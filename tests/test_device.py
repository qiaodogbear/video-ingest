"""设备选择与 GPU 探测的单元测试。

不依赖真实 GPU：通过打桩 probe_gpu 验证决策逻辑。
背景（实测）：
- ctranslate2.get_cuda_device_count() 只查硬件，缺 cuBLAS 时仍返回 1；
- 缺 cuBLAS 时模型**加载**也会成功，只有真正推理才抛
  "Library cublas64_12.dll is not found"。
因此"能否加载模型"不能作为 GPU 可用性的判据。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_ingest import asr  # noqa: E402
from video_ingest.asr import resolve_device  # noqa: E402


def test_explicit_cpu_is_respected():
    """显式指定设备时不做探测，尊重用户选择。"""
    device, ct, info = resolve_device("cpu")
    assert device == "cpu"
    assert ct == "int8"
    assert info.get("available") is None   # 未探测


def test_explicit_cuda_is_respected_even_if_probe_says_no(monkeypatch):
    """用户显式要 cuda 就给他 cuda，交由运行时如实报错，不静默改写。"""
    monkeypatch.setattr(asr, "probe_gpu", lambda: {"available": False, "reason": "no"})
    device, ct, _ = resolve_device("cuda")
    assert device == "cuda"
    assert ct == "float16"


def test_auto_uses_cuda_when_probe_succeeds(monkeypatch):
    monkeypatch.setattr(asr, "probe_gpu",
                        lambda: {"available": True, "cuda_device_count": 1, "reason": None})
    device, ct, info = resolve_device("auto")
    assert device == "cuda"
    assert ct == "float16"          # 默认 float16，不是 CPU 惯用的 int8
    assert info["available"] is True


def test_auto_falls_back_to_cpu_when_probe_fails(monkeypatch):
    monkeypatch.setattr(asr, "probe_gpu",
                        lambda: {"available": False, "cuda_device_count": 1,
                                 "reason": "Library cublas64_12.dll is not found"})
    device, ct, info = resolve_device("auto")
    assert device == "cpu"
    assert ct == "int8"
    assert "cublas" in info["reason"]


def test_probe_reports_reason_when_no_cuda_device(monkeypatch):
    class FakeCT:
        @staticmethod
        def get_cuda_device_count():
            return 0

    monkeypatch.setitem(sys.modules, "ctranslate2", FakeCT)
    result = asr.probe_gpu()
    assert result["available"] is False
    assert result["cuda_device_count"] == 0
    assert "没有可见的 CUDA 设备" in result["reason"]


def test_probe_handles_missing_ctranslate2(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    monkeypatch.delitem(sys.modules, "ctranslate2", raising=False)
    result = asr.probe_gpu()
    assert result["available"] is False
    assert result["reason"]


def _install_probe_stubs(monkeypatch, model_cls, device_count=1):
    """装好 probe_gpu 需要的桩：ctranslate2 / faster_whisper / numpy。

    numpy 也要桩，因为 probe_gpu 用它构造 1 秒静音数组。这样这些用例
    在"最小环境"（未安装 numpy）下也能跑到真正的判定分支，而不是被
    更早的 ImportError 掩盖。
    """
    class FakeCT:
        @staticmethod
        def get_cuda_device_count():
            return device_count

    class FakeNumpy:
        # probe_gpu 会构造 np.zeros(n, dtype=np.float32)
        float32 = "float32"

        @staticmethod
        def zeros(n, dtype=None):
            return [0.0] * int(n)

    fake_fw = type(sys)("faster_whisper")
    fake_fw.WhisperModel = model_cls
    monkeypatch.setitem(sys.modules, "ctranslate2", FakeCT)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    monkeypatch.setitem(sys.modules, "numpy", FakeNumpy)
    monkeypatch.setattr(asr, "_ensure_cuda_dll_path", lambda: [])


def test_probe_explains_missing_cublas(monkeypatch):
    """缺 cuBLAS 时必须给出可执行的修复命令，而不是只抛英文库名。"""
    class FakeModel:
        def __init__(self, *a, **k):
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    _install_probe_stubs(monkeypatch, FakeModel)

    result = asr.probe_gpu()
    assert result["available"] is False
    assert "nvidia-cublas-cu12" in result["reason"]


def test_probe_distinguishes_cache_error_from_cuda_error(monkeypatch):
    """权重下载/缓存失败与 CUDA 不可用是两回事，不能混为一谈。"""
    class FakeModel:
        def __init__(self, *a, **k):
            raise PermissionError(
                "[WinError 5] 拒绝访问。: 'C:\\\\x\\\\huggingface\\\\hub\\\\models--y'")

    _install_probe_stubs(monkeypatch, FakeModel)

    result = asr.probe_gpu()
    assert result["available"] is False
    assert "缓存" in result["reason"]
    assert "nvidia-cublas" not in result["reason"]


def test_probe_succeeds_when_inference_runs(monkeypatch):
    """正向路径：推理真的跑通时判定为可用。"""
    class FakeSegments(list):
        pass

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, language=None, beam_size=None):
            return FakeSegments(), object()

    _install_probe_stubs(monkeypatch, FakeModel)
    result = asr.probe_gpu()
    assert result["available"] is True
    assert result["device"] == "cuda"
    assert result["compute_type"] == "float16"


def test_av_open_patch_drops_unsupported_metadata_errors(monkeypatch):
    """回归：PyAV 19 不接受 metadata_errors，补丁必须静默丢弃它。

    这条路径影响的不仅是文件解码——faster-whisper 传入数组时同样会走
    av.open，所以不打补丁会导致"数组输入"这条路彻底失败。
    """
    fake_av = type(sys)("av")

    def strict_open(file, mode="r", **kwargs):
        if "metadata_errors" in kwargs:
            raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
        return "opened"

    fake_av.open = strict_open
    monkeypatch.setitem(sys.modules, "av", fake_av)
    monkeypatch.setattr(asr, "_av_patched", False)

    assert asr._patch_av_open() is True
    # 打了补丁后，带 metadata_errors 的调用不应再报错
    assert fake_av.open("x", metadata_errors="ignore") == "opened"
    # 其它参数照常透传
    assert fake_av.open("x", mode="r") == "opened"


def test_av_open_patch_is_idempotent(monkeypatch):
    """重复调用不应层层包装 av.open。"""
    fake_av = type(sys)("av")
    fake_av.open = lambda *a, **k: "opened"
    monkeypatch.setitem(sys.modules, "av", fake_av)
    monkeypatch.setattr(asr, "_av_patched", False)

    assert asr._patch_av_open() is True
    first = fake_av.open
    assert asr._patch_av_open() is True
    assert fake_av.open is first      # 不重复包装
    assert fake_av.open("x") == "opened"


def test_model_cache_is_keyed_by_device_and_compute_type(monkeypatch):
    """不同 device/compute_type 不能复用同一个模型实例。"""
    created = []

    class FakeModel:
        def __init__(self, name, device=None, compute_type=None, cpu_threads=None):
            created.append((name, device, compute_type))

    fake_fw = type(sys)("faster_whisper")
    fake_fw.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    monkeypatch.setattr(asr, "_MODEL_CACHE", {})
    monkeypatch.setattr(asr, "_ensure_cuda_dll_path", lambda: [])

    a = asr.load_model("tiny", device="cpu", compute_type="int8")
    b = asr.load_model("tiny", device="cpu", compute_type="int8")
    c = asr.load_model("tiny", device="cuda", compute_type="float16")

    assert a is b                      # 同配置复用
    assert c is not a                  # 不同配置不复用
    assert len(created) == 2


def test_uniform_device_defaults_are_auto_in_cli():
    """回归：CLI 默认必须是 auto。曾经默认 cpu，导致有 GPU 也用不上。"""
    from video_ingest.cli import build_parser

    parser = build_parser()

    def sub_default(cmd: str, opt: str):
        for action in parser._subparsers._group_actions:
            if cmd in action.choices:
                sp = action.choices[cmd]
                for a in sp._actions:
                    if a.dest == opt:
                        return a.default
        raise AssertionError("未找到 %s/%s" % (cmd, opt))

    for cmd in ("ingest", "batch"):
        assert sub_default(cmd, "device") == "auto", cmd
        # compute_type 默认留空，交给设备解析决定
        assert sub_default(cmd, "compute_type") is None, cmd
