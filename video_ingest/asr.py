"""本地 ASR：音频获取、解码、faster-whisper 转写。

关键实现说明（bootstrap §8，实测确认）：
- faster-whisper 1.2.x 的 decode_audio 会调用 av.open(..., metadata_errors="ignore")，
  而 PyAV 19.x 不接受该参数，抛 TypeError。这里不升降级 PyAV，而是自行用 PyAV
  解码成 16kHz 单声道 float32 数组后传给模型，彻底绕开该调用。
- 不用系统 FFmpeg 也能工作（PyAV 自带 FFmpeg 库）。
- 不自造"成功"：VAD 参数如实记录；segments 必须完整迭代后才算转写完成。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .acquire import AcquireError
from .redact import redact

TARGET_RATE = 16000

# 模型缓存：同一进程内复用，避免语言检测与转写各加载一次。
_MODEL_CACHE: dict[tuple, Any] = {}


# --------------------------------------------------------------------------
# 音频获取
# --------------------------------------------------------------------------

def list_audio_formats(url: str, cookies_file: str | None = None,
                       timeout: float = 180.0) -> list[dict[str, Any]]:
    """用 yt-dlp 的 JSON 输出列出可用的纯音频轨道（不下载媒体）。"""
    cmd = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
           "--skip-download", "--no-warnings", "-J"]
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    cmd.append(url)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        raise AcquireError("network_error", "yt-dlp 列举格式超时") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-4:])
        raise AcquireError("network_error", "yt-dlp 列举格式失败: %s" % redact(tail))
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AcquireError("api_changed", "yt-dlp 未返回可解析 JSON: %s" % exc) from exc

    out = []
    for f in info.get("formats") or []:
        if f.get("vcodec") not in (None, "none"):
            continue
        if not f.get("acodec") or f.get("acodec") == "none":
            continue
        out.append({
            "format_id": f.get("format_id"),
            "ext": f.get("ext"),
            "abr": f.get("abr"),
            "acodec": f.get("acodec"),
            "filesize_approx": f.get("filesize_approx"),
        })
    out.sort(key=lambda x: (x.get("abr") or 0), reverse=True)
    return out


def download_audio(
    url: str,
    dest_dir: str | Path,
    cookies_file: str | None = None,
    format_id: str | None = None,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """只下载音频轨道，返回落盘路径与 yt-dlp 元信息摘要。"""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    template = str(dest / "audio.%(ext)s")

    cmd = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
           "--no-warnings", "--print-json"]
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    cmd += ["-f", format_id or "bestaudio/best", "-o", template, "--", url]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        raise AcquireError("network_error", "音频下载超时") from exc

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-4:])
        kind = "needs_login" if "login" in tail.lower() or "cookie" in tail.lower() else "network_error"
        raise AcquireError(kind, "音频下载失败: %s" % redact(tail))

    info: dict[str, Any] = {}
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                info = json.loads(line)
            except json.JSONDecodeError:
                continue

    files = sorted(dest.glob("audio.*"))
    if not files:
        # 有时 yt-dlp 会写成 <id>.<ext>
        files = [p for p in dest.iterdir() if p.is_file() and p.suffix.lower() in
                 (".m4a", ".webm", ".mp3", ".opus", ".ogg", ".aac", ".wav")]
    if not files:
        raise AcquireError("network_error", "音频下载后未找到文件")

    path = files[0]
    return {
        "path": str(path.resolve()),
        "ext": path.suffix.lstrip("."),
        "size_bytes": path.stat().st_size,
        "format_id": info.get("format_id"),
        "acodec": info.get("acodec"),
        "abr": info.get("abr"),
        "duration": info.get("duration"),
        "extractor": info.get("extractor"),
        "title": info.get("title"),
        "uploader": info.get("uploader"),
    }


# --------------------------------------------------------------------------
# 解码（绕开 faster-whisper 的 av.open 调用）
# --------------------------------------------------------------------------

def decode_audio(path: str | Path) -> tuple[Any, int]:
    """解码为 int16 单声道 numpy 数组，返回 (samples, src_rate)。"""
    import av
    import numpy as np

    container = av.open(str(path), mode="r")
    try:
        stream = container.streams.audio[0]
        src_rate = stream.codec_context.sample_rate or 48000
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=src_rate
        )
        chunks = []
        for frame in container.decode(audio=0):
            if frame is None:
                continue
            for rf in resampler.resample(frame):
                if rf is None:
                    continue
                chunks.append(rf.to_ndarray().reshape(-1))
        if not chunks:
            raise AcquireError("failed", "音频解码后没有任何样本: %s" % path)
        pcm = np.concatenate(chunks)
    finally:
        container.close()
    return pcm, src_rate


def resample_linear(samples: Any, src_rate: int, dst_rate: int) -> Any:
    """线性重采样到目标采样率。"""
    import numpy as np

    if src_rate == dst_rate:
        return samples.astype(np.float32)
    n_out = int(round(len(samples) * dst_rate / src_rate))
    if n_out <= 0:
        return samples.astype(np.float32)
    idx = np.linspace(0, len(samples) - 1, n_out, dtype=np.float64)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, len(samples) - 1)
    frac = (idx - lo).astype(np.float32)
    return (samples[lo] * (1 - frac) + samples[hi] * frac).astype(np.float32)


def to_float32_mono16k(path: str | Path) -> tuple[Any, float]:
    """返回 (float32 单声道 16k 数组, 解码时长秒)。"""
    pcm, src_rate = decode_audio(path)
    duration = len(pcm) / float(src_rate)
    audio = resample_linear(pcm.astype("float32") / 32768.0, src_rate, TARGET_RATE)
    return audio, duration


# --------------------------------------------------------------------------
# 设备选择与 GPU 可用性探测
# --------------------------------------------------------------------------

# 实测：CUDA 推理除了 cudnn（随 ctranslate2 打包）之外还需要 cuBLAS，
# 而 pip 装的 ctranslate2 **不含** cuBLAS。nvidia-cublas-cu12 会把它装到
# site-packages/nvidia/cublas/bin，Python 3.8+ 不再从 PATH 之外搜索依赖 DLL，
# 因此必须把该目录显式加进 DLL 搜索路径。
_CUDA_RUNTIME_LIBS = ("cublas64_12.dll", "cublasLt64_12.dll")
_dll_path_registered = False
# 兼容补丁只需打一次
_av_patched = False


def _patch_av_open() -> bool:
    """修补 PyAV 与 faster-whisper 的签名不兼容。

    现象：faster-whisper 1.2.x 内部调用
        av.open(file, mode="r", metadata_errors="ignore")
    而 PyAV 19.x 的 av.open 不接受 metadata_errors，直接抛
        TypeError: open() got an unexpected keyword argument 'metadata_errors'

    影响范围比想象的广：不只是解码文件，**连传入 numpy/列表音频**也会走到
    这条路径（faster-whisper 会先把数组写成临时 wav 再 av.open）。因此
    "自行解码绕开"只能解决文件路径，数组路径仍会失败。

    做法：包一层 av.open，在 PyAV 不接受该参数时静默丢弃它。
    metadata_errors 只影响元数据损坏时的容错，丢弃它对本用途无实质影响。
    不升降级 PyAV 版本——那会牵动 ctranslate2/CUDA 依赖组合。
    """
    global _av_patched
    if _av_patched:
        return True
    try:
        import av
    except Exception:  # noqa: BLE001
        return False

    original = av.open
    if getattr(original, "_video_ingest_patched", False):
        _av_patched = True
        return True

    def patched(*args, **kwargs):
        if "metadata_errors" in kwargs:
            try:
                return original(*args, **kwargs)
            except TypeError:
                kwargs.pop("metadata_errors", None)
        return original(*args, **kwargs)

    patched._video_ingest_patched = True  # type: ignore[attr-defined]
    av.open = patched
    _av_patched = True
    return True


def _ensure_cuda_dll_path() -> list[str]:
    """把 pip 安装的 CUDA 运行时目录加入 DLL 搜索路径。

    返回已加入的目录列表（便于 doctor 报告与排错）。
    """
    global _dll_path_registered
    added: list[str] = []
    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia")
    except Exception:  # noqa: BLE001
        spec = None
    if spec is None or not spec.submodule_search_locations:
        return added

    for root in spec.submodule_search_locations:
        try:
            from pathlib import Path as _Path

            bin_dir = _Path(root) / "cublas" / "bin"
            if not bin_dir.is_dir():
                continue
            # 必须用 add_dll_directory 持有句柄，否则目录会被立刻移除
            os.add_dll_directory(str(bin_dir))
            if str(bin_dir) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            added.append(str(bin_dir))
        except Exception:  # noqa: BLE001
            continue
    _dll_path_registered = bool(added)
    return added


def probe_gpu() -> dict[str, Any]:
    """探测 GPU 是否真的可用。

    **关键**：`ctranslate2.get_cuda_device_count()` 只查硬件，缺 cuBLAS 时
    依然返回 1；而模型**加载**也会成功，只有真正推理才抛
    `Library cublas64_12.dll is not found`。所以不能用"能否加载模型"判断，
    必须实际跑一次极短推理。
    """
    result: dict[str, Any] = {
        "available": False,
        "device": "cpu",
        "compute_type": "int8",
        "reason": None,
        "cuda_device_count": 0,
        "dll_paths": [],
    }
    try:
        import ctranslate2
    except Exception as exc:  # noqa: BLE001
        result["reason"] = "缺少 ctranslate2: %s" % exc
        return result

    try:
        count = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:  # noqa: BLE001
        result["reason"] = "查询 CUDA 设备失败: %s" % exc
        return result
    result["cuda_device_count"] = count
    if count <= 0:
        result["reason"] = "没有可见的 CUDA 设备（未安装驱动或 GPU 不可用）"
        return result

    result["dll_paths"] = _ensure_cuda_dll_path()

    # 真正试一次推理：这是唯一能确认 cuBLAS 可加载的方法。
    try:
        import numpy as np
        from faster_whisper import WhisperModel

        _patch_av_open()
        model = WhisperModel("tiny", device="cuda", compute_type="float16")
        # faster-whisper 只接受文件路径/文件对象/numpy 数组，不接受 list
        silence = np.zeros(TARGET_RATE, dtype=np.float32)   # 1 秒静音
        segments, _info = model.transcribe(silence, language="en", beam_size=1)
        list(segments)   # 必须迭代，否则不会触发推理
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        # 模型权重下载失败与 CUDA 不可用是两回事，不能混为一谈
        if "cublas" in msg.lower() or "cudnn" in msg.lower():
            reason = ("CUDA 运行库缺失：%s；CUDA 推理需要 cuBLAS，安装："
                      "python -m pip install nvidia-cublas-cu12" % msg[:160])
        elif isinstance(exc, (PermissionError, OSError)) and "huggingface" in msg.lower():
            reason = ("探测用的小模型无法下载/写入缓存（%s）。"
                      "这可与 CUDA 无关；请修复缓存目录权限或设 HF_HOME 后重试" % msg[:160])
        else:
            reason = "%s: %s" % (type(exc).__name__, msg[:200])
        result["reason"] = reason
        return result

    result.update(available=True, device="cuda", compute_type="float16", reason=None)
    return result


def resolve_device(device: str) -> tuple[str, str, dict[str, Any]]:
    """把 'auto' 解析成具体设备。

    返回 (device, compute_type, probe_info)。
    - auto：GPU 真能用就用 cuda+float16，否则回退 cpu+int8
    - 显式 cpu/cuda：原样返回，不做探测（尊重用户选择）
    """
    if device != "auto":
        return device, ("int8" if device == "cpu" else "float16"), {"available": None}
    info = probe_gpu()
    if info["available"]:
        return "cuda", "float16", info
    return "cpu", "int8", info


def load_model(
    model_name: str = "medium",
    *,
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: int | None = None,
) -> Any:
    """加载（并缓存）whisper 模型。

    缓存的意义：一次 ingest 里"语言检测"与"转写"需要同一个模型，
    分别加载会让 medium 模型白加载两次（各数秒、数百 MB 内存）。
    键包含 device/compute_type，避免不同配置互相串用。
    """
    from faster_whisper import WhisperModel

    _patch_av_open()
    if device == "cuda":
        _ensure_cuda_dll_path()

    threads = cpu_threads or (os.cpu_count() or 4)
    key = (model_name, device, compute_type, threads)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    model = WhisperModel(
        model_name, device=device, compute_type=compute_type, cpu_threads=threads,
    )
    _MODEL_CACHE[key] = model
    return model


def detect_language(
    audio_path: str | Path,
    *,
    model_name: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
    probe_seconds: float = 30.0,
    cpu_threads: int | None = None,
) -> dict[str, Any]:
    """单独做一次语言检测（不指定 language）。

    **为什么必须单独做**：实测传入 language="zh" 去解码英文音频时，
    info.language 会**回显 "zh" 且 prob=1.000**——也就是说强制指定语言后，
    那对字段不再反映音频真实语言，无法用来发现"指定错了"。

    代价只有一次检测：只取开头 probe_seconds 秒、beam_size=1。
    """
    audio, _dur = to_float32_mono16k(audio_path)
    if probe_seconds and len(audio) > 0:
        # 取开头一小段即可判断语言，避免为检测付整段代价
        n = int(probe_seconds * TARGET_RATE)
        if len(audio) > n:
            audio = audio[:n]

    model = load_model(model_name, device=device, compute_type=compute_type,
                       cpu_threads=cpu_threads)
    _segments, info = model.transcribe(audio, language=None, beam_size=1)
    return {
        "language": getattr(info, "language", None),
        "probability": getattr(info, "language_probability", None),
        "probe_seconds": probe_seconds,
    }


def transcribe_audio(
    audio_path: str | Path,
    *,
    model_name: str = "medium",
    language: str = "zh",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    vad_filter: bool = True,
    initial_prompt: str | None = None,
    cpu_threads: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """转写音频，返回 cues + 实际使用的参数与耗时。

    segments 会被完整迭代（生成器不迭代完不算完成）。
    """

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    started = time.time()
    audio, decoded_duration = to_float32_mono16k(audio_path)
    log("decoded %.1fs audio" % decoded_duration)

    # 若指定了具体语言，先单独检测一次，用于发现"指定语言与音频不符"。
    # 不能依赖 transcribe 返回的 info.language——强制指定时它只是回显。
    detected: dict[str, Any] | None = None
    if language not in ("auto", None):
        try:
            detected = detect_language(
                audio_path, model_name=model_name, device=device,
                compute_type=compute_type, cpu_threads=cpu_threads,
            )
            log("language probe: %s (prob=%s)"
                % (detected["language"], detected["probability"]))
        except Exception as exc:  # noqa: BLE001
            log("language probe failed: %s" % redact(str(exc)))

    # 复用缓存：语言检测那一步已经加载过同一模型，不必再加载一次。
    model = load_model(model_name, device=device, compute_type=compute_type,
                       cpu_threads=cpu_threads)
    log("model %s ready (device=%s, compute_type=%s)" % (model_name, device, compute_type))

    segments, info = model.transcribe(
        audio,
        language=None if language in ("auto", None) else language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": 400} if vad_filter else None,
        initial_prompt=initial_prompt,
        word_timestamps=False,
    )

    cues: list[dict[str, Any]] = []
    for seg in segments:  # 必须完整迭代
        text = (seg.text or "").strip()
        if text:
            cues.append({"start": float(seg.start), "end": float(seg.end), "text": text})

    elapsed = time.time() - started
    log("transcribed %d cues in %.1fs" % (len(cues), elapsed))

    return {
        "cues": cues,
        # transcribe 返回的 detected_language 在强制指定语言时不可信（回显），
        # 因此这里优先用单独检测的结果；language="auto" 时它就是真实检测值。
        "detected_language": (detected or {}).get("language") or getattr(info, "language", None),
        "language_probability": (detected or {}).get("probability")
        if detected else getattr(info, "language_probability", None),
        "decoded_duration": decoded_duration,
        "elapsed_seconds": round(elapsed, 1),
        "params": {
            "model": model_name,
            "language": language,
            "device": device,
            "compute_type": compute_type,
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "vad_parameters": {"min_silence_duration_ms": 400} if vad_filter else None,
            "cpu_threads": cpu_threads or (os.cpu_count() or 4),
        },
    }
