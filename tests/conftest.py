"""全局测试配置。

要点：单元测试**不得依赖 GPU、模型权重或网络**。
`resolve_device("auto")` 会做一次真实推理探测（加载模型，实测秒级到数十秒），
若不禁用，测试套件会在有 GPU 的机器上白等几十秒、在无 GPU 的机器上行为
不稳定。这里统一关掉探测，需要验证探测逻辑的用例自行 monkeypatch。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _no_gpu_probe_and_clean_caches(monkeypatch):
    """整套测试：禁止真实 GPU 探测，并清理模块级缓存。"""
    monkeypatch.setenv("VIDEO_INGEST_SKIP_GPU_PROBE", "1")
    # 避免测试期间意外联网校验模型
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    from video_ingest import asr

    asr._GPU_PROBE_CACHE = None
    yield
    asr._GPU_PROBE_CACHE = None


@pytest.fixture(autouse=True)
def _isolate_hf_home(tmp_path, monkeypatch):
    """把 HuggingFace 缓存指向临时目录，避免测试触碰用户真实缓存。"""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    yield
