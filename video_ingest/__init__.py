"""video_ingest —— 本地优先的视频取材与转录工具。

设计约束（见 SOP.md）：
- 只做确定性、可复现、可校验的取材工作，不调用任何 LLM。
- 原始材料与派生内容严格分层，原文层不被覆盖。
- 处理状态与核查状态分开记录，不把"进程退出码 0"当成"完整准确"。
"""

from __future__ import annotations

# 版本单一来源：优先读安装元数据，未安装时回退到源码声明。
# 不要在多个文件里各写一份版本号——两处会漂移（曾经发生过）。
try:  # pragma: no cover - 取决于是否已安装
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    try:
        __version__ = _dist_version("video-ingest")
    except PackageNotFoundError:
        __version__ = "0.2.0"
except Exception:  # noqa: BLE001  # pragma: no cover
    __version__ = "0.2.0"

__all__ = ["__version__"]
