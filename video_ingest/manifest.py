"""manifest 数据合同。

两个状态维度必须分开（bootstrap §10）：
- processing_status: resource_processed / partial / blocked / failed
  只说明"拿到的资源有没有处理完"。
- verification_status: unreviewed / sample_checked / issues_found
  只说明"核查到什么程度"，不把抽样核查写成逐字准确。

- full_video_coverage: supported / uncertain / partial
  拿到整个字幕文件只证明下载了这个文件，不证明它覆盖视频中所有讲话。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1

PROCESSING_STATUSES = ("resource_processed", "partial", "blocked", "failed")
VERIFICATION_STATUSES = ("unreviewed", "sample_checked", "issues_found")
COVERAGE_STATUSES = ("supported", "uncertain", "partial")

# 字幕/ASR 来源标记。unknown 是合法值，禁止猜成人工字幕（bootstrap §4）。
SOURCE_TYPES = (
    "platform_manual",   # 平台人工字幕
    "platform_auto",     # 平台自动字幕
    "local_asr",         # 本地语音识别
    "unknown",
)

# 枚举字幕失败的原因分类。不得统一吞成空列表（bootstrap §7.1）。
SUBTITLE_FAILURE_KINDS = (
    "no_native_subtitle",             # 平台侧确认没有原语字幕
    "translation_only",               # 只有机器翻译字幕，无原语
    "needs_login",                    # 需要登录才可见
    "network_error",
    "rate_limited",
    "api_changed",
    "content_unavailable",
    "not_probed",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(text: str, max_len: int = 60) -> str:
    """生成兼容 Windows 的文件名片段，保留中文，不保留路径分隔符。"""
    text = unicodedata.normalize("NFC", text or "")
    text = _UNSAFE.sub("_", text)
    text = re.sub(r"\s+", "_", text).strip("._")
    if not text:
        text = "untitled"
    return text[:max_len].rstrip("._") or "untitled"


def make_job_id(platform: str, content_id: str, page: int | None = None) -> str:
    """稳定 ID 与标题分离（bootstrap §9）。"""
    parts = [platform, content_id]
    if page is not None:
        parts.append("p%d" % page)
    return "_".join(parts)


def new_manifest(
    *,
    job_id: str,
    input_url: str | None,
    canonical_url: str | None,
    platform: str,
    media_kind: str,
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "job_id": job_id,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "input_url": input_url,
        "canonical_url": canonical_url,
        "platform": platform,
        "media_kind": media_kind,          # bilibili_video | local_media | ...
        "identity": {
            "content_id": None,            # bvid
            "cid": None,
            "page": None,
            "aid": None,
            "title": None,
            "uploader": None,
        },
        "timing": {
            "claimed_duration": None,      # 平台声称时长（秒）
            "decoded_duration": None,      # 实际解码时长（秒）
            "last_cue_end": None,          # 最后一条字幕结束时间
        },
        "subtitle": {
            "probe_state": "not_probed",   # found | empty | needs_auth_check | not_probed
            "failure_kind": "not_probed",  # 见 SUBTITLE_FAILURE_KINDS
            "language_keys": [],
            "selected_language": None,
            "source_type": "unknown",
            "source_file": None,
            "source_sha256": None,
            "auth_used": False,
        },
        "asr": {
            "used": False,
            "model": None,
            "model_revision": None,
            "compute_type": None,
            "device": None,
            "language": None,
            "detected_language": None,
            "language_mismatch": False,
            "beam_size": None,
            "vad_filter": None,
            "vad_parameters": None,
            "audio_file": None,
            "audio_sha256": None,
        },
        "tooling": {
            "video_ingest_version": None,
            "yt_dlp_version": None,
            "extractor": None,
            "faster_whisper_version": None,
            "python_version": None,
        },
        "counts": {
            "segment_count": 0,
            "chunk_count": 0,
        },
        "visual": {
            "used": False,
            "video_file": None,
            "video_sha256": None,
            "interval": None,
            "scene_threshold": None,
            "scale": None,
            "run_count": None,
            "frames_written": None,
            "frames_skipped_blank": None,
            "orphan_run_count": None,
            "index": None,
            "sampling_policy": "intersection_midpoint",
            "understanding_by": None,   # 必须是 agent；脚本不做视觉理解
            "notes": [],
        },
        "ocr": {
            "used": False,
            "engine": None,
            "frames_read": None,
            "seconds_total": None,
            "seconds_per_frame": None,
            "divergent_count": None,
            "divergence_threshold": None,
            "index": None,
            "understanding_by": None,   # 必须是 agent；OCR 只提取文字
        },
        "files": {},
        "processing_status": "failed",
        "verification_status": "unreviewed",
        "full_video_coverage": "uncertain",
        "coverage_basis": [],
        "errors": [],
        "notes": [],
    }


def add_error(manifest: dict[str, Any], stage: str, kind: str, message: str) -> None:
    """记录错误。message 必须先脱敏（见 redact.py）。"""
    manifest.setdefault("errors", []).append(
        {"at": utc_now_iso(), "stage": stage, "kind": kind, "message": message}
    )
    manifest["updated_at"] = utc_now_iso()


def add_note(manifest: dict[str, Any], note: str) -> None:
    manifest.setdefault("notes", []).append({"at": utc_now_iso(), "note": note})
    manifest["updated_at"] = utc_now_iso()


def add_coverage_basis(manifest: dict[str, Any], basis: str) -> None:
    manifest.setdefault("coverage_basis", []).append(basis)
    manifest["updated_at"] = utc_now_iso()


def save_manifest(job_dir: str | os.PathLike[str], manifest: dict[str, Any]) -> Path:
    manifest["updated_at"] = utc_now_iso()
    path = Path(job_dir) / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)   # 原子替换，避免半写文件被当成成功
    return path


def load_manifest(job_dir: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(job_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("manifest.json 不存在: %s" % path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """返回一致性告警列表（不抛异常，供 validate 命令使用）。"""
    problems: list[str] = []

    ps = manifest.get("processing_status")
    vs = manifest.get("verification_status")
    cov = manifest.get("full_video_coverage")

    if ps not in PROCESSING_STATUSES:
        problems.append("processing_status 非法: %r" % ps)
    if vs not in VERIFICATION_STATUSES:
        problems.append("verification_status 非法: %r" % vs)
    if cov not in COVERAGE_STATUSES:
        problems.append("full_video_coverage 非法: %r" % cov)

    sub = manifest.get("subtitle", {})
    src = sub.get("source_type")
    if src not in SOURCE_TYPES:
        problems.append("subtitle.source_type 非法: %r" % src)
    if sub.get("failure_kind") not in SUBTITLE_FAILURE_KINDS:
        problems.append("subtitle.failure_kind 非法: %r" % sub.get("failure_kind"))

    # 声称"处理完成"但没有片段，是矛盾状态。
    if ps == "resource_processed" and not manifest.get("counts", {}).get("segment_count"):
        problems.append("processing_status=resource_processed 但 segment_count 为 0")

    # 覆盖率 supported 必须有依据。
    if cov == "supported" and not manifest.get("coverage_basis"):
        problems.append("full_video_coverage=supported 但 coverage_basis 为空（缺依据）")

    # 覆盖不是 supported 时，摘要层必须带警示；这里只做标记检查。
    if cov != "supported" and ps == "resource_processed":
        if not any("警示" in n.get("note", "") for n in manifest.get("notes", [])):
            problems.append("覆盖率非 supported，但未记录'输出需带警示'的说明")

    # ASR 与字幕不应同时声称是主要来源。
    if manifest.get("asr", {}).get("used") and src == "platform_manual":
        problems.append("asr.used=true 但来源标为 platform_manual，来源标记矛盾")

    # 语言不符是"内容不可信"级别的信号，必须体现为 issue，不能停留在 unreviewed。
    asr = manifest.get("asr") or {}
    if asr.get("language_mismatch"):
        if manifest.get("verification_status") == "unreviewed":
            problems.append(
                "asr.language_mismatch=true 但 verification_status 仍为 unreviewed"
                "（指定语言与检测语言不符时，转录内容不可信）")
        if not any("语言" in n.get("note", "") for n in manifest.get("notes", [])):
            problems.append("asr.language_mismatch=true 但未记录任何语言警示")

    # 视觉理解必须由 Agent 完成，脚本不得声称做了理解。
    vis = manifest.get("visual") or {}
    if vis.get("used"):
        if not vis.get("index"):
            problems.append("visual.used=true 但缺少 index 路径")
        if vis.get("understanding_by") not in (None, "agent"):
            problems.append("visual.understanding_by 只能是 agent（脚本不做视觉理解）")
        if not vis.get("sampling_policy"):
            problems.append("visual.used=true 但未记录取样策略")
        if vis.get("frames_written") == 0:
            problems.append("visual.used=true 但 frames_written 为 0")

    # OCR 是可选依赖；用了就必须记清楚，且不得声称做了语义理解。
    ocr = manifest.get("ocr") or {}
    if ocr.get("used"):
        if not ocr.get("index"):
            problems.append("ocr.used=true 但缺少 index 路径")
        if ocr.get("understanding_by") not in (None, "agent"):
            problems.append("ocr.understanding_by 只能是 agent（OCR 只提取文字）")
        if ocr.get("frames_read") == 0:
            problems.append("ocr.used=true 但 frames_read 为 0")
        vis2 = manifest.get("visual") or {}
        if not vis2.get("used"):
            problems.append("ocr.used=true 但 visual.used 为 false（OCR 依赖已提取的帧）")

    # 时长核对
    timing = manifest.get("timing", {})
    claimed, decoded = timing.get("claimed_duration"), timing.get("decoded_duration")
    if claimed and decoded and decoded > claimed * 1.05 + 1:
        problems.append("decoded_duration(%.1f) 明显超过 claimed_duration(%.1f)" % (decoded, claimed))

    return problems
