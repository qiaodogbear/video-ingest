"""字幕解析与转录标准化。

职责：
- 解析 SRT / VTT / B 站字幕 JSON（多种形状）为统一片段结构。
- 提供确定性导出：txt / srt / markdown。
- 时间统一为浮点秒；片段 ID 在源文件版本内稳定。

不做的事：不改写文字。校正写在独立层（见 apply_corrections）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

# 统一片段结构键：id, start, end, text, source_type, language

_SRT_TIME = re.compile(
    r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{1,3})"
)
_VTT_TIME = re.compile(
    r"(?:(?P<h>\d+):)?(?P<m>\d{2}):(?P<s>\d{2})[.](?P<ms>\d{1,3})"
)


def _to_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return (
        int(h or 0) * 3600
        + int(m) * 60
        + int(s)
        + int(ms.ljust(3, "0")) / 1000.0
    )


def parse_timestamp(text: str) -> float:
    """解析 "00:01:02.500" / "00:01:02,500" / "01:02.500"。"""
    text = text.strip()
    m = _SRT_TIME.search(text)
    if m:
        return _to_seconds(m.group("h"), m.group("m"), m.group("s"), m.group("ms"))
    m = _VTT_TIME.search(text)
    if m:
        return _to_seconds(m.group("h"), m.group("m"), m.group("s"), m.group("ms"))
    raise ValueError("无法解析时间戳: %r" % text)


def parse_srt(content: str) -> list[dict[str, Any]]:
    if not content or not content.strip():
        return []
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    cues: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", content):
        lines = [l for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        idx = 0
        if lines[0].strip().isdigit():
            idx = 1
        if idx >= len(lines) or "-->" not in lines[idx]:
            continue
        start_s, end_s = lines[idx].split("-->", 1)
        try:
            start = parse_timestamp(start_s)
            end = parse_timestamp(end_s)
        except ValueError:
            continue
        text = "\n".join(lines[idx + 1:]).strip()
        if text:
            cues.append({"start": start, "end": end, "text": text})
    return cues


def parse_vtt(content: str) -> list[dict[str, Any]]:
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    cues: list[dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n", content)
    for block in blocks:
        lines = [l for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        if lines[0].strip().upper().startswith("WEBVTT"):
            lines = lines[1:]
        if not lines:
            continue
        # 跳过 NOTE / STYLE / REGION 块
        if lines[0].strip().upper().startswith(("NOTE", "STYLE", "REGION")):
            continue
        idx = 0
        if "-->" not in lines[idx] and idx + 1 < len(lines):
            idx += 1
        if idx >= len(lines) or "-->" not in lines[idx]:
            continue
        start_s, rest = lines[idx].split("-->", 1)
        end_s = rest.strip().split()[0] if rest.strip() else ""
        try:
            start = parse_timestamp(start_s)
            end = parse_timestamp(end_s)
        except ValueError:
            continue
        text_lines = lines[idx + 1:]
        # 去掉 VTT 内联标签（如 <00:00:01.000><c> ... </c>）
        cleaned = []
        for l in text_lines:
            l = re.sub(r"<[^>]+>", "", l).strip()
            if l:
                cleaned.append(l)
        text = "\n".join(cleaned).strip()
        if text:
            cues.append({"start": start, "end": end, "text": text})
    return _dedupe_rolling(cues)


def _dedupe_rolling(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """VTT 自动字幕常见滚动重复：后一条包含前一条文本时合并。"""
    out: list[dict[str, Any]] = []
    for cue in cues:
        if out and cue["text"] == out[-1]["text"]:
            out[-1]["end"] = cue["end"]
            continue
        out.append(dict(cue))
    return out


def parse_bilibili_subtitle_json(payload: Any) -> list[dict[str, Any]]:
    """B 站字幕 JSON。可见形状：

        {"body": [{"from": 1.2, "to": 3.4, "content": "..."}]}
        {"body": [{"from":..,"to":..,"content":..}], "font_size": ..}

    某些镜像返回 {"data": {"body": [...]}}。无法识别时抛 ValueError，
    不猜格式（bootstrap §4：来源无法识别时不得猜测）。
    """
    body = None
    if isinstance(payload, dict):
        if isinstance(payload.get("body"), list):
            body = payload["body"]
        elif isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("body"), list):
            body = payload["data"]["body"]
    if body is None:
        raise ValueError("无法识别的 B 站字幕 JSON 形状：顶层键 %r" % (
            list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        ))

    cues: list[dict[str, Any]] = []
    for item in body:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("from"))
            end = float(item.get("to"))
        except (TypeError, ValueError):
            continue
        text = str(item.get("content") or "").strip()
        if text:
            cues.append({"start": start, "end": end, "text": text})
    return cues


def parse_subtitle_file(path: str | Path, *, fmt: str | None = None) -> list[dict[str, Any]]:
    """按扩展名或显式 fmt 解析字幕文件。"""
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    fmt = (fmt or p.suffix.lstrip(".")).lower()
    if fmt in ("srt",):
        cues = parse_srt(raw)
    elif fmt in ("vtt", "webvtt"):
        cues = parse_vtt(raw)
    elif fmt in ("json", "bcc"):
        cues = parse_bilibili_subtitle_json(json.loads(raw))
    elif fmt == "xml":
        raise ValueError("XML 通常为弹幕(danmaku)，不是字幕；拒绝按字幕解析")
    else:
        raise ValueError("不支持的字幕格式: %s" % fmt)
    if not cues:
        raise ValueError("字幕文件解析后为空: %s" % p)
    return cues


# --------------------------------------------------------------------------
# 标准化
# --------------------------------------------------------------------------

def normalize_cues(
    cues: Iterable[dict[str, Any]],
    *,
    source_type: str,
    language: str,
    duration: float | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """加上稳定片段 ID 与来源标记，并做完整性检查。

    返回 (segments, issues)。issues 只记录可疑情况，不静默删数据。
    """
    segs: list[dict[str, Any]] = []
    issues: list[str] = []
    previous_end = -1.0

    for i, cue in enumerate(cues):
        start = float(cue["start"])
        end = float(cue["end"])
        text = str(cue["text"])

        if start < 0:
            issues.append("片段 %d：start 为负 (%s)，已截断为 0" % (i, start))
            start = 0.0
        if end < start:
            issues.append("片段 %d：end < start (%s < %s)，按 start 对齐" % (i, end, start))
            end = start
        if duration is not None and start > duration * 1.05 + 1:
            issues.append("片段 %d：start (%s) 超出解码时长 (%s)" % (i, start, duration))
        if start < previous_end - 0.5:
            # 合法重叠不删除，仅登记
            issues.append("片段 %d：与前一片段重叠 %.2fs（保留，未删除）" % (i, previous_end - start))
        previous_end = max(previous_end, end)

        segs.append(
            {
                "id": "seg-%06d" % (i + 1),
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "source_type": source_type,
                "language": language,
            }
        )

    segs.sort(key=lambda s: (s["start"], s["end"]))
    # 排序不改变源文件；ID 在排序后按新顺序重编，保证序列稳定且递增
    for i, s in enumerate(segs):
        s["id"] = "seg-%06d" % (i + 1)
    return segs, issues


def to_plain_text(segments: list[dict[str, Any]]) -> str:
    return "".join(s["text"] for s in segments)


def to_txt(segments: list[dict[str, Any]], *, with_timestamps: bool = True) -> str:
    lines = []
    for s in segments:
        if with_timestamps:
            lines.append("[%s -> %s] %s" % (_mmss(s["start"]), _mmss(s["end"]), s["text"]))
        else:
            lines.append(s["text"])
    return "\n".join(lines) + "\n"


def to_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for i, s in enumerate(segments, 1):
        blocks.append(
            "%d\n%s --> %s\n%s\n" % (i, _srt_time(s["start"]), _srt_time(s["end"]), s["text"])
        )
    return "\n".join(blocks)


def to_markdown(segments: list[dict[str, Any]], *, title: str = "转录全文") -> str:
    lines = ["# %s" % title, "", "> 本文件为原始转录的阅读视图，未改写文字。", ""]
    for s in segments:
        lines.append("**[%s]** %s" % (_mmss(s["start"]), s["text"]))
        lines.append("")
    return "\n".join(lines)


def _mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return "%d:%02d:%02d" % (h, m, s)
    return "%02d:%02d" % (m, s)


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


# --------------------------------------------------------------------------
# 校正层（不覆盖原文）
# --------------------------------------------------------------------------

def apply_corrections(
    segments: list[dict[str, Any]], table: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """生成校正稿与更正记录。原文层不被修改。

    返回 (corrected_segments, correction_records)。
    """
    if not table:
        return [dict(s) for s in segments], []

    ordered = sorted(table, key=len, reverse=True)
    out: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for seg in segments:
        text = seg["text"]
        new_text = text
        for wrong in ordered:
            if wrong and wrong in new_text:
                new_text = new_text.replace(wrong, table[wrong])
                records.append(
                    {
                        "segment_id": seg["id"],
                        "start": seg["start"],
                        "wrong": wrong,
                        "right": table[wrong],
                        "basis": "user_supplied_table",
                    }
                )
        new_seg = dict(seg)
        new_seg["text"] = new_text
        new_seg["corrected"] = new_text != text
        out.append(new_seg)
    return out, records
