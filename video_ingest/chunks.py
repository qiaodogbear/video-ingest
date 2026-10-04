"""确定性分块与覆盖检查。

bootstrap §12 要求：
- 不随机删段，不只把前 N 个字符塞给模型。
- 每个原始片段都必须在主覆盖集合中出现。
- 未知分词器时用保守字符上限，并明确说明它不是 token 数。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 保守字符上限。**这不是 token 数**（bootstrap §12）。
DEFAULT_CHARS_PER_CHUNK = 3000


def build_chunks(
    segments: list[dict[str, Any]],
    *,
    chars_per_chunk: int = DEFAULT_CHARS_PER_CHUNK,
    overlap_segments: int = 1,
) -> dict[str, Any]:
    """按字符预算切块，返回 chunk 索引与覆盖检查。

    规则：
    - 主覆盖集合包含每一个片段 ID，恰好一次。
    - 允许在相邻块之间重复少量片段作为重叠，但重叠不替代主覆盖。
    - 片段不跨块拆分，避免破坏句义边界。
    """
    if chars_per_chunk <= 0:
        raise ValueError("chars_per_chunk 必须为正整数")
    if overlap_segments < 0:
        raise ValueError("overlap_segments 不能为负")

    chunks: list[dict[str, Any]] = []
    covered: list[str] = []
    overlap_ids: list[str] = []

    current: list[dict[str, Any]] = []
    # 当前块中"首次进入主覆盖"的片段 ID。重叠片段不计入主覆盖，
    # 否则同一 ID 会被计入两次，coverage 会误报不完整。
    current_new: list[str] = []
    current_chars = 0

    def flush() -> list[dict[str, Any]]:
        nonlocal current, current_new, current_chars
        if not current:
            return []
        idx = len(chunks) + 1
        chunks.append(
            {
                "chunk_id": "chunk-%03d" % idx,
                "segment_ids": [s["id"] for s in current],
                "new_segment_ids": list(current_new),
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "char_count": sum(len(s["text"]) for s in current),
                "segments": current,
            }
        )
        covered.extend(current_new)
        flushed = current
        current = []
        current_new = []
        current_chars = 0
        return flushed

    for seg in segments:
        seg_chars = len(seg["text"])
        if current and current_chars + seg_chars > chars_per_chunk:
            flushed = flush()
            # 重叠：把上一块末尾若干片段带入下一块，仅作上下文，不替代主覆盖
            if overlap_segments:
                tail = flushed[-overlap_segments:]
                overlap_ids.extend(s["id"] for s in tail)
                current = [dict(s) for s in tail]
                current_chars = sum(len(s["text"]) for s in current)
        if seg["id"] not in current_new:
            current_new.append(seg["id"])
        current.append(seg)
        current_chars += seg_chars
    flush()

    all_ids = [s["id"] for s in segments]
    missing = [sid for sid in all_ids if sid not in set(covered)]
    duplicated = sorted({sid for sid in covered if covered.count(sid) > 1})

    return {
        "chars_per_chunk": chars_per_chunk,
        "chars_per_chunk_is_tokens": False,
        "overlap_segments": overlap_segments,
        "chunk_count": len(chunks),
        "segment_count": len(segments),
        "chunks": chunks,
        "coverage": {
            "all_segment_ids": all_ids,
            "covered_segment_ids": covered,
            "missing_segment_ids": missing,
            "overlap_segment_ids": sorted(set(overlap_ids)),
            "duplicated_in_main_coverage": duplicated,
            "complete": not missing and not duplicated,
        },
    }


def validate_coverage(index: dict[str, Any]) -> list[str]:
    """返回覆盖问题列表（空 = 通过）。"""
    problems: list[str] = []
    cov = index.get("coverage") or {}
    missing = cov.get("missing_segment_ids") or []
    dup = cov.get("duplicated_in_main_coverage") or []

    if missing:
        problems.append("有 %d 个片段未进入任何分块: %s" % (len(missing), ", ".join(missing[:10])))
    if dup:
        problems.append("主覆盖集合中片段重复: %s" % ", ".join(dup[:10]))

    declared = index.get("chunk_count")
    actual = len(index.get("chunks") or [])
    if declared != actual:
        problems.append("chunk_count(%s) 与实际块数(%s) 不一致" % (declared, actual))

    seg_count = index.get("segment_count")
    if isinstance(seg_count, int) and isinstance(cov.get("covered_segment_ids"), list):
        if len(cov["covered_segment_ids"]) != seg_count:
            problems.append(
                "covered_segment_ids 长度(%d) 与 segment_count(%d) 不一致"
                % (len(cov["covered_segment_ids"]), seg_count)
            )
    for ch in index.get("chunks") or []:
        if not ch.get("segment_ids"):
            problems.append("分块 %s 没有片段" % ch.get("chunk_id"))
    return problems


def write_chunks(job_dir: str | Path, index: dict[str, Any]) -> Path:
    """把分块写到 <job>/chunks/：index.json + 每块一个 md。"""
    base = Path(job_dir) / "chunks"
    base.mkdir(parents=True, exist_ok=True)

    public = {k: v for k, v in index.items() if k != "chunks"}
    public["chunks"] = [
        {k: v for k, v in ch.items() if k != "segments"} for ch in index["chunks"]
    ]
    index_path = base / "index.json"
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(public, fh, ensure_ascii=False, indent=2)

    for ch in index["chunks"]:
        lines = [
            "# %s" % ch["chunk_id"],
            "",
            "- 片段区间: %s – %s" % (ch["segment_ids"][0], ch["segment_ids"][-1]),
            "- 时间范围: %.1fs – %.1fs" % (ch["start"], ch["end"]),
            "- 字符数: %d（字符预算，不是 token 数）" % ch["char_count"],
            "",
            "> 本块包含 %d 个片段，每个片段保留原始 ID 与时间戳。" % len(ch["segment_ids"]),
            "",
        ]
        for s in ch["segments"]:
            lines.append("### %s  [%.1fs – %.1fs]" % (s["id"], s["start"], s["end"]))
            lines.append("")
            lines.append(s["text"])
            lines.append("")
        (base / ("%s.md" % ch["chunk_id"])).write_text("\n".join(lines), encoding="utf-8")

    return index_path
