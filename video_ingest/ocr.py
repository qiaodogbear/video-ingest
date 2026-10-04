"""画面文字识别（OCR）与三方绑定。

三方绑定关系
------------
    字幕片段 (segment)  ──时间求交──▶  画面帧 (frame)  ──读取──▶  屏幕文字 (ocr_text)

**重要设计认识**：OCR **不能**用来判断某帧该配哪条字幕。
实测 70 帧里有 7 帧（10%）口播与画面**没有任何共同词汇**——旁白讲方法论、
画面放阶段结论。配对只能靠时间（见 visual.py 的区间求交）；OCR 的价值是
**让画面证据变得可检索**：

- 正向：某条字幕对应期间，屏幕上出现了哪些文字
- 反向：某个词出现在屏幕上的哪些时间点（这正是"必须看画面才能理解"的入口）
- 分歧：口播与画面内容无关的位置，即纯转录必然漏掉的信息

依赖：rapidocr-onnxruntime + onnxruntime。**可选依赖**——缺失时本模块
的检查函数返回不可用原因，命令层给出明确提示而不是崩溃。

原始层不被覆盖：OCR 文本单独存放，不改写 transcript。
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

# 判定"口播与画面无关"的 n-gram 覆盖率阈值
DIVERGENCE_THRESHOLD = 0.05
NGRAM = 4

PUNCT = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]")


class OcrUnavailable(RuntimeError):
    """OCR 依赖缺失或初始化失败。命令层据此给出可执行的修复提示。"""


def ocr_available() -> tuple[bool, str | None]:
    """返回 (是否可用, 不可用原因)。"""
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, (
            "未安装 OCR 依赖（%s）。安装："
            'python -m pip install rapidocr-onnxruntime'
            % type(exc).__name__
        )
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, "缺少 onnxruntime：%s" % exc
    return True, None


def normalize_text(text: str) -> str:
    """只保留中日韩与字母数字，用于宽松比对。"""
    return PUNCT.sub("", text or "").lower()


def ngrams(text: str, n: int = NGRAM) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def overlap_ratio(a_norm: str, b_norm: str) -> float:
    """a 的 n-gram 有多少比例出现在 b 中（对 OCR 错字与换行有一定容忍）。"""
    ga, gb = ngrams(a_norm), ngrams(b_norm)
    if not ga:
        return 0.0
    return len(ga & gb) / len(ga)


class OcrEngine:
    """惰性初始化的 RapidOCR 包装。模型加载较慢，只初始化一次。"""

    def __init__(self) -> None:
        ok, reason = ocr_available()
        if not ok:
            raise OcrUnavailable(reason)
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

    def read(self, path: str | Path) -> dict[str, Any]:
        """识别单张图片，返回文本与行数。"""
        started = time.time()
        result, _elapsed = self._engine(str(path))
        lines = []
        for item in (result or []):
            # RapidOCR 返回 [box, text, score]
            if len(item) >= 2 and item[1]:
                lines.append(str(item[1]).strip())
        text = " ".join(l for l in lines if l)
        return {
            "text": text,
            "line_count": len(lines),
            "seconds": round(time.time() - started, 2),
        }


def run_ocr(
    job_dir: str | Path,
    *,
    frames_dir: str | Path | None = None,
    index_name: str = "frames.index.json",
    on_progress: Callable[[str], None] | None = None,
    limit: int = 0,
) -> dict[str, Any]:
    """对已提取的帧做 OCR，产出 ocr.index.json。"""
    job = Path(job_dir)
    frames_index_path = job / index_name
    if not frames_index_path.exists():
        raise FileNotFoundError(
            "缺少 %s：请先执行 frames 命令提取画面帧" % index_name
        )
    idx = json.loads(frames_index_path.read_text(encoding="utf-8"))
    frames_path = Path(frames_dir) if frames_dir else (job / "frames")
    if not frames_path.exists():
        raise FileNotFoundError("缺少帧目录: %s" % frames_path)

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # 帧 -> 引用它的片段
    frame_refs: dict[str, list[str]] = defaultdict(list)
    seg_by_id: dict[str, dict[str, Any]] = {}
    for seg in idx.get("segments", []):
        seg_by_id[seg["segment_id"]] = seg
        for fr in seg.get("frames", []):
            if not fr.get("skipped"):
                frame_refs[fr["path"]].append(seg["segment_id"])

    files = sorted(frames_path.glob("*.png"))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError("帧目录里没有 PNG: %s" % frames_path)

    engine = OcrEngine()
    records: list[dict[str, Any]] = []
    started_all = time.time()
    for i, path in enumerate(files, 1):
        try:
            read = engine.read(path)
        except Exception as exc:  # noqa: BLE001
            read = {"text": "", "line_count": 0, "seconds": 0.0, "error": str(exc)}
        seg_ids = frame_refs.get(path.name, [])

        # 该帧对应的字幕原文（一帧可能被多条字幕引用）
        seg_text = " ".join(seg_by_id[s]["text"] for s in seg_ids if s in seg_by_id)
        n_frame = normalize_text(read["text"])
        n_seg = normalize_text(seg_text)

        # 双向覆盖率：字幕有多少出现在画面里 / 画面有多少出现在字幕里
        cov_sub = overlap_ratio(n_seg, n_frame) if n_seg else None
        cov_frame = overlap_ratio(n_frame, n_seg) if n_seg else None

        records.append({
            "frame": path.name,
            "segment_ids": seg_ids,
            "segment_start": min(
                (seg_by_id[s]["segment_start"] for s in seg_ids if s in seg_by_id),
                default=None,
            ),
            "segment_text": seg_text,
            "ocr_text": read["text"],
            "ocr_line_count": read.get("line_count", 0),
            "ocr_seconds": read.get("seconds"),
            "subtitle_coverage": None if cov_sub is None else round(cov_sub, 3),
            "frame_coverage": None if cov_frame is None else round(cov_frame, 3),
            "divergent": bool(
                cov_sub is not None
                and cov_sub < DIVERGENCE_THRESHOLD
                and len(read["text"]) >= 40
            ),
            "error": read.get("error"),
        })
        if i % 10 == 0:
            log("OCR %d/%d ..." % (i, len(files)))

    elapsed = time.time() - started_all

    # 关键词 -> 帧，便于反向检索
    keyword_index: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        norm = normalize_text(rec["ocr_text"])
        for kw in _candidate_terms(rec["ocr_text"]):
            if kw in norm:
                keyword_index[kw].append(rec["frame"])

    return {
        "job_id": idx.get("job_id"),
        "source_index": index_name,
        "engine": "rapidocr-onnxruntime",
        "frames_total": len(files),
        "frames_read": len(records),
        "seconds_total": round(elapsed, 1),
        "seconds_per_frame": round(elapsed / max(1, len(records)), 2),
        "divergence_threshold": DIVERGENCE_THRESHOLD,
        "divergent_count": sum(1 for r in records if r["divergent"]),
        "frames": records,
        "keyword_index": dict(sorted(keyword_index.items())),
    }

def _candidate_terms(text: str, min_len: int = 2, max_len: int = 8) -> set[str]:
    """从 OCR 文本里抽取可供检索的候选词。

    不引入中文分词依赖：按标点/空白切块，再对每块取长度窗口。
    宁可多切一些候选，也不漏检——检索是给人用的，噪声由排序处理。
    """
    terms: set[str] = set()
    for piece in re.split(r"[\s,，。、；;：:！!？?（）()\[\]【】/|·\-—…]+", text or ""):
        piece = normalize_text(piece)
        if len(piece) < min_len:
            continue
        if len(piece) <= max_len:
            terms.add(piece)
        for n in range(min_len, min(max_len, len(piece)) + 1):
            for i in range(len(piece) - n + 1):
                terms.add(piece[i:i + n])
    return terms


def search_ocr(payload: dict[str, Any], query: str, *, limit: int = 20) -> list[dict[str, Any]]:
    """在 OCR 结果里检索。返回按时间排序的命中帧。"""
    q = normalize_text(query)
    if not q:
        return []
    hits = []
    for rec in payload.get("frames", []):
        if q in normalize_text(rec["ocr_text"]):
            hits.append({
                "frame": rec["frame"],
                "segment_ids": rec["segment_ids"],
                "segment_start": rec["segment_start"],
                "matched": q,
                "ocr_excerpt": _excerpt(rec["ocr_text"], q),
            })
    hits.sort(key=lambda h: (h["segment_start"] is None, h["segment_start"] or 0))
    return hits[:limit]


def _excerpt(text: str, query: str, width: int = 40) -> str:
    """在原文里定位查询串（忽略标点），给出上下文。"""
    norm = normalize_text(text)
    pos = norm.find(query)
    if pos < 0:
        return text[:width * 2]
    # 用归一化位置近似映射回原文
    ratio = len(text) / max(1, len(norm))
    center = int(pos * ratio)
    start = max(0, center - width)
    end = min(len(text), center + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return "%s%s%s" % (prefix, text[start:end], suffix)


def write_ocr_index(job_dir: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(job_dir) / "ocr.index.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def to_markdown(payload: dict[str, Any], *, title: str = "画面文字证据") -> str:
    """给人读的视图。分歧片段单独成节——那是纯转录必然漏掉的部分。"""
    lines = [
        "# %s" % title,
        "",
        "> 由脚本生成：文字来自画面 OCR，**未做语义解释**。",
        "> 配对靠时间（字幕区间 ∩ 画面曝光段），不靠文字相似度。",
        "",
        "- 引擎：%s" % payload.get("engine"),
        "- 帧数：%d，总耗时 %.1fs（平均 %.2fs/帧）"
        % (payload.get("frames_total", 0), payload.get("seconds_total", 0),
           payload.get("seconds_per_frame", 0)),
        "- 口播与画面无明显关联的帧：%d（阈值 %.2f）"
        % (payload.get("divergent_count", 0), payload.get("divergence_threshold", 0)),
        "",
    ]

    divergent = [r for r in payload.get("frames", []) if r.get("divergent")]
    if divergent:
        lines += [
            "## 口播与画面内容无关的片段",
            "",
            "这些位置的屏幕信息**无法从转录得到**，最值得优先阅读：",
            "",
        ]
        for rec in divergent:
            t = rec.get("segment_start")
            lines.append("- **%s** @ %s" % (rec["frame"], "%.0fs" % t if t is not None else "?"))
            lines.append("  - 口播：%s" % (rec.get("segment_text") or "（无）"))
            lines.append("  - 画面：%s" % rec["ocr_text"][:160])
        lines.append("")

    lines += ["## 逐帧画面文字", ""]
    for rec in payload.get("frames", []):
        t = rec.get("segment_start")
        header = "### %s @ %s" % (rec["frame"], "%.0fs" % t if t is not None else "?")
        lines.append(header)
        lines.append("")
        if rec.get("error"):
            lines.append("- OCR 失败：%s" % rec["error"])
            lines.append("")
            continue
        lines.append("- 关联片段：%s" % (", ".join(rec["segment_ids"]) or "（无）"))
        if rec.get("subtitle_coverage") is not None:
            lines.append("- 字幕覆盖率：%.0f%%%s"
                         % (rec["subtitle_coverage"] * 100,
                            "（口播与画面无关）" if rec.get("divergent") else ""))
        lines.append("")
        lines.append(rec["ocr_text"] or "_（未识别到文字）_")
        lines.append("")
    return "\n".join(lines)
