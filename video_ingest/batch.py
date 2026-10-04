"""多分P批处理的分页参数解析与调度辅助。

设计约束（bootstrap §7.1）：
- **只接受显式范围或列表**，不允许"没指定就全下"——那会悄悄下载整个合集。
- 每个分P独立记录状态，单个分P失败不能中断整批。
- 已处理完成的分P可跳过，不重复跑 ASR。
"""

from __future__ import annotations

from typing import Any, Iterable

MAX_PAGES = 500          # 单次批处理的硬上限，防止误操作拉爆磁盘与时间


class PageSpecError(ValueError):
    pass


def parse_page_spec(spec: str, *, total: int | None = None) -> list[int]:
    """解析分P选择表达式。

    支持：单值 "3"、列表 "3,7"、区间 "1-5"、混合 "1-3,7,10-12"。
    返回去重后按升序排列的页码列表。空表达式一律报错——
    "不指定"不等于"全部"，这是刻意的。
    """
    if spec is None or not str(spec).strip():
        raise PageSpecError(
            "必须显式指定分P范围，例如 --pages 1-5 或 --pages 3,7。"
            "不提供'全部'默认值，以免误下整个合集。"
        )

    pages: set[int] = set()
    for chunk in str(spec).split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            lo_s, hi_s = lo_s.strip(), hi_s.strip()
            if not lo_s.isdigit() or not hi_s.isdigit():
                raise PageSpecError("区间写法非法: %r（应为 1-5 这种形式）" % part)
            lo, hi = int(lo_s), int(hi_s)
            if lo < 1 or hi < 1:
                raise PageSpecError("分P从 1 开始，收到: %r" % part)
            if hi < lo:
                raise PageSpecError("区间上下界颠倒: %r" % part)
            pages.update(range(lo, hi + 1))
        else:
            if not part.isdigit():
                raise PageSpecError("分P必须是正整数，收到: %r" % part)
            n = int(part)
            if n < 1:
                raise PageSpecError("分P从 1 开始，收到: %r" % part)
            pages.add(n)

    if not pages:
        raise PageSpecError("未能从 %r 解析出任何分P" % spec)

    result = sorted(pages)
    if len(result) > MAX_PAGES:
        raise PageSpecError(
            "单次最多处理 %d 个分P，本次请求 %d 个。请缩小范围分批执行。"
            % (MAX_PAGES, len(result))
        )
    if total is not None:
        missing = [p for p in result if p > total]
        if missing:
            raise PageSpecError(
                "该视频只有 %d 个分P，但请求了: %s"
                % (total, ", ".join(str(m) for m in missing))
            )
    return result


def summarize_batch(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """汇总批处理结果。

    逐个分P统计，不把"整批失败"当成单一状态——部分成功是常态。
    """
    rows = list(results)
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r.get("status", "unknown")] = by_status.get(r.get("status", "unknown"), 0) + 1

    failed = [r for r in rows if r.get("status") in ("failed", "blocked")]
    return {
        "requested": len(rows),
        "by_status": by_status,
        "ok_count": by_status.get("ok", 0),
        "reused_count": by_status.get("reused", 0),
        "failed_count": len(failed),
        "failed_pages": [
            {"page": r.get("page"), "status": r.get("status"),
             "kind": r.get("kind"), "message": r.get("message")}
            for r in failed
        ],
        "all_ok": len(failed) == 0,
    }


def should_reuse(manifest: dict[str, Any] | None, *, force: bool = False) -> bool:
    """判断某个分P是否可以复用已有产物。

    只在"确实处理完成"时复用。partial/failed/blocked 一律重跑，
    否则会把上次的失败静默当成成功。
    """
    if force or not manifest:
        return False
    return manifest.get("processing_status") == "resource_processed"
