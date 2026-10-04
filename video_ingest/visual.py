"""画面帧提取与字幕配对。

职责：把"字幕时间轴"与"画面时间轴"对齐，产出可复核的视觉证据。
不调用任何模型：只有解码、指纹、阈值判定。图像"理解"由 Agent 完成。

核心概念
--------
曝光段（exposure run）：画面内容稳定的最长连续区间 [t0, t1]。
    由相邻帧指纹的平均绝对差（MAD）超过阈值来切分。

配对规则：**区间求交**，不是按时间点取帧。
    一条字幕可能跨越转场（对应多张画面），一个画面也可能被多条字幕引用，
    所以字幕与画面是 M:N 关系。按 cue.start 单点取帧在实测中 83% 会落在
    画面变化点 1s 内（转场中），取到的基本是空画面。

帧状态：ok / mid_transition / blank / high_change
    用于让"取错了"可被机器检出，而不是假设取对了。
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

# numpy 只在需要它的函数内部导入。
# 效果：曝光段分割、区间求交、取样规划、状态判定这些纯逻辑不依赖任何
# 第三方库，测试可在最小环境运行（CI 因此不必安装 GB 级依赖）。

# 指纹尺寸：只用于变化检测与粗略分类，不用于读取文字
SIG_W, SIG_H = 64, 36

# 采样步长（秒）。逐帧解码但按步长取样，落点精度只取决于该值。
DEFAULT_INTERVAL = 0.4

# 判定画面变化：相邻指纹平均绝对差阈值（0-255 灰度）
DEFAULT_SCENE_THRESHOLD = 6.0

# 取样点距最近变化点小于该值，判为落在转场中
TRANSITION_GUARD = 0.3

# 曝光段短于该值时，整段都应视为过渡/闪烁，而不是稳定画面
SHORT_RUN = 0.6

# 空白帧判据：亮度极端 + **完全没有结构**（std 与 edge 都极低）
# 注意：黑底白字的幻灯片亮度也很低，但 std/edge 高，因此不会被误判为空白。
BLANK_MEAN_LO = 24.0
BLANK_MEAN_HI = 238.0
BLANK_SIGMA_MAX = 12.0
BLANK_EDGE_MAX = 0.8

# 低细节：有画面但几乎没有可读内容
LOW_DETAIL_EDGE = 0.6

# 段内指纹方差超过该值判为高动态（动画/滚动）
HIGH_CHANGE_VAR = 40.0


def to_rgb(frame: Any):
    import numpy as np

    return np.ascontiguousarray(frame.to_ndarray(format="rgb24"))


def signature(rgb) -> Any:
    """盒式降采样灰度指纹。用于变化检测，不用于阅读文字。"""
    import numpy as np

    h, w, _ = rgb.shape
    gray = rgb.mean(axis=2)
    ys = np.linspace(0, h, SIG_H + 1).astype(int)
    xs = np.linspace(0, w, SIG_W + 1).astype(int)
    ys[-1] = h
    xs[-1] = w
    out = np.empty((SIG_H, SIG_W), dtype=np.float32)
    for i in range(SIG_H):
        y0, y1 = ys[i], max(ys[i] + 1, ys[i + 1])
        band = gray[y0:y1]
        for j in range(SIG_W):
            x0, x1 = xs[j], max(xs[j] + 1, xs[j + 1])
            out[i, j] = band[:, x0:x1].mean()
    return out


def frame_stats(rgb) -> dict[str, float]:
    """帧的粗略统计量。在降采样灰度上进行，成本固定。

    edge 是横向+纵向平均梯度，用来区分"黑底白字"与"纯黑"——
    只靠亮度会把两者混为一谈。
    """
    import numpy as np

    g = _downsample_gray(rgb)
    mean = float(g.mean())
    std = float(g.std())
    edge = float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())
    return {"mean": round(mean, 2), "std": round(std, 2), "edge": round(edge, 3)}


def _downsample_gray(rgb):
    import numpy as np

    h, w, _ = rgb.shape
    step_y = max(1, h // (SIG_H * 2))
    step_x = max(1, w // (SIG_W * 2))
    gray = rgb[::step_y, ::step_x].mean(axis=2)
    return gray.astype(np.float32)


def classify_sig(stats: dict[str, float]) -> str:
    """依据统计量把帧分类为 blank / low-detail / content。

    blank 要求**同时**满足亮度极端与完全没有结构。只看亮度会把
    黑底白字的幻灯片（教学视频里极常见）误判成空白，因此 std 与 edge
    是必要条件：有文字就一定有结构和边缘。
    """
    extreme = stats["mean"] < BLANK_MEAN_LO or stats["mean"] > BLANK_MEAN_HI
    if extreme and stats["std"] < BLANK_SIGMA_MAX and stats["edge"] < BLANK_EDGE_MAX:
        return "blank"
    if stats["edge"] < LOW_DETAIL_EDGE:
        return "low-detail"
    return "content"


# --------------------------------------------------------------------------
# 画面时间轴
# --------------------------------------------------------------------------

def scan_timeline(
    video_path: str | Path,
    *,
    interval: float = DEFAULT_INTERVAL,
    start: float = 0.0,
    end: float | None = None,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    on_progress: Any = None,
) -> dict[str, Any]:
    """逐帧解码，按步长采样，产出画面时间轴。

    不用 container.seek()：seek 依赖关键帧，落点会偏到最近的关键帧，
    而关键帧位置由压缩效率决定，与画面内容起点无关。
    """
    import av
    import numpy as np

    if interval <= 0:
        raise ValueError("interval 必须为正数")

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30.0)
        samples: list[dict[str, Any]] = []
        prev_sig = None
        next_t = start
        decoded = 0

        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if t < next_t:
                continue
            if end is not None and t > end:
                break
            decoded += 1

            rgb = to_rgb(frame)
            sig = signature(rgb)
            stats = frame_stats(rgb)

            diff = float(np.abs(sig - prev_sig).mean()) if prev_sig is not None else 0.0
            samples.append({
                "t": round(t, 3),
                "diff": round(diff, 2),
                "changed": bool(diff >= scene_threshold),
                "kind": classify_sig(stats),
                "sig": sig,
            })
            prev_sig = sig
            next_t += interval

            if on_progress and decoded % 50 == 0:
                on_progress("scanned %.0fs / %s" % (t, end if end is not None else "end"))
    finally:
        container.close()

    if not samples:
        raise ValueError("未取到任何画面样本（区间为空或解码失败）")

    runs = build_exposure_runs(samples)
    return {
        "video": str(Path(video_path).resolve()),
        "fps": fps,
        "interval": interval,
        "scene_threshold": scene_threshold,
        "window": {"start": start, "end": end if end is not None else samples[-1]["t"]},
        "sample_count": len(samples),
        "samples": samples,
        "runs": runs,
    }


def build_exposure_runs(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把采样序列切成曝光段（画面内容稳定的连续区间）。

    段边界取相邻采样点的中点，减少步长带来的量化误差。
    """
    if not samples:
        return []

    runs: list[dict[str, Any]] = []
    cur_start_idx = 0

    def close(end_idx: int) -> None:
        seg = samples[cur_start_idx:end_idx + 1]
        if not seg:
            return
        # 段起点取本段首样本，终点取下一段首样本的中点（或末样本）
        if end_idx + 1 < len(samples):
            t_end = (samples[end_idx]["t"] + samples[end_idx + 1]["t"]) / 2.0
        else:
            t_end = samples[end_idx]["t"]
        stats = [s["sig"] for s in seg]
        var = _variance_of(stats)
        kinds = [s["kind"] for s in seg]
        runs.append({
            "run_id": "run-%04d" % (len(runs) + 1),
            "start": round(float(seg[0]["t"]), 3),
            "end": round(float(t_end), 3),
            "sample_count": len(seg),
            "kinds": kinds,
            "dominant_kind": max(set(kinds), key=kinds.count),
            "sig_variance": round(var, 2),
            "high_change": bool(var >= HIGH_CHANGE_VAR),
        })

    for i in range(1, len(samples)):
        if samples[i]["changed"]:
            close(i - 1)
            cur_start_idx = i
    close(len(samples) - 1)
    return runs


def _variance_of(sigs: list[Any]) -> float:
    """段内指纹的离散程度（逐点标准差再取均值）。

    优先用 numpy（真实指纹是 ndarray）；对不支持的输入回退到纯 Python，
    使该判据在无 numpy 的环境里仍可测。
    """
    if len(sigs) < 2:
        return 0.0
    try:
        import numpy as np

        arr = np.stack(sigs)
        return float(arr.std(axis=0).mean())
    except Exception:  # noqa: BLE001
        return _variance_of_plain(sigs)


def _variance_of_plain(sigs: list[Any]) -> float:
    """纯 Python 版本的段内离散度，支持 (rows, cols) 形状或标量序列。

    sigs 若为实现了 shape/取值协议的对象，这里按其 shape 展开取值。
    """
    values: list[list[float]] = []
    for s in sigs:
        shape = getattr(s, "shape", None)
        if shape:
            n = 1
            for d in shape:
                n *= int(d)
            v = getattr(s, "value", None)
            if v is not None:
                values.append([float(v)] * n)
                continue
        try:
            values.append([float(x) for x in s])
        except TypeError:
            values.append([float(s)])

    if not values:
        return 0.0
    width = min(len(v) for v in values)
    if width == 0:
        return 0.0

    total = 0.0
    for j in range(width):
        col = [v[j] for v in values]
        m = sum(col) / len(col)
        var = sum((x - m) ** 2 for x in col) / len(col)
        total += var ** 0.5
    return total / width


def _uniform_indices(total: int, count: int) -> list[int]:
    """在 [0, total) 上均匀取 count 个下标（含两端），纯 Python 实现。

    用于 max_frames 抽稀：**均匀抽稀而不是砍掉尾部**，以免时间轴后半段
    完全没有画面证据。用纯 Python 是为了让本模块的规划逻辑不依赖 numpy，
    从而在最小环境里可测。count >= total 时返回全部下标。

    ≥1 时返回值必然包含 0 与 total-1，允许重复（total 很小时）。
    """
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


# --------------------------------------------------------------------------
# 区间求交取帧
# --------------------------------------------------------------------------

def intersect_runs(
    runs: list[dict[str, Any]], start: float, end: float, *, min_overlap: float = 0.15
) -> list[dict[str, Any]]:
    """返回与 [start, end] 有实质交集的曝光段，按时间排序。

    min_overlap 过滤掉仅"蹭到"边界的段，避免为 0.05s 的擦边交集取帧。
    """
    out = []
    for run in runs:
        lo = max(start, run["start"])
        hi = min(end, run["end"])
        if hi - lo >= min_overlap:
            out.append({**run, "overlap_start": round(lo, 3), "overlap_end": round(hi, 3),
                        "overlap": round(hi - lo, 3)})
    out.sort(key=lambda r: r["overlap_start"])
    return out


def choose_sample_time(overlap: dict[str, Any]) -> float:
    """取样时刻 = 曝光段的中点（不是交集中点）。

    为什么用曝光段中点：
    1. 曝光段是**原子视觉单位**——同一段就是同一张画面。若按各片段的交集中点
       取样，一张被 N 条字幕引用的幻灯片会被提取 N 次，浪费 N 倍视觉模型调用，
       而且同一个画面出现多个 frame_id，破坏"帧 ↔ 视觉状态"的一一对应。
    2. 中点对转场免疫，始终落在内容稳定区内，远离两端的变化点。

    代价：当曝光段远长于某个字幕片段时，帧仍在同一画面内，但可能与那条字幕
    相隔较远。因此每个条目同时记录 overlap 区间，供阅读时判断时间相关性。
    """
    return round((overlap["start"] + overlap["end"]) / 2.0, 3)


def sample_status(
    sample_t: float,
    runs: list[dict[str, Any]],
    *,
    guard: float = TRANSITION_GUARD,
    run_high_change: bool = False,
) -> str:
    """判定取样点的帧状态。

    mid_transition：该曝光段过短（< SHORT_RUN），整段属于过渡/闪烁，
        不能代表"稳定画面"。**这是"建议复核"标记，不是"丢弃"**——实测被标
        的帧文字仍可能可读，只是可能含弹入/淡入残留或只是短暂一闪。
    blank：该段画面确实没有结构（纯黑/纯白），不值得送模型
    high_change：段内画面剧变（动画/滚动），单帧可能不代表整段
    ok：可用
    """
    for run in runs:
        if run["start"] <= sample_t <= run["end"]:
            if run["dominant_kind"] == "blank":
                return "blank"

            span = run["end"] - run["start"]
            # 段太短：无论取样点在哪，这一段都不是稳定画面。
            # 注意不能用"距边界距离"判断——取样点固定在段中点，
            # 相对位置恒为 0.5，那种判据永远不会触发。
            if span < SHORT_RUN:
                return "mid_transition"
            if min(sample_t - run["start"], run["end"] - sample_t) < guard:
                return "mid_transition"

            if run.get("high_change"):
                return "high_change"
            return "ok"
    return "mid_transition"


def plan_frames(
    segments: list[dict[str, Any]],
    timeline: dict[str, Any],
    *,
    include_blank: bool = False,
    max_frames: int = 0,
) -> dict[str, Any]:
    """为每个片段规划取样点，并给出未被任何片段引用的孤儿曝光段。"""
    runs = timeline["runs"]
    plan: list[dict[str, Any]] = []
    referenced: set[str] = set()

    for seg in segments:
        overlaps = intersect_runs(runs, seg["start"], seg["end"])
        entries = []
        for ov in overlaps:
            referenced.add(ov["run_id"])
            sample_t = choose_sample_time(ov)
            status = sample_status(sample_t, runs, run_high_change=ov.get("high_change", False))
            if status == "blank" and not include_blank:
                entries.append({
                    "run_id": ov["run_id"], "sample_t": sample_t, "status": status,
                    "exposure_run": [ov["start"], ov["end"]],
                    "overlap": [ov["overlap_start"], ov["overlap_end"]],
                    "skipped": True,
                    "skip_reason": "画面为空白/过渡，不提取",
                })
                continue
            entries.append({
                "run_id": ov["run_id"], "sample_t": sample_t, "status": status,
                "exposure_run": [ov["start"], ov["end"]],
                "overlap": [ov["overlap_start"], ov["overlap_end"]],
                "skipped": False,
            })
        plan.append({
            "segment_id": seg["id"],
            "segment_start": seg["start"],
            "segment_end": seg["end"],
            "text": seg["text"],
            "frames": entries,
        })

    orphans = [r for r in runs if r["run_id"] not in referenced]

    # 去重：同一 (run_id, sample_t) 只应提取一次，供多个片段共享
    unique: dict[tuple, dict[str, Any]] = {}
    for item in plan:
        for fr in item["frames"]:
            key = (fr["run_id"], fr["sample_t"])
            fr["frame_id"] = "frame-%s-%d" % (fr["run_id"], int(round(fr["sample_t"] * 1000)))
            if not fr["skipped"]:
                unique.setdefault(key, fr)

    pending = list(unique.values())
    truncated = 0
    if max_frames and len(pending) > max_frames:
        truncated = len(pending) - max_frames
        # 均匀抽稀，保持时间轴覆盖，而不是砍掉尾部
        keep_idx = set(_uniform_indices(len(pending), max_frames))
        pending = [f for i, f in enumerate(pending) if i in keep_idx]

    return {
        "plan": plan,
        "unique_frames": pending,
        "unique_frame_count": len(pending),
        "orphan_runs": [
            {"run_id": r["run_id"], "start": r["start"], "end": r["end"],
             "dominant_kind": r["dominant_kind"]}
            for r in orphans
        ],
        "orphan_run_count": len(orphans),
        "truncated_by_max_frames": truncated,
    }


# --------------------------------------------------------------------------
# 帧落盘
# --------------------------------------------------------------------------

def write_png(path: Path, rgb) -> None:
    """用标准库写 PNG，避免为取帧引入 Pillow 依赖。"""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _resize_nearest(rgb, scale: float):
    import numpy as np

    if scale >= 0.999:
        return rgb
    h, w, _ = rgb.shape
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    ys = np.linspace(0, h - 1, nh).astype(int)
    xs = np.linspace(0, w - 1, nw).astype(int)
    return np.ascontiguousarray(rgb[np.ix_(ys, xs)])


def extract_frames(
    video_path: str | Path,
    frames: list[dict[str, Any]],
    out_dir: Path,
    *,
    scale: float = 1.0,
    on_progress: Any = None,
) -> dict[str, Any]:
    """按规划好的时间点提取帧。

    用一次顺序解码完成全部提取（按目标时间排序），避免逐帧 seek 的
    关键帧偏差与重复解码开销。
    """
    import av

    out_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted(frames, key=lambda f: f["sample_t"])
    if not targets:
        return {"written": 0, "frames": []}

    container = av.open(str(video_path))
    results: list[dict[str, Any]] = []
    try:
        stream = container.streams.video[0]
        ti = 0
        for frame in container.decode(video=0):
            if frame.pts is None or ti >= len(targets):
                continue
            t = float(frame.pts * stream.time_base)
            while ti < len(targets) and t >= targets[ti]["sample_t"]:
                target = targets[ti]
                rgb = _resize_nearest(to_rgb(frame), scale)
                path = out_dir / ("%s.png" % target["frame_id"])
                write_png(path, rgb)
                results.append({
                    **{k: v for k, v in target.items() if k != "sig"},
                    "t_actual": round(t, 3),
                    "path": path.name,
                    "file": str(path),
                    "size_bytes": path.stat().st_size,
                })
                ti += 1
                if on_progress and ti % 25 == 0:
                    on_progress("extracted %d/%d frames" % (ti, len(targets)))
    finally:
        container.close()

    missing = len(targets) - len(results)
    return {"written": len(results), "missing": missing, "frames": results}


def load_segments(job_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(job_dir) / "transcript.raw.json"
    if not path.exists():
        raise FileNotFoundError("缺少 transcript.raw.json: %s" % path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("segments") or []


def to_json(payload: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
