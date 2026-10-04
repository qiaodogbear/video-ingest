"""画面帧对齐逻辑的单元测试。不依赖网络，也不需要真实视频。

重点覆盖"字幕与画面是 M:N 关系"这一核心设计：
- 一条字幕跨越转场时必须取到多帧；
- 一个画面被多条字幕引用时必须去重、只提取一次；
- 取样点必须落在曝光段内部，而不是曝光段边界（转场）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_ingest.visual import (  # noqa: E402
    HIGH_CHANGE_VAR,
    build_exposure_runs,
    choose_sample_time,
    classify_sig,
    intersect_runs,
    plan_frames,
    sample_status,
)


def make_samples(spec, *, interval=0.4, threshold=6.0):
    """用 (时刻, 指纹值) 序列构造采样点。

    指纹用 0-255 的常数填充；相邻不同值即产生变化点。
    """
    samples = []
    prev = None
    for t, val, kind in spec:
        sig = np.full((4, 4), float(val), dtype=np.float32)
        diff = float(np.abs(sig - prev).mean()) if prev is not None else 0.0
        samples.append({
            "t": t, "diff": round(diff, 2), "changed": bool(diff >= threshold),
            "kind": kind, "sig": sig,
        })
        prev = sig
    return samples


# --------------------------------------------------------------------------
# 曝光段分割
# --------------------------------------------------------------------------

def test_build_runs_splits_on_change():
    samples = make_samples([
        (0.0, 10, "content"), (0.4, 10, "content"), (0.8, 10, "content"),
        (1.2, 200, "content"), (1.6, 200, "content"),   # 变化点
        (2.0, 10, "content"),
    ])
    runs = build_exposure_runs(samples)
    assert len(runs) == 3
    assert runs[0]["start"] == 0.0
    # 段边界取相邻采样点中点（0.8 与 1.2 之间 = 1.0）
    assert runs[0]["end"] == pytest.approx(1.0)
    assert runs[1]["start"] == pytest.approx(1.2)
    assert runs[2]["start"] == pytest.approx(2.0)


def test_build_runs_ignores_below_threshold_diff():
    samples = make_samples([
        (0.0, 10, "content"), (0.4, 10, "content"), (0.8, 11, "content"),
        (1.2, 10, "content"),
    ])
    runs = build_exposure_runs(samples)
    assert len(runs) == 1, "低于阈值的变化不应切段"


def test_build_runs_empty_input():
    assert build_exposure_runs([]) == []


def test_build_runs_single_sample():
    runs = build_exposure_runs(make_samples([(3.0, 10, "content")]))
    assert len(runs) == 1
    assert runs[0]["start"] == 3.0 and runs[0]["end"] == 3.0


def test_build_runs_marks_blank_dominant():
    samples = make_samples([
        (0.0, 5, "blank"), (0.4, 5, "blank"), (0.8, 250, "blank"),
    ])
    runs = build_exposure_runs(samples)
    # 亮度接近但 std 为 0、edge 为 0 → 判为 blank 段
    assert runs[0]["dominant_kind"] == "blank"


def test_build_runs_detects_high_change_within_run():
    """段内指纹剧变（动画/滚动）应被标记，提示单帧可能不代表整段。

    构造 0/200 交替序列：每步变化 200，但把阈值设得更高使其不切段；
    段内逐像素标准差的均值足够大，应被标为 high_change。
    注意 sig_variance 是"逐像素标准差再取均值"，不是帧间差值的均值。
    """
    spec = [(i * 0.4, 0 if i % 2 == 0 else 200, "content") for i in range(8)]
    samples = make_samples(spec, threshold=250.0)
    runs = build_exposure_runs(samples)
    assert len(runs) == 1, "每步变化都低于阈值时不应切段"
    assert runs[0]["high_change"] is True
    assert runs[0]["sig_variance"] >= HIGH_CHANGE_VAR


# --------------------------------------------------------------------------
# 区间求交
# --------------------------------------------------------------------------

RUNS = [
    {"run_id": "run-0001", "start": 0.0, "end": 5.0, "dominant_kind": "content", "high_change": False},
    {"run_id": "run-0002", "start": 5.0, "end": 12.0, "dominant_kind": "content", "high_change": False},
    {"run_id": "run-0003", "start": 12.0, "end": 20.0, "dominant_kind": "content", "high_change": False},
]


def test_intersect_returns_overlapping_runs():
    out = intersect_runs(RUNS, 4.0, 6.0)
    assert [r["run_id"] for r in out] == ["run-0001", "run-0002"]
    assert out[0]["overlap_start"] == 4.0 and out[0]["overlap_end"] == 5.0
    assert out[1]["overlap_start"] == 5.0 and out[1]["overlap_end"] == 6.0


def test_intersect_inside_single_run():
    out = intersect_runs(RUNS, 6.0, 8.0)
    assert len(out) == 1
    assert out[0]["run_id"] == "run-0002"


def test_intersect_filters_trivial_touch():
    """只为 0.05s 的擦边交集取帧是浪费，应被过滤。"""
    out = intersect_runs(RUNS, 11.95, 12.05)
    assert out == [] or all(r["overlap"] >= 0.15 for r in out)


def test_intersect_no_overlap():
    assert intersect_runs(RUNS, 50.0, 60.0) == []


def test_intersect_sorted_by_time():
    out = intersect_runs(RUNS, 1.0, 15.0)
    starts = [r["overlap_start"] for r in out]
    assert starts == sorted(starts)


# --------------------------------------------------------------------------
# 取样点
# --------------------------------------------------------------------------

def test_choose_sample_time_is_exposure_run_midpoint():
    """取样点是曝光段中点——曝光段是原子视觉单位。"""
    ov = {"start": 5.0, "end": 12.0, "overlap_start": 5.0, "overlap_end": 6.5}
    assert choose_sample_time(ov) == 8.5


def test_sample_time_lies_inside_exposure_run_not_on_boundary():
    """核心保证：取样点必须在曝光段内部。

    这是不用 cue.start + 固定偏移的原因——固定偏移会落到转场里。
    """
    for cue_start, cue_end in [(4.5, 6.5), (6.0, 7.0), (11.0, 13.0)]:
        for ov in intersect_runs(RUNS, cue_start, cue_end):
            t = choose_sample_time(ov)
            run = next(r for r in RUNS if r["run_id"] == ov["run_id"])
            assert run["start"] <= t <= run["end"]


# --------------------------------------------------------------------------
# 帧状态
# --------------------------------------------------------------------------

def test_sample_status_ok_for_midpoint():
    assert sample_status(6.0, RUNS) == "ok"


def test_sample_status_mid_transition_near_boundary_of_long_run():
    # 5.0 是 run-0001/run-0002 的边界；长段内贴近边界应标记复核
    assert sample_status(5.1, RUNS) == "mid_transition"
    assert sample_status(11.9, RUNS) == "mid_transition"


def test_sample_status_short_run_is_flagged_regardless_of_position():
    """短段整段视为过渡，无论取样点在哪。

    早期实现判定"取样点距边界是否过近"，而取样点固定在段中点，
    相对位置恒为 0.5，导致短段永远不会被标记（实测 102 帧全部 ok）。
    """
    runs = [{"run_id": "r", "start": 10.0, "end": 10.4, "dominant_kind": "content",
             "high_change": False}]
    assert sample_status(10.2, runs) == "mid_transition"   # 中点也必须标


def test_sample_status_long_run_midpoint_is_ok():
    runs = [{"run_id": "r", "start": 10.0, "end": 10.8, "dominant_kind": "content",
             "high_change": False}]
    assert sample_status(10.4, runs) == "ok"


def test_sample_status_long_run_near_boundary_is_flagged():
    runs = [{"run_id": "r", "start": 10.0, "end": 20.0, "dominant_kind": "content",
             "high_change": False}]
    assert sample_status(10.1, runs) == "mid_transition"


def test_sample_status_degenerate_zero_length_run():
    runs = [{"run_id": "r", "start": 10.0, "end": 10.0, "dominant_kind": "content",
             "high_change": False}]
    assert sample_status(10.0, runs) == "mid_transition"


def test_sample_status_blank_run():
    runs = [{"run_id": "r", "start": 0.0, "end": 5.0, "dominant_kind": "blank",
             "high_change": False}]
    assert sample_status(2.5, runs) == "blank"


def test_sample_status_high_change():
    runs = [{"run_id": "r", "start": 0.0, "end": 5.0, "dominant_kind": "content",
             "high_change": True}]
    assert sample_status(2.5, runs) == "high_change"


def test_sample_status_outside_all_runs():
    assert sample_status(100.0, RUNS) == "mid_transition"


def test_classify_sig_blank_requires_no_structure_at_all():
    """blank 需要同时满足亮度极端与完全没有结构。

    只看亮度会把黑底白字的幻灯片误判为空白——教学视频里这是常态。
    """
    assert classify_sig({"mean": 5.0, "std": 1.0, "edge": 0.2}) == "blank"
    assert classify_sig({"mean": 252.0, "std": 2.0, "edge": 0.1}) == "blank"
    # 纯黑但仍有文字结构 → 不是空白
    assert classify_sig({"mean": 5.0, "std": 40.0, "edge": 5.0}) == "content"
    # 亮底幻灯片（有文字）→ 不是空白
    assert classify_sig({"mean": 245.0, "std": 30.0, "edge": 3.0}) == "content"
    # 亮度正常 → 不是空白
    assert classify_sig({"mean": 128.0, "std": 1.0, "edge": 0.1}) == "low-detail"


def test_classify_sig_low_detail():
    assert classify_sig({"mean": 128.0, "std": 20.0, "edge": 0.3}) == "low-detail"


def test_classify_sig_content():
    assert classify_sig({"mean": 200.0, "std": 50.0, "edge": 4.0}) == "content"


# --------------------------------------------------------------------------
# 规划：M:N 配对、去重、空白跳过、孤儿段
# --------------------------------------------------------------------------

TIMELINE = {"runs": RUNS}


def test_plan_gives_multiple_frames_when_cue_spans_transition():
    """一条字幕跨越转场 → 必须取到多帧，单点取帧会漏内容。"""
    segments = [{"id": "seg-000001", "start": 4.0, "end": 6.5, "text": "跨转场"}]
    plan = plan_frames(segments, TIMELINE)
    frames = plan["plan"][0]["frames"]
    assert len(frames) == 2, "跨越两个曝光段必须取两帧"
    assert {f["run_id"] for f in frames} == {"run-0001", "run-0002"}


def test_plan_dedupes_frame_shared_by_segments():
    """一个画面被多条字幕引用时，只应提取一次。

    这是"曝光段为原子视觉单位"的直接后果：同一段就是同一张画面，
    被 N 条字幕引用也只提取 1 帧，否则会浪费 N 倍视觉模型调用。
    """
    segments = [
        {"id": "seg-000001", "start": 6.0, "end": 7.0, "text": "A"},
        {"id": "seg-000002", "start": 6.5, "end": 8.0, "text": "B"},
    ]
    plan = plan_frames(segments, TIMELINE)
    assert len(plan["plan"]) == 2
    total_listed = sum(len(p["frames"]) for p in plan["plan"])
    assert total_listed == 2                    # 每个片段各引用一次
    assert plan["unique_frame_count"] == 1      # 但只提取一次
    ids = {f["frame_id"] for p in plan["plan"] for f in p["frames"]}
    assert len(ids) == 1
    # 两条字幕引用同一画面时，取样时刻必须相同
    times = {f["sample_t"] for p in plan["plan"] for f in p["frames"]}
    assert len(times) == 1


def test_plan_distinct_segments_in_same_run_share_one_frame():
    """同一曝光段内的不同交集区间，仍只能得到一个 frame_id。"""
    segments = [
        {"id": "seg-000001", "start": 5.0, "end": 6.0, "text": "前半"},
        {"id": "seg-000002", "start": 10.0, "end": 11.5, "text": "后半"},
    ]
    plan = plan_frames(segments, TIMELINE)
    assert plan["unique_frame_count"] == 1
    assert plan["plan"][0]["frames"][0]["sample_t"] == plan["plan"][1]["frames"][0]["sample_t"]


def test_plan_skips_blank_frames_by_default():
    runs = [{"run_id": "run-0001", "start": 0.0, "end": 10.0,
             "dominant_kind": "blank", "high_change": False}]
    segments = [{"id": "seg-000001", "start": 1.0, "end": 3.0, "text": "过渡"}]
    plan = plan_frames(segments, {"runs": runs})
    assert plan["unique_frame_count"] == 0
    assert plan["plan"][0]["frames"][0]["skipped"] is True
    assert plan["plan"][0]["frames"][0]["skip_reason"]


def test_plan_keeps_blank_when_requested():
    runs = [{"run_id": "run-0001", "start": 0.0, "end": 10.0,
             "dominant_kind": "blank", "high_change": False}]
    segments = [{"id": "seg-000001", "start": 1.0, "end": 3.0, "text": "过渡"}]
    plan = plan_frames(segments, {"runs": runs}, include_blank=True)
    assert plan["unique_frame_count"] == 1


def test_plan_reports_orphan_runs():
    """没有被任何字幕引用的画面段可能含无口播的屏幕内容，必须报告。"""
    segments = [{"id": "seg-000001", "start": 0.0, "end": 3.0, "text": "只覆盖第一段"}]
    plan = plan_frames(segments, TIMELINE)
    assert plan["orphan_run_count"] == 2
    assert [r["run_id"] for r in plan["orphan_runs"]] == ["run-0002", "run-0003"]


def test_plan_no_orphans_when_all_covered():
    segments = [{"id": "seg-000001", "start": 0.0, "end": 20.0, "text": "全覆盖"}]
    plan = plan_frames(segments, TIMELINE)
    assert plan["orphan_run_count"] == 0


def test_plan_frame_ids_are_stable_and_descriptive():
    segments = [{"id": "seg-000001", "start": 6.0, "end": 8.0, "text": "A"}]
    plan = plan_frames(segments, TIMELINE)
    fid = plan["plan"][0]["frames"][0]["frame_id"]
    assert fid.startswith("frame-run-0002-")
    # 同一输入重复规划必须得到相同 ID
    again = plan_frames(segments, TIMELINE)
    assert again["plan"][0]["frames"][0]["frame_id"] == fid


def test_plan_max_frames_uniformly_thins_not_truncates():
    runs = [{"run_id": "run-%04d" % (i + 1), "start": float(i * 10), "end": float(i * 10 + 10),
             "dominant_kind": "content", "high_change": False} for i in range(10)]
    segments = [{"id": "seg-%06d" % (i + 1), "start": float(i * 10 + 1), "end": float(i * 10 + 2),
                 "text": "x"} for i in range(10)]
    plan = plan_frames(segments, {"runs": runs}, max_frames=4)
    assert plan["unique_frame_count"] == 4
    assert plan["truncated_by_max_frames"] == 6


def test_plan_handles_segment_without_any_run():
    segments = [{"id": "seg-000001", "start": 900.0, "end": 901.0, "text": "越界"}]
    plan = plan_frames(segments, TIMELINE)
    assert plan["plan"][0]["frames"] == []
    assert plan["unique_frame_count"] == 0
    assert plan["orphan_run_count"] == 3


# --------------------------------------------------------------------------
# frames 命令的 manifest 写入（回归）
# --------------------------------------------------------------------------

def _prepare_job(tmp_path: Path) -> Path:
    """构造一个只有 transcript 与旧版 manifest 的任务目录。

    manifest 故意用"没有 visual 键"的旧结构，复现历史 bug：
    早期实现直接 man["visual"].update(...)，遇到旧 manifest 直接 KeyError。
    """
    import json

    job = tmp_path / "job"
    job.mkdir(parents=True, exist_ok=True)
    (job / "transcript.raw.json").write_text(json.dumps({
        "job_id": "j", "source_type": "local_asr", "language": "zh",
        "segments": [
            {"id": "seg-000001", "start": 6.0, "end": 8.0, "text": "甲",
             "source_type": "local_asr", "language": "zh"},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    (job / "manifest.json").write_text(json.dumps({
        "manifest_version": 1,
        "job_id": "j",
        "subtitle": {"source_type": "local_asr", "probe_state": "not_probed",
                     "failure_kind": "not_probed"},
        "asr": {"used": True},
        "counts": {"segment_count": 1, "chunk_count": 1},
        "timing": {"claimed_duration": 20, "decoded_duration": 20, "last_cue_end": 8.0},
        "processing_status": "resource_processed",
        "verification_status": "unreviewed",
        "full_video_coverage": "supported",
        "coverage_basis": ["本地 ASR 覆盖完整音频"],
        "notes": [{"at": "x", "note": "警示：占位"}],
        "files": {},
    }, ensure_ascii=False), encoding="utf-8")
    return job


def test_frames_command_writes_manifest_without_visual_key(tmp_path, monkeypatch):
    """回归：旧 manifest 没有 visual 键时，frames 必须能写入而不是崩掉。"""
    import argparse
    import json
    import sys as _sys

    from video_ingest import __main__ as cli

    job = _prepare_job(tmp_path)
    video = tmp_path / "fake.mp4"
    video.write_bytes(b"\x00" * 32)

    timeline = {"runs": RUNS, "video": str(video), "fps": 30.0, "interval": 0.4,
                "scene_threshold": 6.0, "window": {"start": 0.0, "end": 20.0},
                "sample_count": 50, "samples": []}

    monkeypatch.setattr(_sys.modules["video_ingest.visual"], "scan_timeline",
                        lambda *a, **k: timeline)
    monkeypatch.setattr(_sys.modules["video_ingest.visual"], "extract_frames",
                        lambda path, frames, out, scale=1.0, on_progress=None: {
                            "written": len(frames), "missing": 0,
                            "frames": [{**f, "t_actual": f["sample_t"],
                                        "path": "%s.png" % f["frame_id"],
                                        "file": "x", "size_bytes": 10} for f in frames],
                        })

    args = argparse.Namespace(
        job_dir=str(job), video=str(video), interval=0.4, scene_threshold=6.0,
        start=0.0, end=None, scale=1.0, max_frames=0, include_blank=False,
    )
    rc = cli.cmd_frames(args)
    assert rc == cli.EXIT_OK

    man = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    assert man["visual"]["used"] is True
    assert man["visual"]["frames_written"] > 0
    assert man["visual"]["understanding_by"] == "agent"
    assert man["visual"]["sampling_policy"] == "intersection_midpoint"
    assert man["visual"]["index"] == "frames.index.json"
    assert (job / "frames.index.json").exists()
    assert (job / "visual-index.md").exists()

    # 写入后 manifest 必须自洽（视觉门禁也要通过）
    from video_ingest.manifest import validate_manifest
    problems = validate_manifest(man)
    assert not any("visual" in p for p in problems), problems


def test_frames_command_rejects_job_without_segments(tmp_path):
    import argparse
    import json

    from video_ingest import __main__ as cli

    job = tmp_path / "empty"
    job.mkdir()
    (job / "transcript.raw.json").write_text(
        json.dumps({"segments": []}), encoding="utf-8")

    args = argparse.Namespace(
        job_dir=str(job), video=None, interval=0.4, scene_threshold=6.0,
        start=0.0, end=None, scale=1.0, max_frames=0, include_blank=False,
    )
    assert cli.cmd_frames(args) == cli.EXIT_USAGE


# --------------------------------------------------------------------------
# manifest 视觉门禁
# --------------------------------------------------------------------------

def test_manifest_rejects_visual_without_index():
    from video_ingest.manifest import new_manifest, validate_manifest

    man = new_manifest(job_id="j", input_url=None, canonical_url=None,
                       platform="local", media_kind="local_media")
    man["visual"]["used"] = True
    man["visual"]["index"] = None
    problems = validate_manifest(man)
    assert any("缺少 index" in p for p in problems)


def test_manifest_rejects_script_claiming_visual_understanding():
    """脚本不得声称做了视觉理解——那是 Agent 的职责。"""
    from video_ingest.manifest import new_manifest, validate_manifest

    man = new_manifest(job_id="j", input_url=None, canonical_url=None,
                       platform="local", media_kind="local_media")
    man["visual"].update(used=True, index="frames.index.json",
                         frames_written=5, sampling_policy="intersection_midpoint",
                         understanding_by="script")
    problems = validate_manifest(man)
    assert any("understanding_by" in p for p in problems)


def test_manifest_rejects_visual_with_zero_frames():
    from video_ingest.manifest import new_manifest, validate_manifest

    man = new_manifest(job_id="j", input_url=None, canonical_url=None,
                       platform="local", media_kind="local_media")
    man["visual"].update(used=True, index="frames.index.json", frames_written=0,
                         sampling_policy="intersection_midpoint", understanding_by="agent")
    problems = validate_manifest(man)
    assert any("frames_written 为 0" in p for p in problems)
