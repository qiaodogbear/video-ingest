"""端到端离线测试：从字幕 fixture 或本地音频 fixture 走完整流水线。

这些用例不访问网络，用于验证"字幕路线"与"本地媒体路线"和 ASR 路线
具有同等的覆盖保证。fixtures 未覆盖的路径（实网 cookie）见 SOP 第 7 节。
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_ingest.chunks import build_chunks, validate_coverage, write_chunks  # noqa: E402
from video_ingest.manifest import (  # noqa: E402
    add_coverage_basis,
    add_note,
    new_manifest,
    save_manifest,
    validate_manifest,
)
from video_ingest.transcript import (  # noqa: E402
    apply_corrections,
    normalize_cues,
    parse_subtitle_file,
    to_srt,
    to_txt,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_subtitle_route_produces_full_coverage_guarantee(tmp_path: Path):
    """字幕路线必须和 ASR 路线一样，保证每个片段都进入分块。"""
    cues = parse_subtitle_file(FIXTURES / "sample.srt")
    segments, issues = normalize_cues(
        cues, source_type="platform_manual", language="zh-CN", duration=12.0
    )
    assert len(segments) == 3
    assert not issues

    # 写原始层
    (tmp_path / "transcript.txt").write_text(to_txt(segments), encoding="utf-8")
    (tmp_path / "transcript.srt").write_text(to_srt(segments), encoding="utf-8")

    index = build_chunks(segments, chars_per_chunk=10, overlap_segments=1)
    assert index["coverage"]["complete"] is True
    assert index["coverage"]["missing_segment_ids"] == []
    assert validate_coverage(index) == []
    write_chunks(tmp_path, index)
    assert (tmp_path / "chunks" / "index.json").exists()


def test_subtitle_route_correction_layering(tmp_path: Path):
    """平台字幕同样适用"原文不被覆盖"的规则。"""
    cues = parse_subtitle_file(FIXTURES / "sample.srt")
    segments, _ = normalize_cues(cues, source_type="platform_manual", language="zh-CN")
    original = segments[0]["text"]

    corrected, records = apply_corrections(segments, {"第一段": "第1段"})
    assert segments[0]["text"] == original          # 原文未变
    assert corrected[0]["text"] != original
    assert records[0]["wrong"] == "第一段"


def test_subtitle_manifest_coverage_must_be_uncertain(tmp_path: Path):
    """只拿到字幕文件时，覆盖率不得标 supported。

    这是 bootstrap §10 的核心约束：下载到字幕文件不证明覆盖全部讲话。
    """
    man = new_manifest(
        job_id="bilibili_BVtest_p1", input_url="https://x", canonical_url="https://x",
        platform="bilibili", media_kind="bilibili_video",
    )
    man["subtitle"]["source_type"] = "platform_manual"
    man["subtitle"]["probe_state"] = "found"
    man["subtitle"]["failure_kind"] = "not_probed"
    man["counts"]["segment_count"] = 3
    man["timing"].update(claimed_duration=100, decoded_duration=None, last_cue_end=12.0)
    man["processing_status"] = "resource_processed"
    man["full_video_coverage"] = "uncertain"
    add_coverage_basis(man, "使用平台字幕，仅证明下载了该字幕文件")
    add_note(man, "警示：full_video_coverage=uncertain，输出必须带限制")

    problems = validate_manifest(man)
    assert problems == [], "字幕路线的 manifest 应自洽: %s" % problems

    save_manifest(tmp_path, man)
    assert (tmp_path / "manifest.json").exists()

    # 反过来：若错误地标成 supported 且无依据，必须被检出
    man["full_video_coverage"] = "supported"
    man["coverage_basis"] = []
    problems = validate_manifest(man)
    assert any("coverage_basis 为空" in p for p in problems)


def test_blocked_status_is_distinguishable_from_no_subtitle():
    """鉴权失败必须能和'确实没有字幕'区分开。"""
    man = new_manifest(
        job_id="j", input_url="https://x", canonical_url="https://x",
        platform="bilibili", media_kind="bilibili_video",
    )
    man["processing_status"] = "blocked"
    man["subtitle"]["failure_kind"] = "needs_login"
    man["subtitle"]["source_type"] = "unknown"
    problems = validate_manifest(man)
    assert not any("非法" in p for p in problems)

    # 分类不能是空列表式的一团
    man["subtitle"]["failure_kind"] = "something_vague"
    assert any("failure_kind 非法" in p for p in validate_manifest(man))


def test_unknown_source_is_not_guessed_as_manual():
    """来源未识别时必须保持 unknown，不得默认成人工字幕。"""
    man = new_manifest(
        job_id="j", input_url="https://x", canonical_url="https://x",
        platform="bilibili", media_kind="bilibili_video",
    )
    assert man["subtitle"]["source_type"] == "unknown"
    assert not any("非法" in p for p in validate_manifest(man))


def test_zero_segments_cannot_be_reported_as_success():
    """回归：纯静音/纯音乐音频 ASR 会产出 0 片段。

    此时"流程跑完了"不等于"拿到了内容"。若标成 resource_processed +
    supported，就会给出一份看起来成功、实际没有任何内容的结论。
    """
    man = new_manifest(
        job_id="local_tone", input_url=None, canonical_url=None,
        platform="local", media_kind="local_media",
    )
    man["asr"]["used"] = True
    man["subtitle"]["source_type"] = "local_asr"
    man["subtitle"]["probe_state"] = "not_probed"
    man["subtitle"]["failure_kind"] = "not_probed"
    man["timing"]["decoded_duration"] = 3.0

    # 错误的记法：声称成功且覆盖完整
    man["processing_status"] = "resource_processed"
    man["full_video_coverage"] = "supported"
    add_coverage_basis(man, "本地媒体")
    problems = validate_manifest(man)
    assert any("segment_count 为 0" in p for p in problems), \
        "0 片段却声称处理完成，必须被检出"

    # 正确的记法：如实记为 failed，覆盖率不作 supported 主张
    man["processing_status"] = "failed"
    man["full_video_coverage"] = "uncertain"
    assert not any("segment_count 为 0" in p for p in validate_manifest(man))


# --------------------------------------------------------------------------
# doctor 的环境报告
# --------------------------------------------------------------------------

def _run_doctor(check_gpu: bool = False):
    import argparse
    import io
    import json as _json

    from video_ingest import __main__ as cli

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        cli.cmd_doctor(argparse.Namespace(check_gpu=check_gpu))
    finally:
        sys.stdout = old
    return _json.loads(buf.getvalue())


def test_doctor_reports_test_runner():
    """README 让用户跑 pytest 验证安装，doctor 必须报告它是否可用。"""
    out = _run_doctor()
    assert "test_runner" in out["checks"]
    assert isinstance(out["checks"]["test_runner"]["ok"], bool)


def test_doctor_warns_when_pytest_missing(monkeypatch):
    """缺 pytest 时给出告警与修复提示，但不把它当成取材失败。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pytest":
            raise ImportError("simulated missing pytest")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = _run_doctor()
    assert out["checks"]["test_runner"]["ok"] is False
    assert any("pytest" in w for w in out.get("warnings", []))
    # 缺测试运行器不等于取材依赖不可用。
    # 注意：不要在这里断言某个依赖"已安装"——测试套件本身不依赖
    # yt-dlp / faster-whisper / numpy，CI 与最小环境都应在缺失时也能跑通。
    # 只校验依赖检查自身结构完好、且没有被 pytest 的缺失牵连。
    deps = out["checks"]["dependencies"]
    assert {"yt_dlp", "faster_whisper", "av", "numpy"} <= set(deps)
    assert all("ok" in v for v in deps.values())
