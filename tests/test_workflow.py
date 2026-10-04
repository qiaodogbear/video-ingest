"""分块覆盖、manifest 一致性与脱敏的单元测试。不依赖网络。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_ingest.chunks import build_chunks, validate_coverage, write_chunks  # noqa: E402
from video_ingest.manifest import (  # noqa: E402
    add_coverage_basis,
    add_note,
    new_manifest,
    safe_name,
    validate_manifest,
)
from video_ingest.redact import redact, redact_url  # noqa: E402


def make_segments(n: int, chars: int = 10):
    return [
        {
            "id": "seg-%06d" % (i + 1),
            "start": float(i),
            "end": float(i) + 0.9,
            "text": "x" * chars,
            "source_type": "local_asr",
            "language": "zh",
        }
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# 分块
# --------------------------------------------------------------------------

def test_all_segments_covered_exactly_once():
    segs = make_segments(50)
    index = build_chunks(segs, chars_per_chunk=100, overlap_segments=1)
    cov = index["coverage"]
    assert cov["complete"] is True
    assert cov["missing_segment_ids"] == []
    assert cov["duplicated_in_main_coverage"] == []
    assert len(cov["covered_segment_ids"]) == len(segs)


def test_empty_input_produces_no_chunks_and_is_complete():
    index = build_chunks([])
    assert index["chunk_count"] == 0
    assert index["coverage"]["complete"] is True


def test_no_segment_is_split_across_chunks():
    segs = make_segments(20, chars=30)
    index = build_chunks(segs, chars_per_chunk=100, overlap_segments=0)
    seen = [sid for ch in index["chunks"] for sid in ch["segment_ids"]]
    assert len(seen) == len(set(seen)) == 20


def test_overlap_is_recorded_but_does_not_replace_main_coverage():
    segs = make_segments(30)
    index = build_chunks(segs, chars_per_chunk=80, overlap_segments=2)
    cov = index["coverage"]
    assert cov["complete"] is True
    assert cov["overlap_segment_ids"]          # 有重叠
    assert cov["duplicated_in_main_coverage"] == []   # 主覆盖不重复


def test_large_single_segment_still_placed():
    segs = make_segments(1, chars=10000)
    index = build_chunks(segs, chars_per_chunk=100)
    assert index["chunk_count"] == 1
    assert index["coverage"]["complete"] is True


def test_validate_coverage_detects_missing():
    segs = make_segments(5)
    index = build_chunks(segs, chars_per_chunk=10, overlap_segments=0)
    index["coverage"]["missing_segment_ids"] = ["seg-000003"]
    problems = validate_coverage(index)
    assert any("未进入任何分块" in p for p in problems)


def test_validate_coverage_detects_chunk_count_mismatch():
    segs = make_segments(5)
    index = build_chunks(segs, chars_per_chunk=10)
    index["chunk_count"] = 999
    problems = validate_coverage(index)
    assert any("chunk_count" in p for p in problems)


def test_write_chunks_creates_index_and_markdown(tmp_path: Path):
    segs = make_segments(12)
    index = build_chunks(segs, chars_per_chunk=40, overlap_segments=1)
    index_path = write_chunks(tmp_path, index)
    assert index_path.exists()
    data = json.loads(index_path.read_text(encoding="utf-8"))
    assert data["coverage"]["complete"] is True
    # index.json 不应内嵌全部片段正文
    assert "segments" not in data["chunks"][0]
    md_files = sorted((tmp_path / "chunks").glob("chunk-*.md"))
    assert len(md_files) == index["chunk_count"]
    assert "seg-000001" in md_files[0].read_text(encoding="utf-8")


def test_chars_per_chunk_is_declared_not_tokens():
    index = build_chunks(make_segments(3))
    assert index["chars_per_chunk_is_tokens"] is False


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        build_chunks(make_segments(3), chars_per_chunk=0)
    with pytest.raises(ValueError):
        build_chunks(make_segments(3), overlap_segments=-1)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def make_manifest(**kw):
    man = new_manifest(
        job_id="bilibili_BV1_p1", input_url="https://x/y",
        canonical_url="https://x/y", platform="bilibili", media_kind="bilibili_video",
    )
    man.update(kw)
    return man


def test_new_manifest_is_valid_shape():
    man = make_manifest()
    problems = validate_manifest(man)
    # 默认状态不应触发非法枚举报错
    assert not any("非法" in p for p in problems)


def test_manifest_rejects_invalid_statuses():
    man = make_manifest(processing_status="done", verification_status="fine",
                        full_video_coverage="yes")
    problems = validate_manifest(man)
    assert any("processing_status 非法" in p for p in problems)
    assert any("verification_status 非法" in p for p in problems)
    assert any("full_video_coverage 非法" in p for p in problems)


def test_manifest_flags_processed_without_segments():
    man = make_manifest(processing_status="resource_processed")
    problems = validate_manifest(man)
    assert any("segment_count 为 0" in p for p in problems)


def test_manifest_requires_basis_for_supported_coverage():
    man = make_manifest(full_video_coverage="supported")
    problems = validate_manifest(man)
    assert any("coverage_basis 为空" in p for p in problems)


def test_manifest_requires_warning_note_when_coverage_not_supported():
    man = make_manifest(full_video_coverage="partial", processing_status="resource_processed")
    man["counts"]["segment_count"] = 5
    problems = validate_manifest(man)
    assert any("警示" in p for p in problems)
    add_note(man, "警示：full_video_coverage=partial，输出必须带限制")
    assert not any("警示" in p for p in validate_manifest(man))


def test_manifest_detects_contradictory_source():
    man = make_manifest()
    man["asr"]["used"] = True
    man["subtitle"]["source_type"] = "platform_manual"
    problems = validate_manifest(man)
    assert any("来源标记矛盾" in p for p in problems)


def test_manifest_flags_decoded_longer_than_claimed():
    man = make_manifest()
    man["timing"]["claimed_duration"] = 100
    man["timing"]["decoded_duration"] = 130
    problems = validate_manifest(man)
    assert any("明显超过" in p for p in problems)


def test_manifest_rejects_unknown_source_type():
    man = make_manifest()
    man["subtitle"]["source_type"] = "guessed_manual"
    problems = validate_manifest(man)
    assert any("source_type 非法" in p for p in problems)


def test_coverage_basis_appends():
    man = make_manifest()
    add_coverage_basis(man, "依据一")
    add_coverage_basis(man, "依据二")
    assert man["coverage_basis"] == ["依据一", "依据二"]


# --------------------------------------------------------------------------
# 文件名与脱敏
# --------------------------------------------------------------------------

def test_safe_name_handles_windows_and_chinese():
    assert "/" not in safe_name("a/b")
    assert "\\" not in safe_name("a\\b")
    assert ":" not in safe_name("标题：副标题")
    assert safe_name("中文 标题 带空格") == "中文_标题_带空格"
    assert safe_name("") == "untitled"
    assert safe_name("...") == "untitled"


def test_redact_masks_credentials_in_text():
    out = redact("SESSDATA=abc123; bili_jct=def456 Authorization: Bearer xyz")
    assert "abc123" not in out
    assert "def456" not in out
    assert "***" in out


def test_redact_url_masks_signed_query_but_keeps_identity():
    url = ("https://upos-x.bilivideo.com/upgcxcode/43/69/41883336943/41883336943-1-30280.m4s"
           "?upsig=deadbeef&deadline=1791052498&oi=1908977280&buvid=ABC")
    out = redact_url(url)
    assert "deadbeef" not in out
    assert "1791052498" not in out
    assert "ABC" not in out
    assert "41883336943" in out          # 路径保留，仍可定位
    assert "upos-x.bilivideo.com" in out


def test_redact_url_masks_cookie_style_params():
    out = redact_url("https://x/y?SESSDATA=secret&p=2")
    assert "secret" not in out
    assert "p=2" in out
