"""转录解析、标准化与校正的单元测试。不依赖网络。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_ingest.transcript import (  # noqa: E402
    apply_corrections,
    normalize_cues,
    parse_bilibili_subtitle_json,
    parse_srt,
    parse_subtitle_file,
    parse_timestamp,
    parse_vtt,
    to_plain_text,
    to_srt,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_timestamp_formats():
    assert parse_timestamp("00:00:12,500") == 12.5
    assert parse_timestamp("00:01:02.250") == 62.25
    assert parse_timestamp("01:02.500") == 62.5
    assert parse_timestamp("01:00:00,000") == 3600.0


def test_parse_srt_basic():
    text = (FIXTURES / "sample.srt").read_text(encoding="utf-8")
    cues = parse_srt(text)
    assert len(cues) == 3
    assert cues[0]["start"] == 0.5
    assert "第一段" in cues[0]["text"]
    assert cues[2]["end"] == 12.0


def test_parse_vtt_with_rolling_dedupe():
    text = (FIXTURES / "sample.vtt").read_text(encoding="utf-8")
    cues = parse_vtt(text)
    # 滚动重复应被合并，且内联标签被清除
    assert len(cues) == 3
    assert cues[0]["text"] == "大家好"
    assert all("<" not in c["text"] for c in cues)


def test_parse_bilibili_json_body_shape():
    payload = json.loads((FIXTURES / "bilibili.subtitle.json").read_text(encoding="utf-8"))
    cues = parse_bilibili_subtitle_json(payload)
    assert len(cues) == 2
    assert cues[0]["start"] == 1.2
    assert cues[0]["text"] == "这是一条字幕"


def test_parse_bilibili_json_nested_data_shape():
    payload = {"data": {"body": [{"from": 0, "to": 1, "content": "嵌套"}]}}
    cues = parse_bilibili_subtitle_json(payload)
    assert cues[0]["text"] == "嵌套"


def test_parse_bilibili_json_unknown_shape_raises():
    """来源无法识别时必须报错，不得猜格式。"""
    with pytest.raises(ValueError):
        parse_bilibili_subtitle_json({"unexpected": True})


def test_xml_is_rejected_as_subtitle():
    """弹幕 XML 不是字幕，必须拒绝。"""
    p = FIXTURES / "danmaku.xml"
    with pytest.raises(ValueError):
        parse_subtitle_file(p)


def test_parse_srt_empty_returns_empty_list():
    """低层解析器对空内容返回空列表，由调用方负责判定为失败。"""
    assert parse_srt("") == []
    assert parse_srt("   \n  ") == []


def test_empty_subtitle_file_raises(tmp_path: Path):
    """空字幕文件必须报错，不能当成'成功但没有内容'。"""
    empty = tmp_path / "empty.srt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_subtitle_file(empty)


def test_normalize_assigns_stable_ids_and_sorts():
    cues = [
        {"start": 5.0, "end": 6.0, "text": "后"},
        {"start": 1.0, "end": 2.0, "text": "先"},
    ]
    segs, issues = normalize_cues(cues, source_type="local_asr", language="zh")
    assert [s["text"] for s in segs] == ["先", "后"]
    assert [s["id"] for s in segs] == ["seg-000001", "seg-000002"]
    assert all(s["source_type"] == "local_asr" for s in segs)


def test_normalize_reports_negative_and_reversed_without_deleting():
    cues = [
        {"start": -1.0, "end": 1.0, "text": "负起点"},
        {"start": 5.0, "end": 3.0, "text": "倒挂"},
    ]
    segs, issues = normalize_cues(cues, source_type="platform_auto", language="zh")
    assert len(segs) == 2  # 不静默删数据
    assert segs[0]["start"] == 0.0
    assert any("负" in i for i in issues)
    assert any("end < start" in i for i in issues)


def test_normalize_keeps_legitimate_overlap_and_reports_it():
    cues = [
        {"start": 0.0, "end": 5.0, "text": "A"},
        {"start": 3.0, "end": 8.0, "text": "B"},
    ]
    segs, issues = normalize_cues(cues, source_type="platform_manual", language="zh")
    assert len(segs) == 2
    assert any("重叠" in i for i in issues)


def test_normalize_flags_start_beyond_duration():
    cues = [{"start": 100.0, "end": 101.0, "text": "越界"}]
    _, issues = normalize_cues(cues, source_type="local_asr", language="zh", duration=10.0)
    assert any("超出解码时长" in i for i in issues)


def test_corrections_do_not_modify_original_segments():
    segs, _ = normalize_cues(
        [{"start": 0, "end": 1, "text": "费慢学习和素查表"}],
        source_type="local_asr", language="zh",
    )
    corrected, records = apply_corrections(segs, {"费慢": "费曼", "素查表": "速查表"})
    assert segs[0]["text"] == "费慢学习和素查表"      # 原文未被覆盖
    assert corrected[0]["text"] == "费曼学习和速查表"
    assert len(records) == 2
    assert {r["wrong"] for r in records} == {"费慢", "素查表"}


def test_corrections_prefer_longest_key_first():
    segs, _ = normalize_cues(
        [{"start": 0, "end": 1, "text": "速查表情"}],
        source_type="local_asr", language="zh",
    )
    corrected, _ = apply_corrections(segs, {"速查表": "X", "速查表情": "Y"})
    assert corrected[0]["text"] == "Y"


def test_exports_round_trip():
    segs, _ = normalize_cues(
        [{"start": 0.5, "end": 2.0, "text": "甲"}, {"start": 2.0, "end": 4.0, "text": "乙"}],
        source_type="local_asr", language="zh",
    )
    assert to_plain_text(segs) == "甲乙"
    srt = to_srt(segs)
    assert "00:00:00,500 --> 00:00:02,000" in srt
    # 导出的 SRT 应能被自己重新解析
    assert len(parse_srt(srt)) == 2
