"""字幕探测解析、URL 规范化与判定逻辑的单元测试。不依赖网络。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_ingest.acquire import (  # noqa: E402
    AcquireError,
    _parse_list_subs,
    normalize_url,
)
from video_ingest.__main__ import _combine_subtitle_verdict  # noqa: E402
from video_ingest.workflow import pick_subtitle_key  # noqa: E402


# --------------------------------------------------------------------------
# --list-subs 解析
# --------------------------------------------------------------------------

def test_parse_list_subs_regression_table_header_has_bracket_prefix():
    """回归：表头行形如 '[info] Available subtitles for ...'。

    早期实现先判断"以 [ 开头"再判断表头，导致永远进不了表格，
    danmaku_only 恒为 false。此用例锁死该顺序。
    """
    text = (
        "[BiliBili] Extracting subtitle info 41883336943\n"
        "[info] Available subtitles for BV1qNYC6eEMj:\n"
        "Language Formats\n"
        "danmaku  xml\n"
    )
    keys = _parse_list_subs(text)
    assert keys == ["danmaku"]


def test_parse_list_subs_multiple_languages():
    text = (
        "[info] Available subtitles for BV1gfYu6hEeg:\n"
        "Language Formats\n"
        "ai-zh  json\n"
        "zh-CN  srt\n"
        "danmaku  xml\n"
    )
    assert _parse_list_subs(text) == ["ai-zh", "zh-CN", "danmaku"]


def test_parse_list_subs_no_table_returns_empty():
    assert _parse_list_subs("[BiliBili] nothing here\n") == []


def test_parse_list_subs_stops_at_following_log_line():
    text = (
        "[info] Available subtitles for BVx:\n"
        "Language Formats\n"
        "zh-CN  srt\n"
        "[download] Destination: out.m4a\n"
    )
    assert _parse_list_subs(text) == ["zh-CN"]


def test_parse_list_subs_danmaku_is_recognised_as_not_a_subtitle():
    text = ("[info] Available subtitles for BVx:\n"
            "Language Formats\n"
            "danmaku  xml\n")
    keys = _parse_list_subs(text)
    real = [k for k in keys if k.lower() != "danmaku"]
    assert keys == ["danmaku"] and real == []


# --------------------------------------------------------------------------
# URL 规范化
# --------------------------------------------------------------------------

def test_normalize_strips_tracking_but_keeps_page():
    url = ("https://www.bilibili.com/video/BV1qNYC6eEMj/?trackid=web_pegasus&"
           "spm_id_from=333.1007&vd_source=abc&p=2")
    out = normalize_url(url)
    assert out["bvid"] == "BV1qNYC6eEMj"
    assert out["page"] == 2
    assert "trackid" not in out["canonical_url"]
    assert "vd_source" not in out["canonical_url"]
    assert "p=2" in out["canonical_url"]


def test_normalize_extracts_bvid_from_path():
    assert normalize_url("https://www.bilibili.com/video/BV1qNYC6eEMj")["bvid"] == "BV1qNYC6eEMj"


def test_normalize_handles_av_id():
    assert normalize_url("https://www.bilibili.com/video/av12345")["bvid"] == "av12345"


def test_normalize_rejects_non_http_scheme():
    with pytest.raises(AcquireError):
        normalize_url("file:///C:/secret.mp4")
    with pytest.raises(AcquireError):
        normalize_url("ftp://example.com/x")


def test_normalize_rejects_unsupported_host():
    with pytest.raises(AcquireError):
        normalize_url("https://evil.example.com/video/BV1qNYC6eEMj")


def test_normalize_rejects_localhost():
    with pytest.raises(AcquireError):
        normalize_url("http://localhost:8080/video/BV1qNYC6eEMj")
    with pytest.raises(AcquireError):
        normalize_url("http://127.0.0.1/video/BV1qNYC6eEMj")


def test_normalize_rejects_bad_page_value():
    with pytest.raises(AcquireError):
        normalize_url("https://www.bilibili.com/video/BV1qNYC6eEMj/?p=abc")


def test_normalize_empty_input_rejected():
    with pytest.raises(AcquireError):
        normalize_url("")


# --------------------------------------------------------------------------
# 三态判定
# --------------------------------------------------------------------------

def test_verdict_found_when_any_channel_has_keys():
    player = {"language_keys": [{"lan": "zh-CN", "is_ai": False, "subtitle_url": "//x"}]}
    out = _combine_subtitle_verdict(player, {"keys": []}, cookies_used=False)
    assert out["probe_state"] == "found"
    assert out["recommended_action"] == "download_subtitle"


def test_verdict_needs_auth_check_when_empty_without_cookies():
    """核心约束：无 cookie 时不能断言'没有字幕'。"""
    out = _combine_subtitle_verdict({"language_keys": []}, {"keys": []}, cookies_used=False)
    assert out["probe_state"] == "needs_auth_check"
    assert out["failure_kind"] == "needs_login"
    assert out["recommended_action"] == "provide_cookies_then_reprobe_or_use_asr"


def test_verdict_empty_when_confirmed_with_cookies():
    out = _combine_subtitle_verdict({"language_keys": []}, {"keys": []}, cookies_used=True)
    assert out["probe_state"] == "empty"
    assert out["failure_kind"] == "no_native_subtitle"
    assert out["recommended_action"] == "use_asr"


def test_verdict_ignores_danmaku_from_ytdlp():
    """只有弹幕时不得判为找到字幕。"""
    out = _combine_subtitle_verdict({"language_keys": []}, {"keys": ["danmaku"]},
                                    cookies_used=False)
    assert out["probe_state"] == "needs_auth_check"
    assert out["language_keys"] == []


def test_verdict_picks_up_ytdlp_only_keys():
    out = _combine_subtitle_verdict({"language_keys": []}, {"keys": ["ai-zh"]},
                                    cookies_used=True)
    assert out["probe_state"] == "found"
    assert out["language_keys"][0]["lan"] == "ai-zh"
    assert out["language_keys"][0]["via"] == "yt_dlp"


# --------------------------------------------------------------------------
# 字幕键选择优先级
# --------------------------------------------------------------------------

def test_pick_prefers_manual_native_over_auto():
    keys = [
        {"lan": "ai-zh", "is_ai": True, "subtitle_url": "//a"},
        {"lan": "zh-CN", "is_ai": False, "subtitle_url": "//b"},
    ]
    assert pick_subtitle_key(keys, "zh")["lan"] == "zh-CN"


def test_pick_falls_back_to_auto_when_no_manual():
    keys = [{"lan": "ai-zh", "is_ai": True, "subtitle_url": "//a"}]
    assert pick_subtitle_key(keys, "zh")["lan"] == "ai-zh"


def test_pick_ignores_translation_only():
    """只有机器翻译字幕时不应被当作原语字幕。"""
    keys = [{"lan": "en-US", "is_ai": False, "subtitle_url": "//e"}]
    assert pick_subtitle_key(keys, "zh") is None


def test_pick_returns_none_when_no_url():
    keys = [{"lan": "zh-CN", "is_ai": False, "subtitle_url": None}]
    assert pick_subtitle_key(keys, "zh") is None
