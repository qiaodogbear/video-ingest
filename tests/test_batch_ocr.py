"""批处理分页解析与 OCR 绑定的单元测试。不依赖网络，也不需要真实图片。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_ingest.batch import (  # noqa: E402
    MAX_PAGES,
    PageSpecError,
    parse_page_spec,
    should_reuse,
    summarize_batch,
)
from video_ingest.ocr import (  # noqa: E402
    DIVERGENCE_THRESHOLD,
    _candidate_terms,
    ngrams,
    normalize_text,
    ocr_available,
    overlap_ratio,
    search_ocr,
    to_markdown,
)


# --------------------------------------------------------------------------
# 分页表达式
# --------------------------------------------------------------------------

def test_single_page():
    assert parse_page_spec("3") == [3]


def test_page_list_sorted_and_deduped():
    assert parse_page_spec("7,3,3,1") == [1, 3, 7]


def test_page_range():
    assert parse_page_spec("1-5") == [1, 2, 3, 4, 5]


def test_mixed_ranges_and_values():
    assert parse_page_spec("1-3,7,10-12") == [1, 2, 3, 7, 10, 11, 12]


def test_whitespace_tolerated():
    assert parse_page_spec(" 1 - 3 , 5 ") == [1, 2, 3, 5]


def test_empty_spec_is_rejected_not_treated_as_all():
    """核心约束：不指定 ≠ 全部。空表达式必须报错，避免误下整个合集。"""
    for bad in ("", "   ", None, ","):
        with pytest.raises(PageSpecError):
            parse_page_spec(bad)


def test_rejects_zero_and_negative():
    for bad in ("0", "-1", "0-3"):
        with pytest.raises(PageSpecError):
            parse_page_spec(bad)


def test_rejects_reversed_range():
    with pytest.raises(PageSpecError):
        parse_page_spec("5-2")


def test_rejects_non_numeric():
    for bad in ("a", "1-a", "1--3", "1.5"):
        with pytest.raises(PageSpecError):
            parse_page_spec(bad)


def test_rejects_page_beyond_total():
    with pytest.raises(PageSpecError) as ei:
        parse_page_spec("1-10", total=3)
    assert "只有 3 个分P" in str(ei.value)


def test_accepts_page_within_total():
    assert parse_page_spec("1-3", total=3) == [1, 2, 3]


def test_rejects_too_many_pages():
    with pytest.raises(PageSpecError) as ei:
        parse_page_spec("1-%d" % (MAX_PAGES + 10))
    assert "最多" in str(ei.value)


# --------------------------------------------------------------------------
# 汇总与复用
# --------------------------------------------------------------------------

def test_summarize_batch_counts_each_status():
    results = [
        {"page": 1, "status": "ok"},
        {"page": 2, "status": "ok"},
        {"page": 3, "status": "reused"},
        {"page": 4, "status": "failed", "kind": "network_error", "message": "超时"},
    ]
    s = summarize_batch(results)
    assert s["requested"] == 4
    assert s["ok_count"] == 2
    assert s["reused_count"] == 1
    assert s["failed_count"] == 1
    assert s["all_ok"] is False
    assert s["failed_pages"][0]["page"] == 4


def test_summarize_batch_all_ok():
    s = summarize_batch([{"page": 1, "status": "ok"}, {"page": 2, "status": "reused"}])
    assert s["all_ok"] is True
    assert s["failed_pages"] == []


def test_summarize_batch_empty():
    s = summarize_batch([])
    assert s["requested"] == 0
    assert s["all_ok"] is True


def test_should_reuse_only_when_fully_processed():
    """只有确实处理完成才复用；partial/failed 必须重跑。

    否则会把上一次的失败静默当成成功。
    """
    assert should_reuse({"processing_status": "resource_processed"}) is True
    assert should_reuse({"processing_status": "partial"}) is False
    assert should_reuse({"processing_status": "failed"}) is False
    assert should_reuse({"processing_status": "blocked"}) is False
    assert should_reuse(None) is False
    assert should_reuse({}) is False


def test_should_reuse_force_overrides():
    assert should_reuse({"processing_status": "resource_processed"}, force=True) is False


# --------------------------------------------------------------------------
# OCR 文本工具
# --------------------------------------------------------------------------

def test_normalize_strips_punctuation_and_case():
    assert normalize_text("Hello，世界！ 123") == "hello世界123"
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_ngrams_short_text():
    assert ngrams("ab", n=4) == {"ab"}
    assert ngrams("", n=4) == set()


def test_ngrams_length():
    assert len(ngrams("abcdef", n=4)) == 3


def test_overlap_ratio_full_and_partial():
    assert overlap_ratio("十倍速学习", "我们用十倍速学习法") == 1.0
    assert overlap_ratio("完全无关的内容", "另一段文字") < 0.2
    assert overlap_ratio("", "任意") == 0.0


def test_overlap_tolerates_ocr_typos():
    """OCR 错字不应让覆盖率归零——这是用 n-gram 而不是精确匹配的原因。"""
    a = normalize_text("学习速度是复利的")
    b = normalize_text("学习速度是复利约")     # 末字识别错
    assert overlap_ratio(a, b) > 0.5


def test_candidate_terms_extracts_searchable_pieces():
    terms = _candidate_terms("复利 十倍速学习")
    assert "复利" in terms
    assert any("十倍速" in t for t in terms)
    assert all(len(t) >= 2 for t in terms)


def test_candidate_terms_ignores_punctuation_only():
    assert _candidate_terms("，。！") == set()


def test_search_ocr_finds_and_sorts_by_time():
    payload = {
        "frames": [
            {"frame": "b.png", "ocr_text": "学习速度是复利的", "segment_ids": ["s2"],
             "segment_start": 74.0},
            {"frame": "a.png", "ocr_text": "复利账 干活省的是时间", "segment_ids": ["s1"],
             "segment_start": 12.0},
            {"frame": "c.png", "ocr_text": "无关内容", "segment_ids": [], "segment_start": None},
        ]
    }
    hits = search_ocr(payload, "复利")
    assert [h["frame"] for h in hits] == ["a.png", "b.png"]
    assert hits[0]["segment_start"] == 12.0
    assert search_ocr(payload, "复利")[0]["matched"] == "复利"
    assert search_ocr(payload, "不存在的词") == []


def test_search_ocr_ignores_punctuation_in_query():
    payload = {"frames": [{"frame": "a.png", "ocr_text": "复利账",
                           "segment_ids": [], "segment_start": 1.0}]}
    assert len(search_ocr(payload, "复利，")) == 1


def test_search_respects_limit():
    payload = {"frames": [{"frame": "%d.png" % i, "ocr_text": "复利",
                           "segment_ids": [], "segment_start": float(i)}
                          for i in range(30)]}
    assert len(search_ocr(payload, "复利", limit=5)) == 5


def test_markdown_flags_divergent_frames():
    """口播与画面无关的帧必须单独成节——那是纯转录必然漏掉的信息。"""
    payload = {
        "engine": "rapidocr-onnxruntime", "frames_total": 2, "seconds_total": 4.0,
        "seconds_per_frame": 2.0, "divergent_count": 1,
        "divergence_threshold": DIVERGENCE_THRESHOLD,
        "frames": [
            {"frame": "a.png", "segment_ids": ["s1"], "segment_start": 12.0,
             "segment_text": "口播甲", "ocr_text": "画面文字甲", "divergent": False,
             "subtitle_coverage": 0.9, "error": None},
            {"frame": "b.png", "segment_ids": ["s2"], "segment_start": 42.0,
             "segment_text": "我们从最反直觉的一笔账讲起",
             "ocr_text": "差距=指数级 干活效率·线性 学习速度：复利", "divergent": True,
             "subtitle_coverage": 0.0, "error": None},
        ],
    }
    md = to_markdown(payload)
    assert "口播与画面内容无关的片段" in md
    assert "差距=指数级" in md
    assert "我们从最反直觉的一笔账讲起" in md
    # 分歧帧应出现在分歧节里
    assert md.index("b.png") < md.index("逐帧画面文字")


def test_markdown_reports_ocr_error():
    payload = {
        "engine": "x", "frames_total": 1, "seconds_total": 0.0, "seconds_per_frame": 0.0,
        "divergent_count": 0, "divergence_threshold": DIVERGENCE_THRESHOLD,
        "frames": [{"frame": "bad.png", "segment_ids": [], "segment_start": None,
                    "segment_text": "", "ocr_text": "", "divergent": False,
                    "subtitle_coverage": None, "error": "解码失败"}],
    }
    assert "解码失败" in to_markdown(payload)


def test_ocr_available_returns_reason_when_missing(monkeypatch):
    """可选依赖缺失必须返回可读原因，而不是抛异常。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "rapidocr_onnxruntime":
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ok, reason = ocr_available()
    assert ok is False
    assert reason and "rapidocr" in reason


def test_run_ocr_requires_frames_index(tmp_path):
    from video_ingest.ocr import run_ocr

    with pytest.raises(FileNotFoundError) as ei:
        run_ocr(tmp_path)
    assert "frames" in str(ei.value)


# --------------------------------------------------------------------------
# manifest 的 OCR 门禁
# --------------------------------------------------------------------------

def test_manifest_rejects_ocr_without_visual():
    """OCR 依赖已提取的帧，不能凭空声称用了 OCR。"""
    from video_ingest.manifest import new_manifest, validate_manifest

    man = new_manifest(job_id="j", input_url=None, canonical_url=None,
                       platform="local", media_kind="local_media")
    man["ocr"].update(used=True, index="ocr.index.json", frames_read=10,
                      understanding_by="agent")
    problems = validate_manifest(man)
    assert any("visual.used 为 false" in p for p in problems)


def test_manifest_rejects_ocr_claiming_understanding():
    from video_ingest.manifest import new_manifest, validate_manifest

    man = new_manifest(job_id="j", input_url=None, canonical_url=None,
                       platform="local", media_kind="local_media")
    man["visual"].update(used=True, index="frames.index.json", frames_written=5,
                         sampling_policy="intersection_midpoint", understanding_by="agent")
    man["ocr"].update(used=True, index="ocr.index.json", frames_read=5,
                      understanding_by="script")
    problems = validate_manifest(man)
    assert any("ocr.understanding_by" in p for p in problems)


# --------------------------------------------------------------------------
# 语言不匹配（比"识别不准"更危险的失败模式）
# --------------------------------------------------------------------------

def _asr_out(lang, prob=0.94):
    return {"detected_language": lang, "language_probability": prob}


def test_language_mismatch_flags_and_escalates_verification():
    """强制 zh 解码英文音频会流畅地编造中文——必须被记为 issue。

    实测：传入 language=zh 时 info.language 回显 "zh" 且 prob=1.000，
    因此必须用**单独一次检测**的真实结果来比对，不能用 transcribe 的回显值。
    """
    from video_ingest.manifest import new_manifest, validate_manifest
    from video_ingest.workflow import _check_language_match

    man = new_manifest(job_id="j", input_url=None, canonical_url=None,
                       platform="bilibili", media_kind="bilibili_video")
    _check_language_match(man, requested="zh", asr_out=_asr_out("en"))

    assert man["asr"]["language_mismatch"] is True
    assert man["asr"]["detected_language"] == "en"
    assert man["verification_status"] == "issues_found"
    assert any("语言" in n["note"] for n in man["notes"])
    assert not any("language_mismatch" in p for p in validate_manifest(man))


def test_language_match_does_not_flag():
    from video_ingest.manifest import new_manifest
    from video_ingest.workflow import _check_language_match

    man = new_manifest(job_id="j", input_url=None, canonical_url=None,
                       platform="bilibili", media_kind="bilibili_video")
    _check_language_match(man, requested="zh", asr_out=_asr_out("zh"))
    assert man["asr"]["language_mismatch"] is False
    assert man["verification_status"] == "unreviewed"
    assert not man["notes"]


def test_language_region_variants_are_not_mismatch():
    """zh 与 zh-CN、en 与 en-US 属同一语言，不应报不匹配。"""
    from video_ingest.manifest import new_manifest
    from video_ingest.workflow import _check_language_match

    for req, det in (("zh", "zh-CN"), ("en", "en-US"), ("zh-CN", "zh")):
        man = new_manifest(job_id="j", input_url=None, canonical_url=None,
                           platform="bilibili", media_kind="bilibili_video")
        _check_language_match(man, requested=req, asr_out=_asr_out(det))
        assert man["asr"]["language_mismatch"] is False, (req, det)


def test_language_check_skipped_for_auto():
    """auto 模式下检测值就是实际使用值，不存在"指定错"的问题。"""
    from video_ingest.manifest import new_manifest
    from video_ingest.workflow import _check_language_match

    for req in ("auto", None):
        man = new_manifest(job_id="j", input_url=None, canonical_url=None,
                           platform="bilibili", media_kind="bilibili_video")
        _check_language_match(man, requested=req, asr_out=_asr_out("en"))
        assert man["asr"]["language_mismatch"] is False


def test_language_check_handles_missing_detection():
    from video_ingest.manifest import new_manifest
    from video_ingest.workflow import _check_language_match

    man = new_manifest(job_id="j", input_url=None, canonical_url=None,
                       platform="bilibili", media_kind="bilibili_video")
    _check_language_match(man, requested="zh",
                          asr_out={"detected_language": None, "language_probability": None})
    assert man["asr"]["language_mismatch"] is False


def test_manifest_rejects_mismatch_left_as_unreviewed():
    """回归：不匹配却没升级核查状态，等于把不可信内容当正常结果。"""
    from video_ingest.manifest import new_manifest, validate_manifest

    man = new_manifest(job_id="j", input_url=None, canonical_url=None,
                       platform="bilibili", media_kind="bilibili_video")
    man["asr"].update(used=True, language_mismatch=True)
    man["subtitle"]["source_type"] = "local_asr"
    man["counts"]["segment_count"] = 10
    man["verification_status"] = "unreviewed"
    problems = validate_manifest(man)
    assert any("language_mismatch" in p for p in problems)
