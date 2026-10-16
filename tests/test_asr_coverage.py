"""离线回归：只验证证据门禁与参数传递，不验证真实媒体识别准确性。"""

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from video_ingest import asr, cli, workflow
from video_ingest.manifest import new_manifest, validate_manifest


def manifest():
    man = new_manifest(
        job_id="test", input_url=None, canonical_url=None, platform="local", media_kind="local_media"
    )
    man["asr"].update(used=True, vad_filter=True)
    man["subtitle"]["source_type"] = "local_asr"
    return man


@pytest.mark.parametrize("vad", [True, False])
@pytest.mark.parametrize(
    "intervals,position,duration",
    [
        ([(0, 5), (95, 100)], "internal", 90),
        ([(0, 10)], "trailing", 90),
        ([(90, 100)], "leading", 90),
        ([], "entire", 100),
    ],
)
def test_gaps_are_uncertain_not_confirmed_missing_speech(vad, intervals, position, duration):
    man = manifest()
    man["asr"]["vad_filter"] = vad
    man["timing"].update(decoded_duration=100, claimed_duration=100, last_cue_end=100)
    segments = [{"start": a, "end": b} for a, b in intervals]
    workflow._assess_coverage(man, segments, {}, {}, {"mode": "asr"})
    assert man["full_video_coverage"] == "uncertain"
    evidence = man["coverage_evidence"]
    assert evidence["vad_filter"] is vad
    assert evidence["gaps"] == [
        {
            "start": 10 if position == "trailing" else 5 if position == "internal" else 0,
            "end": 95 if position == "internal" else 90 if position == "leading" else 100,
            "duration": duration,
            "position": position,
        }
    ]
    assert evidence["speech_completeness_verified"] is False
    assert any("原因未核实" in n["note"] for n in man["notes"])
    assert man["verification_status"] == "unreviewed"


@pytest.mark.parametrize("decoded,claimed", [(100, 100), (100, 200), (None, 100), (0, None)])
def test_dense_timestamps_or_missing_duration_do_not_prove_completeness(decoded, claimed):
    man = manifest()
    man["timing"].update(decoded_duration=decoded, claimed_duration=claimed)
    # 重叠与乱序不能重复累计，不能依赖 timing.last_cue_end。
    workflow._assess_coverage(
        man, [{"start": 40, "end": 100}, {"start": 0, "end": 60}], {}, {}, {"mode": "asr"}
    )
    assert man["timing"]["speech_duration"] == 100
    assert man["timing"]["transcribed_end"] == 100
    assert man["full_video_coverage"] == "uncertain"


@pytest.mark.parametrize("vad", [True, False])
@pytest.mark.parametrize("with_segments", [False, True])
@pytest.mark.parametrize("chunk_problem", [False, True])
def test_local_route_persists_gap_evidence(monkeypatch, tmp_path, vad, with_segments, chunk_problem):
    if chunk_problem:
        monkeypatch.setattr(workflow, "validate_coverage", lambda _: ["missing segment"])
    source = tmp_path / "silence.wav"
    # 合成真实零值 PCM；不运行模型，mock 输出不能证明它识别正确。
    import wave

    with wave.open(str(source), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(b"\0\0" * 16000)
    seen = []

    def transcribe(path, **kw):
        seen.append(kw["vad_filter"])
        return {
            "cues": [{"start": 0, "end": 0.1, "text": "测试"}] if with_segments else [],
            "decoded_duration": 1,
            "detected_language": "zh",
            "language_probability": 1,
            "elapsed_seconds": 0,
            "params": {
                "model": "mock",
                "device": "cpu",
                "compute_type": "int8",
                "language": "zh",
                "beam_size": 5,
                "vad_filter": kw["vad_filter"],
                "vad_parameters": None,
            },
        }

    monkeypatch.setattr(workflow, "transcribe_audio", transcribe)
    result = workflow.run_ingest(url=None, media=str(source), output_dir=tmp_path / "out", vad_filter=vad)
    man = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert seen == [vad]
    for key in ("full_video_coverage", "verification_status", "coverage_evidence", "coverage_basis", "notes"):
        assert result[key] == man[key]
    assert result["processing_status"] == (("partial" if chunk_problem else "resource_processed") if with_segments else "failed")
    assert man["full_video_coverage"] == "uncertain"
    assert man["coverage_evidence"]["gaps"][0]["position"] == ("trailing" if with_segments else "entire")
    assert man["asr"]["vad_filter"] is vad
    assert validate_manifest(man) == []


@pytest.mark.parametrize("command", ["ingest", "batch"])
@pytest.mark.parametrize("no_vad", [False, True])
def test_cli_passes_vad(monkeypatch, tmp_path, command, no_vad):
    calls = []

    def run(**kw):
        calls.append(kw)
        return {
            "ok": True,
            "processing_status": "resource_processed",
            "segment_count": 2,
            "chunk_count": 1,
            "job_dir": str(tmp_path),
            "full_video_coverage": "uncertain",
        }

    monkeypatch.setattr(cli, "run_ingest", run)
    monkeypatch.setattr(cli, "build_opener", lambda _: None)
    monkeypatch.setattr(
        cli, "fetch_video_info", lambda *a: {"bvid": "BV1xx411c7mD", "title": "test", "pages": [{"page": 1}]}
    )
    args = [command, "--url", "https://www.bilibili.com/video/BV1xx411c7mD", "--output-dir", str(tmp_path)]
    if command == "batch":
        args += ["--pages", "1", "--force"]
    if no_vad:
        args += ["--no-vad"]
    parsed = cli.build_parser().parse_args(args)
    parsed.func(parsed)
    assert len(calls) == 1
    assert calls[0]["vad_filter"] is (not no_vad)


@pytest.mark.parametrize("vad", [True, False])
def test_backend_receives_vad(monkeypatch, vad):
    seen = {}

    def transcribe(audio, **kw):
        seen.update(kw)
        return iter([]), SimpleNamespace(language="zh", language_probability=1)

    monkeypatch.setattr(asr, "to_float32_mono16k", lambda _: ([], 100))
    monkeypatch.setattr(asr, "load_model", lambda *a, **kw: SimpleNamespace(transcribe=transcribe))
    out = asr.transcribe_audio("mock.wav", language="auto", vad_filter=vad)
    assert seen["vad_filter"] is vad
    assert seen["vad_parameters"] == ({"min_silence_duration_ms": 400} if vad else None)
    assert out["params"]["vad_filter"] is vad


@pytest.mark.parametrize("vad", [True, False])
@pytest.mark.parametrize("chunk_problem", [False, True])
def test_online_asr_route_persists_internal_gap(monkeypatch, tmp_path, vad, chunk_problem):
    if chunk_problem:
        monkeypatch.setattr(workflow, "validate_coverage", lambda _: ["missing segment"])
    audio = tmp_path / "mock.wav"
    audio.write_bytes(b"mock audio")
    monkeypatch.setattr(workflow, "build_opener", lambda _: None)
    monkeypatch.setattr(workflow, "yt_dlp_version", lambda: "mock")
    monkeypatch.setattr(
        workflow,
        "fetch_video_info",
        lambda *a: {
            "bvid": "BV1xx411c7mD",
            "aid": 1,
            "title": "test",
            "uploader": "test",
            "desc": "",
            "duration": 100,
        },
    )
    monkeypatch.setattr(workflow, "resolve_page", lambda *a: {"cid": 1, "page": 1, "duration": 100})
    monkeypatch.setattr(workflow, "probe_player_subtitles", lambda *a: {})
    monkeypatch.setattr(workflow, "probe_ytdlp_subtitles", lambda *a, **kw: {})
    monkeypatch.setattr(workflow, "download_audio", lambda *a, **kw: {"path": str(audio), "size_bytes": 10})
    seen = []

    def transcribe(path, **kw):
        seen.append(kw["vad_filter"])
        return {
            "cues": [{"start": 0, "end": 5, "text": "开头"}, {"start": 95, "end": 100, "text": "结尾"}],
            "decoded_duration": 100,
            "detected_language": "zh",
            "language_probability": 1,
            "elapsed_seconds": 0,
            "params": {
                "model": "mock",
                "device": "cpu",
                "compute_type": "int8",
                "language": "zh",
                "beam_size": 5,
                "vad_filter": kw["vad_filter"],
                "vad_parameters": None,
            },
        }

    monkeypatch.setattr(workflow, "transcribe_audio", transcribe)
    result = workflow.run_ingest(
        url="https://www.bilibili.com/video/BV1xx411c7mD",
        media=None,
        output_dir=tmp_path / "out",
        vad_filter=vad,
    )
    man = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert seen == [vad]
    for key in ("full_video_coverage", "verification_status", "coverage_evidence", "coverage_basis", "notes"):
        assert result[key] == man[key]
    assert result["processing_status"] == ("partial" if chunk_problem else "resource_processed")
    assert result["full_video_coverage"] == "uncertain"
    assert man["timing"]["speech_duration"] == 10
    assert man["coverage_evidence"]["gaps"] == [
        {"start": 5, "end": 95, "duration": 90, "position": "internal"}
    ]
    assert validate_manifest(man) == []


@pytest.mark.parametrize("sample_check", [False, True])
@pytest.mark.parametrize("status", ["unreviewed", "sample_checked", "issues_found"])
@pytest.mark.parametrize("audio_present", [False, True])
def test_validate_is_read_only_and_never_invents_review(tmp_path, capsys, sample_check, status, audio_present):
    from video_ingest.manifest import save_manifest

    man = manifest()
    man.update(verification_status=status, processing_status="resource_processed")
    man["counts"]["segment_count"] = 1
    man["timing"]["decoded_duration"] = 100
    workflow._assess_coverage(man, [{"start": 0, "end": 5}], {}, {}, {"mode": "asr"})
    man["asr"]["audio_file"] = "audio.wav"
    if status == "issues_found":
        man["asr"]["language_mismatch"] = True
        man["notes"].append({"note": "语言不符，需核听"})
    if audio_present:
        (tmp_path / "audio.wav").write_bytes(b"unreviewed audio")
    (tmp_path / "chunks").mkdir()
    (tmp_path / "chunks/index.json").write_text(json.dumps({"coverage": {"complete": True}}))
    for name in ("transcript.raw.json", "transcript.txt", "transcript.srt"):
        (tmp_path / name).write_text("原始证据，不能改变", encoding="utf-8")
    save_manifest(tmp_path, man)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert cli.cmd_validate(SimpleNamespace(job_dir=str(tmp_path), sample_check=sample_check)) == cli.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["language_accuracy"]["status"] == status
    assert report["language_accuracy"]["performed_by_validate"] is False
    assert report["language_accuracy"].get("sampled_segments", 0) == 0
    assert report["resource_processing"]["coverage_evidence"] == man["coverage_evidence"]
    assert report["resource_processing"]["full_video_coverage"] == "uncertain"
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("flag", [None, False, "true", 1])
def test_supported_rejects_timestamp_only_or_nonboolean_completeness(flag):
    man = manifest()
    man.update(full_video_coverage="supported", coverage_basis=["时间轴密集、时长吻合、关闭 VAD"])
    man["coverage_evidence"] = {"method": "asr_timestamp_union_v1", "speech_completeness_verified": flag}
    assert any("supported" in p for p in validate_manifest(man))



@pytest.mark.parametrize("duration,gap,warn", [(100, 30, False), (100, 30.01, True), (1000, 50, False), (1000, 50.01, True)])
def test_long_gap_threshold_is_strictly_exceeded(duration, gap, warn):
    man = manifest()
    man["timing"]["decoded_duration"] = duration
    workflow._assess_coverage(man, [{"start": gap, "end": duration}], {}, {}, {"mode": "asr"})
    assert any("请核听该时段" in n["note"] for n in man["notes"]) is warn
    assert man["full_video_coverage"] == "uncertain"
    assert man["verification_status"] == "unreviewed"


def test_intervals_are_clipped_merged_and_do_not_mutate_raw_evidence():
    import copy

    man = manifest()
    man["timing"]["decoded_duration"] = 100
    segments = [{"start": a, "end": b} for a, b in [(90, 120), (-10, 20), (20, 30), (10, 25), (50, 50), (80, 70), (110, 130)]]
    before = copy.deepcopy(segments)
    workflow._assess_coverage(man, segments, {}, {}, {"mode": "asr"})
    assert segments == before
    assert man["coverage_evidence"]["segment_interval_duration"] == 40
    assert man["coverage_evidence"]["segment_interval_ratio"] == 0.4
    assert man["coverage_evidence"]["gaps"] == [{"start": 30, "end": 90, "duration": 60, "position": "internal"}]


@pytest.mark.parametrize("processing", ["resource_processed", "partial", "failed", "blocked"])
def test_batch_propagates_processing_failures(monkeypatch, tmp_path, capsys, processing):
    monkeypatch.setattr(cli, "build_opener", lambda _: None)
    monkeypatch.setattr(cli, "fetch_video_info", lambda *a: {"bvid": "BV1xx411c7mD", "pages": [{"page": 1}]})
    evidence = {"speech_completeness_verified": False}
    monkeypatch.setattr(cli, "run_ingest", lambda **kw: {
        "processing_status": processing, "full_video_coverage": "uncertain",
        "verification_status": "issues_found", "coverage_evidence": evidence,
        "coverage_basis": ["时间轴不足以证明完整"], "notes": [{"note": "警示：待核听"}],
    })
    args = cli.build_parser().parse_args(["batch", "--url", "https://www.bilibili.com/video/BV1xx411c7mD", "--pages", "1", "--force", "--output-dir", str(tmp_path)])
    code = args.func(args)
    report = json.loads(capsys.readouterr().out)
    assert (code == cli.EXIT_OK) is (processing == "resource_processed")
    assert report["ok"] is (processing == "resource_processed")
    assert report["results"][0]["coverage_evidence"] == evidence
    assert report["results"][0]["verification_status"] == "issues_found"


@pytest.mark.parametrize("legacy_supported", [False, True])
def test_batch_reuse_keeps_evidence_and_surfaces_legacy_inconsistency(monkeypatch, tmp_path, capsys, legacy_supported):
    from video_ingest.manifest import make_job_id, save_manifest

    monkeypatch.setattr(cli, "build_opener", lambda _: None)
    monkeypatch.setattr(cli, "fetch_video_info", lambda *a: {"bvid": "BV1xx411c7mD", "pages": [{"page": 1}]})
    monkeypatch.setattr(cli, "run_ingest", lambda **kw: pytest.fail("reused artifacts must not be overwritten"))
    job = tmp_path / make_job_id("bilibili", "BV1xx411c7mD", 1)
    job.mkdir()
    man = manifest()
    man.update(processing_status="resource_processed", verification_status="issues_found")
    man["counts"]["segment_count"] = 1
    man["timing"]["decoded_duration"] = 100
    workflow._assess_coverage(man, [{"start": 0, "end": 100}], {}, {}, {"mode": "asr"})
    if legacy_supported:
        man["full_video_coverage"] = "supported"
        man["coverage_evidence"] = {}
    save_manifest(job, man)
    before = (job / "manifest.json").read_bytes()
    args = cli.build_parser().parse_args(["batch", "--url", "https://www.bilibili.com/video/BV1xx411c7mD", "--pages", "1", "--output-dir", str(tmp_path)])
    code = args.func(args)
    row = json.loads(capsys.readouterr().out)["results"][0]
    assert row["coverage_evidence"] == man["coverage_evidence"]
    assert row["coverage_basis"] == man["coverage_basis"]
    assert row["notes"] == man["notes"]
    assert row["verification_status"] == "issues_found"
    assert row["full_video_coverage"] == man["full_video_coverage"]
    assert (code == cli.EXIT_OK) is (not legacy_supported)
    assert bool(row["consistency_problems"]) is legacy_supported
    assert (job / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("evidence,status", [
    ({"speech_completeness_verified": True, "method": "asr_timestamp_union_v1"}, "sample_checked"),
    ({"speech_completeness_verified": True, "method": "independent_full_audio_review"}, "unreviewed"),
    ({"speech_completeness_verified": True}, "sample_checked"),
    ("not an evidence object", "sample_checked"),
])
def test_supported_cannot_be_unlocked_by_flag_alone(evidence, status):
    man = manifest()
    man.update(full_video_coverage="supported", verification_status=status,
               coverage_basis=["时间轴密集"], coverage_evidence=evidence)
    assert any("supported" in p for p in validate_manifest(man))


def test_platform_subtitle_acquisition_stays_uncertain(monkeypatch, tmp_path):
    source = tmp_path / "sample.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:05,000\n测试字幕\n", encoding="utf-8")
    monkeypatch.setattr(workflow, "probe_player_subtitles", lambda *a: {
        "language_keys": [{"lan": "zh", "subtitle_url": "https://example.invalid/subtitle"}]})
    monkeypatch.setattr(workflow, "probe_ytdlp_subtitles", lambda *a, **kw: {})
    monkeypatch.setattr(workflow, "download_subtitle", lambda *a: source)
    man = manifest()
    man["asr"]["used"] = False
    man["timing"]["claimed_duration"] = 100
    result = workflow._acquire_transcript(
        man=man, job_dir=tmp_path, opener=None, info={"bvid": "BV1xx411c7mD"},
        page={"cid": 1}, norm={"canonical_url": "https://www.bilibili.com/video/BV1xx411c7mD"},
        url="https://www.bilibili.com/video/BV1xx411c7mD", language="zh", model="mock",
        device="cpu", compute_type="int8", cookies_file=None, force_asr=False,
        corrections={}, initial_prompt=None, log=lambda _: None,
    )
    workflow._assess_coverage(man, result["segments"], {}, {}, result)
    assert result["mode"] == "subtitle"
    assert len(result["segments"]) == 1
    assert man["processing_status"] == "resource_processed"
    assert man["full_video_coverage"] == "uncertain"
    assert man["verification_status"] == "unreviewed"


@pytest.mark.parametrize("evidence", [[], "", False, 0, None])
def test_falsey_malformed_evidence_is_reported(evidence):
    man = manifest()
    man["coverage_evidence"] = evidence
    assert "coverage_evidence 必须为对象" in validate_manifest(man)


@pytest.mark.parametrize("status", ["unreviewed", "sample_checked", "issues_found"])
@pytest.mark.parametrize("sample_check,audio_exists", [(False, True), (True, True), (True, False)])
def test_validate_preserves_review_status_and_evidence(tmp_path, capsys, status, sample_check, audio_exists):
    from video_ingest.manifest import save_manifest

    man = manifest()
    man["verification_status"] = status
    man["timing"].update(decoded_duration=100, claimed_duration=100)
    workflow._assess_coverage(man, [{"start": 0, "end": 5}, {"start": 95, "end": 100}], {}, {}, {"mode": "asr"})
    man["asr"]["audio_file"] = "audio.wav"
    if audio_exists:
        (tmp_path / "audio.wav").write_bytes(b"mock")
    (tmp_path / "transcript.raw.json").write_text("[]")
    (tmp_path / "chunks").mkdir()
    (tmp_path / "chunks/index.json").write_text('{"coverage": {"complete": true}}')
    save_manifest(tmp_path, man)
    before = (tmp_path / "manifest.json").read_bytes()
    assert cli.cmd_validate(SimpleNamespace(job_dir=str(tmp_path), sample_check=sample_check)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["resource_processing"]["full_video_coverage"] == "uncertain"
    assert report["resource_processing"]["coverage_evidence"] == man["coverage_evidence"]
    assert report["resource_processing"]["coverage_notes"] == man["notes"]
    assert report["language_accuracy"]["status"] == status
    if sample_check and audio_exists:
        assert report["language_accuracy"]["sampled_segments"] == 0
    assert (tmp_path / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("legacy", [True, False])
def test_validator_rejects_asr_supported_even_with_basis(legacy):
    man = manifest()
    if not legacy:
        man["timing"]["decoded_duration"] = 100
        workflow._assess_coverage(man, [{"start": 0, "end": 100}], {}, {}, {"mode": "asr"})
    else:
        man.pop("coverage_evidence")
    man["coverage_basis"] = ["时间轴达到 100%"]
    man["full_video_coverage"] = "supported"
    problems = validate_manifest(man)
    assert any("自动 ASR" in p and "supported" in p for p in problems)
    assert any("缺少 coverage_evidence" in p for p in problems) is legacy


@pytest.mark.parametrize("gap,warning", [(29.99, False), (30, False), (30.01, True)])
def test_long_gap_threshold_is_only_a_warning(gap, warning):
    man = manifest()
    man["timing"]["decoded_duration"] = 100
    workflow._assess_coverage(man, [{"start": gap, "end": 100}], {}, {}, {"mode": "asr"})
    assert man["full_video_coverage"] == "uncertain"
    assert man["coverage_evidence"]["gaps"][0]["duration"] == gap
    assert any("原因未核实" in n["note"] for n in man["notes"]) is warning


def test_interval_clipping_preserves_input_and_bounds_union():
    import copy

    man = manifest()
    man["timing"]["decoded_duration"] = 100
    segments = [{"start": -5, "end": 10}, {"start": 90, "end": 120},
                {"start": 110, "end": 130}, {"start": 40, "end": 30}]
    before = copy.deepcopy(segments)
    workflow._assess_coverage(man, segments, {}, {}, {"mode": "asr"})
    assert segments == before
    assert man["coverage_evidence"]["segment_interval_duration"] == 20
    assert man["coverage_evidence"]["segment_interval_ratio"] == 0.2
    assert man["coverage_evidence"]["gaps"] == [
        {"start": 10, "end": 90, "duration": 80, "position": "internal"}]


@pytest.mark.parametrize("duration,threshold", [(100, 30), (1000, 50)])
def test_gap_threshold_scales_with_decoded_duration(duration, threshold):
    man = manifest()
    man["timing"]["decoded_duration"] = duration
    workflow._assess_coverage(man, [{"start": threshold, "end": duration}], {}, {}, {"mode": "asr"})
    assert man["coverage_evidence"]["long_gap_threshold_seconds"] == threshold
    assert not any("原因未核实" in n["note"] for n in man["notes"])
    assert man["full_video_coverage"] == "uncertain"


def test_unknown_duration_does_not_claim_known_tail_or_ratio():
    man = manifest()
    workflow._assess_coverage(man, [{"start": 5, "end": 10}], {}, {}, {"mode": "asr"})
    evidence = man["coverage_evidence"]
    assert evidence["segment_interval_ratio"] is None
    assert evidence["long_gap_threshold_seconds"] is None
    assert evidence["duration_drift_ratio"] is None
    assert evidence["gaps"] == [{"start": 0, "end": 5, "duration": 5, "position": "leading"}]
    assert any("缺少有效解码时长" in n["note"] for n in man["notes"])


def test_validate_rejects_legacy_supported_without_rewriting(tmp_path, capsys):
    from video_ingest.manifest import save_manifest

    man = manifest()
    man["full_video_coverage"] = "supported"
    man["coverage_basis"] = ["解码时长吻合"]
    save_manifest(tmp_path, man)
    before = (tmp_path / "manifest.json").read_bytes()
    assert cli.cmd_validate(SimpleNamespace(job_dir=str(tmp_path), sample_check=False)) == cli.EXIT_FAILED
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert any("自动 ASR" in p for p in report["resource_processing"]["consistency_problems"])
    assert (tmp_path / "manifest.json").read_bytes() == before


def test_validator_rejects_timestamp_evidence_claiming_speech_verified():
    man = manifest()
    man["timing"]["decoded_duration"] = 100
    workflow._assess_coverage(man, [{"start": 0, "end": 100}], {}, {}, {"mode": "asr"})
    man["coverage_evidence"]["speech_completeness_verified"] = True
    assert any("speech_completeness_verified" in p for p in validate_manifest(man))
