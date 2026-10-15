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
def test_local_route_persists_gap_evidence(monkeypatch, tmp_path, vad, with_segments):
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
    man = json.loads(Path(result["manifest"]).read_text())
    assert seen == [vad]
    assert result["processing_status"] == ("resource_processed" if with_segments else "failed")
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
def test_online_asr_route_persists_internal_gap(monkeypatch, tmp_path, vad):
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
    man = json.loads(Path(result["manifest"]).read_text())
    assert seen == [vad]
    assert result["processing_status"] == "resource_processed"
    assert result["full_video_coverage"] == "uncertain"
    assert man["timing"]["speech_duration"] == 10
    assert man["coverage_evidence"]["gaps"] == [
        {"start": 5, "end": 95, "duration": 90, "position": "internal"}
    ]
    assert validate_manifest(man) == []
