"""编排层：把取材、转写、标准化、持久化串成一个带状态的任务。

不做理解与摘要（bootstrap §12：脚本先完成获取与完整性检查，再让 Agent 阅读）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .acquire import (
    AcquireError,
    build_opener,
    fetch_video_info,
    normalize_url,
    probe_player_subtitles,
    probe_ytdlp_subtitles,
    resolve_page,
    yt_dlp_version,
)
from .asr import download_audio, resolve_device, transcribe_audio
from .chunks import build_chunks, validate_coverage, write_chunks
from .manifest import (
    add_coverage_basis,
    add_error,
    add_note,
    make_job_id,
    new_manifest,
    safe_name,
    save_manifest,
    sha256_file,
)
from .redact import redact, redact_url
from .transcript import (
    apply_corrections,
    normalize_cues,
    parse_subtitle_file,
    to_markdown,
    to_srt,
    to_txt,
)

Logger = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# --------------------------------------------------------------------------
# 字幕下载
# --------------------------------------------------------------------------

def download_subtitle(
    opener: Any,
    subtitle_url: str,
    dest: Path,
    cookies_file: str | None = None,
) -> Path:
    """下载 B 站字幕 JSON 到源文件目录。保留原始文件，不就地改写。"""
    import urllib.request

    url = subtitle_url
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        url = "https://www.bilibili.com" + url

    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "subtitle.source.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with opener.open(req, timeout=40) as resp:
            data = resp.read()
    except Exception as exc:  # noqa: BLE001
        raise AcquireError("network_error", "字幕下载失败: %s" % redact(str(exc))) from exc
    target.write_bytes(data)
    return target


def pick_subtitle_key(keys: list[dict[str, Any]], language: str) -> dict[str, Any] | None:
    """按优先级挑字幕键：原语人工 > 原语自动 > 其它。

    bootstrap §4：不能因为文件名含"中文"就把机器翻译字幕当原语。
    """
    def is_native(k: dict[str, Any]) -> bool:
        lan = (k.get("lan") or "").lower()
        return lan.startswith("zh") or lan.startswith("ai-zh")

    native = [k for k in keys if is_native(k)]
    manual = [k for k in native if not k.get("is_ai")]
    auto = [k for k in native if k.get("is_ai")]
    for group in (manual, auto, native):
        for k in group:
            if k.get("subtitle_url"):
                return k
    return None


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------

def run_ingest(
    *,
    url: str | None,
    media: str | None,
    output_dir: Path,
    language: str = "zh",
    model: str = "medium",
    device: str = "auto",
    compute_type: str | None = None,
    cookies_file: str | None = None,
    force_asr: bool = False,
    chars_per_chunk: int = 3000,
    corrections: dict[str, str] | None = None,
    initial_prompt: str | None = None,
    log: Logger = _noop,
) -> dict[str, Any]:
    """执行一次取材任务，返回结果摘要（含 job_dir 与 manifest 路径）。

    device 支持 "auto"：先探测 GPU 是否**真正可用**（缺 cuBLAS 时硬件可见
    但推理会失败），可用则用 cuda+float16，否则回退 cpu+int8。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    corrections = corrections or {}

    requested_device = device
    resolved_device, resolved_ct, gpu_info = resolve_device(device)
    if compute_type:
        resolved_ct = compute_type
    if resolved_device != device:
        log("设备解析：%s -> %s（compute_type=%s）" % (device, resolved_device, resolved_ct))
    if resolved_device == "cuda" and not compute_type:
        # 实测 20 系卡上 GPU 的 int8 路径反而比 float16 慢，故默认 float16
        log("使用 GPU：compute_type=%s（未显式指定时默认 float16）" % resolved_ct)
    device, compute_type = resolved_device, resolved_ct

    if media:
        return _ingest_local_media(
            media=media, output_dir=output_dir, language=language, model=model,
            device=device, compute_type=compute_type, chars_per_chunk=chars_per_chunk,
            corrections=corrections, initial_prompt=initial_prompt, log=log,
            requested_device=requested_device, gpu_info=gpu_info,
        )

    if not url:
        raise AcquireError("invalid_input", "必须提供 --url 或 --media 之一")

    norm = normalize_url(url)
    if not norm["bvid"]:
        raise AcquireError("invalid_input", "无法从 URL 中识别 BV 号")

    opener = build_opener(cookies_file)
    info = fetch_video_info(opener, norm["bvid"])
    page = resolve_page(info, norm["page"])

    job_id = make_job_id("bilibili", info["bvid"], page.get("page", 1))
    job_dir = output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    man = new_manifest(
        job_id=job_id,
        input_url=redact_url(url),
        canonical_url=redact_url(norm["canonical_url"]),
        platform="bilibili",
        media_kind="bilibili_video",
    )
    man["identity"].update(
        content_id=info["bvid"], cid=page.get("cid"), page=page.get("page", 1),
        aid=info["aid"], title=info["title"], uploader=info["uploader"],
    )
    man["timing"]["claimed_duration"] = page.get("duration") or info["duration"]
    man["tooling"].update(
        video_ingest_version=__version__,
        yt_dlp_version=yt_dlp_version(),
        extractor="BiliBili",
        python_version=sys.version.split()[0],
    )
    man["subtitle"]["auth_used"] = bool(cookies_file)

    # 记录设备选择与 GPU 探测结论，便于事后解释"为什么这次跑在 CPU 上"
    man["asr"]["device_requested"] = requested_device
    man["asr"]["gpu_probe"] = {
        "available": gpu_info.get("available"),
        "cuda_device_count": gpu_info.get("cuda_device_count"),
        "reason": gpu_info.get("reason"),
        "dll_paths": gpu_info.get("dll_paths"),
    }
    if gpu_info.get("available") is False and requested_device == "auto":
        add_note(man, "GPU 探测结论为不可用，已回退 CPU。原因：%s" % gpu_info.get("reason"))

    # 保留简介作为合法辅助材料（但不作为全文，bootstrap：简介不等于全文）
    (job_dir / "source").mkdir(exist_ok=True)
    if info.get("desc"):
        (job_dir / "source" / "description.txt").write_text(info["desc"], encoding="utf-8")

    try:
        result = _acquire_transcript(
            man=man, job_dir=job_dir, opener=opener, info=info, page=page,
            norm=norm, url=url, language=language, model=model, device=device,
            compute_type=compute_type, cookies_file=cookies_file, force_asr=force_asr,
            corrections=corrections, initial_prompt=initial_prompt, log=log,
        )
    except AcquireError as exc:
        man["processing_status"] = "blocked" if exc.kind == "needs_login" else "failed"
        add_error(man, "acquire", exc.kind, str(exc))
        save_manifest(job_dir, man)
        raise

    segments = result["segments"]
    man["counts"]["segment_count"] = len(segments)
    man["timing"]["decoded_duration"] = result.get("decoded_duration")
    if segments:
        man["timing"]["last_cue_end"] = segments[-1]["end"]

    _write_transcripts(job_dir, man, segments, info=info, corrections=corrections)

    index = build_chunks(segments, chars_per_chunk=chars_per_chunk)
    problems = validate_coverage(index)
    if problems:
        add_error(man, "chunks", "coverage_incomplete", "; ".join(problems))
        man["processing_status"] = "partial"
    write_chunks(job_dir, index)
    man["counts"]["chunk_count"] = index["chunk_count"]
    man["files"]["chunks_index"] = "chunks/index.json"

    _assess_coverage(man, segments, info, page, result)

    # 注意：new_manifest 的初始 processing_status 是 "failed"（表示尚未成功）。
    # 这里必须显式改判。但"跑完流程"不等于"拿到了内容"：ASR 或字幕解析
    # 产出 0 个片段时资源实际未被处理，必须如实记为 failed，而不是
    # resource_processed（否则会与 segment_count=0 矛盾）。
    if segments:
        man["processing_status"] = "resource_processed"
    else:
        man["processing_status"] = "failed"
        add_error(man, "normalize", "no_segments",
                  "取得了资源但未产出任何片段（可能为纯音乐/静音，或识别失败）")

    save_manifest(job_dir, man)
    return {
        "ok": man["processing_status"] in ("resource_processed", "partial"),
        "job_dir": str(job_dir.resolve()),
        "manifest": str((job_dir / "manifest.json").resolve()),
        "processing_status": man["processing_status"],
        "full_video_coverage": man["full_video_coverage"],
        "segment_count": len(segments),
        "chunk_count": index["chunk_count"],
        "source_type": man["subtitle"]["source_type"],
        "files": man["files"],
    }


def _acquire_transcript(
    *, man: dict, job_dir: Path, opener: Any, info: dict, page: dict, norm: dict,
    url: str, language: str, model: str, device: str, compute_type: str,
    cookies_file: str | None, force_asr: bool, corrections: dict, initial_prompt: str | None,
    log: Logger,
) -> dict[str, Any]:
    """取得转录：优先字幕，必要时 ASR。"""
    cid = page["cid"]
    player = probe_player_subtitles(opener, info["bvid"], cid)
    ytdlp = probe_ytdlp_subtitles(norm["canonical_url"], cookies_file=cookies_file)

    keys = list(player.get("language_keys") or [])
    for k in ytdlp.get("keys") or []:
        if k.lower() == "danmaku":
            continue
        if not any(isinstance(x, dict) and x.get("lan") == k for x in keys):
            keys.append({"lan": k, "lan_doc": None, "subtitle_url": None,
                         "is_ai": str(k).lower().startswith("ai-"),
                         "via": "yt_dlp"})
    man["subtitle"]["language_keys"] = keys

    chosen = None if force_asr else pick_subtitle_key(keys, language)

    if chosen is not None:
        log("使用字幕: %s" % chosen.get("lan"))
        src = download_subtitle(opener, chosen["subtitle_url"], job_dir / "source", cookies_file)
        cues = parse_subtitle_file(src)
        source_type = "platform_auto" if chosen.get("is_ai") else "platform_manual"
        man["subtitle"].update(
            probe_state="found",
            failure_kind="not_probed",
            selected_language=chosen.get("lan"),
            source_type=source_type,
            source_file="source/%s" % src.name,
            source_sha256=sha256_file(src),
        )
        man["files"]["subtitle_source"] = "source/%s" % src.name
        segments, issues = normalize_cues(
            cues, source_type=source_type, language=chosen.get("lan") or language,
            duration=man["timing"].get("claimed_duration"),
        )
        for issue in issues:
            add_note(man, "完整性提示：%s" % issue)
        if not man["asr"]["used"]:
            man["processing_status"] = "resource_processed"
        return {"segments": segments, "mode": "subtitle",
                "decoded_duration": man["timing"].get("claimed_duration")}

    # 无可用字幕 → ASR
    if not cookies_file:
        man["subtitle"].update(probe_state="needs_auth_check", failure_kind="needs_login")
        add_note(man, "未提供 cookie：无法区分'平台侧确实无字幕'与'需登录才可见'；本次改用本地 ASR")
    else:
        man["subtitle"].update(probe_state="empty", failure_kind="no_native_subtitle")
        add_note(man, "已提供 cookie，两个渠道均未枚举到字幕键；本次改用本地 ASR")

    log("未取得字幕，改用本地 ASR（model=%s, device=%s）" % (model, device))
    dl = download_audio(url, job_dir / "source", cookies_file=cookies_file)
    log("音频已下载: %s (%.1f MB)" % (Path(dl["path"]).name, dl["size_bytes"] / 1024 / 1024))

    asr_out = transcribe_audio(
        dl["path"], model_name=model, language=language, device=device,
        compute_type=compute_type, initial_prompt=initial_prompt, on_progress=log,
    )

    man["asr"].update(
        used=True,
        model=asr_out["params"]["model"],
        compute_type=asr_out["params"]["compute_type"],
        device=asr_out["params"]["device"],
        language=asr_out["params"]["language"],
        beam_size=asr_out["params"]["beam_size"],
        vad_filter=asr_out["params"]["vad_filter"],
        vad_parameters=asr_out["params"]["vad_parameters"],
        audio_file="source/%s" % Path(dl["path"]).name,
        audio_sha256=sha256_file(dl["path"]),
    )
    man["subtitle"].update(source_type="local_asr")
    man["files"]["audio_source"] = "source/%s" % Path(dl["path"]).name
    add_note(man, "ASR 检测语言=%s (prob=%s)，耗时 %ss" % (
        asr_out["detected_language"], asr_out["language_probability"], asr_out["elapsed_seconds"]))

    _check_language_match(man, requested=language, asr_out=asr_out)

    # 清理音频以节省空间（可配置；此处保守保留，只在成功且体积大时提示）
    segments, issues = normalize_cues(
        asr_out["cues"], source_type="local_asr", language=language,
        duration=asr_out["decoded_duration"],
    )
    for issue in issues:
        add_note(man, "完整性提示：%s" % issue)

    return {
        "segments": segments,
        "mode": "asr",
        "decoded_duration": asr_out["decoded_duration"],
        "detected_language": asr_out["detected_language"],
    }


def _check_language_match(man: dict, *, requested: str, asr_out: dict) -> None:
    """检测"强制指定语言"与"实际语音语言"不一致的情况。

    为什么必须查：强制用 zh 解码英文音频时，whisper 会**流畅地吐出中文**
    ——音素结构被套进汉字，产出的每一个字都不是原话。实测同一段英文音频
    在 --language zh 下得到的是完全捏造的中文句子。这类错误不会报错、
    看起来还很通顺，是比"识别不准"更危险的失败模式。
    """
    detected = asr_out.get("detected_language")
    prob = asr_out.get("language_probability")
    if not detected or not requested or requested in ("auto", None):
        return
    base_req = str(requested).split("-")[0].lower()
    base_det = str(detected).split("-")[0].lower()
    if base_req == base_det:
        return

    add_note(man, "警示：指定语言 %s，但 ASR 检测到语音为 %s（prob=%s）。"
                  "强制指定语言与音频不符时，识别的**内容**不可信，"
                  "建议改用 --language auto 重跑。" % (requested, detected, prob))
    man.setdefault("asr", {})["language_mismatch"] = True
    man["asr"]["detected_language"] = detected
    if man.get("verification_status") == "unreviewed":
        man["verification_status"] = "issues_found"


def _ingest_local_media(
    *, media: str, output_dir: Path, language: str, model: str, device: str,
    compute_type: str, chars_per_chunk: int, corrections: dict,
    initial_prompt: str | None, log: Logger,
    requested_device: str = "cpu",
    gpu_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """处理用户本地媒体文件（无网络依赖）。"""
    gpu_info = gpu_info or {}
    path = Path(media)
    if not path.exists():
        raise AcquireError("invalid_input", "本地媒体不存在: %s" % path)

    job_id = "local_%s" % safe_name(path.stem, 40)
    job_dir = output_dir / job_id
    (job_dir / "source").mkdir(parents=True, exist_ok=True)

    man = new_manifest(
        job_id=job_id, input_url=None, canonical_url=None,
        platform="local", media_kind="local_media",
    )
    man["identity"]["title"] = path.name
    man["tooling"].update(
        video_ingest_version=__version__,
        faster_whisper_version=None,
        python_version=sys.version.split()[0],
    )
    man["asr"]["device_requested"] = requested_device
    man["asr"]["gpu_probe"] = {
        "available": gpu_info.get("available"),
        "cuda_device_count": gpu_info.get("cuda_device_count"),
        "reason": gpu_info.get("reason"),
        "dll_paths": gpu_info.get("dll_paths"),
    }
    if gpu_info.get("available") is False and requested_device == "auto":
        add_note(man, "GPU 探测结论为不可用，已回退 CPU。原因：%s" % gpu_info.get("reason"))

    asr_out = transcribe_audio(
        path, model_name=model, language=language, device=device,
        compute_type=compute_type, initial_prompt=initial_prompt, on_progress=log,
    )
    man["asr"].update(
        used=True, model=asr_out["params"]["model"], device=asr_out["params"]["device"],
        compute_type=asr_out["params"]["compute_type"], language=asr_out["params"]["language"],
        beam_size=asr_out["params"]["beam_size"], vad_filter=asr_out["params"]["vad_filter"],
        vad_parameters=asr_out["params"]["vad_parameters"],
        audio_file="source/%s" % path.name, audio_sha256=sha256_file(path),
    )
    man["subtitle"].update(source_type="local_asr", probe_state="not_probed",
                           failure_kind="not_probed", auth_used=False)
    man["timing"]["decoded_duration"] = asr_out["decoded_duration"]
    man["asr"]["detected_language"] = asr_out["detected_language"]
    _check_language_match(man, requested=language, asr_out=asr_out)

    segments, issues = normalize_cues(
        asr_out["cues"], source_type="local_asr", language=language,
        duration=asr_out["decoded_duration"],
    )
    for issue in issues:
        add_note(man, "完整性提示：%s" % issue)

    man["counts"]["segment_count"] = len(segments)
    if segments:
        man["timing"]["last_cue_end"] = segments[-1]["end"]

    _write_transcripts(job_dir, man, segments, info={"title": path.name, "desc": ""},
                       corrections=corrections)
    index = build_chunks(segments, chars_per_chunk=chars_per_chunk)
    problems = validate_coverage(index)
    if problems:
        add_error(man, "chunks", "coverage_incomplete", "; ".join(problems))
        man["processing_status"] = "partial"
    write_chunks(job_dir, index)
    man["counts"]["chunk_count"] = index["chunk_count"]
    man["files"]["chunks_index"] = "chunks/index.json"

    man["full_video_coverage"] = "supported" if segments else "uncertain"
    add_coverage_basis(man, "输入为本地完整媒体文件，ASR 覆盖该文件全部时间轴")
    add_note(man, "本地媒体任务：无平台侧字幕信息，来源标记为 local_asr")
    # 同 bilibili 路径：有片段才算处理完成，否则如实记为 failed
    if segments:
        man["processing_status"] = "resource_processed"
    else:
        man["processing_status"] = "failed"
        add_error(man, "normalize", "no_segments",
                  "本地媒体未产出任何片段（可能为纯音乐/静音，或识别失败）")

    save_manifest(job_dir, man)
    return {
        "ok": man["processing_status"] in ("resource_processed", "partial"),
        "job_dir": str(job_dir.resolve()),
        "manifest": str((job_dir / "manifest.json").resolve()),
        "processing_status": man["processing_status"],
        "full_video_coverage": man["full_video_coverage"],
        "segment_count": len(segments), "chunk_count": index["chunk_count"],
        "source_type": "local_asr", "files": man["files"],
    }


# --------------------------------------------------------------------------
# 输出与覆盖评估
# --------------------------------------------------------------------------

def _write_transcripts(
    job_dir: Path, man: dict, segments: list[dict[str, Any]], *,
    info: dict, corrections: dict[str, str],
) -> None:
    """写原始层与派生的可读层。原始层永不被校正稿覆盖。"""
    raw = {"job_id": man["job_id"], "source_type": man["subtitle"]["source_type"],
           "language": man["asr"].get("language"), "segments": segments}
    (job_dir / "transcript.raw.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    (job_dir / "transcript.txt").write_text(to_txt(segments), encoding="utf-8")
    (job_dir / "transcript.srt").write_text(to_srt(segments), encoding="utf-8")
    (job_dir / "transcript.md").write_text(
        to_markdown(segments, title=info.get("title") or man["job_id"]), encoding="utf-8")

    man["files"].update(
        transcript_raw="transcript.raw.json",
        transcript_txt="transcript.txt",
        transcript_srt="transcript.srt",
        transcript_md="transcript.md",
    )
    if info.get("desc"):
        man["files"]["description"] = "source/description.txt"

    if corrections:
        corrected, records = apply_corrections(segments, corrections)
        (job_dir / "transcript.readable.md").write_text(
            to_markdown(corrected, title=(info.get("title") or man["job_id"]) + "（校正稿）"),
            encoding="utf-8")
        (job_dir / "corrections.json").write_text(
            json.dumps({"table": corrections, "records": records}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        man["files"]["transcript_readable"] = "transcript.readable.md"
        man["files"]["corrections"] = "corrections.json"
        add_note(man, "已应用 %d 条校正（独立层，未覆盖原文）；共 %d 处替换"
                 % (len(corrections), len(records)))
    else:
        add_note(man, "未提供校正表；转录为未校正原始层")


def _assess_coverage(
    man: dict, segments: list[dict[str, Any]], info: dict, page: dict, result: dict
) -> None:
    """评估 full_video_coverage 并记录依据（bootstrap §10）。"""
    claimed = man["timing"].get("claimed_duration")
    decoded = man["timing"].get("decoded_duration")
    last_end = man["timing"].get("last_cue_end")

    if not segments:
        man["full_video_coverage"] = "uncertain"
        add_coverage_basis(man, "没有任何片段")
        return

    if result.get("mode") == "asr":
        add_coverage_basis(man, "本地 ASR 覆盖完整音频时长（%s 秒）" % decoded)
        if claimed and decoded:
            drift = abs(decoded - claimed) / claimed
            add_coverage_basis(man, "解码时长与平台声称时长偏差 %.1f%%" % (drift * 100))
            man["full_video_coverage"] = "supported" if drift <= 0.05 else "uncertain"
        else:
            man["full_video_coverage"] = "supported"
    else:
        add_coverage_basis(man, "使用平台字幕，仅证明下载了该字幕文件")
        if claimed and last_end:
            ratio = last_end / claimed
            add_coverage_basis(man, "最后一条字幕结束于 %.1fs / 声称 %.1fs（%.0f%%）"
                               % (last_end, claimed, ratio * 100))
            # 字幕时间稀疏可能因片头片尾无声，不能当成识别准确率
            man["full_video_coverage"] = "uncertain"
            add_coverage_basis(man, "字幕时间密度不能独立证明覆盖全部讲话（片头片尾/无声演示会造成稀疏）")
        else:
            man["full_video_coverage"] = "uncertain"

    if man["full_video_coverage"] != "supported":
        add_note(man, "警示：full_video_coverage=%s，输出结论时必须带上该限制，不得写成'完整视频总结'"
                 % man["full_video_coverage"])
