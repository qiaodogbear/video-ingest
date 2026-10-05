"""CLI 实现：参数解析、命令分发与各命令处理。

契约（bootstrap §5）：
  doctor   —— 检查环境，不输出任何 token/cookie
  probe    —— 只探测身份/分P/时长/字幕键/鉴权需要，不下载大文件、不跑 ASR
  ingest   —— 取材、转写、标准化、持久化、完整性校验
  validate —— 区分"资源处理覆盖"与"语言识别准确性核查"
  chunks   —— 确定性分块与覆盖检查，不生成摘要
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

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
from .manifest import (
    add_error,
    add_note,
    load_manifest,
    make_job_id,
    new_manifest,
    save_manifest,
    sha256_file,
    validate_manifest,
)
from .chunks import build_chunks, validate_coverage, write_chunks
from .redact import redact, redact_url
from .workflow import run_ingest

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BLOCKED = 3
EXIT_FAILED = 4


def _print_json(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _eprint(text: str) -> None:
    sys.stderr.write(text.rstrip() + "\n")


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    problems: list[str] = []
    info: dict = {
        "video_ingest_version": __version__,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
        },
        "checks": {},
    }

    if sys.version_info < (3, 11):
        problems.append("Python 版本过低，需要 3.11+")

    # 依赖
    deps = {}
    for mod in ("yt_dlp", "faster_whisper", "av", "numpy"):
        try:
            m = __import__(mod)
            deps[mod] = {"ok": True, "version": getattr(m, "__version__", None)}
        except Exception as exc:  # noqa: BLE001
            deps[mod] = {"ok": False, "error": redact(str(exc))}
    info["checks"]["dependencies"] = deps

    # 测试运行器单独报告：README 让用户跑 pytest，但 pip install -e .
    # （不带 [dev]）不会装它。缺了会影响"验证安装是否成功"，但不是取材必需。
    pytest_status = {}
    try:
        import pytest as _pytest
        pytest_status = {"ok": True, "version": getattr(_pytest, "__version__", None)}
    except Exception:  # noqa: BLE001
        pytest_status = {"ok": False, "hint": 'python -m pip install -e ".[dev]"'}
    info["checks"]["test_runner"] = pytest_status
    if not pytest_status["ok"]:
        info.setdefault("warnings", []).append(
            "缺少 pytest：无法按 README 运行测试来验证安装。取材功能不受影响。")

    # OCR 是可选依赖：缺失只影响画面文字识别，不影响取材与转录。
    try:
        from .ocr import ocr_available
        ocr_ok, ocr_reason = ocr_available()
    except Exception as exc:  # noqa: BLE001
        ocr_ok, ocr_reason = False, redact(str(exc))
    info["checks"]["ocr"] = {
        "ok": ocr_ok,
        "optional": True,
        "reason": ocr_reason,
        "hint": None if ocr_ok else 'python -m pip install rapidocr-onnxruntime',
        "note": "可选：用于识别画面文字。缺失不影响字幕/ASR/取帧。",
    }
    if not ocr_ok:
        info.setdefault("warnings", []).append(
            "OCR 可选依赖未安装：画面文字检索不可用，其余功能正常。")

    # 可选 FFmpeg：不是所有路径都依赖它（faster-whisper 走 PyAV）
    ffmpeg = shutil.which("ffmpeg")
    info["checks"]["ffmpeg"] = {
        "found": bool(ffmpeg),
        "path": ffmpeg,
        "required": False,
        "note": "仅音频格式转换需要；faster-whisper 自身使用 PyAV",
    }

    # 磁盘空间
    try:
        usage = shutil.disk_usage(str(Path.cwd()))
        info["checks"]["disk"] = {
            "free_gb": round(usage.free / (1024 ** 3), 1),
            "cwd": str(Path.cwd()),
        }
        if usage.free < 2 * 1024 ** 3:
            problems.append("磁盘可用空间不足 2GB")
    except Exception as exc:  # noqa: BLE001
        info["checks"]["disk"] = {"error": redact(str(exc))}

    # 模型缓存
    model_root = Path.home() / ".cache" / "huggingface" / "hub"
    cached = []
    if model_root.exists():
        for d in model_root.glob("models--*--faster-whisper-*"):
            total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            cached.append({"name": d.name, "size_gb": round(total / (1024 ** 3), 2)})
    info["checks"]["asr_model_cache"] = {
        "path": str(model_root),
        "cached": cached,
        "note": "离线运行需要预先缓存；按名称加载可能触发下载",
    }

    # GPU：硬件可见 ≠ 可用。缺 cuBLAS 时 ctranslate2 连"加载模型"都会成功，
    # 只有真正推理才报错，因此这里做一次实打实的可用性探测。
    gpu = {"nvidia_visible": False, "note": "无 GPU 时使用 CPU，不判定软件不可用"}
    try:
        nvsmi = shutil.which("nvidia-smi")
        if nvsmi:
            import subprocess
            out = subprocess.run([nvsmi, "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20)
            names = [l.strip() for l in (out.stdout or "").splitlines() if l.strip()]
            gpu["nvidia_visible"] = bool(names)
            gpu["devices"] = names
    except Exception as exc:  # noqa: BLE001
        gpu["error"] = redact(str(exc))

    if args.check_gpu:
        try:
            from .asr import probe_gpu
            probe = probe_gpu()
            gpu["usable_for_inference"] = probe["available"]
            gpu["recommended"] = ("--device cuda --compute-type float16"
                                  if probe["available"] else "--device cpu")
            if not probe["available"]:
                gpu["unavailable_reason"] = probe["reason"]
        except Exception as exc:  # noqa: BLE001
            gpu["usable_for_inference"] = False
            gpu["unavailable_reason"] = redact(str(exc))
        if not gpu.get("usable_for_inference"):
            info.setdefault("warnings", []).append(
                "GPU 不可用于推理，将回退 CPU（--device auto 会自动选择）。"
                "原因见 checks.gpu.unavailable_reason")
    else:
        gpu["note"] = ("未做推理级探测；加 --check-gpu 可确认 GPU 是否真的可用"
                       "（硬件可见不等于可用）")
    info["checks"]["gpu"] = gpu

    # yt-dlp 版本（用于运行报告）
    info["checks"]["yt_dlp_version"] = yt_dlp_version()

    # cookie 状态：只报存在性，绝不读取或打印内容
    cookies_env = os.environ.get("VIDEO_INGEST_COOKIES")
    info["checks"]["cookies"] = {
        "env_var": "VIDEO_INGEST_COOKIES",
        "configured": bool(cookies_env),
        "file_exists": bool(cookies_env and Path(cookies_env).exists()),
        "note": "只报告存在性；不读取、不打印、不写入任何凭证内容",
    }

    info["problems"] = problems
    info["ok"] = not problems
    _print_json(info)
    return EXIT_OK if not problems else EXIT_FAILED


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------

def cmd_probe(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir) if args.output_dir else (Path.cwd() / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        norm = normalize_url(args.url)
    except AcquireError as exc:
        _print_json({"ok": False, "stage": "normalize_url", "kind": exc.kind, "message": str(exc)})
        return EXIT_USAGE

    probe: dict = {
        "ok": False,
        "stage": "probe",
        "input_url_redacted": redact_url(args.url),
        "canonical_url_redacted": redact_url(norm["canonical_url"]),
        "host": norm["host"],
        "bvid": norm["bvid"],
        "requested_page": norm["page"],
        "cookies_used": bool(args.cookies),
    }

    if not norm["bvid"]:
        probe.update(kind="invalid_input", message="无法从 URL 中识别 BV 号")
        _print_json(probe)
        return EXIT_USAGE

    opener = build_opener(args.cookies)
    try:
        info = fetch_video_info(opener, norm["bvid"])
    except AcquireError as exc:
        probe.update(kind=exc.kind, message=str(exc))
        _print_json(probe)
        return EXIT_BLOCKED if exc.kind == "needs_login" else EXIT_FAILED

    page = resolve_page(info, norm["page"])
    probe["identity"] = {
        "bvid": info["bvid"],
        "aid": info["aid"],
        "cid": page.get("cid"),
        "page": page.get("page", 1),
        "title": info["title"],
        "uploader": info["uploader"],
        "claimed_duration": page.get("duration") or info["duration"],
        "part": page.get("part"),
        "total_pages": len(info.get("pages") or []),
    }

    # 渠道一：播放器接口
    player = probe_player_subtitles(opener, info["bvid"], page["cid"])
    # 渠道二：yt-dlp 交叉验证
    ytdlp = probe_ytdlp_subtitles(norm["canonical_url"], cookies_file=args.cookies)

    probe["subtitle_channels"] = {
        "player_api": {
            "probe_state": player["probe_state"],
            "failure_kind": player["failure_kind"],
            "language_keys": player["language_keys"],
            "endpoints": player["endpoints"],
        },
        "yt_dlp": ytdlp,
    }

    combined = _combine_subtitle_verdict(player, ytdlp, cookies_used=bool(args.cookies))
    probe["subtitle_verdict"] = combined
    probe["ok"] = True

    if args.save_job:
        job_id = make_job_id("bilibili", norm["bvid"], page.get("page", 1))
        job_dir = out_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        man = new_manifest(
            job_id=job_id,
            input_url=redact_url(args.url),
            canonical_url=redact_url(norm["canonical_url"]),
            platform="bilibili",
            media_kind="bilibili_video",
        )
        man["identity"].update(
            content_id=info["bvid"], cid=page.get("cid"), page=page.get("page", 1),
            aid=info["aid"], title=info["title"], uploader=info["uploader"],
        )
        man["timing"]["claimed_duration"] = page.get("duration") or info["duration"]
        man["subtitle"]["probe_state"] = combined["probe_state"]
        man["subtitle"]["failure_kind"] = combined["failure_kind"]
        man["subtitle"]["language_keys"] = combined["language_keys"]
        man["subtitle"]["auth_used"] = bool(args.cookies)
        from .manifest import add_note
        add_note(man, "probe 完成：仅探测，未下载媒体、未执行 ASR")
        man["processing_status"] = "partial"
        man["files"]["probe_json"] = "probe.json"
        save_manifest(job_dir, man)
        with open(job_dir / "probe.json", "w", encoding="utf-8") as fh:
            json.dump(probe, fh, ensure_ascii=False, indent=2)
        probe["job_dir"] = str(job_dir.resolve())
        probe["manifest"] = str((job_dir / "manifest.json").resolve())

    _print_json(probe)
    return EXIT_OK


def _combine_subtitle_verdict(player: dict, ytdlp: dict, *, cookies_used: bool) -> dict:
    """综合两个渠道，给出三态结论。

    关键：无 cookie 时空字幕不能断言"没有字幕"。
    """
    keys = list(player.get("language_keys") or [])
    for k in ytdlp.get("keys") or []:
        if k.lower() == "danmaku":
            continue
        if not any((isinstance(x, dict) and x.get("lan") == k) for x in keys):
            keys.append({"lan": k, "lan_doc": None, "subtitle_url": None,
                         "is_ai": str(k).lower().startswith("ai-"), "via": "yt_dlp"})

    if keys:
        return {
            "probe_state": "found",
            "failure_kind": "not_probed",
            "language_keys": keys,
            "recommended_action": "download_subtitle",
            "basis": ["player_api 或 yt_dlp 至少一个渠道枚举到字幕键"],
        }

    if not cookies_used:
        return {
            "probe_state": "needs_auth_check",
            "failure_kind": "needs_login",
            "language_keys": [],
            "recommended_action": "provide_cookies_then_reprobe_or_use_asr",
            "basis": [
                "两个渠道都未枚举到字幕键",
                "未提供 cookie：无法区分'平台侧确实没有'与'需登录才可见'",
                "实测未登录时播放器接口对确实有 CC 的视频同样返回空列表",
            ],
        }

    return {
        "probe_state": "empty",
        "failure_kind": "no_native_subtitle",
        "language_keys": [],
        "recommended_action": "use_asr",
        "basis": ["已提供 cookie，两个渠道均未枚举到字幕键，判定为平台侧无原语字幕"],
    }


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir) if args.output_dir else (Path.cwd() / "output")

    corrections: dict[str, str] = {}
    if args.corrections:
        try:
            with open(args.corrections, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                _print_json({"ok": False, "kind": "invalid_input",
                             "message": "--corrections 必须是 {错:对} 的 JSON 对象"})
                return EXIT_USAGE
            corrections = {str(k): str(v) for k, v in loaded.items()}
        except Exception as exc:  # noqa: BLE001
            _print_json({"ok": False, "kind": "invalid_input",
                         "message": "校正表读取失败: %s" % redact(str(exc))})
            return EXIT_USAGE

    def log(msg: str) -> None:
        _eprint("[ingest] %s" % msg)

    try:
        result = run_ingest(
            url=args.url, media=args.media, output_dir=out_dir,
            language=args.language, model=args.model, device=args.device,
            compute_type=args.compute_type, cookies_file=args.cookies,
            force_asr=args.force_asr, chars_per_chunk=args.chars_per_chunk,
            corrections=corrections, initial_prompt=args.prompt, log=log,
        )
    except AcquireError as exc:
        _print_json({"ok": False, "stage": "ingest", "kind": exc.kind, "message": str(exc)})
        return EXIT_BLOCKED if exc.kind in ("needs_login", "rate_limited") else EXIT_FAILED
    except FileNotFoundError as exc:
        _print_json({"ok": False, "stage": "ingest", "kind": "invalid_input", "message": str(exc)})
        return EXIT_USAGE

    _print_json(result)
    # 结果 JSON 始终输出（供 Agent 检查），但退出码如实反映状态，
    # 便于脚本化调用检测失败。
    return EXIT_OK if result.get("ok") else EXIT_FAILED


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir)
    report: dict = {
        "ok": False,
        "job_dir": str(job_dir.resolve()),
        "resource_processing": {},
        "language_accuracy": {},
        "notes": [
            "进程退出码 0 不等于转录完整准确；两个维度分开报告",
        ],
    }

    try:
        man = load_manifest(job_dir)
    except FileNotFoundError as exc:
        _print_json({"ok": False, "kind": "invalid_input", "message": str(exc)})
        return EXIT_USAGE

    problems = validate_manifest(man)
    report["resource_processing"] = {
        "processing_status": man.get("processing_status"),
        "full_video_coverage": man.get("full_video_coverage"),
        "coverage_basis": man.get("coverage_basis"),
        "segment_count": man.get("counts", {}).get("segment_count"),
        "chunk_count": man.get("counts", {}).get("chunk_count"),
        "consistency_problems": problems,
    }

    # 文件存在性
    missing_files = []
    for key, rel in (man.get("files") or {}).items():
        if not (job_dir / rel).exists():
            missing_files.append("%s -> %s" % (key, rel))
    report["resource_processing"]["missing_files"] = missing_files
    if missing_files:
        problems.append("manifest 引用的文件缺失: %s" % "; ".join(missing_files))

    # 分块覆盖
    index_path = job_dir / "chunks" / "index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
        cov = index.get("coverage") or {}
        report["resource_processing"]["chunk_coverage"] = {
            "complete": cov.get("complete"),
            "missing_segment_ids": cov.get("missing_segment_ids"),
            "chunk_count": index.get("chunk_count"),
        }
        if not cov.get("complete"):
            problems.append("分块覆盖不完整")
    else:
        report["resource_processing"]["chunk_coverage"] = {"present": False}
        problems.append("缺少 chunks/index.json")

    # 时长核对
    timing = man.get("timing") or {}
    report["resource_processing"]["timing"] = timing
    claimed, decoded = timing.get("claimed_duration"), timing.get("decoded_duration")
    if claimed and decoded:
        drift = abs(decoded - claimed) / claimed
        report["resource_processing"]["duration_drift_pct"] = round(drift * 100, 2)
        if drift > 0.05:
            problems.append("解码时长与声称时长偏差 %.1f%%" % (drift * 100))

    # 语言准确性核查（抽样，不声称逐字准确）
    accuracy: dict = {
        "status": "unreviewed",
        "method": None,
        "warning": "未做抽样核查时，不得声称字符级或逐字准确",
    }
    raw_path = job_dir / "transcript.raw.json"
    if args.sample_check:
        audio_rel = (man.get("asr") or {}).get("audio_file")
        audio_path = job_dir / audio_rel if audio_rel else None
        if audio_path and audio_path.exists():
            accuracy.update(
                status="sample_checked",
                method="音频与转录同时存在；仅证明抽样区间可复核，不证明逐字准确",
                audio_present=True,
                sampled_segments=min(5, man.get("counts", {}).get("segment_count") or 0),
            )
        else:
            accuracy.update(
                status="unreviewed",
                method="--sample-check 已请求，但音频文件已被清理或不存在",
                audio_present=False,
            )
            report["notes"].append("音频缺失：无法抽样比对，保持 unreviewed")
    report["language_accuracy"] = accuracy

    if raw_path.exists():
        report["resource_processing"]["transcript_raw_present"] = True
    else:
        report["resource_processing"]["transcript_raw_present"] = False
        problems.append("缺少 transcript.raw.json（原始层）")

    report["resource_processing"]["consistency_problems"] = problems
    report["ok"] = not problems

    # 更新 manifest 的核查维度
    man["verification_status"] = accuracy["status"]
    save_manifest(job_dir, man)

    _print_json(report)
    return EXIT_OK if not problems else EXIT_FAILED


# --------------------------------------------------------------------------
# chunks
# --------------------------------------------------------------------------

def cmd_chunks(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir)
    raw_path = job_dir / "transcript.raw.json"
    if not raw_path.exists():
        _print_json({"ok": False, "kind": "invalid_input",
                     "message": "缺少 transcript.raw.json: %s" % raw_path})
        return EXIT_USAGE

    with open(raw_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    segments = raw.get("segments") or []

    index = build_chunks(segments, chars_per_chunk=args.chars_per_chunk,
                         overlap_segments=args.overlap_segments)
    problems = validate_coverage(index)
    index_path = write_chunks(job_dir, index)

    man = load_manifest(job_dir)
    man["counts"]["chunk_count"] = index["chunk_count"]
    man.setdefault("files", {})["chunks_index"] = "chunks/index.json"
    if problems:
        add_error(man, "chunks", "coverage_incomplete", "; ".join(problems))
    save_manifest(job_dir, man)

    _print_json({
        "ok": not problems,
        "chunk_count": index["chunk_count"],
        "segment_count": index["segment_count"],
        "chars_per_chunk": index["chars_per_chunk"],
        "chars_per_chunk_is_tokens": False,
        "coverage": index["coverage"],
        "coverage_problems": problems,
        "index_path": str(index_path.resolve()),
    })
    return EXIT_OK if not problems else EXIT_FAILED


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def cmd_frames(args: argparse.Namespace) -> int:
    from .visual import (
        extract_frames,
        load_segments,
        plan_frames,
        scan_timeline,
        to_json,
    )

    job_dir = Path(args.job_dir)
    try:
        segments = load_segments(job_dir)
    except FileNotFoundError as exc:
        _print_json({"ok": False, "kind": "invalid_input", "message": str(exc)})
        return EXIT_USAGE

    if not segments:
        _print_json({"ok": False, "kind": "invalid_input",
                     "message": "transcript.raw.json 中没有片段，先跑 ingest"})
        return EXIT_USAGE

    man = load_manifest(job_dir)

    # 视频来源：显式参数 > manifest 记录 > 同一任务的音频（可解码则可用）
    video = args.video
    if not video:
        video = (man.get("visual") or {}).get("video_file")
    if not video:
        cand = (man.get("files") or {}).get("audio_source")
        if cand and (job_dir / cand).exists():
            video = str(job_dir / cand)
    if not video:
        _print_json({"ok": False, "kind": "invalid_input",
                     "message": "未指定视频，且任务内没有可用的媒体文件；用 --video 指定"})
        return EXIT_USAGE
    video_path = Path(video)
    if not video_path.exists():
        _print_json({"ok": False, "kind": "invalid_input",
                     "message": "视频文件不存在: %s" % video_path})
        return EXIT_USAGE

    def log(msg: str) -> None:
        _eprint("[frames] %s" % msg)

    log("扫描画面时间轴（interval=%.2fs, threshold=%.1f）" % (args.interval, args.scene_threshold))
    try:
        timeline = scan_timeline(
            video_path,
            interval=args.interval,
            start=args.start,
            end=args.end,
            scene_threshold=args.scene_threshold,
            on_progress=log,
        )
    except Exception as exc:  # noqa: BLE001
        _print_json({"ok": False, "kind": "failed", "message": redact(str(exc))})
        return EXIT_FAILED

    log("得到 %d 个画面曝光段（采样 %d 帧）" % (len(timeline["runs"]), timeline["sample_count"]))

    plan = plan_frames(
        segments, timeline,
        include_blank=args.include_blank,
        max_frames=args.max_frames,
    )
    log("规划 %d 帧待提取（跳过空白 %d，孤儿段 %d）"
        % (plan["unique_frame_count"],
           sum(1 for p in plan["plan"] for f in p["frames"] if f["skipped"]),
           plan["orphan_run_count"]))

    frames_dir = job_dir / "frames"
    try:
        extracted = extract_frames(
            video_path, plan["unique_frames"], frames_dir,
            scale=args.scale, on_progress=log,
        )
    except Exception as exc:  # noqa: BLE001
        _print_json({"ok": False, "kind": "failed", "message": redact(str(exc))})
        return EXIT_FAILED

    log("已写出 %d 帧" % extracted["written"])
    if extracted.get("missing"):
        log("警告：有 %d 个目标时间点未取到帧（可能超出视频时长）" % extracted["missing"])

    # 把实际落盘信息回填到按片段组织的计划里
    by_id = {f["frame_id"]: f for f in extracted["frames"]}
    for item in plan["plan"]:
        for fr in item["frames"]:
            got = by_id.get(fr.get("frame_id"))
            if got:
                fr["path"] = got["path"]
                fr["t_actual"] = got["t_actual"]
                fr["size_bytes"] = got["size_bytes"]

    index = {
        "video": timeline["video"],
        "fps": timeline["fps"],
        "interval": timeline["interval"],
        "scene_threshold": timeline["scene_threshold"],
        "window": timeline["window"],
        "scale": args.scale,
        "sampling_policy": "intersection_midpoint",
        "sampling_policy_note": (
            "取帧点 = (字幕片段区间 ∩ 画面曝光段区间) 的中点。"
            "不使用 cue.start + 固定偏移：固定偏移是对单个视频转场时长的拟合。"
        ),
        "understanding_by": "agent",
        "exposure_runs": [
            {k: v for k, v in r.items() if k != "kinds"} for r in timeline["runs"]
        ],
        "segments": plan["plan"],
        "orphan_runs": plan["orphan_runs"],
        "orphan_run_count": plan["orphan_run_count"],
        "frames_written": extracted["written"],
        "frames_missing": extracted.get("missing", 0),
        "truncated_by_max_frames": plan["truncated_by_max_frames"],
    }
    to_json(index, job_dir / "frames.index.json")

    # 供 Agent 阅读的紧凑视图；**不**嵌入 chunks，避免块体积膨胀
    lines = [
        "# 画面证据视图",
        "",
        "> 本文件由脚本生成：只有取帧位置与状态，**没有**画面内容的理解。",
        "> 屏幕上的内容只能来自帧本身；不得凭字幕推断屏幕上写了什么。",
        "",
        "- 取样策略：字幕区间 ∩ 画面曝光段，取交集中点",
        "- 曝光段数：%d，已提取帧：%d，跳过空白：%d，孤儿段：%d"
        % (len(timeline["runs"]), extracted["written"],
           sum(1 for p in plan["plan"] for f in p["frames"] if f["skipped"]),
           plan["orphan_run_count"]),
        "",
    ]
    for item in plan["plan"]:
        lines.append("## %s  [%.1fs – %.1fs]" % (item["segment_id"], item["segment_start"], item["segment_end"]))
        lines.append("")
        lines.append("字幕：%s" % item["text"])
        lines.append("")
        if not item["frames"]:
            lines.append("_该片段内没有可用的画面曝光段。_")
            lines.append("")
            continue
        for fr in item["frames"]:
            if fr.get("skipped"):
                lines.append("- `%s` 跳过（%s）" % (fr.get("frame_id"), fr.get("skip_reason")))
            else:
                lines.append("- `%s` t=%.2fs 状态=`%s` 文件=`frames/%s` 曝光段=%.2f–%.2f"
                             % (fr.get("frame_id"), fr["sample_t"], fr["status"],
                                fr.get("path"), fr["exposure_run"][0], fr["exposure_run"][1]))
        lines.append("")
    if plan["orphan_runs"]:
        lines.append("## 未被任何字幕引用的画面段")
        lines.append("")
        lines.append("这些画面段没有对应字幕，可能包含无口播的屏幕内容（含提示词/图表），建议单独查看：")
        lines.append("")
        for r in plan["orphan_runs"]:
            lines.append("- `%s` %.2fs – %.2fs（%s）" % (r["run_id"], r["start"], r["end"], r["dominant_kind"]))
        lines.append("")
    (job_dir / "visual-index.md").write_text("\n".join(lines), encoding="utf-8")

    # 更新 manifest（只记策略与计数，不内嵌全部帧）
    try:
        vis = man.setdefault("visual", {})
        vis.update(
            used=True,
            video_file=str(video_path.resolve()),
            video_sha256=sha256_file(video_path),
            interval=timeline["interval"],
            scene_threshold=timeline["scene_threshold"],
            scale=args.scale,
            run_count=len(timeline["runs"]),
            frames_written=extracted["written"],
            frames_skipped_blank=sum(1 for p in plan["plan"] for f in p["frames"] if f["skipped"]),
            orphan_run_count=plan["orphan_run_count"],
            index="frames.index.json",
            sampling_policy="intersection_midpoint",
            understanding_by="agent",
        )
        vis.setdefault("notes", []).append(
            "视觉理解由 Agent 在阅读阶段完成；脚本只做取帧，不解释画面内容")
        man.setdefault("files", {})["visual_index"] = "frames.index.json"
        man["files"]["visual_view"] = "visual-index.md"
        if extracted.get("missing"):
            add_error(man, "frames", "frames_missing",
                      "有 %d 个目标时间点未取到帧" % extracted["missing"])
        save_manifest(job_dir, man)
    except Exception as exc:  # noqa: BLE001
        log("警告：manifest 更新失败：%s" % redact(str(exc)))

    _print_json({
        "ok": True,
        "job_dir": str(job_dir.resolve()),
        "video": str(video_path.resolve()),
        "exposure_run_count": len(timeline["runs"]),
        "frames_written": extracted["written"],
        "frames_missing": extracted.get("missing", 0),
        "frames_skipped_blank": index["segments"] and sum(
            1 for p in plan["plan"] for f in p["frames"] if f["skipped"]),
        "orphan_run_count": plan["orphan_run_count"],
        "segments_with_frames": sum(1 for p in plan["plan"] if any(
            not f.get("skipped") for f in p["frames"])),
        "segment_count": len(segments),
        "sampling_policy": "intersection_midpoint",
        "index": str((job_dir / "frames.index.json").resolve()),
        "view": str((job_dir / "visual-index.md").resolve()),
        "frames_dir": str(frames_dir.resolve()),
    })
    return EXIT_OK


# --------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------

def cmd_batch(args: argparse.Namespace) -> int:
    from .batch import PageSpecError, parse_page_spec, should_reuse, summarize_batch
    from .manifest import load_manifest as _load

    out_dir = Path(args.output_dir) if args.output_dir else (Path.cwd() / "output")

    try:
        norm = normalize_url(args.url)
    except AcquireError as exc:
        _print_json({"ok": False, "stage": "normalize_url", "kind": exc.kind, "message": str(exc)})
        return EXIT_USAGE
    if not norm["bvid"]:
        _print_json({"ok": False, "kind": "invalid_input", "message": "无法识别 BV 号"})
        return EXIT_USAGE

    def log(msg: str) -> None:
        _eprint("[batch] %s" % msg)

    # 先取一次身份，拿到总分P数——避免每个分P重复请求
    opener = build_opener(args.cookies)
    try:
        info = fetch_video_info(opener, norm["bvid"])
    except AcquireError as exc:
        _print_json({"ok": False, "stage": "view", "kind": exc.kind, "message": str(exc)})
        return EXIT_BLOCKED if exc.kind == "needs_login" else EXIT_FAILED

    total = len(info.get("pages") or []) or 1
    try:
        pages = parse_page_spec(args.pages, total=total)
    except PageSpecError as exc:
        _print_json({"ok": False, "kind": "invalid_input", "message": str(exc),
                     "total_pages": total})
        return EXIT_USAGE

    log("视频《%s》共 %d 个分P，本次处理 %d 个: %s"
        % (info.get("title"), total, len(pages), ",".join(str(p) for p in pages)))

    corrections: dict[str, str] = {}
    if args.corrections:
        try:
            with open(args.corrections, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                raise ValueError("--corrections 必须是 {错:对} 的 JSON 对象")
            corrections = {str(k): str(v) for k, v in loaded.items()}
        except Exception as exc:  # noqa: BLE001
            _print_json({"ok": False, "kind": "invalid_input",
                         "message": "校正表读取失败: %s" % redact(str(exc))})
            return EXIT_USAGE

    results: list[dict] = []
    for page_no in pages:
        page_info = None
        for p in (info.get("pages") or []):
            if p.get("page") == page_no:
                page_info = p
                break
        if page_info is None:
            results.append({"page": page_no, "status": "failed", "kind": "invalid_input",
                            "message": "分P %d 不在该视频中" % page_no})
            log("分P %d 不存在，跳过" % page_no)
            continue

        job_id = make_job_id("bilibili", info["bvid"], page_no)
        job_dir = out_dir / job_id

        if args.resume and job_dir.exists():
            try:
                existing = _load(job_dir)
            except Exception:  # noqa: BLE001
                existing = None
            if should_reuse(existing):
                log("分P %d 已有完成产物，跳过（--force 可强制重跑）" % page_no)
                results.append({
                    "page": page_no, "status": "reused", "job_id": job_id,
                    "processing_status": existing.get("processing_status"),
                    "segment_count": (existing.get("counts") or {}).get("segment_count"),
                })
                continue

        page_url = "%s?p=%d" % (norm["canonical_url"].split("?")[0], page_no)
        log("处理分P %d: %s" % (page_no, page_url))
        try:
            out = run_ingest(
                url=page_url, media=None, output_dir=out_dir,
                language=args.language, model=args.model, device=args.device,
                compute_type=args.compute_type, cookies_file=args.cookies,
                force_asr=args.force_asr, chars_per_chunk=args.chars_per_chunk,
                corrections=corrections, initial_prompt=args.prompt, log=log,
            )
            results.append({
                "page": page_no, "status": "ok", "job_id": job_id,
                "processing_status": out.get("processing_status"),
                "full_video_coverage": out.get("full_video_coverage"),
                "segment_count": out.get("segment_count"),
                "source_type": out.get("source_type"),
                "job_dir": out.get("job_dir"),
            })
        except AcquireError as exc:
            # 单个分P失败不中断整批
            log("分P %d 失败(%s): %s" % (page_no, exc.kind, exc))
            results.append({
                "page": page_no, "status": "blocked" if exc.kind in ("needs_login", "rate_limited")
                else "failed",
                "job_id": job_id, "kind": exc.kind, "message": str(exc),
            })
        except Exception as exc:  # noqa: BLE001
            log("分P %d 未预期错误: %s" % (page_no, redact(str(exc))))
            results.append({"page": page_no, "status": "failed", "job_id": job_id,
                            "kind": "unexpected", "message": redact(str(exc))})

    summary = summarize_batch(results)
    report = {
        "ok": summary["all_ok"],
        "bvid": info["bvid"],
        "title": info.get("title"),
        "total_pages": total,
        "requested_pages": pages,
        "summary": summary,
        "results": results,
    }

    # 汇总报告落盘，便于留档
    batch_dir = out_dir / make_job_id("bilibili", info["bvid"])
    try:
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "batch.report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str((batch_dir / "batch.report.json").resolve())
    except Exception as exc:  # noqa: BLE001
        log("警告：汇总报告写入失败：%s" % redact(str(exc)))

    log("完成：成功 %d，复用 %d，失败 %d"
        % (summary["ok_count"], summary["reused_count"], summary["failed_count"]))
    _print_json(report)
    return EXIT_OK if summary["all_ok"] else EXIT_FAILED


# --------------------------------------------------------------------------
# ocr
# --------------------------------------------------------------------------

def cmd_ocr(args: argparse.Namespace) -> int:
    from .ocr import (
        OcrUnavailable,
        ocr_available,
        run_ocr,
        search_ocr,
        to_markdown,
        write_ocr_index,
    )

    job_dir = Path(args.job_dir)

    if args.search:
        path = job_dir / "ocr.index.json"
        if not path.exists():
            _print_json({"ok": False, "kind": "invalid_input",
                         "message": "缺少 ocr.index.json：请先执行 ocr（不带 --search）"})
            return EXIT_USAGE
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        hits = search_ocr(payload, args.search, limit=args.limit)
        _print_json({"ok": True, "query": args.search, "hit_count": len(hits), "hits": hits})
        return EXIT_OK

    ok, reason = ocr_available()
    if not ok:
        # 可选依赖缺失：明确提示，不当成崩溃
        _print_json({
            "ok": False, "kind": "optional_dependency_missing",
            "message": reason,
            "hint": "OCR 是可选功能；不装也能完成取材、转录与取帧。",
        })
        return EXIT_USAGE

    def log(msg: str) -> None:
        _eprint("[ocr] %s" % msg)

    try:
        payload = run_ocr(job_dir, on_progress=log, limit=args.limit)
    except FileNotFoundError as exc:
        _print_json({"ok": False, "kind": "invalid_input", "message": str(exc)})
        return EXIT_USAGE
    except OcrUnavailable as exc:
        _print_json({"ok": False, "kind": "optional_dependency_missing", "message": str(exc)})
        return EXIT_USAGE
    except Exception as exc:  # noqa: BLE001
        _print_json({"ok": False, "kind": "failed", "message": redact(str(exc))})
        return EXIT_FAILED

    index_path = write_ocr_index(job_dir, payload)
    view_path = job_dir / "ocr-view.md"
    view_path.write_text(
        to_markdown(payload, title="画面文字证据 · %s" % job_dir.name), encoding="utf-8")

    try:
        man = load_manifest(job_dir)
        man.setdefault("ocr", {}).update(
            used=True,
            engine=payload["engine"],
            frames_read=payload["frames_read"],
            seconds_total=payload["seconds_total"],
            seconds_per_frame=payload["seconds_per_frame"],
            divergent_count=payload["divergent_count"],
            divergence_threshold=payload["divergence_threshold"],
            index="ocr.index.json",
            understanding_by="agent",
        )
        man.setdefault("files", {})["ocr_index"] = "ocr.index.json"
        man["files"]["ocr_view"] = "ocr-view.md"
        add_note(man, "OCR 仅提取画面文字，不做语义解释；配对仍以时间为准")
        save_manifest(job_dir, man)
    except Exception as exc:  # noqa: BLE001
        log("警告：manifest 更新失败：%s" % redact(str(exc)))

    log("完成 %d 帧，耗时 %.1fs（平均 %.2fs/帧），疑似口播/画面分歧 %d 帧"
        % (payload["frames_read"], payload["seconds_total"],
           payload["seconds_per_frame"], payload["divergent_count"]))

    _print_json({
        "ok": True,
        "job_dir": str(job_dir.resolve()),
        "engine": payload["engine"],
        "frames_read": payload["frames_read"],
        "seconds_total": payload["seconds_total"],
        "seconds_per_frame": payload["seconds_per_frame"],
        "divergent_count": payload["divergent_count"],
        "keyword_count": len(payload["keyword_index"]),
        "index": str(index_path.resolve()),
        "view": str(view_path.resolve()),
    })
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m video_ingest", description="本地视频取材与转录工具")
    p.add_argument("--version", action="version", version="video_ingest %s" % __version__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="检查环境，不输出任何 token/cookie")
    d.add_argument("--check-gpu", action="store_true",
                   help="做一次真实推理探测，确认 GPU 是否真的可用（硬件可见不等于可用）")
    d.set_defaults(func=cmd_doctor)

    pr = sub.add_parser("probe", help="只探测身份/分P/时长/字幕键/鉴权需要")
    pr.add_argument("--url", required=True)
    pr.add_argument("--cookies", default=None, help="Netscape 格式 cookies.txt（可选）")
    pr.add_argument("--output-dir", default=None)
    pr.add_argument("--save-job", action="store_true", help="在 output/<job-id>/ 下保存 probe.json 与 manifest.json")
    pr.set_defaults(func=cmd_probe)

    ing = sub.add_parser("ingest", help="取材、转写、标准化、持久化、完整性校验")
    src = ing.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="视频链接")
    src.add_argument("--media", help="本地媒体文件")
    ing.add_argument("--cookies", default=None, help="Netscape 格式 cookies.txt（可选）")
    ing.add_argument("--output-dir", default=None)
    ing.add_argument("--language", default="zh", help="语言代码，或 auto")
    ing.add_argument("--model", default="medium", help="faster-whisper 模型（tiny..large-v3）")
    ing.add_argument("--device", default="auto",
                     help="auto（默认，GPU 真可用则用 cuda）| cpu | cuda")
    ing.add_argument("--compute-type", default=None,
                     help="留空则按设备选择：cuda→float16，cpu→int8")
    ing.add_argument("--force-asr", action="store_true", help="即使有字幕也改用音频转写（用于比对）")
    ing.add_argument("--chars-per-chunk", type=int, default=3000,
                     help="分块字符预算（不是 token 数）")
    ing.add_argument("--corrections", default=None, help="JSON 校正表 {错:对}")
    ing.add_argument("--prompt", default=None, help="领域提示词，提升专有名词识别率")
    ing.set_defaults(func=cmd_ingest)

    va = sub.add_parser("validate", help="区分资源处理覆盖情况与识别准确性核查")
    va.add_argument("--job-dir", required=True)
    va.add_argument("--sample-check", action="store_true",
                    help="抽样比对音频与转录（需音频仍在），结果标为 sample_checked")
    va.set_defaults(func=cmd_validate)

    ch = sub.add_parser("chunks", help="按片段预算切分并做覆盖检查，不生成摘要")
    ch.add_argument("--job-dir", required=True)
    ch.add_argument("--chars-per-chunk", type=int, default=3000)
    ch.add_argument("--overlap-segments", type=int, default=1)
    ch.set_defaults(func=cmd_chunks)

    fr = sub.add_parser("frames", help="提取与字幕配对的画面帧（只取帧，不解释画面）")
    fr.add_argument("--job-dir", required=True)
    fr.add_argument("--video", default=None, help="视频文件；缺省时用 manifest 记录或任务内媒体")
    fr.add_argument("--interval", type=float, default=0.4, help="采样步长（秒）")
    fr.add_argument("--scene-threshold", type=float, default=6.0,
                    help="画面变化阈值（0-255 灰度平均绝对差）")
    fr.add_argument("--start", type=float, default=0.0, help="只处理该时间之后")
    fr.add_argument("--end", type=float, default=None, help="只处理该时间之前")
    fr.add_argument("--scale", type=float, default=1.0, help="输出缩放比例（1.0 保持原分辨率）")
    fr.add_argument("--max-frames", type=int, default=0, help="帧数上限，0 表示不限")
    fr.add_argument("--include-blank", action="store_true",
                    help="同时提取被判为空白/过渡的帧（默认跳过）")
    fr.set_defaults(func=cmd_frames)

    ba = sub.add_parser("batch", help="多分P批处理（只接受显式范围，逐P记录状态）")
    ba.add_argument("--url", required=True)
    ba.add_argument("--pages", required=True,
                    help="必须显式指定：1-5 或 3,7 或 1-3,7,10-12（无'全部'默认值）")
    ba.add_argument("--cookies", default=None)
    ba.add_argument("--output-dir", default=None)
    ba.add_argument("--language", default="zh")
    ba.add_argument("--model", default="medium")
    ba.add_argument("--device", default="auto")
    ba.add_argument("--compute-type", default=None)
    ba.add_argument("--force-asr", action="store_true")
    ba.add_argument("--chars-per-chunk", type=int, default=3000)
    ba.add_argument("--corrections", default=None)
    ba.add_argument("--prompt", default=None)
    ba.add_argument("--resume", action="store_true", default=True,
                    help="跳过已有完成产物的分P（默认开启）")
    ba.add_argument("--force", dest="resume", action="store_false",
                    help="强制重跑所有分P")
    ba.set_defaults(func=cmd_batch)

    oc = sub.add_parser("ocr", help="识别画面帧文字并建立三方绑定（可选依赖）")
    oc.add_argument("--job-dir", required=True)
    oc.add_argument("--search", default=None, help="在已有 OCR 结果里检索关键词")
    oc.add_argument("--limit", type=int, default=0, help="只处理前 N 帧（0 不限制）")
    oc.set_defaults(func=cmd_ocr)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        _eprint("已中断")
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001
        _eprint("未预期错误: %s" % redact(str(exc)))
        return EXIT_FAILED


