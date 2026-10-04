"""取材层：URL 规范化、B 站身份解析、字幕枚举与下载、音频获取。

不依赖系统 FFmpeg；不使用 shell 字符串拼接（bootstrap §14）。
"""

from __future__ import annotations

import http.cookiejar
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .redact import redact, redact_url

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

BILI_API = "https://api.bilibili.com"

# 分享追踪参数，去除但保留 p（bootstrap §7.1）
TRACKING_PARAMS = {
    "trackid", "spm_id_from", "vd_source", "from_source", "seid", "from_spmid",
    "share_source", "share_medium", "share_plat", "share_session_id", "share_tag",
    "bbid", "ts", "unique_k", "buvid", "up_id", "tab", "broadcast_type",
    "is_room_feed", "live_from", "visit_id", "msource", "w_rid", "wts",
}


class AcquireError(RuntimeError):
    """取材失败的统一异常，kind 对应 manifest.SUBTITLE_FAILURE_KINDS 等分类。"""

    def __init__(self, kind: str, message: str):
        super().__init__(redact(message))
        self.kind = kind


# --------------------------------------------------------------------------
# URL 规范化
# --------------------------------------------------------------------------

_BV_RE = re.compile(r"(BV[0-9A-Za-z]{10})")
_AV_RE = re.compile(r"av(\d+)", re.IGNORECASE)


def normalize_url(url: str) -> dict[str, Any]:
    """去追踪参数、保留分P、抽取身份。短链重定向后需重新校验域名。"""
    if not url or not url.strip():
        raise AcquireError("invalid_input", "URL 为空")

    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        raise AcquireError("invalid_input", "只接受 http(s) URL，收到: %s" % parsed.scheme)

    host = (parsed.hostname or "").lower()
    if host not in {"www.bilibili.com", "bilibili.com", "m.bilibili.com", "b23.tv"}:
        raise AcquireError("invalid_input", "非受支持站点: %s" % host)

    # 拒绝私网 / localhost
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or host.endswith(".local"):
        raise AcquireError("invalid_input", "拒绝访问本地地址")

    kept = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS
    ]
    query = urllib.parse.urlencode(kept)

    path = parsed.path
    bvid = None
    m = _BV_RE.search(path) or _BV_RE.search(url)
    if m:
        bvid = m.group(1)
    else:
        m2 = _AV_RE.search(path)
        if m2:
            bvid = "av" + m2.group(1)

    page = None
    for k, v in kept:
        if k.lower() == "p":
            try:
                page = int(v)
            except ValueError as exc:
                raise AcquireError("invalid_input", "分P参数 p 不是整数: %s" % v) from exc

    canonical = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path, query, "")
    )
    return {
        "canonical_url": canonical,
        "host": host,
        "bvid": bvid,
        "page": page,
        "query": query,
        "is_shortlink": host == "b23.tv",
    }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def build_opener(cookies_file: str | None = None) -> urllib.request.OpenerDirector:
    handlers: list[Any] = []
    if cookies_file:
        jar = http.cookiejar.MozillaCookieJar(cookies_file)
        try:
            jar.load(ignore_discard=True, ignore_expires=True)
        except Exception as exc:  # noqa: BLE001
            raise AcquireError("needs_login", "cookie 文件无法解析: %s" % redact(str(exc))) from exc
        handlers.append(urllib.request.HTTPCookieProcessor(jar))
    opener = urllib.request.build_opener(*handlers)
    opener.addheaders = [
        ("User-Agent", USER_AGENT),
        ("Referer", "https://www.bilibili.com/"),
        ("Accept", "application/json, text/plain, */*"),
        ("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8"),
    ]
    return opener


def _classify_http_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            return "needs_login"
        if exc.code == 404:
            return "content_unavailable"
        if exc.code == 429:
            return "rate_limited"
        return "network_error"
    return "network_error"


def api_get(
    opener: urllib.request.OpenerDirector,
    path: str,
    params: dict[str, Any],
    timeout: float = 25.0,
) -> dict[str, Any]:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = "%s%s?%s" % (BILI_API, path, query)
    req = urllib.request.Request(url, method="GET")
    try:
        with opener.open(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001
        raise AcquireError(_classify_http_error(exc), "请求 %s 失败: %s" % (path, redact_url(url))) from exc
    if not isinstance(payload, dict):
        raise AcquireError("api_changed", "接口 %s 返回非 JSON 对象" % path)
    return payload


# --------------------------------------------------------------------------
# B 站身份
# --------------------------------------------------------------------------

def fetch_video_info(
    opener: urllib.request.OpenerDirector, bvid: str
) -> dict[str, Any]:
    payload = api_get(opener, "/x/web-interface/view", {"bvid": bvid})
    code = payload.get("code")
    if code == -404:
        raise AcquireError("content_unavailable", "视频不存在或已删除: %s" % bvid)
    if code == -403:
        raise AcquireError("needs_login", "视频访问受限(可能需登录/会员): %s" % bvid)
    if code != 0:
        raise AcquireError("api_changed", "view 接口返回 code=%s msg=%s" % (code, payload.get("message")))
    data = payload.get("data") or {}
    pages = data.get("pages") or []
    return {
        "aid": data.get("aid"),
        "bvid": data.get("bvid"),
        "cid": data.get("cid"),
        "title": data.get("title"),
        "desc": data.get("desc") or "",
        "duration": data.get("duration"),
        "uploader": (data.get("owner") or {}).get("name"),
        "pubdate": data.get("pubdate"),
        "pages": [
            {
                "page": p.get("page"),
                "cid": p.get("cid"),
                "part": p.get("part"),
                "duration": p.get("duration"),
            }
            for p in pages
        ],
    }


def resolve_page(info: dict[str, Any], page: int | None) -> dict[str, Any]:
    """确定要处理的单个分P。默认只处理 p=1，不悄悄下载全集。"""
    pages = info.get("pages") or []
    if not pages:
        return {"page": 1, "cid": info.get("cid"), "duration": info.get("duration"), "part": info.get("title")}
    wanted = page or 1
    for p in pages:
        if p.get("page") == wanted:
            return p
    raise AcquireError("invalid_input", "分P p=%s 不存在，该视频共 %d 个分P" % (wanted, len(pages)))


# --------------------------------------------------------------------------
# 字幕探测（三态）
# --------------------------------------------------------------------------

def probe_player_subtitles(
    opener: urllib.request.OpenerDirector, bvid: str, cid: int
) -> dict[str, Any]:
    """查询播放器接口的字幕信息。

    返回 probe_state：
      found          —— 平台侧给出可用字幕键
      empty          —— 接口正常应答，但字幕列表为空
      needs_auth_check —— 无法区分"确实没有"与"需登录才可见"

    实测（2026-10）：未登录时新旧两版 player 接口都会返回空 subtitles，
    即使该视频确实存在 CC 字幕。因此空列表不能单独证明"没有字幕"。
    """
    result: dict[str, Any] = {
        "probe_state": "needs_auth_check",
        "failure_kind": "not_probed",
        "language_keys": [],
        "endpoints": {},
        "auth_used": False,
    }

    last_error: AcquireError | None = None
    for name, path in (
        ("player_wbi_v2", "/x/player/wbi/v2"),
        ("player_v2", "/x/player/v2"),
    ):
        try:
            payload = api_get(opener, path, {"bvid": bvid, "cid": cid})
        except AcquireError as exc:
            result["endpoints"][name] = {"ok": False, "kind": exc.kind, "message": str(exc)}
            last_error = exc
            continue

        code = payload.get("code")
        data = payload.get("data") or {}
        sub = data.get("subtitle") or {}
        subs = sub.get("subtitles") or []
        keys = []
        for item in subs:
            if not isinstance(item, dict):
                continue
            keys.append(
                {
                    "lan": item.get("lan"),
                    "lan_doc": item.get("lan_doc"),
                    "subtitle_url": item.get("subtitle_url") or item.get("subtitleUrl"),
                    "is_ai": _looks_auto(item),
                }
            )
        result["endpoints"][name] = {"ok": code == 0, "code": code, "count": len(keys)}
        if code == 0 and keys:
            result.update(
                probe_state="found",
                failure_kind="not_probed",
                language_keys=keys,
            )
            return result
        if code == -403:
            last_error = AcquireError("needs_login", "%s 需要登录" % name)

    if last_error is not None and all(
        not (v.get("ok") if isinstance(v, dict) else False)
        for v in result["endpoints"].values()
    ):
        result["failure_kind"] = last_error.kind
        result["probe_state"] = "needs_auth_check"
    else:
        # 接口正常应答但为空。无 cookie 时无法排除"需登录"。
        result["failure_kind"] = "no_native_subtitle"
        result["probe_state"] = "empty"
    return result


def _looks_auto(item: dict[str, Any]) -> bool:
    lan = (item.get("lan") or "").lower()
    if lan.startswith("ai-"):
        return True
    doc = (item.get("lan_doc") or "")
    return "自动" in doc or "AI" in doc.upper()


# --------------------------------------------------------------------------
# yt-dlp 探测（交叉验证渠道）
# --------------------------------------------------------------------------

def yt_dlp_version(python_exe: str | None = None) -> str | None:
    """读取 yt-dlp 版本，用于运行报告。

    注意：yt_dlp 模块本身没有 __version__；优先用包元数据，
    其次退回 yt_dlp.version.__version__，最后才起子进程。
    """
    try:
        from importlib.metadata import version as _dist_version
        return _dist_version("yt-dlp")
    except Exception:  # noqa: BLE001
        pass
    try:
        from yt_dlp.version import __version__ as v  # type: ignore
        return v
    except Exception:  # noqa: BLE001
        pass
    py = python_exe or sys.executable
    try:
        out = subprocess.run(
            [py, "-m", "yt_dlp", "--version"],
            capture_output=True, text=True, timeout=60,
        )
        return (out.stdout or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def probe_ytdlp_subtitles(
    url: str, python_exe: str | None = None, cookies_file: str | None = None, timeout: float = 120.0
) -> dict[str, Any]:
    """用 yt-dlp 枚举字幕，作为第二个独立渠道。

    注意：B 站提取器可能把平台自动字幕放进常规 subtitles，也可能只放 danmaku。
    danmaku 是弹幕，不是字幕，必须排除（bootstrap §6）。
    """
    py = python_exe or sys.executable
    cmd = [py, "-m", "yt_dlp", "--ignore-config", "--no-playlist", "--skip-download",
           "--list-subs", "--no-warnings"]
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    cmd.append(url)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"ok": False, "kind": "network_error", "keys": [], "raw": "yt-dlp 超时"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "kind": "network_error", "keys": [], "raw": redact(str(exc))}

    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    keys = _parse_list_subs(text)
    real = [k for k in keys if k.lower() != "danmaku"]
    return {
        "ok": proc.returncode == 0,
        "kind": "ok" if real else "no_native_subtitle",
        "keys": real,
        "danmaku_only": bool(keys) and not real,
        "raw_tail": "\n".join(text.strip().splitlines()[-6:]),
    }


def _parse_list_subs(text: str) -> list[str]:
    """从 --list-subs 的人类可读输出里抽取语言键。

    注意匹配顺序：表头行本身形如 "[info] Available subtitles for ..."，
    因此必须**先**判断表头，再判断"以 [ 开头的日志行"，否则永远进不了表格。
    只认 "Language / Formats" 之后、缩进的两列表格行。
    """
    keys: list[str] = []
    in_table = False
    saw_header = False
    for line in text.splitlines():
        stripped = line.strip()

        # 表头必须在通用 "[...]" 过滤之前判断
        if "Available subtitles" in stripped or "Available automatic captions" in stripped:
            in_table = True
            saw_header = False
            continue
        if not in_table:
            continue

        if not stripped:
            continue
        if stripped.startswith("Language") and "Formats" in stripped:
            saw_header = True
            continue
        if stripped.startswith("["):
            # 表格结束后又出现日志行
            in_table = False
            continue
        if not saw_header:
            continue
        if stripped.startswith("-"):
            continue

        parts = stripped.split()
        if not parts:
            continue
        token = parts[0]
        # 表格形如 "<lang>  <formats>"
        if len(parts) >= 2 and token:
            keys.append(token)
    return keys
