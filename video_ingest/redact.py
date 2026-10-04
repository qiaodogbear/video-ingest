"""日志与错误信息的脱敏。

bootstrap §7.2：不要在终端、摘要、模型提示词、错误回传中打印 Cookie / Authorization。
bootstrap §11：敏感 URL 查询参数需要脱敏。
"""

from __future__ import annotations

import re

_SENSITIVE_KEY = re.compile(
    r"(?i)\b(cookie|authorization|auth|token|access_token|refresh_token|sessdata|"
    r"bili_jct|buvid3|dedeuserid|api[_-]?key|secret|password|passwd|pwd)\b"
)

# key=value 形式的敏感参数
_KV = re.compile(
    r"(?i)\b(SESSDATA|bili_jct|buvid3|buvid4|DedeUserID|DedeUserID__ckMd5|sid|"
    r"access_token|refresh_token|api_key|apikey|token|password|pwd|sign|upsig)"
    r"\s*[=:]\s*([^\s;&\"']+)"
)

# 签名类查询参数（B 站 CDN 常见），脱敏后仍可复现来源
_SIGNED_QUERY = re.compile(
    r"(?i)([?&])(upsig|e|deadline|oi|uipk|trid|buvid|bvc|qn_dyeid|og|os|platform|nbs)=([^&\s]*)"
)


def redact(text: str) -> str:
    """对任意文本做脱敏，用于日志、异常与 manifest 错误字段。"""
    if not text:
        return text
    return _KV.sub(lambda m: "%s=***" % m.group(1), text)


def redact_url(url: str) -> str:
    """URL 脱敏：保留主机与路径（可定位），抹掉凭证与签名查询值。"""
    if not url:
        return url
    out = _KV.sub(lambda m: "%s=***" % m.group(1), url)
    return _SIGNED_QUERY.sub(lambda m: "%s%s=***" % (m.group(1), m.group(2)), out)


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: ("***" if _SENSITIVE_KEY.search(k) else v) for k, v in headers.items()}
