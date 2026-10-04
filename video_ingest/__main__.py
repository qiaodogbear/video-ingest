"""`python -m video_ingest` 的入口。

实现全部放在 :mod:`video_ingest.cli`；这里只做薄转发，并保留对历史
导入路径（``video_ingest.__main__``）的兼容，避免破坏已有调用与测试。
"""

from __future__ import annotations

from .cli import (  # noqa: F401  (re-exported for backwards compatibility)
    EXIT_BLOCKED,
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    _combine_subtitle_verdict,
    _eprint,
    _print_json,
    build_parser,
    cmd_batch,
    cmd_chunks,
    cmd_doctor,
    cmd_frames,
    cmd_ingest,
    cmd_ocr,
    cmd_probe,
    cmd_validate,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
