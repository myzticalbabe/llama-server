"""Wrapper that makes scrapling's MCP server run on the installed mcp 1.x.

scrapling 0.5 was written against mcp 2.x (which exposes MCPServer, Icon,
CacheHint and the mcp.server.mcpserver subpackage). This venv ships mcp 1.10.1,
where those names live elsewhere or not at all. Rather than upgrading mcp
(which would risk pams/codegraph/supervisor that are written against 1.x),
this wrapper rewrites the few offending imports in scrapling's server.py on
disk (idempotently) and then delegates to scrapling's own CLI entry point.

If scrapling is ever reinstalled, the rewrite is re-applied automatically on
the next launch.
"""
from __future__ import annotations

import pathlib
import sys

# mcp 1.10.1 does `from pydantic._internal._typing_extra import
# eval_type_backport`; pydantic 2.14 deleted that name, so importing mcp (and
# therefore scrapling's own server module) raises ImportError before the
# wrapper's patch logic ever runs. The vendored backfill must be imported
# BEFORE any mcp import. Same pattern as barta_scrape_mcp.py.
try:
    import compat_pydantic214  # noqa: F401 -- vendored pydantic-2.14 backfill
except ImportError:
    _aztools = "C:/Azazael/azazael/tools"
    if _aztools not in sys.path:
        sys.path.insert(0, _aztools)
    import compat_pydantic214  # noqa: F401

SERVER_PY = pathlib.Path(
    "C:/Azazael/.venvs/py315t/Lib/site-packages/scrapling/core/ai/server.py"
)

_PATCHES = [
    ("from mcp.server import MCPServer",
     "from mcp.server.fastmcp import FastMCP as MCPServer"),
    ("from mcp.server.mcpserver import Image",
     "from mcp.server.fastmcp import Image"),
    ("from mcp.server.mcpserver.exceptions import ToolError",
     "from mcp.server.fastmcp.exceptions import ToolError"),
    ("from mcp.server.caching import CacheHint",
     "class CacheHint:\n    def __init__(self, ttl_ms=0, scope='public'):\n        self.ttl_ms = ttl_ms\n        self.scope = scope"),
    ("from mcp.types import Icon, ImageContent, TextContent, ToolAnnotations",
     "from mcp.types import ImageContent, TextContent, ToolAnnotations\n\n"
     "class Icon:\n    def __init__(self, src='', mime_type=None, **kw):\n        self.src = src\n        self.mime_type = mime_type"),
]


def _ensure_patched() -> None:
    if not SERVER_PY.exists():
        return
    text = SERVER_PY.read_text(encoding="utf-8")
    original = text
    for old, new in _PATCHES:
        if old in text:
            text = text.replace(old, new)
    if text != original:
        SERVER_PY.write_text(text, encoding="utf-8")
        # invalidate stale bytecode
        pyc = SERVER_PY.parent / "__pycache__" / "server.cpython-315.pyc"
        try:
            pyc.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> int:
    _ensure_patched()
    # Delegate to scrapling's own mcp command.
    from scrapling.cli import mcp as scrapling_mcp
    # scrapling_mcp is a click Command; invoke it.
    return scrapling_mcp()


if __name__ == "__main__":
    raise SystemExit(main())
