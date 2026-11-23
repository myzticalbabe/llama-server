#!/usr/bin/env python3
# coding: utf-8
"""pmb_wsl.py — reach the PMB memory from WSL without the daemon.

WHY THIS EXISTS (2026-09-30)
----------------------------
The `pmb` MCP server in ~/.config/kilo/kilo.jsonc is type:remote at
http://127.0.0.1:8765/mcp and points at the WARM DAEMON. From WSL that daemon
is unreachable: the network is NAT, there is no WSLInterop, and
`curl http://127.0.0.1:8765/mcp` returns HTTP 000. So the memory tool is
"present" in the config but dead in practice from this shell.

The daemon is not the memory. The memory is a sqlite store plus a BM25 index
under PMB_HOME/workspaces/<id>/. The `pmb` package itself is pure Python and
is installed in the Windows venv at ~/.pmb-venv. This helper loads THAT
package under the Linux interpreter by ordering sys.path so the Linux
site-packages wins for the compiled dependencies (numpy, pydantic_core, rich)
and the Windows site-packages supplies the pure-Python ones (cyclopts, pmb).

The two env vars that matter:
  PMB_HOME       where the workspaces live. On this box: /mnt/c/Users/clint/.pmb
                 NOT `PMB_WORKSPACE_DIR` -- that one silently selects an empty
                 "root" workspace and recall reports 0 events.
  PMB_VENV       the Windows venv holding the pmb package.

KNOWN LIMITATION
----------------
lancedb's native extension is a Windows .pyd and will not import under Linux,
so the vector half of hybrid recall is unavailable here and recall degrades to
BM25-only (`source: cold (bm25-only)`). Every sqlite-backed surface (goals,
events, lessons, mcp_calls, session briefs) reads exactly.

USAGE
-----
    python3 pmb_wsl.py goals
    python3 pmb_wsl.py recall "why did the chain job fail"
    python3 pmb_wsl.py note "jotted from WSL"
    python3 pmb_wsl.py --where          # print resolved paths, then exit

Any argument is passed straight through to the real `pmb` CLI, so every
`pmb` subcommand works. `--where` is the only flag this wrapper owns.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_PMB_HOME = "/mnt/c/Users/clint/.pmb"
DEFAULT_VENV = "/mnt/c/Users/clint/.pmb-venv"


def _linux_site() -> str:
    """The running interpreter's own site-packages (its compiled wheels win)."""
    import sysconfig

    for key in ("purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p:
            return p
    return sysconfig.get_paths()["stdlib"]


def _venv_site(venv: Path) -> str | None:
    """site-packages inside a venv, whichever layout it uses."""
    for rel in ("Lib/site-packages",                    # Windows venv
                "lib/site-packages",
                *[f"lib/python3.{n}/site-packages" for n in range(20, 10, -1)]):
        cand = venv / rel
        if (cand / "pmb").is_dir():
            return str(cand)
    return None


def _resolve() -> tuple[str, str, str]:
    pmb_home = os.environ.get("PMB_HOME") or DEFAULT_PMB_HOME
    venv = Path(os.environ.get("PMB_VENV") or DEFAULT_VENV)
    win_site = _venv_site(venv)
    if win_site is None:
        die("no pmb package found under %s — set PMB_VENV" % venv)
    return pmb_home, win_site, _linux_site()


def die(msg: str) -> "NoReturn":
    print("pmb_wsl: %s" % msg, file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    argv = sys.argv[1:]
    pmb_home, win_site, linux_site = _resolve()

    # PMB_HOME must be exported for the whole process, not just this call: the
    # engine resolves the workspace from it lazily, deep inside recall().
    os.environ["PMB_HOME"] = pmb_home

    if argv and argv[0] == "--where":
        print("interpreter : %s" % sys.executable)
        print("linux site  : %s" % linux_site)
        print("venv site   : %s" % win_site)
        print("PMB_HOME    : %s" % pmb_home)
        ws = Path(pmb_home) / "current_workspace"
        if ws.is_file():
            print("workspace   : %s" % ws.read_text(encoding="utf-8").strip())
        print("note        : os.name=%s — lancedb vector path %s"
              % (os.name, "available" if os.name == "nt" else "unavailable"))
        return 0

    # Order matters. Linux first so its compiled wheels (numpy, pydantic_core)
    # are the ones imported; the Windows venv last so pure-Python packages that
    # only exist there (cyclopts, pmb itself) are still found.
    sys.path.insert(0, linux_site)
    if win_site not in sys.path:
        sys.path.append(win_site)

    try:
        from pmb.cli import app
    except Exception as exc:  # noqa: BLE001 - surface the real reason, not a bare trace
        die("could not import pmb.cli (%s: %s)" % (type(exc).__name__, exc))

    try:
        app()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
