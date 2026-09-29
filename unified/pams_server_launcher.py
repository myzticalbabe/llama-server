"""pams_server_launcher.py — start pams-server with its data on G:.

WHY THIS EXISTS
---------------
`persistent_ai_memory` resolves its databases through TWO different functions
that disagree:

  * `utils.get_memory_data_dir()`  honours $AI_MEMORY_DATA_DIR
  * `ai_memory_core.get_base_path()` returns `Path(__file__).resolve().parent`
    unconditionally -- the site-packages install directory

`mcp_server.py` does both:

    BASE_PATH      = get_base_path()                              # site-packages
    memory_system  = AIMemorySystem(data_dir=get_memory_data_dir())  # env, correct
    port_manager   = PortManager(memory_data_path=get_memory_data_dir())  # env

and `AIMemorySystem.__init__` accepts `data_dir` and then never reads it --
it derives every path from `get_base_path()`. So the env var is honoured by
the port manager (the port file lands on G:) and silently discarded by the
databases (which land in site-packages). That split is why the corpus
fragmented into 3639 memories in site-packages and 60 on G:.

`pip install --force-reinstall` or a Python upgrade would have deleted the
3639-memory database, because it sits inside the install directory.

WHAT THIS ALSO DOES
-------------------
Marks `mcp_server_port.json` inactive on exit. The port file outlives a hard
kill and keeps claiming `"status": "active"`, so a dead server reads as a live
one -- on 2026-09-29 it asserted an active port 21434 while nothing was
listening and the pid was gone. A graceful exit now corrects the record; a
SIGKILL still cannot be caught, which is why consumers should still probe the
port rather than trust the file.

Env:
    PAMS_ROOT  canonical project root. Default G:\\source-of-truth\\
               persistent-ai-memory
"""
import atexit
import json
import os
import signal
import sys
import time
from pathlib import Path

PAMS_ROOT = Path(os.environ.get(
    "PAMS_ROOT", r"G:\source-of-truth\persistent-ai-memory"))
PAMS_DATA = PAMS_ROOT / "memory_data"
PORT_FILE = PAMS_DATA / "mcp_server_port.json"

PAMS_DATA.mkdir(parents=True, exist_ok=True)
# Both the port manager and anything else reading the env var get the right
# answer; the patch below covers the resolver that ignores it.
os.environ["AI_MEMORY_DATA_DIR"] = str(PAMS_DATA)
os.environ["PAMS_DATA_DIR"] = str(PAMS_DATA)


def _mark_port_file_inactive() -> None:
    """Downgrade the port file so a dead server stops reading as a live one."""
    try:
        data = json.loads(PORT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if data.get("status") == "active":
        data["status"] = "inactive"
        data["stopped_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        data["stopped_pid"] = os.getpid()
        try:
            PORT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass


def _install_exit_hooks() -> None:
    atexit.register(_mark_port_file_inactive)

    def _on_signal(signum, _frame):
        _mark_port_file_inactive()
        # Re-raise with the default disposition so the exit status is honest.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError, AttributeError):
            pass  # not the main thread, or unsupported on this platform


_install_exit_hooks()

# --- patch the resolver that ignores the env var, BEFORE the import ---------
# ai_memory_core.get_base_path is looked up inside AIMemorySystem.__init__ at
# call time, so rebinding it on the module object is enough. It must happen
# before `mcp_server` is imported, because that module builds its databases at
# import time.
import ai_memory_core  # noqa: E402


def _base_path():
    return str(PAMS_ROOT)


ai_memory_core.get_base_path = _base_path

# Import the server by its FULL package path. site-packages also contains a
# top-level `mcp_server.py` (plus decoy `utils.py`, `port_manager.py`,
# `database_maintenance.py`) used by `from ai_memory_core import ...`-style
# internal imports. A bare `import mcp_server` silently binds that decoy --
# which has no `run_main` -- so the explicit path is required, not stylistic.
from persistent_ai_memory import mcp_server  # noqa: E402

# mcp_server computes its own BASE_PATH from its own get_base_path at import.
# Rebind it so logs/keys/config also follow the canonical root.
mcp_server.BASE_PATH = PAMS_ROOT

if __name__ == "__main__":
    sys.exit(mcp_server.run_main())
