"""pams_patch_log_tool_call.py — re-apply and verify the PAMS tool-call logging fix.

WHY THIS FILE EXISTS
--------------------
`persistent_ai_memory` has no source repo on this machine. `G:\\source-of-truth\\
persistent-ai-memory` holds only data (config/, keys/, logs/, memory_data/), so
site-packages is the SOLE copy of the code. Any

    pip install --force-reinstall persistent-ai_memory

reverts every fix, silently. The launcher docstring in pams_server_launcher.py
already records a 3,639-memory database being destroyed this way.

THE BUG
-------
mcp_server.py calls

    self.memory_system.log_tool_call(..., source=source)

but AIMemorySystem.log_tool_call had no `source` parameter, so every call raised
`TypeError` inside a fire-and-forget asyncio task. The task's exception was
never retrieved, so nothing surfaced. Result: mcp_tool_calls.db held 0 rows
forever, and every structured usage query (get_tool_information,
reflect_on_tool_usage, get_ai_insights, export_all_tool_calls) returned empty
while reporting status "healthy".

Usage:
    python pams_patch_log_tool_call.py            # patch if needed, then verify
    python pams_patch_log_tool_call.py --check    # verify only, change nothing
    python pams_patch_log_tool_call.py --status   # show current state, no verify
"""

import argparse
import re
import sys
from pathlib import Path

PKG = Path(sys.prefix) / "Lib" / "site-packages" / "persistent_ai_memory"

CORE = PKG / "ai_memory_core.py"
SERVER = PKG / "mcp_server.py"

HELPER = '''

def _report_background_failure(task):
    """Surface exceptions from fire-and-forget tasks.

    create_task() only fails synchronously on bad args; anything raised once
    the coroutine is scheduled lands on the task. Combined with a
    discard-only done_callback, that exception is never retrieved, so it
    vanishes (and the launcher's stderr=DEVNULL hides the "Task exception
    was never retrieved" notice). That is how a signature mismatch in
    log_tool_call went unnoticed while every structured usage query
    silently returned zero.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Background task failed (%s): %s",
            getattr(task, "__name__", task), exc, exc_info=exc
        )

'''

BROKEN_FACADE = re.compile(
    r"( +)async def log_tool_call\(self, client_id: str, tool_name: str,"
    r" parameters: Dict = None,\n"
    r" +execution_time_ms: float = None, status: str = \"success\",\n"
    r" +result: Any = None, error_message: str = None\) -> str:\n"
    r"( +)\"\"\"Log an MCP tool call for AI self-reflection\"\"\"\n"
    r"\n?"
    r" +return await self\.mcp_db\.log_tool_call\(\n"
    r" +client_id, tool_name, parameters, execution_time_ms,\s*\n"
    r" +status, result, error_message\n"
    r" +\)"
)

FIXED_FACADE_RE = re.compile(
    r"async def log_tool_call\(self, client_id: str, tool_name: str,"
    r" parameters: Dict = None,\s*\n"
    r" +execution_time_ms: float = None, status: str = \"success\",\s*\n"
    r" +result: Any = None, error_message: str = None,\s*\n"
    r" +source: str = \"direct\"\) -> str:"
)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def _unpaired_discard_lines(text: str) -> list[str]:
    """Discard-only done_callbacks that have no failure reporter for them.

    A discard line is PAIRED only when the very next line registers
    _report_background_failure for the SAME task variable. Two ways to get
    this wrong, both of which report a green check while the task is still
    unguarded: counting discard lines outright (every correctly guarded task
    then looks unpaired), and accepting a guard belonging to a different task
    (a misaligned paste reads as fixed).
    """
    unpaired: list[str] = []
    lines = text.splitlines()
    discard_re = re.compile(
        r"^(\s*)(\S+)\.add_done_callback\((self|srv)\._background_tasks\.discard\)\s*$"
    )
    for i, line in enumerate(lines):
        m = discard_re.match(line)
        if not m:
            continue
        task = m.group(2)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if f"{task}.add_done_callback(_report_background_failure)" not in nxt:
            unpaired.append(line.strip())
    return unpaired


def patch_core() -> bool:
    text = _read(CORE)
    if FIXED_FACADE_RE.search(text):
        return False
    new, n = BROKEN_FACADE.subn(
        lambda m: (
            f"{m.group(1)}async def log_tool_call(self, client_id: str, tool_name: str,"
            " parameters: Dict = None,\n"
            f"{m.group(1)}                      execution_time_ms: float = None,"
            ' status: str = "success",\n'
            f"{m.group(1)}                      result: Any = None,"
            " error_message: str = None,\n"
            f'{m.group(1)}                      source: str = "direct") -> str:\n'
            f'{m.group(2)}"""Log an MCP tool call for AI self-reflection"""\n'
            f"\n{m.group(1)}    return await self.mcp_db.log_tool_call(\n"
            f"{m.group(1)}        client_id, tool_name, parameters, execution_time_ms,\n"
            f"{m.group(1)}        status, result, error_message, source\n"
            f"{m.group(1)}    )"
        ),
        text,
    )
    if n != 1:
        raise SystemExit(
            f"refusing to patch: expected exactly 1 facade match, found {n}.\n"
            "The upstream layout changed; patch by hand."
        )
    _write(CORE, new)
    return True


def patch_server() -> bool:
    text = _read(SERVER)
    changed = False
    if "_report_background_failure" not in text:
        anchor = "logger = logging.getLogger(__name__)\n"
        if text.count(anchor) != 1:
            raise SystemExit(
                "refusing to patch: expected exactly 1 logger anchor.\n"
                "The upstream layout changed; patch by hand."
            )
        text = text.replace(anchor, anchor + HELPER, 1)
        changed = True
    guarded = text.count("add_done_callback(_report_background_failure)")
    unpaired = _unpaired_discard_lines(text)
    if len(unpaired) > guarded:
        text = re.sub(
            r"^(\s*)(\S+)\.add_done_callback\((self|srv)\._background_tasks\.discard\)\s*$",
            lambda m: (
                f"{m.group(0)}\n"
                f"{m.group(1)}{m.group(2)}.add_done_callback(_report_background_failure)"
            ),
            text,
            flags=re.M,
        )
        changed = True
    if changed:
        _write(SERVER, text)
    return changed


def check() -> bool:
    core = _read(CORE)
    server = _read(SERVER)
    ok = True
    if not FIXED_FACADE_RE.search(core):
        print("  FAIL  ai_memory_core.log_tool_call missing `source` param")
        ok = False
    else:
        print("  ok    ai_memory_core.log_tool_call accepts `source`")
    if "_report_background_failure" not in server:
        print("  FAIL  mcp_server missing _report_background_failure")
        ok = False
    else:
        n = server.count("add_done_callback(_report_background_failure)")
        print(f"  ok    mcp_server guards {n} background task(s)")
    unpaired = _unpaired_discard_lines(server)
    if unpaired:
        print(f"  FAIL  {len(unpaired)} task(s) still discard-only")
        ok = False
    return ok


def runtime_check() -> bool:
    """Import the real module and prove the caller/callee signatures agree."""
    import asyncio
    from persistent_ai_memory.ai_memory_core import AIMemorySystem

    params = list(__import__("inspect").signature(
        AIMemorySystem.log_tool_call).parameters)
    if "source" not in params:
        print("  FAIL  runtime signature still lacks `source`")
        return False
    print(f"  ok    runtime signature: {', '.join(params)}")

    class Capture:
        async def log_tool_call(self, *a, **kw):
            self.args = a
            return "verify-ok"

    class Fake(AIMemorySystem):
        def __init__(self):
            self.mcp_db = Capture()

    f = Fake()
    got = asyncio.run(f.log_tool_call(
        client_id="verify", tool_name="verify", parameters={},
        execution_time_ms=1.0, status="success", result={}, source="verify"))
    if got != "verify-ok" or len(f.mcp_db.args) != 8 or f.mcp_db.args[7] != "verify":
        print(f"  FAIL  forwarding wrong: {f.mcp_db.args}")
        return False
    print("  ok    source='verify' forwarded to db layer")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="verify only; change nothing")
    ap.add_argument("--status", action="store_true",
                    help="report state without importing the package")
    args = ap.parse_args()

    print(f"package: {PKG}")
    for p in (CORE, SERVER):
        if not p.exists():
            print(f"FATAL: {p} not found")
            return 2

    if args.status:
        core, server = _read(CORE), _read(SERVER)
        print("  facade accepts source : "
              f"{bool(FIXED_FACADE_RE.search(core))}")
        print("  helper present        : "
              f"{'_report_background_failure' in server}")
        print("  guarded tasks         : "
              f"{server.count('add_done_callback(_report_background_failure)')}")
        return 0

    if not args.check:
        touched = []
        if patch_core():
            touched.append("ai_memory_core.py (facade `source` param)")
        if patch_server():
            touched.append("mcp_server.py (task exception guard)")
        if touched:
            for t in touched:
                print(f"  patched  {t}")
        else:
            print("  already patched, nothing to do")

    print("\nstatic check:")
    if not check():
        return 1
    print("\nruntime check:")
    try:
        if not runtime_check():
            return 1
    except Exception as e:
        print(f"  FAIL  {type(e).__name__}: {e}")
        return 1

    print("\nAll checks passed. Restart the pams MCP server to load the change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
