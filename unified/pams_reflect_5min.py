"""PAMS 5-minute reflection driver (tonight-only cadence, 2026-10-28/29).

Invokes the same code path as the MCP `reflect_on_tool_usage` tool
(AIMemorySystem.reflect_on_tool_usage -> mcp_tool_calls.ai_reflections)
without an MCP handshake: constructs AIMemorySystem against the canonical
data dir (AI_MEMORY_DATA_DIR, same default the launcher uses) and runs it.

Run:  C:/Azazael/.venvs/py315t/Scripts/python.exe pams_reflect_5min.py [days]
Scheduled: Task Scheduler "PAMS-Reflect-5min", every 5 min, expires 2026-10-30.
"""

import asyncio
import os
import sys
from pathlib import Path

PAMS_ROOT = Path(os.environ.get("PAMS_ROOT") or (Path.home() / ".pams"))
os.environ["AI_MEMORY_DATA_DIR"] = str(PAMS_ROOT / "memory_data")

from persistent_ai_memory.ai_memory_core import AIMemorySystem  # noqa: E402


async def main(days: int) -> dict:
    ms = AIMemorySystem(data_dir=str(PAMS_ROOT / "memory_data"))
    return await ms.reflect_on_tool_usage(
        days=days,
        user_id="clint",
        model_id="scheduler-5min",
        source="scheduler-5min",
        client_id="tonight-watch",
    )


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    result = asyncio.run(main(days))
    status = result.get("status", result.get("success", "?"))
    print(f"reflection status={status} days={days}")
    for k in ("reflection_id", "total_calls", "error"):
        if k in result:
            print(f"  {k}: {result[k]}")
    if str(status).lower() not in ("success", "true", "1"):
        print(result)
        sys.exit(1)
