"""
pams_client.py — Connect to persistent-ai-memory MCP server from Python.

Spawns pams-server as a subprocess and speaks JSON-RPC directly over stdio,
bypassing the buggy mcp.client.stdio transport.

Usage (interactive):
    python -c "import pams_client; asyncio.run(pams_client.demo())"

Usage (from your script):
    import pams_client
    import asyncio

    async def my_code():
        async with pams_client.PAMS() as pams:
            result = await pams.search("hello")
            print(result)
    asyncio.run(my_code())
"""

import asyncio, json, os, sys, time
from pathlib import Path
from typing import Any

# ThreadSafeDict for caching — from azazael/src/CORE
_ft_path = str(Path(__file__).resolve().parent.parent.parent / 'azazael' / 'src')
if _ft_path not in sys.path:
    sys.path.insert(0, _ft_path)
try:
    from CORE.freethreading import ThreadSafeDict
except ImportError:
    ThreadSafeDict = dict

MEMORY_DATA_DIR = Path(
    os.environ.get('AI_MEMORY_DATA_DIR')
    or os.environ.get('PAMS_DATA_DIR')
    or r"G:\source-of-truth\persistent-ai-memory\memory_data"
)
PAMS_CMD = "pams-server"

# Spawn the repo's launcher instead of the bare console script. The installed
# `persistent_ai_memory` resolves its databases with two disagreeing helpers
# (utils.get_memory_data_dir honours AI_MEMORY_DATA_DIR, but
# ai_memory_core.get_base_path returns the site-packages dir regardless), so
# the bare entry point writes the corpus INTO the install directory. The
# launcher repoints both before the server module is imported.
# See pams_server_launcher.py for the full explanation.
_LAUNCHER = str(Path(__file__).resolve().parent / "pams_server_launcher.py")
PAMS_LAUNCHER = _LAUNCHER if os.path.exists(_LAUNCHER) else PAMS_CMD

# Cache TTLs (seconds) — balances freshness with responsiveness
TOOL_LIST_TTL = 300      # Tool schemas rarely change
SEARCH_TTL = 30           # Memory searches can be slightly stale
HEALTH_TTL = 10           # Health checks should be relatively fresh


class PAMS:
    """Low-level MCP client for persistent-ai-memory.

    Manually handles JSON-RPC over the subprocess's stdio to avoid
    the cancellation-scope bug in the mcp.client.stdio transport.

    Uses ThreadSafeDict for caching tool lists and query results
    across concurrent async calls. Falls back to plain dict if
    freethread_utils is unavailable.
    """

    def __init__(self, memory_data_dir: Path | str = MEMORY_DATA_DIR):
        self._memory_data_dir = Path(memory_data_dir)
        self._proc: asyncio.subprocess.Process | None = None
        self._req_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._cache = ThreadSafeDict()
        self._cache_enabled = True

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()

    async def _reader(self):
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode().strip())
            except json.JSONDecodeError:
                continue
            rid = msg.get("id")
            if rid in self._pending:
                self._pending[rid].set_result(msg)
                del self._pending[rid]

    async def start(self):
        env = os.environ.copy()
        env["AI_MEMORY_DATA_DIR"] = str(self._memory_data_dir)

        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, PAMS_LAUNCHER,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )

        self._reader_task = asyncio.create_task(self._reader())

        result = await self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pams_client.py", "version": "1.0"},
        })
        if "error" in result:
            raise RuntimeError(f"MCP initialize failed: {result['error']}")

        await self._send_notification("notifications/initialized")

    async def stop(self):
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None

    async def _send(self, method: str, params: dict | None = None) -> dict:
        self._req_id += 1
        rid = self._req_id
        req = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            req["params"] = params
        future = asyncio.get_event_loop().create_future()
        self._pending[rid] = future
        raw = json.dumps(req) + "\n"
        self._proc.stdin.write(raw.encode())
        await self._proc.stdin.drain()
        return await asyncio.wait_for(future, timeout=30)

    async def _send_notification(self, method: str, params: dict | None = None):
        req = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            req["params"] = params
        raw = json.dumps(req) + "\n"
        self._proc.stdin.write(raw.encode())
        await self._proc.stdin.drain()

    def _cache_get(self, key: str, ttl: float = 30) -> tuple[bool, Any]:
        """Check cache for a fresh entry. Returns (hit: bool, value: Any)."""
        if not self._cache_enabled:
            return False, None
        entry = self._cache.get(key)
        if entry is None:
            return False, None
        ts, val = entry
        if time.monotonic() - ts < ttl:
            return True, val
        return False, None

    def _cache_set(self, key: str, value: Any) -> None:
        """Store value in cache with current timestamp."""
        if self._cache_enabled:
            self._cache[key] = (time.monotonic(), value)

    def cache_clear(self) -> None:
        """Clear all cached entries. Does not change enabled state."""
        self._cache.clear()

    def cache_disable(self) -> None:
        """Disable caching (for debugging or fresh queries)."""
        self._cache_enabled = False

    async def list_tools(self) -> list[dict]:
        hit, cached = self._cache_get('tools', TOOL_LIST_TTL)
        if hit:
            return cached
        result = await self._send("tools/list")
        if "error" in result:
            raise RuntimeError(f"tools/list failed: {result['error']}")
        tools = result.get("result", {}).get("tools", [])
        parsed = [{"name": t["name"], "description": t["description"],
                   "inputSchema": t.get("inputSchema", {})} for t in tools]
        self._cache_set('tools', parsed)
        return parsed

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        result = await self._send("tools/call", {"name": tool_name, "arguments": kwargs})
        if "error" in result:
            raise RuntimeError(f"tools/call '{tool_name}' failed: {result['error']}")
        content = result.get("result", {}).get("content", [])
        texts = [c["text"] for c in content if c.get("type") == "text"]
        return texts[0] if len(texts) == 1 else texts

    async def search(self, query: str, **kwargs) -> Any:
        cache_key = f'search:{query}:{kwargs.get("limit", "")}:{kwargs.get("offset", "")}'
        hit, cached = self._cache_get(cache_key, SEARCH_TTL)
        if hit:
            return cached
        kwargs.setdefault("user_id", "nate")
        kwargs.setdefault("model_id", "friday")
        result = await self.call_tool("search_memories", query=query, **kwargs)
        self._cache_set(cache_key, result)
        # Invalidate health cache since usage stats may have changed
        if self._cache_enabled:
            self._cache['health'] = (0, None)  # force stale via zero timestamp
        return result

    async def create_memory(self, content: str, **kwargs) -> Any:
        kwargs.setdefault("user_id", "nate")
        kwargs.setdefault("model_id", "friday")
        result = await self.call_tool("create_memory", content=content, **kwargs)
        # Invalidate only search and health caches (tools haven't changed)
        if self._cache_enabled:
            for key in list(self._cache.keys()):
                if key.startswith('search:') or key == 'health':
                    self._cache[key] = (0, None)  # force stale
        return result

    async def health(self) -> str:
        hit, cached = self._cache_get('health', HEALTH_TTL)
        if hit:
            return cached
        result = await self.call_tool("get_system_health", user_id="nate", model_id="friday")
        self._cache_set('health', result)
        return result


async def demo():
    print("Connecting to persistent-ai-memory...")
    async with PAMS() as pams:
        tools = await pams.list_tools()
        print(f"\nAvailable tools ({len(tools)}):")
        for t in tools:
            print(f"  {t['name']}")
        print()

        h = await pams.health()
        print(f"Health:\n{h}\n")

        memories = await pams.search("recent context", limit=3)
        print(f"Recent memories:\n{memories}")


if __name__ == "__main__":
    asyncio.run(demo())
