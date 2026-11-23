"""pams_server_launcher.py — start pams-server with its data on local disk.

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
the port manager (the port file follows the env) and silently discarded by
databases (which land in site-packages). That split is why the corpus
fragmented into 3639 memories in site-packages and 60 on G:.

`pip install --force-reinstall` or a Python upgrade would have deleted the
3639-memory database, because it sits inside the install directory -- and on 2026-10-09 that loss was confirmed: the corpus is gone, only 87 rows survived in a pre-3.14 archive, and this launcher now defaults to ~/.pams.

WHAT THIS ALSO DOES
-------------------
Marks `mcp_server_port.json` inactive on exit. The port file outlives a hard
kill and keeps claiming `"status": "active"`, so a dead server reads as a live
one -- on 2026-09-29 it asserted an active port 21434 while nothing was
listening and the pid was gone. A graceful exit now corrects the record; a
SIGKILL still cannot be caught, which is why consumers should still probe the
port rather than trust the file.

WHAT THE HEARTBEAT ADDS
-----------------------
Exit hooks cannot cover SIGKILL, so the launcher also stamps the record with a
periodic `heartbeat_epoch` (every PAMS_HEARTBEAT_INTERVAL seconds, default 10)
while the server is alive, and rewrites `status` from "active" to "stale" when
it starts and finds a record whose pid is already dead. A record is therefore
verifiable rather than merely asserted: `is_record_live()` requires the status
to be active, the pid to be running, and the stamp to be recent. Consumers that
want the full picture call pmb_daemon_health.check_all().

WHAT THE CALLER-IDENTITY PATCH ADDS
-----------------------------------
`mcp_server._execute_tool` validates that every call carries `user_id` and
`model_id`, then its dispatch for `get_system_health` calls
`memory_system.get_system_health(source=source)` and drops both. The handler
falls back to its own "Nate"/"Friday" defaults, so the health payload reported
`requested_by` as the default rather than the caller. The launcher stashes the
validated identity around each dispatch and patches that one handler to use it,
so attribution survives the drop. site-packages itself is not edited, for the
same reason as the `daemon_health` tool below.

AT STARTUP
----------
`_surface_stale_records()` runs the shared daemon health check and logs a
warning for every advertised-but-dead endpoint. Because kilo and opencode both
spawn this launcher, a downed pmb daemon is surfaced automatically instead of
being silently swallowed.

Env:
    PAMS_ROOT  canonical data root. Defaults to ~/.pams on local disk.
               (G: is slow for a live store; reinstalls wipe site-packages.)
"""
import atexit
import contextvars
import functools
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

try:
    import compat_pydantic214  # noqa: F401 -- vendored pydantic-2.14 backfill
except ImportError:
    _aztools = 'C:/Azazael/azazael/tools'
    if _aztools not in sys.path:
        sys.path.insert(0, _aztools)
    import compat_pydantic214  # noqa: F401

PAMS_ROOT = Path(os.environ.get("PAMS_ROOT") or (Path.home() / ".pams"))
PAMS_DATA = PAMS_ROOT / "memory_data"
PORT_FILE = PAMS_DATA / "mcp_server_port.json"

HEARTBEAT_INTERVAL = float(os.environ.get("PAMS_HEARTBEAT_INTERVAL", "10"))
_stop_heartbeat = threading.Event()

# The caller identity of the MCP request currently being dispatched. Set by the
# `_execute_tool` wrapper and read by the patched write path; see
# "WHAT THE CALLER-IDENTITY PATCH ADDS" in the module docstring.
_CALLER = contextvars.ContextVar("pams_caller", default=None)

PAMS_DATA.mkdir(parents=True, exist_ok=True)
# Both the port manager and anything else reading the env var get the right
# answer; the patch below covers the resolver that ignores it.
os.environ["AI_MEMORY_DATA_DIR"] = str(PAMS_DATA)
os.environ["PAMS_DATA_DIR"] = str(PAMS_DATA)


def _mark_port_file_inactive() -> None:
    """Downgrade the port file so a dead server stops reading as a live one."""
    _stop_heartbeat.set()  # stop stamping before declaring the server stopped
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


# --- liveness and heartbeat ------------------------------------------------
# Every helper below shares one premise: the port file is a HINT. A record only
# proves a live server if its pid is running AND it was stamped recently. A
# SIGKILLed process can leave `"status": "active"` behind forever; it cannot
# leave a fresh heartbeat behind.


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _pid_alive(pid) -> bool:
    """Liveness probe. On Windows os.kill(pid, 0) would TERMINATE the pid."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import psutil  # type: ignore

            return bool(psutil.pid_exists(pid))
        except Exception:
            pass
        try:
            import ctypes

            k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = k32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFO
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == 259  # STILL_ACTIVE
            finally:
                k32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _load_port_file():
    try:
        data = json.loads(PORT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_port_file(data: dict) -> None:
    """Atomic replace, so a consumer never reads a half-written record."""
    tmp = PORT_FILE.with_name(PORT_FILE.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, PORT_FILE)
    except OSError:
        pass


def _heal_stale_record() -> None:
    """Correct a record that a dead process left claiming "active"."""
    data = _load_port_file()
    if not data or str(data.get("status", "")).lower() != "active":
        return
    pid = data.get("pid")
    if pid == os.getpid() or _pid_alive(pid):
        return
    data["status"] = "stale"
    data["stale_reason"] = "pid %s was not running when the launcher started" % pid
    data["observed_at"] = _now()
    _write_port_file(data)


_MY_PORT = None  # our bound port, learned from our own claim (see below)


def _heartbeat_loop() -> None:
    """Refresh the record every HEARTBEAT_INTERVAL seconds while we live."""
    global _MY_PORT
    while not _stop_heartbeat.wait(HEARTBEAT_INTERVAL):
        data = _load_port_file()
        if not data or str(data.get("status", "")).lower() != "active":
            continue
        pid = data.get("pid", data.get("process_id"))
        if isinstance(pid, int) and pid != os.getpid():
            # Another process's claim. Stamp it only if it is OURS... it
            # isn't, so normally skip. EXCEPTION (claim reclaim, 2026-10-29
            # inward sweep): every short-lived MCP spawn rewrites this shared
            # file with its own claim and dies, leaving a live heartbeat on
            # a dead record. If the claimant is dead, the port is closed,
            # and we already know OUR bound port from our own earlier claim,
            # take the record back instead of guarding a corpse. A LIVE
            # claimant is never touched, so two live owners cannot flap.
            try:
                import socket as _sock

                claimed_port = data.get("active_port")
                with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    claimed_open = (s.connect_ex(("127.0.0.1", int(claimed_port))) == 0) \
                        if isinstance(claimed_port, int) else None
            except Exception:
                claimed_open = None
            if (_MY_PORT is not None and not _pid_alive(pid)
                    and claimed_open is False
                    and claimed_port != _MY_PORT):
                data["active_port"] = _MY_PORT
                data["primary_port"] = _MY_PORT
                data["http_url"] = "http://127.0.0.1:%d" % _MY_PORT
                data["process_id"] = os.getpid()
                data["status"] = "active"
                data.pop("stale_reason", None)
            else:
                continue  # never stamp another (live or unknown) record
        if isinstance(data.get("active_port"), int):
            _MY_PORT = data["active_port"]  # learn our claim while we hold it
        data["pid"] = os.getpid()
        data["process_id"] = os.getpid()
        data["pid_alive"] = True
        data["heartbeat_at"] = _now()
        data["heartbeat_epoch"] = round(time.time(), 3)
        _write_port_file(data)


def _start_heartbeat() -> None:
    threading.Thread(target=_heartbeat_loop, name="pams-heartbeat",
                     daemon=True).start()


def is_record_live(max_age=None) -> bool:
    """Public check: active AND pid running AND stamped recently.

    A record with no heartbeat yet falls back to started_at, so one written
    before heartbeats existed is still judged honestly rather than assumed bad.
    """
    max_age = HEARTBEAT_INTERVAL * 3 if max_age is None else float(max_age)
    data = _load_port_file()
    if not data or str(data.get("status", "")).lower() != "active":
        return False
    if not _pid_alive(data.get("pid")):
        return False
    stamp = data.get("heartbeat_epoch") or data.get("started_at")
    if not isinstance(stamp, (int, float)):
        return False
    return (time.time() - float(stamp)) <= max_age


def _surface_stale_records() -> None:
    """Log advertised-but-dead endpoints at startup. Never raises."""
    try:
        import pmb_daemon_health as health

        results = health.check_all(pams_port_file=PORT_FILE, timeout=1.0)
    except Exception as exc:
        print("[pams] daemon health check skipped: %s" % exc, file=sys.stderr)
        return
    for res in results:
        if res.get("status") != "STALE":
            continue
        print("[pams] WARNING stale %s: pid=%s %s:%s claimed=%s (%s)"
              % (res.get("origin"), res.get("pid"), res.get("host"),
                 res.get("port"), res.get("claimed"),                 "; ".join(res.get("reasons") or []) or "unknown"),
              file=sys.stderr)


#: The floor proxy is the piece that makes pmb's relevance floor
#: unconditional: Kilo's `pmb` entry points at 127.0.0.1:8766, so a dead proxy
#: costs every pmb tool, not just the floor. Kept warm here because this
#: launcher is already the process that owns pmb-daemon lifecycle.
FLOOR_PORT = int(os.environ.get("PMB_FLOOR_PORT", "8766"))
FLOOR_PROXY = os.environ.get(
    "PMB_FLOOR_PROXY", "C:/Azazael/azazael/tools/pmb_floor_proxy.py")
FLOOR_PY = os.environ.get(
    "PMB_FLOOR_PY", "C:/Users/clint/.pmb-venv/Scripts/python.exe")


def _floor_port_open(timeout: float = 1.0) -> bool:
    import socket as _socket
    try:
        with _socket.create_connection(("127.0.0.1", FLOOR_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def _keep_floor_warm() -> None:
    """Background: keep the pmb floor proxy listening on 127.0.0.1:FLOOR_PORT.

    Restarts are logged with an attempt count; a missing interpreter or script
    disables the loop outright instead of spinning forever. Opt out with
    PAMS_FLOOR_AUTOSTART=0.
    """
    import subprocess as _sp
    if os.environ.get("PAMS_FLOOR_AUTOSTART", "1") == "0":
        return
    if not (os.path.isfile(FLOOR_PROXY) and os.path.isfile(FLOOR_PY)):
        print("[pams] floor proxy autostart skipped: missing %s or %s"
              % (FLOOR_PROXY, FLOOR_PY), file=sys.stderr)
        return
    attempts = 0
    while True:
        if not _floor_port_open():
            try:
                # Declared, not inferred: this proxy is alive only because a
                # client session started it, so the gate can tell it apart from
                # the one the logon Startup item owns.
                _env = dict(os.environ, PMB_FLOOR_ORIGIN="pams")
                _sp.Popen([FLOOR_PY, FLOOR_PROXY],
                          stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                          close_fds=True, env=_env)
                attempts += 1
                print("[pams] floor proxy started (attempt %d) on 127.0.0.1:%d"
                      % (attempts, FLOOR_PORT), file=sys.stderr)
            except Exception as exc:
                print("[pams] floor proxy spawn failed: %s" % exc,
                      file=sys.stderr)
        time.sleep(60)


def _keep_pmb_warm() -> None:
    """Background: bring the pmb daemon up and keep stamping its heartbeat.

    Deliberately a thread, not a call in the startup path: the daemon loads an
    embedding model, so blocking the MCP handshake on it would risk the client's
    startup timeout. Failures are logged and swallowed -- a missing memory
    daemon must never stop pams from serving.

    Lifecycle transitions -- a start, a restart after a crash -- are written
    through `_audit_and_anchor`, so they land in the same tamper-evident audit
    sidecar the supervisor uses. A healthy tick is NOT recorded: stamping a
    heartbeat every interval would turn the trail into a heartbeat log and bury
    the events that matter. This mirrors supervise()'s live -> gone rule.

    Opt out with PAMS_PMB_AUTOSTART=0.
    """
    try:
        import pmb_daemon_launcher as pmb
    except Exception as exc:
        print("[pams] pmb autostart unavailable: %s" % exc, file=sys.stderr)
        return
    try:
        home = pmb.health.find_pmb_home()
        was_live = pmb.daemon_live(home)
        seen_live = was_live
        state = pmb.ensure_running(home)
        print("[pams] pmb daemon: %s" % state, file=sys.stderr)
        live = pmb.daemon_live(home)
        if live and not was_live:
            _pmb_transition(pmb, "started", home, state)
            seen_live = True
        was_live = live
        while True:
            pmb.stamp_heartbeat(home)
            live = pmb.daemon_live(home)
            if not live:
                if was_live:
                    # The daemon was up and is not any more: trace the edge, not
                    # every tick it stays down, so a restart loop cannot flood
                    # the sidecar.
                    _pmb_transition(pmb, "gone", home, None)
                state = pmb.ensure_running(home)
                print("[pams] pmb daemon: %s" % state, file=sys.stderr)
                live = pmb.daemon_live(home)
                if live:
                    _pmb_transition(pmb, "restarted" if seen_live else "started",
                                    home, state)
                    seen_live = True
            was_live = live
            time.sleep(pmb.DEFAULT_INTERVAL)
    except Exception as exc:                       # pragma: no cover - best effort
        print("[pams] pmb autostart stopped: %s" % exc, file=sys.stderr)


def _pmb_transition(pmb, kind: str, home, ensure) -> None:
    """Record a pmb lifecycle edge. Never raises -- this is a best-effort trace."""
    try:
        event = "pmb_daemon_%s_by_pams" % kind
        fields = {"pmb_home": str(home), "origin": "pams_server_launcher"}
        if ensure is not None:
            fields["ensure"] = ensure
        pmb._audit_and_anchor(event, **fields)
    except Exception as exc:                       # pragma: no cover - best effort
        print("[pams] pmb audit skipped: %s" % exc, file=sys.stderr)


# --- patch the resolver that ignores the env var, BEFORE the import ---------

# PAMS-SOURCE-001: prefer the versioned source tree over site-packages.
# site-packages is a build artifact: `pip install --force-reinstall` silently
# reverts every fix applied there (measured 2026-09-29, 3,639 memories lost
# with the revert). Canonical source is C:/Azazael/pams-source (git-tracked).
# Prepending both the tree root (for `import persistent_ai_memory`) and the
# package dir (for flat `import ai_memory_core`-style imports) makes every
# spawn -- persistent daemon or short-lived MCP child, any env -- resolve to
# source. site-packages stays on sys.path as fallback. Override with
# PAMS_SOURCE (empty string disables: PAMS_SOURCE= forces installed package).
_PAMS_SOURCE = os.environ.get("PAMS_SOURCE", r"C:\Azazael\pams-source")
if _PAMS_SOURCE:
    for _p in (_PAMS_SOURCE,
               os.path.join(_PAMS_SOURCE, "persistent_ai_memory")):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        print("[pams] source tree preferred: %s" % _PAMS_SOURCE,
              file=sys.stderr)
    except Exception:
        pass
del _p, _PAMS_SOURCE

# The relevance-floor proxy is spawned HERE, ahead of the heavy PAMS imports
# below, because its bind time is a client-visible deadline. Measured
# 2026-10-31: opencode's `pmb` entry is a remote MCP server and gives an
# unreachable endpoint three attempts in 1.65 s, after which it marks the
# server failed for the whole session -- while this launcher reached the old
# start site ~3 s in, so every cold start lost the race (ECONNREFUSED at
# 07:38:22.470, daemon up 07:38:24, proxy bound 07:38:25). Spawning at ~0.2 s
# puts :8766 inside the window; the proxy itself waits for the daemon.
_FLOOR_WATCH_STARTED = False


def _start_floor_watch() -> None:
    """Start the keep-warm watcher once. Idempotent: two watchers race."""
    global _FLOOR_WATCH_STARTED
    if _FLOOR_WATCH_STARTED:
        return
    _FLOOR_WATCH_STARTED = True
    threading.Thread(target=_keep_floor_warm, name="floor-proxy-watch",
                     daemon=True).start()


_start_floor_watch()

# ai_memory_core.get_base_path is looked up inside AIMemorySystem.__init__ at
# call time, so rebinding it on the module object is enough. It must happen
# before `mcp_server` is imported, because that module builds its databases at
# import time.
import ai_memory_core  # noqa: E402


def _base_path():
    # Return a Path, matching upstream's `current_file.parent`. A str here
    # breaks `ensure_all_memory_databases_ready`, which does
    # `base_dir / "memory_data"` on the result (str/str TypeError).
    # Every other caller wraps in str()/Path() and tolerates a Path.
    return PAMS_ROOT


ai_memory_core.get_base_path = _base_path


# --- alias the attribute name `database_maintenance` gets wrong --------------
# `database_maintenance._build_tag_registries` and `_build_memory_bank_registries`
# both call `self.memory_system.ai_memories_db.execute_query(...)`, but
# `AIMemorySystem.__init__` defines `self.ai_memory_db` (SINGULAR). The two
# builders therefore always died with AttributeError, so `list_available_tags`
# and `list_available_memory_banks` could never return a non-zero count on ANY
# install -- an empty registry was indistinguishable from a missing one. Alias
# the misspelled name instead of editing the two call sites in site-packages.
if not hasattr(ai_memory_core.AIMemorySystem, "ai_memories_db"):
    ai_memory_core.AIMemorySystem.ai_memories_db = property(
        lambda self: self.ai_memory_db)


# --- repair the two registry builders ----------------------------------------
# `DatabaseMaintenance._build_memory_bank_registries` and `_build_tag_registries`
# call `row.get(...)` on `sqlite3.Row` objects. Those rows have no `.get`, and
# `sqlite3.Row` is an IMMUTABLE type, so it cannot be extended either -- the
# builders died with AttributeError on every run, which made
# `list_available_tags` and `list_available_memory_banks` report an empty registry
# on every install (an empty registry was indistinguishable from a missing one).
# Rebuild both here from plain dicts, matching the shapes the writers produced.
def _install_registry_builder_repair() -> bool:
    """Replace the two builder methods with working ones. Idempotent."""
    cls = getattr(ai_memory_core, "DatabaseMaintenance", None)
    if cls is None or getattr(cls, "_pams_registry_repaired", False):
        return False

    async def _build_memory_bank_registries(self):
        path = PAMS_DATA / "memory_bank_registry.json"
        res = {"main_registry_path": str(path), "docker_registry_path": str(path),
               "banks_found": 0, "total_memories": 0, "errors": []}
        try:
            rows = await self.memory_system.ai_memory_db.execute_query(
                "SELECT memory_bank, COUNT(*) AS count FROM curated_memories "
                "GROUP BY memory_bank", ())
            registry = {}
            for row in rows:
                bank = row["memory_bank"] or "General"
                count = row["count"] or 0
                registry[bank] = {"name": bank, "memory_count": count}
                res["total_memories"] += count
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(registry, indent=2, sort_keys=True),
                            encoding="utf-8")
            res["banks_found"] = len(registry)
        except Exception as exc:                 # pragma: no cover - best effort
            res["errors"].append(str(exc))
        return res

    async def _build_tag_registries(self):
        path = PAMS_DATA / "tag_registry.json"
        res = {"main_registry_path": str(path), "docker_registry_path": str(path),
               "tags_found": 0, "memories_scanned": 0, "errors": []}
        try:
            from tag_manager import TagManager

            rows = await self.memory_system.ai_memory_db.execute_query(
                "SELECT memory_id, content, tags FROM curated_memories", ())
            memories = [dict(row) for row in rows]
            res["memories_scanned"] = len(memories)
            if memories:
                manager = TagManager()
                registry = manager.build_tag_registry(memories)
                res["tags_found"] = len(registry)
                path.parent.mkdir(parents=True, exist_ok=True)
                manager.save_registry(registry, str(path))
                try:
                    res["summary"] = manager.get_registry_summary()
                except Exception:
                    pass
        except Exception as exc:                 # pragma: no cover - best effort
            res["errors"].append(str(exc))
        return res

    cls._build_memory_bank_registries = _build_memory_bank_registries
    cls._build_tag_registries = _build_tag_registries
    cls._pams_registry_repaired = True
    return True


_install_registry_builder_repair()


# --- resilient embeddings: truncate-and-retry, and never store a zero vector -
# Two defects lived here, both of which made long memories unsearchable:
#
#  1. mxbai-embed-large-v1 is trained on 512 tokens and llama.cpp CLAMPS to it
#     (`/props` reports n_ctx=512 even when launched with `-c 4096`). Any memory
#     longer than 512 tokens was rejected with HTTP 500, so the summary-shaped
#     memories -- the ones worth keeping -- could never be embedded.
#  2. On rejection `EmbeddingService.generate_embedding` returns an ALL-ZERO
#     vector, and `_generate_and_store_embedding` counts any non-None result as
#     success. Those failures were written to `curated_memories.embedding` as a
#     zero vector and reported as stored; a zero vector matches nothing, so
#     semantic search silently skipped 11 of the restored memories.
#
# Fix: treat a zero vector as failure, and shrink the text until it fits. The
# first attempt is always the full content, so nothing is truncated needlessly.
def _install_embedding_resilience() -> bool:
    svc = getattr(ai_memory_core, "EmbeddingService", None)
    original = getattr(svc, "generate_embedding", None) if svc else None
    if original is None or getattr(svc, "_pams_embed_resilient", False):
        return False

    @functools.wraps(original)
    async def _generate_embedding(self, text, *args, **kwargs):
        attempt = text
        for _ in range(7):
            vector = await original(self, attempt, *args, **kwargs)
            if vector is not None:
                try:
                    if any(vector):
                        return vector
                except TypeError:
                    return vector
            if not isinstance(attempt, str) or len(attempt) < 300:
                break
            attempt = attempt[: int(len(attempt) * 0.6)]
            print("[pams] embedding rejected (%d chars); retrying with %d"
                  % (len(text), len(attempt)), file=sys.stderr)
        return None

    svc.generate_embedding = _generate_embedding
    svc._pams_embed_resilient = True
    return True


_install_embedding_resilience()

# Import the server by its FULL package path. site-packages also contains a
# top-level `mcp_server.py` (plus decoy `utils.py`, `port_manager.py`,
# `database_maintenance.py`) used by `from ai_memory_core import ...`-style
# internal imports. A bare `import mcp_server` silently binds that decoy --
# which has no `run_main` -- so the explicit path is required, not stylistic.
from persistent_ai_memory import mcp_server  # noqa: E402

# mcp_server computes its own BASE_PATH from its own get_base_path at import.
# Rebind it so logs/keys/config also follow the canonical root.
mcp_server.BASE_PATH = PAMS_ROOT


# --- surface daemon/audit health as an MCP tool -----------------------------
# The daemon/audit verdict lives in these wrappers, not in pams, so a client
# talking to pams cannot see it unless we publish it. We add ONE tool to the
# server we just imported -- from OUTSIDE the package. site-packages is never
# edited (a reinstall would revert it); the two handler methods are patched on
# the class instead. Both MCP handlers dispatch through `self.<method>()`, so a
# patched method is picked up even on an already-constructed server.

HEALTH_TOOL_NAME = "daemon_health"
HEALTH_TOOL_DESCRIPTION = ("Report pmb daemon liveness and audit-chain "
                           "integrity for this host.")


def _daemon_health_payload(timeout: float = 0.5) -> dict:
    """JSON-able daemon + audit verdict. Never raises."""
    payload: dict = {"ok": True, "records": [], "stale": 0, "audit": {}}
    try:
        import pmb_daemon_health as health

        payload["records"] = health.check_all(pams_port_file=PORT_FILE,
                                              timeout=timeout)
    except Exception as exc:
        payload["ok"] = False
        payload["health_error"] = str(exc)
    stale = [r for r in payload["records"] if r.get("status") == "STALE"]
    payload["stale"] = len(stale)
    try:
        import pmb_daemon_launcher as pmb

        chain_ok, chain = pmb.verify_audit_chain()
        anchor_ok, anchor = pmb.verify_against_anchors()
        payload["audit"] = {"chain_ok": chain_ok, "chain": chain,
                            "anchor_ok": anchor_ok, "anchor": anchor,
                            "merkle_root": pmb.merkle_root()}
        payload["ok"] = payload["ok"] and chain_ok and anchor_ok
        # Heal history makes a flapping daemon visible to the client too, not
        # just to whoever reads the audit file.
        heals = pmb.heal_history()
        payload["heals_total"] = len(heals)
        payload["heals_1h"] = len([h for h in heals
                                   if isinstance(h.get("epoch"), (int, float))
                                   and time.time() - float(h["epoch"]) <= 3600])
    except Exception as exc:
        payload["ok"] = False
        payload["audit_error"] = str(exc)
    payload["ok"] = payload["ok"] and not stale
    # A client-facing verdict, so a caller does not have to parse the records.
    if stale:
        payload["severity"] = "critical"
        payload["alert"] = ("%d advertised pmb daemon endpoint(s) are stale"
                            % len(stale))
    elif not payload["ok"]:
        payload["severity"] = "warning"
        payload["alert"] = (payload.get("health_error")
                            or payload.get("audit_error")
                            or "audit trail failed verification")
    else:
        payload["severity"] = "ok"
        payload["alert"] = None
    return payload


def _health_tool_result(payload: dict | None = None):
    """Wrap the payload the way the server's native tools do: a plain dict.

    The low-level SDK (mcp/server/lowlevel/server.py::call_tool) normalizes a
    handler's return value by shape: a dict becomes structuredContent (plus a
    JSON text part); a non-dict iterable is treated as raw content blocks. A
    real CallToolResult object falls in the iterable bucket -- pydantic models
    iterate as (field, value) tuples -- and those tuples were sent to clients
    as content items, breaking daemon_health with 20 validation errors while
    all 28 dict-returning native tools worked. So this returns a dict.
    """
    import json

    payload = _daemon_health_payload() if payload is None else payload
    return {
        "content": [{"type": "text",
                     "text": json.dumps(payload, indent=2, default=str)}],
        "success": payload["ok"],
        "isError": not payload["ok"],
    }


async def _alert_if_unhealthy(server, payload: dict) -> None:
    """Push a client-facing MCP log when the verdict is not OK.

    The tool result is pull-only: a client that never calls daemon_health sees
    nothing. This also PUSHES the alert, so a stale endpoint surfaces in the
    client's log stream. Never fatal -- a client without the logging capability
    raises here, and that must not fail the tool call.
    """
    if payload.get("severity") == "ok":
        return
    try:
        session = server.server.request_context.session
        await session.send_log_message(
            level="error", logger="pams.daemon_health",
            data=payload.get("alert") or "pmb daemon health degraded")
    except Exception:
        pass


def _install_health_tool() -> bool:
    """Add `daemon_health` to the pams MCP server. Best effort, never fatal."""
    try:
        from mcp.types import Tool
    except Exception as exc:
        print("[pams] daemon_health tool unavailable: %s" % exc, file=sys.stderr)
        return False
    server_cls = getattr(mcp_server, "AIMemoryMCPServer", None)
    original_list = getattr(server_cls, "_get_client_tools", None)
    original_call = getattr(server_cls, "_execute_tool", None)
    if not (server_cls and original_list and original_call):
        print("[pams] daemon_health tool unavailable: no AIMemoryMCPServer API",
              file=sys.stderr)
        return False

    async def _get_client_tools(self):
        tools = await original_list(self)
        tools.append(Tool(name=HEALTH_TOOL_NAME,
                          description=HEALTH_TOOL_DESCRIPTION,
                          inputSchema={"type": "object", "properties": {}}))
        return tools

    async def _execute_tool(self, name, arguments):
        if name == HEALTH_TOOL_NAME:
            payload = _daemon_health_payload()
            await _alert_if_unhealthy(self, payload)
            return _health_tool_result(payload)
        # Publish the validated caller for the duration of this dispatch, so
        # handlers whose dispatch branch drops user_id/model_id can still
        # attribute the call. Reset in `finally` -- a leaked token would
        # misattribute every later request on this thread.
        args = arguments if isinstance(arguments, dict) else {}
        token = _CALLER.set((args.get("user_id"), args.get("model_id")))
        try:
            return await original_call(self, name, arguments)
        finally:
            _CALLER.reset(token)

    server_cls._get_client_tools = _get_client_tools
    server_cls._execute_tool = _execute_tool
    return True


#: Identity defaults for clients that cannot know the convention.
#: `mcp_server._execute_tool` / ai_memory_core refuse any call without
#: user_id and model_id -- "MISSING REQUIRED PARAMETERS ... Do not use
#: defaults. Provide the actual user identifier and your model name from the
#: system prompt" -- a contract no MCP client satisfies. Measured live
#: 2026-10-30: a bare `get_system_health` call dies as
#: "Input validation error: 'user_id' is a required property", which takes
#: all 32 tools out of reach for any caller that does not already know the
#: convention. Filling the two at the MCP BOUNDARY keeps the core's rule
#: intact (it still never invents an identity itself) while making every
#: tool callable. Override per host with PAMS_USER_ID / PAMS_MODEL_ID.
DEFAULT_USER_ID = os.environ.get("PAMS_USER_ID", "clint")
DEFAULT_MODEL_ID = os.environ.get("PAMS_MODEL_ID", "mcp-client")
IDENTITY_FIELDS = ("user_id", "model_id")

#: One additive dispatch entry point. Every PAMS tool stays registered and
#: visible -- this adds a single name that a swarm lane can call without
#: carrying 32 schemas, and it forwards to the same `_execute_tool` the
#: individual tools use, so the two paths cannot drift.
ADMIN_TOOL_NAME = "pams_admin"
ADMIN_TOOL_DESCRIPTION = (
    "One entry point for every PAMS action (all 32 pams_* tools). "
    "`action` is the tool name without its prefix -- e.g. search_memories, "
    "create_reminder, get_system_health, get_weather_open_meteo -- and "
    "`params` is that tool's own arguments object. Identity (user_id, "
    "model_id) is filled automatically, so it may be omitted. "
    "Use this when a task needs several PAMS calls: one dispatch per action. "
    "The individual tools remain callable as before."
)


def _install_identity_defaults() -> bool:
    """Fill user_id/model_id at the MCP boundary; add the pams_admin router.

    Best effort, never fatal -- a launcher patch that cannot find its hook
    must leave the server exactly as it was rather than half-modified.

    Installed AFTER `_install_health_tool`, so the wrappers chain:
    this one fills identity, then delegates to the health wrapper (which
    publishes the caller and handles `daemon_health`), then to the original.
    """
    try:
        from mcp.types import Tool
    except Exception as exc:
        print("[pams] identity defaults skipped: %s" % exc, file=sys.stderr)
        return False
    server_cls = getattr(mcp_server, "AIMemoryMCPServer", None)
    original_list = getattr(server_cls, "_get_client_tools", None)
    original_call = getattr(server_cls, "_execute_tool", None)
    if not (server_cls and original_list and original_call):
        print("[pams] identity defaults skipped: no AIMemoryMCPServer API",
              file=sys.stderr)
        return False
    if getattr(server_cls, "_pams_identity_defaults", False):
        return True

    def _fill(arguments):
        args = dict(arguments) if isinstance(arguments, dict) else {}
        for field, default in (("user_id", DEFAULT_USER_ID),
                              ("model_id", DEFAULT_MODEL_ID)):
            if not args.get(field):
                args[field] = default
        return args

    async def _get_client_tools(self):
        tools = await original_list(self)
        for t in tools:
            try:
                schema = t.inputSchema if isinstance(t.inputSchema, dict) else None
                if not schema:
                    continue
                props = schema.get("properties")
                if not isinstance(props, dict):
                    continue
                touched = False
                for field, default in (("user_id", DEFAULT_USER_ID),
                                      ("model_id", DEFAULT_MODEL_ID)):
                    if field in props:
                        props[field].setdefault("default", default)
                        touched = True
                if not touched:
                    continue
                required = schema.get("required")
                if isinstance(required, list):
                    kept = [r for r in required if r not in IDENTITY_FIELDS]
                    if kept:
                        schema["required"] = kept
                    else:
                        schema.pop("required", None)
                t.inputSchema = schema
            except Exception:
                continue  # one odd schema must not cost the whole surface
        if not any(getattr(t, "name", None) == ADMIN_TOOL_NAME for t in tools):
            tools.append(Tool(
                name=ADMIN_TOOL_NAME,
                description=ADMIN_TOOL_DESCRIPTION,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "PAMS tool name without the pams_ "
                                           "prefix, e.g. search_memories",
                        },
                        "params": {
                            "type": "object",
                            "description": "Arguments for that tool",
                        },
                    },
                    "required": ["action"],
                },
            ))
        return tools

    async def _execute_tool(self, name, arguments):
        args = _fill(arguments)
        if name == ADMIN_TOOL_NAME:
            action = str(args.get("action") or "").strip()
            params = args.get("params")
            if not isinstance(params, dict):
                params = {}
            if not action:
                return _admin_text_error(
                    "pams_admin needs an `action` (a pams_* tool name).")
            if action.startswith("pams_"):
                action = action[len("pams_"):]
            if action == ADMIN_TOOL_NAME:
                return _admin_text_error("pams_admin cannot dispatch to itself.")
            if not params.get("user_id"):
                params["user_id"] = args["user_id"]
            if not params.get("model_id"):
                params["model_id"] = args["model_id"]
            return await original_call(self, action, params)
        return await original_call(self, name, args)

    def _admin_text_error(message):
        try:
            from mcp.types import TextContent
            return [TextContent(type="text", text=message)]
        except Exception:
            return message

    server_cls._get_client_tools = _get_client_tools
    server_cls._execute_tool = _execute_tool
    server_cls._pams_identity_defaults = True
    print("[pams] identity defaults installed (user_id=%s model_id=%s) + %s"
          % (DEFAULT_USER_ID, DEFAULT_MODEL_ID, ADMIN_TOOL_NAME), file=sys.stderr)
    return True


def _install_health_alias_fix() -> bool:
    """Teach the health verdict the portfile's real schema. Best effort.

    port_manager writes `active_port`/`process_id`; pmb_daemon_health._check
    reads `port`/`pid`. Without aliases the PAMS record can never verify:
    port_open stays null (UNKNOWN at best), and any record missing the
    heartbeat-added `pid` reads STALE/critical ("record carries neither pid
    nor port") while the daemon is alive -- the exact false critical the
    2026-10-29 inward sweep caught. Fixed from OUTSIDE the checked-out
    packages: wrap check_all (sibling module pmb_daemon_health, same
    monkeypatch pattern as the other launcher installs) and, for
    pams_port_file records with port None, re-derive from a fresh read of
    PORT_FILE plus live pid/port probes, then re-classify with _check's own
    rules. Never fatal; on any error the original verdict stands.
    """
    import functools
    import json

    try:
        import pmb_daemon_health as health
    except ImportError as exc:
        print("[pams] health alias fix skipped: %s" % exc, file=sys.stderr)
        return False
    original = getattr(health, "check_all", None)
    if not original or getattr(health, "_pams_alias_fixed", False):
        return False

    @functools.wraps(original)
    def check_all(pmb_home=None, pams_port_file=None, timeout=2.0):
        results = original(pmb_home, pams_port_file, timeout)
        if not pams_port_file:
            return results
        try:
            data = health._read_json(Path(pams_port_file)) or {}
        except Exception:
            data = {}
        aport = data.get("active_port")
        apid = data.get("process_id", data.get("pid"))
        for r in results:
            if r.get("origin") != "pams_port_file" or r.get("port") is not None:
                continue
            try:
                host = r.get("host") or "127.0.0.1"
                r["port"] = aport
                if r.get("pid") is None:
                    r["pid"] = apid
                alive = health._pid_alive(apid)
                listening = (health._port_open(host, aport, timeout)
                             if isinstance(aport, int) else None)
                r["pid_alive"] = alive
                r["port_open"] = listening
                reasons = []
                if alive is False:
                    reasons.append("pid %s is not running" % apid)
                if listening is False:
                    reasons.append("nothing listening on %s:%s" % (host, aport))
                if alive is None and listening is None:
                    reasons.append("record carries neither pid nor port")
                r["reasons"] = reasons
                claimed = str(data.get("status", "active")).lower()
                r["claimed"] = claimed
                if claimed != "active":
                    r["status"] = "INACTIVE"
                elif alive and listening:
                    r["status"] = "OK"
                elif reasons:
                    r["status"] = "STALE"
                else:
                    r["status"] = "UNKNOWN"
            except Exception:
                pass  # original verdict stands for this record
        return results

    health.check_all = check_all
    health._pams_alias_fixed = True
    return True


def _install_caller_identity() -> bool:
    """Make `get_system_health` name its real caller. Best effort, never fatal.

    `mcp_server._execute_tool` requires every call to carry `user_id` and
    `model_id`, then its dispatch for this ONE tool calls
    `memory_system.get_system_health(source=source)` and drops both. The handler
    then falls back to its own "Nate"/"Friday" defaults, so the payload's
    `requested_by` named the default rather than the caller -- an attested
    caller silently rewritten to somebody else.

    Only this branch was affected: `get_current_time_tool`, `run_maintenance`
    and the weather branch never report an identity at all, and the other 28
    dispatch branches forward user_id/model_id themselves.
    """
    cls = getattr(ai_memory_core, "AIMemorySystem", None)
    original = getattr(cls, "get_system_health", None) if cls else None
    if not (cls and original):
        print("[pams] caller identity patch skipped: no AIMemorySystem API",
              file=sys.stderr)
        return False
    if getattr(cls, "_pams_caller_patched", False):
        return True

    @functools.wraps(original)
    async def _get_system_health(self, user_id=None, model_id=None, **kwargs):
        caller = _CALLER.get()
        if caller:
            caller_user, caller_model = caller
            user_id = user_id or caller_user
            model_id = model_id or caller_model
        return await original(self, user_id=user_id, model_id=model_id, **kwargs)

    try:
        cls.get_system_health = _get_system_health
        cls._pams_caller_patched = True
    except Exception as exc:                      # pragma: no cover - best effort
        print("[pams] caller identity patch skipped: %s" % exc, file=sys.stderr)
        return False
    return True


EMBED_PORT = int(os.environ.get("PAMS_EMBED_PORT", "8080"))
# `-c` is only a REQUEST: mxbai-embed-large-v1 is trained on 512 tokens and
# llama.cpp clamps to that, so `/props` reports n_ctx=512 even with a larger
# value here. Headroom is still asked for so swapping PAMS_EMBED_MODEL to a
# longer-context model takes effect without editing this file. Memories that do
# not fit are handled by truncate-and-retry in _install_embedding_resilience().
EMBED_CTX = int(os.environ.get("PAMS_EMBED_CTX", "4096"))
EMBED_URL = "http://127.0.0.1:%d" % EMBED_PORT
EMBED_BIN = os.environ.get("PAMS_EMBED_BIN", r"C:\Users\clint\llama-server\bin\llama-b10964-cpu\llama-server.exe")
EMBED_MODEL = os.environ.get("PAMS_EMBED_MODEL", r"C:\Users\clint\llama-server\models\mxbai-embed-large-v1\mxbai-embed-large-v1-Q4_k_m.gguf")


def _embedding_healthy(timeout: float = 1.0) -> bool:
    """True when the local embedding endpoint answers /health."""
    try:
        with urllib.request.urlopen(EMBED_URL + "/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _keep_embedding_warm() -> None:
    """Background: make sure the local llama-server embedding endpoint is up.

    WHY: PAMS's EmbeddingService posts to http://127.0.0.1:8080/v1/embeddings.
    Nothing else in the stack starts that server, so after a reboot semantic
    search silently degrades ("embedding_service: unhealthy") with no visible
    cause. This brings the endpoint up and keeps it up.

    Deliberately a thread, not a call in the startup path, for the same reason
    as _keep_pmb_warm: loading the model takes seconds and must not block the
    MCP handshake. Failures are logged and swallowed -- a missing embedder must
    never stop pams from serving. Opt out with PAMS_EMBED_AUTOSTART=0.
    """
    if not (Path(EMBED_BIN).exists() and Path(EMBED_MODEL).exists()):
        print("[pams] embedding autostart skipped: binary or model missing",
              file=sys.stderr)
        return
    while True:
        try:
            if not _embedding_healthy():
                print("[pams] starting embedding server on %s" % EMBED_URL,
                      file=sys.stderr)
                subprocess.Popen(
                    [EMBED_BIN, "-m", EMBED_MODEL, "--host", "127.0.0.1",
                     "--port", str(EMBED_PORT), "--embedding", "--pooling",
                     "mean", "-c", str(EMBED_CTX), "-b", str(EMBED_CTX)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
                for _ in range(60):
                    time.sleep(1.0)
                    if _embedding_healthy():
                        print("[pams] embedding server up", file=sys.stderr)
                        break
        except Exception as exc:              # pragma: no cover - best effort
            print("[pams] embedding autostart error: %s" % exc, file=sys.stderr)
        time.sleep(30)


# --- persistence: start file monitoring on the MCP stdio path ----------------
# `AIMemoryMCPServer.main()` never calls `handle_initialization()`, which is the
# only caller of `memory_system._start_monitoring()`. Every MCP-spawned server
# therefore reported `active_observers: 0, processed_files: 0` forever -- the
# monitor was constructed but never started. Fixed here, never in site-packages:
# wrap `__init__` so each server schedules the real startup on its own running
# loop. Delay is env-tunable for testability; default 15s so even short-lived
# MCP sessions get coverage (upstream's own delayed_start waits 60s for LM
# Studio init, which outlives most stdio sessions).
def _install_monitoring_autostart() -> bool:
    """Schedule file-monitoring startup on every server instance. Idempotent."""
    import asyncio
    import logging

    cls = getattr(mcp_server, "AIMemoryMCPServer", None)
    original = getattr(cls, "__init__", None) if cls else None
    if not (cls and original) or getattr(cls, "_pams_monitor_autostart", False):
        return False
    try:
        delay = float(os.environ.get("PAMS_MONITOR_DELAY", "15"))
    except ValueError:
        delay = 15.0

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # constructed off-loop; nothing to schedule on

        async def _delayed_monitor_start():
            await asyncio.sleep(delay)
            log = logging.getLogger(__name__)
            try:
                await self.memory_system._start_monitoring()
                log.info("File monitoring autostarted by pams_server_launcher")
            except Exception as exc:
                # _start_monitoring() starts the observers FIRST and then runs
                # ensure_all_memory_databases_ready(), which carries a
                # pre-existing str/str path bug. Verify rather than cry wolf.
                fm = getattr(self.memory_system, "file_monitor", None)
                n = len(getattr(fm, "observers", None) or [])
                if n:
                    log.info("File monitoring autostarted (%d observers; "
                             "ensure_databases raised: %s)" % (n, exc))
                else:
                    print("[pams] monitoring autostart failed: %s" % exc,
                          file=sys.stderr)

        loop.create_task(_delayed_monitor_start())

    cls.__init__ = __init__
    cls._pams_monitor_autostart = True
    return True


# --- inflation guard: bus telemetry must never be stored as high-importance memory
# Audit 2026-10-29: 4219/4338 curated rows (97.3%) were bus snapshots pushed
# through promotion (cluster.execute.start x3258, cluster.nature.built, lane.*,
# monitor.*) stamped at a flat importance 6. Recall orders BY importance DESC
# plus importance_boost, so machine snapshots buried the 119 genuine memories
# (quarantined to backups/quarantine_openwebui-*.db, sha-verified). Fixed here,
# never in site-packages: demote bus-topic content to importance 2 + telemetry
# type at the AIMemorySystem.create_memory choke point every path funnels
# through (MCP create_memory, HTTP promote which hardcodes 8). A human who
# truly wants a bus snapshot ranked high can re-raise it afterwards.
_BUS_TOPIC_RE = None  # compiled lazily (keeps import cheap)


def _bus_topic_re():
    import re

    global _BUS_TOPIC_RE
    if _BUS_TOPIC_RE is None:
        _BUS_TOPIC_RE = re.compile(
            r"cluster\.(execute|nature)\.|lane\.|monitor\.|bus\.monitor",
            re.IGNORECASE,
        )
    return _BUS_TOPIC_RE


def _install_promote_guard() -> bool:
    """Demote bus-telemetry content to importance 2 at create time. Idempotent."""
    cls = getattr(ai_memory_core, "AIMemorySystem", None)
    original = getattr(cls, "create_memory", None) if cls else None
    if not (cls and original) or getattr(cls, "_pams_promote_guarded", False):
        return False

    @functools.wraps(original)
    async def create_memory(self, *args, **kwargs):
        # Never pass one parameter both positionally and by keyword into
        # the wrapped call (TypeError: multiple values). Helpers below
        # write each parameter through exactly one channel.
        def _get(name, idx, default):
            if name in kwargs:
                return kwargs[name]
            if len(args) > idx:
                return args[idx]
            return default

        def _set(args, kwargs, name, idx, value):
            if name in kwargs or len(args) <= idx:
                kwargs[name] = value
                return args, kwargs
            return args[:idx] + (value,) + args[idx + 1 :], kwargs

        try:
            content = _get("content", 0, "")
            tags = _get("tags", 3, None)
            importance = _get("importance_level", 2, 5)
            haystack = str(content or "") + " " + " ".join(tags or [])
            if _bus_topic_re().search(haystack) and (importance or 5) >= 5:
                args, kwargs = _set(
                    args, kwargs, "importance_level", 2, 2
                )
                if not _get("memory_type", 1, None):
                    args, kwargs = _set(
                        args, kwargs, "memory_type", 1, "telemetry"
                    )
                t = list(_get("tags", 3, None) or [])
                if "bus-telemetry-guarded" not in t:
                    t.append("bus-telemetry-guarded")
                args, kwargs = _set(args, kwargs, "tags", 3, t)
                print(
                    "[pams] promote guard: bus-telemetry content demoted "
                    "to importance 2 (was %s)" % importance,
                    file=sys.stderr,
                )
        except Exception as exc:  # never block a memory write on the guard
            print("[pams] promote guard bypassed: %s" % exc, file=sys.stderr)
        return await original(self, *args, **kwargs)

    cls.create_memory = create_memory
    cls._pams_promote_guarded = True
    return True


# --- boost retune: importance must break ties, not override similarity --------
# Both search paths add importance/10*factor to the similarity score. With the
# flat-6 corpus that buried everything; with genuine 2-7 grading the old
# factors (0.15 insights / 0.10 memories, i.e. up to +0.105) still let a
# level-7 row outrank a clearly more similar level-2 row. Halve both so
# similarity decides and importance only orders near-ties. Applied as an
# output-side correction (subtract old, add new, re-sort) so no
# site-packages logic is forked; per-result try/except keeps a malformed
# row from failing the whole search.
def _install_boost_retune() -> bool:
    """Halve importance_boost on both search paths. Idempotent."""
    cls = getattr(ai_memory_core, "AIMemorySystem", None)
    if not cls or getattr(cls, "_pams_boost_retuned", False):
        return False
    try:
        new_insights = float(os.environ.get("PAMS_BOOST_INSIGHTS", "0.07"))
        new_memories = float(os.environ.get("PAMS_BOOST_MEMORIES", "0.05"))
    except ValueError:
        new_insights, new_memories = 0.07, 0.05
    targets = {
        "_search_project_insights": (0.15, new_insights),
        "_search_ai_memories": (0.10, new_memories),
    }
    patched = 0
    for name, (old_f, new_f) in targets.items():
        original = getattr(cls, name, None)
        if not original:
            print("[pams] boost retune skipped: no %s" % name, file=sys.stderr)
            continue

        async def _wrapped(self, *args, _orig=original, _old=old_f,
                           _new=new_f, **kwargs):
            results = await _orig(self, *args, **kwargs)
            try:
                for r in results or []:
                    try:
                        lvl = (r.get("data") or {}).get("importance_level", 5)
                        base = r.get("similarity_score", 0) or 0
                        r["similarity_score"] = (
                            base - lvl / 10.0 * _old + lvl / 10.0 * _new
                        )
                    except Exception:
                        continue
                results.sort(
                    key=lambda x: x.get("similarity_score", 0), reverse=True
                )
            except Exception as exc:
                print("[pams] boost resort skipped: %s" % exc, file=sys.stderr)
            return results

        setattr(cls, name, _wrapped)
        patched += 1
    cls._pams_boost_retuned = patched == len(targets)
    return cls._pams_boost_retuned


if __name__ == "__main__":
    import logging as _logging

    # INFO (not WARNING): per-instance proof that observers autostarted must
    # be visible in the launcher log. MCP stdio rides on stdout; the logging
    # module writes to stderr, so this cannot corrupt the protocol.
    # Audit 2026-10-29: without this, "File monitoring autostarted" was
    # suppressed and a live server looked identical to a dead monitor.
    # basicConfig() is a no-op when another module already attached a
    # handler (ai_memory_core/mcp do at import), which silently kept the
    # WARNING level. Force the root level + ensure a stderr handler instead.
    _root = _logging.getLogger()
    _root.setLevel(_logging.INFO)
    if not _root.handlers:
        _handler = _logging.StreamHandler(sys.stderr)
        _handler.setFormatter(
            _logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"
            )
        )
        _root.addHandler(_handler)
    # Surface BEFORE healing: healing rewrites our own record to "stale", and a
    # record that already admits it stopped is no longer worth a warning.
    _surface_stale_records()  # make a downed daemon visible in the logs
    _heal_stale_record()      # correct a record left behind by a dead process
    _start_heartbeat()        # keep our own record provably live
    # The pmb daemon is a Windows venv binary, so only a Windows session can own
    # it. Started in the background; see _keep_pmb_warm for why it is not inline.
    if os.name == "nt" and os.environ.get("PAMS_PMB_AUTOSTART", "1") != "0":
        threading.Thread(target=_keep_pmb_warm, name="pmb-autostart",
                         daemon=True).start()
    # Same reasoning for the embedder: PAMS's EmbeddingService needs
    # http://127.0.0.1:8080/v1/embeddings, and nothing else starts it.
    if os.name == "nt" and os.environ.get("PAMS_EMBED_AUTOSTART", "1") != "0":
        threading.Thread(target=_keep_embedding_warm, name="embedding-autostart",
                         daemon=True).start()
    # The relevance floor lives in a proxy in front of the daemon (pmb-ai has
    # no min-score setting and no newer release carries one). Keep it warm, or
    # every pmb tool goes dark the moment it stops listening. The watcher was
    # already started before the heavy imports above -- see _start_floor_watch.
    _start_floor_watch()
    _install_health_tool()    # publish the daemon/audit verdict to MCP clients
    _install_identity_defaults()  # every tool callable without an identity
    _install_health_alias_fix()  # verdict reads the portfile's real schema
    _install_caller_identity()  # keep the caller's identity on its own calls
    _install_monitoring_autostart()  # persistence: observers actually start
    _install_promote_guard()  # inflation guard: bus telemetry demoted to 2
    _install_boost_retune()  # importance breaks ties, similarity decides
    sys.exit(mcp_server.run_main())
