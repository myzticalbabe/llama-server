"""Tests for the MCP daemon health/heartbeat stack.

Covers three modules that live in this directory:

  * pmb_daemon_health   - record classification + the CLI contract
  * pams_server_launcher - stale-record healing, heartbeat, startup surfacing
  * pmb_daemon_launcher  - registry pruning, heartbeat stamping, ensure/supervise

Run either way:
    pytest test_daemon_health.py -q
    python test_daemon_health.py

Nothing here touches the real registry: every test passes an explicit pmb home
or monkeypatches PORT_FILE, so the live ~/.pmb is only ever read by the operator.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pmb_daemon_health as health  # noqa: E402
import pmb_daemon_launcher as pmb_launcher  # noqa: E402


# --- module under test: the pams launcher ------------------------------------
# PAMS_ROOT must be set before the module executes, because it resolves its
# database paths at import time (and would otherwise mkdir a Windows-style path).

def _load_pams_launcher():
    root = Path(tempfile.mkdtemp(prefix="pams-test-"))
    os.environ["PAMS_ROOT"] = str(root)
    os.environ["PAMS_HEARTBEAT_INTERVAL"] = "0.2"

    core = types.ModuleType("ai_memory_core")
    core.get_base_path = lambda: str(root)
    # The launcher does not only rebind `get_base_path`: it also installs an
    # `ai_memories_db` alias ON the AIMemorySystem class (the installed package
    # defines `ai_memory_db`, SINGULAR, and database_maintenance.py calls the
    # plural name). The double therefore has to carry that class or collection
    # dies with AttributeError before a single test runs.
    class AIMemorySystem:
        def __init__(self):
            self.ai_memory_db = None

    core.AIMemorySystem = AIMemorySystem
    sys.modules["ai_memory_core"] = core

    pkg = types.ModuleType("persistent_ai_memory")
    pkg.__path__ = []
    srv = types.ModuleType("persistent_ai_memory.mcp_server")
    srv.BASE_PATH = None
    srv.run_main = lambda: 0
    pkg.mcp_server = srv
    sys.modules["persistent_ai_memory"] = pkg
    sys.modules["persistent_ai_memory.mcp_server"] = srv

    spec = importlib.util.spec_from_file_location(
        "pams_server_launcher", HERE / "pams_server_launcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PAMS = _load_pams_launcher()
_DEAD_PID = 999_999  # nothing is ever this pid in practice


@pytest.fixture
def listening_port():
    """A port that is genuinely accepting connections for the test's duration."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(5)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def _closed_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _write_servers(home: Path, entries) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "servers.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _isolate_audit(tmp_path, monkeypatch):
    """Never let a test write into the operator's real .audit trail."""
    monkeypatch.setenv("AZAZAEL_AUDIT_DIR", str(tmp_path / "audit"))
    yield


# =============================================================================
# pmb_daemon_health
# =============================================================================

def test_pid_alive_current_process_true_and_dead_pid_false():
    assert health.pid_alive(os.getpid()) is True
    assert health.pid_alive(_DEAD_PID) is False


@pytest.mark.parametrize("pid", [None, 0, -1, "123"])
def test_pid_alive_rejects_non_pids(pid):
    assert health.pid_alive(pid) is False


def test_port_open_true_for_listening_false_for_closed(listening_port):
    assert health.port_open("127.0.0.1", listening_port, 1.0) is True
    assert health.port_open("127.0.0.1", _closed_port(), 0.5) is False


def test_check_all_live_record_is_ok(tmp_path, listening_port):
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": os.getpid(), "host": "127.0.0.1",
                           "port": listening_port}])
    results = health.check_all(pmb_home=home)
    assert len(results) == 1
    assert results[0]["status"] == "OK"
    assert results[0]["pid_alive"] is True and results[0]["port_open"] is True


def test_check_all_flags_stale_pid_and_port(tmp_path):
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": _DEAD_PID, "host": "127.0.0.1",
                           "port": _closed_port(), "status": "active"}])
    result = health.check_all(pmb_home=home)[0]
    assert result["status"] == "STALE"
    assert any("not running" in r for r in result["reasons"])
    assert any("nothing listening" in r for r in result["reasons"])


def test_check_all_pams_record_inactive_is_not_stale(tmp_path):
    home = tmp_path / "empty"
    home.mkdir()
    port_file = tmp_path / "mcp_server_port.json"
    port_file.write_text(json.dumps({"status": "inactive", "pid": _DEAD_PID,
                                     "port": _closed_port()}), encoding="utf-8")
    result = health.check_all(pmb_home=home, pams_port_file=port_file)[0]
    assert result["status"] == "INACTIVE"
    assert result["origin"] == "pams_port_file"


def test_heartbeat_age_is_reported_but_not_decisive(tmp_path, listening_port):
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": os.getpid(), "host": "127.0.0.1",
                           "port": listening_port,
                           "heartbeat_epoch": time.time() - 30}])
    result = health.check_all(pmb_home=home)[0]
    assert result["status"] == "OK"           # a stale heartbeat is not "down"
    assert result["heartbeat_age_seconds"] == pytest.approx(30, abs=5)


def test_cli_exit_codes(tmp_path):
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": _DEAD_PID, "port": _closed_port(),
                           "status": "active"}])
    assert health.main(["--pmb-home", str(home)]) == health.EXIT_STALE

    _write_servers(home, [{"pid": os.getpid(), "host": "127.0.0.1",
                           "port": _closed_port()}])
    # pid alive but nothing listening -> still stale
    assert health.main(["--pmb-home", str(home)]) == health.EXIT_STALE
    # missing registry -> unreadable
    assert health.main(["--pmb-home", str(tmp_path / "absent")]) == health.EXIT_ERROR


def test_cli_json_payload(tmp_path, capsys):
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": _DEAD_PID, "port": _closed_port(),
                           "status": "active"}])
    assert health.main(["--pmb-home", str(home), "--json"]) == health.EXIT_STALE
    payload = json.loads(capsys.readouterr().out)
    assert payload["stale"] == 1 and payload["checked"] == 1


# =============================================================================
# pams_server_launcher - healing and heartbeat
# =============================================================================

def test_heal_rewrites_active_record_left_by_dead_pid(tmp_path, monkeypatch):
    port_file = tmp_path / "mcp_server_port.json"
    monkeypatch.setattr(PAMS, "PORT_FILE", port_file)
    port_file.write_text(json.dumps({"status": "active", "pid": _DEAD_PID}),
                         encoding="utf-8")

    PAMS._heal_stale_record()

    data = json.loads(port_file.read_text(encoding="utf-8"))
    assert data["status"] == "stale"
    assert "not running" in data["stale_reason"]


def test_heal_leaves_a_live_record_alone(tmp_path, monkeypatch):
    port_file = tmp_path / "mcp_server_port.json"
    monkeypatch.setattr(PAMS, "PORT_FILE", port_file)
    port_file.write_text(json.dumps({"status": "active", "pid": os.getpid()}),
                         encoding="utf-8")

    PAMS._heal_stale_record()

    assert json.loads(port_file.read_text(encoding="utf-8"))["status"] == "active"


def test_heartbeat_stamps_then_liveness_holds(tmp_path, monkeypatch):
    port_file = tmp_path / "mcp_server_port.json"
    monkeypatch.setattr(PAMS, "PORT_FILE", port_file)
    port_file.write_text(json.dumps({"status": "active", "pid": os.getpid()}),
                         encoding="utf-8")
    PAMS._stop_heartbeat.clear()
    PAMS._start_heartbeat()
    try:
        deadline = time.time() + 5
        data = {}
        while time.time() < deadline:
            data = json.loads(port_file.read_text(encoding="utf-8"))
            if data.get("heartbeat_epoch"):
                break
            time.sleep(0.05)
        assert data.get("heartbeat_epoch"), "heartbeat never stamped the record"
        assert data["pid_alive"] is True
        assert data["heartbeat_at"]
        assert PAMS.is_record_live() is True
    finally:
        PAMS._stop_heartbeat.set()


def test_is_record_live_rejects_dead_pid_and_inactive(tmp_path, monkeypatch):
    port_file = tmp_path / "mcp_server_port.json"
    monkeypatch.setattr(PAMS, "PORT_FILE", port_file)
    fresh = time.time()

    port_file.write_text(json.dumps({"status": "active", "pid": _DEAD_PID,
                                     "heartbeat_epoch": fresh}), encoding="utf-8")
    assert PAMS.is_record_live() is False

    port_file.write_text(json.dumps({"status": "inactive", "pid": os.getpid(),
                                     "heartbeat_epoch": fresh}), encoding="utf-8")
    assert PAMS.is_record_live() is False

    port_file.write_text(json.dumps({"status": "active", "pid": os.getpid(),
                                     "heartbeat_epoch": fresh}), encoding="utf-8")
    assert PAMS.is_record_live() is True


def test_is_record_live_falls_back_to_started_at(tmp_path, monkeypatch):
    port_file = tmp_path / "mcp_server_port.json"
    monkeypatch.setattr(PAMS, "PORT_FILE", port_file)
    port_file.write_text(json.dumps({"status": "active", "pid": os.getpid(),
                                     "started_at": time.time()}), encoding="utf-8")
    assert PAMS.is_record_live() is True


def test_surface_stale_records_logs_both_sources(tmp_path, monkeypatch, capsys):
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": _DEAD_PID, "host": "127.0.0.1",
                           "port": _closed_port(), "status": "active"}])
    monkeypatch.setenv("PMB_HOME", str(home))

    port_file = tmp_path / "mcp_server_port.json"
    monkeypatch.setattr(PAMS, "PORT_FILE", port_file)
    port_file.write_text(json.dumps({"status": "active", "pid": _DEAD_PID,
                                     "port": _closed_port()}), encoding="utf-8")

    PAMS._surface_stale_records()

    err = capsys.readouterr().err
    assert "WARNING stale servers.json" in err
    assert "WARNING stale pams_port_file" in err


def test_exit_hook_downgrades_and_halts_heartbeat(tmp_path, monkeypatch):
    port_file = tmp_path / "mcp_server_port.json"
    monkeypatch.setattr(PAMS, "PORT_FILE", port_file)
    port_file.write_text(json.dumps({"status": "active", "pid": os.getpid()}),
                         encoding="utf-8")

    PAMS._mark_port_file_inactive()

    data = json.loads(port_file.read_text(encoding="utf-8"))
    assert data["status"] == "inactive" and data["stopped_at"]
    assert PAMS._stop_heartbeat.is_set()
    PAMS._stop_heartbeat.clear()  # leave global state usable for other tests


def test_keep_pmb_warm_anchors_start_gone_and_restart(tmp_path, monkeypatch):
    """The warm loop traces lifecycle EDGES, and anchors each one.

    Scripted liveness: the daemon starts, dies, is restarted, then the script is
    exhausted and the loop is stopped with an exception (which the loop swallows
    by design). Each edge must land in the tamper-evident sidecar exactly once.
    """
    home = tmp_path / "pmbhome"
    home.mkdir()

    monkeypatch.setattr(pmb_launcher.health, "find_pmb_home",
                        lambda *a, **k: home)
    monkeypatch.setattr(pmb_launcher, "stamp_heartbeat", lambda h, now=None: 1)
    monkeypatch.setattr(pmb_launcher, "DEFAULT_INTERVAL", 0)
    monkeypatch.setattr(pmb_launcher, "ensure_running", lambda *a, **k: "started")

    script = iter([False, True, False, True])   # was_live | started | gone | restart

    def fake_live(h, timeout=1.0):
        try:
            return next(script)
        except StopIteration:
            raise RuntimeError("test: scripted liveness exhausted")

    monkeypatch.setattr(pmb_launcher, "daemon_live", fake_live)

    PAMS._keep_pmb_warm()    # raises internally when the script runs out

    audit = tmp_path / "audit" / pmb_launcher.AUDIT_FILENAME
    types = [json.loads(x)["type"]
             for x in audit.read_text(encoding="utf-8").splitlines()]
    assert types == ["pmb_daemon_started_by_pams",
                     "pmb_daemon_gone_by_pams",
                     "pmb_daemon_restarted_by_pams"]
    # Anchored, not just audited: every transition published a head hash.
    anchors = tmp_path / "audit" / pmb_launcher.ANCHORS_FILENAME
    assert len([x for x in anchors.read_text(encoding="utf-8").splitlines()
                if x.strip()]) == 3
    assert pmb_launcher.verify_audit_chain()[0] is True
    assert pmb_launcher.verify_against_anchors()[0] is True


def test_keep_pmb_warm_records_no_edge_when_already_live(tmp_path, monkeypatch):
    """A daemon that is already up is not an event, and healthy ticks are silent."""
    home = tmp_path / "pmbhome"
    home.mkdir()

    monkeypatch.setattr(pmb_launcher.health, "find_pmb_home",
                        lambda *a, **k: home)
    monkeypatch.setattr(pmb_launcher, "stamp_heartbeat", lambda h, now=None: 1)
    monkeypatch.setattr(pmb_launcher, "DEFAULT_INTERVAL", 0)

    script = iter([True, True, True])      # was_live | tick | exhausted

    def fake_live(h, timeout=1.0):
        try:
            return next(script)
        except StopIteration:
            raise RuntimeError("test: scripted liveness exhausted")

    monkeypatch.setattr(pmb_launcher, "daemon_live", fake_live)

    PAMS._keep_pmb_warm()

    audit = tmp_path / "audit" / pmb_launcher.AUDIT_FILENAME
    assert not audit.exists(), "a healthy warm loop must write no events"


def test_daemon_health_payload_flags_a_stale_daemon(tmp_path, monkeypatch):
    """The MCP-visible verdict agrees with the health checker it wraps."""
    home = tmp_path / "pmbhome"
    _write_servers(home, [{"pid": _DEAD_PID, "kind": "daemon",
                           "host": "127.0.0.1", "port": 8765,
                           "status": "active"}])
    monkeypatch.setenv("PMB_HOME", str(home))

    payload = PAMS._daemon_health_payload(timeout=0.3)

    assert payload["ok"] is False          # a stale record must not read as OK
    assert payload["stale"] == 1
    assert payload["audit"]["chain_ok"] is True
    assert isinstance(payload["audit"]["merkle_root"], str)
    # The client-facing verdict is explicit, not left for the caller to infer.
    assert payload["severity"] == "critical"
    assert "stale" in payload["alert"]
    assert payload["heals_total"] == 0


def test_health_tool_alert_pushes_when_stale(tmp_path, monkeypatch):
    """A stale verdict is PUSHED to the client, not just returned."""
    import asyncio
    from types import SimpleNamespace

    _install_fake_mcp(monkeypatch)
    home = tmp_path / "pmbhome"
    _write_servers(home, [{"pid": _DEAD_PID, "kind": "daemon",
                           "host": "127.0.0.1", "port": 8765,
                           "status": "active"}])
    monkeypatch.setenv("PMB_HOME", str(home))

    sent: list[dict] = []

    class Session:
        async def send_log_message(self, **kw):
            sent.append(kw)

    class FakeServer:
        # Mirrors the SDK shape: server.server.request_context.session
        server = SimpleNamespace(
            request_context=SimpleNamespace(session=Session()))

        async def _get_client_tools(self):
            return []

        async def _execute_tool(self, name, arguments):
            return ("orig", name)

    monkeypatch.setattr(PAMS.mcp_server, "AIMemoryMCPServer", FakeServer,
                        raising=False)
    assert PAMS._install_health_tool() is True

    result = asyncio.run(FakeServer()._execute_tool(PAMS.HEALTH_TOOL_NAME, {}))

    assert result["isError"] is True
    assert '"severity": "critical"' in result["content"][0]["text"]
    assert sent and sent[0]["level"] == "error"
    assert sent[0]["logger"] == "pams.daemon_health"
    assert "stale" in sent[0]["data"]


def test_alert_is_silent_for_a_healthy_payload():
    """A healthy verdict must not emit a client alert."""
    import asyncio
    from types import SimpleNamespace

    calls: list[dict] = []
    session = SimpleNamespace(send_log_message=lambda **kw: calls.append(kw))
    # self.server.request_context.session -- the shape the SDK exposes.
    server = SimpleNamespace(server=SimpleNamespace(
        request_context=SimpleNamespace(session=session)))

    asyncio.run(PAMS._alert_if_unhealthy(server, {"severity": "ok"}))

    assert calls == []


# --- a minimal mcp.types stub, so the wrapping test needs no mcp install ---

class _FakeTool:
    def __init__(self, name, description, inputSchema):
        self.name, self.description, self.inputSchema = name, description, inputSchema


class _FakeTextContent:
    def __init__(self, type, text):
        self.type, self.text = type, text


class _FakeCallToolResult:
    def __init__(self, content, isError=False):
        self.content, self.isError = content, isError


def _install_fake_mcp(monkeypatch):
    mcp = types.ModuleType("mcp")
    mcp_types = types.ModuleType("mcp.types")
    mcp_types.Tool = _FakeTool
    mcp_types.TextContent = _FakeTextContent
    mcp_types.CallToolResult = _FakeCallToolResult
    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.types", mcp_types)


def test_install_health_tool_adds_and_dispatches(tmp_path, monkeypatch):
    """daemon_health is added to the tool list and routed without touching
    the package: a foreign tool still reaches the original handler."""
    import asyncio

    _install_fake_mcp(monkeypatch)
    monkeypatch.setenv("PMB_HOME", str(tmp_path / "pmbhome"))

    class FakeServer:
        async def _get_client_tools(self):
            return ["base"]

        async def _execute_tool(self, name, arguments):
            return ("orig", name)

    monkeypatch.setattr(PAMS.mcp_server, "AIMemoryMCPServer", FakeServer,
                        raising=False)

    assert PAMS._install_health_tool() is True

    server = FakeServer()
    tools = asyncio.run(server._get_client_tools())
    assert "base" in tools                        # the original list survives
    assert any(getattr(t, "name", None) == PAMS.HEALTH_TOOL_NAME for t in tools)

    result = asyncio.run(server._execute_tool(PAMS.HEALTH_TOOL_NAME, {}))
    # Native-dict shape: the low-level SDK only accepts dicts from handlers --
    # a CallToolResult object iterates as (field, value) tuples and breaks
    # clients (20 validation errors, Oct 2026). Never return the object here.
    assert isinstance(result, dict)
    assert result["content"][0]["type"] == "text"
    assert '"records"' in result["content"][0]["text"]
    assert result["isError"] in (True, False)

    # Everything else must still be handled by the original method.
    assert asyncio.run(server._execute_tool("other", {})) == ("orig", "other")


def test_install_health_tool_is_safe_without_the_server_api(tmp_path, monkeypatch):
    """No AIMemoryMCPServer (or no mcp) must not raise -- it just no-ops."""
    _install_fake_mcp(monkeypatch)
    monkeypatch.delattr(PAMS.mcp_server, "AIMemoryMCPServer", raising=False)
    assert PAMS._install_health_tool() is False


# =============================================================================
# pmb_daemon_launcher - registry hygiene, heartbeat, ensure/supervise
# =============================================================================

def test_prune_dead_removes_only_dead_entries(tmp_path):
    home = tmp_path / "pmb"
    _write_servers(home, [
        {"pid": _DEAD_PID, "kind": "daemon", "port": 8765},
        {"pid": os.getpid(), "kind": "mcp", "port": 1234},
    ])

    assert pmb_launcher.prune_dead(home) == 1

    entries = pmb_launcher.load_entries(home)
    assert [e["pid"] for e in entries] == [os.getpid()]


def test_stamp_heartbeat_marks_live_entries(tmp_path):
    home = tmp_path / "pmb"
    _write_servers(home, [
        {"pid": os.getpid(), "kind": "daemon", "port": 8765},
        {"pid": _DEAD_PID, "kind": "mcp", "port": 1},
    ])

    assert pmb_launcher.stamp_heartbeat(home, now=1_000_000.0) == 1

    entry = pmb_launcher.load_entries(home)[0]
    assert entry["pid_alive"] is True
    assert entry["heartbeat_epoch"] == 1_000_000.0
    assert entry["heartbeat_at"]


def test_daemon_live_needs_both_a_pid_and_an_open_port(tmp_path, listening_port):
    home = tmp_path / "pmb"

    _write_servers(home, [{"pid": os.getpid(), "kind": "daemon",
                           "host": "127.0.0.1", "port": listening_port}])
    assert pmb_launcher.daemon_live(home) is True

    _write_servers(home, [{"pid": _DEAD_PID, "kind": "daemon",
                           "host": "127.0.0.1", "port": listening_port}])
    assert pmb_launcher.daemon_live(home) is False

    # pid alive, but the port is not answering -> not live
    _write_servers(home, [{"pid": os.getpid(), "kind": "daemon",
                           "host": "127.0.0.1", "port": _closed_port()}])
    assert pmb_launcher.daemon_live(home) is False


def test_daemon_live_ignores_non_daemon_entries(tmp_path, listening_port):
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": os.getpid(), "kind": "mcp",
                           "host": "127.0.0.1", "port": listening_port}])
    assert pmb_launcher.daemon_live(home) is False


@pytest.mark.skipif(os.name == "nt", reason="explicitly the non-Windows branch")
def test_ensure_running_skips_off_windows(tmp_path):
    home = tmp_path / "pmb"
    home.mkdir()
    assert pmb_launcher.ensure_running(home) == "skipped-non-windows"


def test_ensure_running_reports_already_live(tmp_path, listening_port):
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": os.getpid(), "kind": "daemon",
                           "host": "127.0.0.1", "port": listening_port}])
    assert pmb_launcher.ensure_running(home) == "already-live"


def test_supervise_stamps_then_stops_after_max_ticks(tmp_path, listening_port):
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": os.getpid(), "kind": "daemon",
                           "host": "127.0.0.1", "port": listening_port}])
    logged: list[str] = []

    code = pmb_launcher.supervise(home, interval=0.05, timeout=0.5,
                                  max_ticks=2, log=logged.append)

    assert code == pmb_launcher.EXIT_OK
    assert any("already-live" in line for line in logged)
    assert pmb_launcher.load_entries(home)[0]["pid_alive"] is True


def test_supervise_without_restart_reports_down(tmp_path):
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": _DEAD_PID, "kind": "daemon",
                           "host": "127.0.0.1", "port": _closed_port()}])
    logged: list[str] = []

    code = pmb_launcher.supervise(home, interval=0.05, timeout=0.2,
                                  restart=False, max_ticks=1, log=logged.append)

    assert code == pmb_launcher.EXIT_DOWN
    assert any("gone" in line for line in logged)


def test_status_reports_registration_and_liveness(tmp_path, listening_port):
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": os.getpid(), "kind": "daemon",
                           "host": "127.0.0.1", "port": listening_port,
                           "started_at": time.time() - 10}])
    info = pmb_launcher.status(home)
    assert info["registered"] is True
    assert info["live"] is True and info["pid_alive"] is True
    assert info["age_seconds"] == pytest.approx(10, abs=5)


def test_cli_status_exit_code_and_json(tmp_path, capsys):
    down = tmp_path / "down"
    _write_servers(down, [{"pid": _DEAD_PID, "kind": "daemon",
                           "host": "127.0.0.1", "port": _closed_port()}])
    assert pmb_launcher.main(["--pmb-home", str(down), "--json"]) == \
        pmb_launcher.EXIT_DOWN
    assert json.loads(capsys.readouterr().out)["live"] is False


@pytest.mark.skipif(os.name == "nt", reason="explicitly the non-Windows branch")
def test_cli_ensure_skipped_is_success_not_failure(tmp_path, capsys):
    home = tmp_path / "pmb"
    home.mkdir()
    assert pmb_launcher.main(["--pmb-home", str(home), "--ensure"]) == \
        pmb_launcher.EXIT_OK
    assert "skipped-non-windows" in capsys.readouterr().out


def test_stamp_subcommand_is_quiet_success(tmp_path, capsys):
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": os.getpid(), "port": 1}])
    assert pmb_launcher.main(["--pmb-home", str(home), "--stamp"]) == \
        pmb_launcher.EXIT_OK
    assert "stamped 1 live record" in capsys.readouterr().out


# =============================================================================
# pmb_daemon_launcher - audit sidecar
# =============================================================================

def test_audit_sidecar_filename_is_outside_the_session_chain():
    """The whole point: a name no consumer's `session_*.jsonl` glob can match."""
    import fnmatch
    assert not fnmatch.fnmatch(pmb_launcher.AUDIT_FILENAME, "session_*.jsonl")


def test_audit_event_appends_a_verifiable_chain(tmp_path):
    assert pmb_launcher.audit_event("pmb_daemon_started", port=8765) is not None
    assert pmb_launcher.audit_event("pmb_daemon_gone", stamped=1) is not None

    path = tmp_path / "audit" / pmb_launcher.AUDIT_FILENAME
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [e["type"] for e in lines] == ["pmb_daemon_started", "pmb_daemon_gone"]
    assert lines[0]["prev_hash"] == "0" * 64
    assert lines[1]["prev_hash"] == lines[0]["hash"]

    ok, detail = pmb_launcher.verify_audit_chain()
    assert ok is True
    assert "2 entries" in detail


def test_audit_chain_detects_tampering(tmp_path):
    pmb_launcher.audit_event("pmb_daemon_started", port=8765)
    pmb_launcher.audit_event("pmb_registry_pruned", removed=3)
    path = tmp_path / "audit" / pmb_launcher.AUDIT_FILENAME

    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["port"] = 9999                      # edit a hashed field
    path.write_text("\n".join([json.dumps(first, sort_keys=True)] + lines[1:]) + "\n",
                    encoding="utf-8")

    ok, detail = pmb_launcher.verify_audit_chain()
    assert ok is False
    assert "line 1" in detail


def test_verify_audit_chain_missing_file_is_not_failure(tmp_path):
    ok, detail = pmb_launcher.verify_audit_chain(tmp_path / "audit" / "missing.jsonl")
    assert ok is True and "no audit sidecar" in detail


def test_prune_dead_is_audited_only_when_it_removes(tmp_path):
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": os.getpid()}])
    assert pmb_launcher.prune_dead(home) == 0     # nothing removed -> no record
    assert pmb_launcher.verify_audit_chain()[1].startswith("no audit sidecar")

    _write_servers(home, [{"pid": os.getpid()}, {"pid": _DEAD_PID}])
    assert pmb_launcher.prune_dead(home) == 1
    path = tmp_path / "audit" / pmb_launcher.AUDIT_FILENAME
    entry = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["type"] == "pmb_registry_pruned" and entry["removed"] == 1


def test_supervise_audits_the_gone_transition_once(tmp_path, listening_port):
    """A live then dead daemon must produce exactly one gone record."""
    home = tmp_path / "pmb"
    _write_servers(home, [{"pid": os.getpid(), "kind": "daemon",
                           "host": "127.0.0.1", "port": listening_port}])
    calls = {"n": 0}
    real_live = pmb_launcher.daemon_live

    def fake_live(h, timeout=1.0):
        calls["n"] += 1
        return calls["n"] <= 1          # live for the first look, gone after

    pmb_launcher.daemon_live = fake_live
    try:
        pmb_launcher.supervise(home, interval=0.05, timeout=0.2,
                               restart=False, max_ticks=3, log=lambda *_: None)
    finally:
        pmb_launcher.daemon_live = real_live

    path = tmp_path / "audit" / pmb_launcher.AUDIT_FILENAME
    types = [json.loads(x)["type"]
             for x in path.read_text(encoding="utf-8").splitlines()]
    assert types.count("pmb_daemon_gone") == 1


def test_cli_verify_audit_reports_ok(tmp_path, capsys):
    pmb_launcher.audit_event("pmb_daemon_started", port=1)
    assert pmb_launcher.main(["--verify-audit"]) == pmb_launcher.EXIT_OK
    out = capsys.readouterr().out
    assert "chain : True" in out and "anchor: True" in out


# --- anchors layer -----------------------------------------------------------

def test_anchor_publishes_head_in_the_repo_schema(tmp_path):
    pmb_launcher.audit_event("pmb_daemon_started", port=8765)
    pmb_launcher.audit_event("pmb_registry_pruned", removed=1)
    head = pmb_launcher.anchor_sidecar()
    assert head

    anchors = tmp_path / "audit" / pmb_launcher.ANCHORS_FILENAME
    record = json.loads(anchors.read_text(encoding="utf-8").splitlines()[-1])
    # Same five keys audit_trail.anchor_head_hash writes, plus a Merkle commitment.
    assert set(record) >= {"ts", "audit_file", "head_hash", "entry_count",
                           "session_uuid"}
    assert record["head_hash"] == head
    assert record["entry_count"] == 2
    assert record["merkle_root"] == pmb_launcher.merkle_root()
    assert pmb_launcher.verify_against_anchors()[0] is True


def test_anchor_is_a_checkpoint_so_later_entries_still_verify(tmp_path):
    pmb_launcher.audit_event("pmb_daemon_started", port=1)
    anchored = pmb_launcher._head_hash(
        tmp_path / "audit" / pmb_launcher.AUDIT_FILENAME)
    pmb_launcher.anchor_sidecar()

    pmb_launcher.audit_event("pmb_registry_pruned", removed=2)   # append after
    ok, detail = pmb_launcher.verify_against_anchors()
    assert ok is True, detail
    assert anchored[:16] in detail


def test_verify_against_anchors_catches_a_rewrite(tmp_path):
    pmb_launcher.audit_event("pmb_daemon_started", port=1)
    pmb_launcher.audit_event("pmb_daemon_gone", stamped=1)
    pmb_launcher.anchor_sidecar()
    path = tmp_path / "audit" / pmb_launcher.AUDIT_FILENAME
    assert pmb_launcher.verify_against_anchors()[0] is True

    # Drop the anchored head: records below the anchor were rewritten/removed.
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    ok, detail = pmb_launcher.verify_against_anchors()
    assert ok is False and "no longer in the chain" in detail


def test_anchor_reports_no_anchor_without_lying(tmp_path):
    pmb_launcher.audit_event("pmb_daemon_started", port=1)
    ok, detail = pmb_launcher.verify_against_anchors()
    assert ok is True and "no anchor" in detail


def test_merkle_root_matches_the_specified_tree_shape(tmp_path):
    def dsha(a: str, b: str) -> str:
        joined = bytes.fromhex(a) + bytes.fromhex(b)
        return hashlib.sha256(hashlib.sha256(joined).digest()).hexdigest()

    a, b, c = "ab" * 32, "cd" * 32, "ef" * 32
    assert pmb_launcher._compute_merkle_root([]) == ""
    assert pmb_launcher._compute_merkle_root([a]) == a          # single leaf
    assert pmb_launcher._compute_merkle_root([a, b]) == dsha(a, b)
    # an odd count duplicates the last leaf, so 3 leaves == the 4-leaf tree
    assert pmb_launcher._compute_merkle_root([a, b, c]) == \
        pmb_launcher._compute_merkle_root([a, b, c, c])


# --- interoperability with the repo's real trust layer -----------------------

REPO_SRC = Path("/root/azazael/src")


def _repo_audit_trail():
    """The azazael audit_trail module, or None when the repo isn't present."""
    if not REPO_SRC.is_dir():
        return None
    if str(REPO_SRC) not in sys.path:
        sys.path.insert(0, str(REPO_SRC))
    try:
        import CORE.audit_trail as audit_trail
        return audit_trail
    except Exception:
        return None


REPO_AT = _repo_audit_trail()


@pytest.mark.skipif(REPO_AT is None, reason="azazael CORE.audit_trail unavailable")
def test_sidecar_is_accepted_by_the_repos_own_verifiers(tmp_path, monkeypatch):
    """The real trust layer must accept our chain, anchor and Merkle root.

    audit_trail resolves .audit relative to the CWD, so the test runs inside
    tmp_path and points the sidecar at the same directory.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZAZAEL_AUDIT_DIR", str(tmp_path / ".audit"))

    pmb_launcher.audit_event("pmb_daemon_started", port=8765)
    pmb_launcher.audit_event("pmb_daemon_gone", stamped=1)
    pmb_launcher.anchor_sidecar()
    sidecar = tmp_path / ".audit" / pmb_launcher.AUDIT_FILENAME

    # 1. the repo's chain verifier accepts our entries
    chain_ok, error_line = REPO_AT.verify_chain(str(sidecar))
    assert chain_ok is True, "rejected at line %s" % error_line

    # 2. our Merkle root is byte-identical to the repo's construction
    hashes = [json.loads(x)["hash"]
              for x in sidecar.read_text(encoding="utf-8").splitlines()]
    assert pmb_launcher.merkle_root(sidecar) == \
        REPO_AT._compute_merkle_root(hashes)

    # 3. the SHARED anchors.jsonl is left alone. verify_head_hash() is
    #    first-match-per-file and compares against the file's current last
    #    hash, so anchoring a growing sidecar there would make the repo's own
    #    checker accuse intact data of tampering.
    shared = tmp_path / ".audit" / "anchors.jsonl"
    anchored_names = [Path(p).name for p, _h, _c in REPO_AT.get_anchored_files()]
    assert pmb_launcher.AUDIT_FILENAME not in anchored_names
    if shared.exists():
        assert "daemon_health" not in shared.read_text(encoding="utf-8")

    # 4. our own ledger is where the anchor lives, and it verifies
    ledger = tmp_path / ".audit" / pmb_launcher.ANCHORS_FILENAME
    assert ledger.exists()
    ok, detail = pmb_launcher.verify_against_anchors()
    assert ok is True, detail


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
