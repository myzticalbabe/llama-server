"""End-to-end test of the daemon stack, minus the real Windows daemon.

A real TCP listener stands in for the pmb daemon, and the whole chain is driven
end to end:

    registry (servers.json)
        -> pmb_daemon_launcher.supervise()   (heartbeat stamping)
        -> pmb_daemon_health.check_all()     (verdict)
        -> pams_server_launcher startup      (surfacing)
        -> .audit/daemon_health.jsonl        (traceability)

WHY A STAND-IN AND NOT THE REAL THING
-------------------------------------
The genuine Windows end-to-end run cannot be performed from this WSL host, for
two independent reasons:

  1. WSLInterop is not registered in /proc/sys/fs/binfmt_misc, so no Windows
     binary can be executed at all -- `pmb-ai.exe daemon start` fails with
     "Exec format error", as does cmd.exe.
  2. The daemon binds 127.0.0.1 on Windows only. Under NAT networking that
     loopback is not reachable from WSL (probed on 127.0.0.1, the default
     gateway and the DNS address: all refused). Reaching it needs
     `networkingMode=mirrored`, which only applies after `wsl --shutdown` --
     a command that would terminate the session doing the verification.

What this file proves is everything except the daemon process itself: given a
correct registry entry, the heartbeat, verdict, surfacing and audit paths all
behave, including the live -> dead transition. run_checks.py reports the
remaining Windows-side readiness as an advisory check.

Run:  pytest test_daemon_stack_integration.py -q
      python test_daemon_stack_integration.py
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE))

import pmb_daemon_health as health  # noqa: E402
import pmb_daemon_launcher as pmb_launcher  # noqa: E402
from test_daemon_health import PAMS  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("AZAZAEL_AUDIT_DIR", str(tmp_path / "audit"))
    yield


@pytest.fixture(autouse=True)
def _stub_terence(monkeypatch):
    """Never shell out to the ~20s oracle suite from a unit test."""
    import run_checks

    monkeypatch.setattr(
        run_checks, "check_terence",
        lambda strict: run_checks._result("terence oracles", True, "stub", strict),
        raising=False)
    yield


@pytest.fixture
def stand_in_daemon():
    """A listening socket that impersonates the daemon, and its registry entry."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(5)
    entry = {
        "pid": os.getpid(),          # a pid that really is alive
        "transport": "streamable-http",
        "kind": "daemon",
        "host": "127.0.0.1",
        "port": sock.getsockname()[1],
        "path": "/mcp",
        "workspace": "pmb",
        "started_at": time.time(),
    }
    yield sock, entry
    sock.close()


def _write_servers(home: Path, entries) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "servers.json").write_text(json.dumps(entries, indent=2),
                                       encoding="utf-8")


def test_live_daemon_is_ok_then_stale_when_it_dies(tmp_path, monkeypatch,
                                                   stand_in_daemon):
    """The whole chain, across the live -> dead transition."""
    sock, entry = stand_in_daemon
    home = tmp_path / "pmbhome"
    _write_servers(home, [entry])
    monkeypatch.setenv("PMB_HOME", str(home))

    # --- 1. supervisor sees it live and stamps a heartbeat ------------------
    logged: list[str] = []
    assert pmb_launcher.supervise(home, interval=0.05, timeout=0.5,
                                  max_ticks=1, log=logged.append) == \
        pmb_launcher.EXIT_OK
    assert any("already-live" in line for line in logged)

    stamped = pmb_launcher.load_entries(home)[0]
    assert stamped["pid_alive"] is True and stamped["heartbeat_epoch"]

    # --- 2. the health check agrees ----------------------------------------
    results = health.check_all(pmb_home=home)
    assert [r["status"] for r in results] == ["OK"]
    assert results[0]["port"] == entry["port"]

    # --- 3. pams startup stays silent, because nothing is stale ------------
    pams_port_file = tmp_path / "mcp_server_port.json"
    pams_port_file.write_text(json.dumps({"status": "inactive", "pid": 12345}),
                              encoding="utf-8")
    monkeypatch.setattr(PAMS, "PORT_FILE", pams_port_file)
    import contextlib
    import io
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        PAMS._surface_stale_records()
    assert "WARNING stale servers.json" not in buffer.getvalue()

    # --- 4. the daemon dies: port closed, pid still alive ------------------
    sock.close()

    after = health.check_all(pmb_home=home)[0]
    assert after["status"] == "STALE"
    assert after["pid_alive"] is True and after["port_open"] is False
    assert any("nothing listening" in reason for reason in after["reasons"])

    # --- 5. and now pams startup says so ----------------------------------
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        PAMS._surface_stale_records()
    assert "WARNING stale servers.json" in buffer.getvalue()

    # --- 6. the audit sidecar recorded the session and still verifies -----
    ok, detail = pmb_launcher.verify_audit_chain()
    assert ok is True, detail


def test_dead_daemon_entry_is_pruned_and_audited(tmp_path):
    """A registry left behind by a killed daemon is cleaned, not just reported."""
    home = tmp_path / "pmbhome"
    _write_servers(home, [{"pid": 999999, "kind": "daemon",
                           "host": "127.0.0.1", "port": 8765,
                           "started_at": time.time() - 3600}])

    assert pmb_launcher.stamp_heartbeat(home) == 0
    assert pmb_launcher.load_entries(home) == []

    audit = tmp_path / "audit" / pmb_launcher.AUDIT_FILENAME
    types = [json.loads(x)["type"]
             for x in audit.read_text(encoding="utf-8").splitlines()]
    assert types == ["pmb_registry_pruned"]
    assert pmb_launcher.verify_audit_chain()[0] is True


def test_supervise_restarts_a_daemon_that_died(tmp_path, monkeypatch):
    """With restart=True the supervisor tries to bring it back and traces it."""
    home = tmp_path / "pmbhome"
    _write_servers(home, [{"pid": 999999, "kind": "daemon",
                           "host": "127.0.0.1", "port": 8765}])

    attempts: list[str] = []
    real_ensure = pmb_launcher.ensure_running

    def fake_ensure(h, wait=8.0, timeout=1.0, executable=None):
        state = real_ensure(h, wait, timeout, executable)
        attempts.append(state)
        return state

    monkeypatch.setattr(pmb_launcher, "ensure_running", fake_ensure)
    pmb_launcher.supervise(home, interval=0.05, timeout=0.2,
                           max_ticks=2, log=lambda *_: None)

    # one attempt at startup, one after the daemon was found gone
    assert len(attempts) >= 2
    audit = tmp_path / "audit" / pmb_launcher.AUDIT_FILENAME
    types = [json.loads(x)["type"]
             for x in audit.read_text(encoding="utf-8").splitlines()]
    # This daemon was never live, so it is a down-at-start, not a live->gone
    # transition. The dead entry it left behind is pruned and traced.
    assert "pmb_daemon_down_at_start" in types
    assert "pmb_registry_pruned" in types
    assert pmb_launcher.verify_audit_chain()[0] is True


# =============================================================================
# run_checks.py --fix - explicit, opt-in healing of the daemon stack
# =============================================================================

def test_run_checks_fix_prunes_then_ensures_then_reports(tmp_path, monkeypatch):
    """--fix applies pmb's own rules: prune dead entries, then ensure_running."""
    import run_checks

    home = tmp_path / "pmbhome"
    home.mkdir()
    monkeypatch.setattr(run_checks.health, "find_pmb_home",
                        lambda *a, **k: home)

    calls: list[str] = []

    def fake_prune(h):
        calls.append("prune")
        return 2                      # two dead entries removed

    def fake_ensure(h, *a, **k):
        calls.append("ensure")
        return "started"

    monkeypatch.setattr(run_checks.pmb_launcher, "prune_dead", fake_prune)
    monkeypatch.setattr(run_checks.pmb_launcher, "ensure_running", fake_ensure)
    monkeypatch.setattr(run_checks.pmb_launcher, "daemon_live",
                        lambda *a, **k: True)

    actions = run_checks.attempt_fix()

    assert calls == ["prune", "ensure"]     # order matters: heal, then bring up
    assert all(a["ok"] for a in actions)
    assert all(not a["required"] for a in actions)   # fix actions never gate
    assert "removed 2 dead" in actions[0]["detail"]
    assert "live=True" in actions[1]["detail"]


def test_run_checks_fix_is_opt_in(monkeypatch):
    """Without --fix nothing in the environment is touched."""
    import run_checks

    called: list[bool] = []
    monkeypatch.setattr(run_checks, "check_compile",
                        lambda: run_checks._result("compile", True, "stub", True))
    monkeypatch.setattr(run_checks, "environment_checks", lambda strict: [])
    monkeypatch.setattr(run_checks, "attempt_fix",
                        lambda: (called.append(True),
                                 [run_checks._result("fix: prune", True, "x", False)])[1])

    assert run_checks.main(["--no-tests", "--fix"]) == 0
    assert called == [True]

    called.clear()
    assert run_checks.main(["--no-tests"]) == 0
    assert called == []                     # default run must not heal anything


def test_run_checks_watch_fix_heals_then_reports_post_fix_state(monkeypatch,
                                                                capsys):
    """--watch --fix: a failing poll is healed, and the report shows the result."""
    import run_checks

    # Two environment reads per tick when fix is on: the failure, then the
    # post-fix state. The third call ends the watch loop.
    reads = [
        [run_checks._result("daemon health", False, "down", False)],
        [run_checks._result("daemon health", True, "live", False)],
    ]
    calls = {"n": 0}
    fixed: list[bool] = []

    def fake_env(strict):
        if calls["n"] >= len(reads):
            raise KeyboardInterrupt
        out = reads[calls["n"]]
        calls["n"] += 1
        return out

    monkeypatch.setattr(run_checks, "environment_checks", fake_env)
    monkeypatch.setattr(run_checks, "attempt_fix",
                        lambda: (fixed.append(True),
                                 [run_checks._result("fix: ensure", True, "ok", False)])[1])
    monkeypatch.setattr(run_checks, "check_compile",
                        lambda: run_checks._result("compile", True, "stub", True))
    monkeypatch.setattr(run_checks.time, "sleep", lambda _s: None)

    assert run_checks.watch(0.01, fix=True) == 0
    assert fixed == [True]                  # the failing poll was healed
    out = capsys.readouterr().out
    assert "watch stopped" in out
    assert "fix: ensure" in out             # and the fix is visible in the report
    assert "daemon health" in out


def test_run_checks_watch_fix_cooldown_throttles_repeats(monkeypatch, capsys):
    """A failure that persists is not re-healed on every single poll."""
    import run_checks

    def fail():
        return run_checks._result("daemon health", False, "down", False)

    reads = [[fail()], [fail()], [fail()], [fail()]]
    calls = {"n": 0}
    fixes: list[int] = []

    def fake_env(strict):
        if calls["n"] >= len(reads):
            raise KeyboardInterrupt
        out = reads[calls["n"]]
        calls["n"] += 1
        return out

    monkeypatch.setattr(run_checks, "environment_checks", fake_env)
    monkeypatch.setattr(run_checks, "attempt_fix",
                        lambda: (fixes.append(1),
                                 [run_checks._result("fix: ensure", True, "ok", False)])[1])
    monkeypatch.setattr(run_checks, "check_compile",
                        lambda: run_checks._result("compile", True, "stub", True))
    monkeypatch.setattr(run_checks, "check_tests",
                        lambda: run_checks._result("pytest suite", True, "stub", True))
    monkeypatch.setattr(run_checks.time, "sleep", lambda _s: None)

    # The first failing poll heals at once; the cooldown swallows every repeat.
    assert run_checks.watch(0.01, fix=True, cooldown=10_000.0) == 0
    assert len(fixes) == 1


def test_heal_attempts_are_persisted_and_visible(tmp_path):
    """Every heal attempt lands in the tamper-evident sidecar, not just stdout."""
    import run_checks

    run_checks.record_fix_attempt("fresh-failure",
                                  {"daemon health": False},
                                  {"daemon health": True})
    run_checks.record_fix_attempt("cooldown-retry",
                                  {"daemon health": False},
                                  {"daemon health": False})

    history = pmb_launcher.heal_history()
    assert [h["reason"] for h in history] == ["fresh-failure", "cooldown-retry"]
    assert history[0]["failing"] == ["daemon health"]
    assert history[0]["resolved"] == ["daemon health"]
    assert history[1]["still_failing"] == ["daemon health"]
    assert pmb_launcher.verify_audit_chain()[0] is True


def test_watch_fix_records_a_heal_event(monkeypatch):
    """The watch loop persists the heal it performed (via record_fix_attempt)."""
    import run_checks

    reads = [[run_checks._result("daemon health", False, "down", False)],
             [run_checks._result("daemon health", True, "up", False)]]
    calls = {"n": 0}

    def fake_env(strict):
        if calls["n"] >= len(reads):
            raise KeyboardInterrupt
        out = reads[calls["n"]]
        calls["n"] += 1
        return out

    monkeypatch.setattr(run_checks, "environment_checks", fake_env)
    monkeypatch.setattr(run_checks, "attempt_fix",
                        lambda: [run_checks._result("fix: ensure", True, "ok", False)])
    monkeypatch.setattr(run_checks, "check_compile",
                        lambda: run_checks._result("compile", True, "stub", True))
    monkeypatch.setattr(run_checks, "check_tests",
                        lambda: run_checks._result("pytest suite", True, "stub", True))
    monkeypatch.setattr(run_checks.time, "sleep", lambda _s: None)

    assert run_checks.watch(0.01, fix=True) == 0
    history = pmb_launcher.heal_history()
    assert len(history) == 1
    assert history[0]["reason"] == "fresh-failure"


def test_check_audit_flags_flapping_after_repeated_heals(monkeypatch):
    """Repeated heals in the window make the audit check fail as flapping."""
    import run_checks

    for _ in range(run_checks.FLAP_THRESHOLD):
        pmb_launcher.audit_event("watch_fix_applied", reason="retry",
                                 epoch=time.time())

    result = run_checks.check_audit(strict=False)
    assert result["ok"] is False
    assert "FLAPPING" in result["detail"]

    # Below the threshold the same history is not treated as flapping.
    monkeypatch.setattr(run_checks, "FLAP_THRESHOLD", 99)
    assert run_checks.check_audit(strict=False)["ok"] is True


def test_run_checks_wait_for_ready_carries_fix(monkeypatch, capsys):
    """--wait-for-ready --fix heals in the gated run it recurses into."""
    import run_checks

    fixed: list[int] = []
    monkeypatch.setattr(run_checks, "wait_for_ready", lambda i, t: True)
    monkeypatch.setattr(run_checks, "should_gate", lambda strict: True)
    monkeypatch.setattr(run_checks, "check_compile",
                        lambda: run_checks._result("compile", True, "stub", True))
    monkeypatch.setattr(run_checks, "environment_checks", lambda strict: [])
    monkeypatch.setattr(run_checks, "attempt_fix",
                        lambda: (fixed.append(1),
                                 [run_checks._result("fix: prune", True, "x", False)])[1])

    assert run_checks.main(["--wait-for-ready", "--fix", "--no-tests"]) == 0
    assert fixed == [1]        # --fix survived the recursion into --strict


def test_run_checks_json_locks_fix_action_schema(tmp_path, monkeypatch, capsys):
    """The --json contract for fix actions is part of the interface, so pin it."""
    import run_checks

    monkeypatch.setattr(run_checks, "check_compile",
                        lambda: run_checks._result("compile", True, "stub", True))
    monkeypatch.setattr(run_checks, "environment_checks", lambda strict: [])
    monkeypatch.setattr(run_checks.health, "find_pmb_home",
                        lambda *a, **k: tmp_path)
    monkeypatch.setattr(run_checks.pmb_launcher, "prune_dead", lambda h: 3)
    monkeypatch.setattr(run_checks.pmb_launcher, "ensure_running",
                        lambda *a, **k: "started")
    monkeypatch.setattr(run_checks.pmb_launcher, "daemon_live",
                        lambda *a, **k: True)

    assert run_checks.main(["--no-tests", "--fix", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["passed"] is True
    fixes = [c for c in payload["checks"] if c["name"].startswith("fix:")]
    assert [c["name"] for c in fixes] == ["fix: prune", "fix: ensure"]
    for action in fixes:
        assert set(action) == {"name", "ok", "detail", "required"}
        assert action["ok"] is True
        assert action["required"] is False   # a fix never gates the run
    assert "removed 3 dead" in fixes[0]["detail"]


def test_should_gate_forces_strict_and_detects_a_live_daemon(monkeypatch):
    """Gating is on for --strict and for a host where the daemon already answers."""
    import run_checks

    monkeypatch.setattr(run_checks.health, "find_pmb_home",
                        lambda *a, **k: "/nope")
    monkeypatch.setattr(run_checks.pmb_launcher, "daemon_live",
                        lambda *a, **k: True)
    assert run_checks.should_gate(True) is True
    assert run_checks.should_gate(False) is True    # a live daemon => gate

    monkeypatch.setattr(run_checks.pmb_launcher, "daemon_live",
                        lambda *a, **k: False)
    if os.name != "nt":                             # a plain build box
        assert run_checks.should_gate(False) is False


def test_run_checks_auto_gating_reaches_the_environment_checks(monkeypatch,
                                                                capsys):
    """When the host is judged to run the daemon, advisories become gating."""
    import run_checks

    seen: list[bool] = []
    monkeypatch.setattr(run_checks, "should_gate", lambda strict: True)
    monkeypatch.setattr(run_checks, "check_compile",
                        lambda: run_checks._result("compile", True, "stub", True))

    def fake_env(strict):
        seen.append(strict)
        return [run_checks._result("daemon health", False, "down", strict)]

    monkeypatch.setattr(run_checks, "environment_checks", fake_env)

    assert run_checks.main(["--no-tests"]) == 1     # the advisory now gates
    assert seen == [True]
    out = capsys.readouterr().out
    assert "RESULT: FAIL" in out and "daemon health" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
