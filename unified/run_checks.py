#!/usr/bin/env python3
"""run_checks.py - one combined pass/fail for the MCP daemon stack.

Runs, in order:

  compile  - py_compile every module in the stack (required)
  tests    - the pytest suite for the stack (required)
  terence  - the oracle TERENCE's fault count (critical/high only; advisory)
  daemon   - live verdict from pmb_daemon_health.check_all() (advisory)
  audit    - .audit/daemon_health.jsonl hash chain (advisory)
  host     - can THIS host run and reach the pmb daemon (advisory)

Exit status is one bit: 0 only when every REQUIRED check passed. The advisories
default to non-gating because "the daemon is down" is a state of the machine,
not a defect in the code -- CI on a build box would otherwise fail forever.
Pass --strict to gate them too, which is what you want on a host that is
supposed to be running the daemon.

They also gate AUTOMATICALLY on a host that is clearly meant to run the daemon
(Windows, or anywhere a live daemon already answers): there "the daemon is
down" is a real defect, not a property of a build box. --strict forces gating
everywhere, including hosts that only check the stack.

    python run_checks.py
    python run_checks.py --strict
    python run_checks.py --json
    python run_checks.py --no-terence                 # skip the ~20s oracle check
    python run_checks.py --fix                        # heal a stale registry, then report
    python run_checks.py --watch                      # poll, re-report on change
    python run_checks.py --watch --fix                # poll AND heal a dying daemon
    python run_checks.py --wait-for-ready             # block until the daemon is up

--fix is opt-in and never runs by default. It applies pmb's own dead-pid rule
(pmb_daemon_launcher.prune_dead) to drop registry entries left behind by a
killed daemon, then tries to bring the daemon back (ensure_running) -- and only
then reports. It never deletes a live record and never touches the audit trail
beyond the events those two operations already emit.

--watch --fix heals a failing environment before judging it, throttled by
--fix-cooldown (default 60s) so a daemon that is down for a long time is not
re-pruned and re-spawned every interval. A FRESH failure always heals at once.

--wait-for-ready exists because the real end-to-end run has to happen on the
Windows side and cannot be driven from here (see test_daemon_stack_integration).
Start it here, run `pmb-ai daemon start` and `wsl --shutdown` over there, and
this finishes the E2E itself the moment the daemon answers.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import py_compile
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pmb_daemon_health as health  # noqa: E402
import pmb_daemon_launcher as pmb_launcher  # noqa: E402

MODULES = ["pmb_daemon_health.py", "pmb_daemon_launcher.py",
           "pams_server_launcher.py"]
TEST_FILES = ["test_daemon_health.py", "test_daemon_stack_integration.py"]

# The re-engineered error-resolution layer (oracles, "no verifier, no fix").
# Optional: a machine without the azazael tree simply reports it as not found.
TERENCE_ENV = "TERENCE_INVARIANTS"
TERENCE_CANDIDATES = [
    "/mnt/c/Users/clint/azazael/tools/terence_invariants.py",
    str(HERE.parent.parent / "azazael" / "tools" / "terence_invariants.py"),
]


def _result(name: str, ok: bool, detail: str, required: bool) -> dict:
    return {"name": name, "ok": ok, "detail": detail, "required": required}


# --- checks -----------------------------------------------------------------

def check_compile() -> dict:
    compiled, failed = [], []
    for name in MODULES + TEST_FILES:
        path = HERE / name
        if not path.is_file():
            failed.append("%s missing" % name)
            continue
        try:
            py_compile.compile(str(path), doraise=True)
            compiled.append(name)
        except py_compile.PyCompileError as exc:
            failed.append("%s: %s" % (name, str(exc).splitlines()[-1]))
    if failed:
        return _result("compile", False, "; ".join(failed), required=True)
    return _result("compile", True, "%d modules" % len(compiled), required=True)


def check_tests() -> dict:
    cmd = [sys.executable, "-m", "pytest", *TEST_FILES, "-q",
           "--no-header", "-p", "no:cacheprovider"]
    try:
        proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True,
                              text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _result("pytest suite", False, "could not run: %s" % exc,
                       required=True)
    summary = ""
    for line in reversed(proc.stdout.strip().splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            summary = line.strip().strip("=").strip()
            break
    return _result("pytest suite", proc.returncode == 0,
                   summary or "exit %d" % proc.returncode, required=True)


def check_daemon(strict: bool) -> dict:
    try:
        results = health.check_all()
    except Exception as exc:
        return _result("daemon health", False, "error: %s" % exc, required=strict)
    stale = [r for r in results if r["status"] == "STALE"]
    if not results:
        return _result("daemon health", False, "no records advertised",
                       required=strict)
    detail = "%d record(s), %d stale" % (len(results), len(stale))
    if stale:
        detail += ": " + "; ".join(
            "pid=%s %s:%s" % (r["pid"], r["host"], r["port"]) for r in stale)
    return _result("daemon health", not stale, detail, required=strict)


# A daemon that needs healing FLAP_THRESHOLD times inside FLAP_WINDOW_S is not
# unlucky, it is flapping: healing hides the symptom, so the run must say so.
FLAP_WINDOW_S = 3600.0
FLAP_THRESHOLD = 5


def record_fix_attempt(reason: str, before: dict, after: dict) -> None:
    """Persist one heal attempt so flapping is visible over time. Never raises."""
    try:
        pmb_launcher.audit_event(
            "watch_fix_applied", reason=reason,
            epoch=round(time.time(), 3),
            failing=sorted(n for n, ok in before.items() if not ok),
            resolved=sorted(n for n, ok in after.items() if ok),
            still_failing=sorted(n for n, ok in after.items() if not ok),
        )
    except Exception:
        pass


def check_audit(strict: bool) -> dict:
    chain_ok, chain_detail = pmb_launcher.verify_audit_chain()
    anchor_ok, anchor_detail = pmb_launcher.verify_against_anchors()
    heals = pmb_launcher.heal_history()
    now = time.time()
    recent = [h for h in heals
              if isinstance(h.get("epoch"), (int, float))
              and (now - float(h["epoch"])) <= FLAP_WINDOW_S]
    flapping = len(recent) >= FLAP_THRESHOLD
    ok = chain_ok and anchor_ok and not flapping
    detail = "chain: %s | anchor: %s | merkle: %s | heals: %d (%d in 1h)" % (
        chain_detail, anchor_detail, (pmb_launcher.merkle_root()[:16] or "-"),
        len(heals), len(recent))
    if flapping:
        detail += " FLAPPING (%d heals in 1h; see %s)" % (
            len(recent), pmb_launcher.AUDIT_FILENAME)
    return _result("audit trail", ok, detail, required=strict)


def attempt_fix() -> list[dict]:
    """Heal a stale registry and bring the daemon back, then let the caller
    re-judge the environment. Explicit and opt-in (--fix); never destructive.

    Both steps are pmb's own semantics, not new policy: prune_dead drops entries
    whose pid is gone (pmb/mcp/registry.py `_prune`), and ensure_running is the
    idempotent `pmb daemon start`. A no-op fix is reported as a PASS so the
    output never implies success where nothing happened.
    """
    home = health.find_pmb_home()
    actions = []
    try:
        removed = pmb_launcher.prune_dead(home)
        actions.append(_result(
            "fix: prune", True,
            "removed %d dead registry entr%s" % (
                removed, "y" if removed == 1 else "ies") if removed
            else "registry clean (0 removed)", required=False))
    except Exception as exc:
        actions.append(_result("fix: prune", False, "error: %s" % exc,
                               required=False))
    try:
        state = pmb_launcher.ensure_running(home)
        live = pmb_launcher.daemon_live(home, timeout=1.0)
        # A deliberate skip (non-Windows host, no executable) is not a failure
        # here, exactly as in the launcher CLI: there was nothing to ensure.
        actions.append(_result("fix: ensure", live or state.startswith("skipped-"),
                               "%s (live=%s)" % (state, live), required=False))
    except Exception as exc:
        actions.append(_result("fix: ensure", False, "error: %s" % exc,
                               required=False))
    return actions


def find_terence() -> Path | None:
    """Locate the oracle TERENCE: explicit env, then the known trees."""
    env = os.environ.get(TERENCE_ENV)
    if env and Path(env).is_file():
        return Path(env)
    for candidate in TERENCE_CANDIDATES:
        if Path(candidate).is_file():
            return Path(candidate)
    return None


def _pmb_workspace_dir() -> str | None:
    """The daemon workspace holding events.sqlite -- TERENCE's evidence DB."""
    try:
        workspaces = Path(health.find_pmb_home()) / "workspaces"
        if workspaces.is_dir():
            subs = sorted(p for p in workspaces.iterdir() if p.is_dir())
            if subs:
                return str(subs[0])
    except Exception:
        pass
    return None


def check_terence(strict: bool) -> dict:
    """Run the oracle TERENCE and report its fault count. ~20s, advisory."""
    tool = find_terence()
    if tool is None:
        return _result("terence oracles", True,
                       "terence_invariants.py not found (skipped)", required=False)
    env = dict(os.environ)
    if "PMB_WORKSPACE_DIR" not in env:
        workspace = _pmb_workspace_dir()
        if workspace:
            env["PMB_WORKSPACE_DIR"] = workspace
    try:
        proc = subprocess.run([sys.executable, str(tool), "--json"],
                              capture_output=True, text=True, timeout=600, env=env,
                              cwd=str(tool.parent.parent))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _result("terence oracles", False, "could not run: %s" % exc,
                       required=strict)
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        return _result("terence oracles", False,
                       "non-JSON output (rc=%d)" % proc.returncode, required=strict)
    faults = data.get("faults", [])
    crit = sum(1 for f in faults if f.get("severity") == "critical")
    high = sum(1 for f in faults if f.get("severity") == "high")
    detail = "oracles %s/%s | %d faults (%d critical, %d high)" % (
        data.get("oracles_run"), data.get("oracles_total"), len(faults), crit, high)
    # Only critical/high are treated as a failure: the 182 medium wiring faults
    # are a standing backlog, and gating on them would make the check noise.
    return _result("terence oracles", not (crit or high), detail, required=strict)


def check_host(strict: bool) -> dict:
    """Is a real end-to-end run possible from THIS host?"""
    findings = ["%s %s" % (platform.system(), platform.release())]
    interop = Path("/proc/sys/fs/binfmt_misc/WSLInterop")
    in_wsl = "microsoft" in platform.release().lower()
    if in_wsl:
        findings.append("WSLInterop %s" % ("present" if interop.exists()
                                          else "absent (cannot exec Windows binaries)"))
    exe = pmb_launcher.find_pmb_executable(health.find_pmb_home())
    findings.append("pmb exe %s" % (exe or "not found"))
    home = health.find_pmb_home()
    live = pmb_launcher.daemon_live(home, timeout=0.5)
    findings.append("daemon %s" % ("live" if live else "not reachable"))
    return _result("host readiness", live, "; ".join(findings), required=strict)


# --- reporting --------------------------------------------------------------

def render(results: list[dict], strict: bool) -> None:
    width = max(len(r["name"]) for r in results) + 2
    print("%-*s %-7s %-9s %s" % (width, "CHECK", "STATUS", "REQUIRED", "DETAIL"))
    print("-" * (width + 60))
    for r in results:
        status = "PASS" if r["ok"] else ("FAIL" if r["required"] else "WARN")
        print("%-*s %-7s %-9s %s" % (width, r["name"], status,
                                     "yes" if r["required"] else "advisory",
                                     r["detail"]))
    print()
    # Watch mode is often piped to a log or a file, where stdout is block
    # buffered; without this a poll would sit invisible until the buffer fills.
    sys.stdout.flush()


def environment_checks(strict: bool) -> list[dict]:
    return [check_daemon(strict), check_audit(strict), check_host(strict)]


def should_gate(strict: bool) -> bool:
    """Should the daemon/audit advisories gate on THIS host?

    --strict always gates. Otherwise they gate automatically when this host is
    demonstrably part of the stack -- it is the daemon's own platform (Windows),
    or a live daemon already answers here. On a plain build box neither holds,
    so the advisories stay non-gating and CI cannot fail forever on a daemon it
    was never meant to run.
    """
    if strict:
        return True
    if os.name == "nt":
        return True
    try:
        return pmb_launcher.daemon_live(health.find_pmb_home(), timeout=0.5)
    except Exception:
        return False


def wait_for_ready(interval: float, timeout: float) -> bool:
    """Block until the pmb daemon actually answers. 0 timeout = forever."""
    home = health.find_pmb_home()
    deadline = None if timeout <= 0 else time.time() + timeout
    print("waiting for the pmb daemon to answer (Ctrl-C to stop)...")
    while True:
        if pmb_launcher.daemon_live(home, timeout=0.5):
            print("[ready] pmb daemon is live on %s" % home)
            return True
        if deadline is not None and time.time() > deadline:
            print("[timeout] daemon still not reachable after %gs" % timeout)
            return False
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(interval)


def watch(interval: float, fix: bool = False, cooldown: float = 60.0) -> int:
    """Poll the environment checks; print a full report whenever they change.

    With fix=True a failing environment is healed (prune + ensure) BEFORE the
    verdict is taken, so a daemon that dies mid-session is brought back and the
    report shows the post-fix state rather than the corpse.

    Healing is throttled by `cooldown` so a daemon that stays down for hours is
    not re-pruned and re-spawned every interval. A fresh failure (or the first
    poll) always heals immediately; only a failure that persists past the
    cooldown is retried. Output is deduped by state either way.

    A newly failing check also re-runs the code checks, so a daemon that dies
    mid-session is reported together with the suite's current health rather
    than a bare warning.
    """
    previous: dict | None = None
    last_fix = 0.0
    try:
        while True:
            results = environment_checks(strict=False)
            state = {r["name"]: r["ok"] for r in results}
            newly_failing = previous is not None and any(
                state[n] is False and previous.get(n) is True for n in state)

            if fix and any(not ok for ok in state.values()):
                now = time.time()
                fresh = newly_failing or previous is None
                if fresh or (now - last_fix) >= cooldown:
                    # Heal, then re-judge: the report below reflects what the
                    # fix achieved, not the failure it just addressed.
                    before = state
                    results = attempt_fix() + environment_checks(strict=False)
                    state = {r["name"]: r["ok"] for r in results}
                    last_fix = now
                    # Persist the attempt: repeated heals are the only durable
                    # evidence that a daemon is flapping rather than just down.
                    record_fix_attempt(
                        "fresh-failure" if fresh else "cooldown-retry",
                        before, state)
            stamp = time.strftime("%H:%M:%S")
            print("[%s] %s" % (stamp, ", ".join(
                "%s=%s" % (n, "ok" if v else "FAIL") for n, v in state.items())))
            sys.stdout.flush()

            if state != previous or newly_failing:
                print()
                render(results, strict=False)
                if newly_failing:
                    code_results = [check_compile(), check_tests()]
                    render(code_results, strict=True)
                print()
            previous = state
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nwatch stopped")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true",
                        help="make the advisory environment checks gating")
    parser.add_argument("--no-tests", action="store_true",
                        help="skip the pytest suite")
    parser.add_argument("--no-terence", action="store_true",
                        help="skip the oracle TERENCE check (~20s)")
    parser.add_argument("--fix", action="store_true",
                        help="heal a stale registry and restart the daemon before reporting")
    parser.add_argument("--fix-cooldown", type=float, default=60.0,
                        help="min seconds between heals in --watch --fix (default 60)")
    parser.add_argument("--watch", action="store_true",
                        help="poll the environment checks and re-report on change")
    parser.add_argument("--wait-for-ready", action="store_true",
                        help="block until the daemon answers, then run --strict")
    parser.add_argument("--interval", type=float, default=15.0,
                        help="poll/secondary interval in seconds")
    parser.add_argument("--timeout", type=float, default=0.0,
                        help="0 waits forever (--wait-for-ready)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.wait_for_ready:
        try:
            if not wait_for_ready(args.interval, args.timeout):
                return 1
        except KeyboardInterrupt:
            print("\nwait cancelled")
            return 1
        print()
        # Ready: run the real gated suite, carrying --fix/--json/--no-tests
        # through so `--wait-for-ready --fix` heals and then gates in one go.
        follow = ["--strict"]
        if args.fix:
            follow.append("--fix")
        if args.json:
            follow.append("--json")
        if args.no_tests:
            follow.append("--no-tests")
        return main(follow)

    if args.watch:
        return watch(args.interval, fix=args.fix, cooldown=args.fix_cooldown)

    # Auto-gate the environment advisories on a host that is meant to run the
    # daemon; on this WSL box neither Windows nor a live daemon holds, so they
    # stay advisory and the run still reports PASS.
    gating = should_gate(args.strict)

    results = [check_compile()]
    if not args.no_tests:
        results.append(check_tests())
    if not args.no_terence:
        results.append(check_terence(gating))
    if args.fix:
        # Heal BEFORE the environment is judged, so the daemon/audit/host checks
        # below report the post-fix state rather than the state we just acted on.
        results += attempt_fix()
    results += environment_checks(gating)

    failures = [r for r in results if r["required"] and not r["ok"]]
    warnings = [r for r in results if not r["required"] and not r["ok"]]

    if args.json:
        print(json.dumps({"passed": not failures, "strict": gating,
                          "checks": results}, indent=2))
    else:
        render(results, gating)
        verdict = "PASS" if not failures else "FAIL"
        tail = ""
        if warnings:
            tail = " (%d advisory warning(s)%s)" % (
                len(warnings), "" if gating else "; --strict to gate")
        if failures:
            tail = " (%s)" % ", ".join(r["name"] for r in failures)
        print("RESULT: %s%s" % (verdict, tail))

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
