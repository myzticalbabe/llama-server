#!/usr/bin/env python3
"""pmb_daemon_health.py - detect a stale pmb / pams server record.

WHY THIS EXISTS
---------------
Both the pmb daemon and the pams MCP server advertise themselves through a JSON
file that outlives a hard kill:

  * pmb   -> <pmb_home>/servers.json          (a list of records)
  * pams  -> <data_dir>/mcp_server_port.json  (a single record)

When the process is SIGKILLed, or the machine reboots, the file keeps claiming
a live pid and status "active". A consumer that trusts the file then talks to
nothing. Measured here: on 2026-09-29 pams asserted an active port 21434 with
no listener and a dead pid; on 2026-09-30 .pmb/servers.json still claimed pid
7896 on :8765 after daemon.log had already logged "idle timeout reached -
exiting".

The file is therefore a HINT, never the truth. This script re-derives the
truth: for every record it checks (a) whether the pid is alive and (b) whether
the port is actually accepting connections, and reports STALE when the record
claims active but either check fails.

Usage
-----
    python pmb_daemon_health.py                 # auto-detect the pmb home
    python pmb_daemon_health.py --pmb-home /mnt/c/Users/clint/.pmb
    python pmb_daemon_health.py --json          # machine-readable
    python pmb_daemon_health.py --pams-port-file <path/to/mcp_server_port.json>

Exit codes: 0 healthy, 1 stale/dead record found, 2 could not read or parse.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys
import time
from pathlib import Path

EXIT_OK, EXIT_STALE, EXIT_ERROR = 0, 1, 2
_STILL_ACTIVE = 259  # GetExitCodeProcess: process has not exited
DEFAULT_TIMEOUT = 2.0


def _pid_alive(pid: int) -> bool:
    """True only if `pid` is a running process we can see.

    POSIX uses the classic signal-0 probe. Windows asks the kernel directly so
    the check does not depend on psutil being installed.
    """
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
                return code.value == _STILL_ACTIVE
            finally:
                k32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def _port_open(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return sock.connect_ex((host or "127.0.0.1", int(port))) == 0
    except (OSError, ValueError, TypeError):
        return False


def pid_alive(pid) -> bool:
    """Public alias for the liveness probe (used by pmb_daemon_launcher)."""
    return _pid_alive(pid)


def port_open(host: str, port, timeout: float = DEFAULT_TIMEOUT) -> bool:
    """Public alias for the TCP probe (used by pmb_daemon_launcher)."""
    return _port_open(host, port, timeout)


def _find_pmb_home(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("PMB_HOME")
    if env:
        return Path(env)
    candidates = [Path.home() / ".pmb"]
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        candidates.append(Path(userprofile) / ".pmb")
    # Running inside WSL against a Windows-side pmb home.
    candidates.extend(Path(p) for p in sorted(glob.glob("/mnt/c/Users/*/.pmb")))
    for candidate in candidates:
        if (candidate / "servers.json").is_file():
            return candidate
    return candidates[0]


def find_pmb_home(explicit: str | None = None) -> Path:
    """Public alias for the resolver, for other modules to call."""
    return _find_pmb_home(explicit)


def _check(record: dict, origin: str, timeout: float) -> dict:
    """Evaluate one advertised server record and classify it."""
    host = record.get("host") or "127.0.0.1"
    port = record.get("port")
    pid = record.get("pid")
    claimed = str(record.get("status", "active")).lower()

    alive = _pid_alive(pid) if isinstance(pid, int) else None
    listening = _port_open(host, port, timeout) if isinstance(port, int) else None

    age = None
    started = record.get("started_at")
    if isinstance(started, (int, float)) and started > 0:
        age = max(0.0, time.time() - float(started))

    # Informational only: a heartbeat proves the writer was alive after it
    # started. It never decides `status`, because a supervisor only stamps
    # while a session is open and its absence must not read as "down".
    hb_age = None
    hb = record.get("heartbeat_epoch")
    if isinstance(hb, (int, float)):
        hb_age = max(0.0, time.time() - float(hb))

    reasons = []
    if alive is False:
        reasons.append("pid %s is not running" % pid)
    if listening is False:
        reasons.append("nothing listening on %s:%s" % (host, port))
    if alive is None and listening is None:
        reasons.append("record carries neither pid nor port")

    if claimed != "active":
        status = "INACTIVE"  # file already admits it stopped
    elif alive and listening:
        status = "OK"
    elif reasons:
        status = "STALE"
    else:
        status = "UNKNOWN"

    return {
        "origin": origin,
        "pid": pid,
        "host": host,
        "port": port,
        "claimed": claimed,
        "pid_alive": alive,
        "port_open": listening,
        "age_seconds": None if age is None else round(age, 1),
        "heartbeat_age_seconds": None if hb_age is None else round(hb_age, 1),
        "status": status,
        "reasons": reasons,
    }


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise RuntimeError("%s: %s" % (path, exc))


def _collect(pmb_home: Path, pams_port_file: Path | None) -> list[dict]:
    servers_file = pmb_home / "servers.json"
    data = _read_json(servers_file)
    records: list[dict] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                records.append({**item, "pmb_record": True})
    elif isinstance(data, dict):
        records.append({**data, "pmb_record": True})

    if pams_port_file is not None:
        pams = _read_json(pams_port_file)
        if isinstance(pams, dict):
            records.append({**pams, "pmb_record": False})
    return records


def check_all(pmb_home=None, pams_port_file=None, timeout=DEFAULT_TIMEOUT):
    """Re-derive the truth for every advertised record.

    Importable entry point. pams_server_launcher.py calls this at startup, so a
    stale record is logged by a session that kilo or opencode already starts --
    nobody has to remember to run the CLI.
    """
    home = _find_pmb_home(pmb_home)
    pams = Path(pams_port_file) if pams_port_file else None
    return [
        _check(rec, "servers.json" if rec.get("pmb_record") else "pams_port_file",
               timeout)
        for rec in _collect(home, pams)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pmb-home", help="dir holding servers.json (default: auto)")
    parser.add_argument("--pams-port-file", help="path to pams mcp_server_port.json")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="TCP probe timeout in seconds (default 2)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args(argv)

    pmb_home = _find_pmb_home(args.pmb_home)
    pams_port_file = Path(args.pams_port_file) if args.pams_port_file else None

    if not (pmb_home / "servers.json").is_file() and pams_port_file is None:
        print("error: no servers.json under %s and no --pams-port-file given"
              % pmb_home, file=sys.stderr)
        return EXIT_ERROR

    try:
        results = check_all(pmb_home, pams_port_file, args.timeout)
    except RuntimeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return EXIT_ERROR

    if not results:
        print("warning: no records advertised (pmb home: %s)" % pmb_home,
              file=sys.stderr)
        return EXIT_ERROR

    stale = [r for r in results if r["status"] == "STALE"]

    if args.json:
        print(json.dumps({
            "pmb_home": str(pmb_home),
            "checked": len(results),
            "stale": len(stale),
            "results": results,
        }, indent=2))
    else:
        print("pmb/pams daemon health  (pmb home: %s)" % pmb_home)
        for r in results:
            where = "%s:%s" % (r["host"], r["port"])
            age = "" if r["age_seconds"] is None else "  age=%.0fs" % r["age_seconds"]
            print("  [%-8s] pid=%s %s  claimed=%s%s"
                  % (r["status"], r["pid"], where, r["claimed"], age))
            for reason in r["reasons"]:
                print("             - %s" % reason)
        if stale:
            print("\n%d stale record(s): the advertised endpoint is not live. "
                  "Start it (pmb daemon start) or drop the record." % len(stale))
        else:
            print("\nall advertised endpoints are live.")

    return EXIT_STALE if stale else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
