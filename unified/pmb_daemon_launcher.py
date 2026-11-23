#!/usr/bin/env python3
"""pmb_daemon_launcher.py - keep the pmb memory daemon up and provably alive.

WHY THIS EXISTS
---------------
The pmb daemon registers itself in ``$PMB_HOME/servers.json`` and unregisters on
a graceful exit (pmb/mcp/daemon.py: atexit + finally). A SIGKILL or a reboot
leaves the entry behind, and nothing rewrites the file until some process
happens to touch the registry, so the record can advertise a dead pid for days.
Measured here: servers.json still claimed pid 7896 on :8765 long after
daemon.log had logged "idle timeout reached - exiting".

pmb's own registry already knows the fix -- ``_prune`` drops entries whose pid
is not running, and ``list_servers(prune=True)`` applies it on read
(pmb/mcp/registry.py). The gap is that nothing reads it. This launcher closes
that gap, and adds the one thing the registry has no concept of: a heartbeat.

WHAT IT DOES
------------
  prune   - drop entries whose pid is no longer running. That is pmb's own rule,
            not a new one.
  stamp   - annotate every live entry with ``heartbeat_at`` / ``heartbeat_epoch``
            / ``pid_alive``, so an "active" claim carries proof of a recent
            process instead of just an assertion. Writes are atomic.
  ensure  - start the daemon when it is not live (``pmb daemon start``, already
            idempotent) and wait for it to answer.
  supervise - ensure + keep stamping until the daemon goes away.

WHY A SUPERVISOR AND NOT A PATCH
--------------------------------
The daemon lives in the ``.pmb-venv`` site-packages, so editing it directly
would be reverted by the next ``pip install`` -- and silently diverging from the
shipped package is worse than not having a heartbeat. This wrapper sits next to
``pams_server_launcher.py``, which solves the identical problem for pams in the
identical way.

CLI
---
    python pmb_daemon_launcher.py --status       # report only, never starts
    python pmb_daemon_launcher.py --ensure       # start it if it is down
    python pmb_daemon_launcher.py --supervise    # ensure + heartbeat loop
    python pmb_daemon_launcher.py --json

Exit codes: 0 ok (or deliberately skipped), 1 daemon down, 2 error.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pmb_daemon_health as health  # noqa: E402

EXIT_OK, EXIT_DOWN, EXIT_ERROR = 0, 1, 2
DEFAULT_INTERVAL = float(os.environ.get("PMB_HEARTBEAT_INTERVAL", "15"))
_CREATE_NO_WINDOW = 0x08000000
_CREATE_NEW_PROCESS_GROUP = 0x00000200

AUDIT_FILENAME = "daemon_health.jsonl"
ANCHORS_FILENAME = "daemon_health.anchors.jsonl"
_GENESIS = "0" * 64
_AUDIT_LOCK = threading.Lock()
_MERKLE_SKIP = ("chain_checkpoint", "session_end")


def _lock_fd(fd) -> None:
    """Cross-PROCESS advisory lock on byte 0. Best effort, never raises."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError:
        pass  # a filesystem without locking still gets O_APPEND's ordering


def _unlock_fd(fd) -> None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass


@contextlib.contextmanager
def _locked_sidecar_fd(path: Path):
    """Hold the sidecar open and locked for a whole read-head -> append cycle.

    `_AUDIT_LOCK` is a threading lock, so it serialises only ONE process. The
    pmb daemon and the pams launcher both append to this same file, which
    corrupts it in two independent ways:

      * On Windows `open("a")` is seek-to-end-then-write, NOT an atomic append.
        Two processes can seek to the same offset and overwrite each other,
        leaving one record's tail behind as an orphan fragment. That is exactly
        the shape found in .audit/daemon_health.jsonl on 2026-10-09: a complete
        record on line 41 followed by the orphan `_daemon_gone_by_pams"}` on
        line 42 (and again at line 110), after which the chain would not verify.
      * `audit_event` reads the head hash and then appends. Two processes that
        both read the same head would append records sharing one `prev_hash`, so
        the chain would fail even if every individual write were atomic.

    The lock therefore has to span the read as well as the write, which is why
    this yields the locked fd instead of doing a fire-and-forget append. The
    byte-0 advisory lock is the load-bearing part on Windows.

    The handle is opened O_RDWR, not O_WRONLY, because the head hash has to be
    read back THROUGH it (see `_head_hash_fd`).
    """
    flags = (os.O_APPEND | os.O_CREAT | os.O_RDWR
             | getattr(os, "O_BINARY", 0))
    fd = os.open(str(path), flags)
    try:
        _lock_fd(fd)
        try:
            yield fd
        finally:
            _unlock_fd(fd)
    finally:
        os.close(fd)


def _append_locked(fd, line: str) -> None:
    """One contiguous write of a whole record, flushed for crash durability."""
    os.write(fd, line.encode("utf-8"))
    os.fsync(fd)


# --- audit sidecar ----------------------------------------------------------
# The repo's real trail (.audit/session_*.jsonl) is a SHA-256 hash chain with
# Ed25519 signatures, one file per process, written through a sentinel-gated
# handle (azazael/src/CORE/audit_trail.py). Appending to it from HERE would
# CORRUPT it: a foreign process cannot know the owning session's _prev_hash or
# line counter, so verify_chain would fail afterwards and the trust layer would
# report tampering that never happened.
#
# So this writes an ADJACENT, self-chained file instead. Every consumer in the
# repo globs `session_*.jsonl` -- verify_all.check_audit_chain,
# governance._read_audit_trail, direct_memory, data_retention -- so a different
# filename is provably outside the chain. Records are still tamper-evident
# among themselves, using the same canonical-JSON + SHA-256 recipe.


def audit_dir() -> Path:
    explicit = os.environ.get("AZAZAEL_AUDIT_DIR")
    if explicit:
        return Path(explicit)
    home = Path(os.environ.get("USERPROFILE") or Path.home()) / ".audit"
    if home.is_dir():
        return home
    # WSL: the populated trail usually lives on the Windows side.
    import glob as _glob
    for candidate in sorted(_glob.glob("/mnt/c/Users/*/.audit")):
        return Path(candidate)
    return home


def _compute_hash(entry: dict, prev_hash: str) -> str:
    """Same recipe as audit_trail._compute_hash, for the sidecar chain."""
    payload = {k: v for k, v in entry.items()
               if k not in ("hash", "signature", "entry_count")}
    payload["prev_hash"] = prev_hash
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _head_from_bytes(data: bytes) -> str:
    """Head hash from raw sidecar bytes: the last line that IS a record.

    Trailing lines that do not parse are skipped rather than turned into a
    genesis. This is deliberate: `_head_hash` used to fall back to `_GENESIS`
    when the last line was unparseable, so ONE torn line (see
    `_locked_sidecar_fd`) made the next append chain from genesis -- and the
    one after that too, because the new last line chained from genesis rather
    than from the fragment. A single torn write therefore poisoned every
    subsequent record, which is exactly the shape found on 2026-10-03 in
    .audit/daemon_health.jsonl (`_daemon_gone_by_pams"}` on line 42 followed by
    a genesis-linked record on line 43, and again at 110/111).

    Chaining on the last real record keeps the damage to the record that was
    actually torn. The torn line still makes `verify_audit_chain` fail at that
    line -- honestly, since the file IS corrupt there -- but the chain no longer
    grows a fresh break per append. A torn last line is never repaired silently.
    """
    text = data.decode("utf-8", "replace")
    for line in reversed(text.splitlines()):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        return str(entry.get("hash") or _GENESIS)
    return _GENESIS


def _head_hash_fd(fd: int) -> str:
    """Head hash read back through the LOCKED handle that will append.

    A second handle is not usable here: `os.open` on Windows opens the sidecar
    with sharing that denies concurrent readers, so `path.read_text()` raises
    PermissionError while the write handle is open -- and the old
    `except OSError: return _GENESIS` then silently genesis-linked every record.
    Reading through the same fd is also the stricter guarantee: the bytes we
    chain on are the bytes the cross-process lock is held over.
    """
    try:
        size = os.fstat(fd).st_size
        if size <= 0:
            return _GENESIS
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, size)
    except OSError:
        return _GENESIS
    return _head_from_bytes(data)


def _head_hash(path: Path) -> str:
    """Head hash of a sidecar read from disk (no open write handle)."""
    try:
        data = path.read_bytes()
    except OSError:
        return _GENESIS
    return _head_from_bytes(data)


def audit_event(event: str, log=print, **fields) -> str | None:
    """Append one self-chained record to .audit/daemon_health.jsonl.

    Returns the new head hash, or None. Never raises: the audit trail must not
    be able to stop the daemon path it is recording.
    """
    try:
        path = audit_dir() / AUDIT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK, _locked_sidecar_fd(path) as fd:
            prev = _head_hash_fd(fd)
            entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "type": event,
                     "pid": os.getpid(), "source": "pmb_daemon_launcher"}
            entry.update(fields)
            entry["prev_hash"] = prev
            entry["hash"] = _compute_hash(entry, prev)
            line = json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
            _append_locked(fd, line)
        return entry["hash"]
    except Exception as exc:
        try:
            log("[pmb] audit write failed: %s" % exc)
        except Exception:
            pass
        return None


def verify_audit_chain(path=None) -> tuple[bool, str]:
    """Verify the sidecar's own hash chain. Missing file is not a failure.

    Deliberately the same rule as azazael/src/CORE/audit_trail.verify_chain:
    genesis prev_hash, then each entry's prev_hash must equal the previous
    entry's hash, and the hash must recompute. Interoperability with that
    verifier is asserted in the test suite.
    """
    path = Path(path) if path else audit_dir() / AUDIT_FILENAME
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
    except FileNotFoundError:
        return True, "no audit sidecar yet (%s)" % path
    except OSError as exc:
        return False, "unreadable: %s" % exc
    prev = _GENESIS
    for number, line in enumerate(lines, 1):
        try:
            entry = json.loads(line)
        except ValueError:
            return False, "line %d is not JSON" % number
        if entry.get("prev_hash") != prev:
            return False, "line %d: prev_hash does not chain" % number
        if entry.get("hash") != _compute_hash(entry, prev):
            return False, "line %d: hash mismatch (tampered?)" % number
        prev = entry["hash"]
    return True, "%d entries, head %s" % (len(lines), prev[:16])


def _read_entries(path: Path) -> list[dict]:
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
    except (OSError, ValueError):
        return []
    out = []
    for line in lines:
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def heal_history(path=None, limit: int | None = None) -> list[dict]:
    """Recorded heal attempts (from run_checks --watch --fix), oldest first.

    Persisted in the same self-chained sidecar as every other lifecycle event,
    so a daemon that flaps shows up as a run of `watch_fix_applied` rows with
    `epoch` stamps -- and that history is tamper-evident, not a plain log.
    """
    path = Path(path) if path else audit_dir() / AUDIT_FILENAME
    entries = [e for e in _read_entries(path)
               if e.get("type") == "watch_fix_applied"]
    return entries[-limit:] if limit else entries


# --- Merkle layer -----------------------------------------------------------
# Same construction as audit_trail._compute_merkle_root: SHA-256 double-hashed
# parent nodes, last leaf duplicated on an odd count. Reimplemented rather than
# imported because the supervisor runs from the pmb venv, which has no `CORE`
# package; the test suite asserts the two produce identical roots.


def _compute_merkle_root(hashes: list[str]) -> str:
    if not hashes:
        return ""
    level = [bytes.fromhex(h) for h in hashes]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        nxt = []
        for i in range(0, len(level), 2):
            combined = level[i] + level[i + 1]
            nxt.append(bytes.fromhex(
                hashlib.sha256(hashlib.sha256(combined).digest()).hexdigest()))
        level = nxt
    return level[0].hex() if level else ""


def merkle_root(path=None) -> str:
    """Merkle root over the sidecar's entry hashes (checkpoints excluded)."""
    path = Path(path) if path else audit_dir() / AUDIT_FILENAME
    hashes = [e["hash"] for e in _read_entries(path)
              if e.get("hash") and e.get("type") not in _MERKLE_SKIP]
    return _compute_merkle_root(hashes)


# --- anchors layer ----------------------------------------------------------
# The sidecar is anchored in the SAME layer -- same .audit/ directory, same
# record schema as audit_trail.anchor_head_hash, same SHA-256/Merkle
# construction -- but in its own ledger:
#     .audit/daemon_health.anchors.jsonl
#     {ts, audit_file, head_hash, entry_count, session_uuid, merkle_root}
#
# It is deliberately NOT appended to the shared .audit/anchors.jsonl. That file
# is FIRST-MATCH-PER-FILE: verify_head_hash() takes the first record for a path
# and compares it with the file's CURRENT last hash. That is right for the
# repo's model, where a session file is anchored once at close and never grows
# again. The sidecar is append-only and never closes, so a second anchor would
# make verify_head_hash() report a mismatch on perfectly intact data -- a false
# accusation of tampering written into the shared trust file, which is worse
# than keeping our ledger beside it.
#
# In exchange this ledger is checked harder than the repo anchors are:
# verify_against_anchors() requires the anchored head to still EXIST in the
# chain (ancestry, not equality) and rejects truncation below the anchored
# entry count.


def anchors_path() -> Path:
    return audit_dir() / ANCHORS_FILENAME


def anchor_sidecar(head_hash: str | None = None,
                   entry_count: int | None = None) -> str | None:
    """Publish the sidecar head hash to its own anchors ledger. Never raises."""
    path = audit_dir() / AUDIT_FILENAME
    entries = _read_entries(path)
    head = head_hash or (entries[-1]["hash"] if entries else "")
    if not head or head == _GENESIS:
        return None
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "audit_file": str(path),
        "head_hash": head,
        "entry_count": len(entries) if entry_count is None else entry_count,
        "session_uuid": "",
        # Additive key: the repo's readers use .get(), so an extra field is
        # ignored there while making the Merkle commitment visible to us.
        "merkle_root": merkle_root(path),
    }
    try:
        target = anchors_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK, _locked_sidecar_fd(target) as fd:
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            _append_locked(fd, line)
        return head
    except Exception as exc:
        print("[pmb] anchor write failed: %s" % exc)
        return None


def verify_against_anchors(path=None) -> tuple[bool, str]:
    """Check the sidecar against its newest anchor.

    The anchor is a checkpoint, so the test is not "head unchanged" -- more
    entries may legitimately follow. It is "the anchored head is still in the
    chain", which fails exactly when records below the anchor were rewritten,
    plus a truncation check when the anchor recorded an entry count.
    """
    path = Path(path) if path else audit_dir() / AUDIT_FILENAME
    anchor_file = anchors_path()
    stored = None
    for record in _read_entries(anchor_file):
        if os.path.normpath(str(record.get("audit_file", ""))) == \
                os.path.normpath(str(path)):
            stored = record          # newest wins; anchors are append-only
    if stored is None:
        if not anchor_file.exists():
            return True, "no anchors layer yet (%s)" % anchor_file
        return True, "no anchor for this sidecar yet"

    anchored = stored.get("head_hash")
    entries = _read_entries(path)
    if not any(e.get("hash") == anchored for e in entries):
        return False, ("anchored head %s is no longer in the chain "
                       "(records below it were rewritten)" % str(anchored)[:16])
    expected = stored.get("entry_count")
    if isinstance(expected, int) and len(entries) < expected:
        return False, "chain truncated: %d entries < anchored %d" % (
            len(entries), expected)
    return True, "anchor %s intact, %d entries (anchored at %d)" % (
        str(anchored)[:16], len(entries), expected if isinstance(expected, int)
        else len(entries))


def _audit_and_anchor(event: str, **fields) -> str | None:
    """Audit a lifecycle transition and publish the new head to the anchors.

    Anchoring per transition (not per tick) is what makes restarts land in the
    trust root without turning anchors.jsonl into a heartbeat log.
    """
    result = audit_event(event, **fields)
    if result:
        anchor_sidecar(head_hash=result)
    return result


# --- registry ---------------------------------------------------------------

def registry_path(home) -> Path:
    return Path(home) / "servers.json"


def load_entries(home) -> list[dict]:
    try:
        data = json.loads(registry_path(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict)]


def save_entries(home, entries: list[dict]) -> None:
    """Atomic replace; the registry is best-effort and must never raise."""
    path = registry_path(home)
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def prune_dead(home) -> int:
    """Drop entries whose pid is gone. Returns how many were removed."""
    entries = load_entries(home)
    live = [e for e in entries if health.pid_alive(e.get("pid"))]
    removed = len(entries) - len(live)
    if removed:
        save_entries(home, live)
        audit_event("pmb_registry_pruned", pmb_home=str(home), removed=removed)
    return removed


def stamp_heartbeat(home, now: float | None = None) -> int:
    """Prune dead entries, then stamp every live one. Returns live count."""
    prune_dead(home)
    entries = load_entries(home)
    epoch = time.time() if now is None else float(now)
    stamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch))
    for entry in entries:
        entry["pid_alive"] = True
        entry["heartbeat_at"] = stamp_iso
        entry["heartbeat_epoch"] = round(epoch, 3)
    if entries:
        save_entries(home, entries)
    return len(entries)


# --- daemon -----------------------------------------------------------------

def daemon_entry(home) -> dict | None:
    """The newest registered daemon entry, alive or not."""
    found = [e for e in load_entries(home) if e.get("kind") == "daemon"]
    found.sort(key=lambda e: e.get("started_at") or 0, reverse=True)
    return found[0] if found else None


def daemon_live(home, timeout: float = 1.0) -> bool:
    """A daemon is live only if its pid runs AND its port answers.

    The registry's own `alive` flag is pid-only; a pid can be alive while the
    uvloop server is wedged or bound elsewhere, so the port is probed too.
    """
    entry = daemon_entry(home)
    if not entry:
        return False
    if not health.pid_alive(entry.get("pid")):
        return False
    port = entry.get("port")
    if not isinstance(port, int):
        return False
    return health.port_open(entry.get("host") or "127.0.0.1", port, timeout)


def find_pmb_executable(home) -> str | None:
    """Locate the pmb CLI: explicit env, then the sibling venv, then PATH."""
    home = Path(home)
    candidates = [
        os.environ.get("PMB_BIN"),
        str(home.parent / ".pmb-venv" / "Scripts" / "pmb-ai.exe"),
        str(home.parent / ".pmb-venv" / "Scripts" / "pmb-ai"),
        str(home.parent / ".pmb-venv" / "bin" / "pmb-ai"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    for name in ("pmb-ai", "pmb-ai.exe", "pmb"):
        found = shutil.which(name)
        if found:
            return found
    return None


def start_daemon(home, wait: float = 8.0, timeout: float = 1.0,
                 executable: str | None = None) -> bool:
    """Spawn ``pmb daemon start`` and wait until the registry shows it live.

    `pmb daemon start` is idempotent and already detaches the real server, so
    this only has to launch it and poll.
    """
    if os.name != "nt":
        return False  # the daemon is a Windows venv binary; nothing to spawn
    executable = executable or find_pmb_executable(home)
    if not executable:
        return False
    flags = 0
    if os.name == "nt":
        flags = _CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen(
            [executable, "daemon", "start"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, creationflags=flags,
        )
    except OSError as exc:
        _audit_and_anchor("pmb_daemon_start_failed", pmb_home=str(home),
                          reason="spawn-error: %s" % exc)
        return False
    deadline = time.time() + max(0.0, wait)
    while time.time() < deadline:
        time.sleep(0.25)
        if daemon_live(home, timeout):
            break
    started = daemon_live(home, timeout)
    entry = daemon_entry(home) or {}
    _audit_and_anchor("pmb_daemon_started" if started else "pmb_daemon_start_failed",
                      pmb_home=str(home), executable=str(executable),
                      port=entry.get("port"), pid=entry.get("pid"))
    return started


def ensure_running(home, wait: float = 8.0, timeout: float = 1.0,
                   executable: str | None = None) -> str:
    """Bring the daemon up if needed. Returns why, never raises."""
    if daemon_live(home, timeout):
        return "already-live"
    if os.name != "nt":
        return "skipped-non-windows"
    if not (executable or find_pmb_executable(home)):
        return "skipped-no-executable"
    return "started" if start_daemon(home, wait, timeout, executable) else "start-failed"


def supervise(home, interval: float = DEFAULT_INTERVAL, wait: float = 8.0,
              timeout: float = 1.0, restart: bool = True, max_ticks: int | None = None,
              log=print) -> int:
    """Ensure the daemon is up, then heartbeat it until it dies or we stop.

    `max_ticks` exists so the loop is testable; in production it is None and the
    loop runs until KeyboardInterrupt or process exit.
    """
    state = ensure_running(home, wait, timeout)
    log("[pmb] %s" % state)
    was_live = daemon_live(home, timeout)
    if not was_live:
        # One record per supervise run, not one per tick: "this session began
        # with no reachable daemon" is the fact worth tracing.
        _audit_and_anchor("pmb_daemon_down_at_start", pmb_home=str(home),
                          ensure=state)
    ticks = 0
    try:
        while True:
            stamped = stamp_heartbeat(home)
            live_now = daemon_live(home, timeout)
            if not live_now:
                # Audit the live -> gone transition, not every tick it stays
                # gone: a restart loop must not flood the trail.
                if was_live or not restart:
                    log("[pmb] daemon is gone (stamped %d record(s))" % stamped)
                    _audit_and_anchor("pmb_daemon_gone", pmb_home=str(home),
                                      stamped=stamped)
                if not restart:
                    return EXIT_DOWN
                log("[pmb] %s" % ensure_running(home, wait, timeout))
                live_now = daemon_live(home, timeout)
            was_live = live_now
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                return EXIT_OK
            time.sleep(interval)
    except KeyboardInterrupt:
        log("[pmb] stopped")
        return EXIT_OK
    finally:
        # Seal the run: the final head goes into the anchors layer so the whole
        # supervise session is covered by the trust root, not just its events.
        anchor_sidecar()


def status(home, timeout: float = 1.0) -> dict:
    entry = daemon_entry(home)
    alive = health.pid_alive(entry.get("pid")) if entry else False
    live = daemon_live(home, timeout)
    age = None
    if entry and isinstance(entry.get("started_at"), (int, float)):
        age = round(max(0.0, time.time() - float(entry["started_at"])), 1)
    hb = None
    if entry and isinstance(entry.get("heartbeat_epoch"), (int, float)):
        hb = round(max(0.0, time.time() - float(entry["heartbeat_epoch"])), 1)
    return {
        "pmb_home": str(home),
        "executable": find_pmb_executable(home),
        "registered": bool(entry),
        "pid": entry.get("pid") if entry else None,
        "host": entry.get("host") if entry else None,
        "port": entry.get("port") if entry else None,
        "pid_alive": alive,
        "port_open": live,
        "age_seconds": age,
        "heartbeat_age_seconds": hb,
        "live": live,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pmb-home", help="dir holding servers.json (default: auto)")
    parser.add_argument("--status", action="store_true", help="report only")
    parser.add_argument("--ensure", action="store_true", help="start if down")
    parser.add_argument("--supervise", action="store_true", help="ensure + heartbeat")
    parser.add_argument("--stamp", action="store_true", help="heartbeat once, exit")
    parser.add_argument("--verify-audit", action="store_true",
                        help="verify the sidecar chain, its anchor and Merkle root")
    parser.add_argument("--anchor", action="store_true",
                        help="publish the sidecar head to .audit/anchors.jsonl")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--wait", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    home = health.find_pmb_home(args.pmb_home)

    if args.verify_audit:
        chain_ok, chain_detail = verify_audit_chain()
        anchor_ok, anchor_detail = verify_against_anchors()
        root = merkle_root()
        ok = chain_ok and anchor_ok
        if args.json:
            print(json.dumps({"ok": ok, "chain_ok": chain_ok,
                              "chain": chain_detail, "anchor_ok": anchor_ok,
                              "anchor": anchor_detail,
                              "merkle_root": root}, indent=2))
        else:
            print("[pmb] chain : %s -- %s" % (chain_ok, chain_detail))
            print("[pmb] anchor: %s -- %s" % (anchor_ok, anchor_detail))
            print("[pmb] merkle: %s" % (root[:32] or "(no entries)"))
        return EXIT_OK if ok else EXIT_ERROR
    if args.anchor:
        head = anchor_sidecar()
        if args.json:
            print(json.dumps({"anchored": head,
                              "merkle_root": merkle_root()}, indent=2))
        else:
            print("[pmb] anchored %s" % (str(head)[:16] if head
                                         else "nothing (sidecar empty)"))
        return EXIT_OK if head else EXIT_ERROR
    if args.supervise:
        return supervise(home, args.interval, args.wait, args.timeout)
    if args.ensure:
        state = ensure_running(home, args.wait, args.timeout)
        live = daemon_live(home, args.timeout)
        if args.json:
            print(json.dumps({"action": "ensure", "result": state,
                              "live": live}, indent=2))
        else:
            print("[pmb] %s (live=%s)" % (state, live))
        # A deliberate skip is not a failure: on a host where pmb cannot run
        # there was nothing to ensure, and reporting "down" would be misleading.
        return EXIT_OK if (live or state.startswith("skipped-")) else EXIT_DOWN
    if args.stamp:
        count = stamp_heartbeat(home)
        print(json.dumps({"action": "stamp", "live_records": count})
              if args.json else "[pmb] stamped %d live record(s)" % count)
        return EXIT_OK

    info = status(home, args.timeout)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        if not info["registered"]:
            print("[pmb] no daemon registered in %s" % info["pmb_home"])
        else:
            print("[pmb] daemon pid=%s %s:%s pid_alive=%s port_open=%s live=%s"
                  % (info["pid"], info["host"], info["port"], info["pid_alive"],
                     info["port_open"], info["live"]))
    return EXIT_OK if info["live"] else EXIT_DOWN


if __name__ == "__main__":
    raise SystemExit(main())
