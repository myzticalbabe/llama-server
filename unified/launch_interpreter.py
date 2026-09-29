#!/usr/bin/env python3
# coding: utf-8
"""
launch_interpreter.py — shared Python entry point for the auto-loaded
Open Interpreter session.

Referenced by ``unified/Source_of_Truth.md`` since the beginning but never
actually created — which is why the Python interpreter never auto-loaded when
the llama-server loaded a model. This is the missing component.

It reads the *same* ``llama-server/config.toml`` ``[interpreter]`` block that
the Rust server reads, so the Rust start-up hook and this script cannot drift.

What it does
------------
1. Locates and parses ``config.toml``.
2. Resolves ``interpreter.exe`` (config path → venv-interpreter → Python
   Scripts → PATH).
3. Waits for the inference API to report healthy (the model is resident).
4. Launches Open Interpreter against that API.

Usage
-----
    python launch_interpreter.py                 # wait for health, then launch
    python launch_interpreter.py --dry-run       # print the command, don't run
    python launch_interpreter.py --no-wait       # launch without health check
    python launch_interpreter.py --status        # show config + server state
    python launch_interpreter.py --json          # machine-readable result

Exit codes
----------
    0  interpreter launched (or --dry-run / --status succeeded)
    1  configuration or resolution failure
    2  server never became healthy within the timeout
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

# Windows cp1252 consoles mangle the em dashes / arrows in this output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── Defaults (mirror config.rs) ───────────────────────────────────────────
DEFAULT_PORT = 8080
DEFAULT_API_KEY = "dummy"
HEALTH_TIMEOUT_S = 300
HEALTH_INTERVAL_S = 2


# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════

def find_config() -> Path | None:
    """Locate config.toml: $LLAMA_SERVER_CONFIG, repo copy, then CWD."""
    env = os.environ.get("LLAMA_SERVER_CONFIG")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parent.parent / "config.toml")
    candidates.append(Path.home() / "llama-server" / "config.toml")
    candidates.append(Path.cwd() / "config.toml")
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_config() -> tuple[dict, Path | None]:
    """Return (config_dict, config_path). Missing file => empty dict."""
    path = find_config()
    if path is None or tomllib is None:
        return {}, path
    with path.open("rb") as fh:
        return tomllib.load(fh), path


def interpreter_settings(cfg: dict) -> dict:
    """Flatten the config into the values the launcher needs."""
    interp = cfg.get("interpreter", {}) or {}
    port = int(cfg.get("port", DEFAULT_PORT) or DEFAULT_PORT)

    api_base = str(interp.get("api_base", "") or "").strip()
    if not api_base:
        api_base = f"http://127.0.0.1:{port}/v1"

    model = str(interp.get("model", "") or "").strip()
    if not model:
        model_path = str(cfg.get("model_path", "") or "").strip()
        model = Path(model_path).stem if model_path else "model.gguf"

    return {
        "auto_start": bool(interp.get("auto_start", True)),
        "path": str(interp.get("path", "") or "").strip(),
        "profile": str(interp.get("profile", "") or "").strip(),
        "api_base": api_base,
        "model": model,
        "api_key": str(interp.get("api_key", DEFAULT_API_KEY) or DEFAULT_API_KEY),
        "new_console": bool(interp.get("new_console", True)),
        "args": list(interp.get("args", []) or []),
        "port": port,
    }


def resolve_interpreter(configured: str) -> Path | None:
    """Configured path if it exists, else auto-detect. Mirrors config.rs."""
    if configured:
        p = Path(configured)
        if p.is_file():
            return p
    home = Path.home()
    for c in (
        home / "venv-interpreter" / "Scripts" / "interpreter.exe",
        home / "AppData" / "Local" / "Programs" / "Python" / "Python314"
        / "Scripts" / "interpreter.exe",
        home / ".local" / "bin" / "interpreter",
    ):
        if c.is_file():
            return c
    found = shutil.which("interpreter")
    return Path(found) if found else None


# ══════════════════════════════════════════════════════════════════════════
# Server readiness
# ══════════════════════════════════════════════════════════════════════════

def health_url(api_base: str) -> str:
    """http://host:port/v1 -> http://host:port/health"""
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base}/health"


def server_healthy(api_base: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(health_url(api_base), timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_for_server(
    api_base: str,
    timeout: int = HEALTH_TIMEOUT_S,
    interval: int = HEALTH_INTERVAL_S,
    verbose: bool = True,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server_healthy(api_base):
            return True
        if verbose:
            print(f"  … waiting for {health_url(api_base)}", flush=True)
        time.sleep(interval)
    return False


# ══════════════════════════════════════════════════════════════════════════
# Launch
# ══════════════════════════════════════════════════════════════════════════

def build_args(settings: dict) -> list[str]:
    """Explicit flags rather than --profile (0.4.x ignores the 0.2.x YAML)."""
    args = [
        "--api_base", settings["api_base"],
        "--model", settings["model"],
        "--api_key", settings["api_key"],
        "--offline",
    ]
    if settings["profile"]:
        args += ["--profile", settings["profile"]]
    args += settings["args"]
    return args


def running_interpreter_pids() -> list[int]:
    """PIDs of live interpreter.exe processes (Windows only; [] elsewhere)."""
    if os.name != "nt":
        return []
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq interpreter.exe", "/NH", "/FO", "CSV"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    pids: list[int] = []
    for line in out.splitlines():
        cells = [c.strip('"') for c in line.split('","')]
        if len(cells) >= 2 and cells[0].lower().startswith("interpreter.exe"):
            try:
                pids.append(int(cells[1]))
            except ValueError:
                pass
    return pids


def launch(exe: Path, args: list[str], new_console: bool = True) -> int:
    """Spawn Open Interpreter; return its PID (or the launcher's PID).

    With ``new_console`` the REPL is started via ``cmd /c start`` so it gets
    its own console and real standard handles. A plain Popen would inherit
    the caller's stdio — and when the caller is the headless llama-server its
    stdin is already at EOF, so Open Interpreter exits the moment it starts.
    """
    env = dict(os.environ)
    # Open Interpreter's venv keeps its native DLLs beside the exe.
    env["PATH"] = f"{exe.parent}{os.pathsep}{env.get('PATH', '')}"

    if new_console and os.name == "nt":
        proc = subprocess.Popen(
            ["cmd", "/c", "start", "", str(exe), *args], env=env)
        return proc.pid

    kwargs: dict = {"env": env}
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen([str(exe), *args], **kwargs)
    return proc.pid


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def cmd_status(settings: dict, config_path: Path | None, exe: Path | None) -> int:
    print("Open Interpreter auto-load — status")
    print(f"  config.toml : {config_path or '(not found — using defaults)'}")
    print(f"  auto_start  : {settings['auto_start']}")
    print(f"  interpreter : {exe or '(NOT FOUND)'}")
    print(f"  api_base    : {settings['api_base']}")
    print(f"  model       : {settings['model']}")
    print(f"  profile     : {settings['profile'] or '(none — explicit flags)'}")
    print(f"  health      : {health_url(settings['api_base'])} "
          f"-> {'OK' if server_healthy(settings['api_base']) else 'unreachable'}")
    return 0 if exe else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch Open Interpreter against the local llama-server.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command instead of running it")
    parser.add_argument("--no-wait", action="store_true",
                        help="do not wait for the inference API to be healthy")
    parser.add_argument("--timeout", type=int, default=HEALTH_TIMEOUT_S,
                        help=f"health-wait timeout in seconds (default {HEALTH_TIMEOUT_S})")
    parser.add_argument("--status", action="store_true",
                        help="show resolved config and exit")
    parser.add_argument("--json", action="store_true",
                        help="emit a machine-readable result")
    parser.add_argument("--inherit-stdio", action="store_true",
                        help="attach Open Interpreter to this terminal")
    parser.add_argument("--force", action="store_true",
                        help="launch even if an interpreter is already running")
    args = parser.parse_args(argv)

    cfg, config_path = load_config()
    settings = interpreter_settings(cfg)
    exe = resolve_interpreter(settings["path"])

    if args.status:
        if args.json:
            print(json.dumps({
                "config": str(config_path) if config_path else None,
                "interpreter": str(exe) if exe else None,
                "api_base": settings["api_base"],
                "model": settings["model"],
                "auto_start": settings["auto_start"],
                "healthy": server_healthy(settings["api_base"]),
            }, indent=2))
            return 0 if exe else 1
        return cmd_status(settings, config_path, exe)

    if exe is None:
        msg = ("interpreter.exe not found — set [interpreter].path in "
               "config.toml or install Open Interpreter")
        if args.json:
            print(json.dumps({"ok": False, "error": msg}))
        else:
            print(f"  ERROR: {msg}", file=sys.stderr)
        return 1

    cli_args = build_args(settings)

    if args.dry_run:
        cmd = subprocess.list2cmdline([str(exe), *cli_args])
        if args.json:
            print(json.dumps({"ok": True, "dry_run": True, "command": cmd}))
        else:
            print("  Would launch:")
            print(f"    {cmd}")
        return 0

    if not args.no_wait:
        if not args.json:
            print(f"  Waiting for model on {settings['api_base']} …")
        if not wait_for_server(settings["api_base"], timeout=args.timeout,
                               verbose=not args.json):
            msg = (f"server did not become healthy at "
                   f"{health_url(settings['api_base'])} within {args.timeout}s")
            if args.json:
                print(json.dumps({"ok": False, "error": msg}))
            else:
                print(f"  ERROR: {msg}", file=sys.stderr)
            return 2

    # Idempotent: never stack REPLs. The Rust auto-load and this script may
    # both fire for the same model load.
    if not args.force:
        existing = running_interpreter_pids()
        if existing:
            msg = f"interpreter already running (pid {existing[0]})"
            if args.json:
                print(json.dumps({"ok": True, "reused": True, "pid": existing[0]}))
            else:
                print(f"  {msg} — reusing (use --force to launch another)")
            return 0

    pid = launch(exe, cli_args,
                 new_console=settings["new_console"] and not args.inherit_stdio)

    if args.json:
        print(json.dumps({"ok": True, "pid": pid, "interpreter": str(exe),
                          "api_base": settings["api_base"],
                          "model": settings["model"]}))
    else:
        print(f"  Open Interpreter started (pid {pid})")
        print(f"    endpoint : {settings['api_base']}")
        print(f"    model    : {settings['model']}")
        print(f"    exe      : {exe}")
        print("    note     : auto_run is off — it will ask before running code")
    return 0


if __name__ == "__main__":
    sys.exit(main())
