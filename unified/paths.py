"""paths.py — single source of truth for unified-launcher filesystem locations.

All machine-specific locations live here with environment-variable overrides,
so the launcher works on any machine/user instead of hardcoding C:\\Users\\clint.

Environment overrides:
    LLAMA_SERVER_EXE   explicit llama-server binary
    LLAMA_SERVER_DIR   directory containing llama-server.exe
    PYTHON314_DIR      Python 3.14 install dir (for python3.14t.exe)
    OV_LIBS_DIR        OpenVINO libs dir
    PAMS_DATA_DIR      persistent-ai-memory data dir
    LMSTUDIO_MODELS    .lmstudio models dir (for use_model.dat default)
    HARVESTER_OUTPUT   harvester output dir
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

HOME = Path.home()


def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val).expanduser() if val else default


# ── Python 3.14 install (free-threaded python3.14t.exe lives here) ──────────
PYTHON_DIR: Path = _env_path(
    "PYTHON314_DIR",
    Path(r"C:\Users\clint\AppData\Local\Programs\Python\Python314")
    if os.name == "nt"
    else HOME / ".local" / "share" / "python314",
)
PYTHON_FT: Path = PYTHON_DIR / "python3.14t.exe"


def _default_ov_libs() -> Path:
    try:
        import openvino  # noqa: F401
        import openvino as _ov

        libs = Path(_ov.__file__).parent / "libs"
        if libs.exists():
            return libs
    except Exception:
        pass
    return (
        Path(r"C:\Users\clint\AppData\Local\Programs\Python\Python314\Lib\site-packages\openvino\libs")
        if os.name == "nt"
        else HOME / ".local" / "lib" / "openvino"
    )


OV_LIBS: Path = _env_path("OV_LIBS_DIR", _default_ov_libs())

# ── PAMS persistent-memory data dir ─────────────────────────────────────────
PAMS_DATA_DIR: Path = _env_path(
    "PAMS_DATA_DIR",
    Path(r"G:\source-of-truth\persistent-ai-memory\memory_data"),
)

# ── llama-server binary resolution ──────────────────────────────────────────
_LEGACY_SERVER_DIR = HOME / "temp_llama_cpp_build" / "llama.cpp" / "build_server_multi" / "bin" / "Release"
_NEW_SERVER_DIR = HOME / "temp_llama_cpp_build" / "llama.cpp" / "build_all_backends" / "bin" / "Release"


def _candidate_server_exes() -> list[Path]:
    cands: list[Path] = []
    env_exe = os.environ.get("LLAMA_SERVER_EXE")
    if env_exe:
        cands.append(Path(env_exe).expanduser())
    env_dir = os.environ.get("LLAMA_SERVER_DIR")
    if env_dir:
        cands.append(Path(env_dir).expanduser() / "llama-server.exe")
    # Verified install layout on this machine (llama-b10964 builds)
    for variant in ("llama-b10964-cpu", "llama-b10964-vulkan"):
        cands.append(HOME / "llama-server" / "bin" / variant / "llama-server.exe")
    cands.append(HOME / "llama-server" / "llama-server.exe")
    cands.append(_NEW_SERVER_DIR / "llama-server.exe")
    cands.append(_LEGACY_SERVER_DIR / "llama-server.exe")  # legacy, kept last
    which = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if which:
        cands.append(Path(which))
    return cands


def resolve_server_exe() -> Path | None:
    """Return the first existing llama-server binary, or None if none found."""
    for cand in _candidate_server_exes():
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


# Back-compat constants: resolved lazily at import, but callers that need
# freshness (executor start) should call resolve_server_exe() at use time.
_RESOLVED = resolve_server_exe()
SERVER_EXE: Path = _RESOLVED or _candidate_server_exes()[2]
SERVER_DIR: Path = SERVER_EXE.parent


def server_exe_or_raise() -> Path:
    exe = resolve_server_exe()
    if exe is None:
        searched = "\n  ".join(str(c) for c in _candidate_server_exes())
        raise FileNotFoundError(
            "llama-server binary not found. Set LLAMA_SERVER_EXE or "
            f"LLAMA_SERVER_DIR. Searched:\n  {searched}"
        )
    return exe


# ── Model / data defaults ───────────────────────────────────────────────────
LMSTUDIO_MODELS: Path = _env_path("LMSTUDIO_MODELS", HOME / ".lmstudio" / "models")
USE_MODEL_DAT: Path = LMSTUDIO_MODELS / "use_model.dat"
HARVESTER_OUTPUT: Path = _env_path(
    "HARVESTER_OUTPUT",
    Path(os.environ.get("HARVESTER_OUTPUT", str(HOME / "AppData" / "Local" / "Temp" / "harvester_output")))
    if os.name == "nt"
    else HOME / "tmp" / "harvester_output",
)
