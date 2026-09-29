"""
desktop_controller.py  —  Standalone Desktop Controller for llama-server.

Clones and extends switch_model_tui.py with native hardware intelligence:
auto-populates models, mmproj pairs, embedders, backends, PAMS, and
free-threaded Python 3.14.5 agents for parallel processing — all in a
unified terminal interface with minimal resource overhead.

Sections (navigate with 1-9, ESC to quit):
  1. System Overview       — detected hardware, RAM, VRAM, backends
  2. Model Selection       — pick LLM, auto-paired with mmproj
  3. Embedder Selection    — embedding model for semantic search
  4. Backend Config        — CPU / Vulkan / GPU layers / threads
  5. Task Config           — chat / code / vision / embed / fast / research
  6. PAMS Control          — start / stop persistent-ai-memory
  7. Agent Launcher        — free-threaded parallel file/data agents
  8. Instance Monitor      — running servers, health, port status
  9. Full Launch           — combined launch with all selections

Usage:
    python desktop_controller.py
    python desktop_controller.py --reset    # Rescan hardware + models from scratch
"""

import os, sys, json, time, ctypes, subprocess, threading, itertools, webbrowser
from pathlib import Path
from dataclasses import dataclass, field, asdict
from concurrent.futures import as_completed
from typing import Optional

# ── Free-threading imports ────────────────────────────────────────────────
_ft_path = str(Path(__file__).resolve().parent.parent.parent / 'azazael' / 'src')
if _ft_path not in sys.path:
    sys.path.insert(0, _ft_path)
from CORE.freethreading import FreeThreadPool

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
CONFIG_DIR = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'unified-llm'
STATE_FILE = CONFIG_DIR / 'desktop_state.json'
DETECT_CACHE = CONFIG_DIR / 'detect_cache.json'
MODEL_CACHE = CONFIG_DIR / 'models.json'
INSTANCES_FILE = CONFIG_DIR / 'instances.json'

from paths import SERVER_DIR, SERVER_EXE, OV_LIBS, PAMS_DATA_DIR, PYTHON_DIR, PYTHON_FT, resolve_server_exe

PYTHON_EXE = str(PYTHON_DIR / 'python.exe')
PYTHON_FT_EXE = str(PYTHON_FT)

# ── Backend aliases ────────────────────────────────────────────────────────────

MODEL_PREFERENCE = ['gemma4', 'qwen35', 'llama', 'qwen3', 'qwen3vl', 'qwen2vl', 'qwen2']

ARCH_FLAGS = {
    'gemma4': ['--fit', 'off'],
    'qwen3vl': ['--fit', 'off'],
    'qwen2vl': [],
    'llama': [],
    'qwen2': [],
    'qwen3': [],
    'qwen35': [],
    'clip': [],
}

TASK_PRESETS = {
    'chat':     {'ctx': 8192,  'batch': 512,  'ubatch': 128,  'threads': 16, 'desc': 'General chat'},
    'code':     {'ctx': 16384, 'batch': 128,  'ubatch': 64,   'threads': 16, 'desc': 'Code generation'},
    'vision':   {'ctx': 8192,  'batch': 512,  'ubatch': 1024, 'threads': 16, 'desc': 'Vision + multimodal'},
    'embed':    {'ctx': 512,   'batch': 512,  'ubatch': 512,  'threads': 4,  'desc': 'Embeddings only'},
    'fast':     {'ctx': 2048,  'batch': 512,  'ubatch': 256,  'threads': 16, 'desc': 'Quick responses'},
    'research': {'ctx': 32768, 'batch': 512,  'ubatch': 1024, 'threads': 16, 'desc': 'Deep codebase analysis'},
}

VISION_ARCHS = {'qwen3vl', 'qwen2vl', 'gemma4', 'clip'}

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class DesktopState:
    model_path: str = ''
    model_name: str = ''
    model_arch: str = ''
    model_size_gb: float = 0.0
    model_ctx: int = 8192
    model_filename: str = ''
    model_blocks: int = 32

    mmproj_path: str = ''
    mmproj_name: str = ''

    embedder_path: str = ''
    embedder_name: str = ''
    embedder_arch: str = ''

    backend: str = 'cpu'
    ngl: int = 0
    threads: int = 16

    task_mode: str = 'chat'
    port: int = 8080
    ctx_size: int = 8192
    batch_size: int = 512
    ubatch_size: int = 128

    pams_enabled: bool = False
    agents_enabled: bool = False
    agent_count: int = 4
    agent_task: str = 'summarize'

    multi_serve: bool = False
    instance_count: int = 2

    last_updated: str = ''


def _default_state() -> DesktopState:
    return DesktopState()


def _load_state() -> DesktopState:
    if STATE_FILE.exists():
        try:
            d = json.loads(STATE_FILE.read_text())
            return DesktopState(**{k: d.get(k, v) for k, v in asdict(_default_state()).items()})
        except Exception:
            pass
    return _default_state()


def _save_state(state: DesktopState):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    state.last_updated = time.strftime('%Y-%m-%d %H:%M:%S')
    STATE_FILE.write_text(json.dumps(asdict(state), indent=2, default=str))


# ── Scanners ───────────────────────────────────────────────────────────────────

def _run_py(script: str, *args: str) -> tuple[str, str, int]:
    r = subprocess.run(
        [sys.executable, script] + list(args),
        capture_output=True, text=True, timeout=120,
    )
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def scan_hardware(force: bool = False) -> dict:
    if not force and DETECT_CACHE.exists():
        try:
            return json.loads(DETECT_CACHE.read_text())
        except Exception:
            pass
    hw_script = BASE_DIR / 'hardware_detect.py'
    if hw_script.exists():
        out, err, rc = _run_py(str(hw_script), '--json')
        if rc == 0 and out:
            try:
                data = json.loads(out)
                return data
            except json.JSONDecodeError:
                pass
    if DETECT_CACHE.exists():
        try:
            return json.loads(DETECT_CACHE.read_text())
        except Exception:
            pass
    return {'system': {}, 'gpu': [], 'backends': [], 'npu': None}


def scan_models(force: bool = False) -> list[dict]:
    sc_script = BASE_DIR / 'model_scanner.py'
    if force or not MODEL_CACHE.exists():
        if sc_script.exists():
            _run_py(str(sc_script))
    if MODEL_CACHE.exists():
        try:
            return json.loads(MODEL_CACHE.read_text()).get('models', [])
        except Exception:
            pass
    return []


def scan_embedders(models: list[dict]) -> list[dict]:
    embed_keywords = ['embed', 'nomic', 'text-embedding', 'e5-', 'bge-', 'gte-', 'jina-embed', 'mxbai']
    embedders = []
    for m in models:
        fn = m.get('filename', '').lower()
        arch = m.get('metadata', {}).get('architecture', '').lower()
        is_embed = any(k in fn for k in embed_keywords) or 'embed' in arch
        if is_embed and m.get('metadata', {}).get('block_count', 0) < 64:
            embedders.append(m)
            continue
        if m.get('metadata', {}).get('vocab_size', 0) == 0:
            embedders.append(m)
    return embedders


def scan_mmproj(models: list[dict]) -> list[dict]:
    return [m for m in models if m.get('metadata', {}).get('mmproj')]


def find_mmproj_for_model(model_path: str) -> str:
    mp = Path(model_path)
    if not mp.exists():
        return ''
    for f in mp.parent.iterdir():
        if f.suffix == '.gguf' and (f.name.startswith('mmproj') or '.mmproj' in f.name):
            return str(f)
    return ''


def detect_ft_python() -> bool:
    """Python 3.14t is the only target — always returns True."""
    return True


# ── Process helpers ────────────────────────────────────────────────────────────

def _process_exists(pid: int) -> bool:
    r = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                       capture_output=True, text=True, timeout=5)
    return str(pid) in r.stdout


def _kill_process(pid: int) -> bool:
    r = subprocess.run(['taskkill', '/PID', str(pid), '/F'],
                       capture_output=True, text=True, timeout=5)
    return r.returncode == 0


def get_running_instances() -> list[dict]:
    if not INSTANCES_FILE.exists():
        return []
    try:
        data = json.loads(INSTANCES_FILE.read_text())
        alive = []
        for k, v in data.items():
            if _process_exists(v.get('pid', 0)):
                import urllib.request
                healthy = False
                try:
                    r = urllib.request.urlopen(f'http://127.0.0.1:{v.get("port", 8080)}/health', timeout=2)
                    healthy = r.status == 200
                except Exception:
                    pass
                alive.append({
                    'pid': v['pid'], 'port': v['port'],
                    'model': v.get('model', '?'),
                    'backend': v.get('backend', '?'),
                    'healthy': healthy,
                    'label': v.get('label', ''),
                })
        return alive
    except Exception:
        return []


# ── Launchers ──────────────────────────────────────────────────────────────────

def launch_server(state: DesktopState) -> int:
    merged = dict(os.environ)
    merged['GGML_OPENVINO_DISABLE'] = '1'
    merged['GGML_OPENVINO_BLACKLIST'] = '1'
    ov_path = str(OV_LIBS)
    if ov_path not in merged.get('PATH', ''):
        merged['PATH'] = f'{ov_path};{merged["PATH"]}'
    if not state.model_path or not os.path.exists(state.model_path):
        return -1
    args = [
        '--model', state.model_path,
        '--host', '127.0.0.1',
        '--port', str(state.port),
        '--ctx-size', str(state.ctx_size),
        '--threads', str(state.threads),
        '--batch-size', str(state.batch_size),
        '--ubatch-size', str(state.ubatch_size),
        '--no-warmup',
    ]
    if state.mmproj_path and os.path.exists(state.mmproj_path):
        args.extend(['--mmproj', state.mmproj_path])
    elif state.model_arch in VISION_ARCHS:
        auto_mm = find_mmproj_for_model(state.model_path)
        if auto_mm:
            args.extend(['--mmproj', auto_mm])
    if state.task_mode != 'vision':
        args.extend(['--jinja'])
    args.extend(['-ngl', '0'])
    arch_flags = ARCH_FLAGS.get(state.model_arch, [])
    args.extend(arch_flags)
    if state.task_mode == 'embed':
        args.extend(['--embedding', '--pooling', 'mean'])
    proc = subprocess.Popen(
        [str(SERVER_EXE)] + args,
        env=merged,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
    )
    from executor import ServerInstance, _register_instance
    inst = ServerInstance(
        pid=proc.pid, port=state.port, model=state.model_name or state.model_filename,
        backend=state.backend, label=f'llama-server-{state.port}', started=time.time(),
    )
    _register_instance(inst)
    return proc.pid


def wait_health(port: int, timeout: int = 120) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def launch_pams(state: DesktopState) -> int:
    env = dict(os.environ)
    env['AI_MEMORY_DATA_DIR'] = str(PAMS_DATA_DIR)
    launcher = BASE_DIR / '_pams_launcher.py'
    if not launcher.exists():
        return -1
    proc = subprocess.Popen(
        [sys.executable, str(launcher)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
    )
    pid_line = proc.stdout.readline().decode().strip()
    return int(pid_line) if pid_line.isdigit() else 0


def launch_agents(state: DesktopState) -> list[int]:
    ft_path = PYTHON_FT if os.path.exists(PYTHON_FT) else sys.executable
    harvester = BASE_DIR / 'harvester.py'
    if not harvester.exists():
        return []
    pids = []
    with FreeThreadPool() as pool:
        futures = []
        for i in range(state.agent_count):
            f = pool.submit(
                subprocess.Popen,
                [ft_path, str(harvester),
                 '--root', str(Path.home() / 'llama-server'),
                 '--model', state.model_path,
                 '--workers', '1'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            futures.append(f)
        for f in as_completed(futures):
            proc = f.result()
            if proc:
                pids.append(proc.pid)
    return pids


def stop_all_instances():
    if INSTANCES_FILE.exists():
        try:
            data = json.loads(INSTANCES_FILE.read_text())
            for k, v in data.items():
                pid = v.get('pid', 0)
                if pid and _process_exists(pid):
                    _kill_process(pid)
            INSTANCES_FILE.write_text('{}')
        except Exception:
            pass


# ── TUI / Console helpers ──────────────────────────────────────────────────────

def _read_console_key():
    STDIN = -10
    ENABLE_PROCESSED_INPUT = 0x0001
    hIn = ctypes.windll.kernel32.GetStdHandle(STDIN)
    if hIn in (0, -1):
        return None, 0
    orig = ctypes.c_ulong()
    ctypes.windll.kernel32.GetConsoleMode(hIn, ctypes.byref(orig))
    ctypes.windll.kernel32.SetConsoleMode(hIn, ENABLE_PROCESSED_INPUT)
    buf = (ctypes.c_byte * 4)()
    nRead = ctypes.c_ulong()
    ok = ctypes.windll.kernel32.ReadConsoleA(
        ctypes.c_void_p(hIn), ctypes.c_void_p(ctypes.addressof(buf)),
        ctypes.c_ulong(4), ctypes.byref(nRead), None,
    )
    ctypes.windll.kernel32.SetConsoleMode(hIn, orig)
    if not ok or nRead.value == 0:
        return None, 0
    b0 = buf[0] & 0xFF
    if nRead.value == 1:
        return chr(b0), b0
    vk = buf[1] & 0xFF
    return None, vk


def read_key() -> str:
    _VK = {
        0x08: 'Backspace', 0x09: 'Tab', 0x0D: 'Enter',
        0x1B: 'Escape', 0x21: 'PageUp', 0x22: 'PageDown',
        0x23: 'End', 0x24: 'Home', 0x2D: 'Insert', 0x2E: 'Delete',
        0x26: 'UpArrow', 0x28: 'DownArrow',
        0x25: 'LeftArrow', 0x27: 'RightArrow',
    }
    while True:
        _ch, vk = _read_console_key()
        if _ch is None and vk:
            r = _VK.get(vk)
            if r is not None:
                return r
            continue
        if _ch == '\r':
            return 'Enter'
        if _ch == '\x03':
            raise KeyboardInterrupt
        if _ch is not None:
            return _ch


def cls():
    os.system('cls')


def hr(title='', char='=', width=70):
    if title:
        side = (width - len(title) - 2) // 2
        return f'{char * side} {title} {char * side}'
    return char * width


# ── Section Renderers ──────────────────────────────────────────────────────────

def render_system(hw: dict, state: DesktopState) -> str:
    s = hw.get('system', {})
    gpus = hw.get('gpu', [])
    backends = hw.get('backends', [])
    npu = hw.get('npu')
    lines = [
        hr(' System Overview '),
        f'  CPU:  {s.get("cpu_name", "?")}',
        f'  Cores: {s.get("physical_cores", "?")}P / {s.get("logical_cores", "?")}L',
        f'  RAM:  {s.get("total_ram_gb", "?"):>5.1f} GB total, {s.get("available_ram_gb", "?"):>5.1f} GB free',
        f'  Features: {s.get("cpu_features", [])}',
    ]
    for g in gpus:
        vk = 'VK' if g.get('vulkan_available') else '--'
        lines.append(f'  GPU:  {g.get("name", "?")} ({g.get("vram_gb", "?"):>4.1f} GB, {vk})')
    if npu:
        lines.append(f'  NPU:  {npu.get("name", "Intel AI Boost")}')
    lines.append(f'  Python: 3.14.5 + free-threaded')
    lines.append(f'')
    lines.append(hr(' Available Backends '))
    for b in backends:
        icon = 'OK' if b.get('available') else '--'
        eng = f' [{", ".join(b.get("engines", []))}]' if b.get('engines') else ''
        lines.append(f'  {b.get("name", "?"):<20} {icon}{eng}')
    lines.append(f'')
    lines.append(f'  Current selection: {state.backend} ({state.ngl} GPU layers, {state.threads} threads)')
    lines.append(f'  Current task:      {state.task_mode} ({TASK_PRESETS.get(state.task_mode, {}).get("desc", "")})')
    lines.append(f'')
    lines.append(f'  [ENTER] Refresh   [1-9] Navigate   [Q] Quit')
    return '\n'.join(lines)


def render_models(models: list[dict], state: DesktopState, sel: int = 0) -> str:
    lines = [
        hr(' Model Selection '),
        f'  Scanned {len(models)} GGUF models. * = current, ==> = cursor',
        hr(),
        f'  #  {"Name":<50} {"Size":>6}  {"Arch":<12}  {"Ctx":>6}',
        hr('-'),
    ]
    for i, m in enumerate(models):
        meta = m.get('metadata', {})
        name = meta.get('name') or m.get('filename', '?')
        arch = meta.get('architecture', '?')
        ctx = meta.get('context_length', '?')
        size = m.get('size_gb', 0)
        mmproj = ' [V]' if meta.get('mmproj') else ''
        sel_mark = '==>' if i == sel else '   '
        cur_mark = '*' if m.get('path', '') == state.model_path else ' '
        truncated = name[:48] if len(name) > 48 else name
        lines.append(f'  {cur_mark}{sel_mark} {i+1:>3}. {truncated:<48} {size:>5.1f}  {arch:<12} {str(ctx):>6}{mmproj}')
    lines.append(f'')
    lines.append(f'  Current: {state.model_name or state.model_filename or "(none)"}')
    if state.mmproj_path:
        lines.append(f'  mmproj:  {Path(state.mmproj_path).name}')
    lines.append(f'  [UP/DOWN] move   [ENTER] select   [V] view details   [Q] back')
    return '\n'.join(lines)


def render_embedders(embedders: list[dict], state: DesktopState, sel: int = 0) -> str:
    lines = [
        hr(' Embedder Selection '),
        f'  Found {len(embedders)} embedding-capable models.',
        hr(),
    ]
    if not embedders:
        lines.append(f'  No dedicated embedding models found.')
        lines.append(f'  (Will use nomic-embed-text via Ollama or main model for embeddings)')
    else:
        for i, m in enumerate(embedders):
            meta = m.get('metadata', {})
            name = meta.get('name') or m.get('filename', '?')
            arch = meta.get('architecture', '?')
            size = m.get('size_gb', 0)
            sel_mark = '==>' if i == sel else '   '
            cur_mark = '*' if m.get('path', '') == state.embedder_path else ' '
            lines.append(f'  {cur_mark}{sel_mark} {i+1:>3}. {name:<50} {size:>5.1f} GB  {arch}')
    lines.append(f'')
    lines.append(f'  Current: {state.embedder_name or "(none — using Ollama nomic-embed)"}')
    lines.append(f'  [UP/DOWN] move   [ENTER] select   [E] clear   [Q] back')
    return '\n'.join(lines)


def render_backends(hw: dict, state: DesktopState, sel_opt: int = 0) -> str:
    gpus = hw.get('gpu', [])
    vram = gpus[0].get('vram_gb', 0) if gpus else 0
    options = [
        ('CPU', '47.4 tok/s gen, 169.9 tok/s prompt (recommended)'),
        ('CPU (low threads)', '4 threads, for multi-instance'),
    ]
    lines = [
        hr(' Backend Configuration '),
        f'  VRAM: {vram} GB  |  Cores: {hw.get("system", {}).get("physical_cores", 16)}',
        hr(),
        f'  Benchmark (qwen2.5-1.5b, 2026-05-22):',
        f'    CPU:     47.4 tok/s gen, 169.9 tok/s prompt',
        f'    Vulkan:  37.6 tok/s gen,  73.1 tok/s prompt',
        f'  CPU is faster on ALL metrics for this hardware.',
        hr(),
    ]
    for i, (name, desc) in enumerate(options):
        mark = '==>' if i == sel_opt else '   '
        is_cur = (i == 0 and state.backend == 'cpu' and state.threads >= 8) or \
                 (i == 1 and state.backend == 'cpu' and state.threads <= 4)
        cur = '*' if is_cur else ' '
        lines.append(f'  {cur}{mark} {i+1}. {name:<30} — {desc}')
    lines.append(f'')
    lines.append(f'  Current: {state.backend}, ngl=0, threads={state.threads}')
    lines.append(f'  [UP/DOWN] move   [ENTER] select   [Q] back')
    return '\n'.join(lines)


def render_tasks(state: DesktopState, sel: int = 0) -> str:
    modes = list(TASK_PRESETS.keys())
    lines = [
        hr(' Task Configuration '),
        f'  Select a task mode to auto-configure context, batch, and ubatch.',
        hr(),
        f'  #  {"Mode":<12} {"Ctx":>6} {"Batch":>6} {"Ubatch":>6} {"Threads":>8}  Description',
        hr('-'),
    ]
    for i, mode in enumerate(modes):
        cfg = TASK_PRESETS[mode]
        mark = '==>' if i == sel else '   '
        cur = '*' if mode == state.task_mode else ' '
        lines.append(f'  {cur}{mark} {i+1}. {mode:<12} {cfg["ctx"]:>6} {cfg["batch"]:>6} {cfg["ubatch"]:>6} {cfg["threads"]:>8}  {cfg["desc"]}')
    lines.append(f'')
    lines.append(f'  Current: {state.task_mode} — ctx={state.ctx_size} batch={state.batch_size}')
    lines.append(f'  [UP/DOWN] move   [ENTER] select   [C] custom   [Q] back')
    return '\n'.join(lines)


def render_pams(state: DesktopState) -> str:
    pams_running = False
    port_file = PAMS_DATA_DIR / 'mcp_server_port.json'
    pams_port = 0
    if port_file.exists():
        try:
            data = json.loads(port_file.read_text())
            pams_port = int(data.get('active_port', 0))
            if pams_port > 0:
                import urllib.request
                try:
                    r = urllib.request.urlopen(f'http://127.0.0.1:{pams_port}/api/health', timeout=2)
                    pams_running = r.status == 200
                except Exception:
                    pass
        except Exception:
            pass
    instances = get_running_instances()
    pids = [i for i in instances if i.get('label') == 'pams-server']
    lines = [
        hr(' PAMS Persistent Memory '),
        f'  Data dir: {PAMS_DATA_DIR}',
        hr(),
        f'  Server status: {"RUNNING" if pams_running else "STOPPED"}',
        f'  HTTP port:     {pams_port if pams_port else "N/A"}',
        f'  Auto-start:    {"ON" if state.pams_enabled else "OFF"}',
        f'',
        f'  [1] Toggle auto-start (currently {"ON" if state.pams_enabled else "OFF"})',
        f'  [S] Start PAMS server',
        f'  [K] Stop PAMS server',
        f'  [H] Health check',
        f'  [Q] Back',
    ]
    return '\n'.join(lines)


def render_agents(state: DesktopState, sel_count: int = 0, sel_task: int = 0) -> str:
    task_options = ['summarize', 'code-analyze', 'security-scan', 'embed-generate']
    lines = [
        hr(' Agent Launcher (Free-threaded Python 3.14.5) '),
        f'  Free-threaded Python: AVAILABLE',
        f'  Python path: {PYTHON_FT if os.path.exists(PYTHON_FT) else sys.executable}',
        hr(),
        f'  Agent count ({state.agent_count}):',
    ]
    for i in range(8):
        mark = '==>' if i == sel_count else '   '
        num = i + 1
        lines.append(f'  {mark} [{num}] {num} agent{"s" if num > 1 else ""}')
    lines.append(f'')
    lines.append(f'  Task:')
    for i, t in enumerate(task_options):
        mark = '==>' if i == sel_task else '   '
        cur = '*' if t == state.agent_task else ' '
        lines.append(f'  {cur}{mark} [{i+1}] {t}')
    lines.append(f'')
    lines.append(f'  Current: {state.agent_count} agents, task="{state.agent_task}"')
    if state.agents_enabled:
        lines.append(f'  Auto-launch on start: ON')
    lines.append(f'  [UP/DOWN] move count   [LEFT/RIGHT] switch panel   [ENTER] toggle')
    lines.append(f'  [L] Launch agents now   [A] Toggle auto-launch   [Q] Back')
    return '\n'.join(lines)


def render_instances(state: DesktopState) -> str:
    instances = get_running_instances()
    lines = [
        hr(' Instance Monitor '),
        f'  Total running: {len(instances)}',
        hr(),
    ]
    if not instances:
        lines.append(f'  No running instances.')
        lines.append(f'  Use section 9 (Full Launch) to start servers.')
    else:
        for inst in instances:
            icon = '[OK]' if inst.get('healthy') else '[..]'
            lines.append(f'  {icon}  port={inst["port"]:<5}  {inst.get("model", "?"):<45}  '
                         f'{inst.get("backend", "?"):<8}  pid={inst["pid"]}')
    lines.append(f'')
    lines.append(f'  [R] Refresh   [S] Stop all   [K #] Kill by port   [Q] Back')
    return '\n'.join(lines)


def render_launch(state: DesktopState, hw: dict) -> str:
    est_tps = 47.4 if state.model_size_gb <= 2.0 else (22.2 if state.model_size_gb <= 3.0 else 10.0)
    lines = [
        hr(' Full Launch Configuration '),
        f'  {hr(char="-")}',
        hr(' Model '),
        f'    {state.model_name or state.model_filename or "(not selected)"}',
        f'    {state.model_path or ""}',
        f'    mmproj: {Path(state.mmproj_path).name if state.mmproj_path and os.path.exists(state.mmproj_path) else "(none)"}',
        hr(' Configuration '),
        f'    Backend:    {state.backend} ({state.ngl} GPU layers)',
        f'    Task:       {state.task_mode}',
        f'    Context:    {state.ctx_size}  Batch: {state.batch_size}  Ubatch: {state.ubatch_size}',
        f'    Threads:    {state.threads}',
        f'    Port:       {state.port}',
        f'    Speed est:  ~{est_tps} tok/s',
        hr(' PAMS + Agents '),
        f'    PAMS:  {"ON" if state.pams_enabled else "OFF"}',
        f'    Agents: {"ON" if state.agents_enabled else "OFF"} ({state.agent_count})',
        hr(),
        f'  [ENTER] Launch all   [S] Save config   [L] Load config',
        f'  [O] Open browser UI   [Q] Back',
    ]
    return '\n'.join(lines)


# ── Section Handlers ───────────────────────────────────────────────────────────

def section_system(hw: dict, state: DesktopState) -> tuple[DesktopState, str | None]:
    cls()
    print(render_system(hw, state))
    while True:
        key = read_key()
        if key == 'Enter':
            cls()
            print(render_system(scan_hardware(force=True), state))
            continue
        if key in ('Q', 'Escape'):
            return state, None
        if key in ('1','2','3','4','5','6','7','8','9'):
            return state, key


def section_models(models: list[dict], state: DesktopState) -> tuple[DesktopState, str | None]:
    if not models:
        cls()
        print(hr(' Model Selection '))
        print(f'  No models found. Scan with --reset or add GGUF files to:')
        print(f'    {Path.home() / "llama-server" / "models"}')
        print(f'  Press any key...')
        read_key()
        return state, None
    sel = 0
    for i, m in enumerate(models):
        if m.get('path', '') == state.model_path:
            sel = i
            break
    while True:
        cls()
        print(render_models(models, state, sel))
        key = read_key()
        if key == 'UpArrow' and sel > 0:
            sel -= 1
        elif key == 'DownArrow' and sel < len(models) - 1:
            sel += 1
        elif key == 'Enter':
            m = models[sel]
            state.model_path = m.get('path', '')
            state.model_name = m.get('metadata', {}).get('name', '') or m.get('filename', '')
            state.model_arch = m.get('metadata', {}).get('architecture', '')
            state.model_size_gb = m.get('size_gb', 0)
            state.model_ctx = m.get('metadata', {}).get('context_length', 8192)
            state.model_filename = m.get('filename', '')
            state.model_blocks = m.get('metadata', {}).get('block_count', 32)
            mm_name = m.get('metadata', {}).get('mmproj')
            if mm_name:
                state.mmproj_path = str(Path(state.model_path).parent / mm_name)
                state.mmproj_name = mm_name
            else:
                auto_mm = find_mmproj_for_model(state.model_path)
                if auto_mm:
                    state.mmproj_path = auto_mm
                    state.mmproj_name = Path(auto_mm).name
                else:
                    state.mmproj_path = ''
                    state.mmproj_name = ''
            if state.model_arch in VISION_ARCHS and state.task_mode == 'chat':
                state.task_mode = 'vision'
                tc = TASK_PRESETS['vision']
                state.ctx_size = tc['ctx']
                state.batch_size = tc['batch']
                state.ubatch_size = tc['ubatch']
            _save_state(state)
            # Stay in section for quick browsing — use Q or number keys to exit
        elif key == 'V':
            m = models[sel]
            meta = m.get('metadata', {})
            cls()
            print(hr(f' Model Details: {m.get("filename", "?")} '))
            for k, v in meta.items():
                print(f'  {k}: {v}')
            print(f'  size_gb: {m.get("size_gb", 0):.2f}')
            print(f'  path: {m.get("path", "")}')
            print(f'\n  Press any key...')
            read_key()
        elif key in ('Q', 'Escape'):
            return state, None
        elif key in ('1','2','3','4','5','6','7','8','9'):
            return state, key


def section_embedders(embedders: list[dict], state: DesktopState) -> tuple[DesktopState, str | None]:
    sel = 0
    for i, m in enumerate(embedders):
        if m.get('path', '') == state.embedder_path:
            sel = i
            break
    while True:
        cls()
        print(render_embedders(embedders, state, sel))
        key = read_key()
        if key == 'UpArrow' and sel > 0:
            sel -= 1
        elif key == 'DownArrow' and sel < len(embedders) - 1:
            sel += 1
        elif key == 'Enter':
            if embedders:
                m = embedders[sel]
                state.embedder_path = m.get('path', '')
                state.embedder_name = m.get('metadata', {}).get('name', '') or m.get('filename', '')
                state.embedder_arch = m.get('metadata', {}).get('architecture', '')
            _save_state(state)
            return state, None
        elif key == 'E':
            state.embedder_path = ''
            state.embedder_name = ''
            state.embedder_arch = ''
            _save_state(state)
            return state, None
        elif key in ('Q', 'Escape'):
            return state, None
        elif key in ('1','2','3','4','5','6','7','8','9'):
            return state, key


def section_backends(hw: dict, state: DesktopState) -> tuple[DesktopState, str | None]:
    options = ['cpu', 'cpu-low']
    sel = 0
    for i, o in enumerate(options):
        if o == 'cpu' and state.backend == 'cpu' and state.threads >= 8:
            sel = i
        elif o == 'cpu-low' and state.backend == 'cpu' and state.threads <= 4:
            sel = i
    while True:
        cls()
        print(render_backends(hw, state, sel))
        key = read_key()
        if key == 'UpArrow' and sel > 0:
            sel -= 1
        elif key == 'DownArrow' and sel < len(options) - 1:
            sel += 1
        elif key == 'Enter':
            opt = options[sel]
            if opt == 'cpu':
                state.backend = 'cpu'
                state.ngl = 0
                state.threads = 16
            elif opt == 'cpu-low':
                state.backend = 'cpu'
                state.ngl = 0
                state.threads = 4
            _save_state(state)
            return state, None
        elif key in ('Q', 'Escape'):
            return state, None
        elif key in ('1','2','3','4','5','6','7','8','9'):
            return state, key


def section_tasks(state: DesktopState) -> tuple[DesktopState, str | None]:
    modes = list(TASK_PRESETS.keys())
    sel = 0
    for i, m in enumerate(modes):
        if m == state.task_mode:
            sel = i
            break
    while True:
        cls()
        print(render_tasks(state, sel))
        key = read_key()
        if key == 'UpArrow' and sel > 0:
            sel -= 1
        elif key == 'DownArrow' and sel < len(modes) - 1:
            sel += 1
        elif key == 'Enter':
            mode = modes[sel]
            state.task_mode = mode
            cfg = TASK_PRESETS[mode]
            state.ctx_size = cfg['ctx']
            state.batch_size = cfg['batch']
            state.ubatch_size = cfg['ubatch']
            state.threads = cfg['threads']
            _save_state(state)
            return state, None
        elif key == 'C':
            cls()
            print(hr(' Custom Task Configuration '))
            old_ctx, old_batch, old_ubatch, old_threads = state.ctx_size, state.batch_size, state.ubatch_size, state.threads
            try:
                ctx = input(f'  Context size [{state.ctx_size}]: ').strip()
                if ctx: state.ctx_size = int(ctx)
                batch = input(f'  Batch size [{state.batch_size}]: ').strip()
                if batch: state.batch_size = int(batch)
                ubatch = input(f'  Ubatch size [{state.ubatch_size}]: ').strip()
                if ubatch: state.ubatch_size = int(ubatch)
                threads = input(f'  Threads [{state.threads}]: ').strip()
                if threads: state.threads = int(threads)
            except ValueError:
                pass
            if (state.ctx_size != old_ctx or state.batch_size != old_batch or
                state.ubatch_size != old_ubatch or state.threads != old_threads):
                state.task_mode = 'custom'
            _save_state(state)
            return state, None
        elif key in ('Q', 'Escape'):
            return state, None
        elif key in ('1','2','3','4','5','6','7','8','9'):
            return state, key


def section_pams(state: DesktopState) -> tuple[DesktopState, str | None]:
    while True:
        cls()
        print(render_pams(state))
        key = read_key()
        if key == '1':
            state.pams_enabled = not state.pams_enabled
            _save_state(state)
        elif key == 'S':
            cls()
            print(hr(' Starting PAMS... '))
            pid = launch_pams(state)
            if pid > 0:
                print(f'  PAMS PID: {pid}')
            else:
                print(f'  Failed to start PAMS')
            print(f'  Press any key...')
            read_key()
        elif key == 'K':
            try:
                import urllib.request
                port_file = PAMS_DATA_DIR / 'mcp_server_port.json'
                if port_file.exists():
                    data = json.loads(port_file.read_text())
                    port = int(data.get('active_port', 0))
                    if port:
                        r = urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=2)
                        print(f'  Can\'t stop via HTTP — use task manager')
            except Exception:
                pass
            print(f'  Stopping all pams-server instances...')
            r = subprocess.run(['taskkill', '/IM', 'pams-server.exe', '/F'],
                               capture_output=True, text=True, timeout=5)
            print(f'  Result: {r.stdout.strip() or r.stderr.strip() or "OK"}')
            print(f'  Press any key...')
            read_key()
        elif key == 'H':
            cls()
            print(hr(' PAMS Health Check '))
            port_file = PAMS_DATA_DIR / 'mcp_server_port.json'
            if port_file.exists():
                data = json.loads(port_file.read_text())
                print(f'  Port file: {json.dumps(data, indent=2)}')
            else:
                print(f'  No port file found at {port_file}')
            try:
                import urllib.request
                for port in [21435, 21436]:
                    try:
                        r = urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=2)
                        print(f'  Port {port}: HTTP {r.status}')
                    except Exception as e:
                        print(f'  Port {port}: {e}')
            except Exception:
                pass
            print(f'\n  Press any key...')
            read_key()
        elif key in ('Q', 'Escape'):
            return state, None
        elif key in ('2','3','4','5','6','7','8','9'):
            return state, key


# ── Agent panel state persistence ──────────────────────────────────────┐
_last_agent_panel = 0

def section_agents(state: DesktopState) -> tuple[DesktopState, str | None]:
    global _last_agent_panel
    sel_panel = _last_agent_panel  # 0=count, 1=task
    sel_count = max(0, min(state.agent_count - 1, 7))
    task_options = ['summarize', 'code-analyze', 'security-scan', 'embed-generate']
    sel_task = 0
    for i, t in enumerate(task_options):
        if t == state.agent_task:
            sel_task = i
            break
    while True:
        cls()
        print(render_agents(state, sel_count, sel_task))
        key = read_key()
        if key == 'UpArrow':
            if sel_panel == 0 and sel_count > 0:
                sel_count -= 1
            elif sel_panel == 1 and sel_task > 0:
                sel_task -= 1
                _last_agent_panel = sel_panel
        elif key == 'DownArrow':
            if sel_panel == 0 and sel_count < 7:
                sel_count += 1
            elif sel_panel == 1 and sel_task < len(task_options) - 1:
                sel_task += 1
                _last_agent_panel = sel_panel
        elif key == 'LeftArrow' and sel_panel > 0:
            sel_panel -= 1
            _last_agent_panel = sel_panel
        elif key == 'RightArrow' and sel_panel < 1:
            sel_panel += 1
            _last_agent_panel = sel_panel
        elif key == 'Enter':
            state.agent_count = sel_count + 1
            state.agent_task = task_options[sel_task]
            _save_state(state)
            # Stay in section for further adjustments
        elif key == 'L':
            state.agent_count = sel_count + 1
            state.agent_task = task_options[sel_task]
            _save_state(state)
            cls()
            print(hr(f' Launching {state.agent_count} Agents... '))
            pids = launch_agents(state)
            if pids:
                print(f'  Launched {len(pids)} agent(s): {pids}')
            else:
                print(f'  Agent launch failed or harvester.py not found')
            print(f'  Press any key...')
            read_key()
        elif key == 'A':
            state.agents_enabled = not state.agents_enabled
            _save_state(state)
        elif key in ('Q', 'Escape'):
            return state, None
        elif key in ('1','2','3','4','5','6','7','8','9'):
            return state, key


def section_instances(state: DesktopState) -> tuple[DesktopState, str | None]:
    while True:
        cls()
        print(render_instances(state))
        key = read_key()
        if key == 'R':
            continue
        elif key == 'S':
            stop_all_instances()
            cls()
            print(hr(' Stopped All Instances '))
            print(f'  All server instances killed.')
            print(f'  Press any key...')
            read_key()
        elif key == 'K':
            cls()
            print(hr(' Kill Instance '))
            port_str = input(f'  Enter port number to kill: ').strip()
            try:
                port = int(port_str)
                instances = get_running_instances()
                for inst in instances:
                    if inst['port'] == port:
                        _kill_process(inst['pid'])
                        print(f'  Killed PID {inst["pid"]} on port {port}')
                        break
                else:
                    print(f'  No instance on port {port}')
            except ValueError:
                print(f'  Invalid port')
            print(f'  Press any key...')
            read_key()
        elif key in ('Q', 'Escape'):
            return state, None
        elif key in ('1','2','3','4','5','6','7','8','9'):
            return state, key


def section_launch(hw: dict, state: DesktopState) -> tuple[DesktopState, str | None]:
    while True:
        cls()
        print(render_launch(state, hw))
        key = read_key()
        if key == 'Enter':
            cls()
            print(hr(' Launching... '))

            if state.pams_enabled:
                print(f'  Starting PAMS...')
                pid = launch_pams(state)
                print(f'    PAMS PID: {pid if pid > 0 else "FAILED"}')

            if state.model_path and os.path.exists(state.model_path):
                print(f'  Starting llama-server on port {state.port}...')
                pid = launch_server(state)
                if pid > 0:
                    print(f'    PID: {pid}')
                    print(f'    Waiting for health...')
                    if wait_health(state.port):
                        print(f'    READY on http://127.0.0.1:{state.port}')
                    else:
                        print(f'    TIMEOUT')
                else:
                    print(f'    FAILED (model path: {state.model_path})')
            else:
                print(f'    No model selected')

            if state.agents_enabled:
                print(f'  Launching {state.agent_count} agents...')
                pids = launch_agents(state)
                print(f'    Agent PIDs: {pids}')

            print(f'\n  Press any key to return...')
            read_key()

        elif key == 'S':
            _save_state(state)
            cls()
            print(hr(' Saved '))
            print(f'  Config saved to {STATE_FILE}')
            print(f'  Press any key...')
            read_key()
        elif key == 'L':
            state = _load_state()
            cls()
            print(hr(' Loaded '))
            print(f'  Config loaded from {STATE_FILE}')
            print(f'  Press any key...')
            read_key()
        elif key == 'O':
            webbrowser.open(f'http://127.0.0.1:{state.port}/chat')
        elif key in ('Q', 'Escape'):
            return state, None
        elif key in ('1','2','3','4','5','6','7','8','9'):
            return state, key


# ── Main Loop ──────────────────────────────────────────────────────────────────

HEADER_ART = r"""
  +----------------------------------------------------------------------+
  |   UNIFIED DESKTOP CONTROLLER — llama-server                          |
  |   Native model selector  |  mmproj pairing  |  PAMS  |  Free-thread  |
  +----------------------------------------------------------------------+
  |   1 System      2 Model       3 Embedder    4 Backend    5 Task      |
  |   6 PAMS        7 Agents      8 Monitor     9 Launch      Q Quit     |
  +----------------------------------------------------------------------+
"""


def main():
    if '--reset' in sys.argv:
        for f in [DETECT_CACHE, MODEL_CACHE, STATE_FILE]:
            if f.exists():
                f.unlink()
        print('Cache cleared. Re-scanning...')

    hw = scan_hardware()
    models = scan_models()
    embedders = scan_embedders(models)

    state = _load_state()

    # Auto-select best model + mmproj + embedder on first run
    if not state.model_path and models:
        best_score = lambda m: (
            MODEL_PREFERENCE.index(m.get('metadata', {}).get('architecture', ''))
            if m.get('metadata', {}).get('architecture', '') in MODEL_PREFERENCE else 99,
            -m.get('size_bytes', 0),
        )
        scored = sorted(
            [m for m in models if not m['filename'].startswith('mmproj-')
             and m.get('metadata', {}).get('architecture', '') not in ('clip', '')],
            key=best_score,
        )
        if scored:
            best = scored[0]
            state.model_path = best.get('path', '')
            state.model_name = best.get('metadata', {}).get('name', '') or best.get('filename', '')
            state.model_arch = best.get('metadata', {}).get('architecture', '')
            state.model_size_gb = best.get('size_gb', 0)
            state.model_ctx = best.get('metadata', {}).get('context_length', 8192)
            state.model_filename = best.get('filename', '')
            state.model_blocks = best.get('metadata', {}).get('block_count', 32)
            mm_name = best.get('metadata', {}).get('mmproj')
            if mm_name:
                state.mmproj_path = str(Path(state.model_path).parent / mm_name)
                state.mmproj_name = mm_name
            else:
                auto_mm = find_mmproj_for_model(state.model_path)
                if auto_mm:
                    state.mmproj_path = auto_mm
                    state.mmproj_name = Path(auto_mm).name
            if embedders:
                emb = embedders[0]
                state.embedder_path = emb.get('path', '')
                state.embedder_name = emb.get('metadata', {}).get('name', '') or emb.get('filename', '')
                state.embedder_arch = emb.get('metadata', {}).get('architecture', '')
            if state.model_arch in VISION_ARCHS and state.task_mode == 'chat':
                state.task_mode = 'vision'
                tc = TASK_PRESETS['vision']
                state.ctx_size = tc['ctx']
                state.batch_size = tc['batch']
                state.ubatch_size = tc['ubatch']
            _save_state(state)

    while True:
        try:
            cls()
            print(HEADER_ART)

            s = hw.get('system', {})
            gpus = hw.get('gpu', [])
            instances = get_running_instances()
            pams_running = PAMS_DATA_DIR.joinpath('mcp_server_port.json').exists()

            total_models = len(models)
            vision_models = sum(1 for m in models if m.get('metadata', {}).get('mmproj'))
            main_models = total_models - sum(
                1 for m in models
                if m.get('filename', '').startswith('mmproj-')
                or m.get('metadata', {}).get('architecture', '') == 'clip'
            )
            embed_models = len(embedders)

            model_display = state.model_name or state.model_filename or '(none)'
            if len(model_display) > 40:
                model_display = model_display[:37] + '...'

            mm_display = Path(state.mmproj_path).name if state.mmproj_path else '(none)'
            if len(mm_display) > 30:
                mm_display = mm_display[:27] + '...'

            print(f'  Hardware: {s.get("cpu_name", "")[:50]}')
            print(f'  RAM: {s.get("total_ram_gb", "?"):>4.1f} GB  VRAM: {gpus[0].get("vram_gb", "?"):>4.1f} GB  '
                  f'Cores: {s.get("physical_cores", "?")}  FT: Y')
            print(f'  Library: {total_models} GGUF models ({main_models} LLM, {vision_models} vision, {embed_models} embed)')
            print(f'  Active:  {model_display}')
            if state.mmproj_path:
                print(f'  mmproj:  {mm_display}')
            print(f'  Backend: {state.backend} ngl={state.ngl}  Task: {state.task_mode}  '
                  f'Port: {state.port}  PAMS: {"ON" if state.pams_enabled else "OFF"}')
            print(f'  Running: {len(instances)} instance(s)  '
                  f'Agents: {"ON" if state.agents_enabled else "OFF"} ({state.agent_count})')
            print(f'  {hr(char="-")}')
            print(f'  [1] System    [2] Model     [3] Embedder  [4] Backend   [5] Task')
            print(f'  [6] PAMS      [7] Agents    [8] Monitor   [9] Launch    [Q] Quit')
            print(f'  {hr(char="-")}')
            sys.stdout.write('  Select: ')
            sys.stdout.flush()

            key = read_key()

            while True:
                nav_key = None

                if key in ('Q', 'Escape'):
                    cls()
                    print(hr(' Goodbye '))
                    sys.exit(0)

                elif key == '1':
                    state, nav_key = section_system(hw, state)
                    hw = scan_hardware()

                elif key == '2':
                    models = scan_models()
                    state, nav_key = section_models(models, state)

                elif key == '3':
                    models = scan_models()
                    embedders = scan_embedders(models)
                    state, nav_key = section_embedders(embedders, state)

                elif key == '4':
                    hw = scan_hardware()
                    state, nav_key = section_backends(hw, state)

                elif key == '5':
                    state, nav_key = section_tasks(state)

                elif key == '6':
                    state, nav_key = section_pams(state)

                elif key == '7':
                    state, nav_key = section_agents(state)

                elif key == '8':
                    state, nav_key = section_instances(state)

                elif key == '9':
                    hw = scan_hardware()
                    state, nav_key = section_launch(hw, state)

                if nav_key is None:
                    break
                key = nav_key

        except KeyboardInterrupt:
            cls()
            print(hr(' Interrupted '))
            sys.exit(0)


if __name__ == '__main__':
    main()
