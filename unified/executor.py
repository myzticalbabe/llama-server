import json, os, sys, time, threading
import subprocess
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ── Free-threading imports ────────────────────────────────────────────────
_ft_path = str(Path(__file__).resolve().parent.parent.parent / 'azazael' / 'src')
if _ft_path not in sys.path:
    sys.path.insert(0, _ft_path)
from CORE.freethreading import ThreadSafeDict

CONFIG_DIR = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'unified-llm'
PID_FILE = CONFIG_DIR / 'server.pid'
STATUS_FILE = CONFIG_DIR / 'server_status.json'
INSTANCES_FILE = CONFIG_DIR / 'instances.json'
_instances_cache = ThreadSafeDict()
_save_lock = threading.Lock()

from paths import SERVER_DIR, SERVER_EXE, OV_LIBS, PAMS_DATA_DIR, resolve_server_exe


@dataclass
class ServerInstance:
    pid: int
    port: int
    model: str
    backend: str
    label: str
    started: float = 0.0
    healthy: bool = False


class EngineExecutor(ABC):
    @abstractmethod
    def start(self, args: list[str], env: dict | None = None) -> int:
        ...

    @abstractmethod
    def stop(self) -> bool:
        ...

    @abstractmethod
    def health(self, port: int) -> bool:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class LlamaServerExecutor(EngineExecutor):
    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._exe = str(SERVER_EXE)
        self._instance: ServerInstance | None = None

    def name(self) -> str:
        return 'llama-server'

    def start(self, args: list[str], env: dict | None = None) -> int:
        exe = resolve_server_exe()
        if exe is None:
            raise FileNotFoundError(
                "llama-server binary not found. Set LLAMA_SERVER_EXE or LLAMA_SERVER_DIR."
            )
        self._exe = str(exe)
        merged = dict(os.environ)
        merged.setdefault('GGML_OPENVINO_DISABLE', '1')
        merged.setdefault('GGML_OPENVINO_BLACKLIST', '1')
        ov_path = str(OV_LIBS)
        if ov_path not in merged.get('PATH', ''):
            merged['PATH'] = f'{ov_path};{merged["PATH"]}'
        if env:
            merged.update(env)
        merged['GGML_OPENVINO_DISABLE'] = '1'
        merged['GGML_OPENVINO_BLACKLIST'] = '1'

        self._process = subprocess.Popen(
            [self._exe] + args,
            env=merged,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )

        port = 8080
        model_name = 'unknown'
        backend_name = 'cpu'
        for i, a in enumerate(args):
            if a == '--port' and i + 1 < len(args): port = int(args[i + 1])
            if a == '--model' and i + 1 < len(args): model_name = Path(args[i + 1]).name
            if a == '-ngl' and i + 1 < len(args): backend_name = 'vulkan' if int(args[i + 1]) > 0 else 'cpu'

        self._instance = ServerInstance(
            pid=self._process.pid, port=port, model=model_name,
            backend=backend_name, label='llama-server', started=time.time(),
        )
        _register_instance(self._instance)
        return self._process.pid

    def stop(self) -> bool:
        if self._instance:
            _unregister_instance(self._instance.port)
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            return True
        return False

    def health(self, port: int) -> bool:
        import urllib.request
        try:
            r = urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3)
            ok = r.status == 200
            if ok and self._instance:
                self._instance.healthy = True
                _register_instance(self._instance)
            return ok
        except Exception:
            return False

    @staticmethod
    def available() -> bool:
        return resolve_server_exe() is not None


class OpenVINOExecutor(EngineExecutor):
    def name(self) -> str:
        return 'openvino-python'

    def start(self, args: list[str], env: dict | None = None) -> int:
        return -1

    def stop(self) -> bool:
        return True

    def health(self, port: int) -> bool:
        return False

    @staticmethod
    def available() -> bool:
        try:
            import openvino
            return True
        except ImportError:
            return False

    @staticmethod
    def is_model_supported(model_entry: dict) -> tuple[bool, str]:
        arch = model_entry.get('metadata', {}).get('architecture', '')
        if arch == 'gemma4':
            return False, 'gemma-4 has ov::Exception incompatibility'
        return False, 'OpenVINO Python runtime not configured for GGUF models'


def _register_instance(inst: ServerInstance):
    _instances_cache[str(inst.port)] = inst.__dict__
    with _save_lock:
        _save_instances({int(k): ServerInstance(**v) if isinstance(v, dict) else v for k, v in _instances_cache.items()})


def _unregister_instance(port: int):
    _instances_cache.pop(str(port), None)
    with _save_lock:
        _save_instances({int(k): ServerInstance(**v) if isinstance(v, dict) else v for k, v in _instances_cache.items()})


def _load_instances() -> dict:
    if INSTANCES_FILE.exists():
        try:
            data = json.loads(INSTANCES_FILE.read_text())
            return {int(k): ServerInstance(**v) for k, v in data.items()}
        except Exception:
            return {}
    return {}

def _load_instances_from_disk():
    """Populate the in-memory cache from disk at startup."""
    for k, v in _load_instances().items():
        _instances_cache[str(k)] = v.__dict__


def _save_instances(instances: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {str(k): v.__dict__ for k, v in instances.items()}
    INSTANCES_FILE.write_text(json.dumps(data, indent=2, default=str))


def resolve_executor(backend: str) -> EngineExecutor:
    if backend == 'openvino':
        return OpenVINOExecutor()
    return LlamaServerExecutor()


def wait_for_health(port: int, executor: EngineExecutor, timeout: int = 120, interval: int = 2) -> bool:
    for _ in range(timeout // interval):
        if executor.health(port):
            return True
        time.sleep(interval)
    return False


def _process_exists(pid: int) -> bool:
    """Check if process exists on Windows using tasklist."""
    r = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                       capture_output=True, text=True, timeout=5)
    return str(pid) in r.stdout


def _kill_process(pid: int) -> bool:
    """Kill process by PID on Windows."""
    r = subprocess.run(['taskkill', '/PID', str(pid), '/F'],
                       capture_output=True, text=True, timeout=5)
    return r.returncode == 0


def get_status() -> dict:
    instances = _load_instances()
    if not instances:
        return {'running': False, 'instances': []}
    alive = []
    for port, inst in list(instances.items()):
        if _process_exists(inst.pid):
            alive.append({'pid': inst.pid, 'port': inst.port, 'model': inst.model,
                          'backend': inst.backend, 'healthy': inst.healthy})
        else:
            _unregister_instance(port)
    if not alive:
        return {'running': False, 'instances': []}
    return {'running': True, 'instance_count': len(alive), 'instances': alive}


def stop_server() -> bool:
    instances = _load_instances()
    if not instances:
        return False
    for port, inst in list(instances.items()):
        if _process_exists(inst.pid):
            _kill_process(inst.pid)
        _unregister_instance(port)
    return True


def stop_all() -> bool:
    stopped = stop_server()
    pams = get_pams_manager()
    if pams.stop():
        stopped = True
    return stopped


def stop_instance(port: int) -> bool:
    instances = _load_instances()
    inst = instances.get(port)
    if not inst:
        return False
    if _process_exists(inst.pid):
        _kill_process(inst.pid)
    _unregister_instance(port)
    return True


# ── PAMS (persistent-ai-memory) integration ────────────────────────────

PAMS_PORT = 21435
PAMS_LABEL = 'pams-server'
PAMS_LAUNCHER = Path(__file__).parent / '_pams_launcher.py'

class PAMSManager:
    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._pams_pid: int = 0
        self._instance: ServerInstance | None = None
        self._http_port: int = 0
        self._detect_running()

    def _detect_running(self):
        port_file = PAMS_DATA_DIR / 'mcp_server_port.json'
        if port_file.exists():
            try:
                data = json.loads(port_file.read_text())
                port = int(data.get('active_port', 0))
                if port > 0:
                    import urllib.request
                    try:
                        r = urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=2)
                        if r.status == 200:
                            self._http_port = port
                            self._instance = ServerInstance(
                                pid=0, port=PAMS_PORT, model='persistent-ai-memory',
                                backend='pams', label=PAMS_LABEL, healthy=True,
                            )
                    except Exception:
                        pass
            except Exception:
                pass

    @property
    def http_port(self) -> int:
        return self._http_port

    def start(self) -> int:
        if self._process and self._process.poll() is None:
            return self._pams_pid

        if self._http_port > 0:
            return self._pams_pid if self._pams_pid > 0 else -1

        env = dict(os.environ)
        env['AI_MEMORY_DATA_DIR'] = str(PAMS_DATA_DIR)

        self._process = subprocess.Popen(
            [sys.executable, str(PAMS_LAUNCHER)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )

        # Read pams-server PID from launcher's stdout
        pid_line = self._process.stdout.readline().decode().strip()
        self._pams_pid = int(pid_line) if pid_line.isdigit() else 0

        self._http_port = self._read_port_file(timeout=15)

        self._instance = ServerInstance(
            pid=self._pams_pid,
            port=PAMS_PORT,
            model='persistent-ai-memory',
            backend='pams',
            label=PAMS_LABEL,
            started=time.time(),
            healthy=self._http_port > 0,
        )
        _register_instance(self._instance)
        return self._pams_pid

    def _read_port_file(self, timeout: int = 15) -> int:
        port_file = PAMS_DATA_DIR / 'mcp_server_port.json'
        deadline = time.time() + timeout
        while time.time() < deadline:
            if port_file.exists():
                try:
                    data = json.loads(port_file.read_text())
                    return int(data.get('active_port', 0))
                except Exception:
                    pass
            time.sleep(0.5)
        return 0

    def stop(self) -> bool:
        if self._instance:
            _unregister_instance(PAMS_PORT)
        if self._pams_pid and _process_exists(self._pams_pid):
            _kill_process(self._pams_pid)
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self._pams_pid = 0
            self._http_port = 0
            return True
        return False

    def health(self) -> bool:
        port = self._http_port or self._read_port_file(timeout=3)
        if not port:
            return False
        try:
            import urllib.request
            r = urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=3)
            return r.status == 200
        except Exception:
            return False


_PAMS_MANAGER: PAMSManager | None = None

def get_pams_manager() -> PAMSManager:
    global _PAMS_MANAGER
    if _PAMS_MANAGER is None:
        _PAMS_MANAGER = PAMSManager()
    return _PAMS_MANAGER

# Populate cache from disk on module load
_load_instances_from_disk()
