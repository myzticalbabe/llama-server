import subprocess, json, os, sys, re, platform as _platform
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

CACHE_FILE = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'unified-llm' / 'detect_cache.json'

def run_ps(script):
    r = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                        capture_output=True, text=True, timeout=15)
    return r.stdout.strip()

@dataclass
class SystemInfo:
    total_ram_gb: float = 0
    available_ram_gb: float = 0
    total_ram_bytes: int = 0
    available_ram_bytes: int = 0
    logical_cores: int = 0
    physical_cores: int = 0
    cpu_name: str = ''
    cpu_features: list = field(default_factory=list)

@dataclass
class GpuInfo:
    name: str = ''
    vram_bytes: int = 0
    vram_gb: float = 0
    driver_version: str = ''
    vulkan_available: bool = False
    opencl_available: bool = False

@dataclass
class NpuInfo:
    name: str = ''
    available: bool = False
    driver_version: str = ''

@dataclass
class BackendInfo:
    name: str = ''
    available: bool = False
    engines: list = field(default_factory=list)

@dataclass
class HardwareConfig:
    system: SystemInfo = field(default_factory=SystemInfo)
    gpu: list = field(default_factory=list)
    npu: Optional[NpuInfo] = None
    backends: list = field(default_factory=list)
    timestamp: str = ''

def probe_system():
    sysinfo = SystemInfo()
    total_str = run_ps('(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory')
    try:
        total_bytes = int(total_str)
        sysinfo.total_ram_bytes = total_bytes
        sysinfo.total_ram_gb = round(total_bytes / 1e9, 1)
    except: pass
    free_kb = run_ps('(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory')
    try:
        avail_bytes = int(free_kb) * 1024
        sysinfo.available_ram_bytes = avail_bytes
        sysinfo.available_ram_gb = round(avail_bytes / 1e9, 1)
    except: pass
    logical = run_ps('(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors')
    try: sysinfo.logical_cores = int(logical)
    except: pass
    cores = run_ps('(Get-CimInstance Win32_Processor).NumberOfCores')
    try: sysinfo.physical_cores = int(cores)
    except: pass
    sysinfo.cpu_name = run_ps('(Get-CimInstance Win32_Processor).Name')
    features = []
    cpu_info = run_ps(r'Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Caption')
    # Rough feature detection based on generation
    if 'Ultra 9 285H' in sysinfo.cpu_name:
        features = ['avx2', 'avx_vnni', 'vnnip']
    elif 'avx2' in cpu_info.lower():
        features.append('avx2')
    elif 'avx' in cpu_info.lower():
        features.append('avx')
    sysinfo.cpu_features = features
    return sysinfo

def probe_gpu():
    gpus = []
    script = r'''
$gpu = Get-CimInstance Win32_VideoController | Select-Object -First 1
$vram = $gpu.AdapterRAM
if (-not $vram -or $vram -eq 0) { $vram = 2147483648 }
[PSCustomObject]@{ Name = $gpu.Name; RAM = $vram; Driver = $gpu.DriverVersion } | ConvertTo-Json -Compress
'''
    raw = run_ps(script)
    info = GpuInfo()
    try:
        data = json.loads(raw)
        info.name = data.get('Name', '')
        info.vram_bytes = int(data.get('RAM', 0))
        info.vram_gb = round(info.vram_bytes / 1e9, 1)
        info.driver_version = data.get('Driver', '')
    except: pass
    vk = subprocess.run(['vulkaninfo'], capture_output=True, text=True, shell=True, timeout=10)
    info.vulkan_available = vk.returncode == 0
    # Check OpenCL via Python
    try:
        import opencl
        info.opencl_available = True
    except:
        info.opencl_available = False
    gpus.append(info)
    return gpus

def probe_npu():
    script = r'Get-CimInstance Win32_PNPEntity | Where-Object { $_.Name -match "AI Boost|NPU|Intel.*Neural|ComputeAccelerator" } | Select-Object -First 1 Name, Status | ConvertTo-Json -Compress'
    raw = run_ps(script)
    if not raw or raw == 'null' or raw == '[]':
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            data = data[0] if data else {}
        name = data.get('Name', 'Intel AI Boost')
        return NpuInfo(name=name, available=True, driver_version='')
    except:
        return None

def probe_backends():
    backends = []
    sys_name = run_ps('(Get-CimInstance Win32_Processor).Name')
    backends.append(BackendInfo(name='cpu', available=True, engines=[f'CPU ({sys_name})']))
    vk = subprocess.run(['vulkaninfo'], capture_output=True, text=True, shell=True, timeout=10)
    if vk.returncode == 0:
        gpu_name = run_ps('(Get-CimInstance Win32_VideoController).Name')
        backends.append(BackendInfo(name='vulkan', available=True, engines=[gpu_name]))
    else:
        backends.append(BackendInfo(name='vulkan', available=False))
    from paths import SERVER_DIR as _SERVER_DIR
    ov_dll = _SERVER_DIR / 'ggml-openvino.dll'
    backends.append(BackendInfo(name='openvino', available=ov_dll.exists()))
    sycl_path = Path(r'C:\Program Files (x86)\Intel\oneAPI\compiler\2026.0\bin\sycl-ls.exe')
    if sycl_path.exists():
        r = subprocess.run([str(sycl_path)], capture_output=True, text=True, timeout=10)
        has_gpu = 'gpu' in r.stdout.lower()
        backends.append(BackendInfo(name='sycl', available=has_gpu, engines=['oneAPI SYCL'] if has_gpu else []))
    else:
        backends.append(BackendInfo(name='sycl', available=False))
    cuda = subprocess.run(['nvidia-smi'], capture_output=True, text=True, shell=True, timeout=5)
    backends.append(BackendInfo(name='cuda', available=cuda.returncode == 0))
    # Python engine backends
    backends.append(BackendInfo(name='pytorch', available=True, engines=[f'torch {_import_torch_version()}']))
    backends.append(BackendInfo(name='tensorflow', available='tensorflow' in sys.modules or os.system('python -c "import tensorflow" 2>nul') == 0))
    try:
        import openvino
        core = openvino.Core()
        ov_devices = core.available_devices
        full_devices = [f'{d}={core.get_property(d, "FULL_DEVICE_NAME")}' for d in ov_devices]
        backends.append(BackendInfo(name='openvino_python', available=True, engines=full_devices))
    except: pass
    return backends

def _import_torch_version():
    try:
        import torch
        return torch.__version__
    except: return ''

def detect():
    hw = HardwareConfig()
    hw.system = probe_system()
    hw.gpu = probe_gpu()
    hw.npu = probe_npu()
    hw.backends = probe_backends()
    hw.timestamp = __import__('datetime').datetime.now().isoformat()
    return hw

def to_dict(hw):
    return json.loads(json.dumps(hw, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)))

def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except: pass
    return None

def save_cache(data):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data, indent=2, default=str))

def main():
    import datetime
    hw = detect()
    data = to_dict(hw)
    save_cache(data)
    if '--json' in sys.argv:
        print(json.dumps(data, indent=2, default=str))
    else:
        s = hw.system
        print(f'CPU: {s.cpu_name} ({s.physical_cores}C/{s.logical_cores}T, {s.cpu_features})')
        print(f'RAM: {s.total_ram_gb} GB total, {s.available_ram_gb} GB free')
        for g in hw.gpu:
            print(f'GPU: {g.name} ({g.vram_gb} GB, Vulkan: {g.vulkan_available})')
        if hw.npu:
            print(f'NPU: {hw.npu.name}')
        print('Backends:')
        for b in hw.backends:
            icon = 'OK' if b.available else '--'
            eng = f' [{", ".join(b.engines)}]' if b.engines else ''
            print(f'  {b.name:<20} {icon}{eng}')
        print(f'\nCache: {CACHE_FILE}')

if __name__ == '__main__':
    main()
