import json, os, sys, shlex, subprocess, time
from pathlib import Path

CONFIG_DIR = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'unified-llm'
DETECT_CACHE = CONFIG_DIR / 'detect_cache.json'
MODEL_CACHE = CONFIG_DIR / 'models.json'
USE_MODEL_DAT = Path.home() / '.lmstudio' / 'models' / 'use_model.dat'

from paths import SERVER_DIR, SERVER_EXE, OV_LIBS, resolve_server_exe, server_exe_or_raise

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
    'chat':   {'ctx': 8192,  'batch': 512,  'ubatch': 128,  'threads': 16, 'desc': 'General chat'},
    'code':   {'ctx': 16384, 'batch': 128,  'ubatch': 64,   'threads': 16, 'desc': 'Code generation'},
    'vision': {'ctx': 8192,  'batch': 512,  'ubatch': 1024, 'threads': 16, 'desc': 'Vision + multimodal'},
    'embed':  {'ctx': 512,   'batch': 512,  'ubatch': 512,  'threads': 4,  'desc': 'Embeddings only'},
    'fast':   {'ctx': 2048,  'batch': 512,  'ubatch': 256,  'threads': 16, 'desc': 'Quick responses'},
    'research': {'ctx': 32768, 'batch': 512, 'ubatch': 1024, 'threads': 16, 'desc': 'Deep codebase analysis'},
}

DEFAULT_TASK = 'chat'

def _auto_scan():
    base = Path(__file__).parent
    hw_script = base / 'hardware_detect.py'
    sc_script = base / 'model_scanner.py'
    if hw_script.exists() and not DETECT_CACHE.exists():
        subprocess.run([sys.executable, str(hw_script)], capture_output=True, timeout=60)
    if sc_script.exists() and not MODEL_CACHE.exists():
        subprocess.run([sys.executable, str(sc_script)], capture_output=True, timeout=120)

def _prune_orphans(models: list) -> list:
    pruned = [m for m in models if os.path.exists(m.get('path', ''))]
    if len(pruned) != len(models):
        data = json.loads(MODEL_CACHE.read_text())
        data['models'] = pruned
        data['pruned'] = len(models) - len(pruned)
        MODEL_CACHE.write_text(json.dumps(data, indent=2, default=str))
    return pruned

def load_hw():
    if not DETECT_CACHE.exists():
        _auto_scan()
    if DETECT_CACHE.exists():
        return json.loads(DETECT_CACHE.read_text())
    return None

def load_models():
    if not MODEL_CACHE.exists():
        _auto_scan()
    if MODEL_CACHE.exists():
        models = json.loads(MODEL_CACHE.read_text()).get('models', [])
        return _prune_orphans(models)
    return []

def read_use_model():
    if not USE_MODEL_DAT.exists():
        return None
    for line in USE_MODEL_DAT.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('|')
        if parts:
            p = parts[0].strip()
            if os.path.exists(p):
                return p
            if Path.home().name in p:
                p = p.replace('/mnt/c/Users', 'C:/Users').replace('/', '\\')
                if os.path.exists(p):
                    return p
    return None

def find_best_model(models):
    def score(m):
        arch = m.get('metadata', {}).get('architecture', '')
        base = MODEL_PREFERENCE.index(arch) if arch in MODEL_PREFERENCE else 99
        size = m.get('size_bytes', 0)
        return (base, -size)

    scored = [m for m in models if not m['filename'].startswith('mmproj-') and m.get('metadata', {}).get('architecture', '') != 'clip']
    if not scored:
        scored = [m for m in models if not m['filename'].startswith('mmproj-')]
    if not scored:
        return models[0] if models else None
    scored.sort(key=score)
    return scored[0]

def match_model(models, query):
    if not isinstance(query, str):
        return None
    query = query.lower()
    hits = []
    for m in models:
        if m.get('metadata', {}).get('architecture', '') == 'clip':
            continue
        matched = (query in m['path'].lower() or query in m['filename'].lower())
        if not matched:
            meta = m.get('metadata', {})
            for val in [meta.get(k, '') for k in ('name', 'basename', 'architecture')]:
                if query in str(val).lower():
                    matched = True
                    break
        if matched:
            hits.append(m)
    if not hits:
        return None
    # Prefer real weights over MTP/draft files unless explicitly queried.
    # e.g. "qwythos" -> 5.6GB qwen35 main, not 0.7GB unknown MTP.
    want_mtp = 'mtp' in query
    if not want_mtp:
        non_mtp = [m for m in hits if 'mtp' not in m['filename'].lower()]
        if non_mtp:
            hits = non_mtp
    # Largest first so Q4_K_M beats smaller drafts of same family.
    hits.sort(key=lambda m: m.get('size_bytes', 0), reverse=True)
    return hits[0]

def compute_launch(model, hw, mode='server', requested_port=0, task_mode=None, instances=0):
    meta = model.get('metadata', {})
    arch = meta.get('architecture', 'unknown')
    block_count = meta.get('block_count', 32)
    model_size_gb = model.get('size_gb', 5)
    model_path = model['path']
    is_vision = 'mmproj' in meta

    sys_free_gb = hw.get('system', {}).get('available_ram_gb', 8)

    env = {}
    env['GGML_OPENVINO_DISABLE'] = '1'
    ov_path_str = str(OV_LIBS)
    env['PATH'] = f'{ov_path_str};{os.environ.get("PATH", "")}'

    if task_mode is None:
        task_mode = DEFAULT_TASK
        if is_vision and task_mode == 'chat':
            task_mode = 'vision'
    task_cfg = TASK_PRESETS.get(task_mode, TASK_PRESETS[DEFAULT_TASK])

    ctx_size = task_cfg['ctx']
    batch_size = task_cfg['batch']
    ubatch_size = task_cfg['ubatch']
    threads = task_cfg['threads']

    if instances > 1:
        threads = max(1, threads // instances)

    # ── Benchmark-derived backend rules ──────────────────────────────────
    # System: 16 cores, Intel Arc 140T (2.1 GB VRAM), ~40 GB/s DDR5
    #
    # Benchmark (2026-05-22, qwen2.5-1.5b):
    #   CPU:    47.4 tok/s gen, 169.9 tok/s prompt
    #   Vulkan: 37.6 tok/s gen,  73.1 tok/s prompt
    # → CPU beats Vulkan on ALL metrics for this hardware.
    # → Never use Vulkan for pure text inference.
    # → Partial offload is always slower than pure CPU (PCIe overhead).
    # → Vulkan ViT encoding for vision models is unmeasured but
    #   likely slower on this integrated GPU. CPU all the way.

    backend = 'cpu'
    ngl = 0

    # Size-based throughput estimate (CPU, 16 threads)
    if model_size_gb <= 2.0:
        tps_est = 45.0
    elif model_size_gb <= 3.0:
        tps_est = 22.0
    elif model_size_gb <= 4.5:
        tps_est = 15.0
    elif model_size_gb <= 8.0:
        tps_est = 10.0
    else:
        tps_est = 4.0

    args = ['--model' if mode == 'server' else '-m', model_path]
    if mode == 'server':
        port = str(requested_port) if isinstance(requested_port, int) and requested_port > 0 else '8080'
        args = [
            '--model', model_path,
            '--host', '127.0.0.1',
            '--port', port,
            '--ctx-size', str(ctx_size),
            '--threads', str(threads),
            '--batch-size', str(batch_size),
            '--ubatch-size', str(ubatch_size),
            '--no-warmup', '-ngl', '0',
        ]
        if task_mode != 'embed':
            args.extend(['--jinja'])
        if task_mode == 'embed':
            args.extend(['--embedding', '--pooling', 'mean'])
    else:
        args = [
            '-m', model_path,
            '-n', '256',
            '--ctx-size', str(ctx_size),
            '--temp', '0.7', '-ngl', '0',
        ]

    arch_specific = ARCH_FLAGS.get(arch, [])
    args.extend(arch_specific)

    if is_vision:
        mmproj_name = meta.get('mmproj')
        mmproj_path = str(Path(model_path).parent / mmproj_name) if mmproj_name else None
        if mmproj_path and os.path.exists(mmproj_path):
            args.extend(['--mmproj', mmproj_path])

    # P0-7 (2026-09-15): auto-attach LoRA adapter if convert-lora-to-gguf output
    # sits next to the base model (adapter.gguf). Zero-config serve path for
    # train_gemma.py / train_dpo_scale.py outputs.
    for _cand in (str(Path(model_path).parent / "adapter.gguf"),
                  str(Path(model_path).parent / "adapter" / "adapter.gguf")):
        if os.path.exists(_cand):
            args.extend(['--lora', _cand])
            break

    memory_ok = sys_free_gb >= (model_size_gb * 1.2)

    result = {
        'model': model['filename'],
        'model_path': model_path,
        'size_gb': model_size_gb,
        'architecture': arch,
        'block_count': block_count,
        'is_vision': is_vision,
        'mode': mode,
        'task': task_mode,
        'engine': f'llama-server.exe ({SERVER_EXE})' if mode == 'server' else 'llama-cli',
        'backend': backend,
        'n_gpu_layers': ngl,
        'context_size': ctx_size,
        'threads': threads,
        'batch_size': batch_size,
        'ubatch_size': ubatch_size,
        'args': args,
        'args_str': ' '.join(f'"{a}"' if ' ' in str(a) else str(a) for a in args),
        'env': env,
        'tps_est': round(tps_est, 1),
        'memory_adequate': memory_ok,
        'notes': [],
    }

    result['notes'].append(f'Task: {task_mode} ({task_cfg["desc"]})')
    if arch == 'gemma4':
        result['notes'].append('Gemma-4: --fit off')
    if not memory_ok:
        result['notes'].append(f'Low memory: {sys_free_gb:.1f} GB free, model needs ~{model_size_gb * 1.2:.1f} GB')
    if 'mmproj' in meta:
        result['notes'].append(f'Vision model: {meta["mmproj"]}')
    result['notes'].append('CPU-only (benchmarked 47.4 tok/s gen, 169.9 tok/s prompt for 1.5B)')
    if instances > 1:
        result['notes'].append(f'Threads split for {instances} instances: {threads} each')
    result['notes'].append('OpenVINO backend disabled')

    return result

def route(model_query='auto', mode='server', requested_port=0, task_mode=None, instances=0):
    hw = load_hw()
    models = load_models()
    if not models:
        return {'error': 'No models cached. Run model_scanner.py first.'}
    if not hw:
        return {'error': 'No hardware cache. Run hardware_detect.py first.'}

    if model_query == 'auto':
        use_model_path = read_use_model()
        if use_model_path:
            m = None
            for mm in models:
                if mm['path'] == use_model_path:
                    m = mm
                    break
            if m:
                return compute_launch(m, hw, mode, requested_port, task_mode, instances)
        m = find_best_model(models)
    else:
        m = match_model(models, model_query)
    if not m:
        return {'error': f'No model matching "{model_query}". Run model_scanner.py to refresh.'}
    if not os.path.exists(m.get('path', '')):
        return {'error': f'Model file not found on disk: {m["path"]}. Run model_scanner.py to refresh.'}
    return compute_launch(m, hw, mode, requested_port, task_mode, instances)

def main():
    model_query = 'auto'
    mode = 'server'
    port = 0
    task_mode = None
    instances = 0
    args_iter = iter(sys.argv[1:])
    for arg in args_iter:
        if arg.startswith('--mode='):
            mode = arg.split('=', 1)[1]
        elif arg.startswith('--model='):
            model_query = arg.split('=', 1)[1]
        elif arg.startswith('--port='):
            port = int(arg.split('=', 1)[1])
        elif arg == '--port':
            port = int(next(args_iter, 0))
        elif arg.startswith('--task='):
            task_mode = arg.split('=', 1)[1]
        elif arg == '--task':
            task_mode = next(args_iter, None)
        elif arg.startswith('--instances='):
            instances = int(arg.split('=', 1)[1])
        elif arg == '--instances':
            instances = int(next(args_iter, 0))
        elif arg in ('-h', '--help'):
            tasks = '|'.join(TASK_PRESETS.keys())
            print(f'Usage: python router.py [--model=<name|path>] [--mode=server|cli] [--port=N] [--task={tasks}] [--instances=N] [--json] [--args-only]')
            return
    result = route(model_query, mode, port, task_mode, instances)
    if '--args-only' in sys.argv:
        if 'error' in result:
            print(f'ERROR: {result["error"]}', file=sys.stderr)
            sys.exit(1)
        print(result['args_str'])
        return
    if '--json' in sys.argv:
        print(json.dumps(result, indent=2, default=str))
    else:
        if 'error' in result:
            print(f'ERROR: {result["error"]}')
            return
        print(f'Model:  {result["model"]} ({result["size_gb"]} GB, {result["architecture"]})')
        print(f'Engine: {result["engine"]}')
        print(f'Backend: {result["backend"]} ({result["n_gpu_layers"]} GPU layers, ~{result["tps_est"]} tok/s)')
        print(f'Context: {result["context_size"]}, Threads: {result["threads"]}, Batch: {result["batch_size"]}')
        print(f'Memory:  {"OK" if result["memory_adequate"] else "LOW"}')
        print(f'Vision:  {"Yes (with mmproj)" if result["is_vision"] else "No"}')
        print(f'')
        print(f'Launch args ({mode}):')
        print(f'  {result["args_str"]}')
        if result.get('notes'):
            print(f'')
            print('Notes:')
            for n in result['notes']:
                print(f'  - {n}')

if __name__ == '__main__':
    main()
