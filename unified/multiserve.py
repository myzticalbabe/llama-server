"""
multiserve.py  —  Parallel multi-instance server manager.

Python 3.14 free-threaded build enables true parallel threading.
On this system (16 cores, Intel Arc 140T, 16 GB RAM), we can run
multiple llama-server instances simultaneously, each with a different
model and backend, managed by truly concurrent Python threads.

Usage:
    python multiserve.py --plan plan.json         # Launch from plan file
    python multiserve.py --auto --count 2         # Auto-select top 2 models
    python multiserve.py --auto --vision-only     # Run all vision models
    python multiserve.py --stop                   # Stop all instances
    python multiserve.py --status                 # Show all instances
    python multiserve.py --stop --port 18081      # Stop single instance
"""

import json, os, sys, time, itertools, subprocess
from pathlib import Path
from concurrent.futures import as_completed
from dataclasses import dataclass, field

# ── Free-threading imports ────────────────────────────────────────────────
_ft_path = str(Path(__file__).resolve().parent.parent.parent / 'azazael' / 'src')
if _ft_path not in sys.path:
    sys.path.insert(0, _ft_path)
from CORE.freethreading import FreeThreadPool

BASE_DIR = Path(__file__).parent
CONFIG_DIR = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'unified-llm'
ROUTER_PY = BASE_DIR / 'router.py'

from paths import PYTHON_DIR, PYTHON_FT

PYTHON_EXE = str(PYTHON_FT if PYTHON_FT.exists() else sys.executable)

PORT_POOL = iter(range(18080, 18096))

THREAD_NAMES = itertools.cycle([
    'serpent', 'phoenix', 'griffin', 'dragon',
    'kraken', 'leviathan', 'chimera', 'hydra',
])

@dataclass
class ServePlan:
    model_query: str = ''
    port: int = 18080
    backend: str = 'auto'
    task_mode: str = 'chat'
    label: str = ''
    _estimated_gb: float = 0.0


def _resolve_port() -> int:
    return next(PORT_POOL)


def _run_router(model_query: str, port: int, task_mode: str = 'chat') -> dict | None:
    import subprocess, shlex
    args = [PYTHON_EXE, str(ROUTER_PY), '--json',
            f'--model={model_query}', f'--port={port}', f'--task={task_mode}']
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        plan = json.loads(r.stdout)
        if 'error' in plan:
            return None
        return plan
    except Exception:
        return None


def _launch_one(plan: ServePlan, idx: int, total: int) -> dict:
    thread_name = next(THREAD_NAMES)
    print(f'  [{idx}/{total}] {thread_name}: resolving {plan.model_query} on port {plan.port}...')

    route = _run_router(plan.model_query, plan.port, plan.task_mode)
    if not route:
        return {'port': plan.port, 'model': plan.model_query, 'status': 'failed', 'error': 'router returned no plan'}

    from executor import LlamaServerExecutor, resolve_executor, wait_for_health
    executor = resolve_executor(route['backend'])
    args = route['args']

    pid = executor.start(args, route.get('env'))
    if pid is None or pid <= 0:
        return {'port': plan.port, 'model': plan.model_query, 'status': 'failed', 'error': 'executor start failed'}

    healthy = wait_for_health(plan.port, executor, timeout=120)
    label = plan.label or route.get('model', plan.model_query)
    return {
        'port': plan.port,
        'model': route.get('model', plan.model_query),
        'model_path': route.get('model_path', ''),
        'backend': route['backend'],
        'n_gpu_layers': route['n_gpu_layers'],
        'pid': pid,
        'status': 'healthy' if healthy else 'started_unhealthy',
        'task_mode': plan.task_mode,
        'label': label,
        'thread': thread_name,
    }


def _check_memory(plans: list[ServePlan]) -> bool:
    """Warn if total model memory exceeds available RAM. No psutil needed."""
    import subprocess
    r = subprocess.run(['powershell', '-NoProfile', '-Command',
        '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],
        capture_output=True, text=True, timeout=15)
    try:
        avail_kb = int(r.stdout.strip())
        avail_gb = avail_kb * 1024 / 1e9
    except ValueError:
        avail_gb = 8.0
    total_models = sum(p._estimated_gb for p in plans)
    if total_models > avail_gb * 0.85:
        print(f'  [WARN] Total model size ({total_models:.1f} GB) may exceed available RAM ({avail_gb:.1f} GB)')
        print(f'  [WARN] Consider --count fewer or --tasks embed/fast')
        return False
    return True


def launch_all(plans: list[ServePlan]) -> list[dict]:
    total = len(plans)
    if total == 0:
        print('No plans to launch.')
        return []
    _check_memory(plans)
    print(f'\nLaunching {total} server instances in parallel...')
    print(f'  Python:   {PYTHON_EXE}')
    print(f'  Threads:  free-threaded (no GIL)')
    print(f'  Cores:    {os.cpu_count()}')
    print(f'  Workers:  {FreeThreadPool.info()["recommended_workers"]} (auto-sized)')
    print()

    results = []
    with FreeThreadPool() as pool:
        futures = {pool.submit(_launch_one, p, i + 1, total): p for i, p in enumerate(plans)}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r['port'])
    print(f'\n=== Multi-Serve Status ===')
    for r in results:
        icon = '[OK]' if r['status'] == 'healthy' else '[FAIL]'
        print(f'  {icon}  {r["label"]:<40} port={r["port"]}  {r["backend"]:<8} pid={r["pid"]}')
    print()

    report = CONFIG_DIR / 'multiserve_report.json'
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2, default=str))
    return results


def auto_plan(count: int = 2, vision_only: bool = False, task_modes: list[str] | None = None) -> list[ServePlan]:
    cache = CONFIG_DIR / 'models.json'
    if not cache.exists():
        print('No model cache. Run --scan first.')
        return []
    data = json.loads(cache.read_text())
    models = data.get('models', [])
    models = [m for m in models if not m['filename'].startswith('mmproj-')
              and m.get('metadata', {}).get('architecture') not in ('clip',)]

    if vision_only:
        models = [m for m in models if m.get('metadata', {}).get('mmproj')]

    models.sort(key=lambda m: m.get('size_gb', 0), reverse=True)
    models = models[:count]

    modes = task_modes or ['chat'] * len(models)
    plans = []
    for i, m in enumerate(models):
        mode = modes[i] if i < len(modes) else 'chat'
        plans.append(ServePlan(
            model_query=m['path'],
            port=_resolve_port(),
            task_mode=mode,
            label=f'{m.get("metadata", {}).get("name") or m["filename"]}',
            _estimated_gb=m.get('size_gb', 5) * 1.2,
        ))
    return plans


def load_plan(path: str) -> list[ServePlan]:
    with open(path) as f:
        raw = json.load(f)
    plans = []
    for entry in raw if isinstance(raw, list) else raw.get('instances', []):
        plans.append(ServePlan(
            model_query=entry.get('model', 'auto'),
            port=entry.get('port', _resolve_port()),
            backend=entry.get('backend', 'auto'),
            task_mode=entry.get('task_mode', 'chat'),
            label=entry.get('label', ''),
        ))
    return plans


def main():
    import argparse
    parser = argparse.ArgumentParser(prog='multiserve',
        description='Parallel multi-instance server manager (Python 3.14 free-threaded)')
    parser.add_argument('--plan', type=str, help='JSON plan file with instance definitions')
    parser.add_argument('--auto', action='store_true', help='Auto-select top models')
    parser.add_argument('--count', type=int, default=2, help='Number of instances for --auto')
    parser.add_argument('--vision-only', action='store_true', help='Only vision models')
    parser.add_argument('--tasks', type=str, nargs='*',
                        choices=['chat', 'code', 'vision', 'embed', 'fast'],
                        help='Task mode per instance')
    parser.add_argument('--stop', nargs='?', const='all', metavar='PORT',
                        help='Stop all instances (or single port)')
    parser.add_argument('--status', action='store_true', help='Show all running instances')

    args = parser.parse_args()

    if args.status:
        from executor import get_status
        s = get_status()
        if not s.get('running'):
            print('No running instances.')
            return
        print(f'{s["instance_count"]} running instance(s):')
        for inst in s['instances']:
            icon = '[OK]' if inst['healthy'] else '[..]'
            print(f'  {icon}  port={inst["port"]}  {inst["model"]:<45} {inst["backend"]:<8} pid={inst["pid"]}')
        return

    if args.stop:
        from executor import stop_instance, stop_server
        if args.stop == 'all':
            stop_server()
            print('All instances stopped.')
        else:
            port = int(args.stop)
            if stop_instance(port):
                print(f'Instance on port {port} stopped.')
            else:
                print(f'No instance found on port {port}.')
        return

    if args.plan:
        plans = load_plan(args.plan)
    elif args.auto:
        task_modes = args.tasks
        plans = auto_plan(count=args.count, vision_only=args.vision_only, task_modes=task_modes)
    else:
        parser.print_help()
        return

    if not plans:
        print('No plans generated.')
        return

    print(f'Multi-serve plan: {len(plans)} instance(s)')
    for p in plans:
        print(f'  port={p.port}  model={p.model_query:<55} mode={p.task_mode}')
    print()

    results = launch_all(plans)
    healthy = sum(1 for r in results if r['status'] == 'healthy')
    print(f'{healthy}/{len(results)} instances healthy')
    if healthy < len(results):
        sys.exit(1)


if __name__ == '__main__':
    main()
