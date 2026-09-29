import sys, os, json, time, shlex, subprocess, webbrowser
from pathlib import Path

CONFIG_DIR = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'unified-llm'
BASE_DIR = Path(__file__).parent
ROUTER_PY = BASE_DIR / 'router.py'
HARDWARE_PY = BASE_DIR / 'hardware_detect.py'
SCANNER_PY = BASE_DIR / 'model_scanner.py'
SWITCH_PY = (
    Path(os.environ['SWITCH_TUI']).expanduser()
    if os.environ.get('SWITCH_TUI')
    else Path.home() / 'Desktop' / 'Llama Server' / 'switch_model_tui.py'
)

from paths import PYTHON_DIR, PYTHON_FT, USE_MODEL_DAT as _USE_MODEL_DAT

USE_MODEL_DAT = _USE_MODEL_DAT

def _resolve_python():
    """Use free-threaded python3.14t.exe if available, else default python.exe."""
    return str(PYTHON_FT if PYTHON_FT.exists() else sys.executable)

def _run_py(script_path: Path, *args: str, python_exe: str | None = None) -> tuple[str, str, int]:
    py = python_exe or _resolve_python()
    r = subprocess.run(
        [py, str(script_path)] + list(args),
        capture_output=True, text=True, timeout=120,
    )
    return r.stdout.strip(), r.stderr.strip(), r.returncode

def cmd_detect():
    out, err, rc = _run_py(HARDWARE_PY)
    print(out)
    if err:
        print(err, file=sys.stderr)
    return rc

def cmd_scan():
    out, err, rc = _run_py(SCANNER_PY)
    print(out)
    if err:
        print(err, file=sys.stderr)
    return rc

def cmd_list():
    cache = CONFIG_DIR / 'models.json'
    if not cache.exists():
        print('No model cache. Run with --scan first.')
        return 1
    data = json.loads(cache.read_text())
    models = data.get('models', [])
    if not models:
        print('No models found in cache.')
        return 0
    print(f'{len(models)} models cached:\n')
    for m in models:
        meta = m.get('metadata', {})
        name = meta.get('name') or m.get('filename', '?')
        arch = meta.get('architecture', '?')
        size = m.get('size_gb', '?')
        ctx = meta.get('context_length', '?')
        mmproj = ' [V]' if meta.get('mmproj') else ''
        exists = ' ' if os.path.exists(m.get('path', '')) else ' [MISSING]'
        print(f'  {name:<45} {arch:<12} {size:>5.1f} GB  ctx={ctx:<6}{mmproj}{exists}')
    return 0

def cmd_switch():
    if SWITCH_PY.exists():
        subprocess.run([sys.executable, str(SWITCH_PY)])
        return 0
    # Fallback: external TUI not installed — use the cached model list picker.
    print(f'switch_model_tui.py not found at {SWITCH_PY}', file=sys.stderr)
    print('Falling back to cached model list (--list). Set SWITCH_TUI to override.', file=sys.stderr)
    return cmd_list()

def cmd_start(model_query='auto', port=8080):
    router_args = ['--json', f'--model={model_query}', f'--port={port}']
    out, err, rc = _run_py(ROUTER_PY, *router_args)
    if rc != 0:
        print(f'Router error: {err}', file=sys.stderr)
        return 1
    try:
        plan = json.loads(out)
    except json.JSONDecodeError:
        print(f'Router returned invalid JSON: {out[:200]}', file=sys.stderr)
        return 1

    if 'error' in plan:
        print(f'ERROR: {plan["error"]}', file=sys.stderr)
        return 1

    from executor import resolve_executor, wait_for_health
    executor = resolve_executor(plan['backend'])
    args = plan['args']
    args_str = ' '.join(shlex.quote(str(a)) for a in args)

    print(f'Starting: {plan["model"]}')
    print(f'  Backend: {plan["backend"]} ({plan["n_gpu_layers"]} GPU layers)')
    print(f'  Port:    {port}')
    print(f'  Context: {plan["context_size"]} ctx, {plan["batch_size"]} batch')
    print(f'  Speed:   ~{plan.get("tps_est", "?")} tok/s')
    print(f'  Args:    {args_str}')
    for note in plan.get('notes', []):
        print(f'  Note:    {note}')

    pid = executor.start(args, plan.get('env'))
    if pid is None or pid <= 0:
        print(f'  FAILED to start {executor.name()}', file=sys.stderr)
        return 1
    print(f'  PID:     {pid}')

    print(f'\nWaiting for health on port {port}...')
    if wait_for_health(port, executor):
        print(f'  READY on http://127.0.0.1:{port}')
        print(f'  Health: http://127.0.0.1:{port}/health')
        print(f'  Chat:   http://127.0.0.1:{port}/chat')
        return 0
    else:
        print('  TIMEOUT waiting for server', file=sys.stderr)
        return 1

def cmd_switch_and_start(model_query='auto'):
    if model_query == 'auto':
        cmd_switch()
    return cmd_start(model_query)


def stop_all_and_exit():
    from executor import stop_all
    if stop_all():
        print('All servers stopped.')
        return 0
    print('No running servers found.')
    return 0

def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog='unified-launcher',
        description='Unified inference launcher — auto-detect, auto-configure, auto-start',
    )
    parser.add_argument('--start', nargs='?', const='auto', metavar='MODEL',
                        help='Start server with best (or named) model')
    parser.add_argument('--port', type=int, default=8080,
                        help='Server port (default: 8080)')
    parser.add_argument('--stop', action='store_true',
                        help='Stop running server')
    parser.add_argument('--status', action='store_true',
                        help='Check if server is running')
    parser.add_argument('--switch', action='store_true',
                        help='Interactive model picker')
    parser.add_argument('--scan', action='store_true',
                        help='Re-scan model directories')
    parser.add_argument('--detect', action='store_true',
                        help='Re-detect hardware')
    parser.add_argument('--list', action='store_true',
                        help='List cached models')
    parser.add_argument('--open', action='store_true',
                        help='Open server chat in browser (after --start)')
    parser.add_argument('--args-only', nargs='?', const='auto', metavar='MODEL',
                        help='Print launch args for use by launch_chat.bat')
    parser.add_argument('--multi', action='store_true',
                        help='Launch multiple server instances concurrently')
    parser.add_argument('--count', type=int, default=2,
                        help='Number of instances for --multi (default: 2)')
    parser.add_argument('--vision-only', action='store_true',
                        help='With --multi, only launch vision models')
    parser.add_argument('--tasks', type=str, nargs='*',
                        choices=['chat', 'code', 'vision', 'embed', 'fast'],
                        help='Task mode per instance for --multi')
    parser.add_argument('--pams', action='store_true',
                        help='Start/stop persistent-ai-memory alongside server')

    args = parser.parse_args()

    if args.detect:
        return cmd_detect()
    if args.scan:
        return cmd_scan()
    if args.list:
        return cmd_list()
    if args.switch:
        return cmd_switch()
    if args.stop:
        from executor import stop_all
        if stop_all():
            print('All servers stopped.')
            return 0
        print('No running servers found.')
        return 0
    if args.status:
        from executor import get_status, get_pams_manager
        s = get_status()
        pams = get_pams_manager()
        if s.get('instances'):
            for inst in s['instances']:
                icon = '[OK]' if inst['healthy'] else '[..]'
                print(f'  {icon}  port={inst["port"]}  {inst["model"]:<45} {inst["backend"]:<8} pid={inst["pid"]}')
        pams_http = pams.http_port
        if pams_http:
            print(f'  [OK]  PAMS:{pams_http}  persistent-ai-memory')
        if not s.get('instances') and not pams_http:
            print('No running servers.')
        return 0
    if args.args_only is not None:
        router_args = ['--args-only', f'--model={args.args_only}', f'--port={args.port}']
        out, err, rc = _run_py(ROUTER_PY, *router_args)
        if rc != 0:
            print(f'ERROR: {err}', file=sys.stderr) if err else print(f'ERROR: router exited {rc}', file=sys.stderr)
            sys.exit(1)
        print(out)
        return 0
    if args.multi:
        from multiserve import auto_plan, launch_all
        plans = auto_plan(count=args.count, vision_only=args.vision_only, task_modes=args.tasks)
        if not plans:
            print('No plans generated.')
            return 1
        print('Multi-serve plan:')
        for p in plans:
            print(f'  port={p.port}  model={p.model_query:<55} mode={p.task_mode}')
        results = launch_all(plans)
        healthy = sum(1 for r in results if r['status'] == 'healthy')
        if healthy == 0:
            return 1
        return 0
    if args.pams and args.start is None and not args.stop and not args.status:
        from executor import get_pams_manager
        pams = get_pams_manager()
        print('Starting persistent-ai-memory...')
        pid = pams.start()
        if pid > 0:
            print(f'  PAMS PID: {pid}')
            healthy = pams.health()
            if pams.http_port:
                print(f'  HTTP API: http://127.0.0.1:{pams.http_port}/api/health')
            print(f'  Status:   {"healthy" if healthy else "starting"}')
            return 0
        print('  FAILED to start pams-server', file=sys.stderr)
        return 1

    if args.start is not None:
        rc = cmd_start(args.start, args.port)
        if rc == 0 and args.pams:
            from executor import get_pams_manager
            pams = get_pams_manager()
            print('\nStarting persistent-ai-memory...')
            pid = pams.start()
            if pid > 0:
                print(f'  PAMS PID: {pid}')
                if pams.http_port:
                    print(f'  HTTP API: http://127.0.0.1:{pams.http_port}')
        if rc == 0 and args.open:
            webbrowser.open(f'http://127.0.0.1:{args.port}/chat')
        return rc

    parser.print_help()
    return 0

if __name__ == '__main__':
    sys.exit(main())
