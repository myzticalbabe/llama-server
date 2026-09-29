"""
research_analyzer.py  —  Second-pass analyzer using gemma-4 for deep research.

Workflow:
  Harvester (Qwen2.5)  →  merged.jsonl  →  gemma-4 deep analysis  →  enhanced.jsonl

Single model:
    python research_analyzer.py --input harvest.jsonl
    python research_analyzer.py --input harvest.jsonl --prompt "Find security bugs"
    python research_analyzer.py --input harvest.jsonl --prompt "Suggest improvements" --out analysis.jsonl

Dual model (E2B + E4B in parallel):
    python research_analyzer.py --input harvest.jsonl --dual
    python research_analyzer.py --input harvest.jsonl --models gemma4 gemma4-e4b --ports 18090 18091

Use running servers:
    python research_analyzer.py --input harvest.jsonl --no-server
    python research_analyzer.py --input harvest.jsonl --dual --no-server --ports 18090 18091

Start/stop servers:
    python research_analyzer.py --start --model=gemma4 --port 18090
    python research_analyzer.py --stop --port 18090
"""

import json, os, sys, time, argparse, urllib.request, urllib.error, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).parent
ROUTER_PY = BASE_DIR / 'router.py'

DEFAULT_PROMPT = (
    "You are analyzing a codebase. For each file below, provide a brief research-quality analysis: "
    "what it does, how it fits into the larger system, any design patterns or issues you notice. "
    "Be concise (2-4 sentences per file)."
)

MODEL_ALIASES = {
    'e2b': 'gemma4',
    'gemma4': 'gemma4',
    'gemma4-e2b': 'gemma4',
    'e4b': 'gemma-4-E4B-it-Q4_K_M',
    'gemma4-e4b': 'gemma-4-E4B-it-Q4_K_M',
}


class GemmaAnalyzer:
    def __init__(self, port: int = 18090, model_query: str = 'gemma4',
                 max_tokens: int = 256):
        self._port = port
        self._model_query = MODEL_ALIASES.get(model_query, model_query)
        self._base = f'http://127.0.0.1:{port}'
        self._server_proc = None
        self._max_tokens = max_tokens

    def start_server(self) -> bool:
        from executor import LlamaServerExecutor, wait_for_health
        out = subprocess.run(
            [sys.executable, str(ROUTER_PY), '--json', f'--model={self._model_query}',
             f'--port={self._port}', '--task=research'],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            print(f'  Router error for {self._model_query}: {out.stderr.strip() or "exit " + str(out.returncode)}')
            return False
        try:
            plan = json.loads(out.stdout)
        except json.JSONDecodeError:
            print(f'  Router returned invalid JSON for {self._model_query}')
            return False
        if 'error' in plan:
            print(f'  Router: {plan["error"]}')
            return False
        executor = LlamaServerExecutor()
        pid = executor.start(plan['args'], plan.get('env'))
        if pid is None or pid <= 0:
            return False
        self._server_proc = executor
        print(f'  Waiting for {self._model_query} on port {self._port}...')
        ok = wait_for_health(self._port, executor, timeout=180)
        if ok:
            print(f'  {self._model_query} ready on {self._port}')
        return ok

    def stop_server(self):
        if self._server_proc:
            self._server_proc.stop()
            self._server_proc = None

    def analyze(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = json.dumps({
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": 0.2,
            "stream": False,
            "thinking": {"type": "disabled"},
        }).encode()

        req = urllib.request.Request(
            f'{self._base}/v1/chat/completions',
            data=body,
            headers={"Content-Type": "application/json"},
            method='POST',
        )
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            data = json.loads(resp.read())
            msg = data['choices'][0]['message']
            # Thinking models (gemma-4) may return empty content with the
            # text in reasoning_content when the thinking-disable flag is
            # ignored by the server build — fall back instead of recording
            # an empty analysis.
            text = (msg.get('content') or '').strip()
            if not text:
                text = (msg.get('reasoning_content') or '').strip()
            return text
        except Exception as e:
            return f'[ERROR: {e}]'

    def health(self) -> bool:
        try:
            r = urllib.request.urlopen(f'{self._base}/health', timeout=3)
            return r.status == 200
        except Exception:
            return False


def load_entries(path: str) -> list[dict]:
    entries = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_entries(entries: list[dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f'Wrote {len(entries)} entries to {path}')


def build_file_prompt(entry: dict, user_prompt: str) -> str:
    fpath = entry.get('path', '?')
    fname = fpath.split('\\')[-1].split('/')[-1]
    summary = entry.get('summary', '').strip()
    content = f'File: {fname}\nSummary: {summary}' if summary else f'File: {fname}'
    return f'Research question: {user_prompt}\n\n{content}\n\nAnalysis:'


def analyze_single(analyzer: GemmaAnalyzer, entries: list[dict], user_prompt: str,
                   batch_size: int = 1) -> list[dict]:
    results = []
    total = len(entries)
    for i, e in enumerate(entries):
        print(f'  [{analyzer._port}] File {i+1}/{total}: {e.get("path","?").split("\\")[-1].split("/")[-1]}')

        prompt = build_file_prompt(e, user_prompt)
        analysis = analyzer.analyze(prompt)
        results.append({**e, 'deep_analysis': analysis, '_model': analyzer._model_query, '_port': analyzer._port})
    return results


def analyze_dual(analyzers: list[GemmaAnalyzer], entries: list[dict], user_prompt: str,
                 batch_size: int = 1) -> list[dict]:
    n = len(analyzers)
    splits = [entries[i::n] for i in range(n)]
    all_results = []

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {}
        for i, a in enumerate(analyzers):
            f = pool.submit(analyze_single, a, splits[i], user_prompt, batch_size)
            futures[f] = a

        for future in as_completed(futures):
            a = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f'  [{a._port}] Finished {len(results)} entries')
            except Exception as e:
                print(f'  [{a._port}] FAILED: {e}')

    all_results.sort(key=lambda r: r.get('path', ''))
    return all_results


def start_servers(model_queries: list[str], ports: list[int]) -> list[GemmaAnalyzer]:
    analyzers = [GemmaAnalyzer(p, m) for m, p in zip(model_queries, ports)]

    def start_one(analyzer):
        ok = analyzer.start_server()
        return analyzer, ok

    with ThreadPoolExecutor(max_workers=len(analyzers)) as pool:
        futures = {pool.submit(start_one, a): a for a in analyzers}
        for future in as_completed(futures):
            a, ok = future.result()
            if not ok:
                print(f'  Failed to start {a._model_query} on {a._port}', file=sys.stderr)

    healthy = [a for a in analyzers if a.health()]
    return healthy


def main():
    parser = argparse.ArgumentParser(prog='research_analyzer',
        description='Second-pass deep analysis using gemma-4')
    parser.add_argument('--input', help='Harvester JSONL input')
    parser.add_argument('--out', default='', help='Output JSONL (default: input with _analyzed suffix)')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT, help='Research question/prompt for gemma-4')
    parser.add_argument('--model', default='gemma4', help='Model query (gemma4, gemma4-e4b, e2b, e4b)')
    parser.add_argument('--port', type=int, default=18090, help='Port for gemma-4 server')
    parser.add_argument('--dual', action='store_true', help='Run E2B + E4B in parallel')
    parser.add_argument('--models', nargs='*', help='Model queries for --dual')
    parser.add_argument('--ports', type=int, nargs='*', help='Ports for --dual')
    parser.add_argument('--no-server', action='store_true', help='Skip server start (use running instance)')
    parser.add_argument('--batch', type=int, default=5, help='Files per batch (default: 5)')
    parser.add_argument('--max-tokens', type=int, default=256,
                        help='Max tokens per analysis (thinking models need 768+ to finish reasoning AND answer)')
    parser.add_argument('--start', action='store_true', help='Start server(s) and exit')
    parser.add_argument('--stop', nargs='?', const='all', metavar='PORT', help='Stop server instance(s)')

    args = parser.parse_args()

    # --stop
    if args.stop is not None:
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
        return 0

    # --start
    if args.start:
        if args.dual:
            models = [args.model or 'gemma4', 'gemma4-e4b']
            ports = [args.port, (args.ports or [18091])[0]]
        else:
            models = [args.model or 'gemma4']
            ports = [args.port]
        analyzers = start_servers(models, ports)
        for a in analyzers:
            print(f'{a._model_query} running on port {a._port}')
        return 0

    # --input required for analysis
    if not args.input:
        parser.print_help()
        return 1

    entries = load_entries(args.input)
    print(f'Loaded {len(entries)} entries from {args.input}')

    if args.dual:
        models = [args.model or 'gemma4', 'gemma4-e4b']
        ports = [args.port]
        if args.ports:
            ports = args.ports[:2]
        else:
            ports.append(ports[0] + 1)
        models = models[:len(ports)]
        print(f'Dual mode: {dict(zip(ports, models))}')
        analyzers = [GemmaAnalyzer(p, m, args.max_tokens) for p, m in zip(ports, models)]

        if not args.no_server:
            print('Starting gemma-4 servers...')
            analyzers = start_servers(models, ports)
            if not analyzers:
                print('Failed to start any server', file=sys.stderr)
                return 1
        else:
            for a in analyzers:
                if not a.health():
                    print(f'No server on port {a._port}', file=sys.stderr)
                    return 1
                print(f'  Server on {a._port} ({a._model_query}) running')

        try:
            results = analyze_dual(analyzers, entries, args.prompt, args.batch)
        finally:
            if not args.no_server:
                for a in analyzers:
                    a.stop_server()
    else:
        analyzer = GemmaAnalyzer(args.port, args.model, args.max_tokens)
        if not args.no_server:
            if not analyzer.start_server():
                print('Failed to start gemma-4 server', file=sys.stderr)
                return 1
        else:
            if not analyzer.health():
                print(f'No server on port {args.port}', file=sys.stderr)
                return 1

        try:
            results = analyze_single(analyzer, entries, args.prompt, args.batch)
        finally:
            if not args.no_server:
                analyzer.stop_server()

    out_path = args.out or args.input.replace('.jsonl', '_analyzed.jsonl')
    if out_path == args.input:
        out_path = args.input.replace('.jsonl', '_analyzed.jsonl')
    save_entries(results, out_path)

    print(f'\n=== Sample Analysis ===')
    for r in results[:2]:
        model = r.get('_model', r.get('_port', '?'))
        preview = (r.get('deep_analysis') or '')[:200]
        if preview:
            print(f'\n[{model}] {r["path"]}:')
            print(f'  {preview}...')

    return 0


if __name__ == '__main__':
    sys.exit(main())
