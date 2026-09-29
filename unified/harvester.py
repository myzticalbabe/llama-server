"""
harvester.py  —  Parallel file system data harvester.

Uses free-threaded Python 3.14t to run multiple model instances in true
parallel, each processing a different directory subtree. Outputs structured
JSONL results per worker, merged at completion.

Usage:
    python harvester.py --root D:/ --model model.gguf --workers 8
    python harvester.py --root C:/Users/clint/Documents --workers 8 --dry-run
"""

import argparse, json, os, sys, time, threading, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

MODEL_PATH = os.environ.get(
    'HARVESTER_MODEL',
    str(Path.home() / 'llama-server' / 'models' / 'Qwen2.5-1.5B-Instruct' / 'qwen2.5-1.5b-instruct-q4_k_m.gguf'),
)
TEXT_EXTENSIONS = {'.py', '.txt', '.md', '.json', '.yaml', '.yml', '.toml', '.cfg',
                   '.ini', '.conf', '.log', '.csv', '.xml', '.html', '.css', '.js',
                   '.ts', '.jsx', '.tsx', '.sh', '.ps1', '.bat', '.env', '.rst'}
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea',
             '.vscode', 'bin', 'obj', '.cache', '.config', '.local'}

@dataclass
class HarvestResult:
    path: str = ''
    size: int = 0
    ext: str = ''
    summary: str = ''
    topics: list = field(default_factory=list)
    error: str = ''

from paths import HARVESTER_OUTPUT as _HARVESTER_OUTPUT

OUTPUT_DIR = _HARVESTER_OUTPUT

def scan_files(root: Path) -> list[Path]:
    files = []
    for dirpath, dirs, filenames in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in filenames:
            ext = Path(f).suffix.lower()
            if ext in TEXT_EXTENSIONS:
                files.append(Path(dirpath) / f)
    return files

def split_by_subtree(files: list[Path], n: int) -> list[list[Path]]:
    groups = [[] for _ in range(n)]
    for i, f in enumerate(files):
        groups[i % n].append(f)
    return groups

def process_file(model, file_path: Path, max_chars: int = 800) -> HarvestResult:
    result = HarvestResult(path=str(file_path), ext=file_path.suffix.lower(), size=file_path.stat().st_size)
    try:
        content = file_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        result.error = str(e)
        return result
    if not content.strip():
        result.error = 'empty'
        return result
    preview = content[:max_chars]
    prompt = (
        f'<|system|>\nYou analyze text files and output a one-line summary.\n'
        f'<|user|>\nFile: {file_path.name}\nContent:\n{preview}\n'
        f'<|assistant|>\nSummary: '
    )
    try:
        out = model.create_completion(
            prompt=prompt,
            max_tokens=80, temperature=0.1
        )
        result.summary = out['choices'][0]['text'].strip().strip('"')
    except Exception as e:
        result.error = str(e)
    return result

def worker_task(worker_id: int, files: list[Path], output_file: Path, model_path: str, n_threads: int):
    from llama_cpp import Llama
    model = Llama(model_path=model_path, n_ctx=2048, verbose=False, n_threads=n_threads, n_gpu_layers=0)
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path in files:
            result = process_file(model, file_path)
            f.write(json.dumps({'worker': worker_id, **result.__dict__}) + '\n')
            count += 1
            if count % 50 == 0:
                f.flush()
    return count

def harvest(root: Path, model_path: str, workers: int, dry_run: bool = False):
    print(f'Scanning {root}...')
    files = scan_files(root)
    print(f'Found {len(files)} text files')
    if not files:
        return
    groups = split_by_subtree(files, workers)
    per_worker = [len(g) for g in groups]
    print(f'Worker file distribution: min={min(per_worker)}, max={max(per_worker)}, avg={sum(per_worker)/workers:.0f}')

    if dry_run:
        print('Dry run — no processing.')
        return

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    n_threads = max(1, 16 // workers)

    print(f'\nLaunching {workers} workers ({n_threads} threads each)...')
    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, group in enumerate(groups):
            if not group:
                continue
            out = output_dir / f'worker_{i:02d}_{timestamp}.jsonl'
            futures[pool.submit(worker_task, i, group, out, model_path, n_threads)] = i
        for future in as_completed(futures):
            wid = futures[future]
            try:
                n = future.result()
                results.append((wid, n))
            except Exception as e:
                print(f'  Worker {wid} FAILED: {e}')
                results.append((wid, 0))

    elapsed = time.time() - start
    total = sum(n for _, n in results)
    print(f'\nDone: {total} files in {elapsed:.0f}s ({total/elapsed:.1f} files/s)')

    # Merge results
    merge_file = output_dir / f'harvest_{timestamp}.jsonl'
    with open(merge_file, 'w') as out:
        for wid, _ in results:
            src_path = output_dir / f'worker_{wid:02d}_{timestamp}.jsonl'
            if src_path.exists():
                out.write(src_path.read_text())
    print(f'Merged: {merge_file}')
    print(f'Total size: {merge_file.stat().st_size / 1e6:.1f} MB')

def main():
    parser = argparse.ArgumentParser(prog='harvester', description='Parallel file system data harvester')
    parser.add_argument('--root', default=str(Path.cwd()), help='Root directory to scan')
    parser.add_argument('--model', default=MODEL_PATH, help='GGUF model path')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--dry-run', action='store_true', help='Scan only, no model processing')
    args = parser.parse_args()
    harvest(Path(args.root), args.model, args.workers, args.dry_run)

if __name__ == '__main__':
    main()
