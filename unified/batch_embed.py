"""
batch_embed.py  —  Batch file embedder for 4096-dim native embeddings.

Starts llama-server embedding server, reads text files, sends batched
embedding requests, saves results incrementally as JSONL. Resume-safe
via checkpoint file. Recovers from server restarts.

Usage:
    python batch_embed.py                    # Default: scan all model dirs
    python batch_embed.py --root D:/knowledge_base  # Scan specific dir
    python batch_embed.py --resume                   # Resume from checkpoint
    python batch_embed.py --batch 400                # Override batch size
    python batch_embed.py --dry-run                  # Count files only
"""

import os, sys, json, time, datetime, subprocess, signal, atexit
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
from paths import SERVER_EXE, OV_LIBS

import corpus  # shared TEXT_EXTS / SKIP_DIRS + canonical stack doc
TEXT_EXTS = corpus.TEXT_EXTS
SKIP_DIRS = corpus.SKIP_DIRS

MODEL = os.environ.get(
    'EMBED_MODEL_QWEN',
    str(Path.home() / '.lmstudio' / 'models' / 'lmstudio-community' / 'Qwen3.5-9B-GGUF' / 'Qwen3.5-9B-Q4_K_M.gguf'),
)
PORT = 18082
CHECKPOINT = Path.home() / ".config" / "unified-llm" / "embed_checkpoint.json"
OUTPUT = Path.home() / "llama-server" / "embeddings" / "embeddings.jsonl"

MODEL_DIRS = [
    Path.home() / "llama-server" / "models",
    Path.home() / ".lmstudio" / "models",
    Path.home() / "source" / "tensorflow",
    Path.home() / "source" / "langchain",
    Path.home() / "source" / "nomic-embed",
    Path.home() / "source" / "gpt4all",
    Path.home() / "llama.cpp",
    Path.home() / "lora",
]

# Also include specified root dirs
EXTRA_ROOTS = []


def scan_files(roots=None, max_chars=10000):
    files = []
    roots = roots or MODEL_DIRS
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.exists():
            continue
        for dirpath, dirs, filenames in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in filenames:
                ext = Path(f).suffix.lower()
                if ext in TEXT_EXTS:
                    fp = Path(dirpath) / f
                    try:
                        size = fp.stat().st_size
                        if size == 0:
                            continue
                        files.append(fp)
                    except OSError:
                        continue

    # Deduplicate by realpath
    seen = set()
    unique = []
    for fp in files:
        real = os.path.realpath(str(fp))
        if real not in seen:
            seen.add(real)
            unique.append(real)
    return sorted(unique)


def read_file(path, max_chars=10000):
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars:
            content = content[:max_chars]
        return content
    except Exception:
        return ""


def read_checkpoint():
    if CHECKPOINT.exists():
        try:
            return json.loads(CHECKPOINT.read_text())
        except Exception:
            pass
    return {"completed": [], "output_lines": 0}


def write_checkpoint(completed, output_lines):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    data = {"completed": sorted(completed), "output_lines": output_lines,
            "timestamp": datetime.datetime.now().isoformat()}
    CHECKPOINT.write_text(json.dumps(data, indent=2))


def start_server():
    env = dict(os.environ)
    env["GGML_OPENVINO_DISABLE"] = "1"
    env["GGML_OPENVINO_BLACKLIST"] = "1"
    env["PATH"] = f"{OV_LIBS};{env.get('PATH', '')}"

    proc = subprocess.Popen(
        [SERVER_EXE, "--model", MODEL,
         "--host", "127.0.0.1", "--port", str(PORT),
         "--ctx-size", "2048", "--threads", "16",
         "--batch-size", "2048", "--ubatch-size", "1024",
         "--no-warmup", "-ngl", "0",
         "--embedding", "--pooling", "mean"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    return proc


def wait_server(timeout=120):
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def get_embedding(texts):
    import urllib.request
    body = json.dumps({"input": texts, "model": "default"}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=600)
    data = json.loads(resp.read())
    return [e["embedding"] for e in data["data"]]


def main():
    # Parse args
    extra_roots = []
    batch_size = 200
    dry_run = False
    resume = False

    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--root" and i + 1 < len(args):
            extra_roots.append(Path(args[i + 1]).resolve())
        elif a == "--batch" and i + 1 < len(args):
            batch_size = int(args[i + 1])
        elif a == "--dry-run":
            dry_run = True
        elif a == "--resume":
            resume = True

    all_roots = MODEL_DIRS + extra_roots if extra_roots else MODEL_DIRS

    print(f"Scanning {len(all_roots)} directories...")
    files = scan_files(all_roots)
    print(f"Found {len(files)} text files")
    if not files:
        return

    if dry_run:
        print(f"Dry run: would embed {len(files)} files")
        return

    if resume:
        cp = read_checkpoint()
        completed = set(cp.get("completed", []))
        output_lines = cp.get("output_lines", 0)
        files = [f for f in files if f not in completed]
        print(f"Resuming: {len(completed)} done, {len(files)} remaining")
    else:
        completed = set()
        output_lines = 0
        if CHECKPOINT.exists():
            CHECKPOINT.unlink()

    if not files:
        print("All files already embedded!")
        return

    # Start server if not running
    import urllib.request
    server_ok = False
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
        server_ok = r.status == 200
    except Exception:
        pass

    proc = None
    if not server_ok:
        print("Starting llama-server...")
        proc = start_server()
        if not wait_server():
            print("Server failed to start", file=sys.stderr)
            return
        print("Server ready")
    else:
        print("Server already running")

    # Ensure output dir
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total = len(files)
    t_start = time.time()

    try:
        for batch_start in range(0, total, batch_size):
            batch = files[batch_start:batch_start + batch_size]
            batch_files = batch

            texts = []
            valid_paths = []
            for fp in batch_files:
                content = read_file(fp)
                if content:
                    texts.append(content)
                    valid_paths.append(fp)

            if not texts:
                continue

            retries = 3
            for attempt in range(retries):
                try:
                    embeddings = get_embedding(texts)
                    break
                except Exception as e:
                    print(f"  Retry {attempt+1}/{retries}: {e}", flush=True)
                    if attempt < retries - 1:
                        time.sleep(5)
                        # Check if server is still alive
                        try:
                            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
                        except Exception:
                            print("  Server dead, restarting...", flush=True)
                            if proc:
                                proc.terminate()
                                proc.wait()
                            proc = start_server()
                            wait_server()
                    else:
                        print(f"  FAILED batch at file {batch_start+1}", flush=True)
                        raise

            # Write results immediately
            with open(OUTPUT, "a", encoding="utf-8") as f:
                for i, fp in enumerate(valid_paths):
                    rec = {"path": fp, "dim": 4096, "embedding": embeddings[i],
                           "time": datetime.datetime.now().isoformat()}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    completed.add(fp)
                    output_lines += 1

            checkpoint_interval = max(1, batch_size * 5)
            if len(completed) % checkpoint_interval < batch_size:
                write_checkpoint(completed, output_lines)

            # Progress
            elapsed = time.time() - t_start
            done = len(completed) - (output_lines - output_lines)  # FIX: track properly
            rate = len(completed) / elapsed if elapsed > 0 else 0
            remaining = total - len(completed)
            eta = remaining / rate if rate > 0 else 0
            print(f"[{len(completed)}/{total}] "
                  f"{rate:.1f} files/s, "
                  f"ETA: {datetime.timedelta(seconds=int(eta))}  "
                  f"batch {len(texts)} files", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
    finally:
        write_checkpoint(completed, output_lines)
        print(f"Saved {output_lines} embeddings to {OUTPUT}")
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
