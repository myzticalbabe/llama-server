import sys, os, json, time, datetime, numpy as np, ctypes
from pathlib import Path

MODEL_PATH = os.environ.get(
    'EMBED_MODEL_QWEN',
    str(Path.home() / '.lmstudio' / 'models' / 'lmstudio-community' / 'Qwen3.5-9B-GGUF' / 'Qwen3.5-9B-Q4_K_M.gguf'),
)
CHECKPOINT = Path.home() / ".config" / "unified-llm" / "embed_checkpoint.json"
OUTPUT = Path.home() / "llama-server" / "embeddings" / "embeddings.jsonl"
TEXT_EXTS = {'.py', '.txt', '.md', '.json', '.yaml', '.yml', '.toml', '.cfg',
             '.ini', '.conf', '.log', '.csv', '.xml', '.html', '.css', '.js',
             '.ts', '.jsx', '.tsx', '.sh', '.ps1', '.bat', '.env', '.rst',
             '.c', '.cpp', '.h', '.hpp', '.rs', '.go', '.java', '.kt', '.scala',
             '.sql', '.r', '.m', '.swift', '.dart', '.lua', '.rb', '.pl', '.php',
             '.svelte', '.vue'}
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea',
             '.vscode', 'bin', 'obj', '.cache', '.local', '.config'}
MODEL_DIRS = [
    Path.home() / "llama-server",
    Path.home() / ".lmstudio" / "models",
    r"D:\knowledge_base",
    r"D:\llama-server-build",
    Path.home() / "temp_llama_cpp_build" / "llama.cpp",
    Path.home() / "Desktop",
]

MAX_CHARS = 4096

class Embedder:
    """C API wrapper: fresh context per call, n_seq_max=1 for full per-file context."""

    def __init__(self, model_path, n_ctx=8192, n_threads=16):
        import llama_cpp as lc
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.lc = lc
        t0 = time.time()
        params = lc.llama_model_default_params()
        self._model = lc.llama_load_model_from_file(model_path.encode(), params)
        self._vocab = lc.llama_model_get_vocab(self._model)
        self.n_embd = lc.llama_model_n_embd(self._model)
        print(f"Model: {self.n_embd}d, loaded in {time.time()-t0:.1f}s")

    def tokenize(self, text):
        lc = self.lc
        text_bytes = text.encode("utf-8")
        max_tokens = self.n_ctx
        tokens = (lc.llama_token * max_tokens)()
        n = lc.llama_tokenize(self._vocab, text_bytes, len(text_bytes), tokens, max_tokens, True, True)
        if n < 0:
            raise RuntimeError(f"tokenize returned {n}")
        return tokens[:n]

    def embed_one(self, text):
        import numpy as np
        lc = self.lc

        tokens = self.tokenize(text)
        n_tokens = len(tokens)
        if n_tokens > self.n_ctx:
            n_tokens = self.n_ctx
            tokens = tokens[:n_tokens]

        ctx_params = lc.llama_context_default_params()
        ctx_params.n_ctx = max(n_tokens, 256)
        ctx_params.n_batch = n_tokens
        ctx_params.n_ubatch = n_tokens
        ctx_params.n_seq_max = 1
        ctx_params.embeddings = True
        ctx_params.flash_attn = True
        ctx_params.n_threads = self.n_threads
        ctx_params.n_threads_batch = self.n_threads

        ctx = lc.llama_new_context_with_model(self._model, ctx_params)
        try:
            batch = lc.llama_batch_init(n_tokens, 0, 1)
            batch.n_tokens = n_tokens
            for i in range(n_tokens):
                batch.token[i] = tokens[i]
                batch.pos[i] = i
                batch.seq_id[i][0] = 0
                batch.n_seq_id[i] = 1
                batch.logits[i] = True

            ret = lc.llama_decode(ctx, batch)
            lc.llama_batch_free(batch)
            if ret != 0:
                raise RuntimeError(f"llama_decode returned {ret}")

            ptr = lc.llama_get_embeddings_seq(ctx, 0)
            if ptr:
                return np.array(ptr[:self.n_embd], dtype=np.float32)
            ptr = lc.llama_get_embeddings(ctx)
            if not ptr:
                raise RuntimeError("No embeddings returned")
            all_emb = np.array(ptr[:n_tokens * self.n_embd], dtype=np.float32).reshape(n_tokens, self.n_embd)
            return np.mean(all_emb, axis=0)
        finally:
            lc.llama_free(ctx)

    def close(self):
        if hasattr(self, '_model') and self._model:
            self.lc.llama_free_model(self._model)
            self._model = None

def scan_files(roots=None):
    files = []
    for root in (roots or MODEL_DIRS):
        root = Path(root).expanduser().resolve()
        if not root.exists(): continue
        for dirpath, dirs, filenames in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in filenames:
                if Path(f).suffix.lower() in TEXT_EXTS:
                    fp = Path(dirpath) / f
                    try:
                        if fp.stat().st_size > 0: files.append(str(os.path.realpath(fp)))
                    except: pass
    files = list(set(files))
    files.sort(key=lambda f: os.path.getsize(f))
    return files

def read_file(path, max_chars=MAX_CHARS):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[:max_chars]
    except: return ""

def write_checkpoint(completed, output_lines):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"completed": sorted(completed), "output_lines": output_lines,
               "timestamp": datetime.datetime.now().isoformat()},
              CHECKPOINT.open("w"), indent=2)

def full_run(max_chars=MAX_CHARS, resume=False, max_files=0):
    files = scan_files()
    print(f"Found {len(files)} text files")
    if not files: return

    if resume:
        cp = {}
        if CHECKPOINT.exists():
            cp = json.loads(CHECKPOINT.read_text())
        completed = set(cp.get("completed", []))
        output_lines = cp.get("output_lines", 0)
        files = [f for f in files if f not in completed]
        print(f"Resuming: {len(completed)} done, {len(files)} remaining")
    else:
        completed = set()
        output_lines = 0
        if CHECKPOINT.exists(): CHECKPOINT.unlink()

    if max_files > 0: files = files[:max_files]
    if not files: print("All done!"); return

    print("Loading model...")
    embedder = Embedder(MODEL_PATH, n_ctx=max_chars * 4)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    total = len(files)
    t_start = time.time()

    try:
        for idx, fp in enumerate(files):
            content = read_file(fp, max_chars)
            if not content:
                completed.add(fp)
                continue

            try:
                emb = embedder.embed_one(content)
            except Exception as e:
                print(f"  Error at {os.path.basename(fp)}: {e}", flush=True)
                completed.add(fp)
                continue

            with open(OUTPUT, "a", encoding="utf-8") as f:
                rec = {"path": fp, "dim": 4096, "embedding": emb.round(6).tolist(),
                       "time": datetime.datetime.now().isoformat()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            completed.add(fp)
            output_lines += 1

            if output_lines % 50 == 0:
                write_checkpoint(completed, output_lines)

            elapsed = time.time() - t_start
            rate = output_lines / elapsed if elapsed > 0 else 0
            remaining = len(files) - output_lines
            eta = remaining / rate if rate > 0 else 0
            n_tok = len(embedder.tokenize(content))
            fname = os.path.basename(fp).encode('ascii', errors='replace').decode('ascii')
            print(f"[{output_lines}/{total}] {rate:.2f} files/s, ETA: {datetime.timedelta(seconds=int(eta))}, file: {fname} ({n_tok} tok)", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        write_checkpoint(completed, output_lines)
        print(f"\nSaved {output_lines} embeddings to {OUTPUT}")
        embedder.close()

if __name__ == "__main__":
    if "--run" in sys.argv:
        resume = "--resume" in sys.argv
        maxf = 0
        maxc = MAX_CHARS
        for i, a in enumerate(sys.argv):
            if a == "--max" and i+1 < len(sys.argv): maxf = int(sys.argv[i+1])
            if a == "--chars" and i+1 < len(sys.argv): maxc = int(sys.argv[i+1])
        full_run(max_chars=maxc, resume=resume, max_files=maxf)
    else:
        print("Usage:")
        print("  python direct_embed.py --run")
        print("  python direct_embed.py --run --resume")
        print("  python direct_embed.py --run --max 1000")
        print("  python direct_embed.py --run --chars 2048")
