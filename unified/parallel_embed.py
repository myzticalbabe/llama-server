"""
parallel_embed.py  —  Parallel batch embedder using multiple llama-server instances.

Uses free-threaded Python 3.15t to run N embedding servers concurrently and
distribute batches across them for true parallel throughput.
"""

import json, os, sys, time, subprocess, argparse, shlex
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import urllib.request

# Use free-threaded Python
PYTHON_FT = Path(r"C:\Users\clint\AppData\Local\Programs\Python\Python315\python3.15t.exe")
ROUTER_PY = Path(__file__).parent / "router.py"

from paths import SERVER_EXE, OV_LIBS
import semantic  # reuse Chunker, pack_embedding, INDEX_FILE, CHUNK_SIZE, etc.

INDEX_FILE = semantic.INDEX_FILE
CHUNK_SIZE = semantic.CHUNK_SIZE
CHUNK_OVERLAP = semantic.CHUNK_OVERLAP
EMBED_DIM = semantic.EMBED_DIM
pack_embedding = semantic.pack_embedding
TEXT_EXTS = semantic.TEXT_EXTS
SKIP_DIRS = semantic.SKIP_DIRS
INDEX_MAX_MB = semantic.INDEX_MAX_MB
compact_index = semantic.compact_index

# Build output, vendored graph dumps and bulk CSVs are not knowledge. Measured
# 2026-10-31: they were 2,235,928 of 2,325,948 chunks (96%) under the default
# roots -- a single generated SPIR-V C file was worth ~284,300 chunks -- and
# none of the chunks anyone searches for. Excluding them turns a 4-day job into
# an afternoon one.
DEFAULT_EXCLUDES = [
    "build-spark", "/build/", "graphify_codebase_atlas",
    "graph.json", "graph.html", "_full.csv", ".log",
]


def _excluded(fp: Path, patterns: Optional[List[str]]) -> bool:
    if not patterns:
        return False
    low = fp.as_posix().lower()
    return any(p.lower() in low for p in patterns)


def get_embed_plan(port: int) -> Dict[str, Any]:
    """Get router plan for embedding server on given port."""
    result = subprocess.run(
        [str(PYTHON_FT), str(ROUTER_PY), "--json",
         f"--model=mxbai-embed-large-v1", f"--port={port}", "--task=embed"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"Router failed for port {port}: {result.stderr}")
    plan = json.loads(result.stdout)
    if "error" in plan:
        raise RuntimeError(f"Router error for port {port}: {plan['error']}")
    return plan


class EmbeddingServer:
    """Manages one llama-server embedding instance."""

    def __init__(self, port: int, plan: Dict[str, Any], exe: Optional[Path] = None,
                 cmd: Optional[List[str]] = None):
        self.port = port
        self.plan = plan
        self.cmd = cmd
        # plan['exe'] lets the router pick the binary (the vulkan build for
        # embeddings); an explicit --server-exe still wins.
        self.exe = Path(exe) if exe else Path(plan.get("exe") or SERVER_EXE)
        self.proc: subprocess.Popen | None = None
        self.base_url = f"http://127.0.0.1:{port}"

    def start(self) -> bool:
        env = dict(os.environ)
        env["GGML_OPENVINO_DISABLE"] = "1"
        env["GGML_OPENVINO_BLACKLIST"] = "1"
        ov_path = str(OV_LIBS)
        if ov_path not in env.get("PATH", ""):
            env["PATH"] = f"{ov_path};{env['PATH']}"
        if self.plan.get("env"):
            env.update(self.plan["env"])

        # Below-normal priority: this job is designed to be folded into a day of
        # other work, so it must yield to whatever else is running.
        flags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            flags |= subprocess.CREATE_NO_WINDOW
        if hasattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS"):
            flags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS

        # --server-cmd wins: it lets a non-llama-server backend (e.g.
        # ov_embed_server.py on the OpenVINO IR) serve the same /health and
        # /v1/embeddings endpoints. {port} is substituted per instance.
        argv = ([c.format(port=self.port) for c in self.cmd] if self.cmd
                else [str(self.exe)] + self.plan["args"])
        self.proc = subprocess.Popen(
            argv,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        return self.wait_healthy()

    def wait_healthy(self, timeout: int = 120) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = urllib.request.urlopen(f"{self.base_url}/health", timeout=3)
                if r.status == 200:
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def embed(self, texts: List[str]) -> List[List[float]]:
        body = json.dumps({"input": texts, "model": "default"}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())
        return [e["embedding"] for e in data["data"]]

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class ParallelIndexer:
    """Distributes embedding batches across multiple servers."""

    def __init__(self, num_servers: int = 4, batch_size: int = 200,
                 server_exe: Optional[Path] = None, ngl: Optional[int] = None,
                 server_cmd: Optional[str] = None):
        self.server_cmd = shlex.split(server_cmd) if server_cmd else None
        self.num_servers = num_servers
        self.batch_size = batch_size
        self.server_exe = Path(server_exe) if server_exe else None
        self.ngl = ngl
        self.servers: List[EmbeddingServer] = []
        self._server_idx = 0
        self.last_failed = 0

    def start_servers(self, start_port: int = 18083):
        print(f"Starting {self.num_servers} embedding servers on ports {start_port}-{start_port + self.num_servers - 1}...")
        plans = []
        for i in range(self.num_servers):
            port = start_port + i
            print(f"  Getting plan for port {port}...")
            plans.append(get_embed_plan(port))

        if self.ngl is not None:
            # Measured 2026-10-31 on the Arc 140T iGPU: the vulkan build at
            # -ngl 99 embeds at 26.9 chunks/s against 16.5 chunks/s for the same
            # binary at -ngl 0 and 4.75 chunks/s for the cpu-only build. The
            # router still forces cpu from a 2026-05-22 text-generation
            # benchmark, which does not describe this (prompt-only) workload.
            for plan in plans:
                args = list(plan["args"])
                if "-ngl" in args:
                    args[args.index("-ngl") + 1] = str(self.ngl)
                else:
                    args += ["-ngl", str(self.ngl)]
                plan["args"] = args

        with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = {}
            for i, plan in enumerate(plans):
                port = start_port + i
                server = EmbeddingServer(port, plan, exe=self.server_exe,
                                         cmd=self.server_cmd)
                self.servers.append(server)
                futures[executor.submit(server.start)] = server

            for future in as_completed(futures):
                server = futures[future]
                if not future.result():
                    raise RuntimeError(f"Server on port {server.port} failed to start")

        print(f"All {len(self.servers)} servers healthy")

    def _next_server(self) -> EmbeddingServer:
        server = self.servers[self._server_idx % len(self.servers)]
        self._server_idx += 1
        return server

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        server = self._next_server()
        return server.embed(texts)

    def embed_batches_parallel(self, all_texts: List[List[str]]) -> List[List[List[float]]]:
        """Embed multiple batches in parallel across servers."""
        results = [None] * len(all_texts)
        with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
            futures = {}
            for i, texts in enumerate(all_texts):
                server = self.servers[i % len(self.servers)]
                futures[executor.submit(server.embed, texts)] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"  Batch {idx} failed: {e}")
                    results[idx] = []
        return results

    def stop_all(self):
        for server in self.servers:
            server.stop()

    def _index_size_mb(self, output: Path) -> float:
        try:
            return output.stat().st_size / 1e6
        except OSError:
            return 0.0

    def _existing_chunks(self, output: Path) -> set:
        seen = set()
        if not output.exists():
            return seen
        with output.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen.add((rec.get("path"), rec.get("offset")))
        return seen

    def index_chunks(self, chunks: List[Dict], output: Path = INDEX_FILE, resume: bool = True,
                     max_chunks: Optional[int] = None,
                     max_seconds: Optional[float] = None) -> int:
        output.parent.mkdir(parents=True, exist_ok=True)

        if self._index_size_mb(output) > INDEX_MAX_MB:
            before_mb = self._index_size_mb(output)
            print(f"  index is {before_mb:.0f} MB (cap {INDEX_MAX_MB} MB) -- compacting before append", flush=True)
            stats = compact_index(output)
            print(f"  compact: {stats['records_before']} -> {stats['records_after']} records, {before_mb:.0f} -> {stats['size_after_mb']:.0f} MB ({stats['duplicates_dropped']} duplicates dropped)", flush=True)
            if self._index_size_mb(output) > INDEX_MAX_MB:
                print(f"  WARNING: still {self._index_size_mb(output):.0f} MB over the {INDEX_MAX_MB} MB cap with no duplicates left.", flush=True)

        done = self._existing_chunks(output) if resume else set()
        if done:
            before = len(chunks)
            chunks = [c for c in chunks if (c.get("path"), c.get("offset")) not in done]
            print(f"  resume: {before - len(chunks)} chunks already indexed, {len(chunks)} to go", flush=True)
        if not chunks:
            return 0
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
            print(f"  slice: capped at {len(chunks)} chunk(s) for this run", flush=True)

        total = len(chunks)
        indexed = 0
        failed = 0
        t0 = time.time()

        batch_size = self.batch_size
        batches = [chunks[i:i + batch_size] for i in range(0, total, batch_size)]
        print(f"  {len(batches)} batches to embed across {len(self.servers)} servers", flush=True)

        # Incremental: append each wave the moment it finishes. The previous
        # version buffered every embedding in memory and wrote once at the end,
        # so a slice stopped by a timeout or a kill persisted nothing (measured
        # 2026-10-31: a 2-hour run produced 0 records and 0 visible progress).
        # Held in memory it was also ~2.38e9 floats, which cannot fit.
        wave_size = batch_size * max(1, len(self.servers))
        stop_reason = "complete"
        with open(output, "a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=max(1, len(self.servers))) as pool:
                for start in range(0, total, wave_size):
                    wave = batches[start // batch_size:(start + wave_size) // batch_size]
                    futures = {
                        pool.submit(server.embed, [c["text"] for c in batch]): batch
                        for server, batch in zip(self.servers, wave)
                    }
                    for fut in as_completed(futures):
                        batch = futures[fut]
                        try:
                            embeddings = fut.result()
                        except Exception as exc:
                            failed += len(batch)
                            print(f"  batch failed at offset {batch[0].get('offset')}: {str(exc)[:160]}", flush=True)
                            continue
                        if len(embeddings) != len(batch):
                            failed += len(batch)
                            print(f"  short batch {len(embeddings)}/{len(batch)} vectors -- dropped", flush=True)
                            continue
                        for chunk, emb in zip(batch, embeddings):
                            record = {**chunk, "embedding_b64": pack_embedding(emb), "dim": EMBED_DIM}
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            indexed += 1
                    f.flush()

                    elapsed = time.time() - t0
                    rate = indexed / elapsed if elapsed > 0 else 0
                    print(f"  [{indexed}/{total}] {rate:.1f} chunks/s {elapsed:.0f}s", flush=True)
                    if max_seconds is not None and elapsed >= max_seconds:
                        stop_reason = f"slice budget {max_seconds:.0f}s reached"
                        break

        self.last_failed = failed
        elapsed = time.time() - t0
        rate = indexed / elapsed if elapsed > 0 else 0
        print(f"  {stop_reason.upper()}: {indexed}/{total} chunks in {elapsed:.0f}s "
              f"({rate:.1f} chunks/s) -- all on disk", flush=True)
        if failed:
            print(f"  WARNING: {failed} chunk(s) not embedded -- re-run to retry", flush=True)
        return indexed


#: Extension value ranks for `--order value`. Documentation and source answer a
#: search; generated HTML, data dumps and model vocab JSON do not.
VALUE_RANK = {".md": 0, ".rst": 0, ".py": 1, ".js": 2, ".ts": 2, ".tsx": 2,
              ".jsx": 2, ".rs": 2, ".go": 3, ".c": 3, ".h": 3, ".cpp": 3,
              ".toml": 4, ".ini": 4, ".yaml": 4, ".yml": 4, ".bat": 5, ".ps1": 5,
              ".sh": 5, ".html": 6, ".json": 7, ".txt": 7, ".xml": 7, ".csv": 8}


def scan_files(roots: List[Path], excludes: Optional[List[str]] = None,
               order: str = "size") -> List[Path]:
    """Collect text files under `roots`.

    `order` decides what gets embedded FIRST, which is the whole game when the
    corpus is larger than one sitting:
      size  -- smallest first (historical behaviour)
      roots -- walk order, i.e. the order the caller listed the roots
      value -- root order, then extension rank, then size

    Root order used to be silently discarded: the old code always sorted by
    file size, so a priority-ordered root list still embedded a 200-byte
    archived stub before `src/CORE/*.py`. Measured 2026-10-31: with `size`,
    the priority roots were 5.4% durable while `tests/` and `archived_unified/`
    were being embedded.
    """
    files: List[tuple] = []
    for ridx, root in enumerate(roots):
        root = Path(root).expanduser().resolve()
        if not root.exists():
            continue
        for dirpath, dirs, filenames in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in filenames:
                if Path(f).suffix.lower() in TEXT_EXTS:
                    fp = Path(dirpath) / f
                    if _excluded(fp, excludes):
                        continue
                    try:
                        if fp.stat().st_size > 0:
                            files.append((ridx, fp))
                    except OSError:
                        pass
    seen = set()
    unique = []
    for ridx, fp in files:
        real = str(fp.resolve())
        if real not in seen:
            seen.add(real)
            unique.append((ridx, fp))
    def _size(p: Path) -> int:
        try:
            return p.stat().st_size
        except OSError:
            return 0
    if order == "roots":
        return [fp for _, fp in unique]
    if order == "value":
        unique.sort(key=lambda rf: (rf[0], VALUE_RANK.get(rf[1].suffix.lower(), 9),
                                    _size(rf[1])))
        return [fp for _, fp in unique]
    unique.sort(key=lambda rf: _size(rf[1]))
    return [fp for _, fp in unique]


LOCK_FILE = INDEX_FILE.with_name("semantic_index.lock")


def _pid_alive(pid: int) -> bool:
    """True if pid is still running. os.kill(pid, 0) TERMINATES on Windows,
    so this asks the kernel directly instead."""
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    except Exception:
        return True  # cannot tell -> assume alive, refuse to double-run


def _acquire_lock(path: Path) -> bool:
    try:
        if path.exists():
            try:
                pid = int(path.read_text().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and _pid_alive(pid):
                return False
            path.unlink(missing_ok=True)
        path.write_text(str(os.getpid()))
        return True
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Parallel semantic indexer")
    parser.add_argument("--roots", nargs="+", default=[
        str(Path.home() / "llama-server"),
        str(Path.home() / ".lmstudio" / "models"),
        r"D:\knowledge_base",
        r"D:\llama-server-build",
        str(Path.home() / "temp_llama_cpp_build" / "llama.cpp"),
        str(Path.home() / "Desktop"),
    ])
    parser.add_argument("--servers", type=int, default=4, help="Number of embedding servers")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size per request")
    parser.add_argument("--start-port", type=int, default=18083, help="Starting port")
    parser.add_argument("--rebuild", action="store_true", help="Move existing index aside")
    parser.add_argument("--resume", action="store_true", help="Resume from existing index")
    parser.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDES,
                        help="Path substrings to skip (build output, graph dumps, bulk CSVs, logs)")
    parser.add_argument("--no-default-excludes", action="store_true",
                        help="Also index build artefacts and graph dumps (the bulk of the corpus)")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Slice size: stop after this many chunks")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Slice budget: stop after N seconds, progress already on disk")
    parser.add_argument("--loop-sleep", type=float, default=0.0,
                        help="Fold the job into the day: rest N seconds between slices until done")
    parser.add_argument("--server-exe", default=None,
                        help="Override the llama-server binary (e.g. the vulkan build)")
    parser.add_argument("--ngl", type=int, default=None,
                        help="GPU layers to offload (-ngl); needs a binary with a GPU backend")
    parser.add_argument("--order", choices=["size", "roots", "value"], default="value",
                        help="What to embed first: size (smallest), roots (walk order), "
                             "value (root order, then extension rank, then size)")
    parser.add_argument("--server-cmd", default=None,
                        help="Spawn this command instead of llama-server; {port} is substituted "
                             "(e.g. the OpenVINO server: ov_embed_server.py --device GPU ...)")
    args = parser.parse_args()

    excludes = [] if args.no_default_excludes else (args.exclude or [])

    if args.rebuild and INDEX_FILE.exists():
        bak = INDEX_FILE.with_suffix(INDEX_FILE.suffix + ".bak")
        if bak.exists():
            bak.unlink()
        os.replace(INDEX_FILE, bak)
        print(f"  --rebuild: previous index moved to {bak.name} ({bak.stat().st_size/1e6:.0f} MB)")

    if not _acquire_lock(LOCK_FILE):
        print(f"  another slice holds {LOCK_FILE.name} -- refusing to double-run")
        return 1

    chunker = semantic.Chunker()
    indexer = ParallelIndexer(num_servers=args.servers, batch_size=args.batch_size,
                              server_exe=args.server_exe, ngl=args.ngl,
                              server_cmd=args.server_cmd)
    try:
        indexer.start_servers(args.start_port)
        while True:
            t_scan = time.time()
            files = scan_files([Path(r) for r in args.roots], excludes, order=args.order)
            all_chunks = []
            for fp in files:
                all_chunks.extend(chunker.chunk_file(fp))
            print(f"  scan: {len(files)} files -> {len(all_chunks)} chunks "
                  f"in {time.time() - t_scan:.0f}s", flush=True)
            if not all_chunks:
                print("No chunks to index", flush=True)
                return 0

            done = indexer.index_chunks(all_chunks, resume=args.resume,
                                        max_chunks=args.max_chunks,
                                        max_seconds=args.max_seconds)
            if done == 0:
                if indexer.last_failed:
                    print(f"  STOPPED: {indexer.last_failed} chunk(s) failed and none "
                          f"advanced -- check the embedding servers", flush=True)
                    return 1
                print("  index is complete for these roots", flush=True)
                return 0
            if args.loop_sleep <= 0:
                return 0
            print(f"  resting {args.loop_sleep:.0f}s, then the next slice", flush=True)
            time.sleep(args.loop_sleep)
    finally:
        indexer.stop_all()
        LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())