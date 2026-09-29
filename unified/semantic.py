"""
semantic.py  —  Unified semantic pipeline: chunk, embed, hybrid search, rerank.

Architecture:
  raw files → Chunker (512-char chunks) → Indexer (mxbai via llama-server HTTP)
  → index.jsonl (chunk + embedding + raw text) → Searcher (BM25 + vector hybrid
  + cross-encoder rerank)

Usage:
    python semantic.py index ~/llama-server/unified/*.py    # Index small set
    python semantic.py index --root ~/llama-server/unified  # Index directory
    python semantic.py search "how does executor.py work"   # Interactive search
    python semantic.py index --all                          # Full corpus (6797 files)
"""

import json, os, sys, time, re, argparse, textwrap, urllib.request, urllib.error
import subprocess
from pathlib import Path
from typing import Any

import corpus  # shared TEXT_EXTS / SKIP_DIRS + canonical stack doc

# ── Config ──────────────────────────────────────────────────────────────

from paths import SERVER_EXE as _RESOLVED_SERVER_EXE, OV_LIBS as _OV_LIBS_PATH

MODEL_PATH = os.environ.get(
    'EMBED_MODEL',
    str(Path.home() / 'llama-server' / 'models' / 'mxbai-embed-large-v1' / 'mxbai-embed-large-v1-Q4_k_m.gguf'),
)
SERVER_EXE = _RESOLVED_SERVER_EXE
OV_LIBS = str(_OV_LIBS_PATH)

PORT = 18083
EMBED_PORT = 18083      # llama-server embedding endpoint
EMBED_URL = f'http://127.0.0.1:{EMBED_PORT}/v1/embeddings'

INDEX_DIR = Path.home() / 'llama-server' / 'embeddings'
INDEX_FILE = INDEX_DIR / 'semantic_index.jsonl'
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBED_DIM = 1024          # mxbai-embed-large-v1 native dim

TEXT_EXTS = corpus.TEXT_EXTS
SKIP_DIRS = corpus.SKIP_DIRS


# ═══════════════════════════════════════════════════════════════════════
#  CHUNKER
# ═══════════════════════════════════════════════════════════════════════

class Chunker:
    """Split text files into 512-char chunks with 64-char overlap."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap

    def chunk_text(self, text: str, source_path: str) -> list[dict]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append({
                'path': source_path,
                'chunk_id': len(chunks),
                'offset': start,
                'length': len(chunk_text),
                'text': chunk_text,
            })
            if end >= len(text):
                break
            start += self.stride
        return chunks

    def chunk_file(self, file_path: Path) -> list[dict]:
        # Coerce: index_files() is documented as taking file paths, and callers
        # routinely pass str. Without this the except branch below raised
        # AttributeError on `.name`, which both hid the real read error AND
        # aborted the whole run instead of skipping one unreadable file.
        file_path = Path(file_path)
        try:
            text = file_path.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            print(f'  [SKIP] {file_path.name}: {e}')
            return []
        if not text.strip():
            return []
        return self.chunk_text(text, str(file_path))


# ═══════════════════════════════════════════════════════════════════════
#  INDEXER (llama-server HTTP batch)
# ═══════════════════════════════════════════════════════════════════════

class Indexer:
    """Embed chunks via llama-server HTTP API, store as JSONL."""

    def __init__(self, embed_url: str = EMBED_URL, batch_size: int = 200):
        self.embed_url = embed_url
        self.batch_size = batch_size
        self._server_proc = None

    def ensure_server(self) -> bool:
        if self._health_check():
            return True
        print('  Starting llama-server embedding endpoint...')
        env = dict(os.environ)
        env['GGML_OPENVINO_DISABLE'] = '1'
        env['GGML_OPENVINO_BLACKLIST'] = '1'
        ov = str(OV_LIBS)
        env['PATH'] = f'{ov};{env.get("PATH", "")}'

        self._server_proc = subprocess.Popen(
            [str(SERVER_EXE), '--model', MODEL_PATH,
             '--host', '127.0.0.1', '--port', str(PORT),
             '--ctx-size', '2048', '--threads', '16',
             '--batch-size', '2048', '--ubatch-size', '1024',
             '--no-warmup', '-ngl', '0',
             '--embedding', '--pooling', 'mean'],
            env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        deadline = time.time() + 120
        while time.time() < deadline:
            if self._health_check():
                return True
            time.sleep(2)
        return False

    def _health_check(self) -> bool:
        try:
            r = urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=3)
            return r.status == 200
        except Exception:
            return False

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({'input': texts, 'model': 'default'}).encode()
        req = urllib.request.Request(
            self.embed_url, data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        resp = urllib.request.urlopen(req, timeout=600)
        data = json.loads(resp.read())
        return [e['embedding'] for e in data['data']]

    def _existing_chunks(self, output: Path) -> set:
        """Which (path, offset) chunks are already in the index.

        Resume state is read from the index itself rather than a sidecar
        checkpoint. A checkpoint file is a second thing to corrupt, go stale,
        or lose; the index is the artefact that matters, it already carries
        `path` and `offset` for every chunk, and a truncated final line from a
        killed process is simply skipped.
        """
        seen = set()
        if not output.exists():
            return seen
        with output.open('r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue          # torn final line from a hard kill
                seen.add((rec.get('path'), rec.get('offset')))
        return seen

    def index_chunks(self, chunks: list[dict], output: Path = INDEX_FILE,
                     resume: bool = True) -> int:
        output.parent.mkdir(parents=True, exist_ok=True)

        # Without this, a re-run after a crash APPENDS duplicates for every
        # chunk already written -- output is opened in append mode per batch,
        # so a partial index silently becomes a corrupt one.
        done = self._existing_chunks(output) if resume else set()
        if done:
            before = len(chunks)
            chunks = [c for c in chunks
                      if (c.get('path'), c.get('offset')) not in done]
            print(f'  resume: {before - len(chunks)} chunks already indexed, '
                  f'{len(chunks)} to go', flush=True)
        if not chunks:
            return 0

        total = len(chunks)
        indexed = 0
        failed = []

        for batch_start in range(0, total, self.batch_size):
            batch = chunks[batch_start:batch_start + self.batch_size]
            texts = [c['text'] for c in batch]
            try:
                embeddings = self.embed_batch(texts)
            except Exception as e:
                # Was a bare `continue`: those chunks were skipped silently
                # and the run still reported success, so an index could look
                # complete while missing an arbitrary slice of the corpus.
                print(f'  Embedding batch FAILED at chunk {batch_start}: {e}')
                failed.append(batch_start)
                continue

            with open(output, 'a', encoding='utf-8') as f:
                for chunk, emb in zip(batch, embeddings):
                    record = {**chunk, 'embedding': emb, 'dim': EMBED_DIM}
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    indexed += 1

            elapsed = time.time() - self._t0
            rate = indexed / elapsed if elapsed > 0 else 0
            print(f'  [{indexed}/{total}] {rate:.1f} chunks/s', flush=True)

        if failed:
            print(f'  WARNING: {len(failed)} batch(es) failed at offsets '
                  f'{failed[:10]}{"..." if len(failed) > 10 else ""}. '
                  f'Those chunks are NOT in the index; re-run with --resume '
                  f'to fill them.', flush=True)
        return indexed

    def index_files(self, file_paths: list[Path], output: Path = INDEX_FILE) -> dict:
        if not self.ensure_server():
            return {'error': 'Server failed to start'}
        self._t0 = time.time()
        chunker = Chunker()
        all_chunks = []
        for fp in file_paths:
            all_chunks.extend(chunker.chunk_file(fp))
        print(f'  {len(file_paths)} files -> {len(all_chunks)} chunks')
        if not all_chunks:
            return {'files': len(file_paths), 'chunks': 0, 'indexed': 0}
        indexed = self.index_chunks(all_chunks, output=output)
        return {'files': len(file_paths), 'chunks': len(all_chunks), 'indexed': indexed}

    def shutdown(self):
        if self._server_proc:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None


# ═══════════════════════════════════════════════════════════════════════
#  SEARCHER (BM25 + vector hybrid + cross-encoder rerank)
# ═══════════════════════════════════════════════════════════════════════

class Searcher:
    """Hybrid search: BM25 + vector cosine similarity + cross-encoder rerank."""

    def __init__(self, index_path: Path = INDEX_FILE):
        self.index_path = index_path
        self.chunks: list[dict] = []
        self.embeddings: list[list[float]] = []
        self.bm25 = None
        self._tokenized_docs: list[list[str]] = []
        self._loaded = False

    def load(self) -> bool:
        if not self.index_path.exists():
            print(f'  Index not found: {self.index_path}')
            return False
        self.chunks = []
        for line in self.index_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                self.chunks.append(record)
                self.embeddings.append(record['embedding'])
        self._build_bm25()
        self._loaded = True
        print(f'  Loaded {len(self.chunks)} chunks from index')
        return True

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        self._tokenized_docs = [
            self._tokenize(c['text']) for c in self.chunks
        ]
        self.bm25 = BM25Okapi(self._tokenized_docs)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-10)

    def _vector_search(self, query_vec: list[float], top_k: int = 200) -> list[tuple[int, float]]:
        scores = [(i, self._cosine_sim(query_vec, self.embeddings[i]))
                  for i in range(len(self.chunks))]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _bm25_search(self, query: str, top_k: int = 200) -> list[tuple[int, float]]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        max_s = max(scores) if max(scores) > 0 else 1.0
        scored = [(i, s / max_s) for i, s in enumerate(scores)]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def _get_embedding(self, text: str) -> list[float]:
        import urllib.request
        body = json.dumps({'input': [text], 'model': 'default'}).encode()
        req = urllib.request.Request(
            EMBED_URL, data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        return data['data'][0]['embedding']

    def search(self, query: str, top_k: int = 10,
               alpha: float = 0.5, rerank: bool = True,
               embed_online: bool = False) -> list[dict]:
        if not self._loaded:
            self.load()
        if not self.chunks:
            return []

        # 1) BM25
        bm25_results = self._bm25_search(query, top_k=200)
        bm25_map = {i: s for i, s in bm25_results}

        # 2) Vector search
        if embed_online:
            query_vec = self._get_embedding(query)
        else:
            # Fallback: treat query as a pseudo-vector using avg of all embeddings
            # weighted by BM25 score. This is a crude approximation when no
            # online embedding is available.
            query_vec = [0.0] * EMBED_DIM
            for i, s in bm25_results[:50]:
                for d in range(EMBED_DIM):
                    query_vec[d] += s * self.embeddings[i][d]
            norm = sum(abs(v) for v in query_vec)
            if norm > 0:
                query_vec = [v / norm for v in query_vec]

        vec_results = self._vector_search(query_vec, top_k=200)
        vec_map = {i: s for i, s in vec_results}

        # 3) Hybrid fusion (reciprocal rank fusion)
        all_indices = set(i for i, _ in bm25_results) | set(i for i, _ in vec_results)
        hybrid = []
        for i in all_indices:
            bm25_score = bm25_map.get(i, 0.0)
            vec_score = vec_map.get(i, 0.0)
            combined = alpha * bm25_score + (1 - alpha) * vec_score
            hybrid.append((i, combined, bm25_score, vec_score))
        hybrid.sort(key=lambda x: -x[1])
        hybrid = hybrid[:top_k * 3]  # get more for reranking

        # 4) Cross-encoder rerank
        if rerank and hybrid:
            return self._rerank(query, hybrid, top_k)
        else:
            results = []
            for i, combined, bm25_s, vec_s in hybrid[:top_k]:
                c = self.chunks[i]
                results.append({
                    'path': c['path'],
                    'chunk_id': c['chunk_id'],
                    'text': c['text'][:300],
                    'score': round(combined, 4),
                    'score_bm25': round(bm25_s, 4),
                    'score_vec': round(vec_s, 4),
                })
            return results

    def _rerank(self, query: str, candidates: list[tuple], top_k: int = 10) -> list[dict]:
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            pairs = [(query, self.chunks[i]['text'][:512]) for i, *_ in candidates]
            scores = model.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
            results = []
            for (i, combined, bm25_s, vec_s), ce_score in ranked[:top_k]:
                c = self.chunks[i]
                results.append({
                    'path': c['path'],
                    'chunk_id': c['chunk_id'],
                    'text': c['text'][:300],
                    'score': round(float(ce_score), 4),
                    'score_bm25': round(bm25_s, 4),
                    'score_vec': round(vec_s, 4),
                })
            return results
        except Exception as e:
            print(f'  Reranker failed ({e}), falling back to hybrid scores')
            results = []
            for i, combined, bm25_s, vec_s in candidates[:top_k]:
                c = self.chunks[i]
                results.append({
                    'path': c['path'],
                    'chunk_id': c['chunk_id'],
                    'text': c['text'][:300],
                    'score': round(combined, 4),
                    'score_bm25': round(bm25_s, 4),
                    'score_vec': round(vec_s, 4),
                })
            return results


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def _scan_files(roots: list[Path]) -> list[Path]:
    files = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        if root.is_file():
            if root.suffix.lower() in TEXT_EXTS:
                files.append(root)
            continue
        if not root.exists():
            continue
        for dirpath, dirs, filenames in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in filenames:
                ext = Path(f).suffix.lower()
                if ext in TEXT_EXTS:
                    fp = Path(dirpath) / f
                    try:
                        if fp.stat().st_size > 0:
                            files.append(fp)
                    except OSError:
                        pass
    seen = set()
    unique = []
    for fp in files:
        real = str(fp)
        if real not in seen:
            seen.add(real)
            unique.append(fp)
    unique.sort(key=lambda f: f.stat().st_size)
    return unique


def cmd_index(args):
    if args.all:
        roots = [
            Path.home() / 'llama-server',
            Path.home() / '.lmstudio' / 'models',
            r'D:\knowledge_base',
            r'D:\llama-server-build',
            Path.home() / 'temp_llama_cpp_build' / 'llama.cpp',
            Path.home() / 'Desktop',
        ]
    elif args.root:
        roots = [Path(args.root)]
    else:
        roots = [Path(p) for p in args.files] if args.files else [Path.cwd()]

    files = _scan_files(roots)
    if args.max_files > 0:
        files = files[:args.max_files]

    if not files:
        print('No files found.')
        return

    print(f'Indexing {len(files)} files to {INDEX_FILE}')
    print(f'  Model: {MODEL_PATH}')
    print(f'  Chunk: {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap')
    print(f'  Dim:   {EMBED_DIM}')
    print(f'  Server: http://127.0.0.1:{PORT}')

    indexer = Indexer()
    try:
        result = indexer.index_files(files)
        if 'error' in result:
            print(f'ERROR: {result["error"]}')
        else:
            print(f'Done: {result["indexed"]} chunks from {result["files"]} files')
    finally:
        indexer.shutdown()


def cmd_search(args):
    if not args.query:
        while True:
            try:
                q = input('search> ').strip()
                if not q:
                    break
                args.query = q
                _do_search(args)
                args.query = None
            except (KeyboardInterrupt, EOFError):
                break
    else:
        _do_search(args)


def _do_search(args):
    searcher = Searcher()
    if not searcher.load():
        print('Run `python semantic.py index` first.')
        return
    results = searcher.search(
        args.query, top_k=args.top_k,
        alpha=args.alpha, rerank=not args.no_rerank,
    )
    print(f'\nTop {len(results)} results for: {args.query}\n')
    for r in results:
        print(f'  [{r["score"]:.4f}] {r["path"]}:{r["chunk_id"]}')
        print(f'       {r["text"][:200].replace(chr(10), " ")}')
        if r.get('score_bm25'):
            print(f'       BM25={r["score_bm25"]:.3f}  VEC={r["score_vec"]:.3f}')
        print()


def cmd_info(args):
    index = INDEX_FILE
    if not index.exists():
        print(f'No index at {index}')
        return
    lines = [l for l in index.read_text().splitlines() if l.strip()]
    chunks = [json.loads(l) for l in lines]
    paths = set(c['path'] for c in chunks)
    total_size = sum(len(json.dumps(c)) for c in chunks)
    print(f'Index:  {index}')
    print(f'Chunks: {len(chunks)}')
    print(f'Files:  {len(paths)}')
    print(f'Size:   {total_size / 1e6:.1f} MB')
    print(f'Dim:    {chunks[0]["dim"] if chunks else "?"}')
    print(f'Sample paths:')
    for p in sorted(paths)[:5]:
        count = sum(1 for c in chunks if c['path'] == p)
        print(f'  {p} ({count} chunks)')


def cmd_docs(args):
    """Print this file as documentation."""
    print(open(__file__).read())


def main():
    parser = argparse.ArgumentParser(
        prog='semantic',
        description='Unified semantic pipeline: chunk, embed, hybrid search, rerank',
    )
    sub = parser.add_subparsers(dest='cmd')

    # index
    p_idx = sub.add_parser('index', help='Index files into semantic search')
    p_idx.add_argument('files', nargs='*', help='Files or globs to index')
    p_idx.add_argument('--root', help='Directory to scan recursively')
    p_idx.add_argument('--all', action='store_true', help='Index full corpus (6797 files)')
    p_idx.add_argument('--max', type=int, default=0, dest='max_files', help='Max files')

    # search
    p_srch = sub.add_parser('search', help='Search indexed corpus')
    p_srch.add_argument('query', nargs='?', help='Search query')
    p_srch.add_argument('--top-k', type=int, default=10, help='Results to return')
    p_srch.add_argument('--alpha', type=float, default=0.5, help='BM25 weight (0-1) for hybrid')
    p_srch.add_argument('--no-rerank', action='store_true', help='Skip cross-encoder rerank')

    # info
    sub.add_parser('info', help='Show index statistics')

    # docs
    sub.add_parser('docs', help='Print full source as documentation')

    args = parser.parse_args()

    if args.cmd == 'index':
        cmd_index(args)
    elif args.cmd == 'search':
        cmd_search(args)
    elif args.cmd == 'info':
        cmd_info(args)
    elif args.cmd == 'docs':
        cmd_docs(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
