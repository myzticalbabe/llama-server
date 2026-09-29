"""Build the code-scoped file list and index it with semantic.py's own pipeline.

Scope is deliberately NOT `semantic.py index --all`: the azazael tree contains
~5 GB of generated graph snapshots (var/cold/graph-snapshots/*.json, 932 MB
each) and other artifacts. Indexing everything is ~12.4M chunks / ~150 GB
against 64.5 GB free -- infeasible, and it would fill the disk.

Measured on a 40-file pilot: 6.4 chunks/s, 22.9 KB per chunk. This scope is
~116K chunks => ~5 hours, ~2.7 GB.

Reuses Chunker/Indexer from semantic.py so the chunks and embeddings are
identical to what `semantic.py index` would have produced; only the file
selection differs.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import semantic  # noqa: E402

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\clint\azazael")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else semantic.INDEX_FILE
# Extensions to index. Code is the default; docs are a second pass (".md"),
# sized separately because the .md corpus is dominated by generated dumps.
EXTS = {e.strip().lower() for e in
        (sys.argv[3] if len(sys.argv) > 3 else ".py").split(",") if e.strip()}
# Per-file ceiling. A 31.56 MB auto-generated lexicon (BIBL/INDEX.md, 421k
# entries) plus its backup copy was 80% of the .md corpus and ~1.5h of
# embedding for text nobody will ever search. Anything this large is a
# generated dump, not documentation.
MAX_BYTES = 2_000_000

# Directories that are generated output, vendored payloads, or archival --
# none of it is code a question would be asked about.
SKIP_DIRS = set(semantic.SKIP_DIRS) | {
    "var", "archive", "archived", "output", "logs", "state", "reports",
    "backups", "backup", "graphify-out", "tools", "docs", "dist", "build",
    "site-packages", "node_modules", ".git", "data", "BIBL",
    # build/venv caches that are not project code
    ".deps-jiter", ".nox", ".pytest_cache", ".mypy_cache", "htmlcov",
    ".ruff_cache", "__pypackages__", "eggs", ".eggs", ".claude", ".kilo",
}


def scan():
    files = []
    for dirpath, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for n in names:
            if Path(n).suffix.lower() not in EXTS:
                continue
            fp = Path(dirpath) / n
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            if 0 < size <= MAX_BYTES:
                files.append(str(fp))
    return sorted(set(files))


def main() -> int:
    files = scan()
    nbytes = sum(os.path.getsize(f) for f in files)
    print("scope  : %s" % ROOT, flush=True)
    print("exts   : %s" % ",".join(sorted(EXTS)), flush=True)
    print("files  : %d  (%.1f MB)" % (len(files), nbytes / 1e6), flush=True)
    print("output : %s" % OUT, flush=True)
    if not files:
        print("nothing to index")
        return 1

    indexer = semantic.Indexer()
    t0 = time.time()
    try:
        res = indexer.index_files(files, output=OUT)
    finally:
        indexer.shutdown()
    print("RESULT: %s" % res, flush=True)
    print("elapsed: %.0fs" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
