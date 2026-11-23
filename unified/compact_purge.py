#!/usr/bin/env python
# coding: utf-8
"""compact_purge -- remove records for paths the corpus no longer admits.

WHY THIS EXISTS
`parallel_embed --exclude` stops FUTURE writes. It does nothing about records
already on disk: on 2026-10-31 a run started before the corpus was trimmed wrote
archived shards, graph dumps and tokenizer JSON into the index, and every one of
those records keeps answering searches after the exclusion that would have
prevented it. Deduplication cannot fix that -- the records are unique, just
unwanted.

WHAT IT DOES NOT DO
It does not touch the old-location copy (`C:/Users/clint/azazael`). Those 71k
records are a stale tree, not junk: keeping them costs disk, deleting them costs
the only record of files that may no longer exist anywhere.

USAGE
  python compact_purge.py --dry-run     # counts only, writes nothing
  python compact_purge.py               # rewrite the index atomically
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import semantic  # noqa: E402

#: Path substrings whose records are build output, generated viewers, model
#: vocabularies or superseded shard copies -- measured, not guessed. Each entry
#: was 100+ chunks of non-knowledge in the 2026-10-31 index.
JUNK_PREFIXES = [
    "Azazael/azazael/archived/",
    "Azazael/azazael/archived_unified",
    "Azazael/azazael/skills.pkg-archive/",
    "Azazael/azazael/archive/",
    "Azazael/azazael/graphify-out/",
    "Azazael/azazael/src/graphify-out/",
    "Azazael/azazael/src/output/",
    "Azazael/azazael/state/",
    "Azazael/azazael/logs/",
    "Azazael/azazael/research/",
    "Azazael/azazael/nuitka_build_backup/",
    "llama-server/models/",              # tokenizer.json vocab, not knowledge
    "Desktop/AZAZAEL_COSMOS_VIEWER/",    # generated bundle + graph JSON
]


def scan(index: Path) -> dict:
    """Count what WOULD be dropped, without writing. Same rules as the purge."""
    drops = [p.lower() for p in JUNK_PREFIXES]
    by_prefix: dict[str, int] = {}
    total = keep = corrupt = 0
    with index.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                corrupt += 1
                continue
            low = str(rec.get("path") or "").replace("\\", "/").lower()
            hit = next((p for p in drops if p in low), None)
            if hit is not None:
                by_prefix[hit] = by_prefix.get(hit, 0) + 1
            else:
                keep += 1
    return {"records": total, "would_keep": keep,
            "would_drop": total - keep - corrupt, "corrupt": corrupt,
            "by_prefix": by_prefix}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", default=str(semantic.INDEX_FILE))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    idx = Path(args.index)
    if not idx.exists():
        print(f"no index at {idx}", file=sys.stderr)
        return 2

    if args.dry_run:
        r = scan(idx)
        print(f"records {r['records']:,}  keep {r['would_keep']:,}  "
              f"drop {r['would_drop']:,}  corrupt {r['corrupt']}")
        for k, v in sorted(r["by_prefix"].items(), key=lambda kv: -kv[1]):
            print(f"  {v:7,}  {k}")
        return 0

    stats = semantic.compact_index(idx, drop_prefixes=JUNK_PREFIXES)
    print(f"records {stats['records_before']:,} -> {stats['records_after']:,} "
          f"({stats['dropped_total']:,} purged, "
          f"{stats['duplicates_dropped']:,} duplicates, "
          f"{stats['corrupt_lines']:,} corrupt)")
    print(f"size {stats['size_before_mb']:.0f} MB -> {stats['size_after_mb']:.0f} MB")
    for k, v in sorted(stats["dropped_by_prefix"].items(), key=lambda kv: -kv[1]):
        print(f"  purged {v:7,}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
