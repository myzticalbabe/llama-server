"""Smoke test: search without server (BM25 pseudo-vector fallback) + with reranker.

Usage:
    python test_search.py [query]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from semantic import Searcher


def main(query: str = 'what does harvester.py do') -> int:
    s = Searcher()
    if not s.load():
        print('No index — run `python semantic.py index` first.')
        return 1
    print(f'Query: {query}\n')

    print('--- No reranker ---')
    results = s.search(query, top_k=5, rerank=False)
    assert isinstance(results, list), 'search must return a list'
    for r in results:
        p = r['path'].rsplit('\\', 1)[-1]
        t = r['text'][:120].replace('\n', ' ')
        print(f'  [{r["score"]:.4f}] {p}: {t}...')

    print()
    print('--- With cross-encoder reranker ---')
    for r in s.search(query, top_k=5, rerank=True):
        p = r['path'].rsplit('\\', 1)[-1]
        t = r['text'][:120].replace('\n', ' ')
        print(f'  [{r["score"]:.4f}] {p}: {t}...')
    print(f'\nOK: {len(results)} results (no-rerank branch)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else 'what does harvester.py do'))
