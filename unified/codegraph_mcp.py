"""codegraph_mcp.py — expose azazael's code graph as native MCP tools.

WHY THIS EXISTS
---------------
The graphify skill documents an `--mcp` flag for agent access, but the
installed graphify 0.9.72 CLI ships no MCP surface at all — `mcp` appears
nowhere in `graphify --help`. So the graph is only reachable by shelling out
to the CLI, which means paying process start + a 44 MB JSON parse per
question and getting text back instead of tools.

This reads graph.json once, lazily, and exposes the same traversals as
tools: query, explain, path, affected, god-nodes, stats. Same answers, no
CLI round trip, and the caller gets structured fields instead of prose.

Transport is stdio, which is what Kilo launches as a local MCP server.

    python codegraph_mcp.py --graph C:\\path\\to\\graphify-out\\graph.json

Tools
-----
    codegraph_query(question, budget)  BFS from vocabulary-matched seeds
    codegraph_explain(node)            a node, its neighbours, where it lives
    codegraph_path(a, b)               shortest path between two nodes
    codegraph_affected(node, depth)    what a change to X would impact
    codegraph_god_nodes(top)           the most connected nodes
    codegraph_stats()                 graph shape and where the file is
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - surfaced as a clear message
    print("The 'mcp' package is required: pip install mcp", file=sys.stderr)
    raise

mcp = FastMCP("codegraph")

_GRAPH: dict[str, Any] | None = None
_GRAPH_PATH: Path | None = None
_NODES: dict[str, dict] = {}
_ADJ: dict[str, set[str]] = defaultdict(set)
_INDEX: dict[str, set[str]] = defaultdict(set)   # token -> node ids


def _default_graph() -> Path:
    import os
    env = os.environ.get("CODEGRAPH_GRAPH")
    if env:
        return Path(env)
    return Path.home() / "azazael" / "graphify-out" / "graph.json"


def _load() -> None:
    """Parse graph.json once. 44 MB, so it is cached for the process life."""
    global _GRAPH, _NODES, _ADJ, _INDEX
    if _GRAPH is not None:
        return
    path = _GRAPH_PATH or _default_graph()
    if not path.exists():
        raise FileNotFoundError(
            "graph.json not found at %s -- run `graphify <path>` first" % path)
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    nodes = data.get("nodes", [])
    edges = data.get("edges", data.get("links", []))
    for n in nodes:
        nid = n.get("id") or n.get("label")
        if not nid:
            continue
        _NODES[nid] = n
        for tok in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}",
                                  str(n.get("label", "")).lower())):
            _INDEX[tok].add(nid)
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s in _NODES and t in _NODES:
            _ADJ[s].add(t)
            _ADJ[t].add(s)
    _GRAPH = {"nodes": len(_NODES), "edges": len(edges), "path": str(path)}


def _nid(node: str) -> str | None:
    """Resolve a caller-supplied label to a node id, exactly or by case."""
    if node in _NODES:
        return node
    low = node.lower()
    for nid, n in _NODES.items():
        if str(n.get("label", "")).lower() == low:
            return nid
    return None


def _card(nid: str) -> str:
    n = _NODES[nid]
    bits = [str(n.get("label") or nid)]
    if n.get("source_location"):
        bits.append(str(n["source_location"]))
    elif n.get("source_file"):
        bits.append(str(n["source_file"]))
    if n.get("community_name"):
        bits.append("[%s]" % n["community_name"])
    return "  ".join(bits)


# ── tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def codegraph_stats() -> dict:
    """Shape of the loaded graph and where it came from."""
    _load()
    return {
        "nodes": _GRAPH["nodes"],
        "edges": _GRAPH["edges"],
        "graph_file": _GRAPH["path"],
        "connected_nodes": len(_ADJ),
    }


# Words that match thousands of prose nodes and carry no retrieval signal.
_STOP = {"the", "and", "for", "how", "what", "where", "when", "why", "does",
         "did", "are", "was", "were", "you", "your", "this", "that", "with",
         "from", "into", "use", "used", "using", "get", "set", "can", "work",
         "works", "there", "their", "them", "then", "than", "have", "has",
         "had", "but", "not", "all", "any", "its", "out", "off", "one"}


def _seed_rank(nid: str) -> tuple:
    """Prefer code identifiers over prose when seeding a query.

    Token matching alone ranks by frequency, and in this corpus that
    favours document headings: asking "how does the worm registry work"
    seeded on "the", "does" and "work" and returned sections of an
    unrelated markdown file, while the real answer -- worm_registry.py,
    degree 256 -- was never reached.

    Rank by the SHAPE of the label first, not the file it came from. A .py
    file also yields docstring nodes ("Start the alert notifier"), and
    keying on the extension promoted those prose lines above the actual
    identifier; label shape separates them.
    """
    n = _NODES.get(nid, {})
    label = str(n.get("label", "")).strip()
    src = str(n.get("source_file", "")).lower()
    is_code_file = src.endswith((".py", ".ts", ".js", ".rs", ".go", ".c", ".cpp"))
    identifier = bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\(\)?", label))
    code_path = bool(re.fullmatch(r"[\w./\\-]+\.(py|ts|js|rs|go|c|cpp)", label))
    doc_path = bool(re.fullmatch(r"[\w./\\-]+\.md", label))
    if identifier or code_path:
        return 0          # a symbol or a source file: the best possible seed
    if doc_path:
        return 1          # a document that mentions it
    if is_code_file:
        return 2          # a docstring or comment lifted out of source
    return 3              # prose from a document


@mcp.tool()
def codegraph_query(question: str, budget: int = 25) -> dict:
    """Answer a codebase question by traversing from vocabulary matches.

    Seeds are the graph's own node labels that appear in the question, so the
    starting point is grounded in what the graph actually contains rather than
    in a guess. Breadth-first from those seeds, one result per line. Seeds are
    ranked so code identifiers beat prose -- otherwise common words in
    markdown outrank the symbol you are actually asking about.
    """
    _load()
    words = {w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}",
                                           question)} - _STOP
    if not words:
        words = {w.lower() for w in re.findall(r"[A-Za-z_]{3,}", question)}
    seeds: set[str] = set()
    for w in words:
        seeds |= _INDEX.get(w, set())
    # Substring pass. The index is built from whole label tokens, so
    # "CORE/worm_registry.py" indexes as {core, worm_registry, py} and a
    # question saying "worm" matches none of them exactly -- the real
    # answer was unreachable until this was added.
    if len(words) < 12:
        for w in words:
            if len(w) < 4:
                continue
            for tok, ids in _INDEX.items():
                if w in tok:
                    seeds |= ids
    if not seeds:
        return {"seeds": [], "results": [],
                "note": "no graph label matched the question wording"}

    ordered = sorted(seeds, key=lambda s: (_seed_rank(s), str(s)))
    seen: set[str] = set()
    results: list[dict] = []
    queue = deque((s, 0) for s in ordered[:12])
    while queue and len(results) < budget:
        nid, depth = queue.popleft()
        if nid in seen or depth > 2:
            continue
        seen.add(nid)
        results.append({"depth": depth, "node": nid,
                        "label": _NODES[nid].get("label"),
                        "source": _NODES[nid].get("source_location")
                                  or _NODES[nid].get("source_file"),
                        "community": _NODES[nid].get("community_name")})
        for nb in sorted(_ADJ.get(nid, ()))[:8]:
            if nb not in seen:
                queue.append((nb, depth + 1))
    return {"seeds": [str(_NODES[s].get("label", s)) for s in ordered[:12]],
            "results": results}


@mcp.tool()
def codegraph_explain(node: str, neighbors: int = 10) -> dict:
    """What a node is, where it lives, and what it connects to."""
    _load()
    nid = _nid(node)
    if nid is None:
        return {"error": "no node matching %r" % node}
    n = _NODES[nid]
    return {
        "id": nid,
        "label": n.get("label"),
        "file": n.get("source_file"),
        "location": n.get("source_location"),
        "file_type": n.get("file_type"),
        "community": n.get("community_name"),
        "degree": len(_ADJ.get(nid, ())),
        "neighbors": [_card(x) for x in sorted(_ADJ.get(nid, ()))[:neighbors]],
    }


@mcp.tool()
def codegraph_path(a: str, b: str) -> dict:
    """Shortest path between two nodes -- what bridges them."""
    _load()
    src, dst = _nid(a), _nid(b)
    if src is None or dst is None:
        return {"error": "unresolved: %r / %r" % (
            a if src is None else "", b if dst is None else "")}
    if src == dst:
        return {"path": [_card(src)], "length": 0}
    prev: dict[str, str] = {src: ""}
    queue = deque([src])
    while queue:
        cur = queue.popleft()
        for nxt in _ADJ.get(cur, ()):
            if nxt in prev:
                continue
            prev[nxt] = cur
            if nxt == dst:
                path, node = [], dst
                while node:
                    path.append(_card(node))
                    node = prev[node]
                return {"path": list(reversed(path)), "length": len(path) - 1}
            queue.append(nxt)
    return {"path": [], "length": None,
            "note": "no path within the loaded graph"}


@mcp.tool()
def codegraph_affected(node: str, depth: int = 2) -> dict:
    """What a change to this node would impact -- reverse traversal."""
    _load()
    nid = _nid(node)
    if nid is None:
        return {"error": "no node matching %r" % node}
    seen = {nid}
    frontier = {nid}
    out: list[dict] = []
    for d in range(1, max(1, depth) + 1):
        nxt: set[str] = set()
        for x in frontier:
            nxt |= _ADJ.get(x, set())
        nxt -= seen
        for x in sorted(nxt):
            out.append({"depth": d, **_NODES[x],
                        "label": _NODES[x].get("label"),
                        "source": _NODES[x].get("source_location")
                                  or _NODES[x].get("source_file")})
        seen |= nxt
        frontier = nxt
    return {"origin": _card(nid), "impacted": out, "count": len(out)}


@mcp.tool()
def codegraph_god_nodes(top: int = 10) -> list[dict]:
    """The most connected nodes -- architectural hubs and churn centres."""
    _load()
    ranked = sorted(_NODES.items(), key=lambda kv: -len(_ADJ.get(kv[0], ())))
    return [{"label": n.get("label"), "degree": len(_ADJ.get(nid, ())),
             "file": n.get("source_file"),
             "community": n.get("community_name")}
            for nid, n in ranked[:max(1, top)]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=None,
                    help="path to graphify-out/graph.json")
    args = ap.parse_args()
    if args.graph:
        global _GRAPH_PATH
        _GRAPH_PATH = Path(args.graph)
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
