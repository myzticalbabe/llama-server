"""codegraph_mcp.py — expose azazael's code graph as native MCP tools.

WHY THIS EXISTS
---------------
The graphify skill documents an `--mcp` flag for agent access, but the
installed graphify 0.9.72 CLI ships no MCP surface at all — `mcp` appears
nowhere in `graphify --help`. So the graph is only reachable by shelling out
to the CLI, which means paying process start + a 44 MB JSON parse per
question and getting text back instead of tools.

This reads graph.json lazily, and re-reads it when the file's mtime or size
changes, so a `graphify update` no longer needs a server restart. It exposes
the same traversals as tools: query, explain, path, affected, god-nodes,
stats. Same answers, no CLI round trip, and the caller gets structured
fields instead of prose.

Transport is stdio, which is what Kilo launches as a local MCP server.

    python codegraph_mcp.py --graph C:\\path\\to\\graphify-out\\graph.json

Tools
-----
    codegraph_query(question, budget)  BFS from vocabulary-matched seeds
    codegraph_explain(node)            a node, its neighbours, where it lives
    codegraph_path(a, b, relations)    shortest path, code relations only
    codegraph_affected(node, depth, relations)
                                        what a change to X would impact
    codegraph_god_nodes(top)           the most connected nodes
    codegraph_stats()                 graph shape, provenance, and whether the
                                      build commit still matches the checkout's HEAD

`codegraph_path` and `codegraph_affected` are relation-typed and report the
relation that carried every hop. Passing relations=["*"] restores the old
relation-blind traversal, which answers a different question than the one the
tool name implies.

Resources
---------
    codegraph://manifest              index of everything this server exposes
    codegraph://stats                 shape, build commit, histograms
    codegraph://schema                the graph's own vocabulary -- which
                                     relation/kind/file_type values exist
    codegraph://communities           largest communities and their themes
    codegraph://hyperedges            multi-node groupings the graph asserts
    codegraph://node/{label}          full record + edges grouped by relation
    codegraph://file/{path}           per-file members and cross-file links
    codegraph://community/{name}      members of one detected community

Resources carry what the tools deliberately truncate: a tool answers a
question and returns a bounded answer, whereas a resource is a whole
artefact the caller can hold, diff and re-read. Every one of them reports
its own truncation rather than silently capping.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

try:  # mcp 2.x: FastMCP was renamed to MCPServer
    from mcp.server.mcpserver import MCPServer
    mcp = MCPServer("codegraph")
except ImportError:  # mcp 1.x fallback
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:  # pragma: no cover - surfaced as a clear message
        print("The 'mcp' package is required: pip install mcp", file=sys.stderr)
        raise
    mcp = FastMCP("codegraph")

_GRAPH: dict[str, Any] | None = None
_GRAPH_PATH: Path | None = None
# (st_mtime_ns, st_size) of the file the live index was parsed from. This is
# the freshness predicate: equal means serve the cache, different means parse
# again. mtime alone is too coarse (one second, FAT, clock skew); size alone
# misses an in-place edit of the same length.
_GRAPH_STAMP: tuple[int, int] | None = None
_GRAPH_ERROR: str = ""   # why the last reload was skipped, if it was
_LOAD_LOCK = threading.Lock()
_NODES: dict[str, dict] = {}
_ADJ: dict[str, set[str]] = defaultdict(set)
_INDEX: dict[str, set[str]] = defaultdict(set)   # token -> node ids

# Relation-typed, direction-preserving edges. _ADJ alone throws all of this
# away: it is an undirected set, so a path and a "what breaks if I change
# this" answer both run over containment and call edges in one undifferentiated
# pool, and neither says WHICH relation carried the hop.
_EDGES: dict[str, list[tuple]] = defaultdict(list)  # nid -> (other, dir, rel, at)
_FILES: dict[str, set[str]] = defaultdict(set)      # source_file -> node ids
_COMMUNITIES: dict[str, list[str]] = defaultdict(list)  # community_name -> ids
_HYPEREDGES: list[dict] = []
_COMMIT: str = ""


def _default_graph() -> Path:
    import os
    env = os.environ.get("CODEGRAPH_GRAPH")
    if env:
        return Path(env)
    return Path.home() / "azazael" / "graphify-out" / "graph.json"


# A `git rev-parse HEAD` costs ~15 ms here, and codegraph_stats is called often,
# so the answer is kept for a few seconds. The graph's own freshness is EXACT
# (mtime_ns + size, checked on every call); this one is a diagnostic, and being
# up to five seconds behind on "has someone committed" is not a wrong answer to
# a question whose whole point is "does the graph describe this checkout".
_HEAD_TTL_S = 5.0
_HEAD_CACHE: tuple[float, Path, str | None] | None = None


def _repo_root(graph_path: Path) -> Path | None:
    """The git work tree the graph file lives in, or None if there is none."""
    try:
        resolved = graph_path.resolve()
    except OSError:
        return None
    for parent in resolved.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _git_head(root: Path) -> str | None:
    """HEAD sha of `root`, or None when git cannot answer.

    A subprocess rather than reading .git by hand: HEAD can be a loose ref, a
    packed ref, or a `gitdir:` file inside a worktree, and re-deriving that
    resolution is how a staleness warning ends up confidently wrong. A failed
    lookup returns None and is reported as "not checked", never as "current".
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            # This runs under a client with no console; without the flag
            # Windows flashes a console window on every call.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _head_commit(root: Path | None) -> str | None:
    global _HEAD_CACHE
    if root is None:
        return None
    now = time.monotonic()
    if (_HEAD_CACHE is not None and _HEAD_CACHE[1] == root
            and now - _HEAD_CACHE[0] < _HEAD_TTL_S):
        return _HEAD_CACHE[2]
    sha = _git_head(root)
    _HEAD_CACHE = (now, root, sha)
    return sha


# Only consulted when the graph's stamp and HEAD already disagree, so it is one
# extra 15 ms call on the mismatch path, cached on the same TTL.
_COMMIT_CACHE: dict[tuple[Path, str], tuple[float, bool]] = {}


def _commit_in_repo(root: Path, sha: str) -> bool:
    """Is `sha` a commit of `root`? Distinguishes "behind" from "other tree".

    Without this, any graph sitting somewhere under any work tree gets told to
    run `graphify update` -- measured on the first pass: a temp graph under
    C:/Users/clint (its own repo) was compared against the home repo's HEAD and
    told to update a graph that repo had never built. "Differs" is the right
    observation there, but the remedy is different, so the two are separated.
    """
    now = time.monotonic()
    hit = _COMMIT_CACHE.get((root, sha))
    if hit is not None and now - hit[0] < _HEAD_TTL_S:
        return hit[1]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", sha + "^{commit}"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        found = out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        found = False
    if len(_COMMIT_CACHE) > 64:
        _COMMIT_CACHE.clear()
    _COMMIT_CACHE[(root, sha)] = (now, found)
    return found


def _staleness(path: str, built_commit: str) -> dict[str, Any]:
    """Compare the graph's build commit with the checkout's HEAD.

    The gap this closes: a graph can be internally consistent, freshly
    reloaded, and still describe a tree from a hundred commits ago. Every
    answer is then correct about the wrong commit, and nothing in the reply
    says so -- the caller has to already suspect it to check.

    `graph_matches_head` is equality, not ancestry: claiming "behind" would
    need a merge-base check, and a graph built from a newer commit than the
    checkout differs for the same reason and deserves the same sentence.
    """
    root = _repo_root(Path(path))
    head = _head_commit(root)
    if not built_commit:
        return {"head_commit": head, "repo_root": str(root) if root else None,
                "graph_matches_head": None,
                "warning": "graph.json carries no built_at_commit, so whether it "
                           "describes this checkout cannot be measured"}
    if root is None:
        return {"head_commit": None, "repo_root": None, "graph_matches_head": None,
                "warning": "no .git above %s, so graph staleness was not checked" % path}
    if head is None:
        return {"head_commit": None, "repo_root": str(root), "graph_matches_head": None,
                "warning": "git could not report HEAD for %s, so graph staleness "
                           "was not checked" % root}
    if head == built_commit:
        return {"head_commit": head, "repo_root": str(root),
                "graph_matches_head": True, "warning": ""}
    if not _commit_in_repo(root, built_commit):
        return {"head_commit": head, "repo_root": str(root),
                "graph_matches_head": False,
                "warning": "graph.json was built at %s, which is not a commit of %s "
                           "(HEAD %s) -- this graph describes a different tree, so "
                           "its answers may not match this checkout; pass the --graph "
                           "that belongs to it" % (
                               built_commit[:12], root, head[:12])}
    return {"head_commit": head, "repo_root": str(root), "graph_matches_head": False,
            "warning": "graph.json was built at %s but HEAD of %s is %s -- the graph "
                       "does not describe this checkout; run `graphify update` (or "
                       "pass the --graph that belongs to this tree)" % (
                           built_commit[:12], root.name or str(root), head[:12])}


def _file_stamp(path: Path) -> tuple[int, int] | None:
    """(mtime_ns, size), or None when the file cannot be stat'd right now."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _parse_graph(path: Path) -> dict[str, Any]:
    """Build a complete index from one graph.json.

    Parsed into local containers, never into the live globals: a reload that
    died halfway through would otherwise leave every question answered by a
    partial graph that still looks healthy. Build, then swap.
    """
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    nodes_raw = data.get("nodes", [])
    edges_raw = data.get("edges", data.get("links", []))
    nodes: dict[str, dict] = {}
    adj: dict[str, set[str]] = defaultdict(set)
    index: dict[str, set[str]] = defaultdict(set)
    edges: dict[str, list[tuple]] = defaultdict(list)
    files: dict[str, set[str]] = defaultdict(set)
    communities: dict[str, list[str]] = defaultdict(list)
    for n in nodes_raw:
        nid = n.get("id") or n.get("label")
        if not nid:
            continue
        nodes[nid] = n
        for tok in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}",
                                  str(n.get("label", "")).lower())):
            index[tok].add(nid)
        if n.get("source_file"):
            files[str(n["source_file"])].add(nid)
        if n.get("community_name"):
            communities[str(n["community_name"])].append(nid)
    for e in edges_raw:
        s, t = e.get("source"), e.get("target")
        if s in nodes and t in nodes:
            adj[s].add(t)
            adj[t].add(s)
            # The graph object exports with directed=False, but the stored
            # (source, target) tuple still carries semantic orientation:
            # verified across contains/calls/imports/inherits/rationale_for,
            # source is always the container / caller / importer / subclass /
            # rationale side. So these typed edges are usable as directed.
            rel = str(e.get("relation") or "")
            at = e.get("source_location")
            edges[s].append((t, "out", rel, at))
            edges[t].append((s, "in", rel, at))
    return {
        "meta": {"nodes": len(nodes), "edges": len(edges_raw), "path": str(path)},
        "nodes": nodes,
        "adj": adj,
        "index": index,
        "edges": edges,
        "files": files,
        "communities": communities,
        "hyperedges": list(data.get("graph", {}).get("hyperedges")
                           or data.get("hyperedges") or []),
        "commit": str(data.get("built_at_commit", "")),
    }


def _install(parsed: dict[str, Any]) -> None:
    """Make a parsed index live. Every assignment is a rebind, not a mutation."""
    global _GRAPH, _NODES, _ADJ, _INDEX, _EDGES, _FILES, _COMMUNITIES
    global _HYPEREDGES, _COMMIT
    _NODES = parsed["nodes"]
    _ADJ = parsed["adj"]
    _INDEX = parsed["index"]
    _EDGES = parsed["edges"]
    _FILES = parsed["files"]
    _COMMUNITIES = parsed["communities"]
    _HYPEREDGES = parsed["hyperedges"]
    _COMMIT = parsed["commit"]
    _GRAPH = parsed["meta"]


def _load() -> None:
    """(Re)parse graph.json when the file on disk has changed.

    44 MB, so the parsed index is still cached for the process life -- but the
    cache is now keyed on the file's stamp rather than on "have I ever
    loaded". A rebuild under a running server used to leave every answer
    describing a tree the file no longer contained, with nothing in the reply
    to say so; now the next call after the file changes answers from the new
    build.

    Two properties this keeps:

    * Unchanged stamp -> no re-parse. A 94 MB JSON parse per question is the
      cost this cache exists to avoid, so the check is a stat, not a read.
    * A failed reload never takes the server down. If a rebuild is caught
      mid-write, the previous index keeps serving and `_GRAPH_ERROR` says
      why; the stamp still disagrees, so the next call tries again.

    Readers do not take the lock: each container is internally consistent, so
    a request that races the swap can at worst mix ids from two builds in one
    answer, which is what it was already getting before the reload existed.
    """
    global _GRAPH_STAMP, _GRAPH_ERROR
    path = _GRAPH_PATH or _default_graph()
    stamp = _file_stamp(path)
    if stamp is None:
        if _GRAPH is None:
            raise FileNotFoundError(
                "graph.json not found at %s -- run `graphify <path>` first" % path)
        _GRAPH_ERROR = "graph.json unreadable at %s; still serving the copy loaded earlier" % path
        return
    if _GRAPH is not None and stamp == _GRAPH_STAMP:
        return
    with _LOAD_LOCK:
        # Re-stat under the lock: a concurrent caller may have just reloaded,
        # and a second parse of the same 94 MB file is pure waste.
        stamp = _file_stamp(path)
        if stamp is None:
            _GRAPH_ERROR = "graph.json unreadable at %s; still serving the copy loaded earlier" % path
            return
        if _GRAPH is not None and stamp == _GRAPH_STAMP:
            return
        try:
            parsed = _parse_graph(path)
        except (OSError, ValueError) as exc:
            if _GRAPH is None:
                raise
            _GRAPH_ERROR = "reload of %s failed (%s: %s); still serving the copy loaded at %s" % (
                path, type(exc).__name__, exc,
                _COMMIT or "an earlier build")
            return
        _install(parsed)
        # Only trust the stamp if the file did not move while we read it;
        # otherwise leave it unset so the next call parses the settled file.
        _GRAPH_STAMP = stamp if _file_stamp(path) == stamp else None
        _GRAPH_ERROR = ""



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
    """Shape of the loaded graph, where it came from, and whether it is current."""
    _load()
    return {
        "nodes": _GRAPH["nodes"],
        "edges": _GRAPH["edges"],
        "graph_file": _GRAPH["path"],
        "built_at_commit": _COMMIT,
        "graph_mtime_ns": _GRAPH_STAMP[0] if _GRAPH_STAMP else None,
        "connected_nodes": len(_ADJ),
        "reload_error": _GRAPH_ERROR,
        **_staleness(_GRAPH["path"], _COMMIT),
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


# ── relation sets ──────────────────────────────────────────────────────────
# The graph exports 16 relation types. Traversing all of them in one undirected
# pool yields answers that look like code structure and are not. Measured on
# this corpus: codegraph_path("FreeThreadedAgent", "BaseWormMixin") returned
# 5 hops, 4 of which were FILE nodes (FreeThreadedAgent.py -> logging.py ->
# CORE/worm_registry.py -> session_state.py) -- and there is NO `inherits`
# edge between those two nodes at all. The hop count read like a dependency
# chain; it was a walk through documents sharing the same soup. So the
# traversal tools are relation-typed by default and report which relation
# carried every hop.

# Relations meaning "this code actually uses that code".
_CODE_RELS = frozenset({
    "calls", "indirect_call", "imports", "imports_from", "inherits",
    "uses", "defines", "re_exports", "method", "shares_data_with",
    "rationale_for",
})

# The subset meaning "change the target and the source may break" -- what
# `affected` follows. Excludes `contains` (a function living inside a file is
# not a dependency on it) and every document relation.
_DEPENDENCY_RELS = frozenset({
    "calls", "indirect_call", "imports", "imports_from", "inherits",
    "uses", "method", "shares_data_with",
})


def _rel_filter(relations: list[str] | None, default: frozenset) -> set | None:
    """None means 'no filter' (every relation). Empty/None -> the default set."""
    if not relations:
        return set(default)
    if "*" in relations:
        return None
    return set(relations)


def _neighbours(nid: str, rels: set | None, direction: str | None = None,
                bidirectional: bool = False):
    """(other, relation, direction, at) for edges at `nid`, optionally filtered.

    `direction` is the STORED direction: "out" = nid is the edge source,
    "in" = nid is the edge target. The graph object exports undirected, but
    the (source, target) tuple keeps semantic orientation -- source is the
    container / caller / importer / subclass / rationale side.

    When bidirectional=True, both in-edges and out-edges are traversed
    regardless of the `direction` filter, enabling reverse dependency
    discovery (e.g., "what calls this function" from either side).

    Iterates a list, not a set, so traversal order -- and therefore which
    shortest path is returned -- is deterministic.
    """
    for other, d, rel, at in _EDGES.get(nid, ()):
        if rels is not None and rel not in rels:
            continue
        if direction is not None and d != direction and not bidirectional:
            continue
        yield other, rel, d, at


@mcp.tool()
def codegraph_path(a: str, b: str, relations: list[str] | None = None,
                   bidirectional: bool = False) -> dict:
    """Shortest path between two nodes over CODE relations only.

    Every hop reports the relation that carried it, so a real call chain is
    distinguishable from a walk through shared documents. Defaults to the
    code relation set; pass relations=["*"] to search every edge type, or
    e.g. relations=["inherits"] for exactly one.

    When bidirectional=True, traverses both in-edges and out-edges, enabling
    reverse dependency discovery (e.g., finding how A depends on B even when
    the graph only stores B->A edges).
    """
    _load()
    src, dst = _nid(a), _nid(b)
    if src is None or dst is None:
        return {"error": "unresolved: %r / %r" % (
            a if src is None else "", b if dst is None else "")}
    if src == dst:
        return {"length": 0, "hops": [], "nodes": [_card(src)],
                "note": "a and b resolve to the same node"}
    rels = _rel_filter(relations, _CODE_RELS)
    prev: dict[str, tuple] = {src: ("", "", "", None)}
    queue = deque([src])
    while queue:
        cur = queue.popleft()
        for other, rel, direction, at in _neighbours(cur, rels, bidirectional=bidirectional):
            if other in prev:
                continue
            prev[other] = (cur, rel, direction, at)
            if other == dst:
                hops, node = [], dst
                while node != src:
                    parent, r_, d_, at_ = prev[node]
                    hops.append({"relation": r_, "direction": d_, "at": at_,
                                 "from": parent, "to": node})
                    node = parent
                hops.reverse()
                return {
                    "length": len(hops),
                    "relations_used": sorted({h["relation"] for h in hops}),
                    "hops": [{**h, "from_label": _NODES[h["from"]].get("label"),
                              "to_label": _NODES[h["to"]].get("label"),
                              "from_file": _NODES[h["from"]].get("source_file"),
                              "to_file": _NODES[h["to"]].get("source_file")}
                             for h in hops],
                    "nodes": [_card(h["to"]) for h in hops],
                }
            queue.append(other)
    return {
        "length": None, "hops": [], "nodes": [],
        "relations_searched": sorted(rels) if rels else "all",
        "note": "no path under these relations. The nodes are in the same "
                "graph but not connected by a code edge. Retry with "
                'relations=["*"] to include containment and document edges, '
                "or relations=[] for the default code set.",
    }


@mcp.tool()
def codegraph_affected(node: str, depth: int = 2,
                       relations: list[str] | None = None) -> dict:
    """What a change to this node would impact, following DEPENDENCY edges.

    Only INCOMING dependency edges are followed: if `A calls B` then changing
    B may break A, so from B we cross to A. A relation-blind version walked
    containment in both directions and reported a file as "impacted" by its
    own contents, which is never true.
    """
    _load()
    nid = _nid(node)
    if nid is None:
        return {"error": "no node matching %r" % node}
    rels = _rel_filter(relations, _DEPENDENCY_RELS)
    seen = {nid}
    frontier = {nid}
    out: list[dict] = []
    for d in range(1, max(1, depth) + 1):
        found: dict[str, tuple] = {}
        for x in frontier:
            for other, rel, direction, at in _neighbours(x, rels, "in"):
                if other in seen or other in found:
                    continue
                found[other] = (rel, at, x)
        for x in sorted(found):
            rel, at, via = found[x]
            out.append({
                "depth": d,
                "id": x,
                "label": _NODES[x].get("label"),
                "file": _NODES[x].get("source_file"),
                "at": _NODES[x].get("source_location") or at,
                "depends_via": rel,
                "depends_on": _NODES[via].get("label"),
            })
        seen |= set(found)
        frontier = set(found)
    result = {
        "origin": _card(nid),
        "relations_followed": sorted(rels) if rels else "all",
        "impacted": out,
        "count": len(out),
    }
    if not out:
        result["note"] = (
            "nothing depends on this node under these relations. That is a "
            'real answer, not a failure -- but check it is a leaf you expect. '
            'Retry with relations=["*"] to include containment and document '
            "edges, which will produce hits that are not code dependencies.")
    return result


@mcp.tool()
def codegraph_god_nodes(top: int = 10) -> list[dict]:
    """The most connected nodes -- architectural hubs and churn centres."""
    _load()
    ranked = sorted(_NODES.items(), key=lambda kv: -len(_ADJ.get(kv[0], ())))
    return [{"label": n.get("label"), "degree": len(_ADJ.get(nid, ())),
             "file": n.get("source_file"),
             "community": n.get("community_name")}
            for nid, n in ranked[:max(1, top)]]


@mcp.tool()
def codegraph_export_3d(max_nodes: int = 500, community: str | None = None) -> dict:
    """Export graph data in 3D force-directed visualization format.

    Returns nodes (with val=degree for sizing, group=community for coloring)
    and links (with relation type) suitable for vasturiano/3d-force-graph.
    Optionally filter to a single community. Caps at max_nodes to keep
    the visualization responsive.
    """
    _load()
    if community:
        ids = [nid for nid, n in _NODES.items()
               if str(n.get("community_name", "")) == community]
    else:
        # Prioritize high-degree nodes for visualization
        ranked = sorted(_NODES.items(), key=lambda kv: -len(_ADJ.get(kv[0], ())))
        ids = [nid for nid, _ in ranked[:max_nodes]]

    id_set = set(ids)
    nodes = []
    for nid in ids:
        n = _NODES[nid]
        nodes.append({
            "id": nid,
            "label": str(n.get("label", nid))[:60],
            "val": len(_ADJ.get(nid, ())),
            "group": str(n.get("community_name", "unknown"))[:30],
            "file": str(n.get("source_file", ""))[:80],
            "type": str(n.get("file_type", "")),
        })

    links = []
    seen_links = set()
    for nid in ids:
        for other, d, rel, at in _EDGES.get(nid, ()):
            if other not in id_set:
                continue
            key = (nid, other, rel)
            if key in seen_links:
                continue
            seen_links.add(key)
            links.append({
                "source": nid,
                "target": other,
                "relation": rel,
                "direction": d,
            })

    return {
        "nodes": nodes,
        "links": links,
        "node_count": len(nodes),
        "link_count": len(links),
        "truncated": len(_NODES) > len(nodes),
        "communities": sorted(set(n["group"] for n in nodes)),
        "format": "3d-force-graph",
    }


# ── resources ─────────────────────────────────────────────────────────────
# Tools answer a question and return a bounded answer. Resources are the whole
# artefact: relation-typed instead of relation-blind, and complete enough to
# re-derive the tools' answers rather than restating one.
#
# A resource is pulled by exact URI, so anything capped below says so in the
# payload (`truncated`, `shown`, `total`). A silent cap is indistinguishable
# from a node that only ever had 20 edges.

_CAP = 200          # rows before a listing admits it stopped


def _capped(rows: list, cap: int = _CAP) -> dict:
    """Wrap a listing with honest truncation metadata."""
    return {"shown": len(rows[:cap]), "total": len(rows),
            "truncated": len(rows) > cap, "rows": rows[:cap]}


def _hist(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        k = v if v is not None else "(unset)"
        out[str(k)] = out.get(str(k), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _rel_group(nid: str) -> dict:
    """Every edge at `nid`, bucketed by relation and direction."""
    buckets: dict[str, list] = {}
    for other, direction, rel, at in _EDGES.get(nid, ()):
        buckets.setdefault("%s:%s" % (rel, direction), []).append({
            "node": other,
            "label": _NODES[other].get("label"),
            "file": _NODES[other].get("source_file"),
            "at": at,
        })
    return {k: _capped(v) for k, v in sorted(buckets.items())}


def _file_of(nid: str) -> str:
    return str(_NODES.get(nid, {}).get("source_file") or "")


@mcp.resource("codegraph://manifest", mime_type="application/json",
              name="manifest", title="Codegraph server index")
def manifest() -> str:
    """Everything this server exposes, and how the URIs are meant to be filled.

    A templated URI's `{param}` binds to `[^/]+` (see FastMCP
    ResourceTemplate.matches), so a template parameter can never contain a
    slash. `codegraph://file/{path}` therefore takes a percent-encoded path:
    `codegraph://file/algorithms%2Falgorithm_templates%2FSAFE_BATCH_PATTERN.py`.
    Node ids are already URL-safe and are the preferred key.
    """
    _load()
    return _dump({
        "graph_file": _GRAPH["path"],
        "built_at_commit": _COMMIT,
        "tools": {
            "codegraph_stats": "graph shape and provenance",
            "codegraph_query": "BFS from question vocabulary to labelled nodes",
            "codegraph_explain": "one node, bounded neighbour list",
            "codegraph_path": "shortest path between two nodes, code "
                              "relations only, per-hop relation reported",
            "codegraph_affected": "reverse traversal over dependency "
                                  "relations (in-edges only)",
            "codegraph_god_nodes": "highest-degree nodes",
        },
        "resources": {
            "codegraph://stats": "shape, build commit, top-level histograms",
            "codegraph://schema": "relation / node_kind / file_type vocabulary",
            "codegraph://communities": "largest detected communities",
            "codegraph://hyperedges": "asserted multi-node groupings",
            "codegraph://node/{id}": "full record, edges grouped by relation",
            "codegraph://file/{path}": "members + cross-file links (encode /)",
            "codegraph://community/{name}": "members of one community",
        },
        "notes": {
            "edge_direction": "exported undirected, but (source, target) keeps "
                              "semantic orientation; 'out' = as stored",
            "truncation": "every capped listing reports shown/total/truncated",
            "tool_vs_resource": "tools answer a question; resources return the "
                                "artefact. Prefer a resource when you need to "
                                "verify or re-derive a tool's answer.",
        },
    })


@mcp.resource("codegraph://stats", mime_type="application/json",
              name="stats", title="Graph shape and provenance")
def stats_resource() -> str:
    """Full shape plus the histograms the tools never expose."""
    _load()
    nodes = list(_NODES.values())
    return _dump({
        "graph_file": _GRAPH["path"],
        "built_at_commit": _COMMIT,
        "nodes": _GRAPH["nodes"],
        "edges": _GRAPH["edges"],
        "connected_nodes": len(_ADJ),
        "isolated_nodes": sum(1 for n in _NODES if not _ADJ.get(n)),
        "source_files": len(_FILES),
        "communities": len(_COMMUNITIES),
        "hyperedges": len(_HYPEREDGES),
        "file_type": _hist(n.get("file_type") for n in nodes),
        "node_kind": _hist(n.get("node_kind") for n in nodes),
        "origin": _hist(n.get("_origin") for n in nodes),
        "relation": _hist(rel for rels in _EDGES.values()
                          for _, _, rel, _ in rels),
    })


@mcp.resource("codegraph://schema", mime_type="application/json",
              name="schema", title="Graph vocabulary")
def schema_resource() -> str:
    """Which values the `relation`, `node_kind` and `file_type` fields take.

    The tools emit raw relation strings and `file_type`/`node_kind` is never
    exposed at all, so a caller has no way to discover the vocabulary without
    parsing graph.json itself.
    """
    _load()
    rels: dict[str, int] = {}
    for rels_of_node in _EDGES.values():
        for _, _, rel, _ in rels_of_node:
            rels[rel] = rels.get(rel, 0) + 1
    per_file: dict[str, int] = {}
    for rel, n in rels.items():
        per_file[rel] = n
    return _dump({
        "relation": {
            "counts": dict(sorted(per_file.items(), key=lambda kv: -kv[1])),
            "orientation": {
                "out": "target, as stored in the graph edge",
                "in": "source, as stored in the graph edge",
            },
            "note": "counts are doubled versus graph.json because each edge is "
                    "indexed at both endpoints",
        },
        "file_type": _hist(n.get("file_type") for n in _NODES.values()),
        "node_kind": _hist(n.get("node_kind") for n in _NODES.values()),
        "node_fields": _hist(k for n in _NODES.values() for k in n),
        "edge_fields": _hist(k for rows in _EDGES.values() for _, _, _, _ in rows
                             for k in ()),
    })


@mcp.resource("codegraph://communities", mime_type="application/json",
              name="communities", title="Largest communities")
def communities_resource() -> str:
    """The biggest detected communities and what files they cover."""
    _load()
    ranked = sorted(_COMMUNITIES.items(), key=lambda kv: -len(kv[1]))
    rows = []
    for name, ids in ranked[:_CAP]:
        files = sorted({_file_of(i) for i in ids if _file_of(i)})
        rows.append({"community": name, "members": len(ids),
                     "files": files[:10], "file_count": len(files)})
    # Not _capped(rows): that would report total=200/truncated=false and hide
    # the other 3175 communities. The total is the size of the index, not the
    # slice that was rendered.
    return _dump({"total_communities": len(_COMMUNITIES),
                  "shown": len(rows),
                  "truncated": len(ranked) > len(rows),
                  "rows": rows})


@mcp.resource("codegraph://hyperedges", mime_type="application/json",
              name="hyperedges", title="Asserted hyperedges")
def hyperedges_resource() -> str:
    """Multi-node groupings the graph asserts. Loaded but never exposed by any tool."""
    _load()
    return _dump(_capped([
        {"id": h.get("id"), "label": h.get("label"),
         "source_file": h.get("source_file"),
         "confidence": h.get("confidence"),
         "confidence_score": h.get("confidence_score"),
         "nodes": [_card(n) for n in h.get("nodes", []) if n in _NODES]}
        for h in _HYPEREDGES
    ]))


@mcp.resource("codegraph://node/{node}", mime_type="application/json",
              name="node", title="One node in full")
def node_resource(node: str) -> str:
    """Full node record plus every edge, bucketed by relation and direction.

    `codegraph_explain` returns ten neighbours with the relation thrown away,
    so a caller cannot tell a container edge from a call edge. This cannot.
    """
    _load()
    from urllib.parse import unquote
    nid = _nid(unquote(node))
    if nid is None:
        low = unquote(node).lower()
        hits = [n for n, d in _NODES.items()
                if str(d.get("label", "")).lower() == low]
        if len(hits) != 1:
            return _dump({"error": "no unique node matching %r" % node,
                          "candidates": [_card(h) for h in hits[:10]],
                          "hint": "pass the node id, or codegraph_explain"})
        nid = hits[0]
    n = _NODES[nid]
    return _dump({
        "id": nid,
        "record": n,
        "degree": len(_ADJ.get(nid, ())),
        "edge_counts_by_relation": _hist(rel for _, _, rel, _
                                          in _EDGES.get(nid, ())),
        "edges": _rel_group(nid),
        "community_peers": _capped(
            [_card(i) for i in _COMMUNITIES.get(str(n.get("community_name")), ())
             if i != nid]),
    })


@mcp.resource("codegraph://file/{path}", mime_type="application/json",
              name="file", title="One source file")
def file_resource(path: str) -> str:
    """Every node the graph extracted from one file, and what it links to.

    `path` is percent-encoded: template parameters bind to `[^/]+`, so
    `a/b/c.py` must arrive as `a%2Fb%2Fc.py`.
    """
    _load()
    from urllib.parse import unquote
    raw = unquote(path)
    ids = _FILES.get(raw)
    if ids is None:                       # case-insensitive retry
        low = raw.lower()
        for f, members in _FILES.items():
            if f.lower() == low:
                ids, raw = members, f
                break
    if ids is None:                       # last resort: basename
        base = raw.replace("\\", "/").rsplit("/", 1)[-1].lower()
        matches = sorted(f for f in _FILES if f.lower().endswith("/" + base)
                         or f.lower() == base)
        if len(matches) == 1:
            ids, raw = _FILES[matches[0]], matches[0]
        elif matches:
            return _dump({"error": "ambiguous file %r" % path,
                          "candidates": matches[:_CAP],
                          "candidate_count": len(matches)})
        else:
            return _dump({"error": "no file matching %r" % path,
                          "hint": "percent-encode '/' as %%2F",
                          "sample_files": sorted(_FILES)[:10]})
    out: dict[str, int] = {}
    inn: dict[str, int] = {}
    for nid in ids:
        for other, direction, _, _ in _EDGES.get(nid, ()):
            of = _file_of(other)
            if not of or of == raw:
                continue
            bucket = out if direction == "out" else inn
            bucket[of] = bucket.get(of, 0) + 1
    return _dump({
        "file": raw,
        "node_count": len(ids),
        "file_type": _hist(_NODES[i].get("file_type") for i in ids),
        "node_kind": _hist(_NODES[i].get("node_kind") for i in ids),
        "nodes": _capped([{"id": i, "label": _NODES[i].get("label"),
                           "kind": _NODES[i].get("node_kind"),
                           "at": _NODES[i].get("source_location"),
                           "callable": bool(_NODES[i].get("_callable"))}
                          for i in sorted(ids)]),
        "links_out": dict(sorted(out.items(), key=lambda kv: -kv[1])),
        "links_in": dict(sorted(inn.items(), key=lambda kv: -kv[1])),
    })


@mcp.resource("codegraph://community/{name}", mime_type="application/json",
              name="community", title="One community")
def community_resource(name: str) -> str:
    """The members of a single detected community, with what binds them."""
    _load()
    from urllib.parse import unquote
    key = unquote(name)
    ids = _COMMUNITIES.get(key)
    if ids is None:
        low = key.lower()
        for c, members in _COMMUNITIES.items():
            if c.lower() == low:
                ids, key = members, c
                break
    if ids is None:
        near = sorted(c for c in _COMMUNITIES if key.lower() in c.lower())[:10]
        return _dump({"error": "no community matching %r" % name,
                      "did_you_mean": near,
                      "hint": "see codegraph://communities"})
    rels: dict[str, int] = {}
    for nid in ids:
        for other, _, rel, _ in _EDGES.get(nid, ()):
            if other in ids:
                rels[rel] = rels.get(rel, 0) + 1
    return _dump({
        "community": key,
        "member_count": len(ids),
        "files": sorted({_file_of(i) for i in ids if _file_of(i)}),
        "internal_relations": dict(sorted(rels.items(), key=lambda kv: -kv[1])),
        "members": _capped([{"id": i, "label": _NODES[i].get("label"),
                             "kind": _NODES[i].get("node_kind"),
                             "file": _NODES[i].get("source_file"),
                             "at": _NODES[i].get("source_location"),
                             "degree": len(_ADJ.get(i, ()))}
                            for i in sorted(ids)]),
    })


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
