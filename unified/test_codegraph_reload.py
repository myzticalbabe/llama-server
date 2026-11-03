"""Regression test: codegraph_mcp must re-read graph.json when the file changes.

Pins the behaviour added in 1a4b484. Before it, `_load()` was a one-shot cache
(`if _GRAPH is not None: return`), so a `graphify update` under a running server
left every tool answering from the previous build with nothing in the reply to
say so -- measured 2026-11-03, four live servers served the pre-rebuild graph
after two rebuilds, and the only cure was restarting the client's stdio child.

What this covers:
  1. a rebuild on disk is picked up by the next call, with no restart
  2. the new build is what the traversals answer from, not just the counts
  3. an unchanged file causes no re-parse (the 94 MB parse is the whole reason
     the cache exists, so a fix that re-stat'd by re-reading would be a
     regression dressed as a fix)
  4. a rebuild caught mid-write keeps the last good index serving and records
     why, rather than taking the server down or answering from a partial graph
  5. the next whole write recovers and clears the recorded error
  6. a graph.json that disappears is survivable once something is loaded

No server is spawned: the tools are called directly, because the reload lives
in `_load()` and every tool goes through it. `test_codegraph_stdio.py` would be
a different test (transport).

Usage:
    python test_codegraph_reload.py

Exit codes: 0 all checks passed, 1 a check failed, 2 could not run (no `mcp`).
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _import_codegraph():
    """Load codegraph_mcp.py as a module without needing it on sys.path."""
    path = HERE / "codegraph_mcp.py"
    spec = importlib.util.spec_from_file_location("codegraph_mcp_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Checks:
    def __init__(self):
        self.failed = []

    def eq(self, label, got, want):
        ok = got == want
        print(("  PASS " if ok else "  FAIL ") + "%s: got=%r want=%r" % (label, got, want))
        if not ok:
            self.failed.append(label)
        return ok

    def true(self, label, value, detail=""):
        ok = bool(value)
        print(("  PASS " if ok else "  FAIL ") + label + ((" -- " + detail) if detail and not ok else ""))
        if not ok:
            self.failed.append(label)
        return ok


def _write(path, nodes, edges=(), commit="c1", bump_ms=1.0):
    """Write a graph.json and force a distinct stamp on coarse-mtime filesystems."""
    path.write_text(json.dumps({
        "nodes": [{"id": n, "label": n.upper(), "source_file": n + ".py",
                   "community_name": "comm"} for n in nodes],
        "edges": [{"source": s, "target": t, "relation": "calls"} for s, t in edges],
        "built_at_commit": commit,
    }), encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + int(bump_ms * 1e6)))


def main() -> int:
    try:
        cg = _import_codegraph()
    except ImportError as exc:
        print("cannot run: %s" % exc)
        return 2

    c = Checks()
    tmp = Path(tempfile.mkdtemp(prefix="codegraph-reload-"))
    gp = tmp / "graph.json"
    saved_path = cg._GRAPH_PATH
    saved_parse = cg._parse_graph
    try:
        cg._GRAPH_PATH = gp

        print("[1] cold load")
        _write(gp, ["a", "b"], [("a", "b")], commit="build-one")
        s = cg.codegraph_stats()
        c.eq("nodes", s["nodes"], 2)
        c.eq("edges", s["edges"], 1)
        c.eq("built_at_commit", s["built_at_commit"], "build-one")
        c.eq("reload_error", s["reload_error"], "")
        c.true("graph_mtime_ns reported", s["graph_mtime_ns"] is not None)

        print("[2] rebuild on disk is picked up without a restart")
        _write(gp, ["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")],
               commit="build-two", bump_ms=5)
        s = cg.codegraph_stats()
        c.eq("nodes", s["nodes"], 3)
        c.eq("edges", s["edges"], 3)
        c.eq("built_at_commit", s["built_at_commit"], "build-two")
        c.eq("connected_nodes", s["connected_nodes"], 3)
        c.eq("reload_error still clear", s["reload_error"], "")

        print("[3] traversals answer from the new build")
        c.true("new node resolvable", cg._nid("c") is not None)
        c.eq("community rebuilt", len(cg._COMMUNITIES["comm"]), 3)
        gods = cg.codegraph_god_nodes(3)
        c.true("god_nodes sees 3 nodes", len(gods) == 3, repr(gods))
        c.true("query works after reload", isinstance(cg.codegraph_query("alpha", 5), dict))
        c.true("path works after reload", isinstance(cg.codegraph_path("a", "c"), dict))

        print("[4] an unchanged file causes no re-parse")
        calls = {"n": 0}

        def counting(path):
            calls["n"] += 1
            return saved_parse(path)

        cg._parse_graph = counting
        try:
            for _ in range(3):
                cg.codegraph_stats()
            cg.codegraph_god_nodes(2)
        finally:
            cg._parse_graph = saved_parse
        c.eq("parses while unchanged", calls["n"], 0)

        print("[5] a rebuild caught mid-write keeps the last good index")
        gp.write_text('{"nodes": [{"id": "x"', encoding="utf-8")
        st = gp.stat()
        os.utime(gp, ns=(st.st_atime_ns, st.st_mtime_ns + 9_000_000))
        s = cg.codegraph_stats()
        c.eq("nodes unchanged", s["nodes"], 3)
        c.eq("commit unchanged", s["built_at_commit"], "build-two")
        c.true("failure recorded", bool(s["reload_error"]), repr(s["reload_error"]))
        c.true("error names the parse failure", "JSONDecodeError" in s["reload_error"],
               s["reload_error"])
        c.true("partial write never installed", cg._nid("x") is None)

        print("[6] the next whole write recovers and clears the error")
        _write(gp, ["a", "b", "c", "d"], [("a", "b")], commit="build-three", bump_ms=20)
        s = cg.codegraph_stats()
        c.eq("nodes", s["nodes"], 4)
        c.eq("built_at_commit", s["built_at_commit"], "build-three")
        c.eq("reload_error cleared", s["reload_error"], "")
        c.true("new node resolvable", cg._nid("d") is not None)

        print("[7] a vanished graph.json does not kill a loaded server")
        gp.unlink()
        s = cg.codegraph_stats()
        c.eq("nodes still served", s["nodes"], 4)
        c.true("loss recorded", bool(s["reload_error"]), repr(s["reload_error"]))
        _write(gp, ["a", "b"], [], commit="build-four", bump_ms=40)
        s = cg.codegraph_stats()
        c.eq("recovers when the file returns", s["nodes"], 2)
        c.eq("reload_error cleared again", s["reload_error"], "")
    finally:
        cg._GRAPH_PATH = saved_path
        cg._parse_graph = saved_parse
        shutil.rmtree(tmp, ignore_errors=True)

    if c.failed:
        print("\nFAILED %d check(s): %s" % (len(c.failed), ", ".join(c.failed)))
        return 1
    print("\nOK: reload, no-re-parse and corrupt-file behaviour all hold")
    return 0


def test_codegraph_reload() -> None:
    """pytest entry point; the direct-run path is `python test_codegraph_reload.py`."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
