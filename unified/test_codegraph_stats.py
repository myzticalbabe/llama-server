"""Regression test: codegraph_stats must show when the graph predates HEAD.

A graph can be internally consistent, freshly reloaded, and still describe a
tree from a hundred commits ago -- every answer correct about the wrong commit,
with nothing in the reply saying so. codegraph_stats now compares the graph's
`built_at_commit` with the checkout's HEAD and says which of three things is
true: it matches, it is an older commit of the same repo (so `graphify update`
fixes it), or it is not a commit of that repo at all (a graph from a different
tree, where telling the caller to update would be wrong advice).

The four states that must never collapse into each other, and are therefore
each asserted here:
  True  -- matches HEAD, warning empty
  False -- differs, with the two cases distinguished by their message
  None  -- could not be measured, and the warning says which measurement failed
  (plus: a real graph must not be reported as matching by accident)

Every fixture repo is synthetic and disposable. The one assertion about the
real graph is a property (fields agree with each other), never a snapshot of
its commit, because HEAD moves and a test that fails when someone commits is a
test that gets deleted.

Usage:
    python test_codegraph_stats.py

Exit codes: 0 all checks passed, 1 a check failed, 2 could not run (no `mcp`).
"""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL_GRAPH = Path(r"C:\Azazael\azazael\graphify-out\graph.json")


def _import_codegraph():
    path = HERE / "codegraph_mcp.py"
    spec = importlib.util.spec_from_file_location("codegraph_mcp_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Checks:
    def __init__(self):
        self.failed = []
        self.skipped = []

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

    def skip(self, label, why):
        print("  SKIP " + label + " -- " + why)
        self.skipped.append(label)


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def _write_graph(path, nodes, commit):
    path.write_text(json.dumps({
        "nodes": [{"id": n, "label": n.upper()} for n in nodes],
        "edges": [],
        "built_at_commit": commit,
    }), encoding="utf-8")


def main() -> int:
    try:
        cg = _import_codegraph()
    except ImportError as exc:
        print("cannot run: %s" % exc)
        return 2

    def fresh(path):
        """Point the module at `path` with both caches dropped.

        Dropped explicitly rather than by sleeping: the 5 s TTL is deliberate,
        and a test that waits it out is a test that gets marked slow and then
        skipped.
        """
        cg._GRAPH_PATH = Path(path)
        cg._HEAD_CACHE = None
        cg._COMMIT_CACHE.clear()

    c = Checks()
    tmp = Path(tempfile.mkdtemp(prefix="codegraph-stats-"))
    saved = (cg._GRAPH_PATH, cg._git_head, cg._repo_root, cg._commit_in_repo)
    try:
        print("[1] a graph built at HEAD")
        repo = tmp / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "test@example.invalid")
        _git(repo, "config", "user.name", "test")
        _git(repo, "commit", "-q", "--allow-empty", "-m", "one")
        first = _git(repo, "rev-parse", "HEAD")
        gp = repo / "graph.json"
        _write_graph(gp, ["a", "b"], first)
        fresh(gp)
        s = cg.codegraph_stats()
        c.eq("matches HEAD", s["graph_matches_head"], True)
        c.eq("warning is empty", s["warning"], "")
        c.eq("head_commit", s["head_commit"], first)
        c.eq("repo_root", Path(s["repo_root"]).resolve(), repo.resolve())

        print("[2] one commit later, same graph: stale, and the advice applies")
        _git(repo, "commit", "-q", "--allow-empty", "-m", "two")
        second = _git(repo, "rev-parse", "HEAD")
        fresh(gp)
        s = cg.codegraph_stats()
        c.eq("differs from HEAD", s["graph_matches_head"], False)
        c.eq("head_commit is the new one", s["head_commit"], second)
        c.true("warning names the build commit", first[:12] in s["warning"], s["warning"])
        c.true("warning names HEAD", second[:12] in s["warning"], s["warning"])
        c.true("warning names the remedy", "graphify update" in s["warning"], s["warning"])

        print("[3] a graph from a different tree: differs, but not a rebuild")
        # tmp sits under C:/Users/clint, which is itself a git repo, so this
        # dir is inside a work tree that did not build this graph.
        elsewhere = tmp / "plain"
        elsewhere.mkdir()
        gp2 = elsewhere / "graph.json"
        _write_graph(gp2, ["a"], second)
        fresh(gp2)
        s = cg.codegraph_stats()
        c.eq("differs from HEAD", s["graph_matches_head"], False)
        c.true("warning says not a commit here", "not a commit of" in s["warning"], s["warning"])
        c.true("warning withholds the wrong remedy",
               "graphify update" not in s["warning"], s["warning"])

        print("[4] a graph with no build commit")
        _write_graph(gp, ["a"], "")
        fresh(gp)
        s = cg.codegraph_stats()
        c.eq("unmeasurable", s["graph_matches_head"], None)
        c.true("warning says why", "no built_at_commit" in s["warning"], s["warning"])

        print("[5] no git work tree above the graph")
        _write_graph(gp, ["a"], first)      # a stamp to compare, so the walk is what fails
        real_root = cg._repo_root
        cg._repo_root = lambda p: None
        try:
            fresh(gp)
            s = cg.codegraph_stats()
        finally:
            cg._repo_root = real_root
        c.eq("unmeasurable", s["graph_matches_head"], None)
        c.eq("no head claimed", s["head_commit"], None)
        c.true("warning says why", "no .git above" in s["warning"], s["warning"])
        c.eq("walk finds the repo holding a nested graph",
             cg._repo_root(gp), repo)

        print("[6] git present but unable to answer")
        _write_graph(gp, ["a"], first)
        real_head = cg._git_head
        cg._git_head = lambda root: None
        try:
            fresh(gp)
            s = cg.codegraph_stats()
        finally:
            cg._git_head = real_head
        c.eq("unmeasurable", s["graph_matches_head"], None)
        c.true("warning says why", "could not report HEAD" in s["warning"], s["warning"])

        print("[7] git is shelled out once, not per call")
        # Counted at subprocess.run, not at the wrapper: the wrapper is entered
        # on every call by design (it answers from cache), so counting it would
        # measure our own indirection instead of the spawn cost it exists for.
        real_run = subprocess.run
        spawned = []

        def counting_run(*a, **k):
            args = a[0] if a else k.get("args")
            if isinstance(args, (list, tuple)):
                if "rev-parse" in args:
                    spawned.append("rev-parse")
                elif "cat-file" in args:
                    spawned.append("cat-file")
            return real_run(*a, **k)

        subprocess.run = counting_run
        try:
            _write_graph(gp, ["a"], second)   # built at HEAD, so no containment check
            fresh(gp)
            for _ in range(5):
                cg.codegraph_stats()
            c.eq("rev-parse spawns over 5 calls", spawned.count("rev-parse"), 1)
            c.eq("cat-file spawns when it matches", spawned.count("cat-file"), 0)
            spawned.clear()
            fresh(gp2)                     # mismatch: one containment check, then cached
            for _ in range(5):
                cg.codegraph_stats()
            c.eq("rev-parse spawns on mismatch", spawned.count("rev-parse"), 1)
            c.eq("cat-file spawns over 5 mismatched calls", spawned.count("cat-file"), 1)
        finally:
            subprocess.run = real_run

        print("[8] the real graph, as a property not a snapshot")
        if REAL_GRAPH.exists():
            fresh(REAL_GRAPH)
            s = cg.codegraph_stats()
            c.eq("walk finds the azazael root", cg._repo_root(REAL_GRAPH),
                 Path(r"C:\Azazael\azazael"))
            c.true("git answered", isinstance(s["head_commit"], str)
                   and len(s["head_commit"]) == 40, repr(s["head_commit"]))
            c.eq("matches_head agrees with the two commits",
                 s["graph_matches_head"], s["built_at_commit"] == s["head_commit"])
            c.eq("no warning when they agree", s["warning"] == "",
                 s["graph_matches_head"])
            print("     built=%s head=%s" % (s["built_at_commit"][:12],
                                             (s["head_commit"] or "")[:12]))
        else:
            c.skip("real graph", "%s not on this machine" % REAL_GRAPH)
    finally:
        (cg._GRAPH_PATH, cg._git_head, cg._repo_root,
         cg._commit_in_repo) = saved
        shutil.rmtree(tmp, ignore_errors=True)

    for label in c.skipped:
        print("\nSKIPPED: " + label)
    if c.failed:
        print("\nFAILED %d check(s): %s" % (len(c.failed), ", ".join(c.failed)))
        return 1
    print("\nOK: staleness is reported in all four states and never asserted blind")
    return 0


def test_codegraph_stats_staleness() -> None:
    """pytest entry point; the direct-run path is `python test_codegraph_stats.py`."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
