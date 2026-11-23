"""Run every codegraph/search test in one command and fail loudly if any fails.

Three files, three different costs: the graph tests are sub-second, and
test_search.py loads a real index and a cross-encoder reranker, which is slow
on a cold model cache and can wedge indefinitely -- measured 2026-11-03, it
ran past 240 s and took the calling shell with it. A runner that just calls
`for f in tests: run(f)` therefore reports nothing at all when that happens,
so each file runs as a bounded child: it gets its own process, its own wall
clock, and its own verdict. A hung file is NAMED as HUNG, never folded into a
pass and never allowed to eat the whole run.

Exit codes are the contract, and the third one is not a pass:
    0  every test file ran and returned 0
    1  at least one failed, hung, or exited non-zero
    2  the run could not be performed (no test files found, no interpreter)

Usage:
    python run_codegraph_tests.py            # all of them
    python run_codegraph_tests.py -v         # also show passing output
    python run_codegraph_tests.py --timeout 600
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Both patterns on purpose: the graph tests and the search test are the same
# gate, and a new test_*.py matching either name joins it without an edit.
PATTERNS = ("test_codegraph*.py", "test_search*.py")
DEFAULT_TIMEOUT_S = 300
TAIL_LINES = 40


def discover() -> list[Path]:
    found = {p.resolve() for pattern in PATTERNS for p in HERE.glob(pattern)}
    return sorted(found)


def run_one(path: Path, timeout_s: int) -> tuple[str, int, float, str]:
    """(status, exit_code, seconds, output). status is PASS/FAIL/HUNG."""
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(HERE),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        text = exc.stdout or ""
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        tail = exc.stderr or ""
        if isinstance(tail, bytes):
            tail = tail.decode("utf-8", "replace")
        return "HUNG", -1, elapsed, (text + tail)
    elapsed = time.monotonic() - started
    return ("PASS" if proc.returncode == 0 else "FAIL",
            proc.returncode, elapsed, (proc.stdout or "") + (proc.stderr or ""))


def tail(text: str, lines: int = TAIL_LINES) -> str:
    body = [ln for ln in text.splitlines() if ln.strip()]
    if len(body) > lines:
        cut = len(body) - lines
        body = ["... %d earlier line(s) not shown ..." % cut] + body[-lines:]
    return "\n".join("    " + ln for ln in body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show output from passing tests too")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                    help="per-test wall bound in seconds (default %d)" % DEFAULT_TIMEOUT_S)
    args = ap.parse_args()

    files = discover()
    if not files:
        print("could not run: no test files matching %s in %s"
              % (" / ".join(PATTERNS), HERE))
        return 2

    print("running %d test file(s), %ds wall bound each\n" % (len(files), args.timeout))
    bad = []
    for path in files:
        status, code, elapsed, output = run_one(path, args.timeout)
        summary = "%-4s %-28s %6.1fs" % (status, path.name, elapsed)
        if status == "PASS":
            summary += "  exit 0"
        elif status == "HUNG":
            summary += "  no exit within %ds" % args.timeout
        else:
            summary += "  exit %d" % code
        print(summary)
        if status != "PASS":
            bad.append((path.name, status, code))
            print(tail(output))
            print()
        elif args.verbose:
            print(tail(output))
            print()

    if bad:
        print("FAILED: " + ", ".join("%s (%s)" % (n, s) for n, s, _ in bad))
        # Hung is its own sentence: it is not a failing assertion, and fixing
        # it usually means a missing index or a cold model download, not a bug.
        for name, status, code in bad:
            if status == "HUNG":
                print("  %s never finished -- raise --timeout only after finding "
                      "out what it is waiting on" % name)
            elif code == 2:
                print("  %s could not run (exit 2) -- that is not a pass" % name)
        return 1

    print("\nOK: %d test file(s) passed" % len(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
