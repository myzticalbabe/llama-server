#!/usr/bin/env python
"""ov_flag_repeat.py — repeat each flag config enough times that run-to-run noise
cannot be mistaken for an effect.

Measured 2026-10-31: identical configs varied 59.5 -> 73.7 chunks/s (24%), so a
single 400-chunk run per flag is not evidence. This runs each config N times and
prints mean / min / max.

    python ov_flag_repeat.py --repeats 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ov_flag_sweep import export_ir, load_texts, measure, serve  # noqa: E402

CONFIGS = [
    # (label, batch, seq, device, extra flags)
    ("b16 s288 hint=throughput", 16, 288, "GPU", ()),
    ("b16 s288 hint=none", 16, 288, "GPU", ("--perf-hint", "none")),
    ("b16 s288 hint=throughput streams=2", 16, 288, "GPU", ("--streams", "2")),
    ("b8  s288 hint=throughput", 8, 288, "GPU", ()),
    ("b16 s256 hint=throughput", 16, 256, "GPU", ()),
]


def run_one(label, batch, seq, device, extra, port):
    d = export_ir(batch, seq)
    p, ready = serve(d, device, port, batch, seq, extra)
    try:
        texts = load_texts()
        measure(port, texts[:16])
        n, el = measure(port, texts)
        return n / el, ready
    finally:
        p.terminate()
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    print(f"{'config':40s} {'mean':>7s} {'min':>7s} {'max':>7s}  runs", flush=True)
    rows = []
    for i, (label, b, s, dev, extra) in enumerate(CONFIGS):
        rates = []
        for r in range(args.repeats):
            try:
                rate, _ = run_one(label, b, s, dev, extra, 18180 + i * 10 + r)
                rates.append(rate)
            except Exception as exc:
                print(f"  {label}: run {r} failed: {str(exc)[:120]}", flush=True)
        if rates:
            rows.append((label, sum(rates) / len(rates), rates))
            print(f"{label:40s} {sum(rates)/len(rates):7.2f} {min(rates):7.2f} "
                  f"{max(rates):7.2f}  {[round(x, 1) for x in rates]}", flush=True)
    if rows:
        rows.sort(key=lambda r: -r[1])
        print(f"\nbest: {rows[0][0]} at {rows[0][1]:.2f} chunks/s "
              f"(spread {min(rows[0][2]):.1f}-{max(rows[0][2]):.1f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
