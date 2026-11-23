#!/usr/bin/env python
"""ov_flag_sweep.py — decide, by measurement, which OpenVINO embedding flags should be on.

Sweeps static (batch, seq) on the iGPU, then perf-hint / streams / device controls,
and prints a table. Each config spawns ov_embed_server.py on its own port, measures
the same 400 real chunks over HTTP (200 per request, exactly like the index job),
and is torn down afterwards.

    python ov_flag_sweep.py                    # GPU sweep + NPU check
    python ov_flag_sweep.py --device GPU --quick
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HF = Path(r"C:/Users/clint/llama-server/models/ov_export/mxbai_embed_hf")
IR_BASE = Path(r"C:/Users/clint/llama-server/models/ov_export")
SWEEP = IR_BASE / "sweep"

_torch = None
_model = None
_tok = None


def _load_torch_model():
    global _torch, _model, _tok
    if _model is not None:
        return _torch, _model, _tok
    import torch
    from transformers import AutoModel, AutoTokenizer
    _torch = torch
    _tok = AutoTokenizer.from_pretrained(str(HF))
    _model = AutoModel.from_pretrained(str(HF)).eval()
    return _torch, _model, _tok


def export_ir(batch: int, seq: int) -> Path:
    """Write a static-shape IR for (batch, seq) and return its directory."""
    torch, model, _ = _load_torch_model()
    d = SWEEP / f"b{batch}_s{seq}"
    xml = d / "openvino_model.xml"
    if xml.is_file() and (d / "shape.json").is_file():
        if json.loads((d / "shape.json").read_text()).get("shape") == [batch, seq]:
            return d
    import openvino as ov
    d.mkdir(parents=True, exist_ok=True)
    ex = {"input_ids": torch.ones((batch, seq), dtype=torch.long),
          "attention_mask": torch.ones((batch, seq), dtype=torch.long)}
    t0 = time.time()
    with torch.no_grad():
        m = ov.convert_model(model, example_input=ex,
                             input=[("input_ids", [batch, seq]),
                                    ("attention_mask", [batch, seq])])
    ov.save_model(m, str(xml))
    (d / "shape.json").write_text(json.dumps({"shape": [batch, seq]}))
    _ = _tok.save_pretrained(str(d))
    print(f"    exported [{batch},{seq}] in {time.time()-t0:.0f}s", flush=True)
    return d


def serve(model_dir: Path, device: str, port: int, batch: int, seq: int, extra=()):
    cmd = [sys.executable, str(HERE / "ov_embed_server.py"),
           "--model-dir", str(model_dir), "--device", device,
           "--port", str(port), "--batch", str(batch), "--seq", str(seq), *extra]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    while time.time() - t0 < 300:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200:
                return p, time.time() - t0
        except Exception:
            if p.poll() is not None:
                out = p.stdout.read() if p.stdout else ""
                raise RuntimeError(f"server died: {out[-800:]}")
            time.sleep(1)
    p.kill()
    raise RuntimeError("server never became healthy")


def measure(port: int, texts: list[str], per_request: int = 200) -> tuple[int, float]:
    n = 0
    t0 = time.time()
    for i in range(0, len(texts), per_request):
        body = json.dumps({"input": texts[i:i + per_request]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        n += len(json.loads(urllib.request.urlopen(req, timeout=600).read())["data"])
    return n, time.time() - t0


def run_config(name, model_dir, device, port, batch, seq, extra=()):
    try:
        p, ready_s = serve(model_dir, device, port, batch, seq, extra)
    except Exception as exc:
        print(f"  {name:42s} FAILED: {str(exc)[:160]}", flush=True)
        return None
    try:
        texts = load_texts()
        measure(port, texts[:16])                      # warmup
        n, el = measure(port, texts)
        rate = n / el
        print(f"  {name:42s} {rate:6.2f} chunks/s   (ready {ready_s:5.1f}s)", flush=True)
        return rate
    finally:
        p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()


_TEXT_CACHE: list[str] | None = None


def load_texts() -> list[str]:
    global _TEXT_CACHE
    if _TEXT_CACHE is not None:
        return _TEXT_CACHE
    import semantic
    from pathlib import Path as P

    def junk(p):
        s = p.as_posix().lower()
        return ("/build-spark/" in s or "/build/" in s or "graphify_codebase_atlas" in s
                or s.endswith("/graph.json") or s.endswith("/graph.html")
                or s.endswith("_full.csv") or s.endswith(".log"))

    files = [f for f in semantic._scan_files([P.home() / "llama-server", P.home() / "Desktop"])
             if not junk(f)]
    files = [f for f in files if 20000 < f.stat().st_size < 400000][:80]
    ch = semantic.Chunker()
    texts: list[str] = []
    for fp in files:
        texts.extend(c["text"] for c in ch.chunk_file(fp))
    _TEXT_CACHE = texts[:400]
    return _TEXT_CACHE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    _, _, tok = _load_torch_model()
    texts = load_texts()
    lens = sorted(len(tok(t)["input_ids"]) for t in texts)
    p50, p95 = lens[len(lens) // 2], lens[int(len(lens) * 0.95)]
    s50 = max(32, int(math.ceil(p50 / 32) * 32))
    s95 = min(512, int(math.ceil(p95 / 32) * 32))
    print(f"chunks={len(texts)} tokens p50={p50} p95={p95} -> seq candidates {s50} / {s95}\n", flush=True)

    shapes = [(8, s95), (16, s95), (32, s95), (8, s50), (16, s50)]
    if args.quick:
        shapes = [(8, s95)]

    print("== static shape sweep on", args.device, flush=True)
    best = (None, -1, None)
    dirs = {}
    for i, (b, s) in enumerate(shapes):
        d = export_ir(b, s)
        dirs[(b, s)] = d
        r = run_config(f"batch {b}  seq {s}", d, args.device, 18140 + i, b, s)
        if r and r > best[1]:
            best = ((b, s), r, d)

    if best[0] is None:
        print("no config served; stopping", flush=True)
        return 1
    (bb, bs), brate, bdir = best
    print(f"\nbest shape: batch {bb} seq {bs} at {brate:.2f} chunks/s\n", flush=True)

    print("== flag controls at the best shape", flush=True)
    controls = [
        ("perf-hint latency (control)", ("--perf-hint", "latency")),
        ("perf-hint none (control)", ("--perf-hint", "none")),
        ("streams 2 (control)", ("--streams", "2")),
        ("streams 4 (control)", ("--streams", "4")),
    ]
    for i, (name, extra) in enumerate(controls):
        run_config(name, bdir, args.device, 18155 + i, bb, bs, extra)

    print("\n== disabled file cache (control: what --cache-dir buys)", flush=True)
    tmpcache = Path(r"C:/Users/clint/llama-server/models/ov_export/sweep/nocache")
    if tmpcache.exists():
        shutil.rmtree(tmpcache)
    run_config("fresh cache dir (cold compile)", bdir, args.device, 18160, bb, bs,
               ("--cache-dir", str(tmpcache)))
    run_config("same cache dir again (warm)", bdir, args.device, 18161, bb, bs,
               ("--cache-dir", str(tmpcache)))

    print("\n== device comparison at the best shape", flush=True)
    for i, dev in enumerate([d for d in ("GPU", "NPU", "CPU") if d != args.device]):
        run_config(f"device {dev}", bdir, dev, 18165 + i, bb, bs)

    print(f"\nrecommended: --batch {bb} --seq {bs} --device {args.device} "
          f"--perf-hint throughput --cache-dir <model-dir>/ov_cache --normalize --pooling mean",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
