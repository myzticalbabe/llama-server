#!/usr/bin/env python
"""ov_embed_server.py — OpenAI-compatible /v1/embeddings on OpenVINO (GPU / NPU / CPU).

WHY THIS EXISTS
---------------
llama-server is a chat server that happens to embed. Measured 2026-10-31 on this
box (Core Ultra 9 285H, Arc 140T iGPU, AI Boost NPU), mxbai-embed-large-v1,
400 real chunks, mpnet-style 512-char chunks:

    OpenVINO IR fp16, iGPU  (this server)          62.1 chunks/s
    llama.cpp vulkan GGUF Q4_k_m, iGPU             26.7 chunks/s
    OpenVINO IR fp16, NPU                          12.4 chunks/s
    llama.cpp / OpenVINO, true CPU                  4.7 / 3.7 chunks/s

It speaks the same two endpoints parallel_embed.py already uses (/health and
POST /v1/embeddings), so it drops in where a llama-server embedding instance was.

FLAGS THAT SHOULD BE ON (each measured, see --help text and the sweep report):
    --device GPU             the iGPU beats the NPU 5.0x and the CPU 16.6x
    --perf-hint throughput   not LATENCY, which serialises requests
    --cache-dir <dir>        turns the 95 s NPU compile into a one-off cost
    --batch <IR optimum>     static batch from the IR; see OV's OPTIMAL_BATCH_SIZE
    --normalize              cosine-ready vectors (llama.cpp --embedding does the same)
    --pooling mean           mxbai is a mean-pooled sentence-transformer

    python ov_embed_server.py --model-dir C:/.../mxbai_embed_ir --device GPU --port 18100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import openvino as ov

STATE: dict = {}
_LOCK = threading.Lock()


def build(args) -> dict:
    model_dir = Path(args.model_dir)
    xml = model_dir / "openvino_model.xml"
    if not xml.is_file():
        raise SystemExit(f"no openvino_model.xml in {model_dir}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))

    core = ov.Core()
    cache_dir = Path(args.cache_dir) if args.cache_dir else model_dir / "ov_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # CACHE_DIR is a Core property; the rest are compile-time hints. String keys
    # are used because the typed ov.properties.* accessors changed shape between
    # 2024 and 2026 releases (hint.performance_mode() returns a key, not an enum).
    try:
        core.set_property({"CACHE_DIR": str(cache_dir)})
    except Exception as exc:
        print(f"  [warn] CACHE_DIR not settable: {exc}", flush=True)

    config: dict = {}
    if args.perf_hint != "none":
        config["PERFORMANCE_HINT"] = args.perf_hint.upper()
    if args.streams:
        config["NUM_STREAMS"] = str(args.streams)
    if args.inference_precision and args.inference_precision != "auto":
        config["INFERENCE_PRECISION_HINT"] = args.inference_precision

    t0 = time.time()
    compiled = core.compile_model(str(xml), args.device, config)
    compile_s = time.time() - t0

    st = {
        "model_dir": str(model_dir),
        "device": args.device,
        "batch": args.batch,
        "seq": args.seq,
        "pooling": args.pooling,
        "normalize": args.normalize,
        "perf_hint": args.perf_hint,
        "streams": args.streams,
        "compile_s": round(compile_s, 2),
        "tokenizer": tok,
        "compiled": compiled,
        "out": compiled.output(0),
        "cache_dir": str(cache_dir),
    }
    return st


def embed_texts(texts: list[str]) -> list[list[float]]:
    st = STATE
    tok = st["tokenizer"]
    compiled = st["compiled"]
    out = st["out"]
    B, S = st["batch"], st["seq"]
    vecs: list[np.ndarray] = []
    for i in range(0, len(texts), B):
        chunk = list(texts[i:i + B])
        n = len(chunk)
        # The IR has a STATIC batch shape, so a short final batch must be padded
        # to B or the GPU plugin rejects the request outright. Padding is masked
        # out below, then the extra vectors are dropped.
        if n < B:
            chunk += [chunk[-1]] * (B - n)
        enc = tok(chunk, padding="max_length", max_length=S, truncation=True,
                  return_tensors="np")
        res = compiled({"input_ids": enc["input_ids"].astype(np.int64),
                        "attention_mask": enc["attention_mask"].astype(np.int64)})[out]
        mask_np = enc["attention_mask"]
        if st["pooling"] == "mean":
            mask = mask_np[..., None].astype(np.float32)
            v = (res * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
        else:
            v = res[:, 0, :]
        if st["normalize"]:
            v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
        vecs.append(np.asarray(v[:n], dtype=np.float32))
    return np.concatenate(vecs).tolist() if vecs else []


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # keep the console quiet
        pass

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            self._send(200, {"status": "ok", "device": STATE["device"],
                             "batch": STATE["batch"], "seq": STATE["seq"],
                             "pooling": STATE["pooling"],
                             "normalize": STATE["normalize"],
                             "dim": None if not STATE.get("dim") else STATE["dim"]})
        elif self.path.startswith("/v1/models"):
            self._send(200, {"object": "list", "data": [
                {"id": Path(STATE["model_dir"]).name, "object": "model"}]})
        elif self.path.startswith("/props"):
            self._send(200, {k: v for k, v in STATE.items()
                             if isinstance(v, (str, int, float, bool))})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/embeddings"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:
            self._send(400, {"error": f"bad json: {exc}"})
            return
        inp = req.get("input")
        if isinstance(inp, str):
            texts = [inp]
        elif isinstance(inp, list) and all(isinstance(t, str) for t in inp):
            texts = inp
        else:
            self._send(400, {"error": "input must be a string or list of strings"})
            return
        t0 = time.time()
        with _LOCK:
            try:
                vecs = embed_texts(texts)
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
        if vecs:
            STATE.setdefault("dim", len(vecs[0]))
        self._send(200, {
            "object": "list",
            "model": Path(STATE["model_dir"]).name,
            "data": [{"object": "embedding", "index": i, "embedding": v}
                     for i, v in enumerate(vecs)],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
            "elapsed_s": round(time.time() - t0, 4),
        })


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenVINO embedding server (OpenAI-compatible)")
    ap.add_argument("--model-dir", required=True, help="IR dir (openvino_model.xml + tokenizer)")
    ap.add_argument("--device", default="GPU", help="GPU | NPU | CPU (default GPU)")
    ap.add_argument("--port", type=int, default=18100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--batch", type=int, default=8, help="static batch baked into the IR")
    ap.add_argument("--seq", type=int, default=224, help="static sequence length baked into the IR")
    ap.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    ap.add_argument("--normalize", dest="normalize", action="store_true", default=True)
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.add_argument("--perf-hint", choices=["throughput", "latency", "none"], default="throughput")
    ap.add_argument("--streams", type=int, default=0, help="0 = device default")
    ap.add_argument("--inference-precision", choices=["auto", "f16", "f32"], default="auto")
    ap.add_argument("--cache-dir", default=None, help="OV model cache (default <model-dir>/ov_cache)")
    ap.add_argument("--warmup", type=int, default=1, help="warmup inferences before serving")
    args = ap.parse_args()

    print(f"compiling {args.device} b={args.batch} seq={args.seq} hint={args.perf_hint} ...", flush=True)
    STATE.update(build(args))
    print(f"ready on {args.device} (compile {STATE['compile_s']}s, cache {STATE['cache_dir']})", flush=True)

    for _ in range(max(0, args.warmup)):
        embed_texts(["warmup"])

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
