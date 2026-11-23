"""A/B the embedding workload across installed llama-server builds.
Same model, same args, same 400 real chunks; only the binary and -ngl change.
"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

MODEL = r"C:/Users/clint/llama-server/models/mxbai-embed-large-v1/mxbai-embed-large-v1-Q4_k_m.gguf"
EXE = Path(sys.argv[1]); NGL = sys.argv[2]; PORT = int(sys.argv[3]); ARM = sys.argv[4]
BASE = f"http://127.0.0.1:{PORT}"
LOG = Path(rf"C:/Users/clint/llama-server/embeddings/xpu_{ARM}.stderr.log")

sys.path.insert(0, r"C:/Users/clint/llama-server/unified")
import semantic
from pathlib import Path as P

def junk(p):
    s = p.as_posix().lower()
    return ("/build-spark/" in s or "/build/" in s or "graphify_codebase_atlas" in s
            or s.endswith("/graph.json") or s.endswith("/graph.html")
            or s.endswith("_full.csv") or s.endswith(".log"))

ROOTS = [P.home()/"llama-server", P(r"D:/knowledge_base"), P(r"D:/llama-server-build"),
         P.home()/"temp_llama_cpp_build"/"llama.cpp", P.home()/"Desktop"]
files = [f for f in semantic._scan_files(ROOTS) if not junk(f)]
files = [f for f in files if 20000 < f.stat().st_size < 400000][:120]
ch = semantic.Chunker()
texts = []
for fp in files:
    texts.extend(c["text"] for c in ch.chunk_file(fp))
    if len(texts) >= 400:
        break
texts = texts[:400]

args = [str(EXE), "--model", MODEL, "--host", "127.0.0.1", "--port", str(PORT),
        "--ctx-size", "512", "--threads", "4", "--batch-size", "512",
        "--ubatch-size", "512", "--no-warmup", "-ngl", NGL,
        "--embedding", "--pooling", "mean"]
env = dict(os.environ)
env["GGML_OPENVINO_DISABLE"] = "1"
env["GGML_OPENVINO_BLACKLIST"] = "1"
LOG.parent.mkdir(parents=True, exist_ok=True)
logf = open(LOG, "w", encoding="utf-8", errors="replace")
proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
try:
    t0 = time.time()
    up = False
    while time.time() - t0 < 180:
        try:
            if urllib.request.urlopen(f"{BASE}/health", timeout=3).status == 200:
                up = True; break
        except Exception:
            time.sleep(1)
    if not up:
        print(f"{ARM}: FAILED to come up ({EXE.name}, ngl={NGL})"); sys.exit(1)
    boot = time.time() - t0

    def embed(batch):
        body = json.dumps({"input": batch, "model": "default"}).encode()
        req = urllib.request.Request(f"{BASE}/v1/embeddings", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        return json.loads(urllib.request.urlopen(req, timeout=600).read())["data"]

    embed(texts[:8])                                   # warmup, not timed
    batches = [texts[i:i+200] for i in range(0, len(texts), 200)]
    t1 = time.time(); n = 0
    for b in batches:
        n += len(embed(b))
    el = time.time() - t1
    print(f"{ARM}: boot {boot:.1f}s | {n} chunks in {el:.1f}s -> {n/el:.2f} chunks/s  (exe={EXE.parent.name}, ngl={NGL})")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    logf.close()
