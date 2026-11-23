"""Is the vulkan build's -ngl 0 speed CPU, or work still going to the iGPU?"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, r"C:/Users/clint/llama-server/unified")
import semantic
from pathlib import Path as P

MODEL = r"C:/Users/clint/llama-server/models/mxbai-embed-large-v1/mxbai-embed-large-v1-Q4_k_m.gguf"
BIN = Path(r"C:/Users/clint/llama-server/bin")

def junk(p):
    s = p.as_posix().lower()
    return ("/build-spark/" in s or "/build/" in s or "graphify_codebase_atlas" in s
            or s.endswith("/graph.json") or s.endswith("/graph.html")
            or s.endswith("_full.csv") or s.endswith(".log"))
files = [f for f in semantic._scan_files([P.home()/"llama-server", P.home()/"Desktop"]) if not junk(f)]
files = [f for f in files if 20000 < f.stat().st_size < 400000][:80]
ch = semantic.Chunker(); texts = []
for fp in files:
    texts.extend(c["text"] for c in ch.chunk_file(fp))
texts = texts[:400]

def modules(pid):
    ps = ("Get-Process -Id %d -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Modules | "
          "Where-Object { $_.ModuleName -like '*ggml*' } | Select-Object -ExpandProperty ModuleName" % pid)
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    return sorted({l.strip() for l in r.stdout.splitlines() if l.strip()})

def arm(name, exe, ngl, port, extra=()):
    args = [str(exe), "--model", MODEL, "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", "512", "--threads", "4", "--batch-size", "512",
            "--ubatch-size", "512", "--no-warmup", "-ngl", str(ngl),
            "--embedding", "--pooling", "mean", *extra]
    p = subprocess.Popen(args, env=dict(os.environ), stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        t0 = time.time(); up = False
        while time.time() - t0 < 180:
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).status == 200:
                    up = True; break
            except Exception:
                time.sleep(1)
        if not up:
            print(f"{name:34s} FAILED to start"); return
        mods = modules(p.pid)
        def embed(b):
            body = json.dumps({"input": b, "model": "default"}).encode()
            r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings", data=body,
                                       headers={"Content-Type": "application/json"}, method="POST")
            return len(json.loads(urllib.request.urlopen(r, timeout=600).read())["data"])
        embed(texts[:4])
        t1 = time.time(); n = 0
        for i in range(0, len(texts), 200):
            n += embed(texts[i:i+200])
        el = time.time() - t1
        has_vk = "ggml-vulkan.dll" in mods
        print(f"{name:34s} {n/el:6.2f} chunks/s   vulkan-dll={has_vk}")
    finally:
        p.terminate()
        try: p.wait(timeout=10)
        except subprocess.TimeoutExpired: p.kill()

arm("A cpu build, ngl 0", BIN/"llama-b10964-cpu/llama-server.exe", 0, 18120)
arm("B vulkan build, ngl 0", BIN/"llama-b10964-vulkan/llama-server.exe", 0, 18121)
arm("C vulkan build, ngl 0, --device none", BIN/"llama-b10964-vulkan/llama-server.exe", 0, 18122, ("--device", "none"))
arm("D vulkan build, ngl 99", BIN/"llama-b10964-vulkan/llama-server.exe", 99, 18123)
