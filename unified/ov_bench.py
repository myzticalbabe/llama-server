"""NPU vs iGPU (OpenVINO) vs CPU on the same texts; and the length distribution
that decides the static shape the NPU needs."""
import json, math, sys, time
from pathlib import Path
import numpy as np, torch, openvino as ov
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, r"C:/Users/clint/llama-server/unified")
import semantic
from pathlib import Path as P

HF = Path(r"C:/Users/clint/llama-server/models/ov_export/mxbai_embed_hf")
IR = Path(r"C:/Users/clint/llama-server/models/ov_export/mxbai_embed_ir")
tok = AutoTokenizer.from_pretrained(str(HF))

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

lens = sorted(len(tok(t)["input_ids"]) for t in texts)
p95 = lens[int(len(lens)*0.95)]
S = int(math.ceil(min(p95, 512) / 32) * 32)
print(f"chunks={len(texts)} tokens: median={lens[len(lens)//2]} p95={p95} -> static seq={S}", flush=True)

xml = IR / "openvino_model.xml"
if not xml.exists() or json.loads((IR/"shape.json").read_text() if (IR/"shape.json").exists() else "{}").get("S") != S:
    B = 8
    mdl = AutoModel.from_pretrained(str(HF)).eval()
    ex = {"input_ids": torch.ones((B, S), dtype=torch.long),
          "attention_mask": torch.ones((B, S), dtype=torch.long)}
    with torch.no_grad():
        ov.save_model(ov.convert_model(mdl, example_input=ex,
                                       input=[("input_ids", [B, S]), ("attention_mask", [B, S])]),
                      str(xml))
    (IR/"shape.json").write_text(json.dumps({"B": B, "S": S}))
    print(f"IR re-exported at [{B},{S}]", flush=True)

core = ov.Core()
enc = core.compile_model  # noqa

def run(device):
    t0 = time.time()
    try:
        cm = core.compile_model(str(xml), device)
    except Exception as e:
        print(f"{device:4s}  COMPILE FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        return None, None
    compile_s = time.time() - t0
    out = cm.output(0)
    B = 8
    def embed_batch(batch):
        e = tok(batch, padding="max_length", max_length=S, truncation=True, return_tensors="np")
        res = cm({"input_ids": e["input_ids"].astype(np.int64),
                  "attention_mask": e["attention_mask"].astype(np.int64)})[out]
        mask = e["attention_mask"][..., None].astype(np.float32)
        v = (res * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
    embed_batch(texts[:B])                      # warmup / compile at first infer
    t1 = time.time(); n = 0; vecs = []
    for i in range(0, len(texts), B):
        v = embed_batch(texts[i:i+B])
        vecs.append(v); n += len(v)
    el = time.time() - t1
    print(f"{device:4s}  {n/el:6.2f} chunks/s   compile {compile_s:.1f}s   ({n} chunks in {el:.1f}s)", flush=True)
    return n/el, np.concatenate(vecs)

rates = {}
cpu_v = None
for dev in ["NPU", "GPU", "CPU"]:
    rate, v = run(dev)
    if rate: rates[dev] = rate
    if dev == "CPU": cpu_v = v
    if dev != "CPU" and v is not None and cpu_v is not None:
        sims = [float(np.dot(a, b)) for a, b in zip(v, cpu_v)]
        print(f"     cosine {dev} vs CPU: min {min(sims):.6f} mean {sum(sims)/len(sims):.6f}", flush=True)
print("rates:", {k: round(v, 2) for k, v in rates.items()}, flush=True)
