"""Do NPU and iGPU produce the same embeddings as OpenVINO CPU?"""
import sys, numpy as np, openvino as ov
from transformers import AutoTokenizer
from pathlib import Path
IR = Path(r"C:/Users/clint/llama-server/models/ov_export/mxbai_embed_ir")
HF = Path(r"C:/Users/clint/llama-server/models/ov_export/mxbai_embed_hf")
tok = AutoTokenizer.from_pretrained(str(HF))
S = 224
texts = [
    "The board of directors approved the merger at the quarterly meeting in Reading, Pennsylvania.",
    "def embed_batch(texts): return model.encode(texts)",
    "def calculate_total(items):\n    return sum(i.price for i in items)",
    "The facility lost its license after the state audit found unreported incidents.",
    "import numpy as np\nx = np.zeros((8, 512))",
    "Section 4.2 describes the tenant's obligation to maintain the premises.",
    "The reimbursement rate increased from 12.5% to 18.0% effective July 1.",
    "class Embedder:\n    def __init__(self, path):\n        self.path = path",
]
core = ov.Core()
xml = str(IR / "openvino_model.xml")
def vecs(device):
    cm = core.compile_model(xml, device)
    out = cm.output(0)
    e = tok(texts, padding="max_length", max_length=S, truncation=True, return_tensors="np")
    r = cm({"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[out]
    m = e["attention_mask"][..., None].astype(np.float32)
    v = (r * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
cpu = vecs("CPU"); gpu = vecs("GPU"); npu = vecs("NPU")
for name, v in (("GPU", gpu), ("NPU", npu)):
    sims = [float(np.dot(a, b)) for a, b in zip(v, cpu)]
    print(f"{name} vs CPU: min {min(sims):.6f}  mean {sum(sims)/len(sims):.6f}")
print("dims:", cpu.shape)
