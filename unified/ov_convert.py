"""Convert mxbai-embed-large-v1 to OpenVINO IR directly (optimum is broken against
transformers 5.x, so this uses ov.convert_model on the torch model)."""
import json, sys, time
from pathlib import Path
import torch, openvino as ov
from transformers import AutoModel, AutoTokenizer

HF = Path(r"C:/Users/clint/llama-server/models/ov_export/mxbai_embed_hf")
IR = Path(r"C:/Users/clint/llama-server/models/ov_export/mxbai_embed_ir")
IR.mkdir(parents=True, exist_ok=True)
t0 = time.time()

tok = AutoTokenizer.from_pretrained(str(HF))
mdl = AutoModel.from_pretrained(str(HF)).eval()
print(f"loaded torch model ({sum(p.numel() for p in mdl.parameters())/1e6:.0f}M params, "
      f"{next(mdl.parameters()).dtype}) in {time.time()-t0:.0f}s", flush=True)

B, S = 8, 512
ex = {"input_ids": torch.ones((B, S), dtype=torch.long),
      "attention_mask": torch.ones((B, S), dtype=torch.long)}

with torch.no_grad():
    print("converting...", flush=True)
    ov_m = ov.convert_model(mdl, example_input=ex, input=[("input_ids", [B, S]), ("attention_mask", [B, S])])
print("converted", flush=True)
xml = IR / "openvino_model.xml"
ov.save_model(ov_m, str(xml))
tok.save_pretrained(str(IR))
print(f"saved IR {xml.stat().st_size/1e6:.0f} MB + {(IR/'openvino_model.bin').stat().st_size/1e6:.0f} MB bin "
      f"in {time.time()-t0:.0f}s", flush=True)
print("inputs:", [(i.get_any_name(), i.partial_shape) for i in ov_m.inputs], flush=True)
print("outputs:", [(o.get_any_name(), o.partial_shape) for o in ov_m.outputs], flush=True)
