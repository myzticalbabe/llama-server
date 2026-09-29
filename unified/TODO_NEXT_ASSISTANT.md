# TODO for Next Assistant — 2026-05-26

## 1. Embedding Project — PROCESS BROKEN

### Problem
`direct_embed.py` uses Qwen3.5-9B (9B params) for embeddings — way too slow at **0.75 files/s**.
Only **750/6797 files (11%)** done after significant runtime. Full run would take ~2.4 hours.

### Output So Far
- **File**: `C:\Users\clint\llama-server\embeddings\embeddings.jsonl` (61 MB, 755 entries)
- **Checkpoint**: `C:\Users\clint\.config\unified-llm\embed_checkpoint.json` (750 completed)
- **Model**: Qwen3.5-9B Q4_K_M — overkill for embeddings
- **Dim**: 4096 (but most applications only need 768-1024)

### Recommended Fix — Switch to mxbai-embed-large-v1
**Option A (best)**: Use `batch_embed.py` which starts a persistent llama-server embedding endpoint:
```powershell
python batch_embed.py --resume
```
This avoids per-file context creation by sending batches (200 files/batch) via HTTP.

**Option B**: Switch to mxbai-embed-large-v1 GGUF (335M params, 1024 dim, 0.2 GB):
- Model location: `C:\Users\clint\llama-server\models\mxbai-embed-large-v1\mxbai-embed-large-v1-Q4_k_m.gguf`
- Embedding config: `C:\Users\clint\llama-server\unified\embedding_config.json`
- ~10x faster, adequate quality, widely benchmarked SOTA for 335M class

**Option C (instant)**: Ollama nomic-embed-text (768 dim, instant setup):
```powershell
ollama pull nomic-embed-text:latest
```

### File Corpus Stats (for reference)
| Category | Count | Size |
|----------|-------|------|
| .py | 1533 | 17.3 MB |
| .txt | 1472 | 18.3 MB |
| .md | 875 | 11.0 MB |
| .cpp | 721 | 244 MB |
| .json | 628 | 146 MB |
| .svelte | 266 | 0.6 MB |
| .h | 238 | 6.5 MB |
| .ts | 234 | 0.9 MB |
| .html | 168 | 5.0 MB |
| .hpp | 82 | 140 MB |
| Others | 610 | 101 MB |
| **Total** | **6797** | **651.5 MB** |

## 2. Disk Cleanup Candidates

### Delete these .whl files (1.35 GB reclaimable)
- `C:\Users\clint\AppData\Local\Temp\tf_wheel\tensorflow-2.21.0-cp312-cp312-win_amd64.whl` (335 MB)
- `C:\Users\clint\AppData\Local\Temp\tf_wheel\tensorflow-2.21.0-cp314-cp314-win_amd64.whl` (342 MB)
- Both appear twice (dedup=1.35 GB unique, 1.7 GB with dupes)

### Archive/Review Build Artifacts (5+ GB)
- `C:\Users\clint\temp_llama_cpp_build\llama.cpp` (source) — 2.9 GB
- `build_server_simple` — 446 MB (stale, from old config, can delete)
- `build_sycl_test` — 1.0 GB (SYCL experiment, can archive)
- `build_mkl` — 366 MB (oneAPI build, can archive)
- `build_server_multi` — 806 MB (current active build, KEEP)

### Temp Cleanup
- `C:\Users\clint\AppData\Local\Temp` — 5 GB total (includes 107 MB `wctD4D.tmp` from May 15)

## 3. PAMS Memory Status
4 memories saved to persistent-ai-memory DB (port 21434):
- `Memory IDs: 2cfb62df, a9e97bd4, a1675ef0, 8dad3571`

## 4. PAMS Server
PAMS server is running on port 21434 (started by executor.py PAMSManager).
Health check: OK.
To stop: `python -c "from executor import get_pams_manager; get_pams_manager().stop()"`

## 5. semantic.py Status
`semantic.py` was never created — the 4096-dim embedding pipeline needs a semantic search interface.
The existing `pams_client.py` already provides semantic search via PAMS MCP tools.
