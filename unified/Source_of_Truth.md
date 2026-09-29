# Source of Truth — Local AI Inference Architecture

## Port Configuration
| Port | Model | Hardware | Purpose |
|------|-------|----------|---------|
| **18080** | Primary (gemma-4 / Eve / selected) | Vulkan GPU (3 GB) | Deep reasoning, vision, complex code review |
| **18081** | Qwen2.5-1.5B | CPU (1 GB) | Fast chat, lightweight completions, quick code |
| **18082** | mxbai-embed-large-v1 | CPU (335 MB) | Semantic search, RAG, embeddings (POST /v1/embeddings) |

## Launch Scripts
- `launch_chat.bat` — Main launcher with model picker
- `launch_llama_server.ps1` — Spawns llama-server and health-checks on port 18080
- `launch_interpreter.py` — Manages 3 inference servers + persistent memory
- `switch_model_tui.py` — Interactive model switcher
- `select_model.ps1` — PowerShell model picker

## Persistent Memory
- **Server:** `pams-server.exe` (Python 3.14 Scripts)
- **Data:** `G:\source-of-truth\persistent-ai-memory\memory_data`
- **Embeddings:** mxbai-embed-large-v1 via llama-server on port 18082

## Models Directory
`C:\Users\clint\llama-server\models\`
- `gemma-4-E2B/` — Primary chat + mmproj
- `Eve-V2-Unleashed/` — Eve Unleashed + mmproj
- `Qwen2.5-1.5B-Instruct/` — Fast secondary
- `mxbai-embed-large-v1/` — Embedding model
- `nomic-embed-text-v1.5/` — Legacy embedding (Ollama fallback)

## Embedding Provider
Primary: `mxbai-embed-large-v1` served via `llama-server` on `127.0.0.1:18082`
Fallback: `nomic-embed-text` via Ollama on `localhost:11434`
