# Open Interpreter — Local Inference Assistant

## System Overview
You are connected to a local AI inference stack with three servers:

| Port | Model | Purpose |
|------|-------|---------|
| **18080** | Primary chat model (gemma-4 / Eve / selected model) — Vulkan GPU | Deep reasoning, vision, complex code review |
| **18081** | Qwen2.5-1.5B — CPU | Fast chat, lightweight completions, quick code |
| **18082** | mxbai-embed-large-v1 — CPU | Semantic search, RAG, embeddings (POST to /v1/embeddings) |

## Quick Start
1. Double-click `llama-server.lnk` on the desktop
2. Or run `launch_chat.bat` from the Llama Server folder
3. Type `SW` to switch models, `OK` to launch

## Tools Available
- **Vision** — screenshots analyzed via the primary model (mmproj)
- **OCR** — Tesseract OCR for text extraction from images
- **Browser** — Playwright for web automation
- **Audio** — Whisper for microphone transcription
- **Memory** — pams-server for persistent cross-session memory
- **Screen** — PyAutoGUI for desktop control

## Memory Management
Memory is persistent across sessions, stored at:
`G:\source-of-truth\persistent-ai-memory\`

## Configuration
- Switch models via `switch_model_tui.py` or type `SW` in the launcher
- Re-scan hardware/models using the probe scripts on the Desktop
- Edit embedding provider in `embedding_config.json`

## Offline Operation
The entire stack is fully offline — no internet connection needed after initial setup.
