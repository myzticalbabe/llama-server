# Local inference servers — provenance, ports, and roles

**Single source of truth for "which port is what".** If another document
(`Source_of_Truth.md`, `ASSISTANT_INSTRUCTIONS.md`, `config.toml` comments)
disagrees with the table below, this table is correct — it is the only one
verified against running processes.

All binaries below are parts of their respective projects. Nothing here is
deleted or moved; dead ends are quarantined with notes.

---

## Port map (authoritative)

The port confusion came from three different things all having a claim to
"the local model port". They are different layers with different owners:

| Port | Owner | Source of truth | Notes |
|------|-------|-----------------|-------|
| **12780** | native llama.cpp worker, spawned **by** the Rust supervisor | hardcoded — `const NATIVE_PORT: u16 = 12780`, `src/main.rs:462` | **Internal channel.** Not configurable. The supervisor proxies to it. Do NOT point clients here. |
| **8080** | Rust supervisor HTTP proxy | `port` in `config.toml` | The client-facing endpoint. This is the one to hand to clients. |
| **18080 / 18081 / 18082** | older Jan-b9967 launch scripts (`launch_chat.bat`, `select_model.ps1`, …) | `Source_of_Truth.md` | A **separate path** — the Rust supervisor does not use these. Unchanged. |
| **8081–8084** | azazael per-model llama.cpp servers | `START_LOCAL_MODEL.bat` | One GGUF per port (mythos :8081, v2 :8082, qwopus :8083, ornith :8084). |
| **11434** | Ollama | external | Not part of this stack. |
| **1234** | LM Studio | external | Not part of this stack. |

Previously `SERVERS.md` listed 12780 as *"per config.toml, previously verified
Jan port (currently unused)"*. Wrong on both counts: it is used, and it is not
a Jan port — it is this supervisor's own hardcoded worker port.

**Azazael integration:** the `llamaserver` provider in
`azazael/src/CORE/inference/providers.py` defaults to `http://localhost:8080/v1`
— i.e. the Rust supervisor — and is overridable with `LLAMASERVER_BASE_URL`.

---

## Rust supervisor — VERIFIED WORKING (2026-09-18)

**Binary:** `C:/Users/clint/llama-server/target/release/llama-server.exe`
(5,031,936 bytes — genuine Rust: `axum`/`tokio`/`reqwest`; the C++ engine
binary has no Rust markers at all)

```
INFO Launching native: .../llama-b10964-vulkan/llama-server.exe --port 12780 -np 4 -c 8192 -t 8 \
       --model .../gemma-4-E2B-it-IQ4_XS/model.gguf --mmproj .../mmproj.gguf --reasoning off
INFO Native server started  pid=1952  port=12780  backend=CPU
INFO Listening on http://0.0.0.0:18099
```

Verified: `/v1/models` 200, `/v1/chat/completions` 200 (real inference),
`stream:true` returns `text/event-stream` ending in `[DONE]`.

**This entry used to read:** *"Rust launcher … binds its port but spawns the
worker via a bad relative path (`bin/bin/ipex-sycl/...`, os error 267)."*
**That is fixed and this note is obsolete.** The 2026-09-17 audit in
`config.toml` corrected `llama_server_path` and the two dead `llama-cli` paths,
and the spawn now succeeds on the real `llama-b10964-vulkan` build.

### REQUIRED: `reasoning = "off"` in `config.toml`

`reasoning` is only forwarded to the native worker when it is **not** `auto`,
and the schema default is `auto` — which silently breaks every client that reads
only `message.content`. Measured, same request, gemma-4-E2B, through the proxy:

| `reasoning` | `finish_reason` | `content` | `reasoning_content` |
|---|---|---|---|
| `auto` (old default) | `length` | `''` | `Thinking Process:...` |
| `off` (now set) | `stop` | `'Pong'` | `None` |

A thinking model streams its trace into `reasoning_content` and leaves `content`
EMPTY until thinking ends, so under `auto` the client gets a blank assistant turn
and the task silently does nothing. Open Interpreter reads only `content`.

---

## Native engine (the C++ layer)

- **Binary:** `llama-server/bin/llama-b10964-cpu/llama-server.exe` (or `-vulkan`)
- **Project:** llama.cpp b10964, **C++**. Each `.exe` is a 9,216-byte stub; the
  code is in `llama-server-impl.dll` (8.9 MB) + `llama-common.dll` (7.7 MB).
- Measured on this host (Core Ultra 9 285H + Arc 140T):

| Backend | prompt eval (pp256) | generation (tg64) |
|---|---|---|
| CPU (8 threads) | **47.8 t/s** | 8.30 t/s |
| Vulkan (Arc, `-ngl 99`) | ~20 t/s | **10.5 t/s** |

CPU wins prompt eval 2.4×; Vulkan wins generation 1.26×. For agentic use
(open-interpreter re-processes a large system prompt every turn) CPU is the
better default. MTP speculative decoding (`--spec-type draft-mtp`, Qwythos-9B
only) measured **16.35 t/s vs 8.30 t/s = 1.97×**.

## Dead ends (quarantined, do not use)

- **ipex-sycl bundle** (`llama-server/bin/ipex-sycl/.../llama-server.exe`):
  exits 0 with zero output on `--help`/`--version`; never binds. Also verified
  its `llama-bench.exe` **cannot load the `qwen35` architecture** at all — the
  bundle is a June 2025 build predating it. Left in place for the owning project.
- **ipex-sycl `llama-server.exe`** additionally dies on a missing `mtmd.dll`
  (that DLL only ships in the b10964 builds).

## Model alias map (proposal — implement in launcher/config)

Servers report the full GGUF path as the model id, so every client hardcodes
90-char paths. One alias map kills a whole class of 404s.

| alias | GGUF path | ctx | backend |
|-------|-----------|-----|---------|
| `qwen15` | `llama-server/models/Qwen2.5-1.5B-Instruct/qwen2.5-1.5b-instruct-q4_k_m.gguf` | 32768 | vulkan |
| `gemma4` | `AppData/Roaming/Jan/data/llamacpp/models/gemma-4-E2B-it-IQ4_XS/model.gguf` (+ mmproj) | 32768 | vulkan |
| `pixtral12` | `llama-server/models/Pixtral-12B/pixtral-12b-Q4_K_M.gguf` (+ mmproj) | 8192 | vulkan |
| `qwen35` | `.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf` | 32768 | vulkan |
