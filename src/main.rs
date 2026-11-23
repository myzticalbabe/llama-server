//!
//! llama-server — Native llama.cpp SYCL/Level Zero + Rust
//! Copyright © 2026  ·  Intel Core Ultra / Arc 140T / AI Boost NPU
//!

#![allow(unused_imports)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::redundant_closure_call)]

// ── local modules ──────────────────────────────────────────────────────────
mod config;
mod hardware;

// ── std ─────────────────────────────────────────────────────────────────────
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

// ── tokio ─────────────────────────────────────────────────────────────────────
use tokio::net::TcpStream;
use tokio::time::timeout;

// ── axum ────────────────────────────────────────────────────────────────────
use axum::{
    body::{Body, Bytes},
    extract::{FromRequestParts, State},
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
    response::sse::Event,
    routing::{get, post},
    serve,
    Json,
    Router,
};


// ── tracing ─────────────────────────────────────────────────────────────────
use tracing::{error, info, warn};
use tracing_subscriber::{self, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

// ── clap ────────────────────────────────────────────────────────────────────
use clap::Parser;

// ── local ───────────────────────────────────────────────────────────────────
use config::{Backend, ServerConfig};
use hardware::{detect_hardware, print_hardware_info, HardwareInfo};

// ════════════════════════════════════════════════════════════════════════════
// CLI
// ════════════════════════════════════════════════════════════════════════════

#[derive(Parser, Clone, Debug)]
#[command(
    name    = "llama-server",
    about   = "Native llama.cpp (SYCL/Level Zero + CPU) inference server.",
    version
)]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Path to GGUF model to load at startup
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Path to GGUF multimodal projector (auto-detected next to --model if omitted)
    #[arg(long)]
    mmproj: Option<PathBuf>,

    /// Skip the multimodal projector even if the config sets one
    /// (needed when testing a text-only model while the config points
    /// at another model's mmproj — a stale projector kills native load).
    #[arg(long)]
    no_mmproj: bool,

    /// Backend: cpu | sycl | npu | auto
    #[arg(short, long, default_value = "auto")]
    backend: String,

    /// Context window (tokens)
    #[arg(long, default_value_t = 8192)]
    ctx: u32,

    /// Concurrent inference slots
    #[arg(long, default_value_t = 4)]
    slots: u32,

    /// CPU threads
    #[arg(long, default_value_t = 8)]
    threads: u32,

    /// Reasoning/thinking in chat: on | off | auto (default auto).
    /// Passes --reasoning through to the native server. Use off for
    /// thinking models whose trace never terminates into an answer
    /// (observed: gemma-4-12B IQ4_XS thinking-void on 3/5 prompts).
    #[arg(long, default_value = "auto")]
    reasoning: String,

    /// Level Zero device selector (e.g. "level_zero:0")
    #[arg(long)]
    gpu_id: Option<String>,

    /// Print hardware info and exit
    #[arg(long)]
    check_hw: bool,

    /// Quiet start
    #[arg(short, long)]
    quiet: bool,
}

// ════════════════════════════════════════════════════════════════════════════
// Shared state
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct AppState {
    cfg:      Arc<ServerConfig>,
    hw:       Arc<HardwareInfo>,
    backend:  Backend,
}

// ════════════════════════════════════════════════════════════════════════════
// REST types
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ModelsResponse {
    object:   String,
    data:     Vec<ModelInfo>,
    hardware: HardwareInfo,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ModelInfo {
    id:         String,
    object:     String,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_size:  Option<u64>,
}

// ════════════════════════════════════════════════════════════════════════════
// main
// ════════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> anyhow::Result<()> {

    // ── logging ────────────────────────────────────────────────────────────
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "llama_server=info,warn");
    }
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().compact())
        .with(EnvFilter::from_default_env())
        .init();

    // ── args ───────────────────────────────────────────────────────────────
    let args = Args::parse();

    // ── hardware scan ──────────────────────────────────────────────────────
    let hw = detect_hardware()?;

    // ── config ─────────────────────────────────────────────────────────────
    // 2026-09-17: loaded before the banner/check-hw so config.toml actually
    // governs the listen port and backend. Both were dead keys before —
    // main() read args.port / args.backend and never looked at cfg.
    let mut cfg = ServerConfig::load();
    if let Err(ref e) = cfg {
        warn!("Could not load config.toml ({}), using defaults", e);
        cfg = Ok(ServerConfig::default());
    }
    let mut cfg = cfg.unwrap_or_default();

    // CLI overrides
    if args.model.is_some()          { cfg.model_path = args.model; }
    if args.mmproj.is_some()         { cfg.mmproj_path = args.mmproj; }
    if args.no_mmproj                { cfg.mmproj_path = None; }
    if args.ctx     != 8192          { cfg.n_ctx      = args.ctx; }
    if args.slots   != 4             { cfg.n_slots    = args.slots; }
    if args.threads != 8             { cfg.threads    = args.threads; }
    if args.reasoning != "auto"      { cfg.reasoning  = args.reasoning.clone(); }
    if let Some(ref id) = args.gpu_id {
        cfg.oneapi_device_selector = Some(id.clone());
    }

    // Resolve bin paths
    resolve_bins(&mut cfg);
    cfg.resolve_paths();

    // ── effective listen port + backend (CLI wins, else config.toml) ───────
    let listen_port: u16 = if args.port != 8080 { args.port } else { cfg.port };
    cfg.port = listen_port;

    let backend_name = if args.backend != "auto" {
        args.backend.clone()
    } else {
        cfg.backend.to_string()
    };
    let backend = parse_backend(&backend_name, &hw);

    if !args.quiet {
        banner(&hw, listen_port, &backend_name);
    }

    // ── check-hw mode ──────────────────────────────────────────────────────
    if args.check_hw {
        print_hardware_info();
        return Ok(());
    }

    info!("Using backend: {}", backend);
    if cfg.interpreter.auto_start {
        if cfg.interpreter.path.as_os_str().is_empty() {
            warn!("interpreter.auto_start=true but no interpreter.exe was found — \
                   set [interpreter].path in config.toml");
        } else {
            let base = if cfg.interpreter.api_base.is_empty() {
                format!("http://127.0.0.1:{listen_port}/v1")
            } else {
                cfg.interpreter.api_base.clone()
            };
            info!("Interpreter auto-load armed: {} → {}",
                cfg.interpreter.path.display(), base);
        }
    }

    // ── router ─────────────────────────────────────────────────────────────
    let state = AppState {
        cfg:      Arc::new(cfg),
        hw:       Arc::new(hw.clone()),
        backend:  backend.clone(),
    };

    // clone for the startup auto-load task (Router::with_state consumes state)
    let boot = state.clone();

    let app = Router::new()
        .route("/",              get(index))
        .route("/health",        get(|| async { "OK" }))
        .route("/v1/models",     get(models_handler))
        .route("/v1/capabilities", get(capabilities_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state);

    // �� serve �����������������������������������������������������������������
    // ══════════════════════════════════════════════════════════════════════
    // Native backend FIRST, our own listener SECOND
    // ══════════════════════════════════════════════════════════════════════
    //
    // 2026-09-17 fix (port 8080 stayed bound by a dead PID).
    //
    // Ordering matters on Windows: Rust's Command::spawn always calls
    // CreateProcess with bInheritHandles = TRUE (it has to, for the std
    // handles), so every *inheritable* handle is duplicated into the child.
    // Binding port 8080 and then spawning the native server handed the child a
    // duplicate of the listening socket. Killing this process left the orphan
    // holding it: netstat still showed `0.0.0.0:8080 LISTENING <dead pid>` and
    // the next start died with
    //     "Only one usage of each socket address ... (os error 10048)"
    // (the same failure mode as the 700 KB zombie native server).
    //
    // Spawning the child before the bind means the socket does not exist yet,
    // so there is nothing to inherit.
    //
    // The native backend used to spawn only on the first chat completion, so
    // /health never meant "model resident" and there was no hook to attach the
    // interpreter to. It now loads at boot; the weights stream in in the
    // background while we bind and start serving.
    let native_started = match spawn_native_server(&boot).await {
        Ok(pid) => {
            info!("Native backend pid={pid} — loading model in background…");
            true
        }
        Err(e) => {
            warn!("Startup model auto-load skipped: {e}");
            false
        }
    };

    // ── bind our own port (after the child, so it cannot inherit it) ───────
    let addr  = SocketAddr::from(([0, 0, 0, 0], listen_port));
    let listen = tokio::net::TcpListener::bind(addr).await?;
    info!("Listening on http://0.0.0.0:{}\n  Backend : {}", listen_port, backend);

    // ── model auto-load at startup ─────────────────────────────────────────
    // Attach the interpreter once the weights are actually resident.
    if native_started {
        tokio::spawn(async move {
            if wait_native_ready().await {
                info!("Model loaded — native backend ready on port {NATIVE_PORT}");
                match spawn_interpreter(&boot).await {
                    Ok(ipid) => info!("Python interpreter auto-loaded (pid {ipid})"),
                    Err(e)   => warn!("Interpreter auto-load skipped: {e}"),
                }
            } else {
                warn!("Native backend did not report ready within timeout");
            }
        });
    }

    axum::serve(listen, app).await?;
    Ok(())
}

// ════════════════════════════════════════════════════════════════════════════
// helpers
// ════════════════════════════════════════════════════════════════════════════

fn parse_backend(s: &str, _hw: &HardwareInfo) -> Backend {
    match s.to_lowercase().as_str() {
        "cpu"  => Backend::Cpu,
        "sycl" => Backend::Sycl,
        "npu"  => Backend::Npu,
        "auto" => {
            // Measured 2026-09-16 (1.5B: cpu 43.3 vs sycl 39.2 tok/s;
            // 9B: tie 9.1 vs 9.5): CPU wins or ties on shared DDR5, so
            // auto prefers CPU. Override explicitly with --backend sycl.
            Backend::Cpu
        }
        other  => {
            warn!("Unknown backend '{}' — using cpu", other);
            Backend::Cpu
        }
    }
}

/// Print startup banner
fn banner(hw: &HardwareInfo, port: u16, backend: &str) {
    println!();
    println!("  ╔═══════════════════════════════════════════════════════╗");
    println!("  ║          llama-server  v{:<26}║", env!("CARGO_PKG_VERSION"));
    println!("  ╚═══════════════════════════════════════════════════════╝");
    println!();
    println!("  CPU : {:<44}  [{}/{}]",
        hw.cpu.model_name, hw.cpu.vendor_id, hw.cpu.family);
    println!("  GPU : {:<44}  [{}/{}]",
        hw.gpu.name,        hw.gpu.vendor_id,   hw.gpu.device_id);
    println!("  NPU : {:<44}  [{}/{}]",
        hw.npu.name,        hw.npu.vendor_id,   hw.npu.device_id);
    if let Some(ref v) = hw.gpu.vulkan_version {
        println!("  Vulkan : {}", v);
    }
    println!("  SYCL/L0: GPU={} NPU_plugin={}",
        if hw.gpu.sycl_level_zero { "YES" } else { "no" },
        if hw.npu.level_zero { "YES" } else { "no" });
    println!();
    println!("  Backend : {}", backend);
    println!("  Listen  : http://0.0.0.0:{}", port);
    println!();
}

/// Fill unset binary paths. config.rs resolve_paths() already searched
/// exe dir + project root; this only keeps absolute config values as-is.
fn resolve_bins(cfg: &mut ServerConfig) {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    let defaults = [
        (&mut cfg.llama_server_path,  "./bin/ipex-sycl/llama-cpp-ipex-llm-2.3.0b20250612-win/llama-server.exe"),
        (&mut cfg.llama_cli_path,     "./bin/ipex-sycl/llama-cpp-ipex-llm-2.3.0b20250612-win/llama-cli.exe"),
        (&mut cfg.llama_cli_npu_path, "./bin/ipex-npu/llama-cli-npu.exe"),
    ];

    for (slot, rel) in defaults {
        if !slot.exists() {
            let resolved = exe_dir.join(rel.trim_start_matches("./"));
            if resolved.exists() {
                *slot = resolved;
            }
        }
    }
}

/// First .gguf under `dir`
fn first_gguf(dir: &PathBuf) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .find(|e| {
            e.path().extension()
                .and_then(|s| s.to_str())
                .map_or(false, |ext| ext.eq_ignore_ascii_case("gguf"))
        })
        .map(|e| e.path())
}

// ════════════════════════════════════════════════════════════════════════════
// Handlers
// ════════════════════════════════════════════════════════════════════════════

async fn index() -> impl IntoResponse {
    axum::response::Html(HTML_INDEX)
}

/// GET /v1/models  — list all GGUF files and expose hardware info
async fn models_handler(
    State(state): State<AppState>,
) -> Json<ModelsResponse> {
    let mut models = vec![];

    // Specific model from config
    if let Some(ref p) = state.cfg.model_path {
        if p.exists() {
            let sz = std::fs::metadata(p).map(|m| m.len()).ok();
            models.push(ModelInfo {
                id: p.file_stem().and_then(|s| s.to_str()).unwrap_or("?").into(),
                object: "model".into(),
                file_size: sz,
            });
        }
    }

    // Fall back: scan models dir
    if models.is_empty() {
        if let Some(path) = first_gguf(&state.cfg.models_dir) {
            let sz = std::fs::metadata(&path).map(|m| m.len()).ok();
            models.push(ModelInfo {
                id: path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").into(),
                object: "model".into(),
                file_size: sz,
            });
        }
    }

    Json(ModelsResponse {
        object: "list".into(),
        data:   models,
        hardware: (*state.hw).clone(),
    })
}

/// GET /api/capabilities
async fn capabilities_handler(
    State(state): State<AppState>,
) -> Json<HardwareInfo> {
    Json((*state.hw).clone())
}


// ════════════════════════════════════════════════════════════════════════════
// Native server bootstrap
// ════════════════════════════════════════════════════════════════════════════

/// One-shot fork of the native C++ llama-server onto port 12780.
///
/// Protected by `NATIVE_SERVER` (process-wide `std::sync::Mutex<Option<u32>>`).
/// The guard also performs a liveness probe: if the stored PID has died
/// (e.g. because the C++ server crashed at startup), the stale entry is
/// cleared so a fresh instance can be spawned.
static NATIVE_SERVER: std::sync::Mutex<Option<u32>> = std::sync::Mutex::new(None);

/// TCP check port for the native server.
const NATIVE_PORT: u16 = 12780;
/// How long to wait for a TCP response when probing an existing PID.
const NATIVE_PROBE_TIMEOUT: Duration = Duration::from_millis(500);
/// How long to wait for the native backend to report the model as loaded.
const NATIVE_READY_TIMEOUT: Duration = Duration::from_secs(300);

/// Latch: the interpreter is auto-loaded at most once per server lifetime.
///
/// 2026-09-17: the previous guard polled a `Child` handle, but a detached
/// REPL started through `cmd /c start` has no handle to poll. The latch is
/// enough because re-attaching a closed interpreter is a manual action
/// (`python run.py interpreter`), not something a chat request should do.
static INTERPRETER_LAUNCHED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Spawn the interpreter process.
///
/// On Windows the default is `cmd /c start "" <exe> …`: it gives the REPL its
/// own console with real standard handles. A plain `Command::spawn` inherits
/// the server's stdio, and since the server is headless its stdin is at EOF —
/// Open Interpreter would print "Exiting…" and die the moment it loaded.
#[cfg(windows)]
fn spawn_interpreter_process(
    exe: &Path,
    args: &[String],
    envs: &HashMap<String, String>,
    new_console: bool,
) -> std::io::Result<u32> {
    if new_console {
        let mut cmd = std::process::Command::new("cmd.exe");
        cmd.args(["/c", "start", "", &exe.display().to_string()])
            .args(args)
            .envs(envs);
        return cmd.spawn().map(|c| c.id());
    }
    let mut cmd = std::process::Command::new(exe);
    cmd.args(args).envs(envs);
    cmd.spawn().map(|c| c.id())
}

#[cfg(not(windows))]
fn spawn_interpreter_process(
    exe: &Path,
    args: &[String],
    envs: &HashMap<String, String>,
    _new_console: bool,
) -> std::io::Result<u32> {
    let mut cmd = std::process::Command::new(exe);
    cmd.args(args).envs(envs);
    cmd.spawn().map(|c| c.id())
}

fn native_health_url() -> String {
    format!("http://127.0.0.1:{NATIVE_PORT}/health")
}

fn native_chat_url() -> String {
    format!("http://127.0.0.1:{NATIVE_PORT}/v1/chat/completions")
}

/// Poll the native backend's /health until it answers or the timeout expires.
async fn wait_native_ready() -> bool {
    let client   = reqwest::Client::new();
    let deadline = tokio::time::Instant::now() + NATIVE_READY_TIMEOUT;
    while tokio::time::Instant::now() < deadline {
        match client
            .get(native_health_url())
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => return true,
            _ => tokio::time::sleep(Duration::from_secs(2)).await,
        }
    }
    false
}

/// Spawn Open Interpreter against this server once the model is resident.
///
/// Guarded by `INTERPRETER`: the startup task and the first chat request can
/// both reach here, and only one interpreter should ever be running.
async fn spawn_interpreter(state: &AppState) -> anyhow::Result<u32> {
    let cfg = &state.cfg;

    if !cfg.interpreter.auto_start {
        return Err(anyhow::anyhow!("interpreter auto_start is disabled in config.toml"));
    }
    if cfg.interpreter.path.as_os_str().is_empty() {
        return Err(anyhow::anyhow!(
            "no interpreter.exe found (set [interpreter].path in config.toml)"));
    }
    if !cfg.interpreter.path.exists() {
        return Err(anyhow::anyhow!(
            "interpreter not found at {}", cfg.interpreter.path.display()));
    }

    // ── at most one auto-load per server lifetime ──────────────────────────
    if INTERPRETER_LAUNCHED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return Err(anyhow::anyhow!("interpreter already auto-loaded this session"));
    }

    // ── resolve endpoint + model id ────────────────────────────────────────
    let api_base = if cfg.interpreter.api_base.is_empty() {
        format!("http://127.0.0.1:{}/v1", cfg.port)
    } else {
        cfg.interpreter.api_base.clone()
    };

    let model_id = if !cfg.interpreter.model.is_empty() {
        cfg.interpreter.model.clone()
    } else {
        cfg.model_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .and_then(|s| s.to_str())
            .unwrap_or("model.gguf")
            .to_string()
    };

    // Explicit flags rather than --profile: the checked-in local.yaml is the
    // obsolete Open Interpreter 0.2.x schema and 0.4.x ignores it.
    let mut args: Vec<String> = vec![
        "--api_base".into(), api_base.clone(),
        "--model".into(),    model_id.clone(),
        "--api_key".into(),  cfg.interpreter.api_key.clone(),
        "--offline".into(),
    ];
    if !cfg.interpreter.profile.is_empty() {
        args.push("--profile".into());
        args.push(cfg.interpreter.profile.clone());
    }
    args.extend(cfg.interpreter.args.iter().cloned());

    // Open Interpreter's venv keeps its native DLLs beside the exe.
    let mut envs: HashMap<String, String> = HashMap::new();
    if let Some(dir) = cfg.interpreter.path.parent() {
        envs.insert("PATH".into(), format!(
            "{};{}", dir.display(), std::env::var("PATH").unwrap_or_default()));
    }

    let pid = spawn_interpreter_process(
        &cfg.interpreter.path, &args, &envs, cfg.interpreter.new_console,
    )
    .map_err(|e| anyhow::anyhow!(
        "failed to spawn {}: {e}", cfg.interpreter.path.display()))?;

    info!("Launched interpreter{}: {} {}",
        if cfg.interpreter.new_console { " (new console)" } else { "" },
        cfg.interpreter.path.display(), args.join(" "));
    Ok(pid)
}

/// Spawn the native `llama-server.exe` on port 12780 (one-shot spawn, guarded
/// by a process-wide mutex).
/// Returns the OS PID on success.
async fn spawn_native_server(state: &AppState) -> anyhow::Result<u32> {
    // ── liveness check + singleton guard ─────────────────────────────────────
    let stale_pid = {
        let guard = NATIVE_SERVER.lock().unwrap();
        let pid = *guard;
        drop(guard); // MutexGuard is !Send — must drop before .await
        pid
    };

    if let Some(existing_pid) = stale_pid {
        // Probe the TCP port that the native server would be listening on.
        // Fast connect = a live process is holding the port → refuse to double-spawn.
        // Timeout / refused = the process crashed or never reached listen → stale → clear.
        let port_alive = timeout(NATIVE_PROBE_TIMEOUT, async move {
            TcpStream::connect(format!("127.0.0.1:{NATIVE_PORT}")).await.is_ok()
        })
        .await
        .unwrap_or(false); // probe timed out → treat as dead

        if port_alive {
            // 2026-09-12 fix: a live backend is the desired singleton
            // state, not an error. The old code 503'd every request
            // after the first one per supervisor lifetime.
            info!("Native server already running (pid {existing_pid}) - reusing");
            return Ok(existing_pid);
        }

        // Stale PID — crash-orphaned entry. Clear and fall through to re-spawn.
        info!("Native server pid {existing_pid} is dead (port {NATIVE_PORT} not reachable) — clearing stale entry");
        {
            let mut guard = NATIVE_SERVER.lock().unwrap();
            *guard = None;
        }
    }

    let cfg  = &state.cfg;
    let hw   = &state.hw;
    let kind = match state.backend {
        Backend::Sycl => "SYCL",
        Backend::Npu  => "NPU",
        Backend::Cpu  => "CPU",
        Backend::Auto => "AUTO",
    };

    // NPU cannot serve: bin/ipex-npu ships llama-cli only (no HTTP server),
    // so server flags would never bind. Fail fast with guidance instead of
    // a 300s readiness timeout ending in 503.
    if matches!(state.backend, Backend::Npu) {
        return Err(anyhow::anyhow!(
            "NPU serve unsupported: ipex-npu backend is llama-cli only; use --backend cpu|sycl"));
    }

    let (exe, mut envs) = match state.backend {
        Backend::Npu => {
            info!("Backend NPU: {}  [{}/{}]",
                hw.npu.name, hw.npu.vendor_id, hw.npu.device_id);
            (cfg.llama_cli_npu_path.clone(), HashMap::new())
        }
        Backend::Sycl => {
            info!("Backend SYCL: Arc 140T  [{}/{}]",
                hw.gpu.vendor_id, hw.gpu.device_id);
            let mut envs: HashMap<String, String> = HashMap::with_capacity(4);
            envs.insert(
                "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS".into(),
                "1".into(),
            );
            envs.insert(
                "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS".into(),
                "1".into(),
            );
            if let Some(ref sel) = cfg.oneapi_device_selector {
                envs.insert("ONEAPI_DEVICE_SELECTOR".into(), sel.clone());
            }
            (cfg.llama_server_path.clone(), envs)
        }
        _ => {
            info!("Backend CPU: {}", hw.cpu.model_name);
            (cfg.llama_server_path.clone(), HashMap::new())
        }
    };

    // Merge extra env from config.toml [env]
    for (k, v) in &cfg.env {
        envs.insert(k.clone(), v.clone());
    }

    // Build argument list
    let mut args: Vec<String> = vec![
        "--port".into(),     NATIVE_PORT.to_string(),
        "-np".into(),         cfg.n_slots.to_string(),
        "-c".into(),          cfg.n_ctx.to_string(),
        "-t".into(),          cfg.threads.to_string(),
    ];

    // Model
    let model: Option<PathBuf> = if let Some(ref p) = cfg.model_path {
        p.clone().into()
    } else {
        first_gguf(&cfg.models_dir)
    };
    if let Some(ref m) = model {
        args.extend(["--model".into(), m.display().to_string()]);
    } else {
        warn!("No .gguf model found — starting server in live-reload mode.");
    }

    // Multimodal projector
    if let Some(ref p) = cfg.mmproj_path {
        args.extend(["--mmproj".into(), p.display().to_string()]);
    }

    // Reasoning mode (only forwarded when explicitly set; native
    // default is auto-detect from the template).
    if cfg.reasoning != "auto" {
        args.extend(["--reasoning".into(), cfg.reasoning.clone()]);
    }

    // SYCL layers
    if matches!(state.backend, Backend::Sycl) {
        args.extend(["-ngl".into(), "99".into()]);
    }
    // NPU layers
    if matches!(state.backend, Backend::Npu) {
        args.extend(["-ngl".into(), "99".into()]);
    }

    // ── spawn ──────────────────────────────────────────────────────────────
    if let Some(parent) = exe.parent() {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap());

        // OPENVINO DLL directory: ggml-openvino.dll depends on openvino.dll
        // + tbb12.dll which are shipped in the ipex-npu bin directory. They
        // are NOT co-located with the llama-server.exe binary and must be on
        // PATH for the loader to resolve them at process startup.
        //
        // exe_dir = parent of the RUST llama-server binary.
        // target/release/  → ../../bin/ipex-npu  = project-root/bin/ipex-npu  ✓
        static OPENVINO_DLL_DIR: &str = "../../bin/ipex-npu";
        let dll_dir = exe_dir.join(OPENVINO_DLL_DIR);
        if dll_dir.exists() {
            envs.insert("PATH".into(),
                format!("{};{}", dll_dir.display(),
                    envs.get("PATH").cloned().unwrap_or_default()));
        } else {
            warn!("OpenVINO DLL dir not found: {}", dll_dir.display());
        }

        let mut cmd = std::process::Command::new(&exe);
        cmd.current_dir(parent)
            .envs(&envs)
            .args(&args)
            // stdin is null, never inherited: the native server has no console
            // and must not hold the parent's stdin pipe open.
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        info!("Launching native: {} {}", exe.display(), args.join(" "));

        let child = cmd.spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn {:?}: {}", exe, e))?;

        let pid = child.id();
        *NATIVE_SERVER.lock().unwrap() = Some(pid);
        info!("Native server started  pid={}  port={}  backend={}", pid, NATIVE_PORT, kind);
        Ok(pid)
    } else {
        Err(anyhow::anyhow!("No parent dir for {}", exe.display()))
    }
}

// ════════════════════════════════════════════════════════════════════════════
// /v1/chat/completions  —  proxy → native llama-server on port 12780
// ════════════════════════════════════════════════════════════════════════════

#[axum::debug_handler]
async fn chat_completions_handler(
    State(state): State<AppState>,
    body: Bytes,
) -> impl IntoResponse {
    use axum::Json;
    use axum::http::StatusCode as HTTPStatus;

    // launch native server
    let _native_pid = match spawn_native_server(&state).await {
        Ok(pid) => pid,
        Err(e) => {
            error!("Failed to launch native llama-server: {e}");
            return (
                HTTPStatus::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    // readiness wait (2026-09-12 fix): the old code proxied immediately,
    // so the first-ever request always raced model loading and died.
    if !wait_native_ready().await {
        error!("Native server never became ready on port {NATIVE_PORT}");
        return (
            HTTPStatus::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "native backend did not become ready (model still loading or crashed)"})),
        )
            .into_response();
    }

    // model is resident — attach the interpreter if auto-load hasn't run yet
    // (silent when it already has; the latch keeps this off the hot path)
    if !INTERPRETER_LAUNCHED.load(std::sync::atomic::Ordering::SeqCst) {
        if let Err(e) = spawn_interpreter(&state).await {
            warn!("Interpreter auto-load skipped: {e}");
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // proxy the request to the native server — BYTE-LEVEL PASSTHROUGH
    // ══════════════════════════════════════════════════════════════════════
    //
    // 2026-09-17 fix (the "it crashed after input" bug).
    //
    // The old handler buffered the entire upstream body and ran
    // `serde_json::from_str` over it. Open Interpreter — like every OpenAI
    // SDK — sends `stream: true`, so the native server answers with
    // `text/event-stream`:
    //
    //     data: {"choices":[{"delta":{"content":"OK"}}]}\n\n
    //
    // That is not JSON, so the parse failed with
    // "expected value at line 1 column 1" and the handler downgraded a
    // perfectly good HTTP 200 into a 502 `proxy_error`. The interpreter saw a
    // 502 on its first message and died. Buffering also meant token-by-token
    // output was impossible.
    //
    // Now the status, content-type and body stream are forwarded verbatim:
    // SSE stays SSE, plain JSON stays plain JSON. No parse, no buffering.
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        // Deliberately no total timeout — a long generation must not be cut
        // off mid-stream. The native server has its own slot timeouts.
        .build()
        .unwrap_or_default();

    let upstream: reqwest::Response = match client
        .post(native_chat_url())
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream")
        .body(body.to_vec())
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            error!("proxy POST error: {e}");
            return (
                HTTPStatus::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": { "message": format!("{e}"), "type": "proxy_error" }
                })),
            )
                .into_response();
        }
    };

    let status = HTTPStatus::from_u16(upstream.status().as_u16())
        .unwrap_or(HTTPStatus::BAD_GATEWAY);

    // Copy content-type verbatim (text/event-stream, application/json, …).
    // content-length / transfer-encoding are intentionally NOT copied: the
    // body is now a stream of unknown length, so hyper re-frames it itself.
    let content_type = upstream.headers().get(header::CONTENT_TYPE).cloned();

    let mut response = Response::new(Body::from_stream(upstream.bytes_stream()));
    *response.status_mut() = status;
    if let Some(ct) = content_type {
        response.headers_mut().insert(header::CONTENT_TYPE, ct);
    }
    response
}

// ════════════════════════════════════════════════════════════════════════════
// HTML index page — shown on GET /
// ════════════════════════════════════════════════════════════════════════════

static HTML_INDEX: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>llama-server · Intel SYCL/Level Zero</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root{--bg:#0d1117;--fg:#e6edf3;--accent:#58a6ff;--ok:#3fb950;
        --warn:#f0883e;--border:#30363d;--card:#161b22}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Cascadia Code','Fira Code','JetBrains Mono',menlo,
        'Courier New',monospace;background:var(--bg);color:var(--fg);min-height:100vh}
  header{background:var(--card);border-bottom:1px solid var(--border);padding:18px 28px}
  h1{font-size:1.35em;color:var(--accent);margin-bottom:4px}
  .tag{display:inline-block;padding:2px 9px;border-radius:4px;font-size:.78em;
       border:1px solid;margin:1px}
  .tag.ok  {background:#14261e;color:var(--ok);  border-color:var(--ok)  }
  .tag.warn{background:#2a160f;color:var(--warn);border-color:var(--warn)}
  .tag.info{background:#111a2a;color:var(--accent);border-color:var(--accent)}
  main{max-width:820px;margin:24px auto;padding:0 20px}
  table{width:100%;border-collapse:collapse;margin:14px 0 24px;font-size:.9em}
  th,td{padding:7px 12px;text-align:left;border-bottom:1px solid var(--border)}
  th{color:#8b949e;border-bottom:2px solid var(--border)}
  pre{background:var(--card);border:1px solid var(--border);border-radius:6px;
      padding:14px;font-size:.82em;overflow:auto;margin:10px 0}
  code{color:#79c0ff}
  a{color:var(--accent)}
  #cap-wrap{margin:14px 0}
</style>
</head>
<body>
<header>
  <h1>🦙 llama-server <span style="font-size:.6em;color:#8b949e">
    Intel SYCL · NPU · Arc GPU</span></h1>
  <div style="margin-top:6px">
    <span class="tag ok">CPU</span>
    <span class="tag ok">Arc 140T</span>
    <span class="tag ok">AI Boost NPU</span>
    <span class="tag info">SYCL</span>
    <span class="tag info">Level Zero</span>
    <span class="tag info">Rust</span>
  </div>
</header>
<main>
  <div id="cap-wrap"><em>Loading hardware info…</em></div>
  <table>
    <thead><tr><th>Backend</th><th>Device</th><th>Vendor / Device ID</th></tr></thead>
    <tbody>
      <tr><td><span class="tag ok">CPU</span></td>
          <td id="cpu-model" colspan="2">…</td></tr>
      <tr><td><span class="tag ok">GPU</span></td>
          <td id="gpu-name">…</td>
          <td id="gpu-ids">…</td></tr>
      <tr><td><span class="tag ok">NPU</span></td>
          <td id="npu-name">…</td>
          <td id="npu-ids">…</td></tr>
    </tbody>
  </table>
  <h3>API</h3>
  <pre><code>GET    /health
GET    /v1/capabilities
GET    /v1/models
POST   /v1/chat/completions

# Native llama-server port (auto-started on first request):
GET    http://127.0.0.1:12780/health
GET    http://127.0.0.1:12780/v1/models
POST   http://127.0.0.1:12780/v1/chat/completions</code></pre>
  <p></p>
</main>
<script>
(async () => {
  try {
    const res  = await fetch("/v1/capabilities");
    const data = await res.json();
    document.getElementById("cap-wrap").innerHTML =
      "<pre><code>" + JSON.stringify(data, null, 2) + "</code></pre>";
    document.getElementById("cpu-model").textContent =
      data.cpu.model_name + " · " + data.cpu.threads + " threads · " +
      data.cpu.features.join(",");
    document.getElementById("gpu-name").textContent  = data.gpu.name;
    document.getElementById("gpu-ids").textContent   =
      data.gpu.vendor_id + " / " + data.gpu.device_id +
      (data.gpu.vulkan_version ? " · Vulkan " + data.gpu.vulkan_version : "");
    document.getElementById("npu-name").textContent  = data.npu.name;
    document.getElementById("npu-ids").textContent   =
      data.npu.vendor_id + " / " + data.npu.device_id +
      (data.npu.level_zero ? " · L0" : " · OV");
  } catch(e) {
    document.getElementById("cap-wrap").innerHTML =
      "<em style='color:#f0883e'>Failed to load capabilities: " + e + "</em>";
  }
})();
</script>
</body>
</html>"#;
