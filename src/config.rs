//! Config management — reads config.toml, environment overrides.
//! Uses `thiserror` for structured errors; no `anyhow` dependency.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// CPU-only (AVX2/AVX512-NN)
    #[default]
    Cpu,
    /// Intel Arc iGPU / Arc dGPU — SYCL via Level Zero
    Sycl,
    /// Intel AI Boost NPU — OpenVINO
    Npu,
    /// Auto-detect
    Auto,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Cpu  => write!(f, "cpu"),
            Backend::Sycl => write!(f, "sycl"),
            Backend::Npu  => write!(f, "npu"),
            Backend::Auto => write!(f, "auto"),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    /// TCP port for the Rust server
    #[serde(default = "default_port")]
    pub port: u16,

    /// Absolute or relative path to llama-server.exe (SYCL / CPU)
    #[serde(default = "default_server_path")]
    pub llama_server_path: PathBuf,

    /// Absolute or relative path to llama-cli.exe
    #[serde(default = "default_cli_path")]
    pub llama_cli_path: PathBuf,

    /// Absolute or relative path to llama-cli-npu.exe
    #[serde(default = "default_npu_path")]
    pub llama_cli_npu_path: PathBuf,

    /// Hardware backend
    #[serde(default)]
    pub backend: Backend,

    /// Directory scanned for .gguf models
    #[serde(default = "default_models")]
    pub models_dir: PathBuf,

    /// Explicit model to auto-load
    #[serde(default)]
    pub model_path: Option<PathBuf>,

    /// Optional multimodal projector file (auto-resolved next to model_path
    /// when set to "auto" or left derivable from the model directory)
    #[serde(default)]
    pub mmproj_path: Option<PathBuf>,

    /// Concurrent inference slots (= parallel requests)
    #[serde(default = "default_n_slots")]
    pub n_slots: u32,

    /// Context window in tokens
    #[serde(default = "default_ctx")]
    pub n_ctx: u32,

    /// CPU threads for the native llama-server process
    #[serde(default = "default_threads")]
    pub threads: u32,

    /// Reasoning/thinking mode forwarded to the native server
    /// (on | off | auto). Auto preserves existing behavior.
    #[serde(default = "default_reasoning")]
    pub reasoning: String,

    /// Optional: override Level Zero device selector string
    #[serde(default)]
    pub oneapi_device_selector: Option<String>,

    /// Extra env passed to native llama-server child process
    #[serde(default)]
    pub env: HashMap<String, String>,

    /// Python interpreter (Open Interpreter) auto-load settings
    #[serde(default)]
    pub interpreter: InterpreterConfig,
}

/// Python interpreter auto-load settings.
///
/// The Rust server spawns Open Interpreter once the native llama.cpp backend
/// has the model resident, so the model can drive the local environment.
/// `unified/launch_interpreter.py` reads this exact block, keeping the Rust
/// and Python launch paths on one source of truth.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InterpreterConfig {
    /// Launch the interpreter automatically once the model reports ready.
    #[serde(default = "default_true")]
    pub auto_start: bool,

    /// Path to `interpreter.exe` (Open Interpreter).
    /// Empty or missing => auto-detected (venv-interpreter, Python Scripts, PATH).
    #[serde(default)]
    pub path: PathBuf,

    /// Open Interpreter profile name. Empty = rely on the explicit flags below
    /// (the checked-in `local.yaml` is the obsolete 0.2.x schema, so the flags
    /// are the reliable path on Open Interpreter 0.4.x).
    #[serde(default)]
    pub profile: String,

    /// OpenAI-compatible base URL. Empty => derived from `port` at launch.
    #[serde(default)]
    pub api_base: String,

    /// Model id the interpreter requests. Empty => derived from the loaded GGUF.
    #[serde(default)]
    pub model: String,

    /// API key (local llama.cpp ignores it, but the client requires one).
    #[serde(default = "default_interpreter_api_key")]
    pub api_key: String,

    /// Launch the REPL in its own console window (Windows: `cmd /c start`).
    /// Without this the interpreter inherits the headless server's stdio,
    /// hits EOF on stdin and exits immediately.
    #[serde(default = "default_true")]
    pub new_console: bool,

    /// Extra CLI arguments appended verbatim (e.g. "--safe_mode", "ask").
    #[serde(default)]
    pub args: Vec<String>,
}

impl Default for InterpreterConfig {
    fn default() -> Self {
        Self {
            auto_start: default_true(),
            path:       PathBuf::new(),
            profile:    String::new(),
            api_base:    String::new(),
            model:       String::new(),
            api_key:     default_interpreter_api_key(),
            new_console: default_true(),
            args:        Vec::new(),
        }
    }
}

fn default_port()        -> u16 { 8080 }
fn default_true()        -> bool { true }
fn default_interpreter_api_key() -> String { "dummy".into() }
fn default_server_path() -> PathBuf { PathBuf::from(
    "bin/ipex-sycl/llama-cpp-ipex-llm-2.3.0b20250612-win/llama-server.exe"
) }
fn default_cli_path() -> PathBuf { PathBuf::from(
    "bin/ipex-sycl/llama-cpp-ipex-llm-2.3.0b20250612-win/llama-cli.exe"
) }
fn default_npu_path() -> PathBuf {
    PathBuf::from("bin/ipex-npu/llama-cli-npu.exe")
}
fn default_models() -> PathBuf { PathBuf::from("models") }
fn default_n_slots() -> u32 { 4 }
fn default_ctx()    -> u32 { 8192 }
fn default_threads() -> u32 { 8 }
fn default_reasoning() -> String { "auto".into() }

impl Default for ServerConfig {
    fn default() -> Self {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap());

        Self {
            port:               default_port(),
            llama_server_path:  exe_dir.join(default_server_path()),
            llama_cli_path:     exe_dir.join(default_cli_path()),
            llama_cli_npu_path: exe_dir.join(default_npu_path()),
            backend:            Backend::default(),
            models_dir:         exe_dir.join("models"),
            model_path:         None,
            mmproj_path:        None,
            n_slots:            default_n_slots(),
            n_ctx:              default_ctx(),
            threads:            default_threads(),
            reasoning:          default_reasoning(),
            oneapi_device_selector: None,
            env:                HashMap::new(),
            interpreter:        InterpreterConfig::default(),
        }
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    /// TOML parse failure
    #[error("TOML parse error: {0}")]
    Parse(#[from] toml::de::Error),
    /// File I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Best-effort lookup of `interpreter.exe` (Open Interpreter).
///
/// Order: the dedicated venv, the system Python Scripts dir, then PATH.
fn detect_interpreter() -> Option<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()
        .map(PathBuf::from);

    if let Some(ref home) = home {
        let candidates = [
            home.join("venv-interpreter").join("Scripts").join("interpreter.exe"),
            home.join("AppData/Local/Programs/Python/Python314/Scripts/interpreter.exe"),
            home.join(".local/bin/interpreter"),
        ];
        for c in candidates {
            if c.exists() {
                return Some(c);
            }
        }
    }

    which::which("interpreter").ok()
}

impl ServerConfig {
    /// Load `config.toml`: current working directory first, then the
    /// executable's directory (2026-09-12 fix — launching from anywhere
    /// else silently fell back to defaults with unusable backend paths).
    pub fn load() -> Result<Self, ConfigError> {
        let cwd_path = std::path::Path::new("config.toml");
        let exe_path = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("config.toml")));
        // release builds live in target/release: also try the project root
        // (exe -> release -> target -> project root = three levels up).
        let root_path = std::env::current_exe().ok().and_then(|p| {
            p.parent()
                .and_then(|d| d.parent())
                .and_then(|d| d.parent())
                .map(|d| d.join("config.toml"))
        });
        for path in [Some(cwd_path), exe_path.as_deref(), root_path.as_deref()]
            .into_iter()
            .flatten()
        {
            if path.exists() {
                let raw = std::fs::read_to_string(path)?;
                let cfg: Self = toml::from_str(&raw)?;
                return Ok(cfg);
            }
        }
        Ok(Self::default())
    }

    /// Persist to `config.toml`
    #[allow(dead_code)]
    pub fn save(&self) -> Result<(), ConfigError> {
        let raw = toml::to_string_pretty(self)
            .map_err(|e| ConfigError::Io(std::io::Error::new(
                std::io::ErrorKind::Other, e)))?;
        std::fs::write("config.toml", raw)?;
        Ok(())
    }

    /// Resolve relative paths: executable directory first, then walking
    /// up toward the project root (2026-09-12 fix — release binaries in
    /// target/release/ resolved bin/... against themselves, so the
    /// bundled ipex backends at the project root were never found).
    pub fn resolve_paths(&mut self) {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap());

        fn locate(base: &Path, rel: &Path) -> Option<PathBuf> {
            let mut dir: Option<&Path> = Some(base);
            for _ in 0..5 {
                let d = dir?;
                let cand = d.join(rel);
                if cand.exists() {
                    return Some(cand);
                }
                dir = d.parent();
            }
            None
        }

        let resolve = |p: &mut PathBuf| {
            if p.is_absolute() {
                return;
            }
            if let Some(found) = locate(&exe_dir, p) {
                *p = found;
            } else {
                *p = exe_dir.join(&*p);
            }
        };
        resolve(&mut self.llama_server_path);
        resolve(&mut self.llama_cli_path);
        resolve(&mut self.llama_cli_npu_path);
        if !self.models_dir.is_absolute() {
            self.models_dir = exe_dir.join(&self.models_dir);
        }

        // Interpreter: an explicit path must exist, otherwise fall back to
        // auto-detection so a stale config value cannot silently disable
        // auto-load (2026-09-17 fix — the documented launch_interpreter.py
        // never existed, so nothing ever resolved this).
        if self.interpreter.path.as_os_str().is_empty() || !self.interpreter.path.exists() {
            if let Some(found) = detect_interpreter() {
                self.interpreter.path = found;
            } else if !self.interpreter.path.as_os_str().is_empty() {
                self.interpreter.path = PathBuf::new();
            }
        }

        // Auto-detect mmproj: if model_path is set but mmproj_path is not,
        // look for mmproj.gguf — or mmproj-*.gguf (Jan/LM-Studio naming)
        // — in the same directory.
        if self.mmproj_path.is_none() {
            if let Some(ref model) = self.model_path {
                let direct = model.with_file_name("mmproj.gguf");
                if direct.exists() {
                    self.mmproj_path = Some(direct);
                } else if let Some(parent) = model.parent() {
                    let mut cands: Vec<PathBuf> = std::fs::read_dir(parent)
                        .map(|rd| {
                            rd.filter_map(|e| e.ok())
                                .map(|e| e.path())
                                .filter(|p| {
                                    p.extension().and_then(|s| s.to_str())
                                        == Some("gguf")
                                        && p.file_stem()
                                            .and_then(|s| s.to_str())
                                            .map_or(false, |s| {
                                                s.starts_with("mmproj")
                                            })
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    cands.sort();
                    if let Some(first) = cands.into_iter().next() {
                        self.mmproj_path = Some(first);
                    }
                }
            }
        }
    }
}
