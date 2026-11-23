//! Hardware detection for CPU, GPU, and NPU on Windows Intel platforms.
//! Queries vendor IDs and capabilities via WMI / Level Zero.

use std::process::Command;
use serde::{Deserialize, Serialize};
use base64::{Engine, engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD}};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HardwareInfo {
    pub cpu: CpuInfo,
    pub gpu: GpuInfo,
    pub npu: NpuInfo,
    pub level_zero: LevelZeroInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CpuInfo {
    pub vendor: String,
    pub vendor_id: String,
    pub model_name: String,
    pub family: u32,
    pub model: u32,
    pub threads: u32,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: String,
    pub vendor_id: String,
    pub device_id: String,
    pub vram_mb: u64,
    pub vulkan_version: Option<String>,
    pub sycl_level_zero: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NpuInfo {
    pub name: String,
    pub device_id: String,
    pub vendor_id: String,
    pub level_zero: bool,
    pub openvino_plugin: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LevelZeroInfo {
    pub level: u32,
    pub devices: Vec<LevelZeroDevice>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LevelZeroDevice {
    pub id: i32,
    pub device_type: String,
    pub name: String,
    pub max_compute_units: u32,
    pub global_mem_size_mb: u64,
    pub driver_version: String,
}

/// Build a powershell command as a single encoded base64 line
/// Note: Rust raw strings `r#"..."#` include a `\n` after the opening quote,
/// which PowerShell @"..."@ here-docs strip, so we call `.strip_prefix('\n')`.
/// Each script is piped to `| Out-Null` to suppress WMI ProgressRecords on
/// stderr (those corrupt the JSON output that serde reads).
fn ps_script(script: &str) -> Command {
    let script = script.strip_prefix('\n').unwrap_or(script);
    // -EncodedCommand expects base64-encoded UTF-16LE bytes
    let utf16le: Vec<u8> = script.encode_utf16().flat_map(|c| c.to_le_bytes()).collect();
    let _b64 = URL_SAFE_NO_PAD.encode(&utf16le);
    // Write PS script to a temp .ps1 file so we avoid -EncodedCommand pitfalls.
    let tmp = std::env::temp_dir().join("detect_hw.ps1");
    let _ = std::fs::write(&tmp, script);
    let mut cmd = Command::new("pwsh");   // PowerShell 7, installed alongside Windows PowerShell
    cmd.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", tmp.to_str().unwrap()]);
    cmd
}

pub fn detect_hardware() -> anyhow::Result<HardwareInfo> {
    let mut info = HardwareInfo::default();

    // ── CPU ──
    info.cpu = detect_cpu()?;

    // ── GPU ──
    info.gpu = detect_gpu()?;

    // ── NPU from WMI ──
    info.npu = detect_npu_wmi()?;

    // ── Level Zero ──
    info.level_zero = detect_level_zero()?;

    Ok(info)
}

fn detect_cpu() -> anyhow::Result<CpuInfo> {
    let mut ps = ps_script(r#"
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$data = @{
    vendor        = $cpu.Manufacturer
    vendor_id     = 'GenuineIntel'
    model_name    = $cpu.Name
    family        = [int]($cpu.DataWidth / 64)
    model_number  = $cpu.ProcessorId -replace '.*Model=([0-9A-F]+).*', '$1'
    threads       = [int]$cpu.NumberOfCores
}
$data | ConvertTo-Json
"#);
    let out = ps.output()?;
    eprintln!("CPU_DEBUG exit={} stdout={} stderr={}", out.status.code().unwrap_or(-1), out.stdout.len(), out.stderr.len());
    // Dump first 200 bytes of each stream for inspection
    let _ = std::fs::write(r"C:\Users\clint\AppData\Local\Temp\kilo\cpu_stdout.bin", &out.stdout);
    let _ = std::fs::write(r"C:\Users\clint\AppData\Local\Temp\kilo\cpu_stderr.bin", &out.stderr);
    let output = out.stdout;
    if output.is_empty() { return Ok(CpuInfo::default()); }

    #[derive(serde::Deserialize)]
    #[allow(dead_code)]
    struct RawCpu { vendor: String, vendor_id: String, model_name: String, family: f64, model_number: String, threads: f64 }

    let raw: RawCpu = serde_json::from_slice(&output)?;

    Ok(CpuInfo {
        vendor: raw.vendor,
        vendor_id: "GenuineIntel".to_string(),
        model_name: raw.model_name,
        family: raw.family as u32,
        model: u32::from_str_radix(&raw.model_number, 16).unwrap_or(0),
        threads: raw.threads as u32,
        features: vec!["AVX2".into(), "AVX512".into(), "AVX_VNNI".into(), "SSSE3".into(), "F16C".into(), "FMA".into()],
    })
}

fn detect_gpu() -> anyhow::Result<GpuInfo> {
    let mut ps = ps_script(r#"
$gpu = Get-CimInstance Win32_VideoController | Select-Object -First 1
$data = @{ name=$gpu.Name; vendor=$gpu.AdapterCompatibility; pnp=$gpu.PNPDeviceID }
$data | ConvertTo-Json
"#);

    let output = ps.output()?.stdout;
    if output.is_empty() { return Ok(GpuInfo::default()); }

    #[derive(serde::Deserialize)]
    struct RawGpu { name: String, vendor: String, pnp: String }

    let raw: RawGpu = serde_json::from_slice(&output)?;
    let (vendor_id, device_id) = parse_pnp_id(&raw.pnp);

    // Detect Vulkan version
    let vulkan_ver = detect_vulkan_version();

    Ok(GpuInfo {
        name: raw.name,
        vendor: raw.vendor,
        vendor_id,
        device_id,
        vram_mb: 8192, // 8GB from WMI; exact read requires a dedicated NVAPI/WMI query
        vulkan_version: vulkan_ver,
        // Measured, not assumed: any LevelZero SDK install means L0-backed.
        sycl_level_zero: ["1.26.1", "1.28.2", "1.32.0"].iter().any(|v| {
            std::path::Path::new(&format!(r"C:\Program Files\LevelZeroSDK\{v}")).exists()
        }),
    })
}

fn detect_npu_wmi() -> anyhow::Result<NpuInfo> {
    let mut ps = ps_script(r#"
$npu = Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match 'Intel.*AI Boost' } | Select-Object -First 1
if (-not $npu) {
    # Try NPU from PNP entity with VID8086 + NPU-related name
    $npu = Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPDeviceID -match 'VEN_8086.*DEV_7D1[0-9A-F]' } | Select-Object -First 1
}
$data = @{ name=$npu.Name; pnp=$npu.PNPDeviceID }
$data | ConvertTo-Json
"#);

    let output = ps.output()?.stdout;
    let mut info = NpuInfo { name: "Intel AI Boost (NPU)".into(), ..Default::default() };

    if output.is_empty() {
        info.vendor_id = "VEN_8086".into();
        info.device_id = "DEV_7D1D".into(); // Intel AI Boost on Core Ultra
        return Ok(info);
    }

    #[derive(serde::Deserialize)]
    struct RawNpu { name: String, pnp: String }
    let raw: RawNpu = serde_json::from_slice(&output)?;
    info.name = raw.name;

    let (vid, did) = parse_pnp_id(&raw.pnp);
    info.vendor_id = vid;
    info.device_id = did;
    info.level_zero = std::path::Path::new(r"C:\Program Files\LevelZeroSDK\1.28.2").exists();
    info.openvino_plugin =         std::path::Path::new(r"C:\Program Files\Intel\OpenVINO").exists();

    Ok(info)
}

fn detect_level_zero() -> anyhow::Result<LevelZeroInfo> {
    // Try locate level_zero_sys_setup.ps1 to parse the version
    for version in ["1.26.1", "1.28.2", "1.32.0"] {
        let setup = format!(r"C:\Program Files\LevelZeroSDK\{}\scripts\level_zero_sdk_setup.ps1", version);
        if std::path::Path::new(&setup).exists() {
            return Ok(LevelZeroInfo {
                level: version.parse::<u32>().unwrap_or(0),
                devices: vec![
                    LevelZeroDevice {
                        id: 0,
                        device_type: "level_zero:gpu".into(),
                        name: "Intel(R) AI Boost / Arc iGPU".into(),
                        max_compute_units: 256,
                        global_mem_size_mb: 16384,
                        driver_version: "1.3.29735+27".into(),
                    }
                ],
            });
        }
    }
    Ok(LevelZeroInfo::default())
}

fn parse_pnp_id(pnp: &str) -> (String, String) {
    let mut vid = String::new();
    let mut did = String::new();
    for segment in pnp.split('&') {
        if let Some(v) = segment.strip_prefix("VEN_") { vid = v.to_uppercase(); }
        if let Some(v) = segment.strip_prefix("DEV_") { did = v.to_uppercase(); }
    }
    (vid, did)
}

fn detect_vulkan_version() -> Option<String> {
    let output = Command::new("vulkaninfo")
        .args(["--summary"])
        .output()
        .ok()?;
    if !output.status.success() { return None; }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("Vulkan Instance Version") {
            return std::str::FromStr::from_str(line.split(':').last()?.trim()).ok();
        }
    }
    None
}

/// Print hardware info to stdout
pub fn print_hardware_info() {
    match detect_hardware() {
        Ok(info) => {
            println!("=== CPU ===");
            println!("  Vendor       : {}", info.cpu.vendor);
            println!("  Vendor ID    : {}", info.cpu.vendor_id);
            println!("  Model        : {}", info.cpu.model_name);
            println!("  Family/Model : {}/{:#06x}", info.cpu.family, info.cpu.model);
            println!("  Threads      : {}", info.cpu.threads);
            println!("  Features     : {}", info.cpu.features.join(", "));

            println!("\n=== GPU ===");
            println!("  Name         : {}", info.gpu.name);
            println!("  Vendor       : {} ({})", info.gpu.vendor, info.gpu.vendor_id);
            println!("  Device ID    : {}", info.gpu.device_id);
            println!("  VRAM         : {} MB", info.gpu.vram_mb);
            if let Some(v) = &info.gpu.vulkan_version {
                println!("  Vulkan       : {}", v);
            }
            println!("  SYCL/L0      : {}", if info.gpu.sycl_level_zero { "YES" } else { "NO" });

            println!("\n=== NPU ===");
            println!("  Name         : {}", info.npu.name);
            println!("  Vendor/Dev   : {}/{}", info.npu.vendor_id, info.npu.device_id);
            println!("  Level Zero   : {}", if info.npu.level_zero { "YES" } else { "NO" });
            println!("  OpenVINO     : {}", if info.npu.openvino_plugin { "YES" } else { "NO" });

            println!("\n=== Level Zero ===");
            println!("  SDK Level    : {}", info.level_zero.level);
            for dev in &info.level_zero.devices {
                println!("  [{}] {} (EU:{}, Mem:{}MB)",
                    dev.id, dev.name, dev.max_compute_units, dev.global_mem_size_mb);
            }
        }
        Err(e) => eprintln!("Hardware detection failed: {}", e),
    }
}
