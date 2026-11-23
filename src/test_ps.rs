// Test ps_script behavior directly
use std::process::Command;
use std::sync::Once;

static INIT: Once = Once::new();

fn main() {
    INIT.call_once(|| {
        eprintln!("=== ps_script TEST ===");
        
        // What Rust passes to ps_script:
        let script = r#"
$cpu = Get-WmiObject Win32_Processor | Select-Object -First 1
$features = @()
$cpu.ProcessorId -match 'GenuineIntel(.*)'
$data = @{
    vendor        = $cpu.Manufacturer
    vendor_id     = 'GenuineIntel'
    model_name    = $cpu.Name
    family        = [int]($cpu.DataWidth / 64)
    model_number  = $cpu.ProcessorId -replace '.*Model=([0-9A-F]+).*', '$1'
    threads       = [int]$cpu.NumberOfCores
}
$data_json = $data | ConvertTo-Json
Write-Output $data_json
"#;
        
        eprintln!("Script after strip_prefix: {:?}", script.strip_prefix('\n').unwrap_or(script));
        
        let s = script.strip_prefix('\n').unwrap_or(script);
        let fuel: Vec<u8> = s.encode_utf16().flat_map(|c| c.to_le_bytes()).collect();
        // Decode fuel[0..20] as UTF-16LE chars
        let chars: String = fuel.chunks(2)
            .map(|c| char::from_u16(u16::from_le_bytes([c[0], c[1]])).unwrap_or('?'))
            .collect();
        eprintln!("First 80 UTF-16 chars: {}", &chars[..chars.len().min(80)]);
        let each: String = fuel.iter()
            .enumerate()
            .take(20)
            .map(|(i, b)| format!("  [{}]=0x{:02X}", i, b))
            .collect();
        eprintln!("First 20 UTF-16LE bytes:{}", each);
        
        let mut cmd = Command::new("powershell");
        cmd.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand",
            &hex::encode(fuel)]);
        eprintln!("Running PS command...");
        let o = cmd.output().unwrap();
        eprintln!("status: {:?}", o.status);
        eprintln!("stdout ({} bytes): {:?}", o.stdout.len(), String::from_utf8_lossy(&o.stdout));
        eprintln!("stderr ({} bytes): {:?}", o.stderr.len(), String::from_utf8_lossy(&o.stderr));
    });
}
