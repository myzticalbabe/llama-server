//! llama-metadata — prints GGUF model metadata to stdout.
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Args { model: PathBuf }

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("Model: {:?}", args.model);
    Ok(())
}
