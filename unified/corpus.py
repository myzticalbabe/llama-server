"""corpus.py — shared corpus-scan constants + canonical embedder-stack doc.

Two embedder stacks exist and their indexes are NOT interchangeable
(different models, dims, files). Do not "convert" between them — re-index.

Canonical stacks:
    search   mxbai-embed-large-v1 (1024-d) via semantic.py
             index: ~/llama-server/embeddings/semantic_index.jsonl
             use for: BM25+vector hybrid search + rerank (test_search.py).
    longctx  Qwen3.5-9B (4096-d) via batch_embed.py / direct_embed.py
             index: ~/llama-server/embeddings/embeddings.jsonl
             use for: long chunks that clip mxbai's 512-token window.

Rule: new scan code imports TEXT_EXTS / SKIP_DIRS from here so all three
stacks agree on the file set. Stack-specific MODEL_DIRS stay per-script.
"""
from __future__ import annotations

TEXT_EXTS = {'.py', '.txt', '.md', '.json', '.yaml', '.yml', '.toml', '.cfg',
             '.ini', '.conf', '.log', '.csv', '.xml', '.html', '.css', '.js',
             '.ts', '.jsx', '.tsx', '.sh', '.ps1', '.bat', '.env', '.rst',
             '.c', '.cpp', '.h', '.hpp', '.rs', '.go', '.java', '.kt', '.scala',
             '.sql', '.r', '.m', '.swift', '.dart', '.lua', '.rb', '.pl', '.php',
             '.svelte', '.vue'}
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea',
             '.vscode', 'bin', 'obj', '.cache', '.local', '.config'}

STACKS = {
    "search": {
        "model": "mxbai-embed-large-v1-Q4_k_m.gguf",
        "dim": 1024,
        "script": "semantic.py",
        "index": "llama-server/embeddings/semantic_index.jsonl",
    },
    "longctx": {
        "model": "Qwen3.5-9B-Q4_K_M.gguf",
        "dim": 4096,
        "script": "batch_embed.py / direct_embed.py",
        "index": "llama-server/embeddings/embeddings.jsonl",
    },
}
