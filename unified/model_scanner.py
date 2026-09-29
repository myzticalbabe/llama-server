import os, sys, json, datetime
from pathlib import Path
from typing import Optional

CACHE_FILE = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'unified-llm' / 'models.json'
MTIME_FILE = CACHE_FILE.parent / 'models_mtime.json'

MODEL_DIRS = [
    Path.home() / 'llama-server' / 'models',
    Path.home() / '.lmstudio' / 'models',
]

def find_gguf_files(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith('.gguf'):
                yield Path(dirpath) / f

def read_gguf_meta(path):
    import gguf
    r = gguf.GGUFReader(str(path), 'r')
    meta = {}
    KEY_MAP = {
        'general.architecture': 'architecture',
        'general.name': 'name',
        'general.basename': 'basename',
        'general.size_label': 'size_label',
        'general.file_type': 'file_type',
        'general.quantization_version': 'quant_version',
        'general.type': 'type',
        'llama.context_length': 'context_length',
        'llama.embedding_length': 'embedding_length',
        'llama.block_count': 'block_count',
        'llama.attention.head_count': 'head_count',
        'llama.attention.head_count_kv': 'head_count_kv',
        'llama.rope.dimension_count': 'rope_dim',
        'llama.feed_forward_length': 'ff_length',
        'llama.vocab_size': 'vocab_size',
        'general.license': 'license',
        'general.repo_url': 'repo_url',
    }
    for gguf_key, our_key in KEY_MAP.items():
        if gguf_key in r.fields:
            try:
                meta[our_key] = r.fields[gguf_key].contents()
            except Exception:
                pass

    # ── Arch-specific fallback for context/embedding/layer keys ──────────────
    # Some architectures use arch-prefixed keys instead of the
    # generic "llama.*" prefix (e.g. gemma4.context_length).
    # Detect the arch from the GGUF and probe for any missing keys.
    arch = meta.get('architecture', '')
    arch_keys = {
        'context_length':   f'{arch}.context_length',
        'embedding_length': f'{arch}.embedding_length',
        'block_count':      f'{arch}.block_count',
        'head_count':       f'{arch}.attention.head_count',
        'head_count_kv':    f'{arch}.attention.head_count_kv',
        'rope_dim':         f'{arch}.rope.dimension_count',
        'ff_length':        f'{arch}.feed_forward_length',
    }
    for our_key, gguf_key in arch_keys.items():
        if our_key not in meta and gguf_key in r.fields:
            try:
                meta[our_key] = r.fields[gguf_key].contents()
            except Exception:
                pass

    if 'block_count' not in meta:
        blk = set()
        for t in r.tensors:
            for part in t.name.split('.'):
                if part.isdigit():
                    blk.add(int(part))
        if blk:
            meta['block_count'] = max(blk)
    meta['tensor_count'] = len(r.tensors)
    for t in r.tensors:
        if 'blk' in t.name:
            meta['quantization'] = str(t.tensor_type)
            break
    for f in path.parent.iterdir():
        if f.suffix != '.gguf' or f.name == path.name:
            continue
        name = f.name
        if name.startswith('mmproj') or '.mmproj' in name:
            meta['mmproj'] = name
            break
    return meta

def load_mtime_cache():
    if MTIME_FILE.exists():
        try: return json.loads(MTIME_FILE.read_text())
        except: pass
    return {}

def save_mtime_cache(data):
    MTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    MTIME_FILE.write_text(json.dumps(data, indent=2))

def load_model_cache():
    if CACHE_FILE.exists():
        try: return json.loads(CACHE_FILE.read_text())
        except: pass
    return {'models': []}

def save_model_cache(data):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data, indent=2, default=str))

def scan():
    mtime_cache = load_mtime_cache()
    cache = load_model_cache()
    cache_models = {m['path']: m for m in cache.get('models', [])}
    models = list(cache_models.values())
    new_or_changed = 0

    for base in MODEL_DIRS:
        base = Path(base).expanduser().resolve()
        if not base.exists():
            continue
        for gguf_path in find_gguf_files(base):
            if gguf_path.suffix != '.gguf':
                continue
            sp = str(gguf_path)
            last_mtime = mtime_cache.get(sp, 0)
            cur_mtime = gguf_path.stat().st_mtime_ns
            if sp not in cache_models and cur_mtime > last_mtime:
                try:
                    meta = read_gguf_meta(gguf_path)
                except Exception as e:
                    # MTP/draft GGUFs (e.g. Qwythos MTP) can crash gguf_reader
                    # tensor reshape — still register by filename so the
                    # selector/router can match them.
                    meta = {'architecture': 'unknown', 'scan_error': str(e)[:200]}
                new_or_changed += 1
                size = gguf_path.stat().st_size
                model = {
                    'path': sp,
                    'filename': gguf_path.name,
                    'size_bytes': size,
                    'size_gb': round(size / 1e9, 2),
                    'format': 'gguf',
                    'metadata': meta,
                }
                models.append(model)
                cache_models[sp] = model
            mtime_cache[sp] = cur_mtime

    data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_count': len(models),
        'new_scanned': new_or_changed,
        'models': models,
    }
    save_model_cache(data)
    save_mtime_cache(mtime_cache)
    return data

def main():
    data = scan()
    if '--json' in sys.argv:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(f'Found {data["model_count"]} models ({data["new_scanned"]} new scanned)')
        for m in data['models']:
            meta = m['metadata']
            name = meta.get('name') or meta.get('architecture') or m['filename']
            arch = meta.get('architecture', '?')
            blk = meta.get('block_count', '?')
            quant = str(meta.get('quantization', '?')).rsplit('.', 1)[-1]
            ctx = meta.get('context_length', '?')
            print(f'  {m["filename"]:<55} {arch:<15} {blk:>3}L  {quant:<10} ctx={ctx:<6} {m["size_gb"]:>5.1f} GB')
            if meta.get('mmproj'):
                print(f'    + mmproj: {meta["mmproj"]}')
        print(f'\nCache: {CACHE_FILE}')

if __name__ == '__main__':
    main()
