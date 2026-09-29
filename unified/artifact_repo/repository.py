"""Repository business logic — scan, pick, tag, launch, train."""
import os, sys, json, time, subprocess
from pathlib import Path
from typing import Optional

from .database import ArtifactDatabase

BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path.home() / '.config' / 'unified-llm'
MODEL_CACHE = CONFIG_DIR / 'models.json'
USE_MODEL_DAT = Path.home() / '.lmstudio' / 'models' / 'use_model.dat'


class ArtifactRepository:
    def __init__(self, db: Optional[ArtifactDatabase] = None):
        self._db = db or ArtifactDatabase()

    @property
    def db(self) -> ArtifactDatabase:
        return self._db

    # ── Scanning ────────────────────────────────────────────────────────

    def scan_and_import(self, force: bool = False) -> int:
        """Run model_scanner, import results into DB. Returns count of new/changed."""
        scanner = BASE_DIR / 'model_scanner.py'
        if not scanner.exists():
            return 0
        out = subprocess.run(
            [sys.executable, str(scanner), '--json'],
            capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            return 0
        try:
            data = json.loads(out.stdout)
        except json.JSONDecodeError:
            return 0
        imported = 0
        for m in data.get('models', []):
            meta = m.get('metadata', {})
            mid = self._db.upsert_model(
                path=m['path'],
                filename=m['filename'],
                size_bytes=m.get('size_bytes', 0),
                size_gb=m.get('size_gb', 0.0),
                format='gguf',
                architecture=meta.get('architecture', ''),
                block_count=meta.get('block_count', 0),
                context_length=meta.get('context_length', 0),
                quantization=str(meta.get('quantization', '')),
                mmproj=meta.get('mmproj', ''),
                tensor_count=meta.get('tensor_count', 0),
            )
            if meta.get('mmproj'):
                self._db.add_tag(mid, 'vision')
            if meta.get('architecture') in ('gemma4',):
                self._db.add_tag(mid, 'gemma-4')
            if meta.get('architecture', '').startswith('qwen'):
                self._db.add_tag(mid, 'qwen')
            imported += 1
        return imported

    def list_models(self, tag: str = '', search: str = '', sort: str = 'updated') -> list[dict]:
        return self._db.list_models(tag=tag, search=search, sort=sort)

    def get_model(self, model_id: int) -> Optional[dict]:
        return self._db.get_model(model_id)

    def get_model_by_path(self, path: str) -> Optional[dict]:
        return self._db.get_model_by_path(path)

    # ── Selection / Launch ──────────────────────────────────────────────

    def select_and_launch(self, model_id: int, port: int = 8080) -> dict:
        """Mark model as selected and launch via router+executor."""
        model = self._db.get_model(model_id)
        if not model:
            return {'error': 'Model not found'}
        # Write use_model.dat
        os.makedirs(os.path.dirname(USE_MODEL_DAT), exist_ok=True)
        with open(USE_MODEL_DAT, 'w', encoding='ascii') as f:
            f.write(f"{model['path']}|{model['context_length']}|{model['filename']}|{model['size_gb']} GB|{model['filename']}")
        self._db.set_last_used(model_id)
        # Launch via router
        sys.path.insert(0, str(BASE_DIR))
        try:
            from executor import LlamaServerExecutor, wait_for_health
            out = subprocess.run(
                [sys.executable, str(BASE_DIR / 'router.py'),
                 '--json', f'--model={model["path"]}', f'--port={port}', '--task=chat'],
                capture_output=True, text=True, timeout=15)
            plan = json.loads(out.stdout)
            if 'error' in plan:
                return plan
            executor = LlamaServerExecutor()
            pid = executor.start(plan['args'], plan.get('env'))
            if pid and pid > 0:
                ok = wait_for_health(port, executor, timeout=120)
                return {'port': port, 'pid': pid, 'healthy': ok, 'model': model['filename']}
            return {'error': 'Failed to start'}
        except Exception as e:
            return {'error': str(e)}

    def launch_training(self, model_id: int, method: str = 'sft', dataset: str = '',
                        config: dict = None) -> dict:
        """Queue a training run. The actual training happens via trainer.py."""
        model = self._db.get_model(model_id)
        if not model:
            return {'error': 'Model not found'}
        run_id = self._db.add_training_run(
            base_model_id=model_id, method=method, dataset=dataset, config=config)
        # Launch training in background
        trainer = BASE_DIR / 'artifact_repo' / 'trainer.py'
        subprocess.Popen(
            [sys.executable, str(trainer), '--run', str(run_id)],
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )
        return {'run_id': run_id, 'status': 'launched'}

    # ── Tags ────────────────────────────────────────────────────────────

    def list_tags(self) -> list[str]:
        return self._db.list_tags()

    def add_tag(self, model_id: int, tag: str):
        self._db.add_tag(model_id, tag)

    def remove_tag(self, model_id: int, tag: str):
        self._db.remove_tag(model_id, tag)

    def set_rating(self, model_id: int, rating: int):
        self._db.set_rating(model_id, rating)

    def set_notes(self, model_id: int, notes: str):
        self._db.set_notes(model_id, notes)

    # ── Adapters ────────────────────────────────────────────────────────

    def list_adapters(self, base_model_id: int = 0) -> list[dict]:
        return self._db.list_adapters(base_model_id)

    def add_adapter(self, name: str, path: str, base_model_id: int,
                    method: str = 'lora', rank: int = 16, config: dict = None) -> int:
        return self._db.add_adapter(name, path, base_model_id, method, rank, config)

    # ── Training Runs ───────────────────────────────────────────────────

    def list_training_runs(self, base_model_id: int = 0) -> list[dict]:
        return self._db.list_training_runs(base_model_id)

    # ── Stats ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'models': len(self._db.list_models()),
            'tags': len(self._db.list_tags()),
            'adapters': len(self._db.list_adapters()),
            'training_runs': len(self._db.list_training_runs()),
            'datasets': len(self._db.list_datasets()),
        }
