"""SQLite-backed artifact repository database."""
import sqlite3, json, time
from pathlib import Path
from typing import Optional

REPO_DIR = Path.home() / '.config' / 'unified-llm' / 'artifact_repo'
DB_PATH = REPO_DIR / 'repository.db'

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT UNIQUE NOT NULL,
    filename    TEXT NOT NULL,
    size_bytes  INTEGER,
    size_gb     REAL,
    format      TEXT DEFAULT 'gguf',
    architecture TEXT DEFAULT '',
    block_count INTEGER DEFAULT 0,
    context_length INTEGER DEFAULT 0,
    quantization TEXT DEFAULT '',
    mmproj      TEXT DEFAULT '',
    tensor_count INTEGER DEFAULT 0,
    rating      INTEGER DEFAULT 0,
    notes       TEXT DEFAULT '',
    last_used   TEXT,
    created     TEXT DEFAULT (datetime('now')),
    updated     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_tags (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    tag      TEXT NOT NULL,
    UNIQUE(model_id, tag)
);

CREATE TABLE IF NOT EXISTS adapters (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    path          TEXT,
    base_model_id INTEGER REFERENCES models(id),
    method        TEXT DEFAULT 'lora',
    rank          INTEGER DEFAULT 16,
    config        TEXT DEFAULT '{}',
    metrics       TEXT DEFAULT '{}',
    created       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS training_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    base_model_id INTEGER REFERENCES models(id),
    adapter_id    INTEGER REFERENCES adapters(id),
    dataset       TEXT DEFAULT '',
    method        TEXT DEFAULT 'sft',
    config        TEXT DEFAULT '{}',
    status        TEXT DEFAULT 'pending',
    metrics       TEXT DEFAULT '{}',
    created       TEXT DEFAULT (datetime('now')),
    updated       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    path        TEXT DEFAULT '',
    source      TEXT DEFAULT 'local',
    config      TEXT DEFAULT '{}',
    size_bytes  INTEGER DEFAULT 0,
    record_count INTEGER DEFAULT 0,
    created     TEXT DEFAULT (datetime('now'))
);
"""


class ArtifactDatabase:
    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        REPO_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self):
        self._conn.close()

    # ── Models ──────────────────────────────────────────────────────────

    def upsert_model(self, path: str, **kw) -> int:
        existing = self._conn.execute(
            "SELECT id FROM models WHERE path = ?", (path,)).fetchone()
        kw['path'] = path
        kw['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        if existing:
            sets = ', '.join(f"{k}=?" for k in kw)
            vals = list(kw.values()) + [existing['id']]
            self._conn.execute(f"UPDATE models SET {sets} WHERE id=?", vals)
            return existing['id']
        else:
            cols = ', '.join(kw.keys())
            plc = ', '.join('?' * len(kw))
            cur = self._conn.execute(f"INSERT INTO models ({cols}) VALUES ({plc})", list(kw.values()))
            return cur.lastrowid

    def get_model(self, model_id: int) -> Optional[dict]:
        r = self._conn.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
        return dict(r) if r else None

    def get_model_by_path(self, path: str) -> Optional[dict]:
        r = self._conn.execute("SELECT * FROM models WHERE path=?", (path,)).fetchone()
        return dict(r) if r else None

    def list_models(self, tag: str = '', search: str = '', sort: str = 'updated') -> list[dict]:
        q = "SELECT m.*, GROUP_CONCAT(t.tag, ',') as tags FROM models m LEFT JOIN model_tags t ON m.id=t.model_id"
        params = []
        clauses = []
        if tag:
            clauses.append("m.id IN (SELECT model_id FROM model_tags WHERE tag=?)")
            params.append(tag)
        if search:
            clauses.append("(m.filename LIKE ? OR m.architecture LIKE ? OR m.notes LIKE ?)")
            params.extend([f'%{search}%'] * 3)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " GROUP BY m.id"
        order = {'name': 'm.filename', 'size': 'm.size_bytes DESC', 'rating': 'm.rating DESC',
                 'ctx': 'm.context_length DESC', 'created': 'm.created DESC', 'updated': 'm.updated DESC'}
        q += f" ORDER BY {order.get(sort, 'm.updated DESC')}"
        return [dict(r) for r in self._conn.execute(q, params).fetchall()]

    def delete_model(self, model_id: int):
        self._conn.execute("DELETE FROM models WHERE id=?", (model_id,))
        self._conn.commit()

    def set_rating(self, model_id: int, rating: int):
        self._conn.execute("UPDATE models SET rating=?, updated=datetime('now') WHERE id=?",
                           (max(0, min(5, rating)), model_id))
        self._conn.commit()

    def set_notes(self, model_id: int, notes: str):
        self._conn.execute("UPDATE models SET notes=?, updated=datetime('now') WHERE id=?", (notes, model_id))
        self._conn.commit()

    def set_last_used(self, model_id: int):
        self._conn.execute("UPDATE models SET last_used=datetime('now') WHERE id=?", (model_id,))
        self._conn.commit()

    # ── Tags ────────────────────────────────────────────────────────────

    def add_tag(self, model_id: int, tag: str):
        try:
            self._conn.execute("INSERT INTO model_tags (model_id, tag) VALUES (?,?)", (model_id, tag.lower()))
            self._conn.commit()
        except sqlite3.IntegrityError:
            pass

    def remove_tag(self, model_id: int, tag: str):
        self._conn.execute("DELETE FROM model_tags WHERE model_id=? AND tag=?", (model_id, tag.lower()))
        self._conn.commit()

    def list_tags(self) -> list[str]:
        return [r['tag'] for r in self._conn.execute(
            "SELECT tag, COUNT(*) as cnt FROM model_tags GROUP BY tag ORDER BY cnt DESC").fetchall()]

    # ── Adapters ────────────────────────────────────────────────────────

    def add_adapter(self, name: str, path: str, base_model_id: int, method: str = 'lora', rank: int = 16, config: dict = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO adapters (name, path, base_model_id, method, rank, config) VALUES (?,?,?,?,?,?)",
            (name, path, base_model_id, method, rank, json.dumps(config or {})))
        self._conn.commit()
        return cur.lastrowid

    def list_adapters(self, base_model_id: int = 0) -> list[dict]:
        if base_model_id:
            return [dict(r) for r in self._conn.execute(
                "SELECT * FROM adapters WHERE base_model_id=? ORDER BY created DESC", (base_model_id,)).fetchall()]
        return [dict(r) for r in self._conn.execute("SELECT * FROM adapters ORDER BY created DESC").fetchall()]

    # ── Training Runs ───────────────────────────────────────────────────

    def add_training_run(self, base_model_id: int, method: str = 'sft', dataset: str = '',
                         config: dict = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO training_runs (base_model_id, method, dataset, config) VALUES (?,?,?,?)",
            (base_model_id, method, dataset, json.dumps(config or {})))
        self._conn.commit()
        return cur.lastrowid

    def update_training_run(self, run_id: int, **kw):
        kw['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        sets = ', '.join(f"{k}=?" for k in kw)
        self._conn.execute(f"UPDATE training_runs SET {sets} WHERE id=?", list(kw.values()) + [run_id])
        self._conn.commit()

    def list_training_runs(self, base_model_id: int = 0) -> list[dict]:
        if base_model_id:
            return [dict(r) for r in self._conn.execute(
                "SELECT * FROM training_runs WHERE base_model_id=? ORDER BY created DESC", (base_model_id,)).fetchall()]
        return [dict(r) for r in self._conn.execute("SELECT * FROM training_runs ORDER BY created DESC").fetchall()]

    # ── Datasets ────────────────────────────────────────────────────────

    def add_dataset(self, name: str, path: str = '', source: str = 'local',
                    config: dict = None, record_count: int = 0) -> int:
        cur = self._conn.execute(
            "INSERT INTO datasets (name, path, source, config, record_count) VALUES (?,?,?,?,?)",
            (name, path, source, json.dumps(config or {}), record_count))
        self._conn.commit()
        return cur.lastrowid

    def list_datasets(self) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM datasets ORDER BY created DESC").fetchall()]
