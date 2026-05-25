"""SQLite persistence layer for chunk lab."""

import json
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chunk_lab.db"
_INDEXES_DIR = Path(__file__).resolve().parent.parent / "data" / "indexes"
_local = threading.local()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workspaces (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS settings (
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    key          TEXT NOT NULL,
    value        TEXT NOT NULL,
    updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (workspace_id, key)
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    strategy        TEXT NOT NULL,
    eval_mode       TEXT NOT NULL,
    recall_at_k     REAL NOT NULL,
    hits            INTEGER NOT NULL,
    total           INTEGER NOT NULL,
    judge_score     REAL,
    judge_hits      INTEGER,
    judge_pass_rate REAL,
    top_k           INTEGER NOT NULL,
    doc_title       TEXT NOT NULL DEFAULT '',
    run_name        TEXT NOT NULL DEFAULT '',
    qa_source       TEXT NOT NULL DEFAULT 'builtin',
    details_json    TEXT,
    index_id        TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_workspace
    ON eval_runs(workspace_id, created_at DESC);

CREATE TABLE IF NOT EXISTS vector_indexes (
    id                 TEXT PRIMARY KEY,
    workspace_id       TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    strategy           TEXT NOT NULL,
    doc_title          TEXT NOT NULL DEFAULT '',
    run_name           TEXT NOT NULL DEFAULT '',
    chunk_count        INTEGER NOT NULL,
    embedding_provider TEXT NOT NULL,
    embedding_model    TEXT NOT NULL,
    disk_path          TEXT NOT NULL,
    stats_json         TEXT,
    source_eval_id     TEXT,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_vector_indexes_workspace
    ON vector_indexes(workspace_id);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript(_SCHEMA_SQL)
    conn.execute(
        "INSERT OR IGNORE INTO workspaces (id, name) VALUES (?, ?)",
        ("default", "Default Workspace"),
    )
    # Migration: add run_name column if missing (for existing databases)
    try:
        conn.execute("ALTER TABLE eval_runs ADD COLUMN run_name TEXT NOT NULL DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE vector_indexes ADD COLUMN run_name TEXT NOT NULL DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    conn.commit()


# --- Workspace CRUD ---


def list_workspaces() -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name, created_at, updated_at FROM workspaces ORDER BY created_at"
    ).fetchall()
    return [dict(r) for r in rows]


def create_workspace(slug: str, name: str) -> dict[str, Any]:
    conn = get_connection()
    now = _utc_now()
    conn.execute(
        "INSERT INTO workspaces (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (slug, name, now, now),
    )
    conn.commit()
    return {"id": slug, "name": name, "created_at": now, "updated_at": now}


def get_workspace(workspace_id: str) -> dict[str, Any] | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, name, created_at, updated_at FROM workspaces WHERE id = ?",
        (workspace_id,),
    ).fetchone()
    return dict(row) if row else None


def delete_workspace(workspace_id: str) -> bool:
    if workspace_id == "default":
        raise ValueError("不能删除 default workspace")
    conn = get_connection()
    # Clean up index files on disk
    indexes = list_vector_indexes(workspace_id)
    for idx in indexes:
        _remove_index_files(idx["disk_path"])
    conn.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
    conn.commit()
    return True


# --- Settings CRUD ---


def load_settings_overlay(workspace_id: str = "default") -> dict[str, Any]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT key, value FROM settings WHERE workspace_id = ?",
        (workspace_id,),
    ).fetchall()
    overlay: dict[str, Any] = {}
    for row in rows:
        overlay[row["key"]] = json.loads(row["value"])
    return overlay


def save_settings_key(workspace_id: str, key: str, value: Any) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO settings (workspace_id, key, value, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(workspace_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
        (workspace_id, key, json.dumps(value, ensure_ascii=False), _utc_now()),
    )
    conn.commit()


def delete_settings_key(workspace_id: str, key: str) -> None:
    conn = get_connection()
    conn.execute(
        "DELETE FROM settings WHERE workspace_id = ? AND key = ?",
        (workspace_id, key),
    )
    conn.commit()


def clear_settings(workspace_id: str = "default") -> None:
    conn = get_connection()
    conn.execute("DELETE FROM settings WHERE workspace_id = ?", (workspace_id,))
    conn.commit()


# --- Eval runs CRUD ---


def save_eval_run(
    workspace_id: str,
    *,
    strategy: str,
    eval_mode: str,
    recall_at_k: float,
    hits: int,
    total: int,
    top_k: int,
    doc_title: str = "",
    run_name: str = "",
    qa_source: str = "builtin",
    judge_score: float | None = None,
    judge_hits: int | None = None,
    judge_pass_rate: float | None = None,
    details: list[dict] | None = None,
    index_id: str | None = None,
) -> str:
    run_id = uuid.uuid4().hex
    conn = get_connection()
    conn.execute(
        """INSERT INTO eval_runs
           (id, workspace_id, strategy, eval_mode, recall_at_k, hits, total,
            judge_score, judge_hits, judge_pass_rate, top_k, doc_title, run_name,
            qa_source, details_json, index_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id, workspace_id, strategy, eval_mode, recall_at_k, hits, total,
            judge_score, judge_hits, judge_pass_rate, top_k, doc_title, run_name,
            qa_source,
            json.dumps(details, ensure_ascii=False) if details else None,
            index_id, _utc_now(),
        ),
    )
    conn.commit()
    return run_id


def list_eval_runs(
    workspace_id: str = "default",
    *,
    strategy: str | None = None,
    run_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    conn = get_connection()
    sql = "SELECT * FROM eval_runs WHERE workspace_id = ?"
    params: list[Any] = [workspace_id]
    if strategy:
        sql += " AND strategy = ?"
        params.append(strategy)
    if run_name is not None:
        sql += " AND run_name = ?"
        params.append(run_name)
    sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = conn.execute(sql, params).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d.pop("details_json", None)
        results.append(d)
    return results


def list_eval_run_names(workspace_id: str = "default") -> list[dict[str, Any]]:
    """List distinct run_names with summary stats (for task-based grouping)."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT run_name,
                  COUNT(*) as run_count,
                  MAX(recall_at_k) as best_recall,
                  MIN(created_at) as first_run_at,
                  MAX(created_at) as last_run_at
           FROM eval_runs
           WHERE workspace_id = ?
           GROUP BY run_name
           ORDER BY last_run_at DESC""",
        (workspace_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_eval_run(run_id: str) -> dict[str, Any] | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    if d.get("details_json"):
        d["details"] = json.loads(d["details_json"])
    d.pop("details_json", None)
    return d


def delete_eval_run(run_id: str) -> bool:
    conn = get_connection()
    cur = conn.execute("DELETE FROM eval_runs WHERE id = ?", (run_id,))
    conn.commit()
    return cur.rowcount > 0


# --- Vector indexes CRUD ---


def get_index_dir(workspace_id: str, index_id: str) -> Path:
    return _INDEXES_DIR / workspace_id / index_id


def save_vector_index(
    workspace_id: str,
    *,
    strategy: str,
    doc_title: str,
    run_name: str = "",
    chunk_count: int,
    embedding_provider: str,
    embedding_model: str,
    stats: dict | None = None,
    source_eval_id: str | None = None,
) -> tuple[str, Path]:
    """Register a vector index record and return (index_id, disk_path)."""
    index_id = uuid.uuid4().hex[:12]
    disk_path = get_index_dir(workspace_id, index_id)
    disk_path.mkdir(parents=True, exist_ok=True)
    rel_path = str(disk_path.relative_to(_DB_PATH.parent.parent))

    conn = get_connection()
    conn.execute(
        """INSERT INTO vector_indexes
           (id, workspace_id, strategy, doc_title, run_name, chunk_count,
            embedding_provider, embedding_model, disk_path, stats_json,
            source_eval_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            index_id, workspace_id, strategy, doc_title, run_name, chunk_count,
            embedding_provider, embedding_model, rel_path,
            json.dumps(stats, ensure_ascii=False) if stats else None,
            source_eval_id, _utc_now(),
        ),
    )
    conn.commit()
    return index_id, disk_path


def update_vector_index_eval(index_id: str, source_eval_id: str) -> None:
    """Link a vector index to its eval run after the run is saved."""
    conn = get_connection()
    conn.execute(
        "UPDATE vector_indexes SET source_eval_id = ? WHERE id = ?",
        (source_eval_id, index_id),
    )
    conn.commit()


def list_vector_indexes(workspace_id: str = "default") -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM vector_indexes WHERE workspace_id = ? ORDER BY created_at DESC",
        (workspace_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_vector_index(index_id: str) -> dict[str, Any] | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM vector_indexes WHERE id = ?", (index_id,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    if d.get("stats_json"):
        d["stats"] = json.loads(d["stats_json"])
    d.pop("stats_json", None)
    return d


def delete_vector_index(index_id: str) -> bool:
    conn = get_connection()
    row = conn.execute(
        "SELECT disk_path FROM vector_indexes WHERE id = ?", (index_id,)
    ).fetchone()
    if not row:
        return False
    _remove_index_files(row["disk_path"])
    conn.execute("DELETE FROM vector_indexes WHERE id = ?", (index_id,))
    conn.commit()
    return True


def clear_vector_indexes(workspace_id: str = "default") -> int:
    """Delete all vector indexes in a workspace. Returns count deleted."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, disk_path FROM vector_indexes WHERE workspace_id = ?",
        (workspace_id,),
    ).fetchall()
    for row in rows:
        _remove_index_files(row["disk_path"])
    cur = conn.execute(
        "DELETE FROM vector_indexes WHERE workspace_id = ?", (workspace_id,)
    )
    conn.commit()
    return cur.rowcount


def resolve_index_path(index_id: str) -> Path | None:
    info = get_vector_index(index_id)
    if not info:
        return None
    rel = info["disk_path"]
    full = _DB_PATH.parent.parent / rel
    if full.is_dir():
        return full
    return None


def _remove_index_files(rel_path: str) -> None:
    full = _DB_PATH.parent.parent / rel_path
    if full.is_dir():
        shutil.rmtree(full, ignore_errors=True)
