"""Experiment store (SPEC §9): SQLite metadata plus content-addressed blobs (traces, workspaces)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL, config_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    git_rev TEXT, harness_version TEXT, started_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
    history INTEGER, arm TEXT, split TEXT, checkpoint INTEGER, ordinal INTEGER,
    task_id TEXT, family TEXT, cause TEXT, theme TEXT, seed INTEGER, task_hash TEXT,
    model_id TEXT, sandbox_backend TEXT, sandbox_image TEXT,
    outcome TEXT, passed INTEGER, verdict_json TEXT,
    turns INTEGER, tool_calls INTEGER,
    input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
    wall_s REAL, sandbox_cpu_s REAL,
    trace_sha256 TEXT, final_workspace_sha256 TEXT,
    retrieved_memory_ids_json TEXT, memory_snapshot_id TEXT, backend_props_sha256 TEXT,
    memory_props_sha256 TEXT, batch_id TEXT,
    started_at TEXT, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS llm_calls (
    episode_id TEXT NOT NULL, turn INTEGER NOT NULL, request_id TEXT, model TEXT, stop_reason TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
    latency_s REAL, served_json TEXT, PRIMARY KEY (episode_id, turn)
);
CREATE TABLE IF NOT EXISTS memory_snapshots (
    snapshot_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
    history INTEGER NOT NULL, arm TEXT NOT NULL, after_episode INTEGER NOT NULL,
    source_bank_id TEXT NOT NULL, bank_id TEXT NOT NULL,
    bank_export_sha256 TEXT NOT NULL, content_sha256 TEXT NOT NULL, learner_state_sha256 TEXT,
    created_at TEXT NOT NULL, UNIQUE (run_id, history, arm, after_episode)
);
-- Arm D procedures (SPEC §9/§10; elh/learner/workflows.py). One row per workflow; a revision is
-- one whole regenerated library of a (run, history, arm, family) stream and shares one status:
-- candidate -> promoted | rejected; promoted -> superseded | rolled_back; superseded -> promoted
-- (rollback only). Bodies and provenance are never rewritten; only status/validation/timestamps move.
CREATE TABLE IF NOT EXISTS skills (
    skill_id TEXT NOT NULL, revision INTEGER NOT NULL, run_id TEXT NOT NULL,
    history INTEGER NOT NULL, arm TEXT NOT NULL, family TEXT NOT NULL, position INTEGER NOT NULL,
    status TEXT NOT NULL, title TEXT, body TEXT NOT NULL, body_sha256 TEXT NOT NULL,
    library_sha256 TEXT NOT NULL, source_episode_ids_json TEXT NOT NULL, induction_json TEXT,
    validation_json TEXT, created_at TEXT NOT NULL, promoted_at TEXT, rolled_back_at TEXT,
    PRIMARY KEY (run_id, history, arm, family, revision, skill_id)
);
CREATE TABLE IF NOT EXISTS ledger (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, episode_id TEXT,
    category TEXT NOT NULL, component TEXT NOT NULL, units REAL, unit TEXT, usd REAL NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "elh.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._add_columns("episodes", {"memory_props_sha256": "TEXT", "batch_id": "TEXT", "retrieval_json": "TEXT",
                                       "memory_searches_json": "TEXT", "rule_json": "TEXT"})
        # Arm D: the procedure library revision a snapshot pins (NULL: none promoted yet, or not arm D).
        self._add_columns("memory_snapshots", {"library_revision": "INTEGER"})
        # Learning spend with no episode (arm D induction / GEPA reflection): which stream and why.
        self._add_columns("ledger", {"history": "INTEGER", "arm": "TEXT", "purpose": "TEXT"})

    def _add_columns(self, table: str, cols: dict[str, str]) -> None:
        """Columns added after a store was created (CREATE TABLE IF NOT EXISTS keeps the old shape)."""
        have = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
        with self.db:
            for name, kind in cols.items():
                if name not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

    # -- blobs -------------------------------------------------------------------------------
    def put_blob(self, kind: str, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / kind / digest[:2] / digest
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.rename(path)
        return digest

    def get_blob(self, kind: str, digest: str) -> bytes:
        data = (self.root / kind / digest[:2] / digest).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"blob {kind}/{digest} is corrupt")
        return data

    def put_trace(self, events: list[dict]) -> str:
        return self.put_blob("traces", "".join(json.dumps(e, sort_keys=True, default=str) + "\n" for e in events).encode())

    def put_workspace(self, files: dict[str, str]) -> str:
        return self.put_blob("workspaces", json.dumps(files, sort_keys=True).encode())

    def get_workspace(self, digest: str) -> dict[str, str]:
        return json.loads(self.get_blob("workspaces", digest))

    # -- rows --------------------------------------------------------------------------------
    def insert(self, table: str, row: dict) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self.db:
            self.db.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))

    def ensure_run(self, run_id: str, name: str, version: int, config_hash: str, config_json: str,
                   git_rev: str | None, harness_version: str) -> None:
        existing = self.db.execute("SELECT config_hash FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if existing:
            if existing["config_hash"] != config_hash:
                raise RuntimeError(f"run {run_id} was registered with a different config; "
                                   "registration files are immutable once a run starts")
            return
        clash = self.db.execute("SELECT run_id FROM runs WHERE name = ? AND version = ? AND config_hash != ?",
                                (name, version, config_hash)).fetchone()
        if clash:
            raise RuntimeError(f"experiment name {name!r} already ran under a different config "
                               f"(run {clash['run_id']}); bump `version` (or `name`) instead of editing it")
        self.insert("runs", {"run_id": run_id, "name": name, "version": version, "config_hash": config_hash,
                             "config_json": config_json, "git_rev": git_rev,
                             "harness_version": harness_version, "started_at": utcnow()})

    def episode(self, episode_id: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
        if row is None:
            raise KeyError(episode_id)
        return row

    def snapshot(self, snapshot_id: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM memory_snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        return row

    def run_config_json(self, run_id: str) -> str:
        return self.db.execute("SELECT config_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
