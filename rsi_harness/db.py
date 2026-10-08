from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .util import ensure_dir, json_dumps


SCHEMA = r'''
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  status TEXT NOT NULL,
  stage TEXT NOT NULL DEFAULT 'planned',
  title TEXT,
  hypothesis TEXT,
  mechanism TEXT,
  parent_primary TEXT,
  root_idea_id TEXT,
  git_commit TEXT,
  worktree TEXT,
  metric REAL,
  metric_std REAL,
  parent_metric REAL,
  delta_parent REAL,
  delta_baseline REAL,
  confidence REAL,
  novelty REAL,
  gpu_hours REAL NOT NULL DEFAULT 0,
  agent_tokens INTEGER NOT NULL DEFAULT 0,
  failure_reason TEXT,
  lesson TEXT,
  next_steps TEXT,
  changes_json TEXT,
  metadata_json TEXT,
  pinned INTEGER NOT NULL DEFAULT 0,
  validated INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edges (
  parent_id TEXT NOT NULL,
  child_id TEXT NOT NULL,
  edge_type TEXT NOT NULL DEFAULT 'mutation',
  PRIMARY KEY(parent_id, child_id),
  FOREIGN KEY(parent_id) REFERENCES experiments(id),
  FOREIGN KEY(child_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  experiment_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  seed INTEGER,
  metric REAL NOT NULL,
  extra_json TEXT,
  created_at REAL NOT NULL,
  FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  level TEXT NOT NULL,
  type TEXT NOT NULL,
  source TEXT NOT NULL,
  message TEXT NOT NULL,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS agent_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_start REAL NOT NULL,
  ts_end REAL,
  agent_name TEXT NOT NULL,
  task_type TEXT NOT NULL,
  session_id TEXT,
  provider TEXT,
  model_id TEXT,
  thinking TEXT,
  input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  total_tokens INTEGER DEFAULT 0,
  cost REAL DEFAULT 0,
  success INTEGER,
  error TEXT
);

CREATE TABLE IF NOT EXISTS human_guidance (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  experiment_id TEXT,
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  consumed INTEGER NOT NULL DEFAULT 0
);


CREATE TABLE IF NOT EXISTS human_notes (
  id INTEGER PRIMARY KEY CHECK (id=1),
  body TEXT NOT NULL DEFAULT '',
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS idea_queue (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  updated_at REAL NOT NULL,
  text TEXT NOT NULL,
  experiment_id TEXT,
  status TEXT NOT NULL DEFAULT 'draft',
  sent_at REAL,
  metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_idea_queue_status ON idea_queue(status, updated_at);

CREATE TABLE IF NOT EXISTS policy_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  version TEXT NOT NULL UNIQUE,
  policy_json TEXT NOT NULL,
  rationale TEXT,
  active INTEGER NOT NULL DEFAULT 0
);


CREATE TABLE IF NOT EXISTS beliefs (
  key TEXT PRIMARY KEY,
  statement TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.5,
  evidence_json TEXT,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS parameter_effects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  experiment_id TEXT NOT NULL,
  parameter TEXT NOT NULL,
  from_value TEXT,
  to_value TEXT,
  effect REAL,
  confidence REAL,
  stage TEXT,
  created_at REAL NOT NULL,
  FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS dependency_requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  package TEXT NOT NULL,
  reason TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'requested',
  resolved_at REAL,
  note TEXT
);

CREATE TABLE IF NOT EXISTS immutable_hashes (
  path TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL,
  recorded_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_metrics_exp ON metrics(experiment_id);
'''


AGENT_RUN_COLUMNS = {
    "tool_calls": "INTEGER DEFAULT 0",
    "experiment_id": "TEXT",
    "transcript_path": "TEXT",
    "session_file": "TEXT",
}


class ResearchDB:
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)
        self._local = threading.local()
        self.init_schema()

    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def init_schema(self) -> None:
        c = sqlite3.connect(self.path)
        try:
            c.executescript(SCHEMA)
            # Additive migration for databases created before transcript/tool accounting existed.
            have = {r[1] for r in c.execute("PRAGMA table_info(agent_runs)")}
            for col, decl in AGENT_RUN_COLUMNS.items():
                if col not in have:
                    c.execute(f"ALTER TABLE agent_runs ADD COLUMN {col} {decl}")
            c.commit()
        finally:
            c.close()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn().execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.execute(sql, params).fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        row = self.execute(sql, params).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None

    def set_meta(self, key: str, value: Any) -> None:
        v = value if isinstance(value, str) else json_dumps(value)
        self.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, v))

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.one("SELECT value FROM meta WHERE key=?", (key,))
        if not row:
            return default
        v = row["value"]
        try:
            return json.loads(v)
        except Exception:
            return v

    def next_experiment_id(self) -> str:
        row = self.one("SELECT COUNT(*) AS n FROM experiments WHERE id LIKE 'exp_%'") or {"n": 0}
        return f"exp_{int(row['n']) + 1:06d}"

    def add_experiment(self, exp: dict[str, Any], parents: list[str] | None = None) -> str:
        eid = exp.get("id") or self.next_experiment_id()
        ts = time.time()
        parents = parents or ([exp["parent_primary"]] if exp.get("parent_primary") else [])
        fields = {
            "id": eid,
            "created_at": ts,
            "updated_at": ts,
            "status": exp.get("status", "planned"),
            "stage": exp.get("stage", "planned"),
            "title": exp.get("title"),
            "hypothesis": exp.get("hypothesis"),
            "mechanism": exp.get("mechanism"),
            "parent_primary": exp.get("parent_primary") or (parents[0] if parents else None),
            "root_idea_id": exp.get("root_idea_id") or eid,
            "git_commit": exp.get("git_commit"),
            "worktree": exp.get("worktree"),
            "confidence": exp.get("confidence"),
            "novelty": exp.get("novelty"),
            "changes_json": json_dumps(exp.get("changes", {})),
            "metadata_json": json_dumps(exp.get("metadata", {})),
        }
        cols = ",".join(fields)
        qs = ",".join("?" for _ in fields)
        self.execute(f"INSERT INTO experiments({cols}) VALUES({qs})", fields.values())
        for pid in parents:
            self.execute("INSERT OR IGNORE INTO edges(parent_id,child_id,edge_type) VALUES(?,?,?)", (pid, eid, "crossover" if len(parents) > 1 else "mutation"))
        return eid

    def update_experiment(self, eid: str, **changes: Any) -> None:
        if not changes:
            return
        changes["updated_at"] = time.time()
        for key in ["changes", "metadata", "next_steps"]:
            if key in changes:
                new_key = key + "_json" if key in {"changes", "metadata"} else key
                changes[new_key] = json_dumps(changes.pop(key))
        # Column names cannot be bound parameters; restrict them to the real schema.
        allowed = {r["name"] for r in self.query("PRAGMA table_info(experiments)")}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unknown experiment columns: {sorted(unknown)}")
        cols = ",".join(f"{k}=?" for k in changes)
        self.execute(f"UPDATE experiments SET {cols} WHERE id=?", [*changes.values(), eid])  # noqa: S608

    def add_metric(self, eid: str, stage: str, seed: int | None, metric: float, extra: dict[str, Any] | None = None) -> None:
        self.execute(
            "INSERT INTO metrics(experiment_id,stage,seed,metric,extra_json,created_at) VALUES(?,?,?,?,?,?)",
            (eid, stage, seed, metric, json_dumps(extra or {}), time.time()),
        )

    def experiment(self, eid: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM experiments WHERE id=?", (eid,))

    def parents(self, eid: str) -> list[str]:
        return [r["parent_id"] for r in self.query("SELECT parent_id FROM edges WHERE child_id=?", (eid,))]

    def children(self, eid: str) -> list[str]:
        return [r["child_id"] for r in self.query("SELECT child_id FROM edges WHERE parent_id=?", (eid,))]

    def experiments(self, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM experiments ORDER BY created_at ASC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.query(sql)

    def completed_count(self) -> int:
        row = self.one("SELECT COUNT(*) AS n FROM experiments WHERE status IN ('completed','promising','champion','near_miss','rejected','failed','invalid')")
        return int(row["n"] if row else 0)

    def metric_mean(self, experiment_id: str, stage: str) -> float | None:
        row = self.one("SELECT AVG(metric) AS x FROM metrics WHERE experiment_id=? AND stage=?", (experiment_id, stage))
        return None if not row or row.get("x") is None else float(row["x"])

    def best(self, direction: str) -> dict[str, Any] | None:
        order = "DESC" if direction == "maximize" else "ASC"  # fixed literal, never user input
        # Champion comparisons use only publication-comparable full/validated results.
        sql = (
            "SELECT * FROM experiments WHERE metric IS NOT NULL AND validated=1 AND stage='full' "
            "AND status NOT IN ('invalid','failed') ORDER BY metric " + order + " LIMIT 1"
        )
        return self.one(sql)

    def add_guidance(self, text: str, kind: str = "note", experiment_id: str | None = None) -> None:
        self.execute("INSERT INTO human_guidance(ts,experiment_id,kind,text) VALUES(?,?,?,?)", (time.time(), experiment_id, kind, text))

    def get_note(self) -> dict[str, Any]:
        row = self.one("SELECT body,updated_at FROM human_notes WHERE id=1")
        return row or {"body": "", "updated_at": 0.0}

    def save_note(self, body: str) -> None:
        self.execute(
            "INSERT INTO human_notes(id,body,updated_at) VALUES(1,?,?) "
            "ON CONFLICT(id) DO UPDATE SET body=excluded.body,updated_at=excluded.updated_at",
            (body, time.time()),
        )

    def add_idea(self, text: str, experiment_id: str | None = None, metadata: dict[str, Any] | None = None) -> int:
        ts = time.time()
        cur = self.execute(
            "INSERT INTO idea_queue(ts,updated_at,text,experiment_id,status,metadata_json) VALUES(?,?,?,?,?,?)",
            (ts, ts, text, experiment_id, "draft", json_dumps(metadata or {})),
        )
        return int(cur.lastrowid)

    def list_ideas(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if status:
            return self.query(
                "SELECT * FROM idea_queue WHERE status=? ORDER BY updated_at DESC LIMIT ?",
                (status, limit),
            )
        return self.query("SELECT * FROM idea_queue ORDER BY updated_at DESC LIMIT ?", (limit,))

    def send_idea(self, idea_id: int) -> bool:
        row = self.one("SELECT * FROM idea_queue WHERE id=?", (idea_id,))
        if not row or row.get("status") == "sent":
            return False
        now = time.time()
        self.add_guidance(str(row["text"]), kind="idea", experiment_id=row.get("experiment_id"))
        self.execute("UPDATE idea_queue SET status='sent',sent_at=?,updated_at=? WHERE id=?", (now, now, idea_id))
        return True

    def delete_idea(self, idea_id: int) -> bool:
        cur = self.execute("DELETE FROM idea_queue WHERE id=?", (idea_id,))
        return bool(cur.rowcount)

    def recent_guidance(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM human_guidance ORDER BY ts DESC LIMIT ?", (limit,))
