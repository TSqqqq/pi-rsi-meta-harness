from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .db import ResearchDB
from .util import ensure_dir


class EventBus:
    def __init__(self, db: ResearchDB, log_dir: Path, echo: bool = True):
        self.db = db
        self.log_dir = ensure_dir(log_dir)
        self.events_path = self.log_dir / "events.jsonl"
        self.echo = echo
        self._lock = threading.Lock()

    def emit(self, type_: str, message: str, *, source: str = "SYSTEM", level: str = "INFO", payload: dict[str, Any] | None = None) -> None:
        ts = time.time()
        payload = payload or {}
        self.db.execute(
            "INSERT INTO events(ts,level,type,source,message,payload_json) VALUES(?,?,?,?,?,?)",
            (ts, level, type_, source, message, json.dumps(payload, ensure_ascii=False)),
        )
        record = {"ts": ts, "level": level, "type": type_, "source": source, "message": message, "payload": payload}
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if self.echo:
                stamp = time.strftime("%H:%M:%S", time.localtime(ts))
                print(f"[{stamp}] [{source}] {message}", flush=True)

    def recent(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        for r in rows:
            try:
                r["payload"] = json.loads(r.pop("payload_json") or "{}")
            except Exception:
                r["payload"] = {}
        return list(reversed(rows))
