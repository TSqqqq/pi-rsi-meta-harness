from __future__ import annotations

import json
from typing import Any

from .config import Config
from .db import ResearchDB


class ContextBuilder:
    """Build small, task-relevant snapshots instead of dumping the full research history."""

    def __init__(self, cfg: Config, db: ResearchDB):
        self.cfg = cfg
        self.db = db

    def global_state(self) -> dict[str, Any]:
        best = self.db.best(self.cfg.direction)
        baseline = self.db.experiment("baseline")
        recent = self.db.query(
            "SELECT id,status,stage,title,hypothesis,mechanism,metric,delta_baseline,lesson,novelty,confidence "
            "FROM experiments WHERE id!='baseline' ORDER BY updated_at DESC LIMIT 20"
        )
        successes = self.db.query(
            "SELECT id,title,mechanism,stage,metric,delta_baseline,lesson FROM experiments "
            "WHERE status IN ('champion','promising','completed') AND metric IS NOT NULL "
            "ORDER BY delta_baseline DESC LIMIT 8"
        )
        failures = self.db.query(
            "SELECT id,title,mechanism,metric,delta_baseline,failure_reason,lesson FROM experiments "
            "WHERE status IN ('rejected','failed','invalid') ORDER BY updated_at DESC LIMIT 8"
        )
        near = self.db.query(
            "SELECT id,title,mechanism,metric,delta_baseline,lesson FROM experiments "
            "WHERE status='near_miss' ORDER BY updated_at DESC LIMIT 6"
        )
        guidance = self.db.recent_guidance(10)
        beliefs = self.db.query("SELECT key,statement,confidence,evidence_json,updated_at FROM beliefs ORDER BY confidence DESC, updated_at DESC LIMIT 20")
        parameter_effects = self.db.query("SELECT experiment_id,parameter,from_value,to_value,effect,confidence,stage FROM parameter_effects ORDER BY created_at DESC LIMIT 30")
        return {
            "objective": {
                "metric": self.cfg.get("objective", "metric"),
                "direction": self.cfg.direction,
                "target": self.cfg.target,
                "min_meaningful_delta": self.cfg.get("objective", "min_meaningful_delta", 0.0),
            },
            "baseline": baseline,
            "champion": best,
            "recent": recent,
            "top_successes": successes,
            "recent_failures": failures,
            "near_misses": near,
            "human_guidance": guidance,
            "beliefs": beliefs,
            "parameter_effects": parameter_effects,
            "active_policy": self.db.get_meta("active_policy", {}),
        }

    def node_context(self, parent_ids: list[str]) -> dict[str, Any]:
        parents = [self.db.experiment(x) for x in parent_ids]
        ancestors: list[dict[str, Any]] = []
        seen = set(parent_ids)
        frontier = list(parent_ids)
        for _ in range(3):
            nxt: list[str] = []
            for node in frontier:
                for pid in self.db.parents(node):
                    if pid not in seen:
                        seen.add(pid)
                        e = self.db.experiment(pid)
                        if e:
                            ancestors.append(e)
                        nxt.append(pid)
            frontier = nxt
        return {
            "parents": [p for p in parents if p],
            "ancestors": ancestors[:10],
            "global": self.global_state(),
        }

    @staticmethod
    def render(data: dict[str, Any], max_chars: int = 50000) -> str:
        text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        if len(text) > max_chars:
            return text[:max_chars] + "\n...<context truncated by harness>"
        return text
