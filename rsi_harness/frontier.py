from __future__ import annotations

import math
import random
from typing import Any

from .config import Config
from .db import ResearchDB


class FrontierSelector:
    def __init__(self, cfg: Config, db: ResearchDB):
        self.cfg = cfg
        self.db = db

    def select(self, n: int | None = None) -> list[dict[str, Any]]:
        n = n or int(self.cfg.get("search", "frontier_size", 12))
        rows = self.db.query(
            "SELECT * FROM experiments WHERE metric IS NOT NULL AND status NOT IN ('invalid','failed')"
        )
        if not rows:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for r in rows:
            # delta_baseline is stage-normalized, so quick/medium/full nodes remain comparable as search evidence.
            perf_score = float(r.get("delta_baseline") or 0.0)
            novelty = float(r.get("novelty") or 0.0)
            confidence = float(r.get("confidence") or 0.5)
            pinned = 4.0 if int(r.get("pinned") or 0) else 0.0
            status_bonus = {
                "champion": 3.0,
                "promising": 2.0,
                "near_miss": 1.5,
                "completed": 1.0,
                "rejected": -2.0,
            }.get(str(r.get("status")), 0.0)
            children = self.db.one("SELECT COUNT(*) AS n FROM edges WHERE parent_id=?", (r["id"],)) or {"n": 0}
            underexplored = 1.0 / math.sqrt(1.0 + int(children["n"]))
            jitter = random.random() * 0.05
            score = perf_score + 1.2 * novelty + 0.4 * (1.0 - confidence) + status_bonus + underexplored + pinned + jitter
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = [r for _, r in scored[:n]]
        # Always include baseline as a route to genuinely new mechanisms if space allows.
        baseline = self.db.experiment("baseline")
        if baseline and all(r["id"] != "baseline" for r in result) and len(result) < n:
            result.append(baseline)
        return result
