from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import Config
from .db import ResearchDB
from .state import HarnessState


@dataclass
class StopDecision:
    state: HarnessState | None
    reason: str = ""


class StopPolicy:
    def __init__(self, cfg: Config, db: ResearchDB):
        self.cfg = cfg
        self.db = db

    def evaluate(self) -> StopDecision:
        best = self.db.best(self.cfg.direction)
        target = self.cfg.target
        if best and target is not None and best.get("metric") is not None:
            hit = float(best["metric"]) >= target if self.cfg.direction == "maximize" else float(best["metric"]) <= target
            require_full = bool(self.cfg.get("stop", "require_full_stage_for_target", True))
            validated = bool(best.get("validated")) and best.get("stage") == "full"
            if hit and (validated or not require_full):
                return StopDecision(HarnessState.TARGET_REACHED, f"validated target reached: {best['metric']} vs {target}")

        exp_count = self.db.completed_count()
        max_exp = int(self.cfg.get("budget", "max_experiments", 1000))
        if exp_count >= max_exp:
            return StopDecision(HarnessState.BUDGET_EXHAUSTED, f"max experiments reached: {exp_count}/{max_exp}")

        gpu = self.db.one("SELECT COALESCE(SUM(gpu_hours),0) AS x FROM experiments") or {"x": 0}
        max_gpu = float(self.cfg.get("budget", "max_gpu_hours", 1e18))
        if float(gpu["x"]) >= max_gpu:
            return StopDecision(HarnessState.BUDGET_EXHAUSTED, f"GPU-hour budget reached: {gpu['x']:.2f}/{max_gpu:.2f}")

        tok = self.db.one("SELECT COALESCE(SUM(total_tokens),0) AS x FROM agent_runs") or {"x": 0}
        max_tok = int(self.cfg.get("budget", "max_agent_tokens", 2**63-1))
        if int(tok["x"]) >= max_tok:
            return StopDecision(HarnessState.BUDGET_EXHAUSTED, f"agent token budget reached: {tok['x']}/{max_tok}")

        started = float(self.db.get_meta("started_at", time.time()))
        max_wall = float(self.cfg.get("budget", "max_wall_hours", 1e18))
        if (time.time() - started) / 3600 >= max_wall:
            return StopDecision(HarnessState.BUDGET_EXHAUSTED, f"wall-time budget reached: {max_wall}h")

        crashes = self.db.query("SELECT id FROM experiments WHERE status IN ('failed','invalid') ORDER BY updated_at DESC LIMIT ?", (int(self.cfg.get("budget", "max_consecutive_crashes", 8)),))
        # Consecutive crash detection using latest rows rather than any historical failures.
        ncrash = int(self.cfg.get("budget", "max_consecutive_crashes", 8))
        latest = self.db.query("SELECT status FROM experiments WHERE id!='baseline' ORDER BY updated_at DESC LIMIT ?", (ncrash,))
        if len(latest) >= ncrash and all(r["status"] in {"failed", "invalid"} for r in latest):
            return StopDecision(HarnessState.SAFETY_STOP, f"{ncrash} consecutive failed/invalid experiments")

        plateau_n = int(self.cfg.get("stop", "plateau_experiments", 30))
        if best and best.get("created_at") and exp_count >= plateau_n:
            since_best = self.db.one(
                "SELECT COUNT(*) AS n FROM experiments WHERE id!='baseline' AND created_at>? AND status IN ('completed','promising','champion','near_miss','rejected','failed','invalid')",
                (best["created_at"],),
            ) or {"n": 0}
            ack_count = int(self.db.get_meta("plateau_ack_count", 0) or 0)
            since_ack = max(0, exp_count - ack_count)
            if int(since_best["n"]) >= plateau_n and since_ack >= plateau_n:
                if bool(self.cfg.get("stop", "pause_on_plateau", True)):
                    return StopDecision(HarnessState.PAUSED_FOR_HUMAN, f"no new champion for {since_best['n']} completed experiments")
                return StopDecision(HarnessState.BUDGET_EXHAUSTED, f"plateau after {since_best['n']} experiments")

        return StopDecision(None, "")
