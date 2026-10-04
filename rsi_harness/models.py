from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config
from .util import split_model


PLANNER_TASKS = {
    "research_planning",
    "hypothesis_generation",
    "candidate_critique",
    "failure_synthesis",
    "cross_branch_reasoning",
    "meta_rsi",
    "plateau_analysis",
    "paper_validation",
}


@dataclass(frozen=True)
class ModelRoute:
    spec: str | None
    thinking: str | None
    tier: str


class ModelRouter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mode = str(cfg.get("models", "mode", "single"))
        self.default = str(cfg.get("models", "default", "") or "")
        self.planner = str(cfg.get("models", "planner", "") or "")
        self.worker = str(cfg.get("models", "worker", "") or "")
        self.planner_thinking = str(cfg.get("models", "planner_thinking", "high") or "high")
        self.worker_thinking = str(cfg.get("models", "worker_thinking", "minimal") or "minimal")
        self.allow_escalation = bool(cfg.get("models", "allow_escalation", True))
        self.failures_before_escalation = int(cfg.get("models", "max_worker_failures_before_escalation", 2))

    def route(self, task_type: str, failures: int = 0, force_tier: str | None = None) -> ModelRoute:
        if self.mode == "single":
            return ModelRoute(self.default or None, self.planner_thinking if task_type in PLANNER_TASKS else self.worker_thinking, "single")
        tier = force_tier or ("planner" if task_type in PLANNER_TASKS else "worker")
        if tier == "worker" and self.allow_escalation and failures >= self.failures_before_escalation:
            tier = "planner"
        if tier == "planner":
            return ModelRoute(self.planner or self.default or None, self.planner_thinking, "planner")
        return ModelRoute(self.worker or self.default or None, self.worker_thinking, "worker")

    @staticmethod
    def model_key(model_obj: dict[str, Any]) -> str:
        return f"{model_obj.get('provider','')}/{model_obj.get('id','')}"

    def validate_specs(self, available_models: list[dict[str, Any]]) -> list[str]:
        available = {self.model_key(m) for m in available_models}
        errors: list[str] = []
        specs = [s for s in [self.default, self.planner if self.mode == "dual" else "", self.worker if self.mode == "dual" else ""] if s]
        for spec in specs:
            if spec not in available:
                errors.append(f"Configured model not available in Pi: {spec}")
        return errors

    @staticmethod
    def provider_model(spec: str) -> tuple[str, str]:
        return split_model(spec)
