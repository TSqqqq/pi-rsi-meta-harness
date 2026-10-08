from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import resolve


@dataclass
class Config:
    path: Path
    base_dir: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> Config:
        p = Path(path).expanduser().resolve()
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
        return cls(path=p, base_dir=p.parent, raw=raw)

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)

    @property
    def repo(self) -> Path:
        return resolve(self.base_dir, self.get("project", "repo"))

    def project_path(self, key: str, default: str) -> Path:
        return resolve(self.base_dir, self.get("project", key, default))

    @property
    def state_dir(self) -> Path:
        return self.project_path("state_dir", "./state")

    @property
    def db_path(self) -> Path:
        return self.state_dir / "research.sqlite3"

    def campaign_path(self, key: str, name: str) -> Path:
        # Per-campaign data defaults to state_dir so campaigns never share Pi sessions, worktrees or logs.
        raw = self.get("project", key)
        return resolve(self.base_dir, raw) if raw else self.state_dir / name

    @property
    def worktree_dir(self) -> Path:
        return self.campaign_path("worktree_dir", "worktrees")

    @property
    def artifact_dir(self) -> Path:
        return self.campaign_path("artifact_dir", "artifacts")

    @property
    def session_dir(self) -> Path:
        return self.campaign_path("session_dir", "sessions")

    @property
    def transcript_dir(self) -> Path:
        return self.campaign_path("transcript_dir", "transcripts")

    @property
    def log_dir(self) -> Path:
        return self.campaign_path("log_dir", "logs")

    @property
    def direction(self) -> str:
        d = str(self.get("objective", "direction", "maximize")).lower()
        if d not in {"maximize", "minimize"}:
            raise ValueError("objective.direction must be maximize or minimize")
        return d

    @property
    def target(self) -> float | None:
        v = self.get("objective", "target", None)
        return None if v is None else float(v)

    def better(self, a: float, b: float, min_delta: float = 0.0) -> bool:
        return a >= b + min_delta if self.direction == "maximize" else a <= b - min_delta

    def delta(self, value: float, baseline: float) -> float:
        return value - baseline if self.direction == "maximize" else baseline - value

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.repo.exists():
            errors.append(f"project.repo does not exist: {self.repo}")
        if not self.get("objective", "metric"):
            errors.append("objective.metric is required")
        if not self.get("commands", "evaluate"):
            errors.append("commands.evaluate is required")
        if int(self.get("resources", "max_parallel", 1)) < 1:
            errors.append("resources.max_parallel must be >= 1")
        mode = str(self.get("models", "mode", "single"))
        if mode not in {"single", "dual"}:
            errors.append("models.mode must be single or dual")
        return errors
