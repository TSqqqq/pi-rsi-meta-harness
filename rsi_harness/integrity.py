from __future__ import annotations

import fnmatch
import hashlib
from pathlib import Path
from typing import Iterable

from .config import Config
from .db import ResearchDB


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_matching(repo: Path, patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for p in repo.rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        rel = p.relative_to(repo).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in patterns):
            out.append(p)
    return sorted(set(out))


def record_immutable_hashes(cfg: Config, db: ResearchDB) -> int:
    patterns = cfg.get("files", "immutable", []) or []
    count = 0
    for path in iter_matching(cfg.repo, patterns):
        rel = path.relative_to(cfg.repo).as_posix()
        db.execute(
            "INSERT INTO immutable_hashes(path,sha256,recorded_at) VALUES(?,?,strftime('%s','now')) "
            "ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, recorded_at=excluded.recorded_at",
            (rel, sha256_file(path)),
        )
        count += 1
    return count


def verify_immutable_hashes(cfg: Config, db: ResearchDB, worktree: Path | None = None) -> list[str]:
    root = worktree or cfg.repo
    violations: list[str] = []
    rows = db.query("SELECT path,sha256 FROM immutable_hashes")
    recorded = {str(row["path"]): row["sha256"] for row in rows}
    current_paths = {
        path.relative_to(root).as_posix()
        for path in iter_matching(root, cfg.get("files", "immutable", []) or [])
    }
    recorded_paths = set(recorded)
    for rel in sorted(current_paths - recorded_paths):
        violations.append(f"new immutable file: {rel}")
    for rel in sorted(recorded_paths - current_paths):
        violations.append(f"missing immutable file: {rel}")
    for rel, digest in recorded.items():
        if rel not in current_paths:
            continue
        p = root / rel
        if sha256_file(p) != digest:
            violations.append(f"immutable file changed: {rel}")
    return violations
