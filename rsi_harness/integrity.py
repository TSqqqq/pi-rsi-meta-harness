from __future__ import annotations

import fnmatch
import hashlib
import subprocess
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
    # Tracked plus new non-ignored files: build caches (__pycache__, outputs) are not research inputs.
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=repo, check=True, capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    patterns = list(patterns)
    return sorted({
        repo / rel for rel in listed
        if rel and (repo / rel).is_file() and any(fnmatch.fnmatch(rel, pat) for pat in patterns)
    })


def record_immutable_hashes(cfg: Config, db: ResearchDB) -> int:
    patterns = cfg.get("files", "immutable", []) or []
    db.execute("DELETE FROM immutable_hashes")  # a snapshot, not a merge: stale paths must not linger
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
