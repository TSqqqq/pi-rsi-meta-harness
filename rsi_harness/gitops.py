from __future__ import annotations

import fnmatch
import shutil
import subprocess
from pathlib import Path

from .config import Config
from .util import ensure_dir, run_capture


class GitError(RuntimeError):
    pass


def ensure_git_repo(repo: Path) -> None:
    r = run_capture(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo, check=False)
    if r.returncode != 0 or r.stdout.strip() != "true":
        raise GitError(f"Not a git repository: {repo}")


def head_commit(repo: Path) -> str:
    return run_capture(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()


def is_dirty(repo: Path) -> bool:
    return bool(run_capture(["git", "status", "--porcelain"], cwd=repo).stdout.strip())


def create_worktree(cfg: Config, experiment_id: str, parent_commit: str) -> Path:
    ensure_dir(cfg.worktree_dir)
    path = cfg.worktree_dir / experiment_id
    if path.exists():
        return path
    branch = f"rsi/{cfg.state_dir.name}/{experiment_id}"  # campaign-scoped: ids restart per campaign
    r = run_capture(["git", "worktree", "add", "-b", branch, str(path), parent_commit], cwd=cfg.repo, check=False)
    if r.returncode != 0:
        # Branch may already exist after a controller restart.
        r2 = run_capture(["git", "worktree", "add", str(path), branch], cwd=cfg.repo, check=False)
        if r2.returncode != 0:
            raise GitError(r.stderr + "\n" + r2.stderr)
    return path


def diff_files(worktree: Path) -> list[str]:
    out = run_capture(["git", "status", "--porcelain"], cwd=worktree).stdout
    files: list[str] = []
    for line in out.splitlines():
        if len(line) >= 4:
            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            files.append(path)
    return files


def validate_modified_paths(cfg: Config, worktree: Path) -> list[str]:
    immutable = cfg.get("files", "immutable", []) or []
    mutable = cfg.get("files", "mutable", []) or []
    problems: list[str] = []
    for f in diff_files(worktree):
        if any(fnmatch.fnmatch(f, pat) for pat in immutable):
            problems.append(f"immutable path modified: {f}")
        elif mutable and not any(fnmatch.fnmatch(f, pat) for pat in mutable):
            problems.append(f"path outside mutable allowlist modified: {f}")
    return problems


def save_patch(worktree: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tracked = run_capture(["git", "diff", "--binary", "HEAD"], cwd=worktree).stdout
    untracked = run_capture(["git", "ls-files", "--others", "--exclude-standard"], cwd=worktree).stdout.splitlines()
    chunks = [tracked]
    for f in untracked:
        p = worktree / f
        if p.is_file() and p.stat().st_size < 2_000_000:
            # Record untracked file contents in a simple appendix; git commit below will preserve them too.
            chunks.append(f"\n# UNTRACKED FILE: {f}\n")
            try:
                chunks.append(p.read_text(encoding="utf-8"))
            except Exception:
                chunks.append("<binary or unreadable>\n")
    dest.write_text("\n".join(chunks), encoding="utf-8", errors="replace")


def commit_all(worktree: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=worktree, check=True)
    # Use local fallback identity only if repo/user config is absent.
    subprocess.run(["git", "-c", "user.name=Pi RSI Harness", "-c", "user.email=pi-rsi@local", "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", message], cwd=worktree, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return head_commit(worktree)


def remove_worktree(cfg: Config, experiment_id: str, force: bool = False) -> None:
    path = cfg.worktree_dir / experiment_id
    if not path.exists():
        return
    args = ["git", "worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(path))
    subprocess.run(args, cwd=cfg.repo, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
