from __future__ import annotations

import fnmatch
import re
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


def numeric_only_diff(worktree: Path) -> bool:
    """True when every changed line differs from its counterpart only in numeric literals (e.g. 0.5 -> 0.3).

    Such patches are hyperparameter edits in disguise: they belong to harness tuning runs, where the knob is
    passed on the command line and is guaranteed to take effect.
    """
    diff = run_capture(["git", "diff", "-U0", "HEAD"], cwd=worktree).stdout
    removed = [ln[1:] for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    added = [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    if not added or len(added) != len(removed):
        return False
    strip = lambda t: re.sub(r"[-+]?\d+(\.\d+)?(e[-+]?\d+)?", "#", t.strip())
    return all(strip(a) == strip(r) for a, r in zip(added, removed))


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


def sync_harness_files(cfg: Config, worktree: Path, base: str) -> list[str]:
    """Bring the harness-owned files of an experiment worktree up to the current baseline commit.

    A worktree starts from its parent's commit, which can predate a harness change (e.g. a fixed benchmark
    driver or a new adapter). Immutable paths and [files].harness_runtime paths belong to the harness, so
    they are checked out from the baseline; experiment code (the mutable paths) is untouched.
    The resulting change is committed so the experiment's own diff stays clean.
    """
    mutable = cfg.get("files", "mutable", []) or []
    immutable = cfg.get("files", "immutable", []) or []
    # Files the experiment pipeline runs or reads from the worktree besides experiment code (e.g. the adapter);
    # the controller itself always runs from the main checkout, so its own files need no sync.
    runtime = cfg.get("files", "harness_runtime", []) or []
    if not base or not mutable:
        return []  # without a mutable allowlist, harness files cannot be told apart from experiment code
    changed = run_capture(["git", "diff", "--name-only", "HEAD", base], cwd=worktree).stdout.split()
    # Immutable wins over mutable (e.g. src/eval/** inside a mutable src/**): the evaluator is the harness's.
    harness = [f for f in changed if any(fnmatch.fnmatch(f, pat) for pat in [*immutable, *runtime])]
    if not harness:
        return []
    present = set(run_capture(["git", "ls-tree", "-r", "--name-only", base], cwd=worktree).stdout.split())
    keep = [f for f in harness if f in present]
    gone = [f for f in harness if f not in present]
    if keep:
        subprocess.run(["git", "checkout", base, "--", *keep], cwd=worktree, check=True, capture_output=True)
    if gone:
        subprocess.run(["git", "rm", "-q", "--", *gone], cwd=worktree, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Pi RSI Harness", "-c", "user.email=pi-rsi@local", "-c", "commit.gpgsign=false",
                    "commit", "-q", "-m", f"harness files synced to baseline {base[:10]}"],
                   cwd=worktree, check=True, capture_output=True)
    return harness


def remove_worktree(cfg: Config, experiment_id: str, force: bool = False) -> None:
    path = cfg.worktree_dir / experiment_id
    if not path.exists():
        return
    args = ["git", "worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(path))
    subprocess.run(args, cwd=cfg.repo, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
