from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .config import Config
from .db import ResearchDB

SPEC_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[([A-Za-z0-9_,.-]+)\])?(?:(==|>=|<=|~=|>|<)([A-Za-z0-9.*+_-]+))?$")


def load_cfg(path: str | None) -> Config:
    p = path or os.environ.get("RSI_CONFIG") or "research.toml"
    return Config.load(p)


def base_name(spec: str) -> str:
    m = SPEC_RE.fullmatch(spec)
    if not m:
        raise SystemExit("Rejected package spec. Only normal PyPI names, optional extras, and simple version constraints are allowed; URLs/VCS/editable installs are forbidden.")
    return m.group(1).lower().replace("_", "-")


def is_approved(cfg: Config, db: ResearchDB, spec: str) -> bool:
    base = base_name(spec)
    allowed = (cfg.get("dependencies", "allowed_packages", []) or [])
    if any(base_name(str(item)) == base for item in allowed):
        return True
    rows = db.query("SELECT package,status FROM dependency_requests WHERE status='approved' ORDER BY id DESC")
    return any(base_name(str(row["package"])) == base for row in rows)


def request(cfg: Config, spec: str, reason: str) -> int:
    base_name(spec)
    db = ResearchDB(cfg.db_path)
    db.execute("INSERT INTO dependency_requests(ts,package,reason,status) VALUES(?,?,?,'requested')", (time.time(), spec, reason))
    print(f"Dependency requested: {spec}. Waiting for human/controller approval if not allowlisted.")
    return 0


def install(cfg: Config, spec: str) -> int:
    if not bool(cfg.get("dependencies", "allow_dynamic_pip", False)):
        raise SystemExit("Dynamic pip is disabled by research.toml")
    db = ResearchDB(cfg.db_path)
    if not is_approved(cfg, db, spec):
        raise SystemExit(f"Package not approved: {spec}. Run request first, then approve from controller CLI.")
    venv_root_raw = str(cfg.get("dependencies", "venv_root", "") or "")
    venv_root = Path(venv_root_raw).expanduser() if venv_root_raw else (cfg.state_dir / "venvs")
    if not venv_root.is_absolute():
        venv_root = (cfg.base_dir / venv_root).resolve()
    # Worktree basename is the experiment id in the standard harness layout.
    exp_name = Path.cwd().name or "shared"
    venv = venv_root / exp_name
    if not (venv / "bin" / "python").exists():
        args = [sys.executable, "-m", "venv"]
        if bool(cfg.get("dependencies", "inherit_system_site_packages", True)):
            args.append("--system-site-packages")
        subprocess.run([*args, str(venv)], check=True)
    py = venv / "bin" / "python"
    # No shell=True: spec is a single validated argument.
    subprocess.run([str(py), "-m", "pip", "install", spec], check=True)
    print(f"Installed {spec} into {venv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Guarded project-local dependency helper")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rq = sub.add_parser("request")
    rq.add_argument("package")
    rq.add_argument("reason")
    ins = sub.add_parser("install")
    ins.add_argument("package")
    ns = ap.parse_args(argv)
    cfg = load_cfg(ns.config)
    if ns.cmd == "request":
        return request(cfg, ns.package, ns.reason)
    return install(cfg, ns.package)


if __name__ == "__main__":
    raise SystemExit(main())
