from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from .config import Config
from .controller import ResearchController
from .dashboard import DashboardServer
from .db import ResearchDB
from .gitops import head_commit, is_dirty
from .state import HarnessState


def load(args) -> tuple[Config, ResearchDB, Path]:
    cfg = Config.load(args.config)
    root = Path(__file__).resolve().parent.parent
    return cfg, ResearchDB(cfg.db_path), root


def set_control(db: ResearchDB, action: str) -> None:
    db.set_meta("control", {"action": action, "ts": time.time()})


def cmd_status(cfg: Config, db: ResearchDB) -> int:
    best = db.best(cfg.direction)
    gpu = db.one("SELECT COALESCE(SUM(gpu_hours),0) AS x FROM experiments") or {"x": 0}
    tok = db.one("SELECT COALESCE(SUM(total_tokens),0) AS x FROM agent_runs") or {"x": 0}
    out = {
        "state": db.get_meta("state", "NEW"),
        "reason": db.get_meta("pause_reason", ""),
        "best": None if not best else {"id": best["id"], "metric": best["metric"], "stage": best["stage"], "validated": bool(best["validated"])},
        "experiments": db.completed_count(),
        "gpu_hours": gpu["x"],
        "agent_tokens": tok["x"],
        "active_policy": db.get_meta("active_policy", {}),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_rebase_baseline(cfg: Config, db: ResearchDB, ns) -> int:
    """Move the baseline to HEAD so new worktrees get current harness/adapter code. History is kept, not rewritten."""
    if is_dirty(cfg.repo):
        raise SystemExit("commit first: the new baseline must be an exact commit")
    base = db.experiment("baseline") or {}
    old_commit, new_commit = base.get("git_commit"), head_commit(cfg.repo)
    stages = dict(db.get_meta("baseline_stage_metrics", {}) or {})
    old_metric = stages.get(ns.stage)
    if old_metric is not None and abs(float(old_metric) - ns.metric) > ns.tolerance:
        raise SystemExit(f"baseline {ns.stage} changed {old_metric} -> {ns.metric} (> tolerance {ns.tolerance}); not behavior-preserving")
    # The re-run is one more measurement of the same configuration: keep both, compare against their mean.
    db.add_metric("baseline", ns.stage, None, ns.metric, {"rebase_commit": new_commit, "evidence": ns.evidence})
    stages[ns.stage] = db.metric_mean("baseline", ns.stage)
    db.update_experiment("baseline", git_commit=new_commit)
    db.set_meta("baseline_commit", new_commit)
    db.set_meta("baseline_stage_metrics", stages)
    db.execute(
        "INSERT INTO events(ts,level,type,source,message,payload_json) VALUES(?,?,?,?,?,?)",
        (time.time(), "INFO", "baseline_rebased", "HUMAN", f"baseline {old_commit} -> {new_commit}; {ns.stage}={ns.metric}",
         json.dumps({"old_commit": old_commit, "new_commit": new_commit, "old_metric": old_metric, "evidence": ns.evidence})),
    )
    print(f"baseline rebased to {new_commit[:10]}; {ns.stage} mean {old_metric} -> {stages[ns.stage]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pi RSI Meta Harness")
    ap.add_argument("--config", default="research.toml", help="research TOML path")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("run")
    sub.add_parser("status")
    sub.add_parser("pause")
    sub.add_parser("resume")
    sub.add_parser("stop")
    sub.add_parser("dashboard")
    note = sub.add_parser("note"); note.add_argument("text")
    pin = sub.add_parser("pin"); pin.add_argument("experiment_id")
    branch = sub.add_parser("branch"); branch.add_argument("experiment_id"); branch.add_argument("insight")
    steer = sub.add_parser("steer"); steer.add_argument("agent_name"); steer.add_argument("text")
    approve = sub.add_parser("approve-dependency"); approve.add_argument("package")
    reject = sub.add_parser("reject-dependency"); reject.add_argument("package"); reject.add_argument("--reason", default="rejected by human")
    rebase = sub.add_parser("rebase-baseline", help="branch new experiments from the current HEAD after re-measuring the baseline")
    rebase.add_argument("--stage", default="quick"); rebase.add_argument("--metric", type=float, required=True)
    rebase.add_argument("--evidence", required=True, help="evaluation result file of the baseline re-run at HEAD")
    rebase.add_argument("--tolerance", type=float, default=0.01, help="max |old-new| attributed to run-to-run noise")
    ns = ap.parse_args(argv)
    cfg, db, root = load(ns)

    if ns.cmd == "init":
        asyncio.run(ResearchController(cfg, root).initialize()); return 0
    if ns.cmd == "run":
        asyncio.run(ResearchController(cfg, root).run()); return 0
    if ns.cmd == "status": return cmd_status(cfg, db)
    if ns.cmd == "pause":
        set_control(db, "pause"); print("Pause requested: current subprocesses may finish, but no new batch will be scheduled."); return 0
    if ns.cmd == "resume":
        if db.get_meta("state", "") == "PAUSED_FOR_HUMAN":
            db.set_meta("plateau_ack_count", db.completed_count())
        set_control(db, "resume"); print("Resume requested."); return 0
    if ns.cmd == "stop":
        set_control(db, "stop"); print("Stop requested; controller will abort active Pi runs and training processes."); return 0
    if ns.cmd == "note":
        db.add_guidance(ns.text, kind="note"); print("Human research note recorded."); return 0
    if ns.cmd == "pin":
        if not db.experiment(ns.experiment_id): raise SystemExit(f"Unknown experiment: {ns.experiment_id}")
        db.update_experiment(ns.experiment_id, pinned=1); print(f"Pinned {ns.experiment_id} to the frontier."); return 0
    if ns.cmd == "branch":
        if not db.experiment(ns.experiment_id): raise SystemExit(f"Unknown experiment: {ns.experiment_id}")
        db.update_experiment(ns.experiment_id, pinned=1)
        db.add_guidance(ns.insight, kind="branch", experiment_id=ns.experiment_id)
        print(f"Pinned {ns.experiment_id} and recorded branch insight."); return 0
    if ns.cmd == "steer":
        db.add_guidance(ns.text, kind="steer", experiment_id=ns.agent_name)
        print(f"Steering request queued for {ns.agent_name}."); return 0
    if ns.cmd in {"approve-dependency", "reject-dependency"}:
        row = db.one("SELECT id FROM dependency_requests WHERE lower(package)=lower(?) AND status='requested' ORDER BY id DESC LIMIT 1", (ns.package,))
        if not row: raise SystemExit(f"No pending request for {ns.package}")
        status = "approved" if ns.cmd == "approve-dependency" else "rejected"
        note_text = "approved by human" if status == "approved" else ns.reason
        db.execute("UPDATE dependency_requests SET status=?,resolved_at=?,note=? WHERE id=?", (status, time.time(), note_text, row["id"]))
        print(f"{ns.package}: {status}"); return 0
    if ns.cmd == "rebase-baseline":
        return cmd_rebase_baseline(cfg, db, ns)
    if ns.cmd == "dashboard":
        srv = DashboardServer(cfg, db, root / "dashboard" / "index.html")
        srv.start()
        host = cfg.get("dashboard", "host", "127.0.0.1"); port = cfg.get("dashboard", "port", 8765)
        print(f"Dashboard: http://{host}:{port} (Ctrl+C to stop)")
        try:
            while True: time.sleep(3600)
        except KeyboardInterrupt:
            srv.stop()
        return 0
    return 2
