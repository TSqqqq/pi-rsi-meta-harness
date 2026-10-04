from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from .agent_runtime import AgentRuntime
from .config import Config
from .context import ContextBuilder
from .dashboard import DashboardServer
from .db import ResearchDB
from .events import EventBus
from .experiments import ExperimentRunner
from .frontier import FrontierSelector
from .gitops import (
    GitError,
    commit_all,
    create_worktree,
    ensure_git_repo,
    head_commit,
    is_dirty,
    save_patch,
    validate_modified_paths,
)
from .integrity import record_immutable_hashes, verify_immutable_hashes
from .scheduler import GPUScheduler
from .state import HarnessState, TERMINAL_STATES
from .stopping import StopPolicy
from .util import ensure_dir, parse_json_from_agent_text, parse_last_json_object, shell_template


class ResearchController:
    def __init__(self, cfg: Config, harness_root: Path):
        self.cfg = cfg
        self.harness_root = harness_root
        self.db = ResearchDB(cfg.db_path)
        self.events = EventBus(self.db, cfg.log_dir, echo=True)
        self.scheduler = GPUScheduler(cfg)
        self.runner = ExperimentRunner(cfg, self.db, self.events, self.scheduler)
        self.agents = AgentRuntime(cfg, self.db, self.events)
        self.context = ContextBuilder(cfg, self.db)
        self.frontier = FrontierSelector(cfg, self.db)
        self.stop_policy = StopPolicy(cfg, self.db)
        self.dashboard = DashboardServer(cfg, self.db, harness_root / "dashboard" / "index.html")
        self._stop_requested = asyncio.Event()
        self._control_task: asyncio.Task[None] | None = None
        self._last_meta_count = int(self.db.get_meta("last_meta_count", 0) or 0)

    def set_state(self, state: HarnessState, reason: str = "") -> None:
        self.db.set_meta("state", state.value)
        self.db.set_meta("pause_reason", reason)
        self.events.emit("state", f"state={state.value}" + (f" reason={reason}" if reason else ""), source="CONTROLLER")

    def _load_prompt(self, name: str) -> str:
        return (self.harness_root / "prompts" / name).read_text(encoding="utf-8")

    async def initialize(self) -> None:
        errors = self.cfg.validate()
        if errors:
            raise RuntimeError("Configuration errors:\n- " + "\n- ".join(errors))
        ensure_git_repo(self.cfg.repo)
        if is_dirty(self.cfg.repo):
            raise RuntimeError("Target repository has uncommitted changes. Commit/stash them before init so baseline provenance is unambiguous.")
        for p in [self.cfg.state_dir, self.cfg.worktree_dir, self.cfg.artifact_dir, self.cfg.session_dir, self.cfg.log_dir]:
            ensure_dir(p)

        self.events.emit("resources", "harness skills will be loaded additively via Pi --skill; existing Pi skills remain enabled", source="INIT")
        model_errors = await self.agents.validate_models()
        if model_errors:
            raise RuntimeError("Model routing configuration errors:\n- " + "\n- ".join(model_errors))

        baseline_commit = head_commit(self.cfg.repo)
        self.db.set_meta("baseline_commit", baseline_commit)
        n = record_immutable_hashes(self.cfg, self.db)
        self.events.emit("integrity", f"recorded {n} immutable file hashes", source="INIT")
        if not self.db.experiment("baseline"):
            await self._run_baseline(baseline_commit)
        self.db.set_meta("started_at", self.db.get_meta("started_at", time.time()))
        self.set_state(HarnessState.INITIALIZED)
        # `init` should not leave a model-check Pi subprocess behind. run() will create role sessions lazily.
        await self.agents.close()

    async def _run_baseline(self, baseline_commit: str) -> None:
        eval_cmd = str(self.cfg.get("commands", "evaluate", "") or "")
        metric_key = str(self.cfg.get("objective", "metric_json_key", "metric"))
        env = os.environ.copy()
        stage_results: dict[str, tuple[float, float, float]] = {}

        async with self.scheduler.acquire() as alloc:
            if alloc.gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, alloc.gpu_ids))
            for stage in ["quick", "medium", "full"]:
                normal = str(self.cfg.get("commands", stage, "") or "")
                if not normal:
                    continue
                if stage == "full":
                    train_template = str(self.cfg.get("commands", "baseline", "") or normal)
                    seeds = [int(x) for x in (self.cfg.get("stages", "validation_seeds", [1, 2, 3]) or [1])]
                else:
                    train_template = str(self.cfg.get("commands", f"baseline_{stage}", "") or normal)
                    seeds = [1]
                vals_metrics: list[float] = []
                elapsed_total = 0.0
                for seed in seeds:
                    vals = {
                        "repo": self.cfg.repo,
                        "worktree": self.cfg.repo,
                        "experiment_id": "baseline",
                        "stage": stage,
                        "seed": seed,
                        "gpu_ids": ",".join(map(str, alloc.gpu_ids)),
                        "python": sys.executable,
                    }
                    timeout = float(self.cfg.get("stages", f"{stage}_timeout_min", self.cfg.get("stages", "full_timeout_min", 1440))) * 60
                    log = self.cfg.log_dir / "experiments" / f"baseline.{stage}.seed{seed}.train.log"
                    elog = self.cfg.log_dir / "experiments" / f"baseline.{stage}.seed{seed}.eval.log"
                    self.events.emit("baseline", f"calibrating {stage} baseline seed={seed}", source="BASELINE")
                    rc, _, elapsed = await self.runner._run_command(shell_template(train_template, **vals), cwd=self.cfg.repo, env=env, log_path=log, source=f"BASELINE:{stage}:train", timeout_s=timeout)
                    if rc != 0:
                        raise RuntimeError(f"baseline {stage} command failed rc={rc}")
                    rc, out, eval_elapsed = await self.runner._run_command(shell_template(eval_cmd, **vals), cwd=self.cfg.repo, env=env, log_path=elog, source=f"BASELINE:{stage}:eval", timeout_s=max(600, timeout / 2))
                    if rc != 0:
                        raise RuntimeError(f"baseline {stage} evaluator failed rc={rc}")
                    obj = parse_last_json_object(out)
                    metric = float(obj[metric_key])
                    vals_metrics.append(metric)
                    elapsed_total += elapsed + eval_elapsed
                    # baseline row is inserted after all calibration; keep values temporarily.
                    stage_results.setdefault(stage, (0.0, 0.0, 0.0))
                    self.events.emit("baseline_metric", f"{stage} seed={seed} metric={metric}", source="BASELINE")
                mean = statistics.fmean(vals_metrics)
                std = statistics.stdev(vals_metrics) if len(vals_metrics) >= 2 else 0.0
                gpu_hours = elapsed_total * len(alloc.gpu_ids) / 3600
                stage_results[stage] = (mean, std, gpu_hours)

        if not stage_results:
            raise RuntimeError("No configured quick/medium/full stage available for baseline calibration")
        full_stage = "full" if "full" in stage_results else list(stage_results)[-1]
        metric, std, _ = stage_results[full_stage]
        total_gpu_hours = sum(x[2] for x in stage_results.values())
        self.db.add_experiment({
            "id": "baseline",
            "status": "baseline",
            "stage": full_stage,
            "title": "Baseline",
            "hypothesis": "Original upstream implementation",
            "mechanism": "baseline",
            "git_commit": baseline_commit,
            "confidence": 1.0,
            "novelty": 0.0,
        }, parents=[])
        self.db.update_experiment("baseline", metric=metric, metric_std=std, delta_parent=0.0, delta_baseline=0.0, gpu_hours=total_gpu_hours, validated=1 if full_stage == "full" else 0)
        # Re-run evaluators are intentionally not needed: store calibrated aggregate metrics with seed=NULL.
        for stage, (mean, sdev, _) in stage_results.items():
            self.db.add_metric("baseline", stage, None, mean, {"aggregate": True, "std": sdev})
        self.db.set_meta("baseline_stage_metrics", {k: v[0] for k, v in stage_results.items()})
        self.events.emit("baseline", f"baseline calibrated: {self.db.get_meta('baseline_stage_metrics')}", source="BASELINE")

    async def _control_watch(self) -> None:
        last_seen = None
        while not self._stop_requested.is_set():
            steer_rows = self.db.query("SELECT id,experiment_id,text FROM human_guidance WHERE kind='steer' AND consumed=0 ORDER BY id")
            for row in steer_rows:
                try:
                    delivered = await self.agents.steer_by_name(str(row["experiment_id"]), str(row["text"]))
                    if delivered:
                        self.db.execute("UPDATE human_guidance SET consumed=1 WHERE id=?", (row["id"],))
                        self.events.emit("human_steer", f"delivered to {row['experiment_id']}", source="HUMAN")
                except Exception as exc:
                    self.events.emit("human_steer_error", str(exc), source="HUMAN", level="WARN")
            ctl = self.db.get_meta("control", {}) or {}
            marker = (ctl.get("ts"), ctl.get("action"))
            if marker != last_seen and ctl.get("action"):
                last_seen = marker
                action = ctl.get("action")
                if action == "stop":
                    self.set_state(HarnessState.STOPPED_BY_USER, "human stop")
                    self._stop_requested.set()
                    await self.agents.abort_all()
                    await self.runner.abort_all()
                    return
                if action == "pause":
                    self.set_state(HarnessState.PAUSED, "human pause; no new work will be scheduled")
                if action == "resume":
                    self.set_state(HarnessState.RUNNING, "human resume")
            await asyncio.sleep(1)

    async def _wait_if_paused(self) -> bool:
        while True:
            state = str(self.db.get_meta("state", HarnessState.RUNNING.value))
            if state not in {HarnessState.PAUSED.value, HarnessState.PAUSED_FOR_HUMAN.value}:
                return not self._stop_requested.is_set()
            if self._stop_requested.is_set():
                return False
            await asyncio.sleep(2)

    def _scout_focuses(self, n: int) -> list[str]:
        base = [
            "exploit the current champion but seek a mechanistic, not cosmetic, improvement",
            "revive promising or near-miss historical branches using new evidence",
            "seek a genuinely novel mechanism outside the currently dominant branch",
            "look for cross-branch combinations whose mechanisms may be complementary",
        ]
        return [base[i % len(base)] for i in range(n)]

    async def generate_candidates(self) -> list[dict[str, Any]]:
        frontier = self.frontier.select()
        state = self.context.global_state()
        state["frontier"] = frontier
        snapshot = self.context.render(state)
        scout_prompt = self._load_prompt("scout.md")
        nscouts = int(self.cfg.get("search", "scouts", 3))
        tasks = []
        for i, focus in enumerate(self._scout_focuses(nscouts)):
            prompt = scout_prompt + f"\n\nSCOUT FOCUS:\n{focus}\n\nRESEARCH STATE:\n{snapshot}"
            tasks.append(self.agents.run_task(
                agent_name=f"scout-{i+1}", role="scout", task_type="hypothesis_generation",
                prompt=prompt, cwd=self.cfg.repo, session_id=f"rsi-scout-{i+1}", timeout=1800,
            ))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        candidates: list[dict[str, Any]] = []
        fallback_parent = (self.db.best(self.cfg.direction) or self.db.experiment("baseline") or {}) .get("id", "baseline")
        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                self.events.emit("scout_error", str(result), source=f"SCOUT-{idx+1}", level="WARN")
                continue
            try:
                obj = parse_json_from_agent_text(result)
                for c in obj.get("hypotheses", []):
                    if not isinstance(c, dict):
                        continue
                    parents = [p for p in c.get("parent_ids", []) if self.db.experiment(str(p))]
                    if not parents:
                        parents = [fallback_parent]
                    c["parent_ids"] = parents
                    c["source_scout"] = idx + 1
                    candidates.append(c)
            except Exception as exc:
                self.events.emit("scout_parse_error", str(exc), source=f"SCOUT-{idx+1}", level="WARN")
        self.events.emit("candidates", f"generated {len(candidates)} candidate hypotheses", source="CONTROLLER")
        return candidates

    async def select_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        batch = int(self.cfg.get("search", "batch_size", 8))
        if len(candidates) <= batch:
            return candidates
        critic_prompt = self._load_prompt("critic.md") + "\n\nRESEARCH STATE:\n" + self.context.render(self.context.global_state(), 25000)
        critic_prompt += "\n\nCANDIDATES:\n" + json.dumps(list(enumerate(candidates)), ensure_ascii=False, indent=2)[:50000]
        try:
            text = await self.agents.run_task(
                agent_name="critic", role="critic", task_type="candidate_critique", prompt=critic_prompt,
                cwd=self.cfg.repo, session_id="rsi-critic", timeout=1800,
            )
            obj = parse_json_from_agent_text(text)
            picked: list[dict[str, Any]] = []
            for item in sorted(obj.get("selected", []), key=lambda x: x.get("priority", 999)):
                i = int(item["index"])
                if 0 <= i < len(candidates) and candidates[i] not in picked:
                    picked.append(candidates[i])
                if len(picked) >= batch:
                    break
            if picked:
                return picked
        except Exception as exc:
            self.events.emit("critic_fallback", f"critic failed; deterministic fallback: {exc}", source="CRITIC", level="WARN")
        def score(c: dict[str, Any]) -> float:
            return float(c.get("expected_delta", 0)) * max(0.1, float(c.get("confidence", 0.5))) + 0.25 * float(c.get("novelty", 0.5))
        return sorted(candidates, key=score, reverse=True)[:batch]

    async def run_candidate(self, candidate: dict[str, Any]) -> None:
        parents = [str(x) for x in candidate.get("parent_ids", [])]
        parent = self.db.experiment(parents[0]) if parents else self.db.experiment("baseline")
        baseline = self.db.experiment("baseline")
        if not parent or not baseline or parent.get("metric") is None or baseline.get("metric") is None:
            raise RuntimeError("parent/baseline metric unavailable")
        parent_commit = str(parent.get("git_commit") or self.db.get_meta("baseline_commit"))
        eid = self.db.add_experiment({
            "status": "planned",
            "stage": "planned",
            "title": candidate.get("title", "candidate"),
            "hypothesis": candidate.get("hypothesis", ""),
            "mechanism": candidate.get("mechanism", ""),
            "parent_primary": parent["id"],
            "confidence": float(candidate.get("confidence", 0.5)),
            "novelty": float(candidate.get("novelty", 0.5)),
            "changes": candidate.get("changes_hint", {}),
            "metadata": {"candidate": candidate},
        }, parents=parents)
        self.events.emit("experiment_created", f"{candidate.get('title','candidate')} parents={parents}", source=eid)
        worktree = create_worktree(self.cfg, eid, parent_commit)
        self.db.update_experiment(eid, worktree=str(worktree))
        context = self.context.render(self.context.node_context(parents), 30000)
        worker_prompt = self._load_prompt("worker.md") + "\n\nASSIGNED EXPERIMENT:\n" + json.dumps(candidate, ensure_ascii=False, indent=2)
        worker_prompt += "\n\nRELEVANT RESEARCH CONTEXT:\n" + context
        try:
            summary = await self.agents.run_task(
                agent_name=f"worker-{eid}", role="worker", task_type="implementation", prompt=worker_prompt,
                cwd=worktree, session_id=f"rsi-{eid}", timeout=3600,
            )
            problems = validate_modified_paths(self.cfg, worktree)
            problems.extend(verify_immutable_hashes(self.cfg, self.db, worktree))
            if problems:
                self.db.update_experiment(eid, status="invalid", failure_reason="; ".join(problems))
                self.events.emit("safety_violation", "; ".join(problems), source=eid, level="ERROR")
                return
            artifact = ensure_dir(self.cfg.artifact_dir / eid)
            (artifact / "worker_summary.txt").write_text(summary, encoding="utf-8")
            save_patch(worktree, artifact / "patch.diff")
            commit = commit_all(worktree, f"{eid}: {candidate.get('title','research experiment')}")
            self.db.update_experiment(eid, git_commit=commit, status="implemented")
            result = await self.runner.run_pipeline(eid, worktree, str(parent["id"]))
            await self.review_experiment(eid, candidate, result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.db.update_experiment(eid, status="failed", failure_reason=str(exc))
            self.events.emit("experiment_failed", str(exc), source=eid, level="ERROR")
            try:
                await self.review_experiment(eid, candidate, None)
            except Exception:
                pass

    async def review_experiment(self, eid: str, candidate: dict[str, Any], result: Any) -> None:
        exp = self.db.experiment(eid) or {}
        review_payload = {
            "experiment": exp,
            "parents": [self.db.experiment(p) for p in self.db.parents(eid)],
            "candidate": candidate,
            "metrics": self.db.query("SELECT stage,seed,metric,extra_json FROM metrics WHERE experiment_id=? ORDER BY id", (eid,)),
            "recent_relevant_state": self.context.global_state(),
        }
        prompt = self._load_prompt("reviewer.md") + "\n\nEVIDENCE:\n" + self.context.render(review_payload, 40000)
        try:
            current = self.db.experiment(eid) or {}
            force_tier = "planner" if current.get("status") in {"failed", "invalid"} or float(candidate.get("novelty", 0.0)) >= 0.85 else None
            text = await self.agents.run_task(
                agent_name="reviewer", role="reviewer", task_type="experiment_review",
                prompt=prompt, cwd=self.cfg.repo, session_id="rsi-reviewer", timeout=1800, force_tier=force_tier,
            )
            obj = parse_json_from_agent_text(text)
            current = self.db.experiment(eid) or {}
            status = str(obj.get("status") or current.get("status") or "completed")
            # Do not let a prose reviewer convert an invalid safety failure into success.
            if current.get("status") in {"invalid", "failed"}:
                status = current["status"]
            self.db.update_experiment(
                eid,
                status=status,
                lesson=str(obj.get("lesson") or ""),
                failure_reason=obj.get("failure_reason") or current.get("failure_reason"),
                next_steps=obj.get("next_steps", []),
            )
            now = time.time()
            for b in obj.get("belief_updates", []) or []:
                if not isinstance(b, dict) or not b.get("key") or not b.get("statement"):
                    continue
                evidence = json.dumps({"experiment": eid, "attribution": obj.get("attribution")}, ensure_ascii=False)
                self.db.execute(
                    "INSERT INTO beliefs(key,statement,confidence,evidence_json,updated_at) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET statement=excluded.statement,confidence=excluded.confidence,evidence_json=excluded.evidence_json,updated_at=excluded.updated_at",
                    (str(b["key"]), str(b["statement"]), float(b.get("confidence", 0.5)), evidence, now),
                )
            for pe in obj.get("parameter_effects", []) or []:
                if not isinstance(pe, dict) or not pe.get("parameter"):
                    continue
                self.db.execute(
                    "INSERT INTO parameter_effects(experiment_id,parameter,from_value,to_value,effect,confidence,stage,created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (eid, str(pe["parameter"]), str(pe.get("from", "")), str(pe.get("to", "")), float(pe.get("effect", current.get("delta_parent") or 0.0)), float(pe.get("confidence", 0.5)), str(current.get("stage", "")), now),
                )
        except Exception as exc:
            self.events.emit("review_warning", str(exc), source=eid, level="WARN")

    def refresh_champion(self) -> None:
        best = self.db.best(self.cfg.direction)
        if not best:
            return
        old = self.db.query("SELECT id FROM experiments WHERE status='champion' AND id!=?", (best["id"],))
        for row in old:
            self.db.update_experiment(row["id"], status="promising")
        if best["id"] != "baseline" and best.get("status") not in {"failed", "invalid"}:
            self.db.update_experiment(best["id"], status="champion")
        self.events.emit("champion", f"best={best['id']} metric={best.get('metric')}", source="CONTROLLER")

    async def maybe_meta_review(self) -> None:
        every = int(self.cfg.get("search", "meta_review_every", 25))
        count = self.db.completed_count()
        if every <= 0 or count - self._last_meta_count < every:
            return
        aggregate = {
            "state": self.context.global_state(),
            "experiments": self.db.query("SELECT id,status,stage,title,mechanism,metric,delta_parent,delta_baseline,gpu_hours,agent_tokens,lesson FROM experiments ORDER BY created_at"),
            "agent_usage": self.db.query("SELECT agent_name,task_type,provider,model_id,SUM(total_tokens) tokens,SUM(cost) cost,COUNT(*) runs FROM agent_runs GROUP BY agent_name,task_type,provider,model_id"),
        }
        prompt = self._load_prompt("meta_rsi.md") + "\n\nAGGREGATE RESEARCH DATA:\n" + self.context.render(aggregate, 70000)
        try:
            text = await self.agents.run_task(
                agent_name="meta-rsi", role="meta", task_type="meta_rsi", prompt=prompt,
                cwd=self.cfg.repo, session_id=f"rsi-meta-{count}", timeout=2400,
            )
            obj = parse_json_from_agent_text(text)
            policy = obj.get("policy", {})
            version = f"policy_{count:06d}"
            self.db.execute("UPDATE policy_versions SET active=0")
            self.db.execute(
                "INSERT OR REPLACE INTO policy_versions(ts,version,policy_json,rationale,active) VALUES(?,?,?,?,1)",
                (time.time(), version, json.dumps(policy, ensure_ascii=False), str(obj.get("rationale", ""))),
            )
            self.db.set_meta("active_policy", policy)
            for eid in policy.get("revive_nodes", []):
                if self.db.experiment(str(eid)):
                    self.db.update_experiment(str(eid), pinned=1)
            self._last_meta_count = count
            self.db.set_meta("last_meta_count", count)
            self.events.emit("meta_rsi", f"activated {version}", source="META-RSI")
        except Exception as exc:
            self.events.emit("meta_rsi_error", str(exc), source="META-RSI", level="WARN")

    async def analyze_plateau(self, reason: str) -> None:
        count = self.db.completed_count()
        if int(self.db.get_meta("last_plateau_analysis_count", -1) or -1) == count:
            return
        prompt = self._load_prompt("plateau.md") + "\n\nPLATEAU REASON:\n" + reason
        prompt += "\n\nRESEARCH STATE:\n" + self.context.render(self.context.global_state(), 60000)
        try:
            text = await self.agents.run_task(
                agent_name="lead", role="lead", task_type="plateau_analysis", prompt=prompt,
                cwd=self.cfg.repo, session_id="rsi-lead", timeout=2400, force_tier="planner",
            )
            try:
                report = parse_json_from_agent_text(text)
            except Exception:
                report = {"diagnosis": text}
            self.db.set_meta("plateau_report", report)
            self.db.set_meta("last_plateau_analysis_count", count)
            self.db.add_guidance(json.dumps(report, ensure_ascii=False), kind="ai_plateau_report")
            for eid in report.get("promising_revivals", []) if isinstance(report, dict) else []:
                if self.db.experiment(str(eid)):
                    self.db.update_experiment(str(eid), pinned=1)
            self.events.emit("plateau_analysis", "lead produced a plateau report; waiting for human decision", source="LEAD")
        except Exception as exc:
            self.events.emit("plateau_analysis_error", str(exc), source="LEAD", level="WARN")

    async def run(self) -> None:
        if not self.db.experiment("baseline"):
            await self.initialize()
        self.dashboard.start()
        if self.dashboard.address:
            host, port = self.dashboard.address
            self.events.emit("dashboard", f"Research Console: http://{host}:{port}", source="CONTROLLER")
        self.db.set_meta("started_at", self.db.get_meta("started_at", time.time()))
        self.db.set_meta("control", {})  # explicit run ignores stale stop/pause commands from an earlier process
        self.set_state(HarnessState.RUNNING)
        self._control_task = asyncio.create_task(self._control_watch())
        try:
            while not self._stop_requested.is_set():
                if not await self._wait_if_paused():
                    break
                decision = self.stop_policy.evaluate()
                if decision.state:
                    if decision.state == HarnessState.PAUSED_FOR_HUMAN:
                        await self.analyze_plateau(decision.reason)
                    self.set_state(decision.state, decision.reason)
                    if decision.state == HarnessState.PAUSED_FOR_HUMAN:
                        continue
                    if decision.state in TERMINAL_STATES:
                        break
                violations = verify_immutable_hashes(self.cfg, self.db)
                if violations:
                    self.set_state(HarnessState.SAFETY_STOP, "; ".join(violations))
                    break
                candidates = await self.generate_candidates()
                if not candidates:
                    self.set_state(HarnessState.PAUSED_FOR_HUMAN, "agents produced no valid hypotheses; human insight requested")
                    continue
                selected = await self.select_candidates(candidates)
                self.events.emit("batch", f"selected {len(selected)} experiments", source="CONTROLLER")
                tasks = [asyncio.create_task(self.run_candidate(c)) for c in selected]
                await asyncio.gather(*tasks, return_exceptions=True)
                self.refresh_champion()
                await self.maybe_meta_review()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.set_state(HarnessState.FATAL_ERROR, str(exc))
            raise
        finally:
            if self._control_task:
                self._control_task.cancel()
            await self.agents.close()
            self.dashboard.stop()
