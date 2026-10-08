import asyncio
import contextlib
import fcntl
import json
import os
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
from .gitops import (commit_all, create_worktree, ensure_git_repo, head_commit,
                     is_dirty, save_patch, validate_modified_paths)
from .integrity import record_immutable_hashes, verify_immutable_hashes
from .pi_rpc import AgentAborted
from .scheduler import GPUScheduler
from .state import ACTIVE_STATUSES, TERMINAL_STATES, HarnessState
from .stopping import StopPolicy
from .util import (ensure_dir, parse_json_from_agent_text,
                   parse_last_json_object, shell_template)


class MainRepoModified(RuntimeError):
    """An agent edited the main checkout instead of its worktree."""


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
        self._iteration: asyncio.Task[bool] | None = None
        self._last_meta_count = int(self.db.get_meta("last_meta_count", 0) or 0)
        self._worker_failures = 0

    def set_state(self, state: HarnessState, reason: str = "") -> None:
        self.db.set_meta("state", state.value)
        self.db.set_meta("pause_reason", reason)
        self.events.emit("state", f"state={state.value}" + (f" reason={reason}" if reason else ""), source="CONTROLLER")

    def _worktree_rule(self, worktree: Path) -> str:
        return (
            f"\n\nWORKTREE: {worktree}\nEdit files only under this directory, using paths inside it. "
            f"Never modify {self.cfg.repo} outside {self.cfg.worktree_dir}: the controller treats that as a "
            "safety violation and stops the whole campaign.\n"
        )

    def _agent_timeout(self, kind: str) -> float:
        return 60.0 * float(self.cfg.get("agents", f"{kind}_timeout_min", 10))

    def _project_file(self, key: str) -> Path | None:
        raw = self.cfg.get("project", key)
        return (self.cfg.base_dir / raw).resolve() if raw else None

    def _codebase_map(self) -> str:
        """Paper-specific map of what agents may change; see docs/ADAPTER.md. Without it small models wander."""
        path = self._project_file("codebase_map")
        return "\n\n" + path.read_text(encoding="utf-8") if path and path.is_file() else ""

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
                        "device": self.cfg.get("experiment", "device", 0),
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
                    await self._interrupt("human stop")
                    return
                if action == "pause":
                    self.set_state(HarnessState.PAUSED, "human pause; active agents and experiments interrupted")
                    await self._interrupt("human pause")
                if action == "resume":
                    self._worker_failures = 0  # the human has seen the streak; do not re-pause on the next single failure
                    self.set_state(HarnessState.RUNNING, "human resume")
            await asyncio.sleep(1)

    def _settle_inflight(self, reason: str) -> None:
        """Close every in-flight row. Called when nothing can still be running: at startup or after an interrupt."""
        n_runs = self.db.execute(
            "UPDATE agent_runs SET ts_end=?,success=0,error=COALESCE(error,?) WHERE ts_end IS NULL",
            (time.time(), reason),
        ).rowcount
        marks = ",".join("?" * len(ACTIVE_STATUSES))
        rows = self.db.query("SELECT id FROM experiments WHERE status IN (" + marks + ")", ACTIVE_STATUSES)
        for r in rows:
            self.db.update_experiment(r["id"], status="interrupted", failure_reason=reason)
        if n_runs or rows:
            self.events.emit("settled", f"{reason}: {n_runs} agent runs and {len(rows)} experiments marked interrupted", source="CONTROLLER", level="WARN")

    def _guard_main_repo(self, source: str) -> None:
        """Agents run unsandboxed; any edit to the main checkout corrupts every later worktree and baseline."""
        if not is_dirty(self.cfg.repo):
            return
        dest = ensure_dir(self.cfg.artifact_dir / "escapes") / f"{int(time.time())}_{source}.diff"
        save_patch(self.cfg.repo, dest)
        self.set_state(HarnessState.SAFETY_STOP, f"{source} modified the main repository outside its worktree; diff saved to {dest}")
        raise MainRepoModified(str(dest))

    async def _interrupt(self, reason: str) -> None:
        """Hard-stop all in-flight work: cancel the iteration, abort every Pi run, kill experiment processes."""
        task = self._iteration
        if task and not task.done():
            task.cancel()
        await self.agents.abort_all(reason)
        await self.runner.abort_all()
        if task:
            await asyncio.wait({task}, timeout=30)
        self._settle_inflight(f"interrupted by {reason}")

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
        idea_path = self._project_file("idea_file")
        if idea_path and idea_path.is_file():
            state["user_research_idea"] = idea_path.read_text(encoding="utf-8")
            state["user_research_idea_source"] = str(idea_path)
        snapshot = self.context.render(state)
        scout_prompt = self._load_prompt("scout.md") + self._codebase_map()
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
        worker_prompt = self._load_prompt("worker.md") + self._codebase_map()
        worker_prompt += self._worktree_rule(worktree)
        worker_prompt += "\n\nASSIGNED EXPERIMENT:\n" + json.dumps(candidate, ensure_ascii=False, indent=2)
        worker_prompt += "\n\nRELEVANT RESEARCH CONTEXT:\n" + context
        try:
            # The worker must hand over a runnable patch within this window, or the experiment never starts.
            summary = await self.agents.run_task(
                agent_name=f"worker-{eid}", role="worker", task_type="implementation", prompt=worker_prompt,
                cwd=worktree, session_id=f"rsi-{eid}", timeout=self._agent_timeout("worker"), experiment_id=eid,
            )
            self._guard_main_repo(f"worker-{eid}")
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
            self._worker_failures = 0  # the failure counter is for consecutive failures
        except (asyncio.CancelledError, MainRepoModified):
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            # A budget abort means the agent itself is not converging; another agent run would repeat it.
            repaired = not isinstance(exc, AgentAborted) and await self.repair_candidate(eid, candidate, worktree, error)
            if repaired:
                try:
                    result = await self.runner.run_pipeline(eid, worktree, str(parent["id"]))
                    await self.review_experiment(eid, candidate, result)
                    self._worker_failures = 0
                    return
                except Exception as repair_exc:
                    error = f"repair rerun failed: {type(repair_exc).__name__}: {repair_exc}"
            ran = bool(self.db.query("SELECT 1 FROM metrics WHERE experiment_id=? LIMIT 1", (eid,)))
            # Without any metric there is no evidence to review; a reviewer would only invent conclusions.
            lesson = None if ran else f"NOT RUN: no experiment was executed, so the hypothesis is untested. Cause: {error}"
            self.db.update_experiment(eid, status="failed", failure_reason=error, lesson=lesson)
            self.events.emit("experiment_failed", error, source=eid, level="ERROR")
            self._worker_failures += 1
            if self._worker_failures >= int(self.cfg.get("stop", "max_worker_failures", 3)):
                self.set_state(HarnessState.PAUSED_FOR_HUMAN, f"{self._worker_failures} consecutive worker failures")
            if ran:
                with contextlib.suppress(Exception):
                    await self.review_experiment(eid, candidate, None)

    async def repair_candidate(self, eid: str, candidate: dict[str, Any], worktree: Path, error: str) -> bool:
        row = self.db.experiment(eid) or {}
        try:
            metadata = json.loads(row.get("metadata_json") or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid experiment metadata for {eid}") from exc
        attempt = int(metadata.get("repair_attempts", 0)) + 1
        max_attempts = int(self.cfg.get("repair", "max_attempts", 2))
        if attempt > max_attempts:
            return False
        metadata["repair_attempts"] = attempt
        self.db.update_experiment(eid, metadata=metadata, status="repairing", failure_reason=error)
        prompt = (
            "You are repairing a failed experiment, not proposing a new idea.\n"
            "Inspect the existing worktree and the failure below. Make the smallest repair needed.\n"
            "Do not change evaluator, tests, immutable files, research objective, or create a new hypothesis.\n"
            "Your job is the experiment code only. If the failure is in the harness itself, do not fix it; explain it and stop.\n"
            f"{self._worktree_rule(worktree)}\n"
            f"FAILURE:\n{error}\n\nASSIGNED EXPERIMENT:\n{json.dumps(candidate, ensure_ascii=False, indent=2)}"
        )
        try:
            summary = await self.agents.run_task(
                agent_name=f"repair-{eid}-{attempt}", role="worker", task_type="repair", prompt=prompt,
                cwd=worktree, session_id=f"rsi-repair-{eid}-{attempt}", timeout=self._agent_timeout("repair"),
                experiment_id=eid,
            )
            self._guard_main_repo(f"repair-{eid}-{attempt}")
            problems = validate_modified_paths(self.cfg, worktree)
            problems.extend(verify_immutable_hashes(self.cfg, self.db, worktree))
            if problems:
                raise RuntimeError("repair safety check failed: " + "; ".join(problems))
            await self.runner.preflight(eid, worktree)
            artifact = ensure_dir(self.cfg.artifact_dir / eid)
            (artifact / f"repair_{attempt}_summary.txt").write_text(summary, encoding="utf-8")
            save_patch(worktree, artifact / f"repair_{attempt}.patch.diff")
            commit = commit_all(worktree, f"{eid}: repair attempt {attempt}")
            self.db.update_experiment(eid, git_commit=commit, status="implemented", metadata=metadata)
            self.events.emit("repair_succeeded", f"repair attempt {attempt} passed preflight", source=eid)
            return True
        except (asyncio.CancelledError, MainRepoModified):
            raise
        except Exception as exc:
            message = f"repair attempt {attempt} failed: {type(exc).__name__}: {exc}"
            self.db.update_experiment(eid, status="failed", failure_reason=message, metadata=metadata)
            self.events.emit("repair_failed", message, source=eid, level="ERROR")
            return False

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
                experiment_id=eid,
            )
            obj = parse_json_from_agent_text(text)
            current = self.db.experiment(eid) or {}
            status = str(obj.get("status") or current.get("status") or "completed")
            # Do not let a prose reviewer convert an invalid safety failure into success.
            if current.get("status") in {"invalid", "failed"}:
                status = current["status"]
            noise = float(self.cfg.get("objective", "min_meaningful_delta", 0.0))
            delta = current.get("delta_baseline")
            if status in {"promising", "rejected"} and delta is not None and abs(float(delta)) < noise:
                # Within run-to-run noise the evidence supports neither verdict.
                status = "near_miss"
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

    async def _iterate(self) -> bool:
        """One research iteration (plan -> batch -> settle). Returns False when the campaign must end."""
        decision = self.stop_policy.evaluate()
        if decision.state:
            if decision.state == HarnessState.PAUSED_FOR_HUMAN:
                await self.analyze_plateau(decision.reason)
            self.set_state(decision.state, decision.reason)
            return decision.state not in TERMINAL_STATES
        violations = verify_immutable_hashes(self.cfg, self.db)
        if violations:
            self.set_state(HarnessState.SAFETY_STOP, "; ".join(violations))
            return False
        candidates = await self.generate_candidates()
        if not candidates:
            self.set_state(HarnessState.PAUSED_FOR_HUMAN, "agents produced no valid hypotheses; human insight requested")
            return True
        batch_started_at = time.time()
        self.db.set_meta("batch_started_at", batch_started_at)
        selected = await self.select_candidates(candidates)
        slots = int(self.cfg.get("resources", "max_parallel", 1))
        selected = selected[:max(1, slots)]
        self.events.emit("batch", f"selected {len(selected)} experiments; awaiting full batch settlement", source="CONTROLLER")
        results = await asyncio.gather(*(self.run_candidate(c) for c in selected), return_exceptions=True)
        self.refresh_champion()
        batch_rows = self.db.query(
            "SELECT * FROM experiments WHERE created_at>=? AND id!='baseline' ORDER BY created_at",
            (batch_started_at,),
        )
        failed = [r for r in batch_rows if r.get("status") in {"failed", "invalid"}]
        unresolved = [r for r in batch_rows if r.get("status") in ACTIVE_STATUSES]
        if any(isinstance(result, MainRepoModified) for result in results):
            return False
        if any(isinstance(result, BaseException) for result in results):
            self.set_state(HarnessState.PAUSED_FOR_HUMAN, "batch task raised an exception; no new plans will be generated")
            return True
        measured = [r for r in batch_rows if r.get("metric") is not None]
        if unresolved or not measured:
            # Plan again only on new evidence; consecutive-failure streaks are bounded by [stop]/[budget] limits.
            self.set_state(HarnessState.PAUSED_FOR_HUMAN, f"batch produced no usable evidence: failed={len(failed)} unresolved={len(unresolved)}")
            return True
        await self.maybe_meta_review()
        return True

    def _acquire_lock(self) -> None:
        """One controller per campaign: a second one would double-schedule GPUs and agents."""
        self._lock = open(ensure_dir(self.cfg.state_dir) / "controller.lock", "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another controller is already running on {self.cfg.state_dir}; stop it first") from None
        self._lock.write(str(os.getpid()))
        self._lock.flush()

    async def run(self) -> None:
        self._acquire_lock()
        if not self.db.experiment("baseline"):
            await self.initialize()
        elif not self.db.query("SELECT 1 FROM immutable_hashes LIMIT 1"):
            n = record_immutable_hashes(self.cfg, self.db)
            self.events.emit("integrity", f"recorded {n} immutable file hashes for imported baseline", source="INIT")
        self.dashboard.start()
        if self.dashboard.address:
            host, port = self.dashboard.address
            self.events.emit("dashboard", f"Research Console: http://{host}:{port}", source="CONTROLLER")
        self.db.set_meta("started_at", self.db.get_meta("started_at", time.time()))
        self.db.set_meta("control", {})  # explicit run ignores stale stop/pause commands from an earlier process
        self._settle_inflight("controller exited before settlement")
        self.set_state(HarnessState.RUNNING)
        self._control_task = asyncio.create_task(self._control_watch())
        try:
            while not self._stop_requested.is_set():
                if not await self._wait_if_paused():
                    break
                self._iteration = asyncio.create_task(self._iterate())
                await asyncio.wait({self._iteration})
                if self._iteration.cancelled():
                    continue  # pause/stop interrupted this iteration; loop re-checks state
                if not self._iteration.result():
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.set_state(HarnessState.FATAL_ERROR, str(exc))
            raise
        finally:
            if self._control_task:
                self._control_task.cancel()
            if self._iteration and not self._iteration.done():
                await self._interrupt("controller shutdown")
            await self.agents.close()
            # The control task that ran the interrupt may itself have been cancelled mid-way; close what remains.
            self._settle_inflight("controller shutdown")
            self.dashboard.stop()
