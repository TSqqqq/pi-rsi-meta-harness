import asyncio
import ast
import fcntl
import hashlib
import subprocess
from collections import Counter
import json
import os
import re
import shlex
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from .agent_runtime import AgentRuntime
from .config import Config
from .context import KINDS, ContextBuilder, clip
from .dashboard import DashboardServer
from .db import ResearchDB
from .events import EventBus
from .experiments import ExperimentRunner
from .gitops import (commit_all, create_worktree, ensure_git_repo, head_commit, is_dirty,
                     numeric_only_diff, save_patch, sync_harness_files, validate_modified_paths)
from .integrity import record_immutable_hashes, verify_immutable_hashes
from .pi_rpc import AgentAborted
from .scheduler import GPUScheduler
from .state import ACTIVE_STATUSES, TERMINAL_STATES, HarnessState
from .stopping import StopPolicy
from .util import (ensure_dir, json_objects_with, parse_json_from_agent_text,
                   parse_last_json_object, run_capture, shell_template)


class MainRepoModified(RuntimeError):
    """An agent edited the main checkout instead of its worktree."""


class InvalidPatch(RuntimeError):
    """The worker's change broke a harness rule (recorded as an invalid experiment)."""


class ResearchController:
    def __init__(self, cfg: Config, harness_root: Path):
        self.cfg = cfg
        self.harness_root = harness_root
        self.db = ResearchDB(cfg.db_path)
        self.events = EventBus(self.db, cfg.log_dir, echo=True)
        self.scheduler = GPUScheduler(cfg)
        self.runner = ExperimentRunner(cfg, self.db, self.events, self.scheduler)
        self.agents = AgentRuntime(cfg, self.db, self.events)
        self.context = ContextBuilder(cfg, self.db, noise=self.runner.noise)
        self.stop_policy = StopPolicy(cfg, self.db)
        self.dashboard = DashboardServer(cfg, self.db, harness_root / "dashboard" / "index.html")
        self._stop_requested = asyncio.Event()
        self._control_task: asyncio.Task[None] | None = None
        self._iteration: asyncio.Task[bool] | None = None
        self._worker_failures = 0

    def set_state(self, state: HarnessState, reason: str = "") -> None:
        self.db.set_meta("state", state.value)
        self.db.set_meta("pause_reason", reason)
        self.events.emit("state", f"state={state.value}" + (f" reason={reason}" if reason else ""), source="CONTROLLER")

    def _budget_rule(self, kind: str) -> str:
        tools = int(self.cfg.get("agents", "max_tool_calls", 40))
        return f"\n\nBUDGET: at most {tools} tool calls and {self._agent_timeout(kind) / 60:.0f} minutes for this task.\n"

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
        """Paper-specific map of what agents may change. Without it small models wander."""
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

    def _repo_state(self) -> str:
        """Fingerprint of the main checkout's uncommitted changes (tracked diff + untracked files)."""
        out = subprocess.run(["git", "status", "--porcelain=v1", "-uall"], cwd=self.cfg.repo, capture_output=True, text=True).stdout
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=self.cfg.repo, capture_output=True, text=True).stdout
        return hashlib.sha256((out + diff).encode()).hexdigest()

    def _guard_main_repo(self, source: str, before: str) -> None:
        """Detect an unsandboxed agent writing to the main checkout during its own run.

        With the bwrap sandbox the main checkout is read-only for agents, so any change seen there is a
        human's (e.g. a harness edit or commit while workers run) and must not stop the campaign.
        """
        if bool(self.cfg.get("pi", "sandbox", True)) or self._repo_state() == before:
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

    async def _idea_digest(self) -> None:
        """Distil the research note once into a mechanism checklist (cached by content hash)."""
        path = self._project_file("idea_file")
        if not path or not path.is_file():
            return
        raw = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(raw.encode()).hexdigest()
        if (self.db.get_meta("idea_digest", {}) or {}).get("sha256") == digest:
            return
        prompt = self._load_prompt("idea_digest.md") + self._codebase_map() + "\n\nRESEARCH NOTE:\n" + raw[:40000]
        try:
            text = await self.agents.run_task(
                agent_name="idea-digest", role="lead", task_type="research_planning", prompt=prompt,
                cwd=self.cfg.repo, session_id="rsi-idea-digest", timeout=900,
            )
            mechanisms = parse_json_from_agent_text(text, "mechanisms")["mechanisms"]
            self.db.set_meta("idea_digest", {"sha256": digest, "mechanisms": mechanisms})
            self.events.emit("idea_digest", f"research note distilled into {len(mechanisms)} mechanisms", source="LEAD")
        except Exception as exc:
            self.db.set_meta("idea_digest", {"sha256": digest, "raw": raw[:3000]})
            self.events.emit("idea_digest_warning", f"falling back to a clipped note: {exc}", source="LEAD", level="WARN")

    def _normalize(self, c: Any) -> tuple[dict[str, Any] | None, str]:
        """Mechanical validation of one scout hypothesis; returns (candidate, '') or (None, reason).

        Only what the harness must enforce: a known kind, knob values the command line can take, parents that
        have code to start from, and no exact re-run of a measured knob setting. Judgement stays with the scout.
        """
        if not isinstance(c, dict) or not c.get("title"):
            return None, "no title"
        kind = str(c.get("kind") or "").lower()
        if kind not in KINDS:
            return None, f"kind must be one of {KINDS}"
        knobs = self.cfg.section("experiment")
        if kind == "tuning":
            params = c.get("params") if isinstance(c.get("params"), dict) else {}
            bad = [k for k, v in params.items() if k not in knobs or not isinstance(v, (int, float)) or isinstance(v, bool)
                   or (isinstance(knobs[k], int) and float(v) != int(v))]
            if not params or bad:
                return None, (f"tuning needs ONE numeric value per knob over [experiment] keys {sorted(knobs)} "
                              f"(a sweep is several experiments); bad={bad}")
            c["params"] = {k: type(knobs[k])(v) for k, v in params.items()}  # int knobs are parsed with int()
        else:
            c.pop("params", None)  # knobs change only in tuning experiments
        # A never-measured node has no code to start from: build on its nearest measured ancestor instead.
        parents: list[str] = []
        for p in [str(x) for x in c.get("parent_ids") or []] or ["baseline"]:
            node = self.db.experiment(p)
            while node and node.get("metric") is None:
                node = self.db.experiment(str(node.get("parent_primary") or "baseline"))
            if node and node["id"] not in parents:
                parents.append(node["id"])
        c.update(kind=kind, parent_ids=parents or ["baseline"])
        if kind == "tuning":
            parent = self.db.experiment(c["parent_ids"][0]) or {}
            base = {**knobs, **json.loads(parent.get("params_json") or "{}")}
            want = {**base, **c["params"]}
            if want == base:
                return None, f"tuning sets the knobs {c['params']} that {parent.get('id')} already uses"
            for r in self.db.query("SELECT id,params_json FROM experiments WHERE kind='tuning' AND parent_primary=? "
                                   "AND (metric IS NOT NULL OR status IN ('planned','implemented','running','repairing'))", (parent.get("id"),)):
                if {**knobs, **json.loads(r.get("params_json") or "{}")} == want:
                    return None, f"exact re-run of {r['id']} (same parent, same knobs)"
        return c, ""

    async def confirm_promising(self) -> None:
        """Re-measure single-run 'promising' results before anything is built on them (no LLM involved)."""
        want = int(self.cfg.get("stages", "quick_replicates", 1))
        if want < 2:
            return
        todo = []
        for r in self.db.query("SELECT * FROM experiments WHERE status IN ('promising','champion') AND stage='quick'"):
            n = self.db.metric_stats(r["id"], "quick")[2]
            if n < want and r.get("worktree") and Path(r["worktree"]).is_dir():
                todo.append((r, n))
        if not todo:
            return
        self.events.emit("confirm", f"re-measuring {len(todo)} single-run promising results before planning", source="CONTROLLER")

        async def one(row: dict[str, Any], n: int) -> None:
            eid = row["id"]
            await self.runner.run_stage(eid, Path(row["worktree"]), "quick", list(range(n + 1, want + 1)))
            mean, _, n_all = self.db.metric_stats(eid, "quick")
            delta = self.cfg.delta(mean, self.db.metric_mean("baseline", "quick"))
            self.db.update_experiment(eid, metric=mean, delta_baseline=delta)
            status = self._verdict(self.db.experiment(eid) or {})
            self.db.update_experiment(eid, status=status)
            self.events.emit("confirm", f"{eid}: {n_all}-run mean Δ{delta:+.4f} vs noise ±{self.runner.noise('quick', n_all):.4f} -> {status}", source=eid)

        results = await asyncio.gather(*(one(r, n) for r, n in todo), return_exceptions=True)
        for (r, _), res in zip(todo, results):
            if isinstance(res, BaseException) and not isinstance(res, asyncio.CancelledError):
                self.events.emit("confirm_error", f"{r['id']}: {res}", source=r["id"], level="WARN")

    async def plan_batch(self, slots: int) -> list[dict[str, Any]]:
        """One planner-tier scout reads the raw research record and proposes the next batch, best first."""
        await self._idea_digest()
        knobs = ", ".join(f"{k}={v}" for k, v in self.cfg.section("experiment").items())
        prompt = (self._load_prompt("scout.md") + self._codebase_map()
                  + f"\n\nBATCH: propose exactly {slots} hypotheses, best first; they run in parallel."
                  + f"\n\nTUNABLE KNOBS ([experiment] defaults; tuning experiments set them via params): {knobs}"
                  + "\n\n" + self.context.briefing(idea=True))
        try:
            result = await self.agents.run_task(
                agent_name="scout", role="scout", task_type="hypothesis_generation", prompt=prompt,
                cwd=self.cfg.repo, session_id="rsi-scout", timeout=self._agent_timeout("scout"),
            )
        except Exception as exc:
            self.events.emit("scout_error", f"{type(exc).__name__}: {exc}", source="SCOUT", level="WARN")
            return []
        if '"hypotheses"' not in result:
            # The answer was cut (e.g. the output budget ran out while reasoning): ask for it in the same session,
            # which still holds everything read and reasoned so far.
            self.events.emit("scout_retry", "no JSON in the answer; asking for the final JSON only", source="SCOUT", level="WARN")
            try:
                result = await self.agents.run_task(
                    agent_name="scout", role="scout", task_type="hypothesis_generation", cwd=self.cfg.repo,
                    session_id="rsi-scout", timeout=900, continue_session=True,
                    prompt=("Your previous answer was cut off before the JSON. Do not use tools and do not reason again: "
                            f"output only the final STRICT JSON object with the {slots} hypotheses you settled on."),
                )
            except Exception as exc:
                self.events.emit("scout_error", f"retry: {type(exc).__name__}: {exc}", source="SCOUT", level="WARN")
                return []
        try:
            hypotheses = parse_json_from_agent_text(result, "hypotheses")["hypotheses"]
        except ValueError:
            # Small models often break the outer JSON while each hypothesis object is still intact.
            hypotheses = json_objects_with(result, {"title", "hypothesis"})
            if not hypotheses:
                self.events.emit("scout_parse_error", "no hypotheses in the scout answer", source="SCOUT", level="WARN")
                return []
            self.events.emit("scout_salvage", f"outer JSON malformed; salvaged {len(hypotheses)} hypotheses", source="SCOUT", level="WARN")
        chosen: list[dict[str, Any]] = []
        for h in hypotheses if isinstance(hypotheses, list) else []:
            c, why = self._normalize(h)
            if c is not None and len(chosen) >= slots:
                c, why = None, f"beyond the batch size ({slots})"
            if c is None:
                # Kept with the full plan: the next scout sees what was rejected and why.
                hd = h if isinstance(h, dict) else {"title": str(h)}
                self.events.emit("candidate_rejected", f"{clip(hd.get('title'), 70)}: {why}", source="SCOUT",
                                 payload={k: hd.get(k) for k in ("kind", "title", "parent_ids", "params", "mechanism")} | {"reason": why})
                continue
            chosen.append(c)
        self.events.emit("candidates", f"scout planned {len(chosen)} experiments {dict(Counter(c['kind'] for c in chosen))}", source="CONTROLLER")
        return chosen

    async def run_candidate(self, candidate: dict[str, Any]) -> None:
        parents = [str(x) for x in candidate.get("parent_ids", [])]
        parent = self.db.experiment(parents[0]) if parents else self.db.experiment("baseline")
        baseline = self.db.experiment("baseline")
        if not parent or not baseline or parent.get("metric") is None or baseline.get("metric") is None:
            raise RuntimeError("parent/baseline metric unavailable")
        parent_commit = str(parent.get("git_commit") or self.db.get_meta("baseline_commit"))
        # Knobs are inherited from the primary parent; a tuning candidate overrides only the ones it names.
        params = {**json.loads(parent.get("params_json") or "{}"), **(candidate.get("params") or {})}
        eid = self.db.add_experiment({
            "status": "planned",
            "stage": "planned",
            "kind": candidate["kind"],
            "params": params,
            "title": candidate.get("title", "candidate"),
            "hypothesis": candidate.get("hypothesis", ""),
            "mechanism": candidate.get("mechanism", ""),
            "parent_primary": parent["id"],
            "confidence": float(candidate.get("confidence", 0.5)),
            "novelty": float(candidate.get("novelty", 0.5)),
            "changes": candidate.get("changes_hint", {}),
            "metadata": {"candidate": candidate, "batch": int(self.db.get_meta("batch_seq", 0) or 0)},
        }, parents=parents)
        self.events.emit("experiment_created", f"{candidate.get('title','candidate')} parents={parents}", source=eid)
        worktree = create_worktree(self.cfg, eid, parent_commit)
        self.db.update_experiment(eid, worktree=str(worktree))
        base_commit = str(self.db.get_meta("baseline_commit") or "")
        synced = sync_harness_files(self.cfg, worktree, base_commit)
        if synced:
            self.events.emit("harness_sync", f"harness files updated to the baseline commit: {synced}", source=eid)
        migrate = str(self.cfg.get("commands", "migrate", "") or "")
        if migrate:
            # Project hook: adapt inherited experiment code to the current harness interface (e.g. a new driver).
            r = run_capture(shlex.split(shell_template(migrate, python=sys.executable, worktree=worktree, baseline_commit=base_commit,
                                                      repo=self.cfg.repo)), cwd=self.cfg.repo, check=False)
            if r.returncode != 0:
                raise RuntimeError(f"migrate hook failed: {r.stderr.strip()[-500:]}")
            if run_capture(["git", "status", "--porcelain"], cwd=worktree).stdout.strip():
                commit_all(worktree, f"{eid}: inherited code migrated to the current harness interface")
                dropped = r.stdout.strip()
                self.events.emit("harness_migrate", "inherited code migrated" + (f"; dropped parent changes in: {dropped}" if dropped else ""), source=eid)
                if dropped:
                    self.db.update_experiment(eid, metadata={**json.loads((self.db.experiment(eid) or {}).get("metadata_json") or "{}"),
                                                             "migration_dropped": dropped.split(",")})
        meta = json.loads((self.db.experiment(eid) or {}).get("metadata_json") or "{}")
        self.db.update_experiment(eid, metadata={**meta, "start_commit": head_commit(worktree)})
        try:
            if candidate["kind"] == "tuning":
                # Knob changes need no agent: the harness passes them on the command line, so they always take effect.
                summary = f"harness tuning run: {candidate.get('params')} on top of {parent['id']}"
                await self._commit_patch(eid, worktree, summary, f"{eid}: {candidate.get('title','research experiment')}")
                await self.runner.run_pipeline(eid, worktree, str(parent["id"]))
            else:
                await self._implement(eid, candidate, worktree, parents, params, str(parent["id"]))
            self.settle_measured(eid)
            self._worker_failures = 0  # the failure counter is for consecutive failures
        except (asyncio.CancelledError, MainRepoModified):
            raise
        except InvalidPatch:
            return  # recorded as invalid; not a worker failure
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            ran = bool(self.db.query("SELECT 1 FROM metrics WHERE experiment_id=? LIMIT 1", (eid,)))
            lesson = None if ran else f"NOT RUN: no experiment was executed, so the hypothesis is untested. Cause: {error}"
            self.db.update_experiment(eid, status="failed", failure_reason=error, lesson=lesson)
            self.events.emit("experiment_failed", error, source=eid, level="ERROR")
            self._worker_failures += 1
            if self._worker_failures >= int(self.cfg.get("stop", "max_worker_failures", 3)):
                self.set_state(HarnessState.PAUSED_FOR_HUMAN, f"{self._worker_failures} consecutive worker failures")
        finally:
            await self.agents.release(f"worker-{eid}", worktree)

    async def _implement(self, eid: str, candidate: dict[str, Any], worktree: Path, parents: list[str],
                         params: dict[str, Any], parent_id: str) -> None:
        """One worker session implements the hypothesis and fixes its own runtime errors.

        Turn 1 implements; each failure of the runnable checks (preflight, smoke) or of the quick stage is
        sent back to the SAME session with the traceback and the failing code, up to [agents].fix_rounds
        times, so the worker keeps everything it already read and reasoned. All turns share one tool budget.
        """
        prompt = self._load_prompt("worker.md") + self._budget_rule("worker") + self._codebase_map() + self._worktree_rule(worktree)
        prompt += f"\n\nEFFECTIVE KNOBS for this run (passed on the command line): {params}"
        prompt += "\n\nASSIGNED EXPERIMENT:\n" + json.dumps(candidate, ensure_ascii=False, indent=2)
        prompt += "\n\n" + self.context.worker_context(parents, self._patch_of)
        prompt += self._lineage_diff(worktree)
        prompt += self._target_source(worktree, candidate)
        rounds = int(self.cfg.get("agents", "fix_rounds", 5))
        for turn in range(rounds + 1):
            repo_before = self._repo_state()
            summary = await self.agents.run_task(
                agent_name=f"worker-{eid}", role="worker", task_type="implementation", prompt=prompt,
                cwd=worktree, session_id=f"rsi-{eid}", timeout=self._agent_timeout("worker"), experiment_id=eid,
                continue_session=turn > 0, keep_tool_count=turn > 0,
            )
            self._guard_main_repo(f"worker-{eid}", repo_before)
            await self._commit_patch(eid, worktree, summary, f"{eid}: " + (candidate.get("title", "research experiment")
                                                                         if turn == 0 else f"fix {turn}"), turn)
            try:
                await self.runner.run_pipeline(eid, worktree, parent_id)
                return
            except (asyncio.CancelledError, MainRepoModified):
                raise
            except Exception as exc:
                if turn == rounds or self.db.query("SELECT 1 FROM metrics WHERE experiment_id=? LIMIT 1", (eid,)):
                    raise  # out of fix rounds, or it already ran and produced a result: nothing to fix
                error = f"{type(exc).__name__}: {exc}"
                self.db.update_experiment(eid, status="repairing", failure_reason=error)
                self.events.emit("worker_fix", f"round {turn + 1}: {clip(error, 200)}", source=eid, level="WARN")
                prompt = ("Your patch failed when the controller ran it. Fix the cause in the experiment code with the\n"
                          "smallest change; keep the hypothesis. If the failure is in the harness, explain and stop.\n\n"
                          f"FAILURE:\n{error}" + self._traceback_source(worktree, error))

    async def _commit_patch(self, eid: str, worktree: Path, summary: str, message: str, turn: int = 0) -> None:
        """Validate the worktree change, keep its artifacts and commit it; InvalidPatch when it breaks a rule."""
        problems = validate_modified_paths(self.cfg, worktree)
        problems.extend(verify_immutable_hashes(self.cfg, self.db, worktree))
        exp = self.db.experiment(eid) or {}
        if exp.get("kind") != "tuning" and numeric_only_diff(worktree):
            problems.append("numeric-only patch: changed constants instead of mechanism; numeric changes must be tuning experiments")
        if problems:
            self.db.update_experiment(eid, status="invalid", failure_reason="; ".join(problems))
            self.events.emit("safety_violation", "; ".join(problems), source=eid, level="ERROR")
            raise InvalidPatch("; ".join(problems))
        artifact = ensure_dir(self.cfg.artifact_dir / eid)
        suffix = "" if turn == 0 else f"_fix{turn}"
        (artifact / f"worker_summary{suffix}.txt").write_text(summary, encoding="utf-8")
        save_patch(worktree, artifact / f"patch{suffix}.diff")
        exp = self.db.experiment(eid) or {}
        start = (json.loads(exp.get("metadata_json") or "{}")).get("start_commit")
        commit = commit_all(worktree, message)
        if turn and start:
            # patch.diff stays the experiment's complete change vs its starting code, fixes included.
            (artifact / "patch.diff").write_text(run_capture(["git", "diff", start, "HEAD"], cwd=worktree).stdout, encoding="utf-8")
        self.db.update_experiment(eid, git_commit=commit, status="implemented")

    def settle_measured(self, eid: str) -> None:
        """Deterministic bookkeeping for a measured experiment; interpretation is left to the next scouts."""
        exp = self.db.experiment(eid) or {}
        if exp.get("metric") is None:
            return
        if exp.get("status") not in {"invalid", "failed"}:
            self.db.update_experiment(eid, status=self._verdict(exp))

    def _verdict(self, exp: dict[str, Any]) -> str:
        """promising / near_miss (inside noise) / rejected, from the measured mean and the measured noise."""
        stage = str(exp.get("stage") or "quick")
        n = max(1, self.db.metric_stats(exp["id"], stage)[2])
        delta, noise = float(exp.get("delta_baseline") or 0.0), self.runner.noise(stage, n)
        return "promising" if delta >= noise else "near_miss" if delta > -noise else "rejected"

    def _lineage_diff(self, worktree: Path, limit: int = 12000) -> str:
        """The complete code change of the worker's starting point vs baseline (all ancestors together),
        so it never has to reconstruct the lineage from git history."""
        base = str(self.db.get_meta("baseline_commit") or "")
        if not base:
            return ""
        diff = run_capture(["git", "diff", base, "HEAD"], cwd=worktree, check=False).stdout
        if not diff.strip():
            return "\n\nPARENT LINEAGE DIFF: none (you start from the unmodified baseline code).\n"
        cut = "" if len(diff) <= limit else f"\n… ({len(diff) - limit} more chars; run `git diff {base[:12]} HEAD` to see all)"
        return f"\n\nPARENT LINEAGE DIFF (your starting code vs baseline, i.e. every ancestor change already in your worktree):\n{diff[:limit]}{cut}\n"

    def _traceback_source(self, worktree: Path, error: str, limit: int = 3) -> str:
        """Source (with line numbers) of the worktree functions a traceback points into, innermost first,
        so a fix round starts at the failing code instead of searching for it."""
        root = worktree.resolve()
        frames = re.findall(r'File "([^"]+)", line (\d+)', error)
        out, seen = [], set()
        for fname, line in reversed(frames):
            path = Path(fname).resolve()
            if not path.is_relative_to(root) or not path.suffix == ".py" or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
                tree = ast.parse(text)
            except (SyntaxError, UnicodeDecodeError):
                continue
            n = int(line)
            fns = [f for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and f.lineno <= n <= (f.end_lineno or f.lineno)]
            if not fns:
                continue
            fn = max(fns, key=lambda f: f.lineno)  # innermost enclosing function
            key = (str(path), fn.lineno)
            if key in seen:
                continue
            seen.add(key)
            lines = text.splitlines()
            lo, end = fn.lineno, fn.end_lineno or fn.lineno
            hi = min(end, lo + 199)
            body = "\n".join(f"{i:5d}{'>' if i == n else ' '} {lines[i - 1]}" for i in range(lo, hi + 1))
            rel = path.relative_to(root).as_posix()
            out.append(f"FAILING CODE {rel}:{fn.name} (line {n} marked >, lines {lo}-{end}):\n{body}"
                       + ("" if hi == end else f"\n      … (continues to line {end})"))
            if len(out) >= limit:
                break
        return "\n\n" + "\n\n".join(out) + "\n" if out else ""

    def _target_source(self, worktree: Path, candidate: dict[str, Any]) -> str:
        """Source of the function(s) the scout named in changes_hint, with line numbers, so the worker edits
        instead of searching. changes_hint is one {file, function, change} or a list of them (up to 4); every
        identifier in a hint's "function" that names a function/class of its file is shown (600 lines in
        total). Nothing is added when nothing matches."""
        hint = candidate.get("changes_hint")
        # One {file, function} hint, or a list of them when the change spans several places.
        hints = [h for h in (hint if isinstance(hint, list) else [hint]) if isinstance(h, dict)]
        out, budget = [], 600
        for h in hints[:4]:
            block, budget = self._hint_source(worktree, h, budget)
            out += block
        return "\n\n" + "\n\n".join(out) + "\n" if out else ""

    def _hint_source(self, worktree: Path, hint: dict[str, Any], budget: int) -> tuple[list[str], int]:
        rel = str(hint.get("file") or "")
        path = (worktree / rel).resolve()
        if not rel.endswith(".py") or not path.is_file() or not path.is_relative_to(worktree.resolve()):
            return [], budget
        try:
            text = path.read_text(encoding="utf-8")
            defs = {n.name: n for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        except (SyntaxError, UnicodeDecodeError):
            return [], budget
        # "Class.method" names the method; free text like "f + g" names both.
        words = [w.split(".")[-1] for w in re.findall(r"[A-Za-z_][\w.]*", str(hint.get("function") or ""))]
        names = [w for w in dict.fromkeys(words) if w in defs][:3]
        lines, out = text.splitlines(), []
        for name in names:
            node = defs[name]
            lo, end = node.lineno, node.end_lineno or node.lineno
            hi = min(end, lo + budget - 1)
            if hi < lo:
                break
            budget -= hi - lo + 1
            body = "\n".join(f"{i:5d}  {lines[i - 1]}" for i in range(lo, hi + 1))
            more = "" if hi == end else f"\n      … (continues to line {end})"
            out.append(f"TARGET {rel}:{name} (lines {lo}-{end}, as in your worktree):\n{body}{more}")
        return out, budget

    def _patch_of(self, eid: str) -> str:
        path = self.cfg.artifact_dir / eid / "patch.diff"
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""

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

    async def _iterate(self) -> bool:
        """One research iteration (plan -> batch -> settle). Returns False when the campaign must end."""
        decision = self.stop_policy.evaluate()
        if decision.state:
            self.set_state(decision.state, decision.reason)
            return decision.state not in TERMINAL_STATES
        violations = verify_immutable_hashes(self.cfg, self.db)
        if violations:
            self.set_state(HarnessState.SAFETY_STOP, "; ".join(violations))
            return False
        await self.confirm_promising()
        slots = max(1, min(int(self.cfg.get("search", "batch_size", 8)), int(self.cfg.get("resources", "max_parallel", 1))))
        selected = await self.plan_batch(slots)
        if not selected:
            self.set_state(HarnessState.PAUSED_FOR_HUMAN, "the scout produced no valid hypotheses; human insight requested")
            return True
        batch_started_at = time.time()
        self.db.set_meta("batch_started_at", batch_started_at)
        self.db.set_meta("batch_seq", int(self.db.get_meta("batch_seq", 0) or 0) + 1)
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
        errors = [(c.get("title"), r) for c, r in zip(selected, results) if isinstance(r, BaseException)]
        for title, err in errors:
            self.events.emit("batch_task_error", f"{clip(title, 70)}: {type(err).__name__}: {err}", source="CONTROLLER", level="ERROR")
        if errors:
            self.set_state(HarnessState.PAUSED_FOR_HUMAN, f"{len(errors)} batch task(s) raised (see batch_task_error); no new plans will be generated")
            return True
        measured = [r for r in batch_rows if r.get("metric") is not None]
        if unresolved or not measured:
            # Plan again only on new evidence; consecutive-failure streaks are bounded by [stop]/[budget] limits.
            self.set_state(HarnessState.PAUSED_FOR_HUMAN, f"batch produced no usable evidence: failed={len(failed)} unresolved={len(unresolved)}")
            return True
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
