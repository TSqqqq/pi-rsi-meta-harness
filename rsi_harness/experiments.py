from __future__ import annotations

import asyncio
import json
import os
import signal
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .db import ResearchDB
from .events import EventBus
from .scheduler import GPUScheduler
from .util import ensure_dir, parse_last_json_object, shell_template


@dataclass
class StageResult:
    stage: str
    metrics: list[float]
    mean: float
    std: float
    gpu_hours: float
    extra: list[dict[str, Any]]


class ExperimentRunner:
    def __init__(self, cfg: Config, db: ResearchDB, events: EventBus, scheduler: GPUScheduler):
        self.cfg = cfg
        self.db = db
        self.events = events
        self.scheduler = scheduler
        self._active: dict[str, asyncio.subprocess.Process] = {}

    async def _run_command(
        self,
        command: str,
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
        source: str,
        timeout_s: float,
    ) -> tuple[int, str, float]:
        ensure_dir(log_path.parent)
        started = time.time()
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=(os.name != "nt"),
        )
        self._active[source] = proc
        output: list[str] = []

        async def consume() -> None:
            assert proc.stdout
            with log_path.open("a", encoding="utf-8") as f:
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", "replace")
                    output.append(text)
                    f.write(text)
                    f.flush()
                    print(f"[{source}] {text}", end="", flush=True)

        consumer = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            self.events.emit("experiment_timeout", f"timeout after {timeout_s/60:.1f} min", source=source, level="WARN")
            self._terminate(proc, force=False)
            try:
                await asyncio.wait_for(proc.wait(), timeout=15)
            except asyncio.TimeoutError:
                self._terminate(proc, force=True)
                await proc.wait()
        finally:
            await consumer
            self._active.pop(source, None)
        return int(proc.returncode or 0), "".join(output), time.time() - started

    @staticmethod
    def _terminate(proc: asyncio.subprocess.Process, *, force: bool) -> None:
        if proc.returncode is not None:
            return
        if os.name != "nt" and proc.pid:
            try:
                os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass
        if force:
            proc.kill()
        else:
            proc.terminate()

    async def abort_all(self) -> None:
        active = list(self._active.values())
        for proc in active:
            self._terminate(proc, force=False)
        await asyncio.sleep(1)
        for proc in active:
            self._terminate(proc, force=True)

    def _values(self, eid: str, worktree: Path, stage: str, seed: int, gpu_ids: list[int]) -> dict[str, Any]:
        venv_root_raw = str(self.cfg.get("dependencies", "venv_root", "") or "")
        venv_root = Path(venv_root_raw).expanduser() if venv_root_raw else (self.cfg.state_dir / "venvs")
        if not venv_root.is_absolute():
            venv_root = (self.cfg.base_dir / venv_root).resolve()
        vpy = venv_root / eid / "bin" / "python"
        python_bin = str(vpy) if vpy.exists() else sys.executable
        return {
            "repo": self.cfg.repo,
            "worktree": worktree,
            "experiment_id": eid,
            "stage": stage,
            "seed": seed,
            "gpu_ids": ",".join(str(x) for x in gpu_ids),
            "python": python_bin,
        }

    async def preflight(self, eid: str, worktree: Path) -> None:
        template = str(self.cfg.get("commands", "preflight", "") or "")
        if not template:
            return
        log = self.cfg.log_dir / "experiments" / f"{eid}.preflight.log"
        env = os.environ.copy()
        rc, out, _ = await self._run_command(template, cwd=worktree, env=env, log_path=log, source=f"{eid}:PREFLIGHT", timeout_s=600)
        if rc != 0:
            raise RuntimeError(f"preflight failed for {eid}: rc={rc}")

    async def run_stage(self, eid: str, worktree: Path, stage: str, seeds: list[int]) -> StageResult:
        train_template = str(self.cfg.get("commands", stage, "") or "")
        if not train_template:
            raise RuntimeError(f"commands.{stage} is empty")
        eval_template = str(self.cfg.get("commands", "evaluate", "") or "")
        timeout_min = float(self.cfg.get("stages", f"{stage}_timeout_min", 60))
        metric_key = str(self.cfg.get("objective", "metric_json_key", "metric"))
        metrics: list[float] = []
        extras: list[dict[str, Any]] = []
        total_gpu_hours = 0.0

        async with self.scheduler.acquire() as alloc:
            gpu_ids = alloc.gpu_ids
            env = os.environ.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            for seed in seeds:
                vals = self._values(eid, worktree, stage, seed, gpu_ids)
                train_cmd = shell_template(train_template, **vals)
                log = self.cfg.log_dir / "experiments" / f"{eid}.{stage}.seed{seed}.train.log"
                self.events.emit("stage_start", f"{stage} seed={seed} gpu={gpu_ids or 'cpu'}", source=eid)
                rc, _, elapsed = await self._run_command(
                    train_cmd, cwd=worktree, env=env, log_path=log, source=f"{eid}:{stage}:train", timeout_s=timeout_min * 60
                )
                total_gpu_hours += elapsed * len(gpu_ids) / 3600.0
                if rc != 0:
                    raise RuntimeError(f"training failed: stage={stage} seed={seed} rc={rc}")

                eval_cmd = shell_template(eval_template, **vals)
                elog = self.cfg.log_dir / "experiments" / f"{eid}.{stage}.seed{seed}.eval.log"
                rc, out, elapsed_eval = await self._run_command(
                    eval_cmd, cwd=worktree, env=env, log_path=elog, source=f"{eid}:{stage}:eval", timeout_s=max(600, timeout_min * 60 / 2)
                )
                total_gpu_hours += elapsed_eval * len(gpu_ids) / 3600.0
                if rc != 0:
                    raise RuntimeError(f"evaluation failed: stage={stage} seed={seed} rc={rc}")
                obj = parse_last_json_object(out)
                if metric_key not in obj:
                    raise RuntimeError(f"evaluator JSON missing key {metric_key!r}: {obj}")
                metric = float(obj[metric_key])
                extra = {k: v for k, v in obj.items() if k != metric_key}
                metrics.append(metric)
                extras.append(extra)
                self.db.add_metric(eid, stage, seed, metric, extra)
                self.events.emit("metric", f"{stage} seed={seed} {metric_key}={metric:.6g}", source=eid, payload={"metric": metric, "stage": stage, "seed": seed})

        mean = statistics.fmean(metrics)
        std = statistics.stdev(metrics) if len(metrics) >= 2 else 0.0
        return StageResult(stage, metrics, mean, std, total_gpu_hours, extras)

    async def run_pipeline(self, eid: str, worktree: Path, parent_id: str) -> StageResult | None:
        await self.preflight(eid, worktree)
        total_gpu = 0.0
        final: StageResult | None = None
        for stage in ["quick", "medium", "full"]:
            if not self.cfg.get("commands", stage, ""):
                continue
            seeds = [1]
            if stage == "full":
                seeds = [int(x) for x in (self.cfg.get("stages", "validation_seeds", [1, 2, 3]) or [1])]
            baseline_stage = self.db.metric_mean("baseline", stage)
            if baseline_stage is None:
                raise RuntimeError(f"No stage-matched baseline metric for {stage}. Re-run init with baseline calibration for this stage.")
            parent_stage = self.db.metric_mean(parent_id, stage)
            if parent_stage is None:
                # A parent may have been pruned before this fidelity; use stage baseline rather than comparing unlike fidelities.
                parent_stage = baseline_stage
            self.db.update_experiment(eid, stage=stage, status="running")
            result = await self.run_stage(eid, worktree, stage, seeds)
            total_gpu += result.gpu_hours
            delta_parent = self.cfg.delta(result.mean, parent_stage)
            delta_baseline = self.cfg.delta(result.mean, baseline_stage)
            self.db.update_experiment(
                eid,
                metric=result.mean,
                metric_std=result.std,
                parent_metric=parent_stage,
                delta_parent=delta_parent,
                delta_baseline=delta_baseline,
                gpu_hours=total_gpu,
                stage=stage,
            )
            self.events.emit("stage_done", f"{stage} mean={result.mean:.6g} Δparent={delta_parent:+.6g} Δstage-baseline={delta_baseline:+.6g}", source=eid)
            final = result
            if stage == "quick":
                threshold = float(self.cfg.get("stages", "quick_min_delta", -0.05))
                if delta_parent < threshold:
                    self.db.update_experiment(eid, status="rejected", failure_reason=f"quick delta {delta_parent:+.6g} < {threshold:+.6g}")
                    return result
            elif stage == "medium":
                threshold = float(self.cfg.get("stages", "medium_min_delta", 0.05))
                if delta_parent < threshold:
                    status = "near_miss" if delta_parent >= 0 else "rejected"
                    self.db.update_experiment(eid, status=status, failure_reason=f"medium delta {delta_parent:+.6g} < {threshold:+.6g}")
                    return result
        if final:
            self.db.update_experiment(eid, status="completed", validated=1 if final.stage == "full" else 0)
        return final
