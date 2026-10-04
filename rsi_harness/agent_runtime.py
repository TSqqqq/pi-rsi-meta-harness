from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from .config import Config
from .db import ResearchDB
from .events import EventBus
from .models import ModelRouter, ModelRoute
from .pi_rpc import PiRPC


ROLE_SKILLS = {
    "scout": "research-hypothesis",
    "reviewer": "experiment-review",
    "meta": "meta-rsi",
    "validator": "paper-validation",
}


class AgentRuntime:
    def __init__(self, cfg: Config, db: ResearchDB, events: EventBus):
        self.cfg = cfg
        self.db = db
        self.events = events
        self.router = ModelRouter(cfg)
        self.clients: dict[tuple[str, str], PiRPC] = {}
        self.initialized: set[tuple[str, str]] = set()
        self.last_stats: dict[tuple[str, str], dict[str, Any]] = {}

    def _event_callback(self, agent_name: str):
        def cb(rec: dict[str, Any]) -> None:
            t = rec.get("type")
            if t == "tool_execution_start":
                self.events.emit("agent_tool_start", f"tool={rec.get('toolName')}", source=agent_name)
            elif t == "tool_execution_end":
                self.events.emit("agent_tool_end", f"tool={rec.get('toolName')} error={rec.get('isError', False)}", source=agent_name)
            elif t == "rpc_stderr":
                self.events.emit("pi_stderr", str(rec.get("text", ""))[:1000], source=agent_name, level="WARN")
            elif t == "agent_start":
                self.events.emit("agent_start", "agent run started", source=agent_name)
            elif t == "agent_settled":
                self.events.emit("agent_settled", "agent run settled", source=agent_name)
        return cb

    async def client(self, agent_name: str, cwd: Path, session_id: str) -> PiRPC:
        key = (agent_name, str(cwd.resolve()))
        if key not in self.clients:
            harness_root = Path(__file__).resolve().parent.parent
            old_pp = os.environ.get("PYTHONPATH", "")
            pp = str(harness_root) + ((os.pathsep + old_pp) if old_pp else "")
            skill_root = harness_root / ".agents" / "skills"
            skill_paths = [p for p in skill_root.iterdir() if p.is_dir()] if skill_root.exists() else []
            c = PiRPC(
                cwd=cwd,
                session_dir=self.cfg.session_dir,
                session_id=session_id,
                name=agent_name,
                pi_bin=str(self.cfg.get("pi", "binary", "pi")),
                on_event=self._event_callback(agent_name),
                extra_env={"RSI_CONFIG": str(self.cfg.path), "PYTHONPATH": pp},
                skill_paths=skill_paths,
                append_system_prompt=harness_root / "AGENTS.md",
            )
            await c.start()
            self.clients[key] = c
        return self.clients[key]

    async def validate_models(self) -> list[str]:
        c = await self.client("model-check", self.cfg.repo, "rsi-model-check")
        models = await c.available_models()
        return self.router.validate_specs(models)

    async def run_task(
        self,
        *,
        agent_name: str,
        role: str,
        task_type: str,
        prompt: str,
        cwd: Path,
        session_id: str,
        failures: int = 0,
        timeout: float = 1800.0,
        force_tier: str | None = None,
    ) -> str:
        client = await self.client(agent_name, cwd, session_id)
        route = self.router.route(task_type, failures=failures, force_tier=force_tier)
        await client.apply_route(route)
        # Measure before one-time skill initialization so its tokens are part of the task budget.
        before = await client.stats()

        key = (agent_name, str(cwd.resolve()))
        if key not in self.initialized and role in ROLE_SKILLS:
            # Pi supports invoking discovered skill commands by /skill:name through RPC prompt.
            # This one-time role initialization keeps the full skill body out of every later controller message.
            skill = ROLE_SKILLS[role]
            init_prompt = f"/skill:{skill} Initialize this session for the {role} role. Reply only: READY"
            try:
                await client.prompt_and_wait(init_prompt, timeout=timeout)
            except Exception as exc:
                self.events.emit("skill_init_warning", f"{skill}: {exc}", source=agent_name, level="WARN")
            self.initialized.add(key)

        ts = time.time()
        provider = model_id = None
        state = await client.command({"type": "get_state"})
        model = (state.get("data") or {}).get("model") or {}
        provider, model_id = model.get("provider"), model.get("id")
        cur = self.db.execute(
            "INSERT INTO agent_runs(ts_start,agent_name,task_type,session_id,provider,model_id,thinking) VALUES(?,?,?,?,?,?,?)",
            (ts, agent_name, task_type, session_id, provider, model_id, route.thinking),
        )
        run_id = int(cur.lastrowid)
        self.events.emit("agent_task", f"{task_type} via {route.tier} model {provider}/{model_id}", source=agent_name)
        try:
            text = await client.prompt_and_wait(prompt, timeout=timeout)
            after = await client.stats()
            bt = before.get("tokens", {}) or {}
            at = after.get("tokens", {}) or {}
            inp = max(0, int(at.get("input", 0)) - int(bt.get("input", 0)))
            out = max(0, int(at.get("output", 0)) - int(bt.get("output", 0)))
            total = max(0, int(at.get("total", 0)) - int(bt.get("total", 0)))
            cost = max(0.0, float(after.get("cost", 0.0)) - float(before.get("cost", 0.0)))
            usage = after.get("contextUsage") or {}
            compact_at = float(self.cfg.get("sessions", "compact_at_percent", 82))
            if usage.get("percent") is not None and float(usage["percent"]) >= compact_at:
                try:
                    info = await client.compact("Preserve current research role, active hypotheses, unresolved decisions, code changes, and references to durable experiment IDs. The SQLite DAG remains the source of truth.")
                    self.events.emit("session_compact", f"context compacted from {info.get('tokensBefore')} to ~{info.get('estimatedTokensAfter')}", source=agent_name)
                    after = await client.stats()  # include compaction usage in this run's accounting
                    at = after.get("tokens", {}) or {}
                    inp = max(0, int(at.get("input", 0)) - int(bt.get("input", 0)))
                    out = max(0, int(at.get("output", 0)) - int(bt.get("output", 0)))
                    total = max(0, int(at.get("total", 0)) - int(bt.get("total", 0)))
                    cost = max(0.0, float(after.get("cost", 0.0)) - float(before.get("cost", 0.0)))
                except Exception as compact_exc:
                    self.events.emit("session_compact_warning", str(compact_exc), source=agent_name, level="WARN")
            self.db.execute(
                "UPDATE agent_runs SET ts_end=?,input_tokens=?,output_tokens=?,total_tokens=?,cost=?,success=1 WHERE id=?",
                (time.time(), inp, out, total, cost, run_id),
            )
            self.events.emit("agent_done", f"{task_type} done; tokens={total}", source=agent_name)
            return text
        except Exception as exc:
            self.db.execute("UPDATE agent_runs SET ts_end=?,success=0,error=? WHERE id=?", (time.time(), str(exc), run_id))
            self.events.emit("agent_error", str(exc), source=agent_name, level="ERROR")
            raise

    async def steer(self, agent_name: str, cwd: Path, message: str) -> None:
        key = (agent_name, str(cwd.resolve()))
        c = self.clients.get(key)
        if not c:
            raise RuntimeError(f"Agent not running: {agent_name}")
        await c.steer(message)

    async def steer_by_name(self, agent_name: str, message: str) -> bool:
        for (name, _), client in self.clients.items():
            if name == agent_name:
                await client.steer(message)
                return True
        return False

    async def abort_all(self) -> None:
        await asyncio.gather(*(c.abort() for c in list(self.clients.values())), return_exceptions=True)

    async def close(self) -> None:
        await asyncio.gather(*(c.close() for c in list(self.clients.values())), return_exceptions=True)
        self.clients.clear()
        self.initialized.clear()
