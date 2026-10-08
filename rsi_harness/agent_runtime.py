import asyncio
import os
import time
import traceback
from pathlib import Path
from typing import Any

from .config import Config
from .db import ResearchDB
from .events import EventBus
from .models import ModelRouter
from .pi_rpc import PiRPC, PiRPCError


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
        # One Pi session serves one prompt at a time; concurrent callers (e.g. batch reviewers) queue here.
        self.locks: dict[tuple[str, str], asyncio.Lock] = {}

    def _event_callback(self, agent_name: str):
        def cb(rec: dict[str, Any]) -> None:
            t = rec.get("type")
            if t == "tool_execution_start":
                self.events.emit("agent_tool_start", f"tool={rec.get('toolName')}", source=agent_name)
            elif t == "tool_execution_end":
                self.events.emit("agent_tool_end", f"tool={rec.get('toolName')} error={rec.get('isError', False)}", source=agent_name)
            elif t == "rpc_reader_error":
                self.events.emit("pi_reader_error", str(rec.get("error")), source=agent_name, level="ERROR")
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
            # Agents may write only inside their own experiment worktree; scouts/critics/reviewers are read-only.
            in_worktree = cwd.resolve().is_relative_to(self.cfg.worktree_dir.resolve())
            sandbox = [cwd] if in_worktree else []
            no_proxy = ",".join(x for x in [os.environ.get("NO_PROXY", ""), "127.0.0.1,localhost"] if x)
            c = PiRPC(
                cwd=cwd,
                session_dir=self.cfg.session_dir,
                session_id=session_id,
                name=agent_name,
                pi_bin=str(self.cfg.get("pi", "binary", "pi")),
                on_event=self._event_callback(agent_name),
                extra_env={"RSI_CONFIG": str(self.cfg.path), "PYTHONPATH": pp, "NO_PROXY": no_proxy, "no_proxy": no_proxy},
                skill_paths=skill_paths,
                append_system_prompt=harness_root / "AGENTS.md",
                transcript_dir=self.cfg.transcript_dir,
                sandbox_writable=sandbox if bool(self.cfg.get("pi", "sandbox", True)) else None,
                extensions=bool(self.cfg.get("pi", "extensions", False)),
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
        experiment_id: str | None = None,
    ) -> str:
        key = (agent_name, str(cwd.resolve()))
        async with self.locks.setdefault(key, asyncio.Lock()):
            return await self._run_locked(
                key, agent_name=agent_name, role=role, task_type=task_type, prompt=prompt, cwd=cwd,
                session_id=session_id, failures=failures, timeout=timeout, force_tier=force_tier,
                experiment_id=experiment_id,
            )

    async def _run_locked(
        self, key: tuple[str, str], *, agent_name: str, role: str, task_type: str, prompt: str, cwd: Path,
        session_id: str, failures: int, timeout: float, force_tier: str | None, experiment_id: str | None,
    ) -> str:
        client = await self.client(agent_name, cwd, session_id)
        route = self.router.route(task_type, failures=failures, force_tier=force_tier)
        await client.apply_route(route)
        # Measure before one-time skill initialization so its tokens are part of the task budget.
        before = await client.stats()
        state = (await client.command({"type": "get_state"})).get("data") or {}
        model = state.get("model") or {}
        provider, model_id = model.get("provider"), model.get("id")
        allowed = self.cfg.get("models", "allowed_providers")
        if allowed is not None and provider not in allowed:
            # Hard cost guard: never send a prompt to a provider outside the allowlist, whatever the routing said.
            raise PermissionError(f"provider {provider!r} is not in models.allowed_providers={allowed}")
        cur = self.db.execute(
            "INSERT INTO agent_runs(ts_start,agent_name,task_type,session_id,provider,model_id,thinking,"
            "experiment_id,transcript_path,session_file) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (time.time(), agent_name, task_type, session_id, provider, model_id, route.thinking,
             experiment_id, str(client.transcript_path), state.get("sessionFile")),
        )
        run_id = int(cur.lastrowid or 0)
        if not run_id:
            raise RuntimeError("failed to create agent_runs row")
        client.run_id = run_id  # every transcript line from here on is attributable to this row
        client.tool_calls = 0  # the tool budget covers the whole task, including skill initialization
        max_tools = int(self.cfg.get("agents", "max_tool_calls", 40))
        self.events.emit("agent_task", f"{task_type} via {route.tier} model {provider}/{model_id} run#{run_id}", source=agent_name)
        try:
            if key not in self.initialized and role in ROLE_SKILLS:
                # Pi supports invoking discovered skill commands by /skill:name through RPC prompt.
                # This one-time role initialization keeps the full skill body out of every later controller message.
                skill = ROLE_SKILLS[role]
                init_prompt = f"/skill:{skill} Initialize this session for the {role} role. Reply only: READY"
                try:
                    await client.prompt_and_wait(init_prompt, timeout=min(timeout, 300), max_tool_calls=max_tools)
                except PiRPCError as exc:
                    self.events.emit("skill_init_warning", f"{skill}: {exc}", source=agent_name, level="WARN")
                self.initialized.add(key)
            text = await client.prompt_and_wait(prompt, timeout=timeout, max_tool_calls=max_tools)
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
                "UPDATE agent_runs SET ts_end=?,input_tokens=?,output_tokens=?,total_tokens=?,cost=?,success=1,tool_calls=? WHERE id=?",
                (time.time(), inp, out, total, cost, client.tool_calls, run_id),
            )
            self.events.emit("agent_done", f"{task_type} done; tokens={total} tools={client.tool_calls}", source=agent_name)
            return text
        except BaseException as exc:
            # BaseException: a human pause/stop cancels the task, and the row must still be closed.
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.db.execute(
                "UPDATE agent_runs SET ts_end=?,success=0,error=?,tool_calls=? WHERE id=?",
                (time.time(), error_text, client.tool_calls, run_id),
            )
            self.events.emit("agent_error", f"run#{run_id} {type(exc).__name__}: {exc}", source=agent_name, level="ERROR")
            raise
        finally:
            client.run_id = None

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

    async def abort_all(self, reason: str = "aborted by harness") -> None:
        await asyncio.gather(*(c.abort(reason) for c in list(self.clients.values())), return_exceptions=True)

    async def close(self) -> None:
        await asyncio.gather(*(c.close() for c in list(self.clients.values())), return_exceptions=True)
        self.clients.clear()
        self.initialized.clear()
