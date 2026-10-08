from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import ModelRoute


class PiRPCError(RuntimeError):
    pass


class AgentAborted(PiRPCError):
    """The run was aborted by a harness budget (time/tool calls) or by a human pause/stop."""


EventCallback = Callable[[dict[str, Any]], None]
# One JSONL record can carry a whole file read or session dump; asyncio's 64 KiB default kills the reader.
RPC_LINE_LIMIT = 64 * 1024 * 1024


def bwrap_prefix(writable: list[Path]) -> list[str]:
    """Read-only host filesystem; only `writable` (plus a private /tmp) accepts writes.

    The network namespace stays shared because agents must reach the model endpoint.
    """
    if not shutil.which("bwrap"):
        raise PiRPCError("sandbox requested but bwrap is not installed; refusing to run an unsandboxed agent")
    home = Path.home()
    args = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    if (home / ".cache").is_dir():
        args += ["--tmpfs", str(home / ".cache")]
    agent_dir = home / ".pi" / "agent"
    if agent_dir.is_dir():
        # Pi writes lock files beside its config; a tmpfs layer takes them while every real entry stays read-only.
        args += ["--tmpfs", str(agent_dir)]
        for entry in sorted(agent_dir.iterdir()):
            args += ["--ro-bind", str(entry), str(entry)]
    for path in writable:
        args += ["--bind", str(path), str(path)]
    return args + ["--unshare-pid", "--die-with-parent", "--new-session", "--"]


class PiRPC:
    """Minimal, language-independent Pi RPC client using strict JSONL framing."""

    def __init__(
        self,
        *,
        cwd: Path,
        session_dir: Path,
        session_id: str,
        name: str,
        pi_bin: str = "pi",
        on_event: EventCallback | None = None,
        extra_env: dict[str, str] | None = None,
        skill_paths: list[Path] | None = None,
        append_system_prompt: Path | None = None,
        transcript_dir: Path | None = None,
        sandbox_writable: list[Path] | None = None,
        extensions: bool = True,
    ):
        self.cwd = cwd
        self.session_dir = session_dir
        self.session_id = session_id
        self.name = name
        self.pi_bin = pi_bin
        self.on_event = on_event
        self.extra_env = extra_env or {}
        self.skill_paths = skill_paths or []
        self.append_system_prompt = append_system_prompt
        self.sandbox_writable = sandbox_writable  # None = unsandboxed
        self.extensions = extensions
        self.max_tool_calls: int | None = None
        self.tool_calls = 0
        # One local JSONL per session; every line is tagged with the agent_runs id for exact replay.
        self.transcript_path = (transcript_dir or session_dir / "transcripts") / f"{session_id}.jsonl"
        self.abort_reason: str | None = None
        self.run_id: int | None = None  # agent_runs.id of the task currently using this session
        self._abort_task: asyncio.Task[None] | None = None
        self.proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._settled = asyncio.Event()
        self._settled.set()
        self._closed = False

    def _record(self, direction: str, record: dict[str, Any]) -> None:
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        line = {"ts": time.time(), "agent": self.name, "run_id": self.run_id, "direction": direction, "record": record}
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    async def start(self) -> None:
        if self.proc and self.proc.returncode is None:
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(self.extra_env)
        args = [
            self.pi_bin, "--mode", "rpc",
            "--session-id", self.session_id,
            "--session-dir", str(self.session_dir),
            "--name", self.name,
        ]
        if not self.extensions:
            # Extensions can open interactive dialogs that block a headless agent, or add tools such as model switching.
            args.append("--no-extensions")
        # Explicit harness skills are additive: discovered global/project skills remain enabled.
        for skill in self.skill_paths:
            args += ["--skill", str(skill)]
        if self.append_system_prompt:
            args += ["--append-system-prompt", str(self.append_system_prompt)]
        if self.sandbox_writable is not None:
            args = bwrap_prefix([self.session_dir, *self.sandbox_writable]) + args
        self.proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(self.cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=RPC_LINE_LIMIT,
        )
        self._reader_task = asyncio.create_task(self._read_stdout(), name=f"pi-rpc-reader:{self.name}")
        self._stderr_task = asyncio.create_task(self._read_stderr(), name=f"pi-rpc-stderr:{self.name}")
        # Fail early if the process exits immediately.
        await asyncio.sleep(0.05)
        if self.proc.returncode is not None:
            raise PiRPCError(f"Pi RPC process exited during startup: {self.proc.returncode}")

    async def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                if line.endswith(b"\r\n"):
                    line = line[:-2] + b"\n"
                try:
                    rec = json.loads(line.decode("utf-8").rstrip("\n"))
                except Exception as exc:
                    if self.on_event:
                        self.on_event({"type": "rpc_parse_error", "error": str(exc), "raw": line.decode("utf-8", "replace")[:1000]})
                    continue
                if rec.get("type") not in {"message_update", "tool_execution_update", "extension_ui_request"}:
                    # Deltas are redundant: message_end/tool_execution_end carry the authoritative content.
                    self._record("in", rec)
                if rec.get("type") == "tool_execution_start":
                    self.tool_calls += 1
                    if self.max_tool_calls is not None and self.tool_calls > self.max_tool_calls and not self.abort_reason:
                        # Never await a command from the reader: its response is read by this same task.
                        # Pi answers abort with agent_settled, which wakes prompt_and_wait.
                        self._abort_task = asyncio.create_task(self.abort(f"tool call limit exceeded: {self.max_tool_calls}"))
                rid = rec.get("id")
                if rec.get("type") == "response" and rid and rid in self._pending:
                    fut = self._pending.pop(rid)
                    if not fut.done():
                        fut.set_result(rec)
                else:
                    if rec.get("type") == "agent_settled":
                        self._settled.set()
                    if self.on_event:
                        self.on_event(rec)
        except Exception as exc:
            # Surface reader death in the transcript and events instead of an unretrieved task exception.
            rec = {"type": "rpc_reader_error", "error": f"{type(exc).__name__}: {exc}"}
            self._record("harness", rec)
            if self.on_event:
                self.on_event(rec)
        finally:
            err = PiRPCError(f"Pi RPC stdout closed for {self.name}")
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(err)
            self._pending.clear()
            self._settled.set()

    async def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                return
            rec = {"type": "rpc_stderr", "text": line.decode("utf-8", "replace").rstrip()}
            self._record("stderr", rec)
            if self.on_event:
                self.on_event(rec)

    async def command(self, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
        await self.start()
        assert self.proc and self.proc.stdin
        rid = str(payload.get("id") or uuid.uuid4())
        payload = dict(payload)
        payload["id"] = rid
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[rid] = fut
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        self._record("out", payload)
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()
        try:
            response = await asyncio.wait_for(fut, timeout=timeout)
        except Exception:
            self._pending.pop(rid, None)
            raise
        if not response.get("success", False):
            raise PiRPCError(str(response.get("error") or response))
        return response

    async def apply_route(self, route: ModelRoute) -> dict[str, Any] | None:
        model_data = None
        if route.spec:
            if "/" not in route.spec:
                raise PiRPCError(f"Model spec must be provider/model: {route.spec}")
            provider, model_id = route.spec.split("/", 1)
            resp = await self.command({"type": "set_model", "provider": provider, "modelId": model_id})
            model_data = resp.get("data")
        if route.thinking:
            try:
                await self.command({"type": "set_thinking_level", "level": route.thinking})
            except PiRPCError:
                # A non-reasoning model may only expose off. Query and clamp.
                levels = (await self.command({"type": "get_available_thinking_levels"})).get("data", {}).get("levels", [])
                if levels:
                    await self.command({"type": "set_thinking_level", "level": levels[-1]})
        return model_data

    async def prompt_and_wait(self, message: str, timeout: float = 1800.0, max_tool_calls: int | None = None) -> str:
        self._settled.clear()
        self.max_tool_calls = max_tool_calls  # counted against self.tool_calls, which the caller resets per task
        self.abort_reason = None
        resp = await self.command({"type": "prompt", "message": message}, timeout=60.0)
        disposition = (resp.get("data") or {}).get("disposition")
        if disposition == "handled":
            self._settled.set()  # no run was started
        else:
            try:
                await asyncio.wait_for(self._settled.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                await self.abort(f"agent timeout after {timeout:.0f}s")
        if self.abort_reason:
            raise AgentAborted(self.abort_reason)
        last = await self.command({"type": "get_last_assistant_text"}, timeout=30.0)
        return str((last.get("data") or {}).get("text") or "")

    async def steer(self, message: str) -> None:
        await self.command({"type": "prompt", "message": message, "streamingBehavior": "steer"})

    async def abort(self, reason: str = "aborted by harness") -> None:
        """Abort the active run; the pending prompt_and_wait raises AgentAborted(reason)."""
        if self._settled.is_set():
            return  # idle session: nothing to abort, and no stale reason may leak into the next run
        if self.abort_reason is None:
            self.abort_reason = reason
            self._record("harness", {"type": "abort", "reason": reason})
        with contextlib.suppress(Exception):
            # Pi responds only after the session is idle, so the run is over either way.
            await self.command({"type": "abort"}, timeout=10.0)
        self._settled.set()

    async def stats(self) -> dict[str, Any]:
        return (await self.command({"type": "get_session_stats"}, timeout=30.0)).get("data") or {}

    async def available_models(self) -> list[dict[str, Any]]:
        return ((await self.command({"type": "get_available_models"}, timeout=30.0)).get("data") or {}).get("models", [])

    async def compact(self, instructions: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "compact"}
        if instructions:
            payload["customInstructions"] = instructions
        return (await self.command(payload, timeout=600.0)).get("data") or {}

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
                with contextlib.suppress(Exception):
                    await self.proc.stdin.wait_closed()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.proc.kill()
            for task in [self._reader_task, self._stderr_task]:
                if task and not task.done():
                    task.cancel()
