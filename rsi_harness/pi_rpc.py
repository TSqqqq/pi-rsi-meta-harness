from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from .models import ModelRoute


class PiRPCError(RuntimeError):
    pass


EventCallback = Callable[[dict[str, Any]], None]


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
        self.proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._settled = asyncio.Event()
        self._settled.set()
        self._closed = False

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
        # Explicit harness skills are additive: discovered global/project skills remain enabled.
        for skill in self.skill_paths:
            args += ["--skill", str(skill)]
        if self.append_system_prompt:
            args += ["--append-system-prompt", str(self.append_system_prompt)]
        self.proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(self.cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
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
            if self.on_event:
                self.on_event({"type": "rpc_stderr", "text": line.decode("utf-8", "replace").rstrip()})

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

    async def prompt_and_wait(self, message: str, timeout: float = 1800.0) -> str:
        self._settled.clear()
        resp = await self.command({"type": "prompt", "message": message}, timeout=60.0)
        disposition = (resp.get("data") or {}).get("disposition")
        if disposition != "handled":
            await asyncio.wait_for(self._settled.wait(), timeout=timeout)
        last = await self.command({"type": "get_last_assistant_text"}, timeout=30.0)
        return str((last.get("data") or {}).get("text") or "")

    async def steer(self, message: str) -> None:
        await self.command({"type": "prompt", "message": message, "streamingBehavior": "steer"})

    async def abort(self) -> None:
        try:
            await self.command({"type": "abort"}, timeout=10.0)
        except Exception:
            pass

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
                try:
                    await self.proc.stdin.wait_closed()
                except Exception:
                    pass
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
