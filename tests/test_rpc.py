import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from rsi_harness.models import ModelRoute
from rsi_harness.pi_rpc import AgentAborted, PiRPC

ROOT = Path(__file__).resolve().parent.parent
FAKE_PI = str(ROOT / "tests" / "fake_pi.py")


def client(td: Path) -> PiRPC:
    return PiRPC(cwd=ROOT, session_dir=td / "sessions", session_id="test", name="test",
                 pi_bin=FAKE_PI, transcript_dir=td / "transcripts")


class TestPiRPC(unittest.TestCase):
    def run_with_client(self, body):
        with tempfile.TemporaryDirectory() as td_s:
            async def go():
                c = client(Path(td_s))
                try:
                    await c.start()
                    return await body(c)
                finally:
                    await c.close()
            return asyncio.run(go())

    def test_prompt_model_and_transcript(self):
        async def body(c):
            self.assertEqual(len(await c.available_models()), 2)
            await c.apply_route(ModelRoute("fake/worker", "minimal", "worker"))
            c.run_id = 7
            self.assertIn('"ok"', await c.prompt_and_wait("hello"))
            lines = [json.loads(x) for x in c.transcript_path.read_text().splitlines()]
            mine = [x for x in lines if x["run_id"] == 7]
            # The exact prompt sent and the final assistant message are both on disk.
            self.assertTrue(any(x["direction"] == "out" and x["record"].get("message") == "hello" for x in mine))
            self.assertTrue(any(x["record"].get("type") == "message_end" for x in mine))
            self.assertFalse(any(x["record"].get("type") == "message_update" for x in lines))
        self.run_with_client(body)

    def test_tool_call_limit_aborts(self):
        async def body(c):
            with self.assertRaisesRegex(AgentAborted, "tool call limit"):
                await c.prompt_and_wait("LOOP_TOOLS", timeout=10, max_tool_calls=5)
            # The session stays usable and the stale abort reason does not leak.
            c.tool_calls = 0
            self.assertIn('"ok"', await c.prompt_and_wait("hello", timeout=10))
        self.run_with_client(body)

    def test_wrap_up_steers_before_the_hard_limit(self):
        async def body(c):
            c.tool_calls = 0
            with self.assertRaises(AgentAborted) as cm:
                await c.prompt_and_wait("LOOP_TOOLS", timeout=10, max_tool_calls=3, wrap_up="answer now")
            self.assertIn(f"tool call limit exceeded: {3 + c.WRAP_UP_GRACE}", str(cm.exception))
            steers = [json.loads(x)["record"] for x in c.transcript_path.read_text().splitlines()]
            self.assertTrue(any(r.get("streamingBehavior") == "steer" and r.get("message") == "answer now" for r in steers))
        self.run_with_client(body)

    def test_minimal_resources_and_explicit_extensions_on_the_command_line(self):
        captured = {}
        async def fake_exec(*args, **kw):
            captured["args"] = args
            raise RuntimeError("stop after argv")
        c = PiRPC(cwd=ROOT, session_dir=Path("/tmp/x"), session_id="t", name="t", pi_bin="pi", extensions=False,
                  extension_paths=[ROOT / "pi_extensions" / "rsi-context.ts"], minimal_resources=True)
        from unittest import mock
        with mock.patch("asyncio.create_subprocess_exec", fake_exec):
            with self.assertRaises(RuntimeError):
                asyncio.run(c.start())
        args = list(captured["args"])
        for flag in ("--no-extensions", "--no-skills", "--no-prompt-templates", "--no-context-files"):
            self.assertIn(flag, args)
        self.assertEqual(args[args.index("--extension") + 1], str(ROOT / "pi_extensions" / "rsi-context.ts"))

    def test_large_rpc_line(self):
        async def body(c):
            self.assertEqual(len(await c.prompt_and_wait("BIG_LINE", timeout=10)), 200_000)
        self.run_with_client(body)

    def test_timeout_aborts(self):
        async def body(c):
            with self.assertRaisesRegex(AgentAborted, "timeout"):
                await c.prompt_and_wait("HANG", timeout=0.5)
        self.run_with_client(body)

    def test_external_abort(self):
        async def body(c):
            task = asyncio.create_task(c.prompt_and_wait("HANG", timeout=30))
            await asyncio.sleep(0.3)
            await c.abort("human pause")
            with self.assertRaisesRegex(AgentAborted, "human pause"):
                await task
        self.run_with_client(body)


if __name__ == "__main__":
    unittest.main()
