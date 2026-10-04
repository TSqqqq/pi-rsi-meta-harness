import asyncio
import unittest
from pathlib import Path
from rsi_harness.pi_rpc import PiRPC
from rsi_harness.models import ModelRoute

ROOT=Path(__file__).resolve().parent.parent

class TestPiRPC(unittest.TestCase):
    def test_prompt_and_model(self):
        async def go():
            c=PiRPC(cwd=ROOT,session_dir=ROOT/".test-sessions",session_id="test",name="test",pi_bin=str(ROOT/"tests"/"fake_pi.py"))
            try:
                await c.start()
                models=await c.available_models()
                self.assertEqual(len(models),2)
                await c.apply_route(ModelRoute("fake/worker","minimal","worker"))
                text=await c.prompt_and_wait("hello")
                self.assertIn('"ok"',text)
            finally:
                await c.close()
        asyncio.run(go())

if __name__=='__main__': unittest.main()
