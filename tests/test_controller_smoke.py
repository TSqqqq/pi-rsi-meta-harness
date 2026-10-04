import asyncio
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

from rsi_harness.config import Config
from rsi_harness.controller import ResearchController

ROOT = Path(__file__).resolve().parent.parent
FAKE_PI = ROOT / "tests" / "fake_pi.py"


class TestControllerSmoke(unittest.TestCase):
    def test_end_to_end_toy_campaign(self):
        with tempfile.TemporaryDirectory() as td_s:
            td = Path(td_s)
            repo = td / "repo"
            repo.mkdir()
            (repo / "train.py").write_text("print('train ok')\n")
            (repo / "evaluate.py").write_text(textwrap.dedent('''
                import argparse, json
                p=argparse.ArgumentParser(); p.add_argument('--id'); p.add_argument('--stage'); p.add_argument('--seed')
                a=p.parse_args()
                value = 1.0 if a.id == 'baseline' else 1.1
                print(json.dumps({'metric': value, 'stage': a.stage, 'seed': int(a.seed)}))
            '''))
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@local", "commit", "-qm", "baseline"], cwd=repo, check=True)

            cfgp = td / "research.toml"
            cfgp.write_text(textwrap.dedent(f'''
                [pi]
                binary = "{FAKE_PI}"
                [project]
                name = "toy"
                repo = "{repo}"
                state_dir = "{td / 'state'}"
                worktree_dir = "{td / 'worktrees'}"
                artifact_dir = "{td / 'artifacts'}"
                session_dir = "{td / 'sessions'}"
                log_dir = "{td / 'logs'}"
                [objective]
                metric = "score"
                direction = "maximize"
                target = 1.05
                min_meaningful_delta = 0.01
                metric_json_key = "metric"
                [commands]
                baseline = "python train.py"
                quick = "python train.py"
                medium = "python train.py"
                full = "python train.py"
                evaluate = "python evaluate.py --id {{experiment_id}} --stage {{stage}} --seed {{seed}}"
                preflight = "python -m compileall -q ."
                [stages]
                quick_min_delta = -1.0
                medium_min_delta = -1.0
                validation_seeds = [1,2]
                quick_timeout_min = 1
                medium_timeout_min = 1
                full_timeout_min = 1
                [resources]
                gpus = []
                max_parallel = 1
                gpus_per_trial = 1
                [search]
                batch_size = 1
                scouts = 1
                meta_review_every = 100
                frontier_size = 3
                [models]
                mode = "single"
                default = "fake/planner"
                planner_thinking = "high"
                worker_thinking = "minimal"
                allow_escalation = true
                [sessions]
                compact_at_percent = 95
                [budget]
                max_gpu_hours = 10
                max_wall_hours = 1
                max_agent_tokens = 1000000
                max_experiments = 10
                max_consecutive_crashes = 3
                [stop]
                plateau_experiments = 5
                pause_on_plateau = true
                require_full_stage_for_target = true
                [files]
                mutable = ["train.py"]
                immutable = ["evaluate.py"]
                [dependencies]
                allow_dynamic_pip = false
                allowed_packages = []
                require_human_for_unlisted = true
                venv = ".venv"
                [dashboard]
                host = "127.0.0.1"
                port = 0
                refresh_seconds = 1
            '''))
            cfg = Config.load(cfgp)

            async def go():
                ctl = ResearchController(cfg, ROOT)
                await ctl.initialize()
                ctl.db.close()
                ctl = ResearchController(cfg, ROOT)
                await asyncio.wait_for(ctl.run(), timeout=20)
                return ctl

            ctl = asyncio.run(go())
            best = ctl.db.best(cfg.direction)
            self.assertIsNotNone(best)
            self.assertEqual(best["id"], "exp_000001")
            self.assertAlmostEqual(float(best["metric"]), 1.1)
            self.assertEqual(ctl.db.get_meta("state"), "TARGET_REACHED")
            self.assertTrue(bool(best["validated"]))
            self.assertTrue(ctl.db.query("SELECT * FROM beliefs"))
            self.assertTrue(ctl.db.query("SELECT * FROM parameter_effects"))
            ctl.db.close()


if __name__ == "__main__":
    unittest.main()
