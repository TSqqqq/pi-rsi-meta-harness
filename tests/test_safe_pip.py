import tempfile
import unittest
from pathlib import Path
from rsi_harness.config import Config
from rsi_harness.db import ResearchDB
from rsi_harness.safe_pip import base_name, is_approved, request

class TestSafePip(unittest.TestCase):
    def test_specs_and_approval(self):
        self.assertEqual(base_name("ray[tune]"),"ray")
        with self.assertRaises(SystemExit): base_name("git+https://example.com/x")
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); repo=td/"r"; repo.mkdir(); c=td/"research.toml"
            c.write_text(f'''[project]\nrepo="{repo}"\nstate_dir="{td}/state"\n[objective]\nmetric="x"\n[commands]\nevaluate="echo"\n[dependencies]\nallow_dynamic_pip=true\nallowed_packages=["optuna"]\n''')
            cfg=Config.load(c); db=ResearchDB(cfg.db_path)
            self.assertTrue(is_approved(cfg,db,"optuna"))
            self.assertFalse(is_approved(cfg,db,"somepkg"))
            request(cfg,"somepkg","test")
            self.assertEqual(db.one("SELECT status FROM dependency_requests WHERE package='somepkg'")["status"],"requested")

if __name__=='__main__': unittest.main()
