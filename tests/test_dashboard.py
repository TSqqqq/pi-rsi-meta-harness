import json
import tempfile
import unittest
import urllib.request
from pathlib import Path

from rsi_harness.config import Config
from rsi_harness.dashboard import DashboardServer
from rsi_harness.db import ResearchDB

ROOT = Path(__file__).resolve().parent.parent


class TestDashboard(unittest.TestCase):
    def test_console_state_and_controls(self):
        with tempfile.TemporaryDirectory() as td_s:
            td = Path(td_s)
            repo = td / "repo"
            repo.mkdir()
            cfgp = td / "research.toml"
            cfgp.write_text(
                f'''[project]\nrepo="{repo}"\nstate_dir="{td / 'state'}"\nartifact_dir="{td / 'artifacts'}"\nlog_dir="{td / 'logs'}"\n[objective]\nmetric="acc"\ndirection="maximize"\ntarget=2.0\n[commands]\nevaluate="echo"\n[dashboard]\nhost="127.0.0.1"\nport=0\nallow_control=true\n'''
            )
            cfg = Config.load(cfgp)
            db = ResearchDB(cfg.db_path)
            db.add_experiment({"id": "baseline", "status": "baseline", "stage": "full", "title": "Baseline"}, [])
            db.update_experiment("baseline", metric=1.0, validated=1, delta_baseline=0.0, delta_parent=0.0)
            eid = db.add_experiment({"title": "Try A", "status": "near_miss", "stage": "quick", "parent_primary": "baseline"}, ["baseline"])
            db.update_experiment(eid, metric=1.1, delta_parent=0.1, delta_baseline=0.1)
            srv = DashboardServer(cfg, db, ROOT / "dashboard" / "index.html")
            srv.start()
            try:
                host, port = srv.address
                base = f"http://{host}:{port}"
                with urllib.request.urlopen(base + "/api/state") as r:
                    state = json.load(r)
                self.assertEqual(state["experiment_count"], 1)
                self.assertIn("idea_queue", state)

                req = urllib.request.Request(
                    base + "/api/ideas",
                    data=json.dumps({"action": "add", "text": "Explore alternative loss", "experiment_id": eid}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req) as r:
                    out = json.load(r)
                self.assertTrue(out["ok"])
                idea_id = out["id"]

                req = urllib.request.Request(
                    base + "/api/ideas",
                    data=json.dumps({"action": "send", "id": idea_id}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req) as r:
                    self.assertTrue(json.load(r)["ok"])
                self.assertEqual(db.recent_guidance(1)[0]["kind"], "idea")

                req = urllib.request.Request(
                    base + "/api/node",
                    data=json.dumps({"action": "branch", "experiment_id": eid, "insight": "Revisit with better schedule"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req) as r:
                    self.assertTrue(json.load(r)["ok"])
                self.assertEqual(db.experiment(eid)["pinned"], 1)

                req = urllib.request.Request(
                    base + "/api/control",
                    data=json.dumps({"action": "pause"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req) as r:
                    self.assertTrue(json.load(r)["ok"])
                self.assertEqual(db.get_meta("control")["action"], "pause")
            finally:
                srv.stop()
                db.close()


if __name__ == "__main__":
    unittest.main()
