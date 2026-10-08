import json
import tempfile
import unittest
import urllib.request
from pathlib import Path

from rsi_harness.config import Config
from rsi_harness.dashboard import DashboardServer
from rsi_harness.db import ResearchDB

ROOT = Path(__file__).resolve().parent.parent
# Local test server: never route through the host's HTTP proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(base: str, path: str, payload: dict | None = None) -> dict:
    """GET (no payload) or JSON POST against the local test dashboard only."""
    url = base + path
    if not url.startswith("http://127.0.0.1:"):
        raise ValueError(f"refusing non-local URL: {url}")
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(  # noqa: S310 - scheme and host checked above
        url, data=data, headers={"Content-Type": "application/json"}, method="GET" if data is None else "POST"
    )
    with _OPENER.open(req) as r:
        return json.load(r)


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
                address = srv.address
                assert address is not None
                base = f"http://{address[0]}:{address[1]}"

                state = call(base, "/api/state")
                self.assertEqual(state["experiment_count"], 1)
                self.assertIn("idea_queue", state)

                out = call(base, "/api/ideas", {"action": "add", "text": "Explore alternative loss", "experiment_id": eid})
                self.assertTrue(out["ok"])
                self.assertTrue(call(base, "/api/ideas", {"action": "send", "id": out["id"]})["ok"])
                self.assertEqual(db.recent_guidance(1)[0]["kind"], "idea")

                # The page sends experiment_id=null when no node is attached; that must not be read as "None".
                self.assertTrue(call(base, "/api/ideas", {"action": "add", "text": "Unattached idea", "experiment_id": None})["ok"])

                self.assertTrue(call(base, "/api/node", {"action": "branch", "experiment_id": eid, "insight": "Revisit with better schedule"})["ok"])
                exp = db.experiment(eid)
                assert exp is not None
                self.assertEqual(exp["pinned"], 1)

                self.assertTrue(call(base, "/api/control", {"action": "pause"})["ok"])
                self.assertEqual(db.get_meta("control")["action"], "pause")
            finally:
                srv.stop()
                db.close()


if __name__ == "__main__":
    unittest.main()
