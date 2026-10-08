"""Throwaway dashboard + fake controller for the jsdom click test (temp state, no agents, no GPU)."""
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rsi_harness.config import Config
from rsi_harness.dashboard import DashboardServer
from rsi_harness.db import ResearchDB

td = Path(tempfile.mkdtemp(prefix="dashtest-"))
cfgp = td / "research.toml"
cfgp.write_text(
    f'[project]\nrepo="{td}"\nstate_dir="{td / "state"}"\n'
    '[objective]\nmetric="acc"\ndirection="maximize"\ntarget=2.0\n'
    '[commands]\nevaluate="echo"\n[dashboard]\nhost="127.0.0.1"\nport=0\nallow_control=true\n'
)
cfg = Config.load(cfgp)
db = ResearchDB(cfg.db_path)
db.add_experiment({"id": "baseline", "status": "baseline", "stage": "quick", "title": "base"}, parents=[])
db.update_experiment("baseline", metric=0.53)
db.add_experiment({"id": "exp_000001", "status": "near_miss", "stage": "quick", "title": "t1"}, parents=["baseline"])
db.update_experiment("exp_000001", metric=0.535)
db.execute("INSERT INTO agent_runs(ts_start,agent_name,task_type,session_id) VALUES(?,?,?,?)",
           (time.time(), "worker-exp_000002", "implementation", "rsi-exp_000002"))  # an active agent to steer
db.execute("INSERT INTO dependency_requests(ts,package,reason,status) VALUES(?,?,?,?)",
           (time.time(), "scipy", "test", "requested"))
db.set_meta("state", "PAUSED_FOR_HUMAN")

srv = DashboardServer(cfg, db, ROOT / "dashboard" / "index.html")
srv.start()


def fake_controller():
    # Mirrors ResearchController._control_watch: apply the requested state transition.
    seen = None
    target = {"pause": "PAUSED", "resume": "RUNNING", "stop": "STOPPED_BY_USER"}
    while True:
        ctl = db.get_meta("control", {}) or {}
        marker = (ctl.get("ts"), ctl.get("action"))
        if ctl.get("action") and marker != seen:
            seen = marker
            time.sleep(0.6)
            db.set_meta("state", target[ctl["action"]])
        time.sleep(0.2)


threading.Thread(target=fake_controller, daemon=True).start()
print(json.dumps({"port": srv.address[1], "db": str(cfg.db_path)}), flush=True)
threading.Event().wait()
