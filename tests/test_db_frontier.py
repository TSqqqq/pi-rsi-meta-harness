import tempfile
import unittest
from pathlib import Path
from rsi_harness.config import Config
from rsi_harness.db import ResearchDB
from rsi_harness.frontier import FrontierSelector

class TestDBFrontier(unittest.TestCase):
    def test_dag_and_frontier(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); repo=td/"repo"; repo.mkdir()
            cfgp=td/"research.toml"
            cfgp.write_text(f'''[project]\nrepo="{repo}"\n[objective]\nmetric="acc"\ndirection="maximize"\n[commands]\nevaluate="echo"\n[search]\nfrontier_size=3\n''')
            cfg=Config.load(cfgp); db=ResearchDB(td/"x.db")
            db.add_experiment({"id":"baseline","status":"baseline","metric":1.0},[]); db.update_experiment("baseline",metric=1.0)
            a=db.add_experiment({"title":"a","status":"promising","parent_primary":"baseline","novelty":0.7,"confidence":0.7},["baseline"]); db.update_experiment(a,metric=1.2)
            b=db.add_experiment({"title":"b","status":"near_miss","parent_primary":"baseline","novelty":1.0,"confidence":0.4},["baseline"]); db.update_experiment(b,metric=1.1,pinned=1)
            self.assertEqual(db.parents(a),["baseline"])
            ids=[x["id"] for x in FrontierSelector(cfg,db).select(3)]
            self.assertIn(b,ids)

if __name__=='__main__': unittest.main()
