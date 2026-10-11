"""Unit tests for the mechanical rules: noise, briefing, candidate validation, knobs, routing."""
import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

from rsi_harness.config import Config
from rsi_harness.context import ContextBuilder, outcome, section
from rsi_harness.controller import ResearchController
from rsi_harness.gitops import numeric_only_diff

ROOT = Path(__file__).resolve().parent.parent


def make(td: Path) -> ResearchController:
    repo = td / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (td / "research.toml").write_text(textwrap.dedent(f"""
        [pi]
        binary = "{ROOT / 'tests' / 'fake_pi.py'}"
        [project]
        repo = "{repo}"
        state_dir = "{td / 'state'}"
        [objective]
        metric = "acc"
        direction = "maximize"
        target = 1.0
        min_meaningful_delta = 0.005
        noise_sigmas = 2.0
        [commands]
        evaluate = "echo"
        quick = "echo"
        [experiment]
        alpha = 0.5
        layer = 18
        [resources]
        max_parallel = 4
        [search]
        batch_size = 4
        min_structural_ratio = 0.5
        max_tuning_per_batch = 1
        explore_after_flat_batches = 2
    """))
    ctl = ResearchController(Config.load(td / "research.toml"), ROOT)
    db = ctl.db
    db.add_experiment({"id": "baseline", "status": "baseline", "stage": "quick", "title": "base"}, [])
    for v in (0.50, 0.52, 0.51, 0.53):
        db.add_metric("baseline", "quick", None, v)
    db.update_experiment("baseline", metric=0.515)
    return ctl


def exp(ctl, eid, kind, title, delta=None, status="completed", params=None, batch=1, reason=None):
    ctl.db.add_experiment({"id": eid, "status": status, "stage": "quick", "kind": kind, "title": title,
                           "mechanism": title, "params": params or {}, "metadata": {"batch": batch}}, ["baseline"])
    if delta is not None:
        ctl.db.add_metric(eid, "quick", 1, 0.515 + delta)
        ctl.db.update_experiment(eid, metric=0.515 + delta, delta_baseline=delta, delta_parent=delta)
    if reason:
        ctl.db.update_experiment(eid, failure_reason=reason)


class TestResearchRules(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.ctl = make(Path(self.td.name))

    def tearDown(self):
        self.ctl.db.close()
        self.td.cleanup()

    def test_noise_threshold_from_baseline_replicates(self):
        sigma = 0.0129  # std of 0.50,0.52,0.51,0.53
        one = self.ctl.runner.noise("quick", 1)
        self.assertAlmostEqual(one, 2 * sigma * (1 + 1 / 4) ** 0.5, places=3)
        self.assertLess(self.ctl.runner.noise("quick", 3), one)  # replication tightens the threshold

    def test_briefing_is_budgeted_and_complete(self):
        for i in range(80):
            exp(self.ctl, f"exp_{i:06d}", "mechanism", f"mechanism number {i} " + "x" * 200, delta=0.001 * (i % 7),
                status="near_miss")
        text = self.ctl.context.briefing()
        self.assertLess(len(text), 70000)
        self.assertNotIn("truncated by harness", text)
        self.assertIn("OBJECTIVE AND NOISE", text)
        tree = text.split("## EXPERIMENT TREE")[1].split("\n## ")[0]
        self.assertEqual(sum(f"exp_{i:06d}" in tree for i in range(80)), 80)  # 80 nodes all fit, none dropped
        self.ctl.cfg.raw["context"] = {"tree_chars": 8000}  # recent anchors (~400 chars each) must still fit
        tree = self.ctl.context.briefing().split("## EXPERIMENT TREE")[1].split("\n## ")[0]
        self.assertIn("folded", tree)  # over budget: summarized, not silently cut
        self.assertIn("exp_000079", tree)

    def test_not_run_is_separated_from_evidence(self):
        exp(self.ctl, "exp_000001", "mechanism", "a", status="failed", reason="AgentAborted: agent timeout after 600s")
        self.assertEqual(outcome(self.ctl.db.experiment("exp_000001")), "NOT_RUN:worker_timeout")

    def test_normalize_is_mechanical_and_keeps_the_scouts_choices(self):
        exp(self.ctl, "exp_000001", "mechanism", "token exclusion", delta=0.02, status="promising")
        exp(self.ctl, "exp_000002", "tuning", "lower alpha", delta=0.0, params={"alpha": 0.3})
        exp(self.ctl, "exp_000003", "mechanism", "never ran", status="failed", reason="AgentAborted: agent timeout")
        self.ctl.db.update_experiment("exp_000002", parent_primary="baseline")
        self.ctl.db.update_experiment("exp_000003", parent_primary="exp_000001")
        # Similar wording is the scout's call, not a rejection.
        same_words, _ = self.ctl._normalize({"kind": "revival", "title": "token exclusion", "mechanism": "token exclusion"})
        self.assertIsNotNone(same_words)
        bad, why = self.ctl._normalize({"kind": "tuning", "title": "x", "params": {"gamma": 1}})
        self.assertIsNone(bad)
        sweep, why = self.ctl._normalize({"kind": "tuning", "title": "x", "params": {"alpha": [0.2, 0.3]}})
        self.assertIn("ONE numeric value", why)
        rerun, why = self.ctl._normalize({"kind": "tuning", "title": "again", "params": {"alpha": 0.3}})
        self.assertIn("exact re-run of exp_000002", why)
        noop, why = self.ctl._normalize({"kind": "tuning", "title": "noop", "params": {"alpha": 0.5}})
        self.assertIn("already uses", why)
        ok, _ = self.ctl._normalize({"kind": "tuning", "title": "layer shift", "params": {"layer": 16.0}})
        self.assertEqual(ok["params"], {"layer": 16})  # int knob stays int for the command line
        combo, _ = self.ctl._normalize({"kind": "combination", "title": "merge", "parent_ids": ["exp_000002", "exp_000001"]})
        self.assertEqual(combo["parent_ids"], ["exp_000002", "exp_000001"])  # the scout's primary parent is kept
        revive, _ = self.ctl._normalize({"kind": "revival", "title": "retry", "parent_ids": ["exp_000003"], "params": {"alpha": 1}})
        self.assertEqual(revive["parent_ids"], ["exp_000001"])  # never-measured parent -> nearest measured ancestor
        self.assertNotIn("params", revive)  # knobs change only in tuning experiments

    def test_worker_gets_the_full_lineage_diff(self):
        import subprocess
        wt = Path(self.td.name) / "lin"
        wt.mkdir()
        git = lambda *a: subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=a", *a], cwd=wt, check=True, capture_output=True)
        git("init", "-q"); (wt / "m.py").write_text("a = 1\n"); git("add", "-A"); git("commit", "-qm", "base")
        base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=wt, capture_output=True, text=True).stdout.strip()
        self.ctl.db.set_meta("baseline_commit", base)
        self.assertIn("none (you start from the unmodified baseline code)", self.ctl._lineage_diff(wt))
        for i in (2, 3):  # two ancestor experiments
            (wt / "m.py").write_text((wt / "m.py").read_text() + f"step{i} = {i}\n"); git("commit", "-qam", f"exp{i}")
        diff = self.ctl._lineage_diff(wt)
        self.assertIn("+step2 = 2", diff)
        self.assertIn("+step3 = 3", diff)  # every ancestor's change, not just the last commit

    def test_fix_round_gets_the_failing_functions(self):
        wt = Path(self.td.name) / "rep"
        wt.mkdir()
        (wt / "m.py").write_text("def outer():\n    return inner()\n\ndef inner():\n    x = 1\n    return x.missing\n")
        err = (f'Traceback (most recent call last):\n  File "{wt}/m.py", line 2, in outer\n'
               f'  File "{wt}/m.py", line 6, in inner\n  File "/usr/lib/python3/x.py", line 9, in y\n'
               "AttributeError: 'int' object has no attribute 'missing'")
        src = self.ctl._traceback_source(wt, err)
        self.assertLess(src.index("m.py:inner"), src.index("m.py:outer"))  # innermost frame first
        self.assertIn("    6>     return x.missing", src)
        self.assertNotIn("x.py", src)  # library frames are not experiment code

    def test_worktree_gets_current_harness_files_but_keeps_experiment_code(self):
        import subprocess
        from rsi_harness.gitops import sync_harness_files
        wt = Path(self.td.name) / "sync"
        wt.mkdir()
        git = lambda *a: subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=a", *a], cwd=wt, check=True, capture_output=True, text=True)
        (wt / "train.py").write_text("v = 1\n"); (wt / "evaluate.py").write_text("old evaluator\n")
        git("init", "-q"); git("add", "-A"); git("commit", "-qm", "parent")
        parent = git("rev-parse", "HEAD").stdout.strip()
        (wt / "evaluate.py").write_text("fixed evaluator\n"); (wt / "driver.py").write_text("new driver\n")
        git("add", "-A"); git("commit", "-qm", "harness fix")
        base = git("rev-parse", "HEAD").stdout.strip()
        git("checkout", "-q", parent); (wt / "train.py").write_text("v = 2  # experiment code\n"); git("commit", "-qam", "exp")
        self.ctl.cfg.raw["files"] = {"mutable": ["*.py"], "immutable": ["evaluate.py"]}  # immutable wins over mutable
        synced = sync_harness_files(self.ctl.cfg, wt, base)
        self.assertEqual(sorted(synced), ["evaluate.py"])  # driver.py is mutable here: experiment code
        self.assertEqual((wt / "evaluate.py").read_text(), "fixed evaluator\n")
        self.assertEqual((wt / "train.py").read_text(), "v = 2  # experiment code\n")
        self.assertEqual(git("status", "--porcelain").stdout, "")  # committed: the experiment's diff stays clean
        self.assertEqual(sync_harness_files(self.ctl.cfg, wt, base), [])

    def test_worker_gets_the_target_function(self):
        wt = Path(self.td.name) / "wt"
        (wt / "pkg").mkdir(parents=True)
        (wt / "pkg" / "m.py").write_text("x = 1\n\nclass A:\n    def f(self):\n        return 2\n")
        src = self.ctl._target_source(wt, {"changes_hint": {"file": "pkg/m.py", "function": "A.f"}})
        self.assertIn("TARGET pkg/m.py:f (lines 4-5", src)
        self.assertIn("    5          return 2", src)
        many = self.ctl._target_source(wt, {"changes_hint": [{"file": "pkg/m.py", "function": "A"}, {"file": "pkg/m.py", "function": "f"}]})
        self.assertIn("TARGET pkg/m.py:A", many)  # every place a multi-site change touches
        self.assertIn("TARGET pkg/m.py:f", many)
        for hint in ({"file": "pkg/m.py", "function": "missing"}, {"file": "../x.py", "function": "f"}, ):
            self.assertEqual(self.ctl._target_source(wt, {"changes_hint": hint}), "")

    def test_knobs_reach_the_command_line(self):
        exp(self.ctl, "exp_000001", "tuning", "t", params={"alpha": 0.3})
        vals = self.ctl.runner._values("exp_000001", Path("."), "quick", 1, [])
        self.assertEqual((vals["alpha"], vals["layer"]), (0.3, 18))

    def test_briefing_is_plans_results_and_rejections_only(self):
        exp(self.ctl, "exp_000001", "mechanism", "measured idea", delta=0.02)
        exp(self.ctl, "exp_000002", "mechanism", "broken idea", status="failed",
            reason="RuntimeError: smoke run failed: ValueError: only one element tensors can be converted")
        self.ctl.db.update_experiment("exp_000001", mechanism="mask the last 2 tokens of each block",
                                      metadata={"candidate": {"changes_hint": {"file": "src/model.py", "function": "weight_attention"}}})
        self.ctl.events.emit("candidate_rejected", "sweep: bad", source="SCOUT-1",
                             payload={"kind": "tuning", "title": "alpha sweep", "parent_ids": ["exp_000001"],
                                      "params": {"alpha": [0.2, 0.3]}, "mechanism": "scan", "reason": "ONE numeric value per knob"})
        text = self.ctl.context.briefing(idea=True)
        self.assertIn("plan: mask the last 2 tokens of each block @model.py:weight_attention", text)
        self.assertRegex(text, r"exp_000001 .*result: quick n1 mean=0\.\d{4} Δb\+0\.0200")
        self.assertIn("result: NOT RUN (runtime_error)", text)
        self.assertIn("only one element tensors", text)  # the real reason, not just the category
        self.assertIn('alpha sweep parents=exp_000001 params={"alpha": [0.2, 0.3]}', text)
        self.assertIn("REJECTED: ONE numeric value per knob", text)
        self.assertIn("patch.diff", text)  # the scout can read any exact patch on demand
        for gone in ("INSIGHTS", "PATHS", "FRONTIER", "KIND MIX", "POLICY", "PARAMETER EFFECTS", "FAILED TO RUN"):
            self.assertNotIn(gone, text)  # no harness-written interpretation

    def test_tree_shows_lineage_and_folds_least_informative_leaves(self):
        exp(self.ctl, "exp_000001", "mechanism", "root idea", delta=0.03, status="promising")
        prev = "exp_000001"
        for i in range(2, 60):
            eid = f"exp_{i:06d}"
            exp(self.ctl, eid, "mechanism", f"variant {i} " + "y" * 120, delta=0.0 if i % 3 else None,
                status="near_miss" if i % 3 else "failed", reason=None if i % 3 else "AgentAborted: agent timeout")
            self.ctl.db.update_experiment(eid, parent_primary=prev if i < 6 else "exp_000001")
            prev = eid
        lines = self.ctl.context.tree(4000)
        text = "\n".join(lines)
        self.assertLess(len(text), 4200)
        self.assertTrue(any("folded" in x for x in lines))  # overflow is summarized, never silently dropped
        self.assertIn("exp_000001", text)  # the promising root is kept ...
        self.assertIn("exp_000059", text)  # ... and so are the most recent nodes
        full = self.ctl.context.tree(10 ** 6)
        depth = {x.strip().split()[0]: len(x) - len(x.lstrip()) for x in full if x.strip().startswith("exp_")}
        self.assertGreater(depth["exp_000005"], depth["exp_000002"])  # descent is visible as indentation
        self.assertEqual(len(depth), 59)  # with room, every node is shown

    def test_two_tiers(self):
        self.ctl.cfg.raw["models"] = {"mode": "dual", "planner": "p/strong", "worker": "w/code", "allow_escalation": True,
                                      "max_worker_failures_before_escalation": 2}
        from rsi_harness.models import ModelRouter
        r = ModelRouter(self.ctl.cfg)
        tier = lambda t, f=0: r.route(t, failures=f).spec
        for t in ("research_planning", "hypothesis_generation"):
            self.assertEqual(tier(t), "p/strong", t)
        self.assertEqual((tier("implementation"), tier("implementation", 2)), ("w/code", "p/strong"))
        self.ctl.cfg.raw["models"]["tiers"] = {"implementation": "planner"}
        self.assertEqual(ModelRouter(self.ctl.cfg).route("implementation").spec, "p/strong")

    def test_section_cuts_on_line_boundaries(self):
        text = section("T", [f"line {i}" for i in range(100)], 60)
        self.assertTrue(text.splitlines()[-1].startswith("… (+"))

    def test_numeric_only_diff(self):
        repo = Path(self.td.name) / "g"
        repo.mkdir()
        run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
        run("init", "-q")
        (repo / "a.py").write_text("alpha = 0.5\nx = f(y)\n")
        run("add", "-A")
        run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i")
        (repo / "a.py").write_text("alpha = 0.3\nx = f(y)\n")
        self.assertTrue(numeric_only_diff(repo))
        (repo / "a.py").write_text("alpha = 0.3\nx = g(y) * mask\n")
        self.assertFalse(numeric_only_diff(repo))


if __name__ == "__main__":
    unittest.main()
