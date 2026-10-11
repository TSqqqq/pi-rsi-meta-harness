"""The fixed-driver template must not let a method see or exploit the benchmark (no GPU, no model needed).

A fake method tries every known leak: reading labels, earlier predictions and scores, writing outside its
process, and relating grouped items through shared state. Every attempt must fail.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "examples" / "fixed_driver.py"

FAKE_METHOD = '''
import glob, json, os
SEEN = []
def load(args):
    secret = os.environ["LEAK_PROBE_ROOT"]
    probes = {
        "labels": os.path.exists(os.path.join(secret, "data", "test.json")),
        "any_json_in_data": bool(glob.glob(os.path.join(secret, "data", "**", "*.json"), recursive=True)),
        "old_predictions": os.path.exists(os.path.join(secret, "outputs", "old.json")),
        "load_args": sorted(vars(args)),
    }
    for target in (os.path.join(secret, "leak.txt"), os.path.join(os.path.dirname(__file__), "leak.txt")):
        try:
            open(target, "w").write("x")
            probes.setdefault("wrote", []).append(target)
        except OSError:
            pass
    print("PROBES " + json.dumps(probes), flush=True)
    def answer(item):
        # Cheat: the second item of a group is answered opposite to the first.
        seen_before = any(p == item["input_path"] for p in SEEN)
        SEEN.append(item["input_path"])
        return "B" if seen_before else "A"
    return answer
'''


class TestFixedDriver(unittest.TestCase):
    def setUp(self):
        if not shutil.which("bwrap"):
            self.skipTest("bwrap not installed")
        self.td = Path(tempfile.mkdtemp(dir=os.path.expanduser("~"), prefix=".rsi-driver-test-"))
        data = self.td / "data"
        (data / "inputs").mkdir(parents=True)
        items = []
        for i in range(20):
            (data / "inputs" / f"x{i}.bin").write_bytes(b"0")
            items.append({"input": f"x{i}.bin", "prompt": f"q{i}", "label": "A"})
        (data / "test.json").write_text(json.dumps(items))
        (self.td / "outputs").mkdir()
        (self.td / "outputs" / "old.json").write_text("[]")
        (self.td / "code").mkdir()
        (self.td / "code" / "method.py").write_text(textwrap.dedent(FAKE_METHOD))
        (self.td / "model").mkdir()
        shutil.copy(DRIVER, self.td / "fixed_driver.py")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_method_cannot_see_labels_or_write(self):
        r = subprocess.run([sys.executable, str(self.td / "fixed_driver.py"), "--code_dir", str(self.td / "code"),
                            "--model_path", str(self.td / "model"), "--data_dir", str(self.td / "data"),
                            "--inputs_dir", str(self.td / "data" / "inputs"), "--output", str(self.td / "pred.json")],
                           env={**os.environ, "LEAK_PROBE_ROOT": str(self.td)}, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        probes = [json.loads(line[7:]) for line in r.stdout.splitlines() if line.startswith("PROBES ")]
        self.assertTrue(probes)
        for p in probes:
            self.assertFalse(p["labels"] or p["any_json_in_data"] or p["old_predictions"], p)
            self.assertNotIn("wrote", p)
            self.assertEqual(p["load_args"], ["device", "model_path"])
        groups = json.loads((self.td / "pred.json").read_text())
        self.assertEqual(sum(len(g) for g in groups), 20)

    def test_grouped_items_never_share_a_shard(self):
        spec = importlib.util.spec_from_file_location("fixed_driver", DRIVER)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        groups = [[{}, {}] for _ in range(30)] + [[{}, {}, {}] for _ in range(5)]
        shards = mod.shards(groups)
        self.assertEqual(len(shards), 3)
        for shard in shards:
            members = [g for g, _ in shard]
            self.assertEqual(len(members), len(set(members)))  # at most one member of a group per shard
        self.assertEqual(sum(len(s) for s in shards), 30 * 2 + 5 * 3)


if __name__ == "__main__":
    unittest.main()
