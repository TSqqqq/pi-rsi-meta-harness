"""Template: a fixed benchmark driver that a research method cannot exploit.

Put a copy in an immutable path (e.g. `eval/run_benchmark.py`) and run every experiment through it. The method
under research only provides `load(args) -> answer`, where `answer(item_input) -> prediction`; the driver owns
the benchmark:

- Isolation: the method runs in a bwrap sandbox whose home directory is empty except for read-only code, model
  weights and the raw *inputs* (e.g. images, videos, audio). Label files, earlier predictions, scores and the
  research database are invisible, and nothing outside the sandbox's private /tmp is writable.
- No grouping structure: when test items come in groups (paired questions, multi-turn sets, ...), the members of
  each group are dealt to different shards, one shard per process, so no state kept across calls can relate them.
- No order structure: items are shuffled within each shard with a fixed seed.
- Narrow interface: `load()` receives only model/knob settings (no data or output paths), `answer()` only one
  item's input.

Adapt `read_items()` / `item_input()` / `write_predictions()` to your benchmark format. Shards run one after
another so the template also works on GPUs in exclusive-process mode.

Why it matters: an automated research loop will find any shortcut the evaluation leaves open. In one campaign a
"verification gate" reached +25 points by rewriting every answer of the question type whose label is always the
same; the old driver told the method which type each question was. See docs/LESSONS.md.
"""
import argparse
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import types

ORDER_SEED = 1234  # fixed: identical shards and order for every experiment
METHOD_ARGS = ("model_path", "device")  # add your knobs here; never data or output paths
DRIVER_DIR = tempfile.mkdtemp(prefix="rsi-driver-")  # read-only copy of this file for the sandboxed children


# ---------- benchmark format: adapt these three ----------
def read_items(data_dir):
    """Groups of items: [[item, item, ...], ...]. Items of one group must never meet in one process."""
    with open(os.path.join(data_dir, "test.json")) as f:
        return [[x] for x in json.load(f)]  # ungrouped benchmark: every group has one item


def item_input(data_dir, item):
    """What the method sees for one item: inputs only, never the label."""
    return {"input_path": os.path.join(data_dir, "inputs", item["input"]), "prompt": item["prompt"]}


def write_predictions(groups, predictions, out_path):
    for g, group in enumerate(groups):
        for k, item in enumerate(group):
            item["prediction"] = predictions[f"{g}:{k}"]
    with open(out_path, "w") as f:
        json.dump(groups, f, indent=2)


# ---------- fixed part ----------
def load_method(code_dir):
    sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location("method", os.path.join(code_dir, "method.py"))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {code_dir}/method.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shards(groups):
    """As many shards as the largest group; member k of a group goes to a random, distinct shard."""
    rng = random.Random(ORDER_SEED)
    n = max(len(g) for g in groups)
    out = [[] for _ in range(n)]
    for g, group in enumerate(groups):
        for k, slot in zip(range(len(group)), rng.sample(range(n), len(group)), strict=True):
            out[slot].append((g, k))
    for s in out:
        rng.shuffle(s)
    return [s for s in out if s]


def sandbox_prefix(binds, writable):
    if not shutil.which("bwrap"):
        raise SystemExit("bwrap is required to isolate the method from the benchmark labels")
    home = os.path.expanduser("~")
    args = ["bwrap", "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc",
            "--tmpfs", tempfile.gettempdir(), "--tmpfs", home]
    for path in sorted(set(binds)):
        if os.path.exists(path):
            args += ["--ro-bind", path, path]
    args += ["--remount-ro", home, "--bind", writable, writable]
    return args + ["--die-with-parent", "--"]


def run_child(code_dir, method_args, out_path):
    answer = load_method(code_dir).load(types.SimpleNamespace(**method_args))
    jobs = [json.loads(line) for line in sys.stdin]
    with open(out_path, "w") as out:
        for job in jobs:
            out.write(json.dumps({"k": job["k"], "prediction": answer(job["input"])}) + "\n")


def run_shard(args, binds, jobs):
    xdir = tempfile.mkdtemp(prefix="rsi-shard-")
    out_path = os.path.join(xdir, "answers.jsonl")
    cmd = sandbox_prefix(binds + [DRIVER_DIR], xdir) + [
        sys.executable, os.path.join(DRIVER_DIR, os.path.basename(__file__)), "--child", "--code_dir", args.code_dir,
        "--method_args", json.dumps({k: getattr(args, k) for k in METHOD_ARGS}), "--answers", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, cwd=args.code_dir, text=True)
    if proc.stdin is None:
        raise SystemExit("method process has no stdin")
    proc.stdin.write("".join(json.dumps(j) + "\n" for j in jobs))
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"method process failed (rc={proc.returncode})")
    with open(out_path) as f:
        results = {r["k"]: r["prediction"] for r in map(json.loads, f)}
    shutil.rmtree(xdir, ignore_errors=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true")
    p.add_argument("--method_args")
    p.add_argument("--answers")
    p.add_argument("--code_dir", required=True, help="the experiment's method code (only this is visible to it)")
    p.add_argument("--model_path")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--data_dir")
    p.add_argument("--inputs_dir", help="raw inputs the method may read; labels must not be under it")
    p.add_argument("--output", help="prediction file for the fixed scorer")
    args = p.parse_args()
    args.code_dir = os.path.realpath(args.code_dir)
    if args.child:
        return run_child(args.code_dir, json.loads(args.method_args), args.answers)
    shutil.copy(os.path.abspath(__file__), os.path.join(DRIVER_DIR, os.path.basename(__file__)))
    groups = read_items(args.data_dir)
    binds = [args.code_dir, os.path.realpath(args.model_path), os.path.realpath(args.inputs_dir), sys.prefix,
             os.path.realpath(sys.prefix)]
    predictions = {}
    for shard in shards(groups):
        jobs = [{"k": f"{g}:{k}", "input": item_input(args.data_dir, groups[g][k])} for g, k in shard]
        predictions.update(run_shard(args, binds, jobs))
    write_predictions(groups, predictions, args.output)


if __name__ == "__main__":
    main()
