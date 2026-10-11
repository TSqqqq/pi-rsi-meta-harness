CODEBASE MAP (the only code an experiment can change):
- `model/attention.py` (~300 lines): <what it does>.
  - `Attention.forward` (~L120-180): <the mechanism the paper changes; which tensors, which knobs>.
  - `load(args)`: returns `answer(item_input) -> prediction`, the ONLY entry point the benchmark calls.
- `eval/run_benchmark.py` (fixed, read-only; template: examples/fixed_driver.py): runs the method in a sandbox
  that sees only its code, the model and the raw inputs, one item at a time, shuffled, with grouped items in
  separate processes. A gain must come from how the model processes its input, not from benchmark structure.
- `train.py`: CLI used by the harness; knobs `--lr`, `--steps` come from the experiment's params (editing their
  defaults in code does nothing; change a knob with a `tuning` experiment).

Evaluation (fixed, do not touch): <how the metric is computed>. The quick stage runs <subset>; identical
re-runs vary, and the briefing states the measured baseline mean, sigma and noise threshold.

Not present in this codebase: <components the paper/idea mentions that do not exist here>.
Ideas must be mapped onto what exists here, for example: <one concrete mapping>.
