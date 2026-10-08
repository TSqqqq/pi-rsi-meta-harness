CODEBASE MAP (the only code an experiment can change):

- `model/attention.py` (~300 lines): <what it does>.
  - `Attention.forward` (~L120-180): <the mechanism the paper changes; which tensors, which knobs>.
- `train.py`: CLI used by the harness; knobs `--lr`, `--steps`.

Evaluation (fixed, do not touch): <how the metric is computed>. The quick stage runs <subset>; baseline
metric is <value> (re-runs vary by <noise>, so a gain under <min_meaningful_delta> is noise).

Not present in this codebase: <components the paper/idea mentions that do not exist here>.
Ideas must be mapped onto what exists here, for example: <one concrete mapping>.
