---
name: paper-validation
description: Validate a candidate research contribution with multi-seed runs, ablations, fair baselines, compute accounting, and locked held-out evaluation before treating it as publication evidence.
disable-model-invocation: true
---

Before a result becomes publication evidence:

- Require full-fidelity reproduction and configured independent seeds.
- Report mean and variance, not only the best seed.
- Use strong, fair baselines with comparable compute/parameters where relevant.
- Run ablations that isolate the claimed mechanism.
- Keep the final held-out test locked from the iterative tuning loop.
- Record compute cost and failed validation attempts.
- Do not overstate novelty or causality beyond the evidence in the experiment DAG.
