# RSI Research Harness Rules

This repository is a research harness. When working inside a target experiment worktree:

1. Preserve the fixed evaluator, final test split, research objective, safety policy, and budget controls.
2. Do not delete failed experiments or rewrite historical metrics.
3. Before proposing an experiment, consult the supplied research context: relevant ancestors, successes, failures, near-misses, parameter effects, and human guidance.
4. Separate structural/mechanistic hypotheses from numeric hyperparameter tuning.
5. Prefer the cheapest experiment that can falsify a hypothesis.
6. Record negative results. A useful failed experiment should produce a reusable lesson.
7. Do not use the final held-out test set as a tuning loop.
8. Do not run `sudo`, modify host CUDA/drivers, mount host sockets, or change system services.
9. Do not run arbitrary `pip install`. First try `python -m rsi_harness.safe_pip install <package>`; allowlisted packages install into an external experiment virtualenv. If rejected as unapproved, use `python -m rsi_harness.safe_pip request <package> <reason>` and wait for human approval.
10. Existing Pi skills/extensions remain available. Use subagents when they improve analysis, but the top-level experiment lifecycle is controlled by the harness.
