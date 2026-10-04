---
name: experiment-review
description: Review a completed research experiment, attribute its metric change, extract reusable lessons, and decide whether the branch is promising, near-miss, rejected, or needs replication.
disable-model-invocation: true
---

Review experiments scientifically:

- Compare against the correct parent and baseline.
- Use stage fidelity and seed variance when judging evidence.
- Separate implementation bugs, instability, and resource failure from genuine negative scientific results.
- Record lessons from both success and failure.
- Identify plausible interactions with earlier changes, but do not claim causality without an ablation or appropriate comparison.
- Recommend the cheapest next experiment that reduces uncertainty.
- Preserve negative results in the research memory.
