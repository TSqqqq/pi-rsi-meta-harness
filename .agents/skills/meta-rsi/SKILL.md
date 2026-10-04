---
name: meta-rsi
description: Improve the research search policy using aggregate experiment outcomes, branch structure, GPU/token efficiency, and human insights without changing the evaluator or safety boundary.
disable-model-invocation: true
---

Optimize the research process itself:

- Identify branches with high scientific return per GPU-hour and branches that are saturated.
- Revive informative near-misses when new evidence changes their value.
- Recommend cross-branch combinations only when mechanisms plausibly compose.
- Adjust exploration versus exploitation based on evidence.
- Use agent/model cost data to reserve expensive models for tasks that benefit from deeper reasoning.
- Prefer policies that improve expected gain, information gain, and reproducibility per unit compute.
- Never modify evaluator semantics, target metric, final test split, credentials, permission boundaries, or immutable safety policy.
