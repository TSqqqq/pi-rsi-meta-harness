---
name: research-hypothesis
description: Generate falsifiable, mechanism-driven research hypotheses from an experiment DAG while using successes, failures, near-misses, parameter effects, and human guidance.
disable-model-invocation: true
---

When generating research hypotheses:

- Start from evidence in the supplied experiment DAG, not generic brainstorming.
- Keep structural/mechanistic innovation separate from numeric hyperparameter tuning.
- Include the parent node(s), mechanism, expected direction/magnitude, uncertainty, and cheapest falsification experiment.
- Use failures as constraints: if an idea resembles a failed node, explain the material difference.
- Maintain diversity. Do not let all proposals descend from the current champion.
- Consider promising nodes, near-misses, underexplored branches, and combinations of independently useful mechanisms.
- Prefer ideas that can produce interpretable evidence even when they fail.
- Never change evaluator semantics, the final held-out split, or the stated objective to create an apparent gain.
