# Adapting a New Open-Source Research Repository

The harness is intentionally domain-agnostic. The target repository must supply a reproducible baseline and a machine-readable evaluator.

## 1. Reproduce manually first

Before automation, confirm the upstream repository can train and evaluate successfully by hand. Record the environment, dataset version, checkpoint source, and exact baseline metric.

## 2. Configure `research.toml`

The most important fields are:

- `[project].repo`: absolute or config-relative path to the target Git repo.
- `[objective]`: metric name, maximize/minimize, target, meaningful delta.
- `[commands]`: baseline/quick/medium/full/evaluate commands.
- `[files]`: what agents may and may not modify.
- `[budget]`: hard resource/token/wall-time caps.

Command templates receive:

```text
{repo}
{worktree}
{experiment_id}
{stage}
{seed}
{gpu_ids}
{python}
```

plus every key of `[experiment]` (paper-specific knobs such as `{lr}`).

## 2b. Paper-specific agent context

- `[project] codebase_map`: a short Markdown map appended to scout and worker prompts (template:
  `prompts/codebase.example.md`). List the mutable files with key functions and line ranges, how the metric
  is computed, the baseline value and its noise, and the components the paper mentions that are **not** in
  the code. This is the single biggest factor in whether a small worker model finishes within its budget.
- `[project] idea_file`: optional research note injected into every scout turn.
- Keep large model/data caches out of worktrees: link them to one shared, pre-populated cache and run offline,
  otherwise every experiment silently re-downloads (see `docs/LESSONS.md`).
- Run the baseline twice with the same config and set `[objective] min_meaningful_delta` above the observed gap.

## 3. Evaluator contract

`commands.evaluate` may print arbitrary logs, but its **last JSON object** must include the configured metric key:

```json
{"metric": 72.41, "loss": 1.83, "examples": 10000}
```

The controller parses the final JSON object and stores the other fields as extra metric metadata.

For a minimization objective, set:

```toml
[objective]
direction = "minimize"
```

## 4. Fidelity stages

Use quick/medium/full only when lower-fidelity results are meaningfully predictive.

Typical pattern:

- `quick`: short training / small data / cheap proxy, designed to falsify bad ideas.
- `medium`: larger budget for promising candidates.
- `full`: publication-relevant training with multiple configured seeds.

If a repository does not support a useful medium stage, leave `commands.medium` empty.

## 5. Preserve attribution

Worker agents should make one attributable conceptual change per node whenever possible. Numeric tuning should normally happen after a structural mechanism has shown evidence of benefit; otherwise the DAG becomes hard to interpret.

## 6. Crossovers

A DAG node may have multiple parents. The primary parent supplies the initial Git commit. The worker receives all parent IDs and research context and is responsible for implementing the intended combination. This avoids automatic Git merges that can be syntactically correct but scientifically invalid.

## Stage-matched baseline calibration

`init` calibrates the unchanged upstream code at every configured fidelity. If the normal quick/medium command requires an experiment-only config, provide `commands.baseline_quick` and/or `commands.baseline_medium` overrides. The harness never compares a quick proxy directly against a full-fidelity parent score.
