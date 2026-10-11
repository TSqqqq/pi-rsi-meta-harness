# Architecture

## Core idea

Pi RSI Meta Harness is an **agent harness**, not a single autonomous agent. The harness owns durable state and objective control; Pi supplies research reasoning and code-editing capability.

### Durable layers

1. **Experiment DAG** — the scientific history and source of truth: plan, parents, commit, patch, metrics per stage.
2. **Event store** — every state transition, experiment, metric, agent run, rejected candidate and human intervention.
3. **Pi sessions** — working memory for one task. Every task starts a fresh session; only a worker's fix rounds
   continue its own session.

### Agents

```text
Controller (deterministic: lifecycle, validation, noise verdicts, replication, budgets, safety)
├── Idea-digest / once per research-note version: note → mechanism checklist
├── Scout / one per batch, read-only: raw research record → next batch of hypotheses, best first
└── Worker / one per experiment worktree: implement → run → fix its own failures (same session)
```

There is no critic, reviewer, meta-RSI or plateau lead: the scout reads the raw record (every experiment's
plan and result, rejected candidates with the reason, where each exact patch is) and draws its own conclusions.
Strategy guards that used to be hard-coded (kind quotas, focus rotation, fuzzy de-duplication, parent re-ranking)
were removed for the same reason; the controller only rejects what it must (invalid knob values, an exact
re-run of a measured knob setting, a parent without code).

## Why the long-term memory is external

A research campaign can contain hundreds of experiments. The harness stores the full history in SQLite and
renders it for the scout as an indented experiment tree under a character budget: over budget, the least
informative leaves are folded into per-parent summaries, while promising, pinned and recent nodes never fold.
Context inside one agent run is trimmed too (`pi_extensions/rsi-context.ts`): earlier reasoning is dropped and
tool outputs older than the last turns become one-line stubs before each model call.

## Model routing

The model router exposes two logical tiers:

- `planner`: judgement (idea digest, scout).
- `worker`: implementation, including its fix rounds.

In `single` mode both collapse to one model (thinking levels still differ per tier if configured). In `dual`
mode tasks are routed by task type; `[models].tiers` overrides any task. The harness only selects among models
already configured in Pi, and refuses any provider outside `allowed_providers` before a prompt is sent.

## Fixed benchmark driver

The evaluator runs the method only through a fixed, immutable driver (template `examples/fixed_driver.py`):
the method runs in a bwrap sandbox that sees its code, the model and the raw inputs, gets one test item at a
time, shuffled, with grouped items in separate processes. An automated research loop finds any shortcut the
evaluation leaves open; this removes the shortcuts structurally instead of forbidding them in a prompt.

## Event-driven lifecycle

The controller does not repeatedly ask an LLM whether training has finished. Training runs as subprocesses and the controller consumes stdout/stderr directly. LLM calls happen only on meaningful transitions: planning a batch, implementing a candidate, or fixing a failed run.

This is essential for long-running research because a 2-hour training run should cost GPU time, not 2 hours of polling-model tokens.

## Research-state machine

```text
NEW → INITIALIZED → RUNNING
                     ├─ TARGET_REACHED
                     ├─ BUDGET_EXHAUSTED
                     ├─ SAFETY_STOP
                     ├─ STOPPED_BY_USER
                     ├─ PAUSED ↔ RUNNING
                     └─ PAUSED_FOR_HUMAN ↔ RUNNING
```

A plateau defaults to `PAUSED_FOR_HUMAN`, not an infinite brainstorm loop. A human can provide a scientific insight, pin/revive an older node, or change the external configuration and resume.
