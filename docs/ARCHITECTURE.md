# Architecture

## Core idea

Pi RSI Meta Harness is an **agent harness**, not a single autonomous agent. The harness owns durable state and objective control; Pi supplies research reasoning and code-editing capability.

### Durable layers

1. **Experiment DAG** — the scientific history and source of truth.
2. **Event store** — every state transition, experiment, metric, agent run, and human intervention.
3. **Pi sessions** — role-specific working memory. Useful, resumable, but replaceable.
4. **Research policy** — periodically revised meta-level search policy.

### Agent hierarchy

```text
Controller
├── Lead / strategic reasoning
├── Scouts / diverse hypothesis generation
│   └── may call Pi subagents internally
├── Critic / batch selection
├── Worker / one experiment worktree
├── Reviewer / scientific attribution
├── Meta-RSI / search-policy improvement
└── Validator / publication-grade confirmation
```

Top-level agents are separate Pi RPC sessions. A Pi agent may still use installed skills/extensions such as subagents or code navigation. That gives two levels of orchestration without making subagents responsible for global lifecycle.

## Why the long-term memory is external

A research campaign can contain hundreds or thousands of experiments. Feeding all transcripts, diffs, and logs into every model call wastes tokens and degrades decision quality. The harness stores the full history in SQLite and builds compact task-specific contexts containing only relevant ancestors, successful analogues, failures, near-misses, global patterns, and human guidance.

## Model routing

The model router exposes two logical tiers:

- `planner`: insight-heavy tasks, typically expensive/high-reasoning.
- `worker`: mechanical implementation and repetitive tasks, typically cheaper.

In `single` mode both roles collapse to Pi's default or the configured single model. In `dual` mode tasks are routed deterministically by task type. Repeated worker failure may escalate to the planner tier. The harness only selects among models already configured in Pi.

## Event-driven lifecycle

The controller does not repeatedly ask an LLM whether training has finished. Training runs as subprocesses and the controller consumes stdout/stderr directly. LLM calls happen only on meaningful transitions: proposing ideas, selecting a batch, implementing a candidate, reviewing a result, or running a periodic meta-review.

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
