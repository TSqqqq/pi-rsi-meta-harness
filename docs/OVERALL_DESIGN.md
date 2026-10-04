# Overall Design: RSI Meta Harness

## 1. Positioning

This repository is a **meta harness for metric-driven RSI-style research** over an existing open-source research repository.

It is not a single autonomous agent and it does not let an LLM own the lifecycle. It combines:

- Pi RPC as the reasoning / code-editing runtime;
- multiple role-specific Pi sessions;
- a durable experiment DAG in SQLite;
- an event-driven Python controller;
- staged experiments and objective evaluation;
- model routing between planner and worker models already configured in Pi;
- additive Pi skills, including existing user skills such as subagents;
- a browser-based Research Console for observation and human intervention;
- hard stop, budget, evaluator-integrity, and dependency boundaries.

The operating principle is:

> Agents decide **what is worth trying**. The harness decides **when/how it runs, what counts as evidence, what is remembered, and when to stop**.

## 2. Layers

```text
Human / Research Console
        │
        ▼
Meta Harness Controller
  ├─ state machine
  ├─ session manager
  ├─ model router
  ├─ context builder
  ├─ frontier selector
  ├─ GPU/process scheduler
  ├─ stop/budget policy
  ├─ integrity checks
  └─ event stream
        │ Pi RPC
        ▼
Agent hierarchy
  ├─ Lead
  ├─ Scouts
  ├─ Critic
  ├─ Workers
  ├─ Reviewer
  ├─ Meta-RSI
  └─ Validator
        │
        ▼
Git worktrees → staged experiments → fixed evaluator
        │
        ▼
SQLite experiment DAG + beliefs + parameter effects + events
```

## 3. Durable memory

Pi sessions are working memory. SQLite is the source of truth.

The database stores:

- experiment nodes and DAG edges;
- stage/seed metrics;
- hypotheses and mechanisms;
- code provenance and artifact references;
- success, failure, near-miss, crash, and validation state;
- reusable lessons;
- parameter effects;
- research beliefs;
- model/token/cost accounting;
- research-policy versions;
- human guidance;
- private human notes;
- queued human ideas;
- dependency approval requests;
- immutable evaluator hashes;
- the full semantic event timeline.

A failed branch remains useful evidence and can later be revived.

## 4. Experiment DAG, not a flat log

Each experiment points to one or more parents. A node may be:

- baseline;
- planned / implemented / running;
- promising;
- near-miss;
- rejected;
- failed / invalid;
- champion.

Multiple parents represent cross-branch combinations. The frontier selector does not follow only the current champion; it rewards performance, novelty, uncertainty, underexplored nodes, and explicitly pinned human choices.

## 5. Agent hierarchy and session policy

Top-level agents are separate Pi RPC sessions:

- **Lead** — long-lived strategic context and plateau analysis.
- **Scouts** — diverse hypothesis generation; may internally use installed subagent skills.
- **Critic** — ranks a candidate batch.
- **Worker** — one experiment/worktree/session; implementation-oriented.
- **Reviewer** — attributes outcomes and updates lessons/beliefs/parameter effects.
- **Meta-RSI** — periodically improves the search policy.
- **Validator** — publication-grade replication and ablation when enabled.

Existing Pi skills/extensions are not disabled. Harness role skills are loaded additively and marked `disable-model-invocation: true`; the harness explicitly initializes the relevant role skill once per role session, while normal user-installed skills such as subagents remain available.

## 6. Model routing

Default mode uses Pi's normal model for every role.

Optional dual mode defines:

- `planner`: expensive/high-reasoning model for planning, hypotheses, critique, plateau analysis, cross-branch synthesis, and meta-RSI;
- `worker`: cheaper model for implementation and repetitive tasks.

Models must already be configured in Pi. The harness calls Pi RPC model-selection commands and never stores provider keys.

Worker tasks can be escalated to the planner tier after repeated failure. Thinking level is also routed separately, so cheap/minimal, cheap/medium, strong/medium, and strong/high can be treated as different compute tiers.

## 7. Event-driven long-running execution

The controller is the long-running process. Pi is called only at meaningful state transitions.

During a two-hour training run the controller monitors a subprocess and streams logs; it does not spend LLM tokens asking whether the run is done.

A typical cycle is:

```text
choose frontier
→ scouts propose diverse falsifiable hypotheses
→ critic selects a batch
→ create independent git worktrees
→ workers implement candidates
→ integrity / preflight checks
→ quick stage
→ early reject or promote
→ medium stage
→ early reject / near-miss or promote
→ full stage + configured seeds
→ reviewer extracts lessons and parameter effects
→ update DAG / champion
→ periodic meta-RSI review
→ next frontier
```

## 8. Browser-first Research Console

The browser console is the primary human interface. It provides:

- full research DAG from baseline to current frontier;
- metric trend from the beginning of the campaign;
- live semantic event timeline;
- per-node hypothesis, mechanism, stage, metrics, deltas, lessons, parents/children, parameter effects, patch preview, worker summary, and log tail;
- active experiment and Pi-agent status;
- model usage, token usage, cost, and GPU-hour budgets;
- pause / resume / safe stop;
- pin/unpin a historical node;
- restart exploration from a selected node with human insight;
- a private research notebook that is not automatically sent to agents;
- an idea queue that can send one or all drafts into the next research turn;
- immediate RPC steering of a currently running top-level agent;
- dependency approval/rejection;
- plateau reports and active meta-RSI policy.

The console polls SQLite/process logs. This UI refresh consumes no LLM tokens.

## 9. Human intervention semantics

There are three separate mechanisms:

1. **Notes**: private scratchpad. Never injected automatically.
2. **Idea queue**: ideas are drafts until the human clicks Send. Sent ideas become durable human guidance and enter future context-building/planning turns.
3. **Steer now**: the controller sends an RPC steering message to an active Pi session.

This separation prevents every human note from becoming prompt noise while still supporting immediate intervention when needed.

## 10. Stopping policy

Research must not run forever merely because agents can still generate text.

The controller stops or pauses on:

- validated target reached;
- GPU-hour, wall-time, token, or experiment budget exhaustion;
- immutable evaluator/test integrity failure;
- repeated fatal experiment failures;
- explicit human stop;
- plateau.

A plateau normally enters `PAUSED_FOR_HUMAN`. A strong planner model produces a plateau report once, then the system waits. The human can add a new insight, revive a node, change configuration, or stop the campaign.

## 11. Safety and research integrity

The harness may optimize:

- hypotheses;
- frontier allocation;
- exploration/exploitation ratio;
- research prompts/policy;
- model routing;
- experiment budgets;
- search spaces.

It must not autonomously redefine:

- evaluator semantics;
- primary metric;
- final test split;
- safety boundaries;
- provider credentials;
- host permissions;
- immutable files.

The evaluator/test paths are hashed and checked. Final-test tuning and best-seed cherry-picking are explicitly outside the intended workflow.

## 12. Dynamic dependencies

The harness has no mandatory third-party Python dependency.

An agent that truly needs Optuna, Ray, SQLAlchemy, or another package uses `rsi_harness.safe_pip`. Allowlisted packages can be installed into an experiment-local virtual environment; non-allowlisted packages create a human approval request visible in the console.

URLs, VCS installs, editable installs, sudo, and host-driver changes are rejected.

## 13. Reusing on a new open-source project

The reusable boundary is `research.toml`:

- repository path;
- objective metric and direction;
- baseline / quick / medium / full commands;
- evaluator command that emits JSON;
- GPU resources;
- mutable / immutable paths;
- budgets and stop policy;
- model routing.

The target repository should first be reproducible manually. After that:

```bash
cp research.example.toml research.toml
# edit adapter
python controller.py init
python controller.py run
```

Open the local console at `http://127.0.0.1:8765`, or tunnel it from a remote GPU server.

## 14. Why this is an RSI meta harness

There are three levels of optimization:

- **L0 — research object**: improve the target model/system metric.
- **L1 — research policy**: learn which mechanisms, branches, experiment budgets, and search strategies produce useful evidence.
- **L2 — harness policy**: learn when expensive reasoning is worth paying for, which roles need which model tier, and how much exploration/parallelism is efficient.

L1/L2 remain bounded by immutable evaluation, safety, and budget rules. That gives recursive improvement of the research process without giving agents permission to move the goalposts.
