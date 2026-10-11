# Overall Design: RSI Meta Harness

## 1. Positioning

This repository is a **meta harness for metric-driven RSI-style research** over an existing open-source research repository.

It is not a single autonomous agent and it does not let an LLM own the lifecycle. It combines:

- Pi RPC as the reasoning / code-editing runtime;
- two Pi agents: a planner-tier scout that plans each batch and one worker per experiment;
- a durable experiment DAG in SQLite;
- an event-driven Python controller;
- staged experiments and objective evaluation;
- model routing between planner and worker models already configured in Pi;
- a fixed, sandboxed benchmark driver so the method cannot exploit the evaluation;
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
  ├─ state machine          ├─ GPU/process scheduler
  ├─ session manager        ├─ stop/budget policy
  ├─ model router           ├─ noise verdicts + replication
  ├─ context builder        ├─ integrity checks
  └─ candidate validation   └─ event stream
        │ Pi RPC (bwrap sandbox)
        ▼
Agents: idea-digest (once per note) · scout (one per batch) · worker (one per experiment)
        │
        ▼
Git worktrees → staged experiments → fixed driver (sandboxed method) → fixed scorer
        │
        ▼
SQLite experiment DAG + events + agent runs
```

## 3. Durable memory

Pi sessions are working memory. SQLite is the source of truth.

The database stores:

- experiment nodes and DAG edges;
- stage/seed metrics;
- hypotheses and mechanisms;
- code provenance and artifact references;
- success, failure, near-miss, crash, and validation state;
- rejected candidates with the reason;
- model/token/cost accounting (fresh input per model call);
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

Multiple parents represent cross-branch combinations: the worker starts from the first parent's code and the other parents' patches are re-applied. The scout chooses parents itself from the full tree; a never-measured parent is replaced by its nearest measured ancestor.

## 5. Agents and session policy

- **Idea-digest** — runs when the research note's content hash changes; distils the note into 4-10 testable
  mechanisms (claim + mapping onto this codebase). The scout sees the list with how often each was tested.
- **Scout** — one read-only planner-tier call per batch. It gets the raw record only: the experiment tree (each
  node = plan + result, or why it never ran), rejected candidates with reasons, where each exact patch is, the
  idea mechanisms and human guidance. It returns the whole batch, best first. It proposes directions; the
  worker works out the code.
- **Worker** — one per experiment worktree. Its prompt contains the target function sources the scout named and
  the full lineage diff. If preflight, smoke or a stage fails, the traceback and the failing functions' source
  go back to the same session (up to `fix_rounds`); a result that ran is never "fixed".

Every task starts a fresh Pi session; only follow-ups to the same task continue it. There is no critic, reviewer,
meta-RSI or lead: conclusions are the scout's, and the controller keeps no strategy of its own.

## 6. Model routing

`single` mode uses one model for every task (thinking level per tier still applies). `dual` mode routes
judgement (idea-digest, scout) to `planner` and implementation to `worker`; `[models].tiers` overrides any task.
Models must already be configured in Pi; the harness never stores provider keys and refuses providers outside
`allowed_providers` before sending anything.

## 7. Event-driven long-running execution

The controller is the long-running process. Pi is called only at meaningful state transitions.

During a two-hour training run the controller monitors a subprocess and streams logs; it does not spend LLM tokens asking whether the run is done.

A typical cycle is:

```text
scout plans the batch from the raw record
→ mechanical validation (knob values, parents with code, no exact re-run)
→ independent git worktrees (harness files synced, inherited code migrated)
→ workers implement candidates
→ integrity / preflight / smoke checks (failures go back to the worker's session)
→ quick stage → medium stage → full stage + configured seeds
→ noise verdict; gains beyond single-run noise are re-measured
→ update DAG / champion
→ next batch
```

## 8. Browser-first Research Console

The browser console is the primary human interface. It provides:

- full research DAG from baseline to the current champion;
- metric trend from the beginning of the campaign;
- live semantic event timeline;
- per-node hypothesis, mechanism, stage, metrics, deltas, parents/children, patch preview, worker summary, and log tail;
- active experiment and Pi-agent status;
- model usage, token usage, cost, and GPU-hour budgets;
- pause / resume / safe stop;
- pin/unpin a historical node;
- restart exploration from a selected node with human insight;
- a private research notebook that is not automatically sent to agents;
- an idea queue that can send one or all drafts into the next research turn;
- immediate RPC steering of a currently running top-level agent;
- dependency approval/rejection.

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

A plateau normally enters `PAUSED_FOR_HUMAN` and the system waits. The human can add a new insight, revive a node, change configuration, or stop the campaign.

## 11. Safety and research integrity

The scout may choose:

- hypotheses and which branches to extend, revive or combine;
- the split between new mechanisms and knob tuning;
- the order of experiments in a batch.

It must not autonomously redefine:

- evaluator semantics;
- primary metric;
- final test split;
- safety boundaries;
- provider credentials;
- host permissions;
- immutable files.

The evaluator/test paths are hashed and checked. The method is run only through a fixed driver that hides labels, earlier predictions, scores, item order and item grouping (`examples/fixed_driver.py`). Final-test tuning and best-seed cherry-picking are explicitly outside the intended workflow.

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
