# Pi RSI Meta Harness

A reusable **RSI-oriented meta harness** for metric-driven research on top of an existing open-source repository.

The harness separates **research intelligence** from **research control**:

- **Pi agents** decide what to investigate, propose hypotheses, implement changes, review failures, and periodically improve the research policy.
- The **controller** owns lifecycle, budgets, GPU/process management, session management, stop conditions, logging, immutable-evaluator checks, and human intervention.
- A persistent **SQLite experiment DAG** is the long-term memory. Pi sessions are working memory, not the source of truth.
- A built-in **real-time terminal stream + web dashboard** shows the complete path from baseline to current frontier/champion.
- A **model router** supports one-model mode by default and optional planner/worker dual-model mode using models already configured in Pi.

The default runtime uses only the Python standard library. Optional packages such as Optuna or Ray are intentionally **not preinstalled**; agents may request them through the guarded dependency mechanism when a research phase actually needs them.

## Architecture

```text
Human ── pause/resume/note/pin/branch/stop
                    │
                    ▼
          RSI Meta Harness Controller
  ┌───────────────────────────────────────────┐
  │ state machine / budgets / stopping        │
  │ session manager / model router            │
  │ DAG memory / context builder              │
  │ GPU scheduler / process monitor           │
  │ event store / terminal / dashboard        │
  └──────────────────────┬────────────────────┘
                         │ Pi RPC
           ┌─────────────┼──────────────┐
           ▼             ▼              ▼
         Lead          Scouts        Reviewer
           │          (may use         │
           │          subagents)       │
           └─────────────┬──────────────┘
                         ▼
                    Candidate ideas
                         │
                       Critic
                         │
                         ▼
                Worker Pi per experiment
                         │
                  Git worktree + GPU
                         │
                   fixed evaluator
                         │
                         ▼
                 SQLite experiment DAG
```

## Requirements

- Linux/macOS (Linux recommended for GPU research)
- Python 3.11+
- Git
- Pi installed and working: `pi --version`
- The target research repository should already be reproducible manually.
- For GPU scheduling, `nvidia-smi` is optional but recommended.

Pi's existing global/project skills and extensions are **not disabled**. Harness research skills are added with repeatable `--skill` flags, so tools such as subagents or code-navigation skills remain available. The controller remains the top-level orchestrator.

## Quick start

```bash
unzip pi-rsi-meta-harness.zip
cd pi-rsi-meta-harness
cp research.example.toml research.toml
$EDITOR research.toml

python controller.py init
python controller.py run
```

Dashboard (control API is enabled by default on localhost):

```text
http://127.0.0.1:8765
```

For a remote server:

```bash
ssh -L 8765:127.0.0.1:8765 your-server
```

Then open `http://localhost:8765` locally.

## Research Console: primary interface

After `python controller.py run`, the main human interface is the browser console at `http://127.0.0.1:8765`. The terminal remains useful for bootstrap and raw stdout, but normal observation and intervention should happen in the console.

The console shows the complete experiment DAG, metric trend, live event stream, selected-node details, live/latest experiment logs, active Pi sessions, model routing, token/GPU budgets, plateau reports, and RSI policy. It also provides pause/resume/safe-stop, pin/unpin, continue-from-node, a private research notebook, a draft idea queue with send-one/send-all, immediate steering of a running agent, and dependency approvals. Notes remain private until explicitly copied/sent as an idea.

See `docs/CONSOLE.md` and `docs/OVERALL_DESIGN.md`.

## Human controls

From another shell:

```bash
python controller.py status
python controller.py pause
python controller.py resume
python controller.py note "Focus on data curriculum; stop spending budget on normalization variants"
python controller.py pin exp_000117
python controller.py branch exp_000117 "Revisit this node: I suspect the gain came from regularization"
python controller.py steer scout-1 "Prioritize the new data-mixture insight in the current reasoning turn"
python controller.py stop
```

`plateau` does not silently burn tokens forever. The default transition is `PAUSED_FOR_HUMAN`, where the dashboard shows the evidence and waits for a human insight or explicit resume.

## Single-model and dual-model routing

Default: do not override Pi's currently configured default model.

```toml
[models]
mode = "single"
default = ""
```

Optional dual-model mode:

```toml
[models]
mode = "dual"
planner = "rightapi-codex/gpt-6.1-sol"
worker = "rightapi-codex/a-cheaper-model"
planner_thinking = "high"
worker_thinking = "minimal"
allow_escalation = true
max_worker_failures_before_escalation = 2
```

Models must already be configured in Pi. The harness calls Pi RPC `get_available_models`, `set_model`, and `set_thinking_level`; it does not store provider keys.

Planner model is used for high-value reasoning such as research planning, hypothesis generation, cross-branch synthesis, plateau analysis, and meta-RSI. Worker model is used for mechanical implementation, log triage, simple repairs, and repetitive tasks. A worker can be escalated to the planner model after repeated failure.

## Adapting a new open-source project

The project-specific contract is `research.toml`. The harness needs four things:

1. Target repo path.
2. Training commands for quick/medium/full fidelity.
3. An evaluation command whose **last JSON object** on stdout includes `{"metric": <number>}`.
4. Mutable/immutable path rules and budgets.

Example evaluator output:

```json
{"metric": 72.41, "extra": {"loss": 1.83}}
```

See `docs/ADAPTER.md`.


### Stage-matched baselines

Quick/medium/full proxy scores are **not compared against a full-fidelity baseline**. During `init`, the harness calibrates the baseline at every configured fidelity stage and stores stage-matched baseline metrics. A quick-stage candidate is compared to the parent's quick-stage metric (or the quick baseline if the parent was pruned before that stage); the same rule applies to medium/full. Only validated full-stage results can become the global champion or trigger `TARGET_REACHED`.

This prevents a common automated-research error where incomparable proxy and full-budget metrics are mixed in the same leaderboard.

## Stopping logic

The controller can stop/pause on:

- `TARGET_REACHED`: target metric confirmed at full fidelity and required seeds.
- `PLATEAU`: no meaningful gain for the configured number of completed experiments; transitions to `PAUSED_FOR_HUMAN` by default.
- `BUDGET_EXHAUSTED`: GPU-hours, wall time, agent-token budget, or experiment cap reached.
- `SAFETY_STOP`: immutable files changed, repeated fatal errors, or integrity violation.
- `HUMAN_PAUSE` / `HUMAN_STOP`.

No LLM call is made merely to poll a running training process. During long training, the controller monitors subprocesses and streams logs without spending model tokens.

## Dynamic dependencies

The harness itself is stdlib-only. Agents are instructed not to run arbitrary `pip install`. Instead:

```bash
python -m rsi_harness.safe_pip install optuna
```

Allowlisted packages install into an **external per-experiment virtualenv** under the harness state directory (not inside the Git worktree). By default that venv inherits the base environment's site-packages, so large existing ML dependencies do not need to be duplicated. If a package is not allowlisted, request it:

```bash
python -m rsi_harness.safe_pip request somepkg "needed for this experiment"
```

Unlisted packages become human approval requests. `sudo`, URLs, VCS installs, editable installs, and system-site installs are rejected by the helper.

## Tests

```bash
python -m unittest discover -s tests -v
```

The test suite includes a fake Pi RPC process, so tests do not require real model calls.

## Important research-integrity boundary

The harness may optimize research strategy, model routing, frontier allocation, hypothesis prompts, experiment budgets, or search spaces. It must **not** autonomously redefine the metric, alter the locked test split, change evaluator semantics, weaken safety boundaries, or move the goalposts after seeing results.

See `docs/SAFETY.md`.

## Pi references used by this implementation

- RPC mode: https://pi.dev/docs/latest/rpc
- RPC commands: https://pi.dev/docs/latest/rpc-commands
- JSON event stream: https://pi.dev/docs/latest/json
- CLI resources / `--skill`: https://pi.dev/docs/latest/cli
- Skills: https://pi.dev/docs/latest/skills
- Sessions and context: https://pi.dev/docs/latest/sessions
