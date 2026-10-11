# Repository Manifest

Core runtime:

- `controller.py` — entrypoint
- `rsi_harness/controller.py` — event-driven research state machine
- `rsi_harness/pi_rpc.py` — strict JSONL Pi RPC client, bwrap sandbox, tool-call/timeout budgets, full transcripts
- `rsi_harness/agent_runtime.py` — session/role/model routing, provider allowlist, token/tool accounting
- `rsi_harness/db.py` — SQLite experiment DAG, metrics, events, agent runs, human guidance
- `rsi_harness/experiments.py` — staged experiment runner and live terminal output
- `rsi_harness/context.py` — scout briefing: experiment tree (plan + result per node), rejected candidates, idea mechanisms
- `pi_extensions/rsi-context.ts` — per-call context trimming (old reasoning and tool outputs) and git-history block for workers
- `rsi_harness/stopping.py` — target/budget/plateau/safety stop policy
- `rsi_harness/dashboard.py` + `dashboard/index.html` — browser-first research console with DAG, logs, controls, notes, idea queue, and steering
- `rsi_harness/safe_pip.py` — guarded on-demand dependency installation

Research intelligence:

- `prompts/scout.md` — the research lead: reads the raw record, plans each batch (JSON contract, anti-exploit rules)
- `prompts/worker.md` — implements one hypothesis and fixes its own runtime failures in the same session
- `prompts/idea_digest.md` — distils the research note into a mechanism checklist (once per note version)
- `prompts/codebase.example.md` — template for the paper-specific codebase map (`[project] codebase_map`)

Documentation and validation:

- `docs/ARCHITECTURE.md`
- `docs/ADAPTER.md`
- `docs/PI_RPC.md`
- `docs/SAFETY.md`
- `docs/CONSOLE.md`
- `docs/OVERALL_DESIGN.md`
- `docs/LESSONS.md` — integration lessons and fix log
- `examples/fixed_driver.py` — template for an unexploitable benchmark driver (sandboxed method, grouped items split across processes)
- `research.example.toml`
- `tests/fake_pi.py`
- `tests/test_controller_smoke.py` — end-to-end fake-RPC campaign through validated target stop

- `tests/test_dashboard.py` — console state and human-control API
- `tests/test_rpc.py` — transcripts, large RPC lines, tool-call/timeout/external aborts
- `tests/dashboard_e2e/` — jsdom click test of every console button
