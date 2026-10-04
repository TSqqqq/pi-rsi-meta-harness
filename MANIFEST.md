# Repository Manifest

Core runtime:

- `controller.py` — entrypoint
- `rsi_harness/controller.py` — event-driven research state machine
- `rsi_harness/pi_rpc.py` — strict JSONL Pi RPC client
- `rsi_harness/agent_runtime.py` — session/role/model routing and token accounting
- `rsi_harness/db.py` — SQLite DAG, events, beliefs, parameter effects, policies, human guidance
- `rsi_harness/experiments.py` — staged experiment runner and live terminal output
- `rsi_harness/frontier.py` — multi-branch frontier selection
- `rsi_harness/stopping.py` — target/budget/plateau/safety stop policy
- `rsi_harness/dashboard.py` + `dashboard/index.html` — browser-first research console with DAG, logs, controls, notes, idea queue, and steering
- `rsi_harness/safe_pip.py` — guarded on-demand dependency installation

Research intelligence:

- `.agents/skills/*` — compact, role-specific research skills loaded additively into Pi
- `prompts/*` — structured JSON contracts for scouts, critic, worker, reviewer, meta-RSI, plateau review

Documentation and validation:

- `docs/ARCHITECTURE.md`
- `docs/ADAPTER.md`
- `docs/PI_RPC.md`
- `docs/SAFETY.md`
- `docs/CONSOLE.md`
- `docs/OVERALL_DESIGN.md`
- `research.example.toml`
- `tests/fake_pi.py`
- `tests/test_controller_smoke.py` — end-to-end fake-RPC campaign through validated target stop

- `tests/test_dashboard.py` — console state and human-control API
