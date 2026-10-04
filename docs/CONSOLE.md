# RSI Research Console

The browser console is the primary human interface. The terminal is mainly for bootstrap, recovery, and raw stdout.

## What is live

- The experiment DAG updates from the SQLite source of truth.
- The metric trend updates as experiment nodes get stage results.
- The event stream shows controller, agent, evaluator, safety, and human-intervention transitions.
- Raw training/evaluation output is persisted under `logs/experiments/`; selecting a node tails its latest log in the console.
- Active Pi RPC sessions, task types, model routing, tokens, cost, experiment status, and budgets update continuously.

The page polls the local stdlib HTTP API at a short interval. Polling the SQLite state does **not** call an LLM and therefore consumes no model tokens.

## Human controls

The console can:

- pause / resume / safely stop a campaign;
- click any experiment node for hypothesis, metrics, parameter effects, worker summary, patch preview, logs, parents, and children;
- pin/unpin an old node to the frontier;
- continue from a selected node with a human scientific insight;
- maintain a private research notebook that is **not** automatically injected into agent context;
- keep an idea queue, then send one idea or all ideas into the **next research turn**;
- steer an already-running top-level Pi RPC agent immediately;
- approve/reject optional dependency requests.

There are intentionally two different intervention modes:

1. **Idea queue / next turn** — durable guidance is added to SQLite and is seen by future planners/scouts through the context builder. This does not interrupt a current model call.
2. **Steer now** — a `steer` guidance item is delivered by the controller to an active Pi RPC session using the RPC steering mechanism.

## Safety

The console binds to `127.0.0.1` by default and sends no CORS headers. For a remote GPU host use SSH port forwarding:

```bash
ssh -L 8765:127.0.0.1:8765 your-server
```

Do not bind the control console to a public interface without adding your own authenticated reverse proxy. The console can stop jobs and alter the research frontier.

`Stop safely` does not kill processes in the HTTP thread. It records a control request; the controller then aborts Pi runs, terminates experiment subprocesses, and persists the final state.
