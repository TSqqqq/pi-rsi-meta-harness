# Pi RPC Integration

The harness starts each top-level Pi role as a long-lived subprocess:

```bash
pi --mode rpc --session-id rsi-scout-1 --session-dir ./sessions --name scout-1
```

No `--no-skills` flag is used. Existing global/project Pi skills and extensions remain discoverable.

The controller communicates using one-JSON-object-per-line RPC messages. It continuously drains stdout so Pi cannot block on a full pipe buffer.

Important RPC operations used by the harness:

- `prompt`
- `get_last_assistant_text`
- `get_state`
- `get_available_models`
- `set_model`
- `set_thinking_level`
- `get_available_thinking_levels`
- `get_session_stats`
- `abort`

For high-value planner tasks the model router can switch a session to the configured planner model and thinking level. Mechanical tasks can use the worker model. The session remains the same across a model switch.

Role skills are initialized once per controller process by sending `/skill:<name>` to the corresponding long-lived session. Subsequent controller prompts contain the task and compact research state, not another full copy of the skill body.
