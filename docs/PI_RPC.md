# Pi RPC Integration

The harness starts one Pi subprocess per agent (the scout, each experiment's worker), inside a bwrap sandbox:

```bash
pi --mode rpc --session-id rsi-scout --session-dir ./sessions --name scout \
   --no-extensions --extension pi_extensions/rsi-context.ts \
   --no-skills --no-prompt-templates --no-context-files --append-system-prompt AGENTS.md
```

- `--no-extensions` keeps interactive extension dialogs from blocking a headless agent; the harness context
  extension and any `[pi] extra_extensions` are loaded explicitly with `--extension`.
- `--no-skills --no-prompt-templates --no-context-files` (`[pi] minimal_resources`) keep the skill catalogue,
  templates and auto-discovered AGENTS.md out of every turn; the harness rules are appended once.
- Read-only agents get `--tools read,grep,find,ls`.

The controller communicates using one-JSON-object-per-line RPC messages and continuously drains stdout so Pi
cannot block on a full pipe buffer. Every record is written to a per-session transcript tagged with the
`agent_runs` id.

RPC operations used by the harness: `prompt` (and `prompt` with `streamingBehavior: "steer"` for steering and
the tool-budget wrap-up), `new_session`, `get_last_assistant_text`, `get_state`, `get_available_models`,
`set_model`, `set_thinking_level`, `get_available_thinking_levels`, `get_session_stats`, `compact`, `abort`.

Every task sends `new_session` first (long-term memory is the SQLite DAG; Pi would otherwise resume an old
session file after a controller restart). A follow-up that must keep the context (a worker's fix round, a
scout asked again for its cut-off JSON) skips it. A worker's Pi process is stopped when its experiment ends.

Budgets: each task has a tool-call and a wall-time limit. Read-only roles are steered to answer at the
tool-call limit and aborted three calls later; workers are aborted at the limit, which covers all their turns.
