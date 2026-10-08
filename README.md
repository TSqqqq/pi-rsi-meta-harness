# Pi RSI Meta Harness

A reusable **RSI-oriented meta harness** for metric-driven research on top of an existing open-source repository.

The harness separates **research intelligence** from **research control**:

- **Pi agents** decide what to investigate, propose hypotheses, implement changes, review failures, and periodically improve the research policy.
- The **controller** owns lifecycle, budgets, GPU/process management, session management, stop conditions, logging, immutable-evaluator checks, and human intervention.
- A persistent **SQLite experiment DAG** is the long-term memory. Pi sessions are working memory, not the source of truth.
- A built-in **real-time terminal stream + web dashboard** shows the complete path from baseline to current frontier/champion.
- A **model router** supports one-model mode by default and optional planner/worker dual-model mode using models already configured in Pi.

The default runtime uses only the Python standard library. Optional packages such as Optuna or Ray are intentionally **not preinstalled**; agents may request them through the guarded dependency mechanism when a research phase actually needs them.

## 核心思想（一页读懂）

RSI 在这里指：**在固定的评估器下，让 agent 反复改进一份论文代码，同时让搜索策略本身也随证据改进。**

六条设计原则，决定了代码里的每个取舍：

1. **固定目标，迭代手段。** 评估器、测试划分、研究目标不可修改（哈希校验），agent 只能改 `[files] mutable` 允许的代码。
2. **长期记忆在 SQLite，不在会话里。** 每个实验是 DAG 上的一个节点：假设、父节点、git commit、补丁、各阶段指标、结论、参数影响。Pi 会话只是工作记忆，随时可以压缩或丢弃。
3. **两层自我改进。** 知识层：每次复盘写回结论和失败原因，下一轮 scout 必须读到；策略层：meta-RSI 定期调整探索比例、复活哪些分支、组合哪些方向。
4. **控制器管生命周期，agent 只做判断。** 调度、GPU、超时、中断、停止、记账全部是确定性代码；LLM 只负责提假设、选假设、改代码、复盘。等训练时不调用 LLM。
5. **硬边界，不靠 prompt 约束。** 沙箱、工具调用上限、超时、provider 白名单，都在代码里强制执行。实测小模型会无限调用工具、改主仓库，prompt 写了也照样越界。
6. **证据优先于结论。** 先测噪声再判结论；没跑出指标的实验不复盘（否则 reviewer 会编造）；失败实验不删除，修正结论也只追加事件。

### 一轮迭代

```text
停止检查 ── 达标 / 预算用尽 / 连续失败 / 平台期 / 不可修改文件被改
   │
① 汇总状态    SQLite：成功、失败、接近提升、结论、参数影响、人工想法、idea 文件
② 选父节点    frontier 按指标、新颖度、不确定性、置顶打分
③ 提假设      scout ×N（各有侧重：深挖冠军 / 复活接近提升 / 全新机制 / 交叉组合）→ JSON
④ 选假设      critic 选出 ≤ max_parallel 个
⑤ 并行实验    每个假设一条独立流水线：
   │  worktree（来自父实验 commit）→ worker 在沙箱里改代码 → 安全检查 → commit
   │  → 预检 → quick → medium → full（逐级晋升，不达标就止损）→ 固定评估器打分
   │  → reviewer 复盘：状态、结论、结论库、参数影响
⑥ 批次结算    更新冠军；有新数据就进入下一轮，没有就暂停等人；每 N 个实验跑一次 meta-RSI
   └──────────→ 回到停止检查
```

### 失败处理

| 情况 | 处理 |
|---|---|
| worker 超时 / 超出工具调用上限 | 判失败，不修复（再跑一次 agent 只会重复失败） |
| 补丁预检或运行报错 | repair agent 有限次修复（`[repair] max_attempts`），修好就重跑 |
| 没有任何指标 | 记 `NOT RUN`，不复盘 |
| \|Δ\| < `min_meaningful_delta` | 判 `near_miss`（未证实），既不算有效也不算有害 |
| 改了主仓库 | 保存 diff，`SAFETY_STOP` |
| 连续 `max_worker_failures` 次失败 | 暂停等人；点"继续"后计数清零 |
| 进程崩溃 / 被 kill | 下次启动时把未完成的记录标为 `interrupted` |

### 安全与成本（全部可配置，默认开启）

| 机制 | 配置 | 作用 |
|---|---|---|
| bwrap 沙箱 | `[pi] sandbox` | 主机只读，只有自己的 worktree 可写；没有 bwrap 时拒绝启动 |
| 关闭全局扩展 | `[pi] extensions` | 防止交互确认框卡死无人值守的 agent，减少额外工具 |
| provider 白名单 | `[models] allowed_providers` | 不在名单内的 provider 在发送 prompt 前就被拒绝，是唯一的花钱开关 |
| 单任务预算 | `[agents] max_tool_calls / *_timeout_min` | 超出即中止 |
| 主仓库检查 | 内置 | agent 越界修改时存 diff 并安全停止 |
| 单 controller 锁 | `state_dir/controller.lock` | 同一 campaign 不会重复调度 GPU 和 agent |
| 完整对话记录 | `state_dir/transcripts/` | 每次 Pi 调用的 prompt、回复、工具调用都可审计，看板上可直接打开 |

### 接入一篇新论文

母体只放通用框架。与论文代码紧密相关的部分全部放在**目标仓库**和 `research.toml` 里：

1. **adapter 脚本**：统一封装训练/推理/评估命令，评估命令最后打印 `{"metric": ...}`。
2. **`research.toml`**：命令模板、`[experiment]` 参数（作为模板占位符）、可修改/不可修改路径、各阶段晋升阈值和超时。
3. **代码地图** `[project] codebase_map`（模板：`prompts/codebase.example.md`）：可修改的文件、关键函数和行号、评估方式、基线值，以及"代码库里没有的组件"。没有它，小模型 worker 会在工具调用上限内一直搜索代码。
4. **研究想法** `[project] idea_file`（可选）：每轮注入 scout。
5. **基线跑两次**估计噪声，据此设 `min_meaningful_delta`。
6. **验证顺序**：单元测试 → `tests/dashboard_e2e/run.sh` 点击测试 → 用本地模型跑 1 轮真实迭代（确认有指标、主仓库没被改动、花费为 0）→ 再把付费 provider 加入白名单。

接入时踩过的具体坑和修法见 `docs/LESSONS.md`。

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
- Python 3.10+ (3.10 needs `tomli`)
- Git
- `bwrap` (bubblewrap) for the agent sandbox; the controller refuses to start agents without it unless `[pi] sandbox = false`
- Pi installed and working: `pi --version`
- The target research repository should already be reproducible manually.
- For GPU scheduling, `nvidia-smi` is optional but recommended.

Harness research skills are added with repeatable `--skill` flags. Global Pi extensions are **disabled for agents by default** (`[pi] extensions = false`): interactive extensions (e.g. confirmation dialogs) block a headless RPC agent, and extra tools widen what an agent can reach. Set `extensions = true` only for extensions you have verified to be headless-safe.

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
bash tests/dashboard_e2e/run.sh   # clicks every console button in jsdom (installs jsdom into /tmp)
```

The test suite includes a fake Pi RPC process, so tests do not require real model calls. It covers sandbox containment, provider refusal, tool-call/timeout aborts, pause interrupting a hung worker, and main-repo edit detection.

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
