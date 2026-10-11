# Pi RSI Meta Harness

A reusable **RSI-oriented meta harness** for metric-driven research on top of an existing open-source repository.

The harness separates **research intelligence** from **research control**:

- **Two Pi agents do all research judgement.** One planner-tier **scout** reads the raw research record and plans each batch; one **worker** per experiment implements a hypothesis and fixes its own failures.
- The **controller** only does mechanical work: lifecycle, candidate validation, noise-based verdicts and replication, budgets, GPU/process management, sessions, stop conditions, logging, immutable-evaluator checks and human intervention. It writes no interpretation into agent prompts.
- A persistent **SQLite experiment DAG** is the long-term memory. Every agent task starts a fresh Pi session.
- A built-in **real-time terminal stream + web dashboard** shows the complete path from baseline to the current champion.

The default runtime uses only the Python standard library. Optional packages are not preinstalled; agents request them through the guarded dependency mechanism.

## 核心思想（一页读懂）

RSI 在这里指：**在固定的评估器下，让 agent 反复改进一份论文代码；判断全部交给 agent，harness 只做确定性的事。**

七条设计原则：

1. **固定目标，迭代手段。** 评估器、测试划分、研究目标不可修改（哈希校验），agent 只能改 `[files] mutable` 允许的代码。
2. **评测不可被钻空子。** 方法只能通过固定驱动（模板 `examples/fixed_driver.py`）被调用：沙箱里只看得到自己的代码、模型和原始输入，看不到标签、旧预测和分数；同一组的题目拆到不同进程，顺序打乱。scout 的 prompt 明确禁止利用标签分布、题目分组/顺序、跨题状态。实测：没有这层隔离时，一个"验证门控"把总是答"否"的那一类题全部改成"否"，凭空涨了 25 个点。
3. **相信 agent，不写死策略。** 没有 critic、reviewer、meta-RSI、配额、焦点轮换或模糊去重。scout 看到的只有原始记录（每个实验的方案 + 结果、被拒候选及原因、补丁位置），自己比较、总结、规划。
4. **长期记忆在 SQLite，不在会话里。** 每个实验是 DAG 上的一个节点：方案、父节点、git commit、补丁、各阶段指标。每个任务是新会话；worker 修自己的错误时才延续同一会话。
5. **控制器管生命周期，agent 只做判断。** 调度、GPU、超时、中断、停止、记账、噪声判定、复测全部是确定性代码；等实验时不调用 LLM。
6. **硬边界，不靠 prompt 约束。** 沙箱、工具调用上限、超时、provider 白名单都在代码里强制执行。
7. **证据优先于结论。** 先测噪声再判结论：超过单次噪声的提升自动复测；失败实验不删除，修正只追加事件。

### 一轮迭代

```text
停止检查 ── 达标 / 预算用尽 / 连续失败 / 平台期 / 不可修改文件被改
   │
① 规划    scout（planner 档）读完整实验树：每个节点 = 方案 + 结果（未运行的写原因）、被拒候选及原因、
          研究笔记提炼出的机制清单 → 一次给出整批假设（最好的在前）；harness 只做机械校验
② 实验    每个假设一条独立流水线：
   │  worktree（父实验 commit + 同步 harness 文件 + migrate 钩子）
   │  → worker 拿到目标函数源码和祖先 diff 实现 → 安全检查 → commit
   │  → 预检 → smoke（几个样本）→ quick → medium → full → 固定评估器打分
   │  → 报错就把 traceback 和出错函数源码发回同一会话，worker 自己修（最多 fix_rounds 轮）
③ 结算    噪声判定（promising / near_miss / rejected），正向结果自动复测；更新冠军
   └──────────→ 回到停止检查
```

### 失败处理

| 情况 | 处理 |
|---|---|
| 补丁预检、smoke 或运行报错 | 同一 worker 会话里修复，最多 `[agents] fix_rounds` 轮，共用一个工具/时间预算 |
| worker 超时 / 超出工具调用上限 | 判失败（NOT RUN），原因写进实验树 |
| \|Δ\| 在噪声阈值内 | 判 `near_miss`（未证实） |
| 改了主仓库 | 保存 diff，`SAFETY_STOP` |
| 连续 `max_worker_failures` 次失败 | 暂停等人 |
| scout 回答被截断 / JSON 括号缺失 | 同一会话追问一次 / 自动补全括号 |

### 安全与成本（全部可配置，默认开启）

| 机制 | 配置 | 作用 |
|---|---|---|
| bwrap 沙箱 | `[pi] sandbox` | 主机只读，只有自己的 worktree 可写；没有 bwrap 时拒绝启动 |
| 固定驱动 | `[files] immutable` | 方法在沙箱里只看得到输入；同组题目分进程 |
| 上下文精简 | `pi_extensions/rsi-context.ts` | 每次调用前删掉旧推理、把旧工具输出换成一行占位；worker 禁用 git 历史 |
| provider 白名单 | `[models] allowed_providers` | 不在名单内的 provider 在发送 prompt 前就被拒绝 |
| 单任务预算 | `[agents] max_tool_calls / *_timeout_min` | 判断类角色到上限先被要求作答，worker 超出即中止 |
| token 预算 | `[budget] max_agent_tokens` | 只计新增输入 + 输出（重发的会话前缀不重复计算） |
| 单 controller 锁 | `state_dir/controller.lock` | 同一 campaign 不会重复调度 GPU 和 agent |
| 完整对话记录 | `state_dir/transcripts/` | 每次 Pi 调用都可审计 |

### 接入一篇新论文

母体只放通用框架。与论文代码紧密相关的部分全部放在**目标仓库**和 `research.toml` 里：

1. **固定驱动**（模板 `examples/fixed_driver.py`）：评测只通过 `answer(item_input)` 调用方法，放在不可修改路径。
2. **adapter 脚本**：统一封装推理/训练/评估命令，评估命令最后打印 `{"metric": ...}`；列入 `[files] harness_runtime`。
3. **`research.toml`**：命令模板、`[experiment]` 参数、可修改/不可修改路径、各阶段阈值和超时。
4. **代码地图** `[project] codebase_map`（模板 `prompts/codebase.example.md`）：可修改的文件、关键函数、`load()` 入口、评估方式。
5. **研究笔记** `[project] idea_file`（可选）：内容变化时提炼一次成机制清单。
6. **基线跑多次**估计噪声（harness 从重复测量计算阈值）。
7. **验证顺序**：单元测试 → 驱动隔离测试（用一个故意作弊的假方法）→ 用本地模型跑 1 轮真实迭代 → 再考虑付费 provider。

接入时踩过的具体坑和修法见 `docs/LESSONS.md`。

## Architecture

```text
Human ── pause/resume/note/pin/branch/stop
                    │
                    ▼
          RSI Meta Harness Controller
  ┌───────────────────────────────────────────┐
  │ state machine / budgets / stopping        │
  │ sessions / model router / noise verdicts  │
  │ DAG memory / context builder              │
  │ GPU scheduler / process monitor           │
  │ event store / terminal / dashboard        │
  └──────────────────────┬────────────────────┘
                         │ Pi RPC (bwrap sandbox)
                         ▼
     Scout (planner tier, read-only): experiment tree
     + rejected candidates → next batch
                         │
                         ▼
     Worker per experiment: implement → run → fix (same session)
                         │
                  Git worktree + GPU
                         │
          fixed driver (sandboxed method) + scorer
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

Each agent gets one plain prompt in a fresh session (no skill wrappers, `[pi] minimal_resources`). Global Pi extensions are **disabled for agents by default** (`[pi] extensions = false`): interactive extensions (e.g. confirmation dialogs) block a headless RPC agent, and extra tools widen what an agent can reach. Set `extensions = true` only for extensions you have verified to be headless-safe; load specific ones with `[pi] extra_extensions`.

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
python controller.py steer scout "Prioritize the new data-mixture insight in the current reasoning turn"
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
planner = "local-vllm/strong-model"
worker = "local-vllm/coding-model"
planner_thinking = "high"
worker_thinking = "high"
allowed_providers = ["local-vllm"]
```

Models must already be configured in Pi. The harness calls Pi RPC `get_available_models`, `set_model`, and `set_thinking_level`; it does not store provider keys.

The planner tier runs the judgement tasks (idea digest, scout); the worker tier runs implementation and its fix rounds. Override per task with `[models].tiers`.

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

The scout may change research strategy, which branches to extend, and how to split a batch between mechanisms and tuning. Nothing may change the metric, the evaluator, the fixed driver, the test split or the safety boundary.

See `docs/SAFETY.md`.

## Pi references used by this implementation

- RPC mode: https://pi.dev/docs/latest/rpc
- RPC commands: https://pi.dev/docs/latest/rpc-commands
- JSON event stream: https://pi.dev/docs/latest/json
- CLI resources / `--extension`, `--no-skills`: https://pi.dev/docs/latest/cli
- Extensions (`context`, `tool_call` hooks used by `pi_extensions/rsi-context.ts`): https://pi.dev/docs/latest/extensions
- Sessions and context: https://pi.dev/docs/latest/sessions
