# 接入经验与修复记录

来源：第一次把母体接到真实论文代码（training-free 的视频 LLM 推理方法），并用本地 vLLM（9B 模型）跑通完整 RSI 流程。
下面只保留**通用**的问题和修法。论文相关的细节放在各自的项目仓库里。

## 1. 十条经验

1. **agent 必须有硬边界。** 小模型会无限调用工具、改主仓库、改 harness 自己的代码，prompt 写了"只改工作区"也没用。真正有效的是沙箱、工具调用上限和超时。
2. **中断必须能真正打断。** 每轮迭代要做成一个可以 cancel 的 Task。pause/stop 时同时 abort 所有 Pi 调用，并 kill 推理进程组。只改状态位不够。
3. **成本在发送前拦截。** 用 provider 白名单在发送前拒绝，不要靠事后统计 token。
4. **每次 Pi 调用都要留完整对话。** 不看对话，就判断不了 RSI 是否真的在迭代，也查不出 worker 为什么卡住。
5. **先测噪声，再下结论。** 同一配置重跑一次，指标就可能差半个点。不设噪声阈值，reviewer 会把噪声写成"机制有效/有害"，污染下一轮假设。
6. **没跑出指标的实验不复盘。** 没有指标时让 reviewer 写结论，它会编造。直接记为 `NOT RUN`。
7. **worker 需要代码地图。** 没有地图，小模型会用完全部工具调用去找不存在的模块。给出文件、函数和行号后，7–15 次工具调用就能交出补丁。
8. **假设必须落在现有代码上。** idea 里提到的组件（NLI、检索器、校准集）代码里往往没有。scout 的 prompt 要要求假设可以在工具调用上限内实现，并写明改哪个文件、哪个函数。
9. **看板按钮必须等后端确认。** 只发请求、不给反馈，用户会以为"点了没反应"。按钮要有忙碌、成功、失败三种状态；控制类按钮要一直轮询，直到状态真的切换。
10. **用端到端测试验证。** 有些 bug 只有模拟真实点击才会出现，比如刷新时把选中的纠偏目标清空、JSON 的 null 被转成字符串 `"None"`。

## 2. 问题 → 根因 → 修复

### Agent 失控与中断

| 问题 | 根因 | 修复 |
|---|---|---|
| worker 跑了几十分钟、几百次请求也不结束 | 只有总超时，没有工具调用上限 | `[agents] max_tool_calls`、`worker_timeout_min`；超出后 abort 并抛 `AgentAborted` |
| pause/stop 打断不了 worker | 主循环一直等在 `asyncio.gather` 上 | 每轮迭代是一个 Task；`_interrupt()` 依次 cancel、abort Pi、kill 进程组 |
| 超出工具上限后读取协程卡死 | stdout 读取协程里 await 了一个 RPC 命令，而这个命令的响应要靠这个协程自己去读 | 读取协程只 `create_task(abort())`，不 await |
| 上一次的中止原因影响到下一次调用 | 状态没有重置 | 每次 prompt 前清空中止原因；会话空闲时 `abort()` 直接返回 |
| 因预算被中止后又进入 repair，重复耗费 | repair 不区分失败类型 | `AgentAborted` 不进 repair |
| 报错 `Separator is not found, and chunk exceed the limit` | asyncio 默认单行上限是 64 KiB | RPC 读取的 `limit` 调到 64 MiB；子进程输出改为按块读取 |

### 安全隔离

| 问题 | 根因 | 修复 |
|---|---|---|
| worker 直接改了主仓库的文件 | agent 不在沙箱里，路径一写错就改到主仓库 | bwrap：`/` 只读，只有自己的 worktree 和 session 目录可写，`/tmp` 私有；`--unshare-pid --die-with-parent --new-session` |
| repair agent 去改 harness 自己的代码 | 同上，而且 prompt 让它"修好失败" | 沙箱 + prompt 写明"harness 的问题不要修，说明原因后停止" |
| 越界没有被及时发现 | 没有事后检查 | 每次 worker/repair 结束后检查主仓库，有改动就存 diff 到 `artifacts/escapes/` 并 `SAFETY_STOP` |
| 沙箱里 Pi 读不到配置 | `~/.pi/agent` 整个只读，Pi 写锁文件失败 | 在 `~/.pi/agent` 上铺一层 tmpfs，再把每个真实条目只读挂回去 |
| 沙箱里 agent 一调用工具就卡住 | 全局的 auto-mode 扩展弹了交互确认框，没人回应 | agent 一律加 `--no-extensions` |
| 沙箱里本地模型服务返回 502 | 主机设了 `http_proxy`，连 localhost 的请求也走了代理 | 给 agent 设 `NO_PROXY=127.0.0.1,localhost` |

### 实验正确性

| 问题 | 根因 | 修复 |
|---|---|---|
| 实验卡在模型加载后不动 | 新 worktree 里没有模型缓存，正在通过代理下载几个 GB | adapter 把 worktree 的缓存目录软链到共享缓存，并设置 `HF_HUB_OFFLINE=1`，缺文件时立刻报错 |
| 新实验看不到 harness 的修复 | worktree 是从旧的基线 commit 创建的 | `rebase-baseline`：在 HEAD 上重跑基线，结果在容差内才允许切换，并记录事件 |
| 预检报 `{python}: not found` | preflight 命令没做模板替换 | 用 `shell_template` 替换 |
| 噪声被判成提升或下降 | 不清楚噪声有多大 | 基线跑两次，设 `min_meaningful_delta`；在阈值内一律判 `near_miss` |
| 冒烟测试用的小子集一直没换 | 只有 8 道题，区分不出好坏 | quick 阶段要大到能区分出 `min_meaningful_delta` 的差异 |

### 状态一致性

| 问题 | 修复 |
|---|---|
| 进程被 kill 后，实验一直卡在 `planned/running` | 启动时、中断后、关闭时都调用 `_settle_inflight()`，统一标为 `interrupted` |
| 执行 interrupt 的任务自己先被 cancel 了 | 在 `run()` 的 finally 里再收尾一次 |
| 同一个 campaign 起了两个 controller | `state_dir/controller.lock` + `flock` |
| 有一个实验失败就整批暂停 | 只要这一批产生了指标，就继续下一轮 |
| 点"继续"后马上又暂停 | 实验成功或人工点"继续"时，把连续失败计数清零 |
| 新旧 campaign 的数据混在一起 | 默认放在 `<state_dir>/{worktrees,sessions,transcripts,logs,artifacts}` |
| 每次跑完测试都误报 `SAFETY_STOP` | 完整性检查只看 git 能看到的文件；记录哈希时整体替换旧快照；harness 自己的测试不列入不可修改文件 |
| TOML 配置项不生效 | 一个 TOML 小节会一直延续到下一个 `[...]`，新加的小节不要插在原有小节的中间 |

### 看板

| 问题 | 修复 |
|---|---|
| 点了没反应 | 统一的 `run(el, fn)`：忙碌转圈 → 成功绿色脉冲 / 失败抖动 + 提示；控制类按钮一直轮询到状态切换，12 秒内没确认就报错 |
| 实时纠偏永远发不出去 | 刷新时保留当前选中的 agent |
| `null` 被当成实验 ID | 服务端用 `str(x or "")` |
| 条件不满足时点按钮没有任何反应 | 用 `need(x, '原因')` 抛出带说明的错误提示 |
| 单测访问看板返回 502 | 测试里使用 `ProxyHandler({})` |

## 3. 常用排查

```bash
sqlite3 state/<campaign>/research.sqlite3 \
  "select id,agent_name,task_type,tool_calls,success from agent_runs order by id desc limit 10"
ls state/<campaign>/transcripts/          # 每个 session 的完整对话（每行带 agent_runs.id）
ls state/<campaign>/artifacts/escapes/    # agent 越界修改的 diff
tail -f state/<campaign>/logs/experiments/<exp>.quick.seed1.train.log
```

## 4. 已知限制

- 沙箱和主机共享网络（agent 需要访问模型服务），所以 agent 理论上可以访问其他本地服务。
- 沙箱里写不了 `safe_pip` 的 venv 和依赖请求表，需要新依赖时只能人工处理。
- 只有通过 full 阶段验证的结果才会成为冠军；只配了 quick 阶段时，得到的结论都只是代理指标。
