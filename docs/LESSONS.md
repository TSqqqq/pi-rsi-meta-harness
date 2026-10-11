# 接入经验与修复记录

来源：把母体接到真实论文代码（training-free 的视频 LLM 推理方法），先用本地 vLLM 9B、再用 27B 跑通完整 RSI 流程。
下面只保留**通用**的问题和修法。论文相关的细节放在各自的项目仓库里。

## 1. 十条经验

1. **agent 必须有硬边界。** 小模型会无限调用工具、改主仓库、改 harness 自己的代码，prompt 写了"只改工作区"也没用。真正有效的是沙箱、工具调用上限和超时。
2. **中断必须能真正打断。** 每轮迭代要做成一个可以 cancel 的 Task。pause/stop 时同时 abort 所有 Pi 调用，并 kill 推理进程组。只改状态位不够。
3. **成本在发送前拦截。** 用 provider 白名单在发送前拒绝，不要靠事后统计 token。
4. **每次 Pi 调用都要留完整对话。** 不看对话，就判断不了 RSI 是否真的在迭代，也查不出 worker 为什么卡住。
5. **先测噪声，再下结论。** 同一配置重跑一次，指标就可能差半个点。harness 从基线的重复测量算出噪声阈值；超过单次阈值的提升自动复测，阈值内一律判 `near_miss`。
6. **没跑出指标的实验不算证据。** 记为 `NOT RUN` 并写明原因（超时、报错尾部），它说明的是实现失败，不是想法无效。
7. **worker 需要代码地图。** 没有地图，小模型会用完全部工具调用去找不存在的模块。给出文件、函数和行号后，7–15 次工具调用就能交出补丁。
8. **假设必须落在现有代码上。** idea 里提到的组件（NLI、检索器、校准集）代码里往往没有。scout 的 prompt 要要求假设可以在工具调用上限内实现，并写明改哪个文件、哪个函数。
9. **看板按钮必须等后端确认。** 只发请求、不给反馈，用户会以为"点了没反应"。按钮要有忙碌、成功、失败三种状态；控制类按钮要一直轮询，直到状态真的切换。
10. **用端到端测试验证。** 有些 bug 只有模拟真实点击才会出现，比如刷新时把选中的纠偏目标清空、JSON 的 null 被转成字符串 `"None"`。
11. **评测一定会被钻空子，必须结构性地堵死。** 旧推理循环按题对出题，方法能看到当前是"基础题"还是"幻觉题"；而幻觉题的答案全是"否"。scout 提出的"验证门控"把幻觉题的"是"全改成"否"，凭空 +25 分。prompt 禁止不够：评测改成固定驱动（`examples/fixed_driver.py`），方法在沙箱里只拿到一道题的输入，同组题目分到不同进程，顺序打乱；再写一个故意作弊的假方法做回归测试。
12. **相信强模型，删掉写死的策略。** critic、reviewer、meta-RSI、kind 配额、焦点轮换、模糊去重都是 harness 替 agent 做判断，还会重复消耗 token。保留一个 scout：给它原始记录（方案 + 结果 + 被拒原因），让它自己比较和规划；harness 只做机械校验（参数合法、父节点有代码、不重复跑同一组参数）。
13. **实现和修复是同一件事。** 独立的 repair agent 从新会话开始，worker 读过的代码全部丢失，常常把工具预算花在重新找代码上。改为同一个 worker 会话：报错时把 traceback 和出错函数源码发回去，最多 `fix_rounds` 轮。
14. **token 主要花在重发的上下文上。** 每轮工具调用都会重发整个对话。有效的办法：直接给 worker 目标函数源码和祖先 diff（不用自己找、不用翻 git 历史）；每次调用前删掉旧推理、把旧工具输出换成一行占位（`pi_extensions/rsi-context.ts`）；预算按新增输入 + 输出计。

## 2. 问题 → 根因 → 修复

### Agent 失控与中断

| 问题 | 根因 | 修复 |
|---|---|---|
| worker 跑了几十分钟、几百次请求也不结束 | 只有总超时，没有工具调用上限 | `[agents] max_tool_calls`、`worker_timeout_min`；超出后 abort 并抛 `AgentAborted` |
| pause/stop 打断不了 worker | 主循环一直等在 `asyncio.gather` 上 | 每轮迭代是一个 Task；`_interrupt()` 依次 cancel、abort Pi、kill 进程组 |
| 超出工具上限后读取协程卡死 | stdout 读取协程里 await 了一个 RPC 命令，而这个命令的响应要靠这个协程自己去读 | 读取协程只 `create_task(abort())`，不 await |
| 上一次的中止原因影响到下一次调用 | 状态没有重置 | 每次 prompt 前清空中止原因；会话空闲时 `abort()` 直接返回 |
| 因预算被中止后又进入修复，重复耗费 | 不区分失败类型 | 工具/时间预算覆盖实现和所有修复轮次；超出即判失败 |
| scout 用完工具上限后整次规划作废 | 到上限直接 abort，已读内容全部丢失 | 判断类角色到上限先 steer"现在作答"，再给 3 次调用宽限 |
| scout 回答没有 JSON | 推理太长，用完了输出上限 | 提高输出上限；没有 JSON 时在同一会话追问一次 |
| scout 的 4 个假设只解析出 1 个 | 小错误：少了一个右括号 | 解析器按栈补全缺失/多余的括号（字符串内不动） |
| 报错 `Separator is not found, and chunk exceed the limit` | asyncio 默认单行上限是 64 KiB | RPC 读取的 `limit` 调到 64 MiB；子进程输出改为按块读取 |

### 安全隔离

| 问题 | 根因 | 修复 |
|---|---|---|
| worker 直接改了主仓库的文件 | agent 不在沙箱里，路径一写错就改到主仓库 | bwrap：`/` 只读，只有自己的 worktree 和 session 目录可写，`/tmp` 私有；`--unshare-pid --die-with-parent --new-session` |
| 修复轮次去改 harness 自己的代码 | 同上，而且 prompt 让它"修好失败" | 沙箱 + prompt 写明"harness 的问题不要修，说明原因后停止" |
| 人在主仓库改 harness 时误报 `SAFETY_STOP` | 主仓库检查在 agent 运行期间比较的是全局状态 | 开着沙箱时 agent 根本写不了主仓库，不再检查；不开沙箱时只比较 agent 运行前后的状态 |
| 方法能读到题目答案、旧预测和分数 | 推理进程和评估数据在同一个文件系统里 | 固定驱动把方法放进 bwrap：只读挂载代码、模型、原始输入；家目录其余部分为空 |
| 越界没有被及时发现 | 没有事后检查 | 每次 worker/repair 结束后检查主仓库，有改动就存 diff 到 `artifacts/escapes/` 并 `SAFETY_STOP` |
| 沙箱里 Pi 读不到配置 | `~/.pi/agent` 整个只读，Pi 写锁文件失败 | 在 `~/.pi/agent` 上铺一层 tmpfs，再把每个真实条目只读挂回去 |
| 沙箱里 agent 一调用工具就卡住 | 全局的 auto-mode 扩展弹了交互确认框，没人回应 | agent 一律加 `--no-extensions` |
| 沙箱里本地模型服务返回 502 | 主机设了 `http_proxy`，连 localhost 的请求也走了代理 | 给 agent 设 `NO_PROXY=127.0.0.1,localhost` |

### 实验正确性

| 问题 | 根因 | 修复 |
|---|---|---|
| 实验卡在模型加载后不动 | 新 worktree 里没有模型缓存，正在通过代理下载几个 GB | adapter 把 worktree 的缓存目录软链到共享缓存，并设置 `HF_HUB_OFFLINE=1`，缺文件时立刻报错 |
| 新实验看不到 harness 的修复 | worktree 是从父实验的 commit 创建的 | 每个新 worktree 把不可修改文件和 `[files] harness_runtime` 同步到基线 commit；接口变了用 `[commands] migrate` 钩子迁移继承来的代码；`rebase-baseline` 记录基线切换（`--no-new-run` 表示只改了 harness 文件） |
| 输出压缩扩展让 worker 的 edit 全部失败 | logzip 改写了通过 bash 显示的源码（缩进/换行） | 改写文本的压缩器只给只读角色（`readonly_extra_extensions`） |
| 改参数默认值的实验其实没有生效 | adapter 总是通过命令行传入参数，覆盖了代码里的默认值 | 调参由 harness 通过命令行传入（tuning 实验不需要 worker）；只改数字的补丁直接拒绝 |
| 每个实验都留下一个 Pi 进程 | worker 用完从不关闭 | 实验结束后释放该 worker 的 Pi 进程 |
| 预检报 `{python}: not found` | preflight 命令没做模板替换 | 用 `shell_template` 替换 |
| 噪声被判成提升或下降 | 不清楚噪声有多大 | 从基线的重复测量计算噪声阈值；正向结果自动复测到 `quick_replicates` 次 |
| token 预算提前耗尽 | 把每轮重发的会话前缀都算作新输入（实际是 36 倍高估） | 按每次调用的新增输入 + 输出计（`fresh_input_tokens`） |
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
- 固定驱动需要 bwrap；同组题目分进程依次运行，评测时间约为原来的两倍。
- 判断类角色（scout）开 high 思考时，单次规划要十几分钟；规划期间实验 GPU 是空闲的。
