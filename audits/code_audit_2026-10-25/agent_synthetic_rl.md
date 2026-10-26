# 子审查：合成机制、预算调度与旧神经 RL 入口

审查对象：`kkc25/GB_RL_MoE_Loop`，提交 `f2bef422f7315f30618f00c690bf82810d1e85ac`。本报告不修改仓库源码，不使用未上传训练权重。下文的 **当前路径** 指报告实际使用的 E05–E09 合成/重放脚本；**旧入口** 指 `run_experiment.py` 暴露但不能据现有登记确认用于本轮报告的神经训练代码。两者必须分开解释。

## 1. 总判断

1. 当前 E09 NumPy 控制环境的 RLOO 期望梯度、精确梯度与概率归一化可独立复现；报告的 `5.13e-16` 恒等式误差可信。
2. `train/exact.py` 中旧 PyTorch 精确训练入口有确定错误：反传时遗漏路径概率权重。它不仅改变梯度尺度，还能把期望损失朝错误方向优化。此外，原神经枚举器在 `budget < max_calls` 时重复枚举截断后的同一终止路径，概率和会大于 1。这些必须在重启该入口前修复，但不能据此宣布本轮 NumPy E09 结果错误。
3. E07 当前只在两个端点测量之间插值，并非逐轮收益的实测轨迹。脚本对此有明确说明，值得保留。其顺序敏感性遗漏、零最优值时把 regret 设成 0、以及误称 64 请求精确分配不可计算，需要纠正。
4. E08 的不少“门禁”是独立合成例，不调用真实训练实现。它们可以验证数学机制，却不能代替 LM/控制器的集成验收。B3 中反转/随机/重复路径对照还同时改变了题目与答案，未实施固定题目下的强制错误路由干预。
5. E06 的硬整数容量求解具备正确的无重复 Top-2 和候选约束；软矩阵的可行性标签及汇总不够严格。不能把软解的问题误说成已经执行的硬容量约束失效。

## 2. 主要发现

| 编号 | 严重程度 | 确定性 | 作用范围 | 发现 |
|---|---|---|---|---|
| S01 | 高，重启旧入口前必须修 | 实际 PyTorch 入口复现 | 旧神经 `--mode exact` | 精确目标梯度漏乘路径概率，存在上升损失的更新 |
| S02 | 高，相关配置启用前必须修 | 原神经枚举器复现 | 旧神经 oracle/exact/imitation | 硬预算短于最大深度时，截断路径重复计数，概率和不为 1 |
| S03 | 中 | 原函数直接反例 | 当前 E07 评价 | oracle 最优收益为 0 时，负收益启发式被记为相对 regret 0 |
| S04 | 中 | 原函数直接反例 | 当前 E07 敏感性 | 声称离线方法顺序无关而跳过重排，实际 round-robin 余量和分块 DP 依赖顺序 |
| S05 | 中，具有直接补救方案 | 100 个小问题交叉验证及完整窗口计算 | 当前 E07 基线 | 64 请求精确前缀分配可用 O(NBH) 背包，原“不可计算”只是所选状态算法的限制 |
| S06 | 中 | 静态数据流确定 | 当前 E08 B3 | 反转/随机/重复控制改变了题目及标签，未检验同一题目下的路由顺序作用 |
| S07 | 中 | 直接数值反例 | 当前 E06 软参考与汇总 | `feasible=True` 忽略 distinct-edge/结构可行性，且汇总遗漏 P≤1 违反量 |
| S08 | 中 | 源码与上传 registry 确认 | 当前管理/结论汇总 | 顶层状态由最后更新覆盖；聚合 GPU 小时没有接到真实训练 ledger |

### S01：旧精确训练优化的不是它记录的目标

位置：[train/exact.py 第 30–33 行](https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/train/exact.py#L30-L33)。

代码反传的是：

```python
((objective.detach() * log_probability).mean() / len(paths)).backward()
```

因此其梯度为

\[
\frac1{|\mathcal T|}\sum_{\tau\in\mathcal T} F(\tau)\nabla\log\pi(\tau),
\]

而所需目标 `sum π(τ)F(τ)` 的梯度是

\[
\sum_{\tau\in\mathcal T}\pi(\tau)F(\tau)\nabla\log\pi(\tau).
\]

代码紧接着却用 `probability * objective` 记录 `expected_objective`，造成日志目标和优化目标不一致。枚举并非均匀采样；不能用除以路径数替代策略概率。

**已执行复现：**两片叶的成本为 `(1, 2)`，第二片叶的策略概率为 0.8，参数为其 Bernoulli logit。正确梯度为 `+0.16`，原训练函数产生 `−0.200000003`。原函数执行一步 lr=0.01 AdamW 后，真实期望成本由 `1.7999999523` 增至 `1.8015952110`。这不是仅影响收敛速度的尺度误差。

复现文件：`reproduce_legacy_exact.py`，结果 `legacy_exact_checks.json`。该测试调用原 `train_exact`，只在当前测试进程中用两个解析叶子替换枚举器，以隔离生产目标行；不改源码，也不冒充真实 LM 实验。

**最小修正：**对唯一、合法终止路径累加 `probability.detach() * objective.detach() * log_probability`，不再除以叶子总数；或者直接对 `sum(exp(logp) * F.detach())` 自动微分。以同一个生产训练函数做有限差分和概率质量检验。

**不能据此解释的结果：**当前报告的 E09 使用 `experiments/e09_controlled.py` 和 `data/controlled.py`，其精确梯度确实有概率权重，独立复核通过。

### S02：硬预算截断后原枚举器重复计数

位置：[eval/oracle.py 第 35–45 行](https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/eval/oracle.py#L35-L45)；`train/exact.py:20`、`train/imitation.py:21` 同样使用 `all_paths(E,max_calls)`。

枚举生成器始终枚举到 `max_calls`，但模型执行按 `budget` 提前强制退出。例如 E=2、max_calls=2、budget=1：六个名称不同的路径 `(0),(1),(0,0),(0,1),(1,0),(1,1)`，实际只执行第一步。每个终止动作被统计三次。

**原神经枚举器实测：**两层随机小 LoopMoE、E=2、max_calls=2：

| budget | 枚举行数 | 路径概率总和 |
|---|---:|---:|
| 2 | 6 | 0.9999999925 |
| 1 | 6 | 2.9999999106 |

因此即使修好 S01，预算更短的配置仍不构成正确的精确期望。多 token 或多个自适应层时，当前同一 `forced_path` 应用于所有相关位置/层，也不能自动当作完整联合轨迹树。

**最小修正：**首版对旧 exact/oracle 限定单位置、单自适应层，并枚举到 `min(budget,max_calls)`；之后若支持多层/多 token，应按真实联合状态和剩余预算生成唯一终止轨迹，而不是沿用单路径列表。现有 `tests/test_training.py:31–38` 只测 `budget == max_calls`，漏掉此条件。

### S03：零最优值时把真实损害隐藏为零 gap

位置：[experiments/e07_replay.py 第 531–544 行](https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/e07_replay.py#L531-L544)。

```python
a.append((opt - realized) / opt if opt > 1e-12 else 0.0)
```

直接调用原函数：两个请求各一轮，真实收益均为 −1，预测收益均为 +1，总额外预算为 1。oracle 不花预算，`opt=0`；启发式花一轮，`realized=-1`，真实绝对 regret=1。输出却把 equal_depth、single_step_index、two_step_package 的相对 gap 都记为 0。

这不推翻报告的“无稳定优势”方向，但会使统计量低估算法损害，尤其当继续计算普遍无益时。

**最小修正：**永远报告 `absolute_regret=opt-realized`；相对 gap 在分母为零时返回 `null` 并计入单独样本数，或报告有明确定义的其他尺度，不能赋 0 冒充完美。

### S04：离线重排敏感性不应被跳过

位置：[experiments/e07_replay.py 第 576–581 行](https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/e07_replay.py#L576-L581)、第 612–613 行。

说明声称所有 offline allocators 都对顺序不敏感，因此只重复 online ledger。原函数的直接反例：

- `equal_depth_allocation([[10],[0]],budget=1)` 得收益 10；反转两个请求后得 0。不能平均分配的预算余量按请求顺序给出。
- `rolling_window_dp_allocation([[1],[100],[2],[3]],budget=2,dp_window=2)` 得 103；改为 `[[1],[2],[100],[3]]` 得 102。分块改变了局部预算竞争。

随机分配的固定种子也不意味着同一请求在重排后获得同样的分配；argmax 平局同样需要声明 tie-break 语义。

**最小修正：**至少对 equal_depth、random、rolling DP 以及涉及平局的启发式执行相同请求重排；保持 request ID、真标签与预测得分一起移动，分别报告离线分配敏感性与真实到达顺序敏感性。

### S05：完整 64 请求精确分配可以实现

位置：[experiments/e07_replay.py 第 283–290 行](https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/e07_replay.py#L283-L290)、第 508–552 行。

原 `e07_synthetic.exact_dp` 对“每个请求已走多少轮”的联合状态记忆搜索，状态数随请求数指数增长。但当前重放的收益是固定、可分离的每请求前缀数组：对每个请求选择 k∈{0,…,H}，收益为其前 k 轮之和，成本为 k。它是多重选择背包，递推：

\[
D_i(b)=\max_{0\leq k\leq\min(H_i,b)}\{D_{i-1}(b-k)+\sum_{j=1}^{k}\Delta_{ij}\}.
\]

复杂度 O(NBH)，64 请求、H=3、B≤192 约 5 万次标量转移量级；无需复杂 bandit 求解器。

已在 `reproduce_synthetic_audit.py` 中提供 `prefix_knapsack`，与原小型 exact DP 在 100 个含正负收益的随机问题上交叉验证，最大误差 `1.78e-15`。完整 64×3、B=192 在本审查 CPU 的一次运行约 1.37 ms。该时间仅是这个 Python/NumPy 合成分配器的局部测量，不是 GPU serving 时延。

进一步读取已上传 `runs/e01/m105_100m/records_test.jsonl`，按原 E07 随机种子抽出相同 64 状态；已核对 indices 与完整 `runs/e07/m105_replay/e07_replay.json` 一致。R1 NLL=5.33535661，R4 NLL=5.32318328，39/64 个端点尾迹改善为正。沿原脚本的两种插值假设求出完整窗口知标签 oracle：

| 额外轮次总预算 | 递减插值 oracle NLL | 延迟插值 oracle NLL |
|---:|---:|---:|
| 32 | 5.30980916 | 5.31126352 |
| 64 | 5.30453126 | 5.30539446 |
| 128 | 5.30195245 | 5.30195245 |
| 192 | 5.30195245 | 5.30195245 |

这些只是在**已声明的插值环境**里的 label-informed 上界，源 JSONL 为六位小数；不是部署策略，不是新增的逐轮 LM 实测，也不能证明可学性。结果见 `e07_reconstructed_oracle.json`。

### S06：B3 的反转、随机、重复控制没有固定题目

位置：[experiments/e08_ab.py 第 976–1008 行](https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/e08_ab.py#L976-L1008)。

`rollout(seq_list,...)` 用每个 s 同时设置输入描述 `desc=list(s)` 和正确答案 `_b3_compose(s,...)`。因此把 `seq_list` 换成反转程序、随机程序或重复程序，只是测试模型能否做这些新题目，未改变“同一个输入状态选择错误专家顺序”的处理。

例如 x=1，A(x)=2x+1、B(x)=3x，原题 AB 的答案为 9；强制 BA 应输出 7，从而对原题出错。当前 reversed 测试把题目与答案一起改成 BA/7，正确输出 7 仍计成功。它不能检验“交换原题路由会否损害表现”。

**最小修正：**保留原 desc、x0 和 target，给 rollout 增加独立 `forced_action_path` 或路由干预回调，分别强制 reverse/random/repeat/early-stop；将目前指标改名为“反转程序族上的泛化”。精确专家上这个修正无需真实 LM 权重。

### S07：E06 的 soft feasible 与完整 Top-2 可行性不同

位置：[experiments/e05_e06_synthetic.py 第 509–563 行](https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/e05_e06_synthetic.py#L509-L563)、第 765–780 行。

`soft_transport` 解的是行和/列和限制的软参考，未加入 token–expert 边的 P≤1。原测试 `tests/test_e05_e06.py:194–197` 已承认此边界，这本身可以作为放松模型存在。问题是：

1. 只要总容量足够就返回 `feasible=True`，即便 Top-2 distinct/候选图根本无解。
2. 函数内部返回 `max_upper_bound_violation`，但 `run_e06` 汇总时漏掉了它。

直接反例：一个 token、两个专家、k=2、capacity=(2,2)、cost=(0,100)，soft 矩阵约 `(1.999,0)`，边界违反约 0.999，线性目标近 0；合法无重复 Top-2 的成本必为 100。如果只允许第一个专家，soft 仍报 feasible=True，而 exact min-cost flow 正确报告 infeasible。

**必须保留的正面结论：**`solve_additive_assignment` 在 token→expert 边上设置容量 1，正确防止重复选择；流量不足时拒绝不可行实例。`round_fractional` 在无法贪心补全时回退 exact flow，保护整数层的容量。不能把上例说成真实硬约束被击穿。

**最小修正：**明确名为 `relaxed_transport_solved`，同时报告 row/column/edge/forbidden-edge residuals；另用 max-flow 判定合法整数图是否可行。若要称合法 Top-2 soft 可行点，必须把 0≤P≤1 纳入求解。

另一个解释边界：`causal_sanity:632–634` 对相同 costs 重复调用同一个分配器，“suffix invariance”由函数接口不接受 suffix 保证。它是接口级确定性检查，不能验证上游 value predictor 是否已从未来 token 泄漏信息。当前 E08 E1 的两份 suffix toy 测试更有区分力；正式 LM 还需要对真实 predictor→allocator 链做前缀不变性测试。

### S08：状态/资源汇总不等于门禁验收

位置：[experiments/registry.py 第 84–92 行](https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/registry.py#L84-L92)、[analysis/goal_report.py 第 16–42 行](https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/analysis/goal_report.py#L16-L42)。

每次写某 component 时直接 `record['status']=status`；顶层状态并非从所有计划要求、实验范围、统计门槛共同推导。已上传例子：E09 的 neural_lm 为 blocked，但最后 controlled/final_scope 为 finished，顶层就 finished；E11 scale_out 为 stopped，最后 S0 profile finished，顶层仍 finished。报告披露了这些限制，但“6项完成”的数量不是科学方案全面验收计数。

GPU 账本也是手填字段求和。上传 registry 的 E00 `gpu_hours=0.0203`，仅计 C0；100M B2 的约 9.63 GPUh 未加入这一聚合字段，E01/E03/E07 等聚合字段均为 0。因此不能用该 registry 证明相对骨干的 5%/10% 上限已经执行。

**修正：**分列 `implementation_status / execution_scope / gate_status / scientific_conclusion`；从真实 immutable run ledger 自动汇总 GPU 时间及候选标签成本，保留 hash 与去重标识。metadata 缺失应是 unknown，而不是 0。

## 3. 其他旧入口的确定风险

这些尚不能解释本轮已上传 LM 结果，优先级低于主线 E00/E01/E03 审查：

- `run_experiment.py:58–64` 加载 model/controller/ledger/step/offset，但未恢复 optimizer/RNG；`train/router_rloo.py:68` 重新创建 optimizer。把 pretrain checkpoint 给 `--mode rloo` 时，还会把预训练 step 当作路由更新 step，跳过原计划开头更新。应明确区分“阶段间初始化”与“同阶段续训”。本轮 B0 GPU 迁移使用另一个 LM 入口，不能把这项旧风险移植过去。
- `train/router_rloo.py:85–91` 训练向 controller 输入 `price=lambda_cost`；`evaluate:24–31` 没有传 price，默认 0。非零成本训练时评价的是不同 price 条件。当前 p0 配置 lambda=0 不触发，但成本扫描必须修。
- `train/pretrain.py`、`train/exact.py`、`train/imitation.py`、`train/halt.py` 未统一执行 `input_token_limit`；旧 RLOO 的上限还复用前阶段 ledger。相同预算比较需独立阶段预算及总预算。
- 旧 exact/imitate/halt 没有统一的验证选点与模型保存协议；在未补齐前不适合作为正式模型间等预算比较工具。
- 旧报告辅助工具 `analysis/report.py:17,38–45` 在缺少 `math_checks.json` 时令 `math_checks={}`，随后 `math_pass = not math_checks or (...)` 把数学门禁记为通过。直接将原 `build_report` 用于完全空的临时 run 目录，实际返回 `math_gate_pass=true`、`training_records=0`、`p0_learning_gate_pass=false`、`large_scale_gate=not_assessed`。因此只能说该辅助工具的数学门禁对缺失证据错误放行，不能说它也放行了学习或扩规模门禁。修复应把缺失/空检查记为 `unknown` 或 `not_assessed`（若接口必须是布尔则不通过），仅在检查结果完整且达标时记为通过。证据为 `probe_legacy_report.py` 与 `legacy_report_missing_math_checks.json`；没有证据表明当前报告的 NumPy E09 依赖这一辅助输出，且其实际梯度已经独立复核，不能据此推翻 E09。本项作为旧辅助入口补充，不新增主 finding。

## 4. 哪些已有结果仍可相信

### 4.1 数学与实现正向核查

- E05 单玩家异步严格改进与势函数增量一致、同步振荡反例都按固定状态游戏实现。
- E06 additive exact flow 的 distinct Top-2、候选约束和硬容量被直接编码在流网络；现有小实例与 brute-force 相符。
- E07 的实际执行扣账独立于预测收益；本审查未发现该额外预算账本超支。它从已完成一轮的状态开始，结果需按其额外预算口径解释，不能误读为全模型所有层调用。
- E08 矩阵切换反例采用谱范数约束作为充分稳定化对照，而不是把单矩阵谱半径误当任意切换稳定性；此区分正确。
- E09 独立 NumPy 枚举实测：RLOO 与 exact 最大误差 `5.134781488891349e-16`；exact 梯度六坐标有限差分误差 `5.39e-12`；各任务概率和约 1，路径数为 84/84/84/84/4/20。见 `e09_gradient_recheck.json`。

### 4.2 结果解读边界

完整上传 E09 JSON 与报告数字相符：监督目标 0.18320158、exact 0.15095155、RLOO 0.27709466±0.07126624。监督的 6/6 恢复是**设计训练任务**，独立 24 例 held-out 恢复则为监督 6/24、exact 7/24、最佳训练目标 RLOO seed 7/24。后两者比监督高一个样本不具备强统计意义，但足以说明“监督在所有泛化指标更强”并未被建立。

E07 报告所引 `5.3343 vs 5.3370` 是递减插值、预算 1.5 的一个 cell；同 cell 随机额外分配为 5.32948。预算 2 时单步指数 5.33019 又差于等深度 5.32840。报告称“未证明稳定调度优势”是合适的；不应从一个 cell 推导指数优越，也不应从当前插值回放排除真实逐轮收益分配。

## 5. 本轮可复现证据与缺失材料

已跑原库测试（CPU）：E05/E06 22 项、E07 synthetic 16 项、E08 CDE 29 项，共 **67 项通过**。日志是 `tests_e05_e06.log`、`tests_e07.log`、`tests_e08_cde.log`。这组已有测试通过的同时，新反例仍成立，说明需要覆盖生产 loss、边界预算、零最优值和实验干预语义，而不只是增加同型测试数量。

无需任何未上传权重就能确认 S01–S08。要进一步复核真实 LM 的收益预测/动态停止，需要主审查列出的 backbone/student/state tensors；E07 即使拿到原权重，仍需要另外标注 T2/T3 或逐轮后状态，不能从 T1/T4 两个端点恢复这些中间真值。

## 6. 覆盖范围

详细阅读：`experiments/e05_e06_synthetic.py`、`e07_synthetic.py`、`e07_replay.py`、`e08_ab.py`、`e08_cde.py`、`e09_controlled.py`、`registry.py`；`policy/controller.py`、`policy/rollout.py`；`train/router_rloo.py`、`train/exact.py`、`train/halt.py`、`train/imitation.py`、`train/pretrain.py`；`run_experiment.py`；`eval/oracle.py`、`eval/verify_router_math.py`；`analysis/goal_report.py`、`analysis/report.py`。相关 tests 全文或关键路径阅读，E05/E06/E07/E08 CDE 全套执行。`data/controlled.py` 的概率枚举经独立复算，数据生成整体由数据审查组负责；模型、LM训练及反事实学生由其他组负责。

查阅证据：`runs/registry.json`、`runs/goal_status.json`、`runs/e09/controlled.json`、`runs/e07/m105_replay/e07_replay.json`、`runs/e01/m105_100m/records_test.jsonl`、`runs/e03/m105_100m/e03_summary.json`。对照计划为前轮 `research_bundle/experiments/E05…E09` 及公共协议，另核对仓库内原计划与用户上传本轮实验报告。

## 附录 A：最终补覆盖——因果性检查入口与受控环境

完整补读 `eval/causality.py` 及 `data/controlled.py`，并对照 `config.py`、`policy/controller.py`、`model/loop.py`、`model/transformer.py`、`tests/test_model_policy.py`、`tests/test_controlled.py`。未发现会推翻 E09 受控环境概率归一化、STOP/强制退出、预算截断、合法观测、exact/RLOO 梯度结论的新问题。`data/controlled.py:418–480` 在本例的每个预算分别枚举真实终止树，S02 的旧神经入口重计数问题不适用于它。

### A1：因果性聚合 CLI 当前不能完整生成报告（确定，中等维护优先级）

运行 `python eval/causality.py --output <审查目录>/causality_cli.json`，实际退出码为 1。原因是 [eval/causality.py 第 544 行](https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/eval/causality.py#L544) 无条件执行 `check_multi_adaptive_target_only`，该函数在第 497 行构造 `target_only=True, adaptive_layers=(1,3)`，而 [config.py 第 43–44 行](https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/config.py#L43-L44) 已正确拒绝非单个末层的 target-only 模式。错误为 `ValueError: target_only supports exactly one terminal adaptive layer`，最终 JSON 未写出。

原测试第 234–236 行断言该探针应抛出 ValueError，因此测试通过与 CLI 失败完全可以同时发生。应将聚合器中的该项改成“非法配置被正确拒绝”的显式检查，并更新过时探针说明；无需放开生产配置约束。这是验收工具的可用性问题，不是当前 LM 负结果的原因，也不是因果性被破坏的证据。

### A2：明确区分历史注释、测试 monkeypatch 与实际生产路径

`eval/causality.py:10–28` 的 Known core-file issues 已落后于实际代码。不能据此宣称当前仍有 float32 memory、多层共享 rounds 或共享采样流问题：

- 生产 `policy/controller.py:35–36` 使用 GRU 权重的 `new_zeros`，已继承 dtype。审查额外调用 `build_pair(patch_memory_dtype=False)`，确认实例没有 monkeypatch，FP64 memory 与完整 routed forward 均成功，logits 有限。默认测试 helper 中保留的 instance-level patch 是冗余旧代码，而非当前生产正确性依赖。
- `model/transformer.py:180–181` 已每层重置 previous expert 与 rounds；原测试也检查预算 4/6 可完全使用。
- `model/loop.py:38–58` 与 `model/transformer.py:218–225` 已使用前缀/episode key 的确定性随机流。补测同样的 4 个样本分别单独执行、合批执行、反序合批执行：最大调用数差均为 0，单独与合批最大 logits 差 `3.33e-16`，反序重排后最大差 0。未再现注释所述 batch 不变性缺陷。
- 原 `check_sampled_batch_invariance` 只对 batch 第一条与单独第一条做比较，覆盖较弱；增加重排及每条样本比较有价值。此处报告的是已实测通过的生产路径，而非把弱测试推断成实际错误。

这些检查的范围是 CPU、小型随机权重、Top-1、无 KV-cache 的参考路径。`check_replay` 对已有轨迹的 joint log probability 做重复与显式公式复算，支持该范围内重放值一致；它不替代真实 Top-2 LM 训练、KV-cache 或 S01 生产 exact loss 梯度验证。

可复现脚本为 `reproduce_causality_supplement.py`，完整异常栈、未打补丁 FP64 结果及重排结果保存在 `causality_supplement_checks.json`。此次补覆盖没有修改原仓库，也没有新增训练或要求权重。

## 附录 B：E07/E09 两项关键计算的独立复现入口

新增 `recheck_e07_e09.py`，默认读取同级 `GB_RL_MoE_Loop` 快照，可显式传 `--repo` 和 `--output`。仅依赖 Python、NumPy 以及包内已有 `reproduce_synthetic_audit.py` 的 `prefix_knapsack`；不需要 PyTorch、GPU、权重或重新训练。示例（从工作目录运行）：

```bash
python code_audit_2026-10-25/recheck_e07_e09.py --repo GB_RL_MoE_Loop --output code_audit_2026-10-25/e07_e09_recheck.json
```

脚本直接执行原 `data/controlled.py`，并从原 E07/E09 文件中提取未改写的纯函数 AST，以避开无关的包级 PyTorch 导入。E07 的 DP 是独立审查实现，E09 的有限差分是对原期望目标做独立中心差分；这两者不冒充仓库自有实现。

实际复算已完成：E07 的采样索引与上传运行完全一致，8 个完整窗口 oracle cell 与 `e07_reconstructed_oracle.json` 数值差为 0；E09 的 RLOO/exact 最大误差仍为 `5.134781488891349e-16`。新脚本明确固定并记录 6 个参数坐标，对 6 个任务分别有限差分，最大误差约 `7.7882e-12`，低于 `1e-7` 门槛。历史简版 `e09_gradient_recheck.json` 没记录当时所选的 6 个坐标，因此不声称新选择能逐位重现其 `5.385994601703192e-12` 误差尾数；脚本保留旧文件，输出两者门槛比较与每坐标数值。

`e07_e09_recheck.json` 同时记录原源码及输入证据的 SHA-256。E07 的含义严格限于已知、可分离、离线窗口，且中间收益采用原端点插值假设；E09 的含义严格限于当前 NumPy 受控环境，不验证旧 Torch exact 训练入口。
