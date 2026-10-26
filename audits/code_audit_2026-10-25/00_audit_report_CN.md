# GB_RL_MoE_Loop：代码落实、实验有效性与研究主线审计

**审计日期：2026-10-25**  
**固定提交：`f2bef422f7315f30618f00c690bf82810d1e85ac`**  
**仓库：[kkc25/GB_RL_MoE_Loop](https://github.com/kkc25/GB_RL_MoE_Loop/tree/f2bef422f7315f30618f00c690bf82810d1e85ac)**

## 1. 总体判断

**现有证据还不足以判定研究路线失败，也不足以确认“动态停止已获得有用的自适应计算收益”。** 基础 Looped MoE 和默认反事实标注的主要计算语义基本正确；主要问题发生在控制器训练与部署的接口、状态抽样与统计单位、基线配平、训练协议、实验门禁解释，以及少数确定的公式实现错误。

最影响当前决策的五点是：

1. **E11 没有部署 E03 校准的同一个停止策略。** 学生训练和离线校准输入是 raw hidden state，在线却变成 RMSNorm 后的状态。
2. **E01 的“1024 个测试状态”只有 3 个文档，验证集的 1024 个状态几乎来自同一个文档。** 这不足以将某个 MLP 的负收益解释为“当前可观测状态没有可用信息”。
3. **固定 R1 已经落在报告的 0.005 nat 容忍范围内。** 因而“比 R4 少算且均值变化小”尚不能证明按状态分配计算具有必要性。
4. **E12 的专家独立 adapter 首层实际先把所有专家输入相加。** 它与声明的逐专家计算不同；候选空间和优化设置也没有完全匹配。
5. **S67 的 dense 对照没有按 S67 配平，LM 逻辑 batch 也没有达到方案的起点。** 这使“方案已完整执行，结果不佳所以关闭路线”的推断过强。

建议保留目前的所有原始结果与失败记录，将状态改为“工程 smoke / 单骨干 pilot / 未决诊断”，优先修复测量和采样，再复用已训练的 M105 检查点做有界诊断。现在没有理由立即扩大到 300M–1B token、末四层循环或真实 LM RL；同样没有充分理由永久关闭价值监督和有序互补性。

本文不把“代码存在缺陷”直接等同于“它解释了全部负结果”。确定的算法身份错误可以不依赖权重验证；这些错误对已训练模型的具体 NLL 影响，需要下一轮权重才能量化。

## 2. 范围、证据和执行方式

已获取并核验本提交全部 **390 个受版本管理文件**，逐文件 Git blob SHA 匹配，无缺失、无不匹配。包括生产代码、82 个非 plans 目录 Python 文件（含 22 个测试文件）、21 份配置，以及全部已上传 JSON、JSONL、日志、数据 manifest 和 tokenizer。原仓库保持不变；修复建议以独立 patch 交付。核验见 [snapshot_integrity.json](snapshot_integrity.json) 与 [repository_snapshot.json](repository_snapshot.json)。

审查对照包括仓库内研究方案、前一轮逐实验计划及公共协议、实验报告、实际 resolved config、执行日志、指标与逐状态成本记录。用户附件报告与仓库报告仅有行尾空格差异，不存在实质内容版本冲突。

证据分三层：

| 层级 | 本轮实际完成 | 能支持什么 |
|---|---|---|
| 源码与公式审查 | 对训练、执行、数据、评测、登记路径及配置进行交叉检查 | 判定实现语义、候选空间、目标函数、边界条件 |
| 已上传记录复算 | 文档覆盖、成本方差、廉价策略基线、参数/FLOPs、离线 DP 等 | 判定当前证据规模及记录内可验证的数值关系 |
| 原代码 CPU 实跑 | 全部原测试、微型真实 PyTorch 反例、受控梯度核验、补丁回归 | 判定具体代码问题是否真实触发 |

**原测试：424 passed，83.15 秒。** 原实验环境记录为 PyTorch 2.12.1+cu130；审查使用 Python 3.12.14、PyTorch 2.5.1+cpu。没有使用 A6000 重训、没有重跑真实权重 NLL，也没有把 CPU 用时当成 GPU 性能。原测试日志：[existing_tests.log](existing_tests.log)。

测试全过与下文问题并不矛盾：例如动态停止原测试将学生权重归零并用无穷阈值，无法检查有限阈值下的特征契约；E12 原测试主要检查 shape、合法动作和可重复性，没有逐专家独立性断言。

## 3. 核心模型有哪些部分确实落实了

以下部分可以保留为可靠的工程基础：

- 12 层、隐藏维度 512、8 专家、Top-2、SwiGLU、绑定 embedding/head、RoPE/RMSNorm。
- attention 每块执行一次；默认仅末层 FFN 循环，前 11 层为 R1；没有把 attention 隐蔽地重复算四遍。
- dispatch 按选中专家的 token 分组计算，确实执行两个专家；没有先算全部 8 个再做 mask。
- 预训练 `Top2MoE(..., freeze=False)` 允许 gate 与专家参数得到梯度。本轮微型真实训练中二者梯度均非零。
- 参数实际实例化计数：S67 为 **67,170,816**，M105 为 **104,919,552**，与选定结构一致。
- 文档内 causal attention、shifted next-token labels、跨文档与 padding 的监督 mask 基本正确。新的 recovery 格式不把答案后缀作为查询输入。
- E01 默认全部 28 个合法 Top-2 动作包含 baseline；首步强制候选动作，后续按冻结基准策略续算到相同深度；候选混合权重来自冻结 router logits 的重新归一化；学生只读合法状态，标签知情 oracle 单独列示。
- 当前末层干预之后没有后续跨位置 attention，因而从每个动作真实执行后的标签表按列取成本，是该限定场景下有效的离线政策评估方式。

来源：[模型前向][src-model]、[Top-2 执行器][src-top2]、[默认反事实路径][src-cf]。本轮实例计数与梯度证据见 [model_training_torch_checks.json](model_training_torch_checks.json)。

## 4. 主线问题：停止策略、反事实学习与证据充分性

### F01｜确定错误：停止学生的训练—部署特征不一致

`train/value_student.py:35–37` 从缓存读取 raw pre-FFN `h`；E03 也用 raw `h` 校准。`eval/dynamic_halting.py:26–31` 却将 `block.ffn_norm(h)` 输入同一学生，并继续使用 raw-h 的 mean/std。因此训练与校准的函数是

\[
f\big((h-\mu)/\sigma\big),
\]

而部署的是

\[
f\big((\operatorname{RMSNorm}(h)-\mu)/\sigma\big).
\]

二者一般不相等，校准阈值也不能直接迁移。router 使用归一化状态是正确的；需要修的是学生输入，不应把 router 一起改为 raw h。[训练输入][src-student]；[部署输入][src-halting]。

真实原代码微型复现中，按训练契约 7 个 token 都应 R1，原在线执行器却让其中一个 token 走到 R4，最大 logit 差为 **0.00419546**。这是证明错误存在的构造例，不是用户检查点效果估计。[复现记录](repro_counterfactual_torch_results.json)。

**影响范围：**E11 不能被称为对 E03 同一冻结策略的真实前向验证。E03 的离线表不因这个部署错误自动失效，但仍受下述样本与对照问题限制。补丁恢复 raw-h 契约；修复后应使用现有学生与阈值先做同状态逐 token 对照，不需要先重训学生。

### F02｜证据不足：状态数增加没有增加足够独立文档

`gather_states` 随机访问 sequence，然后提取该序列几乎全部合法位置，最后按 `n_states` 截断。它没有文档均衡、每文档上限或随机位置抽样。[采样代码][src-cf-sampling]。

从已上传逐状态 JSONL 复算：

| 运行 | 分区 | 状态数 | 文档数 | 主要文档占用 |
|---|---|---:|---:|---|
| 初始 pilot | train / val / test | 1024 / 256 / 256 | 2 / 1 / 1 | 训练为 1023+1 |
| coverage_remediation | train | 8192 | 15 | 9 个序列，多个长文档各约 1023 状态 |
| coverage_remediation | validation | 1024 | 2 | **1023+1** |
| coverage_remediation | test | 1024 | 3 | **906+116+2** |
| m105_100m | train / val / test | 8192 / 1024 / 1024 | 15 / 2 / 3 | 与上组使用同一批 val/test 文档与位置 |

不能将 1024 个相邻 token 当作 1024 个独立泛化样本。当前代码以 token 数计算普通 SE，未实现原定的文档级配对 bootstrap。只有 3 个测试文档时，即使补上 bootstrap 也不会凭空获得可靠的总体代表性。[统计代码][src-cf-metrics]；[复算脚本](records_only_counterfactual_audit.py)；[完整覆盖与按文档结果](records_only_counterfactual_audit.json)。

当前“路径价值不可有效学习”应收缩为：**该骨干、该样本池、该学生目标和训练配置下，没有观察到部署收益。** 它支持暂停当前配置，不支持排除整个可学性路线。

### F03｜协议偏离：独立探针训练、模型选择、校准分区没有落实

FineWeb 管线只有 train/validation/test 三份；LM 与 E01 读取同一目录的同名分区。M105 的 97,657 个训练序列对应 97,657 次 batch=1 更新，E01 的 train 状态因此来自已经用于骨干训练的文档池。E01 用 validation 选学生，E03 又用同一 validation 选阈值，缺少独立 calibration。[数据分区][src-fineweb]；[LM 入口][src-runlm]；[E01 入口][src-rune01]。

应保持边界准确：这不等于测试标签进入训练；公开记录的三个 split 文档 ID 交集为零。骨干验证仅用前 16 个序列，而 E01 validation 抽到序列 3306/2322，因此不能声称这些具体 token 已用于骨干选点。确定的问题是独立角色未按协议冻结、学生选择与阈值校准复用了同一小验证集。

原方案的六分区是逻辑角色；controller_validation 内再留 calibration。修复应按文档与近重复组建立角色清单，而非仅把现有 token 行随机分成更多文件。

### F04｜新诊断：Q 标签绝大多数方差来自共同 token 难度

写每个状态的 28 动作成本为

\[
Q_{ia}=\bar Q_i+\widetilde Q_{ia},\quad
\bar Q_i=\frac1{28}\sum_aQ_{ia},\quad \sum_a\widetilde Q_{ia}=0.
\]

则把所有状态—动作 cell 等权展开后，成本方差可分解为状态均值方差与动作中心化二阶矩。对 m105 公开记录，复算得到：

| 分区 | \(\mathrm{Var}(\bar Q_i)\) | \(\mathbb E\widetilde Q_{ia}^{2}\) | 比值 |
|---|---:|---:|---:|
| train | 13.937472 | 0.0220776 | 631.29 |
| test | 13.990548 | 0.0180868 | 773.52 |

test 的共同难度部分占总方差约 **99.8709%**。这是标签方差分解，不是学生残差 MSE、梯度或已学内容的分解；后者需要实际预测才能计算。当前学生用绝对成本 MSE 训练并按 validation MSE 选点，因此总体 MSE 的改善可以主要来自“这个 token 难不难”，并不保证专家动作排序更好。[学生目标][src-student]；[本轮分解](records_only_counterfactual_audit.json)。

另做两个预先定义的低成本规则，**只用 train 标签拟合动作，test 不参与选择**：

| m105 test 策略 | 相对 baseline 平均收益，nat/token |
|---|---:|
| 固定采用 train mean Q 最低的 action 20 | −0.00004936 |
| 按原 router 的动作分组，各组采用 train mean Q 最低动作 | −0.00015579 |
| 原即时成本 MLP | −0.00384269 |
| 原长期成本 MLP | −0.00677045 |

这些廉价规则仍没有正收益，不能当新方法成功；但原 MLP 连这些近似保持 baseline 的规则都没有超过。合理的下一步是检查拟合与样本外决策，不是直接将失败归于状态信息不足。JSONL 成本保留 6 位小数，单个差值的量化误差上限约 1e−6；上述新结果属于记录内探索性分析。

建议并列比较 raw Q、动作中心化 Q、相对 baseline 的 advantage，按配对决策 regret 和部署收益选点。对动作选择，减去同状态、所有动作共用的常数不改变 argmin。**这个等价性不能直接用于停止：不能分别中心化 immediate 和 continuation 两个头后再相减。** 停止应直接监督配对的 `loss(T1)−loss(T4)`，或明确保留两个头之间的共同 offset。

### F05｜研究解释：大 oracle gap 不是可部署收益证明

令 \(Z\) 为控制器可见状态，包含确定 baseline 动作所需信息，\(a_0=a_0(Z)\)；\(Y\) 为正确下一 token，\(C(a,Y)\) 为候选终局损失。事后 oracle 计算的是

\[
\mathbb E[C(a_0,Y)-\min_a C(a,Y)],
\]

部署策略能利用的则取决于

\[
\mathbb E[C(a_0,Y)-\min_a\mathbb E(C(a,Y)\mid Z)].
\]

一般有 \(\mathbb E[\min_a C(a,Y)\mid Z]\leq\min_a\mathbb E[C(a,Y)\mid Z]\)。因此 E01 的 **0.1095425 nat** 标签知情 gap 既可能含可预测的动作差异，也可能含知道答案后才能挑出的差异。当前 MLP 负收益同样不能证明后一种解释成立。要判断可部署信号，必须补足文档覆盖、廉价强基线、学习曲线、同输入同容量对照和独立确认。[原 E01 数值][src-e01-result]。

### F06｜正结果的边界：固定 R1 已符合描述性容忍

在 E03 的相同 1024 个 test 状态上：

| 策略 | 平均末层轮数 | 相对固定 T4 的 NLL 变化 |
|---|---:|---:|
| 固定 T1 | 1.00000 | +0.00260160 |
| value student，validation 目标预算 2 | 2.37695 | +0.00099853 |
| 固定 T4 | 4.00000 | 0 |

两者均值都小于 0.005；固定 T1 不需要停止学生。E11 另一份 16,342-token 验证上，R1−R4 仅 **+0.000573586**，dynamic−R4 为 **+0.000074070**。这首先提示 R4 的额外循环在该骨干与评测集上平均收益很小。[E03 结果][src-e03-result]；[E11 结果][src-e11-result]。

这不是说 dynamic 没有潜在价值；应检验它是否在独立文档上超过固定 R1/R2/R4 与相同平均成本的无条件深度混合，并把两个学生网络的推理开销计入。E01 的实际 hidden_dim 没有完整写入已上传 CLI 元数据，不能把默认 1024 当作已证实的运行值。

报告中的 `nll_within_tolerance` 只是均值比较，不能自动推出统计非劣结论，后续需预先约定检验方式。当前 token-IID SE 不适合支持该检验；也不应事后任意选 1.645 或 1.96 来宣布成功或失败。

预算 2 是 validation 平均预算目标；test 实际 2.37695 是分布迁移下的实现值。**这不是硬预算扣账错误。** 应按实际成本作曲线，若部署需要硬上限，再添加合法调度约束。

### F07｜确定测量错误：固定与动态计时工作量不同

固定路径调用 `model(input_ids, labels, ...)`，包含完整 float32 cross-entropy；dynamic 路径只计算 logits。真实 `profile` 复现中，每模式 1 次 warmup+2 次测量，固定 R1/R4 各执行 CE 3 次，dynamic 执行 0 次。即使换成独占 GPU，这个比较仍不公平。[计时代码][src-profile]；[CE 所在前向][src-model]；[复现](model_training_torch_checks.json)。

修复应统一为 input→logits，质量计算放在计时外；每个模式独立重置 peak-memory stats。当前三个模式相同的峰值显存可能只是累计峰值，不能当逐模式测量。

`evaluate_dynamic` 还把全部执行位置的 rounds 相加，除以监督 token 数；16×1024=16384 个执行位置与 16342 个监督位置不等。因此应分开报告监督位置平均深度与全部实际执行位置的成本，不能混称 `mean_rounds`。

## 5. 基础训练、公平对照与成本账本

### F08｜S67 dense 配平指向了不同规模

按真实实例参数与配置复算，FFN 矩阵乘 FLOPs（每 token；一次乘加计 2 FLOPs，不含 attention/head/router）如下：

| 配置 | 参数量 | R1 FFN FLOPs | R4 FFN FLOPs |
|---|---:|---:|---:|
| S67 MoE，m=256 | 67,170,816 | 18,874,368 | 23,592,960 |
| dense B3，m=1024 | 48,247,296 | 37,748,736 | 37,748,736 |
| dense B4，m=4096 | 104,870,400 | 150,994,944 | 150,994,944 |
| recurrent dense B5，m=1024 | 48,247,808 | 37,748,736 | 47,185,920 |
| M105 MoE，m=512 | 104,919,552 | 37,748,736 | 47,185,920 |

B3 的 FFN 计算是 S67 R1 的 2 倍；B4 比 S67 多约 56% 参数。它们可作为某些 M105 比较的起点，例如 B3 匹配 M105 R1 的 FFN 矩阵量，B5 在同深度匹配 M105 的 FFN 矩阵量；不能据此说它们完全无用。但它们不构成报告所称的 S67 成本/参数匹配对照，也不同时匹配 M105 R4、随机深度和完整 wall clock。[E00 配置][src-configs]；[精确算表](model_training_arithmetic.json)。

5M token smoke 的性能记录可以保留；不能用这个配平表推导架构优劣。M105 目前只有 B2 的 100M 单 seed pilot，没有同规模完整对照。

### F09｜逻辑 batch 明显小于方案，深度“random”实际为周期

`train/lm.py` 每次一个 batch 直接 backward/step，没有梯度累积。上传配置都是 batch=1、长度 1024；M105 平均约 **1023.992 输入 token/update、1022.018 监督 token/update**，而原协议的 pilot 起点是 8192 有效 token、主配置 32768。[训练循环][src-lmtrain]；[原共同协议][src-plan-common]。

“每卡 microbatch=1”不等于“逻辑全局 batch=1 条”。这一差别改变优化噪声、学习率的适用范围、更新次数及训练成本，但本轮不能证明它是负结果唯一原因。

`depth_schedule == 'random'` 使用 `1+(update+seed)%max_calls`，是周期覆盖 1–4，边际频率均匀但并非独立随机抽样。周期策略可以作为明示设计，不能以名称代替真实抽样协议。若恢复随机抽样，必须同时恢复 RNG 状态。

### F10｜恢复与账本存在可触发错误

1. checkpoint 在本段 elapsed/GPU-hours 累加前保存；恢复读取旧 ledger，上一段耗时丢失。真实微型训练注入 7 秒与 5 秒两段时钟：第一段 checkpoint 为 0，第二段报告累计 5，而应为 12。
2. `best_nll` 每次进入训练重置为 infinity；恢复后较差验证值可以覆盖历史最佳。受控例中历史最佳 0.9 被恢复后的 2.0 覆盖。
3. 保存 torch/cuda RNG 却未恢复；当前无 dropout 等条件下，不据此断言已有权重必然漂移，但恢复契约不完整。
4. checkpoint 缺少强制的数据内容/tokenizer hash 校验；改变同 shape 数据或 batch 可以静默继续。
5. 学习率按本次 `run.updates` 重算，smoke100→4883 的继续训练发生 schedule 跳变，不能当原 schedule 无缝续接。

上传日志确实有迁移与恢复；B0 初始 100 步的 140.068 秒未包含在最终段 ledger，回滚重算也没有独立 lost-compute 项。时间戳间隔不能直接当精确 GPUh；确定的是存盘顺序与恢复字段不满足累积账本语义。[训练存盘与恢复][src-lmtrain]；[恢复任务同类路径][src-recoverytrain]；[受控证据](model_training_torch_checks.json)。

另外，`expert_calls` 实际累加每 token 的 FFN rounds。Top-2 的单专家-token对应乘 2：M105 记录 **1,350,009,856**，实际同口径为 **2,700,019,712**；S67 B0 为 60,002,304→120,004,608。保留 rounds 与 expert-token-pairs 两列，再按宽度计算 FLOPs。E01 的字段名 `candidate_pair_calls` 本来就是专家对调用，不能误判它也少算一半。

### F11｜采样器重复构造导致不必要的索引开销

每个 update 都新建 sampler 并取第一个剩余 batch；sampler 每次构造、洗牌全部 N 个索引及 batch 列表，使每 epoch 的索引工作由 O(N) 变成约 O(N²/batch)。N=97,657 的真实 sampler 在本机前 12 步复现中，每个 update 重建的 12 步合计约 0.505 秒，复用 iterator 的 12 步合计约 0.036 秒，序列顺序完全相同。这是 CPU 局部开销示例，不外推为训练总加速比。[sampler][src-packed]；[算术复现](model_training_arithmetic.json)。

修复只需每 epoch 构造一次并保存位置；不必为解决这个问题改动数据分布。

## 6. 其他实验的确定问题与可保留部分

### F12｜E12：专家 adapter 首层合并专家输入，且对照未匹配

原代码为：

```python
torch.einsum("...ei,oih->...oh", rows, self.w1)
```

输入专家轴 e 被求和，计算的是每个输出专家 o 对 `sum_e rows_e` 的变换。独立专家应为：

```python
torch.einsum("...oi,oih->...oh", rows, self.w1)
```

两个专家输入 2、7，各自系数 3、5，原结果为 **27、45**，应为 **6、35**。`_apply` 方法还覆盖了 `nn.Module._apply`，使 `.to()`/`.double()`报错。[E12 实现][src-e12]；[真实复现](repro_counterfactual_torch_results.json)。

需要区分模式：无 summary 时相同输入被乘 E，可用权重重参数化吸收，因此不能说函数类完全不能学习；当前 **learned summary** 模式只保留所有摘要的总和，失去专家各自摘要的对应关系；零均值 checkpoint summary 恰好相消为零，是另一个可触发但非本次已执行的模式。当前 CPU 主运行未调用 `.to()`，不能把可移植性报错说成已有 run 崩溃原因。

中央 MLP 部署选全部 28 动作的最小值；报价给定 anchor 后只有 7 个候选。虽然同 cell MSE 对照存在，部署候选空间仍不同。中央学生 weight decay=0，报价 AdamW 未显式传该值时使用 0.01。修复应让中央对照使用相同 anchor 和候选表，另列 unrestricted 中央模型；明确 optimizer 与所有额外参数/推理成本。改正首层后应重新训练报价，不能把旧权重直接装入新公式就称为公平修正结果。

线性专家报价等价于中央线性 router 的理论判断仍成立；非线性报价“已被排除”的证据应撤回为未决。

### F13｜E10：单文档小样本与符号翻转定义不足以排除互补性

train/validation/test 仅 **32/16/16 个状态，各自只有 1 个文档**；约 14.36 万参数头训练 20 个 epoch，每 epoch 一 batch，即 20 次更新。它足以验证 28² 两步成本矩阵能生成，不能支持可靠的模型容量与可学性比较。[E10 实现][src-e10]；[本轮 E10 覆盖与定义敏感性复算](e10_records_recheck.json)。

有序路径枚举、固定参考四项交互与冻结续算主要公式基本正确。但 sign-flip 用“至少一个 >0 且至少一个不 >0”，将正→零也计为正负翻转，没有数值零带。应改成存在 `>eps` 和 `<−eps`，并单列 uncertain/zero。对公开六位小数矩阵用 eps=1e−6 重算约 0.41837，而原指标为 0.5；前者不能冒充修正后的全精度结果。精确复核只需缓存 pair_q，不必重新上传整个 LM。

### F14｜E07：有一条便宜、可直接补足的完整离线 oracle 路线

当前 64 请求回放的 T2/T3 由 T1/T4 端点插值，不是实测逐轮 LM 收益；报告对此已有明确披露。硬额外预算账本本身没有发现超支。[E07 回放][src-e07]。

仍有三项应修：

- oracle 最优收益为 0 时把相对 gap 记为 0，会掩盖算法负收益。直接反例：oracle=0、启发式=−1，真实绝对 regret=1，代码输出 relative gap=0。应始终报告绝对 regret，零分母的相对指标为 null。
- “所有 offline 分配器对顺序不敏感”不成立。等深度余数按顺序分配；rolling window DP 的分块也随顺序改变。已经构造并执行 10→0、103→102 的收益反例。
- 当前可分离、已知固定窗口的完整 oracle 不必用指数级联合状态枚举。设每请求选择 k 轮，其前缀收益为 \(G_i(k)\)，可用

\[
D_i(b)=\max_{0\leq k\leq\min(H_i,b)}\{D_{i-1}(b-k)+G_i(k)\},
\]

使用精确花费状态时，初始化 D₀(0)=0、D₀(b>0)=−∞，最终取 max_{b≤B} D_N(b)，允许不花完预算。复杂度 O(NBH)。本轮提供 `prefix_knapsack`，对 100 个含正负收益的小例与原 exact DP 交叉验证，最大误差 **1.78e−15**。它只解决这个离线、可分离、预算可留空的问题，不替代未知未来请求的在线服务调度。

按原 E07 seed 与上传记录重建同一 64 状态，额外预算 B=64 时，递减/延迟插值的标签知情完整 oracle NLL 分别 **5.30453126 / 5.30539446**。这些是插值环境中的收益上界所对应的 NLL 下界，不能作为新 LM 结果或可部署优势。[复算记录](e07_reconstructed_oracle.json)；[DP 与反例](reproduce_synthetic_audit.py)。

### F15｜E06/E08：算法核心与验证语义要分开

E06 的整数 min-cost flow 对 token→expert 边容量设为 1，正确实现 distinct Top-2 和硬容量。软 transport 是只约束/拟合行列边际的数值放松，未保证每条边 P≤1，却返回 `feasible=True`，汇总还漏掉 upper-bound violation。单 token、k=2、capacity=(2,2)、cost=(0,100) 的软分配可近似为 (1.999,0)，合法硬解成本必为 100；这说明软参考与 Top-2 可行域不同，**不说明硬 flow 违反容量**。应分别报告放松求解与合法整数可行性。[E05/E06][src-e0506]。

E08 B3 的 reverse/random/repeat 同时改程序描述和答案，检验的是另一程序族上的泛化，而非固定原题强制错误专家路径。若要因果判断顺序贡献，应保留 input/target，只干预 action path。当前势函数、硬容量、稳定矩阵切换等合成验证仍有其限定价值。[E08 B3][src-e08]。

### F16｜E09：当前受控概率与梯度核验通过；旧 Torch exact 入口确有错误

当前报告使用的 NumPy controlled E09，经独立复算：RLOO 期望梯度与 exact 最大误差 **5.13e−16**，6 个显式记录坐标的有限差分误差约 **7.79e−12**，每个任务的终止路径概率总和约 1。这个数学验证可以保留。[当前受控实现][src-e09]；[可独立复现的梯度记录](e07_e09_recheck.json)。

监督 6/6 是设计训练任务；held-out 24 例为监督 6/24、exact 7/24、按训练目标选出的 RLOO seed 7/24。因此“本受控训练目标上监督强于当前 RLOO 配置”可以保留，“监督在所有泛化指标都更好”不能成立。真实 LM RL 未运行，不能从这些玩具任务直接判定它无效。

另一个旧神经入口 `train/exact.py` 反传 \(\sum F\nabla\log\pi/|\mathcal T|\)，缺少枚举轨迹概率权重。真实训练函数的两叶构造例中，正确梯度 +0.16，代码 −0.2，走一步后真实期望成本 1.8→1.801595。旧 `eval/oracle.py` 在 budget<max_calls 时还把已截断路径重复计数；E=2、max=2、budget=1 的概率总和为 3。[旧 exact][src-oldexact]；[旧枚举][src-oldoracle]；[反例](legacy_exact_checks.json)。

这两个错误必须在重新开放旧神经 RL 路线前修复，但它们**不解释当前 NumPy E09 或 E01 的负结果**。

### F17｜其他确定入口问题与状态汇总问题

- `run_recovery --mode evaluate --resume ...` 没有加载 checkpoint；加载仅在 train 分支发生。传不存在的 checkpoint 仍以退出码 0 评估随机模型，本轮已实跑。当前上传 C0/C1 来自 train 路径，不能把此错误当作 C1 泛化差的解释。[recovery 入口][src-runrecovery]；[探针结果目录](recovery_evaluate_probe/)。
- `eval/causality.py` 总入口无条件调用已被 config 禁止的 multi-adaptive target-only 探针，导致退出码 1。正确修复是捕获并记录“非法配置被正确拒绝”，不应放开生产约束。头部声称的 memory dtype、共享 rounds、采样 batch 不变性问题已被生产代码修复；本轮关闭 helper monkeypatch 后的 FP64 与样本重排检查通过，不能依过时注释误报。[因果性入口][src-causality]；[复算](causality_supplement_checks.json)。
- `registry` 顶层 status 被最后一次 component 更新覆盖。E09 neural blocked 与 controlled finished 可以同时存在，顶层却显示 finished；E11 scale-out stopped 与 S0 finished 同理。完成数量不能当成所有科学门槛已通过。[登记器][src-registry]。
- registry 的 E00 GPUh 仅约 0.0203（C0），没有包含报告的 M105 B2 约 9.63 GPUh；多项聚合字段为 0。实际发生却未记录的 GPU 计算应记 unknown；纯 CPU 分支 GPUh=0 则合理。不能据此证明 5%/10% 标注或增量预算约束已落实。[上传登记表][src-registry-result]。
- 旧 `analysis/report.py` 在缺失 `math_checks.json` 时仍输出 `math_gate_pass=True`；原 `build_report` 对空目录的实跑已确认。应记为未评估，而非数学验收通过。其学习/扩规模字段并未同时通过；当前 NumPy E09 的独立梯度核验不受这项旧辅助工具错误影响。[复现记录](legacy_report_missing_math_checks.json)。
- 自定义候选表接口接受 pairs 却执行全局 `arange(n_actions)`，会将局部列 ID 当全局动作 ID。默认完整 28 动作未触发；改成节约标注的 4 候选前必须修映射并保证 baseline 在表内。[反事实扩展接口][src-cf]。

## 7. 对现有实验报告应怎样改写

| 原判断 | 建议保留/修改为 |
|---|---|
| E00 完成，67M B0–B5 比较 | E00 工程 smoke 与单 M105 pilot 已完成；配平修复及正式对照待做 |
| 当前可观测状态不足以可靠预测路径 | 当前小文档池和学生设置下未获得部署收益，尚不能定位为信息不足 |
| 动态停止显著减少调用并保持质量 | 离线二元停止在小样本上有描述性均值结果；需固定 R1 对照、文档统计和部署契约修复 |
| 计时只因共享 GPU 不能解释 | 还存在固定路径多算 CE 的系统性不对称，修复后再测 |
| E10/E12 因证据停止 | 当前配置可以暂停，但不能外推机制无效；E10 样本与指标不足，E12 公式与公平性须修复 |
| E09 完成可控部分，登记顶层 finished | NumPy 受控对照已完成；顶层状态需保留范围，真实 LM RL 未执行，旧入口不能直接启用 |
| 44 门禁全部通过 | 已配置的合成检查通过；不等于所有预定因果干预语义与真实 LM 验收已覆盖 |

原报告已经主动披露了 300M/规模扩展未做、E07 插值、E03 二元深度、共享 GPU 和未跑真实 LM RL，这些诚实边界应保留。本审计修改的是结论力度和具体接口问题，不将尚未运行的扩展指认为虚假结果。[原实验报告][src-report]。

## 8. 推荐的研究主线与修复次序

**第一阶段：恢复实验可解释性。** 先修 F01/F07/F12 及账本、动作映射、评估加载入口；原测试保持通过，新增测试必须能在原实现失败、修复后通过。现交付的最小 patch 仅修 raw-h 契约、E12 einsum 与 `_apply` 命名，3 项定向回归在原实现为 2 失败+1 报错，在内存应用补丁后为 3/3 通过；其他问题没有冒充已修复。

**第二阶段：复用 M105 检查点回答两个独立问题。** 一是在覆盖充分的文档上，固定 R1/R2/R3/R4 是否有可重复的收益差；二是在同一个冻结骨干上，动作相对成本是否可由合法状态预测。先补抽样和廉价线性/小 MLP/常量规则，再决定是否更换骨干。当前权重仍可用于诊断其自身，无需因训练协议不足而立即报废。

**第三阶段：只开放通过前置条件的分支。** 路由值有样本外增益再做 E02/E04；R1 不能满足质量而 gain 可预测时再做 E03；只有真实瓶颈和合法服务 trace 才做 E05/E06/E07 系统扩展；E10/E12 保留低成本筛查优先级；LM RL 继续等待明确的监督受限证据。

通常 3–5 张 A6000 足够按这一顺序推进。最有价值的使用是状态标注、独立种子/对照并行与隔离测量，不是把当前接口错误扩展到更多卡。具体任务、验收门槛、数据量与资源安排在 [02_repair_and_rerun_plan.md](02_repair_and_rerun_plan.md)。

## 9. 本轮完成与仍未确认的边界

本轮已经完成上传源码、配置、日志和记录的审查、原测试执行、关键错误反例、参数/成本/数据统计复算、最小可审阅补丁和复测计划。没有向远端仓库推送或修改历史。

仍未确认：raw-h 修复对原检查点真实 NLL 的变化量；学生训练拟合曲线与 hidden_dim；扩大文档后的可部署收益；是否存在实际专家塌缩/状态收缩；修正计时后的 A6000 延迟。它们需要指定权重或新实验，不会在本轮以猜测填数。下一轮最小材料已列在 [03_next_round_artifacts.md](03_next_round_artifacts.md)，不需要上传所有 optimizer 和历史 checkpoint。

---

## 原始源码与结果索引

[src-model]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/model/transformer.py
[src-top2]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/model/top2.py
[src-cf]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/eval/counterfactual.py
[src-cf-sampling]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/eval/counterfactual.py#L133-L151
[src-cf-metrics]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/eval/counterfactual.py#L250-L282
[src-student]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/train/value_student.py
[src-halting]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/eval/dynamic_halting.py#L26-L36
[src-fineweb]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/data/fineweb.py
[src-runlm]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/run_lm.py
[src-rune01]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/run_e01.py
[src-e01-result]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e01/m105_100m/e01_summary.json
[src-e03-result]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e03/m105_100m/e03_summary.json
[src-e11-result]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e11/m105_s0_dynamic_profile.json
[src-profile]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/eval/profile_dynamic.py#L28-L62
[src-configs]: https://github.com/kkc25/GB_RL_MoE_Loop/tree/f2bef422f7315f30618f00c690bf82810d1e85ac/configs
[src-lmtrain]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/train/lm.py#L64-L127
[src-plan-common]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/protocols/01_common_protocol.md
[src-recoverytrain]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/train/recovery.py
[src-packed]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/data/packed_lm.py#L320-L361
[src-e12]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/e12_bidding.py
[src-e10]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/e10_ordered.py
[src-e07]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/e07_replay.py
[src-e0506]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/e05_e06_synthetic.py
[src-e08]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/e08_ab.py#L976-L1008
[src-e09]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/e09_controlled.py
[src-oldexact]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/train/exact.py#L20-L37
[src-oldoracle]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/eval/oracle.py#L22-L45
[src-runrecovery]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/run_recovery.py#L65-L89
[src-causality]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/eval/causality.py#L486-L556
[src-registry]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/registry.py#L84-L92
[src-registry-result]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/registry.json
[src-report]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/Loop_MoE_Experiment_Report_2026-10-25.md
