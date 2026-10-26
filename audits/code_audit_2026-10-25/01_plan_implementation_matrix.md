# E00–E12：原方案、上传实现与证据等级矩阵

审查对象：`kkc25/GB_RL_MoE_Loop`，固定提交 `f2bef422f7315f30618f00c690bf82810d1e85ac`。审查日期：2026-10-25。

本文件对照上一轮 `research_bundle/experiments/E00–E12`、仓库内保留的同版研究方案、已上传源码、配置和运行产物，回答三个不同问题：**算法是否按约定实现、实验实际运行到哪里、证据允许作出什么结论。** 未上传权重不视为实验没有运行；但没有用权重复核的内容，也不写成已经复现模型质量或 GPU 性能。确定错误的最小反例及源码行号见三份详细审查：[模型与训练](agent_model_training.md)、[反事实与控制器](agent_counterfactual.md)、[合成机制与 RL](agent_synthetic_rl.md)。

**总体判断：基础 LM、冻结反事实标注、若干有限机制验证已经真实落地；真实 LM 主线仍处在 smoke／诊断阶段。** 当前材料没有完成原方案中任一真实 LM 方法的确认级比较。若干暂停扩大投入的决定合理，但不能将它们升级为“方案充分执行后已排除该科学假设”。E11 和 E12 还有直接改变被测算法或比较边界的确定实现错误，需要先修正。

## 1. 本矩阵的证据等级

| 等级 | 本文件中的含义 | 不自动意味着什么 |
|---|---|---|
| 未执行 | 没有相应目标实验的运行证据，或明确依条件暂停 | 不代表方法失败，也不要求立即启动 |
| Smoke | 跑通输入、训练、输出和最小执行链，样本／训练量很小 | 不足以证明可学性、不可学性或系统优势 |
| 诊断 | 在明确范围内量化现象、识别可疑点或判断下一步花费 | 不等于按原协议确认科学结论 |
| 门禁 | 特定性质经有限枚举、参考实现或有区分力的测试通过 | 结论只覆盖已检验的机制、输入范围与路径 |
| 确认 | 目标算法、强对照、数据划分、预算与统计协议均足以支持预定结论 | 不由 `finished`、一个阈值布尔值或测试数量决定 |
| 需修复 | 已执行实现或统计／测量定义存在具体错误 | 不代表所有原始数值虚假，需逐项保留可用证据 |

数学构造的有限反例或全状态枚举，可以确认其自身的有限命题，不必机械要求三个随机种子；它们仍不构成真实 LM 收益确认。下文的“门禁”优先表达这种范围限制。

仓库的 [registry 实现][registry] 会在记录组件时直接更新实验顶层 `status`。例如 E09 的神经 LM 部分暂停、受控部分完成，顶层可以成为 `finished`；E11 扩大范围暂停、S0 profile 完成，也会成为 `finished`。因此本表采用实际执行范围，独立于这些汇总标签。

## 2. 全部实验一览

| 实验 | 原方案要回答的问题 | 上传实现／运行范围 | 审查后的证据状态 | 当前能支持的结论 | 最近下一步 |
|---|---|---|---|---|---|
| [E00][p00] | 循环骨干是否训练正常，基础比较是否公平，单卡是否可行 | S67 B0–B5 各 5M；M105 B2 单种子 100M；基础正确性及资源记录 | Smoke＋部分工程门禁；配平、训练及账本需修正 | 模型能训练、末层 FFN 循环和稀疏执行真实存在 | 修 batch／配平／恢复，再补足能区分原因的训练与验证 |
| [E01][p01] | 路径 oracle 空间是否能由合法状态预测 | 全 28 首动作＋冻结 T4 tail；8192／1024／1024 状态，两类 MLP | 诊断；跨文档覆盖不足 | 当前学生未在当前小样本上改善；存在样本内 label-informed oracle 空间 | 文档分散采样、独立划分、简单强基线、学习曲线 |
| [E02][p02] | 多步价值蒸馏是否改善固定成本路由 | 前置 E01 学生；正式 E02 暂停 | 未执行 | 不能判断正式 E02 成败 | 先修 E01 诊断，再决定是否进入完整 rollout distillation |
| [E03][p03] | 状态相关停止是否改善质量／成本权衡 | T1 或 T4 二元离线停止、validation 选阈值 | 诊断；尚无独立校准与确认 | 在已观测样本均值上，很多额外轮次可省；固定 R1 也已在容忍内 | 先比较固定 R1，再考虑直接 delta 头与独立校准 |
| [E04][p04] | 历史是否超出充分当前状态／容量对照 | 神经 LM 历史实验暂停；有合成机制和旧 GRU 基础代码 | 未执行；相关合成门禁有限有效 | 完整／压缩状态的理论区分可以保留；LM 记忆价值未判定 | 只在合法状态与容量对照充分后启动 |
| [E05][p05] | 真实拥塞下价格能否改善资源分配 | 固定状态合成势游戏；真实服务部分未执行 | 有限机制门禁 | 异步势函数性质与同步振荡反例成立于构造环境 | 有真实拥塞证据后再采 trace、比较服务指标 |
| [E06][p06] | 因果、distinct Top-2、硬容量分配是否可行且有收益 | 合成 min-cost flow、soft transport、rounding | 整数门禁有效；软可行性表述需修复 | 整数流保证 distinct 与容量；软解不能自动算合法 Top-2 可行点 | 补 `P≤1` 或改名为放松问题，分别报告可行性 |
| [E07][p07] | 跨请求分配额外计算是否优于简单预算规则 | 合成 DP／扣账；64 状态离线端点插值回放 | 扣账门禁＋代理环境诊断 | 当前额外预算执行不超支；没有稳定调度优势证据 | 修 regret／重排；若继续则标注真实中间轮收益 |
| [E08][p08] | 理论边界、延迟收益、稳定性与实现门禁 | A/B/C/D/E 小型 CPU 机制验证与辅助恢复任务 | 多项有限机制门禁；B3 干预需修正 | 若干构造反例和因果／预算性质有效 | 修固定题目的路由干预；LM 归因等待真实状态证据 |
| [E09][p09] | 冻结专家后策略梯度能否超越监督／蒸馏 | 独立 NumPy 受控梯度与训练比较；神经 LM 暂停 | NumPy 梯度门禁有效；LM 未执行 | 当前 NumPy exact／RLOO 正确；受控优化表现可报告 | 保留该门禁；旧 Torch exact 修好后才可使用 |
| [E10][p10] | 有序两步互补是否存在、可预测且可部署 | S67 32／16／16 状态，28² 前两步＋固定 tail | Smoke／小诊断；符号指标需修正 | 成本矩阵生成与同成本比较链跑通 | 先补跨文档样本与充分拟合，或维持低优先级未决 |
| [E11][p11] | 动态执行能否带来真实端到端净收益 | M105 S0 末层共享 GPU profile；S1 以后未运行 | 需修复；系统 smoke | 当前不能确认 E03 策略正确部署或真实加速 | 修 raw-h 接口和计时边界，公平比较固定 R1／R4 |
| [E12][p12] | 分专家非线性报价是否超出同容量中央路由器 | 线性等价检验＋复用 E01 标签的非线性 adapter 比较 | 线性门禁有效；非线性需修复 | 线性重参数化等价成立；非线性实验尚不能排除独立 adapter | 修 einsum，再同候选、同优化设置比较；无新信号则降优先级 |

## 3. 跨实验协议：主要缺口在哪里

公共协议不是装饰性文档。它规定了各方法之间可比较的单位，以及何种阴性结果才足以影响研究决策。[公共协议][common]、[数据与评估协议][data_protocol]与实际上传运行存在以下关键差别。

| 约定 | 已上传执行 | 对结论的影响 |
|---|---|---|
| 独立 backbone train／controller train／controller validation-calibration／backbone validation／controller test／LM test；控制器探针训练文档不用于骨干更新 | 数据目录使用 train／validation／test 三分区；M105 骨干已遍历同一 train 池；E01 选择学生与 E03 选阈值使用同一 validation 状态 | 与原隔离协议不同；不能说已经证明 test 进入训练，或所有 E01 validation 位置都被骨干选点使用 |
| E00 pilot 8192、正式起点 32768 有效 token／optimizer update | 运行 batch=1，S1024，没有梯度累积；M105 约 1022 个监督 token／update | microbatch=1 被实际用作优化 batch；训练条件与原方案不等价 |
| E00 筛选 1M、正式 10M validation token；文档级 paired bootstrap | 质量评估 16,342 个监督 token；E01／E03 的 test 只有 3 个文档，SE 按 token 计算 | 可报告样本均值；不足以确认细小收益或泛化阴性结论 |
| 确认阶段多个骨干 seed；先看样本量／训练量学习曲线 | 当前主要 LM 路径单 seed，小样本；E10 每模型只有 20 次更新 | 不能区分方法失效、优化不足、骨干未准备好、数据覆盖差 |
| 骨干、标注、控制器、重跑和失败运行统一成本账本 | 训练 checkpoint 时间保存顺序漏记旧段；Top-2 `expert_calls` 部分记成轮数；registry GPUh 聚合不完整 | 不能从聚合表证明 5%／10% 资源上限已执行；缺失应记 unknown |
| 固化数据、骨干、候选、学生和阈值的对应关系 | 部分 hash 已记录，但学生与标签、E11 三类输入之间未全面验证；完整 CLI 不总是保存 | 当前没有证据证明混错文件；未来误配可静默通过，复现实验仍缺精确入口 |

这些差别不要求现在无条件补跑所有原计划。它们要求**先修正确性和可比较性，再用足够的诊断证据选择保留哪条支线**。原方案本来就是按门禁分阶段投入，不是十三个实验必须同时跑满。

## 4. E00：基础模型、训练协议与配平

### 原方案的核心承诺

[E00][p00] 将主模型定为 M105：L12、d512、8 专家、Top-2、专家 FFN m512、最后一层只循环 FFN，最大 4 轮，attention 每块一次。B0 固定 R1、B1 固定 R4、B2 按 batch 均匀采样 R1–R4；B3 比较计算匹配 dense，B4 比较参数匹配 dense，B5 比较共享 dense loop。先做正确性／恢复／资源校准；S67 累计 50M 中前 5–20M 为 smoke；M105 B0／B1／B2 各累计 300M 中前 100M 为 pilot。后续确认与扩展依据这些结果触发。

### 已上传实现与证据

`model/transformer.py`、`model/top2.py`、`train/lm.py`、`run_lm.py` 和 `configs/e00*.json` 实现了真实训练链。[E00 对比结果][r00]显示 S67 三个 MoE 及三个 dense 变体各训练 5M input token，M105 只有 B2 的 100M pilot。基础架构、参数计数、Top-2 两专家执行、稀疏 dispatch、绑定输出头及文档内因果 attention 基本落实；不能把这些实现描述成仅在全专家输出后做 mask。

**dense 配平必须明确对象和深度。** 下表为每 token 所有 FFN 主矩阵的 FLOPs，未把它冒充端到端时延；gate、dispatch、attention 和词表头需另外计入。

| 模型／配置 | 参数量 | R1 FFN FLOPs | R4 FFN FLOPs | 正确比较口径 |
|---|---:|---:|---:|---|
| S67，m256、k2 | 67,170,816 | 18,874,368 | 23,592,960 | 当前 5M MoE 对象 |
| B3 dense，m1024 | 48,247,296 | 37,748,736 | 37,748,736 | 与 **M105 R1** 的 FFN 主矩阵工作匹配；不匹配 S67，也不匹配 M105 R4 |
| B4 dense，m4096 | 104,870,400 | 150,994,944 | 150,994,944 | 与 **M105 参数量**近似匹配；比 S67 多约 56% 参数 |
| B5 shared dense，m1024 | 48,247,808 | 37,748,736 | 47,185,920 | 与 **M105 同深度**的 FFN 主矩阵工作匹配；相同深度约为 S67 的两倍 |
| M105，m512、k2 | 104,919,552 | 37,748,736 | 47,185,920 | 当前仅 B2 100M；没有同训练量的匹配 dense 运行 |

B3 同样不匹配深度均值为 2.5 的 M105 B2 训练工作量。当前 dense 数值可以作为不同结构的 smoke 对照保留，不能标成针对 S67 的计算／参数配平确认。

此外，运行的逻辑 batch 约 1024 input token，而不是计划的 8192／32768 有效 token；所谓 random depth 实际为确定性周期轮转，只有边际频率均匀。恢复功能会漏记历史时间、重置最佳验证值，延长总步数还会改变学习率 schedule；sampler 每 update 重建全数据排列带来不必要的 O(N²／batch) 每 epoch 索引工作。这些均有源码或最小真实执行证据，详见 [模型训练审查](agent_model_training.md)。

### 证据判断与下一步

**状态：smoke／pilot 已运行，部分工程门禁通过；正式公平比较尚未完成。** M105 现有验证 R1 NLL 为 5.0709673、R4 为 5.0703937，差约 0.000574 nat；R2 还略优于 R4。这提示当前骨干的额外循环价值很小，不能独立判断是充分训练后的结构事实还是训练／诊断条件造成。

先修 batch 与恢复契约、按明确 S67 或 M105 与 R 深度重建对照表，输出每轮负载、router 熵、hidden／update 范数和专家重复率。完成短程 batch／LR 与学习曲线校准后，才决定补哪几个模型和多少 token。现有 C0／C1 仿射恢复任务是附加机制检查；C1 的泛化失败不能代替真实 LM 的 E00 验收，也不能由未触发的 `run_recovery --mode evaluate` 加载 bug 来解释。

## 5. E01：oracle 空间与可部署可学性

### 原方案的核心承诺

[E01][p01] 区分单步、28² 有序两步、4 候选固定策略续算、独立 Top-1 精确机制和可学性学习曲线。可学性起点为 100k／20k／20k 状态，来自独立文档；比较线性、小 MLP 与容量匹配 MLP，保留原路由、随机、标签打乱等负对照。所有候选同成本；oracle 只作知标签上界，不能当部署收益。进入 E02 需要持出文档上的稳定学生改善，0.01 nat/token 是原方案的初始实用门槛。

### 已上传实现与证据

`run_e01.py`、`eval/counterfactual.py`、`train/value_student.py` 已实现冻结骨干，在最后 FFN 前缓存 raw h，对全部 28 个首动作执行后按冻结 π0 续算至 T4，并分别回归即时成本和长期成本。默认 28 动作、包含基准动作、固定续算和 label-informed oracle 单列均基本正确。[M105 summary][r01] 的状态数为 8192／1024／1024；这是一种比 4 候选更广的首动作诊断，但不是原方案所有阶段都已完成。

关键问题在抽样：每抽到一个 packed sequence，连续收集其全部监督位置，达到数量后截断。因此 M105 train 只有 15 个文档，validation 只有 2 个，test 只有 3 个，其中 test 的 906／1024 状态来自同一文档。`coverage_remediation` 增加了 token 数，但没有解决独立文档覆盖；它与 M105 run 还复用了相同 validation／test 位置。

上传 M105 oracle gap 为约 0.10954 nat；即时学生相对基准增益 −0.00384，长期学生 −0.00677。可以忠实报告这些样本均值，但不能将 token-IID SE 当作文档级独立证据。独立的已上传记录分析还发现，绝对 Q 的 test 总方差约 99.87% 来自各动作共享的 token 难度；这提示绝对 MSE 可能不善于选择决策模型，**并不证明中心化监督一定会成功**。

### 证据判断与下一步

**状态：有用诊断，尚未达到“状态不可学”的排除门槛。** 当前能说的是“两名特定学生在极少文档的测试上没有改善”，以及“这些样本存在知标签的候选选择空间”。暂停昂贵 E02／RL 扩跑是合理成本控制；把该暂停写成所有可观测状态都不足则超出证据。

下一步按文档分散抽样并限制每文档贡献，明确 train／selection／calibration／test；先补线性、小 MLP、固定动作与原 router 条件基线，再看容量和样本量曲线。路由可比较动作优势或中心化 Q 与实际 decision regret；停止分数则须保留跨目标共同偏移，不能把两套 Q 各自中心化后直接相减。若使用原方案 4 候选节省标注，应先修自定义候选 action ID 映射；默认全 28 的当前结果不受该潜在接口错误影响。

## 6. E02：续算价值蒸馏与有限规划

### 原方案的核心承诺

[E02][p02] 以固定 T4 的 Q^π0／优势监督为主线，从当前状态低成本提议候选，部署时不额外试算未选专家。控制器应见合法 h、轮数和预算；按实际部署 NLL 选模型；覆盖各轮状态，必要时作有限次数 on-policy 数据聚合。原方案另外定义至多 16 条两步组合加基准的可选规划，以及确认后才启用的联合微调。必须与即时价值、EPO-style、容量匹配等强对照等预算比较。

### 已上传实现与证据

E01 的 Q 学生提供了这条路线的一部分基础，但未形成完整多轮部署、状态分布覆盖、数据聚合及 E02 对照矩阵。registry 的 E02 明确因前置证据暂停；没有相应独立正式运行产物。[状态记录][goal_status]与源码支持“尚未启动完整 E02”，不支持“E02 已运行失败”。

### 证据判断与下一步

**状态：正式实验未执行。** 应继续保留条件启动，先修 E01 的样本覆盖和学习诊断。若持出文档上的信号仍缺失，则记录适用范围内的阴性诊断并暂停；若出现稳定改善，再完成固定成本 Q^π0 部署与简单强对照。没有必要先上联合微调或更大搜索树来掩盖前置问题。

## 7. E03：价值校准的停止

### 原方案的核心承诺

[E03][p03] 使用“现在停止”相对“继续到指定尾迹”的配对收益，并将完整剩余调用成本计入；各可停止轮有合法状态输入，小 delta 头或分类头可部署。通过独立 calibration 调阈值达到目标平均轮数，测试必须报告实际预算漂移。固定 R1–R4、等平均成本随机策略、熵／margin／hidden-change 等都应作为对照。初始 NLL 容忍为相对 R4 增加不超过 0.005 nat；**若 R1 已同质量，应优先固定 R1。**

### 已上传实现与证据

`experiments/e03_halting.py` 明确复用 E01 的 T1／T4 标签，只在第 1 轮选择立即停止或一直算到第 4 轮；它没有伪造 T2／T3 标签。这是清楚披露的二元 pilot。[M105 summary][r03] 显示，在相同 1024 test 状态上：

| 方法 | 实际平均轮数 | 相对 T4 的 NLL 增量 |
|---|---:|---:|
| 固定 T1 | 1.00000 | +0.00260160 |
| target 1.5 的动态策略 | 1.71191 | +0.00238314 |
| target 2 的动态策略 | 2.37695 | +0.00099853 |

这里 `target` 是 validation 上的目标平均成本，不是 test 的硬上限。实际轮数偏离 target 是需要报告和评估的分布漂移，**不是硬预算执行 bug**。硬请求预算／跨请求扣账分别是另一类约束。

当前停止器由两个全成本 MLP 相减；预测尺度与真实 delta 的分箱明显不一致，尚不能把阈值搜索叫作价值已校准。validation 同时用于 E01 学生选择和 400 点阈值搜索，也没有落实独立 calibration。`within_tolerance=True` 只比较样本均值，没有形成原计划的文档级不确定性结论；不能事后任选单侧／双侧 CI 规则宣布通过或失败。

### 证据判断与下一步

**状态：二元离线停止诊断，未确认自适应计算的必要性。** 固定 R1 的均值已在同一容忍内，动态方案尚未显示相对更便宜固定规则的净价值。E11 的输入契约错误影响在线实现，不应反过来声称 E03 的离线查表均值本身因该错误而无效。

优先补多文档固定 R1／R2／R4、等成本无条件混合，预先确定推断方式。如仍有按状态分配计算的价值，再训练直接配对 delta 头、独立校准，比较有限阈值下离线／在线逐 token 决策。后者需要代表性权重和状态，但本轮已经足以判定现有接口错误。

## 8. E04：历史记忆与转移结构

### 原方案的核心承诺

[E04][p04] 在同标签、同调用数下比较 h-only／h+t+budget、容量匹配更大 MLP、前一专家、无序次数、顺序 GRU 记忆、MoUE-style、无关历史和额外 h0。必须把“历史增加信息”“历史使有限函数类更易读取”“历史只增加容量”区分开；参数差控制在 1%，不能只增加在线计算后声称记忆有效。

### 已上传实现与证据

神经 LM 记忆分支因前置条件暂停；`policy/controller.py` 中存在 GRU 并不等于 E04 对照矩阵完成。E08 中完整状态／信息擦除／函数类受限的合成实验与 E04 有理论联系，但不是其真实 LM 替代结果。

### 证据判断与下一步

**状态：真实 LM E04 未执行。** 不能据现有 E01 MLP 阴性推导“必须加记忆”，也不能据有限充分状态证明推导“真实 hidden state 下记忆永远无用”。只有 E01 状态、轮数和预算接口明确，且容量对照足够后，再比较最便宜历史特征。没有新可学性信号时保持暂停合理。

## 9. E05：价格与拥塞游戏

### 原方案的核心承诺

[E05][p05] 首先确认真实服务中存在稀缺资源和可测拥塞；固定状态下研究异步改进与势函数，和同步振荡分开。进入服务实验后，只使用因果请求前沿，价格读已完成窗口，质量值与价格使用一致单位；硬容量由可行分配保证。比较原 Top-2、硬约束 greedy、ALF-style、价值减价格、异步迭代及 E06，报告质量、p95、fallback 和控制开销。

### 已上传实现与证据

`experiments/e05_e06_synthetic.py` 与 [合成结果][r0506] 完成固定状态势游戏部分；单玩家异步严格改进的势函数增量和同步振荡反例在构造环境中正确。文件明确写 `real_lm_serving_part=not_executed`，没有把这些 CPU 数值伪装成 GPU 服务收益。

### 证据判断与下一步

**状态：有限机制门禁有效，真实服务研究尚未执行。** 可以保留固定游戏命题；不能据此声称动态 hidden state 上一定收敛、存在经济激励相容性或优于无价格容量调度。

继续条件应是 E00／E11 profile 暴露真实拥塞，且有稳定价值输入。届时采真实请求 trace、统一 nat 质量损失与成本，再比较有无价格；只有负载 CV 下降不足以支持服务收益。当前不应为了完整编号而强行部署这条支线。

## 10. E06：因果 transport 与容量分配

### 原方案的核心承诺

[E06][p06] 的合法 Top-2 分配要求每 token 行和 2、每专家列和不超过容量，且 token–expert 边满足 0≤P≤1、禁用边为 0；总容量足够只是必要条件。加性代理与集合 Q 需要分开；整数 min-cost flow 为参考，软迭代、rounding、fallback 必须分别验可行性。服务实验还需检查因果前缀、质量与开销。

### 已上传实现与证据

整数 `solve_additive_assignment` 在 token→expert 边上设容量 1，正确防重复专家；流量不足拒绝不可行实例，rounding 失败可回退 exact flow。这些是有效正面门禁。

软 `soft_transport` 未纳入 P≤1，且只按总容量给 `feasible`，汇总又漏掉已计算的 upper-bound violation。例如一个 token、两个专家、k=2、容量 (2,2)、成本 (0,100)，软解可接近 (2,0)，而合法 distinct Top-2 必须 (1,1)。因此当前软输出可以是一个放松问题的解，不能直接称合法 Top-2 可行点。该反例**没有击穿整数 flow 的硬容量保障**。详见 [合成审查 S07](agent_synthetic_rl.md)。

### 证据判断与下一步

**状态：整数求解器门禁成立；软解的可行性定义和汇总需修复；真实服务未运行。** 最小修正是纳入边上界，或明确改名为 relaxed transport，并单列 row／column／edge／forbidden residual 和整数图可行性。之后才比较 soft objective 与整数最优值；上游 predictor→allocator 的真实因果性还需端到端测试，重复调用同一不接收 suffix 的函数不足以验证上游。

## 11. E07：跨请求预算与收益指数

### 原方案的核心承诺

[E07][p07] 区分已知服务窗口与因果在线流；每个请求已有最少计算后，调度其额外 Top-2 调用组。单步指数、完整 tail 指数、两步 package 必须与各自真实时间跨度和成本相配，不花费预测非正收益预算。先有限 DP、再真实逐轮 LM replay，与 uniform／random／简单分数／预算 MLP 比较；另做重排、长度混合、分桶及实际 p95。

### 已上传实现与证据

`experiments/e07_synthetic.py` 有有限 DP 和实际扣账；`experiments/e07_replay.py`、[M105 replay][r07] 用 64 个 E01 test 状态组成窗口，只有实测 T1／T4 两端点，将中间收益分别按 diminishing 和 delayed 权重插值。文件明确披露这些是假设，并没有实测 T2／T3、请求执行器或 GPU 分桶时延。当前预算单位为末层最少一轮之后的额外 Top-2 组；不能拿它等同全模型所有层成本。

审查未发现当前执行账本超支。但有两处统计／设计问题：opt=0 时把相对 gap 强设 0，会隐藏实际负收益；声明所有 offline allocators 重排不敏感，漏测了余量按顺序给出的 equal-depth 和分块 DP 等情形。

此外，完整 64 请求离线 oracle 在当前**已知、固定、可分离的每请求前缀收益数组**上可写成 O(NBH) 多重选择背包，不需要枚举所有联合状态。审查包已给出与原小 DP 交叉核对的实现和回放结果。这只补强已声明插值环境的上界，不是在线预知未来的可部署策略，更不是新增 LM 中间轮实测。

### 证据判断与下一步

**状态：额外预算门禁通过，插值环境诊断完成；真实 LM 调度优越性未确认。** 原报告“没有稳定优势”的描述适合当前回放；一个预算 cell 的改善既未稳定超过 random，也不能排除真实逐轮收益驱动的调度。

先修绝对 regret／相对 regret 的零分母、对相同 request ID 重排并保留对齐，加入完整窗口离线 oracle。若仍值得继续，采集每个真实后状态的逐轮增量与预测，最后才做 GPU grouped execution；缺权重并不是当前中间真值缺失的全部原因，原端点数据本来就不能唯一恢复中间值。

## 12. E08：合成机制、稳定性与实现门禁

### 原方案的核心承诺

[E08][p08] 分别验证完整充分状态、信息擦除、有限函数类可读性、延迟收益、不可学习 oracle、非交换有序组合、固定／变化的资源问题、切换稳定性，以及真实执行的因果与预算门禁。数学构造和自然语言外推必须分开；若连接 LM 稳定性，应有实际范数／Jacobian／信息保留证据。

### 已上传实现与证据

`experiments/e08_ab.py`、`experiments/e08_cde.py` 及 [A/B 结果][r08ab]、[C/D/E 结果][r08cde] 覆盖多项小型机制。异步／同步区别、单矩阵稳定不保证切换稳定、谱范数约束与谱半径不同、因果前沿和执行扣账等范围内的正面结论可保留。这些文件也明确其 CPU 合成范围。

需要修正 B3 干预语义：当前 reverse／random／repeat 同时改变程序描述与正确答案，测试的是另一族程序的求解，而非“同一题目强制错误专家顺序”。例如原题 AB 的答案是 9，强制 BA 得 7 本应错；目前把题目也换 BA／7，就不能测试顺序损害。应保持输入、目标和任务不动，另传 forced action path。

辅助 C0／C1 recovery 是不同神经任务；C1 IID 泛化弱不能否定精确机制反例。旧 `eval/causality.py` 聚合 CLI 还有一个因非法多自适应层 target-only 配置被生产约束拒绝而退出的问题；这是门禁入口维护问题，不是因果性被破坏。过时注释提到的 memory dtype、每层 rounds、共享采样流问题也不能在已经修正的生产路径上重新认定为现存缺陷。

### 证据判断与下一步

**状态：若干有限命题与实现门禁有效，部分学习／干预还在诊断阶段。** 修 B3 路由干预与因果聚合入口后，可保留为低成本回归工具。只有真实 LM 的 hidden norm、更新范数、专家序列与误差显示相关结构时，才提出 LM 稳定性归因；当前没有这类证据时，扩大控制器或循环深度缺少依据。

## 13. E09：策略梯度与受控精确参考

### 原方案的核心承诺

[E09][p09] 以冻结 LM 专家、Top-2 集合策略和已训练 E02 初始化为对象，比较 REINFORCE／LOO 与监督、蒸馏、扩大候选、搜索蒸馏；先用完整有限树核对概率、梯度和预算。E01 可学性和 E02 覆盖不足时应延后神经 RL。正式比较有 rollout／候选 CE／GPU-hour 上限。

### 已上传实现与证据

必须区分两套代码：

| 路径 | 实际范围 | 审查结论 |
|---|---|---|
| `experiments/e09_controlled.py`＋`data/controlled.py` | 已上传的独立 NumPy 受控 E09 | 真实终止树概率归一化；exact 使用路径概率权重；RLOO／exact 与有限差分一致 |
| `train/exact.py`＋`eval/oracle.py` 等旧 Torch 入口 | 较早神经 Top-1／GRU 基础设施，不是当前 NumPy run | exact loss 缺 π 权重；预算小于 max_calls 时枚举重复截断轨迹；不能未经修正直接作未来精确参考 |
| 原计划冻结 Top-2 LM RL | 前置条件未满足，明确暂停 | 没有完整目标实验执行证据 |

[当前 controlled JSON][r09] 中监督训练任务恢复 6／6；exact 的训练目标优于监督，RLOO 在该设定下更差且有方差。独立 24 例 held-out 则为监督 6／24、exact 7／24、按训练目标选出的 RLOO seed 7／24；这些小差异不支持泛化优越性，但也不允许说监督在所有泛化指标都更强。

### 证据判断与下一步

**状态：NumPy 梯度门禁通过，受控训练比较可保留；真实 LM E09 未执行。** 旧 Torch 错误不能解释当前 NumPy 结果，当前 NumPy 通过也不能替旧 Torch 背书。保留正确受控模块作未来门禁；若以后使用 Torch exact，先补 π 权重、唯一终止轨迹及阶段恢复语义，再把同一目标的梯度与有限差分对齐。真实 RL 继续条件仍应来自 E01／E02，而不是因为 RL 代码存在就投入 LM rollout。

## 14. E10：有序互补与交互预测

### 原方案的核心承诺

[E10][p10] 用同一参考动作定义四项交互 Γ，分别测顺序效应；两轮路径保持等成本，oracle 单列。起点约 256 个状态／seed、多个骨干 seed，对比 additive、低秩与容量匹配 MLP，并看预测动作的实际 NLL；非零 Γ 本身不能证明可部署价值。

### 已上传实现与证据

`experiments/e10_ordered.py` 和 [pilot summary][r10] 在 S67 上采 train 32、validation 16、test 16 状态，每个 split 仅一个文档。完整枚举前两轮的 28² 组合，之后按 π0 续到 T4；这是一种**等总 4 轮 tail 的两步交互变体**，不应与纯两轮实验混为同一个条件。前缀共享、四项同参考动作和成本、重复动作、oracle 分列等主要结构合理。

两种约 143k 参数的预测头每 epoch 只有一 batch，20 epoch 即 20 次 optimizer update。该训练量和文档覆盖不足以排除互补可学性。`sign_flip` 又把正与非正共存算为翻转，Γ=[1,0] 也被计入；应要求严格正与严格负并设数值零带。上传成本只保留六位小数，不能用其重算数冒充原全精度修复后实验。

### 证据判断与下一步

**状态：流水线 smoke／极小诊断。** 当前可以确认成本矩阵生成链已运行，不支持“互补性不可学习”或“顺序结构没有价值”。若继续，先修符号定义、补多文档样本，验证训练拟合与样本量曲线；未见充分拟合前不必换更复杂头。考虑主线资源，维持低优先级未决也合理，但应把暂停与证伪分开。

## 15. E11：动态执行与真实系统收益

### 原方案的核心承诺

[E11][p11] 从 S0 单末层逐步升级到多末层／更大模型；前一阶段未证明净收益，不自动扩展。中间层反事实必须重新计算受影响 suffix。动态执行应真实跳过／分桶，统一统计控制器、输出头和调度成本；teacher forcing／prefill／decode 分开。确认测量要求独占或可控环境、充分热身与重复，目标同质量端到端净加速至少 5%，不确定性跨 0 则不作成功结论。

### 已上传实现与证据

`eval/dynamic_halting.py` 确实只对 continue 的位置继续执行额外 FFN；[S0 profile][r11] 在共享 GPU 上测 M105 末层，S1 以后未执行。但两处确定问题使现有比较不能验收：

1. E01／E03 学生训练和校准输入 raw pre-FFN h；在线却输入 `ffn_norm(h)`，还用 raw-h 的 mean／std。部署的已不是同一个阈值策略。
2. `profile_dynamic.py` 固定 R1／R4 调用模型时传 labels，额外执行 FP32 cross-entropy；dynamic 只输出 logits。固定和动态计时边界不同。即使改为独占 GPU，这项偏差也仍然存在。

另外，`mean_rounds` 的分子包含所有执行位置，分母仅为监督位置；peak memory 没有按模式重置；当前 warmup／steps／重复范围不足以满足原确认协议。共享负载是另一项已披露限制，不能用它代替上述代码解释。

### 证据判断与下一步

**状态：真实动态稀疏执行的系统 smoke，算法接入与性能比较需修复。** 现有 dynamic 比 R1 更快的数值不能用于正式速度结论，也不能据其 NLL 在阈值内判定 E03 已成功部署。修后究竟快多少、质量变多少，需要原权重与公平 GPU 测量；本轮不虚构这个幅度。

先共享 raw-h 特征接口，用有限 tau 验证离线／在线分数与 stop mask；固定和动态统一 input→logits 计时边界，质量另算。重新比较固定 R1／R2／R4 与动态，报告真实执行专家-token 对及监督位置平均深度。只有相对最便宜合格固定策略仍有净价值，才进入 S1；3–5 张动态分配 A6000 可用于独立实验／种子，不能把多卡占用自动当作同一模型吞吐扩展收益。

## 16. E12：专家独立报价与中央路由对照

### 原方案的核心承诺

[E12][p12] 首先验证线性专家报价堆叠与中央线性 router 的等价性，再研究非线性、摘要与历史是否构成有价值的归纳偏置。Top-2 条件报价使用相同 anchor、合法状态、候选、标签、容量及计算预算；不执行未选专家来估价，也不把普通分布式打分称为策略经济机制。

### 已上传实现与证据

`experiments/e12_equivalence.py` 的线性等价检验有效；`experiments/e12_bidding.py` 与 [非线性 summary][r12] 复用 E01 8192／1024／1024 状态比较中央 MLP 与分专家 adapter，实际运行的摘要模式是 learned。

关键 einsum 使用 `...ei,oih->...oh`，先把所有专家输入沿 e 求和再给每个 o 的权重；正确独立 adapter 应对齐同一个专家轴。无摘要时相同 h 被放大 8 倍，虽可重参数化，优化尺度已变；learned 摘要也只以跨专家总和进入，不能代表逐专家独有摘要。**零均值 checkpoint 摘要完全相消是另一个未执行模式的风险，不是已运行 learned 摘要必为零。** 类还覆盖 `nn.Module._apply`，影响 `.to()`／`.cuda()`，但当前 CPU run 没触发，不能把它说成已有运行失败原因。

部署对照也未匹配候选：中央 MLP 从 28 对全局选优，报价只能在给定 anchor 的 7 对里选。原实现的同 cell MSE 对比成立，但不足以使部署 NLL 比较自动公平。中央／报价 weight decay 不一致，计算估计与实际 adapter 数也有差别；参数与 anchor 开销则已在报告中披露，不应说它们被隐藏。

### 证据判断与下一步

**状态：线性重参数化等价门禁有效，非线性分专家比较需重做。** 修 einsum 会改变旧模型函数和训练过程，不能直接把旧权重不经处理当作修复后公平模型。重命名 `_apply`、显式优化超参数，为每个 anchor 规则配 central-restricted argmin，unrestricted central 另列。若在修复和公平对照后仍无收益，可停止该归纳偏置分支；现阶段的负结果不足以排除正确实现的独立 adapter。

## 17. 建议怎样更新研究状态

下一版状态表宜分别保存 `implementation_status`、`execution_scope`、`gate_status` 和 `scientific_conclusion`，附数据／代码／骨干／学生／阈值 hash，避免一个总标签混合四件事。

目前最需要补齐的顺序是：

1. **修确定错误和公平性。** E11 raw-h 接口与计时、E12 独立 adapter；E00 训练 batch、对照配平和恢复／成本账本；保留旧 Torch RL 和其他扩展入口的明确禁用或修复状态。
2. **修证据采样。** 文档分散、独立 selection／calibration／test、足够验证 token；固定 R1 与简单策略先作为强基线，预先确定统计判据。
3. **再判断是否存在值得学习的状态相关剩余收益。** 同轮路由比较优势／排序／regret，停止直接比较配对 delta；学习曲线用于区分优化和泛化，而不是一见负值就提高模型复杂度。
4. **条件恢复支线。** 有路由信号才进入 E02／E04／E09；有额外计算收益才扩 E03／E07／E11；有真实拥塞才推进 E05／E06；E10／E12 保持低成本、强对照验证。

这个顺序保留了仓库已经积累的模型、标注和有限机制工具，同时避免把当前代码偏离、样本不足和科学假设本身混为一类结论。具体缺失材料与最小复核范围见 [下一轮材料清单](03_next_round_artifacts.md)。

---

## 固定提交证据索引

所有外部源码与运行链接均固定到本次审查提交，防止后续仓库变化改变本文件的含义。详细源文件行号、反例、补丁范围及版本限制见三份分项审查；本文只作研究承诺与证据范围映射，不新增实验结果。

[p00]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/experiments/E00_feasibility_and_baselines.md
[p01]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/experiments/E01_oracle_and_learnability.md
[p02]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/experiments/E02_counterfactual_rollout_distillation.md
[p03]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/experiments/E03_value_calibrated_halting.md
[p04]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/experiments/E04_memory_and_transition_graph.md
[p05]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/experiments/E05_prices_and_congestion_games.md
[p06]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/experiments/E06_causal_transport_assignment.md
[p07]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/experiments/E07_budget_scheduling.md
[p08]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/experiments/E08_stability_and_synthetic_mechanisms.md
[p09]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/experiments/E09_router_only_policy_gradient.md
[p10]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/experiments/E10_ordered_complementarity.md
[p11]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/experiments/E11_systems_and_scale_validation.md
[p12]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/experiments/E12_expert_self_bidding.md
[common]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/protocols/01_common_protocol.md
[data_protocol]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/plans/loop_moe_research_plan_2026-10-25/protocols/02_data_and_evaluation.md
[registry]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/experiments/registry.py#L84-L92
[goal_status]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/goal_status.json
[r00]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e00/all_lm_comparison.json
[r01]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e01/m105_100m/e01_summary.json
[r03]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e03/m105_100m/e03_summary.json
[r0506]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e05_e06/seed0.json
[r07]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e07/m105_replay/e07_replay.json
[r08ab]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e08_ab/e08_ab_results.json
[r08cde]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e08_cde_cde.json
[r09]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e09/controlled.json
[r10]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e10/pilot/e10_summary.json
[r11]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e11/m105_s0_dynamic_profile.json
[r12]: https://github.com/kkc25/GB_RL_MoE_Loop/blob/f2bef422f7315f30618f00c690bf82810d1e85ac/runs/e12/bidding/e12_bidding_summary.json
