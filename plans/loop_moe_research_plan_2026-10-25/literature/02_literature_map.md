# 循环 MoE 研究文献地图与基线优先级

更新日期：2026-10-25。此文件将文献按“会否推翻创新声明／改变实验”组织。下列一手来源已实际查询；访问深度标为 **正文相关节**、**摘要／出版页**或**受限索引**。表中“启示”“优先级”为本项目判断，不是原论文的结论。

## 1. 最靠近拟议研究的文献

| 文献 | 可核实内容与访问深度 | 对本项目的约束 | 实验优先级 |
|---|---|---|---|
| [CoE，2506.18945](https://arxiv.org/abs/2506.18945)，Zihan Wang 等，2025 | 单层内多轮专家处理；逐轮独立路由与内部残差。正文相关节 | FFN-only 循环的直接架构先例 | 必做同骨干机制对照 |
| [EPO / When Are Experts Misrouted?，2605.07260](https://arxiv.org/abs/2605.07260)，Youngsik Yoon 等，2026 | 固定专家，最终层等基数候选集合构造偏好监督。正文相关节 | 单步反事实集合优化已被提出 | 必做 router-only 一步偏好基线 |
| [MoUE，2603.04971](https://arxiv.org/abs/2603.04971)，Yilong Chen 等，2026 | 跨层通用专家池及轨迹 fast-weight router。正文 §4.4、Appendix E | “带路径历史”本身创新不足；其跨 token 状态更新需独立检查因果性 | 必做状态读出类比；完整架构可后置 |
| [TaH2，2609.35748](https://arxiv.org/abs/2609.35748)，Yichen You 等，2026-09 | lookahead 深度收益监督，训练额外一步无梯度预测用于停止标签。正文相关节 | “预测继续收益再停止”已有直接先例 | 停止路线必做机制基线 |
| [Adaptive Depth in Looped Transformers，2607.20519](https://arxiv.org/abs/2607.20519)，Andrei Cristian Popescu 等，2026 | 固定先验深度监督、冻结轨迹 gate 与多种读出比较。正文相关节 | 强制拆分“轨迹是否有用”和“gate 能否读出” | 必做固定深度、置信度、冻结轨迹控制 |
| [LOOM，2610.01153](https://arxiv.org/abs/2610.01153)，Di He 等，2026-10 | 整块循环的稳定性和路由多样化 recipe。正文相关节 | 不能把训练塌缩误当成路径价值无效，也不能把修复稳定性冒充路由创新 | 必做逐轮 router 和稳定 residual 的机制对照 |
| [Sparse Layers are Critical to Scaling Looped Language Models，2605.09165](https://arxiv.org/abs/2605.09165)，Ryan Lee 等，2026 | 四类架构比较和 loop 边界熵退出；正文明确未测端到端吞吐。正文相关节 | Looped-MoE+early exit 已有先例；模型 FLOPs 与墙钟应分列 | 必比强 loop 基线；系统验证另做 |
| [Path-Constrained MoE，2603.18297](https://arxiv.org/abs/2603.18297)，Zijin Gu 等，2026 | 跨层路径结构与共享 router。摘要／元数据 | 路径集中、路径熵和共享路由不能单独代表新颖性 | 路径结构路线必比 |

### 这些工作之间不能混为一谈

- **CoE** 重复的是层内专家处理；**LOOM／MoR** 主要涉及共享 Transformer 堆栈。它们的注意力、KV cache 和训练成本不同。
- **EPO** 的候选“route”主要是一个 MoE 层内的专家集合；本项目若讨论路径顺序，须定义多轮集合序列及中间状态变化。
- **MoUE** 是跨层共享专家与状态路由；“使用历史”已被覆盖，但其状态与本项目每 token 的专家调用历史不是同一对象。
- **TaH2** 对“是否再循环”提供直接监督；若只在它的停止器里多拼一个 history 向量，仍需充分证据证明贡献。

## 2. 自适应计算与参数共享的基础

| 文献 | 作者／时间 | 已核验作用 | 不宜外推的内容 |
|---|---|---|---|
| [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983) | Alex Graves，2016/2017 | 确定、可微的计算步数自适应；摘要 | 不保证所有 LM 任务能获得显著收益 |
| [Universal Transformers](https://arxiv.org/abs/1807.03819) | Mostafa Dehghani 等，2018；ICLR 2019 | 深度递归与逐位置停止；摘要／元数据 | 非固定深度架构本身不是新点 |
| [PonderNet](https://arxiv.org/abs/2107.05407) | Andrea Banino 等，2021 | 停止概率、预测与计算权衡；摘要 | 稠密训练评估多个深度不等于训练节省 |
| [MoEUT](https://arxiv.org/abs/2405.16039) | Róbert Csordás 等，2024；NeurIPS 2024 | 共享层 UT 与 MoE 已结合；摘要／会议元数据 | MoE+共享层组合的新颖性不能重提 |
| [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) | David Raposo 等，2024 | 在固定总体容量内分配 token 计算；摘要 | 跨序列位置竞争要处理因果性和训练部署差异 |
| [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) | Sangmin Bae 等，2025 | 动态 token 递归深度、递归 KV 设计；摘要／作者库 | 把循环 Attention 视为“完全无法实现”不准确，只是首轮工程负担更高 |
| [T-LoopFormer](https://arxiv.org/abs/2609.15160) | Mingqian Yu 等，2026-09，v3 2026-10-14 | token-choice 动态深度和 recursion-wise cache；摘要 | 已有动态深度方案不能漏作相关工作 |

## 3. 路由学习、可微选择和专家分配

| 文献 | 作者／时间 | 已核验作用 | 对研究的具体影响 |
|---|---|---|---|
| [ReMoE](https://arxiv.org/abs/2412.14711) | Ziteng Wang、Jun Zhu、Jianfei Chen；2024/2025；ICLR 2025 | ReLU 路由与自适应稀疏；摘要／正式会议信息 | 不能把非 Top-k 可微选择视为新贡献；作为梯度优化备选 |
| [ProbMoE](https://proceedings.mlr.press/v306/zhao26ap.html) | Heng Zhao、Zilei Shao、Guy Van den Broeck、Zhe Zeng；ICML 2026 | 在基数约束子集上采样，exact marginals 提供代理梯度；正文方法 | 明确“精确边缘概率”不等于“真实任务目标的精确无偏梯度”；是无 RL 强基线 |
| [Routing-Free MoE](https://arxiv.org/abs/2604.00801) | Yilun Liu 等，2026 | 激活逻辑置于专家内部；摘要 | “专家报价／自激活”已有近邻；实现也可能退化成另一种中心评分器 |
| [Routing Manifold Alignment（RoMA）](https://arxiv.org/abs/2511.07419) | Zhongyang Li、Ziyue Li、Tianyi Zhou；2025 | 冻结骨干，用成功邻居的路由改善 router；摘要、方法目录及相关段 | 路由模仿和 router-only 后训练已有先例；与逐 token 多步候选搜索区分 |
| [Awakening Dormant Experts / CoR](https://arxiv.org/abs/2604.14246) | Wentao Hu 等，2026 | 用反事实专家影响信号重分配资源；摘要 | “反事实发现被遗漏专家”不是空白；长尾事实问题与本项目通用 LM 目标不同 |
| [DAG-MoE](https://arxiv.org/abs/2606.01062) | Jiarui Feng 等，2026；作者元数据称 ICML 2026 | 学习选定专家之间的结构化聚合；摘要 | 专家互补／结构化组合须与聚合图先例区分 |
| [MoE Pathfinder](https://arxiv.org/abs/2512.18425) | Xican Yang、Yuanhe Tian、Yan Song；2025 | 利用跨层轨迹图作专家剪枝；摘要 | “把专家放到图上做路径优化”不是创新；其目标是离线剪枝而非循环执行策略 |

## 4. 容量、市场、博弈和最优传输

| 文献 | 已核验作用 | 本项目应如何使用 |
|---|---|---|
| [BASE Layers，ICML 2021](https://proceedings.mlr.press/v139/lewis21a.html) | 平衡线性指派；出版页 | 离线小批精确分配参考；不能无条件用于教师强制整序列 |
| [ALF-LB，2408.15664](https://arxiv.org/abs/2408.15664) | 最近负载驱动专家 bias；摘要 | 每专家价钱路线的必要强基线 |
| [ALF-LB 理论，2512.03915](https://arxiv.org/abs/2512.03915) | 单步原始—对偶解释；摘要 | 博弈路线首先与已有在线优化相区分；不重复证明同一静态结果 |
| [Selective Sinkhorn Routing，2511.08972](https://arxiv.org/abs/2511.08972) | 传输图同时用于分配与权重；摘要 | 如做 OT，需要有明确跨轮价值增量与真实稀疏执行 |
| [Mean Field Equilibria for MoE](https://openreview.net/forum?id=Mtaa2o5aVc) | workshop 身份由[官方接受名单](https://sites.google.com/view/nextgame-icml26/accepted-papers)核对；正文受限索引 | 只作为已存在“博弈解释”的证明，详细结果不构成本方案的性能依据 |
| [Three Phases of Expert Routing，2604.04230](https://arxiv.org/abs/2604.04230) | 拥塞参数解释训练阶段；摘要 | 区分解释性理论和产生更优路由的算法 |
| [Alloc-MoE，2604.08133](https://arxiv.org/abs/2604.08133) | 层级 sensitivity profiling+DP，token 级重分配；正文相关节；元数据标 ACL 2026 main | 预算调度的直接邻近；其 additive sensitivity 目标最优不等于完整网络损失全局最优 |
| [SeqTopK，2511.06494](https://arxiv.org/abs/2511.06494) | 序列总体专家预算及在线缓存分数；正文 §3.2 | offline 与 online 分开比较；实际执行必须有预算账本，不直接继承文中累计界 |

## 5. 规模和系统比较

| 文献 | 已核验作用 | 对 A6000 实验的启示 |
|---|---|---|
| [SMELT，2609.01343](https://arxiv.org/abs/2609.01343) | matched per-token FLOPs、非 embedding 参数和 KV 的循环 MoE 比较；摘要 | 公平比较至少报告三类预算；不要求小团队复现 54B 规模 |
| [Scaling Laws for Looped Mixture of Experts，2609.40316](https://arxiv.org/abs/2609.40316) | 联合稀疏度、递归、参数与数据的 scaling；摘要 | “循环 MoE 的 scaling law”已有人研究，本项目不宜全面争夺该大规模主线 |
| [Read-ME，2410.19123](https://arxiv.org/abs/2410.19123) | pre-gating、专家缓存与 lookahead 调度；摘要 | 预测已选专家以预取，不等于按任务损失搜索更好的专家路径 |
| [Dynamic Experts Search，2509.22572](https://arxiv.org/abs/2509.22572) | 在生成搜索中变化专家数量；摘要和方法相关范围 | 输出搜索预算与内部循环路径预算不同；可作为将来推理扩展，首轮不混入 |
| [Rewiring Experts on the Fly，2510.14853](https://arxiv.org/abs/2510.14853) | 用已生成上下文自监督更新部分 routing logits；摘要 | 在线适应已有近邻；部署适应开销和分布变化应单独计入 |

## 6. 只有概念借鉴，不应误列为内部 MoE 直接基线

[Lookahead Routing for Large Language Models，2510.19506](https://arxiv.org/abs/2510.19506)（Canbin Huang 等，2025）选择的是异构完整 LLM 对请求的服务模型。它预测潜在输出表示以改善 query-level routing，**不是**同一神经网络内部的循环专家分配。它对“预测执行结果后再选动作”有启发，但不能据此断言本项目已被完整完成，也不能拿其 7.7% 作为内部 MoE 可得收益。

[PADD，2606.10369](https://arxiv.org/abs/2606.10369)（Xinyue Peng 等，2026）被一手索引列为 dense-to-MoE 蒸馏及 path-refined policy optimization。本轮该条目的直接打开失败，已核实范围限于一手检索摘要；**不把它列为已读全文或用它的性能数字支撑决策**。正式投稿前列入继续核验清单。

## 7. 最小而有力的基线集合

目标不是复现上面每篇论文的完整规模。对于 3–5 张 A6000，本项目建议在同一 105M 左右骨干内实现以下机制比较，并明确名称加 `-style`：

1. 非循环 Top-2 与固定循环 Top-2；同时给出同参数和同实际调用数比较。
2. CoE-style 逐轮 router 与 residual；控制循环结构差异。
3. EPO-style 一步候选集合偏好学习；控制额外反事实监督的作用。
4. 容量相同的 `h-only`、`h+time/budget/input`、`h+history` 控制；另加 MoUE-style 状态方案的因果实现。
5. 固定轨迹的 entropy／max-prob／margin 停止与 TaH2-style 一步收益监督。
6. ALF-LB 风格价钱更新；如资源分配有增量，再加可部署的预算调度和小规模 OT。
7. 只有在新 router 的优化方式本身成为结论关键时，才加入 ReMoE 或 ProbMoE 的训练比较，避免每条支线都重训。

必须将“本项目的简化机制复现”和“原论文完整实验复现”区别写清。针对 FFN-only 主体迁移 LOOM 的 residual scaling，不应自称复现 LOOM 整块架构。

## 8. 搜索记录与覆盖限制

已检索的关键词族包括：`looped mixture-of-experts`、`adaptive depth looped transformers`、`counterfactual routing`、`Expert Preference Optimization`、`trajectory state router`、`expert path planning`、`routing lookahead`、`expert beam search`、`routing imitation`、`budget allocation MoE`、`SeqTopK`、`ProbMoE`。首先核对给定链接，再对最新近邻、方法词和文献中的参考线索作追溯。

主要证据来自 arXiv 原论文页面及 HTML、PMLR／NeurIPS 正式出版页、作者仓库和 workshop 官方名单；未将二手技术博客当作性能或创新性的证据。没有把“搜索未命中”当成“不存在该方法”。截至此日期，本项目仍不能声称已穷尽未公开投稿、所有 2026 新论文、跨领域控制／元推理文献，或全部作者实现。

## 9. 经一手论文链接确认的代码入口

| 项目 | 官方入口 | 使用原则 |
|---|---|---|
| CoE | <https://github.com/ZihanWang314/coe> | 先核对当前实现与论文公式；移植小模块优先于照搬全部训练栈 |
| LOOM | <https://github.com/hed-ucas/LOOM> | 固定 commit，记录 full-block 与本项目 FFN-only 的差异 |
| MoR | <https://github.com/raymin0223/mixture_of_recursions> | 循环 Attention 扩展时参考 KV 管理 |
| TaH / TaH2 论文所链仓库 | <https://github.com/thu-nics/TaH> | 仓库首页名称／分支可能对应 TaH；需确认 TaH2 具体实现可得性，不能只因有链接便称完整代码已复现 |
| ReMoE | <https://github.com/thu-ml/ReMoE> | 原库基于 Megatron；小模型实现需先验证数值等价 |
| ProbMoE | <https://github.com/HengHugoZhao/ProbMoE> | 区分 subset 分布精确计算与 surrogate gradient |
| SeqTopK | <https://github.com/Y-Research-SBU/SeqTopK> | 对 online 预算做独立单元验证；不以文字保证替代验证 |
| MoEUT | <https://github.com/robertcsordas/moeut_training_code> | 历史架构基线；多处改动可能让机制归因变难 |

这些是检索入口；本轮没有安装、训练或认证它们在用户服务器上的可运行性。

## 10. 交付前新增的直接近邻与定位修订

以下追加保留以上全部旧内容，但进一步收窄研究定位。详细边界见 [04_latest_overlap_and_adjustments.md](04_latest_overlap_and_adjustments.md)。尤其不能将一般性的 oracle—selector gap、搜索蒸馏或反事实停止宣称为本项目首创。

| 文献 | 已核验范围 | 对计划的调整 |
|---|---|---|
| [Dynamic Routing as a New Dimension for Test-time Versatility of LLMs，2610.04751](https://arxiv.org/abs/2610.04751)，Michal Štefánik 等，2026-10-03 | 正文 §3.4、4.3、Appendix A/D/E/G：hindsight／二次采样 oracle、选择偏差、反事实探针和层程序预测器 | 一般 oracle 可学性诊断与负面结果已有先例；其 MoE token router 参数／规则未改，选中专家仍可能随层路径间接变化。二次采样诊断不是本项目可部署策略的普遍 Bayes 界 |
| [How to Loop MoE: Flatten the Experts, Untie the Attention（Foil），2609.35751](https://arxiv.org/abs/2609.35751)，Shouren Wang 等，2026-09-28 | 正文 §2、Appendix J：专家池／层数／循环次数的匹配变换，逐轮独立 attention | 架构扩展时控制 attention 共享和专家池大小；末层 FFN-only 不称复现其完整架构 |
| [LatentLoop，2610.19975](https://arxiv.org/abs/2610.19975)，Zefeng He 等，2026-10-14 | 正文 §2–3、Appendix B：非对称循环、首轮压缩、共享 MoE／独立 attention、因果增量解码、训练时间 | 非对称标度与计算分配本身不是新点；与冻结专家动作价值区别，不能泛称现有循环 MoE 都只测理论 FLOPs |
| [Rethinking Routing Exploration / MirrorMoE，2610.22353](https://arxiv.org/abs/2610.22353)，Bingcong Li 等，2026-10-15 | 正文 §2–3、Appendix B.6：shared experts 与组合探索；镜像选择／原分数聚合 | 对含 shared experts 的预训练最相关；有效贡献减少不等于实际专家调用减少 |
| [PoLar，2606.06574](https://arxiv.org/abs/2606.06574)，Ziyue Li、Yang Li、Tianyi Zhou，2026-06-04 | 正文 §3、Appendix D：MCTS 有效程序与单次预测器 | 离线多步搜索蒸馏已有直接先例；必须给首动作模仿等强对照 |
| [Dr.LLM，2510.12773](https://arxiv.org/abs/2510.12773)，Ahmed Heakl 等，2025 | 原摘要与作者库：冻结骨干、MCTS 标签、逐层 skip／execute／repeat 控制 | 不能将状态依赖的预算路由监督视为未开发领域 |
| [RE-PoLar，2609.31360](https://arxiv.org/abs/2609.31360)，Justus Westerhoff 等，2026-09-25 | 一手摘要索引和[作者库](https://github.com/DATEXIS/RE-PoLar)，正文直取受限：报告单次预测器未重现搜索候选收益 | “oracle 大而学生失败”已有负面复现；本项目需 MoE 特有、可行动的增量证据 |
| [LatentHarness／Counterfactual Policy Distillation，2609.39740](https://arxiv.org/abs/2609.39740)，Xiaoqiang Wang、Suyuchen Wang、Bang Liu，2026-09-30 | 正文 §2.2、3.2：一步潜在动作分支、收益与成本 soft teacher、KL 蒸馏；与完成分支比较 | 反事实动作收益监督／停止已有直接方法；可在同候选标签上加入便宜的一步 soft-policy 控制，不必复现完整 GRPO 系统 |

**更新后的可争取边界：**稀疏 MoE 有序集合在受控计算下的延迟收益是否可预测，以及由专家结构、出口训练、状态漂移或标签成本决定的成功／失败边界。它是待验证的研究问题，不是已经成立的新颖性结论。
