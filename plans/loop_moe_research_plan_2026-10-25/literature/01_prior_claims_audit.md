# 前次讨论的文献与论断审计

核验日期：2026-10-25。范围：用户提供的前次对话中可见的文献与主要推断，以及会实质改变研究方案的新近直接先例。

## 1. 审计结论

此前可见对话中的具体 arXiv 编号均已在一手论文页面找到与标题一致的记录，**本轮没有发现这些编号被错配到无关论文**。需要修正的主要是新颖性边界、论证强度与实验公平性，而不是把 2026 年文献一概判作不存在。

但是，继续检索发现此前未充分讨论的近邻：TaH2 已有收益监督停止；MoUE 已有轨迹状态路由；Sparse Layers、SMELT 和 Loop Scaling Laws 已系统研究循环 MoE 的计算比较。因此，“MoE + Loop + 路径记忆 + 非 RL 停止”整体不能直接作为创新声明。研究应缩小到：**等计算预算下，有序多步专家路径的可学习价值，何时超越一步路由改进，并能否在真实执行中兑现。**这是根据下面文献作出的本项目判断，尚不是已被实验支持的结论。

### 证据等级

- **A：一手元数据、摘要与相关正文段落核验。**不表示全文所有推导均被复现。
- **B：一手摘要／会议出版页核验。**只支持摘要范围内的叙述。
- **C：一手网站索引或官方名单核验，正文访问受限。**只保留已能确认的狭窄事实。
- 本轮没有复现任何论文训练结果；文献报告的收益不能作为本项目或 RTX A6000 的实测结果。

## 2. 前次具体引用逐项核验

| 原讨论引用 | 作者、时间与版本 | 核验范围 | 可以保留的判断 | 必须收紧的判断 |
|---|---|---|---|---|
| [When Are Experts Misrouted? Counterfactual Routing Analysis in Mixture-of-Experts Language Models](https://arxiv.org/abs/2605.07260) | Youngsik Yoon、Siwei Wang、Wei Chen、Jungseul Ok；2026-05-08，v1 | A；摘要、§3、§5、局限 | 冻结模型中存在等专家数的更优替代选择；提出最终层 router-only 的 EPO | 主要干预是同一层的专家集合，不是循环中的有序路径规划；使用真实后续 token 的 oracle 差距不等于可部署收益 |
| [Bridging Game Theory and Transformer Routing: Mean Field Equilibria for Mixture of Experts](https://openreview.net/forum?id=Mtaa2o5aVc) | Nevroz Sen；2026-06；NExT-Game@ICML26 workshop | C；官方接受名单及 OpenReview 索引，正文打开遇验证页 | “MoE 路由的平均场博弈解释已有先例”；[官方名单](https://sites.google.com/view/nextgame-icml26/accepted-papers)确认 workshop poster | 不能称 ICML 主会论文；索引能核对 1.65× 开销字样，但本轮未独立核查完整实验条件，不将其当通用成本系数 |
| [Three Phases of Expert Routing](https://arxiv.org/abs/2604.04230) | Charafeddine Mouzouni；2026-04-05，v1 | B | 拥塞博弈用于解释负载平衡的训练演化 | 摘要明确单类型模型预测几乎与温度 softmax 相同，不能引用为“博弈路由必然更优”的证据 |
| [BASE Layers](https://proceedings.mlr.press/v139/lewis21a.html) | Mike Lewis 等；ICML 2021，PMLR 139 | B；正式出版页 | 平衡线性指派是成熟先例 | “全局匹配”或“均衡容量”本身没有新颖性；训练指派与自回归部署须另行核对 |
| [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) | Lean Wang 等；2024-08-28，v1 | B | 专家偏置按近期负载更新，已有低成本平衡机制 | 不能把“动态价格／偏置”本身包装成新算法 |
| [A Theoretical Framework for Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2512.03915) | X. Y. Han、Yuan Zhong；2025-12-03，v3 为 2026-04-26 | B | 将 ALF-LB 写作原始—对偶更新，研究确定性与在线设置 | 条件化 regret／平衡性质不自动覆盖带非凸隐藏状态演化的整个网络 |
| [Selective Sinkhorn Routing](https://arxiv.org/abs/2511.08972) | Duc Anh Nguyen 等；2025-11-12，v2 为 2026-06-04 | B | 最优传输和 Sinkhorn 用于稀疏专家分配已有直接先例 | “动态 OT”必须说明跨轮状态、代价与稀疏执行的新增内容；不能只改名 |
| [Path-Constrained Mixture-of-Experts](https://arxiv.org/abs/2603.18297) | Zijin Gu 等；2026-03-18，v2 为 2026-04-03 | B；元数据标注 under review | 跨层专家路径有结构；共享 router 限制路径空间 | 路径聚类、路径约束、共享 router 均已有先例；与循环内的同池重复调用仍需区分 |
| [Routing-Free Mixture-of-Experts](https://arxiv.org/abs/2604.00801) | Yilun Liu 等；2026-04-01，v1 | B | 专家内部激活、自主竞争及可微分配已有先例 | “去掉中央 Router”不是空白；也不能因连续表达就推断 GPU 稀疏执行高效 |
| [Chain-of-Experts](https://arxiv.org/abs/2506.18945) | Zihan Wang 等；2025-06-23，v1 | A；§3–4 与消融 | 单层 FFN 内顺序调用共享专家、逐轮独立 router、内部 residual 是直接先例 | 我们的 FFN-only loop 设计必须承认是 CoE 家族；不能将顺序专家组合本身作为新贡献 |
| [Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts](https://arxiv.org/abs/2610.01153)（LOOM） | Di He 等；2026-10-01，v1 | A；方法、计算比较、消融 | 残差缩放、输入再注入、逐轮 router、历史 residual 与分段反传是已公开组合 | LOOM 循环整块 Attention+MoE，与 FFN-only 需作机制移植对照；near-iso-FLOP 和非等 FLOP 结果必须分开 |
| [Adaptive Computation Time](https://arxiv.org/abs/1603.08983)（ACT） | Alex Graves；2016-03-29，v6 为 2017-02-21 | B | 可微自适应计算并不依赖 RL | ACT 不是本研究的创新；不能凭“可微”保证新设置中训练稳定 |
| [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407) | Andrea Banino、Jan Balaguer、Charles Blundell；2021-07-12，v2 为 2021-09-02 | B；元数据标 workshop | 学习停止分布并兼顾计算代价已有先例 | learned halting 不是新问题；训练中加权所有深度的损失也未必减少训练 FLOPs |
| [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)（MoR） | Sangmin Bae 等；2025-07-14，v3 为 2025-10-25 | B；作者仓库标 NeurIPS 2025 | 参数共享、token 动态递归深度与 KV 方案已有直接先例 | 本项目应具体声称专家身份／顺序选择的增量，而非宽泛的 token 自适应循环 |
| [Adaptive Depth in Looped Transformers](https://arxiv.org/abs/2607.20519) | Andrei Cristian Popescu、Haitz Sáez de Ocáriz Borde、Pietro Liò；2026-07-08，v1 | A；方法及 Appendix H | 轨迹训练和退出读出应被分别诊断；冻结轨迹 gate 与置信度基线很重要 | 不能概括为“作者证明一切设置必须解耦”；它提供的是所测模型中的实证诊断 |

## 3. 对前次概念表述的必要修正

### 3.1 Top-k 算子简单，并不代表目标必然短视

Top-k 是最终选择算子。其分数网络可由最终任务损失训练，输入隐藏状态也可能已编码历史。因此不能把所有 top-k router 都等同于“只最大化当前一步真实收益”。应比较三件分开的东西：**分数的监督来源、输入信息、离散选择算子**。一种多步价值模型最终也完全可以用 top-k 执行；研究对象是分数有没有学到有用的延迟收益。

### 3.2 历史路径有信息，需要条件化检验

不能从“不同路径产生不同计算”推出“在给定当前完整状态后，历史仍有新增价值”。如果受控计算的完整状态确为马尔可夫充分状态，则历史没有额外最优决策信息。有限维表示、有限容量读出器、被省略的层／轮次／预算／原始输入、或非马尔可夫机制，才可能使历史表现出作用。实验必须把这些混杂因素分开，不能只比较小型 `h` 线性头与更大 `h+history` MLP。

### 3.3 单轮势博弈不等于多轮系统稳定

固定状态、共同的可分离拥塞成本、有限动作下的 Rosenthal 型势函数，支持的是该简化分配博弈的性质。跨轮隐藏状态变化会改变效用和博弈；同步最佳响应也不同于异步严格改进。有限轮价钱更新更不自动给出当轮硬容量可行性。具体假设与可证明结论应见理论文件，不从博弈名称推出保证。

### 3.4 “市场中的长期价值”应有可校准单位

传统 router logit 的绝对尺度不是天然的损失下降，不能直接和 FLOPs、毫秒、拥塞价格相减而不标定。建议先以 held-out 反事实数据回归 nat/token 的收益，再用明确的乘子进行预算权衡；同时把 task-value 和系统成本分开估计。只学习现有 router 的分数，不能自称得到任务长期价值。

### 3.5 顺序路径的候选数量

对于 8 个专家、每轮 Top-2 且轮内无序，一个动作有 `C(8,2)=28` 种；两轮有 `28^2=784` 条有序“专家集合序列”，四轮有 `28^4=614656` 条。`3^2=9` 仅在每层预先限制为 3 个候选动作时成立。“3 个候选专家”与“3 个候选 Top-2 集合”不同。该计数是本项目的组合计算。

### 3.6 Oracle 改进并不保证可学习、可部署

使用真实标签从多个候选中挑最优，会产生一个知晓实现标签的最优选择。它既可能反映真实的上下文可预测机会，也可能只利用样本级噪声。必须报告 oracle gap、学生能恢复的 gap、未见文档上收益、真实前向计算成本，而不是仅给出 oracle 最优损失。

## 4. 新颖性声明应如何改写

可以作为**待验证假设**：

> 在相同专家调用数与端到端推理预算下，用训练期有限范围的有序路径反事实监督学习一个因果路由策略，能否获得超越一步集合偏好优化、已有轨迹状态路由和已有停止读出的收益？该收益在何种状态信息、路径互补性与成本条件下存在？

目前不应写成论文贡献的说法：

- 首次将 MoE 与循环结合。
- 首次把历史路径加入 router。
- 首次用反事实专家选择训练 router。
- 首次不用 RL 学习停止。
- 首次用博弈／价格／OT 达成专家负载均衡。
- 8 张 A6000 可以直接替代 8 张 H200 获得相同训练速度。

本轮检索不是全球不存在性证明。正式投稿前应再次更新最近邻和公开代码；若多步学习不超过强的一步基线，应收敛成机制诊断与负结果论文，而非堆叠新模块掩盖阴性结果。
