# E09：冻结专家后的策略梯度对照

**备选路线；不作为主项目启动条件；不在3–5卡上默认训练完整LLM GRPO。**

## 何时值得做

只有E01显示可学空间、E02已完成候选覆盖/状态漂移补救，却仍有明确长期信用分配或探索瓶颈时，才投入此对照。若监督方法已经稳定且便宜，RL不是必要步骤。

路由RL已有 [Exploration-Driven Reinforcement Learning for Expert Routing Improvement in Mixture-of-Experts Language Models](https://aclanthology.org/2025.findings-emnlp.1282/) 和 [Expert-Space Exploration in MoE Reinforcement Learning](https://arxiv.org/abs/2609.13058)。论文“用了RL”不构成创新；要回答的是**相同专家冻结、相同额外训练计算下，策略梯度是否提供监督反事实方法没有的收益**。

## 最小可行环境

- 专家、attention、embedding/head、norm全部冻结；只训练小路由controller，初始化为E02学生：π_ref(a|s)=softmax(−Qhat(s,a)/T_policy)，temperature在validation上选定并锁定，明确成本越低概率越高。
- 最后FFN块，Tmax4，每步Top-2集合为一个离散动作；先固定深度，后加stop。
- 状态是h,t,budget；动作空间28集合或事先合法裁剪池。选择整集合的分类分布，使logprob定义明确；不把两个独立有放回expert抽样误当无重复Top-2。
- reward首先用teacher-forced终局 \(-\ell(y)-\lambda C\)。y是训练奖励，不是部署输入。此环境不生成长CoT，计算量远小于完整序列GRPO。

## 算法与步骤

1. 对同一合法初始状态采样4条路径；每条路径最多4轮，总专家calls逐条计入。
2. 使用REINFORCE配状态baseline，或leave-one-out的同状态回报均值作为baseline；后者对该样本独立，避免把本样本纳入baseline带来的缩放/偏差解释问题。
3. loss为 \(-\sum_t\log\pi_\theta(a_t|s_t)\operatorname{stopgrad}(R-b)\)，加适度KL至初始π_ref与可选熵正则。
4. 初始lr候选{1e-4,3e-4}；KL权重{0,0.01,0.1}只在pilot小网格，正式锁定。clip grad1。固定总rollout预算与候选CE评估数。
5. rollout保存实际动作序列、旧logprob、states、reward和骨干hash。若多epoch使用旧数据，需合适importance ratio/PPO目标并报告偏差；首版每批一次更新，避免不必要off-policy复杂性。
6. 每1000更新评估固定验证样本；达到总GPU-hour上限或连续3次无改善停止。所有超参失败也计成本。
7. 推理取greedy controller，额外做采样策略对照；采样策略必须固定temperature并计多次尝试费用。

## 等预算对照

| 方案 | 使用同样额外资源 |
|---|---|
| E02监督价值学习 | 同样次数的候选终局评价 |
| 一步EPO-style偏好 | 同样候选/标注GPU小时 |
| 更大候选池监督 | 检验RL仅因探索多 |
| router端到端CE直传 | 检验普通梯度是否足够 |
| 随机探索+模仿最优候选 | 强而简单的搜索蒸馏 |
| REINFORCE warm start | 本实验 |
| REINFORCE scratch | 仅小pilot，判断初始化依赖 |

主协议冻结混合权重且只有hard集合选择时，CE对controller没有可用梯度；“直传”对照必须另开允许可微gate-weight更新的组，并与原式EPO重算门权组公平比较。“直传”只能对实际可微混合权重/选中logits求梯度；hard未选动作没有自动梯度。使用straight-through等代理时必须说明其偏差，不声称离散策略的精确梯度。

## 是否需要真实任务终局奖励

如果想测试CE无法表达的组合正确性，先用E08可精确判分的有限算法任务；奖励为answer exact-match或程序执行成功。把输出长度限制在16/32/64tokens，最大采样数4，仍冻结骨干。

105M语言模型如果无法完成任务，应先有能力基线，否则0/1稀疏reward不能回答路由学习问题。大型数学推理GRPO不在默认计划内；也不能让RL额外使用解题verifier而监督对照没有相同信息。

## 评价与资源

固定预算的NLL/任务正确率、每GPU-hour样本效率、梯度方差、KL漂移、专家熵与repeat rate、跨seed方差、rollout/更新状态一致性。专家熵更高不自动代表更好。

第一阶段1卡，三卡可平行seeds；预设最多相应冻结骨干训练成本的10%或48 GPU-hours/方法，取较小值。这是止损上限，不是预测耗时。

若RL只胜弱Top-k而不胜同预算E02/EPO-style，结论是当前设置无需RL。若RL只在合成延迟奖励有效，范围限定于该环境。若有效但训练更贵，给出所需部署调用量的摊销区间，不隐藏成本。

输出：等评估次数/等GPU小时学习曲线、reward分布、方差和成功/失败原因、精确rollout账本。
