# 循环 MoE：研究主线与 RTX A6000 实验方案包

**版本：2026-10-25。面向通常可用 3–5 张、最多 8 张 RTX A6000 的研究条件。**

本包包含文献核验、数学论证、逐项实验方案、硬件预算、执行协议及可复算的小型计算。它是一份可交给研究生实施的研究计划；**尚未在你们的服务器上训练模型，所有速度、显存余量与语言模型收益均须实测**。已执行的 CPU 数值检查与待执行 GPU 实验在结果文件中明确区分。

## 建议先读

1. [研究主线与决策树](00_research_roadmap.md)：决定做什么、为什么、什么证据出现后继续。
2. [此前引用核验](literature/01_prior_claims_audit.md) 与 [最近邻文献地图](literature/02_literature_map.md)：避免重复已有工作。
3. [共同实验协议](protocols/01_common_protocol.md)：所有方案共用的模型、数据、因果性、公平比较和统计规则。
4. [硬件与预算](resources/01_a6000_capacity_and_budget.md)：参数、FLOPs、显存与 GPU 小时的可复算估计。
5. [E00 基线与可行性](experiments/E00_feasibility_and_baselines.md)、[E01 可学性诊断](experiments/E01_oracle_and_learnability.md)、[E02 多步价值学习](experiments/E02_counterfactual_rollout_distillation.md)。

## 实验清单

“可行”表示能够在小规模、有限预算下检验，不表示全部都值得训练，更不表示都应塞入同一篇论文。

| 编号 | 独立方案 | 定位 | 开始条件 |
|---|---|---|---|
| E00 | [基础模型、资源校准与强基线](experiments/E00_feasibility_and_baselines.md) | 必做 | 立即 |
| E01 | [反事实空间与可部署收益诊断](experiments/E01_oracle_and_learnability.md) | 必做 | 有合格冻结检查点 |
| E02 | [基准策略 rollout 价值蒸馏与两步规划](experiments/E02_counterfactual_rollout_distillation.md) | 推荐论文核心 | E01 的样本外可学性通过 |
| E03 | [收益校准的自适应停止](experiments/E03_value_calibrated_halting.md) | 主线扩展 | 基础循环确实有用 |
| E04 | [路径记忆、专家转移图与充分状态诊断](experiments/E04_memory_and_transition_graph.md) | 机制与必要消融 | 同一冻结骨干/相同输入接口 |
| E05 | [价格、拥塞势博弈与硬容量分配](experiments/E05_prices_and_congestion_games.md) | 条件系统扩展 | 实测负载瓶颈存在 |
| E06 | [因果容量约束的传输／匹配](experiments/E06_causal_transport_assignment.md) | E05 的替代路线 | 可同时观测的请求前沿足够大 |
| E07 | [跨请求的预算分配与收益指数](experiments/E07_budget_scheduling.md) | 停止／调度替代路线 | 多请求服务负载存在 |
| E08 | [稳定性与合成机制实验](experiments/E08_stability_and_synthetic_mechanisms.md) | 低成本理论支线 | 可与 E00 同时做 |
| E09 | [冻结专家后的策略梯度对照](experiments/E09_router_only_policy_gradient.md) | 备选，不是前置条件 | 监督方法有明确受限证据 |
| E10 | [有序专家互补性](experiments/E10_ordered_complementarity.md) | 小型机制筛查 | E01 已有可信反事实数据 |
| E11 | [扩大循环范围与实际延迟验证](experiments/E11_systems_and_scale_validation.md) | 效率论文必须完成 | 单末层主机制通过 |
| E12 | [专家自报价与等价性检验](experiments/E12_expert_self_bidding.md) | 可选结构筛查 | E01 标签可复用 |

E00–E12共13份独立实验方案。自报价先做等价性检查，避免把普通多输出Router的重新参数化当成新架构。

## 理论、执行与复算材料

- [形式化与命题](theory/01_formalization_and_propositions.md)
- [反例与可证伪条件](theory/02_counterexamples_and_falsification.md)
- [数据与评测规范](protocols/02_data_and_evaluation.md)
- [实验登记、预算与报告模板](protocols/03_run_registry_and_budget.md)
- [可复现执行与动态 GPU 数量](resources/02_reproducible_execution.md)
- [最近邻与可学性专项核查](literature/03_nearest_neighbor_and_learnability.md)
- [最后补查的近期重叠与路线修正](literature/04_latest_overlap_and_adjustments.md)
- [最小配置](configs/core_experiments.yaml)：说明性配置，不冒充已实现训练器。
- [独立反例与组合计数脚本](code/verify_design.py)
- [CPU 检查结果](results/design_checks.json)
- [交付检查与已知局限](DELIVERY_NOTES.md)

## 这份计划最重要的约束

先判断“额外计算是否存在可学、可部署的收益”，再研究分配机制。真实标签挑出的最优路径只是诊断上界；更长路径、更多搜索、更多标签构造成本，都必须进入计算账本。Top-k 可以保留为最后的稀疏执行器；需要检验的是它所依据的分数是否应从常规Router评分改为显式续算价值监督。

本文献核验截至上述日期；近期预印本的结论按作者报告记录，不视为独立复现。各文件使用稳定原始 URL，离线解压后仍可识别来源。压缩包不附第三方论文全文或未核验的模型权重。
