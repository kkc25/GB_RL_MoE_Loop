# Mixtral RL + Loop + MoE 有界测试报告

- 模型：Mixtral-8x7B-v0.1 Base，冻结全部主干，末层 8 专家 Top-2。
- 后端：grouped_mm；M0 已验证受控 R1 与原生逐位一致。
- 数据：2 篇 selection 文档、254 个有效 token；未使用 newtest。
- 本轮是开发集 smoke，不是正式 R2 训练或统计质量确认。

## R1 RL

64/64 updates 完成，RLOO、4 contexts × 4 samples、32 个有界训练 context。best 为 update 0；0、8、16、24、32、40、48、56、64 步的 selection greedy NLL 均为 `1.5314759093986592`，没有可见学习收益。

Best actor 独立在线评估：原生 164.455 ms，learned 165.562 ms，actor/特征开销约 1.107 ms（0.67%），无对应质量收益。

## Loop 探针

将 R1 actor 直接接入末层第二轮 loop；这不是 R2 训练。

| 路径 | NLL | 平均轮数 |
|---|---:|---:|
| 原路由 R1 | 1.5314759094 | 1 |
| 原路由强制两轮 | 7.3935911920 | 2 |
| Best actor R1 | 1.5314759094 | 1 |
| Best actor 直接两轮（未训练 STOP/R2） | 7.3935911920 | 2 |
| 原生第一轮后 STOP | 1.5314759094 | 1 |

强制第二轮使 NLL 增加约 `5.8621 nat`，严重破坏质量；best actor 因初始化并停留在原生路由，第二轮结果与强制原生第二轮相同。STOP 路径正确恢复到 R1 质量和 1 轮执行。

## 结论

- Grouped_mm + R1 快速路径的执行正确性与此前 1.35× 相对 eager 的速度优势成立。
- 本轮 64 步 R1 RL 没有学习收益；未训练的额外 loop 不但没有收益，反而严重损害 Mixtral 原模型输出。
- 当前不应扩大 R2 训练或声称 RL+loop 改善质量。若要继续研究 loop，需要先分析第二轮路由/残差目标并设计显式 STOP 的 R2 训练与固定 R1 质量门槛，不能把 R1 actor 直接两轮执行当成正式 R2 方案。

产物：`train_grouped_r1_u64/`、`profile_grouped_r1_best/`、`loop_r2_probe/report.json`。
