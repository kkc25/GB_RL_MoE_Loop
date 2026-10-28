# Mixtral 可学习性、Oracle 与第二轮适配实验报告

- 代码：`main@971c295`
- 模型：Mixtral-8x7B-v0.1 Base，grouped_mm，冻结主干。
- 数据：开发角色文档；未使用 newtest。
- 新增测试：40/40 通过。

## Oracle smoke

8 个 calibration context，R1 枚举 28 对，R2 在固定原生 R1 后枚举 28 对+STOP，共 464 次评分调用。

- R1：8/8 context 超过 0.005 nat 实用阈值；平均 label-aware headroom `0.308722 nat`，中位数 `0.284068`，范围 `0.077044–0.706422`。
- R2：仅 1/8 context 超过阈值；平均 headroom `0.143128 nat`，但 7/8 选择 STOP（stop fraction 0.875）。

Oracle 使用标签和 singleton counterfactual 评分，不是可部署策略收益，也不能与完整文档 NLL 直接相减。结果支持继续 R1 可学习性 pilot，不支持扩大强制冻结重复 FFN。

## 新采样 R1 pilot

每角色 8 篇、序列长 256；256 个文档平衡随机位置训练 context；64/64 updates 完成。

| Checkpoint | Selection NLL | 相对原生改善 | Greedy 与原路由一致率 |
|---|---:|---:|---:|
| 原生/初始 | 1.6885652605 | 0 | 100% |
| Best（update 56） | **1.6867131842** | **0.0018520764 nat** | 99.8443% |
| Latest（update 64） | 1.6869189291 | 0.0016463314 nat | 99.7405% |

这是首次在 Mixtral 完整 selection 在线评估中观察到正的方向，但改善低于预设 `0.005 nat` 实用阈值，只有约 3/1927 个 token 的 greedy 路由不同，尚无多 seed/置信区间，不能声称显著或 SOTA。0–40 步 greedy 未变，48 步开始变化，56 步最好，64 步略有回落。

## 独立低秩残差适配器

Rank=8、65,536 个可训练参数、32 context、4/4 updates。初始严格等于 R1。

- 初始/best：1.5314759094（best update 0）。
- Latest：1.5315929211，略有恶化。
- 固定 R1/STOP：1.5314759094；未训练原生强制 R2：7.3935911920。

本轮适配器没有 selection 收益。它是新增低秩修正，不是第二次专家调用或 learned STOP，不能用于证明或否定所有 loop 结构。

## 结论

1. 冻结 R1 确实存在很大的标签知情 oracle 空间，但当前策略只实现了很小一部分，可学习性和动作预测仍是瓶颈。
2. 新采样让 greedy 决策真正发生改变，并获得约 0.00185 nat 的开发集改善；但仍低于实用门槛，需要独立 seed 和更广样本验证，不应直接扩大为正式结论。
3. 冻结第二轮 87.5% oracle 选择 STOP；强制重复严重恶化。停止该路线。
4. 4 步低秩适配器 smoke 无收益，不扩大。
5. 下一步优先检查 oracle 动作的合法特征可预测性/蒸馏基线和多 seed R1，而不是盲目增加相同 RLOO 步数或训练强制 R2。
