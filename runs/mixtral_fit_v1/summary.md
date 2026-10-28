# Mixtral Fit-first 64-context 实验报告

- 最新代码：`main@fb7ecd6`（PR #10）
- 模型：Mixtral-8x7B-v0.1 Base，grouped_mm，冻结主干。
- 协议：train-only fit → 冻结 train-best 的 selection generalize → 质量通过才 profile。
- 未使用 newtest；generalize 未更新参数或重选 checkpoint。
- 新 fit-first 测试：97/97 通过。

## Smoke

Router 与 adapter smoke 均通过 fit：router 训练 greedy CE 改善 0.15396 nat；adapter train probe NLL 改善 0.31939 nat。只用于验证流程，不作为研究结论。

## 64-context fit

| 方法 | best | 训练结果 | Fit gate |
|---|---:|---|---|
| 全 28 动作 router | update 256 | greedy CE 改善 0.0710327 nat；51/64 context 改变动作 | 通过 |
| Rank-8 adapter | update 256 | train probe NLL 1.8825349 → 0.0265989，改善 1.8559360 nat | 通过 |

PR #10 后两者均在新代码身份下重新生成 fit 产物；旧 PR #9 结果保留。

## 冻结 checkpoint 的 generalize

16 篇 selection 文档、3967 个有效 token。原生与 controlled native 均为 `1.8169399446`。

| 方法 | Candidate NLL | 相对原生 | 预设门槛 | 结果 |
|---|---:|---:|---:|---|
| Router | 1.8442564545 | 恶化 0.0273165 nat/token | 改善 ≥0.001 | **未通过** |
| Adapter | 2.4897283903 | 恶化 0.6727884 nat/token | 改善 ≥0.001 | **未通过** |

Router 在 selection 上只有 14.54% token 保持原生 greedy 路由，说明训练将动作改得非常激进；13/16 文档恶化，3/16 小幅改善。Adapter 16/16 文档全部恶化，单篇恶化 0.4474–0.9964 nat，呈现严重训练位置过拟合。

## 决策

两条路线均未达到 generalization 门槛，**均未执行 profile**。不能以训练拟合收益或潜在速度亮点替代质量通过。

结论：全动作期望 CE 和 rank-8 残差都能拟合固定训练问题，但当前设置不能泛化到独立开发文档。下一步应优先正则化/早停（仅根据 train）、降低路由动作偏移、增加训练覆盖并做多 seed，而不是扩大相同配置或进入部署测量。
