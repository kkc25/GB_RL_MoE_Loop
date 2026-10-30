# PR #14 真正第二轮专家 FFN 与硬门实验报告

- 原始固定提交：`336db2abfab962afa226feb46029d4a4e500c599`。
- 首次 run 完成训练和门拟合，但在逐 token 证据导出时因整数字典键报错，保留 failed 记录。
- 导出修复提交：`3559a307cb8af347444b09444044bfac31084cc0`；修复后全量测试 1200 项，1199 通过、1 跳过。
- 修复后使用全新 run 完整重跑，状态 complete；未访问 newtest。

## 训练与数值验证

- 真实 eager 后端反传通过，backbone 无梯度，adapter up-gradient norm 为 0.91356。
- R2 与参数匹配的一遍控制均完成 8 遍覆盖：每位置 min visits=8，context visits=46,872。
- 总 optimizer updates=2,960；硬门额外完成 512 updates。
- 训练局部收益资格门通过（eligible=true）。
- 修复后的证据导出包含 680 条记录、9,355,902 bytes，SHA256 `b97809f679e43754ad49c21115c7ce25465d15b4ad75ec5a501f999139060922`。

## Development 结果

| 路径 | Development NLL/结论 |
|---|---|
| Native/STOP | 1.9741908402 |
| 训练后 R2 best | 2.2567136345 |
| 参数匹配的一遍低秩控制 best | 1.9642836026 |
| 硬门候选 | 最终选择全 STOP，continued_tokens=0 |

真实第二轮训练能拟合局部训练位置，但其 development 表现比 native 差；参数匹配的一遍低秩控制反而改善约 0.00991 nat。硬门没有获得可选择的正收益，最终 `scientific_outcome=no_gate_selected`。

## 决策

按预设关卡，**不执行 selection evaluate/profile**。本轮不能宣称真正第二次专家 FFN 或硬门改善质量，也不能把训练局部收益或门开启率当作泛化收益。与此前结果一致，新增低秩修正有开发收益，但真正重复专家计算尚未证明独立价值。
