# Loop-scale v2 准备与 Preflight 结果

- 固定代码：`ec7f78a79ef6b9bbaaa098b2da76f3d3a5f2b2a1`。
- 全量测试：1269 项，1268 通过、1 跳过。
- 新候选语料：8,000 篇显式 `id/source_id/text` 文档；prepare 完成并冻结，未使用旧 56 篇文档或旧 cache。

## 冻结数据

- 最大 adapter-train：900,059 有效位置、3,667 来源组。
- 最大 gate-fit：100,127 位置、407 来源组。
- Development calibration：50,231；selection：50,072。
- 最终 test：100,025，继续封存。
- 100k/300k/1m 嵌套清单及 freeze hash 已生成。

## Preflight 失败

模型与 eager 后端加载正常，主干无梯度，真实 adapter 反传非零。STOP 零额外专家与全 CONTINUE 通过，但混合位置门的紧凑执行与“全部计算后按门掩码”不一致：

- CE 最大绝对差：0.094923，71/255 个位置不同；
- Hidden 最大绝对差：16.0，109,130 个元素不同；
- 预声明容差：atol=rtol=1e-4；
- `conditional_parity.passed=false`，因此整个 preflight failed。

这是条件执行数值/实现检查失败，不是训练质量结论。按协议未执行 cache、100k、300k 或后续训练，未放宽门槛、未切换后端、未缩小数据。
