# PR #17 专家逐算子与精度对照报告

- 固定代码：`d33248074ef46fcf874ea756d7668bf5adc3849f`。
- 全量测试：1296 项，1295 通过、1 跳过。
- 新旧冻结清单一致；完成 8 个严格 parity case，仍全部失败，未启动 cache/fit。
- Observer 验证通过：带观察器的 dense/compact 输出与未观测执行逐元素一致，因此算子定位输入有效。

## 首个失败混合掩码的逐算子观察

- 输入和路由固定，抽样前 8 个 CONTINUE token。
- Expert 0 的首次精确/容差差异位于 `gate_up`；随后 down/weighted/accumulation 也出现差异。
- Expert 1 的首次差异位于 `down`。
- `accumulation` 继承前序误差，不能据此认定 index_add 自身出错。
- FP32 参考使用同一已量化输入/权重且 TF32 关闭，不是数学真值或新的验收标准。

## BF16 reduced-precision reduction 对照

原设置为 `allow_bf16_reduced_precision_reduction=true`，诊断中临时改为 false：

| 指标 | 原设置 | 关闭 reduction |
|---|---:|---:|
| Dense 与 compact 不同元素 | 159,456 / 524,288 | 6,280 / 524,288 |
| 最大绝对差 | 16.0 | 8.0 |
| 对公共 FP32 参考最大绝对差 | 7.2717 | 7.2717 |

关闭 reduced-precision reduction 显著减少 dense/compact 差异，但没有消除差异，也仍超过严格容差；临时设置改变不是生产修复，不能自动启用。

## 结论

当前证据进一步支持专家 BF16 算子和归约路径对行数/批形状敏感。Reduced-precision reduction 是重要影响因素，但不是唯一问题；路由和 adapter 已排除为首个差异来源，未发现 observer 自身错误。严格 gate 仍失败，100k 训练仍未开始。下一步需要针对专家精度/归约设计生产修复和独立验收，而不是仅调整容差。
