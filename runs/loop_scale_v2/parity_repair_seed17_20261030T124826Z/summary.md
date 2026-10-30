# 混合门执行一致性增强诊断报告

- 固定代码：`31b796274720f47a437c8f6805726ac953b657f6`。
- 全量测试：1281 项，1280 通过、1 跳过。
- 新旧冻结清单逐角色一致：有效位置、文档数、来源组数及文档 ID 顺序完全相同；最终 test 未评估。
- 完成 4 篇文档 × 2 种幅度共 8 个 case，全部保留严格 parity 失败，未进入 cache/训练。

## 首个失败 case 的分层证据

在幅度 1、更新前、交替混合掩码下：

| 层级 | 结果 |
|---|---|
| Normalized input | 完全一致 |
| FP32 adapter correction | 完全一致 |
| Expert input（修正后） | 完全一致 |
| Router logits | 完全一致 |
| Router pairs（专家编号） | **完全一致** |
| Router weights | 完全一致 |
| Expert output | **首次精确差异和首次容差失败** |
| 回填 hidden/CE | 不同 |

更新后同样首先失败于 expert output。STOP 和全 CONTINUE 通过；交替及首/尾单位置掩码也失败。

## 隔离对照

`fixed_dense_inputs_routes_expert_only` 固定相同输入、专家编号和权重，只改变专家调用行数，仍出现与整体路径相同的 expert output 差异：159,457 个元素不同，最大绝对差 16.0。

`fixed_dense_hidden_head_only` 固定 hidden，仅改变 readout 行数，CE 差为 0，说明本次首个 case 的输出头不是首个差异来源。

## 结论

当前证据不支持 adapter 修正、router Top-2 改变、输出头或掩码回填是首要原因。差异被定位到**专家计算在不同行数/批形状下的数值行为**。这与 BF16 专家 GEMM/分桶形状敏感性一致，但不能把该观察自动等同于已完成根因修复或证明所有 case 只有同一机制。

严格 GPU gate 仍未通过，因此 100k 训练仍未开始。需要修改专家执行/验证协议，而不是放宽阈值或继续扩大数据。
