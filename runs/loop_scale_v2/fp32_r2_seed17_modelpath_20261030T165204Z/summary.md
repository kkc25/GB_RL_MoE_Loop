# FP32 R2 候选 Precision-screen 报告

- 固定代码：`4628a0e03074a8993fbeb43f87110b62629cdd9f`。
- 全量测试：1318 项，1317 通过、1 跳过。
- 新旧冻结清单一致；完成 4 文档 × 2 幅度 × 2 精度共 16 个 case。
- `numerical_checks_passed.native_bf16=false`，`numerical_checks_passed.fp32_experts=false`。未启动 cache/fit。

## FP32 的改善与残余失败

FP32 显著降低专家差异，但没有通过完整 parity。首个 case（幅度1、更新前混合掩码）从原 BF16 hidden 最大差 32 降至 0.125，但仍有 395 个不同元素；singleton-last hidden 最大差 8.0、CE 最大差 0.0521。

更新后，该 case mixed hidden 最大差 0.25，singleton-last hidden 最大差 2.0。其他 FP32 case 也存在 hidden/CE 超差，说明只提高专家精度不能消除 native BF16 边界和其余形状相关问题。

梯度结果并非全部通过：部分 case 在更新前/后均通过（如 case 9、10），部分更新前通过但更新后失败，部分两阶段均失败。所有完成 case 的主干参数版本保持不变。

## 成本

FP32 case 每例约 2.89–3.17 秒，native BF16 约 1.48–1.87 秒，包含多次正确性检查和反传，不是独立推理吞吐。末层设备 allocated 从 native 约 17.87–17.87 GiB 增至 FP32 约 18.31 GiB，说明候选增加显存开销；没有 OOM，但这不证明完整训练预算可行。

## 结论

FP32 是有效的减差手段，但不是完整修复：hidden/CE 和梯度严格检查仍未全部通过，成本也更高。这支持“专家 BF16 是主要差异来源”，同时证实 native BF16 残差/head 和其他形状效应不能忽略。不能把 FP32 筛查称为成功，也不能启动 100k 训练。下一步需要处理剩余 BF16 边界或采用明确的新固定形状/混合精度协议。
