# R1 最终 hidden 与 head 精度闭环报告

**日期：2026-10-26**

## 结论

R1 的 BF16 决策闭环继续通过：test/validation 的 baseline、stop/continue 和 rounds 不一致数均为 0，缓存 pre-FFN h 差为 0，最大学生 delta 差约 1e-5。

此前约 0.148/0.195 nat 的生产 CE 与缓存成本差，主要来自 **BF16 head 与 FP32 head 的精度差异**，不是专家 kernel 形状。旧“差异主要由后续专家 kernel 形状造成”的解释已由直接对照替代。

## 修复

- 附加 FP32 readout 改用选定轮数后的 `trace["hidden"]`，不再误用 pre-FFN `trace["h"]`。
- `readout_ce` 显式关闭 autocast，保证 final_norm、 tied head 和 CE 的 FP32 契约。
- 同一 final hidden、同一逐状态 `(1,V)` head 形状下比较 FP32/BF16。
- 分别记录生产 final hidden、replay final hidden、两条路径逐轮动作、hidden hash 和 full-batch 生产 CE。
- 执行身份增加未跟踪 Python 文件内容 hash 和入口文件 hash。

## 关键数值

| 最大差异 | test | validation |
|---|---:|---:|
| 修正后 production final hidden＋FP32 head vs offline | 0.01082 | 0.04209 |
| 原 full-batch BF16 生产 CE vs offline | 0.14759 | 0.19498 |
| 同一 hidden：FP32 vs BF16 head | 0.14762 | 0.19535 |
| 逐状态 BF16 head vs full-batch CE（形状效应） | 0.00000 | 0.00058 |
| production vs replay hidden，FP32 head（路径数值） | 0.01081 | 0.04210 |
| replay final hidden＋FP32 head vs offline 缓存成本 | **0.00077** | **0.00170** |

production/replay final hidden 的最大绝对元素差仍存在，但逐轮动作不匹配状态为 0；这些差异属于单行重放与完整 batch BF16 执行的数值边界，没有改变本轮路由决策。

## 边界

- 仍只覆盖原缓存的 2 条 packed 序列和少量文档，不推广到任意输入、batch 或 dtype。
- 逐轮动作由 pre-FFN h 重放重建；round-1 与正式 trace 全部一致，重放与生产 hidden 差另行记录。
- 不重跑方案 C 主 NLL；F1/F2 的附加诊断修复不改变 C 的完整生产评估结果。

## 证据

- `runs/next_2026_10_26/readout/r1_bfloat16_test/`
- `runs/next_2026_10_26/readout/r1_bfloat16_validation/`
- `runs/next_2026_10_26/readout/targeted_tests.log`
- 实现：`experiments/r1_closed_loop.py`、`eval/counterfactual.py`
