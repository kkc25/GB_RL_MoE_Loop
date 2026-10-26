# Loop-MoE 当前状态与下一触发条件

**日期：2026-10-26**

**当前工作分支：`research/sota-validation-2026-10-26`**

本文件是当前状态索引；A/B/C/D 报告保留各自阶段的历史“尚未执行”条目，不应再按那些条目判断当前进度。

## 已完成

| 工作 | 当前结论 | 最终证据 |
|---|---|---|
| A：接口与恢复 | raw-h 决策契约、候选映射、exact/price/registry、v2 恢复与真实 packed 校验已修 | `Loop_MoE_Followup_A_Repair_Report_2026-10-26.md`、`runs/next_2026_10_26/readout/` |
| B：廉价基线与成本匹配 | 学习模型未超过修正后的廉价规则；动态未超过实际成本匹配混合 | `Loop_MoE_Followup_B_Repair_Report_2026-10-26.md` |
| C：固定深度质量 | 原 router＋固定 R1 在冻结自然流上非劣；990 文档 ID/988 文本组均通过 | `Loop_MoE_Followup_C_Confirmation_Report_2026-10-26.md`、`runs/next_2026_10_26/plan_c_group_sensitivity.json` |
| D：有界可学性挽救 | candidate-aware 改善 h-only，但未通过部署门槛；停止扩张 | `Loop_MoE_Followup_D_Learnability_Report_2026-10-26.md` |
| R1 读出闭环 | CE 差主要来自 BF16 vs FP32 head，不是专家 kernel 形状 | `Loop_MoE_Readout_Closure_Report_2026-10-26.md` |
| 端到端效率 | 独占 A6000 上 R1 相对 R2 加速 7.48%，五轮均为正 | `Loop_MoE_Efficiency_Profile_Report_2026-10-26.md` |
| 路由×深度机制 | 续算可补偿较差首步，但最终同深度收益仅约 0.0005 nat；不补 T2/T3 | `Loop_MoE_Depth_Interaction_Report_2026-10-26.md` |
| 数据与执行身份 | 实际 packed 字节、config、frozen IDs、cache/backbone、学生和未跟踪源码均纳入校验 | `data/packed_lm.py`、各 run manifest |

## 当前工作默认

> **原 router＋固定 R1。** 仅适用于当前 M105、已评估数据域、BF16 生产实现和 0.005 nat/token 容忍值；R2 保留为质量参照。

质量非劣和约 7.5% 的完整前向效率收益是内部对照结果，不是超越 SOTA 的证明。

## 当前停止的方向

- 不扩 32k 学生，不启动真实 LM RL，不重开 E12。
- 不为路由×深度交互追加 T2/T3；交互量不等于最终部署收益。
- 不重复 C 的 1.1M token 主前向，不把历史诊断 newtest 改称确认集。
- 不直接跳转到 300M 或多卡训练。

## 进入 SOTA/强基线比较前的必要条件

1. 固定当前修复版本，保留完整源码、配置、输入字节和环境身份。
2. 预先指定有实际可比性的同规模强基线、数据、训练预算和评估协议；内部 R1/R2 对比不能替代外部 SOTA。
3. 若提出新控制器，必须在同实际成本下同时超过原 router 和最强廉价规则，并有独立确认。
4. 若扩大骨干，先固定有效 batch、深度分布、dense/MoE 对照、种子和恢复协议，并证明新增机制的增益覆盖其成本。

当前只满足进入“实验设计与强基线确定”的条件，尚未满足“宣布超越 SOTA”的证据条件。

## 100M 强基线对照

- MoE−dense 的 R1 NLL 差：**+0.116340157 nat**（负值有利于 MoE）。
- MoE 相对 dense 的完整 R1 前向速度差：**-74.14%**（正值有利于 MoE）。
- 冻结门槛判定：**stop_or_fundamental_redesign**——停止当前架构的直接扩张，回到机制或训练目标 redesign。

本结果是同数据、同训练预算的内部两组对照，不是外部 SOTA 认证。
