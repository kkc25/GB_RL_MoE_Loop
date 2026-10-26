# Loop-MoE 方案 D：有界可学性诊断报告

**日期：2026-10-26**

**结论：冻结的 development 门槛未通过，停止当前骨干/接口的可学性扩展，不扩 32k，不进入真实 LM RL。**

## 1. 有界比较设计

本轮只改变候选表示/评分结构，两支模型学习同一目标：

`A(s,a) = Q^pi0(s,a) − Q^pi0(s,a0)`。

- **h-only：**raw h 输入，四候选 advantage 输出，hidden=128，82,692 参数。
- **candidate-aware：**raw h、真实 expert-pair ID embedding、router logits、候选对 score/margin，共享标量评分器，hidden=120，79,769 参数。
- 训练规模：2,000 和 8,000 states。
- 学生 seeds：0、1、2；不是三个骨干 seeds。
- 相同 AdamW、lr=0.001、weight decay=0、batch=256、10 epochs、固定最终权重。
- 最强廉价规则按 selection 选择；test 不参与扩展决定。

训练前门槛已冻结在 `plan_d/d_plan.json`：8k 时三个 candidate-aware seeds 都必须在 selection 上同时超过 0、对应 h-only 和最强廉价规则，并且 8k 均值优于 2k，才允许另开扩展与独立确认。

## 2. Selection 学习曲线

| 训练规模 | 最强廉价规则 | candidate-aware seed 0 | seed 1 | seed 2 | 三 seed 均值 |
|---|---|---:|---:|---:|---:|
| 2k | −0.000205 | +0.000152 | +0.000229 | +0.000092 | +0.000158 |
| 8k | +0.000411 | +0.000926 | −0.000187 | −0.001588 | **−0.000283** |

8k 的最强廉价规则为修正后的固定全局动作。只有 seed 0 同时超过 0、h-only 和最强廉价规则；seed 1/2 未通过。均值从 2k 到 8k 没有改善，反而转负，因此学习曲线门槛也未通过。

## 3. 已查看 newtest 的诊断

以下为诊断，不参与门槛：

| 训练规模 | 方法 | seed 0 | seed 1 | seed 2 |
|---|---|---:|---:|---:|
| 2k | h-only | −0.005655 | −0.002431 | −0.004670 |
| 2k | candidate-aware | +0.000637 | −0.000888 | −0.001198 |
| 8k | h-only | −0.003581 | −0.002558 | −0.001781 |
| 8k | candidate-aware | −0.002245 | −0.002345 | −0.001548 |

candidate-aware 在部分对照中改善了 h-only 的负结果，但 8k 的三个 test 结果仍全部为负。seed 0 的 selection 正值也没有复现为 test 正值。不能据此宣称表示修复已经产生可部署收益。

## 4. 停止决定

- 不扩展到 32k；现有训练文档和预登记门槛也不支持自动扩张。
- 不打开新的确认集，不把当前 newtest 改称确认集。
- 不进入一步 vs 多步续算价值比较或真实 LM RL。
- 方案 C 的固定 R1 非劣结果不受影响，继续作为当前 M105 的工作默认。
- 本次只是一轮限定的表示/优化诊断，不证明所有 candidate-aware 结构或其他骨干都不可学习。

证据：

- `runs/followup_2026_10_26/plan_d/d_plan.json`
- `runs/followup_2026_10_26/plan_d/d_summary.json`
- `runs/followup_2026_10_26/plan_d/run_manifest.json`
- `runs/followup_2026_10_26/plan_d_console.log`
