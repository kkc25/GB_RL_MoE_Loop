# Router 二进制监督产物诊断报告

- 诊断代码：`d51d9fc`（独立 worktree `diagnose/router-artifact-margins`）。
- 输入：原始全文 router pilot 的 report、186MB train surrogate table、latest/best checkpoint，来自同一 run。
- 只读 CPU 有效行计算，5,859 个训练位置；不加载模型/tokenizer、不执行 GPU、不重训、不重新选 checkpoint、不访问 newtest。
- 诊断测试：17/17 通过。

## 标签确实偏好替代动作

- 5,565/5,859（94.98%）位置的唯一 soft-target argmax 不是 native。
- 5,673/5,859（96.83%）位置存在成本更低的替代动作。
- native 仅为 186 个位置的精确成本最小者（含 ties），196 个位置在 1e-6 容差内近似最优。
- 因此不能把没有路由变化归因于“标签全都太平”或“没有替代收益”。这些成本是 uniform-action 监督 surrogate，不是 mixed-policy 部署收益。

## Latest 的参数变了，但没有跨过 greedy 决策边界

- update 0 的 native 与最佳替代 logit margin：2.079442。
- update 192 的同一 margin 平均降至 1.347724，最小仍为 0.406777。
- 5,859 个位置的数学 argmax 改变数为 0，near tie 为 0。
- Candidate native-reference KL 为 0.036869，说明 actor 分布确实变化，不是参数没有更新。
- Best（update 0）保持原 prior，KL 约为 0。

这直接支持“残差有效但还不足以改变 argmax”，不是优化器完全没有工作。该 CPU 结论不宣称 GPU padded-batch 动作等价，但原报告的实际 full-document greedy 也全部保持 native。

## 目标与成本出现错位

- Soft-target CE：3.4830004 → 3.4103634，下降 0.0726370。
- Native-reference KL：0 → 0.036869，权重 0.01 的惩罚为 0.0003687；配置目标为 3.4107320。
- 但 expected uniform-action CE surrogate 从 1.8462314 小幅升至 1.8475036。

因此当前优化更接近拟合 soft-target 分布和缩小 margin，而不是直接降低部署成本或改变 greedy。Soft-target 改善不能被解释为实际执行收益。

## 结论与后续约束

Router pilot 停止不是收敛：192 步才首次完整覆盖 5,859 个位置，development patience 已在更早时耗尽，停止由覆盖条件和耐心共同触发。后续若要继续 router，需要预先设计新的决策边界/目标实验，而不是增加相同训练步数；本诊断本身不授权新训练或修改旧配置。
