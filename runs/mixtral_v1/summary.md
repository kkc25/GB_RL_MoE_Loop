# Mixtral-8x7B 冻结路由与运行时优化首轮报告

- 代码：`main@d33efa8`
- 模型：`mistralai/Mixtral-8x7B-v0.1` Base
- 官方 revision：`fc7ac94680e38d7348cfa806e51218e6273104b0`
- 本地模型：`/nvme-1/zhoukegu/models/Mixtral-8x7B-v0.1`，19/19 safetensors 分片，完整文件哈希见各 report。
- 设备：物理 GPU `0,3,5,7`（4×RTX A6000），BF16，逻辑设备 0–3。
- 数据：FineWeb-Edu 开发角色文档，train/selection/calibration 各 64 篇；本轮每组取 2 篇、序列长 128；未使用 newtest。
- 本报告是 M0/M1 和 4-update M2 smoke，不是正式质量确认。

## M0：真实 CUDA 一致性

Eager 与 grouped_mm 分别独立通过 `native_equivalence.eligible=true`：每个后端内，原生与受控 R1 快速路径的 logits、逐 token CE、NLL 差异均为 0，原生末层专家对轨迹相同，动作和轮数合法。

Eager selection NLL：原生/受控均为 `1.52973584451299`。Grouped_mm selection NLL：原生/受控均为 `1.5314759093986592`。不同后端之间 NLL 不完全相同，因此不能跨后端混用基线。

Grouped_mm 实际观察到 `torch.nn.functional.grouped_mm` BF16 分派，四卡均有记录；该证据是 HF dispatch 分支观察，不是底层 CUDA profiler kernel trace。

## M1：完整序列 prefill 性能

Batch 1、序列长 128、到完整 logits、无 KV cache、计时不含 CE；每次 3 个测量，单位 ms。

| 后端 | 原生 | 受控 R1 快速路径 | 受控额外开销 |
|---|---:|---:|---:|
| Eager | 221.923 | 221.971 | 0.048 ms（0.02%） |
| Grouped_mm | 163.974 | 164.083 | 0.109 ms（0.07%） |

Grouped_mm 原生相对 eager 原生从 221.923 降至 163.974 ms，约 **1.353×**；受控 R1 从 221.971 降至 164.083 ms，同样约 **1.352×**。R1 快速路径复用 norm/gate 并移除无策略时的无用 observation/STOP 管理，未引入可测的明显额外开销。

Eager 各卡峰值 allocated 约 21.9/24.4/24.4/16.5 GiB；grouped_mm 为 21.9/24.4/24.4/16.5 GiB（四舍五入后相同）。这是绝对 allocator 峰值，不是每路径独立增量。

## M2：有界 R1 路由训练

默认 reference 与 `deduplicate_checked_v1` 均完成 4/4 updates；两者 best 都是 update 0，best selection NLL 分别为其初始基线，未见策略收益。4 步 smoke 不能证明路由不可学习。

### 评分去重核验

- 参考：80 requested/scored rows，8 score calls，wall 1.571 s。
- 去重候选：51 cache hits，但因保留完整参考校验，总计 93 scored rows、12 calls，wall 1.627 s。
- 仅 1/4 updates 通过容差并使用 candidate；3/4 因评分误差超过 `atol=rtol=1e-5` 回退。
- 四次最大评分误差：0.001395、0.028505、0.053058、2.49e-6。

去重减少了理论重复计算，但 BF16 批形状变化造成不可忽略的数值差异，完整校验又增加实际工作；本轮没有加速，不应推广为默认模式，也不能放宽容差掩盖失败。

## 结论与下一步

1. **可以采用 grouped_mm + R1 快速路径作为后续开发候选**：同一后端内数值一致，且本轮比 eager 快约 35%。仍需更多重复和更长序列验证稳定性能。
2. **评分去重保持实验候选**：需要设计不破坏数值口径的复用方式，并去除“为验证而重复计算全部参考”的成本后才能评价真实加速。
3. **尚不能声称策略质量提升或超越 SOTA**：当前只有 4 步训练，best 仍是初始 actor，未进行多 seed、置信区间或 newtest 确认。
4. 下一轮应在冻结配置下扩大训练 updates/context，并独立报告 grouped_mm 原路由与 learned actor 的 selection 质量；若 best 仍为初始状态，按停止规则不扩大训练。

关键产物：`runs/mixtral_v1/{validate_eager,profile_eager,validate_grouped_mm,profile_grouped_mm,train_reference,train_dedup_checked}/`。
