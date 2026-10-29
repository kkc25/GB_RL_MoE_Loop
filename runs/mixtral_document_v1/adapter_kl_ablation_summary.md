# 全文 Adapter Native-KL 消融报告

- 协议：`main@8e79971` 的 PR#12 全文训练；主结果上传于 `a2459d9`。
- 单因素变化：`native_kl_beta=0.1 → 0`；其余数据、rank=8、seed=17、更新上限和 gate 保持一致。
- 未访问 newtest。

| 指标 | KL=0.1 主实验 | KL=0 消融 |
|---|---:|---:|
| Best update | 192 | 192 |
| Development NLL | 1.9612595685 | 1.9607290156 |
| Development 改善 | 0.0111061912 | 0.0116367440 |
| Development 改善文档 | 7/8 | 7/8 |
| Selection NLL | 1.8060992430 | 1.8058764957 |
| Selection 改善 | 0.0108407016 | 0.0110634489 |
| Selection 改善文档 | 15/16 | 14/16 |
| 冻结 replay | 通过 | 通过 |
| Candidate 平均延迟 | 179.355 ms | 179.497 ms |

两者 native 基线均为 development 1.9723657597、selection 1.8169399446。KL=0 的 profile native 为 179.251 ms，candidate 增加约 0.246 ms（0.137%）；主实验 candidate 相对其 native 增加约 0.119 ms（0.067%）。

## 结论

KL=0 也通过了 development、selection 和 profile，平均 NLL 改善略高约 0.00022 nat/token，但改善文档更少，且只有一个 seed，不能据此判定去掉 native KL 更优。当前证据表明全文训练与 development 选模是主要有效因素；native KL 是否提供额外稳健收益需要 seed 17/29/43 的配对实验，而不是继续基于 selection 调整配置。
