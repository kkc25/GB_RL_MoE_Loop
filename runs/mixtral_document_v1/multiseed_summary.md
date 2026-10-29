# 全文 Adapter 多 Seed 与 Native-KL 对照报告

协议：PR#12 全文训练，Mixtral-8x7B Base、grouped_mm、rank=8、development-best。所有候选均通过 development、外层 selection、冻结 replay 和 profile；未访问 newtest。

| Seed | KL | Development 改善 | Development 改善文档 | Selection 改善 | Selection 改善文档 | Candidate 平均延迟 |
|---|---|---:|---:|---:|---:|---:|
| 17 | 0.1 | 0.0111062 | 7/8 | 0.0108407 | 15/16 | 179.355 ms |
| 17 | 0 | 0.0116367 | 7/8 | 0.0110634 | 14/16 | 179.497 ms |
| 29 | 0.1 | 0.0105136 | 8/8 | 0.0106330 | 15/16 | 179.425 ms |
| 29 | 0 | 0.0108817 | 8/8 | 0.0105622 | 15/16 | 179.430 ms |
| 43 | 0.1 | 0.0097098 | 8/8 | 0.0084306 | 14/16 | 179.570 ms |
| 43 | 0 | 0.0095165 | 8/8 | 0.0087729 | 14/16 | 179.437 ms |

Native selection NLL 均为 1.8169399446；development 初始 NLL 均为 1.9723657597。Seed 17/29 的 best 为 update 192，seed 43 为 update 208；profile 每个 variant 96 次测量。

## 结论

- 三个 seed、两种 KL 配置均稳定取得开发集正收益，全文 adapter 的结果不是单次 seed 偶然。
- KL=0 与 KL=0.1 在不同 seed 上互有高低，没有稳定的 KL 优劣方向，不能宣称去掉 native KL 更好。
- 延迟开销仍很小，各 candidate 与对应 native 差异约 -0.003–0.25 ms。
- 这仍是新增低秩残差修正的开发集结果，不是 RL 路由或 loop 收益，也不是最终 held-out/SOTA 证明。三个预声明 seed 已完成，下一步应冻结最终确认范围，而不是继续基于 selection 调参。
