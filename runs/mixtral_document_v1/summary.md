# PR#12 全文训练实验报告

- 代码：`main@8e79971`
- 模型：Mixtral-8x7B-v0.1 Base，grouped_mm，冻结主干。
- 数据：24 train / 8 development / 16 selection / 8 calibration，物理分离 manifest；未访问 newtest。
- 新协议测试：46/46 通过。

## Smoke

Router/adapter 的同协议 initial/frozen replay 均通过；4 步 smoke 均无 development 收益，仅作为软件检查，不作为否定方法的证据。

## 主 pilot：development 选择

| 路线 | best | 初始 development NLL | best development NLL | 改善 | Gate |
|---|---:|---:|---:|---:|---|
| Router | update 0 | 1.9723657597 | 1.9723657597 | 0 | 未通过；不 evaluate/profile |
| Adapter | update 192 | 1.9723657597 | 1.9612595685 | **0.0111062 nat/token** | 通过 |

Router 在 192 次更新后 development patience 停止。Adapter 完成 256 次更新并覆盖全部训练位置；development 7/8 文档改善、1 篇轻微恶化。

## Adapter 外层 selection evaluate

同一冻结 development-best（update 192），不更新或重选：

- Native/controlled：1.8169399446
- Candidate：1.8060992430
- 改善：**0.0108407016 nat/token**，超过 0.001 门槛。
- 15/16 文档改善，1 篇恶化 0.0009804 nat；最大单篇改善 0.0243180 nat。
- 冻结候选 replay 的 logits/hidden/CE 全部零差异。

## Profile

16 篇 selection、完整 fresh prefix + 全文 terminal tail/head，3 轮交错 × 32 次，共 96 次测量/variant：

| 路径 | 平均延迟 |
|---|---:|
| Native | 179.236 ms |
| Controlled original R1 | 179.284 ms |
| Frozen adapter | 179.355 ms |

Adapter 相对 native 增加约 0.119 ms（0.067%），在当前测量噪声下开销很小。该结果支持“新增低秩修正以很小推理开销获得开发集质量改善”，但不证明统计显著、最终 held-out 或 SOTA，也不证明 RL 路由或 loop 收益。

## 结论

全文形状、全有效位置 native KL 约束与 development 选模解决了旧 64 点适配器的严重过拟合问题，并首次得到同时通过 development 和外层 selection 的候选。Router 当前没有收益，保留原模型。后续应进行多 seed、KL=0 单因素消融和独立最终确认，不能把当前开发结果直接称为生产或 SOTA 成功。
