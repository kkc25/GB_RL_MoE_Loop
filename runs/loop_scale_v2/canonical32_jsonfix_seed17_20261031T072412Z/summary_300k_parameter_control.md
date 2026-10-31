# 300k parameter_control 阶段报告

- 固定代码：`510194a3dc95f8c8b5df0d50ea91ae299f78243b`；冻结数据身份不变。
- 本 cell 已完成，报告摘要校验通过；300k 的两个 R2 cell 尚未全部完成，不能据此判断是否进入 1M。
- 实际 adapter exposure 为 540,216（270,108 个独立位置，两遍），完成 2,112 次更新，gate 更新为 0。
- 核心 wall 为 1,019.722 秒，失败与未知操作均为 0。

## 固定 checkpoint 结果

| Epoch | Train NLL | Development NLL |
| ---: | ---: | ---: |
| 0 | 1.8276525955 | 1.8757864007 |
| 1 | 1.7885945164 | 1.8604516243 |
| 2 | 1.7596783149 | 1.8751814209 |

原生开发集 NLL 为 `1.8757864007`。按既定规则选择 epoch 1，改善约 `0.0153347764 nat`；100k 参数控制为 `1.8595192098`，因此 300k 参数控制相对 100k 没有改善，而是高约 `0.0009324145 nat`。这不代表两个 R2 cell 的趋势，不能单独用于关闭 1M 入口。

所有 epoch 的 conditional parity 均通过。逐文档/逐位置证据、ledger、报告和每卡显存位于 `cells/300000/parameter_control/seed17/`；本地 checkpoint 仅以 SHA256 记录在 `checkpoint_manifest_300k_parameter_control.json`，checkpoint 与 cache 未进入 Git。

下一阶段 `r2_full` 正在运行。300k 三 cell 完成并形成同幅度、同结果类型趋势前，不启动 1M。
