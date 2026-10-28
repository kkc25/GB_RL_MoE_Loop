# 合作者更新方案（integrity v2）重跑报告

- 代码：`main@9ad0825`（PR #1–#3）
- 日期：2026-10-27
- 配置：`configs/rl_lm_v2/`
- 新 cache：`data/rl_lm_pair_v2/cache`、`data/rl_lm_single_v2/cache`
- 新 run：`runs/integrity_v2/`
- 历史 run 未覆盖；newtest 未参与训练、选择或评估。

## 1. 验证与数据契约

全量测试：871 项，870 通过、1 跳过、0 失败。日志：`runs/integrity_v2_full_tests.log`。

Pair cache v2 包含 65536 train、8192 selection、2048 calibration states；preflight/run contract 为 ready。R2 第一次启动因新目录缺少独立 preflight 被拒绝，补齐后重启，未绕过校验。

`sparse_active_merged_v3` 真实 100M GPU 验证：2 序列、128 位置、0 gate tie、0 action difference；相对 canonical 的 CE 平均/最大绝对差为 `6.25e-7 / 6.68e-5`。结果：`validation_active_v3/summary.json`。

## 2. 训练质量

| 实验 | 完成 | best | greedy NLL | 结论 |
|---|---:|---:|---:|---|
| Pair R1 | 2000/2000 | update 1900 | 4.173301637 | 初始 4.173303085，仅改善 1.448e-6 nat，远低于 0.005 门槛 |
| Pair R2 | 500/500 | update 0 | 4.173301637 | 与新 R1 best 相同，无新增质量收益 |
| Single 适配 | 1000/1000 | gate passed | 恶化 0.008036772 nat | 通过 ≤0.02 gate |
| Single RLOO | 500/500 | update 0 | 4.181444545 | 未超越初始 actor，无质量收益 |

Single 适配 student SHA256：`22a51e90ce5e5185a5e4fdf4e4da3dce81acc039874a2f2a7f1e7644254709f9`。适配末层执行计时为 4.116 → 2.316 ms（1.78×），不是全模型加速结论。

## 3. 四格端到端 profile

32 个 selection 序列，warmup 30、5 轮交错、每轮 200 measurements；输出 schema v4。以下单位为 ms，越低越好。

### Batch 1, sequence 1024

| Policy | legacy + v1 | legacy + v3 | merged + v1 | merged + v3 |
|---|---:|---:|---:|---:|
| Canonical R1 | 55.436 | 53.918 | 44.008 | **42.268** |
| Learned R1 | 56.185 | 54.492 | 44.573 | **42.762** |
| Canonical R2 | 62.312 | 58.971 | 50.819 | **47.285** |
| Learned R2 | 60.836 | 57.811 | 49.144 | **46.096** |

对照：原始 MoE R1 48.983 ms，原始 MoE R2 52.657 ms，dense 13.357 ms。

### Batch 8, sequence 1024

| Policy | legacy + v1 | legacy + v3 | merged + v1 | merged + v3 |
|---|---:|---:|---:|---:|
| Canonical R1 | 106.985 | 105.352 | 95.498 | **93.848** |
| Learned R1 | 107.703 | 106.033 | 96.191 | **94.536** |
| Canonical R2 | 122.726 | 119.418 | 111.230 | **107.935** |
| Learned R2 | 121.701 | 117.862 | 110.201 | **106.348** |

对照：原始 MoE R1 67.628 ms，原始 MoE R2 72.117 ms，dense 46.383 ms。

主要收益来自 merged prefix，v3 terminal 进一步小幅降低延迟。Learned R1 最优组合相对旧默认 learned R1：batch 1 从 56.185 降至 42.762 ms（约 1.31×），相对原始 MoE R1 从 48.983 降至 42.762 ms（约 1.15×）。但最优 learned R1 仍明显慢于 dense（42.762 vs 13.357 ms），不能声称超越 dense 或 SOTA。

## 4. 成本与限制

- R1：166.4 s wall time，0 KL backoff；pair ledger 的 `gpu_hours=0.0` 是当前实现记录值，报告同时保留实际 wall time，不把 0 当成真实零成本。
- Single 适配：0.28064 GPU·h；single RLOO：0.01054 GPU·h。
- Profile：0.84806 GPU·h。
- 新后端是显式 runtime profiling 候选，没有写入模型权重，也没有自动推广到训练。
- 没有进行 newtest 正式确认；没有冻结新的 confirmation candidate/scope。

## 5. 总结

合作者的更新成功强化了数据/run 契约，并带来可实测的推理延迟改善；但重新训练仍未带来有意义的 NLL 收益。当前最可靠结论是：**保留历史负质量结果；merged prefix + active v3 可作为后续独立验证的部署候选，但尚不能宣称质量或 dense 端到端性能超越。**
