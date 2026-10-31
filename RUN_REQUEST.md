# 当前运行请求：单 token 固定分块候选修复

**新候选已实现，真实 GPU 一致性仍待验收；没有放宽容差。**

请按 [RUN_REQUEST_CANONICAL32.md](RUN_REQUEST_CANONICAL32.md)，固定交付 SHA，在新目录先执行 prepare → precision-screen。

- 新配置：`configs/mixtral/loop_scale_canonical32.json`，精度模式必须是 `fp32_canonical32`。
- 新分支：`fix/mixtral-canonical32-r2`，叠在 PR #18 上，不必先合并才能执行。
- R2 的 norm、correction、router、每个专家桶，以及输出 norm/head 均按固定 32 行处理。仅补齐选中的行/专家分配，真实补齐成本单独统计。
- 原 BF16 和普通 FP32 作为对照，新配置共筛查 4 文档 × 2 幅度 × 3 模式 = 24 个案例。
- 最多 12 次实际单 token 路径诊断；若仍失败，报告首次差异及固定 hidden 的 readout 隔离证据。
- 同模式原生基线/STOP 使用相同 readout，同时保留与原 pretrained 输出的严格检查，不能把原生基线漂移藏起来。
- screen 不签发训练凭据。只有新模式全部数值检查、资源审阅及独立正式 preflight 通过，才考虑进入原计划 100k 档；命令不自动串联训练。

旧入口逐字节归档：[PR #18 入口](docs/history/RUN_REQUEST_FP32_R2_4628a0e0.md)。旧配置默认路径和历史结果不变。
