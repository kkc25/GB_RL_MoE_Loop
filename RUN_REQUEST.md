# 当前运行请求：混合门执行一致性诊断补丁

**状态：CPU 验证后交付；真实 GPU 数值根因未确认，尚不能称已修复。**

本次先运行有界 preflight，不运行 cache/fit/screen/final-test。完整步骤见 [RUN_REQUEST_PARITY_REPAIR.md](RUN_REQUEST_PARITY_REPAIR.md)。

- 分支：`repair/mixtral-conditional-parity-diagnostics`。
- 执行模块仍为 `python -m experiments.mixtral_loop_scale`。
- 用本次交付完整 SHA，新 run_id 和新 freeze_dir。不得恢复旧失败目录或覆盖其凭据。
- 生产公式、专家后端、数据预算、门策略及 `atol=rtol=1e-4` 不变；没有慢速逐 token 后端替换，没有 compute-all 冒充稀疏执行。
- 更新前后均检查 STOP、全开、两种交替位置和首尾单位置；遇数值差异保存逐层隔离诊断，在预算内继续其他文档/幅度。主干完整性、非有限值、异常和预算失败仍立即停止。
- 即使完整收集了诊断，只要一项未通过，仍不签发正式 preflight 通过凭据。

原 v2 当前入口逐字节保存在 [历史入口](docs/history/RUN_REQUEST_LOOP_SCALE_V2_51cc1afc.md)；研究方案和数据学习曲线仍见 [v2 运行请求](RUN_REQUEST_LOOP_SCALE_V2.md)。本次先排查执行路径，不能以 GPU 检查失败评价数据规模或 loop 的效果。
