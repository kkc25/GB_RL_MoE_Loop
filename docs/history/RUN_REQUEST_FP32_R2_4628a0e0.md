# 当前运行请求：FP32 第二轮专家候选修复

**生产候选代码已实现；真实 GPU 是否通过仍待复验。**

完整步骤见 [RUN_REQUEST_FP32_R2.md](RUN_REQUEST_FP32_R2.md)。本次只执行新目录的 prepare → precision-screen，不启动正式训练，不修改严格容差。

- 分支：`fix/mixtral-fp32-r2-candidate`，代码叠在 PR #17 上；使用交付的完整 SHA，不用 main 的旧执行器。
- 新配置：`configs/mixtral/loop_scale_fp32_experts.json`，显式指定 `r2_precision=fp32_experts`。旧配置默认仍走原路径。
- 只提高新增第二轮中的 correction、专家投影、激活、加权和累加精度，保留原模型、路由及残差出口的 BF16 边界。
- 反向传播也采用 FP32；每次只物化一个投影的 FP32 权重，不保留整套高精度专家副本。
- 配对筛查 4 文档 × 2 幅度 × 2 精度，检查更新前后 hidden/CE/梯度、同步耗时和峰显存。它不签发 preflight 证书。
- 原 strict gate 不降低；候选通过筛查后仍需审阅并单独通过绑定新配置的 preflight，不能自动续跑 cache/fit。

上一版入口逐字节保存于 [历史入口](docs/history/RUN_REQUEST_EXPERT_PRECISION_d3324807.md)。旧失败结果、冻结输入与最终测试封存均保留。
