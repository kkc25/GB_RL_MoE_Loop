# 当前运行请求：专家逐算子与精度对照诊断

**这是诊断补丁，不是已经验证有效的 GPU 数值修复。**

请按 [RUN_REQUEST_EXPERT_PRECISION.md](RUN_REQUEST_EXPERT_PRECISION.md)，固定此次交付的完整提交 SHA，在新的 worktree 和新输出目录中只执行 prepare → preflight。不要启动 cache、fit 或最终测试。

- 分支：`fix/mixtral-expert-precision-diagnostics`。
- 实际 dense/compact 专家调用保持原行数和相同输入、路由、权重；观察内部 ATen 算子，不替换专家 forward。
- 整轮最多两次深入诊断，每次只保存前八个选中 token 的聚合统计；FP32 参考使用同一份已经量化的权重/输入，不是数学真值。
- BF16 reduction 开关仅在隔离诊断中关闭，退出或异常后恢复。生产设置、严格容差、训练门槛均不变。
- 全量 BF16 不再被称作正确答案；两条路径分别与公共 FP32 参考比较，但任何诊断改善都不改变严格 preflight 失败。
- 旧数据、结果、代码和最终测试封存保留。上次当前入口逐字节归档为 [历史入口](docs/history/RUN_REQUEST_PARITY_31b79627.md)。
