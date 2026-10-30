# 100M 运行时诊断报告（PR #4）

- 独立 worktree：`54f0cec`；测试 9/9 通过。
- 真实 100M MoE/dense/R1 actor，物理 GPU 3，selection 开发数据；未访问 newtest。
- Smoke 首次因 GPU 0 被占用而停止，随后在 GPU 3 的 `smoke_002` 完成。

## 原生模型

Smoke batch 1：legacy 47.314 ms、全层 merged 34.683 ms、dense 12.927 ms。扩大测量：legacy 47.496 ms、merged 35.134 ms、dense 13.055 ms。Merged 约快 1.35×，但两个 smoke 样本及扩大检查均未通过数值门槛，不能标为等价部署后端；dense 仍最快。

## 在线 head 与 batch 诊断

扩大版 16 序列、batch 1/8、head 64/256/1024/full、BF16 cache、5×50 次测量，共 35 个 variant、17,500 个 E2E timed calls。

Batch 8 的 learned legacy：
- head64 autocast 103.952 ms；
- head256 autocast 82.946 ms；
- full autocast 77.497 ms；
- full + BF16 cache 77.400 ms。

Batch 8 的 learned merged：
- head64 autocast 93.068 ms；
- head256 cached 67.392 ms；
- full cached 66.440 ms。

说明在线 head 分块和转换是重要成本，BF16 cache 常驻权重为 33,554,432 bytes。可是 merged 的整体数值筛查未通过；batch 1 的部分 legacy 在线候选也存在整体参考差异，不能仅凭延迟结果推广。

## 独立 trace

采集 `learned_merged_head64_cached_bf16` 的独立 profiler trace（非正式计时路径）。主要 CPU 观测：
- 714 次 `aten::linear`，38.321 ms；
- 2,751 次 `cudaLaunchKernel`，15.279 ms；
- `aten::to/_to_copy` 合计约 28.271 ms；
- `aten::nonzero` 6.743 ms，`index_select` 5.019 ms。

这支持 100M 慢速来自大量小矩阵、启动/同步、动态路由索引和类型转换，而不是激活 FFN FLOP 更大。Trace 的 instrumented 时间不能当作不带 profiler 的部署延迟。

## 结论

原生 merged 和在线 head 优化都显示实际速度空间，但数值筛查限制仍然存在。下一步应先解决参考/候选数值口径和真正 grouped/fused 专家内核，不能把当前 merged 结果称为超过 dense 或可部署优化。
