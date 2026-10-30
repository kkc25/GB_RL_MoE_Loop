# Expected-CE Router Pilot 结果

- 配置：`document_router_expected_ce_pilot`，seed=17，grouped_mm，全文统一动作监督。
- 完成 192 次更新后因 development patience 停止；best 为 update 0。
- 初始/best development NLL：`1.9723657597`，改善 `0`，未达到 `0.001 nat/token` 门槛。
- 最新实际全文 greedy 与原路由一致率仍为 `1.0`，train NLL 保持 `1.8155064459`，没有实际路由变化。
- 按 gate 保留初始模型，**未执行 selection evaluate/profile**，未访问 newtest。

与 soft-target CE 主实验一样，expected-CE 对照也未显示可部署路由收益；不能把优化目标下降或非零梯度当作质量改善。
