# RL-LM pair run summary

- run_id: `rl-lm-8287d3f97d70-20261027T192701`
- status: **completed**
- stage: r2, initialization: warmstart
- successful updates: 500 / planned 500 (stop-after cap 500)
- LR scale: 1.0, cumulative KL backoffs: 0
- selection status: selected
- best: update 0, greedy_nll 4.173301637056055, mean_rounds 1.0

## 已实现
- on-policy RLOO 采样→replay→单次受监控 update;原子 checkpoint 与精确恢复。

## 已验证
- CPU toy 契约测试见 tests/rl_lm/test_resume.py (T19) 与 test_update_transaction.py (T20)。

## 尚未运行
- 真实 100M checkpoint 上的 GPU 训练(任务卡 E)不在本交付内。

## 未通过
- 无(若本 run 以 failure 结束,见 failure.json / update_rejected.json)。
