# 任务卡 C 验收记录

阶段：C——RLOO 与 replay，完成。未进行长训练。

本次修改文件与用途：

- `rl_lm/rng.py`：SHA256+Philox 确定性随机流、float64 inverse-CDF 采样。
- `rl_lm/objectives.py`：reward、G 维 LOO、无归一化 PG loss 和精确目标参照。
- `rl_lm/replay.py`：冻结 records 上的 actor log-prob 重算、microbatch 梯度累计。
- `tests/rl_lm/test_rloo_math.py`、`test_replay.py`：T11–T18。

已实现的规范条款：第 9–11 节及任务卡 C；没有 PPO、critic、多 epoch、组 std 或轨迹长度归一化。

执行的实际命令：

```bash
.venv/bin/python -m unittest tests.rl_lm.test_rloo_math tests.rl_lm.test_replay -v
.venv/bin/python -m unittest discover -s tests/rl_lm -v
.venv/bin/python -m unittest discover -s tests -t .
```

通过的测试及实际数值：

- 本卡复核：27/27 通过；子任务全项目：685 项总计、684 通过、1 跳过、0 失败。
- T13：14 条终端路径、全部 G=2 组合，exact 梯度范数 2.313183e-01；RLOO 与 exact 最大误差 **9.714e-17**。
- T14：trajectory-length 平均与 exact 的 L2 误差 **0.1384340**；组 std whitening 与 exact 的 L2 误差 **0.0822502**，均为明确负对照。
- T15：更新前 replay 与 rollout old joint 最大差 **0.0**；错误 mask、概率规则和动作均触发失败。
- T16：actor 至少一个参数有非零有限梯度；LM 参数 grad=None，参数 hash 不变。
- T17：重排、microbatch、相邻提前 STOP 不改变本 episode 随机流；边界 u 行为与参考一致。
- T18：microbatch 与全 batch 累计梯度最大差 **2.274e-13**。
- T12：全同 returns 的 advantage 逐位为 0。

未运行/skip/blocked：没有本卡 blocked；长训练和真实 checkpoint/GPU 训练不属于本卡。

与规范的差异：

1. `rloo_advantage` 对浮点输入保留原 dtype，以允许 T13 的 FP64 独立验收；FP32 行为与规范参考一致。
2. T17 放在 `tests/rl_lm/test_replay.py` 的 `RngStreamTests` 中，而不是单独 `test_rng.py` 文件。

剩余步骤：任务卡 D 完成训练事务、恢复与 CLI。

输出文件与 checkpoint 身份：本卡没有训练 run 或 actor checkpoint；测试日志 `runs/rl_lm_pair_v1_task_c_tests.log`；未提交或推送。
