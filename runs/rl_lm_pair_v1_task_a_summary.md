# 任务卡 A 验收记录

阶段：A——schema、动作和概率，已完成。

本次修改文件与用途：

- `rl_lm/config.py`：严格嵌套 schema、未知字段拒绝、阶段/动作/dtype/hash 约束；独立 `RLConfig` 与 `CorpusConfig`。
- `rl_lm/types.py`：`ContextBatch`、`PolicyState`、`DecisionRecords`、`RolloutBatch` 的 shape/dtype 检查。
- `rl_lm/actions.py`：STOP=0、pair=1–28 的唯一映射、合法 mask 和动作表身份。
- `rl_lm/selector.py`：canonical pair 与 0.9+0.1/28 reference prior。
- `rl_lm/actor.py`：FP32 两层 MLP、零初始化 heads、联合 29 类 log-probs 和联合 argmax。
- `configs/rl_lm/`：完整 R1/R2/corpus 配置、动作表和 hash。
- `tests/rl_lm/`、`tests/__init__.py`：T01–T04/T24 与类型测试；避免测试包遮蔽生产包。

已实现的规范条款：第 5 节动作编号、第 6 节张量边界、第 7 节策略与联合概率、第 13 节配置约束、第 19 节任务卡 A。未写训练主循环、PPO 或专家适配。

执行的实际命令：

```bash
.venv/bin/python -m unittest discover -s tests/rl_lm -v
.venv/bin/python -m unittest discover -s tests -t . -v
```

通过的测试及实际数值：

- 定向：16/16 通过。
- 全项目：608 项总计，607 通过、1 跳过、0 失败。
- `(0,1)→1`、`(0,7)→7`、`(1,2)→8`、`(6,7)→28`；全 28 对逆映射一致。
- r0 STOP 概率严格为 0；合法分布概率和为 1；hazard=.4 且 pair 均匀时联合 greedy 选择 STOP。
- 零初始化 greedy 为 canonical R1；R2 warmstart 保留 R1 策略且 stop_head 重置并解冻。
- 动作表 SHA256：`9a377c254de5c1c5a1900fdfa48d52a78a262258e85e2d81d235ae3245c51127`。
- R1 配置 SHA256：`32be8874ebe92e255a160517c66087ebaa3c7572540a7ba6474b907a1e976d2c`。
- R2 配置 SHA256：`55ec0d89706f3aab197b41b310176105dda5ad10192e9906cb3a8bd061783b6d`。
- corpus 配置 SHA256：`0bd7bdc46fa3f8b164ddb2234950b1189e606c3af4891b3eee48aebbbe4ac7ea`。

未运行/skip/blocked 项目及具体原因：全量 1 项既有 skip；没有执行真实 checkpoint/GPU 训练，因为任务卡 A 只要求 toy 数学与 schema 验收，不属于 blocked。

与规范的差异：无。checkpoint/corpus/tokenizer 路径按规范保留 null，待任务卡 B/E 的真实入口填入，未使用假路径。

剩余步骤：进入任务卡 B，完成冻结环境、数据身份、prefix cache、canonical executor/readout 和 R1 online forward。

输出文件与 checkpoint 身份：

- `runs/rl_lm_pair_v1_task_a_tests.log`
- `runs/rl_lm_pair_v1_task_a_full_tests.log`
- `configs/rl_lm/action_table.json`、`action_table.sha256`
- 当前分支 `research/rl-lm-pair-v1`；未创建 actor checkpoint，未提交或推送。
