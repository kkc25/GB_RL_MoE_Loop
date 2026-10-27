# 任务卡 B 验收记录

阶段：B——冻结环境与数据身份，代码与 toy/小样本验收完成；完整新语料与 65k cache 尚未生成，preflight 正确标记为 `not_built`。

本次修改文件与用途：

- `rl_lm/corpus.py`、`experiments/rl_lm_corpus.py`、`tests/rl_lm/test_corpus*.py`：本地原文 JSONL、三种历史 manifest 适配、doc/text/near hash 排除、near_hash 重建、staging 提交与拒绝测试。
- `rl_lm/cache.py`、`tests/rl_lm/test_cache.py`：均匀合法 token 抽样、非零 offset 全局序号、FP32 `h0` cache、legacy/canonical baseline、完整身份 hash 与加载校验。
- `rl_lm/executor.py`、`environment.py`、`inference.py`、`readout.py`、`ledger.py` 及对应测试：冻结观察、canonical/sparse executor、R1/R2 状态机、生产 BF16 head/FP32 CE、R1 在线 forward、逻辑/物理账本。
- `rl_lm/cli.py`、`experiments/rl_lm_prepare.py`、`experiments/rl_lm_preflight.py`：路径覆盖、resolved config、模型与数据预检、cache 准备入口。
- `experiments/rl_lm_verify_r1.py`：真实 100M R1 最小在线闭环核验。

已实现的规范条款：第 4 节数据/标签/cache 契约、第 5 节 R1/R2 状态机、第 7–8 节冻结观察与生产读出、第 10 节 rollout 状态机、第 14 节账本、T05–T10、T21–T23、T22 的 R1 部分。未实现训练主循环和 R2 实验。

执行的实际命令：

```bash
.venv/bin/python -m unittest discover -s tests/rl_lm -v
.venv/bin/python -m unittest discover -s tests -t . -v
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m experiments.rl_lm_verify_r1
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m experiments.rl_lm_preflight \
  --config configs/rl_lm/pair_r1.json \
  --checkpoint runs/next_2026_10_26/baseline_100m/loop_r1/latest.pt \
  --corpus-dir runs/data/fineweb_edu_100m \
  --tokenizer runs/data/fineweb_edu_100m/tokenizer.json \
  --device cuda:0
```

通过的测试及实际数值：

- B 卡定向：65/65 通过。
- 全项目：658 项总计，657 通过、1 跳过、0 失败。
- cache identity 覆盖 checkpoint、gamma、tokenizer、packed 字节、backend chunk 的拒绝；非零 offset 与全局 sequence index 通过。
- 真实 R1 在线核验：2 sequences、128 个合法位置、0 gate ties、0 canonical/legacy pair differences、0 initial-actor/canonical action differences。
- 真实 R1 CE 绝对差：canonical−legacy mean 0.001758 / p95 0.004105 / max 0.048073；sparse−legacy mean 0.001143 / p95 0.002543 / max 0.047712。这些数值报告为跨后端/形状差异，不作为收益。
- preflight：模型与原始数据 verified，三个训练相关 cache 均为 `not_built`，没有伪造 baseline。

未运行/skip/blocked 项目及具体原因：

- 完整新四角色 RL corpus 尚未构建：`configs/rl_lm/corpus.json` 的 source/revision 与所有排除来源仍为 null/占位，按规范拒绝正式构建。
- 65,536/8,192/2,048 状态 prefix cache 尚未构建，等待新 corpus。
- T22 的 R2 在线部分属于任务卡 F；R1 部分已完成。
- 真实训练未开始，属于后续任务卡 E。

与规范的差异：历史 manifest 格式拒绝测试按文件职责放在 `tests/rl_lm/test_corpus_manifest.py`，不是 `test_config_and_roles.py`；其余无。

剩余步骤：任务卡 C 实现 RLOO、随机流、replay 与 T11–T18；再进入 D 训练事务。

输出文件与 checkpoint 身份：

- 真实 R1 结果：`runs/rl_lm_pair_v1/task_b/online_r1/summary.json`、`per_state.jsonl`。
- preflight：`runs/rl_lm_pair_v1/r1_seed17/preflight.json`，状态 `not_built`。
- 测试日志：`runs/rl_lm_pair_v1_task_b_tests.log`、`runs/rl_lm_pair_v1_task_b_full_tests.log`。
- 使用 checkpoint：`runs/next_2026_10_26/baseline_100m/loop_r1/latest.pt`；SHA256 `0ccb04fe0a98f9147d252f526a360ef9a3e780c01dccb569caf77163457b6202`。
- 当前分支 `research/rl-lm-pair-v1`；未提交或推送。
