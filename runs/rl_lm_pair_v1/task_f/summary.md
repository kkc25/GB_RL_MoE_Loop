# 任务卡 F 验收记录：R2、完整 forward 与 profile

阶段：F——R2 warmstart/STOP 闭环、500-update lambda=0 pilot 和三层 profile，已完成。R2 pilot 未取得增量收益，未扩展 price grid 或多 seeds；newtest 保持封存。

## 本次修改文件与用途

- `experiments/rl_lm_profile.py`：selector-only、executor-only、完整端到端三层计时；补充 `--physical-gpu-index`，正确处理 `CUDA_VISIBLE_DEVICES` 的物理/逻辑 GPU 映射。
- `rl_lm/ledger.py`：累计真实 wall_time 对应的 GPU hours，不再保留字段恒为 0 的假账本。
- `tests/rl_lm/test_online_parity.py`：R2 STOP/最多两轮/STOP 后 hidden 不变的闭环检查。
- `runs/rl_lm_pair_v1/task_f/`：R2 评估、三层 profile、newtest 未使用审计及本报告。

已实现的规范条款：第 10.2、11、12、16、17 节任务卡 F；没有扩展 price、seed 或更大训练。

## 执行的实际命令

```bash
.venv/bin/python -m experiments.rl_lm_train \
  --config configs/rl_lm/pair_r2.json \
  --checkpoint runs/next_2026_10_26/baseline_100m/loop_r1/latest.pt \
  --corpus-dir data/rl_lm_pair_v1/corpus \
  --tokenizer runs/data/fineweb_edu_100m/tokenizer.json \
  --warmstart-actor runs/rl_lm_pair_v1/r1_seed17/checkpoint_best.pt \
  --stop-after 500 --device cuda:0

CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m experiments.rl_lm_profile \
  --config configs/rl_lm/pair_r2.json \
  --checkpoint runs/next_2026_10_26/baseline_100m/loop_r1/latest.pt \
  --corpus-dir data/rl_lm_pair_v1/corpus \
  --tokenizer runs/data/fineweb_edu_100m/tokenizer.json \
  --device cuda:0 --physical-gpu-index 2 \
  --r1-actor runs/rl_lm_pair_v1/r1_seed17/checkpoint_best.pt \
  --r2-actor runs/rl_lm_pair_v1/r2_lambda0_seed17/checkpoint_best.pt \
  --sequences 32 --warmup 30 --repeats 5 --measurements 200 --batch8

.venv/bin/python -m unittest discover -s tests -t . -v
```

## 通过的测试及实际数值

- 全项目：698 项总计，697 通过、1 跳过、0 失败。
- R2 STOP 闭环：slot 1 全 STOP，STOP 后 hidden 与 P1 best 逐位一致，logits 与 P1 best 逐位一致；没有第三个 STOP log-prob。
- R2 pilot：500/500 updates 完成，0 重试；best 仍为 update 0 初始 actor。
- R2 learned greedy NLL 4.173289299，等于 P1 best；没有新增质量收益。
- canonical teacher R2 NLL 4.217555523，比 R1 差约 0.044252 nat；固定 R2 的额外计算没有质量收益。
- R2 initial/best 与 P1 best 的 paired delta 为 0，故保留初始候选，不扩展 price/seed。

### 三层正式 profile

协议：32 固定输入，30 warmup，5×200 次正式测量/变体，batch=1 主结果及 batch=8 对照，BF16 生产口径，物理 GPU 2 独占。

Selector-only（batch=1, S=1024）：

- legacy gate+top-k：0.267 ms
- canonical pair selector：0.278 ms
- RL 特征+actor+joint：2.832 ms，actor 开销明显高于选择器本身。

Executor-only（batch=1, S=1024）：

- 原两槽：3.389 ms
- merged dispatch：3.835 ms
- grouped canonical：6.325 ms（训练用全专家固定形状成本更高，仅作后端对照）。

End-to-end（batch=1, S=1024）：

| 候选 | 平均延迟 | 有效 tokens/s |
|---|---:|---:|
| 原 MoE R1 | 46.989 ms | 21,792 |
| 原 MoE R2 | 50.633 ms | 20,224 |
| 参数匹配 dense | **12.954 ms** | **79,052** |
| canonical R1 | 53.855 ms | 19,014 |
| canonical R2 | 60.257 ms | 16,994 |
| learned R1 | 54.370 ms | 18,834 |
| learned R2 | 58.791 ms | 17,418 |

batch=8 下 learned R1/R2 分别为 106.896/120.806 ms；dense 为 46.536 ms。actor 开销使 learned 比原 MoE 更慢，dense 仍显著更快。

profile 的 GPU hours 为 0.30693，来自真实 wall_time；原始每次 CUDA event/wall time、GPU UUID、软件版本与显存均保存在 `task_f/profile/profile_report.json`。

## 未运行/skip/blocked 项目及具体原因

- price grid、更多 seeds、R2 续训和独立 newtest 确认：R2 pilot 没有增益，按第 17 节停止。
- newtest 仍封存：已核验训练/选择/评估均未读取，只有未来冻结 candidate manifest 且门槛通过后才能打开。
- S1 单专家分支属于独立任务卡 G，不在 F 中启动。

## 与规范的差异

- profile CLI 新增 `--physical-gpu-index`，用于正确选择 nvidia-smi 的物理 GPU；不改变计时范围或算法。
- 使用物理 GPU 2，因 GPU 1 有其他进程，未共享计时卡。

## 剩余步骤

进入独立任务卡 G（S1）：先监督适配最后专家池，再决定是否启动 single-RL。

## 输出文件与 checkpoint 身份

- R2 评估：`task_f/evaluate_r2_selection.json`
- R2 STOP 核验：`task_f/verify_r2/summary.json`
- 三层 profile：`task_f/profile/profile_report.json`
- newtest 审计：`task_f/newtest_audit.json`
- 全量回归：`runs/rl_lm_pair_v1_task_f_full_tests.log`
- P1 best SHA256：`290e1f25f0320701988a9b28121aad0e591f2bd9ea0647dd0f9dd3fe0ac8af09`
- R2 initial/best SHA256：`837372ef20ff7330fb52db85e490fc86de273fb96e50e37e34760a26bd577aa1`
- 当前分支 `research/rl-lm-pair-v1`，未提交或推送。
