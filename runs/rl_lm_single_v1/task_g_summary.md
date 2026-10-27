# 任务卡 G 验收记录：S1 单专家分支（适配 + single-RL pilot）

阶段：G——独立 S1 单专家分支（`spec_version=RL-LM-Single-v1.0`），已完成监督适配、质量门槛与 500-update single-RLOO pilot。pair-v1 文件与结果未修改、未提交、未推送。

## 本次修改文件与用途

全部为新增文件（pair-v1 代码零改动）：

- `rl_lm/single_executor.py`：`SingleExpertExecutor`（canonical 全 8 专家训练后端 + sparse 部署后端），明确专家 ID、幅度恒 1，不调用旧 `MoE.forward` 幅度路径；`frozen_single_observe`（previous-expert one-hot 特征）；`single_argmax_expert`（argmax gate，tie 取最低 ID）；`single_reference`（0.9 argmax + 0.1/8）；`SingleLedger`（logical single round 与物理 8 倍计账）；`single_online_forward`（joint-9 在线闭环 greedy forward，前 11 层保持原 backbone 函数，仅 terminal executor override）；`ArgmaxSingleTeacherPolicy`。
- `rl_lm/single_readout.py`：可微 `differentiable_logits`（FP32 tied head，冻结权重但保留对其输入的反传）、`adapt_ce_loss`、`adapt_kl_loss`（FP32 softmax/log_softmax，温度 1，逐 token 求和再平均，teacher 必须 detach）、`relative_update_error`（逐 token 相对平方误差再平均，非整 batch 总范数比）。
- `rl_lm/single_actor.py`：joint-9 分布（STOP + 8 专家，factorized，同 pair 的 stop-logit 结构）、`SingleActor`（hidden 128，expert head 8 维，R1 冻结 stop head）、`SingleDecisionRecords`。
- `rl_lm/single_config.py`：`SingleAdaptConfig`/`SingleRLConfig` schema 与固定常数校验；`single_cache_identity`（新 `rl-lm-single-cache-v1` / `single8_stop_v1` 身份，逐一核对 prefix 依赖 hash 后才允许从 pair payload 派生）。
- `experiments/rl_lm_single_adapt.py`：监督适配 CLI（白名单==blocks.11.moe.pool.w1/w2/w3，AdamW 1e-5/(.9,.95)/wd.01/clip1/batch256/≤1000 updates/50 warmup/cosine→1e-6，每 100 步完整 selection 评价，每专家样本/梯度/权重变化记录，adapter checkpoint + resume，执行时间对照，gate.json）。
- `experiments/rl_lm_single_train.py`：single RLOO CLI（仅 gate 通过可启动；reference=适配后固定单专家 R1 CE；独立 run/config/checkpoint/resume；500 步 pilot）。
- `configs/rl_lm/single_adapt.json`、`configs/rl_lm/single_rloo.json`。
- `tests/rl_lm/test_single_actions.py`、`test_single_executor.py`、`test_single_adapt.py`、`test_single_online.py`、`test_single_rloo.py`（40 项新 toy 契约测试）。
- 结果：`runs/rl_lm_single_v1/adapt/`、`runs/rl_lm_single_v1/rloo_seed17/`、日志 `runs/rl_lm_single_v1_task_g_{adapt,rloo,full_tests}.log`；新 cache `data/rl_lm_single_v1/cache/`（新身份 manifest + 重算 single baseline）。

## 已实现的规范条款

第 18.1 节（单专家函数 h'=h+γE_e(RMSNorm(h))、幅度 1、明确 ID、terminal override only、动作 0 STOP / 1–8 专家 0–7）；18.2（teacher 冻结、student 仅末层池 w1/w2/w3 可训、同一 h0、argmax tie 最低 ID、L=CE+KL+0.1·逐token相对update误差、teacher 全 detach、可微 readout、固定超参、每专家样本/梯度/权重记录、新 cache/environment 身份、prefix hash 核对后派生、不沿用 pair 冻结 manifest）；18.3（先适配后 RL 的门槛：完整 selection NLL 恶化 ≤0.02 nat 且真实执行更快；joint9、prior 0.9 argmax+0.1/8、prev expert one-hot、每轮 1 次 expert call、lambda 单位 extra_single_expert_round、reference=适配单专家 R1、canonical 全 8 选 1 物理比例 8 计账、独立 run/resume）。backbone 配置保持 top_k=2，未全局改动。

## 执行的实际命令

```bash
# toy 契约测试（CPU）
.venv/bin/python -m unittest tests.rl_lm.test_single_actions tests.rl_lm.test_single_executor \
  tests.rl_lm.test_single_adapt tests.rl_lm.test_single_online tests.rl_lm.test_single_rloo
# 全项目回归
.venv/bin/python -m unittest discover -s tests -t .   # 日志 runs/rl_lm_single_v1_task_g_full_tests.log
# 真实适配 pilot（物理 GPU 2，CUDA_VISIBLE_DEVICES=2 -> 进程内 cuda:0）
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m experiments.rl_lm_single_adapt \
  --config configs/rl_lm/single_adapt.json \
  --checkpoint runs/next_2026_10_26/baseline_100m/loop_r1/latest.pt \
  --corpus-dir data/rl_lm_pair_v1/corpus \
  --tokenizer runs/data/fineweb_edu_100m/tokenizer.json --device cuda:0
# gate 通过后：single RLOO 500-update pilot
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m experiments.rl_lm_single_train \
  --config configs/rl_lm/single_rloo.json \
  --checkpoint runs/next_2026_10_26/baseline_100m/loop_r1/latest.pt \
  --corpus-dir data/rl_lm_pair_v1/corpus \
  --tokenizer runs/data/fineweb_edu_100m/tokenizer.json --device cuda:0
```

## 通过的测试及实际数值

- 全项目 738 项测试通过、1 跳过、0 失败（新增 40 项 single 契约：编号/STOP、tie 最低 ID、9 类分布归一、prior 数值、执行器幅度恰为 1 且不调用 MoE.forward（monkeypatch 负对照）、canonical/sparse 一致、物理/逻辑=8、terminal-only、前层不动、可微 readout 梯度到专家 slice 且冻结 head 输入有梯度、teacher 全 detach、逐 token 相对误差、适配白名单精确、扰动 student 蒸馏回收、adapt resume 逐位一致、在线闭环 R1/R2 STOP 语义、RLOO rollout/replay/resume 逐位一致）。
- 适配 pilot（真实 100M，1000/1000 updates，0.305 GPU·h < 6 上限）：
  - update 0（未适配）degradation 已 0.011155 nat；随训练单调下降：100→0.010517，500→0.009613，1000→0.009523 nat（完整 selection 256 序列 / 261637 token，pair teacher NLL 4.179894）。
  - 最终 CE 4.481、KL 8.7e-4、逐 token 相对 update 误差 1.1e-3（batch 均值）。
  - 每专家样本总数（256000 次）：[7638, 3725, 3344, 135729, 3183, 51320, 43825, 7236]；平均梯度范数与样本数一致（expert 3 最高 0.533）；所有 slice 的 |Δw|（含未选中 slice，weight decay 所致，规范允许）终值 0.276–0.289。
  - 执行时间（8192 行、BF16、CUDA events、30 warmup + 200 次）：原 top-2 末层 FFN 3.507 ms，单专家 sparse 1.866 ms，加速 1.880×。
  - **gate.json：passed=true**（质量 0.009523 ≤ 0.02 且执行更快），student sha256 `2ad634de…d2469`。
- single RLOO pilot（500/500 updates，0 回滚 0 拒绝，训练段 37.2 s / 0.0103 GPU·h；总 CLI 53 s 含 reference 预计算与评价初始化）：
  - reference（适配单专家 R1 argmax）greedy NLL 4.182843（128 条 selection 序列）；pair teacher 外部对照 4.173380。
  - learned greedy NLL 在 0/100/…/500 均为 4.182858（best=update 0）：500 步内 greedy 未离开 argmax 先验，无新增质量收益，也无退化；采样 reward 均值后期略负（探索采到非 argmax 专家的正常代价）。
  - 探索分布：专家使用总计 [9103, 5532, 5333, 130950, 4995, 49212, 42131, 8744]，teacher 一致率末步 0.955。
  - 账本：logical single rounds=logical expert rows=256000，physical expert rows=2048000，**物理/逻辑恰为 8**；head 物理行 256000；resume/warmstart 互斥、原子 checkpoint、frozen hash 稳定。

## 未运行/skip/blocked 项目及具体原因

- 全项目回归中 1 项 skip 为 pair-v1 既有跳过项，与 S1 无关。
- single-RL 未扩展到更多 updates/seeds/lambda：500 步 pilot greedy 无增益，按第 17 节停止条件不扩大。
- R2/多轮 single（STOP 实际启用）不在 S1 第一版（配置固定 R1，min=max=1）。

## 与规范的差异

- 规范 18.2 未固定适配阶段 head 精度；实现采用显式 FP32 head（teacher/student 同一函数），仅 KL 按要求 FP32——production 评价仍走 BF16 head 口径。
- 适配阶段专家 matmul 在 BF16 autocast 下计算（与生产环境 frozen_compute_dtype 一致）。
- SingleTrainer 的 `gpu_hours` 在首次 pilot 运行为 0（只记 wall_time_seconds）；修正 `_finish` 记录实测 wall time 后按同 seed 重跑，结果逐位一致，现账本 gpu_hours=0.01033。旧运行目录已删除重跑，无伪造历史。
- toy 测试配置保留生产 schema 值（如 adaptive_layer: 11），terminal 约束由执行器在真实模型上强制——沿用 pair-v1 测试的既有做法。

## 剩余步骤

- S1 上线候选评价（更多 seeds、newtest 确认）需按第 16/17 节另行决定；newtest 保持封存。
- 联合更新 actor+experts 与前层 RL 属 18.4 明确排除项，未进入。

## 输出文件与 checkpoint 身份

- 适配：`runs/rl_lm_single_v1/adapt/`（gate.json、student_final.pt sha256 `2ad634de69d5c420ccaf34490ee9d289fe966753116a7bd0de7e54d1e35d2469`、metrics/evaluations.jsonl、cost_ledger.json、executor_timing.json、pair_teacher_selection.json、preflight.json）。
- RL：`runs/rl_lm_single_v1/rloo_seed17/`（checkpoint_best=initial，sha256 `134815cb17fc4d3025a3fd2032d2ae38a58db600854b50d2329576c425e80a71`、reference_ce.{npy,json}、selection_manifest.json、final_summary.json、cost_ledger.json）。
- cache：`data/rl_lm_single_v1/cache/{controller_train,selection,calibration}`（`rl-lm-single-cache-v1` 身份 + 派生自 pair cache 的 manifest hash 记录）。
- 当前分支 `research/rl-lm-pair-v1`，未提交或推送。
