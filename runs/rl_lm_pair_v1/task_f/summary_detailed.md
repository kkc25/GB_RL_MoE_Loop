# 任务卡 F 详细验收记录（附录于 summary.md）：R2 warmstart/STOP 闭环/500 步 λ=0 pilot/三层 profile

阶段：F，完成。R2 λ=0 pilot 为**负结果**（相对 P1 best 增益恰 0.0，CI [0,0]；固定 teacher R2 恶化
+0.044 nat），按 §17.3 与任务指令停止：不开 price grid、不多 seed、不 resume 至 2000、不打开 newtest。
本文件为详细版；`summary.md` 为并行执行者所写的概要版，两者数值一致（profile 数值以本文件/
`profile/profile_report.json` 的最终版为准，理由见"差异 3"）。

## 本次修改文件与用途

- `rl_lm/ledger.py`（仅修 gpu_hours 累计）：新增 `RLLedger.record_wall_time(seconds, gpus=1)`
  唯一累计入口，只接受实测 wall time；不加字段（旧 checkpoint 的 ledger 段仍可 `from_dict`），历史 run
  的 0.0 记录不回改，权威成本取各 run `cost_ledger.json` 的 `wall_time_seconds`；对账见
  `task_f/gpu_hours_reconciliation.json`（R1 159.42 s→0.0443 GPU·h；R2 45.45 s→0.0126 GPU·h）。
- `tests/rl_lm/test_online_parity.py`（T22 R2 部分；原 R1 测试逐字未动）：新增 `R2OnlineClosedLoopTests`
  5 项 + `ScriptedPolicy`。
- `experiments/rl_lm_profile.py`（新增，与并行执行者合稿）：§16.4 三层计时；`--physical-gpu-index`
  （对方补充，处理 CUDA_VISIBLE_DEVICES 映射）；`torch.cuda.set_device`（修正无参 CUDA event/显存计数
  默认落到 cuda:0 的问题）；独占检查忽略自身 pid。
- `runs/rl_lm_pair_v1/task_f/`：`verify_r2_parity.py`、`evaluate_r2_candidates.py`、
  `newtest_audit.json`、`gpu_hours_reconciliation.json`、cache 哈希记录、本附录。

## 已实现的规范条款

- §5.2/10.2（T22 R2 toy，CPU FP32 逐位断言）：
  - `[pair,STOP]` 恰 1 轮；STOP 后 h 逐位不变（与同 pair 的 R1 forward `torch.equal`，logits 同）；
  - `[pair,pair]` 恰 2 轮；actor 调用计数=2（**无第三个 STOP log-prob**）；
  - 混合批次中已终止行 h 在其他行第二轮执行期间逐位不变；r=0 STOP fail-fast；
  - canonical teacher R2 与独立逐轮参考（每轮重算 gate→canonical_pair→executor.step）动作完全一致、
    hidden/logits 逐位一致。
- 真实 checkpoint 小规模 R2 online parity（`verify_r2/summary.json`；selection 2 序列×128 合法位置，
  sparse_merged_v1 + bf16_linear_fp32_ce_v1）：canonical R2 全部 rounds=2，与独立单行两轮参考动作
  不一致 **0/128**；CE |差| mean 3.20e-4 / max 3.09e-2 / p95 5.96e-5；hidden |差| mean 7.71e-4 /
  max 1.23e-2（与任务 E 的 R1 实现差 4.5e-4 同量级，bf16 batch 形状抖动）；**STOP 闭环在真实
  checkpoint 上逐位成立**：R2 `checkpoint_initial.pt`（warmstart actor）greedy 全部 rounds=1、slot1 全
  STOP，final hidden/logits 与 P1 best（max_calls=1）`torch.equal`。
- §12.3 warmstart：`--warmstart-actor` 指向 P1 best（update 1100，sha256 `290e1f25…af09`，与 manifest
  一致）；只继承非 stop_head 参数、stop_head 清零（初始 hazard 0.5）；新 optimizer/scheduler、数据游标
  与账本重置；`warmstart_from` 记录 parent run_id/update 1100；与 `--resume` 互斥；**schedule 不变**
  （planned 2000、peak LR 5e-5；`--stop-after 500` 仅为生命周期上限，未衰减到 500）。
- §13.2：pair_r2.json 恰为规范差异集。
- §16.1：update=0 先评估并保存 `checkpoint_initial.pt`；候选含初始 actor；选择仅用 selection 128 序列
  完整生产 greedy NLL（λ=0 → 纯 NLL），tie 1e-6 先少 rounds 再早 update；七候选并列比较见下。
- §16.4：三层 profile（下）；32 固定 selection 输入、30 预热、5×200、batch1 主/batch8 单列、CUDA
  events+wall 双记录、原始时间全量保存、独占卡（开始前无其它进程）、无 compile/profiler。
- §17.3：门槛未过 → 停止（下）。

## 执行的实际命令

```bash
.venv/bin/python -m unittest tests.rl_lm.test_online_parity            # 6/6 OK
.venv/bin/python -m unittest discover -s tests/rl_lm                   # 106/106 OK
.venv/bin/python -m unittest discover -s tests -t .                    # 698 项, 697 过 + 1 既有 GPU skip
PYTHONPATH=. .venv/bin/python runs/rl_lm_pair_v1/task_f/verify_r2_parity.py --device cuda:1
.venv/bin/python -m experiments.rl_lm_prepare --config configs/rl_lm/pair_r1.json --checkpoint .../loop_r1/latest.pt \
  --corpus-dir data/rl_lm_pair_v1/corpus --tokenizer runs/data/fineweb_edu_100m/tokenizer.json --device cuda:1
.venv/bin/python -m experiments.rl_lm_preflight --config configs/rl_lm/pair_r2.json <同上>   # ready
.venv/bin/python -m experiments.rl_lm_train --config configs/rl_lm/pair_r2.json <同上> \
  --warmstart-actor runs/rl_lm_pair_v1/r1_seed17/checkpoint_best.pt --stop-after 500
PYTHONPATH=. .venv/bin/python runs/rl_lm_pair_v1/task_f/evaluate_r2_candidates.py --device cuda:1
.venv/bin/python -m experiments.rl_lm_profile --config configs/rl_lm/pair_r2.json <同上> \
  --device cuda:1 --physical-gpu-index 1 \
  --r1-actor runs/rl_lm_pair_v1/r1_seed17/checkpoint_best.pt \
  --r2-actor runs/rl_lm_pair_v1/r2_lambda0_seed17/checkpoint_best.pt --batch8 \
  --output-dir runs/rl_lm_pair_v1/task_f/profile
```

## 通过的测试及实际数值

### 训练健康度（runs/rl_lm_pair_v1/r2_lambda0_seed17/）

500/500 updates 全成功；optimizer_attempts=500（0 重试/0 KL 回滚/0 backoff）；step KL mean 最大 7.7e-6、
max 最大 1.57e-4（阈值 0.02/0.2）；grad clip 从未触发（norm mean 0.165 / max 0.561）；冻结 LM hash
全程不变；warmstart 后 update-0 greedy NLL=4.173289050794628 与 P1 best 的 selection NLL 完全一致。
replay_max_abs_error mean 7.3e-7 / max 2.9e-6（R1 为逐位 0；R2 因每批 decision 数随 STOP 比例变化、
replay microbatch 边界移动产生 FP32 结合序差；远低于容差，如实记录）。

### 探索/梯度/成本（λ=0）

- stop_fraction（采样，r=1 处）0.271→0.426（mean 0.368）；rounds 直方图 [0,278,234]→[0,436,76]
  （/512 episodes）——λ=0 下 actor 自发学会更多 STOP，与"第二轮有害"方向一致；
  teacher_pair_agreement 0.957→0.966；entropy 0.574→0.286；joint_max_prob 0.723→0.907；
  distinct_paths/context 2.10→1.44；zero_adv 0.070→0.633。
- 逻辑/物理账本：episodes 256,000；logical_pair_rounds 323,389（26.3% 轨迹第二轮）；
  logical_expert_rows 646,778；physical_expert_rows 4,096,000（canonical 固定形状 8×64×chunks 计账）；
  physical_head_rows 256,000；峰值显存 allocated 21.35 GiB / reserved 23.55 GiB；wall 45.45 s
  （≈0.0126 GPU·h，M3 预算 2 GPU·h）。gpu_hours 字段仍为 0.0（trainer 不在本卡可改范围；真实值见
  对账 JSON）。

### 学习曲线（selection 128 序列完整生产 greedy）

| update | greedy NLL | mean rounds |
|---:|---:|---:|
| 0（warmstart 初始，best） | **4.173289051** | 1.0000 |
| 100 | 4.173376453 | 1.0027 |
| 200 | 4.175666167 | 1.0617 |
| 300 | 4.176102685 | 1.0726 |
| 400 | 4.176053460 | 1.0731 |
| 500 | 4.176056520 | 1.0755 |

best=update 0（初始候选），`checkpoint_best.pt == checkpoint_initial.pt`（`837372ef…7aa1`），
标记"未取得增益"；没有任何训练 checkpoint 胜过 warmstart 初始 actor。

### 全候选完整生产 greedy 比较（evaluate_r2_selection.json；128 序列 / 130,816 有效 token）

| 候选 | NLL | rounds | Δ vs canonical_r1 |
|---|---:|---:|---:|
| legacy_top2_r1 | 4.173295975 | 1.0 | −7.2e-6 |
| legacy_top2_r2 | 4.217524529 | 2.0 | +4.422e-2 |
| canonical_pair_r1（ref） | 4.173303127 | 1.0 | 0 |
| canonical_pair_r2（固定 teacher R2） | 4.217555523 | 2.0 | +4.425e-2 |
| r1_best_greedy（P1 update 1100） | 4.173289299 | 1.0 | −1.38e-5 |
| r2_initial_greedy（warmstart） | 4.173289299 | 1.0 | −1.38e-5 |
| r2_learned_greedy（best=初始） | 4.173289299 | 1.0 | −1.38e-5 |

配对组级 bootstrap（129 组、2000 次、95%）：r2_learned−r1_best Δ=0.0 [0,0]；
r2_learned−canonical_r1 Δ=−1.40e-5 [−3.48e-5, +4.8e-7]（P1 既有的 1.4e-5，非 R2 新增）；
canonical_r2−canonical_r1 Δ=**+0.04425** [+0.04129, +0.04710]。sampled 诊断（2048×16 固定流）：
r2 learned/initial sampled_nll 4.2116、mean_rounds 1.499、vs ref_ce −2.66e-2。

**§17.3 判定**：pilot 相对最佳 R1 改善 0（≪0.005 nat），teacher R2 自身 +0.044 nat 有害，训练候选
NLL 更差且 rounds 更多 → 停止；不启动三 price、不 resume、不跑 seed 23/41、不打开 newtest。

### 三层 profile 最终版（profile/profile_report.json；物理 GPU 1 独占，UUID
GPU-03416495-c0b6-fb40-2fe4-8df8c8a5aed2，driver 595.58.03，1800/2100 MHz，torch 2.12.1+cu130，
python 3.12.15；数值为 CUDA event mean/p50/p95 ms；与并行执行者在物理 GPU 2 的独立运行交叉一致，
如 original_moe_r1 46.55 vs 46.99 ms）

batch=1, S=1024（主）：

| 层 | 变体 | mean | p50 | p95 | tokens/s |
|---|---|---:|---:|---:|---:|
| selector | shared norm+gate | 0.165 | 0.163 | 0.173 | 6.22M |
| selector | legacy gate+top-k | 0.243 | 0.242 | 0.250 | 4.21M |
| selector | canonical pair selector | 0.255 | 0.252 | 0.261 | 4.01M |
| selector | RL 全特征+actor+joint | 2.593 | 2.588 | 2.612 | 0.395M |
| executor | 原两槽 | 3.164 | 3.162 | 3.197 | 0.324M |
| executor | merged（sparse_merged_v1） | 3.795 | 3.798 | 3.832 | 0.270M |
| executor | grouped（canonical 固定形状） | 6.274 | 6.268 | 6.311 | 0.163M |
| e2e | 原 MoE R1 | 46.553 | 46.580 | 47.068 | 21,996 |
| e2e | 原 MoE R2 | 50.280 | 50.019 | 51.075 | 20,366 |
| e2e | dense_parameter | 12.681 | 12.678 | 12.788 | 80,748 |
| e2e | canonical_pair_r1 | 53.379 | 52.754 | 59.891 | 19,184 |
| e2e | canonical_pair_r2 | 59.437 | 59.038 | 62.766 | 17,228 |
| e2e | learned_r1（P1 best） | 53.603 | 53.256 | 54.741 | 19,104 |
| e2e | learned_r2（P2 best=初始） | 58.022 | 57.936 | 59.092 | 17,648 |

batch=8, S=1024（单列）：selector RL 10.891 vs gate 系 0.290–0.363 ms；executor 两槽 3.800 /
merged 4.452 / grouped 49.192 ms；e2e 原 R1 65.954、原 R2 70.436、dense 45.764、canonical_r1
105.413、canonical_r2 120.747、learned_r1 106.053、learned_r2 119.840 ms。

显存：各变体稳态 allocated 1.96–2.35 GiB / reserved 3.69–4.22 GiB（逐变体记录于 JSON）；全程峰值
allocated 3.11 GiB / reserved 4.22 GiB；profile 实测 gpu_hours=0.3034（record_wall_time 累计）。

结论（§16.4 要求的如实报告）：
1. RL 选择器 2.59 ms/千 token，约为 gate 系（≤0.26 ms）的 10–16 倍；开销在全量特征拼接（1044 维）
   与 actor MLP。
2. 同 route 执行器：merged 在 batch1 **慢于**原两槽（3.80 vs 3.16 ms，16 次小 kernel 发射），grouped
   最慢（6.27 ms）；batch8 下 grouped 急剧放大（49.2 ms）。当前实现下 sparse_merged_v1 无时间收益。
3. e2e：**所有 RL/online 候选慢于原生产模型**（learned_r1 53.6 vs 原 R1 46.6 ms，+15%；learned_r2
   58.0 ms）；dense_parameter 12.7 ms 比全部 MoE 变体快 ~3.7–4.7 倍。actor 开销超过本环境可省工作，
   按 §16.4 记为负面时间结果。learned_r2 快于 canonical_r2（58.0 vs 59.4 ms）仅因 1 轮即 STOP，
   仍慢于原 R1。

## 未运行/skip/blocked 项目及具体原因

- price grid/多 seed/resume 至 2000/M4/M5/newtest 确认：§17.3 门槛未过（增益 0.0）。
- newtest：未打开。已完成未使用核实（`newtest_audit.json`）；因无候选过门槛，未冻结 candidate
  manifest、未执行 `--split newtest`；确认集封存（若主 agent 认为零结果也需确认，请先冻结 manifest
  再开）。
- 全项目 1 项 skip 为既有 GPU 标记测试（同任务 E）。

## 与规范的差异

1. **cache 因源码身份绑定重建 4 次**（§4.5：identity 含 untracked python 哈希，任务 F 必须改
   ledger/test/新增 profile 与辅助脚本）。每次均在代码冻结后重建；首次误用 pair_r2.json 使 corpus_id
   变化导致 episode_ids 改变，已弃用并改用 pair_r1.json 重建以保持 episode id/data order/随机流规范与
   R1 完全一致。最终 18/18 payload 文件（含 episode_ids）与任务 E 逐字节一致
   （`cache_payload_hashes_before.txt` vs `cache_payload_hashes_final3.txt`）；仅 manifest 的 source
   身份重绑定。r1_seed17/resolved_config.json 被 prepare 重写但逐字节相同（`820a5a54…` 前后一致）。
2. R2 run 内 evaluations.jsonl 的 `reference_nll` 为 canonical teacher **R2**（4.21756，rounds=2，
   trainer 既有行为：reference 随 max_calls）；与 canonical R1/R1 best 的比较由
   `evaluate_r2_selection.json` 补齐；选择规则本身不变。
3. profile 过程中发现并修复两处计时基建问题（均不改变被测算法）：
   a) 无参 `torch.cuda.memory_*`/`reset_peak_memory_stats` 与 CUDA event 默认落 cuda:0，非 0 号卡时
      显存读数为 0——加 `torch.cuda.set_device` 后重测，最终版数值正确；中间两版 JSON 已被最终版覆盖；
   b) 独占检查把自身 CUDA 上下文（set_device 建立，~262 MiB）当作占用——改为忽略自身 pid。
   并行执行者在物理 GPU 2 的独立运行（`profile_full.log`，仅摘要行）与最终版方向/量级一致。
4. `gpu_hours`：trainer 不可改，run 账本字段仍为 0.0；累计方法已入 `rl_lm/ledger.py` 并被 profile
   实际使用（0.3034 GPU·h 实测）；历史 run 以 wall_time 对账，不回改。
5. R2 replay_max_abs_error ~1e-6 而非 R1 的逐位 0（microbatch 边界随 STOP 比例移动）；低于容差，
   如实并列记录。
6. 工作区并行：另一执行者同任务覆盖了 `summary.md` 与中间版 `profile_report.json`；本附录为详细版，
   最终 profile JSON 为修正确认后的版本（含全部原始时间）。

## 剩余步骤

- pair-RL 线（R1/R2）在 seed17 上质量与时间均无收益；是否转向任务卡 G（S1 单专家分支，独立预算）
  或终止，需主 agent/用户决策。
- newtest 封存；未来有候选过 §17.3 门槛时先冻结 candidate manifest 再打开。

## 输出文件与 checkpoint 身份

- `runs/rl_lm_pair_v1/r2_lambda0_seed17/`：resolved_config、preflight（ready）、metrics.jsonl（500）、
  evaluations.jsonl（6）、cost_ledger、selection_manifest、checkpoint_manifest、final_summary 等。
- checkpoint：initial=best（update 0，warmstart）`837372ef20ff7330fb52db85e490fc86de273fb96e50e37e34760a26bd577aa1`；
  latest（update 500）`bb418284b151c763cb19d9e612c771964933235304d2efcf055e8b0a929a9cfc`；
  parent（P1 update 1100）`290e1f25f0320701988a9b28121aad0e591f2bd9ea0647dd0f9dd3fe0ac8af09`。
- `runs/rl_lm_pair_v1/task_f/`：summary.md（并行执行者概要）、本附录、verify_r2/、
  evaluate_r2_selection.json、profile/profile_report.json（全原始时间）、newtest_audit.json、
  gpu_hours_reconciliation.json、cache_payload_hashes_{before,after,final,final2,final3}.txt、
  old_cache_manifests/、两个辅助脚本、profile_smoke/（冒烟，非正式）。
- 日志：`runs/rl_lm_pair_v1/task_f_{prepare..prepare5,train,profile,preflight_final}.log`、
  `runs/rl_lm_pair_v1_task_f_tests.log`。未 commit/push。
