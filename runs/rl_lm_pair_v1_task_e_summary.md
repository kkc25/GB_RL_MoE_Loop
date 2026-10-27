# 任务卡 E 验收记录：真实 R1 pilot(seed17, λ=0)

阶段：E——真实 R1 pilot，完成。2000/2000 updates 在真实 100M checkpoint 与新四角色语料上跑完；
结论为**负结果**(greedy 自然语言收益 ≈1.4e-5 nat,远低于 0.005 nat 预登记阈值),按指令不启动 R2/RL 扩展。

## 本次修改文件与用途

- `experiments/export_rl_lm_source.py`(新增):经 `data.prepare_fineweb.fetch_documents` 固定下载器导出
  本地 UTF-8 原文 JSONL;revision 必填且对 Hub 校验,记录文件 SHA256/shard/行数,拒绝覆盖。
- `experiments/rl_lm_evaluate.py`(新增):规范 13.4/16.1–16.3/20 的评估入口。输出 legacy_top2_r1、
  canonical_pair_r1、initial_actor_greedy、learned_greedy(在线完整 forward,deployment backend)与
  learned_sampled/initial_actor_sampled(cache 侧固定评估随机流 G_eval=16 诊断);主 NLL 为 token 加权,
  配对区间先按近重复组聚合再 2000 次组级 bootstrap。`--split newtest` 必须给 `--candidate-manifest`
  且对 `development_only` 语料拒绝;trainer 仍无任何读取 newtest 的路径(本任务未打开 newtest)。
- `configs/rl_lm/corpus.json`(填入真实值):source_jsonl、revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`、
  tokenizer、四个排除目录(旧 LM 100m、旧 LM debug、旧 controller、旧 confirmation)。token 预算保持
  train 1,048,576、其余各 262,144。
- `experiments/rl_lm_train.py`(入口 bug 修复,不动 D 的算法):`OnlineSelectionEvaluator._forward_nll`
  原来对 [128,1024,32768] logits 一次性 `.float()`+全量 CE,真实规模下需 ~17GiB 额外显存,update-0
  评估即 OOM(见 failed_attempt_1/)。改为按 8192 行分块累计 CE;评估口径(token 加权 mean NLL、
  参照缓存、tie 规则)不变。
- `tests/rl_lm/test_config_and_roles.py`(适配):T24 原断言"签出的 corpus.json 必须为 null 模板";
  任务 E 按规范 4.6 要求在正式运行前填入真实路径。负对照改为内存中构造 null/空排除配置,
  并新增正对照(填好的真实配置必须通过校验)。
- `runs/rl_lm_pair_v1/task_e/`(新增报告与产物,见末尾清单)。

## 已实现的规范条款

- §4.1/4.6:本地原文 JSONL(记录实际 SHA256 `de1276dcae11a365755ebcd55f6e24b520467f52b92fb9495b179f799c62542a`,
  726,000 行,shard `sample/10BT/000_00000.parquet`,revision 逐字记录);四种历史格式(旧 LM
  fineweb-pipeline-v1 ×2、controller-corpus-v1、fixed-depth-confirmation-v1)全部按 doc map 实际 used
  rank 连接;旧 LM 缺 near_hash → 先完整扫描本次原文重建(202,569 条全部重建,0 未解析),
  再排除再分配;`development_only=false`;四角色与排除集合的 doc/text/near 交集全为 0;
  排除 108,621 文档,保留 617,379;四角色 tokens 全部达到请求值(不少训、不复制补齐)。
- §13.1/13.4:R1 默认配置,null 路径经 CLI 填入并写 `resolved_config.json`;checkpoint 实际 SHA256
  `0ccb04fe…7b6202` 与期望一致(preflight/load_backbone 双重核对)。
- §15.1:新语料上 2 序列 × 128 合法位置四方对照(见下)。
- §16.1:update=0 先评估并存 `checkpoint_initial.pt` 初始化 best;best 含初始 actor 候选;
  统一 delta/gain 符号;固定 128 条 selection 序列完整生产 greedy forward。
- §17.2:单 seed17、λ=0、最多 2000 updates、B=128 G=4、on-policy RLOO(无 oracle 训练、无 PPO 复用)。

## 执行的实际命令

```bash
.venv/bin/python -m experiments.export_rl_lm_source --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --limit-shards 1 --output data/rl_lm_pair_v1/source/fineweb_edu_sample10bt_shard0_rev87f09149.jsonl
.venv/bin/python -m experiments.rl_lm_corpus --config configs/rl_lm/corpus.json
.venv/bin/python -m experiments.rl_lm_preflight --config configs/rl_lm/pair_r1.json --checkpoint .../loop_r1/latest.pt \
  --corpus-dir data/rl_lm_pair_v1/corpus --tokenizer runs/data/fineweb_edu_100m/tokenizer.json --device cuda:1   # not_built
.venv/bin/python -m experiments.rl_lm_prepare  <同上>                                                            # 建三角色 cache
.venv/bin/python -m experiments.rl_lm_preflight <同上>                                                            # ready
.venv/bin/python -m experiments.rl_lm_verify_r1 --data-dir data/rl_lm_pair_v1/corpus --split selection \
  --sequences 2 --max-positions 128 --device cuda:1 --output-dir runs/rl_lm_pair_v1/task_e/verify_r1
.venv/bin/python -m unittest discover -s tests/rl_lm && .venv/bin/python -m unittest discover -s tests -t .
.venv/bin/python -m experiments.rl_lm_train --config configs/rl_lm/pair_r1.json --checkpoint ... --corpus-dir ... \
  --tokenizer ... --device cuda:1                                                                               # 2000 updates
.venv/bin/python -m experiments.rl_lm_evaluate --config runs/rl_lm_pair_v1/r1_seed17/resolved_config.json \
  --actor-checkpoint runs/rl_lm_pair_v1/r1_seed17/checkpoint_best.pt --split selection --mode online --device cuda:1
.venv/bin/python runs/rl_lm_pair_v1/task_e/oracle_pair_sweep.py.txt   # oracle 上限诊断(非 actor 输入)
```

## 通过的测试及实际数值

- 测试:`tests/rl_lm` 101/101 通过;全项目 693 项,692 通过 + 1 既有 GPU skip,0 失败
  (`runs/rl_lm_pair_v1_task_e_tests.log`)。
- Preflight:model_and_raw_data=verified;三角色 cache ready(65,536 / 8,192 / 2,048 状态)。
- §15.1 新语料 128 状态核验(`task_e/verify_r1/summary.json`):gate ties=1(按 canonical tie-break 解决,
  canonical/legacy pair 差异 0);初始 actor 与 canonical 动作差异 0;canonical−legacy CE |差|
  mean 4.52e-4 / max 1.98e-2 / p95 1.32e-3;initial−canonical mean 4.47e-6;backend bf16_linear_fp32_ce_v1。
- 训练:2000/2000 updates 全成功,optimizer_attempts=2000(0 重试、0 KL 回滚、0 backoff),
  step KL mean 最大 2.41e-6(阈值 0.02/0.2);replay 前后一致误差全程逐位 0;冻结 LM 参数 hash 不变;
  replay_max_abs_error=0;grad clip 从未触发(mean 0.044,max 0.411)。

### 学习曲线(每 100 update 节选;完整序列在 metrics.jsonl / evaluations.jsonl / task_e/learning_curve.json)

| update | reward(λ=0) | zero_adv | teacher一致率 | 熵 | joint最大概率 | 梯度范数 | selection greedy NLL | gain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0/1 | −0.0128 | 0.625 | 0.895 | 0.635 | 0.904 | 0.014 | 4.173303085 | 0 |
| 500 | −0.0034 | 0.828 | 0.955 | 0.289 | 0.957 | 0.034 | 4.173303085 | 0 |
| 1000 | −0.0027 | 0.883 | 0.963 | 0.225 | 0.960 | 0.022 | 4.173307370 | −4.3e-6 |
| 1100(best) | — | — | — | — | — | — | 4.173289051 | +1.40e-5 |
| 1500 | −0.0051 | 0.883 | 0.963 | 0.209 | 0.966 | 0.068 | 4.173295755 | +7.3e-6 |
| 2000 | −0.0054 | 0.852 | 0.945 | 0.218 | 0.960 | 0.036 | 4.173297053 | +6.0e-6 |

探索:R1 固定 1 轮、STOP 在 r=0 非法(全程 stop_fraction=0,rounds 恒为 1);每 context G=4 的
distinct paths 均值 1.18;zero_adv 比例 0.63→0.85(信号稀疏,多数 context 四次采样同动作)。

### 最终 selection 评估(`task_e/evaluate_selection_best.json`,128 序列、130,816 有效 token,sparse_merged_v1)

| 候选 | NLL | vs canonical |
|---|---:|---:|
| legacy_top2_r1 | 4.173295975 | +7.2e-6(实现/后端差) |
| canonical_pair_r1(reference) | 4.173303127 | 0 |
| initial_actor_greedy | 4.173303127 | 0(与 canonical 逐位一致) |
| learned_greedy(best, update 1100) | 4.173289299 | **+1.38e-5** |
| learned_sampled(2048 状态×16,固定流) | 4.188727 | −3.69e-3 vs ref_ce |
| initial_actor_sampled(同状态同流) | 4.194385 | −9.35e-3 vs ref_ce |

配对 bootstrap(129 个近重复组、2000 次、95%):learned−canonical Δ=−1.40e-5,
CI [−3.48e-5, +4.8e-7]。收益比预登记实际阈值 0.005 nat 小约 350 倍,且与 legacy/canonical 实现差
(7e-6)同量级 → 按 §15.1/16.3 判为**未证实/负结果**。

### Oracle 上限诊断(任务 E 失败分类用,`task_e/oracle_r1/`,读 label,仅诊断)

128 train + 128 selection 状态穷举 28 pair:93.8%/96.9% 状态下 canonical pair 不是逐 token 最优;
oracle 平均上限 **+0.270 / +0.291 nat/token**(中位 ~0.22)。即动作空间内确实存在大幅逐 token 收益
(样本上界,非可学习性保证)。

## 负结果分类(§17.2 三分法)

1. 接口错误——排除:全部契约测试、§15.1 核验、replay 逐位一致、KL 监控、冻结 hash 均通过。
2. 策略未学会已有信号——部分排除:同一固定评估流/状态上 sampled 惩罚从 −9.35e-3 收窄到 −3.69e-3 nat,
   熵 0.635→0.218、teacher 一致率 0.89→0.95,RLOO 确实把概率质量推向更优 pair;但 greedy NLL 几乎不动。
3. 动作空间收益不足——被 oracle 诊断否定(上界 ~0.27 nat/token)。
   结论:在 seed17/λ=0/2000 updates/B128 G4/先验 ε=0.1 的预算内,actor 只学到了收益的极小份额;
   可能原因包括探索信号稀疏(zero_adv 均值 0.84)、单步 pair 选择对整句 NLL 的摊薄、训练预算不足。
   按指令与 §17.2 记录负结果,不自动扩大预算/多 price/多 seed,不启动 R2。

## 未运行/skip/blocked 项目及具体原因

- R2/R2 pilot/price 扫描/多 seed/profile:任务指令禁止扩展;且 M2 未达 0.005 nat 继续阈值。
- newtest 评估:方案未冻结,且规范要求冻结 candidate manifest 后才打开;本任务只读 selection。
- 速度三层测量(§16.4):属任务卡 F;R1 不改变计算量(固定 1 轮),本阶段无计时声明。
- `tests/rl_lm` 1 项 skip 为既有 GPU 标记测试,与本卡无关。

## 与规范的差异

1. `configs/rl_lm/corpus.json` 由 null 模板填入真实值(规范 4.6 明确要求正式运行前填入);
   T24 负对照相应改为内存构造(见"修改文件")。
2. `experiments/rl_lm_train.py` 评估入口的分块 CE 修复(真实规模 OOM;算法与口径不变)。
3. 账本 `gpu_hours` 字段在 `rl_lm/ledger.py` 中只有声明、无累计代码(D 的既有缺口,未重写);
   权威成本为 `cost_ledger.json` 的 `wall_time_seconds=159.4`(单卡 A6000 ≈ 0.044 GPU·h,
   峰值显存 allocated 21.35 GiB/reserved 23.56 GiB),加 cache 构建 ~6.5 min 与评估 <1 min,
   远低于 M2 的 4 GPU·h 预算。
4. 训练 159s 完成 2000 updates(全部命中 h0 cache 的单步环境,rollout 均值 9.3ms、replay+step
   均值 14ms);metrics.jsonl 逐 update 含逻辑/物理账本(1,024,000 episodes;物理专家行
   8,192,000 = 8×逻辑行,canonical 后端如实计账)。
5. 训练器 update-0 greedy 4.173303085 与 evaluate 重算 canonical 4.173303127 差 4e-8:
   两次独立 bf16 forward 的数值抖动,小于 tie 容差 1e-6,如实并列记录。

## 失败与修复记录(未伪造、已修复契约)

- attempt 1:update-0 评估 OOM(16 GiB CE 缓冲),trainer 正常写 `failure.json`;
  存 `task_e/failed_attempt_1/`。修复:入口分块 CE。
- attempt 2:修复入口后未重建 cache,`load_cache` 按 §4.5 正确拒绝(source 身份中
  untracked python hash 变化),存 `task_e/failed_attempt_2_cache_identity.log`;
  处理:冻结全部代码后删除旧 cache 重建并重跑 preflight。此后未再改动任何 .py。

## 剩余步骤

- 任务卡 F(R2 warmstart/STOP 闭环/稀疏后端计时):需主 agent 明确决策;按当前负结果与指令不启动。
- 若未来继续:可考虑在同一冻结协议下调查"信号稀疏"(zero_adv 0.84)与预算/先验的关系;
  oracle 上界表明动作空间不是瓶颈。
- newtest 保持未开封;confirmation 入口已能对 development_only 拒绝,本语料 development_only=false。

## 输出文件与 checkpoint 身份

- run 目录 `runs/rl_lm_pair_v1/r1_seed17/`:resolved_config.json、preflight.json(status=ready)、
  metrics.jsonl(2000 行)、evaluations.jsonl(21 点)、cost_ledger.json、selection_manifest.json、
  checkpoint_manifest.json、final_summary.md、source_identity.json。
- checkpoint:initial `00d71938c683ba911bdde7e917b34ae546ab2a4685b4f1261950b24189013732`;
  **best(update 1100)`290e1f25f0320701988a9b28121aad0e591f2bd9ea0647dd0f9dd3fe0ac8af09`**;
  latest(update 2000)`dbc353b4a162b88f8fededa38bb71214c00e0f0cce938f50d9e883610b015d6a`。
  停止判定:跑满计划 2000 updates(status=completed),无早停;best 由 selection greedy NLL
  最低规则选出(tie 容差 1e-6,候选含初始 actor);未取得 ≥0.005 nat 增益,best 仅比初始好 1.4e-5。
- 语料 `data/rl_lm_pair_v1/corpus/`(manifest + exclusion_coverage.json;development_only=false);
  cache `data/rl_lm_pair_v1/cache/{controller_train,selection,calibration}/`(manifest 身份含 checkpoint/
  config/tokenizer/packed/documents hash、layer 11、gamma 0.5、prefix bf16→FP32 存储)。
- 报告与产物:`runs/rl_lm_pair_v1/task_e/`(本摘要、learning_curve.json、evaluate_selection_best.json、
  exclusion_coverage.json、source_manifest.json、verify_r1/、oracle_r1/、oracle_pair_sweep.py.txt、
  failed_attempt_1/、failed_attempt_2_cache_identity.log、final_summary.md、selection_manifest.json、
  cost_ledger.json、preflight.json、resolved_config.json 副本)。
- 日志:`runs/rl_lm_pair_v1/task_e_{export,corpus,prepare,prepare2,train,evaluate}.log`、
  `runs/rl_lm_pair_v1_task_e_tests.log`。未 commit/push。
