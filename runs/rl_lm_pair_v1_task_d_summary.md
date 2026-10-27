# 任务卡 D 验收记录

阶段：D——训练事务与恢复，完成。未进行长训练；真实 GPU/checkpoint 训练属任务卡 E。

本次修改文件与用途：

- `rl_lm/checkpoint.py`(新增):原子保存(tmp+fsync+`os.replace`+指针 manifest)、`resume_exact`、
  `warmstart_actor`、完整 RNG 捕获/恢复(Python/NumPy/torch CPU/CUDA + stateless sampler 版本)。
- `rl_lm/trainer.py`(新增):采样→replay→单次受监控 update 的训练状态机;按 successful update 计数的
  warmup+cosine LR schedule 与持久 `lr_scale`;step 前 deepcopy 快照、完整 categorical KL(old‖new、
  FP64 累加、跳过非法 STOP)、至多一次重试、run 级累计 backoff 上限(默认 4)、回滚不抹除资源账本;
  固定 epoch permutation(Philox 按 `(actor_seed, epoch)` 重建)、data cursor、`checkpoint_initial/
  latest/best` 与 best 选择记录;`metrics.jsonl`/`evaluations.jsonl`/`cost_ledger.json`/
  `selection_manifest.json`/`final_summary.md`/`failure.json`/`update_rejected.json`。
- `experiments/rl_lm_train.py`(新增):训练 CLI,只做接线;`--resume`/`--warmstart-actor` 互斥,
  `--stop-after` 为绝对上限;要求 preflight `ready`;resume 时除 `runtime.device` 外配置必须逐字段一致。
- `tests/rl_lm/test_resume.py`(新增):T19 及 warmstart 语义测试,含子进程驱动模式。
- `tests/rl_lm/test_update_transaction.py`(新增):T20 KL 回滚事务测试。

已实现的规范条款:11.2(固定 update 顺序)、11.4(单步 KL 监控与一次回滚)、11.5(schedule 按成功
update、`j=successful_updates+1`、LR 倍率持久、累计回退上限)、12.1–12.4(checkpoint 字段、原子保存、
resume 与 warmstart 分离、恢复验收)、13.4(CLI 互斥与绝对上限)、14.1–14.3(逐 update 日志、逻辑/物理
账本、run 目录文件)。无 PPO、critic、多 epoch 复用。

执行的实际命令:

```bash
.venv/bin/python -m unittest tests.rl_lm.test_update_transaction -v
.venv/bin/python -m unittest tests.rl_lm.test_resume -v
.venv/bin/python -m unittest discover -s tests/rl_lm
.venv/bin/python -m unittest discover -s tests -t .
.venv/bin/python -m experiments.rl_lm_train --help
.venv/bin/python -m experiments.rl_lm_train --config configs/rl_lm/pair_r1.json   # null 路径按契约报错
```

通过的测试及实际数值:

- T19(`test_resume.py`,4 项):CPU toy(tiny LoopMoE,B=8、G=4、microbatch=3、N=64)连续 20 步 vs
  10 步→保存→**新进程**恢复→10 步:逐 update 的 episode ID 哈希、uniform 表哈希、动作哈希、loss、LR
  全部逐位一致;最终 actor 参数、AdamW state(含 step/moments)、scheduler_state(lr_scale/backoffs)、
  training_state(epoch=2、cursor=32、visited 集合)、torch/NumPy/Python RNG、selection_state(best)与
  ledger 完全一致。恢复跨过两次 epoch 重建边界。最终 actor_loss=8.092681e-05。
- T20(`test_update_transaction.py`,4 项):toy 配置 peak LR=0.05 使单步 KL 可测。
  - 重试成功:全 LR 步 KL mean=6.612e-07,半 LR 步 1.341e-07;阈值置两者之间 → 第一次违规、快照完整
    恢复、恰好一次重试成功;actor 与独立模拟的半 LR 克隆逐位一致;`lr_scale=0.5` 持续到下一 update
    (实测下一 update LR=base_lr(3)×0.5);ledger.optimizer_attempts 按 2 次记账。
  - 重试耗尽:阈值 0 → 恰好 2 次 attempt 后 `UpdateRejected`;actor/optimizer 逐位恢复;`lr_scale` 回到 1;
    `update_rejected.json`(update=2,reason=kl_retry_exhausted);`checkpoint_latest.pt` 仍停留在 update 1;
    被拒 update 的 rollout 成本(64 episodes、物理专家/读出行)保留在账本中。
  - run 级预算:`max_lr_backoffs_per_run=0` → 首次违规即终止,仅 1 次 attempt,reason=kl_budget_exhausted。
  - 失败 run:写出 `failure.json` 与 `final_summary.md`,保留 update-0 的完整 checkpoint。
- CLI:`--resume` 与 `--warmstart-actor` 同给 → argparse 退出码 2("not allowed with argument")。
- 全量回归:`tests/rl_lm` 101 项全部通过;`tests` 全项目 693 项,692 通过、1 skip(既有 GPU 跳过项),
  0 失败。任务卡 C 验收时为 685 项,本卡新增 8 项。

未运行/skip/blocked 项目及具体原因:

- 真实 100M checkpoint 与 GPU 上的训练/评估:属任务卡 E,本机此阶段未执行;CLI 路径(preflight 门控、
  真实 cache provider、online selection 评估器)已实现但未经真实数据验收。
- NaN 强制终止路径:代码已实现(step 后参数有限性检查→恢复→`update_rejected.json`),未构造可靠 NaN
  用例,toy 环境无法自然触发。
- `tests/rl_lm` 中 1 项 skip 为既有 GPU 标记测试,与本卡无关。

与规范的差异:

1. 评估器为可接线参数:CLI 使用规范 16.1 的在线完整 forward selection 评估
   (`forward_with_last_ffn_policy`,deployment backend,token 加权 NLL,canonical teacher 参照);
   toy 测试使用 `GreedyCacheEvaluator`(cache 侧联合 argmax NLL 诊断)。正式 run 的 checkpoint 选择始终
   走在线评估;toy 评估器仅用于 T19/T20 的 CPU 契约验证。
2. toy 配置保留 `backbone.adaptive_layer=11`(RLConfig  schema 的生产值);实际模型的终端层约束由
   executor 按 `layer == model.config.layers - 1` 对 tiny 模型(第 1 层)强制。配置与模型的一致性在
   CLI `load_backbone` 中对真实 backbone 校验。
3. `ledger.gpu_hours` 仅在 CUDA 设备上累计;CPU toy 运行保持 0.0,以保证 T19 账本逐位可比(wall time
   另记于 metrics/cost_ledger,不参与恢复等价断言)。
4. `metrics.jsonl` 每个 update 写完整条目(规范 14.1 的"每次 update 至少包含");`logging.interval_updates`
   仅控制 stdout 摘要频率。

剩余步骤:任务卡 E——真实 checkpoint/corpus/tokenizer 填入后跑 preflight、128 状态核验与 R1 pilot;
F(R2 闭环与 profile);G(可选 S1)。

输出文件与 checkpoint 身份:本卡无真实训练 run;toy run 目录含 `resolved_config.json`、
`source_identity.json`、`metrics.jsonl`、`evaluations.jsonl`、`cost_ledger.json`、
`checkpoint_initial/latest/best.pt`、`checkpoint_manifest.json`、`selection_manifest.json`、
`final_summary.md`(失败时另有 `failure.json`/`update_rejected.json`)。测试日志:
`runs/rl_lm_pair_v1_task_d_tests.log`。未 commit/push。
