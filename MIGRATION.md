# 迁移清单：数据完整性与可选执行优化

适用于 PR #2 的新契约。迁移不是覆盖旧实验，也不是把旧结果改称新实现的结果。
**先锁定最终合并后的 commit，再创建新缓存、新 run；不要在运行中途 git pull。**
现有来源身份包含 Git tree、工作区差异等，文档或工作区变化也可能影响严格续跑。

## 1. 兼容性总表

| 项目 | 迁移规则 |
|---|---|
| 历史报告、日志、负结果、checkpoint | 原样保留，关联原 commit/配置；不覆盖、不删除、不重写历史结论 |
| 原始 corpus、tokenizer、预训练 backbone | 文件哈希、模型配置和数据分割验证一致时可复用；无需因为缓存升级而盲目重新下载或预训练 |
| 旧 pair/single 派生 cache | 分别重建为 pair `rl-lm-cache-v2` / single `rl-lm-single-cache-v2`，使用新目录；旧 manifest 不得仅补版本号/伪造校验信息后继续使用 |
| 新 cache | 校验七个 payload 的文件哈希、数组类型/形状、记录数、有限值；实际内容身份进入 checkpoint |
| 旧 reference_ce.npy / JSON sidecar | 在新 single-RL run 中重新计算；新 `reference-ce-artifact-v3` 绑定 NPY 文件和数组内容、类型/形状等 |
| 旧单专家适配模型 | 冻结 single teacher 的旧目标不能冒充 pair teacher；新适配目标为 `original-pair-teacher-ce-kl-relative-update-v2` |
| 旧 R2 checkpoint | 缺少新选择规则、固定 R1 参照或实际输入身份时，不能直接 exact resume |
| 已使用新契约的 checkpoint | 只有实际数据/模型/数值/动作/特征等身份匹配才能 exact resume；改变输入必须新建 run |
| 正式确认 candidate | 新建、预先固定 `confirmation_scope`；旧候选缺少范围身份时不直接沿用 |
| 新 runtime 后端 | 默认关闭；先独立验证，不能把旧训练/测量结果改称使用了新后端 |

**不要仅凭 checkpoint 的 `format_version` 字符串判断可续跑。** 某些外层格式名称
保留，但新增的 verified-cache、reference、selection 等身份字段仍然是必需契约。
重新计算文件哈希也不能使替换后的数据匹配已保存 checkpoint 的旧身份。

## 2. 开始前：备份与目录

- 记录原 commit、配置、数据/backbone/tokenizer 哈希和原 run 路径。
- 复制配置到新的文件，不要直接修改历史 resolved_config.json。
- 为本轮指定全新的 `output_dir`，例如 `runs/integrity_v2/pair_r1_seed17`。
- Pair 配置中的 `data.cache_root` 指向新的 pair-cache 目录。
- Single 配置中的 `data.pair_cache_root` 指向新 pair cache；`data.cache_root`
  指向新的派生 single cache。适配与 RLOO 使用不同的新 output_dir。
- Single RLOO 的 `adapted.adapt_run_dir`、`adapted.checkpoint_path` 及预期哈希
  必须对应新适配输出及其通过的 gate，不能混用另一模型的 gate.json。
- 不要关闭只为绕过错误的校验，不要把 CUDA 配置改成 CPU 来假装完成正式 GPU 验收。

## 3. Pair：准备、检查、训练的顺序

以下只是操作者后续执行的模板，本次修改没有在服务器上执行这些命令，也没有启动训练。
先将新配置中的 checkpoint/corpus/tokenizer/device 路径填写完整，或使用对应 CLI 参数。

```bash
# 新配置已设置新的 data.cache_root 和 output_dir。
python -m experiments.rl_lm_prepare --config PATH_TO_NEW_PAIR_CONFIG
python -m experiments.rl_lm_preflight --config PATH_TO_NEW_PAIR_CONFIG
# 只有 preflight.json 状态为 ready，且预算/任务获准，才启动训练。
python -m experiments.rl_lm_train --config PATH_TO_NEW_PAIR_R1_CONFIG
```

R2 必须预先选择兼容的 R1 checkpoint，并固定这个参照：

```bash
python -m experiments.rl_lm_train --config PATH_TO_NEW_PAIR_R2_CONFIG \
  --r1-reference-checkpoint PATH_TO_PRESELECTED_R1_CHECKPOINT
```

所有新 R2（包括 lambda=0）都在 `NLL <= 固定R1 NLL + 0.005` 的可行候选中最小化
`NLL + lambda * (mean_rounds - 1)`。没有可行候选时会报告
`no_feasible_candidate`，不能把不存在的 best 当成通过。

## 4. Single：重新建立完整依赖链

顺序是：已验证 pair cache → 新 pair-teacher adaptation → 通过质量/计时 gate →
新 single-RL run / reference CE。新适配入口会生成并验证派生 single cache。

```bash
python -m experiments.rl_lm_single_adapt --config PATH_TO_NEW_SINGLE_ADAPT_CONFIG
# 先检查新的 gate.json；未通过时不要跳过 gate 强行启动。
python -m experiments.rl_lm_single_train --config PATH_TO_NEW_SINGLE_RLOO_CONFIG
```

reference CE 缺失/格式旧/内容不一致时，按错误提示在新 run 重建；不要把任意旧
reference_ce.npy 复制到新目录。已存在 run 的恢复也不会默默重新认证缺失的旧 sidecar。

## 5. Exact resume 与 warmstart 不同

- Exact resume 必须指向同一 output_dir 内的匹配 checkpoint，并保留其数据、模型、
  参考、配置、选择规则和 RNG 状态。仅允许接口明确支持的 device 等例外。
- 先验证身份，再恢复状态；不应靠删掉 manifest 或改 checkpoint 字段来绕过拒绝。
- Warmstart 是一个**新实验**，不是延续原 optimizer/步数/历史。Pair 的
  `--warmstart-actor` 只继承 actor；必须保留父模型来源并检查兼容性，不得把任意
  同维度模型当成已证明兼容。默认迁移路线是新 run，不是自动 warmstart。
- 对需要复查的旧实验，使用其原代码和原环境，且不要覆盖新版本产物。

## 6. 正式确认：先固定范围，再看结果

新 candidate 除原有 actor/reference 哈希、price、backend、head contract、selection
rule 外，还必须包含 `confirmation_scope`。用
`experiments.rl_lm_evaluate.verified_evaluation_scope(corpus_dir, "newtest", count)`
的第一个返回值生成该字段；这个过程只校验数据，不计算 actor 指标。

范围记录包含实际 split 文件、corpus manifest、文档分组哈希，以及序列数和序列 ID。
先固定候选与范围，再执行确认；不能看过结果后缩小样本或换分组。传入的序列数
必须与固定范围一致。R1 参照如被固定，CLI 也必须提供匹配的 checkpoint。

新报告默认 `evaluate_<split>_metrics_v3.json`；旧文件已存在时会拒绝覆盖。
`--overwrite-output` 只用于操作者明确决定的替换，常规迁移应选择新输出路径。
它不会允许绕过输入身份或确认范围检查。

## 7. 新后端的启用与退出

| 路径 | 默认 | 新候选 |
|---|---|---|
| Pair 训练 reward | `canonical_all_experts_v1` | 不替换，仍为固定形状参考 |
| Pair 末层部署 | `sparse_merged_v1` | 显式选择 `sparse_active_merged_v3`，STOP 行真正不计算专家 |
| Prefix 派发 | `legacy` | profiler 使用 `--prefix-backend merged` |

先只切末层，再只切 prefix，分别比较后再考虑组合。新后端改变 GEMM 批形状，CPU
通过不等于 CUDA BF16 路径一致或 GPU 加速。优化 prefix 当前是显式运行时 profiling
选择，不写入模型权重；canonical cache 构建拒绝非 legacy prefix，不能暗中推广到训练。

不启用这些候选就继续使用默认参考。若候选验收失败，回到默认后端和新的评估输出，
保留失败记录；这不等于可把已改变配置的 run 当作原 run exact resume。

## 8. 验收清单

- [ ] 新配置通过校验；目录没有与历史记录混用。
- [ ] cache/reference/schema/实际内容哈希匹配；篡改和重哈希替换的负测试能拒绝。
- [ ] 同输入连续训练与中断续跑一致；不同输入不能 exact resume。
- [ ] CPU 全套测试通过，并单独列出 CUDA 跳过项。
- [ ] 在目标 GPU 上验证动作、STOP、隐藏状态、生产 CE、尾块/偏斜路由和梯度路径。
- [ ] 公平比较同一工作负载的 selector、executor、full-model，保留重复测量与显存口径。
- [ ] 在规定的数据角色和预算内重新验证质量；不把逻辑省算量写成实际端到端提速。

本轮实现的本地验证为 834 项测试：826 通过、8 项 CUDA 跳过。最终 GitHub CI
结果见 PR #2。尚无本轮 GPU 速度、显存实测或重新训练质量结论。
