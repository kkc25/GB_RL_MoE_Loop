# 复现、证据等级与覆盖说明

## 1. 固定对象和环境

仓库：`kkc25/GB_RL_MoE_Loop`。审查 commit：`f2bef422f7315f30618f00c690bf82810d1e85ac`；tree：`62bb1f1e9cd48cff238ea5492338c22536bcf552`。全部 390 个受版本管理文件字节经 Git blob SHA 检验，合计 37,893,278 bytes。下载方式不同不会改变这些字节；审查中的仓库文件没有被补丁覆盖。

| 环境 | Python | PyTorch | 设备 | 用途 |
|---|---|---|---|---|
| 上传运行记录 | 3.12.15 | 2.12.1+cu130 | 用户 RTX A6000 | 已上传训练与评测证据 |
| 本轮审查 | 3.12.14 | 2.5.1+cpu | CPU | 原测试、微型真实代码反例、公式/记录复算 |

本轮另使用 pytest 8.3.5、tokenizers 0.21.4。NumPy 的实际版本记录在 `verification_summary.json`。版本与设备差异不妨碍源码中求和轴、数据覆盖、是否调用 CE 等确定事实；真实 GPU 数值与性能仍需在用户环境复测。

## 2. 目录约定

多数审计脚本默认在自身目录的上一级查找 `GB_RL_MoE_Loop`，因此将压缩包解到仓库的同级目录：

```text
<replay_root>/GB_RL_MoE_Loop
<replay_root>/code_audit_2026-10-25
```

若现有工作树已经修改，使用独立 worktree 或 clone 检出固定提交；不要为了复现把正在使用的工作树强制重置。以下命令是可选的本地建立方式，`/path/to/...` 需替换成实际目录：

```bash
git -C /path/to/existing/GB_RL_MoE_Loop worktree add --detach /path/to/replay_root/GB_RL_MoE_Loop f2bef422f7315f30618f00c690bf82810d1e85ac
unzip GB_RL_MoE_Loop_code_audit_2026-10-25.zip -d /path/to/replay_root
```

仅使用 NumPy 的脚本有些支持 `--repo`，命令表中有说明。运行会在解压后的审计目录生成或覆盖对应结果 JSON/日志；如果希望逐字节保留交付证据，另复制一份审计目录再跑。不会改动生产源码。

现有训练环境可以用于复现。若另建环境，需按下表安装相应依赖；本审计的 PyTorch 2.5.1+cpu 是已验证版本，不意味着必须替换用户的 CUDA 环境。完全不需要下载 LM 权重即可执行本节的所有反例。

## 3. 纯记录与数学复算

从 `code_audit_2026-10-25` 目录执行：

| 命令 | 依赖 | 主要结果 |
|---|---|---|
| `python records_only_counterfactual_audit.py --repo ../GB_RL_MoE_Loop` | NumPy | 三套 E01 的文档覆盖、split 交集、Q 方差分解、train-only 常量/分组策略与逐文档结果 |
| `python recheck_e10_records.py --repo ../GB_RL_MoE_Loop` | NumPy | E10 32/16/16 的单文档覆盖、六位小数表的严格符号翻转敏感性 |
| `python reproduce_synthetic_audit.py --repo ../GB_RL_MoE_Loop --output synthetic_audit_checks.json` | NumPy | E06 soft 边界、E07 gap/顺序反例、prefix knapsack 与小 DP 对照 |
| `python recheck_e07_e09.py --repo ../GB_RL_MoE_Loop --output e07_e09_recheck.json` | NumPy | 原 E07 64 状态窗口完整 oracle；当前 NumPy E09 梯度/概率与显式坐标有限差分 |
| `python repro_counterfactual_numpy.py` | NumPy | 特征变换、einsum、零翻转、动作编号的解析小反例 |
| `python reproduce_model_training_arithmetic.py` | NumPy | 参数/FFN FLOPs/调用整数核对与实际 sampler 的 12 步 CPU 比较 |
| `python probe_legacy_report.py --repo ../GB_RL_MoE_Loop` | 标准库 | 原旧报告函数对缺数学检查的空目录误报 math pass |

`reproduce_synthetic_audit.py` 与 `recheck_e07_e09.py` 对依赖 Torch 导入的文件使用原函数 AST 提取，函数体没有改写；报告将这种调用与独立 DP、独立有限差分明确区分。它验证的是列明的函数，不冒充运行整个 GPU 训练 CLI。

`repro_counterfactual_numpy.py` 是独立表达式反例；对应关键错误另有真实 PyTorch 原函数复现，不能单靠复制一个错误公式来声称执行了原系统。

E09 旧简版 `e09_gradient_recheck.json` 没记录 6 个有限差分坐标，误差为约 5.39e−12；新脚本记录固定坐标，误差约 7.79e−12。两者均通过 `<1e−7` 门槛，不能声称这些误差尾数逐位相同。RLOO/exact 的约 5.13e−16 及概率质量一致。

## 4. 原代码 PyTorch 复现

在同样目录、可导入 Torch/NumPy 的 Python 环境执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python repro_counterfactual_torch.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python reproduce_model_training_torch.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python reproduce_legacy_exact.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python reproduce_causality_supplement.py
```

每个脚本内部将 sibling 仓库加入模块搜索路径。含下列有区分力的检查：

| 检查 | 原函数如何被调用 | 结果的解释边界 |
|---|---|---|
| 停止 raw/normalized h | 真正 `DynamicHaltingPolicy`，小型随机模型、非零学生、有限 tau | 证明执行契约不一致；不是用户训练权重的 NLL 估计 |
| E12 独立性、摘要、`.to` | 真正 `ExpertBidAdapters` | 证明求和轴和 Module API 问题；区分 learned 与潜在 checkpoint-summary 模式 |
| 计时 CE 工作量 | 原 `profile`，包装 CE 计数 | 固定路径 3 次、dynamic 0 次；不是 CPU 与 A6000 时延比较 |
| checkpoint 账本与 best | 原 `train_lm`，注入确定时钟与验证分数 | 7+5 应累计 12；0.9 不应被 2.0 覆盖；注入数不是性能结果 |
| 旧 exact 梯度 | 原 `train_exact`，进程内替换成两叶解析枚举器 | 隔离生产 loss；缺概率权重产生错误方向 |
| 旧 budget 截断枚举 | 原神经模型与 oracle，短预算 | 终止路径重复计数，质量和可超过 1 |
| 因果性工具 | 原聚合器以及无 memory monkeypatch 的生产路径 | 聚合入口报错；生产 FP64/批重排检查通过，排除过时注释误报 |

Recovery evaluate 入口另已实际调用原 CLI，传不存在的 checkpoint 仍以 0 退出；当时完整输出保存在 `recovery_evaluate_probe/`。复现方式是对独立临时输出目录运行原 `run_recovery.py --mode evaluate --device cpu --config configs/recovery_c0_identity_dense.json --resume <不存在路径> --output-dir <临时目录>`。切勿以已有训练输出目录接收这次故意的随机评估，以免覆盖配置或 summary。

## 5. 原测试与最小补丁回归

原测试执行环境设置为 CPU，实际命令等价于从仓库目录执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests
```

结果为 **424 passed in 83.15s**，没有将用户报告中的 skip 数照抄为本轮结果。另附三组早期分工测试日志，其测试已包含在完整 424 中，不能相加成更多独立测试。

从审计目录执行：

```bash
python validate_minimal_patches.py
python validate_minimal_patches.py --patched
```

第一条预期观察到 2 failures+1 error，脚本在成功复现这些缺陷时正常返回；第二条预期 3 tests pass。它们在内存加载模块，不写生产源码。若用户已经修过源码，应换回固定快照复现，或直接在修复分支运行补丁附带的回归测试，不能把“原缺陷不再复现”误认为补丁失败。

补丁可应用性只需在匹配提交的工作树运行：

```bash
git apply --check ../code_audit_2026-10-25/patches/minimal_correctness_fixes_with_tests.patch
```

本轮该检查已通过。实际应用属于后续实施；这里交付的是可审阅差分，不代表所有发现已修复或远端仓库已更新。

## 6. 审查覆盖与不能从中推出的事

| 区域 | 覆盖内容 | 主要文件/附录 |
|---|---|---|
| 模型/策略 | causal attention、loop、稀疏 dispatch、Top-2、GRU/采样、预算与强制退出 | `model/`、`policy/`，模型/合成附录 |
| LM 与 recovery | 配置、训练、标签、优化、恢复、评估、数据 sampler | 四个 run 入口、`train/`、配置和模型附录 |
| 数据 | affine/recovery/controlled、FineWeb 分区与 tokenizer、packing/边界、manifest | `data/`、总报告 F02/F03 与覆盖结果 |
| 反事实与学生 | 缓存、候选、固定续算、学生训练、停止部署、E10/E12 | 反事实附录 |
| 合成与 RL | 势博弈/流、调度/DP、稳定性、当前受控 RL 与旧神经路径 | 合成附录 |
| 测量与汇总 | profile、资源计数、恢复指标、registry/报告生成 | 总报告 F07–F11/F17 |
| 测试与记录 | 全 22 个测试文件、全部公开 run 记录与代码/配置对照 | 424 测试日志、记录复算 JSON |

82 个非 plans Python 文件包括 60 个生产/初始化文件和 22 个测试文件；plans 中另外 5 个 Python 是先前方案的预算/验证脚本，按设计材料核对，不算本次 LM 训练实现。具体文件见 `source_inventory.json`。

不存在“静态检查已证明所有代码没有其他 bug”的承诺。审计的完整性指覆盖所提供提交、对关键科学主张和执行链做系统交叉检查，并将未证实内容显式列出；没有大权重时不能判断真实专家塌缩、误差归因比例或修复后的最终收益。

所有结果文件中的 CPU 微秒/毫秒、采样器用时均是审查主机观测。`proposed_label_budget.json` 中 future 行是调用量规划，**不是已运行实验**。E10 量化表重算不是原 FP32 成本矩阵的精确替代；E07 插值 oracle 不是实际 T2/T3 LM 结果。主报告与三个附录的表述均保留这些限制。
