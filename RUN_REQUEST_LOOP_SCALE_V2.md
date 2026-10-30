# 运行请求：Mixtral 数据规模与固定幅度实验 v2

**状态：待服务器执行；本次任务仅完成开发、CPU 测试和 GitHub 上传。** 不把 CI、微型模型测试或上轮离线复算写成新 GPU 效果。完整研究规则见 `LOOP_SCALE_EXPERIMENTS.md`，迁移说明见 `LOOP_SCALE_MIGRATION.md`。旧 `RUN_REQUEST.md` 原样保留。

## 执行身份与范围

使用本次交付的完整 40 位代码 SHA，建立新的独立 worktree。基础 main 为 `ac80b445376b05e2a974226bbea86fcc8daaecaa`，必要代码依赖 PR #14 头 `3559a307cb8af347444b09444044bfac31084cc0`；新分支 `experiment/mixtral-loop-data-scale-v2-20261030` 已同时保留这两条历史。不要修改旧 worktree、旧结果、main 或 #14，不强推、不删分支、不自动合并。

本请求针对现有四张 A6000 上的 Mixtral-8x7B-v0.1 BF16 resident 多卡环境。模型路径 `/nvme-1/zhoukegu/models/Mixtral-8x7B-v0.1`，revision `fc7ac94680e38d7348cfa806e51218e6273104b0`；不重新下载，不 CPU/磁盘卸载，不改成旧 100M 模型。使用现有 `ENVIRONMENT.md` 和 `requirements-mixtral.txt`，无权重下载或环境自动迁移。

## 准备输入与预算

配置模板为 `configs/mixtral/loop_scale_v2.json` 和 `local_configs/loop_scale_v2.json`。输入数据须由操作者显式提供现有本地 JSONL 路径；每行具有非空字符串 `id`、`source_id`、`text`。仓库未附带百万位置的新语料，代码不会下载或复制小数据补满。

把实际路径写入外置冻结配置；不要在运行中修改该文件。`source_files`、模型/tokenizer 路径、独立 `freeze_dir` 和 `run_root` 必须指向真实位置，历史 report 路径必须指向上轮完整报告。准备命令会按实际文件内容绑定权重、tokenizer、语料、历史排除和清单哈希。每份清单至少 30 个独立来源组；输入不足、来源不明或去重后短缺均应失败，不能报告“已准备”。

训练总预算为 100k/300k/1m 个去重有效预测位置，内部 90% adapter-train、10% 独立 gate-fit，二者分别嵌套；开发为来源隔离的 50k 校准和 50k 选择；最终测试 100k 封存。文档边界允许每份清单至多 254 个位置超额，按实际计数。准备阶段排除全部既有小试验文档身份/内容，尤其历史 8 篇开发文档；不能把旧 selection 称新测试。

默认硬预算：每 cell 核心训练/评估 7,200 秒、缓存阶段 7,200 秒、预检 300 秒、整个 run_root 86,400 秒；cell 核心 ledger、缓存与 suite 从各自不可变首次启动时间计时；fit 命令之前的模型加载和缓存身份扫描仍完整计入 stage receipt/suite，保守包括暂停时间；预检 300 秒为模型加载后的核心检查上限，加载/哈希另计入 suite 及阶段 receipt。这些是上限，不是耗时保证。先用四篇文档做吞吐/显存/梯度预检，再据实判断能否完成冻结阶段；预算不足即停止并回传，不自动延长或缩数据。单 cell 的更新上限及缓存字节上限见冻结配置和报告。

## 独立 checkout 与 CPU 快速检查

以下命令由服务器操作者执行；本次开发任务没有执行服务器命令。`CODE_SHA` 必须替换为最终交付的完整 SHA；不要使用 main、旧代码 SHA 或浮动分支代替。

```bash
set -euo pipefail
REPO=$(pwd)
PY=$(command -v python)
CODE_SHA=REPLACE_WITH_DELIVERED_FULL_40_HEX_SHA
git fetch origin experiment/mixtral-loop-data-scale-v2-20261030
WORKTREE="${REPO%/*}/Mixtral-loop-scale-v2-${CODE_SHA:0:12}"
git worktree add --detach "$WORKTREE" "$CODE_SHA"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$CODE_SHA"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY -m pip check
$PY -m unittest discover -s tests -t . -v
$PY -m compileall -q mixtral_lab experiments tests
$PY -m experiments.mixtral_loop_scale --help
```

已存在 worktree 时先核查身份，不删除重建。保存命令、完整日志、退出码、runtime 和设备清单。CPU 条件跳过的 CUDA/多 GPU 测试不算通过了真实 GPU 验证。

## 外置配置与正式冻结

为本轮创建全新输出父目录，将模板复制为该目录中的配置，再填写实际 `source_files` 等路径。`RUNROOT`、`FROZEN` 和 `CONFIG` 应位于独立 worktree 之外，不能覆盖历史路径；下面是命名示例，不表示这些目录已有数据。

```bash
RUN_ID=loop_scale_v2_seed17_$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="${REPO%/*}/mixtral-loop-scale-v2-runs/$RUN_ID"
FROZEN="${RUNROOT}_frozen"
CONFIG="${RUNROOT}_config.json"
mkdir -p "$(dirname "$CONFIG")"
cp local_configs/loop_scale_v2.json "$CONFIG"
```

复制后，先在外置配置中填写真实输入，并把 `run_root`、`freeze_dir` 分别设为上述 `RUNROOT`、`FROZEN`。不能只设置 shell 变量却漏改 JSON。研究参数、角色配额、幅度、阈值和种子保持冻结值；准备成功后不得再编辑配置。

```bash
$PY -m experiments.mixtral_loop_scale --config "$CONFIG" --stage validate
$PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage prepare
$PY -m experiments.mixtral_loop_scale --config "$CONFIG" --stage validate
$PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage preflight
```

第一条 `validate` 通过只说明配置合法；只有实际 `prepare` 成功并再次校验全部非测试清单后，才称数据已冻结。核对 `freeze.json`、`preparation.json`、历史排除、实际来源/文档/位置数量、权重字节哈希及 preflight 结论。清单校验不读取最终测试用于调参，默认训练/开发入口拒绝加载测试。

若预检发现 native parity、冻结专家反传、适配器有效更新、BF16 或紧凑执行不通过，停止。不要切 backend、增大容差或下载替代权重后继续同一运行。

## 先执行 100k 三个 cell

`fit` 完整执行一个 cell：固定两遍分支学习，epoch0/1/2 评估；R2 的 epoch1/2 各自在独立 gate-fit 学门、校准阈值，再在选择集选条件候选。无需额外调用旧 v1 的 fit/evaluate/profile。

```bash
$PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage cache
for CELL in parameter_control r2_full r2_reduced; do
  $PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage fit --budget 100000 --cell "$CELL" --seed 17
done
$PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage report --budget 100000
```

`cache` 一次生成最大冻结训练池（900k adapter、100k gate-fit）和开发两半的 prefix，并绑定模型与运行环境；不捕获测试。先核对这个前置成本，不能称为只缓存 100k。300k/1m 复用同一缓存，不重复执行 cache。

审阅三个完整 cell 与全部候选，不只看赢家。对 forced-all 和 gated outcome 使用各自预定选择规则；全 STOP、负收益、失败、unknown 都是需回传结果。门只用 pre-R2 `h1`；其标签来源不能回流 adapter-train。

100k 三个 cell 完成后，预先授权继续运行一次 300k 三 cell 的有限对比，不要求 100k 已优于 native。这是为了检验增加独立数据能否修复负收益；数值、身份或硬预算失败仍停止。不会无条件自动运行 1m。

## 有条件进入 300k 与 1m

下面的命令不是自动续跑脚本。只有读取上一阶段报告并核对推进条件后，才单独执行下一段；CLI 也检查阶段凭据。

```bash
for CELL in parameter_control r2_full r2_reduced; do
  $PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage fit --budget 300000 --cell "$CELL" --seed 17
done
$PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage report --budget 300000
```

300k 完成后，只在同一 R2 幅度、同一结果类型（forced/gated）内比较 300k 与 100k。至少一项选择集 NLL 改善 ≥0.001 nat/有效位置，且来源簇配对 95% 区间下界 >0，才允许进入 1m；此时允许仍差于 native，因为这里检验数据改善趋势。没有这种趋势则暂停扩容，保留阴性结果诊断。开发数据参与这些决策，区间属探索性证据；三种子条件确认仍单独要求对 native 正收益。

```bash
for CELL in parameter_control r2_full r2_reduced; do
  $PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage fit --budget 1000000 --cell "$CELL" --seed 17
done
$PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage report --budget 1000000
```

1m 若不改善，就停止，不自动进入 300万或500万位置。两种幅度都从头训练，不能把未训练小幅度诊断冒充已训练幅度对照。

## 等曝光对照与三种子确认

若进入 1m，必须把 100k 小池重复到与实际 1m adapter-train 两遍相同的有效位置曝光和更新数；每个候选门也重复小 gate-fit 池至相同曝光/更新数，用于区分数据与额外计算。`--matched-exposure` 只对 100k 档开放，目标从冻结 1m 实际清单计数获得，不能手工估算整数倍。

```bash
for CELL in parameter_control r2_full r2_reduced; do
  $PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage fit --budget 100000 --cell "$CELL" --seed 17 --matched-exposure
done
```

在已完成的最大获准档位，按条件 NLL 固定唯一幅度 cell（强制赢家另报）。只有条件赢家相对 native 改善 ≥0.001 nat、文档区间下界 >0 才允许三种子确认。预先固定唯一有希望的 cell，再做 seed29、43，与 seed17 构成三个种子。下例的 `SELECTED_BUDGET`、`SELECTED_CELL` 必须来自冻结选择凭据，而非在各 seed 中再挑赢家；最终原样保留失败种子。

```bash
SELECTED_BUDGET=REPLACE_WITH_FROZEN_SELECTED_BUDGET
SELECTED_CELL=REPLACE_WITH_FROZEN_SELECTED_CELL
for SEED in 29 43; do
  $PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage fit --budget "$SELECTED_BUDGET" --cell "$SELECTED_CELL" --seed "$SEED"
  $PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage fit --budget "$SELECTED_BUDGET" --cell parameter_control --seed "$SEED"
done
$PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage report --budget "$SELECTED_BUDGET"
```

## 冻结最终候选与一次测试

finalize 还要求 100k 和 300k 三 cell 的有限主对比已完整完成，与是否优于 native 无关。先核对冻结配置、全 cohort 完整性、参数控制、必要的等曝光对照、三种子确认及全部 checkpoint 哈希，再生成不可变 finalization lock。最终测试只接受该 lock，不接受临时指定阈值或 checkpoint。

```bash
$PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage finalize --budget "$SELECTED_BUDGET" --cell "$SELECTED_CELL" --seed 17
```

`finalize` 输出 `finalization_lock` 路径，固定文件为 `$RUNROOT/final_selection.json`；不能手改 lock。测试不是新一轮选模，测试完成后不能回看结果继续挑幅度、epoch、seed 或阈值并仍称同一次确认。

```bash
LOCK="$RUNROOT/final_selection.json"
$PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage final-test --lock "$LOCK"
$PY -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage report --budget "$SELECTED_BUDGET"
```

## 恢复、验收与回传

恢复先检查 `stage_receipts.json`、cell 日志与检查点，确认上次提交的游标、优化器、RNG、已完成阶段、成本和所有身份。只对相同配置/代码/数据/模型的未完成命令追加 `--resume`；完成命令不重训。已完成最终测试再次 `--resume` 只核验并返回既有证据，默认重复执行被拒绝；锁定后同一 run_root 禁止新增拟合。unknown attempt 不按零成本处理，累计墙钟不重置。改变任何研究参数或代码须新 run_id，不能覆写失败目录。

验收至少包含：STOP 零额外专家；CONTINUE 真实 Top-2 计算；无 padding 计分；冻结主干无变化；梯度经过专家进入适配器；单篇惰性缓存与有界分块；epoch 固定评估；独立 gate-fit；校准与选择分离；最终测试封存/lock；完整尝试、完成、失败、未知和成本记录。匹配参数、曝光、调用数和真实 FLOPs 是不同结论，不能混写。

回传完整代码 SHA、运行配置/清单/checkpoint/证据哈希、完整逐文档及可复算逐位置数据、命令/日志/退出码、每卡显存、实际吞吐与所有失败限制。大模型、源语料、隐藏缓存和 `.pt` 不进 Git，保留可访问的受控原始副本和哈希。代码与结果分开提交；结果上传后重新核对远端精确 SHA、逐文件字节及 CI 终态。草稿 PR 保持未合并。
