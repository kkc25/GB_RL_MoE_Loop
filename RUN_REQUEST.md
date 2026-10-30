# 运行请求：先训练真实第二轮，再学习硬停止门

**状态：待执行。不要把CPU测试/CI通过写成GPU效果。** 使用本次交付的完整SHA，在独立worktree运行。设计、竞争解释和预设判据见 `TRAINABLE_LOOP_EXPERIMENTS.md`。本轮不实现可学习幅度门，不自动合并。

## 基线与目的

基线 `7f21de3b4a14a5f954cbeae51db03a4377ab4ec6`；最近R1 expected-CE结果 `205a567453c41abb768828920ccd12ca212c6767`。旧R1仅训练约1.044遍、动作未变；本轮检验“真正额外一次末层专家FFN是否可以通过小输入适配器得到可预测局部收益，再由硬门分配调用”。不是检验整个decoder循环，也不预设存在必须先爬出的局部最优。

分开回答：软件能否正确反传和执行STOP；第二轮能否拟合；是否优于参数匹配的一遍修正；门能否优于同成本随机分配；开发/selection收益和实际成本是否值得。失败原样保留，不自动加幅度门或延长预算。

## 不变条件和预算

- 24训练/8开发，16 selection，8 calibration；既有数据身份、去重及冻结规则保留，最终测试不访问。16篇selection已用于历史研究，不能称新最终测试。
- rank8，固定8遍覆盖、每次监督最多32个有效位置、每个分支最多2048更新。实际前向及反传使用完整padding文档（默认256token），不是32token激活。预估超过更新上限则先拒绝，不截断后冒充8遍。
- 真正R2与参数匹配无额外FFN控制各自完成同样更新预算；这不是算力匹配。只训练小适配器，主干/专家/输出头冻结但保留到输入的反传。
- 无STOP训练机会竞争，无开发集patience提前结束分支拟合。开发集从实际R2 checkpoint0..8中选分支；原模型另存为回退候选。
- 训练局部收益资格达标后才训练硬门：512步、batch256，使用第二轮前h1，不用标签/未来token/第二轮结果作推理特征。阈值固定网格及全STOP，开发集选定后不再用selection调阈值。
- fit/evaluate/profile链累计墙钟预算7200秒；按阶段凭据扣除已消耗时间。边界检查不是抢占式GPU看门狗，单个正在执行的操作可能跨过截止时间；超时必须记录为未完成并保留成本，不能报告正常完成或自动续跑。
- 序列256、训练有效位置上限8192、缓存1GiB、logit元素上限16,777,216。原始模型已在服务器，禁止重新下载大型权重；不CPU/磁盘卸载，不租新算力。
- 四张可用RTX A6000、每卡配置44GiB预算，BF16；不能假定机器始终空闲。先确认可用卡，不干扰其他任务。实际CUDA反传/显存/数值必须由本机执行验证，CPU测试不能替代。

## 环境与依赖

不改变现有依赖版本。沿用 `ENVIRONMENT.md`、`requirements-mixtral.txt` 和已有服务器环境；当前仓库记录Python3.12、PyTorch2.12.1+cu130、Transformers5.15.1。不要把别的GraphJit项目环境套到此处。

模型revision固定 `fc7ac94680e38d7348cfa806e51218e6273104b0`。配置 `local_configs/trainable_loop_pilot.json` 提供现有服务器路径；通用模板在 `configs/mixtral/`（以实际交付文件为准）。输入来自已有本地数据，不从旧100M缓存迁移，不把历史报告用作训练输入。

如本机模型/数据路径不同，只在执行前明确调整路径并保存最终配置及哈希。模型revision、数据划分、训练预算、门阈值等研究参数不临时调整。不得修改运行中的worktree或旧输出。

## 固定提交与快速检查

在现有仓库、激活正确服务器环境后执行。`CODE_SHA`替换为本次交付完整40位SHA，不能用main、基线SHA或浮动分支名。

```bash
set -euo pipefail
REPO=$(pwd)
PY=$(command -v python)
CODE_SHA=REPLACE_WITH_DELIVERED_FULL_40_HEX_SHA
git fetch origin experiment/trainable-loop-hard-gate-v1
WORKTREE="${REPO%/*}/Mixtral-trainable-loop-${CODE_SHA:0:12}"
git worktree add --detach "$WORKTREE" "$CODE_SHA"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$CODE_SHA"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY -m pip check
$PY -m unittest discover -s tests -t . -v
$PY -m compileall -q mixtral_lab/trainable_loop.py experiments/mixtral_trainable_loop.py
$PY -m experiments.mixtral_trainable_loop --help
```

若worktree已存在，核查身份，不删除重建。保存完整测试日志、退出码、runtime、设备清单及源码身份。CUDA专用测试在CPU跳过不算已验证；真实服务器上按适用条件运行。出现不支持的专家反传、OOM、非有限值、原模型parity失败即停止，保留诊断，不自动换backend或缩数据直到通过。

## 正式步骤

同一PY/worktree/CODE_SHA，使用全新独立目录。以下没有自动恢复或重训；不得复用旧输出。

```bash
set -euo pipefail
RUN_ID=trainable_loop_v1_seed17_$(date -u +%Y%m%dT%H%M%SZ)
OUT="$REPO/runs/mixtral_trainable_loop_v1/$RUN_ID"
CONFIG=local_configs/trainable_loop_pilot.json
$PY -m experiments.mixtral_trainable_loop --config "$CONFIG" --expected-commit "$CODE_SHA" --stage fit --output-dir "$OUT/fit"
```

先阅读 `fit/report.json`：确认complete、真实R2与控制的完整覆盖、梯度/冻结权重、实际成本、训练资格门、开发集门选模结果。全STOP、资格失败、partial/failed都是应回传的结果，不要求强行进入下一阶段。

仅当前序预设关卡通过，才执行：

```bash
$PY -m experiments.mixtral_trainable_loop --config "$CONFIG" --expected-commit "$CODE_SHA" --stage evaluate --output-dir "$OUT/evaluate" --fit-report "$OUT/fit/report.json"
```

只有评估质量关卡通过，才进入性能阶段：

```bash
$PY -m experiments.mixtral_trainable_loop --config "$CONFIG" --expected-commit "$CODE_SHA" --stage profile --output-dir "$OUT/profile" --fit-report "$OUT/fit/report.json" --evaluation-report "$OUT/evaluate/report.json"
```

各阶段都写 `report.json`，后续绑定上游report与内嵌checkpoint路径/hash，不允许手挑替换checkpoint。阶段采用同一冻结配置和代码SHA；保留累计墙钟凭据。不要改report让stage gate通过。

## 验收与失败恢复

- 检查STOP零额外专家调用、CONTINUE真实执行额外Top-2专家、主干无更新、适配器有效更新、无padding计分/标签泄漏。
- 保留强制全开、参数控制、固定预算、学习门与匹配随机的全部结果，不只报赢家。回顾性按文档精确匹配调用数的随机对照不是部署策略。
- 真实压缩执行是候选路径；全部计算后mask仅作数值诊断，不能把它的耗时当条件计算速度。数值差异须报告，不以忽略差异或放宽旧parity门槛消除失败。
- 开发集用于选模；selection只评估已冻结候选。达到0.001 nat/token及预设文档配对区间条件才允许相应质量结论，记录全部对照和局限；不声称最终测试成功。
- 保存每阶段模型/配置/数据/源码/checkpoint身份，所有成本预约与完成记录、峰显存、未完成操作、失败原因、日志、命令和退出码。未知硬中断成本标未知，不能算零或假装完成。
- 本轮没有自动resume。失败保留原目录；需要修复时先诊断，新的代码/参数/预算需重新冻结并另建run_id，不能删除失败后悄悄重新计预算。
- 回传执行代码SHA、结果提交SHA、完整报告和可复算证据清单；权重/大checkpoint遵守既有仓库存储规则，保留可访问的受控原始副本与哈希，不能只发截图或摘要。
- 结果上传后检查远端字节/hash与CI，草稿PR保持未合并，不强推、不删历史分支、不改旧阴性结果。正式执行完成前状态始终待执行，不写已有成绩。

## 专家后端冻结

本轮交付配置固定 `experts_implementation=eager`，不沿用旧实验的grouped_mm作为未经验证的训练后端。真实服务器仍需通过配置后端反传预检，确认适配器梯度非零且主干无梯度，再进行完整训练。所有质量/性能对照在本轮同一后端重新计算；不能直接把本轮NLL或耗时与旧grouped_mm运行当作单因素比较。切换后端属于新数值协议，需要重新冻结配置/运行身份，不在失败时静默回退。
