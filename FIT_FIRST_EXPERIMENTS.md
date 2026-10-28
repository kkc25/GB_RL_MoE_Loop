# Mixtral：先拟合训练问题，再检验泛化，最后评估成本

本协议新增独立入口 `experiments.mixtral_fit_experiment`，不改变旧
`experiments.mixtral_experiment` 的默认行为。它把三个问题按顺序拆开：

1. **Fit**：给方法可观察、可诊断的训练机会，证明它能改善固定训练问题。
2. **Generalize**：冻结训练期选出的同一个 checkpoint，只在 selection 开发文档
   上检查配对质量变化；本阶段不得再更新参数或挑选训练 checkpoint。
3. **Profile**：只有前两关通过，才测这个冻结 checkpoint 的时间与成本。

本次代码交付、tiny CPU 测试与配置校验不构成真实 Mixtral 实验结果。
没有自动启动 GPU、下载权重或读取最终 held-out/newtest。下述命令是供操作者
逐条执行的手册，不是训练、评估、计时自动串联脚本。

## 1. 为什么需要新的拟合关

此前 4 步适配器 smoke、很小的训练覆盖、强原路由先验及不同 mini-batch 上的
几个 loss 点，只能说明流程是否运行，不能说明已充分优化，更不能作为否定方法的
证据。此前固定主干的强制第二轮 FFN 恶化，也不能替代可训练适配器的容量检验。

新的起点是一个有界、能反复观察的训练问题：先 64 个固定 context、最多256次更新，
再在诊断支持时扩展到 256 个 context。**这不是“256 步必然足够”的声明。**
是否已拟合由固定探针、完整训练集指标、覆盖、数值有效性与学习曲线共同判断。
smoke 仍可先运行，但不能被当成阴性研究结果，也不降低质量门槛来制造通过。

若 plateau 或 best 仍在 update 0，先区分：实际访问覆盖不足、特征不可预测、
先验 margin 太强、梯度/优化问题、BF16 修正被量化吞没、动作 CE 几乎相同、
或模型容量不足。有限的学习率/尺度/预算诊断应预先声明并保留全部失败记录；
不能仅凭一个设置没通过就自动停止整条研究路线，也不能无界追加算力。

## 2. 冻结骨干的 R1 全动作拟合

新 R1 训练不依赖有放回的少量动作采样来碰运气。在 **train** 上为每个固定
context 枚举全部 28 个无序专家对，构建动作 CE 表。训练目标为
`sum_a pi(a|x) * CE(x,a)`：只有路由参数可训练，候选 CE 与主干均冻结。
这是完整动作的期望 CE 优化，不是原 RLOO，也不是声称部署时知道正确答案。
它是独立的容量/可学习性基线，**不是自适应探索机制的单因素因果检验**：目标、
先验、动作评分覆盖和优化器设置都改变，若有收益不能只归因于其中一项。
本版本提供有界、train-only 的学习率/plateau预算控制（见第3.1节），
但未实现自动 epsilon/G 控制器，也不自动调所有参数；此训练拟合基线
没有旧 RLOO 的单步 KL hard stop，保留数值检查、梯度裁剪与有界预算。
开发泛化安全门槛是另一个阶段，不能将两种约束混称。

- 策略仅使用现有合法 hidden/router 特征。训练时目标 token 只用于构造 train
  CE 监督；generalize/profile 的动作选择不能读取 target、CE 表或 oracle 标签。
- `epsilon=0.8` 是显式较弱的原路由先验，不是 80% 贪心部署随机率。初始化时
  原生对仍为 greedy；原生对概率约 0.2286，其余每对约 0.02857。它是待检验的
  研究默认值，不是已经选出的最优探索参数。
- 每个 context 使用固定 singleton 评分形状：28 个候选及 2 次 native anchor，
  共 **30 次 score call / 30 个 score row**。64/256 个 context 的表构建上限为
  1,920/7,680 次调用；后续读缓存优化不重新支付这些 head call。
- 表、context 身份、输入 tensor、来源与数值契约均绑定。单行 BF16 分数是
  这个形状下的训练目标，不能当成 full-document 在线 CE，也不能跨后端或批形状
  复用以声称等价。重复 anchor 只证明该评分协议内部的检查通过。
- 每次更新都报告完整有限训练集的 greedy/expected CE。best 按训练 greedy CE
  优先、expected CE 次优选择；初始化参与，持平不应被写成学习收益。
- 固定 seed 的训练 probe 子集保留 before/best/latest 诊断；每步曲线来自完整
  训练集。probe 不替代完整训练指标，也不是一个独立泛化集。训练 gate 要求
  达到配置的实际 greedy CE 改善并确实改变动作；
  仅 expected CE 降低不足以证明部署的 greedy 策略已经改善。

读表训练的优化步很便宜，不应把其迭代速度和每步重新算动作的 RLOO 直接比较。
新表也不同于旧 `oracle` 阶段：旧 calibration R1/R2+STOP oracle 保持原样；
这里是 train-only、R1-only 的监督训练输入，不提供第二轮标签或 STOP 策略。

## 3. 适配器：固定 train probe 与实际 BF16 有效修正

新适配器路线仍为 native R1 后的低秩残差修正，默认 rank=8、
`residual_scale=1.0`、64 个固定 train context、最多256次更新。
原生一次 FFN 得到的 h1 被缓存；只训练新增两个低秩矩阵，经冻结 norm/head
反传训练 CE。它**不再次调用原专家、不训练 STOP，也不是已经成立的循环机制**。

- 用固定、完整的 train context probe 选择 best；update 0 参与，持平保留最早。
  `eval_interval=16`，最后一步也要检查，不能比较来自不同随机 batch 的零散 loss
  就声称收敛。probe 的分块大小及尾块 padding 属于固定数值口径。
- 记录真实访问的 context ID/索引、访问数、distinct 覆盖、梯度、参数变化及曲线。
  训练 gate 要求数值运行完成、所有训练 context 被实际访问，并达到训练 probe
  改善门槛。缓存了 64 个 context 不等于训练实际使用了 64 个。
- 同时看 FP32 提议修正和**转换并相加后**的有效变化：原生 dtype 有效改变率/RMS，
  以及 BF16 cast + BF16 残差相加后的有效改变率/RMS。真实 BF16 主干时后者是
  实际执行诊断；tiny FP32 模型时只是 BF16 反事实诊断，不冒充 GPU 证据。
- 非零梯度或非零 FP32 correction norm 不保证 BF16 输出发生改变。若修正无效，
  先检查尺度、学习率与数值协议；不要直接推断适配容量无用。

即使适配器先拟合、再在开发集改善，也只能支持“新增单次修正容量有用”。
要证明 **loop 本身**有收益，还需要参数量/实际计算匹配的单次适配对照，
再与真正重复专家并学习轮次/STOP 的模型比较。本版本不替代这个因果实验。

### 3.1 有界自适应：只根据训练证据调学习率与停止预算

非smoke模板启用共同的确定性控制器，smoke保持关闭。它是一组可审计的
train-only反馈规则，不是自动寻找每个阶段/模块“最合适参数”的全局搜索器，
也不保证比固定设置更快、更省钱或取得更好的解。

| 模块/阶段 | 控制信号 | 允许自动调整 | 保持固定 |
|---|---|---|---|
| R1 fit | 固定完整train表上的expected CE，每16步观察 | 有界学习率；满足最低观察机会后可在plateau停止 | 全28动作、合法特征、epsilon、batch、评分形状、质量门槛 |
| Adapter fit | 固定完整train probe CE，每16步观察；实际BF16有效修正与梯度 | 有界学习率；持续无效修正时有限加大学习率；满足条件后可在plateau停止 | rank、residual_scale、probe分块、特征/目标与质量门槛 |
| Generalize | 只评估已冻结的train-best | 不允许调参、更新或重选checkpoint | 同一候选、开发文档、配对质量门槛 |
| Profile | 只计时质量通过的同一候选 | 不反馈调参，不搜索后端或计时设置 | 同一候选、精度、设备映射、工作负载与预声明重复数 |

- 初始学习率由各模块预先给定：router为0.01，界限`[1e-5, 0.02]`；
  adapter为0.001，界限`[1e-5, 0.01]`。连续3次固定训练观察没有达到声明的
  最小改善`min_delta=1e-4`，触发学习率乘以`lr_factor=0.5`。不能用不同mini-batch loss或不变的greedy动作
  代替稳定的训练目标来判断plateau。
- Adapter的加大学习率只用于可解释的数值反馈：非零有效梯度存在，但实际
  BF16 cast与残差相加后没有有效修正，且该情况持续、train probe未恶化。
  梯度证据累计自上次固定probe之后已接受的更新；当前更新通过数值检查后也
  纳入判断。不能只看最后一个mini-batch，也不能沿用上一观察区间的证据。
  每次probe观察后清空区间，失败或被拒绝的更新不计入有效梯度证据。
  只有实际native BF16执行且完整probe有效改变率为0才可触发；tiny FP32模型
  计算出的BF16反事实零改变不能触发增长。
  每次增长系数`growth_factor=2`、最多`max_growths=2`次，并受学习率上界限制；
  FP32 correction非零不是BF16有效性的充分证据。
  它不是自动调整rank/scale，不是BF16精度修复的保证，也不能借开发NLL反馈。
- Plateau提前停止必须同时达到至少64次更新、完整训练访问覆盖、学习率下界
  以及持续plateau；最低64步不是自动成功或自动停止点。无论是否触发提前停止，
  `updates=256`仍是硬上限，不自动扩大context数、调用预算或GPU作业。
  当前默认降幅/间隔下，256步内可能尚未降到学习率下界，因此不会保证发生
  plateau提前停止；训练仍可运行到硬上限。不能为强行省步数而过早结束拟合。
- 调整与停止理由、训练观察值、学习率、计数器与决策历史进入训练产物和冻结
  checkpoint。到generalize/profile时不能继续控制器，也不能用其结果重新选择
  训练checkpoint。best选择与train-fit质量gate仍按前文原规则，不被控制器放宽。
  最后一步学习率、下一步计划学习率与生成best权重那一步的学习率不是同一概念，
  必须按相应checkpoint/日志读取，不能把最终控制器状态写成best的训练设置。

固定与自适应模式的比较仍需相同实际评分/时间预算和多个seed。控制器的边界、
耐心次数和阈值本身也是预声明超参数；不能将本轮启发式称作全面自调参，
更不能从一个受限搜索结果推断全局最优或排除其他设置。

配置位于`fit.training.adaptive`：非smoke显式`enabled=true`，
`eval_interval=16`、`patience=3`、`stop_patience=3`；adapter启用时共享的
`eval_interval`必须与外层`fit.training.eval_interval`相同，不能暗中用两套
probe频率。smoke显式`adaptive=null`关闭；设置`enabled=false`也保留固定行为。
Runner会将缺省、`null`、空对象和默认关闭形式规范化为完整的关闭配置后绑定，
因此原始模板中的`null`在解析后报告中表现为`enabled=false`的配置对象。

## 4. 数据角色、冻结身份与质量关

沿用本地、显式、互不重叠的 `train`、`selection`、`calibration` 三个开发角色。
预检可能读取、分词、哈希这三者并核验重复；“train-only”指损失监督与训练期
best 选择只用 train，不是宣称从未打开其他开发文件。
最终 held-out/newtest 不在此入口支持范围内，不能改名后拿来调参。
三个阶段都先在 calibration 上检查 native/controlled 一致性；fit 不执行
selection NLL。通过 calibration 检查也不代表所有 selection 文档都必然一致，
因此 generalize 还保留逐文档的 native/controlled NLL screen。

文档由显式有界清单给出，当前 runner 取每角色清单前 N 篇，再在入选文档中
按 seed 文档平衡/位置随机抽 context。不要将其称作对全语料随机抽文档。
精确文档/有效 token 去重与同角色重复统计，不等于语义独立性证明。

| 关口 | 决策依据 | 不允许的解释 |
|---|---|---|
| Fit | train-only best、覆盖/数值诊断、独立配置的训练改善门槛 | 拟合成功等于泛化成功；smoke 未过等于方法无效 |
| Generalize | 同一冻结 best，在相同 selection 文档上与 native、controlled native 配对比较 | selection 等于未触碰的最终测试；在该阶段挑另一个 checkpoint |
| Profile | fit 与开发质量门槛均通过，同一 checkpoint 与契约 | 未达质量标准先找速度亮点；换 checkpoint 后沿用旧门槛证明 |

模板将 `fit.training.fit_gain_threshold=0.01`，将
`fit.generalization_min_gain=0.001` nat/token 分别显式设置。后者是操作者在看结果
前声明的**允许进入性能测量的开发筛选阈值**，不是统计显著性门槛，也不是部署
验收标准。它不从 `oracle.practical_gain` 借用或自动继承；训练 CE、oracle 上限
和完整文档开发 NLL 不是相同口径。所有初始值只是研究默认值，并无最优性证明。
改变阈值或其他训练/数值设置，应在新的独立 run 中预先固定并保留原结果。

Generalize 必须报告每篇文档的配对 NLL/gain，而不只报告 token 总均值；
全体 token 加权 NLL与文档均值不同，应分开读取。少数文档的很多 token 不等于
很多独立样本。报告描述性配对结果，不自动产生“正式泛化显著”的结论。
即便均值超过筛选阈值，也要检查文档覆盖、离群恶化和 seed 稳定性。

Fit 后冻结 checkpoint 与 manifest。Generalize 不更新它；profile 要求提供
同一 fit 报告、同一 checkpoint 与相匹配的 generalize 报告。新模型/数据/源码/
数值身份不匹配时失败，不允许“改 binding”或用旧 selection-best 冒充。
每个阶段使用全新的输出目录；新拟合 checkpoint 格式与旧 router/adapter
checkpoint 不互通，也不提供 exact resume。

## 5. 预设配置：smoke 与拟合机会分开

| 配置（`configs/mixtral/`） | 方法 | Train contexts / 最大updates | train / selection / calibration 文档数 | 长度 |
|---|---|---:|---:|---:|
| `fit_router_smoke.json` | 全 28 动作 R1 | 8 / 4 | 2 / 2 / 2 | 128 |
| `fit_adapter_smoke.json` | rank8 单次残差 | 8 / 4 | 2 / 2 / 2 | 128 |
| `fit_router_64.json` | 全 28 动作 R1 | 64 / 256 | 8 / 16 / 8 | 256 |
| `fit_adapter_64.json` | rank8 单次残差 | 64 / 256 | 8 / 16 / 8 | 256 |
| `fit_router_256.json` | 全 28 动作 R1，扩大覆盖 | 256 / 256 | 32 / 32 / 8 | 256 |

模板都采用 seed 17、batch 8、eager BF16，并保留旧 schema 所需的 `training`、
`oracle`、`second_round` 配置。**本入口实际训练只读取 `fit.training`**；旧
`training` 不是额外执行一次 RLOO，`second_round.enabled=false` 不会禁用新的
`fit.kind=adapter`。不要将这些新模板交给旧入口来解释结果。

非 smoke 模板预先固定 `warmup=3`、`repeats=3`、`measurements=32`：三个计时
variant 的执行顺序轮换三次，每个variant记录96个正式测量；每轮32次覆盖本模板
全部16/32篇selection文档。smoke保留1次预热、1轮×3次，仅检查计时流程，
不能用其覆盖与噪声水平得出速度结论。这些性能参数在fit前就要固定，不能等到
profile再修改；实际GPU耗时以实测计费，不承诺完成时长或成本。

64→256 是扩大问题覆盖，不是同一个 run 的续训，也不是自动执行的下一步。
扩大后单个 context 的平均更新访问次数可能下降，必须重新观察覆盖和曲线。
learning rate、先验、rank/scale、更新数与 probe 设置应分别记录，失败时做有界
诊断，不把某个预设当成充分训练的通用保证。
非smoke启用的自适应只在预定预算内部工作；对照固定模式时应新建run，
预先关闭控制器并保留其他可比设置，不在一次run中途切换控制规则。

至少手动准备 seed **17、29、43** 的独立 run：同时修改顶层 `seed`、
`training.seed` 与 `fit.training.seed`，采用不同 output_dir。每个 seed 都按
fit→generalize→profile 的条件单独判断；不从 selection 选一个最好 seed
当成总体结果。比较方法时尽量使用同一 context 抽样清单、文档、数值口径与
seed 配对；多 seed 的 context 有重叠时也不能当成完全独立文档样本。

## 6. 公平预算与解释边界

不要用“更新数相同”代替成本公平。完整动作表一次性支付 30×contexts 的评分，
RLOO 每步采样会重新评分，适配器包含可微 head/backward 与固定 probe。
报告实际 score/head 调用及 row 数、表构建、训练、probe、评估 wall time，
同时说明缓存命中、prefix 缓存/模型加载是否包含在所报区间。
这是正式公平比较需要交代的完整账本，不是说当前所有分阶段计时均已实现。
当前产物提供训练总时间、stage总时间/模型加载时间及各训练器已记录的工作量；
若要区分表构建、优化、probe、评估的独立wall time，必须增加一致的测量，
不能从总时间倒推出未记录的分项或声称已经完成matched-budget比较。

若比较 RLOO、全动作 CE、适配器，应预先设立**相同实际评分调用预算**和
**相同 wall-time 预算**的对照切片，并按真实消耗对齐；两者不能同时恰好相等时，
分别给出 matched-score 和 matched-wall 比较，而不是假装更新数已控制成本。
前向 call、带梯度 head call 与不同 batch row 的成本不相同，仍需报告工作量和
实际时间。现有模板本身不是自动的等预算benchmark；有界plateau停止只控制
当前fit的预算使用，不是自动搜索最优跨方法算力分配或自动开展等预算对照。

Profile 只比较同一工作负载、精度、device map、后端与冻结 checkpoint。
保留原生、受控原路由和候选，不混用单行训练表、完整文档前向或生成服务吞吐。
当前计时为完整序列prefill到所有logits，包含输入传输，不含CE，不测生成/KV cache；
它仍然使用开发工作负载，不是最终独立确认。
多卡层分片不是 tensor/expert parallel；逻辑少算量不等于实测端到端提速。
只有质量通过才有必要讨论成本是否值得，但通过不预设速度必然更快。

## 7. 操作者运行顺序

先按 [MIXTRAL_EXPERIMENTS.md](MIXTRAL_EXPERIMENTS.md) 和
[ENVIRONMENT.md](ENVIRONMENT.md) 准备完整本地 Base snapshot、tokenizer、
合法数据与依赖。当前交付不会替操作者占用 GPU。下面以 R1 64-context、seed17
为例；适配器改用 `fit_adapter_64.json`，不是追加第二轮专家开关。

### 7.1 复制并核对配置

```bash
mkdir -p local_configs
cp configs/mixtral/fit_router_64.json local_configs/fit_router64_s17.json
python -m experiments.mixtral_fit_experiment --help
```

编辑复制件，填写真实 `model_path`、`tokenizer_path`、40 位 `model_revision`、
三个开发文件及 GPU `max_memory`。本例将 `output_dir` 设为
`runs/fit-router64-s17-fit-001`。示例 revision/路径不是本机已具备资源的承诺。
max_memory 中编号为进程可见逻辑 GPU 编号；只有已获准且空闲的设备才能使用。

仅检查新配置结构（需要已安装依赖，不读模型/语料，也不启动 GPU）：

```bash
python - <<'PYCONFIG'
import json
from pathlib import Path
from mixtral_lab.fit_protocol import validate_fit_config
path = Path('local_configs/fit_router64_s17.json')
config = validate_fit_config(json.loads(path.read_text()))
print('fit config valid:', config['fit']['kind'], config['output_dir'])
PYCONFIG
```

结构通过不代表本地资源齐全、数据独立、显存足够或 checkpoint 已通过质量关。

### 7.2 Fit：先看训练曲线与 gate

以下命令加载真实模型并可能执行 CUDA，不是 CPU 结构检查：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.mixtral_fit_experiment \
  --config local_configs/fit_router64_s17.json --stage fit
```

人工查看 fit 报告、train 曲线和 `fit_summary.train_fit_passed`，并确认
`fit_summary.best_update > 0`。未通过就停在训练诊断，
不要绕过门槛进入 generalize/profile，也不要把数值正常退出当成拟合成功。
如需调整，复制新配置并使用新输出目录，不覆盖失败证据。

### 7.3 Generalize：不重新选择 checkpoint

复制 fit 配置，仅把 `output_dir` 改为 `runs/fit-router64-s17-generalize-001`，
保存为 `local_configs/generalize_router64_s17.json`；其他冻结设置保持相同。
显式传入该 fit 训练期选出的 best，而不是 latest 或旧格式 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.mixtral_fit_experiment \
  --config local_configs/generalize_router64_s17.json --stage generalize \
  --fit-report runs/fit-router64-s17-fit-001/report.json \
  --checkpoint runs/fit-router64-s17-fit-001/fit/checkpoint_best.pt
```

检查相同 selection 文档上对 native 与 controlled native 的配对结果、
`generalization.quality_gate.passed` 以及冻结身份。它仍是开发集筛选，不得写为
官方泛化结论。看过 selection 后的
方案调整也属于开发过程；最终独立确认需另行预注册、冻结配置/候选/范围。

### 7.4 Profile：同一候选，质量关之后

仅在 generalize 允许进入 profile 后，复制相同配置到
`local_configs/profile_router64_s17.json`，只改 `output_dir` 为
`runs/fit-router64-s17-profile-001`，再手动执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.mixtral_fit_experiment \
  --config local_configs/profile_router64_s17.json --stage profile \
  --fit-report runs/fit-router64-s17-fit-001/report.json \
  --checkpoint runs/fit-router64-s17-fit-001/fit/checkpoint_best.pt \
  --generalization-report runs/fit-router64-s17-generalize-001/report.json
```

不要在一个 shell 命令里自动串联三步，也不要为了让 profile 运行而降低事后门槛。
适配器和其他 seed 同样分别手动执行，不复用另一个候选的 fit/generalize 报告。

## 8. 预期产物与报告读取

成功阶段的产物如下；失败可能只保留 `report.json` 或部分中间文件，不能要求
尚未完成的阶段已经生成全部文件，也不能仅凭某个 `.pt` 存在就判定成功。

| 阶段 | 必看产物 | 读取重点 |
|---|---|---|
| 全部 | `report.json`、`train_contexts.json`、`freeze_manifest.json` | `status=complete`、校准一致性、配置/数据/源码身份；失败时看 `failure_phase` 与 `model_load_started` |
| router fit | `fit/training_contract.json`、`fit/train_score_table.pt`、`fit/train_score_table.json` | 全28动作表、singleton形状、native anchor、实际评分预算与表哈希 |
| router fit | `fit/metrics.json`、`fit/train_probe.json`、`fit/summary.json` | 每步完整train曲线、固定probe before/best/latest、greedy动作变化与train gate |
| adapter fit | `fit/metrics.json`、`fit/summary.json` | batch访问索引、覆盖、固定probe历史、实际BF16改变、train gate |
| 自适应 fit | 同一`fit/metrics.json`、`fit/summary.json`与checkpoint | 实际使用/下一步学习率、`adaptive_decision`、控制器快照/历史与`stop_reason`；best快照不等于latest快照 |
| 两类 fit | `fit/checkpoint_initial.pt`、`fit/checkpoint_best.pt`、`fit/checkpoint_latest.pt` | initial/best/latest不同；后续只接受freeze绑定的best，不能手工替换 |
| generalize | `report.json` 中的 `generalization` | `per_document`配对明细、三路径aggregate、`quality_gate`、router动作诊断 |
| profile | `report.json` 中的 `timing`、`quality_gate`、`evidence` | 同一冻结候选/质量证据、native/controlled/candidate耗时、原始重复测量与范围 |

Generalize 的逐文档字段 `candidate_minus_native_nll` 和
`candidate_minus_controlled_nll` 为**负值更好**；gate 的
`gain_vs_native_nats` / `gain_vs_controlled_nats` 为**正值更好**，不要读反符号。
Aggregate NLL 为 token 加权；若另算文档等权均值，需要明确标注为不同汇总。
profile gate 要求两种 gain 都不小于冻结的阈值，且
`controlled_native_nll_screen_passed=true`；不存在 override 开关。

Router summary中的控制器快照为`adaptive`，checkpoint为`adaptive_audit_state`；
adapter summary为`adaptive_controller_state`及`best_adaptive_controller_state`，
checkpoint为`adaptive_controller_state`。它们是绑定到训练证据的审计状态，
不是恢复优化器训练的接口。Adapter的`work_accounting`还分别记录包含padding的
cache FFN rows、训练head/backward rows、probe head/scored rows与修正诊断工作量；
不同项目不能当成等价单价的GPU调用相加。

`report.json` 的完成状态表示该阶段正常产生产物，不等于训练/开发关必然通过。
例如 fit 正常完成但 best=0、或 generalize 正常完成但 gain 不足，都是合法的
阴性开发记录，后续阶段会拒绝。不得删除或改写这些证据来重开门槛。
报告/manifest/checkpoint 的哈希绑定用于完整性校验，不是抵抗主动伪造的签名系统；
仍需保留不可变本地 snapshot、代码 commit 与原始运行目录。

## 9. CPU 验证与最终独立确认

运行 tiny CPU 测试，不需要语料、模型权重或 GPU：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -v
```

测试应覆盖完整动作 CE 目标、只更新新增参数、train-only best、固定 probe、
实际 BF16 改变诊断、缓存/来源/状态篡改拒绝、旧 checkpoint 拒绝、阶段门槛与
同一候选绑定；同时保留旧入口兼容性检查。具体通过/跳过数量以当前测试报告为准。
CPU 通过不证明真实 46.7B BF16、显存、多卡数值或速度通过。

最终研究结论仍需要：冻结协议后的独立确认文档、多个 seed、按文档的配对不确定性、
预定最小效应及公平成本比较。此入口不自动执行独立 held-out 确认，不因先前 smoke
失败就排除方法，也不因训练过拟合或开发均值通过就宣告模型有效。
