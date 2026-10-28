# Mixtral：可改善空间、可学习性与可选残差适配

后续独立四角色 recovery 协议见 [RECOVERY_EXPERIMENTS.md](RECOVERY_EXPERIMENTS.md)：先做历史 cache/在线桥接，再由独立 development 选 checkpoint；本文原 baseline 与命令保持不变。

后续实验优先使用 [FIT_FIRST_EXPERIMENTS.md](FIT_FIRST_EXPERIMENTS.md) 的独立
fit→generalize→profile 路线。它新增 train-only 全28动作期望CE拟合与固定train
probe适配器，不把4步smoke未改善当作方法否定。本文保留旧入口与原始方案边界，
其中selection-best适配器不等同于新train-best checkpoint协议。
新非smoke模板还提供train-only的有界学习率/plateau控制，不自动调rank、scale、
epsilon或全部超参数；开发泛化和性能阶段仍只接收冻结候选。

同一角色内截断后的重复输入现在写入数据身份的 `effective_input_statistics`：
包含文档数、唯一有效序列数、冗余文档比例和重复组。该统计不改变采样权重，
不自动删除文档，也不证明语义独立性；跨角色重复仍直接拒绝。

## 结论边界与本轮目标

`runs/mixtral_v1/` 的 64 步实验只使用第一篇开头 32 个 token 位置；两篇
selection 的 greedy NLL 未改善。强制重复冻结末层 FFN 则显著恶化。这支持
停止盲目增加同一小样本的步数，不支持否定所有 RL 或经适配的循环结构。

本方案分开回答三个问题：
1. 固定骨干、固定评分口径下，替代专家动作有多少可改善空间？
2. 不使用答案的策略能否预测有益动作并在独立开发文档上泛化？
3. 新增少量可训练修正参数后能否改善输出？这与只换路由是不同实验。

本次交付只提供代码、配置和 CPU 测试，不自动下载权重、执行 GPU 作业或打开
newtest。CPU tiny 模型正确性不能代替真实 46.7B CUDA 数值与性能验证。

### 启动前检查与阶段依赖

先加载本地 tokenizer/config、分词并检查有效位置与本阶段配额，再加载大模型
权重。`oracle` 仅规划其指定开发角色的 oracle context；`validate` 和无 actor 的
`profile` 不强制规划训练缓存。`train`、`adapter` 以及带 actor checkpoint 的
`profile` 仍严格重建训练采样清单，保持 checkpoint 数据绑定。

所有阶段继续保留 train/selection/calibration 的来源与跨集合重复检查。
配置结构错误仍可能在建立输出目录前失败；进入 preflight 后的错误会保留在
新目录 `report.json` 中，包含 phase、preflight 状态和 `model_load_started`，
可区分输入失败与模型加载/执行失败。失败目录不能原地覆盖。

本地 tokenizer/config 在预检和后续装载间复用并复核。权重只做一次完整内容
hash，另以文件签名检查加载期间的替换/写入；依赖不可变的本地 snapshot，
不是抵抗主动伪造文件元数据的安全证明，也不重复读取整套权重做第二次 hash。

## A. 修正训练覆盖：文档平衡、位置随机

`context_sampling.strategy=document_balanced_uniform_positions` 使用独立 seed RNG，
在各文档有效 next-token 位置中无放回抽取，不再顺序截取前 N 个位置。
预算足够时每篇符合条件的文档都有配额；短文档不足的配额重新分配。
若预算小于文档数，则随机选择文档子集，并记录未覆盖的文档；总有效位置不足
直接拒绝，不能悄悄缩小样本数。此分布是文档平衡，不等于全语料 token 均匀。

`train_contexts.json` 保存角色、seed、文档 ID、token 位置、目标、配额、实际
覆盖和 sampling hash。Prefix 每个入选文档只计算一次。训练、oracle 以及
适配器都使用这一采样接口；不要把 context visits 当作独立样本数。

旧配置没有采样字段时会规范化到新策略，新 checkpoint 身份包含采样清单。
这改变了实际训练输入，必须使用新输出目录和新 run。旧 first-N 结果保留原样，
不能编辑 binding 来让旧 actor 冒充新实验；需要复核旧实验时使用原 commit。

## B. 标签知情 oracle：只测空间，不当策略成绩

`--stage oracle` 默认只在 calibration 上运行；不默认计算 selection NLL。
只允许 calibration/selection 开发角色，不接受 train/newtest。选择 selection
用于 oracle 后，它就是诊断开发集，不能再称作未触碰的最终确认集。

每个 context 枚举：
- R1：全部 28 个专家对，含原生对这一真实 fallback。
- R2：先固定执行一次原生 R1，再枚举 28 对与 STOP。STOP 分数精确复用 R1
  参考；收益不足 tie tolerance 时优先 STOP。

R2 是**固定原生第一轮条件下**的空间，不是同时搜索两轮 28×28 路径。
Oracle 使用真实目标标签挑选动作，只是该数值协议下的上限，不是可部署收益。
它不能告诉我们合法输入特征是否足以预测动作，也没有训练 STOP 策略。

为避免枚举批大小任意变化，每个评分调用只使用一行 context。报告明确标记
为 singleton counterfactual baseline：不是完整序列在线 NLL。BF16 GEMM 形状
与路由近似并列可能影响分数；重复原生 anchor 只验证单行协议内部一致，不能
证明它与生产批形状等价。不能直接减去此前 full-document NLL 当作收益。

每个 context 有 58 次评分调用，另外有一次原生状态推进和 STOP readout 检查。
`max_score_calls` 约束的是评分调用，不是总 GPU 算子数或账单。报告保存全部
候选 CE、动作、精确最小值、阈值化收益、收益分位数、文档分组、tensor/来源
hash 以及实际调用数量，不只保存一个漂亮的平均值。

## C. R1 路由训练：观察是否真的改变决策

先看多文档 oracle 的收益分布，再决定是否扩大固定骨干的 R1 RLOO。
不要仅因为 4/64 步 smoke 没收益就增加数量级预算，也不要只调大学习率。

训练结束分别重载 best 和 latest，报告 selection NLL、实际执行 greedy 动作
与原路由一致率，以及原路由相对最佳替代的 log-prob margin。诊断在实际完整
padding 形状中计算 gate/actor，再遮罩监督位置，避免为了诊断而改变 GEMM
形状。更新成功不等于 greedy 改变；best=0 也不等于参数完全没有更新。

收益上限高但策略未改善时，再比较合法特征的预测能力、探索先验、采样信息量
和优化过程。本版本没有实现 oracle 标签蒸馏或完整的特征可预测性基准；不能
把新增日志当成这些研究问题已经解决。最终确认仍需独立冻结 candidate/scope。

## D. 可选适配器：训练修正能力，不冒充循环收益

`--stage adapter` 必须显式设置 `second_round.enabled=true`，默认关闭。
结构为：固定原生 R1 得到 h1，再加一个可训练低秩残差修正。

- 原 Mixtral 全部冻结，只更新新增的两个低秩矩阵；默认 rank=8。
- 一个矩阵使用局部随机初始化，另一个为零，因此初始化输出等于 R1；不能
  两个矩阵都初始化为零而使梯度失效。
- 适配器用 FP32 运算，修正量转换回原隐藏 dtype 后做残差相加，保持初始
  BF16 数值口径。损失通过冻结的 final norm/head 对修正量反传；冻结权重
  不等于把这段计算放进 no_grad。
- 仅用 train context 的 next-token CE 训练；selection 选择 best，update 0
  参与，持平保留最早。保存 initial/best/latest，使用独立绑定格式且不支持
  resume。记录实际 batch、参数量、梯度、成本和 selection 改善标记。
- 对照包括固定 R1/STOP、未经训练的原生 FFN 强制重复、零初始化适配器及
  训练后适配器。所有对照保存，不只展示最佳结果。

**这个适配器不会再次调用原末层专家，也没有 learned STOP。**它检验新增修正
能力，不是已经训练好的 recurrent MoE。若有收益，还需要参数/计算匹配的
单次适配对照，才能讨论 loop 本身的价值；本版本尚未实现该因果对照。
冻结路由 oracle 无收益也不能排除新增参数适配的收益，二者假设空间不同。
selection 改善不等于统计显著、实际可用或 held-out 改善。

## 配置与执行顺序

沿用 [MIXTRAL_EXPERIMENTS.md](MIXTRAL_EXPERIMENTS.md) 的离线权重、依赖、数据
去重和多卡约束。每一步复制配置后修改本地模型/数据路径及**全新的 output_dir**。
以下仅为操作者手动运行命令，本次交付未执行 GPU 实验。

| 配置 | 用途与规模 |
|---|---|
| `learnability_smoke.json` | 新采样与诊断的流程短测，不是质量结论 |
| `oracle_smoke.json` | 8 个 context、至多 464 次评分调用 |
| `learnability_pilot.json` | 每角色 8 篇、长 256；256 个训练 context、64 次 R1 更新；oracle 64 个 context/3712 次评分调用 |
| `second_round_adapter_smoke.json` | 默认独立输出；显式打开 rank8 适配器，32 context、4 次更新 |

配置位于 `configs/mixtral/`。Pilot 仍是有界开发诊断，不是充分训练的保证。
保持 backend 固定，eager 与 grouped_mm 的 NLL 不能混作一个基线；改变后端
需要新配置、新结果与重新检查。四卡层分片不是 TP/EP，不保证四倍速度。

```bash
# 先复制配置并填入真实路径；每个目标配置设置不同的新 output_dir。
python -m experiments.mixtral_experiment --config local_configs/validate.json --stage validate
python -m experiments.mixtral_experiment --config local_configs/oracle.json --stage oracle
# 只有诊断与预定预算支持继续时，分别手动选择以下实验：
python -m experiments.mixtral_experiment --config local_configs/router.json --stage train
python -m experiments.mixtral_experiment --config local_configs/adapter.json --stage adapter
```

`adapter.json` 必须来自或显式等价于 enabled 的适配器配置。Oracle 用当前实际
模型、tokenizer、数据及采样清单重新绑定身份，不复用旧训练报告来替代检查。
历史 `runs/mixtral_v1/loop_probe.py` 保留为当时的证据，不作为正式新入口。

## 预先声明的决策规则

1. 原生一致性失败、输入身份不符或预算不足：停止并保留失败，不放宽门槛掩盖。
2. 冻结 R2 oracle 基本选择 STOP、收益不足实用阈值：不扩大相同冻结重复方案。
3. Oracle 有空间但 R1 路由不泛化：优先检查可预测性和探索，不把 oracle 当成果。
4. 适配器训练只改善训练集、best 仍为初始化：保留负结果，不声称已学会循环。
5. 所有正结果仍需更广样本、多 seed、置信区间及独立确认；本代码不自动批准
   增加算力、扩大实验、训练 R2/STOP 或对外发布结论。
