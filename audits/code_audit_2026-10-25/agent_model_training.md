# 模型、预训练与系统测量审查

审查对象：`kkc25/GB_RL_MoE_Loop`，提交 `f2bef422f7315f30618f00c690bf82810d1e85ac`。本报告只读审查源码与已上传 JSON/日志；没有读取未上传权重，没有在 A6000 上重跑训练。

本分工覆盖模型、LM 优化器/恢复、E00 配平、E11 动态执行与计时。E01/E03 的统计、标签和学生训练由另一个分工主审；本报告独立复核了影响 E11 的特征接口问题。

## 1. 主要判断

核心语言模型结构基本落实了方案：12 层、512 隐藏维度、8 专家、Top-2、末层只循环 FFN、attention 每块一次、固定残差系数、绑定词表头、文档内因果 attention。S67/M105 参数量与方案的精确计数一致。没有发现“名为 Top-2 实际只执行一个专家”或“每次循环把 attention 也重复执行”的问题。

但是，现有结果不能视为完整执行计划后的路线否定。实际优化 batch 明显缩小，dense 对照配平指向错误规模，验证只有 16,342 token，恢复成本账本漏计；E11 存在训练/部署输入不一致和固定/动态计时不对称两个确定问题。应先修正接口和测量，再重新决定停止或扩大研究。

严重程度约定：P1 直接使关键比较/算法身份不成立；P2 影响训练条件、效率、复现或证据力度；P3 为未触发的扩展限制。严重程度不表示已经证明该问题是所有负结果的唯一原因。

## 2. P1：动态停止学生训练使用 raw h，部署却使用 RMSNorm(h)

**证据链**：

- `eval/counterfactual.py:80–86,105–120`：缓存的是末层 FFN 前、`ffn_norm` 前的 raw hidden state；`states['h']` 保存该 raw h。
- `train/value_student.py:35–43,73–75,108–110`：学生输入 raw h，并使用训练集 raw h 的每维 mean/std 标准化。
- `experiments/e03_halting.py:165–173`：离线校准仍输入 raw h。
- `eval/dynamic_halting.py:26–31`：在线却先算 `normalized = block.ffn_norm(h)`，再把 `flat_normalized` 传给两名学生；mean/std 仍来自 raw h。

这不是合法的等价变换。RMSNorm 包含逐 token 均方根缩放和训练后的逐维权重，通常与 raw h 的线性标准化不相同。在线得到的是另一套输入分布上的停止策略，E03 校准的 tau 对该策略没有原有保证。此 bug 独立于是否存在未来信息泄漏，也不需要任何大权重即可确认。

**影响**：`runs/e11/m105_s0_dynamic_profile.json` 的 NLL 与平均深度，虽然可能是真实运行得到的数值，但不能作为“E03 校准停止器接入模型后被复现”的证据。代码改变停止规则后，仅靠 NLL 仍在容忍阈值内不足以修复算法身份。

**最小修改**：router 仍使用 `ffn_norm(h)`；学生使用 `h.reshape(-1, h.shape[-1]).float()`。最好以共享 `student_features` 接口和明确 `feature_version='pre_ffn_raw_v1'` 约束缓存、校准和部署。

**验收**：相同真实 prefix 同时走“缓存离线预测”和“在线预测”，逐 token 比较两名学生预测、delta、停止 mask、最终 logits/NLL。使用有限 tau，并确保既有 stop 又有 continue。现有 `tests/test_dynamic_halting.py:13–34` 只测正负无穷阈值，此时输入再错也会全部 stop/continue，因此无法覆盖本错误。需要权重才能量化旧结果的改变幅度，但无需权重就能判定接口有错。

## 3. P1：S67 的 dense 基线没有实现宣称的参数/计算匹配

实际 `configs/e00_dense_b3.json:2`、B4:2、B5:2 以及各自 `runs/e00/*/resolved_config.json` 确认 dense 宽度分别为 1024、4096、1024。它们对应 M105 的活跃宽度/参数量，实际比较的 MoE 却是 m=256 的 S67。

| 配置 | 参数 | 每 token FFN 主矩阵 FLOPs，R1 | 每 token FFN 主矩阵 FLOPs，R4 |
|---|---:|---:|---:|
| S67 MoE，E8/k2/m256 | 67,170,816 | 18,874,368 | 23,592,960 |
| B3 dense，m1024 | 48,247,296 | 37,748,736 | 37,748,736 |
| B4 dense，m4096 | 104,870,400 | 150,994,944 | 150,994,944 |
| B5 shared dense，m1024 | 48,247,808 | 37,748,736 | 47,185,920 |
| M105 MoE，E8/k2/m512 | 104,919,552 | 37,748,736 | 47,185,920 |

表中只比较 FFN 主矩阵 FLOPs，不冒充端到端 FLOPs或延迟；attention/head 是共同的额外部分，router 和 dispatch 也尚未计入。B4 比 S67 多约 56% 参数。B3 比 S67 R1 的 FFN 算量多一倍，比 S67 R4 多 60%。B5 相同深度 FFN 算量为 S67 的两倍。B5 的单专家 softmax 恰等于 1，因此其末层实现可视为 dense FFN；该结构命名本身没有问题。

**影响**：S67 的这组 NLL 可保留为不同结构 smoke 结果，不能标为配平对照后据此否定循环/稀疏价值。M105 目前只有 B2 的 100M pilot，也不能借用 S67 的 5M dense 数字完成 M105 对照。

**修正**：先选明确比较轴。S67 R1 的 dense 活跃宽度应约 512；参数匹配 dense 宽度约 2048（MoE router 的小额参数差单列）。若匹配 S67 R4，应按总 FFN 算量调最后层宽度/深度，不能仍称所有层 m1024 是算量匹配。M105 的现有 B3/B4/B5 配置可保留，但须与 M105 同 token/同预算运行。

独立算术复现见 `reproduce_model_training_arithmetic.py` 和 `model_training_arithmetic.json`，与上传 run_summary 参数一致，不需要权重。

## 4. P2：实际 global batch 约 1024 token，没有梯度累积

`lm_config.py:17` 默认 batch=1，所有已运行 E00 resolved config 也均为 1；`train/lm.py:87–105` 每一条 packed sequence 立即 zero_grad、backward、optimizer.step，没有 gradient accumulation 或 DDP 汇总。

M105 日志：100,000,000 input token / 97,657 updates = **1023.992 input token/update**；99,807,178 supervised token / 97,657 = **1022.018 有效 token/update**。方案 common protocol 第 45 行要求 pilot 8192、主实验 32768 有效 token；microbatch=1 是显存校准起点，不是要求 optimizer batch=1。

**影响**：实际 pilot 的逻辑 batch 约为方案的 1/8，与正式起点相比约 1/32；相同 3e-4 学习率不能自动视为同一优化条件。高噪声、小批次、过多 optimizer/validation/checkpoint 次数及 dispatch 开销都可能影响收敛/效率，但仅凭源码不能证明修改 batch 必然使路径收益转正。

**修正**：配置应显式区分 microbatch_sequences、gradient_accumulation、target_effective_tokens_per_update。按有效监督 token 加权累计 loss，保证最后不足 batch 的归一化正确。先做相同 token 预算下的 batch/LR 小范围校准；不需要直接训 1B 才能排查。

另 `depth_schedule='random'` 实际是 `1 + (update + seed) % max_calls`（train/lm.py:92–95），即周期轮转。它有均匀边际频率，但不是方案的每 batch 独立均匀抽样。应明确改名 cyclic 或换成可恢复 RNG 抽样；不能把此差别单独断言成负结果根因。log_interval=10 与周期4相互作用，使 seed17 的训练日志只记录深度2/4，虽然训练实际也执行1/3。

## 5. P1/P2：E11 固定路径计时额外算 loss，动态路径没有

`eval/profile_dynamic.py:47–50`：dynamic 只调用 `policy.forward` 输出 logits；固定 R1/R4 调用 `model(input_ids, labels, ...)`。后者触发 `model/transformer.py:321–324` 的完整 float32 cross_entropy 与归约，前者没有该工作。

因此即使 GPU 独占，当前比较仍不公平。报告发现 dynamic 比 R1 快时，把问题仅归为共享负载不足以解释测量设计。不能根据目前代码估计不公平 CE 占了多少毫秒；这需要修正后测量。

**真实执行复现**：在审查环境 PyTorch 2.5.1+cpu，使用真实 `profile` 函数和2层16维微型模型，包装 `torch.nn.functional.cross_entropy` 计数；每模式1次warmup+2次计时，固定R1/R4均调用CE 3次，dynamic为0次。没有拿CPU延迟冒充GPU延迟。

**最小修正**：计时中固定路径不传 labels，只测相同 input→logits 边界；若需要计入 loss，则所有模式统一计入。质量评估在计时外独立运行。每模式重置 peak-memory stats，随机交错测量，按方案做相同热身、长度、重复数和隔离 GPU。

现有 profile 没有 reset peak stats，三个模式相同的 peak memory 值可能只是跨模式累计峰值，不能用于逐方法显存比较。默认 warmup20/steps100 也小于 E11 文档 warmup30、200步或30秒并重复5次的确认门槛。

## 6. P2：动态平均深度分子、分母掩码不一致

`eval/profile_dynamic.py:33–37` 累加所有位置 `rounds.sum()`，却除以仅 `labels != -100` 的监督 token 数。packed sequence 最末位置、文档边界和 padding 都可能不被监督；所以即使全部 R1，当前公式也可得到大于1的“mean_rounds”，全部 R4 可大于4。

**修正**：分别报告 `rounds[supervision_mask].mean()`（质量对应的位置平均深度），及所有实际执行位置的 expert-token-pair 总数（真实成本，包括 padding）。如希望成本/监督 token，则明确指标单位，不称其为平均深度。

例如本次16个1024位置共16384个执行位置，而监督计数16342；两种分母已有可见差异。这个偏差不足以解释所有停止现象，但在0.005 nat/小幅系统收益研究中应统一口径。

## 7. P2：名为 expert_calls 的账本实际记录 FFN rounds

`model/transformer.py:139,301,305` 每执行一轮累加1；Top2MoE 实际通过 `model/top2.py:80–82` 执行两个专家。`train/lm.py:109` 直接把 output.calls 求和记入 `expert_calls`，没有乘 Top-k，也没有区分 dense/MoE。

- S67 B0 记录60,002,304，恰等于4883×1024×12；实际执行专家-token对为120,004,608。
- M105 B2 记录1,350,009,856；同口径实际专家-token对为2,700,019,712。

这些计数包含实际执行的 padding 位置，因此不要再拿 real input token 去强行乘除而隐藏 padding。修正应保留 `ffn_rounds` 并新增 `expert_token_pairs`，dense 执行另标口径；不同专家宽度还需要 FLOPs，不能仅比较调用数。

此 bug 不改变模型梯度，但影响资源公平性与后续成本约束核算。

## 8. P2：恢复 checkpoint 漏掉已用时间，best 值及随机态恢复不完整

`train/lm.py:115–118` 保存 checkpoint 时，ledger.elapsed_seconds / GPU-hours 还没有累加本段运行；直到训练退出的121–125行才更新，并仅写 JSON，没有重新保存 checkpoint。`resume` 的77–81行从旧 checkpoint ledger 恢复，因此已经完成的训练段成本被遗失。`train/recovery.py:149–157` 有同样顺序问题。

上传日志已确认实际使用该路径：各 E00 resolved config 均含 resume。B0 初始100步 console 记录140.068秒；其最终 ledger 只有末次迁移后的1466.721秒。B0 metrics 还显示迁移前update1190，恢复后从1150继续，至少一段回滚重算没有 lost-compute 项。这里不拿跨进程含空闲的总时间戳差直接冒充精确 GPU-hour，但保存顺序已足以确定旧段漏计。

其他恢复问题：

- 每次 train_lm 的 `best_nll` 重置为 infinity（86行），未保存最佳值，下一次 validation 可以把更差的权重覆盖到 best_validation.pt。
- 保存了 torch_rng/cuda_rng（58–59行），加载时没有恢复。当前 LM 无 dropout、深度周期式、sampler 确定，因此不能据此声称当前恢复权重必然漂移；但恢复契约不完整，改成真随机深度后会直接受影响。
- checkpoint 内没有固化并校验 tokenizer/data-content hash；run_lm 只把当前 manifest hash 写进 environment。更换同 shape 的 token stream/词表或 batch_size 可以静默恢复，数据索引语义也随之改变。
- learning_rate_at 依赖当前 `run.updates`，而已运行100步 smoke 的总steps在继续训练时改为4883。日志直接显示step90学习率3.6877e-5、恢复step100学习率2.999997e-4。此跳变不是严格延续原 schedule，应固定全程计划或显式登记新的阶段 schedule。

**真实执行复现**：真实 train_lm 微型训练，注入仅用于账本测试的确定时钟和验证分数。第一段时钟7秒，返回/JSON为7秒，但checkpoint保存0秒；恢复后第二段5秒，返回“累计”5秒而不是12秒。第一段最佳验证0.9出现在update2，恢复后的注入验证2.0更差，仍覆盖best_validation为update3。注入值不是实际LM性能数据，而是隔离验证保存/恢复语义的受控输入。复现脚本和JSON已随报告提供。

**修正**：每次存盘前累计本段时间并重置段计时；保存/恢复 scheduler/best/RNG/sampler state 与有效 batch/token hashes；真实丢失的更新和重新执行成本另列。模型/optimizer 的加载机制本身存在，不能说仓库完全没有恢复能力。

## 9. P2：每个 update 重新打乱全部数据，数据采样复杂度过高

`train/lm.py:87–90` 每个 update 新建 sampler 并取 next。`data/packed_lm.py:343–361` 每次迭代都会完整构造N个索引、洗牌、构造全部batch列表；下一次 update 将相同工作再次做一遍。每 epoch 从应有 O(N) 变成 O(N²/batch_size) 的索引工作。

通过真实 sampler 源码的 CPU-only 复现，N=97,657（上传100M manifest中的真实序列数）、前12步：首次记录每步重建总计0.526秒，复用一个 iterator 总计0.038秒，输出顺序完全相同。这个时间只属于审查主机，不外推成 A6000 的确定加速比；复杂度错误则由代码确定。重新运行的具体时间保存在JSON，允许正常波动。

**修正**：按 epoch 创建一次 sampler/iterator，从保存的 start_batch 继续；跨 epoch 再换顺序。DataLoader/prefetch 可以后续做，不需要先改变数据统计定义。

## 10. P2：当前 E00 证据仍是 smoke/pilot，没有完成质量否定门槛

- S67只有5M token；方案E00将5M–20M定义为smoke，累计50M才是debug模型阶段目标。
- M105只有B2一个seed的100M pilot，没有同规模B0/B1或dense同预算对照。
- 每次 NLL 验证只有16,342监督token；方案筛选固定1M前缀，正式候选10M验证，并要求文档级损失及 paired bootstrap。
- run_lm 只输出R1/R2/R4，缺R3；缺逐专家负载、router熵、hidden/update范数、重复专家率等计划要求的诊断。没有这些信息时无法区分“循环没有价值”和“循环/专家没有学好”。
- M105现有R1相对R4只差+0.000573586 nat，本身已远低于停止容忍0.005；R2甚至比R4低0.000657247。故“动态达到阈值并省调用”必须进一步与固定R1/R2及等成本随机策略比较，不能只对较冗余的R4报告优势。此处是对上传指标的直接运算，不表示已做新的显著性检验。

建议把状态标记为“E00 smoke/pilot 已完成；正式比较待补充”，不需要把所有扩展无条件跑满。只有在修正实现并补齐足以支撑决策的少量实验后，才重新判断E02/E11是否有必要继续。

## 11. P2：恢复任务 CLI 的 evaluate 模式没有加载 checkpoint

`run_recovery.py:65–80` 总是新建随机模型。只有 mode=train 时才调用 train_recovery，而真正读取 run.resume 的代码仅在 `train/recovery.py:96–104`。mode=evaluate 直接进入86–89行评估随机模型；即使用户传 `--resume` 也没有加载。默认 output_dir 下环境、配置和 stage_result 还可能被此次随机评估覆盖。

**真实执行复现**：对原CLI运行 `--mode evaluate --device cpu --config configs/recovery_c0_identity_dense.json --resume <不存在的文件>`，命令仍以0退出，输出随机模型的0/97正确率；optimizer_updates=0。证据文件保存在 `recovery_evaluate_probe/`。若真正加载resume，必须因文件不存在报错。这验证了CLI忽略checkpoint，并非只凭分支阅读推测。

这是确定的复现入口 bug，但当前上传 C0/C1 结果来自 train 路径，因此不能说它解释了已有 C1 泛化差。修正 evaluate 分支显式要求 checkpoint，加载并核验 data_hash/model config，输出到新的评估目录；用随机初始化与刻意改变的权重做断言测试即可，无需大权重。

## 12. 模型核心已落实/未发现问题的部分

1. `model/transformer.py:295–305` attention 在外层 block 循环执行一次，末层fixed_depth只重复FFN，符合主方案。
2. `model/top2.py:16–37,80–83` 28个无序不同专家pair映射及两专家加权和逻辑正确；组合内部交换顺序不改变同权重对应输出。
3. `Block` 用 `Top2MoE(..., freeze=False)`，训练时 gate 混合权重保留梯度；不能误把默认冻结 wrapper 当成LM预训练router被冻结。
4. 专家dispatch是真正按被选token分组执行，不是先算全部8个专家再mask。当前Python逐专家实现仍有系统开销，但少算机制真实。
5. 文档分段mask是same-document与causal交集；默认全序列路径没有读未来标签作为输入。
6. LM外部 `lm_loss` 按有效监督token求和再归一化，校正了模型内部按序列均值在一般不同长度时可能有的权重差异；当前训练batch1不触发序列权重差。
7. 绑定词表头、参数计数、SwiGLU三矩阵、两norm/块+final norm均与已选105M规范一致。

## 13. 精确验收与下一轮所需材料

本轮已运行：纯Python参数/FLOPs/账本整数核对、真实sampler顺序等价及CPU采样开销复现；随后使用可用的PyTorch 2.5.1+cpu运行真实profile路径CE计数、真实train_lm保存/恢复账本与best覆盖测试、meta模型真实参数计数、微型Top-2训练gate/专家梯度非零测试、原recovery CLI evaluate入口复现。原运行环境是PyTorch2.12.1+cu130，版本与CPU/GPU差异明确保留；本轮不提供A6000速度/质量复现实验。证据见 `reproduce_model_training_torch.py` 和 `model_training_torch_checks.json`。

下一轮首选最小材料：M105已使用的模型 state_dict（可不含optimizer）、E01 student权重及mean/std、与E11同一批prefix对应的 input_ids/segment_ids/labels、小量缓存raw h与Q标签、完整hash/version metadata。这样可量化feature修正影响、统一BF16/FP32路径差异、逐token比较离线/在线停止，而无需上传所有训练历史。

若要进一步审查训练本身，再提供代表性 checkpoint 的逐层norm/路由负载摘要，以及保存/恢复前后的完整checkpoint metadata；仅凭现有json无法确认是否真的发生专家塌缩、隐藏状态收缩或router泛化失败。

## 14. 本分工覆盖清单

完整读取：`config.py`、`lm_config.py`、`recovery_config.py`、`run_lm.py`、`run_recovery.py`；`model/{__init__,moe,top2,loop,transformer}.py`；`train/{lm,common,recovery,recovery_common}.py`；`eval/{dynamic_halting,profile_dynamic,profile,resource_accounting}.py`；全部7个 `configs/e00*.json`；`tests/{test_top2,test_model_policy,test_dynamic_halting,test_training,test_recovery_model,test_math_resources}.py`；`data/packed_lm.py` 的采样器与标签接口。

交叉读取：`eval/counterfactual.py` 缓存及主标签路径，`train/value_student.py`，`experiments/e03_halting.py` 的学生加载/预测。其余E01/E03主审不由本报告重复承担。

证据读取：全部E00已上传resolved config/run_summary/cost ledger、S67所有metrics和console、M105 summary/config、E11 profile JSON；对照 prior research_bundle 中E00/E11与common protocol。对权重缺失有实质影响的点均明确标出。
