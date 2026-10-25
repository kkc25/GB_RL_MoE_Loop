# 可复现的数据、训练、恢复与评测协议

本文件约束所有实验。默认 2026-10-25 形成的设计；具体依赖版本、数据 revision 和 GPU 拓扑必须在实际运行时记录，而不是从当前网页的 `latest` 推断。

## 1. 数据路线：先让变量可控

### 1.1 主语料

主线使用 [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 的 `sample-10BT` 配置，固定数据集 revision。该配置的 10B 是来源数据卡按 GPT-2 tokenizer 计数的量级；本实验一律重新用自己的 tokenizer 计数，不能直接复制来源 token 数。

建议流程：

1. 固定 revision 与 parquet 分片清单，保存 URL、大小、校验和、原始文档 ID。
2. 在任何 tokenizer 训练和序列切块之前，以规范化文档内容 SHA-256 去除完全重复，以稳定文档 ID/内容 hash 划分 train/validation/test。
3. 例如 hash 模 10,000 后，0–9799 为 train、9800–9899 为 validation、9900–9999 为 test；此分配仅是项目选择，不是来源官方 split。
4. 主训练先截取 300M 个**本项目 token**的固定前缀作筛选，其中前 100M 是 pilot，不另加 100M；确认运行升到累计 1B。保留同序列前缀，避免方法 A/B 读到不同难度的数据。只有继续同一配置、样本流和优化器状态的运行才可抵扣已有训练；被放弃的学习率试验、重新初始化或重新训练仍须单列成本。
5. validation/test 各固定 10M token；开发初期使用其中固定 1M validation 前缀，正式确认再使用完整集。test 在超参、门槛、预算和主图定义冻结后才使用。
6. 保存去重和 split 的清单；如果先按 shard 切训练/测试再去重，应检查跨 shard 重复，否则低估泛化损失。

数据卡声明 ODC-By 1.0，并说明还受 Common Crawl 使用条款约束；发布数据处理脚本、文档 ID/分片清单和引用信息。此处只陈述数据卡，不对原网页内容给出统一权利保证。主线需要的是可复算的数据来源，不是重新分发完整网页集合。

### 1.2 Tokenizer

- 在 train 中确定性抽取约 100M 字符的样本，训练 byte-level BPE，目标词表 32,768，特殊 token 也计入其中。
- 固定 normalization、pre-tokenization、特殊 token ID、BOS/EOS 插入规则；保存 `tokenizer.json`、配置、训练文档清单与 SHA-256。
- 所有实验使用同一份 tokenizer，不能针对某条方法重新训练。
- 按文档边界追加 EOS，先确定 attention 是否允许跨文档，再统一所有方法。最简单基线是拼接后 causal attention；如采用文档隔离 mask，应在所有方法和速度测试中一致执行。
- HF 的 [Tokenizers 官方指南](https://huggingface.co/docs/tokenizers/quicktour)提供 BPE 训练机制。实际可用版本必须在本机锁定后测试；不把当前文档的可用 API 自动等同已安装软件。

可选复用 [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) tokenizer 以减少准备工作，但其 [官方 config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json)词表为 49,152，不能仍沿用 V=32,768 的参数、读出成本或直接比较不同 tokenizer 的 PPL。选择此替代方案时应整套实验重算并固定，不能中途混用。

### 1.3 备用语料与分布外测试

[SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)有 `fineweb-edu-dedup`、`cosmopedia-v2`、`python-edu`。其 FineWeb-Edu 去重子集与主语料来自相关来源，**不能把它未经去重直接称为独立 OOD 测试**。备用路线优先取主语料的不同 Common Crawl dump/domain 分组，经跨集合去重后作分布移位验证；如果使用 Cosmopedia 或代码语料，分别核对该子集说明，报告语言/领域/tokenizer 差异。

算法任务（置换组合、模运算、计数、有限状态机）使用自行生成的数据、固定生成器 seed、训练/测试规则分离。它们用于辨识机制与组合泛化，不代替自然语言结论。

## 2. 主训练规格

| 项目 | 默认值 | 说明 |
|---|---|---|
| 模型 | M105，具体见容量文件 | 最后一块 FFN 循环是核心阶段 |
| 序列长度 | 1,024 | 先 S512 smoke test；正式比较统一 S1024 |
| 优化器 | AdamW | 固定 betas=(0.9,0.95)、eps=1e-8、weight_decay=0.1；bias/norm不衰减 |
| 峰值学习率 | 候选 3e-4、6e-4 | 仅在公共基线小预算 pilot 选一次，所有方法共享，避免对新方法投入更多搜索 |
| 学习率日程 | 按累计有效训练 token 数；前2% warmup，之后 cosine至峰值的10% | 这些是起始超参，不是保证最优值 |
| 全局有效 batch | 主实验 32,768；pilot 8,192 非 padding token/step | 以 token 数记录；变 GPU 数时保持当前阶段逻辑 batch 不变，pilot→主实验的切换对所有方法一致 |
| 单卡 microbatch | 从 1 条序列起校准，可升至 2/4/8/16 | 由真实峰值显存选择；用 gradient accumulation达到全局 batch |
| 梯度裁剪 | global norm 1.0 | 记录裁剪触发率 |
| 训练精度 | BF16 autocast，FP32 optimizer/master状态 | router logits/softmax、损失归约和价差计算按需FP32 |
| dropout | 主机制实验 0 | 消除分支反事实标签的 dropout噪声；泛化扩展可另测 |
| 种子 | 17、29、43 | 相同种子的方法配对；数据排序可固定公共seed17 |
| 调试预算 | S67 累计 50M token，其中包含 5M–20M smoke | smoke 是调试训练的前缀，不重复加算 |
| 筛选预算 | M105 累计 300M token，其中包含 100M pilot | 单种子筛选只决定后续投入，不作为最终统计结论 |
| 确认预算 | 每种子1B token | 是否加长取决于学习曲线与项目GPU-hour账目 |

这是预注册起点，应在 E00 后冻结。若改变学习率/数据/深度日程以修复不稳定，所有对照从相同公共 checkpoint 分支或重跑，并在运行表标记改变。不能把较晚训练、更长调参或更多监督成本归入模型结构收益。

## 3. 单卡与多卡实现顺序

1. 先完成单卡 dense reference、Top-2 reference、固定循环、缓存末端分支的数值一致性。
2. 路由先用透明的 PyTorch 实现：FP32 router打分、Top-k、仅在选中集合重归一化、按专家分组计算、原位置 scatter-add。确认输出与小批量逐token参考一致后，才切换 grouped GEMM等优化。
3. 小模型所有专家保留在同一 GPU，不需要专家并行。
4. 如启用 DDP，官方文档说明它按副本同步梯度，不会自动把模型参数分片。动态 token路由可能使部分专家参数在某一步未使用，必要时设置 `find_unused_parameters=True`，且不要提前声称 `static_graph=True` 适用；固定图条件需要实际验证。参见 [DDP 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。
5. 权重在循环中复用时，activation checkpoint采用经验证的非重入方式；检验一轮、多轮、某专家空负载、不同 rank专家使用集合不同的情形。为追求 static graph而强行让所有专家执行，必须重新计算实际开销。
6. 使用 [SDPA 官方接口](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)先建立可运行路径。内核自动 fallback或重新编译可能改变耗时；记录选择的backend和dtype，不把框架名当成真实内核证据。

主结果能用单卡完成时，三个种子并行通常更简单；DDP速度是否有益由 E00 的 `q_N/(Nq_1)` 决定。

## 4. GPU 资源变化与恢复

### 4.1 默认策略

不在 optimizer step 中间改变 world size。任务只能在完整 optimizer step 边界存盘；资源变化后从该 checkpoint 重新启动。即使支持 elastic workers，也不能把“进程自动重启”当“数据和梯度统计自然保持一致”。

存盘必须包含：

- 模型、optimizer、scheduler、混合精度相关状态；若使用 FP16还包含grad scaler。
- 全局 optimizer step、累计有效token、累计实际GPU-hour、样本游标/文档序列ID、打包器当前位置。
- Python/NumPy/PyTorch/每GPU RNG，数据增强/合成任务RNG。
- 路由温度、专家价格、均衡器、运行统计、控制器参数与校准阈值；若这些依赖过去负载，不能只保存主模型。
- 数据/代码/tokenizer hash、配置、依赖、GPU UUID与拓扑、调用计划、训练成本计数器。
- 原子完成标识：临时文件写完并验证可加载后再切换 `latest`；不能把部分写入文件当最近可恢复状态。

### 4.2 world size 变化时的 batch 规则

例如 S1024、每卡microbatch=8：单卡累积4次得到32,768 token；两卡各累积2次得到相同全局batch；四卡各累积1次。3或5卡整齐均分32条序列不成立，因此核心阶段优先独立单卡任务；若确需3/5卡DDP，使用明确的可变microbatch或不同固定全局batch并重新设计公平对照，不可偷偷把batch改小。pilot的8,192-token逻辑batch单独按当前序列长度和GPU数分解，不能沿用主实验累积次数。

在不同rank有效token数不等时，不能平均各rank的平均loss后直接假定得到全局token平均梯度；需按本地有效token数加权，并补偿DDP默认rank平均，或采用明确的全局sum/count方案。完整数据顺序由全局样本索引决定，再分配给rank；不要仅从各rank sampler游标猜测已消费数据。

相同模型状态、相同全局batch内容和相同日程允许语义一致地恢复；不同浮点归约顺序、随机数分片和内核可能导致轨迹不逐位相同。[PyTorch 可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)明确不保证跨版本/平台完全重现，因此报告“可恢复、预算一致、统计复现”，不承诺world size变更后bitwise一致。

[PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)及 [DeepSpeed Universal Checkpointing](https://www.deepspeed.ai/tutorials/universal-checkpointing/)支持特定分片状态/并行配置的恢复，但必须做本项目实际模型、optimizer、数据游标的恢复演练，不能把通用文档当作本模型已测试证明。当前小模型可先用完整状态checkpoint避开分片复杂性。

## 5. 因果性、缓存与运行状态

必须有三类检查：

1. **同一前缀、不同未来后缀：**在推理模式且无随机性时，对共享前缀的模型输出与路由应相同（允许指定数值容差）。训练梯度依赖所有训练token是正常的；但当前token的前向router直接读到同序列未来token统计是不允许的。
2. **不同批次组成：**将同一请求放入不同batch。如果方法显式允许可同时观测请求之间竞争容量，输出改变应被记录为该算法性质；不得将这种条件化的结果当作严格请求独立模型的无代价替代。在线价格只能来自已允许观察的信息。
3. **末端分支缓存：**缓存最后attention之后的每token状态，分别执行候选FFN路径和独立LM head，与从头运行对应完整模型的该位置输出比较。只有最后块无后续跨token算子的情况下，才能以此构造精确单位置反事实标签。

推理输出不能读取真实下一个token；真实标签只用于训练/离线标注和事后评测。label-generation代码与inference代码应有不同函数签名，避免误把标签传进控制器。

## 6. 统计与评测

### 6.1 准确率/损失

主指标是相同tokenizer下的平均NLL（nats/token）；PPL作为指数变换同时报告。不同tokenizer的PPL不直接比较；可补充每字节损失作解释，但主结果仍统一tokenizer。

逐方法输出每个seed的结果和配对差值。文档内token相关，因此对测试集误差条采用**文档级**bootstrap或分块bootstrap，不能把数百万token当独立样本。文档级CI描述固定训练模型的评估不确定性，三个训练seed描述训练不确定性；二者应分开显示。三种子不足以支持很精细的总体统计结论，不能靠海量token产生极小p值掩盖种子不稳定。

建议在pilot后冻结一个主要预算点、一个主要比较和一个实用效果阈值，其他预算点构成曲线而非反复挑最优。相同终点性能下比较成本，或相同成本下比较NLL，不用“更低loss”和“更低成本”分别取不同运行点拼接。

### 6.2 真实速度

分别测：

- training：完整优化步，记录有效tokens/s、路由耗时、专家耗时、attention/LM head耗时、显存峰值、数据等待。
- prefill：固定prompt长度S=128/512/1024及batch=1/8/32，报告有效输入tokens/s和整请求耗时。
- decode：固定上下文与生成长度（如prefix512、generate128），batch=1/8/32，记录每输出token时延与总请求时延。
- dynamic batching：只有系统实验才加入，并使用相同请求流和batch规则；报告p50/p95，不把排队时间与纯GPU时间混为一谈。

每个配置预热至少20个完整步骤，再测不少于100个完整步骤，重复3次；若编译/缓存仍未稳定，延长到稳定并记录。选择预热长度是项目协议，不是通用定理。计时前后同步CUDA，排除首次编译但单独报告编译成本。不要对每个micro-kernel都同步后把其总和当正常运行时延。对存盘/评估/数据I/O另外记录端到端账目。

所有方法使用同一硬件、精度、batch、上下文、内核版本、padding和输出长度；若动态退出需要token compaction，比较“仅mask继续全部算”和“真正压紧跳过计算”，前者不能声称节约执行FLOPs。

### 6.3 能耗与成本

本项目首轮用GPU-hour与实际算量即可；若讨论能耗，读实际功率采样积分，不用300W乘时长当已测能耗。记录失败实验和反事实标签生成成本；主方法不能通过离线搜索隐性获得大量未计价算力。

## 7. 失败后的具体替代顺序

| 问题 | 优先替代 | 保留的科学问题 |
|---|---|---|
| grouped GEMM/特定MoE库安装失败 | PyTorch按专家分组reference；先小batch/单卡 | 路径是否有可学习价值 |
| BF16内核不兼容 | 核验驱动与构建；其次FP16+scaler，最后FP32小模型 | 正确性与相对收益；速度结论延期 |
| 动态DDP出现unused参数/同步问题 | 单卡独立种子；之后单独修DDP | 核心算法仍能完成 |
| 动态token compaction慢 | 固定深度桶或每序列深度决策 | 粗粒度预算分配 |
| 反事实logits内存过大 | 分位置/候选chunk，在线计算CE后丢弃logits | 标签定义保持精确 |
| 标注吞吐太低 | 从最后块、Top-1机制模型、少数状态候选开始 | 先验证oracle gap与可蒸馏性 |
| 全12层循环失稳 | 保留末1/末4层，减残差步长、限制最大轮数 | 受控递归计算，而非强推规模 |
| 自然语言预算不足以出现稳定差异 | 先报告机制诊断与失败边界，使用合成任务验证理论 | 不把合成成功宣称LM普遍改进 |

所有替代均登记为变更，不在论文中只保留最有利的一条运行轨迹。
