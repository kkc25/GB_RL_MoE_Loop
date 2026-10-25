# E00：可行性校准、基础模型与不可省略的对照

## 0. 实验身份

- 级别：所有后续路线的共同前置实验。
- 科学问题：既定资源下，能否得到训练正常、循环稳定、可精确构造末端反事实、且成本可测的 MoE？
- 主要输出：公共 checkpoint、冻结数据/配置、GPU 吞吐与显存曲线、基础深度—NLL—成本曲线、后续标注可用的缓存接口。
- 成功不等于：证明新 router 有用；E00 不承担创新结论。
- 设备：先1卡，重复实验并行使用3卡。2卡DDP仅在单卡通过后比较；最多8卡不是要求。
- 本文件是可实施的实验设计，当前未在用户GPU上执行。

## 1. 模型和变量

采用 [容量文件](../resources/01_a6000_capacity_and_budget.md)中的 S67/M105/M180。正式主模型 M105 精确参数104,919,552，词表32,768，L12,d512,E8,k2,m512，固定S1024。

核心仅最后 Transformer block 的FFN可循环：先计算该块causal attention一次，再在每个token的隐藏状态上执行共享专家FFN最多4次。前11块均R1。最后块之后只有逐token final RMSNorm与绑定词表头。

建议先固定最后块循环残差系数 \(\alpha=0.5\)，并对所有相同结构的对照保持一致；其他层常规残差。这个数是稳定性起点，需要E00验证，不能当成已证最优。若改为0.25/1.0应共用小范围pilot并将其成本记账，不只为新方法挑优。循环步编码/控制器参数以后另计，基准模型不含它们。

### 基础对照矩阵

| ID | 结构/训练 | 部署深度 | 验证目的 |
|---|---|---|---|
| B0 | 常规MoE，每层R1 | R1 | 不加递归的参照 |
| B1 | 最后块固定R4训练 | R4 | 固定最大深度参照 |
| B2 | 最后块每batch均匀抽R∈{1,2,3,4}，只在抽到终点计算LM loss | 分别R1/R2/R3/R4 | 用一个公共模型获得各深度受过监督的可比较轨迹 |
| B3 | dense计算匹配，最后FFN宽度/深度按实际FLOPs选择 | 固定 | 区分稀疏参数与计算预算收益 |
| B4 | dense参数匹配 | 固定 | 展示容量匹配，不宣称算量相等 |
| B5 | dense共享FFN循环，采用与B2相同深度训练 | R1–R4 | 区分递归与专家组合的贡献 |

B2是后续冻结主干、训练路径/停止控制器的首选公共checkpoint。**B1与B2训练目标不同**，不可把B2早退表现改善全归因于停止算法。后续方法与其固定深度参照应从同一B2权重分支。

B0/B1/B2若做科学性结果，既报告等token训练，也报告训练成本；等token下B1更多计算是事实，不能称作严格等训练FLOPs。论文主要创新对比使用公共checkpoint，使主干学习差异不干扰路由机制判断。

## 2. 数据与训练步骤

共享 [执行协议](../resources/02_reproducible_execution.md)，FineWeb-Edu固定revision，train-only32k tokenizer，固定文档级split与序列顺序。学习率pilot候选为{3e-4,6e-4}；主实验逻辑global batch为32,768有效token，pilot为8,192；每卡microbatch从1条序列校准。

### 阶段 A：正确性和微型过拟合

1. 用极小尺寸模型（例如L2,d64,E4,k1,m64）实现逐token Python reference。
2. 对同一权重、输入、专家索引比较reference与batch/分组实现：输出、最终loss、关键参数梯度。
3. 测空专家、全部token选同一专家、不同专家选择、同一专家重复调用，以及循环R1/R4。梯度检查使用FP32小尺寸；BF16只做容差一致性，不要求逐位相等。
4. 固定少量训练序列，确认loss显著下降、router与被选专家获得有限梯度。不要要求小样本训练准确率作为自然语言泛化证据。
5. 输入相同前缀而替换未来后缀，确认前缀位置输出/路由不变；固定随机性与价格状态。
6. 缓存最后attention输出，执行一条强制专家路径，与完整前向中同路径该位置输出比较。容差先由纯reference的舍入误差标定，而不是为了通过测试任意调大。

这些是必要正确性检查，不构成大规模训练结果。任何一项失败先修实现，不进入创新路线。

### 阶段 B：资源校准

1. 运行 `resources/code/probe_hardware.py`，保存真实GPU型号、驱动、BF16支持、拓扑。
2. S67、M105分别以S512/S1024测last1 R1/R4；microbatch1开始，逐级2/4/8/16。
3. 预热至少20个完整步骤，测100个完整优化步骤、重复3次；记录有效token/s、峰值allocated/reserved显存、总GPU占用、router/FFN/head时间、数据等待。
4. `mask-only`实现与真正token compaction分开测；masked token仍执行矩阵乘法时不能用低平均轮数冒充省算。
5. 在已稳定配置下测1卡与2卡DDP；把两卡总吞吐除以两卡单卡总吞吐，得到并行效率。固定有效global batch，不为加速偷偷改变优化问题。
6. 跑存盘—重启—继续演练，比较恢复前后token游标、optimizer step、loss趋势和路由价格状态；如计划world-size变更，再独立验证1↔2卡恢复。

**规划门槛：**核心M105至少一种S1024配置在单卡可稳定运行，峰值留约15%余量。若不足，依次减microbatch、增加gradient accumulation、chunk logits/CE、checkpoint；不是直接等待8卡。

### 阶段 C：模型筛选与公共checkpoint

1. S67累计50M token确认训练曲线正常，其中前5M–20M为smoke检查，不额外重复计账；这是代码与训练信号检查，不承担最终性能结论。
2. M105的B0/B1/B2先各一个seed累计跑300M token，其中前100M为pilot，不额外重复计账；固定公共学习率pilot选择、相同数据前缀与基本优化器配置。若pilot后的配置或初始化变化要求重跑，放弃运行的成本另列，不能将所有候选pilot视为免费前缀。
3. 在validation的固定1M前缀计算各深度NLL；正式候选使用完整10M validation。
4. B2通过后保存三个初始化seed的公共checkpoint；核心方法在各seed权重上分别分支。
5. 确认实验升到1B token/seed；并将B3/B5计算匹配基线加入最终预算图。B4是容量对照，预算不足时可列为后续，不抢占核心三种子资源。
6. 不要求B2的每个token都随深度单调改善；真实现象可能是部分token过度计算变差。记录个体分布以及平均曲线。

## 3. 必须导出的诊断

每个模型/seed/深度记录：

- validation NLL、PPL与每文档损失；训练loss曲线、梯度norm、溢出/NaN次数。
- 每专家被选次数、负载均值/最大值/变异系数、未使用专家比例；router熵、Top1–Top2 margin。
- 隐藏状态范数、循环增量范数、相邻轮cosine、同一专家重复率、路径频率。
- 理论执行次数、真实执行的expert-token pairs、padding浪费、额外词表头次数。
- 训练tokens/s、prefill/decode速度、controller/routing占比和峰值内存。
- 累计训练FLOPs粗估、profiler覆盖率、GPU-hour；未覆盖算子明确标注。

如果router始终选同样专家，不先称“专家专业化”；检查表示/梯度/负载和数据相关性。均匀分配也不等于有用分工。

## 4. 基线训练与成本公平性

### 4.1 三种比较轴同时保存

1. 同参数/同主干checkpoint比较：适合判断控制器的因果增益。
2. 同推理FLOPs或同实际时延比较：适合判断效率。
3. 同总训练预算比较：包含标注、搜索、蒸馏、失败超参和controller训练，判断收益是否只是额外算力。

“所有指标完全相同”一般做不到，必须选择主比较轴并透明列出其余指标。Dense参数匹配通常不算力匹配；多轮共享MoE虽然参数不变但算量上升。

### 4.2 早退简单基线

后续停止实验至少继承：固定R1/2/3/4、按validation选择的固定深度、相同平均深度的随机停止、隐藏状态增量阈值、预测熵阈值。熵基线每轮计算完整词表头的开销须计入；不能让它免费读出再与只读隐藏状态的控制器比较。

随机停止的分布应在validation上匹配平均执行成本，不使用test标签。若采用随机深度但不做真实压紧，仅能作为loss–算术诊断，不能代表部署速度。

## 5. 何时进入后续路线

| 发现 | 决策 | 说明 |
|---|---|---|
| 正确性、因果性、训练和恢复均通过 | 进入E01/E02的oracle与可学习性检查 | 此时才值得研究复杂router |
| R4整体略优但部分tokenR1更好 | 优先停止/预算分配路线 | 深度异质性提供可检验机会 |
| R1–R4几乎无差异 | 先检查模型训练、残差缩放、任务和专家容量；仍无差异则降低停止路线优先级 | 不用复杂博弈掩盖没有可分配的价值 |
| 强制替代路径没有oracle改善 | 保留资源价格/系统优化作为另一问题；停止长期路径价值主张 | 无oracle gap时蒸馏不应凭空产生收益 |
| 路径有oracle改善但正常状态下控制器不可预测 | 考虑更强状态摘要或短期搜索教师；仍无效则报告不可蒸馏边界 | 不将label-dependent oracle当部署性能 |
| 少算理论有收益、实测变慢 | 先采用固定深度桶/每序列控制；必要时把系统效率结论限定为FLOPs | 不伪称真实加速 |

停止一条研究假设不代表停止整个项目：机制、系统、理论三条结果可分开形成科学结论，但必须保留相应证据边界。

## 6. 资源账目与产物

E00初始训练量：S67累计50M（含5M–20M smoke），加M105 B0/B1/B2各累计300M（每条含100M pilot）共900M，合计950M token。该数假设所计pilot沿同一运行继续；被放弃的超参试验另列。后续确认按已有合格checkpoint的增量计账，不预先把全部可选基线一次训满。校准吞吐后按容量文件的公式换算GPU-hour；各模型吞吐不同，不能将一个q乘所有配置。

应产出的文件（这是未来运行的规定输出，不表示现已生成）：

```text
hardware_inventory.json
resolved_environment.txt
data_manifest.json
tokenizer.json
base_config.yaml
throughput_profile.jsonl
baseline_runs.csv
validation_depth_metrics.jsonl
checkpoint_<seed>_<tokens>/
causality_and_cache_checks.json
cost_ledger.csv
```

最重要的预注册结果图：同一B2模型的NLL–深度曲线、NLL–真实FLOPs曲线、NLL–时延曲线，以及R4→R1最大可节省比例。最后块M105在指定S1024解析模型下最多节约约7.98%前向主算术，因此此阶段以机制验证为主要目标；末4/全12块扩展才检验更大的总体效率收益。

## 7. 文献与工具依据

- [NVIDIA RTX A6000 数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20(1).pdf)：显存、互联与稀疏峰值脚注。
- [FineWeb-Edu 数据卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)：来源、采样配置和许可声明。
- [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)：数据并行语义与unused-parameter条件。
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：attention实现和backend限制。
- [PyTorch 可复现性](https://docs.pytorch.org/docs/stable/notes/randomness.html)：数值复现边界。

本实验的模型规模、预算、阈值和调度是本次研究设计，不是上述来源已经证明的推荐最优配置。
