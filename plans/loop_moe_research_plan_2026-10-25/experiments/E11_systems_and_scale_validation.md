# E11：扩大循环范围与真实延迟验证

**若要声称高效架构，这是必要确认；常用3–5张A6000足够，不把8张作为前置条件。**

## 研究问题

E02/E03在末层证实的质量或调用节省，能否在更大计算占比下保持，并转化为实际吞吐/延迟收益？最后层反事实结论不能自动推广到全层。

[LOOM](https://arxiv.org/abs/2610.01153) 已研究循环稳定性，[Sparse Layers are Critical to Scaling Looped Language Models](https://arxiv.org/abs/2605.09165) 讨论循环与早退；需要区分其理论FLOPs结果与本实验真正测量的系统性能。

最新结构近邻还包括[Foil](https://arxiv.org/abs/2609.35751)与[LatentLoop](https://arxiv.org/abs/2610.19975)。前者扩大专家池并解耦逐轮attention，后者做逐次非对称计算。它们与本方案末层FFN-only不同；若进入完整block循环，要加入相应机制对照，不能把未覆盖的attention设计差异归给Router。首轮无需复现其大规模训练。

## 范围逐级升级

| 阶段 | 模型与循环块 | 目的 |
|---|---|---|
| S0 | 105M，最后1块 | 复现主机制 |
| S1 | 105M，最后4块 | 提高可节省计算占比 |
| S2 | 180M，最后4块 | 一次规模确认 |
| S3 | 105/180M，全12块 | 条件架构扩展 |
| S4 | 0.5B或更大 | 仅S1/S2已稳定且剩余资源足够 |

不默认做S4；7B从头训练不是本项目完成条件。每个阶段只保留“最强基线＋主方法”及必要一个停止对照，三seeds优先于堆更多弱基线。

## 重要改变：中间层反事实不再局部

若在倒数第4层改变token i，后续attention可让其影响j≥i。精确标签应重新执行受影响后缀、使用正确因果KV；不能只替换最终token的局部FFN输出。

标签先明确目标：若测局部next-token路由，只比较该token CE但重跑到其最终输出；若声称完整序列价值，则使用ΔL=Σ_{j≥i}(CE′_j−CE_j)，在固定受影响后缀长度上归一化，并将所有受影响位置纳入重算。两个目标分别报告，不能沿用末层局部标签却称完整序列改进。

可行选择按优先级：

1. **冻结控制器泛化诊断：**先将末层方法按相同接口移到末4层，真实完整前向评估；这检查转移，不称新层最优。
2. **稀疏采样层与位置：**训练标注只选一个layer/token干预，缓存前缀后重算受影响后缀；标注成本重新登记。
3. **块级动作/停止：**同请求一个短块共享深度，可降低动态调度，但深度只能由块开始时已经可见的prefix决定；不能用块内所有未来位置的features。定义变了，要比较质量差异。
4. **局部代理loss：**作为廉价近似，必须与少量精确重算标签测相关性与选择regret，不能当作真实终局loss。

若标注成本超门槛，优先采用E03固定路由停止或直接可微训练末4块；不强求把末层oracle流程照搬全网。

## 执行实现与参考路径

- 正确性参考：按token/专家循环实现，功能明确但可能慢，只作数值对照。
- 工程版本：按专家分组的稀疏dispatch、gather/scatter，动态退出时压缩活跃token；统计实际执行，不能用dense mask伪装少算。
- 分桶版本：按预估深度把token分组，代价是预先决策误差与组内padding。
- 若torch.compile/CUDA graphs对动态shape不友好，采用固定bucket与mask，但只在真正不执行已停专家时记节省。编译时间单列不并入steady-state，可另报首次使用时延。
- 所有8专家默认驻留每张GPU。只有规模/负载必要才启用expert parallel，且与同样模型的data parallel分别计。

代码实现可参考官方 [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 与 [FlashAttention](https://github.com/Dao-AILab/flash-attention)，但先检查A6000实际驱动、dtype和内核支持；不要为了使用某个新内核更换硬件假设。

## 实测矩阵

| 场景 | batch/request数 | 长度 | 主要指标 |
|---|---|---|---|
| teacher-forcing评估 | 1/4/8 | 512/1024/2048 | NLL、tokens/s、峰值显存 |
| prefill | 1/4/16 | 128/512/2048 | TTFT、prompt吞吐 |
| decode | 1/8/32/64 | 固定prefix512、生成128 | 每步p50/p95、tokens/s |
| 混合服务trace | 预定到达率梯度 | 长短请求混合 | 排队、goodput、tail latency |

prefill通常不对每个prompt位置都读完整词表；训练/likelihood评估则需要逐位置head。资源文件中的全位置head FLOPs表不直接作为所有serving场景的上限。

固定generation长度、EOS规则、采样参数与同一输入；预热至少30步，正式测量至少200步或30秒，重复5次。计时前后CUDA同步。测性能时隔离其他GPU任务，记录功率/温度/显存竞争。无法隔离则标注共享负载并给出较大波动，不报精确到小数点的普遍加速比。

## 账本与Amdahl分析

拆成 attention、专家matmul、dispatch、控制器、LM head、通信、host/kernel overhead。记录加速前后真实时间占比f；即使循环部分完全消除，理论总加速也不超过1/(1−f)。

在统一105M、S1024、逐位置单次head的解析计算下，R4→R1最后1/4/12块的最大算术节省分别约7.98%/25.74%/50.98%。这些不是实测速度，也不包括stop额外head/controller；真实分桶padding与小batch可能使节省更低。

## GPU调度与资源门槛

3卡：每seed单卡；4卡：2+1+1或4个独立单卡；5卡：2+2+1；8卡：并行确认或有拓扑证据时4卡一组。RTX A6000 NVLink为两卡桥接，不假定8卡NVSwitch互联或384GB统一显存。

S1先1seed300M tokens筛查，成功再三seed累计1B；S2只选两个方法三seed，tokens上限由实测q与剩余GPU-hour决定。S3/S4各自需要新的go/no-go，不预先全排满。

每次改变world size仅在完整optimizer step边界恢复。验证相同逻辑batch与数据游标；无需声称不同collective顺序下位级一致。

## 成功标准与转向

- 同质量约束下总体速度提升目标≥5%，含所有控制与调度成本，置信区间不跨0；或同延迟下NLL明确改善。
- 若NLL/FLOPs改善但wall clock不改善，结果归为算法质量—计算前沿，明确系统限制。
- 若只大batch受益，给出适用batch阈值，不泛称低延迟加速。
- 若末4层机制失效，研究层位置依赖、后续attention耦合与标签代理误差；不要把末层结果包装成全网路由定理。

输出：端到端可重复benchmark命令配置（待实现训练器补齐）、分阶段profile、latency分布、质量—时间曲线、拓扑与规模限制表。
