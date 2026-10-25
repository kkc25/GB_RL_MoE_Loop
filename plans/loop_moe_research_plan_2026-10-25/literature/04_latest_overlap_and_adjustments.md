# 近期直接近邻：新增重叠与研究边界修订

更新：2026-10-25。此补充会收窄 `02_literature_map.md` 中的初步定位；它不撤销旧文献，也不把任何未执行实验写成结果。以下“研究调整”属于本项目判断。

## 1. 最重要的修订

**不能再把“oracle 很好但部署选择器学不到”“搜索多步路径后蒸馏”“反事实增益监督停止”作为独立新颖性。**这些问题已有直接方法、诊断和负面复现。保留的研究问题必须进一步限定为：在冻结稀疏 MoE 中，改变**有序专家集合**的多轮决策，相对同预算的一步监督和简单循环结构，何时产生可部署且足以支付额外成本的增益。

这里只确认有研究价值的问题，**不宣称该限定已经证明全球首次**。如果实验只重现“上界大、学生差”，它可以是诚实的复现与边界报告，但尚不足以预设为独立机制论文。

## 2. 新增四篇直接相关论文

| 文献与核验深度 | 已覆盖的内容 | 本项目应保留的边界 |
|---|---|---|
| [Dynamic Routing as a New Dimension for Test-time Versatility of LLMs](https://arxiv.org/abs/2610.04751)，Michal Štefánik、Marek Kadlčík、Josef Kuchař、Michal Spiegel，2026-10-03；work in progress。[正文](https://arxiv.org/html/2610.04751v1) §3.4、4.3、Appendix A/D/E/G 已读 | 已区分 hindsight 与二次独立采样的 transferable oracle，分析选择偏差、选择信号与跨问题探针，并比较训练得到的层程序预测器。 | Appendix A 明确保留 MoE 层自身的 token router 参数／规则；层路径变化仍可能间接改变选中专家。本项目直接干预专家集合并研究多轮集合交互，仍是不同动作空间，但一般性的 oracle—selector gap 已非空白。 |
| [How to Loop MoE: Flatten the Experts, Untie the Attention（Foil）](https://arxiv.org/abs/2609.35751)，Shouren Wang 等，2026-09-28。[正文](https://arxiv.org/html/2609.35751v1) §2、Appendix J 已读 | 在专家参数和调用预算受控时，将层数减半、每层专家数与循环次数加倍；专家和 router 跨轮共享，attention 逐轮独立。 | 专家池宽度、循环深度、attention 共享已经是明确架构变量。末层 FFN-only 实验不等于复现 Foil；扩展 full-block 时需检查 attention 共享是否解释收益。 |
| [LatentLoop: Scaling Looped MoE Models with Asymmetric Compute Allocation](https://arxiv.org/abs/2610.19975)，Zefeng He 等，2026-10-14。[正文](https://arxiv.org/html/2610.19975v1) §2–3、Appendix B 已读 | 非对称循环 residual、首轮序列压缩、可学习写回、共享 MoE 与独立 attention；给出因果增量解码和训练时间。 | 不能将“不同轮次分配不同计算”或非对称 residual 本身作为新点。其重配置架构与本项目冻结骨干上的状态依赖专家选择不同；预测的 scaling 节省、实测训练时间和本项目 decode 延迟分别报告。 |
| [Rethinking Routing Exploration in Modern Mixture-of-Experts（MirrorMoE）](https://arxiv.org/abs/2610.22353)，Bingcong Li、Hao Ma、Michael Muehlebach、Niao He、Yi Ma，2026-10-15。[正文](https://arxiv.org/html/2610.22353v1) §2–3、Appendix B.6 已读 | 研究共享专家与细分专家组合空间之间的探索问题；镜像 logits 选专家，原评分决定聚合贡献，抑制部分专家的有效贡献。 | 其“有效专家减少”不等于实际调用减少。主问题是含 shared experts 的预训练；本项目默认无 shared expert、冻结专家的 E=8/k=2 设置，不可直接继承其结论或把不适用的移植列成完整复现。 |

### 2.1 Transferable Oracle 的正确使用

上述 Dynamic Routing 的二次采样诊断用于识别随机生成中的偶然成功，不能直接改名为本项目可部署策略的 Bayes 上界或下界。它仍然使用答案判断候选成功，并不证明可从部署可见状态识别成功候选。

一个简单的独立推导说明为何没有普遍界：若只有一个准确率为 p 的随机动作，连续两次正确率是 p²，低于其部署准确率 p；若多个动作准确率都相同且无法预先识别成功者，“至少一个动作两次都正确”的概率又可高于任一部署单选策略。因此不继承原文中未经适用条件证明的 deployment ceiling 表述。

本项目在固定权重、固定候选、关闭 dropout 的 teacher-forced CE 下通常是确定性评估。重复同一 CE 不会像生成重采样那样检验可学性。应通过文档持出、部署可见输入、候选覆盖对照、标签随机化和真实学生的 NLL 检验；有限学生失败也不证明所有可部署策略都失败。此前 `03_nearest_neighbor_and_learnability.md` 的反例是说明性论证，不应被包装成首次发现这一一般问题。

## 3. 沿引用追到的更直接方法先例

| 原始来源 | 已核验方法及边界 | 对实验的具体约束 |
|---|---|---|
| [Skip a Layer or Loop It? Learning Program-of-Layers in LLMs（PoLar）](https://arxiv.org/abs/2606.06574)，Ziyue Li、Yang Li、Tianyi Zhou，2026-06-04；[正文 §3 与 Appendix D](https://arxiv.org/html/2606.06574v1) | MCTS 产生有效层执行程序；独立冻结 embedding encoder 和可训练预测器输出分段及 skip/keep/repeat 标签。已核验[作者代码入口](https://github.com/tianyi-lab/PoLar)。 | “离线多步搜索→轻量预测器→部署无搜索”不是新范式；必须比较首动作模仿、一步成本回归、两步价值预测，而不能只比较原 Top-k。外置 encoder 的参数与时间也需计入，不能只数预测头。 |
| [Dr.LLM: Dynamic Layer Routing in LLMs](https://arxiv.org/abs/2510.12773)，Ahmed Heakl 等，2025；[作者仓库](https://github.com/parameterlab/dr-llm) | 原论文摘要与作者库已核验：冻结骨干、MCTS 监督、逐层 skip/execute/repeat router；Dynamic Routing 的 Appendix D 是其另一个实验组的复现描述，不能替代原方法。 | 以当前状态作预算感知的多步路由监督已存在；本项目差别须落实在稀疏专家集合、终局目标和等标签成本。 |
| [Programs-of-Layers in LLMs through the Lens of Cortical Areas（RE-PoLar）](https://arxiv.org/abs/2609.31360)，Justus Westerhoff 等，2026-09-25；一手摘要索引与[作者库](https://github.com/DATEXIS/RE-PoLar) 已核验，arXiv 正文直取失败 | 复现者报告程序搜索存在收益，但其 top-ranked 学习预测器退回标准路径，候选集合的存在性收益未全部转化为单次选择收益。 | 已有相近负面复现。只能表述为该复现设置中的发现，不能据此断言 PoLar 原结论普遍错误；我们的阴性路线还需 MoE 特有机制与可执行干预证据。 |
| [LatentHarness: Learning Latent Actions for Memory and Reasoning via Counterfactual Policy Distillation](https://arxiv.org/abs/2609.39740)，Xiaoqiang Wang、Suyuchen Wang、Bang Liu，2026-09-30；work in progress。[正文 §2.2、3.2](https://arxiv.org/html/2609.39740v1) 已读 | 对 THINK/RECALL/EXIT 做一步分支，用已发出 token 的 logprob 变化、整轨迹 advantage 与动作成本形成 soft teacher；KL 蒸馏与 GRPO、骨干／记忆更新共同使用。另有一步分支与完成到答案的分支对照。 | 不是纯监督冻结 MoE，也不是专家集合规划；但直接覆盖反事实动作增益、停止和分支视野比较。不能把“一步换多步”本身写成首次。其一步优于完成分支的结果也不能外推为所有 MoE 的定理。 |

PoLar 与 LatentHarness 的出处分别来自引用追溯和题名检索；本轮没有在 Dynamic Routing 正文中找到对 LatentHarness／Counterfactual Policy Distillation 的直接引用，不能写成它的参考文献已经包含后者。

## 4. 仍可检验的 MoE 具体问题

这些问题是**条件性贡献候选**，每个都需要真实实验支持。

| 可检验命题 | 所需新增证据 | 不够的结果 |
|---|---|---|
| 专家集合的顺序交互使长视野控制有价值 | 同一冻结骨干、固定轮数、同候选覆盖下，先差后优的排序翻转可由合法输入预测；学生胜过一步偏好／回归／soft-policy 蒸馏及参数匹配控制 | 已知 y 后挑出的最佳路径更好；仅展示少数成功样例 |
| 更多标签成本值得支付 | 分别画等候选终局评价、等 GPU-hours、等部署调用的质量曲线；记录 LM head 成本与收支平衡点 | 多步方法用更多搜索后胜弱 Top-k |
| 收益由特定 MoE 结构控制 | 小规模改变集合大小、专家池、循环范围或 shared-expert 配置，匹配相应参数／调用轴，并证实预测器排序或收益随机制干预改变 | 仅把不同规模模型的相关性称为因果机制 |
| 提高可部署性有明确办法 | 将候选覆盖、学生容量、循环稳定、出口训练、状态漂移分别干预，观察哪种修复能稳定恢复 NLL 收益 | 泛称 oracle 不可学；用反例证明所有真实模型不可学 |
| 阴性结论能指导资源选择 | 给出在测试范围内“何时采用一步控制、何时固定循环、何时停止做规划”的稳健边界，并有同预算强基线 | 单一 checkpoint、单一 seed 或一次训练失败 |

默认模型仅约 67M／105M／180M，结论必须限定这一训练和动作空间范围；规模较小是可执行条件，不能单独充当研究新颖性。

## 5. 对现有实验的最小调整

1. **E00/E11：**把逐轮 residual 标度作为明确控制；若扩展至 full-block 循环，再考虑 Foil-style 独立 attention。压缩序列的 LatentLoop 属于单独架构，不能为了补基线在末层原型中随意替换任务。
2. **E01：**将一般 oracle 可学性诊断写成借鉴与复核；主要新测量是稀疏集合序列的可预测顺序收益、候选覆盖及标签成本。按问题／文档持出仍是必需协议。
3. **E02/E03：**在已有同一候选表上补一个便宜的一步 soft-policy 蒸馏控制，例如以负即时成本构造 soft teacher；明确这是 CPD-inspired 受控改编，没有复现 LatentHarness 的 GRPO advantage 或记忆。不要新增大模型 RL 作为前置条件。
4. **E04/E10：**保持容量匹配、Markov 输入和同调用交互对照；“路径记忆”“有序程序”“互补”分别已有相关先例，贡献落在额外可预测性及其具体机制。
5. **阴性转向：**可以保留负面结果，但需要多 seed、同预算强基线和预登记的机制干预。若只重复已有 oracle—selector 落差，应将其写成复现／范围限制，而非自动升级成新理论主线。

## 6. 覆盖限制

此轮追加了四篇近期直接论文及四条方法追溯，已足以改变创新声明。没有尝试复现其训练结果，也未认证仓库可运行；RE-PoLar 的正文获取受限已明确标注。严禁把仍未覆盖的预印本、投稿或实现差异当成不存在。本文件的作用是防止过度主张，并把有限 GPU 预算集中到真正能够区分机制的实验。
