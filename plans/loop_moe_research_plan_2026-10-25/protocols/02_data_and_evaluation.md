# 数据、任务与评测设计

## 1. 先用小模型能产生有效信号的任务

核心主指标为持出自然文本NLL及等预算路径收益。算法合成任务用于检验延迟收益、组合长度、状态记忆与预算约束。不要把AIME、复杂代码生成或大模型长CoT作为105M原型的首要成功条件；若所有方法几乎零分，这种任务不能有效区分路由机制。

| 用途 | 数据/任务 | 计划规模 | 说明 |
|---|---|---|---|
| Debug | TinyStories固定版本，或主语料小样本 | 累计≤50M tokens | 只做管线与机制；不作为广域结论 |
| 自然文本预训练 | FineWeb-Edu的固定sample/shards | 300M筛选，最多1B/seed确认 | tokenizer重计有效tokens |
| 主同分布评测 | 主来源未训练文档 | 至少1M tokens pilot，5M–10M确认 | 固定文档ID |
| 来源变化评测 | WikiText-103-raw-v1正式test | 全部锁定test，按文章处理 | 去重/污染审查后才称来源外测试 |
| 机制任务 | 有限状态机、函数组合、拷贝/反转、计数/进位 | 每族100k train、10k val、10k test起点 | 自有生成器，模板族/长度分开 |
| 下游次指标 | harness中的简单语言/常识选择题 | pilot有信号后选2–3任务 | 固定commit/task/version，避免大规模刷榜 |

表中数量是研究设计，不是来源作者推荐配置。

## 2. 一手来源与版本

**FineWeb-Edu。**作者数据卡：[HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)，数据论文：[The FineWeb Datasets](https://arxiv.org/abs/2406.17557)。选定revision与`sample-10BT`或确定shards清单，按需要下载/流式处理，不默认拉取全库。卡片ODC-BY及来源信息写进data manifest。原token_count以其tokenizer为口径，本项目重新编码后计数。

**TinyStories。**原论文：[Eldan & Li, 2023](https://arxiv.org/abs/2305.07759)，[作者数据卡](https://huggingface.co/datasets/roneneldan/TinyStories)。原版与V2-GPT4版不能混用。官方train/validation之外，要在train中另留不调参test；其简化语言分布适合小模型调试，不代表广泛推理能力。记录卡片CDLA-Sharing-1.0及下载日期。

**WikiText。**原论文：[Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843)，[Salesforce数据卡](https://huggingface.co/datasets/Salesforce/wikitext)。显式指定`wikitext-103-raw-v1`。原103M是词级统计，不是本项目32k tokenizer的token数。保留来源/数据卡列明的CC-BY-SA/GFDL信息。

**可选SmolLM-Corpus。**[作者数据卡](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)。其中有FineWeb-Edu相关子集，未经文档去重不能作为主FineWeb训练的独立来源外验证。使用公开SmolLM2 tokenizer会改变词表规模；本计划固定自训32768，不直接套其词表参数预算。

这些信息用于准确记录数据来源与复现条件，不替代机构对数据使用的既有安排。包内不再分发这些数据原文。

## 3. 推荐六分区，避免控制器“偷用测试集”

先做文档规范化与近重复分组，分组ID哈希后分区，最后才切片。推荐在所取主语料上用10000个hash buckets：

| bucket区间 | 比例 | 用途 |
|---|---:|---|
| 0000–8999 | 90% | backbone_train，仅这里训练tokenizer/骨干 |
| 9000–9499 | 5% | controller_train，构造反事实监督 |
| 9500–9699 | 2% | controller_validation，其中一半单作calibration |
| 9700–9799 | 1% | backbone_validation，选骨干与稳定超参 |
| 9800–9899 | 1% | controller_test，锁定路由/停止确认 |
| 9900–9999 | 1% | lm_test，锁定整网NLL确认 |

该细分是正式协议；资源文档的外层train/validation/test比例只作管线示例，以这里的manifest为准。不得从LM validation/test中抽文档训练controller。测试集可以供预登记方法共同评测，但不得用其结果选择阈值/超参后仍声称独立确认。

同一文档的所有prefix、路径、counterfactual、候选行必须在同一分区。近重复文档也应同组；若发现跨组近重复，按固定规则移除训练侧并记录，而非挑对结果有利的处理。

## 4. Tokenizer、packing与计数

从backbone_train抽固定文本训练byte-level BPE；词表总大小含特殊符号恰为32768，保存tokenizer模型和hash。若实际大小不同，资源脚本重算参数与head成本。所有方法共用同一tokenizer。

S1024为主。跨文档packing须明确EOS和attention语义：优先block-diagonal attention隔离不同文档，或将跨文档连续训练作为统一配置。若后者，报告不能把随机文档拼接边界当成自然上下文。bootstrap仍按源文档汇总。

每个token只统计一次监督；stride滚动评估时避免重复loss，左侧上下文token不重复记分。padding、被mask标签、BOS不记入有效训练tokens与NLL分母。

训练数据顺序由全局document/segment manifest固定，不随GPU卡数改变。每个逻辑optimizer update记录token区间；重启从该区间之后继续，禁止重复/跳过还不记录。

## 5. 合成任务的可实现规范

### 5.1 算法任务组

生成字母表大小8/16的字符串，输入长度训练8–32，验证同范围，长度外测试48/64。任务可为reverse、rotate、stable filter、模q累计和、有限状态机接受。每个任务明确唯一输出并有CPU精确checker。

### 5.2 组合任务组

从4种明确操作中抽长度1–3的程序；预先留出若干有序二元组合，训练其单操作与其他组合；测试留出组合及长度4–6。按程序模板族分区，避免只换输入值就声称未见程序。

模型的expert并不预先对应操作。输出任务正确率/长度泛化才是评价，专家路径解释只作后续关联分析。为研究纯机制，可另设人工已知专家环境，但与学习得到的LM专家分开。

### 5.3 延迟收益组

使用E08的有限状态环境，让有些最优路径需要先走即时loss较差的动作；再加入即时贪心已经最优的负对照和所有动作等价的负对照。生成器参数、分布与完整oracle记录，防止只挑有利任务。

## 6. NLL、PPL与harness不能混算

\[
\operatorname{NLL}=\frac{\sum_{i\in\mathcal T}-\log p(y_i\mid x_{\le i})}{|\mathcal T|},\quad
\operatorname{PPL}_{token}=\exp(\operatorname{NLL}).
\]

这是固定tokenizer下的token PPL。不同tokenizer之间不能直接比较这个数；需要byte/word口径并说明预处理。

[LM Evaluation Harness官方库](https://github.com/EleutherAI/lm-evaluation-harness)及[模型接口](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md)可包装自定义模型。当前[WikiText任务YAML](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/wikitext.yaml)指向`wikitext-2-raw-v1`，报告word/byte perplexity与bits_per_byte；**`--tasks wikitext`并不自动评估WikiText-103，也不是本项目token PPL。** 使用103需自定义明确task/评测脚本。

WikiText-2和WikiText-103的验证、测试文章相同，不能当作两个独立复现。HF的一行也不等于一个文档；按文章边界重建再bootstrap。

## 7. 污染与来源变化

FineWeb可能含Wikipedia或转载内容。若WikiText用作来源外测试，先排除已识别Wikipedia来源并对测试文章做文本近重复检查；保存匹配阈值与排除量。如果无法排除重叠，只称“外部评测集”，不声称无污染OOD。

控制器oracle训练不使用测试答案；合成生成器的测试seed、模板族、长度区间冻结。对自然文本重复调优的validation效果只用于选择，确认test按预登记一次生成主要表。

## 8. 报告最小集合

每个方法：三个骨干seed的NLL/PPL、总体与分来源指标、预算点、文档级配对区间、实际calls、FLOPs、head次数、GPU小时、吞吐/延迟。另给未成功的任务/预算点，不以剔除困难集来提升均值。

保持结果空表直到真正运行；本方案包中的CPU玩具复算不能填入语言模型性能表。
