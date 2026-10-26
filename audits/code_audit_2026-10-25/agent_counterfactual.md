# 反事实、停止、互补与报价代码审查

审查提交：`f2bef422f7315f30618f00c690bf82810d1e85ac`。仓库只读；未使用未上传的 checkpoint；未重训任何 LM。以下区分确定代码错误、实验设计偏离和证据不足。

## 关键结论

E01 的默认全 28 动作、冻结基准续算、标签知情 oracle 单列等关键逻辑基本落实。当前阴性结论的主要缺陷是样本来自极少文档，无法据此推出“状态不含可用信息”或“路径价值不可学习”。E03 的离线停止和 E11 的在线停止输入契约发生确定性错误；E12 分专家 adapter 的 einsum 实现不符合独立专家计算。此外 E10 的 32/16/16 token 训练验证测试几乎只是流水线 smoke。

## C01 — P1，已执行路径：停止学生训练/校准用 raw h，部署用 RMSNorm(h)

- **证据**：`train/value_student.py:35–37` 的 `student_features` 返回原始 pre-FFN `h`；`experiments/e03_halting.py:165–173` 使用同一 raw `h`；`eval/dynamic_halting.py:26–31` 却将 `block.ffn_norm(h)` 送入学生，并继续用 raw-h 统计的 mean/std 标准化。
- **数学影响**：训练策略为 `f((h-mu)/sigma)`，部署变成 `f((RMSNorm(h)-mu)/sigma)`。非线性网络和阈值都没有这种不变性，故 E11 并未执行 E03 冻结阈值所校准的同一个策略。
- **最小反例**：h=(2,0)，RMSNorm权重全1，mean=0，std=1，gain预测器读第一坐标，tau=1.7。训练/校准输入得到 gain=2→继续，部署输入约1.414→停止。NumPy精确表达式复现见 `repro_counterfactual_numpy.py`。
- **为什么原测试未发现**：`tests/test_dynamic_halting.py:18–28` 将全部学生权重归零，并仅测试±∞阈值。此时学生任何输入都不影响结果，所以只能检验执行器两端一致性，不能检验部署决策一致性。
- **最小修复**：router仍用normalized状态；学生改用 `flat_h=h.reshape(-1,h.shape[-1])`。增加同一批缓存状态、相同dtype、有限tau的离线/在线delta、decision mask、rounds和CE对照。
- **结论界限**：该错误足以使“E03策略已正确接入真实前向”的结论失效。没有实际权重不能量化它对论文报告 NLL/深度的贡献；不能宣称全部数值偏差由它解释。

## C02 — P1，已执行实验：E01 coverage_remediation没有真正修复文档覆盖

代码 `eval/counterfactual.py:133–151` 随机访问序列，但每次提取该序列全部监督token；到达总数后直接 `[:n_states]` 截断。它没有对文档或状态作均衡抽样。公开 JSONL 给出：

| 运行 | split | token状态数 | 文档数 | 文档内状态数 | 序列状态数 |
|---|---|---:|---:|---|---|
| E01 pilot | train | 1024 | 2 | 1023 / 1 | 1023 / 1 |
| E01 pilot | validation | 256 | 1 | 256 | 256 |
| E01 pilot | test | 256 | 1 | 256 | 256 |
| coverage_remediation | train | 8192 | 15 | 多个文档各约1023，最大1023 | 9序列 |
| coverage_remediation | validation | 1024 | 2 | 1023 / 1 | 1023 / 1 |
| coverage_remediation | test | 1024 | 3 | 906 / 116 / 2 | 1022 / 2 |
| m105_100m | train | 8192 | 15 | 多个文档各约1023，最大1023 | 9序列 |
| m105_100m | validation | 1024 | 2 | 1023 / 1 | 1023 / 1 |
| m105_100m | test | 1024 | 3 | 906 / 116 / 2 | 1022 / 2 |

这是扩充 token 数而非扩充文档覆盖。m105与coverage_remediation的validation/test还使用完全相同文档与位置；两个骨干阶段并非两次独立数据复核。

- **统计错误**：`evaluate_policies:264` / `oracle_metrics:279` 及E03 `mean_se:176–181` 都以token数开平方给SE，未作预定的paired文档bootstrap；相邻token一般相关。3文档也不足以可信估计域级不确定性。
- **方案差异**：E01-E要求100k/20k/20k状态起点、文档级拆分、线性/小MLP/同规模MLP对照、三骨干seed确认。当前E01两种目标均用同一MLP类，seed=0的设置没有线性及不同容量恢复曲线。少量pilot可以暂停花费，但应标“未获得充分可学性证据”，而非证明“可观测状态不足”。
- **修复建议**：每文档最多16或32个随机合法位置；先覆盖至少数百/数千独立文档，报告doc数、token数、最大文档权重；分开controller_train/selection/calibration/test；对子任务以paired document bootstrap并列骨干seed差异。将最终test冻结，当前已多次分析的test仅作为开发诊断。
- **当前无权重可做**：已从JSONL核实文档覆盖；收集新隐藏状态需要checkpoint。仅上传现有states不能补足文档代表性。

## C03 — P1，研究解释：E03停止“成功”没有排除固定R1这一更便宜解释

E03 m105在同一1024状态上的公开标签：固定T1−T4=0.0026016 nat，已小于0.005容忍。预算2动态为平均2.37695轮、增量0.00099853 nat。原方案E03明确规定“R1已与R4同质量：优先固定R1，停止器没有存在必要”。

这并不说明动态值函数完全无用，仍需在相同调用预算下比较固定深度/无条件随机混合和直接delta回归。当前仅能说“在报告容忍下删掉大部分循环可行”，不能说已证实按状态调计算有净价值。报告的E11验证表更是R1−R4约0.00058，全部只跑R1同样满足该阈值。

- `experiments/e03_halting.py:362–365` 仅检查均值≤容忍；该flag不能自行升级为统计非劣保证。原方案没有固定单侧/双侧置信判据，不能事后换用某个倍数的SE宣布通过或失败。当前仅3文档，所报token-IID SE也不满足预定paired文档推断；应先扩充独立文档并预先固定推断方式。
- 预算是validation目标，不是test硬约束：target1.5实际1.71191；target2实际2.37695。代码如实报实际值，这本身不是造假，但不能声称这些配置在部署满足1.5/2预算。
- 两个全成本MLP相减并非方案要求的小delta头；公开分箱最左预测−0.203而实际−0.00386，最右预测+0.205而实际+0.00425，不能把阈值搜索等同于价值已校准。
- 同一validation既用于E01 best-MSE checkpoint selection，又用于400点阈值搜索（`run_e01.py:149–154`，E03 `337–358`），未实现独立calibration split；不构成test标签直接泄漏，但增加阈值过拟合风险。

## C04 — P1，已执行路径：E12的分专家首层错误地对所有专家输入求和

`experiments/e12_bidding.py:149`：

```python
torch.einsum("...ei,oih->...oh", rows, self.w1)
```

输入专家轴`e`不在输出里，也不与权重专家轴`o`对应，因此计算的是 `W_o * sum_e rows_e`。每专家独立的正确表达式应为 `...oi,oih->...oh`。

- 无summary时每行相同，输入被乘E=8；这可通过改变首层权重重参数化，不意味着函数完全不能训练，但它改变初始化/优化尺度，不能按独立adapter公式理解运行结果。
- 有summary时所有专家的summary先被求和。公开实际run使用 `summary_mode=learned`，因此已执行问题是learned summary只以总和生效，无法提供每专家独有的摘要；并非断言该run摘要数值必为0。另一个尚未执行的 `checkpoint` 模式按专家维零均值标准化，故其摘要部分理论上相消为0（浮点误差除外），已用小型反例验证该接口错误。
- **确定反例**：专家输入[2,7]、各自首层系数[3,5]，现有输出[27,45]，应为[6,35]。只改变专家1输入会使专家0输出也改变，违反实现注释的专家独立性。
- **补充API错误**：类将计算函数命名 `_apply` 覆盖 `torch.nn.Module._apply`，因此 `.to()`/`.cuda()`/`.double()`会向它传函数而非Tensor。现有E12主运行在CPU且没有调用to，不应将其说成既有run崩溃原因；它是确定的可移植性bug。修复时重命名为 `_forward_rows`。
- 原测试只检查输入维度、输出shape、动作合法性、结果有限/重复运行一致，没有逐专家手工reference与独立性验证（`tests/test_e12_bidding.py:91–113`）。

## C05 — P2，已执行对照：E12候选空间、优化设定未匹配

- 文档声称“同候选”，但中央MLP选28动作全局argmin（`e12_bidding.py:514`），报价只能选给定anchor的7对（`328–340`）。同cell MSE对比确实落实（`476–491`），但它不能让部署NLL比较自动变成同候选。
- 修复应为每个anchor规则同时计算central-restricted argmin与bid argmin，另列central unrestricted上界/设计变体；若用learned anchor，中央对照也承担完全相同anchor规则/成本。
- 中央 `fit_value_student` 默认weight_decay=0，报价/anchor直接AdamW不传weight_decay，使用默认0.01；不是完全相同优化设置。需要明确并固定这个超参数。
- 报告的adapter参数差已如实写出，包括anchor额外4104参数，这点不应错误指责为隐藏参数。FLOPs/MACs公式还把单Linear anchor当三层网络，并按7个adapter计，而实际一次计算8个后遮蔽；只宜作理论估计，与实测开销分开。

## C06 — P1，已执行实验：E10只有单文档32/16/16 token，不能形成排除路线证据

公开records：train32状态全是doc219/seq181；validation16全是doc187008/seq3306；test16全是doc120588/seq2086。143640参数低秩头与143584参数MLP训练20epoch，每epoch一batch，即每模型20个optimizer steps，未形成可靠学习能力比较。

两轮28²的执行与baseline第二步选择、相同总4轮续算、固定anchor交互公式、顺序打乱、oracle分列等代码设计基本正确；其结果可作为“精确成本矩阵生成管线已跑通”。低秩头负收益不等于互补性不可学；需先在多文档样本和足够优化下证明训练拟合、看样本量学习曲线，或保留为低优先级未决分支。

## C07 — P2：E10 sign_flip把零也当成负，而且没有数值零带

`e10_ordered.py:237–238` 用 `positive.any & (~positive).any`。同一路径在两个anchor下Gamma=[1,0]也被叫sign flip，但并未出现负值。anchor行列按定义恰好为0；浮点相消可能给±1e-7。应使用 `(Gamma>eps).any & (Gamma<−eps).any`，并报告zero/uncertain状态。

公开pair_q只保留6位小数，无法精确恢复原0.5；对此量化表用1e-6零带重算严格正负翻转约0.41837，显示指标对零处理敏感。这个数不能冒充修复后全精度实验结果；应从states_test.pt重新计算。可复算脚本见`recheck_e10_records.py`，输出`e10_records_recheck.json`同时保存覆盖与源文件SHA256。

## C08 — P2，扩展接口潜在错误：自定义候选actions未真正执行

`eval/counterfactual.py:199–203` 接受自定义pairs表，但 `217` 总是构造 `arange(n_actions)`，`222` 将其解释为全局Top2动作ID，而非所给pairs。传入[(0,1),(6,7)]会实际执行全表的第0/1动作，即[(0,1),(0,2)]，却用输入actions作元数据返回。baseline_action仍是全局ID，还可能超出列数。

当前E01默认全部28合法动作按固定顺序执行，因此该问题**不解释当前结果**。它会阻碍原方案4候选节省标注的扩展。修复应明确global action IDs、local column IDs与pairs的映射并检验基准动作包含。

## C09 — P2， provenance和边界保护不足

- `run_e01.py:98` 只保存LM配置，不保存实际CLI：state数量虽在summary，hidden_dim、lr、epochs、dtype、state/depth/chunk、seed等无完整统一记录。student权重文件有hidden_dim/history但无骨干hash。
- E03 `load_artifacts`验证三份states骨干hash，不验证students与states同hash；E11 `profile_dynamic.py:75–83`不验证checkpoint/E01学生/E03tau三者对应关系。没有证据表明当前混错，但错用时静默接受。
- `last_adaptive_layer`只验证Top2 executor，不验证该层必须是模型最后一层；若未来配置使adaptive最后层早于模型末层，cache/readout会错误跳过后续blocks。现有末层配置满足要求，所以仅为扩展保护。
- 计算账本名为candidate_pair_calls，确实是Top2集合调用次数；要转换单专家calls需×2。当前代码的ledger本身自洽，不应因命名混淆误判为漏算一半。

## C10 — P1，协议偏离：没有独立的controller探针训练与calibration分区

`run_lm.py:41–42`骨干读同一data_dir的train/validation；`run_e01.py:123–129`再读同名train/validation/test。FineWeb只生成这三份（`data/fineweb.py:172,572–575`），没有落实原方案六分区。

m105 manifest里train有97657序列；训练summary有97657 optimizer updates、batch_size=1；按`train/lm.py:87–90`每epoch遍历整个排列，E01训练状态来自已被骨干更新过的同一100M训练池。这不等于test标签泄漏，但与方案“未参与骨干训练的探针文档”不同，可能造成controller训练与部署状态/损失分布差异。

需要保持准确：骨干验证只检查前16序列，而E01 validation实际来自序列3306/2322。共享validation池本身不能证明这些具体E01验证token已经被骨干选点使用；没有证据说test进训练。独立calibration缺失则是确定的：E01选学生与E03选tau使用同一1024状态。

## C11 — 探索性诊断：Q标签方差几乎全来自共同token难度

仅使用上传JSONL、无需任何权重的重新计算：

| m105分区 | Var(每状态28动作平均Q) | E[动作中心化Q²] | 前者/后者 |
|---|---:|---:|---:|
| train | 13.937472 | 0.0220776 | 631.29 |
| test | 13.990548 | 0.0180868 | 773.52 |

在test上共同token难度占所有Q标签总方差的99.8709%。这不是已训练学生残差MSE或梯度分量的占比；具体误差分解需要学生预测。对全成本MSE，共同难度预测改善有可能压过重要的动作排序误差。这个分解不是不可学习性证明；它提示加入保持argmin不变的 `Q(s,a)-mean_a Q(s,a)` 或 `Q(s,a)-Q(s,a0)` 监督与决策regret/排序指标，而非仅按绝对MSE选点。

为核查学生是否至少超过低成本基线，预先定义两个规则，只从**train标签**拟合，test未参与选动作：

1. 所有状态使用train mean Q最低的固定动作（m105为action20）。
2. 输入仅为冻结原router的baseline_action；每个baseline_action组选择train内mean Q最低动作；train未出现组保留原动作。

| m105 held-out test规则 | 相对原router平均增益（nat/token） |
|---|---:|
| train固定动作20 | −0.00004936 |
| train按原router动作分组 | −0.00015579 |
| 已报告即时成本MLP | −0.00384269 |
| 已报告长期成本MLP | −0.00677045 |

这两个简单规则仍无正收益，因此不能宣传新路线有效；但它们在这3文档test上几乎保持原router表现，而MLP恶化更多。这进一步表明当前结果不足以把“某个学生未学好”归因为“状态没有信息”。动作中心化只保证同一轮同一候选集的路由argmin不变，**不能把immediate与Q_pi0分别中心化后再相减作停止分数**：那会删掉真正stop-vs-continue共同偏移。停止应直接监督配对delta，或保留并正确恢复每个目标的offset。公开CE保留6位小数，差值每状态最多有1e-6量化误差；上述量级与精度已分开说明。完整映射、每文档结果和三套run保存在`records_only_counterfactual_audit.json`，脚本同名.py；三split的记录文档ID交集均为0。

## 已确认落实的部分

1. E01/E10模型参数冻结，反事实路径标注在no_grad下执行；学生特征只读h，不把y/成本当部署输入。
2. E01默认28候选包含baseline；首动作强制后余下按冻结pi0续算，所有候选相同轮数；按列取对应真实执行的成本作为离线政策评估，在末层无后续attention的设置下有效。
3. E10 prefix sharing、第二动作完整28²枚举、a=b重复可用、相同b0的四项Gamma定义、reverse control基本正确。
4. E03没有伪造T2/T3标签，显式限定T1/T4二元决策；阈值在validation选择后才用于test，oracle独立标记。
5. E12线性报价等价性结论本身成立；非线性比较的实现和对照需重做。

## 下一轮最小文件需求

优先原始E01 `states_{train,validation,test}.pt` 与 `students.pt`（105M运行），以及其完整CLI或shell日志。它们可验证状态分布、学生训练曲线、delta校准、修正后决策差异，不一定需要optimizer状态。重跑真实在线前向与扩充跨文档抽样时才需要105M骨干 `latest.pt` 或仅model state_dict、tokenizer及小份打包输入和doc_map。E10若要精确修正符号统计，只需states_test.pt内pair_q，不需要完整骨干。

## 覆盖文件

完整审读：run_e01.py；eval/counterfactual.py；train/value_student.py；experiments/e03_halting.py；experiments/e10_ordered.py；experiments/e12_bidding.py；experiments/e12_equivalence.py；model/top2.py；eval/dynamic_halting.py；eval/profile_dynamic.py；tests/test_counterfactual.py；tests/test_e03_halting.py；tests/test_e10_ordered.py；tests/test_e12_bidding.py；tests/test_dynamic_halting.py。相关阅读：data/packed_lm.py、data/fineweb.py的split部分、方案E01/E03/E10/E12、上述运行summary/records/registry。模型主干和数据完整审查由其他审查者覆盖。

验证脚本 `repro_counterfactual_numpy.py` 已执行通过，结果保存为 `repro_counterfactual_numpy_results.json`。随后在CPU PyTorch **2.5.1+cpu**（用户训练环境2.12.1+cu130）直接导入未改动仓库运行 `repro_counterfactual_torch.py`，已复现有限阈值决策差异、E12独立性错误、Module.to报错、摘要相消、自定义候选错位、零误算sign-flip等。随机小模型中raw-h契约本应7token都R1，原动态代码第5token执行R4，最大logit差0.00419546；这只是反例，不是原checkpoint效果估计。

最小修复差分见 `patches/minimal_correctness_fixes_with_tests.patch`，另有两个独立.patch。`validate_minimal_patches.py`在内存中应用修改，不改仓库：原实现3项回归=2失败+1报错；补丁后3/3通过。测试只覆盖raw/normalized特征契约、独立专家首层和Module.to，不改变候选空间、训练协议、统计门槛或任何上传结果。日志与机器可读summary分别为`minimal_patch_tests_{original,patched}.log/.json`。
