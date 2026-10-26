# 下一轮最小文件清单与验证目标

本轮代码审查已经完成。下列文件用于下一轮量化已定位问题、复核真实权重上的行为；不需要为此补齐所有历史checkpoint或立即继续大规模训练。

审查源码固定为：`kkc25/GB_RL_MoE_Loop` 的提交 `f2bef422f7315f30618f00c690bf82810d1e85ac`。

## 一、已经拿到的文件，不必重复上传

本轮已读取该提交全部390个Git文件，并核对Git blob哈希。下列文件已经足够，不需要再次上传：

- `runs/data/fineweb_edu_100m/manifest.json`、`tokenizer.json`，以及debug数据对应的两份文件。
- `configs/*.json`、各run的`resolved_config.json`、`environment.json`、`cost_ledger.json`。
- E00/E01/E03/E07/E09/E10/E11/E12的已上传summary、日志、registry。
- E01的`pilot`、`coverage_remediation`、`m105_100m`中已上传的`records_{train,validation,test}.jsonl`。
- E10 `pilot/records_{train,validation,test}.jsonl`。
- 实验报告、研究方案和全部源码/测试。

如果服务器上的这些文件在本次Git提交之后发生变化，则只提供变更版并说明对应新提交或修改日期，不要混作原来运行的证据。

## 二、第一优先级：先上传105M的学生和缓存状态

这些文件按当前源码确实会保存，但未包含在Git提交中：

| 优先级 | 服务器上的现有路径 | 能验证什么 | 是否需要完整骨干 |
|---|---|---|---|
| A1 | `runs/e01/m105_100m/students.pt` | 两个学生的实际hidden_dim、训练history、选中权重、mean/std；消除“源码默认值就是实际运行值”的猜测 | 否 |
| A1 | `runs/e01/m105_100m/states_validation.pt` | 按原始FP32标签重现E03阈值选择、学生预测与验证分箱；核对骨干hash/rounds/action表 | 否 |
| A1 | `runs/e01/m105_100m/states_test.pt` | 重现原E01学生选择和真实已标注路径成本、E03离线决策与质量；比JSONL保留更完整精度 | 否 |
| A2 | `runs/e01/m105_100m/states_train.pt` | 训练曲线/拟合诊断，低容量头和动作中心化监督的受控重训；保持相同原始标签比较 | 否 |

`states_*.pt`保存的内容包括h、target、文档/序列/位置、全部28动作的immediate和Q_pi0成本、baseline_action、rounds、backbone_hash。10240个512维FP32 h本体约21MB，连同成本和索引通常是几十MB量级；准确体积以实际文件为准。学生文件大小由实际hidden_dim决定，不以源码默认1024假定它已经使用该宽度。

**最小起步包是A1三份文件。** A2有助于比较如何修学生，但不是复现已有离线结果的前提。缓存状态本身仍只有原来的15/2/3文档覆盖；上传它们不能补足统计代表性。

现有E03只验证三份states的hash一致，并不验证students来源。因此建议随学生文件附一个短metadata JSON，列出该文件的SHA256、来源E01目录、原骨干文件SHA256和实际命令。无需重新导出已有JSONL。

## 三、第二优先级：一个105M骨干的model-only导出

下一步要定量检查动态停止的raw-h/RMSNorm-h输入差异、验证补丁后在线/离线一致性、扩充跨文档状态时，需要对应骨干。

目前日志和E01环境共同记录的原文件是：

```text
runs/e00/m105_b2/latest.pt
原文件 SHA256:
47f6ec019b4d72a3229076e4f60d3dd7e41c26e6a14c12dc964536e65de33324
更新数: 97657
参数量: 104919552
```

优先导出单个 `m105_b2_model_only.pt`，包含`model` state_dict、当时的model/config、update和原checkpoint SHA256。**不需要optimizer状态、所有历史checkpoint、所有67M基线权重。** FP32独立参数按4 bytes估算约420MB；实际序列化大小还受共享storage及元数据影响。不要为了减小上传体积擅自转BF16，这会让精确一致性复核与原FP32权重不同。

如目前只想检查两个学生因输入归一化造成的决策变化，可以先从同一骨干导出极小的末层`ffn_norm.weight`及`norm_eps`，与A1一起计算raw-h和normalized-h预测。它只能验证输入变换导致的策略差异，不能重跑真实专家/词表head、验证端到端NLL或重新采样隐藏状态。若已提供model-only骨干，就不必另传这个小片段。

### 三种hash必须区分

| 标识 | 含义 | 应如何记录 |
|---|---|---|
| `source_commit` | 执行/审查的源码Git提交 | 本轮固定提交为`f2bef422…`；如训练期间源码不同需明确提供差异 |
| `source_checkpoint_sha256` | 原始`latest.pt`整文件字节哈希 | 本轮记录为上面的`47f6ec…`，用于衔接既有states标签 |
| `export_file_sha256` | 新model-only导出文件的字节哈希 | 导出后重新计算，**正常情况下不会等于原checkpoint哈希** |

从checkpoint抽取state_dict再保存，即使权重完全相同，文件结构与字节已变，不能要求新文件仍有`47f6ec…`。若要比较“权重内容一致”，应另定义明确的规范张量哈希，固定key顺序、dtype、shape和原始tensor bytes；不要把任意重新`torch.save`得到的文件hash称为规范权重hash。

原E01 `backbone_hash`本质上是checkpoint文件hash，不是源码hash，也不是独立定义的参数hash。现有states的来源hash应保留；新增导出metadata通过`source_checkpoint_sha256`连接它，不能粗暴改写旧标签hash来“对齐”。

## 四、配合骨干上传一小份原始打包输入

仅有权重和已缓存h能检查末层反事实，但要验证“从输入到缓存h到输出”的全链路，需要少量输入。Git中没有上传原始`*.input_ids.u16.bin`、`*.segment_ids.i32.bin`或doc_map；无需为了第一轮核对上传完整100M训练数据。

建议从原数据导出一个小型 `audit_batches.pt` 或 `.npz`，每条样本保留：

- `input_ids`、`labels`、`segment_ids`、`doc_ids`。
- 原始split名、sequence index、每条序列文档映射；明确保存的是移位后的next-token标签。
- 原数据目录及manifest hash；如果只导出部分序列，另给导出文件hash。

先包含以下集合即可：

1. `fineweb_edu_100m/validation` 的前16个序列，即E00和E11实际质量评估使用的窗口。
2. E01 validation的sequence **3306、2322**；test的sequence **2086、3112**，用于核对缓存状态/标签与原序列。

这些是从已上传records和评估代码读到的序列ID，不是建议重新选取有利样本。它们仅用于复现已有结果；真正新测试应另行封存。

完整原始打包数据只有在决定大规模重采样时才需要。届时可以在你们服务器运行导出器，仅上传新采样的状态和对应小型原始序列审计样本。

## 五、需要补充的运行命令与配置，不是重复现有JSON

现有 `runs/e01/m105_100m/resolved_config.json` 由LM配置保存，包含的`updates=100`并非实际骨干训练的97657更新；它也没有完整记录E01 CLI覆盖值。请补充当时实际命令或等价结构化配置：

| 运行 | 目前缺少的关键信息 |
|---|---|
| E01 m105_100m | `--checkpoint`、`--train-states/validation-states/test-states`、`--epochs`、`--hidden-dim`、`--lr`、`--student-batch-size`、`--seed`、`--dtype`、`--depth`、`--rounds`及chunk设定 |
| E03 m105_100m | 实际`--input-dir`、budgets、tau-grid-size、max-states、nll-tolerance；阈值来源文件hash |
| E11动态profile | 实际config/checkpoint/E01目录/E03 summary路径、warmup、steps、validation-batches、运行时源码版本 |
| 如需要继续E10/E12 | 各自完整CLI、源码版本和使用的E01/骨干hash，不与105M结果混为同一条件 |

A1学生文件可以直接查hidden_dim和history长度；记录命令仍有价值，因为目前summary并没有完整、可审计的参数清单。如果shell日志不在，请如实写“未知”，不根据当前默认值回填为已知参数。

## 六、暂不优先上传的文件

- **所有optimizer和历史checkpoint**：本轮发现不依赖它们；只有要精确续训或复现资源迁移时才需要。
- **E12所谓最终adapter checkpoint**：当前`experiments/e12_bidding.py`只保存summary，未实现保存这些训练完的adapter/anchor权重。不要寻找一个源码没有生成的文件；修复后重跑时新增权重与完整配置保存即可。
- **全部67M基线权重**：等105M管线核对清楚后，再决定是否需要架构对比复核。
- **E10全套权重**：若只修正参考符号统计，上传`runs/e10/pilot/states_test.pt`中的全精度`pair_q`及metadata即可，不需要完整骨干。原E10骨干hash是`7c417e7bcac3469596275ccba6f05b965e6bdfeefd0046de033d74a16910d9d9`，不同于105M的hash；样本量不足的问题仍需新实验解决。

## 七、哪些问题本轮已经验证，权重不会改变其代码事实

1. 停止学生训练/校准读raw h，动态部署读normalized h；已用真实原代码、小型随机权重和有限阈值复现决策差异。
2. E12 einsum混合输入专家轴、覆盖Module._apply；原代码测试失败，最小补丁后对应回归通过。
3. E12中央28候选与anchor条件7候选不一致；优化器weight_decay配置没有显式对齐。
4. E01/E03/E10实际文档覆盖过少、token级SE替代文档统计、独立controller/calibration分区缺失。
5. E10零被计入sign-flip、自定义候选actions未正确映射的接口错误。
6. 当前报告的E11计时路径不对称、固定路径额外计算loss；这需要改测量代码，并非再上传权重就能使旧计时可比。

权重能回答的是“这些错误使实际NLL/预算/策略改变多少”“修复后结果能否重现”“学生是否优化充分”等量化问题。未上传权重不妨碍本轮对源码和已上传记录完成审查。

## 八、后续重新校准与新test应怎样分开

把本轮反复分析过的样本作为开发诊断集。不要在同一旧test上不断修模型、选阈值后再称其为最终持出测试。

固定现有骨干时，优先从新取且未被骨干更新过的文档池中，按稳定的文档hash划分：controller训练、模型/超参数选择、阈值calibration、封存新test。已正式锁定的LM validation/test不转作controller_train；旧池如需重分角色，必须显式登记新协议，并确认其此前未锁定、未使用。此前已分析过的文档整体排除出新test；不能仅换取同一文档的另外几个token。采样时给每文档的状态数量设上限，同时记录实际文档数与最大文档权重。

先在训练和selection split确定学生及监督目标，再只用calibration确定阈值，最后一次性在新test上计算文档配对差、预先约定的置信方式及实际预算。已有0.005 nat阈值可继续作为应用容忍，但统计非劣检验的单侧/双侧区间和成功条件应在看新test之前写明。

### 动作中心化用于路由时，停止必须保留目标间的offset

对于同一个状态、同一终止深度的候选动作表，减去共同标量不改变动作选择：

\[
\arg\min_a Q(s,a)=\arg\min_a\left[Q(s,a)-\overline Q(s)\right].
\]

所以可以用动作中心化成本或相对原router动作的advantage改善路由回归；这是需要实验验证的优化改进，不是已证明能得到正收益。

但是，**不能将immediate和Q_pi0分别中心化，再直接用两者预测之差作为停止收益**。两个目标的共同均值差恰恰可能包含“继续是否总体有益”的主要信号。停止应直接训练配对的

\[
\Delta(s)=L_{T1}(s)-L_{T4}(s),
\]

或同时预测并保留各自offset，复原到同一CE单位后再比较。所有成本与收益都必须对应同一续算时域。这样才能把“改进动作排序”与“决定是否继续计算”两个目标分别检验清楚。
