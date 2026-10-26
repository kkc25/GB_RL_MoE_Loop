# Loop-MoE 代码审计修复与有界复验报告

> **后续复审提示：** 本文保留 2026-10-25 的阶段记录。后续审计发现 R2 廉价基线映射、R1 闭环入口和 R4 实际成本对照仍需修正，相关 A 方案修复与验收见 [2026-10-26 方案 A 返修报告](Loop_MoE_Followup_A_Repair_Report_2026-10-26.md)。本文中的对应结论不得脱离该后续报告单独引用。

**日期：2026-10-25**

**项目：`/nvme-1/zhoukegu/GB_RL_MoE_Loop`**

**修复分支：`repair/code-audit-2026-10-25`**

**原始提交：`f2bef422f7315f30618f00c690bf82810d1e85ac`**

## 1. 范围与结论

本轮依据合作者审计直接修复代码、补充回归测试，并在现有 M105 检查点上完成有界复验。没有启动 300M、126M/280M、末四层、真实 LM RL 或大规模重训；旧实验目录均保留。

当前结论：

1. **raw-h 停止契约已修复并完成在线/离线闭环验收。** 在旧 validation/test 各 1,024 个状态上，修复后的离线决策与在线执行不一致数均为 0，最大 predicted-delta 差均为 0。
2. **当前四候选、raw-h 接口没有显示可部署的动作价值收益。** 新诊断 test 的 raw MLP 部署改善为 −0.003933 nat，文档级 95% CI 为 [−0.006136, −0.001617]；其他学习模型也未超过简单规则。因此不重启 E02 或 LM RL。
3. **动态停止没有超过固定深度对照。** direct-delta 策略相对固定 R2 的 NLL 增加 0.001121 nat；等成本随机深度混合也差于固定 R2。固定 R1 与 R2 的 NLL 差约 0.001214 nat，已低于 0.005 nat 容忍值，当前优先保留简单固定深度。
4. **修复并重训后的 E12 仍不支持专家自报价优势。** 分专家 adapter 在同 cell MSE 和公平部署对照中均未稳定超过中央 MLP。
5. **最终回归为 521 项通过、1 项跳过、0 项失败。** 这只是代码正确性证据，不等于所有旧科学结论已经重新验证。

这些是单一 M105 骨干、当前状态接口和诊断样本上的限域结论，不是“循环机制普遍无效”或“状态在数学上无信息”的证明。

## 2. 主要代码修复

| 范围 | 修复与验收 |
|---|---|
| E11 动态停止 | 学生训练、校准和部署统一使用 raw pre-FFN `h`；router 仍使用 RMSNorm 后的输入 |
| E11 系统测量 | 固定/动态路径均只测 input→logits；CE 排除在计时外；逐模式重置显存；分开记录监督位置深度、全位置 FFN rounds 和 expert-token pairs |
| E12 报价 adapter | 修复专家轴错误合并，恢复 `nn.Module._apply` 语义；`.to()`/dtype 转换可用；新增同 anchor、同候选的 central restricted 对照 |
| 反事实动作 | 修复 global/local action ID、逐状态候选和非前缀候选；每个状态保存候选映射 |
| 精确 RL 玩具代码 | 按唯一合法路径概率加权梯度；短 budget 枚举概率质量恢复为 1；训练/评估 price 一致 |
| LM 训练与恢复 | sampler 按 epoch 复用；累计存盘时间；恢复 best、optimizer、RNG、数据位置和 hash；分离深度与调用账本 |
| 显式评估 | `run_recovery.py --mode evaluate` 必须加载并校验 checkpoint，不再评估随机初始化模型 |
| E07/E08/E10 等 | 修复离线重排、prefix knapsack、strict sign flip、固定题目强制路径干预、软放松残差和 regret 口径 |
| registry/report | 顶层状态不再被最后组件覆盖；缺失 GPUh 记为 unknown；缺失数学检查不再错误放行 |
| R2/R4 数据契约 | 状态文件绑定 backbone hash 和 corpus doc-map hash；R4 拒绝陈旧语料或不匹配骨干；固定深度与动态策略从同一缓存状态配对计算 |

对应实现主要位于 `eval/`、`experiments/`、`train/`、`data/controller_corpus.py` 和 `eval/state_sampling.py`。原审计及逐项证据保留在 `audits/code_audit_2026-10-25/`。

## 3. R1：raw-h 在线/离线闭环

复用 M105、旧学生和旧 validation/test 状态，只验收实现契约，不重新选择阈值。

| 分区 | 状态数 | raw-h 决策不一致 | 最大 delta 差 | normalized-h 决策改变 | raw-h 平均深度 | normalized-h 平均深度 |
|---|---:|---:|---:|---:|---:|---:|
| validation | 1,024 | 0 | 0.0 | 169 | 1.9727 | 1.7412 |
| test | 1,024 | 0 | 0.0 | 217 | 2.4502 | 2.1133 |

这说明旧 normalized-h 部署确实改变了大量决策，而修复后在线策略与离线参考一致。每个旧分区只覆盖 2 个 packed sequences，因此这里只证明实现闭环，不证明跨文档泛化或系统加速。

证据：

- `runs/repair_2026_10_25/r1_validation/summary.json`
- `runs/repair_2026_10_25/r1_bf16/summary.json`

## 4. R2/R3：多文档动作价值复验

### 4.1 数据与角色

从固定 FineWeb-Edu revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` 建立新的 controller 语料，复用原 BPE，排除实际进入原 train/validation/test 的文档及其近重复组。

| 角色 | 语料文档数 | 采样状态数 | 实际覆盖文档 | 最大文档权重 | 重复位置 |
|---|---:|---:|---:|---:|---:|
| controller_train | 873 | 8,000 | 873 | 0.00125 | 0 |
| selection | 412 | 2,000 | 412 | 0.00250 | 0 |
| calibration | 405 | 2,000 | 405 | 0.00250 | 0 |
| newtest（诊断） | 385 | 4,000 | 385 | 0.00275 | 0 |

每文档最多 16 个合法位置，采样优先覆盖文档。区间按文档 paired bootstrap 估计，不再把 token 数当独立样本数。

**确认集限制：** smoke 和正式诊断均访问过 `newtest`，所以它是新文档诊断集，不是完全未查看的确认集。后续若做确认性声明，必须另封存未查看文档。本轮也未完成多骨干 seed、完整学习曲线或自然 token 流 1M token 的独立质量确认。

语料证据：`runs/repair_2026_10_25/controller_corpus/manifest.json`。

### 4.2 部署结果

所有学习方法共享同一逐状态四候选池，池内包含 baseline；改善定义为相对原 router 的 NLL 降低，正值较好。

| 策略 | 部署改善（nat） | 文档级 95% CI |
|---|---:|---:|
| 常量规则 | +0.000512 | [−0.000588, +0.001544] |
| 按 baseline 分组规则 | +0.000483 | [−0.000648, +0.001528] |
| raw MLP | −0.003933 | [−0.006136, −0.001617] |
| centered MLP | −0.002124 | [−0.004592, +0.000147] |
| raw / centered 线性 | −0.001542 | [−0.003868, +0.000500] |
| 随机候选 | −0.001830 | [−0.004152, +0.000323] |

答案知情四候选 oracle gap 为 0.034400 nat，但它不可部署，且不能与旧 28 候选 oracle gap 直接比较。简单规则的小正均值区间均跨零，不能称为阳性；raw MLP 则显著差于 baseline。

**决定：** 当前配置下不开放 E02/E04，不启动真实 LM RL。该负结果只关闭当前骨干、候选和状态接口的路线，不证明新骨干或新表示也不可学习。

证据：`runs/repair_2026_10_25/r2_full/summary.json`。

### 4.3 实际标注账本

- backbone prefix tokens：1,942,528
- candidate pair calls：256,000
- head readouts：128,000
- 两个 MLP 的学生训练样本计数：160,000
- 学生 optimizer updates：640
- R2 总耗时：66.52 秒
- 峰值 CUDA 已分配显存：1,742,155,776 bytes

证据：`runs/repair_2026_10_25/r2_full/cost_ledger.json`。这些是本次单卡作业记录，不是其他 GPU 或大规模训练的时间估计。

## 5. R4：固定深度与动态停止

直接学生学习 `Delta_1_to_4 = immediate − Q^pi0`；selection 用于选学生，calibration 用于选择预算 2 的阈值。固定 R1–R4 与动态策略均从同一 newtest 缓存状态计算；R1/R4 直接复用原配对标签，R2/R3 新测真实基线路径，没有用端点插值伪造标签。

| 策略 | 平均 rounds | NLL | 相对固定 R2 的 NLL 增量 |
|---|---:|---:|---:|
| 固定 R1 | 1.0000 | 4.968461 | +0.001214 |
| 固定 R2 | 2.0000 | **4.967247** | 0 |
| 固定 R3 | 3.0000 | 4.967272 | +0.000025 |
| 固定 R4 | 4.0000 | 4.967844 | +0.000596 |
| direct-delta 动态 | 1.6195 | 4.968369 | +0.001121 |
| 随机 R1/R4 混合 | 2.0110 | 4.968562 | +0.001314 |

文档级配对区间（改善 = 对照 NLL − 策略 NLL）：

- direct-delta 相对固定 R1：+0.000092 nat，95% CI [−0.001072, +0.001308]。
- direct-delta 相对固定 R2：−0.001121 nat，95% CI [−0.002196, +0.000069]。
- 随机混合相对固定 R2：−0.001314 nat，95% CI [−0.002516, −0.000178]。

**决定：** 当前证据不支持动态停止的状态条件增量价值。固定 R1 相对 R2 的差距已低于 0.005 nat 容忍值，优先保留固定 R1；若追求本诊断集上的最低均值，固定 R2 略优，但差异很小。本实验仍是 R1/R4 二选一的 direct-delta 诊断，不是完整多轮停止器，也没有形成真实 wall-clock 加速证据。

证据：`runs/repair_2026_10_25/r4_full/summary.json`。

## 6. E12：修复后重新训练

使用修复后的独立专家计算重新训练全部 adapter，没有复用旧 adapter 权重。训练/validation/test 状态数为 8,192/1,024/1,024；各组 epochs、lr、batch、weight decay 和 seed 一致，并保存新权重及配置。

| 对照 | 结果 |
|---|---:|
| 中央 MLP 同 cell MSE | **12.87886** |
| 分专家报价同 cell MSE | 12.91294 |
| 带摘要报价同 cell MSE | 13.12227 |
| 中央 unrestricted 部署改善 | −0.008009 nat |
| 中央 restricted，learned anchor | −0.000747 nat |
| 报价，learned anchor | −0.003054 nat |
| 中央 restricted，baseline-first anchor | −0.002451 nat |
| 报价，baseline-first anchor | +0.000130 ± 0.003491 nat（state-level SE） |
| 带摘要报价，baseline-first anchor | −0.003195 nat |

唯一略正的均值远小于其标准误，而且该实验仍使用原 E01 状态分布，不是新多文档确认。结论仍是停止专家自报价结构路线，不宣称激励兼容、市场机制或分布式分配优势。

证据：`runs/repair_2026_10_25/e12_bidding/e12_bidding_summary.json`。

## 7. 旧结论的保留、修正与撤回

| 旧内容 | 当前处理 |
|---|---|
| E03/E11 使用 normalized-h 部署得到的动态数值 | 撤回其实现一致性依据；由 raw-h 闭环和本轮 R4 限域结果替代 |
| “动态停止已证明有质量—计算优势” | 撤回为一般结论；本轮 direct-delta 未胜固定 R2，固定 R1 已满足 0.005 nat 容忍 |
| E01“状态无信息/路径不可学习” | 改为当前学生、样本和接口未显示部署收益；不得外推为数学不可能 |
| E01/E10 以大量 token 作为独立泛化证据 | 不再采用；新诊断按文档配对统计，E12 旧分布结果明确限域 |
| 旧 E12 adapter 数值与权重 | 由修复公式后的重训结果替代 |
| 共享 GPU 延迟和显存比较 | 不支持真实加速声明；须在隔离 GPU 上按统一口径重测 |
| 旧 Torch exact 玩具训练数值 | 不作为精确 RL 证据；概率加权与合法枚举修复后的测试通过 |
| 当前 NumPy E09 梯度/概率恒等式 | 保留为合成数学核验，不外推为 LM RL 效果 |
| E00 可训练性与原 M105 训练记录 | 保留历史记录；旧 dense 配平不足，不据此宣称架构普遍优劣 |
| E07/E08/E10 等旧实验数值 | 代码回归已修复并通过，但未全部重新执行科学实验，不升级为新的自然 LM 证据 |

原报告 `Loop_MoE_Experiment_Report_2026-10-25.md` 已添加审计提示，作为历史记录保留。

## 8. 测试、登记与复现

最终命令：

```bash
.venv/bin/python -m unittest discover -s tests -v \
  > runs/repair_2026_10_25/full_tests_final.log 2>&1
```

结果：**Ran 521 tests，OK (skipped=1)**。`git diff --check` 通过。后台任务已全部结束。

复验登记：`runs/repair_2026_10_25/registry.json`。其中 GPUh 未可靠记录时保持 `null/unknown`，不把缺失值写成 0。

主要复现命令（均在项目根目录执行）：

```bash
.venv/bin/python -m experiments.prepare_controller_corpus \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --tokenizer runs/data/fineweb_edu_100m/tokenizer.json \
  --exclude-documents runs/data/fineweb_edu_100m/documents.jsonl runs/data/fineweb_edu_debug/documents.jsonl \
  --output-dir runs/repair_2026_10_25/controller_corpus \
  --controller-train-tokens 1000000 --selection-tokens 400000 \
  --calibration-tokens 400000 --newtest-tokens 400000

CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m experiments.r2_value \
  --config configs/e00_m105_b2.json --checkpoint runs/e00/m105_b2/latest.pt \
  --corpus-dir runs/repair_2026_10_25/controller_corpus \
  --output-dir runs/repair_2026_10_25/r2_full --device cuda \
  --train-states 8000 --selection-states 2000 \
  --calibration-states 2000 --test-states 4000 --epochs 10

CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m experiments.r4_quality \
  --config configs/e00_m105_b2.json --checkpoint runs/e00/m105_b2/latest.pt \
  --corpus-dir runs/repair_2026_10_25/controller_corpus \
  --r2-dir runs/repair_2026_10_25/r2_full \
  --output-dir runs/repair_2026_10_25/r4_full --device cuda --epochs 10

.venv/bin/python -m experiments.e12_bidding \
  --input-dir runs/e01/m105_100m \
  --output-dir runs/repair_2026_10_25/e12_bidding --epochs 5
```

`.pt`、checkpoint、原始数据和 `.venv` 均被 Git 忽略；小型 JSON、日志和 Markdown 报告可进入版本控制。

## 9. 仍未完成的事项

- 未做隔离 GPU 延迟/吞吐确认，不能声明系统净加速。
- 未做新的自然 token 流 1M token 固定深度质量确认、多骨干 seed 或完整学习曲线。
- 当前 newtest 已用于诊断，后续确认需另封存未查看文档。
- 未启动 E02/E04、真实 LM RL、300M、126M/280M、末四层或全块循环；当前证据不满足开放门槛。
- 修复通过 `repair/code-audit-2026-10-25` 分支同步；默认分支是否合并由仓库所有者另行决定。
