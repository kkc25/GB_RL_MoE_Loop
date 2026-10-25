# 实验登记、资源闸门与报告模板

## 1. 用累计GPU小时管理，不按卡数假设无限扩展

训练run j实际吞吐q_j为有效tokens/s（包含选定训练路径），新增D_j tokens用时

\[
H_{train}=\sum_j\frac{D_j}{3600q_j}\times G_j,
\]

其中q_j是该整个G_j卡作业的aggregate吞吐。单卡时G_j=1。不能把多卡总吞吐当单卡吞吐又不乘GPU数。标签、搜索、评测、失败调参和控制器训练按实测GPU占用单独记；不把其样本数按LM q折算成真实耗时。

资源文件中的2k/8k/20k tokens/s都是代数情景，**没有测过你们机器的实际值**。

## 2. 默认训练量与升级路径

| 计划 | 累计训练token量 | 具体构成 | 结论等级 |
|---|---:|---|---|
| 初筛 | 0.95B | S67共50M；M105 B0/B1/B2各300M | 选择主线，1seed探索 |
| 最小机制确认 | 3.65B | 初筛＋B2三seed各累计1B，抵扣已有B2的300M | 同骨干controller比较可三seed确认 |
| 扩充架构对照 | 5.65B | 上述＋两个新强架构baseline各1B单seed | 架构对照仍探索性 |
| 完整架构确认 | 9.65B | 最小机制确认＋两个强架构baseline各3seed×1B | 架构比较也三seed |

不要求全部走到最末。若300M已经能稳定区分主要方法，可将确认token数下调并预先登记，节省量按各run计算。

50M包含5–20M smoke；300M包含100M pilot；只有同模型配置/数据流/优化器状态的连续训练可抵扣。若两个强架构baseline就是可延续的初筛B0/B1，还可抵扣2×300M，变为5.05B或9.05B。不得把已丢弃重跑的超参成本抵掉。

## 3. 代数情景表

为直观展示数量级，暂假设所有独立单卡run都具有相同q（实际必须逐run替换）：

| 计划 | q=2,000 | q=8,000 | q=20,000 |
|---|---:|---:|---:|
| 初筛0.95B | 131.94 GPU-h | 32.99 GPU-h | 13.19 GPU-h |
| 最小确认3.65B | 506.94 GPU-h | 126.74 GPU-h | 50.69 GPU-h |
| 架构对照5.65B | 784.72 GPU-h | 196.18 GPU-h | 78.47 GPU-h |
| 完整确认9.65B | 1340.28 GPU-h | 335.07 GPU-h | 134.03 GPU-h |

这些不是预计达到的吞吐档位，不能据此承诺工期。日历小时还要除以可用并行卡数和实际利用率，并计排队/抢占；3–5卡动态分配更适合维护GPU-hour队列。

## 4. 控制器研究的额外预算上限

以对应公共骨干实际训练GPU小时H_B2为参考：

| 活动 | 增量预算上限 | 触发 |
|---|---:|---|
| E01诊断 | 5% H_B2 | 骨干可用 |
| E02-A主价值学习 | 5% H_B2 | E01可学性通过 |
| E02-B两步路径 | 另5% H_B2 | A有效且长期信号存在 |
| E03停止 | 5% H_B2 | 深度收益异质 |
| E04记忆消融 | 1% H_B2 | 复用相同hash标签 |

不含B时核心上限16% H_B2，含B为21%。这不是总项目上限的16/21%，也不意味着实际一定要用满。每份文档提的单个状态数只是起点，先碰到GPU上限就缩小样本并报告统计能力。

复用同一hash候选成本只按唯一run收费一次。300M骨干升级1B后须重新标注，其旧标签成本已经发生，不能假装免费迁移。为未明确的运维/失败/评测保留最多25%项目规划余量；已列入独立活动账本的成本不重复再收同一项。

E05/E06/E07初筛各最多24 GPU-h，E08学习分支最多24 GPU-h，**这些是有触发条件的替代项，不建议同时占满**。E09上限见其文档；E10/E12每项约1% H_B2小型筛查。E11是另立训练run，按其新模型/循环范围q实测核算。

## 5. 建议队列

**3卡常态：**卡A跑当前骨干/seed，卡B跑第二seed或强baseline，卡C跑第三seed/已冻结标签；不得让频繁性能测量与同卡训练竞争。

**5卡常态：**两个实测有收益的2卡作业＋1卡标签/评测；若2卡DDP不节省GPU小时则五个独立单卡。推理benchmark单独保留短隔离窗口。

**临时8卡：**优先并行确认seeds、补强基线或做已准备好的profile，不在短暂窗口里启动依赖8卡永久持有的核心训练。

## 6. Run登记模板

```yaml
run_id: "E02A_B2_seed17_ckpt1B_v1"
status: planned  # planned / running / finished / stopped / failed
hypothesis: "多步续算目标优于等标签预算一步目标"
primary_comparison: "EPO-style_controlled"
backbone_hash: null
code_commit: null
tokenizer_hash: null
data_manifest_hash: null
controller_train_split: controller_train
evaluation_split: controller_test
seed_backbone: 17
seed_controller: 101
target_loop_blocks: 1
top_k: 2
max_loops: 4
candidate_policy_version: "causal_b4_with_baseline_v1"
continuation_policy_hash: null
loss_target: "conditional_mean_terminal_nll"
regression_loss: mse
trainable_parameters: "controller_only"
label_budget_gpu_hours: null  # E00实测后填写
actual_gpu_hours: null
actual_expert_calls: null
actual_head_evaluations: null
primary_metric: "paired_delta_nll_at_fixed_calls"
stop_rule: "预登记预算或连续验证无增益"
result: null
```

这是登记格式，不是可直接调用的训练CLI。正式训练器未在本包实现，命令名和checkpoint路径必须由实现时填入的真实对象产生。

## 7. 结果主表空模板

| method | backbone seed | total params | training GPU-h | probe GPU-h | NLL | PPL | expert calls/token | forward FLOPs/token | head calls | tokens/s | p95 ms | max GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 待执行 |  |  |  |  |  |  |  |  |  |  |  |  |

配套记录：每条运行的真实model hash、失败原因、预登记变更、数据/阈值访问次数、controller开销、预算/容量违反量、重启造成的样本处理变化。

## 8. Checkpoint与中断恢复

发生实质性新结果、路线排除、关键计算完成、抢占风险时更新一条短checkpoint：

```text
时间：
完成/新发现：
证据文件与run_id：
对当前主线的影响：
下一步与触发条件：
已消耗/剩余GPU-hour：
可恢复checkpoint及数据游标：
```

发现预算不足时，优先降低试验分支数、保留最强对照与seeds，再降低确认tokens；不可通过悄悄改测试集、漏记失败成本或把假定吞吐当实测来继续。
