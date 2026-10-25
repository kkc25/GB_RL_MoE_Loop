# 交付说明与证据范围

## 本次已经完成

- 以前面对话为起点，核验所列主要论文，并补充截至2026-10-25的直接近邻。
- 形成有明确否定条件的研究主线；近期PoLar、LatentHarness等进一步重叠后，重新收紧创新表述和强基线。
- 写成E00–E12共13份独立Markdown实验方案，包括数据、训练对象、方法、对照、消融、评价、资源与停止/转向条件。
- 给出RTX A6000官方规格边界、67M/105M/180M参数计算、FFN-only循环FLOPs与显存规划。
- 推导充分状态、近似价值改进、固定凸分配、预算账本及收益贪心的适用条件，提供反例。
- 实际执行CPU组合计数、参数/FLOPs计算、标签oracle反例、预算反例和切换稳定性等验证脚本。
- 检查Markdown内部链接、数学定界符、代码围栏、Python语法、JSON格式，以及压缩包CRC/逐文件哈希。

## 尚未完成且没有声称完成

- 没有连接、占用或benchmark你们的RTX A6000服务器。
- 没有训练本计划的语言模型、下载训练语料或生成真实LM的反事实标签。
- 没有测得真实tokens/s、峰值显存、NLL、PPL、accuracy或推理加速。
- 没有实现完整训练器；配置与run模板是实施规范，不是可直接调用的训练程序。
- 没有证明此组合在文献中首次出现，也没有保证实验成功或发表。

这些不是工作遗漏：本次请求的交付目标是文献调研、严格研究设计与逐项实验方案。真正GPU实验应先执行E00校准，再按决策树启动。

## 文献的核验层级

每篇重要工作在文献文件中区分摘要、方法节/全文、正式出版页或作者库。访问受阻的OpenReview条目保留具体限制；未核验的开销数字不进入预算。近期预印本按作者报告表述。

“能找到论文”与“验证作者结论”是两件事。本包没有独立复现外部论文数值。对SeqTopK公开公式的预算反例，只限定公式及其文字保证，未运行或断言其仓库代码存在同样问题。

## 数值与预算口径

1. 精确参数计数只对定义好的bias-free、tied-head基础模型成立，controller/step编码额外列。
2. FLOPs表是矩阵计算估算，非GPU测量；attention实现、activation、dispatch、通信、checkpoint重算另计。
3. 7.98%的末层节省上限限定于S1024、逐位置单次head的计算模型，不代表所有prefill/decode情形。
4. GPU-hour表中的2k/8k/20k吞吐是代数情景，真正预算由E00实测逐run替换。
5. 5%、0.01nat、0.005nat等为预登记建议阈值，不是经验定律。
6. 两步候选扩展最多16组合加1个基准策略候选；普通E02-A只有4个首动作加固定尾策略，二者不混账。

## 复算

从解压后`research_bundle`目录执行：

```bash
python3 code/verify_design.py --output results/design_checks.json
python3 resources/code/estimate_budget.py --loop-blocks 1 --loops 4 --tokens 1000000000
python3 theory/verify_counterexamples.py
python3 code/validate_bundle.py
```

前三个是CPU解析/玩具计算。`resources/code/probe_hardware.py`需要在你们的GPU机器运行才能提供真实设备信息；它不是训练吞吐benchmark。E00说明如何另测训练与推理性能。

`MANIFEST_SHA256.json`记录交付时每个文件的字节哈希（不递归包含自身）。重新运行会写结果的脚本后，部分文件哈希可能变化，这是预期行为；如要验证原始包，请先验证再修改。

## 最小启动安排

先确定一个维护实现的人与一个核对统计/数学的人。三张卡先各自独立运行校准、骨干和后续seed，不要求专家并行。首轮只落实E00、E01、E02强对照与E08门禁；其余方案按触发条件启动。所有日志使用唯一run_id并保存完整checkpoint与数据游标。
