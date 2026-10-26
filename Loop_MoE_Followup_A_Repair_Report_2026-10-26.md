# Loop-MoE 后续审计方案 A 返修报告

**日期：2026-10-26**

**分支：`repair/followup-audit-2026-10-26`**

**基线提交：`661eb57dd0234d4d2b2ea635845e2592ed82546d`**

## 1. 执行范围

本轮只执行后续审计的方案 A：最小正确性收尾、真实 R1 入口验收、实验身份绑定及训练恢复契约。没有执行方案 B 的 R2/R4 缓存重算、方案 C 的独立自然 token 流确认、方案 D 的学生挽救实验或方案 E 的新骨干训练。

最终全量回归为 **539 项总计，538 项通过、1 项跳过、0 项失败**。GPU 工作使用现有 M105、学生和小型合成训练流，没有重训骨干。

## 2. R2 基线与候选动作契约

### 已修复

- 固定候选排名规则只在 train 上选择排名，部署时保持该排名语义。
- 固定全局动作在每个训练状态上按完整策略评分：候选存在时执行，缺失时回退原 router。
- 分组规则按 baseline global action ID 分组，默认最少 32 个训练状态；稀疏组回退到全局规则。
- 所有 global→local 转换均逐行完成，不再共享会被覆盖的 lookup。
- `evaluate_policies` 拒绝负数或超过候选数的 selection，不再静默执行最后候选。
- 二维共享候选与三维逐状态候选统一按显式 expert pairs 执行。

### 验收

覆盖行置换、拆批、候选列置换、缺失动作回退、稀疏组回退、非法索引以及二维/三维非前缀候选数值等价。R2/反事实定向测试共 21 项通过。

**本轮没有重算正式 R2 基线数值。** 旧常量 `+0.000512` 和分组 `+0.000483` 不再作为有效结论；raw/centered MLP 相对原 router 的旧比较不因该映射错误直接失效。正确数值留待方案 B 复用缓存计算。

## 3. R1 真实生产闭环

### 接口修复

- 在线侧实际调用 `DynamicHaltingPolicy.forward`，不再使用脚本内复制的相似前向。
- 正式策略增加可选 trace，导出 h、baseline action、delta、mask、rounds 和最终 hidden。
- 离线侧固定使用原缓存 h、baseline、学生预测及选中成本。
- 使用原 packed sampler 的真实 batch，不再通过重复同一序列构造 batch。
- CE 显式使用 `(1,V)` logits 与 `(1,)` target。
- 学生预测在外层 BF16 autocast 下仍固定为 FP32，与原离线学生口径一致。
- 每份结果包含 `run_manifest.json`，绑定源码 commit/tree、未提交 diff hash、完整配置、命令、checkpoint、学生、状态、数据/tokenizer、dtype 和 sampler。

### BF16 决策验收

| 分区 | 状态数 | baseline 不一致 | 决策不一致 | rounds 不一致 | 最大缓存 h 差 | 最大 delta 差 |
|---|---:|---:|---:|---:|---:|---:|
| test | 1,024 | 0 | 0 | 0 | 0.0 | 9.54e-6 |
| validation | 1,024 | 0 | 0 | 0 | 0.0 | 1.05e-5 |

两个分区各只有 1 个阈值附近状态，均未发生决策分歧。平均 rounds 也与原缓存决策完全一致。

这证明：**在原缓存对应的 BF16 骨干、FP32 学生和真实 packed batch 下，正式 raw-h 策略复现了离线 baseline、delta、mask 和 rounds。** 它不再只是同一 `replay_h` 上的自洽比较。

### FP32 与 CE 边界

把生产骨干改为 FP32 后，缓存 h 不再一致：test 有 11 个 baseline 和 5 个决策差异，validation 有 4 个 baseline 和 3 个决策差异。这说明原缓存对应的是 BF16 骨干路径，FP32 结果只能作为精度敏感性诊断，不能混入 BF16 契约验收。

BF16 的完整 batch 生产 CE 与缓存选中成本仍有最大约 0.148 nat（test）和 0.195 nat（validation）的差异。对最大误差代表状态进一步重放发现：

- 从原缓存 h 独立重放 baseline R1/R4，可以复现原缓存成本到 BF16 舍入误差；
- 差异发生在共同 h 之后，完整 batch BF16 续算与逐状态反事实重放使用不同 kernel 形状，后续轮次出现数值漂移。

因此，**决策契约已通过，但不声称完整 batch logits/CE 与逐状态缓存标签位级一致。** 两种 CE 口径分开记录，未用平均 NLL 接近替代逐状态检查。

证据：

- `runs/followup_2026_10_26/r1_bfloat16_test/`
- `runs/followup_2026_10_26/r1_bfloat16_validation/`
- `runs/followup_2026_10_26/r1_float32_test/`
- `runs/followup_2026_10_26/r1_float32_validation/`
- `runs/followup_2026_10_26/r1_ce_diagnostics.json`

## 4. exact、price 与 registry

- exact/oracle 枚举增加明确配置 guard：只允许单 adaptive layer、`target_only=True` 且每序列恰好一个监督位置；不支持的联合路径树明确拒绝。
- 通用 router 评估现在把 `lambda_cost` 传给模型，训练与通用评估 price 一致。
- registry 将 `stop_reason` 和 `next_trigger` 保存在组件级，再按聚合状态选择，后续 finished 组件不会清空 blocked 原因。
- `goal_report` 支持显式 `--repo-root`，嵌套 registry 的仓库相对证据路径可正确解析。

实际对 `runs/repair_2026_10_25/registry.json` 重建报告，结果为 `missing_evidence=0`。相关辅助契约测试通过。

## 5. 训练数据与恢复契约

### 修复内容

- RNG 恢复前显式转回 CPU ByteTensor，兼容 `torch.load(map_location=cuda)`。
- 可见 GPU 数量与 checkpoint CUDA RNG 数量不一致时明确报错。
- 新数据指纹覆盖 train/validation 的 input IDs、segment IDs、doc-map、seq_len、文档池和 tokenizer，不再只哈希未截断文档池。
- 新 checkpoint 记录版本和 stream metadata：序列/样本数、batch size、sampler seed 等。
- batch size、seed、数据目录或序列数不支持的变化会拒绝“无缝续训”。
- 旧文档池哈希只在明确的 v1 兼容条件下迁移，并记录 migration；新保存统一使用 packed-stream 指纹。
- 小数据集不足一个完整 batch 时给出明确错误，不再除零。

### 真实 GPU 验收

使用小型合成 packed LM，在一张 A6000 上执行两个独立进程：

1. 首次训练 2 updates 并保存 checkpoint；
2. 新进程从 `latest.pt` 恢复到 4 updates。

恢复成功。随后再以 `map_location=cuda` 加载，确认 CPU RNG 被映射成 CUDA 张量后仍能正确恢复。checkpoint 为 version 2，update=4，stream metadata 为 16 sequences、batch=1、seed=17。

证据：

- `runs/followup_2026_10_26/gpu_lm_initial.log`
- `runs/followup_2026_10_26/gpu_lm_resumed.log`
- `runs/followup_2026_10_26/gpu_lm_resume/latest.pt`（本地保留，不进入 Git）

## 6. 测试与交付物

最终命令：

```bash
.venv/bin/python -m unittest discover -s tests -v \
  > runs/followup_2026_10_26/full_tests_final.log 2>&1
```

结果：**Ran 539 tests，OK (skipped=1)**，即 538 通过、1 跳过。

统一契约摘要：`runs/followup_2026_10_26/contract_checks.json`。

本轮新增/更新测试覆盖 R2 映射、二维候选、正式策略 trace、CE 形状、exact 配置、price、registry、packed 数据指纹和恢复不变量。`git diff --check` 对本轮代码与报告通过；审计归档 patch 原文不参与空白符改写。

## 7. 对旧报告的修订

- “R1 零决策差已经证明完整闭环”修订为：旧脚本证据不足；本轮正式入口在匹配 BF16/FP32 学生口径后通过决策闭环，CE 数值边界单列。
- “常量/分组简单规则已验证”撤回，正确重算属于方案 B。
- “二维/三维候选均已修复”现在有显式 pairs 和数值等价测试支持。
- “训练/评估 price 一致”现在覆盖通用 evaluate；任意联合 exact 树不作声明。
- “数据与恢复契约完整”收窄为：新 v2 检查点和真实单卡同配置恢复通过；改变卡数、batch 或训练流不自动宣称等价。
- 测试计数统一写“总数、通过数、跳过数”，不再把总项数写成通过数。

## 8. 尚未执行

- **方案 B：**正确 R2 廉价基线、解析随机期望和 R4 实际成本匹配区间。
- **方案 C：**未查看文档上的至少 1M 自然监督 token 固定深度确认。
- **方案 D：**candidate-aware advantage 的有界可学性实验。
- **方案 E：**新骨干、300M、末四层或真实 LM RL。

方案 A 的完成不表示动态停止、动作价值或专家报价出现新的正向科学证据。当前分支尚未提交或推送。
