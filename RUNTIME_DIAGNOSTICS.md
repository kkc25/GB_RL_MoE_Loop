# 100M 模型运行时诊断实验

## 目的与边界

本实验针对 `runs/integrity_v2/` 暴露的两个独立问题：普通 MoE 比参数匹配
dense 慢，以及 RL 在线路径还有额外成本。目标是定位可测量的执行开销，
不是扩大模型、继续训练或宣称 RL 获得质量收益。

保留现有 100M 配置：12 层、hidden 512、每层 8 个宽度 512 的专家、Top-2；
dense FFN 宽度为 4096。两者总参数接近，但 MoE 每 token 激活的 FFN 参数约为
dense 的四分之一。这个比例不适用于整个模型的 FLOP，更不能直接换算成延迟。

新的诊断入口独立于训练、缓存生成和正式确认。它不自动推广任何后端，不改写
历史 run，也不改变训练奖励的固定形状、head chunk 或数值契约。本次代码提交
不代表已经运行服务器/GPU 实验；实际 CUDA 测量必须由操作者另行启动。

## 对照必须回答的问题

1. **原生 MoE 的实现开销**：完全不经过 RL 在线封装，对照所有 12 层均为
   legacy 与所有 12 层均为 merged，保留原生单次输出层。再与相同输入的
   dense 原生路径比较。这补上旧四格实验缺少的无 RL、全层 merged 对照。
2. **在线封装成本**：保持骨干、数据、策略权重、最大调用轮数和数值配置一致，
   比较在线参考与候选。canonical 策略也走在线特征/输出层，因此它不是无
   RL 封装的原生模型对照。
3. **输出层分块与转换成本**：分别改变 head chunk 和实验性 BF16 权重缓存。
   先一次只改变一个因素，再看组合；不要把分块、缓存和专家后端的共同变化
   归因给其中某一项。
4. **开销位置**：分别观察 prefix、特征/路由、actor、专家执行和 head 的
   固定输入微基准，并在单独的 trace 中检查转换、索引、同步和矩阵乘法。

`merged` 是合并 Top-2 分配桶，不是 fused/grouped GEMM 内核。本实验不把
历史 `grouped_canonical`（计算所有专家）当作成熟稀疏 grouped GEMM 对照。
R1 不允许 STOP，因此 R1 的 v3 收益不能称为 STOP compaction 的收益。

## 数据、身份与结果保护

- 当前入口只使用配置中的 selection 开发集，不提供 calibration 切换参数，
  不读取训练集或 confirmation。禁止把 newtest 或配置中的
  confirmation split 用于该诊断；不得根据正式确认集挑选性能方案。
- 固定数据内容、序列数与顺序、batch、序列长度、骨干和 actor checkpoint
  SHA、精度、TF32 设置、后端及软件/GPU 环境。保存实际身份，而不是只保存
  操作者希望使用的参数。
- 在同一 GPU、同一输入工作负载下比较；检查其他 GPU 任务，避免把资源争用
  误认为实现退化。不要用瞬时功耗读数代替整个测量区间的利用率证据。
- 每次使用新的输出目录。保留原始计时和检查结果，不覆盖旧 profile 或实验
  checkpoint。先完成短 smoke，再考虑增加输入和重复次数。
- 本地未上传的骨干/actor 权重和语料仍是运行前提。仓库里的 SHA manifest
  不等于权重文件；不能把缺文件造成的失败当成模型或性能结果。

## 数值检查先于性能结论

BF16 权重缓存只用于候选推理路径，不能改写原模型参数、checkpoint 或默认
readout。缓存建立不计入稳态延迟，但必须明确这是预先准备的缓存；首次调用
成本和额外常驻显存不能被解释为不存在。

每项候选先与明确命名的参考比较：动作、STOP/轮数，以及相同样本的 FP32 CE
或 NLL 差异。改变 head chunk、专家桶形状或精度缓存可能改变 BF16 舍入；
不得通过放松门槛后只报告“通过”来掩盖差异。保留原始误差与所用门槛。

所有在线候选都做两重检查：相同 prefix/terminal 下的 head64 参考，以及固定
legacy prefix + v1 terminal + head64 的整体参考。最终资格要求两者均通过，
包括只更换 terminal 的 legacy-prefix 候选。原生模型比较只检查输出 CE/NLL；
其接口没有暴露完整路由轨迹，因此动作差字段为空，不表示已验证所有路由相同。

短 smoke 的通过仅说明所测样本通过门槛，不证明所有数据等价，更不证明质量
提升。若数值检查未通过，该候选不能被标为可部署或与参考等价；性能数值
即使可供排查，也不能单独支持推广。后续扩大验证范围仍只使用开发集，正式
确认必须另行冻结 candidate/scope，并遵循 [RUN_CONTRACT.md](RUN_CONTRACT.md)。

## 如何解释计时

- 原生/在线端到端均应明确测到完整 logits，是否包含 CE 必须单独标注。
  不把一个输出 CE 的路径与另一个只输出 logits 的路径直接比较。
- 固定输入的组件微基准**不可相加成端到端时间**：中间张量复用、缓存、
  kernel 排队和分配行为与完整执行不同，也不能用两个不匹配的组件时间
  相减来声称得到纯 actor 开销。
- torch.profiler trace 与正式延迟测量分开运行。trace 用于定位算子与同步，
  其运行时间不能充当不带 instrumentation 的性能结果。
- 报告 CUDA-event 与同步 wall time、原始重复测量、分位数、吞吐与工作负载。
  分清预热、缓存构建、runtime 切换与实际计时区间。
- allocator 的 reserved/peak 往往包含共享模型、预先缓存和其他候选留下的
  分配；它不是某个候选独占显存。解释缓存方案时同时考虑额外常驻权重。

## 运行方法

入口：`python -m experiments.rl_lm_diagnose_runtime --help`。下面命令是给有
数据和 checkpoint 的服务器操作者使用的示例，**本次提交没有执行这些 GPU
命令**。从仓库根目录运行，按实际服务器调整设备和输入路径。actor 使用同一
阶段的 selected checkpoint，不能随意把 R2 权重配到 R1 配置。

最小 smoke（默认 2 个序列、batch 1、1 次预热、1 轮×3 次计时）：

```bash
python -m experiments.rl_lm_diagnose_runtime \
  --config configs/rl_lm_v2/pair_r1.json \
  --device cuda:0 \
  --dense-config runs/next_2026_10_26/baseline_100m/dense_parameter.json \
  --dense-checkpoint runs/next_2026_10_26/baseline_100m/dense_parameter/latest.pt \
  --actor runs/integrity_v2/pair_r1_seed17/checkpoint_best.pt \
  --output-dir runs/runtime_diagnostics/smoke_001
```

没有 actor 文件时可省略 `--actor`，仍测原生 legacy、原生全层 merged、dense
及 canonical 在线控制；这不代表测过 learned policy。可用 `--checkpoint`、
`--corpus-dir`、`--tokenizer` 覆盖服务器上的输入路径，但不能绕过身份检查。

smoke 正常后，再明确开启缓存候选、扩展样本和重复次数：

```bash
python -m experiments.rl_lm_diagnose_runtime \
  --config configs/rl_lm_v2/pair_r1.json \
  --device cuda:0 \
  --dense-config runs/next_2026_10_26/baseline_100m/dense_parameter.json \
  --dense-checkpoint runs/next_2026_10_26/baseline_100m/dense_parameter/latest.pt \
  --actor runs/integrity_v2/pair_r1_seed17/checkpoint_best.pt \
  --sequences 16 --batches 1,8 \
  --head-chunks 64,256,1024,full --cached-bf16-head \
  --warmup 10 --repeats 5 --measurements 50 \
  --output-dir runs/runtime_diagnostics/expanded_001
```

序列数必须能被每个 batch 大小整除。`full` 表示当前 batch 的全部 token 行，
不是改变输入序列。候选数量会随 prefix、policy、chunk 和缓存开关相乘；
增大重复次数前先查看报告中的 `variant_count` 和 `planned_e2e_timed_calls`。
该计数不包含额外数值验证、预热和组件测量，不能用它直接推算总账单。
扩大版是建议协议，不是统计充分性保证；实际耗时/显存以服务器测量为准。

单独采集一个候选的 trace，例如：

```bash
python -m experiments.rl_lm_diagnose_runtime \
  --config configs/rl_lm_v2/pair_r1.json \
  --device cuda:0 \
  --dense-config runs/next_2026_10_26/baseline_100m/dense_parameter.json \
  --dense-checkpoint runs/next_2026_10_26/baseline_100m/dense_parameter/latest.pt \
  --actor runs/integrity_v2/pair_r1_seed17/checkpoint_best.pt \
  --cached-bf16-head --trace \
  --trace-variant learned_merged_head64_cached_bf16 \
  --output-dir runs/runtime_diagnostics/trace_001
```

`--trace-variant` 必须是本次生成的完整候选名，且必须与 `--trace` 一起使用。
不指定该名称的 `--trace` 会为各 E2E 候选采集 trace，文件数量与额外运行量
都会增加。trace 采集与计时区间分离；上例仍会先完成短诊断，不能视为零成本。
输出目录一旦存在即拒绝使用，包括空目录和未完成目录；失败后排查原因，
再选择新目录，不要删掉失败证据来伪装成一次成功运行。

## 报告阅读顺序

主报告为 `diagnostic_report.json`，schema 为 `rl_lm_runtime_diagnostic_v1`。
先看 `status` 与输入/源码身份，再看每个 workload 的数值验证和参考，再看
`end_to_end`；最后用 `components` 和可选 trace 解释现象。保留 matched-prefix
head64 与 combined-runtime 的不同参考范围，不要只挑一个通过项。

默认绝对 NLL 与 CE 平均绝对差门槛均为 `1e-4`，只是可配置的本地筛查规则，
不是已批准的研究等价标准。未通过筛查的候选仍可能保留诊断计时，但必须带着
失败标记解释，不能据此推广。额外 BF16 head 缓存字节单独记录；无论哪个候选
正在计时，共享模型/缓存均可能常驻，不把 allocator 数字解读成独占显存。

## 决策顺序

1. 先排除输入、权重、有效后端或计时范围不一致。
2. 若原生全层 merged 已明显改善，说明非 RL 的执行结构值得继续优化；
   仍不能把它描述为 learned policy 的贡献。
3. 若 head 分块/缓存变化明显改善同一在线路径，结合独立 trace 检验转换与
   小 GEMM 是否减少。不能仅凭总延迟下降确定单个算子的贡献比例。
4. 再考虑真正 grouped/fused 专家内核。若涉及新的数值行为，保留旧参考并
   单独验证；不要直接替换训练奖励后端。
5. 只有这些损耗被量化后，才考虑在同等预算下做宽度、模型规模和 batch 扫描。
   100M 结果不能推出更大模型一定更快，也不能推出 MoE 理论上不可行。

这些是诊断与决策规则，不是“修复以后必然超过 dense”的承诺。研究质量
收益、部署延迟和实验完整性分别报告。
