# 专家逐算子与精度对照：服务器复验请求

## 背景与本次目的

代码基线为 PR #16 的 `31b796274720f47a437c8f6805726ac953b657f6`，新 PR 叠在其分支上。结果证据位于 main `a75c6b306ead1ef142832e0c7e7bb1987f46f0db`；main 尚未合并这套实验代码，不能误用 main 的旧执行器。新提交自带 v2 与上一版诊断代码，无须先合并任何 PR。上轮 4 篇文档 × 2 幅度、更新前后共 16 份诊断显示，固定专家输入、Top-2 编号与权重后，仅改变调用行数仍有差异。它定位到专家模块，但没有证明具体 GPU 内核有错误。

本次不修改生产 forward、专家后端、训练目标、数据划分或 `atol=rtol=1e-4`。不使用全量计算冒充稀疏执行，不以放宽阈值使训练启动。没有宣称性能、显存或质量改善。

## 观察的实际运算

仅支持固定依赖中的 Transformers MixtralExperts eager + SiLU、BF16/FP32、唯一 Top-2。用 TorchDispatchMode 观察真实专家 forward：

1. 专家按原实现顺序处理；聚集行按 top-k slot 优先、token 次序对齐。
2. 记录 packed gate/up 矩阵乘法、SiLU、gate × up、down 矩阵乘法、路由权重乘法、回转 BF16、累计 index_add。
3. **实际调用仍使用完整 dense/compact 行数**。仅中间观测和 FP32 参考采样前 min(8, selected_count) 个选中 token，覆盖这些 token 的两位专家；不是把两个调用都缩成八行。
4. 检查观测运行的完整输出与此前未观测输出逐元素一致，同时核对最后累计结果与模块输出。若不一致，标为 `observer_mismatch`，不给出首个差异算子的定位结论。另一种算子序列直接失败，不猜测融合算子的含义。
5. 每专家的 `first_exact_difference` / `first_tolerance_failure` 只表示抽样观察到的位置，不保证其他行没有更早差异。`accumulation` 是累计值，可继承先前专家的误差，不应当作该专家 index_add 自身出错的证明。

## 高精度参考与开关实验

- FP32 参考升精度的是**同一份已经量化的权重与输入**，不恢复模型量化前的信息；固定相同路由与权重，FP32 连续计算、累计，不插入 BF16 中间舍入。
- dense 与 compact 分别相对这一公共参考报告偏差；它不是精确真值，也不是新的通过标准。FP32 抽样行数本身与原执行不同，仍可能有浮点批形状误差。不能仅因更接近 FP32 就宣布生产修复成立。
- FP32 参考临时禁用 TF32。原生产配置早已关闭 TF32，这不是新发现的修复。
- CUDA BF16 下，另做一对相同输入/路由的真实 expert forward，临时关闭 `allow_bf16_reduced_precision_reduction`，比较两边与参考及原输出。记录原设置；如果原本已关闭，必须注明没有改变。
- CPU 不执行 CUDA mitigation，标记 `not_cuda_bf16_tested`。全局精度/reduction 设置在正常和异常退出均恢复；必须串行执行，不与其他线程的 forward 共用这些全局设置。

## 有界成本与失败处理

- 整轮 preflight **共享最多两次**深入诊断额度，通常是第一篇文档、幅度 1 的更新前后首个失败混合掩码。后续仍保留旧逐层诊断，深诊断标记额度用尽；若早期没有失败，额度会留给后续首两个失败状态。不能把未深查的其他 case 说成已完成逐算子定位。
- 每次额外两次带观察器的真实专家 forward；CUDA BF16 再额外两次关闭 reduction 的 forward。FP32 参考只处理最多八个 token 的实际路由专家。每项有独立 attempted/completed/token/projection 计数。
- 只转换一个专家的一次投影为 FP32，释放后再转换下一次，不升精度整个专家库。标准 Mixtral 的 gate/up 投影约 448 MiB、down 约 224 MiB；这是单次权重临时量，不是整轮峰值保证。模型原分片和显存限制不变，OOM 即停止，不迁移模型或自动降规格。
- 深入诊断截止时间取原 300 秒 preflight 预算与不可变 suite 截止时间的较早者，算子边界检查。已发出的 GPU kernel、同步和分配不可抢占，不能宣称严格硬中断；实际耗时仍需回传。
- 异常/超时保留已完成 layer/head 证据、深诊断阶段、完成计数和 work，写入 `preflight_progress.json` 的 `diagnostic_evidence`；并继续抛错。当前探针未完成的运算不冒充完成。历史 case 结果仍保留。
- 原 strict parity gate 独立于探针；失败仍无 `preflight.json`，不允许 cache/fit。即使 reduction 诊断改善，也不自动启用为生产设置。

## 固定版本运行

沿用上轮本地候选语料、历史排除输入、固定模型快照、四张 A6000、依赖和研究参数；不下载权重。只改新 run_root / freeze_dir，保留旧失败目录和凭据。配置置于 worktree 外，执行前后保留配置哈希。重新 prepare 并核对各角色清单与旧冻结内容一致，不评估最终 test。

```bash
set -euo pipefail
CODE_SHA=REPLACE_WITH_DELIVERED_FULL_40_HEX_SHA
git fetch origin fix/mixtral-expert-precision-diagnostics
git worktree add --detach ../mixtral-expert-${CODE_SHA:0:12} "$CODE_SHA"
cd ../mixtral-expert-${CODE_SHA:0:12}
test "$(git rev-parse HEAD)" = "$CODE_SHA"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pip check
python -m unittest tests.test_expert_precision_diagnostics tests.test_loop_parity_diagnostics -v
python -m unittest discover -s tests -t . -v
python -m compileall -q mixtral_lab experiments
CONFIG=/absolute/path/to/new-expert-precision-config.json
python -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage prepare
python -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage preflight
```

命令失败不加 `|| true` 继续训练，不复用失败输出目录。该过程无自动 GPU 执行授权扩张；由已有服务器操作者按此请求决定执行，不自动合并 PR。

## 回传与审阅

回传完整代码 SHA、环境/模型身份、结果提交 SHA、stage_receipts、preflight_progress、运行日志、耗时与峰显存（缺失需注明）。深入诊断在失败混合掩码的 `layer_diagnostic.expert_precision` 中。先看 observer 校验，再看抽样各阶段、FP32 偏差和 reduction 的真实设置与结果。失败中断看顶层 `diagnostic_evidence`。

下一步根据证据选生产修复和验收设计；本补丁本身不提供 GPU 根因已解决、训练可继续或全数据泛化改善的结论。CPU 注入误差测试只能证明诊断定位与保护流程能工作。
