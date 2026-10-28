# 为什么当前 MoE 比 dense 慢：代码与日志核查

核查对象：`main@9ad0825`、`runs/integrity_v2/profile_factorial_v4/profile_report.json`、模型与在线推理实现。

## 结论

慢的主要原因不是 MoE 的激活计算量更大，而是当前实现用大量小 GEMM、Python 专家循环和动态路由/拷贝代替了 dense 的少量大 GEMM；RL 在线路径还增加 actor、特征构造和分块 head。稀疏 FLOP 优势没有转化为 GPU 墙钟时间优势。

## 1. 两个模型是参数匹配，不是激活 FLOP 匹配

- MoE：104,919,552 参数，12 层全部为 8 专家 Top-2，单个专家 intermediate=512。
- Dense：104,870,400 参数，12 层 dense SwiGLU，intermediate=4096。
- 配置：`runs/next_2026_10_26/baseline_100m/{loop_r1,dense_parameter}.json`。

Dense 每 token 的 FFN 激活 FLOP 约为 MoE Top-2 的 4 倍，但 dense 的参数矩阵能形成适合 GPU Tensor Core 的大 GEMM。参数量相近并不意味着延迟相近。

## 2. 旧 MoE dispatch 产生大量小矩阵运算

`model/moe.py:29-41` 的 `ExpertPool.forward` 对每个专家执行 Python 循环，每个专家有 `nonzero`、`index_select`、3 个线性层和 `index_copy_`。

`model/top2.py:140-143` 的旧 Top-2 路径对两个 slot 分别调用一次 `ExpertPool`。因此每层最多是 2×8 个专家组、48 次专家 GEMM；12 层约 576 次专家 GEMM，还未计入路由、归一化、attention 和拷贝。Dense 每层只有 3 次大 FFN GEMM，12 层共 36 次。

`merged` prefix 将两个 slot 合并到同一专家桶（`model/top2.py:22-38`），把每层专家组数从最多 16 降为 8，这是四格 profile 中 merged prefix 明显更快的主要原因，但仍保留 Python/nonzero 和可变形状小 GEMM。

## 3. RL 在线路径还有额外开销

- `rl_lm/inference.py:126-197`：prefix 后将 hidden 转 FP32、构造观察、actor 推理、joint argmax、执行末层，再经过分块 head。
- `rl_lm/executor.py:129-155`：构造 `n/n0`、中心化的 gate、previous-pair multi-hot、drift 和拼接特征。
- `rl_lm/executor.py:298-306`：旧 terminal v1 仍按 slot×expert 双重 Python 循环。
- `rl_lm/readout.py:120-137`：在线 full-logits 路径使用 `head_chunk=64`，1024 行需要 16 次 head GEMM；dense 原生 forward 使用一次完整 head。

Profile 的组件日志也支持这一判断：batch 1 时 `legacy_gate_top2` 仅 0.257 ms，而 `rl_features_actor_joint` 为 2.650 ms；末层 executor 约 3–7 ms，但完整在线模型为 42–62 ms，说明开销分布在所有 prefix 层、actor/特征和 head，而不只是末层专家。

## 4. 日志中的实际差距

Batch 1、S=1024：

- 原始 MoE R1：48.983 ms；dense：13.357 ms，MoE 约 3.67× 慢。
- 最优 learned R1（merged prefix + active v3）：42.762 ms，仍约 3.20× 慢于 dense。
- merged+v3 将 learned R1 从旧默认 56.185 ms 降至 42.762 ms，证明 dispatch 和 STOP compaction 有效，但不足以消除实现开销。

Batch 8、S=1024：原始 MoE R1 67.628 ms、dense 46.383 ms，差距缩小到约 1.46×。这符合小 batch 下 kernel launch/Python/动态路由开销占比更高的判断。

Profile 明确记录未在计时区使用 torch.compile 或 profiler，采用 CUDA event + 同步墙钟、交错重复测量；dense 与 MoE 都是完整输入到 logits、排除 CE，因此差距不是“一个算 CE、另一个不算”造成的。

## 5. 后续优化优先级

1. 用 grouped/batched expert GEMM 或成熟 MoE kernel 取代专家 Python 循环和小 GEMM；这是最大瓶颈。
2. 在完整 prefix 上也使用真正的 grouped dispatch，并验证 BF16 数值一致性；不能只依据 CPU parity。
3. 推理部署使用一次完整 head GEMM，或在保持数值契约的前提下增大 head chunk；训练/正式有界评估仍保留现有流式路径。
4. 精简在线 actor 特征（避免不必要的 FP32 拼接、norm 和重复状态拷贝），再评估 CUDA Graph/torch.compile；先固定数值与动作契约。
5. 分别报告 prefix、terminal、actor、head 的耗时，避免用末层 1.78× 加速代替全模型加速。

当前结果不能解释为“MoE 理论上必然比 dense 慢”，只能解释为：**这个参数匹配、未经高度优化的 MoE/RL 实现，在 RTX A6000 和当前 batch/序列工作负载上比 dense 慢。**
