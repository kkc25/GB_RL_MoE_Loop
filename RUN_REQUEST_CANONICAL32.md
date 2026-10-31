# 单 token 与批量执行：固定 32 行候选协议

## 依据与边界

代码基线 PR #18：`4628a0e03074a8993fbeb43f87110b62629cdd9f`。结果证据 main：`a4b71694a9019c3ac8e5df70e9642b5e1bed9676`。上一轮普通 FP32 expert 已使两种交替掩码的梯度前后均通过，剩余梯度失败集中单 token；hidden/CE 仍未全部通过。没有证据把剩余根因单独锁定为 norm、router、残差出口或 head。

此次是**可执行的候选修复 + 有界真实路径定位**，不是预先宣布解决 GPU 问题。目标是消除行数切换带来的算子形状变化，同时保留真正只对选中 token 计算第二轮的机制。固定形状仍不保证 GPU 在不同行内位置、后端或参数梯度求和顺序下逐 bit 相同，必须实测。

## 新模式 fp32_canonical32

- 固定 TILE=32，由模式标识和代码 SHA 共同绑定，不按结果调 tile 大小。
- 仅在原 native prefix 后的 token-local R2 中，对 active 行分块并补零至 32：post-attention norm、correction、Top-2 router 都执行同一形状。
- 每个实际使用的专家桶独立分块至 32。packed gate/up、SiLU、乘法、down、路由加权均在完整块内执行；裁掉填充行后，按原专家升序、slot-major 顺序回填。每 token 在同一专家内最多一项，不把不同专家的贡献混在一个无序归约中。
- 保留 PR #18 的 FP32 专家/适配器一阶 forward/backward 和单投影临时转换，不保存整套 FP32 专家 bank。norm/router/head 保留原模型权重 dtype；BF16 residual 出口也保留。
- 输出 norm/head 也固定 32 行。同模式的 native baseline、STOP、候选训练/评估和 gate gain 使用相同 readout；显式不匹配精度模式的 branch/config 被拒绝。
- **原 pretrained 输出仍单独比较，原生保真检查不移除。** 如果固定分块 readout 改变了原生模型损失并超出原容差，preflight 仍失败，不能通过重定义 baseline 把它藏起来。
- 默认 native_bf16、旧 fp32_experts 数值路径保持；新配置是新的数值协议，旧配置/旧 checkpoint/旧 preflight 不隐式迁移。

## 稀疏性与成本必须如实解释

不把未选中 token 放入生产 compact 专家计算。只补齐 active 行和实际路由的专家桶；dense 路径仍仅作为计算全部后掩码的参照。

例如只选中一个 token、路由到两位专家：逻辑专家分配为 2，但实际执行 2×32=64 行，而不是 2 行。STOP 为零额外专家；对于大量选中 token，实际成本是 sum_e ceil(n_e/32)×32。它有固定分块开销，不能按理想 token 比例宣称省算力。

work 中同时记录各阶段 `logical_row_attempts`、`padded_row_attempts`、`tile_attempts`、`tile_completed`，旧逻辑 token 计数继续保留。head 分配检查按累计补齐行数×vocab 约束；这不是整轮显存峰值保证，输出拼接、激活、梯度和 allocator cache 仍占内存。

算子执行临时禁用 autocast、使用最高 FP32 matmul 精度并恢复原设置；冻结投影的 backward 仍有独立精度保护。原 BF16 reduction 设置不擅自改变，记录在运行时身份中。必须串行执行，勿在同一进程并发切换全局精度设置。

## 剩余失败的真实路径观察

新可选 trace 直接采集实际执行产生的：normalized_input、correction_fp32、router logits/weights/pairs、expert input/output、FP32 residual、cast hidden。它支持重复 tile 调用和自定义 FP32 helper，不使用只能捕获一次 native expert 调用的旧 hooks 冒充新模式证据。

对失败的 first/last singleton，整轮共享最多 12 次探针：
1. 在同一适配器状态下运行 dense/compact 实际路径，比较同一选中行；
2. 另运行不带观察器的实际路径，确认完整输出一致；观察器不一致时不输出首次定位结论；
3. 固定相同 dense hidden，分别整批与单行通过该模式的 readout，比较最终 norm、logits、CE，以隔离输出端形状影响。

只保存聚合差异和形状/dtype，不导出 token 文本、hidden 张量或权重。`first_*` 是最早观察到的差异位置，不等于底层最终根因。预算用尽明确标为 skipped，不能说所有失败都做过逐层定位。异常保留已完成 conditional 证据、当前 mask 和 work；继续抛错而不签发通过。

## 三组筛查与验收

采用新配置时，precision-screen 在同一批 4 个 adapter-train 文档、两种幅度上对照 native_bf16、fp32_experts、fp32_canonical32，共 24 cases；组序循环轮换。每组各自一次丢弃的临时更新，不是正式训练，各组更新后参数可能不同，不当作纯精度下的相同 checkpoint 对比。

保留原 `atol=rtol=1e-4`：
- 更新前后 STOP/all/even/odd/first/last singleton 的 hidden/CE；
- 更新前后 even/odd/first/last singleton 的相同标量损失及两个适配器因子的梯度；
- 实际有效输入变化、有限且非零 up 梯度、所有主干无梯度且参数版本不变；
- 原生 prefix/CE 保真检查，候选不能绕过梯度检查。

TILE 边界、所有行内位置、任意重排/分段、零/非零 correction、选中与未选中独立性，以及真实补齐成本在 CPU 回归测试中覆盖；CUDA 专项测试必须在服务器实际执行，不用 CPU 结果代替。

时间与内存：每例同步所有可见 GPU，报告完整检查 seconds、单独 path_diagnostic_seconds 和扣除此探针后的 seconds_excluding_path_diagnostics。探针按失败触发，不能直接比较含不等诊断工作的总耗时。扣除后仍是多次正确性检查和反传，**不是推理吞吐**。峰 allocated/reserved 与起始占用同时回传；`peak_memory_includes_path_diagnostics` 标明额外观察器可能影响峰值。

共同遵守原 300 秒筛查预算与不可变 suite 截止时间，操作边界检查而非抢占 GPU kernel。异常/非有限值/主干变化/OOM/超时停止并保留输出；不重置预算重跑未知工作。

screen 的 status=complete 只表示所有案例收集完成。查看 `numerical_checks_passed.fp32_canonical32`，不是旧模式字段。**screen 永不创建 preflight.json，不会自动启动 cache/fit。** 全部数值及资源条件满足后，审阅并独立运行同配置正式 preflight；其通过后可推进原计划 100k 阶段，不必因无关的位级差异无限新增诊断，但不能靠放宽本轮门槛宣布通过。

## 执行请求

固定交付 SHA，新 worktree/new run_root/new freeze_dir。沿用已存在的同一模型/tokenizer 路径、8,000 篇候选语料、历史排除、设备预算与数据划分；先确认路径存在，**不要直接使用示例占位路径，也不要下载模型**。最终 test 质量仍封存，重新 prepare 后逐角色核对冻结文件与上一轮一致。

```bash
set -euo pipefail
CODE_SHA=REPLACE_WITH_DELIVERED_FULL_40_HEX_SHA
git fetch origin fix/mixtral-canonical32-r2
git worktree add --detach ../mixtral-canonical-${CODE_SHA:0:12} "$CODE_SHA"
cd ../mixtral-canonical-${CODE_SHA:0:12}
test "$(git rev-parse HEAD)" = "$CODE_SHA"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pip check
python -m unittest tests.test_canonical_execution tests.test_fp32_experts tests.test_loop_parity_diagnostics -v
python -m unittest discover -s tests -t . -v
python -m compileall -q mixtral_lab experiments
# 从 loop_scale_canonical32.json 复制到 worktree 外，落实原输入和全新输出路径。
CONFIG=/absolute/path/to/new-canonical32-config.json
python -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage prepare
python -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage precision-screen
```

到此停止并回传完整 SHA、配置/环境身份、screen/progress、stage_receipts、运行日志、冻结核对结果。既有输出不覆盖，不自动合并 PR，不串联正式训练。GPU 通过与资源可行性尚未验证。
