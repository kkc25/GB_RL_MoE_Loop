# Canonical32 R2 Precision-screen 报告

## 执行身份与范围

- PR：#19；固定代码：`fea92b07288f4b1dedaea4f8b480ecc1578934d6`。
- 配置：`config.json`；配置 SHA256：`4fa1b23d275f32ec03e0d25b3a035d8a3d4819e84b5b48e8c1af5c0a6c40cf44`。
- 模型：本地 Mixtral-8x7B-v0.1；运行时为 Python 3.12.15、torch 2.12.1+cu130、transformers 5.15.1，物理 GPU 3/4/5/7（4 张 NVIDIA RTX A6000）。
- 严格保持 `atol=rtol=1e-4`；只执行 `prepare` 和 `precision-screen`，未执行 cache、fit、正式 preflight 或 100k 训练，未访问 test 样本内容。

## 软件验证

- 依赖检查与编译检查通过。
- 专项测试：53 项，51 通过、2 项 GPU 条件测试跳过。
- 完整 CPU 测试：1,336 项，1,321 通过、15 项跳过；完整日志见 `full_tests.log`。CPU 测试不代替后续真实 GPU 筛查。

## 数据核对

重新冻结同一批 8,000 篇候选语料。新旧 `freeze.json` 及所有角色清单逐字段一致，`freeze_sha256=e5ffd918ee9c0e1dafec89436e2e39613ed9e922de94543cbe15ca9253790ba2`。输入 8,000 行，去除 1 个重复行，7,999 个唯一规范化内容；development 和 test 身份未改变。

## 数值结果

`precision-screen` 完成 4 文档 × 2 幅度 × 3 模式，共 24/24 case。模式完成状态为：

- `fp32_canonical32`：8/8 case 通过，`numerical_checks_passed.fp32_canonical32=true`。
- `fp32_experts`：0/8 case 通过。
- `native_bf16`：0/8 case 通过。

Canonical32 在更新前后的 STOP/all/even/odd/first/last singleton 检查中，compact 与 compute-all-then-mask 的 hidden 和 CE 最大差均为 `0.0`，没有超差元素。更新前后四类梯度比较全部通过；梯度最大绝对差为 `1.1920928955078125e-07`，低于既定容差。有效输入改变、非零 up 梯度、一次临时更新、STOP 零额外专家、主干无梯度和参数版本不变均通过。

原生 prefix/CE 保真检查也全部为零差，说明候选没有通过重定义原生基线隐藏差异。与 PR #18 相比，固定 32 行协议消除了此前 singleton 和混合掩码的剩余数值不一致。

## 时间与资源

- Canonical32 每例 3.414–3.694 秒，均值 3.574 秒；native BF16 均值 1.227 秒，普通 FP32 experts 均值 2.208 秒。
- 这些时间包含多次正确性检查与反传，不是推理吞吐。额外路径诊断仅在失败时触发；Canonical32 没有触发该诊断。
- Canonical32 记录的单设备峰值上限为 allocated 24.368 GiB、reserved 28.711 GiB，低于 44 GiB 配置预算，无 OOM。
- 固定分块带来真实额外工作：每个 case 的逻辑专家分配为 13,152–15,360 行，补齐后的物理执行为 18,752–21,184 行，即约 1.308–1.426 倍。不能把逻辑稀疏 token 比例直接解释为实际省算力。

## 结论与下一步

固定 32 行 FP32 候选在本轮严格数值筛查中通过，解决了 PR #18 遗留的完整 parity 阻塞；这不是训练质量改善或 SOTA 的证据，也不是 100k 训练完成证书。

按 PR #19 的执行边界，本轮在筛查后停止。下一步应在同一固定代码、配置和冻结数据上独立运行正式 preflight；只有该关卡通过并完成单独审阅后，才决定是否进入原计划 100k 阶段。
