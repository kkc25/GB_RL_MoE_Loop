# Canonical32 首轮 100k 执行与中断报告

## 结论

本轮已完成真实 CUDA 专项测试和独立正式 preflight，但在 `cache` 开始前的 preflight 身份复核处失败。**没有执行任何 100k 训练、配套对照或开发集评估，也没有进入 300k、1M 或最终 test。**

失败不是数值、梯度、STOP、主干冻结、OOM 或总时限问题。根因是 `require_preflight` 将内存中的模型身份直接同 JSON 回读身份比较，`model_config.id2label` 的整数键在 JSON 中变为字符串，造成相同身份被误判为不匹配。

目前不能判断是否值得扩大训练档位。数值关卡通过只支持“修复后重新执行 100k”，不支持扩大到 300k，也不提供任何质量收益或 SOTA 结论。

## 固定执行身份

- 原固定代码：`fea92b07288f4b1dedaea4f8b480ecc1578934d6`，独立 worktree 未修改。
- 模式：`fp32_canonical32`；未切换到 `native_bf16` 或普通 `fp32_experts`。
- 配置：`config.json`，SHA256 `4fa1b23d275f32ec03e0d25b3a035d8a3d4819e84b5b48e8c1af5c0a6c40cf44`。
- 冻结身份：`e5ffd918ee9c0e1dafec89436e2e39613ed9e922de94543cbe15ca9253790ba2`；配置、数据路径、模型、种子、幅度、超参数及 `atol=rtol=1e-4` 均未修改。
- 运行环境：Python 3.12.15、torch 2.12.1+cu130、transformers 5.15.1，`CUDA_VISIBLE_DEVICES=3,4,5,7`，四张 RTX A6000。
- 不可变总截止时间由 `stage_receipts.json` 的创建时间与 86,400 秒共同确定，为北京时间 2026-11-01 13:12:31。执行时尚未到期，未修改时间戳或预算。

## 1. 真实 CUDA 测试

`cuda_tests.log` 显示 CUDA 可用、四张 A6000 可见，以下两项均实际执行为 `ok`，不是 skipped：

- `tests.test_canonical_execution.CanonicalExecutionTests.test_cuda_bf16_singleton_permutation`
- `tests.test_fp32_experts.PrecisionScreenTests.test_cuda_bf16_frozen_linear_backward`

合计 `Ran 2 tests in 0.648s`，结果 `OK`。

## 2. 独立正式 preflight

`preflight.json` 和 `preflight_progress.json` 均为 `status=complete`，共 8/8 案例通过：

- 更新前后 conditional parity 与梯度检查通过；保持既定容差。
- STOP 零额外专家通过。
- 主干无梯度且参数版本不变。
- 核心检查耗时 31.161 秒；含身份哈希和模型加载的 stage wall 为 148.241 秒。
- 逻辑卡 0/1/2/3 的峰值 allocated 分别约为 23.62/26.08/26.08/18.21 GiB，均低于 44 GiB 预算。

完整逐案例数据、模型内容哈希、设备映射和每卡峰值见 `preflight.json`；加载与命令输出见 `preflight.log`。

## 3. Cache 失败与停止边界

`cache` 在 `require_preflight` 抛出：

```text
ValueError: successful source/model/data/runtime-bound preflight required
```

失败 stage wall 为 115.145 秒，完整 traceback 见 `cache.log`，四次阶段尝试见 `stage_receipts_100k_attempt.json`。

没有 `cache_complete.json`、cache 目录、cells 目录或训练 checkpoint。故以下训练记录不存在，不能按零成本或已完成处理：实际训练 token、更新次数、训练 loss、训练梯度、训练各卡峰值、训练逻辑/补齐专家工作量、配套对照及开发集评估。

## 4. 身份诊断

`identity_diagnostic.json` 与 `formal_scope_identity_diagnostic.log` 记录了正式 `precision_scope(False)` 下的只读验证：

- 代码身份和冻结数据身份一致，preflight 状态完整。
- 原始内存对象与 JSON 回读对象不相等。
- 对实际模型身份进行同样的 JSON 规范化后，与 preflight 完全一致。
- 具体字段为 `model_config.id2label`：内存键为整数 `0/1`，JSON 键为字符串 `"0"/"1"`。

两份未使用正式精度作用域的早期诊断日志也原样保留，但不用于根因结论；不能将其中出现的 `cudnn_tf32=true` 误认为正式 cache 的设置。

## 5. 独立修复与验证

未修改原固定代码或失败运行。独立分支 `fix/mixtral-preflight-json-identity` 已推送，提交：

`510194a3dc95f8c8b5df0d50ea91ae299f78243b`

修改仅在 `experiments/mixtral_loop_scale.py` 的 `require_preflight` 中先对实际模型身份做同样的 JSON 规范化，再执行严格相等比较；不忽略任何运行时字段，不放宽数值容差。新增回归测试同时证明：

1. 相同 JSON 规范化身份可以通过；
2. 真实 `cudnn_tf32` 变化仍被拒绝。

验证结果：

- 定向回归：1 项通过。
- 完整 CPU 回归：1,337 项，1,322 通过、15 跳过，见 `preflight_json_fix_tests.log`。
- 使用真实 Mixtral 重新生成身份并调用修复后的 `require_preflight`：`PASS`，见 `preflight_json_fix_real_model_verification.log`。该验证没有执行 cache、fit 或修改原阶段回执。
- 修复分支未自动合并成 PR，也未用于恢复本轮失败运行。

## 6. 是否扩大与下一步

**不建议扩大训练。** 本轮没有任何训练质量或开发集收益证据，300k 不应启动。

可保留的积极证据仅是 canonical32 的严格数值筛查和正式 preflight 均通过。若继续研究，应使用修复后的固定提交建立新的运行身份，按协议重新 prepare/preflight，并只重试首个 100k 三 cell；不能把旧 preflight 直接冒充新代码的正式凭据，也不能跳过这些关卡。
