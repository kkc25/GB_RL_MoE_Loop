# 混合门执行一致性：诊断与验收修补

## 基线、问题与边界

代码基线 `51cc1afcb63958f8575cf0a041a42a6f2968ccb2`（PR #15）。失败证据在 main 提交 `636023994099ac37ef9965cb9588eef2cb92bed8` 的 `runs/loop_scale_v2/loop_scale_v2_seed17_20261030T104832Z/`。该 run 执行 `ec7f78a79ef6b9bbaaa098b2da76f3d3a5f2b2a1`；它与代码基线的差异仅为文档和状态。失败数据和旧分支一律保留。

当前证据只覆盖第一篇文档、幅度 1 的更新后混合门。相同数学表达式在不同批次形状的浮点计算中不保证一致，但这不是对当前失败根因的证明。参考 [PyTorch 2.12 数值说明](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)。

竞争解释：归一化/输入适配器产生批次相关差异；路由 Top-2 或权重改变；固定路由下专家 GEMM 仍产生差异；输出头批次形状差异；掩码/回填语义错误。此次不预设其中任何一种正确，不降低容差，不更换计算后端。

**本补丁修复的是诊断与验收覆盖缺口，不是声称已验证了 GPU 数值根因修复。** CPU 人工注入批次相关误差只能验证诊断能发现问题。

## 改动

1. preflight 在临时 optimizer 更新前、后都检查条件执行，不再拿零初始化通过代替训练后通过。
2. 固定六种掩码：STOP/all/even/odd/singleton-first/singleton-last。单位置及交替位置只选有效监督位置；all 保留整文档形状以对照实际实现。
3. 对每文档每参数状态的首个失败混合掩码，hooks 观察实际 forward 中的 norm、FP32 correction、router logits/weights/pairs、expert input/output 和回填后 hidden。只保存聚合统计，不导出隐藏张量或模型权重。
4. 专家隔离对照固定 dense 路径的相同输入、专家编号及权重，仅改变专家调用的行数；输出头隔离对照固定 hidden，仅改变 readout 行数。它们仅诊断，不是生产策略，也不作为通过标准的替代。
5. 数值失败不再掩盖其余三篇文档和幅度 0.1 的覆盖。在原 300 秒核心预算内收集最多 4×2 个 case；异常、主干变化、非有限值、预算不足仍停止。未完成项不能说成通过。
6. 所有原始严格失败仍保留；诊断结束也不使 `passed=false` 变为 true。任何失败不生成 `preflight.json`，cache/fit 继续被阻止。

## 运行身份与输入

保留旧失败目录。使用已存在的 8,000 篇候选 JSONL、同一历史排除输入、同一模型快照、同一固定科学配置；只能更新本次 `run_root` 和 `freeze_dir` 为全新路径。准备阶段重新生成绑定新代码的 manifest，并核对每角色的文档/有效位置清单与旧清单一致（不要读取最终测试的质量结果）。数据或划分不一致即停止，不悄悄补数据或改变预算。

模型仍是既有 Mixtral-8x7B-v0.1、固定 revision、BF16/eager、四张 A6000；不下载模型、不切换为 100M。使用 v2 环境依赖，未增加包依赖。代码 SHA 必须完整固定，输入/模型/配置由既有身份机制哈希。CPU 修复测试可在无 GPU 环境运行。

```bash
set -euo pipefail
CODE_SHA=REPLACE_WITH_DELIVERED_FULL_40_HEX_SHA
git fetch origin repair/mixtral-conditional-parity-diagnostics
git worktree add --detach ../mixtral-parity-${CODE_SHA:0:12} "$CODE_SHA"
cd ../mixtral-parity-${CODE_SHA:0:12}
test "$(git rev-parse HEAD)" = "$CODE_SHA"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pip check
python -m unittest tests.test_loop_parity_diagnostics tests.test_loop_scale -v
python -m unittest discover -s tests -t . -v
python -m compileall -q mixtral_lab experiments
```

配置保存在新 worktree 以外，保留最终配置哈希。`CONFIG` 是新配置的绝对路径；仅路径变化，研究参数、模型和原输入不变。先核对可用设备，不干扰其他任务。

```bash
CONFIG=/absolute/path/to/new-parity-run-config.json
python -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage prepare
python -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage preflight
```

这两条命令不串联训练；preflight 失败返回非零是正确行为，不能用 `|| true` 继续正式实验。共享 suite 预算与 stage receipts 不重置。300 秒预算可能不足以收集所有 case；此时回传 timeout、已完成 case 及成本，而不是延长到通过。每个 case 的额外 trace 和隔离专家/head 调用均计入 work 计数。没有自动恢复失败 preflight；保留其输出，再审阅决定新的诊断版本/预算。

## 验收、失败和后续

- 回传完整固定代码 SHA、结果发布 SHA、stage_receipts、preflight_progress、运行日志、环境/模型身份、各 case 完成状态、峰显存和耗时（如因异常未写完整，明确缺失）。
- 每个失败 trace 查看 `first_exact_difference` 与 `first_tolerance_failure`；首次出现差异的层不自动等于最终根因。
- `fixed_dense_inputs_routes_expert_only` 超差说明即使输入/路由固定，专家批次计算仍敏感；未超差也不证明整体正确。`fixed_dense_hidden_head_only` 单独检查 readout 行数影响。
- 同时核对 router pairs 是否完全相同，不仅看权重接近。不得把统计容差满足代替专家编号一致。
- STOP 必须零额外专家，未选位置不应被修改；原严格整体 parity 仍须通过。任何异常/非有限值/主干变化立即停止。
- 成功收集诊断但 parity 失败：状态为失败待处理，不能称 GPU 修复完成，不能启动学习曲线。所有 case 通过后，也需审阅来源和成本，再由操作者明确执行 v2 后续步骤。
- 不对当前无 GPU 复验的补丁宣称速度、显存或质量改善。最终测试集保持封存，本次没有使用其损失作判断。
