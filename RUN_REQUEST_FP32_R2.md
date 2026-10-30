# FP32 第二轮专家候选：数值执行修复与有界筛查

## 依据和状态

代码基线为 PR #17 `d33248074ef46fcf874ea756d7668bf5adc3849f`。结果 main `b6d81043628aed63d0f7dc2413be5ee5d8628754` 显示，抽样专家的首个差异位于 gate/up 或 down 矩阵乘法；关闭 BF16 reduced-precision reduction 减少了约 96% 的超容差元素，但完整专家输出仍不一致，抽样 FP32 参考的最大偏差未下降。该结果不能证明任一具体 GPU 内核错误，也不能证明 FP32 候选一定通过。

这是一个**显式的新数值协议候选**，不是把旧结果重新判为通过。生产路径中既有 rounding 边界发生受控改变；不宣称与原 BF16 执行逐 bit 等价，不复用旧适配器 checkpoint 或旧 preflight 凭据。

## 执行变化

原模型仍冻结、保持 BF16 常驻四张 A6000，native attention/prefix、post-attention norm、Top-2 路由和输出头不变。新策略仅作用于 `ScaleBranch(kind='r2')` 的新增第二轮：

1. 归一化结果仍来自 native norm。correction 的两层线性以及反传都采用 FP32、TF32 关闭；将 correction 加至 norm.float 后，**不再先回转 BF16 再进入专家**。
2. 按原 eager 的专家顺序与 slot-major 聚集顺序，packed gate/up 投影、SiLU、乘积、down 投影、路由权重乘法与累计输出全部 FP32。
3. 残差幅度和相加保持 FP32，最后仍在原残差出口回转 native dtype；原输出头保持原精度。这个 BF16 边界及 native norm/head 仍可能产生形状相关差异，所以必须保留原 hidden/CE 严格检查。
4. Dense 参考与真正 compact 训练/推理走同一候选函数。STOP 不调用专家，未选行保持原值；没有 padding 全量执行或逐 token 慢速伪硬门。
5. 路由使用原 norm 的结果并 detach，不对专家选择或路由权重反传；只训练既有 correction 参数，参数控制分支保持原执行。

## 内存与反向传播

`mixtral_lab/fp32_experts.py` 的 first-order 自定义 linear 在 forward 和 backward 独立禁用 autocast/TF32，并恢复精度状态。冻结专家 linear 只保存原 BF16 权重 view，不保存完整 FP32 权重副本；需要时转换一个投影，算完释放。小型可训练 correction 的 linear 另保存输入以计算权重梯度。

标准 Mixtral 单个 gate/up FP32 投影约 448 MiB，down 约 224 MiB。激活、梯度、分配器缓存与原模型仍占内存，这两个数字不是总峰值保证。CUDA allocator 可能保留已释放块；报告 allocated 与 reserved，不用 empty_cache 隐藏成本。拒绝可训练专家权重和非支持 eager/SiLU 布局，无自动降级。

只支持本实验所需的一阶反传，不提供二阶梯度。全局精度设置必须串行使用，不能在同一进程并发运行其他不同精度 forward。CPU 测试检查了 frozen/trainable linear 的梯度方程、无 FP32 冻结权重保留、异常恢复等，但不能替代真实 GPU 验收。

## 新配置、身份和验收

- `r2_precision` 只允许 `native_bf16` 或 `fp32_experts`，旧 JSON 未指定时为前者。
- 精度字段进入 ScaleConfig、checkpoint、候选评估和 resume 身份；源码/配置/冻结数据/模型绑定继续有效。新增字段使旧 checkpoint 身份不兼容，必须新 run，不隐式迁移。
- 候选的普通 preflight 强制执行梯度一致性，即使底层调用者传 gradient_checks=false 也不能关闭。
- hidden/CE 仍检查 STOP/all/even/odd/first-singleton/last-singleton，容差仍 `atol=rtol=1e-4`。梯度检查 even/odd/first/last singleton 的相同选中 token 平均 CE，分别比较 correction 的 down/up 参数梯度，更新前后都执行。空的选中集合明确为不适用，不伪造梯度通过。
- 零 up 初始化时 down 梯度为零是正常的；up 必须有限且非零。使用 autograd.grad，不污染已有 `.grad`。保留一次临时 optimizer 更新以验证有效输入变化，候选按真实 FP32 输入度量，不再错误回转 BF16 后计数。
- 验收同时要求主干参数版本不变、无主干梯度。原 native observer 不能观察替换后的候选函数，因此候选中禁用旧 hooks 诊断，不能拿旧 BF16 重构结果冒充候选证据。

## 配对 precision-screen

先运行独立筛查阶段：4 篇原 adapter-train 文档 × 幅度 1/0.1 × native/candidate，共 16 个 case；两种精度的先后顺序交替。各 case 使用相同初始化和各自一次临时更新，更新后参数可能不同。因此训练文档 NLL 快照不代表相同参数下的纯精度效应，更不是开发集泛化收益。

每个 case 报告：
- 原严格 hidden/CE、更新前后梯度检查、有效输入变化、主干完整性；
- train 文档 native/R2 更新前/R2 更新后的平均 NLL（仅辅助诊断）；
- 对所有可见 GPU 同步的完整检查耗时；重置 peak 后的各 GPU allocated/reserved 峰值及起始占用；
- 精度模式、代码/配置/模型/冻结数据身份，尝试和完成计数。

耗时包含正确性检查和多次反传，**不是独立推理吞吐或端到端速度**。不得因检查耗时较小就声称训练或稀疏推理更快。

筛查共用原 300 秒预算和不可变 suite 截止时间的较早者，两种精度均在操作边界检查；已发出的 GPU kernel 无法抢占。完整候选未通过时仍保留全部预算内案例；异常、非有限损失、主干变化、OOM、超时立即停止并保留已完成案例/当前尝试。不自动恢复未知工作或重置预算。

输出 `precision_screen.json` 与 `precision_screen_progress.json`。`status=complete` 只表示筛查完成，必须另看 `numerical_checks_passed.fp32_experts`。**无论筛查结果如何，都不创建 preflight.json，也不允许直接训练。** 后续须审阅数值和成本，再由操作者明确执行同配置的普通 preflight；其通过才满足原 cache/fit 的机器门槛。

## 运行步骤

不要覆盖上一轮结果。固定此次交付 SHA，新 worktree、新 run_root、新 freeze_dir；沿用同一候选语料、历史排除、模型快照、设备预算和研究参数，不下载权重或使用最终 test 质量。新旧冻结清单应逐角色核对一致。

```bash
set -euo pipefail
CODE_SHA=REPLACE_WITH_DELIVERED_FULL_40_HEX_SHA
git fetch origin fix/mixtral-fp32-r2-candidate
git worktree add --detach ../mixtral-fp32-${CODE_SHA:0:12} "$CODE_SHA"
cd ../mixtral-fp32-${CODE_SHA:0:12}
test "$(git rev-parse HEAD)" = "$CODE_SHA"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pip check
python -m unittest tests.test_fp32_experts tests.test_loop_scale tests.test_loop_parity_diagnostics -v
python -m unittest discover -s tests -t . -v
python -m compileall -q mixtral_lab experiments
# 从新配置复制到 worktree 外，仅落实原输入/模型和全新输出路径。
CONFIG=/absolute/path/to/new-fp32-r2-config.json
python -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage prepare
python -m experiments.mixtral_loop_scale --config "$CONFIG" --expected-commit "$CODE_SHA" --stage precision-screen
```

到此停止。回传完整代码 SHA、结果 SHA、两个 screen JSON、stage_receipts、prepare/筛查日志、实际环境及清单核对记录。若中断，保留全部输出，不用 `|| true` 串联训练。不合并 PR、不自动启动学习曲线；这次仍不评价增加数据是否成功。
