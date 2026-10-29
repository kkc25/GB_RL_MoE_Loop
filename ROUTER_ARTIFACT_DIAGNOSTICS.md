# Router 产物诊断：只读 CPU 运行与迁移说明

## 1. 当前状态与边界

这是诊断工具和运行说明，**不是新的实机实验结果**。仓库已有 JSON 报告，但没有
服务器保存的二进制监督表和 actor；真实产物诊断仍待操作者运行。小型/合成测试只
验证软件行为，不证明真实模型收益。

工具仅在 CPU 上读取已保存的有效行特征、成本与 actor，不加载预训练模型或
tokenizer，不执行 GPU forward、不重训、不重新选 checkpoint、不修改历史运行。
CPU 有效行计算**不等同于原完整 padded-document GPU 策略执行**，也不保证逐位
数值相同；实际 train/development 行为仍以原执行报告为准。

## 2. 已核实的历史结论

以下来自固定源码基线 `0888b293bec27340e4cdc8a3810f3ad4abb48911` 中的历史报告，
不是新诊断结果。

| Adapter seed | Native KL 权重 | Development-best update | Selection 收益（nat/token） | 来源 |
|---|---:|---:|---:|---|
| 17 | .1 | 192 | .01084070 | [evaluate](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/runs/mixtral_document_v1/document_adapter_pilot_evaluate/report.json), [tune](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/runs/mixtral_document_v1/document_adapter_pilot/tune/summary.json) |
| 17 | 0 | 192 | .01106345 | [evaluate](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/runs/mixtral_document_v1/document_adapter_kl0_pilot_evaluate/report.json), [tune](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/runs/mixtral_document_v1/document_adapter_kl0_pilot/tune/summary.json) |
| 29 | .1 | 192 | .01063295 | [evaluate](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/runs/mixtral_document_v1/document_adapter_pilot_seed29_evaluate/report.json), [tune](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/runs/mixtral_document_v1/document_adapter_pilot_seed29/tune/summary.json) |
| 29 | 0 | 192 | .01056218 | [evaluate](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/runs/mixtral_document_v1/document_adapter_kl0_pilot_seed29_evaluate/report.json), [tune](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/runs/mixtral_document_v1/document_adapter_kl0_pilot_seed29/tune/summary.json) |

四组复用相同的 16 篇 selection 文档（3,967 个有效 next-token 位置），不是四次
独立数据集重复实验；协议窗口是 256 token，不能外推到无限长文档。四组 best 均为
192，继续到 256 的 development 退化仍存在。KL 优劣排序在 seed 17/29 间翻转，
尚不能声称 KL 稳定占优。保留现有 seed-43 adapter 配对与 expected-CE router
下一轮计划，不重复启动、不替换正在运行的配置、不用 selection 反向调参。
本 checkout 本身不能证明服务器 seed-43 配置当前是否可用。

Router 的 192 次更新均有非零梯度；覆盖 5,859/5,859 个位置，总访问 6,115 次。
固定 surrogate objective 从 3.4830004 降至 3.4107320，实际执行策略熵从
3.0800471 升至 3.2064578；但实际 greedy 动作全部保持 native，train/development
NLL 持平，best 保留 update 0。176 步覆盖率仅 0.961256，192 是首次覆盖完整的
评估点，此时 development patience 早已耗尽。因此这是**覆盖条件满足后触发耐心
停止，不是优化器收敛**；非零梯度也不等于 greedy 路由改变。
见 [router summary](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/runs/mixtral_document_v1/document_router_pilot/tune/summary.json)、
[metrics](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/runs/mixtral_document_v1/document_router_pilot/tune/metrics.json)、
[停止条件源码](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/mixtral_lab/document_router.py#L309-L345)。

## 3. 输入文件与来源绑定

[原始保存代码](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/mixtral_lab/document_router.py#L295-L338)
确认同一次服务器 router run 下的文件名：

- `report.json`：完成的 tune 报告，在 run 根目录，**不在 `tune/` 下**。
- `tune/train_surrogate_table.pt`：特征、native pairs、成本、binding 与 provenance。
- `tune/checkpoint_latest.pt`：最后记录的 actor；本 pilot 为 update 192。
- `tune/checkpoint_best.pt`：development 选中的 actor；本 pilot 为 update 0。

先核实服务器真实 run 根目录。报告、表和 checkpoint 必须来自同一次原始运行；
同名文件不代表同一身份。binding、tensor/state hash、配置、provenance 或原始
源码摘要不匹配时应停止诊断，不得修改历史身份来绕过检查。

历史 `.identity.code.commit` 可以不同于后来上传结果的 commit。本 router 记录为
`8e799711b6ebd32ae46fb707dddec9a1a1379945`，但其报告记录的每个原始源码文件摘要
均与上述基线一致。新增诊断文件应单独记录版本；不要为通过诊断而改旧 trainer、
科学配置或报告。见 [原报告](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/runs/mixtral_document_v1/document_router_pilot/report.json)。

监督表约 193 MB。安全预检同时限制表文件与 ZIP 声明的解压总量不超过 **512 MiB**，
checkpoint 上限为 64 MiB；仅接受普通非符号链接、stored-only ZIP 文件。使用 CPU
`weights_only` 加载原始可信 tensor 产物，不执行任意 pickle。检查失败先调查原因，
不要关闭验证或篡改身份。

## 4. 独立 worktree 运行，不干扰训练

**建议在独立诊断 worktree 操作，绝不要在正在训练的 worktree 中 checkout/pull。**
准备包含新诊断代码的独立 worktree；其旧科学源码字节必须匹配原报告记录的摘要。
原训练 worktree、配置和运行目录保持原样。诊断 PR 是否合并与本步骤无关；本流程
不授权合并或开启 auto-merge。

在独立诊断 worktree 根目录、可用的 CPU PyTorch 环境中执行。`RUN` 必须指向
**旧服务器 run 的绝对路径**，不是新 worktree 下碰巧同名的目录。以下路径均为
占位符，先替换为核实后的路径：

```bash
RUN=/absolute/path/to/runs/mixtral_document_v1/document_router_pilot
OUT=/absolute/path/to/new/router_artifact_diagnostics_2026_10_29
# OUT 必须不存在；不要使用旧 run/tune 目录，也不要覆盖已有诊断。
test -f "$RUN/report.json" && test ! -e "$OUT" || exit 1
mkdir "$OUT"
for KIND in latest best; do
  CUDA_VISIBLE_DEVICES='' python -m experiments.mixtral_router_diagnose \
    --tune-report "$RUN/report.json" \
    --checkpoint "$RUN/tune/checkpoint_${KIND}.pt" \
    --table "$RUN/tune/train_surrogate_table.pt" \
    --output-dir "$OUT/$KIND" \
    --batch-size 256 --near-tie-tolerance 1e-6 || exit 1
done
```

每次调用在全新子目录写入 `diagnostics.json`、`summary.md`。API：
`diagnose_router_artifacts(tune_report, checkpoint, table, output_dir,
batch_size=256, near_tie_tolerance=1e-6)`。

latest/best 分开输出，原输入保持只读；分析/上传时一并保留输出、准确命令、诊断
代码版本与输入摘要。本步骤不启动 seed-43/expected-CE 实验，也不引入其他项目的
运行请求流程。

## 5. 指标解释与下一步判断

**两类 KL 不能混淆：**

- Adapter `native_kl_beta=.1` 或 `0` 是正则**权重**，不是已测得的散度；对应
  `KL(native-token distribution || corrected-token distribution)`。
- Router `reference_kl_weight=.01` 对应专家对动作分布上的
  `KL(current routing policy || native-smoothed routing prior)`。

两者分布和方向不同，都是辅助优化惩罚；checkpoint 按 development NLL 选，不按
最小 KL 选。应分开报告原始 KL、系数、加权贡献，以及目标交叉熵；不要把标签交叉熵
与 expected table CE 混为一谈。见 [adapter KL](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/mixtral_lab/document_training.py#L297-L300)、
[router KL](https://github.com/kkc25/GB_RL_MoE_Loop/blob/0888b293bec27340e4cdc8a3810f3ad4abb48911/mixtral_lab/recovery_router.py#L209-L226)。

重点看：

1. **软标签是否要求切换：** target entropy、native mass、target argmax。平坦标签
   的 argmax 只会返回第一个并列索引，不能据此说 teacher 偏好非 native。结合
   `native_is_uniform_cost_minimizer`、`native_within_best_cost_tolerance`、
   `unique_target_argmax_differs_native`，区分精确并列、容差内近似并列和唯一替代偏好。
2. **是否越过决策边界：** native 与最佳非 native 动作的 logit/probability margin、
   声明容差内的 near tie。CPU 近似并列不能证明 GPU 动作改变。
3. **残差是否有效：** residual norm，latest/best 与原 native prior 的对比；区分
   actor 没有变化与残差已经变化但不足以改变 argmax，不能只看总 objective。
4. **成本的适用范围：** expected/greedy/minimum table cost 只是 uniform-action
   监督表统计，**不是部署收益，也不是 oracle bound**。表没有枚举 mixed routing。
   在这里的末端逐 token 计算路径，不应虚构 attention 跨 token 因果反馈；相关数值
   差异应表述为**不同专家批量形状导致浮点差异**。实际 train/development 重放需要
   另行授权运行，不能由本 CPU 诊断替代。

原始二进制诊断完成前，不要宣布失败机制已定位。拿到结果后再区分标签过平、先验
过强、margin 太小或目标不匹配；保留阴性结果和现有下一轮版本，任何新科学实验
使用新的运行身份。
