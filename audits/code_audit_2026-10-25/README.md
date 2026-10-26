# GB_RL_MoE_Loop 代码与实验审计包

**2026-10-25｜固定提交 `f2bef422f7315f30618f00c690bf82810d1e85ac`**

核心判断：基础 Looped MoE 与默认全 28 动作反事实标注主要语义已经实现；停止部署、E12 adapter、数据覆盖、对照配平和测量协议存在具体问题。当前负结果不足以排除整条研究路线，当前停止正结果也尚未排除固定 R1 的简单解释。

## 阅读顺序

| 文件 | 用途 |
|---|---|
| [00_audit_report_CN.md](00_audit_report_CN.md) | 主报告：17 组 finding、原始来源、关键复算、研究结论边界 |
| [01_plan_implementation_matrix.md](01_plan_implementation_matrix.md) | E00–E12 全部 13 项原承诺、实际实现、证据等级与下一步 |
| [02_repair_and_rerun_plan.md](02_repair_and_rerun_plan.md) | 修复顺序、复用 M105 的实验、3–5/最多 8 张 A6000 分配 |
| [03_next_round_artifacts.md](03_next_round_artifacts.md) | 下一轮最小权重/缓存/输入清单；已上传材料不用重复提供 |
| [04_reproduction_and_scope.md](04_reproduction_and_scope.md) | 环境、复现命令、原测试与新增反例为何不矛盾、覆盖与局限 |
| [agent_model_training.md](agent_model_training.md) | 模型、优化、恢复、成本与系统测量详细附录 |
| [agent_counterfactual.md](agent_counterfactual.md) | 反事实、停止、互补、报价与统计详细附录 |
| [agent_synthetic_rl.md](agent_synthetic_rl.md) | 合成机制、调度、当前 NumPy/旧 Torch RL 详细附录 |

## 已实际完成的验证

- 仓库全部 390 个 Git 文件逐 blob SHA 核验，无缺失或不匹配；见 [snapshot_integrity.json](snapshot_integrity.json)。
- 原仓库测试 **424 passed in 83.15s**；见 [existing_tests.log](existing_tests.log)。
- 原代码 CPU 反例：停止特征契约、E12 专家轴与 `.to()`、不公平计时、checkpoint 时间/best 恢复、旧 exact 梯度与预算概率质量、评估入口、因果性总入口等。
- 仅用上传记录复算：E01/E10 文档覆盖、Q 标签方差、train-only 廉价规则、E10 符号定义敏感性、E07 全窗口离线 oracle。
- 当前 NumPy E09 梯度恒等式和有限差分再次核验通过。
- 两个生产文件的最小修复建议与 3 个回归测试：原实现 2 失败+1 报错，内存应用补丁后 3/3 通过。

**没有执行的事：**未使用用户未上传的大权重，未在 A6000 重训/计时，未把 proposed 实验当结果，未修改或推送远端仓库。本包不重复附整个 Git 仓库，也不包含依赖安装目录或训练检查点。

## 补丁范围

[patches/minimal_correctness_fixes_with_tests.patch](patches/minimal_correctness_fixes_with_tests.patch) 包含：

1. 动态停止学生读 raw h，router 保持 normalized h。
2. E12 首层按对应专家计算，并重命名覆盖了 `nn.Module._apply` 的方法。
3. 有限阈值特征契约、专家独立性、dtype/device 转换三个回归检查。

也提供两个独立 patch。它们不含所有审计问题的修复；训练协议、候选映射、计时、账本及统计设计仍需按计划实施。E12 修正后需重新训练相应比较，不能把旧权重直接用于新公式并当作公平结论。

## 文件完整性

`MANIFEST_SHA256.json` 记录包内交付文件的 SHA-256（不包含它自身，避免递归）；`artifact_validation.json` 记录打包检查。`source_inventory.json` 列出固定提交的文件分类与来源，不把“文件已取得”混同于“实际权重已复现”。原检查点和 A6000 结果的边界在主报告及复现说明中明确列出。

固定源码：[GitHub commit](https://github.com/kkc25/GB_RL_MoE_Loop/commit/f2bef422f7315f30618f00c690bf82810d1e85ac)。文档中的源代码链接使用固定提交；未来 main 的变化不会静默改变本报告的审查对象。
