# Mixtral 冻结主干路由实验

最新的分层采样、R1/R2+STOP oracle 与默认关闭的残差适配器方案见
[LEARNABILITY_PLAN.md](LEARNABILITY_PLAN.md)。下文保留原始基础协议与运行边界；
适配器是新增修正容量的独立实验，不代表已训练的循环专家或 learned STOP。

## 本轮要回答什么

采用 `mistralai/Mixtral-8x7B-v0.1` **预训练 Base**，而不是 Instruct 版本。
官方参数为约 46.7B 总参数、12.9B 每 token 激活参数；每层 8 个专家选 2。
这保留了当前研究中 28 种无序专家对的动作结构，但不表示现有 100M 模型的
执行器、数值契约或 checkpoint 可以直接复用。

本轮依次回答：

1. 原生 Mixtral 在当前硬件和工作负载下的基线质量与耗时是多少？
2. 将末层 FFN 接入受控执行路径、仍选原路由时，输出是否在预先规定的误差内？
3. 冻结主干，只训练末层 R1 路由 actor，能否在 selection 上改善原路由？

本轮**不做**从零预训练、全参数 RL、自动下载大权重或自动启动服务器作业。
R2/STOP、多轮执行、单专家适配和模型规模扫描应在 R1 基线与可学习收益明确后
另行设计；不能把未执行或未实现的阶段写成成功结果。

## 硬件与预算边界

仓库硬件记录列出 8 张 RTX A6000、每卡约 48GB；研究计划记录通常动态可用
3–5 张，最多 8 张。这不是当前空闲卡数或独占承诺。记录中的 GPU 0–3 与
4–7 分属两个 PCIe 拓扑组，组内 PIX、跨组 SYS，没有记录到 NVLink 连接。
运行前需重新核对可用设备、显存和竞争进程。

BF16 参数按两字节估算，46.7B 权重本身约 93.4GB；这还不包含激活、临时
logits、KV cache、actor、梯度或优化器。四卡是预留空间的规划起点，不是已
验证的吞吐承诺；不能把 8 卡显存相加当作透明的单块显存。

第一轮保持短序列、小样本和明确的更新上限。多卡同步 wall time 才能覆盖
实际分片路径；不能用一张卡的 CUDA event 代表全部设备完成的端到端时间。
记录实际 device map、精度与软件版本；跨卡放置、通信和内核实现变化必须与
路由策略变化分开解释。

`device_map` 是按层放置/分片，不是 tensor parallel 或 expert parallel；
它主要解决装载容量，不保证多卡加速。原生与受控路径必须使用相同放置方案。
默认 `eager` 是正确性基线，仍可能包含 Python 专家循环，不能据此断言 MoE
架构在优化部署后没有速度优势。若使用 `grouped_mm` 候选后端，需独立配置、
重新检查原生一致性与 NLL；累加方式变化不应被静默当成同一基线。

## 不可沿用的 100M 假设

- 使用 Mixtral 自己的 tokenizer。旧 100M tokenizer、packed cache、token ID
  和 reference CE 不可直接搬用；重新建立独立数据身份。
- 保留原生 RMSNorm、Top-2 混合、残差与输出头。不能沿用自定义模型的 gamma
  0.5；原生残差系数为 1。Mixtral 的输出头与输入 embedding 不绑权。
- 只控制末层 FFN，之前的 attention 和专家层保持冻结。不能把“冻结参数”
  误写成没有计算或显存成本。
- 原生路由对照与受控原路由对照都要保留。受控执行器的捕获、路由和调用
  开销属于该研究实现，不能直接视为生产推理框架的最佳性能。
- 同一模型、同一 tokenizer 下比较 token NLL；不同 tokenizer 模型的原始
  token NLL 不能直接作质量排名。dense 对照需要另行固定质量与计算比较口径。

## 数据与来源

只读取操作者准备的本地模型、tokenizer 和文本文件；禁用远程自定义代码，
不在实验命令中自动下载权重或语料。模型目录、tokenizer、数据、配置与源码
身份应写入结果，固定 revision 标签不能代替实际文件的内容校验。

train、selection、calibration 使用显式且互不重叠的文档集合。不要把
newtest 或最终确认数据用于调参、策略选择或性能方案选择。不同文件名不等于
数据独立，必须检查文档 ID 和内容重叠；这些精确检查仍不能证明没有近重复。

每次选择新的输出目录，不覆盖历史 run 或失败证据。新 Mixtral 实验与 100M
结果独立保存，不能把新模型结果追加到旧 actor/checkpoint 下冒充续训。

## 阶段与决策

### M0：离线检查和原生一致性

先确认模型架构、数据和设备身份，再运行短测试。比较原生前向与受控原路由
R1，报告动作范围/轮数合法性、输出/CE 偏差及其门槛。临时 hook 独立捕获
末层原生 gate 的专家对，并与受控路径按无序专家对 ID 比较；该检查也是
门槛的一部分。数值通过只适用于实测样本，不能证明
全语料等价。若检查失败，保留失败报告，停止后续训练，不通过放宽门槛掩盖
模型或实现变化。

### M1：固定工作负载性能基线

在相同输入、batch、长度、精度和设备映射下比较原生与受控路径。预热、
正式计时、质量评估应明确区分；不能将一个路径算 CE、另一个只算 logits。
分别说明测量是完整序列前向还是逐 token 生成，不能把前者当作服务吞吐。
保留原始重复测量，不能只选择最快一次。

### M2：冻结主干的有界 R1 策略实验

先记录未训练 actor/原路由的 selection 基线，再进行有限更新。RLOO 的
leave-one-out 对照应在同一 context 的独立采样之间计算，奖励与优势停止
梯度，只有 actor 更新；不能让梯度进入专家或语言模型主干。

记录成功更新、实际耗时、采样轨迹、奖励与选择结果。用 selection 决定 best，
保留 initial、best 和 final 的区别。若 best 仍为初始状态，应报告没有可见
学习收益，而不是把训练完成或训练损失下降当作策略成功。

末层上下文缓存、冻结输出头及每次候选动作的 readout/CE 都有成本；
冻结不等于免费。先用有界 context 缓存做可学习性短测，不把此缓存式训练的
吞吐当作完整模型在线训练或自回归生成吞吐。

预先固定样本数、序列长度、采样数、更新数、评估频率与数值门槛。非有限数值、
身份不符或一致性失败应失败退出；达到预算上限即停止。单步 KL 超限时恢复该步之前的 actor/optimizer 并停止，
在 `router/summary.json` 查看 `stopped_step_kl`；外层报告完成仅表示流程退出，
不代表全部预定更新成功或获得正收益。短测结果不构成统计
显著性或实际质量收益，后续确认还需要独立多 seed/置信区间与冻结候选流程。

## 运行手册（操作者在服务器执行）

### 1. 准备环境与独立数据

按 [ENVIRONMENT.md](ENVIRONMENT.md) 配置匹配服务器驱动的 PyTorch CUDA wheel，
再安装本实验的额外依赖：

```bash
python -m pip install -r requirements-mixtral.txt
python -m experiments.mixtral_experiment --help
```

依赖安装可能访问包源；实验运行只访问本地模型和数据，不下载模型。
模型目录必须是完整、已核验的 Mixtral Base snapshot；tokenizer 目录可指向
同一 snapshot。`model_revision` 填该 snapshot 的真实 40 位 Git commit，
不要填 `main`、随意的 40 位字符串或把声明当作来源证明。

分别准备 train、selection、calibration 三个 JSONL 文件，每行格式：

```json
{"id": "train-000001", "text": "此处是已获得使用许可的独立训练文档正文。"}
```

以上仅为格式说明，不是训练语料。三组的 ID、正文都要不同；不要把同一正文
换一个 ID 放入另一组。每篇文档独立编码、截断、右侧 padding；只计算有效
相邻 token 的 shifted CE，不跨文档拼接。每个文件的全文档数和每篇长度也
受配置上限约束，不是只检查被抽取的前几篇。不要将最终 newtest 放进这些文件。

### 2. 复制配置，冻结运行契约

```bash
mkdir -p local_configs
cp configs/mixtral/frozen_router_smoke.json local_configs/mixtral_validate.json
```

默认 smoke 仅使用每组 2 篇、长度 128、32 个训练 context、4 次更新、
每次 4 个 context × 4 个采样；它是流程短测，不是足以检验研究假设的正式实验。
`configs/mixtral/grouped_mm_candidate.json` 是独立后端候选；先做 eager，
候选不支持当前设备或未通过门槛时保留失败，不静默回退成 eager。

编辑复制件，至少设置 `model_path`、`tokenizer_path`、`model_revision`、三个
`data_files` 路径、`output_dir`；检查 `max_memory` 与实际可用卡一致。
GPU 键是当前进程的**可见设备逻辑编号**。例如仅当物理卡 0–3 已获准使用且
确实可用时，才用 `CUDA_VISIBLE_DEVICES=0,1,2,3` 配合配置键 0–3。
`max_memory` 是装载预算，不是显存占用上限保证，需保留前向与运行时余量。

先仅检查 JSON/配置结构（不加载权重、不运行 GPU）：

```bash
python - <<'PYCONFIG'
from mixtral_lab.protocol import load_config
config = load_config('local_configs/mixtral_validate.json')
print('config schema valid:', config['output_dir'])
PYCONFIG
```

这个检查不证明模型文件齐全、数据独立或显存足够。不要因结构检查通过就
跳过后面的原生一致性门槛。

### 3. 逐阶段执行，不自动串联

在明确设备授权、空闲状态后执行 M0；`validate` 会装载真实模型并运行 CUDA，
不是只解析配置：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.mixtral_experiment \
  --config local_configs/mixtral_validate.json --stage validate
```

检查该配置 `output_dir/report.json` 的 `status` 与
`native_equivalence.eligible`。通过后复制配置为
`local_configs/mixtral_profile.json`，**仅更换成未存在的输出目录**，执行 M1：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.mixtral_experiment \
  --config local_configs/mixtral_profile.json --stage profile
```

本版本计时为 batch=1 的完整序列前向，禁用 KV cache；包含输入传输以及
完整 logits 返回输入设备的传输，所有模型 GPU 同步后记录 wall time。计时
区间不包含 CE；不是逐 token 解码 benchmark。保留原始重复时间。
`timing.peak_memory_bytes` 报告预热后全部计时路径合并的各卡 allocator
峰值 allocated/reserved；这是绝对占用，不是每条路径的额外显存或整机独占
显存，不证明无竞争作业。竞争进程与设备状态仍需服务器监控另存。

`native_selection` 是未修改的原生模型评估，`controlled_original_selection`
是受控原路由评估，不应混为同一条结果。`model_load_wall_seconds` 与
`stage_wall_seconds_excluding_model_load` 分开记录；后者包含阶段内的数据
准备、门槛与评估等，不能当作纯训练或纯前向计时。
先看绝对耗时和受控路径增加的开销，再决定是否值得进行学习实验。

确定预算后复制为 `local_configs/mixtral_train.json`，再次使用新输出目录，
核对 `training` 的更新数、context batch、RLOO group size、学习率、评估间隔
及 KL 上限；`max_train_contexts` 与训练配置一致。执行 M2：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.mixtral_experiment \
  --config local_configs/mixtral_train.json --stage train
```

`profile` 和 `train` 都会再次跑原生一致性检查，失败不继续。所有阶段均无
resume；不要删除失败目录再原地重跑，改用新的输出路径。只改输出目录即可
区分运行；改变样本、后端或精度则是新实验条件，需重新审查比较口径。

训练产物在该次输出目录的 `router/` 下，包括
`checkpoint_initial.pt`、`checkpoint_best.pt` 与 `checkpoint_latest.pt`；
latest 是最近记录的已接受状态，不一定是 best；失败时不保证所有文件齐全。
它们是 actor，不含完整 Mixtral 权重，也不能用于续训。需要测 best actor 时，
复制相同身份与数值契约的配置到新的 profile 输出路径，并使用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.mixtral_experiment \
  --config local_configs/mixtral_profile_best.json --stage profile \
  --actor-checkpoint /actual/train-output/router/checkpoint_best.pt
```

将示例绝对路径替换为实际文件；不要加载不可信的 checkpoint。代码、模型、
分词和数据等绑定不符应直接拒绝，不能绕过校验来比较。

### 4. 保存完整证据

保留配置、`report.json`、训练的 `training_contract.json`、`evaluations.json`、
训练指标和 actor checkpoints。初始化/装载阶段失败时可能尚未建立报告目录，
仍应另存 stderr 和设备状态；已有报告标记失败也必须保留。不要只上传 best
NLL，不上传失败或初始基线。仓库只提交小型报告和必要配置，不提交大模型权重、
私有语料、凭据或巨大的上下文缓存。

## 解释结果

需要分别报告：执行一致性、实测性能、策略质量、资源成本。即使规模增加后
MoE 更快，也不能单凭不同预训练模型的一次对比证明“100M 太小”是唯一原因。
本轮优先比较同一个预训练 Mixtral 的原路由与受控策略，以减少架构和训练
历史的混杂因素。

本次代码提交与 CPU 测试不代表服务器上的 46.7B GPU 实验已执行，也不构成
加速保证。历史负结果应保留。

## 完整性修补与 R1 快速路径

- 模型路径的祖先可以包含 `.cache`，包括正常 Hugging Face snapshot；仅过滤
  模型根目录内部的缓存子目录，符号链接文件仍校验所指向的实际字节。
- 原始文档 ID/内容去重之外，分词、截断后还检查不同数据角色的**有效 token
  序列**是否相同，不把文档 ID 或 padding 混入重复判定。被拒绝时重新构建
  开发集，不能靠改 ID 或增添会被截掉的尾部文字绕过检查。此检查仍不是
  所有部分重叠、近重复或语义泄漏的证明。
- 独立 profile 的 `actor_checkpoint` 记录实际载入文件的 SHA、binding SHA、
  更新步数、epsilon、训练配置和 actor state SHA。文件 hash 与反序列化来自
  同一个字节快照，不能只用 `best` 或 `latest` 文件名代替身份。
- R1 默认快速路径复用一次末层 norm/gate；原路由无策略时不构建无用特征和
  STOP 状态。`adapter.forward(..., reference=True)` 保留通用对照，R2 路径不变。
  CPU tiny 模型中 FP32/BF16、eager/grouped_mm 及全部专家对已检查精确一致；
  这不是完整 Mixtral CUDA 加速证明，服务器仍需执行原生一致性门槛。
- `model.expert_backend_evidence` 只是请求/配置证据。校准的每个 group 中
  `native_expert_dispatch` 另记录已观察的 HF 分派分支、设备与 dtype；未观察到
  或不支持的版本明确标为 `unverified`。分支观察不等于底层 CUDA profiler
  kernel 证据；临时观测在 finally 中恢复，正式计时不带此 instrumentation。

### 评分去重核验候选：不是默认提速模式

默认 `training.score_mode` 仍为 `reference`。新配置
`configs/mixtral/score_reuse_checked_candidate.json` 显式选择
`deduplicate_checked_v1`：在单次更新内复用同一 context、同一专家对的评分，
保留原采样顺序和 RLOO 权重，缓存不会跨更新增长。

该候选改变专家/head 的批形状，因此每步仍计算完整参考分数，对每个分数检查
`score_atol + score_rtol * abs(reference)`；任一分数失败就回退到参考，并记录
失败原因。默认容差各为 `1e-5`，不构成普适 GPU 等价标准。

**完整参考校验也消耗计算，所以这个版本是去重的数值核验候选，不声称训练
总体更快。** `score_work` 明确区分 requested/scored/validation rows、调用数、
缓存命中、逐项误差及 candidate 是否实际采用；汇总见 `score_work_totals`。
不要把候选理论减少的行数当成包含校验后的真实速度收益。未经目标 GPU 校准，
不提供自动关闭核验并推广的模式。

更新代码后继续使用新的输出目录。旧报告、旧 checkpoint 和旧数据保留原样；
源码身份不匹配时在原 commit 复查旧实验或新建运行，不修改 binding 来强行加载。

## 参考

- [Mixtral 官方模型卡](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- [Mixtral 官方架构与参数说明](https://mistral.ai/news/mixtral-of-experts/)
- [仓库硬件记录](runs/bundle_validation/hardware_inventory.json)
- [A6000 容量与预算规划](plans/loop_moe_research_plan_2026-10-25/resources/01_a6000_capacity_and_budget.md)
