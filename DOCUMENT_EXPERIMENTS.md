# Mixtral 全文形状训练：先统一执行，再验证泛化

新入口 `python -m experiments.mixtral_document`，schema
`mixtral-full-document-recovery-v1`。本轮提供实验代码、配置和 CPU 测试，
**没有下载模型、启动服务器训练，也没有得到新 GPU 泛化或速度结论**。
旧 fit-first、recovery 和历史 bridge 的代码、配置、报告与 checkpoint 保持独立。

## 1. 为什么升级协议

`7ee8d14` 上传的[历史桥接结果](runs/mixtral_bridge_v1/summary.md)表明：

- 历史 router 和 adapter 的缓存指标都能复现，缓存 h0 与全文对应输入一致。
- 在全文执行的同一 64 个训练位置，router NLL 从 1.890769 降到 1.817432，
  adapter 降到 0.026535，拟合收益并非纯缓存假象。
- 同文档其余 1828 个位置，router 从 1.929946 升到 1.957412，adapter 升到
  3.137868。这里的未选中位置仍来自训练文档，**不是独立开发或最终测试**。
- 旧 singleton/probe 与全文的数值检查未通过；全文内 native/controlled/zero
  则逐位一致。不能靠直接放宽旧容差继续训练，也不能无限等待不再采用的旧
  singleton 路径与全文等价。

新协议改变的是缓存和计算的单位，不改写旧结果：缓存完整、固定 padding 文档，
保留全文尾部的计算形状；只在计算损失时用 mask 选择监督位置。末层 FFN 之后
没有跨 token attention，因此不虚构“其他位置经后续 attention 影响当前 token”。
专家批量大小和数值后端仍可能影响浮点结果，必须实测。

## 2. 同协议正确性与历史跨形状诊断是两回事

新训练前必须验证当前实际模型、数据、dtype、设备映射和后端下：

1. 完整模型原生执行与 controlled/零修正执行一致。
2. 原完整文档前缀加尾部，与缓存完整 h0 加**同一尾部**一致。
3. 非零修正/选路也经过缓存重放与完整模型对照，冻结候选再次复核。

这些同协议重放检查实际运行在入选的 **train 完整文档**上，是这些数据与
形状的数值 screen，不证明任意输入、设备或将来运行全局等价。

固定文档 B/S、padding、token 顺序、dtype、expert backend、设备映射和输出头
计算方式；不能先抽出若干 token 再计算 FFN 或输出头，却把结果叫作全文训练。
同形状只是必要工程约束，不是 GPU 精确一致的证明；以报告中的实际差值为准。
数值失败保留失败报告，不能修改报告、自动扩大容差或改写 checkpoint 身份。

旧 `experiments.mixtral_bridge` 仍用于历史 singleton/full-document 对照。
其失败不被删除或改称成功，但它不再充当新全文协议的训练入口门槛。
新协议有自己的实测 gate；这不是绕过正确性检查。

## 3. 四角色与新的数据清单

| 角色 | 用途 |
|---|---|
| train | 更新参数、构建 router 监督成本、训练诊断 |
| development | 全文在线 NLL 选 checkpoint 和早停；update 0 参选 |
| selection | 对唯一冻结 development-best 做外层开发筛选，不重新选模型 |
| calibration | 保留来源角色，绑定哈希并参与泄漏预检；此新 CLI 不对其执行模型评估 |

`calibration` 不是本实现额外跑过的一组 parity 结果。当前同协议 replay 的模型
检查使用 train，报告不应宣称 calibration 模型评估通过。

四个角色来源的 ID、规范化文本和有效 token 都检查跨角色重复，未被 first-N
选中的来源行也必须遵守隔离。精确去重不证明语义独立；应人工避免同篇文档片段、
同一问题改写或近重复跨角色。已反复查看的旧 selection 不能因更名而变成最终测试。
此入口不接受 newtest，正式最终测试需要另行冻结协议与从未使用的来源。

下面只创建新的 manifest，不修改历史文件。输入目录必须替换为真实历史开发
数据位置；目标目录必须不存在。至少需要旧 train 32 篇、selection 16 篇、
calibration 8 篇，不足时停止，不复制补齐。

```bash
python - <<'PY'
import json
from pathlib import Path
source = Path('/data/mixtral/dev')
out = Path('/data/mixtral/document-v1')
def read(name, count):
    rows = [json.loads(x) for x in (source / (name + '.jsonl')).read_text().splitlines() if x.strip()]
    if len(rows) < count:
        raise ValueError(f'{name}: need {count}, got {len(rows)}')
    return rows
train = read('train', 32)
parts = {'train': train[:24], 'development': train[24:32],
         'selection': read('selection', 16)[:16],
         'calibration': read('calibration', 8)[:8]}
seen_ids, seen_text = set(), set()
for role, rows in parts.items():
    for row in rows:
        ident, text = row['id'], ' '.join(row['text'].split())
        if ident in seen_ids or text in seen_text:
            raise ValueError('duplicate ID or normalized raw text')
        seen_ids.add(ident); seen_text.add(text)
out.mkdir(parents=True, exist_ok=False)
for role, rows in parts.items():
    (out / (role + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(role, len(rows))
# Effective token checks still run in the experiment preflight.
PY
```

配置选择每个来源文件的 first-N 文档，文档选择本身不是随机语料抽样。
24 train / 8 development 是起始小规模研究，不代表数据量足够或已排除过拟合。
训练位置随遍历轮换，报告实际独立位置覆盖和总访问次数；重复访问不算新增独立数据。
逐更新位置审计采用有界日志，明示截断/记录数量；汇总覆盖不能用日志中可见行数代替。

## 4. Adapter：完整文档缓存，末端选损失

缓存完整原生末层 h1，整个文档执行低秩残差与冻结 norm/head，再按训练 mask
取监督 CE；原生 KL 与相对修正正则覆盖该篇文档的**全部有效位置**，无需再增加
输出头调用。目标保留原生全词表输出约束：

`CE(gold, corrected) + beta * KL(p_native || p_corrected) + correction_penalty`。

固定 train probe 使用全有效位置 CE + KL/修正正则，优化步骤的 CE 使用轮换 mask，
二者不能混成同一个 mini-batch 损失。报告中“已访问/未访问位置”按累计 **CE 监督**
定义；未被 CE 选中的位置也会受到 KL 约束，不能称作完全未训练。

Teacher detach，主干与输出头参数冻结；输出头对修正隐藏状态保留梯度。
rank、residual scale、KL 权重是预声明参数，不通过开发集无限搜索。
零初始化保持 native，update 0 与所有评估过的候选共同参与开发集选模；没有
改善时允许保留原模型。非零 FP32 梯度与修正不保证 BF16 相加后发生有效变化。

这仍是单次 native FFN 后的新增低秩容量，不是第二轮专家执行、learned STOP
或已证明的 loop 收益。应先独立检验它，再研究与路由组合，不能把两种改动同时
打开后无法归因。

## 5. Router：监督表不是部署成本的承诺

保留 28 个无序专家对，紧凑 R1 特征、缩放残差 logits、全支持 native prior。
成本软标签提供训练监督，不把 gold label 或成本表传给推理策略。
成本表按完整文档尾部执行生成；它描述预声明的反事实执行条件，仍然是
**监督 surrogate**，不能承诺对任意 mixed-policy 逐位数值等价。

原因是其他 token 的专家选择改变实际 expert batch。把 singleton 换成整篇
文档并不自动消除这种区别。因此 train probe、development 选模和外层评估
必须走真实全文 mixed-policy，分别记录表内预测收益与真实在线收益。
不能因为 expected/greedy table CE 改善就宣布在线质量改善。

## 6. 阶段、预算与通过条件

- `tune`：同协议数值关通过后训练；独立 development 全文 NLL 选择 best，
  初始原模型参与，严格改善才替换。记录停止原因、位置覆盖、实际训练/teacher/
  表构建/probe/development 成本。
- `evaluate`：只接受身份匹配的冻结 development-best 与 tune 报告；对
  selection 做 native/controlled/candidate 配对全文评估，不反传、不另选权重。
- `profile`：需要同一冻结 checkpoint、匹配的 tune/evaluate 报告，以及通过的
  数值与质量关。不得为计时事后降低阈值，完成状态不等于科学成功。

默认最小开发/selection 改善 0.001 nat/token 是预声明工程筛选值，不是统计
显著性、部署许可或保证有效。除了 token 加权平均值，还应看逐文档损益、
训练已/未接受 CE 监督的位置以及独立开发文档，避免少数位置撑起平均数。

有界学习率控制与 development 选模是不同机制。学习率边界、观察间隔、耐心、
硬更新上限必须预先固定，不能用 selection 或 profile 耗时反馈调参。自动调整
不意味着寻找全局最优，也不自动扩大数据、rank、训练轮数或 GPU 预算。

### 6.1 六个有界模板

| 模板 | 用途 | 训练/开发文档 | 更新上限 | mask 单位 |
|---|---|---:|---:|---|
| `document_adapter_smoke.json` | 软件路径检查，不是效果实验 | 2 / 2 | 4 | 每次选中文档 8 个位置 |
| `document_router_smoke.json` | 软件路径检查，不是效果实验 | 2 / 2 | 4 | 全训练有效位置中每次 8 个 |
| `document_adapter_pilot.json` | 原生 KL 约束适配器主实验 | 24 / 8 | 256 | 每次选中文档 32 个位置 |
| `document_adapter_kl0_pilot.json` | 仅去掉 native KL 的消融 | 24 / 8 | 256 | 同 adapter 主实验 |
| `document_router_pilot.json` | cost-soft-target 路由主实验 | 24 / 8 | 256 | 全训练有效位置中每次 32 个 |
| `document_router_expected_ce_pilot.json` | 仅改 expected CE 的消融 | 24 / 8 | 256 | 同 router 主实验 |

两个 pilot 的 `256×32=8192` 只是最大监督位置访问量，不是 8192 个独立位置，
也不是固定 512-context 缓存。Adapter 每步处理一篇完整文档，在文档内轮换 mask；
router 优化监督表时轮换位置，probe 另跑实际全文策略。不同模块和目标的实际
调用量不同。默认全长 256，最多 24×255=6120 个有效 next-token 位置，短文档
会更少；独立文档量仍仅 24。

Router 的 uniform-action 标签生成预算为每篇 **29 次完整尾部调用**：28 个统一
动作加一个 native 对照。Smoke 为 58 次，pilot 为 696 次，不是每位置再乘 28；
它只覆盖这种明确的反事实条件，不枚举任意联合路由。训练有效 token 上界为
smoke 512 / pilot 8192，超界失败而非静默截断。

共同缓存上限 1 GiB、每次完整 logits 上限 16,777,216 个元素，不是整机显存需求
估算；模型权重、算子临时张量和反向开销另计。默认序列 256、词表 32000 的一次
完整输出为 8,192,000 个元素。不能在显存不足时私自改成 singleton 评分。

两个 smoke 固定学习率，`adaptive=null`；pilot 显式打开有界 train-probe 学习率
控制：初始 0.001，每 16 步观察，patience=3、min_delta=0.0001、下降因子 0.5，
边界 `[0.00001, 0.002]`；adapter 至多一次在非零梯度却无实际 BF16 修正等
条件成立时恢复性增长，router 不使用这个增长触发器。最少 64 步及完整
位置覆盖只是停止的必要条件，256 步硬上限不变。开发集耐心 4 次、min_delta
0.0001；具体停止原因与实际调度写入报告，不保证一定提前停止或省算力。
目标/KL 消融只改一个科学配置字段和输出路径，但优化反馈可能导致不同学习率
轨迹、早停步数和耗时；必须报告这些差异，不能因此声称严格等成本。

所有模板同协议重放默认绝对/相对容差均为 0。它是可证伪的严格起点，
不是已在真实多 GPU 上验证的承诺。若失败，先检查差异来源并预声明另一个
数值协议、新建 run；没有自动放宽功能。

## 7. 运行方法

从仓库根目录运行，使用仓库记录的依赖与本地完整 Mixtral 权重。先复制模板到
`local_configs/` 并修改真实模型、tokenizer、数据、设备与输出路径；不要把机器
专属路径改回提交的通用模板。每个阶段输出目录必须全新；三个阶段只有
`output_dir` 可以改变，其他合同字段保持一致。

```bash
mkdir -p local_configs
cp configs/mixtral/document_adapter_smoke.json local_configs/document_adapter_smoke.json
# 编辑 local_configs/document_adapter_smoke.json 中的本机路径和显卡映射。
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.mixtral_document \
  --config local_configs/document_adapter_smoke.json --stage tune
```

显卡编号只是新运行示例，不是服务器当前可用性证明。新实验三个阶段必须使用
一致的映射、模型文件与配置。历史 bridge 仍使用历史报告绑定的 GPU 映射，不能
用这个新运行示例覆盖旧身份。

后续 evaluate/profile 命令仅在前序实际 gate 通过时执行；必须使用报告记录的
精确 checkpoint 路径，不凭文件名推断哪个是有效 best。

```bash
# 先从同一 tune 配置创建两个副本，仅更改 output_dir 到两个全新目录。
python -m experiments.mixtral_document --config local_configs/document_adapter_evaluate.json \
  --stage evaluate --tune-report /absolute/path/to/tune/report.json \
  --checkpoint /absolute/path/to/frozen/best.pt
python -m experiments.mixtral_document --config local_configs/document_adapter_profile.json \
  --stage profile --tune-report /absolute/path/to/tune/report.json \
  --checkpoint /absolute/path/to/frozen/best.pt \
  --evaluation-report /absolute/path/to/evaluate/report.json
```

上面的绝对路径占位符必须换成真实产物，并不代表已经存在这些文件。
先 smoke，再分别运行 adapter/router pilot；不要一开始同时启动所有模板。
多 seed 17/29/43 是手动预声明的独立运行，seed 与输出目录都要更新，不能覆盖。
相同步数不等于公平预算：router 成本表、adapter teacher、不同吞吐和开发评估
需要分开计入总墙钟与模型调用成本。

## 8. 迁移与结果解释

新格式 checkpoint 与旧 fit/recovery 格式分离，不提供 exact resume。
旧文件只用于对应历史版本的复核，不能改写其来源哈希导入新训练器。
本地权重/数据/源码/数值身份变化应新建 run；模型上次加载成功不代表本次文件
未变化。不要删去阴性结果或失败报告，只保留成功 seed。

如果仍无开发集收益，应依据三类证据定位：

1. 同协议数值关失败：先解决执行/设备/形状问题，不分析训练效果。
2. train 全文指标不改善：调查目标、梯度、覆盖、数值有效更新和优化预算。
3. train 改善而独立 development 退化：优先研究覆盖与正则/容量，而不是继续
   只记忆旧位置。多 seed 无稳定开发收益时保留 baseline，暂不做成本优越性宣称。

CPU/tiny 模型测试用于软件正确性；真实多 GPU BF16 parity、学习收益、速度与显存
仍需服务器实测。只有质量稳定后再判断性能成本，不能把 kernel 较快替代质量验证。
