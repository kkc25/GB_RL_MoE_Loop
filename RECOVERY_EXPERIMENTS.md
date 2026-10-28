# Mixtral recovery：先桥接，再用独立开发文档选 checkpoint

这是独立、版本化的研究入口 `experiments.mixtral_recovery`，配置 schema 为
`mixtral-generalization-recovery-v1`，顺序为 **历史桥接 → tune → evaluate → profile**。
它不是修补旧结果、更改旧 fit-first 的 best 定义或将旧 checkpoint 接着训练。
旧 [FIT_FIRST_EXPERIMENTS.md](FIT_FIRST_EXPERIMENTS.md) 的 train-only baseline 保持原样。

**本次交付是代码、配置和 CPU 验证，不是新一轮 GPU 效果结果。** 不自动下载权重、
启动 GPU、追加预算或读取最终 held-out/newtest。所有参数都是待验证的研究默认值，
不保证收敛、泛化、数值等价、加速或最优。

## 1. 现有结果究竟说明什么

结果上传 commit 为 `bdca6a3`；其中 `runs/mixtral_fit_v1/*_64_pr10/` 的实际执行
源码仍是 `fb7ecd6`。上传版本与执行源码不是同一概念，不应把这解释为身份错配，
也不能把旧产物改成新源码身份。

- Router 的 64-context train greedy CE 平均改善约 **0.071033 nat**，但原 oracle
  gap 约 0.180609，仍剩 0.109576；只有 3/64 动作与精确 oracle 一致。熵从约
  3.08 在 8 步内降到 0.0524，少数位置主导收益：两个已记录 probe token 的
  收益合计约占完整 64-context **净收益的 69.83%**。这是过早饱和、局部/集中
  改善与剩余可改善空间的联合证据，不是“已经全面拟合”；仅低熵本身并非失败。
  近似并列的动作也会影响 oracle agreement，因此要同时看 cost-weighted regret。
- Router 在 16 篇 selection 开发文档、3967 个有效 token 上，NLL 从
  **1.816940 → 1.844256**，13/16 篇恶化。训练收益不能替代部署形状的质量收益。
- Adapter 的 train probe NLL **1.882535 → 0.026599**，selection NLL 却达到
  **2.489728**，16/16 篇全部恶化。这支持“记住少量训练位置、未泛化”的诊断。
- 两者实际只有 **8 篇文档里的 64 个位置**，占这些文档 1892 个有效位置约
  **3.38%**；256 次 batch=8 更新相当于 32 个位置遍历，不是 32 份独立数据。

上述结果不支持继续照原样堆迭代，也不否定所有路由、残差适配或循环方法。
新协议先排除 cache/在线评分差异，再将 checkpoint 选择从 train 转移到独立
`development` 文档；这不是宣称 native 正则一定能修复泛化。

## 2. 桥接：把数值/执行问题与学习问题分开

### 2.1 历史 checkpoint 的独立诊断

`python -m experiments.mixtral_bridge` 读取原 fit 配置、原报告和原 checkpoint，
即使旧 fit 或 generalize gate 未通过，也可以诊断。它不改写历史 binding，
不启动新训练、不拿 selection 来挑候选，也不将旧格式导入新训练器。

在**原来采样的同一文档、同一 token 位置**比较 cache 与完整文档执行：
原生 logits/NLL/隐藏状态/特征、强制相同动作的结果、以及适配器修正结果。
必须固定上下文、position/mask、next-token label shift、dtype、eval mode、
执行后端与损失范围。单点干预和联合干预分开报告；完整文档总 NLL 不能冒充
“原 64 个位置”的指标。

这里干预的是**最后一个 transformer block 的 pointwise FFN**。它之后没有
新的跨 token attention：同文档同时改多个末层 FFN 位置，并不会形成额外的
跨 token attention 反馈。单点/联合差异可能来自执行批形状、GEMM/路由数值等，
必须测量，不能发明“后续 attention 分布漂移”来解释该实现。

报告分别保留历史模型/数据/权重/源码身份与当前诊断源码身份；检查旧训练所依赖
的核心源码字节，并明确历史版本与新诊断模块的边界。报告/哈希是完整性检查，
不是抵抗主动伪造的签名认证。

### 2.2 新 tune 的 fail-closed 前置关

新入口用**当前**实际模型、tokenizer、四角色数据、采样位置、数值及源码重新
建立身份并做校准/桥接。历史桥接成功不是所有未来输入/形状都等价的通行证。
只有当前 native/zero 前置检查通过，才允许训练；evaluate/profile 也不能复用过期身份。
训练后还检查冻结 learned candidate 的桥接：如果此项失败，保留
`training_path_equivalence_established=false`，禁止把 cache 拟合改善转述为在线
等价收益；但 development 质量通过时仍可对唯一冻结候选做完整在线 outer
evaluate，以诊断实际部署形状的质量。**profile 则同时要求 learned 数值 screen
通过和 outer selection 质量通过。** 即使 screen 通过，也只是测过的位置/形状在
声明容差内，不自动建立严格数值等价或推广保证。

配置必须显式声明 `recovery.bridge_tolerances` 的
`hidden_max_abs`、`feature_max_abs`、`ce_max_abs`；同时保留顶层
`logit_max_abs_tolerance`、`ce_mean_abs_tolerance`、`nll_tolerance`。
模板分别使用 `1e-6 / 1e-6 / 1e-4` 与 `1e-3 / 1e-4 / 1e-4`，
**不是已经验证的 CUDA BF16 容差**。真实 GPU 的 singleton/full-document 批形状
差异可能使运行在训练前被拒绝；这是有意义的结果，不是自动放宽容差的理由。
native/zero screen 失败只说明当前训练前提不满足，不宣布方法不可学习。
若要研究另一数值协议，必须先解释差异、预声明新阈值/形状、新建 run 并保留失败，
不得通过修改旧报告、强行重绑定或自动 widening 让实验继续。

## 3. 四个数据角色与实际文件划分

| 角色 | 可以做什么 | 不能做什么 |
|---|---|---|
| `train` | 固定位置监督、构建全动作成本表、训练损失与诊断 | 单凭 train loss 选最终 checkpoint |
| `development` | 完整文档在线 NLL；checkpoint 选择、早停，step 0 参选 | 与 train 共用文档或有效输入 |
| `selection` | 外层开发筛选；只评估一个已冻结 development-best | 在 evaluate 中挑另一个 checkpoint；叫作独立最终测试 |
| `calibration` | 原生/controlled 数值一致性检查 | 训练损失、checkpoint 选择 |

这四个角色全部属于开发协议。尤其先前已经看过的 16 篇 selection，不能因改名
就重新成为“未触碰测试”。本入口不支持最终 held-out/newtest；最终确认需要
另行冻结方案、范围和独立数据，不在本轮实现或授权执行范围内。

预检对四个角色的**完整有界来源文件**做 ID、规范化原文与截断后有效 token
检查，包括未被 `sequences` 的 first-N 选中的行。跨角色重复直接拒绝，同角色
入选文档的有效输入重复统计进入身份。不同文件名/不同 ID 不保证独立，精确去重也不证明
语义独立性。训练过程中的 `train-only` 监督不意味着预检没有打开其他角色文件。

**必须物理拆分新 manifest**：从旧 train 划给 development 的文档，必须从新的
train 来源文件中移除。只把旧 train 的“未采样”行复制到 dev，同时继续将整个
旧 train.jsonl 配成新 train，是错误的；即使 train 只选前 8 行也会被拒绝。
历史源文件保留不动，不能改写旧实验输入。

历史报告的完整来源身份记录 train/selection/calibration 各 64 篇，因此下面
24+8 的拆分不要求凭空增加超过现有 train 文件的文档数。脚本要求实际源文件至少
32/16/8 篇，不足时停止，绝不复制补齐：

```bash
# 在模型/数据所在机器执行。先人工确认这三个源文件确实是历史开发数据。
# 仅产生新的数据 manifest，不训练、不评估；目标目录必须尚不存在。
python - <<'PY'
import json
from pathlib import Path
source = Path('/data/mixtral/dev')       # 改为真实历史开发数据目录
out = Path('/data/mixtral/recovery-v1')  # 与新配置保持一致

def read(name, count):
    rows = [json.loads(line) for line in (source / (name + '.jsonl')).read_text().splitlines() if line.strip()]
    if len(rows) < count:
        raise ValueError(f'{name}: need >= {count} source documents, got {len(rows)}')
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
            raise ValueError('duplicate ID or normalized raw document in new manifests')
        seen_ids.add(ident); seen_text.add(text)
out.mkdir(parents=True, exist_ok=False)
for role, rows in parts.items():
    (out / (role + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    print(role, 'source documents =', len(rows))
# Tokenizer-dependent full-source effective-input checks still run in recovery preflight.
PY
```

新来源清单固定为 **24 train / 8 development / 16 selection / 8 calibration**。
64-position 诊断只取新 train 前 8 篇，但完整 24 篇仍接受去重/身份检查；
512-position 模板取全部 24 篇。每角色先显式取前 N 篇，再对 train 做 seed 固定的
文档平衡、位置无放回采样，不是对全语料随机抽文档。剩余历史源文档不在本次新
manifest 中，也不被悄悄称作独立最终测试。有效位置总数不足时必须停止。

## 4. Router：稳定监督与合法紧凑特征

新 router 保留全部 28 个无序专家对以及真实 native fallback；只训练新增路由
参数。它仅接受 R1 状态，从旧 `2D+20` 特征中保留 `n` 与 8 维居中 gate logits，
得到 `D+8` 维紧凑特征，做逐行 RMS 归一化。R1 中 `n0=n`、前一专家/轮次等
常量不再重复拼接；`n0=n` 本身不是旧实现 bug，也不是新增状态证据。
残差线性 logits 显式乘 **`1/sqrt(D+8)`**，零初始化保持原生 greedy。

两个有界对照使用相同特征、初始化、数据、LR、更新上限、开发评估频率与 prior：

- `expected_ce`：`sum_a pi(a|x) * c_a`，保留原目标作独立对照；
- `soft_target_ce`：`q = softmax(-(c - row_min(c))/temperature)`，
  `L = -sum_a q_a log pi_a`。减最小值只是稳定数值；等价于减 native cost，
  不改变 q。该目标的 logit 梯度为 `pi-q`，避免只依赖概率很小动作的期望损失梯度。

训练成本表来自 gold next-token label，**只是 train 监督**。推理签名没有 label、
成本表、oracle 或事后收益；不能用真实目标选择动作或开启适配器。

可选正则为 `reference_kl_weight * KL(pi || p0)`，其中
`p0=(1-epsilon)*onehot(native)+epsilon/28`，模板 `epsilon=.8` 为全支持静态先验。
它不是与 one-hot native 的无穷 KL，也不是 step KL/current-versus-previous 的
trust region。`.8` 也不是部署随机探索率；静态 prior 与持续 KL 正则须分开理解。
模板比较 `.01` 与可选 `0`，温度 `.1`、LR `.001` 均未证明最优。

固定训练表记录 greedy/expected CE、cost-weighted regret/oracle gap、动作
变化、熵、概率/margin、按位置和文档收益分布、访问覆盖与数值状态。训练拟合
标签还要求至少 3 个正收益位置、正收益比例至少 .5、平均改善至少 .005 nat；
这是预声明诊断，不用两点拉动的均值代表全面拟合，也不能替代 development gate。

## 5. Adapter：native 输出约束，不再只记住 64 个点

新适配器仍是 native R1 后的低秩单次残差，模板 rank=8、scale=.1：

`L = CE(gold, corrected) + beta * KL(p_native || p_corrected)
     + lambda * mean(||correction||² / max(||h1||², 1e-12))`

- native teacher 是相同缓存 h1、相同分块形状经过冻结 norm/head 的**全词表软
  输出**，T=1、detach；不是 native argmax 硬标签，也不反传进主干。
- `native_kl_beta=.1`，可选 `relative_correction_l2=0` 默认关闭。teacher 按块
  重新计算，不保存全训练集 logits；额外 head 成本必须单列。
- 一个低秩矩阵随机初始化、另一个零初始化，step 0 精确保持 native；不能两者
  同时全零导致无梯度。development-best 可以是 0，持平保留最早；没有可靠改善
  时保留 native，不能强迫选非零更新。
- 固定训练 probe、实际访问覆盖、FP32 提议修正与 native/BF16 cast+相加后的
  有效改变同时记录；非零 FP32 norm 不等于实际 BF16 输出变化。

本实现不重跑原专家、不训练 STOP、不加入跨 token attention。它检验新增修正
容量，不是已实现 recurrent MoE 或 loop 的因果收益。更小 rank、其他 scale/KL
可以作为另行预声明的小规模消融；此交付不启动无界网格搜索。

## 6. Checkpoint 选择、早停与阶段门槛

每个新训练器保存 initial/latest/best 以及被评估更新的 checkpoint。训练期按
固定间隔在完整 `development` 文档上在线评估；不是缓存 train CE，也不是
selection。step 0 参与，持平保留最早；`development_min_delta` 与
`development_patience`、`min_updates` 预先固定，最晚受 `updates` 硬上限约束。`development_min_delta` 控制耐心计数，
不是挑选 best 的最小差值：任何严格较低的 development NLL 都可成为 best。

模板固定 LR、`adaptive=null`，不自动扩大预算。实现允许可选 train-only 有界
LR/plateau 控制；它可以独立以 `adaptive_train_plateau` 停止训练，但不能选
checkpoint 或把 train plateau 叫作最佳泛化点。development 早停与硬预算另有
独立原因记录；development patience 须达到 min_updates 且完整训练覆盖后才触发。
不同 batch 的零散损失不是早停依据。

- **tune**：前置原生一致性和 cache/在线桥接通过后训练；development 选择冻结
  best，并检查 `development_min_gain=.001 nat/token`。如果 best=0 或质量不够，
  保留正常的阴性记录/native fallback；不进入 profile。
- **evaluate**：只接收同一个已冻结 development-best，使用外层 selection 做
  native/controlled/candidate 完整文档配对检查，不更新参数、不另挑 checkpoint。
  `selection_min_gain=.001` 是预声明开发筛选阈值，不是显著性或部署验收标准。
- **profile**：要求匹配的 tune/evaluate 报告、通过的质量关、learned bridge 数值
  screen 通过，以及同一 checkpoint。
  不能只看到 `status=complete` 就认为质量通过，不能为了计时降低事后阈值。

development gate 计算 step0→best 的完整文档 NLL 改善；step0 是 native 初始化。
selection gate 明确要求相对 native 和 controlled 两种 gain 均达到阈值。
两者都保留逐文档 native/controlled 数值 screen、配对结果与总体 token 加权 NLL；
文档等权均值另列，不能混用。
观察过 selection 再调整方案仍属于开发过程，不因此获得新测试资格。

## 7. 有界 run matrix 与成本台账

完整结果应**先列实际工作量/成本**，再解释质量，不用相同步数伪装预算匹配。
所有下表都是计划上限；实际 early stop、数值 gate、padding 与后端会改变成本。

| 成本项 | Router | Adapter | 必须如何报告 |
|---|---|---|---|
| 固定训练输入 | train 文档 prefix/cache | train 文档 prefix/cache + native h1 | 实际文档/有效位置、cache rows、墙钟时间 |
| 一次性监督 | 每 context 28 候选 + 2 native anchor；30 score calls/rows | 无全动作表 | 表构建与优化分账，不是总 GPU 调用数 |
| 每次训练更新 | 读固定表的轻量路由计算 | corrected head 前向/反向 + native KL teacher head 前向 | head/backward/teacher rows、padding 与耗时分别列 |
| 固定 train probe | 全表诊断 | corrected/native teacher readout + 修正诊断 | 不与 development 文档评估混算 |
| development 选择 | 完整文档在线 candidate；native/controlled 对照 | 相同角色与协议 | 实际评估次数/文档/有效 token/耗时；不是免费调用 |
| 桥接/校准/evaluate/profile | 各自独立阶段 | 各自独立阶段 | 另列，不并进训练步数声称预算相同 |

CPU tiny 测试不提供 GPU-hours、真实 46.7B 成本或速度。只有实际测量才能填这些
数值，不把 score calls、head rows、文档前向或逻辑专家调用当同价单位直接相加。

| `configs/mixtral/` 模板 | 目的 | contexts / updates 上限 | 实际选用文档 train/dev/selection/cal | 长度 |
|---|---|---:|---:|---:|
| `recovery_router_smoke.json` | router 流程检查 | 8 / 4 | 2/2/2/2 | 128 |
| `recovery_adapter_smoke.json` | adapter 流程检查 | 8 / 4 | 2/2/2/2 | 128 |
| `recovery_router_expected_ce_64.json` | 相同紧凑特征上的期望 CE 对照 | 64 / 256 | 8/8/16/8 | 256 |
| `recovery_router_soft_target_ce_64.json` | cost soft-target CE，KL=.01 | 64 / 256 | 8/8/16/8 | 256 |
| `recovery_router_soft_target_ce_64_kl0.json` | 可选：只去掉 native prior KL | 64 / 256 | 8/8/16/8 | 256 |
| `recovery_adapter_anchored_64.json` | native 输出 KL 锚定适配器 | 64 / 256 | 8/8/16/8 | 256 |
| `recovery_router_soft_target_ce_512.json` | 有诊断支持后扩大位置与文档覆盖 | 512 / 256 | 24/8/16/8 | 256 |
| `recovery_adapter_anchored_512.json` | 锚定适配器扩大覆盖 | 512 / 256 | 24/8/16/8 | 256 |

不是要求把 8 项全部跑完：先每种 smoke，再有界 64-position 诊断；KL0 是可选
单因素对照，512 是另一个新 run，不是续训或自动下一步。新旧 router 同时改变
特征/缩放/监督/选择规则，不能把整体差异都归因于 soft targets；只有新矩阵中
expected-versus-soft 那一对保持其余设置一致。

Router 表构建预算为 8→240、64→1920、512→15360 score calls；每个非smoke
最多 256×8=2048 个训练 context visits。512 位置平均访问机会更少，不能声称
同样充分拟合。非smoke每16步评估、含step0且最后一步检查，最多17次预定开发
观察；另为一次 off-schedule 早停检查保留有界额度，runner 上限18次；实际
次数以报告为准。smoke每2步、最多3次预定观察，runner 总预算4次。
非smoke profile 预声明 warmup=3、repeats=3、measurements=32；smoke=1/1/3，
只验证流程，不足以得出速度结论。

## 8. 手动执行命令

先在 CPU 验证配置；以下所有 `CUDA_VISIBLE_DEVICES` 命令是操作者手动启动的
真实 GPU 工作，**不是本次交付已经执行的操作**。每个阶段必须使用全新目录，
禁止一个 shell 串自动从失败训练继续跑 evaluate/profile。

### 8.1 先诊断历史产物

保持历史配置内容和绑定不动；`--output-dir` 仅指定新诊断目录。显式容差文件
先写好，再运行；不要看过差异后原地放宽。
本仓库两份 PR #10 历史报告分别绑定 router 的 `CUDA_VISIBLE_DEVICES=0,3,5,7`
与 adapter 的 `1,2,4,6`，下面保留原值。还需保留原 runtime、模型/数据路径与
不可变文件身份；原环境或产物不可用时，不能改写绑定来强行通过预检。

```bash
cat > local_configs/historical_bridge_tolerances.json <<'JSON'
{"hidden_max_abs":1e-6,"feature_max_abs":1e-6,"ce_max_abs":1e-4,"ce_mean_abs":1e-4,"logit_max_abs":1e-3}
JSON

CUDA_VISIBLE_DEVICES=0,3,5,7 python -m experiments.mixtral_bridge \
  --config local_configs/fit_router_64_pr10.json \
  --fit-report runs/mixtral_fit_v1/router_64_pr10/report.json \
  --checkpoint runs/mixtral_fit_v1/router_64_pr10/fit/checkpoint_best.pt \
  --output-dir runs/recovery-historical-router-bridge-001 \
  --tolerances local_configs/historical_bridge_tolerances.json
```

适配器另一次人工运行（不要与上一条自动串联）：

```bash
CUDA_VISIBLE_DEVICES=1,2,4,6 python -m experiments.mixtral_bridge \
  --config local_configs/fit_adapter_64_pr10.json \
  --fit-report runs/mixtral_fit_v1/adapter_64_pr10/report.json \
  --checkpoint runs/mixtral_fit_v1/adapter_64_pr10/fit/checkpoint_best.pt \
  --output-dir runs/recovery-historical-adapter-bridge-001 \
  --tolerances local_configs/historical_bridge_tolerances.json
```

若历史 checkpoint 二进制不在当前检出目录，先由持有原始产物的机器提供原文件；
不能用随机初始化、新训练或修改报告来补出“历史 checkpoint”。

### 8.2 准备新的 recovery 配置

下面选 soft-target64 作为命令示例；先 smoke 时把模板换成相应 smoke，并使用
另一组新目录。预先修改真实模型/tokenizer 路径、数据路径、backend、GPU 映射与
预算；所有改变必须在 tune 前完成。

```bash
python - <<'PY'
import json
from pathlib import Path
from mixtral_lab.recovery_protocol import validate_recovery_config
c = json.loads(Path('configs/mixtral/recovery_router_soft_target_ce_64.json').read_text())
c['model_path'] = c['tokenizer_path'] = '/models/Mixtral-8x7B-v0.1'  # 改为真实离线目录
c['output_dir'] = 'runs/recovery-router-soft64-s17-tune-001'
validate_recovery_config(c)  # 结构校验；不加载权重、不启动 GPU
Path('local_configs').mkdir(exist_ok=True)
p = Path('local_configs/recovery_router_soft64_s17_tune.json')
with p.open('x') as f:
    json.dump(c, f, indent=2)
print(p)
PY

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.mixtral_recovery \
  --config local_configs/recovery_router_soft64_s17_tune.json --stage tune
```

人工检查 tune 的数值/桥接结果、训练分布诊断、development 曲线、best_update
以及 `development_gate`。best=0 或关口失败是需要保留的结果，不自动追加训练。

### 8.3 Evaluate：只评估一个冻结候选

仅从相同已验证配置复制并改变 `output_dir`；不更改 seed、数据、容差、选择阈值、
训练设置或 profile 参数。可预先创建两个下游配置，但不要自动执行阶段：

```bash
python - <<'PY'
import json
from pathlib import Path
c = json.loads(Path('local_configs/recovery_router_soft64_s17_tune.json').read_text())
for stage in ('evaluate', 'profile'):
    d = dict(c)
    d['output_dir'] = f'runs/recovery-router-soft64-s17-{stage}-001'
    with Path(f'local_configs/recovery_router_soft64_s17_{stage}.json').open('x') as f:
        json.dump(d, f, indent=2)
PY

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.mixtral_recovery \
  --config local_configs/recovery_router_soft64_s17_evaluate.json --stage evaluate \
  --tune-report runs/recovery-router-soft64-s17-tune-001/report.json \
  --checkpoint runs/recovery-router-soft64-s17-tune-001/tune/checkpoint_best.pt
```

### 8.4 Profile：质量门槛之后，同一个候选

人工检查 evaluate 逐文档与总体质量关以及 learned bridge 数值 screen；任一
profile 前提未通过就停止计时，保留诊断结果。两者都通过后另行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.mixtral_recovery \
  --config local_configs/recovery_router_soft64_s17_profile.json --stage profile \
  --tune-report runs/recovery-router-soft64-s17-tune-001/report.json \
  --checkpoint runs/recovery-router-soft64-s17-tune-001/tune/checkpoint_best.pt \
  --evaluation-report runs/recovery-router-soft64-s17-evaluate-001/report.json
```

Adapter、expected_ce、KL0、512 和多 seed 都用各自模板、新输出和各自证据，
逐项人工决定是否执行。多 seed 17/29/43 要同时改顶层 `seed` 与
`recovery.training.seed`；其余对照设置固定。不提供自动 GPU sweep/resume。

## 9. 产物、迁移与验证边界

每阶段 `report.json` 是首要入口；失败时查看阶段/原因与是否已经加载模型，
而不是要求尚未完成的阶段存在所有训练产物。tune 的 `tune/` 下保存训练指标、
initial/best/latest、被评估更新的 checkpoint，router 还有固定训练成本表。
Router 的评估快照位于 `tune/evaluation_checkpoints/checkpoint_XXXXXX.pt`；
Adapter 位于 `tune/checkpoint_eval_XXXXXX.pt`。Router summary 的
`score_work_totals`、Adapter 的 `work_accounting` 与顶层
`report.work_accounting` 是不同层次的实际工作量，不能只读其中一项。
冻结信息绑定实际 checkpoint 文件/状态、四角色输入、采样、模型/tokenizer、
源码、数值和完整实验契约；只有 output_dir 允许在三个阶段之间变化。

- 新 router/adapter 各有独立 checkpoint format，不接受历史 train-best 或
  selection-best 格式作为新训练输入；历史 bridge 是诊断例外，不是迁移入口。
- evaluate/profile 重建当前身份并核验关联报告/文件；不能拿另一个 seed 的报告
  给当前候选开门，也不能换 latest 或任意被评估 checkpoint 再沿用旧 best 证据。
- 不提供 exact resume；失败目录不可覆盖。修改源码/依赖/数值/数据后重新运行，
  保留历史目录，不修 binding 伪装“同一个实验”。
- `status=complete`、训练有非零梯度、dev均值改善、CPU通过分别代表不同证据，
  没有一个能自动推出正式泛化、CUDA正确、部署可用或 loop 因果收益。

```bash
# 不加载真实权重的配置测试
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest tests.test_mixtral_recovery_configs -v
# 全套测试；CUDA 项的跳过必须与通过项分别报告
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -v
```

还未实现/尚未建立的内容包括：独立最终 held-out 确认、正式文档聚类不确定性分析、
通用超参数最优性、GPU质量/速度收益、真实重复专家+learned STOP、参数/实际计算
匹配的 loop 因果对照。更保守的约束只是一种可检验假设；若开发文档没有可靠改善，
保持 native 是正确输出，不是实验需要绕过的障碍。
