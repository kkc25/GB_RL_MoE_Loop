# PR#11 历史 checkpoint 缓存／全文桥接诊断

- 诊断代码：`main@2511dd2` + 多卡索引设备修复。
- 历史对象：`fb7ecd6` 下生成的 router/adapter 64-context train-best checkpoint。
- 不训练、不用 selection 选模型、不访问 newtest。
- 初次诊断因 native logits 在 cuda:0、末层索引在 cuda:3 而失败；修复后 bridge 测试 17/17 通过，以下为新目录的有效诊断。

## 历史缓存复现

两者均精确复现历史 cache 指标，说明 checkpoint、训练位置和缓存绑定正确：

- Router：native `1.8767921282`、learned `1.8057594121`。
- Adapter：native `1.8825348532`、learned `0.0265989018`。
- Router 的 native/zero/learned 动作在 64/64 位置全部一致。

## Cache 与完整文档执行的差异

两者的 `numerical_screen_passed=false`、`zero_screen_passed=false`，不能宣称缓存训练分数与完整在线执行等价。

| 诊断项 | Router | Adapter |
|---|---:|---:|
| cached h0 最大差 | 0 | 0 |
| native hidden 最大差 | 8.0 | 8.0 |
| native CE 最大差 | 0.1787 | 0.1810 |
| native CE 平均差 | 0.0222 | 0.0186 |
| learned CE 最大差 | 0.1396 | 0.00492 |
| learned logits 最大差 | 0.3203 | 0.25 |

同一完整文档内 native/controlled/zero 的 logits、hidden、CE 均逐位一致；差异发生在 cached singleton/probe 与 full-document 的执行形状之间，不是原生受控路径本身错误。

## 训练位置与未训练位置

### Router

- 64 个训练位置：在线 native 1.890769 → learned 1.817432，改善 0.073337 nat；缓存改善 0.071033 nat。
- 同文档其余 1828 个未训练位置：native 1.929946 → learned 1.957412，恶化 0.027466 nat。

### Adapter

- 64 个训练位置：在线 native 1.890769 → learned 0.026535，改善 1.864234 nat；缓存改善 1.855936 nat。
- 其余 1828 个未训练位置：native 1.929946 → learned 3.137868，恶化 1.207922 nat。

这直接证明历史拟合收益在完整文档训练位置上也存在，不只是缓存假象；但策略/修正严重专门化到训练位置，对同文档其他位置造成损失。过拟合是主要研究问题，cache/full 数值差异也需要独立处理，二者不能互相替代解释。

## 决策

按 PR#11 fail-closed 规则，当前 native/zero 桥接容差未通过，**不启动 recovery tune**，也不放宽容差强行训练。后续若要新训练，需要先预先声明并验证一致的评分/在线形状协议；模型学习方面则应优先解决训练位置集中和 development 泛化，而非继续优化旧 64 点。
