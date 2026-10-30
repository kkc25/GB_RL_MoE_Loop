# 当前运行请求：Mixtral 数据规模与固定幅度实验 v2

**当前状态：代码与 CPU 验证已交付，GPU 正式实验尚未执行；新语料尚未准备或冻结。**

本文件是当前默认入口。完整准备、预检、正式执行、恢复、预算及验收流程见 **[RUN_REQUEST_LOOP_SCALE_V2.md](RUN_REQUEST_LOOP_SCALE_V2.md)**，研究协议见 [LOOP_SCALE_EXPERIMENTS.md](LOOP_SCALE_EXPERIMENTS.md)。

- 精确分支：`experiment/mixtral-loop-data-scale-v2-20261030`。
- 执行模块：`python -m experiments.mixtral_loop_scale`；配置入口：`local_configs/loop_scale_v2.json` 或 `configs/mixtral/loop_scale_v2.json`。
- 执行身份：使用本 PR 最终交付的完整 40 位 commit SHA 创建独立 worktree，并传入 `--expected-commit`；不得使用浮动分支、旧 v1 SHA 或已有旧输出目录代替。
- 当前实验：固定 100k→300k 的有限数据对比，按同结构改善趋势决定是否进入 1m；每档 90% adapter-train、10% 独立 gate-fit；开发校准/选择分离，最终测试封存。固定两遍训练，epoch0/1/2 检查点，支持绑定 optimizer/RNG/cursor/cost 的显式恢复。
- 资源：现有四张 A6000，Mixtral-8x7B-v0.1 BF16、固定 eager 专家后端；只使用已存在本地模型和显式输入，不自动下载或启动服务器。

旧 24 篇/8 epoch/no-resume 的 v1 请求已逐字节存档到 **[历史 RUN_REQUEST v1](docs/history/RUN_REQUEST_TRAINABLE_LOOP_V1_3559a307.md)**，仅用于追溯，不能作为本轮执行指令。其内容来自 PR #14 的 `3559a307cb8af347444b09444044bfac31084cc0`；旧代码、阴性结果及历史分支保持原样。
