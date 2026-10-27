# 运行环境（Environment）

## 硬件与驱动

- GPU：NVIDIA RTX A6000（实验使用多卡，`CUDA_VISIBLE_DEVICES` 指定卡号）
- 驱动：595.58.03

## 软件版本

- Python 3.12.15
- PyTorch 2.12.1+cu130（CUDA 13.0）
- torchvision 0.27.1+cu130
- transformers 5.15.1 / tokenizers 0.22.2 / numpy 2.5.2
- 完整依赖版本见 `requirements.txt`（`pip freeze` 全量固定版本）

## 安装步骤

```bash
# 1. 创建虚拟环境
python3.12 -m venv .venv
source .venv/bin/activate

# 2. 安装 PyTorch（CUDA 13.0 版本，+cu130 后缀需要官方 wheel 源）
pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
    torch==2.12.1+cu130 torchvision==0.27.1+cu130

# 3. 安装其余依赖
pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
    -r requirements.txt
```

注意：`torch==2.12.1+cu130` 带本地版本后缀，PyPI 上没有，
必须加 `--extra-index-url https://download.pytorch.org/whl/cu130`，否则会报找不到版本。

## 验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 预期输出: 2.12.1+cu130 True

# 运行 RL 分支完整测试（不依赖 GPU 的部分）
python -m unittest discover -s tests -t . -v
```

## 分支与数据

- RL 代码在分支 `research/rl-lm-pair-v1`；`main` 为此前 MoE 基线工作。
- 原始数据、packed cache（`data/rl_lm_pair_v1/`、`data/rl_lm_single_v1/`，约 7GB）与 checkpoint（`*.pt`）不进入 Git。
  复现实验时需用 `experiments/rl_lm_prepare.py` / `rl_lm_corpus.py` 重新生成，
  数据源为 HuggingFace `fineweb-edu`（sample-10BT），身份见各 run 目录下的
  `source_identity.json` / `source_manifest.json`。

## CPU correctness checks (not an experiment reproduction)

The CPU-only suite can be run without installing the CUDA freeze or downloading
training data/checkpoints:

```bash
python3.12 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -v
```

GitHub Actions runs this same CPU suite. Passing it does not establish CUDA BF16
parity, A6000 speed, or training quality; those require the separately recorded
GPU validation protocol. The original CUDA requirements remain unchanged.
