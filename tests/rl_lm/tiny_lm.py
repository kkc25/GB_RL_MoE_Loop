"""Shared tiny real LoopMoE fixture for rl_lm contract tests.

A small but structurally real model: MoE blocks with the Top-2 executor, a
terminal adaptive layer, 8 experts, RMSNorm/gate/gamma exactly as in the
100M backbone.  CPU + FP32 compute keeps the math tests exact; the BF16
production contract is exercised separately on GPU.
"""

from __future__ import annotations

import torch

from config import ModelConfig
from model.transformer import LoopMoE

TINY_DIM = 32
TINY_VOCAB = 64
TINY_EXPERTS = 8


def build_tiny_model(seed: int = 17, layers: int = 2) -> LoopMoE:
    torch.manual_seed(seed)
    config = ModelConfig(
        layers=layers,
        hidden_size=TINY_DIM,
        attention_heads=4,
        head_dim=TINY_DIM // 4,
        vocab_size=TINY_VOCAB,
        intermediate_size=48,
        experts_per_layer=TINY_EXPERTS,
        top_k=2,
        adaptive_layers=(layers - 1,),
        moe_layers=tuple(range(layers)),
        max_calls=2,
        target_only=False,
    )
    model = LoopMoE(config)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
