from __future__ import annotations

import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ModelConfig
from model import LoopMoE
from model.moe import DenseSwiGLU, MoE
from model.transformer import Attention


class RecoveryModelTests(unittest.TestCase):
    def test_dense_and_single_expert_dispatch_match_forward_and_gradients(self):
        torch.manual_seed(19)
        dense = DenseSwiGLU(16, 24).double()
        moe = MoE(16, 24, 1).double()
        with torch.no_grad():
            moe.pool.w1[0].copy_(dense.w1.weight)
            moe.pool.w3[0].copy_(dense.w3.weight)
            moe.pool.w2[0].copy_(dense.w2.weight)
        x_dense = torch.randn(3, 5, 16, dtype=torch.float64, requires_grad=True)
        x_moe = x_dense.detach().clone().requires_grad_(True)
        dense_output = dense(x_dense)[0]
        moe_output = moe(x_moe, torch.zeros(3, 5, dtype=torch.long))[0]
        torch.testing.assert_close(moe_output, dense_output, rtol=1e-12, atol=1e-12)
        dense_output.square().sum().backward()
        moe_output.square().sum().backward()
        torch.testing.assert_close(x_moe.grad, x_dense.grad, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(moe.pool.w1.grad[0], dense.w1.weight.grad, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(moe.pool.w3.grad[0], dense.w3.weight.grad, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(moe.pool.w2.grad[0], dense.w2.weight.grad, rtol=1e-10, atol=1e-10)

    def test_final_moe_uses_dense_prefix(self):
        config = ModelConfig(moe_layers=(3,))
        model = LoopMoE(config)
        self.assertTrue(all(isinstance(model.blocks[index].moe, DenseSwiGLU) for index in range(3)))
        self.assertIsInstance(model.blocks[3].moe, MoE)

    def test_main_parameter_counts_unchanged_by_default_moe_layers(self):
        config = ModelConfig(layers=8, hidden_size=512, attention_heads=8, head_dim=64, vocab_size=32768, intermediate_size=1024, experts_per_layer=8, adaptive_layers=(4, 5, 6, 7), max_calls=4, target_only=False)
        with torch.device("meta"):
            model = LoopMoE(config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 125870592)

    def test_document_blocks_are_isolated(self):
        config = ModelConfig(layers=2, hidden_size=32, attention_heads=2, head_dim=16, intermediate_size=48, experts_per_layer=2, adaptive_layers=(1,), target_only=False, vocab_size=64)
        attention = Attention(config).double().eval()
        first = torch.randn(1, 8, 32, dtype=torch.float64)
        changed = first.clone()
        changed[:, :4] = torch.randn(1, 4, 32, dtype=torch.float64)
        segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        with torch.no_grad():
            original_output = attention(first, segments)
            changed_output = attention(changed, segments)
        torch.testing.assert_close(original_output[:, 4:], changed_output[:, 4:], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
