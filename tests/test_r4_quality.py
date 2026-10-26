import unittest

import torch

from config import ModelConfig
from eval.counterfactual import PathCosts, candidate_actions
from experiments.r4_quality import depth_metrics
from model import LoopMoE


class DepthMetricsTests(unittest.TestCase):
    def test_returns_metrics_for_every_depth(self):
        torch.manual_seed(7)
        config = ModelConfig(
            layers=3,
            hidden_size=32,
            attention_heads=2,
            head_dim=16,
            vocab_size=64,
            intermediate_size=48,
            experts_per_layer=4,
            top_k=2,
            adaptive_layers=(2,),
            moe_layers=(0, 1, 2),
            max_calls=4,
            target_only=False,
            max_position_embeddings=32,
        )
        model = LoopMoE(config).eval()
        h = torch.randn(2, config.hidden_size)
        actions = candidate_actions(model, h)
        costs = PathCosts(
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            torch.zeros(2, dtype=torch.long),
            actions,
            4,
        )
        states = {
            "h": h,
            "targets": torch.tensor([1, 2]),
            "costs": costs,
        }
        metrics = depth_metrics(model, states, (2, 3), torch.device("cpu"), "float32")
        self.assertEqual(set(metrics), {2, 3})
        for depth in (2, 3):
            self.assertEqual(metrics[depth]["losses"].shape, (2,))
            self.assertTrue(torch.isfinite(metrics[depth]["losses"]).all())
            self.assertGreater(metrics[depth]["nll"], 0)


if __name__ == "__main__":
    unittest.main()
