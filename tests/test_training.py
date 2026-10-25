from __future__ import annotations

import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ControllerConfig, ModelConfig
from data.affine import sample_batch
from eval.oracle import all_paths, exact_path_batch
from model import LoopMoE
from policy import Controller
from train.halt import HaltHead, halt_distribution


class TrainingObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = ModelConfig(layers=2, hidden_size=32, attention_heads=2, head_dim=16, intermediate_size=48, experts_per_layer=2, adaptive_layers=(1,), max_calls=2, target_only=True)
        self.model = LoopMoE(self.config).eval()
        self.controller = Controller(self.config, ControllerConfig(projected_hidden=16, gru_hidden_size=16)).eval()
        batch = sample_batch("train", 2, 19, 0, (1, 2))
        self.input_ids = batch["input_ids"]
        self.labels = batch["labels"]

    def test_expected_path_count(self):
        self.assertEqual(len(all_paths(4, 3)), 84)

    def test_exact_path_probability_mass(self):
        total = torch.zeros(2)
        count = 0
        for _, _, log_probability in exact_path_batch(self.model, self.controller, self.input_ids, self.labels, 2, 0.0, torch.device("cpu"), "float32"):
            total += log_probability.exp()
            count += 1
        self.assertEqual(count, 6)
        torch.testing.assert_close(total, torch.ones_like(total), rtol=1e-5, atol=1e-5)

    def test_exact_halt_probabilities_sum_to_one(self):
        head = HaltHead(self.config.hidden_size)
        losses, probabilities, _ = halt_distribution(self.model, head, self.input_ids, self.labels, torch.device("cpu"), "float32")
        self.assertEqual(losses.shape, (2, 2))
        torch.testing.assert_close(probabilities.sum(dim=0), torch.ones(2), rtol=1e-5, atol=1e-5)

    def test_rloo_leave_one_out_formula(self):
        objectives = torch.tensor([[1.0, 3.0], [2.0, 1.0]])
        loo = (objectives.sum(dim=0, keepdim=True) - objectives) / 1
        advantages = objectives - loo
        torch.testing.assert_close(advantages, torch.tensor([[-1.0, 2.0], [1.0, -2.0]]))
        torch.testing.assert_close(advantages.mean(dim=0), torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
