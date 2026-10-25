from __future__ import annotations

import unittest
import torch

from config import ModelConfig
from eval.dynamic_halting import DynamicHaltingPolicy
from model import LoopMoE
from train.value_student import ValueStudent


class DynamicHaltingTests(unittest.TestCase):
    def test_threshold_extremes_match_fixed_depths(self):
        torch.manual_seed(3)
        config = ModelConfig(layers=2, hidden_size=32, attention_heads=2, head_dim=16, vocab_size=64, intermediate_size=24, experts_per_layer=4, top_k=2, adaptive_layers=(1,), max_calls=4, target_only=False)
        model = LoopMoE(config).eval()
        students = {}
        for name in ("immediate", "q_pi0"):
            student = ValueStudent(32, 6, 8)
            with torch.no_grad():
                for parameter in student.parameters():
                    parameter.zero_()
            students[name] = {"student": student, "mean": torch.zeros(32), "std": torch.ones(32)}
        input_ids = torch.randint(0, 64, (1, 7))
        segment_ids = torch.zeros_like(input_ids)
        with torch.no_grad():
            stop_logits, stop_rounds = DynamicHaltingPolicy(students, float("inf")).forward(model, input_ids, segment_ids)
            continue_logits, continue_rounds = DynamicHaltingPolicy(students, float("-inf")).forward(model, input_ids, segment_ids)
            fixed_one = model(input_ids, depth=1).logits
            fixed_four = model(input_ids, depth=4).logits
        torch.testing.assert_close(stop_logits, fixed_one)
        torch.testing.assert_close(continue_logits, fixed_four)
        self.assertTrue(bool((stop_rounds == 1).all()))
        self.assertTrue(bool((continue_rounds == 4).all()))


if __name__ == "__main__":
    unittest.main()
