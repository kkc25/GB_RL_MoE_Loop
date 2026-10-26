from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from config import ModelConfig
from eval.counterfactual import cache_last_ffn_input
from eval.dynamic_halting import DynamicHaltingPolicy
from experiments.e03_halting import load_students
from model import LoopMoE
from train.value_student import ValueStudent, predict


def tiny_config() -> ModelConfig:
    return ModelConfig(layers=2, hidden_size=32, attention_heads=2, head_dim=16, vocab_size=64, intermediate_size=24, experts_per_layer=4, top_k=2, adaptive_layers=(1,), max_calls=4, target_only=False)


def zero_students() -> dict:
    students = {}
    for name in ("immediate", "q_pi0"):
        student = ValueStudent(32, 6, 8)
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.zero_()
        students[name] = {"student": student, "mean": torch.zeros(32), "std": torch.ones(32)}
    return students


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

    def test_finite_tau_matches_offline_calibration(self):
        """Students were trained and tau was calibrated on raw pre-FFN h; the
        online policy must reproduce the offline delta/mask/rounds exactly."""
        torch.manual_seed(3)
        config = tiny_config()
        model = LoopMoE(config).eval()
        students = zero_students()
        with torch.no_grad():
            immediate = students["immediate"]["student"]
            immediate.net[0].weight[0, 0] = 1
            immediate.net[2].weight[0, 0] = 1
            immediate.net[4].weight[:, 0] = 1
        input_ids = torch.randint(0, 64, (1, 7))
        segment_ids = torch.zeros_like(input_ids)
        with torch.no_grad():
            h = cache_last_ffn_input(model, input_ids, segment_ids)
        block = model.blocks[config.adaptive_layers[-1]]
        flat_h = h.reshape(-1, h.shape[-1])
        actions = block.executor.baseline_actions(block.ffn_norm(h).reshape(-1, h.shape[-1]))
        rows = torch.arange(actions.shape[0])
        stop_pred = predict(students["immediate"]["student"], flat_h, students["immediate"]["mean"], students["immediate"]["std"])
        cont_pred = predict(students["q_pi0"]["student"], flat_h, students["q_pi0"]["mean"], students["q_pi0"]["std"])
        delta = stop_pred[rows, actions] - cont_pred[rows, actions]
        wrong_delta = predict(students["immediate"]["student"], block.ffn_norm(h).reshape(-1, h.shape[-1]),
                              students["immediate"]["mean"], students["immediate"]["std"])[:, 0]
        index = int((delta - wrong_delta).abs().argmax())
        self.assertNotEqual(float(delta[index]), float(wrong_delta[index]))
        tau = float((delta[index] + wrong_delta[index]) / 2)
        continue_mask = delta > tau
        expected_rounds = 1 + (config.max_calls - 1) * continue_mask.long()
        trace = {}
        with torch.no_grad():
            logits, rounds = DynamicHaltingPolicy(students, tau).forward(model, input_ids, segment_ids, trace=trace)
        torch.testing.assert_close(rounds.reshape(-1), expected_rounds)
        torch.testing.assert_close(trace["h"], h)
        self.assertTrue(torch.equal(trace["baseline_action"].reshape(-1), actions))
        torch.testing.assert_close(trace["delta"].reshape(-1), delta)
        self.assertTrue(torch.equal(trace["continue_mask"].reshape(-1), continue_mask))
        self.assertTrue(torch.equal(trace["rounds"], rounds))
        self.assertEqual(trace["hidden"].shape, h.shape)
        with torch.no_grad():
            expected_h, _ = model.fixed_depth(h, block, expected_rounds, None, model.gammas[config.adaptive_layers[-1]])
            expected_logits = F.linear(model.final_norm(expected_h), model.embedding.weight)
        torch.testing.assert_close(logits, expected_logits, rtol=1e-5, atol=1e-6)

    def test_real_e03_students_artifact_loads_into_policy(self):
        """The E01 ``students.pt`` payload, loaded through the real E03
        ``load_students`` interface, must drive the policy unchanged."""
        torch.manual_seed(5)
        config = tiny_config()
        model = LoopMoE(config).eval()
        payload = {}
        for name in ("immediate", "q_pi0"):
            student = ValueStudent(32, 6, 8)
            payload[name] = {
                "state_dict": student.state_dict(),
                "mean": torch.zeros(1, 32),
                "std": torch.ones(1, 32).clamp_min(1e-6),
                "hidden_dim": 8,
            }
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(payload, Path(tmp) / "students.pt")
            students = load_students(Path(tmp))
        input_ids = torch.randint(0, 64, (1, 7))
        segment_ids = torch.zeros_like(input_ids)
        with torch.no_grad():
            logits, rounds = DynamicHaltingPolicy(students, 0.0).forward(model, input_ids, segment_ids)
        self.assertEqual(tuple(logits.shape), (1, 7, config.vocab_size))
        self.assertEqual(tuple(rounds.shape), (1, 7))
        self.assertTrue(bool((rounds >= 1).all()) and bool((rounds <= config.max_calls).all()))


if __name__ == "__main__":
    unittest.main()
