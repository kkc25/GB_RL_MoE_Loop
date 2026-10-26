import hashlib
import unittest
from pathlib import Path

import torch

from config import ModelConfig
from eval.counterfactual import readout_ce
from eval.dynamic_halting import DynamicHaltingPolicy
from experiments.r1_closed_loop import (
    final_hidden_readout_diagnostics,
    position_cross_entropy,
    replay_baseline_continuation,
    source_identity,
    tensor_sha256,
)
from model import LoopMoE
from train.value_student import ValueStudent


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


class ClosedLoopHelperTests(unittest.TestCase):
    def test_position_ce_preserves_explicit_batch_dimensions(self):
        logits = torch.randn(3, 32)
        targets = torch.tensor([1, 2, 3])
        expected = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        for index in range(3):
            actual = position_cross_entropy(logits[index : index + 1], targets[index : index + 1])
            self.assertEqual(actual.shape, (1,))
            self.assertAlmostEqual(float(actual[0]), float(expected[index]), places=6)

    def test_position_ce_rejects_ambiguous_shapes(self):
        with self.assertRaises(ValueError):
            position_cross_entropy(torch.randn(32), torch.tensor([1]))
        with self.assertRaises(ValueError):
            position_cross_entropy(torch.randn(2, 32), torch.tensor([1]))

    def test_tensor_hash_is_content_sensitive(self):
        self.assertEqual(tensor_sha256(torch.ones(2)), tensor_sha256(torch.ones(2)))
        self.assertNotEqual(tensor_sha256(torch.ones(2)), tensor_sha256(torch.zeros(2)))


class FinalHiddenReadoutTests(unittest.TestCase):
    """Regression: the auxiliary FP32 readout must consume the final hidden
    (trace["hidden"]), never the pre-FFN h (trace["h"])."""

    def setUp(self):
        torch.manual_seed(3)
        self.config = tiny_config()
        self.model = LoopMoE(self.config).eval()
        self.input_ids = torch.randint(0, 64, (1, 7))
        self.segment_ids = torch.zeros_like(self.input_ids)
        self.labels = torch.full((1, 7), -100)
        self.labels[0, 1:] = self.input_ids[0, 1:]

    def fixed_depth_ce(self, depth: int, position: int) -> float:
        with torch.no_grad():
            output = self.model(self.input_ids, depth=depth, segment_ids=self.segment_ids)
        logits = output.logits[0, position].float().unsqueeze(0)
        target = self.labels[0, position].reshape(1)
        return float(torch.nn.functional.cross_entropy(logits, target))

    def check_depth(self, tau: float, depth: int) -> None:
        trace = {}
        with torch.no_grad():
            DynamicHaltingPolicy(zero_students(), tau).forward(self.model, self.input_ids, self.segment_ids, trace=trace)
        position = 3
        target = self.labels[0, position].reshape(1)
        truth = self.fixed_depth_ce(depth, position)
        diagnostics = final_hidden_readout_diagnostics(
            self.model, trace["hidden"][0, position], trace["hidden"][0, position], target, "float32"
        )
        self.assertAlmostEqual(diagnostics["production_final_hidden_fp32_head_ce"], truth, places=5)
        # The pre-FFN h would give a materially different CE: feeding it to
        # this diagnostic is the bug this test guards against.
        misused = float(readout_ce(self.model, trace["h"][0, position].float().unsqueeze(0), target))
        self.assertGreater(abs(misused - truth), 1e-4)

    def test_fp32_readout_matches_depth1_forward_only_with_final_hidden(self):
        self.check_depth(float("inf"), 1)

    def test_fp32_readout_matches_depth4_forward_only_with_final_hidden(self):
        self.check_depth(float("-inf"), 4)

    def test_replay_baseline_continuation_reproduces_production_hidden_and_actions(self):
        trace = {}
        with torch.no_grad():
            _, rounds = DynamicHaltingPolicy(zero_students(), float("-inf")).forward(
                self.model, self.input_ids, self.segment_ids, trace=trace
            )
        position = 4
        replay_hidden, actions = replay_baseline_continuation(
            self.model, trace["h"][0, position], int(trace["rounds"][0, position]), "float32"
        )
        torch.testing.assert_close(replay_hidden, trace["hidden"][0, position], rtol=1e-5, atol=1e-6)
        self.assertEqual(len(actions), int(rounds[0, position]))
        self.assertEqual(actions[0], int(trace["baseline_action"][0, position]))
        with self.assertRaises(ValueError):
            replay_baseline_continuation(self.model, trace["h"][0, position], 0, "float32")
        with self.assertRaises(ValueError):
            replay_baseline_continuation(self.model, trace["h"][0], 1, "float32")

    def test_fp32_readout_ignores_outer_autocast_on_same_final_hidden(self):
        trace = {}
        with torch.no_grad():
            DynamicHaltingPolicy(zero_students(), float("inf")).forward(
                self.model, self.input_ids, self.segment_ids, trace=trace
            )
        final_hidden = trace["hidden"][0, 2].float().unsqueeze(0)
        target = self.labels[0, 2].reshape(1)
        reference = readout_ce(self.model, final_hidden, target)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            under_autocast = readout_ce(self.model, final_hidden, target)
        torch.testing.assert_close(under_autocast, reference, rtol=0, atol=0)


class SourceIdentityTests(unittest.TestCase):
    def test_source_identity_binds_entrypoint_and_untracked_python(self):
        identity = source_identity()
        root = Path(__file__).resolve().parents[1]
        entrypoint = root / "experiments" / "r1_closed_loop.py"
        self.assertEqual(identity["entrypoint"], "experiments/r1_closed_loop.py")
        self.assertEqual(
            identity["entrypoint_sha256"],
            hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
        )
        probe = root / "untracked_identity_probe_r1.py"
        probe.write_text("# identity probe\n")
        try:
            updated = source_identity()
            self.assertEqual(updated["commit"], identity["commit"])
            self.assertEqual(updated["uncommitted_diff_sha256"], identity["uncommitted_diff_sha256"])
            self.assertEqual(
                updated["untracked_python_sha256"].get("untracked_identity_probe_r1.py"),
                hashlib.sha256(b"# identity probe\n").hexdigest(),
            )
        finally:
            probe.unlink()


if __name__ == "__main__":
    unittest.main()
