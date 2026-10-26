from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from config import ModelConfig
from eval import profile_dynamic
from eval.dynamic_halting import DynamicHaltingPolicy
from model import LoopMoE
from train.lm import lm_loss
from train.value_student import ValueStudent


def tiny_model() -> LoopMoE:
    torch.manual_seed(3)
    config = ModelConfig(layers=2, hidden_size=32, attention_heads=2, head_dim=16, vocab_size=64, intermediate_size=24, experts_per_layer=4, top_k=2, adaptive_layers=(1,), max_calls=4, target_only=False)
    return LoopMoE(config).eval()


def zero_students() -> dict:
    students = {}
    for name in ("immediate", "q_pi0"):
        student = ValueStudent(32, 6, 8)
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.zero_()
        students[name] = {"student": student, "mean": torch.zeros(32), "std": torch.ones(32)}
    return students


def tiny_train_config() -> SimpleNamespace:
    return SimpleNamespace(train=SimpleNamespace(batch_size=2, dtype="float32"))


class FakeDataset:
    """Minimal PackedLMDataset stand-in: sequence dicts of numpy arrays."""

    def __init__(self, sequences: list[dict]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict:
        return self.sequences[index]


def fake_sequences() -> list[dict]:
    rng = np.random.default_rng(0)
    sequences = []
    for labels in ([10, 11, -100, 12, 13], [-100, 20, 21, -100, 22]):
        sequences.append({
            "input_ids": rng.integers(0, 64, size=5).astype(np.int64),
            "labels": np.array(labels, dtype=np.int64),
            "segment_ids": np.zeros(5, dtype=np.int64),
            "n_tokens": 5,
            "n_supervised_tokens": int(sum(label != -100 for label in labels)),
        })
    return sequences


class ProfileDynamicTests(unittest.TestCase):
    def test_profile_times_input_to_logits_only(self):
        """The timed region must need only input_ids (+segment_ids): fixed and
        dynamic modes are both input->logits, with no labels/CE inside."""
        model = tiny_model()
        policy = DynamicHaltingPolicy(zero_students(), float("inf"))
        config = tiny_train_config()
        device = torch.device("cpu")
        batch = {
            "input_ids": torch.randint(0, 64, (1, 7)),
            "segment_ids": torch.zeros(1, 7, dtype=torch.long),
        }
        for mode in (1, 4, "dynamic"):
            result = profile_dynamic.profile(model, policy, batch, config, device, mode, warmup=1, steps=2)
            self.assertEqual(result["steps"], 2)
            self.assertGreaterEqual(result["step_ms_mean"], 0.0)
            self.assertIn("peak_memory_bytes", result)

    def test_profile_resets_peak_memory_per_invocation(self):
        """Each mode's profile must reset peak-memory stats so peaks are
        per-mode measurements, not a cumulative maximum across modes."""
        model = tiny_model()
        policy = DynamicHaltingPolicy(zero_students(), float("inf"))
        config = tiny_train_config()
        device = torch.device("cpu")
        batch = {
            "input_ids": torch.randint(0, 64, (1, 7)),
            "segment_ids": torch.zeros(1, 7, dtype=torch.long),
        }
        with mock.patch.object(profile_dynamic, "reset_peak_memory") as reset:
            for mode in (1, 4, "dynamic"):
                profile_dynamic.profile(model, policy, batch, config, device, mode, warmup=0, steps=1)
        self.assertEqual(reset.call_count, 3)

    def test_evaluate_dynamic_separates_supervised_and_executed_cost(self):
        """Mean rounds over supervised positions must be reported separately
        from the all-position executed FFN rounds / expert-token pairs."""
        model = tiny_model()
        policy = DynamicHaltingPolicy(zero_students(), float("inf"))
        config = tiny_train_config()
        device = torch.device("cpu")
        dataset = FakeDataset(fake_sequences())
        result = profile_dynamic.evaluate_dynamic(model, policy, dataset, config, device, batches=1)
        supervised = 7
        executed = 10
        self.assertEqual(result["supervised_tokens"], supervised)
        self.assertEqual(result["executed_positions"], executed)
        self.assertEqual(result["mean_rounds_supervised"], 1.0)
        self.assertEqual(result["ffn_rounds_total"], executed)
        self.assertEqual(result["expert_token_pairs"], executed * model.config.top_k)
        input_ids = torch.from_numpy(np.stack([sequence["input_ids"] for sequence in dataset.sequences]))
        labels = torch.from_numpy(np.stack([sequence["labels"] for sequence in dataset.sequences]))
        segments = torch.from_numpy(np.stack([sequence["segment_ids"] for sequence in dataset.sequences]))
        with torch.no_grad():
            reference = model(input_ids, depth=1, segment_ids=segments).logits
        _, loss_sum, count = lm_loss(reference, labels)
        self.assertAlmostEqual(result["nll"], (loss_sum / count).item(), places=6)


if __name__ == "__main__":
    unittest.main()
