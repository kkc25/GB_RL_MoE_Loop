"""Regression tests for the 2026-10-25 code audit items F10/F11 covering
train/lm.py, train/recovery.py and train/recovery_common.py.

Each test encodes a failure reproduced against the legacy code; they fail on
the pre-repair tree and must pass after the repair:

* the LM loop rebuilt the batch sampler on every update (O(N^2/batch) index
  work per epoch); the sampler must be built once per epoch and the
  deterministic batch order must be unchanged, including mid-epoch resume;
* checkpoints were saved before the current segment's elapsed time was
  accumulated into the ledger, so a resume silently dropped the previous
  segment's wall time;
* best-validation state, RNG state and the LR schedule were not restored /
  recorded across resume, so a worse validation NLL could overwrite the
  historical best and a continued run silently jumped to a new schedule;
* LM resume did not verify a data hash, silently continuing on different data;
* the ledger field ``expert_calls`` actually counted per-token FFN rounds;
  the ledger now carries ``ffn_rounds`` and ``expert_token_pairs``
  (rounds x top_k) instead, and legacy checkpoints are migrated.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train.lm as lm_module
from config import ModelConfig
from data.recovery import build_datasets, build_manifest, recovery_collate
from lm_config import LMExperimentConfig, LMTrainConfig
from model import LoopMoE
from recovery_config import RecoveryConfig, RecoveryRunConfig, RecoveryTaskConfig
from train.common import MetricWriter
from train.lm import train_lm
from train.recovery import train_recovery
from train.recovery_common import RecoveryLedger


class FakePackedDataset:
    """In-memory stand-in for PackedLMDataset (12 sequences of length 8)."""

    def __init__(self, n_sequences=12, seq_len=8, vocab_size=64):
        rng = np.random.RandomState(0)
        self.input_ids = rng.randint(1, vocab_size, size=(n_sequences, seq_len)).astype(np.int64)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        ids = self.input_ids[index]
        labels = np.concatenate([ids[1:], np.array([-100], dtype=np.int64)])
        return {
            "index": index,
            "input_ids": ids,
            "labels": labels,
            "segment_ids": np.zeros(self.seq_len, dtype=np.int32),
            "n_tokens": self.seq_len,
            "n_supervised_tokens": self.seq_len - 1,
        }


class FakeClock:
    """Deterministic clock that advances 1s per model forward call."""

    def __init__(self):
        self.now = 0.0
        self.forwards = 0

    def monotonic(self):
        return self.now

    def hook(self, model):
        original = model.forward

        def forward(*args, **kwargs):
            self.forwards += 1
            self.now += 1.0
            return original(*args, **kwargs)

        model.forward = forward
        return model


def lm_config(output_dir, updates, validation_interval=2, top_k=1, seed=17, learning_rate=1e-3):
    model = ModelConfig(
        layers=2, hidden_size=32, attention_heads=2, head_dim=16, vocab_size=64,
        intermediate_size=48, experts_per_layer=2, top_k=top_k,
        adaptive_layers=(1,), max_calls=2, target_only=False,
    )
    train = LMTrainConfig(
        output_dir=str(output_dir), device="cpu", dtype="float32", seed=seed,
        batch_size=2, updates=updates, learning_rate=learning_rate,
        weight_decay=0.0, warmup_fraction=0.0, gradient_clip=1.0,
        validation_interval=validation_interval, validation_batches=1,
        log_interval=1, depth_schedule="fixed", fixed_depth=1,
    )
    return LMExperimentConfig(model=model, train=train)


class LMSamplerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output_dir = Path(self.tmp.name) / "lm"
        self.dataset = FakePackedDataset()

    def reference_order(self, updates, batch_size=2, seed=17):
        """Batch order produced by the legacy per-update sampler construction."""
        sampler_type = lm_module.PackedLMBatchSampler
        batches_per_epoch = len(self.dataset) // batch_size
        expected = []
        for update in range(updates):
            epoch, start_batch = divmod(update, batches_per_epoch)
            sampler = sampler_type(
                len(self.dataset), batch_size, shuffle=True, seed=seed,
                epoch=epoch, start_batch=start_batch,
            )
            expected.append(next(iter(sampler)))
        return expected

    def train_recording(self, config, consumed, constructions):
        real_sampler = lm_module.PackedLMBatchSampler
        real_packed_batch = lm_module.packed_batch

        class CountingSampler(real_sampler):
            def __init__(self, *args, **kwargs):
                if kwargs.get("shuffle", True):
                    constructions.append((kwargs.get("epoch", 0), kwargs.get("start_batch", 0)))
                super().__init__(*args, **kwargs)

        torch.manual_seed(config.train.seed)
        model = LoopMoE(config.model)

        def recording_packed_batch(dataset, indices, device):
            if model.training:
                consumed.append(list(indices))
            return real_packed_batch(dataset, indices, device)

        writer = MetricWriter(Path(config.train.output_dir) / "metrics.jsonl")
        with mock.patch.object(lm_module, "PackedLMBatchSampler", CountingSampler), \
                mock.patch.object(lm_module, "packed_batch", recording_packed_batch):
            return train_lm(model, self.dataset, self.dataset, config, torch.device("cpu"), writer)

    def test_sampler_built_once_per_epoch_with_unchanged_order(self):
        config = lm_config(self.output_dir, updates=14)
        consumed, constructions = [], []
        self.train_recording(config, consumed, constructions)
        self.assertEqual(consumed, self.reference_order(14))
        # 6 batches/epoch over 14 updates touches epochs 0, 1, 2 exactly once.
        self.assertEqual(constructions, [(0, 0), (1, 0), (2, 0)])

    def test_mid_epoch_resume_preserves_order_and_reuses_sampler(self):
        consumed1, constructions1 = [], []
        self.train_recording(lm_config(self.output_dir, updates=8), consumed1, constructions1)
        config2 = lm_config(self.output_dir, updates=14)
        config2.train.resume = str(self.output_dir / "latest.pt")
        consumed2, constructions2 = [], []
        self.train_recording(config2, consumed2, constructions2)
        self.assertEqual(consumed1 + consumed2, self.reference_order(14))
        # Resume re-enters epoch 1 at batch 2, then builds epoch 2 once.
        self.assertEqual(constructions1, [(0, 0), (1, 0)])
        self.assertEqual(constructions2, [(1, 2), (2, 0)])


class LMResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output_dir = Path(self.tmp.name) / "lm"
        self.dataset = FakePackedDataset()

    def train(self, config, clock=None, data_hash=None):
        torch.manual_seed(config.train.seed)
        model = LoopMoE(config.model)
        if clock is not None:
            clock.hook(model)
        writer = MetricWriter(Path(config.train.output_dir) / "metrics.jsonl")
        return train_lm(model, self.dataset, self.dataset, config, torch.device("cpu"), writer, data_hash=data_hash)

    def test_elapsed_accumulated_before_checkpoint_and_across_resume(self):
        clock = FakeClock()
        with mock.patch("time.monotonic", clock.monotonic):
            self.train(lm_config(self.output_dir, updates=2), clock=clock)
        run1_forwards = clock.forwards
        self.assertEqual(run1_forwards, 3)  # 2 updates + 1 validation batch
        checkpoint = torch.load(self.output_dir / "latest.pt", weights_only=False)
        self.assertEqual(
            checkpoint["ledger"]["elapsed_seconds"], float(run1_forwards),
            "checkpoint saved before the current segment's elapsed time was accumulated",
        )
        config2 = lm_config(self.output_dir, updates=4)
        config2.train.resume = str(self.output_dir / "latest.pt")
        with mock.patch("time.monotonic", clock.monotonic):
            ledger = self.train(config2, clock=clock)
        self.assertEqual(ledger["elapsed_seconds"], float(clock.forwards))
        saved = json.loads((self.output_dir / "cost_ledger.json").read_text())
        self.assertEqual(saved["elapsed_seconds"], float(clock.forwards))

    def test_resume_restores_torch_rng(self):
        self.train(lm_config(self.output_dir, updates=2))
        checkpoint = torch.load(self.output_dir / "latest.pt", weights_only=False)
        expected_rng = checkpoint["torch_rng"]
        torch.manual_seed(999)
        torch.rand(64)
        config2 = lm_config(self.output_dir, updates=2)  # no remaining updates
        config2.train.resume = str(self.output_dir / "latest.pt")
        self.train(config2)
        self.assertTrue(
            torch.equal(torch.get_rng_state(), expected_rng),
            "resume must restore the torch RNG state stored in the checkpoint",
        )

    def test_historical_best_survives_resume(self):
        self.train(lm_config(self.output_dir, updates=2))
        checkpoint = torch.load(self.output_dir / "latest.pt", weights_only=False)
        self.assertIn("best_nll", checkpoint, "checkpoint must carry the historical best NLL")
        checkpoint["best_nll"] = -1.0  # unreachable: no future validation may beat it
        torch.save(checkpoint, self.output_dir / "latest.pt")
        best_before = torch.load(self.output_dir / "best_validation.pt", weights_only=False)["update"]
        config2 = lm_config(self.output_dir, updates=4)
        config2.train.resume = str(self.output_dir / "latest.pt")
        self.train(config2)
        best_after = torch.load(self.output_dir / "best_validation.pt", weights_only=False)["update"]
        self.assertEqual(best_after, best_before, "a worse validation NLL overwrote the historical best")

    def test_resume_rejects_data_hash_mismatch(self):
        self.train(lm_config(self.output_dir, updates=2), data_hash="a" * 64)
        checkpoint = torch.load(self.output_dir / "latest.pt", weights_only=False)
        self.assertEqual(checkpoint.get("data_hash"), "a" * 64)
        config2 = lm_config(self.output_dir, updates=4)
        config2.train.resume = str(self.output_dir / "latest.pt")
        with self.assertRaises(ValueError):
            self.train(config2, data_hash="b" * 64)
        ledger = self.train(config2, data_hash="a" * 64)
        self.assertEqual(ledger["updates"], 4)

    def test_schedule_change_is_recorded_as_explicit_phase(self):
        self.train(lm_config(self.output_dir, updates=2, learning_rate=1e-3))
        config2 = lm_config(self.output_dir, updates=4, learning_rate=2e-3)
        config2.train.resume = str(self.output_dir / "latest.pt")
        ledger = self.train(config2)
        self.assertEqual(len(ledger["schedule_phases"]), 1)
        phase = ledger["schedule_phases"][0]
        self.assertEqual(phase["schedule"]["updates"], 2)
        self.assertEqual(phase["schedule"]["learning_rate"], 1e-3)
        self.assertEqual(phase["ended_at_update"], 2)
        records = [json.loads(line) for line in (self.output_dir / "metrics.jsonl").read_text().splitlines()]
        self.assertTrue(any(record.get("phase") == "schedule" for record in records))

    def test_same_schedule_resume_adds_no_phase(self):
        self.train(lm_config(self.output_dir, updates=2))
        config2 = lm_config(self.output_dir, updates=2)
        config2.train.resume = str(self.output_dir / "latest.pt")
        ledger = self.train(config2)
        self.assertEqual(ledger["schedule_phases"], [])


class LMLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output_dir = Path(self.tmp.name) / "lm"
        self.dataset = FakePackedDataset()

    def train(self, config, data_hash=None):
        torch.manual_seed(config.train.seed)
        model = LoopMoE(config.model)
        writer = MetricWriter(Path(config.train.output_dir) / "metrics.jsonl")
        return train_lm(model, self.dataset, self.dataset, config, torch.device("cpu"), writer, data_hash=data_hash)

    def test_ledger_separates_ffn_rounds_and_expert_token_pairs(self):
        # 2 updates x 2 sequences x 8 tokens x (1 non-adaptive + depth 1) = 64 rounds.
        ledger = self.train(lm_config(self.output_dir, updates=2, top_k=2))
        self.assertEqual(ledger["ffn_rounds"], 64)
        self.assertEqual(ledger["expert_token_pairs"], 128)
        self.assertNotIn("expert_calls", ledger)

    def test_legacy_checkpoint_migrates_expert_calls(self):
        self.train(lm_config(self.output_dir, updates=2, top_k=2))
        checkpoint = torch.load(self.output_dir / "latest.pt", weights_only=False)
        ledger = checkpoint["ledger"]
        checkpoint["ledger"] = {
            "updates": ledger["updates"],
            "input_tokens": ledger["input_tokens"],
            "supervised_tokens": ledger["supervised_tokens"],
            "expert_calls": ledger["ffn_rounds"],  # legacy field actually held rounds
            "elapsed_seconds": ledger["elapsed_seconds"],
            "allocated_gpu_hours": 0.0,
            "peak_memory_bytes": 0,
        }
        for key in ("best_nll", "schedule", "data_hash"):
            checkpoint.pop(key, None)
        torch.save(checkpoint, self.output_dir / "latest.pt")
        config2 = lm_config(self.output_dir, updates=4, top_k=2)
        config2.train.resume = str(self.output_dir / "latest.pt")
        migrated = self.train(config2)
        self.assertEqual(migrated["ffn_rounds"], 64 + 64)
        self.assertEqual(migrated["expert_token_pairs"], 256)


class RecoveryResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output_dir = Path(self.tmp.name) / "recovery"
        self.datasets = build_datasets("c0_identity")
        self.data_hash = build_manifest("c0_identity")["sha256"]

    def config(self, updates, validation_interval=2, warmup_updates=1, learning_rate=1e-3, top_k=2):
        model = ModelConfig(
            layers=2, hidden_size=32, attention_heads=2, head_dim=16, vocab_size=128,
            intermediate_size=48, experts_per_layer=2, top_k=top_k,
            adaptive_layers=(1,), max_calls=2, target_only=True,
        )
        run = RecoveryRunConfig(
            output_dir=str(self.output_dir), device="cpu", dtype="float32", seed=11,
            batch_size=16, updates=updates, learning_rate=learning_rate,
            warmup_updates=warmup_updates, validation_interval=validation_interval,
            log_interval=1, fixed_depth=1, depth_schedule="fixed",
        )
        return RecoveryConfig(model=model, task=RecoveryTaskConfig(stage="c0_identity"), run=run)

    def train(self, config, clock=None, data_hash=None):
        torch.manual_seed(config.run.seed)
        model = LoopMoE(config.model)
        if clock is not None:
            clock.hook(model)
        ledger = RecoveryLedger()
        writer = MetricWriter(Path(config.run.output_dir) / "metrics.jsonl")
        train_recovery(
            model, self.datasets, recovery_collate, config, torch.device("cpu"),
            ledger, writer, self.data_hash if data_hash is None else data_hash,
        )
        return ledger

    def test_data_hash_mismatch_rejected(self):
        self.train(self.config(updates=2))
        config2 = self.config(updates=4)
        config2.run.resume = str(self.output_dir / "latest.pt")
        with self.assertRaises(ValueError):
            self.train(config2, data_hash="0" * 64)
        ledger = self.train(config2)
        self.assertEqual(ledger.optimizer_updates, 4)

    def test_elapsed_accumulated_before_checkpoint_and_across_resume(self):
        clock = FakeClock()
        with mock.patch("time.monotonic", clock.monotonic):
            self.train(self.config(updates=2), clock=clock)
        run1_forwards = clock.forwards  # 2 updates + 7 validation batches
        checkpoint = torch.load(self.output_dir / "latest.pt", weights_only=False)
        self.assertEqual(
            checkpoint["ledger"]["elapsed_seconds"], float(run1_forwards),
            "checkpoint saved before the current segment's elapsed time was accumulated",
        )
        config2 = self.config(updates=4)
        config2.run.resume = str(self.output_dir / "latest.pt")
        with mock.patch("time.monotonic", clock.monotonic):
            ledger = self.train(config2, clock=clock)
        self.assertEqual(ledger.elapsed_seconds, float(clock.forwards))

    def test_resume_restores_torch_rng(self):
        self.train(self.config(updates=2))
        checkpoint = torch.load(self.output_dir / "latest.pt", weights_only=False)
        expected_rng = checkpoint["torch_rng"]
        torch.manual_seed(999)
        torch.rand(64)
        config2 = self.config(updates=2)  # no remaining updates
        config2.run.resume = str(self.output_dir / "latest.pt")
        self.train(config2)
        self.assertTrue(
            torch.equal(torch.get_rng_state(), expected_rng),
            "resume must restore the torch RNG state stored in the checkpoint",
        )

    def test_historical_best_survives_resume(self):
        self.train(self.config(updates=2))
        checkpoint = torch.load(self.output_dir / "latest.pt", weights_only=False)
        self.assertIn("best_validation_loss", checkpoint, "checkpoint must carry the historical best loss")
        checkpoint["best_validation_loss"] = -1.0  # unreachable
        torch.save(checkpoint, self.output_dir / "latest.pt")
        best_before = torch.load(self.output_dir / "best_validation.pt", weights_only=False)["update"]
        config2 = self.config(updates=4)
        config2.run.resume = str(self.output_dir / "latest.pt")
        self.train(config2)
        best_after = torch.load(self.output_dir / "best_validation.pt", weights_only=False)["update"]
        self.assertEqual(best_after, best_before, "a worse validation loss overwrote the historical best")

    def test_schedule_change_is_recorded_as_explicit_phase(self):
        self.train(self.config(updates=2, warmup_updates=1))
        config2 = self.config(updates=4, warmup_updates=3)
        config2.run.resume = str(self.output_dir / "latest.pt")
        ledger = self.train(config2)
        self.assertEqual(len(ledger.schedule_phases), 1)
        phase = ledger.schedule_phases[0]
        self.assertEqual(phase["schedule"]["updates"], 2)
        self.assertEqual(phase["schedule"]["warmup_updates"], 1)
        self.assertEqual(phase["ended_at_update"], 2)
        records = [json.loads(line) for line in (self.output_dir / "metrics.jsonl").read_text().splitlines()]
        self.assertTrue(any(record.get("phase") == "schedule" for record in records))

    def test_ledger_separates_ffn_rounds_and_expert_token_pairs(self):
        ledger = self.train(self.config(updates=2, top_k=2))
        self.assertGreater(ledger.ffn_rounds, 0)
        self.assertEqual(ledger.expert_token_pairs, 2 * ledger.ffn_rounds)
        self.assertFalse(hasattr(ledger, "expert_calls"))

    def test_legacy_ledger_migrates_expert_calls(self):
        self.train(self.config(updates=2))
        checkpoint = torch.load(self.output_dir / "latest.pt", weights_only=False)
        legacy = {
            "optimizer_updates": 2,
            "supervised_targets": 32,
            "input_tokens": 1000,
            "rollouts": 0,
            "expert_calls": 500,  # legacy field actually held rounds
            "elapsed_seconds": 1.0,
            "allocated_gpu_hours": 0.0,
            "lost_compute_seconds": 0.0,
            "peak_memory_bytes": 0,
        }
        checkpoint["ledger"] = legacy
        for key in ("best_validation_loss", "schedule"):
            checkpoint.pop(key, None)
        torch.save(checkpoint, self.output_dir / "latest.pt")
        config2 = self.config(updates=4)
        config2.run.resume = str(self.output_dir / "latest.pt")
        ledger = self.train(config2)
        self.assertGreater(ledger.ffn_rounds, 500)
        self.assertEqual(ledger.expert_token_pairs, 2 * ledger.ffn_rounds)
        self.assertEqual(ledger.elapsed_seconds >= 1.0, True)


if __name__ == "__main__":
    unittest.main()
