import tempfile
import unittest
from pathlib import Path

import torch

from data.packed_lm import legacy_data_fingerprint, packed_data_fingerprint
from data.recovery import build_datasets, build_manifest, recovery_collate
from model import LoopMoE
from tests.test_lm_recovery import FakePackedDataset, lm_config
from train.common import MetricWriter
from train.lm import train_lm
from train.recovery import train_recovery
from train.recovery_common import RecoveryLedger
from recovery_config import RecoveryConfig, RecoveryRunConfig, RecoveryTaskConfig
from config import ModelConfig


def manifest(train_hash):
    return {
        "document_list_sha256": "a" * 64,
        "tokenizer": {"sha256": "b" * 64},
        "splits": {
            split: {
                "input_ids_sha256": train_hash if split == "train" else "c" * 64,
                "segment_ids_sha256": "d" * 64,
                "doc_map_sha256": "e" * 64,
                "seq_len": 1024,
            }
            for split in ("train", "validation")
        },
    }


class PackedFingerprintTests(unittest.TestCase):
    def test_packed_hash_distinguishes_truncated_streams_with_same_document_pool(self):
        small = manifest("1" * 64)
        large = manifest("2" * 64)
        self.assertEqual(legacy_data_fingerprint(small), legacy_data_fingerprint(large))
        self.assertNotEqual(packed_data_fingerprint(small), packed_data_fingerprint(large))


class StreamResumeTests(unittest.TestCase):
    def test_lm_rejects_batch_size_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "lm"
            dataset = FakePackedDataset()
            config = lm_config(output, updates=2)
            model = LoopMoE(config.model)
            train_lm(model, dataset, dataset, config, torch.device("cpu"), MetricWriter(output / "metrics.jsonl"))
            resumed = lm_config(output, updates=4)
            resumed.train.batch_size = 4
            resumed.train.resume = str(output / "latest.pt")
            model = LoopMoE(resumed.model)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                train_lm(model, dataset, dataset, resumed, torch.device("cpu"), MetricWriter(output / "metrics.jsonl"))

    def test_lm_rejects_sequence_count_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "lm"
            dataset = FakePackedDataset()
            config = lm_config(output, updates=2)
            model = LoopMoE(config.model)
            train_lm(model, dataset, dataset, config, torch.device("cpu"), MetricWriter(output / "metrics.jsonl"))
            checkpoint = torch.load(output / "latest.pt", weights_only=False)
            checkpoint["stream_metadata"]["n_sequences"] += 1
            torch.save(checkpoint, output / "latest.pt")
            resumed = lm_config(output, updates=4)
            resumed.train.resume = str(output / "latest.pt")
            model = LoopMoE(resumed.model)
            with self.assertRaisesRegex(ValueError, "n_sequences"):
                train_lm(model, dataset, dataset, resumed, torch.device("cpu"), MetricWriter(output / "metrics.jsonl"))

    def test_recovery_rejects_batch_size_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "recovery"
            model_config = ModelConfig(
                layers=2,
                hidden_size=32,
                attention_heads=2,
                head_dim=16,
                vocab_size=128,
                intermediate_size=48,
                experts_per_layer=2,
                adaptive_layers=(1,),
                max_calls=2,
                target_only=True,
            )
            run = RecoveryRunConfig(
                output_dir=str(output),
                device="cpu",
                dtype="float32",
                batch_size=16,
                updates=2,
                validation_interval=2,
                log_interval=1,
            )
            config = RecoveryConfig(model=model_config, task=RecoveryTaskConfig(stage="c0_identity"), run=run)
            datasets = build_datasets("c0_identity")
            data_hash = build_manifest("c0_identity")["sha256"]
            model = LoopMoE(config.model)
            train_recovery(model, datasets, recovery_collate, config, torch.device("cpu"), RecoveryLedger(), MetricWriter(output / "metrics.jsonl"), data_hash)
            config.run.updates = 4
            config.run.batch_size = 8
            config.run.resume = str(output / "latest.pt")
            model = LoopMoE(config.model)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                train_recovery(model, datasets, recovery_collate, config, torch.device("cpu"), RecoveryLedger(), MetricWriter(output / "metrics.jsonl"), data_hash)


if __name__ == "__main__":
    unittest.main()
