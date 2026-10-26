import json
import tempfile
import unittest
from pathlib import Path

import torch

from data.fineweb import SplitWriter
from data.packed_lm import legacy_data_fingerprint, packed_data_fingerprint, verify_packed_split, verify_packed_splits
from data.recovery import build_datasets, build_manifest, recovery_collate
from model import LoopMoE
from run_lm import mark_unverified_data_migrations
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


class PackedIntegrityTests(unittest.TestCase):
    """verify_packed_split must accept intact splits and reject any drift from
    the manifest digests, including byte edits that keep file length unchanged."""

    DOCS = {
        "train": [(0, [10, 11, 12]), (1, [20, 21])],
        "validation": [(2, [30, 31, 32, 33])],
    }

    def build_root(self, root: Path) -> dict:
        manifest = {"pipeline_version": "unit-test", "splits": {}}
        for split, docs in self.DOCS.items():
            writer = SplitWriter(root, split, 4)
            for rank, tokens in docs:
                writer.add_document(rank, tokens)
            manifest["splits"][split] = writer.finish()
        (root / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_intact_splits_verify_and_return_actual_digests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.build_root(root)
            for split in ("train", "validation"):
                actual = verify_packed_split(root, split)
                declared = manifest["splits"][split]
                self.assertEqual(actual["input_ids_sha256"], declared["input_ids_sha256"])
                self.assertEqual(actual["segment_ids_sha256"], declared["segment_ids_sha256"])
                self.assertEqual(actual["doc_map_sha256"], declared["doc_map_sha256"])
            both = verify_packed_splits(root)
            self.assertEqual(set(both), {"train", "validation"})

    def tamper(self, path: Path, offset: int = 0):
        data = bytearray(path.read_bytes())
        data[offset] ^= 0x01  # equal-length byte edit
        path.write_bytes(bytes(data))

    def test_equal_length_input_ids_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_root(root)
            self.tamper(root / "train.input_ids.u16.bin")
            with self.assertRaisesRegex(ValueError, "input_ids_sha256"):
                verify_packed_split(root, "train")

    def test_equal_length_segment_ids_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_root(root)
            self.tamper(root / "validation.segment_ids.i32.bin")
            with self.assertRaisesRegex(ValueError, "segment_ids_sha256"):
                verify_packed_split(root, "validation")

    def test_equal_length_doc_map_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_root(root)
            path = root / "train.doc_map.jsonl"
            text = path.read_text()
            self.assertIn('"length": 3', text)
            path.write_text(text.replace('"length": 3', '"length": 2', 1))
            with self.assertRaisesRegex(ValueError, "doc_map_sha256"):
                verify_packed_split(root, "train")

    def test_missing_declaration_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.build_root(root)
            del manifest["splits"]["train"]["doc_map_sha256"]
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "doc_map_sha256"):
                verify_packed_split(root, "train")
            with self.assertRaisesRegex(ValueError, "unknown"):
                verify_packed_split(root, "unknown")


class DataMigrationMarkingTests(unittest.TestCase):
    def test_v1_migration_marked_as_unverified_compatibility_stage(self):
        ledger = {"data_migrations": [{"from": "document_pool_v1", "to": "packed_stream_v2", "at_update": 4}]}
        marked = mark_unverified_data_migrations(ledger)
        entry = marked["data_migrations"][0]
        self.assertEqual(entry["from"], "document_pool_v1")
        self.assertEqual(entry["compatibility"], "unverified")
        self.assertEqual(entry["equivalence"], "not_verified")
        self.assertIn("not strict equivalence", entry["note"])

    def test_ledger_without_migrations_untouched(self):
        self.assertEqual(mark_unverified_data_migrations({"data_migrations": []}), {"data_migrations": []})
        self.assertEqual(mark_unverified_data_migrations({}), {})


if __name__ == "__main__":
    unittest.main()
