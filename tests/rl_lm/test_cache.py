import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.fineweb import SplitWriter
from data.packed_lm import PackedLMDataset
from lm_config import LMExperimentConfig, LMTrainConfig
from rl_lm.cache import build_cache, cache_identity, load_cache, plan_contexts
from tests.rl_lm.tiny_lm import build_tiny_model


class CacheTests(unittest.TestCase):
    def make_dataset(self, root):
        writer = SplitWriter(root, "train", 8)
        writer.add_document(0, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        writer.add_document(1, [15, 16, 17, 18])
        writer.add_document(2, [19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
        writer.add_document(3, [29, 30, 31, 32])
        info = writer.finish()
        (root / "manifest.json").write_text(json.dumps({"splits": {"train": info}}))
        return PackedLMDataset(root, "train", offset=1)

    def test_plan_uses_global_index_and_no_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self.make_dataset(Path(temporary))
            contexts = plan_contexts(dataset, 4, seed=3, corpus_id="toy")
            self.assertEqual(len(contexts), 4)
            self.assertEqual(len({(row["sequence_index"], row["position"]) for row in contexts}), 4)
            self.assertTrue(all(row["sequence_index"] >= 1 for row in contexts))
            self.assertTrue(all(row["episode_id"].startswith("toy:") for row in contexts))

    def test_build_and_load_cache_with_offset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.make_dataset(root)
            contexts = plan_contexts(dataset, 3, seed=4, corpus_id="toy")
            model = build_tiny_model()
            checkpoint = root / "model.pt"
            torch.save({"model": model.state_dict()}, checkpoint)
            config_path = root / "config.json"
            LMExperimentConfig(model=model.config, train=LMTrainConfig(data_dir=str(root))).save(config_path)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("{}")
            output = root / "cache"
            identity = cache_identity(model, model.config, checkpoint_path=checkpoint, config_path=config_path, tokenizer_path=tokenizer, data_dir=root, split="train", sampling_seed=4, backend_version="toy-v1", head_contract="bf16_linear_fp32_ce_v1", expert_chunk=64)
            build_cache(model, dataset, contexts, output_dir=output, identity=identity, dtype="float32", sequence_batch=2)
            expected = cache_identity(model, model.config, checkpoint_path=checkpoint, config_path=config_path, tokenizer_path=tokenizer, data_dir=root, split="train", sampling_seed=4, backend_version="toy-v1", head_contract="bf16_linear_fp32_ce_v1", expert_chunk=64)
            cache = load_cache(output, expected)
            self.assertEqual(cache["h0"].shape, (3, model.config.hidden_size))
            self.assertTrue(np.array_equal(cache["sequence_indices"], [row["sequence_index"] for row in contexts]))
            self.assertEqual(len(cache["baseline"]), 3)
            self.assertEqual(cache["sampling_report"]["duplicate_positions"], 0)
            expected["backend_version"] = "different"
            with self.assertRaises(ValueError):
                load_cache(output, expected)

    def test_checkpoint_identity_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.make_dataset(root)
            contexts = plan_contexts(dataset, 2, seed=4, corpus_id="toy")
            model = build_tiny_model()
            checkpoint = root / "model.pt"
            torch.save({"model": model.state_dict()}, checkpoint)
            config_path = root / "config.json"
            LMExperimentConfig(model=model.config, train=LMTrainConfig(data_dir=str(root))).save(config_path)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("{}")
            output = root / "cache"
            identity = cache_identity(model, model.config, checkpoint_path=checkpoint, config_path=config_path, tokenizer_path=tokenizer, data_dir=root, split="train", sampling_seed=4, backend_version="toy-v1", head_contract="bf16_linear_fp32_ce_v1", expert_chunk=64)
            build_cache(model, dataset, contexts, output_dir=output, identity=identity, dtype="float32")
            with self.assertRaises(ValueError):
                load_cache(output, identity | {"checkpoint_sha256": "0" * 64})

    def test_gamma_tokenizer_packed_and_chunk_identity_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.make_dataset(root)
            contexts = plan_contexts(dataset, 2, seed=4, corpus_id="toy")
            model = build_tiny_model()
            checkpoint = root / "model.pt"
            torch.save({"model": model.state_dict()}, checkpoint)
            config_path = root / "config.json"
            LMExperimentConfig(model=model.config, train=LMTrainConfig(data_dir=str(root))).save(config_path)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("{}")
            identity = cache_identity(model, model.config, checkpoint_path=checkpoint, config_path=config_path, tokenizer_path=tokenizer, data_dir=root, split="train", sampling_seed=4, backend_version="toy-v1", head_contract="bf16_linear_fp32_ce_v1", expert_chunk=64)
            build_cache(model, dataset, contexts, output_dir=root / "cache", identity=identity, dtype="float32")
            packed_changed = dict(identity["packed_sha256"])
            packed_changed["input_ids_sha256"] = "0" * 64
            for changed in (
                identity | {"gamma": identity["gamma"] + 0.1},
                identity | {"tokenizer_sha256": "0" * 64},
                identity | {"packed_sha256": packed_changed},
                identity | {"expert_chunk": 32},
            ):
                with self.assertRaises(ValueError):
                    load_cache(root / "cache", changed)


if __name__ == "__main__":
    unittest.main()
