import json
import re
import tempfile
import unittest
from pathlib import Path

import torch

from data.confirmation_corpus import prepare_confirmation_corpus
from data.controller_corpus import build_doc_records
from data.packed_lm import PackedLMDataset
from experiments.evaluate_plan_c import consistency_sample, supervised_positions, verify_frozen_inputs
from experiments.prepare_plan_c import write_controller_exclusions
from tests.test_controller_corpus import synthetic_docs, synthetic_tokenizer
from train.recovery_common import sha256_file


class ConfirmationCorpusTests(unittest.TestCase):
    def test_pipeline_excludes_prior_documents_and_packs_natural_stream(self):
        docs = synthetic_docs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = root / "prior.jsonl"
            record = build_doc_records([docs[5]])[0]
            prior.write_text(json.dumps(record) + "\n")
            output = root / "confirmation"
            manifest = prepare_confirmation_corpus(
                lambda: iter(docs),
                output,
                tokenizer=synthetic_tokenizer(),
                revision="synthetic-v1",
                source="unit-test",
                confirmation_tokens=48,
                pool_bytes=10000,
                exclude_documents=[prior],
                seq_len=16,
            )
            dataset = PackedLMDataset(output, "confirmation")
            self.assertEqual(manifest["splits"]["confirmation"]["n_tokens"], 48)
            self.assertEqual(manifest["config"]["confirmation_tokens"], 48)
            self.assertGreater(manifest["n_frozen_documents"], 0)
            frozen = json.loads((output / "frozen_document_ids.json").read_text())
            self.assertNotIn("doc-5", frozen["document_ids"])
            self.assertEqual(len(frozen["document_ranks"]), manifest["n_frozen_documents"])
            self.assertGreater(dataset.n_supervised_tokens, 0)

    def test_controller_exclusions_cover_heldout_and_sampled_train_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {"rank": 0, "role": "controller_train", "doc_hash": "a"},
                {"rank": 1, "role": "selection", "doc_hash": "b"},
                {"rank": 2, "role": "controller_train", "doc_hash": "c"},
                {"rank": 3, "role": "newtest", "doc_hash": "d"},
            ]
            (root / "documents.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            states = root / "states.pt"
            torch.save({"stats": {"doc_ids": [2]}}, states)
            output, counts = write_controller_exclusions(root, states, root / "excluded.jsonl")
            excluded = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual({row["rank"] for row in excluded}, {1, 2, 3})
            self.assertEqual(counts, {"heldout_roles": 2, "controller_train_states": 1})


class ConfirmationStatisticsTests(unittest.TestCase):
    def test_document_accumulation_preserves_natural_token_weights(self):
        import numpy as np
        from experiments.evaluate_plan_c import accumulate_document_values

        sums = np.zeros(2)
        counts = np.zeros(2)
        accumulate_document_values(sums, counts, np.array([0, 0, 1]), np.array([1.0, 2.0, 4.0]))
        self.assertTrue(np.array_equal(sums, [3.0, 4.0]))
        self.assertTrue(np.array_equal(counts, [2.0, 1.0]))
        accumulate_document_values(sums, None, np.array([0, 1]), np.array([0.5, 0.5]))
        self.assertTrue(np.array_equal(sums, [3.5, 4.5]))
        self.assertTrue(np.array_equal(counts, [2.0, 1.0]))

    def test_bootstrap_uses_document_ratio_of_sums_and_one_sided_upper(self):
        import numpy as np
        from experiments.evaluate_plan_c import document_bootstrap

        result = document_bootstrap(np.array([1.0, 3.0]), np.array([1.0, 1.0]), 1000, seed=0)
        self.assertAlmostEqual(result["estimate"], 2.0)
        self.assertEqual(result["replicates"], 1000)
        self.assertEqual(result["upper_95"], 3.0)


def build_corpus(root: Path, tokens: int = 256, tokenizer_path=None):
    output = root / "confirmation"
    manifest = prepare_confirmation_corpus(
        lambda: iter(synthetic_docs()),
        output,
        tokenizer=synthetic_tokenizer(),
        revision="synthetic-v1",
        source="unit-test",
        confirmation_tokens=tokens,
        pool_bytes=100000,
        exclude_documents=[],
        seq_len=16,
        tokenizer_path=tokenizer_path,
    )
    return output, manifest


class ConsistencySamplingTests(unittest.TestCase):
    def test_supervised_positions_drop_segment_final_tokens(self):
        segments = [{"doc": 0, "offset": 0, "length": 3}, {"doc": 1, "offset": 0, "length": 2}, {"doc": 2, "offset": 0, "length": 1}]
        self.assertEqual(supervised_positions(segments), [0, 1, 3])

    def test_sample_is_deterministic_and_spans_sequences_and_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, _ = build_corpus(Path(temporary))
            dataset = PackedLMDataset(output, "confirmation")
            first = consistency_sample(dataset, max_states=16, seed=0)
            second = consistency_sample(dataset, max_states=16, seed=0)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 16)
            sequences = {entry[0] for entry in first}
            docs = {entry[2] for entry in first}
            self.assertGreater(len(sequences), 1)
            self.assertGreater(len(docs), 1)
            # The legacy auxiliary check took the first 32 supervised states,
            # all inside sequence 0; the pre-fixed sample must not.
            self.assertNotEqual(sequences, {0})
            for sequence_index, position, doc_id in first:
                item = dataset[sequence_index]
                self.assertNotEqual(int(item["labels"][position]), -100)
                self.assertEqual(int(item["doc_ids"][position]), doc_id)

    def test_seed_changes_sample_and_fill_pass_covers_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, _ = build_corpus(Path(temporary))
            dataset = PackedLMDataset(output, "confirmation")
            n_documents = len({seg["doc"] for i in range(len(dataset)) for seg in dataset[i]["segments"]})
            self.assertLess(n_documents, 40)
            filled = consistency_sample(dataset, max_states=40, seed=0)
            self.assertEqual(len(filled), 40)  # second pass fills past one-per-document
            other = consistency_sample(dataset, max_states=40, seed=1)
            self.assertNotEqual(filled, other)


class FrozenInputVerificationTests(unittest.TestCase):
    def build_inputs(self, root: Path):
        tokenizer_path = root / "tokenizer.json"
        tokenizer_path.write_bytes(b"synthetic tokenizer bytes")
        output, _manifest = build_corpus(root, tokenizer_path=str(tokenizer_path))
        config = root / "config.json"
        config.write_text('{"model": {}, "train": {}}\n')
        checkpoint = root / "checkpoint.pt"
        checkpoint.write_bytes(b"fake checkpoint bytes")
        plan = {
            "corpus_manifest_sha256": sha256_file(output / "manifest.json"),
            "checkpoint_sha256": sha256_file(checkpoint),
            "config_sha256": sha256_file(config),
            "frozen_document_ids_sha256": sha256_file(output / "frozen_document_ids.json"),
        }
        return plan, output, config, checkpoint, tokenizer_path

    def test_intact_inputs_verify(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, output, config, checkpoint, _ = self.build_inputs(Path(temporary))
            verified = verify_frozen_inputs(plan, output, config, checkpoint)
            self.assertEqual(verified["config_sha256"], plan["config_sha256"])
            self.assertEqual(verified["frozen_document_ids_sha256"], plan["frozen_document_ids_sha256"])
            self.assertEqual(set(verified["packed_confirmation_sha256"]), {"input_ids_sha256", "segment_ids_sha256", "doc_map_sha256"})
            self.assertTrue(verified["tokenizer_sha256"])

    def test_equal_length_config_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, output, config, checkpoint, _ = self.build_inputs(Path(temporary))
            config.write_text('{"model": {}, "train": {}}\r')  # same length, different bytes
            with self.assertRaisesRegex(ValueError, "config"):
                verify_frozen_inputs(plan, output, config, checkpoint)

    def test_equal_length_frozen_ids_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, output, config, checkpoint, _ = self.build_inputs(Path(temporary))
            frozen = output / "frozen_document_ids.json"
            text = frozen.read_text()
            digit = re.search(r"\d", text)
            self.assertIsNotNone(digit)
            swapped = text[: digit.start()] + str((int(digit.group()) + 1) % 10) + text[digit.end() :]
            self.assertEqual(len(swapped), len(text))
            frozen.write_text(swapped)
            with self.assertRaisesRegex(ValueError, "frozen document IDs"):
                verify_frozen_inputs(plan, output, config, checkpoint)

    def test_equal_length_packed_byte_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, output, config, checkpoint, _ = self.build_inputs(Path(temporary))
            packed = output / "confirmation.input_ids.u16.bin"
            data = bytearray(packed.read_bytes())
            data[0] ^= 0x01  # equal-length edit
            packed.write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_frozen_inputs(plan, output, config, checkpoint)

    def test_tokenizer_identity_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, output, config, checkpoint, tokenizer_path = self.build_inputs(Path(temporary))
            tokenizer_path.write_bytes(b"synthetic tokenizer BYTES")  # equal length
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                verify_frozen_inputs(plan, output, config, checkpoint)
            tokenizer_path.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_frozen_inputs(plan, output, config, checkpoint)

    def test_corpus_manifest_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, output, config, checkpoint, _ = self.build_inputs(Path(temporary))
            plan["corpus_manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "corpus"):
                verify_frozen_inputs(plan, output, config, checkpoint)


if __name__ == "__main__":
    unittest.main()
