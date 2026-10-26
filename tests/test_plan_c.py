import json
import tempfile
import unittest
from pathlib import Path

import torch

from data.confirmation_corpus import prepare_confirmation_corpus
from data.controller_corpus import build_doc_records
from data.packed_lm import PackedLMDataset
from experiments.prepare_plan_c import write_controller_exclusions
from tests.test_controller_corpus import synthetic_docs, synthetic_tokenizer


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


if __name__ == "__main__":
    unittest.main()
