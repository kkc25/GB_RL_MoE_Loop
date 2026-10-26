"""Unit tests for the controller-role corpus (data/controller_corpus.py) and
document-coverage-first state sampling (eval/state_sampling.py).

All tests run on CPU with synthetic documents, a hand-constructed (never
trained here, byte-alphabet) BPE tokenizer, and a tiny Top-2 LoopMoE.  No
network access, no pre-existing corpus files.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ModelConfig
from data.controller_corpus import (
    ROLES,
    assign_roles,
    build_doc_records,
    encode_document,
    load_exclusion_hashes,
    mark_excluded,
    near_duplicate_hash,
    normalize_near_text,
    prepare_controller_corpus,
)
from data.fineweb import SplitWriter, doc_fingerprint, text_fingerprint
from data.packed_lm import PackedLMDataset
from eval.counterfactual import gather_states
from eval.state_sampling import (
    document_paired_bootstrap,
    plan_state_sampling,
    sample_states,
    supervised_positions_by_doc,
)
from model import LoopMoE


def synthetic_tokenizer():
    """A byte-level BPE with the full byte alphabet and no merges.

    Constructed directly (no training): encoding maps every byte to its own
    token id, which is exactly what the corpus builder needs from a
    caller-supplied tokenizer.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    vocab = {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2}
    for char in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[char] = len(vocab)
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def synthetic_docs(n: int = 80):
    return [(f"doc-{i}", f"content of document number {i} with some text") for i in range(n)]


def tiny_model(seed: int = 7) -> LoopMoE:
    torch.manual_seed(seed)
    config = ModelConfig(
        layers=3,
        hidden_size=32,
        attention_heads=2,
        head_dim=16,
        vocab_size=300,
        intermediate_size=48,
        experts_per_layer=4,
        top_k=2,
        adaptive_layers=(2,),
        moe_layers=(0, 1, 2),
        max_calls=4,
        target_only=False,
        max_position_embeddings=32,
    )
    return LoopMoE(config).eval()


class NearDuplicateTests(unittest.TestCase):
    def test_case_punctuation_whitespace_variants_share_group(self):
        base = "The Quick Brown Fox, jumps over 13 dogs!"
        variants = [
            base,
            "the quick brown fox jumps over 13 dogs",
            "  THE   QUICK\nbrown FOX jumps   over 13 DOGS ",
        ]
        hashes = {near_duplicate_hash(v) for v in variants}
        self.assertEqual(len(hashes), 1)

    def test_distinct_texts_usually_distinct_groups(self):
        self.assertNotEqual(
            near_duplicate_hash("alpha beta gamma"), near_duplicate_hash("delta epsilon zeta")
        )

    def test_same_group_same_role_regardless_of_source_order(self):
        docs = synthetic_docs(30)
        docs.append(("dup-a", "Shared duplicate text here"))
        docs.append(("dup-b", "shared DUPLICATE text here."))  # near-dup of dup-a
        records_a = build_doc_records(docs)
        records_b = build_doc_records(list(reversed(docs)))
        kwargs = dict(newtest_bytes=200, calibration_bytes=200, selection_bytes=200)
        roles_a = assign_roles(records_a, **kwargs)
        roles_b = assign_roles(records_b, **kwargs)
        role_of_a = {rec["id"]: role for role, recs in roles_a.items() for rec in recs}
        role_of_b = {rec["id"]: role for role, recs in roles_b.items() for rec in recs}
        self.assertEqual(role_of_a, role_of_b)
        self.assertEqual(role_of_a["dup-a"], role_of_a["dup-b"])

    def test_exact_duplicates_share_role(self):
        docs = synthetic_docs(30) + [("x1", "identical body"), ("x2", "identical body")]
        records = build_doc_records(docs)
        roles = assign_roles(records, newtest_bytes=200, calibration_bytes=200, selection_bytes=200)
        role_of = {rec["id"]: role for role, recs in roles.items() for rec in recs}
        self.assertEqual(role_of["x1"], role_of["x2"])


class ExclusionTests(unittest.TestCase):
    def test_load_exclusion_hashes_reads_doc_and_text_hashes(self):
        docs = synthetic_docs(3)
        records = build_doc_records(docs)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "documents.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps({**rec, "split": "train"}) + "\n")
            exclusion = load_exclusion_hashes(path)
        self.assertEqual(exclusion["doc_hashes"], {rec["doc_hash"] for rec in records})
        self.assertEqual(exclusion["text_hashes"], {rec["text_hash"] for rec in records})

    def test_exclusion_is_group_level(self):
        docs = synthetic_docs(10)
        target_text = docs[3][1]
        near_variant = ("near-3", target_text.upper().replace(",", " "))  # same near group
        records = build_doc_records(docs + [near_variant])
        exclusion = {
            "doc_hashes": {doc_fingerprint(docs[3][0], target_text)},
            "text_hashes": set(),
        }
        excluded = mark_excluded(records, exclusion)
        self.assertTrue(excluded[3])
        self.assertTrue(excluded[len(docs)])  # near-duplicate excluded with its group
        self.assertEqual(sum(excluded), 2)
        roles = assign_roles(
            records, newtest_bytes=100, calibration_bytes=100, selection_bytes=100, excluded=excluded
        )
        assigned_ids = {rec["id"] for recs in roles.values() for rec in recs}
        self.assertNotIn(docs[3][0], assigned_ids)
        self.assertNotIn("near-3", assigned_ids)


class PrepareControllerCorpusTests(unittest.TestCase):
    def run_pipeline(self, root: Path, docs=None, **overrides):
        docs = synthetic_docs() if docs is None else docs
        tokenizer = synthetic_tokenizer()
        kwargs = dict(
            tokenizer=tokenizer,
            revision="synthetic-rev-0",
            source="unit-test",
            seq_len=16,
            selection_tokens=48,
            calibration_tokens=48,
            newtest_tokens=64,
        )
        kwargs.update(overrides)
        manifest = prepare_controller_corpus(lambda: iter(docs), root, **kwargs)
        return manifest, tokenizer, docs

    def test_roles_are_written_and_readable_by_packed_lm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, _ = self.run_pipeline(root)
            self.assertEqual(set(manifest["splits"]), set(ROLES))
            for role in ("newtest", "calibration", "selection"):
                self.assertEqual(manifest["splits"][role]["n_tokens"], manifest["config"][f"{role}_tokens"])
            for role in ROLES:
                dataset = PackedLMDataset(root, role)  # seq_len inferred from manifest
                self.assertEqual(dataset.seq_len, 16)
                self.assertGreater(len(dataset), 0)
                self.assertEqual(dataset.n_tokens, manifest["splits"][role]["n_tokens"])

    def test_roles_are_document_disjoint_and_cover_all_docs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_pipeline(root)
            rows = [json.loads(line) for line in (root / "documents.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 80)
            self.assertEqual({row["role"] for row in rows}, set(ROLES))
            # One role per document id; near-duplicates of a doc share its role.
            self.assertEqual(len({row["id"] for row in rows}), 80)

    def test_deterministic_across_runs(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            manifest_a, _, _ = self.run_pipeline(Path(a))
            manifest_b, _, _ = self.run_pipeline(Path(b))
            for role in ROLES:
                for key in ("input_ids_sha256", "segment_ids_sha256", "doc_map_sha256"):
                    self.assertEqual(manifest_a["splits"][role][key], manifest_b["splits"][role][key])
            self.assertEqual(manifest_a["document_list_sha256"], manifest_b["document_list_sha256"])

    def test_tokenizer_is_never_trained_or_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, tokenizer, _ = self.run_pipeline(root)
            self.assertFalse((root / "tokenizer.json").exists())
            self.assertEqual(manifest["tokenizer"]["trained_by"], "caller")
            self.assertEqual(manifest["tokenizer"]["vocab_size"], tokenizer.get_vocab_size())
            # Encoding still works and carries the special-token wrapper.
            ids = encode_document(tokenizer, "hello")
            self.assertEqual(ids[0], tokenizer.token_to_id("<|bos|>"))
            self.assertEqual(ids[-1], tokenizer.token_to_id("<|eos|>"))

    def test_exclusion_file_removes_docs_from_every_role(self):
        docs = synthetic_docs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = root / "prior_documents.jsonl"
            excluded_records = build_doc_records([docs[5]])
            with open(prior, "w", encoding="utf-8") as f:
                for rec in excluded_records:
                    f.write(json.dumps({**rec, "split": "train"}) + "\n")
            # A near-duplicate of the excluded doc must leave with it.
            docs = docs + [("near-5", docs[5][1].upper())]
            manifest, _, _ = self.run_pipeline(root, docs=docs, exclude_documents=prior)
            self.assertEqual(manifest["n_excluded_documents"], 2)
            rows = [json.loads(line) for line in (root / "documents.jsonl").read_text().splitlines()]
            role_of = {row["id"]: row["role"] for row in rows}
            self.assertEqual(role_of["doc-5"], "excluded")
            self.assertEqual(role_of["near-5"], "excluded")
            self.assertEqual(manifest["exclusion"]["n_doc_hashes"], 1)

    def test_exact_budget_raises_when_pool_too_small(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                self.run_pipeline(Path(temporary), newtest_tokens=10**9)


class StateSamplingTests(unittest.TestCase):
    DOCS = [
        (0, [5, 6, 7, 8, 9, 10]),              # 6 tokens -> 5 supervised
        (1, [11, 12, 13, 14]),                 # 4 tokens -> 3 supervised
        (2, list(range(15, 25))),              # 10 tokens, spans sequences
        (3, [30, 31, 32, 33, 34]),
        (4, [40, 41, 42, 43, 44, 45, 46]),
    ]

    def build_root(self, directory: Path) -> None:
        writer = SplitWriter(directory, "probe", 8)
        for rank, tokens in self.DOCS:
            writer.add_document(rank, tokens)
        writer.finish()
        manifest = {"splits": {"probe": {"seq_len": 8}}}
        (directory / "manifest.json").write_text(json.dumps(manifest))

    def test_plan_covers_documents_before_repeats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_root(root)
            dataset = PackedLMDataset(root, "probe")
            plan = plan_state_sampling(dataset, 5, seed=0, max_per_doc=2)
            docs_first_round = [doc for _, doc, _ in plan["selection"]]
            self.assertEqual(sorted(docs_first_round), [0, 1, 2, 3, 4])
            # Per-document cap is respected and short docs contribute less.
            self.assertTrue(all(count <= 2 for count in plan["doc_counts"].values()))

    def test_plan_is_deterministic_and_raises_when_exhausted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_root(root)
            dataset = PackedLMDataset(root, "probe")
            plan_a = plan_state_sampling(dataset, 6, seed=3, max_per_doc=2)
            plan_b = plan_state_sampling(dataset, 6, seed=3, max_per_doc=2)
            self.assertEqual(plan_a["selection"], plan_b["selection"])
            total = sum(min(2, len(v)) for v in supervised_positions_by_doc(dataset).values())
            with self.assertRaises(ValueError):
                plan_state_sampling(dataset, total + 1, seed=0, max_per_doc=2)

    def test_sampled_states_match_dataset_and_stats(self):
        model = tiny_model()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_root(root)
            dataset = PackedLMDataset(root, "probe")
            states = sample_states(
                model, dataset, 8, split="probe", device=torch.device("cpu"), seed=0, max_per_doc=2
            )
            self.assertEqual(states["h"].shape[0], 8)
            self.assertEqual(states["h"].shape[1], model.config.hidden_size)
            self.assertTrue(torch.isfinite(states["h"]).all())
            stats = states["stats"]
            self.assertEqual(stats["n_documents"], 5)
            self.assertLessEqual(stats["max_doc_states"], 2)
            self.assertAlmostEqual(stats["max_doc_weight"], stats["max_doc_states"] / 8)
            self.assertEqual(stats["n_duplicate_positions"], 0)
            self.assertEqual(stats["doc_id_intersection"], [])
            self.assertEqual(stats["excluded_doc_ids_present"], [])
            for i in range(8):
                item = dataset[int(states["seq_indices"][i])]
                position = int(states["positions"][i])
                self.assertNotEqual(int(item["labels"][position]), -100)
                self.assertEqual(int(item["labels"][position]), int(states["targets"][i]))
                self.assertEqual(int(item["doc_ids"][position]), int(states["doc_ids"][i]))

    def test_sample_states_matches_gather_states_contract(self):
        model = tiny_model()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_root(root)
            dataset = PackedLMDataset(root, "probe")
            legacy = gather_states(model, dataset, 4, split="probe", device=torch.device("cpu"), seed=0)
            sampled = sample_states(model, dataset, 4, split="probe", device=torch.device("cpu"), seed=0)
            for key in ("h", "targets", "positions", "doc_ids", "seq_indices", "split"):
                self.assertIn(key, sampled)
                self.assertEqual(type(sampled[key]), type(legacy[key]))
            self.assertEqual(sampled["h"].dtype, legacy["h"].dtype)
            # Coverage-first: 4 states come from 4 distinct documents, unlike
            # the sequence-order walk of gather_states.
            self.assertEqual(len(set(sampled["doc_ids"].tolist())), 4)

    def test_intersection_reporting(self):
        model = tiny_model()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_root(root)
            dataset = PackedLMDataset(root, "probe")
            states = sample_states(
                model, dataset, 5, split="probe", device=torch.device("cpu"), seed=0,
                reference_doc_ids={0, 2, 99}, exclude_doc_ids={4},
            )
            sampled_docs = set(states["stats"]["doc_ids"])
            self.assertEqual(set(states["stats"]["doc_id_intersection"]), sampled_docs & {0, 2, 99})
            self.assertEqual(set(states["stats"]["excluded_doc_ids_present"]), sampled_docs & {4})


class DocumentPairedBootstrapTests(unittest.TestCase):
    def test_estimate_is_token_weighted_ratio_of_sums(self):
        # doc A: diffs [1, 3] weights [1, 1]; doc B: diff [4] weight [2]
        result = document_paired_bootstrap(
            [2.0, 4.0, 5.0], [1.0, 1.0, 1.0], ["A", "A", "B"],
            weights=[1.0, 1.0, 2.0], n_bootstrap=50,
        )
        self.assertAlmostEqual(result["estimate"], (1 + 3 + 2 * 4) / (1 + 1 + 2))
        self.assertEqual(result["n_documents"], 2)
        self.assertEqual(result["doc_diff_sum"], [4.0, 8.0])
        self.assertEqual(result["doc_tokens"], [2.0, 2.0])
        self.assertAlmostEqual(result["max_doc_token_share"], 0.5)
        self.assertAlmostEqual(result["effective_documents"], 2.0)

    def test_interval_covers_true_effect_and_is_deterministic(self):
        rng = np.random.default_rng(0)
        n_docs, per_doc = 40, 5
        doc_effects = rng.normal(0.5, 0.1, n_docs)
        diffs = np.concatenate([rng.normal(mu, 0.3, per_doc) for mu in doc_effects])
        docs = np.repeat(np.arange(n_docs), per_doc)
        zeros = np.zeros_like(diffs)
        result = document_paired_bootstrap(diffs, zeros, docs, n_bootstrap=500, seed=1)
        repeat = document_paired_bootstrap(diffs, zeros, docs, n_bootstrap=500, seed=1)
        self.assertEqual((result["ci_lo"], result["ci_hi"]), (repeat["ci_lo"], repeat["ci_hi"]))
        self.assertLess(result["ci_lo"], 0.5)
        self.assertGreater(result["ci_hi"], 0.5)
        self.assertLess(result["ci_lo"], result["estimate"])
        self.assertGreater(result["ci_hi"], result["estimate"])

    def test_interval_is_document_level_not_token_level(self):
        # 3 documents with very different means, 1000 tokens each: a token-level
        # interval would be razor-thin; the document bootstrap must stay wide.
        rng = np.random.default_rng(1)
        diffs = np.concatenate([rng.normal(mu, 0.01, 1000) for mu in (0.0, 1.0, 2.0)])
        docs = np.repeat(np.arange(3), 1000)
        result = document_paired_bootstrap(diffs, np.zeros_like(diffs), docs, n_bootstrap=1000, seed=2)
        self.assertGreater(result["se"], 0.2)
        self.assertEqual(result["n_documents"], 3)
        self.assertEqual(result["n_rows"], 3000)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            document_paired_bootstrap([1.0], [1.0, 2.0], ["A", "B"])
        with self.assertRaises(ValueError):
            document_paired_bootstrap([], [], [])
        with self.assertRaises(ValueError):
            document_paired_bootstrap([1.0], [0.0], ["A"], weights=[0.0])


if __name__ == "__main__":
    unittest.main()
