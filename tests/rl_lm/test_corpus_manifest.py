"""T24 data-role / historical manifest format tests for rl_lm.corpus.

Covers the three historical corpus formats (original LM ``fineweb-pipeline-v1``,
four-role ``controller-corpus-v1``, and ``fixed-depth-confirmation-v1``), hash
verification of every declared file, in-corpus rank joins, and the additional
exclusion JSONL path.  All fixtures are synthetic and local; no network.

Fixture builders live here and are reused by ``test_corpus.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data import fineweb
from data.controller_corpus import (
    build_doc_records,
    prepare_controller_corpus,
)
from data.fineweb import SplitWriter, doc_fingerprint, sha256_file, text_fingerprint
from rl_lm.corpus import (
    CONFIRMATION_PIPELINE_VERSION,
    CorpusBuildError,
    collect_exclusions,
)
from tests.test_controller_corpus import synthetic_docs, synthetic_tokenizer

SEQ_LEN = 16


# --------------------------------------------------------------------------
# Synthetic historical corpus fixtures (all three formats)
# --------------------------------------------------------------------------

def make_lm_corpus(root: Path, docs, *, tokenizer=None, n_train=3, n_val=2,
                   extra_tokenizer_docs=()) -> Path:
    """Original LM format: train/validation/test doc maps, no near_hash column."""
    tokenizer = tokenizer or synthetic_tokenizer()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = fineweb.build_doc_records(docs)
    splits = {
        "train": records[:n_train],
        "validation": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:],
    }
    split_stats = {}
    for split, recs in splits.items():
        writer = SplitWriter(root, split, SEQ_LEN)
        for rec in recs:
            writer.add_document(rec["rank"], fineweb.encode_document(tokenizer, docs[rec["rank"]][1]))
        split_stats[split] = writer.finish()
    doc_list_sha = fineweb.write_document_list(records, splits, root / "documents.jsonl")
    tokenizer_hashes = [recs["doc_hash"] for recs in splits["train"][:1]]
    tokenizer_hashes += [
        doc_fingerprint(doc_id, text) for doc_id, text in extra_tokenizer_docs
    ]
    manifest = {
        "pipeline_version": "fineweb-pipeline-v1",
        "document_list_file": "documents.jsonl",
        "document_list_sha256": doc_list_sha,
        "tokenizer_corpus": {"doc_hashes": tokenizer_hashes},
        "splits": split_stats,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


def make_controller_corpus(root: Path, docs=None, *, tokenizer=None) -> Path:
    """Four-role controller format with near_hash in the document list."""
    tokenizer = tokenizer or synthetic_tokenizer()
    docs = docs or synthetic_docs(40)
    prepare_controller_corpus(
        lambda: iter(docs),
        Path(root),
        tokenizer=tokenizer,
        revision="hist-controller-rev",
        source="unit-test",
        seq_len=SEQ_LEN,
        selection_tokens=32,
        calibration_tokens=32,
        newtest_tokens=32,
    )
    return Path(root)


def make_confirmation_corpus(root: Path, docs, used_ranks, *, shuffle=True) -> Path:
    """Hand-built fixed-depth-confirmation-v1 fixture.

    ``confirmation_documents.jsonl`` is deliberately written out of rank
    order; only ``used_ranks`` appear in the doc map, so selected-but-unpacked
    documents exist and must not be counted as used.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = build_doc_records(docs)
    order = list(reversed(records)) if shuffle else list(records)
    table = root / "confirmation_documents.jsonl"
    with open(table, "w", encoding="utf-8") as handle:
        for rec in order:
            handle.write(json.dumps(rec | {"role": "confirmation"}, sort_keys=True) + "\n")
    doc_map = root / "confirmation.doc_map.jsonl"
    with open(doc_map, "w", encoding="utf-8") as handle:
        for seq, rank in enumerate(sorted(used_ranks)):
            handle.write(json.dumps({
                "seq": seq,
                "segments": [{"doc": rank, "offset": 0, "length": 3}],
            }) + "\n")
    manifest = {
        "pipeline_version": CONFIRMATION_PIPELINE_VERSION,
        "confirmation_documents_sha256": sha256_file(table),
        "splits": {
            "confirmation": {
                "seq_len": SEQ_LEN,
                "doc_map_file": doc_map.name,
                "doc_map_sha256": sha256_file(doc_map),
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


def make_real_confirmation_corpus(root: Path, docs, *, tokenizer=None,
                                  confirmation_tokens=64, pool_bytes=10**6) -> Path:
    """Confirmation corpus produced by the real data.confirmation_corpus builder."""
    from data.confirmation_corpus import prepare_confirmation_corpus

    tokenizer = tokenizer or synthetic_tokenizer()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    empty = root / "empty_exclusions.jsonl"
    empty.write_text("")
    prepare_confirmation_corpus(
        lambda: iter(docs),
        root,
        tokenizer=tokenizer,
        revision="hist-confirmation-rev",
        source="unit-test",
        confirmation_tokens=confirmation_tokens,
        pool_bytes=pool_bytes,
        exclude_documents=empty,
        seq_len=SEQ_LEN,
    )
    return root


def doc_map_ranks(corpus_dir: Path, split: str):
    ranks = set()
    with open(corpus_dir / f"{split}.doc_map.jsonl", encoding="utf-8") as handle:
        for line in handle:
            for segment in json.loads(line)["segments"]:
                ranks.add(int(segment["doc"]))
    return ranks


def rewrite_manifest(corpus_dir: Path, mutate) -> None:
    path = Path(corpus_dir) / "manifest.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


class LmFormatTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.docs = synthetic_docs(6)
        self.extra_doc = ("tok-only", "tokenizer training only document text")
        make_lm_corpus(self.root / "lm", self.docs, extra_tokenizer_docs=[self.extra_doc])

    def test_collects_used_docs_from_all_three_splits(self):
        exclusions = collect_exclusions([self.root / "lm"])
        records = fineweb.build_doc_records(self.docs)
        expected_doc = {rec["doc_hash"] for rec in records}
        expected_text = {rec["text_hash"] for rec in records}
        self.assertTrue(expected_doc <= exclusions.doc_hashes)
        self.assertTrue(expected_text <= exclusions.text_hashes)
        # The original LM format has no near_hash: every used doc is pending.
        self.assertEqual(exclusions.near_hashes, set())
        self.assertGreaterEqual(len(exclusions.pending_near), len(self.docs))
        source = exclusions.sources[0]
        self.assertEqual(source["format"], "fineweb-pipeline-v1")
        self.assertEqual(source["splits_traversed"], ["test", "train", "validation"])
        self.assertEqual(source["n_used_documents"], len(self.docs))

    def test_tokenizer_corpus_doc_hashes_are_excluded(self):
        exclusions = collect_exclusions([self.root / "lm"])
        extra_hash = doc_fingerprint(*self.extra_doc)
        self.assertIn(extra_hash, exclusions.doc_hashes)
        # The extra doc is in the document table? No: it was never packed nor
        # listed, so only its content hash identity is excluded.
        source = exclusions.sources[0]
        self.assertEqual(source["n_tokenizer_corpus_doc_hashes"], 2)


class ControllerFormatTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.docs = synthetic_docs(40)
        make_controller_corpus(self.root / "controller", self.docs)

    def test_all_four_role_doc_maps_are_traversed(self):
        corpus = self.root / "controller"
        exclusions = collect_exclusions([corpus])
        source = exclusions.sources[0]
        self.assertEqual(
            source["splits_traversed"],
            ["calibration", "controller_train", "newtest", "selection"],
        )
        # A document used only in newtest would be invisible to the old
        # train/validation/test helper; here it must be excluded.
        newtest_ranks = doc_map_ranks(corpus, "newtest")
        train_ranks = doc_map_ranks(corpus, "controller_train")
        only_newtest = newtest_ranks - train_ranks
        self.assertTrue(only_newtest)
        rows = [
            json.loads(line)
            for line in (corpus / "documents.jsonl").read_text().splitlines()
        ]
        by_rank = {int(row["rank"]): row for row in rows}
        for rank in only_newtest:
            self.assertIn(by_rank[rank]["doc_hash"], exclusions.doc_hashes)
            self.assertIn(by_rank[rank]["near_hash"], exclusions.near_hashes)
        # Controller format carries near_hash: nothing is pending.
        self.assertEqual(exclusions.pending_near, [])
        self.assertEqual(source["n_missing_near_hash"], 0)


class ConfirmationFormatTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.docs = synthetic_docs(8)

    def test_inline_rank_not_file_order_and_selected_not_used(self):
        used = {1, 4, 6}
        corpus = make_confirmation_corpus(self.root / "conf", self.docs, used, shuffle=True)
        exclusions = collect_exclusions([corpus])
        records = build_doc_records(self.docs)
        used_hashes = {records[rank]["doc_hash"] for rank in used}
        unused_hashes = {rec["doc_hash"] for rank, rec in enumerate(records)} - used_hashes
        self.assertTrue(used_hashes <= exclusions.doc_hashes)
        self.assertFalse(unused_hashes & exclusions.doc_hashes)
        self.assertEqual(exclusions.sources[0]["format"], CONFIRMATION_PIPELINE_VERSION)
        self.assertEqual(exclusions.sources[0]["n_used_documents"], len(used))
        # Confirmation rows carry near_hash; nothing pending.
        self.assertEqual(exclusions.pending_near, [])

    def test_real_builder_output(self):
        corpus = make_real_confirmation_corpus(
            self.root / "conf_real", synthetic_docs(20), confirmation_tokens=64
        )
        frozen = json.loads((corpus / "frozen_document_ids.json").read_text())
        exclusions = collect_exclusions([corpus])
        self.assertTrue(set(frozen["document_hashes"]) <= exclusions.doc_hashes)
        self.assertEqual(
            exclusions.sources[0]["n_used_documents"], len(frozen["document_hashes"])
        )
        # Selected-but-never-packed documents are not part of the exposure.
        n_selected = json.loads((corpus / "manifest.json").read_text())["n_selected_documents"]
        self.assertGreaterEqual(n_selected, len(frozen["document_hashes"]))


class HashAndFormatFailureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.docs = synthetic_docs(8)

    def test_doc_map_hash_corruption_fails(self):
        corpus = make_lm_corpus(self.root / "lm", self.docs)
        with open(corpus / "train.doc_map.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"seq": 99, "segments": []}) + "\n")
        with self.assertRaises(CorpusBuildError):
            collect_exclusions([corpus])

    def test_missing_doc_map_file_fails(self):
        corpus = make_lm_corpus(self.root / "lm", self.docs)
        (corpus / "validation.doc_map.jsonl").unlink()
        with self.assertRaises(CorpusBuildError):
            collect_exclusions([corpus])

    def test_undeclared_doc_map_fails(self):
        corpus = make_lm_corpus(self.root / "lm", self.docs)
        rewrite_manifest(corpus, lambda m: m["splits"]["test"].pop("doc_map_file"))
        with self.assertRaises(CorpusBuildError):
            collect_exclusions([corpus])

    def test_document_list_hash_corruption_fails(self):
        corpus = make_lm_corpus(self.root / "lm", self.docs)
        with open(corpus / "documents.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"rank": 100, "doc_hash": "x"}) + "\n")
        with self.assertRaises(CorpusBuildError):
            collect_exclusions([corpus])

    def test_confirmation_documents_hash_corruption_fails(self):
        corpus = make_confirmation_corpus(self.root / "conf", self.docs, {0, 1})
        with open(corpus / "confirmation_documents.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"rank": 100, "doc_hash": "x"}) + "\n")
        with self.assertRaises(CorpusBuildError):
            collect_exclusions([corpus])

    def test_rank_join_failure_fails(self):
        corpus = make_lm_corpus(self.root / "lm", self.docs)
        doc_map = corpus / "train.doc_map.jsonl"
        with open(doc_map, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"seq": 0, "segments": [{"doc": 999, "offset": 0, "length": 1}]}) + "\n")
        rewrite_manifest(
            corpus, lambda m: m["splits"]["train"].update(doc_map_sha256=sha256_file(doc_map))
        )
        with self.assertRaises(CorpusBuildError):
            collect_exclusions([corpus])

    def test_unknown_format_fails(self):
        corpus = make_lm_corpus(self.root / "lm", self.docs)
        rewrite_manifest(corpus, lambda m: (m.pop("document_list_file"), m.pop("pipeline_version")))
        with self.assertRaises(CorpusBuildError):
            collect_exclusions([corpus])

    def test_missing_manifest_fails(self):
        with self.assertRaises(CorpusBuildError):
            collect_exclusions([self.root / "does-not-exist"])


class AdditionalExclusionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_hashes_collected_and_file_recorded(self):
        path = self.root / "extra.jsonl"
        rows = [
            {"doc_hash": "d1", "text_hash": "t1", "near_hash": "n1"},
            {"doc_hash": "d2"},  # no near_hash -> pending rebuild
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        exclusions = collect_exclusions([], [path])
        self.assertEqual(exclusions.doc_hashes, {"d1", "d2"})
        self.assertEqual(exclusions.text_hashes, {"t1"})
        self.assertEqual(exclusions.near_hashes, {"n1"})
        self.assertEqual(len(exclusions.pending_near), 1)
        report = exclusions.additional_files[0]
        self.assertEqual(report["sha256"], sha256_file(path))
        self.assertEqual(report["n_rows"], 2)
        self.assertEqual(report["n_missing_near_hash"], 1)

    def test_row_without_any_hash_fails(self):
        path = self.root / "bad.jsonl"
        path.write_text(json.dumps({"rank": 3}) + "\n")
        with self.assertRaises(CorpusBuildError):
            collect_exclusions([], [path])

    def test_missing_file_fails(self):
        with self.assertRaises(CorpusBuildError):
            collect_exclusions([], [self.root / "nope.jsonl"])


if __name__ == "__main__":
    unittest.main()
