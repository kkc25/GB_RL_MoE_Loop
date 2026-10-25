"""Unit tests for the FineWeb-Edu data pipeline (data/fineweb.py + CLI).

All tests run offline on small synthetic documents in temporary directories:
no network access, no large downloads, no model training.
"""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import fineweb as FW
from data import prepare_fineweb as CLI


def synth_docs(n=60, seed_len=5):
    """Deterministic synthetic documents with varied lengths."""
    return [
        (f"doc{i}", f"The quick brown fox number {i} jumps over the lazy dog. " * (3 + i % seed_len))
        for i in range(n)
    ]


def small_kwargs(**overrides):
    cfg = dict(
        revision="unit-test-revision",
        seq_len=64,
        vocab_size=300,
        tokenizer_bytes=4000,
        val_tokens=200,
        test_tokens=200,
        train_tokens=1000,
        val_pool_bytes=1500,
        test_pool_bytes=1500,
    )
    cfg.update(overrides)
    return cfg


class TestFingerprints(unittest.TestCase):
    def test_normalization_collapses_whitespace(self):
        a = FW.text_fingerprint("hello   world\nagain ")
        b = FW.text_fingerprint(" hello world again")
        self.assertEqual(a, b)

    def test_normalization_is_unicode_nfc(self):
        composed = "é"
        decomposed = "é"
        self.assertEqual(FW.text_fingerprint(composed), FW.text_fingerprint(decomposed))

    def test_fingerprints_are_deterministic(self):
        self.assertEqual(
            FW.doc_fingerprint("id1", "some text"), FW.doc_fingerprint("id1", "some text")
        )
        self.assertEqual(
            FW.text_fingerprint("x"),
            hashlib.sha256(FW.normalize_text("x").encode("utf-8")).hexdigest(),
        )

    def test_duplicate_texts_share_text_fingerprint_but_not_doc_fingerprint(self):
        self.assertEqual(FW.text_fingerprint("dup"), FW.text_fingerprint("dup"))
        self.assertNotEqual(
            FW.doc_fingerprint("id-a", "dup"), FW.doc_fingerprint("id-b", "dup")
        )

    def test_split_key_is_stable_across_calls(self):
        self.assertEqual(FW.split_key("abc"), FW.split_key("abc"))
        self.assertNotEqual(FW.split_key("abc"), FW.split_key("abd"))


class TestSplitAssignment(unittest.TestCase):
    def test_duplicates_land_in_the_same_split(self):
        docs = synth_docs(40) + [("dup-a", "identical content here"), ("dup-b", "identical content here")]
        records = FW.build_doc_records(docs)
        splits = FW.assign_splits(records, val_bytes=500, test_bytes=500)
        split_of = {
            rec["id"]: name for name, recs in splits.items() for rec in recs
        }
        self.assertEqual(split_of["dup-a"], split_of["dup-b"])

    def test_assignment_is_invariant_to_input_order(self):
        docs = synth_docs(50)
        records_a = FW.build_doc_records(docs)
        records_b = FW.build_doc_records(list(reversed(docs)))
        splits_a = FW.assign_splits(records_a, val_bytes=500, test_bytes=500)
        splits_b = FW.assign_splits(records_b, val_bytes=500, test_bytes=500)
        for name in ("train", "validation", "test"):
            self.assertEqual(
                sorted(r["id"] for r in splits_a[name]),
                sorted(r["id"] for r in splits_b[name]),
            )

    def test_splits_are_disjoint_and_exhaustive(self):
        records = FW.build_doc_records(synth_docs(50))
        splits = FW.assign_splits(records, val_bytes=800, test_bytes=800)
        hashes = [r["doc_hash"] for name in splits for r in splits[name]]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual(len(hashes), len(records))

    def test_pool_byte_targets_are_met(self):
        records = FW.build_doc_records(synth_docs(80))
        splits = FW.assign_splits(records, val_bytes=1000, test_bytes=1000)
        val_bytes = sum(r["n_bytes"] for r in splits["validation"])
        test_bytes = sum(r["n_bytes"] for r in splits["test"])
        self.assertGreaterEqual(val_bytes, 1000)
        self.assertGreaterEqual(test_bytes, 1000)


class TestTokenizer(unittest.TestCase):
    def _train(self, tmp, texts, vocab_size=300):
        return FW.train_byte_bpe(texts, Path(tmp) / "tokenizer.json", vocab_size=vocab_size)

    def test_vocab_size_includes_special_tokens(self):
        # The corpus needs enough distinct byte pairs for the requested number
        # of merges; with a rich corpus the trained vocab must hit the budget
        # exactly, special tokens included (3 specials + 256 bytes + merges).
        rich = [f"document {i}: the {i * 7 % 97} quick foxes jump over {i % 13} lazy dogs." for i in range(200)]
        with tempfile.TemporaryDirectory() as td:
            info = self._train(td, rich, vocab_size=300)
            self.assertEqual(info["vocab_size"], 300)
            tok = FW.load_tokenizer(info["path"])
            for i, special in enumerate(FW.SPECIAL_TOKENS):
                self.assertEqual(tok.token_to_id(special), i)
            self.assertLess(tok.token_to_id("<|eos|>"), info["vocab_size"])

    def test_vocab_size_is_capped_when_merges_run_out(self):
        # A degenerate corpus with too few distinct pairs must not exceed the
        # requested size either; the trainer stops early and we report the
        # actual size rather than padding the vocab with fake tokens.
        with tempfile.TemporaryDirectory() as td:
            info = self._train(td, ["hello world"] * 50, vocab_size=300)
            self.assertLessEqual(info["vocab_size"], 300)
            self.assertGreaterEqual(info["vocab_size"], len(FW.SPECIAL_TOKENS) + 256)

    def test_special_tokens_get_first_ids(self):
        with tempfile.TemporaryDirectory() as td:
            info = self._train(td, ["hello world"] * 50)
            tok = FW.load_tokenizer(info["path"])
            self.assertEqual(tok.token_to_id("<|pad|>"), FW.PAD_ID)
            self.assertEqual(tok.token_to_id("<|bos|>"), FW.BOS_ID)
            self.assertEqual(tok.token_to_id("<|eos|>"), FW.EOS_ID)

    def test_byte_level_roundtrip_including_unicode(self):
        with tempfile.TemporaryDirectory() as td:
            info = self._train(td, ["héllo wörld 123!"] * 30)
            tok = FW.load_tokenizer(info["path"])
            text = "héllo wörld 123! 未見文字"
            decoded = tok.decode(tok.encode(text, add_special_tokens=False).ids)
            self.assertEqual(decoded, text)

    def test_training_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            texts = [f"document {i} with some text" for i in range(40)]
            a = self._train(td, texts, vocab_size=290)
            b = self._train(td, texts, vocab_size=290)
            self.assertEqual(a["sha256"], b["sha256"])

    def test_encode_document_wraps_bos_eos(self):
        with tempfile.TemporaryDirectory() as td:
            info = self._train(td, ["abc def"] * 20)
            tok = FW.load_tokenizer(info["path"])
            ids = FW.encode_document(tok, "abc def")
            self.assertEqual(ids[0], FW.BOS_ID)
            self.assertEqual(ids[-1], FW.EOS_ID)
            self.assertEqual(ids[1:-1], tok.encode("abc def", add_special_tokens=False).ids)

    def test_corpus_takes_deterministic_byte_budget_and_train_docs_only(self):
        docs = synth_docs(30)
        records = FW.build_doc_records(docs)
        splits = FW.assign_splits(records, val_bytes=500, test_bytes=500)
        train_hashes = {r["doc_hash"] for r in splits["train"]}
        texts, used, total = FW.collect_tokenizer_corpus(iter(docs), train_hashes, 2000)
        self.assertLessEqual(total, 2000)
        self.assertGreater(total, 0)
        self.assertTrue(set(used) <= train_hashes)
        # Same call again -> identical corpus.
        texts2, used2, total2 = FW.collect_tokenizer_corpus(iter(docs), train_hashes, 2000)
        self.assertEqual((texts, used, total), (texts2, used2, total2))


class TestPacking(unittest.TestCase):
    def test_boundaries_offsets_and_reconstruction(self):
        docs = [(0, list(range(10, 15))), (1, list(range(100, 140))), (2, [7, 8, 9])]
        seqs, stats = FW.pack_documents(docs, 16, keep_partial=True)
        # 48 real tokens -> 3 sequences of 16, no padding.
        self.assertEqual(stats["n_tokens"], 48)
        self.assertEqual(stats["n_sequences"], 3)
        self.assertEqual(stats["n_padding"], 0)
        for seq in seqs:
            self.assertEqual(len(seq.input_ids), 16)
            self.assertEqual(len(seq.segment_ids), 16)
            self.assertTrue((seq.segment_ids >= 0).all())
        # Reconstruct each document from the segment metadata.
        for rank, tokens in docs:
            rebuilt = []
            for seq in seqs:
                for pos, seg in enumerate(seq.segments):
                    if seg["doc"] != rank:
                        continue
                    mask = seq.segment_ids == pos
                    rebuilt.extend(seq.input_ids[mask].tolist())
                    self.assertEqual(len(seg), 3)
                    self.assertEqual(seg["length"], int(mask.sum()))
            self.assertEqual(rebuilt, tokens)

    def test_segment_ids_mark_attention_blocks(self):
        docs = [(0, [1] * 10), (1, [2] * 10)]
        seqs, _ = FW.pack_documents(docs, 16, keep_partial=True)
        seq = seqs[0]
        # tokens 0..9 belong to segment 0, tokens 10..15 to segment 1.
        self.assertEqual(seq.segment_ids[:10].tolist(), [0] * 10)
        self.assertEqual(seq.segment_ids[10:].tolist(), [1] * 6)
        # Block-diagonal attention: equal segment ids <-> may attend.
        block = seq.segment_ids[:, None] == seq.segment_ids[None, :]
        self.assertFalse(bool(block[0, 10]))
        self.assertTrue(bool(block[0, 9]))

    def test_partial_sequence_padding(self):
        seqs, stats = FW.pack_documents([(0, [5] * 20)], 16, keep_partial=True)
        self.assertEqual(stats["n_sequences"], 2)
        self.assertEqual(stats["n_padding"], 12)
        tail = seqs[-1]
        self.assertEqual(tail.segment_ids[4:].tolist(), [-1] * 12)
        self.assertEqual(tail.input_ids[4:].tolist(), [FW.PAD_ID] * 12)

    def test_drop_partial(self):
        seqs, stats = FW.pack_documents([(0, [5] * 20)], 16, keep_partial=False)
        self.assertEqual(stats["n_sequences"], 1)
        self.assertEqual(stats["n_dropped_tokens"], 4)
        self.assertEqual(stats["n_tokens"], 16)

    def test_document_spanning_sequences(self):
        seqs, stats = FW.pack_documents([(3, [9] * 40)], 16, keep_partial=True)
        offsets = [seg["offset"] for seq in seqs for seg in seq.segments]
        self.assertEqual(offsets, [0, 16, 32])
        lengths = [seg["length"] for seq in seqs for seg in seq.segments]
        self.assertEqual(lengths, [16, 16, 8])
        self.assertEqual(stats["n_tokens"], 40)
        self.assertTrue(all(seg["doc"] == 3 for seq in seqs for seg in seq.segments))

    def test_document_spanning_sequences_drop_partial_loses_tail(self):
        seqs, stats = FW.pack_documents([(3, [9] * 40)], 16, keep_partial=False)
        offsets = [seg["offset"] for seq in seqs for seg in seq.segments]
        self.assertEqual(offsets, [0, 16])  # the last 8 tokens are dropped
        self.assertEqual(stats["n_dropped_tokens"], 8)
        self.assertEqual(stats["n_tokens"], 32)


class TestTokenBudgets(unittest.TestCase):
    def test_take_first_tokens_exact_and_prefix(self):
        docs = [(0, list(range(10))), (1, list(range(100, 110))), (2, list(range(200, 210)))]
        trimmed, n = FW.take_first_tokens(docs, 15)
        self.assertEqual(n, 15)
        self.assertEqual(trimmed[0], docs[0])
        self.assertEqual(trimmed[1], (1, list(range(100, 105))))
        self.assertEqual(len(trimmed), 2)

    def test_split_writer_matches_batch_pipeline(self):
        docs = [(i, [i + 1] * (7 + 3 * i)) for i in range(6)]
        trimmed, _ = FW.take_first_tokens(docs, 40)
        seqs, stats = FW.pack_documents(trimmed, 16, keep_partial=True)
        with tempfile.TemporaryDirectory() as td:
            writer = FW.SplitWriter(Path(td), "validation", 16, max_tokens=40)
            for rank, tokens in docs:
                writer.add_document(rank, tokens)
            info = writer.finish()
            self.assertEqual(info["n_tokens"], stats["n_tokens"])
            self.assertEqual(info["n_sequences"], stats["n_sequences"])
            disk_ids = np.fromfile(Path(td) / "validation.input_ids.u16.bin", dtype=np.uint16)
            mem_ids = np.concatenate([s.input_ids for s in seqs])
            self.assertTrue((disk_ids == mem_ids).all())
            disk_seg = np.fromfile(Path(td) / "validation.segment_ids.i32.bin", dtype=np.int32)
            mem_seg = np.concatenate([s.segment_ids for s in seqs])
            self.assertTrue((disk_seg == mem_seg).all())

    def test_split_writer_stops_at_budget(self):
        with tempfile.TemporaryDirectory() as td:
            writer = FW.SplitWriter(Path(td), "train", 16, max_tokens=10)
            self.assertTrue(writer.add_document(0, [1] * 6))
            self.assertFalse(writer.add_document(1, [2] * 8))  # trimmed to 4
            self.assertFalse(writer.add_document(2, [3] * 8))  # rejected
            info = writer.finish()
            self.assertEqual(info["n_tokens"], 10)
            self.assertEqual(info["n_documents"], 2)


class TestPrepareEndToEnd(unittest.TestCase):
    def _run(self, tmp, docs=None, **overrides):
        docs = synth_docs() if docs is None else docs
        return FW.prepare_documents(lambda: iter(docs), Path(tmp), **small_kwargs(**overrides))

    def test_exact_split_token_counts_and_files(self):
        with tempfile.TemporaryDirectory() as td:
            m = self._run(td)
            self.assertEqual(m["splits"]["validation"]["n_tokens"], 200)
            self.assertEqual(m["splits"]["test"]["n_tokens"], 200)
            self.assertEqual(m["splits"]["train"]["n_tokens"], 1000)
            for split in ("train", "validation", "test"):
                info = m["splits"][split]
                ids = np.fromfile(Path(td) / info["input_ids_file"], dtype=np.uint16)
                seg = np.fromfile(Path(td) / info["segment_ids_file"], dtype=np.int32)
                self.assertEqual(ids.shape, (info["n_sequences"] * 64,))
                self.assertEqual(seg.shape, ids.shape)
                self.assertEqual(int((seg >= 0).sum()), info["n_tokens"])
                self.assertLess(int(ids.max()), 300)
                # Padding only in the final sequence.
                self.assertTrue((seg[:-64] >= 0).all())
                lines = (Path(td) / info["doc_map_file"]).read_text().strip().splitlines()
                self.assertEqual(len(lines), info["n_sequences"])

    def test_provenance_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            m = self._run(td)
            self.assertEqual(m["revision"], "unit-test-revision")
            self.assertEqual(m["source"], FW.REPO_ID)
            self.assertEqual(m["source_config"], FW.SOURCE_CONFIG)
            self.assertEqual(m["config"]["seq_len"], 64)
            self.assertEqual(m["config"]["vocab_size"], 300)
            self.assertEqual(m["config"]["train_tokens"], 1000)
            doc_file = Path(td) / "documents.jsonl"
            self.assertEqual(m["document_list_sha256"], FW.sha256_file(doc_file))
            self.assertEqual(m["tokenizer"]["sha256"], FW.sha256_file(Path(td) / "tokenizer.json"))
            rows = [json.loads(x) for x in doc_file.read_text().splitlines()]
            self.assertEqual(len(rows), m["n_documents"])
            self.assertTrue(all({"id", "text_hash", "doc_hash", "split", "rank"} <= set(r) for r in rows))
            self.assertEqual(
                sum(m["split_documents"].values()), m["n_documents"]
            )
            on_disk = json.loads((Path(td) / "manifest.json").read_text())
            self.assertEqual(on_disk["revision"], m["revision"])

    def test_split_before_tokenizer_fit(self):
        with tempfile.TemporaryDirectory() as td:
            m = self._run(td)
            rows = [json.loads(x) for x in (Path(td) / "documents.jsonl").read_text().splitlines()]
            train_hashes = {r["doc_hash"] for r in rows if r["split"] == "train"}
            corpus = set(m["tokenizer_corpus"]["doc_hashes"])
            self.assertTrue(corpus <= train_hashes)
            self.assertTrue(corpus)  # non-empty corpus
            held_out = {r["doc_hash"] for r in rows if r["split"] != "train"}
            self.assertTrue(corpus.isdisjoint(held_out))

    def test_pipeline_is_deterministic(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ma = self._run(a)
            mb = self._run(b)
            self.assertEqual(ma["document_list_sha256"], mb["document_list_sha256"])
            self.assertEqual(ma["tokenizer"]["sha256"], mb["tokenizer"]["sha256"])
            for split in ("train", "validation", "test"):
                self.assertEqual(
                    ma["splits"][split]["input_ids_sha256"], mb["splits"][split]["input_ids_sha256"]
                )
                self.assertEqual(
                    ma["splits"][split]["doc_map_sha256"], mb["splits"][split]["doc_map_sha256"]
                )

    def test_revision_is_required(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                self._run(td, revision="")

    def test_insufficient_validation_pool_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, "validation"):
                self._run(td, docs=synth_docs(20), val_tokens=10**9, val_pool_bytes=100)

    def test_train_without_cap_uses_everything(self):
        with tempfile.TemporaryDirectory() as td:
            m = self._run(td, train_tokens=None)
            rows = [json.loads(x) for x in (Path(td) / "documents.jsonl").read_text().splitlines()]
            n_train = sum(1 for r in rows if r["split"] == "train")
            self.assertEqual(m["splits"]["train"]["n_documents"], n_train)


class TestCLI(unittest.TestCase):
    def test_revision_is_required_by_parser(self):
        parser = CLI.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--out-dir", "x"])

    def test_revision_has_no_default(self):
        parser = CLI.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_main_from_jsonl_runs_offline(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw.jsonl"
            with open(raw, "w", encoding="utf-8") as f:
                for doc_id, text in synth_docs():
                    f.write(json.dumps({"id": doc_id, "text": text}) + "\n")
            out = Path(td) / "out"
            cfg = small_kwargs()
            m = CLI.main([
                "--revision", "explicit-test-rev",
                "--from-jsonl", str(raw),
                "--out-dir", str(out),
                "--seq-len", str(cfg["seq_len"]),
                "--vocab-size", str(cfg["vocab_size"]),
                "--tokenizer-bytes", str(cfg["tokenizer_bytes"]),
                "--val-tokens", str(cfg["val_tokens"]),
                "--test-tokens", str(cfg["test_tokens"]),
                "--train-tokens", str(cfg["train_tokens"]),
                "--val-pool-bytes", str(cfg["val_pool_bytes"]),
                "--test-pool-bytes", str(cfg["test_pool_bytes"]),
            ])
            self.assertEqual(m["revision"], "explicit-test-rev")
            self.assertTrue((out / "manifest.json").exists())
            self.assertEqual(m["splits"]["validation"]["n_tokens"], 200)

    @unittest.skipIf(importlib.util.find_spec("pyarrow") is not None, "pyarrow installed")
    def test_fetch_fails_clearly_without_pyarrow(self):
        with self.assertRaisesRegex(RuntimeError, "pyarrow"):
            CLI.fetch_documents("some-revision")


if __name__ == "__main__":
    unittest.main()
