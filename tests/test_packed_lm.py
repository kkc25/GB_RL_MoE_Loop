"""Unit tests for the packed-split LM reader (data/packed_lm.py).

All tests run offline: synthetic packed splits are written to a temporary
directory with ``data.fineweb.SplitWriter`` (the same writer the real
pipeline uses), plus one hand-written manifest.json.  No network, no
tokenizer training, no model.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def _load_module(name: str, path: Path):
    """Load a data/*.py module directly, bypassing data/__init__.py.

    The package __init__ pulls in torch (via data/affine.py), which the
    packed-LM reader and these tests do not need; loading the module files
    directly keeps the tests runnable in minimal numpy-only environments.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass processing looks the module up here
    spec.loader.exec_module(module)
    return module


FW = _load_module("fineweb", _REPO_ROOT / "data" / "fineweb.py")
PLM = _load_module("packed_lm", _REPO_ROOT / "data" / "packed_lm.py")

SEQ_LEN = 8
# Deliberately irregular lengths so documents span sequences, share
# sequences, and leave a padded tail.  Token ids stay below 2**16.
DOCS = [
    (0, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),  # 10 tokens: spans seqs 0-1
    (1, [20, 21, 22]),                              # 3 tokens: shares seq 1
    (2, [30]),                                      # 1 token: shares seq 1
    (3, [40, 41, 42, 43, 44, 45, 46, 47, 48]),      # 9 tokens: spans seqs 2-3
]
TOTAL_TOKENS = sum(len(t) for _, t in DOCS)  # 23


def build_split(root: Path, split: str = "train") -> dict:
    writer = FW.SplitWriter(root, split, SEQ_LEN)
    for rank, tokens in DOCS:
        writer.add_document(rank, tokens)
    info = writer.finish()
    manifest = {
        "pipeline_version": "unit-test",
        "config": {"seq_len": SEQ_LEN},
        "splits": {split: info},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return info


def n_segments_on_disk(root: Path, split: str = "train") -> int:
    with open(root / f"{split}.doc_map.jsonl", encoding="utf-8") as f:
        return sum(len(json.loads(line)["segments"]) for line in f if line.strip())


def reference_item(root: Path, split: str, global_index: int) -> dict:
    """Independent, deliberately naive recomputation of one expected item."""
    ids = np.fromfile(
        root / f"{split}.input_ids.u16.bin", dtype=np.uint16
    ).reshape(-1, SEQ_LEN)[global_index].astype(np.int64)
    segs = np.fromfile(
        root / f"{split}.segment_ids.i32.bin", dtype=np.int32
    ).reshape(-1, SEQ_LEN)[global_index]
    with open(root / f"{split}.doc_map.jsonl", encoding="utf-8") as f:
        doc_map = [json.loads(line)["segments"] for line in f if line.strip()]
    segments = doc_map[global_index]

    labels = np.full(SEQ_LEN, PLM.IGNORE_INDEX, dtype=np.int64)
    doc_ids = np.full(SEQ_LEN, -1, dtype=np.int64)
    for t in range(SEQ_LEN - 1):
        if segs[t] >= 0 and segs[t + 1] == segs[t]:
            labels[t] = ids[t + 1]
    for k, seg in enumerate(segments):
        for t in range(SEQ_LEN):
            if segs[t] == k:
                doc_ids[t] = seg["doc"]
    return {
        "input_ids": ids,
        "segment_ids": segs,
        "labels": labels,
        "doc_ids": doc_ids,
        "segments": segments,
    }


class PackedLMTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.info = build_split(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def open(self, **kwargs):
        return PLM.PackedLMDataset(self.root, "train", **kwargs)


class TestReading(PackedLMTestCase):
    def test_layout_matches_writer_stats(self):
        ds = self.open()
        self.assertEqual(len(ds), self.info["n_sequences"])
        self.assertEqual(ds.seq_len, SEQ_LEN)
        self.assertEqual(ds.n_sequences_disk, self.info["n_sequences"])
        # 23 real tokens over ceil(23/8) = 3 sequences.
        self.assertEqual(len(ds), 3)

    def test_seq_len_inferred_from_manifest(self):
        ds = self.open()  # no seq_len passed
        self.assertEqual(ds.seq_len, SEQ_LEN)

    def test_seq_len_required_without_manifest(self):
        (self.root / "manifest.json").unlink()
        with self.assertRaises(ValueError):
            self.open()
        ds = self.open(seq_len=SEQ_LEN)  # explicit fallback works
        self.assertEqual(len(ds), 3)

    def test_missing_files_raise(self):
        with self.assertRaises(FileNotFoundError):
            PLM.PackedLMDataset(self.root, "nonexistent", seq_len=SEQ_LEN)

    def test_memmap_backing(self):
        ds = self.open()
        self.assertIsInstance(ds._ids, np.memmap)
        self.assertIsInstance(ds._segs, np.memmap)

    def test_items_match_reference(self):
        ds = self.open()
        for i in range(len(ds)):
            ref = reference_item(self.root, "train", i)
            item = ds[i]
            self.assertEqual(item["index"], i)
            np.testing.assert_array_equal(item["input_ids"], ref["input_ids"])
            np.testing.assert_array_equal(item["segment_ids"], ref["segment_ids"])
            np.testing.assert_array_equal(item["labels"], ref["labels"])
            np.testing.assert_array_equal(item["doc_ids"], ref["doc_ids"])
            self.assertEqual(item["segments"], ref["segments"])
            self.assertEqual(item["input_ids"].dtype, np.int64)
            self.assertEqual(item["labels"].dtype, np.int64)

    def test_negative_indexing(self):
        ds = self.open()
        np.testing.assert_array_equal(ds[-1]["input_ids"], ds[len(ds) - 1]["input_ids"])
        with self.assertRaises(IndexError):
            ds[len(ds)]


class TestLabelSemantics(PackedLMTestCase):
    def test_labels_are_inputs_shifted_by_one_within_segment(self):
        ds = self.open()
        for i in range(len(ds)):
            item = ds[i]
            ids, labels, segs = item["input_ids"], item["labels"], item["segment_ids"]
            for t in range(SEQ_LEN - 1):
                if segs[t] >= 0 and segs[t + 1] == segs[t]:
                    self.assertEqual(labels[t], ids[t + 1])

    def test_document_ends_are_not_supervised(self):
        ds = self.open()
        supervised = 0
        for i in range(len(ds)):
            item = ds[i]
            segs, labels = item["segment_ids"], item["labels"]
            pos = 0
            for seg in item["segments"]:
                last = pos + seg["length"] - 1
                self.assertEqual(labels[last], PLM.IGNORE_INDEX)
                self.assertGreaterEqual(segs[last], 0)  # a real segment token
                pos += seg["length"]
            supervised += item["n_supervised_tokens"]
        # Every on-disk segment loses exactly its final-token label; a
        # document cut across sequences contributes one segment per piece.
        self.assertEqual(supervised, TOTAL_TOKENS - n_segments_on_disk(self.root))

    def test_no_cross_document_supervision(self):
        ds = self.open()
        for i in range(len(ds)):
            item = ds[i]
            doc_ids, labels = item["doc_ids"], item["labels"]
            for t in range(SEQ_LEN - 1):
                if doc_ids[t] != doc_ids[t + 1] or doc_ids[t] < 0:
                    self.assertEqual(labels[t], PLM.IGNORE_INDEX)

    def test_padding_is_not_supervised(self):
        ds = self.open()
        last = ds[len(ds) - 1]  # the padded tail sequence
        pad = last["segment_ids"] < 0
        self.assertTrue(pad.any())
        self.assertTrue((last["labels"][pad] == PLM.IGNORE_INDEX).all())
        self.assertTrue((last["doc_ids"][pad] == -1).all())
        self.assertEqual(
            last["n_tokens"] + int(pad.sum()), SEQ_LEN
        )
        self.assertEqual(last["labels"][SEQ_LEN - 1], PLM.IGNORE_INDEX)

    def test_doc_identity_per_token(self):
        ds = self.open()
        seen = set()
        for i in range(len(ds)):
            item = ds[i]
            for seg in item["segments"]:
                seen.add(seg["doc"])
                mask = item["doc_ids"] == seg["doc"]
                self.assertEqual(int(mask.sum()), seg["length"])
        self.assertEqual(seen, {rank for rank, _ in DOCS})


class TestTokenCounts(PackedLMTestCase):
    def test_aggregate_counts(self):
        ds = self.open()
        counts = ds.token_counts()
        self.assertEqual(counts["n_tokens"], TOTAL_TOKENS)
        self.assertEqual(
            counts["n_supervised_tokens"], TOTAL_TOKENS - n_segments_on_disk(self.root)
        )
        self.assertEqual(
            counts["n_padding"], len(ds) * SEQ_LEN - TOTAL_TOKENS
        )
        per_item = sum(ds[i]["n_tokens"] for i in range(len(ds)))
        self.assertEqual(per_item, counts["n_tokens"])

    def test_doc_map_lengths_match_real_tokens(self):
        ds = self.open()
        for i in range(len(ds)):
            item = ds[i]
            self.assertEqual(
                sum(seg["length"] for seg in item["segments"]), item["n_tokens"]
            )


class TestOffsetAndLimit(PackedLMTestCase):
    def test_offset_view(self):
        ds = self.open(offset=1)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0]["index"], 1)
        full = self.open()
        np.testing.assert_array_equal(ds[0]["input_ids"], full[1]["input_ids"])

    def test_offset_and_limit(self):
        ds = self.open(offset=1, limit=1)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]["index"], 1)

    def test_limit_clipped_at_end(self):
        ds = self.open(offset=2, limit=100)
        self.assertEqual(len(ds), 1)

    def test_offset_counts_only_cover_view(self):
        ds = self.open(offset=2)
        # From sequence 2 on, only doc 3's tail (7 tokens) is present.
        expected = sum(s["length"] for s in ds[0]["segments"])
        self.assertEqual(ds.n_tokens, expected)
        self.assertEqual(expected, 7)

    def test_offset_out_of_range(self):
        with self.assertRaises(ValueError):
            self.open(offset=4)


class TestAttentionMaskMetadata(PackedLMTestCase):
    def test_block_diagonal_mask(self):
        seg_ids = np.array([0, 0, 0, 1, 1, -1, -1, -1], dtype=np.int32)
        mask = PLM.block_diagonal_attention_mask(seg_ids)
        self.assertEqual(mask.shape, (SEQ_LEN, SEQ_LEN))
        self.assertEqual(mask.dtype, bool)
        self.assertTrue(mask[0, 0] and mask[2, 0] and mask[0, 2])
        self.assertTrue(mask[3, 4] and mask[4, 3])
        self.assertFalse(mask[2, 3])  # document boundary
        self.assertFalse(mask[5, 5])  # padding attends to nothing
        self.assertFalse(mask[0, 5])

    def test_mask_matches_item_segments(self):
        ds = self.open()
        item = ds[1]  # shared by docs 0 (tail), 1 and 2
        mask = PLM.block_diagonal_attention_mask(item["segment_ids"])
        same_doc = item["doc_ids"][:, None] == item["doc_ids"][None, :]
        real = item["doc_ids"] >= 0
        np.testing.assert_array_equal(mask, same_doc & real[:, None] & real[None, :])


class TestBatchSampler(PackedLMTestCase):
    def test_sequential_no_shuffle(self):
        sampler = PLM.PackedLMBatchSampler(10, 3, shuffle=False, drop_last=False)
        batches = list(sampler)
        self.assertEqual(batches, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])

    def test_drop_last(self):
        sampler = PLM.PackedLMBatchSampler(10, 3, shuffle=False, drop_last=True)
        batches = list(sampler)
        self.assertEqual(batches, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.assertEqual(sampler.batches_per_epoch, 3)

    def test_deterministic_for_same_seed_and_epoch(self):
        a = list(PLM.PackedLMBatchSampler(50, 4, seed=7, epoch=0))
        b = list(PLM.PackedLMBatchSampler(50, 4, seed=7, epoch=0))
        self.assertEqual(a, b)

    def test_permutation_is_valid(self):
        sampler = PLM.PackedLMBatchSampler(50, 4, seed=7, epoch=0, drop_last=False)
        flat = [i for batch in sampler for i in batch]
        self.assertEqual(sorted(flat), list(range(50)))

    def test_epoch_changes_permutation(self):
        e0 = list(PLM.PackedLMBatchSampler(50, 4, seed=7, epoch=0))
        e1 = list(PLM.PackedLMBatchSampler(50, 4, seed=7, epoch=1))
        self.assertNotEqual(e0, e1)
        sampler = PLM.PackedLMBatchSampler(50, 4, seed=7, epoch=0)
        sampler.set_epoch(1)
        self.assertEqual(list(sampler), e1)

    def test_start_batch_resumes(self):
        full = list(PLM.PackedLMBatchSampler(20, 4, seed=3, epoch=2))
        resumed = list(PLM.PackedLMBatchSampler(20, 4, seed=3, epoch=2, start_batch=2))
        self.assertEqual(resumed, full[2:])

    def test_len_counts_remaining_batches(self):
        sampler = PLM.PackedLMBatchSampler(20, 4, seed=3, start_batch=2)
        self.assertEqual(sampler.batches_per_epoch, 5)
        self.assertEqual(len(sampler), 3)
        self.assertEqual(len(list(sampler)), 3)

    def test_accepts_dataset(self):
        ds = self.open()
        sampler = PLM.PackedLMBatchSampler(ds, 2, shuffle=False, drop_last=False)
        batches = list(sampler)
        items = [ds[i] for batch in batches for i in batch]
        self.assertEqual([it["index"] for it in items], [0, 1, 2])

    def test_rejects_bad_batch_size(self):
        with self.assertRaises(ValueError):
            PLM.PackedLMBatchSampler(10, 0)


if __name__ == "__main__":
    unittest.main()
