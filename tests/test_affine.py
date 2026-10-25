"""Unit tests for data.affine (P0 mod-97 affine program task)."""

import itertools
import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import affine as A


class TestTokenLayout(unittest.TestCase):
    def test_vocab_size_and_symbol_range(self):
        self.assertEqual(A.VOCAB_SIZE, 512)
        self.assertEqual(A.sym(0), 5)
        self.assertEqual(A.sym(96), 101)
        self.assertLess(A.sym(96), A.VOCAB_SIZE)
        for v in range(A.MODULUS):
            self.assertEqual(A.inv_sym(A.sym(v)), v)

    def test_special_tokens_do_not_collide_with_symbols(self):
        specials = {A.PAD, A.BOS, A.EOS, A.QRY, A.FNC}
        self.assertEqual(len(specials), 5)
        symbol_tokens = {A.sym(v) for v in range(A.MODULUS)}
        self.assertTrue(specials.isdisjoint(symbol_tokens))

    def test_sym_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            A.sym(97)
        with self.assertRaises(ValueError):
            A.inv_sym(A.QRY)


class TestAffineMath(unittest.TestCase):
    def test_apply_matches_definition(self):
        rng = random.Random(0)
        for _ in range(500):
            a, b, z = (rng.randrange(97) for _ in range(3))
            self.assertEqual(A.AffineFunction(a, b).apply(z), (a * z + b) % 97)

    def test_inverse_roundtrip(self):
        rng = random.Random(1)
        for _ in range(200):
            a = rng.randrange(1, 97)
            b = rng.randrange(97)
            f = A.AffineFunction(a, b)
            g = f.inverse()
            for z in (0, 1, 50, 96):
                self.assertEqual(g.apply(f.apply(z)), z)
                self.assertEqual(f.apply(g.apply(z)), z)

    def test_inverse_of_constant_raises(self):
        with self.assertRaises(ValueError):
            A.AffineFunction(0, 5).inverse()

    def test_compose_matches_sequential_application(self):
        rng = random.Random(2)
        for _ in range(200):
            f = A.AffineFunction(rng.randrange(97), rng.randrange(97))
            g = A.AffineFunction(rng.randrange(97), rng.randrange(97))
            fg = f.compose(g)
            for z in (0, 7, 96):
                self.assertEqual(fg.apply(z), f.apply(g.apply(z)))

    def test_program_run_matches_brute_force(self):
        rng = random.Random(3)
        for _ in range(100):
            k = rng.randrange(1, 5)
            funcs = tuple(
                A.AffineFunction(rng.randrange(1, 97), rng.randrange(97))
                for _ in range(k)
            )
            p = A.Program(rng.randrange(97), funcs)
            z = p.z0
            for f in funcs:
                z = (f.a * z + f.b) % 97
            self.assertEqual(p.run(), z)


class TestEncoding(unittest.TestCase):
    def test_known_program_encoding(self):
        # f(z) = 2z + 1 on z0 = 3 -> 7, then g(z) = z + 5 -> 12.
        p = A.Program(3, (A.AffineFunction(2, 1), A.AffineFunction(1, 5)))
        ids, labels = A.encode_program(p)
        expected_ids = [
            A.BOS, A.sym(3),
            A.FNC, A.sym(2), A.sym(1),
            A.FNC, A.sym(1), A.sym(5),
            A.QRY, A.sym(12), A.EOS,
        ]
        self.assertEqual(ids, expected_ids)
        self.assertEqual(p.run(), 12)

    def test_labels_predict_answer_at_query_delimiter(self):
        rng = random.Random(4)
        for k in (1, 2, 3, 4):
            funcs = tuple(
                A.AffineFunction(rng.randrange(1, 97), rng.randrange(97))
                for _ in range(k)
            )
            p = A.Program(rng.randrange(97), funcs)
            ids, labels = A.encode_program(p)
            self.assertEqual(len(ids), A.sequence_length(k))
            supervised = [i for i, y in enumerate(labels) if y != A.IGNORE_INDEX]
            self.assertEqual(supervised, [A.answer_position(k) - 1])
            pos = supervised[0]
            self.assertEqual(labels[pos], A.sym(p.run()))
            self.assertEqual(ids[pos], A.QRY)
            self.assertEqual(ids[pos + 1], A.sym(p.run()))
            self.assertEqual(ids[pos + 2], A.EOS)

    def test_all_token_ids_within_vocab(self):
        ds = A.AffineDataset("train", num_samples=200, seed=11)
        for i in range(len(ds)):
            ids = ds[i]["input_ids"]
            self.assertTrue(bool(((0 <= ids) & (ids < A.VOCAB_SIZE)).all()))


class TestSplits(unittest.TestCase):
    def test_family_assignment_disjoint_and_complete(self):
        for k in (1, 2, 3, 4):
            m = A.family_split_map(k, seed=0)
            train, val, test = set(m["train"]), set(m["validation"]), set(m["test"])
            self.assertTrue(train.isdisjoint(val))
            self.assertTrue(train.isdisjoint(test))
            self.assertTrue(val.isdisjoint(test))
            self.assertEqual(train | val | test, set(A.all_families(k)))
            self.assertTrue(val and test)  # stratified: each split non-empty

    def test_assignment_is_deterministic(self):
        for k in (1, 2, 3):
            self.assertEqual(A.family_split_map(k, seed=7), A.family_split_map(k, seed=7))

    def test_dataset_items_respect_split_families(self):
        for split in ("train", "validation", "test"):
            ds = A.AffineDataset(split, num_samples=400, seed=5)
            for i in range(len(ds)):
                prog = ds.make_program(i)
                allowed = set(A.families_for_split(split, prog.length, seed=5))
                self.assertIn(prog.family, allowed)

    def test_family_overlap_between_instance_samples(self):
        # Instance-level overlap across splits is possible, but no sampled
        # train program family may ever appear in val/test and vice versa.
        fams = {}
        for split in ("train", "validation", "test"):
            ds = A.AffineDataset(split, num_samples=2000, seed=9)
            fams[split] = {ds.make_program(i).family for i in range(len(ds))}
        self.assertTrue(fams["train"].isdisjoint(fams["validation"]))
        self.assertTrue(fams["train"].isdisjoint(fams["test"]))
        self.assertTrue(fams["validation"].isdisjoint(fams["test"]))

    def test_length_stratification(self):
        ds = A.AffineDataset("train", num_samples=400, seed=0, lengths=(1, 2, 3, 4))
        counts = {k: 0 for k in (1, 2, 3, 4)}
        for i in range(len(ds)):
            counts[ds.make_program(i).length] += 1
        self.assertTrue(all(c == 100 for c in counts.values()))

    def test_difficulty_metadata(self):
        ds = A.AffineDataset("train", num_samples=50, seed=3)
        for i in range(len(ds)):
            item = ds[i]
            length, n_mult = item["difficulty"]
            self.assertEqual(length, item["length"])
            self.assertEqual(n_mult, sum(c != "T" for c in item["family"]))


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_items(self):
        d1 = A.AffineDataset("train", num_samples=100, seed=42, offset=0)
        d2 = A.AffineDataset("train", num_samples=100, seed=42, offset=0)
        for i in range(100):
            self.assertTrue(torch.equal(d1[i]["input_ids"], d2[i]["input_ids"]))
            self.assertTrue(torch.equal(d1[i]["labels"], d2[i]["labels"]))

    def test_different_seed_differs(self):
        d1 = A.AffineDataset("train", num_samples=100, seed=1)
        d2 = A.AffineDataset("train", num_samples=100, seed=2)
        same = sum(
            bool(torch.equal(d1[i]["input_ids"], d2[i]["input_ids"])) for i in range(100)
        )
        self.assertLess(same, 100)

    def test_offset_shifts_stream(self):
        d0 = A.AffineDataset("train", num_samples=50, seed=6, offset=0)
        d7 = A.AffineDataset("train", num_samples=50, seed=6, offset=7)
        self.assertTrue(
            torch.equal(d0[7]["input_ids"], d7[0]["input_ids"])
        )

    def test_batch_api_deterministic(self):
        b1 = A.sample_batch("validation", 32, seed=13, offset=64)
        b2 = A.sample_batch("validation", 32, seed=13, offset=64)
        self.assertTrue(torch.equal(b1["input_ids"], b2["input_ids"]))
        self.assertTrue(torch.equal(b1["labels"], b2["labels"]))
        self.assertTrue(torch.equal(b1["attention_mask"], b2["attention_mask"]))

    def test_batch_api_padding_and_shapes(self):
        batch = A.sample_batch("train", 16, seed=0, lengths=(1, 4))
        ids, labels, mask = (
            batch["input_ids"], batch["labels"], batch["attention_mask"]
        )
        self.assertEqual(ids.shape, labels.shape)
        self.assertEqual(ids.shape[0], 16)
        self.assertEqual(ids.shape[1], A.sequence_length(4))  # padded to longest
        # Padding rows: PAD tokens where mask is 0, labels -100 there.
        self.assertTrue(bool((ids[mask == 0] == A.PAD).all()))
        self.assertTrue(bool((labels[mask == 0] == A.IGNORE_INDEX).all()))
        # Exactly one supervised position per row.
        self.assertTrue(
            bool(((labels != A.IGNORE_INDEX).sum(dim=1) == 1).all())
        )
        self.assertEqual(len(batch["meta"]), 16)

    def test_collate_matches_dataset_items(self):
        ds = A.AffineDataset("test", num_samples=8, seed=21, lengths=(2,))
        batch = A.affine_collate([ds[i] for i in range(8)])
        for i in range(8):
            self.assertTrue(torch.equal(batch["input_ids"][i], ds[i]["input_ids"]))
            self.assertEqual(batch["meta"][i]["answer"], ds[i]["answer"])


class TestDiagnostics(unittest.TestCase):
    def test_complementary_program_is_identity(self):
        rng = random.Random(8)
        for _ in range(100):
            z0 = rng.randrange(97)
            a, b = rng.randrange(1, 97), rng.randrange(97)
            p = A.complementary_program(z0, a, b)
            self.assertEqual(p.length, 2)
            self.assertEqual(p.run(), z0)

    def test_default_noncommuting_pair_known_values(self):
        f, g = A.noncommuting_pair()
        # f(z)=2z+1, g(z)=z+1: f∘g(z)=2z+3, g∘f(z)=2z+2.
        self.assertEqual(f.compose(g), A.AffineFunction(2, 3))
        self.assertEqual(g.compose(f), A.AffineFunction(2, 2))
        # Program(z0, (f, g)) applies f first, then g: result is (g∘f)(z0).
        p_f_then_g, p_g_then_f = A.order_diagnostic_programs(10, f, g)
        self.assertEqual(p_f_then_g.run(), (2 * 10 + 2) % 97)
        self.assertEqual(p_g_then_f.run(), (2 * 10 + 3) % 97)
        self.assertNotEqual(p_f_then_g.run(), p_g_then_f.run())

    def test_noncommuting_pair_rejects_commuting(self):
        with self.assertRaises(ValueError):
            A.noncommuting_pair(1, 3, 1, 5)  # two translations always commute

    def test_order_diagnostic_matches_composition(self):
        rng = random.Random(12)
        tried = 0
        while tried < 50:
            f = A.AffineFunction(rng.randrange(1, 97), rng.randrange(97))
            g = A.AffineFunction(rng.randrange(1, 97), rng.randrange(97))
            if f.compose(g) == g.compose(f):
                continue
            tried += 1
            z0 = rng.randrange(97)
            p_f_then_g, p_g_then_f = A.order_diagnostic_programs(z0, f, g)
            # [f, g] applies f first, so the result is (g∘f)(z0), and vice versa.
            self.assertEqual(p_f_then_g.run(), g.compose(f).apply(z0))
            self.assertEqual(p_g_then_f.run(), f.compose(g).apply(z0))
            self.assertNotEqual(p_f_then_g.run(), p_g_then_f.run())


if __name__ == "__main__":
    unittest.main()
