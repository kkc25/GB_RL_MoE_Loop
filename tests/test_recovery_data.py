"""Unit tests for data.recovery and eval.recovery_metrics (recovery plan
sec. 4/5, appendices A/B)."""

import math
import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import affine as A
from data import recovery as R
from eval import recovery_metrics as M


def make_logits(probs):
    """[1, 1, V] logits whose softmax is the given probability list."""
    return torch.log(torch.tensor(probs, dtype=torch.float64)).float().view(1, 1, -1)


class TestTokenContract(unittest.TestCase):
    def test_vocab_layout_reused(self):
        self.assertEqual(R.VOCAB_SIZE, 512)
        self.assertEqual((R.PAD, R.BOS, R.QRY, R.FNC), (0, 1, 3, 4))
        self.assertEqual(A.sym(0), 5)
        for p in (11, 13, 97):
            self.assertLessEqual(A.sym(p - 1), 101)

    def test_input_has_no_answer(self):
        for stage in R.STAGES:
            ds = R.build_datasets(stage)[R.TRAIN]
            for i in (0, len(ds) // 2, len(ds) - 1):
                item = ds[i]
                ids = item["input_ids"].tolist()
                # [BOS, x, program, QRY]: starts with BOS, ends with QRY,
                # no answer token or EOS appended (sec. 4.1).
                self.assertEqual(ids[0], A.BOS)
                self.assertEqual(ids[-1], A.QRY)
                self.assertNotIn(A.EOS, ids)
                self.assertEqual(item["query_position"], len(ids) - 1)
                self.assertEqual(len(ids), 3 + 3 * item["meta"]["program_length"])

    def test_encode_decode_roundtrip(self):
        rng = random.Random(0)
        for p in (11, 13, 97):
            for _ in range(50):
                k = rng.randrange(0, 4)
                ops = tuple(
                    (rng.randrange(p), rng.randrange(p)) for _ in range(k)
                )
                if k and ops[0][0] >= p:
                    continue
                inst = R.Instance(rng.randrange(p), ops, p, R.TRAIN)
                x, decoded = R.decode_input(R.encode_instance(inst), p)
                self.assertEqual((x, decoded), (inst.x, inst.ops))

    def test_decode_rejects_malformed(self):
        with self.assertRaises(ValueError):
            R.decode_input([A.BOS, A.sym(1)], 13)  # missing QRY
        with self.assertRaises(ValueError):
            R.decode_input([A.BOS, A.sym(1), A.FNC, A.sym(1), A.QRY], 13)  # truncated op


class TestInterpreter(unittest.TestCase):
    def test_run_matches_canonical(self):
        rng = random.Random(1)
        for p in (11, 13, 97):
            for _ in range(200):
                k = rng.randrange(0, 4)
                ops = tuple(
                    (rng.randrange(p), rng.randrange(p)) for _ in range(k)
                )
                x = rng.randrange(p)
                A_, B_ = R.canonical_affine(ops, p)
                self.assertEqual(R.run_program(x, ops, p), (A_ * x + B_) % p)

    def test_ops_do_not_commute(self):
        # Appendix C: every ordered pair of distinct primitives differs.
        import itertools

        for p in (13, 97):
            for f, g in itertools.permutations(R.COMPOSITION_OPS, 2):
                for x in range(p):
                    self.assertNotEqual(
                        R.run_program(x, [f, g], p), R.run_program(x, [g, f], p)
                    )

    def test_identities(self):
        inst = R.Instance(5, ((2, 1), (3, 1)), 13, R.TRAIN)
        self.assertEqual(inst.program_id, "f2,1;f3,1")
        self.assertEqual(inst.instance_id, "p=13|f2,1;f3,1|x=5")
        self.assertEqual(R.program_id(()), "id")
        self.assertEqual(R.canonical_affine((), 13), (1, 0))
        # (2,1) then (3,1): A=6, B=3*1+1=4
        self.assertEqual(inst.canonical, (6 % 13, 4))


class TestC0(unittest.TestCase):
    def test_identity_full_table(self):
        ds = R.build_datasets("c0_identity")[R.TRAIN]
        self.assertEqual(len(ds), 97)
        for i in range(97):
            meta = ds[i]["meta"]
            self.assertEqual(meta["x"], meta["answer"])
            self.assertEqual(meta["program_id"], "id")

    def test_fixed_affine_full_table(self):
        ds = R.build_datasets("c0_fixed")[R.TRAIN]
        self.assertEqual(len(ds), 97)
        xs = set()
        for i in range(97):
            meta = ds[i]["meta"]
            xs.add(meta["x"])
            self.assertEqual(meta["answer"], (2 * meta["x"] + 1) % 97)
            self.assertEqual(meta["ops"], ((2, 1),))
        self.assertEqual(xs, set(range(97)))


class TestC1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = R.build_instances("c1")

    def test_counts(self):
        self.assertEqual(len(self.splits[R.TRAIN]), 847)
        self.assertEqual(len(self.splits[R.ID_VALIDATION]), 182)
        self.assertEqual(len(self.splits[R.ID_TEST]), 181)
        self.assertEqual(sum(len(v) for v in self.splits.values()), 1210)

    def test_disjoint_and_complete(self):
        ids = [i.instance_id for s in self.splits.values() for i in s]
        self.assertEqual(len(ids), len(set(ids)))
        expected = {
            R.instance_id(((a, b),), x, 11)
            for a in range(1, 11)
            for b in range(11)
            for x in range(11)
        }
        self.assertEqual(set(ids), expected)

    def test_nonzero_multiplier(self):
        for insts in self.splits.values():
            for inst in insts:
                self.assertNotEqual(inst.ops[0][0], 0)

    def test_train_covers_digits_and_roles(self):
        train = self.splits[R.TRAIN]
        self.assertEqual({i.ops[0][0] for i in train}, set(range(1, 11)))
        self.assertEqual({i.ops[0][1] for i in train}, set(range(11)))
        self.assertEqual({i.x for i in train}, set(range(11)))

    def test_labels_match_interpreter(self):
        for insts in self.splits.values():
            for inst in insts:
                self.assertEqual(inst.answer, (inst.ops[0][0] * inst.x + inst.ops[0][1]) % 11)

    def test_deterministic_seed(self):
        again = R.build_instances("c1")
        for name in self.splits:
            self.assertEqual(
                [i.instance_id for i in self.splits[name]],
                [i.instance_id for i in again[name]],
            )


class TestCompositionStages(unittest.TestCase):
    def check_stage(self, stage, p, expected, primitive_count, total):
        splits = R.build_instances(stage)
        for name, count in expected.items():
            self.assertEqual(len(splits[name]), count, f"{stage}/{name}")
        self.assertEqual(sum(len(v) for v in splits.values()), total)

        # Instance identity is unique; splits are pairwise disjoint.
        ids = [i.instance_id for s in splits.values() for i in s]
        self.assertEqual(len(ids), len(set(ids)))

        train = splits[R.TRAIN]
        # Primitive full table is training-only and complete.
        primitives = {i.instance_id for i in train if i.program_length == 1}
        self.assertEqual(len(primitives), primitive_count)
        for split in (R.ID_VALIDATION, R.ID_TEST, R.PROGRAM_VALIDATION, R.PROGRAM_TEST):
            self.assertTrue(
                all(i.program_length > 1 for i in splits[split]),
                f"primitives leaked into {split}",
            )
        # All four primitives appear among train programs of every length.
        for length in (1, 2, 3):
            used = {
                op for i in train if i.program_length == length for op in i.ops
            }
            self.assertEqual(used, set(R.COMPOSITION_OPS))

        train_prog_ids = {i.program_id for i in train if i.program_length > 1}
        # ID holdout: same program strings as train, disjoint x.
        for split in (R.ID_VALIDATION, R.ID_TEST):
            prog_ids = {i.program_id for i in splits[split]}
            self.assertTrue(prog_ids <= train_prog_ids)
            held = {(i.program_id, i.x) for i in splits[split]}
            trained = {(i.program_id, i.x) for i in train}
            self.assertTrue(held.isdisjoint(trained))
        # Program holdout: unseen operation strings.
        for split in (R.PROGRAM_VALIDATION, R.PROGRAM_TEST):
            prog_ids = {i.program_id for i in splits[split]}
            self.assertTrue(prog_ids.isdisjoint(train_prog_ids))
        self.assertTrue(
            {i.program_id for i in splits[R.PROGRAM_VALIDATION]}.isdisjoint(
                {i.program_id for i in splits[R.PROGRAM_TEST]}
            )
        )
        # Per train program, x split counts are exact.
        import collections

        by_prog = collections.defaultdict(lambda: collections.Counter())
        for split in (R.TRAIN, R.ID_VALIDATION, R.ID_TEST):
            for i in splits[split]:
                if i.program_length > 1:
                    by_prog[i.program_id][split] += 1
        n_xt, n_xv, n_xs = R.X_SPLIT_COUNTS[p]
        for prog_id, counts in by_prog.items():
            self.assertEqual(counts[R.TRAIN], n_xt, prog_id)
            self.assertEqual(counts[R.ID_VALIDATION], n_xv, prog_id)
            self.assertEqual(counts[R.ID_TEST], n_xs, prog_id)
        self.assertEqual(
            len(by_prog),
            R.LENGTH2_PROGRAM_COUNTS[0] + R.LENGTH3_PROGRAM_COUNTS[0],
        )
        # Labels match the independent interpreter.
        for insts in splits.values():
            for inst in insts:
                self.assertEqual(inst.answer, R.run_program(inst.x, inst.ops, p))
        return splits

    def test_c2(self):
        self.check_stage(
            "c2",
            13,
            {"train": 556, "id_validation": 112, "id_test": 112,
             "program_validation": 156, "program_test": 156},
            primitive_count=52,
            total=1092,
        )

    def test_c3(self):
        self.check_stage(
            "c3",
            97,
            {"train": 4140, "id_validation": 840, "id_test": 840,
             "program_validation": 1164, "program_test": 1164},
            primitive_count=388,
            total=8148,
        )

    def test_program_split_aligned_across_moduli(self):
        # Same seed => same held-out program strings for p=13 and p=97.
        for split in (R.PROGRAM_VALIDATION, R.PROGRAM_TEST):
            c2 = {i.program_id for i in R.build_instances("c2")[split]}
            c3 = {i.program_id for i in R.build_instances("c3")[split]}
            self.assertEqual(c2, c3)

    def test_canonical_matches_instances(self):
        for inst in R.build_instances("c2")[R.TRAIN][:50]:
            A_, B_ = inst.canonical
            self.assertEqual(
                inst.answer, (A_ * inst.x + B_) % 13
            )


class TestCollate(unittest.TestCase):
    def test_right_padding_and_positions(self):
        splits = R.build_instances("c2")
        items = []
        datasets = R.build_datasets("c2")
        # Mixed program lengths 1/2/3 from the train split.
        by_len = {}
        for i in range(len(datasets[R.TRAIN])):
            item = datasets[R.TRAIN][i]
            by_len.setdefault(item["meta"]["program_length"], item)
            if len(by_len) == 3:
                break
        items = [by_len[k] for k in sorted(by_len)]
        batch = R.recovery_collate(items)
        max_len = max(it["input_ids"].numel() for it in items)
        self.assertEqual(batch["input_ids"].shape, (3, max_len))
        for i, item in enumerate(items):
            n = item["input_ids"].numel()
            self.assertTrue(torch.equal(batch["input_ids"][i, :n], item["input_ids"]))
            self.assertTrue(bool((batch["input_ids"][i, n:] == A.PAD).all()))
            self.assertEqual(batch["attention_mask"][i].tolist(), [1] * n + [0] * (max_len - n))
            q = int(batch["query_positions"][i])
            self.assertEqual(q, item["query_position"])
            self.assertEqual(int(batch["input_ids"][i, q]), A.QRY)
            self.assertEqual(int(batch["target_answer_ids"][i]), item["target_answer_id"])
        self.assertEqual(batch["meta"][0]["instance_id"], items[0]["meta"]["instance_id"])

    def test_batch_invariance(self):
        ds = R.build_datasets("c1")[R.TRAIN]
        item = ds[0]
        alone = R.recovery_collate([item])
        mixed = R.recovery_collate([ds[1], item, ds[2]])
        n = item["input_ids"].numel()
        self.assertTrue(torch.equal(alone["input_ids"][0, :n], mixed["input_ids"][1, :n]))
        self.assertEqual(int(alone["query_positions"][0]), int(mixed["query_positions"][1]))
        self.assertEqual(int(alone["target_answer_ids"][0]), int(mixed["target_answer_ids"][1]))

    def test_audit_batch_clean(self):
        ds = R.build_datasets("c2")[R.TRAIN]
        batch = R.recovery_collate([ds[i] for i in range(16)])
        self.assertEqual(R.audit_batch(batch, 13), [])


class TestManifest(unittest.TestCase):
    def test_counts_match_plan(self):
        for stage, expected in R.EXPECTED_COUNTS.items():
            manifest = R.build_manifest(stage)
            for name, count in expected.items():
                self.assertEqual(manifest["splits"][name]["count"], count)
            self.assertEqual(
                manifest["total_instances"], sum(expected.values())
            )
            self.assertEqual(manifest["modulus"], R.STAGE_MODULUS[stage])
            self.assertEqual(manifest["label_contract"], "answer_at_qry_not_in_input")

    def test_hash_stable_and_sensitive(self):
        m1 = R.build_manifest("c2")
        m2 = R.build_manifest("c2")
        self.assertEqual(m1["sha256"], m2["sha256"])
        self.assertNotEqual(m1["sha256"], R.build_manifest("c3")["sha256"])
        self.assertNotEqual(m1["sha256"], R.build_manifest("c2", seed=1)["sha256"])


class TestAudit(unittest.TestCase):
    class _PatchedDataset(R.RecoveryDataset):
        """RecoveryDataset with per-index item mutation (special methods
        resolve on the class, so instance attribute patching would not)."""

        def __init__(self, *args, patches=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._patches = patches or {}

        def __getitem__(self, index):
            item = dict(super().__getitem__(index))
            if index in self._patches:
                self._patches[index](item)
            return item

    def test_c1_full_domain_passes(self):
        # 1,210 examples > 1,000 minimum; zero mismatches required.
        splits = R.build_instances("c1")
        combined = R.RecoveryDataset(
            [i for s in splits.values() for i in s], "c1", "all"
        )
        report = R.audit_dataset(combined)
        self.assertEqual(report.checked, 1210)
        self.assertTrue(report.meets_min_count)
        report.assert_ok()

    def test_c0_small_full_table_passes(self):
        for stage in ("c0_identity", "c0_fixed"):
            report = R.audit_dataset(R.build_datasets(stage)[R.TRAIN])
            self.assertEqual(report.checked, 97)
            self.assertFalse(report.meets_min_count)  # small full table
            report.assert_ok()

    def test_c2_all_splits_pass(self):
        for name, ds in R.build_datasets("c2").items():
            R.audit_dataset(ds).assert_ok()

    def test_detects_corrupted_label(self):
        insts = list(R.build_instances("c1")[R.TRAIN])

        def corrupt(item):
            item["target_answer_id"] = A.sym((insts[0].answer + 1) % 11)

        ds = self._PatchedDataset(insts, "c1", R.TRAIN, patches={0: corrupt})
        report = R.audit_dataset(ds)
        self.assertEqual(len(report.label_mismatches), 1)
        self.assertFalse(report.ok)

    def test_detects_prefix_conflict(self):
        # Same visible prefix, two different labels: must be flagged.
        insts = list(R.build_instances("c1")[R.TRAIN])
        ref = R.RecoveryDataset(insts[:10], "c1", R.TRAIN)[0]

        def conflict(item):
            item["input_ids"] = ref["input_ids"]
            item["query_position"] = ref["query_position"]
            item["target_answer_id"] = A.sym((insts[0].answer + 1) % 11)
            item["meta"] = dict(item["meta"], instance_id="conflict-copy")

        ds = self._PatchedDataset(insts[:10], "c1", R.TRAIN, patches={1: conflict})
        report = R.audit_dataset(ds)
        self.assertTrue(report.prefix_conflicts)
        self.assertFalse(report.ok)

    def test_detects_query_position_error(self):
        insts = list(R.build_instances("c1")[R.TRAIN])

        def shift(item):
            item["query_position"] = item["query_position"] - 1

        ds = self._PatchedDataset(insts[:5], "c1", R.TRAIN, patches={2: shift})
        report = R.audit_dataset(ds)
        self.assertEqual(len(report.query_errors), 1)
        self.assertFalse(report.ok)


class TestMetrics(unittest.TestCase):
    def test_hand_computed_identity(self):
        # Appendix C example: full CE = conditional CE - log(answer mass).
        probs = [0.1, 0.2, 0.05, 0.65]
        answer_ids = torch.tensor([0, 1, 2])
        logits = make_logits(probs)
        input_ids = torch.tensor([[A.QRY]])
        loss, metrics = M.query_loss_and_metrics(
            logits, input_ids, torch.tensor([0]), torch.tensor([1]), answer_ids
        )
        ce_full = -math.log(0.2)
        mass = 0.35
        ce_cond = -math.log(0.2 / mass)
        self.assertAlmostEqual(float(metrics["ce_full"]), ce_full, places=6)
        self.assertAlmostEqual(float(metrics["ce_answer_conditional"]), ce_cond, places=6)
        self.assertAlmostEqual(float(metrics["answer_mass"]), mass, places=6)
        self.assertAlmostEqual(
            ce_full - (float(metrics["ce_answer_conditional"]) - math.log(mass)),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(float(loss), ce_full, places=6)
        self.assertEqual(int(metrics["valid_target_count"]), 1)
        self.assertEqual(int(metrics["eval_count"]), 1)
        self.assertEqual(int(metrics["correct_count"]), 0)  # argmax is token 3

    def test_correct_count(self):
        probs = [0.1, 0.2, 0.05, 0.65]
        _, metrics = M.query_loss_and_metrics(
            make_logits(probs), torch.tensor([[A.QRY]]), torch.tensor([0]),
            torch.tensor([3]), torch.tensor([0, 1, 2, 3]),
        )
        self.assertEqual(int(metrics["correct_count"]), 1)

    def test_contract_assertions(self):
        logits = torch.zeros(1, 1, 8)
        with self.assertRaises(AssertionError):
            M.query_loss_and_metrics(  # position does not hold QRY
                logits, torch.tensor([[A.BOS]]), torch.tensor([0]),
                torch.tensor([5]), torch.tensor([5, 6]),
            )
        with self.assertRaises(AssertionError):
            M.query_loss_and_metrics(  # target outside legal answer set
                logits, torch.tensor([[A.QRY]]), torch.tensor([0]),
                torch.tensor([7]), torch.tensor([5, 6]),
            )
        with self.assertRaises(AssertionError):
            M.query_loss_and_metrics(  # duplicate answer ids
                logits, torch.tensor([[A.QRY]]), torch.tensor([0]),
                torch.tensor([5]), torch.tensor([5, 5]),
            )

    def test_aggregation_is_sum_then_divide(self):
        # Plan appendix C example: batch of 3 with CE 1, batch of 7 with CE 3.
        def batch_metrics(ce_values):
            n = len(ce_values)
            logits = torch.full((n, 1, 8), -1e9)
            targets = torch.full((n,), 5)
            # Two-token support {5, 6}: p_target = exp(-ce).
            for i, ce in enumerate(ce_values):
                p = math.exp(-ce)
                logits[i, 0, 5] = 0.0
                logits[i, 0, 6] = math.log((1 - p) / p) if p < 1 else -1e9
            answer_ids = torch.tensor([5, 6])
            input_ids = torch.full((n, 1), A.QRY)
            _, metrics = M.query_loss_and_metrics(
                logits, input_ids, torch.zeros(n, dtype=torch.long),
                targets, answer_ids,
            )
            return metrics

        b1 = batch_metrics([1.0] * 3)
        b2 = batch_metrics([3.0] * 7)
        agg = M.MetricsAggregator().update(b1).update(b2).compute()
        correct_global = (3 * 1.0 + 7 * 3.0) / 10
        incorrect_mean_of_means = (float(b1["ce_full"]) + float(b2["ce_full"])) / 2
        self.assertAlmostEqual(agg["ce_full"], correct_global, places=5)
        self.assertNotAlmostEqual(agg["ce_full"], incorrect_mean_of_means, places=3)
        self.assertEqual(agg["eval_count"], 10)
        self.assertEqual(agg["num_batches"], 2)
        # aggregate_batches helper agrees.
        self.assertAlmostEqual(
            M.aggregate_batches([b1, b2])["ce_full"], agg["ce_full"], places=7
        )

    def test_aggregator_empty_raises(self):
        with self.assertRaises(ValueError):
            M.MetricsAggregator().compute()

    def test_uniform_baseline(self):
        # Uniform over 512 with 97 legal answers: CE = log 512, mass = 97/512,
        # conditional CE = log 97 (plan sec. 2.1/eq. 1).
        logits = torch.zeros(4, 1, A.VOCAB_SIZE)
        input_ids = torch.full((4, 1), A.QRY)
        targets = torch.tensor([A.sym(0), A.sym(1), A.sym(2), A.sym(96)])
        answer_ids = torch.tensor([A.sym(v) for v in range(97)])
        _, metrics = M.query_loss_and_metrics(
            logits, input_ids, torch.zeros(4, dtype=torch.long), targets, answer_ids
        )
        self.assertAlmostEqual(float(metrics["ce_full"]), math.log(512), places=5)
        self.assertAlmostEqual(
            float(metrics["ce_answer_conditional"]), math.log(97), places=5
        )
        self.assertAlmostEqual(float(metrics["answer_mass"]), 97 / 512, places=5)


class TestEndToEnd(unittest.TestCase):
    def test_collate_feeds_metrics(self):
        # Full path: dataset -> collate -> metric function, on padded batches.
        ds = R.build_datasets("c2")[R.ID_VALIDATION]
        items = [ds[i] for i in range(8)]
        batch = R.recovery_collate(items)
        torch.manual_seed(0)
        logits = torch.randn(8, batch["input_ids"].shape[1], A.VOCAB_SIZE)
        answer_ids = torch.tensor([A.sym(v) for v in range(13)])
        loss, metrics = M.query_loss_and_metrics(
            logits,
            batch["input_ids"],
            batch["query_positions"],
            batch["target_answer_ids"],
            answer_ids,
        )
        self.assertEqual(int(metrics["valid_target_count"]), 8)
        self.assertEqual(loss.shape, ())
        # Predicting the true answer token at every position gives accuracy 1.
        oracle_logits = torch.full((8, batch["input_ids"].shape[1], A.VOCAB_SIZE), -10.0)
        for i in range(8):
            oracle_logits[i, int(batch["query_positions"][i]), int(batch["target_answer_ids"][i])] = 10.0
        _, oracle = M.query_loss_and_metrics(
            oracle_logits, batch["input_ids"], batch["query_positions"],
            batch["target_answer_ids"], answer_ids,
        )
        self.assertEqual(int(oracle["correct_count"]), 8)
        self.assertLess(float(oracle["ce_full"]), 0.01)


if __name__ == "__main__":
    unittest.main()
