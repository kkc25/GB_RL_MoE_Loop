import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data.affine import affine_collate
from eval import causality as cz
from model.loop import validate_budget

STRUCTURAL_TOL = 1e-9   # FP64 deterministic identities (observed exactly 0.0)
REPLAY_TOL = 1e-5       # replay joint accumulates in float32 (observed ~6e-7)


class SharedSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = cz.make_items(4, seed=0)
        batch = affine_collate(cls.items)
        cls.ids, cls.labels = batch["input_ids"], batch["labels"]
        cls.model_to, cls.controller_to = cz.build_pair(target_only=True, seed=0)
        cls.model_fs, cls.controller_fs = cz.build_pair(target_only=False, seed=0)
        cls.model_multi, cls.controller_multi = cz.build_pair(
            adaptive_layers=(1, 3), target_only=False, seed=0)


class TestFixedDepthCalls(SharedSetup):
    def test_target_only_call_counts(self):
        report = cz.check_fixed_depth_calls(self.model_to, self.ids, self.labels)
        for depth in (1, 2, 3):
            entry = report[str(depth)]
            self.assertEqual(entry["target_min"], entry["expected_target"])
            self.assertEqual(entry["target_max"], entry["expected_target"])
            self.assertEqual(entry["nontarget_min"], entry["expected_nontarget"])
            self.assertEqual(entry["nontarget_max"], entry["expected_nontarget"])

    def test_target_only_nontarget_executes_exactly_one_ffn_per_layer(self):
        # With depth=3 the target gets 3 calls at the adaptive layer; every
        # non-target position must still get exactly one FFN call per layer.
        report = cz.check_fixed_depth_calls(self.model_to, self.ids, self.labels, depths=(3,))
        entry = report["3"]
        self.assertEqual(entry["nontarget_min"], self.model_to.config.layers)
        self.assertEqual(entry["nontarget_max"], self.model_to.config.layers)

    def test_full_sequence_call_counts(self):
        report = cz.check_fixed_depth_calls_full_sequence(self.model_fs, self.ids)
        for depth in (1, 2, 3):
            entry = report[str(depth)]
            self.assertEqual(entry["min"], entry["expected"])
            self.assertEqual(entry["max"], entry["expected"])


class TestBudget(SharedSetup):
    def test_zero_violations_greedy_and_sampled(self):
        report = cz.check_budget(self.model_to, self.controller_to, self.ids, self.labels,
                                 budgets=(1, 2, 3), seeds=(0, 1, 2, 3))
        for budget, variants in report.items():
            for name, entry in variants.items():
                with self.subTest(budget=budget, variant=name):
                    self.assertEqual(entry["violations"], 0)
                    self.assertGreaterEqual(entry["min_calls"], 1)
                    self.assertLessEqual(entry["max_calls"], int(budget))

    def test_validate_budget_bounds(self):
        validate_budget(torch.tensor([1, 2, 3]), 1, 3)
        validate_budget(torch.tensor([2, 6]), 2, 3)
        with self.assertRaises(ValueError):
            validate_budget(torch.tensor([0]), 1, 3)
        with self.assertRaises(ValueError):
            validate_budget(torch.tensor([4]), 1, 3)
        with self.assertRaises(ValueError):
            validate_budget(torch.tensor([1]), 2, 3)
        with self.assertRaises(ValueError):
            validate_budget(torch.tensor([7]), 2, 3)

    def test_supervised_positions_contract(self):
        batch, length = self.ids.shape
        none_supervised = torch.full((batch, length), -100, dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model_to.supervised_positions(none_supervised, batch, length, self.ids.device)
        two_supervised = torch.full((batch, length), -100, dtype=torch.long)
        two_supervised[:, 0] = 1
        two_supervised[:, 1] = 2
        with self.assertRaises(ValueError):
            self.model_to.supervised_positions(two_supervised, batch, length, self.ids.device)


class TestPrefixInvariance(SharedSetup):
    def test_fixed_depth_prefix_invariance(self):
        report = cz.check_prefix_invariance(self.model_fs, self.controller_fs,
                                            self.ids, self.labels, cut=10)
        self.assertLessEqual(report["fixed_max_logit_diff"], STRUCTURAL_TOL)
        self.assertEqual(report["fixed_max_call_diff"], 0)

    def test_routed_greedy_prefix_invariance(self):
        report = cz.check_prefix_invariance(self.model_fs, self.controller_fs,
                                            self.ids, self.labels, cut=10)
        self.assertLessEqual(report["routed_max_logit_diff"], STRUCTURAL_TOL)
        self.assertEqual(report["routed_max_call_diff"], 0)


class TestBatchInvariance(SharedSetup):
    def test_fixed_depth_batch_invariance(self):
        report = cz.check_batch_invariance(self.model_fs, self.controller_fs, self.items)
        for i, entry in enumerate(report["fixed"]):
            with self.subTest(item=i):
                self.assertLessEqual(entry["max_logit_diff"], STRUCTURAL_TOL)
                self.assertEqual(entry["max_call_diff"], 0)

    def test_routed_greedy_batch_invariance(self):
        report = cz.check_batch_invariance(self.model_fs, self.controller_fs, self.items)
        for i, entry in enumerate(report["routed"]):
            with self.subTest(item=i):
                self.assertLessEqual(entry["max_logit_diff"], STRUCTURAL_TOL)
                self.assertEqual(entry["max_call_diff"], 0)

    def test_sampled_batch_invariance(self):
        report = cz.check_sampled_batch_invariance(self.model_fs, self.controller_fs, self.items)
        self.assertLessEqual(report["max_logit_diff"], STRUCTURAL_TOL)
        self.assertEqual(report["max_call_diff"], 0)

    def test_sampled_reproducibility(self):
        report = cz.check_sampled_reproducibility(self.model_to, self.controller_to,
                                                  self.ids, self.labels)
        self.assertEqual(report["max_logit_diff"], 0.0)
        self.assertEqual(report["max_call_diff"], 0)


class TestZeroDeltaGreedy(SharedSetup):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.report = cz.check_zero_delta_greedy(cls.model_to, cls.controller_to,
                                                cls.ids, cls.labels)

    def test_stop_after_forced_round_matches_fixed_depth_1(self):
        self.assertLessEqual(self.report["stop_after_forced_round_logit_diff"], STRUCTURAL_TOL)
        self.assertEqual(self.report["stop_after_forced_round_call_diff"], 0)

    def test_forced_continue_matches_fixed_max_depth(self):
        self.assertLessEqual(self.report["forced_continue_logit_diff"], STRUCTURAL_TOL)
        self.assertEqual(self.report["forced_continue_call_diff"], 0)

    def test_greedy_experts_match_gate_argmax(self):
        self.assertTrue(self.report["experts_match_gate_argmax"])


class TestReplayAndAccounting(SharedSetup):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.report = cz.check_replay(cls.model_to, cls.controller_to, cls.ids, cls.labels, seed=0)

    def test_replay_matches_per_decision_log_prob(self):
        self.assertLessEqual(self.report["replay_vs_decision_log_prob"], REPLAY_TOL)

    def test_replay_matches_manual_recomputation(self):
        self.assertLessEqual(self.report["replay_vs_manual"], REPLAY_TOL)

    def test_reference_formula_matches(self):
        self.assertLessEqual(self.report["reference_formula_max_diff"], 1e-6)

    def test_forced_stop_not_double_counted(self):
        # #decisions == #calls + #stops: when the loop ends on legality
        # (budget / max_calls exhausted, i.e. a forced STOP) no extra decision
        # or logprob is recorded.
        self.assertTrue(self.report["counts_equal_calls_plus_stops"])
        self.assertTrue(self.report["at_most_one_stop"])
        self.assertTrue(self.report["stop_is_terminal"])

    def test_first_decision_is_forced_continue(self):
        self.assertTrue(self.report["first_decision_forced"])
        self.assertTrue(self.report["exactly_one_forced"])


class TestRLOOAuxiliary(unittest.TestCase):
    def test_groups_require_identical_input_and_budget(self):
        ids = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 4], [1, 2, 3]])
        budgets = torch.tensor([2, 2, 2, 3])
        self.assertEqual(cz.rloo_groups(ids, budgets), [[0, 1], [2], [3]])

    def test_leave_one_out_hand_values(self):
        rewards = torch.tensor([1.0, -0.5, 3.0, 10.0, 20.0, 30.0])
        groups = [[0, 1], [2], [3, 4, 5]]
        advantages = cz.leave_one_out_advantages(rewards, groups)
        expected = torch.tensor([1.5, -1.5, 0.0, -15.0, 0.0, 15.0], dtype=torch.float64)
        self.assertTrue(torch.equal(advantages, expected))

    def test_zero_sum_within_group(self):
        rewards = torch.tensor([0.3, -1.2, 2.5, 0.9])
        groups = [[0, 1, 2, 3]]
        advantages = cz.leave_one_out_advantages(rewards, groups)
        self.assertAlmostEqual(float(advantages.sum()), 0.0, places=15)

    def test_shift_invariance_within_group(self):
        rewards = torch.tensor([0.3, -1.2, 2.5], dtype=torch.float64)
        groups = [[0, 1, 2]]
        base = cz.leave_one_out_advantages(rewards, groups)
        shifted = cz.leave_one_out_advantages(rewards + 4.7, groups)
        self.assertLessEqual(float((base - shifted).abs().max()), 1e-12)


class TestMultiAdaptive(SharedSetup):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.report = cz.check_multi_adaptive(cls.model_multi, cls.controller_multi,
                                             cls.ids, cls.labels)

    def test_budget_reserve_zero_violations(self):
        for budget, entry in self.report.items():
            with self.subTest(budget=budget):
                self.assertEqual(entry["violations"], 0)
                self.assertGreaterEqual(entry["min_adaptive_calls"], 1)
                self.assertLessEqual(entry["max_adaptive_calls"], int(budget))

    def test_full_sequence_replay_shapes(self):
        batch, length = self.ids.shape
        for budget, entry in self.report.items():
            with self.subTest(budget=budget):
                self.assertEqual(entry["episodes"], entry["expected_episodes"])
                self.assertEqual(entry["expected_episodes"], batch * length)
                self.assertEqual(entry["joint_shape"], [batch * length])
                self.assertTrue(entry["joint_finite"])
                self.assertEqual(entry["decision_index_max"], batch * length - 1)
                self.assertEqual(entry["calls_shape"], [batch, length])

    def test_budget_above_max_calls_fully_usable(self):
        self.assertTrue(self.report["4"]["budget_fully_usable"])
        self.assertTrue(self.report["6"]["budget_fully_usable"])

    def test_target_only_multi_adaptive_is_rejected(self):
        with self.assertRaises(ValueError):
            cz.check_multi_adaptive_target_only(self.ids, self.labels)

    def test_initial_memory_follows_module_dtype(self):
        model, controller = cz.build_pair(target_only=True, seed=0, patch_memory_dtype=False)
        with torch.no_grad():
            model(self.ids, labels=self.labels, controller=controller, budget=2, sample=False)


if __name__ == "__main__":
    unittest.main()
