import unittest

import torch

from eval.counterfactual import PathCosts
from experiments.mechanism_depth_interaction import (
    changed_mask,
    depth_contrast,
    depth_gain,
    doc_direction,
    leave_max_doc,
)
from experiments.r2_value import candidate_rank_rule, group_rules


def make_costs(immediate, q_pi0, baseline_action, global_action_ids):
    n_states, n_actions = q_pi0.shape
    return PathCosts(
        immediate,
        q_pi0,
        baseline_action,
        torch.zeros(n_states, n_actions, 2, dtype=torch.long),
        4,
        global_action_ids,
    )


class FormulaTests(unittest.TestCase):
    def test_gain_and_contrast_definitions(self):
        l1_base = torch.tensor([5.0, 6.0])
        l4_base = torch.tensor([4.0, 4.5])
        l1_rule = torch.tensor([6.0, 5.0])
        l4_rule = torch.tensor([4.5, 4.0])
        self.assertTrue(torch.equal(depth_gain(l1_base, l4_base), torch.tensor([1.0, 1.5])))
        contrast = depth_contrast(l1_rule, l4_rule, l1_base, l4_base)
        self.assertTrue(torch.equal(contrast["d_l1"], torch.tensor([1.0, -1.0])))
        self.assertTrue(torch.equal(contrast["d_l4"], torch.tensor([0.5, -0.5])))
        # I = G_rule - G_base = d_l1 - d_l4
        expected_g = depth_gain(l1_rule, l4_rule) - depth_gain(l1_base, l4_base)
        self.assertTrue(torch.allclose(contrast["d_g"], expected_g))
        self.assertTrue(torch.allclose(contrast["d_g"], contrast["d_l1"] - contrast["d_l4"]))
        with self.assertRaises(ValueError):
            depth_gain(torch.ones(2), torch.ones(3))


class ChangedOnlyTests(unittest.TestCase):
    def test_changed_mask_compares_executed_global_actions(self):
        baseline_action = torch.tensor([0, 0, 2])
        global_action_ids = torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
        costs = make_costs(torch.zeros(3, 3), torch.zeros(3, 3), baseline_action, global_action_ids)
        # row 0: local 2 -> global 12 differs from baseline global 10 (changed)
        # row 1: local 0 -> global 20 equals baseline global 20 (fallback-looking but same executed action, not changed)
        # row 2: local 2 -> baseline itself (not changed)
        selection = torch.tensor([2, 0, 2])
        mask = changed_mask(costs, selection)
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_changed_mask_all_baseline_is_empty(self):
        baseline_action = torch.tensor([1, 0])
        global_action_ids = torch.tensor([[5, 6], [7, 8]])
        costs = make_costs(torch.zeros(2, 2), torch.zeros(2, 2), baseline_action, global_action_ids)
        self.assertFalse(bool(changed_mask(costs, baseline_action).any()))


class LeaveMaxDocTests(unittest.TestCase):
    def test_leave_max_doc_drops_max_token_share_document(self):
        diff = torch.tensor([1.0, 1.0, 1.0, -1.0, 3.0])
        doc_ids = torch.tensor([0, 0, 0, 1, 2])
        result = leave_max_doc(diff, doc_ids)
        self.assertEqual(result["dropped_doc_id"], 0)
        self.assertAlmostEqual(result["dropped_token_share"], 0.6)
        self.assertAlmostEqual(result["estimate_without_doc"], 1.0)
        self.assertAlmostEqual(result["delta_vs_full"], 1.0 - 1.0)
        with self.assertRaises(ValueError):
            leave_max_doc(torch.tensor([1.0]), torch.tensor([0]))

    def test_doc_direction_counts_per_document_signs(self):
        diff = torch.tensor([1.0, 2.0, -5.0, 0.0, 0.0])
        doc_ids = torch.tensor([0, 0, 1, 1, 2])
        direction = doc_direction(diff, doc_ids)
        self.assertEqual(direction["n_documents"], 3)
        self.assertEqual(direction["n_positive_rule_worse"], 1)
        self.assertEqual(direction["n_negative_rule_better"], 1)
        self.assertEqual(direction["n_zero"], 1)


class NoLabelSelectionTests(unittest.TestCase):
    def _train_costs(self):
        q_pi0 = torch.tensor([
            [4.0, 4.2, 4.1, 4.3],
            [4.1, 4.0, 4.2, 4.4],
            [4.2, 4.1, 4.0, 4.3],
            [4.0, 4.3, 4.1, 4.2],
        ])
        immediate = q_pi0 + 1.0
        baseline_action = torch.tensor([0, 1, 2, 0])
        global_action_ids = torch.tensor([[10, 11, 12, 13], [10, 11, 12, 13], [10, 11, 12, 13], [10, 11, 12, 13]])
        return make_costs(immediate, q_pi0, baseline_action, global_action_ids)

    def _test_costs(self, fill):
        q_pi0 = torch.full((3, 4), fill)
        immediate = torch.full((3, 4), fill + 1.0)
        baseline_action = torch.tensor([0, 1, 2])
        global_action_ids = torch.tensor([[10, 11, 12, 13], [10, 11, 12, 13], [20, 21, 22, 23]])
        return make_costs(immediate, q_pi0, baseline_action, global_action_ids)

    def test_rules_ignore_test_cost_values(self):
        train = self._train_costs()
        test_a = self._test_costs(4.0)
        test_b = self._test_costs(99.0)  # adversarially different test costs / labels
        rank_a, meta_a = candidate_rank_rule(train, test_a)
        rank_b, meta_b = candidate_rank_rule(train, test_b)
        self.assertTrue(torch.equal(rank_a, rank_b))
        self.assertEqual(meta_a, meta_b)
        constant_a, group_a, group_meta_a = group_rules(train, test_a, min_group_states=1)
        constant_b, group_b, group_meta_b = group_rules(train, test_b, min_group_states=1)
        self.assertTrue(torch.equal(constant_a, constant_b))
        self.assertTrue(torch.equal(group_a, group_b))
        self.assertEqual(group_meta_a["constant_global_action"], group_meta_b["constant_global_action"])

    def test_rules_depend_only_on_train_costs_for_action_choice(self):
        train = self._train_costs()
        test = self._test_costs(4.0)
        rank, meta = candidate_rank_rule(train, test)
        self.assertEqual(meta["rank"], int(train.q_pi0.mean(dim=0).argmin()))
        self.assertTrue(bool((rank == meta["rank"]).all()))
        constant, _group, group_meta = group_rules(train, test, min_group_states=1)
        # constant global action is the train-mean-argmin global action
        from experiments.r2_value import baseline_global_ids, policy_cost_for_global

        global_actions = train.global_action_ids.unique()
        costs = [policy_cost_for_global(train, int(action)).mean() for action in global_actions]
        expected = int(global_actions[torch.stack(costs).argmin()])
        self.assertEqual(group_meta["constant_global_action"], expected)
        self.assertEqual(constant.shape, test.baseline_action.shape)

    def test_uniform_expectation_is_analytic_column_mean(self):
        test = self._test_costs(0.0)
        test.q_pi0 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 8.0], [2.0, 2.0, 2.0, 2.0]])
        uniform = test.q_pi0.mean(dim=1)
        self.assertTrue(torch.allclose(uniform, torch.tensor([2.5, 2.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
