import unittest

import torch

from eval.counterfactual import PathCosts, evaluate_policies
from experiments.r2_value import (
    baseline_global_ids,
    candidate_rank_rule,
    group_rules,
    local_action_or_baseline,
)


def costs(global_ids, baseline, q=None):
    global_ids = torch.as_tensor(global_ids, dtype=torch.long)
    baseline = torch.as_tensor(baseline, dtype=torch.long)
    n, a = global_ids.shape
    q = torch.arange(n * a, dtype=torch.float32).reshape(n, a) if q is None else torch.as_tensor(q, dtype=torch.float32)
    pairs = torch.zeros(n, a, 2, dtype=torch.long)
    return PathCosts(q.clone(), q, baseline, pairs, 4, global_ids)


class ActionMappingTests(unittest.TestCase):
    def test_global_action_uses_each_rows_own_candidate_columns(self):
        ids = torch.tensor([[10, 20, 21, 22], [23, 10, 24, 25]])
        selection = local_action_or_baseline(ids, 10, torch.tensor([0, 0]))
        self.assertEqual(selection.tolist(), [0, 1])
        missing = local_action_or_baseline(ids, 999, torch.tensor([0, 1]))
        self.assertEqual(missing.tolist(), [0, 1])

    def test_rules_are_row_permutation_and_split_batch_invariant(self):
        train = costs(
            [[10, 20, 21, 22]] * 4,
            [0, 0, 0, 0],
            [[5.0, 1.0, 3.0, 4.0]] * 4,
        )
        test = costs([[10, 20, 21, 22], [23, 10, 24, 25]], [0, 1])
        constant, group, _ = group_rules(train, test, min_group_states=1)
        self.assertEqual(constant.tolist(), [1, 1])
        self.assertEqual(group.tolist(), [1, 1])
        permutation = torch.tensor([1, 0])
        shuffled = costs(test.global_action_ids[permutation], test.baseline_action[permutation], test.q_pi0[permutation])
        shuffled_constant, shuffled_group, _ = group_rules(train, shuffled, min_group_states=1)
        self.assertEqual(shuffled_constant.tolist(), constant[permutation].tolist())
        self.assertEqual(shuffled_group.tolist(), group[permutation].tolist())
        for row in range(2):
            single = costs(test.global_action_ids[row : row + 1], test.baseline_action[row : row + 1], test.q_pi0[row : row + 1])
            one_constant, one_group, _ = group_rules(train, single, min_group_states=1)
            self.assertEqual(one_constant[0], constant[row])
            self.assertEqual(one_group[0], group[row])

    def test_candidate_column_permutation_preserves_executed_global_action(self):
        train = costs([[10, 20, 21, 22]] * 2, [0, 0], [[5.0, 1.0, 3.0, 4.0]] * 2)
        test = costs([[10, 20, 21, 22]], [0])
        expected_constant, expected_group, _ = group_rules(train, test, min_group_states=1)
        order = torch.tensor([2, 0, 3, 1])
        reordered = costs(test.global_action_ids[:, order], test.baseline_action, test.q_pi0[:, order])
        constant, group, _ = group_rules(train, reordered, min_group_states=1)
        baseline = baseline_global_ids(test)
        expected = test.global_action_ids.gather(1, expected_group.unsqueeze(1))
        actual = reordered.global_action_ids.gather(1, group.unsqueeze(1))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(baseline.item(), 10)

    def test_sparse_group_falls_back_to_global_rule(self):
        train = costs([[10, 20, 21, 22]] * 3, [0, 1, 1], [[5.0, 1.0, 3.0, 4.0]] * 3)
        test = costs([[10, 20, 21, 22]], [0])
        constant, group, metadata = group_rules(train, test, min_group_states=2)
        self.assertEqual(group.tolist(), constant.tolist())
        self.assertEqual(metadata["group_train_support"], {"10": 1})

    def test_candidate_rank_is_selected_on_train_only(self):
        train = costs([[10, 20], [20, 10]], [0, 1], [[4.0, 1.0], [3.0, 4.0]])
        test = costs([[20, 10]], [0])
        selection, metadata = candidate_rank_rule(train, test)
        self.assertEqual(selection.tolist(), [1])
        self.assertEqual(metadata, {"rank": 1})

    def test_evaluation_rejects_invalid_selection_indices(self):
        test = costs([[10, 20]], [0])
        for invalid in (torch.tensor([-1]), torch.tensor([2])):
            with self.assertRaises(ValueError):
                evaluate_policies(test, {"invalid": invalid})


if __name__ == "__main__":
    unittest.main()
