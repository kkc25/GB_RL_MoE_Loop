import itertools
import unittest

import torch

from rl_lm.actions import action_table_identity, legal_mask, make_pair_table, pair_action_to_ids, pair_for_action, pair_index


class ActionTests(unittest.TestCase):
    def test_full_table_and_inverse_mapping(self):
        table = make_pair_table()
        expected = list(itertools.combinations(range(8), 2))
        self.assertEqual(table.tolist(), [list(pair) for pair in expected])
        for index, pair in enumerate(expected):
            self.assertEqual(pair_index(*pair), index)
            self.assertEqual(pair_action_to_ids(pair), index + 1)
            self.assertEqual(pair_for_action(index + 1, table).tolist(), list(pair))

    def test_required_examples(self):
        self.assertEqual(pair_action_to_ids((0, 1)), 1)
        self.assertEqual(pair_action_to_ids((0, 7)), 7)
        self.assertEqual(pair_action_to_ids((1, 2)), 8)
        self.assertEqual(pair_action_to_ids((6, 7)), 28)

    def test_legal_mask_only_stop_changes(self):
        mask = legal_mask(torch.tensor([False, True]))
        self.assertFalse(bool(mask[0, 0]))
        self.assertTrue(bool(mask[1, 0]))
        self.assertTrue(bool(mask[:, 1:].all()))

    def test_action_table_identity_is_stable(self):
        first = action_table_identity()
        self.assertEqual(first, action_table_identity())
        self.assertEqual(len(first["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
