import unittest

import torch

from rl_lm.types import ContextBatch, DecisionRecords, PolicyState, RolloutBatch


class TypeTests(unittest.TestCase):
    def test_valid_structures(self):
        context = ContextBatch(torch.randn(2, 4), torch.tensor([1, 2]), torch.randn(2), ("a", "b"))
        self.assertEqual(context.episode_ids, ("a", "b"))
        PolicyState(torch.randn(2, 4), torch.randn(2, 4), torch.zeros(2, dtype=torch.long), -torch.ones(2, 2, dtype=torch.long))
        DecisionRecords(torch.randn(2, 6), torch.rand(2, 28), torch.tensor([False, True]), torch.tensor([1, 0]), torch.tensor([0, 1]), torch.randn(2, 29))
        RolloutBatch(torch.randn(2, 4), torch.ones(2, 4, dtype=torch.long), torch.randn(2, 4), torch.randn(2, 4), 0)

    def test_bad_shape_dtype_and_padding_fail(self):
        with self.assertRaises(ValueError):
            ContextBatch(torch.randn(2, 4), torch.tensor([1]), torch.randn(2), ("a", "b"))
        with self.assertRaises(ValueError):
            ContextBatch(torch.randn(2, 4).requires_grad_(), torch.tensor([1, 2]), torch.randn(2), ("a", "b"))
        with self.assertRaises(ValueError):
            DecisionRecords(torch.randn(2, 6), torch.rand(2, 28), torch.tensor([False, True]), torch.tensor([-1, 0]), torch.tensor([0, 1]), torch.randn(2, 29))


if __name__ == "__main__":
    unittest.main()
