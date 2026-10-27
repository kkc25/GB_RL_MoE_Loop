import unittest

import torch

from rl_lm.executor import FrozenLastFFNExecutor
from tests.rl_lm.tiny_lm import TINY_DIM, build_tiny_model


class NumericalContractTests(unittest.TestCase):
    def test_canonical_row_is_independent_of_other_episode_actions(self):
        model = build_tiny_model()
        executor = FrozenLastFFNExecutor(model, chunk=3, compute_dtype="float32")
        torch.manual_seed(11)
        h = torch.randn(5, TINY_DIM)
        rounds = torch.zeros(5, dtype=torch.int64)
        previous = -torch.ones(5, 2, dtype=torch.int64)
        observation = executor.observe(h, h.clone(), rounds, previous, max_calls=1)
        actions_a = torch.tensor([1, 7, 1, 8, 1], dtype=torch.int64)
        actions_b = torch.tensor([1, 2, 1, 3, 1], dtype=torch.int64)
        proposed_a = executor.step_pair(h, observation, actions_a)
        proposed_b = executor.step_pair(h, observation, actions_b)
        # Rows 0, 2 and 4 select the same action while other rows change.
        for row in (0, 2, 4):
            torch.testing.assert_close(proposed_a[row], proposed_b[row])

    def test_fixed_shape_backend_computes_proposal_for_every_row(self):
        model = build_tiny_model()
        executor = FrozenLastFFNExecutor(model, chunk=2, compute_dtype="float32")
        h = torch.randn(3, TINY_DIM)
        observation = executor.observe(h, h.clone(), torch.zeros(3, dtype=torch.int64), -torch.ones(3, 2, dtype=torch.int64), max_calls=1)
        proposed = executor.step_pair(h, observation, torch.ones(3, dtype=torch.int64))
        self.assertEqual(proposed.shape, h.shape)
        self.assertFalse(torch.allclose(proposed[1], h[1]))


if __name__ == "__main__":
    unittest.main()
