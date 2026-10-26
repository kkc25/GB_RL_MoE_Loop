import unittest

import torch

from eval.counterfactual import PathCosts
from experiments.followup_b_recompute import analytic_mix, selection_diagnostics


class FollowupBTests(unittest.TestCase):
    def test_analytic_mix_is_exact_per_state_expectation(self):
        stop = torch.tensor([1.0, 3.0])
        cont = torch.tensor([5.0, 7.0])
        self.assertTrue(torch.equal(analytic_mix(stop, cont, 0.0), stop))
        self.assertTrue(torch.equal(analytic_mix(stop, cont, 1.0), cont))
        self.assertTrue(torch.equal(analytic_mix(stop, cont, 0.25), torch.tensor([2.0, 4.0])))
        with self.assertRaises(ValueError):
            analytic_mix(stop, cont, 1.5)

    def test_selection_diagnostics_uses_actual_global_ids_and_fallback(self):
        costs = PathCosts(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            torch.tensor([0, 1]),
            torch.zeros(2, 3, 2, dtype=torch.long),
            4,
            torch.tensor([[10, 20, 21], [23, 10, 24]]),
        )
        constant = selection_diagnostics(costs, torch.tensor([1, 1]), 20)
        self.assertEqual(constant["actual_global_action_counts"], {"10": 1, "20": 1})
        self.assertEqual(constant["fallback_fraction"], 0.5)
        group = selection_diagnostics(costs, torch.tensor([0, 1]), torch.tensor([10, 10]))
        self.assertEqual(group["fallback_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
