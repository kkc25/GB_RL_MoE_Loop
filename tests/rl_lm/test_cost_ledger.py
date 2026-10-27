import unittest

import torch

from rl_lm.executor import FrozenLastFFNExecutor
from rl_lm.ledger import RLLedger
from rl_lm.readout import terminal_ce
from tests.rl_lm.tiny_lm import TINY_DIM, build_tiny_model


class CostLedgerTests(unittest.TestCase):
    def test_logical_and_physical_counts_are_separate(self):
        model = build_tiny_model()
        ledger = RLLedger()
        executor = FrozenLastFFNExecutor(model, chunk=3, compute_dtype="float32", ledger=ledger)
        torch.manual_seed(5)
        h = torch.randn(4, TINY_DIM)
        observation = executor.observe(h, h.clone(), torch.zeros(4, dtype=torch.int64), -torch.ones(4, 2, dtype=torch.int64), max_calls=1)
        executor.step_pair(h, observation, torch.tensor([1, 2, 3, 4], dtype=torch.int64))
        ledger.logical_pair_rounds = 4
        ledger.logical_expert_rows = 8
        ledger.actor_decision_rows = 4
        ledger.rollout_episodes = 4
        terminal_ce(model, h, torch.randint(0, 64, (4,)), chunk=3, ledger=ledger)
        self.assertEqual(ledger.physical_expert_rows, 8 * 3 * 2)
        self.assertEqual(ledger.physical_head_rows, 3 * 2)
        self.assertEqual(ledger.logical_expert_rows, 2 * ledger.logical_pair_rounds)
        self.assertGreater(ledger.physical_expert_rows, ledger.logical_expert_rows)


if __name__ == "__main__":
    unittest.main()
