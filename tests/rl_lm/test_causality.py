import unittest

import torch

from rl_lm.actor import PairActor
from rl_lm.environment import LastFFNEnvironment
from rl_lm.executor import FrozenLastFFNExecutor
from rl_lm.types import ContextBatch
from tests.rl_lm.tiny_lm import TINY_DIM, TINY_VOCAB, build_tiny_model


class CausalityTests(unittest.TestCase):
    def contexts(self, targets):
        torch.manual_seed(19)
        return ContextBatch(
            h0=torch.randn(2, TINY_DIM),
            targets=targets,
            ref_ce=torch.rand(2),
            episode_ids=("toy:seq-a:pos-a", "toy:seq-b:pos-b"),
        )

    def test_permuting_targets_cannot_change_actions_or_hidden(self):
        model = build_tiny_model()
        executor = FrozenLastFFNExecutor(model, chunk=64, compute_dtype="float32")
        environment = LastFFNEnvironment(model, max_calls=1, executor=executor, head_chunk=64)
        torch.manual_seed(17)
        actor = PairActor(2 * TINY_DIM + 2 * 8 + 4, train_stop=False)
        context_a = self.contexts(torch.tensor([1, 2]))
        context_b = self.contexts(torch.tensor([2, 1]))
        uniforms = torch.zeros(2, 2, 1, dtype=torch.float64)
        rollout_a, records_a = environment.rollout(context_a, actor, uniforms)
        rollout_b, records_b = environment.rollout(context_b, actor, uniforms)
        self.assertTrue(torch.equal(records_a.actions, records_b.actions))
        self.assertTrue(torch.equal(rollout_a.rounds, rollout_b.rounds))
        # Only the final CE may change because the labels changed.
        self.assertFalse(torch.allclose(rollout_a.terminal_ce, rollout_b.terminal_ce))

    def test_target_values_do_not_enter_episode_ids(self):
        context = self.contexts(torch.tensor([1, 2]))
        self.assertNotEqual(context.episode_ids[0], context.episode_ids[1])
        self.assertNotIn(str(int(context.targets[0])), context.episode_ids[0])
        self.assertNotIn(str(int(context.targets[1])), context.episode_ids[1])


if __name__ == "__main__":
    unittest.main()
