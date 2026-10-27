"""T05/T06: pair transition math and the STOP/limit state machine."""

from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from rl_lm.actions import pair_action_to_ids
from rl_lm.environment import LastFFNEnvironment
from rl_lm.executor import FrozenLastFFNExecutor, SparseMergedExecutor
from rl_lm.types import ContextBatch
from tests.rl_lm.tiny_lm import TINY_DIM, TINY_VOCAB, build_tiny_model


def manual_pair_step(model, h, pair):
    """Independent hand computation: norm -> gate -> pair softmax -> SwiGLU -> residual."""
    block = model.blocks[model.config.adaptive_layers[-1]]
    gamma = model.gammas[model.config.adaptive_layers[-1]]
    n = block.ffn_norm(h)
    z = F.linear(n, block.moe.gate.weight).float()
    alpha = F.softmax(z[:, list(pair)], dim=-1)
    outputs = []
    for expert in pair:
        hidden = F.silu(F.linear(n, block.moe.pool.w1[expert])) * F.linear(n, block.moe.pool.w3[expert])
        outputs.append(F.linear(hidden, block.moe.pool.w2[expert]).float())
    return h + gamma * (alpha[:, 0:1] * outputs[0] + alpha[:, 1:2] * outputs[1])


class ScriptActor:
    """Deterministic joint log-probs: all mass on scripted actions per slot."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = 0

    def __call__(self, features, pair_reference, stop_allowed):
        actions = self.scripts[self.calls]
        self.calls += 1
        if actions.shape[0] != features.shape[0]:
            raise ValueError("script does not cover every row")
        log_probs = torch.full((features.shape[0], 29), -torch.inf)
        log_probs[torch.arange(features.shape[0]), actions] = 0.0
        return log_probs


class TransitionMathTests(unittest.TestCase):
    """T05: executor transition equals hand computation and the top2 reference."""

    def setUp(self):
        self.model = build_tiny_model()
        torch.manual_seed(7)
        self.h = torch.randn(5, TINY_DIM)
        self.pair = (2, 5)
        self.action = int(pair_action_to_ids(self.pair))

    def _observe(self, executor, h, h0):
        return executor.observe(
            h,
            h0,
            torch.zeros(h.shape[0], dtype=torch.int64),
            torch.full((h.shape[0], 2), -1, dtype=torch.int64),
            max_calls=1,
        )

    def test_canonical_step_matches_hand_computation(self):
        executor = FrozenLastFFNExecutor(self.model, chunk=64, compute_dtype="float32")
        observation = self._observe(executor, self.h, self.h)
        proposed = executor.step_pair(self.h, observation, torch.full((5,), self.action, dtype=torch.int64))
        expected = manual_pair_step(self.model, self.h, self.pair)
        torch.testing.assert_close(proposed, expected, atol=1e-5, rtol=1e-5)

    def test_canonical_step_matches_top2_reference(self):
        from model.top2 import top2_dispatch

        executor = FrozenLastFFNExecutor(self.model, chunk=64, compute_dtype="float32")
        observation = self._observe(executor, self.h, self.h)
        proposed = executor.step_pair(self.h, observation, torch.full((5,), self.action, dtype=torch.int64))
        block = self.model.blocks[1]
        pairs = torch.tensor([list(self.pair)] * 5, dtype=torch.int64)
        update, weights, _ = top2_dispatch(block.moe.pool, observation.n, pairs=pairs, gate_logits=observation.z)
        gamma = self.model.gammas[1]
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(5))
        torch.testing.assert_close(proposed, self.h + gamma * update.float(), atol=1e-5, rtol=1e-5)

    def test_sparse_backend_matches_hand_computation(self):
        executor = SparseMergedExecutor(self.model, chunk=64, compute_dtype="float32")
        observation = self._observe(executor, self.h, self.h)
        proposed = executor.step_pair(self.h, observation, torch.full((5,), self.action, dtype=torch.int64))
        expected = manual_pair_step(self.model, self.h, self.pair)
        torch.testing.assert_close(proposed, expected, atol=1e-5, rtol=1e-5)

    def test_observation_feature_order(self):
        executor = FrozenLastFFNExecutor(self.model, chunk=64, compute_dtype="float32")
        torch.manual_seed(11)
        h0 = torch.randn(3, TINY_DIM)
        h = torch.cat((manual_pair_step(self.model, h0[:2], (1, 3)), h0[2:]))
        rounds = torch.tensor([1, 1, 0], dtype=torch.int64)
        previous = torch.tensor([[1, 3], [1, 3], [-1, -1]], dtype=torch.int64)
        observation = executor.observe(h, h0, rounds, previous, max_calls=2)
        d, e = TINY_DIM, 8
        features = observation.features
        self.assertEqual(features.shape, (3, 2 * d + 2 * e + 4))
        self.assertTrue(torch.equal(features[:, :d], observation.n))
        self.assertTrue(torch.equal(features[:, d:2 * d], observation.n0))
        torch.testing.assert_close(features[:, 2 * d:2 * d + e], observation.z - observation.z.mean(dim=-1, keepdim=True))
        multi = features[:, 2 * d + e:2 * d + 2 * e]
        self.assertTrue(torch.equal(multi[0], torch.tensor([0, 1, 0, 1, 0, 0, 0, 0]).float()))
        self.assertTrue(torch.equal(multi[2], torch.zeros(e)))
        scalars = features[:, -4:]
        torch.testing.assert_close(scalars[:2, 0], torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(scalars[:2, 1], torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(scalars[:, 2], torch.tensor([0.0, 0.0, 1.0]))
        expected_drift = torch.log1p((h - h0).norm(dim=-1) / (h0.norm(dim=-1) + 1e-8))
        torch.testing.assert_close(scalars[:, 3], expected_drift)
        self.assertTrue((observation.pair_reference > 0).all())

    def test_terminal_state_is_rejected(self):
        executor = FrozenLastFFNExecutor(self.model, chunk=64, compute_dtype="float32")
        with self.assertRaises(ValueError):
            executor.observe(
                self.h,
                self.h,
                torch.full((5,), 1, dtype=torch.int64),
                torch.full((5, 2), 0, dtype=torch.int64),
                max_calls=1,
            )


class StateMachineTests(unittest.TestCase):
    """T06: [pair,STOP] runs one round; [pair,pair] two rounds, no third decision."""

    def setUp(self):
        self.model = build_tiny_model()
        torch.manual_seed(13)
        self.h0 = torch.randn(2, TINY_DIM)
        self.targets = torch.randint(0, TINY_VOCAB, (2,))
        self.ref_ce = torch.rand(2) + 1.0

    def _contexts(self):
        return ContextBatch(
            h0=self.h0,
            targets=self.targets,
            ref_ce=self.ref_ce,
            episode_ids=("t06/0/0", "t06/0/1"),
        )

    def _rollout(self, scripts, max_calls=2):
        executor = FrozenLastFFNExecutor(self.model, chunk=64, compute_dtype="float32")
        environment = LastFFNEnvironment(self.model, max_calls=max_calls, executor=executor, head_chunk=64)
        actor = ScriptActor(scripts)
        uniforms = torch.zeros(2, 2, max_calls, dtype=torch.float64)
        return environment.rollout(self._contexts(), actor, uniforms), actor

    def test_pair_then_stop_vs_pair_pair(self):
        # flat 0: [3, STOP] -> 1 round; flat 1: [5, 9] -> 2 rounds
        # flat 2: [1, STOP] -> 1 round; flat 3: [2, 28] -> 2 rounds
        scripts = [
            torch.tensor([3, 5, 1, 2], dtype=torch.int64),
            torch.tensor([0, 9, 0, 28], dtype=torch.int64),
        ]
        (rollout, records), actor = self._rollout(scripts)
        self.assertEqual(actor.calls, 2, "exactly Rmax decision slots, no extra STOP probe")
        torch.testing.assert_close(rollout.rounds, torch.tensor([[1, 2], [1, 2]]))
        self.assertEqual(records.features.shape[0], 8, "2 decisions per episode, none after termination")
        per_episode = records.flat_episode_ids.bincount(minlength=4)
        torch.testing.assert_close(per_episode, torch.tensor([2, 2, 2, 2]))
        first_slot = records.stop_allowed[:4]
        second_slot = records.stop_allowed[4:]
        self.assertFalse(bool(first_slot.any()), "r=0 decisions must not allow STOP")
        self.assertTrue(bool(second_slot.all()), "r=1 decisions must allow STOP")
        torch.testing.assert_close(records.actions, torch.tensor([3, 5, 1, 2, 0, 9, 0, 28]))

        # Independently recomputed terminal states: [pair,STOP] gets exactly
        # one update; [pair,pair] gets two.  Compare through the production
        # readout (hidden states are internal to the environment).
        # flat_id = b*G+g, so flats 0/1 start from context 0 and flats 2/3
        # from context 1.  action a -> pair index a-1: 3->(0,3), 5->(0,5),
        # 9->(1,3), 1->(0,1), 2->(0,2), 28->(6,7).
        from rl_lm.readout import terminal_ce

        expected_flat0 = manual_pair_step(self.model, self.h0[:1], (0, 3))
        expected_flat1 = manual_pair_step(self.model, manual_pair_step(self.model, self.h0[:1], (0, 5)), (1, 3))
        expected_flat2 = manual_pair_step(self.model, self.h0[1:2], (0, 1))
        expected_flat3 = manual_pair_step(self.model, manual_pair_step(self.model, self.h0[1:2], (0, 2)), (6, 7))
        expected_h = torch.cat((expected_flat0, expected_flat1, expected_flat2, expected_flat3))
        expected_ce = terminal_ce(self.model, expected_h, self.targets.repeat_interleave(2), chunk=64)
        torch.testing.assert_close(rollout.terminal_ce.reshape(-1), expected_ce, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rollout.old_log_joint, torch.zeros(2, 2))
        torch.testing.assert_close(rollout.returns, self.ref_ce[:, None] - rollout.terminal_ce)

    def test_stopped_episode_hidden_not_touched_by_later_rounds(self):
        # flats 1/3 continue to round 2 while flats 0/2 STOP; the stopped
        # episodes must equal their exact one-step results (a second hidden
        # change would alter their CE).
        from rl_lm.readout import terminal_ce

        scripts = [
            torch.tensor([4, 6, 3, 5], dtype=torch.int64),
            torch.tensor([0, 12, 0, 9], dtype=torch.int64),
        ]
        (rollout, _), _actor = self._rollout(scripts)
        # action 4 -> (0,4); 6 -> (0,6); 12 -> (1,6); 3 -> (0,3); 5 -> (0,5); 9 -> (1,3)
        expected_flat0 = manual_pair_step(self.model, self.h0[:1], (0, 4))
        expected_flat1 = manual_pair_step(self.model, manual_pair_step(self.model, self.h0[:1], (0, 6)), (1, 6))
        expected_flat2 = manual_pair_step(self.model, self.h0[1:2], (0, 3))
        expected_flat3 = manual_pair_step(self.model, manual_pair_step(self.model, self.h0[1:2], (0, 5)), (1, 3))
        expected_h = torch.cat((expected_flat0, expected_flat1, expected_flat2, expected_flat3))
        expected_ce = terminal_ce(self.model, expected_h, self.targets.repeat_interleave(2), chunk=64)
        torch.testing.assert_close(rollout.terminal_ce.reshape(-1), expected_ce, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rollout.rounds, torch.tensor([[1, 2], [1, 2]]))

    def test_r1_environment_has_single_forced_pair_decision(self):
        scripts = [torch.tensor([7, 28, 1, 2], dtype=torch.int64)]
        executor = FrozenLastFFNExecutor(self.model, chunk=64, compute_dtype="float32")
        environment = LastFFNEnvironment(self.model, max_calls=1, executor=executor, head_chunk=64)
        rollout, records = environment.rollout(self._contexts(), ScriptActor(scripts), torch.zeros(2, 2, 1, dtype=torch.float64))
        torch.testing.assert_close(rollout.rounds, torch.ones(2, 2, dtype=torch.int64))
        self.assertEqual(records.features.shape[0], 4)
        self.assertFalse(bool(records.stop_allowed.any()))
        self.assertFalse(bool((records.actions == 0).any()), "R1 has no STOP decision")

    def test_illegal_stop_at_r0_fails_fast(self):
        scripts = [torch.tensor([0, 5, 1, 2], dtype=torch.int64)]
        executor = FrozenLastFFNExecutor(self.model, chunk=64, compute_dtype="float32")
        environment = LastFFNEnvironment(self.model, max_calls=1, executor=executor, head_chunk=64)
        with self.assertRaises(RuntimeError):
            environment.rollout(self._contexts(), ScriptActor(scripts), torch.zeros(2, 2, 1, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
