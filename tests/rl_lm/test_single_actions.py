"""Single-branch action numbering, tie rules, prior and joint-9 distribution."""

import unittest

import torch

from rl_lm.single_actor import SingleActor, single_joint_log_probs
from rl_lm.single_executor import (
    JOINT_SINGLE_ACTIONS,
    SINGLE_ACTIONS,
    STOP_ACTION,
    expert_for_action,
    single_argmax_expert,
    single_greedy_lowest_id,
    single_reference,
)


class SingleNumberingTests(unittest.TestCase):
    def test_action_zero_is_stop(self):
        self.assertEqual(STOP_ACTION, 0)
        with self.assertRaises(ValueError):
            expert_for_action(0)

    def test_actions_1_to_8_map_to_experts_0_to_7(self):
        for action in range(1, 9):
            self.assertEqual(expert_for_action(action), action - 1)
        with self.assertRaises(ValueError):
            expert_for_action(9)
        with self.assertRaises(ValueError):
            expert_for_action(-1)

    def test_joint_action_count(self):
        self.assertEqual(SINGLE_ACTIONS, 8)
        self.assertEqual(JOINT_SINGLE_ACTIONS, 9)


class SingleArgmaxTieTests(unittest.TestCase):
    def test_argmax_gate(self):
        z = torch.tensor([[0.1, 0.5, 0.2, 0.0, -1.0, 0.3, 0.4, -0.2]])
        self.assertEqual(int(single_argmax_expert(z)[0]), 1)

    def test_tie_goes_to_lowest_id(self):
        z = torch.zeros(1, 8)
        z[0, 5] = 1.0
        z[0, 2] = 1.0
        self.assertEqual(int(single_argmax_expert(z)[0]), 2)
        z = torch.zeros(3, 8)
        self.assertTrue(bool((single_argmax_expert(z) == 0).all()))


class SingleReferenceTests(unittest.TestCase):
    def test_prior_is_090_argmax_plus_010_over_8(self):
        z = torch.tensor([[0.0, 1.0, -1.0, 0.5, 0.2, 0.3, -0.4, 0.1]])
        q = single_reference(z)
        self.assertEqual(tuple(q.shape), (1, 8))
        self.assertAlmostEqual(float(q.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(q[0, 1]), 0.9 + 0.1 / 8, places=6)
        for expert in (0, 2, 3, 4, 5, 6, 7):
            self.assertAlmostEqual(float(q[0, expert]), 0.1 / 8, places=6)

    def test_epsilon_fixed(self):
        z = torch.zeros(1, 8)
        with self.assertRaises(ValueError):
            single_reference(z, epsilon=0.2)


class SingleJointDistributionTests(unittest.TestCase):
    def test_nine_class_probabilities_sum_to_one(self):
        torch.manual_seed(0)
        raw_expert = torch.randn(5, 8)
        raw_stop = torch.randn(5)
        reference = single_reference(torch.randn(5, 8))
        stop_allowed = torch.tensor([True, False, True, False, True])
        log_probs = single_joint_log_probs(raw_expert, raw_stop, reference, stop_allowed)
        self.assertEqual(tuple(log_probs.shape), (5, 9))
        probs = log_probs.exp()
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(5), atol=1e-6))
        self.assertTrue(bool(torch.isneginf(log_probs[~stop_allowed, 0]).all()))
        self.assertTrue(bool((probs[~stop_allowed, 0] == 0).all()))

    def test_zero_actor_logits_reproduce_prior_times_continue(self):
        reference = single_reference(torch.randn(4, 8))
        stop_allowed = torch.zeros(4, dtype=torch.bool)
        log_probs = single_joint_log_probs(torch.zeros(4, 8), torch.zeros(4), reference, stop_allowed)
        self.assertTrue(torch.allclose(log_probs[:, 1:], reference.log(), atol=1e-6))
        stop_allowed_on = torch.ones(4, dtype=torch.bool)
        log_probs_on = single_joint_log_probs(torch.zeros(4, 8), torch.zeros(4), reference, stop_allowed_on)
        expected = torch.log(0.5 * reference)
        self.assertTrue(torch.allclose(log_probs_on[:, 1:], expected, atol=1e-6))
        self.assertTrue(torch.allclose(log_probs_on[:, 0], torch.full((4,), torch.log(torch.tensor(0.5))), atol=1e-6))

    def test_greedy_lowest_id_tie(self):
        log_probs = torch.zeros(2, 9)
        self.assertTrue(bool((single_greedy_lowest_id(log_probs) == 0).all()))
        log_probs[0, 3] = 0.1
        log_probs[0, 5] = 0.1
        self.assertEqual(int(single_greedy_lowest_id(log_probs)[0]), 3)


class SingleActorTests(unittest.TestCase):
    def test_actor_output_shape_and_determinism(self):
        torch.manual_seed(17)
        actor = SingleActor(2 * 32 + 2 * 8 + 4)
        features = torch.randn(6, actor.feature_dim)
        reference = single_reference(torch.randn(6, 8))
        stop_allowed = torch.ones(6, dtype=torch.bool)
        first = actor(features, reference, stop_allowed)
        second = actor(features, reference, stop_allowed)
        self.assertEqual(tuple(first.shape), (6, 9))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.allclose(first.exp().sum(-1), torch.ones(6), atol=1e-6))

    def test_stop_head_frozen_by_default_in_r1(self):
        actor = SingleActor(2 * 32 + 2 * 8 + 4, train_stop=False)
        self.assertFalse(actor.stop_head.weight.requires_grad)
        self.assertTrue(actor.expert_head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
