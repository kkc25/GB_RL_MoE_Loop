import unittest

import torch

from rl_lm.actor import PairActor, greedy_actions, joint_log_probs
from rl_lm.selector import canonical_pair, pair_reference


class PolicyDistributionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.features = torch.randn(3, 1044)
        self.gate = torch.randn(3, 8)
        self.q = pair_reference(self.gate)
        self.actor = PairActor(1044, train_stop=False)

    def test_legal_distribution(self):
        first = self.actor(self.features, self.q, torch.tensor([False, False, False]))
        second = self.actor(self.features, self.q, torch.tensor([True, True, True]))
        self.assertTrue(torch.isneginf(first[:, 0]).all())
        self.assertTrue((second[:, 0].exp() > 0).all())
        for probs in (first, second):
            self.assertFalse(torch.isnan(probs).any())
            torch.testing.assert_close(probs.exp().sum(dim=-1), torch.ones(3), atol=1e-6, rtol=1e-6)

    def test_joint_greedy_stop_not_hazard_threshold(self):
        raw_pair = torch.zeros(1, 28)
        raw_stop = torch.logit(torch.tensor([0.4]))
        q = torch.full((1, 28), 1 / 28)
        log_probs = joint_log_probs(raw_pair, raw_stop, q, torch.tensor([True]))
        self.assertEqual(int(greedy_actions(log_probs)[0]), 0)

    def test_lowest_action_wins_ties(self):
        log_probs = torch.log(torch.full((1, 29), 1 / 29))
        self.assertEqual(int(greedy_actions(log_probs)[0]), 0)
        pair_only = torch.full((1, 29), -torch.inf)
        pair_only[:, 1:] = torch.log(torch.tensor(1 / 28))
        self.assertEqual(int(greedy_actions(pair_only)[0]), 1)

    def test_zero_initialization_matches_canonical_teacher(self):
        chosen = canonical_pair(self.gate)
        actions = greedy_actions(self.actor(self.features, self.q, torch.zeros(3, dtype=torch.bool)))
        self.assertTrue(torch.equal(actions, chosen + 1))
        self.assertFalse(self.actor.stop_head.weight.requires_grad)
        self.assertFalse(self.actor.stop_head.bias.requires_grad)

    def test_r2_warmstart_preserves_r1_policy_and_resets_stop(self):
        with torch.no_grad():
            self.actor.pair_head.weight.normal_(0, 0.01)
            self.actor.pair_head.bias.normal_(0, 0.01)
        r2 = PairActor(1044, train_stop=True)
        r2.trunk.load_state_dict(self.actor.trunk.state_dict())
        r2.pair_head.load_state_dict(self.actor.pair_head.state_dict())
        nn_stop_weight, nn_stop_bias = r2.stop_head.weight.clone(), r2.stop_head.bias.clone()
        self.assertTrue(torch.equal(nn_stop_weight, torch.zeros_like(nn_stop_weight)))
        self.assertTrue(torch.equal(nn_stop_bias, torch.zeros_like(nn_stop_bias)))
        stop_allowed = torch.zeros(3, dtype=torch.bool)
        expected = self.actor(self.features, self.q, stop_allowed)
        actual = r2(self.features, self.q, stop_allowed)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(r2.stop_head.weight.requires_grad)
        self.assertTrue(r2.stop_head.bias.requires_grad)


if __name__ == "__main__":
    unittest.main()
