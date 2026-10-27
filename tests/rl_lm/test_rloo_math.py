"""T11-T14: RLOO advantage/loss math against independent exact references.

T11 sign & grouping, T12 zero signal, T13 exact gradient on an enumerated
variable-length toy tree MDP (spec appendix B: 2 experts, at most 3 expert
calls, 14 terminal paths, all 14^2 G=2 groups), T14 negative controls
(trajectory-length averaging and group-std whitening must NOT match the
exact gradient).

All tests are pure CPU math; the exact references come from enumeration and
hand derivation, not from the functions under test.
"""

from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from rl_lm.actor import PairActor
from rl_lm.objectives import (
    compute_returns,
    exact_path_objective,
    policy_gradient_loss,
    rloo_advantage,
)

# --- Toy tree MDP (spec appendix B) -----------------------------------------
# Round 0: two expert actions (STOP illegal).  Rounds 1 and 2: STOP plus the
# two experts.  A third expert call hits the call budget and terminates
# without a further decision.  Terminal paths: 2 + 4 + 8 = 14.
# Nodes: 0 = root (2 actions); 1..2 = after the first call (3 actions);
# 3..6 = after the second call (3 actions).

NODE_ACTIONS = (2, 3, 3, 3, 3, 3, 3)
THETA_SIZE = sum(NODE_ACTIONS)  # 20


def enumerate_paths():
    """All 14 terminal paths as tuples of (node, action) decisions."""
    paths = []
    for a in range(2):
        paths.append(((0, a), (1 + a, 0)))  # 1 call, then STOP
    for a in range(2):
        for b in range(2):
            paths.append(((0, a), (1 + a, 1 + b), (3 + 2 * a + b, 0)))  # 2 calls, STOP
    for a in range(2):
        for b in range(2):
            for c in range(2):
                paths.append(((0, a), (1 + a, 1 + b), (3 + 2 * a + b, 1 + c)))  # 3 calls, forced end
    return tuple(paths)


PATHS = enumerate_paths()
# Deterministic, non-monotone in depth: the best (0.1) and worst (2.5) paths
# are both deep, so "deep paths are uniformly bad" bugs cannot pass by luck.
PATH_COSTS = torch.tensor(
    [
        1.7, 0.4,  # 1 call
        0.9, 2.3, 0.2, 1.1,  # 2 calls
        1.9, 0.6, 2.5, 0.1, 1.3, 2.1, 0.8, 1.5,  # 3 calls
    ],
    dtype=torch.float64,
)


def node_logits(theta, node):
    start = sum(NODE_ACTIONS[:node])
    return theta[start : start + NODE_ACTIONS[node]]


def path_log_probs(theta):
    """Joint log-prob l_i of every terminal path; only real decisions count."""
    logs = []
    for path in PATHS:
        total = torch.zeros((), dtype=theta.dtype)
        for node, action in path:
            total = total + F.log_softmax(node_logits(theta, node), dim=0)[action]
        logs.append(total)
    return torch.stack(logs)


def fresh_theta():
    generator = torch.Generator().manual_seed(20261027)
    theta = (torch.randn(THETA_SIZE, generator=generator, dtype=torch.float64) * 0.4).requires_grad_(True)
    return theta


def expected_rloo_gradient(log_probs, costs, log_prob_transform=None, advantage_fn=None):
    """Sum over all 14^2 G=2 groups of detach(p_i*p_j) * PG-surrogate."""
    probs = log_probs.detach().exp()
    total = torch.zeros((), dtype=log_probs.dtype)
    for i in range(len(PATHS)):
        for j in range(len(PATHS)):
            li = log_probs[i] if log_prob_transform is None else log_prob_transform(log_probs[i], i)
            lj = log_probs[j] if log_prob_transform is None else log_prob_transform(log_probs[j], j)
            log_joint = torch.stack((li, lj)).reshape(1, 2)
            returns = torch.tensor([[-float(costs[i]), -float(costs[j])]], dtype=log_probs.dtype)
            advantage = rloo_advantage(returns) if advantage_fn is None else advantage_fn(returns)
            total = total + probs[i] * probs[j] * policy_gradient_loss(log_joint, advantage)
    total.backward()
    return total


def whitened_advantage(returns):
    """Wrong on purpose (T14): GRPO-style group-std whitening."""
    mean = returns.mean(dim=1, keepdim=True)
    std = returns.std(dim=1, keepdim=True, unbiased=True)
    return torch.where(std > 0, (returns - mean) / std.clamp_min(1e-12), torch.zeros_like(returns)).detach()


def decision_count(path_index):
    return len(PATHS[path_index])


class RewardShapeTests(unittest.TestCase):
    """Spec 9.1: R = ref_ce - ce - lambda * (rounds - 1)."""

    def test_reward_form_matches_hand_computation(self):
        ref_ce = torch.tensor([2.0, 1.0])
        terminal_ce = torch.tensor([1.5, 0.25])
        rounds = torch.tensor([1, 3])
        expected = torch.tensor([2.0 - 1.5 - 0.02 * 0, 1.0 - 0.25 - 0.02 * 2])
        result = compute_returns(ref_ce, terminal_ce, rounds, price_lambda=0.02)
        torch.testing.assert_close(result, expected, rtol=0, atol=1e-7)

    def test_reference_ce_cancels_in_leave_one_out(self):
        torch.manual_seed(5)
        returns = torch.randn(3, 4)
        shifted = returns + torch.tensor([1.25, -0.5, 3.0])[:, None]
        torch.testing.assert_close(rloo_advantage(returns), rloo_advantage(shifted), rtol=0, atol=1e-6)


class SignAndGroupingTests(unittest.TestCase):
    """T11: sign of the PG loss and strict per-context G-dim grouping."""

    def test_leave_one_out_values_by_hand(self):
        returns = torch.tensor([[0.0, 1.0, 2.0, 3.0], [5.0, 5.0, 2.0, 2.0]])
        # row 0: sum 6 -> baselines 2, 5/3, 4/3, 1; row 1: sum 14 -> 3, 3, 4, 4
        expected = torch.tensor(
            [[-2.0, 1.0 - 5.0 / 3.0, 2.0 - 4.0 / 3.0, 2.0], [2.0, 2.0, -2.0, -2.0]]
        )
        torch.testing.assert_close(rloo_advantage(returns), expected, rtol=0, atol=1e-6)

    def test_other_contexts_do_not_change_this_context(self):
        returns = torch.tensor([[0.0, 1.0, 2.0, 3.0], [5.0, 5.0, 2.0, 2.0]])
        modified = returns.clone()
        modified[1] = torch.tensor([-100.0, 0.0, 50.0, 7.0])
        before = rloo_advantage(returns)
        after = rloo_advantage(modified)
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertFalse(torch.equal(before[1], after[1]))

    def test_descent_increases_high_reward_action_probability(self):
        # Two-action tabular softmax, G=2 samples: action 0 got reward 2,
        # action 1 got reward 1.  Exact hand-derived SGD outcome.
        theta = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        actions = torch.tensor([0, 1])
        returns = torch.tensor([[2.0, 1.0]], dtype=torch.float64)
        advantage = rloo_advantage(returns)
        self.assertTrue(torch.equal(advantage, torch.tensor([[1.0, -1.0]], dtype=torch.float64)))
        log_joint = F.log_softmax(theta, dim=0)[actions].reshape(1, 2)
        loss = policy_gradient_loss(log_joint, advantage)
        loss.backward()
        # pi = [0.5, 0.5] -> d loss/d theta = [-0.5, +0.5]; lr 0.1 -> [0.05, -0.05]
        lr = 0.1
        new_theta = theta.detach() - lr * theta.grad
        expected = torch.tensor([0.05, -0.05], dtype=torch.float64)
        torch.testing.assert_close(new_theta, expected, rtol=0, atol=1e-15)
        probabilities = F.softmax(new_theta, dim=0)
        expected_p0 = torch.exp(expected[0]) / torch.exp(expected).sum()
        torch.testing.assert_close(probabilities[0], expected_p0, rtol=0, atol=1e-15)
        self.assertGreater(float(probabilities[0]), 0.5)
        self.assertLess(float(probabilities[1]), 0.5)

    def test_descent_direction_through_pair_actor(self):
        # Same state sampled twice (G=2): high-reward action 5, low-reward 9.
        feature_dim = 2 * 8 + 2 * 8 + 4  # small stand-in dimension
        actor = PairActor(feature_dim, train_stop=True)
        torch.manual_seed(11)
        feature = torch.randn(1, feature_dim)
        features = feature.repeat(2, 1)
        q = torch.full((2, 28), 1.0 / 28.0)
        stop_allowed = torch.zeros(2, dtype=torch.bool)  # r=0: STOP illegal
        actions = torch.tensor([5, 9])
        returns = torch.tensor([[2.0, 1.0]])
        advantage = rloo_advantage(returns)
        before = actor(features, q, stop_allowed).detach()
        log_joint = actor(features, q, stop_allowed).gather(1, actions[:, None]).squeeze(1).reshape(1, 2)
        loss = policy_gradient_loss(log_joint, advantage)
        actor.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for parameter in actor.parameters():
                if parameter.grad is not None:
                    parameter -= 0.05 * parameter.grad
        after = actor(features, q, stop_allowed).detach()
        self.assertGreater(float(after[0, 5]), float(before[0, 5]))
        self.assertLess(float(after[0, 9]), float(before[0, 9]))
        self.assertEqual(float(after[0, 0]), float("-inf"))


class ZeroSignalTests(unittest.TestCase):
    """T12: identical in-group returns give bitwise-zero advantage."""

    def test_constant_group_is_exactly_zero(self):
        returns = torch.full((1, 4), 1.7)
        advantage = rloo_advantage(returns)
        self.assertTrue(torch.equal(advantage, torch.zeros_like(advantage)))

    def test_constant_rows_zero_mixed_rows_untouched(self):
        returns = torch.tensor([[1.7, 1.7, 1.7, 1.7], [0.0, 1.0, 2.0, 3.0]])
        advantage = rloo_advantage(returns)
        self.assertTrue(torch.equal(advantage[0], torch.zeros(4)))
        self.assertFalse(bool((advantage[1] == 0).all()))

    def test_zero_advantage_gives_exactly_zero_gradient(self):
        theta = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        actions = torch.tensor([0, 1, 0, 1])
        log_joint = F.log_softmax(theta, dim=0)[actions].reshape(1, 4)
        advantage = rloo_advantage(torch.full((1, 4), 1.7, dtype=torch.float64))
        loss = policy_gradient_loss(log_joint, advantage)
        loss.backward()
        self.assertTrue(torch.equal(theta.grad, torch.zeros_like(theta.grad)))


class ExactGradientTests(unittest.TestCase):
    """T13: enumerated toy tree MDP; RLOO expectation equals exact gradient."""

    def test_rloo_matches_exact_gradient(self):
        theta = fresh_theta()
        log_probs = path_log_probs(theta)
        probs = log_probs.detach().exp()
        torch.testing.assert_close(probs.sum(), torch.ones((), dtype=torch.float64), rtol=0, atol=1e-12)

        objective = exact_path_objective(log_probs, PATH_COSTS)
        exact_grad, = torch.autograd.grad(objective, theta, retain_graph=True)

        expected_rloo_gradient(log_probs, PATH_COSTS)
        rloo_grad = theta.grad
        max_error = float((rloo_grad - exact_grad).abs().max())
        exact_norm = float(exact_grad.norm())
        print(f"\nT13 exact |grad|={exact_norm:.6e} max|rloo-exact|={max_error:.3e}")
        self.assertLess(max_error, 1e-10)
        self.assertGreater(exact_norm, 1e-3)  # the signal is genuinely nonzero


class WrongObjectiveControlTests(unittest.TestCase):
    """T14: length-averaged and std-whitened variants must mismatch exact."""

    def _wrong_gradient(self, log_prob_transform=None, advantage_fn=None):
        theta = fresh_theta()
        log_probs = path_log_probs(theta)
        objective = exact_path_objective(log_probs, PATH_COSTS)
        exact_grad, = torch.autograd.grad(objective, theta, retain_graph=True)
        expected_rloo_gradient(log_probs, PATH_COSTS, log_prob_transform, advantage_fn)
        return exact_grad, theta.grad

    def test_length_averaged_log_prob_is_wrong(self):
        exact_grad, wrong_grad = self._wrong_gradient(
            log_prob_transform=lambda lp, i: lp / decision_count(i)
        )
        error = float((wrong_grad - exact_grad).norm())
        print(f"\nT14 length-averaged L2 error={error:.6e}")
        self.assertGreater(error, 1e-6)

    def test_group_std_whitening_is_wrong(self):
        exact_grad, wrong_grad = self._wrong_gradient(advantage_fn=whitened_advantage)
        error = float((wrong_grad - exact_grad).norm())
        print(f"\nT14 std-whitened L2 error={error:.6e}")
        self.assertGreater(error, 1e-6)


if __name__ == "__main__":
    unittest.main()
