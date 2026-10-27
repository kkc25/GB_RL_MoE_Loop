"""SingleExpertExecutor contracts: amplitude 1, explicit expert id, backends."""

import unittest

import torch
from torch.nn import functional as F

from rl_lm.single_executor import (
    SINGLE_CANONICAL_BACKEND,
    SINGLE_SPARSE_BACKEND,
    SingleLedger,
    make_single_executor,
    single_argmax_expert,
    single_ffn_update,
)
from tests.rl_lm.tiny_lm import build_tiny_model


def _observation(executor, h, h0):
    rounds = torch.zeros(h.shape[0], dtype=torch.int64)
    previous = torch.full((h.shape[0],), -1, dtype=torch.int64)
    return executor.observe(h, h0, rounds, previous, max_calls=1)


class SingleExecutorAmplitudeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = build_tiny_model()
        self.layer = self.model.config.layers - 1
        self.block = self.model.blocks[self.layer]
        self.gamma = float(self.model.gammas[self.layer])
        self.h = torch.randn(10, 32)
        self.h0 = torch.randn(10, 32)

    def test_amplitude_is_exactly_one(self):
        executor = make_single_executor(self.model, SINGLE_SPARSE_BACKEND, compute_dtype="float32")
        observation = _observation(executor, self.h, self.h0)
        actions = torch.arange(1, 11) % 8 + 1
        out = executor.step_single(self.h, observation, actions)
        n = self.block.ffn_norm(self.h).float()
        for row in range(10):
            expert = int(actions[row]) - 1
            x = n[row]
            hidden = F.silu(F.linear(x, self.block.moe.pool.w1[expert])) * F.linear(x, self.block.moe.pool.w3[expert])
            manual = self.h[row] + self.gamma * F.linear(hidden, self.block.moe.pool.w2[expert]).float()
            torch.testing.assert_close(out[row], manual, atol=1e-6, rtol=1e-6)

    def test_not_the_legacy_amplitude_path(self):
        """The single executor must not call MoE.forward nor multiply gate probs."""
        executor = make_single_executor(self.model, SINGLE_SPARSE_BACKEND, compute_dtype="float32")
        observation = _observation(executor, self.h, self.h0)
        actions = torch.full((10,), 3, dtype=torch.int64)

        def forbidden(*args, **kwargs):
            raise AssertionError("legacy MoE.forward amplitude path was called")

        original = self.block.moe.forward
        self.block.moe.forward = forbidden
        try:
            out = executor.step_single(self.h, observation, actions)
        finally:
            self.block.moe.forward = original
        n = self.block.ffn_norm(self.h).float()
        with torch.no_grad():
            _, _, probabilities, _ = original(n, torch.full((10,), 2, dtype=torch.int64))
        amplitude = probabilities[:, 2].float()
        x = n
        hidden = F.silu(F.linear(x, self.block.moe.pool.w1[2])) * F.linear(x, self.block.moe.pool.w3[2])
        legacy = self.h + self.gamma * (F.linear(hidden, self.block.moe.pool.w2[2]).float() * amplitude[:, None])
        self.assertFalse(torch.allclose(out, legacy, atol=1e-4))

    def test_canonical_and_sparse_backends_agree(self):
        canonical = make_single_executor(self.model, SINGLE_CANONICAL_BACKEND, chunk=4, compute_dtype="float32")
        sparse = make_single_executor(self.model, SINGLE_SPARSE_BACKEND, compute_dtype="float32")
        observation_c = _observation(canonical, self.h, self.h0)
        observation_s = _observation(sparse, self.h, self.h0)
        actions = torch.arange(1, 11) % 8 + 1
        out_c = canonical.step_single(self.h, observation_c, actions)
        out_s = sparse.step_single(self.h, observation_s, actions)
        torch.testing.assert_close(out_c, out_s, atol=1e-6, rtol=1e-6)

    def test_physical_logical_ratio_eight_for_canonical(self):
        ledger = SingleLedger()
        canonical = make_single_executor(self.model, SINGLE_CANONICAL_BACKEND, chunk=4, compute_dtype="float32", ledger=ledger)
        observation = _observation(canonical, self.h, self.h0)
        actions = torch.ones(10, dtype=torch.int64)
        canonical.step_single(self.h, observation, actions)
        # 10 rows in chunks of 4 -> 3 chunks x 4 rows x 8 experts.
        self.assertEqual(ledger.physical_expert_rows, 8 * 4 * 3)
        ledger_sparse = SingleLedger()
        sparse = make_single_executor(self.model, SINGLE_SPARSE_BACKEND, compute_dtype="float32", ledger=ledger_sparse)
        sparse.step_single(self.h, _observation(sparse, self.h, self.h0), actions)
        self.assertEqual(ledger_sparse.physical_expert_rows, 10)

    def test_explicit_expert_id_validation(self):
        executor = make_single_executor(self.model, SINGLE_SPARSE_BACKEND, compute_dtype="float32")
        observation = _observation(executor, self.h, self.h0)
        with self.assertRaises(ValueError):
            executor.step_single(self.h, observation, torch.zeros(10, dtype=torch.int64))
        with self.assertRaises(ValueError):
            executor.step_single(self.h, observation, torch.full((10,), 9, dtype=torch.int64))

    def test_only_terminal_layer_is_controllable(self):
        with self.assertRaises(ValueError):
            make_single_executor(self.model, SINGLE_SPARSE_BACKEND, layer=0, compute_dtype="float32")

    def test_front_layers_untouched(self):
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        executor = make_single_executor(self.model, SINGLE_CANONICAL_BACKEND, compute_dtype="float32")
        observation = _observation(executor, self.h, self.h0)
        executor.step_single(self.h, observation, torch.full((10,), 5, dtype=torch.int64))
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter.detach(), before[name]), name)


class SingleObservationTests(unittest.TestCase):
    def test_previous_expert_one_hot_feature(self):
        torch.manual_seed(3)
        model = build_tiny_model()
        executor = make_single_executor(model, SINGLE_SPARSE_BACKEND, compute_dtype="float32")
        h = torch.randn(4, 32)
        h0 = torch.randn(4, 32)
        rounds = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
        previous = torch.tensor([-1, 3, 0, -1], dtype=torch.int64)
        observation = executor.observe(h, h0, rounds, previous, max_calls=2)
        dim = 32
        one_hot = observation.features[:, 2 * dim + 8:2 * dim + 16]
        self.assertTrue(bool((one_hot[0] == 0).all()))
        self.assertTrue(bool((one_hot[3] == 0).all()))
        self.assertEqual(float(one_hot[1, 3]), 1.0)
        self.assertEqual(float(one_hot[1].sum()), 1.0)
        self.assertEqual(float(one_hot[2, 0]), 1.0)

    def test_round_zero_requires_no_previous(self):
        model = build_tiny_model()
        executor = make_single_executor(model, SINGLE_SPARSE_BACKEND, compute_dtype="float32")
        h = torch.randn(2, 32)
        h0 = torch.randn(2, 32)
        with self.assertRaises(ValueError):
            executor.observe(h, h0, torch.zeros(2, dtype=torch.int64), torch.tensor([0, -1]), max_calls=1)


class SingleFFNUpdateGradTests(unittest.TestCase):
    def test_gradient_flows_to_selected_expert_slices(self):
        torch.manual_seed(7)
        model = build_tiny_model()
        pool = model.blocks[model.config.layers - 1].moe.pool
        for weight in (pool.w1, pool.w2, pool.w3):
            weight.requires_grad_(True)
        n = torch.randn(6, 32)
        experts = torch.tensor([0, 1, 1, 3, 3, 3])
        update = single_ffn_update(pool, n, experts)
        update.square().sum().backward()
        for expert in (0, 1, 3):
            self.assertGreater(float(pool.w1.grad[expert].abs().sum()), 0.0)
        for expert in (2, 4, 5, 6, 7):
            self.assertEqual(float(pool.w1.grad[expert].abs().sum()), 0.0)
            self.assertEqual(float(pool.w2.grad[expert].abs().sum()), 0.0)
            self.assertEqual(float(pool.w3.grad[expert].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
