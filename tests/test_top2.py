import itertools
import os
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import top2
from model.moe import ExpertPool, MoE


def ref_expert(pool, expert, x):
    hidden = F.silu(F.linear(x, pool.w1[expert])) * F.linear(x, pool.w3[expert])
    return F.linear(hidden, pool.w2[expert])


def ref_top2_update(pool, x, pairs, weights):
    flat_x = x.reshape(-1, x.shape[-1])
    flat_pairs = pairs.reshape(-1, 2)
    flat_weights = weights.reshape(-1, 2).float()
    out = torch.zeros(flat_x.shape, dtype=torch.float32)
    for expert in range(pool.experts):
        for slot in (0, 1):
            mask = flat_pairs[:, slot] == expert
            if not mask.any():
                continue
            selected = ref_expert(pool, expert, flat_x[mask]).float()
            out[mask] += flat_weights[mask, slot].unsqueeze(-1) * selected
    return out.to(x.dtype).reshape(x.shape)


class TestActionTable(unittest.TestCase):
    def test_enumerates_all_28_unordered_pairs(self):
        table = top2.enumerate_top2_actions(8)
        self.assertEqual(tuple(table.shape), (28, 2))
        self.assertEqual(top2.num_top2_actions(8), 28)
        expected = list(itertools.combinations(range(8), 2))
        self.assertEqual(table.tolist(), [list(p) for p in expected])
        as_sets = {frozenset(p) for p in table.tolist()}
        self.assertEqual(len(as_sets), 28)

    def test_action_pair_roundtrip(self):
        actions = torch.arange(28)
        pairs = top2.action_to_pair(actions, experts=8)
        self.assertTrue(torch.equal(top2.pair_to_action(pairs, experts=8), actions))

    def test_pair_order_does_not_matter(self):
        pairs = torch.tensor([[3, 5], [5, 3], [0, 7]])
        actions = top2.pair_to_action(pairs, experts=8)
        self.assertEqual(actions[0].item(), actions[1].item())
        self.assertEqual(actions[2].item(), 6)

    def test_invalid_pair_rejected(self):
        with self.assertRaises(ValueError):
            top2.pair_to_action(torch.tensor([[4, 4]]), experts=8)
        with self.assertRaises(ValueError):
            top2.pair_to_action(torch.tensor([[1, 8]]), experts=8)


class TestWeights(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(3, 4, 8)

    def test_router_weights_are_renormalized_softmax(self):
        pairs = top2.action_to_pair(torch.randint(0, 28, (3, 4)))
        weights = top2.top2_weights(self.logits, pairs)
        gathered = self.logits.gather(-1, pairs)
        expected = F.softmax(gathered, dim=-1)
        self.assertEqual(weights.dtype, torch.float32)
        self.assertTrue(torch.allclose(weights, expected, atol=1e-7))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-7))

    def test_router_weights_do_not_backprop_into_frozen_router(self):
        logits = self.logits.clone().requires_grad_(True)
        pairs = top2.action_to_pair(torch.randint(0, 28, (3, 4)))
        weights = top2.top2_weights(logits, pairs)
        self.assertFalse(weights.requires_grad)
        self.assertIsNone(weights.grad_fn)

    def test_uniform_weights(self):
        pairs = top2.action_to_pair(torch.randint(0, 28, (3, 4)))
        weights = top2.top2_weights(None, pairs, rule=top2.UNIFORM_WEIGHTS)
        self.assertTrue(torch.equal(weights, torch.full_like(weights, 0.5)))

    def test_unknown_rule_rejected(self):
        with self.assertRaises(ValueError):
            top2.top2_weights(self.logits, torch.zeros(2, 2, dtype=torch.long), rule="bogus")


class TestDispatch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.pool = ExpertPool(dim=8, intermediate=16, experts=8)
        self.x = torch.randn(2, 5, 8)
        self.logits = torch.randn(2, 5, 8)

    def test_dropless_dispatch_matches_manual_reference(self):
        actions = torch.randint(0, 28, (2, 5))
        update, weights, pairs = top2.top2_dispatch(self.pool, self.x, actions=actions, gate_logits=self.logits)
        expected = ref_top2_update(self.pool, self.x, pairs, weights)
        self.assertTrue(torch.allclose(update, expected, atol=1e-6))

    def test_every_token_gets_both_experts(self):
        actions = torch.full((2, 5), 3)
        update, weights, pairs = top2.top2_dispatch(self.pool, self.x, actions=actions, gate_logits=self.logits)
        e0, e1 = pairs[0, 0].tolist()
        for expert, slot in ((e0, 0), (e1, 1)):
            with torch.no_grad():
                self.pool.w2[expert].mul_(1.5)
                changed, _, _ = top2.top2_dispatch(self.pool, self.x, actions=actions, gate_logits=self.logits)
                self.pool.w2[expert].div_(1.5)
            self.assertFalse(torch.allclose(update, changed, atol=1e-6), f"expert {expert} slot {slot} had no effect")

    def test_unselected_experts_receive_zero_gradient(self):
        self.pool.zero_grad()
        actions = torch.full((2, 5), 0)  # only experts 0 and 1
        update, _, _ = top2.top2_dispatch(self.pool, self.x, actions=actions, gate_logits=self.logits)
        update.sum().backward()
        for expert in range(2, 8):
            self.assertEqual(self.pool.w1.grad[expert].abs().max().item(), 0.0)
            self.assertEqual(self.pool.w2.grad[expert].abs().max().item(), 0.0)
            self.assertEqual(self.pool.w3.grad[expert].abs().max().item(), 0.0)

    def test_forced_baseline_action_matches_baseline(self):
        gate = torch.nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            gate.weight.normal_(std=0.5)
        logits = gate(self.x)
        top = logits.topk(2, dim=-1).indices
        base_weights = F.softmax(logits.gather(-1, top), dim=-1)
        baseline = ref_top2_update(self.pool, self.x, top, base_weights)

        forced = top2.pair_to_action(top.sort(dim=-1).values, experts=8)
        update, _, _ = top2.top2_dispatch(self.pool, self.x, actions=forced, gate_logits=logits)
        self.assertTrue(torch.allclose(update, baseline, atol=1e-6))

        swapped = top[..., [1, 0]]
        update_swapped, _, _ = top2.top2_dispatch(self.pool, self.x, pairs=swapped, gate_logits=logits)
        self.assertTrue(torch.allclose(update_swapped, baseline, atol=1e-6))

    def test_module_defaults_to_baseline(self):
        moe = MoE(dim=8, intermediate=16, experts=8)
        module = top2.Top2MoE(moe, freeze=False)
        update, logits, weights, pairs = module(self.x)
        top = logits.topk(2, dim=-1).indices
        base_weights = F.softmax(logits.gather(-1, top), dim=-1)
        baseline = ref_top2_update(moe.pool, self.x, top, base_weights)
        self.assertTrue(torch.allclose(update, baseline, atol=1e-6))
        self.assertTrue(torch.equal(pairs, top.sort(dim=-1).values))
        self.assertTrue(torch.equal(module.baseline_actions(self.x), top2.pair_to_action(pairs, experts=8)))

    def test_module_freezes_router_and_experts(self):
        moe = MoE(dim=8, intermediate=16, experts=8)
        module = top2.Top2MoE(moe)
        self.assertTrue(all(not p.requires_grad for p in moe.parameters()))

    def test_uniform_ablation(self):
        actions = torch.randint(0, 28, (2, 5))
        update, weights, pairs = top2.top2_dispatch(
            self.pool, self.x, actions=actions, gate_logits=self.logits, weight_rule=top2.UNIFORM_WEIGHTS
        )
        self.assertTrue(torch.equal(weights, torch.full_like(weights, 0.5)))
        expected = ref_top2_update(self.pool, self.x, pairs, weights)
        self.assertTrue(torch.allclose(update, expected, atol=1e-6))
        router_update, _, _ = top2.top2_dispatch(self.pool, self.x, actions=actions, gate_logits=self.logits)
        self.assertFalse(torch.allclose(update, router_update, atol=1e-6))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            top2.top2_dispatch(self.pool, self.x, actions=torch.randint(0, 28, (2, 4)), gate_logits=self.logits)
        with self.assertRaises(ValueError):
            top2.top2_dispatch(self.pool, self.x)
        with self.assertRaises(ValueError):
            top2.top2_dispatch(self.pool, self.x, actions=torch.randint(0, 28, (2, 5)))


class TestDensePositiveControl(unittest.TestCase):
    def test_e2_matches_dense_forward_and_gradients(self):
        torch.manual_seed(2)
        pool = ExpertPool(dim=8, intermediate=16, experts=2)
        x = torch.randn(3, 4, 8, requires_grad=True)
        logits = torch.randn(3, 4, 2)

        update, weights, pairs = top2.top2_dispatch(pool, x, actions=torch.zeros(3, 4, dtype=torch.long), gate_logits=logits)
        self.assertTrue(torch.equal(pairs, torch.tensor([0, 1]).expand(3, 4, 2)))
        dense = ref_top2_update(pool, x, pairs, weights)
        self.assertTrue(torch.allclose(update, dense, atol=1e-6))

        upstream = torch.randn(3, 4, 8)
        (update * upstream).sum().backward()

        x_ref = x.detach().clone().requires_grad_(True)
        pool_ref = ExpertPool(dim=8, intermediate=16, experts=2)
        with torch.no_grad():
            pool_ref.w1.copy_(pool.w1.detach())
            pool_ref.w3.copy_(pool.w3.detach())
            pool_ref.w2.copy_(pool.w2.detach())
        weights_ref = F.softmax(logits, dim=-1)
        ref = ref_top2_update(pool_ref, x_ref, pairs, weights_ref)
        (ref * upstream).sum().backward()

        self.assertTrue(torch.allclose(x.grad, x_ref.grad, atol=1e-6))
        for name in ("w1", "w3", "w2"):
            grad = getattr(pool, name).grad
            grad_ref = getattr(pool_ref, name).grad
            self.assertTrue(torch.allclose(grad, grad_ref, atol=1e-6), name)
            self.assertGreater(grad.abs().max().item(), 0.0, f"{name} received no gradient")


class TestBF16Safety(unittest.TestCase):
    def test_bf16_buffer_safety(self):
        torch.manual_seed(3)
        pool = ExpertPool(dim=8, intermediate=16, experts=8)
        x = torch.randn(2, 5, 8)
        logits = torch.randn(2, 5, 8)
        actions = torch.randint(0, 28, (2, 5))
        reference, _, _ = top2.top2_dispatch(pool, x, actions=actions, gate_logits=logits)

        pool_bf16 = ExpertPool(dim=8, intermediate=16, experts=8)
        with torch.no_grad():
            pool_bf16.w1.copy_(pool.w1)
            pool_bf16.w3.copy_(pool.w3)
            pool_bf16.w2.copy_(pool.w2)
        pool_bf16.to(torch.bfloat16)
        update, weights, _ = top2.top2_dispatch(
            pool_bf16, x.to(torch.bfloat16), actions=actions, gate_logits=logits.to(torch.bfloat16)
        )
        self.assertEqual(update.dtype, torch.bfloat16)
        self.assertEqual(weights.dtype, torch.float32)
        self.assertTrue(torch.isfinite(update).all())
        self.assertTrue(torch.allclose(update.float(), reference, rtol=2e-2, atol=1e-2))

    def test_bf16_module_matches_fp32_baseline(self):
        torch.manual_seed(4)
        moe = MoE(dim=8, intermediate=16, experts=8)
        x = torch.randn(2, 5, 8)
        module = top2.Top2MoE(moe)
        reference, _, _, _ = module(x)

        moe_bf16 = MoE(dim=8, intermediate=16, experts=8)
        moe_bf16.load_state_dict(moe.state_dict())
        moe_bf16.to(torch.bfloat16)
        module_bf16 = top2.Top2MoE(moe_bf16)
        update, _, weights, _ = module_bf16(x.to(torch.bfloat16))
        self.assertEqual(update.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(update).all())
        self.assertTrue(torch.allclose(update.float(), reference, rtol=2e-2, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
