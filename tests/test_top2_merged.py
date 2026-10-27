"""CPU FP32 parity for the explicitly opt-in common/prefix dispatcher."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from config import ModelConfig
from model.moe import ExpertPool, MoE
from model.top2 import (Top2MoE, top2_dispatch, set_top2_dispatch_backend,
                        TOP2_DISPATCH_CONTRACTS, top2_dispatch_identity)
from model.transformer import LoopMoE


class TestMergedTop2(unittest.TestCase):
    def assert_optional_close(self, a, b):
        self.assertEqual(a is None, b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)

    def test_forward_and_all_gradients(self):
        for layout in ('mixed', 'skew', 'duplicate', 'empty', 'noncontiguous'):
            for rule in ('explicit', 'uniform', 'router_detached', 'router_live'):
                with self.subTest(layout=layout, rule=rule):
                    torch.manual_seed(23)
                    pool = ExpertPool(6, 9, 5)
                    other = copy.deepcopy(pool)
                    # Scale up to make gradient comparisons nontrivial.
                    with torch.no_grad():
                        for p, q in zip(pool.parameters(), other.parameters()):
                            p.mul_(8); q.copy_(p)
                    n = 0 if layout == 'empty' else 12
                    x = torch.randn(3, n, 6).transpose(0, 1).requires_grad_()
                    pairs = torch.randint(0, 5, (n, 3, 2))
                    if layout == 'skew':
                        pairs[:] = torch.tensor([4, 0])
                    if layout == 'duplicate':
                        pairs[:] = 2
                    if layout == 'noncontiguous':
                        pairs = pairs.transpose(0, 1)
                        x = x.transpose(0, 1).detach().requires_grad_()
                        x = x.transpose(0, 1).contiguous().transpose(0, 1).detach().requires_grad_()
                    y = x.detach().clone(memory_format=torch.preserve_format).requires_grad_()
                    logits = torch.randn(*pairs.shape[:-1], 5, requires_grad=True)
                    logits2 = logits.detach().clone().requires_grad_()
                    weights = torch.randn(*pairs.shape, requires_grad=True)
                    weights2 = weights.detach().clone().requires_grad_()
                    kwargs = dict(pairs=pairs, weight_rule='uniform' if rule == 'uniform' else 'router',
                                  detach_weights=rule != 'router_live')
                    a, aw, ap = top2_dispatch(pool, x, gate_logits=logits,
                                             weights=weights if rule == 'explicit' else None, **kwargs)
                    b, bw, bp = top2_dispatch(other, y, gate_logits=logits2, backend='merged',
                                             weights=weights2 if rule == 'explicit' else None, **kwargs)
                    torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)
                    self.assertTrue(torch.equal(aw, bw)); self.assertTrue(torch.equal(ap, bp))
                    self.assertEqual(a.dtype, x.dtype)
                    if a.requires_grad:
                        probe = torch.randn_like(a)
                        (a * probe).sum().backward(); (b * probe).sum().backward()
                    for u, v in [(x, y), (logits, logits2), (weights, weights2),
                                 *zip(pool.parameters(), other.parameters())]:
                        self.assert_optional_close(u.grad, v.grad)

    def test_cpu_dtype_and_autocast_contract(self):
        torch.manual_seed(29)
        for dtype in (torch.float32, torch.bfloat16):
            pool = ExpertPool(6, 9, 4).to(dtype)
            x = torch.randn(7, 6).to(dtype)
            pairs = torch.randint(0, 4, (7, 2))
            with torch.autocast('cpu', dtype=torch.bfloat16):
                a, aw, _ = top2_dispatch(pool, x, pairs=pairs, weight_rule='uniform')
                b, bw, _ = top2_dispatch(pool, x, pairs=pairs, weight_rule='uniform', backend='merged')
            self.assertEqual(a.dtype, dtype)
            self.assertEqual(b.dtype, dtype)
            self.assertEqual(aw.dtype, torch.float32)
            self.assertEqual(bw.dtype, torch.float32)
            torch.testing.assert_close(a, b, rtol=0.02, atol=2e-5)

    def test_one_expert_bucket_across_both_slots(self):
        pool = ExpertPool(6, 9, 4)
        x = torch.randn(8, 6)
        pairs = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]] * 2)
        with patch('model.top2.F.linear', wraps=F.linear) as linear:
            top2_dispatch(pool, x, pairs=pairs, weight_rule='uniform', backend='merged')
            self.assertEqual(linear.call_count, 3 * pool.experts)
        with patch.object(pool, 'forward', wraps=pool.forward) as forward:
            top2_dispatch(pool, x, pairs=pairs, weight_rule='uniform')
            self.assertEqual(forward.call_count, 2)

    def test_actions_ties_freeze_and_runtime_contract(self):
        for freeze in (True, False):
            legacy = Top2MoE(MoE(6, 9, 5), freeze=freeze)
            merged = copy.deepcopy(legacy)
            with torch.no_grad():
                legacy.moe.gate.weight.zero_(); merged.moe.gate.weight.zero_()
            keys = list(merged.state_dict())
            self.assertEqual(legacy.dispatch_backend, 'legacy')
            self.assertEqual(set_top2_dispatch_backend(merged, 'merged'), 1)
            self.assertEqual(keys, list(merged.state_dict()))
            x = torch.randn(7, 6)
            for actions in (None, torch.tensor([0, 1, 2, 3, 4, 5, 6])):
                for a, b in zip(legacy(x, actions=actions), merged(x, actions=actions)):
                    torch.testing.assert_close(a, b)
            self.assertEqual(set_top2_dispatch_backend(merged, 'legacy'), 1)
            self.assertEqual(merged.dispatch_backend, 'legacy')
            self.assertIn('merged', TOP2_DISPATCH_CONTRACTS, top2_dispatch_identity)
            with self.assertRaises(ValueError):
                set_top2_dispatch_backend(merged, 'typo')
            self.assertEqual(merged.dispatch_backend, 'legacy')

    def test_whole_model_all_eleven_prefix_layers_and_gradients(self):
        torch.manual_seed(7)
        cfg = ModelConfig(layers=12, hidden_size=8, attention_heads=2, head_dim=4,
                          intermediate_size=12, experts_per_layer=4, vocab_size=23,
                          top_k=2, adaptive_layers=(11,), target_only=False)
        legacy = LoopMoE(cfg)
        merged = copy.deepcopy(legacy)
        self.assertEqual(set_top2_dispatch_backend(merged.blocks[:11], 'merged'), 11)
        identity = top2_dispatch_identity(merged)
        self.assertEqual(len(identity['executors']), 12)
        self.assertEqual([e['module'] for e in identity['executors'][:11]],
                         [f'blocks.{i}.executor' for i in range(11)])
        self.assertEqual([e['backend'] for e in identity['executors']], ['merged'] * 11 + ['legacy'])
        self.assertEqual(merged.blocks[-1].executor.dispatch_backend, 'legacy')
        self.assertTrue(all(b.executor.dispatch_backend == 'legacy' for b in legacy.blocks))
        self.assertEqual(list(legacy.state_dict()), list(merged.state_dict()))
        ids = torch.randint(0, 23, (2, 5))
        a, b = legacy(ids, labels=ids), merged(ids, labels=ids)
        torch.testing.assert_close(a.logits, b.logits, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(a.loss, b.loss)
        self.assertTrue(torch.equal(a.calls, b.calls))
        a.loss.backward(); b.loss.backward()
        for (name, p), (other_name, q) in zip(legacy.named_parameters(), merged.named_parameters()):
            self.assertEqual(name, other_name)
            with self.subTest(parameter=name):
                self.assert_optional_close(p.grad, q.grad)


if __name__ == '__main__':
    unittest.main()
