"""CPU numerical/shape evidence only; no CUDA speed or parity claim."""
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from rl_lm.executor import ACTIVE_MERGED_BACKEND, MERGED_BACKEND
from rl_lm.inference import (CanonicalTeacherPolicy, OnlineCEOutput, OnlineOutput,
                             forward_with_last_ffn_policy)
from rl_lm.ledger import RLLedger
from rl_lm.readout import production_logits, production_token_ce
from tests.rl_lm.test_online_parity import ScriptedPolicy
from tests.rl_lm.tiny_lm import TINY_DIM, TINY_VOCAB, build_tiny_model


def reference_ce(logits, labels, window=8192):
    return torch.cat([F.cross_entropy(logits[s:s + window].float(), labels[s:s + window],
                                     reduction='none', ignore_index=-100)
                      for s in range(0, len(labels), window)]).cpu() if len(labels) else torch.empty(0)


class VocabularyAllocationGuard(TorchDispatchMode):
    """Observe all dispatcher tensor results, including cast/empty/cat copies."""
    def __init__(self, limit):
        self.limit = limit
        self.seen = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        def check(value):
            if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[1] == TINY_VOCAB:
                self.seen.append((str(func), tuple(value.shape)))
                if value.shape[0] > self.limit:
                    raise AssertionError(f'unbounded vocabulary tensor: {func} {value.shape}')
            elif isinstance(value, (tuple, list)):
                for item in value:
                    check(item)
        # The tied weight transpose is a view, not a token-logit buffer.
        if func != torch.ops.aten.t.default:
            check(output)
        return output


class StreamingReadoutTests(unittest.TestCase):
    def test_exact_windows_tail_padding_ignore_and_no_grad(self):
        model = build_tiny_model()
        model.embedding.weight.requires_grad_(True)
        for dtype in ('float32', 'bfloat16'):
            # Include a head chunk crossing a CE window boundary, multiple
            # CE windows, and the real production 8192 boundary.
            for count, chunk, window in ((0, 4, 8), (1, 4, 8), (8, 4, 8),
                                         (19, 3, 8), (8193, 64, 8192)):
                with self.subTest(dtype=dtype, count=count, chunk=chunk):
                    h = torch.randn(count, TINY_DIM, requires_grad=True)
                    labels = torch.arange(count, dtype=torch.int64) % TINY_VOCAB
                    labels[::3] = -100
                    expected = reference_ce(production_logits(model, h, chunk=chunk, compute_dtype=dtype),
                                            labels, window)
                    ledger = RLLedger()
                    guard = VocabularyAllocationGuard(window)
                    with guard, patch('rl_lm.readout.F.cross_entropy', wraps=F.cross_entropy) as ce:
                        actual = production_token_ce(model, h, labels, chunk=chunk,
                                                     compute_dtype=dtype, ce_window=window, ledger=ledger)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(float(actual.sum()), float(expected.sum()))
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertEqual(actual.device.type, 'cpu')
                    self.assertFalse(actual.requires_grad)
                    self.assertIsNone(actual.grad_fn)
                    self.assertIsNone(h.grad)
                    self.assertIsNone(model.embedding.weight.grad)
                    self.assertEqual(ledger.physical_head_rows, ((count + chunk - 1) // chunk) * chunk)
                    self.assertEqual([c.args[0].shape[0] for c in ce.call_args_list],
                                     [min(window, count - s) for s in range(0, count, window)])
                    self.assertTrue(bool((actual[labels == -100] == 0).all()))
                    if count:
                        self.assertTrue(guard.seen)
                        buffers = [shape for op, shape in guard.seen if op == 'aten.new_empty.default']
                        self.assertEqual(buffers, [(min(count, window), TINY_VOCAB)])

    def test_validation(self):
        model = build_tiny_model()
        h = torch.zeros(2, TINY_DIM)
        for labels, kwargs in ((torch.tensor([-1, 0]), {}), (torch.tensor([64, 0]), {}),
                               (torch.zeros(2), {}), (torch.zeros(2, dtype=torch.long), {'ce_window': 0}),
                               (torch.zeros(2, dtype=torch.long), {'chunk': 9, 'ce_window': 8}),
                               (torch.zeros(2, dtype=torch.long), {'compute_dtype': 'float16'})):
            with self.assertRaises(ValueError):
                production_token_ce(model, h, labels, **kwargs)

    def test_online_modes_identical_actions_hidden_cost_and_rng(self):
        model = build_tiny_model()
        ids = torch.arange(15).reshape(3, 5)
        labels = ids.clone()
        labels[:, -1] = -100
        for dtype in ('float32', 'bfloat16'):
            for backend in ('sparse_merged_v1', MERGED_BACKEND, ACTIVE_MERGED_BACKEND):
                for canonical in (False, True):
                    with self.subTest(dtype=dtype, backend=backend, canonical=canonical):
                        def policy():
                            return CanonicalTeacherPolicy() if canonical else ScriptedPolicy(
                                [1, [0, 2, 0, 3, 0] * 3], ids.numel())
                        old_ledger, new_ledger = RLLedger(), RLLedger()
                        kwargs = dict(max_calls=2, backend=backend, compute_dtype=dtype,
                                      return_trace=True, head_chunk=4)
                        old = forward_with_last_ffn_policy(model, policy(), ids, torch.zeros_like(ids),
                                                           ledger=old_ledger, **kwargs)
                        rng = torch.get_rng_state().clone()
                        with patch('rl_lm.inference.production_logits', side_effect=AssertionError('full logits')):
                            new = forward_with_last_ffn_policy(model, policy(), ids, torch.zeros_like(ids),
                                output_mode='token_ce', labels=labels, ledger=new_ledger, **kwargs)
                        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                        self.assertIsInstance(old, OnlineOutput)
                        self.assertIsInstance(new, OnlineCEOutput)
                        self.assertFalse(hasattr(new, 'logits'))
                        for field in ('rounds', 'actions', 'decision_mask', 'final_hidden'):
                            self.assertTrue(torch.equal(getattr(old, field), getattr(new, field)), field)
                        self.assertEqual(old_ledger.to_dict(), new_ledger.to_dict())
                        expected = reference_ce(old.logits.reshape(-1, TINY_VOCAB), labels.reshape(-1))
                        self.assertTrue(torch.equal(expected.reshape_as(labels), new.token_ce))
                        self.assertFalse(new.token_ce.requires_grad)


class SingleStreamingReadoutTests(unittest.TestCase):
    def test_single_closed_loop_modes_exact_and_no_full_logits(self):
        from rl_lm.single_executor import (ArgmaxSingleTeacherPolicy, SingleLedger,
            SINGLE_CANONICAL_BACKEND, SINGLE_SPARSE_BACKEND, single_greedy_lowest_id,
            single_online_forward)
        from tests.rl_lm.test_single_online import ScriptedSinglePolicy
        model = build_tiny_model()
        ids = torch.arange(15).reshape(3, 5)
        labels = ids.clone()
        labels[:, -1] = -100
        for dtype in ('float32', 'bfloat16'):
            for backend in (SINGLE_CANONICAL_BACKEND, SINGLE_SPARSE_BACKEND):
                for teacher in (False, True):
                    with self.subTest(dtype=dtype, backend=backend, teacher=teacher):
                        decisions = []
                        def greedy(log_probs):
                            result = single_greedy_lowest_id(log_probs)
                            decisions.append(result.clone())
                            return result
                        def policy():
                            return ArgmaxSingleTeacherPolicy() if teacher else ScriptedSinglePolicy(
                                [1, [0, 2, 0, 3, 0] * 3], ids.numel())
                        kwargs = dict(max_calls=2, backend=backend, compute_dtype=dtype, head_chunk=4)
                        old_ledger, new_ledger = SingleLedger(), SingleLedger()
                        with patch('rl_lm.single_executor.single_greedy_lowest_id', side_effect=greedy):
                            logits, old_rounds = single_online_forward(model, policy(), ids,
                                torch.zeros_like(ids), ledger=old_ledger, **kwargs)
                        old_decisions = decisions[:]
                        decisions.clear()
                        rng = torch.get_rng_state().clone()
                        with patch('rl_lm.readout.production_logits', side_effect=AssertionError('full logits')), \
                             patch('rl_lm.single_executor.single_greedy_lowest_id', side_effect=greedy):
                            ce, rounds = single_online_forward(model, policy(), ids, torch.zeros_like(ids),
                                labels=labels, output_mode='token_ce', ledger=new_ledger, **kwargs)
                        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                        self.assertTrue(torch.equal(rounds, old_rounds))
                        self.assertEqual(len(decisions), len(old_decisions))
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip(decisions, old_decisions)))
                        expected = reference_ce(logits.reshape(-1, TINY_VOCAB), labels.reshape(-1))
                        self.assertTrue(torch.equal(ce, expected.reshape_as(labels)))
                        self.assertEqual(float(ce.sum()), float(expected.sum()))
                        self.assertFalse(ce.requires_grad)
                        self.assertIsNone(ce.grad_fn)
                        self.assertEqual(ce.device.type, 'cpu')
                        self.assertEqual(old_ledger.to_dict(), new_ledger.to_dict())

    def test_single_label_mode_validation(self):
        from rl_lm.single_executor import ArgmaxSingleTeacherPolicy, single_online_forward
        model = build_tiny_model()
        ids = torch.arange(4).reshape(1, 4)
        for mode, labels in (('bad', None), ('token_ce', None), ('token_ce', ids.float()),
                             ('token_ce', ids.reshape(-1)), ('logits', ids)):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                single_online_forward(model, ArgmaxSingleTeacherPolicy(), ids, None,
                    max_calls=1, backend='sparse_single_v1', output_mode=mode, labels=labels)


if __name__ == '__main__':
    unittest.main()
