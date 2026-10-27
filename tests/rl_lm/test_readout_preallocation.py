"""Compare preallocation against the original exact padded head computation."""
import unittest

import torch
from torch.nn import functional as F

from rl_lm.ledger import RLLedger
from rl_lm.readout import production_logits
from tests.rl_lm.tiny_lm import TINY_DIM, build_tiny_model


@torch.no_grad()
def original_logits(model, h, chunk, dtype):
    if len(h) == 0:
        return h.new_empty((0, model.embedding.weight.shape[0]))
    pieces = []
    for start in range(0, len(h), chunk):
        valid = min(chunk, len(h) - start)
        hc = h[start:start + valid]
        if valid < chunk:
            hc = torch.cat((hc, h.new_zeros((chunk-valid, h.shape[1]))))
        with torch.autocast(device_type=h.device.type, dtype=torch.bfloat16, enabled=dtype == 'bfloat16'):
            logits = F.linear(model.final_norm(hc), model.embedding.weight)
        pieces.append(logits[:valid])
    return torch.cat(pieces)


class ReadoutPreallocationTests(unittest.TestCase):
    device = 'cpu'

    def test_exact_dtype_values_padding_empty_and_no_grad(self):
        model = build_tiny_model().to(self.device)
        model.embedding.weight.requires_grad_(True)
        for dtype in ('float32', 'bfloat16'):
            for count in (0, 1, 4, 5, 17):
                with self.subTest(dtype=dtype, count=count):
                    h = torch.randn(count, TINY_DIM, device=self.device, requires_grad=True)
                    ledger = RLLedger()
                    actual = production_logits(model, h, chunk=4, compute_dtype=dtype, ledger=ledger)
                    expected = original_logits(model, h, 4, dtype)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertFalse(actual.requires_grad)
                    self.assertIsNone(actual.grad_fn)
                    self.assertEqual(ledger.physical_head_rows, ((count + 3) // 4) * 4)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; no GPU memory claim')
class ReadoutPreallocationCUDA(ReadoutPreallocationTests):
    device = 'cuda'

    def test_peak_allocation_lower_than_list_cat(self):
        model = build_tiny_model().cuda()
        h = torch.randn(8192, TINY_DIM, device='cuda')
        def peak(fn):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            output = fn()
            torch.cuda.synchronize()
            used = torch.cuda.max_memory_allocated() - baseline
            del output
            return used
        # Warm up both paths before comparing live allocation peaks.
        production_logits(model, h, chunk=64)
        original_logits(model, h, 64, 'bfloat16')
        old = peak(lambda: original_logits(model, h, 64, 'bfloat16'))
        new = peak(lambda: production_logits(model, h, chunk=64))
        self.assertLess(new, old)
