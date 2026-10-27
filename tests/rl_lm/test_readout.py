import unittest

import torch

from rl_lm.executor import FrozenLastFFNExecutor, freeze_backbone
from rl_lm.ledger import RLLedger
from rl_lm.readout import diagnostic_fp32_ce, production_logits, terminal_ce
from tests.rl_lm.tiny_lm import TINY_DIM, TINY_VOCAB, build_tiny_model


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.model = build_tiny_model()
        freeze_backbone(self.model)
        self.h = torch.randn(5, TINY_DIM)
        self.targets = torch.randint(0, TINY_VOCAB, (5,))
        self.ledger = RLLedger()

    def test_last_ffn_change_changes_final_readout(self):
        observation = FrozenLastFFNExecutor(self.model, chunk=64, compute_dtype="float32").observe(
            self.h, self.h, torch.zeros(5, dtype=torch.int64), -torch.ones(5, 2, dtype=torch.int64), max_calls=1
        )
        changed = FrozenLastFFNExecutor(self.model, chunk=64, compute_dtype="float32").step_pair(
            self.h, observation, torch.full((5,), 2, dtype=torch.int64)
        )
        ce_original = terminal_ce(self.model, self.h, self.targets)
        ce_changed = terminal_ce(self.model, changed, self.targets)
        self.assertFalse(torch.allclose(ce_original, ce_changed))
        self.assertFalse(ce_changed.requires_grad)

    def test_production_and_diagnostic_dtyps_and_ledger(self):
        terminal = terminal_ce(self.model, self.h, self.targets, chunk=3, ledger=self.ledger)
        self.assertEqual(terminal.dtype, torch.float32)
        self.assertEqual(self.ledger.physical_head_rows, 6)
        diagnostic = diagnostic_fp32_ce(self.model, self.h, self.targets, chunk=2)
        self.assertEqual(diagnostic.dtype, torch.float32)
        logits = production_logits(self.model, self.h, chunk=2, compute_dtype="float32")
        self.assertEqual(logits.shape, (5, TINY_VOCAB))
        self.assertEqual(logits.dtype, torch.float32)

    def test_bad_targets_rejected(self):
        with self.assertRaises(ValueError):
            terminal_ce(self.model, self.h, torch.tensor([-1, 1, 2, 3, 4]))
        with self.assertRaises(ValueError):
            terminal_ce(self.model, self.h.to(torch.bfloat16), self.targets)

    @unittest.skipUnless(torch.cuda.is_available(), "BF16 head dtype contract requires CUDA")
    def test_bf16_head_dtype_on_gpu(self):
        device = torch.device("cuda")
        model = build_tiny_model().to(device)
        h = self.h.to(device)
        logits = production_logits(model, h, chunk=2, compute_dtype="bfloat16")
        self.assertEqual(logits.dtype, torch.bfloat16)
        ce = terminal_ce(model, h, self.targets.to(device))
        self.assertEqual(ce.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
