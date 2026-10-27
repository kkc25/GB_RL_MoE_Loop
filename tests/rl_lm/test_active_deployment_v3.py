"""Active dispatch correctness; CPU tests do not establish CUDA speed/quality."""
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from rl_lm.actor import PairActor
from rl_lm.executor import ACTIVE_MERGED_BACKEND, MERGED_BACKEND, make_executor
from rl_lm.inference import forward_with_last_ffn_policy
from rl_lm.ledger import RLLedger
from tests.rl_lm.test_online_parity import ScriptedPolicy
from tests.rl_lm.tiny_lm import TINY_DIM, build_tiny_model


class ActiveDeploymentTests(unittest.TestCase):
    device = 'cpu'
    compute_dtype = 'float32'
    tolerance = 1e-5

    def setUp(self):
        self.model = build_tiny_model().to(self.device)
        self.ids = torch.arange(8, device=self.device).reshape(1, 8)

    def run_policy(self, policy, backend=ACTIVE_MERGED_BACKEND, calls=2):
        ledger = RLLedger()
        # ScriptedPolicy is intentionally CPU-based; move its output for GPU.
        def wrapped(*args):
            return policy(*args).to(self.device)
        output = forward_with_last_ffn_policy(
            self.model, wrapped, self.ids, torch.zeros_like(self.ids),
            max_calls=calls, backend=backend, return_trace=True,
            compute_dtype=self.compute_dtype, ledger=ledger)
        return output, ledger

    def assert_parity(self, actual, expected):
        for field in ('actions', 'rounds', 'decision_mask'):
            self.assertTrue(torch.equal(getattr(actual, field), getattr(expected, field)), field)
        for field in ('final_hidden', 'logits'):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field),
                                       atol=self.tolerance, rtol=self.tolerance)
        targets = self.ids.reshape(-1)
        a = F.cross_entropy(actual.logits.reshape(8, -1).float(), targets, reduction='none')
        b = F.cross_entropy(expected.logits.reshape(8, -1).float(), targets, reduction='none')
        torch.testing.assert_close(a, b, atol=self.tolerance, rtol=self.tolerance)

    def test_stop_continue_mixed_physical_and_closed_loop_parity(self):
        for second, physical in ((0, 16), (1, 32), ([0, 2, 0, 28, 0, 3, 0, 8], 24)):
            with self.subTest(second=second):
                actual, ledger = self.run_policy(ScriptedPolicy([1, second], 8))
                expected, legacy = self.run_policy(ScriptedPolicy([1, second], 8), MERGED_BACKEND)
                self.assert_parity(actual, expected)
                self.assertEqual(ledger.physical_expert_rows, physical)
                self.assertEqual(ledger.logical_expert_rows, physical)
                self.assertEqual(legacy.physical_expert_rows, 32)
                first, _ = self.run_policy(ScriptedPolicy([1], 8), calls=1)
                stopped = actual.rounds == 1
                self.assertTrue(torch.equal(actual.final_hidden[stopped], first.final_hidden[stopped]))

    def test_finished_rows_across_later_slots_and_ties(self):
        class TiedPolicy:
            def __call__(self, features, reference, allowed):
                logits = features.new_full((len(features), 29), -torch.inf)
                logits[:, 1:3] = 0  # lowest pair action must win at r0
                logits[allowed, 0] = 0  # STOP wins ties at r1
                return logits
        actual, ledger = self.run_policy(TiedPolicy(), calls=3)
        self.assertTrue((actual.actions[:, :, :2] == torch.tensor([1, 0], device=self.device)).all())
        self.assertTrue((actual.actions[:, :, 2] == -1).all())
        self.assertEqual(ledger.physical_expert_rows, 16)
        script = [1, [0, 3, 0, 3, 0, 3, 0, 3], 1]
        actual, ledger = self.run_policy(ScriptedPolicy(script, 8), calls=3)
        expected, _ = self.run_policy(ScriptedPolicy(script, 8), MERGED_BACKEND, calls=3)
        self.assert_parity(actual, expected)
        self.assertEqual(ledger.physical_expert_rows, 32)
        self.assertEqual(ledger.actor_padding_rows, 4)

    def test_real_actor_closed_loop(self):
        actor = PairActor(2 * TINY_DIM + 20).to(self.device).eval()
        with torch.no_grad():
            actor.pair_head.weight.normal_(std=.02)
            actor.stop_head.bias.fill_(-8)  # continue, recompute state-dependent pairs
        actual, _ = self.run_policy(actor, calls=3)
        expected, _ = self.run_policy(actor, MERGED_BACKEND, calls=3)
        self.assert_parity(actual, expected)

    def test_physical_ledger_matches_actual_expert_linear_rows(self):
        ledger = RLLedger()
        executor = make_executor(self.model, ACTIVE_MERGED_BACKEND,
                                 compute_dtype=self.compute_dtype, ledger=ledger)
        h = torch.randn(8, TINY_DIM, device=self.device)
        obs = executor.observe(h, h, torch.zeros(8, dtype=torch.int64, device=self.device),
                               torch.full((8, 2), -1, dtype=torch.int64, device=self.device), max_calls=2)
        actions = torch.ones(8, dtype=torch.int64, device=self.device)
        for mask in (torch.zeros(8, dtype=torch.bool, device=self.device),
                     torch.arange(8, device=self.device) % 2 == 1,
                     torch.ones(8, dtype=torch.bool, device=self.device)):
            ledger.physical_expert_rows = 0
            with patch('rl_lm.executor.F.linear', wraps=F.linear) as linear:
                executor.step_active(h, obs, actions, mask)
            # Each expert assignment executes three linears (w1, w3, w2).
            actual_linear_rows = sum(call.args[0].shape[0] for call in linear.call_args_list)
            self.assertEqual(actual_linear_rows, 3 * ledger.physical_expert_rows)
            self.assertEqual(ledger.physical_expert_rows, 2 * int(mask.sum()))
        with self.assertRaisesRegex(ValueError, 'take_pair'):
            executor.step_active(h, obs, actions, torch.ones(8, device=self.device))

    def test_compaction_empty_skew_and_row_order(self):
        for count in (0, 1, 8, 28, 65):
            ledger = RLLedger()
            executor = make_executor(self.model, ACTIVE_MERGED_BACKEND,
                                     compute_dtype=self.compute_dtype, ledger=ledger)
            h = torch.randn(count, TINY_DIM * 2, device=self.device)[:, ::2]
            obs = executor.observe(h, h, torch.zeros(count, dtype=torch.int64, device=self.device),
                                   torch.full((count, 2), -1, dtype=torch.int64, device=self.device), max_calls=2)
            for skew in (True, False):
                actions = torch.ones(count, dtype=torch.int64, device=self.device) if skew else torch.arange(count, device=self.device) % 28 + 1
                mask = torch.arange(count, device=self.device) % 3 == 1
                expected = executor.step_pair(h, obs, actions)
                ledger.physical_expert_rows = 0
                actual = executor.step_active(h, obs, actions, mask)
                self.assertEqual(ledger.physical_expert_rows, 2 * int(mask.sum()))
                self.assertTrue(torch.equal(actual[~mask], h[~mask]))
                torch.testing.assert_close(actual[mask], expected[mask], atol=self.tolerance, rtol=self.tolerance)
                with patch.object(executor, 'pair_update', side_effect=AssertionError('empty dispatch')):
                    unchanged = executor.step_active(h, obs, actions, torch.zeros_like(mask))
                self.assertTrue(torch.equal(unchanged, h))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; no GPU parity claim')
class ActiveDeploymentCUDA(ActiveDeploymentTests):
    device = 'cuda'
    compute_dtype = 'bfloat16'
    tolerance = .02  # bounded tiny fixture only, not production acceptance
