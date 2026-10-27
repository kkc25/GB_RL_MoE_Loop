"""Single-branch online closed loop: greedy joint-9 forward, STOP semantics."""

import unittest

import torch
from torch.nn import functional as F

from rl_lm.readout import production_logits
from rl_lm.single_executor import (
    JOINT_SINGLE_ACTIONS,
    STOP_ACTION,
    ArgmaxSingleTeacherPolicy,
    SingleLedger,
    single_argmax_expert,
    single_online_forward,
)
from tests.rl_lm.tiny_lm import build_tiny_model


class ScriptedSinglePolicy:
    """Joint-9 log-probs concentrated on scripted per-slot actions."""

    feature_dim = None

    def __init__(self, slot_actions, count):
        self.slot_actions = slot_actions
        self.count = count
        self.calls = 0

    def __call__(self, features, reference, stop_allowed):
        slot = self.calls
        self.calls += 1
        if slot >= len(self.slot_actions):
            raise AssertionError("policy invoked past the scripted decision slots")
        scripted = self.slot_actions[slot]
        if isinstance(scripted, int):
            scripted = [scripted] * self.count
        log_probs = torch.full((self.count, JOINT_SINGLE_ACTIONS), -torch.inf)
        for row, action in enumerate(scripted):
            if action == STOP_ACTION and not bool(stop_allowed[row]):
                raise AssertionError("script requests an illegal STOP at r=0")
            log_probs[row, action] = 0.0
        return log_probs


class SingleOnlineClosedLoopTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.model = build_tiny_model()
        self.input_ids = torch.randint(0, 64, (2, 8))
        self.segment_ids = torch.zeros_like(self.input_ids)

    def test_argmax_rule_matches_manual_single_expert(self):
        from eval.counterfactual import cache_last_ffn_input

        logits, rounds = single_online_forward(
            self.model,
            ArgmaxSingleTeacherPolicy(),
            self.input_ids,
            self.segment_ids,
            max_calls=1,
            backend="sparse_single_v1",
            compute_dtype="float32",
        )
        self.assertTrue(bool((rounds == 1).all()))
        layer = self.model.config.layers - 1
        block = self.model.blocks[layer]
        gamma = float(self.model.gammas[layer])
        with torch.no_grad():
            prefix = cache_last_ffn_input(self.model, self.input_ids, self.segment_ids)
            n = block.ffn_norm(prefix.float()).float()
            z = block.moe.gate(n)
            experts = single_argmax_expert(z.float().reshape(-1, 8)).reshape(2, 8)
            updated = prefix.float().clone()
            for position in range(prefix.shape[1]):
                for row in range(prefix.shape[0]):
                    e = int(experts[row, position])
                    x = n[row, position]
                    hidden = F.silu(F.linear(x, block.moe.pool.w1[e])) * F.linear(x, block.moe.pool.w3[e])
                    updated[row, position] = prefix[row, position].float() + gamma * F.linear(hidden, block.moe.pool.w2[e]).float()
            manual = production_logits(self.model, updated.reshape(-1, 32), compute_dtype="float32").reshape(2, 8, -1)
        torch.testing.assert_close(logits.float(), manual.float(), atol=1e-5, rtol=1e-5)

    def test_r2_stop_closes_episode_and_hidden_stays_fixed(self):
        count = 2 * 8
        policy_r1 = ScriptedSinglePolicy([[3] * count], count)
        logits_r1, rounds_r1 = single_online_forward(
            self.model, policy_r1, self.input_ids, self.segment_ids,
            max_calls=1, backend="sparse_single_v1", compute_dtype="float32",
        )
        policy_r2 = ScriptedSinglePolicy([[3] * count, [STOP_ACTION] * count], count)
        logits_r2, rounds_r2 = single_online_forward(
            self.model, policy_r2, self.input_ids, self.segment_ids,
            max_calls=2, backend="sparse_single_v1", compute_dtype="float32",
        )
        self.assertEqual(policy_r2.calls, 2)
        self.assertTrue(bool((rounds_r2 == 1).all()))
        torch.testing.assert_close(logits_r2.float(), logits_r1.float(), atol=0, rtol=0)

    def test_two_rounds_apply_two_single_updates(self):
        count = 2 * 8
        policy = ScriptedSinglePolicy([[2] * count, [5] * count], count)
        logits, rounds = single_online_forward(
            self.model, policy, self.input_ids, self.segment_ids,
            max_calls=2, backend="sparse_single_v1", compute_dtype="float32",
        )
        self.assertTrue(bool((rounds == 2).all()))
        policy_once = ScriptedSinglePolicy([[2] * count, [STOP_ACTION] * count], count)
        logits_once, _ = single_online_forward(
            self.model, policy_once, self.input_ids, self.segment_ids,
            max_calls=2, backend="sparse_single_v1", compute_dtype="float32",
        )
        self.assertFalse(torch.allclose(logits.float(), logits_once.float()))

    def test_illegal_stop_at_r0_fails_fast(self):
        count = 2 * 8

        class BadPolicy:
            feature_dim = None

            def __call__(self, features, reference, stop_allowed):
                log_probs = torch.full((features.shape[0], JOINT_SINGLE_ACTIONS), -torch.inf)
                log_probs[:, STOP_ACTION] = 0.0
                return log_probs

        with self.assertRaises(RuntimeError):
            single_online_forward(
                self.model, BadPolicy(), self.input_ids, self.segment_ids,
                max_calls=1, backend="sparse_single_v1", compute_dtype="float32",
            )

    def test_ledger_counts_single_rounds(self):
        ledger = SingleLedger()
        single_online_forward(
            self.model,
            ArgmaxSingleTeacherPolicy(),
            self.input_ids,
            self.segment_ids,
            max_calls=1,
            backend="sparse_single_v1",
            compute_dtype="float32",
            ledger=ledger,
        )
        self.assertEqual(ledger.logical_single_rounds, 16)
        self.assertEqual(ledger.logical_expert_rows, 16)
        self.assertEqual(ledger.physical_expert_rows, 16)


if __name__ == "__main__":
    unittest.main()
