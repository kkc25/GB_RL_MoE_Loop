import unittest

import torch

from rl_lm.actions import JOINT_ACTIONS, STOP_ACTION
from rl_lm.executor import make_executor
from rl_lm.inference import CanonicalTeacherPolicy, forward_with_last_ffn_policy
from rl_lm.ledger import RLLedger
from rl_lm.readout import production_logits
from rl_lm.selector import canonical_pair
from tests.rl_lm.tiny_lm import TINY_DIM, build_tiny_model


class OnlineParityTests(unittest.TestCase):
    def test_r1_canonical_online_matches_full_forward(self):
        torch.manual_seed(3)
        model = build_tiny_model()
        input_ids = torch.randint(0, 64, (1, 8))
        segment_ids = torch.zeros_like(input_ids)
        with torch.no_grad():
            reference = model(input_ids, labels=None, depth=1, segment_ids=segment_ids)
        ledger = RLLedger()
        output = forward_with_last_ffn_policy(
            model,
            CanonicalTeacherPolicy(),
            input_ids,
            segment_ids,
            max_calls=1,
            backend="sparse_merged_v1",
            return_trace=True,
            compute_dtype="float32",
            ledger=ledger,
        )
        self.assertEqual(tuple(output.logits.shape), (1, 8, 64))
        self.assertTrue(bool((output.rounds == 1).all()))
        self.assertTrue(bool(output.actions[:, :, 0].gt(0).all()))
        torch.testing.assert_close(output.logits, reference.logits, atol=1e-5, rtol=1e-5)
        self.assertGreater(ledger.prefix_input_tokens, 0)
        self.assertEqual(ledger.logical_pair_rounds, 8)
        self.assertEqual(ledger.physical_expert_rows, 2 * 8)


class ScriptedPolicy:
    """Joint log-probs concentrated on a scripted action per (slot, row).

    ``slot_actions[slot]`` is either a single action id or a list of one
    action id per flat row.  Every invocation is counted so tests can prove
    the loop never asks for a third (post-terminal) decision.
    """

    feature_dim = None

    def __init__(self, slot_actions, count: int):
        self.slot_actions = slot_actions
        self.count = count
        self.calls = 0

    def __call__(self, features, pair_reference, stop_allowed):
        slot = self.calls
        self.calls += 1
        if slot >= len(self.slot_actions):
            raise AssertionError("policy invoked past the scripted decision slots")
        scripted = self.slot_actions[slot]
        if isinstance(scripted, int):
            scripted = [scripted] * self.count
        if len(scripted) != self.count or features.shape[0] != self.count:
            raise AssertionError("scripted actions must cover every flat row")
        log_probs = torch.full((self.count, JOINT_ACTIONS), -torch.inf)
        for row, action in enumerate(scripted):
            if action == STOP_ACTION and not bool(stop_allowed[row]):
                raise AssertionError("script requests an illegal STOP at r=0")
            log_probs[row, action] = 0.0
        return log_probs


class R2OnlineClosedLoopTests(unittest.TestCase):
    """T22 R2 part: STOP closed loop of the online forward (spec 5.2/10.2)."""

    def setUp(self):
        torch.manual_seed(5)
        self.model = build_tiny_model()
        self.input_ids = torch.randint(0, 64, (1, 8))
        self.segment_ids = torch.zeros_like(self.input_ids)
        self.count = 8

    def _run(self, policy, *, ledger=None):
        return forward_with_last_ffn_policy(
            self.model,
            policy,
            self.input_ids,
            self.segment_ids,
            max_calls=2,
            backend="sparse_merged_v1",
            return_trace=True,
            compute_dtype="float32",
            ledger=ledger,
        )

    def test_pair_stop_executes_exactly_one_round(self):
        # [pair, STOP]: one executed round, a recorded voluntary STOP, no
        # further computation; the hidden after STOP is untouched, so the
        # whole R2 episode is bitwise identical to the same single pair in R1.
        ledger = RLLedger()
        output = self._run(ScriptedPolicy([5, STOP_ACTION], self.count), ledger=ledger)
        self.assertTrue(bool((output.rounds == 1).all()))
        self.assertTrue(bool((output.actions[:, :, 0] == 5).all()))
        self.assertTrue(bool((output.actions[:, :, 1] == STOP_ACTION).all()))
        self.assertTrue(bool(output.decision_mask[:, :, :].all()))
        r1 = forward_with_last_ffn_policy(
            self.model,
            ScriptedPolicy([5], self.count),
            self.input_ids,
            self.segment_ids,
            max_calls=1,
            backend="sparse_merged_v1",
            return_trace=True,
            compute_dtype="float32",
        )
        self.assertTrue(torch.equal(output.final_hidden, r1.final_hidden))
        self.assertTrue(torch.equal(output.logits, r1.logits))
        self.assertEqual(ledger.logical_pair_rounds, self.count)
        self.assertEqual(ledger.logical_expert_rows, 2 * self.count)
        self.assertEqual(ledger.actor_decision_rows, 2 * self.count)  # pair + voluntary STOP
        self.assertEqual(ledger.actor_padding_rows, 0)

    def test_pair_pair_two_rounds_and_no_third_decision(self):
        # [pair, pair]: exactly two executed rounds; reaching Rmax terminates
        # the episode without any further actor call (no third STOP log-prob).
        policy = ScriptedPolicy([5, 9], self.count)
        ledger = RLLedger()
        output = self._run(policy, ledger=ledger)
        self.assertEqual(policy.calls, 2)
        self.assertTrue(bool((output.rounds == 2).all()))
        self.assertTrue(bool((output.actions[:, :, 0] == 5).all()))
        self.assertTrue(bool((output.actions[:, :, 1] == 9).all()))
        self.assertTrue(bool(output.decision_mask.all()))
        self.assertEqual(ledger.logical_pair_rounds, 2 * self.count)
        # Independent two-round reference: step the executor by hand through
        # the same two pairs, re-observing the gate after round one.
        executor = make_executor(self.model, "sparse_merged_v1", chunk=64, compute_dtype="float32")
        from eval.counterfactual import cache_last_ffn_input

        prefix = cache_last_ffn_input(self.model, self.input_ids, self.segment_ids)
        h0 = prefix.float().reshape(self.count, TINY_DIM).contiguous()
        rounds = torch.zeros(self.count, dtype=torch.int64)
        previous = torch.full((self.count, 2), -1, dtype=torch.int64)
        obs0 = executor.observe(h0, h0, rounds, previous, max_calls=2)
        h1 = executor.step_pair(h0, obs0, torch.full((self.count,), 5, dtype=torch.int64))
        obs1 = executor.observe(h1, h0, torch.ones(self.count, dtype=torch.int64), executor.table[[4] * self.count], max_calls=2)
        h2 = executor.step_pair(h1, obs1, torch.full((self.count,), 9, dtype=torch.int64))
        reference = production_logits(self.model, h2, chunk=64, compute_dtype="float32")
        self.assertTrue(torch.equal(output.final_hidden.reshape(self.count, TINY_DIM), h2))
        self.assertTrue(torch.equal(output.logits.reshape(self.count, -1), reference))

    def test_finished_rows_hidden_stays_untouched(self):
        # Mixed batch: even rows stop after one pair, odd rows take a second
        # pair.  Rows that stopped must keep their hidden bitwise while the
        # remaining rows execute round two.
        stop_rows = [row for row in range(self.count) if row % 2 == 0]
        slot1 = [STOP_ACTION if row % 2 == 0 else 9 for row in range(self.count)]
        mixed = self._run(ScriptedPolicy([5, slot1], self.count))
        self.assertTrue(bool((mixed.rounds.reshape(-1)[stop_rows] == 1).all()))
        self.assertTrue(bool((mixed.rounds.reshape(-1)[1::2] == 2).all()))
        one_round = self._run(ScriptedPolicy([5, STOP_ACTION], self.count))
        self.assertTrue(torch.equal(mixed.final_hidden[:, stop_rows, :], one_round.final_hidden[:, stop_rows, :]))
        # Stopped rows took no slot-1 pair action in the trace.
        self.assertTrue(bool((mixed.actions[:, :, 1].reshape(-1)[stop_rows] == STOP_ACTION).all()))
        self.assertTrue(bool((mixed.actions[:, :, 1].reshape(-1)[1::2] == 9).all()))

    def test_stop_at_r0_fails_fast(self):
        class StopAtR0:
            feature_dim = None

            def __call__(self, features, pair_reference, stop_allowed):
                log_probs = torch.full((features.shape[0], JOINT_ACTIONS), -torch.inf)
                log_probs[:, STOP_ACTION] = 0.0
                return log_probs

        with self.assertRaises(RuntimeError):
            self._run(StopAtR0())

    def test_canonical_r2_matches_independent_two_round_reference(self):
        # Canonical teacher never STOPs: R2 online must execute exactly the
        # canonical pair of the current gate twice, matching an independent
        # per-round reference path (spec 16.1 canonical_pair_r2 baseline).
        output = self._run(CanonicalTeacherPolicy())
        self.assertTrue(bool((output.rounds == 2).all()))
        executor = make_executor(self.model, "sparse_merged_v1", chunk=64, compute_dtype="float32")
        from eval.counterfactual import cache_last_ffn_input

        prefix = cache_last_ffn_input(self.model, self.input_ids, self.segment_ids)
        h0 = prefix.float().reshape(self.count, TINY_DIM).contiguous()
        rounds = torch.zeros(self.count, dtype=torch.int64)
        previous = torch.full((self.count, 2), -1, dtype=torch.int64)
        h = h0.clone()
        for round_index in range(2):
            observation = executor.observe(h, h0, rounds, previous, max_calls=2)
            pairs = canonical_pair(observation.z, executor.table)
            actions = pairs + 1
            self.assertTrue(bool((output.actions[:, :, round_index].reshape(-1) == actions).all()))
            h = executor.step_pair(h, observation, actions)
            rounds = rounds + 1
            previous = executor.table[pairs]
        reference = production_logits(self.model, h, chunk=64, compute_dtype="float32")
        self.assertTrue(torch.equal(output.final_hidden.reshape(self.count, TINY_DIM), h))
        self.assertTrue(torch.equal(output.logits.reshape(self.count, -1), reference))


if __name__ == "__main__":
    unittest.main()
