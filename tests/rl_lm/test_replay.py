"""T15-T18: replay consistency, gradient isolation, RNG streams, microbatch.

T15 replay on the pre-update actor reproduces the rollout's old_log_joint
(spec 11.1 tolerance FP32 atol=rtol=1e-5), and tampering with the recorded
mask / probability rule / sampled actions must break that equality.
T16 only actor parameters receive gradients; the frozen LM keeps grad=None
and a bitwise parameter hash.  T17 deterministic SHA256+Philox streams are
invariant to batch order, subsetting and neighbours' early STOP, uniforms
stay in [0,1), and the inverse-CDF boundary behaviour matches spec
appendix A.  T18 microbatching the replay at 32 contexts (last one short)
accumulates the same gradients as the full-batch backward, using the global
B*G reduction of spec 11.3.

NOTE (spec deviation, flagged to the coordinator): the spec places T17 in
``tests/rl_lm/test_rng.py``; this task card's editable file set does not
include that file, so T17 lives here as ``RngStreamTests`` and can be moved
verbatim.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import unittest

import numpy as np
import torch

from rl_lm.actor import PairActor
from rl_lm.environment import LastFFNEnvironment, inverse_cdf_sample
from rl_lm.executor import FrozenLastFFNExecutor
from rl_lm.objectives import policy_gradient_loss, rloo_advantage
from rl_lm.replay import (
    iter_context_microbatches,
    microbatch_pg_loss,
    replay_log_joint,
)
from rl_lm.rng import make_uniform_table, sample_from_log_probs
from rl_lm.types import ContextBatch, DecisionRecords
from tests.rl_lm.tiny_lm import TINY_DIM, TINY_VOCAB, build_tiny_model

FEATURE_DIM = 2 * TINY_DIM + 2 * 8 + 4


def build_environment(max_calls=2):
    model = build_tiny_model()
    executor = FrozenLastFFNExecutor(model, chunk=64, compute_dtype="float32")
    environment = LastFFNEnvironment(model, max_calls=max_calls, executor=executor, head_chunk=64)
    return model, environment


def make_contexts(batch, seed=19):
    torch.manual_seed(seed)
    return ContextBatch(
        h0=torch.randn(batch, TINY_DIM),
        targets=torch.randint(0, TINY_VOCAB, (batch,)),
        ref_ce=torch.rand(batch) + 0.5,
        episode_ids=tuple(f"toy-v1:seq-{i}:pos-{i}" for i in range(batch)),
    )


def model_hash(model):
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class ReplayConsistencyTests(unittest.TestCase):
    """T15: pre-update replay equals rollout old log-joint; tampering fails."""

    def setUp(self):
        self.model, self.environment = build_environment(max_calls=2)
        torch.manual_seed(17)
        self.actor = PairActor(FEATURE_DIM, train_stop=True)
        self.batch, self.groups = 3, 4
        contexts = make_contexts(self.batch)
        uniforms = make_uniform_table(17, 0, contexts.episode_ids, self.groups, 2)
        self.rollout, self.records = self.environment.rollout(contexts, self.actor, uniforms)

    def test_replay_matches_rollout_before_update(self):
        joint = replay_log_joint(self.actor, self.records, self.batch, self.groups)
        max_diff = float((joint.detach() - self.rollout.old_log_joint).abs().max())
        print(f"\nT15 max|replay-rollout|={max_diff:.3e}")
        torch.testing.assert_close(joint, self.rollout.old_log_joint, atol=1e-5, rtol=1e-5)

    def _tampered(self, **overrides):
        fields = dict(
            features=self.records.features,
            pair_reference=self.records.pair_reference,
            stop_allowed=self.records.stop_allowed,
            actions=self.records.actions,
            flat_episode_ids=self.records.flat_episode_ids,
            old_log_probs_all=self.records.old_log_probs_all,
        )
        fields.update(overrides)
        return DecisionRecords(**fields)

    def test_flipped_stop_mask_must_fail(self):
        tampered = self._tampered(stop_allowed=~self.records.stop_allowed)
        joint = replay_log_joint(self.actor, tampered, self.batch, self.groups)
        self.assertFalse(
            bool(torch.isclose(joint, self.rollout.old_log_joint, atol=1e-5, rtol=1e-5).all()),
            "flipping stop_allowed must break replay consistency",
        )

    def test_changed_probability_rule_must_fail(self):
        uniform_q = torch.full_like(self.records.pair_reference, 1.0 / 28.0)
        tampered = self._tampered(pair_reference=uniform_q)
        joint = replay_log_joint(self.actor, tampered, self.batch, self.groups)
        self.assertFalse(
            bool(torch.isclose(joint, self.rollout.old_log_joint, atol=1e-5, rtol=1e-5).all()),
            "changing the reference probabilities must break replay consistency",
        )

    def test_changed_action_must_fail(self):
        actions = self.records.actions.clone()
        actions[0] = actions[0] % 28 + 1  # still a legal pair action for that row
        tampered = self._tampered(actions=actions)
        joint = replay_log_joint(self.actor, tampered, self.batch, self.groups)
        self.assertFalse(
            bool(torch.isclose(joint, self.rollout.old_log_joint, atol=1e-5, rtol=1e-5).all()),
            "changing a sampled action must break replay consistency",
        )


class GradientIsolationTests(unittest.TestCase):
    """T16: actor is the only gradient owner; the frozen LM never changes."""

    def test_only_actor_gets_gradients(self):
        model, environment = build_environment(max_calls=2)
        torch.manual_seed(17)
        actor = PairActor(FEATURE_DIM, train_stop=True)
        batch, groups = 3, 4
        contexts = make_contexts(batch)
        uniforms = make_uniform_table(17, 0, contexts.episode_ids, groups, 2)
        rollout, records = environment.rollout(contexts, actor, uniforms)
        advantage = rloo_advantage(rollout.returns)
        lm_hash_before = model_hash(model)

        joint = replay_log_joint(actor, records, batch, groups)
        loss = policy_gradient_loss(joint, advantage)
        actor.zero_grad(set_to_none=True)
        loss.backward()

        grads = [(name, p.grad) for name, p in actor.named_parameters() if p.requires_grad]
        self.assertTrue(all(g is not None for _, g in grads))
        self.assertTrue(all(bool(torch.isfinite(g).all()) for _, g in grads))
        nonzero = [name for name, g in grads if bool((g != 0).any())]
        self.assertTrue(nonzero, "at least one actor parameter must have a nonzero grad")

        for name, parameter in model.named_parameters():
            self.assertIsNone(parameter.grad, f"frozen LM parameter {name} got a grad")
        self.assertEqual(model_hash(model), lm_hash_before, "backward changed LM weights")

        actor_hash_before = hashlib.sha256(
            b"".join(p.detach().numpy().tobytes() for p in actor.parameters())
        ).hexdigest()
        with torch.no_grad():
            for parameter in actor.parameters():
                if parameter.grad is not None:
                    parameter -= 0.05 * parameter.grad
        actor_hash_after = hashlib.sha256(
            b"".join(p.detach().numpy().tobytes() for p in actor.parameters())
        ).hexdigest()
        self.assertNotEqual(actor_hash_before, actor_hash_after, "actor update had no effect")
        self.assertEqual(model_hash(model), lm_hash_before, "actor step changed LM weights")


class RngStreamTests(unittest.TestCase):
    """T17: deterministic SHA256+Philox streams and inverse-CDF boundary."""

    EPISODES = ("corpus-v1:seq-3:pos-11", "corpus-v1:seq-9:pos-2", "corpus-v1:seq-0:pos-7")

    def test_same_ids_give_identical_table(self):
        first = make_uniform_table(17, 0, self.EPISODES, 4, 2)
        second = make_uniform_table(17, 0, self.EPISODES, 4, 2)
        self.assertTrue(torch.equal(first, second))

    def test_rows_follow_episode_id_not_batch_position(self):
        full = make_uniform_table(17, 0, self.EPISODES, 4, 2)
        permuted = make_uniform_table(17, 0, (self.EPISODES[2], self.EPISODES[0], self.EPISODES[1]), 4, 2)
        self.assertTrue(torch.equal(full[2], permuted[0]))
        self.assertTrue(torch.equal(full[0], permuted[1]))
        subset = make_uniform_table(17, 0, (self.EPISODES[1],), 4, 2)  # microbatch of one
        self.assertTrue(torch.equal(full[1], subset[0]))

    def test_stream_depends_on_seed_update_episode_and_group(self):
        base = make_uniform_table(17, 0, self.EPISODES, 4, 2)
        self.assertFalse(torch.equal(base, make_uniform_table(18, 0, self.EPISODES, 4, 2)))
        self.assertFalse(torch.equal(base, make_uniform_table(17, 1, self.EPISODES, 4, 2)))
        self.assertFalse(
            torch.equal(base, make_uniform_table(17, 0, ("corpus-v1:seq-3:pos-12",) + self.EPISODES[1:], 4, 2))
        )
        self.assertFalse(torch.equal(base[:, 0], base[:, 1]))

    def test_uniforms_stay_in_unit_interval(self):
        table = make_uniform_table(23, 7, self.EPISODES, 4, 3)
        self.assertEqual(table.dtype, torch.float64)
        self.assertTrue(bool((table >= 0).all()) and bool((table < 1).all()))

    def test_table_is_stable_across_processes(self):
        code = (
            "import hashlib;"
            "from rl_lm.rng import make_uniform_table;"
            "t = make_uniform_table(17, 0, ('corpus-v1:seq-3:pos-11',), 4, 2);"
            "print(hashlib.sha256(t.numpy().tobytes()).hexdigest())"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True, cwd=_REPO_ROOT
        )
        local = hashlib.sha256(make_uniform_table(17, 0, (self.EPISODES[0],), 4, 2).numpy().tobytes()).hexdigest()
        self.assertEqual(out.stdout.strip(), local)

    def test_neighbour_early_stop_keeps_this_episodes_actions(self):
        model, environment = build_environment(max_calls=2)
        torch.manual_seed(17)
        actor = PairActor(FEATURE_DIM, train_stop=True)
        contexts = make_contexts(2)
        groups = 2
        base = make_uniform_table(17, 0, contexts.episode_ids, groups, 2)
        stopped = base.clone()
        stopped[1, :, 0] = 0.3  # context 1: pair in round 0 (STOP illegal there)
        stopped[1, :, 1] = 0.01  # context 1: voluntary STOP in round 1 (hazard 0.5)
        continued = base.clone()
        continued[1, :, 0] = 0.3
        continued[1, :, 1] = 0.99  # context 1: keeps going
        rollout_a, records_a = environment.rollout(contexts, actor, stopped)
        rollout_b, records_b = environment.rollout(contexts, actor, continued)

        self.assertTrue(bool((rollout_a.rounds[1] == 1).all()), "neighbour must stop early")
        self.assertTrue(bool((rollout_b.rounds[1] == 2).all()), "neighbour must continue")
        keep_a = records_a.flat_episode_ids < groups
        keep_b = records_b.flat_episode_ids < groups
        self.assertTrue(torch.equal(records_a.flat_episode_ids[keep_a], records_b.flat_episode_ids[keep_b]))
        self.assertTrue(torch.equal(records_a.actions[keep_a], records_b.actions[keep_b]))
        self.assertTrue(torch.equal(rollout_a.terminal_ce[0], rollout_b.terminal_ce[0]))

    def test_inverse_cdf_boundary_and_illegal_stop(self):
        # r=0 row: STOP has -inf log-prob (appendix A reference behaviour).
        log_probs = torch.tensor([[-torch.inf] + [float(np.log(1.0 / 28.0))] * 28])
        u_zero = torch.tensor([0.0], dtype=torch.float64)
        u_top = torch.tensor([np.nextafter(1.0, 0.0)], dtype=torch.float64)
        self.assertEqual(int(sample_from_log_probs(log_probs, u_zero)[0]), 1)
        self.assertEqual(int(sample_from_log_probs(log_probs, u_top)[0]), 28)
        # r>=1 row: hazard 0.5, uniform pairs.
        allowed = torch.tensor([[float(np.log(0.5))] + [float(np.log(0.5 / 28.0))] * 28])
        self.assertEqual(int(sample_from_log_probs(allowed, torch.tensor([0.25], dtype=torch.float64))[0]), 0)
        self.assertGreaterEqual(int(sample_from_log_probs(allowed, torch.tensor([0.75], dtype=torch.float64))[0]), 1)

    def test_sampler_matches_environment_reference(self):
        torch.manual_seed(3)
        logits = torch.randn(64, 29)
        logits[:, 0] = -torch.inf
        log_probs = torch.log_softmax(logits.masked_fill(torch.isinf(logits), -torch.inf), dim=-1)
        uniforms = torch.rand(64, dtype=torch.float64)
        self.assertTrue(torch.equal(sample_from_log_probs(log_probs, uniforms), inverse_cdf_sample(log_probs, uniforms)))

    def test_sampler_rejects_out_of_range_or_rounded_uniforms(self):
        log_probs = torch.zeros(2, 29)
        with self.assertRaises(ValueError):
            sample_from_log_probs(log_probs, torch.tensor([0.5, 1.0], dtype=torch.float64))
        with self.assertRaises(ValueError):
            sample_from_log_probs(log_probs, torch.tensor([0.5, -0.1], dtype=torch.float64))
        with self.assertRaises(ValueError):
            sample_from_log_probs(log_probs, torch.rand(2))  # float32 is forbidden


class MicrobatchTests(unittest.TestCase):
    """T18: 32-context microbatches accumulate the full-batch gradient."""

    def test_microbatch_equals_full_batch(self):
        model, environment = build_environment(max_calls=2)
        torch.manual_seed(17)
        actor = PairActor(FEATURE_DIM, train_stop=True)
        batch, groups, micro = 72, 4, 32
        contexts = make_contexts(batch)
        uniforms = make_uniform_table(17, 0, contexts.episode_ids, groups, 2)
        rollout, records = environment.rollout(contexts, actor, uniforms)
        advantage = rloo_advantage(rollout.returns)
        self.assertGreater(float(advantage.abs().sum()), 0.0, "test needs a nonzero signal")

        joint = replay_log_joint(actor, records, batch, groups)
        full_loss = policy_gradient_loss(joint, advantage)
        actor.zero_grad(set_to_none=True)
        full_loss.backward()
        grads_full = {name: p.grad.clone() for name, p in actor.named_parameters() if p.grad is not None}

        actor.zero_grad(set_to_none=True)
        sizes = []
        micro_loss_total = torch.zeros(())
        for context_ids in iter_context_microbatches(batch, micro):
            sizes.append(int(context_ids.numel()))
            micro_loss = microbatch_pg_loss(actor, records, advantage, batch, groups, context_ids)
            micro_loss.backward()
            micro_loss_total = micro_loss_total + micro_loss.detach()
        self.assertEqual(sizes, [32, 32, 8], "must include a short final microbatch")

        torch.testing.assert_close(micro_loss_total, full_loss.detach(), rtol=1e-6, atol=1e-7)
        max_diff = 0.0
        for name, parameter in actor.named_parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad)
                continue
            self.assertIsNotNone(parameter.grad, name)
            diff = float((parameter.grad - grads_full[name]).abs().max())
            max_diff = max(max_diff, diff)
            torch.testing.assert_close(parameter.grad, grads_full[name], rtol=1e-5, atol=1e-7)
        print(f"\nT18 max|microbatch-full| grad diff={max_diff:.3e}")


_REPO_ROOT = __file__.rsplit("/tests/", 1)[0]

if __name__ == "__main__":
    unittest.main()
