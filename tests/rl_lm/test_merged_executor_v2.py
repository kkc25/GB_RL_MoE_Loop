"""Bounded CPU fixtures for the opt-in merged backend (not a GPU speed gate)."""
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from eval.counterfactual import cache_last_ffn_input
from rl_lm.environment import LastFFNEnvironment
from rl_lm.executor import (
    CANONICAL_BACKEND, SPARSE_BACKEND, MERGED_BACKEND,
    FrozenLastFFNExecutor, SparseMergedExecutor, SparseMergedV2Executor,
    make_executor,
)
from rl_lm.inference import CanonicalTeacherPolicy, forward_with_last_ffn_policy
from rl_lm.ledger import RLLedger
from rl_lm.readout import terminal_ce
from rl_lm.selector import canonical_pair
from rl_lm.types import ContextBatch
from tests.rl_lm.test_online_parity import ScriptedPolicy
from tests.rl_lm.test_transition import manual_pair_step
from tests.rl_lm.tiny_lm import TINY_DIM, build_tiny_model


class MergedExecutorTests(unittest.TestCase):
    def setUp(self):
        self.model = build_tiny_model()
        torch.manual_seed(43)

    def executor(self, backend=MERGED_BACKEND, **kwargs):
        return make_executor(self.model, backend, compute_dtype="float32", **kwargs)

    def observe(self, executor, h):
        return executor.observe(h, h, torch.zeros(len(h), dtype=torch.int64),
                                torch.full((len(h), 2), -1, dtype=torch.int64), max_calls=2)

    def test_backend_identities_remain_distinct(self):
        for name, cls in ((CANONICAL_BACKEND, FrozenLastFFNExecutor),
                          (SPARSE_BACKEND, SparseMergedExecutor),
                          (MERGED_BACKEND, SparseMergedV2Executor)):
            self.assertIs(type(self.executor(name)), cls)
            self.assertEqual(self.executor(name).backend, name)
        with self.assertRaises(ValueError):
            self.executor("sparse_merged")

    def test_config_opt_in_does_not_change_training_or_default(self):
        from rl_lm.config import RLConfig
        path = Path(__file__).resolve().parents[2] / "configs/rl_lm/pair_r1.json"
        original = RLConfig.load(path).validate()
        self.assertEqual(original.get("environment.deployment_backend"), SPARSE_BACKEND)
        data = original.to_dict()
        data["environment"]["deployment_backend"] = MERGED_BACKEND
        candidate = RLConfig.from_dict(data).validate()
        self.assertNotEqual(candidate.sha256, original.sha256)
        data["environment"]["training_backend"] = MERGED_BACKEND
        with self.assertRaisesRegex(ValueError, "canonical backend"):
            RLConfig.from_dict(data).validate()
        data["environment"]["training_backend"] = CANONICAL_BACKEND
        data["environment"]["deployment_backend"] = "unknown"
        with self.assertRaisesRegex(ValueError, "deployment backend"):
            RLConfig.from_dict(data).validate()

    def test_all_pairs_sizes_noncontiguous_and_skew(self):
        # Existing FP32 contract tolerance; no universal BF16/GPU bound implied.
        for count in (0, 1, 28, 63, 64, 65):
            for skew in (False, True):
                with self.subTest(count=count, skew=skew):
                    h = torch.randn(count, TINY_DIM * 2)[:, ::2]
                    executor = self.executor()
                    obs = self.observe(executor, h)
                    actions = torch.ones(count, dtype=torch.int64) if skew else torch.arange(count) % 28 + 1
                    actual = executor.step_pair(h, obs, actions)
                    for backend in (CANONICAL_BACKEND, SPARSE_BACKEND):
                        expected = self.executor(backend).step_pair(h, obs, actions)
                        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                    for row in range(min(count, 28)):
                        pair = tuple(executor.table[actions[row] - 1].tolist())
                        torch.testing.assert_close(actual[row:row+1], manual_pair_step(self.model, h[row:row+1], pair), atol=1e-5, rtol=1e-5)

    def test_expert_buckets_really_merge_slots_and_ledger(self):
        h = torch.randn(28, TINY_DIM)
        actions = torch.arange(1, 29)
        for backend, expected_linears in ((SPARSE_BACKEND, 42), (MERGED_BACKEND, 24)):
            ledger = RLLedger()
            executor = self.executor(backend, ledger=ledger)
            obs = self.observe(executor, h)
            with patch("rl_lm.executor.F.linear", wraps=F.linear) as linear:
                executor.step_pair(h, obs, actions)
            self.assertEqual(linear.call_count, expected_linears)
            self.assertEqual(ledger.physical_expert_rows, 56)

    def test_pre_gamma_update_reuses_explicit_observation(self):
        executor = self.executor(gamma=0.0)
        h = torch.randn(28, TINY_DIM)
        obs = self.observe(executor, h)
        actions = torch.arange(1, 29)
        pairs = executor.table[actions - 1]
        weights = F.softmax(obs.z.gather(1, pairs), dim=-1)
        with patch.object(executor.block.moe.gate, "forward", side_effect=AssertionError("gate recomputed")):
            update = executor.pair_update(obs.n, pairs, weights)
            actual = executor.step_pair(h, obs, actions)
        self.assertTrue(torch.equal(actual, h))
        self.assertGreater(float(update.abs().sum()), 0)
        self.assertFalse(update.requires_grad)
        # Row permutation and changed bucket composition preserve FP32 math.
        permutation = torch.randperm(28)
        torch.testing.assert_close(executor.pair_update(obs.n[permutation], pairs[permutation], weights[permutation]), update[permutation], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(executor.pair_update(obs.n[:1], pairs[:1], weights[:1]), update[:1], atol=1e-5, rtol=1e-5)

    def test_invalid_actions_and_routing_rejected(self):
        executor = self.executor()
        h = torch.randn(1, TINY_DIM)
        obs = self.observe(executor, h)
        for action in (-1, 0, 29):
            with self.assertRaises(ValueError):
                executor.step_pair(h, obs, torch.tensor([action]))
        for pair in ((0, 0), (2, 1), (-1, 1), (0, 8)):
            with self.assertRaises(ValueError):
                executor.pair_update(obs.n, torch.tensor([pair]), torch.full((1, 2), 0.5))

    def online(self, policy, backend=MERGED_BACKEND, max_calls=2):
        return forward_with_last_ffn_policy(self.model, policy, self.tokens, self.segments,
            backend=backend, max_calls=max_calls, compute_dtype="float32", return_trace=True)

    def test_closed_loop_actions_stop_hidden_nll_and_replay(self):
        self.tokens = torch.randint(0, 64, (1, 8))
        self.segments = torch.zeros_like(self.tokens)
        targets = torch.randint(0, 64, (8,))
        script = [5, [0, 9, 0, 9, 0, 28, 0, 1]]
        actual = self.online(ScriptedPolicy(script, 8))
        for backend in (CANONICAL_BACKEND, SPARSE_BACKEND):
            expected = self.online(ScriptedPolicy(script, 8), backend)
            self.assertTrue(torch.equal(actual.actions, expected.actions))
            self.assertTrue(torch.equal(actual.rounds, expected.rounds))
            self.assertTrue(torch.equal(actual.decision_mask, expected.decision_mask))
            torch.testing.assert_close(actual.final_hidden, expected.final_hidden, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(F.cross_entropy(actual.logits.reshape(8, -1), targets), F.cross_entropy(expected.logits.reshape(8, -1), targets), atol=1e-5, rtol=1e-5)
        one = self.online(ScriptedPolicy([5], 8), max_calls=1)
        self.assertTrue(torch.equal(actual.final_hidden[:, ::2], one.final_hidden[:, ::2]))
        with torch.no_grad():
            h0 = cache_last_ffn_input(self.model, self.tokens, self.segments).float().reshape(8, TINY_DIM)
        contexts = ContextBatch(h0=h0, targets=targets, ref_ce=torch.ones(8), episode_ids=tuple(f"v2/{i}" for i in range(8)))
        env = LastFFNEnvironment(self.model, max_calls=2, executor=self.executor(), head_chunk=64)
        rollout, records = env.rollout(contexts, ScriptedPolicy(script, 8), torch.zeros(8, 1, 2, dtype=torch.float64))
        self.assertTrue(torch.equal(records.actions, actual.actions.reshape(8, 2).T.reshape(-1)))
        self.assertTrue(torch.equal(rollout.rounds[:, 0], actual.rounds.reshape(-1)))
        # Compare production BF16 head to the same head, not diagnostic FP32 NLL.
        expected_ce = terminal_ce(self.model, actual.final_hidden.reshape(8, TINY_DIM), targets)
        torch.testing.assert_close(rollout.terminal_ce[:, 0], expected_ce, atol=0, rtol=0)

    def test_closed_loop_teacher_exact_and_near_ties(self):
        self.tokens = torch.randint(0, 64, (1, 8))
        self.segments = torch.zeros_like(self.tokens)
        gate = self.model.blocks[-1].moe.gate.weight
        for perturbation in (0.0, 1e-7):
            with torch.no_grad():
                gate.zero_()
                gate[7, 0] = perturbation
            expected = self.online(CanonicalTeacherPolicy(), CANONICAL_BACKEND)
            actual = self.online(CanonicalTeacherPolicy())
            self.assertTrue(torch.equal(actual.actions, expected.actions))
            torch.testing.assert_close(actual.logits, expected.logits, atol=1e-5, rtol=1e-5)
            if perturbation == 0:
                self.assertTrue(bool((actual.actions == 1).all()))
        # Independent selector ties and representable near ties.
        z = torch.zeros(2, 8)
        z[1, 7] = 1e-7
        self.assertEqual(canonical_pair(z).tolist(), [0, 6])


if __name__ == "__main__":
    unittest.main()
