"""CPU, no downloads: policy math, real updates, isolation and artifact contracts."""
import copy
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from mixtral_lab.training import (ResidualPairRouter, RouterTrainingConfig,
    pair_actions, rloo_advantage, train_router, load_actor_checkpoint,
    load_actor_checkpoint_with_metadata, _sampled_scores)


IDENTITIES = {k: 'toy-'+k for k in ('model', 'tokenizer', 'train', 'selection', 'code')}


class ToyAdapter:
    def __init__(self):
        self.model = nn.Linear(3, 3).requires_grad_(False)
    def features(self, h):
        return h
    def native_pair(self, h):
        return torch.tensor([0, 1], device=h.device).expand(len(h), -1)
    def score_pairs(self, h, targets, pairs):
        # Intentionally differentiable inputs; trainer must stop reward gradients.
        return (pair_actions(pairs).float() / 28 + h.sum(-1) * 0).square()


class TrainingTests(unittest.TestCase):
    def test_checked_reuse_counts_order_and_fail_closed(self):
        class Counting(ToyAdapter):
            def __init__(self, perturb=False):
                super().__init__()
                self.rows = []
                self.perturb = perturb
            def score_pairs(self, h, targets, pairs):
                self.rows.append(len(h))
                value = super().score_pairs(h, targets, pairs)
                return value + (self.perturb if self.perturb and len(self.rows) == 3 else 0.)
        h, y = torch.ones(2, 3), torch.zeros(2, dtype=torch.long)
        # Native duplicates + nonnative duplicates, with different row orders.
        actions = torch.tensor([[0, 1, 1, 0], [2, 0, 2, 0]])
        cfg = RouterTrainingConfig(score_mode='deduplicate_checked_v1', score_atol=0., score_rtol=0.)
        for perturb in (0., .1, float('nan')):
            adapter = Counting(perturb)
            native = adapter.native_pair(h)
            native_ce = adapter.score_pairs(h, y, native)
            ce, work = _sampled_scores(adapter, h, y, native, native_ce, actions, cfg)
            expected = (actions.float() / 28).square()
            torch.testing.assert_close(ce, expected, atol=0, rtol=0)
            self.assertEqual(adapter.rows, [2, 8, 2])
            self.assertEqual(work['requested_rows'], 10)
            self.assertEqual(work['scored_rows'], 12)
            self.assertEqual(work['candidate_scored_rows'], 4)
            self.assertEqual(work['validation_rows'], 8)
            self.assertEqual(work['cache_hits'], 6)
            self.assertEqual(work['native_cache_hits'], 4)
            self.assertEqual(work['score_calls'], 3)
            self.assertEqual(work['candidate_used'], not bool(perturb))
            self.assertEqual(work['candidate_eligible'], not bool(perturb))
            json.dumps(work, allow_nan=False)
            self.assertEqual(len(work['absolute_errors']), 2)
        adapter = Counting()
        native = adapter.native_pair(h)
        ce, work = _sampled_scores(adapter, h, y, native,
            adapter.score_pairs(h, y, native), torch.zeros_like(actions), cfg)
        self.assertEqual(adapter.rows, [2, 8])
        self.assertEqual(work['candidate_scored_rows'], 2)
        self.assertEqual(work['cache_hits'], 8)

    def test_checked_reuse_preserves_toy_updates_and_rng(self):
        from dataclasses import replace
        h, y = torch.ones(8, 3, requires_grad=True), torch.zeros(8, dtype=torch.long)
        config = RouterTrainingConfig(updates=3, batch_contexts=8, group_size=16,
            epsilon=.8, learning_rate=.01, eval_interval=1, max_step_kl=1.)
        states, runs = [], []
        with tempfile.TemporaryDirectory() as temp:
            for mode in ('reference', 'deduplicate_checked_v1'):
                adapter = ToyAdapter()
                initial = copy.deepcopy(adapter.model.state_dict())
                rng_before = torch.get_rng_state().clone()
                actor, summary = train_router(adapter, h, y, lambda a: {'nll':1.},
                    replace(config, score_mode=mode), IDENTITIES, Path(temp)/mode)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
                self.assertIsNone(h.grad)
                self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))
                for key, value in adapter.model.state_dict().items():
                    self.assertTrue(torch.equal(value, initial[key]))
                states.append(copy.deepcopy(actor.state_dict()))
                runs.append(json.loads((Path(temp)/mode/'metrics.json').read_text()))
                self.assertEqual(summary['score_work_totals']['requested_rows'], 3*8*17)
            for key in states[0]:
                torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)
            for reference, checked in zip(*runs):
                self.assertTrue(checked['score_work']['candidate_used'])
                self.assertEqual({k:v for k,v in reference.items() if k != 'score_work'},
                                 {k:v for k,v in checked.items() if k != 'score_work'})

    def test_pair_mapping_and_native_initialization(self):
        pairs = torch.combinations(torch.arange(8), r=2)
        self.assertTrue(torch.equal(pair_actions(pairs.flip(-1)), torch.arange(28)))
        actor = ResidualPairRouter(3, epsilon=.8)
        x = torch.randn(28, 3)
        self.assertTrue(torch.equal(actor.greedy(x, pairs.flip(-1)), torch.arange(1, 29)))
        self.assertTrue(torch.allclose(actor.log_probs(x, pairs).exp().sum(-1), torch.ones(28)))

    def test_within_context_rloo_and_stop_gradient(self):
        rewards = torch.tensor([[1., 2., 6.], [100., 100., 100.]], requires_grad=True)
        advantage = rloo_advantage(rewards)
        self.assertTrue(torch.equal(advantage, torch.tensor([[-3., -1.5, 4.5], [0., 0., 0.]])))
        self.assertFalse(advantage.requires_grad)
        with self.assertRaises(ValueError):
            rloo_advantage(torch.ones(2, 1))

    def test_updates_isolation_best_zero_and_checkpoint_identity(self):
        adapter = ToyAdapter()
        initial = copy.deepcopy(adapter.model.state_dict())
        h = torch.ones(8, 3, requires_grad=True)
        config = RouterTrainingConfig(updates=3, batch_contexts=8, group_size=16,
            epsilon=.8, learning_rate=.01, eval_interval=1, max_step_kl=1.)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'run'
            actor, summary = train_router(adapter, h, torch.zeros(8, dtype=torch.long),
                lambda actor: {'nll': 1.+float(actor.weight.detach().abs().sum())}, config, IDENTITIES, output)
            self.assertGreater(float(actor.weight.detach().abs().sum()), 0)
            self.assertEqual(summary['successful_updates'], 3)
            self.assertEqual(summary['best_update'], 0)
            self.assertIsNone(h.grad)
            for k, v in adapter.model.state_dict().items():
                self.assertTrue(torch.equal(initial[k], v))
            self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))
            best = load_actor_checkpoint(output/'checkpoint_best.pt', IDENTITIES)
            self.assertEqual(float(best.weight.detach().abs().sum()), 0.)
            verified, metadata = load_actor_checkpoint_with_metadata(output/'checkpoint_best.pt', IDENTITIES)
            self.assertEqual(metadata['checkpoint_sha256'], hashlib.sha256((output/'checkpoint_best.pt').read_bytes()).hexdigest())
            self.assertEqual(metadata['binding_sha256'], summary['binding_sha256'])
            self.assertEqual(metadata['update'], 0)
            self.assertEqual(metadata['training_config']['epsilon'], config.epsilon)
            self.assertEqual(len(metadata['actor_state_sha256']), 64)
            self.assertTrue(torch.equal(verified.weight, best.weight))
            self.assertEqual(len(json.loads((output/'evaluations.json').read_text())), 4)
            metrics = json.loads((output/'metrics.json').read_text())
            self.assertGreater(metrics[0]['distinct_sampled_actions_mean'], 1)
            with self.assertRaises(ValueError):
                load_actor_checkpoint(output/'checkpoint_best.pt', dict(IDENTITIES, train='wrong'))
            payload = torch.load(output/'checkpoint_best.pt', weights_only=True)
            payload['binding']['config']['epsilon'] = .7
            torch.save(payload, output/'tampered.pt')
            with self.assertRaises(ValueError):
                load_actor_checkpoint(output/'tampered.pt', IDENTITIES)
            with self.assertRaises(FileExistsError):
                train_router(adapter, h, torch.zeros(8, dtype=torch.long), lambda a: {'nll':1.}, config, IDENTITIES, output)

    def test_real_tiny_hf_adapter_training_integration(self):
        from transformers import MixtralConfig, MixtralForCausalLM
        from mixtral_lab.adapter import MixtralAdapter
        torch.manual_seed(41)
        cfg = MixtralConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            num_local_experts=8, num_experts_per_tok=2, max_position_embeddings=32)
        cfg._attn_implementation = 'eager'
        cfg._experts_implementation = 'eager'
        adapter = MixtralAdapter(MixtralForCausalLM(cfg).eval(), head_chunk=4)
        ids = torch.tensor([[2, 3, 4, 5, 6]])
        h = adapter.capture(ids)[:, :-1].reshape(-1, 16)
        y = ids[:, 1:].reshape(-1)
        original = copy.deepcopy(adapter.model.state_dict())
        native = adapter.forward(ids).logits
        def selection(actor):
            logits = adapter.forward(ids, policy=actor).logits[:, :-1]
            return {'nll': float(torch.nn.functional.cross_entropy(logits.reshape(-1, 32), y))}
        config = RouterTrainingConfig(updates=2, batch_contexts=4, group_size=16,
            epsilon=.8, learning_rate=.01, max_step_kl=1., score_mode='deduplicate_checked_v1')
        with tempfile.TemporaryDirectory() as temp:
            actor, summary = train_router(adapter, h, y, selection, config, IDENTITIES, Path(temp)/'run')
            initial = load_actor_checkpoint(Path(temp)/'run/checkpoint_initial.pt', IDENTITIES)
            torch.testing.assert_close(adapter.forward(ids, policy=initial).logits, native, atol=0, rtol=0)
            self.assertEqual(summary['successful_updates'], 2)
            metrics = json.loads((Path(temp)/'run/metrics.json').read_text())
            for row in metrics:
                self.assertTrue(row['score_work']['candidate_eligible'])
                self.assertLess(row['score_work']['max_absolute_error'], 1e-5)
            self.assertGreater(float(actor.weight.detach().abs().sum()), 0.)
        for key, value in adapter.model.state_dict().items():
            self.assertTrue(torch.equal(original[key], value))
        self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))

    def test_config_rejects_invalid(self):
        for kwargs in ({'group_size':1}, {'epsilon':1.}, {'epsilon':float('nan')},
                       {'score_mode':'unchecked'}, {'score_atol':-1}, {'score_rtol':float('nan')}, {'updates':True}, {'max_step_kl':0.}, {'learning_rate':float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RouterTrainingConfig(**kwargs).validate()

    def test_kl_rollback(self):
        adapter = ToyAdapter()
        config = RouterTrainingConfig(updates=2, batch_contexts=8, group_size=16,
            epsilon=.8, learning_rate=1., max_step_kl=1e-10)
        with tempfile.TemporaryDirectory() as temp:
            actor, result = train_router(adapter, torch.ones(8,3), torch.zeros(8,dtype=torch.long),
                lambda a: {'nll':1.}, config, IDENTITIES, Path(temp)/'run')
            self.assertEqual(result['status'], 'stopped_step_kl')
            self.assertEqual(result['successful_updates'], 0)
            self.assertEqual(float(actor.weight.detach().abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
