"""CPU, no downloads: policy math, real updates, isolation and artifact contracts."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from mixtral_lab.training import (ResidualPairRouter, RouterTrainingConfig,
    pair_actions, rloo_advantage, train_router, load_actor_checkpoint)


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
            epsilon=.8, learning_rate=.01, max_step_kl=1.)
        with tempfile.TemporaryDirectory() as temp:
            actor, summary = train_router(adapter, h, y, selection, config, IDENTITIES, Path(temp)/'run')
            initial = load_actor_checkpoint(Path(temp)/'run/checkpoint_initial.pt', IDENTITIES)
            torch.testing.assert_close(adapter.forward(ids, policy=initial).logits, native, atol=0, rtol=0)
            self.assertEqual(summary['successful_updates'], 2)
            self.assertGreater(float(actor.weight.detach().abs().sum()), 0.)
        for key, value in adapter.model.state_dict().items():
            self.assertTrue(torch.equal(original[key], value))
        self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))

    def test_config_rejects_invalid(self):
        for kwargs in ({'group_size':1}, {'epsilon':1.}, {'epsilon':float('nan')},
                       {'updates':True}, {'max_step_kl':0.}, {'learning_rate':float('inf')}):
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
