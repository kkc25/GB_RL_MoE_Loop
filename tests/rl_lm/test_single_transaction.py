"""Single-policy rollback invariants, independent of pair-policy semantics."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.rl_lm_single_train import UpdateRejected
from rl_lm.trainer import categorical_kl
from tests.rl_lm.test_resume import compare_optimizer_state
from tests.rl_lm.test_single_rloo import make_trainer, toy_config_dict


def config(path):
    data = toy_config_dict(path)
    data['optimization']['learning_rate'] = 0.05
    return data


def simulated_step(trainer, records, scale):
    actor = copy.deepcopy(trainer.actor)
    optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],
                                 lr=0.0, betas=trainer.betas, eps=trainer.eps,
                                 weight_decay=trainer.weight_decay)
    optimizer.load_state_dict(copy.deepcopy(trainer.optimizer.state_dict()))
    for cloned, live in zip(actor.parameters(), trainer.actor.parameters()):
        if live.grad is not None:
            cloned.grad = live.grad.detach().clone()
    for group in optimizer.param_groups:
        group['lr'] = trainer.base_lr(trainer.successful_updates + 1) * scale
    optimizer.step()
    with torch.no_grad():
        new = actor(records.features, records.reference, records.stop_allowed)
    kl = categorical_kl(records.old_log_probs_all, new)
    return float(kl.mean()), float(kl.max()), actor, optimizer


class SingleTransactionTests(unittest.TestCase):
    def test_retry_matches_independent_half_step_and_retains_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = make_trainer(config(Path(tmp) / 'run'), stop_after=10)
            trainer._update()
            info = trainer._begin_update()
            full_mean, full_max, _, _ = simulated_step(trainer, info.records, 1.0)
            half_mean, half_max, actor, optimizer = simulated_step(trainer, info.records, 0.5)
            self.assertGreater(full_mean, half_mean)
            self.assertGreater(full_max, half_max)
            trainer.max_kl_mean = (full_mean + half_mean) / 2
            trainer.max_kl_state = (full_max + half_max) / 2
            grads = [None if p.grad is None else p.grad.clone() for p in trainer.actor.parameters()]
            attempts = trainer.ledger.optimizer_attempts
            with patch.object(trainer, '_begin_update', return_value=info):
                step = trainer._update()
            self.assertEqual(step['retry_count'], 1)
            self.assertEqual(trainer.ledger.optimizer_attempts, attempts + 2)
            self.assertEqual((trainer.lr_scale, trainer.lr_backoffs), (0.5, 1))
            for name, value in trainer.actor.state_dict().items():
                self.assertTrue(torch.equal(value, actor.state_dict()[name]), name)
            compare_optimizer_state(trainer.optimizer.state_dict(), optimizer.state_dict(), 'retry')
            for p, grad in zip(trainer.actor.parameters(), grads):
                if grad is not None:
                    self.assertTrue(torch.equal(p.grad, grad))
            trainer.max_kl_mean = trainer.max_kl_state = 1.0
            third = trainer._update()
            self.assertEqual(third['lr'], trainer.base_lr(3) * 0.5)
            self.assertEqual(third['retry_count'], 0)
            self.assertEqual(trainer.lr_backoffs, 1)

    def test_rejection_restores_state_retains_costs_and_checkpoint(self):
        for budget, attempts, reason in [(4, 2, 'kl_retry_exhausted'), (0, 1, 'kl_budget_exhausted')]:
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'run'
                data = config(path)
                data['optimization']['max_lr_backoffs_per_run'] = budget
                trainer = make_trainer(data, stop_after=10)
                trainer._update()
                trainer._save_checkpoint('latest')
                actor = copy.deepcopy(trainer.actor.state_dict())
                optimizer = copy.deepcopy(trainer.optimizer.state_dict())
                ledger = trainer.ledger.to_dict()
                trainer.max_kl_mean = trainer.max_kl_state = 0.0
                with self.assertRaises(UpdateRejected):
                    trainer._update()
                for name, value in trainer.actor.state_dict().items():
                    self.assertTrue(torch.equal(value, actor[name]), name)
                compare_optimizer_state(trainer.optimizer.state_dict(), optimizer, 'rejected')
                self.assertEqual((trainer.lr_scale, trainer.lr_backoffs), (1.0, 0))
                self.assertEqual(trainer.successful_updates, 1)
                self.assertEqual(trainer.ledger.optimizer_attempts, ledger['optimizer_attempts'] + attempts)
                self.assertEqual(trainer.ledger.rollout_episodes, ledger['rollout_episodes'] + trainer.batch * trainer.groups)
                self.assertGreater(trainer.ledger.physical_expert_rows, ledger['physical_expert_rows'])
                self.assertGreater(trainer.ledger.physical_head_rows, ledger['physical_head_rows'])
                self.assertEqual(json.loads((path / 'update_rejected.json').read_text())['reason'], reason)
                saved = torch.load(path / 'checkpoint_latest.pt', weights_only=False)
                self.assertEqual(saved['saved_at_update'], 1)
