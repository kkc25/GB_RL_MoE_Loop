"""T20: the single-step KL rollback transaction (spec 11.4).

Artificially oversized steps are forced by tightening the KL thresholds.
Verified properties:

- retry succeeds: the first attempt at the scheduled LR violates the
  threshold, the actor/optimizer/scheduler deepcopy snapshot is fully
  restored, the LR multiplier is halved and *persists* into later updates,
  exactly one retry happens on the same old parameters and gradients, and
  the resulting actor equals an independently simulated halved-LR step;
- retry exhausted: with an impossible threshold the update is rejected after
  exactly one retry (two attempts), the pre-update state is restored
  bitwise, no half-updated checkpoint is written, and the rollout/optimizer
  resource costs stay in the ledger;
- run-level backoff budget: with ``max_lr_backoffs_per_run = 0`` the first
  violation terminates immediately without consuming a retry.

The halved-LR prediction uses a deepcopy of the live actor plus a fresh
AdamW loaded with the live optimizer state and the live gradients, stepped
on a clone — the real trainer path is then checked against it bitwise.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from rl_lm.trainer import Trainer, UpdateRejected, categorical_kl
from tests.rl_lm.test_resume import compare_optimizer_state, make_trainer, toy_config_dict


def transaction_config_dict(workdir) -> dict:
    """Toy config whose steps have a measurable KL (peak LR above the default)."""
    data = toy_config_dict(workdir)
    data["optimization"]["learning_rate"] = 0.05
    return data


def clone_step_kl(trainer: Trainer, records, scale: float):
    """Simulate the pending update on a deepcopy at ``scale * base_lr``."""
    actor = copy.deepcopy(trainer.actor)
    params = [p for p in actor.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=0.0, betas=trainer.betas, eps=trainer.eps, weight_decay=trainer.weight_decay
    )
    optimizer.load_state_dict(copy.deepcopy(trainer.optimizer.state_dict()))
    for cloned, live in zip(actor.parameters(), trainer.actor.parameters()):
        if live.grad is not None:
            cloned.grad = live.grad.detach().clone()
    lr = trainer.base_lr(trainer.successful_updates + 1) * scale
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    with torch.no_grad():
        new_log_probs = actor(records.features, records.pair_reference, records.stop_allowed)
    kl = categorical_kl(records.old_log_probs_all, new_log_probs)
    return float(kl.mean()), float(kl.max()), actor


class UpdateTransactionTests(unittest.TestCase):
    def test_retry_success_halves_lr_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = make_trainer(transaction_config_dict(Path(tmp) / "run"), stop_after=10)
            first = trainer._update()  # update 1 with default thresholds
            self.assertEqual(first["retry_count"], 0)

            info = trainer._begin_update()  # update 2, gradients ready
            kl_full_mean, kl_full_max, _ = clone_step_kl(trainer, info.records, 1.0)
            kl_half_mean, kl_half_max, half_actor = clone_step_kl(trainer, info.records, 0.5)
            self.assertGreater(kl_full_mean, 0.0, "test needs a nonzero step KL")
            self.assertGreater(kl_full_mean, kl_half_mean)
            self.assertGreater(kl_full_max, kl_half_max)
            # First attempt (scale 1.0) violates, retry (scale 0.5) passes.
            trainer.max_kl_mean = (kl_full_mean + kl_half_mean) / 2.0
            trainer.max_kl_state = (kl_full_max + kl_half_max) / 2.0

            attempts_before = trainer.ledger.optimizer_attempts
            step = trainer._guarded_step(info.records)
            self.assertEqual(step["attempts"], 2, "exactly one retry")
            self.assertEqual(trainer.ledger.optimizer_attempts, attempts_before + 2)
            self.assertEqual(trainer.lr_scale, 0.5)
            self.assertEqual(trainer.lr_backoffs, 1)
            self.assertAlmostEqual(step["lr"], trainer.base_lr(2) * 0.5)
            self.assertAlmostEqual(step["kl_mean"], kl_half_mean, places=12)
            for name, parameter in trainer.actor.state_dict().items():
                self.assertTrue(
                    torch.equal(parameter, half_actor.state_dict()[name]),
                    f"retry outcome does not match the simulated halved step at {name}",
                )
            trainer._commit_update(info, step)
            entry = trainer._metrics_entry(info, step)
            trainer._append_metrics(entry)

            # LR scale persists: update 3 must run at base_lr(3) * 0.5.
            third = trainer._update()
            self.assertAlmostEqual(third["lr"], trainer.base_lr(3) * 0.5)
            self.assertEqual(third["retry_count"], 0)
            self.assertEqual(trainer.lr_backoffs, 1)
            self.assertEqual(trainer.ledger.rollout_episodes, 3 * trainer.batch * trainer.groups)
            print(
                f"\nT20 retry-success: KL full={kl_full_mean:.3e} half={kl_half_mean:.3e},"
                f" attempts=2, lr_scale persisted at {trainer.lr_scale}"
            )

    def test_retry_exhausted_restores_and_keeps_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            trainer = make_trainer(transaction_config_dict(run_dir), stop_after=10)
            trainer._update()
            trainer._save_checkpoint("latest")
            pre_actor = copy.deepcopy(trainer.actor.state_dict())
            pre_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
            ledger_before = trainer.ledger.to_dict()

            trainer.max_kl_mean = 0.0
            trainer.max_kl_state = 0.0
            with self.assertRaises(UpdateRejected):
                trainer._update()

            # One retry maximum: exactly two optimizer attempts for this update.
            self.assertEqual(trainer.ledger.optimizer_attempts, ledger_before["optimizer_attempts"] + 2)
            self.assertEqual(trainer.successful_updates, 1)
            for name, parameter in trainer.actor.state_dict().items():
                self.assertTrue(torch.equal(parameter, pre_actor[name]), f"actor not restored at {name}")
            compare_optimizer_state(trainer.optimizer.state_dict(), pre_optimizer, "restored_optimizer")
            self.assertEqual(trainer.lr_scale, 1.0, "rejected update must not keep the halved scale")

            rejected = json.loads((run_dir / "update_rejected.json").read_text())
            self.assertEqual(rejected["update"], 2)
            self.assertEqual(rejected["attempts"], 2)
            self.assertEqual(rejected["reason"], "kl_retry_exhausted")
            self.assertGreater(rejected["kl"]["kl_mean"], 0.0)

            # No half-updated checkpoint: latest still holds update 1.
            latest = torch.load(run_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(latest["training_state"]["successful_updates"], 1)
            self.assertEqual(latest["saved_at_update"], 1)

            # Resource costs of the rejected update are not erased.
            self.assertEqual(
                trainer.ledger.rollout_episodes,
                ledger_before["rollout_episodes"] + trainer.batch * trainer.groups,
            )
            self.assertGreater(trainer.ledger.physical_expert_rows, ledger_before["physical_expert_rows"])
            self.assertGreater(trainer.ledger.physical_head_rows, ledger_before["physical_head_rows"])
            print(
                "\nT20 retry-exhausted: attempts=2, state restored bitwise,"
                f" rollout cost retained ({trainer.ledger.rollout_episodes} episodes)"
            )

    def test_run_backoff_budget_blocks_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            config_dict = transaction_config_dict(run_dir)
            config_dict["optimization"]["max_lr_backoffs_per_run"] = 0
            trainer = make_trainer(config_dict, stop_after=10)
            trainer._update()
            trainer.max_kl_mean = 0.0
            trainer.max_kl_state = 0.0
            attempts_before = trainer.ledger.optimizer_attempts
            with self.assertRaises(UpdateRejected):
                trainer._update()
            self.assertEqual(
                trainer.ledger.optimizer_attempts,
                attempts_before + 1,
                "exhausted run-level backoff budget must block even the first retry",
            )
            self.assertEqual(trainer.lr_scale, 1.0)
            rejected = json.loads((run_dir / "update_rejected.json").read_text())
            self.assertEqual(rejected["reason"], "kl_budget_exhausted")

    def test_failed_run_writes_failure_and_keeps_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            config_dict = transaction_config_dict(run_dir)
            config_dict["optimization"]["max_lr_backoffs_per_run"] = 0
            trainer = make_trainer(config_dict, stop_after=10)
            trainer.max_kl_mean = 0.0
            trainer.max_kl_state = 0.0
            with self.assertRaises(UpdateRejected):
                trainer.run()
            failure = json.loads((run_dir / "failure.json").read_text())
            self.assertEqual(failure["exception_type"], "UpdateRejected")
            self.assertEqual(failure["successful_updates_completed"], 0)
            self.assertIsNotNone(failure["last_complete_checkpoint"])
            latest = torch.load(run_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(latest["saved_at_update"], 0)
            self.assertTrue((run_dir / "final_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
