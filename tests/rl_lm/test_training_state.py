"""Shared seams preserve policy RNG domains and reusable rollback state."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rl_lm.checkpoint import capture_rng_state
from rl_lm.training_state import (
    advance_epoch_batch, capture_update_snapshot, epoch_permutation, restore_update_snapshot,
)
from tests.rl_lm.test_resume import compare_optimizer_state
from tests.rl_lm.test_single_rloo import make_trainer, toy_config_dict


class TrainingStateTests(unittest.TestCase):
    def test_permutations_match_legacy_algorithms_without_global_rng_consumption(self):
        state = copy.deepcopy(np.random.get_state())
        permutations = []
        for domain in ['rl-lm-epoch-permutation-v1', 'rl-lm-single-epoch-permutation-v1']:
            for epoch in [0, 1, 10]:
                key = json.dumps([domain, 17, epoch], separators=(',', ':'))
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], 'little')
                expected = np.random.Generator(np.random.Philox(seed)).permutation(19)
                actual = epoch_permutation(domain=domain, actor_seed=17, epoch=epoch, count=19)
                np.testing.assert_array_equal(actual, expected)
                permutations.append(actual)
        self.assertFalse(np.array_equal(permutations[0], permutations[3]))
        current = np.random.get_state()
        self.assertEqual(state[0], current[0])
        np.testing.assert_array_equal(state[1], current[1])
        self.assertEqual(state[2:], current[2:])

    def test_epoch_cursor_drops_tail_without_padding(self):
        permutation = np.array([8, 2, 1, 4, 6, 3, 7, 0, 5])
        indices, epoch, cursor, cached = advance_epoch_batch(
            permutation, epoch=2, cursor=0, batch=4, batches_per_epoch=2)
        self.assertEqual((indices, epoch, cursor), ([8, 2, 1, 4], 2, 4))
        self.assertIs(cached, permutation)
        indices, epoch, cursor, cached = advance_epoch_batch(
            cached, epoch=epoch, cursor=cursor, batch=4, batches_per_epoch=2)
        self.assertEqual((indices, epoch, cursor, cached), ([6, 3, 7, 0], 3, 0, None))

    def test_snapshot_is_reusable_preserves_gradients_and_optimizer_device(self):
        actor = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=0.1)
        actor(torch.ones(1, 2)).sum().backward()
        optimizer.step()  # materialize Adam moments
        snapshot = capture_update_snapshot(actor, optimizer, lr_scale=0.5, lr_backoffs=2)
        grads = [p.grad.clone() for p in actor.parameters()]
        for _ in range(2):
            optimizer.step()
            self.assertEqual(restore_update_snapshot(actor, optimizer, snapshot, device='cpu'), (0.5, 2))
            for name, value in actor.state_dict().items():
                self.assertTrue(torch.equal(value, snapshot['actor'][name]))
            compare_optimizer_state(optimizer.state_dict(), snapshot['optimizer'], 'snapshot')
            for p, grad in zip(actor.parameters(), grads):
                self.assertTrue(torch.equal(p.grad, grad))
            for state in optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        self.assertEqual(value.device.type, 'cpu')

    def test_single_resume_matches_optimizer_rng_and_cursor_across_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / 'continuous'
            b = Path(tmp) / 'interrupted'
            continuous = make_trainer(toy_config_dict(a), stop_after=11)
            continuous.run()
            expected_rng = capture_rng_state()
            make_trainer(toy_config_dict(b), stop_after=5).run()
            resumed = make_trainer(toy_config_dict(b), resume_path=b / 'checkpoint_latest.pt', stop_after=11)
            resumed.run()
            compare_optimizer_state(continuous.optimizer.state_dict(), resumed.optimizer.state_dict(), 'resume')
            for name, value in continuous.actor.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.actor.state_dict()[name]), name)
            self.assertEqual((continuous.epoch, continuous.epoch_cursor), (resumed.epoch, resumed.epoch_cursor))
            self.assertEqual((resumed.epoch, resumed.epoch_cursor), (1, 24))
            actual_rng = capture_rng_state()
            # NumPy's RNG tuple contains an ndarray, unlike optimizer state.
            np.testing.assert_equal(expected_rng.pop('numpy'), actual_rng.pop('numpy'))
            compare_optimizer_state(expected_rng, actual_rng, 'rng')
            self.assertEqual(continuous.lr_scale, resumed.lr_scale)
            self.assertEqual(continuous.lr_backoffs, resumed.lr_backoffs)
