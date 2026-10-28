"""Offline real tiny HF regression tests for the optional residual phase."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.second_round import (SecondRoundConfig, SecondRoundResidual,
    differentiable_readout, load_second_round_checkpoint, train_second_round)
from tests.test_mixtral_adapter import tiny


class SecondRoundTests(unittest.TestCase):
    def test_initial_identity_gradients_and_rng(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                adapter = MixtralAdapter(tiny(dtype))
                ids = torch.tensor([[1, 2, 3, 4]])
                h0 = adapter.capture(ids).reshape(-1, adapter.hidden_size)
                h1 = adapter.step_pairs(h0, adapter.native_pair(h0))
                original = {k: v.clone() for k, v in adapter.model.state_dict().items()}
                rng = torch.random.get_rng_state().clone()
                residual = SecondRoundResidual(16, SecondRoundConfig(rank=4))
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                self.assertTrue(torch.equal(h1, residual(h1)))
                # Patch catches accidental use of the no_grad convenience readout.
                expected = adapter.readout(h1)
                with patch.object(adapter, 'readout', side_effect=AssertionError('no_grad readout used')):
                    logits = differentiable_readout(adapter, residual(h1))
                    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
                    F.cross_entropy(logits.float(), ids.flatten()).backward()
                self.assertGreater(float(residual.up.grad.norm()), 0)
                self.assertEqual(float(residual.down.grad.norm()), 0)
                opt = torch.optim.SGD(residual.parameters(), lr=.1)
                opt.step()
                opt.zero_grad()
                F.cross_entropy(differentiable_readout(adapter, residual(h1)).float(), ids.flatten()).backward()
                self.assertGreater(float(residual.down.grad.norm()), 0)
                for k, v in adapter.model.state_dict().items():
                    self.assertTrue(torch.equal(v, original[k]))
                self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))

    def test_train_checkpoint_and_fixed_stop_baseline(self):
        adapter = MixtralAdapter(tiny())
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        h0 = adapter.capture(ids)[:, :-1].reshape(-1, 16)
        labels = ids[:, 1:].flatten()
        h1 = adapter.step_pairs(h0, adapter.native_pair(h0))
        original = {k: v.clone() for k, v in adapter.model.state_dict().items()}
        def selection(module):
            return {'nll': float(F.cross_entropy(adapter.readout(module(h1)).float(), labels))}
        config = SecondRoundConfig(rank=4, updates=3, batch_contexts=4,
                                   learning_rate=.1, eval_interval=1)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'
            rng = torch.random.get_rng_state().clone()
            module, summary = train_second_round(adapter, h0, labels, selection, config, {'test':'one'}, output)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertEqual(summary['successful_updates'], 3)
            self.assertLess(summary['best_selection_nll'], summary['initial_selection_nll'])
            self.assertEqual(summary['trainable_parameters'], 128)
            loaded = load_second_round_checkpoint(output/'checkpoint_best.pt', {'test':'one'})
            torch.testing.assert_close(loaded(h1), module(h1), rtol=0, atol=0)
            latest = load_second_round_checkpoint(output/'checkpoint_latest.pt', {'test':'one'},
                expected_config=config, expected_hidden_size=16)
            self.assertEqual(latest.hidden_size, 16)
            with self.assertRaisesRegex(ValueError, 'expected configuration'):
                load_second_round_checkpoint(output/'checkpoint_best.pt', {'test':'one'},
                    expected_config=SecondRoundConfig(rank=2))
            initial = load_second_round_checkpoint(output/'checkpoint_initial.pt', {'test':'one'})
            torch.testing.assert_close(initial(h1), h1, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'identity'):
                load_second_round_checkpoint(output/'checkpoint_best.pt', {'test':'other'})
            with self.assertRaises(FileExistsError):
                train_second_round(adapter, h0, labels, selection, config, {'test':'one'}, output)
            rows = json.loads((output/'metrics.json').read_text())
            self.assertGreater(rows[0]['gradient_norm'], 0)
            self.assertGreater(rows[0]['correction_rms'], 0)
        for k, v in adapter.model.state_dict().items():
            self.assertTrue(torch.equal(v, original[k]))
        self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))

    def test_update_zero_wins_ties(self):
        adapter = MixtralAdapter(tiny())
        h0 = adapter.capture(torch.tensor([[1, 2]])).reshape(-1, 16)
        with tempfile.TemporaryDirectory() as tmp:
            module, result = train_second_round(adapter, h0, torch.tensor([2, 3]),
                lambda m: {'nll': 1.}, SecondRoundConfig(rank=2, updates=1), {'test':'one'}, Path(tmp)/'run')
            self.assertEqual(result['best_update'], 0)
            self.assertEqual(float(module.up.detach().abs().sum()), 0)

    def test_checkpoint_rejects_mutation_and_nonfinite_gradients(self):
        adapter = MixtralAdapter(tiny())
        h0 = adapter.capture(torch.tensor([[1, 2]])).reshape(-1, 16)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'
            # A nonfinite gradient cannot create a deployable contaminated best state.
            real = differentiable_readout
            def bad_gradient(a, h):
                h.register_hook(lambda g: torch.full_like(g, float('nan')))
                return real(a, h)
            with patch('mixtral_lab.second_round.differentiable_readout', side_effect=bad_gradient):
                module, result = train_second_round(adapter, h0, torch.tensor([2, 3]),
                    lambda m: {'nll': 1.}, SecondRoundConfig(rank=2, updates=1), {'test':'one'}, output)
            self.assertEqual(result['status'], 'stopped_nonfinite_gradient')
            self.assertEqual(result['successful_updates'], 0)
            self.assertEqual(float(module.up.detach().abs().sum()), 0)
            path = output/'checkpoint_best.pt'
            data = torch.load(path, weights_only=True)
            data['config']['residual_scale'] = .5
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, 'configuration binding'):
                load_second_round_checkpoint(path, {'test':'one'})

    def test_atomic_checkpoint_failure_preserves_previous(self):
        from mixtral_lab.second_round import _save
        module = SecondRoundResidual(16, SecondRoundConfig(rank=2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'best.pt'
            _save(path, module, {'test': 'one'}, 0, 1.)
            before = path.read_bytes()
            def incomplete_save(data, target):
                Path(target).write_bytes(b'incomplete')
                raise OSError('simulated full disk')
            with patch('mixtral_lab.second_round.torch.save', side_effect=incomplete_save):
                with self.assertRaises(OSError):
                    _save(path, module, {'test': 'one'}, 1, .9)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(tmp).iterdir()), [path])
            with self.assertRaisesRegex(ValueError, 'nonempty identity'):
                load_second_round_checkpoint(path, {})
        with self.assertRaises(ValueError):
            SecondRoundResidual(True, SecondRoundConfig(rank=1))

    def test_validation(self):
        for options in ({'rank':True}, {'rank':257}, {'updates':10001}, {'seed':-1},
                        {'residual_scale':2}, {'learning_rate':float('nan')}, {'batch_contexts':0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SecondRoundConfig(**options).validate()
        with self.assertRaises(ValueError):
            SecondRoundResidual(4, SecondRoundConfig(rank=8))


if __name__ == '__main__':
    unittest.main()
