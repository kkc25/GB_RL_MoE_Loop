"""Offline train-only fit diagnostic: tiny fake and genuine HF CPU contracts."""
from dataclasses import replace
from importlib.util import find_spec
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from mixtral_lab.adapter_fit import (AdapterFitConfig, _correction_totals,
    _diagnostics, _json_hash, _save_checkpoint, _training_probe,
    load_adapter_fit_checkpoint, train_adapter_fit)
from mixtral_lab.adaptive_fit import AdaptiveFitController
from mixtral_lab.second_round import SecondRoundResidual, differentiable_readout


class ScaleNorm(nn.Module):
    def __init__(self, hidden, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden, dtype=dtype))

    def forward(self, value):
        return value * self.weight


class FakeAdapter:
    """Frozen deterministic head; no Hub, CUDA, selection split, or random state."""
    def __init__(self, dtype=torch.float32):
        self.hidden_size, self.device = 4, torch.device('cpu')
        self.model = nn.Module()
        self.model.model = nn.Module()
        self.model.model.norm = ScaleNorm(4, dtype)
        self.model.lm_head = nn.Linear(4, 4, bias=False, dtype=dtype)
        with torch.no_grad():
            self.model.lm_head.weight.copy_(torch.eye(4, dtype=dtype))
        self.model.config = SimpleNamespace(vocab_size=4)
        self.model.requires_grad_(False).eval()
        self.native_shapes = []

    def native_pair(self, hidden):
        return torch.zeros((len(hidden), 2), dtype=torch.long)

    def step_pairs(self, hidden, pairs):
        self.native_shapes.append(tuple(hidden.shape))
        return hidden.detach().clone()

    def readout(self, hidden):
        raise AssertionError('no_grad convenience readout must not be used for training')


def fixture(count=5, dtype=torch.float32):
    hidden = torch.zeros(count, 4, dtype=dtype)
    hidden[:, 0] = 1.
    hidden[:, 2] = torch.arange(count, dtype=torch.float32).to(dtype) / 10.
    return FakeAdapter(dtype), hidden, torch.ones(count, dtype=torch.long)


def config(**kwargs):
    return replace(AdapterFitConfig(rank=2, updates=12, batch_contexts=2,
                                    probe_chunk_contexts=3, eval_interval=3,
                                    learning_rate=.05), **kwargs)


class AdapterFitTests(unittest.TestCase):
    def test_train_decreases_fixed_probe_and_preserves_backbone(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                adapter, h0, targets = fixture(dtype=dtype)
                original = {k: v.clone() for k, v in adapter.model.state_dict().items()}
                rng = torch.random.get_rng_state().clone()
                out = Path(tmp) / 'fit'
                module, summary = train_adapter_fit(adapter, h0, targets, config(), {'split': 'train'}, out)
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                self.assertEqual(summary['status'], 'completed')
                self.assertTrue(summary['train_fit_passed'])
                self.assertLess(summary['best_train_probe_nll'], summary['initial_train_probe_nll'] - .1)
                self.assertEqual(summary['successful_updates'], 12)
                self.assertEqual(summary['context_visit_counts'], [4] * 5)
                self.assertEqual(summary['completed_epochs'], 4)
                self.assertEqual(summary['unique_contexts_visited'], 5)
                self.assertEqual(adapter.native_shapes, [(3, 4), (3, 4)])
                work = summary['work_accounting']
                self.assertEqual(work['native_cache_contexts_including_padding'], 6)
                self.assertEqual(work['optimizer_head_contexts'], 20)
                self.assertEqual(work['optimizer_backward_contexts'], 20)
                self.assertEqual(work['train_probe_calls'], 5)
                self.assertEqual(work['train_probe_scored_contexts'], 25)
                self.assertEqual(work['train_probe_head_contexts_including_padding'], 30)
                self.assertEqual(work['total_head_forward_contexts_including_padding'], 50)
                self.assertEqual(work['total_adapter_correction_contexts_including_diagnostics_and_padding'], 100)
                self.assertFalse(summary['selection_used'])
                self.assertFalse(summary['recurrent_ffn'])
                self.assertFalse(summary['learned_stop'])
                self.assertTrue(summary['added_capacity'])
                self.assertFalse(summary['heldout_gain_established'])
                self.assertFalse(summary['practical_gain_established'])
                self.assertFalse(any(p.requires_grad for p in module.parameters()))
                loaded, metadata = load_adapter_fit_checkpoint(out / 'checkpoint_best.pt', {'split': 'train'},
                    expected_config=config(), expected_hidden_size=4)
                torch.testing.assert_close(loaded(h0), module(h0), rtol=0, atol=0)
                self.assertEqual(metadata['update'], summary['best_update'])
                self.assertEqual(len(metadata['checkpoint_sha256']), 64)
                self.assertEqual(metadata['train_inputs_sha256'], summary['train_inputs_sha256'])
                initial, initial_meta = load_adapter_fit_checkpoint(out / 'checkpoint_initial.pt', {'split': 'train'})
                self.assertEqual(initial_meta['update'], 0)
                torch.testing.assert_close(initial(h0), h0, rtol=0, atol=0)
                self.assertEqual(summary['initial_correction_diagnostics']['native_effective_change_rate'], 0.)
                diagnostics = summary['best_correction_diagnostics']
                self.assertGreater(diagnostics['fp32_proposed_correction_rms'], 0.)
                self.assertGreater(diagnostics['bf16_effective_change_rate'], 0.)
                self.assertEqual(diagnostics['bf16_diagnostic_is_actual_native'], dtype == torch.bfloat16)
                for key, value in adapter.model.state_dict().items():
                    torch.testing.assert_close(value, original[key], rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))
                rows = json.loads((out / 'metrics.json').read_text())
                self.assertTrue(all(row['accepted'] for row in rows))
                self.assertTrue(all(row['gradient_norm'] > 0 for row in rows))
                self.assertEqual(summary, json.loads((out / 'summary.json').read_text()))

    def test_without_replacement_remainder_and_reproducible_visits(self):
        logs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                adapter, h0, targets = fixture(7)
                out = Path(tmp) / 'fit'
                _, summary = train_adapter_fit(adapter, h0, targets,
                    config(updates=6, batch_contexts=3), {'test': 'epoch'}, out)
                rows = json.loads((out / 'metrics.json').read_text())
                self.assertEqual([r['batch_contexts'] for r in rows], [3, 3, 1, 3, 3, 1])
                for epoch in (1, 2):
                    visited = [i for r in rows if r['epoch'] == epoch for i in r['visited_indices']]
                    self.assertEqual(sorted(visited), list(range(7)))
                self.assertEqual(summary['context_visit_counts'], [2] * 7)
                logs.append([r['visited_indices'] for r in rows])
        self.assertEqual(logs[0], logs[1])

    def test_fixed_full_probe_masks_padding_and_uses_stable_shapes(self):
        adapter, h0, labels = fixture(5)
        module = SecondRoundResidual(4, config())
        shapes = []
        real = differentiable_readout
        def record(a, h):
            shapes.append(tuple(h.shape))
            return real(a, h)
        with patch('mixtral_lab.adapter_fit.differentiable_readout', side_effect=record):
            first = _training_probe(adapter, module, h0, labels, config())
            second = _training_probe(adapter, module, h0, labels, config())
        expected = float(F.cross_entropy(real(adapter, h0).float(), labels, reduction='none').double().mean())
        self.assertAlmostEqual(first['nll'], expected, places=10)
        self.assertEqual(first, second)
        self.assertEqual(shapes, [(3, 4)] * 4)

    def test_best_checkpoint_uses_train_probe_not_latest_or_external_selection(self):
        adapter, h0, targets = fixture(2)
        real = _training_probe
        scores = iter([3., 1., 2., 2.5])
        def staged_probe(*args):
            result = real(*args)
            result['nll'] = next(scores)
            return result
        with tempfile.TemporaryDirectory() as tmp, patch('mixtral_lab.adapter_fit._training_probe', side_effect=staged_probe):
            out = Path(tmp) / 'fit'
            module, result = train_adapter_fit(adapter, h0, targets,
                config(updates=3, batch_contexts=2, eval_interval=1), {'test': 'best'}, out)
            best, best_meta = load_adapter_fit_checkpoint(out / 'checkpoint_best.pt', {'test': 'best'})
            latest, latest_meta = load_adapter_fit_checkpoint(out / 'checkpoint_latest.pt', {'test': 'best'})
            self.assertEqual(result['best_update'], 1)
            self.assertEqual(best_meta['train_probe_nll'], 1.)
            self.assertEqual(latest_meta['update'], 3)
            self.assertEqual(latest_meta['train_probe_nll'], 2.5)
            self.assertFalse(torch.equal(best.up, latest.up))
            torch.testing.assert_close(module.up, best.up, rtol=0, atol=0)

    def test_update_zero_wins_ties_and_fit_threshold_is_independent(self):
        adapter, h0, targets = fixture(2)
        real = _training_probe
        def tied_probe(*args):
            result = real(*args)
            result['nll'] = 1.
            return result
        with tempfile.TemporaryDirectory() as tmp, patch('mixtral_lab.adapter_fit._training_probe', side_effect=tied_probe):
            module, result = train_adapter_fit(adapter, h0, targets,
                config(updates=2, eval_interval=1), {'test': 'tie'}, Path(tmp) / 'fit')
            self.assertEqual(result['best_update'], 0)
            self.assertEqual(float(module.up.abs().sum()), 0.)
            self.assertFalse(result['train_fit_passed'])
            self.assertEqual(result['fit_evidence_status'], 'threshold_not_reached')
        with tempfile.TemporaryDirectory() as tmp:
            _, result = train_adapter_fit(adapter, h0, targets,
                config(updates=2, fit_gain_threshold=100.), {'test': 'threshold'}, Path(tmp) / 'fit')
            self.assertGreater(result['train_fit_gain'], 0.)
            self.assertFalse(result['train_fit_passed'])

    def test_incomplete_coverage_is_not_negative_fit_evidence(self):
        adapter, h0, targets = fixture(7)
        with tempfile.TemporaryDirectory() as tmp:
            _, result = train_adapter_fit(adapter, h0, targets,
                config(updates=1, batch_contexts=2, fit_gain_threshold=1e-9),
                {'test': 'partial'}, Path(tmp) / 'fit')
            self.assertGreater(result['train_fit_gain'], 1e-9)
            self.assertFalse(result['train_fit_passed'])
            self.assertEqual(result['fit_evidence_status'], 'incomplete_coverage')
            self.assertEqual(result['unique_contexts_visited'], 2)
            self.assertEqual(result['completed_epochs'], 0)

    def test_bf16_post_add_quantization_is_measured_not_assumed_fixed(self):
        adapter, h0, targets = fixture(2, torch.bfloat16)
        h0.fill_(1.)
        module = SecondRoundResidual(4, config())
        with torch.no_grad():
            module.down.fill_(.01)
            module.up.fill_(1e-4)
            totals = _correction_totals(module, h0)
            diagnostics = _diagnostics(totals, h0.dtype)
        self.assertGreater(diagnostics['fp32_proposed_correction_rms'], 0.)
        self.assertEqual(diagnostics['bf16_effective_change_rms'], 0.)
        self.assertEqual(diagnostics['bf16_effective_change_rate'], 0.)
        self.assertEqual(diagnostics['native_effective_change_rate'], 0.)
        self.assertTrue(diagnostics['bf16_diagnostic_is_actual_native'])

    def test_zero_init_native_gradients_flow_to_both_factors_after_first_step(self):
        for dtype in (torch.float32, torch.bfloat16):
            adapter, h0, targets = fixture(3, dtype)
            module = SecondRoundResidual(4, config())
            self.assertTrue(torch.equal(module(h0), h0))
            optimizer = torch.optim.SGD(module.parameters(), lr=.1)
            F.cross_entropy(differentiable_readout(adapter, module(h0)).float(), targets).backward()
            self.assertGreater(float(module.up.grad.norm()), 0.)
            self.assertEqual(float(module.down.grad.norm()), 0.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(differentiable_readout(adapter, module(h0)).float(), targets).backward()
            self.assertGreater(float(module.down.grad.norm()), 0.)
            self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))

    def test_nonfinite_gradients_cannot_pass_or_contaminate_checkpoint(self):
        adapter, h0, targets = fixture(3)
        real = differentiable_readout
        def poisoned(a, h):
            if h.requires_grad:
                h.register_hook(lambda grad: torch.full_like(grad, float('nan')))
            return real(a, h)
        with tempfile.TemporaryDirectory() as tmp, patch('mixtral_lab.adapter_fit.differentiable_readout', side_effect=poisoned):
            out = Path(tmp) / 'fit'
            module, result = train_adapter_fit(adapter, h0, targets, config(), {'test': 'gradient'}, out)
            self.assertEqual(result['status'], 'stopped_nonfinite_gradient')
            self.assertEqual(result['successful_updates'], 0)
            self.assertFalse(result['train_fit_passed'])
            self.assertEqual(result['fit_evidence_status'], 'numerical_guard_stopped')
            self.assertEqual(result['context_visit_counts'], [0, 0, 0])
            self.assertTrue(torch.equal(module(h0), h0))
            latest, meta = load_adapter_fit_checkpoint(out / 'checkpoint_latest.pt', {'test': 'gradient'})
            self.assertEqual(meta['update'], 0)
            self.assertTrue(torch.equal(latest(h0), h0))

    def test_nonfinite_parameter_step_rolls_back(self):
        adapter, h0, targets = fixture(2)
        def bad_step(optimizer, closure=None):
            with torch.no_grad():
                optimizer.param_groups[0]['params'][0].fill_(float('nan'))
        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.optim.AdamW, 'step', bad_step):
            module, result = train_adapter_fit(adapter, h0, targets, config(), {'test': 'nan'}, Path(tmp) / 'fit')
            self.assertEqual(result['status'], 'stopped_nonfinite_parameters')
            self.assertEqual(result['successful_updates'], 0)
            self.assertTrue(torch.equal(module(h0), h0))

    def test_nonfinite_probe_falls_back_to_valid_checkpoint_and_never_passes(self):
        adapter, h0, targets = fixture(2)
        real = _training_probe
        calls = 0
        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise FloatingPointError('simulated full probe overflow')
            return real(*args)
        with tempfile.TemporaryDirectory() as tmp, patch('mixtral_lab.adapter_fit._training_probe', side_effect=fail_second):
            out = Path(tmp) / 'fit'
            module, result = train_adapter_fit(adapter, h0, targets,
                config(updates=3, eval_interval=1), {'test': 'probe-failure'}, out)
            self.assertEqual(result['status'], 'stopped_nonfinite_probe')
            self.assertFalse(result['train_fit_passed'])
            self.assertEqual(result['latest_update'], 0)
            self.assertTrue(torch.equal(module(h0), h0))
            latest, metadata = load_adapter_fit_checkpoint(out / 'checkpoint_latest.pt', {'test': 'probe-failure'})
            self.assertEqual(metadata['update'], 0)
            self.assertTrue(torch.equal(latest(h0), h0))

    def test_unscheduled_guard_scores_last_accepted_weights(self):
        adapter, h0, targets = fixture(2)
        real = differentiable_readout
        train_calls = 0
        def fail_second(a, h):
            nonlocal train_calls
            if h.requires_grad:
                train_calls += 1
                if train_calls == 2:
                    h.register_hook(lambda grad: torch.full_like(grad, float('nan')))
            return real(a, h)
        with tempfile.TemporaryDirectory() as tmp, patch('mixtral_lab.adapter_fit.differentiable_readout', side_effect=fail_second):
            out = Path(tmp) / 'fit'
            _, result = train_adapter_fit(adapter, h0, targets,
                config(updates=3, eval_interval=3), {'test': 'late-guard'}, out)
            self.assertEqual(result['status'], 'stopped_nonfinite_gradient')
            self.assertEqual(result['successful_updates'], 1)
            self.assertEqual(result['latest_update'], 1)
            self.assertEqual(result['best_update'], 1)
            self.assertFalse(result['train_fit_passed'])
            window = result['train_probe_history'][-1]['probe_interval_gradients']
            self.assertEqual(window['accepted_updates'], 1)
            self.assertEqual(window['nonzero_gradient_updates'], 1)
            self.assertGreater(window['max_gradient_norm'], 0.)
            self.assertEqual(result['adaptive_controller_state']['observation_count'], 1)
            _, metadata = load_adapter_fit_checkpoint(out / 'checkpoint_latest.pt', {'test': 'late-guard'})
            self.assertEqual(metadata['update'], 1)
            self.assertEqual(metadata['train_probe_nll'], result['latest_train_probe_nll'])

    def test_checkpoint_rejects_identity_configuration_tensor_state_and_metadata_tamper(self):
        adapter, h0, targets = fixture(2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'fit'
            train_adapter_fit(adapter, h0, targets, config(updates=1), {'test': 'integrity'}, out)
            original_path = out / 'checkpoint_best.pt'
            original = torch.load(original_path, weights_only=True)
            for key, mutate in [
                ('config', lambda data: data['config'].__setitem__('residual_scale', .5)),
                ('tensor', lambda data: data.__setitem__('train_inputs_sha256', '0' * 64)),
                ('state', lambda data: data['state']['up'].add_(1.)),
                ('metadata', lambda data: data.__setitem__('update', 0)),
                ('format', lambda data: data.__setitem__('format', 'mixtral-second-round-residual-v1')),
                ('adaptive', lambda data: data['adaptive_controller_state'].__setitem__('current_lr', .123)),
            ]:
                with self.subTest(key=key):
                    import copy
                    changed = copy.deepcopy(original)
                    mutate(changed)
                    path = out / f'tampered-{key}.pt'
                    torch.save(changed, path)
                    with self.assertRaises(ValueError):
                        load_adapter_fit_checkpoint(path, {'test': 'integrity'})
            with self.assertRaisesRegex(ValueError, 'identity'):
                load_adapter_fit_checkpoint(original_path, {'test': 'wrong'})
            with self.assertRaisesRegex(ValueError, 'expected configuration'):
                load_adapter_fit_checkpoint(original_path, {'test': 'integrity'}, expected_config=config())
            with self.assertRaisesRegex(ValueError, 'expected hidden'):
                load_adapter_fit_checkpoint(original_path, {'test': 'integrity'}, expected_hidden_size=16)
            with self.assertRaisesRegex(ValueError, 'nonempty identity'):
                load_adapter_fit_checkpoint(original_path, {})
            with self.assertRaises(FileExistsError):
                train_adapter_fit(adapter, h0, targets, config(updates=1), {'test': 'integrity'}, out)

    def test_atomic_save_failure_preserves_complete_previous_checkpoint(self):
        adapter, h0, targets = fixture(2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'fit'
            module, _ = train_adapter_fit(adapter, h0, targets, config(updates=1), {'test': 'atomic'}, out)
            path = out / 'checkpoint_best.pt'
            before = path.read_bytes()
            data = torch.load(path, weights_only=True)
            def interrupted(payload, target):
                Path(target).write_bytes(b'incomplete')
                raise OSError('simulated interruption')
            with patch('mixtral_lab.adapter_fit.torch.save', side_effect=interrupted):
                with self.assertRaises(OSError):
                    _save_checkpoint(path, module, {'test': 'atomic'},
                        {k: data[k] for k in ('train_inputs_sha256', 'cached_native_sha256')},
                        data['probe_spec'], data['update'],
                        {'nll': data['train_probe_nll'], 'diagnostics': data['probe_diagnostics']},
                        controller_state=data['adaptive_controller_state'])
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(any(p.name.endswith('.tmp') for p in out.iterdir()))

    def test_config_and_input_budgets_fail_before_creating_run(self):
        for overrides in [{'rank': True}, {'rank': 257}, {'updates': 0}, {'updates': 10001},
                          {'batch_contexts': 0}, {'max_train_contexts': 4097}, {'probe_chunk_contexts': 1025},
                          {'learning_rate': 1.01}, {'gradient_clip': float('inf')},
                          {'residual_scale': 1.01}, {'fit_gain_threshold': 0.},
                          {'fit_gain_threshold': float('nan')}, {'seed': -1}, {'seed': True}]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                config(**overrides).validate()
        adapter, h0, targets = fixture(5)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'fit'
            cases = [(h0, targets, config(max_train_contexts=4), {'test': 'budget'}),
                     (h0.to(torch.float64), targets, config(), {'test': 'dtype'}),
                     (h0, targets.float(), config(), {'test': 'labels'}),
                     (h0, targets + 4, config(), {'test': 'vocab'}),
                     (h0, targets, config(), {}),
                     (h0, targets, config(), {'bad': float('nan')}),
                     (torch.full_like(h0, float('nan')), targets, config(), {'test': 'finite'})]
            for hidden, labels, cfg, identities in cases:
                with self.assertRaises(ValueError):
                    train_adapter_fit(adapter, hidden, labels, cfg, identities, out)
                self.assertFalse(out.exists())
            adapter.model.model.norm.train()
            with self.assertRaisesRegex(ValueError, 'eval mode'):
                train_adapter_fit(adapter, h0, targets, config(), {'test': 'mode'}, out)
            adapter.model.eval()
            adapter.model.lm_head.weight.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, 'frozen'):
                train_adapter_fit(adapter, h0, targets, config(), {'test': 'freeze'}, out)
            adapter.model.lm_head.weight.requires_grad_(False)
            adapter.model.lm_head.weight.grad = torch.zeros_like(adapter.model.lm_head.weight)
            with self.assertRaisesRegex(ValueError, 'gradients'):
                train_adapter_fit(adapter, h0, targets, config(), {'test': 'grad'}, out)
            self.assertFalse(out.exists())

    def test_adaptive_disabled_is_bitwise_fixed_rate_behavior(self):
        states, results = [], []
        for adaptive in (None, {'enabled': False}):
            adapter, h0, targets = fixture(5)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'fit'
                module, result = train_adapter_fit(adapter, h0, targets,
                    config(adaptive=adaptive), {'test': 'fixed-default'}, out)
                states.append({key: value.clone() for key, value in module.state_dict().items()})
                results.append(result)
                rows = json.loads((out / 'metrics.json').read_text())
                self.assertTrue(all(row['learning_rate'] == .05 for row in rows))
                self.assertFalse(result['adaptive_stopped'])
                self.assertEqual(result['stop_reason'], 'update_budget_exhausted')
                self.assertEqual(result['successful_updates'], 12)
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
        self.assertEqual(results[0]['best_train_probe_nll'], results[1]['best_train_probe_nll'])
        self.assertEqual(results[0]['context_visit_counts'], results[1]['context_visit_counts'])

    def test_adaptive_lr_uses_only_fixed_train_probe_and_applies_next_update(self):
        adapter, h0, targets = fixture(5)
        actual_probe, observations = _training_probe, []
        class RecordingController(AdaptiveFitController):
            def observe(self, update, train_loss, coverage_complete, ineffective_update=False):
                observations.append((update, train_loss, coverage_complete, ineffective_update))
                return super().observe(update, train_loss, coverage_complete, ineffective_update)
        def constant_probe(*args):
            result = actual_probe(*args)
            result['nll'] = 1.
            return result
        settings = {'enabled': True, 'eval_interval': 3, 'patience': 1,
                    'max_lr': .1, 'min_lr': .001, 'min_updates': 100}
        with tempfile.TemporaryDirectory() as tmp, \
                patch('mixtral_lab.adapter_fit._training_probe', side_effect=constant_probe), \
                patch('mixtral_lab.adapter_fit.AdaptiveFitController', RecordingController):
            out = Path(tmp) / 'fit'
            _, result = train_adapter_fit(adapter, h0, targets,
                config(updates=6, adaptive=settings), {'test': 'train-feedback'}, out)
            rows = json.loads((out / 'metrics.json').read_text())
            self.assertEqual([item[0] for item in observations], [0, 3, 6])
            self.assertEqual([item[1] for item in observations], [1., 1., 1.])
            self.assertTrue(any(row['train_batch_nll'] != 1. for row in rows))
            self.assertTrue(all(row['learning_rate'] == .05 for row in rows[:3]))
            self.assertEqual(rows[2]['adaptive_decision']['action'], 'reduce_lr')
            self.assertEqual(rows[2]['next_learning_rate'], .025)
            self.assertEqual(rows[3]['learning_rate'], .025)
            self.assertEqual(result['best_update'], 0)
            # Audit state is bound to the checkpoint, not an inference-time tuner.
            _, latest = load_adapter_fit_checkpoint(out / 'checkpoint_latest.pt', {'test': 'train-feedback'})
            self.assertEqual(latest['adaptive_controller_state'], result['adaptive_controller_state'])
            _, best = load_adapter_fit_checkpoint(out / 'checkpoint_best.pt', {'test': 'train-feedback'})
            self.assertEqual(best['adaptive_controller_state'], result['best_adaptive_controller_state'])
            self.assertEqual(best['adaptive_controller_state']['last_update'], 0)

    def test_adaptive_growth_requires_actual_bf16_zero_post_add_and_nonzero_gradient(self):
        for dtype, expect_growth in [(torch.bfloat16, True), (torch.float32, False)]:
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                adapter, h0, targets = fixture(2, dtype)
                h0.fill_(1.)  # no zero base entries where tiny BF16 corrections would survive
                settings = {'enabled': True, 'eval_interval': 1, 'patience': 1,
                            'min_lr': 1e-10, 'max_lr': 1e-4, 'min_updates': 64}
                out = Path(tmp) / 'fit'
                _, result = train_adapter_fit(adapter, h0, targets,
                    config(updates=3, eval_interval=1, learning_rate=1e-8, adaptive=settings),
                    {'test': 'bf16-growth'}, out)
                rows = json.loads((out / 'metrics.json').read_text())
                self.assertTrue(all(row['gradient_norm'] > 0 for row in rows))
                decisions = result['adaptive_controller_state']['decisions']
                self.assertEqual(any(d['action'] == 'grow_lr' for d in decisions), expect_growth)
                self.assertTrue(all(d['ineffective_update'] == expect_growth for d in decisions[1:]))
                self.assertTrue(all(1e-10 <= d['lr'] <= 1e-4 for d in decisions))
                if expect_growth:
                    self.assertEqual(result['best_correction_diagnostics']['native_effective_change_rate'], 0.)

    def test_adaptive_growth_uses_accepted_probe_interval_not_final_minibatch(self):
        for dtype, expect_growth in [(torch.bfloat16, True), (torch.float32, False)]:
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                adapter = FakeAdapter(dtype)
                h0 = torch.tensor([[1.] * 4, [0.] * 4], dtype=dtype)
                targets = torch.ones(2, dtype=torch.long)
                settings = {'enabled': True, 'eval_interval': 2, 'patience': 1,
                            'min_lr': 1e-10, 'max_lr': 1e-4}
                out = Path(tmp) / 'fit'
                _, result = train_adapter_fit(adapter, h0, targets,
                    config(updates=2, batch_contexts=1, probe_chunk_contexts=2,
                           eval_interval=2, learning_rate=1e-8, seed=0, adaptive=settings),
                    {'test': 'gradient-window'}, out)
                rows = json.loads((out / 'metrics.json').read_text())
                self.assertEqual([row['visited_indices'] for row in rows], [[0], [1]])
                self.assertGreater(rows[0]['gradient_norm'], 0.)
                self.assertEqual(rows[1]['gradient_norm'], 0.)
                interval = dict(start_update_exclusive=0, accepted_updates=2,
                                nonzero_gradient_updates=1,
                                max_gradient_norm=rows[0]['gradient_norm'])
                self.assertEqual(rows[1]['probe_interval_gradients'], interval)
                probe = result['train_probe_history'][-1]
                self.assertEqual(probe['probe_interval_gradients'], interval)
                self.assertEqual(probe['diagnostics']['native_effective_change_rate'], 0.)
                self.assertEqual(probe['diagnostics']['bf16_effective_change_rate'], 0.)
                decision = result['adaptive_controller_state']['decisions'][-1]
                self.assertEqual(decision['ineffective_update'], expect_growth)
                self.assertEqual(decision['action'], 'grow_lr' if expect_growth else 'reduce_lr')
                self.assertEqual(decision['lr'], 2e-8 if expect_growth else 5e-9)
                self.assertEqual(result['best_update'], 0)
                self.assertFalse(result['train_fit_passed'])
                _, latest = load_adapter_fit_checkpoint(out / 'checkpoint_latest.pt',
                                                        {'test': 'gradient-window'})
                self.assertEqual(latest['adaptive_controller_state'], result['adaptive_controller_state'])
                _, best = load_adapter_fit_checkpoint(out / 'checkpoint_best.pt',
                                                      {'test': 'gradient-window'})
                self.assertEqual(best['adaptive_controller_state']['last_update'], 0)

    def test_adaptive_gradient_window_resets_and_all_zero_gradients_do_not_grow(self):
        for active_first_window, interval in ((True, 2), (False, 2), (True, 3), (False, 3)):
            with self.subTest(active=active_first_window, interval=interval), tempfile.TemporaryDirectory() as tmp:
                adapter = FakeAdapter(torch.bfloat16)
                h0 = torch.zeros(4, 4, dtype=torch.bfloat16)
                if active_first_window:
                    h0[0].fill_(1.)
                settings = {'enabled': True, 'eval_interval': interval, 'patience': 1,
                            'min_lr': 1e-10, 'max_lr': 1e-4}
                out = Path(tmp) / 'fit'
                _, result = train_adapter_fit(adapter, h0, torch.ones(4, dtype=torch.long),
                    config(updates=4, batch_contexts=1, probe_chunk_contexts=4,
                           eval_interval=interval, learning_rate=1e-8, seed=0, adaptive=settings),
                    {'test': 'window-reset'}, out)
                rows = json.loads((out / 'metrics.json').read_text())
                self.assertEqual([row['visited_indices'] for row in rows], [[0], [1], [3], [2]])
                probes = result['train_probe_history'][1:]
                windows = [probe['probe_interval_gradients'] for probe in probes]
                self.assertEqual([window['start_update_exclusive'] for window in windows], [0, interval])
                self.assertEqual([window['accepted_updates'] for window in windows], [interval, 4 - interval])
                self.assertEqual([window['nonzero_gradient_updates'] for window in windows],
                                 [int(active_first_window), 0])
                self.assertEqual(windows[1]['max_gradient_norm'], 0.)
                self.assertTrue(all(probe['diagnostics']['native_effective_change_rate'] == 0.
                                    for probe in probes))
                decisions = result['adaptive_controller_state']['decisions'][1:]
                self.assertEqual([decision['ineffective_update'] for decision in decisions],
                                 [active_first_window, False])
                self.assertEqual([decision['action'] for decision in decisions],
                                 ['grow_lr' if active_first_window else 'reduce_lr', 'reduce_lr'])

    def test_adaptive_gradient_window_excludes_every_rejected_update(self):
        for guard in ('loss', 'gradient', 'parameters', 'correction', 'probe'):
            with self.subTest(guard=guard), tempfile.TemporaryDirectory() as tmp:
                adapter = FakeAdapter(torch.bfloat16)
                # The accepted first update has zero gradient. Only the rejected
                # second update could supply nonzero evidence.
                h0 = torch.tensor([[0.] * 4, [1.] * 4], dtype=torch.bfloat16)
                calls = 0
                if guard in ('loss', 'gradient'):
                    def poisoned(a, h):
                        nonlocal calls
                        if h.requires_grad:
                            calls += 1
                            if calls == 2:
                                if guard == 'loss':
                                    return differentiable_readout(a, h) * float('nan')
                                h.register_hook(lambda grad: torch.full_like(grad, float('nan')))
                        return differentiable_readout(a, h)
                    patcher = patch('mixtral_lab.adapter_fit.differentiable_readout', side_effect=poisoned)
                elif guard == 'parameters':
                    real_step = torch.optim.AdamW.step
                    def bad_step(optimizer, closure=None):
                        nonlocal calls
                        real_step(optimizer, closure)
                        calls += 1
                        if calls == 2:
                            with torch.no_grad():
                                optimizer.param_groups[0]['params'][0].fill_(float('nan'))
                    patcher = patch.object(torch.optim.AdamW, 'step', bad_step)
                elif guard == 'correction':
                    def bad_correction(*args, **kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == 3:  # initial probe, first update, rejected update
                            raise FloatingPointError('simulated correction overflow')
                        return _correction_totals(*args, **kwargs)
                    patcher = patch('mixtral_lab.adapter_fit._correction_totals', side_effect=bad_correction)
                else:
                    def bad_probe(*args):
                        nonlocal calls
                        calls += 1
                        if calls == 2:
                            raise FloatingPointError('simulated full probe overflow')
                        return _training_probe(*args)
                    patcher = patch('mixtral_lab.adapter_fit._training_probe', side_effect=bad_probe)
                settings = {'enabled': True, 'eval_interval': 2, 'patience': 1,
                            'min_lr': 1e-10, 'max_lr': 1e-4}
                out = Path(tmp) / 'fit'
                with patcher:
                    module, result = train_adapter_fit(adapter, h0, torch.ones(2, dtype=torch.long),
                        config(updates=2, batch_contexts=1, probe_chunk_contexts=2,
                               eval_interval=2, learning_rate=1e-8, seed=0, adaptive=settings),
                        {'test': 'rejected-gradient-window'}, out)
                rows = json.loads((out / 'metrics.json').read_text())
                self.assertEqual(result['status'], 'stopped_nonfinite_' + guard)
                self.assertEqual(result['successful_updates'], 1)
                self.assertEqual([row['accepted'] for row in rows], [True, False])
                self.assertNotIn('probe_interval_gradients', rows[1])
                if guard not in ('loss', 'gradient'):
                    self.assertGreater(rows[1]['gradient_norm'], 0.)
                self.assertEqual(result['adaptive_controller_state']['observation_count'], 1)
                self.assertEqual(result['adaptive_controller_state']['growths'], 0)
                self.assertFalse(result['train_fit_passed'])
                self.assertEqual(result['fit_evidence_status'], 'numerical_guard_stopped')
                final_probe = result['train_probe_history'][-1]
                window = final_probe['probe_interval_gradients']
                self.assertEqual(window['accepted_updates'], 0 if guard == 'probe' else 1)
                self.assertEqual(window['nonzero_gradient_updates'], 0)
                self.assertEqual(window['max_gradient_norm'], 0.)
                latest, metadata = load_adapter_fit_checkpoint(out / 'checkpoint_latest.pt',
                                                               {'test': 'rejected-gradient-window'})
                self.assertEqual(metadata['update'], 0 if guard == 'probe' else 1)
                self.assertEqual(metadata['adaptive_controller_state'], result['adaptive_controller_state'])
                self.assertTrue(torch.equal(latest(h0), h0))
                self.assertTrue(torch.equal(module(h0), h0))

    def test_adaptive_gradient_window_allows_recovery_instead_of_floor_stop(self):
        adapter = FakeAdapter(torch.bfloat16)
        h0 = torch.tensor([[1.] * 4, [0.] * 4], dtype=torch.bfloat16)
        settings = {'enabled': True, 'eval_interval': 2, 'patience': 1,
                    'min_lr': .001, 'max_lr': .01, 'min_updates': 2, 'stop_patience': 1}
        with tempfile.TemporaryDirectory() as tmp:
            _, result = train_adapter_fit(adapter, h0, torch.ones(2, dtype=torch.long),
                config(updates=8, batch_contexts=1, probe_chunk_contexts=2, eval_interval=2,
                       learning_rate=.001, seed=0, fit_gain_threshold=1e-5, adaptive=settings),
                {'test': 'window-floor-recovery'}, Path(tmp) / 'fit')
            first = result['adaptive_controller_state']['decisions'][1]
            self.assertEqual(first['update'], 2)
            self.assertTrue(first['ineffective_update'])
            self.assertEqual(first['action'], 'grow_lr')
            self.assertGreater(result['successful_updates'], 2)
            self.assertGreaterEqual(result['train_fit_gain'], 1e-5)
            self.assertTrue(result['train_fit_passed'])

    def test_adaptive_stop_waits_for_coverage_and_keeps_fit_gain_gate(self):
        for improved in (False, True):
            adapter, h0, targets = fixture(7)
            real, count = _training_probe, 0
            def plateau_probe(*args):
                nonlocal count
                result = real(*args)
                result['nll'] = 3. if count == 0 or not improved else 2.
                count += 1
                return result
            settings = {'enabled': True, 'eval_interval': 1, 'patience': 1,
                        'min_lr': .05, 'max_lr': .1, 'min_updates': 2, 'stop_patience': 1}
            with self.subTest(improved=improved), tempfile.TemporaryDirectory() as tmp, \
                    patch('mixtral_lab.adapter_fit._training_probe', side_effect=plateau_probe):
                _, result = train_adapter_fit(adapter, h0, targets,
                    config(updates=10, eval_interval=1, adaptive=settings),
                    {'test': 'adaptive-stop'}, Path(tmp) / 'fit')
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(result['stop_reason'], 'adaptive_plateau')
                self.assertTrue(result['adaptive_stopped'])
                self.assertTrue(result['coverage_complete'])
                self.assertGreaterEqual(result['successful_updates'], 4)
                self.assertLess(result['successful_updates'], 10)
                self.assertEqual(result['train_fit_passed'], improved)
                self.assertTrue(result['adaptive_controller_state']['stopped'])

    def test_adaptive_nonfinite_gradient_cannot_trigger_growth_or_false_completion(self):
        adapter, h0, targets = fixture(2, torch.bfloat16)
        h0.fill_(1.)
        real = differentiable_readout
        def poisoned(a, h):
            if h.requires_grad:
                h.register_hook(lambda grad: torch.full_like(grad, float('nan')))
            return real(a, h)
        settings = {'enabled': True, 'eval_interval': 1, 'patience': 1, 'max_lr': .1}
        with tempfile.TemporaryDirectory() as tmp, \
                patch('mixtral_lab.adapter_fit.differentiable_readout', side_effect=poisoned):
            _, result = train_adapter_fit(adapter, h0, targets,
                config(updates=3, eval_interval=1, adaptive=settings),
                {'test': 'adaptive-nan'}, Path(tmp) / 'fit')
            self.assertEqual(result['status'], 'stopped_nonfinite_gradient')
            self.assertFalse(result['adaptive_stopped'])
            self.assertFalse(result['train_fit_passed'])
            self.assertEqual(result['adaptive_controller_state']['growths'], 0)
            self.assertEqual(result['adaptive_controller_state']['observation_count'], 1)

    def test_adaptive_worsening_full_probe_never_grows_lr(self):
        adapter, h0, targets = fixture(2, torch.bfloat16)
        h0.fill_(1.)
        real, calls = _training_probe, 0
        def worsening(*args):
            nonlocal calls
            result = real(*args)
            result['nll'] = 3. + calls
            calls += 1
            return result
        settings = {'enabled': True, 'eval_interval': 1, 'patience': 1,
                    'min_lr': 1e-10, 'max_lr': 1e-4, 'min_updates': 64}
        with tempfile.TemporaryDirectory() as tmp, \
                patch('mixtral_lab.adapter_fit._training_probe', side_effect=worsening):
            _, result = train_adapter_fit(adapter, h0, targets,
                config(updates=3, eval_interval=1, learning_rate=1e-8, adaptive=settings),
                {'test': 'worse'}, Path(tmp) / 'fit')
            decisions = result['adaptive_controller_state']['decisions']
            self.assertTrue(all(d['ineffective_update'] for d in decisions[1:]))
            self.assertFalse(any(d['action'] == 'grow_lr' for d in decisions))
            self.assertTrue(all(d['lr'] <= 1e-8 for d in decisions))

    def test_checkpoint_replays_adaptive_audit_and_checks_binding_even_with_new_metadata_hash(self):
        import copy
        adapter, h0, targets = fixture(2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'fit'
            train_adapter_fit(adapter, h0, targets, config(updates=1), {'test': 'audit'}, out)
            original = torch.load(out / 'checkpoint_best.pt', weights_only=True)
            changed_counter = copy.deepcopy(original['adaptive_controller_state'])
            changed_counter['growths'] += 1
            wrong_lr = AdaptiveFitController(initial_lr=.01)
            wrong_lr.observe(0, 3., False)
            wrong_config = AdaptiveFitController({'patience': 1}, initial_lr=.05)
            wrong_config.observe(0, 3., False)
            future = AdaptiveFitController(initial_lr=.05)
            future.observe(0, 3., False)
            future.observe(original['update'] + 1, 2., True)
            for number, audit in enumerate((changed_counter, wrong_lr.snapshot(),
                                             wrong_config.snapshot(), future.snapshot())):
                with self.subTest(number=number):
                    changed = copy.deepcopy(original)
                    changed['adaptive_controller_state'] = audit
                    changed['metadata_sha256'] = _json_hash({key: value for key, value in changed.items()
                                                            if key not in ('state', 'metadata_sha256')})
                    path = out / f'invalid-audit-{number}.pt'
                    torch.save(changed, path)
                    with self.assertRaisesRegex(ValueError, 'adaptive'):
                        load_adapter_fit_checkpoint(path, {'test': 'audit'})

    def test_adaptive_config_rejects_conflicting_probes_and_unbounded_initial_lr(self):
        for settings in ({'enabled': True, 'eval_interval': 1},
                         {'enabled': True, 'eval_interval': 3, 'max_lr': .02},
                         {'enabled': True, 'eval_interval': 3, 'mystery': 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                config(adaptive=settings).validate()

    @unittest.skipUnless(find_spec('transformers') is not None, 'optional tiny HF dependency not installed')
    def test_real_tiny_mixtral_train_probe_fit_and_native_identity(self):
        from mixtral_lab.adapter import MixtralAdapter
        from tests.test_mixtral_adapter import tiny
        for dtype in (torch.float32, torch.bfloat16):
            adapter = MixtralAdapter(tiny(dtype))
            ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
            h0 = adapter.capture(ids)[:, :-1].reshape(-1, 16)
            labels = ids[:, 1:].flatten()
            cfg = config(rank=4, updates=24, batch_contexts=5, probe_chunk_contexts=5,
                         learning_rate=.05, eval_interval=4)
            original = {key: value.clone() for key, value in adapter.model.state_dict().items()}
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'fit'
                module, result = train_adapter_fit(adapter, h0, labels, cfg, {'test': 'tiny'}, out)
                self.assertTrue(result['train_fit_passed'])
                self.assertLess(result['best_train_probe_nll'], result['initial_train_probe_nll'])
                initial, _ = load_adapter_fit_checkpoint(out / 'checkpoint_initial.pt', {'test': 'tiny'})
                h1 = adapter.step_pairs(h0, adapter.native_pair(h0))
                torch.testing.assert_close(initial(h1), h1, rtol=0, atol=0)
                torch.testing.assert_close(differentiable_readout(adapter, initial(h1)), adapter.readout(h1), rtol=0, atol=0)
            for key, value in adapter.model.state_dict().items():
                self.assertTrue(torch.equal(value, original[key]))
            self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))


if __name__ == '__main__':
    unittest.main()
