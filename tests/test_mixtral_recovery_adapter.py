"""Offline recovery-adapter objective, split, selection and fail-closed contracts."""
from dataclasses import replace
from importlib.util import find_spec
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from mixtral_lab.recovery_adapter import (
    RecoveryAdapterConfig, _loss_components, _training_probe,
    load_recovery_adapter_checkpoint, train_recovery_adapter)
from mixtral_lab.second_round import SecondRoundResidual, differentiable_readout
from tests.test_mixtral_adapter_fit import fixture


def config(**kwargs):
    return replace(RecoveryAdapterConfig(rank=2, updates=6, batch_contexts=2,
        probe_chunk_contexts=3, eval_interval=2, learning_rate=.05,
        min_updates=32), **kwargs)


def identities():
    return dict(model={'id': 'tiny'}, tokenizer={'id': 'tiny'}, code={'sha': 'unit'},
                train={'selected_document_ids': ['train-a', 'train-b']},
                development={'selected_document_ids': ['development-a']})


def evaluator(scores=None):
    contract = dict(role='development', document_ids=['development-a'],
                    protocol='full_document_online_nll')
    values = iter(scores) if scores is not None else None
    def evaluate(module):
        assert not module.training
        assert not torch.is_grad_enabled()
        return dict(contract, status='ok', nll=next(values) if values is not None else 1.)
    evaluate.contract = contract
    return evaluate


class RecoveryAdapterTests(unittest.TestCase):
    def test_dev_best_not_train_best_or_latest_and_all_checkpoints(self):
        adapter, h0, labels = fixture(2)
        original = {k: v.clone() for k, v in adapter.model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        cfg = config(updates=3, eval_interval=1)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            module, summary = train_recovery_adapter(adapter, h0, labels,
                evaluator([3., 1., 2., 2.5]), cfg, identities(), out)
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['best_update'], 1)
            self.assertEqual(summary['best_development_nll'], 1.)
            self.assertLess(summary['latest_train_probe']['ce'], summary['best_train_probe']['ce'])
            best, best_meta = load_recovery_adapter_checkpoint(out / 'checkpoint_best.pt', identities(),
                expected_config=cfg, expected_hidden_size=4)
            latest, latest_meta = load_recovery_adapter_checkpoint(out / 'checkpoint_latest.pt', identities())
            self.assertEqual(latest_meta['update'], 3)
            self.assertEqual(best_meta['development_metrics'], summary['best_development_metrics'])
            torch.testing.assert_close(module.up, best.up, rtol=0, atol=0)
            self.assertFalse(torch.equal(module.up, latest.up))
            self.assertFalse(any(p.requires_grad for p in module.parameters()))
            self.assertEqual([e['update'] for e in summary['evaluation_history']], [0, 1, 2, 3])
            for entry in summary['evaluation_history']:
                loaded, meta = load_recovery_adapter_checkpoint(out / entry['checkpoint'], identities())
                self.assertEqual(meta['update'], entry['update'])
                self.assertEqual(meta['development'], entry['development'])
            self.assertEqual(json.loads((out / 'summary.json').read_text()), summary)
            self.assertFalse(summary['heldout_used'])
            self.assertFalse(summary['selection_used'])
            self.assertFalse(summary['recurrent_ffn'])
            self.assertTrue(summary['added_capacity'])
            self.assertFalse(summary['native_teacher']['persistent_logits_cached'])
            self.assertEqual(summary['native_teacher']['direction'], 'KL(native||corrected)')
            with self.assertRaises(FileExistsError):
                train_recovery_adapter(adapter, h0, labels, evaluator(), cfg, identities(), out)
            self.assertEqual(json.loads((out / 'summary.json').read_text()), summary)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for key, value in adapter.model.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))

    def test_native_update_zero_wins_ties_and_worse_development(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                adapter, hidden, labels = fixture(2, dtype)
                module, summary = train_recovery_adapter(adapter, hidden, labels,
                    evaluator([1., 1., 2.]), config(updates=2, eval_interval=1), identities(), Path(tmp)/'r')
                self.assertEqual(summary['best_update'], 0)
                self.assertFalse(summary['development_improved'])
                torch.testing.assert_close(module(hidden), hidden, rtol=0, atol=0)
                self.assertLess(summary['latest_train_probe']['nll'], summary['initial_train_probe']['nll'])
                diag = summary['latest_train_probe']['diagnostics']
                self.assertEqual(diag['bf16_diagnostic_is_actual_native'], dtype == torch.bfloat16)
                self.assertGreater(diag['correction_to_hidden_norm'], 0.)

    def test_full_soft_native_kl_direction_detach_and_gradient(self):
        adapter, hidden, labels = fixture(2)
        hidden.requires_grad_(True)
        cfg = config(native_kl_beta=.7, relative_correction_l2=.4, residual_scale=.1)
        module = SecondRoundResidual(4, cfg)
        with torch.no_grad():
            module.up.fill_(.1)
            module.up[0].fill_(.4)
        calls = []
        def readout(a, h):
            calls.append((tuple(h.shape), h.requires_grad, torch.is_grad_enabled()))
            return differentiable_readout(a, h)
        with patch('mixtral_lab.recovery_adapter.differentiable_readout', side_effect=readout):
            parts = _loss_components(adapter, module, hidden, labels, cfg)
        self.assertEqual(calls, [((2, 4), False, False), ((2, 4), True, True)])
        native_logp = F.log_softmax(differentiable_readout(adapter, hidden.detach()).float(), -1)
        student = differentiable_readout(adapter, module(hidden)).float()
        expected_kl = (native_logp.exp() * (native_logp - F.log_softmax(student, -1))).sum(-1)
        expected_ce = F.cross_entropy(student, labels, reduction='none')
        expected_relative = module.correction(hidden).square().sum(-1) / hidden.detach().square().sum(-1)
        torch.testing.assert_close(parts['native_kl'], expected_kl)
        torch.testing.assert_close(parts['ce'], expected_ce)
        torch.testing.assert_close(parts['relative_correction_l2'], expected_relative)
        torch.testing.assert_close(parts['loss'], expected_ce + .7 * expected_kl + .4 * expected_relative)
        parts['loss'].mean().backward()
        self.assertIsNone(hidden.grad)
        self.assertGreater(float(module.up.grad.norm()), 0.)
        self.assertGreater(float(module.down.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))

    def test_initial_identity_zero_kl_and_rank_scale_penalty(self):
        adapter, hidden, labels = fixture(2)
        small = SecondRoundResidual(4, config(rank=1, residual_scale=.1))
        large = SecondRoundResidual(4, config(rank=1, residual_scale=1.))
        initial = _loss_components(adapter, small, hidden, labels, small.config)
        torch.testing.assert_close(initial['native_kl'], torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(small(hidden), hidden, rtol=0, atol=0)
        with torch.no_grad():
            small.up.fill_(.1)
            large.load_state_dict(small.state_dict())
        torch.testing.assert_close(large.correction(hidden), small.correction(hidden)*10)
        a = _loss_components(adapter, small, hidden, labels, small.config)
        b = _loss_components(adapter, large, hidden, labels, large.config)
        torch.testing.assert_close(b['relative_correction_l2'], 100*a['relative_correction_l2'])
        self.assertEqual(sum(p.numel() for p in small.parameters()), 8)
        self.assertEqual(sum(p.numel() for p in SecondRoundResidual(4, config(rank=2)).parameters()), 16)
        # Inference has no labels, reference distribution, or evaluator inputs.
        self.assertEqual(small(hidden).shape, hidden.shape)

    def test_epoch_coverage_padding_chunks_and_full_teacher_work(self):
        adapter, hidden, labels = fixture(5)
        cfg = config(updates=6, eval_interval=3)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'r'
            _, summary = train_recovery_adapter(adapter, hidden, labels, evaluator(), cfg, identities(), out)
            rows = json.loads((out/'metrics.json').read_text())
            self.assertEqual([r['batch_contexts'] for r in rows], [2, 2, 1, 2, 2, 1])
            self.assertEqual(summary['context_visit_counts'], [2]*5)
            self.assertTrue(summary['coverage_complete'])
            self.assertEqual(summary['completed_epochs'], 2)
            for epoch in (1, 2):
                self.assertEqual(sorted(i for r in rows if r['epoch'] == epoch for i in r['visited_indices']), list(range(5)))
            work = summary['work_accounting']
            self.assertEqual(adapter.native_shapes, [(3, 4)]*2)
            self.assertEqual(work['optimizer_head_contexts_including_padding'], 36)
            self.assertEqual(work['optimizer_backward_valid_contexts'], 10)
            self.assertEqual(work['train_probe_head_contexts_including_padding'], 36)
            self.assertEqual(work['teacher_head_contexts_including_padding'], 36)
            self.assertEqual(work['train_probe_scored_contexts'], 15)
            self.assertEqual(work['development_evaluator_calls'], 3)
            self.assertEqual(work['total_cached_head_contexts_including_padding'], 72)

    def test_fixed_probe_padding_matches_unpadded_weighted_components(self):
        adapter, hidden, labels = fixture(5)
        cfg = config()
        module = SecondRoundResidual(4, cfg)
        with torch.no_grad():
            module.up.fill_(.1)
            expected = _loss_components(adapter, module, hidden, labels, cfg)
            actual = _training_probe(adapter, module, hidden, labels, cfg)
        for key in ('loss', 'ce', 'native_kl', 'relative_correction_l2'):
            self.assertAlmostEqual(actual[key], float(expected[key].double().mean()), places=7)

    def test_development_patience_minimum_updates_and_min_delta(self):
        adapter, hidden, labels = fixture(2)
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = train_recovery_adapter(adapter, hidden, labels, evaluator(),
                config(updates=9, eval_interval=1, development_patience=2, min_updates=4), identities(), Path(tmp)/'r')
            self.assertEqual(summary['successful_updates'], 4)
            self.assertEqual(summary['stop_reason'], 'development_patience_exhausted')
            self.assertEqual(summary['best_update'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = train_recovery_adapter(adapter, hidden, labels, evaluator([1., .9999, .99995]),
                config(updates=9, eval_interval=1, development_patience=2, min_updates=2,
                       development_min_delta=.01), identities(), Path(tmp)/'r')
            self.assertEqual(summary['successful_updates'], 2)
            self.assertEqual(summary['best_update'], 1)

    def test_adaptive_train_stop_distinct_and_does_not_select_checkpoint(self):
        adapter, hidden, labels = fixture(2)
        cfg = config(updates=6, eval_interval=1, learning_rate=.001, adaptive=dict(
            enabled=True, eval_interval=1, min_lr=.001, max_lr=.001, min_updates=0,
            stop_patience=1, patience=1, min_delta=100.))
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = train_recovery_adapter(adapter, hidden, labels, evaluator([2., 1.]),
                cfg, identities(), Path(tmp)/'r')
            self.assertEqual(summary['stop_reason'], 'adaptive_train_plateau')
            self.assertEqual(summary['successful_updates'], 1)
            self.assertEqual(summary['best_update'], 1)
            self.assertTrue(summary['adaptive_stopped'])
            self.assertEqual(summary['adaptive_controller_state']['config']['min_updates'], 0)

    def test_bf16_growth_uses_all_accepted_gradient_window(self):
        adapter, hidden, labels = fixture(2, torch.bfloat16)
        hidden.fill_(1.)
        cfg = config(updates=2, eval_interval=2, batch_contexts=2, learning_rate=1e-6,
            adaptive=dict(enabled=True, eval_interval=2, patience=1, min_delta=1e-6,
                          min_lr=1e-6, max_lr=.01, min_updates=0))
        real = _loss_components
        training_calls = 0
        def partial_zero(*args):
            nonlocal training_calls
            parts = real(*args)
            if torch.is_grad_enabled():
                training_calls += 1
                if training_calls == 2:
                    parts['loss'] = parts['loss'] * 0.
            return parts
        with tempfile.TemporaryDirectory() as tmp, patch('mixtral_lab.recovery_adapter._loss_components', side_effect=partial_zero):
            _, summary = train_recovery_adapter(adapter, hidden, labels, evaluator(), cfg, identities(), Path(tmp)/'r')
            event = summary['evaluation_history'][-1]
            self.assertEqual(event['probe_interval_gradients']['accepted_updates'], 2)
            self.assertEqual(event['probe_interval_gradients']['nonzero_gradient_updates'], 1)
            self.assertEqual(event['adaptive_decision']['action'], 'grow_lr')
            self.assertEqual(event['train_probe']['diagnostics']['native_effective_change_rate'], 0.)

    def test_guard_after_improvement_returns_native_and_does_not_count_bad_update(self):
        adapter, hidden, labels = fixture(2)
        good = evaluator([2., 1.])
        calls = 0
        def bad_dev(module):
            nonlocal calls
            calls += 1
            if calls == 3:
                return dict(good.contract, status='ok', nll=float('nan'))
            return good(module)
        bad_dev.contract = good.contract
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'r'
            module, summary = train_recovery_adapter(adapter, hidden, labels, bad_dev,
                config(updates=3, eval_interval=1), identities(), out)
            self.assertEqual(summary['status'], 'stopped_guard')
            self.assertEqual(summary['stop_reason'], 'guard_development')
            self.assertEqual(summary['successful_updates'], 1)
            self.assertEqual(summary['best_update'], 0)
            self.assertEqual(summary['selected_before_guard']['update'], 1)
            self.assertFalse(summary['deployment_eligible'])
            torch.testing.assert_close(module(hidden), hidden, rtol=0, atol=0)
            for name in ('best', 'latest'):
                loaded, meta = load_recovery_adapter_checkpoint(out/f'checkpoint_{name}.pt', identities())
                self.assertEqual(meta['run_status'], 'guard_fallback_native')
                self.assertEqual(meta['update'], 0)
                torch.testing.assert_close(loaded(hidden), hidden, rtol=0, atol=0)
            rows = json.loads((out/'metrics.json').read_text())
            self.assertFalse(rows[-1]['accepted'])
            self.assertNotIn('probe_interval_gradients', rows[-1])

    def test_nonfinite_teacher_student_gradients_and_parameters_fail_closed(self):
        for mode in ('teacher', 'student', 'gradient', 'parameters'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                adapter, hidden, labels = fixture(2)
                real_readout = differentiable_readout
                real_step = torch.optim.AdamW.step
                def readout(a, h):
                    logits = real_readout(a, h)
                    if torch.is_grad_enabled() and mode == 'student':
                        return logits * float('nan')
                    if torch.is_grad_enabled() and mode == 'gradient':
                        h.register_hook(lambda grad: torch.full_like(grad, float('nan')))
                    # Native teacher readout runs no_grad even during training.
                    if mode == 'teacher' and torch.is_grad_enabled() is False and calls[0] >= 2:
                        return logits * float('nan')
                    calls[0] += 1
                    return logits
                def step(optimizer, *a, **kw):
                    result = real_step(optimizer, *a, **kw)
                    if mode == 'parameters':
                        with torch.no_grad():
                            optimizer.param_groups[0]['params'][0].fill_(float('nan'))
                    return result
                calls = [0]
                with patch('mixtral_lab.recovery_adapter.differentiable_readout', side_effect=readout), patch.object(torch.optim.AdamW, 'step', step):
                    module, summary = train_recovery_adapter(adapter, hidden, labels, evaluator(),
                        config(updates=1, eval_interval=1), identities(), Path(tmp)/'r')
                self.assertEqual(summary['status'], 'stopped_guard')
                self.assertEqual(summary['successful_updates'], 0)
                torch.testing.assert_close(module(hidden), hidden, rtol=0, atol=0)

    def test_split_contract_provenance_and_initial_failure_summary(self):
        adapter, hidden, labels = fixture(2)
        bad_ids = identities()
        bad_ids['development']['selected_document_ids'] = ['train-a']
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'r'
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                train_recovery_adapter(adapter, hidden, labels, evaluator(), config(), bad_ids, out)
            self.assertFalse(out.exists())
            wrong = evaluator()
            wrong.contract = dict(wrong.contract, role='selection')
            with self.assertRaisesRegex(ValueError, 'declare'):
                train_recovery_adapter(adapter, hidden, labels, wrong, config(), identities(), out)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                train_recovery_adapter(adapter, hidden, labels, evaluator(), config(), identities(), out,
                    provenance=[dict(role='train', document_id='development-a', token_position=i, target=1) for i in range(2)])
            self.assertFalse(out.exists())
            invalid = evaluator()
            def callback(module):
                return dict(invalid.contract, status='failed', nll=1.)
            callback.contract = invalid.contract
            with self.assertRaisesRegex(ValueError, 'status'):
                train_recovery_adapter(adapter, hidden, labels, callback, config(), identities(), out)
            self.assertFalse(list(out.glob('*.pt')))
            failure = json.loads((out/'summary.json').read_text())
            self.assertEqual(failure['stop_reason'], 'guard_initial_development')
            self.assertFalse(failure['deployment_eligible'])
            self.assertEqual(failure['work_accounting']['development_evaluator_calls'], 1)

    def test_development_cannot_mutate_module_or_declared_contract(self):
        for mode in ('weight', 'contract', 'config', 'hidden'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                adapter, hidden, labels = fixture(2)
                callback = evaluator()
                base = callback
                def bad(module):
                    result = base(module)
                    if mode == 'weight':
                        module.up.add_(1.)
                    elif mode == 'config':
                        module.config = replace(module.config, residual_scale=.5)
                    elif mode == 'hidden':
                        module.hidden_size = 8
                    else:
                        bad.contract['protocol'] = 'cached_nll'
                    return result
                bad.contract = copy.deepcopy(base.contract)
                with self.assertRaisesRegex(ValueError, 'mutated'):
                    train_recovery_adapter(adapter, hidden, labels, bad, config(), identities(), Path(tmp)/'r')

    def test_native_cache_and_initial_probe_failures_have_summary_no_checkpoint(self):
        for mode in ('cache', 'probe'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                adapter, hidden, labels = fixture(2)
                out = Path(tmp)/'r'
                target = ('mixtral_lab.recovery_adapter._training_probe' if mode == 'probe'
                          else 'tests.test_mixtral_adapter_fit.FakeAdapter.step_pairs')
                with patch(target, side_effect=RuntimeError('synthetic initial failure')):
                    with self.assertRaisesRegex(RuntimeError, 'initial failure'):
                        train_recovery_adapter(adapter, hidden, labels, evaluator(), config(), identities(), out)
                failure = json.loads((out/'summary.json').read_text())
                self.assertEqual(failure['status'], 'failed')
                self.assertFalse(failure['deployment_eligible'])
                self.assertFalse(list(out.glob('*.pt')))

    def test_development_stop_waits_for_complete_context_coverage(self):
        adapter, hidden, labels = fixture(5)
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = train_recovery_adapter(adapter, hidden, labels, evaluator(),
                config(updates=9, eval_interval=1, batch_contexts=1, min_updates=0,
                       development_patience=1), identities(), Path(tmp)/'r')
            self.assertEqual(summary['successful_updates'], 5)
            self.assertEqual(summary['context_visit_counts'], [1]*5)
            self.assertEqual(summary['stop_reason'], 'development_patience_exhausted')

    def test_callback_exception_after_update_zero_is_a_native_guard_fallback(self):
        adapter, hidden, labels = fixture(2)
        callback = evaluator()
        calls = 0
        def broken(module):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise KeyError('malformed dev input')
            return callback(module)
        broken.contract = callback.contract
        with tempfile.TemporaryDirectory() as tmp:
            module, summary = train_recovery_adapter(adapter, hidden, labels, broken,
                config(updates=1, eval_interval=1), identities(), Path(tmp)/'r')
            self.assertEqual(summary['status'], 'stopped_guard')
            self.assertEqual(summary['stop_reason'], 'guard_development')
            torch.testing.assert_close(module(hidden), hidden, rtol=0, atol=0)

    def test_checkpoint_tampering_strict_loading_and_no_resume(self):
        adapter, hidden, labels = fixture(2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'r'
            train_recovery_adapter(adapter, hidden, labels, evaluator(), config(updates=1), identities(), out)
            path = out/'checkpoint_best.pt'
            original = path.read_bytes()
            for mode in ('state', 'config', 'source', 'development', 'shape', 'extra'):
                path.write_bytes(original)
                data = torch.load(path, weights_only=True)
                if mode == 'state':
                    data['state']['up'].add_(1.)
                elif mode == 'config':
                    data['config']['native_kl_beta'] = .5
                elif mode == 'source':
                    data['source_file_sha256']['recovery_adapter.py'] = '0'*64
                elif mode == 'development':
                    data['development']['nll'] += 1.
                elif mode == 'shape':
                    data['state']['down'] = data['state']['down'][:1]
                else:
                    data['optimizer_state'] = {}
                torch.save(data, path)
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    load_recovery_adapter_checkpoint(path, identities())
            path.write_bytes(original)
            with patch('mixtral_lab.recovery_adapter._source_hashes', return_value={}):
                with self.assertRaisesRegex(ValueError, 'source implementation'):
                    load_recovery_adapter_checkpoint(path, identities())
            with self.assertRaisesRegex(ValueError, 'expected configuration'):
                load_recovery_adapter_checkpoint(path, identities(), expected_config=config(rank=1))
            with self.assertRaisesRegex(ValueError, 'hidden size'):
                load_recovery_adapter_checkpoint(path, identities(), expected_hidden_size=8)
            other = identities()
            other['model']['id'] = 'different'
            with self.assertRaisesRegex(ValueError, 'identity'):
                load_recovery_adapter_checkpoint(path, other)

    def test_atomic_failed_write_preserves_previous_checkpoint(self):
        from mixtral_lab.recovery_adapter import _save_checkpoint
        module = SecondRoundResidual(4, config())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'test.pt'
            _save_checkpoint(path, module, label='first')
            original = path.read_bytes()
            def incomplete_save(data, target):
                Path(target).write_bytes(b'incomplete')
                raise OSError('disk full')
            with patch('mixtral_lab.recovery_adapter.torch.save', side_effect=incomplete_save):
                with self.assertRaises(OSError):
                    _save_checkpoint(path, module, label='next')
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_config_validation_bounds_and_provisional_defaults(self):
        defaults = RecoveryAdapterConfig()
        self.assertEqual((defaults.rank, defaults.residual_scale, defaults.native_kl_beta), (8, .1, .1))
        for kwargs in ({'updates': 10001}, {'updates': 10000, 'eval_interval': 1},
                       {'rank': True}, {'rank': 257}, {'native_kl_beta': -1.},
                       {'relative_correction_l2': float('nan')}, {'residual_scale': 2.},
                       {'probe_chunk_contexts': 65}, {'development_patience': 0},
                       {'min_updates': -1}, {'learning_rate': True},
                       {'adaptive': {'enabled': True, 'eval_interval': 2}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(defaults, **kwargs).validate()
        replace(defaults, updates=10000, eval_interval=16).validate()


@unittest.skipUnless(find_spec('transformers') is not None, 'optional tiny HF dependencies absent')
class RecoveryAdapterTinyHFTests(unittest.TestCase):
    def test_real_hf_objective_zero_native_and_full_document_callback(self):
        from mixtral_lab.adapter import MixtralAdapter
        from tests.test_mixtral_adapter import tiny
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                adapter = MixtralAdapter(tiny(dtype))
                train_ids = torch.tensor([[1, 2, 3, 4]])
                dev_ids = torch.tensor([[8, 9, 10, 11]])
                hidden = adapter.capture(train_ids)[:, :-1].reshape(-1, 16)
                labels = train_ids[:, 1:].flatten()
                callback = evaluator()
                def full_document(module):
                    result = adapter.forward(dev_ids, max_calls=1)
                    logits = differentiable_readout(adapter, module(result.final_hidden))
                    nll = float(F.cross_entropy(logits[:, :-1].float().reshape(-1, 32), dev_ids[:, 1:].reshape(-1)))
                    return dict(callback.contract, status='ok', nll=nll)
                full_document.contract = callback.contract
                original = {k: v.clone() for k, v in adapter.model.state_dict().items()}
                module, summary = train_recovery_adapter(adapter, hidden, labels, full_document,
                    config(updates=2, eval_interval=1, rank=4, learning_rate=.01), identities(), Path(tmp)/'r')
                self.assertEqual(summary['status'], 'completed')
                self.assertEqual(summary['initial_train_probe']['native_kl'], 0.)
                self.assertEqual(summary['trainable_parameters'], 128)
                for key, value in adapter.model.state_dict().items():
                    torch.testing.assert_close(value, original[key], rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))
