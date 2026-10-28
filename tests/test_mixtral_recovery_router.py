"""CPU, synthetic and tiny-HF recovery tests; no pretrained downloads or GPU."""
import copy
from dataclasses import replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from mixtral_lab.adapter import MixtralAdapter, action_to_pair
from mixtral_lab.full_action_fit import full_action_expected_ce
from mixtral_lab.recovery_router import (
    FORMAT, DEVELOPMENT_PROTOCOL, RecoveryRouterConfig, RecoveryPairRouter,
    compact_r1_features, cost_soft_targets, soft_target_cross_entropy,
    native_reference_kl, train_recovery_router, load_recovery_checkpoint,
    _diagnostics, _fit_passed, _digest, _tensor_hash, _tensor_hashes, _atomic_write,
)
from mixtral_lab.training import pair_actions


IDENTITIES = {key: 'toy-'+key for key in ('model', 'tokenizer', 'code')}
IDENTITIES.update(train={'selected_document_ids': [f'train-{i}' for i in range(4)]},
                  development={'selected_document_ids': ['dev-0', 'dev-1']})


def full_features(h):
    gate = h.new_zeros((len(h), 8))
    gate[:, 0], gate[:, 1] = h[:, 0], -h[:, 0]
    return torch.cat((h, h, gate, h.new_zeros((len(h), 8)),
                      h.new_tensor([0., 1., 1., 0.]).expand(len(h), -1)), -1)


def inputs(count=8):
    sign = torch.tensor([1. if i % 2 == 0 else -1. for i in range(count)])
    h = torch.stack((sign, torch.ones(count), sign*.5), -1).requires_grad_()
    y = torch.arange(count, dtype=torch.long) % 2
    provenance = [{'document_id': f'train-{i//2}', 'token_position': i,
                   'target': int(y[i]), 'role': 'train'} for i in range(count)]
    return h, y, provenance


class Landscape:
    def __init__(self, *, flat=False):
        self.model = nn.Linear(3, 3).eval().requires_grad_(False)
        self.calls, self.feature_calls = [], []
        self.flat = flat

    def features(self, h):
        assert not torch.is_grad_enabled()
        self.feature_calls.append(tuple(h.shape))
        return full_features(h)

    def native_pair(self, h):
        assert not torch.is_grad_enabled()
        return torch.tensor([[1, 0]], device=h.device).expand(len(h), -1)

    def score_pairs(self, h, targets, pairs):
        assert not torch.is_grad_enabled()
        assert h.shape == (1, 3) and targets.shape == (1,) and pairs.shape == (1, 2)
        action = pair_actions(pairs)
        self.calls.append((int(targets[0]), int(action[0]), tuple(h.shape)))
        if self.flat:
            return h.new_tensor([1.])
        oracle = torch.where(targets == 0, 1, 27)
        return torch.where(action == oracle, .25, torch.where(action == 0, 1.5, 2.75)) + h.sum(-1)*0.


class Development:
    """Controlled independent metric; it never reads the train table."""
    def __init__(self, nlls=(3., 2., 1.)):
        self.contract = {'role': 'development', 'protocol': DEVELOPMENT_PROTOCOL,
                         'document_ids': ['dev-0', 'dev-1']}
        self.nlls, self.calls = list(nlls), []

    def __call__(self, actor):
        assert not torch.is_grad_enabled()
        assert not actor.training
        assert set(actor.state_dict()) == {'weight', 'bias'}
        assert not hasattr(actor, 'costs') and not hasattr(actor, 'targets')
        self.calls.append(copy.deepcopy(actor.state_dict()))
        return {**self.contract, 'status': 'ok', 'nll': self.nlls[min(len(self.calls)-1, len(self.nlls)-1)]}


def fit(path, *, adapter=None, development=None, config=None, h=None, y=None, provenance=None, identities=None):
    default_h, default_y, default_rows = inputs()
    return train_recovery_router(adapter or Landscape(), default_h if h is None else h,
        default_y if y is None else y, development or Development(),
        config or RecoveryRouterConfig(updates=40, learning_rate=.08, eval_interval=20),
        IDENTITIES if identities is None else identities, path,
        provenance=default_rows if provenance is None else provenance)


class RecoveryRouterTests(unittest.TestCase):
    def test_soft_target_ce_gradient_is_p_minus_q_and_costs_detached(self):
        logits = torch.linspace(-2., 2., 56, dtype=torch.float64).reshape(2, 28).requires_grad_()
        costs = torch.linspace(.01, 3., 56, dtype=torch.float64).reshape(2, 28).requires_grad_()
        temperature = .23
        q = cost_soft_targets(costs, temperature)
        self.assertFalse(q.requires_grad)
        loss = soft_target_cross_entropy(logits.log_softmax(-1), costs, temperature)
        loss.backward()
        torch.testing.assert_close(logits.grad, (logits.detach().softmax(-1)-q)/2, rtol=1e-12, atol=1e-14)
        self.assertIsNone(costs.grad)
        shifted = cost_soft_targets(costs.detach()+1000., temperature)
        torch.testing.assert_close(q, shifted, rtol=1e-12, atol=1e-14)

    def test_soft_target_recovers_bad_saturation_expected_cost_cannot(self):
        initial = torch.full((1, 28), -40., dtype=torch.float64)
        initial[0, 0] = 40.
        costs = torch.full((1, 28), 2., dtype=torch.float64)
        costs[0, 27] = 0.
        saturated = initial.clone().requires_grad_()
        full_action_expected_ce(saturated.log_softmax(-1), costs).backward()
        self.assertLess(float(saturated.grad.abs().max()), 1e-30)
        logits = nn.Parameter(initial.clone())
        optimizer = torch.optim.SGD([logits], lr=5.)
        for _ in range(30):
            optimizer.zero_grad()
            soft_target_cross_entropy(logits.log_softmax(-1), costs, .1).backward()
            self.assertTrue(bool(torch.isfinite(logits.grad).all()))
            optimizer.step()
        self.assertEqual(int(logits.argmax(-1)), 27)
        self.assertGreater(float(logits.detach().softmax(-1)[0, 27]), .99)

    def test_compact_features_scale_and_zero_native_identity(self):
        h = torch.ones((2, 1024))
        features = full_features(h)
        native = torch.tensor([[7, 6], [1, 0]])
        router = RecoveryPairRouter(features.shape[1])
        self.assertEqual(router.compact_feature_dim, 1032)
        self.assertEqual(router.weight.shape, (28, 1032))
        self.assertEqual(router.residual_scale, 1/math.sqrt(1032))
        self.assertTrue(torch.equal(router.greedy(features, native), pair_actions(native)+1))
        compact = compact_r1_features(features)
        expected_compact = torch.cat((h, features[:, 2048:2056]), -1)
        expected_compact *= torch.rsqrt(expected_compact.square().mean(-1, keepdim=True)+1e-6)
        torch.testing.assert_close(compact, expected_compact)
        with torch.no_grad():
            router.weight.fill_(.01)
            router.bias.fill_(.1)
        residual = router.logits(features, native)-router.reference_log_probs(native)
        unscaled = torch.nn.functional.linear(compact, router.weight, router.bias)
        torch.testing.assert_close(residual, unscaled/math.sqrt(1032))
        self.assertGreater(float((unscaled.abs().max()/residual.abs().max()).detach()), 30.)
        for column in (1024, -1, -5):
            bad = features.clone()
            bad[:, column] += 1
            with self.assertRaisesRegex(ValueError, 'R1 observations'):
                router.log_probs(bad, native)

    def test_static_reference_kl_is_not_step_kl(self):
        h, _, _ = inputs()
        actor = RecoveryPairRouter(26)
        features, native = full_features(h), torch.tensor([[0, 1]]).expand(8, -1)
        before = actor.log_probs(features, native).detach()
        static = actor.reference_log_probs(native)
        self.assertLess(abs(float(native_reference_kl(before, static))), 1e-6)
        with torch.no_grad():
            actor.bias[27] = 20.
        current = actor.log_probs(features, native)
        self.assertGreater(float(native_reference_kl(current, static).detach()), 1.)
        self.assertEqual(float(native_reference_kl(current, current.detach()).detach()), 0.)

    def test_fits_frozen_table_but_checkpoint_selected_only_by_development(self):
        h, y, rows = inputs()
        adapter = Landscape()
        original = copy.deepcopy(adapter.model.state_dict())
        rng = torch.get_rng_state().clone()
        development = Development((1., 2., 3.))  # Deliberately oppose improving train fit.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            actor, summary = fit(out, adapter=adapter, development=development, h=h, y=y, provenance=rows)
            self.assertEqual(summary['best_update'], 0)
            self.assertEqual(summary['best_development_nll'], 1.)
            self.assertFalse(summary['train_fit_passed'])
            self.assertTrue(summary['latest_train_fit_passed'])
            self.assertLess(summary['latest_train_metrics']['greedy_ce'], summary['initial_train_metrics']['greedy_ce'])
            self.assertEqual(summary['initial_train_metrics']['greedy_change_count'], 0)
            self.assertTrue(torch.equal(actor.weight, torch.zeros_like(actor.weight)))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertIsNone(h.grad)
            for key, value in adapter.model.state_dict().items():
                self.assertTrue(torch.equal(value, original[key]))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in adapter.model.parameters()))
            self.assertEqual(len(adapter.calls), 240)
            self.assertEqual(adapter.feature_calls, [(1, 3)]*8)
            self.assertEqual(summary['score_work_totals']['training_score_calls'], 0)
            self.assertEqual(summary['training_coverage']['minimum_context_visits'], 40)
            self.assertEqual(len(development.calls), 3)
            self.assertEqual([row['update'] for row in summary['development_history']], [0, 20, 40])
            self.assertEqual(len(list((out/'evaluation_checkpoints').glob('checkpoint_*.pt'))), 3)
            for name in ('initial', 'best', 'latest'):
                loaded, meta = load_recovery_checkpoint(out/f'checkpoint_{name}.pt', IDENTITIES)
                self.assertFalse(loaded.training)
                self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
                self.assertEqual(meta['format'], FORMAT)
                self.assertEqual(meta['historical_cached_inputs'], {'h0': _tensor_hash(h), 'targets': _tensor_hash(y)})
                self.assertIn('bridge.py', meta['source_sha256'])
                self.assertIn('recovery_protocol.py', meta['source_sha256'])
                self.assertEqual(meta['checkpoint_sha256'], hashlib.sha256((out/f'checkpoint_{name}.pt').read_bytes()).hexdigest())
            self.assertFalse(list(out.rglob('*.tmp')))
            records = json.loads((out/'metrics.json').read_text())
            self.assertEqual(len(records), 41)
            for key in ('gain_mean_nats', 'gain_median_nats', 'gain_worst_nats', 'gain_positive_fraction',
                        'top2_positive_gain_share', 'greedy_oracle_regret_nats', 'oracle_agreement',
                        'entropy', 'max_abs_logit', 'probability_saturation_fraction',
                        'native_reference_kl', 'step_kl_current_to_previous'):
                self.assertTrue(math.isfinite(records[-1]['train'][key]), key)

    def test_development_best_can_be_intermediate_and_ties_choose_earliest(self):
        cfg = RecoveryRouterConfig(updates=12, eval_interval=4, learning_rate=.08)
        with tempfile.TemporaryDirectory() as tmp:
            actor, summary = fit(Path(tmp)/'run', config=cfg, development=Development((3., 1., 1., 2.)))
            self.assertEqual(summary['best_update'], 4)
            loaded, _ = load_recovery_checkpoint(Path(tmp)/'run'/'evaluation_checkpoints'/'checkpoint_000004.pt', IDENTITIES)
            for key in actor.state_dict():
                torch.testing.assert_close(actor.state_dict()[key], loaded.state_dict()[key], rtol=0, atol=0)

    def test_development_patience_waits_for_min_updates_and_zero_included(self):
        cfg = RecoveryRouterConfig(updates=20, eval_interval=2, development_patience=2, min_updates=6)
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = fit(Path(tmp)/'run', config=cfg, development=Development((1.,)))
            self.assertEqual(summary['successful_updates'], 6)
            self.assertEqual(summary['stop_reason'], 'development_patience')
            self.assertEqual(summary['best_update'], 0)
            self.assertEqual([row['update'] for row in summary['development_history']], [0, 2, 4, 6])
            _, zero = fit(Path(tmp)/'zero', config=replace(cfg, updates=0))
            self.assertEqual(zero['successful_updates'], 0)
            self.assertEqual(len(zero['development_history']), 1)

    def test_development_stale_accumulates_before_min_updates_but_requires_coverage(self):
        cfg = RecoveryRouterConfig(updates=10, eval_interval=1, development_patience=2, min_updates=4)
        with tempfile.TemporaryDirectory() as tmp:
            _, covered = fit(Path(tmp)/'covered', config=cfg, development=Development((1.,)))
            self.assertEqual(covered['successful_updates'], 4)
            self.assertEqual(covered['stop_reason'], 'development_patience')
            self.assertEqual([row['patience_observations'] for row in covered['development_history']], [0, 1, 2, 3, 4])
            # Eight rows and batch one: min_updates and patience are already
            # satisfied at four, but stopping must wait until all eight rows
            # have received an optimizer update.
            _, incomplete = fit(Path(tmp)/'coverage', config=replace(cfg, batch_contexts=1),
                                development=Development((1.,)))
            self.assertEqual(incomplete['successful_updates'], 8)
            self.assertEqual(incomplete['stop_reason'], 'development_patience')
            self.assertEqual(incomplete['training_coverage']['minimum_context_visits'], 1)
            self.assertEqual(incomplete['development_history'][-1]['patience_observations'], 8)
            self.assertEqual(incomplete['best_update'], 0)

    def test_adaptive_feedback_is_fixed_train_objective_and_not_development(self):
        cfg = RecoveryRouterConfig(updates=4, eval_interval=2, adaptive={
            'enabled': True, 'eval_interval': 1, 'min_updates': 0, 'patience': 1,
            'stop_patience': 2, 'min_delta': 1., 'min_lr': .001, 'max_lr': .001})
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = fit(Path(tmp)/'run', config=cfg, development=Development((100., 50., 1.)))
            records = json.loads((Path(tmp)/'run'/'metrics.json').read_text())
            for decision in summary['adaptive']['decisions']:
                row = next(row for row in records if row['update'] == decision['update'])
                self.assertEqual(decision['train_loss'], row['fixed_train_objective'])
            self.assertEqual(summary['stop_reason'], 'adaptive_train_plateau')
            load_recovery_checkpoint(Path(tmp)/'run'/'checkpoint_latest.pt', IDENTITIES)

    def test_fit_gate_rejects_two_token_concentration(self):
        cfg = RecoveryRouterConfig()
        self.assertFalse(_fit_passed({'gain_mean_nats': 10., 'gain_positive_fraction': .9,
                                     'gain_positive_count': 2}, cfg))
        self.assertFalse(_fit_passed({'gain_mean_nats': 10., 'gain_positive_fraction': .1,
                                     'gain_positive_count': 10}, cfg))
        self.assertTrue(_fit_passed({'gain_mean_nats': 1., 'gain_positive_fraction': .75,
                                    'gain_positive_count': 6}, cfg))

    def test_diagnostics_greedy_matches_deployed_fp32_near_tie(self):
        h, _, _ = inputs()
        features = full_features(h[:1])
        native = action_to_pair(torch.tensor([11]))
        actor = RecoveryPairRouter(features.shape[1])
        with torch.no_grad():
            actor.bias.copy_(-actor.reference_log_probs(native)[0]/actor.residual_scale)
            actor.bias[10] = torch.nextafter(actor.bias[10], actor.bias.new_tensor(float('inf')))
        # Valid finite checkpoint weights make the native raw logit slightly
        # larger, but FP32 log-softmax rounds it into a tie with action one.
        self.assertEqual(int(actor.greedy(features, native)), 1)
        self.assertEqual(int(actor.logits(features, native).double().log_softmax(-1).argmax(-1))+1, 11)
        costs = torch.ones((1, 28), dtype=torch.float64)
        costs[0, 10] = .1
        metrics, _ = _diagnostics(actor, {'features': features, 'native_pairs': native, 'costs': costs})
        self.assertEqual(metrics['greedy_change_count'], 1)
        self.assertEqual(metrics['greedy_action_counts'][0], 1)
        self.assertAlmostEqual(metrics['greedy_gain_nats'], -.9)

    def test_distribution_diagnostics_expose_concentrated_gains_and_regret(self):
        h, _, _ = inputs()
        actor = RecoveryPairRouter(26)
        with torch.no_grad():
            actor.bias[1] = 40.
        costs = torch.full((8, 28), 4., dtype=torch.float64)
        costs[:, 0] = 1.
        costs[:2, 1], costs[2:, 1] = 0., 1.1
        table = {'features': full_features(h), 'native_pairs': torch.tensor([[0, 1]]).expand(8, -1),
                 'costs': costs}
        metrics, _ = _diagnostics(actor, table)
        self.assertAlmostEqual(metrics['gain_mean_nats'], .175)
        self.assertAlmostEqual(metrics['gain_median_nats'], -.1)
        self.assertAlmostEqual(metrics['gain_worst_nats'], -.1)
        self.assertEqual(metrics['gain_positive_fraction'], .25)
        self.assertEqual(metrics['top2_positive_gain_share'], 1.)
        self.assertAlmostEqual(metrics['greedy_oracle_regret_nats'], .075)
        self.assertEqual(metrics['probability_saturation_fraction'], 1.)
        self.assertFalse(_fit_passed(metrics, RecoveryRouterConfig()))

    def test_identity_provenance_and_development_leakage_rejected_before_scoring(self):
        cases = []
        selection = Development(); selection.contract['role'] = 'selection'; cases.append(selection)
        test = Development(); test.contract['role'] = 'final_test'; cases.append(test)
        overlap = Development(); overlap.contract['document_ids'] = ['train-0']; cases.append(overlap)
        invalid = Development(); invalid.contract['protocol'] = 'cached_singletons'; cases.append(invalid)
        for development in cases:
            adapter = Landscape()
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                fit(Path(tmp)/'run', adapter=adapter, development=development)
            self.assertEqual(adapter.calls, [])
        h, y, rows = inputs()
        rows[0]['role'] = 'selection'
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'train provenance'):
            fit(Path(tmp)/'run', provenance=rows)
        signature = inspect.signature(train_recovery_router)
        self.assertEqual([key for key in signature.parameters if 'evaluator' in key], ['development_evaluator'])

    def test_cost_budget_and_nonfinite_inputs_scores_results_fail_closed(self):
        cfg = RecoveryRouterConfig(max_score_calls=239)
        adapter = Landscape()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'budget'):
            fit(Path(tmp)/'run', adapter=adapter, config=cfg)
        self.assertEqual(adapter.calls, [])
        for value in (float('nan'), float('inf'), -1.):
            adapter = Landscape()
            with tempfile.TemporaryDirectory() as tmp, patch.object(adapter, 'score_pairs', return_value=torch.tensor([value])):
                with self.assertRaisesRegex(ValueError, 'score'):
                    fit(Path(tmp)/'run', adapter=adapter)
                result = json.loads((Path(tmp)/'run'/'summary.json').read_text())
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['score_work_totals']['actual_score_calls'], 1)
        development = Development((float('nan'),))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            fit(Path(tmp)/'run', development=development)
        h, y, rows = inputs(); h = h.detach(); h[0, 0] = torch.inf
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'finite'):
            fit(Path(tmp)/'run', h=h)

    def test_callback_mutation_and_protocol_change_rejected(self):
        class Mutator(Development):
            def __call__(self, actor):
                result = super().__call__(actor)
                actor.bias[0] += 1
                return result
        class DifferentDocs(Development):
            def __call__(self, actor):
                result = super().__call__(actor)
                result['document_ids'] = ['wrong']
                return result
        class ConfigMutator(Development):
            def __call__(self, actor):
                result = super().__call__(actor)
                actor.epsilon = .5
                return result
        class FailedResult(Development):
            def __call__(self, actor):
                return {**super().__call__(actor), 'status': 'failed'}
        for development in (Mutator(), DifferentDocs(), ConfigMutator(), FailedResult()):
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                fit(Path(tmp)/'run', development=development)

    def test_native_anchor_frozen_backbone_and_identity_coverage_checks(self):
        adapter = Landscape()
        calls = 0
        original = adapter.score_pairs
        def inconsistent(*args):
            nonlocal calls
            calls += 1
            return original(*args) + (0.1 if calls == 30 else 0.)
        with tempfile.TemporaryDirectory() as tmp, patch.object(adapter, 'score_pairs', side_effect=inconsistent):
            with self.assertRaisesRegex(ValueError, 'anchor'):
                fit(Path(tmp)/'run', adapter=adapter)
        adapter = Landscape(); adapter.model.train()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'frozen'):
            fit(Path(tmp)/'run', adapter=adapter)
        missing = copy.deepcopy(IDENTITIES)
        missing['train']['selected_document_ids'] = ['train-0']
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'provenance documents'):
            fit(Path(tmp)/'run', identities=missing)
        overlap = copy.deepcopy(IDENTITIES)
        overlap['train']['selected_document_ids'].append('dev-0')
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'disjoint'):
            fit(Path(tmp)/'run', identities=overlap)

    def test_checkpoint_tamper_identity_state_shape_source_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            fit(out, config=RecoveryRouterConfig(updates=0))
            original = torch.load(out/'checkpoint_best.pt', weights_only=True)
            with self.assertRaisesRegex(ValueError, 'identities'):
                load_recovery_checkpoint(out/'checkpoint_best.pt', {**IDENTITIES, 'train': 'wrong'})
            tampered = copy.deepcopy(original); tampered['actor_state_dict']['weight'][0, 0] = 1.
            torch.save(tampered, out/'bad.pt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_recovery_checkpoint(out/'bad.pt', IDENTITIES)
            tampered = copy.deepcopy(original); tampered['update'] = 1
            torch.save(tampered, out/'bad.pt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_recovery_checkpoint(out/'bad.pt', IDENTITIES)
            tampered = copy.deepcopy(original)
            tampered['actor_state_dict']['weight'] = torch.zeros((28, 12))
            tampered['actor_state_hashes'] = _tensor_hashes(tampered['actor_state_dict'])
            tampered['payload_sha256'] = _digest({key: value for key, value in tampered.items()
                                                 if key not in ('payload_sha256', 'actor_state_dict')})
            torch.save(tampered, out/'bad.pt')
            with self.assertRaisesRegex(ValueError, 'state'):
                load_recovery_checkpoint(out/'bad.pt', IDENTITIES)
            with patch('mixtral_lab.recovery_router._source_hashes', return_value={'changed': 'source'}):
                with self.assertRaisesRegex(ValueError, 'source'):
                    load_recovery_checkpoint(out/'checkpoint_best.pt', IDENTITIES)

    def test_atomic_write_failure_preserves_previous_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.pt'; path.write_bytes(b'previous')
            def broken(temporary):
                temporary.write_bytes(b'partial')
                raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError):
                _atomic_write(path, broken)
            self.assertEqual(path.read_bytes(), b'previous')
            self.assertEqual([entry.name for entry in Path(tmp).iterdir()], ['checkpoint.pt'])

    def test_configuration_is_bounded_and_expected_ce_arm_runs(self):
        invalid = ({'updates': True}, {'updates': 10001}, {'temperature': 0.}, {'temperature': 1e-10},
                   {'epsilon': 1.}, {'learning_rate': float('nan')}, {'reference_kl_weight': -1.},
                   {'updates': 10000, 'eval_interval': 1}, {'min_updates': -1}, {'fit_min_positive_contexts': 2}, {'objective': 'oracle_feature'},
                   {'adaptive': {'enabled': True, 'min_lr': .02}}, {'development_patience': 0})
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RecoveryRouterConfig(**kwargs).validate()
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = fit(Path(tmp)/'run', config=RecoveryRouterConfig(updates=2, objective='expected_ce'))
            self.assertEqual(summary['status'], 'complete')

    def test_tiny_hf_zero_router_exact_online_native_and_roundtrip(self):
        from tests.test_mixtral_adapter import tiny
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                adapter = MixtralAdapter(tiny(dtype))
                ids = torch.tensor([[1, 2, 3, 4]])
                actor = RecoveryPairRouter(adapter.feature_dim)
                native = adapter.forward(ids)
                routed = adapter.forward(ids, policy=actor)
                torch.testing.assert_close(routed.logits, native.logits, rtol=0, atol=0)
                h = adapter.capture(ids).reshape(-1, adapter.hidden_size)[:3]
                targets = ids.flatten()[1:]
                rows = [{'document_id': 'train-hf', 'token_position': index, 'role': 'train'} for index in range(3)]
                with tempfile.TemporaryDirectory() as tmp:
                    trained, summary = fit(Path(tmp)/'run', adapter=adapter, h=h, y=targets, provenance=rows,
                        config=RecoveryRouterConfig(updates=0, batch_contexts=3),
                        identities={**IDENTITIES, 'train': {'selected_document_ids': ['train-hf']}})
                    loaded, _ = load_recovery_checkpoint(Path(tmp)/'run'/'checkpoint_best.pt',
                        {**IDENTITIES, 'train': {'selected_document_ids': ['train-hf']}})
                    torch.testing.assert_close(adapter.forward(ids, policy=loaded).logits, native.logits, rtol=0, atol=0)
                    self.assertEqual(summary['score_work_totals']['actual_score_calls'], 90)


if __name__ == '__main__':
    unittest.main()
