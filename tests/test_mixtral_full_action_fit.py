"""CPU/no downloads: exhaustive train fit, isolation, budgets and artifact safety."""
import copy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from mixtral_lab.adapter import PAIR_TABLE, MixtralAdapter
from mixtral_lab.adaptive_fit import AdaptiveFitController
from mixtral_lab.full_action_fit import (
    FORMAT, FullActionFitConfig, full_action_expected_ce,
    train_full_action_router, load_full_action_checkpoint,
    _digest, _tensor_hash, _source_hashes,
)
from mixtral_lab.training import ResidualPairRouter, pair_actions


IDENTITIES = {key: 'toy-'+key for key in ('model', 'tokenizer', 'train', 'code')}


def inputs(count=8):
    sign = torch.tensor([1. if i % 2 == 0 else -1. for i in range(count)])
    h = torch.stack((sign, torch.ones(count), sign*.5), -1).requires_grad_()
    y = torch.arange(count, dtype=torch.long) % 2
    provenance = [{'document_id': f'doc-{i//2}', 'token_position': i,
                   'target': int(y[i]), 'role': 'train'} for i in range(count)]
    return h, y, provenance


class Landscape:
    """A linearly learnable, target-aware cost table, not a real Mixtral score."""
    def __init__(self, *, flat=False):
        self.model = nn.Linear(3, 3).eval().requires_grad_(False)
        self.calls, self.feature_calls, self.native_calls = [], [], []
        self.flat = flat

    def features(self, h):
        assert not torch.is_grad_enabled()
        self.feature_calls.append(tuple(h.shape))
        return h

    def native_pair(self, h):
        assert not torch.is_grad_enabled()
        self.native_calls.append(tuple(h.shape))
        # Reversed native order exercises canonical candidate identity.
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


def fit(directory, *, adapter=None, config=None, h=None, y=None, provenance=None, identities=None):
    default_h, default_y, default_provenance = inputs()
    return train_full_action_router(
        adapter or Landscape(), default_h if h is None else h, default_y if y is None else y,
        config or FullActionFitConfig(updates=40, learning_rate=.08),
        IDENTITIES if identities is None else identities, directory,
        provenance=default_provenance if provenance is None else provenance)


class FullActionTests(unittest.TestCase):
    def test_exact_expected_ce_gradient_all_actions_and_detached_costs(self):
        logits = torch.linspace(-1., 1., 56, dtype=torch.float64).reshape(2, 28).requires_grad_()
        costs = (torch.arange(56, dtype=torch.float64).reshape(2, 28)/10).requires_grad_()
        logp = logits.log_softmax(-1)
        loss = full_action_expected_ce(logp, costs)
        p = logits.softmax(-1).detach()
        expected = (p*costs.detach()).sum(-1)
        torch.testing.assert_close(loss, expected.mean())
        loss.backward()
        analytic = p*(costs.detach()-expected[:, None])/2
        torch.testing.assert_close(logits.grad, analytic, atol=1e-14, rtol=1e-12)
        self.assertEqual(int(torch.count_nonzero(logits.grad)), 56)
        self.assertIsNone(costs.grad)
        with self.assertRaises(ValueError):
            full_action_expected_ce(logp[:, :27], costs[:, :27])

    def test_learns_greedy_train_fit_without_backbone_or_global_rng_changes(self):
        h, y, provenance = inputs()
        adapter = Landscape()
        original = copy.deepcopy(adapter.model.state_dict())
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'fit'
            actor, summary = fit(output, adapter=adapter, h=h, y=y, provenance=provenance)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertIsNone(h.grad)
            self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))
            for key, value in adapter.model.state_dict().items():
                self.assertTrue(torch.equal(original[key], value))
            self.assertEqual(summary['status'], 'complete')
            self.assertTrue(summary['train_fit_passed'])
            self.assertGreater(summary['best_update'], 0)
            before, after = summary['initial_train_metrics'], summary['best_train_metrics']
            self.assertEqual(before['greedy_change_count'], 0)
            self.assertAlmostEqual(before['greedy_ce'], 1.5)
            self.assertEqual(after['greedy_change_count'], 8)
            self.assertAlmostEqual(after['greedy_ce'], .25)
            self.assertAlmostEqual(after['oracle_ce'], .25)
            self.assertAlmostEqual(after['greedy_oracle_regret_nats'], 0.)
            self.assertGreater(after['greedy_gain_nats'], 1.2)
            self.assertLess(after['expected_ce'], before['expected_ce'])
            self.assertGreater(after['expected_ce'], after['greedy_ce'])
            self.assertEqual(len(adapter.calls), 240)
            self.assertEqual(adapter.feature_calls, [(1, 3)]*8)
            self.assertEqual(adapter.native_calls, [(1, 3)]*8)
            for key in ('actual_score_calls', 'actual_pair_rows', 'actual_head_calls', 'actual_head_rows'):
                self.assertEqual(summary['score_work_totals'][key], 240)
            self.assertEqual(summary['score_work_totals']['training_score_calls'], 0)
            self.assertEqual(summary['training_coverage']['optimizer_context_visits'], 40*8)
            self.assertEqual(summary['training_coverage']['minimum_context_visits'], 40)
            self.assertEqual(summary['training_coverage']['unique_train_contexts_visited'], 8)
            self.assertFalse(actor.training)
            self.assertEqual(set(actor.state_dict()), {'weight', 'bias'})
            records = json.loads((output/'metrics.json').read_text())
            self.assertEqual([row['update'] for row in records], list(range(41)))
            best = min(records, key=lambda row: (row['train']['greedy_ce'], row['train']['expected_ce']))
            self.assertEqual(best['update'], summary['best_update'])
            self.assertGreater(records[1]['gradient_norm'], 0.)
            probe = json.loads((output/'train_probe.json').read_text())
            self.assertEqual(probe['before']['greedy_change_count'], 0)
            self.assertEqual(probe['best']['greedy_change_count'], 8)
            self.assertEqual([row['context_index'] for row in probe['before']['rows']], probe['indices'])
            loaded, metadata = load_full_action_checkpoint(output/'checkpoint_best.pt', IDENTITIES)
            self.assertTrue(metadata['train_fit_passed'])
            self.assertEqual(metadata['update'], summary['best_update'])
            self.assertEqual(metadata['binding_sha256'], summary['binding_sha256'])
            self.assertEqual(metadata['checkpoint_sha256'], hashlib.sha256((output/'checkpoint_best.pt').read_bytes()).hexdigest())
            for key, value in actor.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
            self.assertFalse(list(output.glob('*.tmp')))

    def test_table_binding_canonical_native_anchors_and_train_only_api(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'fit'
            fit(output, config=FullActionFitConfig(updates=0))
            contract = json.loads((output/'training_contract.json').read_text())
            table = torch.load(output/'train_score_table.pt', weights_only=True)
            text = json.loads((output/'train_score_table.json').read_text())
            self.assertEqual(table['binding'], contract)
            self.assertEqual(contract['role'], 'train')
            self.assertEqual(contract['source_sha256'], _source_hashes())
            self.assertFalse(contract['numeric_protocol']['cross_shape_equivalence_established'])
            for key, tensor in table['tensors'].items():
                self.assertEqual(contract['score_table_hashes'][key], _tensor_hash(tensor))
                self.assertFalse(tensor.requires_grad)
            self.assertEqual(contract['train_targets'], _tensor_hash(table['targets']))
            self.assertEqual(contract['provenance_sha256'], _digest(table['provenance']))
            self.assertEqual(text['binding_sha256'], contract['sha256'])
            for row in text['contexts']:
                self.assertEqual(row['native_action'], 1)
                self.assertEqual(row['native_pair'], [1, 0])
                self.assertEqual(row['scores_nats'][0], 1.5)
                self.assertEqual(row['native_anchors'], [1.5, 1.5])
        signature = str(inspect.signature(train_full_action_router))
        self.assertNotIn('selection', signature)
        self.assertNotIn('evaluator', signature)

    def test_fixed_local_rng_deterministic_probe_and_actor(self):
        cfg = FullActionFitConfig(updates=8, batch_contexts=3, learning_rate=.08, probe_contexts=3)
        with tempfile.TemporaryDirectory() as temp:
            actors, records, probes = [], [], []
            for index in range(2):
                torch.rand(5+index)  # different global RNG states do not affect fit
                output = Path(temp)/str(index)
                actor, _ = fit(output, config=cfg)
                actors.append(copy.deepcopy(actor.state_dict()))
                records.append(json.loads((output/'metrics.json').read_text()))
                probes.append(json.loads((output/'train_probe.json').read_text()))
            self.assertEqual(records[0], records[1])
            self.assertEqual(probes[0], probes[1])
            for key in actors[0]:
                torch.testing.assert_close(actors[0][key], actors[1][key], atol=0, rtol=0)

    def test_expected_ce_improvement_without_greedy_change_does_not_pass(self):
        cfg = FullActionFitConfig(updates=1, learning_rate=1e-4, fit_gain_threshold=1e-9)
        with tempfile.TemporaryDirectory() as temp:
            _, result = fit(Path(temp)/'run', config=cfg)
            self.assertLess(result['best_train_metrics']['expected_ce'], result['initial_train_metrics']['expected_ce'])
            self.assertEqual(result['best_train_metrics']['greedy_change_count'], 0)
            self.assertFalse(result['train_fit_passed'])
            _, high_threshold = fit(Path(temp)/'threshold', config=replace(cfg, updates=40, learning_rate=.08, fit_gain_threshold=2.))
            self.assertGreater(high_threshold['best_train_metrics']['greedy_change_count'], 0)
            self.assertFalse(high_threshold['train_fit_passed'])

    def test_update_zero_is_a_checkpoint_and_cannot_pass_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'zero'
            actor, result = fit(output, config=FullActionFitConfig(updates=0))
            self.assertEqual(result['best_update'], 0)
            self.assertEqual(result['successful_updates'], 0)
            self.assertFalse(result['train_fit_passed'])
            self.assertEqual(int(torch.count_nonzero(actor.weight)), 0)
            for kind in ('initial', 'best', 'latest'):
                _, metadata = load_full_action_checkpoint(output/f'checkpoint_{kind}.pt', IDENTITIES)
                self.assertEqual(metadata['update'], 0)
                self.assertFalse(metadata['train_fit_passed'])
            self.assertEqual(len(json.loads((output/'metrics.json').read_text())), 1)

    def test_update_zero_remains_best_when_updates_worsen_train_greedy_ce(self):
        def bad_step(optimizer, closure=None):
            with torch.no_grad():
                # Deliberately bad optimizer proposal, still finite and R1-legal.
                weight, bias = optimizer.param_groups[0]['params']
                weight.zero_()
                bias.zero_()
                bias[5] = 20.
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'regression'
            with patch.object(torch.optim.AdamW, 'step', bad_step):
                actor, summary = fit(output, config=FullActionFitConfig(updates=2))
            self.assertEqual(summary['best_update'], 0)
            self.assertFalse(summary['train_fit_passed'])
            self.assertGreater(summary['latest_train_metrics']['greedy_ce'],
                               summary['initial_train_metrics']['greedy_ce'])
            self.assertEqual(int(torch.count_nonzero(actor.weight)), 0)
            self.assertEqual(int(torch.count_nonzero(actor.bias)), 0)
            _, best = load_full_action_checkpoint(output/'checkpoint_best.pt', IDENTITIES)
            _, latest = load_full_action_checkpoint(output/'checkpoint_latest.pt', IDENTITIES)
            self.assertEqual(best['update'], 0)
            self.assertEqual(latest['update'], 2)

    def test_budget_preflight_and_failed_actual_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = Landscape()
            output = Path(temp)/'over'
            with self.assertRaisesRegex(ValueError, 'budget'):
                fit(output, adapter=adapter, config=FullActionFitConfig(max_score_calls=239))
            self.assertEqual(adapter.calls, [])
            self.assertFalse(output.exists())
            for index, bad in enumerate((float('nan'), float('inf'), -1.)):
                output = Path(temp)/f'bad-{index}'
                with patch.object(adapter, 'score_pairs', return_value=torch.tensor([bad])):
                    with self.assertRaisesRegex(ValueError, 'singleton train score'):
                        fit(output, adapter=adapter)
                result = json.loads((output/'summary.json').read_text())
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['train_fit_passed'])
                self.assertEqual(result['score_work_totals']['actual_score_calls'], 1)
                self.assertFalse((output/'checkpoint_best.pt').exists())
            output = Path(temp)/'anchor'
            original = adapter.score_pairs
            calls = 0
            def mismatch(*args):
                nonlocal calls
                calls += 1
                return original(*args) + (1. if calls == 30 else 0.)
            with patch.object(adapter, 'score_pairs', side_effect=mismatch):
                with self.assertRaisesRegex(ValueError, 'native anchor CE mismatch'):
                    fit(output, adapter=adapter)
            self.assertEqual(json.loads((output/'summary.json').read_text())['score_work_totals']['actual_score_calls'], 30)

    def test_identity_state_config_source_and_metadata_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'fit'
            fit(output, config=FullActionFitConfig(updates=0))
            path = output/'checkpoint_best.pt'
            with self.assertRaisesRegex(ValueError, 'identities'):
                load_full_action_checkpoint(path, dict(IDENTITIES, train='wrong'))
            changes = [lambda value: value['actor_state_dict']['weight'].add_(1.),
                       lambda value: value['binding']['config'].update(epsilon=.7),
                       lambda value: value.update(update=1),
                       lambda value: value.update(feature_dim=4),
                       lambda value: value.update(train_fit_passed=True),
                       lambda value: value['train_metrics'].update(greedy_ce=0.),
                       lambda value: value.update(format='mixtral-r1-router-v1')]
            original = torch.load(path, weights_only=True)
            for index, change in enumerate(changes):
                payload = copy.deepcopy(original)
                change(payload)
                bad = output/f'bad-{index}.pt'
                torch.save(payload, bad)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    load_full_action_checkpoint(bad, IDENTITIES)
            with patch('mixtral_lab.full_action_fit._source_hashes', return_value={'changed': 'source'}):
                with self.assertRaisesRegex(ValueError, 'source checksum'):
                    load_full_action_checkpoint(path, IDENTITIES)
            # Corrupted nonfinite state remains invalid even if hashes are recalculated.
            payload = copy.deepcopy(original)
            payload['actor_state_dict']['weight'][0, 0] = float('nan')
            payload['actor_state_hashes']['weight'] = _tensor_hash(payload['actor_state_dict']['weight'])
            header = {key: value for key, value in payload.items() if key not in ('payload_sha256', 'actor_state_dict')}
            payload['payload_sha256'] = _digest(header)
            bad = output/'nonfinite.pt'
            torch.save(payload, bad)
            with self.assertRaisesRegex(ValueError, 'nonfinite actor'):
                load_full_action_checkpoint(bad, IDENTITIES)
            for dimension in (True, 1000001, 1000000):
                payload = copy.deepcopy(original)
                payload['feature_dim'] = dimension
                header = {key: value for key, value in payload.items() if key not in ('payload_sha256', 'actor_state_dict')}
                payload['payload_sha256'] = _digest(header)
                bad = output/'bad-dimension.pt'
                torch.save(payload, bad)
                # Metadata must be checked before potentially huge allocation.
                with patch('mixtral_lab.full_action_fit.ResidualPairRouter', side_effect=AssertionError('allocated')):
                    with self.assertRaises(ValueError):
                        load_full_action_checkpoint(bad, IDENTITIES)
            for index, change in enumerate((
                    lambda value: value['adaptive_audit_state'].update(last_update=999),
                    lambda value: value['adaptive_audit_state']['config'].update(enabled=True),
                    lambda value: value.update(learning_rate_next=.5),
                    lambda value: value.update(learning_rate_used=.01))):
                payload = copy.deepcopy(original)
                change(payload)
                header = {key: value for key, value in payload.items() if key not in ('payload_sha256', 'actor_state_dict')}
                payload['payload_sha256'] = _digest(header)
                bad = output/f'audit-{index}.pt'
                torch.save(payload, bad)
                with self.subTest(audit=index), self.assertRaises(ValueError):
                    load_full_action_checkpoint(bad, IDENTITIES)
            with self.assertRaises(FileExistsError):
                fit(output)

    def test_invalid_config_provenance_frozen_model_and_features(self):
        for kwargs in ({'updates': -1}, {'updates': True}, {'epsilon': 1.}, {'epsilon': float('nan')},
                       {'max_score_calls': True}, {'batch_contexts': 0}, {'probe_contexts': 0},
                       {'fit_gain_threshold': 0.}, {'fit_gain_threshold': float('inf')},
                       {'anchor_atol': -1.}, {'learning_rate': 0.}, {'seed': 2**63},
                       {'updates': 10001}, {'max_train_contexts': 4097}, {'probe_contexts': 4097},
                       {'max_score_calls': 122881}, {'learning_rate': 1.01}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FullActionFitConfig(**kwargs).validate()
        with tempfile.TemporaryDirectory() as temp:
            h, y, provenance = inputs()
            for index, change in enumerate((lambda p: p[0].update(role='selection'),
                                           lambda p: p[0].update(role='held_out'),
                                           lambda p: p[0].update(target=999),
                                           lambda p: p.__setitem__(0, copy.deepcopy(p[1])),
                                           lambda p: p[0].update(document_id=''))):
                rows = copy.deepcopy(provenance)
                change(rows)
                with self.assertRaises(ValueError):
                    fit(Path(temp)/f'provenance-{index}', provenance=rows)
            for index, mutate in enumerate((lambda a: a.model.train(),
                                           lambda a: a.model.requires_grad_(True),
                                           lambda a: setattr(a.model.weight, 'grad', torch.ones_like(a.model.weight)))):
                adapter = Landscape()
                mutate(adapter)
                with self.assertRaisesRegex(ValueError, 'frozen|gradients'):
                    fit(Path(temp)/f'model-{index}', adapter=adapter)
                self.assertFalse(adapter.calls)
            adapter = Landscape()
            with patch.object(adapter, 'features', return_value=torch.full((1, 3), float('nan'))):
                with self.assertRaisesRegex(ValueError, 'features'):
                    fit(Path(temp)/'features', adapter=adapter)
            with self.assertRaisesRegex(ValueError, 'identity'):
                fit(Path(temp)/'identity', identities={'model': 'x'})
            with self.assertRaisesRegex(ValueError, 'context count'):
                fit(Path(temp)/'contexts', config=FullActionFitConfig(max_train_contexts=7, batch_contexts=7))

    def test_weaker_native_prior_is_explicit_and_initially_native(self):
        features = torch.ones(1, 3)
        native = torch.tensor([[1, 0]])
        for epsilon, odds in ((.2, 113.), (.8, 8.)):
            actor = ResidualPairRouter(3, epsilon)
            probabilities = actor.log_probs(features, native).exp()[0]
            self.assertAlmostEqual(float((probabilities[0]/probabilities[1]).detach()), odds, places=4)
            self.assertEqual(int(actor.greedy(features, native)), 1)

    def test_adaptive_disabled_preserves_fixed_weights_and_diagnostics(self):
        config = FullActionFitConfig(updates=4, learning_rate=.08)
        with tempfile.TemporaryDirectory() as temp:
            actor, fixed = fit(Path(temp)/'fixed', config=config)
            disabled_actor, disabled = fit(Path(temp)/'disabled', config=replace(config,
                adaptive={'enabled': False}))
            for key, value in actor.state_dict().items():
                torch.testing.assert_close(value, disabled_actor.state_dict()[key], atol=0, rtol=0)
            self.assertEqual(fixed['best_train_metrics'], disabled['best_train_metrics'])
            self.assertEqual(fixed['stop_reason'], 'max_updates')
            self.assertEqual(disabled['successful_updates'], 4)
            first = json.loads((Path(temp)/'fixed/metrics.json').read_text())
            second = json.loads((Path(temp)/'disabled/metrics.json').read_text())
            self.assertEqual(first, second)
            self.assertTrue(all(row['adaptive_decision'] is None for row in first))
            self.assertTrue(all(row['learning_rate_next'] == .08 for row in first))

    def test_adaptive_expected_ce_progress_is_not_flat_greedy_plateau(self):
        config = FullActionFitConfig(updates=4, learning_rate=1e-4, probe_contexts=2,
            adaptive={'enabled': True, 'eval_interval': 1, 'patience': 1,
                      'min_delta': 1e-12, 'min_updates': 1, 'stop_patience': 1,
                      'min_lr': 1e-5, 'max_lr': .001})
        observed = []
        original = AdaptiveFitController.observe
        def observe(controller, update, train_loss, **kwargs):
            observed.append((update, train_loss, kwargs['coverage_complete']))
            return original(controller, update, train_loss, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'progress'
            with patch.object(AdaptiveFitController, 'observe', observe):
                _, result = fit(output, config=config)
            records = json.loads((output/'metrics.json').read_text())
            self.assertEqual(result['stop_reason'], 'max_updates')
            self.assertEqual(result['successful_updates'], 4)
            self.assertFalse(result['train_fit_passed'])
            self.assertEqual(result['best_train_metrics']['greedy_change_count'], 0)
            self.assertLess(result['best_train_metrics']['expected_ce'], result['initial_train_metrics']['expected_ce'])
            self.assertEqual(len(observed), 5)
            for (update, loss, _), row in zip(observed, records):
                self.assertEqual(update, row['update'])
                self.assertEqual(loss, row['train']['expected_ce'])
                self.assertEqual(row['train']['contexts'], 8)  # not the 2-row diagnostic subset
                self.assertEqual(row['adaptive_decision']['action'], 'hold')
                self.assertEqual(row['learning_rate_next'], 1e-4)
            self.assertFalse(observed[0][2])
            self.assertTrue(all(row[2] for row in observed[1:]))

    def test_adaptive_plateau_reduces_lr_and_stops_only_after_coverage_and_warmup(self):
        config = FullActionFitConfig(updates=30, batch_contexts=1, learning_rate=.02,
            adaptive={'enabled': True, 'eval_interval': 2, 'patience': 1,
                      'min_delta': 1e-6, 'lr_factor': .5, 'min_updates': 12,
                      'stop_patience': 1, 'min_lr': .01, 'max_lr': .02})
        adapter = Landscape(flat=True)
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'plateau'
            _, result = fit(output, config=config, adapter=adapter)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['stop_reason'], 'adaptive_plateau')
            self.assertGreaterEqual(result['successful_updates'], 12)
            self.assertLess(result['successful_updates'], config.updates)
            self.assertEqual(result['training_coverage']['unique_train_contexts_visited'], 8)
            self.assertFalse(result['train_fit_passed'])
            self.assertEqual(result['score_work_totals']['actual_score_calls'], 240)
            records = json.loads((output/'metrics.json').read_text())
            events = [row['adaptive_decision'] for row in records if row['adaptive_decision'] is not None]
            self.assertIn('reduce_lr', [event['action'] for event in events])
            self.assertEqual(events[-1]['action'], 'stop')
            self.assertEqual(records[-1]['learning_rate_next'], .01)
            self.assertEqual([row['update'] for row in records if row['adaptive_decision'] is not None],
                             list(range(0, result['successful_updates']+1, 2)))
            for before, after in zip(records, records[1:]):
                self.assertEqual(after['learning_rate_used'], before['learning_rate_next'])
            _, metadata = load_full_action_checkpoint(output/'checkpoint_latest.pt', IDENTITIES)
            self.assertEqual(metadata['update'], result['successful_updates'])
            self.assertEqual(metadata['learning_rate_used'], records[-1]['learning_rate_used'])
            self.assertEqual(metadata['learning_rate_next'], records[-1]['learning_rate_next'])
            self.assertEqual(metadata['adaptive_audit_state'], result['adaptive'])
            payload = torch.load(output/'checkpoint_latest.pt', weights_only=True)
            payload['adaptive_audit_state']['tampered'] = True
            torch.save(payload, output/'tampered-controller.pt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_full_action_checkpoint(output/'tampered-controller.pt', IDENTITIES)
            self.assertTrue(all(parameter.grad is None for parameter in adapter.model.parameters()))

    def test_real_tiny_mixtral_singleton_and_reusable_inference(self):
        from transformers import MixtralConfig, MixtralForCausalLM
        # Initialization is isolated from caller RNG too; no pretrained download.
        with torch.random.fork_rng():
            torch.manual_seed(41)
            config = MixtralConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                num_local_experts=8, num_experts_per_tok=2, max_position_embeddings=32)
            config._attn_implementation = 'eager'
            config._experts_implementation = 'eager'
            model = MixtralForCausalLM(config).eval()
        adapter = MixtralAdapter(model, head_chunk=4)
        ids = torch.tensor([[2, 3, 4]])
        h = adapter.capture(ids)[:, :-1].reshape(-1, 16)
        y = ids[:, 1:].reshape(-1)
        provenance = [{'document_id': 'tiny', 'token_position': i, 'target': int(y[i])} for i in range(2)]
        original = copy.deepcopy(model.state_dict())
        native_logits = adapter.forward(ids).logits
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'tiny'
            actor, summary = fit(output, adapter=adapter, h=h, y=y, provenance=provenance,
                config=FullActionFitConfig(updates=3, batch_contexts=2, learning_rate=.05))
            self.assertEqual(summary['score_work_totals']['actual_score_calls'], 60)
            initial, _ = load_full_action_checkpoint(output/'checkpoint_initial.pt', IDENTITIES)
            torch.testing.assert_close(adapter.forward(ids, policy=initial).logits, native_logits, atol=0, rtol=0)
            logits = adapter.forward(ids, policy=actor).logits
            self.assertEqual(logits.shape, (1, 3, 32))
            self.assertTrue(torch.isfinite(logits).all())
            metrics = json.loads((output/'metrics.json').read_text())
            self.assertGreater(metrics[1]['gradient_norm'], 0.)
            self.assertFalse(summary['held_out_confirmation'])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(original[key], value))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
