"""Offline tiny-HF numerical bridge tests; no pretrained weights or GPU jobs."""
import copy
from contextlib import ExitStack
import subprocess
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments import mixtral_bridge as cli
from mixtral_lab.adapter import MixtralAdapter, pair_to_action
from mixtral_lab.adapter_fit import AdapterFitConfig, _tensor_hash
from mixtral_lab.bridge import (
    bridge_tolerances, diagnose_bridge, forced_cached_policy, prepare_frozen_train,
    safe_path, verify_historical_sources,
)
from mixtral_lab.fit_protocol import FREEZE_SCHEMA, REPORT_SCHEMA, experiment_contract, seal_report, validate_fit_config
from mixtral_lab.protocol import canonical_hash, prepare_data, sha256_file
from mixtral_lab.sampling import collect_stratified_contexts, plan_stratified_contexts
from mixtral_lab.second_round import SecondRoundResidual
from mixtral_lab.training import ResidualPairRouter
from tests.test_mixtral_adapter import tiny
from tests.test_mixtral_protocol import Tokenizer, config as base_config


def rows():
    # Include left padding, right padding, an internal mask gap, and remainder.
    return [
        {'id': 'train-a', 'input_ids': torch.tensor([0, 1, 2, 3, 4, 0]),
         'attention_mask': torch.tensor([0, 1, 1, 1, 1, 0])},
        {'id': 'train-b', 'input_ids': torch.tensor([5, 6, 7, 8, 9, 0]),
         'attention_mask': torch.tensor([1, 1, 0, 1, 1, 0])},
    ]


def settings(kind='router'):
    training = {} if kind == 'router' else asdict(AdapterFitConfig(rank=2, updates=1,
        batch_contexts=2, max_train_contexts=3, probe_chunk_contexts=2, seed=17))
    return {'fit': {'kind': kind, 'training': training}, 'max_train_contexts': 3, 'seed': 17,
            'nll_tolerance': 1e-6, 'ce_mean_abs_tolerance': 1e-6, 'logit_max_abs_tolerance': 1e-6}


class BridgeTests(unittest.TestCase):
    def test_router_fp32_bf16_masks_labels_and_zero_identity(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                adapter = MixtralAdapter(tiny(dtype))
                candidate = ResidualPairRouter(adapter.feature_dim, .8)
                before = {name: value.clone() for name, value in adapter.model.state_dict().items()}
                result = diagnose_bridge(adapter, rows(), candidate, settings())
                self.assertEqual(result['status'], 'complete')
                self.assertFalse(result['deployment_approved'])
                self.assertFalse(result['cross_shape_equivalence_asserted'])
                self.assertEqual(result['cached']['native']['nll'], result['cached']['zero']['nll'])
                self.assertEqual(result['online']['selected']['native']['nll'], result['online']['selected']['zero']['nll'])
                self.assertEqual(result['online']['selected']['native']['tokens'], 3)
                self.assertEqual(result['online']['unselected']['native']['tokens'], 2)
                self.assertEqual(result['action_agreement']['zero']['fraction'], 1.)
                mapping = {r['id']: r for r in rows()}
                for context in result['contexts']:
                    row = mapping[context['document_id']]; p = context['token_position']
                    self.assertEqual(context['target'], int(row['input_ids'][p+1]))
                    self.assertTrue(row['attention_mask'][p] and row['attention_mask'][p+1])
                for name, value in adapter.model.state_dict().items():
                    self.assertTrue(torch.equal(value, before[name]))
                self.assertTrue(all(p.grad is None for p in candidate.parameters()))
                json.dumps(result, allow_nan=False)

    def test_adapter_fixed_tail_padding_and_cache_hashes(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                adapter = MixtralAdapter(tiny(dtype))
                config = settings('adapter')
                candidate = SecondRoundResidual(adapter.hidden_size, AdapterFitConfig(**config['fit']['training']))
                sizes = []
                original = adapter.step_pairs
                def observe(h, pairs):
                    sizes.append(len(h))
                    return original(h, pairs)
                with patch.object(adapter, 'step_pairs', side_effect=observe):
                    result = diagnose_bridge(adapter, rows(), candidate, config)
                self.assertEqual(sizes[:2], [2, 2], 'tail cache call must retain original probe shape')
                self.assertEqual(result['work']['cached_native_chunks'], 2)
                self.assertEqual(result['score_protocol']['probe_chunk_contexts'], 2)
                self.assertEqual(result['cached']['native']['nll'], result['cached']['zero']['nll'])
                self.assertEqual(result['cached']['native']['nll'], result['cached']['learned']['nll'])
                # A frozen tensor identity cannot be waved away by wider tolerances.
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    diagnose_bridge(adapter, rows(), candidate, config,
                                    checkpoint_metadata={'cached_native_sha256': '0'*64})

    def test_recovery_adapter_scientific_settings_preserved(self):
        from mixtral_lab.recovery_adapter import RecoveryAdapterConfig
        config = settings('adapter')
        training = RecoveryAdapterConfig(rank=2, updates=1, min_updates=0,
            batch_contexts=2, max_train_contexts=3, probe_chunk_contexts=2,
            native_kl_beta=.2, relative_correction_l2=.01, development_patience=3)
        config['fit']['training'] = asdict(training)
        adapter = MixtralAdapter(tiny())
        candidate = SecondRoundResidual(adapter.hidden_size, training)
        result = diagnose_bridge(adapter, rows(), candidate, config)
        self.assertEqual(result['candidate_training_config'], asdict(training))
        self.assertEqual(result['score_protocol']['probe_chunk_contexts'], 2)
        bad = copy.deepcopy(config); bad['fit']['training']['native_kl_beta'] = .3
        with self.assertRaisesRegex(ValueError, 'scientific settings'):
            diagnose_bridge(adapter, rows(), candidate, bad)

    def test_learned_router_and_forced_action_drift_are_visible(self):
        adapter = MixtralAdapter(tiny())
        candidate = ResidualPairRouter(adapter.feature_dim, .8)
        # Shape-dependent actor fake: cached probes have three rows, documents six.
        def policy(features, native):
            return torch.full((len(features),), 1 if len(features) == 3 else 28, dtype=torch.long)
        with patch.object(candidate, 'greedy', side_effect=policy):
            result = diagnose_bridge(adapter, rows(), candidate, settings())
        self.assertEqual(result['action_agreement']['learned']['fraction'], 0.)
        self.assertIn('forced_cached', result['online']['selected'])
        self.assertIn('cached_vs_online.forced_cached.ce', result['absolute_differences'])
        for context in result['contexts']:
            self.assertEqual(context['cached_learned_action'], 1)
            self.assertEqual(context['online_learned_action'], 28)
            self.assertIn('online_forced_cached_ce', context)
        self.assertTrue(any(abs(r['online_learned_ce'] - r['online_forced_cached_ce']) > 1e-7
                            for r in result['contexts']))

    def test_forcing_changes_only_requested_positions(self):
        adapter = MixtralAdapter(tiny())
        obs = adapter.observe(torch.randn(6, 16))
        original = lambda obs: pair_to_action(obs.native_pairs)
        policy = forced_cached_policy(original, torch.tensor([1, 4]), torch.tensor([1, 28]))
        actual = policy(obs); expected = original(obs)
        expected[1] = 1; expected[4] = 28
        self.assertTrue(torch.equal(actual, expected))
        for invalid in (torch.tensor([6]), torch.tensor([-1]), torch.tensor([1, 1]), torch.tensor([1.])):
            with self.assertRaises(ValueError):
                forced_cached_policy(original, invalid, torch.ones(len(invalid), dtype=torch.long))(obs)

    def test_shape_dependent_fake_scores_fail_declared_screen(self):
        adapter = MixtralAdapter(tiny())
        candidate = ResidualPairRouter(adapter.feature_dim, .8)
        original = adapter.score_pairs
        def score(h, y, pairs):
            return original(h, y, pairs) + (0.1 if len(h) == 1 else 0.)
        with patch.object(adapter, 'score_pairs', side_effect=score):
            result = diagnose_bridge(adapter, rows(), candidate, settings())
        self.assertFalse(result['numerical_screen_passed'])
        self.assertFalse(result['zero_screen_passed'])
        self.assertGreater(result['absolute_differences']['cached_readout_vs_score_pairs.ce']['max_abs'], .09)
        self.assertEqual(result['tolerances']['ce_max_abs'], 1e-6)
        self.assertEqual(result['work']['cached_readout_calls'], 24)
        self.assertEqual(result['work']['cached_terminal_ffn_calls'], 24)
        self.assertEqual(result['work']['cached_score_pairs_calls'], 12)

    def test_nonfinite_rejected_and_hooks_cleaned(self):
        adapter = MixtralAdapter(tiny())
        candidate = ResidualPairRouter(adapter.feature_dim, .8)
        with torch.no_grad():
            candidate.weight[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            diagnose_bridge(adapter, rows(), candidate, settings())
        self.assertEqual(len(adapter.model.model.norm._forward_pre_hooks), 0)

    def test_manifest_target_position_and_nontrain_rejected(self):
        adapter = MixtralAdapter(tiny())
        candidate = ResidualPairRouter(adapter.feature_dim, .8)
        manifest = plan_stratified_contexts(rows(), 3, 17, 'train')
        manifest['contexts'][0]['target'] += 1
        with self.assertRaisesRegex(ValueError, 'manifest'):
            diagnose_bridge(adapter, rows(), candidate, settings(), manifest=manifest)
        data = rows(); data[0]['role'] = 'selection'
        with self.assertRaisesRegex(ValueError, 'train rows only'):
            diagnose_bridge(adapter, data, candidate, settings())

    def test_all_selected_empty_unselected_is_not_nan(self):
        adapter = MixtralAdapter(tiny())
        config = settings(); config['max_train_contexts'] = 5
        result = diagnose_bridge(adapter, rows(), ResidualPairRouter(adapter.feature_dim, .8), config)
        self.assertIsNone(result['online']['unselected']['native']['nll'])
        json.dumps(result, allow_nan=False)

    def test_tolerances_are_explicit_not_auto_widened(self):
        config = settings()
        self.assertEqual(bridge_tolerances(config)['hidden_max_abs'], 1e-6)
        self.assertEqual(bridge_tolerances(config, {'hidden_max_abs': 0.})['hidden_max_abs'], 0.)
        for override in ({'unknown': 1.}, {'ce_max_abs': float('inf')}, {'ce_max_abs': -1.}, {'ce_max_abs': True}):
            with self.assertRaises(ValueError):
                bridge_tolerances(config, override)


class HistoricalBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_sources_ignore_new_modules_and_commit_but_reject_old_tamper(self):
        (self.root/'original.py').write_text('ORIGINAL = True\n')
        identity = {'commit': 'historical commit', 'sources': {'original.py': sha256_file(self.root/'original.py')}}
        (self.root/'new_diagnostic.py').write_text('NEW = True\n')
        self.assertEqual(verify_historical_sources(identity, self.root), identity)
        (self.root/'original.py').write_text('ORIGINAL = False\n')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            verify_historical_sources(identity, self.root)

    def test_source_and_report_path_traversal_and_symlink_escape(self):
        for name in ('../outside.py', '/tmp/outside.py', 'a/../../outside.py', r'..\outside.py'):
            with self.assertRaises(ValueError):
                verify_historical_sources({'sources': {name: '0'*64}}, self.root)
        outside = self.root.parent/(self.root.name + '-outside.py')
        outside.write_text('outside')
        self.addCleanup(outside.unlink)
        (self.root/'escape.py').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            verify_historical_sources({'sources': {'escape.py': sha256_file(outside)}}, self.root)
        with self.assertRaisesRegex(ValueError, 'traversal'):
            safe_path('report/../report.json', 'report')

    def test_only_train_tokenized_all_roles_hash_checked(self):
        config = base_config(self.root)
        _, identity = prepare_data(config, Tokenizer())
        class TrainOnlyTokenizer(Tokenizer):
            def encode(self, text, **kw):
                if 'document 0 ' not in text:
                    raise AssertionError('selection/calibration tokenization forbidden')
                return super().encode(text, **kw)
        prepared, current = prepare_frozen_train(config, TrainOnlyTokenizer(), identity)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(current, identity)
        Path(config['data_files']['selection']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'selection data identity'):
            prepare_frozen_train(config, TrainOnlyTokenizer(), identity)

    def _historical_fit(self, kind):
        # Exercise the genuine old artifact writers/loaders, with only the heavy
        # GPU/model-input boundary replaced by a random tiny CPU HF model.
        from experiments import mixtral_fit_experiment as fit
        config = base_config(self.root)
        common = {'updates': 1, 'batch_contexts': 2, 'max_train_contexts': 4,
                  'fit_gain_threshold': 100., 'seed': 17}
        training = ({**common, 'max_score_calls': 120, 'probe_contexts': 4}
                    if kind == 'router' else {**common, 'rank': 2, 'probe_chunk_contexts': 3})
        config['fit'] = {'kind': kind, 'training': training, 'generalization_min_gain': .001}
        config['output_dir'] = str(self.root/'old-fit')
        path = self.root/'old-config.json'; path.write_text(json.dumps(config))
        model = tiny(); adapter = MixtralAdapter(model); tokenizer = Tokenizer()
        info = {'model': {'content_sha256': 'a'*64}, 'tokenizer': {'content_sha256': 'b'*64},
                'cuda_devices': []}
        inputs = {'metadata': model.config, 'tokenizer': tokenizer,
                  'identity': {'model': info['model'], 'tokenizer': info['tokenizer']}}
        root = Path(cli.__file__).resolve().parents[1]
        tracked = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-r', '--name-only', 'HEAD', 'mixtral_lab']).decode().splitlines()
        names = [name for name in tracked if name.endswith('.py')]
        names += ['experiments/mixtral_experiment.py', 'experiments/mixtral_fit_experiment.py']
        historical = {'commit': 'original-artifact-commit-not-todays-commit',
                      'sources': {name: sha256_file(root/name) for name in names}}
        stack = self.enterContext(ExitStack())
        for module in (fit, cli):
            stack.enter_context(patch.object(module, 'load_local_inputs', return_value=inputs))
            stack.enter_context(patch.object(module, 'load_local_model', return_value=(model, tokenizer, adapter, info)))
            stack.enter_context(patch.object(module, 'normalized_artifact_stats', return_value={'immutable': 'tiny-test-only'}))
        stack.enter_context(patch.object(fit, 'fit_code_identity', return_value=historical))
        stack.enter_context(patch.object(fit, 'validate_native', return_value={'eligible': True}))
        stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
        stack.enter_context(patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False))
        report = fit.main(['--config', str(path), '--stage', 'fit'])
        self.assertFalse(report['fit_summary']['train_fit_passed'])
        return path, self.root/'old-fit'/'report.json', self.root/'old-fit'/'fit'/'checkpoint_best.pt'

    def test_historical_failed_fit_gate_router_and_adapter_run_without_selection(self):
        for kind in ('router', 'adapter'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                previous = self.root; self.root = Path(directory)
                path, report_path, checkpoint = self._historical_fit(kind)
                with patch('experiments.mixtral_experiment.evaluate', side_effect=AssertionError('no selection evaluation')), \
                     patch('experiments.mixtral_fit_experiment.evaluate_generalization', side_effect=AssertionError('no generalization')):
                    result = cli.main(['--config', str(path), '--fit-report', str(report_path),
                        '--checkpoint', str(checkpoint), '--output-dir', str(self.root/'bridge')])
                self.assertEqual(result['status'], 'complete')
                self.assertFalse(result['fit_gate_required'])
                self.assertFalse(result['deployment_approved'])
                self.assertNotEqual(result['identity']['code']['commit'], result['diagnostic_source']['commit'])
                self.assertEqual(result['preflight']['data_access']['selection'], 'original bound file hash only')
                self.assertEqual(result['bridge']['chosen_update'], result['checkpoint']['update'])
                self.assertEqual(result['identity'], json.loads(report_path.read_text())['identity'])
                self.root = previous

    def test_historical_checkpoint_tamper_rejected_before_model_load(self):
        path, report_path, checkpoint = self._historical_fit('router')
        checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
        with patch.object(cli, 'load_local_model') as heavy:
            with self.assertRaisesRegex(ValueError, 'checkpoint bytes changed'):
                cli.main(['--config', str(path), '--fit-report', str(report_path),
                    '--checkpoint', str(checkpoint), '--output-dir', str(self.root/'bridge')])
        heavy.assert_not_called()
        failed = json.loads((self.root/'bridge'/'report.json').read_text())
        self.assertFalse(failed['model_load_started'])
        self.assertEqual(failed['status'], 'failed')

    def test_cli_rejects_report_tamper_without_loading_and_records_failure(self):
        config = base_config(self.root)
        config['fit'] = {'kind': 'router', 'training': {'updates': 1, 'batch_contexts': 2,
            'max_train_contexts': 4, 'max_score_calls': 120, 'probe_contexts': 4, 'seed': 17},
            'generalization_min_gain': .001}
        config_path = self.root/'config.json'; config_path.write_text(json.dumps(config))
        report = seal_report({'schema': REPORT_SCHEMA, 'stage': 'fit', 'status': 'complete',
            'model_load_started': True, 'native_equivalence': {'eligible': True}})
        report['unexpected'] = 'tampered after seal'
        report_path = self.root/'report.json'; report_path.write_text(json.dumps(report))
        with patch.object(cli, 'load_local_model') as heavy, patch.object(cli, 'load_local_inputs') as light:
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                cli.main(['--config', str(config_path), '--fit-report', str(report_path),
                          '--checkpoint', str(self.root/'checkpoint.pt'), '--output-dir', str(self.root/'diagnosis')])
        heavy.assert_not_called(); light.assert_not_called()
        failed = json.loads((self.root/'diagnosis'/'report.json').read_text())
        self.assertEqual(failed['status'], 'failed'); self.assertFalse(failed['model_load_started'])
        self.assertEqual(failed['failure_phase'], 'preflight')
        with self.assertRaises(FileExistsError):
            cli.main(['--config', str(config_path), '--fit-report', str(report_path),
                      '--checkpoint', str(self.root/'checkpoint.pt'), '--output-dir', str(self.root/'diagnosis')])


if __name__ == '__main__':
    unittest.main()
