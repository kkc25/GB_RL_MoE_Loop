"""CPU-only staging contracts: no model download or real GPU allocation."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from experiments import mixtral_fit_experiment as runner
from mixtral_lab import fit_protocol as protocol
from mixtral_lab.protocol import canonical_hash, sha256_file
from mixtral_lab.full_action_fit import FullActionFitConfig
from tests.test_mixtral_protocol import config as base_config, Tokenizer
from tests.test_mixtral_adapter import tiny
from mixtral_lab.adapter import MixtralAdapter


class FitProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = base_config(self.root)
        self.config['fit'] = {'kind': 'router', 'training': {
            'updates': 2, 'batch_contexts': 2, 'max_train_contexts': 4,
            'max_score_calls': 120, 'probe_contexts': 4, 'seed': 17},
            'generalization_min_gain': .001}

    def install_runtime(self, *, fit_pass=True, candidate_nll=1.8, actual_training=False):
        """Real data preflight + fake heavy boundary, not a fake passing GPU test."""
        stack = self.enterContext(ExitStack())
        model = tiny()
        adapter = MixtralAdapter(model)
        info = {'model': {'content_sha256': 'a'*64}, 'tokenizer': {'content_sha256': 'b'*64},
                'cuda_devices': []}
        inputs = {'metadata': model.config, 'tokenizer': Tokenizer(),
                  'identity': {'model': {'files': {'config.json': 'a'}}, 'tokenizer': {'files': {'tokenizer.json': 'b'}}}}
        stack.enter_context(patch.object(runner, 'load_local_inputs', return_value=inputs))
        self.heavy = stack.enter_context(patch.object(runner, 'load_local_model', return_value=(model, inputs['tokenizer'], adapter, info)))
        stack.enter_context(patch.object(runner, 'normalized_artifact_stats', return_value={'model': {'files': ['immutable']}}))
        stack.enter_context(patch.object(runner, 'fit_code_identity', return_value={'commit': 'fixed', 'sources': {'new_runner.py': 'f'*64}}))
        stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
        stack.enter_context(patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False))
        self.parity = stack.enter_context(patch.object(runner, 'validate_native', return_value={'eligible': True}))
        def capture(a, rows, count, seed, role):
            self.assertEqual(role, 'train')
            return torch.ones(count, 16), torch.ones(count, dtype=torch.long), runner.plan_stratified_contexts(rows, count, seed, role)
        if not actual_training:
            self.capture = stack.enter_context(patch.object(runner, 'collect_stratified_contexts', side_effect=capture))
        self.training_identities = []
        self.settings = None
        def train(a, h0, y, config, identities, output, *, provenance):
            self.assertEqual(len(provenance), 4)
            self.training_identities.append(identities)
            self.settings = dict(config.__dict__)
            output.mkdir()
            (output/'checkpoint_best.pt').write_bytes(b'frozen best train checkpoint')
            return object(), {'status': 'complete', 'train_fit_passed': fit_pass, 'best_update': 1 if fit_pass else 0,
                'best_train_metrics': {'greedy_gain_nats': .1, 'greedy_change_count': 1}}
        def load(path, identities, device='cpu'):
            self.assertEqual(identities, self.training_identities[0])
            return object(), {'checkpoint_sha256': sha256_file(path), 'update': 1 if fit_pass else 0,
                              'training_config': self.settings, 'train_fit_passed': fit_pass,
                              'train_metrics': {'greedy_gain_nats': .1, 'greedy_change_count': 1}}
        if not actual_training:
            self.trainer = stack.enter_context(patch.object(runner, 'trainer_api', return_value=(FullActionFitConfig, train, load)))
        def evaluate(adapter, rows, policy=None, *, native=False):
            self.assertTrue(all(row['id'] == 'selection' for row in rows))
            nll = 2. if native or policy is None else candidate_nll
            return {'nll': nll, 'tokens': 4, 'sequences': 1}
        self.evaluator = stack.enter_context(patch.object(runner, 'evaluate', side_effect=evaluate))
        self.routes = stack.enter_context(patch.object(runner, 'route_diagnostics', return_value={'greedy_native_agreement': .5}))
        self.timing = stack.enter_context(patch.object(runner, 'measure_calls', return_value={'variants': {'native': {'wall_ms': [1.]}}}))

    def run_stage(self, stage, *, fit=None, generalization=None, name=None):
        c = copy.deepcopy(self.config)
        c['output_dir'] = str(self.root/(name or stage))
        path = self.root/((name or stage)+'.json')
        path.write_text(json.dumps(c))
        argv = ['--config', str(path), '--stage', stage]
        if fit:
            argv += ['--fit-report', str(fit/'report.json'), '--checkpoint', str(fit/'fit'/'checkpoint_best.pt')]
        if generalization:
            argv += ['--generalization-report', str(generalization/'report.json')]
        return runner.main(argv)

    def rewrite_report(self, directory, edit, reseal=True):
        path = directory/'report.json'
        report = json.loads(path.read_text())
        edit(report)
        if reseal:
            report = protocol.seal_report(report)
        path.write_text(json.dumps(report))

    def test_order_fit_generalize_profile_and_no_selection_during_fit(self):
        self.install_runtime()
        fit = self.run_stage('fit')
        self.evaluator.assert_not_called()
        self.routes.assert_not_called()
        self.timing.assert_not_called()
        self.assertTrue(fit['fit_summary']['train_fit_passed'])
        self.assertEqual(fit['identity']['context_sampling']['role'], 'train')
        generalize = self.run_stage('generalize', fit=self.root/'fit')
        self.assertEqual(self.evaluator.call_count, 3)
        self.assertTrue(generalize['generalization']['quality_gate']['passed'])
        self.assertEqual(len(generalize['generalization']['per_document']), 1)
        before = self.evaluator.call_count
        profile = self.run_stage('profile', fit=self.root/'fit', generalization=self.root/'generalize')
        self.assertEqual(self.evaluator.call_count, before, 'profile must not redo selection/model selection')
        self.timing.assert_called_once()
        self.assertEqual(profile['status'], 'complete')

    def test_actual_router_fit_checkpoint_and_downstream_integration(self):
        self.config['fit']['training'].update(updates=24, learning_rate=.2, epsilon=.99, fit_gain_threshold=1e-7,
            adaptive={'enabled': True, 'eval_interval': 2, 'patience': 1, 'min_delta': 100.,
                'min_lr': .05, 'max_lr': .2, 'min_updates': 8, 'stop_patience': 2})
        self.install_runtime(actual_training=True)
        fit = self.run_stage('fit')
        self.evaluator.assert_not_called()
        self.assertTrue(fit['fit_summary']['train_fit_passed'])
        self.assertIn('best_train_metrics', fit['fit_summary'])
        self.assertGreater(fit['fit_summary']['adaptive']['reductions'], 0)
        self.assertLess(fit['fit_summary']['successful_updates'], self.config['fit']['training']['updates'])
        self.assertEqual(fit['checkpoint']['update'], fit['fit_summary']['best_update'])
        self.run_stage('generalize', fit=self.root/'fit')
        self.run_stage('profile', fit=self.root/'fit', generalization=self.root/'generalize')
        self.timing.assert_called_once()

    def test_actual_adapter_fit_checkpoint_and_downstream_integration(self):
        self.config['fit'].update(kind='adapter', training={'rank': 2, 'updates': 24,
            'batch_contexts': 2, 'max_train_contexts': 4, 'probe_chunk_contexts': 2,
            'eval_interval': 2, 'learning_rate': .1, 'residual_scale': 1.,
            'fit_gain_threshold': 1e-7, 'seed': 17,
            'adaptive': {'enabled': True, 'eval_interval': 2, 'patience': 1, 'min_delta': 100.,
                'min_lr': .05, 'max_lr': .1, 'min_updates': 8, 'stop_patience': 2}})
        self.install_runtime(actual_training=True)
        fit = self.run_stage('fit')
        self.evaluator.assert_not_called()
        self.assertTrue(fit['fit_summary']['train_fit_passed'])
        self.assertTrue(fit['fit_summary']['coverage_complete'])
        self.assertGreater(fit['fit_summary']['adaptive_controller_state']['reductions'], 0)
        self.assertTrue(fit['fit_summary']['adaptive_stopped'])
        # Development metrics are mocked only to exercise a positive profile gate;
        # fit above executes the real tiny HF prefix, frozen readout and trainer.
        with patch.object(runner, 'evaluate_second_round', return_value={'nll': 1.8, 'tokens': 4, 'sequences': 1}):
            self.run_stage('generalize', fit=self.root/'fit')
        self.run_stage('profile', fit=self.root/'fit', generalization=self.root/'generalize')
        self.timing.assert_called_once()

    def test_failed_train_fit_blocks_before_model_load(self):
        self.install_runtime(fit_pass=False)
        self.run_stage('fit')
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'train-fit gate'):
            self.run_stage('generalize', fit=self.root/'fit')
        self.heavy.assert_not_called()
        report = json.loads((self.root/'generalize'/'report.json').read_text())
        self.assertFalse(report['model_load_started'])
        self.assertEqual(report['failure_phase'], 'preflight')

    def test_zero_best_update_even_with_pass_flag_is_blocked(self):
        self.install_runtime()
        self.run_stage('fit')
        self.rewrite_report(self.root/'fit', lambda r: r['fit_summary'].update(best_update=0))
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'best train-fit update'):
            self.run_stage('generalize', fit=self.root/'fit')
        self.heavy.assert_not_called()

    def test_negative_or_subthreshold_development_gain_blocks_timing(self):
        self.install_runtime(candidate_nll=1.9995)
        self.run_stage('fit')
        generalization = self.run_stage('generalize', fit=self.root/'fit')
        self.assertFalse(generalization['generalization']['quality_gate']['passed'])
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'quality gate'):
            self.run_stage('profile', fit=self.root/'fit', generalization=self.root/'generalize')
        self.heavy.assert_not_called()
        self.timing.assert_not_called()

    def test_missing_stage_inputs_records_failure_without_loading(self):
        self.install_runtime()
        with self.assertRaisesRegex(ValueError, 'require --fit-report'):
            self.run_stage('generalize')
        self.heavy.assert_not_called()
        self.assertEqual(json.loads((self.root/'generalize'/'report.json').read_text())['status'], 'failed')

    def test_modified_checkpoint_rejected_before_loading(self):
        self.install_runtime()
        self.run_stage('fit')
        (self.root/'fit'/'fit'/'checkpoint_best.pt').write_bytes(b'replaced')
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'checkpoint bytes changed'):
            self.run_stage('generalize', fit=self.root/'fit')
        self.heavy.assert_not_called()

    def test_report_tamper_missing_digest_and_legacy_schema_rejected(self):
        self.install_runtime()
        self.run_stage('fit')
        self.rewrite_report(self.root/'fit', lambda r: r['fit_summary'].update(best_update=2), reseal=False)
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            self.run_stage('generalize', fit=self.root/'fit')
        self.heavy.assert_not_called()
        path = self.root/'fit'/'report.json'
        report = json.loads(path.read_text())
        report['schema'] = 'mixtral-experiment-report-v1'
        path.write_text(json.dumps(protocol.seal_report(report)))
        with self.assertRaisesRegex(ValueError, 'requires complete fit report'):
            protocol.read_report(path, 'fit')

    def test_changed_sources_artifacts_runtime_and_data_fail_preflight(self):
        self.install_runtime()
        self.run_stage('fit')
        for field in ('code', 'artifact_stats', 'runtime_versions', 'data'):
            with self.subTest(field=field):
                original = (self.root/'fit'/'report.json').read_text()
                def mutate(r):
                    r['freeze_manifest'][field] = {'changed': True}
                    r['freeze_manifest_sha256'] = canonical_hash(r['freeze_manifest'])
                self.rewrite_report(self.root/'fit', mutate)
                self.heavy.reset_mock()
                with self.assertRaisesRegex(ValueError, f'frozen {field} mismatch'):
                    self.run_stage('generalize', fit=self.root/'fit', name='changed_'+field)
                self.heavy.assert_not_called()
                (self.root/'fit'/'report.json').write_text(original)

    def test_mismatched_generalization_fit_reference_blocks_before_load(self):
        self.install_runtime()
        self.run_stage('fit')
        self.run_stage('generalize', fit=self.root/'fit')
        self.rewrite_report(self.root/'generalize', lambda r: r['evidence'].update(fit_report_sha256='0'*64))
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'different fit report'):
            self.run_stage('profile', fit=self.root/'fit', generalization=self.root/'generalize')
        self.heavy.assert_not_called()
        self.timing.assert_not_called()

    def test_resealed_but_inconsistent_per_document_metrics_block_profile(self):
        self.install_runtime()
        self.run_stage('fit')
        self.run_stage('generalize', fit=self.root/'fit')
        self.rewrite_report(self.root/'generalize', lambda r: r['generalization']['per_document'][0].update(candidate_nll=1.0))
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'paired delta mismatch'):
            self.run_stage('profile', fit=self.root/'fit', generalization=self.root/'generalize')
        self.heavy.assert_not_called()
        self.timing.assert_not_called()

    def test_fit_summary_must_match_loaded_checkpoint_metrics(self):
        self.install_runtime()
        self.run_stage('fit')
        self.rewrite_report(self.root/'fit', lambda r: r['fit_summary']['best_train_metrics'].update(greedy_gain_nats=.2))
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'metrics/verdict mismatch'):
            self.run_stage('generalize', fit=self.root/'fit')
        self.heavy.assert_not_called()

    def test_checkpoint_mutation_during_execution_leaves_failed_report(self):
        self.install_runtime()
        self.run_stage('fit')
        def changed_checkpoint(*args, **kwargs):
            (self.root/'fit'/'fit'/'checkpoint_best.pt').write_bytes(b'changed during evaluation')
            return {'greedy_native_agreement': .5}
        self.routes.side_effect = changed_checkpoint
        with self.assertRaisesRegex(ValueError, 'checkpoint bytes changed during'):
            self.run_stage('generalize', fit=self.root/'fit')
        report = json.loads((self.root/'generalize'/'report.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertNotIn('report_sha256', report)

    def test_large_adaptive_report_is_readable_but_size_remains_bounded(self):
        # A max-budget fixed-probe history can legitimately exceed the old 16 MiB
        # limit. Preserve the exact seal/read contract without a real GPU run.
        self.install_runtime()
        report = self.run_stage('fit')
        report['fit_summary']['train_probe_history'] = [
            {'update': i, 'nll': 2. - i / 10000., 'diagnostics': {'finite_audit_note': 'x' * 1800}}
            for i in range(10001)]
        report = protocol.seal_report(report)
        path = self.root/'large-report.json'
        path.write_text(json.dumps(report))
        self.assertGreater(path.stat().st_size, 16 * 1024 * 1024)
        self.assertEqual(protocol.read_report(path, 'fit')['report_sha256'], report['report_sha256'])
        # Sparse oversized file: exercise refusal without allocating a 64 MiB JSON.
        with path.open('wb') as stream:
            stream.seek(protocol.MAX_REPORT_BYTES)
            stream.write(b'!')
        with self.assertRaisesRegex(ValueError, '64 MiB bound'):
            protocol.read_report(path, 'fit')

    def test_scientific_config_is_fixed_except_fresh_output(self):
        self.install_runtime()
        self.run_stage('fit')
        self.config['fit']['generalization_min_gain'] = .0001
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'scientific configuration mismatch'):
            self.run_stage('generalize', fit=self.root/'fit')
        self.heavy.assert_not_called()

    def test_preflight_actual_context_quota_before_heavy_load(self):
        self.install_runtime()
        self.config['max_train_contexts'] = self.config['training']['max_train_contexts'] = 5
        self.config['fit']['training'].update(max_train_contexts=5, max_score_calls=150)
        with self.assertRaisesRegex(ValueError, 'only 4 valid next-token positions'):
            self.run_stage('fit')
        self.heavy.assert_not_called()
        self.evaluator.assert_not_called()

    def test_default_off_adaptive_policies_have_one_canonical_contract(self):
        contracts = []
        for policy in ('missing', None, {}, {'enabled': False}):
            c = copy.deepcopy(self.config)
            if policy != 'missing':
                c['fit']['training']['adaptive'] = policy
            protocol.validate_fit_config(c)
            contracts.append(protocol.experiment_contract(c))
        self.assertTrue(all(contract == contracts[0] for contract in contracts))
        self.assertFalse(contracts[0]['fit']['training']['adaptive']['enabled'])
        self.assertIn('min_updates', contracts[0]['fit']['training']['adaptive'])

    def test_adaptive_policy_change_after_fit_is_rejected_before_load(self):
        self.install_runtime()
        self.run_stage('fit')
        self.config['fit']['training']['adaptive'] = {'enabled': True}
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'scientific configuration mismatch'):
            self.run_stage('generalize', fit=self.root/'fit')
        self.heavy.assert_not_called()

    def test_invalid_budget_and_nonfinite_config_leave_early_failure_report(self):
        self.install_runtime()
        self.config['fit']['training']['max_score_calls'] = 119
        with self.assertRaisesRegex(ValueError, '30 singleton'):
            self.run_stage('fit')
        self.heavy.assert_not_called()
        self.config['fit']['training']['max_score_calls'] = 120
        self.config['fit']['generalization_min_gain'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.run_stage('fit', name='nan_config')
        self.assertEqual(json.loads((self.root/'nan_config'/'report.json').read_text())['status'], 'failed')

    def test_nonfinite_development_evaluation_cannot_produce_quality_evidence(self):
        self.install_runtime(candidate_nll=float('nan'))
        self.run_stage('fit')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.run_stage('generalize', fit=self.root/'fit')
        report = json.loads((self.root/'generalize'/'report.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertNotIn('generalization', report)
        self.timing.assert_not_called()

    def test_all_roles_checked_for_leakage_even_during_fit(self):
        self.install_runtime()
        Path(self.config['data_files']['selection']).write_text(Path(self.config['data_files']['train']).read_text())
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.run_stage('fit')
        self.heavy.assert_not_called()
        self.evaluator.assert_not_called()

    def test_fresh_output_protects_failure_evidence(self):
        self.install_runtime()
        self.run_stage('fit')
        before = (self.root/'fit'/'report.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_stage('fit')
        self.assertEqual(before, (self.root/'fit'/'report.json').read_bytes())

    def test_adapter_profile_closure_matches_online_adapter_path(self):
        from mixtral_lab.second_round import SecondRoundConfig, SecondRoundResidual
        adapter = MixtralAdapter(tiny())
        module = SecondRoundResidual(adapter.hidden_size, SecondRoundConfig(rank=2))
        with torch.no_grad():
            module.up.add_(.001)
        row = {'id': 'x', 'input_ids': torch.tensor([1,2,3,0]), 'attention_mask': torch.tensor([1,1,1,0])}
        functions = runner.profile_functions(adapter, module, 'adapter')
        with torch.no_grad():
            candidate_logits = functions['frozen_candidate_adapter'](row)
            h0 = adapter.capture(**runner.batch_row(row, 'cpu'))
            h = h0.reshape(-1, adapter.hidden_size)
            reference = adapter.readout(module(adapter.step_pairs(h, adapter.native_pair(h)))).reshape(1, 4, -1)
        torch.testing.assert_close(candidate_logits, reference, atol=0, rtol=0)
        self.assertEqual(set(functions), {'native', 'controlled_original_r1', 'frozen_candidate_adapter'})

    def test_real_full_document_generalization_metrics_for_both_candidate_types(self):
        from mixtral_lab.training import ResidualPairRouter
        from mixtral_lab.second_round import SecondRoundConfig, SecondRoundResidual
        adapter = MixtralAdapter(tiny())
        rows = [{'id': 'one', 'input_ids': torch.tensor([1,2,3,0,0]),
                 'attention_mask': torch.tensor([1,1,1,0,0])},
                {'id': 'two', 'input_ids': torch.tensor([2,4,6,8,10]),
                 'attention_mask': torch.tensor([1,1,1,1,1])}]
        actor = ResidualPairRouter(adapter.feature_dim, .8)
        for kind, candidate in [('router', actor), ('adapter', SecondRoundResidual(
                adapter.hidden_size, SecondRoundConfig(rank=2)))]:
            c = copy.deepcopy(self.config)
            c['fit']['kind'] = kind
            result = runner.evaluate_generalization(adapter, rows, candidate, c)
            self.assertEqual([r['document_id'] for r in result['per_document']], ['one', 'two'])
            self.assertEqual([r['tokens'] for r in result['per_document']], [2, 4])
            self.assertEqual(result['candidate']['tokens'], 6)
            self.assertEqual(result['candidate']['sequences'], 2)
            self.assertFalse(result['quality_gate']['passed'])
            self.assertTrue(result['quality_gate']['controlled_native_nll_screen_passed'])
            self.assertAlmostEqual(result['candidate']['nll'], result['native']['nll'], places=6)

    def test_sources_include_the_new_runner_and_training_modules(self):
        sources = runner.fit_code_identity()['sources']
        for name in ('experiments/mixtral_fit_experiment.py', 'mixtral_lab/fit_protocol.py',
                     'mixtral_lab/full_action_fit.py', 'mixtral_lab/adapter_fit.py', 'mixtral_lab/adaptive_fit.py'):
            self.assertEqual(sources[name], sha256_file(name))

    def test_gate_cannot_be_satisfied_by_flag_alone_or_inconsistent_metrics(self):
        quality = {'role': 'selection_development', 'native': {'nll': 2., 'tokens': 4, 'sequences': 1},
                   'controlled_original': {'nll': 2., 'tokens': 4, 'sequences': 1},
                   'candidate': {'nll': 1., 'tokens': 4, 'sequences': 1},
                   'quality_gate': {'passed': True, 'controlled_native_nll_screen_passed': True,
                     'threshold_nats': .001, 'gain_vs_native_nats': 1., 'gain_vs_controlled_nats': 1.}}
        quality['per_document'] = [{'document_id': 'selection', 'tokens': 4, 'native_nll': 2.,
            'controlled_nll': 2., 'candidate_nll': 1., 'candidate_minus_native_nll': -1.,
            'candidate_minus_controlled_nll': -1.}]
        report = {'generalization': quality, 'config': self.config,
                  'identity': {'selection': {'selected_document_ids': ['selection']}}}
        protocol.require_quality_evidence(report, .001)
        for key, value in [('gain_vs_native_nats', float('nan')), ('gain_vs_native_nats', .002),
                           ('controlled_native_nll_screen_passed', False)]:
            bad = copy.deepcopy(quality)
            bad['quality_gate'][key] = value
            with self.assertRaises(ValueError):
                protocol.require_quality_evidence({**report, 'generalization': bad}, .001)


if __name__ == '__main__':
    unittest.main()
