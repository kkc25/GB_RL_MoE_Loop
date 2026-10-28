"""CPU-only four-role/true-checkpoint integration; never launch/download a model."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from experiments import mixtral_recovery as runner
from mixtral_lab import recovery_protocol as protocol
from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.protocol import canonical_hash, sha256_file
from tests.test_mixtral_adapter import tiny
from tests.test_mixtral_protocol import config as base_config, Tokenizer


def recovery_config(root):
    config = base_config(root)
    for key in ('training', 'fit', 'oracle', 'second_round'):
        config.pop(key, None)
    config['schema'] = protocol.SCHEMA
    development = root/'development.jsonl'
    development.write_text(json.dumps({'id': 'development', 'text': 'document 3 independent checkpoint development'})+'\n')
    config['data_files']['development'] = str(development)
    config['sequences']['development'] = 1
    config['recovery'] = {'kind': 'router', 'training': {
        'updates': 2, 'batch_contexts': 2, 'max_train_contexts': 4, 'max_score_calls': 120,
        'eval_interval': 1, 'development_patience': 2, 'min_updates': 0, 'learning_rate': .1,
        'epsilon': .99, 'seed': 17}, 'development_min_gain': .001, 'selection_min_gain': .001,
        'bridge_tolerances': {'hidden_max_abs': 1e-5, 'feature_max_abs': 1e-5, 'ce_max_abs': 1e-5}}
    return config


class RecoveryProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = recovery_config(self.root)

    def install_runtime(self, *, development_nll=1.8, selection_nll=1.8,
                        zero_pass=True, learned_pass=True, actual_bridge=False):
        stack = self.enterContext(ExitStack())
        model = tiny()
        adapter = MixtralAdapter(model)
        self.adapter = adapter
        info = {'model': {'content_sha256': 'a'*64}, 'tokenizer': {'content_sha256': 'b'*64}, 'cuda_devices': []}
        inputs = {'metadata': model.config, 'tokenizer': Tokenizer(),
                  'identity': {'model': {'files': {'config.json': 'a'}}, 'tokenizer': {'files': {'tokenizer.json': 'b'}}}}
        stack.enter_context(patch.object(runner, 'load_local_inputs', return_value=inputs))
        self.heavy = stack.enter_context(patch.object(runner, 'load_local_model', return_value=(model, inputs['tokenizer'], adapter, info)))
        stack.enter_context(patch.object(runner, 'normalized_artifact_stats', return_value={'model': {'files': ['immutable']}}))
        stack.enter_context(patch.object(runner, 'recovery_code_identity', return_value={'commit': 'fixed', 'sources': {'recovery': 'fixed'}}))
        stack.enter_context(patch.object(runner, 'runtime_versions', return_value={'torch': 'test', 'cuda': None}))
        stack.enter_context(patch.object(runner.torch.cuda, 'is_available', return_value=True))
        stack.enter_context(patch.object(runner.torch.cuda, 'is_current_stream_capturing', return_value=False))
        stack.enter_context(patch.object(runner, 'validate_native', return_value={'eligible': True}))
        self.route = stack.enter_context(patch.object(runner, 'route_diagnostics', return_value={'greedy_native_agreement': .5}))
        self.timing = stack.enter_context(patch.object(runner, 'measure_calls', return_value={'variants': {}}))
        self.evaluated_roles = []
        def metric(adapter, rows, policy=None, *, native=False):
            self.evaluated_roles.extend(row['id'] for row in rows)
            changed = policy is not None and bool(torch.count_nonzero(getattr(policy, 'weight', getattr(policy, 'up', torch.zeros(1)))))
            score = 2. if not changed else (development_nll if rows[0]['id'] == 'development' else selection_nll)
            return {'nll': score, 'tokens': 4, 'sequences': len(rows)}
        self.evaluate = stack.enter_context(patch.object(runner, 'evaluate', side_effect=metric))
        self.evaluate_adapter = stack.enter_context(patch.object(runner, 'evaluate_second_round', side_effect=lambda a,r,c: metric(a,r,c)))
        if not actual_bridge:
            def bridge(a, rows, candidate, c, manifest, metadata=None):
                tolerances = {'logit_max_abs': c['logit_max_abs_tolerance'], 'ce_mean_abs': c['ce_mean_abs_tolerance'],
                              **c['recovery']['bridge_tolerances']}
                keys = {'cached_vs_online.h0', 'cached_native_vs_zero.ce'}
                for prefix in ('native_vs_controlled','native_vs_zero','cached_vs_online.native','cached_vs_online.zero','cached_vs_online.learned'):
                    keys.update(prefix + '.' + suffix for suffix in ('logits','hidden','ce'))
                if c['recovery']['kind'] == 'router':
                    keys.update({'cached_vs_online.features', 'cached_readout_vs_score_pairs.ce'})
                    keys.update('cached_vs_online.forced_cached.' + suffix for suffix in ('logits','hidden','ce'))
                else:
                    keys.add('cached_vs_online.h1')
                mapping = {'logits':'logit_max_abs', 'hidden':'hidden_max_abs', 'h0':'hidden_max_abs',
                           'h1':'hidden_max_abs', 'features':'feature_max_abs', 'ce':'ce_max_abs'}
                differences = {}
                for key in keys:
                    suffix = key.rsplit('.', 1)[-1]
                    error = 1. if ((not zero_pass and key == 'native_vs_zero.ce') or
                        (metadata is not None and not learned_pass and key == 'cached_vs_online.learned.ce')) else 0.
                    row = {'elements':4, 'max_abs':error, 'mean_abs':error, 'max_abs_tolerance':tolerances[mapping[suffix]],
                           'within_declared_tolerance':error == 0.}
                    if suffix == 'ce': row['mean_abs_tolerance'] = tolerances['ce_mean_abs']
                    differences[key] = row
                historical = {} if metadata is None else (
                    {'native':metadata['train_metrics']['native_ce'], 'learned':metadata['train_metrics']['greedy_ce']}
                    if c['recovery']['kind'] == 'router' else {'learned':metadata['train_probe']['nll']})
                reproduction = {name:{'historical_nll':value, 'recomputed_nll':value, 'absolute_difference':0.,
                    'tolerance':c['nll_tolerance'], 'within_declared_tolerance':True} for name,value in historical.items()}
                return {'schema':'mixtral-cache-online-bridge-v1', 'status': 'complete', 'role': 'train_only',
                        'kind':c['recovery']['kind'], 'sampling_sha256': manifest['sampling_sha256'],
                        'zero_screen_passed': zero_pass, 'numerical_screen_passed': zero_pass and (metadata is None or learned_pass),
                        'chosen_update': None if metadata is None else metadata['update'], 'tolerances':tolerances,
                        'absolute_differences':differences, 'historical_metric_reproduction':reproduction,
                        'cached':{name:{'tokens':manifest['actual_count'], 'nll':historical.get(name, 2.)} for name in ('native','zero','learned')},
                        'candidate_training_config':copy.deepcopy(c['recovery']['training']), 'action_agreement_screen_passed':True,
                        'action_agreement':({k:{'count':4,'agree':4,'fraction':1.} for k in ('native','zero','learned')} if c['recovery']['kind'] == 'router' else {})}
            self.bridge = stack.enter_context(patch.object(runner, 'run_bridge', side_effect=bridge))

    def use_adapter(self):
        self.config['recovery'].update(kind='adapter', training={'rank': 2, 'updates': 2, 'batch_contexts': 2,
            'max_train_contexts': 4, 'probe_chunk_contexts': 2, 'eval_interval': 1, 'min_updates': 0,
            'development_patience': 2, 'learning_rate': .1, 'residual_scale': .1, 'native_kl_beta': .1, 'seed': 17})

    def run_stage(self, stage, *, name=None, tune=None, evaluation=None):
        config = copy.deepcopy(self.config)
        config['output_dir'] = str(self.root/(name or stage))
        config_path = self.root/((name or stage)+'.json')
        config_path.write_text(json.dumps(config))
        argv = ['--config', str(config_path), '--stage', stage]
        if tune:
            argv += ['--tune-report', str(tune/'report.json'), '--checkpoint', str(tune/'tune'/'checkpoint_best.pt')]
        if evaluation:
            argv += ['--evaluation-report', str(evaluation/'report.json')]
        return runner.main(argv)

    def edit_report(self, directory, edit, *, reseal=True):
        path = directory/'report.json'
        report = json.loads(path.read_text())
        edit(report)
        path.write_text(json.dumps(protocol.seal_report(report) if reseal else report))

    def test_true_router_checkpoint_tune_evaluate_profile_no_selection_during_tune(self):
        self.install_runtime()
        tuned = self.run_stage('tune')
        self.assertEqual(set(self.evaluated_roles), {'development'})
        self.route.assert_not_called(); self.timing.assert_not_called()
        self.assertEqual(tuned['work_accounting']['outer_selection_evaluations'], 0)
        self.assertGreater(tuned['checkpoint']['update'], 0)
        self.assertTrue(tuned['development_gate']['passed'])
        self.assertEqual(tuned['identity']['context_sampling']['role'], 'train')
        original_manifest = tuned['identity']['context_sampling']
        self.assertTrue(all('role' not in row for row in original_manifest['contexts']))
        self.assertEqual(original_manifest['sampling_sha256'], canonical_hash({k:v for k,v in original_manifest.items() if k != 'sampling_sha256'}))
        self.assertEqual(set(tuned['preflight']['data']), set(protocol.ROLES))
        self.assertTrue((self.root/'tune'/'tune'/'evaluation_checkpoints'/'checkpoint_000000.pt').exists())
        self.assertEqual(tuned['baseline_checkpoint']['update'], 0)
        self.evaluated_roles.clear()
        evaluated = self.run_stage('evaluate', tune=self.root/'tune')
        self.assertEqual(self.evaluated_roles, ['selection']*3)
        self.assertTrue(evaluated['evaluation']['quality_gate']['passed'])
        self.assertEqual(evaluated['evaluation']['document_statistics']['native']['improved_documents'], 1)
        self.evaluated_roles.clear()
        profiled = self.run_stage('profile', tune=self.root/'tune', evaluation=self.root/'evaluate')
        self.assertEqual(self.evaluated_roles, [])
        self.timing.assert_called_once()
        self.assertEqual(profiled['status'], 'complete')

    def test_true_adapter_checkpoint_integration(self):
        self.use_adapter(); self.install_runtime()
        tuned = self.run_stage('tune')
        self.assertTrue(tuned['development_gate']['passed'])
        self.assertIn('native_teacher', tuned['tune_summary'])
        self.assertEqual(set(self.evaluated_roles), {'development'})
        self.assertTrue((self.root/'tune'/'tune'/'checkpoint_eval_000000.pt').exists())
        self.run_stage('evaluate', tune=self.root/'tune')
        self.run_stage('profile', tune=self.root/'tune', evaluation=self.root/'evaluate')
        self.timing.assert_called_once()

    def test_update_zero_fallback_retains_training_diagnostics_and_blocks_outer(self):
        self.install_runtime(development_nll=2.1)
        tuned = self.run_stage('tune')
        self.assertEqual(tuned['checkpoint']['update'], 0)
        self.assertFalse(tuned['development_gate']['passed'])
        self.assertGreater(tuned['tune_summary']['successful_updates'], 0)
        self.assertIn('latest_train_metrics', tuned['tune_summary'])
        self.assertEqual(len(tuned['tune_summary']['development_history']), 3)
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'positive development gain gate'):
            self.run_stage('evaluate', tune=self.root/'tune')
        self.heavy.assert_not_called()

    def test_native_bridge_screen_fails_before_any_training_callback(self):
        self.install_runtime(zero_pass=False)
        with self.assertRaisesRegex(ValueError, 'zero_screen_passed failed'):
            self.run_stage('tune')
        self.evaluate.assert_not_called()
        self.assertFalse((self.root/'tune'/'tune').exists())
        report = json.loads((self.root/'tune'/'report.json').read_text())
        self.assertEqual(report['failure_phase'], 'bridge_pretraining')
        self.assertFalse(report['bridge_pretraining']['zero_screen_passed'])
        self.assertIn('total_wall_seconds', report)

    def test_learned_bridge_failure_allows_outer_diagnosis_but_blocks_profile(self):
        self.install_runtime(learned_pass=False)
        tuned = self.run_stage('tune')
        self.assertFalse(tuned['training_path_equivalence_established'])
        evaluated = self.run_stage('evaluate', tune=self.root/'tune')
        self.assertTrue(evaluated['evaluation']['quality_gate']['passed'])
        self.assertFalse(evaluated['training_path_equivalence_established'])
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'numerical_screen_passed failed'):
            self.run_stage('profile', tune=self.root/'tune', evaluation=self.root/'evaluate')
        self.heavy.assert_not_called(); self.timing.assert_not_called()

    def test_negative_outer_quality_blocks_profile_before_weights(self):
        self.install_runtime(selection_nll=2.01)
        self.run_stage('tune')
        evaluated = self.run_stage('evaluate', tune=self.root/'tune')
        self.assertFalse(evaluated['evaluation']['quality_gate']['passed'])
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'quality gate'):
            self.run_stage('profile', tune=self.root/'tune', evaluation=self.root/'evaluate')
        self.heavy.assert_not_called(); self.timing.assert_not_called()

    def test_subthreshold_development_gain_blocks_outer(self):
        self.install_runtime(development_nll=1.9995)
        self.run_stage('tune')
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'positive development gain gate'):
            self.run_stage('evaluate', tune=self.root/'tune')
        self.heavy.assert_not_called()

    def test_stale_code_artifact_data_runtime_and_contract_rejected_before_load(self):
        self.install_runtime(); self.run_stage('tune')
        path = self.root/'tune'/'report.json'; original = path.read_bytes()
        for key in ('code', 'artifact_stats', 'data', 'local_inputs', 'train_contexts', 'runtime_versions'):
            with self.subTest(key=key):
                def edit(r):
                    r['freeze_manifest'][key] = {'changed': True}
                    r['freeze_manifest_sha256'] = canonical_hash(r['freeze_manifest'])
                self.edit_report(self.root/'tune', edit)
                self.heavy.reset_mock()
                with self.assertRaisesRegex(ValueError, 'frozen '+key+' mismatch'):
                    self.run_stage('evaluate', tune=self.root/'tune', name='changed_'+key)
                self.heavy.assert_not_called()
                path.write_bytes(original)
        self.config['recovery']['selection_min_gain'] = .002
        with self.assertRaisesRegex(ValueError, 'scientific configuration mismatch'):
            self.run_stage('evaluate', tune=self.root/'tune', name='changed_config')

    def test_actual_model_identity_mismatch_fails_after_loading_not_as_success(self):
        self.install_runtime(); self.run_stage('tune')
        model, tokenizer, adapter, info = self.heavy.return_value
        self.heavy.return_value = model, tokenizer, adapter, {**info, 'model': {'content_sha256': 'changed'}}
        with self.assertRaisesRegex(ValueError, 'loaded model/tokenizer/runtime identity'):
            self.run_stage('evaluate', tune=self.root/'tune')
        self.assertEqual(json.loads((self.root/'evaluate'/'report.json').read_text())['status'], 'failed')

    def test_modified_best_or_baseline_checkpoint_rejected_before_load(self):
        for kind in ('best', 'initial'):
            with self.subTest(kind=kind):
                self.install_runtime()
                destination = 'tune_'+kind
                self.run_stage('tune', name=destination)
                (self.root/destination/'tune'/('checkpoint_'+kind+'.pt')).write_bytes(b'replaced')
                self.heavy.reset_mock()
                with self.assertRaisesRegex(ValueError, 'checkpoint bytes changed'):
                    self.run_stage('evaluate', name='evaluate_'+kind, tune=self.root/destination)
                self.heavy.assert_not_called()

    def test_resealed_initial_nll_tamper_cannot_invent_positive_gain(self):
        self.install_runtime(); self.run_stage('tune')
        def edit(r):
            r['tune_summary']['initial_development_nll'] = 99.
            r['development_gate']['gain_nats'] = 99. - r['tune_summary']['best_development_nll']
        self.edit_report(self.root/'tune', edit)
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'initial development NLL'):
            self.run_stage('evaluate', tune=self.root/'tune')
        self.heavy.assert_not_called()

    def test_resealed_development_best_metric_tamper_fails_cpu_checkpoint_preflight(self):
        self.install_runtime(); self.run_stage('tune')
        def edit(r):
            r['tune_summary']['best_development_nll'] = 1.
            r['development_gate']['gain_nats'] = 1.
        self.edit_report(self.root/'tune', edit)
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'checkpoint development-best metrics'):
            self.run_stage('evaluate', tune=self.root/'tune')
        self.heavy.assert_not_called()

    def test_resealed_outer_per_document_tamper_and_wrong_parent_block_profile(self):
        self.install_runtime(); self.run_stage('tune'); self.run_stage('evaluate', tune=self.root/'tune')
        path = self.root/'evaluate'/'report.json'; original = path.read_bytes()
        self.edit_report(self.root/'evaluate', lambda r: r['evaluation']['per_document'][0].update(candidate_nll=.5))
        self.heavy.reset_mock()
        with self.assertRaisesRegex(ValueError, 'paired delta mismatch'):
            self.run_stage('profile', tune=self.root/'tune', evaluation=self.root/'evaluate')
        self.heavy.assert_not_called()
        path.write_bytes(original)
        self.edit_report(self.root/'evaluate', lambda r: r['evidence'].update(tune_report_sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'different tune report'):
            self.run_stage('profile', name='profile_parent', tune=self.root/'tune', evaluation=self.root/'evaluate')
        self.heavy.assert_not_called()

    def test_missing_inputs_nonfinite_config_and_legacy_schema_leave_failure_evidence(self):
        self.install_runtime()
        with self.assertRaisesRegex(ValueError, 'require --tune-report'):
            self.run_stage('evaluate')
        self.config['recovery']['development_min_gain'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.run_stage('tune', name='nan')
        self.assertFalse(json.loads((self.root/'nan'/'report.json').read_text())['model_load_started'])
        self.config['recovery']['development_min_gain'] = .001
        self.config['schema'] = 'mixtral-frozen-routing-v1'
        with self.assertRaisesRegex(ValueError, 'unknown recovery schema'):
            self.run_stage('tune', name='legacy')
        self.heavy.assert_not_called()

    def test_all_four_roles_raw_and_effective_leakage_checked_before_weights(self):
        self.install_runtime()
        dev = Path(self.config['data_files']['development'])
        original = dev.read_bytes()
        dev.write_bytes(Path(self.config['data_files']['train']).read_bytes())
        with self.assertRaisesRegex(ValueError, 'duplicate document'):
            self.run_stage('tune')
        dev.write_text(json.dumps({'id':'different', 'text':'document 0 truncation duplicate with different raw tail'})+'\n')
        with self.assertRaisesRegex(ValueError, 'duplicate effective token'):
            self.run_stage('tune', name='effective_duplicate')
        self.heavy.assert_not_called(); self.evaluate.assert_not_called()
        dev.write_bytes(original)

    def test_unselected_raw_and_effective_sources_also_checked(self):
        train = Path(self.config['data_files']['train'])
        train.write_text(train.read_text()+json.dumps({'id':'unselected', 'text':'document 3 collision with development'})+'\n')
        with self.assertRaisesRegex(ValueError, 'duplicate effective token'):
            protocol.prepare_data(self.config, Tokenizer())

    def test_actual_context_and_token_tensor_quota_before_weights(self):
        self.install_runtime()
        self.config['max_train_contexts'] = 5
        self.config['recovery']['training'].update(max_train_contexts=5, max_score_calls=150)
        with self.assertRaisesRegex(ValueError, 'only 4 valid next-token positions'):
            self.run_stage('tune')
        self.heavy.assert_not_called()
        bad = Tokenizer()
        bad.encode = lambda *a, **k: [1]*7
        with self.assertRaisesRegex(ValueError, 'tensor quota'):
            protocol.prepare_data(self.config, bad)

    def test_schema_disallows_legacy_sections_newtest_and_duplicate_files(self):
        for field in ('training', 'fit', 'oracle', 'second_round'):
            c = copy.deepcopy(self.config); c[field] = {}
            with self.assertRaisesRegex(ValueError, 'legacy'):
                protocol.validate_recovery_config(c)
        c = copy.deepcopy(self.config); c['data_files']['newtest'] = 'forbidden'
        with self.assertRaisesRegex(ValueError, 'newtest'):
            protocol.validate_recovery_config(c)
        c = copy.deepcopy(self.config); c['data_files']['development'] = c['data_files']['train']
        with self.assertRaisesRegex(ValueError, 'four recovery roles'):
            protocol.validate_recovery_config(c)
        c = copy.deepcopy(self.config); c['recovery']['bridge_tolerances'] = {}
        with self.assertRaisesRegex(ValueError, 'explicitly specify'):
            protocol.validate_recovery_config(c)

    def test_same_role_effective_duplicates_are_reported_without_resampling(self):
        self.config['sequences']['development'] = 2
        dev = Path(self.config['data_files']['development'])
        dev.write_text(dev.read_text()+json.dumps({'id':'development2','text':'document 3 distinct full raw wording'})+'\n')
        data, identity = protocol.prepare_data(self.config, Tokenizer())
        stats = identity['development']['effective_input_statistics']
        self.assertEqual(stats['redundant_documents'], 1)
        self.assertFalse(stats['sampling_changed'])
        self.assertEqual([r['id'] for r in data['development']], ['development','development2'])

    def test_report_digest_and_bounds_and_fresh_output(self):
        self.install_runtime(); self.run_stage('tune')
        path = self.root/'tune'/'report.json'; before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_stage('tune')
        self.assertEqual(path.read_bytes(), before)
        self.edit_report(self.root/'tune', lambda r: r.update(note='tamper'), reseal=False)
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            protocol.read_report(path, 'tune')
        big = self.root/'oversized.json'
        with big.open('wb') as stream:
            stream.seek(protocol.MAX_REPORT_BYTES); stream.write(b'!')
        with self.assertRaisesRegex(ValueError, '64 MiB bound'):
            protocol.read_report(big, 'tune')

    def test_callback_budget_and_role_contract(self):
        self.install_runtime()
        config = protocol.validate_recovery_config(self.config)
        rows = protocol.prepare_data(config, Tokenizer())[0]['development']
        work = {}
        callback = runner.make_development_evaluator(self.adapter, rows, config, work)
        self.assertEqual(callback.contract, {'role':'development','document_ids':['development'],'protocol':protocol.PROTOCOL})
        actor = runner.zero_candidate(self.adapter, config)
        for _ in range(work['development_evaluation_budget']):
            callback(actor)
        with self.assertRaisesRegex(ValueError, 'budget'):
            callback(actor)
        self.assertEqual(work['development_document_forwards'], work['development_document_forward_budget'])

    def test_real_full_document_metrics_both_candidate_types(self):
        adapter = MixtralAdapter(tiny())
        rows = [{'id':'one','input_ids':torch.tensor([1,2,3,0,0]),'attention_mask':torch.tensor([1,1,1,0,0])},
                {'id':'two','input_ids':torch.tensor([2,4,6,8,10]),'attention_mask':torch.tensor([1,1,1,1,1])}]
        for kind in ('router', 'adapter'):
            if kind == 'adapter': self.use_adapter()
            config = protocol.validate_recovery_config(self.config)
            zero = runner.zero_candidate(adapter, config)
            result = runner.evaluate_documents(adapter, rows, zero, config, role='development')
            self.assertEqual([r['tokens'] for r in result['per_document']], [2,4])
            self.assertEqual(result['candidate']['tokens'], 6)
            self.assertAlmostEqual(result['nll'], result['native']['nll'], places=6)
            self.assertEqual(result['document_statistics']['unit'], 'document')
            self.assertIsNone(result['document_statistics']['confidence_interval'])

    def test_resealed_inconsistent_or_missing_bridge_evidence_blocks_before_weights(self):
        self.install_runtime(); self.run_stage('tune')
        path = self.root/'tune'/'report.json'; original = path.read_bytes()
        edits = [lambda b: b['absolute_differences']['cached_vs_online.learned.ce'].update(max_abs=1.),
                 lambda b: b.update(absolute_differences={}),
                 lambda b: b.update(sampling_sha256='changed'),
                 lambda b: b.update(chosen_update=999),
                 lambda b: b['tolerances'].update(ce_max_abs=999.),
                 lambda b: b['cached']['learned'].update(tokens=99),
                 lambda b: b['historical_metric_reproduction']['learned'].update(historical_nll=99.)]
        for index, edit in enumerate(edits):
            self.edit_report(self.root/'tune', lambda r: edit(r['bridge_frozen_candidate']))
            self.heavy.reset_mock()
            with self.assertRaisesRegex(ValueError, 'bridge'):
                self.run_stage('evaluate', tune=self.root/'tune', name='bad_bridge_'+str(index))
            self.heavy.assert_not_called()
            path.write_bytes(original)

    def test_real_bridge_native_preflight_with_true_router_training(self):
        self.install_runtime(actual_bridge=True)
        result = self.run_stage('tune')
        self.assertTrue(result['bridge_pretraining']['zero_screen_passed'])
        self.assertEqual(result['bridge_pretraining']['sampling_sha256'], result['identity']['context_sampling']['sampling_sha256'])
        self.assertEqual(set(self.evaluated_roles), {'development'})
        self.assertTrue(result['bridge_frozen_candidate']['numerical_screen_passed'])
        self.run_stage('evaluate', tune=self.root/'tune')
        self.run_stage('profile', tune=self.root/'tune', evaluation=self.root/'evaluate')

    def test_real_bridge_native_preflight_with_true_adapter_training(self):
        self.use_adapter(); self.install_runtime(actual_bridge=True)
        result = self.run_stage('tune')
        self.assertTrue(result['bridge_pretraining']['zero_screen_passed'])
        self.assertIn('train_inputs_sha256', result['bridge_frozen_candidate']['cache_binding'])
        self.assertEqual(set(self.evaluated_roles), {'development'})
        self.assertTrue(result['bridge_frozen_candidate']['numerical_screen_passed'])
        self.run_stage('evaluate', tune=self.root/'tune')
        self.run_stage('profile', tune=self.root/'tune', evaluation=self.root/'evaluate')

    def test_source_identity_covers_new_runner_protocol_trainers_and_bridge(self):
        sources = runner.recovery_code_identity()['sources']
        for name in ('experiments/mixtral_recovery.py','experiments/mixtral_fit_experiment.py',
                     'mixtral_lab/recovery_protocol.py','mixtral_lab/recovery_router.py',
                     'mixtral_lab/recovery_adapter.py','mixtral_lab/bridge.py'):
            self.assertEqual(sources[name], sha256_file(name))


if __name__ == '__main__':
    unittest.main()
