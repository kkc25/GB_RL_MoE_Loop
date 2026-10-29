"""Offline artifact audit tests: tiny legacy trainer, never real model weights.

Only fixture creation executes a tiny random CPU model. The diagnostic invocation
is separately guarded against model loading, model execution and CUDA access.
"""
import copy
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from experiments import mixtral_document as legacy_runner
from mixtral_lab import document_protocol
from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.document_router import _digest, _tensor_hashes, _sources
from mixtral_lab import router_artifact_diagnostics as diagnosis
from mixtral_lab.recovery_router import RecoveryPairRouter, native_reference_kl
from mixtral_lab.document_router import DocumentRouterConfig, _objective
from tests.test_mixtral_adapter import tiny
from tests.test_mixtral_document_protocol import document_config, paired
from tests.test_mixtral_protocol import Tokenizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy_fixture(root):
    """Build authentic old-format artifacts and a fully sealed tiny tune report."""
    raw = document_config(root, 'router')
    raw['output_dir'] = str(root / 'run')
    cfg = root / 'config.json'
    cfg.write_text(json.dumps(raw))
    model = tiny()
    adapter = MixtralAdapter(model)
    info = {'model': {'content_sha256': 'a' * 64},
            'tokenizer': {'content_sha256': 'b' * 64}, 'cuda_devices': []}
    inputs = {'metadata': model.config, 'tokenizer': Tokenizer(),
              'identity': {'local': 'stable'}}
    def evaluate(adapter, rows, candidate=None, **kwargs):
        state = getattr(candidate, 'weight', torch.zeros(1))
        # Deliberately make update zero development-best, not latest.
        return paired(rows, bool(torch.count_nonzero(state)), score=2.1)
    with ExitStack() as stack:
        for name, value in (
                ('load_local_inputs', inputs),
                ('load_local_model', (model, inputs['tokenizer'], adapter, info)),
                ('normalized_artifact_stats', {'stable': True}),
                ('document_code_identity', {'commit': 'historical', 'sources': {'mixtral_lab/' + k: v for k, v in _sources().items()}}),
                ('runtime_versions', {'runtime': 'cpu-fixture'})):
            stack.enter_context(patch.object(legacy_runner, name, return_value=value))
        stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
        stack.enter_context(patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False))
        stack.enter_context(patch('mixtral_lab.document_training.evaluate_documents', side_effect=evaluate))
        result = legacy_runner.main(['--config', str(cfg), '--stage', 'tune'])
    assert result['status'] == 'complete'
    assert result['tune_summary']['best_update'] == 0
    assert result['tune_summary']['successful_updates'] == 2
    return root / 'run'


class RouterArtifactDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.artifact_run = legacy_fixture(cls.root)
        cls.originals = {p: p.read_bytes() for p in cls.artifact_run.rglob('*') if p.is_file()}

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(dir=self.root))
        self.addCleanup(shutil.rmtree, self.work)

    def tearDown(self):
        # Mutation tests must restore source evidence, even after a failure.
        for path, raw in self.originals.items():
            path.write_bytes(raw)

    def table(self):
        return torch.load(self.artifact_run / 'tune/train_surrogate_table.pt', map_location='cpu', weights_only=True)

    def checkpoint(self, name='latest'):
        return torch.load(self.artifact_run / f'tune/checkpoint_{name}.pt', map_location='cpu', weights_only=True)

    def save_checkpoint(self, value, name='latest', reseal=False):
        if reseal:
            value['actor_state_hashes'] = _tensor_hashes(value['actor_state_dict'])
            value['payload_sha256'] = _digest({k: v for k, v in value.items()
                                               if k not in ('payload_sha256', 'actor_state_dict')})
        torch.save(value, self.artifact_run / f'tune/checkpoint_{name}.pt')

    def save_report(self, value, reseal=True):
        if reseal:
            value = document_protocol.seal_report(value)
        (self.artifact_run / 'report.json').write_text(json.dumps(value))

    def report(self):
        return json.loads((self.artifact_run / 'report.json').read_text())

    def diagnose(self, name='latest', **kwargs):
        return diagnosis.diagnose_router_artifacts(
            self.artifact_run / 'report.json', self.artifact_run / f'tune/checkpoint_{name}.pt',
            self.artifact_run / 'tune/train_surrogate_table.pt', self.work / 'diagnostic', **kwargs)

    def test_valid_legacy_inputs_cpu_only_and_bitwise_nonmutating(self):
        before = {p: sha(p) for p in self.originals}
        real_load = torch.load
        with ExitStack() as stack:
            for target in ('transformers.AutoModelForCausalLM.from_pretrained',
                           'transformers.AutoTokenizer.from_pretrained',
                           'experiments.mixtral_document.load_local_model',
                           'mixtral_lab.adapter.MixtralAdapter._forward_r1',
                           'torch.cuda.init', 'torch.cuda.is_available'):
                stack.enter_context(patch(target, side_effect=AssertionError('forbidden model/GPU execution')))
            loader = stack.enter_context(patch('torch.load', wraps=real_load))
            result = self.diagnose(batch_size=3)
        self.assertGreaterEqual(loader.call_count, 3)
        for call in loader.call_args_list:
            self.assertEqual(call.kwargs.get('map_location'), 'cpu')
            self.assertIs(call.kwargs.get('weights_only'), True)
        self.assertEqual(before, {p: sha(p) for p in self.originals})
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['checkpoint_update'], 2)
        self.assertIn('latest', result['checkpoint_meaning'])
        self.assertEqual(result['numeric_protocol']['batch_size'], 3)
        self.assertFalse(result['numeric_protocol']['deployment_batch_parity_claim'])
        self.assertFalse(result['model_or_tokenizer_loaded'])
        self.assertEqual(result['gpu_calls'], 0)
        self.assertEqual(result['training_updates'], 0)
        self.assertFalse(result['selection_or_calibration_or_test_evaluated'])
        self.assertEqual(result['original_executed_full_document_evidence']['source'],
                         'sealed tune report, not recomputed')
        self.assertIn('not deployment gain', result['label'])
        self.assertTrue((self.work / 'diagnostic/summary.md').is_file())
        disk = json.loads((self.work / 'diagnostic/diagnostics.json').read_text())
        self.assertEqual(result, disk)
        self.assertEqual(result['diagnostic_sha256'],
                         _digest({k: v for k, v in result.items() if k != 'diagnostic_sha256'}))

    def test_cli_writes_report_with_explicit_checkpoint_and_cpu_batch(self):
        from experiments.mixtral_router_diagnose import main
        code = main(['--tune-report', str(self.artifact_run / 'report.json'),
                     '--checkpoint', str(self.artifact_run / 'tune/checkpoint_latest.pt'),
                     '--table', str(self.artifact_run / 'tune/train_surrogate_table.pt'),
                     '--output-dir', str(self.work / 'cli'), '--batch-size', '2'])
        self.assertEqual(code, 0)
        result = json.loads((self.work / 'cli/diagnostics.json').read_text())
        self.assertEqual(result['numeric_protocol']['batch_size'], 2)
        self.assertEqual(result['checkpoint_update'], 2)

    def test_strict_legacy_loader_consumes_checked_snapshot_not_original_path(self):
        original = self.artifact_run / 'tune/checkpoint_latest.pt'
        original_bytes = original.read_bytes()
        legacy_load = diagnosis.load_document_router_checkpoint
        def checked(path, *args, **kwargs):
            self.assertNotEqual(Path(path).resolve(), original.resolve())
            self.assertEqual(Path(path).read_bytes(), original_bytes)
            return legacy_load(path, *args, **kwargs)
        with patch.object(diagnosis, 'load_document_router_checkpoint', side_effect=checked) as loader:
            self.diagnose()
        loader.assert_called_once()

    def test_best_is_zero_fallback_not_mislabelled_latest(self):
        result = self.diagnose('best')
        self.assertEqual(result['checkpoint_update'], 0)
        self.assertIn('development-best', result['checkpoint_meaning'])
        self.assertEqual(result['metrics']['counts']['candidate_changed_argmax_from_update_zero'], 0)
        # Adding this diagnostic module did not invalidate historical source lists.
        self.assertNotIn('mixtral_lab/router_artifact_diagnostics.py',
                         self.report()['identity']['code']['sources'])

    def test_fresh_output_required_before_loading(self):
        (self.work / 'diagnostic').mkdir()
        with patch.object(diagnosis, 'validate_artifacts', side_effect=AssertionError('must preflight')):
            with self.assertRaisesRegex(ValueError, 'fresh'):
                self.diagnose()

    def test_report_corruption_missing_and_unsealed_fail_closed(self):
        path = self.artifact_run / 'report.json'
        original = path.read_bytes()
        for mode in ('missing', 'corrupt', 'seal', 'failed', 'wrong_stage'):
            with self.subTest(mode=mode):
                path.write_bytes(original)
                if mode == 'missing':
                    path.unlink()
                elif mode == 'corrupt':
                    path.write_text('{broken')
                else:
                    value = self.report()
                    if mode == 'seal': value['report_sha256'] = '0' * 64
                    elif mode == 'failed': value['status'] = 'failed'
                    else: value['stage'] = 'evaluate'
                    self.save_report(value, reseal=mode != 'seal')
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.diagnose()
                self.assertFalse((self.work / 'diagnostic').exists())
        path.write_bytes(original)

    def test_table_tampering_rejected(self):
        path = self.artifact_run / 'tune/train_surrogate_table.pt'
        original = path.read_bytes()
        for mode in ('tensor_hash', 'provenance_target', 'provenance_id', 'wrong_run',
                     'shape', 'nan', 'negative', 'metadata', 'native_pair'):
            with self.subTest(mode=mode):
                path.write_bytes(original)
                value = self.table()
                if mode == 'tensor_hash': value['tensors']['costs'][0, 0] += .1
                elif mode == 'provenance_target': value['provenance'][0]['target'] += 1
                elif mode == 'provenance_id': value['provenance'][0]['document_id'] = 'other-run'
                elif mode == 'wrong_run': value['binding']['identities']['model'] = {'wrong': True}
                elif mode == 'shape': value['tensors']['costs'] = value['tensors']['costs'][:, :27]
                elif mode == 'nan': value['tensors']['features'][0, 0] = math.nan
                elif mode == 'negative': value['tensors']['costs'][0, 0] = -1
                elif mode == 'metadata': value['scope'] = 'deployment oracle'
                else: value['tensors']['native_pairs'][0] = 0
                torch.save(value, path)
                with self.assertRaises(ValueError): self.diagnose()
        path.write_bytes(original)

    def test_checkpoint_metadata_config_and_state_tampering_rejected(self):
        path = self.artifact_run / 'tune/checkpoint_latest.pt'
        original = path.read_bytes()
        for mode in ('epsilon', 'update', 'train_metrics', 'nan', 'shape', 'identity', 'source'):
            with self.subTest(mode=mode):
                path.write_bytes(original)
                value = self.checkpoint()
                if mode == 'epsilon': value['epsilon'] = .5
                elif mode == 'update': value['update'] = 1
                elif mode == 'train_metrics': value['train_metrics']['tokens'] += 1
                elif mode == 'nan': value['actor_state_dict']['bias'][0] = math.nan
                elif mode == 'shape': value['actor_state_dict']['bias'] = torch.zeros(27)
                elif mode == 'identity': value['binding']['identities']['train']['selected_document_ids'] = ['wrong']
                else: value['binding']['sources']['document_router.py'] = '0' * 64
                self.save_checkpoint(value, reseal=True)
                with self.assertRaises(ValueError): self.diagnose()
        path.write_bytes(original)

    def test_report_config_disagreement_even_with_valid_seal(self):
        value = self.report()
        value['config']['document']['training']['temperature'] *= 2
        self.save_report(value)
        with self.assertRaises(ValueError): self.diagnose()


class RouterDiagnosticMathematicsTests(unittest.TestCase):
    def inputs(self):
        actor = RecoveryPairRouter(22, .8)
        features = torch.zeros(3, 22)
        features[:, -4:] = torch.tensor([0., 1., 1., 0.])
        pairs = torch.tensor([[0, 1]] * 3)
        costs = torch.ones(3, 28, dtype=torch.float64)
        costs[1, 0] = 2.
        costs[2, 1] = 0.
        tensors = {'features': features, 'native_pairs': pairs, 'costs': costs,
                   'native_losses': torch.tensor([1., 2., 1.], dtype=torch.float64)}
        config = {'temperature': .5, 'reference_kl_weight': .01, 'objective': 'soft_target_ce'}
        return actor, tensors, config

    def test_handcomputed_soft_targets_prior_barrier_and_objectives(self):
        actor, tensors, config = self.inputs()
        with torch.no_grad(): actor.bias[1] = 2.
        result = diagnosis.compute_diagnostics(actor, tensors, config, batch_size=2)
        # Explicit scalar math, independent of the implementation's tensor metrics.
        prior = [8 / 35] + [1 / 35] * 27
        logits = [math.log(v) + (2 / 3 if i == 1 else 0) for i, v in enumerate(prior)]
        total = sum(math.exp(v) for v in logits)
        prob = [math.exp(v) / total for v in logits]
        ce, entropy, expected = [], [], []
        for row in tensors['costs'].tolist():
            weights = [math.exp(-c / .5) for c in row]
            q = [v / sum(weights) for v in weights]
            entropy.append(-sum(v * math.log(v) for v in q))
            ce.append(-sum(v * math.log(p) for v, p in zip(q, prob)))
            expected.append(sum(p * c for p, c in zip(prob, row)))
        kl = sum(p * math.log(p / r) for p, r in zip(prob, prior))
        metrics = result['distributions']
        self.assertAlmostEqual(metrics['target_entropy']['mean'], sum(entropy) / 3, places=12)
        self.assertAlmostEqual(metrics['update_zero_prior.native_vs_best_alternate_margin']['mean'], math.log(8), places=6)
        self.assertAlmostEqual(metrics['candidate.native_vs_best_alternate_margin']['mean'], math.log(8) - 2 / 3, places=6)
        objective = result['objective_components']['candidate']
        self.assertAlmostEqual(objective['soft_target_ce'], sum(ce) / 3, places=6)
        self.assertAlmostEqual(objective['native_reference_kl'], kl, places=6)
        self.assertAlmostEqual(objective['expected_uniform_action_ce_surrogate'], sum(expected) / 3, places=6)
        self.assertAlmostEqual(objective['configured_objective'], sum(ce) / 3 + .01 * kl, places=6)
        self.assertEqual(result['counts']['positions'], 3)
        self.assertEqual(result['counts']['zero_cost_std'], 1)
        self.assertEqual(result['counts']['native_best_alternate_exact_ties'], 1)
        self.assertEqual(result['counts']['alternate_cost_better_than_native'], 2)
        self.assertEqual(result['counts']['candidate_changed_argmax_from_update_zero'], 0)
        config['objective'] = 'expected_ce'
        changed = diagnosis.compute_diagnostics(actor, tensors, config)
        self.assertAlmostEqual(changed['objective_components']['candidate']['configured_objective'],
                               sum(expected) / 3 + .01 * kl, places=6)

    def test_reported_objective_matches_exact_historical_training_math(self):
        actor, tensors, config = self.inputs()
        with torch.no_grad():
            actor.bias.copy_(torch.linspace(-2., 3., 28))
        for objective in ('soft_target_ce', 'expected_ce'):
            config['objective'] = objective
            result = diagnosis.compute_diagnostics(actor, tensors, config, batch_size=2)
            expected = float(_objective(actor, tensors['features'], tensors['native_pairs'],
                                        tensors['costs'], DocumentRouterConfig(**config)).detach())
            reported = result['objective_components']['candidate']
            self.assertAlmostEqual(reported['configured_objective'], expected, places=12)
            logp = actor.log_probs(tensors['features'], tensors['native_pairs'])
            reference = actor.reference_log_probs(tensors['native_pairs'])
            self.assertAlmostEqual(reported['native_reference_kl'],
                                   float(native_reference_kl(logp, reference).detach()), places=12)

    def test_flat_costs_nonzero_native_index_are_minimizers_not_unique_alternates(self):
        actor, tensors, config = self.inputs()
        tensors['costs'].fill_(1.)
        tensors['native_pairs'][:] = torch.tensor([6, 7])
        result = diagnosis.compute_diagnostics(actor, tensors, config, batch_size=2)
        counts = result['counts']
        self.assertEqual(counts['target_native_argmax'], 0)
        self.assertEqual(counts['native_is_uniform_cost_minimizer'], 3)
        self.assertEqual(counts['native_within_best_cost_tolerance'], 3)
        self.assertEqual(counts['unique_target_argmax_differs_native'], 0)
        self.assertEqual(counts['target_best_second_cost_near_ties'], 3)
        self.assertEqual(counts['native_best_alternate_exact_ties'], 3)
        self.assertEqual(counts['candidate_changed_argmax_from_update_zero'], 0)
        self.assertEqual(counts['candidate_near_ties'], 0)
        self.assertAlmostEqual(result['distributions']['target_native_mass']['mean'], 1 / 28)

    def test_near_ties_are_explicit_cpu_math_not_executed_actions(self):
        actor, tensors, config = self.inputs()
        with torch.no_grad(): actor.bias[1] = 3 * math.log(8)
        result = diagnosis.compute_diagnostics(actor, tensors, config, batch_size=1, near_tie_tolerance=1e-5)
        self.assertEqual(result['counts']['candidate_near_ties'], 3)
        self.assertEqual(result['counts']['target_best_second_cost_near_ties'], 2)
        self.assertIn('SUPERVISION', result['label'])

    def test_chunk_sizes_preserve_all_rows_and_forbidden_parameters(self):
        actor, tensors, config = self.inputs()
        results = [diagnosis.compute_diagnostics(actor, tensors, config, batch_size=n) for n in (1, 2, 3, 256)]
        for result in results:
            self.assertEqual(result['counts'], results[0]['counts'])
            self.assertEqual(result['distributions']['target_entropy']['count'], 3)
        for kwargs in ({'batch_size': True}, {'batch_size': 0}, {'batch_size': 4097},
                       {'near_tie_tolerance': math.nan}, {'near_tie_tolerance': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                diagnosis.compute_diagnostics(actor, tensors, config, **kwargs)


class RouterArtifactArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'artifact.pt'

    def test_weights_only_cpu_load_and_size_budget(self):
        torch.save({'x': torch.ones(2)}, self.path)
        with patch('torch.load', wraps=torch.load) as loader:
            value, digest = diagnosis.bounded_torch_load(self.path)
        self.assertEqual(digest, sha(self.path))
        self.assertTrue(torch.equal(value['x'], torch.ones(2)))
        self.assertEqual(loader.call_args.kwargs, {'map_location': 'cpu', 'weights_only': True})
        with patch('torch.load', side_effect=AssertionError('must preflight')):
            with self.assertRaises(ValueError):
                diagnosis.bounded_torch_load(self.path, max_bytes=self.path.stat().st_size - 1)

    def test_compressed_uncompressed_budget_and_malformed_archives_preflight(self):
        for mode in ('compressed', 'uncompressed_budget', 'traversal', 'missing_pickle', 'bad_crc', 'nonzip'):
            with self.subTest(mode=mode):
                if mode == 'nonzip':
                    self.path.write_bytes(b'not a torch zip')
                else:
                    compression = zipfile.ZIP_DEFLATED if mode in ('compressed', 'uncompressed_budget') else zipfile.ZIP_STORED
                    with zipfile.ZipFile(self.path, 'w', compression=compression) as z:
                        z.writestr('archive/data.pkl' if mode != 'missing_pickle' else 'archive/other', b'0')
                        z.writestr('../escape' if mode == 'traversal' else 'archive/data/0', b'0' * 20000)
                    if mode == 'bad_crc':
                        raw = self.path.read_bytes()
                        # Alter stored member bytes without updating its ZIP CRC.
                        self.path.write_bytes(raw.replace(b'0' * 100, b'1' + b'0' * 99, 1))
                with patch('torch.load', side_effect=AssertionError('must preflight')):
                    with self.assertRaises(ValueError):
                        diagnosis.bounded_torch_load(self.path, max_bytes=4096 if mode == 'uncompressed_budget' else 100000)

    def test_malformed_storage_is_rejected_not_executed(self):
        torch.save({'x': torch.ones(2)}, self.path)
        with zipfile.ZipFile(self.path) as z:
            contents = {name: z.read(name) for name in z.namelist()}
        storage = next(name for name in contents if '/data/' in name)
        contents[storage] = b''
        with zipfile.ZipFile(self.path, 'w') as z:
            for name, raw in contents.items(): z.writestr(name, raw)
        with self.assertRaises((ValueError, RuntimeError)):
            diagnosis.bounded_torch_load(self.path)


if __name__ == '__main__':
    unittest.main()
