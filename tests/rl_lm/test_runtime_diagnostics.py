"""CPU correctness/protocol tests; not evidence of GPU speed or full-size parity."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments import rl_lm_diagnose_runtime as diagnostic
from model import LoopMoE
from model.top2 import top2_dispatch_identity
from rl_lm import inference
from rl_lm.readout import production_logits
from rl_lm.runtime_diagnostics import (CachedBF16Head, Variant, runtime_scope, parity_metrics,
                                       parse_chunks, fresh_output, preserve_precision)
from tests.rl_lm.tiny_lm import build_tiny_model, TINY_DIM, TINY_VOCAB


class RuntimeDiagnosticTests(unittest.TestCase):
    def test_all_layer_native_vs_prefix_and_exception_restoration(self):
        model = build_tiny_model()
        model.blocks[0].executor.dispatch_backend = 'merged'
        before = top2_dispatch_identity(model)
        original = inference.production_logits
        cache = CachedBF16Head(model)
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with runtime_scope(model, Variant('test', 'online', 'merged', cached_head=True), cache) as (_, verify):
                verify()
                self.assertEqual(model.blocks[-1].executor.dispatch_backend, 'legacy')
                self.assertIs(inference.production_logits, cache)
                raise RuntimeError('test failure')
        self.assertEqual(top2_dispatch_identity(model), before)
        self.assertIs(inference.production_logits, original)
        with runtime_scope(model, Variant('all', 'native', 'merged')) as (identity, verify):
            verify()
            self.assertTrue(all(e['backend'] == 'merged' for e in identity['dispatch']['executors']))
            model.blocks[-1].executor.dispatch_backend = 'legacy'
            with self.assertRaises(RuntimeError):
                verify()
        self.assertEqual(top2_dispatch_identity(model), before)

    def test_cached_head_exact_same_chunk_cpu_bf16_and_state_unchanged(self):
        model = build_tiny_model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        h = torch.randn(13, TINY_DIM)
        cache = CachedBF16Head(model)
        expected = production_logits(model, h, chunk=5, compute_dtype='bfloat16')
        actual = cache(model, h, chunk=5)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(cache.bytes, TINY_VOCAB*TINY_DIM*2)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        self.assertEqual(cache(model, h[:0]).shape, (0, TINY_VOCAB))
        with torch.no_grad():
            model.embedding.weight.add_(1)
        with self.assertRaises(RuntimeError):
            cache(model, h)

    def test_cache_rejects_replacement_and_training(self):
        model = build_tiny_model()
        cache = CachedBF16Head(model)
        model.embedding.weight = torch.nn.Parameter(model.embedding.weight.detach().clone(), requires_grad=False)
        with self.assertRaises(RuntimeError):
            cache(model, torch.zeros(1, TINY_DIM))
        model.embedding.weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            CachedBF16Head(model)

    def test_cancellation_and_actions_not_hidden_by_nll(self):
        a = SimpleNamespace(logits=torch.tensor([[[2., 0.], [0., 2.]]]),
                            actions=torch.tensor([[1, 1]]), rounds=torch.ones(1, 2))
        b = SimpleNamespace(logits=a.logits.flip(1), actions=a.actions, rounds=a.rounds)
        result = parity_metrics(a, b, torch.zeros(1, 2, dtype=torch.long), nll_tolerance=1e-4)
        self.assertAlmostEqual(result['delta_nll'], 0, places=6)
        self.assertGreater(result['ce_mean_abs'], 1)
        self.assertFalse(result['eligible'])
        b.logits = a.logits
        b.actions = torch.tensor([[1, 2]])
        self.assertFalse(parity_metrics(a, b, torch.zeros(1, 2, dtype=torch.long),
                                       nll_tolerance=0, require_actions=True)['eligible'])
        with self.assertRaises(ValueError):
            parity_metrics(a, b, torch.zeros(1, 2, dtype=torch.long), nll_tolerance=True)

    def test_invalid_arguments_and_protected_outputs(self):
        for value in ('0', '64,064', '-1', 'nan', '64,64', ''):
            with self.assertRaises(ValueError):
                parse_chunks(value)
        args = diagnostic.parser().parse_args(['--config', 'c', '--dense-config', 'd', '--dense-checkpoint', 'w', '--output-dir', 'o'])
        self.assertEqual(diagnostic.validate_args(args), ([1], ['64', 'full']))
        args.sequences = True
        with self.assertRaises(ValueError):
            diagnostic.validate_args(args)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            fresh_output(root)
            (root/'evidence').write_text('keep')
            with self.assertRaises(FileExistsError):
                fresh_output(root)
            self.assertEqual((root/'evidence').read_text(), 'keep')

    def test_real_tiny_online_and_native_comparisons(self):
        model = build_tiny_model()
        ids = torch.randint(TINY_VOCAB, (1, 7))
        batch = (ids, torch.zeros_like(ids), ids)
        policies = {'canonical_r1': inference.CanonicalTeacherPolicy()}
        candidate = Variant('test', 'online', 'merged', 'full', False, 'canonical_r1', 1)
        result = diagnostic.validate_variant(candidate, model, model, policies, [batch], 'float32', 4, None, 1e-5)
        self.assertTrue(result['eligible'])
        self.assertEqual(result['reference']['prefix'], 'merged')
        self.assertEqual(result['groups'][0]['action_differences'], 0)
        native = diagnostic.validate_variant(Variant('native', 'native', 'merged'), model, model, policies,
                                              [batch], 'float32', 4, None, 1e-5)
        self.assertTrue(native['eligible'])
        self.assertIsNone(native['groups'][0]['action_differences'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for cached-head GPU parity')
    def test_cached_head_cuda_same_chunk(self):
        model = build_tiny_model().cuda()
        h = torch.randn(17, TINY_DIM, device='cuda')
        cache = CachedBF16Head(model)
        expected = production_logits(model, h, chunk=8, compute_dtype='bfloat16')
        self.assertTrue(torch.equal(expected, cache(model, h, chunk=8)))

    def test_existing_cli_output_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(diagnostic, 'resolve_config') as load:
            with self.assertRaises(FileExistsError):
                diagnostic.main(['--config', 'c', '--dense-config', 'd', '--dense-checkpoint', 'w',
                                 '--output-dir', tmp])
            load.assert_not_called()

    def test_mocked_cli_full_wiring_on_cpu(self):
        model = build_tiny_model()
        dense_config = replace(model.config, moe_layers=(), adaptive_layers=(), experts_per_layer=1, top_k=1)
        dense = LoopMoE(dense_config).eval()
        config_values = {'runtime.device': 'cuda', 'runtime.allow_tf32': False,
                         'environment.frozen_compute_dtype': 'bfloat16', 'environment.expert_chunk': 4,
                         'environment.max_calls': 1, 'data.selection_split': 'selection',
                         'data.confirmation_split': 'newtest', 'data.corpus_dir': 'fixture',
                         'backbone.checkpoint_path': 'fixture.pt'}
        config = SimpleNamespace(get=config_values.__getitem__, to_dict=lambda: config_values)
        row = {'input_ids': np.arange(4, dtype=np.int64), 'segment_ids': np.zeros(4, dtype=np.int64),
               'labels': np.arange(4, dtype=np.int64)}
        def fake_measure(fn, **kwargs):
            if kwargs.get('before_call'):
                kwargs['before_call']()
            fn()
            return {'raw': [{'event_ms': [1.], 'wall_ms': [1.]}], 'cuda_event': {'mean_ms': 1.}}
        real_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if args and isinstance(args[0], torch.device) and args[0].type == 'cuda':
                args = (torch.device('cpu'),) + args[1:]
            return real_to(tensor, *args, **kwargs)
        before = top2_dispatch_identity(model)
        original = inference.production_logits
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(diagnostic, 'resolve_config', return_value=config), \
             patch.object(diagnostic, 'verified_evaluation_scope', return_value=({'split': 'selection'}, [row, row], {})), \
             patch.object(diagnostic, 'load_backbone', return_value=(model, {})), \
             patch.object(diagnostic.LMExperimentConfig, 'load', return_value=SimpleNamespace(model=dense_config)), \
             patch.object(diagnostic.torch, 'load', return_value={'model': dense.state_dict()}), \
             patch.object(diagnostic.torch.cuda, 'set_device'), \
             patch.object(diagnostic, 'check_exclusive'), \
             patch.object(diagnostic, 'gpu_metadata', return_value={'test': 'cpu mock'}), \
             patch.object(diagnostic, 'source_identity', return_value={'test': True}), \
             patch.object(diagnostic, 'sha256_file', return_value='fixture'), \
             patch.object(diagnostic, 'measure', side_effect=fake_measure), \
             patch.object(torch.Tensor, 'to', cpu_to):
            output = Path(tmp)/'diagnostic'
            diagnostic.main(['--config', 'c', '--dense-config', 'd', '--dense-checkpoint', 'w',
                             '--output-dir', str(output), '--cached-bf16-head', '--head-chunks', '64',
                             '--warmup', '0', '--measurements', '1'])
            report = json.loads((output/'diagnostic_report.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['variant_count'], 7)
            workload = report['workloads']['batch1_s4']
            self.assertEqual(len(workload['end_to_end']), 7)
            self.assertIn('head64_cached_bf16', workload['components'])
            self.assertEqual(len(workload['combined_runtime_verification']), 4)
        self.assertIs(inference.production_logits, original)
        self.assertEqual(top2_dispatch_identity(model), before)


if __name__ == '__main__':
    unittest.main()
