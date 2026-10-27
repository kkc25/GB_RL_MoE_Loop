"""CPU-only validation/protocol regressions; no performance claims."""
import unittest
from unittest.mock import patch

import torch

from rl_lm.config import RLConfig
from rl_lm.single_config import SingleAdaptConfig, SingleRLConfig
from experiments.rl_lm_profile import measure_variants, variant_schedule, validate_profile_protocol, profile_ledger_summary


class ValidationTests(unittest.TestCase):
    def configs(self):
        return [(RLConfig, 'pair_r1'), (RLConfig, 'pair_r2'),
                (SingleRLConfig, 'single_rloo'), (SingleAdaptConfig, 'single_adapt')]

    def test_shipped_configs(self):
        for cls, name in self.configs():
            cls.load(f'configs/rl_lm/{name}.json').validate()

    def test_strict_numeric_types_and_domains(self):
        for cls, name in self.configs():
            original = cls.load(f'configs/rl_lm/{name}.json')
            section = 'adaptation' if name == 'single_adapt' else 'optimization'
            cases = [('data.train_states', 0), ('data.sampling_seed', -1), ('environment.expert_chunk', -1),
                     ('environment.head_chunk', 0), ('logging.interval_updates', 0),
                     (f'{section}.learning_rate', True), (f'{section}.learning_rate', float('nan')),
                     (f'{section}.epsilon', float('inf')), (f'{section}.epsilon', 0),
                     (f'{section}.weight_decay', -1), (f'{section}.batch_contexts', 0),
                     (f'{section}.betas', [True, .9]), (f'{section}.betas', [.9, float('nan')]),
                     (f'{section}.betas', [.9]), (f'{section}.betas', [.9, 1]),
                     (f'{section}.warmup_updates', -1), (f'{section}.warmup_updates', 999999)]
            for path, value in cases:
                with self.subTest(config=name, path=path, value=value), self.assertRaises(ValueError):
                    data = original.to_dict()
                    group, key = path.split('.')
                    data[group][key] = value
                    cls(data).validate()

    def test_rloo_semantics(self):
        for cls, name in self.configs()[:3]:
            for group, key, value in [('optimization', 'algorithm', 'adam'),
                                      ('optimization', 'final_lr_ratio', 1.1),
                                      ('optimization', 'max_retries_per_update', -1),
                                      ('evaluation', 'interval_updates', 0)]:
                with self.subTest(config=name, key=key), self.assertRaises(ValueError):
                    data = cls.load(f'configs/rl_lm/{name}.json').to_dict()
                    data[group][key] = value
                    cls(data).validate()

    def test_load_backbone_rejects_before_allocation(self):
        from rl_lm.cli import load_backbone
        config = RLConfig.load("configs/rl_lm/pair_r1.json")
        config.data["environment"]["head_chunk"] = 0
        with patch("rl_lm.cli.LoopMoE") as model, patch("rl_lm.cli.LMExperimentConfig.load") as load:
            with self.assertRaises(ValueError):
                load_backbone(config, "cpu")
            model.assert_not_called()
            load.assert_not_called()

    def test_revalidate_mutated_schema(self):
        config = RLConfig.load('configs/rl_lm/pair_r1.json')
        config.data['optimization']['updates'] = True
        with self.assertRaises(ValueError):
            config.validate()


class ProfileProtocolTests(unittest.TestCase):
    def test_ledger_peaks_include_earlier_windows_and_work_is_unknown(self):
        from rl_lm.ledger import RLLedger
        ledger = RLLedger(peak_allocated_bytes=2, peak_reserved_bytes=3, gpu_hours=.25)
        layers = {"early": {"a": {"peak_allocated_bytes": 90, "peak_reserved_bytes": 100}},
                  "late": {"b": {"peak_allocated_bytes": 10, "peak_reserved_bytes": 80}}}
        result = profile_ledger_summary(ledger, layers)
        self.assertEqual(result["peak_allocated_bytes"], 90)
        self.assertEqual(result["peak_reserved_bytes"], 100)
        self.assertEqual(result["gpu_hours"], .25)
        self.assertIsNone(result["physical_expert_rows"])
        self.assertIsNone(result["physical_head_rows"])
        self.assertIn("physical_expert_rows", result["uninstrumented_fields"])
        self.assertEqual(result["work_counters_status"], "not_instrumented_in_timed_region")
        self.assertEqual(ledger.physical_expert_rows, 0)  # no training ledger mutation
        self.assertIsNone(profile_ledger_summary(ledger, {})["peak_allocated_bytes"])

    def test_schedule_rotates(self):
        self.assertEqual(variant_schedule(['a', 'b', 'c'], 4),
                         [['a', 'b', 'c'], ['b', 'c', 'a'], ['c', 'a', 'b'], ['a', 'b', 'c']])

    def test_invalid_protocol(self):
        for field, value in [('warmup', -1), ('repeats', 0), ('measurements', False), ('sequences', 0)]:
            args = dict(warmup=0, repeats=1, measurements=1, sequences=8)
            args[field] = value
            with self.assertRaises(ValueError):
                validate_profile_protocol(**args)
        with self.assertRaises(ValueError):
            validate_profile_protocol(warmup=0, repeats=1, measurements=1, sequences=9, batch8=True)

    def test_runner_no_grad_same_inputs_and_aggregation(self):
        calls, windows = [], []
        actor = torch.nn.Linear(1, 1).eval()
        def build(name):
            def run(i):
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(actor(torch.ones(1, 1)).requires_grad)
                calls.append((name, i))
            return run
        def fake_measure(fn, **kwargs):
            start = len(calls)
            for _ in range(kwargs['measurements']):
                fn()
            windows.append(calls[start:])
            return dict(raw=[{'event_ms': [1., 3., 2.], 'wall_ms': [2., 4., 3.]}],
                        tokens_per_call=6, measurements_per_repeat=3,
                        allocated_bytes=10, reserved_bytes=20, peak_allocated_bytes=len(windows), peak_reserved_bytes=30)
        results = measure_variants([('a', build('a')), ('b', build('b'))], groups=2,
                                   warmup=2, repeats=2, measurements=3, tokens=6, measure_fn=fake_measure)
        self.assertEqual([w[0][0] for w in windows], ['a', 'b', 'b', 'a'])
        for w in windows:
            self.assertEqual([i for _, i in w], [0, 1, 0])
        self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(results['a']['peak_allocated_bytes'], 4)
        self.assertEqual(results['a']['cuda_event']['mean_ms'], 2.)
        self.assertEqual(results['a']['cuda_event']['tokens_per_second'], 3000.)
        self.assertEqual(results['a']['repeat_order_positions'], [0, 1])
        self.assertEqual(len(results['a']['raw']), 2)


if __name__ == '__main__':
    unittest.main()
