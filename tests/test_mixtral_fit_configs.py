"""Shipped fit-first presets remain bounded, explicit and CPU-validatable."""
import copy
import json
from pathlib import Path
import unittest

from mixtral_lab.fit_protocol import experiment_contract, validate_fit_config
from mixtral_lab.protocol import validate_config
from mixtral_lab.adaptive_fit import AdaptiveFitController


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs' / 'mixtral'


class FitPresetTests(unittest.TestCase):
    def load(self, name):
        return json.loads((CONFIGS / name).read_text())

    def test_all_shipped_fit_presets_validate_without_model_or_data(self):
        paths = sorted(CONFIGS.glob('fit_*.json'))
        self.assertEqual(len(paths), 5)
        destinations = set()
        for path in paths:
            with self.subTest(path=path.name):
                config = validate_fit_config(json.loads(path.read_text()))
                self.assertNotIn(config['output_dir'], destinations)
                destinations.add(config['output_dir'])
                self.assertEqual(config['seed'], 17)
                self.assertEqual(config['fit']['training']['seed'], 17)
                self.assertEqual(config['fit']['training']['max_train_contexts'], config['max_train_contexts'])
                self.assertEqual(config['fit']['training']['fit_gain_threshold'], .01)
                self.assertEqual(config['fit']['generalization_min_gain'], .001)
                self.assertNotEqual(config['fit']['generalization_min_gain'], config['oracle']['practical_gain'])
                self.assertEqual(set(config['data_files']), {'train', 'selection', 'calibration'})
                self.assertFalse(config['second_round']['enabled'])
                if '_smoke' in path.stem:
                    self.assertEqual((config['warmup'], config['repeats'], config['measurements']), (1, 1, 3))
                    self.assertFalse(config['fit']['training']['adaptive']['enabled'])
                    self.assertIsNone(json.loads(path.read_text())['fit']['training']['adaptive'])
                else:
                    self.assertEqual((config['warmup'], config['repeats'], config['measurements']), (3, 3, 32))
                    self.assertGreaterEqual(config['measurements'], config['sequences']['selection'])
                    adaptive = config['fit']['training']['adaptive']
                    self.assertTrue(adaptive['enabled'])
                    self.assertEqual(adaptive['eval_interval'], 16)
                    self.assertEqual(adaptive['patience'], 3)
                    self.assertEqual(adaptive['min_updates'], 64)
                    self.assertLessEqual(adaptive['min_updates'], config['fit']['training']['updates'])
                    self.assertEqual(adaptive['min_lr'], 1e-5)
                    self.assertEqual(adaptive['max_lr'], .02 if config['fit']['kind'] == 'router' else .01)
                    self.assertLessEqual(adaptive['min_lr'], config['fit']['training']['learning_rate'])
                    self.assertGreaterEqual(adaptive['max_lr'], config['fit']['training']['learning_rate'])
                    if config['fit']['kind'] == 'adapter':
                        self.assertEqual(adaptive['eval_interval'], config['fit']['training']['eval_interval'])
                if config['fit']['kind'] == 'router':
                    self.assertEqual(config['fit']['training']['epsilon'], .8)
                    self.assertEqual(config['fit']['training']['max_score_calls'], 30 * config['max_train_contexts'])
                else:
                    self.assertEqual(config['fit']['training']['rank'], 8)
                    self.assertEqual(config['fit']['training']['residual_scale'], 1.)

    def test_smoke_is_separate_from_bounded_fit_opportunity(self):
        for kind in ('router', 'adapter'):
            smoke = self.load(f'fit_{kind}_smoke.json')
            fit = self.load(f'fit_{kind}_64.json')
            self.assertEqual((smoke['max_train_contexts'], smoke['fit']['training']['updates']), (8, 4))
            self.assertEqual((fit['max_train_contexts'], fit['fit']['training']['updates']), (64, 256))
            self.assertGreater(fit['sequences']['train'], smoke['sequences']['train'])
            self.assertEqual(fit['fit']['generalization_min_gain'], smoke['fit']['generalization_min_gain'])
        larger = self.load('fit_router_256.json')
        self.assertEqual((larger['max_train_contexts'], larger['fit']['training']['updates']), (256, 256))
        self.assertGreater(larger['sequences']['train'], self.load('fit_router_64.json')['sequences']['train'])

    def test_manual_seeds_and_output_only_stage_copy(self):
        for kind in ('router', 'adapter'):
            for seed in (17, 29, 43):
                with self.subTest(kind=kind, seed=seed):
                    config = self.load(f'fit_{kind}_64.json')
                    config['seed'] = config['training']['seed'] = config['fit']['training']['seed'] = seed
                    validate_fit_config(config)
                    downstream = copy.deepcopy(config)
                    downstream['output_dir'] += '-generalize'
                    validate_fit_config(downstream)
                    self.assertEqual(experiment_contract(config), experiment_contract(downstream))

    def test_default_adaptation_does_not_force_premature_plateau_stop(self):
        # A 64-step minimum is not a target stop. Default reductions may not
        # reach the floor within 256 steps, so the trainer's ceiling still wins.
        for kind in ('router', 'adapter'):
            with self.subTest(kind=kind):
                training = self.load(f'fit_{kind}_64.json')['fit']['training']
                controller = AdaptiveFitController(training['adaptive'], initial_lr=training['learning_rate'])
                for update in range(0, training['updates'] + 1, training['adaptive']['eval_interval']):
                    decision = controller.observe(update, 1., coverage_complete=update > 0)
                    self.assertFalse(decision['stop'])
                self.assertLess(controller.lr, training['learning_rate'])
                self.assertGreaterEqual(controller.lr, training['adaptive']['min_lr'])

    def test_legacy_presets_remain_valid_with_legacy_validator(self):
        for path in sorted(CONFIGS.glob('*.json')):
            if path.name.startswith(('fit_', 'recovery_', 'document_', 'trainable_loop_', 'loop_scale_')):
                continue
            with self.subTest(path=path.name):
                validate_config(json.loads(path.read_text()))


if __name__ == '__main__':
    unittest.main()
