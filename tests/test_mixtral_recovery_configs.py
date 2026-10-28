"""Shipped recovery research settings are bounded, isolated and not GPU jobs."""
import copy
import json
from pathlib import Path
import unittest

from mixtral_lab.recovery_protocol import (
    ROLES, SCHEMA, experiment_contract, validate_recovery_config,
)
from mixtral_lab.protocol import validate_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs' / 'mixtral'


class RecoveryPresetTests(unittest.TestCase):
    def load(self, name):
        return json.loads((CONFIGS / name).read_text())

    def test_eight_presets_validate_without_model_data_or_gpu(self):
        paths = sorted(CONFIGS.glob('recovery_*.json'))
        self.assertEqual(len(paths), 8)
        destinations = set()
        for path in paths:
            with self.subTest(path=path.name):
                original = json.loads(path.read_text())
                c = validate_recovery_config(copy.deepcopy(original))
                self.assertEqual(c['schema'], SCHEMA)
                self.assertNotIn(c['output_dir'], destinations)
                destinations.add(c['output_dir'])
                self.assertEqual(set(c['data_files']), set(ROLES))
                self.assertEqual(set(c['sequences']), set(ROLES))
                self.assertEqual(len(set(c['data_files'].values())), 4)
                self.assertTrue(all('/recovery-v1/' in p for p in c['data_files'].values()))
                self.assertTrue(all('newtest' not in p.lower() for p in c['data_files'].values()))
                for forbidden in ('fit', 'training', 'second_round', 'oracle'):
                    self.assertNotIn(forbidden, c)
                t = c['recovery']['training']
                self.assertEqual(t['max_train_contexts'], c['max_train_contexts'])
                self.assertEqual(t['seed'], c['seed'])
                self.assertEqual(c['seed'], 17)
                self.assertIsNone(original['recovery']['training']['adaptive'])
                self.assertEqual(c['recovery']['development_min_gain'], .001)
                self.assertEqual(c['recovery']['selection_min_gain'], .001)
                self.assertEqual(c['recovery']['bridge_tolerances'], {
                    'hidden_max_abs': 1e-6, 'feature_max_abs': 1e-6, 'ce_max_abs': 1e-4})
                if c['recovery']['kind'] == 'router':
                    self.assertEqual(t['max_score_calls'], 30 * c['max_train_contexts'])
                    self.assertEqual(t['epsilon'], .8)
                    self.assertGreater(t['epsilon'], 0.)
                    self.assertLess(t['epsilon'], 1.)

    def test_smoke_and_diagnostic_and_broader_coverage_are_separate(self):
        for kind in ('router', 'adapter'):
            suffix = 'soft_target_ce' if kind == 'router' else 'anchored'
            smoke = self.load(f'recovery_{kind}_smoke.json')
            diagnostic = self.load(f'recovery_{kind}_{suffix}_64.json')
            broader = self.load(f'recovery_{kind}_{suffix}_512.json')
            for c, contexts, updates in ((smoke, 8, 4), (diagnostic, 64, 256), (broader, 512, 256)):
                self.assertEqual((c['max_train_contexts'], c['recovery']['training']['updates']), (contexts, updates))
                self.assertLessEqual(contexts, c['sequences']['train'] * (c['sequence_length'] - 1))
            self.assertEqual(smoke['sequences'], dict.fromkeys(ROLES, 2))
            self.assertEqual((smoke['warmup'], smoke['repeats'], smoke['measurements']), (1, 1, 3))
            self.assertEqual(diagnostic['sequences'], dict(train=8, development=8, selection=16, calibration=8))
            self.assertEqual(broader['sequences'], dict(train=24, development=8, selection=16, calibration=8))
            # The documented physical partition needs only 32 of the historical
            # source's 64 train documents, not 64 train + additional dev documents.
            self.assertEqual(broader['sequences']['train'] + broader['sequences']['development'], 32)
            for c in (diagnostic, broader):
                self.assertEqual((c['warmup'], c['repeats'], c['measurements']), (3, 3, 32))
                self.assertGreaterEqual(c['measurements'], c['sequences']['selection'])
                self.assertEqual(c['recovery']['training']['eval_interval'], 16)
                self.assertEqual(c['recovery']['training']['development_patience'], 4)

    def test_router_objective_pair_is_one_factor_comparison(self):
        expected = self.load('recovery_router_expected_ce_64.json')
        soft = self.load('recovery_router_soft_target_ce_64.json')
        self.assertEqual(expected['recovery']['training']['objective'], 'expected_ce')
        self.assertEqual(soft['recovery']['training']['objective'], 'soft_target_ce')
        for c in (expected, soft):
            c.pop('output_dir')
            c['recovery']['training'].pop('objective')
        self.assertEqual(expected, soft)

    def test_optional_native_prior_ablation_changes_only_coefficient(self):
        anchored = self.load('recovery_router_soft_target_ce_64.json')
        zero = self.load('recovery_router_soft_target_ce_64_kl0.json')
        self.assertEqual(anchored['recovery']['training']['reference_kl_weight'], .01)
        self.assertEqual(zero['recovery']['training']['reference_kl_weight'], 0.)
        for c in (anchored, zero):
            c.pop('output_dir')
            c['recovery']['training'].pop('reference_kl_weight')
        self.assertEqual(anchored, zero)

    def test_adapter_presets_are_native_anchored_not_extra_expert_rounds(self):
        for path in CONFIGS.glob('recovery_adapter_*.json'):
            t = json.loads(path.read_text())['recovery']['training']
            self.assertEqual(t['rank'], 8)
            self.assertEqual(t['residual_scale'], .1)
            self.assertEqual(t['native_kl_beta'], .1)
            self.assertEqual(t['relative_correction_l2'], 0.)
            self.assertEqual(t['probe_chunk_contexts'], 8)

    def test_manual_seeds_and_output_only_stage_copies(self):
        for name in ('recovery_router_soft_target_ce_64.json', 'recovery_adapter_anchored_64.json'):
            for seed in (17, 29, 43):
                with self.subTest(name=name, seed=seed):
                    c = self.load(name)
                    c['seed'] = c['recovery']['training']['seed'] = seed
                    validate_recovery_config(c)
                    downstream = copy.deepcopy(c)
                    downstream['output_dir'] += '-evaluate'
                    validate_recovery_config(downstream)
                    self.assertEqual(experiment_contract(c), experiment_contract(downstream))
                    downstream['recovery']['selection_min_gain'] = .002
                    self.assertNotEqual(experiment_contract(c), experiment_contract(downstream))

    def test_bridge_tolerances_must_be_explicit_finite_nonnegative(self):
        for key in ('hidden_max_abs', 'feature_max_abs', 'ce_max_abs'):
            for invalid in (None, -1., float('nan'), float('inf'), True):
                with self.subTest(key=key, invalid=invalid):
                    c = self.load('recovery_router_smoke.json')
                    if invalid is None:
                        del c['recovery']['bridge_tolerances'][key]
                    else:
                        c['recovery']['bridge_tolerances'][key] = invalid
                    with self.assertRaises(ValueError):
                        validate_recovery_config(c)

    def test_role_aliases_and_newtest_are_rejected(self):
        for key in ('data_files', 'sequences'):
            c = self.load('recovery_router_smoke.json')
            c[key]['newtest'] = c[key].pop('development')
            with self.assertRaises(ValueError):
                validate_recovery_config(c)
        c = self.load('recovery_router_smoke.json')
        c['data_files']['development'] = c['data_files']['train']
        with self.assertRaises(ValueError):
            validate_recovery_config(c)
        c['data_files']['development'] = '/data/newtest-renamed-as-development.jsonl'
        with self.assertRaises(ValueError):
            validate_recovery_config(c)

    def test_legacy_schema_and_recovery_schema_do_not_silently_mix(self):
        with self.assertRaises(ValueError):
            validate_recovery_config(self.load('fit_router_64.json'))
        with self.assertRaises(ValueError):
            validate_config(self.load('recovery_router_smoke.json'))
        for section in ('fit', 'training', 'second_round', 'oracle'):
            c = self.load('recovery_router_smoke.json')
            c[section] = {}
            with self.assertRaises(ValueError):
                validate_recovery_config(c)


if __name__ == '__main__':
    unittest.main()
