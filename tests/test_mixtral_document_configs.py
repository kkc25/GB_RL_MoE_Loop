"""Full-document templates are explicit, bounded and separate from old caches."""
import copy
import json
from pathlib import Path
import unittest

from mixtral_lab.document_protocol import SCHEMA, validate_document_config
from mixtral_lab.fit_protocol import experiment_contract
from mixtral_lab.protocol import validate_config
from mixtral_lab.recovery_protocol import validate_recovery_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs' / 'mixtral'
ROLES = ('train', 'development', 'selection', 'calibration')


class DocumentPresetTests(unittest.TestCase):
    def load(self, name):
        return json.loads((CONFIGS / name).read_text())

    def test_six_presets_validate_without_gpu_model_or_data(self):
        paths = sorted(CONFIGS.glob('document_*.json'))
        self.assertEqual(len(paths), 6)
        destinations = set()
        for path in paths:
            with self.subTest(path=path.name):
                c = validate_document_config(json.loads(path.read_text()))
                self.assertEqual(c['schema'], SCHEMA)
                self.assertNotIn(c['output_dir'], destinations)
                destinations.add(c['output_dir'])
                self.assertEqual(set(c['data_files']), set(ROLES))
                self.assertEqual(set(c['sequences']), set(ROLES))
                self.assertEqual(len(set(c['data_files'].values())), 4)
                self.assertTrue(all('/document-v1/' in x for x in c['data_files'].values()))
                for field in ('recovery', 'fit', 'training', 'oracle', 'second_round', 'context_sampling', 'max_train_contexts'):
                    self.assertNotIn(field, c)
                t = c['document']['training']
                self.assertEqual(t['seed'], c['seed'])
                self.assertEqual(t['max_cache_bytes'], 1024**3)
                self.assertEqual(t['max_logit_elements'], 16_777_216)
                self.assertEqual((t['parity_atol'], t['parity_rtol']), (0., 0.))
                self.assertEqual(c['document']['development_min_gain'], .001)
                self.assertEqual(c['document']['selection_min_gain'], .001)

    def test_smoke_is_not_pilot_or_claim_of_sufficient_budget(self):
        for kind in ('adapter', 'router'):
            s = self.load(f'document_{kind}_smoke.json')
            p = self.load(f'document_{kind}_pilot.json')
            self.assertEqual(s['sequences'], dict.fromkeys(ROLES, 2))
            self.assertEqual(p['sequences'], dict(train=24, development=8, selection=16, calibration=8))
            self.assertEqual((s['warmup'], s['repeats'], s['measurements']), (1,1,3))
            self.assertEqual((p['warmup'], p['repeats'], p['measurements']), (3,3,32))
            st, pt = s['document']['training'], p['document']['training']
            self.assertEqual((st['updates'], st['eval_interval']), (4,2))
            self.assertIsNone(st['adaptive'])
            self.assertEqual((pt['updates'], pt['eval_interval']), (256,16))
            self.assertEqual(pt['min_updates'], 64)
            self.assertTrue(pt['adaptive']['enabled'])
            self.assertEqual(pt['adaptive']['eval_interval'], pt['eval_interval'])
            self.assertLessEqual(pt['adaptive']['min_lr'], pt['learning_rate'])
            self.assertLessEqual(pt['learning_rate'], pt['adaptive']['max_lr'])
            self.assertEqual(pt['adaptive']['max_growths'], 1)
            self.assertEqual(p['sequences']['train'] + p['sequences']['development'], 32)

    def test_router_label_work_is_whole_document_calls_not_singletons(self):
        for path in CONFIGS.glob('document_router_*.json'):
            c=json.loads(path.read_text()); t=c['document']['training']
            self.assertEqual(t['max_table_calls'], 29*c['sequences']['train'])
            self.assertGreaterEqual(t['max_train_tokens'], c['sequences']['train']*(c['sequence_length']-1))
            self.assertLessEqual(t['positions_per_update'], t['max_train_tokens'])
            self.assertEqual(t['epsilon'], .8)
            self.assertEqual(t['reference_kl_weight'], .01)
            self.assertNotIn('positions_per_document', t)

    def test_adapter_is_small_native_anchored_residual_with_rotating_masks(self):
        for path in CONFIGS.glob('document_adapter_*.json'):
            c=json.loads(path.read_text()); t=c['document']['training']
            self.assertEqual(t['rank'], 8)
            self.assertEqual(t['residual_scale'], .1)
            self.assertEqual(t['relative_correction_l2'], 0.)
            self.assertLessEqual(t['positions_per_document'], c['sequence_length']-1)
            self.assertNotIn('positions_per_update', t)
            self.assertNotIn('max_train_contexts', t)

    def test_two_ablations_change_only_one_scientific_field(self):
        for ordinary, ablation, field, baseline, alternative in (
            ('document_router_pilot.json','document_router_expected_ce_pilot.json','objective','soft_target_ce','expected_ce'),
            ('document_adapter_pilot.json','document_adapter_kl0_pilot.json','native_kl_beta',.1,0.),
        ):
            a,b=self.load(ordinary),self.load(ablation)
            self.assertEqual(a['document']['training'][field],baseline)
            self.assertEqual(b['document']['training'][field],alternative)
            for c in (a,b):
                c.pop('output_dir'); c['document']['training'].pop(field)
            self.assertEqual(a,b)

    def test_stage_copies_change_output_only_and_seeds_are_manual(self):
        for kind in ('adapter','router'):
            for seed in (17,29,43):
                c=self.load(f'document_{kind}_pilot.json')
                c['seed']=c['document']['training']['seed']=seed
                validate_document_config(c)
                for stage in ('evaluate','profile'):
                    other=copy.deepcopy(c);other['output_dir']+='-'+stage
                    validate_document_config(other)
                    self.assertEqual(experiment_contract(c),experiment_contract(other))
                changed=copy.deepcopy(c);changed['document']['selection_min_gain']=.002
                self.assertNotEqual(experiment_contract(c),experiment_contract(changed))

    def test_legacy_templates_and_stale_sampling_fields_rejected(self):
        for name in ('fit_adapter_64.json','recovery_adapter_smoke.json'):
            with self.assertRaises(ValueError): validate_document_config(self.load(name))
        c=self.load('document_adapter_smoke.json')
        with self.assertRaises(ValueError): validate_config(c)
        with self.assertRaises(ValueError): validate_recovery_config(c)
        for field in ('recovery','fit','training','context_sampling','max_train_contexts'):
            bad=copy.deepcopy(c);bad[field]={}
            with self.assertRaises(ValueError):validate_document_config(bad)

    def test_role_alias_and_newtest_not_allowed(self):
        c=self.load('document_adapter_smoke.json')
        for key in ('data_files','sequences'):
            bad=copy.deepcopy(c);bad[key]['newtest']=bad[key].pop('development')
            with self.assertRaises(ValueError):validate_document_config(bad)
        bad=copy.deepcopy(c);bad['data_files']['development']=bad['data_files']['train']
        with self.assertRaises(ValueError):validate_document_config(bad)
        bad=copy.deepcopy(c);bad['data_files']['development']='/data/newtest-renamed-dev.jsonl'
        with self.assertRaises(ValueError):validate_document_config(bad)


if __name__ == '__main__':
    unittest.main()
