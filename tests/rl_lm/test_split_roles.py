"""Data-role isolation must hold before model allocation and all later stages."""
import itertools
import unittest
from unittest.mock import patch

from rl_lm.config import RLConfig
from rl_lm.single_config import SingleAdaptConfig, SingleRLConfig
from rl_lm.validation import validate_split_roles


CONFIGS = ((RLConfig, 'pair_r1'), (RLConfig, 'pair_r2'),
           (SingleAdaptConfig, 'single_adapt'), (SingleRLConfig, 'single_rloo'))
ROLES = ('train', 'selection', 'calibration', 'confirmation')


class SplitRoleTests(unittest.TestCase):
    def test_all_shipped_configs_and_distinct_custom_names(self):
        for cls, name in CONFIGS:
            c = cls.load(f'configs/rl_lm/{name}.json')
            c.validate()
            c.data['data'].update({f'{role}_split': f'{role}_custom' for role in ROLES})
            c.validate()
            self.assertEqual(validate_split_roles(c)['confirmation'], 'confirmation_custom')

    def test_every_role_alias_is_rejected(self):
        for cls, name in CONFIGS:
            for left, right in itertools.combinations(ROLES, 2):
                with self.subTest(name=name, left=left, right=right):
                    c = cls.load(f'configs/rl_lm/{name}.json')
                    c.data['data'][f'{left}_split'] = c.data['data'][f'{right}_split']
                    with self.assertRaisesRegex(ValueError, 'distinct'):
                        c.validate()

    def test_reserved_holdout_cannot_drive_selection_or_calibration(self):
        for cls, name in CONFIGS:
            for role in ('train', 'selection', 'calibration'):
                for value in ('newtest', 'NEWTEST'):
                    with self.subTest(name=name, role=role, value=value):
                        c = cls.load(f'configs/rl_lm/{name}.json')
                        c.data['data']['confirmation_split'] = 'final_holdout'
                        c.data['data'][f'{role}_split'] = value
                        with self.assertRaisesRegex(ValueError, 'confirmation'):
                            c.validate()

    def test_leaf_names_and_case_aliases(self):
        for value in ('', ' ', ' selection', 'selection ', '.', '..', 'x/../newtest', 'x\\newtest', 'x\0'):
            c = RLConfig.load('configs/rl_lm/pair_r1.json')
            c.data['data']['selection_split'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                c.validate()
        c = RLConfig.load('configs/rl_lm/pair_r1.json')
        c.data['data']['selection_split'] = c.data['data']['train_split'].upper()
        with self.assertRaisesRegex(ValueError, 'distinct'):
            c.validate()

    def test_streaming_head_window_limit_fails_before_allocation(self):
        from rl_lm.cli import load_backbone
        for cls, name in CONFIGS:
            c = cls.load(f'configs/rl_lm/{name}.json')
            c.data['environment']['head_chunk'] = 8193
            with self.subTest(name=name), patch('rl_lm.cli.LMExperimentConfig.load') as config_load:
                with self.assertRaisesRegex(ValueError, 'head_chunk'):
                    load_backbone(c, 'cpu')
                config_load.assert_not_called()

    def test_holdout_alias_rejected_before_backbone_load(self):
        from rl_lm.cli import load_backbone
        c = RLConfig.load('configs/rl_lm/pair_r1.json')
        c.data['data']['selection_split'] = 'newtest'
        with patch('rl_lm.cli.LMExperimentConfig.load') as config_load, patch('rl_lm.cli.LoopMoE') as model:
            with self.assertRaises(ValueError):
                load_backbone(c, 'cpu')
            config_load.assert_not_called()
            model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
