import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from rl_lm.reporting import stop_metrics, evaluation_readout_identity
from experiments.rl_lm_evaluate import _online_nll, _validate_r1_reference, _validate_actor_reference
from rl_lm.actions import action_table_identity
from experiments.rl_lm_train import OnlineSelectionEvaluator
from experiments.rl_lm_single_train import (
    validate_adaptation_gate, validate_adapted_objective,
    reference_ce_identity, validate_reference_ce_identity,
    ADAPT_OBJECTIVE_VERSION, ADAPT_CHECKPOINT_FORMAT,
)


class ReportingTests(unittest.TestCase):
    def test_evaluation_readout_is_versioned_and_numerically_explicit(self):
        config = {'environment.head_chunk': 64, 'environment.frozen_compute_dtype': 'bfloat16'}
        identity = evaluation_readout_identity(config)
        self.assertEqual(identity['version'], 'streaming_fixed_head_fp32_ce8192_v1')
        self.assertEqual(identity['head_chunk'], 64)
        self.assertEqual(identity['ce_window'], 8192)
        self.assertEqual(identity['compute_dtype'], 'bfloat16')
        self.assertFalse(identity['materializes_full_logits'])

    def test_stop_denominators_are_distinct(self):
        m = stop_metrics(torch.tensor([1, 2, 0, 1]), torch.tensor([False, False, True, True]), max_calls=2)
        self.assertEqual(m['stop_fraction'], .25)
        self.assertEqual(m['stop_fraction_per_decision'], .25)
        self.assertEqual(m['stop_fraction_conditional_r1'], .5)
        self.assertEqual(m['r1_decision_count'], 2)
        r1 = stop_metrics(torch.tensor([1, 2]), torch.tensor([False, False]), max_calls=1)
        self.assertIsNone(r1['stop_fraction_conditional_r1'])
        with self.assertRaises(ValueError):
            stop_metrics(torch.tensor([0]), torch.tensor([False]), max_calls=2)

    def test_online_explicit_r1_overrides_r2_config(self):
        config = {'environment.max_calls': 2, 'environment.expert_chunk': 64, 'environment.head_chunk': 64}
        output = SimpleNamespace(token_ce=torch.tensor([[1.0986123, 0.]]), rounds=torch.ones(1, 2, dtype=torch.long))
        batch = {'input_ids': torch.ones(1, 2, dtype=torch.long), 'segment_ids': None,
                 'labels': torch.tensor([[0, -100]]), 'valid': torch.tensor([[True, False]])}
        with patch('experiments.rl_lm_evaluate.forward_with_last_ffn_policy', return_value=output) as f:
            _online_nll(config, None, None, batch, 'cpu', max_calls=1)
        self.assertEqual(f.call_args.kwargs['max_calls'], 1)

    def test_training_legacy_gain_is_labeled_and_r1_separate(self):
        e = OnlineSelectionEvaluator.__new__(OnlineSelectionEvaluator)
        e.config = {'environment.max_calls': 2}
        e.reference_nll = e.reference_rounds = e.r1_reference_nll = None
        e.r1_actor = e.r1_identity = None
        e.input_ids = torch.zeros(1, 2)
        e.valid = torch.ones(1, 2, dtype=torch.bool)
        e._forward_nll = lambda p, max_calls=None: (5.0, 1.0) if max_calls == 1 else (6.0, 2.0)
        m = e(None, None, 0)
        self.assertEqual(m['evaluation_readout']['version'], 'streaming_fixed_head_fp32_ce8192_v1')
        self.assertEqual(m['gain_nll'], 0.0)
        self.assertEqual(m['legacy_gain_reference']['max_calls'], 2)
        self.assertEqual(m['vs_canonical_r1']['gain_nll'], -1.0)
        self.assertIsNone(m['vs_selected_r1'])

    def test_selected_r1_validates_source_contract_not_only_inference_depth(self):
        import copy
        class Config(dict):
            def to_dict(self):
                return {'policy': {'prior': 'fixed'}}
        config = Config({'policy.feature_schema': 'features', 'stage': 'r1',
                         'environment.max_calls': 1, 'cost.lambda': 0})
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=4))
        payload = {'stage': 'r1', 'resolved_config': {'stage': 'r1',
                   'environment': {'min_calls': 1, 'max_calls': 1},
                   'cost': {'lambda': 0}, 'policy': {'prior': 'fixed'}},
                   'action_contract': action_table_identity(), 'feature_schema': 'features',
                   'feature_dim': 28, 'backbone_identity': {'checkpoint_sha256': 'sha',
                   'config_sha256': 'sha', 'tokenizer_sha256': 'sha'},
                   'data_identity': {'controller_train': {'packed_sha256': 'data'},
                                     'selection': {'packed_sha256': 'data'}}}
        with patch('experiments.rl_lm_evaluate.sha256_file', return_value='sha'), \
             patch('experiments.rl_lm_evaluate.cache_identity', return_value={'packed_sha256': 'data'}):
            _validate_r1_reference(payload, config, model)
            _validate_actor_reference(payload, config, model)
            for field, value in [('stage', 'r2'), ('environment.max_calls', 2), ('cost.lambda', .02)]:
                with self.assertRaises(ValueError):
                    _validate_actor_reference(payload, Config(dict(config, **{field: value})), model)
            for key, value in [('stage', 'r2'), ('feature_dim', 42), ('action_contract', {}),
                               ('backbone_identity', {}), ('data_identity', {})]:
                bad = copy.deepcopy(payload)
                bad[key] = value
                with self.assertRaises(ValueError):
                    _validate_r1_reference(bad, config, model)
                with self.assertRaises(ValueError):
                    _validate_actor_reference(bad, config, model)

    def test_gate_and_objective_bound_to_student(self):
        gate = {'single_rl_allowed': True, 'objective_version': ADAPT_OBJECTIVE_VERSION,
                'student_checkpoint_sha256': 'chosen'}
        validate_adaptation_gate(gate, 'chosen')
        with self.assertRaisesRegex(ValueError, 'different student'):
            validate_adaptation_gate(gate, 'other')
        with self.assertRaises(ValueError):
            validate_adapted_objective({'spec_version': 'RL-LM-Single-v1.0'})
        validate_adapted_objective({'format_version': ADAPT_CHECKPOINT_FORMAT,
                                   'objective_version': ADAPT_OBJECTIVE_VERSION})

    def test_reference_identity_rejects_changed_cache_and_old_metadata(self):
        cache = {'manifest': {'version': 1}, 'h0': np.zeros((2, 3), dtype=np.float32),
                 'targets': np.array([0, 1]), 'episode_ids': ['a', 'b']}
        identity = reference_ce_identity(cache, 'student', 'float32')
        validate_reference_ce_identity({'identity': identity}, identity)
        for key, value in [('targets', np.array([1, 0])), ('episode_ids', ['b', 'a']),
                           ('h0', np.ones((2, 3), dtype=np.float32))]:
            changed = reference_ce_identity(dict(cache, **{key: value}), 'student', 'float32')
            with self.assertRaises(ValueError):
                validate_reference_ce_identity({'identity': identity}, changed)
        with self.assertRaises(ValueError):
            validate_reference_ce_identity({'adapted_checkpoint_sha256': 'student'}, identity)


if __name__ == '__main__':
    unittest.main()
