"""CPU evaluation integration parity; not production CUDA validation."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.rl_lm_evaluate import (
    BOOTSTRAP_SEED, _grouped_ce, _online_nll, _paired_bootstrap, _legacy_nll, _token_ce,
)
from experiments.rl_lm_train import OnlineSelectionEvaluator
from rl_lm.inference import CanonicalTeacherPolicy
from tests.rl_lm.test_online_parity import ScriptedPolicy
from tests.rl_lm.tiny_lm import build_tiny_model


class StreamingEvaluationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model = build_tiny_model()
        self.ids = torch.arange(70).remainder(64).reshape(2, 35)
        self.labels = self.ids.clone()
        self.labels[:, ::4] = -100
        self.batch = dict(input_ids=self.ids, segment_ids=torch.zeros_like(self.ids),
                          labels=self.labels, valid=self.labels != -100)
        self.config = {'environment.max_calls': 2, 'environment.expert_chunk': 64,
                       'environment.head_chunk': 64,
                       'environment.deployment_backend': 'sparse_active_merged_v3',
                       'environment.frozen_compute_dtype': 'float32'}

    def policy(self, calls):
        return ScriptedPolicy([1] if calls == 1 else [1, [0, 2] * 35], 70)

    def test_online_and_selection_reference_parity_and_no_full_logits(self):
        for calls in (1, 2):
            with self.subTest(calls=calls):
                reference = _online_nll(self.config, self.model, self.policy(calls),
                                        self.batch, 'cpu', max_calls=calls,
                                        full_logits_reference=True)
                with patch('rl_lm.inference.production_logits', side_effect=AssertionError('full logits')):
                    actual = _online_nll(self.config, self.model, self.policy(calls),
                                         self.batch, 'cpu', max_calls=calls)
                torch.testing.assert_close(actual['ce'], reference['ce'], atol=0, rtol=0)
                self.assertEqual(actual['nll'], reference['nll'])
                self.assertEqual(actual['mean_rounds'], reference['mean_rounds'])
                self.assertTrue(bool(actual['ce'][~self.batch['valid']].eq(0).all()))
                evaluator = OnlineSelectionEvaluator.__new__(OnlineSelectionEvaluator)
                evaluator.config = self.config
                evaluator._model = self.model
                evaluator.input_ids = self.ids
                evaluator.segment_ids = self.batch['segment_ids']
                evaluator.labels = self.labels
                evaluator.valid = self.batch['valid']
                expected = evaluator._forward_nll(self.policy(calls), max_calls=calls,
                                                  full_logits_reference=True)
                with patch('rl_lm.inference.production_logits', side_effect=AssertionError('full logits')):
                    got = evaluator._forward_nll(self.policy(calls), max_calls=calls)
                self.assertEqual(got, expected)

    def test_native_legacy_skips_unused_loss_without_changing_readout(self):
        from train.common import autocast_context
        with torch.no_grad(), autocast_context(torch.device('cpu'), 'bfloat16'):
            old = self.model(self.ids, self.labels, depth=1,
                             segment_ids=self.batch['segment_ids'])
        with patch.object(self.model, 'forward', wraps=self.model.forward) as forward:
            actual = _legacy_nll(self.model, self.batch, torch.device('cpu'))
        self.assertIsNone(forward.call_args.kwargs['labels'])
        expected = _token_ce(old.logits, self.labels)
        torch.testing.assert_close(actual['ce'], expected, atol=0, rtol=0)
        self.assertEqual(actual['nll'], float(expected.sum() / self.batch['valid'].sum()))

    def test_single_evaluator_streaming_matches_explicit_reference(self):
        from experiments.rl_lm_single_train import SingleOnlineSelectionEvaluator
        from tests.rl_lm.test_single_online import ScriptedSinglePolicy
        evaluator = SingleOnlineSelectionEvaluator.__new__(SingleOnlineSelectionEvaluator)
        evaluator.config = dict(self.config, **{'environment.deployment_backend': 'sparse_single_v1'})
        evaluator.input_ids = self.ids
        evaluator.segment_ids = self.batch['segment_ids']
        evaluator.labels = self.labels
        evaluator.valid = self.batch['valid']
        for calls in (1, 2):
            evaluator.config['environment.max_calls'] = calls
            def policy():
                return ScriptedSinglePolicy([1] if calls == 1 else [1, [0, 2] * 35], 70)
            reference = evaluator._forward_nll(self.model, policy(), full_logits_reference=True)
            with patch('rl_lm.readout.production_logits', side_effect=AssertionError('full logits')):
                actual = evaluator._forward_nll(self.model, policy())
            self.assertEqual(actual, reference)

    def test_adaptation_selection_streams_without_changing_sequence_batches(self):
        from experiments.rl_lm_single_adapt import pair_teacher_selection_nll, single_selection_nll
        ids = torch.arange(9 * 12).remainder(64).reshape(9, 12)
        segments = torch.zeros_like(ids)
        labels = ids.clone()
        labels[:, ::3] = -100
        kwargs = dict(compute_dtype='float32', chunk=64, head_chunk=64)
        for function, forward_path, logits_path in (
            (pair_teacher_selection_nll, 'experiments.rl_lm_single_adapt.forward_with_last_ffn_policy',
             'rl_lm.inference.production_logits'),
            (single_selection_nll, 'experiments.rl_lm_single_adapt.single_online_forward',
             'rl_lm.readout.production_logits'),
        ):
            import experiments.rl_lm_single_adapt as module
            forward = getattr(module, forward_path.rsplit('.', 1)[1])
            reference = function(self.model, ids, segments, labels,
                                 full_logits_reference=True, **kwargs)
            with patch(logits_path, side_effect=AssertionError('full logits')), \
                 patch(forward_path, wraps=forward) as wrapped:
                actual = function(self.model, ids, segments, labels, **kwargs)
            self.assertEqual(actual, reference)
            self.assertEqual([call.args[2].shape[0] for call in wrapped.call_args_list], [8, 1])
            self.assertTrue(all(call.kwargs['output_mode'] == 'token_ce'
                                for call in wrapped.call_args_list))

    def test_independent_group_aggregation_and_paired_bootstrap(self):
        # Unequal group sizes, masked positions, and two ranks sharing a group.
        learned = _online_nll(self.config, self.model, self.policy(2), self.batch, 'cpu')
        reference = _online_nll(self.config, self.model, CanonicalTeacherPolicy(),
                                self.batch, 'cpu', max_calls=1)
        docs = torch.arange(70).reshape(2, 35).remainder(5)
        mapping = {0: 'a', 1: 'a', 2: 'b', 3: 'c', 4: 'c'}
        groups = sorted(set(mapping.values()))
        expected = []
        for result in (learned, reference):
            sums, counts = {}, {}
            for group in groups:
                mask = self.batch['valid'] & torch.tensor(
                    [[mapping[int(rank)] == group for rank in row] for row in docs])
                sums[group] = sum(float(x) for x in result['ce'][mask])
                counts[group] = int(mask.sum())
            aggregate = _grouped_ce(result['ce'], self.batch['valid'], docs, mapping)
            self.assertEqual(aggregate, {'sum': sums, 'n': counts})
            expected.append(aggregate)
        actual = _paired_bootstrap(*expected, repetitions=127, confidence=.95)
        diff = np.array([expected[0]['sum'][g] - expected[1]['sum'][g] for g in groups])
        counts = np.array([expected[0]['n'][g] for g in groups])
        rng = np.random.Generator(np.random.Philox(BOOTSTRAP_SEED))
        values = []
        for _ in range(127):
            sample = rng.integers(0, len(groups), len(groups))
            values.append(sum(diff[i] for i in sample) / sum(counts[i] for i in sample))
        self.assertEqual(actual['tokens'], int(self.batch['valid'].sum()))
        self.assertEqual(actual['groups'], 3)
        self.assertEqual(actual['delta_nll'], float(diff.sum() / counts.sum()))
        self.assertAlmostEqual(actual['ci_low'], float(np.quantile(values, .025)))
        self.assertAlmostEqual(actual['ci_high'], float(np.quantile(values, .975)))


if __name__ == '__main__':
    unittest.main()
