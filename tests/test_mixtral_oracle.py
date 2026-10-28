"""CPU-only oracle checks, including an independent scripted action landscape."""
import math
import unittest
from unittest.mock import patch

import torch
from mixtral_lab.adapter import MixtralAdapter, pair_to_action
from mixtral_lab.oracle import OracleConfig, run_oracle


class Landscape:
    def __init__(self, best1=None, best2=None):
        self.model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        self.best1, self.best2 = best1, best2
        self.calls = []

    def native_pair(self, h):
        return torch.tensor([[0, 1]], device=h.device)

    def step_pairs(self, h, pairs):
        assert not torch.is_grad_enabled()
        return h + 1

    def readout(self, h):
        assert h.shape == (1, 2)
        return torch.zeros(1, 2)

    def score_pairs(self, h, targets, pairs):
        assert not torch.is_grad_enabled()
        assert h.shape == (1, 2) and targets.shape == (1,) and pairs.shape == (1, 2)
        action = pair_to_action(pairs).item()
        second = bool(h[0, 0] == 1)
        self.calls.append((second, action))
        best = self.best2 if second else self.best1
        loss = .2 if action == best else (math.log(2) if action == 1 else 1.)
        return torch.tensor([loss])


def run(adapter=None, **kwargs):
    return run_oracle(adapter or Landscape(), torch.zeros(2, 2), torch.tensor([0, 1]),
        [{'document_id': 'a', 'token_position': 0}, {'document_id': 'b', 'token_position': 2}],
        {'model': 'frozen-test', 'data': 'development'}, role='calibration', **kwargs)


class OracleTests(unittest.TestCase):
    def test_known_best_and_counts(self):
        adapter = Landscape(best1=7, best2=28)
        report = run(adapter)
        for row in report['contexts']:
            self.assertEqual(row['r1_action'], 7)
            self.assertEqual(row['r2_action'], 28)
            self.assertAlmostEqual(row['r1_gain_nats'], math.log(2)-.2)
            self.assertEqual(len(row['r1_scores_nats']), 28)
            self.assertEqual(len(row['r2_scores_nats']), 29)
            self.assertEqual(row['r2_scores_nats'][0], row['native_r1_ce'])
        self.assertEqual(report['budget']['actual_score_calls'], 116)
        self.assertEqual(report['budget']['actual_head_rows'], 118)
        self.assertEqual(len(adapter.calls), 116)
        self.assertEqual(report['summary']['r2']['fraction_above_practical_threshold'], 1.)
        self.assertEqual(set(report['document_groups']), {'a', 'b'})

    def test_no_headroom_stop_wins_ties(self):
        report = run()
        for row in report['contexts']:
            self.assertEqual((row['r1_action'], row['r2_action']), (1, 0))
            self.assertEqual((row['r1_gain_nats'], row['r2_gain_nats']), (0., 0.))
        self.assertEqual(report['summary']['stop_fraction'], 1.)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in Landscape().model.parameters()))

    def test_budget_preflight_and_config(self):
        adapter = Landscape()
        with self.assertRaisesRegex(ValueError, 'budget'):
            run(adapter, config=OracleConfig(max_score_calls=115))
        self.assertEqual(adapter.calls, [])
        for config in (OracleConfig(max_contexts=0), OracleConfig(max_score_calls=True),
                       OracleConfig(practical_gain=float('nan')), OracleConfig(anchor_atol=-1)):
            with self.assertRaises(ValueError):
                run(config=config)

    def test_nonfinite_and_inconsistent_anchor(self):
        for bad in (float('nan'), float('inf')):
            adapter = Landscape()
            with patch.object(adapter, 'score_pairs', return_value=torch.tensor([bad])):
                with self.assertRaisesRegex(ValueError, 'nonfinite'):
                    run(adapter)
        adapter = Landscape()
        with patch.object(adapter, 'readout', return_value=torch.tensor([[4., 0.]])):
            with self.assertRaisesRegex(ValueError, 'anchor CE mismatch'):
                run(adapter)

    def test_invalid_inputs_and_model(self):
        args = [Landscape(), torch.zeros(1, 2), torch.tensor([0]),
                [{'document_id': 'a', 'token_position': 0}], {'model': 'test'}]
        for role in ('train', 'newtest', 'test', ''):
            with self.assertRaises(ValueError):
                run_oracle(*args, role=role)
        args[0].model.train()
        with self.assertRaisesRegex(ValueError, 'frozen'):
            run_oracle(*args, role='selection')
        args[0].model.eval().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen'):
            run_oracle(*args, role='selection')
        args[0] = Landscape()
        args[1][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            run_oracle(*args, role='selection')

    def test_provenance_identity_validation(self):
        h = torch.zeros(2, 2)
        y = torch.tensor([0, 1])
        for rows in (
            [{'document_id': 'a', 'token_position': 0}]*2,
            [{'document_id': 'a', 'token_position': 0, 'target': 1},
             {'document_id': 'b', 'token_position': 0}],
            [{'document_id': 'a', 'token_position': 0, 'role': 'newtest'},
             {'document_id': 'b', 'token_position': 0}],
        ):
            with self.assertRaises(ValueError):
                run_oracle(Landscape(), h, y, rows, {'model': 'test'}, role='selection')

    def test_tiny_real_hf_frozen_and_chunk_independent(self):
        for dtype in (torch.float32, torch.bfloat16):
            for backend in ('eager', 'grouped_mm'):
                with self.subTest(dtype=dtype, backend=backend):
                    self.check_tiny(dtype, backend)

    def check_tiny(self, dtype, backend):
        from transformers import MixtralConfig, MixtralForCausalLM
        torch.manual_seed(19)
        config = MixtralConfig(vocab_size=16, hidden_size=16, intermediate_size=24,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            num_local_experts=8, num_experts_per_tok=2)
        config._attn_implementation = 'eager'
        config._experts_implementation = backend
        model = MixtralForCausalLM(config).to(dtype=dtype).eval()
        adapter = MixtralAdapter(model, head_chunk=1)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        ids = torch.tensor([[1, 2, 3]])
        h0 = adapter.capture(ids, torch.ones_like(ids))[:, 0, :]
        args = (adapter, h0, torch.tensor([2]),
                [{'document_id': 'tiny', 'token_position': 0}], {'model': 'tiny-random'})
        report = run_oracle(*args, role='selection')
        adapter.head_chunk = 64
        repeated = run_oracle(*args, role='selection')
        self.assertEqual(report['contexts'], repeated['contexts'])
        row = report['contexts'][0]
        self.assertEqual(row['max_anchor_error'], 0.)
        self.assertGreaterEqual(row['r1_gain_nats'], 0.)
        self.assertGreaterEqual(row['r2_gain_nats'], 0.)
        self.assertTrue(all(math.isfinite(x) for x in row['r1_scores_nats']+row['r2_scores_nats']))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
