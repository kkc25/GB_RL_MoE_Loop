"""Unit tests for E10 ordered complementarity (experiments/e10_ordered.py).

All tests run on CPU with a tiny Top-2 LoopMoE (4 experts -> 6 actions,
36 ordered two-round paths) and a synthetic batch, mirroring the E01
counterfactual test fixtures.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ModelConfig
from eval.counterfactual import CounterfactualLedger, extract_states
from experiments.e10_ordered import (
    AdditivePredictor,
    LowRankInteractionPredictor,
    PairMLP,
    argmin_pairs,
    baseline_pair_selection,
    evaluate_pair_policies,
    executed_pair_costs,
    fit_pair_predictor,
    interaction_gamma,
    mlp_hidden_dim_for_budget,
    oracle_pair_metrics,
    order_effect,
    order_shuffle_control,
    pair_mse,
    predict_pairs,
    random_pair_selection,
    reference_sensitivity,
    two_round_path_costs,
)
from model import LoopMoE
from model.top2 import baseline_top2_pairs, pair_to_action

EXPERTS = 4
N_ACTIONS = EXPERTS * (EXPERTS - 1) // 2  # 6
ROUNDS = 4


def tiny_model(seed: int = 7) -> LoopMoE:
    torch.manual_seed(seed)
    config = ModelConfig(
        layers=3,
        hidden_size=32,
        attention_heads=2,
        head_dim=16,
        vocab_size=64,
        intermediate_size=48,
        experts_per_layer=EXPERTS,
        top_k=2,
        adaptive_layers=(2,),
        moe_layers=(0, 1, 2),
        max_calls=4,
        target_only=False,
        max_position_embeddings=32,
    )
    return LoopMoE(config).eval()


def tiny_batch(batch: int = 2, length: int = 12, seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, 64, (batch, length), generator=generator)
    labels = torch.full((batch, length), -100)
    labels[:, 1:] = input_ids[:, 1:]
    labels[0, 5] = -100  # unsupervised hole inside the sequence
    segment_ids = torch.zeros(batch, length, dtype=torch.long)
    segment_ids[0, 8:] = 1
    segment_ids[1, :4] = -1  # padding
    labels[segment_ids < 0] = -100
    labels[0, 8] = -100  # document boundary: no supervision across segments
    return input_ids, labels, segment_ids


def ce_per_position(model, input_ids, labels, segment_ids, depth):
    with torch.no_grad():
        output = model(input_ids, labels, depth=depth, segment_ids=segment_ids)
    losses = F.cross_entropy(
        output.logits.float().reshape(-1, output.logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    return losses


class TwoRoundCostTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.input_ids, self.labels, self.segment_ids = tiny_batch()
        self.states = extract_states(self.model, self.input_ids, self.labels, self.segment_ids)
        self.costs = two_round_path_costs(self.model, self.states["h"], self.states["targets"], rounds=ROUNDS)

    def test_baseline_pair_path_matches_fixed_depth_forward(self):
        # The pair (a*, pi0-after-a*) is the model's normal forward: its T=2
        # and T=rounds costs must equal the fixed-depth baseline CE exactly.
        mask = self.labels != -100
        depth2 = ce_per_position(self.model, self.input_ids, self.labels, self.segment_ids, depth=2)[mask]
        depth4 = ce_per_position(self.model, self.input_ids, self.labels, self.segment_ids, depth=4)[mask]
        selection = baseline_pair_selection(self.costs)
        executed_q, executed_two_round = executed_pair_costs(self.costs, selection)
        torch.testing.assert_close(executed_two_round, depth2, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(executed_q, depth4, rtol=1e-5, atol=1e-6)

    def test_shapes_finiteness_and_candidate_set(self):
        n_states = int((self.labels != -100).sum())
        self.assertEqual(tuple(self.costs.immediate.shape), (n_states, N_ACTIONS))
        self.assertEqual(tuple(self.costs.pair_cost.shape), (n_states, N_ACTIONS, N_ACTIONS))
        self.assertEqual(tuple(self.costs.pair_q.shape), (n_states, N_ACTIONS, N_ACTIONS))
        self.assertEqual(tuple(self.costs.second_baseline.shape), (n_states, N_ACTIONS))
        self.assertTrue(torch.isfinite(self.costs.pair_cost).all())
        self.assertTrue(torch.isfinite(self.costs.pair_q).all())
        self.assertTrue(bool(((self.costs.second_baseline >= 0) & (self.costs.second_baseline < N_ACTIONS)).all()))

    def test_chunking_does_not_change_costs(self):
        chunked = two_round_path_costs(
            self.model, self.states["h"], self.states["targets"], rounds=ROUNDS,
            state_chunk=1, readout_chunk=3,
        )
        torch.testing.assert_close(chunked.immediate, self.costs.immediate, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(chunked.pair_cost, self.costs.pair_cost, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(chunked.pair_q, self.costs.pair_q, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(chunked.second_baseline, self.costs.second_baseline))

    def test_second_baseline_is_router_choice_after_forcing(self):
        # Replay one state manually: force action k at round 1, then the
        # router's Top-2 at round 2 must equal second_baseline[0, k].
        layer = self.model.config.adaptive_layers[-1]
        block = self.model.blocks[layer]
        gamma = self.model.gammas[layer]
        h = self.states["h"][:1]
        k = 3
        with torch.no_grad():
            update, _, _, _ = block.executor(block.ffn_norm(h), torch.tensor([k]))
            after_first = h + gamma * update
            logits = block.executor.gate_logits(block.ffn_norm(after_first))
            expected = pair_to_action(baseline_top2_pairs(logits.float()), experts=EXPERTS)
        self.assertEqual(int(expected[0]), int(self.costs.second_baseline[0, k]))

    def test_gamma_definition_and_anchor_structure(self):
        gamma = interaction_gamma(self.costs.pair_q, 0, 0)
        manual = (
            self.costs.pair_q[:, :, 0].unsqueeze(-1)
            + self.costs.pair_q[:, 0, :].unsqueeze(-2)
            - self.costs.pair_q
            - self.costs.pair_q[:, 0, 0].view(-1, 1, 1)
        )
        torch.testing.assert_close(gamma, manual)
        # Any path containing an anchor action has zero interaction by construction.
        torch.testing.assert_close(gamma[:, 0, :], torch.zeros_like(gamma[:, 0, :]))
        torch.testing.assert_close(gamma[:, :, 0], torch.zeros_like(gamma[:, :, 0]))
        # Gamma depends on the reference: a different anchor gives a different tensor.
        other = interaction_gamma(self.costs.pair_q, 1, 2)
        self.assertFalse(torch.allclose(gamma, other))
        with self.assertRaises(ValueError):
            interaction_gamma(self.costs.pair_q, N_ACTIONS, 0)

    def test_reference_sensitivity_reports_anchor_spread(self):
        report = reference_sensitivity(self.costs.pair_q, anchors=((0, 0), (1, 1), (2, 3)))
        self.assertEqual(len(report["anchors"]), 3)
        self.assertEqual(report["anchors"][0]["a0"], 0)
        self.assertGreaterEqual(report["pair_mean_std_across_anchors"], 0.0)
        self.assertGreaterEqual(report["sign_flip_pair_fraction"], 0.0)
        self.assertLessEqual(report["sign_flip_pair_fraction"], 1.0)
        for entry in report["anchors"]:
            # Paths containing an anchor action have Gamma == 0 by construction,
            # so positive + negative fractions need not sum to one.
            self.assertLessEqual(entry["frac_positive"] + entry["frac_negative"], 1.0 + 1e-6)
        self.assertGreater(report["anchors"][0]["frac_positive"] + report["anchors"][0]["frac_negative"], 0.0)

    def test_order_effect_is_antisymmetric(self):
        order = order_effect(self.costs.pair_q)
        torch.testing.assert_close(order, -order.transpose(-1, -2))
        diagonal = torch.diagonal(order, dim1=-2, dim2=-1)
        torch.testing.assert_close(diagonal, torch.zeros_like(diagonal))
        # O_{a,b} = L_{a,b} - L_{b,a} by direct lookup.
        self.assertAlmostEqual(float(order[0, 1, 2]), float(self.costs.pair_q[0, 1, 2] - self.costs.pair_q[0, 2, 1]), places=6)

    def test_ledger_counts_real_work(self):
        ledger = CounterfactualLedger()
        two_round_path_costs(self.model, self.states["h"], self.states["targets"], rounds=ROUNDS, ledger=ledger)
        n_states = self.states["h"].shape[0]
        # Round 1: A forced calls per state; rounds 2..T: A x A calls per state each.
        self.assertEqual(ledger.candidate_pair_calls, n_states * N_ACTIONS + n_states * N_ACTIONS * N_ACTIONS * (ROUNDS - 1))
        # Readouts: after round 1 (A), after round 2 (A x A), after round T (A x A).
        self.assertEqual(ledger.head_readouts, n_states * (N_ACTIONS + 2 * N_ACTIONS * N_ACTIONS))


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.input_ids, self.labels, self.segment_ids = tiny_batch()
        self.states = extract_states(self.model, self.input_ids, self.labels, self.segment_ids)
        self.costs = two_round_path_costs(self.model, self.states["h"], self.states["targets"], rounds=ROUNDS)
        self.features = self.states["h"].float()

    def test_mlp_stays_within_lowrank_budget(self):
        input_dim = self.features.shape[1]
        lowrank = LowRankInteractionPredictor(input_dim, N_ACTIONS, rank=2)
        budget = sum(parameter.numel() for parameter in lowrank.parameters())
        hidden = mlp_hidden_dim_for_budget(input_dim, N_ACTIONS, budget)
        mlp = PairMLP(input_dim, N_ACTIONS, hidden)
        mlp_count = sum(parameter.numel() for parameter in mlp.parameters())
        self.assertLessEqual(mlp_count, budget)
        # The search is maximal: one more hidden unit would exceed the budget.
        wider = PairMLP(input_dim, N_ACTIONS, hidden + 1)
        wider_count = sum(parameter.numel() for parameter in wider.parameters())
        self.assertGreater(wider_count, budget)
        # The additive model is strictly smaller than the interaction budget.
        additive = AdditivePredictor(input_dim, N_ACTIONS)
        additive_count = sum(parameter.numel() for parameter in additive.parameters())
        self.assertLess(additive_count, budget)

    def test_lowrank_with_zero_interaction_equals_additive(self):
        input_dim = self.features.shape[1]
        torch.manual_seed(0)
        additive = AdditivePredictor(input_dim, N_ACTIONS)
        lowrank = LowRankInteractionPredictor(input_dim, N_ACTIONS, rank=2)
        lowrank.u.load_state_dict(additive.u.state_dict())
        lowrank.v.load_state_dict(additive.v.state_dict())
        with torch.no_grad():
            lowrank.p.weight.zero_()
            lowrank.p.bias.zero_()
            lowrank.q.weight.zero_()
            lowrank.q.bias.zero_()
        with torch.no_grad():
            torch.testing.assert_close(lowrank(self.features), additive(self.features))

    def test_fit_predict_and_select_legal_pairs(self):
        results = {}
        for name, predictor in (
            ("additive", AdditivePredictor(self.features.shape[1], N_ACTIONS)),
            ("lowrank", LowRankInteractionPredictor(self.features.shape[1], N_ACTIONS, rank=2)),
            ("mlp", PairMLP(self.features.shape[1], N_ACTIONS, hidden_dim=16)),
        ):
            result = fit_pair_predictor(
                predictor, self.features, self.costs.pair_q,
                epochs=3, batch_size=8, seed=0,
                validation=(self.features, self.costs.pair_q),
            )
            prediction = predict_pairs(result, self.features)
            self.assertEqual(tuple(prediction.shape), (self.features.shape[0], N_ACTIONS, N_ACTIONS))
            self.assertTrue(torch.isfinite(prediction).all())
            selection = argmin_pairs(prediction)
            self.assertEqual(tuple(selection.shape), (self.features.shape[0], 2))
            self.assertTrue(bool(((selection >= 0) & (selection < N_ACTIONS)).all()))
            self.assertTrue(all(entry["train_mse"] == entry["train_mse"] for entry in result.history))
            results[name] = result
        # Fitting reduces the training loss from the initial epoch.
        for result in results.values():
            self.assertLess(result.history[-1]["train_mse"], result.history[0]["train_mse"])

    def test_additive_fits_additive_targets(self):
        # Sanity of the regression itself: a table that IS u_a(s) + v_b(s)
        # must be fit far better than a nonzero residual floor.
        generator = torch.Generator().manual_seed(5)
        features = torch.randn(24, self.features.shape[1], generator=generator)
        w_u = torch.randn(self.features.shape[1], N_ACTIONS, generator=generator)
        w_v = torch.randn(self.features.shape[1], N_ACTIONS, generator=generator)
        costs = (features @ w_u).unsqueeze(-1) + (features @ w_v).unsqueeze(-2)
        result = fit_pair_predictor(
            AdditivePredictor(features.shape[1], N_ACTIONS), features, costs,
            epochs=1500, lr=1e-2, batch_size=24, seed=0,
        )
        final = pair_mse(result, features, costs)
        self.assertLess(final, 1e-3 * costs.var().item())

    def test_executed_cost_is_measured_table_lookup(self):
        n_states = self.costs.pair_q.shape[0]
        selection = random_pair_selection(n_states, N_ACTIONS, seed=0, device=self.costs.pair_q.device)
        executed_q, executed_two_round = executed_pair_costs(self.costs, selection)
        rows = torch.arange(n_states)
        torch.testing.assert_close(executed_q, self.costs.pair_q[rows, selection[:, 0], selection[:, 1]])
        torch.testing.assert_close(executed_two_round, self.costs.pair_cost[rows, selection[:, 0], selection[:, 1]])
        with self.assertRaises(ValueError):
            executed_pair_costs(self.costs, selection[: n_states - 1])

    def test_evaluate_pair_policies_and_shuffle_control(self):
        n_states = self.costs.pair_q.shape[0]
        selections = {
            "baseline": baseline_pair_selection(self.costs),
            "random": random_pair_selection(n_states, N_ACTIONS, seed=1, device=self.costs.pair_q.device),
        }
        policies = evaluate_pair_policies(self.costs, selections)
        self.assertNotIn("oracle", policies)
        self.assertAlmostEqual(policies["baseline"]["deploy_improvement"], 0.0, places=7)
        control = order_shuffle_control(self.costs, selections["random"])
        # Swapping a symmetric (a == b) selection changes nothing.
        same = torch.full((n_states, 2), 2, dtype=torch.long)
        identity = order_shuffle_control(self.costs, same)
        self.assertAlmostEqual(identity["mean_order_gain"], 0.0, places=7)
        self.assertIn("swapped_mean_cost", control)

    def test_oracle_pair_is_not_a_deploy_policy(self):
        oracle = oracle_pair_metrics(self.costs)
        self.assertFalse(oracle["deployable"])
        self.assertGreaterEqual(oracle["oracle_gap"], -1e-6)
        self.assertGreaterEqual(oracle["mean_oracle_pair_is_baseline"], 0.0)
        self.assertLessEqual(oracle["mean_oracle_pair_is_baseline"], 1.0)


if __name__ == "__main__":
    unittest.main()
