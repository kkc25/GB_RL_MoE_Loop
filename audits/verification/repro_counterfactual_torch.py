"""Run tiny deterministic repros against the unchanged reviewed repository.

Usage:
  python repro_counterfactual_torch.py

These random-weight/synthetic checks prove source-level defects, not the size
of their effect on the unavailable trained checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "GB_RL_MoE_Loop"
sys.path.insert(0, str(REPO))

import torch
from torch.nn import functional as F

from config import ModelConfig
from eval.counterfactual import cache_last_ffn_input, counterfactual_path_costs
from eval.dynamic_halting import DynamicHaltingPolicy
from experiments.e10_ordered import reference_sensitivity
from experiments.e12_bidding import ExpertBidAdapters, bidding_actions, pair_action_matrix
from model import LoopMoE
from model.top2 import enumerate_top2_actions
from train.value_student import ValueStudent, predict


def tiny_model() -> LoopMoE:
    torch.manual_seed(3)
    config = ModelConfig(
        layers=2, hidden_size=32, attention_heads=2, head_dim=16,
        vocab_size=64, intermediate_size=24, experts_per_layer=4,
        top_k=2, adaptive_layers=(1,), max_calls=4, target_only=False,
    )
    return LoopMoE(config).eval()


def main() -> None:
    torch.set_num_threads(1)
    results = {"torch_version": torch.__version__, "source_commit": "f2bef422f7315f30618f00c690bf82810d1e85ac", "trained_checkpoint_used": False}
    model = tiny_model()
    students = {}
    for name in ("immediate", "q_pi0"):
        student = ValueStudent(32, 6, 8)
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.zero_()
            if name == "immediate":
                student.net[0].weight[0, 0] = 1
                student.net[2].weight[0, 0] = 1
                student.net[4].weight[:, 0] = 1
        students[name] = {"student": student, "mean": torch.zeros(32), "std": torch.ones(32)}
    input_ids = torch.randint(0, 64, (1, 7))
    segments = torch.zeros_like(input_ids)
    h = cache_last_ffn_input(model, input_ids, segments)
    normalized = model.blocks[1].ffn_norm(h)
    raw_gain = predict(students["immediate"]["student"], h.reshape(-1, 32), torch.zeros(32), torch.ones(32))[:, 0]
    normalized_gain = predict(students["immediate"]["student"], normalized.reshape(-1, 32), torch.zeros(32), torch.ones(32))[:, 0]
    i = (raw_gain - normalized_gain).abs().argmax()
    tau = float((raw_gain[i] + normalized_gain[i]) / 2)
    logits, actual_rounds = DynamicHaltingPolicy(students, tau).forward(model, input_ids, segments)
    expected_rounds = 1 + 3 * (raw_gain > tau).long()
    assert not torch.equal(actual_rounds.flatten(), expected_rounds)
    assert torch.equal(actual_rounds.flatten(), 1 + 3 * (normalized_gain > tau).long())
    with torch.no_grad():
        reference_h, _ = model.fixed_depth(h, model.blocks[1], expected_rounds, None, model.gammas[1])
        reference_logits = F.linear(model.final_norm(reference_h), model.embedding.weight)
    results["dynamic_feature_mismatch"] = {
        "tau": tau, "raw_gains": raw_gain.tolist(), "normalized_gains": normalized_gain.tolist(),
        "expected_rounds_from_training_contract": expected_rounds.tolist(),
        "actual_deployed_rounds": actual_rounds.flatten().tolist(),
        "max_logit_difference": float((logits - reference_logits).abs().max()),
    }

    adapter = ExpertBidAdapters(1, 2, 1)
    rows = torch.tensor([[[2.0, 0.0, 0.0], [7.0, 0.0, 0.0]]])
    with torch.no_grad():
        adapter.w1.zero_()
        adapter.w1[:, 0, 0] = torch.tensor([3.0, 5.0])
        adapter.w2.fill_(1.0)
        adapter.w3.fill_(1.0)
    actual = adapter._apply(rows)
    expected_first = torch.einsum("...oi,oih->...oh", rows, adapter.w1) + adapter.b1
    expected = (F.gelu(F.gelu(expected_first)) * adapter.w3.squeeze(-1)).squeeze(-1)
    assert not torch.allclose(actual, expected)
    results["expert_adapter_einsum"] = {"actual_final_bids": actual.tolist(), "independent_adapter_final_bids": expected.tolist()}
    try:
        adapter.to("cpu")
    except (TypeError, AttributeError) as error:
        results["module_apply_override"] = {"reproduced": True, "exception": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError("Expected the reviewed _apply override to break Module.to")

    summary_adapter = ExpertBidAdapters(1, 2, 1, summary_mode="checkpoint", checkpoint_summary=torch.tensor([[1.0], [-1.0]]))
    with torch.no_grad():
        summary_adapter.w1.zero_()
        summary_adapter.w1[:, -1, 0] = 1.0
        summary_adapter.w2.fill_(1.0)
        summary_adapter.w3.fill_(1.0)
    summary_bids = summary_adapter.bids_for_anchor(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))
    assert torch.equal(summary_bids, torch.zeros_like(summary_bids))
    results["checkpoint_summary_cancels"] = {"expert_summaries": [[1.0], [-1.0]], "actual_bids": summary_bids.tolist()}

    actions = enumerate_top2_actions(4)
    targets = torch.randint(0, 64, (h.numel() // 32,))
    full = counterfactual_path_costs(model, h.reshape(-1, 32), targets, rounds=2)
    subset = counterfactual_path_costs(model, h.reshape(-1, 32), targets, rounds=2, actions=actions[[0, 5]])
    torch.testing.assert_close(subset.q_pi0[:, 1], full.q_pi0[:, 1])
    correct_difference = float((subset.q_pi0[:, 1] - full.q_pi0[:, 5]).abs().max())
    assert correct_difference > 1e-7
    results["custom_action_subset"] = {"requested_actions": actions[[0, 5]].tolist(), "actual_global_action_ids": [0, 1], "max_cost_error_vs_requested_second_action": correct_difference}

    costs = torch.tensor([[[0.0, 0.0], [0.0, -1.0]]], dtype=torch.float64)
    # For pair (1,1), anchors (0,0),(0,1) give +1 and 0, not opposite signs.
    coded = reference_sensitivity(costs, anchors=((0, 0), (0, 1)))
    results["reference_zero_counted_as_sign_flip"] = {"costs": costs.tolist(), "reported_fraction": coded["sign_flip_pair_fraction"], "strict_positive_negative_flip_fraction": 0.0}
    assert coded["sign_flip_pair_fraction"] == 0.25

    central_predictions = torch.tensor([[3.0, 4.0, 5.0, 6.0, 7.0, 0.0]])
    anchor = torch.tensor([0])
    matrix = pair_action_matrix(4)
    legal = matrix[anchor, :].flatten()
    legal = legal[legal >= 0]
    chosen_global = int(central_predictions.argmin(-1)[0])
    chosen_restricted = int(legal[central_predictions[0, legal].argmin()])
    assert chosen_global not in legal.tolist()
    results["central_vs_anchor_candidate_set"] = {"central_global_argmin": chosen_global, "anchor_legal_actions": legal.tolist(), "central_restricted_argmin": chosen_restricted}

    output = Path(__file__).with_name("repro_counterfactual_torch_results.json")
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
