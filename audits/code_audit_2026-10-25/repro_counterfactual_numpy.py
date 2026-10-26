"""Deterministic arithmetic repros for reviewed source expressions; no weights needed.

This script uses NumPy to reproduce the exact einsum/index/boolean semantics.
It does not claim to execute the full PyTorch model or reproduce the LM metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> None:
    results = {}
    rows = np.array([[[2.0], [7.0]]])
    w1 = np.array([[[3.0]], [[5.0]]])
    actual = np.einsum("...ei,oih->...oh", rows, w1)
    expected = np.einsum("...oi,oih->...oh", rows, w1)
    changed = rows.copy()
    changed[0, 1, 0] += 1.0
    actual_changed = np.einsum("...ei,oih->...oh", changed, w1)
    expected_changed = np.einsum("...oi,oih->...oh", changed, w1)
    results["e12_expert_axis"] = {
        "source": "experiments/e12_bidding.py:149",
        "actual_first_projection": actual.tolist(),
        "independent_adapter_projection": expected.tolist(),
        "change_expert1_input_actual_delta": (actual_changed - actual).tolist(),
        "change_expert1_input_expected_delta": (expected_changed - expected).tolist(),
    }
    assert actual.tolist() == [[[27.0], [45.0]]]
    assert expected.tolist() == [[[6.0], [35.0]]]
    assert actual_changed[0, 0, 0] != actual[0, 0, 0]
    assert expected_changed[0, 0, 0] == expected[0, 0, 0]

    h = np.array([[2.0, 0.0]])
    norm = h / np.sqrt(np.mean(h**2, axis=-1, keepdims=True) + 1e-6)
    mean = np.zeros((1, 2))
    std = np.ones((1, 2))
    tau = 1.7
    train_feature = (h - mean) / std
    deploy_feature = (norm - mean) / std
    results["e11_feature_contract"] = {
        "source": "train/value_student.py:35-37 vs eval/dynamic_halting.py:26-31",
        "illustrative_gain_predictor": "delta_hat(z) = z[0]",
        "raw_h_gain": float(train_feature[0, 0]),
        "normalized_h_gain": float(deploy_feature[0, 0]),
        "tau": tau,
        "raw_h_continue": bool(train_feature[0, 0] > tau),
        "normalized_h_continue": bool(deploy_feature[0, 0] > tau),
    }
    assert bool(train_feature[0, 0] > tau) != bool(deploy_feature[0, 0] > tau)

    pair_means = np.array([[[1.0]], [[0.0]]])
    positive = pair_means > 0
    coded_flip = positive.any(axis=0) & (~positive).any(axis=0)
    true_flip = (pair_means > 0).any(axis=0) & (pair_means < 0).any(axis=0)
    results["e10_sign_flip"] = {
        "source": "experiments/e10_ordered.py:237-238",
        "pair_mean_gamma_across_anchors": [1.0, 0.0],
        "coded_sign_flip": bool(coded_flip[0, 0]),
        "positive_and_negative_exists": bool(true_flip[0, 0]),
    }
    assert bool(coded_flip[0, 0]) and not bool(true_flip[0, 0])

    top2_table = np.array([(a, b) for a in range(8) for b in range(a + 1, 8)])
    custom_actions = top2_table[[0, 27]]
    coded_forced = np.arange(len(custom_actions))
    results["e01_custom_action_table"] = {
        "source": "eval/counterfactual.py:199-203,217,222",
        "requested_pairs": custom_actions.tolist(),
        "actually_forced_pairs": top2_table[coded_forced].tolist(),
        "note": "Default all-28 action order works; a non-prefix subset is mislabeled.",
    }
    assert not np.array_equal(custom_actions, top2_table[coded_forced])

    path = Path(__file__).with_name("repro_counterfactual_numpy_results.json")
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
