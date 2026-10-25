#!/usr/bin/env python3
"""CPU-only, standard-library checks of design arithmetic and counterexamples.

These are exact toy/analytic checks, not measurements of a GPU or trained LM.
Run from the bundle root: python3 code/verify_design.py --output results/design_checks.json
"""
import argparse
import json
import math
from pathlib import Path


def parameter_count(m):
    layers, d, experts, vocab = 12, 512, 8, 32768
    return vocab*d + layers*(4*d*d + 3*d*m*experts + d*experts + 2*d) + d


def seq_topk_counterexample():
    rows = [[0.40, 0.30, 0.20, 0.10], [0.49, 0.48, 0.02, 0.01]]
    k, executed = 1, 0
    trace = []
    ledger, actual_ledger_calls = 0, 0
    for m in range(1, len(rows)+1):
        pairs = sorted(((rows[t][e], t, e)
                        for t in range(m) for e in range(4)), reverse=True)
        selected = pairs[:m*k]
        requested = sum(t == m-1 for _, t, _ in selected)
        executed += requested
        ledger += k
        allowed = min(requested, ledger)
        ledger -= allowed
        actual_ledger_calls += allowed
        assert ledger >= 0 and actual_ledger_calls <= m*k
        trace.append(dict(prefix=m, requested_current=requested,
                          actual_cumulative_without_ledger=executed,
                          nominal_prefix_budget=m*k,
                          actual_cumulative_with_ledger=actual_ledger_calls))
    assert executed == 3 and len(rows)*k == 2
    return {"source_scope": "public Eq.7, not an audit of authors' implementation",
            "probabilities": rows, "K": k, "trace": trace}


def label_oracle_counterexample():
    # The two actions predict opposite labels. X has no information about Y.
    fixed = -0.5*(math.log(0.9)+math.log(0.1))
    clairvoyant = -math.log(0.9)
    assert math.isclose(fixed-clairvoyant, math.log(3), abs_tol=1e-12)
    return {"actions_predict_p_y1": [0.1, 0.9],
            "true_y1_probability": 0.5,
            "best_label_blind_route_nll": fixed,
            "clairvoyant_route_nll": clairvoyant,
            "unlearnable_oracle_gap_nats": fixed-clairvoyant,
            "scope": "two-route action space; averaging outputs adds another action"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    expected = {256: 67170816, 512: 104919552, 1024: 180417024}
    counts = {str(m): parameter_count(m) for m in expected}
    assert all(counts[str(m)] == p for m, p in expected.items())
    n_actions = math.comb(8, 2)
    assert n_actions == 28 and n_actions**2 == 784 and n_actions**4 == 614656
    d, m, v = 512, 512, 32768
    expert_f = 6*d*m
    head_f = 2*d*v
    pair_f = 2*expert_f
    shared_tree_f = (n_actions+n_actions**2)*pair_f+n_actions**2*head_f
    train_budget = {name: {str(q): tokens/(3600*q) for q in [2000, 8000, 20000]}
                    for name, tokens in {"screen_0.95B": 950000000,
                                         "minimal_3.65B": 3650000000,
                                         "architecture_5.65B": 5650000000,
                                         "expanded_9.65B": 9650000000}.items()}
    result = {
        "status": "executed_cpu_analytic_checks_only",
        "not_executed": ["GPU training", "LM accuracy evaluation", "throughput benchmark"],
        "parameters_by_expert_width": counts,
        "path_counts": {"top1_four_round_leaves": 8**4,
                        "top1_including_stop_prefixes": sum(8**i for i in range(5)),
                        "top2_actions": n_actions,
                        "top2_two_round_leaves": n_actions**2,
                        "top2_four_round_leaves": n_actions**4},
        "matrix_forward_flops": {"single_expert": expert_f, "vocabulary_head": head_f,
                                  "top2_ffn_and_router": pair_f+2*d*8,
                                  "head_over_top2_ffn_and_router": head_f/(pair_f+2*d*8),
                                  "exhaustive_two_round_tree_per_state": shared_tree_f,
                                  "tree_excludes": "backbone, router, norm, activation, softmax and dispatch"},
        "label_oracle_counterexample": label_oracle_counterexample(),
        "seqtopk_budget_counterexample": seq_topk_counterexample(),
        "gpu_hours_algebraic_scenarios_not_measured": train_budget,
        "all_assertions_passed": True,
    }
    out = json.dumps(result, indent=2, ensure_ascii=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
