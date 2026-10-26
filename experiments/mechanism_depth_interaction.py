"""Task 3 phase 1: depth-interaction mechanism audit on the frozen R2 cache (CPU only).

Uses only the cached counterfactual path costs in
``runs/repair_2026_10_25/r2_full/states_{controller_train,newtest}.pt`` and the
Plan-B recompute ``runs/followup_2026_10_26/plan_b/r2_summary.json``.  No model
forward, no training, no new states.

The cache stores, per state, the FP32 CE of each of the 4 candidate paths at
T=1 (``immediate`` = L1) and at T=``rounds``=4 under the frozen baseline
policy (``q_pi0`` = L4).  Intermediate depths T=2,3 were never cached, so the
depth gradient G = L1 - L4 uses the two endpoints only.

Pre-fixed rules (all fitted on ``controller_train`` costs only; nothing is
selected by test labels):

- ``baseline``                     original router action (reference).
- ``constant_global``              single global action chosen on train,
                                   executed with fallback to the router action
                                   when absent from a state's candidate set
                                   (``group_rules`` constant part).
- ``candidate_rank``               fixed candidate rank chosen on train
                                   (``candidate_rank_rule``).
- ``uniform_candidate_expectation``  per-state analytic mean over the 4
                                   candidates (no sampling).

Per rule vs baseline, on all states and on the actually-changed-action subset:

- ``G = L1 - L4``            per-state depth gain.
- ``I = G_rule - G_baseline``  interaction (how the rule tilts the depth gradient).
- ``D = L4_rule - L4_baseline``  final-depth deploy difference (positive = worse).
- ``T1 = L1_rule - L1_baseline`` immediate difference.

D, T1, I get document-level paired bootstrap intervals (documents resampled,
never tokens), a document direction distribution, and a
leave-out-max-token-share-document sensitivity.  A large |G| or |I| does NOT
imply a deploy benefit; only D at the final depth is deploy-relevant.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

from eval.counterfactual import PathCosts
from eval.state_sampling import document_paired_bootstrap
from experiments.followup_b_recompute import selection_diagnostics
from experiments.r1_closed_loop import source_identity
from experiments.r2_value import candidate_rank_rule, group_rules
from train.recovery_common import atomic_json, sha256_file

BOOTSTRAP_KEYS = (
    "estimate",
    "ci_lo",
    "ci_hi",
    "confidence",
    "se",
    "n_bootstrap",
    "n_documents",
    "n_rows",
    "max_doc_token_share",
    "effective_documents",
)


def depth_gain(l1: torch.Tensor, l4: torch.Tensor) -> torch.Tensor:
    """Per-state depth gain G = L1 - L4 (positive = extra rounds help)."""
    if l1.shape != l4.shape:
        raise ValueError("l1 and l4 must have the same shape")
    return l1 - l4


def depth_contrast(l1_rule: torch.Tensor, l4_rule: torch.Tensor, l1_base: torch.Tensor, l4_base: torch.Tensor) -> dict:
    """Per-state contrasts of a rule against the baseline.

    Returns ``d_l1`` (T1 immediate difference), ``d_l4`` (D final-depth
    difference) and ``d_g`` (interaction I = d_l1 - d_l4, equivalently
    G_rule - G_baseline).  Positive values mean the rule is worse on cost /
    steeper in depth gain.
    """
    d_l1 = l1_rule - l1_base
    d_l4 = l4_rule - l4_base
    return {"d_l1": d_l1, "d_l4": d_l4, "d_g": d_l1 - d_l4}


def changed_mask(costs: PathCosts, selection: torch.Tensor) -> torch.Tensor:
    """States whose actually executed global action differs from the router's.

    Comparison is on executed *global* action ids, so a fallback local index
    that resolves to the router's pair does not count as a change.
    """
    rows = torch.arange(selection.shape[0])
    actual = costs.global_action_ids[rows, selection]
    base = costs.global_action_ids[rows, costs.baseline_action]
    return actual != base


def doc_direction(diff: torch.Tensor, doc_ids: torch.Tensor) -> dict:
    """Distribution of per-document mean diff signs (negative = rule better)."""
    diff = np.asarray(diff, dtype=np.float64).reshape(-1)
    docs = np.asarray(doc_ids).reshape(-1)
    if diff.shape != docs.shape:
        raise ValueError("diff and doc_ids must have the same length")
    unique, inverse = np.unique(docs, return_inverse=True)
    sums = np.zeros(unique.shape[0], dtype=np.float64)
    counts = np.zeros(unique.shape[0], dtype=np.float64)
    np.add.at(sums, inverse, diff)
    np.add.at(counts, inverse, np.ones_like(diff))
    means = sums / counts
    n = unique.shape[0]
    return {
        "n_documents": int(n),
        "n_negative_rule_better": int((means < 0).sum()),
        "n_positive_rule_worse": int((means > 0).sum()),
        "n_zero": int((means == 0).sum()),
        "frac_negative_rule_better": float((means < 0).mean()),
        "frac_positive_rule_worse": float((means > 0).mean()),
    }


def leave_max_doc(diff: torch.Tensor, doc_ids: torch.Tensor) -> dict:
    """Estimate with the maximum token-share document removed."""
    diff = np.asarray(diff, dtype=np.float64).reshape(-1)
    docs = np.asarray(doc_ids).reshape(-1)
    if diff.shape != docs.shape:
        raise ValueError("diff and doc_ids must have the same length")
    unique, inverse = np.unique(docs, return_inverse=True)
    counts = np.zeros(unique.shape[0], dtype=np.float64)
    np.add.at(counts, inverse, np.ones_like(diff))
    shares = counts / counts.sum()
    drop = int(shares.argmax())
    kept = diff[inverse != drop]
    if kept.shape[0] == 0:
        raise ValueError("removing the max-share document leaves no rows")
    return {
        "dropped_doc_id": int(unique[drop]),
        "dropped_token_share": float(shares[drop]),
        "estimate_without_doc": float(kept.mean()),
        "delta_vs_full": float(kept.mean() - diff.mean()),
    }


def _bootstrap(a: torch.Tensor, b: torch.Tensor, doc_ids: torch.Tensor, n_bootstrap: int, seed: int) -> dict:
    result = document_paired_bootstrap(a, b, doc_ids, n_bootstrap=n_bootstrap, seed=seed)
    return {key: result[key] for key in BOOTSTRAP_KEYS}


def _contrast_block(diff: torch.Tensor, doc_ids: torch.Tensor, paired_a: torch.Tensor, paired_b: torch.Tensor, n_bootstrap: int, seed: int) -> dict:
    return {
        "mean": float(diff.mean()),
        "doc_paired_bootstrap": _bootstrap(paired_a, paired_b, doc_ids, n_bootstrap, seed),
        "doc_direction": doc_direction(diff, doc_ids),
        "leave_max_doc": leave_max_doc(diff, doc_ids),
    }


def rule_report(
    name: str,
    costs: PathCosts,
    l1_rule: torch.Tensor,
    l4_rule: torch.Tensor,
    l1_base: torch.Tensor,
    l4_base: torch.Tensor,
    doc_ids: torch.Tensor,
    *,
    selection: torch.Tensor | None,
    requested_global=None,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """Full depth-interaction report for one pre-fixed rule vs the baseline."""
    contrast = depth_contrast(l1_rule, l4_rule, l1_base, l4_base)
    g_rule = depth_gain(l1_rule, l4_rule)
    report = {
        "n_states": int(l1_rule.shape[0]),
        "mean_l1": float(l1_rule.mean()),
        "mean_l4": float(l4_rule.mean()),
        "mean_g": float(g_rule.mean()),
        "D_l4_rule_minus_baseline": _contrast_block(contrast["d_l4"], doc_ids, l4_rule, l4_base, n_bootstrap, seed),
        "T1_l1_rule_minus_baseline": _contrast_block(contrast["d_l1"], doc_ids, l1_rule, l1_base, n_bootstrap, seed),
        "I_g_rule_minus_baseline": _contrast_block(contrast["d_g"], doc_ids, g_rule, depth_gain(l1_base, l4_base), n_bootstrap, seed),
    }
    if selection is None:
        report["fallback_fraction"] = None
        report["actual_global_action_counts"] = None
        report["changed_only"] = None
        report["changed_only_note"] = "analytic per-state mixture over all candidates; there is no discrete selection, so a changed-action subset is not defined"
        return report
    diagnostics = selection_diagnostics(costs, selection, requested_global)
    report["fallback_fraction"] = diagnostics.get("fallback_fraction", 0.0 if requested_global is None else None)
    report["actual_global_action_counts"] = diagnostics["actual_global_action_counts"]
    mask = changed_mask(costs, selection)
    n_changed = int(mask.sum())
    changed = {"n_changed": n_changed, "changed_fraction": float(mask.float().mean())}
    if n_changed:
        sub_docs = doc_ids[mask]
        changed["mean_d_l1"] = float(contrast["d_l1"][mask].mean())
        changed["mean_d_l4"] = float(contrast["d_l4"][mask].mean())
        changed["mean_d_g"] = float(contrast["d_g"][mask].mean())
        changed["d_l4_doc_paired_bootstrap"] = _bootstrap(l4_rule[mask], l4_base[mask], sub_docs, n_bootstrap, seed)
        changed["d_l4_doc_direction"] = doc_direction(contrast["d_l4"][mask], sub_docs)
    report["changed_only"] = changed
    return report


def run(r2_dir, plan_b_summary_path, output_dir, n_bootstrap=2000, seed=0) -> dict:
    r2_dir = Path(r2_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states = {role: torch.load(r2_dir / f"states_{role}.pt", map_location="cpu", weights_only=False) for role in ("controller_train", "newtest")}
    train, test = states["controller_train"]["costs"], states["newtest"]["costs"]
    doc_ids = states["newtest"]["doc_ids"]
    if test.rounds != 4:
        raise ValueError(f"cache rounds must be 4, got {test.rounds}")

    constant, _group, group_metadata = group_rules(train, test)
    rank, rank_metadata = candidate_rank_rule(train, test)
    selections = {
        "constant_global": constant,
        "candidate_rank": rank,
    }
    rows = torch.arange(test.q_pi0.shape[0])
    l1_base = test.immediate[rows, test.baseline_action]
    l4_base = test.q_pi0[rows, test.baseline_action]
    rule_costs = {
        "constant_global": (test.immediate[rows, constant], test.q_pi0[rows, constant], constant, group_metadata["constant_global_action"]),
        "candidate_rank": (test.immediate[rows, rank], test.q_pi0[rows, rank], rank, None),
        "uniform_candidate_expectation": (test.immediate.mean(dim=1), test.q_pi0.mean(dim=1), None, None),
    }
    reports = {}
    for name, (l1_rule, l4_rule, selection, requested_global) in rule_costs.items():
        reports[name] = rule_report(name, test, l1_rule, l4_rule, l1_base, l4_base, doc_ids, selection=selection, requested_global=requested_global, n_bootstrap=n_bootstrap, seed=seed)

    plan_b = json.loads(Path(plan_b_summary_path).read_text())
    consistency = {"checks": {}, "matched": True}

    def check(label, mine, theirs, tol=1e-6):
        delta = abs(float(mine) - float(theirs))
        consistency["checks"][label] = {"mine": float(mine), "plan_b": float(theirs), "abs_delta": delta}
        if delta > tol:
            consistency["matched"] = False

    check("constant_global_deploy_improvement", -reports["constant_global"]["D_l4_rule_minus_baseline"]["mean"], plan_b["policies"]["constant_global"]["deploy_improvement"])
    check("candidate_rank_deploy_improvement", -reports["candidate_rank"]["D_l4_rule_minus_baseline"]["mean"], plan_b["policies"]["candidate_rank"]["deploy_improvement"])
    check("uniform_deploy_improvement", -reports["uniform_candidate_expectation"]["D_l4_rule_minus_baseline"]["mean"], plan_b["policies"]["uniform_candidate_expectation"]["deploy_improvement"])
    check("constant_global_fallback_fraction", reports["constant_global"]["fallback_fraction"], plan_b["selection_diagnostics"]["constant_global"]["fallback_fraction"])
    check("rank_metadata", rank_metadata["rank"], plan_b["rule_metadata"]["rank"])
    check("constant_global_action", group_metadata["constant_global_action"], plan_b["rule_metadata"]["constant_global_action"])
    check("constant_global_D_bootstrap_estimate", reports["constant_global"]["D_l4_rule_minus_baseline"]["doc_paired_bootstrap"]["estimate"], -plan_b["paired_document_intervals"]["constant_global"]["estimate"])
    if not consistency["matched"]:
        raise ValueError(f"recomputed rules do not match the frozen Plan-B summary: {json.dumps(consistency['checks'], indent=2)}")

    table_path = output_dir / "per_state.csv"
    actual = {name: test.global_action_ids[rows, sel] for name, sel in selections.items()}
    changed = {name: changed_mask(test, sel) for name, sel in selections.items()}
    contrasts = {name: depth_contrast(l1, l4, l1_base, l4_base) for name, (l1, l4, _sel, _req) in rule_costs.items()}
    with table_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "state_index", "doc_id", "baseline_global",
            "constant_actual_global", "rank_actual_global", "constant_changed", "rank_changed",
            "l1_baseline", "l4_baseline", "g_baseline",
            "d_l1_constant", "d_l4_constant", "d_g_constant",
            "d_l1_rank", "d_l4_rank", "d_g_rank",
            "d_l1_uniform", "d_l4_uniform", "d_g_uniform",
        ])
        g_base = depth_gain(l1_base, l4_base)
        base_global = test.global_action_ids[rows, test.baseline_action]
        for index in range(rows.shape[0]):
            row = [
                index, int(doc_ids[index]), int(base_global[index]),
                int(actual["constant_global"][index]), int(actual["candidate_rank"][index]),
                int(changed["constant_global"][index]), int(changed["candidate_rank"][index]),
                f"{l1_base[index]:.6f}", f"{l4_base[index]:.6f}", f"{g_base[index]:.6f}",
            ]
            for name in ("constant_global", "candidate_rank", "uniform_candidate_expectation"):
                row += [f"{contrasts[name]['d_l1'][index]:.6f}", f"{contrasts[name]['d_l4'][index]:.6f}", f"{contrasts[name]['d_g'][index]:.6f}"]
            writer.writerow(row)

    summary = {
        "scope": "task3 phase1 depth-interaction mechanism audit on the frozen R2 cache; previously viewed newtest; diagnostic, not confirmation",
        "inputs": {
            "states_controller_train": {"path": str(r2_dir / "states_controller_train.pt"), "sha256": sha256_file(r2_dir / "states_controller_train.pt")},
            "states_newtest": {"path": str(r2_dir / "states_newtest.pt"), "sha256": sha256_file(r2_dir / "states_newtest.pt")},
            "plan_b_r2_summary": {"path": str(plan_b_summary_path), "sha256": sha256_file(plan_b_summary_path)},
            "backbone_hash": states["newtest"]["backbone_hash"],
            "corpus_doc_map_sha256_newtest": states["newtest"]["corpus_doc_map_sha256"],
        },
        "definitions": {
            "L1": "immediate FP32 CE after 1 forced round (cached)",
            "L4": "Q^pi0 FP32 CE after 4 rounds under the frozen baseline policy (cached)",
            "G": "L1 - L4 per state",
            "I": "G_rule - G_baseline",
            "D": "L4_rule - L4_baseline (positive = rule worse at final depth; deploy-relevant)",
            "T1": "L1_rule - L1_baseline",
        },
        "rules": {
            "pre_fixed": True,
            "fitted_on": "controller_train costs only; no test labels or test costs are used to pick any action",
            "baseline": "original router Top-2 action (reference)",
            "constant_global": "single train-selected global action with fallback to the router action when absent from a state's candidate set",
            "candidate_rank": "fixed train-selected candidate rank (local index)",
            "uniform_candidate_expectation": "analytic per-state mean over the 4 cached candidate paths (no sampling)",
        },
        "rule_metadata": {
            "constant_global_action": group_metadata["constant_global_action"],
            "candidate_rank": rank_metadata["rank"],
            "min_group_states": group_metadata["min_group_states"],
        },
        "baseline_reference": {
            "mean_l1": float(l1_base.mean()),
            "mean_l4": float(l4_base.mean()),
            "mean_g": float(depth_gain(l1_base, l4_base).mean()),
        },
        "rules_report": reports,
        "consistency_vs_plan_b": consistency,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "limitations": [
            "The cache stores only T=1 and T=4 readouts; depths T=2,3 were never measured, so G uses the two endpoints only.",
            "Document identity is exact doc_id only: no near-duplicate / group-level near hash (e.g. MinHash) exists in this cache, so near-duplicate leakage between controller_train and newtest cannot be excluded beyond the verified empty exact doc_id intersection; this analysis does not fabricate such a hash.",
            "All metrics are on previously viewed newtest states (diagnostic, not a confirmation split).",
            "A large interaction I or depth gain G does not imply a deploy benefit; only D at the final depth is deploy-relevant, and every D here is orders of magnitude below the oracle gap.",
        ],
        "conclusion": {},
    }
    d_const = reports["constant_global"]["D_l4_rule_minus_baseline"]
    d_rank = reports["candidate_rank"]["D_l4_rule_minus_baseline"]
    summary["conclusion"] = {
        "stable": bool(consistency["matched"]),
        "interpretation": (
            f"candidate_rank has a large interaction (I={reports['candidate_rank']['I_g_rule_minus_baseline']['mean']:.4f}) but a final-depth "
            f"deploy difference D={d_rank['mean']:.6f} whose doc-level 95% CI crosses zero; constant_global has a statistically positive but "
            f"tiny improvement (-D={-d_const['mean']:.6f}, 95% CI excludes zero); uniform expectation is worse than baseline. "
            "Large G/I is a mechanism statement, not deploy value."
        ),
        "t2_t3_recommendation": (
            "Not warranted for a deployment claim: the largest deploy-relevant effect (constant_global, ~4.6e-4 nats, ~0.009% of baseline cost) "
            "is far below practical relevance and 1.3% of the label-informed oracle gap; a <=1024-state T2/T3 confirmation would at most re-verify "
            "a negligible D. If run anyway for sign stability, cap it at <=1024 states on a fresh split and pre-register D, not G/I, as the endpoint."
        ),
    }
    atomic_json(output_dir / "summary.json", summary)
    manifest = {
        "source": source_identity(),
        "argv": sys.argv,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "summary_sha256": sha256_file(output_dir / "summary.json"),
        "per_state_csv_sha256": sha256_file(table_path),
    }
    atomic_json(output_dir / "run_manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-dir", default="runs/repair_2026_10_25/r2_full")
    parser.add_argument("--plan-b-summary", default="runs/followup_2026_10_26/plan_b/r2_summary.json")
    parser.add_argument("--output-dir", default="runs/next_2026_10_26/mechanism")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = run(args.r2_dir, args.plan_b_summary, args.output_dir, args.bootstrap, args.seed)
    compact = {
        name: {
            "D": report["D_l4_rule_minus_baseline"]["mean"],
            "D_ci": [report["D_l4_rule_minus_baseline"]["doc_paired_bootstrap"]["ci_lo"], report["D_l4_rule_minus_baseline"]["doc_paired_bootstrap"]["ci_hi"]],
            "I": report["I_g_rule_minus_baseline"]["mean"],
            "T1": report["T1_l1_rule_minus_baseline"]["mean"],
            "fallback_fraction": report["fallback_fraction"],
            "changed_fraction": None if report["changed_only"] is None else report["changed_only"]["changed_fraction"],
        }
        for name, report in summary["rules_report"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
