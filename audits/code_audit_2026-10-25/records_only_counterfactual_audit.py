"""Inspect uploaded JSONL labels and evaluate prespecified train-only cheap policies.

No model weights are needed. No test label is used to choose an action.
JSONL costs were rounded to six decimal places by the experiment writer.
The held-out examples cover very few documents, so this is diagnostic only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["q_pi0"] for row in rows], dtype=np.float64)


def describe(rows: list[dict]) -> dict:
    q = matrix(rows)
    common = q.mean(axis=1)
    centered = q - common[:, None]
    common_var = float(common.var())
    action_var = float(np.mean(centered**2))
    return {
        "n_states": len(rows),
        "n_documents": len({row["doc"] for row in rows}),
        "n_sequences": len({row["seq"] for row in rows}),
        "document_counts": dict(Counter(str(row["doc"]) for row in rows)),
        "sequence_counts": dict(Counter(str(row["seq"]) for row in rows)),
        "variance_of_per_state_mean_q": common_var,
        "mean_action_centered_squared_q": action_var,
        "common_to_action_variance_ratio": common_var / action_var if action_var else None,
        "common_fraction_of_total_q_variance": common_var / (common_var + action_var),
    }


def policy_metrics(rows: list[dict], selected: np.ndarray) -> dict:
    q = matrix(rows)
    base_action = np.array([row["baseline_action"] for row in rows])
    positions = np.arange(len(rows))
    base_q = q[positions, base_action]
    chosen_q = q[positions, selected]
    gains = base_q - chosen_q
    per_doc = {}
    for doc in sorted({row["doc"] for row in rows}):
        mask = np.array([row["doc"] == doc for row in rows])
        per_doc[str(doc)] = {"states": int(mask.sum()), "gain_nat": float(gains[mask].mean())}
    return {
        "mean_cost": float(chosen_q.mean()),
        "baseline_mean_cost": float(base_q.mean()),
        "deploy_improvement_nat": float(gains.mean()),
        "selected_action_counts": dict(Counter(map(str, selected.tolist()))),
        "by_document": per_doc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent / "GB_RL_MoE_Loop")
    args = parser.parse_args()
    output = {
        "scope": "Exploratory, source-record-only checks; test labels never used for policy fitting; no evidence of population-level improvement is claimed.",
        "cost_precision": "Published writer rounded each CE cost to 6 decimals; paired differences have at most 1e-6 rounding error per state.",
        "runs": {},
    }
    for name in ("pilot", "coverage_remediation", "m105_100m"):
        root = args.repo / "runs" / "e01" / name
        rows = {split: load(root / f"records_{split}.jsonl") for split in ("train", "validation", "test")}
        train_q = matrix(rows["train"])
        n_actions = train_q.shape[1]
        train_base = np.array([row["baseline_action"] for row in rows["train"]])
        constant_action = int(train_q.mean(axis=0).argmin())
        group_rule = np.arange(n_actions, dtype=np.int64)
        group_counts = {}
        for action in range(n_actions):
            mask = train_base == action
            group_counts[str(action)] = int(mask.sum())
            if mask.any():
                group_rule[action] = train_q[mask].mean(axis=0).argmin()
        result = {
            "coverage_and_variance": {split: describe(data) for split, data in rows.items()},
            "train_only_constant_action": constant_action,
            "train_only_group_action_map": group_rule.tolist(),
            "train_group_counts": group_counts,
            "group_rule": "At inference get the frozen router's baseline action, then select the action with minimum train-mean Q within that baseline-action group. Unseen train groups fall back to their baseline action.",
            "evaluation": {},
        }
        doc_sets = {split: {row["doc"] for row in data} for split, data in rows.items()}
        result["document_overlap_counts"] = {f"{a}_{b}": len(doc_sets[a] & doc_sets[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
        for split in ("validation", "test"):
            base_actions = np.array([row["baseline_action"] for row in rows[split]])
            result["evaluation"][split] = {
                "train_mean_constant": policy_metrics(rows[split], np.full(len(rows[split]), constant_action)),
                "train_mean_by_baseline_action": policy_metrics(rows[split], group_rule[base_actions]),
            }
        output["runs"][name] = result

    path = Path(__file__).with_name("records_only_counterfactual_audit.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for name, run in output["runs"].items():
        print(json.dumps({"run": name, "train_coverage": run["coverage_and_variance"]["train"], "test_variance": {k: v for k, v in run["coverage_and_variance"]["test"].items() if "variance" in k or "squared" in k}, "constant_action": run["train_only_constant_action"], "gains": {split: {method: metrics["deploy_improvement_nat"] for method, metrics in methods.items()} for split, methods in run["evaluation"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
