from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import torch

from data.packed_lm import PackedLMDataset
from eval.counterfactual import CounterfactualLedger, candidate_actions, counterfactual_path_costs, evaluate_policies, oracle_metrics, random_selection
from eval.state_sampling import document_paired_bootstrap, sample_states
from lm_config import LMExperimentConfig
from model import LoopMoE
from train.recovery_common import sha256_file
from train.value_student import fit_value_student, predict


def linear_fit(features, costs):
    x = torch.cat([features, torch.ones(features.shape[0], 1)], dim=-1).double()
    weights = torch.linalg.lstsq(x, costs.double()).solution
    return weights


def linear_predict(features, weights):
    x = torch.cat([features, torch.ones(features.shape[0], 1)], dim=-1).double()
    return (x @ weights).float()


def baseline_global_ids(costs):
    rows = torch.arange(costs.baseline_action.shape[0], device=costs.baseline_action.device)
    return costs.global_action_ids[rows, costs.baseline_action]


def local_action_or_baseline(global_action_ids, global_action, baseline_local):
    if isinstance(global_action, int):
        matches = global_action_ids == global_action
    else:
        matches = global_action_ids == global_action.reshape(-1, 1)
    local = matches.long().argmax(dim=1)
    return torch.where(matches.any(dim=1), local, baseline_local)


def policy_cost_for_global(costs, global_action, rows=None):
    baseline = costs.baseline_action if rows is None else costs.baseline_action[rows]
    selection = local_action_or_baseline(costs.global_action_ids if rows is None else costs.global_action_ids[rows], global_action, baseline)
    index = torch.arange(selection.shape[0], device=selection.device)
    costs_q = costs.q_pi0 if rows is None else costs.q_pi0[rows]
    return costs_q[index, selection]


def candidate_rank_rule(train_costs, test_costs):
    rank = int(train_costs.q_pi0.mean(dim=0).argmin())
    return torch.full_like(test_costs.baseline_action, rank), {"rank": rank}


def group_rules(train_costs, test_costs, min_group_states=32):
    train_baseline_global = baseline_global_ids(train_costs)
    test_baseline_global = baseline_global_ids(test_costs)
    global_actions = train_costs.global_action_ids.unique()
    constant_costs = [policy_cost_for_global(train_costs, int(action)).mean() for action in global_actions]
    constant_global = int(global_actions[torch.stack(constant_costs).argmin()])
    constant = local_action_or_baseline(test_costs.global_action_ids, constant_global, test_costs.baseline_action)
    group_global = torch.full_like(test_baseline_global, constant_global)
    group_support = {}
    for baseline_global in test_baseline_global.unique().tolist():
        rows = torch.nonzero(train_baseline_global == baseline_global, as_tuple=False).squeeze(-1)
        group_support[str(baseline_global)] = int(rows.numel())
        if rows.numel() < min_group_states:
            continue
        costs = [policy_cost_for_global(train_costs, int(action), rows).mean() for action in global_actions]
        group_global[test_baseline_global == baseline_global] = int(global_actions[torch.stack(costs).argmin()])
    group = local_action_or_baseline(test_costs.global_action_ids, group_global, test_costs.baseline_action)
    metadata = {
        "constant_global_action": constant_global,
        "group_global_actions": group_global.tolist(),
        "group_train_support": group_support,
        "min_group_states": min_group_states,
    }
    return constant, group, metadata


def run(config_path, checkpoint_path, corpus_dir, output_dir, device_name, train_states, selection_states, calibration_states, test_states, epochs, seed):
    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = LMExperimentConfig.load(config_path)
    device = torch.device(device_name)
    model = LoopMoE(config.model).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    backbone_hash = sha256_file(checkpoint_path)
    roles = {
        "controller_train": train_states,
        "selection": selection_states,
        "calibration": calibration_states,
        "newtest": test_states,
    }
    costs = {}
    ledger = CounterfactualLedger()
    corpus_manifest = json.loads((Path(corpus_dir) / "manifest.json").read_text())
    for role, n_states in roles.items():
        dataset = PackedLMDataset(corpus_dir, role)
        states = sample_states(model, dataset, n_states, split=role, device=device, dtype=config.train.dtype, seed=seed, ledger=ledger)
        actions = candidate_actions(model, states["h"].to(device))
        role_costs = counterfactual_path_costs(model, states["h"].to(device), states["targets"].to(device), actions=actions, dtype=config.train.dtype, ledger=ledger)
        costs[role] = role_costs
        payload = dict(states)
        payload["costs"] = role_costs
        payload["backbone_hash"] = backbone_hash
        payload["corpus_doc_map_sha256"] = corpus_manifest["splits"][role]["doc_map_sha256"]
        torch.save(payload, output_dir / f"states_{role}.pt")
    train = costs["controller_train"]
    selection = costs["selection"]
    test = costs["newtest"]
    train_features = torch.load(output_dir / "states_controller_train.pt", map_location="cpu", weights_only=False)["h"]
    selection_features = torch.load(output_dir / "states_selection.pt", map_location="cpu", weights_only=False)["h"]
    test_features = torch.load(output_dir / "states_newtest.pt", map_location="cpu", weights_only=False)["h"]
    q_train = train.q_pi0.cpu()
    q_selection = selection.q_pi0.cpu()
    q_test = test.q_pi0.cpu()
    centered_train = q_train - q_train.mean(dim=1, keepdim=True)
    centered_selection = q_selection - q_selection.mean(dim=1, keepdim=True)
    centered_test = q_test - q_test.mean(dim=1, keepdim=True)
    torch.manual_seed(seed)
    raw_student = fit_value_student(train_features, q_train, hidden_dim=128, epochs=epochs, validation=(selection_features, q_selection))
    centered_student = fit_value_student(train_features, centered_train, hidden_dim=128, epochs=epochs, validation=(selection_features, centered_selection))
    linear_raw = linear_fit(train_features, q_train)
    linear_centered = linear_fit(train_features, centered_train)
    torch.save({"raw": raw_student, "centered": centered_student, "linear_raw": linear_raw, "linear_centered": linear_centered}, output_dir / "students.pt")
    test_baseline = test.baseline_action
    constant, group, rule_metadata = group_rules(train, test)
    candidate_rank, rank_metadata = candidate_rank_rule(train, test)
    selections = {
        "baseline": test_baseline,
        "random": random_selection(test_baseline.shape[0], test.q_pi0.shape[1], seed, test.q_pi0.device),
        "candidate_rank": candidate_rank,
        "constant": constant,
        "group": group,
        "raw_mlp": predict(raw_student.student, test_features, raw_student.mean, raw_student.std).argmin(dim=-1),
        "centered_mlp": predict(centered_student.student, test_features, centered_student.mean, centered_student.std).argmin(dim=-1),
        "linear_raw": linear_predict(test_features, linear_raw).argmin(dim=-1),
        "linear_centered": linear_predict(test_features, linear_centered).argmin(dim=-1),
    }
    policies = evaluate_policies(test, selections)
    oracle = oracle_metrics(test)
    doc_ids = torch.load(output_dir / "states_newtest.pt", map_location="cpu", weights_only=False)["doc_ids"]
    rows_cpu = torch.arange(test_baseline.shape[0])
    rows = rows_cpu.to(test.q_pi0.device)
    baseline_cost = test.q_pi0[rows, test_baseline].cpu()
    intervals = {}
    for name, selection in selections.items():
        executed = test.q_pi0[rows, selection.to(test.q_pi0.device)].cpu()
        intervals[name] = document_paired_bootstrap(baseline_cost, executed, doc_ids, n_bootstrap=500, seed=seed)
    ledger.student_train_samples = 2 * epochs * train_features.shape[0]
    ledger.student_optimizer_updates = raw_student.optimizer_updates + centered_student.optimizer_updates
    ledger.elapsed_seconds = time.monotonic() - started
    ledger.merge_peak(device)
    summary = {
        "backbone_hash": backbone_hash,
        "config_path": str(config_path),
        "corpus_dir": str(corpus_dir),
        "requested_states": roles,
        "epochs": epochs,
        "seed": seed,
        "cheap_rule_metadata": {**rule_metadata, **rank_metadata},
        "policies": policies,
        "oracle": oracle,
        "paired_document_intervals": intervals,
        "coverage": {role: torch.load(output_dir / f"states_{role}.pt", map_location="cpu", weights_only=False)["stats"] for role in roles},
        "ledger": dict(ledger.__dict__),
        "scope": "four_candidate_document_coverage_screening_not_E02",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "cost_ledger.json").write_text(json.dumps(dict(ledger.__dict__), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-states", type=int, default=8000)
    parser.add_argument("--selection-states", type=int, default=2000)
    parser.add_argument("--calibration-states", type=int, default=2000)
    parser.add_argument("--test-states", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.corpus_dir, args.output_dir, args.device, args.train_states, args.selection_states, args.calibration_states, args.test_states, args.epochs, args.seed)


if __name__ == "__main__":
    main()
