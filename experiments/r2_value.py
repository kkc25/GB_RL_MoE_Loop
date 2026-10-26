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


def group_rules(train_costs, train_baseline, test_baseline, actions):
    means = train_costs.q_pi0.mean(dim=0)
    constant_global = int(train_costs.global_action_ids[0, means.argmin()])
    group_means = {}
    for global_id in actions.unique():
        mask = train_baseline == global_id
        if bool(mask.any()):
            group_means[int(global_id)] = int(train_costs.global_action_ids[mask][0, train_costs.q_pi0[mask].mean(dim=0).argmin()])
    constant = test_baseline.clone()
    group = test_baseline.clone()
    lookup = torch.full((int(actions.max()) + 1,), -1, dtype=torch.long)
    lookup[actions] = torch.arange(actions.shape[1]).repeat(actions.shape[0]).reshape(actions.shape)
    for row, baseline in enumerate(test_baseline):
        constant[row] = lookup[constant_global]
        group[row] = lookup[group_means.get(int(baseline), int(baseline))]
    return constant, group


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
    test_baseline = test.baseline_action.cpu()
    constant, group = group_rules(train, train.baseline_action.cpu(), test_baseline, test.global_action_ids.cpu())
    selections = {
        "baseline": test_baseline,
        "random": random_selection(test_baseline.shape[0], test.q_pi0.shape[1], seed, test.q_pi0.device),
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
    rows = torch.arange(test_baseline.shape[0])
    baseline_cost = test.q_pi0[rows, test_baseline].cpu()
    intervals = {}
    for name, selection in selections.items():
        executed = test.q_pi0[rows, selection].cpu()
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
