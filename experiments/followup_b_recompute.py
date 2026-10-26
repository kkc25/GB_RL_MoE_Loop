from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

from eval.counterfactual import evaluate_policies, oracle_metrics
from eval.state_sampling import document_paired_bootstrap
from experiments.r1_closed_loop import source_identity
from experiments.r2_value import baseline_global_ids, candidate_rank_rule, group_rules, linear_predict
from experiments.r4_quality import depth_metrics
from lm_config import LMExperimentConfig
from model import LoopMoE
from train.recovery_common import atomic_json, sha256_file
from train.value_student import predict


def validate_states_identity(states, *, checkpoint_path=None, corpus_dir=None):
    """Reject cached states whose recorded identity does not match the frozen inputs.

    Each ``states_{role}.pt`` carries ``backbone_hash`` (sha256 of the backbone
    checkpoint the states were cached with) and ``corpus_doc_map_sha256``
    (sha256 of the doc-map of the corpus split the states were sampled from).
    Both are verified before any model forward or downstream computation
    consumes the states, so a wrong checkpoint or corpus is rejected at the
    entry point.  Returns the verified identity digests for the run manifest.
    """
    identity = {}
    backbone_hash = None
    if checkpoint_path is not None:
        backbone_hash = sha256_file(checkpoint_path)
        identity["checkpoint_sha256"] = backbone_hash
    corpus_manifest = None
    if corpus_dir is not None:
        manifest_path = Path(corpus_dir) / "manifest.json"
        corpus_manifest = json.loads(manifest_path.read_text())
        identity["corpus_manifest_sha256"] = sha256_file(manifest_path)
    for role, state in states.items():
        recorded = state.get("backbone_hash")
        if backbone_hash is not None and recorded != backbone_hash:
            raise ValueError(f"{role} states backbone hash {recorded} does not match checkpoint {backbone_hash}")
        if corpus_manifest is not None:
            expected = corpus_manifest["splits"][role]["doc_map_sha256"]
            recorded_doc_map = state.get("corpus_doc_map_sha256")
            if recorded_doc_map != expected:
                raise ValueError(f"{role} states doc-map hash {recorded_doc_map} does not match corpus manifest {expected}")
    return identity


def validate_backbone_config(config, checkpoint):
    """Reject a config whose model section differs from the checkpoint's embedded one."""
    embedded = checkpoint.get("config", {}).get("model")
    if embedded is not None and asdict(config.model) != embedded:
        raise ValueError("config model section does not match the checkpoint's embedded config")


def analytic_mix(stop, cont, probability):
    if not 0 <= probability <= 1:
        raise ValueError("mixture probability must be in [0, 1]")
    return (1 - probability) * stop + probability * cont


def selection_diagnostics(costs, selection, requested_global=None):
    rows = torch.arange(selection.shape[0], device=selection.device)
    actual = costs.global_action_ids[rows, selection]
    result = {
        "actual_global_action_counts": {str(int(action)): int((actual == action).sum()) for action in actual.unique()},
    }
    if requested_global is not None:
        if isinstance(requested_global, int):
            present = costs.global_action_ids == requested_global
        else:
            requested = torch.as_tensor(requested_global, device=costs.global_action_ids.device)
            present = costs.global_action_ids == requested.reshape(-1, 1)
        result["fallback_fraction"] = float((~present.any(dim=1)).float().mean())
    return result


def run_r2(input_dir, output_dir, n_bootstrap, *, checkpoint_path=None, corpus_dir=None):
    input_dir = Path(input_dir)
    states = {role: torch.load(input_dir / f"states_{role}.pt", map_location="cpu", weights_only=False) for role in ("controller_train", "newtest")}
    validate_states_identity(states, checkpoint_path=checkpoint_path, corpus_dir=corpus_dir)
    train, test = states["controller_train"]["costs"], states["newtest"]["costs"]
    students = torch.load(input_dir / "students.pt", map_location="cpu", weights_only=False)
    features = states["newtest"]["h"]
    constant, group, group_metadata = group_rules(train, test)
    rank, rank_metadata = candidate_rank_rule(train, test)
    selections = {
        "baseline": test.baseline_action,
        "candidate_rank": rank,
        "constant_global": constant,
        "baseline_group": group,
        "raw_mlp": predict(students["raw"].student, features, students["raw"].mean, students["raw"].std).argmin(dim=-1),
        "centered_mlp": predict(students["centered"].student, features, students["centered"].mean, students["centered"].std).argmin(dim=-1),
        "linear_raw": linear_predict(features, students["linear_raw"]).argmin(dim=-1),
        "linear_centered": linear_predict(features, students["linear_centered"]).argmin(dim=-1),
    }
    policies = evaluate_policies(test, selections)
    rows = torch.arange(test.q_pi0.shape[0])
    baseline_cost = test.q_pi0[rows, test.baseline_action]
    uniform_cost = test.q_pi0.mean(dim=1)
    policies["uniform_candidate_expectation"] = {
        "mean_cost": float(uniform_cost.mean()),
        "mean_immediate_cost": float(test.immediate.mean()),
        "deploy_improvement": float((baseline_cost - uniform_cost).mean()),
        "deploy_improvement_se": None,
    }
    intervals = {}
    for name in (*selections, "uniform_candidate_expectation"):
        executed = uniform_cost if name == "uniform_candidate_expectation" else test.q_pi0[rows, selections[name]]
        intervals[name] = document_paired_bootstrap(baseline_cost, executed, states["newtest"]["doc_ids"], n_bootstrap=n_bootstrap)
    diagnostics = {
        "candidate_rank": selection_diagnostics(test, rank),
        "constant_global": selection_diagnostics(test, constant, group_metadata["constant_global_action"]),
        "baseline_group": selection_diagnostics(test, group, group_metadata["group_global_actions"]),
    }
    summary = {
        "scope": "corrected cheap-baseline diagnostic on previously viewed newtest; not confirmation",
        "policies": policies,
        "paired_document_intervals": intervals,
        "oracle": oracle_metrics(test),
        "rule_metadata": group_metadata | rank_metadata,
        "selection_diagnostics": diagnostics,
        "n_bootstrap": n_bootstrap,
    }
    atomic_json(Path(output_dir) / "r2_summary.json", summary)
    torch.save({"selections": selections, "uniform_cost": uniform_cost}, Path(output_dir) / "r2_selections.pt")
    return summary


def run_r4(config_path, checkpoint_path, corpus_dir, r2_dir, r4_input_dir, output_dir, device_name, n_bootstrap):
    config = LMExperimentConfig.load(config_path)
    device = torch.device(device_name)
    model = LoopMoE(config.model).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_backbone_config(config, checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    states = {role: torch.load(Path(r2_dir) / f"states_{role}.pt", map_location="cpu", weights_only=False) for role in ("calibration", "newtest")}
    validate_states_identity(states, checkpoint_path=checkpoint_path, corpus_dir=corpus_dir)
    test = states["newtest"]
    costs = test["costs"]
    metrics = depth_metrics(model, states["newtest"], (1, 2, 3, 4), device, config.train.dtype)
    old_summary = json.loads((Path(r4_input_dir) / "summary.json").read_text())
    tau = float(old_summary["direct_delta_student"]["tau"])
    budget = 2.0
    student_payload = torch.load(Path(r4_input_dir) / "student.pt", map_location="cpu", weights_only=False)["student"]
    predicted = predict(student_payload.student, test["h"], student_payload.mean, student_payload.std).squeeze(1)
    calibration_predicted = predict(student_payload.student, states["calibration"]["h"], student_payload.mean, student_payload.std).squeeze(1)
    continue_mask = predicted > tau
    calibration_continue = calibration_predicted > tau
    rows = torch.arange(costs.q_pi0.shape[0])
    stop = costs.immediate[rows, costs.baseline_action]
    cont = costs.q_pi0[rows, costs.baseline_action]
    dynamic = torch.where(continue_mask, cont, stop)
    p_actual = float(continue_mask.float().mean())
    p_nominal = (budget - 1) / 3
    matched = analytic_mix(stop, cont, p_actual)
    nominal = analytic_mix(stop, cont, p_nominal)
    fixed_r2 = metrics[2]["losses"]
    summary = {
        "scope": "descriptive actual-cost-matched diagnostic on previously viewed newtest; mixture interval conditions on fixed observed p",
        "fixed_depths": {str(depth): {"nll": metrics[depth]["nll"]} for depth in (1, 2, 3, 4)},
        "direct_delta": {
            "tau": tau,
            "nll": float(dynamic.mean()),
            "mean_rounds": float(1 + 3 * continue_mask.float().mean()),
            "continue_fraction": p_actual,
            "calibration_continue_fraction": float(calibration_continue.float().mean()),
            "paired_bootstrap_vs_r1": document_paired_bootstrap(stop, dynamic, test["doc_ids"], n_bootstrap=n_bootstrap),
            "paired_bootstrap_vs_r2": document_paired_bootstrap(fixed_r2, dynamic, test["doc_ids"], n_bootstrap=n_bootstrap),
        },
        "actual_cost_matched_mix": {
            "probability": p_actual,
            "mean_rounds": 1 + 3 * p_actual,
            "nll": float(matched.mean()),
            "dynamic_minus_mix": float((dynamic - matched).mean()),
            "paired_bootstrap_dynamic_vs_mix": document_paired_bootstrap(matched, dynamic, test["doc_ids"], n_bootstrap=n_bootstrap),
            "interval_note": "conditions on the observed test continue fraction; descriptive, not a frozen calibration probability",
        },
        "nominal_budget_mix": {
            "probability": p_nominal,
            "mean_rounds": 1 + 3 * p_nominal,
            "nll": float(nominal.mean()),
            "paired_bootstrap_dynamic_vs_mix": document_paired_bootstrap(nominal, dynamic, test["doc_ids"], n_bootstrap=n_bootstrap),
        },
        "fixed_r1_minus_r2": document_paired_bootstrap(metrics[1]["losses"], metrics[2]["losses"], test["doc_ids"], n_bootstrap=n_bootstrap),
        "n_bootstrap": n_bootstrap,
    }
    output_dir = Path(output_dir)
    atomic_json(output_dir / "r4_summary.json", summary)
    torch.save({"depth_losses": {depth: metrics[depth]["losses"] for depth in (1, 2, 3, 4)}, "predicted": predicted, "continue_mask": continue_mask, "dynamic": dynamic, "matched_mix": matched, "nominal_mix": nominal}, output_dir / "r4_losses.pt")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-dir", default="runs/repair_2026_10_25/r2_full")
    parser.add_argument("--r4-input-dir", default="runs/repair_2026_10_25/r4_full")
    parser.add_argument("--config", default="configs/e00_m105_b2.json")
    parser.add_argument("--checkpoint", default="runs/e00/m105_b2/latest.pt")
    parser.add_argument("--corpus-dir", default="runs/repair_2026_10_25/controller_corpus")
    parser.add_argument("--output-dir", default="runs/followup_2026_10_26/plan_b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    r2 = run_r2(args.r2_dir, output_dir, args.bootstrap, checkpoint_path=args.checkpoint, corpus_dir=args.corpus_dir)
    r4 = run_r4(args.config, args.checkpoint, args.corpus_dir, args.r2_dir, args.r4_input_dir, output_dir, args.device, args.bootstrap)
    r2_dir = Path(args.r2_dir)
    manifest = {
        "source": source_identity(),
        "argv": sys.argv,
        "config_sha256": sha256_file(args.config),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "corpus_manifest_sha256": sha256_file(Path(args.corpus_dir) / "manifest.json"),
        "states_sha256": {
            role: sha256_file(r2_dir / f"states_{role}.pt")
            for role in ("controller_train", "selection", "calibration", "newtest")
            if (r2_dir / f"states_{role}.pt").exists()
        },
        "students_sha256": sha256_file(r2_dir / "students.pt"),
        "r4_student_sha256": sha256_file(Path(args.r4_input_dir) / "student.pt"),
        "r2_summary_sha256": sha256_file(output_dir / "r2_summary.json"),
        "r4_summary_sha256": sha256_file(output_dir / "r4_summary.json"),
    }
    atomic_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps({"r2": r2["policies"], "r4_dynamic": r4["direct_delta"]["nll"], "r4_matched_mix": r4["actual_cost_matched_mix"]}, indent=2))


if __name__ == "__main__":
    main()
