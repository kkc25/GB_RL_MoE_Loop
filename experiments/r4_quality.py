from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from eval.counterfactual import counterfactual_path_costs
from eval.state_sampling import document_paired_bootstrap
from lm_config import LMExperimentConfig
from model import LoopMoE
from train.recovery_common import sha256_file
from train.value_student import fit_value_student, predict


def depth_metrics(model, states, depths, device, dtype):
    costs = states["costs"]
    rows = torch.arange(costs.baseline_action.shape[0])
    baseline_pairs = costs.actions[rows, costs.baseline_action].unsqueeze(1).to(device)
    metrics = {}
    for depth in depths:
        if depth == 1:
            losses = costs.immediate[rows, costs.baseline_action].cpu()
        elif depth == costs.rounds:
            losses = costs.q_pi0[rows, costs.baseline_action].cpu()
        else:
            measured = counterfactual_path_costs(
                model,
                states["h"].to(device),
                states["targets"].to(device),
                rounds=depth,
                actions=baseline_pairs,
                dtype=dtype,
            )
            losses = measured.q_pi0[:, 0].cpu()
        metrics[depth] = {"nll": losses.mean().item(), "losses": losses}
    return metrics


def direct_delta(costs):
    rows = torch.arange(costs.q_pi0.shape[0])
    return costs.immediate[rows, costs.baseline_action] - costs.q_pi0[rows, costs.baseline_action]


def threshold_for_budget(predicted, stop, cont, budget, calibration_nll):
    candidates = torch.quantile(predicted, torch.linspace(0, 1, 201))
    best = None
    for tau in candidates:
        continue_mask = predicted > tau
        rounds = 1 + 3 * continue_mask.long()
        nll = torch.where(continue_mask, cont, stop).mean()
        mean_rounds = rounds.float().mean()
        if mean_rounds <= budget and (best is None or nll < best[0]):
            best = (nll.item(), tau.item(), mean_rounds.item())
    if best is None:
        raise ValueError("no feasible threshold")
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--r2-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    config = LMExperimentConfig.load(args.config)
    device = torch.device(args.device)
    model = LoopMoE(config.model).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    corpus_manifest = json.loads((Path(args.corpus_dir) / "manifest.json").read_text())
    backbone_hash = sha256_file(args.checkpoint)
    state_splits = {
        role: torch.load(Path(args.r2_dir) / f"states_{role}.pt", map_location="cpu", weights_only=False)
        for role in ("controller_train", "selection", "calibration", "newtest")
    }
    for role, state in state_splits.items():
        expected_hash = corpus_manifest["splits"][role]["doc_map_sha256"]
        if state.get("backbone_hash") != backbone_hash:
            raise ValueError(f"{role} states do not match the checkpoint; regenerate R2 states")
        if state.get("corpus_doc_map_sha256") != expected_hash:
            raise ValueError(f"{role} states do not match the current corpus doc_map; regenerate R2 states")
    states = state_splits["newtest"]
    metrics = depth_metrics(model, states, (1, 2, 3, 4), device, config.train.dtype)
    costs = {role: state["costs"] for role, state in state_splits.items()}
    train_delta = direct_delta(costs["controller_train"])
    selection_delta = direct_delta(costs["selection"])
    calibration_delta = direct_delta(costs["calibration"])
    test_delta = direct_delta(costs["newtest"])
    train_features = state_splits["controller_train"]["h"]
    selection_features = state_splits["selection"]["h"]
    test_features = states["h"]
    student = fit_value_student(train_features, train_delta.unsqueeze(1), hidden_dim=128, epochs=args.epochs, validation=(selection_features, selection_delta.unsqueeze(1)))
    predicted = predict(student.student, test_features, student.mean, student.std).squeeze(1)
    stop = costs["newtest"].immediate[torch.arange(test_delta.shape[0]), costs["newtest"].baseline_action]
    cont = costs["newtest"].q_pi0[torch.arange(test_delta.shape[0]), costs["newtest"].baseline_action]
    torch.testing.assert_close(metrics[1]["losses"], stop, rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(metrics[4]["losses"], cont, rtol=1e-3, atol=2e-3)
    fixed_r2 = metrics[2]["losses"]
    calibration_predicted = predict(student.student, state_splits["calibration"]["h"], student.mean, student.std).squeeze(1)
    tau = threshold_for_budget(calibration_predicted, costs["calibration"].immediate[torch.arange(calibration_delta.shape[0]), costs["calibration"].baseline_action], costs["calibration"].q_pi0[torch.arange(calibration_delta.shape[0]), costs["calibration"].baseline_action], args.budget, None)
    continue_mask = predicted > tau
    dynamic_cost = torch.where(continue_mask, cont, stop)
    rounds = 1 + 3 * continue_mask.long()
    random_generator = torch.Generator().manual_seed(0)
    random_mask = torch.rand(rounds.shape, generator=random_generator) < (args.budget - 1) / 3
    random_cost = torch.where(random_mask, cont, stop)
    doc_ids = states["doc_ids"]
    summary = {
        "fixed_depths": {str(depth): {"nll": metrics[depth]["nll"]} for depth in (1, 2, 3, 4)},
        "direct_delta_student": {
            "tau": tau,
            "nll": dynamic_cost.mean().item(),
            "mean_rounds": rounds.float().mean().item(),
            "delta_vs_fixed_r1": (dynamic_cost - stop).mean().item(),
            "delta_vs_fixed_r2": (dynamic_cost - fixed_r2).mean().item(),
            "delta_vs_fixed_r4": (dynamic_cost - cont).mean().item(),
            "paired_bootstrap_vs_r1": document_paired_bootstrap(stop, dynamic_cost, doc_ids, n_bootstrap=500),
            "paired_bootstrap_vs_r2": document_paired_bootstrap(fixed_r2, dynamic_cost, doc_ids, n_bootstrap=500),
        },
        "random_depth_mix": {
            "nll": random_cost.mean().item(),
            "mean_rounds": (1 + 3 * random_mask.long()).float().mean().item(),
            "paired_bootstrap_vs_r1": document_paired_bootstrap(stop, random_cost, doc_ids, n_bootstrap=500),
            "paired_bootstrap_vs_r2": document_paired_bootstrap(fixed_r2, random_cost, doc_ids, n_bootstrap=500),
        },
        "fixed_r1_nll": stop.mean().item(),
        "fixed_r2_nll": fixed_r2.mean().item(),
        "fixed_r4_nll": cont.mean().item(),
        "corpus_doc_map_sha256": {role: state["corpus_doc_map_sha256"] for role, state in state_splits.items()},
        "scope": "two_depth_direct_delta_not_multi_depth_halting",
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"student": student}, Path(args.output_dir) / "student.pt")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
