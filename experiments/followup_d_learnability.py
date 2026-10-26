from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

from eval.counterfactual import PathCosts, evaluate_policies
from eval.state_sampling import document_paired_bootstrap
from experiments.r1_closed_loop import source_identity
from experiments.r2_value import candidate_rank_rule, group_rules
from lm_config import LMExperimentConfig
from model import LoopMoE
from train.recovery_common import atomic_json, sha256_file
from train.value_student import fit_value_student, predict, standardization


def subset_costs(costs, indices):
    return PathCosts(
        costs.immediate[indices],
        costs.q_pi0[indices],
        costs.baseline_action[indices],
        costs.actions[indices],
        costs.rounds,
        costs.global_action_ids[indices],
    )


def advantage_targets(costs):
    rows = torch.arange(costs.q_pi0.shape[0])
    return costs.q_pi0 - costs.q_pi0[rows, costs.baseline_action].unsqueeze(1)


def pair_router_features(router_logits, actions):
    pair_logits = router_logits.gather(1, actions.reshape(actions.shape[0], -1)).reshape(actions.shape)
    return torch.stack((pair_logits.sum(dim=-1), pair_logits[..., 0] - pair_logits[..., 1]), dim=-1)


class CandidateAdvantageScorer(nn.Module):
    def __init__(self, h_dim, router_dim, n_global_actions, hidden_dim=120, embedding_dim=16):
        super().__init__()
        self.action_embedding = nn.Embedding(n_global_actions, embedding_dim)
        input_dim = h_dim + router_dim + 2 + embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, router_logits, pair_features, action_ids):
        batch, candidates = action_ids.shape
        embedding = self.action_embedding(action_ids)
        h = h.unsqueeze(1).expand(-1, candidates, -1)
        router_logits = router_logits.unsqueeze(1).expand(-1, candidates, -1)
        features = torch.cat((h, router_logits, pair_features, embedding), dim=-1)
        return self.net(features).squeeze(-1)


def candidate_predict(model, h, router_logits, pair_features, action_ids, normalization):
    h_mean, h_std, router_mean, router_std, pair_mean, pair_std = normalization
    model.eval()
    with torch.no_grad():
        return model((h - h_mean) / h_std, (router_logits - router_mean) / router_std, (pair_features - pair_mean) / pair_std, action_ids)


def fit_candidate(h, router_logits, pair_features, action_ids, targets, *, epochs, seed, hidden_dim=120, batch_size=256, lr=1e-3):
    h_mean, h_std = standardization(h)
    router_mean, router_std = standardization(router_logits)
    pair_mean, pair_std = standardization(pair_features.reshape(pair_features.shape[0], -1))
    pair_mean = pair_mean.reshape(1, pair_features.shape[1], 2)
    pair_std = pair_std.reshape(1, pair_features.shape[1], 2)
    normalization = (h_mean, h_std, router_mean, router_std, pair_mean, pair_std)
    torch.manual_seed(seed)
    model = CandidateAdvantageScorer(h.shape[1], router_logits.shape[1], 28, hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    x = ((h - h_mean) / h_std, (router_logits - router_mean) / router_std, (pair_features - pair_mean) / pair_std)
    history = []
    for epoch in range(epochs):
        order = torch.randperm(h.shape[0], generator=generator)
        total = 0.0
        for start in range(0, h.shape[0], batch_size):
            indices = order[start : start + batch_size]
            prediction = model(x[0][indices], x[1][indices], x[2][indices], action_ids[indices])
            loss = (prediction - targets[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(indices)
        history.append({"epoch": epoch, "train_mse": total / h.shape[0]})
    return {"model": model, "normalization": normalization, "history": history, "optimizer_updates": epochs * ((h.shape[0] + batch_size - 1) // batch_size)}


def router_features(model, states, device):
    layer = model.config.adaptive_layers[-1]
    block = model.blocks[layer]
    pieces = []
    with torch.no_grad():
        for start in range(0, states["h"].shape[0], 256):
            h = states["h"][start : start + 256].to(device)
            pieces.append(block.executor.gate_logits(block.ffn_norm(h)).float().cpu())
    return torch.cat(pieces)


def compact_interval(interval):
    return {key: interval[key] for key in ("estimate", "ci_lo", "ci_hi", "se", "n_documents", "n_rows")}


def selection_metrics(costs, selection, doc_ids, n_bootstrap=500):
    report = evaluate_policies(costs, {"policy": selection})["policy"]
    rows = torch.arange(costs.q_pi0.shape[0])
    interval = document_paired_bootstrap(costs.q_pi0[rows, costs.baseline_action], costs.q_pi0[rows, selection], doc_ids, n_bootstrap=n_bootstrap)
    report["document_interval"] = compact_interval(interval)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-dir", default="runs/repair_2026_10_25/r2_full")
    parser.add_argument("--config", default="configs/e00_m105_b2.json")
    parser.add_argument("--checkpoint", default="runs/e00/m105_b2/latest.pt")
    parser.add_argument("--output-dir", default="runs/followup_2026_10_26/plan_d")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "d_plan.json"
    if plan_path.exists():
        raise FileExistsError(f"frozen Plan D already exists: {plan_path}")
    seeds = [0, 1, 2]
    sizes = [2000, 8000]
    plan = {
        "protocol": "candidate-aware-advantage-bounded-v1",
        "created_before_training": True,
        "source": source_identity(),
        "training_sizes": sizes,
        "student_seeds": seeds,
        "epochs": args.epochs,
        "optimizer": "AdamW lr=0.001 weight_decay=0 batch_size=256 fixed final weights",
        "target": "Q(s,a) - Q(s,a0)",
        "h_only": "512-d raw h -> four candidate advantage outputs, hidden=128",
        "candidate_aware": "raw h + actual action-ID embedding + router logits + pair score/margin -> shared scalar advantage, hidden=120",
        "development_gate": {
            "at_8k_all_three_candidate_seeds": "selection improvement > 0, > paired h-only seed, and > best cheap-rule mean",
            "learning_curve": "mean candidate-aware selection improvement at 8k > 2k",
            "uses_test": False,
        },
        "on_pass": "eligible for new-document expansion and a separately frozen confirmation; not automatic confirmation",
        "on_fail": "stop current backbone/interface learnability expansion; no 32k and no RL",
    }
    atomic_json(plan_path, plan)

    config = LMExperimentConfig.load(args.config)
    device = torch.device(args.device)
    model = LoopMoE(config.model).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    roles = ("controller_train", "selection", "newtest")
    states = {role: torch.load(Path(args.r2_dir) / f"states_{role}.pt", map_location="cpu", weights_only=False) for role in roles}
    router = {role: router_features(model, states[role], device) for role in roles}
    atomic_torch = output_dir / "router_features.pt"
    torch.save(router, atomic_torch)

    results = {}
    saved_models = {}
    for size in sizes:
        train_states = states["controller_train"]
        train_costs = subset_costs(train_states["costs"], torch.arange(size))
        train_h = train_states["h"][:size]
        train_router = router["controller_train"][:size]
        train_actions = train_costs.actions
        train_global = train_costs.global_action_ids
        train_pair = pair_router_features(train_router, train_actions)
        train_target = advantage_targets(train_costs)
        cheap = {}
        constant, group, group_metadata = group_rules(train_costs, states["selection"]["costs"])
        rank, _ = candidate_rank_rule(train_costs, states["selection"]["costs"])
        for name, selection in (("constant_global", constant), ("baseline_group", group), ("candidate_rank", rank)):
            cheap[name] = selection_metrics(states["selection"]["costs"], selection, states["selection"]["doc_ids"])
        best_cheap_name = max(cheap, key=lambda name: cheap[name]["deploy_improvement"])
        best_cheap_mean = cheap[best_cheap_name]["deploy_improvement"]
        size_result = {"cheap": cheap, "best_cheap_name": best_cheap_name, "best_cheap_mean": best_cheap_mean, "seeds": {}}
        for seed in seeds:
            h_student = fit_value_student(train_h, train_target, hidden_dim=128, epochs=args.epochs, seed=seed)
            candidate_student = fit_candidate(train_h, train_router, train_pair, train_global, train_target, epochs=args.epochs, seed=seed)
            saved_models[f"{size}_{seed}"] = {"h_only": h_student, "candidate_aware": candidate_student}
            seed_result = {}
            for role in ("selection", "newtest"):
                role_states = states[role]
                role_costs = role_states["costs"]
                role_router = router[role]
                role_pair = pair_router_features(role_router, role_costs.actions)
                h_prediction = predict(h_student.student, role_states["h"], h_student.mean, h_student.std)
                candidate_prediction = candidate_predict(candidate_student["model"], role_states["h"], role_router, role_pair, role_costs.global_action_ids, candidate_student["normalization"])
                seed_result[role] = {
                    "h_only": selection_metrics(role_costs, h_prediction.argmin(dim=-1), role_states["doc_ids"]),
                    "candidate_aware": selection_metrics(role_costs, candidate_prediction.argmin(dim=-1), role_states["doc_ids"]),
                    "h_only_mse": float((h_prediction - advantage_targets(role_costs)).square().mean()),
                    "candidate_aware_mse": float((candidate_prediction - advantage_targets(role_costs)).square().mean()),
                }
            seed_result["parameters"] = {
                "h_only": sum(parameter.numel() for parameter in h_student.student.parameters()),
                "candidate_aware": sum(parameter.numel() for parameter in candidate_student["model"].parameters()),
            }
            seed_result["optimizer_updates"] = {"h_only": h_student.optimizer_updates, "candidate_aware": candidate_student["optimizer_updates"]}
            size_result["seeds"][str(seed)] = seed_result
        results[str(size)] = size_result
        torch.save(saved_models, output_dir / "d_students.pt")

    eight = results["8000"]["seeds"]
    two_mean = sum(results["2000"]["seeds"][str(seed)]["selection"]["candidate_aware"]["deploy_improvement"] for seed in seeds) / len(seeds)
    eight_mean = sum(eight[str(seed)]["selection"]["candidate_aware"]["deploy_improvement"] for seed in seeds) / len(seeds)
    per_seed_pass = {
        str(seed): (
            eight[str(seed)]["selection"]["candidate_aware"]["deploy_improvement"] > 0
            and eight[str(seed)]["selection"]["candidate_aware"]["deploy_improvement"] > eight[str(seed)]["selection"]["h_only"]["deploy_improvement"]
            and eight[str(seed)]["selection"]["candidate_aware"]["deploy_improvement"] > results["8000"]["best_cheap_mean"]
        )
        for seed in seeds
    }
    gate_pass = all(per_seed_pass.values()) and eight_mean > two_mean
    summary = {
        "scope": "development diagnostic only; newtest previously viewed and never used for expansion gate",
        "results": results,
        "development_gate": {
            "per_seed_pass": per_seed_pass,
            "candidate_aware_mean_improvement_2k": two_mean,
            "candidate_aware_mean_improvement_8k": eight_mean,
            "learning_curve_pass": eight_mean > two_mean,
            "pass": gate_pass,
        },
        "next_step": "eligible_for_separate_expansion_and_confirmation" if gate_pass else "stop_no_32k_no_RL",
        "d_plan_sha256": sha256_file(plan_path),
    }
    atomic_json(output_dir / "d_summary.json", summary)
    atomic_json(output_dir / "run_manifest.json", {
        "source": source_identity(),
        "argv": sys.argv,
        "d_plan_sha256": sha256_file(plan_path),
        "d_summary_sha256": sha256_file(output_dir / "d_summary.json"),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "router_features_sha256": sha256_file(atomic_torch),
    })
    print(json.dumps(summary["development_gate"], indent=2))


if __name__ == "__main__":
    main()
