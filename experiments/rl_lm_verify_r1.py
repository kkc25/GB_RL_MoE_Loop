from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from data.packed_lm import PackedLMDataset
from eval.counterfactual import cache_last_ffn_input
from lm_config import LMExperimentConfig
from model import LoopMoE
from model.top2 import baseline_top2_pairs, pair_to_action
from rl_lm.actions import make_pair_table
from rl_lm.actor import PairActor
from rl_lm.executor import CANONICAL_BACKEND, SPARSE_BACKEND, MERGED_BACKEND
from rl_lm.inference import CanonicalTeacherPolicy, forward_with_last_ffn_policy
from rl_lm.selector import canonical_pair
from train.common import autocast_context
from train.recovery_common import atomic_json, sha256_file


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="runs/next_2026_10_26/baseline_100m/loop_r1.json")
    parser.add_argument("--checkpoint", default="runs/next_2026_10_26/baseline_100m/loop_r1/latest.pt")
    parser.add_argument("--data-dir", default="runs/data/fineweb_edu_100m")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sequences", type=int, default=2)
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None,
                        help="Default: historical online_r1 for v1; online_r1_sparse_merged_v2 for v2")
    parser.add_argument("--deployment-backend", choices=(SPARSE_BACKEND, MERGED_BACKEND), default=SPARSE_BACKEND,
                        help="Opt-in sparse executor to compare against the fixed-shape canonical reference")
    return parser


def main():
    args = build_parser().parse_args()
    if args.sequences <= 0 or args.max_positions <= 0:
        raise ValueError("sequences and max-positions must be positive")
    default_output = "runs/rl_lm_pair_v1/task_b/online_r1"
    if args.deployment_backend != SPARSE_BACKEND:
        default_output += "_" + args.deployment_backend
    output = Path(args.output_dir or default_output)
    output.mkdir(parents=True, exist_ok=True)
    config = LMExperimentConfig.load(args.config)
    device = torch.device(args.device)
    model = LoopMoE(config.model).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    model.requires_grad_(False)
    dataset = PackedLMDataset(args.data_dir, args.split)
    if args.sequences > len(dataset):
        raise ValueError("sequences exceeds available dataset rows")
    input_ids = torch.from_numpy(__import__("numpy").stack([dataset[index]["input_ids"] for index in range(args.sequences)])).to(device)
    segment_ids = torch.from_numpy(__import__("numpy").stack([dataset[index]["segment_ids"] for index in range(args.sequences)])).to(device)
    labels = torch.from_numpy(__import__("numpy").stack([dataset[index]["labels"] for index in range(args.sequences)])).to(device)
    if not bool((labels != -100).any()):
        raise ValueError("diagnostic selection contains no valid target positions")
    with torch.no_grad(), autocast_context(device, "bfloat16"):
        legacy = model(input_ids, labels, depth=1, segment_ids=segment_ids)
    canonical = forward_with_last_ffn_policy(model, CanonicalTeacherPolicy(), input_ids, segment_ids, max_calls=1, backend=CANONICAL_BACKEND, return_trace=True, compute_dtype="bfloat16", chunk=64, head_chunk=64)
    sparse = forward_with_last_ffn_policy(model, CanonicalTeacherPolicy(), input_ids, segment_ids, max_calls=1, backend=args.deployment_backend, return_trace=True, compute_dtype="bfloat16", chunk=64, head_chunk=64)
    actor = PairActor(2 * model.config.hidden_size + 2 * 8 + 4, train_stop=False).to(device)
    initial_actor = forward_with_last_ffn_policy(model, actor, input_ids, segment_ids, max_calls=1, backend=args.deployment_backend, return_trace=True, compute_dtype="bfloat16", chunk=64, head_chunk=64)
    with torch.no_grad(), autocast_context(device, "bfloat16"):
        prefix = cache_last_ffn_input(model, input_ids, segment_ids, depth=1)
    block = model.blocks[model.config.adaptive_layers[-1]]
    rows = []
    valid_rows, valid_cols = torch.nonzero(labels != -100, as_tuple=True)
    for index in range(min(args.max_positions, int(valid_rows.numel()))):
        b, s = int(valid_rows[index]), int(valid_cols[index])
        h = prefix[b, s].float()
        with torch.no_grad(), autocast_context(device, "bfloat16"):
            gate = block.executor.gate_logits(block.ffn_norm(h.unsqueeze(0))).float()
        margin = float((gate.topk(2).values[0, 0] - gate.topk(2).values[0, 1]).item())
        canonical_pair_index = int(canonical_pair(gate)[0])
        pair_difference = not bool(torch.equal(baseline_top2_pairs(gate)[0], make_pair_table().to(device)[canonical_pair_index]))
        row = {
            "sequence": b,
            "position": s,
            "target": int(labels[b, s]),
            "gate_margin": margin,
            "gate_tie": margin == 0.0,
            "pair_difference": pair_difference,
            "legacy_ce": float(F.cross_entropy(legacy.logits[b, s].float().unsqueeze(0), labels[b, s].unsqueeze(0))),
            "canonical_ce": float(F.cross_entropy(canonical.logits[b, s].float().unsqueeze(0), labels[b, s].unsqueeze(0))),
            "sparse_ce": float(F.cross_entropy(sparse.logits[b, s].float().unsqueeze(0), labels[b, s].unsqueeze(0))),
            "initial_actor_ce": float(F.cross_entropy(initial_actor.logits[b, s].float().unsqueeze(0), labels[b, s].unsqueeze(0))),
            "canonical_hidden_abs": float((canonical.final_hidden[b, s] - legacy.hidden[b, s].float()).abs().max()),
            "sparse_canonical_hidden_abs": float((sparse.final_hidden[b, s] - canonical.final_hidden[b, s]).abs().max()),
            "sparse_action": int(sparse.actions[b, s, 0]),
            "sparse_hidden_abs": float((sparse.final_hidden[b, s] - legacy.hidden[b, s].float()).abs().max()),
            "initial_actor_action": int(initial_actor.actions[b, s, 0]),
            "canonical_action": int(canonical.actions[b, s, 0]),
            "legacy_action": int(pair_to_action(baseline_top2_pairs(gate)[0].unsqueeze(0), experts=8)[0]) + 1,
        }
        rows.append(row)
    errors = {
        "canonical_minus_legacy_ce_abs": [abs(row["canonical_ce"] - row["legacy_ce"]) for row in rows],
        "sparse_minus_canonical_ce_abs": [abs(row["sparse_ce"] - row["canonical_ce"]) for row in rows],
        "sparse_canonical_hidden_abs": [row["sparse_canonical_hidden_abs"] for row in rows],
        "sparse_minus_legacy_ce_abs": [abs(row["sparse_ce"] - row["legacy_ce"]) for row in rows],
        "initial_actor_minus_canonical_ce_abs": [abs(row["initial_actor_ce"] - row["canonical_ce"]) for row in rows],
    }
    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "sequences": args.sequences,
        "positions": len(rows),
        "gate_ties": sum(row["gate_tie"] for row in rows),
        "canonical_legacy_pair_differences": sum(row["pair_difference"] for row in rows),
        "initial_actor_canonical_action_differences": sum(row["initial_actor_action"] != row["canonical_action"] for row in rows),
        # Retain the historical backend field (a readout identity) for readers.
        "backend": "bf16_linear_fp32_ce_v1",
        "head_contract": "bf16_linear_fp32_ce_v1",
        "canonical_backend": CANONICAL_BACKEND,
        "deployment_backend": args.deployment_backend,
        "sparse_canonical_action_differences": sum(row["sparse_action"] != row["canonical_action"] for row in rows),
        "numerical_contract": {
            name: {
                "mean": sum(values) / len(values),
                "max": max(values),
                "p95": sorted(values)[int(0.95 * (len(values) - 1))],
            }
            for name, values in errors.items()
        },
    }
    atomic_json(output / "summary.json", summary)
    with (output / "per_state.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
