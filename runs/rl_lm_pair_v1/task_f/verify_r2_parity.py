"""Task card F: real-checkpoint R2 online parity (spec T22 R2 / 15.1 / 16.1).

On the real frozen 100M backbone and the new corpus selection split:

1. canonical_pair_r2 online forward (max_calls=2, sparse_merged_v1, BF16
   production contract) is compared per position against an independent
   single-row two-round reference (re-observe the gate after round one, take
   the canonical pair each round, production head readout).
2. STOP closed loop on the real checkpoint: the warmstarted R2 initial actor
   (P1 best weights, zeroed stop head) must execute exactly one round and
   then voluntarily STOP; its hidden/logits must be bitwise identical to the
   P1 best actor run with max_calls=1 (h unchanged after STOP).
3. Legacy depth=1 / depth=2 production forwards are reported as references.

Auxiliary diagnostic script; it changes no algorithm and writes only into
this task directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data.packed_lm import PackedLMDataset
from eval.counterfactual import cache_last_ffn_input
from lm_config import LMExperimentConfig
from model import LoopMoE
from rl_lm.actor import PairActor
from rl_lm.inference import CanonicalTeacherPolicy, forward_with_last_ffn_policy
from rl_lm.readout import production_logits
from rl_lm.selector import canonical_pair
from train.common import autocast_context
from train.recovery_common import atomic_json, sha256_file

FEATURE_DIM = 2 * 512 + 2 * 8 + 4


def load_actor(path: Path, *, train_stop: bool, device) -> PairActor:
    """Load an actor checkpoint as stored (full state dict, stop head included)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = PairActor(int(payload["feature_dim"]), train_stop=train_stop)
    actor.load_state_dict(payload["actor_state_dict"])
    return actor.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="runs/next_2026_10_26/baseline_100m/loop_r1.json")
    parser.add_argument("--checkpoint", default="runs/next_2026_10_26/baseline_100m/loop_r1/latest.pt")
    parser.add_argument("--p1-best", default="runs/rl_lm_pair_v1/r1_seed17/checkpoint_best.pt")
    parser.add_argument("--r2-initial", default="runs/rl_lm_pair_v1/r2_lambda0_seed17/checkpoint_initial.pt",
                        help="R2 run's own warmstarted initial actor (preferred over re-deriving warmstart)")
    parser.add_argument("--data-dir", default="data/rl_lm_pair_v1/corpus")
    parser.add_argument("--split", default="selection")
    parser.add_argument("--sequences", type=int, default=2)
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="runs/rl_lm_pair_v1/task_f/verify_r2")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = LMExperimentConfig.load(args.config)
    device = torch.device(args.device)
    model = LoopMoE(config.model).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    model.requires_grad_(False)

    dataset = PackedLMDataset(args.data_dir, args.split)
    input_ids = torch.from_numpy(np.stack([dataset[i]["input_ids"] for i in range(args.sequences)])).to(device)
    segment_ids = torch.from_numpy(np.stack([dataset[i]["segment_ids"] for i in range(args.sequences)])).to(device)
    labels = torch.from_numpy(np.stack([dataset[i]["labels"] for i in range(args.sequences)])).to(device)

    def online(policy, max_calls):
        return forward_with_last_ffn_policy(
            model, policy, input_ids, segment_ids,
            max_calls=max_calls, backend="sparse_merged_v1", return_trace=True,
            compute_dtype="bfloat16", chunk=64, head_chunk=64,
        )

    with torch.no_grad(), autocast_context(device, "bfloat16"):
        legacy_r1 = model(input_ids, labels, depth=1, segment_ids=segment_ids)
        legacy_r2 = model(input_ids, labels, depth=2, segment_ids=segment_ids)
    canonical_r1 = online(CanonicalTeacherPolicy(), 1)
    canonical_r2 = online(CanonicalTeacherPolicy(), 2)
    p1_best_actor = load_actor(Path(args.p1_best), train_stop=False, device=device)
    r2_warm_actor = load_actor(Path(args.r2_initial), train_stop=True, device=device)
    p1_best_online = online(p1_best_actor, 1)
    r2_warm_online = online(r2_warm_actor, 2)

    # STOP closed loop on the real checkpoint: warmstarted R2 initial actor
    # must run exactly one pair round and then voluntarily STOP, bitwise.
    stop_checks = {
        "rounds_all_one": bool((r2_warm_online.rounds == 1).all()),
        "slot1_actions_all_stop": bool((r2_warm_online.actions[:, :, 1] == 0).all()),
        "hidden_bitwise_p1_best": bool(torch.equal(r2_warm_online.final_hidden, p1_best_online.final_hidden)),
        "logits_bitwise_p1_best": bool(torch.equal(r2_warm_online.logits, p1_best_online.logits)),
    }

    # Independent single-row two-round reference for canonical_pair_r2.
    executor_block = model.blocks[model.config.adaptive_layers[-1]]
    from rl_lm.executor import make_executor

    executor = make_executor(model, "sparse_merged_v1", chunk=64, compute_dtype="bfloat16")
    with torch.no_grad(), autocast_context(device, "bfloat16"):
        prefix = cache_last_ffn_input(model, input_ids, segment_ids, depth=1)
    valid_rows, valid_cols = torch.nonzero(labels != -100, as_tuple=True)
    rows = []
    for index in range(min(args.max_positions, int(valid_rows.numel()))):
        b, s = int(valid_rows[index]), int(valid_cols[index])
        h0 = prefix[b, s].float().reshape(1, -1)
        h = h0.clone()
        rounds = torch.zeros(1, dtype=torch.int64, device=device)
        previous = torch.full((1, 2), -1, dtype=torch.int64, device=device)
        actions = []
        for _ in range(2):
            observation = executor.observe(h, h0, rounds, previous, max_calls=2)
            pair = canonical_pair(observation.z, executor.table)
            action = int(pair[0]) + 1
            actions.append(action)
            h = executor.step_pair(h, observation, torch.tensor([action], dtype=torch.int64, device=device))
            rounds = rounds + 1
            previous = executor.table[[int(pair[0])]]
        reference_logits = production_logits(model, h, chunk=64, compute_dtype="bfloat16")
        target = labels[b, s].reshape(1)
        rows.append({
            "sequence": b,
            "position": s,
            "online_actions": [int(canonical_r2.actions[b, s, 0]), int(canonical_r2.actions[b, s, 1])],
            "reference_actions": actions,
            "online_rounds": int(canonical_r2.rounds[b, s]),
            "hidden_abs_max": float((canonical_r2.final_hidden[b, s].float() - h.reshape(-1).float()).abs().max()),
            "online_ce": float(F.cross_entropy(canonical_r2.logits[b, s].float().reshape(1, -1), target)),
            "reference_ce": float(F.cross_entropy(reference_logits.reshape(-1).float().reshape(1, -1), target)),
            "legacy_r1_ce": float(F.cross_entropy(legacy_r1.logits[b, s].float().reshape(1, -1), target)),
            "legacy_r2_ce": float(F.cross_entropy(legacy_r2.logits[b, s].float().reshape(1, -1), target)),
            "canonical_r1_ce": float(F.cross_entropy(canonical_r1.logits[b, s].float().reshape(1, -1), target)),
        })

    def stats(values):
        return {"mean": float(np.mean(values)), "max": float(np.max(values)),
                "p95": float(sorted(values)[int(0.95 * (len(values) - 1))])}

    action_mismatch = sum(row["online_actions"] != row["reference_actions"] for row in rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "p1_best": str(args.p1_best),
        "p1_best_sha256": sha256_file(args.p1_best),
        "r2_initial": str(args.r2_initial),
        "r2_initial_sha256": sha256_file(args.r2_initial),
        "split": args.split,
        "sequences": args.sequences,
        "positions": len(rows),
        "backend": "sparse_merged_v1",
        "head_contract": "bf16_linear_fp32_ce_v1",
        "canonical_r2_rounds_all_two": bool((canonical_r2.rounds == 2).all()),
        "canonical_r2_action_mismatch_vs_independent_reference": action_mismatch,
        "canonical_r2_vs_reference_ce_abs": stats([abs(r["online_ce"] - r["reference_ce"]) for r in rows]),
        "canonical_r2_vs_reference_hidden_abs": stats([r["hidden_abs_max"] for r in rows]),
        "legacy_r1_minus_canonical_r1_ce_abs": stats([abs(r["legacy_r1_ce"] - r["canonical_r1_ce"]) for r in rows]),
        "stop_closed_loop": stop_checks,
    }
    atomic_json(output / "summary.json", summary)
    with (output / "per_state.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
