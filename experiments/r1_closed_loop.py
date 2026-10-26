from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn import functional as F

from data.packed_lm import PackedLMBatchSampler, PackedLMDataset
from eval.counterfactual import readout_ce
from eval.dynamic_halting import DynamicHaltingPolicy
from experiments.e03_halting import baseline_costs, load_students, student_predictions
from lm_config import LMExperimentConfig
from model import LoopMoE
from model.top2 import pair_to_action
from train.common import autocast_context
from train.recovery_common import atomic_json, sha256_file


def position_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] != 1 or target.shape != (1,):
        raise ValueError(f"expected logits (1, V) and target (1,), got {tuple(logits.shape)} and {tuple(target.shape)}")
    return F.cross_entropy(logits.float(), target, reduction="none")


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().cpu().numpy().tobytes()).hexdigest()


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
    return {
        "commit": commit,
        "tree": tree,
        "uncommitted_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "has_uncommitted_diff": bool(diff),
    }


@torch.no_grad()
def run(config_path, checkpoint_path, states_path, e01_dir, e03_summary, output_dir, device_name, split, max_sequences, dtype_name=None, state_seed=0, near_threshold=1e-4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = LMExperimentConfig.load(config_path)
    if dtype_name is not None:
        config.train.dtype = dtype_name
    device = torch.device(device_name)
    model = LoopMoE(config.model).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    backbone_hash = sha256_file(checkpoint_path)
    states = torch.load(states_path, map_location="cpu", weights_only=False)
    if states["backbone_hash"] != backbone_hash:
        raise ValueError("states and checkpoint have different backbone hashes")
    if int(states["rounds"]) != config.model.max_calls:
        raise ValueError("cached continuation rounds do not match model max_calls")

    students = load_students(Path(e01_dir))
    offline_stop_prediction, offline_continue_prediction = student_predictions(students, states)
    offline_delta = offline_stop_prediction - offline_continue_prediction
    summary_source = json.loads(Path(e03_summary).read_text())
    tau = float(summary_source["calibration"]["value_student"]["budgets"]["2"]["tau"])
    offline_continue = offline_delta > tau
    offline_rounds = 1 + (int(states["rounds"]) - 1) * offline_continue.long()
    stop_cost, continue_cost = baseline_costs(states)
    offline_selected_cost = torch.where(offline_continue, continue_cost, stop_cost)
    policy = DynamicHaltingPolicy(students, tau)

    dataset = PackedLMDataset(config.train.data_dir, split)
    required_sequences = set(states["seq_indices"].unique().tolist())
    sampler = PackedLMBatchSampler(dataset, 2, shuffle=True, seed=state_seed, epoch=0, drop_last=False)
    ordered_sequences = [index for batch in sampler for index in batch if index in required_sequences]
    if set(ordered_sequences) != required_sequences:
        raise ValueError("packed sampler did not cover every cached sequence")
    if max_sequences:
        ordered_sequences = ordered_sequences[:max_sequences]

    records = []
    for start in range(0, len(ordered_sequences), 2):
        indices = ordered_sequences[start : start + 2]
        items = [dataset[index] for index in indices]
        input_ids = torch.from_numpy(__import__("numpy").stack([item["input_ids"] for item in items])).to(device)
        segment_ids = torch.from_numpy(__import__("numpy").stack([item["segment_ids"] for item in items])).to(device)
        trace = {}
        with autocast_context(device, config.train.dtype):
            logits, _ = policy.forward(model, input_ids, segment_ids, trace=trace)
        for batch_row, sequence_id in enumerate(indices):
            selected = torch.nonzero(states["seq_indices"] == sequence_id, as_tuple=False).squeeze(-1)
            for state_index_tensor in selected:
                state_index = int(state_index_tensor)
                position = int(states["positions"][state_index])
                target = states["targets"][state_index].reshape(1).to(device)
                cached_h = states["h"][state_index]
                online_h = trace["h"][batch_row, position]
                online_delta = trace["delta"][batch_row, position]
                online_continue = bool(trace["continue_mask"][batch_row, position])
                online_rounds = int(trace["rounds"][batch_row, position])
                online_baseline_local = int(trace["baseline_action"][batch_row, position])
                offline_baseline_local = int(states["baseline_action"][state_index])
                offline_baseline_global = int(pair_to_action(states["actions"][offline_baseline_local], experts=config.model.experts_per_layer))
                online_baseline_global = int(pair_to_action(states["actions"][online_baseline_local], experts=config.model.experts_per_layer))
                online_ce = position_cross_entropy(logits[batch_row, position].unsqueeze(0), target)
                fp32_readout_ce = readout_ce(model, online_h.float().unsqueeze(0), target)
                online_cached_cost = states["immediate"][state_index, online_baseline_local] if online_rounds == 1 else states["q_pi0"][state_index, online_baseline_local]
                records.append({
                    "split": split,
                    "sequence_id": int(sequence_id),
                    "doc_id": int(states["doc_ids"][state_index]),
                    "position": position,
                    "valid_label_mask": True,
                    "raw_h_hash": tensor_sha256(cached_h),
                    "feature_contract_version": "raw-pre-ffn-h-v1",
                    "offline_baseline_global_action": offline_baseline_global,
                    "online_baseline_global_action": online_baseline_global,
                    "offline_delta": float(offline_delta[state_index]),
                    "online_delta": float(online_delta),
                    "tau": tau,
                    "near_threshold": bool(abs(float(offline_delta[state_index]) - tau) <= near_threshold),
                    "offline_continue": bool(offline_continue[state_index]),
                    "online_continue": online_continue,
                    "offline_rounds": int(offline_rounds[state_index]),
                    "online_rounds": online_rounds,
                    "offline_selected_cost": float(offline_selected_cost[state_index]),
                    "online_cached_cost": float(online_cached_cost),
                    "online_ce": float(online_ce),
                    "online_fp32_readout_ce": float(fp32_readout_ce),
                    "max_abs_cached_online_h_diff": float((online_h.float().cpu() - cached_h).abs().max()),
                })

    if not records:
        raise ValueError("no states were evaluated")
    with (output_dir / "states.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    summary = {
        "split": split,
        "states": len(records),
        "sequences": len(ordered_sequences),
        "tau": tau,
        "online_dtype": config.train.dtype,
        "offline_reference": "original cached h, baseline, student predictions and selected costs",
        "online_entry": "eval.dynamic_halting.DynamicHaltingPolicy.forward",
        "baseline_global_mismatches": sum(r["offline_baseline_global_action"] != r["online_baseline_global_action"] for r in records),
        "decision_mismatches": sum(r["offline_continue"] != r["online_continue"] for r in records),
        "round_mismatches": sum(r["offline_rounds"] != r["online_rounds"] for r in records),
        "max_cached_online_h_diff": max(r["max_abs_cached_online_h_diff"] for r in records),
        "max_offline_online_delta_diff": max(abs(r["offline_delta"] - r["online_delta"]) for r in records),
        "max_online_ce_vs_offline_selected_cost_diff": max(abs(r["online_ce"] - r["offline_selected_cost"]) for r in records),
        "max_fp32_readout_vs_offline_selected_cost_diff": max(abs(r["online_fp32_readout_ce"] - r["offline_selected_cost"]) for r in records),
        "near_threshold_states": sum(r["near_threshold"] for r in records),
        "near_threshold_decision_mismatches": sum(r["near_threshold"] and r["offline_continue"] != r["online_continue"] for r in records),
        "offline_mean_rounds": sum(r["offline_rounds"] for r in records) / len(records),
        "online_mean_rounds": sum(r["online_rounds"] for r in records) / len(records),
        "offline_mean_selected_cost": sum(r["offline_selected_cost"] for r in records) / len(records),
        "online_mean_ce": sum(r["online_ce"] for r in records) / len(records),
    }
    atomic_json(output_dir / "summary.json", summary)

    data_manifest_path = Path(config.train.data_dir) / "manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_identity(),
        "argv": sys.argv,
        "config": asdict(config),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path),
        "backbone_sha256": backbone_hash,
        "states_path": str(states_path),
        "states_sha256": sha256_file(states_path),
        "students_path": str(Path(e01_dir) / "students.pt"),
        "students_sha256": sha256_file(Path(e01_dir) / "students.pt"),
        "e03_summary_path": str(e03_summary),
        "e03_summary_sha256": sha256_file(e03_summary),
        "data_manifest_path": str(data_manifest_path),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "data_split": data_manifest["splits"][split],
        "tokenizer": data_manifest["tokenizer"],
        "state_sampler_seed": state_seed,
        "state_sampler_batch_size": 2,
        "near_threshold": near_threshold,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "summary_sha256": sha256_file(output_dir / "summary.json"),
        "states_jsonl_sha256": sha256_file(output_dir / "states.jsonl"),
    }
    atomic_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--e01-dir", required=True)
    parser.add_argument("--e03-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--state-seed", type=int, default=0)
    parser.add_argument("--near-threshold", type=float, default=1e-4)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.states, args.e01_dir, args.e03_summary, args.output_dir, args.device, args.split, args.max_sequences, args.dtype, args.state_seed, args.near_threshold)


if __name__ == "__main__":
    main()
