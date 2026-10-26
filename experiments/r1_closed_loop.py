from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F

from eval.counterfactual import cache_last_ffn_input, last_adaptive_layer
from experiments.e03_halting import load_students
from lm_config import LMExperimentConfig
from model import LoopMoE
from data.packed_lm import PackedLMDataset
from train.value_student import predict
from train.common import autocast_context


def forward_with_features(model, students, tau, input_ids, segment_ids, feature_mode, cached_h=None):
    layer = last_adaptive_layer(model)
    block = model.blocks[layer]
    h = cache_last_ffn_input(model, input_ids, segment_ids, depth=1) if cached_h is None else cached_h
    normalized = block.ffn_norm(h)
    flat_normalized = normalized.reshape(-1, normalized.shape[-1])
    actions = block.executor.baseline_actions(flat_normalized)
    features = h.reshape(-1, h.shape[-1]) if feature_mode == "raw" else flat_normalized
    stop_cost = predict(students["immediate"]["student"], features, students["immediate"]["mean"], students["immediate"]["std"])
    continue_cost = predict(students["q_pi0"]["student"], features, students["q_pi0"]["mean"], students["q_pi0"]["std"])
    rows = torch.arange(actions.shape[0], device=actions.device)
    delta = stop_cost[rows, actions] - continue_cost[rows, actions]
    continue_mask = delta > tau
    update, _, _, _ = block.executor(normalized)
    current = h + model.gammas[layer] * update
    current_flat = current.reshape(-1, current.shape[-1])
    rounds = torch.ones(actions.shape, dtype=torch.long, device=actions.device)
    for _ in range(1, model.config.max_calls):
        if not bool(continue_mask.any()):
            break
        selected = block.ffn_norm(current_flat[continue_mask])
        update, _, _, _ = block.executor(selected)
        current_flat = current_flat.clone()
        current_flat[continue_mask] = current_flat[continue_mask] + model.gammas[layer] * update
        rounds[continue_mask] += 1
    logits = F.linear(model.final_norm(current_flat.reshape(h.shape)), model.embedding.weight)
    return logits, rounds.reshape(-1), delta


@torch.no_grad()
def run(config_path, checkpoint_path, states_path, e01_dir, e03_summary, output_dir, device_name, split, max_sequences, dtype_name=None):
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
    states = torch.load(states_path, map_location="cpu", weights_only=False)
    students = load_students(Path(e01_dir))
    for entry in students.values():
        entry["student"].to(device)
        entry["mean"] = entry["mean"].to(device)
        entry["std"] = entry["std"].to(device)
    summary_source = json.loads(Path(e03_summary).read_text())
    tau = summary_source["calibration"]["value_student"]["budgets"]["2"]["tau"]
    dataset = PackedLMDataset(config.train.data_dir, split)
    layer = last_adaptive_layer(model)
    block = model.blocks[layer]
    records = []
    for seq_index, indices in enumerate(states["seq_indices"].unique().tolist()):
        if max_sequences and seq_index >= max_sequences:
            break
        item = dataset[int(indices)]
        input_ids = torch.from_numpy(item["input_ids"]).unsqueeze(0).repeat(2, 1).to(device)
        segment_ids = torch.from_numpy(item["segment_ids"]).unsqueeze(0).repeat(2, 1).to(device)
        selected = torch.nonzero(states["seq_indices"] == indices, as_tuple=False).squeeze(-1)
        positions = states["positions"][selected].to(device)
        targets = states["targets"][selected].to(device)
        baseline = states["baseline_action"][selected].to(device)
        cached_h = states["h"][selected].to(device)
        with autocast_context(device, config.train.dtype):
            replay_h = cache_last_ffn_input(model, input_ids, segment_ids, depth=1)[0, positions]
        rows = torch.arange(len(selected), device=device)
        online_baseline = block.executor.baseline_actions(block.ffn_norm(replay_h))
        with autocast_context(device, config.train.dtype):
            offline_stop = predict(students["immediate"]["student"], replay_h, students["immediate"]["mean"], students["immediate"]["std"])[rows, online_baseline]
            offline_continue = predict(students["q_pi0"]["student"], replay_h, students["q_pi0"]["mean"], students["q_pi0"]["std"])[rows, online_baseline]
            offline_delta = offline_stop - offline_continue
            outputs = {mode: forward_with_features(model, students, tau, input_ids, segment_ids, mode, replay_h) for mode in ("raw", "normalized")}
        for local_index, state_index in enumerate(selected.tolist()):
            record = {
                "split": split,
                "sequence_id": int(indices),
                "doc_id": int(states["doc_ids"][state_index]),
                "position": int(states["positions"][state_index]),
                "baseline_global_action": int(states["baseline_action"][state_index]),
                "raw_h_max_abs_replay_diff": float((replay_h[local_index] - cached_h[local_index]).abs().max()),
                "offline_delta": float(offline_delta[local_index]),
                "tau": tau,
                "offline_continue": bool(offline_delta[local_index] > tau),
            }
            for mode, (logits, rounds, online_delta) in outputs.items():
                online_ce = F.cross_entropy(logits[local_index].float(), targets[local_index].reshape(1), reduction="none")
                record[f"{mode}_delta"] = float(online_delta[local_index])
                record[f"{mode}_continue"] = bool(rounds[local_index] > 1)
                record[f"{mode}_rounds"] = int(rounds[local_index])
                record[f"{mode}_ce"] = float(online_ce)
                reference = states["immediate"][state_index, baseline[local_index]] if rounds[local_index] == 1 else states["q_pi0"][state_index, baseline[local_index]]
                record[f"{mode}_selected_cost"] = float(reference)
            records.append(record)
    with (output_dir / "states.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    raw_mismatch = sum(record["offline_continue"] != record["raw_continue"] for record in records)
    normalized_mismatch = sum(record["offline_continue"] != record["normalized_continue"] for record in records)
    changed = sum(record["raw_continue"] != record["normalized_continue"] for record in records)
    summary = {
        "split": split,
        "states": len(records),
        "sequences": len(states["seq_indices"].unique()),
        "tau": tau,
        "raw_offline_decision_mismatches": raw_mismatch,
        "normalized_offline_decision_mismatches": normalized_mismatch,
        "raw_vs_normalized_decision_changes": changed,
        "max_raw_h_replay_diff": max(record["raw_h_max_abs_replay_diff"] for record in records),
        "max_raw_offline_delta_diff": max(abs(record["offline_delta"] - record["raw_delta"]) for record in records),
        "max_raw_ce_vs_selected_cost_diff": max(abs(record["raw_ce"] - record["raw_selected_cost"]) for record in records),
        "raw_mean_rounds": sum(record["raw_rounds"] for record in records) / len(records),
        "normalized_mean_rounds": sum(record["normalized_rounds"] for record in records) / len(records),
        "raw_mean_ce": sum(record["raw_ce"] for record in records) / len(records),
        "normalized_mean_ce": sum(record["normalized_ce"] for record in records) / len(records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.states, args.e01_dir, args.e03_summary, args.output_dir, args.device, args.split, args.max_sequences, args.dtype)


if __name__ == "__main__":
    main()
