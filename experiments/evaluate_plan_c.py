from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from data.packed_lm import PackedLMDataset
from eval.counterfactual import cache_last_ffn_input, counterfactual_path_costs
from experiments.r1_closed_loop import source_identity
from lm_config import LMExperimentConfig
from model import LoopMoE
from model.top2 import baseline_top2_pairs
from train.common import autocast_context
from train.recovery_common import atomic_json, sha256_file


def accumulate_document_values(loss_sums, counts, indices, losses, doc_counts=None):
    np.add.at(loss_sums, indices, losses)
    if counts is not None:
        np.add.at(counts, indices, 1 if doc_counts is None else doc_counts)


def document_bootstrap(diff_sums, counts, replicates, seed=0):
    diff_sums = np.asarray(diff_sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    estimate = float(diff_sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for index in range(replicates):
        selected = rng.integers(0, len(counts), len(counts))
        draws[index] = diff_sums[selected].sum() / counts[selected].sum()
    return {
        "estimate": estimate,
        "upper_95": float(np.quantile(draws, 0.95)),
        "ci_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "replicates": replicates,
        "seed": seed,
    }


@torch.no_grad()
def production_consistency(model, dataset, depths, device, dtype, max_states=32):
    records = []
    layer = model.config.adaptive_layers[-1]
    block = model.blocks[layer]
    for sequence_index in range(len(dataset)):
        item = dataset[sequence_index]
        input_ids = torch.from_numpy(item["input_ids"]).unsqueeze(0).to(device)
        labels = torch.from_numpy(item["labels"]).unsqueeze(0).to(device)
        segment_ids = torch.from_numpy(item["segment_ids"]).unsqueeze(0).to(device)
        supervised = torch.nonzero(labels[0] != -100, as_tuple=False).squeeze(-1)
        with autocast_context(device, dtype):
            cached_h = cache_last_ffn_input(model, input_ids, segment_ids, depth=1)
        for position in supervised.tolist():
            if len(records) >= max_states:
                return records
            target = labels[0, position].reshape(1)
            with autocast_context(device, dtype):
                normalized = block.ffn_norm(cached_h[:, position])
                pair = baseline_top2_pairs(block.executor.gate_logits(normalized).float()).unsqueeze(0)
                replay = {depth: counterfactual_path_costs(model, cached_h[:, position], target, rounds=depth, actions=pair, dtype=dtype) for depth in depths}
            record = {"sequence_id": sequence_index, "doc_id": int(item["doc_ids"][position]), "position": position, "depths": {}}
            for depth in depths:
                with autocast_context(device, dtype):
                    output = model(input_ids, labels, depth=depth, segment_ids=segment_ids)
                full_ce = F.cross_entropy(output.logits[0, position].float().unsqueeze(0), target, reduction="none")
                replay_ce = replay[depth].q_pi0[0, 0]
                record["depths"][str(depth)] = {"full_ce": float(full_ce), "cached_replay_ce": float(replay_ce), "abs_diff": float(abs(full_ce - replay_ce))}
            records.append(record)
    return records


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", default="runs/followup_2026_10_26/plan_c")
    parser.add_argument("--config", default="configs/e00_m105_b2.json")
    parser.add_argument("--checkpoint", default="runs/e00/m105_b2/latest.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    plan_dir = Path(args.plan_dir)
    plan_path = plan_dir / "evaluation_plan.json"
    plan = json.loads(plan_path.read_text())
    corpus_dir = plan_dir / "confirmation_corpus"
    if sha256_file(corpus_dir / "manifest.json") != plan["corpus_manifest_sha256"]:
        raise ValueError("confirmation corpus changed after preregistration")
    if sha256_file(args.checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from preregistration")
    config = LMExperimentConfig.load(args.config)
    config.train.dtype = plan["dtype"]
    config.train.batch_size = plan["evaluation_batch_size"]
    device = torch.device(args.device)
    model = LoopMoE(config.model).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    dataset = PackedLMDataset(corpus_dir, "confirmation")
    doc_keys = np.array(sorted(json.loads((corpus_dir / "frozen_document_ids.json").read_text())["document_ranks"]), dtype=np.int64)
    depths = plan["depths"]
    loss_sums = {depth: np.zeros(len(doc_keys), dtype=np.float64) for depth in depths}
    counts = np.zeros(len(doc_keys), dtype=np.float64)
    elapsed = {depth: 0.0 for depth in depths}
    for sequence_index in range(len(dataset)):
        item = dataset[sequence_index]
        input_ids = torch.from_numpy(item["input_ids"]).unsqueeze(0).to(device)
        labels = torch.from_numpy(item["labels"]).unsqueeze(0).to(device)
        segment_ids = torch.from_numpy(item["segment_ids"]).unsqueeze(0).to(device)
        doc_indices = np.searchsorted(doc_keys, item["doc_ids"])
        valid = item["labels"] != -100
        for depth in depths:
            started = time.monotonic()
            with autocast_context(device, config.train.dtype):
                output = model(input_ids, labels, depth=depth, segment_ids=segment_ids)
            losses = F.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none").reshape(labels.shape)[0].cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed[depth] += time.monotonic() - started
            accumulate_document_values(loss_sums[depth], counts if depth == depths[0] else None, doc_indices[valid], losses[valid])
    total_count = float(counts.sum())
    if total_count < plan["minimum_supervised_tokens"]:
        raise RuntimeError(f"only {total_count} valid supervised tokens; preregistration requires at least {plan['minimum_supervised_tokens']}")
    if not np.allclose(sum(counts), dataset.n_supervised_tokens):
        raise RuntimeError("per-document counts do not match dataset supervised tokens")
    nll = {depth: float(loss_sums[depth].sum() / total_count) for depth in depths}
    primary = document_bootstrap(loss_sums[1] - loss_sums[2], counts, plan["bootstrap_replicates"], plan["bootstrap_seed"])
    secondary = {
        "r1_minus_r4": document_bootstrap(loss_sums[1] - loss_sums[4], counts, plan["bootstrap_replicates"], plan["bootstrap_seed"]),
        "r2_minus_r4": document_bootstrap(loss_sums[2] - loss_sums[4], counts, plan["bootstrap_replicates"], plan["bootstrap_seed"]),
    }
    consistency = production_consistency(model, dataset, depths, device, config.train.dtype)
    summary = {
        "protocol": plan["protocol"],
        "evaluation_plan_sha256": sha256_file(plan_path),
        "sequences": len(dataset),
        "documents": len(doc_keys),
        "supervised_tokens": total_count,
        "packed_real_tokens": dataset.n_tokens,
        "max_document_token_weight": float(counts.max() / total_count),
        "effective_documents": float(1 / ((counts / total_count) ** 2).sum()),
        "nll": {str(depth): nll[depth] for depth in depths},
        "primary_r1_minus_r2": primary,
        "tolerance_nat_per_token": plan["tolerance_nat_per_token"],
        "non_inferiority_pass": primary["upper_95"] < plan["tolerance_nat_per_token"],
        "secondary": secondary,
        "production_consistency": {
            "states": len(consistency),
            "max_abs_diff_by_depth": {str(depth): max(record["depths"][str(depth)]["abs_diff"] for record in consistency) for depth in depths},
            "full_forward_is_primary": True,
        },
        "elapsed_seconds_by_depth": elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    atomic_json(plan_dir / "summary.json", summary)
    per_document = {
        "document_ranks": doc_keys.tolist(),
        "counts": counts.tolist(),
        "loss_sums": {str(depth): loss_sums[depth].tolist() for depth in depths},
    }
    atomic_json(plan_dir / "per_document.json", per_document)
    atomic_json(plan_dir / "production_consistency.json", {"records": consistency})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_identity(),
        "argv": sys.argv,
        "evaluation_plan_sha256": sha256_file(plan_path),
        "summary_sha256": sha256_file(plan_dir / "summary.json"),
        "per_document_sha256": sha256_file(plan_dir / "per_document.json"),
        "production_consistency_sha256": sha256_file(plan_dir / "production_consistency.json"),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "corpus_manifest_sha256": sha256_file(corpus_dir / "manifest.json"),
        "torch_version": torch.__version__,
        "device": str(device),
    }
    atomic_json(plan_dir / "run_manifest.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
