from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import torch

from data.packed_lm import PackedLMBatchSampler, PackedLMDataset
from eval.dynamic_halting import DynamicHaltingPolicy
from experiments.e03_halting import load_students
from lm_config import LMExperimentConfig
from model import LoopMoE
from train.common import autocast_context
from train.lm import lm_loss, packed_batch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@torch.no_grad()
def evaluate_dynamic(model, policy, dataset, config, device, batches):
    loss_sum, count = 0.0, 0
    supervised_rounds_sum, executed_rounds_sum, executed_positions = 0, 0, 0
    sampler = PackedLMBatchSampler(dataset, config.train.batch_size, shuffle=False, drop_last=False)
    for index, indices in enumerate(sampler):
        if index >= batches:
            break
        batch = packed_batch(dataset, indices, device)
        with autocast_context(device, config.train.dtype):
            logits, rounds = policy.forward(model, batch["input_ids"], batch["segment_ids"])
        _, batch_sum, batch_count = lm_loss(logits, batch["labels"])
        loss_sum += batch_sum.item()
        count += batch_count.item()
        supervised_rounds_sum += rounds[batch["labels"] != -100].sum().item()
        executed_rounds_sum += rounds.sum().item()
        executed_positions += rounds.numel()
    return {
        "nll": loss_sum / count,
        "mean_rounds_supervised": supervised_rounds_sum / count,
        "supervised_tokens": count,
        "executed_positions": executed_positions,
        "ffn_rounds_total": executed_rounds_sum,
        "expert_token_pairs": executed_rounds_sum * model.config.top_k,
    }


@torch.no_grad()
def profile(model, policy, batch, config, device, mode, warmup, steps):
    """Time input->logits only, for every mode; quality/CE stays outside."""
    reset_peak_memory(device)
    times = []
    for step in range(warmup + steps):
        synchronize(device)
        started = time.perf_counter()
        with autocast_context(device, config.train.dtype):
            if mode == "dynamic":
                policy.forward(model, batch["input_ids"], batch["segment_ids"])
            else:
                model(batch["input_ids"], depth=mode, segment_ids=batch["segment_ids"])
        synchronize(device)
        if step >= warmup:
            times.append(time.perf_counter() - started)
    ordered = sorted(times)
    return {
        "step_ms_mean": 1000 * sum(times) / len(times),
        "step_ms_p50": 1000 * ordered[len(ordered) // 2],
        "step_ms_p95": 1000 * ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "steps": steps,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--e01-dir", required=True)
    parser.add_argument("--e03-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=16)
    args = parser.parse_args()
    config = LMExperimentConfig.load(args.config)
    device = torch.device(config.train.device)
    model = LoopMoE(config.model).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    summary = json.loads(Path(args.e03_summary).read_text())
    tau = summary["calibration"]["value_student"]["budgets"]["2"]["tau"]
    policy = DynamicHaltingPolicy(load_students(Path(args.e01_dir)), tau)
    dataset = PackedLMDataset(config.train.data_dir, "validation")
    indices = next(iter(PackedLMBatchSampler(dataset, config.train.batch_size, shuffle=False, drop_last=False)))
    batch = packed_batch(dataset, indices, device)
    result = {"tau": tau, "scope": "single_last_ffn_S0_shared_GPU_not_isolated", "quality": {}, "profile": {}}
    for mode in (1, 4, "dynamic"):
        if mode == "dynamic":
            quality = evaluate_dynamic(model, policy, dataset, config, device, args.validation_batches)
        else:
            from train.lm import evaluate_lm
            quality = evaluate_lm(model, dataset, config, device, mode, args.validation_batches)
        result["quality"][str(mode)] = quality
        result["profile"][str(mode)] = profile(model, policy, batch, config, device, mode, args.warmup, args.steps)
    result["dynamic_vs_r4_latency_ratio"] = result["profile"]["dynamic"]["step_ms_mean"] / result["profile"]["4"]["step_ms_mean"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
