from __future__ import annotations

import argparse
import json
import time
import torch

from config import ExperimentConfig
from data.affine import sample_batch
from model import LoopMoE
from policy import Controller
from train.common import autocast_context, move_batch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def profile_forward(model, controller, config, device, warmup=20, steps=100):
    batch = sample_batch("validation", config.train.batch_size, config.data.seed, 0, (config.data.max_depth,))
    input_ids, labels = move_batch(batch, device)
    times = []
    with autocast_context(device, config.train.dtype):
        for step in range(warmup + steps):
            synchronize(device)
            started = time.perf_counter()
            output = model(input_ids, labels, controller=controller, budget=config.train.budget, sample=False)
            synchronize(device)
            if step >= warmup:
                times.append(time.perf_counter() - started)
    ordered = sorted(times)
    tokens = input_ids.numel()
    return {
        "steps": steps,
        "tokens_per_step": tokens,
        "tokens_per_second": tokens / (sum(times) / len(times)),
        "step_ms_mean": 1000 * sum(times) / len(times),
        "step_ms_p50": 1000 * ordered[len(ordered) // 2],
        "step_ms_p95": 1000 * ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "mean_calls": output.calls.float().mean().item(),
        "scope": "teacher_forced_full_forward_not_decode",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    device = torch.device(config.train.device)
    model = LoopMoE(config.model).to(device).eval()
    controller = Controller(config.model, config.controller).to(device).eval()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("controller") is not None:
            controller.load_state_dict(checkpoint["controller"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = profile_forward(model, controller, config, device, args.warmup, args.steps)
    result["device"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    Path = __import__("pathlib").Path
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
