from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from data.packed_lm import PackedLMDataset, split_paths
from experiments.r1_closed_loop import source_identity
from lm_config import LMExperimentConfig
from model import LoopMoE
from train.common import autocast_context
from train.recovery_common import atomic_json, sha256_file


def randomized_depth_order(depths, repeat, iteration, seed=0):
    order = list(depths)
    random.Random(f"fixed-depth-profile-v1:{seed}:{repeat}:{iteration}").shuffle(order)
    return order


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)), "mean": float(values.mean())}


def latency_summary(records, depths):
    result = {}
    for depth in depths:
        rows = [row for row in records if row["depth"] == depth]
        event_ms = np.array([row["event_ms"] for row in rows])
        wall_ms = np.array([row["wall_ms"] for row in rows])
        repeats = sorted({row["repeat"] for row in rows})
        repeat_means = [float(event_ms[[row["repeat"] for row in rows] == np.array(repeat)].mean()) for repeat in repeats]
        result[str(depth)] = {
            "calls": len(rows),
            "event_ms": percentile_summary(event_ms),
            "wall_ms": percentile_summary(wall_ms),
            "effective_tokens_per_second": float(sum(row["valid_tokens"] for row in rows) / event_ms.sum() * 1000),
            "repeat_mean_event_ms": repeat_means,
        }
    reference = result["2"]["repeat_mean_event_ms"]
    for depth in depths:
        ratios = [baseline / value - 1 for baseline, value in zip(reference, result[str(depth)]["repeat_mean_event_ms"])]
        result[str(depth)]["speedup_vs_r2"] = {"mean": float(np.mean(ratios)), "min": float(np.min(ratios)), "max": float(np.max(ratios))}
    return result


def gpu_state(physical_index):
    output = subprocess.check_output([
        "nvidia-smi", f"--id={physical_index}",
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    fields = [field.strip() for field in output.split(",")]
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True).splitlines()
    return {
        "index": fields[0], "uuid": fields[1], "name": fields[2], "memory_used_mib": int(fields[3]),
        "utilization_percent": int(fields[4]), "temperature_c": int(fields[5]), "power_w": float(fields[6]),
        "sm_clock_mhz": int(fields[7]),
        "compute_processes": [line.strip() for line in processes if line.strip() and fields[1] in line],
    }


def assert_exclusive(state):
    others = [line for line in state["compute_processes"] if f", {os.getpid()}," not in line]
    if others:
        raise RuntimeError(f"GPU is not exclusive: {others}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e00_m105_b2.json")
    parser.add_argument("--checkpoint", default="runs/e00/m105_b2/latest.pt")
    parser.add_argument("--corpus-dir", default="runs/followup_2026_10_26/plan_c/confirmation_corpus")
    parser.add_argument("--split", default="confirmation")
    parser.add_argument("--output-dir", default="runs/next_2026_10_26/profile")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physical-gpu-index", type=int, default=1)
    parser.add_argument("--sequences", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    depths = [1, 2, 4]
    packed_paths = split_paths(args.corpus_dir, args.split)
    protocol = {
        "protocol": "fixed-depth-production-profile-v1",
        "created_before_measurement": True,
        "source": source_identity(),
        "entrypoint": {"path": __file__, "sha256": sha256_file(__file__)},
        "argv": sys.argv,
        "config_sha256": sha256_file(args.config),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "corpus_manifest_sha256": sha256_file(Path(args.corpus_dir) / "manifest.json"),
        "packed_file_sha256": {name: sha256_file(path) for name, path in packed_paths.items()},
        "depths": depths,
        "warmup_per_depth": args.warmup,
        "repeats": args.repeats,
        "iterations_per_repeat": args.iterations,
        "sequence_count": args.sequences,
        "sequence_indices": list(range(args.sequences)),
        "dtype": "bfloat16",
        "batch_size": 1,
        "scope": "input already on GPU to full logits, all layers and vocabulary head; no CE; no autoregressive decode",
        "randomization": "depth order shuffled independently per repeat/iteration with frozen seed",
        "seed": args.seed,
    }
    atomic_json(output_dir / "profile_protocol.json", protocol)
    device = torch.device(args.device)
    config = LMExperimentConfig.load(args.config)
    config.train.dtype = "bfloat16"
    model = LoopMoE(config.model).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    hardware_start = gpu_state(args.physical_gpu_index)
    assert_exclusive(hardware_start)
    dataset = PackedLMDataset(args.corpus_dir, args.split)
    inputs = []
    for sequence_index in range(args.sequences):
        started = time.monotonic()
        item = dataset[sequence_index]
        record = {
            "sequence_index": sequence_index,
            "input_ids": torch.from_numpy(item["input_ids"]).unsqueeze(0).to(device),
            "segment_ids": torch.from_numpy(item["segment_ids"]).unsqueeze(0).to(device),
            "valid_tokens": int((item["labels"] != -100).sum()),
            "data_prepare_seconds": time.monotonic() - started,
        }
        inputs.append(record)
    raw_path = output_dir / "measurements.jsonl"
    records = []
    with raw_path.open("w") as handle:
        for depth in depths:
            for index in range(args.warmup):
                item = inputs[index % len(inputs)]
                with torch.no_grad(), autocast_context(device, config.train.dtype):
                    model(item["input_ids"], depth=depth, segment_ids=item["segment_ids"])
            torch.cuda.synchronize(device)
        for repeat in range(args.repeats):
            for iteration in range(args.iterations):
                item = inputs[iteration % len(inputs)]
                for depth in randomized_depth_order(depths, repeat, iteration, args.seed):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    wall_start = time.perf_counter()
                    with torch.no_grad(), autocast_context(device, config.train.dtype):
                        start_event.record()
                        output = model(item["input_ids"], depth=depth, segment_ids=item["segment_ids"])
                        end_event.record()
                    torch.cuda.synchronize(device)
                    wall_ms = (time.perf_counter() - wall_start) * 1000
                    event_ms = float(start_event.elapsed_time(end_event))
                    row = {
                        "repeat": repeat, "iteration": iteration, "depth": depth,
                        "sequence_index": item["sequence_index"], "valid_tokens": item["valid_tokens"],
                        "event_ms": event_ms, "wall_ms": wall_ms,
                    }
                    records.append(row)
                    handle.write(json.dumps(row) + "\n")
                del output
    peak = {}
    for depth in depths:
        torch.cuda.reset_peak_memory_stats(device)
        item = inputs[0]
        with torch.no_grad(), autocast_context(device, config.train.dtype):
            model(item["input_ids"], depth=depth, segment_ids=item["segment_ids"])
        torch.cuda.synchronize(device)
        peak[str(depth)] = torch.cuda.max_memory_allocated(device)
    hardware_end = gpu_state(args.physical_gpu_index)
    exclusive_end = not [line for line in hardware_end["compute_processes"] if f", {os.getpid()}," not in line]
    summary = {
        "protocol_sha256": sha256_file(output_dir / "profile_protocol.json"),
        "measurements_sha256": sha256_file(raw_path),
        "sequences": args.sequences,
        "valid_tokens_per_pass": sum(item["valid_tokens"] for item in inputs),
        "data_prepare_seconds": sum(item["data_prepare_seconds"] for item in inputs),
        "latency": latency_summary(records, depths),
        "peak_memory_bytes": peak,
        "hardware_start": hardware_start,
        "hardware_end": hardware_end,
        "exclusive_at_start": True,
        "exclusive_at_end": exclusive_end,
        "valid_for_latency_claims": exclusive_end,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["latency"], indent=2))


if __name__ == "__main__":
    main()
