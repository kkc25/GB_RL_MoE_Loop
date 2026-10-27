"""Task card F: three-layer speed profile for RL-LM-Pair-v1 (spec 16.4).

Layers:

1. selector-only — legacy gate+top-k, canonical pair selector, and the full
   RL observation (features + actor + joint argmax).  The shared norm+gate
   is measured on its own as well, so the accounting is consistent and no
   common component is either double-counted or silently dropped.
2. executor-only — on one fixed route: the original two-slot dispatch
   (``top2_dispatch``), the merged dispatch (``sparse_merged_v1``) and the
   fixed-shape grouped candidate (``canonical_all_experts_v1``).
3. end-to-end — full input to all logits: original MoE depth 1/2, the
   parameter-matched dense baseline, canonical teacher R1/R2 and the learned
   actor candidates through ``forward_with_last_ffn_policy``.

Protocol: 32 fixed selection inputs, 30 warmup iterations, 5 repeats of 200
measurements; main workload batch=1/S=1024, batch=8 reported separately
(``--batch8``).  CUDA events and wall time are both recorded per call; raw
timings, mean/p50/p95, effective tokens/s, peak allocated/reserved memory,
GPU UUID/clocks/power and software versions are saved.  The selected GPU
must be exclusive: the script refuses to run when another process holds the
device.  No torch.compile or profiler is used in the timed region.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from data.packed_lm import PackedLMDataset
from eval.counterfactual import cache_last_ffn_input
from lm_config import LMExperimentConfig
from model import LoopMoE
from model.top2 import baseline_top2_pairs, top2_dispatch
from rl_lm.actor import PairActor
from rl_lm.cli import load_backbone, resolve_config
from rl_lm.executor import make_executor
from rl_lm.inference import CanonicalTeacherPolicy, forward_with_last_ffn_policy, greedy_lowest_id
from rl_lm.ledger import RLLedger
from rl_lm.selector import canonical_pair
from train.common import autocast_context
from train.recovery_common import atomic_json, sha256_file


def gpu_metadata(device: torch.device, physical_index: int | None = None) -> dict:
    index = physical_index if physical_index is not None else device.index if device.index is not None else torch.cuda.current_device()
    meta = {
        "device": str(device),
        "name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
    }
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=uuid,driver_version,clocks.sm,clocks.max.sm,power.draw,power.limit,memory.used",
                "--format=csv,noheader",
            ],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        uuid, driver, clocks, clocks_max, power, power_limit, mem_used = [part.strip() for part in out.split(",")]
        meta |= {
            "gpu_uuid": uuid,
            "driver_version": driver,
            "clocks_sm_mhz": clocks,
            "clocks_max_sm_mhz": clocks_max,
            "power_draw": power,
            "power_limit": power_limit,
            "memory_used_at_query": mem_used,
        }
    except Exception as exc:  # metadata only; never fail the profile on nvidia-smi
        meta["nvidia_smi_error"] = str(exc)
    return meta


def check_exclusive(device: torch.device, physical_index: int | None = None) -> None:
    index = physical_index if physical_index is not None else device.index if device.index is not None else torch.cuda.current_device()
    out = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    # Our own process appears once its CUDA context exists (set_device);
    # exclusivity means no *other* process holds the device.
    others = [line for line in out.splitlines() if line.strip() and int(line.split(",")[0].strip()) != os.getpid()]
    if others:
        raise RuntimeError(f"GPU {index} is not exclusive; compute apps still present: {others}")


def measure(fn, *, warmup: int, repeats: int, measurements: int, tokens: int) -> dict:
    """Time ``fn`` with CUDA events and wall clock; raw values are kept."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    repeats_out = []
    for _ in range(repeats):
        event_ms = []
        wall_ms = []
        for _ in range(measurements):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            t0 = time.perf_counter()
            fn()
            end.record()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            event_ms.append(float(start.elapsed_time(end)))
            wall_ms.append(float((t1 - t0) * 1e3))
        repeats_out.append({"event_ms": event_ms, "wall_ms": wall_ms})
    flat_event = [v for rep in repeats_out for v in rep["event_ms"]]
    flat_wall = [v for rep in repeats_out for v in rep["wall_ms"]]

    def stats(values):
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean())
        return {
            "mean_ms": mean,
            "p50_ms": float(np.quantile(arr, 0.5)),
            "p95_ms": float(np.quantile(arr, 0.95)),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "tokens_per_second": tokens / (mean / 1e3),
        }

    return {
        "tokens_per_call": tokens,
        "warmup": warmup,
        "repeats": repeats,
        "measurements_per_repeat": measurements,
        "cuda_event": stats(flat_event),
        "wall": stats(flat_wall),
        # Absolute steady-state levels after the measurement loop; the peaks
        # above are window maxima after a reset and stay near zero when the
        # caching allocator reuses blocks (no new allocations per call).
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "raw": repeats_out,
    }


def load_actor_payload(path: str, *, train_stop: bool, device) -> PairActor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = PairActor(int(payload["feature_dim"]), train_stop=train_stop)
    actor.load_state_dict(payload["actor_state_dict"])
    return actor.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--corpus-dir")
    parser.add_argument("--tokenizer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--dense-config", default="runs/next_2026_10_26/baseline_100m/dense_parameter.json")
    parser.add_argument("--dense-checkpoint", default="runs/next_2026_10_26/baseline_100m/dense_parameter/latest.pt")
    parser.add_argument("--deployment-backend", default="sparse_merged_v1", choices=["sparse_merged_v1", "sparse_merged_v2"])
    parser.add_argument("--r1-actor", default=None, help="P1 best actor checkpoint (max_calls=1 candidate)")
    parser.add_argument("--r2-actor", default=None, help="P2 candidate actor checkpoint (max_calls=2)")
    parser.add_argument("--sequences", type=int, default=32, help="fixed profile inputs from the selection split")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--measurements", type=int, default=200)
    parser.add_argument("--batch8", action="store_true", help="also run the batch=8 secondary workload")
    parser.add_argument("--output-dir", default=None, help="new output directory; default is metric/backend-versioned")
    args = parser.parse_args()

    config = resolve_config(args.config, checkpoint=args.checkpoint, corpus_dir=args.corpus_dir, tokenizer=args.tokenizer, device=args.device)
    device = torch.device(config.get("runtime.device"))
    if device.type != "cuda":
        raise RuntimeError("the formal profile requires CUDA")
    # Make the profiled device current so CUDA events and all memory counters
    # (no-arg memory_allocated/max_memory_allocated/reset_peak_memory_stats)
    # refer to the profiled GPU even when it is not cuda:0.
    torch.cuda.set_device(device)
    check_exclusive(device, args.physical_gpu_index)
    output_dir = Path(args.output_dir or f"runs/rl_lm_pair_v1/task_f/profile_metrics_v2_{args.deployment_backend}")
    if (output_dir / "profile_report.json").exists():
        raise FileExistsError("profile report already exists; choose a new --output-dir to preserve it")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _ = load_backbone(config, device)
    max_calls = int(config.get("environment.max_calls"))
    chunk = int(config.get("environment.expert_chunk"))
    head_chunk = int(config.get("environment.head_chunk"))
    compute_dtype = config.get("environment.frozen_compute_dtype")

    dense_config = LMExperimentConfig.load(args.dense_config)
    dense_model = LoopMoE(dense_config.model).to(device).eval()
    dense_model.load_state_dict(torch.load(args.dense_checkpoint, map_location=device, weights_only=False)["model"])
    dense_model.requires_grad_(False)

    actors = {}
    if args.r1_actor:
        actors["learned_r1"] = (load_actor_payload(args.r1_actor, train_stop=False, device=device), 1)
    if args.r2_actor:
        actors["learned_r2"] = (load_actor_payload(args.r2_actor, train_stop=True, device=device), 2)

    dataset = PackedLMDataset(config.get("data.corpus_dir"), config.get("data.selection_split"), limit=args.sequences)
    if len(dataset) < args.sequences:
        raise ValueError("selection split smaller than the fixed profile input set")
    items = [dataset[i] for i in range(args.sequences)]
    all_input_ids = [torch.from_numpy(item["input_ids"]).unsqueeze(0).to(device) for item in items]
    all_segment_ids = [torch.from_numpy(item["segment_ids"]).unsqueeze(0).to(device) for item in items]

    block = model.blocks[model.config.adaptive_layers[-1]]
    ledger = RLLedger()
    wall_start = time.time()

    report = {
        "spec": "16.4",
        "metric_schema_version": "rl_lm_profile_v2",
        "deployment_backend": args.deployment_backend,
        "timing_scopes": {
            "selector_only": "terminal norm + gate + named selection/actor work; no experts or head",
            "executor_only": "terminal expert dispatch + pair mixing + residual; precomputed norm/gate/routes",
            "end_to_end": "full model prefix + terminal policy/all rounds + final norm/head; excludes CE",
        },
        "variant_semantics": {"grouped_canonical": "canonical_all_experts_v1 dense physical work; not sparse grouped GEMM",
                              "merged_dispatch": args.deployment_backend},
        "gpu": gpu_metadata(device, args.physical_gpu_index),
        "backbone_checkpoint_sha256": sha256_file(config.get("backbone.checkpoint_path")),
        "dense_checkpoint_sha256": sha256_file(args.dense_checkpoint),
        "r1_actor_sha256": sha256_file(args.r1_actor) if args.r1_actor else None,
        "r2_actor_sha256": sha256_file(args.r2_actor) if args.r2_actor else None,
        "protocol": {
            "fixed_inputs": args.sequences,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "measurements_per_repeat": args.measurements,
            "compute_dtype": compute_dtype,
            "expert_chunk": chunk,
            "head_chunk": head_chunk,
            "compile_or_profiler_in_timed_region": False,
        },
        "layers": {},
    }

    def run_workload(name: str, builders, batch: int) -> None:
        """builders: list of (variant_name, callable(input_index)->None)."""
        tokens = batch * 1024
        groups = max(1, args.sequences // batch)
        results = {}
        for variant, fn in builders:
            counter = {"i": 0}

            def call():
                fn(counter["i"] % groups)
                counter["i"] += 1

            results[variant] = measure(
                call, warmup=args.warmup, repeats=args.repeats,
                measurements=args.measurements, tokens=tokens,
            )
        report["layers"][name] = results
        print(json.dumps({name: {v: {"event_mean_ms": r["cuda_event"]["mean_ms"], "tokens_per_second": r["cuda_event"]["tokens_per_second"]} for v, r in results.items()}}))

    # ---------------------------------------------------------------- layer 1
    def hidden_batches(batch: int):
        """[batch*1024, D] frozen prefix rows, one per fixed input group."""
        rows = []
        for start in range(0, args.sequences, batch):
            with torch.no_grad(), autocast_context(device, compute_dtype):
                prefix = cache_last_ffn_input(
                    model,
                    torch.cat(all_input_ids[start : start + batch]),
                    torch.cat(all_segment_ids[start : start + batch]),
                )
            rows.append(prefix.float().reshape(-1, model.config.hidden_size).contiguous())
        return rows

    def selector_variants(batch: int):
        h0s = hidden_batches(batch)

        def shared(index):
            h = h0s[index]
            with torch.no_grad(), autocast_context(device, compute_dtype):
                n = block.ffn_norm(h)
                z = block.moe.gate(n).float()
            return n, z

        def legacy(index):
            _, z = shared(index)
            baseline_top2_pairs(z)

        def canonical(index):
            _, z = shared(index)
            canonical_pair(z)

        executor = make_executor(model, args.deployment_backend, chunk=chunk, compute_dtype=compute_dtype)
        rl_actor = actors.get("learned_r2", actors.get("learned_r1", (None, None)))[0]
        rounds0 = torch.zeros(batch * 1024, dtype=torch.int64, device=device)
        prev0 = torch.full((batch * 1024, 2), -1, dtype=torch.int64, device=device)

        def rl_full(index):
            h = h0s[index]
            observation = executor.observe(h, h, rounds0, prev0, max_calls=max_calls)
            if rl_actor is None:
                raise RuntimeError("no actor provided")
            log_probs = rl_actor(observation.features, observation.pair_reference, rounds0 >= 1)
            greedy_lowest_id(log_probs)

        variants = [("shared_norm_gate", lambda i: shared(i)), ("legacy_gate_top2", legacy), ("canonical_pair_selector", canonical)]
        if rl_actor is not None:
            variants.append(("rl_features_actor_joint", rl_full))
        return variants

    # ---------------------------------------------------------------- layer 2
    def executor_variants(batch: int):
        rows = batch * 1024
        h0s = hidden_batches(batch)
        norms = [block.ffn_norm(h) for h in h0s]
        with torch.no_grad(), autocast_context(device, compute_dtype):
            gates = [block.moe.gate(n).float() for n in norms]
        routes = [canonical_pair(z) for z in gates]
        actions = [r + 1 for r in routes]
        tables = [make_executor(model, args.deployment_backend, chunk=chunk, compute_dtype=compute_dtype).table[a - 1] for a in actions]
        merged = make_executor(model, args.deployment_backend, chunk=chunk, compute_dtype=compute_dtype)
        grouped = make_executor(model, "canonical_all_experts_v1", chunk=chunk, compute_dtype=compute_dtype)
        rounds0 = torch.zeros(rows, dtype=torch.int64, device=device)
        prev0 = torch.full((rows, 2), -1, dtype=torch.int64, device=device)
        observations = [merged.observe(h, h, rounds0, prev0, max_calls=max_calls) for h in h0s]
        gamma = float(model.gammas[model.config.adaptive_layers[-1]])

        def two_slot(index):
            n = norms[index]
            with torch.no_grad(), autocast_context(device, compute_dtype):
                mixed, _, _ = top2_dispatch(block.moe.pool, n, pairs=tables[index], gate_logits=gates[index])
            h0s[index] + gamma * mixed.float()

        def merged_dispatch(index):
            merged.step_pair(h0s[index], observations[index], actions[index])

        def grouped_dispatch(index):
            grouped.step_pair(h0s[index], observations[index], actions[index])

        return [("original_two_slot", two_slot), ("merged_dispatch", merged_dispatch), ("grouped_canonical", grouped_dispatch)]

    # ---------------------------------------------------------------- layer 3
    def end_to_end_variants(batch: int):
        if batch == 1:
            ids, segs = all_input_ids, all_segment_ids
        else:
            ids = [torch.cat(all_input_ids[i:i + batch]) for i in range(0, args.sequences, batch)]
            segs = [torch.cat(all_segment_ids[i:i + batch]) for i in range(0, args.sequences, batch)]

        def legacy_r1(index):
            with torch.no_grad(), autocast_context(device, compute_dtype):
                model(ids[index], None, depth=1, segment_ids=segs[index])

        def legacy_r2(index):
            with torch.no_grad(), autocast_context(device, compute_dtype):
                model(ids[index], None, depth=2, segment_ids=segs[index])

        def dense(index):
            with torch.no_grad(), autocast_context(device, compute_dtype):
                dense_model(ids[index], None, depth=1, segment_ids=segs[index])

        def online(policy, calls):
            def run(index):
                forward_with_last_ffn_policy(
                    model, policy, ids[index], segs[index],
                    max_calls=calls, backend=args.deployment_backend,
                    compute_dtype=compute_dtype, chunk=chunk, head_chunk=head_chunk,
                )
            return run

        variants = [
            ("original_moe_r1", legacy_r1),
            ("original_moe_r2", legacy_r2),
            ("dense_parameter", dense),
            ("canonical_pair_r1", online(CanonicalTeacherPolicy(), 1)),
            ("canonical_pair_r2", online(CanonicalTeacherPolicy(), 2)),
        ]
        for name, (actor, calls) in actors.items():
            variants.append((name, online(actor, calls)))
        return variants

    for batch in ([1, 8] if args.batch8 else [1]):
        tag = f"batch{batch}_s1024"
        run_workload(f"selector_only_{tag}", selector_variants(batch), batch)
        run_workload(f"executor_only_{tag}", executor_variants(batch), batch)
        run_workload(f"end_to_end_{tag}", end_to_end_variants(batch), batch)

    ledger.record_wall_time(time.time() - wall_start, gpus=1)
    ledger.merge_peak(device)
    report["profile_ledger"] = ledger.to_dict()
    atomic_json(output_dir / "profile_report.json", report)
    print(json.dumps({"ledger": report["profile_ledger"]}))


if __name__ == "__main__":
    main()
