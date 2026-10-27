"""Task card F: selection-split comparison of all registered R2 candidates.

Spec 16.1 / 17.3: complete production greedy forward (deployment backend) on
the fixed 128 selection sequences, token-weighted NLL and mean rounds for
legacy top-2 R1/R2, canonical teacher R1/R2, the P1 best actor (R1 policy),
the R2 initial (warmstarted) actor and the R2 learned best actor; paired
near-duplicate-group bootstrap intervals against canonical_pair_r1 and the
P1 best; cache-side sampled diagnostics on the fixed evaluation stream.
Selection uses this selection split only; newtest is not touched here.

Auxiliary script: imports the task-E evaluator helpers unchanged and writes
only into this task directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.packed_lm import PackedLMDataset
from experiments.rl_lm_evaluate import _grouped_ce, _load_actor, _paired_bootstrap, _sampled_cache_nll, _token_ce
from rl_lm.cache import cache_identity, load_cache
from rl_lm.cli import load_backbone, resolve_config
from rl_lm.inference import CanonicalTeacherPolicy, forward_with_last_ffn_policy
from train.common import autocast_context
from train.recovery_common import atomic_json, sha256_file


def online_nll(model, policy, batch, *, max_calls, backend, compute_dtype, chunk, head_chunk):
    output = forward_with_last_ffn_policy(
        model, policy, batch["input_ids"], batch["segment_ids"],
        max_calls=max_calls, backend=backend, compute_dtype=compute_dtype,
        chunk=chunk, head_chunk=head_chunk,
    )
    ce = _token_ce(output.logits, batch["labels"])
    valid = batch["valid"]
    return {
        "nll": float(ce.sum() / valid.sum()),
        "mean_rounds": float(output.rounds.reshape(-1)[valid.reshape(-1)].float().mean()),
        "ce": ce,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1-config", default="runs/rl_lm_pair_v1/r1_seed17/resolved_config.json")
    parser.add_argument("--r2-config", default="runs/rl_lm_pair_v1/r2_lambda0_seed17/resolved_config.json")
    parser.add_argument("--r1-best", default="runs/rl_lm_pair_v1/r1_seed17/checkpoint_best.pt")
    parser.add_argument("--r2-initial", default="runs/rl_lm_pair_v1/r2_lambda0_seed17/checkpoint_initial.pt")
    parser.add_argument("--r2-best", default="runs/rl_lm_pair_v1/r2_lambda0_seed17/checkpoint_best.pt")
    parser.add_argument("--sequences", type=int, default=128)
    parser.add_argument("--sampled-states", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="runs/rl_lm_pair_v1/task_f/evaluate_r2_selection.json")
    args = parser.parse_args()

    r1_config = resolve_config(args.r1_config, device=args.device)
    r2_config = resolve_config(args.r2_config, device=args.device)
    device = torch.device(r2_config.get("runtime.device"))
    model, _ = load_backbone(r2_config, device)
    backend = r2_config.get("environment.deployment_backend")
    compute_dtype = r2_config.get("environment.frozen_compute_dtype")
    chunk = int(r2_config.get("environment.expert_chunk"))
    head_chunk = int(r2_config.get("environment.head_chunk"))

    corpus_dir = Path(r2_config.get("data.corpus_dir"))
    split = r2_config.get("data.selection_split")
    dataset = PackedLMDataset(corpus_dir, split, limit=args.sequences)
    items = [dataset[i] for i in range(len(dataset))]
    batch = {
        "input_ids": torch.from_numpy(np.stack([item["input_ids"] for item in items])).to(device),
        "segment_ids": torch.from_numpy(np.stack([item["segment_ids"] for item in items])).to(device),
        "labels": torch.from_numpy(np.stack([item["labels"] for item in items])).to(device),
    }
    batch["valid"] = batch["labels"] != -100
    doc_ids = torch.from_numpy(np.stack([item["doc_ids"] for item in items])).to(device)
    group_of_rank = {}
    with open(corpus_dir / "documents.jsonl", "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("role") == split:
                group_of_rank[int(row["rank"])] = str(row["near_hash"])

    def legacy(depth):
        with torch.no_grad(), autocast_context(device, "bfloat16"):
            out = model(batch["input_ids"], batch["labels"], depth=depth, segment_ids=batch["segment_ids"])
        ce = _token_ce(out.logits, batch["labels"])
        return {"nll": float(ce.sum() / batch["valid"].sum()), "mean_rounds": float(depth), "ce": ce}

    r1_best_actor = _load_actor(r1_config, Path(args.r1_best), device)
    r2_initial_actor = _load_actor(r2_config, Path(args.r2_initial), device)
    r2_best_actor = _load_actor(r2_config, Path(args.r2_best), device)

    results = {
        "legacy_top2_r1": legacy(1),
        "legacy_top2_r2": legacy(2),
        "canonical_pair_r1": online_nll(model, CanonicalTeacherPolicy(), batch, max_calls=1, backend=backend, compute_dtype=compute_dtype, chunk=chunk, head_chunk=head_chunk),
        "canonical_pair_r2": online_nll(model, CanonicalTeacherPolicy(), batch, max_calls=2, backend=backend, compute_dtype=compute_dtype, chunk=chunk, head_chunk=head_chunk),
        "r1_best_greedy": online_nll(model, r1_best_actor, batch, max_calls=1, backend=backend, compute_dtype=compute_dtype, chunk=chunk, head_chunk=head_chunk),
        "r2_initial_greedy": online_nll(model, r2_initial_actor, batch, max_calls=2, backend=backend, compute_dtype=compute_dtype, chunk=chunk, head_chunk=head_chunk),
        "r2_learned_greedy": online_nll(model, r2_best_actor, batch, max_calls=2, backend=backend, compute_dtype=compute_dtype, chunk=chunk, head_chunk=head_chunk),
    }

    reference = results["canonical_pair_r1"]["nll"]
    summary = {
        name: {
            "nll": res["nll"],
            "mean_rounds": res["mean_rounds"],
            "delta_nll_vs_canonical_r1": res["nll"] - reference,
            "gain_nll_vs_canonical_r1": reference - res["nll"],
            "delta_nll_vs_r1_best": res["nll"] - results["r1_best_greedy"]["nll"],
            "selection_metric_lambda0": res["nll"],  # mean_NLL + 0 * extra rounds
        }
        for name, res in results.items()
    }

    def grouped(name):
        return _grouped_ce(results[name]["ce"], batch["valid"], doc_ids, group_of_rank)

    repetitions = int(r2_config.get("evaluation.bootstrap_repetitions"))
    confidence = float(r2_config.get("evaluation.bootstrap_confidence"))
    canonical_groups = grouped("canonical_pair_r1")
    bootstraps = {
        "r2_learned_minus_canonical_r1": _paired_bootstrap(grouped("r2_learned_greedy"), canonical_groups, repetitions=repetitions, confidence=confidence),
        "r2_learned_minus_r1_best": _paired_bootstrap(grouped("r2_learned_greedy"), grouped("r1_best_greedy"), repetitions=repetitions, confidence=confidence),
        "r2_initial_minus_r1_best": _paired_bootstrap(grouped("r2_initial_greedy"), grouped("r1_best_greedy"), repetitions=repetitions, confidence=confidence),
        "canonical_r2_minus_canonical_r1": _paired_bootstrap(grouped("canonical_pair_r2"), canonical_groups, repetitions=repetitions, confidence=confidence),
    }

    expected = cache_identity(
        model, model.config,
        checkpoint_path=r2_config.get("backbone.checkpoint_path"),
        config_path=r2_config.get("backbone.config_path"),
        tokenizer_path=r2_config.get("data.tokenizer_path"),
        data_dir=r2_config.get("data.corpus_dir"),
        split=split,
        sampling_seed=r2_config.get("data.sampling_seed"),
        backend_version=r2_config.get("environment.training_backend"),
        head_contract=r2_config.get("environment.head_contract"),
        expert_chunk=chunk,
    )
    cache = load_cache(Path(r2_config.get("data.cache_root")) / split, expected)
    sampled = {
        "r2_initial_sampled": _sampled_cache_nll(r2_config, model, r2_initial_actor, cache, device, args.sampled_states),
        "r2_learned_sampled": _sampled_cache_nll(r2_config, model, r2_best_actor, cache, device, args.sampled_states),
    }

    report = {
        "spec": "16.1/17.3 task F candidate comparison",
        "split": split,
        "sequences": int(batch["input_ids"].shape[0]),
        "valid_tokens": int(batch["valid"].sum()),
        "backend": backend,
        "head_contract": r2_config.get("environment.head_contract"),
        "checkpoints": {
            "r1_best": {"path": args.r1_best, "sha256": sha256_file(args.r1_best)},
            "r2_initial": {"path": args.r2_initial, "sha256": sha256_file(args.r2_initial)},
            "r2_best": {"path": args.r2_best, "sha256": sha256_file(args.r2_best)},
        },
        "online": summary,
        "paired_bootstrap": bootstraps,
        "sampled_cache": sampled,
        "minimum_practical_nll_gain": float(r2_config.get("evaluation.minimum_practical_nll_gain")),
        "r2_quality_tolerance_nll": float(r2_config.get("evaluation.r2_quality_tolerance_nll")),
    }
    atomic_json(Path(args.output), report)
    print(json.dumps({k: v for k, v in report.items() if k != "sampled_cache"}, indent=2))
    print(json.dumps({"sampled_cache": sampled}, indent=2))


if __name__ == "__main__":
    main()
