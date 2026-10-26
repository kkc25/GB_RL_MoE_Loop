"""Targeted CPU execution of existing repository paths; no source modifications.

python reproduce_model_training_torch.py
The tiny bookkeeping test injects a deterministic clock and validation scores,
so it proves checkpoint semantics rather than measuring language-model quality.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parent / "GB_RL_MoE_Loop"
sys.path.insert(0, str(REPO))

from config import ModelConfig
from eval.dynamic_halting import DynamicHaltingPolicy
from eval.profile_dynamic import profile
from lm_config import LMExperimentConfig, LMTrainConfig
from model import LoopMoE
from train.common import MetricWriter
import train.lm as lm
from train.value_student import ValueStudent


class TinyDataset:
    def __len__(self):
        return 4

    def __getitem__(self, i):
        ids = np.array([1, 2+i, 7, 4, 9, 3], dtype=np.int64)
        labels = np.r_[ids[1:], np.int64(-100)]
        return {"input_ids": ids, "labels": labels, "segment_ids": np.zeros(6, dtype=np.int64), "n_tokens": 6, "n_supervised_tokens": 5}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    cfg = ModelConfig(layers=2, hidden_size=16, attention_heads=2, head_dim=8,
                      vocab_size=32, intermediate_size=16, experts_per_layer=4,
                      top_k=2, adaptive_layers=(1,), max_calls=4, target_only=False)
    model = LoopMoE(cfg).eval()
    run = LMTrainConfig(device="cpu", dtype="float32", batch_size=1)
    config = LMExperimentConfig(model=cfg, train=run)
    batch = lm.packed_batch(TinyDataset(), [0], torch.device("cpu"))
    students = {name: {"student": ValueStudent(16, 6, 8), "mean": torch.zeros(1, 16), "std": torch.ones(1, 16)} for name in ("immediate", "q_pi0")}
    policy = DynamicHaltingPolicy(students, tau=0.0)
    real_ce = torch.nn.functional.cross_entropy
    counts = {}
    for mode in (1, 4, "dynamic"):
        calls = []
        def observed_ce(*args, **kwargs):
            calls.append(1)
            return real_ce(*args, **kwargs)
        with patch("torch.nn.functional.cross_entropy", side_effect=observed_ce):
            profile(model, policy, batch, config, torch.device("cpu"), mode, warmup=1, steps=2)
        counts[str(mode)] = len(calls)
    assert counts == {"1": 3, "4": 3, "dynamic": 0}, counts

    model.train()
    model.zero_grad(set_to_none=True)
    out = model(batch["input_ids"], batch["labels"], depth=4, segment_ids=batch["segment_ids"])
    out.loss.backward()
    gate_gradient = float(model.blocks[-1].moe.gate.weight.grad.abs().sum())
    expert_gradient = float(model.blocks[-1].moe.pool.w1.grad.abs().sum())
    assert gate_gradient > 0 and expert_gradient > 0

    full_counts = {}
    for filename in ["e00_debug67_b2.json", "e00_m105_b2.json", "e00_dense_b3.json", "e00_dense_b4.json", "e00_dense_b5.json"]:
        full_config = LMExperimentConfig.load(REPO / "configs" / filename)
        with torch.device("meta"):
            full_model = LoopMoE(full_config.model)
        full_counts[filename] = sum(p.numel() for p in full_model.parameters())

    with tempfile.TemporaryDirectory(prefix="ledger_probe_", dir=AUDIT) as directory:
        bookkeeping_config = LMExperimentConfig(model=cfg, train=LMTrainConfig(output_dir=directory, device="cpu", dtype="float32", batch_size=1, updates=2, validation_interval=1, validation_batches=1, depth_schedule="fixed", fixed_depth=1))
        bookkeeping_model = LoopMoE(cfg)
        clock = iter([100.0, 107.0])
        vals = iter([1.0, 0.9])
        def deterministic_validation(*args, **kwargs):
            return {"nll": next(vals), "ppl": 0.0, "supervised_tokens": 5, "depth": 1}
        with patch.object(lm, "time", SimpleNamespace(monotonic=lambda: next(clock))), patch.object(lm, "evaluate_lm", deterministic_validation):
            first = lm.train_lm(bookkeeping_model, TinyDataset(), TinyDataset(), bookkeeping_config, torch.device("cpu"), MetricWriter(Path(directory)/"metrics.jsonl"))
        first_checkpoint = torch.load(Path(directory)/"latest.pt", map_location="cpu", weights_only=False)
        first_best = torch.load(Path(directory)/"best_validation.pt", map_location="cpu", weights_only=False)
        assert first["elapsed_seconds"] == 7 and first_checkpoint["ledger"]["elapsed_seconds"] == 0
        bookkeeping_config.train.resume = str(Path(directory)/"latest.pt")
        bookkeeping_config.train.updates = 3
        clock = iter([200.0, 205.0])
        vals = iter([2.0])
        with patch.object(lm, "time", SimpleNamespace(monotonic=lambda: next(clock))), patch.object(lm, "evaluate_lm", deterministic_validation):
            second = lm.train_lm(bookkeeping_model, TinyDataset(), TinyDataset(), bookkeeping_config, torch.device("cpu"), MetricWriter(Path(directory)/"metrics.jsonl"))
        second_best = torch.load(Path(directory)/"best_validation.pt", map_location="cpu", weights_only=False)
        assert second["elapsed_seconds"] == 5
        assert first_best["update"] == 2 and second_best["update"] == 3
        bookkeeping = {"clock_injected": True, "validation_scores_injected": [1.0, 0.9, 2.0], "first_segment_in_memory_seconds": first["elapsed_seconds"], "first_segment_saved_checkpoint_seconds": first_checkpoint["ledger"]["elapsed_seconds"], "second_segment_reported_cumulative_seconds": second["elapsed_seconds"], "expected_cumulative_seconds": 12.0, "old_best_update": first_best["update"], "new_best_update_despite_worse_validation": second_best["update"]}

    result = {"torch_audit_version": torch.__version__, "repository_original_torch_version": "2.12.1+cu130", "device": "cpu", "profile_cross_entropy_invocations_for_three_forwards": counts, "lm_training_gate_gradient_abs_sum": gate_gradient, "lm_training_expert_gradient_abs_sum": expert_gradient, "real_module_parameter_counts": full_counts, "checkpoint_bookkeeping_probe": bookkeeping}
    (AUDIT/"model_training_torch_checks.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
