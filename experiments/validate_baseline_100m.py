from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from data.packed_lm import PackedLMBatchSampler, PackedLMDataset, verify_packed_splits
from experiments.r1_closed_loop import source_identity
from lm_config import LMExperimentConfig
from train.recovery_common import atomic_json, sha256_file


def validate_arm(root, arm, protocol):
    root = Path(root)
    arm_dir = root / arm
    config = LMExperimentConfig.load(root / f"{arm}.json")
    resolved = LMExperimentConfig.load(arm_dir / "resolved_config.json")
    summary = json.loads((arm_dir / "run_summary.json").read_text())
    checkpoint_path = arm_dir / "latest.pt"
    checkpoint = __import__("torch").load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset = PackedLMDataset(config.train.data_dir, "train")
    sampler = PackedLMBatchSampler(dataset, protocol["shared"]["batch_size"], shuffle=True, seed=protocol["shared"]["seed"], epoch=0, drop_last=True)
    expected_tokens = 0
    supervised_tokens = 0
    for indices in sampler:
        for index in indices:
            expected_tokens += dataset[index]["n_tokens"]
            supervised_tokens += dataset[index]["n_supervised_tokens"]
    padding_tokens = protocol["shared"]["total_input_tokens"] - expected_tokens
    assert padding_tokens == dataset.token_counts()["n_padding"]
    assert summary["ledger"]["updates"] == protocol["shared"]["updates"] == 12207
    assert summary["ledger"]["input_tokens"] == expected_tokens
    assert summary["ledger"]["supervised_tokens"] == supervised_tokens
    assert summary["ledger"]["elapsed_seconds"] > 0
    assert summary["ledger"]["allocated_gpu_hours"] > 0
    assert summary["ledger"]["peak_memory_bytes"] > 0
    assert summary["ledger"]["schedule_phases"] == []
    assert summary["ledger"]["data_migrations"] == []
    assert summary["environment"]["parameters"] == sum(parameter.numel() for parameter in __import__("model").LoopMoE(config.model).parameters())
    assert summary["environment"]["packed_sha256"] == verify_packed_splits(config.train.data_dir, ("train", "validation"))
    assert asdict(resolved.model) == asdict(config.model)
    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["update"] == 12207
    assert checkpoint["data_hash"] == summary["environment"]["data_hash"]
    assert checkpoint["config"]["model"] == asdict(config.model)
    assert checkpoint["stream_metadata"]["n_sequences"] == len(dataset)
    assert checkpoint["stream_metadata"]["batch_size"] == 8
    assert checkpoint["stream_metadata"]["sampler_seed"] == 17
    validation = summary["validation"]["depth_1"]
    assert validation["supervised_tokens"] == 130834
    assert validation["depth"] == 1
    assert validation["nll"] == validation["nll"] and validation["nll"] > 0
    metrics = [json.loads(line) for line in (arm_dir / "metrics.jsonl").read_text().splitlines()]
    validations = [row for row in metrics if row.get("phase") == "validation"]
    assert validations[-1]["update"] == 12207
    assert validations[-1]["nll"] == validation["nll"]
    return {
        "parameters": summary["environment"]["parameters"],
        "input_tokens": summary["ledger"]["input_tokens"],
        "padding_tokens": padding_tokens,
        "supervised_tokens": summary["ledger"]["supervised_tokens"],
        "r1_nll": validation["nll"],
        "elapsed_seconds": summary["ledger"]["elapsed_seconds"],
        "allocated_gpu_hours": summary["ledger"]["allocated_gpu_hours"],
        "peak_memory_bytes": summary["ledger"]["peak_memory_bytes"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_update": checkpoint["update"],
        "stream_metadata": checkpoint["stream_metadata"],
        "packed_identity_verified": True,
        "config_identity_verified": True,
    }


def validate_profile(root, arm):
    root = Path(root)
    profile_dir = root / arm / "profile_final"
    summary = json.loads((profile_dir / "summary.json").read_text())
    protocol = json.loads((profile_dir / "profile_protocol.json").read_text())
    assert summary["valid_for_latency_claims"]
    assert summary["exclusive_at_start"] and summary["exclusive_at_end"]
    assert protocol["config_sha256"] == sha256_file(root / f"{arm}.json")
    assert protocol["checkpoint_sha256"] == sha256_file(root / arm / "latest.pt")
    assert len(protocol["packed_file_sha256"]) == 3
    for depth in (1, 2, 4):
        assert summary["latency"][str(depth)]["calls"] == 1000
    assert summary["measurements_sha256"] == sha256_file(profile_dir / "measurements.jsonl")
    return {
        "mean_event_ms_r1": summary["latency"]["1"]["event_ms"]["mean"],
        "effective_tokens_per_second_r1": summary["latency"]["1"]["effective_tokens_per_second"],
        "speedup_vs_r2": summary["latency"]["1"]["speedup_vs_r2"],
        "summary_sha256": sha256_file(profile_dir / "summary.json"),
    }


def decide_baseline(delta, speedup):
    if delta < 0 and speedup > 0:
        return "controlled_replication"
    if delta >= 0 and speedup > 0.05:
        return "efficiency_only_redesign"
    return "stop_or_fundamental_redesign"


def main():
    root = Path("runs/next_2026_10_26/baseline_100m")
    output = Path("runs/next_2026_10_26/baseline_100m_validation.json")
    protocol = json.loads((root / "protocol.json").read_text())
    results = {arm: validate_arm(root, arm, protocol) for arm in ("loop_r1", "dense_parameter")}
    profiles = {arm: validate_profile(root, arm) for arm in results}
    result = {
        "source": source_identity(),
        "protocol_sha256": sha256_file(root / "protocol.json"),
        "arms": results,
        "profiles": profiles,
        "loop_minus_dense_nll": results["loop_r1"]["r1_nll"] - results["dense_parameter"]["r1_nll"],
        "dense_is_better": results["dense_parameter"]["r1_nll"] < results["loop_r1"]["r1_nll"],
        "loop_speedup_vs_dense": profiles["dense_parameter"]["mean_event_ms_r1"] / profiles["loop_r1"]["mean_event_ms_r1"] - 1,
        "decision_protocol_sha256": sha256_file(root / "decision_protocol.json"),
        "next_stage_decision": decide_baseline(results["loop_r1"]["r1_nll"] - results["dense_parameter"]["r1_nll"], profiles["dense_parameter"]["mean_event_ms_r1"] / profiles["loop_r1"]["mean_event_ms_r1"] - 1),
        "all_completion_checks_passed": True,
    }
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
