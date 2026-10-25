from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import torch

from data.recovery import audit_batch, audit_dataset, build_datasets, build_manifest, recovery_collate
from model import LoopMoE
from recovery_config import RecoveryConfig
from train.common import MetricWriter, seed_everything
from train.recovery import evaluate_recovery, train_recovery
from train.recovery_common import RecoveryLedger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("d0", "train", "evaluate"), default="train")
    parser.add_argument("--output-dir")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--fixed-depth", type=int)
    parser.add_argument("--depth-schedule", choices=("fixed", "random"))
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RecoveryConfig.load(args.config)
    run = config.run
    for name in ("output_dir", "updates", "batch_size", "learning_rate", "device", "dtype", "fixed_depth", "depth_schedule", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(run, name, value)
    output_dir = Path(run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "resolved_config.json")
    datasets = build_datasets(config.task.stage, config.task.seed)
    manifest = build_manifest(config.task.stage, config.task.seed)
    (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    audits = {}
    failures = []
    for split, dataset in datasets.items():
        report = audit_dataset(dataset)
        audits[split] = asdict(report)
        failures.extend(f"{split}: {error}" for field in (report.label_mismatches, report.prefix_conflicts, report.query_errors, report.target_errors) for error in field)
        for start in range(0, len(dataset), run.batch_size):
            batch = recovery_collate([dataset[index] for index in range(start, min(start + run.batch_size, len(dataset)))])
            failures.extend(f"{split}: {error}" for error in audit_batch(batch, dataset.modulus))
    (output_dir / "d0_audit.json").write_text(json.dumps({"splits": audits, "failures": failures}, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"D0 audit failed with {len(failures)} errors")
    summary = {"stage": config.task.stage, "data_hash": manifest["sha256"], "d0_pass": True, "splits": {name: len(dataset) for name, dataset in datasets.items()}}
    if args.mode == "d0":
        (output_dir / "stage_result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return
    seed_everything(run.seed)
    device = torch.device(run.device)
    model = LoopMoE(config.model).to(device)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "moe_layers": config.model.moe_layers,
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    writer = MetricWriter(output_dir / "metrics.jsonl")
    ledger = RecoveryLedger()
    if args.mode == "train":
        train_recovery(model, datasets, recovery_collate, config, device, ledger, writer, manifest["sha256"])
        best_path = output_dir / "best_validation.pt"
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(best_checkpoint["model"])
            summary["checkpoint_selection"] = "best_validation"
    final_metrics = {}
    for split, dataset in datasets.items():
        final_metrics[split] = evaluate_recovery(model, dataset, recovery_collate, config, device, run.fixed_depth)
        writer.write(phase="final", split=split, **final_metrics[split])
    summary["environment"] = environment
    summary["metrics"] = final_metrics
    summary["ledger"] = asdict(ledger)
    if config.task.stage.startswith("c0"):
        summary["stage_pass"] = final_metrics["train"]["accuracy"] >= 0.99
    elif config.task.stage == "c1":
        summary["stage_pass"] = final_metrics["train"]["accuracy"] >= 0.99 and final_metrics["id_validation"]["accuracy"] >= 0.90
    else:
        summary["stage_pass"] = "not_preregistered"
    (output_dir / "stage_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
