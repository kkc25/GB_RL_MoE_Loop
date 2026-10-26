from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import torch

from data.packed_lm import PackedLMDataset
from lm_config import LMExperimentConfig
from model import LoopMoE
from train.common import MetricWriter, seed_everything
from train.lm import evaluate_lm, train_lm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--resume")
    parser.add_argument("--validation-batches", type=int)
    args = parser.parse_args()
    config = LMExperimentConfig.load(args.config)
    run = config.train
    for name in ("output_dir", "updates", "batch_size", "device", "learning_rate", "resume", "validation_batches"):
        value = getattr(args, name)
        if value is not None:
            setattr(run, name, value)
    output_dir = Path(run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "resolved_config.json")
    manifest = json.loads((Path(run.data_dir) / "manifest.json").read_text())
    seed_everything(run.seed)
    device = torch.device(run.device)
    model = LoopMoE(config.model).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    train_dataset = PackedLMDataset(run.data_dir, "train")
    validation_dataset = PackedLMDataset(run.data_dir, "validation")
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "parameters": parameters,
        "data_revision": manifest["revision"],
        "data_hash": manifest["document_list_sha256"],
        "tokenizer_hash": manifest["tokenizer"]["sha256"],
    }
    data_hash = hashlib.sha256(f'{manifest["document_list_sha256"]}:{manifest["tokenizer"]["sha256"]}'.encode()).hexdigest()
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    writer = MetricWriter(output_dir / "metrics.jsonl")
    ledger = train_lm(model, train_dataset, validation_dataset, config, device, writer, data_hash=data_hash)
    final = {}
    for depth in (1, 2, 4):
        final[f"depth_{depth}"] = evaluate_lm(model, validation_dataset, config, device, depth, run.validation_batches)
        writer.write(phase="final", **final[f"depth_{depth}"])
    summary = {"environment": environment, "ledger": ledger, "validation": final}
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
