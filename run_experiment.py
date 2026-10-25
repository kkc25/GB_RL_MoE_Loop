from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
import torch

from config import ExperimentConfig
from eval.verify_router_math import run as verify_math
from model import LoopMoE
from policy import Controller
from train.common import CostLedger, MetricWriter, seed_everything
from train.exact import train_exact
from train.halt import train_halt
from train.imitation import train_imitation
from train.pretrain import pretrain
from train.router_rloo import evaluate, train_rloo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("p0", "pretrain", "rloo", "exact", "halt", "imitation", "evaluate"), default="p0")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--updates", type=int)
    parser.add_argument("--pretrain-updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-token-limit", type=int)
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.load(args.config)
    train = config.train
    for name in ("output_dir", "device", "dtype", "updates", "pretrain_updates", "batch_size", "input_token_limit", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(train, name, value)
    output_dir = Path(train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "resolved_config.json")
    math_result = verify_math()
    (output_dir / "math_checks.json").write_text(json.dumps(math_result, indent=2) + "\n")
    seed_everything(train.seed)
    device = torch.device(train.device)
    model = LoopMoE(config.model).to(device)
    controller = Controller(config.model, config.controller).to(device)
    optimizer = None
    ledger = CostLedger()
    start_step, start_offset = 0, 0
    if train.resume:
        checkpoint = torch.load(train.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        controller.load_state_dict(checkpoint["controller"])
        ledger = CostLedger(**checkpoint["ledger"])
        start_step = checkpoint["step"]
        start_offset = checkpoint["offset"]
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "controller_parameters": sum(parameter.numel() for parameter in controller.parameters()),
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    writer = MetricWriter(output_dir / "metrics.jsonl")
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if args.mode in ("p0", "pretrain"):
        pretrain(model, config, device, ledger, writer, controller)
    if args.mode in ("p0", "pretrain", "evaluate"):
        for depth in range(1, config.model.max_calls + 1):
            original = config.model.max_calls
            config.model.max_calls = depth
            metrics = evaluate(model, None, config, device, routed=False)
            config.model.max_calls = original
            writer.write(phase="validation", algorithm=f"fixed_depth_{depth}", **metrics)
    if args.mode in ("p0", "rloo"):
        train_rloo(model, controller, config, device, ledger, writer, start_step, start_offset)
    elif args.mode == "exact":
        train_exact(model, controller, config, device, ledger, writer)
    elif args.mode == "halt":
        train_halt(model, config, device, ledger, writer)
    elif args.mode == "imitation":
        train_imitation(model, controller, config, device, ledger, writer)
    ledger.elapsed_seconds += time.monotonic() - started
    ledger.update_peak(device)
    ledger.save(output_dir / "cost_ledger.json")
    final = {"mode": args.mode, "environment": environment, "ledger": asdict(ledger)}
    if args.mode in ("p0", "rloo", "exact", "imitation", "evaluate"):
        final["routed_validation"] = evaluate(model, controller, config, device)
        (output_dir / "final_validation.json").write_text(json.dumps(final["routed_validation"], indent=2) + "\n")
    (output_dir / "run_summary.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
