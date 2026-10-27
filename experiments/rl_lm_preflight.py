from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import torch

from rl_lm.run_contract import CONTRACT_SCHEMA, approval_record, load_role_caches
from rl_lm.cli import load_backbone, resolve_config, write_resolved_config
from rl_lm.run_directory import guard_run_directory
from train.recovery_common import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--corpus-dir")
    parser.add_argument("--tokenizer")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = resolve_config(args.config, checkpoint=args.checkpoint, corpus_dir=args.corpus_dir, tokenizer=args.tokenizer, device=args.device)
    output_dir = Path(config.get("output_dir"))
    guard_run_directory(output_dir)
    # Invalidate any previous approval before model, raw-data or cache checks.
    # Atomic replacement also makes crashes fail closed ("checking", not ready).
    atomic_json(output_dir / "preflight.json", {"status": "checking", "contract_schema": CONTRACT_SCHEMA})
    try:
        write_resolved_config(config, output_dir / "resolved_config.json")
        device = torch.device(config.get("runtime.device"))
        model, _ = load_backbone(config, device)
        caches, contract = load_role_caches(config, model)
        environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        }
        report = approval_record(contract) | {
            "model_and_raw_data": "verified", "environment": environment,
            "caches": {role: {"status": "ready", "states": cache["manifest"]["n"],
                              "expected_states": contract["roles"][role]["requested_states"]}
                       for role, cache in caches.items()},
        }
        atomic_json(output_dir / "environment.json", environment)
        atomic_json(output_dir / "preflight.json", report)
    except Exception as exc:
        atomic_json(output_dir / "preflight.json", {
            "status": "failed", "contract_schema": CONTRACT_SCHEMA,
            "exception_type": type(exc).__name__, "message": str(exc),
        })
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
