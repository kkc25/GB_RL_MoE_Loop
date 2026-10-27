from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import torch

from rl_lm.cache import cache_identity, load_cache
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
    write_resolved_config(config, output_dir / "resolved_config.json")
    device = torch.device(config.get("runtime.device"))
    model, _ = load_backbone(config, device)
    caches = {}
    for role, states_key in (("controller_train", "train_states"), ("selection", "selection_states"), ("calibration", "calibration_states")):
        split = config.get(f"data.{role}_split") if role != "controller_train" else config.get("data.train_split")
        cache_dir = Path(config.get("data.cache_root")) / split
        if not (cache_dir / "manifest.json").exists():
            caches[role] = {"status": "not_built"}
            continue
        expected = cache_identity(model, model.config, checkpoint_path=config.get("backbone.checkpoint_path"), config_path=config.get("backbone.config_path"), tokenizer_path=config.get("data.tokenizer_path"), data_dir=config.get("data.corpus_dir"), split=split, sampling_seed=config.get("data.sampling_seed"), backend_version=config.get("environment.training_backend"), head_contract=config.get("environment.head_contract"), expert_chunk=config.get("environment.expert_chunk"))
        cache = load_cache(cache_dir, expected)
        caches[role] = {"status": "ready", "states": cache["manifest"]["n"], "expected_states": config.get(f"data.{states_key}"), "cache_dir": str(cache_dir)}
        if cache["manifest"]["n"] != config.get(f"data.{states_key}"):
            raise ValueError(f"{role} cache state count mismatch")
    report = {
        "status": "ready" if all(item["status"] == "ready" for item in caches.values()) else "not_built",
        "model_and_raw_data": "verified",
        "caches": caches,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    }
    atomic_json(output_dir / "preflight.json", report)
    atomic_json(output_dir / "environment.json", report["environment"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
