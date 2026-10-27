from __future__ import annotations

from pathlib import Path
import json

import torch

from lm_config import LMExperimentConfig
from model import LoopMoE
from rl_lm.config import RLConfig
from train.recovery_common import sha256_file


def resolve_config(path, *, checkpoint=None, corpus_dir=None, tokenizer=None, device=None):
    config = RLConfig.load(path)
    overrides = {
        "backbone.checkpoint_path": checkpoint,
        "data.corpus_dir": corpus_dir,
        "data.tokenizer_path": tokenizer,
        "runtime.device": device,
    }
    for dotted, value in overrides.items():
        if value is None:
            continue
        parts = dotted.split(".")
        target = config.data
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"unknown CLI override: {dotted}")
        target[parts[-1]] = value
    config.validate_required_paths()
    config.validate()
    return config


def load_backbone(config, device):
    model_config = LMExperimentConfig.load(config.get("backbone.config_path"))
    model = LoopMoE(model_config.model).to(device).eval()
    checkpoint_path = config.get("backbone.checkpoint_path")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    config.validate(backbone_model=model_config.model, runtime=True)
    config.validate_checkpoint_hash(sha256_file(checkpoint_path))
    return model, checkpoint


def write_resolved_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
