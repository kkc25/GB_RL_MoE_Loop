"""S1 single-expert branch: config schemas, validation and cache identity.

``RL-LM-Single-v1.0`` is an independent spec version (spec section 18).  It
never reuses the pair-v1 frozen-environment manifest: the S1 cache carries
``cache_format = "rl-lm-single-cache-v1"`` and ``environment_version =
"single8_stop_v1"``.  The prefix payload (``h0``) is derived from the pair
cache only after every prefix-defining hash (checkpoint/config/tokenizer/
packed data/layer/gamma/prefix depth and dtype) is re-verified against the
loaded teacher; the single baseline records (argmax expert, single-expert
teacher CE at amplitude 1) are always recomputed, never inherited.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch

from rl_lm.single_executor import (
    SINGLE_CANONICAL_BACKEND,
    SINGLE_FEATURE_SCHEMA,
    SINGLE_SPARSE_BACKEND,
    SINGLE_SPEC_VERSION,
)

SINGLE_CACHE_FORMAT = "rl-lm-single-cache-v1"
SINGLE_ENVIRONMENT_VERSION = "single8_stop_v1"

_BACKBONE = {
    "config_path": str,
    "checkpoint_path": (str, type(None)),
    "expected_checkpoint_sha256": str,
    "adaptive_layer": int,
    "freeze_all": bool,
    "prefix_depth": int,
}
_DATA = {
    "corpus_dir": (str, type(None)),
    "tokenizer_path": (str, type(None)),
    "pair_cache_root": str,
    "cache_root": str,
    "train_split": str,
    "selection_split": str,
    "calibration_split": str,
    "confirmation_split": str,
    "train_states": int,
    "selection_states": int,
    "calibration_states": int,
    "state_sampling": str,
    "sampling_seed": int,
    "prefix_sequence_batch": int,
}
_ENVIRONMENT = {
    "min_calls": int,
    "max_calls": int,
    "gamma_source": str,
    "training_backend": str,
    "deployment_backend": str,
    "expert_chunk": int,
    "head_chunk": int,
    "frozen_compute_dtype": str,
    "residual_dtype": str,
    "head_contract": str,
}

ADAPT_SCHEMA = {
    "spec_version": str,
    "stage": str,
    "backbone": _BACKBONE,
    "data": _DATA,
    "adaptation": {
        "learning_rate": (float, int),
        "final_learning_rate": (float, int),
        "betas": list,
        "epsilon": (float, int),
        "weight_decay": (float, int),
        "gradient_clip": (float, int),
        "batch_contexts": int,
        "max_updates": int,
        "warmup_updates": int,
        "kl_coefficient": (float, int),
        "update_error_coefficient": (float, int),
        "update_error_epsilon": (float, int),
        "eval_interval_updates": int,
        "selection_gate_tolerance_nll": (float, int),
        "trainable_whitelist": list,
    },
    "environment": _ENVIRONMENT,
    "runtime": {"device": str, "require_cuda": bool, "allow_tf32": bool},
    "logging": {"interval_updates": int},
    "output_dir": str,
}

SINGLE_RLOO_SCHEMA = {
    "spec_version": str,
    "stage": str,
    "actor_seed": int,
    "backbone": _BACKBONE,
    "adapted": {
        "adapt_run_dir": str,
        "checkpoint_path": (str, type(None)),
        "expected_checkpoint_sha256": (str, type(None)),
    },
    "data": _DATA,
    "policy": {
        "action_space": str,
        "feature_schema": str,
        "hidden_width": int,
        "reference_epsilon": (float, int),
        "distribution": str,
        "greedy_rule": str,
        "dropout": (float, int),
        "price_as_input": bool,
    },
    "environment": _ENVIRONMENT,
    "optimization": {
        "algorithm": str,
        "groups": int,
        "batch_contexts": int,
        "replay_microbatch_contexts": int,
        "updates": int,
        "learning_rate": (float, int),
        "betas": list,
        "epsilon": (float, int),
        "weight_decay": (float, int),
        "warmup_updates": int,
        "final_lr_ratio": (float, int),
        "gradient_clip": (float, int),
        "entropy_coefficient": (float, int),
        "reference_kl_coefficient": (float, int),
        "advantage_normalization": str,
        "updates_per_rollout": int,
        "max_update_kl_mean": (float, int),
        "max_update_kl_state": (float, int),
        "max_retries_per_update": int,
        "max_lr_backoffs_per_run": int,
    },
    "cost": {"lambda": (float, int), "unit": str},
    "evaluation": {
        "interval_updates": int,
        "online_selection_sequences": int,
        "selection_tie_tolerance": (float, int),
    },
    "runtime": {"device": str, "require_cuda": bool, "allow_tf32": bool},
    "logging": {"interval_updates": int},
    "output_dir": str,
}


def _check_schema(value, schema, path="config"):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    unknown = set(value) - set(schema)
    if unknown:
        raise ValueError(f"unknown fields at {path}: {sorted(unknown)}")
    missing = set(schema) - set(value)
    if missing:
        raise ValueError(f"missing fields at {path}: {sorted(missing)}")
    for key, expected in schema.items():
        item = value[key]
        if isinstance(expected, dict):
            _check_schema(item, expected, f"{path}.{key}")
        elif expected is list:
            if not isinstance(item, list):
                raise ValueError(f"{path}.{key} must be a list")
        elif not isinstance(item, expected) or isinstance(item, bool) and expected is not bool and expected != (float, int):
            raise ValueError(f"{path}.{key} has invalid type {type(item).__name__}")


def _canonical_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _BaseSingleConfig:
    SCHEMA = None

    def __init__(self, data):
        self.data = copy.deepcopy(data)
        _check_schema(self.data, self.SCHEMA)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def to_dict(self):
        return copy.deepcopy(self.data)

    @property
    def sha256(self):
        return _canonical_hash(self.data)

    def get(self, dotted):
        value = self.data
        for part in dotted.split("."):
            value = value[part]
        return value

    def _validate_common(self, *, runtime=False, backbone_model=None):
        if self.get("spec_version") != SINGLE_SPEC_VERSION:
            raise ValueError("unsupported spec_version for the single branch")
        if self.get("backbone.adaptive_layer") != 11 or not self.get("backbone.freeze_all") or self.get("backbone.prefix_depth") != 1:
            raise ValueError("only terminal layer 11 with prefix_depth 1 is supported")
        if self.get("environment.frozen_compute_dtype") != "bfloat16" or self.get("environment.residual_dtype") != "float32":
            raise ValueError("mixed or unsupported environment dtype")
        if self.get("environment.head_contract") != "bf16_linear_fp32_ce_v1":
            raise ValueError("unsupported head contract")
        if self.get("environment.training_backend") != SINGLE_CANONICAL_BACKEND:
            raise ValueError("single training backend must be canonical_all_experts_single_v1")
        if self.get("environment.deployment_backend") != SINGLE_SPARSE_BACKEND:
            raise ValueError("single deployment backend must be sparse_single_v1")
        if self.get("environment.min_calls") != 1 or self.get("environment.max_calls") != 1:
            raise ValueError("S1 first version is R1 only (min_calls == max_calls == 1)")
        if self.get("data.train_split") == "newtest" or self.get("data.train_split") == self.get("data.confirmation_split"):
            raise ValueError("training cannot read confirmation data")
        if self.get("data.state_sampling") != "uniform_valid_token_without_replacement":
            raise ValueError("unsupported sampling")
        if backbone_model is not None:
            if getattr(backbone_model, "top_k", None) != 2 or getattr(backbone_model, "experts_per_layer", None) != 8:
                raise ValueError("backbone must keep top_k=2 with 8 experts; only the terminal executor is overridden")
            if tuple(getattr(backbone_model, "adaptive_layers", ())) != (11,):
                raise ValueError("backbone terminal adaptive layer mismatch")
        if runtime and self.get("runtime.require_cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but unavailable")
        return self

    def validate_required_paths(self):
        for key in ("backbone.checkpoint_path", "data.corpus_dir", "data.tokenizer_path"):
            if self.get(key) is None:
                raise ValueError(f"{key} must be provided before running")
        return self

    def validate_checkpoint_hash(self, actual_sha256):
        if actual_sha256 != self.get("backbone.expected_checkpoint_sha256"):
            raise ValueError("actual checkpoint hash does not match the frozen expectation")
        return self


class SingleAdaptConfig(_BaseSingleConfig):
    SCHEMA = ADAPT_SCHEMA

    def validate(self, *, runtime=False, backbone_model=None):
        if self.get("stage") != "adapt":
            raise ValueError("stage must be adapt")
        self._validate_common(runtime=runtime, backbone_model=backbone_model)
        fixed = {
            "adaptation.learning_rate": 1e-5,
            "adaptation.final_learning_rate": 1e-6,
            "adaptation.betas": [0.9, 0.95],
            "adaptation.weight_decay": 0.01,
            "adaptation.gradient_clip": 1.0,
            "adaptation.batch_contexts": 256,
            "adaptation.warmup_updates": 50,
            "adaptation.kl_coefficient": 1.0,
            "adaptation.update_error_coefficient": 0.1,
            "adaptation.update_error_epsilon": 1e-8,
            "adaptation.selection_gate_tolerance_nll": 0.02,
        }
        for key, expected in fixed.items():
            if self.get(key) != expected:
                raise ValueError(f"{key} is fixed at {expected!r} by spec 18.2/18.3")
        if not 1 <= self.get("adaptation.max_updates") <= 1000:
            raise ValueError("adaptation.max_updates must be in [1, 1000]")
        if self.get("adaptation.warmup_updates") >= self.get("adaptation.max_updates"):
            raise ValueError("warmup must be shorter than max_updates")
        expected_whitelist = [
            "blocks.11.moe.pool.w1",
            "blocks.11.moe.pool.w2",
            "blocks.11.moe.pool.w3",
        ]
        if list(self.get("adaptation.trainable_whitelist")) != expected_whitelist:
            raise ValueError("trainable whitelist must equal model.blocks[11].moe.pool w1/w2/w3")
        if self.get("adaptation.eval_interval_updates") != 100:
            raise ValueError("full selection evaluation is fixed at every 100 updates")
        return self


class SingleRLConfig(_BaseSingleConfig):
    SCHEMA = SINGLE_RLOO_SCHEMA

    def validate(self, *, runtime=False, backbone_model=None):
        if self.get("stage") != "r1":
            raise ValueError("single-v1 RL stage must be r1")
        self._validate_common(runtime=runtime, backbone_model=backbone_model)
        if self.get("policy.action_space") != "single8_stop_v1" or self.get("policy.distribution") != "factorized_joint9":
            raise ValueError("invalid single-v1 action distribution")
        if self.get("policy.feature_schema") != SINGLE_FEATURE_SCHEMA:
            raise ValueError("invalid single-v1 feature schema")
        if self.get("policy.hidden_width") != 128 or self.get("policy.dropout") != 0:
            raise ValueError("actor architecture mismatch")
        if self.get("policy.reference_epsilon") != 0.1:
            raise ValueError("prior constants changed")
        if self.get("cost.unit") != "extra_single_expert_round" or self.get("cost.lambda") < 0:
            raise ValueError("invalid cost unit or lambda")
        if self.get("cost.lambda") != 0:
            raise ValueError("R1 requires lambda=0")
        if self.get("optimization.groups") < 2:
            raise ValueError("groups must be >=2")
        if self.get("optimization.updates_per_rollout") != 1:
            raise ValueError("exactly one update per rollout is required")
        for key in ("entropy_coefficient", "reference_kl_coefficient"):
            if self.get(f"optimization.{key}") != 0:
                raise ValueError(f"{key} must be zero in single-v1")
        if self.get("optimization.advantage_normalization") != "none":
            raise ValueError("advantage normalization is forbidden")
        if self.get("optimization.replay_microbatch_contexts") <= 0:
            raise ValueError("invalid replay microbatch")
        return self

    def validate_adapted_hash(self, actual_sha256):
        expected = self.get("adapted.expected_checkpoint_sha256")
        if expected is not None and actual_sha256 != expected:
            raise ValueError("adapted student checkpoint hash does not match the config")
        return self


def single_cache_identity(model, *, checkpoint_path, config_path, tokenizer_path, data_dir, split, sampling_seed, expert_chunk, pair_manifest):
    """New S1 cache identity; prefix fields are re-verified against the pair manifest.

    ``pair_manifest`` is the pair-v1 cache manifest the payload derives from.
    Every prefix-defining field must match the freshly loaded teacher; the
    returned identity has the new cache format and environment version, so
    the pair frozen-environment manifest can never validate as an S1 cache.
    """
    from train.recovery_common import sha256_file

    layer = int(model.config.adaptive_layers[-1])
    prefix_fields = (
        "checkpoint_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "packed_sha256",
        "document_manifest_sha256",
        "layer_index",
        "gamma",
        "prefix_depth",
        "prefix_dtype",
    )
    fresh = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "packed_sha256": pair_manifest.get("packed_sha256"),
        "document_manifest_sha256": sha256_file(Path(data_dir) / "manifest.json"),
        "layer_index": layer,
        "gamma": float(model.gammas[layer]),
        "prefix_depth": 1,
        "prefix_dtype": "bfloat16",
    }
    for key in prefix_fields:
        if pair_manifest.get(key) != fresh[key]:
            raise ValueError(f"pair cache prefix field {key!r} does not match the loaded teacher; refusing to derive")
    return fresh | {
        "backend_version": SINGLE_CANONICAL_BACKEND,
        "expert_chunk": int(expert_chunk),
        "head_contract": "bf16_linear_fp32_ce_v1",
        "cache_format": SINGLE_CACHE_FORMAT,
        "environment_version": SINGLE_ENVIRONMENT_VERSION,
        "split": split,
        "sampling_seed": int(sampling_seed),
        "derived_from_pair_cache_sha256": hashlib.sha256(
            json.dumps(pair_manifest, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest(),
    }


def resolve_single_config(path, cls, *, checkpoint=None, corpus_dir=None, tokenizer=None, device=None):
    config = cls.load(path)
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
