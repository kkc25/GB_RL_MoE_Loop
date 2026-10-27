from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch

from rl_lm.validation import check_schema as _check_schema, validate_numeric_settings

from rl_lm.executor import CANONICAL_BACKEND, SPARSE_BACKEND, MERGED_BACKEND, ACTIVE_MERGED_BACKEND


BACKBONE = {
    "config_path": str,
    "checkpoint_path": (str, type(None)),
    "expected_checkpoint_sha256": str,
    "adaptive_layer": int,
    "freeze_all": bool,
    "prefix_depth": int,
}
DATA = {
    "corpus_dir": (str, type(None)),
    "tokenizer_path": (str, type(None)),
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
POLICY = {
    "action_space": str,
    "feature_schema": str,
    "hidden_width": int,
    "reference_epsilon": (float, int),
    "initial_stop_hazard": (float, int),
    "distribution": str,
    "greedy_rule": str,
    "dropout": (float, int),
    "price_as_input": bool,
}
ENVIRONMENT = {
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
OPTIMIZATION = {
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
}
ROOT = {
    "spec_version": str,
    "stage": str,
    "actor_seed": int,
    "backbone": BACKBONE,
    "data": DATA,
    "policy": POLICY,
    "environment": ENVIRONMENT,
    "optimization": OPTIMIZATION,
    "cost": dict,
    "evaluation": dict,
    "runtime": dict,
    "logging": dict,
    "output_dir": str,
}
FREE = {
    "cost": {"lambda": (float, int), "unit": str},
    "evaluation": {
        "interval_updates": int,
        "stochastic_samples_per_context": int,
        "online_selection_sequences": int,
        "bootstrap_repetitions": int,
        "bootstrap_confidence": (float, int),
        "minimum_practical_nll_gain": (float, int),
        "r2_quality_tolerance_nll": (float, int),
        "selection_tie_tolerance": (float, int),
    },
    "runtime": {"device": str, "require_cuda": bool, "allow_tf32": bool},
    "logging": {"interval_updates": int},
}
SCHEMA = ROOT | FREE


def _canonical_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class RLConfig:
    def __init__(self, data):
        self.data = copy.deepcopy(data)
        _check_schema(self.data, SCHEMA)

    @classmethod
    def from_dict(cls, data):
        return cls(data)

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

    def validate(self, *, runtime=False, backbone_model=None):
        _check_schema(self.data, SCHEMA)
        validate_numeric_settings(self)
        if self.get("policy.feature_schema") != "h_h0_gate_previouspair_scalars_v1":
            raise ValueError("invalid pair feature schema")
        stage = self.get("stage")
        if stage not in ("r1", "r2"):
            raise ValueError("stage must be r1 or r2")
        expected_calls = 1 if stage == "r1" else 2
        if self.get("environment.min_calls") != 1 or self.get("environment.max_calls") != expected_calls:
            raise ValueError("stage and call limits disagree")
        if self.get("spec_version") != "RL-LM-Pair-v1.0":
            raise ValueError("unsupported spec_version")
        if self.get("optimization.groups") < 2:
            raise ValueError("groups must be >=2")
        if self.get("optimization.batch_contexts") <= 0:
            raise ValueError("batch_contexts must be positive")
        if self.get("optimization.updates_per_rollout") != 1:
            raise ValueError("exactly one update per rollout is required")
        for key in ("entropy_coefficient", "reference_kl_coefficient"):
            if self.get(f"optimization.{key}") != 0:
                raise ValueError(f"{key} must be zero in pair-v1")
        if self.get("optimization.advantage_normalization") != "none":
            raise ValueError("advantage normalization is forbidden")
        if self.get("policy.action_space") != "pair28_stop_v1" or self.get("policy.distribution") != "factorized_joint29":
            raise ValueError("invalid pair-v1 action distribution")
        if self.get("policy.hidden_width") != 128 or self.get("policy.dropout") != 0:
            raise ValueError("actor architecture mismatch")
        if self.get("policy.reference_epsilon") != 0.1 or self.get("policy.initial_stop_hazard") != 0.5:
            raise ValueError("prior constants changed")
        if self.get("environment.frozen_compute_dtype") != "bfloat16" or self.get("environment.residual_dtype") != "float32":
            raise ValueError("mixed or unsupported environment dtype")
        if self.get("environment.training_backend") != CANONICAL_BACKEND:
            raise ValueError("training requires the fixed-shape canonical backend")
        if self.get("environment.deployment_backend") not in (CANONICAL_BACKEND, SPARSE_BACKEND, MERGED_BACKEND, ACTIVE_MERGED_BACKEND):
            raise ValueError("unsupported pair deployment backend")
        if self.get("environment.head_contract") != "bf16_linear_fp32_ce_v1":
            raise ValueError("unsupported head contract")
        if self.get("cost.unit") != "extra_pair_round" or self.get("cost.lambda") < 0:
            raise ValueError("invalid cost unit or lambda")
        if stage == "r1" and self.get("cost.lambda") != 0:
            raise ValueError("R1 requires lambda=0")
        if self.get("backbone.adaptive_layer") != 11 or not self.get("backbone.freeze_all") or self.get("backbone.prefix_depth") != 1:
            raise ValueError("only frozen terminal layer 11 is supported")
        if self.get("data.train_split") == "newtest" or self.get("data.train_split") == self.get("data.confirmation_split"):
            raise ValueError("training cannot read confirmation data")
        if self.get("data.state_sampling") != "uniform_valid_token_without_replacement":
            raise ValueError("unsupported sampling")
        if self.get("optimization.replay_microbatch_contexts") <= 0:
            raise ValueError("invalid replay microbatch")
        if backbone_model is not None:
            if getattr(backbone_model, "top_k", None) != 2 or getattr(backbone_model, "experts_per_layer", None) != 8:
                raise ValueError("backbone must use top-2 with 8 experts")
            if tuple(getattr(backbone_model, "adaptive_layers", ())) != (11,):
                raise ValueError("backbone terminal adaptive layer mismatch")
        if runtime and self.get("runtime.require_cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but unavailable")
        return self

    def validate_checkpoint_hash(self, actual_sha256):
        if actual_sha256 != self.get("backbone.expected_checkpoint_sha256"):
            raise ValueError("actual checkpoint hash does not match the frozen expectation")
        return self

    def validate_required_paths(self):
        for key in ("backbone.checkpoint_path", "data.corpus_dir", "data.tokenizer_path"):
            if not self.get(key):
                raise ValueError(f"{key} must be provided before running")
        return self


class CorpusConfig:
    SCHEMA = {
        "spec_version": str,
        "source_jsonl": (str, type(None)),
        "source_id": str,
        "source_revision": (str, type(None)),
        "tokenizer_path": (str, type(None)),
        "exclude_corpus_dirs": list,
        "additional_exclusion_jsonl": list,
        "output_dir": str,
        "sequence_length": int,
        "tokens": dict,
        "allow_existing_output": bool,
    }

    def __init__(self, data):
        self.data = copy.deepcopy(data)
        _check_schema(self.data, self.SCHEMA)
        _check_schema(self.data["tokens"], {"controller_train": int, "selection": int, "calibration": int, "newtest": int}, "tokens")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def to_dict(self):
        return copy.deepcopy(self.data)

    @property
    def sha256(self):
        return _canonical_hash(self.data)

    def validate_required(self):
        _check_schema(self.data, self.SCHEMA)
        _check_schema(self.data["tokens"], {"controller_train": int, "selection": int, "calibration": int, "newtest": int}, "tokens")
        if self.data["sequence_length"] <= 0 or any(value <= 0 for value in self.data["tokens"].values()):
            raise ValueError("corpus sequence length and token budgets must be positive")
        for key in ("exclude_corpus_dirs", "additional_exclusion_jsonl"):
            if any(not isinstance(value, str) or not value for value in self.data[key]):
                raise ValueError(f"{key} must contain nonempty paths")
        if self.data["spec_version"] != "RL-LM-Corpus-v1.0":
            raise ValueError("unsupported corpus spec")
        for key in ("source_jsonl", "source_revision", "tokenizer_path"):
            if self.data[key] is None:
                raise ValueError(f"{key} is required")
        if not self.data["exclude_corpus_dirs"] and not self.data["additional_exclusion_jsonl"]:
            raise ValueError("real runs require at least one exclusion source")
        if self.data["allow_existing_output"]:
            raise ValueError("existing output cannot be overwritten")
        return self
