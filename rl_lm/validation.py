"""Shared strict input checks; semantic checks run on explicit validation."""
import math


def check_schema(value, schema, path="config"):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    unknown, missing = set(value) - set(schema), set(schema) - set(value)
    if unknown:
        raise ValueError(f"unknown fields at {path}: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing fields at {path}: {sorted(missing)}")
    for key, expected in schema.items():
        item, name = value[key], f"{path}.{key}"
        if isinstance(expected, dict):
            check_schema(item, expected, name)
        elif not isinstance(item, expected) or (isinstance(item, bool) and expected is not bool):
            raise ValueError(f"{name} has invalid type {type(item).__name__}")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{name} must be finite")


def validate_numeric_settings(config):
    """Operational domains, not fixed architecture/spec constraints."""
    data = config.data

    def bound(path, lower=0, *, strict=False, upper=None):
        value = config.get(path)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(f"{path} must be a finite number")
        if value < lower or (strict and value == lower) or (upper is not None and value > upper):
            raise ValueError(f"{path} is outside its supported range")

    bound("data.sampling_seed")
    if "actor_seed" in data:
        bound("actor_seed", upper=2**32 - 1)  # numpy.random.seed contract
    for key in ("train_states", "selection_states", "calibration_states", "prefix_sequence_batch"):
        bound(f"data.{key}", strict=True)
    for key in ("expert_chunk", "head_chunk"):
        bound(f"environment.{key}", strict=True)
    bound("logging.interval_updates", strict=True)
    if config.get("environment.gamma_source") != "backbone_config":
        raise ValueError("unsupported gamma_source")
    if "evaluation" in data:
        for key in data["evaluation"]:
            path = f"evaluation.{key}"
            if key == "bootstrap_confidence":
                bound(path, strict=True, upper=1)
                if config.get(path) == 1:
                    raise ValueError("bootstrap_confidence must be below 1")
            else:
                bound(path, strict=key in ("interval_updates", "online_selection_sequences", "stochastic_samples_per_context", "bootstrap_repetitions"))
    section = "optimization" if "optimization" in data else "adaptation"
    opt = data[section]
    for key in ("learning_rate", "epsilon", "gradient_clip", "batch_contexts"):
        bound(f"{section}.{key}", strict=True)
    bound(f"{section}.weight_decay")
    updates_key = "updates" if section == "optimization" else "max_updates"
    bound(f"{section}.{updates_key}", strict=True)
    bound(f"{section}.warmup_updates")
    if opt["warmup_updates"] >= opt[updates_key]:
        raise ValueError(f"{section}.warmup_updates must be shorter than {updates_key}")
    betas = opt["betas"]
    if len(betas) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1 or (isinstance(v, float) and not math.isfinite(v)) for v in betas):
        raise ValueError(f"{section}.betas must contain two finite numbers in [0, 1)")
    if section == "optimization":
        if opt["algorithm"] != "on_policy_rloo":
            raise ValueError("unsupported optimization.algorithm")
        bound("optimization.final_lr_ratio", upper=1)
        for key in ("max_update_kl_mean", "max_update_kl_state", "max_retries_per_update", "max_lr_backoffs_per_run"):
            bound(f"optimization.{key}")
        if config.get("policy.greedy_rule") != "joint_argmax_lowest_id" or config.get("policy.price_as_input"):
            raise ValueError("unsupported greedy rule or price input")
    else:
        for key in ("final_learning_rate", "update_error_epsilon", "eval_interval_updates"):
            bound(f"adaptation.{key}", strict=True)
        if opt["final_learning_rate"] > opt["learning_rate"]:
            raise ValueError("final_learning_rate exceeds learning_rate")
