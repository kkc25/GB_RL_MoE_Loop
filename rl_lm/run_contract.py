"""Role-aware verified cache loading and fail-closed run approval contracts.

The contract binds the *resolved* configuration and verified payload identities,
not a config filename or a stale ``status: ready`` flag. Confirmation is named
by the configuration but is deliberately never loaded by training/preflight.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rl_lm.cache import cache_identity, load_cache

CONTRACT_SCHEMA = "rl-lm-run-contract-v1"
ROLES = (
    ("controller_train", "train_split", "train_states"),
    ("selection", "selection_split", "selection_states"),
    ("calibration", "calibration_split", "calibration_states"),
)


def contract_hash(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require_role_count(config, role, cache):
    states_key = next(states for name, _, states in ROLES if name == role)
    requested = config.get(f"data.{states_key}")
    actual = cache["manifest"]["n"]
    if actual != requested:
        raise ValueError(f"{role} cache state count mismatch: requested {requested}, actual {actual}")
    return actual


def load_role_caches(config, model, *, single=False):
    """Verify model/raw-data/payload identity and exact count for every role."""
    caches = {}
    for role, split_key, _ in ROLES:
        split = config.get(f"data.{split_key}")
        common = dict(checkpoint_path=config.get("backbone.checkpoint_path"),
                      config_path=config.get("backbone.config_path"),
                      tokenizer_path=config.get("data.tokenizer_path"),
                      data_dir=config.get("data.corpus_dir"), split=split,
                      sampling_seed=config.get("data.sampling_seed"),
                      expert_chunk=config.get("environment.expert_chunk"))
        if single:
            from rl_lm.single_config import single_cache_identity
            # Derived identities must refer to verified source payloads too.
            pair = load_cache(Path(config.get("data.pair_cache_root")) / split, {})
            require_role_count(config, role, pair)
            expected = single_cache_identity(model, pair_manifest=pair["manifest"], **common)
        else:
            expected = cache_identity(model, model.config,
                                      backend_version=config.get("environment.training_backend"),
                                      head_contract=config.get("environment.head_contract"), **common)
        cache = load_cache(Path(config.get("data.cache_root")) / split, expected)
        require_role_count(config, role, cache)
        caches[role] = cache
    return caches, build_run_contract(config, caches)


def build_run_contract(config, caches):
    roles = {}
    for role, split_key, states_key in ROLES:
        cache = caches[role]
        actual = require_role_count(config, role, cache)
        if not cache.get("verified_identity"):
            raise ValueError(f"{role} requires a verified cache identity")
        roles[role] = {"split": config.get(f"data.{split_key}"),
                       "requested_states": config.get(f"data.{states_key}"),
                       "actual_states": actual,
                       "verified_identity": cache["verified_identity"]}
    resolved = config.to_dict()
    # Match exact-resume semantics: moving the same run between devices is
    # allowed, but dtype/backend/chunks/require_cuda and every other setting
    # remain bound. The preflight environment still records its actual device.
    resolved["runtime"]["device"] = "<device-agnostic>"
    return {"schema": CONTRACT_SCHEMA, "resolved_config": resolved, "roles": roles}


def run_contract_identity(contract):
    return {"contract_schema": CONTRACT_SCHEMA, "contract_sha256": contract_hash(contract)}


def approval_record(contract):
    return {"status": "ready", **run_contract_identity(contract), "contract": contract}


def require_approval(report, contract=None):
    if report.get("status") != "ready":
        raise RuntimeError(f"preflight status is {report.get('status')!r}; trainer requires 'ready'")
    stored = report.get("contract")
    if (report.get("contract_schema") != CONTRACT_SCHEMA or not isinstance(stored, dict)
            or stored.get("schema") != CONTRACT_SCHEMA
            or report.get("contract_sha256") != contract_hash(stored)):
        raise RuntimeError("preflight approval contract missing or invalid; rerun preflight in a fresh output directory (legacy populated runs cannot be recertified in place)")
    if contract is not None and (stored != contract or report["contract_sha256"] != contract_hash(contract)):
        raise RuntimeError("preflight approval does not match current resolved config/verified caches; rerun preflight")
