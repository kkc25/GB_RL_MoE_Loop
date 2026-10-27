"""Atomic checkpoint save, exact resume and actor-only warmstart (spec 12).

``save_atomic`` writes a temporary file in the same directory, flushes and
fsyncs it, then ``os.replace``-es it onto the target; the pointer manifest
(``checkpoint_manifest.json``) is updated only after the data file is in
place.  A checkpoint is always written after a fully successful update —
never in the middle of a rollout or a rejected optimizer transaction.

``resume_exact`` reloads a checkpoint of the *same* run: format/spec
versions, the resolved config (only ``runtime.device`` may differ) and the
internal consistency of the stored counters are validated before anything is
handed back.  ``warmstart_actor`` is a different operation: it loads only the
actor weights plus the parent identity for a *new* run and refuses contract
mismatches.  The two entry points are never interchangeable.

The RNG bundle covers Python ``random``, NumPy's global generator, torch CPU
and every visible CUDA device, plus the version of the stateless Philox
sampler (``rl-lm-rng-v1``) used for rollout uniforms.
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from train.recovery_common import atomic_json, sha256_file

CHECKPOINT_FORMAT = "rl-lm-checkpoint-v1"
SPEC_VERSION = "RL-LM-Pair-v1.0"
SAMPLER_VERSION = "rl-lm-rng-v1"
POINTER_MANIFEST = "checkpoint_manifest.json"


def save_atomic(path, payload: dict) -> Path:
    """Write ``payload`` to ``path`` atomically (tmp file + fsync + replace)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def update_pointer_manifest(run_dir, kind: str, checkpoint_path) -> None:
    """Record ``kind -> {file, sha256}`` after the data file is fully written."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / POINTER_MANIFEST
    manifest = json_load(manifest_path) if manifest_path.exists() else {}
    checkpoint_path = Path(checkpoint_path)
    manifest[kind] = {"file": checkpoint_path.name, "sha256": sha256_file(checkpoint_path)}
    atomic_json(manifest_path, manifest)


def json_load(path):
    return json.loads(Path(path).read_text())


def capture_rng_state() -> dict:
    """Snapshot every RNG the trainer may touch plus the sampler version."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": list(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,
        "sampler": SAMPLER_VERSION,
    }


def restore_rng_state(state: dict) -> None:
    if state.get("sampler") != SAMPLER_VERSION:
        raise ValueError(f"stateless sampler version mismatch: {state.get('sampler')!r}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda = state.get("cuda")
    if cuda is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint carries CUDA RNG state but CUDA is unavailable")
        if len(cuda) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG count does not match visible GPU count")
        torch.cuda.set_rng_state_all([entry.cpu() for entry in cuda])


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move loaded optimizer state tensors onto the actor's device in place."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _normalized_for_resume(config_dict: dict) -> dict:
    normalized = copy.deepcopy(config_dict)
    normalized["runtime"]["device"] = "<device-agnostic>"
    return normalized


def configs_match_for_resume(stored: dict, current: dict) -> bool:
    """Exact resume may change only ``runtime.device`` (spec 13.4)."""
    return _normalized_for_resume(stored) == _normalized_for_resume(current)


def load_checkpoint(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if payload.get("spec_version") != SPEC_VERSION:
        raise ValueError(f"unsupported spec_version in checkpoint: {payload.get('spec_version')!r}")
    return payload


def _check_counter_consistency(payload: dict, path) -> None:
    training = payload["training_state"]["successful_updates"]
    scheduler = payload["scheduler_state"]["successful_updates"]
    ledger = payload["ledger"]["successful_updates"]
    if not (training == scheduler == ledger):
        raise ValueError(
            f"{path}: inconsistent successful_updates "
            f"(training={training}, scheduler={scheduler}, ledger={ledger})"
        )


def resume_exact(path, *, resolved_config: dict) -> dict:
    """Load a checkpoint of the same run and validate the resume contract."""
    payload = load_checkpoint(path)
    if not configs_match_for_resume(payload["resolved_config"], resolved_config):
        raise ValueError(
            "exact resume requires the identical resolved configuration; "
            "only runtime.device and the --stop-after lifetime cap may change"
        )
    for key in (
        "actor_state_dict",
        "optimizer_state_dict",
        "scheduler_state",
        "training_state",
        "rng_state",
        "selection_state",
        "ledger",
    ):
        if key not in payload:
            raise ValueError(f"checkpoint {path} is missing required section {key!r}")
    _check_counter_consistency(payload, path)
    return payload


def warmstart_actor(path, *, feature_schema: str, feature_dim: int, action_sha256: str) -> dict:
    """Load *only* the actor weights of a parent checkpoint for a new run."""
    payload = load_checkpoint(path)
    contract = payload.get("action_contract") or {}
    if contract.get("sha256") != action_sha256:
        raise ValueError("parent checkpoint action table does not match this run's contract")
    if payload.get("feature_schema") != feature_schema or payload.get("feature_dim") != feature_dim:
        raise ValueError("parent checkpoint feature schema/dim does not match this run")
    if "actor_state_dict" not in payload:
        raise ValueError("parent checkpoint has no actor_state_dict")
    _check_counter_consistency(payload, path)
    return payload
