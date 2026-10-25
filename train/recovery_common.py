from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import torch


@dataclass
class RecoveryLedger:
    optimizer_updates: int = 0
    supervised_targets: int = 0
    input_tokens: int = 0
    rollouts: int = 0
    expert_calls: int = 0
    elapsed_seconds: float = 0.0
    allocated_gpu_hours: float = 0.0
    lost_compute_seconds: float = 0.0
    peak_memory_bytes: int = 0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, destination)


def atomic_torch_save(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def recovery_checkpoint(path, config, model, optimizer, update, ledger, data_hash):
    atomic_torch_save(
        path,
        {
            "config": asdict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "ledger": asdict(ledger),
            "data_hash": data_hash,
            "label_contract_version": "query-prefix-v1",
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    )


def query_labels(batch: dict) -> torch.Tensor:
    if "labels" in batch:
        return batch["labels"]
    labels = torch.full_like(batch["input_ids"], -100)
    rows = torch.arange(batch["input_ids"].shape[0], device=batch["input_ids"].device)
    labels[rows, batch["query_positions"]] = batch["target_answer_ids"]
    return labels
