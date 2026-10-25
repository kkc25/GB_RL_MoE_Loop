from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
import torch

from config import ExperimentConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, dtype: str):
    if device.type != "cuda" or dtype == "float32":
        return nullcontext()
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    return torch.autocast(device_type="cuda", dtype=dtypes[dtype])


def move_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["input_ids"].to(device), batch["labels"].to(device)


@dataclass
class CostLedger:
    input_tokens: int = 0
    path_forward_tokens: int = 0
    backward_tokens: int = 0
    oracle_forward_tokens: int = 0
    optimizer_updates: int = 0
    elapsed_seconds: float = 0.0
    peak_allocated_bytes: int = 0

    def update_peak(self, device: torch.device) -> None:
        if device.type == "cuda":
            self.peak_allocated_bytes = max(self.peak_allocated_bytes, torch.cuda.max_memory_allocated(device))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


class MetricWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **metrics) -> None:
        record = {"time": time.time(), **metrics}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")


def learning_rate_at(step: int, total: int, peak: float, warmup_fraction: float) -> float:
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


def save_checkpoint(path: str | Path, config: ExperimentConfig, model, controller, optimizer, step: int, offset: int, ledger: CostLedger) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "model": model.state_dict(),
        "controller": controller.state_dict() if controller is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "offset": offset,
        "ledger": asdict(ledger),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, destination)
