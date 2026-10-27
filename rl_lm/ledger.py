"""Logical vs physical resource ledger for the pair-RL environment.

Counts follow spec section 14.2: logical counters describe the actions the
environment actually committed (pair rounds, expert-token calls, recorded
decisions), while physical counters describe everything the fixed-shape
backend really computed, including STOP filler rows, zero padding, reference
and diagnostic head readouts.  Discarded results still count physically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import torch


@dataclass
class RLLedger:
    prefix_input_tokens: int = 0
    unique_train_contexts: int = 0
    context_visits: int = 0
    rollout_episodes: int = 0
    logical_pair_rounds: int = 0
    logical_expert_rows: int = 0
    physical_expert_rows: int = 0
    physical_head_rows: int = 0
    actor_decision_rows: int = 0
    actor_padding_rows: int = 0
    successful_updates: int = 0
    optimizer_attempts: int = 0
    gpu_hours: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0

    def merge_peak(self, device: torch.device) -> None:
        if device.type == "cuda":
            self.peak_allocated_bytes = max(self.peak_allocated_bytes, torch.cuda.max_memory_allocated(device))
            self.peak_reserved_bytes = max(self.peak_reserved_bytes, torch.cuda.max_memory_reserved(device))

    def record_wall_time(self, wall_time_seconds: float, *, gpus: int = 1) -> None:
        """Accumulate GPU-hours from an actually measured wall-time interval.

        Task card E found that ``gpu_hours`` was declared but never
        accumulated; this is the single accumulation entry point.  Only
        measured durations may be passed: runs saved before this method
        existed keep their recorded (zero) value, and their authoritative
        cost stays the ``wall_time_seconds`` written next to the ledger in
        ``cost_ledger.json`` — history is not retroactively fabricated here.
        """
        seconds = float(wall_time_seconds)
        if not seconds >= 0.0:
            raise ValueError("wall_time_seconds must be nonnegative")
        if int(gpus) < 1:
            raise ValueError("gpus must be a positive integer")
        self.gpu_hours += seconds * int(gpus) / 3600.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RLLedger":
        known = {field.name for field in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown ledger fields: {sorted(unknown)}")
        missing = known - set(data)
        if missing:
            raise ValueError(f"missing ledger fields: {sorted(missing)}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "RLLedger":
        return cls.from_dict(json.loads(Path(path).read_text()))
