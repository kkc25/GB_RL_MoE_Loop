from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from config import ModelConfig


@dataclass
class RecoveryTaskConfig:
    stage: str = "c0_identity"
    seed: int = 20261024


@dataclass
class RecoveryRunConfig:
    output_dir: str = "runs/recovery"
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 11
    batch_size: int = 128
    updates: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_updates: int = 20
    gradient_clip: float = 1.0
    validation_interval: int = 100
    log_interval: int = 20
    fixed_depth: int = 1
    depth_schedule: str = "fixed"
    shuffle_each_epoch: bool = True
    resume: str | None = None
    gpu_hour_cap: float = 12.0


@dataclass
class RecoveryConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    task: RecoveryTaskConfig = field(default_factory=RecoveryTaskConfig)
    run: RecoveryRunConfig = field(default_factory=RecoveryRunConfig)

    @classmethod
    def load(cls, path: str | Path) -> RecoveryConfig:
        raw = json.loads(Path(path).read_text())
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            task=RecoveryTaskConfig(**raw.get("task", {})),
            run=RecoveryRunConfig(**raw.get("run", {})),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n")
