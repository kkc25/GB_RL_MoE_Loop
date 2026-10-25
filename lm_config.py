from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from config import ModelConfig


@dataclass
class LMTrainConfig:
    data_dir: str = "runs/data/fineweb_edu_debug"
    output_dir: str = "runs/e00/debug67_b2"
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 17
    batch_size: int = 1
    updates: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_fraction: float = 0.02
    gradient_clip: float = 1.0
    validation_interval: int = 50
    validation_batches: int = 16
    log_interval: int = 10
    depth_schedule: str = "random"
    fixed_depth: int = 4
    resume: str | None = None


@dataclass
class LMExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: LMTrainConfig = field(default_factory=LMTrainConfig)

    @classmethod
    def load(cls, path: str | Path) -> LMExperimentConfig:
        raw = json.loads(Path(path).read_text())
        return cls(model=ModelConfig(**raw.get("model", {})), train=LMTrainConfig(**raw.get("train", {})))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n")
