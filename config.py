from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class ModelConfig:
    layers: int = 4
    hidden_size: int = 128
    attention_heads: int = 4
    head_dim: int = 32
    vocab_size: int = 512
    intermediate_size: int = 256
    experts_per_layer: int = 4
    top_k: int = 1
    adaptive_layers: tuple[int, ...] = (3,)
    moe_layers: tuple[int, ...] | None = None
    min_calls: int = 1
    max_calls: int = 3
    max_position_embeddings: int = 256
    norm_eps: float = 1e-5
    rope_base: float = 10000.0
    adaptive_residual_scale: float = 0.5
    nonadaptive_residual_scale: float = 1.0
    dropout: float = 0.0
    target_only: bool = True

    def __post_init__(self) -> None:
        self.adaptive_layers = tuple(self.adaptive_layers)
        self.moe_layers = tuple(range(self.layers)) if self.moe_layers is None else tuple(self.moe_layers)
        if not set(self.adaptive_layers).issubset(self.moe_layers):
            raise ValueError("adaptive layers must use MoE")
        if self.hidden_size != self.attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal attention_heads * head_dim")
        if self.adaptive_layers and (min(self.adaptive_layers) < 0 or max(self.adaptive_layers) >= self.layers):
            raise ValueError("adaptive layer index out of range")
        if self.min_calls < 1 or self.max_calls < self.min_calls:
            raise ValueError("invalid call limits")
        if self.top_k not in (1, 2) or self.top_k > self.experts_per_layer:
            raise ValueError("top_k must be 1 or 2 and no larger than the expert pool")
        if self.target_only and self.adaptive_layers != (self.layers - 1,):
            raise ValueError("target_only supports exactly one terminal adaptive layer")


@dataclass
class ControllerConfig:
    projected_hidden: int = 128
    gru_hidden_size: int = 128
    reference_temperature: float = 0.25
    reference_tempered_weight: float = 0.9
    initial_stop_hazard: float = 0.5


@dataclass
class DataConfig:
    sequence_length: int = 32
    train_programs: int = 4096
    validation_programs: int = 1024
    test_programs: int = 1024
    min_depth: int = 1
    max_depth: int = 4
    seed: int = 20261024


@dataclass
class TrainConfig:
    output_dir: str = "runs/p0"
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 11
    batch_size: int = 64
    updates: int = 1000
    learning_rate: float = 3e-4
    controller_learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_fraction: float = 0.02
    gradient_clip: float = 1.0
    validation_interval: int = 100
    validation_batches: int = 8
    rollout_groups: int = 2
    budget: int = 3
    lambda_cost: float = 0.0
    input_token_limit: int = 20_000_000
    pretrain_updates: int = 0
    resume: str | None = None
    log_interval: int = 10


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, raw: dict) -> ExperimentConfig:
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            controller=ControllerConfig(**raw.get("controller", {})),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open() as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n")
