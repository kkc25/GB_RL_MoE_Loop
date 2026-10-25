"""E01 value students: equal-capacity MLPs regressing per-action costs.

Two students of identical capacity are trained on the same legal deploy-time
input — the cached pre-FFN hidden state ``h`` only, never labels, costs of
other states, or any future information.  One regresses the immediate T=1
cost of each candidate action, the other regresses Q^{pi0} (the T=rounds
cost with baseline continuation).  At deployment each student picks the
argmin of its predicted cost vector and the chosen path is really executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


class ValueStudent(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def student_features(states: dict) -> torch.Tensor:
    """The only legal deploy-time input: the cached hidden state h."""
    return states["h"].float()


def standardization(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std computed on the train split only (no validation/test leakage)."""
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return mean, std


@dataclass
class StudentResult:
    student: ValueStudent
    mean: torch.Tensor
    std: torch.Tensor
    history: list = field(default_factory=list)
    best_validation_mse: float = float("inf")
    optimizer_updates: int = 0


def fit_value_student(
    features: torch.Tensor,
    costs: torch.Tensor,
    *,
    hidden_dim: int,
    epochs: int,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    weight_decay: float = 0.0,
    validation: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> StudentResult:
    """Fit one student by MSE on per-action costs; keep best-validation weights."""
    if features.shape[0] != costs.shape[0] or features.ndim != 2 or costs.ndim != 2:
        raise ValueError(f"bad shapes: features {tuple(features.shape)}, costs {tuple(costs.shape)}")
    device = features.device
    mean, std = standardization(features)
    x = (features - mean) / std
    y = costs.float()
    torch.manual_seed(seed)
    student = ValueStudent(features.shape[1], costs.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = StudentResult(student, mean, std)
    best_state = None
    validation_mse = float("nan")
    for epoch in range(epochs):
        order = torch.randperm(x.shape[0], generator=generator).to(device)
        total_loss = 0.0
        for start in range(0, x.shape[0], batch_size):
            indices = order[start : start + batch_size]
            prediction = student(x[indices])
            loss = F.mse_loss(prediction, y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            result.optimizer_updates += 1
            total_loss += loss.item() * indices.numel()
        train_mse = total_loss / x.shape[0]
        if validation is not None:
            validation_mse = mse(student, validation[0], validation[1], mean, std)
            if validation_mse < result.best_validation_mse:
                result.best_validation_mse = validation_mse
                best_state = {key: value.detach().clone() for key, value in student.state_dict().items()}
        result.history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": validation_mse})
    if best_state is not None:
        student.load_state_dict(best_state)
    return result


@torch.no_grad()
def predict(student: ValueStudent, features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    student.eval()
    return student((features.float() - mean) / std)


@torch.no_grad()
def mse(student: ValueStudent, features: torch.Tensor, costs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> float:
    return F.mse_loss(predict(student, features, mean, std), costs.float()).item()
