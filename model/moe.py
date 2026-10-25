from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class ExpertPool(nn.Module):
    def __init__(self, dim: int, intermediate: int, experts: int) -> None:
        super().__init__()
        self.experts = experts
        self.w1 = nn.Parameter(torch.empty(experts, intermediate, dim))
        self.w3 = nn.Parameter(torch.empty(experts, intermediate, dim))
        self.w2 = nn.Parameter(torch.empty(experts, dim, intermediate))
        for weight in (self.w1, self.w3, self.w2):
            nn.init.normal_(weight, std=0.02)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat_x = x.reshape(-1, shape[-1])
        flat_indices = indices.reshape(-1)
        output = torch.zeros_like(flat_x)
        for expert in range(self.experts):
            locations = torch.nonzero(flat_indices == expert, as_tuple=False).squeeze(-1)
            if locations.numel() == 0:
                continue
            selected = flat_x.index_select(0, locations)
            hidden = F.silu(F.linear(selected, self.w1[expert])) * F.linear(selected, self.w3[expert])
            output.index_copy_(0, locations, F.linear(hidden, self.w2[expert]).to(flat_x.dtype))
        return output.reshape(shape)


class MoE(nn.Module):
    def __init__(self, dim: int, intermediate: int, experts: int) -> None:
        super().__init__()
        self.pool = ExpertPool(dim, intermediate, experts)
        self.gate = nn.Linear(dim, experts, bias=False)

    def forward(self, x: torch.Tensor, indices: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(x)
        probabilities = F.softmax(gate_logits.float(), dim=-1).to(x.dtype)
        if indices is None:
            indices = gate_logits.argmax(dim=-1)
        amplitude = probabilities.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        update = self.pool(x, indices) * amplitude.unsqueeze(-1)
        return update, gate_logits, probabilities, indices


class DenseSwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate, bias=False)
        self.w3 = nn.Linear(dim, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x: torch.Tensor, indices: torch.Tensor | None = None):
        update = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return update, None, None, indices
