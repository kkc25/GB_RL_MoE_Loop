from __future__ import annotations

import itertools

import torch
from torch import nn
from torch.nn import functional as F

from .moe import ExpertPool, MoE

DEFAULT_EXPERTS = 8
ROUTER_WEIGHTS = "router"
UNIFORM_WEIGHTS = "uniform"


def num_top2_actions(experts: int = DEFAULT_EXPERTS) -> int:
    return experts * (experts - 1) // 2


def enumerate_top2_actions(experts: int = DEFAULT_EXPERTS, device: torch.device | None = None) -> torch.Tensor:
    pairs = list(itertools.combinations(range(experts), 2))
    return torch.tensor(pairs, dtype=torch.long, device=device)


def action_to_pair(actions: torch.Tensor, experts: int = DEFAULT_EXPERTS) -> torch.Tensor:
    table = enumerate_top2_actions(experts, device=actions.device)
    return table[actions]


def pair_to_action(pairs: torch.Tensor, experts: int = DEFAULT_EXPERTS) -> torch.Tensor:
    low = torch.minimum(pairs[..., 0], pairs[..., 1])
    high = torch.maximum(pairs[..., 0], pairs[..., 1])
    if bool((low == high).any()):
        raise ValueError("a Top-2 action must contain two distinct experts")
    if bool((high >= experts).any()) or bool((low < 0).any()):
        raise ValueError(f"expert index out of range for {experts} experts")
    return low * experts - low * (low + 1) // 2 + (high - low - 1)


def baseline_top2_pairs(gate_logits: torch.Tensor) -> torch.Tensor:
    top = gate_logits.topk(2, dim=-1).indices
    return top.sort(dim=-1).values


def top2_weights(gate_logits: torch.Tensor | None, pairs: torch.Tensor, rule: str = ROUTER_WEIGHTS, detached: bool = True) -> torch.Tensor:
    if rule == UNIFORM_WEIGHTS:
        return torch.full(pairs.shape, 0.5, dtype=torch.float32, device=pairs.device)
    if rule == ROUTER_WEIGHTS:
        if gate_logits is None:
            raise ValueError("gate_logits are required for router weights")
        source = gate_logits.detach() if detached else gate_logits
        gathered = source.float().gather(-1, pairs)
        return F.softmax(gathered, dim=-1)
    raise ValueError(f"unknown weight rule: {rule!r}")


def top2_dispatch(
    pool: ExpertPool,
    x: torch.Tensor,
    actions: torch.Tensor | None = None,
    pairs: torch.Tensor | None = None,
    gate_logits: torch.Tensor | None = None,
    weight_rule: str = ROUTER_WEIGHTS,
    weights: torch.Tensor | None = None,
    detach_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pairs is None:
        if actions is None:
            raise ValueError("either actions or pairs must be given")
        pairs = action_to_pair(actions, experts=pool.experts)
    if pairs.shape[:-1] != x.shape[:-1] or pairs.shape[-1] != 2:
        raise ValueError(f"pairs shape {tuple(pairs.shape)} incompatible with x shape {tuple(x.shape)}")
    if weights is None:
        if weight_rule == ROUTER_WEIGHTS and gate_logits is None:
            raise ValueError("gate_logits are required for router weights")
        weights = top2_weights(gate_logits, pairs, rule=weight_rule, detached=detach_weights)
    shape = x.shape
    flat_x = x.reshape(-1, shape[-1])
    flat_weights = weights.reshape(-1, 2).float()
    first = pool(flat_x, pairs[..., 0].reshape(-1))
    second = pool(flat_x, pairs[..., 1].reshape(-1))
    mixed = flat_weights[:, 0].unsqueeze(-1) * first.float() + flat_weights[:, 1].unsqueeze(-1) * second.float()
    return mixed.to(x.dtype).reshape(shape), weights, pairs


class Top2MoE(nn.Module):
    def __init__(self, moe: MoE, weight_rule: str = ROUTER_WEIGHTS, freeze: bool = True) -> None:
        super().__init__()
        self.moe = moe
        self.weight_rule = weight_rule
        self.freeze = freeze
        if freeze:
            for parameter in self.moe.parameters():
                parameter.requires_grad_(False)

    @property
    def experts(self) -> int:
        return self.moe.pool.experts

    def gate_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.moe.gate(x)
        return logits.detach() if self.freeze else logits

    def baseline_actions(self, x: torch.Tensor) -> torch.Tensor:
        return pair_to_action(baseline_top2_pairs(self.gate_logits(x)), experts=self.experts)

    def forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor | None = None,
        pairs: torch.Tensor | None = None,
        weight_rule: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate_logits(x)
        if actions is None and pairs is None:
            pairs = baseline_top2_pairs(logits)
        update, weights, pairs = top2_dispatch(
            self.moe.pool,
            x,
            actions=actions,
            pairs=pairs,
            gate_logits=logits,
            weight_rule=weight_rule or self.weight_rule,
            detach_weights=self.freeze,
        )
        return update, logits, weights, pairs
