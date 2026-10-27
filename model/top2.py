from __future__ import annotations

import itertools

import torch
from torch import nn
from torch.nn import functional as F

from .moe import ExpertPool, MoE

DEFAULT_EXPERTS = 8
ROUTER_WEIGHTS = "router"
UNIFORM_WEIGHTS = "uniform"

# Runtime-only execution identities: never persisted in model/checkpoint config.
TOP2_DISPATCH_CONTRACTS = {
    "legacy": "top2_two_slot_expert_pool_v1",
    "merged": "top2_merged_assignment_buckets_v1",
}


def _merged_slot_outputs(pool: ExpertPool, flat_x: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """One bucket/expert across both slots, without duplicating the full input.

    Retain separate slot outputs and the legacy FP32 two-term weighted sum,
    rather than atomic/scatter accumulation whose reduction order can differ.
    Expert computation and output cast match ExpertPool (including autocast).
    """
    assignments = pairs.reshape(-1)
    output = flat_x.new_zeros((assignments.numel(), flat_x.shape[-1]))
    for expert in range(pool.experts):
        locations = torch.nonzero(assignments == expert, as_tuple=False).squeeze(-1)
        if locations.numel() == 0:
            continue
        selected = flat_x.index_select(0, locations // 2)
        hidden = F.silu(F.linear(selected, pool.w1[expert])) * F.linear(selected, pool.w3[expert])
        output.index_copy_(0, locations, F.linear(hidden, pool.w2[expert]).to(flat_x.dtype))
    return output.reshape(flat_x.shape[0], 2, flat_x.shape[-1])


def set_top2_dispatch_backend(module: nn.Module, backend: str) -> int:
    """Explicitly select all descendant Top2MoE executors (returns their count).

    Pass only the intended prefix to limit scope. This does not alter standalone
    top2_dispatch callers, routing, config, state_dict, or canonical defaults.
    Call again with ``legacy`` to restore the original runtime path.
    """
    if backend not in TOP2_DISPATCH_CONTRACTS:
        raise ValueError(f"unknown Top-2 dispatch backend: {backend!r}")
    executors = [child for child in module.modules() if isinstance(child, Top2MoE)]
    for child in executors:
        child.dispatch_backend = backend
    return len(executors)


def top2_dispatch_identity(module: nn.Module) -> dict:
    """JSON-safe effective runtime identity; names identify selected layer paths.

    Model config/state_dict intentionally do not encode this execution choice.
    Profilers must record this identity; canonical cache/training integrations
    must explicitly reject nonlegacy executors until a promotion is validated.
    """
    return {
        "schema": "top2_dispatch_runtime_v1",
        "executors": [
            {"module": name, "backend": child.dispatch_backend,
             "contract": TOP2_DISPATCH_CONTRACTS[child.dispatch_backend]}
            for name, child in module.named_modules() if isinstance(child, Top2MoE)
        ],
    }


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
    backend: str = "legacy",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if backend not in TOP2_DISPATCH_CONTRACTS:
        raise ValueError(f"unknown Top-2 dispatch backend: {backend!r}")
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
    if backend == "legacy":
        first = pool(flat_x, pairs[..., 0].reshape(-1))
        second = pool(flat_x, pairs[..., 1].reshape(-1))
    else:
        slots = _merged_slot_outputs(pool, flat_x, pairs)
        first, second = slots[:, 0], slots[:, 1]
    mixed = flat_weights[:, 0].unsqueeze(-1) * first.float() + flat_weights[:, 1].unsqueeze(-1) * second.float()
    return mixed.to(x.dtype).reshape(shape), weights, pairs


class Top2MoE(nn.Module):
    def __init__(self, moe: MoE, weight_rule: str = ROUTER_WEIGHTS, freeze: bool = True) -> None:
        super().__init__()
        self.moe = moe
        self.weight_rule = weight_rule
        self.freeze = freeze
        self.dispatch_backend = "legacy"
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
            backend=self.dispatch_backend,
        )
        return update, logits, weights, pairs
