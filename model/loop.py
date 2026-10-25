from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class BudgetState:
    remaining: torch.Tensor
    requested: torch.Tensor
    rounds: torch.Tensor
    future_layers: torch.Tensor

    def can_continue(self, max_calls: int) -> torch.Tensor:
        return (self.rounds < max_calls) & (self.remaining >= self.future_layers + 1)


def initial_budget(batch: int, tokens: int, budget: int | torch.Tensor, adaptive_layers: int, device: torch.device) -> BudgetState:
    if isinstance(budget, int):
        requested = torch.full((batch, tokens), budget, dtype=torch.long, device=device)
    else:
        requested = budget.to(device=device, dtype=torch.long)
    return BudgetState(
        remaining=requested.clone(),
        requested=requested,
        rounds=torch.zeros_like(requested),
        future_layers=torch.full_like(requested, adaptive_layers - 1),
    )


def validate_budget(requested: torch.Tensor, adaptive_layers: int, max_calls: int) -> None:
    minimum = adaptive_layers
    maximum = adaptive_layers * max_calls
    if bool(((requested < minimum) | (requested > maximum)).any()):
        raise ValueError(f"budget must be in [{minimum}, {maximum}]")


def episode_random_keys(input_ids: torch.Tensor, requested: torch.Tensor, route_seed: int, episode_ids: torch.Tensor | None = None, supervised: torch.Tensor | None = None) -> torch.Tensor:
    if episode_ids is not None:
        return episode_ids.reshape(-1).to(device=input_ids.device, dtype=torch.long) ^ route_seed
    keys = torch.full(input_ids.shape, route_seed, dtype=torch.long, device=input_ids.device)
    running = torch.full((input_ids.shape[0],), route_seed, dtype=torch.long, device=input_ids.device)
    for position in range(input_ids.shape[1]):
        running = running * 1099511628211 ^ (input_ids[:, position] + 31 * position + 1)
        keys[:, position] = running
    if supervised is not None:
        keys = keys[torch.arange(input_ids.shape[0], device=input_ids.device), supervised]
    else:
        keys = keys.reshape(-1)
    return keys ^ (requested * 0x9E3779B1)


def splitmix_uniform(keys: torch.Tensor, layer: int, rounds: torch.Tensor, counters: torch.Tensor, stream: int) -> torch.Tensor:
    values = keys + layer * 0x9E3779B97F4A7C15 + rounds * 0xBF58476D1CE4E5B9 + counters * 0x94D049BB133111EB + stream * 0xD6E8FEB86659FD93
    values = (values ^ (values >> 30)) * 0xBF58476D1CE4E5B9
    values = (values ^ (values >> 27)) * 0x94D049BB133111EB
    values = values ^ (values >> 31)
    return (values & ((1 << 53) - 1)).double() / float(1 << 53)
