from __future__ import annotations

import itertools
import torch

from policy.rollout import replay_logprob
from train.common import autocast_context


def all_paths(experts: int, max_calls: int) -> list[tuple[int, ...]]:
    return [path for depth in range(1, max_calls + 1) for path in itertools.product(range(experts), repeat=depth)]


def legal_paths(experts: int, budget: int, max_calls: int) -> list[tuple[int, ...]]:
    """Unique executable paths under a budget: forced paths deeper than
    min(budget, max_calls) are truncated by the legality rule, so deeper
    enumerations only duplicate shallower trajectories."""
    return all_paths(experts, min(budget, max_calls))


@torch.no_grad()
def oracle_path_metrics(model, controller, input_ids, labels, budget: int, price: float, device: torch.device, dtype: str) -> dict[str, float]:
    paths = legal_paths(model.config.experts_per_layer, budget, model.config.max_calls)
    losses, costs = [], []
    with autocast_context(device, dtype):
        for path in paths:
            output = model(input_ids, labels, controller=controller, budget=budget, price=price, forced_path=path, collect_rollout=True, keep_logits=False)
            losses.append(output.token_losses.float())
            costs.append(output.rollout.costs.float())
    objectives = torch.stack(losses) + price * torch.stack(costs)
    best, indices = objectives.min(dim=0)
    reference = objectives[0]
    return {
        "paths": len(paths),
        "oracle_loss": best.mean().item(),
        "reference_loss": reference.mean().item(),
        "oracle_gap": (reference - best).mean().item(),
        "mean_oracle_depth": torch.tensor([len(paths[index]) for index in indices], device=device).float().mean().item(),
    }


def exact_path_batch(model, controller, input_ids, labels, budget: int, price: float, device: torch.device, dtype: str):
    paths = legal_paths(model.config.experts_per_layer, budget, model.config.max_calls)
    for path in paths:
        with torch.no_grad(), autocast_context(device, dtype):
            output = model(input_ids, labels, controller=controller, budget=budget, price=price, forced_path=path, collect_rollout=True, keep_logits=False)
        objective = output.rollout.token_losses.float() + price * output.rollout.costs.float()
        with autocast_context(device, dtype):
            log_probability = replay_logprob(controller, output.rollout)
            if not model.config.target_only:
                log_probability = log_probability.reshape(input_ids.shape[0], input_ids.shape[1]).sum(dim=-1)
        yield path, objective, log_probability
