from __future__ import annotations

import hashlib
import itertools
import json

import torch

STOP_ACTION = 0
PAIR_ACTIONS = 28
JOINT_ACTIONS = 29


def make_pair_table(experts: int = 8) -> torch.Tensor:
    if experts != 8:
        raise ValueError("pair-v1 requires exactly 8 experts")
    return torch.tensor(list(itertools.combinations(range(experts), 2)), dtype=torch.long)


def pair_index(first: int, second: int, experts: int = 8) -> int:
    if not 0 <= first < second < experts:
        raise ValueError("expected 0 <= first < second < experts")
    return first * (2 * experts - first - 1) // 2 + second - first - 1


def pair_action_to_ids(pair: tuple[int, int] | torch.Tensor, experts: int = 8) -> int | torch.Tensor:
    if isinstance(pair, tuple):
        return 1 + pair_index(*pair, experts=experts)
    pair = torch.as_tensor(pair, dtype=torch.long)
    first, second = pair.unbind(dim=-1)
    if bool(((first < 0) | (second <= first) | (second >= experts)).any()):
        raise ValueError("invalid expert pair")
    return 1 + first * (2 * experts - first - 1) // 2 + second - first - 1


def pair_for_action(action: int, table: torch.Tensor) -> torch.Tensor:
    if not 1 <= action <= PAIR_ACTIONS:
        raise ValueError("pair action must be in [1,28]")
    if table.shape != (PAIR_ACTIONS, 2):
        raise ValueError("expected pair table [28,2]")
    return table[action - 1].clone()


def legal_mask(stop_allowed: torch.Tensor) -> torch.Tensor:
    if stop_allowed.ndim != 1 or stop_allowed.dtype != torch.bool:
        raise ValueError("stop_allowed must be [N] bool")
    mask = torch.ones(stop_allowed.shape[0], JOINT_ACTIONS, dtype=torch.bool, device=stop_allowed.device)
    mask[:, STOP_ACTION] = stop_allowed
    return mask


def action_table_identity() -> dict:
    table = make_pair_table().tolist()
    encoded = json.dumps(table, separators=(",", ":")).encode()
    return {"table": table, "sha256": hashlib.sha256(encoded).hexdigest(), "feature_contract": "pair28_stop_v1"}
