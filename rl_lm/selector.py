from __future__ import annotations

import torch

from rl_lm.actions import PAIR_ACTIONS, make_pair_table


def canonical_pair(gate_logits: torch.Tensor, table: torch.Tensor | None = None) -> torch.Tensor:
    if gate_logits.ndim != 2 or gate_logits.shape[1] != 8:
        raise ValueError("gate_logits must be [N,8]")
    if gate_logits.dtype != torch.float32:
        gate_logits = gate_logits.float()
    table = make_pair_table() if table is None else table.to(gate_logits.device)
    if table.shape != (PAIR_ACTIONS, 2):
        raise ValueError("expected pair table [28,2]")
    scores = gate_logits[:, table].sum(dim=-1)
    return scores.argmax(dim=-1)


def pair_reference(gate_logits: torch.Tensor, table: torch.Tensor | None = None, epsilon: float = 0.1) -> torch.Tensor:
    if epsilon != 0.1:
        raise ValueError("pair-v1 reference epsilon is fixed at 0.1")
    chosen = canonical_pair(gate_logits, table)
    reference = torch.full((gate_logits.shape[0], PAIR_ACTIONS), epsilon / PAIR_ACTIONS, dtype=torch.float32, device=gate_logits.device)
    reference.scatter_(1, chosen[:, None], 0.9 + epsilon / PAIR_ACTIONS)
    return reference
