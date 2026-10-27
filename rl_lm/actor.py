from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rl_lm.actions import JOINT_ACTIONS, PAIR_ACTIONS


def joint_log_probs(raw_pair, raw_stop, pair_reference, stop_allowed):
    if raw_pair.ndim != 2 or raw_pair.shape[1] != PAIR_ACTIONS:
        raise ValueError("raw_pair must have shape [N,28]")
    n = raw_pair.shape[0]
    if raw_stop.shape != (n,) or stop_allowed.shape != (n,) or stop_allowed.dtype != torch.bool:
        raise ValueError("raw_stop and stop_allowed must be [N] with bool mask")
    if pair_reference.shape != raw_pair.shape:
        raise ValueError("pair_reference shape mismatch")
    with torch.autocast(device_type=raw_pair.device.type, enabled=False):
        q = pair_reference.detach().float()
        if not bool(torch.isfinite(q).all()) or not bool((q > 0).all()):
            raise ValueError("reference must be finite and positive")
        if not torch.allclose(q.sum(-1), torch.ones_like(q[:, 0]), atol=1e-6, rtol=1e-6):
            raise ValueError("reference must sum to one")
        pair = F.log_softmax(q.log() + raw_pair.float(), dim=-1)
        stop = F.logsigmoid(raw_stop.float())
        cont = F.logsigmoid(-raw_stop.float())
        first = torch.where(stop_allowed, stop, torch.full_like(stop, -torch.inf))
        rest = torch.where(stop_allowed[:, None], cont[:, None] + pair, pair)
        return torch.cat((first[:, None], rest), dim=-1)


class PairActor(nn.Module):
    def __init__(self, feature_dim: int, hidden_width: int = 128, train_stop: bool = False):
        super().__init__()
        if hidden_width != 128:
            raise ValueError("pair-v1 actor hidden width is fixed at 128")
        self.feature_dim = feature_dim
        self.trunk = nn.Sequential(nn.Linear(feature_dim, hidden_width), nn.SiLU(), nn.Linear(hidden_width, hidden_width), nn.SiLU())
        self.pair_head = nn.Linear(hidden_width, PAIR_ACTIONS)
        self.stop_head = nn.Linear(hidden_width, 1)
        nn.init.zeros_(self.pair_head.weight)
        nn.init.zeros_(self.pair_head.bias)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.zeros_(self.stop_head.bias)
        self.stop_head.requires_grad_(train_stop)
        self.float()

    def forward(self, features, pair_reference, stop_allowed):
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"features must be [N,{self.feature_dim}]")
        with torch.autocast(device_type=features.device.type, enabled=False):
            hidden = self.trunk(features.detach().float())
            raw_pair = self.pair_head(hidden)
            raw_stop = self.stop_head(hidden).squeeze(-1)
            return joint_log_probs(raw_pair, raw_stop, pair_reference, stop_allowed)


def greedy_actions(log_probs: torch.Tensor) -> torch.Tensor:
    if log_probs.ndim != 2 or log_probs.shape[1] != JOINT_ACTIONS:
        raise ValueError("expected joint log-probs [N,29]")
    return log_probs.argmax(dim=-1)
