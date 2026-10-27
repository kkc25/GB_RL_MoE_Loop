"""S1 single-expert actor: joint 9-class distribution (spec 18.3).

Action 0 is STOP; actions 1..8 select experts 0..7.  The actor head over
experts is 8-dimensional; the joint distribution over 9 actions is
factorized exactly like the pair branch:

    log p(0)      = logsigmoid(raw_stop)                      (if allowed)
    log p(a>0)    = logsigmoid(-raw_stop)
                    + log_softmax(log q + raw_expert)[a - 1]

with the prior ``q = 0.9 * one_hot(argmax gate) + 0.1 / 8``.  STOP is
illegal at ``r == 0`` (its joint entry is ``-inf``); reaching ``Rmax``
terminates without an extra STOP log-prob (handled by the environment).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rl_lm.single_executor import JOINT_SINGLE_ACTIONS, SINGLE_ACTIONS


def single_joint_log_probs(raw_expert, raw_stop, reference, stop_allowed):
    if raw_expert.ndim != 2 or raw_expert.shape[1] != SINGLE_ACTIONS:
        raise ValueError("raw_expert must have shape [N,8]")
    n = raw_expert.shape[0]
    if raw_stop.shape != (n,) or stop_allowed.shape != (n,) or stop_allowed.dtype != torch.bool:
        raise ValueError("raw_stop and stop_allowed must be [N] with bool mask")
    if reference.shape != raw_expert.shape:
        raise ValueError("reference shape mismatch")
    with torch.autocast(device_type=raw_expert.device.type, enabled=False):
        q = reference.detach().float()
        if not bool(torch.isfinite(q).all()) or not bool((q > 0).all()):
            raise ValueError("reference must be finite and positive")
        if not torch.allclose(q.sum(-1), torch.ones_like(q[:, 0]), atol=1e-6, rtol=1e-6):
            raise ValueError("reference must sum to one")
        expert = F.log_softmax(q.log() + raw_expert.float(), dim=-1)
        stop = F.logsigmoid(raw_stop.float())
        cont = F.logsigmoid(-raw_stop.float())
        first = torch.where(stop_allowed, stop, torch.full_like(stop, -torch.inf))
        rest = torch.where(stop_allowed[:, None], cont[:, None] + expert, expert)
        return torch.cat((first[:, None], rest), dim=-1)


class SingleActor(nn.Module):
    def __init__(self, feature_dim: int, hidden_width: int = 128, train_stop: bool = False):
        super().__init__()
        if hidden_width != 128:
            raise ValueError("single-v1 actor hidden width is fixed at 128")
        self.feature_dim = feature_dim
        self.trunk = nn.Sequential(nn.Linear(feature_dim, hidden_width), nn.SiLU(), nn.Linear(hidden_width, hidden_width), nn.SiLU())
        self.expert_head = nn.Linear(hidden_width, SINGLE_ACTIONS)
        self.stop_head = nn.Linear(hidden_width, 1)
        nn.init.zeros_(self.expert_head.weight)
        nn.init.zeros_(self.expert_head.bias)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.zeros_(self.stop_head.bias)
        self.stop_head.requires_grad_(train_stop)
        self.float()

    def forward(self, features, reference, stop_allowed):
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"features must be [N,{self.feature_dim}]")
        with torch.autocast(device_type=features.device.type, enabled=False):
            hidden = self.trunk(features.detach().float())
            raw_expert = self.expert_head(hidden)
            raw_stop = self.stop_head(hidden).squeeze(-1)
            return single_joint_log_probs(raw_expert, raw_stop, reference, stop_allowed)


def single_greedy_actions(log_probs: torch.Tensor) -> torch.Tensor:
    if log_probs.ndim != 2 or log_probs.shape[1] != JOINT_SINGLE_ACTIONS:
        raise ValueError("expected joint log-probs [N,9]")
    return log_probs.argmax(dim=-1)


class SingleDecisionRecords:
    """Decision records for the single-expert branch (9 joint actions)."""

    def __init__(self, features, reference, stop_allowed, actions, flat_episode_ids, old_log_probs_all):
        self.features = features
        self.reference = reference
        self.stop_allowed = stop_allowed
        self.actions = actions
        self.flat_episode_ids = flat_episode_ids
        self.old_log_probs_all = old_log_probs_all
        if features.ndim != 2:
            raise ValueError("features must be [K,F]")
        count = features.shape[0]
        if features.dtype != torch.float32 or reference.dtype != torch.float32 or old_log_probs_all.dtype != torch.float32:
            raise ValueError("records tensors must be FP32")
        if tuple(reference.shape) != (count, SINGLE_ACTIONS):
            raise ValueError("reference must be [K,8]")
        if tuple(stop_allowed.shape) != (count,) or stop_allowed.dtype != torch.bool:
            raise ValueError("stop_allowed must be [K] bool")
        if tuple(actions.shape) != (count,) or actions.dtype != torch.int64:
            raise ValueError("actions must be int64 [K]")
        if tuple(flat_episode_ids.shape) != (count,) or flat_episode_ids.dtype != torch.int64:
            raise ValueError("flat_episode_ids must be int64 [K]")
        if tuple(old_log_probs_all.shape) != (count, JOINT_SINGLE_ACTIONS):
            raise ValueError("old_log_probs_all must be [K,9]")
        if bool((actions < 0).any()) or bool((flat_episode_ids < 0).any()):
            raise ValueError("records cannot contain padding")

    @property
    def K(self) -> int:
        return self.features.shape[0]
