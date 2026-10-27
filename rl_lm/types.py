from __future__ import annotations

from dataclasses import dataclass

import torch


def _shape(value, expected, name):
    if tuple(value.shape) != tuple(expected):
        raise ValueError(f"{name} must have shape {tuple(expected)}, got {tuple(value.shape)}")


def _dtype(value, expected, name):
    if value.dtype != expected:
        raise ValueError(f"{name} must be {expected}, got {value.dtype}")


@dataclass
class ContextBatch:
    h0: torch.Tensor
    targets: torch.Tensor
    ref_ce: torch.Tensor
    episode_ids: tuple[str, ...]

    def __post_init__(self):
        if self.h0.ndim != 2:
            raise ValueError("h0 must be [B,D]")
        batch = self.h0.shape[0]
        _dtype(self.h0, torch.float32, "h0")
        _shape(self.targets, (batch,), "targets")
        _dtype(self.targets, torch.int64, "targets")
        _shape(self.ref_ce, (batch,), "ref_ce")
        _dtype(self.ref_ce, torch.float32, "ref_ce")
        if len(self.episode_ids) != batch:
            raise ValueError("episode_ids length must equal B")
        if self.h0.device != self.targets.device or self.h0.device != self.ref_ce.device:
            raise ValueError("context tensors must share device")
        if self.h0.requires_grad or self.targets.requires_grad or self.ref_ce.requires_grad:
            raise ValueError("context tensors must be detached")


@dataclass
class PolicyState:
    h: torch.Tensor
    h0: torch.Tensor
    rounds: torch.Tensor
    previous_pair: torch.Tensor

    def __post_init__(self):
        if self.h.ndim != 2 or self.h0.shape != self.h.shape:
            raise ValueError("h and h0 must be matching [N,D]")
        count = self.h.shape[0]
        _dtype(self.h, torch.float32, "h")
        _dtype(self.h0, torch.float32, "h0")
        _shape(self.rounds, (count,), "rounds")
        _dtype(self.rounds, torch.int64, "rounds")
        _shape(self.previous_pair, (count, 2), "previous_pair")
        _dtype(self.previous_pair, torch.int64, "previous_pair")
        if self.h.requires_grad or self.h0.requires_grad:
            raise ValueError("policy hidden state must be detached")


@dataclass
class DecisionRecords:
    features: torch.Tensor
    pair_reference: torch.Tensor
    stop_allowed: torch.Tensor
    actions: torch.Tensor
    flat_episode_ids: torch.Tensor
    old_log_probs_all: torch.Tensor

    def __post_init__(self):
        if self.features.ndim != 2:
            raise ValueError("features must be [K,F]")
        count = self.features.shape[0]
        _dtype(self.features, torch.float32, "features")
        _shape(self.pair_reference, (count, 28), "pair_reference")
        _dtype(self.pair_reference, torch.float32, "pair_reference")
        _shape(self.stop_allowed, (count,), "stop_allowed")
        _dtype(self.stop_allowed, torch.bool, "stop_allowed")
        _shape(self.actions, (count,), "actions")
        _dtype(self.actions, torch.int64, "actions")
        _shape(self.flat_episode_ids, (count,), "flat_episode_ids")
        _dtype(self.flat_episode_ids, torch.int64, "flat_episode_ids")
        _shape(self.old_log_probs_all, (count, 29), "old_log_probs_all")
        _dtype(self.old_log_probs_all, torch.float32, "old_log_probs_all")
        if bool((self.actions < 0).any()) or bool((self.flat_episode_ids < 0).any()):
            raise ValueError("records cannot contain padding")


@dataclass
class RolloutBatch:
    terminal_ce: torch.Tensor
    rounds: torch.Tensor
    returns: torch.Tensor
    old_log_joint: torch.Tensor
    policy_version: int

    def __post_init__(self):
        if self.terminal_ce.ndim != 2:
            raise ValueError("terminal_ce must be [B,G]")
        if self.rounds.shape != self.terminal_ce.shape or self.returns.shape != self.terminal_ce.shape or self.old_log_joint.shape != self.terminal_ce.shape:
            raise ValueError("rollout tensors must share [B,G] shape")
        _dtype(self.terminal_ce, torch.float32, "terminal_ce")
        _dtype(self.rounds, torch.int64, "rounds")
        _dtype(self.returns, torch.float32, "returns")
        _dtype(self.old_log_joint, torch.float32, "old_log_joint")
        if self.policy_version < 0:
            raise ValueError("policy_version must be nonnegative")
