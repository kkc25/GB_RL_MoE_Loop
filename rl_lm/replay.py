"""Replay of decision records through the current actor (spec section 11).

Replay recomputes only the actor: the detached features / pair_reference /
stop_allowed stored in ``DecisionRecords`` are fed to the current actor, the
sampled action's log-prob is gathered per decision row, and rows are summed
into per-trajectory joint log-probs by their global ``flat_id = b*G + g``.
No expert re-execution, no reward recomputation, no hidden rebuild.

Before the optimizer step, replay on the same actor weights must reproduce
the rollout's ``old_log_joint`` (spec 11.1, FP32 atol=rtol=1e-5); after the
step a difference is expected.  Microbatching (spec 11.3) keeps the global
flat ids: a filtered record subset is replayed with the full ``(B, G)``
shape, positions outside the microbatch stay zero, and the loss is divided
by the global ``B_effective * G`` -- never by the microbatch size.
"""

from __future__ import annotations

import torch

from rl_lm.actions import JOINT_ACTIONS
from rl_lm.types import DecisionRecords


def replay_log_joint(actor, records: DecisionRecords, batch: int, groups: int) -> torch.Tensor:
    """Current-actor per-trajectory joint log-probs, ``[B, G]``.

    Rows of ``records`` keep their global ``flat_episode_ids``; episodes
    without rows in ``records`` contribute exactly zero.
    """
    if int(batch) < 1 or int(groups) < 2:
        raise ValueError("batch must be >= 1 and groups >= 2")
    total = int(batch) * int(groups)
    if records.flat_episode_ids.numel() and not bool((records.flat_episode_ids < total).all()):
        raise ValueError("flat_episode_ids exceed B*G - 1")
    if bool((records.actions < 0).any()) or bool((records.actions >= JOINT_ACTIONS).any()):
        raise ValueError("actions must be legal joint actions in [0, 28]")
    log_probs = actor(records.features, records.pair_reference, records.stop_allowed)
    if tuple(log_probs.shape) != (records.features.shape[0], JOINT_ACTIONS):
        raise ValueError("actor must return joint log-probs [K,29]")
    chosen = log_probs.gather(1, records.actions[:, None]).squeeze(1)
    joint = chosen.new_zeros(total)
    joint = joint.index_add(0, records.flat_episode_ids, chosen)
    return joint.reshape(int(batch), int(groups))


def context_record_mask(records: DecisionRecords, context_ids: torch.Tensor, groups: int) -> torch.Tensor:
    """Boolean row mask selecting every decision of the given global contexts.

    A context contributes all of its G trajectories; G is never split across
    microbatches.  ``context_ids`` are 0-based context (``b``) indices.
    """
    context_ids = torch.as_tensor(context_ids, dtype=torch.int64, device=records.flat_episode_ids.device)
    if context_ids.numel() and (bool((context_ids < 0).any())):
        raise ValueError("context_ids must be nonnegative")
    contexts = records.flat_episode_ids // int(groups)
    return (contexts[:, None] == context_ids[None, :]).any(dim=1)


def filter_records(records: DecisionRecords, mask: torch.Tensor) -> DecisionRecords:
    """Subset a DecisionRecords by a boolean row mask, keeping global ids."""
    if mask.dtype != torch.bool or mask.shape != (records.features.shape[0],):
        raise ValueError("mask must be bool [K]")
    return DecisionRecords(
        features=records.features[mask],
        pair_reference=records.pair_reference[mask],
        stop_allowed=records.stop_allowed[mask],
        actions=records.actions[mask],
        flat_episode_ids=records.flat_episode_ids[mask],
        old_log_probs_all=records.old_log_probs_all[mask],
    )


def iter_context_microbatches(batch: int, microbatch: int):
    """Yield 0-based context index tensors of size ``microbatch`` (last one shorter)."""
    if int(microbatch) < 1 or int(batch) < 1:
        raise ValueError("batch and microbatch must be positive")
    for start in range(0, int(batch), int(microbatch)):
        yield torch.arange(start, min(start + int(microbatch), int(batch)), dtype=torch.int64)


def microbatch_pg_loss(
    actor,
    records: DecisionRecords,
    advantage: torch.Tensor,
    batch: int,
    groups: int,
    context_ids: torch.Tensor,
) -> torch.Tensor:
    """Spec 11.3 microbatch loss: ``-sum_{micro,g} A*L / (B_effective * G)``.

    The microbatch's records are replayed at the full ``[B, G]`` shape via
    their global flat ids, so the multiplication with the full advantage
    tensor zeroes out everything outside the microbatch and the reduction
    stays the global one.  Call ``backward`` per microbatch and clip/step
    once at the end; never average per-microbatch means.
    """
    if advantage.shape != (int(batch), int(groups)):
        raise ValueError("advantage must be the full [B_effective, G] tensor")
    if advantage.requires_grad:
        raise ValueError("advantage must be detached")
    mask = context_record_mask(records, context_ids, groups)
    sub = filter_records(records, mask)
    joint = replay_log_joint(actor, sub, batch, groups)
    return -(advantage * joint).sum() / (int(batch) * int(groups))
