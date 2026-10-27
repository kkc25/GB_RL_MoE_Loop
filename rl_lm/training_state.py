"""Policy-independent transaction snapshots and deterministic epoch cursors.

Deliberately not a trainer superclass: policy evaluation, gradients, KL decisions,
selection, and accounting remain with each policy. Rollback excludes gradients,
RNG and resource ledgers: retry reuses the same gradients and consumed work must
not disappear. Checkpoint RNG capture/restore remains a separate concern.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from rl_lm.checkpoint import move_optimizer_state


def epoch_permutation(*, domain: str, actor_seed: int, epoch: int, count: int) -> np.ndarray:
    """Preserve each policy's existing domain-separated Philox permutation."""
    key = json.dumps([domain, actor_seed, epoch], ensure_ascii=True, separators=(',', ':'))
    seed = int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:16], 'little')
    return np.random.Generator(np.random.Philox(seed)).permutation(count)


def advance_epoch_batch(permutation, *, epoch: int, cursor: int, batch: int, batches_per_epoch: int):
    """Take one full batch, dropping the remainder without padding/repetition.

    Returns indices, next epoch, next cursor and cached permutation. Returning
    None at the boundary makes the caller regenerate with its own RNG domain.
    """
    indices = [int(i) for i in permutation[cursor:cursor + batch]]
    cursor += batch
    if cursor >= batches_per_epoch * batch:
        return indices, epoch + 1, 0, None
    return indices, epoch, cursor, permutation


def capture_update_snapshot(actor, optimizer, *, lr_scale: float, lr_backoffs: int) -> dict:
    """Deep copies, including Adam moments; never aliases the live state."""
    return {
        'actor': copy.deepcopy(actor.state_dict()),
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'lr_scale': lr_scale,
        'lr_backoffs': lr_backoffs,
    }


def restore_update_snapshot(actor, optimizer, snapshot, *, device):
    """Restore reusable snapshot, preserve live gradients, and relocate moments."""
    actor.load_state_dict(copy.deepcopy(snapshot['actor']))
    optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
    move_optimizer_state(optimizer, device)
    return snapshot['lr_scale'], snapshot['lr_backoffs']
