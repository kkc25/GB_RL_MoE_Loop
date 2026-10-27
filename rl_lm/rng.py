"""Deterministic per-episode random streams and the reference sampler.

Spec section 10.1: every random stream is a function of
``(actor_seed, successful_update, episode_id, group_id, decision_slot)`` only.
It must not depend on the row position inside the active batch, on how many
neighbours stopped early, or on the microbatch split.  The first version
serializes ``["rl-lm-rng-v1", seed, update, episode_id, g]`` with
``json.dumps(..., ensure_ascii=True, separators=(",", ":"))``, hashes the
UTF-8 bytes with SHA256, takes the first 16 digest bytes as a little-endian
integer and uses it as the seed of a NumPy Philox generator that produces a
fixed number of float64 uniforms per decision slot.  Python's built-in
``hash()`` is never used, so streams are stable across processes.

Sampling from the joint 29-class log-probs is by inverse CDF on float64
uniforms in ``[0, 1)`` (spec appendix A): probabilities are converted to
float64, normalized, cumulated with the last column set to exactly 1, and the
action is ``count(cdf <= u)``.  Uniforms must never be rounded to FP32 first.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

RNG_DOMAIN = "rl-lm-rng-v1"


def stream_seed(actor_seed: int, update: int, episode_id: str, group_id: int) -> int:
    """128-bit Philox seed for one (episode, group) stream of one update."""
    if int(update) < 0:
        raise ValueError("update must be nonnegative")
    key = json.dumps(
        [RNG_DOMAIN, int(actor_seed), int(update), str(episode_id), int(group_id)],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "little")


def stream_key(actor_seed: int, update: int, episode_id: str, group_id: int) -> str:
    """The exact serialized key; exposed for audits and cross-process checks."""
    return json.dumps(
        [RNG_DOMAIN, int(actor_seed), int(update), str(episode_id), int(group_id)],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def make_uniform_table(
    actor_seed: int,
    update: int,
    episode_ids,
    groups: int,
    slots: int,
) -> torch.Tensor:
    """Float64 uniforms ``[B, G, T]`` in ``[0, 1)`` for one update.

    ``episode_ids[b]`` alone determines row ``b``; regenerating the table with
    a reordered, subset, or extended episode list leaves every surviving
    episode's numbers unchanged.  Each ``g`` uses an independent Philox
    stream; the decision slot is the position inside that stream, so an early
    STOP in a neighbouring episode cannot shift this episode's numbers.
    """
    episode_ids = tuple(str(e) for e in episode_ids)
    if len(episode_ids) == 0:
        raise ValueError("episode_ids must be nonempty")
    if int(groups) < 1 or int(slots) < 1:
        raise ValueError("groups and slots must be positive")
    table = np.empty((len(episode_ids), int(groups), int(slots)), dtype=np.float64)
    for b, episode_id in enumerate(episode_ids):
        for g in range(int(groups)):
            generator = np.random.Generator(np.random.Philox(stream_seed(actor_seed, update, episode_id, g)))
            table[b, g] = generator.random(int(slots), dtype=np.float64)
    return torch.from_numpy(table)


@torch.no_grad()
def sample_from_log_probs(log_probs: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF categorical sample: [N,A] log-probs, [N] float64 u -> [N].

    Rows whose illegal actions carry ``-inf`` log-prob get zero probability
    mass; every row needs at least one legal action.
    """
    if log_probs.ndim != 2 or uniforms.shape != (log_probs.shape[0],):
        raise ValueError("bad sampler shapes")
    if uniforms.dtype != torch.float64:
        raise ValueError("uniforms must remain float64")
    if not bool(torch.isfinite(uniforms).all()):
        raise ValueError("nonfinite uniforms")
    if not bool(((uniforms >= 0) & (uniforms < 1)).all()):
        raise ValueError("uniforms must lie in [0,1)")
    if bool(torch.isnan(log_probs).any()) or bool(torch.isposinf(log_probs).any()):
        raise ValueError("invalid log probabilities")
    probabilities = log_probs.detach().double().exp()
    normalizer = probabilities.sum(-1, keepdim=True)
    if not bool((normalizer > 0).all()):
        raise ValueError("each row needs at least one legal action")
    probabilities = probabilities / normalizer
    cdf = probabilities.cumsum(-1)
    cdf[:, -1] = 1.0
    return (cdf <= uniforms[:, None]).sum(-1).long()


# Continuity with the appendix-A name used inside environment.rollout.
inverse_cdf_sample = sample_from_log_probs
