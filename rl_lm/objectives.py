"""RLOO objective: reward shape, leave-one-out advantage, PG loss.

Spec section 9:

- reward is ``R = ref_ce - terminal_ce - lambda * (rounds - 1)`` (nats; the
  first pair round is the mandatory base computation, lambda is paid per
  *extra* pair round, never per expert call);
- the baseline is leave-one-out strictly along the G dimension of the same
  context: ``A[b,g] = R[b,g] - mean_{g' != g} R[b,g']``; no flattening of
  B x G into one group, no group-std whitening, no division by trajectory
  length, no batch whitening, no clipping;
- the policy-gradient loss is reward-form with an explicit negative sign:
  ``-(1 / (B*G)) * sum stopgrad(A) * L`` where ``L`` is the sum of the
  actually sampled stochastic-decision log-probs (a voluntary STOP counts,
  the forced termination at Rmax does not).

``rloo_advantage``/``policy_gradient_loss`` follow appendix A, with one
extension: the advantage computation preserves the input floating dtype
instead of unconditionally casting to FP32, so the FP64 exact-gradient
contract test (T13) can exercise the same production function; FP32 inputs
produce bitwise the same results as the appendix reference.
"""

from __future__ import annotations

import torch


def compute_returns(
    ref_ce: torch.Tensor,
    terminal_ce: torch.Tensor,
    rounds: torch.Tensor,
    price_lambda: float,
) -> torch.Tensor:
    """Spec 9.1 reward: ref_ce - ce - lambda * (rounds - 1), elementwise."""
    if ref_ce.shape != terminal_ce.shape or ref_ce.shape != rounds.shape:
        raise ValueError("ref_ce, terminal_ce and rounds must share shape")
    if price_lambda < 0:
        raise ValueError("lambda must be nonnegative (nat per extra pair round)")
    if not ref_ce.is_floating_point() or not terminal_ce.is_floating_point():
        raise ValueError("CE values must be floating point")
    returns = ref_ce - terminal_ce - float(price_lambda) * (rounds - 1).to(ref_ce.dtype)
    if not bool(torch.isfinite(returns).all()):
        raise ValueError("nonfinite returns")
    return returns.detach()


def rloo_advantage(returns: torch.Tensor) -> torch.Tensor:
    """Detached leave-one-out advantages along G; [B,G] -> [B,G], G >= 2.

    A context whose G returns are all identical gets exactly zero advantages
    (bitwise), which makes the batch's PG gradient exactly zero; this is a
    legal event and must be logged, not forced away.
    """
    if returns.ndim != 2 or returns.shape[1] < 2:
        raise ValueError("returns must be [B,G], G >= 2")
    r = returns.detach()
    if not r.is_floating_point():
        r = r.float()
    if not bool(torch.isfinite(r).all()):
        raise ValueError("nonfinite reward")
    loo = (r.sum(dim=1, keepdim=True) - r) / (r.shape[1] - 1)
    same = (r == r[:, :1]).all(dim=1, keepdim=True)
    return torch.where(same, torch.zeros_like(r), r - loo).detach()


def policy_gradient_loss(log_joint: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    """Reward-form PG loss: ``-(advantage * log_joint).mean()`` over [B,G].

    ``log_joint`` must carry the actor graph; ``advantage`` must be detached.
    High-reward trajectories get their probability increased, low-reward ones
    decreased; never mix this with a cost-form positive-sign loss.
    """
    if log_joint.shape != advantage.shape:
        raise ValueError("log_joint and advantage must share shape")
    if not log_joint.requires_grad:
        raise ValueError("log_joint must carry gradients")
    if advantage.requires_grad:
        raise ValueError("advantage must be detached")
    return -(advantage * log_joint).mean()


def exact_path_objective(path_log_probs: torch.Tensor, path_costs: torch.Tensor) -> torch.Tensor:
    """Exact enumerated-tree objective ``J(theta) = sum_i p_i(theta) * c_i``.

    Reference target for the toy tree MDP of spec appendix B: ``path_log_probs``
    is the joint log-prob of every terminal path (with gradient),
    ``path_costs`` is detached.  Its autograd gradient is the exact policy
    gradient that the RLOO expectation must reproduce (T13).
    """
    if path_log_probs.ndim != 1 or path_costs.shape != path_log_probs.shape:
        raise ValueError("expected 1-D path log-probs and matching costs")
    if not path_log_probs.requires_grad:
        raise ValueError("path_log_probs must carry gradients")
    if path_costs.requires_grad:
        raise ValueError("path_costs must be detached")
    return (path_log_probs.exp() * path_costs.detach()).sum()
