"""Online full-input forward with an RL policy on the terminal FFN.

``forward_with_last_ffn_policy`` replays the frozen prefix up to (but not
including) the terminal FFN via ``cache_last_ffn_input``, runs the joint
29-action greedy policy on **every** position (padding positions included,
keeping the cost accounting comparable across candidates), executes the
chosen pairs with the requested backend, and finishes with the final norm
plus tied head.  The original model is never run end-to-end and then
overwritten — the terminal FFN is computed exactly once, through this entry.

R1 (max_calls=1) is the validated path; the R2 interface (max_calls=2, STOP
decisions) is implemented here for task card F but no R2 experiment is
extended in this stage.  Deployment greedy uses the joint argmax with
lowest-action-id tie breaking (never the legacy hazard-threshold rule).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from eval.counterfactual import cache_last_ffn_input
from rl_lm.actions import JOINT_ACTIONS, STOP_ACTION
from rl_lm.environment import action_for_safe_execution, make_safe_observation_state
from rl_lm.executor import ACTIVE_MERGED_BACKEND, _frozen_compute, make_executor
from rl_lm.ledger import RLLedger
from rl_lm.readout import production_logits, production_token_ce


@dataclass
class OnlineOutput:
    logits: torch.Tensor  # [B,S,V], head output dtype (BF16 in production)
    rounds: torch.Tensor  # [B,S] int64 executed pair rounds per position
    actions: torch.Tensor | None  # [B,S,Rmax] int64, -1 where no decision
    decision_mask: torch.Tensor | None  # [B,S,Rmax] bool
    final_hidden: torch.Tensor | None  # [B,S,D] FP32, debug/trace only
    backend: str
    compute_dtype: str
    max_calls: int


@dataclass
class OnlineCEOutput:
    """Explicit evaluation output: CPU FP32 losses, no full vocabulary logits."""

    token_ce: torch.Tensor  # [B,S], ignore_index=-100 positions are zero
    rounds: torch.Tensor
    actions: torch.Tensor | None
    decision_mask: torch.Tensor | None
    final_hidden: torch.Tensor | None
    backend: str
    compute_dtype: str
    max_calls: int


class CanonicalTeacherPolicy:
    """Joint log-probs equal to log q with STOP forbidden.

    Greedy over this joint distribution selects the canonical teacher pair of
    the current gate every round, i.e. the ``canonical_pair_r1`` /
    ``canonical_pair_r2`` baselines with explicit lexicographic tie breaking.
    """

    feature_dim = None

    def __call__(self, features: torch.Tensor, pair_reference: torch.Tensor, stop_allowed: torch.Tensor) -> torch.Tensor:
        q = pair_reference.detach().float()
        if not bool((q > 0).all()):
            raise ValueError("pair reference must be strictly positive")
        stop = torch.full((q.shape[0], 1), -torch.inf, dtype=torch.float32, device=q.device)
        return torch.cat((stop, q.log()), dim=-1)


def greedy_lowest_id(log_probs: torch.Tensor) -> torch.Tensor:
    """Joint argmax with explicit lowest-action-id tie breaking."""
    if log_probs.ndim != 2 or log_probs.shape[1] != JOINT_ACTIONS:
        raise ValueError("expected joint log-probs [N,29]")
    best = log_probs.max(dim=-1, keepdim=True).values
    ties = log_probs == best
    indices = torch.arange(JOINT_ACTIONS, device=log_probs.device)
    return torch.where(ties, indices, JOINT_ACTIONS).min(dim=-1).values


@torch.no_grad()
def forward_with_last_ffn_policy(
    model,
    actor,
    input_ids: torch.Tensor,
    segment_ids: torch.Tensor | None,
    *,
    max_calls: int,
    backend: str,
    return_trace: bool = False,
    compute_dtype: str = "bfloat16",
    chunk: int = 64,
    head_chunk: int = 64,
    ledger: RLLedger | None = None,
    output_mode: str = "logits",
    labels: torch.Tensor | None = None,
) -> OnlineOutput | OnlineCEOutput:
    """Execute every input position, optionally streaming CPU token losses.

    The default ``output_mode="logits"`` preserves the public OnlineOutput
    contract. ``output_mode="token_ce"`` requires labels [B,S] and returns
    OnlineCEOutput instead, without constructing [B,S,V]. Labels do not
    change prefix execution, policy decisions, batching, or physical rows.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [batch, sequence]")
    if segment_ids is not None and tuple(segment_ids.shape) != tuple(input_ids.shape):
        raise ValueError("segment_ids must match input_ids")
    if max_calls < 1:
        raise ValueError("max_calls must be >= 1")
    if output_mode not in ("logits", "token_ce"):
        raise ValueError("output_mode must be logits or token_ce")
    if output_mode == "token_ce":
        if labels is None or labels.shape != input_ids.shape:
            raise ValueError("token_ce requires labels matching input_ids")
        if labels.dtype != torch.int64 or labels.device != input_ids.device:
            raise ValueError("labels must be int64 on the input device")
    elif labels is not None:
        raise ValueError("labels are only accepted in token_ce output mode")
    batch, length = input_ids.shape
    device = input_ids.device
    with _frozen_compute(device, compute_dtype):
        prefix = cache_last_ffn_input(model, input_ids, segment_ids)
    dim = prefix.shape[-1]
    count = batch * length
    h0 = prefix.float().reshape(count, dim).contiguous()
    h = h0.clone()
    rounds = torch.zeros(count, dtype=torch.int64, device=device)
    done = torch.zeros(count, dtype=torch.bool, device=device)
    previous_pair = torch.full((count, 2), -1, dtype=torch.int64, device=device)
    actions_trace = torch.full((count, max_calls), -1, dtype=torch.int64, device=device) if return_trace else None
    decision_mask = torch.zeros(count, max_calls, dtype=torch.bool, device=device) if return_trace else None

    executor = make_executor(model, backend, chunk=chunk, compute_dtype=compute_dtype, ledger=ledger)
    if executor.layer != model.config.layers - 1:
        raise ValueError("the controlled layer must be terminal")

    for slot in range(max_calls):
        active = ~done
        if not bool(active.any()):
            break
        safe_h, safe_h0, safe_rounds, safe_previous = make_safe_observation_state(h, h0, rounds, previous_pair, active)
        observation = executor.observe(safe_h, safe_h0, safe_rounds, safe_previous, max_calls=max_calls)
        stop_allowed = safe_rounds >= 1
        log_probs = actor(observation.features, observation.pair_reference, stop_allowed)
        if tuple(log_probs.shape) != (count, JOINT_ACTIONS):
            raise ValueError("policy must return joint log-probs [N,29]")
        if bool(torch.isnan(log_probs).any()) or bool(torch.isposinf(log_probs).any()):
            raise ValueError("policy produced invalid log-probs")
        actions = greedy_lowest_id(log_probs)
        if bool(((actions == STOP_ACTION) & ~stop_allowed & active).any()):
            raise RuntimeError("illegal STOP selected at r=0; fail fast instead of remapping")
        take_pair = active & (actions > STOP_ACTION)
        voluntary_stop = active & (actions == STOP_ACTION)
        safe_actions = action_for_safe_execution(actions, active)
        if backend == ACTIVE_MERGED_BACKEND:
            # Preserve real stopped hidden states, not their safe observation
            # placeholders. Only continuing rows are gathered into expert GEMMs.
            h = executor.step_active(h, observation, safe_actions, take_pair)
        else:
            proposed = executor.step_pair(safe_h, observation, safe_actions)
            h = torch.where(take_pair[:, None], proposed, h)
        pairs = executor.table[(actions.clamp_min(1) - 1)]
        previous_pair = torch.where(take_pair[:, None], pairs, previous_pair)
        rounds = rounds + take_pair.to(torch.int64)
        done = done | voluntary_stop | (rounds >= max_calls)
        if return_trace:
            actions_trace[:, slot] = torch.where(active, actions, torch.full_like(actions, -1))
            decision_mask[:, slot] = active
        if ledger is not None:
            ledger.actor_decision_rows += int(active.sum())
            ledger.actor_padding_rows += int(count - int(active.sum()))
            ledger.logical_pair_rounds += int(take_pair.sum())
            ledger.logical_expert_rows += 2 * int(take_pair.sum())

    if not bool(done.all()):
        raise RuntimeError("online forward ended with unfinished positions")
    if ledger is not None:
        ledger.prefix_input_tokens += int(input_ids.numel())
    metadata = dict(
        rounds=rounds.reshape(batch, length),
        actions=actions_trace.reshape(batch, length, max_calls) if return_trace else None,
        decision_mask=decision_mask.reshape(batch, length, max_calls) if return_trace else None,
        final_hidden=h.reshape(batch, length, dim) if return_trace else None,
        backend=backend,
        compute_dtype=compute_dtype,
        max_calls=max_calls,
    )
    if output_mode == "token_ce":
        ce = production_token_ce(model, h, labels.reshape(-1), chunk=head_chunk,
                                 compute_dtype=compute_dtype, ledger=ledger)
        return OnlineCEOutput(token_ce=ce.reshape(batch, length), **metadata)
    logits = production_logits(model, h, chunk=head_chunk, compute_dtype=compute_dtype, ledger=ledger)
    return OnlineOutput(logits=logits.reshape(batch, length, model.embedding.weight.shape[0]), **metadata)
