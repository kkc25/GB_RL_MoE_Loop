"""MDP state machine for the frozen terminal-FFN pair environment.

Implements the rollout loop of spec section 10.2: ``r`` counts executed pair
FFN updates, ``decision_count`` counts sampled actions (they differ), action
0 is STOP and is illegal at ``r == 0``, reaching ``Rmax`` terminates without
an extra STOP log-prob, and a terminated episode's hidden/rounds/history are
never touched again.  The reward backend runs at a fixed shape: every
decision slot feeds the whole flat batch through ``frozen_observe``, the
actor and the executor with safe filler rows, records only active rows, and
writes expert results back only where a real pair action was taken.

Sampling uses the inverse-CDF reference from spec appendix A on float64
uniforms in ``[0, 1)``; the uniform tables themselves are produced by the
RNG module (task card C) and passed in, so this state machine has no hidden
randomness.
"""

from __future__ import annotations

import torch

from rl_lm.actions import JOINT_ACTIONS, PAIR_ACTIONS, STOP_ACTION
from rl_lm.executor import FrozenLastFFNExecutor
from rl_lm.ledger import RLLedger
from rl_lm.readout import terminal_ce
from rl_lm.types import ContextBatch, DecisionRecords, RolloutBatch


@torch.no_grad()
def inverse_cdf_sample(log_probs: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    """[N,A] log-probs and [N] float64 u in [0,1) -> [N] actions."""
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


def make_safe_observation_state(
    h: torch.Tensor,
    h0: torch.Tensor,
    rounds: torch.Tensor,
    previous_pair: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed-shape view: inactive rows become the all-zero r=0 filler state.

    Active rows keep their true values.  The real environment state is never
    modified here; filler outputs are computed but neither recorded nor
    written back.
    """
    safe_h = torch.where(active[:, None], h, torch.zeros_like(h))
    safe_h0 = torch.where(active[:, None], h0, torch.zeros_like(h0))
    safe_rounds = torch.where(active, rounds, torch.zeros_like(rounds))
    safe_previous = torch.where(active[:, None], previous_pair, torch.full_like(previous_pair, -1))
    return safe_h, safe_h0, safe_rounds, safe_previous


def action_for_safe_execution(actions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Replace STOP and inactive-row actions with the fixed safe action 1."""
    safe = torch.where(actions > STOP_ACTION, actions, torch.ones_like(actions))
    return torch.where(active, safe, torch.ones_like(actions))


class LastFFNEnvironment:
    """Frozen terminal-FFN environment for R1 (Rmax=1) and R2 (Rmax=2)."""

    def __init__(
        self,
        model,
        *,
        max_calls: int,
        executor: FrozenLastFFNExecutor,
        head_chunk: int = 64,
        price_lambda: float = 0.0,
        ledger: RLLedger | None = None,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if head_chunk <= 0:
            raise ValueError("head_chunk must be positive")
        if price_lambda < 0:
            raise ValueError("lambda must be nonnegative (nat per extra pair round)")
        if executor.model is not model:
            raise ValueError("executor must wrap the same model")
        self.model = model
        self.max_calls = int(max_calls)
        self.executor = executor
        self.head_chunk = int(head_chunk)
        self.price_lambda = float(price_lambda)
        self.ledger = ledger

    @torch.no_grad()
    def rollout(
        self,
        contexts: ContextBatch,
        actor,
        uniforms: torch.Tensor,
        *,
        policy_version: int = 0,
    ) -> tuple[RolloutBatch, DecisionRecords]:
        """Run G independent rollouts per context for Rmax decision slots.

        ``uniforms`` is float64 ``[B, G, Rmax]`` in ``[0, 1)``; episode flat
        order is ``flat_id = b * G + g``.  Returns the per-trajectory
        ``RolloutBatch`` and the concatenated valid ``DecisionRecords``.
        """
        if uniforms.ndim != 3:
            raise ValueError("uniforms must be [B,G,Rmax]")
        batch, groups, slots = uniforms.shape
        if contexts.h0.shape[0] != batch or slots != self.max_calls:
            raise ValueError("uniforms shape does not match contexts or Rmax")
        if uniforms.dtype != torch.float64:
            raise ValueError("uniforms must be float64")
        if not bool(((uniforms >= 0) & (uniforms < 1)).all()):
            raise ValueError("uniforms must lie in [0,1)")
        device = contexts.h0.device
        count = batch * groups
        u = uniforms.reshape(count, slots)
        h0 = contexts.h0.repeat_interleave(groups, dim=0)
        targets = contexts.targets.repeat_interleave(groups)
        ref_ce = contexts.ref_ce.repeat_interleave(groups)
        h = h0.clone()
        rounds = torch.zeros(count, dtype=torch.int64, device=device)
        done = torch.zeros(count, dtype=torch.bool, device=device)
        previous_pair = torch.full((count, 2), -1, dtype=torch.int64, device=device)

        rec_features, rec_reference, rec_stop_allowed = [], [], []
        rec_actions, rec_flat_ids, rec_log_probs = [], [], []

        for slot in range(slots):
            active = ~done
            if not bool(active.any()):
                break
            safe_h, safe_h0, safe_rounds, safe_previous = make_safe_observation_state(h, h0, rounds, previous_pair, active)
            observation = self.executor.observe(safe_h, safe_h0, safe_rounds, safe_previous, max_calls=self.max_calls)
            stop_allowed = safe_rounds >= 1
            log_probs = actor(observation.features, observation.pair_reference, stop_allowed)
            if tuple(log_probs.shape) != (count, JOINT_ACTIONS):
                raise ValueError("actor must return joint log-probs [N,29]")
            if bool(torch.isnan(log_probs).any()) or bool(torch.isposinf(log_probs).any()):
                raise ValueError("actor produced invalid log-probs")
            feature_dim = getattr(actor, "feature_dim", None)
            if feature_dim is not None and feature_dim != observation.features.shape[1]:
                raise ValueError("actor feature dim does not match the observation schema")
            actions = inverse_cdf_sample(log_probs, u[:, slot])
            if bool(((actions == STOP_ACTION) & ~stop_allowed & active).any()):
                raise RuntimeError("illegal STOP sampled at r=0; fail fast instead of remapping")

            rows = torch.nonzero(active, as_tuple=False).squeeze(-1)
            rec_features.append(observation.features[rows].clone())
            rec_reference.append(observation.pair_reference[rows].clone())
            rec_stop_allowed.append(stop_allowed[rows].clone())
            rec_actions.append(actions[rows].clone())
            rec_flat_ids.append(rows.clone())
            rec_log_probs.append(log_probs[rows].clone())

            take_pair = active & (actions > STOP_ACTION)
            voluntary_stop = active & (actions == STOP_ACTION)
            proposed = self.executor.step_pair(safe_h, observation, action_for_safe_execution(actions, active))
            h = torch.where(take_pair[:, None], proposed, h)
            pairs = self.executor.table[(actions.clamp_min(1) - 1)]
            previous_pair = torch.where(take_pair[:, None], pairs, previous_pair)
            rounds = rounds + take_pair.to(torch.int64)
            done = done | voluntary_stop | (rounds >= self.max_calls)

            if self.ledger is not None:
                self.ledger.actor_decision_rows += int(rows.numel())
                self.ledger.actor_padding_rows += int(count - rows.numel())
                self.ledger.logical_pair_rounds += int(take_pair.sum())
                self.ledger.logical_expert_rows += 2 * int(take_pair.sum())

        if not bool(done.all()):
            raise RuntimeError("environment ended with unfinished episodes")
        ce = terminal_ce(self.model, h, targets, chunk=self.head_chunk, ledger=self.ledger)
        returns = ref_ce - ce - self.price_lambda * (rounds - 1).float()

        actions_all = torch.cat(rec_actions)
        log_probs_all = torch.cat(rec_log_probs)
        flat_ids_all = torch.cat(rec_flat_ids)
        chosen = log_probs_all.gather(1, actions_all[:, None]).squeeze(1)
        joint = chosen.new_zeros(count)
        joint = joint.index_add(0, flat_ids_all, chosen)

        records = DecisionRecords(
            features=torch.cat(rec_features),
            pair_reference=torch.cat(rec_reference),
            stop_allowed=torch.cat(rec_stop_allowed),
            actions=actions_all,
            flat_episode_ids=flat_ids_all,
            old_log_probs_all=log_probs_all,
        )
        rollout = RolloutBatch(
            terminal_ce=ce.reshape(batch, groups),
            rounds=rounds.reshape(batch, groups),
            returns=returns.reshape(batch, groups),
            old_log_joint=joint.reshape(batch, groups),
            policy_version=policy_version,
        )
        if self.ledger is not None:
            self.ledger.rollout_episodes += count
        return rollout, records
