"""S1 single-expert branch: executor, observation, ledger and online forward.

Spec section 18 (``RL-LM-Single-v1.0``).  The controlled function is

    h' = h + gamma * E_e(RMSNorm(h))

with an explicit expert id ``e`` and amplitude fixed at 1.  The gate softmax
probability is *never* multiplied into the update, and the legacy
``MoE.forward(x, indices)`` amplitude path is never called.  Action 0 is
STOP; actions 1..8 select experts 0..7.

Only the terminal FFN layer is overridden; the first 11 layers keep the
backbone's original top-2 function (the backbone config keeps ``top_k=2``).

Two execution backends share the same math:

- ``canonical_all_experts_single_v1`` (training reward): fixed shape, all 8
  experts computed per chunk, then the selected expert is gathered.  The
  physical/logical expert-row ratio is exactly 8 and is recorded in the
  ledger.
- ``sparse_single_v1`` (deployment candidate): each expert runs only on the
  rows that selected it.

``single_ffn_update`` is the differentiable sparse path used by supervised
adaptation; it keeps gradients flowing into the selected expert's pool
slices.  ``single_online_forward`` is the greedy joint-9 closed-loop forward
used for selection evaluation and online parity tests.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from rl_lm.executor import _frozen_compute

STOP_ACTION = 0
SINGLE_EXPERTS = 8
SINGLE_ACTIONS = 8
JOINT_SINGLE_ACTIONS = 9
SINGLE_CANONICAL_BACKEND = "canonical_all_experts_single_v1"
SINGLE_SPARSE_BACKEND = "sparse_single_v1"
SINGLE_FEATURE_SCHEMA = "h_h0_gate_previoussingle_scalars_v1"
SINGLE_SPEC_VERSION = "RL-LM-Single-v1.0"


def expert_for_action(action: int) -> int:
    """Action ``a`` in [1,8] selects expert ``a - 1``; action 0 is STOP."""
    if not 1 <= int(action) <= SINGLE_ACTIONS:
        raise ValueError("single expert action must be in [1,8]")
    return int(action) - 1


def single_argmax_expert(gate_logits: torch.Tensor) -> torch.Tensor:
    """Fixed selection: largest gate logit, ties broken to the lowest id."""
    if gate_logits.ndim != 2 or gate_logits.shape[1] != SINGLE_EXPERTS:
        raise ValueError("gate_logits must be [N,8]")
    z = gate_logits.float()
    best = z.max(dim=-1, keepdim=True).values
    is_best = z == best
    ids = torch.arange(SINGLE_EXPERTS, device=z.device)
    return torch.where(is_best, ids, SINGLE_EXPERTS).min(dim=-1).values


def single_reference(gate_logits: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
    """Prior ``0.9 * one_hot(argmax gate) + epsilon / 8`` over the 8 experts."""
    if float(epsilon) != 0.1:
        raise ValueError("single-v1 reference epsilon is fixed at 0.1")
    chosen = single_argmax_expert(gate_logits)
    reference = torch.full(
        (gate_logits.shape[0], SINGLE_EXPERTS), epsilon / SINGLE_EXPERTS,
        dtype=torch.float32, device=gate_logits.device,
    )
    reference.scatter_(1, chosen[:, None], 1.0 - epsilon + epsilon / SINGLE_EXPERTS)
    return reference


def single_ffn_update(pool, n: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    """Differentiable sparse single-expert FFN update ``E_e(n)``, amplitude 1.

    ``n`` is the FP32 RMSNorm output ``[N, D]``; ``expert_ids`` are explicit
    expert ids in [0,7].  Rows are dispatched per expert with ``F.linear`` on
    the pool slices, so gradients flow into exactly the selected experts'
    ``w1/w2/w3`` slices.  This function never calls ``MoE.forward`` and never
    multiplies a gate probability.
    """
    if n.ndim != 2 or expert_ids.shape != (n.shape[0],):
        raise ValueError("expected n[N,D] and expert_ids[N]")
    if expert_ids.dtype != torch.int64:
        raise ValueError("expert_ids must be int64")
    if bool(((expert_ids < 0) | (expert_ids >= pool.experts)).any()):
        raise ValueError("expert id out of range")
    out = torch.zeros_like(n)
    for expert in range(pool.experts):
        rows = torch.nonzero(expert_ids == expert, as_tuple=False).squeeze(-1)
        if rows.numel() == 0:
            continue
        selected = n.index_select(0, rows)
        hidden = F.silu(F.linear(selected, pool.w1[expert])) * F.linear(selected, pool.w3[expert])
        out.index_copy_(0, rows, F.linear(hidden, pool.w2[expert]).to(out.dtype))
    return out


@dataclass
class SingleObservation:
    """Detached frozen view of one single-expert decision point."""

    n: torch.Tensor
    n0: torch.Tensor
    z: torch.Tensor
    features: torch.Tensor
    reference: torch.Tensor

    def __post_init__(self) -> None:
        if self.n.ndim != 2:
            raise ValueError("n must be [N,D]")
        count, dim = self.n.shape
        if self.n0.shape != self.n.shape:
            raise ValueError("n0 must match n")
        for name, value in (("n", self.n), ("n0", self.n0), ("z", self.z), ("features", self.features), ("reference", self.reference)):
            if value.dtype != torch.float32:
                raise ValueError(f"{name} must be FP32")
            if value.requires_grad:
                raise ValueError(f"{name} must be detached")
        if tuple(self.z.shape) != (count, SINGLE_EXPERTS):
            raise ValueError("z must be [N,8]")
        if tuple(self.features.shape) != (count, 2 * dim + 2 * SINGLE_EXPERTS + 4):
            raise ValueError("features must be [N,2D+2E+4]")
        if tuple(self.reference.shape) != (count, SINGLE_ACTIONS):
            raise ValueError("reference must be [N,8]")
        if not bool((self.reference > 0).all()):
            raise ValueError("reference must be strictly positive")


@torch.no_grad()
def frozen_single_observe(
    block,
    h: torch.Tensor,
    h0: torch.Tensor,
    rounds: torch.Tensor,
    previous_expert: torch.Tensor,
    *,
    max_calls: int,
    chunk: int = 64,
    compute_dtype: str = "bfloat16",
) -> SingleObservation:
    """Build the SingleObservation for rows ``h`` with initial state ``h0``.

    Mirrors ``frozen_observe`` from the pair branch, but the previous-action
    feature is a single-expert one-hot (8 dims; all-zero at ``r == 0``) and
    the canonical reference is the single-expert prior ``q``.
    """
    if h.ndim != 2 or h0.shape != h.shape:
        raise ValueError("h and h0 must be matching [N,D]")
    if h.dtype != torch.float32 or h0.dtype != torch.float32:
        raise ValueError("h and h0 must be FP32")
    count, dim = h.shape
    if tuple(rounds.shape) != (count,) or rounds.dtype != torch.int64:
        raise ValueError("rounds must be int64 [N]")
    if tuple(previous_expert.shape) != (count,) or previous_expert.dtype != torch.int64:
        raise ValueError("previous_expert must be int64 [N]")
    if max_calls < 1 or chunk <= 0:
        raise ValueError("bad max_calls or chunk")
    if bool((rounds < 0).any()) or bool((rounds >= max_calls).any()):
        raise ValueError("terminal (r >= Rmax) states must not be observed")
    has_previous = previous_expert >= 0
    if bool(((rounds == 0) & has_previous).any()) or bool(((rounds >= 1) & ~has_previous).any()):
        raise ValueError("previous_expert must be -1 iff r == 0")
    if bool((previous_expert >= SINGLE_EXPERTS).any()):
        raise ValueError("previous_expert out of range")

    n = block.ffn_norm(h).float()
    n0 = block.ffn_norm(h0).float()
    z_pieces = []
    for start in range(0, count, chunk):
        valid = min(chunk, count - start)
        nc = n[start:start + valid]
        if valid < chunk:
            nc = torch.cat((nc, n.new_zeros((chunk - valid, dim))))
        with _frozen_compute(h.device, compute_dtype):
            zc = block.moe.gate(nc)
        z_pieces.append(zc.float()[:valid])
    z = torch.cat(z_pieces) if z_pieces else h.new_zeros((0, SINGLE_EXPERTS))

    centered = z - z.mean(dim=-1, keepdim=True)
    one_hot = h.new_zeros((count, SINGLE_EXPERTS))
    if bool(has_previous.any()):
        rows = torch.nonzero(has_previous, as_tuple=False).squeeze(-1)
        one_hot[rows] = one_hot[rows].scatter(1, previous_expert[rows, None], 1.0)
    r = rounds.float()
    drift = torch.log1p((h - h0).norm(dim=-1) / (h0.norm(dim=-1) + 1e-8))
    scalars = torch.stack(
        (r / max_calls, (max_calls - r) / max_calls, (rounds == 0).float(), drift),
        dim=-1,
    )
    features = torch.cat((n, n0, centered, one_hot, scalars), dim=-1)
    q = single_reference(z)
    return SingleObservation(n=n, n0=n0, z=z, features=features, reference=q)


@dataclass
class SingleLedger:
    """Logical vs physical ledger for the single-expert environment.

    ``logical_single_rounds`` counts committed single-expert FFN updates;
    ``logical_expert_rows`` equals it (exactly one expert call per round).
    ``physical_expert_rows`` counts everything the backend really computed:
    the canonical training backend computes all 8 experts per row, so its
    physical/logical ratio is exactly 8 and must be accounted as such.
    ``cost.lambda`` is denominated per *extra single-expert round*.
    """

    prefix_input_tokens: int = 0
    unique_train_contexts: int = 0
    context_visits: int = 0
    rollout_episodes: int = 0
    logical_single_rounds: int = 0
    logical_expert_rows: int = 0
    physical_expert_rows: int = 0
    physical_head_rows: int = 0
    actor_decision_rows: int = 0
    actor_padding_rows: int = 0
    successful_updates: int = 0
    optimizer_attempts: int = 0
    gpu_hours: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0

    def merge_peak(self, device: torch.device) -> None:
        if device.type == "cuda":
            self.peak_allocated_bytes = max(self.peak_allocated_bytes, torch.cuda.max_memory_allocated(device))
            self.peak_reserved_bytes = max(self.peak_reserved_bytes, torch.cuda.max_memory_reserved(device))

    def record_wall_time(self, wall_time_seconds: float, *, gpus: int = 1) -> None:
        seconds = float(wall_time_seconds)
        if not seconds >= 0.0:
            raise ValueError("wall_time_seconds must be nonnegative")
        if int(gpus) < 1:
            raise ValueError("gpus must be a positive integer")
        self.gpu_hours += seconds * int(gpus) / 3600.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SingleLedger":
        known = {field.name for field in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown ledger fields: {sorted(unknown)}")
        missing = known - set(data)
        if missing:
            raise ValueError(f"missing ledger fields: {sorted(missing)}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "SingleLedger":
        return cls.from_dict(json.loads(Path(path).read_text()))


class _BaseSingleExecutor:
    backend = "abstract"

    def __init__(
        self,
        model,
        *,
        layer: int | None = None,
        chunk: int = 64,
        compute_dtype: str = "bfloat16",
        gamma: float | None = None,
        ledger: SingleLedger | None = None,
    ) -> None:
        layer = int(model.config.adaptive_layers[-1]) if layer is None else int(layer)
        if layer != model.config.layers - 1:
            raise ValueError("only the terminal layer is controllable")
        block = model.blocks[layer]
        pool = block.moe.pool
        if pool.experts != SINGLE_EXPERTS:
            raise ValueError("single-v1 requires exactly 8 experts")
        if chunk <= 0:
            raise ValueError("chunk must be positive")
        if compute_dtype not in ("bfloat16", "float32"):
            raise ValueError("compute_dtype must be bfloat16 or float32")
        self.model = model
        self.block = block
        self.pool = pool
        self.layer = layer
        self.chunk = int(chunk)
        self.compute_dtype = compute_dtype
        self.gamma = float(model.gammas[layer]) if gamma is None else float(gamma)
        self.ledger = ledger

    def observe(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        rounds: torch.Tensor,
        previous_expert: torch.Tensor,
        *,
        max_calls: int,
    ) -> SingleObservation:
        return frozen_single_observe(
            self.block,
            h,
            h0,
            rounds,
            previous_expert,
            max_calls=max_calls,
            chunk=self.chunk,
            compute_dtype=self.compute_dtype,
        )

    def _validate_step(self, h: torch.Tensor, observation: SingleObservation, actions: torch.Tensor) -> None:
        if h.ndim != 2 or h.dtype != torch.float32:
            raise ValueError("h must be FP32 [N,D]")
        count = h.shape[0]
        if observation.n.shape != h.shape or observation.z.shape != (count, SINGLE_EXPERTS):
            raise ValueError("observation does not match h")
        if tuple(actions.shape) != (count,) or actions.dtype != torch.int64:
            raise ValueError("actions must be int64 [N]")
        if bool(((actions < 1) | (actions > SINGLE_ACTIONS)).any()):
            raise ValueError("executor only accepts explicit single actions in [1,8]")

    def step_single(self, h: torch.Tensor, observation: SingleObservation, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CanonicalSingleExecutor(_BaseSingleExecutor):
    """Fixed-shape training backend (``canonical_all_experts_single_v1``).

    Every chunk computes all 8 experts at the fixed chunk shape; the tail
    chunk is padded with zero hidden and safe action 1.  The selected expert
    is gathered with amplitude 1; discarded rows still count as physical
    expert rows (physical/logical ratio exactly 8).
    """

    backend = SINGLE_CANONICAL_BACKEND

    @torch.no_grad()
    def step_single(self, h: torch.Tensor, observation: SingleObservation, actions: torch.Tensor) -> torch.Tensor:
        self._validate_step(h, observation, actions)
        count, dim = h.shape
        proposed = h.clone()
        chunks = 0
        for start in range(0, count, self.chunk):
            valid = min(self.chunk, count - start)
            nc = observation.n[start:start + valid]
            hc = h[start:start + valid]
            ac = actions[start:start + valid]
            if valid < self.chunk:
                pad = self.chunk - valid
                nc = torch.cat((nc, nc.new_zeros((pad, dim))))
                hc = torch.cat((hc, hc.new_zeros((pad, dim))))
                ac = torch.cat((ac, ac.new_ones((pad,))))
            with _frozen_compute(h.device, self.compute_dtype):
                up = torch.matmul(nc.unsqueeze(0), self.pool.w1.transpose(1, 2))
                gate = torch.matmul(nc.unsqueeze(0), self.pool.w3.transpose(1, 2))
                hidden = F.silu(up) * gate
                out_all = torch.matmul(hidden, self.pool.w2.transpose(1, 2))
            out_all = out_all.float().permute(1, 0, 2)  # [chunk, E, D]
            rows = torch.arange(self.chunk, device=h.device)
            chosen = out_all[rows, ac - 1]
            proposed[start:start + valid] = (hc + self.gamma * chosen)[:valid]
            chunks += 1
        if self.ledger is not None:
            self.ledger.physical_expert_rows += SINGLE_EXPERTS * self.chunk * chunks
        return proposed


class SparseSingleExecutor(_BaseSingleExecutor):
    """Deployment candidate (``sparse_single_v1``): per-expert dispatch."""

    backend = SINGLE_SPARSE_BACKEND

    @torch.no_grad()
    def step_single(self, h: torch.Tensor, observation: SingleObservation, actions: torch.Tensor) -> torch.Tensor:
        self._validate_step(h, observation, actions)
        count = h.shape[0]
        update = torch.zeros_like(h)
        with _frozen_compute(h.device, self.compute_dtype):
            for expert in range(SINGLE_EXPERTS):
                rows = torch.nonzero(actions == expert + 1, as_tuple=False).squeeze(-1)
                if rows.numel() == 0:
                    continue
                selected = observation.n[rows]
                hidden = F.silu(F.linear(selected, self.pool.w1[expert])) * F.linear(selected, self.pool.w3[expert])
                out = F.linear(hidden, self.pool.w2[expert])
                update[rows] = out.float()
        if self.ledger is not None:
            self.ledger.physical_expert_rows += count
        return h + self.gamma * update


def make_single_executor(model, backend: str, **kwargs) -> _BaseSingleExecutor:
    if backend == SINGLE_CANONICAL_BACKEND:
        return CanonicalSingleExecutor(model, **kwargs)
    if backend == SINGLE_SPARSE_BACKEND:
        return SparseSingleExecutor(model, **kwargs)
    raise ValueError(f"unknown single backend: {backend!r}")


def single_greedy_lowest_id(log_probs: torch.Tensor) -> torch.Tensor:
    """Joint-9 argmax with explicit lowest-action-id tie breaking."""
    if log_probs.ndim != 2 or log_probs.shape[1] != JOINT_SINGLE_ACTIONS:
        raise ValueError("expected joint log-probs [N,9]")
    best = log_probs.max(dim=-1, keepdim=True).values
    ties = log_probs == best
    indices = torch.arange(JOINT_SINGLE_ACTIONS, device=log_probs.device)
    return torch.where(ties, indices, JOINT_SINGLE_ACTIONS).min(dim=-1).values


class ArgmaxSingleTeacherPolicy:
    """Joint-9 log-probs equal to log q with STOP forbidden.

    Greedy over this joint distribution executes the fixed selection rule
    (argmax gate, lowest-id tie) every round: the adapted/frozen single
    expert R1 reference.
    """

    feature_dim = None

    def __call__(self, features: torch.Tensor, reference: torch.Tensor, stop_allowed: torch.Tensor) -> torch.Tensor:
        q = reference.detach().float()
        if not bool((q > 0).all()):
            raise ValueError("single reference must be strictly positive")
        stop = torch.full((q.shape[0], 1), -torch.inf, dtype=torch.float32, device=q.device)
        return torch.cat((stop, q.log()), dim=-1)


def _single_safe_state(h, h0, rounds, previous_expert, active):
    safe_h = torch.where(active[:, None], h, torch.zeros_like(h))
    safe_h0 = torch.where(active[:, None], h0, torch.zeros_like(h0))
    safe_rounds = torch.where(active, rounds, torch.zeros_like(rounds))
    safe_previous = torch.where(active, previous_expert, torch.full_like(previous_expert, -1))
    return safe_h, safe_h0, safe_rounds, safe_previous


def _single_safe_actions(actions, active):
    safe = torch.where(actions > STOP_ACTION, actions, torch.ones_like(actions))
    return torch.where(active, safe, torch.ones_like(actions))


@torch.no_grad()
def single_online_forward(
    model,
    policy,
    input_ids: torch.Tensor,
    segment_ids: torch.Tensor | None,
    *,
    max_calls: int,
    backend: str,
    compute_dtype: str = "bfloat16",
    chunk: int = 64,
    head_chunk: int = 64,
    ledger: SingleLedger | None = None,
    output_mode: str = "logits",
    labels: torch.Tensor | None = None,
):
    """Greedy joint-9 closed-loop forward with a single-expert terminal FFN.

    Replays the frozen prefix once, then loops the joint-9 greedy policy on
    every position, executing the chosen expert with amplitude 1 through the
    requested backend, and finishes with the production head readout.  The
    first 11 layers always run their original backbone function.

    Default returns (logits [B,S,V], rounds [B,S]) unchanged. Explicit
    output_mode="token_ce" requires labels [B,S] and instead returns
    (CPU FP32 per-token CE [B,S], rounds [B,S]), with bounded 8192-row
    evaluation CE windows. Labels never alter the closed-loop execution.
    """
    from eval.counterfactual import cache_last_ffn_input
    from rl_lm.readout import production_logits, production_token_ce

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
    previous_expert = torch.full((count,), -1, dtype=torch.int64, device=device)

    executor = make_single_executor(model, backend, chunk=chunk, compute_dtype=compute_dtype, ledger=ledger)
    if executor.layer != model.config.layers - 1:
        raise ValueError("the controlled layer must be terminal")

    for _slot in range(max_calls):
        active = ~done
        if not bool(active.any()):
            break
        safe_h, safe_h0, safe_rounds, safe_previous = _single_safe_state(h, h0, rounds, previous_expert, active)
        observation = executor.observe(safe_h, safe_h0, safe_rounds, safe_previous, max_calls=max_calls)
        stop_allowed = safe_rounds >= 1
        log_probs = policy(observation.features, observation.reference, stop_allowed)
        if tuple(log_probs.shape) != (count, JOINT_SINGLE_ACTIONS):
            raise ValueError("policy must return joint log-probs [N,9]")
        if bool(torch.isnan(log_probs).any()) or bool(torch.isposinf(log_probs).any()):
            raise ValueError("policy produced invalid log-probs")
        actions = single_greedy_lowest_id(log_probs)
        if bool(((actions == STOP_ACTION) & ~stop_allowed & active).any()):
            raise RuntimeError("illegal STOP selected at r=0; fail fast instead of remapping")
        take = active & (actions > STOP_ACTION)
        voluntary_stop = active & (actions == STOP_ACTION)
        proposed = executor.step_single(safe_h, observation, _single_safe_actions(actions, active))
        h = torch.where(take[:, None], proposed, h)
        previous_expert = torch.where(take, actions - 1, previous_expert)
        rounds = rounds + take.to(torch.int64)
        done = done | voluntary_stop | (rounds >= max_calls)
        if ledger is not None:
            ledger.actor_decision_rows += int(active.sum())
            ledger.actor_padding_rows += int(count - int(active.sum()))
            ledger.logical_single_rounds += int(take.sum())
            ledger.logical_expert_rows += int(take.sum())

    if not bool(done.all()):
        raise RuntimeError("single online forward ended with unfinished positions")
    if ledger is not None:
        ledger.prefix_input_tokens += int(input_ids.numel())
    if output_mode == "token_ce":
        ce = production_token_ce(model, h, labels.reshape(-1), chunk=head_chunk,
                                 compute_dtype=compute_dtype, ledger=ledger)
        return ce.reshape(batch, length), rounds.reshape(batch, length)
    logits = production_logits(model, h, chunk=head_chunk, compute_dtype=compute_dtype, ledger=ledger)
    return logits.reshape(batch, length, model.embedding.weight.shape[0]), rounds.reshape(batch, length)


class CudaWallTimer:
    """CUDA-event interval timer that accumulates into a ledger's gpu_hours."""

    def __init__(self, device: torch.device, ledger: SingleLedger | None = None, gpus: int = 1) -> None:
        self.device = device
        self.ledger = ledger
        self.gpus = gpus
        self.elapsed_seconds = 0.0

    def __enter__(self) -> "CudaWallTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed_seconds = time.perf_counter() - self._start
        if self.ledger is not None:
            self.ledger.record_wall_time(self.elapsed_seconds, gpus=self.gpus)
