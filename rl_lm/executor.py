"""Frozen observation and last-FFN pair executors.

``frozen_observe`` is the single place that turns the frozen hidden state
into actor inputs (spec section 7.1): it computes ``n = ffn_norm(h)`` and
``n0 = ffn_norm(h0)`` in FP32, runs the frozen gate linear in fixed chunks
under the frozen compute dtype (BF16 in production), casts the gate logits
back to FP32, and assembles the feature vector in the fixed order
``[n, n0, z - mean(z), previous-pair multi-hot, 4 scalars]`` together with
the canonical teacher reference ``q``.  The same observation's ``n``/``z``
must be reused by the executor so that actor and executor never see two
different gate precisions.

Four execution backends share the same pair/weight/residual math:

- ``canonical_all_experts_v1`` (training reward): fixed shape, chunk=64,
  zero padding, all 8 experts computed per chunk, then the actual pair is
  gathered and mixed with the original FP32 top-2 softmax weights.  Stopped
  rows keep their hidden; their computed results are discarded but counted
  as physical work.
- ``sparse_merged_v1`` (historical deployment candidate): sparse dispatch
  separately for each slot/expert; its identity and arithmetic are retained.
- ``sparse_merged_v2`` (opt-in deployment candidate): both assignment slots
  are dispatched together per expert, reconstructed by row/slot, then mixed.
  Variable expert batch shapes may change rounding; this is not a claim of
  bitwise parity or a measured speedup.
- ``sparse_active_merged_v3`` (opt-in deployment): online inference compacts
  continuing rows before merged dispatch; STOP rows execute no experts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from rl_lm.actions import PAIR_ACTIONS, make_pair_table
from rl_lm.ledger import RLLedger
from rl_lm.selector import pair_reference

CANONICAL_BACKEND = "canonical_all_experts_v1"
SPARSE_BACKEND = "sparse_merged_v1"
MERGED_BACKEND = "sparse_merged_v2"
ACTIVE_MERGED_BACKEND = "sparse_active_merged_v3"
EXPERTS = 8


@contextmanager
def _frozen_compute(device: torch.device, compute_dtype: str):
    if compute_dtype == "bfloat16":
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            yield
    elif compute_dtype == "float32":
        with torch.autocast(device_type=device.type, enabled=False):
            yield
    else:
        raise ValueError("compute_dtype must be bfloat16 or float32")


@dataclass
class FrozenObservation:
    """Detached frozen view of one decision point; executor reuses n and z."""

    n: torch.Tensor
    n0: torch.Tensor
    z: torch.Tensor
    features: torch.Tensor
    pair_reference: torch.Tensor

    def __post_init__(self) -> None:
        if self.n.ndim != 2:
            raise ValueError("n must be [N,D]")
        count, dim = self.n.shape
        if self.n0.shape != self.n.shape:
            raise ValueError("n0 must match n")
        for name, value in (("n", self.n), ("n0", self.n0), ("z", self.z), ("features", self.features), ("pair_reference", self.pair_reference)):
            if value.dtype != torch.float32:
                raise ValueError(f"{name} must be FP32")
            if value.requires_grad:
                raise ValueError(f"{name} must be detached")
        if tuple(self.z.shape) != (count, EXPERTS):
            raise ValueError("z must be [N,8]")
        if tuple(self.features.shape) != (count, 2 * dim + 2 * EXPERTS + 4):
            raise ValueError("features must be [N,2D+2E+4]")
        if tuple(self.pair_reference.shape) != (count, PAIR_ACTIONS):
            raise ValueError("pair_reference must be [N,28]")
        if not bool((self.pair_reference > 0).all()):
            raise ValueError("pair_reference must be strictly positive")


@torch.no_grad()
def frozen_observe(
    block,
    h: torch.Tensor,
    h0: torch.Tensor,
    rounds: torch.Tensor,
    previous_pair: torch.Tensor,
    *,
    max_calls: int,
    chunk: int = 64,
    compute_dtype: str = "bfloat16",
    pair_table: torch.Tensor | None = None,
) -> FrozenObservation:
    """Build the FrozenObservation for rows ``h`` with initial state ``h0``.

    ``rounds`` is the number of completed pair updates (``r``); states with
    ``r >= max_calls`` are terminal and must never be observed.  Rows with
    ``r == 0`` must carry ``previous_pair == (-1, -1)`` and get an all-zero
    multi-hot; rows with ``r >= 1`` must carry a valid pair.
    """
    if h.ndim != 2 or h0.shape != h.shape:
        raise ValueError("h and h0 must be matching [N,D]")
    if h.dtype != torch.float32 or h0.dtype != torch.float32:
        raise ValueError("h and h0 must be FP32")
    count, dim = h.shape
    if tuple(rounds.shape) != (count,) or rounds.dtype != torch.int64:
        raise ValueError("rounds must be int64 [N]")
    if tuple(previous_pair.shape) != (count, 2) or previous_pair.dtype != torch.int64:
        raise ValueError("previous_pair must be int64 [N,2]")
    if max_calls < 1 or chunk <= 0:
        raise ValueError("bad max_calls or chunk")
    if bool((rounds < 0).any()) or bool((rounds >= max_calls).any()):
        raise ValueError("terminal (r >= Rmax) states must not be observed")
    has_pair = (previous_pair >= 0).all(dim=-1)
    if bool(((rounds == 0) & has_pair).any()) or bool(((rounds >= 1) & ~has_pair).any()):
        raise ValueError("previous_pair must be (-1,-1) iff r == 0")

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
    z = torch.cat(z_pieces) if z_pieces else h.new_zeros((0, EXPERTS))

    centered = z - z.mean(dim=-1, keepdim=True)
    multi_hot = h.new_zeros((count, EXPERTS))
    if bool(has_pair.any()):
        rows = torch.nonzero(has_pair, as_tuple=False).squeeze(-1)
        multi_hot[rows] = multi_hot[rows].scatter(1, previous_pair[rows], 1.0)
    r = rounds.float()
    drift = torch.log1p((h - h0).norm(dim=-1) / (h0.norm(dim=-1) + 1e-8))
    scalars = torch.stack(
        (r / max_calls, (max_calls - r) / max_calls, (rounds == 0).float(), drift),
        dim=-1,
    )
    features = torch.cat((n, n0, centered, multi_hot, scalars), dim=-1)
    q = pair_reference(z, pair_table)
    return FrozenObservation(n=n, n0=n0, z=z, features=features, pair_reference=q)


class _BaseExecutor:
    backend = "abstract"

    def __init__(
        self,
        model,
        *,
        layer: int | None = None,
        chunk: int = 64,
        compute_dtype: str = "bfloat16",
        gamma: float | None = None,
        ledger: RLLedger | None = None,
    ) -> None:
        layer = int(model.config.adaptive_layers[-1]) if layer is None else int(layer)
        if layer != model.config.layers - 1:
            raise ValueError("only the terminal layer is controllable")
        block = model.blocks[layer]
        pool = block.moe.pool
        if pool.experts != EXPERTS:
            raise ValueError("pair-v1 requires exactly 8 experts")
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
        self.table = make_pair_table().to(model.embedding.weight.device)

    def observe(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        rounds: torch.Tensor,
        previous_pair: torch.Tensor,
        *,
        max_calls: int,
    ) -> FrozenObservation:
        return frozen_observe(
            self.block,
            h,
            h0,
            rounds,
            previous_pair,
            max_calls=max_calls,
            chunk=self.chunk,
            compute_dtype=self.compute_dtype,
            pair_table=self.table,
        )

    def _validate_step(self, h: torch.Tensor, observation: FrozenObservation, actions: torch.Tensor) -> None:
        if h.ndim != 2 or h.dtype != torch.float32:
            raise ValueError("h must be FP32 [N,D]")
        count = h.shape[0]
        if observation.n.shape != h.shape or observation.z.shape != (count, EXPERTS):
            raise ValueError("observation does not match h")
        if tuple(actions.shape) != (count,) or actions.dtype != torch.int64:
            raise ValueError("actions must be int64 [N]")
        if bool(((actions < 1) | (actions > PAIR_ACTIONS)).any()):
            raise ValueError("executor only accepts safe pair actions in [1,28]")

    def step_pair(self, h: torch.Tensor, observation: FrozenObservation, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FrozenLastFFNExecutor(_BaseExecutor):
    """Fixed-shape canonical training backend (``canonical_all_experts_v1``)."""

    backend = CANONICAL_BACKEND

    @torch.no_grad()
    def step_pair(self, h: torch.Tensor, observation: FrozenObservation, actions: torch.Tensor) -> torch.Tensor:
        """Proposed ``h + gamma * mix(E_i(n), E_j(n))`` for every row.

        Every chunk computes all 8 experts at the fixed chunk shape; the tail
        chunk is padded with zero hidden and safe action 1.  Results for
        padding rows are discarded but counted as physical expert rows.  The
        caller decides which rows are actually written back (STOP or finished
        rows must ignore the proposal); nothing here depends on other rows'
        actions.
        """
        self._validate_step(h, observation, actions)
        count, dim = h.shape
        proposed = h.clone()
        chunks = 0
        for start in range(0, count, self.chunk):
            valid = min(self.chunk, count - start)
            nc = observation.n[start:start + valid]
            zc = observation.z[start:start + valid]
            hc = h[start:start + valid]
            ac = actions[start:start + valid]
            if valid < self.chunk:
                pad = self.chunk - valid
                nc = torch.cat((nc, nc.new_zeros((pad, dim))))
                zc = torch.cat((zc, zc.new_zeros((pad, EXPERTS))))
                hc = torch.cat((hc, hc.new_zeros((pad, dim))))
                ac = torch.cat((ac, ac.new_ones((pad,))))
            with _frozen_compute(h.device, self.compute_dtype):
                up = torch.matmul(nc.unsqueeze(0), self.pool.w1.transpose(1, 2))
                gate = torch.matmul(nc.unsqueeze(0), self.pool.w3.transpose(1, 2))
                hidden = F.silu(up) * gate
                out_all = torch.matmul(hidden, self.pool.w2.transpose(1, 2))
            out_all = out_all.float().permute(1, 0, 2)  # [chunk, E, D]
            pairs = self.table[ac - 1]  # [chunk, 2]
            rows = torch.arange(self.chunk, device=h.device)
            first = out_all[rows, pairs[:, 0]]
            second = out_all[rows, pairs[:, 1]]
            weights = F.softmax(zc.gather(1, pairs).float(), dim=-1)
            mixed = weights[:, 0:1] * first + weights[:, 1:2] * second
            proposed[start:start + valid] = (hc + self.gamma * mixed)[:valid]
            chunks += 1
        if self.ledger is not None:
            self.ledger.physical_expert_rows += EXPERTS * self.chunk * chunks
        return proposed


class SparseMergedExecutor(_BaseExecutor):
    """Historical ``sparse_merged_v1``: slot-by-slot sparse dispatch.

    The misleading historical class/backend names are kept for compatibility.
    Use SparseMergedV2Executor for genuinely merged expert buckets.
    """

    backend = SPARSE_BACKEND

    @torch.no_grad()
    def step_pair(self, h: torch.Tensor, observation: FrozenObservation, actions: torch.Tensor) -> torch.Tensor:
        """Same pair/weight/residual math as the canonical backend, but each
        expert runs only on the rows that selected it in each separate slot."""
        self._validate_step(h, observation, actions)
        count = h.shape[0]
        pairs = self.table[actions - 1]  # [N,2]
        weights = F.softmax(observation.z.gather(1, pairs).float(), dim=-1)
        mixed = torch.zeros_like(h)
        with _frozen_compute(h.device, self.compute_dtype):
            for slot in (0, 1):
                for expert in range(EXPERTS):
                    rows = torch.nonzero(pairs[:, slot] == expert, as_tuple=False).squeeze(-1)
                    if rows.numel() == 0:
                        continue
                    selected = observation.n[rows]
                    hidden = F.silu(F.linear(selected, self.pool.w1[expert])) * F.linear(selected, self.pool.w3[expert])
                    out = F.linear(hidden, self.pool.w2[expert])
                    mixed[rows] += weights[rows, slot:slot + 1] * out.float()
        if self.ledger is not None:
            self.ledger.physical_expert_rows += 2 * count
        return h + self.gamma * mixed


class SparseMergedV2Executor(_BaseExecutor):
    """Opt-in merged dispatch; leaves canonical and historical v1 unchanged.

    Each expert receives one batch containing assignments from both slots.
    Outputs are written to unique row/slot positions, not atomically added;
    the final FP32 mixture retains the canonical first-plus-second order.
    Python/nonzero and variable-shape GEMMs remain, so performance and CUDA
    numerical drift must be measured before deployment promotion.
    """

    backend = MERGED_BACKEND

    @torch.no_grad()
    def pair_update(self, n: torch.Tensor, pairs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Pre-gamma update from normalized FP32 rows and explicit routing.

        No norm, gate, softmax, residual, or STOP handling occurs here. Pairs
        must be distinct ascending IDs, and weights are the caller's FP32
        pair-softmax weights. Ledger counts actual assignments, including
        safe proposals that a transition later discards for stopped rows.
        """
        if n.ndim != 2 or n.dtype != torch.float32:
            raise ValueError("n must be FP32 [N,D]")
        count, dim = n.shape
        if pairs.shape != (count, 2) or pairs.dtype != torch.int64:
            raise ValueError("pairs must be int64 [N,2]")
        if weights.shape != (count, 2) or weights.dtype != torch.float32:
            raise ValueError("weights must be FP32 [N,2]")
        if pairs.device != n.device or weights.device != n.device:
            raise ValueError("routing and normalized states must share a device")
        if bool(((pairs < 0) | (pairs >= EXPERTS)).any()) or bool((pairs[:, 0] >= pairs[:, 1]).any()):
            raise ValueError("pairs must contain ascending distinct expert IDs in [0,7]")
        slot_outputs = n.new_empty((count * 2, dim))
        assignments = pairs.reshape(-1)
        with _frozen_compute(n.device, self.compute_dtype):
            for expert in range(EXPERTS):
                positions = torch.nonzero(assignments == expert, as_tuple=False).squeeze(-1)
                if positions.numel() == 0:
                    continue
                rows = positions // 2
                selected = n[rows]
                hidden = F.silu(F.linear(selected, self.pool.w1[expert])) * F.linear(selected, self.pool.w3[expert])
                out = F.linear(hidden, self.pool.w2[expert])
                slot_outputs[positions] = out.float()
        slot_outputs = slot_outputs.reshape(count, 2, dim)
        mixed = weights[:, 0:1] * slot_outputs[:, 0] + weights[:, 1:2] * slot_outputs[:, 1]
        if self.ledger is not None:
            self.ledger.physical_expert_rows += 2 * count
        return mixed

    @torch.no_grad()
    def step_pair(self, h: torch.Tensor, observation: FrozenObservation, actions: torch.Tensor) -> torch.Tensor:
        self._validate_step(h, observation, actions)
        pairs = self.table[actions - 1]
        weights = F.softmax(observation.z.gather(1, pairs).float(), dim=-1)
        return h + self.gamma * self.pair_update(observation.n, pairs, weights)


class SparseActiveMergedV3Executor(SparseMergedV2Executor):
    """Deployment-only active-row compaction with the v2 merged pair math.

    Online inference calls ``step_active`` so STOP/finished rows never reach
    an expert. Observation/actor shapes remain unchanged. Variable expert
    batch shapes can change rounding; v1/v2 retain their reference identities.
    """

    backend = ACTIVE_MERGED_BACKEND

    @torch.no_grad()
    def step_active(self, h: torch.Tensor, observation: FrozenObservation,
                    actions: torch.Tensor, take_pair: torch.Tensor) -> torch.Tensor:
        """Apply safe pair actions only at ``take_pair``; preserve other rows.

        Actions use the existing safe [1,28] contract, even for skipped rows.
        Physical accounting is performed by pair_update on compact rows only.
        """
        self._validate_step(h, observation, actions)
        if take_pair.shape != (h.shape[0],) or take_pair.dtype != torch.bool or take_pair.device != h.device:
            raise ValueError("take_pair must be bool [N] on the hidden device")
        rows = torch.nonzero(take_pair, as_tuple=False).squeeze(-1)
        proposed = h.clone()
        if rows.numel() == 0:
            return proposed
        pairs = self.table[actions.index_select(0, rows) - 1]
        z = observation.z.index_select(0, rows)
        weights = F.softmax(z.gather(1, pairs).float(), dim=-1)
        update = self.pair_update(observation.n.index_select(0, rows), pairs, weights)
        proposed.index_copy_(0, rows, h.index_select(0, rows) + self.gamma * update)
        return proposed


def make_executor(model, backend: str, **kwargs) -> _BaseExecutor:
    if backend == ACTIVE_MERGED_BACKEND:
        return SparseActiveMergedV3Executor(model, **kwargs)
    if backend == CANONICAL_BACKEND:
        return FrozenLastFFNExecutor(model, **kwargs)
    if backend == SPARSE_BACKEND:
        return SparseMergedExecutor(model, **kwargs)
    if backend == MERGED_BACKEND:
        return SparseMergedV2Executor(model, **kwargs)
    raise ValueError(f"unknown backend: {backend!r}")


def freeze_backbone(model) -> None:
    """Freeze every LM parameter and switch to eval mode (idempotent)."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
