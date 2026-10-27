"""Production-precision and diagnostic readouts for the frozen backbone.

``terminal_ce`` is the production reward readout (spec section 8.3): the tied
head runs under BF16 autocast and cross-entropy is computed on FP32 logits,
in fixed chunks with zero padding.  It is intentionally distinct from the
explicit-FP32 ``eval.counterfactual.readout_ce``.  ``diagnostic_fp32_ce``
disables autocast entirely and exists only to separate hidden-state
differences from head-precision differences; its results must be reported
under a separate name, never mixed with production rewards.

``production_logits`` materializes full-vocabulary logits for every position
(the default online forward); ``production_token_ce`` instead streams the
same head chunks through bounded evaluation CE windows. Both preserve the
BF16-head contract; evaluation CE windows remain distinct from reward CE.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from rl_lm.ledger import RLLedger


def _validate_rows(h: torch.Tensor, targets: torch.Tensor, chunk: int) -> None:
    if h.ndim != 2 or targets.shape != (h.shape[0],):
        raise ValueError("expected h[N,D] and targets[N]")
    if h.dtype != torch.float32 or targets.dtype != torch.int64:
        raise ValueError("expected FP32 h and int64 targets")
    if h.device != targets.device or chunk <= 0:
        raise ValueError("bad device or chunk")


def _validate_targets_in_vocab(targets: torch.Tensor, vocab: int) -> None:
    if targets.numel() and not bool(((targets >= 0) & (targets < vocab)).all()):
        raise ValueError("targets outside the vocabulary; padding must use a legal id")


@torch.no_grad()
def terminal_ce(model, h: torch.Tensor, targets: torch.Tensor, *, chunk: int = 64, ledger: RLLedger | None = None) -> torch.Tensor:
    """Production readout: BF16 autocast tied head, FP32 cross-entropy.

    All targets must be valid vocabulary ids; the tail chunk is padded with
    zero hidden and target id 0, and padded rows are discarded after the
    readout but still counted in ``physical_head_rows``.
    """
    _validate_rows(h, targets, chunk)
    _validate_targets_in_vocab(targets, model.embedding.weight.shape[0])
    if h.shape[0] == 0:
        return h.new_empty((0,))
    pieces = []
    chunks = 0
    for start in range(0, h.shape[0], chunk):
        valid = min(chunk, h.shape[0] - start)
        hc = h[start:start + valid]
        yc = targets[start:start + valid]
        if valid < chunk:
            hc = torch.cat((hc, h.new_zeros((chunk - valid, h.shape[1]))))
            yc = torch.cat((yc, targets.new_zeros((chunk - valid,))))
        with torch.autocast(device_type=h.device.type, dtype=torch.bfloat16):
            logits = F.linear(model.final_norm(hc), model.embedding.weight)
        with torch.autocast(device_type=h.device.type, enabled=False):
            losses = F.cross_entropy(logits.float(), yc, reduction="none")
        pieces.append(losses[:valid])
        chunks += 1
    if ledger is not None:
        ledger.physical_head_rows += chunks * chunk
    return torch.cat(pieces)


@torch.no_grad()
def diagnostic_fp32_ce(model, h: torch.Tensor, targets: torch.Tensor, *, chunk: int = 64, ledger: RLLedger | None = None) -> torch.Tensor:
    """Explicit-FP32 head readout; diagnostics only, never the reward."""
    _validate_rows(h, targets, chunk)
    _validate_targets_in_vocab(targets, model.embedding.weight.shape[0])
    if h.shape[0] == 0:
        return h.new_empty((0,))
    pieces = []
    chunks = 0
    for start in range(0, h.shape[0], chunk):
        valid = min(chunk, h.shape[0] - start)
        hc = h[start:start + valid]
        yc = targets[start:start + valid]
        if valid < chunk:
            hc = torch.cat((hc, h.new_zeros((chunk - valid, h.shape[1]))))
            yc = torch.cat((yc, targets.new_zeros((chunk - valid,))))
        with torch.autocast(device_type=h.device.type, enabled=False):
            logits = F.linear(model.final_norm(hc).float(), model.embedding.weight.float())
            losses = F.cross_entropy(logits, yc, reduction="none")
        pieces.append(losses[:valid])
        chunks += 1
    if ledger is not None:
        ledger.physical_head_rows += chunks * chunk
    return torch.cat(pieces)


@torch.no_grad()
def production_logits(
    model,
    h: torch.Tensor,
    *,
    chunk: int = 64,
    compute_dtype: str = "bfloat16",
    ledger: RLLedger | None = None,
) -> torch.Tensor:
    """Full-vocabulary tied-head logits for every row of ``h`` [N,D] -> [N,V].

    Under ``compute_dtype="bfloat16"`` the head linear runs in BF16 autocast,
    matching the production head contract; the returned logits keep the
    autocast output dtype.  The tail chunk is zero padded and discarded.
    """
    if h.ndim != 2 or h.dtype != torch.float32 or chunk <= 0:
        raise ValueError("expected FP32 h[N,D] and positive chunk")
    if compute_dtype not in ("bfloat16", "float32"):
        raise ValueError("compute_dtype must be bfloat16 or float32")
    if h.shape[0] == 0:
        return h.new_empty((0, model.embedding.weight.shape[0]))
    output = None
    chunks = 0
    for start in range(0, h.shape[0], chunk):
        valid = min(chunk, h.shape[0] - start)
        hc = h[start:start + valid]
        if valid < chunk:
            hc = torch.cat((hc, h.new_zeros((chunk - valid, h.shape[1]))))
        if compute_dtype == "bfloat16":
            with torch.autocast(device_type=h.device.type, dtype=torch.bfloat16):
                logits = F.linear(model.final_norm(hc), model.embedding.weight)
        else:
            with torch.autocast(device_type=h.device.type, enabled=False):
                logits = F.linear(model.final_norm(hc), model.embedding.weight)
        if output is None:
            # Allocate in the actual autocast output dtype, not h.dtype.
            output = logits.new_empty((h.shape[0], logits.shape[1]))
        output[start:start + valid].copy_(logits[:valid])
        # Do not retain the previous chunk while computing the next one.
        del logits
        chunks += 1
    if ledger is not None:
        ledger.physical_head_rows += chunks * chunk
    return output


@torch.no_grad()
def production_token_ce(
    model,
    h: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk: int = 64,
    compute_dtype: str = "bfloat16",
    ce_window: int = 8192,
    ledger: RLLedger | None = None,
) -> torch.Tensor:
    """Bounded head -> CE, returning CPU FP32 per-token losses [N].

    Head GEMMs, autocast and tail padding match ``production_logits``.
    A reusable head-dtype buffer holds at most ``ce_window`` rows; FP32 CE
    uses the same position windows as the full-logits evaluation path (8192
    by default), including ignored labels. No valid-token compaction occurs.
    Head chunks may cross CE boundaries without changing their GEMM shape.
    This is evaluation-only, not the chunk-64 ``terminal_ce`` reward API.
    """
    _validate_rows(h, targets, chunk)
    if compute_dtype not in ("bfloat16", "float32"):
        raise ValueError("compute_dtype must be bfloat16 or float32")
    if ce_window <= 0 or chunk > ce_window:
        raise ValueError("ce_window must be positive and at least head chunk")
    vocab = model.embedding.weight.shape[0]
    if targets.numel() and not bool(((targets == -100) | ((targets >= 0) & (targets < vocab))).all()):
        raise ValueError("targets must be vocabulary ids or ignore_index=-100")
    count = h.shape[0]
    result = torch.empty((count,), dtype=torch.float32, device="cpu")
    if count == 0:
        return result
    buffer = None
    filled = 0
    window_start = 0
    for start in range(0, count, chunk):
        valid = min(chunk, count - start)
        hc = h[start:start + valid]
        if valid < chunk:
            hc = torch.cat((hc, h.new_zeros((chunk - valid, h.shape[1]))))
        with torch.autocast(device_type=h.device.type, dtype=torch.bfloat16,
                            enabled=compute_dtype == "bfloat16"):
            logits = F.linear(model.final_norm(hc), model.embedding.weight)
        if ledger is not None:
            ledger.physical_head_rows += chunk
        if buffer is None:
            buffer = logits.new_empty((min(count, ce_window), vocab))
        consumed = 0
        while consumed < valid:
            take = min(valid - consumed, buffer.shape[0] - filled)
            buffer[filled:filled + take].copy_(logits[consumed:consumed + take])
            filled += take
            consumed += take
            if filled == buffer.shape[0] or start + consumed == count:
                with torch.autocast(device_type=h.device.type, enabled=False):
                    losses = F.cross_entropy(
                        buffer[:filled].float(), targets[window_start:window_start + filled],
                        reduction="none", ignore_index=-100,
                    )
                result[window_start:window_start + filled].copy_(losses.cpu())
                del losses
                window_start += filled
                filled = 0
        del logits
    return result
