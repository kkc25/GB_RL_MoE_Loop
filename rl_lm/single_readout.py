"""Differentiable readouts and the supervised adaptation loss (spec 18.2).

The pair branch's ``readout.terminal_ce`` is decorated with ``@no_grad`` and
cannot be used for adaptation.  ``differentiable_logits`` here runs the same
tied-head shape (final RMSNorm then embedding matrix) with gradients enabled:
the head weights stay frozen (``requires_grad=False``) but gradients flow
through them into the hidden state and from there into the terminal expert
pool slices.  Frozen weights do not mean a cut compute graph.

All adaptation readouts are explicit FP32: the KL term uses FP32
``softmax``/``log_softmax``, summed over the vocabulary and averaged over
valid tokens, at fixed temperature 1.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def differentiable_logits(model, h: torch.Tensor, *, chunk: int = 256) -> torch.Tensor:
    """FP32 tied-head logits ``[N, V]`` with gradients flowing into ``h``.

    The frozen head (``final_norm`` + tied ``embedding.weight``) is evaluated
    in FP32 with autocast disabled.  Padding tail rows are discarded.
    """
    if h.ndim != 2 or h.dtype != torch.float32:
        raise ValueError("expected FP32 h[N,D]")
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    if h.shape[0] == 0:
        return h.new_zeros((0, model.embedding.weight.shape[0]))
    pieces = []
    for start in range(0, h.shape[0], chunk):
        hc = h[start:start + chunk]
        with torch.autocast(device_type=h.device.type, enabled=False):
            logits = F.linear(model.final_norm(hc), model.embedding.weight)
        pieces.append(logits.float())
    return torch.cat(pieces)


def adapt_ce_loss(student_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-token CE averaged over valid tokens (all rows here are valid)."""
    if student_logits.ndim != 2 or targets.shape != (student_logits.shape[0],):
        raise ValueError("expected logits [N,V] and targets [N]")
    if targets.dtype != torch.int64:
        raise ValueError("targets must be int64")
    vocab = student_logits.shape[1]
    if bool(((targets < 0) | (targets >= vocab)).any()):
        raise ValueError("targets outside the vocabulary")
    return F.cross_entropy(student_logits.float(), targets, reduction="mean")


def adapt_kl_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """KL(p_teacher || p_student), FP32, temperature 1, per-token then mean.

    ``teacher_logits`` must already be detached; the function refuses a
    tensor that still carries a graph so teacher gradients are impossible.
    """
    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim != 2:
        raise ValueError("teacher/student logits shape mismatch")
    if teacher_logits.requires_grad:
        raise ValueError("teacher logits must be detached")
    with torch.autocast(device_type=student_logits.device.type, enabled=False):
        log_t = F.log_softmax(teacher_logits.float(), dim=-1)
        log_s = F.log_softmax(student_logits.float(), dim=-1)
        kl = (log_t.exp() * (log_t - log_s)).sum(dim=-1)
    return kl.mean()


def relative_update_error(u_student: torch.Tensor, u_teacher: torch.Tensor) -> torch.Tensor:
    """Per-token relative squared error of the pre-gamma FFN updates.

    ``||u_S - u_T||_2^2 / (||u_T||_2^2 + 1e-8)`` is computed per token along
    the hidden dimension and then averaged over tokens — never as one global
    batch norm ratio.  ``u_teacher`` must be detached.
    """
    if u_student.shape != u_teacher.shape or u_student.ndim != 2:
        raise ValueError("u_student/u_teacher must share [N,D] shape")
    if u_teacher.requires_grad:
        raise ValueError("teacher update must be detached")
    numerator = (u_student.float() - u_teacher.float()).square().sum(dim=-1)
    denominator = u_teacher.float().square().sum(dim=-1) + 1e-8
    return (numerator / denominator).mean()
