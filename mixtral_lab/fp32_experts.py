"""Opt-in FP32 R2 arithmetic on frozen, already-quantized expert weights.

No master-weight bank: each linear materializes one FP32 projection for its
forward or backward and releases it. This is a candidate numerical protocol,
not a claim of batch invariance. Global precision flags require serial use.
"""
from contextlib import contextmanager
import time

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def check_deadline(deadline):
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError('FP32 R2 deadline exhausted between operations')


@contextmanager
def precision_scope(device_type):
    old = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(old)


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, deadline):
        if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
            raise ValueError('FP32 linear requires aligned 2D tensors')
        if x.device != weight.device or not x.is_floating_point() or not weight.is_floating_point():
            raise ValueError('FP32 linear requires floating tensors on one device')
        check_deadline(deadline)
        ctx.deadline = deadline
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        # Frozen expert W: save its original BF16 view, never W.float() nor X.
        # Small trainable correction W: X is needed for its weight gradient.
        ctx.save_for_backward(weight, x if ctx.needs_input_grad[1] else x.new_empty(0))
        with precision_scope(x.device.type):
            result = F.linear(x.float(), weight.float())
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        check_deadline(ctx.deadline)
        weight, x = ctx.saved_tensors
        gx = gw = None
        with precision_scope(grad.device.type):
            if ctx.needs_input_grad[0]:
                gx = torch.matmul(grad.float(), weight.float()).to(ctx.input_dtype)
            if ctx.needs_input_grad[1]:
                gw = torch.matmul(grad.float().transpose(0, 1), x.float()).to(ctx.weight_dtype)
        return gx, gw, None


def precise_linear(x, weight, *, deadline=None):
    return _Linear.apply(x, weight, deadline)


def frozen_linear(x, weight, *, deadline=None):
    if weight.requires_grad:
        raise ValueError('expert projection must remain frozen')
    return precise_linear(x, weight, deadline=deadline)


def correction_fp32(correction, x, *, deadline=None):
    # Explicit custom backward also preserves precision for both adapter factors.
    hidden = precise_linear(x.detach().float(), correction.down, deadline=deadline)
    return precise_linear(hidden, correction.up, deadline=deadline)


def experts_fp32(experts, h, pairs, weights, *, deadline=None, work=None):
    """Same eager Top-2 formula/order, without BF16 intermediate rounding."""
    if h.ndim != 2 or h.dtype != torch.float32:
        raise ValueError('FP32 expert inputs must be flat float32')
    if pairs.shape != (len(h), 2) or weights.shape != pairs.shape or pairs.dtype != torch.long:
        raise ValueError('aligned Top-2 routes required')
    if pairs.device != h.device or weights.device != h.device:
        raise ValueError('expert inputs/routes must share a device')
    if (bool((pairs[:, 0] == pairs[:, 1]).any()) or
            not bool(((pairs >= 0) & (pairs < experts.num_experts)).all())):
        raise ValueError('distinct in-range Top-2 experts required')
    from transformers.models.mixtral.modeling_mixtral import MixtralExperts
    if (type(experts) is not MixtralExperts or
            getattr(experts.config, '_experts_implementation', None) != 'eager' or
            experts.config.hidden_act != 'silu'):
        raise ValueError('only packed eager Mixtral SiLU experts supported')
    if any(p.requires_grad for p in experts.parameters()):
        raise ValueError('all expert parameters must remain frozen')
    if experts.gate_up_proj.device != h.device or experts.down_proj.device != h.device:
        raise ValueError('resident expert projections must share input device')
    if work is not None:
        work['fp32_r2_forward_attempts'] = work.get('fp32_r2_forward_attempts', 0) + 1
    check_deadline(deadline)
    with precision_scope(h.device.type):
        mask = F.one_hot(pairs.detach(), num_classes=experts.num_experts).permute(2, 1, 0)
        out = torch.zeros_like(h)
        for e in range(experts.num_experts):
            slots, tokens = torch.where(mask[e])
            if not len(tokens):
                continue
            check_deadline(deadline)
            if work is not None:
                work['fp32_r2_expert_token_calls'] = work.get('fp32_r2_expert_token_calls', 0) + len(tokens)
            projected = frozen_linear(h[tokens], experts.gate_up_proj[e], deadline=deadline)
            gate, up = projected.chunk(2, dim=-1)
            product = F.silu(gate) * up
            update = frozen_linear(product, experts.down_proj[e], deadline=deadline)
            weighted = update * weights[tokens, slots, None].detach().float()
            out = out.index_add(0, tokens, weighted)
    if work is not None:
        work['fp32_r2_forward_completed'] = work.get('fp32_r2_forward_completed', 0) + 1
    return out
