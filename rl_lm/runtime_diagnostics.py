"""Opt-in runtime diagnostics. Never used by training or reward evaluation.

Scoped readout overrides are process-local and NOT thread-safe. The standalone
CLI is single-threaded; no model parameter or checkpoint is converted in place.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from model.top2 import Top2MoE, set_top2_dispatch_backend, top2_dispatch_identity
from rl_lm import inference


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def parse_chunks(value):
    chunks = value.split(',')
    if not chunks or len(chunks) != len(set(chunks)):
        raise ValueError('head chunks must be nonempty and unique')
    for chunk in chunks:
        if chunk != 'full' and (not chunk.isdecimal() or int(chunk) < 1):
            raise ValueError('head chunks must be positive integers or full')
    normalized = [v if v == 'full' else str(int(v)) for v in chunks]
    if len(normalized) != len(set(normalized)):
        raise ValueError('duplicate effective chunk sizes')
    return normalized


def fresh_output(path):
    path = Path(path)
    # Even an incomplete previous run is evidence; never overwrite it.
    if path.exists():
        raise FileExistsError('choose a new output directory; existing paths are protected')
    path.mkdir(parents=True)
    return path


class CachedBF16Head:
    """Snapshot of frozen tied-head weights; build outside timed regions."""
    def __init__(self, model):
        if model.embedding.weight.requires_grad:
            raise ValueError('cached head requires frozen embedding weights')
        self.model = model
        self.weight = model.embedding.weight.detach().to(torch.bfloat16).clone()
        self.bytes = self.weight.numel() * self.weight.element_size()
        self.source = model.embedding.weight
        self.source_version = self.source._version
        self.source_signature = (self.source.data_ptr(), self.source.dtype, self.source.device, tuple(self.source.shape))

    @torch.no_grad()
    def __call__(self, model, h, *, chunk=64, compute_dtype='bfloat16', ledger=None):
        source = model.embedding.weight
        if (model is not self.model or source is not self.source or source.requires_grad or
                source._version != self.source_version or
                (source.data_ptr(), source.dtype, source.device, tuple(source.shape)) != self.source_signature):
            raise RuntimeError('cached head source changed')
        if compute_dtype != 'bfloat16':
            raise ValueError('cached BF16 head only supports bfloat16')
        positive_int(chunk, 'chunk')
        if h.ndim != 2 or h.dtype != torch.float32:
            raise ValueError('expected FP32 hidden [N,D]')
        output = torch.empty((len(h), self.weight.shape[0]), device=h.device, dtype=torch.bfloat16)
        for start in range(0, len(h), chunk):
            valid = min(chunk, len(h) - start)
            hc = h[start:start+valid]
            if valid < chunk:
                hc = torch.cat((hc, h.new_zeros((chunk-valid, h.shape[1]))))
            with torch.autocast(device_type=h.device.type, dtype=torch.bfloat16):
                logits = F.linear(model.final_norm(hc), self.weight)
            output[start:start+valid].copy_(logits[:valid])
            if ledger is not None:
                ledger.physical_head_rows += chunk
        return output


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str  # native, dense, online
    prefix: str = 'legacy'
    head_chunk: str = '64'
    cached_head: bool = False
    policy: str = 'canonical'
    max_calls: int = 1
    terminal: str = 'sparse_active_merged_v3'


@contextmanager
def runtime_scope(model, variant, cached_head=None):
    """Configure before timing and restore heterogeneous runtimes on every exit."""
    previous = [(m, m.dispatch_backend) for m in model.modules() if isinstance(m, Top2MoE)]
    previous_readout = inference.production_logits
    try:
        set_top2_dispatch_backend(model, 'legacy')
        if variant.kind == 'native':
            set_top2_dispatch_backend(model, variant.prefix)
        elif variant.kind == 'online':
            terminal = model.config.adaptive_layers[-1]
            for block in model.blocks[:terminal]:
                set_top2_dispatch_backend(block, variant.prefix)
        elif variant.kind != 'dense':
            raise ValueError('unknown variant kind')
        if variant.cached_head:
            if variant.kind != 'online' or cached_head is None:
                raise ValueError('cached head requires an online candidate and prepared cache')
            inference.production_logits = cached_head
        identity = {'variant': variant.__dict__, 'dispatch': top2_dispatch_identity(model),
                    'head_weight_cache_bytes': cached_head.bytes if variant.cached_head else 0}
        def verify():
            if top2_dispatch_identity(model) != identity['dispatch']:
                raise RuntimeError('runtime dispatch changed during measurement')
            expected = cached_head if variant.cached_head else previous_readout
            if inference.production_logits is not expected:
                raise RuntimeError('runtime readout changed during measurement')
        verify()
        yield identity, verify
    finally:
        inference.production_logits = previous_readout
        for module, backend in previous:
            module.dispatch_backend = backend


def logits_of(output):
    return output.logits


@torch.no_grad()
def parity_metrics(reference, candidate, labels, *, nll_tolerance, require_actions=False, ce_mean_abs_tolerance=None):
    """Token-weighted CE deltas, not a statistical test or equivalence proof."""
    if ce_mean_abs_tolerance is None:
        ce_mean_abs_tolerance = nll_tolerance
    if isinstance(ce_mean_abs_tolerance, bool) or not math.isfinite(ce_mean_abs_tolerance) or ce_mean_abs_tolerance < 0:
        raise ValueError('CE mean absolute tolerance must be finite and nonnegative')
    if isinstance(nll_tolerance, bool) or not math.isfinite(nll_tolerance) or nll_tolerance < 0:
        raise ValueError('nll tolerance must be finite and nonnegative')
    a, b = logits_of(reference), logits_of(candidate)
    if a.shape != b.shape or a.shape[:-1] != labels.shape:
        raise ValueError('logits and labels shapes differ')
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError('nonfinite candidate/reference logits')
    valid = labels.reshape(-1) != -100
    if not bool(valid.any()):
        raise ValueError('verification requires supervised tokens')
    # Bounded CE temporaries; retain the same 8192-row evaluation windows.
    losses = []
    for logits in (a, b):
        flat = logits.reshape(-1, logits.shape[-1])
        chunks = []
        for start in range(0, len(flat), 8192):
            chunks.append(F.cross_entropy(flat[start:start+8192].float(), labels.reshape(-1)[start:start+8192],
                                          reduction='none', ignore_index=-100))
        losses.append(torch.cat(chunks)[valid].double())
    diff = losses[1] - losses[0]
    action_diff = round_diff = None
    if require_actions:
        if reference.actions is None or candidate.actions is None:
            raise ValueError('action verification requires return_trace=True')
        action_diff = int((reference.actions != candidate.actions).sum())
        round_diff = int((reference.rounds != candidate.rounds).sum())
    delta = float(diff.mean())
    return {'tokens': int(valid.sum()), 'reference_nll': float(losses[0].mean()),
            'candidate_nll': float(losses[1].mean()), 'delta_nll': delta,
            'ce_mean_abs': float(diff.abs().mean()), 'ce_max_abs': float(diff.abs().max()), 'action_differences': action_diff,
            'round_differences': round_diff, 'nll_tolerance': nll_tolerance, 'ce_mean_abs_tolerance': ce_mean_abs_tolerance,
            'eligible': abs(delta) <= nll_tolerance and float(diff.abs().mean()) <= ce_mean_abs_tolerance and action_diff in (None, 0) and round_diff in (None, 0)}


@contextmanager
def preserve_precision():
    precision = torch.get_float32_matmul_precision()
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
