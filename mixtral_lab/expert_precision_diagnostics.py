"""Read-only, sampled ATen traces of the supported eager Mixtral experts.

Never a production backend or a parity certificate. FP32 uses the same quantized
weights and inputs. Backend flags are process-global: run these probes serially.
"""
from contextlib import contextmanager
import time

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from .loop_parity_diagnostics import _difference

# Actual eager operators, not a substituted expert forward. Fail closed if the
# installed backend uses another operation/order (including fused activation).
_OPS = ('aten.mm.default', 'aten.silu.default', 'aten.mul.Tensor',
        'aten.mm.default', 'aten.mul.Tensor', 'aten.index_add_.default')
_NAMES = ('gate_up', 'activation', 'product', 'down', 'weighted', 'accumulation')
_WATCH = set(_OPS)


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('expert diagnostic deadline exhausted between operations')


def _bump(work, key, count=1):
    work[key] = work.get(key, 0) + count


@contextmanager
def _flags(*, reduction=None, tf32=None):
    backend = torch.backends.cuda.matmul
    old_reduction = backend.allow_bf16_reduced_precision_reduction
    old_precision = torch.get_float32_matmul_precision()
    try:
        if reduction is not None:
            backend.allow_bf16_reduced_precision_reduction = reduction
        if tf32 is not None:
            torch.set_float32_matmul_precision('high' if tf32 else 'highest')
        yield
    finally:
        backend.allow_bf16_reduced_precision_reduction = old_reduction
        torch.set_float32_matmul_precision(old_precision)


def _plan(pairs, sample_positions, num_experts):
    """Exactly HF's expert order and slot-major gather order."""
    mask = F.one_hot(pairs, num_classes=num_experts).permute(2, 1, 0)
    result = []
    for expert in range(num_experts):
        slots, tokens = torch.where(mask[expert])
        if not tokens.numel():
            continue
        # Mapping from actual gathered rows to the common sample ordering.
        match = tokens[:, None] == sample_positions[None, :]
        rows, sample = torch.where(match)
        result.append(dict(expert=expert, rows=rows, tokens=tokens,
                           slots=slots, sample=sample))
    return result


class _Trace(TorchDispatchMode):
    def __init__(self, plan, sample_positions, deadline):
        self.plan = plan
        self.sample_positions = sample_positions
        self.deadline = deadline
        self.event = 0
        self.records = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if name not in _WATCH:
            return func(*args, **(kwargs or {}))
        _check_deadline(self.deadline)
        group, step = divmod(self.event, len(_OPS))
        if group >= len(self.plan) or name != _OPS[step]:
            raise ValueError('unsupported eager expert operator sequence: ' + name)
        entry = self.plan[group]
        e = entry['expert']
        record = self.records.setdefault(e, {})
        if step == 0:
            record['input'] = args[0][entry['rows']].detach().clone()
        if step == 5:
            if not torch.equal(args[2], entry['tokens']):
                raise ValueError('expert gather/index_add ordering differs from supported eager path')
            record['cast'] = args[3][entry['rows']].detach().clone()
        out = func(*args, **(kwargs or {}))
        rows = self.sample_positions if step == 5 else entry['rows']
        record[_NAMES[step]] = out[rows].detach().clone()
        record['gathered_rows'] = len(entry['tokens'])
        record['sample_slots'] = entry['slots'][entry['rows']].detach().clone()
        record['sample_indices'] = entry['sample'].detach().clone()
        self.event += 1
        return out

    def finish(self):
        if self.event != len(self.plan) * len(_OPS):
            raise ValueError('incomplete eager expert operator trace')


def _call(experts, h, pairs, weights, work, deadline, trace=None):
    _check_deadline(deadline)
    _bump(work, 'precision_expert_forward_attempts')
    _bump(work, 'precision_expert_token_calls', pairs.numel())
    if trace is None:
        out = experts(h, pairs, weights)
    else:
        with trace:
            out = experts(h, pairs, weights)
        trace.finish()
    _bump(work, 'precision_expert_forward_completed')
    if not bool(torch.isfinite(out).all()):
        raise ValueError('nonfinite expert diagnostic output')
    _check_deadline(deadline)
    return out


def _fp32_reference(experts, h, pairs, weights, work, deadline):
    """At most eight tokens; convert/release ONE projection at a time."""
    _bump(work, 'precision_reference_attempts')
    out = torch.zeros_like(h, dtype=torch.float32)
    stages = {}
    plan = _plan(pairs, torch.arange(len(h), device=h.device), experts.num_experts)
    with _flags(tf32=False), torch.autocast(device_type=h.device.type, enabled=False):
        for entry in plan:
            _check_deadline(deadline)
            e, tokens, slots = entry['expert'], entry['tokens'], entry['slots']
            x = h[tokens].float()
            weight = experts.gate_up_proj[e].float()
            _bump(work, 'precision_reference_projection_attempts')
            projected = F.linear(x, weight)
            _bump(work, 'precision_reference_projection_calls')
            del weight
            gate, up = projected.chunk(2, dim=-1)
            activated = F.silu(gate)
            product = activated * up
            _check_deadline(deadline)
            weight = experts.down_proj[e].float()
            _bump(work, 'precision_reference_projection_attempts')
            down = F.linear(product, weight)
            _bump(work, 'precision_reference_projection_calls')
            del weight
            weighted = down * weights[tokens, slots, None].float()
            out.index_add_(0, tokens, weighted)
            stages[e] = dict(input=x, gate_up=projected, activation=activated,
                             product=product, down=down, weighted=weighted,
                             cast=weighted, accumulation=out.clone())
            _bump(work, 'precision_reference_token_calls', len(tokens))
    _bump(work, 'precision_reference_completed')
    if not bool(torch.isfinite(out).all()):
        raise ValueError('nonfinite FP32 reference')
    return out, stages


def _stats(a, b, atol, rtol):
    result = _difference(a, b, atol, rtol)
    if result['nonfinite_elements']:
        raise ValueError('nonfinite expert diagnostic stage')
    return result


@torch.no_grad()
def diagnose_expert_precision(experts, h, pairs, weights, selected, *,
                              dense_output, compact_output, work=None,
                              deadline=None, atol=1e-4, rtol=1e-4, progress=None):
    """Trace actual eager BF16/FP32 forwards, retaining <=8 sampled tokens.

    Full dense and compact row counts are preserved; ONLY retained observations
    and the FP32 reference are sampled. No original tensor or route is exported.
    A first divergent observed primitive is not proof of an underlying kernel bug.
    """
    if work is None:
        work = {}
    if progress is None:
        progress = {}
    progress.update(status='incomplete', stage='validation')
    if h.ndim != 2 or not 1 < len(h) <= 256:
        raise ValueError('expert diagnostic requires 2..256 flat rows')
    if (selected.ndim != 1 or selected.dtype != torch.long or
            not 0 < len(selected) < len(h) or selected.device != h.device or
            not bool(((selected >= 0) & (selected < len(h))).all()) or
            not torch.equal(selected, selected.unique(sorted=True))):
        raise ValueError('sorted unique mixed selected indices required')
    if (pairs.shape != (len(h), 2) or pairs.dtype != torch.long or
            weights.shape != pairs.shape or pairs.device != h.device or weights.device != h.device or
            not bool(((pairs >= 0) & (pairs < experts.num_experts)).all()) or
            bool((pairs[:, 0] == pairs[:, 1]).any())):
        raise ValueError('unique Top-2 routes on input device required')
    if h.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('only BF16/FP32 eager diagnostic supported')
    from transformers.models.mixtral.modeling_mixtral import MixtralExperts
    if type(experts) is not MixtralExperts or getattr(experts.config, '_experts_implementation', None) != 'eager':
        raise ValueError('only installed Transformers MixtralExperts eager implementation supported')
    if torch.is_autocast_enabled(h.device.type):
        raise ValueError('autocast must be disabled for expert diagnostics')
    if not all(bool(torch.isfinite(x).all()) for x in (h, weights)):
        raise ValueError('finite expert inputs required')
    _bump(work, 'precision_probe_attempts')
    sample = selected[:8]
    compact_sample = torch.arange(len(sample), device=h.device)
    dense_plan = _plan(pairs, sample, experts.num_experts)
    compact_plan = _plan(pairs[selected], compact_sample, experts.num_experts)
    baseline = dict(allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    progress.update(stage='actual_operator_traces', sampled_tokens=len(sample))
    traces = []
    for x, p, w, plan, positions in (
        (h, pairs, weights, dense_plan, sample),
        (h[selected], pairs[selected], weights[selected], compact_plan, compact_sample)):
        trace = _Trace(plan, positions, deadline)
        output = _call(experts, x, p, w, work, deadline, trace)
        traces.append((output, trace.records))
        progress['completed_traced_forwards'] = len(traces)
    dense, compact = traces
    observer = dict(dense=_stats(dense[0], dense_output, 0., 0.),
                    compact=_stats(compact[0], compact_output, 0., 0.),
                    dense_final_accumulation=_stats(dense[1][dense_plan[-1]['expert']]['accumulation'], dense[0][sample], 0., 0.),
                    compact_final_accumulation=_stats(compact[1][compact_plan[-1]['expert']]['accumulation'], compact[0][compact_sample], 0., 0.))
    verified = all(v['exact_differences'] == 0 for v in observer.values())
    progress.update(stage='fp32_reference', observer_comparisons=observer, observer_matches_untraced=verified)
    reference, ref_stages = _fp32_reference(experts, h[sample], pairs[sample], weights[sample], work, deadline)
    progress.update(stage='stage_comparisons', fp32_reference_completed=True)
    stages = []
    # Reference local gather ordering is identical after restricting to sampled tokens.
    for e, ref in ref_stages.items():
        a, b = dense[1][e], compact[1][e]
        if not (torch.equal(a['sample_indices'], b['sample_indices']) and
                torch.equal(a['sample_slots'], b['sample_slots'])):
            raise ValueError('sampled expert routing alignment failed')
        comparisons = {}
        for name in ('input', 'gate_up', 'activation', 'product', 'down', 'weighted', 'cast', 'accumulation'):
            comparisons[name] = dict(compact_vs_dense=_stats(b[name], a[name], atol, rtol),
                                     dense_vs_fp32=_stats(a[name], ref[name], atol, rtol),
                                     compact_vs_fp32=_stats(b[name], ref[name], atol, rtol))
        stages.append(dict(expert=e, dense_gathered_rows=a['gathered_rows'],
                           compact_gathered_rows=b['gathered_rows'], sampled_contributions=len(a['sample_indices']),
                           comparisons=comparisons,
                           first_exact_difference=(next((k for k,v in comparisons.items() if v['compact_vs_dense']['exact_differences']), None) if verified else None),
                           first_tolerance_failure=(next((k for k,v in comparisons.items() if v['compact_vs_dense']['mismatched_elements']), None) if verified else None)))
    def ref_metrics(d, c):
        return dict(dense_vs_fp32=_stats(d[sample], reference, atol, rtol),
                    compact_vs_fp32=_stats(c[compact_sample], reference, atol, rtol),
                    compact_vs_dense=_stats(c, d[selected], atol, rtol))
    progress.update(stage='reduction_probe', experts=stages, baseline_output=ref_metrics(dense_output,compact_output))
    reduction = dict(status='not_cuda_bf16_tested')
    if h.device.type == 'cuda' and h.dtype == torch.bfloat16:
        with _flags(reduction=False):
            d = _call(experts, h, pairs, weights, work, deadline)
            c = _call(experts, h[selected], pairs[selected], weights[selected], work, deadline)
            reduction = dict(status='complete', effective_setting=False,
                             changed_from_baseline=baseline['allow_bf16_reduced_precision_reduction'],
                             comparisons=ref_metrics(d,c),
                             dense_change=_stats(d,dense_output,atol,rtol),
                             compact_change=_stats(c,compact_output,atol,rtol))
    progress.update(status='complete', stage='finished')
    _bump(work, 'precision_probe_completed')
    return dict(status='complete' if verified else 'observer_mismatch',
                observer_matches_untraced=verified, observer_comparisons=observer,
                scope='sampled actual ATen observations, not a production repair or acceptance gate',
                sampled_tokens=len(sample), selected_tokens=len(selected), total_tokens=len(h),
                sample_rule='first min(8, selected_count) selected tokens; no token identities exported',
                reference='FP32 arithmetic on identical already-quantized inputs/weights and fixed routes; TF32 off; not exact ground truth',
                baseline_flags=baseline, experts=stages,
                baseline_output=ref_metrics(dense_output,compact_output), reduction_probe=reduction)
