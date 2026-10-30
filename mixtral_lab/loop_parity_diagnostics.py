"""Bounded, read-only causal probes of actual compact execution.

These probes never replace production computation, relax acceptance tolerances,
or certify GPU correctness. No token text, model weights or hidden tensors are
serialized. A mismatch remains a failed preflight even when probes explain it.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def _difference(left, right, atol, rtol):
    if left.shape != right.shape:
        raise ValueError('diagnostic shapes differ')
    a, b = left.detach().double(), right.detach().double()
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = (a-b).abs()
    allowance = atol + rtol*b.abs()
    ok = finite & (delta <= allowance)
    return dict(shape=list(a.shape), left_dtype=str(left.dtype), right_dtype=str(right.dtype),
                elements=a.numel(), nonfinite_elements=int((~finite).sum()),
                exact_differences=int((left != right).sum()),
                mismatched_elements=int((~ok).sum()),
                max_abs=float(delta[finite].max()) if bool(finite.any()) else None,
                reference_max_abs=float(b[finite].abs().max()) if bool(finite.any()) else None)


@torch.no_grad()
def _capture(adapter, branch, h1, mask, compact, work):
    """Hooks observe the *actual* forward; duplicated equations are not the oracle."""
    captured = {}; handles = []
    def put(name, value):
        if name in captured:
            raise ValueError('unexpected repeated diagnostic module call: '+name)
        captured[name] = value.detach().clone()
    def norm(_, args, out): put('normalized_input', out)
    def correction(_, args, out): put('correction_fp32', out)
    def gate(_, args, out):
        if not isinstance(out, tuple) or len(out) != 3:
            raise ValueError('unsupported router output contract')
        for name, value in zip(('router_logits','router_weights','router_pairs'), out): put(name, value)
    def expert_in(_, args): put('expert_input', args[0])
    def expert_out(_, args, out): put('expert_output', out)
    try:
        handles.append(adapter.layer.post_attention_layernorm.register_forward_hook(norm))
        handles.append(branch.correction.register_forward_hook(correction))
        handles.append(adapter.moe.gate.register_forward_hook(gate))
        handles.append(adapter.moe.experts.register_forward_pre_hook(expert_in))
        handles.append(adapter.moe.experts.register_forward_hook(expert_out))
        captured['hidden'] = branch(adapter,h1,mask,compact=compact,work=work).detach().clone()
    finally:
        for handle in handles: handle.remove()
    return captured


@torch.no_grad()
def diagnose_compaction(adapter, cache, branch, mask, *, atol=1e-4, rtol=1e-4, work=None):
    """One mixed mask, <=256 rows. Freeze identical expert inputs/routes to
    isolate expert batching; compare identical hidden rows through the head.
    Neither isolation experiment is a deployable policy or a proposed fallback.
    """
    from .loop_scale import _ce
    if branch.kind != 'r2': raise ValueError('R2 diagnostic required')
    h1 = cache.native_h1.detach().to(adapter.device)
    n = h1.numel() // adapter.hidden_size
    if n > 256 or h1.shape[-1] != adapter.hidden_size:
        raise ValueError('diagnostic limited to one document of at most256 rows')
    if mask.dtype != torch.bool or mask.shape != (n,) or mask.device != h1.device:
        raise ValueError('explicit flat diagnostic mask required')
    idx = mask.nonzero().flatten()
    if not 0 < len(idx) < n: raise ValueError('diagnostic requires a mixed mask')
    if work is None: work = {}
    work['diagnostic_attempts'] = work.get('diagnostic_attempts',0)+1
    compact = _capture(adapter,branch,h1,mask,True,work)
    dense = _capture(adapter,branch,h1,mask,False,work)
    comparisons = {}
    ordered = ('normalized_input','correction_fp32','router_logits','router_weights',
               'router_pairs','expert_input','expert_output')
    for name in ordered:
        tol = (0.,0.) if name == 'router_pairs' else (atol,rtol)
        comparisons[name] = _difference(compact[name],dense[name][idx],*tol)
    comparisons['hidden'] = _difference(compact['hidden'].reshape(n,-1)[idx],
                                         dense['hidden'].reshape(n,-1)[idx],atol,rtol)
    # Expert-only shape probe: SAME exact input, pairs and weights; only row count differs.
    work['extra_expert_forward_attempts'] = work.get('extra_expert_forward_attempts',0)+1
    work['extra_expert_token_calls'] = work.get('extra_expert_token_calls',0)+2*len(idx)
    fixed = adapter.moe.experts(dense['expert_input'][idx],dense['router_pairs'][idx],dense['router_weights'][idx])
    work['extra_expert_forward_completed'] = work.get('extra_expert_forward_completed',0)+1
    expert_only = _difference(fixed,dense['expert_output'][idx],atol,rtol)
    # Head-only shape probe: SAME hidden rows, no R2 or gate recomputation.
    target = cache.targets.to(adapter.device).reshape(-1)
    full_ce = _ce(adapter,dense['hidden'],target,work=work)
    sliced_ce = _ce(adapter,dense['hidden'].reshape(n,-1)[idx],target[idx],work=work)
    valid = cache.valid_mask.to(idx.device)[idx]
    head_only = _difference(sliced_ce.reshape(-1)[valid],full_ce.reshape(-1)[idx][valid],atol,rtol)
    work['diagnostic_completed'] = work.get('diagnostic_completed',0)+1
    return dict(status='complete',scope='diagnostic only; no tolerance or execution fallback',
                selected_rows=int(len(idx)),total_rows=n,tolerances=dict(atol=atol,rtol=rtol),
                comparisons=comparisons,
                first_exact_difference=next((k for k,v in comparisons.items() if v['exact_differences']),None),
                first_tolerance_failure=next((k for k,v in comparisons.items() if v['mismatched_elements']),None),
                fixed_dense_inputs_routes_expert_only=expert_only,
                fixed_dense_hidden_head_only=head_only,
                interpretation='First divergence is observed location, not proof of ultimate root cause; expert/head isolation changes row count only.')
