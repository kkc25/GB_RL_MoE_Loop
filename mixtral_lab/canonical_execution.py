"""Opt-in fixed-row arithmetic for genuinely selected token-local R2 work.

Padding is limited to active rows/expert assignments; it is counted, not hidden.
A fixed shape is a candidate numerical protocol, NOT a GPU parity guarantee.
"""
import time
import torch
import torch.nn.functional as F
from .fp32_experts import check_deadline, correction_fp32, frozen_linear, precision_scope

TILE = 32
MODE = 'fp32_canonical32'


def tiled(x, function, *, deadline=None, work=None, label='canonical'):
    if x.ndim != 2 or not len(x):
        raise ValueError('canonical tiles require nonempty flat rows')
    outputs = None
    for start in range(0, len(x), TILE):
        check_deadline(deadline)
        block = x[start:start+TILE]
        count = len(block)
        padded = F.pad(block, (0, 0, 0, TILE-count)) if count < TILE else block
        if work is not None:
            for suffix, amount in (('logical_row_attempts',count),('padded_row_attempts',TILE),('tile_attempts',1)):
                key=label+'_'+suffix;work[key]=work.get(key,0)+amount
        with precision_scope(x.device.type):
            value = function(padded)
        values = value if isinstance(value,tuple) else (value,)
        if outputs is None:outputs=[[] for _ in values]
        if len(outputs)!=len(values) or any(v.shape[0]!=TILE for v in values):
            raise ValueError('canonical function changed tile contract')
        for bucket,v in zip(outputs,values):bucket.append(v[:count])
        if work is not None:
            key=label+'_tile_completed';work[key]=work.get(key,0)+1
        check_deadline(deadline)
    result=tuple(torch.cat(bucket,dim=0) for bucket in outputs)
    return result if isinstance(value,tuple) else result[0]


def norm_and_correction(adapter, branch, active, work):
    deadline=branch.scale_config.absolute_deadline_utc
    x=tiled(active,adapter.layer.post_attention_layernorm,deadline=deadline,work=work,label='canonical_norm')
    correction=tiled(x,lambda rows:correction_fp32(branch.correction,rows,deadline=deadline),
        deadline=deadline,work=work,label='canonical_correction')
    return x,correction


def route(adapter, x, *, deadline=None, work=None):
    with torch.no_grad():
        return tiled(x,adapter.moe.gate,deadline=deadline,work=work,label='canonical_router')


def experts(adapter, h, pairs, weights, *, deadline=None, work=None):
    e=adapter.moe.experts
    from transformers.models.mixtral.modeling_mixtral import MixtralExperts
    if (type(e) is not MixtralExperts or e.config.hidden_act!='silu' or
            e.config._experts_implementation!='eager' or any(p.requires_grad for p in e.parameters())):
        raise ValueError('canonical protocol requires frozen packed eager Mixtral SiLU experts')
    if (h.ndim!=2 or h.dtype!=torch.float32 or pairs.shape!=(len(h),2) or weights.shape!=pairs.shape or
            pairs.dtype!=torch.long or pairs.device!=h.device or weights.device!=h.device or
            bool((pairs[:,0]==pairs[:,1]).any()) or not bool(((pairs>=0)&(pairs<e.num_experts)).all())):
        raise ValueError('canonical FP32 inputs and distinct Top-2 routes required')
    if e.gate_up_proj.device!=h.device or e.down_proj.device!=h.device:
        raise ValueError('canonical experts must reside on input device')
    mask=F.one_hot(pairs.detach(),num_classes=e.num_experts).permute(2,1,0)
    out=torch.zeros_like(h)
    with precision_scope(h.device.type):
        for expert in range(e.num_experts):
            slots,tokens=torch.where(mask[expert])
            if not len(tokens):continue
            # Include weights in the tile so activation AND weighting use fixed rows.
            packed=torch.cat((h[tokens],weights[tokens,slots,None].detach().float()),dim=1)
            def compute(rows):
                projected=frozen_linear(rows[:,:-1].contiguous(),e.gate_up_proj[expert],deadline=deadline)
                gate,up=projected.chunk(2,-1)
                down=frozen_linear(F.silu(gate)*up,e.down_proj[expert],deadline=deadline)
                return down*rows[:,-1:]
            weighted=tiled(packed,compute,deadline=deadline,work=work,label='canonical_expert')
            # At most one assignment per token within an expert. Expert order is fixed.
            out=out.index_add(0,tokens,weighted)
    return out


def readout(adapter, hidden, config, *, work=None, trace=None):
    norm,head=adapter.model.model.norm,adapter.model.lm_head
    np,hp=next(norm.parameters()),next(head.parameters())
    if any(p.requires_grad for p in adapter.model.parameters()):
        raise ValueError('canonical readout requires frozen backbone')
    vocab=int(adapter.model.config.vocab_size)
    n=hidden.numel()//adapter.hidden_size
    if ((n+TILE-1)//TILE)*TILE*vocab>config.max_logit_elements:
        raise ValueError('canonical padded/full logits exceed allocation bound')
    def compute(rows):
        normalized=norm(rows.to(device=np.device,dtype=np.dtype))
        logits=head(normalized.to(device=hp.device,dtype=hp.dtype))
        return (normalized,logits) if trace is not None else logits
    value=tiled(hidden.reshape(-1,adapter.hidden_size),compute,
        deadline=config.absolute_deadline_utc,work=work,label='canonical_head')
    if trace is not None:
        normalized,logits=value
        trace['readout_normalized']=normalized.detach().clone()
    else:logits=value
    return logits.reshape(*hidden.shape[:-1],vocab)


def observe(trace,name,value):
    if trace is not None:trace[name]=value.detach().clone()


@torch.no_grad()
def diagnose_path(adapter, cache, branch, mask, *, atol=1e-4,rtol=1e-4,work=None):
    """Observe actual implementations with identical adapter state; <=256 rows."""
    from .loop_scale import _ce
    from .loop_parity_diagnostics import _difference
    if torch.device(adapter.device).type=='cuda':
        for device in range(torch.cuda.device_count()):torch.cuda.synchronize(device)
    started=time.perf_counter()
    h=cache.native_h1.detach().to(adapter.device);n=h.numel()//adapter.hidden_size
    if n>256 or mask.dtype!=torch.bool or mask.shape!=(n,) or mask.device!=h.device:
        raise ValueError('bounded flat path diagnostic required')
    idx=mask.nonzero().flatten()
    if len(idx)!=1:raise ValueError('singleton path diagnostic required')
    if work is None:work={}
    work['path_diagnostic_attempts']=work.get('path_diagnostic_attempts',0)+1
    dense_trace={};compact_trace={}
    dense=branch(adapter,h,mask,compact=False,work=work,trace=dense_trace)
    compact=branch(adapter,h,mask,compact=True,work=work,trace=compact_trace)
    dense_check=branch(adapter,h,mask,compact=False,work=work)
    compact_check=branch(adapter,h,mask,compact=True,work=work)
    observer=dict(dense=_difference(dense,dense_check,0.,0.),compact=_difference(compact,compact_check,0.,0.))
    verified=all(v['exact_differences']==0 and v['nonfinite_elements']==0 for v in observer.values())
    names=('normalized_input','correction_fp32','router_logits','router_weights','router_pairs',
           'expert_input','expert_output','residual_fp32','cast_hidden')
    comparisons={}
    for name in names:
        tolerance=(0.,0.) if name=='router_pairs' else (atol,rtol)
        comparisons[name]=_difference(compact_trace[name],dense_trace[name][idx],*tolerance)
    # SAME hidden values, independent of R2/route recomputation: isolate readout shape.
    target=cache.targets.to(adapter.device).reshape(-1);head_dense={};head_small={}
    config=branch.scale_config
    ce_dense=_ce(adapter,dense,target,work=work,config=config,trace=head_dense)
    ce_small=_ce(adapter,dense.reshape(n,-1)[idx],target[idx],work=work,config=config,trace=head_small)
    head={name:_difference(head_small[name].reshape(len(idx),-1),head_dense[name].reshape(n,-1)[idx],atol,rtol)
          for name in ('readout_normalized','logits')}
    head['ce']=_difference(ce_small.reshape(-1),ce_dense.reshape(-1)[idx],atol,rtol)
    if any(v['nonfinite_elements'] for v in [*comparisons.values(),*head.values(),*observer.values()]):
        raise FloatingPointError('nonfinite path diagnostic')
    if h.device.type=='cuda':
        for device in range(torch.cuda.device_count()):torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-started
    work['path_diagnostic_seconds']=work.get('path_diagnostic_seconds',0.)+elapsed
    work['path_diagnostic_completed']=work.get('path_diagnostic_completed',0)+1
    return dict(status='complete' if verified else 'observer_mismatch',r2_precision=config.r2_precision,
        seconds=elapsed,observer_matches_untraced=verified,observer=observer,comparisons=comparisons,
        first_exact_difference=next((k for k,v in comparisons.items() if v['exact_differences']),None) if verified else None,
        first_tolerance_failure=next((k for k,v in comparisons.items() if v['mismatched_elements']),None) if verified else None,
        fixed_hidden_readout=head,scope='first observed stage, not proven root cause; no tensors or text serialized')
