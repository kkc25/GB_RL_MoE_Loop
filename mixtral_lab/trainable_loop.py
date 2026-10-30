"""Actual second expert-FFN training, then label-free hard conditional execution.

Native R1, backbone and native R2 pair weights are frozen. The learned rank-r
input correction does NOT replace the second expert call. No amplitude gate,
policy KL, STOP option or development early-stopping exists during branch fit.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from .adapter_fit import _assert_frozen, _tensor_hash, _atomic_write
from .document_training import (_integer, _number, _validate_cache, _cache_binding,
    cache_documents, document_losses, _comparison, _inputs)
from .second_round import differentiable_readout
from .recovery_adapter import _development_contract, _validate_development

FORMAT = 'mixtral-trainable-expert-loop-v1'

@dataclass(frozen=True)
class LoopConfig:
    rank: int = 8
    epochs: int = 8
    positions_per_document: int = 32
    max_updates: int = 2048
    learning_rate: float = 1e-3
    gradient_clip: float = 1.
    seed: int = 17
    gate_updates: int = 512
    gate_batch: int = 256
    gate_learning_rate: float = 1e-3
    eligibility_fraction: float = .05
    eligibility_gain: float = .01
    eligibility_top_fraction: float = .25
    eligibility_top_mean: float = .005
    thresholds: tuple = (0., .001, .005, .01, .02, .05)
    max_cache_bytes: int = 1024**3
    max_logit_elements: int = 16_777_216
    max_train_tokens: int = 8192
    parity_atol: float = 1e-4
    parity_rtol: float = 1e-4

    def validate(self):
        for name, low, high in [('rank',1,256),('epochs',1,32),('positions_per_document',1,4096),
                ('max_updates',1,10000),('seed',0,2**63-1),('gate_updates',1,4096),
                ('gate_batch',1,8192),('max_cache_bytes',1,16*1024**3),
                ('max_logit_elements',1,134217728),('max_train_tokens',1,100000)]:
            _integer(getattr(self,name),name,low,high)
        for name in ('learning_rate','gate_learning_rate','gradient_clip'):
            _number(getattr(self,name),name,0.,1.,True)
        for name in ('eligibility_fraction','eligibility_gain','eligibility_top_fraction',
                     'eligibility_top_mean','parity_atol','parity_rtol'):
            _number(getattr(self,name),name,0.,1.,False)
        if not 0 < self.eligibility_top_fraction <= 1: raise ValueError('invalid top fraction')
        if not isinstance(self.thresholds,(tuple,list)) or not 1<=len(self.thresholds)<=16:
            raise ValueError('bounded frozen threshold grid required')
        for x in self.thresholds: _number(x,'threshold',0.,1.)
        if list(self.thresholds)!=sorted(set(self.thresholds)): raise ValueError('thresholds must be sorted unique')

class InputCorrection(nn.Module):
    """FP32 rank-r linear correction, zero up init, fixed scale exactly one."""
    def __init__(self, hidden_size, config, device='cpu'):
        super().__init__(); config.validate()
        if type(hidden_size) is not int or not config.rank<=hidden_size<=65536: raise ValueError('invalid hidden dimension/rank')
        self.config=config; self.hidden_size=hidden_size
        g=torch.Generator().manual_seed(config.seed)
        self.down=nn.Parameter((torch.randn(config.rank,hidden_size,generator=g)/math.sqrt(hidden_size)).to(device))
        self.up=nn.Parameter(torch.zeros(hidden_size,config.rank,device=device))
    def forward(self,x):
        return F.linear(F.linear(x.detach().float(),self.down),self.up)

class GainGate(nn.Module):
    """Pre-R2 h1 only; neither labels nor extra FFN are needed for inference."""
    def __init__(self, hidden_size, device='cpu'):
        super().__init__(); self.hidden_size=hidden_size
        self.weight=nn.Parameter(torch.zeros(hidden_size,device=device))
        self.bias=nn.Parameter(torch.zeros(1,device=device))
    @staticmethod
    def features(h1):
        x=h1.detach().float()
        return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
    def forward(self,h1):
        return F.linear(self.features(h1),self.weight[None,:],self.bias).squeeze(-1)/math.sqrt(self.hidden_size)

class LoopBranch(nn.Module):
    def __init__(self, hidden_size, config, kind='r2', device='cpu'):
        super().__init__()
        if kind not in ('r2','parameter_control'): raise ValueError('unknown branch kind')
        self.kind=kind; self.config=config; self.hidden_size=hidden_size
        self.correction=InputCorrection(hidden_size,config,device)
    def forward(self, adapter, h1, mask=None, *, compact=True, work=None):
        _assert_frozen(adapter)
        if h1.shape[-1]!=self.hidden_size or h1.requires_grad: raise ValueError('detached native h1 required')
        flat=h1.reshape(-1,self.hidden_size)
        if mask is None: mask=torch.ones(len(flat),dtype=torch.bool,device=flat.device)
        if mask.dtype!=torch.bool or mask.shape!=(len(flat),) or mask.device!=flat.device:
            raise ValueError('boolean flat mask on terminal device required')
        selected=torch.nonzero(mask).flatten()
        if work is not None:
            work['candidate_calls']=work.get('candidate_calls',0)+1
            work['r2_selected_tokens']=work.get('r2_selected_tokens',0)+(len(selected) if self.kind=='r2' else 0)
        if not len(selected): return h1.clone()
        # Forced-on full shape and conditional compact shape are separately audited.
        active=flat[selected] if compact and len(selected)!=len(flat) else flat
        x=adapter.layer.post_attention_layernorm(active)
        correction=self.correction(x)
        if self.kind=='r2':
            with torch.no_grad(): _,weights,pairs=adapter.moe.gate(x)
            if work is not None:
                work['extra_expert_forward_attempts']=work.get('extra_expert_forward_attempts',0)+1
                work['extra_expert_token_calls']=work.get('extra_expert_token_calls',0)+2*len(active)
            update=adapter.moe.experts((x.float()+correction).to(x.dtype),pairs.detach(),weights.detach())
            if work is not None: work['extra_expert_forward_completed']=work.get('extra_expert_forward_completed',0)+1
        else: update=correction
        changed=active+update if self.kind=='r2' else (active.float()+update).to(active.dtype)
        if len(selected)==len(flat): return changed.reshape(h1.shape)
        # torch.index_copy retains gradient to the active correction/expert inputs.
        values=changed if compact else changed[selected]
        return flat.index_copy(0,selected,values).reshape(h1.shape)


def check_deadline(work):
    if work is not None and work.get('_deadline_monotonic') is not None and time.monotonic() >= work['_deadline_monotonic']:
        raise TimeoutError('shared formal wall-time budget exhausted; incomplete, not a scientific success')


def losses(adapter, cache, branch=None, *, mask=None, compact=True, work=None):
    check_deadline(work)
    hidden=cache.native_h1 if branch is None else branch(adapter,cache.native_h1,mask,compact=compact,work=work)
    if work is not None: work['head_forward_attempts']=work.get('head_forward_attempts',0)+1
    logits=differentiable_readout(adapter,hidden)
    if work is not None: work['head_forward_completed']=work.get('head_forward_completed',0)+1
    ce=document_losses(logits,cache)
    if not bool(torch.isfinite(ce).all()): raise FloatingPointError('nonfinite CE')
    return ce,hidden


def _write_json(path,value):
    _atomic_write(Path(path),lambda p:p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'))


def _sources():
    return {p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in
            ('trainable_loop.py','adapter.py','document_training.py','second_round.py')}


def save_checkpoint(path,branch,identity,epoch,*,gate=None,threshold=None,selection=None):
    state={k:v.detach().cpu().clone() for k,v in branch.state_dict().items()}
    payload=dict(format=FORMAT,kind=branch.kind,config=asdict(branch.config),hidden_size=branch.hidden_size,
        identity=copy.deepcopy(identity),sources=_sources(),epoch=epoch,state=state,state_sha256=_tensor_hash(state),
        gate=None if gate is None else {k:v.detach().cpu().clone() for k,v in gate.state_dict().items()},
        threshold=threshold,selection=selection)
    if payload['gate'] is not None: payload['gate_sha256']=_tensor_hash(payload['gate'])
    payload['metadata_sha256']=hashlib.sha256(json.dumps({k:v for k,v in payload.items() if k not in ('state','gate')},sort_keys=True,allow_nan=False).encode()).hexdigest()
    _atomic_write(Path(path),lambda p:torch.save(payload,p))
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(path,identity,config,device='cpu'):
    config.validate(); path=Path(path)
    if not path.is_file() or path.stat().st_size>160*1024**2: raise ValueError('checkpoint missing/too large')
    data=torch.load(io.BytesIO(path.read_bytes()),map_location='cpu',weights_only=True)
    # JSON normalization permits tuple/list roundtrips, not a scientific config change.
    metadata={k:v for k,v in data.items() if k not in ('state','gate','metadata_sha256')}
    if data.get('metadata_sha256')!=hashlib.sha256(json.dumps(metadata,sort_keys=True,allow_nan=False).encode()).hexdigest(): raise ValueError('checkpoint metadata digest mismatch')
    same=lambda a,b: json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
    if data.get('format')!=FORMAT or not same(data.get('identity'),identity) or not same(data.get('config'),asdict(config)) or data.get('sources')!=_sources():
        raise ValueError('checkpoint source/config/identity binding mismatch')
    if data.get('state_sha256')!=_tensor_hash(data['state']): raise ValueError('state digest mismatch')
    _integer(data.get('epoch'),'checkpoint epoch',0,config.epochs)
    if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in data['state'].values()): raise ValueError('nonfinite or non-FP32 state')
    branch=LoopBranch(data['hidden_size'],config,data['kind'],device); branch.load_state_dict(data['state'],strict=True)
    gate=None
    if data.get('gate') is not None:
        _number(data.get('threshold'),'frozen threshold',0.,1.)
        if data['threshold'] not in config.thresholds: raise ValueError('threshold outside frozen grid')
        if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in data['gate'].values()): raise ValueError('nonfinite gate')
        if data.get('gate_sha256')!=_tensor_hash(data['gate']): raise ValueError('gate digest mismatch')
        gate=GainGate(data['hidden_size'],device);gate.load_state_dict(data['gate'],strict=True);gate.eval()
    branch.eval(); return branch,gate,data


@torch.no_grad()
def evaluate(adapter,rows,branch=None,*,role='development',gate=None,threshold=None,mode='forced',work=None,config=None):
    """Fresh-prefix full-document NLL; gate only compacts the real second FFN."""
    config=config or (branch.config if branch is not None else LoopConfig())
    if role not in ('train','development','selection','calibration'): raise ValueError('final test forbidden')
    if mode not in ('forced','stop','gate','matched_random','fixed_hash_25','fixed_hash_50','fixed_hash_100'): raise ValueError('unknown evaluation mode')
    if mode in ('gate','matched_random') and (gate is None or threshold is None): raise ValueError('gate and frozen threshold required')
    records=[]
    for row in rows:
        check_deadline(work)
        cache=cache_documents(adapter,[row],max_cache_bytes=config.max_cache_bytes,max_logit_elements=config.max_logit_elements,work=work)[0]
        native,_=losses(adapter,cache,work=work)
        n=cache.native_h1.numel()//adapter.hidden_size
        mask=torch.ones(n,dtype=torch.bool,device=adapter.device);predictions=None
        if mode=='stop': mask.zero_()
        elif mode.startswith('fixed_hash_'):
            fraction=int(mode.rsplit('_',1)[1])/100
            mask=torch.tensor([int(hashlib.sha256((str(config.seed)+':'+row['id']+':'+str(i)).encode()).hexdigest()[:16],16)/2**64 < fraction for i in range(n)],device=adapter.device)
        elif mode in ('gate','matched_random'):
            predictions=gate(cache.native_h1).reshape(-1)
            mask=(predictions>threshold)
            if mode=='matched_random':
                seed=int(hashlib.sha256((str(config.seed)+':'+row['id']).encode()).hexdigest()[:15],16)
                g=torch.Generator().manual_seed(seed); count=int(mask.sum())
                indices=torch.randperm(n,generator=g)[:count].to(mask.device);mask.zero_();mask[indices]=True
        candidate,_=losses(adapter,cache,branch,mask=mask,work=work)
        valid=cache.valid_mask.to(candidate.device); tokens=int(valid.sum())
        records.append(dict(document_id=row['id'],tokens=tokens,native_nll=float(native[valid].double().mean()),
            candidate_nll=float(candidate[valid].double().mean()),continued_tokens=int(mask.sum()),padded_tokens=n,
            continued_supervised_tokens=int(mask[cache.valid_mask.to(mask.device)].sum()),
            valid_position_indices=torch.nonzero(cache.valid_mask).flatten().tolist(),
            native_ce=native[valid].detach().cpu().tolist(),candidate_ce=candidate[valid].detach().cpu().tolist(),
            continue_mask=mask[cache.valid_mask.to(mask.device)].cpu().tolist(),
            gate_predictions=None if predictions is None else predictions[cache.valid_mask.to(predictions.device)].cpu().tolist(),
            padded_continue_mask=mask.cpu().tolist(),padded_gate_predictions=None if predictions is None else predictions.cpu().tolist()))
    tokens=sum(r['tokens'] for r in records)
    if not tokens: raise ValueError('empty evaluation')
    return dict(status='ok',role=role,protocol='full_document_online_nll',document_ids=[r['document_id'] for r in records],
        nll=sum(r['candidate_nll']*r['tokens'] for r in records)/tokens,
        native_nll=sum(r['native_nll']*r['tokens'] for r in records)/tokens,tokens=tokens,per_document=records,
        continued_supervised_tokens=sum(r['continued_supervised_tokens'] for r in records),continued_tokens=sum(r['continued_tokens'] for r in records),padded_tokens=sum(r['padded_tokens'] for r in records),
        control_scope=('retrospective per-document exact-count matched random; not causal token streaming' if mode=='matched_random' else 'label-free token-local rule'))


@torch.no_grad()
def probe(adapter,caches,branch,work=None):
    gains=[]; records=[];effective=elements=0
    for c in caches:
        base,_=losses(adapter,c,work=work); ce,_=losses(adapter,c,branch,work=work)
        valid=c.valid_mask.to(ce.device); gain=(base-ce)[valid].detach().cpu();gains.append(gain)
        x=adapter.layer.post_attention_layernorm(c.native_h1.reshape(-1,adapter.hidden_size))
        reference=x if branch.kind=='r2' else c.native_h1.reshape(-1,adapter.hidden_size)
        effective+=int((((reference.float()+branch.correction(x)).to(reference.dtype))!=reference).sum());elements+=reference.numel()
        records.append(dict(document_id=c.document_id,tokens=len(gain),nll=float(ce[valid].double().mean()),native_nll=float(base[valid].double().mean()),
            valid_position_indices=torch.nonzero(c.valid_mask).flatten().tolist(),native_ce=base[valid].cpu().tolist(),candidate_ce=ce[valid].cpu().tolist(),
            continue_mask=[True]*len(gain),gate_predictions=None))
    g=torch.cat(gains);config=branch.config;k=max(1,math.ceil(len(g)*config.eligibility_top_fraction))
    fraction=float((g>config.eligibility_gain).double().mean());top=float(g.topk(k).values.double().mean())
    return dict(nll=sum(r['nll']*r['tokens'] for r in records)/len(g),tokens=len(g),per_document=records,
        useful_token_fraction=fraction,top_fraction_mean_gain=top,
        adapter_input_effective_change_fraction=effective/elements if branch.kind=='r2' else None,
        parameter_control_hidden_effective_change_fraction=effective/elements if branch.kind=='parameter_control' else None,gate_eligible=fraction>=config.eligibility_fraction and top>=config.eligibility_top_mean,
        eligibility_scope='train only; localized improvement, not proof of generalization')


def fit_branch(adapter,caches,development_evaluator,config,identity,output,*,kind='r2',work=None):
    config.validate(); _assert_frozen(adapter)
    check_deadline(work)
    contract=_development_contract(identity,development_evaluator)
    if [c.document_id for c in caches]!=identity['train']['selected_document_ids']: raise ValueError('train IDs mismatch')
    for c in caches: _validate_cache(adapter,c)
    required=sum(c.h0.numel()*c.h0.element_size()+c.native_h1.numel()*c.native_h1.element_size()+c.input_ids.numel()*25 for c in caches)
    if required>config.max_cache_bytes or any(c.input_ids.numel()*adapter.model.config.vocab_size>config.max_logit_elements for c in caches): raise ValueError('cache/logit quota exceeded')
    count=sum(int(c.valid_mask.sum()) for c in caches)
    if count>config.max_train_tokens: raise ValueError('train token quota exceeded')
    per_epoch=sum(math.ceil(int(c.valid_mask.sum())/config.positions_per_document) for c in caches)
    if per_epoch*config.epochs>config.max_updates: raise ValueError('budget cannot complete frozen epochs')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    work={} if work is None else work;work_initial=dict(work);started=time.perf_counter();binding=_cache_binding(caches)
    branch=LoopBranch(adapter.hidden_size,config,kind,adapter.device)
    optimizer=torch.optim.AdamW(branch.parameters(),lr=config.learning_rate,weight_decay=0.)
    history=[];updates=[];best=math.inf;best_epoch=0;best_state=None;visits=[torch.zeros(len(c.valid_mask),dtype=torch.long) for c in caches]
    def audit(epoch):
        nonlocal best,best_epoch,best_state
        branch.eval(); before=_tensor_hash(branch.state_dict())
        with torch.no_grad(): dev=_validate_development(development_evaluator(branch),contract); train=probe(adapter,caches,branch,work)
        if _tensor_hash(branch.state_dict())!=before or branch.training: raise ValueError('evaluation mutated branch')
        _assert_frozen(adapter)
        if dev['nll']<best:
            best=dev['nll'];best_epoch=epoch;best_state=copy.deepcopy(branch.state_dict())
            save_checkpoint(output/'checkpoint_best.pt',branch,identity,epoch)
        history.append(dict(epoch=epoch,update=len(updates),development=dev,train=train))
        _write_json(output/'history.json',history)
    try:
        audit(0)
        for epoch in range(config.epochs):
            branch.train()
            for ci,c in enumerate(caches):
                valid=torch.nonzero(c.valid_mask).flatten()
                # Every supervised position exactly once per epoch; no first-document bias.
                gen=torch.Generator().manual_seed(config.seed+epoch*len(caches)+ci)
                valid=valid[torch.randperm(len(valid),generator=gen)]
                for start in range(0,len(valid),config.positions_per_document):
                    positions=valid[start:start+config.positions_per_document]
                    optimizer.zero_grad(set_to_none=True)
                    ce,_=losses(adapter,c,branch,work=work)
                    loss=ce[positions.to(ce.device)].mean()
                    work['backward_attempts']=work.get('backward_attempts',0)+1
                    loss.backward()
                    work['backward_completed']=work.get('backward_completed',0)+1
                    norm=torch.nn.utils.clip_grad_norm_(branch.parameters(),config.gradient_clip,error_if_nonfinite=True)
                    work['optimizer_step_attempts']=work.get('optimizer_step_attempts',0)+1
                    optimizer.step()
                    if not all(bool(torch.isfinite(p).all()) for p in branch.parameters()): raise FloatingPointError('nonfinite parameters')
                    visits[ci][positions]+=1
                    updates.append(dict(update=len(updates)+1,epoch=epoch+1,document_id=c.document_id,positions=positions.tolist(),
                        ce=float(loss.detach()),gradient_norm=float(norm),learning_rate=config.learning_rate))
                    work['optimizer_updates']=work.get('optimizer_updates',0)+1
                    work['optimizer_step_completed']=work.get('optimizer_step_completed',0)+1
            audit(epoch+1)
        if _cache_binding(caches)!=binding: raise ValueError('training cache mutated')
        save_checkpoint(output/'checkpoint_latest.pt',branch,identity,config.epochs)
        branch.load_state_dict(best_state);branch.eval()
        summary=dict(status='complete',kind=kind,recurrent_ffn=kind=='r2',learned_stop=False,amplitude_gate=False,
            config=asdict(config),best_epoch=best_epoch,updates=len(updates),epochs_completed=config.epochs,
            context_visits=sum(int(v.sum()) for v in visits),min_visits=min(int(v[c.valid_mask].min()) for v,c in zip(visits,caches)),
            zero_gradient_updates=sum(u['gradient_norm']==0 for u in updates),best_development_nll=best,selected_train_probe=probe(adapter,caches,branch,work),
            stop_reason='fixed_epoch_budget_completed',wall_seconds=time.perf_counter()-started,work={k:v for k,v in work.items() if not k.startswith('_')},
            work_delta={k:v-work_initial.get(k,0) for k,v in work.items() if not k.startswith('_') and type(v) in (int,float)},
            work_scope='work is cumulative snapshot; work_delta is this branch fit including development/probes',cache_sha256=binding)
        _write_json(output/'optimization_history.json',updates);_write_json(output/'summary.json',summary)
        return branch,summary
    except Exception as exc:
        _write_json(output/'failure.json',dict(status='failed',type=type(exc).__name__,message=str(exc),updates=len(updates),work=work))
        _write_json(output/'optimization_history.json',updates)
        raise


def fit_gate(adapter,caches,branch,config,work=None):
    """Train-only CE advantage regression. Fixed branch, finite bounded budget."""
    config.validate();branch.eval();before=_tensor_hash(branch.state_dict())
    if branch.kind!='r2' or asdict(branch.config)!=asdict(config): raise ValueError('gate requires matching selected real-R2 branch')
    features=[];labels=[];records=[]
    with torch.no_grad():
        for c in caches:
            _validate_cache(adapter,c)
            base,_=losses(adapter,c,work=work);ce,_=losses(adapter,c,branch,work=work)
            mask=c.valid_mask.to(adapter.device)
            features.append(c.native_h1.reshape(-1,adapter.hidden_size)[mask].detach().cpu())
            labels.append((base-ce)[mask.to(ce.device)].detach().float().cpu())
            for pos in torch.nonzero(c.valid_mask).flatten().tolist():
                records.append(dict(document_id=c.document_id,position=pos,native_ce=float(base[pos]),forced_r2_ce=float(ce[pos]),gain=float(base[pos]-ce[pos])))
    x=torch.cat(features);y=torch.cat(labels)
    if not 1<=len(y)<=config.max_train_tokens: raise ValueError('gate train quota exceeded')
    gate=GainGate(adapter.hidden_size,adapter.device);opt=torch.optim.AdamW(gate.parameters(),lr=config.gate_learning_rate,weight_decay=0.)
    gen=torch.Generator().manual_seed(config.seed);history=[]
    with torch.no_grad(): initial_mse=float(F.mse_loss(gate(x.to(adapter.device)),y.to(adapter.device)))
    for step in range(config.gate_updates):
        check_deadline(work)
        indices=torch.randint(len(y),(min(config.gate_batch,len(y)),),generator=gen)
        opt.zero_grad(set_to_none=True);pred=gate(x[indices].to(adapter.device));target=y[indices].to(adapter.device)
        loss=F.mse_loss(pred,target)
        if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite gate loss')
        if work is not None: work['gate_backward_attempts']=work.get('gate_backward_attempts',0)+1
        loss.backward()
        if work is not None: work['gate_backward_completed']=work.get('gate_backward_completed',0)+1
        norm=torch.nn.utils.clip_grad_norm_(gate.parameters(),config.gradient_clip,error_if_nonfinite=True)
        if work is not None: work['gate_optimizer_attempts']=work.get('gate_optimizer_attempts',0)+1
        opt.step()
        if work is not None: work['gate_optimizer_completed']=work.get('gate_optimizer_completed',0)+1
        if not all(bool(torch.isfinite(p).all()) for p in gate.parameters()): raise FloatingPointError('nonfinite gate parameters')
        history.append(dict(update=step+1,mse=float(loss.detach()),gradient_norm=float(norm)))
    if _tensor_hash(branch.state_dict())!=before: raise ValueError('gate fit mutated frozen branch')
    gate.eval()
    with torch.no_grad():
        predicted=gate(x.to(adapter.device)).cpu(); predictions=predicted.tolist()
        final_mse=float(F.mse_loss(predicted,y))
    for record,prediction in zip(records,predictions): record['predicted_gain']=prediction
    return gate,dict(status='complete',updates=config.gate_updates,tokens=len(y),initial_train_mse=initial_mse,final_train_mse=final_mse,prediction_min=float(predicted.min()),prediction_max=float(predicted.max()),positive_prediction_fraction=float((predicted>0).double().mean()),labels_sha256=_tensor_hash({'labels':y}),
        feature_sha256=_tensor_hash({'h1':x}),per_token=records,label_scope='train only, native CE minus selected forced-R2 CE',history=history)


@torch.no_grad()
def verify_conditional(adapter,caches,branch,gate,threshold,config,work=None):
    records=[]
    for c in caches:
        mask=gate(c.native_h1).reshape(-1)>threshold
        compact,h=losses(adapter,c,branch,mask=mask,compact=True,work=work)
        dense,k=losses(adapter,c,branch,mask=mask,compact=False,work=work)
        metrics=dict(hidden=_comparison(h,k,config.parity_atol,config.parity_rtol),ce=_comparison(compact,dense,config.parity_atol,config.parity_rtol))
        records.append(dict(document_id=c.document_id,continued_tokens=int(mask.sum()),padded_tokens=len(mask),comparisons=metrics))
    return dict(status='complete',passed=all(m['mismatched_elements']==0 for r in records for m in r['comparisons'].values()),
        tolerances=dict(atol=config.parity_atol,rtol=config.parity_rtol),documents=records,
        scope='real compact conditional expert execution vs compute-both+mask; no GPU effect claim')


@torch.no_grad()
def verify_replay(adapter,caches,branch,config,work=None):
    """Fresh prefix vs cached real-R2 path and native untouched-model parity."""
    from .bridge import _native_document
    records=[]
    for c in caches:
        check_deadline(work)
        fresh=cache_documents(adapter,[dict(id=c.document_id,input_ids=c.input_ids,attention_mask=c.attention_mask)],max_cache_bytes=config.max_cache_bytes,max_logit_elements=config.max_logit_elements,work=work)[0]
        a,h=losses(adapter,c,branch,work=work);b,k=losses(adapter,fresh,branch,work=work)
        native_hidden,native_logits=_native_document(adapter,_inputs(adapter,c))
        native_ce=document_losses(native_logits,c);controlled_ce,_=losses(adapter,c,work=work)
        metrics=dict(h0=_comparison(c.h0,fresh.h0,config.parity_atol,config.parity_rtol),
            native_hidden=_comparison(c.native_h1,native_hidden,config.parity_atol,config.parity_rtol),
            native_ce=_comparison(controlled_ce,native_ce,config.parity_atol,config.parity_rtol),
            candidate_hidden=_comparison(h,k,config.parity_atol,config.parity_rtol),
            candidate_ce=_comparison(a,b,config.parity_atol,config.parity_rtol))
        records.append(dict(document_id=c.document_id,comparisons=metrics))
    return dict(status='complete',passed=all(m['mismatched_elements']==0 for r in records for m in r['comparisons'].values()),
        documents=records,tolerances=dict(atol=config.parity_atol,rtol=config.parity_rtol),
        scope='fresh full-document prefix/native reference and cached actual R2, not singleton replay')


@torch.no_grad()
def online_logits(adapter,row,branch=None,*,gate=None,threshold=None,mode='forced',config=None,work=None):
    """One real prefix/native R1 and one head, suitable for end-to-end timing."""
    config=config or (branch.config if branch else LoopConfig())
    check_deadline(work)
    c=cache_documents(adapter,[row],max_cache_bytes=config.max_cache_bytes,max_logit_elements=config.max_logit_elements,work=work)[0]
    n=c.native_h1.numel()//adapter.hidden_size
    if mode=='stop': mask=torch.zeros(n,dtype=torch.bool,device=adapter.device)
    elif mode=='forced': mask=torch.ones(n,dtype=torch.bool,device=adapter.device)
    elif mode=='gate':
        if gate is None or threshold is None: raise ValueError('frozen gate/threshold required')
        mask=gate(c.native_h1).reshape(-1)>threshold
    elif mode.startswith('fixed_hash_') and mode in ('fixed_hash_25','fixed_hash_50','fixed_hash_100'):
        fraction=int(mode.rsplit('_',1)[1])/100
        mask=torch.tensor([int(hashlib.sha256((str(config.seed)+':'+row['id']+':'+str(i)).encode()).hexdigest()[:16],16)/2**64 < fraction for i in range(n)],device=adapter.device)
    else: raise ValueError('unsupported online profiling mode')
    h=c.native_h1 if branch is None else branch(adapter,c.native_h1,mask,work=work)
    if work is not None: work['head_forward_attempts']=work.get('head_forward_attempts',0)+1
    logits=differentiable_readout(adapter,h)
    if work is not None: work['head_forward_completed']=work.get('head_forward_completed',0)+1
    return logits


def verify_backend_backward(adapter,cache,config,work=None):
    """One TRAIN-only actual-backend backward; no optimizer step or base mutation."""
    _validate_cache(adapter,cache);_assert_frozen(adapter)
    branch=LoopBranch(adapter.hidden_size,config,'r2',adapter.device)
    with torch.enable_grad():
        ce,_=losses(adapter,cache,branch,work=work)
        if work is not None: work['preflight_backward_attempts']=work.get('preflight_backward_attempts',0)+1
        ce[cache.valid_mask.to(ce.device)].mean().backward()
        if work is not None: work['preflight_backward_completed']=work.get('preflight_backward_completed',0)+1
    gradient=branch.correction.up.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()) or float(gradient.abs().sum())<=0:
        raise ValueError('actual expert backend failed finite nonzero input-adapter gradient preflight')
    _assert_frozen(adapter)
    return dict(status='complete',passed=True,document_id=cache.document_id,
        backend=getattr(adapter.model.config,'_experts_implementation',None),dtype=str(cache.native_h1.dtype),
        device=str(adapter.device),up_gradient_norm=float(gradient.norm()),optimizer_steps=0,
        backbone_gradients_absent=True,scope='actual configured backend, train-only backward capability; not fit/gain evidence')
