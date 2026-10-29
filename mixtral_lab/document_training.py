"""Full padded-document terminal-tail training, never singleton scoring.

Every FFN, residual and head sees [1,S,D] (FFN internally flattens all S rows).
Loss masks are applied ONLY after full-document logits. Cached prefix tensors
are detached; no transformer prefix or native weight is trained. Historical
singleton bridge results are independent diagnostics, not this protocol's gate.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import copy
import hashlib
import io
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .adapter_fit import _assert_frozen, _atomic_write, _json_hash, _tensor_hash
from .adaptive_fit import AdaptiveFitConfig, AdaptiveFitController
from .bridge import _native_document
from .recovery_adapter import _development_contract, _validate_development
from .second_round import SecondRoundResidual, differentiable_readout

FORMAT = 'mixtral-document-adapter-v1'
MAX_CHECKPOINT_BYTES = 160 * 1024 * 1024
SOURCE_FILES = ('document_training.py', 'adapter.py', 'second_round.py', 'adaptive_fit.py', 'adapter_fit.py', 'recovery_adapter.py', 'bridge.py')


@dataclass(frozen=True)
class DocumentAdapterConfig:
    rank: int = 8
    updates: int = 256
    positions_per_document: int = 32
    learning_rate: float = 1e-3
    gradient_clip: float = 1.
    residual_scale: float = .1
    native_kl_beta: float = .1
    relative_correction_l2: float = 0.
    eval_interval: int = 16
    development_patience: int = 4
    min_updates: int = 32
    development_min_delta: float = 1e-4
    seed: int = 17
    adaptive: dict | None = None
    max_cache_bytes: int = 1024**3
    max_logit_elements: int = 16_777_216
    parity_atol: float = 0.
    parity_rtol: float = 0.

    def validate(self):
        for name, low, high in (('rank',1,256), ('updates',1,10000),
                ('positions_per_document',1,4096), ('eval_interval',1,10000),
                ('development_patience',1,10000), ('min_updates',0,10000),
                ('seed',0,2**63-1), ('max_cache_bytes',1,16*1024**3),
                ('max_logit_elements',1,134217728)):
            _integer(getattr(self,name), name, low, high)
        if 1 + math.ceil(self.updates/self.eval_interval) > 1024:
            raise ValueError('at most 1024 development evaluations')
        for name, low, high, strict in (('learning_rate',0.,1.,True),
                ('gradient_clip',0.,1e6,True), ('residual_scale',0.,1.,True),
                ('native_kl_beta',0.,100.,False), ('relative_correction_l2',0.,100.,False),
                ('development_min_delta',0.,100.,False), ('parity_atol',0.,1.,False),
                ('parity_rtol',0.,1.,False)):
            _number(getattr(self,name),name,low,high,strict)
        adaptive = AdaptiveFitConfig.from_dict(self.adaptive)
        if adaptive.enabled and adaptive.eval_interval != self.eval_interval:
            raise ValueError('adaptive eval_interval must equal eval_interval')
        AdaptiveFitController(adaptive, initial_lr=self.learning_rate)


def _integer(value,name,low,high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be integer in [{low},{high}]')


def _number(value,name,low=0.,high=1e6,strict=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low <= value <= high or (strict and value == low):
        raise ValueError(f'invalid finite bounded {name}')


def _sources():
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCE_FILES}


@dataclass
class DocumentCache:
    document_id: str
    h0: torch.Tensor
    native_h1: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def _inputs(adapter, document):
    device = adapter.model.get_input_embeddings().weight.device
    return dict(input_ids=document.input_ids.to(device), attention_mask=document.attention_mask.to(device))


def _row(row):
    if not isinstance(row,dict) or not isinstance(row.get('id'),str) or not row['id']:
        raise ValueError('document requires nonempty id')
    ids,mask=row.get('input_ids'),row.get('attention_mask')
    if not isinstance(ids,torch.Tensor) or not isinstance(mask,torch.Tensor):
        raise ValueError('document tensors required')
    if ids.ndim==1: ids=ids.unsqueeze(0)
    if mask.ndim==1: mask=mask.unsqueeze(0)
    if ids.dtype!=torch.long or ids.ndim!=2 or ids.shape[0]!=1 or not 2<=ids.shape[1]<=4096 or mask.shape!=ids.shape:
        raise ValueError('expected one bounded padded document [1,S]')
    if bool(((mask!=0)&(mask!=1)).any()) or bool((ids<0).any()):
        raise ValueError('invalid input IDs or binary attention mask')
    valid=torch.zeros(ids.shape[1],dtype=torch.bool)
    valid[:-1]=(mask[0,:-1].cpu()!=0)&(mask[0,1:].cpu()!=0)
    if not bool(valid.any()): raise ValueError('document has no next-token targets')
    targets=torch.zeros(ids.shape[1],dtype=torch.long)
    targets[:-1]=ids[0,1:].cpu()
    return ids.detach().cpu().clone(),mask.detach().cpu().clone(),targets,valid


def _counted(work,name,call):
    if work is not None: work[name+'_attempted']=work.get(name+'_attempted',0)+1
    value=call()
    if work is not None: work[name+'_completed']=work.get(name+'_completed',0)+1
    return value


def cache_documents(adapter,rows,*,max_cache_bytes=1024**3,max_logit_elements=16_777_216,work=None):
    """Bound allocation before capturing; retain full h0 and native h1 only."""
    _assert_frozen(adapter)
    _integer(max_cache_bytes,'max_cache_bytes',1,16*1024**3)
    _integer(max_logit_elements,'max_logit_elements',1,134217728)
    if not isinstance(rows,list) or not 1<=len(rows)<=256: raise ValueError('1..256 documents required')
    seen=set(); prepared=[]; required=0
    dtype=next(adapter.layer.post_attention_layernorm.parameters()).dtype
    if dtype not in (torch.float32,torch.bfloat16): raise ValueError('FP32/BF16 backbone required')
    vocab=int(adapter.model.config.vocab_size)
    for row in rows:
        ids,mask,targets,valid=_row(row)
        if row['id'] in seen: raise ValueError('duplicate document id')
        seen.add(row['id'])
        if int(ids.max())>=vocab: raise ValueError('token outside vocabulary')
        if ids.numel()*vocab>max_logit_elements: raise ValueError('full-document logits exceed allocation bound')
        required+=2*ids.numel()*adapter.hidden_size*torch.empty((),dtype=dtype).element_size()+ids.numel()*25
        if required>max_cache_bytes: raise ValueError('full-document cache exceeds allocation bound')
        prepared.append(DocumentCache(row['id'],None,None,targets,valid,ids,mask))
    with torch.no_grad():
        for cache in prepared:
            cache.h0=_counted(work,'prefix_capture',lambda:adapter.capture(**_inputs(adapter,cache))).detach().clone()
            cache.native_h1=_counted(work,'native_terminal_no_head',lambda:_native_tail_hidden(adapter,cache.h0)).detach().clone()
            _validate_cache(adapter,cache)
    return prepared


@torch.no_grad()
def _native_tail_hidden(adapter,h0):
    flat=adapter._flat(h0)
    n=adapter.layer.post_attention_layernorm(flat)
    _,weights,pairs=adapter.moe.gate(n)
    return (flat+adapter.moe.experts(n,pairs,weights)).reshape(h0.shape)


def _validate_cache(adapter,cache):
    if not isinstance(cache,DocumentCache): raise ValueError('DocumentCache required')
    shape=(*cache.input_ids.shape,adapter.hidden_size)
    for name in ('h0','native_h1'):
        tensor=getattr(cache,name)
        if tensor.shape!=shape or tensor.device!=adapter.device or tensor.requires_grad or not bool(torch.isfinite(tensor).all()):
            raise ValueError('invalid full-document '+name)
    if cache.h0.dtype!=cache.native_h1.dtype: raise ValueError('cache dtype mismatch')
    _,_,targets,valid=_row(dict(id=cache.document_id,input_ids=cache.input_ids,attention_mask=cache.attention_mask))
    if not torch.equal(targets,cache.targets.cpu()) or not torch.equal(valid,cache.valid_mask.cpu()):
        raise ValueError('cache next-token target/mask mismatch')


def document_forward(adapter,cache,candidate=None,kind='adapter'):
    """Full shape shared by training, cached probes and fresh-prefix evaluation."""
    if kind=='router':
        with torch.no_grad():
            out=adapter._forward_r1(cache.h0,candidate)
        return out.logits,out.final_hidden
    if kind!='adapter': raise ValueError('unknown document candidate kind')
    hidden=cache.native_h1 if candidate is None else candidate(cache.native_h1)
    # Deliberately differentiable; unlike adapter.readout this preserves gradients.
    return differentiable_readout(adapter,hidden),hidden


def document_losses(logits,cache):
    if logits.ndim!=3 or logits.shape[:2]!=cache.input_ids.shape: raise ValueError('full document logits required')
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),cache.targets.to(logits.device),reduction='none')


def _comparison(a,b,atol,rtol):
    if a.shape!=b.shape: raise ValueError('parity tensor shape mismatch')
    a,b=a.detach().double().cpu(),b.detach().double().cpu()
    if not bool(torch.isfinite(a).all()&torch.isfinite(b).all()): raise ValueError('nonfinite parity input')
    delta=(a-b).abs(); excess=(delta-(atol+rtol*b.abs())).clamp_min(0.)
    return dict(elements=a.numel(),mismatched_elements=int((excess>0).sum()),max_abs=float(delta.max()) if a.numel() else 0.,max_excess=float(excess.max()) if a.numel() else 0.)


@torch.no_grad()
def verify_document_replay(adapter,caches,candidate=None,kind='adapter',atol=0.,rtol=0.):
    """Same-protocol correctness, not old singleton/full numerical screening."""
    _number(atol,'atol',0.,1.); _number(rtol,'rtol',0.,1.)
    comparisons=[]
    for cache in caches:
        _validate_cache(adapter,cache)
        fresh_h0=adapter.capture(**_inputs(adapter,cache))
        fresh_native=_native_tail_hidden(adapter,fresh_h0)
        fresh=DocumentCache(cache.document_id,fresh_h0,fresh_native,cache.targets,cache.valid_mask,cache.input_ids,cache.attention_mask)
        native_hidden,native_logits=_native_document(adapter,_inputs(adapter,cache))
        controlled_logits,controlled_hidden=document_forward(adapter,cache,None,kind)
        cached_logits,cached_hidden=document_forward(adapter,cache,candidate,kind)
        full_logits,full_hidden=document_forward(adapter,fresh,candidate,kind)
        # Zero residual identity and native routing control share the full-tail path.
        if kind=='adapter':
            zero_logits,zero_hidden=document_forward(adapter,cache,lambda h:h+torch.zeros_like(h),kind)
        else:
            from .adapter import pair_to_action
            zero_logits,zero_hidden=document_forward(adapter,cache,lambda obs:pair_to_action(obs.native_pairs),kind)
        pairs={'native_vs_controlled':(native_logits,controlled_logits,native_hidden,controlled_hidden),
               'zero_vs_native':(controlled_logits,zero_logits,controlled_hidden,zero_hidden),
               'cached_vs_full':(full_logits,cached_logits,full_hidden,cached_hidden)}
        metrics={}
        for name,(a,b,h,k) in pairs.items():
            metrics[name+'.logits']=_comparison(a,b,atol,rtol)
            metrics[name+'.hidden']=_comparison(h,k,atol,rtol)
            metrics[name+'.ce']=_comparison(document_losses(a,cache),document_losses(b,cache),atol,rtol)
        metrics['cached_vs_full.h0']=_comparison(fresh_h0,cache.h0,atol,rtol)
        if kind=='adapter':
            probe=SecondRoundResidual(adapter.hidden_size,DocumentAdapterConfig(rank=min(2,adapter.hidden_size)),adapter.device)
            probe.up.fill_(1.)
        else:
            def probe(obs):
                from .adapter import pair_to_action
                native=pair_to_action(obs.native_pairs)
                return native.remainder(28)+1
        a,h=document_forward(adapter,cache,probe,kind)
        b,k=document_forward(adapter,fresh,probe,kind)
        changed=int((h!=cache.native_h1).sum())
        probe_record=dict(effective_changed_elements=changed,action_change_count=cache.input_ids.numel() if kind=='router' else None,
            passed=(changed>0 if kind=='adapter' else True),
            state_sha256=_tensor_hash(probe.state_dict()) if kind=='adapter' else None)
        metrics['probe_cached_vs_full.logits']=_comparison(a,b,atol,rtol)
        metrics['probe_cached_vs_full.hidden']=_comparison(h,k,atol,rtol)
        metrics['probe_cached_vs_full.ce']=_comparison(document_losses(a,cache),document_losses(b,cache),atol,rtol)
        comparisons.append(dict(document_id=cache.document_id,comparisons=metrics,nonzero_probe=probe_record))
    passed=all(m['mismatched_elements']==0 for d in comparisons for m in d['comparisons'].values()) and all(d['nonzero_probe']['passed'] for d in comparisons)
    return dict(status='complete',passed=passed,protocol='full_padded_document_tail_v1',
        tolerances=dict(atol=atol,rtol=rtol),documents=comparisons,probe='deterministic_nonzero_adapter_or_shifted_native_pair',
        candidate_checked=candidate is not None,kind=kind,old_singleton_gate_used=False)


@torch.no_grad()
def online_logits(adapter,row,candidate=None,kind='adapter',max_logit_elements=16_777_216):
    ids,mask,targets,valid=_row(row)
    if ids.numel()*int(adapter.model.config.vocab_size)>max_logit_elements: raise ValueError('logits exceed allocation bound')
    cache=DocumentCache(row['id'],None,None,targets,valid,ids,mask)
    cache.h0=adapter.capture(**_inputs(adapter,cache))
    if kind=='adapter': cache.native_h1=_native_tail_hidden(adapter,cache.h0)
    return document_forward(adapter,cache,candidate,kind)[0]


@torch.no_grad()
def evaluate_documents(adapter,rows,candidate=None,kind='adapter',max_logit_elements=16_777_216,role=None,work=None):
    """Fresh prefix and full-tail paired metrics, never selected-row head."""
    per_document=[]
    for row in rows:
        cache=cache_documents(adapter,[row],max_cache_bytes=16*1024**3,max_logit_elements=max_logit_elements,work=work)[0]
        _,native=_counted(work,'native_full_model',lambda:_native_document(adapter,_inputs(adapter,cache)))
        controlled,_=_counted(work,'controlled_tail_with_head',lambda:document_forward(adapter,cache,None,kind))
        logits,_=_counted(work,'candidate_tail_with_head',lambda:document_forward(adapter,cache,candidate,kind))
        metrics={}
        for name,value in (('native',native),('controlled',controlled),('candidate',logits)):
            values=document_losses(value,cache)[cache.valid_mask.to(value.device)]
            metrics[name+'_nll']=float(values.double().mean())
        per_document.append(dict(document_id=cache.document_id,tokens=len(values),**metrics,
            candidate_minus_native_nll=metrics['candidate_nll']-metrics['native_nll'],
            candidate_minus_controlled_nll=metrics['candidate_nll']-metrics['controlled_nll']))
    tokens=sum(d['tokens'] for d in per_document)
    if not tokens: raise ValueError('no evaluation tokens')
    aggregates={key:dict(nll=sum(d[name+'_nll']*d['tokens'] for d in per_document)/tokens,tokens=tokens,sequences=len(per_document))
        for name,key in (('native','native'),('controlled','controlled_original'),('candidate','candidate'))}
    return dict(status='ok',role=role,document_ids=[d['document_id'] for d in per_document],
        nll=aggregates['candidate']['nll'],tokens=tokens,sequences=len(per_document),**aggregates,
        per_document=per_document,protocol='full_document_online_nll')


def _cache_binding(caches):
    return _json_hash([dict(document_id=c.document_id,tensors=_tensor_hash({k:getattr(c,k) for k in ('h0','native_h1','targets','valid_mask','input_ids','attention_mask')})) for c in caches])


def _loss_components(adapter,module,cache,config):
    teacher_logits,_=document_forward(adapter,cache,None)
    with torch.no_grad(): teacher_logp=F.log_softmax(teacher_logits.float(),-1).detach()
    logits,corrected=document_forward(adapter,cache,module)
    ce=document_losses(logits,cache)
    kl=F.kl_div(F.log_softmax(logits.float(),-1),teacher_logp,reduction='none',log_target=True).sum(-1).reshape(-1)
    proposed=module.correction(cache.native_h1)
    relative=proposed.float().square().sum(-1).reshape(-1)/cache.native_h1.float().square().sum(-1).reshape(-1).clamp_min(1e-12)
    loss=ce+config.native_kl_beta*kl+config.relative_correction_l2*relative.to(ce.device)
    if not all(bool(torch.isfinite(x).all()) for x in (loss,corrected,proposed)): raise FloatingPointError('nonfinite document loss/correction')
    return dict(loss=loss,ce=ce,native_ce=document_losses(teacher_logits,cache).detach(),native_kl=kl,relative_correction_l2=relative,corrected=corrected)


def _probe(adapter,module,caches,config,visits,work=None):
    totals=dict(loss=0.,ce=0.,native_kl=0.,relative_correction_l2=0.);count=0
    selected_sum=unselected_sum=0.;selected_n=unselected_n=0;changed=elements=0
    per_document=[];anchor_sum=anchor_native=0.;anchor_n=0;never_native=seen_native=0.
    with torch.no_grad():
        for cache,seen in zip(caches,visits):
            if work is not None:
                work['train_probe_documents']+=1;work['teacher_full_document_forwards']+=1
            parts=_loss_components(adapter,module,cache,config);mask=cache.valid_mask
            if work is not None: work['completed_train_probe_documents']+=1
            n=int(mask.sum());count+=n
            for key in totals: totals[key]+=float(parts[key][mask.to(parts[key].device)].double().sum())
            values=parts['ce'].detach().cpu();seen_mask=torch.tensor(seen)>0
            sel=mask&seen_mask;unsel=mask&~seen_mask
            selected_sum+=float(values[sel].double().sum());selected_n+=int(sel.sum())
            native=parts['native_ce'].cpu()
            seen_native+=float(native[sel].double().sum());never_native+=float(native[unsel].double().sum())
            anchor=torch.nonzero(mask,as_tuple=False).flatten()[:config.positions_per_document]
            anchor_sum+=float(values[anchor].double().sum());anchor_native+=float(native[anchor].double().sum());anchor_n+=len(anchor)
            unselected_sum+=float(values[unsel].double().sum());unselected_n+=int(unsel.sum())
            delta=parts['corrected'].detach()-cache.native_h1
            changed+=int((delta!=0).sum());elements+=delta.numel()
            per_document.append(dict(document_id=cache.document_id,tokens=n,nll=float(values[mask].double().mean())))
    return {**{k:v/count for k,v in totals.items()},'nll':totals['ce']/count,'tokens':count,
        'selected':dict(tokens=selected_n,nll=selected_sum/selected_n if selected_n else None),
        'unselected':dict(tokens=unselected_n,nll=unselected_sum/unselected_n if unselected_n else None),
        'actual_hidden_change_rate':changed/elements,'per_document':per_document,
        'population_semantics':'selected = ever CE-supervised; unselected = never CE-supervised; all valid positions receive native KL/L2 on document visits',
        'selected_gain_vs_native':(seen_native-selected_sum)/selected_n if selected_n else None,
        'unselected_gain_vs_native':(never_native-unselected_sum)/unselected_n if unselected_n else None,
        'fixed_anchor':dict(rule='first positions_per_document valid positions per document, diagnostic only',tokens=anchor_n,nll=anchor_sum/anchor_n,gain_vs_native=(anchor_native-anchor_sum)/anchor_n)}


def _dev(adapter,module,evaluator,contract,caches,binding):
    module.eval();state=_tensor_hash(module.state_dict());cfg=asdict(module.config);hidden=module.hidden_size
    model_versions=[(id(p),p._version,p.requires_grad) for p in adapter.model.parameters()]
    with torch.no_grad(): result=evaluator(module)
    _assert_frozen(adapter)
    if model_versions!=[(id(p),p._version,p.requires_grad) for p in adapter.model.parameters()]: raise ValueError('development callback mutated backbone')
    if module.training or module.hidden_size!=hidden or not all(p.requires_grad for p in module.parameters()) or _tensor_hash(module.state_dict())!=state or asdict(module.config)!=cfg or getattr(evaluator,'contract',None)!=contract:
        raise ValueError('development callback mutated candidate or contract')
    if _cache_binding(caches)!=binding: raise ValueError('development callback mutated training cache')
    return _validate_development(result,contract)


def _save(path,module,identities,binding,update,probe,development,source,status):
    state={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
    data=dict(format=FORMAT,config=asdict(module.config),hidden_size=module.hidden_size,
        identities=copy.deepcopy(identities),binding_sha256=_json_hash(identities),cache_sha256=binding,
        update=update,train_probe=probe,development=development,development_metrics=development,
        source_file_sha256=source,state_sha256=_tensor_hash(state),status=status,inference_only=True,
        checkpoint_selection='minimum_development_full_document_nll')
    data['metadata_sha256']=_json_hash(data);data['state']=state
    _atomic_write(path,lambda tmp:torch.save(data,tmp))


def load_document_adapter_checkpoint(path,identities,device='cpu',expected_config=None,expected_hidden_size=None):
    path=Path(path)
    if (path.parent/'failure.json').exists(): raise ValueError('checkpoint belongs to failed document training run')
    if path.stat().st_size>MAX_CHECKPOINT_BYTES: raise ValueError('checkpoint exceeds bound')
    payload=path.read_bytes()
    data=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    expected_keys={'format','config','hidden_size','identities','binding_sha256','cache_sha256','update','train_probe','development','development_metrics','source_file_sha256','state_sha256','status','inference_only','checkpoint_selection','metadata_sha256','state'}
    if len(payload)>MAX_CHECKPOINT_BYTES or not isinstance(data,dict) or set(data)!=expected_keys: raise ValueError('invalid document checkpoint')
    state=data.pop('state',None);digest=data.pop('metadata_sha256',None)
    if digest!=_json_hash(data) or not isinstance(state,dict) or data.get('state_sha256')!=_tensor_hash(state): raise ValueError('checkpoint integrity mismatch')
    if data.get('format')!=FORMAT or data.get('inference_only') is not True or data.get('identities')!=identities or data.get('binding_sha256')!=_json_hash(identities): raise ValueError('checkpoint identity mismatch')
    if data.get('source_file_sha256')!=_sources(): raise ValueError('checkpoint source mismatch')
    if data.get('status') not in ('evaluated','completed'): raise ValueError('checkpoint not eligible')
    config=DocumentAdapterConfig(**data['config']);config.validate()
    if expected_config is not None and config!=expected_config: raise ValueError('checkpoint config mismatch')
    hidden=data.get('hidden_size');_integer(hidden,'hidden_size',config.rank,65536)
    if expected_hidden_size is not None and hidden!=expected_hidden_size: raise ValueError('hidden size mismatch')
    _integer(data.get('update'),'update',0,config.updates)
    _validate_development(data.get('development'),_development_contract(identities))
    if data['development_metrics']!=data['development'] or data['checkpoint_selection']!='minimum_development_full_document_nll': raise ValueError('checkpoint development metadata mismatch')
    module=SecondRoundResidual(hidden,config,device)
    for tensor in state.values():
        if not isinstance(tensor,torch.Tensor) or tensor.dtype!=torch.float32 or not bool(torch.isfinite(tensor).all()): raise ValueError('invalid state tensor')
    module.load_state_dict(state,strict=True);module.requires_grad_(False).eval()
    return module,{**data,'metadata_sha256':digest,'checkpoint_sha256':hashlib.sha256(payload).hexdigest()}


def train_document_adapter(adapter,caches,development_evaluator,config,identities,output_dir):
    """One full document/update, rotating supervised positions, dev-only best."""
    config.validate();_assert_frozen(adapter)
    contract=_development_contract(identities,development_evaluator)
    if [c.document_id for c in caches]!=identities['train']['selected_document_ids']: raise ValueError('training cache/document identity mismatch')
    required=0
    for cache in caches:
        _validate_cache(adapter,cache)
        required+=sum(getattr(cache,k).numel()*getattr(cache,k).element_size() for k in ('h0','native_h1','targets','valid_mask','input_ids','attention_mask'))
        if required>config.max_cache_bytes or cache.input_ids.numel()*int(adapter.model.config.vocab_size)>config.max_logit_elements:
            raise ValueError('training cache/logits exceed configured allocation bound')
    binding=_cache_binding(caches);source=_sources()
    module=SecondRoundResidual(adapter.hidden_size,config,adapter.device)
    parity=verify_document_replay(adapter,caches,module,'adapter',config.parity_atol,config.parity_rtol)
    if not parity['passed']: raise ValueError('full-document zero/cache correctness failed')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    work=dict(optimizer_documents=0,completed_optimizer_documents=0,teacher_full_document_forwards=0,train_probe_documents=0,completed_train_probe_documents=0,development_evaluations=0,completed_development_evaluations=0,counts_are_attempted_unless_completed=True,parity_and_cache_build_excluded=True)
    try:
        return _train(adapter,caches,development_evaluator,config,identities,output,module,contract,binding,source,parity,work)
    except Exception as exc:
        _atomic_write(output/'failure.json',lambda p:p.write_text(json.dumps(dict(status='failed',error=type(exc).__name__+': '+str(exc),work_accounting=work),indent=2,allow_nan=False)))
        raise


def _train(adapter,caches,evaluator,config,identities,output,module,contract,binding,source,parity,work):
    optimizer=torch.optim.AdamW(module.parameters(),lr=config.learning_rate,weight_decay=0.)
    controller=AdaptiveFitController(AdaptiveFitConfig.from_dict(config.adaptive),initial_lr=config.learning_rate)
    rng=torch.Generator(device='cpu').manual_seed(config.seed)
    visits=[[0]*c.input_ids.numel() for c in caches]
    orders=[[] for c in caches];cursors=[0]*len(caches);doc_order=[];doc_cursor=0
    initial_state=copy.deepcopy(module.state_dict())
    def probe():
        return _probe(adapter,module,caches,config,visits,work)
    def dev():
        work['development_evaluations']+=1
        result=_dev(adapter,module,evaluator,contract,caches,binding)
        work['completed_development_evaluations']+=1
        return result
    module.eval();initial_probe=probe();initial_dev=dev()
    controller.observe(0,initial_probe['loss'],coverage_complete=False)
    best_probe=latest_probe=initial_probe;best_dev=latest_dev=initial_dev
    best_update=latest_update=0;best_state=copy.deepcopy(initial_state)
    def save(name,update,p,d,status='evaluated'): _save(output/name,module,identities,binding,update,p,d,source,status)
    for name in ('checkpoint_initial.pt','checkpoint_best.pt','checkpoint_eval_000000.pt'): save(name,0,initial_probe,initial_dev)
    history=[dict(update=0,train_probe=initial_probe,development=initial_dev,checkpoint='checkpoint_eval_000000.pt')]
    rows=[];stale=0;anchor=initial_dev['nll'];window_nonzero=0;stop_reason='update_budget_exhausted'
    for update in range(1,config.updates+1):
        if doc_cursor==len(doc_order): doc_order=torch.randperm(len(caches),generator=rng).tolist();doc_cursor=0
        d=doc_order[doc_cursor];doc_cursor+=1;cache=caches[d]
        if cursors[d]==len(orders[d]):
            valid=torch.nonzero(cache.valid_mask.cpu(),as_tuple=False).flatten()
            orders[d]=valid[torch.randperm(len(valid),generator=rng)].tolist();cursors[d]=0
        positions=orders[d][cursors[d]:cursors[d]+config.positions_per_document];cursors[d]+=len(positions)
        module.train();optimizer.zero_grad(set_to_none=True)
        work['optimizer_documents']+=1;work['teacher_full_document_forwards']+=1
        part=_loss_components(adapter,module,cache,config)
        idx=torch.tensor(positions,device=part['loss'].device)
        valid=cache.valid_mask.to(part['ce'].device)
        # Supervised CE rotates; preservation anchors ALL valid positions.
        loss=(part['ce'].index_select(0,idx).mean()+config.native_kl_beta*part['native_kl'][valid].mean()
            +config.relative_correction_l2*part['relative_correction_l2'][cache.valid_mask.to(part['relative_correction_l2'].device)].mean().to(part['ce'].device))
        loss.backward();_assert_frozen(adapter)
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in module.parameters()): raise FloatingPointError('invalid adapter gradient')
        grad=torch.nn.utils.clip_grad_norm_(module.parameters(),config.gradient_clip)
        if not bool(torch.isfinite(grad)): raise FloatingPointError('nonfinite gradient norm')
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in module.parameters()): raise FloatingPointError('nonfinite adapter parameters')
        work['completed_optimizer_documents']+=1
        for p in positions: visits[d][p]+=1
        window_nonzero+=int(float(grad)>0)
        rows.append(dict(update=update,document_id=cache.document_id,position_count=len(positions),positions_preview=positions[:32],positions_sha256=_json_hash(positions),loss=float(loss.detach()),gradient_norm=float(grad),learning_rate=optimizer.param_groups[0]['lr']))
        del part,loss
        if update%config.eval_interval==0 or update==config.updates:
            module.eval();latest_probe=probe();latest_dev=dev();latest_update=update
            covered=all(all(v>0 for v,m in zip(seen,c.valid_mask.tolist()) if m) for seen,c in zip(visits,caches))
            decision=controller.observe(update,latest_probe['loss'],covered,caches[0].h0.dtype==torch.bfloat16 and latest_probe['actual_hidden_change_rate']==0. and window_nonzero>0)
            window_nonzero=0
            for g in optimizer.param_groups:g['lr']=decision['lr']
            if latest_dev['nll']<anchor-config.development_min_delta:anchor=latest_dev['nll'];stale=0
            else:stale+=1
            file=f'checkpoint_eval_{update:06d}.pt';save(file,update,latest_probe,latest_dev)
            history.append(dict(update=update,train_probe=latest_probe,development=latest_dev,adaptive_decision=decision,checkpoint=file))
            if latest_dev['nll']<best_dev['nll']:
                best_update=update;best_dev=copy.deepcopy(latest_dev);best_probe=copy.deepcopy(latest_probe);best_state=copy.deepcopy(module.state_dict());save('checkpoint_best.pt',update,best_probe,best_dev)
            if covered and update>=config.min_updates and stale>=config.development_patience:stop_reason='development_patience_exhausted';break
            if decision['stop']:stop_reason='adaptive_train_plateau';break
    save('checkpoint_latest.pt',latest_update,latest_probe,latest_dev,'completed')
    module.load_state_dict(best_state)
    learned_parity=verify_document_replay(adapter,caches,module,'adapter',config.parity_atol,config.parity_rtol)
    if not learned_parity['passed']: raise ValueError('selected candidate full-document replay failed')
    save('checkpoint_best.pt',best_update,best_probe,best_dev,'completed')
    module.zero_grad(set_to_none=True);module.requires_grad_(False).eval();_assert_frozen(adapter)
    summary=dict(format=FORMAT,status='completed',stop_reason=stop_reason,config=asdict(config),
        initial_development_nll=initial_dev['nll'],best_development_nll=best_dev['nll'],latest_development_nll=latest_dev['nll'],
        best_update=best_update,latest_update=latest_update,initial_development_metrics=initial_dev,best_development_metrics=best_dev,latest_development_metrics=latest_dev,
        initial_train_probe=initial_probe,best_train_probe=best_probe,latest_train_probe=latest_probe,evaluation_history=history,updates=rows,
        successful_updates=len(rows),context_visits=sum(sum(v) for v in visits),unique_contexts_visited=sum(sum(x>0 for x in v) for v in visits),
        train_contexts=sum(int(c.valid_mask.sum()) for c in caches),context_visit_counts=visits,
        coverage_complete=all(all(v>0 for v,m in zip(seen,c.valid_mask.tolist()) if m) for seen,c in zip(visits,caches)),
        checkpoint_selection='minimum_development_full_document_nll',development_improved=best_dev['nll']<initial_dev['nll'],
        cache_sha256=binding,source_file_sha256=source,work_accounting=work,initial_parity=parity,selected_parity=learned_parity,
        adaptive_controller_state=controller.snapshot(),selection_used=False,heldout_used=False,practical_gain_established=False,
        added_capacity=True,recurrent_ffn=False,learned_stop=False,
        limits=['development is not final test','one full padded document per optimizer update','CE masks rotate; native KL and correction L2 anchor all valid positions; never-supervised counts may reach zero','native KL and residual scale are provisional settings'])
    _atomic_write(output/'summary.json',lambda p:p.write_text(json.dumps(summary,indent=2,allow_nan=False)))
    return module,summary
