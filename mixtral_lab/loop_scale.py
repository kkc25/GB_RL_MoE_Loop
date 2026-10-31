"""Bounded token-local R2 learning curves with durable update transactions.

Only detached native terminal states are cached. Expert/head batches contain
supervised positions; no attention is recomputed on these selected positions.
This is a new numerical protocol and its GPU compaction parity must be checked.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .adapter_fit import _assert_frozen, _atomic_write, _tensor_hash
from .document_training import _integer, _number
from .second_round import differentiable_readout
from .trainable_loop import LoopBranch, LoopConfig, GainGate

FORMAT = 'mixtral-loop-scale-v2'


def digest(value):
    """Content digest including tensor bytes, shape and dtype, independent of pickle."""
    def encode(x):
        if isinstance(x, torch.Tensor):
            return {'tensor_shape': list(x.shape), 'dtype': str(x.dtype),
                    'sha256': hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        if isinstance(x, dict):
            return {str(k): encode(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [encode(v) for v in x]
        if isinstance(x, Path):
            return str(x)
        return x
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, lambda p: p.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                                       allow_nan=False) + '\n'))


def _save(path, value):
    payload = {'format': FORMAT, 'payload': value, 'sha256': digest(value)}
    _atomic_write(Path(path), lambda p: torch.save(payload, p))


def _load(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('format') != FORMAT or data.get('sha256') != digest(data.get('payload')):
        raise ValueError('checkpoint content digest mismatch')
    return data['payload']


def _cpu(state):
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().clone()
    if isinstance(state, dict):
        return {k: _cpu(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_cpu(v) for v in state]
    if isinstance(state, tuple):
        return tuple(_cpu(v) for v in state)
    return copy.deepcopy(state)


@dataclass(frozen=True)
class ScaleConfig:
    rank: int = 8
    epochs: int = 2
    positions_per_update: int = 256
    learning_rate: float = 1e-3
    gradient_clip: float = 1.
    seed: int = 17
    max_updates: int = 100000
    max_wall_seconds: float = 43200.
    absolute_deadline_utc: float | None = None
    gate_epochs: int = 1
    gate_learning_rate: float = 1e-3
    thresholds: tuple = (0., .001, .005, .01, .02, .05)
    target_exposures: int | None = None
    gate_target_exposures: int | None = None
    max_logit_elements: int = 16_777_216
    eligibility_gain: float = .01
    eligibility_fraction: float = .05
    r2_precision: str = 'native_bf16'

    def validate(self):
        if self.r2_precision not in ('native_bf16', 'fp32_experts', 'fp32_canonical32'):
            raise ValueError('unknown R2 precision protocol')
        for name, low, high in (('rank', 1, 256), ('epochs', 1, 2),
                ('positions_per_update', 1, 4096), ('seed', 0, 2**63-1),
                ('max_updates', 1, 1000000), ('gate_epochs', 1, 2),
                ('max_logit_elements', 1, 134217728)):
            _integer(getattr(self, name), name, low, high)
        for name in ('learning_rate', 'gate_learning_rate', 'gradient_clip'):
            _number(getattr(self, name), name, 0., 1., True)
        _number(self.max_wall_seconds, 'max_wall_seconds', 0., 43200., True)
        if self.absolute_deadline_utc is not None:
            _number(self.absolute_deadline_utc,'absolute_deadline_utc',0.,1e12,True)
        _number(self.eligibility_gain, 'eligibility_gain', 0., 1.)
        _number(self.eligibility_fraction, 'eligibility_fraction', 0., 1.)
        for name in ('target_exposures','gate_target_exposures'):
            if getattr(self,name) is not None:
                _integer(getattr(self,name),name,1,100000000)
        if not isinstance(self.thresholds, (list, tuple)) or not self.thresholds:
            raise ValueError('frozen threshold grid required')
        for threshold in self.thresholds:
            _number(threshold, 'threshold', 0., 1.)
        if list(self.thresholds) != sorted(set(self.thresholds)) or len(self.thresholds)>16:
            raise ValueError('threshold grid must be sorted, unique and bounded')

    def loop_config(self):
        self.validate()
        return LoopConfig(rank=self.rank, epochs=self.epochs, seed=self.seed)


class ScaleBranch(LoopBranch):
    """Real frozen expert R2, with a predeclared fixed residual amplitude."""
    def __init__(self, hidden_size, config, kind='r2', amplitude=.1, device='cpu'):
        config.validate()
        if amplitude not in (.1, 1.) or isinstance(amplitude, bool):
            raise ValueError('only frozen amplitudes 0.1 and 1.0 are supported')
        if kind == 'parameter_control' and amplitude != 1.:
            raise ValueError('parameter control has fixed amplitude 1')
        super().__init__(hidden_size, config.loop_config(), kind, device)
        self.scale_config = config
        self.amplitude = float(amplitude)

    def forward(self, adapter, h1, mask=None, *, compact=True, work=None, trace=None):
        _check_config_deadline(self.scale_config)
        _assert_frozen(adapter)
        if h1.requires_grad or h1.shape[-1] != self.hidden_size:
            raise ValueError('matching detached native terminal state required')
        flat = h1.reshape(-1, self.hidden_size)
        if mask is None:
            mask = torch.ones(len(flat), dtype=torch.bool, device=flat.device)
        if mask.dtype != torch.bool or mask.shape != (len(flat),) or mask.device != flat.device:
            raise ValueError('flat boolean mask on terminal device required')
        selected = mask.nonzero().flatten()
        if not len(selected):
            return h1.clone()
        active = flat[selected] if compact and len(selected) != len(flat) else flat
        from .canonical_execution import MODE, norm_and_correction, route, observe
        canonical = self.scale_config.r2_precision == MODE
        if canonical:
            x,correction=norm_and_correction(adapter,self,active,work)
        else:
            x=adapter.layer.post_attention_layernorm(active)
            if self.kind=='r2' and self.scale_config.r2_precision=='fp32_experts':
                from .fp32_experts import correction_fp32
                correction=correction_fp32(self.correction,x,deadline=self.scale_config.absolute_deadline_utc)
            else:correction=self.correction(x)
        observe(trace,'normalized_input',x);observe(trace,'correction_fp32',correction)
        if self.kind == 'r2':
            with torch.no_grad():
                if canonical:logits,weights,pairs=route(adapter,x,deadline=self.scale_config.absolute_deadline_utc,work=work)
                else:logits,weights,pairs=adapter.moe.gate(x)
            observe(trace,'router_logits',logits);observe(trace,'router_weights',weights);observe(trace,'router_pairs',pairs)
            if work is not None:
                work['extra_expert_forward_attempts'] = work.get('extra_expert_forward_attempts', 0) + 1
                work['extra_expert_token_calls'] = work.get('extra_expert_token_calls', 0) + 2*len(active)
            expert_input=(x.float()+correction) if self.scale_config.r2_precision!='native_bf16' else (x.float()+correction).to(x.dtype)
            observe(trace,'expert_input',expert_input)
            if canonical:
                from .canonical_execution import experts
                update=experts(adapter,expert_input,pairs.detach(),weights.detach(),
                    deadline=self.scale_config.absolute_deadline_utc,work=work)
            elif self.scale_config.r2_precision == 'fp32_experts':
                from .fp32_experts import experts_fp32
                update = experts_fp32(adapter.moe.experts, expert_input,
                    pairs.detach(), weights.detach(), deadline=self.scale_config.absolute_deadline_utc, work=work)
            else:
                update = adapter.moe.experts(expert_input, pairs.detach(), weights.detach())
            observe(trace,'expert_output',update)
            if work is not None:
                work['extra_expert_forward_completed'] = work.get('extra_expert_forward_completed', 0) + 1
        else:
            update = correction
        # Apply amplitude BEFORE the residual addition and BF16 rounding.
        residual=active.float()+self.amplitude*update.float()
        observe(trace,'residual_fp32',residual)
        changed=residual.to(active.dtype)
        observe(trace,'cast_hidden',changed)
        if len(selected) == len(flat):
            return changed.reshape(h1.shape)
        return flat.index_copy(0, selected, changed if compact else changed[selected]).reshape(h1.shape)


def _positions(cache):
    valid = cache.valid_mask.detach().cpu()
    if valid.dtype != torch.bool or valid.ndim != 1 or not bool(valid.any()):
        raise ValueError('cache has invalid supervision mask')
    positions = valid.nonzero().flatten()
    if cache.native_h1.requires_grad or cache.native_h1.shape[-2] != len(valid):
        raise ValueError('cache must contain detached document terminal states')
    if not bool(torch.isfinite(cache.native_h1).all()):
        raise ValueError('nonfinite cached terminal states')
    return positions


def dataset_spec(caches):
    """Only small document IDs/counts are retained; never corpus hidden states."""
    ids, counts, hashes, source_ids = [], [], [], []
    for i in range(len(caches)):
        cache = caches[i]
        ids.append(cache.document_id)
        source_ids.append(_source_id(caches,i,cache.document_id))
        counts.append(len(_positions(cache)))
        hashes.append(digest(dict(input_ids=cache.input_ids,attention_mask=cache.attention_mask,
                                  targets=cache.targets,native_h1=cache.native_h1)))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('nonempty distinct source documents required')
    identity = getattr(caches, 'identity', None)
    return dict(document_ids=ids, counts=counts, unique_positions=sum(counts),
                unique_documents=len(ids), unique_source_documents=len(set(source_ids)),
                source_document_ids=source_ids,document_sha256=hashes,cache_identity=identity)


def _segments(spec, exposure, count, seed):
    """Deterministic repeated-source position schedule spanning document boundaries."""
    total = spec['unique_positions']
    while count:
        cycle, offset = divmod(exposure, total)
        document = 0
        while offset >= spec['counts'][document]:
            offset -= spec['counts'][document]
            document += 1
        take = min(count, spec['counts'][document]-offset)
        yield document, cycle, offset, take
        exposure += take
        count -= take


def _batch(adapter, cache, positions, config):
    h1 = cache.native_h1.reshape(-1, adapter.hidden_size)[positions.to(cache.native_h1.device)]
    h1 = h1.detach().to(adapter.device).reshape(1, len(positions), adapter.hidden_size)
    target = cache.targets.reshape(-1)[positions.to(cache.targets.device)].to(adapter.device)
    if len(positions)*int(adapter.model.config.vocab_size) > config.max_logit_elements:
        raise ValueError('supervised-position logits exceed fixed allocation bound')
    return h1, target


def _ce(adapter, h1, target, branch=None, mask=None, work=None, compact=True, config=None, trace=None):
    if config is not None and branch is not None and config.r2_precision!=branch.scale_config.r2_precision:
        raise ValueError('readout and branch precision protocols differ')
    config=config or (branch.scale_config if branch is not None else None)
    if config is not None:_check_config_deadline(config)
    changed = h1 if branch is None else branch(adapter, h1, mask=mask, work=work,compact=compact,trace=trace)
    if work is not None:
        work['head_forward_attempts']=work.get('head_forward_attempts',0)+1
        work['head_supervised_position_visits']=work.get('head_supervised_position_visits',0)+len(target)
    if config is not None and config.r2_precision=='fp32_canonical32':
        from .canonical_execution import readout
        logits=readout(adapter,changed,config,work=work,trace=trace)
    else:
        handle=None
        if trace is not None:
            def record_norm(module,args,out):trace['readout_normalized']=out.detach().clone()
            handle=adapter.model.model.norm.register_forward_hook(record_norm)
        try:logits=differentiable_readout(adapter,changed)
        finally:
            if handle is not None:handle.remove()
    if trace is not None:trace['logits']=logits.detach().clone()
    if work is not None:
        work['head_forward_completed']=work.get('head_forward_completed',0)+1
    if branch is not None:_check_config_deadline(branch.scale_config)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.to(logits.device), reduction='none')
    if not bool(torch.isfinite(ce).all()):
        raise FloatingPointError('nonfinite cross entropy')
    if config is not None:_check_config_deadline(config)
    return ce


class RunLedger:
    """Write-ahead attempts, immutable wall origin, and unknown crash accounting.

    Wall budget conservatively includes downtime. Therefore an unknown crash
    interval is never silently charged as zero. Operations cannot preempt a CUDA
    kernel; checks occur before/after each bounded operation.
    """
    def __init__(self, root, binding, max_wall_seconds, *, resume=False, absolute_deadline_utc=None):
        self.root = Path(root)
        self.path = self.root/'ledger.json'
        self.max_wall_seconds = max_wall_seconds
        self.absolute_deadline_utc = absolute_deadline_utc
        self.root.mkdir(parents=True, exist_ok=resume)
        if self.path.exists():
            if not resume:
                raise FileExistsError('existing run requires explicit resume')
            self.data = json.loads(self.path.read_text())
            if self.data.get('binding') != binding:
                raise ValueError('ledger identity mismatch')
            for event in self.data['events']:
                if event['status'] == 'attempted':
                    event['status'] = 'unknown_after_interruption'
                    event['unknown_cost'] = True
            self.flush()
        else:
            if resume:
                raise ValueError('resume ledger missing')
            self.data = dict(format=FORMAT, binding=binding, started_utc=time.time(), events=[])
            self.flush()

    def elapsed(self):
        return max(0., time.time()-self.data['started_utc'])

    def check(self):
        if self.absolute_deadline_utc is not None and time.time()>=self.absolute_deadline_utc:
            raise TimeoutError('immutable overall suite wall deadline exhausted')
        if self.elapsed() >= self.max_wall_seconds:
            raise TimeoutError('immutable cell wall budget exhausted (downtime included)')

    def flush(self):
        self.data['conservative_wall_seconds'] = self.elapsed()
        _json(self.path, self.data)

    @contextmanager
    def operation(self, key, work=None):
        self.check()
        event = dict(key=key, attempt=len(self.data['events'])+1, status='attempted', started_utc=time.time())
        self.data['events'].append(event)
        self.flush()
        started = time.perf_counter()
        work_before={} if work is None else dict(work)
        try:
            yield event
            event.update(status='completed', measured_seconds=time.perf_counter()-started)
            if work is not None:
                event['work']={k:v-work_before.get(k,0) for k,v in work.items()}
            self.flush()
            self.check()
        except BaseException as exc:
            # KeyboardInterrupt and controlled exceptions are known attempts.
            if event['status'] != 'completed':
                event.update(status='failed', measured_seconds=time.perf_counter()-started,
                             failure_kind=type(exc).__name__, failure=str(exc))
                if work is not None:
                    event['work']={k:v-work_before.get(k,0) for k,v in work.items()}
                self.flush()
            raise

    def summary(self):
        return dict(wall_seconds=self.elapsed(), wall_scope='conservative elapsed wall including downtime',
                    attempted_operations=len(self.data['events']),
                    completed_operations=sum(x['status']=='completed' for x in self.data['events']),
                    failed_operations=sum(x['status']=='failed' for x in self.data['events']),
                    unknown_operations=sum(x['status']=='unknown_after_interruption' for x in self.data['events']),
                    work={k:sum(e.get('work',{}).get(k,0) for e in self.data['events'])
                          for k in {key for e in self.data['events'] for key in e.get('work',{})}})


def _source_id(caches,index,document_id):
    metadata=caches.document_metadata(index) if hasattr(caches,'document_metadata') else {}
    return metadata.get('source_group_sha256') or metadata.get('source_id') or document_id


def _evidence_path(output, index):
    return Path(output)/f'document-{index:06d}.json'


@torch.no_grad()
def evaluate(adapter, caches, branch=None, *, gate=None, threshold=None, mode='forced',
             role='development', config=None, output=None, ledger=None, allow_final_test=False):
    """Streaming cached-prefix evaluation, immutable evidence per completed document."""
    config = config or (branch.scale_config if branch is not None else ScaleConfig())
    config.validate()
    if branch is not None and config.r2_precision!=branch.scale_config.r2_precision:
        raise ValueError('evaluation and branch precision protocols differ')
    if role == 'final_test' and not allow_final_test:
        raise ValueError('sealed final test requires separate explicit finalized command')
    if role not in ('train', 'gate_fit', 'calibration', 'development', 'final_test'):
        raise ValueError('unsupported data role')
    if mode not in ('native','forced','gate','matched_random','fixed_hash_25','fixed_hash_50','fixed_hash_100'):
        raise ValueError('unsupported evaluation control')
    if mode in ('gate', 'matched_random') and (gate is None or threshold is None):
        raise ValueError('frozen gate and threshold required')
    binding = digest(dict(config=asdict(config), mode=mode, role=role, threshold=threshold,
        branch=None if branch is None else _tensor_hash(branch.state_dict()),
        amplitude=None if branch is None else branch.amplitude,
        gate=None if gate is None else _tensor_hash(gate.state_dict()), cache=getattr(caches,'identity',None)))
    if output is not None:
        Path(output).mkdir(parents=True, exist_ok=True)
    records = []
    work = {}
    document_work=[]
    def compact_record(e):
        return {k:e[k] for k in ('document_id','source_document_id','tokens','native_nll','candidate_nll','continued_tokens')} | {
            'useful_positions':sum(a-b>config.eligibility_gain for a,b in zip(e['native_ce'],e['candidate_ce']))}
    for ci in range(len(caches)):
        path = None if output is None else _evidence_path(output, ci)
        cache = caches[ci]
        if path is not None and path.exists():
            row = json.loads(path.read_text())
            if row.get('binding') != binding or row.get('document_id') != cache.document_id:
                raise ValueError('completed evaluation evidence identity mismatch')
            evidence = row['evidence']
            if row.get('sha256') != digest(evidence):
                raise ValueError('completed evaluation evidence digest mismatch')
            records.append(compact_record(evidence))
            document_work.append(row.get('work',{}))
            continue
        context = ledger.operation('evaluate:'+str(output)+':'+str(ci),work=work) if ledger else _nullcontext()
        with context:
            before_work=dict(work)
            positions = _positions(cache)
            native, candidate, predictions, continues = [], [], [], []
            # Gate masks are label-free. Matched random is retrospective per doc.
            h = cache.native_h1.reshape(-1, adapter.hidden_size)[positions.to(cache.native_h1.device)]
            if gate is not None:
                work['gate_forward_attempts']=work.get('gate_forward_attempts',0)+1
                work['gate_position_visits']=work.get('gate_position_visits',0)+len(positions)
            pred = None if gate is None else gate(h.to(adapter.device)).cpu()
            if gate is not None:
                work['gate_forward_completed']=work.get('gate_forward_completed',0)+1
            if mode == 'native' or branch is None:
                mask = torch.zeros(len(positions), dtype=torch.bool)
            elif mode in ('gate','matched_random'):
                mask = pred > threshold
                if mode == 'matched_random':
                    g=torch.Generator().manual_seed(int(hashlib.sha256((str(config.seed)+cache.document_id).encode()).hexdigest()[:15],16))
                    indices=torch.randperm(len(mask),generator=g)[:int(mask.sum())]
                    mask=torch.zeros_like(mask);mask[indices]=True
            elif mode.startswith('fixed_hash_'):
                fraction=int(mode.rsplit('_',1)[1])/100
                mask=torch.tensor([int(hashlib.sha256(f'{config.seed}:{cache.document_id}:{int(p)}'.encode()).hexdigest()[:16],16)/2**64<fraction for p in positions])
            else:
                mask=torch.ones(len(positions),dtype=torch.bool)
            for offset in range(0,len(positions),config.positions_per_update):
                selected=positions[offset:offset+config.positions_per_update]
                h1,target=_batch(adapter,cache,selected,config)
                base=_ce(adapter,h1,target,work=work,config=config)
                local=mask[offset:offset+len(selected)].to(adapter.device)
                values=base if mode=='native' or branch is None else _ce(adapter,h1,target,branch,local,work)
                native.extend(base.double().cpu().tolist());candidate.extend(values.double().cpu().tolist())
            if pred is not None:
                predictions=pred.tolist()
            evidence=dict(document_id=cache.document_id,source_document_id=_source_id(caches,ci,cache.document_id),tokens=len(positions),valid_position_indices=positions.tolist(),
                native_ce=native,candidate_ce=candidate,gate_predictions=predictions or None,
                continue_mask=mask.tolist(),continued_tokens=int(mask.sum()),
                native_nll=sum(native)/len(native),candidate_nll=sum(candidate)/len(candidate))
            if path is not None:
                _json(path,dict(binding=binding,document_id=cache.document_id,evidence=evidence,sha256=digest(evidence),
                               work={k:v-before_work.get(k,0) for k,v in work.items()}))
            document_work.append({k:v-before_work.get(k,0) for k,v in work.items()})
            records.append(compact_record(evidence))
    tokens=sum(r['tokens'] for r in records)
    if not tokens:
        raise ValueError('empty evaluation population')
    result=dict(status='complete',role=role,mode=mode,tokens=tokens,documents=len(records),
        nll=sum(r['candidate_nll']*r['tokens'] for r in records)/tokens,
        native_nll=sum(r['native_nll']*r['tokens'] for r in records)/tokens,
        continued_tokens=sum(r['continued_tokens'] for r in records),
        useful_positions=sum(r['useful_positions'] for r in records),
        per_document=[{k:r[k] for k in ('document_id','source_document_id','tokens','candidate_nll','native_nll','continued_tokens')} for r in records],
        evidence_binding=binding,evidence_directory=None if output is None else str(Path(output).resolve()),
        work={key:sum(w.get(key,0) for w in document_work) for key in {k for w in document_work for k in w}},protocol='cached native full-document prefix; supervised token-local tail microbatches',
        cost_scope='cached tail only; prefix cache preparation cost must be added separately',
        matched_random_scope='retrospective exact supervised-token count per document; not deployment' if mode=='matched_random' else None)
    if output is not None:
        _json(Path(output)/'summary.json',result)
    return result


@contextmanager
def _nullcontext():
    yield None


def paired_comparison(candidate, reference, *, seed=17, draws=2000):
    """Source-document paired token-weighted gain with clustered bootstrap."""
    a,b=candidate['per_document'],reference['per_document']
    key=lambda x:(x['document_id'],x['tokens'],x.get('source_document_id',x['document_id']))
    if [key(x) for x in a] != [key(x) for x in b]:
        raise ValueError('document comparison identity mismatch')
    clusters={}
    for x,y in zip(a,b):
        source=x.get('source_document_id',x['document_id'])
        numerator,tokens=clusters.get(source,(0.,0))
        clusters[source]=(numerator+(y['candidate_nll']-x['candidate_nll'])*x['tokens'],tokens+x['tokens'])
    if not clusters:
        raise ValueError('empty paired comparison')
    values=list(clusters.values())
    gain=sum(numerator for numerator,n in values)/sum(n for numerator,n in values)
    g=random.Random(seed);boot=[]
    for _ in range(draws):
        sample=[values[g.randrange(len(values))] for _ in values]
        boot.append(sum(numerator for numerator,n in sample)/sum(n for numerator,n in sample))
    boot.sort()
    return dict(gain_nats=gain,ci95=[boot[int(.025*draws)],boot[min(draws-1,int(.975*draws))]],
                documents=len(a),independent_source_documents=len(values),
                method='paired source-document cluster bootstrap of token-weighted mean',
                scope='source-document sampling only; does not cover training-seed or selection uncertainty')


def _rng():
    return dict(torch=torch.get_rng_state(),python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    if state.get('cuda'):
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('resume CUDA RNG device topology mismatch')
        torch.cuda.set_rng_state_all(state['cuda'])


def _snapshot(path, branch, identity, config, epoch, gate=None):
    value=dict(identity=identity,config=asdict(config),epoch=epoch,kind=branch.kind,
               amplitude=branch.amplitude,hidden_size=branch.hidden_size,state=_cpu(branch.state_dict()),
               gate=None if gate is None else _cpu(gate.state_dict()))
    if Path(path).exists():
        if digest(_load(path)) != digest(value):
            raise ValueError('attempt to replace immutable candidate checkpoint')
    else:
        _save(path,value)


def load_candidate(path, config, identity, device='cpu'):
    data=_load(path)
    if digest(data['identity'])!=digest(identity) or digest(data['config'])!=digest(asdict(config)):
        raise ValueError('candidate identity/config mismatch')
    branch=ScaleBranch(data['hidden_size'],config,data['kind'],data['amplitude'],device)
    branch.load_state_dict(data['state'],strict=True);branch.eval()
    gate=None
    if data['gate'] is not None:
        gate=GainGate(data['hidden_size'],device);gate.load_state_dict(data['gate'],strict=True);gate.eval()
    return branch,gate,data


def _opt_to(opt,device):
    for state in opt.state.values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor) and key!='step':
                state[key]=value.to(device)


def run_cell(adapter, train_caches, gate_caches, dev_caches, config, identity, output, *,
             calibration_caches=None, kind='r2', amplitude=.1, resume=False,
             interrupt_after_updates=None):
    """Execute one frozen data/amplitude cell; adapter and gate populations differ.

    Returns a JSON report. Epoch0 and every fixed exposure checkpoint receive
    full train/development curves. Conditional threshold uses calibration only;
    development selects the epoch among calibrated candidates and native STOP.
    """
    config.validate();_assert_frozen(adapter)
    if calibration_caches is None:
        raise ValueError('source-disjoint calibration population is required')
    output=Path(output)
    specs={name:dataset_spec(caches) for name,caches in
           (('train',train_caches),('gate_fit',gate_caches),('calibration',calibration_caches),('development',dev_caches))}
    for i,(role,spec) in enumerate(specs.items()):
        for other in list(specs)[i+1:]:
            if set(spec['document_ids'])&set(specs[other]['document_ids']) or set(spec['source_document_ids'])&set(specs[other]['source_document_ids']):
                raise ValueError(f'population document leakage: {role}/{other}')
    total=config.target_exposures or specs['train']['unique_positions']*config.epochs
    boundaries=([total] if config.epochs==1 else [total//2,total])
    if any(x<=0 for x in boundaries) or len(set(boundaries))!=len(boundaries):
        raise ValueError('positive distinct checkpoint exposure boundaries required')
    required=sum(math.ceil((b-(boundaries[i-1] if i else 0))/config.positions_per_update) for i,b in enumerate(boundaries))
    gate_target=config.gate_target_exposures or specs['gate_fit']['unique_positions']*config.gate_epochs
    gate_required=math.ceil(gate_target/config.positions_per_update)*len(boundaries) if kind=='r2' else 0
    if required+gate_required>config.max_updates:
        raise ValueError('frozen update budget cannot finish adapter and candidate-gate exposure')
    binding=digest(dict(format=FORMAT,config=asdict(config),identity=identity,specs=specs,kind=kind,
                        amplitude=amplitude,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if output.exists() and not resume:
        raise FileExistsError('cell output exists; explicit resume required')
    ledger=RunLedger(output,binding,config.max_wall_seconds,resume=resume,absolute_deadline_utc=config.absolute_deadline_utc)
    state_path=output/'state.pt'
    branch=ScaleBranch(adapter.hidden_size,config,kind,amplitude,adapter.device)
    if resume:
        state=_load(state_path)
        if state['binding']!=binding:
            raise ValueError('resume state source/config/data/model identity mismatch')
        if not 0<=state.get('exposure',-1)<=total or not 0<=state.get('adapter_updates',-1)<=required:
            raise ValueError('resume cursor outside frozen schedule')
        if state.get('audited') != list(range(len(state.get('audited',[])))):
            raise ValueError('resume checkpoint audit sequence mismatch')
        branch.load_state_dict(state['branch'],strict=True)
        _restore_rng(state['rng'])
        if state.get('status')=='complete':
            report=json.loads((output/'report.json').read_text())
            if report.get('sha256')!=digest({k:v for k,v in report.items() if k!='sha256'}):
                raise ValueError('completed report digest mismatch')
            return report
    else:
        state=dict(binding=binding,status='running',exposure=0,adapter_updates=0,gate_updates=0,
                   branch=_cpu(branch.state_dict()),optimizer=None,rng=_rng(),work={},audited=[],gate_completed=[],history=[])
        _save(state_path,state)
    def commit():
        state['branch']=_cpu(branch.state_dict());state['rng']=_rng();state['cost']=ledger.summary()
        _save(state_path,state)
    try:
        optimizer=torch.optim.AdamW(branch.parameters(),lr=config.learning_rate,weight_decay=0.)
        if state['optimizer'] is not None:
            optimizer.load_state_dict(state['optimizer']);_opt_to(optimizer,adapter.device)
        for index,boundary in enumerate([0]+boundaries):
            if boundary:
                branch.train()
                while state['exposure']<boundary:
                    ledger.check()
                    take=min(config.positions_per_update,boundary-state['exposure'])
                    with ledger.operation(f'adapter-update:{state["adapter_updates"]+1}',work=state['work']):
                        optimizer.zero_grad(set_to_none=True)
                        loss_sum=0.
                        for ci,cycle,offset,length in _segments(specs['train'],state['exposure'],take,config.seed):
                            cache=train_caches[ci];positions=_positions(cache)
                            gen=torch.Generator().manual_seed(config.seed+cycle*len(train_caches)+ci)
                            positions=positions[torch.randperm(len(positions),generator=gen)][offset:offset+length]
                            h1,target=_batch(adapter,cache,positions,config)
                            ce=_ce(adapter,h1,target,branch,work=state['work'])
                            state['work']['backward_attempts']=state['work'].get('backward_attempts',0)+1
                            (ce.sum()/take).backward()
                            state['work']['backward_completed']=state['work'].get('backward_completed',0)+1
                            loss_sum+=float(ce.detach().double().sum())
                        norm=torch.nn.utils.clip_grad_norm_(branch.parameters(),config.gradient_clip,error_if_nonfinite=True)
                        if float(norm)==0:
                            state['work']['zero_gradient_updates']=state['work'].get('zero_gradient_updates',0)+1
                            raise FloatingPointError('zero adapter gradient; preserve failed learning attempt')
                        state['work']['optimizer_step_attempts']=state['work'].get('optimizer_step_attempts',0)+1
                        optimizer.step()
                        state['work']['optimizer_step_completed']=state['work'].get('optimizer_step_completed',0)+1
                        _assert_frozen(adapter)
                        if not all(bool(torch.isfinite(p).all()) for p in branch.parameters()):
                            raise FloatingPointError('nonfinite trainable adapter')
                        state['exposure']+=take;state['adapter_updates']+=1
                        state['optimizer']=_cpu(optimizer.state_dict())
                        state['last_update']=dict(exposure=state['exposure'],mean_online_ce=loss_sum/take,gradient_norm=float(norm))
                        commit()
                    if interrupt_after_updates is not None and state['adapter_updates']>=interrupt_after_updates:
                        raise InterruptedError('requested controlled CPU interruption after committed update')
            if index not in state['audited']:
                branch.eval()
                path=output/f'candidate-epoch{index}.pt'
                _snapshot(path,branch,identity,config,index)
                with ledger.operation(f'conditional-parity-epoch{index}',work=state['work']):
                    parity=verify_conditional(adapter,[train_caches[0],calibration_caches[0]],branch,work=state['work'])
                    if not parity['passed']:
                        raise ValueError('trained compact/full conditional numerical parity failed')
                train=evaluate(adapter,train_caches,branch,role='train',config=config,
                               output=output/f'audit-epoch{index}'/'train',ledger=ledger)
                dev=evaluate(adapter,dev_caches,branch,role='development',config=config,
                             output=output/f'audit-epoch{index}'/'development',ledger=ledger)
                state['history'].append(dict(epoch=index,exposure=state['exposure'],updates=state['adapter_updates'],train=train,development=dev,conditional_parity=parity))
                state['audited'].append(index);commit()
        # The final forced branch is fixed; gate fitting never changes it.
        for epoch in range(1,len(boundaries)+1):
            if kind!='r2' or epoch in state['gate_completed']:
                continue
            candidate,_,_=load_candidate(output/f'candidate-epoch{epoch}.pt',config,identity,adapter.device)
            gp=state.get('gate_progress')
            gate=GainGate(adapter.hidden_size,adapter.device)
            gateopt=torch.optim.AdamW(gate.parameters(),lr=config.gate_learning_rate,weight_decay=0.)
            if gp is not None:
                if gp['epoch']!=epoch:
                    raise ValueError('gate cursor epoch mismatch')
                gate.load_state_dict(gp['state'],strict=True);gateopt.load_state_dict(gp['optimizer']);_opt_to(gateopt,adapter.device)
                exposure=gp['exposure']
            else:
                exposure=0
            gate_total=gate_target
            while exposure<gate_total:
                take=min(config.positions_per_update,gate_total-exposure)
                with ledger.operation(f'gate-epoch{epoch}-update:{exposure}',work=state['work']):
                    gateopt.zero_grad(set_to_none=True)
                    for ci,cycle,offset,length in _segments(specs['gate_fit'],exposure,take,config.seed):
                        cache=gate_caches[ci];positions=_positions(cache)
                        gen=torch.Generator().manual_seed(config.seed+cycle*len(gate_caches)+ci)
                        positions=positions[torch.randperm(len(positions),generator=gen)][offset:offset+length]
                        h1,target=_batch(adapter,cache,positions,config)
                        with torch.no_grad():
                            gain=(_ce(adapter,h1,target,work=state['work'],config=config)-_ce(adapter,h1,target,candidate,work=state['work'])).detach().to(adapter.device)
                        state['work']['gate_forward_attempts']=state['work'].get('gate_forward_attempts',0)+1
                        state['work']['gate_position_visits']=state['work'].get('gate_position_visits',0)+length
                        prediction=gate(h1).reshape(-1)
                        state['work']['gate_forward_completed']=state['work'].get('gate_forward_completed',0)+1
                        state['work']['gate_backward_attempts']=state['work'].get('gate_backward_attempts',0)+1
                        (F.mse_loss(prediction,gain,reduction='sum')/take).backward()
                        state['work']['gate_backward_completed']=state['work'].get('gate_backward_completed',0)+1
                    torch.nn.utils.clip_grad_norm_(gate.parameters(),config.gradient_clip,error_if_nonfinite=True)
                    state['work']['gate_optimizer_attempts']=state['work'].get('gate_optimizer_attempts',0)+1
                    gateopt.step()
                    state['work']['gate_optimizer_completed']=state['work'].get('gate_optimizer_completed',0)+1
                    if not all(bool(torch.isfinite(p).all()) for p in gate.parameters()):
                        raise FloatingPointError('nonfinite gain gate')
                    exposure+=take;state['gate_updates']+=1
                    state['gate_progress']=dict(epoch=epoch,exposure=exposure,state=_cpu(gate.state_dict()),optimizer=_cpu(gateopt.state_dict()))
                    commit()
            gate.eval();_snapshot(output/f'gated-epoch{epoch}.pt',candidate,identity,config,epoch,gate)
            state['gate_completed'].append(epoch);state['gate_progress']=None;commit()
        forced=[h for h in state['history'] if h['epoch']>0]
        forced_winner=min(forced,key=lambda h:(h['development']['nll'],h['epoch']))
        # Native STOP is a separate candidate, never an untrained R2 epoch0.
        native=evaluate(adapter,dev_caches,mode='native',role='development',config=config,output=output/'selection'/'native',ledger=ledger)
        forced_choice=dict(mode='forced',epoch=forced_winner['epoch'],nll=forced_winner['development']['nll'],
                           checkpoint=str((output/f'candidate-epoch{forced_winner["epoch"]}.pt').resolve()))
        calibrated=[];selection=[dict(mode='native',epoch=None,threshold=None,development=native)]
        if kind=='r2':
            for epoch in range(1,len(boundaries)+1):
                candidate,gate,_=load_candidate(output/f'gated-epoch{epoch}.pt',config,identity,adapter.device)
                cal_native=evaluate(adapter,calibration_caches,mode='native',role='calibration',config=config,
                                    output=output/f'calibration-epoch{epoch}'/'native',ledger=ledger)
                choices=[dict(mode='native',threshold=None,calibration=cal_native)]
                for ti,threshold in enumerate(config.thresholds):
                    result=evaluate(adapter,calibration_caches,candidate,gate=gate,threshold=threshold,mode='gate',
                        role='calibration',config=config,output=output/f'calibration-epoch{epoch}'/f'threshold-{ti}',ledger=ledger)
                    choices.append(dict(mode='gate',threshold=threshold,calibration=result))
                choice=min(choices,key=lambda c:(c['calibration']['nll'],c['mode']!='native',c['threshold'] or 0))
                calibrated.append(dict(epoch=epoch,candidates=choices,selected={k:v for k,v in choice.items() if k!='calibration'}))
                if choice['mode']=='native':
                    development=native
                else:
                    development=evaluate(adapter,dev_caches,candidate,gate=gate,threshold=choice['threshold'],mode='gate',
                        role='development',config=config,output=output/'selection'/f'epoch-{epoch}',ledger=ledger)
                selection.append(dict(epoch=epoch,mode=choice['mode'],threshold=choice['threshold'],development=development))
        chosen=min(selection,key=lambda c:(c['development']['nll'],c['mode']!='native',c['epoch'] or 0))
        chosen={k:v for k,v in chosen.items() if k!='development'}|{'nll':chosen['development']['nll']}
        controls={}
        if kind=='r2' and chosen['mode']=='gate':
            candidate,gate,_=load_candidate(output/f'gated-epoch{chosen["epoch"]}.pt',config,identity,adapter.device)
            for mode in ('matched_random','fixed_hash_25','fixed_hash_50','fixed_hash_100'):
                controls[mode]=evaluate(adapter,dev_caches,candidate,gate=gate,threshold=chosen['threshold'],mode=mode,
                    role='development',config=config,output=output/'controls'/mode,ledger=ledger)
        report=dict(format=FORMAT,status='complete',binding=binding,identity=identity,config=asdict(config),
            kind=kind,amplitude=amplitude,specs=specs,adapter_exposures=state['exposure'],
            unique_adapter_positions=specs['train']['unique_positions'],unique_gate_positions=specs['gate_fit']['unique_positions'],
            adapter_updates=state['adapter_updates'],gate_updates=state['gate_updates'],checkpoint_exposures=boundaries,
            gate_exposures_per_candidate=gate_target if kind=='r2' else 0,
            forced_selection=forced_choice,conditional_selection=chosen,calibration=calibrated,development_candidates=selection,
            history=state['history'],controls=controls,gated_selection=chosen,
            selection_results=dict(native=native,forced=forced_winner['development'],
                gate=min(selection,key=lambda c:(c['development']['nll'],c['mode']!='native',c['epoch'] or 0))['development'],**controls),
            cost=ledger.summary(),work=state['work'],
            matched_exposure=config.target_exposures is not None,
            limitations=['CPU tests cannot establish GPU quality, memory, compaction parity or speed',
                        'parameter/update matching is not FLOPs matching',
                        'cached-tail costs require adding cache preparation and final online profiling'])
        report=json.loads(json.dumps(report,allow_nan=False))
        report['sha256']=digest(report);_json(output/'report.json',report)
        state['status']='complete';commit()
        return report
    except BaseException as exc:
        _json(output/'failure.json',dict(status='interrupted' if isinstance(exc,InterruptedError) else 'failed',
            kind=type(exc).__name__,message=str(exc),binding=binding,committed_exposure=state['exposure'],
            committed_adapter_updates=state['adapter_updates'],cost=ledger.summary(),work=state['work']))
        raise


def load_selected(output, config, identity, device='cpu', *, which='gated'):
    report=json.loads((Path(output)/'report.json').read_text())
    if report.get('sha256')!=digest({k:v for k,v in report.items() if k!='sha256'}):
        raise ValueError('selected report digest mismatch')
    if digest(report['identity'])!=digest(identity) or digest(report['config'])!=digest(asdict(config)):
        raise ValueError('selected report identity/config mismatch')
    if which=='forced':
        selection=report['forced_selection']
        branch,gate,_=load_candidate(Path(output)/f'candidate-epoch{selection["epoch"]}.pt',config,identity,device)
    elif which=='gated':
        selection=report['conditional_selection']
        if selection['mode']=='native':
            return None,None,selection
        branch,gate,_=load_candidate(Path(output)/f'gated-epoch{selection["epoch"]}.pt',config,identity,device)
    else:
        raise ValueError('selector must be forced or gated')
    return branch,gate,selection


def preflight(adapter, cache, config, *, amplitude=.1, atol=1e-4, rtol=1e-4, expert_probe_budget=None, gradient_checks=None, diagnostics=True, path_probe_budget=None):
    """Actual-backend gradient plus full/compacted native and R2 CE audit.

    Run on the real target GPUs before accepting throughput. A failure is an
    integrity stop; changing the tolerance/backend requires a new protocol.
    """
    from .document_training import cache_documents, _comparison, _inputs, document_losses
    from .bridge import _native_document
    config.validate();_check_config_deadline(config);_assert_frozen(adapter)
    backbone_versions={name:p._version for name,p in adapter.model.named_parameters()}
    positions=_positions(cache)
    h1=cache.native_h1.detach().to(adapter.device)
    branch=ScaleBranch(adapter.hidden_size,config,'r2',amplitude,adapter.device)
    work={}
    if gradient_checks is None:
        gradient_checks = config.r2_precision in ('fp32_experts','fp32_canonical32')
    if config.r2_precision in ('fp32_experts','fp32_canonical32'):
        gradient_checks = True
        diagnostics = False  # Old observer hooks observe only native moe.experts.
    with torch.no_grad():
        native_hidden,native_logits=_native_document(adapter,_inputs(adapter,cache))
        _check_config_deadline(config)
        reference_native=document_losses(native_logits,cache)[positions.to(native_logits.device)]
        full=_ce(adapter,h1,cache.targets.to(adapter.device),branch,work=work)
        full_native=_ce(adapter,h1,cache.targets.to(adapter.device),config=config)
        compact=[];compact_native=[]
        for start in range(0,len(positions),config.positions_per_update):
            _check_config_deadline(config)
            hh,tt=_batch(adapter,cache,positions[start:start+config.positions_per_update],config)
            compact.append(_ce(adapter,hh,tt,branch,work=work));compact_native.append(_ce(adapter,hh,tt,config=config))
        compact=torch.cat(compact);compact_native=torch.cat(compact_native)
        metrics=dict(native_hidden=_comparison(h1,native_hidden,atol,rtol),
                     native_ce=_comparison(compact_native,reference_native,atol,rtol),
                     full_native_ce=_comparison(full_native[positions.to(full_native.device)],compact_native,atol,rtol),
                     full_vs_compact_r2_ce=_comparison(full[positions.to(full.device)],compact,atol,rtol))
        mask=torch.zeros(h1.shape[-2],dtype=torch.bool,device=adapter.device)
        before=dict(work);stopped=branch(adapter,h1,mask,work=work)
        stop_ok=torch.equal(stopped,h1) and work==before
    conditional_before=verify_conditional(adapter,[cache],branch,atol=atol,rtol=rtol,work=work,diagnostics=diagnostics,expert_probe_budget=expert_probe_budget,path_probe_budget=path_probe_budget)
    gradients_before=verify_gradient_parity(adapter,cache,branch,atol=atol,rtol=rtol,work=work) if gradient_checks else None
    branch.train();hh,tt=_batch(adapter,cache,positions[:config.positions_per_update],config)
    ce=_ce(adapter,hh,tt,branch,work=work)
    _check_config_deadline(config)
    ce.mean().backward()
    _check_config_deadline(config)
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in branch.parameters()):
        raise FloatingPointError('nonfinite ephemeral adapter gradient')
    gradient=branch.correction.up.grad
    gradient_ok=gradient is not None and bool(torch.isfinite(gradient).all()) and float(gradient.abs().sum())>0
    with torch.no_grad():
        if config.r2_precision=='fp32_canonical32':
            from .canonical_execution import tiled
            xx=tiled(hh.reshape(-1,adapter.hidden_size),adapter.layer.post_attention_layernorm,deadline=config.absolute_deadline_utc)
        else:xx=adapter.layer.post_attention_layernorm(hh.reshape(-1,adapter.hidden_size))
        before_input=_effective_expert_input(branch,xx).clone()
        before_hidden=branch(adapter,hh,work=work).clone()
    opt=torch.optim.AdamW(branch.parameters(),lr=config.learning_rate,weight_decay=0.)
    work['preflight_optimizer_attempts']=work.get('preflight_optimizer_attempts',0)+1
    _check_config_deadline(config)
    opt.step()
    work['preflight_optimizer_completed']=work.get('preflight_optimizer_completed',0)+1
    with torch.no_grad():
        after_input=_effective_expert_input(branch,xx)
        after_hidden=branch(adapter,hh,work=work)
    effective=int((before_input!=after_input).sum())
    hidden_changes=int((before_hidden!=after_hidden).sum())
    conditional=verify_conditional(adapter,[cache],branch,atol=atol,rtol=rtol,work=work,diagnostics=diagnostics,expert_probe_budget=expert_probe_budget,path_probe_budget=path_probe_budget)
    gradients_after=verify_gradient_parity(adapter,cache,branch,atol=atol,rtol=rtol,work=work) if gradient_checks else None
    _assert_frozen(adapter)
    with torch.no_grad():
        post_ce=_ce(adapter,h1,cache.targets.to(adapter.device),branch,work=work) if gradient_checks else None
    unchanged=all(p._version==backbone_versions[name] for name,p in adapter.model.named_parameters())
    backbone_gradients_absent=all(p.grad is None for p in adapter.model.parameters())
    gradient_parity_ok=not gradient_checks or (gradients_before['passed'] and gradients_after['passed'])
    return dict(status='complete',passed=backbone_gradients_absent and gradient_parity_ok and stop_ok and gradient_ok and effective>0 and unchanged and conditional_before['passed'] and conditional['passed'] and all(x['mismatched_elements']==0 for x in metrics.values()),
                comparisons=metrics,stop_zero_experts=stop_ok,up_gradient_norm=None if gradient is None else float(gradient.norm()),
                backbone_gradients_absent=backbone_gradients_absent,
                r2_precision=config.r2_precision,canonical_tile_rows=32 if config.r2_precision=='fp32_canonical32' else None,
                train_document_loss_snapshot=dict(native=float(full_native[positions.to(full_native.device)].mean()),
                    r2_before_update=float(full[positions.to(full.device)].mean()),
                    r2_after_update=None if post_ce is None else float(post_ce[positions.to(post_ce.device)].mean()),
                    scope='one train document; own ephemeral update per precision; not validation quality'),
                gradient_parity_before_update=gradients_before,
                gradient_parity_after_update=gradients_after,
                diagnostic_scope='native observer only' if diagnostics else 'legacy observer disabled; hidden/CE and gradient checks retained',
                optimizer_steps=1,optimizer_scope='ephemeral discarded train-only branch; no formal update',
                effective_expert_input_changed_elements=effective,post_update_hidden_changed_elements=hidden_changes,
                backbone_parameter_versions_unchanged=unchanged,conditional_parity=conditional,
                conditional_parity_before_update=conditional_before,
                work=work,amplitude=amplitude,tolerances=dict(atol=atol,rtol=rtol),
                scope='actual backend/shape correctness and gradient only; no quality or speed claim')


@torch.no_grad()
def verify_conditional(adapter,caches,branch,*,atol=1e-4,rtol=1e-4,work=None,diagnostics=False,expert_probe_budget=None,path_probe_budget=None):
    """Full-shape masked versus genuinely compacted trained STOP/mixed/all R2."""
    from .document_training import _comparison
    records=[]
    for cache in caches:
        positions=_positions(cache)
        h1=cache.native_h1.detach().to(adapter.device)
        n=h1.numel()//adapter.hidden_size
        masks={'stop':torch.zeros(n,dtype=torch.bool,device=adapter.device),
               'all':torch.ones(n,dtype=torch.bool,device=adapter.device),
               'mixed':torch.zeros(n,dtype=torch.bool,device=adapter.device)}
        masks['mixed'][positions[::2].to(adapter.device)]=True
        for suffix,selected in (('singleton_first',positions[:1]),('singleton_last',positions[-1:]),('mixed_odd',positions[1::2])):
            masks[suffix]=torch.zeros(n,dtype=torch.bool,device=adapter.device)
            masks[suffix][selected.to(adapter.device)]=True
        diagnostic_used=False
        for name,mask in masks.items():
            _check_config_deadline(branch.scale_config)
            before={} if work is None else dict(work)
            compact=branch(adapter,h1,mask,compact=True,work=work)
            dense=branch(adapter,h1,mask,compact=False,work=work)
            ce1=_ce(adapter,compact,cache.targets.to(adapter.device),work=work,config=branch.scale_config)
            ce2=_ce(adapter,dense,cache.targets.to(adapter.device),work=work,config=branch.scale_config)
            metrics=dict(hidden=_comparison(compact,dense,atol,rtol),
                         ce=_comparison(ce1[positions.to(ce1.device)],ce2[positions.to(ce2.device)],atol,rtol))
            stop_ok=(name!='stop' or (torch.equal(compact,h1) and (work is None or work.get('extra_expert_token_calls',0)==before.get('extra_expert_token_calls',0))))
            records.append(dict(document_id=cache.document_id,mask=name,continued_tokens=int(mask.sum()),
                                comparisons=metrics,stop_zero_experts=stop_ok))
            # Trace at most one failing mixed mask per document and parameter state.
            if diagnostics and not diagnostic_used and 0<int(mask.sum())<n and any(m['mismatched_elements'] for m in metrics.values()):
                from .loop_parity_diagnostics import diagnose_compaction
                records[-1]['layer_diagnostic']=diagnose_compaction(adapter,cache,branch,mask,atol=atol,rtol=rtol,work=work,expert_probe_budget=expert_probe_budget)
                diagnostic_used=True
            if path_probe_budget is not None and name.startswith('singleton'):
                if any(m['mismatched_elements'] for m in metrics.values()):
                    if path_probe_budget['remaining']>0:
                        path_probe_budget['remaining']-=1
                        from .canonical_execution import diagnose_path
                        try:
                            records[-1]['path_diagnostic']=diagnose_path(adapter,cache,branch,mask,atol=atol,rtol=rtol,work=work)
                        except Exception as error:
                            error.diagnostic_evidence=dict(phase='singleton_path',mask=name,r2_precision=branch.scale_config.r2_precision,
                                work=dict(work or {}),conditional_records=records)
                            raise
                    else:records[-1]['path_diagnostic']={'status':'skipped_probe_budget_exhausted'}
    return dict(status='complete',passed=all(r['stop_zero_experts'] and all(m['mismatched_elements']==0 for m in r['comparisons'].values()) for r in records),
                documents=records,tolerances=dict(atol=atol,rtol=rtol),scope='STOP/all/even/odd/singleton-first/singleton-last; compact versus compute-all then mask')


def _effective_expert_input(branch, x):
    if branch.scale_config.r2_precision=='fp32_canonical32':
        from .canonical_execution import tiled
        from .fp32_experts import correction_fp32
        return x.float()+tiled(x,lambda z:correction_fp32(branch.correction,z,deadline=branch.scale_config.absolute_deadline_utc),
            deadline=branch.scale_config.absolute_deadline_utc)
    if branch.scale_config.r2_precision in ('fp32_experts','fp32_canonical32'):
        from .fp32_experts import correction_fp32
        return x.float()+correction_fp32(branch.correction,x,deadline=branch.scale_config.absolute_deadline_utc)
    return (x.float()+branch.correction(x)).to(x.dtype)


def verify_gradient_parity(adapter, cache, branch, *, atol=1e-4, rtol=1e-4, work=None):
    """Same parameters and selected-token CE; autograd.grad never fills .grad."""
    from .loop_parity_diagnostics import _difference
    h=cache.native_h1.detach().to(adapter.device)
    n=h.numel()//adapter.hidden_size
    positions=_positions(cache).to(adapter.device)
    parameters=tuple(branch.correction.parameters())
    records=[]
    for name, chosen in (('mixed',positions[::2]),('mixed_odd',positions[1::2]),
                         ('singleton_first',positions[:1]),('singleton_last',positions[-1:])):
        _check_config_deadline(branch.scale_config)
        if not len(chosen):
            records.append(dict(mask=name,status='not_applicable_no_selected_positions',passed=True))
            continue
        mask=torch.zeros(n,dtype=torch.bool,device=adapter.device);mask[chosen]=True
        values=[]
        for compact in (False,True):
            if work is not None:work['gradient_parity_backward_attempts']=work.get('gradient_parity_backward_attempts',0)+1
            ce=_ce(adapter,h,cache.targets.to(adapter.device),branch,mask,work=work,compact=compact)
            loss=ce.reshape(-1)[chosen].mean()
            _check_config_deadline(branch.scale_config)
            gradients=torch.autograd.grad(loss,parameters,allow_unused=False)
            _check_config_deadline(branch.scale_config)
            if any(not bool(torch.isfinite(g).all()) for g in gradients):
                raise FloatingPointError('nonfinite gradient parity evidence')
            if work is not None:work['gradient_parity_backward_completed']=work.get('gradient_parity_backward_completed',0)+1
            values.append((loss.detach(),gradients))
        comparisons={key:_difference(b,a,atol,rtol) for key,a,b in zip(
            ('down','up'),values[0][1],values[1][1])}
        loss_comparison=_difference(values[1][0],values[0][0],atol,rtol)
        norms={kind:[float(g.detach().float().norm()) for g in values[i][1]] for i,kind in enumerate(('dense','compact'))}
        finite=all(v['nonfinite_elements']==0 for v in [loss_comparison,*comparisons.values()])
        records.append(dict(mask=name,positions=len(chosen),comparisons=comparisons,loss=loss_comparison,
            gradient_norms=norms,passed=finite and norms['dense'][1]>0 and norms['compact'][1]>0 and
            all(v['mismatched_elements']==0 for v in [loss_comparison,*comparisons.values()])))
    return dict(passed=all(r['passed'] for r in records),records=records,tolerances=dict(atol=atol,rtol=rtol),
        scope='same adapter state, same selected-token mean CE; even/odd and first/last singleton; zero down gradient at zero-up initialization is expected')


def _check_config_deadline(config):
    if config.absolute_deadline_utc is not None and time.time()>=config.absolute_deadline_utc:
        raise TimeoutError('R2 protocol shared deadline exhausted between operations')
