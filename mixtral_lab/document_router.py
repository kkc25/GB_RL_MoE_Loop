"""Full-document R1 router experiment with explicitly surrogate cost labels.

Each label column executes one uniform action over the ENTIRE padded document.
The resulting table is not the numerical cost of an arbitrary mixed policy:
expert batch occupancy changes with other tokens' actions. Every reported train
NLL and development checkpoint choice instead executes the real mixed policy.
Only actor features/cost labels are masked; expert and output-head inputs are not.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import time

import torch

from .adaptive_fit import AdaptiveFitConfig, AdaptiveFitController
from .document_training import DocumentCache, document_forward, document_losses, _validate_cache
from .full_action_fit import _digest, _tensor_hash, _tensor_hashes, _frozen, full_action_expected_ce
from .recovery_router import (RecoveryPairRouter, compact_r1_features,
    soft_target_cross_entropy, native_reference_kl, _number, _integer,
    _development_contract, _development, _json, _save)
from .training import pair_actions

FORMAT = 'mixtral-document-router-v1'
TABLE_FORMAT = 'mixtral-document-uniform-action-surrogate-v1'
DEVELOPMENT_PROTOCOL = 'full_document_online_nll'
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
SCOPE = 'Full-document uniform-action supervision surrogate; actual mixed-policy train probes and development-only model selection; no final test or deployment claim.'


@dataclass(frozen=True)
class DocumentRouterConfig:
    updates: int = 256
    positions_per_update: int = 32
    learning_rate: float = 1e-3
    epsilon: float = .8
    temperature: float = .1
    reference_kl_weight: float = .01
    objective: str = 'soft_target_ce'
    max_train_tokens: int = 16384
    max_table_calls: int = 4096
    max_cache_bytes: int = 1024 * 1024 * 1024
    max_logit_elements: int = 16_000_000
    parity_atol: float = 0.
    parity_rtol: float = 0.
    eval_interval: int = 16
    development_patience: int = 5
    min_updates: int = 32
    development_min_delta: float = 1e-4
    gradient_clip: float = 1.
    seed: int = 17
    adaptive: dict | None = None

    def validate(self):
        _integer(self.updates, 'updates', 0, 10000)
        _integer(self.positions_per_update, 'positions_per_update', 1, 16384)
        _integer(self.max_train_tokens, 'max_train_tokens', 1, 16384)
        _integer(self.max_table_calls, 'max_table_calls', 1, 4096)
        _integer(self.max_cache_bytes, 'max_cache_bytes', 1, 8 * 1024**3)
        _integer(self.max_logit_elements, 'max_logit_elements', 1, 64_000_000)
        _number(self.parity_atol, 'parity_atol', upper=1.)
        _number(self.parity_rtol, 'parity_rtol', upper=1.)
        for key in ('eval_interval', 'development_patience'):
            _integer(getattr(self, key), key, 1, 10000)
        _integer(self.min_updates, 'min_updates', 0, 10000)
        _integer(self.seed, 'seed', 0, 2**63-1)
        if self.positions_per_update > self.max_train_tokens:
            raise ValueError('positions_per_update exceeds max_train_tokens')
        if 1 + math.ceil(self.updates/self.eval_interval) > 1024:
            raise ValueError('development evaluation budget exceeds 1024')
        for key, upper in (('learning_rate', 1.), ('epsilon', 1.), ('temperature',100.), ('gradient_clip', 1e6)):
            _number(getattr(self, key), key, upper=upper, strict=True)
        if self.epsilon >= 1 or self.temperature < 1e-6:
            raise ValueError('epsilon must be below 1; temperature at least 1e-6')
        for key in ('reference_kl_weight', 'development_min_delta'):
            _number(getattr(self, key), key, upper=100.)
        if self.objective not in ('soft_target_ce', 'expected_ce'):
            raise ValueError('unknown router objective')
        adaptive = AdaptiveFitConfig.from_dict(self.adaptive)
        if adaptive.enabled and adaptive.eval_interval != self.eval_interval:
            raise ValueError('adaptive eval_interval must equal eval_interval')
        AdaptiveFitController(adaptive, initial_lr=self.learning_rate)


def _sources():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
        ('document_router.py', 'document_training.py', 'document_protocol.py', 'recovery_router.py',
         'adaptive_fit.py', 'adapter.py', 'full_action_fit.py', 'training.py')}


def _cache_binding(caches, config):
    if not isinstance(caches, (list, tuple)) or not 1 <= len(caches) <= 128:
        raise ValueError('expected bounded nonempty full-document cache list')
    ids, binding, provenance, count = [], [], [], 0
    for cache in caches:
        if not isinstance(cache, DocumentCache):
            raise ValueError('DocumentCache required')
        if not isinstance(cache.document_id, str) or not cache.document_id or cache.document_id in ids:
            raise ValueError('unique document IDs required')
        ids.append(cache.document_id)
        h = cache.h0
        if (not isinstance(h, torch.Tensor) or h.ndim != 3 or h.shape[0] != 1
                or not 2 <= h.shape[1] <= 4096 or not h.is_floating_point()
                or not bool(torch.isfinite(h).all())):
            raise ValueError('finite padded h0 [1,S,D] required')
        if (cache.targets.shape != (h.shape[1],) or cache.targets.dtype != torch.long
                or cache.valid_mask.shape != cache.targets.shape or cache.valid_mask.dtype != torch.bool
                or bool(cache.valid_mask[-1]) or not bool(cache.valid_mask.any())
                or bool((cache.targets < 0).any())):
            raise ValueError('invalid full-document target/mask contract')
        pos = cache.valid_mask.nonzero().flatten().cpu().tolist()
        count += len(pos)
        provenance += [{'document_id':cache.document_id, 'token_position':p,
                        'target':int(cache.targets[p]), 'role':'train'} for p in pos]
        binding.append({'document_id':cache.document_id, 'h0':_tensor_hash(h),
            'native_h1':_tensor_hash(cache.native_h1), 'targets':_tensor_hash(cache.targets),
            'valid_mask':_tensor_hash(cache.valid_mask), 'input_ids':_tensor_hash(cache.input_ids),
            'attention_mask':_tensor_hash(cache.attention_mask)})
    if count > config.max_train_tokens:
        raise ValueError('full-document train token budget exceeded')
    if 29*len(caches) > config.max_table_calls:
        raise ValueError('uniform-action table requires 29 whole-document calls per document')
    return binding, provenance, count


def _objective(actor, features, pairs, costs, config):
    logp = actor.log_probs(features, pairs)
    loss = (soft_target_cross_entropy(logp, costs, config.temperature) if config.objective == 'soft_target_ce'
            else full_action_expected_ce(logp, costs))
    return loss + config.reference_kl_weight*native_reference_kl(logp, actor.reference_log_probs(pairs))


def _call(adapter, cache, policy, work, phase):
    work[phase+'_attempted_document_forwards'] += 1
    logits, hidden = document_forward(adapter, cache, policy, kind='router')
    losses = document_losses(logits, cache)
    if losses.shape != cache.valid_mask.shape or not bool(torch.isfinite(losses).all()):
        raise ValueError('nonfinite or malformed full-document router losses')
    work[phase+'_completed_document_forwards'] += 1
    work[phase+'_head_rows'] += cache.h0.shape[1]
    return losses


@torch.no_grad()
def build_document_cost_table(adapter, caches, config, work):
    """Uniform-action FULL-document labels; no singleton scoring or head slicing."""
    features, pairs, costs, native, offsets = [], [], [], [], []
    cursor = 0
    for cache in caches:
        h = cache.h0
        # Feature creation, norm and gate always see the full document too.
        obs = adapter.observe(h)
        compact_r1_features(obs.features)
        mask = cache.valid_mask.to(h.device)
        features.append(obs.features[mask].detach())
        pairs.append(obs.native_pairs[mask].detach())
        native_loss = _call(adapter, cache, None, work, 'table')
        native.append(native_loss[mask.to(native_loss.device)].detach().to(h.device).double())
        columns = []
        for action in range(1,29):
            def constant(observation, action=action):
                return torch.full((len(observation.features),), action, dtype=torch.long,
                                  device=observation.features.device)
            loss = _call(adapter, cache, constant, work, 'table')
            columns.append(loss[mask.to(loss.device)].detach().to(h.device).double())
        values = torch.stack(columns, 1)
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError('invalid full-document surrogate costs')
        costs.append(values)
        offsets.append((cursor, cursor+len(values)))
        cursor += len(values)
    return {'features':torch.cat(features), 'native_pairs':torch.cat(pairs),
            'costs':torch.cat(costs), 'native_losses':torch.cat(native)}, offsets


@torch.no_grad()
def _probe(adapter, caches, actor, table, visits, work):
    losses, executed_actions, executed_native, entropies = [], [], [], []
    per_document = []
    cursor = 0
    for cache in caches:
        captured = {}
        def instrumented(observation):
            # These are the actions actually returned to the complete padded
            # tail. Never recompute their argmax on concatenated/masked rows:
            # FP32 GEMM batch shape can change a near-tie decision.
            actions = actor(observation)
            logp = actor.log_probs(observation.features, observation.native_pairs)
            captured['actions'] = actions.detach().clone()
            captured['native'] = pair_actions(observation.native_pairs).detach().clone()
            captured['entropy'] = -(logp.exp()*logp).sum(-1)
            return actions
        values = _call(adapter, cache, instrumented, work, 'probe')
        if not captured or captured['actions'].shape != cache.valid_mask.shape:
            raise ValueError('full-document policy action evidence missing or malformed')
        mask = cache.valid_mask.to(captured['actions'].device)
        executed_actions.append((captured['actions'][mask]-1).to(table['costs'].device))
        executed_native.append(captured['native'][mask].to(table['costs'].device))
        entropies.append(captured['entropy'][mask].to(table['costs'].device))
        valid = values[cache.valid_mask.to(values.device)].double().to(table['costs'].device)
        losses.append(valid)
        base = table['native_losses'][cursor:cursor+len(valid)]
        per_document.append({'document_id':cache.document_id,'tokens':len(valid),
            'native_nll':float(base.mean()), 'nll':float(valid.mean()), 'gain_nats':float((base-valid).mean())})
        cursor += len(valid)
    actual = torch.cat(losses)
    native = table['native_losses']
    actions, native_actions = torch.cat(executed_actions), torch.cat(executed_native)
    surrogate = table['costs'].gather(1,actions[:,None]).squeeze(1)
    gains = native-actual
    positives = gains.clamp_min(0)
    seen = visits.to(actual.device)>0
    populations = {}
    for name,mask in (('all_train_positions',torch.ones_like(seen)),('optimizer_seen_positions',seen),
                      ('optimizer_unseen_positions',~seen)):
        populations[name] = {'tokens':int(mask.sum()),
            'native_nll':float(native[mask].mean()) if bool(mask.any()) else None,
            'nll':float(actual[mask].mean()) if bool(mask.any()) else None}
    return {'tokens':len(actual), 'native_nll':float(native.mean()), 'nll':float(actual.mean()),
        'gain_nats':float(gains.mean()), 'gain_median_nats':float(gains.quantile(.5)),
        'gain_worst_nats':float(gains.min()), 'positive_fraction':float((gains>0).double().mean()),
        'top2_positive_gain_share':float(positives.topk(min(2,len(actual))).values.sum()/positives.sum()) if float(positives.sum()) else 0.,
        'native_agreement':float((actions==native_actions).double().mean()),
        'entropy':float(torch.cat(entropies).double().mean()),
        'action_diagnostics_protocol':'executed full-padded-document policy, valid positions only',
        'greedy_action_counts':torch.bincount(actions.cpu(),minlength=28).tolist(),
        'surrogate_nll':float(surrogate.mean()),
        'surrogate_minus_actual_nll':float((surrogate-actual).mean()),
        'surrogate_actual_max_abs':float((surrogate-actual).abs().max()),
        'surrogate_is_deployment_cost':False, 'per_document':per_document,'populations':populations}


def _payload(actor,binding,update,train,dev,controller,lr_used):
    state = {k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
    header = {'format':FORMAT,'binding':binding,'feature_dim':actor.feature_dim,'epsilon':actor.epsilon,
        'update':update,'train_metrics':train,'development_metrics':dev,'resume_supported':False,
        'actor_state_hashes':_tensor_hashes(state),'adaptive_audit_state':controller.snapshot(),
        'learning_rate_used':lr_used,'learning_rate_next':controller.lr}
    return {**header,'payload_sha256':_digest(header),'actor_state_dict':state}


def train_document_router(adapter, caches, development_evaluator, config, identities, output_dir):
    """Development-select a router; input documents and output directory are frozen."""
    if not isinstance(config,DocumentRouterConfig):
        raise ValueError('DocumentRouterConfig required')
    config.validate()
    cache_identity,provenance,count = _cache_binding(caches,config)
    for cache in caches:
        _validate_cache(adapter,cache)
        if cache.h0.shape[1] * int(adapter.model.config.vocab_size) > config.max_logit_elements:
            raise ValueError('full-document logits exceed configured allocation bound')
    cache_bytes = sum(t.numel()*t.element_size() for c in caches for t in
        (c.h0,c.native_h1,c.targets,c.valid_mask,c.input_ids,c.attention_mask))
    if cache_bytes > config.max_cache_bytes:
        raise ValueError('full-document cache exceeds configured allocation bound')
    if not isinstance(identities,dict) or json.loads(json.dumps(identities,allow_nan=False)) != identities:
        raise ValueError('canonical frozen identities required')
    for key in ('model','tokenizer','train','development','code'):
        if not identities.get(key):
            raise ValueError(f'missing identity {key}')
    contract = _development_contract(development_evaluator,provenance,identities)
    if [c.document_id for c in caches] != identities['train']['selected_document_ids']:
        raise ValueError('cache order differs from frozen train document identity')
    output=Path(output_dir)
    if output.exists():
        raise FileExistsError('new router output directory required; no resume')
    _frozen(adapter.model)
    controller=AdaptiveFitController(AdaptiveFitConfig.from_dict(config.adaptive),initial_lr=config.learning_rate)
    binding={'format':FORMAT,'identities':copy.deepcopy(identities),'config':asdict(config),'sources':_sources(),
        'cache_documents':cache_identity,'provenance_sha256':_digest(provenance),
        'development_contract':contract,'checkpoint_rule':'minimum full-document development NLL; earliest tie; includes update zero',
        'numeric_protocol':{'input':'whole padded document [1,S,D]','labels':'uniform whole-document action costs',
            'surrogate_is_deployment_cost':False,'training_mask':'only actor objective positions; never FFN/head input slicing',
            'evaluation':'actual mixed-policy whole-document tail and head'},'scope':SCOPE}
    output.mkdir(parents=True)
    work={phase+'_'+suffix:0 for phase in ('table','probe') for suffix in
          ('attempted_document_forwards','completed_document_forwards','head_rows')}
    visits=torch.zeros(count,dtype=torch.long)
    status,failure,stop_reason='running',None,'max_updates'
    best_state=best_train=best_dev=initial_train=latest_train=initial_dev=None
    best_update=successful=stale=0
    best_nll=progress_nll=float('inf')
    history,metrics,optimization_history=[],[],[]
    start=time.perf_counter()
    def coverage():
        return {'eligible_train_positions':count,'optimizer_position_visits':int(visits.sum()),
            'unique_positions_visited':int((visits>0).sum()),'coverage_fraction':float((visits>0).double().mean()),
            'minimum_visits':int(visits.min()),'maximum_visits':int(visits.max())}
    try:
        table,offsets=build_document_cost_table(adapter,caches,config,work)
        binding['table_hashes']=_tensor_hashes(table)
        binding['sha256']=_digest(binding)
        _json(output/'training_contract.json',binding)
        _save(output/'train_surrogate_table.pt',{'format':TABLE_FORMAT,'binding':binding,'provenance':provenance,
            'tensors':{k:v.detach().cpu() for k,v in table.items()},'scope':SCOPE})
        actor=RecoveryPairRouter(table['features'].shape[1],config.epsilon,device=table['features'].device)
        optimizer=torch.optim.AdamW(actor.parameters(),lr=config.learning_rate,weight_decay=0.)
        generator=torch.Generator(device='cpu').manual_seed(config.seed)
        order,cursor=torch.randperm(count,generator=generator),0
        (output/'evaluation_checkpoints').mkdir()
        def record(update,lr_used=None):
            nonlocal best_state,best_train,best_dev,best_update,best_nll,progress_nll,stale,initial_dev
            train=_probe(adapter,caches,actor,table,visits,work)
            with torch.no_grad():
                objective=float(_objective(actor,table['features'],table['native_pairs'],table['costs'],config))
            if not math.isfinite(objective):
                raise ValueError('nonfinite fixed full train surrogate objective')
            decision=None
            if controller.config.enabled:
                decision=controller.observe(update,objective,coverage_complete=bool((visits>0).all()))
                for group in optimizer.param_groups:
                    group['lr']=controller.lr
            dev=_development(development_evaluator,actor,contract)
            if update == 0:
                initial_dev = copy.deepcopy(dev)
            _frozen(adapter.model)
            if dev['nll'] < progress_nll-config.development_min_delta:
                progress_nll,stale=dev['nll'],0
            else:
                stale+=1
            row={'update':update,**dev,'patience_observations':stale}
            history.append(row)
            payload=_payload(actor,binding,update,train,dev,controller,lr_used)
            _save(output/'evaluation_checkpoints'/f'checkpoint_{update:06d}.pt',payload)
            _save(output/'checkpoint_latest.pt',payload)
            if update==0:
                _save(output/'checkpoint_initial.pt',payload)
            if dev['nll'] < best_nll:
                best_nll,best_update,best_train,best_dev=dev['nll'],update,train,dev
                best_state={k:v.detach().clone() for k,v in actor.state_dict().items()}
                _save(output/'checkpoint_best.pt',payload)
            metrics.append({'update':update,'train':train,'fixed_surrogate_objective':objective,
                'learning_rate_used':lr_used,'learning_rate_next':controller.lr,'adaptive_decision':decision,
                'training_coverage':coverage()})
            _json(output/'metrics.json',metrics)
            _json(output/'development_history.json',history)
            return train,bool(update>=config.min_updates and bool((visits>0).all()) and stale>=config.development_patience),bool(decision and decision['stop'])
        initial_train,_,_=record(0)
        latest_train=initial_train
        if initial_train['native_agreement'] != 1.:
            raise ValueError('zero router must preserve native greedy')
        for update in range(1,config.updates+1):
            if cursor==count:
                order,cursor=torch.randperm(count,generator=generator),0
            indices=order[cursor:cursor+config.positions_per_update].to(table['features'].device)
            cursor+=len(indices)
            loss=_objective(actor,table['features'][indices],table['native_pairs'][indices],table['costs'][indices],config)
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite router objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in actor.parameters()):
                raise ValueError('nonfinite or missing router gradients')
            norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),config.gradient_clip,error_if_nonfinite=True)
            lr_used=optimizer.param_groups[0]['lr']
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in actor.parameters()):
                raise ValueError('nonfinite router weights after update')
            _frozen(adapter.model)
            visits.index_add_(0,indices.cpu(),torch.ones(len(indices),dtype=torch.long))
            successful=update
            optimization_history.append({'update':update,'objective_before_update':float(loss.detach()),
                'gradient_norm':float(norm),'learning_rate_used':lr_used,
                'optimizer_positions':len(indices),'training_coverage':coverage()})
            _json(output/'optimization_history.json',optimization_history)
            due=update%config.eval_interval==0 or update==config.updates
            if due:
                latest_train,dev_stop,train_stop=record(update,lr_used)
                if dev_stop or train_stop:
                    stop_reason='development_patience' if dev_stop else 'adaptive_train_plateau'
                    break
        actor.load_state_dict(best_state,strict=True)
        actor.requires_grad_(False).eval()
        status='complete'
    except Exception as error:
        status,failure,stop_reason='failed',f'{type(error).__name__}: {error}','failure'
        raise
    finally:
        summary={'format':FORMAT,'status':status,'failure':failure,'stop_reason':stop_reason,
            'successful_updates':successful,'best_update':best_update,'initial_train_metrics':initial_train,
            'nonzero_gradient_updates':sum(row['gradient_norm']>0 for row in optimization_history),
            'best_train_metrics':best_train,'latest_train_metrics':latest_train,
            'initial_development_nll':history[0]['nll'] if history else None,
            'best_development_nll':best_dev['nll'] if best_dev else None,
            'initial_development_metrics':initial_dev,'best_development_metrics':best_dev,
            'development_history':history,'training_coverage':coverage(),'work_totals':work,
            'adaptive':controller.snapshot(),'binding_sha256':binding.get('sha256'),
            'checkpoint_rule':binding['checkpoint_rule'],'surrogate_is_deployment_cost':False,
            'scope':SCOPE,'final_test_evaluated':False,'resume_supported':False,
            'wall_seconds':time.perf_counter()-start}
        _json(output/'summary.json',summary)
    return actor,summary


def load_document_router_checkpoint(path,identities,device='cpu'):
    path=Path(path)
    if path.stat().st_size>MAX_CHECKPOINT_BYTES:
        raise ValueError('document router checkpoint exceeds size bound')
    snapshot=path.read_bytes()
    value=torch.load(io.BytesIO(snapshot),map_location='cpu',weights_only=True)
    expected={'format','binding','feature_dim','epsilon','update','train_metrics','development_metrics',
        'resume_supported','actor_state_hashes','adaptive_audit_state','learning_rate_used','learning_rate_next',
        'payload_sha256','actor_state_dict'}
    if not isinstance(value,dict) or set(value)!=expected or value['format']!=FORMAT:
        raise ValueError('unsupported document router checkpoint')
    header={k:v for k,v in value.items() if k not in ('payload_sha256','actor_state_dict')}
    if value['payload_sha256']!=_digest(header):
        raise ValueError('checkpoint payload checksum mismatch')
    binding=copy.deepcopy(value['binding'])
    digest=binding.pop('sha256',None)
    if not digest or digest!=_digest(binding) or binding.get('format')!=FORMAT:
        raise ValueError('checkpoint binding checksum or format mismatch')
    if binding.get('identities')!=identities or binding.get('sources')!=_sources():
        raise ValueError('checkpoint identities or source mismatch')
    try:
        config=DocumentRouterConfig(**binding['config'])
        config.validate()
    except (TypeError,KeyError) as error:
        raise ValueError('invalid document router checkpoint config') from error
    _integer(value['update'],'checkpoint update',0,config.updates)
    if value['epsilon']!=config.epsilon or value['resume_supported'] is not False:
        raise ValueError('checkpoint runtime config mismatch')
    actor=RecoveryPairRouter(value['feature_dim'],value['epsilon'],device=device)
    state=value['actor_state_dict']
    if not isinstance(state,dict) or set(state)!=set(actor.state_dict()):
        raise ValueError('invalid router state keys')
    for key,tensor in state.items():
        if (not isinstance(tensor,torch.Tensor) or tensor.shape!=actor.state_dict()[key].shape
                or tensor.dtype!=torch.float32 or not bool(torch.isfinite(tensor).all())):
            raise ValueError('invalid router state tensor')
    if value['actor_state_hashes']!=_tensor_hashes(state):
        raise ValueError('router state checksum mismatch')
    controller=AdaptiveFitController.from_snapshot(value['adaptive_audit_state'])
    if (controller.config!=AdaptiveFitConfig.from_dict(config.adaptive) or controller.initial_lr!=config.learning_rate
            or controller.lr!=value['learning_rate_next'] or controller.last_update is not None and controller.last_update>value['update']):
        raise ValueError('adaptive audit/config mismatch')
    expected_used=config.learning_rate
    for decision in controller.decisions:
        if decision['update']<value['update']:
            expected_used=decision['lr']
    if value['learning_rate_used']!=(expected_used if value['update'] else None):
        raise ValueError('checkpoint learning rate mismatch')
    dev=value['development_metrics']
    if not isinstance(dev,dict) or dev.get('status')!='ok':
        raise ValueError('checkpoint lacks completed development selection')
    _number(dev.get('nll'),'checkpoint development nll')
    for key in ('role','protocol','document_ids'):
        if dev.get(key)!=binding['development_contract'].get(key):
            raise ValueError('checkpoint development contract mismatch')
    actor.load_state_dict(state,strict=True)
    actor.requires_grad_(False).eval()
    metadata={k:copy.deepcopy(v) for k,v in header.items() if k!='binding'}
    metadata.update(path=str(path.resolve()),checkpoint_sha256=hashlib.sha256(snapshot).hexdigest(),
        binding_sha256=digest,training_config=asdict(config),sources=binding['sources'],
        actor_state_sha256=_digest(_tensor_hashes(state)),development_contract=binding['development_contract'],scope=SCOPE)
    return actor,metadata
