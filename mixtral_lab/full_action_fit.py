"""Train-only R1 fit diagnostic with exact expectation over all 28 pairs.

The expensive, label-aware table uses SINGLETON features, routing and scoring.
It is not a deployable oracle: inference uses only ResidualPairRouter and legal
adapter observations. No selection/held-out tensors or evaluator are accepted.
A fit pass is a training result, never a generalization or speed claim.
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

from .adapter import PAIR_TABLE
from .adaptive_fit import AdaptiveFitConfig, AdaptiveFitController
from .training import ResidualPairRouter, pair_actions


FORMAT = 'mixtral-r1-full-action-fit-v1'
TABLE_FORMAT = 'mixtral-r1-full-action-train-table-v1'
SCOPE = ('fixed singleton train counterfactuals; not online/batched native NLL or '
         'actor-routing equivalence, held-out generalization, or deployment speed')
MAX_CONTEXTS = 4096
MAX_UPDATES = 10000
MAX_FEATURE_DIM = 1000000


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be positive' if positive else f'{name} must be nonnegative')


@dataclass(frozen=True)
class FullActionFitConfig:
    updates: int = 100
    batch_contexts: int = 8
    learning_rate: float = .01
    # A deliberate weaker native prior: 8:1 odds at .8, rather than 113:1 at .2.
    epsilon: float = .8
    max_train_contexts: int = 256
    max_score_calls: int = 7680
    gradient_clip: float = 1.
    seed: int = 17
    anchor_atol: float = 1e-6
    fit_gain_threshold: float = .005
    probe_contexts: int = 32
    adaptive: dict | None = None

    def validate(self):
        for name in ('batch_contexts', 'max_train_contexts', 'max_score_calls', 'probe_contexts'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.updates) is not int or self.updates < 0:
            raise ValueError('updates must be a nonnegative integer')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be a nonnegative int64')
        for name in ('learning_rate', 'gradient_clip', 'epsilon', 'fit_gain_threshold'):
            _number(getattr(self, name), name, positive=True)
        _number(self.anchor_atol, 'anchor_atol')
        if self.epsilon >= 1:
            raise ValueError('epsilon must be below 1, preserving initial native greedy')
        if self.batch_contexts > self.max_train_contexts:
            raise ValueError('batch_contexts exceeds max_train_contexts')
        if self.updates > MAX_UPDATES:
            raise ValueError(f'updates exceeds hard bound {MAX_UPDATES}')
        if any(getattr(self, name) > MAX_CONTEXTS for name in
               ('batch_contexts', 'max_train_contexts', 'probe_contexts')):
            raise ValueError(f'context configuration exceeds hard bound {MAX_CONTEXTS}')
        if self.max_score_calls > 30*MAX_CONTEXTS:
            raise ValueError('max_score_calls exceeds hard bound 122880')
        if self.learning_rate > 1.:
            raise ValueError('learning_rate exceeds hard bound 1')
        # Pure validation; the controller never reads selection data or RNG.
        AdaptiveFitController(AdaptiveFitConfig.from_dict(self.adaptive), initial_lr=self.learning_rate)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _tensor_hash(value):
    cpu = value.detach().contiguous().cpu()
    return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),
            'sha256': hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()}


def _tensor_hashes(values):
    return {key: _tensor_hash(value) for key, value in sorted(values.items())}


def _source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('full_action_fit.py', 'adaptive_fit.py', 'adapter.py', 'training.py')}


def _json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _frozen(model):
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('full-action fit requires an eval model with all backbone weights frozen')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('frozen backbone contains gradients')


def _provenance(rows, targets):
    if not isinstance(rows, (list, tuple)) or len(rows) != len(targets):
        raise ValueError('train provenance must identify every context')
    result, seen = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError('train provenance rows must be dictionaries')
        row = json.loads(json.dumps(row, allow_nan=False))
        doc, position = row.get('document_id', row.get('doc_id')), row.get('token_position')
        if (not isinstance(doc, (str, int)) or isinstance(doc, bool) or str(doc) == ''
                or type(position) is not int or position < 0):
            raise ValueError('train provenance needs document_id and nonnegative token_position')
        if row.get('role', 'train') != 'train':
            raise ValueError('only train provenance is allowed, never selection or held-out')
        target = int(targets[index].item())
        if 'target' in row and (type(row['target']) is not int or row['target'] != target):
            raise ValueError('provenance target does not match train tensor')
        key = str(doc), position
        if key in seen:
            raise ValueError('duplicate train context provenance')
        seen.add(key)
        row.update(document_id=str(doc), token_position=position, target=target, role='train')
        result.append(row)
    return result


@torch.no_grad()
def _score_table(adapter, h0, targets, config, work):
    """No cross-shape/cross-identity reuse; every adapter call sees one row."""
    features, native_pairs, costs, anchors = [], [], [], []
    pairs = torch.tensor(PAIR_TABLE, dtype=torch.long, device=h0.device)

    def score(h, y, pair):
        if work['actual_score_calls'] >= config.max_score_calls:
            raise ValueError('full-action score budget exhausted')
        # Charge attempted calls too, including any call that fails validation.
        work['actual_score_calls'] += 1
        work['actual_pair_rows'] += 1
        work['actual_head_calls'] += 1
        work['actual_head_rows'] += 1
        value = adapter.score_pairs(h, y, pair)
        if (not isinstance(value, torch.Tensor) or value.shape != (1,)
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())
                or bool((value < 0).any())):
            raise ValueError('nonfinite, negative or malformed singleton train score')
        return float(value.item())

    for index in range(len(h0)):
        h, y = h0[index:index+1].detach(), targets[index:index+1].detach()
        work['actual_feature_calls'] += 1
        feat = adapter.features(h)
        work['actual_native_pair_calls'] += 1
        native = adapter.native_pair(h)
        if (not isinstance(feat, torch.Tensor) or feat.ndim != 2 or feat.shape[0] != 1
                or not 1 <= feat.shape[1] <= MAX_FEATURE_DIM or not feat.is_floating_point()
                or feat.device != h0.device or not bool(torch.isfinite(feat).all())):
            raise ValueError('invalid legal singleton features')
        if not isinstance(native, torch.Tensor) or native.shape != (1, 2) or native.device != h0.device:
            raise ValueError('invalid singleton native pairs')
        native_id = int(pair_actions(native).item())
        if features and feat.shape[1] != features[0].shape[1]:
            raise ValueError('feature dimension changed across train contexts')
        first = score(h, y, native)
        row = [score(h, y, pair[None]) for pair in pairs]
        repeated = score(h, y, native)
        reference = row[native_id]  # exact fallback is a real candidate, not a clipped gain
        if max(abs(first-reference), abs(repeated-reference)) > config.anchor_atol:
            raise ValueError('native anchor CE mismatch: singleton train protocol inconsistent')
        features.append(feat.detach().float().clone())
        native_pairs.append(native.detach().clone())
        costs.append(row)
        anchors.append([first, repeated])
    return {'features': torch.cat(features), 'native_pairs': torch.cat(native_pairs),
            'costs': torch.tensor(costs, dtype=torch.float64, device=h0.device),
            'native_anchors': torch.tensor(anchors, dtype=torch.float64, device=h0.device)}


def full_action_expected_ce(log_probs, costs):
    """Exact finite-action expectation, not sampled RL or greedy-CE surrogate.

    The table is detached even if an untrusted caller supplies a graph. All 28
    action probabilities contribute gradients; no labels enter actor features.
    """
    if log_probs.ndim != 2 or log_probs.shape[1] != 28 or costs.shape != log_probs.shape:
        raise ValueError('expected [N,28] full-action log probabilities and costs')
    # Normalize in double so roundoff in float32 exp(logp) cannot create
    # an apparent expectation below the minimum of the cost table.
    return (log_probs.double().softmax(-1) * costs.detach().double()).sum(-1).mean()


@torch.no_grad()
def _diagnostics(actor, table, indices=None, *, details=False):
    features, native, costs = (table[key] for key in ('features', 'native_pairs', 'costs'))
    if indices is not None:
        features, native, costs = (value.index_select(0, indices.to(value.device))
                                   for value in (features, native, costs))
    log_probs = actor.log_probs(features, native)
    if not bool(torch.isfinite(log_probs).all()):
        raise ValueError('nonfinite trained router probabilities')
    probs = log_probs.double().softmax(-1)
    native_ids = pair_actions(native)
    greedy_ids = log_probs.argmax(-1)
    oracle_ids = costs.argmin(-1)
    reference = costs.gather(1, native_ids[:, None]).squeeze(1)
    greedy = costs.gather(1, greedy_ids[:, None]).squeeze(1)
    oracle = costs.min(-1).values
    expected = (probs*costs).sum(-1)
    native_logp = log_probs.gather(1, native_ids[:, None]).squeeze(1)
    other = log_probs.clone()
    other.scatter_(1, native_ids[:, None], -torch.inf)
    margin = native_logp-other.max(-1).values
    result = {
        'contexts': len(features), 'native_ce': float(reference.mean()),
        'greedy_ce': float(greedy.mean()), 'expected_ce': float(expected.mean()),
        'oracle_ce': float(oracle.mean()), 'oracle_headroom_nats': float((reference-oracle).mean()),
        'greedy_gain_nats': float((reference-greedy).mean()),
        'expected_gain_vs_native_nats': float((reference-expected).mean()),
        'greedy_oracle_regret_nats': float((greedy-oracle).mean()),
        'expected_oracle_regret_nats': float((expected-oracle).mean()),
        'greedy_change_count': int((greedy_ids != native_ids).sum()),
        'greedy_change_fraction': float((greedy_ids != native_ids).double().mean()),
        'teacher_agreement': float((greedy_ids == native_ids).double().mean()),
        'oracle_agreement': float((greedy_ids == oracle_ids).double().mean()),
        'native_probability_mean': float(probs.gather(1, native_ids[:, None]).mean()),
        'oracle_probability_mean': float(probs.gather(1, oracle_ids[:, None]).mean()),
        'entropy': float(-(probs*log_probs.double()).sum(-1).mean()),
        'native_vs_best_other_logit_margin_mean': float(margin.mean()),
        'greedy_action_counts': torch.bincount(greedy_ids.cpu(), minlength=28).tolist(),
    }
    if details:
        result['rows'] = [
            {'context_index': int(indices[j]) if indices is not None else j,
             'native_action': int(native_ids[j])+1, 'greedy_action': int(greedy_ids[j])+1,
             'oracle_action': int(oracle_ids[j])+1, 'native_ce': float(reference[j]),
             'greedy_ce': float(greedy[j]), 'expected_ce': float(expected[j]),
             'oracle_ce': float(oracle[j]), 'native_vs_best_other_logit_margin': float(margin[j])}
            for j in range(len(features))]
    return result


def _fit_passed(metrics, config):
    return (metrics['greedy_gain_nats'] >= config.fit_gain_threshold
            and metrics['greedy_change_count'] > 0)


def _payload(actor, binding, update, metrics, config, adaptive_state, *, learning_rate_used=None, learning_rate_next=None):
    state = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
    header = {'format': FORMAT, 'binding': binding, 'feature_dim': actor.feature_dim,
              'epsilon': actor.epsilon, 'update': update, 'train_metrics': metrics,
              'train_fit_passed': _fit_passed(metrics, config), 'resume_supported': False,
              'actor_state_hashes': _tensor_hashes(state),
              'adaptive_audit_state': copy.deepcopy(adaptive_state),
              'learning_rate_used': learning_rate_used, 'learning_rate_next': learning_rate_next}
    return {**header, 'payload_sha256': _digest(header), 'actor_state_dict': state}


def train_full_action_router(adapter, train_h0, train_targets, config, identities, output_dir, *, provenance):
    """Fit only supplied TRAIN contexts; return the best training-fit actor.

    The best checkpoint minimizes (greedy train CE, expected train CE), evaluated
    on every fixed train row after every update, including update zero. The
    deterministic probe is diagnostic only and cannot select checkpoints.
    Optional adaptation observes full-train expected CE, never flat greedy CE
    or a minibatch. Only LR/stop decisions may change; the fit gate is unchanged.
    No resume, selection callback, or score-table import/reuse is supported.
    h0 is post-attention/pre-FFN and targets are already next-token aligned.
    """
    config.validate()
    adaptive_config = AdaptiveFitConfig.from_dict(config.adaptive)
    controller = AdaptiveFitController(adaptive_config, initial_lr=config.learning_rate)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError('full-action output must be a new directory; resume is unsupported')
    if (not isinstance(train_h0, torch.Tensor) or train_h0.ndim != 2
            or not train_h0.is_floating_point() or train_h0.shape[1] < 1
            or not bool(torch.isfinite(train_h0).all())):
        raise ValueError('train h0 must be finite floating [N,D]')
    count = len(train_h0)
    if not config.batch_contexts <= count <= config.max_train_contexts:
        raise ValueError('train context count outside bounded configuration')
    if (not isinstance(train_targets, torch.Tensor) or train_targets.shape != (count,)
            or train_targets.dtype != torch.long or train_targets.device != train_h0.device
            or bool((train_targets < 0).any())):
        raise ValueError('train targets must be nonnegative int64 [N] on h0 device')
    required_calls = count*30
    if required_calls > config.max_score_calls:
        raise ValueError(f'full-action table needs {required_calls} singleton score calls, exceeds budget')
    if not isinstance(identities, dict):
        raise ValueError('frozen experiment identities required')
    identities = json.loads(json.dumps(identities, allow_nan=False))
    for key in ('model', 'tokenizer', 'train', 'code'):
        if not identities.get(key):
            raise ValueError(f'missing full-action identity: {key}')
    clean_provenance = _provenance(provenance, train_targets)
    _frozen(adapter.model)
    work = {'required_score_calls': required_calls, 'max_score_calls': config.max_score_calls,
            'actual_score_calls': 0, 'actual_pair_rows': 0, 'actual_head_calls': 0,
            'actual_head_rows': 0, 'actual_feature_calls': 0, 'actual_native_pair_calls': 0,
            'training_score_calls': 0, 'cross_run_cache_hits': 0}
    protocol = {'context_rows_per_call': 1, 'pair_rows_per_call': 1, 'head_rows_per_call': 1,
                'features_rows_per_call': 1, 'native_pair_rows_per_call': 1,
                'candidates_per_context': 28, 'native_anchor_calls_per_context': 2,
                'hidden_dtype': str(train_h0.dtype), 'ce_dtype': 'float32 in MixtralAdapter',
                'table_dtype': 'float64 (preserves returned singleton CE values)',
                'actor_training_rows_per_call_max': config.batch_contexts,
                'actor_train_metrics_rows_per_call': count,
                'actor_probe_rows_per_call': min(count, config.probe_contexts),
                'actor_inference_cross_shape_equivalence_established': False,
                'native_reference': 'exact native candidate in 28-action table',
                'anchor_atol': config.anchor_atol, 'cross_shape_equivalence_established': False,
                'feature_inputs': 'singleton h0 only; no target, oracle action, CE or provenance features',
                'table_reuse': 'this train invocation only; no import/resume or cross-identity reuse'}
    binding = {'format': FORMAT, 'identities': identities, 'config': asdict(config), 'role': 'train',
               'source_sha256': _source_hashes(), 'train_h0': _tensor_hash(train_h0),
               'train_targets': _tensor_hash(train_targets), 'provenance_sha256': _digest(clean_provenance),
               'numeric_protocol': protocol, 'scope': SCOPE,
               'checkpoint_rule': 'minimum (full train greedy CE, full train expected CE); earliest exact tie',
               'adaptive_contract': {'enabled': adaptive_config.enabled,
                    'observation': 'full fixed train expected CE; never greedy CE or minibatch loss',
                    'eval_interval': adaptive_config.eval_interval,
                    'allowed_changes': ['learning_rate', 'early_stop_after_coverage_and_min_updates'],
                    'frozen': ['epsilon', 'batch_contexts', 'features', 'action_set', 'architecture']}}
    output.mkdir(parents=True)
    start = time.perf_counter()
    metrics, best_metrics, initial_metrics, best_state = [], None, None, None
    best_update, successful, best_key = 0, 0, (float('inf'), float('inf'))
    status, failure, stop_reason = 'running', None, 'max_updates'
    adaptive_should_stop = False
    visits = torch.zeros(count, dtype=torch.long)

    def coverage():
        return {'train_table_contexts': count, 'optimizer_context_visits': int(visits.sum()),
                'unique_train_contexts_visited': int((visits > 0).sum()),
                'context_coverage_fraction': float((visits > 0).double().mean()),
                'effective_epochs': float(visits.sum())/count,
                'minimum_context_visits': int(visits.min()), 'maximum_context_visits': int(visits.max())}

    try:
        table = _score_table(adapter, train_h0, train_targets, config, work)
        _frozen(adapter.model)
        binding['score_table_hashes'] = _tensor_hashes(table)
        probe_generator = torch.Generator(device='cpu').manual_seed(config.seed)
        probe_indices = torch.randperm(count, generator=probe_generator)[:min(count, config.probe_contexts)]
        binding['probe_indices'] = probe_indices.tolist()
        binding['sha256'] = _digest(binding)
        _json(output/'training_contract.json', binding)
        cpu_table = {key: value.detach().cpu() for key, value in table.items()}
        _save(output/'train_score_table.pt', {'format': TABLE_FORMAT, 'binding': binding,
              'provenance': clean_provenance, 'targets': train_targets.detach().cpu(),
              'tensors': cpu_table, 'score_work': copy.deepcopy(work)})
        _json(output/'train_score_table.json', {'format': TABLE_FORMAT, 'binding_sha256': binding['sha256'],
              'numeric_protocol': protocol, 'score_work': copy.deepcopy(work), 'contexts': [
                  {'provenance': clean_provenance[i], 'native_action': int(pair_actions(table['native_pairs'][i:i+1]))+1,
                   'native_pair': cpu_table['native_pairs'][i].tolist(),
                   'scores_nats': cpu_table['costs'][i].tolist(),
                   'native_anchors': cpu_table['native_anchors'][i].tolist()}
                  for i in range(count)]})
        actor = ResidualPairRouter(table['features'].shape[1], config.epsilon, device=table['features'].device)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=config.learning_rate, weight_decay=0.)
        order_generator = torch.Generator(device='cpu').manual_seed(config.seed)
        probe_before = _diagnostics(actor, table, probe_indices, details=True)

        def record(update, *, gradient_norm=None, objective=None, learning_rate_used=None):
            nonlocal best_key, best_metrics, best_update, best_state, adaptive_should_stop
            result = _diagnostics(actor, table)
            decision = None
            if adaptive_config.enabled and (update == 0 or update % adaptive_config.eval_interval == 0
                                            or update == config.updates):
                decision = controller.observe(update, result['expected_ce'],
                    coverage_complete=bool((visits > 0).all()))
                for group in optimizer.param_groups:
                    group['lr'] = decision['lr']
                adaptive_should_stop = decision['stop']
            adaptive_state = controller.snapshot()
            row = {'update': update, 'train': result, 'training_coverage': coverage(),
                   'gradient_norm': gradient_norm,
                   'learning_rate_used': learning_rate_used,
                   'learning_rate_next': optimizer.param_groups[0]['lr'],
                   'adaptive_decision': decision,
                   'minibatch_expected_ce_before_update': objective}
            metrics.append(row)
            candidate_key = result['greedy_ce'], result['expected_ce']
            if candidate_key < best_key:
                best_key, best_metrics, best_update = candidate_key, result, update
                best_state = {key: value.detach().clone() for key, value in actor.state_dict().items()}
                _save(output/'checkpoint_best.pt', _payload(actor, binding, update, result, config, adaptive_state,
                    learning_rate_used=learning_rate_used, learning_rate_next=optimizer.param_groups[0]['lr']))
            _save(output/'checkpoint_latest.pt', _payload(actor, binding, update, result, config, adaptive_state,
                    learning_rate_used=learning_rate_used, learning_rate_next=optimizer.param_groups[0]['lr']))
            _json(output/'metrics.json', metrics)
            return result

        initial_metrics = record(0)
        if initial_metrics['greedy_change_count'] != 0:
            raise ValueError('initial actor must retain exact singleton native greedy actions')
        _save(output/'checkpoint_initial.pt', _payload(actor, binding, 0, initial_metrics, config, controller.snapshot(),
              learning_rate_next=optimizer.param_groups[0]['lr']))
        order, cursor = torch.randperm(count, generator=order_generator), 0
        for update in range(1, config.updates+1):
            if cursor == count:
                order, cursor = torch.randperm(count, generator=order_generator), 0
            indices = order[cursor:cursor+config.batch_contexts].to(table['features'].device)
            cursor += len(indices)  # include the final short batch, never discard train rows
            features = table['features'].index_select(0, indices)
            native = table['native_pairs'].index_select(0, indices)
            costs = table['costs'].index_select(0, indices)
            log_probs = actor.log_probs(features, native)
            loss = full_action_expected_ce(log_probs, costs)
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite full-action expected-CE objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in actor.parameters()):
                raise ValueError('nonfinite or missing full-action actor gradients')
            norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), config.gradient_clip, error_if_nonfinite=True)
            learning_rate_used = optimizer.param_groups[0]['lr']
            optimizer.step()
            _frozen(adapter.model)
            visits.index_add_(0, indices.cpu(), torch.ones(len(indices), dtype=torch.long))
            record(update, gradient_norm=float(norm), objective=float(loss.detach()),
                   learning_rate_used=learning_rate_used)
            successful = update
            if adaptive_should_stop:
                stop_reason = 'adaptive_plateau'
                break
        probe_latest = _diagnostics(actor, table, probe_indices, details=True)
        actor.load_state_dict(best_state, strict=True)
        actor.eval()
        probe_best = _diagnostics(actor, table, probe_indices, details=True)
        _json(output/'train_probe.json', {'role': 'train', 'scope': SCOPE, 'indices': probe_indices.tolist(),
              'binding_sha256': binding['sha256'], 'before': probe_before, 'best': probe_best,
              'latest': probe_latest, 'best_update': best_update, 'latest_update': successful,
              'selection_role': 'diagnostic only; full fixed train set selects checkpoint'})
        status = 'complete'
    except Exception as error:
        status, failure, stop_reason = 'failed', f'{type(error).__name__}: {error}', 'failure'
        raise
    finally:
        summary = {'format': FORMAT, 'status': status, 'failure': failure,
                   'stop_reason': stop_reason, 'adaptive': controller.snapshot(),
                   'successful_updates': successful, 'best_update': best_update,
                   'train_fit_passed': status == 'complete' and _fit_passed(best_metrics, config),
                   'fit_gain_threshold_nats': config.fit_gain_threshold,
                   'fit_gate': 'best greedy train CE gain vs native >= threshold AND actual greedy change count > 0',
                   'initial_train_metrics': initial_metrics, 'best_train_metrics': best_metrics,
                   'latest_train_metrics': metrics[-1]['train'] if metrics else None,
                   'binding_sha256': binding.get('sha256'), 'score_work_totals': work,
                   'training_coverage': coverage(),
                   'wall_seconds': time.perf_counter()-start, 'scope': SCOPE,
                   'held_out_confirmation': False, 'resume_supported': False,
                   'expected_ce_is_not_greedy_ce': True}
        _json(output/'summary.json', summary)
    return actor, summary


def load_full_action_checkpoint(path, identities, device='cpu'):
    """Safe inference-only load, bound to exact identities/source/config/tensors.

    Embedded hashes detect accidental corruption, not an attacker who rewrites
    all hashes. Use trusted experiment artifacts; weights_only forbids objects.
    No optimizer, train table, labels, or RNG state is attached to the actor.
    """
    snapshot = Path(path).read_bytes()
    payload = torch.load(io.BytesIO(snapshot), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != FORMAT:
        raise ValueError('unsupported full-action checkpoint format')
    header = {key: value for key, value in payload.items() if key not in ('payload_sha256', 'actor_state_dict')}
    if payload.get('payload_sha256') != _digest(header):
        raise ValueError('full-action checkpoint payload checksum mismatch')
    binding = copy.deepcopy(payload.get('binding', {}))
    checksum = binding.pop('sha256', None)
    if not checksum or checksum != _digest(binding):
        raise ValueError('full-action checkpoint binding checksum mismatch')
    if binding.get('identities') != identities:
        raise ValueError('full-action checkpoint experiment identities mismatch')
    if binding.get('source_sha256') != _source_hashes():
        raise ValueError('full-action checkpoint source checksum mismatch')
    if binding.get('role') != 'train' or binding.get('format') != FORMAT:
        raise ValueError('full-action checkpoint must bind train-only fit')
    config = FullActionFitConfig(**binding['config'])
    config.validate()
    if payload.get('epsilon') != config.epsilon or payload.get('resume_supported') is not False:
        raise ValueError('full-action checkpoint configuration mismatch')
    update = payload.get('update')
    if type(update) is not int or not 0 <= update <= config.updates:
        raise ValueError('invalid full-action checkpoint update')
    state = payload.get('actor_state_dict')
    if (not isinstance(state, dict) or set(state) != {'weight', 'bias'}
            or any(not isinstance(value, torch.Tensor) for value in state.values())
            or payload.get('actor_state_hashes') != _tensor_hashes(state)):
        raise ValueError('full-action checkpoint actor state checksum mismatch')
    feature_dim = payload.get('feature_dim')
    if type(feature_dim) is not int or not 1 <= feature_dim <= MAX_FEATURE_DIM:
        raise ValueError('invalid full-action checkpoint feature dimension')
    # Reject header/shape mismatches before allocating an actor from metadata.
    for key, shape in (('weight', (28, feature_dim)), ('bias', (28,))):
        if state[key].shape != shape or state[key].dtype != torch.float32 or not bool(torch.isfinite(state[key]).all()):
            raise ValueError('full-action checkpoint malformed or nonfinite actor state')
    _number(payload.get('learning_rate_next'), 'checkpoint next learning rate', positive=True)
    if update:
        _number(payload.get('learning_rate_used'), 'checkpoint producing learning rate', positive=True)
    audit = payload.get('adaptive_audit_state')
    audited_controller = AdaptiveFitController.from_snapshot(audit)
    expected_adaptive = AdaptiveFitConfig.from_dict(config.adaptive)
    if (audited_controller.config != expected_adaptive
            or audit['initial_lr'] != config.learning_rate
            or audit['current_lr'] != payload.get('learning_rate_next')):
        raise ValueError('full-action checkpoint adaptive configuration mismatch')
    if audit['last_update'] is not None and audit['last_update'] > update:
        raise ValueError('full-action checkpoint adaptive state comes from a later update')
    expected_used_lr = config.learning_rate
    for event in audit['decisions']:
        if event['update'] < update:
            expected_used_lr = event['lr']
    if payload.get('learning_rate_used') != (expected_used_lr if update else None):
        raise ValueError('full-action checkpoint producing learning rate mismatch')
    actor = ResidualPairRouter(feature_dim, config.epsilon, device=device)
    actor.load_state_dict(state, strict=True)
    metrics = payload['train_metrics']
    if payload.get('train_fit_passed') is not _fit_passed(metrics, config):
        raise ValueError('full-action checkpoint fit verdict mismatch')
    metadata = {'path': str(Path(path).resolve()), 'checkpoint_sha256': hashlib.sha256(snapshot).hexdigest(),
                'binding_sha256': checksum, 'actor_state_sha256': _digest(_tensor_hashes(state)),
                'update': update, 'epsilon': config.epsilon, 'training_config': asdict(config),
                'train_metrics': copy.deepcopy(metrics), 'train_fit_passed': payload['train_fit_passed'],
                'source_sha256': binding['source_sha256'], 'scope': SCOPE,
                'adaptive_audit_state': copy.deepcopy(payload['adaptive_audit_state']),
                'learning_rate_used': payload['learning_rate_used'], 'learning_rate_next': payload['learning_rate_next'],
                'score_table_hashes': binding['score_table_hashes'], 'resume_supported': False}
    return actor.eval(), metadata
