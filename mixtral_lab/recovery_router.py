"""Isolated R1 recovery experiment: cost supervision, development-only selection.

Gold-token costs are fixed SINGLETON training labels, never inference inputs.
The soft-target loss has logit gradient p-q even for a badly saturated policy.
The native regularizer is KL(pi || static full-support native prior), NOT step KL.
Step KL is reported independently and does not silently act as a trust region.
Only distinct, complete development documents can select a checkpoint; update
zero participates. This module never accepts selection/final-test evaluators.
Defaults are provisional research settings, not established hyperparameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch
from torch import nn
from torch.nn import functional as F

from .adaptive_fit import AdaptiveFitConfig, AdaptiveFitController
from .full_action_fit import (
    _digest, _tensor_hash, _tensor_hashes, _score_table, _provenance, _frozen,
    full_action_expected_ce,
)
from .training import pair_actions

FORMAT = 'mixtral-r1-recovery-router-v1'
TABLE_FORMAT = 'mixtral-r1-recovery-train-table-v1'
DEVELOPMENT_PROTOCOL = 'full_document_online_nll'
MAX_CONTEXTS = 4096
MAX_UPDATES = 10000
MAX_DEVELOPMENT_EVALUATIONS = 1024
MAX_HIDDEN_SIZE = 65536
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
SCOPE = ('fixed singleton training supervision; full-document online development '
         'checkpoint selection; no final-test or deployment-speed claim')


def _number(value, name, lower=0., upper=1e6, *, strict=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value > upper
            or (value <= lower if strict else value < lower)):
        raise ValueError(f'{name} must be finite in {"(" if strict else "["}{lower},{upper}]')


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f'{name} must be integer in [{lower},{upper}]')


@dataclass(frozen=True)
class RecoveryRouterConfig:
    updates: int = 256
    batch_contexts: int = 8
    learning_rate: float = 1e-3
    epsilon: float = .8
    temperature: float = .1
    reference_kl_weight: float = .01
    objective: str = 'soft_target_ce'
    max_train_contexts: int = 256
    max_score_calls: int = 7680
    eval_interval: int = 16
    development_patience: int = 5
    min_updates: int = 32
    development_min_delta: float = 1e-4
    gradient_clip: float = 1.
    anchor_atol: float = 1e-6
    fit_gain_threshold: float = .005
    fit_min_positive_fraction: float = .5
    fit_min_positive_contexts: int = 3
    seed: int = 17
    adaptive: dict | None = None

    def validate(self):
        _integer(self.updates, 'updates', 0, MAX_UPDATES)
        for name in ('batch_contexts', 'max_train_contexts', 'fit_min_positive_contexts'):
            _integer(getattr(self, name), name, 1, MAX_CONTEXTS)
        for name in ('eval_interval', 'development_patience'):
            _integer(getattr(self, name), name, 1, MAX_UPDATES)
        _integer(self.min_updates, 'min_updates', 0, MAX_UPDATES)
        _integer(self.max_score_calls, 'max_score_calls', 1, 30 * MAX_CONTEXTS)
        _integer(self.seed, 'seed', 0, 2**63 - 1)
        for name, upper in (('learning_rate', 1.), ('epsilon', 1.),
                            ('temperature', 100.), ('gradient_clip', 1e6),
                            ('fit_gain_threshold', 100.)):
            _number(getattr(self, name), name, upper=upper, strict=True)
        for name, upper in (('reference_kl_weight', 100.), ('anchor_atol', 1.),
                            ('development_min_delta', 100.), ('fit_min_positive_fraction', 1.)):
            _number(getattr(self, name), name, upper=upper)
        if self.temperature < 1e-6:
            raise ValueError('temperature must be at least 1e-6')
        if self.epsilon >= 1:
            raise ValueError('epsilon must be below 1 to preserve native greedy at zero')
        if 1 + math.ceil(self.updates/self.eval_interval) > MAX_DEVELOPMENT_EVALUATIONS:
            raise ValueError('development evaluation checkpoint budget exceeds 1024')
        if self.batch_contexts > self.max_train_contexts:
            raise ValueError('batch_contexts exceeds max_train_contexts')
        if self.fit_min_positive_contexts < 3:
            raise ValueError('fit needs at least three positive contexts, not two-token gains')
        if self.objective not in ('soft_target_ce', 'expected_ce'):
            raise ValueError('objective must be soft_target_ce or expected_ce')
        AdaptiveFitController(AdaptiveFitConfig.from_dict(self.adaptive), initial_lr=self.learning_rate)


def compact_r1_features(features):
    """Keep normalized h (n) and the eight centered gate logits, nothing else.

    Adapter width is 2D+20. In R1 n0 duplicates n; previous-pair and round/drift
    fields are constants. Reject R2 observations rather than discard real state.
    Row RMS normalization is retained, then residual logits receive 1/sqrt(D+8).
    """
    if (not isinstance(features, torch.Tensor) or features.ndim != 2
            or not features.is_floating_point() or features.shape[1] < 22
            or (features.shape[1] - 20) % 2
            or not bool(torch.isfinite(features).all())):
        raise ValueError('expected finite R1 adapter features [N,2D+20]')
    hidden = (features.shape[1] - 20) // 2
    if hidden > MAX_HIDDEN_SIZE:
        raise ValueError('hidden size exceeds recovery router bound')
    value = features.detach().float()
    expected = value.new_tensor([0., 1., 1., 0.])
    if (not torch.equal(value[:, :hidden], value[:, hidden:2*hidden])
            or bool((value[:, 2*hidden+8:2*hidden+16] != 0).any())
            or bool((value[:, -4:] != expected).any())):
        raise ValueError('recovery router accepts first-round R1 observations only')
    compact = torch.cat((value[:, :hidden], value[:, 2*hidden:2*hidden+8]), -1)
    # Double accumulation prevents overflow for finite but pathological inputs.
    scale = torch.rsqrt(compact.double().square().mean(-1, keepdim=True) + 1e-6)
    return (compact.double() * scale).float()


class RecoveryPairRouter(nn.Module):
    """Inference-only inputs match MixtralAdapter; labels/costs never enter here."""
    def __init__(self, feature_dim, epsilon=.8, *, device=None):
        super().__init__()
        _integer(feature_dim, 'feature_dim', 22, 2 * MAX_HIDDEN_SIZE + 20)
        if (feature_dim - 20) % 2:
            raise ValueError('feature_dim must equal 2D+20')
        _number(epsilon, 'epsilon', upper=1., strict=True)
        if epsilon >= 1:
            raise ValueError('epsilon must be below 1')
        self.feature_dim, self.epsilon = feature_dim, float(epsilon)
        self.hidden_size = (feature_dim - 20) // 2
        self.compact_feature_dim = self.hidden_size + 8
        self.residual_scale = 1. / math.sqrt(self.compact_feature_dim)
        self.weight = nn.Parameter(torch.zeros(28, self.compact_feature_dim, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.zeros(28, dtype=torch.float32, device=device))

    def reference_log_probs(self, native_pairs):
        actions = pair_actions(native_pairs)
        prior = torch.full((len(actions), 28), self.epsilon/28,
                           dtype=torch.float32, device=native_pairs.device)
        prior.scatter_add_(1, actions[:, None], prior.new_full((len(actions), 1), 1-self.epsilon))
        return prior.log()

    def logits(self, features, native_pairs):
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError('recovery router feature shape mismatch')
        if (not isinstance(native_pairs, torch.Tensor) or native_pairs.shape != (len(features), 2)
                or native_pairs.device != features.device or features.device != self.weight.device):
            raise ValueError('native pair rows/device must match router features')
        compact = compact_r1_features(features)
        residual = F.linear(compact, self.weight, self.bias) * self.residual_scale
        logits = self.reference_log_probs(native_pairs) + residual
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('nonfinite recovery router logits')
        return logits

    def log_probs(self, features, native_pairs):
        return self.logits(features, native_pairs).log_softmax(-1)

    @torch.no_grad()
    def greedy(self, features, native_pairs):
        return self.log_probs(features, native_pairs).argmax(-1) + 1

    def forward(self, observation):
        return self.greedy(observation.features, observation.native_pairs)


def cost_soft_targets(costs, temperature):
    _number(temperature, 'temperature', upper=100., strict=True)
    if temperature < 1e-6:
        raise ValueError('temperature must be at least 1e-6')
    if (not isinstance(costs, torch.Tensor) or costs.ndim != 2 or costs.shape[1] != 28
            or not len(costs) or not costs.is_floating_point()
            or not bool(torch.isfinite(costs).all()) or bool((costs < 0).any())):
        raise ValueError('costs must be finite nonnegative [N,28]')
    costs = costs.detach().double()
    return (-(costs-costs.min(-1, keepdim=True).values)/temperature).softmax(-1)


def soft_target_cross_entropy(log_probs, costs, temperature=.1):
    """Stable CE(q,pi), not CE over pi probabilities; labels always detached."""
    if (not isinstance(log_probs, torch.Tensor) or log_probs.shape != costs.shape
            or not log_probs.is_floating_point() or not bool(torch.isfinite(log_probs).all())
            or log_probs.device != costs.device):
        raise ValueError('finite matching log probabilities and costs required')
    return -(cost_soft_targets(costs, temperature)*log_probs.double()).sum(-1).mean()


def native_reference_kl(log_probs, reference_log_probs):
    """KL(current pi || fixed smoothed native p0), never KL to previous pi."""
    if (log_probs.shape != reference_log_probs.shape or log_probs.ndim != 2
            or log_probs.shape[1] != 28 or not len(log_probs)
            or not bool(torch.isfinite(log_probs).all())
            or not bool(torch.isfinite(reference_log_probs).all())):
        raise ValueError('finite [N,28] policy and reference log probabilities required')
    logp = log_probs.double().log_softmax(-1)
    reference = reference_log_probs.detach().double().log_softmax(-1)
    return (logp.exp()*(logp-reference)).sum(-1).mean()


def _objective(actor, features, native, costs, config):
    logp = actor.log_probs(features, native)
    base = (soft_target_cross_entropy(logp, costs, config.temperature)
            if config.objective == 'soft_target_ce' else full_action_expected_ce(logp, costs))
    reference_kl = native_reference_kl(logp, actor.reference_log_probs(native))
    return base + config.reference_kl_weight*reference_kl


def _source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('recovery_router.py', 'full_action_fit.py', 'adaptive_fit.py', 'adapter.py', 'training.py',
             'recovery_protocol.py', 'bridge.py')}


def _atomic_write(path, writer):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        writer(Path(name))
        with open(name, 'rb') as handle:
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _json(path, value):
    serialized = json.dumps(value, indent=2, allow_nan=False) + '\n'
    _atomic_write(path, lambda temporary: temporary.write_text(serialized))


def _save(path, value):
    _atomic_write(path, lambda temporary: torch.save(value, temporary))


def _identity_document_ids(identities, role):
    value = identities.get(role)
    docs = value.get('selected_document_ids') if isinstance(value, dict) else None
    if (not isinstance(docs, list) or not docs or len(docs) > MAX_CONTEXTS
            or any(not isinstance(doc, str) or not doc for doc in docs)
            or len(set(docs)) != len(docs)):
        raise ValueError(f'{role} identity needs unique nonempty selected_document_ids')
    return docs


def _development_contract(evaluator, train_provenance, identities):
    if not callable(evaluator):
        raise ValueError('a development-only evaluator is required')
    contract = getattr(evaluator, 'contract', None)
    if not isinstance(contract, dict):
        raise ValueError('development evaluator must expose its document/protocol contract')
    contract = json.loads(json.dumps(contract, allow_nan=False))
    if contract.get('role') != 'development' or contract.get('protocol') != DEVELOPMENT_PROTOCOL:
        raise ValueError('evaluator must use full_document_online_nll on development only')
    docs = contract.get('document_ids')
    if (not isinstance(docs, list) or not docs or len(docs) > MAX_CONTEXTS
            or any(not isinstance(doc, str) or not doc for doc in docs)
            or len(docs) != len(set(docs))):
        raise ValueError('development requires unique nonempty document_ids')
    train_docs = _identity_document_ids(identities, 'train')
    development_docs = _identity_document_ids(identities, 'development')
    if docs != development_docs:
        raise ValueError('development evaluator document_ids must match frozen identity')
    if set(train_docs) & set(development_docs):
        raise ValueError('train and development documents must be disjoint')
    if not {row['document_id'] for row in train_provenance} <= set(train_docs):
        raise ValueError('train provenance documents do not match frozen train identity')
    return contract


def _development(evaluator, actor, contract):
    before = actor.training
    before_state = _tensor_hashes(actor.state_dict())
    config_fields = ('feature_dim', 'compact_feature_dim', 'hidden_size', 'epsilon', 'residual_scale')
    before_config = tuple(getattr(actor, key) for key in config_fields)
    try:
        actor.eval()
        with torch.no_grad():
            result = evaluator(actor)
    finally:
        actor.train(before)
    if (before_state != _tensor_hashes(actor.state_dict())
            or before_config != tuple(getattr(actor, key) for key in config_fields)):
        raise ValueError('development evaluator mutated router weights or configuration')
    if not isinstance(result, dict):
        raise ValueError('development evaluator must return metrics dictionary')
    result = json.loads(json.dumps(result, allow_nan=False))
    _number(result.get('nll'), 'development nll')
    if result.get('status') != 'ok':
        raise ValueError('development evaluator did not report successful completion')
    for key in ('role', 'protocol', 'document_ids'):
        if result.get(key) != contract[key]:
            raise ValueError(f'development result contract mismatch: {key}')
    if getattr(evaluator, 'contract', None) != contract:
        raise ValueError('development evaluator changed its fixed document contract')
    return result


@torch.no_grad()
def _diagnostics(actor, table, *, previous_log_probs=None):
    features, native, costs = (table[key] for key in ('features', 'native_pairs', 'costs'))
    logits = actor.logits(features, native)
    policy_logp = logits.log_softmax(-1)
    # Argmax must use the deployed FP32 path: a double log-softmax can
    # break ties that the deployed FP32 log-softmax rounds together.
    native_ids, greedy_ids = pair_actions(native), policy_logp.argmax(-1)
    logp = policy_logp.double().log_softmax(-1)
    probs = logp.exp()
    oracle_ids = costs.argmin(-1)
    reference = costs.gather(1, native_ids[:, None]).squeeze(1)
    greedy = costs.gather(1, greedy_ids[:, None]).squeeze(1)
    oracle = costs.min(-1).values
    expected = (probs*costs).sum(-1)
    gain = reference-greedy
    positive = gain.clamp_min(0)
    positive_sum = float(positive.sum())
    step_kl = (float((probs*(logp-previous_log_probs.double())).sum(-1).mean())
               if previous_log_probs is not None else 0.)
    metrics = {
        'contexts': len(features), 'native_ce': float(reference.mean()),
        'greedy_ce': float(greedy.mean()), 'expected_ce': float(expected.mean()),
        'oracle_ce': float(oracle.mean()), 'oracle_headroom_nats': float((reference-oracle).mean()),
        'greedy_gain_nats': float(gain.mean()), 'gain_mean_nats': float(gain.mean()),
        'gain_median_nats': float(gain.quantile(.5)), 'gain_worst_nats': float(gain.min()),
        'gain_positive_fraction': float((gain > 0).double().mean()),
        'gain_positive_count': int((gain > 0).sum()),
        'top2_positive_gain_share': float(positive.topk(min(2, len(gain))).values.sum())/positive_sum if positive_sum else 0.,
        'greedy_oracle_regret_nats': float((greedy-oracle).mean()),
        'expected_oracle_regret_nats': float((expected-oracle).mean()),
        'greedy_change_count': int((greedy_ids != native_ids).sum()),
        'greedy_change_fraction': float((greedy_ids != native_ids).double().mean()),
        'native_agreement': float((greedy_ids == native_ids).double().mean()),
        'oracle_agreement': float((greedy_ids == oracle_ids).double().mean()),
        'entropy': float(-(probs*logp).sum(-1).mean()),
        'max_abs_logit': float(logits.abs().max()), 'max_logit': float(logits.max()),
        'max_logit_span': float((logits.max(-1).values-logits.min(-1).values).max()),
        'max_probability': float(probs.max()),
        'probability_saturation_fraction': float((probs.max(-1).values >= .999).double().mean()),
        'near_zero_action_probability_fraction': float((probs <= 1e-6).double().mean()),
        'native_reference_kl': float(native_reference_kl(logp, actor.reference_log_probs(native))),
        'step_kl_current_to_previous': step_kl,
        'greedy_action_counts': torch.bincount(greedy_ids.cpu(), minlength=28).tolist(),
    }
    # Fail closed before serializing metrics, including any later added fields.
    json.dumps(metrics, allow_nan=False)
    return metrics, logp.detach()


def _fit_passed(metrics, config):
    return bool(metrics and metrics['gain_mean_nats'] >= config.fit_gain_threshold
                and metrics['gain_positive_fraction'] >= config.fit_min_positive_fraction
                and metrics['gain_positive_count'] >= config.fit_min_positive_contexts)


def _payload(actor, binding, update, train_metrics, development_metrics, config, controller,
             learning_rate_used, learning_rate_next):
    state = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
    header = {'format': FORMAT, 'binding': binding, 'feature_dim': actor.feature_dim,
              'compact_feature_dim': actor.compact_feature_dim, 'residual_scale': actor.residual_scale,
              'epsilon': actor.epsilon, 'update': update, 'train_metrics': train_metrics,
              'development_metrics': development_metrics,
              'train_fit_passed': _fit_passed(train_metrics, config), 'resume_supported': False,
              'actor_state_hashes': _tensor_hashes(state),
              'adaptive_audit_state': controller.snapshot(),
              'learning_rate_used': learning_rate_used, 'learning_rate_next': learning_rate_next}
    return {**header, 'payload_sha256': _digest(header), 'actor_state_dict': state}


def train_recovery_router(adapter, train_h0, train_targets, development_evaluator,
                          config, identities, output_dir, *, provenance):
    """Return the best full-document development-NLL actor (including update 0).

    The evaluator is callable(actor) with a fixed .contract mapping containing
    role='development', protocol='full_document_online_nll', document_ids=[...].
    Its result repeats those fields plus status='ok', finite nonnegative nll; the runner owns
    full-document execution. No train labels/tables are passed to the evaluator.
    Train metrics diagnose fit only. Optional LR adaptation sees a fixed full
    TRAIN objective; development patience controls independent early stopping.
    """
    if not isinstance(config, RecoveryRouterConfig):
        raise ValueError('RecoveryRouterConfig required')
    config.validate()
    controller = AdaptiveFitController(AdaptiveFitConfig.from_dict(config.adaptive), initial_lr=config.learning_rate)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError('recovery output must be new; resume is unsupported')
    if (not isinstance(train_h0, torch.Tensor) or train_h0.ndim != 2
            or not train_h0.is_floating_point() or not 1 <= train_h0.shape[1] <= MAX_HIDDEN_SIZE
            or not bool(torch.isfinite(train_h0).all())):
        raise ValueError('train h0 must be finite floating [N,D]')
    count = len(train_h0)
    if not config.batch_contexts <= count <= config.max_train_contexts:
        raise ValueError('train context count outside bounded configuration')
    if (not isinstance(train_targets, torch.Tensor) or train_targets.shape != (count,)
            or train_targets.dtype != torch.long or train_targets.device != train_h0.device
            or bool((train_targets < 0).any())):
        raise ValueError('train targets must be nonnegative int64 [N] on h0 device')
    required_calls = 30*count
    if required_calls > config.max_score_calls:
        raise ValueError(f'full-action table needs {required_calls} singleton score calls, exceeds budget')
    if not isinstance(identities, dict):
        raise ValueError('frozen experiment identities required')
    clean_identities = json.loads(json.dumps(identities, allow_nan=False))
    if clean_identities != identities:
        raise ValueError('experiment identities must contain canonical JSON values')
    identities = clean_identities
    for key in ('model', 'tokenizer', 'train', 'development', 'code'):
        if not identities.get(key):
            raise ValueError(f'missing recovery identity: {key}')
    clean_provenance = _provenance(provenance, train_targets)
    development_contract = _development_contract(development_evaluator, clean_provenance, identities)
    _frozen(adapter.model)
    work = {'required_score_calls': required_calls, 'max_score_calls': config.max_score_calls,
            'actual_score_calls': 0, 'actual_pair_rows': 0, 'actual_head_calls': 0,
            'actual_head_rows': 0, 'actual_feature_calls': 0, 'actual_native_pair_calls': 0,
            'training_score_calls': 0, 'cross_run_cache_hits': 0}
    binding = {'format': FORMAT, 'identities': identities, 'config': asdict(config),
               'role': 'train_with_development_selection', 'source_sha256': _source_hashes(),
               'train_h0': _tensor_hash(train_h0), 'train_targets': _tensor_hash(train_targets),
               'provenance_sha256': _digest(clean_provenance), 'development_contract': development_contract,
               'numeric_protocol': {'context_rows_per_call': 1, 'candidates_per_context': 28,
                   'native_anchor_calls_per_context': 2, 'anchor_atol': config.anchor_atol,
                   'train_table_dtype': 'float64 preserves singleton scores',
                   'feature_layout': 'R1 n[D] + centered_gate[8]; row RMS; residual/sqrt(D+8)',
                   'cross_shape_equivalence_established': False,
                   'feature_inputs': 'h0 only; no target, cost, oracle action or provenance',
                   'static_reference_kl': 'KL(pi_current || p0_native_smoothed)',
                   'step_kl': 'KL(pi_current || pi_previous_update); diagnostic only'},
               'checkpoint_rule': 'minimum full-document development nll; earliest exact tie; includes update zero',
               'adaptive_observation': 'fixed full-train objective only; never development or minibatch loss',
               'scope': SCOPE}
    output.mkdir(parents=True)
    start = time.perf_counter()
    visits = torch.zeros(count, dtype=torch.long)
    metrics, dev_history = [], []
    best_state, best_metrics, best_dev, initial_metrics, latest_metrics = None, None, None, None, None
    best_update, successful, stale = 0, 0, 0
    best_nll, progress_nll = float('inf'), float('inf')
    status, failure, stop_reason = 'running', None, 'max_updates'
    previous_logp = None

    def coverage():
        return {'train_table_contexts': count, 'optimizer_context_visits': int(visits.sum()),
                'unique_train_contexts_visited': int((visits > 0).sum()),
                'context_coverage_fraction': float((visits > 0).double().mean()),
                'effective_epochs': float(visits.sum())/count,
                'minimum_context_visits': int(visits.min()), 'maximum_context_visits': int(visits.max())}

    try:
        table = _score_table(adapter, train_h0, train_targets, config, work)
        compact_r1_features(table['features'])
        if table['features'].shape[1] != 2*train_h0.shape[1]+20:
            raise ValueError('adapter feature width must be 2*hidden_size+20')
        _frozen(adapter.model)
        binding['score_table_hashes'] = _tensor_hashes(table)
        binding['sha256'] = _digest(binding)
        _json(output/'training_contract.json', binding)
        _save(output/'train_score_table.pt', {'format': TABLE_FORMAT, 'binding': binding,
              'provenance': clean_provenance, 'targets': train_targets.detach().cpu(),
              'tensors': {key: value.detach().cpu() for key, value in table.items()},
              'score_work': copy.deepcopy(work)})
        actor = RecoveryPairRouter(table['features'].shape[1], config.epsilon, device=train_h0.device)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=config.learning_rate, weight_decay=0.)
        generator = torch.Generator(device='cpu').manual_seed(config.seed)
        (output/'evaluation_checkpoints').mkdir()

        def record(update, *, gradient_norm=None, objective=None, learning_rate_used=None, force_dev=False):
            nonlocal previous_logp, best_nll, progress_nll, best_update, best_state, best_metrics, best_dev, stale
            result, previous_logp = _diagnostics(actor, table, previous_log_probs=previous_logp)
            with torch.no_grad():
                train_objective = float(_objective(actor, table['features'], table['native_pairs'], table['costs'], config))
            if not math.isfinite(train_objective):
                raise ValueError('nonfinite fixed train objective')
            decision = None
            if controller.config.enabled and (update == 0 or update % controller.config.eval_interval == 0
                                               or update == config.updates):
                decision = controller.observe(update, train_objective, coverage_complete=bool((visits > 0).all()))
                for group in optimizer.param_groups:
                    group['lr'] = decision['lr']
            should_eval = force_dev or update == 0 or update % config.eval_interval == 0 or update == config.updates
            dev = None
            if should_eval:
                dev = _development(development_evaluator, actor, development_contract)
                _frozen(adapter.model)
                if dev['nll'] < progress_nll-config.development_min_delta:
                    progress_nll, stale = dev['nll'], 0
                else:
                    # Count all post-baseline checks; min_updates/coverage
                    # gate stopping, not the accumulated patience evidence.
                    stale += 1
                row = {'update': update, **dev, 'patience_observations': stale}
                dev_history.append(row)
                payload = _payload(actor, binding, update, result, dev, config, controller,
                                   learning_rate_used, optimizer.param_groups[0]['lr'])
                _save(output/'evaluation_checkpoints'/f'checkpoint_{update:06d}.pt', payload)
                if dev['nll'] < best_nll:
                    best_nll, best_update, best_metrics, best_dev = dev['nll'], update, result, dev
                    best_state = {key: value.detach().clone() for key, value in actor.state_dict().items()}
                    _save(output/'checkpoint_best.pt', payload)
                _json(output/'development_history.json', dev_history)
            payload = _payload(actor, binding, update, result, dev, config, controller,
                               learning_rate_used, optimizer.param_groups[0]['lr'])
            _save(output/'checkpoint_latest.pt', payload)
            if update == 0:
                _save(output/'checkpoint_initial.pt', payload)
            metrics.append({'update': update, 'train': result, 'fixed_train_objective': train_objective,
                            'minibatch_objective_before_update': objective, 'gradient_norm': gradient_norm,
                            'learning_rate_used': learning_rate_used,
                            'learning_rate_next': optimizer.param_groups[0]['lr'],
                            'adaptive_decision': decision, 'training_coverage': coverage()})
            _json(output/'metrics.json', metrics)
            return result, bool(should_eval and update >= config.min_updates
                                and bool((visits > 0).all())
                                and stale >= config.development_patience), bool(decision and decision['stop'])

        initial_metrics, _, _ = record(0)
        latest_metrics = initial_metrics
        if initial_metrics['greedy_change_count']:
            raise ValueError('initial recovery actor must retain exact native greedy')
        order, cursor = torch.randperm(count, generator=generator), 0
        for update in range(1, config.updates+1):
            if cursor == count:
                order, cursor = torch.randperm(count, generator=generator), 0
            indices = order[cursor:cursor+config.batch_contexts].to(train_h0.device)
            cursor += len(indices)
            batch = {key: table[key].index_select(0, indices) for key in ('features', 'native_pairs', 'costs')}
            loss = _objective(actor, batch['features'], batch['native_pairs'], batch['costs'], config)
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite recovery objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in actor.parameters()):
                raise ValueError('nonfinite or missing recovery actor gradients')
            norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), config.gradient_clip, error_if_nonfinite=True)
            learning_rate_used = optimizer.param_groups[0]['lr']
            optimizer.step()
            _frozen(adapter.model)
            visits.index_add_(0, indices.cpu(), torch.ones(len(indices), dtype=torch.long))
            latest_metrics, dev_stop, train_stop = record(update, gradient_norm=float(norm),
                objective=float(loss.detach()), learning_rate_used=learning_rate_used)
            successful = update
            if dev_stop or train_stop:
                # Adaptive stops outside the dev schedule still get a terminal
                # full-document evaluation and retained retrospective snapshot.
                if train_stop and dev_history[-1]['update'] != update:
                    dev = _development(development_evaluator, actor, development_contract)
                    _frozen(adapter.model)
                    dev_history.append({'update': update, **dev, 'patience_observations': stale})
                    payload = _payload(actor, binding, update, latest_metrics, dev, config, controller,
                                       learning_rate_used, optimizer.param_groups[0]['lr'])
                    _save(output/'evaluation_checkpoints'/f'checkpoint_{update:06d}.pt', payload)
                    _save(output/'checkpoint_latest.pt', payload)
                    if dev['nll'] < best_nll:
                        best_nll, best_update, best_metrics, best_dev = dev['nll'], update, latest_metrics, dev
                        best_state = {key: value.detach().clone() for key, value in actor.state_dict().items()}
                        _save(output/'checkpoint_best.pt', payload)
                    _json(output/'development_history.json', dev_history)
                stop_reason = 'development_patience' if dev_stop else 'adaptive_train_plateau'
                break
        actor.load_state_dict(best_state, strict=True)
        actor.requires_grad_(False).eval()
        status = 'complete'
    except Exception as error:
        status, failure, stop_reason = 'failed', f'{type(error).__name__}: {error}', 'failure'
        raise
    finally:
        summary = {'format': FORMAT, 'status': status, 'failure': failure, 'stop_reason': stop_reason,
                   'successful_updates': successful, 'best_update': best_update,
                   'best_development_nll': best_dev['nll'] if best_dev else None,
                   'initial_development_nll': dev_history[0]['nll'] if dev_history else None,
                   'initial_development_metrics': dev_history[0] if dev_history else None,
                   'best_development_metrics': best_dev, 'development_history': dev_history,
                   'initial_train_metrics': initial_metrics, 'best_train_metrics': best_metrics,
                   'latest_train_metrics': latest_metrics,
                   'train_fit_passed': status == 'complete' and _fit_passed(best_metrics, config),
                   'latest_train_fit_passed': status == 'complete' and _fit_passed(latest_metrics, config),
                   'fit_gate': 'train mean gain >= threshold AND positive fraction >= configured minimum AND at least three positive contexts; never checkpoint selection',
                   'checkpoint_rule': binding['checkpoint_rule'], 'adaptive': controller.snapshot(),
                   'binding_sha256': binding.get('sha256'), 'score_work_totals': work,
                   'training_coverage': coverage(), 'wall_seconds': time.perf_counter()-start,
                   'scope': SCOPE, 'final_test_evaluated': False, 'held_out_confirmation': False,
                   'resume_supported': False, 'defaults_are_provisional': True}
        _json(output/'summary.json', summary)
    return actor, summary


def load_recovery_checkpoint(path, identities, device='cpu'):
    """Inference-only weights_only load; binds config/identity/tensors/source/state.

    Embedded checksums detect corruption, not an adversary rewriting all hashes.
    Use trusted experiment artifacts. No table, targets, optimizer or RNG state
    is attached to the returned frozen actor. Resume is deliberately unsupported.
    """
    path = Path(path)
    if path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError('recovery checkpoint exceeds size budget')
    snapshot = path.read_bytes()
    payload = torch.load(io.BytesIO(snapshot), map_location='cpu', weights_only=True)
    expected = {'format', 'binding', 'feature_dim', 'compact_feature_dim', 'residual_scale', 'epsilon',
                'update', 'train_metrics', 'development_metrics', 'train_fit_passed', 'resume_supported',
                'actor_state_hashes', 'adaptive_audit_state', 'learning_rate_used', 'learning_rate_next',
                'payload_sha256', 'actor_state_dict'}
    if not isinstance(payload, dict) or set(payload) != expected or payload.get('format') != FORMAT:
        raise ValueError('unsupported recovery checkpoint structure/format')
    header = {key: value for key, value in payload.items() if key not in ('payload_sha256', 'actor_state_dict')}
    if payload['payload_sha256'] != _digest(header):
        raise ValueError('recovery checkpoint payload checksum mismatch')
    binding = copy.deepcopy(payload['binding'])
    checksum = binding.pop('sha256', None)
    if not checksum or checksum != _digest(binding):
        raise ValueError('recovery checkpoint binding checksum mismatch')
    if binding.get('identities') != identities:
        raise ValueError('recovery checkpoint experiment identities mismatch')
    if binding.get('source_sha256') != _source_hashes():
        raise ValueError('recovery checkpoint source checksum mismatch')
    if binding.get('role') != 'train_with_development_selection' or binding.get('format') != FORMAT:
        raise ValueError('recovery checkpoint role mismatch')
    try:
        config = RecoveryRouterConfig(**binding['config'])
        config.validate()
    except (TypeError, KeyError) as exc:
        raise ValueError('invalid recovery checkpoint config') from exc
    if payload['epsilon'] != config.epsilon or payload['resume_supported'] is not False:
        raise ValueError('recovery checkpoint configuration mismatch')
    _integer(payload['update'], 'checkpoint update', 0, config.updates)
    feature_dim = payload['feature_dim']
    _integer(feature_dim, 'checkpoint feature_dim', 22, 2*MAX_HIDDEN_SIZE+20)
    if (feature_dim-20) % 2:
        raise ValueError('invalid recovery checkpoint feature width')
    compact_dim = (feature_dim-20)//2 + 8
    if payload['compact_feature_dim'] != compact_dim or payload['residual_scale'] != 1/math.sqrt(compact_dim):
        raise ValueError('recovery checkpoint compact feature/scale mismatch')
    state = payload['actor_state_dict']
    if not isinstance(state, dict) or set(state) != {'weight', 'bias'}:
        raise ValueError('invalid recovery checkpoint actor state')
    for key, shape in (('weight', (28, compact_dim)), ('bias', (28,))):
        value = state[key]
        if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())):
            raise ValueError('malformed or nonfinite recovery checkpoint actor state')
    if payload['actor_state_hashes'] != _tensor_hashes(state):
        raise ValueError('recovery checkpoint actor state checksum mismatch')
    audit = payload['adaptive_audit_state']
    controller = AdaptiveFitController.from_snapshot(audit)
    if (controller.config != AdaptiveFitConfig.from_dict(config.adaptive)
            or controller.initial_lr != config.learning_rate
            or controller.lr != payload['learning_rate_next']
            or controller.last_update is not None and controller.last_update > payload['update']):
        raise ValueError('recovery checkpoint adaptive configuration mismatch')
    expected_used = config.learning_rate
    for decision in audit['decisions']:
        if decision['update'] < payload['update']:
            expected_used = decision['lr']
    if payload['learning_rate_used'] != (expected_used if payload['update'] else None):
        raise ValueError('recovery checkpoint producing learning rate mismatch')
    if payload['train_fit_passed'] is not _fit_passed(payload['train_metrics'], config):
        raise ValueError('recovery checkpoint fit verdict mismatch')
    dev = payload['development_metrics']
    if dev is not None:
        _number(dev.get('nll'), 'checkpoint development nll')
        contract = binding['development_contract']
        for key in ('role', 'protocol', 'document_ids'):
            if dev.get(key) != contract.get(key):
                raise ValueError('recovery checkpoint development contract mismatch')
    actor = RecoveryPairRouter(feature_dim, config.epsilon, device=device)
    actor.load_state_dict(state, strict=True)
    actor.requires_grad_(False).eval()
    metadata = {key: copy.deepcopy(value) for key, value in header.items() if key != 'binding'}
    metadata.update(path=str(path.resolve()), checkpoint_sha256=hashlib.sha256(snapshot).hexdigest(),
                    binding_sha256=checksum, training_config=asdict(config),
                    actor_state_sha256=_digest(_tensor_hashes(state)), source_sha256=binding['source_sha256'],
                    score_table_hashes=binding['score_table_hashes'],
                    historical_cached_inputs={'h0': binding['train_h0'], 'targets': binding['train_targets']},
                    development_contract=binding['development_contract'], scope=SCOPE)
    return actor, metadata
