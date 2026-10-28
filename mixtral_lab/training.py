"""Small R1 routing experiment; frozen Mixtral, never full-parameter RL.

No resume is supported. A new directory and content-bound contexts are required.
The adapter owns native Mixtral arithmetic; this module trains only a residual
28-way router. Evaluation is supplied by the selection-only experiment runner.
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
from torch import nn


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be positive' if positive else f'{name} must be nonnegative')


@dataclass(frozen=True)
class RouterTrainingConfig:
    updates: int = 20
    batch_contexts: int = 8
    group_size: int = 4
    learning_rate: float = 1e-4
    epsilon: float = 0.2
    eval_interval: int = 5
    max_train_contexts: int = 256
    gradient_clip: float = 1.0
    max_step_kl: float = 0.02
    seed: int = 17
    score_mode: str = "reference"
    score_atol: float = 1e-5
    score_rtol: float = 1e-5

    def validate(self):
        for name in ('updates', 'batch_contexts', 'group_size', 'eval_interval', 'max_train_contexts'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be positive integer')
        if self.score_mode not in ('reference', 'deduplicate_checked_v1'):
            raise ValueError('unsupported frozen score mode')
        _finite(self.score_atol, 'score_atol')
        _finite(self.score_rtol, 'score_rtol')
        if self.group_size < 2:
            raise ValueError('RLOO requires group_size >= 2')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('seed must be nonnegative integer')
        for name in ('learning_rate', 'gradient_clip', 'max_step_kl'):
            _finite(getattr(self, name), name, positive=True)
        _finite(self.epsilon, 'epsilon', positive=True)
        if self.epsilon >= 1:
            raise ValueError('epsilon must be below 1 to preserve initial native greedy action')
        if self.batch_contexts > self.max_train_contexts:
            raise ValueError('batch_contexts exceeds max_train_contexts')


def _pairs(device):
    return torch.combinations(torch.arange(8, device=device), r=2)


def pair_actions(pairs):
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
        raise ValueError('native pairs must be int64 [N,2]')
    pairs = pairs.sort(dim=-1).values
    if bool(((pairs < 0) | (pairs >= 8)).any()) or bool((pairs[:, 0] == pairs[:, 1]).any()):
        raise ValueError('invalid expert pairs')
    a, b = pairs.unbind(-1)
    return a * 8 - a * (a + 1) // 2 + b - a - 1


class ResidualPairRouter(nn.Module):
    """Native-pair anchored exploration; zero residual initializes native greedy.

    Unlike the original 100M actor, feature width is inferred from the adapter.
    Action ids emitted to the adapter are 1..28 (zero/STOP is never used in R1).
    """
    def __init__(self, feature_dim, epsilon=0.2, *, device=None):
        super().__init__()
        _finite(epsilon, 'epsilon', positive=True)
        if epsilon >= 1 or isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError('invalid feature dimension or epsilon')
        self.feature_dim, self.epsilon = feature_dim, float(epsilon)
        # Explicit zero parameters avoid consuming global initialization RNG.
        self.weight = nn.Parameter(torch.zeros(28, feature_dim, device=device))
        self.bias = nn.Parameter(torch.zeros(28, device=device))

    def log_probs(self, features, native_pairs):
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError('router feature shape mismatch')
        if native_pairs.shape != (len(features), 2):
            raise ValueError('native pair count mismatch')
        features = features.detach().float()
        if not bool(torch.isfinite(features).all()):
            raise ValueError('nonfinite router features')
        # Stable row normalization, not learned backbone normalization.
        features = features * torch.rsqrt(features.square().mean(-1, keepdim=True) + 1e-6)
        anchor = torch.full((len(features), 28), self.epsilon / 28, device=features.device)
        anchor.scatter_add_(1, pair_actions(native_pairs)[:, None],
                            torch.full((len(features), 1), 1-self.epsilon, device=features.device))
        residual = torch.nn.functional.linear(features, self.weight, self.bias)
        return (anchor.log() + residual).log_softmax(-1)

    @torch.no_grad()
    def greedy(self, features, native_pairs):
        return self.log_probs(features, native_pairs).argmax(-1) + 1

    def forward(self, observation):
        return self.greedy(observation.features, observation.native_pairs)


def rloo_advantage(rewards):
    if rewards.ndim != 2 or rewards.shape[1] < 2 or not rewards.is_floating_point():
        raise ValueError('rewards must be floating [B,G], G >= 2')
    rewards = rewards.detach()
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError('nonfinite rewards')
    advantage = rewards - (rewards.sum(1, keepdim=True)-rewards)/(rewards.shape[1]-1)
    return torch.where((rewards == rewards[:, :1]).all(1, keepdim=True), torch.zeros_like(advantage), advantage).detach()


def _json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _tensor_digest(tensor):
    cpu = tensor.detach().contiguous().cpu()
    return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),
            'sha256': hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()}


def _selection(evaluator, actor):
    before = actor.training
    try:
        actor.eval()
        with torch.no_grad():
            result = evaluator(actor)
    finally:
        actor.train(before)
    if not isinstance(result, dict) or 'nll' not in result:
        raise ValueError('selection evaluator must return a dictionary with nll')
    _finite(result['nll'], 'selection nll')
    return result


@torch.no_grad()
def _sampled_scores(adapter, h, y, native, reference_ce, actions, config):
    """Bounded update-local reuse diagnostic, never an unchecked speed claim.

    Sampled actions and their order are untouched. Native CE is reusable only
    for the same context index (not merely equal hidden values). Full reference
    validation is charged as real work. A failed check uses reference scores.
    Changing batch shapes can change GPU arithmetic; this is not bitwise parity.
    """
    batch, group = actions.shape
    device = actions.device
    pairs = _pairs(device)
    reference = adapter.score_pairs(h.repeat_interleave(group, 0),
        y.repeat_interleave(group), pairs[actions.reshape(-1)]).detach().to(device).reshape(batch, group)
    requested = batch + actions.numel()
    stats = {'mode': config.score_mode, 'version': 1, 'requested_rows': requested,
             'scored_rows': requested, 'score_calls': 2, 'cache_hits': 0,
             'validation_rows': 0, 'candidate_scored_rows': 0,
             'candidate_eligible': None, 'candidate_used': False}
    if config.score_mode == 'reference':
        return reference, stats
    # The key encodes context index and unordered pair id; storage is bounded
    # by B * (G + 1), and discarded before the next update.
    contexts = torch.arange(batch, device=device)[:, None].expand_as(actions)
    native_ids = pair_actions(native).to(device)
    native_hits = actions == native_ids[:, None]
    keys = (contexts * 28 + actions).reshape(-1)
    missing = ~native_hits.reshape(-1)
    unique, inverse = torch.unique(keys[missing], sorted=True, return_inverse=True)
    candidate = reference_ce[:, None].expand(batch, group).clone().reshape(-1)
    extra_calls = 0
    if unique.numel():
        rows = torch.div(unique, 28, rounding_mode='floor')
        values = adapter.score_pairs(h.index_select(0, rows.to(h.device)),
            y.index_select(0, rows.to(y.device)), pairs[unique % 28]).detach().to(device)
        candidate[missing] = values[inverse]
        extra_calls = 1
    candidate = candidate.reshape(batch, group)
    error = (candidate - reference).abs()
    tolerance = config.score_atol + config.score_rtol * reference.abs()
    finite = torch.isfinite(candidate) & torch.isfinite(reference) & torch.isfinite(error)
    eligible = bool((finite & (error <= tolerance)).all())
    raw_errors = error.cpu().tolist()
    raw_errors = [[v if math.isfinite(v) else None for v in row] for row in raw_errors]
    stats.update(candidate_scored_rows=batch + unique.numel(),
        scored_rows=requested + unique.numel(), score_calls=2 + extra_calls,
        validation_rows=actions.numel(), cache_hits=actions.numel() - unique.numel(),
        native_cache_hits=int(native_hits.sum()), candidate_eligible=eligible,
        candidate_used=eligible, fallback_reason=None if eligible else 'score_tolerance_or_nonfinite',
        atol=config.score_atol, rtol=config.score_rtol,
        absolute_errors=raw_errors, nonfinite_score_count=int((~finite).sum()),
        max_absolute_error=float(error.max()) if bool(torch.isfinite(error).all()) else None)
    return candidate if eligible else reference, stats


def train_router(adapter, train_h0, train_targets, selection_evaluator, config, identities, output_dir):
    """Train R1 actor on bounded train contexts and select on fixed selection data.

    Adapter contract: features(h0), native_pair(h0), score_pairs(h0,targets,pairs),
    plus a frozen .model. score_pairs must retain native pair weighting/order.
    train_h0 contains post-attention, pre-FFN residuals, already aligned with
    next-token targets by the runner. No shifting is performed here.
    """
    config.validate()
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError('router output must be a new directory; resume is unsupported')
    if train_h0.ndim != 2 or not train_h0.is_floating_point() or train_targets.shape != (len(train_h0),) or train_targets.dtype != torch.long:
        raise ValueError('expected floating train h0[N,D] and int64 targets[N]')
    if not config.batch_contexts <= len(train_h0) <= config.max_train_contexts:
        raise ValueError('train context count outside bounded configuration')
    if not bool(torch.isfinite(train_h0).all()) or bool((train_targets < 0).any()):
        raise ValueError('invalid cached training contexts')
    for key in ('model', 'tokenizer', 'train', 'selection', 'code'):
        if not identities.get(key):
            raise ValueError(f'missing training identity: {key}')
    model = adapter.model
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Mixtral backbone must be fully frozen')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('backbone contains pre-existing gradients')
    binding = {'identities': copy.deepcopy(identities), 'config': asdict(config),
               'train_h0': _tensor_digest(train_h0), 'train_targets': _tensor_digest(train_targets)}
    binding['sha256'] = _digest(binding)
    with torch.no_grad():
        features = adapter.features(train_h0[:1])
    actor = ResidualPairRouter(features.shape[1], config.epsilon, device=features.device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=config.learning_rate, weight_decay=0.)
    generator = torch.Generator(device=features.device).manual_seed(config.seed)
    order_generator = torch.Generator(device='cpu').manual_seed(config.seed)
    output.mkdir(parents=True)
    _json(output/'training_contract.json', binding)
    metrics, evaluations = [], []
    best_nll, best_update, successful = float('inf'), 0, 0
    status = 'running'
    start_time = time.perf_counter()
    def payload(update):
        return {'format': 'mixtral-r1-router-v1', 'binding': binding,
                'feature_dim': actor.feature_dim, 'epsilon': actor.epsilon,
                'actor_state_dict': {k: v.detach().cpu().clone() for k,v in actor.state_dict().items()},
                'update': update, 'resume_supported': False}
    def evaluate(update):
        nonlocal best_nll, best_update
        result = _selection(selection_evaluator, actor)
        evaluations.append({'update': update, **result})
        if result['nll'] < best_nll:
            best_nll, best_update = float(result['nll']), update
            _save(output/'checkpoint_best.pt', payload(update))
        _json(output/'evaluations.json', evaluations)
    try:
        evaluate(0)
        _save(output/'checkpoint_initial.pt', payload(0))
        order = torch.randperm(len(train_h0), generator=order_generator)
        cursor = 0
        for update in range(1, config.updates+1):
            if cursor + config.batch_contexts > len(order):
                order = torch.randperm(len(train_h0), generator=order_generator)
                cursor = 0
            indices = order[cursor:cursor+config.batch_contexts].to(train_h0.device)
            cursor += config.batch_contexts
            h = train_h0.index_select(0, indices)
            y = train_targets.index_select(0, indices.to(train_targets.device))
            with torch.no_grad():
                feats = adapter.features(h).detach()
                native = adapter.native_pair(h)
                reference_ce = adapter.score_pairs(h, y, native).detach().to(feats.device)
            log_probs = actor.log_probs(feats, native)
            actions = torch.multinomial(log_probs.detach().exp(), config.group_size, replacement=True, generator=generator)
            ce, score_stats = _sampled_scores(adapter, h, y, native, reference_ce, actions, config)
            rewards = reference_ce[:, None] - ce
            advantage = rloo_advantage(rewards)
            loss = -(advantage * log_probs.gather(1, actions)).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite actor loss')
            before = copy.deepcopy(actor.state_dict())
            optimizer_before = copy.deepcopy(optimizer.state_dict())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in actor.parameters()):
                raise ValueError('nonfinite or missing actor gradients')
            norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                after_log_probs = actor.log_probs(feats, native)
                kl = (log_probs.detach().exp()*(log_probs.detach()-after_log_probs)).sum(-1)
            accepted = bool(torch.isfinite(after_log_probs).all()) and bool((kl <= config.max_step_kl).all())
            if not accepted:
                actor.load_state_dict(before)
                optimizer.load_state_dict(optimizer_before)
                status = 'stopped_step_kl'
            else:
                successful = update
            row = {'score_work': score_stats, 'attempt': update, 'accepted': accepted, 'successful_updates': successful,
                   'loss': float(loss.detach()), 'reward_mean': float(rewards.mean()),
                   'teacher_agreement': float((actions == pair_actions(native)[:,None]).float().mean()),
                   'zero_advantage_fraction': float((advantage == 0).all(1).float().mean()),
                   'entropy': float(-(log_probs.detach().exp()*log_probs.detach()).sum(-1).mean()),
                   'gradient_norm': float(norm), 'step_kl_max': float(kl.max()) if bool(torch.isfinite(kl).all()) else None,
                   'distinct_sampled_actions_mean': sum(a.unique().numel() for a in actions) / len(actions),
                   'context_visits': update*config.batch_contexts, 'sampled_episodes': update*config.batch_contexts*config.group_size}
            # Finiteness guards above fail closed instead of emitting invalid JSON.
            metrics.append(row)
            _json(output/'metrics.json', metrics)
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('frozen backbone acquired gradients')
            _save(output/'checkpoint_latest.pt', payload(successful))
            if not accepted:
                # A rejected proposal must not skip selection of the last
                # accepted actor when it falls between evaluation intervals.
                if evaluations[-1]['update'] != successful:
                    evaluate(successful)
                break
            if update % config.eval_interval == 0 or update == config.updates:
                evaluate(update)
        if status == 'running':
            status = 'complete'
    except Exception:
        status = 'failed'
        raise
    finally:
        elapsed = time.perf_counter()-start_time
        devices = {str(p.device) for p in model.parameters() if p.device.type == 'cuda'}
        _json(output/'summary.json', {'status': status, 'successful_updates': successful,
              'best_update': best_update, 'best_selection_nll': best_nll if math.isfinite(best_nll) else None,
              'wall_seconds': elapsed, 'reserved_device_hours_estimate': elapsed*len(devices)/3600,
              'cuda_devices': sorted(devices), 'binding_sha256': binding['sha256'],
              'score_mode': config.score_mode,
              'score_candidate_used_updates': sum(row['score_work']['candidate_used'] for row in metrics),
              'score_candidate_fallback_updates': sum(row['score_work']['candidate_eligible'] is False for row in metrics),
              'score_work_totals': {key: sum(row['score_work'][key] for row in metrics)
                  for key in ('requested_rows', 'scored_rows', 'score_calls', 'cache_hits', 'validation_rows')},
              'score_reuse_note': 'checked candidate includes full reference validation cost; not a production speedup claim',
              'scope': 'R1 frozen-backbone routing only; no resume, R2 training or held-out confirmation'})
    return actor, json.loads((output/'summary.json').read_text())


def load_actor_checkpoint_with_metadata(path, identities, device="cpu"):
    """Load an inference-only actor, requiring the exact experiment identities.

    This is deliberately not optimizer/RNG resume. Only trusted local experiment
    checkpoints should be supplied, and pickle objects are disallowed.
    """
    snapshot = Path(path).read_bytes()
    payload = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "mixtral-r1-router-v1":
        raise ValueError("unsupported router checkpoint")
    binding = copy.deepcopy(payload.get("binding", {}))
    checksum = binding.pop("sha256", None)
    if not checksum or checksum != _digest(binding):
        raise ValueError("checkpoint binding checksum mismatch")
    if binding.get("identities") != identities:
        raise ValueError("checkpoint experiment identities mismatch")
    cfg = RouterTrainingConfig(**binding["config"])
    cfg.validate()
    if payload.get("epsilon") != cfg.epsilon:
        raise ValueError("checkpoint exploration config mismatch")
    actor = ResidualPairRouter(payload["feature_dim"], cfg.epsilon, device=device)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    if any(not bool(torch.isfinite(p).all()) for p in actor.parameters()):
        raise ValueError("checkpoint contains nonfinite actor weights")
    update = payload.get('update')
    if isinstance(update, bool) or not isinstance(update, int) or not 0 <= update <= cfg.updates:
        raise ValueError('invalid checkpoint update')
    metadata = {'path': str(Path(path).resolve()),
                'checkpoint_sha256': hashlib.sha256(snapshot).hexdigest(),
                'binding_sha256': checksum, 'update': update, 'epsilon': cfg.epsilon,
                'training_config': copy.deepcopy(binding['config']),
                'actor_state_sha256': _digest({key: _tensor_digest(value)
                    for key, value in sorted(payload['actor_state_dict'].items())})}
    return actor.eval(), metadata


def load_actor_checkpoint(path, identities, device="cpu"):
    """Backward-compatible actor-only inference loader (not a resume API)."""
    return load_actor_checkpoint_with_metadata(path, identities, device)[0]
