"""Train-only learnability diagnostic for added low-rank residual capacity.

This is fixed native R1 plus a new adapter, NOT recurrent FFN execution, learned
STOP, or evidence of held-out/efficiency gains. A fixed all-training-context
probe selects checkpoints; this API intentionally has no selection callback.
BF16 post-add diagnostics measure quantization, not a promised remedy for it.
"""
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
from torch.nn import functional as F

from .adaptive_fit import AdaptiveFitConfig, AdaptiveFitController
from .second_round import SecondRoundResidual, differentiable_readout

FORMAT = 'mixtral-adapter-fit-v1'
_MAX_HIDDEN_SIZE = 65536
_MAX_CHECKPOINT_BYTES = 160 * 1024 * 1024


@dataclass(frozen=True)
class AdapterFitConfig:
    rank: int = 8
    updates: int = 256
    batch_contexts: int = 8
    max_train_contexts: int = 64
    probe_chunk_contexts: int = 8
    eval_interval: int = 16
    learning_rate: float = 1e-3
    gradient_clip: float = 1.0
    residual_scale: float = 1.0
    fit_gain_threshold: float = 0.01
    seed: int = 17
    adaptive: dict | None = None

    def validate(self):
        for name, upper in [('rank', 256), ('updates', 10000),
                            ('batch_contexts', 1024), ('max_train_contexts', 4096),
                            ('probe_chunk_contexts', 1024), ('eval_interval', 10000)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ValueError(f'{name} must be integer in [1,{upper}]')
        for name, upper in [('learning_rate', 1.), ('gradient_clip', 1e6),
                            ('residual_scale', 1.), ('fit_gain_threshold', 100.)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= upper:
                raise ValueError(f'{name} must be finite in (0,{upper}]')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be nonnegative int64')
        adaptive = AdaptiveFitConfig.from_dict(self.adaptive)
        if adaptive.enabled and adaptive.eval_interval != self.eval_interval:
            raise ValueError('adaptive eval_interval must match the fixed train probe eval_interval')
        # Validate the starting LR before creating an output directory.
        AdaptiveFitController(adaptive, initial_lr=self.learning_rate)


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _identities(value):
    if not isinstance(value, dict) or not value:
        raise ValueError('nonempty identity mapping is required')
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
        if json.loads(encoded) != value:
            raise ValueError('identity mapping must contain JSON values')
        return _json_hash(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('identity mapping must contain finite JSON values and string keys') from exc


def _tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)],
                                 separators=(',', ':')).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_write(path, writer):
    """An interrupted write never replaces the previous complete artifact."""
    path = Path(path)
    fd, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        with temporary.open('rb') as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_contract(data):
    return {key: data[key] for key in ('format', 'identities', 'config', 'hidden_size',
                                      'train_inputs_sha256', 'cached_native_sha256', 'probe_spec')}


def _save_checkpoint(path, module, identities, tensor_binding, probe_spec, update, probe, controller_state):
    state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    data = dict(format=FORMAT, config=asdict(module.config), hidden_size=module.hidden_size,
                identities=copy.deepcopy(identities), binding_sha256=_identities(identities),
                **tensor_binding, probe_spec=probe_spec, update=update,
                train_probe_nll=probe['nll'], probe_diagnostics=probe['diagnostics'],
                state=state, state_sha256=_tensor_hash(state),
                adaptive_controller_state=copy.deepcopy(controller_state))
    data['contract_sha256'] = _json_hash(_checkpoint_contract(data))
    data['metadata_sha256'] = _json_hash({key: value for key, value in data.items() if key != 'state'})
    _atomic_write(path, lambda temporary: torch.save(data, temporary))


def _sha256_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load_adapter_fit_checkpoint(path, identities, device='cpu', expected_config=None,
                                expected_hidden_size=None):
    """Load only for frozen inference; return (module, JSON-safe metadata).

    Hashes detect artifact corruption/mismatch, not a malicious party able to
    rewrite the complete artifact. No optimizer state or resume path is exposed.
    """
    binding = _identities(identities)
    path = Path(path)
    if path.stat().st_size > _MAX_CHECKPOINT_BYTES:
        raise ValueError('adapter-fit checkpoint exceeds size budget')
    payload = path.read_bytes()
    data = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    expected_keys = {'format', 'config', 'hidden_size', 'identities', 'binding_sha256',
                     'train_inputs_sha256', 'cached_native_sha256', 'probe_spec', 'update',
                     'train_probe_nll', 'probe_diagnostics', 'state', 'state_sha256',
                     'contract_sha256', 'metadata_sha256', 'adaptive_controller_state'}
    if not isinstance(data, dict) or set(data) != expected_keys:
        raise ValueError('invalid adapter-fit checkpoint structure')
    if data['format'] != FORMAT or data['binding_sha256'] != binding or data['identities'] != identities:
        raise ValueError('adapter-fit checkpoint identity/format mismatch')
    if data['contract_sha256'] != _json_hash(_checkpoint_contract(data)):
        raise ValueError('adapter-fit configuration/tensor binding mismatch')
    metadata = {key: value for key, value in data.items() if key not in ('state', 'metadata_sha256')}
    if data['metadata_sha256'] != _json_hash(metadata):
        raise ValueError('adapter-fit metadata binding mismatch')
    if not isinstance(data['adaptive_controller_state'], dict):
        raise ValueError('invalid adapter-fit adaptive audit state')
    try:
        config = AdapterFitConfig(**data['config'])
        config.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid adapter-fit configuration') from exc
    hidden = data['hidden_size']
    if isinstance(hidden, bool) or not isinstance(hidden, int) or not config.rank <= hidden <= _MAX_HIDDEN_SIZE:
        raise ValueError('invalid adapter-fit hidden size')
    if expected_config is not None and data['config'] != asdict(expected_config):
        raise ValueError('adapter-fit expected configuration mismatch')
    if expected_hidden_size is not None and hidden != expected_hidden_size:
        raise ValueError('adapter-fit expected hidden size mismatch')
    if any(not _sha256_string(data[key]) for key in ('train_inputs_sha256', 'cached_native_sha256', 'state_sha256')):
        raise ValueError('invalid adapter-fit tensor digest')
    spec = data['probe_spec']
    if (not isinstance(spec, dict) or set(spec) != {'split', 'contexts', 'chunk_contexts', 'last_chunk_padding', 'order'}
            or spec['split'] != 'train' or spec['order'] != 'cached_index'
            or spec['last_chunk_padding'] != 'repeat_last_masked'
            or isinstance(spec['contexts'], bool) or not isinstance(spec['contexts'], int)
            or not 1 <= spec['contexts'] <= config.max_train_contexts
            or spec['chunk_contexts'] != config.probe_chunk_contexts):
        raise ValueError('invalid train-only probe contract')
    update, nll = data['update'], data['train_probe_nll']
    if isinstance(update, bool) or not isinstance(update, int) or not 0 <= update <= config.updates:
        raise ValueError('invalid checkpoint update')
    if isinstance(nll, bool) or not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0:
        raise ValueError('invalid train probe NLL')
    audit = AdaptiveFitController.from_snapshot(data['adaptive_controller_state'])
    if (audit.config != AdaptiveFitConfig.from_dict(config.adaptive)
            or audit.initial_lr != config.learning_rate
            or audit.last_update is None or audit.last_update > update):
        raise ValueError('adaptive checkpoint audit/configuration mismatch')
    state = data['state']
    if not isinstance(state, dict) or set(state) != {'down', 'up'}:
        raise ValueError('invalid adapter-fit state keys')
    for name, shape in [('down', (config.rank, hidden)), ('up', (hidden, config.rank))]:
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('invalid adapter-fit weight shape/dtype/values')
    if _tensor_hash(state) != data['state_sha256']:
        raise ValueError('adapter-fit state digest mismatch')
    module = SecondRoundResidual(hidden, config, device)
    module.load_state_dict(state, strict=True)
    module.requires_grad_(False).eval()
    metadata['metadata_sha256'] = data['metadata_sha256']
    metadata['checkpoint_sha256'] = hashlib.sha256(payload).hexdigest()
    return module, metadata


def _fixed_chunks(hidden, labels, chunk_contexts):
    """Padding fixes GEMM dimensions; padded rows are NEVER scored or trained."""
    for start in range(0, len(hidden), chunk_contexts):
        h, y = hidden[start:start + chunk_contexts], labels[start:start + chunk_contexts]
        valid = len(h)
        if valid < chunk_contexts:
            h = torch.cat((h, h[-1:].expand(chunk_contexts - valid, -1)))
            y = torch.cat((y, y[-1:].expand(chunk_contexts - valid)))
        yield h, y, valid


def _correction_totals(module, native, proposed=None):
    proposed = module.correction(native) if proposed is None else proposed
    corrected = native + proposed.to(native.dtype)
    # Explicit post-add BF16 change, not merely a cast of the correction.
    bf16_base = native.to(torch.bfloat16)
    bf16_corrected = bf16_base + proposed.to(torch.bfloat16)
    effective = corrected.float() - native.float()
    bf16_effective = bf16_corrected.float() - bf16_base.float()
    if not all(bool(torch.isfinite(value).all()) for value in (proposed, corrected, bf16_corrected)):
        raise FloatingPointError('nonfinite proposed/effective correction')
    return dict(elements=proposed.numel(),
                proposed_square_sum=float(proposed.double().square().sum()),
                native_square_sum=float(effective.double().square().sum()),
                native_changed=int(torch.count_nonzero(effective)),
                bf16_square_sum=float(bf16_effective.double().square().sum()),
                bf16_changed=int(torch.count_nonzero(bf16_effective)))


def _diagnostics(totals, dtype):
    count = totals['elements']
    return dict(fp32_proposed_correction_rms=math.sqrt(totals['proposed_square_sum'] / count),
                native_effective_change_rms=math.sqrt(totals['native_square_sum'] / count),
                native_effective_change_rate=totals['native_changed'] / count,
                bf16_effective_change_rms=math.sqrt(totals['bf16_square_sum'] / count),
                bf16_effective_change_rate=totals['bf16_changed'] / count,
                bf16_diagnostic_is_actual_native=(dtype == torch.bfloat16),
                native_dtype=str(dtype))


def _training_probe(adapter, module, hidden, labels, config, work=None):
    """Fixed, full TRAIN cache only; weighted valid-row mean, including remainder."""
    module.eval()
    if work is not None:
        work['train_probe_calls'] += 1
    losses, totals = [], None
    with torch.no_grad():
        for h, y, valid in _fixed_chunks(hidden, labels, config.probe_chunk_contexts):
            if work is not None:
                work['train_probe_head_batches'] += 1
                work['train_probe_head_contexts_including_padding'] += len(h)
            logits = differentiable_readout(adapter, module(h))
            loss = F.cross_entropy(logits.float(), y.to(logits.device), reduction='none')[:valid]
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError('nonfinite train probe NLL')
            losses.append(loss.double().cpu())
            if work is not None:
                work['train_probe_scored_contexts'] += valid
                work['train_probe_diagnostic_contexts_including_padding'] += len(h)
            # Diagnostics use the same padded correction GEMM as the readout.
            proposed = module.correction(h)[:valid]
            part = _correction_totals(module, h[:valid], proposed=proposed)
            totals = part if totals is None else {key: totals[key] + value for key, value in part.items()}
    if work is not None:
        work['completed_train_probe_calls'] += 1
    return dict(nll=float(torch.cat(losses).mean()), diagnostics=_diagnostics(totals, hidden.dtype))


def _assert_frozen(adapter):
    if any(module.training for module in adapter.model.modules()):
        raise ValueError('backbone and submodules must be in eval mode')
    if any(parameter.requires_grad for parameter in adapter.model.parameters()):
        raise ValueError('backbone must be frozen')
    if any(parameter.grad is not None for parameter in adapter.model.parameters()):
        raise ValueError('backbone gradients must be absent')


def train_adapter_fit(adapter, train_h0, train_targets, config, identities, output_dir):
    """Fit added capacity on TRAIN only and return training-probe-best weights.

    Epochs shuffle without replacement and consume the remainder before a new
    epoch. Zero participates in checkpoint selection and wins ties. Insufficient
    coverage or numerical guard stops are not negative learnability evidence.
    Inputs are cached terminal-prefix contexts and already-shifted labels.
    """
    config.validate()
    _assert_frozen(adapter)
    _identities(identities)
    if isinstance(adapter.hidden_size, bool) or not isinstance(adapter.hidden_size, int) or not config.rank <= adapter.hidden_size <= _MAX_HIDDEN_SIZE:
        raise ValueError('invalid hidden size/rank budget')
    if (not isinstance(train_h0, torch.Tensor) or train_h0.ndim != 2
            or train_h0.shape[1] != adapter.hidden_size
            or not 1 <= len(train_h0) <= config.max_train_contexts
            or train_h0.dtype not in (torch.float32, torch.bfloat16)):
        raise ValueError('invalid/beyond-budget FP32/BF16 training contexts')
    if (not isinstance(train_targets, torch.Tensor) or train_targets.shape != (len(train_h0),)
            or train_targets.dtype != torch.long):
        raise ValueError('training targets must be int64 [contexts], already shifted')
    if not bool(torch.isfinite(train_h0).all()):
        raise ValueError('nonfinite training contexts')
    if bool(((train_targets < 0) | (train_targets >= adapter.model.config.vocab_size)).any()):
        raise ValueError('targets outside vocabulary')
    tensor_binding = dict(train_inputs_sha256=_tensor_hash({'h0': train_h0, 'targets': train_targets}))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)  # no overwrite or resume
    start = time.perf_counter()
    h0, labels = train_h0.detach().to(adapter.device), train_targets.detach().to(adapter.device)
    # Complete native R1 ONCE. Nothing below calls an expert again.
    with torch.no_grad():
        pieces = []
        for h, _, valid in _fixed_chunks(h0, labels, config.probe_chunk_contexts):
            native = adapter.step_pairs(h, adapter.native_pair(h))
            if native.shape != h.shape or native.dtype != h.dtype or not bool(torch.isfinite(native).all()):
                raise ValueError('invalid cached native R1 completion')
            pieces.append(native[:valid].detach())
        h1 = torch.cat(pieces)
    tensor_binding['cached_native_sha256'] = _tensor_hash({'h1': h1})
    probe_spec = dict(split='train', contexts=len(h1), chunk_contexts=config.probe_chunk_contexts,
                      last_chunk_padding='repeat_last_masked', order='cached_index')
    module = SecondRoundResidual(adapter.hidden_size, config, adapter.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=config.learning_rate, weight_decay=0.)
    controller = AdaptiveFitController(AdaptiveFitConfig.from_dict(config.adaptive),
                                       initial_lr=config.learning_rate)
    generator = torch.Generator(device='cpu').manual_seed(config.seed)
    work = dict(native_cache_terminal_ffn_batches=len(pieces),
                native_cache_valid_contexts=len(h1),
                native_cache_contexts_including_padding=len(pieces) * config.probe_chunk_contexts,
                optimizer_head_contexts=0, optimizer_backward_contexts=0,
                postupdate_diagnostic_contexts=0, train_probe_calls=0, completed_train_probe_calls=0,
                train_probe_head_batches=0, train_probe_head_contexts_including_padding=0,
                train_probe_scored_contexts=0, train_probe_diagnostic_contexts_including_padding=0)
    initial_probe = best_probe = latest_probe = _training_probe(adapter, module, h1, labels, config, work)
    initial_decision = controller.observe(0, initial_probe['nll'], coverage_complete=False)
    best_controller_state = latest_controller_state = controller.snapshot()
    best_state = copy.deepcopy(module.state_dict())
    best_update = latest_probe_update = successful = completed_epochs = 0
    visits = [0] * len(h1)
    probe_interval_gradients = dict(start_update_exclusive=0, accepted_updates=0,
                                    nonzero_gradient_updates=0, max_gradient_norm=0.)
    rows, probes = [], [dict(update=0, **initial_probe, adaptive_decision=initial_decision,
                            probe_interval_gradients=probe_interval_gradients.copy())]
    for name in ('initial', 'best'):
        _save_checkpoint(output / f'checkpoint_{name}.pt', module, identities, tensor_binding,
                         probe_spec, 0, initial_probe, controller_state=best_controller_state)
    status, stop_reason, adaptive_stopped = 'completed', 'update_budget_exhausted', False
    permutation, cursor, epoch = [], len(h1), 0
    for update in range(1, config.updates + 1):
        if cursor == len(h1):
            permutation = torch.randperm(len(h1), generator=generator).tolist()
            cursor, epoch = 0, epoch + 1
        indices = permutation[cursor:cursor + config.batch_contexts]
        cursor += len(indices)
        idx = torch.tensor(indices, dtype=torch.long, device=h1.device)
        row = dict(update=update, epoch=epoch, visited_indices=indices,
                   batch_contexts=len(indices), accepted=False,
                   learning_rate=optimizer.param_groups[0]['lr'])
        rows.append(row)
        module.train()
        optimizer.zero_grad(set_to_none=True)
        before = copy.deepcopy(module.state_dict())
        work['optimizer_head_contexts'] += len(indices)
        logits = differentiable_readout(adapter, module(h1[idx]))
        loss = F.cross_entropy(logits.float(), labels[idx].to(logits.device))
        if not bool(torch.isfinite(loss)):
            status = 'stopped_nonfinite_loss'
            break
        row['train_batch_nll'] = float(loss.detach())
        work['optimizer_backward_contexts'] += len(indices)
        loss.backward()
        _assert_frozen(adapter)
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in module.parameters()):
            status = 'stopped_nonfinite_gradient'
            break
        grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), config.gradient_clip)
        if not bool(torch.isfinite(grad_norm)):
            status = 'stopped_nonfinite_gradient'
            break
        row['gradient_norm'] = float(grad_norm)
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in module.parameters()):
            module.load_state_dict(before)
            status = 'stopped_nonfinite_parameters'
            break
        try:
            with torch.no_grad():
                work['postupdate_diagnostic_contexts'] += len(indices)
                row['correction_diagnostics'] = _diagnostics(_correction_totals(module, h1[idx]), h1.dtype)
        except FloatingPointError:
            module.load_state_dict(before)
            status = 'stopped_nonfinite_correction'
            break
        stop_after_update = False
        probe_due = update % config.eval_interval == 0 or update == config.updates
        if probe_due:
            try:
                probe = _training_probe(adapter, module, h1, labels, config, work)
            except FloatingPointError:
                module.load_state_dict(best_state)
                latest_probe, latest_probe_update = best_probe, best_update
                latest_controller_state = copy.deepcopy(best_controller_state)
                status = 'stopped_nonfinite_probe'
                break
        # Count only updates that passed every applicable numerical guard,
        # including the current full probe. A zero-gradient final minibatch
        # must not erase evidence from earlier accepted updates in this window.
        probe_interval_gradients['accepted_updates'] += 1
        probe_interval_gradients['nonzero_gradient_updates'] += int(row['gradient_norm'] > 0.)
        probe_interval_gradients['max_gradient_norm'] = max(
            probe_interval_gradients['max_gradient_norm'], row['gradient_norm'])
        row['probe_interval_gradients'] = probe_interval_gradients.copy()
        if probe_due:
            # Full train probe ONLY. A BF16 counterfactual on FP32 cannot justify growth.
            covered_after_update = all(count > 0 or index in indices
                                       for index, count in enumerate(visits))
            ineffective = (h1.dtype == torch.bfloat16
                           and probe['diagnostics']['native_effective_change_rate'] == 0.
                           and probe_interval_gradients['nonzero_gradient_updates'] > 0)
            decision = controller.observe(update, probe['nll'],
                coverage_complete=covered_after_update, ineffective_update=ineffective)
            for group in optimizer.param_groups:
                group['lr'] = decision['lr']
            row['adaptive_decision'] = decision
            row['next_learning_rate'] = decision['lr']
            stop_after_update = decision['stop']
            latest_probe, latest_probe_update = probe, update
            latest_controller_state = controller.snapshot()
            probes.append(dict(update=update, **probe, adaptive_decision=decision,
                               probe_interval_gradients=probe_interval_gradients.copy()))
            probe_interval_gradients = dict(start_update_exclusive=update, accepted_updates=0,
                                            nonzero_gradient_updates=0, max_gradient_norm=0.)
            row['train_probe_nll'] = probe['nll']
            if probe['nll'] < best_probe['nll']:
                best_probe, best_update = probe, update
                best_state = copy.deepcopy(module.state_dict())
                best_controller_state = copy.deepcopy(latest_controller_state)
                _save_checkpoint(output / 'checkpoint_best.pt', module, identities, tensor_binding,
                                 probe_spec, best_update, best_probe, controller_state=best_controller_state)
        row['accepted'] = True
        successful += 1
        for index in indices:
            visits[index] += 1
        if cursor == len(h1):
            completed_epochs += 1
        row['unique_contexts_visited'] = sum(count > 0 for count in visits)
        row['completed_epochs'] = completed_epochs
        if stop_after_update:
            stop_reason, adaptive_stopped = 'adaptive_plateau', True
            break
    # Score the actual last accepted weights after an unscheduled guard stop.
    if status != 'stopped_nonfinite_probe' and latest_probe_update != successful:
        try:
            latest_probe = _training_probe(adapter, module, h1, labels, config, work)
            latest_probe_update = successful
            latest_controller_state = controller.snapshot()
            probes.append(dict(update=successful, **latest_probe,
                               probe_interval_gradients=probe_interval_gradients.copy()))
        except FloatingPointError:
            module.load_state_dict(best_state)
            latest_probe, latest_probe_update = best_probe, best_update
            latest_controller_state = copy.deepcopy(best_controller_state)
            status = 'stopped_nonfinite_probe'
        if latest_probe['nll'] < best_probe['nll']:
            best_probe, best_update = latest_probe, latest_probe_update
            best_state = copy.deepcopy(module.state_dict())
            best_controller_state = copy.deepcopy(latest_controller_state)
            _save_checkpoint(output / 'checkpoint_best.pt', module, identities, tensor_binding,
                             probe_spec, best_update, best_probe, controller_state=best_controller_state)
    _save_checkpoint(output / 'checkpoint_latest.pt', module, identities, tensor_binding,
                     probe_spec, latest_probe_update, latest_probe, controller_state=latest_controller_state)
    module.load_state_dict(best_state)
    module.zero_grad(set_to_none=True)
    module.requires_grad_(False).eval()
    _assert_frozen(adapter)
    unique = sum(count > 0 for count in visits)
    gain = initial_probe['nll'] - best_probe['nll']
    passed = status == 'completed' and unique == len(h1) and gain >= config.fit_gain_threshold
    evidence = ('numerical_guard_stopped' if status != 'completed' else
                'incomplete_coverage' if unique != len(h1) else
                'fit_established' if passed else 'threshold_not_reached')
    work['optimizer_accepted_context_visits'] = sum(visits)
    work['total_head_forward_contexts_including_padding'] = (work['optimizer_head_contexts']
        + work['train_probe_head_contexts_including_padding'])
    work['total_adapter_correction_contexts_including_diagnostics_and_padding'] = (
        work['total_head_forward_contexts_including_padding'] + work['postupdate_diagnostic_contexts']
        + work['train_probe_diagnostic_contexts_including_padding'])
    summary = dict(status=status, stop_reason=stop_reason if status == 'completed' else status,
                   adaptive_stopped=adaptive_stopped,
                   adaptive_controller_state=controller.snapshot(),
                   best_adaptive_controller_state=best_controller_state, experiment='train_only_fixed_native_r1_adapter_fit',
                   config=asdict(config), checkpoint_selection='minimum_fixed_train_probe_nll',
                   initial_train_probe_nll=initial_probe['nll'], best_train_probe_nll=best_probe['nll'],
                   latest_train_probe_nll=latest_probe['nll'], best_update=best_update,
                   latest_update=latest_probe_update, train_fit_gain=gain,
                   train_fit_passed=passed, fit_evidence_status=evidence,
                   successful_updates=successful, attempted_updates=len(rows),
                   train_contexts=len(h1), unique_contexts_visited=unique,
                   context_visit_counts=visits, context_visits=sum(visits),
                   minimum_context_visits=min(visits), effective_train_epochs=sum(visits) / len(h1),
                   work_accounting=work,
                   completed_epochs=completed_epochs, coverage_complete=unique == len(h1),
                   train_probe_history=probes, probe_spec=probe_spec,
                   trainable_parameters=sum(p.numel() for p in module.parameters()),
                   initial_correction_diagnostics=initial_probe['diagnostics'],
                   best_correction_diagnostics=best_probe['diagnostics'],
                   latest_correction_diagnostics=latest_probe['diagnostics'],
                   added_capacity=True, recurrent_ffn=False, learned_stop=False,
                   selection_used=False, heldout_gain_established=False,
                   practical_gain_established=False, backbone_gradients_absent=True,
                   arithmetic='FP32 adapter; correction cast then native residual add; frozen differentiable norm/head',
                   limits=['train fit is not generalization', 'threshold miss applies only to this bounded adapter fit',
                           'no recurrent FFN or learned STOP', 'added capacity requires parameter-matched controls',
                           'optional bounded LR feedback does not find globally optimal hyperparameters',
                           'rank, residual scale, architecture and stage gates never adapt',
                           'BF16 post-add diagnostics are not a quantization fix',
                           'fixed padded cache/probe shapes may differ numerically from full-document deployment',
                           'work counts include padding/probes; they are not FLOPs or a speed/cost conclusion'],
                   binding_sha256=_identities(identities), **tensor_binding,
                   elapsed_seconds=time.perf_counter() - start)
    for name, value in [('metrics.json', rows), ('summary.json', summary)]:
        _atomic_write(output / name, lambda temporary, value=value:
                      temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'))
    return module, summary
