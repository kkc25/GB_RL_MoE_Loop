"""Bounded native-regularized residual training with independent development selection.

This is added low-rank capacity after fixed native R1, not recurrent experts.
The full-vocabulary soft native teacher is recomputed per fixed-size chunk, on
exactly the SAME cached h1 and readout shape as the student. No teacher logits
are retained between chunks. Labels are training-only; inference is module(h1).
Development is full-document online NLL, never cached train CE or final test.
"""
from dataclasses import asdict, dataclass
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .adapter_fit import (_assert_frozen, _atomic_write, _correction_totals,
                          _diagnostics, _fixed_chunks, _identities, _json_hash,
                          _sha256_string, _tensor_hash)
from .adaptive_fit import AdaptiveFitConfig, AdaptiveFitController
from .second_round import SecondRoundResidual, differentiable_readout

FORMAT = 'mixtral-generalization-recovery-adapter-v1'
_MAX_HIDDEN_SIZE = 65536
_MAX_CHECKPOINT_BYTES = 160 * 1024 * 1024
_DEVELOPMENT_PROTOCOL = 'full_document_online_nll'
_SOURCE_FILES = ('recovery_adapter.py', 'adapter_fit.py', 'second_round.py', 'adaptive_fit.py')


@dataclass(frozen=True)
class RecoveryAdapterConfig:
    rank: int = 8
    updates: int = 256
    batch_contexts: int = 8
    max_train_contexts: int = 4096
    probe_chunk_contexts: int = 8
    eval_interval: int = 16
    learning_rate: float = 1e-3
    gradient_clip: float = 1.
    residual_scale: float = .1
    native_kl_beta: float = .1
    relative_correction_l2: float = 0.
    development_patience: int = 4
    min_updates: int = 32
    development_min_delta: float = 0.
    seed: int = 17
    adaptive: dict | None = None

    def validate(self):
        for name, lower, upper in (
                ('rank', 1, 256), ('updates', 1, 10000), ('batch_contexts', 1, 1024),
                ('max_train_contexts', 1, 100000), ('probe_chunk_contexts', 1, 64),
                ('eval_interval', 1, 10000), ('development_patience', 1, 10000),
                ('min_updates', 0, 10000), ('seed', 0, 2**63 - 1)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f'{name} must be integer in [{lower},{upper}]')
        if 1 + math.ceil(self.updates / self.eval_interval) > 1024:
            raise ValueError('at most 1024 evaluated checkpoints are allowed')
        for name, lower, upper, strict in (
                ('learning_rate', 0., 1., True), ('gradient_clip', 0., 1e6, True),
                ('residual_scale', 0., 1., True), ('native_kl_beta', 0., 100., False),
                ('relative_correction_l2', 0., 100., False),
                ('development_min_delta', 0., 100., False)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not lower <= value <= upper
                    or (strict and value == lower)):
                raise ValueError(f'invalid finite bounded {name}')
        adaptive = AdaptiveFitConfig.from_dict(self.adaptive)
        if adaptive.enabled and adaptive.eval_interval != self.eval_interval:
            raise ValueError('adaptive eval_interval must match fixed training probe interval')
        AdaptiveFitController(adaptive, initial_lr=self.learning_rate)


def _document_ids(value, name):
    if (not isinstance(value, list) or not value or len(value) > 100000
            or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value))):
        raise ValueError(f'{name} must be nonempty unique document identifiers')
    return value


def _development_contract(identities, evaluator=None):
    _identities(identities)
    for name in ('model', 'tokenizer', 'train', 'development', 'code'):
        if not isinstance(identities.get(name), dict) or not identities[name]:
            raise ValueError(f'nonempty {name} identity object is required')
    train = _document_ids(identities['train'].get('selected_document_ids'), 'train.selected_document_ids')
    dev = _document_ids(identities['development'].get('selected_document_ids'), 'development.selected_document_ids')
    for key, expected in [('train_document_ids', train), ('development_document_ids', dev)]:
        if key in identities and identities[key] != expected:
            raise ValueError('conflicting document identity alias')
    if set(train) & set(dev):
        raise ValueError('train/development documents must be disjoint')
    contract = dict(role='development', document_ids=list(dev), protocol=_DEVELOPMENT_PROTOCOL)
    if evaluator is not None and (not callable(evaluator) or getattr(evaluator, 'contract', None) != contract):
        raise ValueError('development evaluator must declare matching full-document contract')
    return contract


def _source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in _SOURCE_FILES}


def _finite_number(value, name, nonnegative=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (nonnegative and value < 0)):
        raise ValueError(f'{name} must be finite' + (' nonnegative' if nonnegative else ''))
    return float(value)


def _validate_development(result, contract):
    if (not isinstance(result, dict) or result.get('status') != 'ok'
            or any(result.get(key) != value for key, value in contract.items())):
        raise ValueError('development callback status/role/documents/protocol mismatch')
    _finite_number(result.get('nll'), 'development nll')
    try:
        encoded = json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('development result must contain finite JSON values') from exc
    if len(encoded.encode()) > 1024 * 1024:
        raise ValueError('development result exceeds audit size budget')
    return copy.deepcopy(result)


def _evaluate_development(adapter, module, evaluator, contract, work):
    module.eval()
    before = _tensor_hash(module.state_dict())
    config_before, hidden_before = copy.deepcopy(asdict(module.config)), module.hidden_size
    started = time.perf_counter()
    work['development_evaluator_calls'] += 1
    try:
        try:
            with torch.no_grad():
                result = evaluator(module)
        except Exception as exc:
            raise ValueError('development evaluator failed: ' + type(exc).__name__ + ': ' + str(exc)) from exc
        _assert_frozen(adapter)
        if getattr(evaluator, 'contract', None) != contract:
            raise ValueError('development evaluator contract mutated during evaluation')
        if (not isinstance(module.config, RecoveryAdapterConfig)
                or asdict(module.config) != config_before or module.hidden_size != hidden_before
                or module.training or _tensor_hash(module.state_dict()) != before
                or not all(parameter.requires_grad for parameter in module.parameters())):
            raise ValueError('development evaluator mutated the training module')
        result = _validate_development(result, contract)
    finally:
        work['development_evaluator_seconds'] += time.perf_counter() - started
    work['completed_development_evaluator_calls'] += 1
    return result


def _loss_components(adapter, module, native, labels, config):
    """One bounded chunk: FP32 CE + beta KL(native || student) + relative L2.

    The native distribution is detached, full-support, full-vocabulary, T=1.
    Returning only per-row scalars/proposed residual does not cache the teacher.
    Caller must backward each training chunk before evaluating the next chunk.
    """
    proposed = module.correction(native)
    corrected = native.detach() + proposed.to(native.dtype)
    if not all(bool(torch.isfinite(x).all()) for x in (proposed, corrected)):
        raise FloatingPointError('nonfinite correction')
    # Reference uses the same cached native and row shape, not another FFN pass.
    with torch.no_grad():
        teacher_logits = differentiable_readout(adapter, native.detach()).float()
        teacher_logp = F.log_softmax(teacher_logits, dim=-1)
    student_logits = differentiable_readout(adapter, corrected).float()
    if not all(bool(torch.isfinite(x).all()) for x in (teacher_logits, teacher_logp, student_logits)):
        raise FloatingPointError('nonfinite native/student readout')
    ce = F.cross_entropy(student_logits, labels.to(student_logits.device), reduction='none')
    kl = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_logp,
                  reduction='none', log_target=True).sum(-1)
    relative_l2 = (proposed.square().sum(-1)
                   / native.detach().float().square().sum(-1).clamp_min(1e-12))
    loss = ce + config.native_kl_beta * kl + config.relative_correction_l2 * relative_l2.to(ce.device)
    if not all(bool(torch.isfinite(x).all()) for x in (ce, kl, relative_l2, loss)):
        raise FloatingPointError('nonfinite loss components')
    return dict(loss=loss, ce=ce, native_kl=kl, relative_correction_l2=relative_l2,
                proposed=proposed)


def _training_probe(adapter, module, hidden, labels, config, work=None):
    """Stable padded full-training probe; padding never contributes to means."""
    module.eval()
    sums = dict(loss=0., ce=0., native_kl=0., relative_correction_l2=0.)
    totals, hidden_square_sum = None, 0.
    if work is not None:
        work['train_probe_calls'] += 1
    with torch.no_grad():
        for h, y, valid in _fixed_chunks(hidden, labels, config.probe_chunk_contexts):
            if work is not None:
                work['train_probe_head_contexts_including_padding'] += 2 * len(h)
                work['teacher_head_contexts_including_padding'] += len(h)
            part = _loss_components(adapter, module, h, y, config)
            for key in sums:
                sums[key] += float(part[key][:valid].double().sum())
            contribution = _correction_totals(module, h[:valid], proposed=part['proposed'][:valid])
            totals = contribution if totals is None else {k: totals[k] + v for k, v in contribution.items()}
            hidden_square_sum += float(h[:valid].double().square().sum())
    diagnostics = _diagnostics(totals, hidden.dtype)
    diagnostics['correction_to_hidden_norm'] = math.sqrt(totals['proposed_square_sum'] / max(hidden_square_sum, 1e-12))
    diagnostics['native_effective_to_hidden_norm'] = math.sqrt(totals['native_square_sum'] / max(hidden_square_sum, 1e-12))
    diagnostics['bf16_effective_to_hidden_norm'] = math.sqrt(totals['bf16_square_sum'] / max(hidden_square_sum, 1e-12))
    if work is not None:
        work['completed_train_probe_calls'] += 1
        work['train_probe_scored_contexts'] += len(hidden)
    return {**{key: value / len(hidden) for key, value in sums.items()},
            'nll': sums['ce'] / len(hidden), 'diagnostics': diagnostics}


def _checkpoint_metadata(module, identities, tensor_binding, probe_spec, contract,
                         update, probe, development, controller, source, run_status):
    return dict(format=FORMAT, config=asdict(module.config), hidden_size=module.hidden_size,
                identities=copy.deepcopy(identities), binding_sha256=_identities(identities),
                **tensor_binding, probe_spec=copy.deepcopy(probe_spec),
                development_contract=copy.deepcopy(contract), update=update,
                train_probe=copy.deepcopy(probe), development=copy.deepcopy(development),
                checkpoint_selection='minimum_development_full_document_nll',
                adaptive_controller_state=copy.deepcopy(controller),
                development_metrics=copy.deepcopy(development),
                source_sha256=_json_hash(source), source_file_sha256=source,
                run_status=run_status, inference_only=True)


def _save_checkpoint(path, module, **metadata):
    state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    data = {**metadata, 'state_sha256': _tensor_hash(state)}
    data['metadata_sha256'] = _json_hash(data)
    data['state'] = state
    _atomic_write(path, lambda temporary: torch.save(data, temporary))


def load_recovery_adapter_checkpoint(path, identities, device='cpu', expected_config=None,
                                     expected_hidden_size=None):
    """Restricted weights-only inference loader; no resume or optimizer state.

    Hashes bind corruption/mismatches, not malicious complete-artifact rewrites.
    Source hashes must match this implementation; they are not remote attestation.
    """
    contract = _development_contract(identities)
    path = Path(path)
    if path.stat().st_size > _MAX_CHECKPOINT_BYTES:
        raise ValueError('recovery adapter checkpoint exceeds size budget')
    payload = path.read_bytes()
    data = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    keys = {'format', 'config', 'hidden_size', 'identities', 'binding_sha256',
            'train_inputs_sha256', 'cached_native_sha256', 'provenance_sha256', 'probe_spec',
            'development_contract', 'update', 'train_probe', 'development',
            'checkpoint_selection', 'adaptive_controller_state', 'source_sha256',
            'source_file_sha256', 'run_status', 'inference_only', 'state_sha256',
            'metadata_sha256', 'state', 'development_metrics'}
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError('invalid recovery adapter checkpoint structure')
    metadata = {key: value for key, value in data.items() if key not in ('state', 'metadata_sha256')}
    if data['metadata_sha256'] != _json_hash(metadata):
        raise ValueError('recovery adapter metadata binding mismatch')
    if (data['format'] != FORMAT or data['identities'] != identities
            or data['binding_sha256'] != _identities(identities)
            or data['development_contract'] != contract):
        raise ValueError('recovery adapter identity/format mismatch')
    try:
        config = RecoveryAdapterConfig(**data['config'])
        config.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid recovery adapter configuration') from exc
    hidden, update = data['hidden_size'], data['update']
    if type(hidden) is not int or not config.rank <= hidden <= _MAX_HIDDEN_SIZE:
        raise ValueError('invalid recovery adapter hidden size')
    if type(update) is not int or not 0 <= update <= config.updates:
        raise ValueError('invalid recovery adapter update')
    if expected_config is not None and data['config'] != asdict(expected_config):
        raise ValueError('recovery adapter expected configuration mismatch')
    if expected_hidden_size is not None and hidden != expected_hidden_size:
        raise ValueError('recovery adapter expected hidden size mismatch')
    if (data['inference_only'] is not True
            or data['checkpoint_selection'] != 'minimum_development_full_document_nll'
            or data['run_status'] not in ('evaluated', 'completed', 'guard_fallback_native')):
        raise ValueError('invalid recovery adapter deployment contract')
    if data['run_status'] == 'guard_fallback_native' and update != 0:
        raise ValueError('guard fallback must be native update zero')
    for key in ('train_inputs_sha256', 'cached_native_sha256', 'provenance_sha256', 'state_sha256', 'source_sha256'):
        if not _sha256_string(data[key]):
            raise ValueError('invalid recovery adapter digest')
    source = data['source_file_sha256']
    if (not isinstance(source, dict) or set(source) != set(_SOURCE_FILES)
            or any(not _sha256_string(value) for value in source.values())
            or data['source_sha256'] != _json_hash(source)):
        raise ValueError('recovery adapter source binding mismatch')
    if source != _source_hashes():
        raise ValueError('recovery adapter source implementation mismatch')
    spec = data['probe_spec']
    if (not isinstance(spec, dict) or set(spec) != {'split', 'contexts', 'chunk_contexts', 'last_chunk_padding', 'order'}
            or spec['split'] != 'train' or type(spec['contexts']) is not int
            or not 1 <= spec['contexts'] <= config.max_train_contexts
            or spec['chunk_contexts'] != config.probe_chunk_contexts
            or spec['last_chunk_padding'] != 'repeat_last_masked' or spec['order'] != 'cached_index'):
        raise ValueError('invalid recovery adapter probe contract')
    probe = data['train_probe']
    if not isinstance(probe, dict) or set(probe) != {'loss', 'ce', 'native_kl', 'relative_correction_l2', 'nll', 'diagnostics'}:
        raise ValueError('invalid recovery training probe')
    for key in ('ce', 'relative_correction_l2', 'nll'):
        _finite_number(probe[key], 'probe ' + key)
    _finite_number(probe['loss'], 'probe loss', nonnegative=False)
    _finite_number(probe['native_kl'], 'probe native_kl', nonnegative=False)
    if probe['ce'] != probe['nll'] or not isinstance(probe['diagnostics'], dict):
        raise ValueError('invalid recovery training probe diagnostics')
    _validate_development(data['development'], contract)
    if data['development_metrics'] != data['development']:
        raise ValueError('development metrics binding mismatch')
    audit = AdaptiveFitController.from_snapshot(data['adaptive_controller_state'])
    if (audit.config != AdaptiveFitConfig.from_dict(config.adaptive) or audit.initial_lr != config.learning_rate
            or audit.last_update != update):
        raise ValueError('recovery adaptive audit/configuration mismatch')
    state = data['state']
    if not isinstance(state, dict) or set(state) != {'down', 'up'}:
        raise ValueError('invalid recovery adapter state keys')
    for key, shape in [('down', (config.rank, hidden)), ('up', (hidden, config.rank))]:
        value = state[key]
        if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())):
            raise ValueError('invalid recovery adapter weight shape/dtype/values')
    if _tensor_hash(state) != data['state_sha256']:
        raise ValueError('recovery adapter state digest mismatch')
    if update == 0 and bool(torch.count_nonzero(state['up'])):
        raise ValueError('initial/native fallback has nonzero correction')
    module = SecondRoundResidual(hidden, config, device)
    module.load_state_dict(state, strict=True)
    module.requires_grad_(False).eval()
    return module, {**metadata, 'metadata_sha256': data['metadata_sha256'],
                    'checkpoint_sha256': hashlib.sha256(payload).hexdigest()}


def _train_recovery_adapter(adapter, train_h0, train_targets, development_evaluator,
                            config, identities, output_dir, failure_audit, provenance):
    """Fit train-only objective; select ONLY by independent full development.

    Development patience, optional training-optimizer plateau and the hard budget
    are distinct stopping reasons. Neither training stop nor train loss selects
    the returned checkpoint.

    Callback declares ``.contract={role:'development',document_ids:[...],
    protocol:'full_document_online_nll'}`` and returns these fields plus
    ``status:'ok',nll:finite_nonnegative``. Identities bind disjoint nonempty
    train/development.selected_document_ids. There is no selection/test
    callback. All evaluated checkpoints are recorded; update zero wins ties.
    Numerical or callback guard failures return exact native update zero and
    overwrite best/latest with audited native fallback. No resume/overwrite.
    """
    config = copy.deepcopy(config)
    config.validate()
    identities = copy.deepcopy(identities)
    _assert_frozen(adapter)
    contract = _development_contract(identities, development_evaluator)
    if type(adapter.hidden_size) is not int or not config.rank <= adapter.hidden_size <= _MAX_HIDDEN_SIZE:
        raise ValueError('invalid hidden size/rank budget')
    if (not isinstance(train_h0, torch.Tensor) or train_h0.ndim != 2
            or train_h0.shape[1] != adapter.hidden_size
            or not 1 <= len(train_h0) <= config.max_train_contexts
            or train_h0.dtype not in (torch.float32, torch.bfloat16)
            or not bool(torch.isfinite(train_h0).all())):
        raise ValueError('invalid/beyond-budget finite FP32/BF16 training contexts')
    if (not isinstance(train_targets, torch.Tensor) or train_targets.dtype != torch.long
            or train_targets.shape != (len(train_h0),)):
        raise ValueError('training targets must be int64 [contexts], already shifted')
    vocab = adapter.model.config.vocab_size
    if type(vocab) is not int or not 1 <= vocab <= 262144:
        raise ValueError('vocabulary exceeds bounded full-soft-teacher budget')
    if bool(((train_targets < 0) | (train_targets >= vocab)).any()):
        raise ValueError('training targets outside vocabulary')
    if provenance is not None:
        if not isinstance(provenance, list) or len(provenance) != len(train_h0):
            raise ValueError('provenance must contain one record per training context')
        seen = set()
        for index, record in enumerate(provenance):
            if (not isinstance(record, dict) or record.get('role') != 'train'
                    or record.get('document_id') not in identities['train']['selected_document_ids']
                    or type(record.get('token_position')) is not int or record['token_position'] < 0
                    or type(record.get('target')) is not int or record['target'] != int(train_targets[index])):
                raise ValueError('invalid training provenance/label/role')
            point = (record['document_id'], record['token_position'])
            if point in seen:
                raise ValueError('duplicate training provenance point')
            seen.add(point)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    failure_audit.update(created=True, stage='initialization')
    start = time.perf_counter()
    source = _source_hashes()
    h0, labels = train_h0.detach().to(adapter.device), train_targets.detach().to(adapter.device)
    binding = dict(train_inputs_sha256=_tensor_hash({'h0': train_h0, 'targets': train_targets}),
                   provenance_sha256=_json_hash(provenance))
    failure_audit['stage'] = 'native_cache'
    failure_audit['native_cache_completed_batches'] = 0
    with torch.no_grad():
        pieces = []
        for h, _, valid in _fixed_chunks(h0, labels, config.probe_chunk_contexts):
            native = adapter.step_pairs(h, adapter.native_pair(h))
            if native.shape != h.shape or native.dtype != h.dtype or not bool(torch.isfinite(native).all()):
                raise ValueError('invalid cached native R1 completion')
            pieces.append(native[:valid].detach())
            failure_audit['native_cache_completed_batches'] += 1
        h1 = torch.cat(pieces)
    binding['cached_native_sha256'] = _tensor_hash({'h1': h1})
    spec = dict(split='train', contexts=len(h1), chunk_contexts=config.probe_chunk_contexts,
                last_chunk_padding='repeat_last_masked', order='cached_index')
    work = dict(native_cache_terminal_ffn_batches=len(pieces), native_cache_valid_contexts=len(h1),
                native_cache_contexts_including_padding=len(pieces) * config.probe_chunk_contexts,
                optimizer_head_contexts_including_padding=0, optimizer_backward_valid_contexts=0,
                teacher_head_contexts_including_padding=0, train_probe_calls=0,
                completed_train_probe_calls=0, train_probe_head_contexts_including_padding=0,
                train_probe_scored_contexts=0, development_evaluator_calls=0,
                completed_development_evaluator_calls=0, development_evaluator_seconds=0.)
    failure_audit['work_accounting'] = work
    module = SecondRoundResidual(adapter.hidden_size, config, adapter.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=config.learning_rate, weight_decay=0.)
    controller = AdaptiveFitController(AdaptiveFitConfig.from_dict(config.adaptive), initial_lr=config.learning_rate)
    rng = torch.Generator(device='cpu').manual_seed(config.seed)
    initial_state = copy.deepcopy(module.state_dict())
    # Invalid initial evaluation raises before publishing any inference checkpoint.
    failure_audit['stage'] = 'initial_training_probe'
    initial_probe = _training_probe(adapter, module, h1, labels, config, work)
    failure_audit['stage'] = 'initial_development'
    initial_dev = _evaluate_development(adapter, module, development_evaluator, contract, work)
    decision = controller.observe(0, initial_probe['loss'], coverage_complete=False)
    initial_controller = controller.snapshot()
    best_state = copy.deepcopy(initial_state)
    best_probe = latest_probe = initial_probe
    best_dev = latest_dev = initial_dev
    best_controller = latest_controller = initial_controller
    best_update = latest_update = successful = completed_epochs = 0
    rows = []
    history = [dict(update=0, train_probe=initial_probe, development=initial_dev,
                    adaptive_decision=decision, development_improved=False,
                    checkpoint='checkpoint_eval_000000.pt')]

    def save(name, update, probe, dev, audit, run_status='evaluated'):
        metadata = _checkpoint_metadata(module, identities, binding, spec, contract,
                                        update, probe, dev, audit, source, run_status)
        _save_checkpoint(output / name, module, **metadata)

    failure_audit['stage'] = 'checkpoint_or_training_io'
    for name in ('checkpoint_initial.pt', 'checkpoint_best.pt', 'checkpoint_eval_000000.pt'):
        save(name, 0, initial_probe, initial_dev, initial_controller)
    status, stop_reason, guard_detail = 'completed', 'update_budget_exhausted', None
    permutation, cursor, epoch = [], len(h1), 0
    visits, dev_stale, progress_anchor = [0] * len(h1), 0, initial_dev['nll']
    gradient_window = dict(start_update_exclusive=0, accepted_updates=0,
                           nonzero_gradient_updates=0, max_gradient_norm=0.)
    for update in range(1, config.updates + 1):
        if cursor == len(h1):
            permutation = torch.randperm(len(h1), generator=rng).tolist()
            cursor, epoch = 0, epoch + 1
        indices = permutation[cursor:cursor + config.batch_contexts]
        cursor += len(indices)
        idx = torch.tensor(indices, dtype=torch.long, device=h1.device)
        row = dict(update=update, epoch=epoch, visited_indices=indices, batch_contexts=len(indices),
                   accepted=False, learning_rate=optimizer.param_groups[0]['lr'])
        rows.append(row)
        module.train()
        optimizer.zero_grad(set_to_none=True)
        stage = 'loss'
        try:
            batch_sums = dict(loss=0., ce=0., native_kl=0., relative_correction_l2=0.)
            for h, y, valid in _fixed_chunks(h1[idx], labels[idx], config.probe_chunk_contexts):
                work['optimizer_head_contexts_including_padding'] += 2 * len(h)
                work['teacher_head_contexts_including_padding'] += len(h)
                parts = _loss_components(adapter, module, h, y, config)
                for key in batch_sums:
                    batch_sums[key] += float(parts[key][:valid].detach().double().sum())
                (parts['loss'][:valid].sum() / len(indices)).backward()
                work['optimizer_backward_valid_contexts'] += valid
                del parts
            row['train_batch'] = {key: value / len(indices) for key, value in batch_sums.items()}
            stage = 'gradient'
            _assert_frozen(adapter)
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in module.parameters()):
                raise FloatingPointError('nonfinite or absent adapter gradient')
            grad = torch.nn.utils.clip_grad_norm_(module.parameters(), config.gradient_clip)
            if not bool(torch.isfinite(grad)):
                raise FloatingPointError('nonfinite gradient norm')
            row['gradient_norm'] = float(grad)
            stage = 'parameters'
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in module.parameters()):
                raise FloatingPointError('nonfinite adapter parameters')
            stage = 'correction'
            with torch.no_grad():
                totals = _correction_totals(module, h1[idx])
                diagnostic = _diagnostics(totals, h1.dtype)
                diagnostic['correction_to_hidden_norm'] = math.sqrt(
                    totals['proposed_square_sum'] / max(float(h1[idx].double().square().sum()), 1e-12))
                row['correction_diagnostics'] = diagnostic
            probe_due = update % config.eval_interval == 0 or update == config.updates
            if probe_due:
                stage = 'training_probe'
                probe = _training_probe(adapter, module, h1, labels, config, work)
                stage = 'development'
                development = _evaluate_development(adapter, module, development_evaluator, contract, work)
            # Only fully guarded, accepted updates may support BF16 LR growth.
            gradient_window['accepted_updates'] += 1
            gradient_window['nonzero_gradient_updates'] += int(row['gradient_norm'] > 0.)
            gradient_window['max_gradient_norm'] = max(gradient_window['max_gradient_norm'], row['gradient_norm'])
            row['probe_interval_gradients'] = gradient_window.copy()
            if probe_due:
                covered = all(count > 0 or i in indices for i, count in enumerate(visits))
                ineffective = (h1.dtype == torch.bfloat16
                    and probe['diagnostics']['native_effective_change_rate'] == 0.
                    and gradient_window['nonzero_gradient_updates'] > 0)
                decision = controller.observe(update, probe['loss'], covered, ineffective)
                for group in optimizer.param_groups:
                    group['lr'] = decision['lr']
                latest_controller = controller.snapshot()
                latest_probe, latest_dev, latest_update = probe, development, update
                improved = development['nll'] < best_dev['nll']
                if development['nll'] < progress_anchor - config.development_min_delta:
                    progress_anchor, dev_stale = development['nll'], 0
                else:
                    dev_stale += 1
                filename = f'checkpoint_eval_{update:06d}.pt'
                save(filename, update, probe, development, latest_controller)
                history.append(dict(update=update, train_probe=probe, development=development,
                    adaptive_decision=decision, probe_interval_gradients=gradient_window.copy(),
                    development_improved=improved, development_stale_evaluations=dev_stale,
                    checkpoint=filename))
                row['adaptive_decision'] = decision
                row['next_learning_rate'] = decision['lr']
                row['development_nll'] = development['nll']
                if improved:
                    best_probe, best_dev, best_update = probe, development, update
                    best_state = copy.deepcopy(module.state_dict())
                    best_controller = copy.deepcopy(latest_controller)
                    save('checkpoint_best.pt', update, probe, development, best_controller)
                gradient_window = dict(start_update_exclusive=update, accepted_updates=0,
                    nonzero_gradient_updates=0, max_gradient_norm=0.)
            row['accepted'] = True
            successful += 1
            for i in indices:
                visits[i] += 1
            completed_epochs += int(cursor == len(h1))
            row['unique_contexts_visited'] = sum(count > 0 for count in visits)
            row['completed_epochs'] = completed_epochs
            if (probe_due and update >= config.min_updates
                    and dev_stale >= config.development_patience and all(count > 0 for count in visits)):
                stop_reason = 'development_patience_exhausted'
                break
            if probe_due and decision['stop']:
                stop_reason = 'adaptive_train_plateau'
                break
        except (FloatingPointError, ValueError, RuntimeError) as exc:
            status, stop_reason = 'stopped_guard', 'guard_' + stage
            guard_detail = type(exc).__name__ + ': ' + str(exc)
            row['guard'] = dict(stage=stage, detail=guard_detail)
            break
    selected_before_guard = dict(update=best_update, development_nll=best_dev['nll'])
    if status != 'completed':
        # Never publish a trained candidate after a numerical/callback failure.
        module.config, module.hidden_size = config, adapter.hidden_size
        module.load_state_dict(initial_state)
        best_update = latest_update = 0
        best_probe = latest_probe = initial_probe
        best_dev = latest_dev = initial_dev
        best_controller = latest_controller = initial_controller
        best_state = copy.deepcopy(initial_state)
        for name in ('checkpoint_best.pt', 'checkpoint_latest.pt'):
            save(name, 0, initial_probe, initial_dev, initial_controller, 'guard_fallback_native')
    else:
        save('checkpoint_latest.pt', latest_update, latest_probe, latest_dev, latest_controller, 'completed')
        module.load_state_dict(best_state)
        save('checkpoint_best.pt', best_update, best_probe, best_dev, best_controller, 'completed')
    module.load_state_dict(best_state)
    module.zero_grad(set_to_none=True)
    module.requires_grad_(False).eval()
    _assert_frozen(adapter)
    work['optimizer_accepted_context_visits'] = sum(visits)
    work['total_cached_head_contexts_including_padding'] = (
        work['optimizer_head_contexts_including_padding'] + work['train_probe_head_contexts_including_padding'])
    summary = dict(format=FORMAT, experiment='native_regularized_adapter_development_selection',
        status=status, stop_reason=stop_reason, guard_detail=guard_detail, config=asdict(config),
        checkpoint_selection='minimum_development_full_document_nll',
        development_contract=contract, initial_development_nll=initial_dev['nll'],
        best_development_nll=best_dev['nll'], latest_development_nll=latest_dev['nll'],
        best_update=best_update, latest_update=latest_update,
        development_improved=best_dev['nll'] < initial_dev['nll'],
        initial_development_metrics=initial_dev, best_development_metrics=best_dev,
        latest_development_metrics=latest_dev,
        development_stale_evaluations=dev_stale,
        deployment_eligible=status == 'completed', selected_before_guard=selected_before_guard,
        initial_train_probe=initial_probe, best_train_probe=best_probe, latest_train_probe=latest_probe,
        successful_updates=successful, attempted_updates=len(rows), train_contexts=len(h1),
        unique_contexts_visited=sum(count > 0 for count in visits), context_visit_counts=visits,
        context_visits=sum(visits), completed_epochs=completed_epochs,
        coverage_complete=all(count > 0 for count in visits), effective_train_epochs=sum(visits) / len(h1),
        evaluation_history=history, train_probe_history=[dict(update=e['update'], **e['train_probe']) for e in history],
        probe_spec=spec, work_accounting=work,
        adaptive_controller_state=controller.snapshot(), best_adaptive_controller_state=best_controller,
        adaptive_controls='train_probe_total_objective_learning_rate_and_optimizer_plateau_stop',
        adaptive_stopped=stop_reason == 'adaptive_train_plateau',
        development_controls='checkpoint_selection_and_early_stopping',
        trainable_parameters=sum(p.numel() for p in module.parameters()),
        source_sha256=_json_hash(source), source_file_sha256=source,
        binding_sha256=_identities(identities), **binding,
        native_teacher=dict(distribution='full_vocabulary', direction='KL(native||corrected)',
            temperature=1., detached=True, persistent_logits_cached=False,
            chunk_contexts=config.probe_chunk_contexts, max_logit_elements_per_tensor=config.probe_chunk_contexts * vocab,
            reference='same_cached_h1_and_same_readout_shape_as_corrected'),
        added_capacity=True, recurrent_ffn=False, learned_stop=False, backbone_gradients_absent=True,
        selection_used=False, heldout_used=False, heldout_gain_established=False, practical_gain_established=False,
        limits=['development is not independent test', 'beta/rank/scale defaults are provisional ablations, not validated optima',
                'chunked full-vocabulary teacher adds a frozen head forward per training/probe chunk',
                'counts exclude evaluator internals; development calls and wall time are reported separately',
                'BF16 effective residual diagnostics are measurements, not a quantization fix',
                'full-document deployment can differ from cached context arithmetic',
                'no recurrent FFN, learned STOP, optimizer resume, or automatic update-budget expansion'],
        elapsed_seconds=time.perf_counter() - start)
    for name, value in [('metrics.json', rows), ('summary.json', summary)]:
        _atomic_write(output / name, lambda p, value=value: p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'))
    return module, summary


def train_recovery_adapter(adapter, train_h0, train_targets, development_evaluator,
                           config, identities, output_dir, *, provenance=None):
    """Public guarded trainer; see implementation for the evaluator contract.

    Initialization/callback failures are reported then raised, without publishing
    an uncertified inference checkpoint. Existing output directories are never
    modified. Later numerical failures return an audited exact-native fallback.
    """
    audit = {'created': False, 'stage': 'validation'}
    started = time.perf_counter()
    try:
        return _train_recovery_adapter(adapter, train_h0, train_targets,
            development_evaluator, config, identities, output_dir, audit, provenance)
    except Exception as exc:
        if audit['created']:
            failure = dict(format=FORMAT, status='failed', stop_reason='guard_' + audit['stage'],
                guard_detail=type(exc).__name__ + ': ' + str(exc), deployment_eligible=False,
                initialization_or_io_failure=True, config=asdict(config),
                work_accounting=audit.get('work_accounting', {}),
                native_cache_completed_batches=audit.get('native_cache_completed_batches', 0),
                elapsed_seconds=time.perf_counter() - started)
            _atomic_write(Path(output_dir) / 'summary.json', lambda p:
                p.write_text(json.dumps(failure, indent=2, allow_nan=False) + '\n'))
        raise
