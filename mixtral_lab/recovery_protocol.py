"""Versioned four-role development-best recovery contracts.

Training, checkpoint-development, outer selection, and numerical calibration are
separate populations. Outer selection was historically inspected: it is NOT a
final held-out test. No newtest input is supported. Evidence is integrity checked,
not authenticated. Local artifacts must remain immutable; stat snapshots reject
ordinary changes and the model loader hashes actual weight bytes once per stage.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch
from mixtral_lab.fit_protocol import finite_tree, experiment_contract, seal_report
from mixtral_lab.protocol import canonical_hash, integer, sha256_file, validate_config

SCHEMA = 'mixtral-generalization-recovery-v1'
REPORT_SCHEMA = 'mixtral-recovery-report-v1'
FREEZE_SCHEMA = 'mixtral-development-best-freeze-v1'
ROLES = ('train', 'development', 'selection', 'calibration')
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_SOURCE_BYTES = 128 * 1024 * 1024
PROTOCOL = 'full_document_online_nll'


def trainer_api(kind):
    if kind == 'router':
        from mixtral_lab.recovery_router import RecoveryRouterConfig, train_recovery_router, load_recovery_checkpoint
        return RecoveryRouterConfig, train_recovery_router, load_recovery_checkpoint
    if kind == 'adapter':
        from mixtral_lab.recovery_adapter import RecoveryAdapterConfig, train_recovery_adapter, load_recovery_adapter_checkpoint
        return RecoveryAdapterConfig, train_recovery_adapter, load_recovery_adapter_checkpoint
    raise ValueError('recovery.kind must be router or adapter')


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite positive')
    return value


def validate_recovery_config(config):
    finite_tree(config, 'config')
    if not isinstance(config, dict) or config.get('schema') != SCHEMA:
        raise ValueError('unknown recovery schema; legacy fit-first configs are not recovery configs')
    for key in ('data_files', 'sequences'):
        if not isinstance(config.get(key), dict) or set(config[key]) != set(ROLES):
            raise ValueError(f'{key} must specify exactly train, development, selection, calibration; newtest is forbidden')
    # Validate all original global safety bounds on a stripped compatibility view.
    # Neither legacy train-best/selection-best settings nor their default behavior
    # are scientific inputs to this new protocol.
    if any(key in config for key in ('fit', 'training', 'second_round', 'oracle')):
        raise ValueError('legacy fit/training/second_round/oracle sections are not accepted in recovery')
    base = deepcopy(config)
    base['schema'] = 'mixtral-frozen-routing-v1'
    for key in ('data_files', 'sequences'):
        base[key] = {role: config[key][role] for role in ('train', 'selection', 'calibration')}
    base['training'] = {'batch_contexts': 1, 'max_train_contexts': config.get('max_train_contexts'), 'seed': config.get('seed')}
    validate_config(base, stage='train')
    config['context_sampling'] = deepcopy(base['context_sampling'])
    resolved = []
    for role in ROLES:
        integer(config['sequences'][role], f'sequences.{role}', 1, config['max_documents_per_split'])
        path = config['data_files'][role]
        if not isinstance(path, str) or not path.strip() or 'newtest' in path.casefold():
            raise ValueError('invalid recovery data path or forbidden newtest path')
        resolved.append(str(Path(path).resolve()))
    if len(set(resolved)) != len(ROLES):
        raise ValueError('all four recovery roles must use different files')
    settings = config.get('recovery')
    required = {'kind', 'training', 'development_min_gain', 'selection_min_gain', 'bridge_tolerances'}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError('recovery must specify exactly kind, training, development_min_gain, selection_min_gain, bridge_tolerances')
    for key in ('development_min_gain', 'selection_min_gain'):
        positive(settings[key], 'recovery.' + key)
    tolerances = settings['bridge_tolerances']
    allowed_tolerances = {'hidden_max_abs', 'feature_max_abs', 'ce_max_abs'}
    if not isinstance(tolerances, dict) or set(tolerances) != allowed_tolerances:
        raise ValueError('bridge_tolerances must explicitly specify hidden_max_abs, feature_max_abs, ce_max_abs')
    for key, value in tolerances.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'bridge_tolerances.{key} must be finite nonnegative')
    if not isinstance(settings['training'], dict):
        raise ValueError('recovery.training must be an object')
    config_class, _, _ = trainer_api(settings['kind'])
    training = config_class(**settings['training'])
    training.validate()
    integer(training.updates, 'recovery.updates', 1, 10000)
    integer(training.batch_contexts, 'recovery.batch_contexts', 1, 1024)
    if training.max_train_contexts != config['max_train_contexts'] or training.seed != config['seed']:
        raise ValueError('recovery training and runner max_train_contexts/seed must match')
    if training.batch_contexts > training.max_train_contexts:
        raise ValueError('recovery batch_contexts exceeds exact training context budget')
    if settings['kind'] == 'router':
        integer(training.max_score_calls, 'recovery.max_score_calls', 30 * config['max_train_contexts'], 30 * 4096)
    settings['training'] = asdict(training)
    return config


def read_documents(config, *, include_unselected=False):
    """Check the complete bounded raw sources, not just requested first rows."""
    documents, identities, ids, contents = {}, {}, set(), set()
    for role in ROLES:
        path = Path(config['data_files'][role])
        if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError('recovery JSONL source missing or exceeds 128 MiB bound')
        source = path.read_bytes()
        if len(source) > MAX_SOURCE_BYTES:
            raise ValueError('recovery JSONL source exceeds 128 MiB bound')
        rows = []
        for line_number, line in enumerate(source.decode('utf-8').splitlines(), 1):
            if not line.strip():
                continue
            if len(rows) >= config['max_documents_per_split']:
                raise ValueError('document file exceeds declared safety limit; prepare an explicit bounded source')
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{role}:{line_number} must be a document object')
            ident, text = row.get('id'), row.get('text')
            if not isinstance(ident, str) or not ident.strip() or not isinstance(text, str) or not text.strip():
                raise ValueError(f'{role}:{line_number} requires nonempty string id and text')
            if len(text) > config['max_text_characters']:
                raise ValueError('document exceeds character limit')
            digest = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
            if ident in ids or digest in contents:
                raise ValueError('duplicate document id or normalized content within/across recovery roles')
            ids.add(ident); contents.add(digest)
            rows.append({'id': ident, 'text': text, 'text_sha256': digest})
        count = config['sequences'][role]
        if len(rows) < count:
            raise ValueError(f'{role} has fewer documents than requested sequences')
        identities[role] = {'path': str(path.resolve()), 'file_sha256': hashlib.sha256(source).hexdigest(),
            'documents': [{'id': r['id'], 'text_sha256': r['text_sha256']} for r in rows],
            'selected_document_ids': [r['id'] for r in rows[:count]], 'declared_role': role}
        documents[role] = rows if include_unselected else rows[:count]
    return documents, identities


def prepare_data(config, tokenizer):
    """All-role raw AND effective-token disjointness before any weight loading.

    Effective-token checks cover every bounded source row, including unselected
    rows. Duplicate truncations within a role are reported, never silently removed.
    This catches exact equality, not partial-prefix overlap or semantic duplicates.
    """
    documents, identities = read_documents(config, include_unselected=True)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if type(pad) is not int or pad < 0:
        raise ValueError('tokenizer needs a nonnegative pad or eos token')
    result, effective_sequences = {}, {}
    for role, rows in documents.items():
        prepared, within_role, all_tokens = [], {}, []
        for index, row in enumerate(rows):
            ids = tokenizer.encode(row['text'], add_special_tokens=True, truncation=True, max_length=config['sequence_length'])
            if (not isinstance(ids, list) or len(ids) < 2 or len(ids) > config['sequence_length']
                    or any(type(i) is not int or i < 0 for i in ids)):
                raise ValueError('tokenizer returned invalid token IDs or violated the declared sequence tensor quota')
            effective = tuple(ids)
            previous = effective_sequences.get(effective)
            if previous is not None and previous[0] != role:
                raise ValueError(f'duplicate effective token sequence across recovery roles: {previous[0]}:{previous[1]} and {role}:{row["id"]}')
            effective_sequences[effective] = (role, row['id'])
            all_tokens.append({'id': row['id'], 'tokens_sha256': canonical_hash(ids)})
            if index >= config['sequences'][role]:
                continue
            within_role.setdefault(effective, []).append(row['id'])
            valid = len(ids)
            prepared.append({'id': row['id'],
                'input_ids': torch.tensor(ids + [pad] * (config['sequence_length'] - valid), dtype=torch.long),
                'attention_mask': torch.tensor([1] * valid + [0] * (config['sequence_length'] - valid), dtype=torch.long)})
        token_identity = [{'id': r['id'], 'input_ids': r['input_ids'].tolist(), 'attention_mask': r['attention_mask'].tolist()} for r in prepared]
        identities[role].update(tokenized_sha256=canonical_hash(token_identity),
            all_source_effective_tokens_sha256=canonical_hash(all_tokens),
            supervised_tokens=sum(int((r['attention_mask'][:-1] * r['attention_mask'][1:]).sum()) for r in prepared))
        groups = [{'effective_tokens_sha256': canonical_hash(list(tokens)), 'document_ids': doc_ids,
                   'effective_token_count': len(tokens)} for tokens, doc_ids in within_role.items() if len(doc_ids) > 1]
        identities[role]['effective_input_statistics'] = {'documents': len(prepared),
            'unique_effective_sequences': len(within_role), 'redundant_documents': len(prepared) - len(within_role),
            'redundant_document_fraction': (len(prepared) - len(within_role)) / len(prepared),
            'documents_in_duplicate_groups': sum(len(g['document_ids']) for g in groups), 'duplicate_groups': groups,
            'scope': 'exact effective token equality after truncation, excluding padding; all source rows checked across roles',
            'sampling_changed': False}
        result[role] = prepared
    return result, identities


def read_report(path, stage):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError('recovery report missing or exceeds 64 MiB bound')
    snapshot = path.read_bytes()
    if len(snapshot) > MAX_REPORT_BYTES:
        raise ValueError('recovery report exceeds 64 MiB bound')
    report = json.loads(snapshot)
    finite_tree(report, 'report')
    if (not isinstance(report, dict) or report.get('schema') != REPORT_SCHEMA or report.get('stage') != stage
            or report.get('status') != 'complete' or report.get('model_load_started') is not True):
        raise ValueError(f'requires complete {stage} recovery report, not failed/running/legacy evidence')
    if report.get('report_sha256') != seal_report(report)['report_sha256']:
        raise ValueError('recovery report digest mismatch')
    if (report.get('config_sha256') != canonical_hash(report.get('config', {}))
            or report.get('experiment_contract_sha256') != canonical_hash(experiment_contract(report.get('config', {})))):
        raise ValueError('recovery report scientific config digest mismatch')
    if report.get('native_equivalence', {}).get('eligible') is not True:
        raise ValueError('recovery report lacks passed native equivalence')
    require_bridge_evidence(report.get('bridge_pretraining'), native_only=True, config=report['config'],
        manifest=report.get('identity', {}).get('context_sampling'))
    return report


def validate_bridge_evidence(bridge, *, config=None, manifest=None, chosen_update=None):
    """Recompute numerical verdicts, not authenticity or mathematical equivalence."""
    finite_tree(bridge, 'bridge')
    if (not isinstance(bridge, dict) or bridge.get('status') != 'complete'
            or bridge.get('schema') != 'mixtral-cache-online-bridge-v1'
            or bridge.get('role') != 'train_only' or bridge.get('kind') not in ('router', 'adapter')):
        raise ValueError('missing complete cache/online bridge diagnostic')
    if config is not None:
        expected = {'logit_max_abs': config['logit_max_abs_tolerance'], 'ce_mean_abs': config['ce_mean_abs_tolerance'],
                    **config['recovery']['bridge_tolerances']}
        if (bridge.get('tolerances') != expected or bridge['kind'] != config['recovery']['kind']
                or bridge.get('candidate_training_config') != config['recovery']['training']):
            raise ValueError('bridge kind/tolerances differ from frozen scientific contract')
    if manifest is not None and bridge.get('sampling_sha256') != manifest['sampling_sha256']:
        raise ValueError('bridge sampling identity differs from frozen train contexts')
    if bridge.get('chosen_update') != chosen_update:
        raise ValueError('bridge chosen update differs from frozen checkpoint')
    tolerances = bridge.get('tolerances', {})
    metric_keys = {'logits': 'logit_max_abs', 'hidden': 'hidden_max_abs', 'h0': 'hidden_max_abs',
                   'h1': 'hidden_max_abs', 'features': 'feature_max_abs', 'ce': 'ce_max_abs'}
    required = {'cached_vs_online.h0', 'cached_native_vs_zero.ce'}
    for prefix in ('native_vs_controlled', 'native_vs_zero', 'cached_vs_online.native',
                   'cached_vs_online.zero', 'cached_vs_online.learned'):
        required.update(prefix + '.' + suffix for suffix in ('logits', 'hidden', 'ce'))
    if bridge['kind'] == 'router':
        required.update({'cached_vs_online.features', 'cached_readout_vs_score_pairs.ce'})
        required.update('cached_vs_online.forced_cached.' + suffix for suffix in ('logits', 'hidden', 'ce'))
    else:
        required.add('cached_vs_online.h1')
    cached = bridge.get('cached')
    if not isinstance(cached, dict) or set(cached) != {'native', 'zero', 'learned'}:
        raise ValueError('bridge cached metric populations missing')
    for metric in cached.values():
        integer(metric.get('tokens'), 'bridge cached tokens', 1)
        if manifest is not None and metric['tokens'] != manifest['actual_count']:
            raise ValueError('bridge cached population differs from sampled contexts')
        value = metric.get('nll')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError('invalid bridge cached NLL')
    if bridge['kind'] == 'adapter' and bridge.get('action_agreement') != {}:
        raise ValueError('adapter bridge cannot claim learned routing agreement')
    differences = bridge.get('absolute_differences')
    if not isinstance(differences, dict) or not required.issubset(differences):
        raise ValueError('bridge primary numerical comparisons are missing')
    zero_prefixes = ('native_vs_controlled.', 'native_vs_zero.', 'cached_native_vs_zero.',
                     'cached_vs_online.native.', 'cached_vs_online.zero.', 'cached_vs_online.h0',
                     'cached_vs_online.h1', 'cached_vs_online.features', 'cached_readout_vs_score_pairs.')
    passed, zero_passed, action_passed = [], [], []
    try:
        for name, metric in differences.items():
            integer(metric.get('elements'), 'bridge comparison elements', 1)
            for key in ('mean_abs', 'max_abs'):
                value = metric[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                    raise ValueError('invalid bridge primary absolute difference')
            if metric['mean_abs'] > metric['max_abs']:
                raise ValueError('bridge mean difference exceeds maximum')
            suffix = name.rsplit('.', 1)[-1]
            threshold = tolerances[metric_keys[suffix]]
            if metric.get('max_abs_tolerance') != threshold:
                raise ValueError('bridge comparison tolerance mismatch')
            verdict = metric['max_abs'] <= threshold
            if suffix == 'ce':
                if metric.get('mean_abs_tolerance') != tolerances['ce_mean_abs']:
                    raise ValueError('bridge CE mean tolerance mismatch')
                verdict = verdict and metric['mean_abs'] <= tolerances['ce_mean_abs']
            if metric.get('within_declared_tolerance') is not verdict:
                raise ValueError('bridge primary comparison/verdict mismatch')
            passed.append(verdict)
            if name.startswith(zero_prefixes):
                zero_passed.append(verdict)
        if bridge['kind'] == 'router':
            for key in ('native', 'zero', 'learned'):
                agreement = bridge['action_agreement'][key]
                integer(agreement.get('count'), 'bridge action count', 1)
                integer(agreement.get('agree'), 'bridge action agreement', 0, agreement['count'])
                if manifest is not None and agreement['count'] != manifest['actual_count']:
                    raise ValueError('bridge action population differs from sampled contexts')
                if not math.isclose(agreement.get('fraction', -1.), agreement['agree'] / agreement['count'], abs_tol=1e-12, rel_tol=1e-12):
                    raise ValueError('bridge action agreement fraction mismatch')
                action_passed.append(agreement['agree'] == agreement['count'])
                if key in ('native', 'zero'):
                    zero_passed.append(agreement['agree'] == agreement['count'])
    except (KeyError, TypeError) as exc:
        raise ValueError('malformed bridge primary numerical evidence') from exc
    if bridge.get('action_agreement_screen_passed') is not all(action_passed):
        raise ValueError('bridge action agreement verdict mismatch')
    passed.extend(action_passed)
    historical = bridge.get('historical_metric_reproduction')
    required_metrics = set() if chosen_update is None else ({'native', 'learned'} if bridge['kind'] == 'router' else {'learned'})
    if not isinstance(historical, dict) or set(historical) != required_metrics:
        raise ValueError('bridge historical metric reproduction missing or has wrong population')
    for name, metric in historical.items():
        for key in ('historical_nll', 'recomputed_nll', 'absolute_difference', 'tolerance'):
            value = metric.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError('invalid bridge historical metric')
        if metric['recomputed_nll'] != cached[name]['nll']:
            raise ValueError('bridge historical reproduction differs from cached primary NLL')
        difference = abs(metric['historical_nll'] - metric['recomputed_nll'])
        if (not math.isclose(difference, metric['absolute_difference'], abs_tol=1e-12, rel_tol=1e-12)
                or (config is not None and metric['tolerance'] != config['nll_tolerance'])):
            raise ValueError('bridge historical metric/tolerance mismatch')
        verdict = difference <= metric['tolerance']
        if metric.get('within_declared_tolerance') is not verdict:
            raise ValueError('bridge historical metric verdict mismatch')
        passed.append(verdict)
    if bridge.get('numerical_screen_passed') is not all(passed) or bridge.get('zero_screen_passed') is not all(zero_passed):
        raise ValueError('bridge aggregate numerical verdict disagrees with primary evidence')
    return bridge


def require_bridge_evidence(bridge, *, native_only=False, config=None, manifest=None, chosen_update=None):
    validate_bridge_evidence(bridge, config=config, manifest=manifest, chosen_update=chosen_update)
    key = 'zero_screen_passed' if native_only else 'numerical_screen_passed'
    if bridge.get(key) is not True:
        raise ValueError(f'cache/online bridge {key} failed; incompatible cached/native shapes cannot support recovery claims')


def validate_paired_metrics(value, role, expected_ids, nll_tolerance):
    """Recompute primary metrics from identical full-document populations."""
    finite_tree(value, role + ' metrics')
    try:
        if (value.get('role') != role or value.get('protocol') != PROTOCOL or value.get('status') != 'ok'
                or value.get('document_ids') != expected_ids or len(set(expected_ids)) != len(expected_ids)):
            raise ValueError('full-document role/protocol/document population mismatch')
        records = value['per_document']
        if not isinstance(records, list) or [r['document_id'] for r in records] != expected_ids:
            raise ValueError('full-document per-document evidence missing or population mismatch')
        for row in records:
            integer(row.get('tokens'), 'per-document tokens', 1)
            for name in ('native', 'controlled', 'candidate'):
                v = row[name + '_nll']
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                    raise ValueError('invalid full-document NLL')
            if abs(row['native_nll'] - row['controlled_nll']) > nll_tolerance:
                raise ValueError('controlled/native per-document screen failed')
            for baseline in ('native', 'controlled'):
                if not math.isclose(row['candidate_nll'] - row[baseline + '_nll'], row['candidate_minus_' + baseline + '_nll'], abs_tol=1e-12, rel_tol=1e-12):
                    raise ValueError('per-document paired delta mismatch')
        tokens = sum(row['tokens'] for row in records)
        integer(tokens, 'full-document tokens', 1)
        for name, key in (('native', 'native'), ('controlled', 'controlled_original'), ('candidate', 'candidate')):
            metric = value[key]
            integer(metric.get('tokens'), 'aggregate tokens', 1)
            integer(metric.get('sequences'), 'aggregate sequences', 1)
            if isinstance(metric.get('nll'), bool) or not isinstance(metric.get('nll'), (int, float)) or metric['nll'] < 0:
                raise ValueError('invalid aggregate NLL')
            if metric['tokens'] != tokens or metric['sequences'] != len(records):
                raise ValueError('full-document metric populations differ')
            mean = sum(row['tokens'] * row[name + '_nll'] for row in records) / tokens
            if not math.isclose(metric['nll'], mean, abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError('per-document and aggregate metrics disagree')
        if not math.isclose(value['nll'], value['candidate']['nll'], abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError('development selection NLL differs from candidate primary metric')
    except (KeyError, TypeError) as exc:
        raise ValueError('malformed full-document primary metrics') from exc
    return value


def require_development_evidence(report):
    summary = report.get('tune_summary', {})
    gate = report.get('development_gate', {})
    threshold = report['config']['recovery']['development_min_gain']
    positive(threshold, 'development gain threshold')
    if summary.get('status') not in ('complete', 'completed'):
        raise ValueError('development tuning did not complete')
    integer(summary.get('best_update'), 'best development update', 0, 10000)
    if report.get('checkpoint', {}).get('update') != summary['best_update']:
        raise ValueError('development best update and checkpoint update mismatch')
    initial, best = summary.get('initial_development_nll'), summary.get('best_development_nll')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in (initial, best)):
        raise ValueError('missing finite development checkpoint evidence')
    gain = initial - best
    if (summary['best_update'] == 0 or gain < threshold or gate.get('passed') is not True
            or gate.get('threshold_nats') != threshold or not math.isclose(gate.get('gain_nats', math.inf), gain, abs_tol=1e-12, rel_tol=1e-12)):
        raise ValueError('positive development gain gate not passed; retain update 0 and block selection/profile')
    bridge = report.get('bridge_frozen_candidate')
    validate_bridge_evidence(bridge, config=report['config'], manifest=report['identity']['context_sampling'],
        chosen_update=report['checkpoint']['update'])
    if report.get('training_path_numerical_screen_passed') is not (bridge.get('numerical_screen_passed') is True):
        raise ValueError('frozen learned-path numerical screen verdict mismatch')


def require_quality_evidence(report, threshold):
    positive(threshold, 'selection gain threshold')
    quality = report.get('evaluation', {})
    validate_paired_metrics(quality, 'selection', report['identity']['selection']['selected_document_ids'], report['config']['nll_tolerance'])
    gate = quality.get('quality_gate', {})
    if gate.get('threshold_nats') != threshold or gate.get('passed') is not True:
        raise ValueError('outer selection quality gate not passed; profiling is blocked')
    for key, baseline in (('gain_vs_native_nats', 'native'), ('gain_vs_controlled_nats', 'controlled_original')):
        measured = quality[baseline]['nll'] - quality['candidate']['nll']
        if measured < threshold or not math.isclose(gate.get(key, math.inf), measured, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError('outer selection quality gate metrics disagree or threshold not passed')
    if gate.get('controlled_native_nll_screen_passed') is not True:
        raise ValueError('outer selection controlled/native numerical screen failed')
    require_bridge_evidence(report.get('bridge_frozen_candidate'), config=report['config'],
        manifest=report['identity']['context_sampling'], chosen_update=report['checkpoint']['update'])


def verify_frozen_report(report, config, checkpoint, current):
    """Reject stale source/data/config/runtime/checkpoint evidence before weights."""
    if experiment_contract(report.get('config', {})) != experiment_contract(config):
        raise ValueError('frozen scientific configuration mismatch (only output_dir may change)')
    freeze = report.get('freeze_manifest')
    if not isinstance(freeze, dict) or freeze.get('schema') != FREEZE_SCHEMA:
        raise ValueError('missing recovery checkpoint freeze manifest')
    if report.get('freeze_manifest_sha256') != canonical_hash(freeze):
        raise ValueError('checkpoint freeze manifest digest mismatch')
    if freeze.get('experiment_contract_sha256') != canonical_hash(experiment_contract(config)):
        raise ValueError('checkpoint scientific contract mismatch')
    identity = report.get('identity')
    if not isinstance(identity, dict) or freeze.get('identity_sha256') != canonical_hash(identity):
        raise ValueError('frozen identity mismatch')
    for key in ('code', 'local_inputs', 'artifact_stats', 'data', 'train_contexts', 'runtime_versions'):
        if freeze.get(key) != current.get(key):
            raise ValueError(f'frozen {key} mismatch before model load')
    if identity.get('code') != current['code']:
        raise ValueError('frozen source identity mismatch')
    for role in ROLES:
        if identity.get(role) != current['data'][role]:
            raise ValueError('frozen four-role data identity mismatch')
    if identity.get('context_sampling') != current['train_contexts']:
        raise ValueError('frozen train sampling identity mismatch')
    baseline = report.get('baseline_checkpoint')
    if not isinstance(baseline, dict) or freeze.get('baseline_checkpoint') != baseline or baseline.get('update') != 0:
        raise ValueError('missing frozen update-zero baseline checkpoint')
    baseline_path = Path(baseline.get('path', ''))
    if (not baseline_path.is_file() or baseline_path.stat().st_size > 64 * 1024 * 1024
            or str(baseline_path.resolve()) != baseline['path']
            or sha256_file(baseline_path) != baseline.get('sha256')):
        raise ValueError('frozen update-zero baseline checkpoint bytes changed')
    info = report.get('checkpoint', {})
    if info.get('sha256') != freeze.get('checkpoint_sha256'):
        raise ValueError('frozen checkpoint/report mismatch')
    if str(Path(checkpoint).resolve()) != freeze.get('checkpoint_path'):
        raise ValueError('checkpoint path differs from the immutable frozen development-best checkpoint')
    if not Path(checkpoint).is_file() or Path(checkpoint).stat().st_size > 64 * 1024 * 1024:
        raise ValueError('recovery checkpoint missing or exceeds 64 MiB bound')
    if sha256_file(checkpoint) != freeze.get('checkpoint_sha256'):
        raise ValueError('frozen checkpoint bytes changed')
    return identity
