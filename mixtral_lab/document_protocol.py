"""Versioned full-document execution and development-selected experiment contracts.

Legacy singleton bridges are historical diagnostics, never evidence for this
new execution path. Integrity hashes detect changes; they are not authentication.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path

from .fit_protocol import finite_tree, experiment_contract, seal_report
from .protocol import canonical_hash, integer, sha256_file, validate_config
from .recovery_protocol import ROLES, PROTOCOL, read_documents, prepare_data, validate_paired_metrics

SCHEMA = 'mixtral-full-document-recovery-v1'
REPORT_SCHEMA = 'mixtral-document-report-v1'
FREEZE_SCHEMA = 'mixtral-document-freeze-v1'
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 160 * 1024 * 1024


def trainer_api(kind):
    if kind == 'adapter':
        from .document_training import DocumentAdapterConfig, train_document_adapter, load_document_adapter_checkpoint
        return DocumentAdapterConfig, train_document_adapter, load_document_adapter_checkpoint
    if kind == 'router':
        from .document_router import DocumentRouterConfig, train_document_router, load_document_router_checkpoint
        return DocumentRouterConfig, train_document_router, load_document_router_checkpoint
    raise ValueError('document.kind must be adapter or router')


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite positive')


def validate_document_config(raw):
    finite_tree(raw, 'config')
    config = deepcopy(raw)
    if not isinstance(config, dict) or config.get('schema') != SCHEMA:
        raise ValueError('requires the full-document experiment schema; legacy artifacts cannot be relabeled')
    if any(k in config for k in ('fit', 'training', 'recovery', 'second_round', 'oracle', 'context_sampling', 'max_train_contexts')):
        raise ValueError('legacy singleton sampling/training fields are forbidden in document protocol')
    for key in ('data_files', 'sequences'):
        if not isinstance(config.get(key), dict) or set(config[key]) != set(ROLES):
            raise ValueError('exactly train/development/selection/calibration roles required; newtest forbidden')
    base = deepcopy(config)
    base.update(schema='mixtral-frozen-routing-v1', max_train_contexts=1,
                training={'batch_contexts': 1, 'max_train_contexts': 1, 'seed': config.get('seed')})
    for key in ('data_files', 'sequences'):
        base[key] = {r: config[key][r] for r in ('train', 'selection', 'calibration')}
    validate_config(base, stage='train')
    paths = []
    for role in ROLES:
        integer(config['sequences'][role], 'sequences.' + role, 1, config['max_documents_per_split'])
        path = config['data_files'][role]
        if not isinstance(path, str) or not path.strip() or 'newtest' in path.casefold():
            raise ValueError('invalid document data path or forbidden newtest')
        paths.append(str(Path(path).resolve()))
    if len(set(paths)) != len(ROLES):
        raise ValueError('all four roles require distinct source files')
    settings = config.get('document')
    if not isinstance(settings, dict) or set(settings) != {'kind', 'training', 'development_min_gain', 'selection_min_gain'}:
        raise ValueError('document settings require kind, training, development_min_gain, selection_min_gain')
    for name in ('development_min_gain', 'selection_min_gain'):
        positive(settings[name], name)
    if not isinstance(settings['training'], dict):
        raise ValueError('document.training must be a dictionary')
    cls, _, _ = trainer_api(settings['kind'])
    training = cls(**settings['training'])
    training.validate()
    if training.seed != config['seed']:
        raise ValueError('training and runner seeds must match')
    settings['training'] = asdict(training)
    return config


def training_population(config, data_identity):
    return {'schema': 'mixtral-document-population-v1', 'role': 'train',
            'document_ids': data_identity['train']['selected_document_ids'],
            'tokenized_sha256': data_identity['train']['tokenized_sha256'],
            'supervised_tokens': data_identity['train']['supervised_tokens'],
            'sequence_length': config['sequence_length'], 'batch_documents': 1,
            'mask_policy': 'deterministic rotating train-only positions; full padded tail before loss masking',
            'training_config_sha256': canonical_hash(config['document']['training'])}


def validate_replay_evidence(evidence, config, expected_ids):
    """Recompute pass from primary discrepancy counts, not a trusted boolean.

    Nonzero relative tolerances are frozen in config. The numerical core records
    the elementwise |x-y| > atol + rtol*|reference| count for each comparison.
    """
    finite_tree(evidence, 'document replay')
    if (not isinstance(evidence, dict) or evidence.get('status') != 'complete'
            or evidence.get('protocol') != 'full_padded_document_tail_v1'
            or evidence.get('kind') != config['document']['kind']
            or evidence.get('old_singleton_gate_used') is not False
            or evidence.get('probe') != 'deterministic_nonzero_adapter_or_shifted_native_pair'):
        raise ValueError('missing complete full-document replay evidence')
    training = config['document']['training']
    if evidence.get('tolerances') != {'atol': training['parity_atol'], 'rtol': training['parity_rtol']}:
        raise ValueError('document replay tolerances differ from scientific contract')
    if (not isinstance(expected_ids, list) or not expected_ids or len(set(expected_ids)) != len(expected_ids)):
        raise ValueError('replay requires nonempty unique document identifiers')
    records = evidence.get('documents')
    if not isinstance(records, list) or [r.get('document_id') for r in records] != expected_ids:
        raise ValueError('document replay population mismatch')
    passed = True
    required = {f'{path}.{metric}' for path in ('native_vs_controlled', 'zero_vs_native', 'cached_vs_full', 'probe_cached_vs_full')
                for metric in ('hidden', 'logits', 'ce')}
    required.add('cached_vs_full.h0')
    for row in records:
        probe = row.get('nonzero_probe')
        if not isinstance(probe, dict):
            raise ValueError('nonzero replay probe evidence missing')
        changed = probe.get('effective_changed_elements')
        integer(changed, 'nonzero probe changed elements', 0)
        if config['document']['kind'] == 'router':
            integer(probe.get('action_change_count'), 'nonzero router probe changed actions', 0)
            probe_passed = probe['action_change_count'] > 0
            if probe.get('state_sha256') is not None:
                raise ValueError('router deterministic probe must not claim adapter state')
        else:
            digest = probe.get('state_sha256')
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest) or probe.get('action_change_count') is not None):
                raise ValueError('nonzero adapter probe state binding missing')
            probe_passed = changed > 0
        if probe.get('passed') is not probe_passed:
            raise ValueError('nonzero probe verdict differs from primary effect evidence')
        passed = passed and probe_passed
        comparisons = row.get('comparisons')
        if not isinstance(comparisons, dict) or not required.issubset(comparisons):
            raise ValueError('document replay required numerical comparisons missing')
        for name, metric in comparisons.items():
            n, failed = metric.get('elements'), metric.get('mismatched_elements')
            integer(n, 'replay comparison elements', 1)
            integer(failed, 'replay mismatched elements', 0, n)
            for field in ('max_abs', 'max_excess'):
                value = metric.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                    raise ValueError('invalid primary replay numerical metric')
            if metric['max_excess'] > metric['max_abs']:
                raise ValueError('replay excess cannot exceed absolute discrepancy')
            if training['parity_rtol'] == 0 and not math.isclose(metric['max_excess'],
                    max(metric['max_abs']-training['parity_atol'], 0.), rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError('replay absolute discrepancy disagrees with declared tolerance')
            if (failed == 0) != (metric['max_excess'] == 0):
                raise ValueError('replay discrepancy count and tolerance excess disagree')
            passed = passed and failed == 0
    if evidence.get('passed') is not passed:
        raise ValueError('document replay aggregate verdict disagrees with primary comparisons')
    return evidence


def require_replay_evidence(evidence, config, expected_ids):
    validate_replay_evidence(evidence, config, expected_ids)
    if evidence['passed'] is not True:
        raise ValueError('same-protocol full-document numerical screen failed')


def read_report(path, stage):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError('document report missing or exceeds 64 MiB')
    raw = path.read_bytes()
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError('document report exceeds bound')
    value = json.loads(raw)
    finite_tree(value, 'document report')
    if (not isinstance(value, dict) or value.get('schema') != REPORT_SCHEMA or value.get('stage') != stage
            or value.get('status') != 'complete' or value.get('model_load_started') is not True):
        raise ValueError('requires complete document-stage report, not legacy or failed evidence')
    config = validate_document_config(value.get('config'))
    if config != value['config'] or value.get('report_sha256') != seal_report(value)['report_sha256']:
        raise ValueError('document report digest or normalized configuration mismatch')
    if (value.get('config_sha256') != canonical_hash(config)
            or value.get('experiment_contract_sha256') != canonical_hash(experiment_contract(config))):
        raise ValueError('document report scientific configuration digest mismatch')
    ids = value.get('identity', {}).get('train', {}).get('selected_document_ids')
    if not ids:
        raise ValueError('missing train population')
    require_replay_evidence(value.get('replay_initial'), config, ids)
    require_replay_evidence(value.get('replay_frozen'), config, ids)
    validate_replay_bindings(value)
    return value


def validate_replay_bindings(report):
    for name, candidate in (('replay_initial', None), ('replay_frozen', report.get('checkpoint'))):
        replay = report.get(name, {})
        expected = {'identity_sha256': canonical_hash(report.get('identity', {})),
            'experiment_contract_sha256': canonical_hash(experiment_contract(report['config'])),
            'checkpoint': candidate, 'candidate_scope': 'initial zero' if candidate is None else 'frozen development-best'}
        if replay.get('binding') != expected or replay.get('candidate_checked') is not (candidate is not None):
            raise ValueError('document replay candidate/config/runtime identity binding mismatch')


def require_development_evidence(report):
    summary, gate = report.get('tune_summary', {}), report.get('development_gate', {})
    if summary.get('status') not in ('complete', 'completed'):
        raise ValueError('document training did not complete')
    update = summary.get('best_update')
    integer(update, 'best development update', 0, 10000)
    if update != report.get('checkpoint', {}).get('update'):
        raise ValueError('development-best checkpoint update mismatch')
    first, best = summary.get('initial_development_nll'), summary.get('best_development_nll')
    for n in (first, best):
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0:
            raise ValueError('invalid development NLL')
    threshold = report['config']['document']['development_min_gain']
    gain = first - best
    if (update == 0 or gain < threshold or gate.get('passed') is not True
            or gate.get('threshold_nats') != threshold
            or not math.isclose(gate.get('gain_nats', math.inf), gain, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError('positive development gain required; update-zero fallback blocks outer evaluation/profile')
    require_replay_evidence(report.get('replay_frozen'), report['config'], report['identity']['train']['selected_document_ids'])


def require_quality_evidence(report):
    config = report['config']
    result = report.get('evaluation', {})
    validate_paired_metrics(result, 'selection', report['identity']['selection']['selected_document_ids'], config['nll_tolerance'])
    gate = result.get('quality_gate', {})
    threshold = config['document']['selection_min_gain']
    if gate.get('passed') is not True or gate.get('threshold_nats') != threshold:
        raise ValueError('outer quality gate not passed')
    for name, key in (('native', 'gain_vs_native_nats'), ('controlled_original', 'gain_vs_controlled_nats')):
        measured = result[name]['nll'] - result['nll']
        if measured < threshold or not math.isclose(gate.get(key, math.inf), measured, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError('outer quality gain disagrees with primary evidence')
    require_replay_evidence(report.get('replay_frozen'), config, report['identity']['train']['selected_document_ids'])


def _check_artifact(info, name):
    if not isinstance(info, dict) or not isinstance(info.get('path'), str):
        raise ValueError(f'missing {name} checkpoint')
    path = Path(info['path'])
    if (not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES or str(path.resolve()) != info['path']
            or sha256_file(path) != info.get('sha256')):
        raise ValueError(f'{name} checkpoint bytes/path changed')


def verify_frozen_report(report, config, checkpoint, current):
    if experiment_contract(report['config']) != experiment_contract(config):
        raise ValueError('frozen scientific configuration mismatch; only output_dir may change')
    freeze = report.get('freeze_manifest', {})
    if freeze.get('schema') != FREEZE_SCHEMA or report.get('freeze_manifest_sha256') != canonical_hash(freeze):
        raise ValueError('document freeze manifest missing or digest mismatch')
    identity = report.get('identity', {})
    if freeze.get('identity_sha256') != canonical_hash(identity):
        raise ValueError('frozen identity digest mismatch')
    if freeze.get('experiment_contract_sha256') != canonical_hash(experiment_contract(config)):
        raise ValueError('frozen contract digest mismatch')
    for key in ('code', 'local_inputs', 'artifact_stats', 'data', 'training_population', 'runtime_versions'):
        if freeze.get(key) != current.get(key):
            raise ValueError(f'frozen {key} changed before weight loading')
    if identity.get('code') != current['code'] or identity.get('training_population') != current['training_population']:
        raise ValueError('frozen source/training population mismatch')
    for role in ROLES:
        if identity.get(role) != current['data'][role]:
            raise ValueError('frozen data role identity mismatch')
    for name in ('baseline_checkpoint', 'checkpoint'):
        if freeze.get(name) != report.get(name):
            raise ValueError('checkpoint freeze binding mismatch')
        _check_artifact(report[name], name)
    if report['baseline_checkpoint'].get('update') != 0:
        raise ValueError('immutable initial checkpoint required')
    if str(Path(checkpoint).resolve()) != report['checkpoint']['path']:
        raise ValueError('only immutable frozen development-best checkpoint permitted')
    return identity
