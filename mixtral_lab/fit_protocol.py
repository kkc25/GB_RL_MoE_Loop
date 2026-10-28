"""Fail-closed contracts for train-fit -> development generalization -> cost.

This is deliberately separate from the historical selection-best protocol.
Reports are integrity checked, not authenticated. Local model snapshots must be
immutable: inode/size/time checks reject ordinary changes before a 93 GB load;
the legacy loader independently hashes actual bytes once per invocation.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

from mixtral_lab.protocol import canonical_hash, integer, sha256_file, validate_config

REPORT_SCHEMA = 'mixtral-fit-first-report-v1'
FREEZE_SCHEMA = 'mixtral-train-fit-freeze-v1'
# A legal 10,000-probe adaptive fit can exceed 16 MiB once its audit
# histories, identities and freeze manifest are retained. Still bound input.
MAX_REPORT_BYTES = 64 * 1024 * 1024


def finite_tree(value, name='value'):
    """Reject nonfinite numeric evidence before it can enter a report or gate."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'{name} contains nonfinite evidence')
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f'{name}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tree(item, f'{name}[{index}]')
    return value


def trainer_api(kind):
    if kind == 'router':
        from mixtral_lab.full_action_fit import FullActionFitConfig, train_full_action_router, load_full_action_checkpoint
        return FullActionFitConfig, train_full_action_router, load_full_action_checkpoint
    if kind == 'adapter':
        from mixtral_lab.adapter_fit import AdapterFitConfig, train_adapter_fit, load_adapter_fit_checkpoint
        return AdapterFitConfig, train_adapter_fit, load_adapter_fit_checkpoint
    raise ValueError('fit.kind must be router or adapter')


def validate_fit_config(config):
    """Retain *all* legacy global input/budget bounds without altering its code."""
    validate_config(config, stage='train')
    finite_tree(config, 'config')
    fit = config.get('fit')
    if not isinstance(fit, dict) or set(fit) != {'kind', 'training', 'generalization_min_gain'}:
        raise ValueError('fit must specify exactly kind, training, generalization_min_gain')
    threshold = fit['generalization_min_gain']
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold <= 0:
        raise ValueError('generalization_min_gain must be finite positive; independent of oracle.practical_gain')
    if not isinstance(fit['training'], dict):
        raise ValueError('fit.training must be an object')
    config_class, _, _ = trainer_api(fit['kind'])
    training = config_class(**fit['training'])
    training.validate()
    integer(training.updates, 'fit.updates', 1, 10000)
    integer(training.batch_contexts, 'fit.batch_contexts', 1, 1024)
    if fit['kind'] == 'router':
        integer(training.probe_contexts, 'fit.probe_contexts', 1, 4096)
    if training.max_train_contexts != config['max_train_contexts'] or training.seed != config['seed']:
        raise ValueError('fit training and runner max_train_contexts/seed must match')
    if training.batch_contexts > training.max_train_contexts:
        raise ValueError('fit batch_contexts exceeds exact training context budget')
    if fit['kind'] == 'router' and training.max_score_calls < 30 * config['max_train_contexts']:
        raise ValueError('fit max_score_calls must cover 30 singleton score calls per train context')
    if fit['kind'] == 'router':
        integer(training.max_score_calls, 'fit.max_score_calls', 1, 30 * 4096)
    config['fit']['training'] = asdict(training)
    # Omitted and explicit default-off policies are the same scientific contract.
    from mixtral_lab.adaptive_fit import AdaptiveFitConfig
    config['fit']['training']['adaptive'] = asdict(AdaptiveFitConfig.from_dict(training.adaptive))
    return config


def experiment_contract(config):
    """Only the fresh destination may change between stages."""
    return {key: value for key, value in config.items() if key != 'output_dir'}


def seal_report(report):
    finite_tree(report, 'report')
    result = {key: value for key, value in report.items() if key != 'report_sha256'}
    result['report_sha256'] = canonical_hash(result)
    return result


def read_report(path, stage):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError('stage report missing or exceeds 64 MiB bound')
    snapshot = path.read_bytes()
    if len(snapshot) > MAX_REPORT_BYTES:
        raise ValueError('stage report exceeds 64 MiB bound')
    report = json.loads(snapshot)
    finite_tree(report, 'report')
    if not isinstance(report, dict):
        raise ValueError('stage report must be an object')
    if (report.get('schema') != REPORT_SCHEMA or report.get('stage') != stage
            or report.get('status') != 'complete' or report.get('model_load_started') is not True):
        raise ValueError(f'requires complete {stage} report, not failed/running/legacy evidence')
    if report.get('report_sha256') != seal_report(report)['report_sha256']:
        raise ValueError('stage report digest mismatch')
    if report.get('native_equivalence', {}).get('eligible') is not True:
        raise ValueError('stage report lacks passed native equivalence')
    return report


def require_fit_evidence(report):
    summary = report.get('fit_summary', {})
    if summary.get('status') not in ('complete', 'completed') or summary.get('train_fit_passed') is not True:
        raise ValueError('train-fit gate not passed; generalization/profile are blocked')
    integer(summary.get('best_update'), 'best train-fit update', 1, 10000)
    if report.get('checkpoint', {}).get('update') != summary['best_update']:
        raise ValueError('fit best update and frozen checkpoint update mismatch')


def verify_checkpoint_fit_metrics(metadata, report):
    """The gate flag must agree with the frozen checkpoint's measured train data."""
    summary = report['fit_summary']
    threshold = report['config']['fit']['training']['fit_gain_threshold']
    if report['config']['fit']['kind'] == 'router':
        metrics = summary.get('best_train_metrics')
        if (metadata.get('train_fit_passed') is not True or not isinstance(metrics, dict)
                or metadata.get('train_metrics') != metrics):
            raise ValueError('frozen router train-fit metrics/verdict mismatch')
        gain = metrics.get('greedy_gain_nats')
        changes = metrics.get('greedy_change_count')
        if (isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain)
                or gain < threshold or type(changes) is not int or changes <= 0):
            raise ValueError('frozen router does not meet the train-fit threshold')
    else:
        best = summary.get('best_train_probe_nll')
        initial = summary.get('initial_train_probe_nll')
        if (metadata.get('train_probe_nll') != best or summary.get('coverage_complete') is not True
                or isinstance(initial, bool) or not isinstance(initial, (int, float))
                or isinstance(best, bool) or not isinstance(best, (int, float))
                or not math.isfinite(initial) or not math.isfinite(best) or initial - best < threshold):
            raise ValueError('frozen adapter does not meet the train-fit threshold/coverage')


def require_quality_evidence(report, threshold):
    quality = report.get('generalization', {})
    gate = quality.get('quality_gate', {})
    finite_tree(quality, 'generalization')
    if gate.get('threshold_nats') != threshold:
        raise ValueError('generalization threshold differs from the frozen contract')
    for key in ('gain_vs_native_nats', 'gain_vs_controlled_nats'):
        value = gate.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < threshold:
            raise ValueError('generalization quality gate not passed; timing is blocked')
    if (gate.get('passed') is not True or gate.get('controlled_native_nll_screen_passed') is not True
            or quality.get('role') != 'selection_development'):
        raise ValueError('requires passing development generalization evidence, not held-out claims')
    # Recompute the gate from primary aggregate metrics rather than a lone flag.
    try:
        native, controlled, candidate = (quality[key] for key in ('native', 'controlled_original', 'candidate'))
        for metric in (native, controlled, candidate):
            integer(metric.get('tokens'), 'development tokens', 1)
            integer(metric.get('sequences'), 'development sequences', 1)
            nll = metric.get('nll')
            if isinstance(nll, bool) or not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0:
                raise ValueError('invalid development NLL')
        for key, baseline in (('gain_vs_native_nats', native), ('gain_vs_controlled_nats', controlled)):
            if baseline['tokens'] != candidate['tokens'] or baseline['sequences'] != candidate['sequences']:
                raise ValueError('generalization populations differ')
            if not math.isclose(baseline['nll'] - candidate['nll'], gate[key], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('generalization gate metrics disagree')
        rows = quality.get('per_document')
        if not isinstance(rows, list) or len(rows) != candidate['sequences']:
            raise ValueError('generalization per-document evidence missing')
        expected_ids = report['identity']['selection']['selected_document_ids']
        if [row['document_id'] for row in rows] != expected_ids or len(set(expected_ids)) != len(expected_ids):
            raise ValueError('generalization document population mismatch')
        for row in rows:
            integer(row.get('tokens'), 'per-document tokens', 1)
            if abs(row['native_nll'] - row['controlled_nll']) > report['config']['nll_tolerance']:
                raise ValueError('controlled/native per-document screen failed')
            for baseline in ('native', 'controlled'):
                if not math.isclose(row['candidate_nll'] - row[baseline + '_nll'],
                        row['candidate_minus_' + baseline + '_nll'], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError('per-document paired delta mismatch')
        if sum(row['tokens'] for row in rows) != candidate['tokens']:
            raise ValueError('generalization per-document token total mismatch')
        for name, metric in (('native', native), ('controlled', controlled), ('candidate', candidate)):
            measured = sum(row[name + '_nll'] * row['tokens'] for row in rows) / candidate['tokens']
            if not math.isclose(measured, metric['nll'], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('generalization per-document and aggregate metrics disagree')
    except (KeyError, TypeError) as exc:
        raise ValueError('malformed generalization primary metrics') from exc


def verify_frozen_report(report, config, checkpoint, current):
    """Cheap preflight check, before a real model is allocated or deserialized."""
    if experiment_contract(report.get('config', {})) != experiment_contract(config):
        raise ValueError('frozen scientific configuration mismatch (only output_dir may change)')
    freeze = report.get('freeze_manifest')
    if not isinstance(freeze, dict) or freeze.get('schema') != FREEZE_SCHEMA:
        raise ValueError('missing or malformed checkpoint freeze manifest')
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
    for role in ('train', 'selection', 'calibration'):
        if identity.get(role) != current['data'][role]:
            raise ValueError('frozen data identity mismatch')
    if identity.get('context_sampling') != current['train_contexts']:
        raise ValueError('frozen train sampling identity mismatch')
    info = report.get('checkpoint', {})
    if info.get('sha256') != freeze.get('checkpoint_sha256'):
        raise ValueError('frozen checkpoint/report mismatch')
    if str(Path(checkpoint).resolve()) != freeze.get('checkpoint_path'):
        raise ValueError('checkpoint path differs from the immutable frozen best checkpoint')
    if sha256_file(checkpoint) != freeze.get('checkpoint_sha256'):
        raise ValueError('frozen checkpoint bytes changed')
    return identity
