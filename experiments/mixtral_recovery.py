"""Offline development-best recovery: tune, frozen outer evaluate, then profile.

Imports never launch GPU jobs. All stages require a fresh destination. The four
roles are disjoint; historically inspected outer selection is not final heldout.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from experiments.mixtral_experiment import (
    atomic_json, code_identity, evaluate, evaluate_second_round, input_device,
    load_local_inputs, load_local_model, measure_calls, precision_scope,
    route_diagnostics, validate_native,
)
from experiments.mixtral_fit_experiment import normalized_artifact_stats, profile_functions, runtime_versions
from mixtral_lab.recovery_protocol import (
    FREEZE_SCHEMA, PROTOCOL, REPORT_SCHEMA, ROLES, experiment_contract, finite_tree,
    prepare_data, read_documents, read_report, require_bridge_evidence,
    require_development_evidence, require_quality_evidence, seal_report, trainer_api,
    validate_paired_metrics, validate_recovery_config, verify_frozen_report, validate_bridge_evidence,
)
from mixtral_lab.protocol import canonical_hash, fresh_output, sha256_file
from mixtral_lab.sampling import collect_stratified_contexts, plan_stratified_contexts


def parser():
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument('--config', required=True)
    result.add_argument('--stage', required=True, choices=('tune', 'evaluate', 'profile'))
    result.add_argument('--tune-report', help='completed recovery tuning report; required after tune')
    result.add_argument('--checkpoint', help='immutable development-best checkpoint named by tune report')
    result.add_argument('--evaluation-report', help='positive frozen outer-selection report; required for profile')
    return result


def recovery_code_identity():
    identity = code_identity()
    root = Path(__file__).resolve().parents[1]
    # code_identity already includes every mixtral_lab source. Include helpers
    # actually reused by this runner, not just the top-level entry point.
    for name in ('experiments/mixtral_recovery.py', 'experiments/mixtral_fit_experiment.py'):
        identity['sources'][name] = sha256_file(root/name)
    return identity


def build_identity(config, model_info, current):
    return {'model': model_info['model'], 'tokenizer': model_info['tokenizer'],
        **current['data'], 'code': current['code'], 'context_sampling': current['train_contexts'],
        'recovery_contract': experiment_contract(config),
        'numerical_contract': {**{key: config[key] for key in ('dtype', 'allow_tf32', 'experts_implementation',
            'sequence_length', 'sequences', 'max_train_contexts')}, 'attention': 'sdpa',
            'runtime_versions': current['runtime_versions'], 'actual_device_map': model_info.get('device_map'),
            'cuda_devices': model_info.get('cuda_devices'),
            'checkpoint_selection': 'development full-document online token-weighted NLL; update 0 participates',
            'selection_scope': 'one frozen development-best candidate; historically inspected outer selection is not final heldout',
            'cross_shape_equivalence': 'measured explicitly by matched-position cache/online bridge; never assumed'}}


def validate_stage_arguments(args):
    if args.stage == 'tune' and (args.tune_report or args.checkpoint or args.evaluation_report):
        raise ValueError('tune starts fresh; report/checkpoint inputs and resume are forbidden')
    if args.stage != 'tune' and not (args.tune_report and args.checkpoint):
        raise ValueError('evaluate/profile require --tune-report and --checkpoint')
    if args.stage == 'evaluate' and args.evaluation_report:
        raise ValueError('evaluate may not select candidates or reuse outer evaluation reports')
    if args.stage == 'profile' and not args.evaluation_report:
        raise ValueError('profile requires --evaluation-report; no quality-gate override exists')


def check_checkpoint_metadata(metadata, config, expected_sha, summary=None):
    finite_tree(metadata, 'checkpoint metadata')
    if metadata.get('checkpoint_sha256') != expected_sha:
        raise ValueError('checkpoint changed between preflight hashing and deserialization')
    if metadata.get('training_config', metadata.get('config')) != config['recovery']['training']:
        raise ValueError('checkpoint training configuration mismatch')
    if summary is not None:
        if metadata.get('update') != summary.get('best_update'):
            raise ValueError('checkpoint is not the frozen development-best update')
        development = metadata.get('development_metrics', metadata.get('development'))
        if (not isinstance(development, dict) or development.get('nll') != summary.get('best_development_nll')
                or development.get('role') != 'development' or development.get('protocol') != PROTOCOL):
            raise ValueError('checkpoint development-best metrics mismatch')
        expected = summary.get('best_development_metrics')
        if expected is not None and development != expected:
            raise ValueError('checkpoint development metrics differ from tuning summary')


def downstream_preflight(args, config, current):
    tune = read_report(args.tune_report, 'tune')
    require_development_evidence(tune)
    identities = verify_frozen_report(tune, config, args.checkpoint, current)
    _, _, loader = trainer_api(config['recovery']['kind'])
    _, initial = loader(tune['baseline_checkpoint']['path'], identities, 'cpu')
    check_checkpoint_metadata(initial, config, tune['baseline_checkpoint']['sha256'])
    initial_metrics = initial.get('development_metrics', initial.get('development'))
    validate_paired_metrics(initial_metrics, 'development', identities['development']['selected_document_ids'], config['nll_tolerance'])
    if initial.get('update') != 0 or initial_metrics['nll'] != tune['tune_summary']['initial_development_nll']:
        raise ValueError('initial development NLL differs from immutable update-zero checkpoint')
    _, metadata = loader(args.checkpoint, identities, 'cpu')
    check_checkpoint_metadata(metadata, config, tune['checkpoint']['sha256'], tune['tune_summary'])
    validate_paired_metrics(metadata.get('development_metrics', metadata.get('development')),
        'development', identities['development']['selected_document_ids'], config['nll_tolerance'])
    evidence = {'tune_report_path': str(Path(args.tune_report).resolve()), 'tune_report_sha256': tune['report_sha256']}
    evaluation = None
    if args.stage == 'profile':
        evaluation = read_report(args.evaluation_report, 'evaluate')
        if evaluation.get('evidence', {}).get('tune_report_sha256') != tune['report_sha256']:
            raise ValueError('outer evaluation refers to a different tune report')
        verify_frozen_report(evaluation, config, args.checkpoint, current)
        if (evaluation.get('identity') != identities or evaluation.get('checkpoint') != tune.get('checkpoint')
                or evaluation.get('freeze_manifest') != tune.get('freeze_manifest')):
            raise ValueError('outer evaluation and tune identities/checkpoints differ')
        if evaluation.get('bridge_frozen_candidate') != tune.get('bridge_frozen_candidate'):
            raise ValueError('outer evaluation changed frozen bridge evidence')
        require_quality_evidence(evaluation, config['recovery']['selection_min_gain'])
        evidence.update(evaluation_report_path=str(Path(args.evaluation_report).resolve()),
                        evaluation_report_sha256=evaluation['report_sha256'])
    return tune, evaluation, evidence


def aggregate_documents(records, name):
    tokens = sum(row['tokens'] for row in records)
    if not records or tokens <= 0:
        raise ValueError('empty full-document evaluation')
    return {'nll': sum(row[name + '_nll'] * row['tokens'] for row in records) / tokens,
            'tokens': tokens, 'sequences': len(records)}


def document_statistics(records):
    """Descriptive paired document statistics, never token-iid confidence bounds."""
    result = {'unit': 'document', 'confidence_interval': None,
              'interpretation': 'descriptive only; documents, not tokens, are the independent evaluation units'}
    for baseline in ('native', 'controlled'):
        deltas = [r['candidate_minus_' + baseline + '_nll'] for r in records]
        mean = sum(deltas) / len(deltas)
        benefits = sorted([max(-d, 0.) * r['tokens'] for d, r in zip(deltas, records)], reverse=True)
        total = sum(benefits)
        result[baseline] = {'document_mean_delta_nll': mean,
            'document_population_std_delta_nll': math.sqrt(sum((d - mean) ** 2 for d in deltas) / len(deltas)),
            'minimum_delta_nll': min(deltas), 'maximum_delta_nll': max(deltas),
            'improved_documents': sum(d < 0 for d in deltas), 'degraded_documents': sum(d > 0 for d in deltas),
            'unchanged_documents': sum(d == 0 for d in deltas),
            'positive_token_gain_total_nats': total,
            'top_document_positive_gain_fraction': benefits[0] / total if total else 0.,
            'top_two_documents_positive_gain_fraction': sum(benefits[:2]) / total if total else 0.}
    return result


def evaluate_documents(adapter, rows, candidate, config, *, role, work=None):
    if role not in ('development', 'selection'):
        raise ValueError('candidate evaluator accepts development or outer selection, never train/calibration/newtest')
    records = []
    counter = 'development' if role == 'development' else 'outer_selection'
    def measured(path, operation):
        if work is not None:
            key = counter + '_document_forwards'
            work[key] = work.get(key, 0) + 1
            by_path = work.setdefault(counter + '_attempted_forwards_by_path', {})
            by_path[path] = by_path.get(path, 0) + 1
        result = operation()
        if work is not None:
            key = counter + '_completed_document_forwards'
            work[key] = work.get(key, 0) + 1
        return result
    for row in rows:
        native = measured('native', lambda: evaluate(adapter, [row], native=True))
        control = measured('controlled', lambda: evaluate(adapter, [row]))
        learned = measured('candidate', lambda: (
            evaluate(adapter, [row], candidate) if config['recovery']['kind'] == 'router'
            else evaluate_second_round(adapter, [row], candidate)))
        finite_tree([native, control, learned], role + ' metrics')
        for metric in (native, control, learned):
            nll = metric.get('nll')
            if (isinstance(nll, bool) or not isinstance(nll, (int, float)) or nll < 0
                    or type(metric.get('tokens')) is not int or metric['tokens'] <= 0 or metric.get('sequences') != 1):
                raise ValueError('invalid full-document metric')
        if len({metric['tokens'] for metric in (native, control, learned)}) != 1:
            raise ValueError('full-document metric populations differ')
        records.append({'document_id': row['id'], 'tokens': native['tokens'],
            'native_nll': native['nll'], 'controlled_nll': control['nll'], 'candidate_nll': learned['nll'],
            'candidate_minus_native_nll': learned['nll'] - native['nll'],
            'candidate_minus_controlled_nll': learned['nll'] - control['nll']})
    result = {'status': 'ok', 'role': role, 'protocol': PROTOCOL, 'document_ids': [r['id'] for r in rows],
        'per_document': records, 'native': aggregate_documents(records, 'native'),
        'controlled_original': aggregate_documents(records, 'controlled'), 'candidate': aggregate_documents(records, 'candidate'),
        'document_statistics': document_statistics(records),
        'numerical_scope': 'identical full padded document inputs and supervised positions for native/control/candidate',
        'selection_policy': ('checkpoint selection and early stopping' if role == 'development'
                             else 'one frozen development-best candidate; no fitting, early stopping, model selection, or parameter changes'),
        'inference_limit': 'historically inspected outer selection is not final heldout; no multi-seed or deployment advantage claim'}
    result['nll'] = result['candidate']['nll']
    # A numerical screen failure is not merely an uncompetitive candidate; fail
    # the evaluation instead of allowing a questionable baseline into selection.
    validate_paired_metrics(result, role, result['document_ids'], config['nll_tolerance'])
    if role == 'selection':
        threshold = config['recovery']['selection_min_gain']
        gn = result['native']['nll'] - result['nll']
        gc = result['controlled_original']['nll'] - result['nll']
        result['quality_gate'] = {'threshold_nats': threshold, 'gain_vs_native_nats': gn,
            'gain_vs_controlled_nats': gc, 'controlled_native_nll_screen_passed': True,
            'passed': gn >= threshold and gc >= threshold,
            'threshold_source': 'recovery.selection_min_gain; independent of development checkpoint selection'}
    return finite_tree(result, role + ' evaluation')


def make_development_evaluator(adapter, rows, config, work):
    """Closure contains ONLY development rows, never the outer selection set."""
    contract = {'role': 'development', 'document_ids': [r['id'] for r in rows], 'protocol': PROTOCOL}
    training = config['recovery']['training']
    # Initial, scheduled/final, plus at most one off-schedule early-stop probe.
    maximum = 2 + math.ceil(training['updates'] / training['eval_interval'])
    work['development_evaluation_budget'] = maximum
    work['development_evaluations'] = 0
    work['development_document_forwards'] = 0
    work['development_document_forward_budget'] = maximum * len(rows) * 3
    def development(candidate):
        if work['development_evaluations'] >= maximum:
            raise ValueError('development evaluator exceeded preregistered update/interval budget')
        work['development_evaluations'] += 1
        result = evaluate_documents(adapter, rows, candidate, config, role='development', work=work)
        work['completed_development_evaluations'] = work.get('completed_development_evaluations', 0) + 1
        return result
    development.contract = contract
    return development


def bridge_config(config):
    view = deepcopy(config)
    view['fit'] = {'kind': config['recovery']['kind'], 'training': config['recovery']['training']}
    return view


def run_bridge(adapter, rows, candidate, config, manifest, metadata=None):
    from mixtral_lab.bridge import diagnose_bridge
    if metadata is not None and config['recovery']['kind'] == 'adapter':
        metadata = {**metadata, 'historical_cached_metrics': {'learned': metadata['train_probe']['nll']}}
    return finite_tree(diagnose_bridge(adapter, rows, candidate, bridge_config(config), manifest=manifest,
        checkpoint_metadata=metadata, tolerances=config['recovery']['bridge_tolerances']), 'cache/online bridge')


def zero_candidate(adapter, config):
    if config['recovery']['kind'] == 'router':
        from mixtral_lab.recovery_router import RecoveryPairRouter
        return RecoveryPairRouter(adapter.feature_dim, config['recovery']['training']['epsilon'], device=adapter.device).eval()
    from mixtral_lab.second_round import SecondRoundResidual
    config_class, _, _ = trainer_api('adapter')
    return SecondRoundResidual(adapter.hidden_size, config_class(**config['recovery']['training']), adapter.device).eval()


def assert_inputs_unchanged(config, current):
    if recovery_code_identity() != current['code'] or normalized_artifact_stats(config) != current['artifact_stats']:
        raise ValueError('code or immutable local artifacts changed during recovery stage')
    for role, info in current['data'].items():
        if sha256_file(config['data_files'][role]) != info['file_sha256']:
            raise ValueError('recovery data source changed during stage')


def main(argv=None):
    args = parser().parse_args(argv)
    raw = json.loads(Path(args.config).read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get('output_dir'), str) or not raw['output_dir'].strip():
        raise ValueError('config must contain a usable fresh output_dir for failure evidence')
    output = fresh_output(raw['output_dir'])
    report = {'schema': REPORT_SCHEMA, 'stage': args.stage, 'status': 'running', 'phase': 'preflight',
        'model_load_started': False, 'work_accounting': {'outer_selection_evaluations': 0, 'outer_selection_document_forwards': 0, 'forward_counter_scope': 'attempted per-document model-path forwards, with separate completed counters; bridge/trainer internal work retained in their own reports'},
        'limitations': ['four roles are disjoint; development alone selects checkpoints; fixed development patience and configured train plateau can stop training',
            'outer selection was historically inspected and is NOT final heldout',
            'newtest unsupported; no final confirmation, recurrent FFN, learned STOP, or deployment speed claim',
            'local artifacts must remain immutable; report hashes provide integrity, not authentication']}
    atomic_json(output/'report.json', report)
    started = time.perf_counter()
    try:
        validate_stage_arguments(args)
        config = validate_recovery_config(raw)
        report.update(config=config, config_sha256=canonical_hash(config),
                      experiment_contract_sha256=canonical_hash(experiment_contract(config)))
        read_documents(config)
        inputs = load_local_inputs(config)
        data, data_identity = prepare_data(config, inputs['tokenizer'])
        manifest = plan_stratified_contexts(data['train'], config['max_train_contexts'], config['seed'], 'train')
        atomic_json(output/'train_contexts.json', manifest)
        current = {'code': recovery_code_identity(), 'local_inputs': inputs['identity'],
            'artifact_stats': normalized_artifact_stats(config), 'data': data_identity,
            'train_contexts': manifest, 'runtime_versions': runtime_versions()}
        report['preflight'] = {'status': 'running', **current,
            'tensor_quotas': {'max_train_contexts': config['max_train_contexts'], 'sequence_length': config['sequence_length'],
                'padded_input_tokens_by_role': {role: len(data[role]) * config['sequence_length'] for role in ROLES},
                'supervised_tokens_by_role': {role: data_identity[role]['supervised_tokens'] for role in ROLES}},
            'weight_identity_scope': 'immutable path/inode/size/mtime/ctime snapshot before weights; actual bytes hashed once by loader'}
        tune = evaluation = None
        if args.stage != 'tune':
            tune, evaluation, report['evidence'] = downstream_preflight(args, config, current)
        report['preflight'].update(status='complete', wall_seconds=time.perf_counter()-started)
        atomic_json(output/'report.json', report)
        report['phase'] = 'model_load'
        if not torch.cuda.is_available():
            raise RuntimeError('real Mixtral recovery requires CUDA; tiny CPU integration is not a GPU experiment')
        torch.manual_seed(config['seed'])
        with precision_scope(config['allow_tf32']):
            report['model_load_started'] = True
            atomic_json(output/'report.json', report)
            load_started = time.perf_counter()
            model, tokenizer, adapter, model_info = load_local_model(config, local_inputs=inputs)
            report['model_load_wall_seconds'] = time.perf_counter() - load_started
            report['phase'] = 'execution'
            assert_inputs_unchanged(config, current)
            identity = build_identity(config, model_info, current)
            if tune is not None and identity != tune['identity']:
                raise ValueError('loaded model/tokenizer/runtime identity differs from frozen tune')
            report.update(identity=identity, model=model_info, environment={'runtime_versions': runtime_versions(),
                'cuda_build': torch.version.cuda,
                'cuda_visible_devices': current['runtime_versions'].get('cuda_visible_devices'),
                'device_index_scope': 'visible logical GPU indices; CUDA_VISIBLE_DEVICES records physical mapping when set',
                'gpus': {str(i): {'name': torch.cuda.get_device_name(i),
                    'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                    for i in model_info['cuda_devices']}})
            parity = finite_tree(validate_native(adapter, data['calibration'], config), 'native equivalence')
            report['native_equivalence'] = parity
            report['work_accounting']['calibration_documents'] = len(data['calibration'])
            atomic_json(output/'report.json', report)
            if parity.get('eligible') is not True:
                raise RuntimeError('native equivalence screen failed; recovery tune/evaluate/profile blocked')
            config_class, trainer, loader = trainer_api(config['recovery']['kind'])
            if args.stage == 'tune':
                report['phase'] = 'bridge_pretraining'
                zero = zero_candidate(adapter, config)
                report['bridge_pretraining'] = run_bridge(adapter, data['train'], zero, config, manifest)
                del zero
                atomic_json(output/'report.json', report)
                require_bridge_evidence(report['bridge_pretraining'], native_only=True, config=config, manifest=manifest)
                report['phase'] = 'tuning'
                h0, targets, captured = collect_stratified_contexts(adapter, data['train'], config['max_train_contexts'], config['seed'], 'train')
                if captured != manifest:
                    raise ValueError('captured train contexts differ from preflight manifest')
                development = make_development_evaluator(adapter, data['development'], config, report['work_accounting'])
                kwargs = {'provenance': [{**row, 'role': 'train'} for row in manifest['contexts']]}
                _, summary = trainer(adapter, h0, targets, development, config_class(**config['recovery']['training']),
                                     identity, output/'tune', **kwargs)
                report['tune_summary'] = finite_tree(summary, 'tune summary')
                atomic_json(output/'report.json', report)
                if summary.get('status') not in ('complete', 'completed'):
                    raise RuntimeError('tuning stopped on an execution/numerical guard; diagnostics retained')
                checkpoint = output/'tune'/'checkpoint_best.pt'
                candidate, metadata = loader(checkpoint, identity, adapter.device)
                checkpoint_sha = sha256_file(checkpoint)
                check_checkpoint_metadata(metadata, config, checkpoint_sha, summary)
                validate_paired_metrics(metadata.get('development_metrics', metadata.get('development')),
                    'development', identity['development']['selected_document_ids'], config['nll_tolerance'])
                initial_checkpoint = output/'tune'/'checkpoint_initial.pt'
                _, initial_metadata = loader(initial_checkpoint, identity, 'cpu')
                initial_sha = sha256_file(initial_checkpoint)
                check_checkpoint_metadata(initial_metadata, config, initial_sha)
                initial_metrics = initial_metadata.get('development_metrics', initial_metadata.get('development'))
                validate_paired_metrics(initial_metrics, 'development', identity['development']['selected_document_ids'], config['nll_tolerance'])
                if initial_metadata.get('update') != 0 or initial_metrics['nll'] != summary['initial_development_nll']:
                    raise ValueError('initial development metrics differ from update-zero checkpoint')
                report['baseline_checkpoint'] = {'path': str(initial_checkpoint.resolve()), 'sha256': initial_sha, 'update': 0}
                report['checkpoint'] = {'path': str(checkpoint.resolve()), 'sha256': checkpoint_sha, 'update': metadata['update'],
                    'selection_criterion': 'minimum full-document online development NLL, including update 0; never outer selection'}
                report['phase'] = 'bridge_frozen_candidate'
                report['bridge_frozen_candidate'] = run_bridge(adapter, data['train'], candidate, config, manifest, metadata)
                validate_bridge_evidence(report['bridge_frozen_candidate'], config=config, manifest=manifest, chosen_update=metadata['update'])
                report['training_path_numerical_screen_passed'] = report['bridge_frozen_candidate'].get('numerical_screen_passed') is True
                report['training_path_equivalence_established'] = False
                report['cached_train_fit_transfer_claim'] = 'not established; cached train metrics remain diagnostics, not outer quality evidence'
                atomic_json(output/'report.json', report)
                initial, best = summary['initial_development_nll'], summary['best_development_nll']
                gain = initial - best
                threshold = config['recovery']['development_min_gain']
                report['development_gate'] = {'threshold_nats': threshold, 'gain_nats': gain,
                    'passed': summary['best_update'] > 0 and gain >= threshold,
                    'fallback': 'update 0 retained when development does not improve; no outer selection or timing'}
                freeze = {'schema': FREEZE_SCHEMA, 'baseline_checkpoint': report['baseline_checkpoint'], 'checkpoint_path': str(checkpoint.resolve()),
                    'checkpoint_sha256': checkpoint_sha, 'identity_sha256': canonical_hash(identity),
                    'experiment_contract_sha256': canonical_hash(experiment_contract(config)), **current}
                report.update(freeze_manifest=freeze, freeze_manifest_sha256=canonical_hash(freeze))
                atomic_json(output/'freeze_manifest.json', freeze)
            else:
                candidate, metadata = loader(args.checkpoint, identity, adapter.device)
                check_checkpoint_metadata(metadata, config, tune['checkpoint']['sha256'], tune['tune_summary'])
                for key in ('checkpoint', 'baseline_checkpoint', 'freeze_manifest', 'freeze_manifest_sha256', 'bridge_pretraining', 'bridge_frozen_candidate', 'development_gate', 'training_path_numerical_screen_passed', 'training_path_equivalence_established', 'cached_train_fit_transfer_claim'):
                    report[key] = deepcopy(tune[key])
                atomic_json(output/'freeze_manifest.json', report['freeze_manifest'])
                if args.stage == 'evaluate':
                    report['phase'] = 'outer_selection'
                    report['work_accounting'].update(outer_selection_evaluations=1, outer_selection_document_forward_budget=3 * len(data['selection']))
                    report['evaluation'] = evaluate_documents(adapter, data['selection'], candidate, config, role='selection', work=report['work_accounting'])
                    report['work_accounting']['completed_outer_selection_evaluations'] = 1
                    if config['recovery']['kind'] == 'router':
                        report['evaluation']['route_diagnostics'] = route_diagnostics(adapter, data['selection'], candidate)
                        report['work_accounting']['selection_route_diagnostic_document_forwards'] = len(data['selection'])
                    else:
                        report['evaluation']['route_diagnostics'] = {'routing': 'native R1 plus residual capacity, not a second expert call'}
                else:
                    report['phase'] = 'profiling'
                    require_quality_evidence(evaluation, config['recovery']['selection_min_gain'])
                    report['quality_gate'] = evaluation['evaluation']['quality_gate']
                    report['work_accounting']['profile_forward_budget'] = 3 * (config['warmup'] + config['repeats'] * config['measurements'])
                    report['timing'] = finite_tree(measure_calls(profile_functions(adapter, candidate, config['recovery']['kind']),
                        data['selection'], model_info['cuda_devices'], config), 'timing')
                    report['timing']['output_device'] = str(input_device(model))
                    report['timing']['candidate_scope'] = ('native R1 plus residual capacity' if config['recovery']['kind'] == 'adapter'
                                                           else 'one learned R1 pair per token')
            assert_inputs_unchanged(config, current)
            if sha256_file(report['checkpoint']['path']) != report['checkpoint']['sha256']:
                raise ValueError('frozen checkpoint bytes changed during stage')
            report.update(status='complete', phase='complete', total_wall_seconds=time.perf_counter()-started)
            report = seal_report(report)
            atomic_json(output/'report.json', report)
    except Exception as exc:
        # A trainer may fail after retaining useful train metrics. Do not discard
        # those simply because development-best remains zero or a guard fires.
        summary_path = output/'tune'/'summary.json'
        if summary_path.is_file() and summary_path.stat().st_size <= 64 * 1024 * 1024 and 'tune_summary' not in report:
            try:
                report['tune_summary'] = finite_tree(json.loads(summary_path.read_text()), 'partial tune summary')
            except (ValueError, TypeError):
                report['tune_summary_artifact'] = str(summary_path)
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', failure_phase=report['phase'], total_wall_seconds=time.perf_counter()-started)
        report.pop('report_sha256', None)
        if report['phase'] == 'preflight':
            report.setdefault('preflight', {}).update(status='failed')
        atomic_json(output/'report.json', report)
        raise
    print(json.dumps({'report': str(output/'report.json'), 'status': report['status']}))
    return report


if __name__ == '__main__':
    main()
