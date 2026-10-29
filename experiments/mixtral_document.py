"""Full-document tail training, frozen development-best evaluation, then profiling.

This entrypoint never reuses singleton scoring or historical bridge gates. GPU
execution is explicit; importing it starts no model loading or experiments.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from experiments.mixtral_experiment import (atomic_json, code_identity, input_device,
    load_local_inputs, load_local_model, measure_calls, precision_scope)
from experiments.mixtral_fit_experiment import normalized_artifact_stats, runtime_versions
from experiments.mixtral_recovery import document_statistics
from mixtral_lab.document_protocol import (FREEZE_SCHEMA, PROTOCOL, REPORT_SCHEMA, ROLES,
    experiment_contract, finite_tree, prepare_data, read_documents, read_report,
    require_development_evidence, require_quality_evidence, require_replay_evidence,
    seal_report, trainer_api, training_population, validate_document_config,
    validate_paired_metrics, verify_frozen_report)
from mixtral_lab.protocol import canonical_hash, fresh_output, sha256_file


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config', required=True)
    p.add_argument('--stage', required=True, choices=('tune', 'evaluate', 'profile'))
    p.add_argument('--tune-report')
    p.add_argument('--checkpoint')
    p.add_argument('--evaluation-report')
    return p


def validate_stage_arguments(args):
    if args.stage == 'tune' and (args.tune_report or args.checkpoint or args.evaluation_report):
        raise ValueError('tune starts fresh; no resume or historical checkpoint inputs')
    if args.stage != 'tune' and not (args.tune_report and args.checkpoint):
        raise ValueError('evaluate/profile require --tune-report and --checkpoint')
    if args.stage == 'evaluate' and args.evaluation_report:
        raise ValueError('evaluate may not select checkpoints from previous outer evaluations')
    if args.stage == 'profile' and not args.evaluation_report:
        raise ValueError('profile requires frozen passing --evaluation-report')


def document_code_identity():
    value = code_identity()
    root = Path(__file__).resolve().parents[1]
    for name in ('experiments/mixtral_document.py', 'experiments/mixtral_recovery.py', 'experiments/mixtral_fit_experiment.py'):
        value['sources'][name] = sha256_file(root/name)
    return value


def build_identity(config, model_info, current):
    return {'model': model_info['model'], 'tokenizer': model_info['tokenizer'], **current['data'],
        'code': current['code'], 'training_population': current['training_population'],
        'document_contract': experiment_contract(config),
        'numerical_contract': {'batch_documents': 1, 'sequence_length': config['sequence_length'],
            'dtype': config['dtype'], 'allow_tf32': config['allow_tf32'], 'attention': 'sdpa',
            'experts_implementation': config['experts_implementation'],
            'actual_device_map': model_info.get('device_map'), 'cuda_devices': model_info.get('cuda_devices'),
            'runtime_versions': current['runtime_versions'],
            'tail_protocol': 'full padded [1,S,D] terminal tail and head before loss mask; no singleton replay',
            'router_target_scope': 'uniform-action full-document cost table is a supervised surrogate, not arbitrary mixed-policy cost',
            'checkpoint_selection': 'minimum full-document development NLL including update0',
            'selection_scope': 'one frozen candidate on previously inspected outer development, not final heldout'}}


def preflight_quotas(config, inputs, data):
    training = config['document']['training']
    metadata = inputs['metadata']
    sequence = config['sequence_length']
    logits = sequence * metadata.vocab_size
    # Cache owns full h0 and h1 (BF16), CPU ids/targets/masks, plus conservative metadata allowance.
    estimated = len(data['train']) * sequence * (metadata.hidden_size * 4 + 32)
    if logits > training['max_logit_elements'] or estimated > training['max_cache_bytes']:
        raise ValueError('full-document cache/logit quota exceeded before weight loading')
    if config['document']['kind'] == 'router':
        if sum(int((r['attention_mask'][:-1]*r['attention_mask'][1:]).sum()) for r in data['train']) > training['max_train_tokens']:
            raise ValueError('router full train-token table quota exceeded before weight loading')
        if 29 * len(data['train']) > training['max_table_calls']:
            raise ValueError('router action-table forward quota exceeded before weight loading')
    return {'estimated_train_cache_bytes': estimated, 'full_document_logit_elements': logits,
            'padded_input_tokens_by_role': {r: len(data[r]) * sequence for r in ROLES}}


def evaluate_documents(adapter, rows, candidate, config, *, role, work=None):
    from mixtral_lab.document_training import evaluate_documents as core_evaluate
    if role not in ('development', 'selection'):
        raise ValueError('only development and frozen outer selection may use candidate evaluator')
    kwargs = dict(kind=config['document']['kind'], max_logit_elements=config['document']['training']['max_logit_elements'])
    counter = None if work is None else work.setdefault(role + '_execution', {})
    records = core_evaluate(adapter, rows, candidate, role=role, work=counter, **kwargs)
    # The core reports per-document primary metrics; no cached train score is substituted.
    if isinstance(records, dict):
        records = records['per_document']
    tokens = sum(row['tokens'] for row in records)
    if not records or tokens <= 0:
        raise ValueError('empty document evaluation')
    result = {'status': 'ok', 'role': role, 'protocol': PROTOCOL,
        'document_ids': [r['id'] for r in rows], 'per_document': records,
        'document_statistics': document_statistics(records),
        'execution': 'fresh full prefix plus shared full padded terminal tail/head; no cached development scoring',
        'scope': 'independent development selection' if role == 'development' else 'previously inspected outer development; not final heldout'}
    for name, key in (('native', 'native'), ('controlled', 'controlled_original'), ('candidate', 'candidate')):
        result[key] = {'nll': sum(r[name+'_nll']*r['tokens'] for r in records)/tokens,
                       'tokens': tokens, 'sequences': len(records)}
    result['nll'] = result['candidate']['nll']
    validate_paired_metrics(result, role, result['document_ids'], config['nll_tolerance'])
    if role == 'selection':
        threshold = config['document']['selection_min_gain']
        native_gain = result['native']['nll']-result['nll']
        controlled_gain = result['controlled_original']['nll']-result['nll']
        result['quality_gate'] = {'threshold_nats': threshold, 'gain_vs_native_nats': native_gain,
            'gain_vs_controlled_nats': controlled_gain, 'passed': native_gain >= threshold and controlled_gain >= threshold}
    return finite_tree(result, 'full-document evaluation')


def make_development_evaluator(adapter, rows, config, work):
    training = config['document']['training']
    maximum = 2 + math.ceil(training['updates']/training['eval_interval'])
    work['development_evaluation_budget'] = maximum
    work['development_evaluations_attempted'] = 0
    def development(candidate):
        if work['development_evaluations_attempted'] >= maximum:
            raise ValueError('development evaluation budget exceeded')
        work['development_evaluations_attempted'] += 1
        result = evaluate_documents(adapter, rows, candidate, config, role='development', work=work)
        work['development_evaluations_completed'] = work.get('development_evaluations_completed', 0)+1
        return result
    development.contract = {'role': 'development', 'document_ids': [r['id'] for r in rows], 'protocol': PROTOCOL}
    return development


def replay(adapter, caches, candidate, config, identity, *, checkpoint=None):
    from mixtral_lab.document_training import verify_document_replay
    training = config['document']['training']
    result = verify_document_replay(adapter, caches, candidate, kind=config['document']['kind'],
        atol=training['parity_atol'], rtol=training['parity_rtol'])
    result['binding'] = {'identity_sha256': canonical_hash(identity),
        'experiment_contract_sha256': canonical_hash(experiment_contract(config)),
        'checkpoint': deepcopy(checkpoint), 'candidate_scope': 'initial zero' if checkpoint is None else 'frozen development-best'}
    require_replay_evidence(result, config, identity['train']['selected_document_ids'])
    return result


def check_checkpoint_metadata(metadata, config, sha, summary=None):
    finite_tree(metadata, 'checkpoint metadata')
    if metadata.get('checkpoint_sha256') != sha:
        raise ValueError('checkpoint changed during loading')
    if metadata.get('training_config', metadata.get('config')) != config['document']['training']:
        raise ValueError('checkpoint configuration mismatch')
    if summary is not None:
        if metadata.get('update') != summary['best_update']:
            raise ValueError('checkpoint is not development best')
        metrics = metadata.get('development_metrics', metadata.get('development'))
        if not isinstance(metrics, dict) or metrics.get('nll') != summary['best_development_nll']:
            raise ValueError('checkpoint development-best metrics mismatch')
        if summary.get('best_development_metrics') != metrics:
            raise ValueError('checkpoint primary development metrics differ from tuning summary')


def verify_checkpoint_metrics(metadata, config, identity):
    metrics = metadata.get('development_metrics', metadata.get('development'))
    return validate_paired_metrics(metrics, 'development', identity['development']['selected_document_ids'], config['nll_tolerance'])


def downstream_preflight(args, config, current):
    tune = read_report(args.tune_report, 'tune')
    require_development_evidence(tune)
    identity = verify_frozen_report(tune, config, args.checkpoint, current)
    _, _, loader = trainer_api(config['document']['kind'])
    _, initial = loader(tune['baseline_checkpoint']['path'], identity, 'cpu')
    check_checkpoint_metadata(initial, config, tune['baseline_checkpoint']['sha256'])
    first = verify_checkpoint_metrics(initial, config, identity)
    if (initial.get('update') != 0 or first['nll'] != tune['tune_summary']['initial_development_nll']
            or first != tune['tune_summary'].get('initial_development_metrics')):
        raise ValueError('initial checkpoint metrics differ from development baseline')
    _, best = loader(args.checkpoint, identity, 'cpu')
    check_checkpoint_metadata(best, config, tune['checkpoint']['sha256'], tune['tune_summary'])
    verify_checkpoint_metrics(best, config, identity)
    evaluation = None
    evidence = {'tune_report_path': str(Path(args.tune_report).resolve()), 'tune_report_sha256': tune['report_sha256']}
    if args.stage == 'profile':
        evaluation = read_report(args.evaluation_report, 'evaluate')
        verify_frozen_report(evaluation, config, args.checkpoint, current)
        if (evaluation.get('evidence', {}).get('tune_report_sha256') != tune['report_sha256']
                or evaluation['identity'] != identity or evaluation['freeze_manifest'] != tune['freeze_manifest']
                or evaluation['replay_initial'] != tune['replay_initial'] or evaluation['replay_frozen'] != tune['replay_frozen']):
            raise ValueError('outer evaluation is not bound to this frozen tune')
        require_quality_evidence(evaluation)
        evidence.update(evaluation_report_path=str(Path(args.evaluation_report).resolve()), evaluation_report_sha256=evaluation['report_sha256'])
    return tune, evaluation, evidence


def profile_functions(adapter, candidate, config):
    from mixtral_lab.document_training import online_logits
    from mixtral_lab.protocol import batch_row
    device = input_device(adapter.model)
    def native(row):
        return adapter.model(**batch_row(row, device), use_cache=False, output_router_logits=False).logits.to(device)
    def controlled(row):
        return online_logits(adapter, row, None, kind=config['document']['kind'], max_logit_elements=config['document']['training']['max_logit_elements']).to(device)
    def learned(row):
        return online_logits(adapter, row, candidate, kind=config['document']['kind'], max_logit_elements=config['document']['training']['max_logit_elements']).to(device)
    return {'native': native, 'controlled_original_r1': controlled, 'frozen_candidate_'+config['document']['kind']: learned}


def assert_inputs_unchanged(config, current):
    if document_code_identity() != current['code'] or normalized_artifact_stats(config) != current['artifact_stats']:
        raise ValueError('source or immutable model artifacts changed during document stage')
    for role in ROLES:
        if sha256_file(config['data_files'][role]) != current['data'][role]['file_sha256']:
            raise ValueError('document data changed during stage')


def main(argv=None):
    args = parser().parse_args(argv)
    raw = json.loads(Path(args.config).read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get('output_dir'), str) or not raw['output_dir'].strip():
        raise ValueError('configuration needs fresh output_dir for failure evidence')
    output = fresh_output(raw['output_dir'])
    report = {'schema': REPORT_SCHEMA, 'stage': args.stage, 'status': 'running', 'phase': 'preflight',
        'model_load_started': False, 'work_accounting': {'calibration_evaluations': 0},
        'calibration_scope': {'evaluated': False, 'purpose': 'reserved source, identity and leakage checks only; same-protocol numerical screens use train documents'},
        'limitations': ['historical singleton cache bridge is not this protocol and is not rewritten',
            'full-document runtime parity is measured, not a proof across all GPUs and batch shapes',
            'outer selection is previously inspected development, not final heldout; newtest unsupported',
            'added adapter capacity is not a recurrent expert loop; no automatic deployment approval',
            'source/report hashes provide integrity, not authentication; local model files must remain immutable']}
    atomic_json(output/'report.json', report)
    started = time.perf_counter()
    try:
        validate_stage_arguments(args)
        config = validate_document_config(raw)
        report.update(config=config, config_sha256=canonical_hash(config), experiment_contract_sha256=canonical_hash(experiment_contract(config)))
        read_documents(config)
        inputs = load_local_inputs(config)
        data, data_identity = prepare_data(config, inputs['tokenizer'])
        population = training_population(config, data_identity)
        current = {'code': document_code_identity(), 'local_inputs': inputs['identity'], 'artifact_stats': normalized_artifact_stats(config),
                   'data': data_identity, 'training_population': population, 'runtime_versions': runtime_versions()}
        quotas = preflight_quotas(config, inputs, data)
        report['preflight'] = {**current, 'tensor_quotas': quotas, 'status': 'running'}
        tune = evaluation = None
        if args.stage != 'tune':
            tune, evaluation, report['evidence'] = downstream_preflight(args, config, current)
        report['preflight'].update(status='complete', wall_seconds=time.perf_counter()-started)
        atomic_json(output/'report.json', report)
        if not torch.cuda.is_available():
            raise RuntimeError('real Mixtral document experiments require CUDA; CPU tests are not GPU experiments')
        torch.manual_seed(config['seed'])
        with precision_scope(config['allow_tf32']):
            report.update(phase='model_load', model_load_started=True)
            atomic_json(output/'report.json', report)
            load_started = time.perf_counter()
            model, _, adapter, model_info = load_local_model(config, local_inputs=inputs)
            report['model_load_wall_seconds'] = time.perf_counter()-load_started
            assert_inputs_unchanged(config, current)
            identity = build_identity(config, model_info, current)
            if tune is not None and tune['identity'] != identity:
                raise ValueError('loaded model/runtime differs from frozen training')
            report.update(identity=identity, model=model_info)
            cls, trainer, loader = trainer_api(config['document']['kind'])
            training = config['document']['training']
            if args.stage == 'tune':
                from mixtral_lab.document_training import cache_documents
                report['phase'] = 'cache_and_initial_replay'
                caches = cache_documents(adapter, data['train'], max_cache_bytes=training['max_cache_bytes'], max_logit_elements=training['max_logit_elements'], work=report['work_accounting'].setdefault('train_cache', {}))
                report['replay_initial'] = replay(adapter, caches, None, config, identity)
                atomic_json(output/'report.json', report)
                report['phase'] = 'tuning'
                evaluator = make_development_evaluator(adapter, data['development'], config, report['work_accounting'])
                _, summary = trainer(adapter, caches, evaluator, cls(**training), identity, output/'tune')
                report['tune_summary'] = finite_tree(summary, 'tune summary')
                atomic_json(output/'report.json', report)
                if summary.get('status') not in ('complete', 'completed'):
                    raise RuntimeError('training execution/gradient guard failed; diagnostics retained')
                checkpoint = output/'tune'/'checkpoint_best.pt'
                candidate, metadata = loader(checkpoint, identity, adapter.device)
                sha = sha256_file(checkpoint)
                check_checkpoint_metadata(metadata, config, sha, summary)
                verify_checkpoint_metrics(metadata, config, identity)
                initial_path = output/'tune'/'checkpoint_initial.pt'
                _, initial = loader(initial_path, identity, 'cpu')
                initial_sha = sha256_file(initial_path)
                check_checkpoint_metadata(initial, config, initial_sha)
                metrics = verify_checkpoint_metrics(initial, config, identity)
                if (initial.get('update') != 0 or metrics['nll'] != summary['initial_development_nll']
                        or metrics != summary.get('initial_development_metrics')):
                    raise ValueError('initial checkpoint development metrics mismatch')
                report['baseline_checkpoint'] = {'path': str(initial_path.resolve()), 'sha256': initial_sha, 'update': 0}
                report['checkpoint'] = {'path': str(checkpoint.resolve()), 'sha256': sha, 'update': metadata['update']}
                report['phase'] = 'frozen_replay'
                report['replay_frozen'] = replay(adapter, caches, candidate, config, identity, checkpoint=report['checkpoint'])
                gain = summary['initial_development_nll']-summary['best_development_nll']
                threshold = config['document']['development_min_gain']
                report['development_gate'] = {'threshold_nats': threshold, 'gain_nats': gain,
                    'passed': summary['best_update'] > 0 and gain >= threshold, 'fallback': 'retain initial model; do not evaluate/profile'}
                freeze = {'schema': FREEZE_SCHEMA, 'baseline_checkpoint': report['baseline_checkpoint'], 'checkpoint': report['checkpoint'],
                    'identity_sha256': canonical_hash(identity), 'experiment_contract_sha256': canonical_hash(experiment_contract(config)), **current}
                report.update(freeze_manifest=freeze, freeze_manifest_sha256=canonical_hash(freeze))
            else:
                candidate, metadata = loader(args.checkpoint, identity, adapter.device)
                check_checkpoint_metadata(metadata, config, tune['checkpoint']['sha256'], tune['tune_summary'])
                for key in ('baseline_checkpoint', 'checkpoint', 'freeze_manifest', 'freeze_manifest_sha256',
                            'replay_initial', 'replay_frozen', 'development_gate', 'tune_summary'):
                    report[key] = deepcopy(tune[key])
                if args.stage == 'evaluate':
                    report['phase'] = 'outer_selection'
                    report['work_accounting']['outer_evaluations_attempted'] = 1
                    report['evaluation'] = evaluate_documents(adapter, data['selection'], candidate, config, role='selection', work=report['work_accounting'])
                    report['work_accounting']['outer_evaluations_completed'] = 1
                else:
                    report['phase'] = 'profiling'
                    require_quality_evidence(evaluation)
                    report['quality_gate'] = evaluation['evaluation']['quality_gate']
                    report['timing'] = finite_tree(measure_calls(profile_functions(adapter, candidate, config), data['selection'], model_info['cuda_devices'], config), 'timing')
                    report['timing']['scope'] = 'complete fresh prefix plus full-document terminal tail/head; no cached-tail latency substitution'
            atomic_json(output/'freeze_manifest.json', report['freeze_manifest'])
            assert_inputs_unchanged(config, current)
            for checkpoint_key in ('baseline_checkpoint', 'checkpoint'):
                if sha256_file(report[checkpoint_key]['path']) != report[checkpoint_key]['sha256']:
                    raise ValueError('frozen checkpoint changed during stage')
            report.update(status='complete', phase='complete', total_wall_seconds=time.perf_counter()-started)
            report = seal_report(report)
            atomic_json(output/'report.json', report)
    except Exception as exc:
        summary_path = output/'tune'/'summary.json'
        if summary_path.is_file() and summary_path.stat().st_size <= 64*1024*1024 and 'tune_summary' not in report:
            try:
                report['tune_summary'] = finite_tree(json.loads(summary_path.read_text()), 'partial tune summary')
            except (ValueError, TypeError):
                report['partial_summary_path'] = str(summary_path)
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', failure_phase=report['phase'], total_wall_seconds=time.perf_counter()-started)
        report.pop('report_sha256', None)
        atomic_json(output/'report.json', report)
        raise
    print(json.dumps({'report': str(output/'report.json'), 'status': report['status']}))
    return report


if __name__ == '__main__':
    main()
