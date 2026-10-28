"""Offline fit-first experiment: train-only fit, frozen development check, then cost.

No GPU jobs are started by importing this module. Selection is development data,
not untouched final confirmation. Newtest is never accepted by this protocol.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import torch
from experiments.mixtral_experiment import (
    atomic_json, code_identity, evaluate, evaluate_second_round, input_device,
    load_local_inputs, load_local_model, local_artifact_stats, measure_calls,
    precision_scope, route_diagnostics, validate_native,
)
from mixtral_lab.fit_protocol import (
    FREEZE_SCHEMA, REPORT_SCHEMA, experiment_contract, finite_tree, read_report,
    require_fit_evidence, require_quality_evidence, seal_report, trainer_api,
    validate_fit_config, verify_frozen_report, verify_checkpoint_fit_metrics,
)
from mixtral_lab.protocol import (
    batch_row, canonical_hash, fresh_output, prepare_data, read_documents, sha256_file,
)
from mixtral_lab.sampling import collect_stratified_contexts, plan_stratified_contexts


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=('fit', 'generalize', 'profile'), required=True)
    p.add_argument('--fit-report', help='completed train-only fit report; required after fit')
    p.add_argument('--checkpoint', help='immutable checkpoint_best.pt frozen by --fit-report')
    p.add_argument('--generalization-report', help='passing frozen development report; required for profile')
    return p


def fit_code_identity():
    identity = code_identity()
    root = Path(__file__).resolve().parents[1]
    identity['sources'][str(Path(__file__).resolve().relative_to(root))] = sha256_file(__file__)
    return identity


def normalized_artifact_stats(config):
    # JSON reports normalize tuple metadata to arrays; compare exactly that form.
    return json.loads(json.dumps(local_artifact_stats(config), sort_keys=True))


def runtime_versions():
    return {**{name: version(name) for name in ('torch', 'transformers', 'accelerate', 'safetensors')},
            'cuda_build': torch.version.cuda, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def build_identity(config, model_info, current):
    return {'model': model_info['model'], 'tokenizer': model_info['tokenizer'],
            **current['data'], 'code': current['code'],
            'context_sampling': current['train_contexts'],
            'fit_contract': experiment_contract(config),
            'numerical_contract': {
                'sampler': config['context_sampling'],
                **{key: config[key] for key in ('dtype', 'allow_tf32', 'experts_implementation',
                    'sequence_length', 'sequences', 'max_train_contexts')},
                'attention': 'sdpa', 'runtime_versions': runtime_versions(),
                'actual_device_map': model_info.get('device_map'), 'cuda_devices': model_info.get('cuda_devices'),
                'fit_scope': ('singleton cached train contexts for all-action targets; online policy uses full padded documents'
                              if config['fit']['kind'] == 'router' else
                              'fixed cached train contexts and fixed probe chunk shape; online adapter uses full padded documents'),
                'cross_shape_equivalence': 'not asserted; train-fit and online development are distinct measured questions'}}


def check_checkpoint_metadata(metadata, config, expected_sha, expected_update=None):
    finite_tree(metadata, 'checkpoint metadata')
    if metadata.get('checkpoint_sha256') != expected_sha:
        raise ValueError('checkpoint changed between preflight hashing and deserialization')
    if expected_update is not None and metadata.get('update') != expected_update:
        raise ValueError('loaded checkpoint is not the frozen train-best update')
    settings = metadata.get('training_config', metadata.get('config'))
    if settings != config['fit']['training']:
        raise ValueError('checkpoint training configuration mismatch')


def validate_stage_arguments(args):
    if args.stage == 'fit' and (args.fit_report or args.checkpoint or args.generalization_report):
        raise ValueError('fit starts fresh: report/checkpoint inputs and resume are forbidden')
    if args.stage != 'fit' and not (args.fit_report and args.checkpoint):
        raise ValueError('generalize/profile require --fit-report and --checkpoint')
    if args.stage == 'generalize' and args.generalization_report:
        raise ValueError('generalize cannot reuse a generalization report or select among candidates')
    if args.stage == 'profile' and not args.generalization_report:
        raise ValueError('profile requires --generalization-report; there is no quality-gate override')


def downstream_preflight(args, config, current):
    """No model execution: reject stale reports/checkpoints before heavy loading."""
    fit_report = read_report(args.fit_report, 'fit')
    require_fit_evidence(fit_report)
    identities = verify_frozen_report(fit_report, config, args.checkpoint, current)
    _, _, loader = trainer_api(config['fit']['kind'])
    _, metadata = loader(args.checkpoint, identities, 'cpu')
    check_checkpoint_metadata(metadata, config, fit_report['checkpoint']['sha256'],
                              fit_report['fit_summary']['best_update'])
    verify_checkpoint_fit_metrics(metadata, fit_report)
    evidence = {'fit_report_path': str(Path(args.fit_report).resolve()),
                'fit_report_sha256': fit_report['report_sha256']}
    generalization = None
    if args.stage == 'profile':
        generalization = read_report(args.generalization_report, 'generalize')
        if generalization.get('evidence', {}).get('fit_report_sha256') != fit_report['report_sha256']:
            raise ValueError('generalization report refers to a different fit report')
        verify_frozen_report(generalization, config, args.checkpoint, current)
        if generalization.get('identity') != identities:
            raise ValueError('generalization and fit identities differ')
        require_quality_evidence(generalization, config['fit']['generalization_min_gain'])
        evidence.update(generalization_report_path=str(Path(args.generalization_report).resolve()),
                        generalization_report_sha256=generalization['report_sha256'])
    return fit_report, generalization, evidence


def aggregate_documents(records, name):
    tokens = sum(row['tokens'] for row in records)
    if not records or not tokens:
        raise ValueError('empty development evaluation')
    return {'nll': sum(row[name + '_nll'] * row['tokens'] for row in records) / tokens,
            'tokens': tokens, 'sequences': len(records)}


def evaluate_generalization(adapter, rows, candidate, config):
    """Every selected development document, once per path; never checkpoint selection."""
    records = []
    for row in rows:
        native = evaluate(adapter, [row], native=True)
        control = evaluate(adapter, [row])
        learned = (evaluate(adapter, [row], candidate) if config['fit']['kind'] == 'router'
                   else evaluate_second_round(adapter, [row], candidate))
        finite_tree([native, control, learned], 'development metrics')
        for metric in (native, control, learned):
            nll = metric.get('nll')
            if (isinstance(nll, bool) or not isinstance(nll, (int, float)) or nll < 0
                    or type(metric.get('tokens')) is not int or metric['tokens'] <= 0
                    or metric.get('sequences') != 1):
                raise ValueError('invalid full-document development metric')
        if len({metric['tokens'] for metric in (native, control, learned)}) != 1:
            raise ValueError('development metric populations differ')
        records.append({'document_id': row['id'], 'tokens': native['tokens'],
            'native_nll': native['nll'], 'controlled_nll': control['nll'], 'candidate_nll': learned['nll'],
            'candidate_minus_native_nll': learned['nll'] - native['nll'],
            'candidate_minus_controlled_nll': learned['nll'] - control['nll']})
    result = {'role': 'selection_development', 'per_document': records,
              'native': aggregate_documents(records, 'native'),
              'controlled_original': aggregate_documents(records, 'controlled'),
              'candidate': aggregate_documents(records, 'candidate'),
              'numerical_scope': 'full padded document shape for all three paths; not equivalent to cached/singleton train-fit scores',
              'selection_policy': 'one frozen train-best checkpoint; no fitting, tuning, early stopping or model selection here',
              'inference_limit': 'development evidence only; no token-independence CI, final held-out claim or multi-seed confirmation'}
    gain_native = result['native']['nll'] - result['candidate']['nll']
    gain_controlled = result['controlled_original']['nll'] - result['candidate']['nll']
    threshold = config['fit']['generalization_min_gain']
    control_equivalent = all(abs(row['native_nll'] - row['controlled_nll']) <= config['nll_tolerance'] for row in records)
    result['quality_gate'] = {'threshold_nats': threshold, 'gain_vs_native_nats': gain_native,
        'gain_vs_controlled_nats': gain_controlled, 'controlled_native_nll_screen_passed': control_equivalent,
        'passed': control_equivalent and gain_native >= threshold and gain_controlled >= threshold,
        'threshold_source': 'fit.generalization_min_gain; independent of oracle.practical_gain'}
    if config['fit']['kind'] == 'router':
        result['route_diagnostics'] = route_diagnostics(adapter, rows, candidate)
    else:
        result['route_diagnostics'] = {'routing': 'fixed native R1; residual correction adds capacity, not a learned route/STOP or second expert call'}
    return finite_tree(result, 'generalization')


def profile_functions(adapter, candidate, kind):
    """Equal returned-logits destination, no CE in any timed variant."""
    model = adapter.model
    device = input_device(model)
    def native(row):
        return model(**batch_row(row, device), use_cache=False, output_router_logits=False).logits.to(device)
    def controlled(row):
        return adapter.forward(**batch_row(row, device), max_calls=1).logits.to(device)
    if kind == 'router':
        def learned(row):
            return adapter.forward(**batch_row(row, device), policy=candidate, max_calls=1).logits.to(device)
    else:
        def learned(row):
            h0 = adapter.capture(**batch_row(row, device))
            shape = h0.shape
            h = h0.reshape(-1, shape[-1])
            h1 = adapter.step_pairs(h, adapter.native_pair(h))
            logits = adapter.readout(candidate(h1)).reshape(*shape[:-1], -1)
            return logits.to(device)
    return {'native': native, 'controlled_original_r1': controlled, 'frozen_candidate_' + kind: learned}


def assert_inputs_unchanged(config, current):
    if fit_code_identity() != current['code'] or normalized_artifact_stats(config) != current['artifact_stats']:
        raise ValueError('code or local immutable model artifacts changed during this stage')
    for role, info in current['data'].items():
        if sha256_file(config['data_files'][role]) != info['file_sha256']:
            raise ValueError('development input file changed during this stage')


def main(argv=None):
    args = parser().parse_args(argv)
    # A parseable output path is necessary to retain even early configuration failures.
    raw = json.loads(Path(args.config).read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get('output_dir'), str) or not raw['output_dir'].strip():
        raise ValueError('config must contain a usable fresh output_dir for failure evidence')
    output = fresh_output(raw['output_dir'])
    report = {'schema': REPORT_SCHEMA, 'stage': args.stage, 'status': 'running',
              'phase': 'preflight', 'model_load_started': False,
              'limitations': ['selection is development, not final held-out confirmation',
                  'newtest remains untouched; future final confirmation requires a separately frozen protocol',
                  'no full-backbone training, learned STOP, loop advantage or deployment speed claim']}
    atomic_json(output/'report.json', report)
    started = time.perf_counter()
    try:
        validate_stage_arguments(args)
        config = validate_fit_config(raw)
        report.update(config=config, config_sha256=canonical_hash(config),
                      experiment_contract_sha256=canonical_hash(experiment_contract(config)))
        # Every source role is checked even though fit never evaluates selection.
        read_documents(config)
        inputs = load_local_inputs(config)
        data, data_identity = prepare_data(config, inputs['tokenizer'])
        manifest = plan_stratified_contexts(data['train'], config['max_train_contexts'], config['seed'], 'train')
        atomic_json(output/'train_contexts.json', manifest)
        current = {'code': fit_code_identity(), 'local_inputs': inputs['identity'],
                   'artifact_stats': normalized_artifact_stats(config), 'data': data_identity,
                   'train_contexts': manifest, 'runtime_versions': runtime_versions()}
        report['preflight'] = {'status': 'running', **current,
            'weight_identity_scope': 'ordinary changes rejected by immutable path/inode/size/mtime/ctime snapshot before loading; actual bytes hashed once by model loader'}
        fit_report = generalization_report = None
        if args.stage != 'fit':
            fit_report, generalization_report, report['evidence'] = downstream_preflight(args, config, current)
        report['preflight'].update(status='complete', wall_seconds=time.perf_counter()-started)
        atomic_json(output/'report.json', report)
        report['phase'] = 'model_load'
        if not torch.cuda.is_available():
            raise RuntimeError('real Mixtral experiments require CUDA; CPU tests do not constitute a model experiment')
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
            if fit_report is not None and identity != fit_report['identity']:
                raise ValueError('loaded model/tokenizer/numerical identity differs from frozen fit')
            report.update(identity=identity, model=model_info,
                          environment={'runtime_versions': runtime_versions(), 'cuda': torch.version.cuda,
                              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                              'gpu_device_index_scope': 'visible logical indices; CUDA_VISIBLE_DEVICES records the physical-device mapping when set',
                              'gpus': {str(i): {'name': torch.cuda.get_device_name(i),
                                  'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                                  for i in model_info['cuda_devices']}})
            parity = finite_tree(validate_native(adapter, data['calibration'], config), 'native equivalence')
            report['native_equivalence'] = parity
            atomic_json(output/'report.json', report)
            if parity.get('eligible') is not True:
                raise RuntimeError('native equivalence screen failed; no training, generalization or timing')
            config_class, trainer, loader = trainer_api(config['fit']['kind'])
            if args.stage == 'fit':
                h0, targets, captured = collect_stratified_contexts(adapter, data['train'],
                    config['max_train_contexts'], config['seed'], 'train')
                if captured != manifest:
                    raise ValueError('captured train contexts differ from exact preflight manifest')
                kwargs = {'provenance': manifest['contexts']} if config['fit']['kind'] == 'router' else {}
                _, summary = trainer(adapter, h0, targets, config_class(**config['fit']['training']),
                                     identity, output/'fit', **kwargs)
                finite_tree(summary, 'train-fit summary')
                if summary.get('status') not in ('complete', 'completed'):
                    raise RuntimeError('train-fit stopped on a numerical or execution guard')
                report['fit_summary'] = summary
                checkpoint = output/'fit'/'checkpoint_best.pt'
                _, metadata = loader(checkpoint, identity, 'cpu')
                checkpoint_sha = sha256_file(checkpoint)
                check_checkpoint_metadata(metadata, config, checkpoint_sha, summary.get('best_update'))
                report['checkpoint'] = {'path': str(checkpoint.resolve()), 'sha256': checkpoint_sha,
                    'update': metadata['update'], 'selection_criterion': (
                        'all fixed train contexts: minimum greedy CE, then expected CE; never selection'
                        if config['fit']['kind'] == 'router' else 'minimum fixed train-only probe NLL; never selection')}
                freeze = {'schema': FREEZE_SCHEMA, 'checkpoint_path': str(checkpoint.resolve()),
                    'checkpoint_sha256': checkpoint_sha, 'identity_sha256': canonical_hash(identity),
                    'experiment_contract_sha256': canonical_hash(experiment_contract(config)), **current}
                report.update(freeze_manifest=freeze, freeze_manifest_sha256=canonical_hash(freeze))
                atomic_json(output/'freeze_manifest.json', freeze)
            else:
                candidate, metadata = loader(args.checkpoint, identity, adapter.device)
                check_checkpoint_metadata(metadata, config, fit_report['checkpoint']['sha256'],
                                          fit_report['fit_summary']['best_update'])
                report.update(checkpoint=fit_report['checkpoint'], freeze_manifest=fit_report['freeze_manifest'],
                              freeze_manifest_sha256=fit_report['freeze_manifest_sha256'])
                atomic_json(output/'freeze_manifest.json', report['freeze_manifest'])
                if args.stage == 'generalize':
                    report['generalization'] = evaluate_generalization(adapter, data['selection'], candidate, config)
                else:
                    # Recheck the preregistered gate before the first warmup/timing call.
                    require_quality_evidence(generalization_report, config['fit']['generalization_min_gain'])
                    report['quality_gate'] = generalization_report['generalization']['quality_gate']
                    report['timing'] = finite_tree(measure_calls(profile_functions(adapter, candidate, config['fit']['kind']),
                        data['selection'], model_info['cuda_devices'], config), 'timing')
                    report['timing']['output_device'] = str(input_device(model))
                    report['timing']['candidate_scope'] = ('native R1 plus fixed residual capacity; no second expert call'
                        if config['fit']['kind'] == 'adapter' else 'one frozen learned R1 pair per token')
            assert_inputs_unchanged(config, current)
            if sha256_file(report['checkpoint']['path']) != report['checkpoint']['sha256']:
                raise ValueError('frozen checkpoint bytes changed during this stage')
            report.update(status='complete', phase='complete', total_wall_seconds=time.perf_counter()-started)
            report = seal_report(report)
            atomic_json(output/'report.json', report)
    except Exception as exc:
        report.update(status='failed', error=str(exc), failure_phase=report['phase'])
        report.pop('report_sha256', None)
        if report['phase'] == 'preflight':
            report.setdefault('preflight', {}).update(status='failed')
        atomic_json(output/'report.json', report)
        raise
    print(json.dumps({'report': str(output/'report.json'), 'status': report['status']}))
    return report


if __name__ == '__main__':
    main()
