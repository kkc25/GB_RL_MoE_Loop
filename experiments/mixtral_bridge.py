"""Diagnose one immutable fit-first checkpoint on its original TRAIN positions.

No training, calibration/development evaluation, newtest, checkpoint selection,
resume, automatic tolerance widening, or deployment approval is performed.
"""
from __future__ import annotations

import argparse
import copy
import json
import io
from pathlib import Path
import subprocess
import time

import torch

from experiments.mixtral_experiment import atomic_json, load_local_inputs, load_local_model, precision_scope
from experiments.mixtral_fit_experiment import (
    build_identity, check_checkpoint_metadata, normalized_artifact_stats, runtime_versions,
)
from mixtral_lab.bridge import (
    MAX_CHECKPOINT_BYTES, bridge_tolerances, diagnose_bridge, prepare_frozen_train,
    safe_path, verify_historical_sources,
)
from mixtral_lab.fit_protocol import (
    experiment_contract, read_report, seal_report, trainer_api, validate_fit_config, verify_frozen_report,
)
from mixtral_lab.protocol import canonical_hash, fresh_output, sha256_file
from mixtral_lab.sampling import plan_stratified_contexts

REPORT_SCHEMA = 'mixtral-cache-online-bridge-report-v1'


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config', required=True, help='original fit-first configuration; no new config schema')
    p.add_argument('--fit-report', required=True, help='sealed original fit report; fit gate need not pass')
    p.add_argument('--checkpoint', required=True, help='the immutable checkpoint_best.pt bound by the report')
    p.add_argument('--output-dir', required=True, help='fresh diagnosis destination; old fit config remains unchanged')
    p.add_argument('--tolerances', help='optional JSON object of fixed diagnostic tolerances, declared before execution')
    return p


def diagnostic_source_identity():
    root = Path(__file__).resolve().parents[1]
    files = ['experiments/mixtral_bridge.py', 'mixtral_lab/bridge.py']
    return {'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD']).decode().strip(),
            'sources': {name: sha256_file(root/name) for name in files},
            'scope': 'current diagnostic implementation; original source bytes separately verified, not rebound'}


def load_frozen_candidate(args, config, fit_report, current):
    """Verify historical bindings before loading even the small candidate."""
    checkpoint = safe_path(args.checkpoint, 'checkpoint')
    report_path = safe_path(args.fit_report, 'fit report').resolve()
    declared = safe_path(fit_report.get('checkpoint', {}).get('path'), 'reported checkpoint')
    frozen = safe_path(fit_report.get('freeze_manifest', {}).get('checkpoint_path'), 'frozen checkpoint')
    if (checkpoint.resolve() != declared.resolve() or checkpoint.resolve() != frozen.resolve()
            or not checkpoint.resolve().is_relative_to(report_path.parent)):
        raise ValueError('checkpoint path is not the original frozen artifact under its fit report directory')
    if not checkpoint.is_file() or checkpoint.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError('checkpoint missing or exceeds diagnostic size bound')
    identities = verify_frozen_report(fit_report, config, checkpoint, current)
    _, _, loader = trainer_api(config['fit']['kind'])
    candidate, metadata = loader(checkpoint, identities, 'cpu')
    expected_update = fit_report.get('checkpoint', {}).get('update')
    if type(expected_update) is not int or not 0 <= expected_update <= config['fit']['training']['updates']:
        raise ValueError('invalid frozen chosen update')
    if fit_report.get('fit_summary', {}).get('best_update') != expected_update:
        raise ValueError('frozen chosen update differs from original fit summary')
    check_checkpoint_metadata(metadata, config, fit_report['checkpoint']['sha256'], expected_update)
    # Verify the checkpoint's measured values agree without requiring improvement.
    summary = fit_report['fit_summary']
    if config['fit']['kind'] == 'router':
        if (metadata.get('train_metrics') != summary.get('best_train_metrics')
                or metadata.get('train_fit_passed') != summary.get('train_fit_passed')):
            raise ValueError('frozen router fit metrics/verdict mismatch')
    elif metadata.get('train_probe_nll') != summary.get('best_train_probe_nll'):
        raise ValueError('frozen adapter probe metric mismatch')
    if config['fit']['kind'] == 'adapter':
        metadata['historical_cached_metrics'] = {'zero': summary['initial_train_probe_nll'],
                                                 'learned': summary['best_train_probe_nll']}
    if config['fit']['kind'] == 'router':
        payload = checkpoint.read_bytes()
        import hashlib
        if hashlib.sha256(payload).hexdigest() != fit_report['checkpoint']['sha256']:
            raise ValueError('checkpoint changed while reading cache binding')
        binding = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)['binding']
        metadata['historical_cached_inputs'] = {'h0': binding['train_h0'], 'targets': binding['train_targets']}
    return candidate, metadata


def assert_unchanged(config, current, diagnostic):
    root = Path(__file__).resolve().parents[1]
    verify_historical_sources(current['code'], root)
    if diagnostic_source_identity() != diagnostic:
        raise ValueError('diagnostic source changed during execution')
    if normalized_artifact_stats(config) != current['artifact_stats']:
        raise ValueError('immutable local model artifacts changed during diagnosis')
    for role, info in current['data'].items():
        if sha256_file(config['data_files'][role]) != info['file_sha256']:
            raise ValueError('bound data file changed during diagnosis')


def main(argv=None):
    args = parser().parse_args(argv)
    output = fresh_output(safe_path(args.output_dir, 'output directory'))
    report = {'schema': REPORT_SCHEMA, 'stage': 'bridge', 'status': 'running',
              'phase': 'preflight', 'model_load_started': False,
              'fit_gate_required': False, 'deployment_approved': False,
              'limitations': ['TRAIN-only numerical diagnosis, not training or development checkpoint selection',
                  'selection/calibration files only hash-checked against the original freeze; no model execution or tokenization on those roles',
                  'no newtest access; no equivalence, held-out gain or deployment approval claim']}
    atomic_json(output/'report.json', report)
    started = time.perf_counter()
    try:
        raw = json.loads(safe_path(args.config, 'config').read_text())
        config = validate_fit_config(raw)
        # Only diagnostic destination may differ from the historical contract.
        config['output_dir'] = str(output)
        report.update(config=config, config_sha256=canonical_hash(config),
                      experiment_contract_sha256=canonical_hash(experiment_contract(config)))
        overrides = json.loads(safe_path(args.tolerances, 'tolerances').read_text()) if args.tolerances else None
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError('tolerances must be a JSON object')
        tolerances = bridge_tolerances(config, overrides)
        report['tolerances'] = tolerances
        fit_report = read_report(safe_path(args.fit_report, 'fit report'), 'fit')
        source = verify_historical_sources(fit_report['identity']['code'], Path(__file__).resolve().parents[1])
        diagnostic = diagnostic_source_identity()
        report.update(diagnostic_source=diagnostic,
            historical_source_verification={'verified': True, 'identity': source,
                'rule': 'every originally listed source byte hash must match; original commit is provenance, not current-checkout equality'},
            evidence={'fit_report_path': str(Path(args.fit_report).resolve()),
                      'fit_report_sha256': fit_report['report_sha256']})
        inputs = load_local_inputs(config)
        rows, data_identity = prepare_frozen_train(config, inputs['tokenizer'], fit_report['freeze_manifest']['data'])
        manifest = plan_stratified_contexts(rows, config['max_train_contexts'], config['seed'], 'train')
        current = {'code': source, 'local_inputs': inputs['identity'],
                   'artifact_stats': normalized_artifact_stats(config), 'data': data_identity,
                   'train_contexts': manifest, 'runtime_versions': runtime_versions()}
        candidate, metadata = load_frozen_candidate(args, config, fit_report, current)
        report.update(preflight={'status': 'complete', **current,
            'data_access': {'train': 'hash, tokenization and original sampled-position plan',
                            'selection': 'original bound file hash only', 'calibration': 'original bound file hash only'}},
            checkpoint=fit_report['checkpoint'], freeze_manifest=fit_report['freeze_manifest'],
            freeze_manifest_sha256=fit_report['freeze_manifest_sha256'], chosen_update=metadata['update'])
        atomic_json(output/'train_contexts.json', manifest)
        atomic_json(output/'report.json', report)
        report['phase'] = 'model_load'
        if not torch.cuda.is_available():
            raise RuntimeError('real Mixtral diagnostics require CUDA; tiny CPU tests are not model experiments')
        torch.manual_seed(config['seed'])
        with precision_scope(config['allow_tf32']):
            report['model_load_started'] = True
            atomic_json(output/'report.json', report)
            model, tokenizer, adapter, model_info = load_local_model(config, local_inputs=inputs)
            assert_unchanged(config, current, diagnostic)
            identity = build_identity(config, model_info, current)
            if identity != fit_report['identity']:
                raise ValueError('loaded model/tokenizer/numerical identity differs from frozen fit')
            report.update(identity=identity, model=model_info, phase='diagnosis')
            candidate = candidate.to(adapter.device).requires_grad_(False).eval()
            report['bridge'] = diagnose_bridge(adapter, rows, candidate, config, manifest=manifest,
                                               checkpoint_metadata=metadata, tolerances=tolerances)
            assert_unchanged(config, current, diagnostic)
            if sha256_file(args.checkpoint) != fit_report['checkpoint']['sha256']:
                raise ValueError('frozen checkpoint changed during diagnosis')
            # Re-read the sealed input, detecting normal edits during diagnosis.
            if read_report(args.fit_report, 'fit') != fit_report:
                raise ValueError('fit report changed during diagnosis')
            report.update(status='complete', phase='complete', total_wall_seconds=time.perf_counter()-started)
            report = seal_report(report)
            atomic_json(output/'report.json', report)
    except Exception as exc:
        report.update(status='failed', error=str(exc), failure_phase=report['phase'],
                      total_wall_seconds=time.perf_counter()-started)
        report.pop('report_sha256', None)
        atomic_json(output/'report.json', report)
        raise
    print(json.dumps({'report': str(output/'report.json'), 'status': report['status'],
                      'numerical_screen_passed': report['bridge']['numerical_screen_passed']}))
    return report


if __name__ == '__main__':
    main()
