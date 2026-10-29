"""Read-only CPU supervision diagnostics; never execute a Mixtral model or select a policy.

Stored valid-row features and CPU GEMM batches are NOT the original padded GPU
policy execution. All argmax/margin findings here are mathematical diagnostics.
Hashes establish artifact integrity and binding, not authenticity.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import pickletools
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

import torch

from .document_protocol import FREEZE_SCHEMA, experiment_contract, read_report
from .document_router import TABLE_FORMAT, SCOPE, _sources, load_document_router_checkpoint
from .full_action_fit import _digest, _tensor_hashes
from .recovery_router import RecoveryPairRouter, compact_r1_features
from .training import pair_actions

MAX_ARTIFACT_BYTES = 512 * 1024**2
MAX_CHECKPOINT_BYTES = 64 * 1024**2
LABEL = 'SUPERVISION DIAGNOSTIC — not deployment gain'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def _bounded_snapshot(path, *, max_bytes=MAX_ARTIFACT_BYTES):
    """Snapshot a regular bounded ZIP, preflight declared allocation, then weights_only.

    Legacy pickle archives, links, duplicate members and compressed archives are
    intentionally unsupported. Every declared uncompressed byte counts toward the
    bound (not just tensor storage); this is conservative for torch.save artifacts.
    """
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), 'artifact must be a regular non-symlink file')
    _require(stat.S_ISREG(path.stat().st_mode) and 0 < path.stat().st_size <= max_bytes,
             'artifact file exceeds size bound or is not regular')
    with path.open('rb') as stream:
        snapshot = stream.read(max_bytes + 1)
    _require(len(snapshot) <= max_bytes, 'artifact grew beyond size bound')
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
            infos = archive.infolist()
            _require(0 < len(infos) <= 4096, 'invalid archive member count')
            names = [i.filename for i in infos]
            _require(len(set(names)) == len(names), 'duplicate archive members')
            _require(sum(i.file_size for i in infos) <= max_bytes,
                     'ZIP declared uncompressed storage exceeds byte budget')
            for member in infos:
                parts = PurePosixPath(member.filename).parts
                mode = member.external_attr >> 16
                _require(parts and not member.filename.startswith('/') and '..' not in parts
                         and '\\' not in member.filename and not member.is_dir()
                         and not stat.S_ISLNK(mode) and not (member.flag_bits & 1)
                         and member.compress_type == zipfile.ZIP_STORED,
                         'unsupported or malformed ZIP member')
            _require(all(i.file_size <= 8*1024**2 for i in infos if i.filename.endswith('/data.pkl')),
                     'torch pickle metadata exceeds 8 MiB bound')
            roots = {PurePosixPath(n).parts[0] for n in names}
            _require(len(roots) == 1 and any(n.endswith('/data.pkl') for n in names),
                     'not a torch ZIP archive')
            _require(archive.testzip() is None, 'ZIP CRC mismatch')
            metadata = archive.read(next(n for n in names if n.endswith('/data.pkl')))
            for index, (opcode, argument, _) in enumerate(pickletools.genops(metadata)):
                _require(index < 1_000_000, 'torch pickle opcode budget exceeded')
                _require(not (opcode.name == 'GLOBAL' and 'sparse' in str(argument)),
                         'sparse tensor pickle metadata unsupported')
    except (zipfile.BadZipFile, RuntimeError, EOFError) as error:
        raise ValueError('invalid bounded torch archive') from error
    return snapshot


def bounded_torch_load(path, *, max_bytes=MAX_ARTIFACT_BYTES):
    """Return a weights-only CPU artifact after bounded ZIP preflight."""
    snapshot = _bounded_snapshot(path, max_bytes=max_bytes)
    value = torch.load(io.BytesIO(snapshot), map_location='cpu', weights_only=True)
    return value, hashlib.sha256(snapshot).hexdigest()


def _validate_report(report):
    _require(report['config']['document']['kind'] == 'router', 'router tune report required')
    summary, identity = report['tune_summary'], report['identity']
    _require(summary.get('status') == 'complete', 'completed router tuning required')
    freeze = report.get('freeze_manifest', {})
    _require(freeze.get('schema') == FREEZE_SCHEMA and report.get('freeze_manifest_sha256') == _digest(freeze),
             'freeze manifest checksum mismatch')
    _require(freeze.get('identity_sha256') == _digest(identity)
             and freeze.get('experiment_contract_sha256') == _digest(experiment_contract(report['config'])),
             'frozen identity or contract mismatch')
    for name in ('checkpoint', 'baseline_checkpoint'):
        _require(freeze.get(name) == report.get(name), 'frozen checkpoint identity mismatch')
    _require(report['baseline_checkpoint']['update'] == 0
             and report['checkpoint']['update'] == summary['best_update'], 'report checkpoint update mismatch')
    _require(freeze.get('code') == identity.get('code'), 'frozen source identity mismatch')
    sources = identity.get('code', {}).get('sources', {})
    _require(isinstance(sources, dict) and sources, 'missing report source hashes')
    root = Path(__file__).resolve().parents[1]
    for name, digest in sources.items():
        path = root / name
        _require(isinstance(name, str) and not Path(name).is_absolute() and '..' not in Path(name).parts
                 and path.resolve().is_relative_to(root) and path.is_file() and _sha(path) == digest,
                 'report source identity mismatch: ' + str(name))
    expected = {'mixtral_lab/' + name: digest for name, digest in _sources().items()}
    _require(all(sources.get(k) == v for k, v in expected.items()), 'incomplete report trainer sources')
    return identity


def validate_artifacts(tune_report, checkpoint, table):
    """Load only train table/router weights after sealed report/source validation."""
    report = read_report(tune_report, 'tune')
    identity = _validate_report(report)
    snapshot = _bounded_snapshot(checkpoint, max_bytes=MAX_CHECKPOINT_BYTES)
    checkpoint_sha = hashlib.sha256(snapshot).hexdigest()
    raw = torch.load(io.BytesIO(snapshot), map_location='cpu', weights_only=True)
    # Strict legacy loader reads only our checked immutable snapshot, not a path
    # which another process could replace between preflight and torch.load.
    with tempfile.TemporaryDirectory(prefix='router-diagnostic-') as temporary:
        checked = Path(temporary)/'checkpoint.pt'
        checked.write_bytes(snapshot)
        actor, metadata = load_document_router_checkpoint(checked, identity, device='cpu')
    metadata['path'] = str(Path(checkpoint).resolve())
    _require(metadata['checkpoint_sha256'] == checkpoint_sha, 'checkpoint changed during strict loading')
    binding = raw['binding']
    summary = report['tune_summary']
    _require(binding['config'] == report['config']['document']['training']
             and binding['sha256'] == summary['binding_sha256'], 'training contract/report binding mismatch')
    update = metadata['update']
    if checkpoint_sha == report['checkpoint']['sha256']:
        meaning = 'development-best (may be update-zero fallback; not latest training state)'
        _require(update == summary['best_update'] and metadata['development_metrics'] == summary['best_development_metrics']
                 and metadata['train_metrics'] == summary['best_train_metrics'], 'best checkpoint metrics mismatch')
    else:
        meaning = 'latest evaluated training state (not deployment-selected)'
        history = summary['development_history']
        _require(history and update == history[-1]['update'] == summary['successful_updates'],
                 'checkpoint is neither frozen best nor terminal latest')
        _require(metadata['train_metrics'] == summary['latest_train_metrics'], 'latest train metrics mismatch')
        _require(all(metadata['development_metrics'].get(k) == v for k, v in history[-1].items()
                     if k not in ('update', 'patience_observations')), 'latest development metrics mismatch')
    value, table_sha = bounded_torch_load(table)
    _require(isinstance(value, dict) and set(value) == {'format','binding','provenance','tensors','scope'}
             and value['format'] == TABLE_FORMAT and value['scope'] == SCOPE, 'unsupported surrogate table')
    _require(value['binding'] == binding, 'table/checkpoint training contract mismatch')
    tensors, provenance = value['tensors'], value['provenance']
    _require(isinstance(tensors, dict) and set(tensors) == {'features','native_pairs','costs','native_losses'},
             'invalid table tensor keys')
    _require(all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' and t.layout == torch.strided
                 for t in tensors.values()), 'dense CPU tensors required')
    f, p, c, n = (tensors[k] for k in ('features','native_pairs','costs','native_losses'))
    count = len(f)
    _require(f.ndim == 2 and 0 < count <= binding['config']['max_train_tokens']
             and f.shape[1] == actor.feature_dim and p.shape == (count,2)
             and c.shape == (count,28) and n.shape == (count,), 'table shape mismatch')
    _require(all(t.is_floating_point() and bool(torch.isfinite(t).all()) for t in (f,c,n))
             and not bool((c < 0).any()) and not bool((n < 0).any()), 'invalid table numeric values')
    pair_actions(p)
    _require(_tensor_hashes(tensors) == binding['table_hashes'], 'table tensor checksum mismatch')
    _require(isinstance(provenance,list) and len(provenance) == count
             and _digest(provenance) == binding['provenance_sha256'], 'table provenance checksum mismatch')
    ids = identity['train']['selected_document_ids']
    _require(ids and len(set(ids)) == len(ids) and [v['document_id'] for v in binding['cache_documents']] == ids,
             'train document binding mismatch')
    forbidden = {v for role in ('development','selection','calibration')
                 for v in identity.get(role, {}).get('selected_document_ids', [])}
    _require(not set(ids) & forbidden, 'train/nontrain identity overlap')
    observed, seen, last = [], set(), {}
    for row in provenance:
        _require(isinstance(row,dict) and set(row) == {'document_id','token_position','target','role'}, 'invalid provenance row')
        doc, pos = row['document_id'], row['token_position']
        _require(doc in ids and row['role'] == 'train' and type(pos) is int and 0 <= pos < report['config']['sequence_length']-1
                 and type(row['target']) is int and row['target'] >= 0 and (doc,pos) not in seen,
                 'nontrain or invalid token provenance')
        _require(pos > last.get(doc,-1), 'unordered token provenance')
        if not observed or observed[-1] != doc:
            observed.append(doc)
        seen.add((doc,pos)); last[doc] = pos
    _require(observed == ids and count == summary['training_coverage']['eligible_train_positions'], 'train population mismatch')
    return actor, tensors, {'report':report, 'metadata':metadata, 'checkpoint_meaning':meaning,
        'checkpoint_sha256':checkpoint_sha,'table_sha256':table_sha,'binding':binding}


def _distribution(values):
    values = values.detach().double().flatten()
    quantiles = values.quantile(torch.tensor([0.,.01,.1,.5,.9,.99,1.], dtype=torch.float64)).tolist()
    return {'count':len(values),'mean':float(values.mean()),'std_population':float(values.std(unbiased=False)),
            'quantiles':dict(zip(('min','p01','p10','p50','p90','p99','max'),quantiles))}


@torch.no_grad()
def compute_diagnostics(actor, tensors, config, *, batch_size=256, near_tie_tolerance=1e-6):
    """Deterministic input-order CPU batches; no batch-shape/deployment parity claim."""
    _require(type(batch_size) is int and 1 <= batch_size <= 4096, 'batch_size must be integer 1..4096')
    _require(type(near_tie_tolerance) in (int,float) and math.isfinite(near_tie_tolerance)
             and 0 <= near_tie_tolerance <= 1., 'invalid near-tie tolerance')
    _require(actor.weight.device.type == 'cpu', 'CPU actor required')
    prior = RecoveryPairRouter(actor.feature_dim, actor.epsilon, device='cpu').eval()
    parts = {}
    def add(name, value):
        parts.setdefault(name, []).append(value.detach().double())
    for start in range(0,len(tensors['features']),batch_size):
        sl = slice(start,start+batch_size)
        f,p,c,n = (tensors[k][sl] for k in ('features','native_pairs','costs','native_losses'))
        compact = compact_r1_features(f)
        native = pair_actions(p)
        rows = torch.arange(len(f))
        qlog = (-(c.double()-c.double().min(-1,keepdim=True).values)/config['temperature']).log_softmax(-1)
        q = qlog.exp()
        native_cost = c[rows,native].double()
        alt_cost = c.double().clone(); alt_cost[rows,native] = math.inf
        advantage = native_cost-alt_cost.min(-1).values
        top_cost = c.double().sort(-1).values
        add('target_entropy',-torch.special.xlogy(q,q).sum(-1)); add('target_max_probability',q.max(-1).values)
        add('target_native_mass',q[rows,native]); add('target_native_argmax_agreement',(q.argmax(-1)==native))
        add('target_native_is_cost_minimizer',native_cost == top_cost[:,0])
        add('target_native_within_best_cost_tolerance',native_cost-top_cost[:,0] <= near_tie_tolerance)
        add('unique_target_argmax_differs_native',
            (q.argmax(-1)!=native) & (top_cost[:,1]-top_cost[:,0] > near_tie_tolerance))
        add('target_cost_gap_best_second',top_cost[:,1]-top_cost[:,0])
        add('native_minus_best_alternate_cost',advantage)
        add('native_advantage_temperature_normalized',advantage/config['temperature'])
        spread = c.double().std(-1,unbiased=False)
        add('cost_action_std',spread)
        add('native_advantage_cost_std_normalized',torch.where(spread>0,advantage/spread,torch.zeros_like(spread)))
        add('zero_cost_std',spread==0)
        add('native_uniform_action_cost',native_cost)
        add('native_executed_anchor_loss',n.double())
        add('native_uniform_cost_minus_native_execution',native_cost-n.double())
        add('compact_row_rms',compact.double().square().mean(-1).sqrt())
        for name, model in (('update_zero_prior',prior),('candidate',actor)):
            logits = model.logits(f,p)
            logp = logits.log_softmax(-1); prob = logp.exp()
            alternatives = logits.clone(); alternatives[rows,native] = -math.inf
            margin = logits[rows,native]-alternatives.max(-1).values
            add(name+'.native_vs_best_alternate_margin',margin)
            alternate_probability = prob.clone(); alternate_probability[rows,native] = -math.inf
            add(name+'.native_vs_best_alternate_probability_margin',prob[rows,native]-alternate_probability.max(-1).values)
            add(name+'.native_argmax_agreement',logits.argmax(-1)==native)
            add(name+'.near_tie',margin.abs()<=near_tie_tolerance)
            top_logits = logits.topk(2,dim=-1).values
            add(name+'.top_two_logit_gap',top_logits[:,0]-top_logits[:,1])
            add(name+'.soft_target_ce',-(q*logp.double()).sum(-1))
            normalized_logp = logp.double().log_softmax(-1)
            normalized_reference = model.reference_log_probs(p).double().log_softmax(-1)
            kl = (normalized_logp.exp()*(normalized_logp-normalized_reference)).sum(-1)
            add(name+'.native_reference_kl',kl)
            add(name+'.expected_uniform_action_ce_surrogate',(logp.double().softmax(-1)*c.double()).sum(-1))
            add(name+'.entropy',-(prob.double()*logp.double()).sum(-1))
        reference = prior.log_probs(f,p).exp()
        delta = actor.log_probs(f,p).exp()-reference
        residual = actor.logits(f,p)-actor.reference_log_probs(p)
        add('candidate.probability_delta_l1',delta.abs().sum(-1))
        add('candidate.native_probability_delta',delta[rows,native])
        add('candidate.residual_logit_l2',residual.norm(dim=-1))
        add('candidate.residual_logit_max_abs',residual.abs().max(-1).values)
    values = {k:torch.cat(v) for k,v in parts.items()}
    distributions = {k:_distribution(v) for k,v in values.items()}
    counts = {'positions':len(tensors['features']),
        'target_native_argmax':int(values['target_native_argmax_agreement'].sum()),
        'target_native_is_cost_minimizer_including_ties':int(values['target_native_is_cost_minimizer'].sum()),
        'native_is_uniform_cost_minimizer':int(values['target_native_is_cost_minimizer'].sum()),
        'native_within_best_cost_tolerance':int(values['target_native_within_best_cost_tolerance'].sum()),
        'unique_target_argmax_differs_native':int(values['unique_target_argmax_differs_native'].sum()),
        'candidate_top_two_logit_near_ties':int((values['candidate.top_two_logit_gap']<=near_tie_tolerance).sum()),
        'candidate_changed_argmax_from_update_zero':int((values['candidate.native_argmax_agreement']==0).sum()),
        'candidate_near_ties':int(values['candidate.near_tie'].sum()),
        'target_best_second_cost_near_ties':int((values['target_cost_gap_best_second']<=near_tie_tolerance).sum()),
        'native_cost_better_than_every_alternate':int((values['native_minus_best_alternate_cost']<0).sum()),
        'alternate_cost_better_than_native':int((values['native_minus_best_alternate_cost']>0).sum()),
        'native_best_alternate_exact_ties':int((values['native_minus_best_alternate_cost']==0).sum()),
        'zero_cost_std':int(values['zero_cost_std'].sum())}
    objective = {}
    for name in ('update_zero_prior','candidate'):
        ce = distributions[name+'.soft_target_ce']['mean']; kl = distributions[name+'.native_reference_kl']['mean']
        expected = distributions[name+'.expected_uniform_action_ce_surrogate']['mean']
        objective[name] = {'soft_target_ce':ce, 'native_reference_kl':kl,
            'weighted_reference_kl':config['reference_kl_weight']*kl,
            'expected_uniform_action_ce_surrogate':expected,
            'configured_objective':(ce if config['objective']=='soft_target_ce' else expected)+config['reference_kl_weight']*kl}
    return {'label':LABEL,'distributions':distributions,'counts':counts,'objective_components':objective,
        'tie_convention':'argmax uses first action index; target native minimum including exact ties is separately counted. near_tie_tolerance is applied separately to logit margins (log probability units) and target cost gaps (NLL nats), not a selection threshold.',
        'kl_definition':'KL(router policy || fixed full-support native-action prior), normalized in float64; not adapter token-distribution KL.',
        'normalization':'Native advantage = native uniform-action cost minus best alternate cost; positive favors alternate. Temperature normalized and population action-cost-std normalized; zero std mapped to zero and counted. No threshold sweep or action selection.'}


def diagnose_router_artifacts(tune_report, checkpoint, table, output_dir, *, batch_size=256, near_tie_tolerance=1e-6):
    output = Path(output_dir)
    _require(not output.exists(), 'fresh output directory required')
    paths = [Path(p) for p in (tune_report,checkpoint,table)]
    for p, limit in zip(paths, (64*1024**2, MAX_CHECKPOINT_BYTES, MAX_ARTIFACT_BYTES)):
        _require(p.is_file() and not p.is_symlink() and stat.S_ISREG(p.stat().st_mode)
                 and 0 < p.stat().st_size <= limit, 'missing, nonregular or oversized input artifact: '+str(p))
    input_hashes = {str(p.resolve()):_sha(p) for p in paths}
    actor,tensors,evidence = validate_artifacts(tune_report,checkpoint,table)
    before_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        metrics = compute_diagnostics(actor,tensors,evidence['binding']['config'], batch_size=batch_size,
                                      near_tie_tolerance=near_tie_tolerance)
    finally:
        torch.set_num_threads(before_threads)
    _require(all(_sha(path)==digest for path,digest in input_hashes.items()), 'input artifact changed during diagnosis')
    report = evidence['report']
    result = {'schema':'mixtral-router-artifact-diagnostics-v1','status':'complete','label':LABEL,
        'input_sha256':input_hashes,'tune_report_sha256':report['report_sha256'],
        'diagnostic_sources':{name:_sha(Path(__file__).resolve().parents[1]/name) for name in
            ('mixtral_lab/router_artifact_diagnostics.py','experiments/mixtral_router_diagnose.py')},
        'binding_sha256':evidence['binding']['sha256'],'checkpoint_update':evidence['metadata']['update'],
        'checkpoint_meaning':evidence['checkpoint_meaning'],'training_config':evidence['binding']['config'],
        'numeric_protocol':{'device':'cpu','actor_dtype':'torch.float32','cost_target_dtype':'torch.float64',
            'stored_feature_dtype':str(tensors['features'].dtype),'batch_size':batch_size,'threads':1,
            'feature_dim':actor.feature_dim,'compact_feature_dim':actor.compact_feature_dim,
            'residual_scale':actor.residual_scale,
            'prior_native_probability':1-actor.epsilon+actor.epsilon/28,
            'prior_alternate_probability':actor.epsilon/28,
            'prior_native_alternate_odds':(1-actor.epsilon+actor.epsilon/28)/(actor.epsilon/28),
            'prior_native_alternate_log_margin':math.log((1-actor.epsilon+actor.epsilon/28)/(actor.epsilon/28)),
            'row_order':'original saved valid train rows, consecutive batches, final short batch retained',
            'torch_version':str(torch.__version__),'near_tie_tolerance':near_tie_tolerance,
            'compact_features':'R1 normalized h plus 8 centered gate logits; row RMS with +1e-6; residual 1/sqrt(D+8)',
            'deployment_batch_parity_claim':False},
        'original_executed_full_document_evidence':{'source':'sealed tune report, not recomputed',
            'initial_train':report['tune_summary']['initial_train_metrics'],
            'latest_train':report['tune_summary']['latest_train_metrics'],
            'best_update':report['tune_summary']['best_update']},
        'metrics':metrics,'model_or_tokenizer_loaded':False,'gpu_calls':0,'training_updates':0,
        'selection_or_calibration_or_test_evaluated':False,
        'limitations':['Valid-row CPU mathematical logits are not executed full-padded GPU policy actions; near ties can depend on batch shape and device.',
            'Uniform-action costs are supervision labels, not a deployable token-wise oracle or costs of arbitrary mixed policies.',
            'No checkpoint selection, hyperparameter change, new training, or deployment gain claim.',
            'Integrity hashes are not authenticity guarantees. Existing model/tokenizer files were not loaded or independently revalidated.']}
    result['diagnostic_sha256'] = _digest(result)
    serialized = json.dumps(result,indent=2,allow_nan=False)+'\n'
    output.mkdir(parents=True, exist_ok=False)
    (output/'diagnostics.json').write_text(serialized)
    counts = metrics['counts']; obj = metrics['objective_components']
    summary = (f'# {LABEL}\n\nCheckpoint update: {result["checkpoint_update"]} — {result["checkpoint_meaning"]}.\n'
        f'CPU valid-row positions: {counts["positions"]}; mathematical changed argmax: {counts["candidate_changed_argmax_from_update_zero"]}; near ties: {counts["candidate_near_ties"]}.\n\n'
        f'Soft-target CE: {obj["update_zero_prior"]["soft_target_ce"]:.9g} → {obj["candidate"]["soft_target_ce"]:.9g}. '
        f'Candidate native-reference KL: {obj["candidate"]["native_reference_kl"]:.9g}.\n\n'
        'These are stored train supervision diagnostics, NOT deployment actions or deployment gain. '
        'See JSON for separate original executed-policy evidence, complete distributions, normalization, bindings and numerical protocol. '
        'No model/tokenizer loading, GPU calls, retraining, selection/calibration/test evaluation, or input writes.\n')
    (output/'summary.md').write_text(summary)
    return result
