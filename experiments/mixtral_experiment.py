"""Offline, bounded Mixtral Base validation/profile/frozen-router experiments.

No weight download, held-out confirmation, full-backbone training or deployment.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from mixtral_lab.protocol import (load_config, directory_identity, prepare_data, batch_row,
                                 token_ce, canonical_hash, fresh_output, read_documents)


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=('validate', 'profile', 'train'), default='validate')
    p.add_argument('--actor-checkpoint', help='optional bound actor checkpoint for profile; no resume')
    return p


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def code_identity():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    files = [Path(__file__), *sorted((root / 'mixtral_lab').glob('*.py'))]
    return {'commit': git('rev-parse', 'HEAD').decode().strip(),
            'sources': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


@contextmanager
def precision_scope(allow_tf32):
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def require_complete_checkpoint(loading_info):
    """Fail closed: HF may otherwise initialize absent weights and merely warn."""
    required = ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')
    if not isinstance(loading_info, dict) or any(key not in loading_info for key in required):
        raise ValueError('unrecognized checkpoint loading diagnostics')
    bad = {key: repr(loading_info[key]) for key in required if loading_info[key]}
    if loading_info.get('conversion_errors'):
        bad['conversion_errors'] = repr(loading_info['conversion_errors'])
    if bad:
        raise ValueError('incomplete or incompatible pretrained checkpoint: ' + json.dumps(bad, sort_keys=True))
    return {key: [] for key in required}


def load_local_model(config):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from mixtral_lab.adapter import MixtralAdapter
    # Real experiment paths are offline and explicit. No model-id fallback.
    model_path, tokenizer_path = Path(config['model_path']), Path(config['tokenizer_path'])
    if not model_path.is_dir() or not tokenizer_path.is_dir():
        raise ValueError('download/pin the model and tokenizer separately; local directories are required')
    metadata = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    expected = {'model_type': 'mixtral', 'num_local_experts': 8, 'num_experts_per_tok': 2,
                'hidden_size': 4096, 'num_hidden_layers': 32}
    for name, value in expected.items():
        if getattr(metadata, name, None) != value:
            raise ValueError(f'expected Mixtral-8x7B Base architecture: {name}={value}')
    if getattr(metadata, 'quantization_config', None):
        raise ValueError('quantized checkpoints require a separate experiment')
    model_identity = directory_identity(model_path)
    tokenizer_identity = directory_identity(tokenizer_path) if tokenizer_path.resolve() != model_path.resolve() else model_identity
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    memory = {int(k): v for k, v in config['max_memory'].items()}
    if any(index >= torch.cuda.device_count() for index in memory):
        raise ValueError('configured GPU index is not currently visible')
    model, loading_info = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, dtype=torch.bfloat16,
        device_map=config['device_map'], max_memory=memory, attn_implementation='sdpa',
        experts_implementation=config['experts_implementation'], output_loading_info=True)
    loading_info = require_complete_checkpoint(loading_info)
    model.eval().requires_grad_(False)
    backends = [layer.mlp.experts.config._experts_implementation for layer in model.model.layers]
    if any(value != config['experts_implementation'] for value in backends):
        raise ValueError('requested expert backend was not installed on every layer')
    placement = getattr(model, 'hf_device_map', {})
    if not placement or any(str(v) in ('cpu', 'disk', 'meta') for v in placement.values()):
        raise ValueError('CPU/disk offload or unresolved placement is not a comparable GPU experiment')
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == 'cuda'})
    if any(p.device.type != 'cuda' for p in model.parameters()) or not set(devices).issubset(memory):
        raise ValueError('model placement must use only explicitly budgeted GPUs')
    return model, tokenizer, MixtralAdapter(model), {'model': model_identity, 'tokenizer': tokenizer_identity,
        'declared_upstream_revision': config['model_revision'],
        'revision_status': 'operator-declared; actual local artifact bytes independently hashed',
        'device_map': {k: str(v) for k, v in placement.items()}, 'cuda_devices': devices,
        'checkpoint_loading_info': loading_info, 'model_config': metadata.to_dict(), 'experts_implementation_by_layer': backends}


def input_device(model):
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def evaluate(adapter, rows, policy=None, *, native=False):
    total, tokens = 0., 0
    for row in rows:
        batch = batch_row(row, input_device(adapter.model))
        logits = (adapter.model(**batch, use_cache=False, output_router_logits=False).logits if native
                  else adapter.forward(**batch, policy=policy, max_calls=1).logits)
        values = token_ce(logits, batch['input_ids'], batch['attention_mask'])
        total += float(values.double().sum()); tokens += len(values)
    return {'nll': total/tokens, 'tokens': tokens, 'sequences': len(rows), 'max_calls': 1}


@torch.no_grad()
def validate_native(adapter, rows, config):
    records = []
    for row in rows:
        batch = batch_row(row, input_device(adapter.model))
        from mixtral_lab.adapter import pair_to_action
        native_pairs = []
        hook = adapter.moe.gate.register_forward_hook(lambda module, inputs, result: native_pairs.append(result[2].detach().cpu()))
        try:
            native = adapter.model(**batch, use_cache=False, output_router_logits=False).logits
        finally:
            hook.remove()
        if len(native_pairs) != 1:
            raise ValueError('expected one native terminal routing trace')
        controlled = adapter.forward(**batch, policy=None, max_calls=1)
        a = token_ce(native, batch['input_ids'], batch['attention_mask'])
        b = token_ce(controlled.logits, batch['input_ids'], batch['attention_mask'])
        delta = (b-a).double()
        max_abs = 0.
        exact = True
        # Bounded float32 comparison temporaries, including padding positions.
        x, y = native.reshape(-1, native.shape[-1]), controlled.logits.reshape(-1, native.shape[-1])
        for start in range(0, len(x), 64):
            left = x[start:start+64].cpu()
            right = y[start:start+64].cpu()
            exact = exact and torch.equal(left, right)
            error = (left.float()-right.float()).abs()
            if not bool(torch.isfinite(error).all()):
                raise ValueError('nonfinite parity logits')
            max_abs = max(max_abs, float(error.max()))
        rounds_ok = bool((controlled.rounds == 1).all())
        actions_ok = bool(((controlled.actions >= 1) & (controlled.actions <= 28)).all())
        trace_equal = torch.equal(pair_to_action(native_pairs[0]), controlled.actions.reshape(-1).cpu())
        record = {'document_id': row['id'], 'tokens': len(a), 'native_nll': float(a.double().mean()),
                  'controlled_nll': float(b.double().mean()), 'delta_nll': float(delta.mean()),
                  'ce_mean_abs': float(delta.abs().mean()), 'ce_max_abs': float(delta.abs().max()),
                  'logit_max_abs': max_abs, 'exact_logits': exact,
                  'r1_rounds_valid': rounds_ok, 'pair_actions_valid': actions_ok, 'native_pair_trace_equal': trace_equal}
        record['eligible'] = (rounds_ok and actions_ok and trace_equal and abs(record['delta_nll']) <= config['nll_tolerance']
                              and record['ce_mean_abs'] <= config['ce_mean_abs_tolerance']
                              and max_abs <= config['logit_max_abs_tolerance'])
        records.append(record)
    return {'groups': records, 'eligible': all(r['eligible'] for r in records),
            'scope': 'local calibration documents; numerical screen, not held-out quality equivalence',
            'native_action_trace': 'terminal native gate pairs independently captured with a temporary hook; compared as unordered pair IDs'}


def synchronize(devices):
    for device in devices:
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_calls(functions, rows, devices, config):
    """Same inputs, rotating variant order, synchronized multi-GPU wall only."""
    result = {name: {'wall_ms': []} for name in functions}
    names = list(functions)
    for name in names:
        for i in range(config['warmup']):
            functions[name](rows[i % len(rows)])
    synchronize(devices)
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    for repeat in range(config['repeats']):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in order:
            for i in range(config['measurements']):
                synchronize(devices)
                start = time.perf_counter()
                functions[name](rows[i % len(rows)])
                synchronize(devices)
                result[name]['wall_ms'].append((time.perf_counter()-start)*1000)
    for name, row in result.items():
        values = torch.tensor(row['wall_ms'], dtype=torch.float64)
        row.update(mean_ms=float(values.mean()), p50_ms=float(values.quantile(.5)),
                   p95_ms=float(values.quantile(.95)), measurements=len(values))
    return {'variants': result, 'scope': 'full sequence prefill to all logits; no CE; includes input transfers; no generation/KV cache',
            'peak_memory_bytes': {str(device): {'allocated': torch.cuda.max_memory_allocated(device), 'reserved': torch.cuda.max_memory_reserved(device)} for device in devices},
            'clock': 'host wall time synchronized across all model GPUs; not a single-device CUDA event',
            'exclusive_gpu_status': 'operator must ensure no competing jobs; not enforced by this runner'}


@torch.no_grad()
def collect_train_contexts(adapter, rows, count):
    hidden, targets, left = [], [], count
    for row in rows:
        batch = batch_row(row, input_device(adapter.model))
        h0 = adapter.capture(**batch)
        valid = (batch['attention_mask'][:, :-1].bool() & batch['attention_mask'][:, 1:].bool()).reshape(-1)
        h = h0[:, :-1].reshape(-1, h0.shape[-1])[valid.to(h0.device)][:left]
        y = batch['input_ids'][:, 1:].reshape(-1)[valid][:left].to(h0.device)
        hidden.append(h.detach()); targets.append(y.detach()); left -= len(y)
        if left == 0:
            break
    if left:
        raise ValueError('not enough supervised train positions for max_train_contexts')
    return torch.cat(hidden), torch.cat(targets)


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if Path(config['output_dir']).exists():
        raise FileExistsError('choose a new output_dir; historical outputs are protected')
    if args.actor_checkpoint and args.stage != 'profile':
        raise ValueError('--actor-checkpoint is only for profile, never resume')
    if not torch.cuda.is_available():
        raise RuntimeError('real Mixtral experiments require CUDA; use CPU tiny-model unit tests for code checks')
    # Reject split leakage/oversized source files before allocating model weights.
    read_documents(config)
    torch.manual_seed(config['seed'])
    with precision_scope(config['allow_tf32']):
        load_started = time.perf_counter()
        model, tokenizer, adapter, model_info = load_local_model(config)
        load_seconds = time.perf_counter() - load_started
        stage_started = time.perf_counter()
        data, data_identity = prepare_data(config, tokenizer)
        output = fresh_output(config['output_dir'])
        identity = {'model': model_info['model'], 'tokenizer': model_info['tokenizer'],
                    'train': data_identity['train'], 'selection': data_identity['selection'],
                    'calibration': data_identity['calibration'], 'code': code_identity(),
                    'numerical_contract': {**{key: config[key] for key in ('dtype', 'allow_tf32', 'experts_implementation', 'sequence_length', 'sequences', 'max_train_contexts')}, 'attention': 'sdpa', 'head_chunk': 128, 'runtime_versions': {name: version(name) for name in ('torch', 'transformers', 'accelerate', 'safetensors')}}}
        report = {'schema': 'mixtral-experiment-report-v1', 'stage': args.stage, 'status': 'running',
                  'model_load_wall_seconds': load_seconds, 'config': config, 'config_sha256': canonical_hash(config), 'identity': identity,
                  'model': model_info, 'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'packages': {name: version(name) for name in ('torch', 'transformers', 'accelerate', 'safetensors')},
                      'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'gpus': {str(i): {'name': torch.cuda.get_device_name(i),
                               'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory} for i in model_info['cuda_devices']}},
                  'limitations': ['no held-out confirmation', 'no full-backbone training', 'no model-scale causal claim']}
        atomic_json(output/'report.json', report)
        try:
            report['native_equivalence'] = validate_native(adapter, data['calibration'], config)
            atomic_json(output/'report.json', report)
            if not report['native_equivalence']['eligible']:
                raise RuntimeError('native equivalence screen failed; stop before profiling/training')
            report['native_selection'] = evaluate(adapter, data['selection'], native=True)
            report['controlled_original_selection'] = evaluate(adapter, data['selection'])
            if args.stage == 'profile':
                def native(row):
                    batch = batch_row(row, input_device(model))
                    return model(**batch, use_cache=False, output_router_logits=False).logits.to(input_device(model))
                def controlled(row):
                    return adapter.forward(**batch_row(row, input_device(model)), max_calls=1).logits.to(input_device(model))
                functions = {'native': native, 'controlled_original_route_r1': controlled}
                if args.actor_checkpoint:
                    from mixtral_lab.training import load_actor_checkpoint
                    actor = load_actor_checkpoint(args.actor_checkpoint, identity, adapter.device)
                    report['learned_selection'] = evaluate(adapter, data['selection'], actor)
                    functions['learned_route_r1'] = lambda row: adapter.forward(**batch_row(row, input_device(model)), policy=actor, max_calls=1).logits.to(input_device(model))
                report['timing'] = measure_calls(functions, data['selection'], model_info['cuda_devices'], config)
                report['timing']['output_device'] = str(input_device(model))
            elif args.stage == 'train':
                from mixtral_lab.training import RouterTrainingConfig, train_router
                h0, targets = collect_train_contexts(adapter, data['train'], config['max_train_contexts'])
                training_config = RouterTrainingConfig(**config['training'])
                _, report['training'] = train_router(adapter, h0, targets, lambda actor: evaluate(adapter, data['selection'], actor),
                    training_config, identity, output/'router')
                from mixtral_lab.training import load_actor_checkpoint
                best = load_actor_checkpoint(output/'router'/'checkpoint_best.pt', identity, adapter.device)
                report['selected_best_development'] = evaluate(adapter, data['selection'], best)
            report['status'] = 'complete'
            report['stage_wall_seconds_excluding_model_load'] = time.perf_counter() - stage_started
            atomic_json(output/'report.json', report)
        except Exception as exc:
            report['status'], report['error'] = 'failed', str(exc)
            atomic_json(output/'report.json', report)
            raise
    print(json.dumps({'report': str(output/'report.json'), 'status': 'complete'}))


if __name__ == '__main__':
    main()
