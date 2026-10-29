"""Bounded, opt-in 100M runtime diagnosis; no training or backend promotion.

Run with python -m experiments.rl_lm_diagnose_runtime --help. Timings exclude
CE and trace instrumentation. Numerical validation is selection-only, not a
held-out quality claim. Component fixtures are isolated and non-additive.
"""
from __future__ import annotations
import argparse
import hashlib
from dataclasses import asdict
import json
import math
import subprocess
from pathlib import Path

import torch
from eval.counterfactual import cache_last_ffn_input
from experiments.rl_lm_evaluate import verified_evaluation_scope, _load_actor
from experiments.rl_lm_profile import measure, gpu_metadata, check_exclusive, variant_schedule
from lm_config import LMExperimentConfig
from model import LoopMoE
from rl_lm.cli import load_backbone, resolve_config
from rl_lm.executor import make_executor
from rl_lm.inference import CanonicalTeacherPolicy, forward_with_last_ffn_policy, greedy_lowest_id
from rl_lm.readout import production_logits
from rl_lm.runtime_diagnostics import (Variant, CachedBF16Head, runtime_scope, parity_metrics,
                                       positive_int, parse_chunks, fresh_output, preserve_precision)
from train.common import autocast_context
from train.recovery_common import atomic_json, sha256_file


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--corpus-dir')
    p.add_argument('--tokenizer')
    p.add_argument('--device', default='cuda')
    p.add_argument('--physical-gpu-index', type=int)
    p.add_argument('--dense-config', required=True)
    p.add_argument('--dense-checkpoint', required=True)
    p.add_argument('--actor', help='optional selected actor matching --config stage/provenance')
    p.add_argument('--output-dir', required=True, help='must not already exist')
    p.add_argument('--sequences', type=int, default=2)
    p.add_argument('--batches', default='1')
    p.add_argument('--head-chunks', default='64,full')
    p.add_argument('--cached-bf16-head', action='store_true', help='add isolated cached-weight candidates')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--measurements', type=int, default=3)
    p.add_argument('--nll-tolerance', type=float, default=1e-4,
                   help='absolute mean NLL deviation limit, not a statistical equivalence claim')
    p.add_argument('--ce-mean-abs-tolerance', type=float, default=1e-4)
    p.add_argument('--trace-variant', help='trace only this exact variant name; requires --trace')
    p.add_argument('--trace', action='store_true', help='one separate trace per E2E variant/batch, never timed')
    return p


def validate_args(args):
    for name in ('sequences', 'repeats', 'measurements'):
        positive_int(getattr(args, name), name)
    if isinstance(args.warmup, bool) or not isinstance(args.warmup, int) or args.warmup < 0:
        raise ValueError('warmup must be nonnegative integer')
    batches = parse_chunks(args.batches)
    if 'full' in batches:
        raise ValueError('batches must be integers')
    batches = [int(v) for v in batches]
    if any(args.sequences % batch for batch in batches):
        raise ValueError('sequences must be divisible by every batch size')
    chunks = parse_chunks(args.head_chunks)
    if isinstance(args.ce_mean_abs_tolerance, bool) or not math.isfinite(args.ce_mean_abs_tolerance) or args.ce_mean_abs_tolerance < 0:
        raise ValueError('CE mean absolute tolerance must be finite and nonnegative')
    if isinstance(args.nll_tolerance, bool) or not math.isfinite(args.nll_tolerance) or args.nll_tolerance < 0:
        raise ValueError('nll tolerance must be finite and nonnegative')
    return batches, chunks


def build_variants(chunks, *, cached, learned, max_calls):
    variants = [Variant('native_legacy', 'native'), Variant('native_all_layers_merged', 'native', 'merged'),
                Variant('dense', 'dense')]
    policies = [('canonical_r1', 1)] + ([('learned', max_calls)] if learned else [])
    for policy, calls in policies:
        for prefix in ('legacy', 'merged'):
            for chunk in chunks:
                for cache in ([False, True] if cached else [False]):
                    name = f'{policy}_{prefix}_head{chunk}_{"cached_bf16" if cache else "autocast"}'
                    variants.append(Variant(name, 'online', prefix, chunk, cache, policy, calls))
    return variants


def reference_variant(variant):
    if variant.kind == 'online':
        return Variant('reference_matched_online_head64', 'online', variant.prefix, '64', False,
                       variant.policy, variant.max_calls, variant.terminal)
    return Variant('reference_' + variant.kind, variant.kind)


@torch.no_grad()
def run_output(variant, model, dense, policies, batch, dtype, expert_chunk, *, trace=False):
    ids, segments, _ = batch
    target = dense if variant.kind == 'dense' else model
    if variant.kind in ('native', 'dense'):
        with autocast_context(ids.device, dtype):
            return target(ids, None, depth=1, segment_ids=segments)
    chunk = ids.numel() if variant.head_chunk == 'full' else int(variant.head_chunk)
    return forward_with_last_ffn_policy(model, policies[variant.policy], ids, segments,
        max_calls=variant.max_calls, backend=variant.terminal, compute_dtype=dtype,
        chunk=expert_chunk, head_chunk=chunk, return_trace=trace)


@torch.no_grad()
def validate_variant(variant, model, dense, policies, batches, dtype, expert_chunk, cached, tolerance, reference=None, ce_mean_abs_tolerance=None):
    reference = reference_variant(variant) if reference is None else reference
    records = []
    target = dense if variant.kind == 'dense' else model
    for batch in batches:
        with runtime_scope(target, reference):
            a = run_output(reference, model, dense, policies, batch, dtype, expert_chunk, trace=True)
        with runtime_scope(target, variant, cached):
            b = run_output(variant, model, dense, policies, batch, dtype, expert_chunk, trace=True)
        records.append(parity_metrics(a, b, batch[2], nll_tolerance=tolerance,
                                      require_actions=variant.kind == 'online', ce_mean_abs_tolerance=ce_mean_abs_tolerance))
        del a, b
    tokens = sum(r['tokens'] for r in records)
    return {'reference': reference.__dict__, 'groups': records,
            'tokens': tokens, 'delta_nll': sum(r['delta_nll']*r['tokens'] for r in records)/tokens,
            'eligible': all(r['eligible'] for r in records),
            'scope': 'all requested fixed selection inputs; no confidence interval or held-out confirmation'}


@torch.no_grad()
def stage_functions(model, policy, batch, dtype, expert_chunk, cached, max_calls=1):
    """R0 component microbenchmarks on a common legacy-prefix fixture."""
    ids, segments, _ = batch
    with autocast_context(ids.device, dtype):
        h = cache_last_ffn_input(model, ids, segments).float().reshape(-1, model.config.hidden_size)
    executor = make_executor(model, 'sparse_active_merged_v3', chunk=expert_chunk, compute_dtype=dtype)
    rounds = torch.zeros(len(h), dtype=torch.long, device=h.device)
    previous = torch.full((len(h), 2), -1, dtype=torch.long, device=h.device)
    observation = executor.observe(h, h, rounds, previous, max_calls=max_calls)
    stop = rounds >= 1
    actions = greedy_lowest_id(policy(observation.features, observation.pair_reference, stop))
    active = torch.ones(len(h), dtype=torch.bool, device=h.device)
    def prefix():
        with autocast_context(ids.device, dtype):
            return cache_last_ffn_input(model, ids, segments)
    return {
        'prefix_legacy': prefix,
        'observation_including_chunked_gate': lambda: executor.observe(h, h, rounds, previous, max_calls=max_calls),
        'actor_and_argmax_precomputed_observation': lambda: greedy_lowest_id(policy(observation.features, observation.pair_reference, stop)),
        'dispatch_v3_precomputed_actions': lambda: executor.step_active(h, observation, actions, active),
        'head64_autocast': lambda: production_logits(model, h, chunk=64, compute_dtype=dtype),
        'head_full_autocast': lambda: production_logits(model, h, chunk=len(h), compute_dtype=dtype),
        **({'head64_cached_bf16': lambda: cached(model, h, chunk=64, compute_dtype=dtype)} if cached else {}),
    }


def source_identity():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'), 'tree': git('rev-parse', 'HEAD^{tree}'),
            'workspace_status': git('status', '--porcelain'),
            'workspace_diff_sha256': hashlib.sha256(subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--'])).hexdigest(),
            'diagnostic_source_sha256': sha256_file(__file__),
            'helper_source_sha256': sha256_file(root / 'rl_lm/runtime_diagnostics.py')}


@preserve_precision()
@torch.no_grad()
def main(argv=None):
    args = parser().parse_args(argv)
    batch_sizes, chunks = validate_args(args)
    # Fail before any CUDA allocation or file write when destination exists.
    if Path(args.output_dir).exists():
        raise FileExistsError('choose a new --output-dir')
    config = resolve_config(args.config, checkpoint=args.checkpoint, corpus_dir=args.corpus_dir,
                            tokenizer=args.tokenizer, device=args.device)
    device = torch.device(config.get('runtime.device'))
    if device.type != 'cuda':
        raise RuntimeError('formal diagnostic measurements require CUDA')
    dtype = config.get('environment.frozen_compute_dtype')
    if args.cached_bf16_head and dtype != 'bfloat16':
        raise ValueError('cached BF16 candidates require bfloat16 config')
    split = config.get('data.selection_split')
    if split.casefold() in ('newtest', config.get('data.confirmation_split').casefold()):
        raise ValueError('diagnostics must not use confirmation data')
    scope, dataset, _ = verified_evaluation_scope(config.get('data.corpus_dir'), split, args.sequences)
    allow_tf32 = config.get('runtime.allow_tf32')
    torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.cuda.set_device(device)
    check_exclusive(device, args.physical_gpu_index)
    model, _ = load_backbone(config, device)
    dense_config = LMExperimentConfig.load(args.dense_config)
    if dense_config.model.moe_layers:
        raise ValueError('dense control must not contain MoE layers')
    for field in ('layers', 'hidden_size', 'attention_heads', 'vocab_size', 'max_position_embeddings'):
        if getattr(dense_config.model, field) != getattr(model.config, field):
            raise ValueError(f'dense/MoE shared architecture mismatch: {field}')
    dense = LoopMoE(dense_config.model).to(device).eval()
    dense.load_state_dict(torch.load(args.dense_checkpoint, map_location=device, weights_only=False)['model'])
    dense.requires_grad_(False)
    if dense.config.vocab_size != model.config.vocab_size:
        raise ValueError('dense and MoE vocabulary sizes differ')
    policies = {'canonical_r1': CanonicalTeacherPolicy()}
    if args.actor:
        policies['learned'] = _load_actor(config, Path(args.actor), device, model=model)
    cached = CachedBF16Head(model) if args.cached_bf16_head else None
    expert_chunk = int(config.get('environment.expert_chunk'))
    variants = build_variants(chunks, cached=cached is not None, learned=args.actor is not None,
                              max_calls=int(config.get('environment.max_calls')))
    if args.trace_variant and (not args.trace or args.trace_variant not in [v.name for v in variants]):
        raise ValueError('--trace-variant requires --trace and an existing variant name')
    output = fresh_output(args.output_dir)
    report = {'schema': 'rl_lm_runtime_diagnostic_v1', 'status': 'running',
              'source': source_identity(), 'arguments': vars(args), 'resolved_config': config.to_dict() if hasattr(config, 'to_dict') else config.values,
              'models': {'moe': asdict(model.config), 'dense': asdict(dense.config),
                         'moe_parameters': sum(p.numel() for p in model.parameters()),
                         'dense_parameters': sum(p.numel() for p in dense.parameters())},
              'precision': {'float32_matmul_precision': torch.get_float32_matmul_precision(),
                            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32},
              'dense_tokenizer_note': 'same verified input token IDs and vocabulary size; checkpoint training tokenizer identity must be established by operator',
              'scope': scope, 'gpu': gpu_metadata(device, args.physical_gpu_index),
              'checkpoint_sha256': sha256_file(config.get('backbone.checkpoint_path')),
              'dense_checkpoint_sha256': sha256_file(args.dense_checkpoint),
              'actor_sha256': sha256_file(args.actor) if args.actor else None,
              'planned_e2e_timed_calls': len(batch_sizes)*len(variants)*args.repeats*args.measurements,
              'variant_count': len(variants), 'cached_head_persistent_bytes': cached.bytes if cached else 0,
              'timing_scope': 'input to all logits, no CE, no trace instrumentation; runtime setup/verification excluded',
              'memory_scope': 'absolute process allocator levels; shared models/cache stay live in all variants',
              'speed_claim': 'not established; eligibility is local numerical screening only', 'workloads': {}}
    atomic_json(output / 'diagnostic_report.json', report)
    try:
        for batch_size in batch_sizes:
            batches = []
            for start in range(0, args.sequences, batch_size):
                rows = [dataset[i] for i in range(start, start+batch_size)]
                batches.append(tuple(torch.stack([torch.as_tensor(row[key].copy()) for row in rows]).to(device)
                                     for key in ('input_ids', 'segment_ids', 'labels')))
            tokens = batches[0][0].numel()
            name = f'batch{batch_size}_s{batches[0][0].shape[1]}'
            result = {'verification': {}, 'end_to_end': {}, 'components': {}, 'combined_runtime_verification': {},
                      'component_scope': 'first selection batch, legacy prefix, R0 only; non-additive isolated microbenchmarks',
                      'component_policy': 'learned' if args.actor else 'canonical_r1',
                      'component_max_calls': int(config.get('environment.max_calls')) if args.actor else 1}
            report['workloads'][name] = result
            for variant in variants:
                result['verification'][variant.name] = validate_variant(variant, model, dense, policies, batches,
                                            dtype, expert_chunk, cached, args.nll_tolerance, ce_mean_abs_tolerance=args.ce_mean_abs_tolerance)
                if variant.kind == 'online':
                    legacy = Variant('reference_legacy_v1_head64', 'online', 'legacy', '64', False,
                                     variant.policy, variant.max_calls, 'sparse_merged_v1')
                    result['combined_runtime_verification'][variant.name] = validate_variant(
                        variant, model, dense, policies, batches, dtype, expert_chunk, cached,
                        args.nll_tolerance, reference=legacy, ce_mean_abs_tolerance=args.ce_mean_abs_tolerance)
            # All candidates are measured for diagnosis, but failed screening is explicit.
            for repeat, order in enumerate(variant_schedule([v.name for v in variants], args.repeats)):
                for variant_name in order:
                    variant = next(v for v in variants if v.name == variant_name)
                    target = dense if variant.kind == 'dense' else model
                    counter = [0]
                    def call():
                        index = counter[0] % len(batches)
                        counter[0] += 1
                        return run_output(variant, model, dense, policies, batches[index], dtype, expert_chunk)
                    with runtime_scope(target, variant, cached) as (identity, verify):
                        if repeat == 0:
                            for _ in range(args.warmup):
                                verify(); call()
                        counter[0] = 0
                        timing = measure(call, warmup=0, repeats=1, measurements=args.measurements,
                                         tokens=tokens, before_call=verify)
                    record = result['end_to_end'].setdefault(variant.name, {'identity': identity, 'repeats': [],
                                  'numerically_eligible': result['verification'][variant.name]['eligible'] and result['combined_runtime_verification'].get(variant.name, {'eligible': True})['eligible']})
                    record['repeats'].append(timing)
            with runtime_scope(model, Variant('components', 'native')):
                functions = stage_functions(model, policies.get('learned', policies['canonical_r1']), batches[0], dtype, expert_chunk, cached, max_calls=int(config.get('environment.max_calls')) if args.actor else 1)
                for stage, fn in functions.items():
                    result['components'][stage] = measure(fn, warmup=args.warmup, repeats=args.repeats,
                                                        measurements=args.measurements, tokens=tokens)
            if args.trace:
                result['traces'] = {}
                for variant in variants:
                    if args.trace_variant and variant.name != args.trace_variant:
                        continue
                    target = dense if variant.kind == 'dense' else model
                    filename = f'{name}_{variant.name}.trace.json'
                    with runtime_scope(target, variant, cached):
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA],
                                                    record_shapes=True, profile_memory=True) as prof:
                            run_output(variant, model, dense, policies, batches[0], dtype, expert_chunk)
                            torch.cuda.synchronize()
                    prof.export_chrome_trace(str(output / filename))
                    result['traces'][variant.name] = filename
            atomic_json(output / 'diagnostic_report.json', report)
        report['status'] = 'complete'
        atomic_json(output / 'diagnostic_report.json', report)
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        atomic_json(output / 'diagnostic_report.json', report)
        raise
    print(json.dumps({'report': str(output / 'diagnostic_report.json'), 'status': 'complete'}))


if __name__ == '__main__':
    main()
