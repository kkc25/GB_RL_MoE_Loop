"""Train-only cache/full-document numerical diagnosis, never a training gate.

At the terminal last FFN positions are pointwise: changing another position
cannot cause attention or recurrent-state feedback. Different expert/head batch
shapes can still change arithmetic. A tolerance screen is not an equivalence
proof, a fit result, development selection, or deployment approval.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .adapter import action_to_pair, pair_to_action
from .adapter_fit import AdapterFitConfig, _fixed_chunks, _tensor_hash as adapter_tensor_hash
from .fit_protocol import finite_tree
from .full_action_fit import _tensor_hash as router_tensor_hash
from .protocol import batch_row, canonical_hash, integer, sha256_file
from .sampling import collect_stratified_contexts, plan_stratified_contexts
from .second_round import SecondRoundResidual, differentiable_readout
from .training import ResidualPairRouter

SCHEMA = 'mixtral-cache-online-bridge-v1'
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


def _index_select_own_device(tensor, indices):
    return tensor.index_select(0, indices.to(tensor.device))


def safe_path(path, name='path'):
    """Reject traversal spellings rather than silently normalizing report paths."""
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError(f'{name} must be a nonempty local path')
    value = Path(path)
    if '..' in value.parts or '\\' in str(path) or '\x00' in str(path):
        raise ValueError(f'{name} contains path traversal or an invalid separator')
    return value


def verify_historical_sources(identity, root):
    """Verify every originally bound file, not today's glob or git commit.

    Adding diagnostics or changing an unrelated commit must not rewrite a
    historical artifact's identity. A changed/missing historical source still
    fails closed. Git's old commit is retained as provenance, not verified as
    the current checkout commit; byte hashes are the source evidence.
    """
    if not isinstance(identity, dict) or not isinstance(identity.get('sources'), dict) or not identity['sources']:
        raise ValueError('historical source identity missing')
    root = Path(root).resolve()
    if len(identity['sources']) > 256:
        raise ValueError('historical source list exceeds bound')
    for name, digest in identity['sources'].items():
        relative = safe_path(name, 'historical source')
        if relative.is_absolute() or relative.suffix != '.py':
            raise ValueError('historical source must be a relative Python file')
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('historical source escapes repository')
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f'historical source checksum mismatch: {name}')
    return copy.deepcopy(identity)


def prepare_frozen_train(config, tokenizer, frozen_data):
    """Hash all already-bound inputs, tokenize TRAIN only; never evaluate others."""
    if set(frozen_data) != {'train', 'selection', 'calibration'}:
        raise ValueError('frozen data roles mismatch')
    for role, identity in frozen_data.items():
        path = safe_path(config['data_files'][role], f'{role} input')
        if str(path.resolve()) != identity.get('path') or sha256_file(path) != identity.get('file_sha256'):
            raise ValueError(f'frozen {role} data identity mismatch')
    path = safe_path(config['data_files']['train'])
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError('train input exceeds bounded JSONL size')
    documents = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(documents) > config['max_documents_per_split']:
        raise ValueError('train document count exceeds bound')
    expected = frozen_data['train']
    parsed = [{'id': r['id'], 'text_sha256': hashlib.sha256(' '.join(r['text'].split()).encode()).hexdigest()}
              for r in documents]
    if parsed != expected.get('documents'):
        raise ValueError('frozen train document identity mismatch')
    selected = documents[:config['sequences']['train']]
    if [r['id'] for r in selected] != expected.get('selected_document_ids'):
        raise ValueError('frozen train selected document order mismatch')
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError('tokenizer needs pad or eos token')
    rows = []
    for row in selected:
        ids = tokenizer.encode(row['text'], add_special_tokens=True, truncation=True,
                               max_length=config['sequence_length'])
        if not 2 <= len(ids) <= config['sequence_length']:
            raise ValueError('invalid bounded train tokenization')
        valid = len(ids)
        ids += [pad] * (config['sequence_length'] - valid)
        rows.append({'id': row['id'], 'input_ids': torch.tensor(ids, dtype=torch.long),
                     'attention_mask': torch.tensor([1]*valid + [0]*(len(ids)-valid), dtype=torch.long)})
    token_identity = [{'id': r['id'], 'input_ids': r['input_ids'].tolist(),
                       'attention_mask': r['attention_mask'].tolist()} for r in rows]
    tokens = sum(int((r['attention_mask'][:-1] * r['attention_mask'][1:]).sum()) for r in rows)
    if canonical_hash(token_identity) != expected.get('tokenized_sha256') or tokens != expected.get('supervised_tokens'):
        raise ValueError('frozen train tokenizer/label identity mismatch')
    return rows, copy.deepcopy(frozen_data)


def _finite(value, name):
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise ValueError(f'nonfinite or malformed {name}')
    return value


def bridge_tolerances(config, overrides=None):
    values = {'logit_max_abs': config['logit_max_abs_tolerance'],
              'ce_mean_abs': config['ce_mean_abs_tolerance'],
              'ce_max_abs': config['nll_tolerance'],
              'hidden_max_abs': 1e-6, 'feature_max_abs': 1e-6}
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError('bridge tolerance overrides must be an object')
    if overrides:
        if set(overrides) - set(values):
            raise ValueError('unknown bridge tolerance')
        values.update(overrides)
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid bridge tolerance: {name}')
    return values


class _Differences:
    def __init__(self):
        self.values = {}

    def add(self, name, a, b):
        if a.shape != b.shape:
            raise ValueError(f'{name} comparison shape mismatch')
        _finite(a, name); _finite(b, name)
        if not a.numel():
            return
        value = self.values.setdefault(name, {'elements': 0, 'sum_abs': 0., 'max_abs': 0.})
        # Avoid vocabulary-sized FP32/FP64 CPU copies for full document logits.
        flat_a, flat_b = a.detach().reshape(-1), b.detach().reshape(-1)
        for start in range(0, a.numel(), 1024 * 1024):
            delta = (flat_a[start:start+1024*1024].float().cpu()
                     - flat_b[start:start+1024*1024].float().cpu()).abs().double()
            value['elements'] += delta.numel()
            value['sum_abs'] += float(delta.sum())
            value['max_abs'] = max(value['max_abs'], float(delta.max()))

    def report(self, tolerances):
        result = {}
        for name, value in self.values.items():
            metric = name.rsplit('.', 1)[-1]
            threshold = tolerances[{'logits': 'logit_max_abs', 'hidden': 'hidden_max_abs',
                                    'h0': 'hidden_max_abs', 'h1': 'hidden_max_abs',
                                    'features': 'feature_max_abs', 'ce': 'ce_max_abs'}[metric]]
            mean = value['sum_abs'] / value['elements']
            passed = value['max_abs'] <= threshold
            entry = {'elements': value['elements'], 'mean_abs': mean,
                     'max_abs': value['max_abs'], 'max_abs_tolerance': threshold}
            if metric == 'ce':
                entry['mean_abs_tolerance'] = tolerances['ce_mean_abs']
                passed = passed and mean <= tolerances['ce_mean_abs']
            entry['within_declared_tolerance'] = passed
            result[name] = entry
        return result


def _ce(logits, targets, *, chunk=None):
    _finite(logits, 'logits')
    chunk = len(logits) if chunk is None else chunk
    values = torch.cat([F.cross_entropy(logits[start:start+chunk].float(),
                        targets[start:start+chunk].to(logits.device), reduction='none').detach().cpu()
                        for start in range(0, len(logits), chunk)])
    return _finite(values, 'CE')


def _native_document(adapter, inputs):
    """Capture the genuine native pre-final-norm hidden with a removable hook."""
    hidden = []
    def capture(module, args):
        hidden.append(args[0].detach())
    handle = adapter.model.model.norm.register_forward_pre_hook(capture)
    try:
        output = adapter.model(**inputs, use_cache=False, output_router_logits=False)
    finally:
        handle.remove()
    if len(hidden) != 1:
        raise ValueError('native final hidden hook did not run exactly once')
    return hidden[0], output.logits


def forced_cached_policy(policy, positions, actions):
    """Force only sampled actions; all other full-document actions stay online."""
    if (not isinstance(positions, torch.Tensor) or positions.dtype != torch.long
            or positions.ndim != 1 or bool((positions < 0).any())
            or len(torch.unique(positions)) != len(positions)
            or not isinstance(actions, torch.Tensor) or actions.shape != positions.shape):
        raise ValueError('forced cached positions/actions must be distinct nonnegative int64 vectors')
    action_to_pair(actions)
    def forced(observation):
        output = policy(observation).clone()
        indices = positions.to(output.device)
        if output.ndim != 1 or len(indices) != len(actions) or bool((indices >= len(output)).any()):
            raise ValueError('forced cached action/position mismatch')
        output[indices] = actions.to(output.device)
        action_to_pair(output)
        return output
    return forced


def _population_metrics(losses, selected, valid_positions):
    output = {}
    selected_set = set(selected)
    for name, indices in (('selected', selected), ('unselected', [p for p in valid_positions if p not in selected_set])):
        values = losses[torch.tensor(indices, dtype=torch.long)] if indices else torch.empty(0)
        output[name] = {'tokens': len(indices), 'nll': float(values.double().mean()) if len(indices) else None}
    return output


def _aggregate(documents):
    result = {}
    for population in ('selected', 'unselected'):
        variants = sorted({name for doc in documents for name in doc['online']})
        result[population] = {}
        for variant in variants:
            rows = [doc['online'][variant][population] for doc in documents]
            count = sum(r['tokens'] for r in rows)
            result[population][variant] = {'tokens': count,
                'nll': sum(r['nll']*r['tokens'] for r in rows if r['tokens']) / count if count else None}
        native = result[population]['native']['nll']
        for variant, value in result[population].items():
            value['minus_native_nll'] = value['nll'] - native if native is not None else None
    return result


@torch.no_grad()
def diagnose_bridge(adapter, train_rows, candidate, config, *, manifest=None,
                    checkpoint_metadata=None, tolerances=None):
    """Fixed-work diagnosis on exact TRAIN sampling, with no optimizer/evaluator.

    Only one document's full logits are retained at a time. Cached adapter probe
    chunks that span documents are recomputed in their original global order and
    padded shape, never compacted to the selected document. Report size is O(train
    contexts + documents), not O(vocabulary). No optional selection callback exists.
    """
    kind = config['fit']['kind']
    if kind not in ('router', 'adapter'):
        raise ValueError('bridge kind must be router or adapter')
    rows = list(train_rows)
    integer(config['max_train_contexts'], 'bridge contexts', 1, 4096)
    if not 1 <= len(rows) <= 4096 or any(len(row['input_ids']) > 4096 for row in rows):
        raise ValueError('bridge train document/length budget exceeded')
    if any(row.get('role', 'train') != 'train' for row in rows):
        raise ValueError('bridge accepts train rows only')
    expected = plan_stratified_contexts(rows, config['max_train_contexts'], config['seed'], 'train')
    if manifest is not None and manifest != expected:
        raise ValueError('bridge manifest differs from exact train sampling')
    manifest = expected
    tolerances = bridge_tolerances(config, tolerances)
    if any(p.requires_grad or p.grad is not None for p in adapter.model.parameters()) or any(m.training for m in adapter.model.modules()):
        raise ValueError('bridge requires frozen eval backbone with no gradients')
    candidate.eval()
    if any(not bool(torch.isfinite(p).all()) for p in candidate.parameters()):
        raise ValueError('nonfinite candidate weights')
    h0, labels, captured = collect_stratified_contexts(adapter, rows, config['max_train_contexts'], config['seed'], 'train')
    if captured != manifest:
        raise ValueError('captured train sampling mismatch')
    _finite(h0, 'cached h0')
    if h0.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('bridge requires FP32 or BF16 cached hidden states')
    metadata = checkpoint_metadata or {}
    groups = {}
    for index, context in enumerate(manifest['contexts']):
        groups.setdefault(context['document_id'], []).append((index, context['token_position']))
    cache, binding = {}, {'h0': router_tensor_hash(h0), 'targets': router_tensor_hash(labels)}
    for key in ('h0', 'targets'):
        if key in metadata.get('historical_cached_inputs', {}) and metadata['historical_cached_inputs'][key] != binding[key]:
            raise ValueError(f'reconstructed cached router {key} checksum mismatch')
    work = {'cached_prefix_documents': len(groups), 'online_documents': 0,
            'cached_readout_calls': 0, 'cached_readout_rows_including_padding': 0,
            'online_variant_calls': 0, 'cached_native_chunks': 0, 'cached_score_pairs_calls': 0,
            'cached_terminal_ffn_calls': 0, 'cached_terminal_ffn_rows_including_padding': 0,
            'prefix_capture_calls': len(groups), 'native_model_calls': 0,
            'online_terminal_ffn_calls': 0, 'online_readout_calls': 0}
    if kind == 'router':
        zero = type(candidate)(candidate.feature_dim, candidate.epsilon, device=adapter.device).eval()
        features = torch.cat([_finite(adapter.features(h0[i:i+1]), 'cached features') for i in range(len(h0))])
        pairs = torch.cat([adapter.native_pair(h0[i:i+1]) for i in range(len(h0))])
        cache.update(features=features, pairs=pairs,
            native_actions=pair_to_action(pairs), zero_actions=zero.greedy(features, pairs),
            # Original fixed table probes evaluate actor on all cached rows.
            learned_actions=candidate.greedy(features, pairs))
        binding.update(features=router_tensor_hash(features), native_pairs=router_tensor_hash(pairs))
        prior = metadata.get('score_table_hashes', {})
        for key in ('features', 'native_pairs'):
            if key in prior and prior[key] != binding[key]:
                raise ValueError(f'reconstructed cached router {key} checksum mismatch')
    else:
        settings = candidate.config
        settings.validate()
        if asdict(settings) != config['fit']['training']:
            raise ValueError('candidate adapter configuration differs from declared scientific settings')
        if settings.max_train_contexts != config['max_train_contexts'] or settings.seed != config['seed']:
            raise ValueError('candidate adapter sampling configuration mismatch')
        zero = SecondRoundResidual(adapter.hidden_size, settings, adapter.device).eval()
        native_chunks = []
        for h, y, valid in _fixed_chunks(h0, labels, settings.probe_chunk_contexts):
            native_chunks.append(_finite(adapter.step_pairs(h, adapter.native_pair(h)), 'cached h1')[:valid])
            work['cached_native_chunks'] += 1
            work['cached_terminal_ffn_calls'] += 1
            work['cached_terminal_ffn_rows_including_padding'] += len(h)
        cache['h1'] = torch.cat(native_chunks)
        binding.update(train_inputs_sha256=adapter_tensor_hash({'h0': h0, 'targets': labels}),
                       cached_native_sha256=adapter_tensor_hash({'h1': cache['h1']}))
        for name in ('train_inputs_sha256', 'cached_native_sha256'):
            if name in metadata and metadata[name] != binding[name]:
                raise ValueError(f'reconstructed adapter {name} checksum mismatch')
    diffs = _Differences()
    records = []
    cached_losses = {name: torch.full((len(h0),), float('nan')) for name in ('native', 'zero', 'learned')}
    agreements = {}
    context_records = [dict(context, context_index=i) for i, context in enumerate(manifest['contexts'])]
    input_device = adapter.model.get_input_embeddings().weight.device
    for row in rows:
        selected = groups.get(row['id'])
        if not selected:
            continue  # unselected positions means SAME sampled documents, not extra documents
        work['online_documents'] += 1
        global_ids = torch.tensor([i for i, _ in selected], dtype=torch.long)
        positions = torch.tensor([p for _, p in selected], dtype=torch.long)
        inputs = batch_row(row, input_device)
        work['prefix_capture_calls'] += 1
        online_h0 = _finite(adapter.capture(**inputs), 'online h0')
        if online_h0.shape != (1, len(row['input_ids']), adapter.hidden_size):
            raise ValueError('online prefix shape does not match original padded document')
        flat_h0 = online_h0.reshape(-1, adapter.hidden_size)
        diffs.add('cached_vs_online.h0', h0[global_ids.to(h0.device)], flat_h0[positions.to(flat_h0.device)])
        valid = torch.nonzero(row['attention_mask'][:-1].bool() & row['attention_mask'][1:].bool()).flatten().tolist()
        if not set(positions.tolist()).issubset(valid):
            raise ValueError('selected bridge positions are masked or have no next-token label')
        doc = {'document_id': row['id'], 'padded_length': len(row['input_ids']),
               'selected_positions': positions.tolist(), 'online': {}, 'paired_deltas': {}}
        if kind == 'router':
            observation = adapter.observe(flat_h0)
            diffs.add('cached_vs_online.features', features[global_ids.to(features.device)],
                      observation.features[positions.to(observation.features.device)])
            online_actions = {'native': pair_to_action(observation.native_pairs),
                              'zero': zero(observation), 'learned': candidate(observation)}
            for name, actions in online_actions.items():
                cached = cache[name + '_actions'][global_ids.to(h0.device)]
                current = actions[positions.to(actions.device)]
                entry = agreements.setdefault(name, {'agree': 0, 'count': 0})
                entry['agree'] += int((cached == current).sum()); entry['count'] += len(cached)
                for j, index in enumerate(global_ids.tolist()):
                    context_records[index].update({f'cached_{name}_action': int(cached[j]),
                                                   f'online_{name}_action': int(current[j])})
            variants = ('native', 'controlled', 'zero', 'learned', 'forced_cached')
        else:
            work['online_terminal_ffn_calls'] += 1
            online_h1 = _finite(adapter.step_pairs(flat_h0, adapter.native_pair(flat_h0)), 'online h1')
            diffs.add('cached_vs_online.h1', cache['h1'][global_ids.to(h0.device)], online_h1[positions.to(h0.device)])
            variants = ('native', 'controlled', 'zero', 'learned')
        native_ce = native_logits = native_hidden = None
        for name in variants:
            work['online_variant_calls'] += 1
            if name == 'native':
                work['native_model_calls'] += 1
                work['online_terminal_ffn_calls'] += 1
                work['online_readout_calls'] += 1
                hidden, logits = _native_document(adapter, inputs)
            elif kind == 'router':
                policy = None if name == 'controlled' else zero if name == 'zero' else candidate
                if name == 'forced_cached':
                    policy = forced_cached_policy(candidate, positions, cache['learned_actions'][global_ids.to(h0.device)])
                work['prefix_capture_calls'] += 1
                work['online_terminal_ffn_calls'] += 1
                work['online_readout_calls'] += 1
                output = adapter.forward(**inputs, policy=policy, max_calls=1)
                hidden, logits = output.final_hidden, output.logits
                if name in ('zero', 'learned', 'forced_cached'):
                    measured_actions = output.actions.reshape(-1)[positions.to(output.actions.device)]
                    expected_actions = (cache['learned_actions'][global_ids.to(h0.device)] if name == 'forced_cached'
                                        else online_actions[name][positions.to(h0.device)])
                    if not torch.equal(measured_actions, expected_actions):
                        raise ValueError('online actions changed between observation and execution')
            else:
                hidden = online_h1 if name == 'controlled' else (zero if name == 'zero' else candidate)(online_h1)
                work['online_readout_calls'] += 1
                logits = adapter.readout(hidden).reshape(*online_h0.shape[:-1], -1)
                hidden = hidden.reshape(online_h0.shape)
            _finite(hidden, f'online {name} hidden'); _finite(logits, f'online {name} logits')
            flat_logits = logits.reshape(-1, logits.shape[-1])
            all_ce = _ce(flat_logits[:-1], row['input_ids'][1:], chunk=256)
            doc['online'][name] = _population_metrics(all_ce, positions.tolist(), valid)
            if native_ce is None:
                native_ce, native_logits, native_hidden = all_ce, flat_logits, hidden.reshape(-1, adapter.hidden_size)
            if name in ('controlled', 'zero'):
                valid_tensor = torch.tensor(valid, dtype=torch.long, device=flat_logits.device)
                diffs.add(f'native_vs_{name}.logits', _index_select_own_device(native_logits, valid_tensor),
                          _index_select_own_device(flat_logits, valid_tensor))
                hidden_valid = valid_tensor.to(hidden.device)
                diffs.add(f'native_vs_{name}.hidden', native_hidden.index_select(0, hidden_valid.to(native_hidden.device)),
                          hidden.reshape(-1, adapter.hidden_size).index_select(0, hidden_valid))
                diffs.add(f'native_vs_{name}.ce', native_ce[valid], all_ce[valid])
            for pop, metric in doc['online'][name].items():
                doc['paired_deltas'].setdefault(pop, {})[name + '_minus_native_nll'] = (
                    metric['nll'] - doc['online']['native'][pop]['nll'] if metric['tokens'] else None)
            if name == 'controlled':
                continue
            cache_name = 'learned' if name == 'forced_cached' else name
            selected_online_logits = flat_logits.index_select(0, positions.to(flat_logits.device))
            selected_online_hidden = hidden.reshape(-1, adapter.hidden_size).index_select(0, positions.to(hidden.device))
            if kind == 'router':
                chunks = [(i, h0[i:i+1], labels[i:i+1], 1) for i in global_ids.tolist()]
            else:
                needed = {i // settings.probe_chunk_contexts for i in global_ids.tolist()}
                chunks = [(index * settings.probe_chunk_contexts, h, y, count)
                          for index, (h, y, count) in enumerate(_fixed_chunks(cache['h1'], labels, settings.probe_chunk_contexts))
                          if index in needed]
            by_global = {index: local for local, index in enumerate(global_ids.tolist())}
            for start, h, y, count in chunks:
                if kind == 'router':
                    actions = cache[cache_name + '_actions'][start:start+1]
                    cached_hidden = adapter.step_pairs(h, action_to_pair(actions))
                    cached_logits = adapter.readout(cached_hidden)
                    # Authoritative old score_pairs path, not an invented replacement CE.
                    work['cached_score_pairs_calls'] += 1
                    losses = _finite(adapter.score_pairs(h, y, action_to_pair(actions)), 'cached singleton score').cpu()
                    diffs.add('cached_readout_vs_score_pairs.ce', losses, _ce(cached_logits, y))
                else:
                    cached_hidden = h if cache_name == 'native' else (zero if cache_name == 'zero' else candidate)(h)
                    cached_logits = differentiable_readout(adapter, cached_hidden)
                    losses = _ce(cached_logits, y)
                # score_pairs itself executes one additional FFN and readout.
                multiplier = 2 if kind == 'router' else 1
                work['cached_readout_calls'] += multiplier
                work['cached_readout_rows_including_padding'] += multiplier * len(h)
                if kind == 'router':
                    work['cached_terminal_ffn_calls'] += 2
                    work['cached_terminal_ffn_rows_including_padding'] += 2 * len(h)
                for offset in range(count):
                    index = start + offset
                    if index not in by_global:
                        continue
                    local = by_global[index]
                    if name != 'forced_cached':
                        cached_losses[cache_name][index] = losses[offset]
                    prefix = f'cached_vs_online.{name}'
                    diffs.add(prefix + '.logits', cached_logits[offset], selected_online_logits[local])
                    diffs.add(prefix + '.hidden', cached_hidden[offset], selected_online_hidden[local])
                    diffs.add(prefix + '.ce', losses[offset:offset+1], all_ce[positions[local]:positions[local]+1])
                    context_records[index][f'cached_{cache_name}_ce'] = float(losses[offset])
                    context_records[index][f'online_{name}_ce'] = float(all_ce[positions[local]])
            del logits, flat_logits, selected_online_logits
        del native_logits, native_hidden
        records.append(doc)
    for name, value in cached_losses.items():
        _finite(value, f'complete cached {name} losses')
    diffs.add('cached_native_vs_zero.ce', cached_losses['native'], cached_losses['zero'])
    differences = diffs.report(tolerances)
    zero_prefixes = ('native_vs_controlled.', 'native_vs_zero.', 'cached_native_vs_zero.',
                     'cached_vs_online.native.', 'cached_vs_online.zero.', 'cached_vs_online.h0',
                     'cached_vs_online.h1', 'cached_vs_online.features', 'cached_readout_vs_score_pairs.')
    zero_screen = all(value['within_declared_tolerance'] for key, value in differences.items()
                      if key.startswith(zero_prefixes))
    zero_screen = zero_screen and all(value['agree'] == value['count']
                                     for key, value in agreements.items() if key in ('native', 'zero'))
    reproduced = {}
    historical = metadata.get('historical_cached_metrics', {})
    if not historical and kind == 'router' and 'train_metrics' in metadata:
        historical = {'native': metadata['train_metrics']['native_ce'],
                      'learned': metadata['train_metrics']['greedy_ce']}
    elif not historical and kind == 'adapter' and 'train_probe_nll' in metadata:
        historical = {'learned': metadata['train_probe_nll']}
    finite_tree(historical, 'historical cached metrics')
    for name, expected_value in historical.items():
        if name not in cached_losses or isinstance(expected_value, bool) or not isinstance(expected_value, (int, float)):
            raise ValueError('invalid historical cached metric')
        measured = float(cached_losses[name].double().mean())
        reproduced[name] = {'historical_nll': expected_value, 'recomputed_nll': measured,
                            'absolute_difference': abs(measured - expected_value),
                            'tolerance': config['nll_tolerance'],
                            'within_declared_tolerance': abs(measured - expected_value) <= config['nll_tolerance']}
    for value in agreements.values():
        value['fraction'] = value['agree'] / value['count']
    result = {'schema': SCHEMA, 'status': 'complete', 'kind': kind, 'role': 'train_only',
        'sampling_sha256': manifest['sampling_sha256'], 'chosen_update': metadata.get('update'),
        'fit_gate_required': False, 'deployment_approved': False,
        'historical_train_fit_passed': metadata.get('train_fit_passed'),
        'cached': {name: {'tokens': len(values), 'nll': float(values.double().mean())} for name, values in cached_losses.items()},
        'online': _aggregate(records), 'per_document': records, 'contexts': context_records,
        'action_agreement': agreements, 'absolute_differences': differences,
        'tolerances': tolerances, 'numerical_screen_passed': (all(v['within_declared_tolerance'] for v in differences.values())
                                     and all(v['within_declared_tolerance'] for v in reproduced.values())
                                     and all(v['agree'] == v['count'] for v in agreements.values())),
        'action_agreement_screen_passed': all(v['agree'] == v['count'] for v in agreements.values()),
        'historical_metric_reproduction': reproduced,
        'zero_screen_passed': zero_screen, 'cross_shape_equivalence_asserted': False, 'cache_binding': binding, 'work': work,
        'work_bound': {'sampled_contexts_max': 4096, 'sampled_documents_max': 4096,
                       'padded_document_length_max': 4096, 'online_variants_per_document': 5 if kind == 'router' else 4,
                       'router_cached_score_calls_max': 4*len(h0) if kind == 'router' else 0,
                       'adapter_cached_probe_chunk_replays_max': 3*len(h0) if kind == 'adapter' else 0,
                       'report_tensor_policy': 'no vocabulary-sized tensors serialized'},
        'candidate_training_config': copy.deepcopy(config['fit']['training']),
        'score_protocol': {'router': 'singleton features/native pair/score_pairs; actor uses all cached rows as original probe',
                           'adapter': 'original probe_chunk_contexts and repeat-last masked tail for native cache and readout',
                           'probe_chunk_contexts': config['fit']['training'].get('probe_chunk_contexts'),
                           'full_document_ce_chunk_rows': 256},
        'interpretation': ['Paired measurements on identical sampled train positions; unselected positions are in the same sampled train documents.',
            'Terminal last FFN is pointwise: other-position changes cannot cause attention/state feedback; batch arithmetic may differ.',
            'Forced cached actions isolate action choice drift from same-action scoring drift; no automatic tolerance widening.',
            'Tolerance screens are numerical diagnostics, not proof of equivalence, learnability, held-out gain or deployment approval.']}
    return finite_tree(result, 'bridge')
