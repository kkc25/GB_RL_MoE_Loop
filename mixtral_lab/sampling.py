"""Reproducible document-balanced next-token context sampling.

The positive budget is exact: insufficient positions fail before prefix execution.
Empty/non-supervisable documents are recorded, never silently counted as covered.
When the budget is smaller than the eligible document count, a seeded uniform
subset of documents is selected. Otherwise every eligible document is represented.
Positions are uniform without replacement within each document; this is *document
balanced*, not an unbiased sample of the token-weighted corpus distribution.
Only train, selection and calibration roles are accepted; held-out data is not a
valid input role. Callers remain responsible for supplying the correct split.
"""
import hashlib
import json
import random

import torch


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _validate_rows(rows):
    result, seen = [], set()
    for row in rows:
        doc_id = row.get('id')
        if not isinstance(doc_id, str) or not doc_id or doc_id in seen:
            raise ValueError('document IDs must be unique nonempty strings')
        seen.add(doc_id)
        ids, mask = row.get('input_ids'), row.get('attention_mask')
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError('input_ids must be one-dimensional int64 tensors')
        if not isinstance(mask, torch.Tensor) or mask.ndim != 1 or mask.shape != ids.shape:
            raise ValueError('attention_mask must match one-dimensional input_ids')
        ids_list, mask_list = ids.detach().cpu().tolist(), mask.detach().cpu().tolist()
        if any(x not in (0, 1) for x in mask_list):
            raise ValueError('attention_mask must be binary')
        if any(x < 0 for x in ids_list):
            raise ValueError('input_ids must be nonnegative')
        positions = [i for i in range(len(ids_list) - 1) if mask_list[i] and mask_list[i + 1]]
        result.append({'id': doc_id, 'ids': ids_list, 'positions': positions,
                       'content_sha256': _hash({'input_ids': ids_list, 'attention_mask': mask_list})})
    return result


def plan_stratified_contexts(rows, count, seed, role):
    """Return JSON-serializable selection/provenance without executing the model."""
    if type(count) is not int or count <= 0:
        raise ValueError('requested context count must be a positive integer')
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    if role not in ('train', 'selection', 'calibration'):
        raise ValueError('context role must be train, selection or calibration (never held-out)')
    docs = _validate_rows(rows)
    eligible = [i for i, doc in enumerate(docs) if doc['positions']]
    available = sum(len(doc['positions']) for doc in docs)
    if count > available:
        raise ValueError(f'requested {count} contexts but only {available} valid next-token positions available')
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = eligible[:min(count, len(eligible))]
    quotas = dict.fromkeys(selected, 0)
    remaining = count
    # Water filling: short documents saturate, their remaining share is reassigned.
    while remaining:
        for index in selected:
            if quotas[index] < len(docs[index]['positions']):
                quotas[index] += 1
                remaining -= 1
                if not remaining:
                    break
    contexts = []
    for index in selected:
        doc = docs[index]
        for position in rng.sample(doc['positions'], quotas[index]):
            contexts.append({'document_id': doc['id'], 'token_position': position,
                             'target': doc['ids'][position + 1],
                             'content_sha256': doc['content_sha256']})
    rng.shuffle(contexts)
    manifest = {
        'version': 1, 'strategy': 'document_balanced_uniform_positions',
        'role': role, 'seed': seed, 'requested_count': count, 'actual_count': len(contexts),
        'available_positions': available, 'document_count': len(docs),
        'eligible_document_count': len(eligible), 'covered_document_count': len(selected),
        'document_coverage': len(selected) / len(eligible),
        'documents': [{'document_id': doc['id'], 'content_sha256': doc['content_sha256'],
                       'available_positions': len(doc['positions']), 'selected_count': quotas.get(i, 0)}
                      for i, doc in enumerate(docs)],
        'contexts': contexts,
    }
    manifest['sampling_sha256'] = _hash(manifest)
    return manifest


@torch.no_grad()
def collect_stratified_contexts(adapter, rows, count, seed, role):
    """Capture once per selected document; return detached contiguous h0, y, manifest.

    Input tokens follow the embedding device; indices and targets follow captured
    terminal states (not the input device), supporting layer-dispatched models.
    Output order exactly matches ``manifest['contexts']``.
    """
    rows = list(rows)
    manifest = plan_stratified_contexts(rows, count, seed, role)
    by_id = {row['id']: row for row in rows}
    groups = {}
    for output_index, context in enumerate(manifest['contexts']):
        groups.setdefault(context['document_id'], []).append((output_index, context['token_position']))
    chunks, targets, output_indices = [], [], []
    input_device = adapter.model.get_input_embeddings().weight.device
    for doc_id, selected in groups.items():
        row = by_id[doc_id]
        captured = adapter.capture(input_ids=row['input_ids'].unsqueeze(0).to(input_device),
                                   attention_mask=row['attention_mask'].unsqueeze(0).to(input_device))
        if captured.ndim != 3 or captured.shape[:2] != (1, len(row['input_ids'])):
            raise ValueError('captured prefix shape does not match document')
        positions = torch.tensor([position for _, position in selected], device=captured.device)
        chunks.append(captured[0].index_select(0, positions).detach())
        targets.append(row['input_ids'].to(captured.device).index_select(0, positions + 1).detach())
        output_indices.extend(index for index, _ in selected)
    hidden = torch.cat(chunks)
    labels = torch.cat(targets)
    inverse = torch.tensor(output_indices, device=hidden.device).argsort()
    return (hidden.index_select(0, inverse).detach().contiguous(),
            labels.index_select(0, inverse).detach().contiguous(), manifest)
