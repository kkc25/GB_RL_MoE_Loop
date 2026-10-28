"""Bounded, label-aware diagnostics, NOT a deployable routing policy.

Every expert execution and vocabulary-head score uses one context row. This is
a singleton counterfactual baseline, NOT online/batched native NLL; batch GEMM
shapes can change router near-ties and CE. Repeated anchors only check internal
singleton consistency, not cross-shape equivalence. R2 is
conditional on FIXED native R1, not a joint search over two-round trajectories.
Only development calibration/selection contexts may be inspected here.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from collections import defaultdict

import torch
from torch.nn import functional as F
from .adapter import PAIR_TABLE, pair_to_action


@dataclass(frozen=True)
class OracleConfig:
    max_contexts: int = 32
    max_score_calls: int = 1856
    practical_gain: float = 0.005
    tie_tolerance: float = 1e-7
    anchor_atol: float = 1e-6

    def validate(self):
        for name in ('max_contexts', 'max_score_calls'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('practical_gain', 'tie_tolerance', 'anchor_atol'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


def _digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(tensor.dtype), list(tensor.shape))).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _summary(rows, threshold):
    result = {'contexts': len(rows)}
    for kind in ('r1', 'r2'):
        gains = torch.tensor([row[f'{kind}_gain_nats'] for row in rows], dtype=torch.float64)
        result[kind] = {
            'mean_gain_nats': gains.mean().item(),
            'fraction_above_practical_threshold': (gains > threshold).double().mean().item(),
            'gain_quantiles_nats': {str(q): torch.quantile(gains, q).item()
                                    for q in (0., .25, .5, .75, .9, 1.)},
        }
    result['stop_fraction'] = sum(row['r2_action'] == 0 for row in rows) / len(rows)
    return result


@torch.no_grad()
def run_oracle(adapter, h0, targets, provenance, identities, *, role, config=None):
    """Return JSON-safe exhaustive scores on an explicitly bounded development set.

    ``provenance`` contains document_id (or doc_id) and token_position per row.
    ``identities`` binds the caller's frozen model/data/source manifest. Tensor
    and provenance hashes are added here; this function never chooses contexts.
    """
    config = config or OracleConfig()
    config.validate()
    if role not in ('calibration', 'selection'):
        raise ValueError('oracle is label-aware: only calibration/selection roles are allowed')
    if not isinstance(identities, dict) or not identities:
        raise ValueError('nonempty frozen identities are required')
    identities = json.loads(json.dumps(identities, allow_nan=False))
    if h0.ndim != 2 or not h0.is_floating_point() or not torch.isfinite(h0).all():
        raise ValueError('h0 must be finite floating [N,D]')
    count = len(h0)
    if not 0 < count <= config.max_contexts:
        raise ValueError('context count exceeds bounded oracle budget or is empty')
    if targets.shape != (count,) or targets.dtype != torch.long or (targets < 0).any():
        raise ValueError('targets must be nonnegative int64 [N]')
    if targets.device != h0.device:
        raise ValueError('targets and h0 must share a device')
    if len(provenance) != count:
        raise ValueError('provenance must identify every context')
    clean_provenance = []
    seen = set()
    for index, row in enumerate(provenance):
        row = json.loads(json.dumps(row, allow_nan=False))
        doc = row.get('document_id', row.get('doc_id'))
        pos = row.get('token_position')
        if not isinstance(doc, (str, int)) or isinstance(doc, bool) or isinstance(pos, bool) or not isinstance(pos, int) or pos < 0:
            raise ValueError('provenance needs document_id and nonnegative token_position')
        if 'target' in row and row['target'] != int(targets[index].item()):
            raise ValueError('provenance target does not match frozen tensor target')
        if 'role' in row and row['role'] != role:
            raise ValueError('provenance role does not match oracle role')
        key = (str(doc), pos)
        if key in seen:
            raise ValueError('duplicate context provenance')
        seen.add(key)
        row['document_id'] = str(doc)
        clean_provenance.append(row)
    if any(p.requires_grad for p in adapter.model.parameters()) or adapter.model.training:
        raise ValueError('oracle requires an eval model with all weights frozen')
    required_calls = count * 58  # two native anchors plus 28 R1 and 28 R2
    if required_calls > config.max_score_calls:
        raise ValueError(f'oracle needs {required_calls} score calls, exceeds budget')
    calls = 0
    def score(h, y, pair):
        nonlocal calls
        calls += 1
        values = adapter.score_pairs(h, y, pair)
        if values.shape != (1,) or not torch.isfinite(values).all():
            raise ValueError('nonfinite or malformed oracle score')
        return values.item()
    def consistent(a, b):
        if abs(a-b) > config.anchor_atol:
            raise ValueError('native anchor CE mismatch: numerical protocol is not consistent')
    rows = []
    for index in range(count):
        h = h0[index:index+1].detach()
        y = targets[index:index+1]
        native = adapter.native_pair(h)
        native_action = int(pair_to_action(native).item())
        initial_reference = score(h, y, native)
        r1 = [score(h, y, torch.tensor([pair], device=h.device, dtype=torch.long)) for pair in PAIR_TABLE]
        repeated_reference = score(h, y, native)
        # The canonical baseline is the native candidate itself. Thus the
        # fallback is a real member of the candidate set, never a clipped gain.
        reference = r1[native_action-1]
        consistent(reference, initial_reference)
        consistent(reference, repeated_reference)
        h1 = adapter.step_pairs(h, native)
        logits = adapter.readout(h1)
        stop_check = F.cross_entropy(logits.float(), y.to(logits.device)).item()
        if not math.isfinite(stop_check):
            raise ValueError('nonfinite STOP score')
        consistent(reference, stop_check)
        r2 = [score(h1, y, torch.tensor([pair], device=h.device, dtype=torch.long)) for pair in PAIR_TABLE]
        best1 = min(range(28), key=lambda j: (r1[j], j))
        action1 = best1+1 if reference-r1[best1] > config.tie_tolerance else native_action
        best2 = min(range(28), key=lambda j: (r2[j], j))
        action2 = best2+1 if reference-r2[best2] > config.tie_tolerance else 0
        rows.append({
            'provenance': clean_provenance[index], 'context_sha256': _digest(h),
            'target': int(y.item()), 'native_action': native_action,
            'native_pair': native[0].tolist(), 'native_r1_ce': reference,
            'anchor_initial_ce': initial_reference, 'anchor_repeat_ce': repeated_reference,
            'stop_readout_check_ce': stop_check,
            'max_anchor_error': max(abs(reference-v) for v in (initial_reference, repeated_reference, stop_check)),
            'r1_scores_nats': r1, 'r2_scores_nats': [reference]+r2,
            'r1_action': action1, 'r2_action': action2,
            'r1_minimum_ce': min(r1), 'r2_minimum_ce': min(reference, min(r2)),
            'r1_exact_headroom_nats': reference-min(r1),
            'r2_exact_headroom_nats': reference-min(reference, min(r2)),
            'r1_gain_nats': reference-r1[action1-1],
            'r2_gain_nats': 0. if action2 == 0 else reference-r2[action2-1],
        })
    groups = defaultdict(list)
    for row in rows:
        groups[row['provenance']['document_id']].append(row)
    return {
        'schema_version': 1, 'diagnosis': 'label-aware development oracle; not a deployable policy or measured training gain',
        'baseline_scope': 'singleton counterfactual native R1, not production online/batched native NLL',
        'r2_scope': '28 pairs plus STOP after fixed singleton-native R1; not joint two-step oracle',
        'limitations': ['Repeated native anchors establish internal singleton consistency only, not cross-shape equivalence.',
            'Batch GEMM shapes can change native routing near ties and CE; no online headroom or deployment gain is established.',
            'No cross-shape reference was evaluated; GPU, throughput and end-to-end quality require separate experiments.'],
        'role': role, 'identities': identities,
        'context_tensor_sha256': _digest(h0), 'targets_sha256': _digest(targets),
        'provenance_sha256': hashlib.sha256(json.dumps(clean_provenance, sort_keys=True).encode()).hexdigest(),
        'numeric_protocol': {'context_rows_per_call': 1, 'pair_rows_per_call': 1,
            'head_rows_per_call': 1, 'ce_dtype': 'float32', 'hidden_dtype': str(h0.dtype),
            'anchor_atol': config.anchor_atol, 'tie_tolerance': config.tie_tolerance,
            'gain_rule': 'selected-action gains use tie_tolerance; exact minima also retained',
            'stop_reference': 'exact canonical native R1 CE', 'r1_reference': 'singleton native candidate in exhaustive R1 set'},
        'budget': {'max_contexts': config.max_contexts, 'max_score_calls': config.max_score_calls,
            'actual_contexts': count, 'actual_score_calls': calls,
            'actual_pair_rows': calls+count, 'actual_head_calls': calls+count,
            'actual_head_rows': calls+count, 'extra_native_steps': count,
            'candidate_pairs_per_round': 28},
        'actions': {'0': 'STOP', **{str(i+1): list(pair) for i, pair in enumerate(PAIR_TABLE)}},
        'practical_gain_threshold_nats': config.practical_gain,
        'summary': _summary(rows, config.practical_gain),
        'document_groups': {doc: _summary(group, config.practical_gain) for doc, group in groups.items()},
        'contexts': rows,
    }
