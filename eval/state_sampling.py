"""Document-coverage-first state sampling and document-level paired bootstrap.

Repair plan R2 (audits/code_audit_2026-10-25 sec. 4.2): states for the
controller probes must *cover documents first* instead of walking packed
sequences in order.  ``plan_state_sampling`` picks, per document, up to
``max_per_doc`` (default 16) random *legal* positions -- supervised positions,
``labels != -100`` -- and orders them round-robin across documents, so the
first ``min(n_states, n_documents)`` states all come from distinct documents.
Shorter documents contribute only their legal positions; the same position is
never repeated to fill a quota.

``sample_states`` executes that plan against a frozen LoopMoE backbone with
the same hidden-state caching as ``eval.counterfactual.gather_states`` and
returns a dict with the same state keys (``h``, ``targets``, ``positions``,
``doc_ids``, ``seq_indices``, ``split``) plus a ``stats`` record carrying the
required diagnostics: document count, sequence count, maximum document
weight, duplicate-position count, and the document-id intersection with
caller-supplied reference / exclusion pools (both must be empty for a clean
split; reported, not silently dropped).

``document_paired_bootstrap`` implements the document-level paired interval:
the estimand is ``sum(loss_diff) / sum(valid_tokens)``, recomputed on every
bootstrap resample *of documents* (never token-level resampling, never
averaging per-document perplexities).
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from eval.counterfactual import cache_last_ffn_input
from train.common import autocast_context

DEFAULT_MAX_PER_DOC = 16


# --------------------------------------------------------------------------
# Sampling plan (pure metadata; no model forward)
# --------------------------------------------------------------------------

def supervised_positions_by_doc(dataset) -> Dict[int, List[Tuple[int, int]]]:
    """Map document id -> [(sequence index, position)] with a supervised label.

    Positions are exactly those with ``labels != -100`` in the dataset view
    (segment-final tokens, padding, and the final sequence position excluded).
    """
    by_doc: Dict[int, List[Tuple[int, int]]] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        labels = item["labels"]
        doc_ids = item["doc_ids"]
        for position in np.nonzero(labels != -100)[0]:
            by_doc.setdefault(int(doc_ids[position]), []).append((index, int(position)))
    return by_doc


def plan_state_sampling(
    dataset,
    n_states: int,
    *,
    seed: int = 0,
    max_per_doc: int = DEFAULT_MAX_PER_DOC,
) -> Dict[str, object]:
    """Deterministic coverage-first sampling plan over one packed split view.

    Per document the legal positions are shuffled with a seeded RNG and capped
    at ``max_per_doc``; the capped lists are then interleaved round-robin over
    documents (document order itself shuffled by the same seed), so document
    coverage is maximized before any document contributes a second state.
    Returns the ordered ``selection`` of ``(sequence index, doc id, position)``
    triples plus per-document candidate counts.  Raises ``ValueError`` when the
    split cannot supply ``n_states`` legal states under the cap.
    """
    if n_states <= 0:
        raise ValueError("n_states must be positive")
    if max_per_doc <= 0:
        raise ValueError("max_per_doc must be positive")
    rng = random.Random(f"state-sampling-v1:{seed}")

    by_doc = supervised_positions_by_doc(dataset)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    capped: Dict[int, List[Tuple[int, int]]] = {}
    for doc in docs:
        candidates = list(by_doc[doc])
        rng.shuffle(candidates)
        capped[doc] = candidates[:max_per_doc]

    available = sum(len(entries) for entries in capped.values())
    if available < n_states:
        raise ValueError(
            f"split view only yields {available} legal states from {len(docs)} documents "
            f"with max_per_doc={max_per_doc}, requested {n_states}"
        )

    selection: List[Tuple[int, int, int]] = []
    round_index = 0
    while len(selection) < n_states:
        progressed = False
        for doc in docs:
            entries = capped[doc]
            if round_index < len(entries):
                seq, pos = entries[round_index]
                selection.append((seq, doc, pos))
                progressed = True
                if len(selection) == n_states:
                    break
        if not progressed:
            break  # unreachable given the availability check
        round_index += 1
    return {"selection": selection, "doc_counts": {doc: len(capped[doc]) for doc in docs}}


# --------------------------------------------------------------------------
# Model execution of a sampling plan
# --------------------------------------------------------------------------

@torch.no_grad()
def sample_states(
    model,
    dataset,
    n_states: int,
    *,
    split: str,
    device: torch.device,
    dtype: str = "float32",
    seed: int = 0,
    seq_batch: int = 2,
    depth: int = 1,
    max_per_doc: int = DEFAULT_MAX_PER_DOC,
    ledger=None,
    reference_doc_ids: Optional[Iterable[int]] = None,
    exclude_doc_ids: Optional[Iterable[int]] = None,
) -> dict:
    """Gather ``n_states`` document-coverage-first states from one packed split.

    Same return contract as ``eval.counterfactual.gather_states`` -- keys
    ``h``, ``targets``, ``positions``, ``doc_ids``, ``seq_indices``, ``split``
    -- plus a ``stats`` dict:

      ``n_documents``            distinct documents contributing states
      ``n_sequences``            distinct packed sequences contributing states
      ``max_doc_states``         largest per-document state count
      ``max_doc_weight``         max_doc_states / n_states
      ``n_duplicate_positions``  states sharing a (sequence, position) pair
                                 (0 by construction; asserted)
      ``doc_id_intersection``    sampled doc ids also in ``reference_doc_ids``
      ``excluded_doc_ids_present`` sampled doc ids also in ``exclude_doc_ids``
    """
    plan = plan_state_sampling(dataset, n_states, seed=seed, max_per_doc=max_per_doc)
    selection: List[Tuple[int, int, int]] = plan["selection"]

    by_sequence: Dict[int, List[Tuple[int, int, int]]] = {}
    for order, (seq, doc, pos) in enumerate(selection):
        by_sequence.setdefault(seq, []).append((order, doc, pos))

    h_rows: List[torch.Tensor] = [None] * n_states
    targets = torch.empty(n_states, dtype=torch.long)
    positions = torch.empty(n_states, dtype=torch.long)
    doc_column = torch.empty(n_states, dtype=torch.long)
    seq_column = torch.empty(n_states, dtype=torch.long)

    ordered_sequences = sorted(by_sequence)
    for start in range(0, len(ordered_sequences), seq_batch):
        chunk = ordered_sequences[start : start + seq_batch]
        items = [dataset[seq] for seq in chunk]
        input_ids = torch.from_numpy(np.stack([item["input_ids"] for item in items])).to(device)
        segment_ids = torch.from_numpy(np.stack([item["segment_ids"] for item in items])).to(device)
        with autocast_context(device, dtype):
            h = cache_last_ffn_input(model, input_ids, segment_ids, depth)
        if ledger is not None:
            ledger.backbone_prefix_tokens += int(input_ids.numel())
        for row, (seq, item) in enumerate(zip(chunk, items)):
            labels = torch.from_numpy(np.asarray(item["labels"]))
            for order, doc, pos in by_sequence[seq]:
                if int(labels[pos]) == -100:
                    raise RuntimeError(f"plan selected unsupervised position (seq {seq}, pos {pos})")
                h_rows[order] = h[row, pos].float().cpu()
                targets[order] = int(labels[pos])
                positions[order] = pos
                doc_column[order] = doc
                seq_column[order] = seq

    sampled_docs = [int(doc) for _, doc, _ in selection]
    doc_counts: Dict[int, int] = {}
    for doc in sampled_docs:
        doc_counts[doc] = doc_counts.get(doc, 0) + 1
    pairs = [(int(seq), int(pos)) for seq, _, pos in selection]
    n_duplicate_positions = len(pairs) - len(set(pairs))
    if n_duplicate_positions:
        raise RuntimeError(f"sampling plan produced {n_duplicate_positions} duplicate positions")

    reference = {int(d) for d in reference_doc_ids} if reference_doc_ids is not None else set()
    excluded = {int(d) for d in exclude_doc_ids} if exclude_doc_ids is not None else set()
    sampled_doc_set = set(doc_counts)

    stats = {
        "n_documents": len(sampled_doc_set),
        "n_sequences": len(by_sequence),
        "max_doc_states": max(doc_counts.values()),
        "max_doc_weight": max(doc_counts.values()) / n_states,
        "n_duplicate_positions": n_duplicate_positions,
        "max_per_doc": max_per_doc,
        "doc_id_intersection": sorted(sampled_doc_set & reference),
        "excluded_doc_ids_present": sorted(sampled_doc_set & excluded),
        "doc_ids": sorted(sampled_doc_set),
    }
    return {
        "h": torch.stack(h_rows),
        "targets": targets,
        "positions": positions,
        "doc_ids": doc_column,
        "seq_indices": seq_column,
        "split": split,
        "stats": stats,
    }


# --------------------------------------------------------------------------
# Document-level paired bootstrap
# --------------------------------------------------------------------------

def document_paired_bootstrap(
    loss_a,
    loss_b,
    doc_ids,
    *,
    weights=None,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Paired document-level bootstrap interval for the mean per-token loss difference.

    ``loss_a`` / ``loss_b`` are paired per-token (or per-state) losses,
    ``doc_ids`` their document membership, ``weights`` the valid-token weight
    of each row (default 1).  The estimand is the ratio
    ``sum(weights * (loss_a - loss_b)) / sum(weights)``; documents are
    resampled with replacement and the ratio is recomputed from the resampled
    documents' sums on every replicate, so the interval reflects document-level
    variation rather than token count.  Returns the point estimate, percentile
    interval, bootstrap standard error, and per-document effective token
    shares (plus their max and the effective number of documents,
    ``1 / sum(share**2)``).
    """
    a = np.asarray(loss_a, dtype=np.float64).reshape(-1)
    b = np.asarray(loss_b, dtype=np.float64).reshape(-1)
    docs = np.asarray(doc_ids).reshape(-1)
    if not (a.shape == b.shape == docs.shape):
        raise ValueError("loss_a, loss_b and doc_ids must have the same length")
    if weights is None:
        w = np.ones_like(a)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != a.shape:
            raise ValueError("weights must have the same length as loss_a")
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
    if a.shape[0] == 0:
        raise ValueError("no observations")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("losses must be finite")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    unique_docs, inverse = np.unique(docs, return_inverse=True)
    n_docs = unique_docs.shape[0]
    diff = a - b
    doc_diff = np.zeros(n_docs, dtype=np.float64)
    doc_tokens = np.zeros(n_docs, dtype=np.float64)
    np.add.at(doc_diff, inverse, diff * w)
    np.add.at(doc_tokens, inverse, w)
    total_tokens = float(doc_tokens.sum())
    if total_tokens <= 0:
        raise ValueError("total valid-token weight must be positive")

    estimate = float(doc_diff.sum() / total_tokens)
    shares = doc_tokens / total_tokens

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_bootstrap, dtype=np.float64)
    for replicate in range(n_bootstrap):
        sample = rng.integers(0, n_docs, n_docs)
        replicates[replicate] = doc_diff[sample].sum() / doc_tokens[sample].sum()
    alpha = 1.0 - confidence
    lo, hi = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "estimate": estimate,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "confidence": confidence,
        "se": float(replicates.std(ddof=1)),
        "n_bootstrap": n_bootstrap,
        "n_documents": int(n_docs),
        "n_rows": int(a.shape[0]),
        "doc_ids": unique_docs.tolist(),
        "doc_diff_sum": doc_diff.tolist(),
        "doc_tokens": doc_tokens.tolist(),
        "doc_token_shares": shares.tolist(),
        "max_doc_token_share": float(shares.max()),
        "effective_documents": float(1.0 / (shares**2).sum()),
    }
