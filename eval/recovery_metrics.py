"""QRY-position loss and metric contract (recovery plan appendix B, eq. 1).

One supervised target per example: full-vocabulary CE at the QRY position,
answer-conditional CE, and legal-answer-set mass, related by

    ce_full = ce_answer_conditional - log(answer_mass).

Aggregation rule (appendix B): sum ``ce_sum`` / counts across batches first,
then divide — never average per-batch means over uneven batches.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F

from data.affine import QRY

METRIC_SUM_KEYS = (
    "ce_sum",
    "ce_conditional_sum",
    "answer_mass_sum",
    "correct_count",
    "valid_target_count",
    "eval_count",
)


def query_loss_and_metrics(
    logits: torch.Tensor,            # [batch, sequence, vocab], prefix-only forward
    input_ids: torch.Tensor,         # [batch, sequence]
    query_positions: torch.Tensor,   # [batch], QRY indices in the padded tensor
    target_token_ids: torch.Tensor,  # [batch], vocabulary ids (not 0..p-1 class indices)
    answer_token_ids: torch.Tensor,  # [p], legal numeric answer ids for this task
    query_token_id: int = QRY,
):
    """Single-batch loss and metrics at the QRY position.

    Returns ``(loss, metrics)`` where ``loss`` is the mean full-vocabulary CE
    (one target per example) and ``metrics`` carries both per-batch means and
    the sums/counts needed for cross-batch aggregation.
    """
    batch = torch.arange(logits.shape[0], device=logits.device)
    assert torch.all(input_ids[batch, query_positions] == query_token_id)
    assert answer_token_ids.unique().numel() == answer_token_ids.numel()
    assert torch.isin(target_token_ids, answer_token_ids).all()

    query_logits = logits[batch, query_positions].float()
    per_example_ce = F.cross_entropy(
        query_logits, target_token_ids, reduction="none"
    )
    legal_logits = query_logits.index_select(-1, answer_token_ids)
    log_answer_mass = (
        torch.logsumexp(legal_logits, dim=-1)
        - torch.logsumexp(query_logits, dim=-1)
    )
    conditional_ce = per_example_ce + log_answer_mass

    # Full-vocabulary accuracy is the deployed prediction metric here.
    predicted_token = query_logits.argmax(dim=-1)
    correct = predicted_token.eq(target_token_ids)

    loss = per_example_ce.mean()  # one supervised target per example
    n = target_token_ids.numel()
    metrics = {
        # Sums and counts: the only quantities safe to aggregate across
        # uneven batches.
        "ce_sum": per_example_ce.detach().sum(),
        "ce_conditional_sum": conditional_ce.detach().sum(),
        "answer_mass_sum": log_answer_mass.detach().exp().sum(),
        "correct_count": correct.detach().sum(),
        "valid_target_count": n,
        "eval_count": n,
        # Per-batch means for logging; do not average these across batches.
        "ce_full": per_example_ce.detach().mean(),
        "ce_answer_conditional": conditional_ce.detach().mean(),
        "answer_mass": log_answer_mass.detach().exp().mean(),
        "accuracy": correct.detach().float().mean(),
    }
    return loss, metrics


class MetricsAggregator:
    """Accumulates metric sums across batches and divides once at the end.

    Use for multi-batch evaluation (and as the single-process analogue of the
    global-target normalization in plan sec. 8.5): ``update`` with per-batch
    metric dicts, then ``compute`` for dataset-level values.
    """

    def __init__(self):
        self._sums = {key: 0.0 for key in METRIC_SUM_KEYS}
        self.num_batches = 0

    def update(self, metrics: Dict) -> "MetricsAggregator":
        for key in METRIC_SUM_KEYS:
            value = metrics[key]
            if isinstance(value, torch.Tensor):
                value = value.item()
            self._sums[key] += float(value)
        self.num_batches += 1
        return self

    def update_many(self, batches: Iterable[Dict]) -> "MetricsAggregator":
        for metrics in batches:
            self.update(metrics)
        return self

    @property
    def eval_count(self) -> int:
        return int(self._sums["eval_count"])

    def compute(self) -> Dict[str, float]:
        n = self._sums["eval_count"]
        if n <= 0:
            raise ValueError("no examples accumulated")
        if self._sums["valid_target_count"] != n:
            raise ValueError("valid target count differs from eval count")
        return {
            "ce_full": self._sums["ce_sum"] / n,
            "ce_answer_conditional": self._sums["ce_conditional_sum"] / n,
            "answer_mass": self._sums["answer_mass_sum"] / n,
            "accuracy": self._sums["correct_count"] / n,
            "ce_sum": self._sums["ce_sum"],
            "correct_count": int(self._sums["correct_count"]),
            "eval_count": int(n),
            "valid_target_count": int(self._sums["valid_target_count"]),
            "num_batches": self.num_batches,
        }


def aggregate_batches(batches: Iterable[Dict]) -> Dict[str, float]:
    """Convenience wrapper: aggregate per-batch metric dicts into dataset
    metrics (sum ce_sum/counts, then divide)."""
    return MetricsAggregator().update_many(batches).compute()
