"""E10 ordered complementarity pilot: exact two-round Top-2 path costs.

Real-LM counterpart of
plans/loop_moe_research_plan_2026-10-25/experiments/E10_ordered_complementarity.md

On a frozen LoopMoE backbone, states are the cached hidden states entering
the last adaptive (looping) FFN, exactly as in E01 (eval/counterfactual.py).
Because that layer is last, every supervised position continues independently
and the full two-round action grid can be executed exactly:

* round 1 forces each of the 28 Top-2 actions ``a``,
* round 2 forces each of the 28 Top-2 actions ``b`` (784 ordered paths),
* rounds 3..``rounds`` follow the frozen baseline policy pi0, so every path
  uses the same number of expert calls (matched compute, T=``rounds``).

The cost after round 2 (``pair_cost``) and after the pi0 continuation
(``pair_q``) are both measured in FP32 CE, chunked so the full
states x paths x vocab logits tensor is never materialized.

With a fixed label-free anchor (a0, b0) — plain legal action IDs, identical
for all four terms — the ordered interaction and order effect are

    Gamma_{a,b} = L_{a,b0} + L_{a0,b} - L_{a,b} - L_{a0,b0}
    O_{a,b}     = L_{a,b} - L_{b,a}

Gamma > 0 means a positive interaction RELATIVE TO THE CHOSEN ANCHOR, the
input distribution and the loss scale; it is not an intrinsic expert
property, so Gamma is recomputed over a small anchor grid (reference
sensitivity).  a == b is allowed (repeating a Top-2 set may be optimal).
The (a, b) vs (b, a) comparison is the order-shuffled control with matched
call count and repetition rate.

Three cost predictors are trained on train states by per-path squared loss
and compared at EQUAL capacity where the plan requires it:

* additive:   L_hat(s,a,b) = u_a(s) + v_b(s)
* low-rank:   u_a(s) + v_b(s) + p_a(s)^T q_b(s)   (rank r)
* MLP:        a generic MLP predicting all 784 path costs, hidden width
              chosen so its parameter count does not exceed the low-rank
              predictor's.

Each predictor selects the argmin pair per held-out state; since every path
was really executed, the executed cost of a selected pair is the measured
cost of that path.  Reported: held-out real combination cost of every
policy vs the baseline pair (a*, pi0-after-a*), the label-informed oracle
pair (separately, never a deploy gain), per-policy order-shuffle controls,
reference sensitivity of Gamma, negative-interaction and repeated-set
(a == b) statistics, predictor MSE at equal parameters, and the JSON cost
ledger.

Example (small CPU pilot, defaults are already small):

    python experiments/e10_ordered.py --config configs/e00_debug67_b2.json \
        --checkpoint runs/e00/debug67_b2/latest.pt --device cpu \
        --train-states 32 --validation-states 16 --test-states 16 \
        --output-dir runs/e10/pilot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import platform
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.packed_lm import PackedLMDataset
from eval.counterfactual import (
    CounterfactualLedger,
    gather_states,
    last_adaptive_layer,
    readout_ce,
)
from lm_config import LMExperimentConfig
from model import LoopMoE
from model.top2 import baseline_top2_pairs, enumerate_top2_actions, pair_to_action
from train.common import MetricWriter, autocast_context, seed_everything
from train.recovery_common import atomic_json, sha256_file
from train.value_student import standardization, student_features

PRIMARY_ANCHOR = (0, 0)
SENSITIVITY_ANCHORS = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 2))


@dataclass
class PairPathCosts:
    """Real measured costs of every ordered two-round path, per state.

    ``immediate`` is (n_states, A): FP32 CE after forcing action ``a`` at
    round 1.  ``pair_cost`` and ``pair_q`` are (n_states, A, A): CE after
    forcing ``a`` then ``b`` (T=2), and after continuing with pi0 up to
    ``rounds`` total calls (T=``rounds``).  ``second_baseline[:, a]`` is the
    action pi0 would take at round 2 after forcing ``a``, so the ordinary
    fixed-depth baseline cost is ``pair_q[s, a*, second_baseline[s, a*]]``
    with ``a* = baseline_action[s]``; all terms share the same call count.
    """

    immediate: torch.Tensor
    pair_cost: torch.Tensor
    pair_q: torch.Tensor
    baseline_action: torch.Tensor
    second_baseline: torch.Tensor
    actions: torch.Tensor
    rounds: int


@torch.no_grad()
def two_round_path_costs(
    model,
    h: torch.Tensor,
    targets: torch.Tensor,
    *,
    rounds: int = 4,
    state_chunk: int = 4,
    readout_chunk: int = 256,
    dtype: str = "float32",
    ledger: CounterfactualLedger | None = None,
) -> PairPathCosts:
    """Measure exact costs of all A x A ordered two-round paths per state.

    Round 1 forces action ``a``, round 2 forces action ``b``, rounds
    3..``rounds`` follow the frozen baseline policy.  Computation is shared
    across ``b`` for each forced ``a`` (round 1 runs A times per state, not
    A x A), candidates are processed ``state_chunk`` states at a time and
    head readouts are accumulated per ``readout_chunk`` rows, so the full
    path x vocab logits tensor is never materialized.
    """
    if rounds < 2:
        raise ValueError("rounds must be >= 2 for a two-round path")
    layer = last_adaptive_layer(model)
    block = model.blocks[layer]
    gamma = model.gammas[layer]
    device = h.device
    experts = model.config.experts_per_layer
    actions = enumerate_top2_actions(experts, device=device)
    n_actions = actions.shape[0]
    n_states = h.shape[0]
    hidden = h.shape[-1]
    immediate = torch.empty(n_states, n_actions, dtype=torch.float32, device=device)
    pair_cost = torch.empty(n_states, n_actions, n_actions, dtype=torch.float32, device=device)
    pair_q = torch.empty(n_states, n_actions, n_actions, dtype=torch.float32, device=device)
    baseline_action = torch.empty(n_states, dtype=torch.long, device=device)
    second_baseline = torch.empty(n_states, n_actions, dtype=torch.long, device=device)
    for start in range(0, n_states, state_chunk):
        hc = h[start : start + state_chunk]
        tc = targets[start : start + state_chunk]
        size = hc.shape[0]
        with autocast_context(device, dtype):
            logits = block.executor.gate_logits(block.ffn_norm(hc))
            baseline_action[start : start + size] = pair_to_action(baseline_top2_pairs(logits.float()), experts=experts)
            expanded = hc.unsqueeze(1).expand(size, n_actions, hidden).reshape(-1, hidden)
            forced_first = torch.arange(n_actions, device=device).repeat(size)
            update, _, _, _ = block.executor(block.ffn_norm(expanded), forced_first)
            after_first = expanded + gamma * update  # (size * A, hidden)
            second_logits = block.executor.gate_logits(block.ffn_norm(after_first))
            second_baseline[start : start + size] = pair_to_action(
                baseline_top2_pairs(second_logits.float()), experts=experts
            ).reshape(size, n_actions)
            expanded_second = after_first.unsqueeze(1).expand(size * n_actions, n_actions, hidden).reshape(-1, hidden)
            forced_second = torch.arange(n_actions, device=device).repeat(size * n_actions)
            update, _, _, _ = block.executor(block.ffn_norm(expanded_second), forced_second)
            after_second = expanded_second + gamma * update  # (size * A * A, hidden)
            current = after_second
            for _ in range(rounds - 2):
                update, _, _, _ = block.executor(block.ffn_norm(current))
                current = current + gamma * update
            if ledger is not None:
                ledger.candidate_pair_calls += size * n_actions + size * n_actions * n_actions * (rounds - 1)
        targets_first = tc.unsqueeze(1).expand(size, n_actions).reshape(-1)
        targets_second = tc.view(size, 1, 1).expand(size, n_actions, n_actions).reshape(-1)
        immediate[start : start + size] = readout_ce(model, after_first, targets_first, readout_chunk, ledger).reshape(size, n_actions)
        pair_cost[start : start + size] = readout_ce(model, after_second, targets_second, readout_chunk, ledger).reshape(size, n_actions, n_actions)
        pair_q[start : start + size] = readout_ce(model, current, targets_second, readout_chunk, ledger).reshape(size, n_actions, n_actions)
    return PairPathCosts(immediate, pair_cost, pair_q, baseline_action, second_baseline, actions, rounds)


def interaction_gamma(costs: torch.Tensor, a0: int, b0: int) -> torch.Tensor:
    """Gamma_{a,b} = L_{a,b0} + L_{a0,b} - L_{a,b} - L_{a0,b0}, per state."""
    n_actions = costs.shape[-1]
    if not (0 <= a0 < n_actions and 0 <= b0 < n_actions):
        raise ValueError(f"anchor ({a0}, {b0}) outside [0, {n_actions})")
    return (
        costs[:, :, b0].unsqueeze(-1)
        + costs[:, a0, :].unsqueeze(-2)
        - costs
        - costs[:, a0, b0].view(-1, 1, 1)
    )


def order_effect(costs: torch.Tensor) -> torch.Tensor:
    """O_{a,b} = L_{a,b} - L_{b,a}; antisymmetric, zero on the a == b diagonal."""
    return costs - costs.transpose(-1, -2)


def gamma_summary(gamma: torch.Tensor) -> dict:
    """Distribution of Gamma over states and pairs; a == b pairs reported apart."""
    n_actions = gamma.shape[-1]
    diagonal = torch.eye(n_actions, dtype=torch.bool, device=gamma.device)
    off_diagonal = gamma.masked_select(~diagonal.expand_as(gamma))
    diagonal_values = gamma.masked_select(diagonal.expand_as(gamma))
    return {
        "mean": gamma.mean().item(),
        "std": gamma.std(unbiased=False).item(),
        "frac_positive": (gamma > 0).float().mean().item(),
        "frac_negative": (gamma < 0).float().mean().item(),
        "repeat_set_mean": diagonal_values.mean().item(),
        "repeat_set_frac_positive": (diagonal_values > 0).float().mean().item(),
        "distinct_set_mean": off_diagonal.mean().item(),
        "distinct_set_frac_positive": (off_diagonal > 0).float().mean().item(),
    }


def reference_sensitivity(costs: torch.Tensor, anchors: tuple[tuple[int, int], ...] = SENSITIVITY_ANCHORS,
                          eps: float = 1e-6) -> dict:
    """Gamma stability under anchor changes, on the given (held-out) costs.

    Per anchor: mean over states of Gamma, then across anchors: spread of the
    per-pair mean Gamma and the fraction of (a, b) pairs whose mean Gamma
    changes sign between anchors.  A Gamma that only exists for one anchor is
    a reference artifact, not an interaction.

    Sign flips are STRICT with a numerical zero band: a pair counts as a flip
    only if its mean Gamma exceeds ``eps`` under at least one anchor and falls
    below ``-eps`` under another.  Pairs whose mean Gamma ever lies inside the
    zero band [-eps, eps] are reported separately as uncertain/zero.  The
    historical inclusive definition (any >0 together with any <=0, so a
    positive-to-zero transition already counted as a flip) is retained under
    an explicit name for comparison with earlier outputs only.
    """
    per_anchor = []
    pair_means = []
    for a0, b0 in anchors:
        gamma = interaction_gamma(costs, a0, b0)
        mean_over_states = gamma.mean(dim=0)
        pair_means.append(mean_over_states)
        per_anchor.append({"a0": a0, "b0": b0, **gamma_summary(gamma)})
    stacked = torch.stack(pair_means)
    strictly_positive = stacked > eps
    strictly_negative = stacked < -eps
    strict_flip = strictly_positive.any(dim=0) & strictly_negative.any(dim=0)
    zero_band = (stacked.abs() <= eps).any(dim=0)
    inclusive_positive = stacked > 0
    inclusive_flip = inclusive_positive.any(dim=0) & (~inclusive_positive).any(dim=0)
    return {
        "anchors": per_anchor,
        "pair_mean_std_across_anchors": stacked.std(dim=0, unbiased=False).mean().item(),
        "pair_mean_abs_across_anchors": stacked.abs().mean().item(),
        "sign_flip_pair_fraction": strict_flip.float().mean().item(),
        "sign_flip_definition": ("strict: a pair flips only if some anchor has mean Gamma > eps "
                                 "and another has mean Gamma < -eps"),
        "sign_flip_eps": eps,
        "zero_or_uncertain_pair_fraction": zero_band.float().mean().item(),
        "inclusive_sign_flip_pair_fraction_positive_to_zero_counts": inclusive_flip.float().mean().item(),
    }


class AdditivePredictor(nn.Module):
    """L_hat(s, a, b) = u_a(s) + v_b(s); no interaction term."""

    def __init__(self, input_dim: int, n_actions: int) -> None:
        super().__init__()
        self.u = nn.Linear(input_dim, n_actions)
        self.v = nn.Linear(input_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.u(x).unsqueeze(-1) + self.v(x).unsqueeze(-2)


class LowRankInteractionPredictor(nn.Module):
    """L_hat(s, a, b) = u_a(s) + v_b(s) + p_a(s)^T q_b(s) with rank ``rank``."""

    def __init__(self, input_dim: int, n_actions: int, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.n_actions = n_actions
        self.rank = rank
        self.u = nn.Linear(input_dim, n_actions)
        self.v = nn.Linear(input_dim, n_actions)
        self.p = nn.Linear(input_dim, n_actions * rank)
        self.q = nn.Linear(input_dim, n_actions * rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[0]
        p = self.p(x).view(size, self.n_actions, self.rank)
        q = self.q(x).view(size, self.n_actions, self.rank)
        return self.u(x).unsqueeze(-1) + self.v(x).unsqueeze(-2) + torch.einsum("nar,nbr->nab", p, q)


class PairMLP(nn.Module):
    """Generic MLP predicting all A x A path costs (same skeleton as E01 students)."""

    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_actions * n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.n_actions, self.n_actions)


def mlp_hidden_dim_for_budget(input_dim: int, n_actions: int, budget: int) -> int:
    """Largest PairMLP hidden width whose parameter count stays within ``budget``."""

    def count(hidden_dim: int) -> int:
        probe = PairMLP(input_dim, n_actions, hidden_dim)
        return sum(parameter.numel() for parameter in probe.parameters())

    hidden_dim = 1
    while count(hidden_dim + 1) <= budget:
        hidden_dim += 1
    return hidden_dim


@dataclass
class PairPredictorResult:
    predictor: nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    history: list = field(default_factory=list)
    best_validation_mse: float = float("inf")
    optimizer_updates: int = 0

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.predictor.parameters())


def fit_pair_predictor(
    predictor: nn.Module,
    features: torch.Tensor,
    costs: torch.Tensor,
    *,
    epochs: int,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
    weight_decay: float = 0.0,
    validation: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> PairPredictorResult:
    """Fit one pair predictor by per-path squared loss; keep best-validation weights."""
    if features.shape[0] != costs.shape[0] or features.ndim != 2 or costs.ndim != 3:
        raise ValueError(f"bad shapes: features {tuple(features.shape)}, costs {tuple(costs.shape)}")
    device = features.device
    mean, std = standardization(features)
    x = (features - mean) / std
    y = costs.float().reshape(costs.shape[0], -1)
    torch.manual_seed(seed)
    predictor = predictor.to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = PairPredictorResult(predictor, mean, std)
    best_state = None
    validation_mse = float("nan")
    for epoch in range(epochs):
        order = torch.randperm(x.shape[0], generator=generator).to(device)
        total_loss = 0.0
        for start in range(0, x.shape[0], batch_size):
            indices = order[start : start + batch_size]
            prediction = predictor(x[indices])
            loss = F.mse_loss(prediction.reshape(indices.numel(), -1), y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            result.optimizer_updates += 1
            total_loss += loss.item() * indices.numel()
        train_mse = total_loss / x.shape[0]
        if validation is not None:
            validation_mse = pair_mse(result, validation[0], validation[1])
            if validation_mse < result.best_validation_mse:
                result.best_validation_mse = validation_mse
                best_state = {key: value.detach().clone() for key, value in predictor.state_dict().items()}
        result.history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": validation_mse})
    if best_state is not None:
        predictor.load_state_dict(best_state)
    return result


@torch.no_grad()
def predict_pairs(result: PairPredictorResult, features: torch.Tensor) -> torch.Tensor:
    result.predictor.eval()
    return result.predictor((features.float() - result.mean) / result.std)


@torch.no_grad()
def pair_mse(result: PairPredictorResult, features: torch.Tensor, costs: torch.Tensor) -> float:
    prediction = predict_pairs(result, features)
    return F.mse_loss(prediction.reshape(prediction.shape[0], -1), costs.float().reshape(costs.shape[0], -1)).item()


def argmin_pairs(prediction: torch.Tensor) -> torch.Tensor:
    """(n, A, A) predicted costs -> (n, 2) selected action indices."""
    flat = prediction.reshape(prediction.shape[0], -1)
    selected = flat.argmin(dim=-1)
    n_actions = prediction.shape[-1]
    return torch.stack([selected // n_actions, selected % n_actions], dim=-1)


def random_pair_selection(n_states: int, n_actions: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(n_actions, (n_states, 2), generator=generator).to(device)


def baseline_pair_selection(costs: PairPathCosts) -> torch.Tensor:
    """The ordinary fixed-depth policy: a* at round 1, pi0-after-a* at round 2."""
    rows = torch.arange(costs.baseline_action.shape[0], device=costs.baseline_action.device)
    second = costs.second_baseline[rows, costs.baseline_action]
    return torch.stack([costs.baseline_action, second], dim=-1)


def executed_pair_costs(costs: PairPathCosts, selection: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Measured (pair_q, pair_cost) of the selected pair per state — real execution."""
    if selection.shape != (costs.baseline_action.shape[0], 2):
        raise ValueError(f"selection has shape {tuple(selection.shape)}")
    rows = torch.arange(selection.shape[0], device=costs.pair_q.device)
    return costs.pair_q[rows, selection[:, 0], selection[:, 1]], costs.pair_cost[rows, selection[:, 0], selection[:, 1]]


def evaluate_pair_policies(costs: PairPathCosts, selections: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Held-out real executed cost of deployable label-free pair policies."""
    baseline, _ = executed_pair_costs(costs, baseline_pair_selection(costs))
    report = {}
    for name, selection in selections.items():
        executed, immediate = executed_pair_costs(costs, selection)
        improvement = baseline - executed
        count = max(1, improvement.numel())
        report[name] = {
            "mean_cost": executed.mean().item(),
            "mean_two_round_cost": immediate.mean().item(),
            "deploy_improvement": improvement.mean().item(),
            "deploy_improvement_se": (improvement.std(unbiased=False) / count**0.5).item(),
        }
    return report


def order_shuffle_control(costs: PairPathCosts, selection: torch.Tensor) -> dict:
    """Execute the swapped pair (b, a): matched calls and repetition, order broken."""
    swapped = torch.stack([selection[:, 1], selection[:, 0]], dim=-1)
    forward, _ = executed_pair_costs(costs, selection)
    backward, _ = executed_pair_costs(costs, swapped)
    gain = backward - forward
    count = max(1, gain.numel())
    return {
        "selected_mean_cost": forward.mean().item(),
        "swapped_mean_cost": backward.mean().item(),
        "mean_order_gain": gain.mean().item(),
        "mean_order_gain_se": (gain.std(unbiased=False) / count**0.5).item(),
    }


def oracle_pair_metrics(costs: PairPathCosts) -> dict:
    """Label-informed oracle pair gap; reported separately, never a deploy gain."""
    baseline, _ = executed_pair_costs(costs, baseline_pair_selection(costs))
    flat = costs.pair_q.reshape(costs.pair_q.shape[0], -1)
    best, selected = flat.min(dim=-1)
    gap = baseline - best
    count = max(1, gap.numel())
    n_actions = costs.pair_q.shape[-1]
    return {
        "mean_cost": best.mean().item(),
        "oracle_gap": gap.mean().item(),
        "oracle_gap_se": (gap.std(unbiased=False) / count**0.5).item(),
        "mean_oracle_pair_is_baseline": (
            (selected // n_actions == costs.baseline_action)
            & (selected % n_actions == costs.second_baseline[torch.arange(costs.pair_q.shape[0], device=costs.pair_q.device), costs.baseline_action])
        ).float().mean().item(),
        "deployable": False,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="default: <train.output_dir>/latest.pt from the config")
    parser.add_argument("--output-dir", default="runs/e10/pilot")
    parser.add_argument("--train-states", type=int, default=32)
    parser.add_argument("--validation-states", type=int, default=16)
    parser.add_argument("--test-states", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1, help="fixed depth of earlier adaptive layers, if any")
    parser.add_argument("--rank", type=int, default=4, help="interaction rank of the low-rank predictor")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--predictor-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seq-batch", type=int, default=2)
    parser.add_argument("--state-chunk", type=int, default=4)
    parser.add_argument("--readout-chunk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def save_records(path: Path, states: dict, costs: PairPathCosts, backbone_hash: str) -> None:
    with path.open("w") as handle:
        for index in range(states["h"].shape[0]):
            record = {
                "split": states["split"],
                "seq": int(states["seq_indices"][index]),
                "doc": int(states["doc_ids"][index]),
                "position": int(states["positions"][index]),
                "target": int(states["targets"][index]),
                "baseline_action": int(costs.baseline_action[index]),
                "second_baseline": [int(value) for value in costs.second_baseline[index].tolist()],
                "immediate": [round(value, 6) for value in costs.immediate[index].tolist()],
                "pair_q": [round(value, 6) for value in costs.pair_q[index].reshape(-1).tolist()],
                "backbone_hash": backbone_hash,
            }
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = LMExperimentConfig.load(args.config)
    config.save(output_dir / "resolved_config.json")
    checkpoint_path = Path(args.checkpoint or f"{config.train.output_dir}/latest.pt")
    backbone_hash = sha256_file(checkpoint_path)
    seed_everything(args.seed)
    device = torch.device(args.device)
    ledger = CounterfactualLedger()
    writer = MetricWriter(output_dir / "metrics.jsonl")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LoopMoE(config.model)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint_path),
        "backbone_hash": backbone_hash,
    }
    atomic_json(output_dir / "environment.json", environment)

    requested = {"train": args.train_states, "validation": args.validation_states, "test": args.test_states}
    states, costs = {}, {}
    for offset, (split, n_states) in enumerate(requested.items()):
        dataset = PackedLMDataset(config.train.data_dir, split)
        states[split] = gather_states(
            model, dataset, n_states, split=split, device=device, dtype=args.dtype,
            seed=args.seed + 101 * offset, seq_batch=args.seq_batch, depth=args.depth, ledger=ledger,
        )
        costs[split] = two_round_path_costs(
            model, states[split]["h"], states[split]["targets"],
            rounds=args.rounds, state_chunk=args.state_chunk, readout_chunk=args.readout_chunk,
            dtype=args.dtype, ledger=ledger,
        )
        torch.save(
            {**{key: value for key, value in states[split].items() if key != "split"},
             "split": split, "immediate": costs[split].immediate, "pair_cost": costs[split].pair_cost,
             "pair_q": costs[split].pair_q, "baseline_action": costs[split].baseline_action,
             "second_baseline": costs[split].second_baseline, "actions": costs[split].actions,
             "rounds": args.rounds, "backbone_hash": backbone_hash},
            output_dir / f"states_{split}.pt",
        )
        save_records(output_dir / f"records_{split}.jsonl", states[split], costs[split], backbone_hash)
        ledger.merge_peak(device)

    n_actions = costs["train"].actions.shape[0]
    input_dim = states["train"]["h"].shape[-1]
    train_x = student_features(states["train"])
    validation_x = student_features(states["validation"])
    test_x = student_features(states["test"])

    lowrank_probe = LowRankInteractionPredictor(input_dim, n_actions, args.rank)
    budget = sum(parameter.numel() for parameter in lowrank_probe.parameters())
    mlp_hidden = mlp_hidden_dim_for_budget(input_dim, n_actions, budget)
    predictors = {
        "additive": AdditivePredictor(input_dim, n_actions),
        "lowrank": lowrank_probe,
        "mlp": PairMLP(input_dim, n_actions, mlp_hidden),
    }
    results = {}
    for name, predictor in predictors.items():
        result = fit_pair_predictor(
            predictor, train_x, costs["train"].pair_q,
            epochs=args.epochs, lr=args.lr, batch_size=args.predictor_batch_size, seed=args.seed,
            validation=(validation_x, costs["validation"].pair_q),
        )
        ledger.student_optimizer_updates += result.optimizer_updates
        ledger.student_train_samples += train_x.shape[0] * args.epochs
        results[name] = result
        writer.write(phase="predictor", predictor=name, parameters=result.parameter_count,
                     best_validation_mse=result.best_validation_mse)
    torch.save(
        {name: {"state_dict": result.predictor.state_dict(), "mean": result.mean, "std": result.std,
                "class": type(result.predictor).__name__, "parameter_count": result.parameter_count,
                "rank": args.rank, "mlp_hidden_dim": mlp_hidden, "history": result.history}
         for name, result in results.items()},
        output_dir / "predictors.pt",
    )

    test_costs = costs["test"]
    n_test = test_costs.pair_q.shape[0]
    selections = {"baseline": baseline_pair_selection(test_costs),
                  "random": random_pair_selection(n_test, n_actions, args.seed, device)}
    for name, result in results.items():
        selections[name] = argmin_pairs(predict_pairs(result, test_x))
    policies = evaluate_pair_policies(test_costs, selections)
    shuffle_controls = {name: order_shuffle_control(test_costs, selections[name]) for name in ("additive", "lowrank", "mlp")}
    oracle = oracle_pair_metrics(test_costs)
    predictor_mse = {name: pair_mse(result, test_x, test_costs.pair_q) for name, result in results.items()}
    a0, b0 = PRIMARY_ANCHOR
    gamma = interaction_gamma(test_costs.pair_q, a0, b0)
    gamma_two_round = interaction_gamma(test_costs.pair_cost, a0, b0)
    sensitivity = reference_sensitivity(test_costs.pair_q)
    order = order_effect(test_costs.pair_q)
    writer.write(phase="heldout", **{f"policy_{key}": value["mean_cost"] for key, value in policies.items()})
    writer.write(phase="heldout_oracle", **oracle)

    ledger.elapsed_seconds = time.monotonic() - started
    ledger.merge_peak(device)
    ledger.save(output_dir / "cost_ledger.json")

    summary = {
        "experiment": "E10 ordered complementarity: exact two-round Top-2 path costs and interaction predictability",
        "checkpoint": str(checkpoint_path),
        "backbone_hash": backbone_hash,
        "rounds": args.rounds,
        "n_actions": n_actions,
        "n_states": requested,
        "matched_compute": "every path forces rounds 1-2 and follows pi0 to T=rounds; all Gamma terms use the same number of expert calls",
        "primary_anchor": {"a0": a0, "b0": b0},
        "gamma_t_final": gamma_summary(gamma),
        "gamma_two_round": gamma_summary(gamma_two_round),
        "order_effect": {
            "mean_abs": order.abs().mean().item(),
            "std": order.std(unbiased=False).item(),
        },
        "reference_sensitivity": sensitivity,
        "policies": policies,
        "order_shuffle_controls": shuffle_controls,
        "oracle": {**oracle, "note": "label-informed lower bound on cost; reported separately and never counted as deploy improvement"},
        "predictors": {
            name: {
                "test_mse": predictor_mse[name],
                "best_validation_mse": result.best_validation_mse,
                "parameter_count": result.parameter_count,
                "parameter_budget": budget if name in ("lowrank", "mlp") else result.parameter_count,
            }
            for name, result in results.items()
        },
        "interpretation_note": "Gamma > 0 is relative to the anchor, input distribution and loss scale; a nonzero Gamma distribution is neither predictability nor deploy gain by itself",
        "ledger": ledger.__dict__,
    }
    atomic_json(output_dir / "e10_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
