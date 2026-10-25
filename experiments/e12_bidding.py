"""E12 nonlinear bidding: central MLP vs per-expert conditional-bid adapters.

Follow-up to the linear equivalence screen (``e12_equivalence.py``): per the
E12 plan, linear per-expert bids are a central linear router, so this script
compares *nonlinear* cost predictors on the E01 same-hash labels:

* ``central`` — one MLP reads the legal deploy-time input ``h`` and predicts
  all 28 candidate Top-2 costs (the E01 value-student architecture).
* ``bidding`` — 8 per-expert small adapter MLPs.  An anchor expert is first
  fixed by a legal h-rule; adapter ``e`` then reads ``h`` plus the anchor
  one-hot and predicts the cost of pair ``{anchor, e}`` (conditional marginal
  bid, in CE units — the same labels as E01).  Only 7 bids per state are
  deployed; no expert output is executed to produce a bid.
* ``bidding_summary`` — same adapters with an extra per-expert summary input
  (learned embedding by default, or real expert-weight statistics with
  ``--summary-mode checkpoint --checkpoint ...``).
* ``central_irrelevant`` — central MLP with irrelevant fields (position, doc,
  sequence index) appended, as an input-dimension control.

All arms use the same candidates, the same E01 labels, and the same training
budget (epochs, lr, batch-over-states, optimizer, seed).  Parameters are
matched as closely as integer hidden widths allow and the exact difference is
reported.  Deployment selections are really executed against the measured
path costs.  This is a joint supervised prediction problem with a shared
objective: no incentive compatibility, dominant strategy, or market-mechanism
claim is made anywhere.

Example:

    python -m experiments.e12_bidding \
        --input-dir runs/e01/coverage_remediation \
        --output-dir runs/e12/bidding
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from eval.counterfactual import (
    PathCosts,
    evaluate_policies,
    oracle_metrics,
    random_selection,
)
from train.value_student import fit_value_student, mse, predict, standardization

SPLITS = ("train", "validation", "test")


# ---------------------------------------------------------------- data

def load_states(input_dir: str | Path) -> dict[str, dict]:
    """Load E01 states splits; verify they share one backbone hash."""
    input_dir = Path(input_dir)
    states = {}
    for split in SPLITS:
        path = input_dir / f"states_{split}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run run_e01.py first")
        states[split] = torch.load(path, map_location="cpu", weights_only=False)
    hashes = {split: states[split]["backbone_hash"] for split in SPLITS}
    if len(set(hashes.values())) != 1:
        raise ValueError(f"backbone hash mismatch across splits: {hashes}")
    reference = states["train"]
    for split in SPLITS:
        if not torch.equal(states[split]["actions"], reference["actions"]):
            raise ValueError(f"candidate action table of split {split!r} differs from train")
        if states[split]["rounds"] != reference["rounds"]:
            raise ValueError(f"rounds of split {split!r} differs from train")
    return states


def irrelevant_features(states: dict) -> torch.Tensor:
    """Fields with no causal role in the label; input-dimension control only."""
    return torch.stack(
        [
            states["positions"].float(),
            states["doc_ids"].float(),
            states["seq_indices"].float(),
        ],
        dim=-1,
    )


def pair_action_matrix(experts: int) -> torch.Tensor:
    """(experts, experts) action index of pair {a, e}; -1 on the diagonal."""
    matrix = torch.full((experts, experts), -1, dtype=torch.long)
    for anchor in range(experts):
        for second in range(experts):
            if anchor == second:
                continue
            low, high = min(anchor, second), max(anchor, second)
            matrix[anchor, second] = low * experts - low * (low + 1) // 2 + (high - low - 1)
    return matrix


# ---------------------------------------------------------------- models

class ExpertBidAdapters(nn.Module):
    """One small MLP per expert: [h, one_hot(anchor), summary_e] -> bid for {anchor, e}.

    Parameters are stored per expert (``w1[e]`` is expert ``e``'s own first
    layer), so each adapter is an independent module that happens to be
    evaluated in one batched GPU kernel — no inter-expert communication.
    """

    def __init__(
        self,
        h_dim: int,
        experts: int,
        hidden_dim: int,
        summary_mode: str = "none",
        summary_dim: int = 9,
        checkpoint_summary: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.experts = experts
        self.summary_mode = summary_mode
        if summary_mode == "checkpoint":
            if checkpoint_summary is None or checkpoint_summary.shape[0] != experts:
                raise ValueError("checkpoint summary mode requires a (experts, S) summary tensor")
            self.register_buffer("summary", checkpoint_summary.float())
        elif summary_mode == "learned":
            self.summary = nn.Parameter(torch.randn(experts, summary_dim) * 0.1)
        elif summary_mode == "none":
            self.summary = None
        else:
            raise ValueError(f"unknown summary mode {summary_mode!r}")
        summary_size = 0 if self.summary is None else self.summary.shape[1]
        input_dim = h_dim + experts + summary_size
        self.input_dim = input_dim
        self.w1 = nn.Parameter(torch.randn(experts, input_dim, hidden_dim) * input_dim**-0.5)
        self.b1 = nn.Parameter(torch.zeros(experts, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(experts, hidden_dim, hidden_dim) * hidden_dim**-0.5)
        self.b2 = nn.Parameter(torch.zeros(experts, hidden_dim))
        self.w3 = nn.Parameter(torch.randn(experts, hidden_dim, 1) * hidden_dim**-0.5)
        self.b3 = nn.Parameter(torch.zeros(experts, 1))

    def _apply(self, rows: torch.Tensor) -> torch.Tensor:
        """rows (..., experts, input_dim) -> bids (..., experts)."""
        hidden = F.gelu(torch.einsum("...ei,oih->...oh", rows, self.w1) + self.b1)
        hidden = F.gelu(torch.einsum("...oh,ohk->...ok", hidden, self.w2) + self.b2)
        return (torch.einsum("...ok,okj->...oj", hidden, self.w3) + self.b3).squeeze(-1)

    def _rows(self, h: torch.Tensor, anchor_onehot: torch.Tensor) -> torch.Tensor:
        """h (B, h_dim), anchor_onehot (..., experts) -> rows (..., experts, input_dim)."""
        experts = self.experts
        shape = anchor_onehot.shape[:-1] + (experts,)
        parts = [
            h.reshape(h.shape[0], *([1] * (len(shape) - 1)), h.shape[1]).expand(*shape, h.shape[1]),
            anchor_onehot.unsqueeze(-2).expand(*shape, experts),
        ]
        if self.summary is not None:
            parts.append(self.summary.reshape(*([1] * (len(shape) - 1)), experts, -1).expand(*shape, self.summary.shape[1]))
        return torch.cat(parts, dim=-1)

    def bids_all_anchors(self, h: torch.Tensor) -> torch.Tensor:
        """Bids for every (anchor, second) orientation: (B, experts, experts)."""
        eye = torch.eye(self.experts, device=h.device, dtype=h.dtype)
        anchor_onehot = eye.unsqueeze(0).expand(h.shape[0], self.experts, self.experts)
        return self._apply(self._rows(h, anchor_onehot))

    def bids_for_anchor(self, h: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Conditional bids of every second expert given anchors: (B, experts)."""
        anchor_onehot = F.one_hot(anchors, self.experts).to(h.dtype)
        return self._apply(self._rows(h, anchor_onehot))


def central_parameter_count(input_dim: int, hidden_dim: int, n_actions: int) -> int:
    return (input_dim + 1) * hidden_dim + (hidden_dim + 1) * hidden_dim + (hidden_dim + 1) * n_actions


def adapter_parameter_count(input_dim: int, hidden_dim: int) -> int:
    return (input_dim + 1) * hidden_dim + (hidden_dim + 1) * hidden_dim + (hidden_dim + 1) * 1


def match_adapter_hidden(central_params: int, adapter_input_dim: int, experts: int, cap: int = 4096) -> int:
    """Integer hidden width bringing experts x adapter params closest to the central MLP."""
    return min(
        range(1, cap + 1),
        key=lambda width: (abs(experts * adapter_parameter_count(adapter_input_dim, width) - central_params), width),
    )


def mlp_macs(input_dim: int, hidden_dim: int, output_dim: int) -> int:
    """Multiply-accumulates of the two-hidden-layer MLP forward (1 MAC ~ 2 FLOPs)."""
    return input_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * output_dim


# ---------------------------------------------------------------- training

def fit_anchor_scorer(
    h: torch.Tensor,
    costs: torch.Tensor,
    actions: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    validation: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """Linear h-rule picking the anchor expert.

    Soft-target cross-entropy against the two members of the train state's
    cheapest pair (label-informed target construction is train-time only;
    deployment reads h alone).  Same optimizer/schedule as the other arms.
    """
    experts = actions.max().item() + 1
    mean, std = standardization(h)
    x = (h - mean) / std
    best_pairs = actions[costs.argmin(dim=-1)]
    target = torch.zeros(h.shape[0], experts).scatter_(1, best_pairs, 0.5)
    torch.manual_seed(seed)
    scorer = nn.Linear(h.shape[1], experts)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history, updates = [], 0
    best_loss, best_state = float("inf"), None

    def soft_ce(features: torch.Tensor) -> float:
        with torch.no_grad():
            pairs = actions[features[1].argmin(dim=-1)]
            targets = torch.zeros(features[1].shape[0], experts).scatter_(1, pairs, 0.5)
            logits = scorer((features[0] - mean) / std)
            return -(targets * F.log_softmax(logits, dim=-1)).sum(-1).mean().item()

    for epoch in range(epochs):
        order = torch.randperm(x.shape[0], generator=generator)
        total = 0.0
        for start in range(0, x.shape[0], batch_size):
            indices = order[start : start + batch_size]
            logits = scorer(x[indices])
            loss = -(target[indices] * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            total += loss.item() * indices.numel()
        validation_loss = soft_ce(validation) if validation is not None else float("nan")
        if validation is not None and validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in scorer.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": total / x.shape[0], "validation_loss": validation_loss})
    if best_state is not None:
        scorer.load_state_dict(best_state)
    return {"scorer": scorer, "mean": mean, "std": std, "history": history,
            "best_validation_loss": best_loss, "optimizer_updates": updates}


def fit_bid_adapters(
    h: torch.Tensor,
    costs: torch.Tensor,
    pa: torch.Tensor,
    *,
    hidden_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    summary_mode: str = "none",
    summary_dim: int = 9,
    checkpoint_summary: torch.Tensor | None = None,
    validation: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """Fit per-expert bid adapters by MSE on conditional pair costs.

    Mirrors ``fit_value_student``: same standardization, optimizer, lr, seed,
    epochs and batch-over-states schedule, best-validation weight selection.
    Each state contributes every (anchor, second) orientation so the adapters
    stay calibrated for whichever anchor the legal h-rule picks at deployment;
    supervision still comes only from the same per-pair E01 labels.
    """
    experts = pa.shape[0]
    off_diagonal = ~torch.eye(experts, dtype=torch.bool)
    mean, std = standardization(h)
    x = (h - mean) / std
    targets = costs.float()[:, pa.clamp(min=0)]  # (N, anchor, second); diagonal unused
    torch.manual_seed(seed)
    adapters = ExpertBidAdapters(
        h.shape[1], experts, hidden_dim, summary_mode, summary_dim, checkpoint_summary
    )
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history, updates = [], 0
    best_mse, best_state = float("inf"), None

    @torch.no_grad()
    def full_mse(features: torch.Tensor, labels: torch.Tensor) -> float:
        bids = adapters.bids_all_anchors((features - mean) / std)
        goal = labels.float()[:, pa.clamp(min=0)]
        return (bids - goal).square()[:, off_diagonal].mean().item()

    for epoch in range(epochs):
        order = torch.randperm(x.shape[0], generator=generator)
        total = 0.0
        for start in range(0, x.shape[0], batch_size):
            indices = order[start : start + batch_size]
            bids = adapters.bids_all_anchors(x[indices])
            loss = (bids - targets[indices]).square()[:, off_diagonal].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            total += loss.item() * indices.numel()
        validation_mse = full_mse(*validation) if validation is not None else float("nan")
        if validation is not None and validation_mse < best_mse:
            best_mse = validation_mse
            best_state = {key: value.detach().clone() for key, value in adapters.state_dict().items()}
        history.append({"epoch": epoch, "train_mse": total / x.shape[0], "validation_mse": validation_mse})
    if best_state is not None:
        adapters.load_state_dict(best_state)
    return {"adapters": adapters, "mean": mean, "std": std, "history": history,
            "best_validation_mse": best_mse, "optimizer_updates": updates}


# ---------------------------------------------------------------- deployment

@torch.no_grad()
def bidding_actions(
    adapters: ExpertBidAdapters,
    h: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    anchors: torch.Tensor,
    pa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the cheapest second expert given anchors; return (actions, seconds)."""
    bids = adapters.bids_for_anchor((h - mean) / std, anchors)
    bids[torch.arange(h.shape[0]), anchors] = float("inf")
    seconds = bids.argmin(dim=-1)
    return pa[anchors, seconds], seconds


def checkpoint_expert_summary(checkpoint_path: str | Path, experts: int) -> torch.Tensor:
    """Per-expert weight statistics of the last adaptive layer's expert pool.

    Nine raw statistics (norm/mean/std of w1, w3, w2) per expert, standardized
    across experts per dimension.  Fixed at load time; not learned.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    layers = set()
    for key in state:
        match = re.match(r"blocks\.(\d+)\.executor\.moe\.pool\.(w1|w2|w3)$", key)
        if match:
            layers.add(int(match.group(1)))
    if not layers:
        raise ValueError(f"no expert pool weights found in {checkpoint_path}")
    layer = max(layers)
    pieces = []
    for name in ("w1", "w3", "w2"):
        weight = state[f"blocks.{layer}.executor.moe.pool.{name}"].float()
        if weight.shape[0] != experts:
            raise ValueError(f"checkpoint has {weight.shape[0]} experts, states have {experts}")
        pieces.append(torch.stack([weight.norm(dim=(1, 2)), weight.mean(dim=(1, 2)), weight.std(dim=(1, 2))], dim=-1))
    summary = torch.cat(pieces, dim=-1)
    return (summary - summary.mean(dim=0, keepdim=True)) / summary.std(dim=0, keepdim=True).clamp_min(1e-6)


# ---------------------------------------------------------------- experiment

def run_experiment(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    hidden_dim: int = 128,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    label: str = "q_pi0",
    summary_mode: str = "learned",
    summary_dim: int = 9,
    checkpoint: str | Path | None = None,
) -> dict:
    started = time.monotonic()
    states = load_states(input_dir)
    if label not in ("q_pi0", "immediate"):
        raise ValueError(f"unknown label {label!r}")
    backbone_hash = states["train"]["backbone_hash"]
    actions = states["train"]["actions"]
    experts = int(actions.max().item()) + 1
    n_actions = actions.shape[0]
    pa = pair_action_matrix(experts)
    h = {split: states[split]["h"].float() for split in SPLITS}
    y = {split: states[split][label].float() for split in SPLITS}

    budget = {"epochs": epochs, "lr": lr, "batch_size": batch_size, "seed": seed,
              "optimizer": "AdamW", "schedule": "batch-over-states, shared by all arms"}
    report_systems: dict[str, dict] = {}
    runtimes: dict[str, float] = {}

    # ---- arm 1: central MLP on legal h only
    tic = time.monotonic()
    central = fit_value_student(
        h["train"], y["train"], hidden_dim=hidden_dim, epochs=epochs, lr=lr,
        batch_size=batch_size, seed=seed, validation=(h["validation"], y["validation"]),
    )
    runtimes["central"] = time.monotonic() - tic
    central_params = sum(p.numel() for p in central.student.parameters())
    report_systems["central"] = {
        "architecture": "single MLP h -> all-candidate costs (E01 value-student architecture)",
        "input_dim": h["train"].shape[1], "hidden_dim": hidden_dim,
        "parameters": central_params,
        "macs_per_state": mlp_macs(h["train"].shape[1], hidden_dim, n_actions),
        "best_validation_mse": central.best_validation_mse,
        "optimizer_updates": central.optimizer_updates,
        "test_mse_all_candidates": mse(central.student, h["test"], y["test"], central.mean, central.std),
    }

    # ---- anchor rules (legal functions of h; scorer trained on train split only)
    tic = time.monotonic()
    anchor_fit = fit_anchor_scorer(
        h["train"], y["train"], actions, epochs=epochs, lr=lr, batch_size=batch_size,
        seed=seed, validation=(h["validation"], y["validation"]),
    )
    runtimes["anchor_scorer"] = time.monotonic() - tic
    scorer_params = sum(p.numel() for p in anchor_fit["scorer"].parameters())
    with torch.no_grad():
        learned_anchor = anchor_fit["scorer"]((h["test"] - anchor_fit["mean"]) / anchor_fit["std"]).argmax(dim=-1)
    baseline_anchor = actions[states["test"]["baseline_action"]][:, 0]
    anchor_rules = {"learned": learned_anchor, "baseline_pair_first": baseline_anchor}

    # ---- arms 2/3: per-expert bid adapters, without and with expert summary
    if summary_mode == "checkpoint":
        if checkpoint is None:
            raise ValueError("--summary-mode checkpoint requires --checkpoint")
        checkpoint_summary = checkpoint_expert_summary(checkpoint, experts)
        summary_size = checkpoint_summary.shape[1]
    elif summary_mode == "learned":
        checkpoint_summary, summary_size = None, summary_dim
    else:
        raise ValueError(f"unknown summary mode {summary_mode!r}")

    bidding_results: dict[str, dict] = {}
    for arm, mode, size in (("bidding", "none", 0), ("bidding_summary", summary_mode, summary_size)):
        tic = time.monotonic()
        adapter_input = h["train"].shape[1] + experts + size
        adapter_hidden = match_adapter_hidden(central_params, adapter_input, experts)
        fit = fit_bid_adapters(
            h["train"], y["train"], pa, hidden_dim=adapter_hidden, epochs=epochs, lr=lr,
            batch_size=batch_size, seed=seed, summary_mode=mode, summary_dim=summary_dim,
            checkpoint_summary=checkpoint_summary, validation=(h["validation"], y["validation"]),
        )
        runtimes[arm] = time.monotonic() - tic
        bidding_results[arm] = {"fit": fit, "hidden": adapter_hidden, "input": adapter_input}

    # ---- arm 4: central MLP + irrelevant fields (input-dimension control)
    tic = time.monotonic()
    extra = {split: irrelevant_features(states[split]) for split in SPLITS}
    wide = {split: torch.cat([h[split], extra[split]], dim=-1) for split in SPLITS}
    control = fit_value_student(
        wide["train"], y["train"], hidden_dim=hidden_dim, epochs=epochs, lr=lr,
        batch_size=batch_size, seed=seed, validation=(wide["validation"], y["validation"]),
    )
    runtimes["central_irrelevant"] = time.monotonic() - tic
    report_systems["central_irrelevant"] = {
        "architecture": "central MLP with irrelevant fields (position, doc, sequence) appended",
        "input_dim": wide["train"].shape[1], "hidden_dim": hidden_dim,
        "parameters": sum(p.numel() for p in control.student.parameters()),
        "macs_per_state": mlp_macs(wide["train"].shape[1], hidden_dim, n_actions),
        "best_validation_mse": control.best_validation_mse,
        "optimizer_updates": control.optimizer_updates,
        "test_mse_all_candidates": mse(control.student, wide["test"], y["test"], control.mean, control.std),
    }

    # ---- MSE comparison on identical cells
    off_diagonal = ~torch.eye(experts, dtype=torch.bool)
    with torch.no_grad():
        central_test = predict(central.student, h["test"], central.mean, central.std)
    goal = y["test"][:, pa.clamp(min=0)]
    central_on_cells = (central_test[:, pa.clamp(min=0)] - goal).square()[:, off_diagonal].mean().item()
    mse_comparison = {
        "cells": "every ordered (anchor, second) orientation of all 28 test pairs; "
                 "central predictions gathered on the same cells (each pair counted twice in both arms)",
        "central_on_same_cells": central_on_cells,
    }
    for arm, record in bidding_results.items():
        fit = record["fit"]
        with torch.no_grad():
            bids = fit["adapters"].bids_all_anchors((h["test"] - fit["mean"]) / fit["std"])
        mse_comparison[f"{arm}_on_same_cells"] = (bids - goal).square()[:, off_diagonal].mean().item()
        adapter_params = sum(p.numel() for p in fit["adapters"].parameters())
        deployed_macs = mlp_macs(h["train"].shape[1], experts, experts) + (experts - 1) * mlp_macs(record["input"], record["hidden"], 1)
        report_systems[arm] = {
            "architecture": f"{experts} per-expert adapter MLPs; conditional bid for pair {{anchor, e}}",
            "input_dim": record["input"], "hidden_dim": record["hidden"],
            "summary_mode": mode if arm == "bidding_summary" else "none",
            "parameters": adapter_params,
            "macs_per_state_deployed": deployed_macs,
            "macs_note": "anchor scorer + (experts-1) adapter forwards; batched implementation computes one extra masked diagonal bid",
            "best_validation_mse": fit["best_validation_mse"],
            "optimizer_updates": fit["optimizer_updates"],
        }

    # ---- deployment: selections really executed against measured path costs
    test_costs = PathCosts(
        states["test"]["immediate"], states["test"]["q_pi0"],
        states["test"]["baseline_action"], actions, states["test"]["rounds"],
    )
    n_test = y["test"].shape[0]
    selections = {
        "baseline": states["test"]["baseline_action"],
        "random": random_selection(n_test, n_actions, seed, torch.device("cpu")),
        "central_argmin": central_test.argmin(dim=-1),
        "central_irrelevant_argmin": predict(control.student, wide["test"], control.mean, control.std).argmin(dim=-1),
    }
    deployed_cell_mse: dict[str, dict] = {}
    for rule_name, anchors in anchor_rules.items():
        for arm, record in bidding_results.items():
            fit = record["fit"]
            selected, _ = bidding_actions(fit["adapters"], h["test"], fit["mean"], fit["std"], anchors, pa)
            selections[f"{arm}_{rule_name}_anchor"] = selected
            cells = pa[anchors.unsqueeze(-1), torch.arange(experts).unsqueeze(0)]
            valid = cells >= 0
            bids = fit["adapters"].bids_for_anchor((h["test"] - fit["mean"]) / fit["std"], anchors)
            bid_cell_mse = ((bids - y["test"].gather(1, cells.clamp(min=0))).square()[valid]).mean().item()
            central_cell_mse = ((central_test.gather(1, cells.clamp(min=0)) - y["test"].gather(1, cells.clamp(min=0))).square()[valid]).mean().item()
            deployed_cell_mse[f"{arm}_{rule_name}_anchor"] = {
                "bidding_mse_on_deployed_cells": bid_cell_mse,
                "central_mse_on_same_deployed_cells": central_cell_mse,
            }
    policies = evaluate_policies(test_costs, selections)
    oracle = oracle_metrics(test_costs)

    oracle_pairs = actions[y["test"].argmin(dim=-1)]
    anchor_diagnostics = {
        "rule_agreement_fraction": (learned_anchor == baseline_anchor).float().mean().item(),
        "learned_anchor_in_label_informed_best_pair_fraction": (oracle_pairs == learned_anchor.unsqueeze(-1)).any(dim=-1).float().mean().item(),
        "note": "label-informed diagnostics, reported separately; never counted as deploy improvement",
    }

    parameter_matching = {
        "central_parameters": central_params,
        "note": "integer hidden widths cannot match exactly; differences reported, not hidden",
    }
    for arm, record in bidding_results.items():
        adapter_params = report_systems[arm]["parameters"]
        parameter_matching[arm] = {
            "adapter_parameters_total": adapter_params,
            "difference_vs_central": adapter_params - central_params,
            "relative_difference": (adapter_params - central_params) / central_params,
            "anchor_scorer_parameters_extra": scorer_params,
            "matched_hidden_dim": record["hidden"],
        }

    summary = {
        "experiment": "E12 nonlinear bidding: central MLP vs per-expert conditional-bid adapters",
        "scope": "capacity/input-matched supervised comparison; NOT a market mechanism and no incentive-compatibility claim",
        "backbone_hash": backbone_hash,
        "label": label,
        "rounds": states["train"]["rounds"],
        "n_states": {split: int(h[split].shape[0]) for split in SPLITS},
        "n_candidates": n_actions,
        "experts": experts,
        "training_budget": budget,
        "label_scalars_per_state_per_epoch": {
            "central": n_actions,
            "bidding": experts * (experts - 1),
            "bidding_note": "each pair supervised once per orientation (56 = 2 x 28 for 8 experts) so adapters stay calibrated for any legal anchor rule",
        },
        "systems": report_systems,
        "parameter_matching": parameter_matching,
        "input_matching": {
            "central_input_dim": h["train"].shape[1],
            "bidding_input_dim": bidding_results["bidding"]["input"],
            "bidding_summary_input_dim": bidding_results["bidding_summary"]["input"],
            "difference": "adapters additionally read the anchor one-hot (and expert summary in bidding_summary); central reads h only",
        },
        "mse_comparison": mse_comparison,
        "deployed_cell_mse": deployed_cell_mse,
        "policies": policies,
        "oracle": {**oracle, "note": "label-informed lower bound; reported separately, never a deploy gain"},
        "anchor_rules": {
            "learned": "linear scorer on h, trained on train split only; counted as control overhead",
            "baseline_pair_first": "first-listed member of the frozen baseline router's Top-2 pair (a legal function of h); zero parameters",
            "diagnostics": anchor_diagnostics,
        },
        "control_overhead": {
            "bids_collected_per_state": {"central": 0, "bidding": experts - 1},
            "scores_computed_per_state": {"central": n_actions, "bidding": experts - 1},
            "anchor_scorer": {"parameters": scorer_params, "macs_per_state": mlp_macs(h["train"].shape[1], experts, experts),
                              "optimizer_updates": anchor_fit["optimizer_updates"]},
            "optimizer_updates": {arm: report_systems[arm]["optimizer_updates"] for arm in report_systems},
            "runtime_seconds": {key: round(value, 3) for key, value in runtimes.items()},
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "e12_bidding_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="E01 output dir containing states_{train,validation,test}.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=128, help="central MLP hidden width; adapter width is matched")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", choices=("q_pi0", "immediate"), default="q_pi0")
    parser.add_argument("--summary-mode", choices=("learned", "checkpoint"), default="learned")
    parser.add_argument("--summary-dim", type=int, default=9)
    parser.add_argument("--checkpoint", help="E00 checkpoint for --summary-mode checkpoint")
    args = parser.parse_args()
    summary = run_experiment(
        args.input_dir, args.output_dir, hidden_dim=args.hidden_dim, epochs=args.epochs,
        lr=args.lr, batch_size=args.batch_size, seed=args.seed, label=args.label,
        summary_mode=args.summary_mode, summary_dim=args.summary_dim, checkpoint=args.checkpoint,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
