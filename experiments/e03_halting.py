"""E03 pilot: value-calibrated halting from frozen E01 artifacts.

This pilot reuses the E01 ``coverage_remediation`` artifacts as-is:

* ``states_{train,validation,test}.pt`` carry the cached hidden state ``h``
  (the only legal deploy-time input), the measured per-action FP32 CE labels
  ``immediate`` (stop after T=1) and ``q_pi0`` (continue to T=rounds with the
  frozen baseline policy pi0), the baseline action, and the backbone hash.
* ``students.pt`` carries the two frozen E01 value students (equal-capacity
  MLPs trained on the train split only, input ``h`` only).

Nothing is relabeled and no backbone/head forward is re-executed here: the
stop/continue costs are read from the stored label matrices at the baseline
action, after verifying that every split carries the same backbone hash.

Because the E01 labels exist for exactly two depths (T=1 and T=rounds), the
halting decision in this pilot is binary: stop after round 1, or continue
with pi0 to round ``rounds``.  Intermediate-depth stopping labels do not
exist in the artifacts and are not fabricated.

Compared halt signals:

* fixed T1 / fixed T4 (no signal),
* value students: predicted stop cost and predicted continue cost at the
  baseline action, continue iff ``delta_hat = stop_pred - continue_pred >
  tau`` (tau absorbs lambda * extra-continuation-cost, per the E03 spec),
* entropy / max-prob / logit-margin / hidden-change thresholds: checked
  against the loaded artifacts; reported as unavailable with an explicit
  reason when the required quantities are absent — never fabricated.

The threshold tau is calibrated on the validation split for each target
average-rounds budget, frozen, and only then evaluated on the test split.
The label-informed oracle is reported in a clearly separated section and is
never mixed into the deployable (student) results.

Example:

    python -m experiments.e03_halting \
        --input-dir runs/e01/coverage_remediation --output-dir runs/e03/pilot
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from train.recovery_common import atomic_json
from train.value_student import ValueStudent, predict

NLL_TOLERANCE = 0.005
DEFAULT_BUDGETS = (1.5, 2.0, 3.0, 4.0)
SPLITS = ("train", "validation", "test")

# Cheap-threshold signals from the full E03 design, what they require, and
# which artifact key would provide them.  Availability is checked against the
# actually loaded state dicts; a missing signal is reported unavailable with
# its reason and is never fabricated.
CHEAP_SIGNALS = {
    "entropy": {
        "key": "entropy",
        "direction": "continue iff entropy > tau",
        "missing_reason": "full-vocab head logits per state were never materialized by E01 "
                          "(see eval/counterfactual.py); states_*.pt store only h, targets and CE labels, "
                          "so entropy cannot be computed without re-executing the head, which is out of "
                          "scope for this reuse-only pilot",
    },
    "maxprob": {
        "key": "maxprob",
        "direction": "continue iff -maxprob > tau (low confidence continues)",
        "missing_reason": "full-vocab head probabilities per state are not stored in the E01 artifacts "
                          "and are not recomputed here",
    },
    "logit_margin": {
        "key": "logit_margin",
        "direction": "continue iff -margin > tau (small top1-top2 margin continues)",
        "missing_reason": "top-2 vocabulary logits per state are not stored in the E01 artifacts "
                          "and are not recomputed here",
    },
    "hidden_change": {
        "key": "hidden_change",
        "direction": "continue iff ||h_t - h_{t-1}|| > tau",
        "missing_reason": "only the pre-FFN h entering the adaptive layer is cached; no per-round hidden "
                          "states exist in the artifacts",
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="runs/e01/coverage_remediation")
    parser.add_argument("--output-dir", default="runs/e03/pilot")
    parser.add_argument("--budgets", type=float, nargs="+", default=list(DEFAULT_BUDGETS),
                        help="target average rounds per state (default: 1.5 2 3 4)")
    parser.add_argument("--nll-tolerance", type=float, default=NLL_TOLERANCE,
                        help="allowed mean NLL increase vs fixed T4, nat/token")
    parser.add_argument("--tau-grid-size", type=int, default=400,
                        help="number of candidate thresholds from validation quantiles")
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=2, help="experts called per round (Top-2)")
    parser.add_argument("--max-states", type=int, default=0,
                        help="cap states per split for debugging (0 = use all)")
    return parser.parse_args(argv)


def load_split(input_dir: Path, split: str, max_states: int = 0) -> dict:
    path = input_dir / f"states_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing E01 states file: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = {"h", "targets", "immediate", "q_pi0", "baseline_action", "actions", "rounds", "backbone_hash"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    if max_states and data["h"].shape[0] > max_states:
        n = data["h"].shape[0]
        data = {key: (value[:max_states] if key != "actions" and torch.is_tensor(value)
                      and value.ndim >= 1 and value.shape[0] == n else value)
                for key, value in data.items()}
    return data


def load_students(input_dir: Path) -> dict:
    path = input_dir / "students.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing E01 students file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    students = {}
    for name in ("immediate", "q_pi0"):
        entry = payload[name]
        student = ValueStudent(entry["mean"].shape[1], entry["state_dict"]["net.4.bias"].shape[0], entry["hidden_dim"])
        student.load_state_dict(entry["state_dict"])
        student.eval()
        students[name] = {"student": student, "mean": entry["mean"], "std": entry["std"]}
    return students


def load_artifacts(input_dir: str | Path, max_states: int = 0) -> tuple[dict, dict, str]:
    """Load the three state splits and the frozen students; verify one shared hash."""
    input_dir = Path(input_dir)
    splits = {split: load_split(input_dir, split, max_states) for split in SPLITS}
    hashes = {split: splits[split]["backbone_hash"] for split in SPLITS}
    if len(set(hashes.values())) != 1:
        raise ValueError(f"backbone hash mismatch across splits: {hashes}; refusing to mix PathCosts")
    rounds = {split: int(splits[split]["rounds"]) for split in SPLITS}
    if len(set(rounds.values())) != 1:
        raise ValueError(f"rounds mismatch across splits: {rounds}")
    n_actions = {split: splits[split]["immediate"].shape[1] for split in SPLITS}
    if len(set(n_actions.values())) != 1:
        raise ValueError(f"action-count mismatch across splits: {n_actions}")
    return splits, load_students(input_dir), hashes[SPLITS[0]]


def baseline_costs(states: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-state realized stop (T=1) and continue (T=rounds, pi0 tail) costs."""
    rows = torch.arange(states["immediate"].shape[0])
    stop = states["immediate"][rows, states["baseline_action"]].double()
    cont = states["q_pi0"][rows, states["baseline_action"]].double()
    return stop, cont


def student_predictions(students: dict, states: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Predicted stop/continue costs at the baseline action, from h only."""
    rows = torch.arange(states["h"].shape[0])
    features = states["h"].float()
    predictions = {}
    for name in ("immediate", "q_pi0"):
        entry = students[name]
        predictions[name] = predict(entry["student"], features, entry["mean"], entry["std"]).double()
    return predictions["immediate"][rows, states["baseline_action"]], predictions["q_pi0"][rows, states["baseline_action"]]


def mean_se(values: torch.Tensor) -> dict:
    values = values.double()
    n = values.numel()
    if n == 0:
        return {"mean": None, "se": None, "count": 0}
    return {"mean": values.mean().item(), "se": (values.std(unbiased=False) / math.sqrt(n)).item(), "count": n}


def evaluate_decisions(continue_mask: torch.Tensor, stop_cost: torch.Tensor, cont_cost: torch.Tensor,
                       *, rounds: int, top_k: int, fixed_t4_cost: torch.Tensor) -> dict:
    """Realized metrics of a stop/continue decision rule on frozen labels.

    ``continue_mask`` selects the measured q_pi0 (T=rounds) cost; otherwise the
    measured immediate (T=1) cost.  Errors are split into premature stops
    (stopped although continuing was strictly better, magnitude in nats lost)
    and wasteful continuations (continued although stopping was at least as
    good, magnitude in extra expert calls plus any excess nats).
    """
    if continue_mask.shape != stop_cost.shape or continue_mask.dtype != torch.bool:
        raise ValueError("continue_mask must be a bool tensor with one entry per state")
    true_delta = stop_cost - cont_cost
    cost = torch.where(continue_mask, cont_cost, stop_cost)
    n_rounds = torch.where(continue_mask, float(rounds), 1.0)
    premature = (~continue_mask) & (true_delta > 0)
    wasteful = continue_mask & (true_delta <= 0)
    delta_vs_t4 = cost - fixed_t4_cost
    return {
        "n_states": int(stop_cost.numel()),
        "mean_nll": mean_se(cost),
        "mean_rounds": mean_se(n_rounds),
        "mean_expert_calls": mean_se(n_rounds * top_k),
        "stop_rate": (~continue_mask).double().mean().item(),
        "nll_delta_vs_fixed_t4": mean_se(delta_vs_t4),
        "premature_stop": {
            "rate": premature.double().mean().item(),
            "nll_lost": mean_se(true_delta[premature]),
        },
        "wasteful_continue": {
            "rate": wasteful.double().mean().item(),
            "extra_expert_calls_each": (rounds - 1) * top_k,
            "excess_nll": mean_se((cont_cost - stop_cost)[wasteful]),
        },
    }


def tau_candidates(delta_hat: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Candidate thresholds: validation quantiles plus both degenerate ends."""
    if grid_size < 2:
        raise ValueError("tau grid needs at least 2 points")
    quantiles = torch.linspace(0.0, 1.0, grid_size, dtype=torch.float64)
    grid = torch.quantile(delta_hat.double(), quantiles).unique()
    eps = max(1e-12, float(grid.abs().max()) * 1e-9)
    return torch.cat([grid[:1] - eps, grid, grid[-1:] + eps])


def calibration_frontier(delta_hat: torch.Tensor, stop_cost: torch.Tensor, cont_cost: torch.Tensor,
                         *, rounds: int, grid_size: int) -> list[dict]:
    """Validation quality/compute point for every candidate threshold."""
    delta_hat = delta_hat.double()
    frontier = []
    n = delta_hat.numel()
    for tau in tau_candidates(delta_hat, grid_size):
        continue_mask = delta_hat > tau
        cost = torch.where(continue_mask, cont_cost, stop_cost)
        n_rounds = torch.where(continue_mask, float(rounds), 1.0)
        frontier.append({
            "tau": float(tau),
            "mean_nll": cost.mean().item(),
            "mean_rounds": n_rounds.mean().item(),
            "stop_rate": 1.0 - continue_mask.double().mean().item(),
            "n_states": n,
        })
    return frontier


def select_tau(frontier: list[dict], budget: float) -> dict | None:
    """Lowest-validation-NLL threshold whose mean rounds fit the budget.

    Ties prefer fewer rounds (cheaper, more conservative).  The achieved mean
    rounds is reported as-is; a controller that cannot hit the budget exactly
    is never silently rescaled.
    """
    feasible = [point for point in frontier if point["mean_rounds"] <= budget + 1e-12]
    if not feasible:
        return None
    return min(feasible, key=lambda point: (point["mean_nll"], point["mean_rounds"]))


def calibration_bins(delta_hat: torch.Tensor, true_delta: torch.Tensor, n_bins: int) -> list[dict]:
    """Binned calibration on a holdout: predicted vs realized continue gain."""
    if n_bins < 1:
        raise ValueError("need at least one calibration bin")
    order = torch.argsort(delta_hat)
    bins = []
    for chunk in torch.chunk(order, n_bins):
        bins.append({
            "count": int(chunk.numel()),
            "predicted_delta_mean": delta_hat[chunk].mean().item(),
            "realized_delta": mean_se(true_delta[chunk]),
        })
    return [entry for entry in bins if entry["count"] > 0]


def build_signals(splits: dict, students: dict) -> tuple[dict, dict]:
    """Per-split halt signals (larger = more inclined to continue) + availability.

    A signal is only built from tensors actually present in the loaded
    artifacts; anything missing is reported unavailable with its reason.
    """
    signals, availability = {}, {}
    predicted = {split: student_predictions(students, states) for split, states in splits.items()}
    signals["value_student"] = {split: predicted[split][0] - predicted[split][1] for split in splits}
    availability["value_student"] = {
        "available": True,
        "direction": "continue iff predicted (stop_nll - continue_nll) > tau",
        "source": "students.pt frozen by E01; trained on the train split with h as the only input",
    }
    for name, spec in CHEAP_SIGNALS.items():
        key = spec["key"]
        if all(key in states and torch.is_tensor(states[key]) and states[key].ndim == 1
               for states in splits.values()):
            values = {split: splits[split][key].double() for split in splits}
            if name in ("maxprob", "logit_margin"):
                values = {split: -value for split, value in values.items()}
            signals[name] = values
            availability[name] = {"available": True, "direction": spec["direction"],
                                  "source": f"states_*.pt[{key!r}]"}
        else:
            availability[name] = {"available": False, "reason": spec["missing_reason"]}
    return signals, availability


def sanitize(payload):
    """JSON-safe copy: non-finite floats become None."""
    if isinstance(payload, float) and not math.isfinite(payload):
        return None
    if isinstance(payload, dict):
        return {key: sanitize(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [sanitize(value) for value in payload]
    return payload


def run(input_dir: str | Path, budgets=DEFAULT_BUDGETS, nll_tolerance: float = NLL_TOLERANCE,
        tau_grid_size: int = 400, n_calibration_bins: int = 10, top_k: int = 2, max_states: int = 0) -> dict:
    splits, students, backbone_hash = load_artifacts(input_dir, max_states)
    rounds = int(splits["validation"]["rounds"])
    realized = {split: baseline_costs(states) for split, states in splits.items()}
    true_delta = {split: realized[split][0] - realized[split][1] for split in SPLITS}
    signals, availability = build_signals(splits, students)

    test_stop, test_cont = realized["test"]
    fixed = {
        "T1": evaluate_decisions(torch.zeros_like(test_stop, dtype=torch.bool), test_stop, test_cont,
                                 rounds=rounds, top_k=top_k, fixed_t4_cost=test_cont),
        "T4": evaluate_decisions(torch.ones_like(test_stop, dtype=torch.bool), test_stop, test_cont,
                                 rounds=rounds, top_k=top_k, fixed_t4_cost=test_cont),
    }

    calibration, test_policies, oracle = {}, {}, {}
    for signal_name, values in signals.items():
        frontier = calibration_frontier(values["validation"], *realized["validation"],
                                        rounds=rounds, grid_size=tau_grid_size)
        budgets_report = {}
        for budget in budgets:
            key = f"{budget:g}"
            chosen = select_tau(frontier, budget)
            if chosen is None:
                budgets_report[key] = {"budget_rounds": budget, "feasible": False,
                                       "reason": "no threshold reached this average-rounds budget on validation"}
                continue
            tau = chosen["tau"]
            budgets_report[key] = {
                "budget_rounds": budget,
                "feasible": True,
                "tau": tau,
                "tau_meaning": "continue iff signal > tau; for the value student the signal is the "
                               "predicted (stop_nll - continue_nll), so tau absorbs lambda * extra "
                               "continuation cost",
                "validation_mean_nll": chosen["mean_nll"],
                "validation_mean_rounds": chosen["mean_rounds"],
                "frozen_before_test": True,
            }
            continue_mask = values["test"] > tau
            metrics = evaluate_decisions(continue_mask, test_stop, test_cont,
                                         rounds=rounds, top_k=top_k, fixed_t4_cost=test_cont)
            metrics["nll_within_tolerance_vs_fixed_t4"] = (
                metrics["nll_delta_vs_fixed_t4"]["mean"] is not None
                and metrics["nll_delta_vs_fixed_t4"]["mean"] <= nll_tolerance
            )
            test_policies[f"{signal_name}_budget_{key}"] = metrics
            if signal_name == "value_student":
                oracle_mask = true_delta["test"] > tau
                oracle[f"oracle_budget_{key}"] = evaluate_decisions(
                    oracle_mask, test_stop, test_cont, rounds=rounds, top_k=top_k, fixed_t4_cost=test_cont)
        calibration[signal_name] = {
            "split": "validation",
            "selection_rule": "lowest validation mean NLL among thresholds with validation mean rounds "
                              "<= budget; ties prefer fewer rounds; thresholds frozen before test",
            "budgets": budgets_report,
            "frontier": frontier,
        }
    oracle["oracle_tau_0"] = evaluate_decisions(true_delta["test"] > 0, test_stop, test_cont,
                                                rounds=rounds, top_k=top_k, fixed_t4_cost=test_cont)

    summary = {
        "experiment": "E03 pilot: value-calibrated halting on frozen E01 artifacts",
        "input_dir": str(input_dir),
        "backbone_hash": backbone_hash,
        "hash_verified_across_splits": True,
        "labels": {
            "stop": "immediate[baseline_action]: measured FP32 CE after T=1",
            "continue": f"q_pi0[baseline_action]: measured FP32 CE after T={rounds} with frozen pi0 tail",
            "reused_not_relabeled": True,
            "depths_available": [1, rounds],
            "binary_decision_note": "E01 artifacts contain exactly two depths, so this pilot compares "
                                    "stop-after-round-1 vs continue-to-rounds; no intermediate-depth "
                                    "stopping labels exist and none are fabricated",
        },
        "n_states": {split: int(splits[split]["h"].shape[0]) for split in SPLITS},
        "nll_tolerance_nat": nll_tolerance,
        "top_k": top_k,
        "signals": availability,
        "fixed": fixed,
        "calibration": calibration,
        "test": test_policies,
        "calibration_curve_test": calibration_bins(signals["value_student"]["test"], true_delta["test"],
                                                   n_calibration_bins),
        "oracle": {
            **oracle,
            "deployable": False,
            "note": "label-informed (uses realized deltas on test); an upper-bound reference only, "
                    "never mixed into the deployable student results above",
        },
        "scope_notes": [
            "debug-scale pilot: counts and SEs come from the small E01 coverage_remediation splits only",
            "only the last adaptive layer is studied; no wall-clock/FLOPs claims (see E11 for system-level)",
            "quality-compute trade-off is measured in realized NLL vs average rounds/expert calls, "
            "not in latency",
        ],
    }
    return sanitize(summary)


def main(argv=None) -> None:
    args = parse_args(argv)
    summary = run(
        args.input_dir,
        budgets=tuple(args.budgets),
        nll_tolerance=args.nll_tolerance,
        tau_grid_size=args.tau_grid_size,
        n_calibration_bins=args.calibration_bins,
        top_k=args.top_k,
        max_states=args.max_states,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "e03_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
