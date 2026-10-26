"""E07 offline trace replay: 64-request service window from frozen E03/E01 artifacts.

This is the real-LM service-window part of E07 (plan step 3): a 64-request
window is drawn from the frozen E01 ``coverage_remediation`` test split, the
per-request continue gains come from the E03-calibrated value students
(deployable signal) and the realized stop/continue labels (evaluation only),
and cross-request budget allocators are compared at four budget tiers — the
same average-round budgets as E03 (1.5 / 2 / 3 / 4).

Depth labels: the E01 artifacts contain realized costs at exactly two depths
(T=1 ``immediate`` and T=rounds ``q_pi0``), so each request has R = rounds-1
extra rounds whose per-round gains are NOT measured.  To compare the
single-step index d1/c1, the fixed two-step package d2/c2 and the full-tail
index, per-round gain profiles are built by multiplying the endpoint delta by
a weight vector summing to 1, under two explicit endpoint-anchored
assumptions — "diminishing" and "delayed" — reported side by side as a
sensitivity axis.  These are assumptions, not measured intermediate labels.

Compared allocators (decisions on predicted gains, evaluated on realized):

* equal_depth: blind round-robin (max-min fair, per-request rounds differ by
  at most 1); the equal-depth baseline when the budget does not divide evenly,
* random: blind random extra allocation, controls for the budget itself,
* single_step_index: gain index I_i = d1_i / c1, recomputed after each round,
* two_step_package: fixed two-step package d2/c2 reserving two rounds,
* tail_index: indivisible full-continuation package, I = d_tail / c_tail,
* rolling_window_dp: rolling-horizon exact DP over arrival-ordered chunks of
  ``dp_window`` requests on predicted gains — an explicit approximation of
  the joint allocation; the exact full-window optimum on realized gains is
  tractable separately via the separable prefix DP (prefix_knapsack, O(N B H))
  and is reported per tier as the label-informed oracle (never deployable).

The small-window exact DP (``dp_window`` requests per instance, realized
gains) is reported per budget tier as a per-instance gap diagnostic:
absolute regret is always reported, and the relative gap is null on
zero-optimum instances instead of being masked as 0.

Hard prefix ledger: the online single-step index allocator is replayed as a
stream — each request arrival supplies its share of the tier's expert-call
credits, executions are debited only when affordable, and the invariant
cum_executed <= cum_supply is checked on every prefix.  The whole ledger is
kept, not just the average.

Sensitivity: request reordering (shuffle seeds and adversarial length/doc
orderings), arrival batch size, and position-quartile (length proxy)
stratified quality.

OFFLINE REPLAY: all costs are in units of Top-2 expert calls and decision
epochs.  No wall-clock, GPU-latency, throughput or bucketing-overhead
measurement is made or implied; same-round GPU bucketing (plan step 5) is not
executed here.

Example:

    python -m experiments.e07_replay \
        --input-dir runs/e01/coverage_remediation \
        --e03-summary runs/e03/pilot/e03_summary.json \
        --output-dir runs/e07/replay
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import random
import statistics
from pathlib import Path

from experiments import e03_halting as e03
from experiments import e07_synthetic as e07syn
from train.recovery_common import atomic_json

DEFAULT_BUDGET_ROUNDS = (1.5, 2.0, 3.0, 4.0)  # average rounds/request, as in E03
DEFAULT_BATCH_SIZES = (1, 4, 16)
PROFILE_RATIO = 0.5  # geometric weight ratio of the per-round gain profiles
PROFILES = ("diminishing", "delayed")

LATENCY_NOTE = (
    "Offline trace replay: all costs are in units of Top-2 expert calls and "
    "decision epochs. No wall-clock, GPU-latency, throughput or "
    "bucketing-overhead measurement is made or implied; treating these "
    "numbers as real GPU latency is invalid. Same-round GPU bucketing and "
    "its grouping overhead (plan step 5) are not executed."
)
INTERMEDIATE_DEPTH_NOTE = (
    "E01 artifacts carry realized costs at exactly two depths (T=1 and "
    "T=rounds). Per-round gains inside the continuation tail are not "
    "measured; they are endpoint-anchored assumptions (a weight vector "
    "summing to 1 multiplied by the endpoint delta), reported under both "
    "'diminishing' and 'delayed' profiles as a sensitivity axis."
)


# --------------------------------------------------------------------------
# Window construction from frozen artifacts
# --------------------------------------------------------------------------

def profile_weights(n_rounds: int, profile: str, ratio: float = PROFILE_RATIO) -> list[float]:
    """Per-round weight vector summing to 1 for an endpoint-anchored profile."""
    if n_rounds < 1:
        raise ValueError("need at least one extra round")
    if profile == "diminishing":
        raw = [ratio ** h for h in range(n_rounds)]
    elif profile == "delayed":
        raw = [ratio ** (n_rounds - 1 - h) for h in range(n_rounds)]
    else:
        raise ValueError(f"unknown profile: {profile}")
    total = sum(raw)
    return [w / total for w in raw]


def trajectory(delta: float, n_rounds: int, profile: str) -> list[float]:
    """Endpoint-anchored per-round gains: delta times the profile weights."""
    return [delta * w for w in profile_weights(n_rounds, profile)]


def load_window(input_dir: str | Path, e03_summary_path: str | Path,
                window_size: int, seed: int, split: str = "test") -> dict:
    """Sample the service window and compute predicted/realized continue gains.

    The backbone hash and rounds of the artifacts are verified against the
    frozen E03 summary; students are used as-is (no refitting).  Sampling is
    without replacement, seeded.
    """
    input_dir = Path(input_dir)
    with open(e03_summary_path) as f:
        e03_summary = json.load(f)
    states = e03.load_split(input_dir, split)
    students = e03.load_students(input_dir)
    if states["backbone_hash"] != e03_summary["backbone_hash"]:
        raise ValueError(
            f"backbone hash mismatch: artifacts {states['backbone_hash']} vs "
            f"E03 summary {e03_summary['backbone_hash']}; refusing to mix artifacts")
    rounds = int(states["rounds"])
    n_states = states["h"].shape[0]
    if not 1 <= window_size <= n_states:
        raise ValueError(f"window_size must be in [1, {n_states}], got {window_size}")

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(n_states), window_size))

    realized_stop, realized_cont = e03.baseline_costs(states)
    pred_stop, pred_cont = e03.student_predictions(students, states)
    positions = states["positions"] if "positions" in states else None
    doc_ids = states["doc_ids"] if "doc_ids" in states else None

    requests = []
    for arrival, i in enumerate(indices):
        requests.append({
            "state_index": int(i),
            "arrival": arrival,
            "stop_cost": float(realized_stop[i]),
            "continue_cost": float(realized_cont[i]),
            "true_delta": float(realized_stop[i] - realized_cont[i]),
            "pred_delta": float(pred_stop[i] - pred_cont[i]),
            "position": int(positions[i]) if positions is not None else None,
            "doc_id": int(doc_ids[i]) if doc_ids is not None else None,
        })
    return {
        "split": split,
        "rounds": rounds,
        "extra_rounds": rounds - 1,
        "backbone_hash": states["backbone_hash"],
        "e03_summary_path": str(e03_summary_path),
        "seed": seed,
        "requests": requests,
    }


def window_trajectories(window: dict, profile: str, key: str) -> list[list[float]]:
    """Per-request predicted (key='pred_delta') or realized ('true_delta') trajectories."""
    return [trajectory(r[key], window["extra_rounds"], profile) for r in window["requests"]]


# --------------------------------------------------------------------------
# Offline allocators: decide on predicted trajectories, return steps
# --------------------------------------------------------------------------

def _wrap_with_steps(deltas, rounds_list, order):
    """Rebuild a step list for an allocation executed in the given request order."""
    steps = []
    remaining = list(rounds_list)
    epoch = 0
    while True:
        progressed = False
        for i in order:
            if remaining[i] > 0:
                steps.append({"request": i, "round": rounds_list[i] - remaining[i] + 1,
                              "gain": deltas[i][rounds_list[i] - remaining[i]], "epoch": epoch})
                remaining[i] -= 1
                epoch += 1
                progressed = True
        if not progressed:
            break
    return steps


def equal_depth_allocation(deltas, budget):
    """Blind round-robin (max-min fair); the equal-depth baseline."""
    total, state, used = e07syn.round_robin_allocation(deltas, budget)
    # replay round-robin exactly: cycle through requests in arrival order
    steps = _wrap_with_steps(deltas, state, list(range(len(deltas))))
    return total, state, used, steps


def random_allocation(deltas, budget, rng):
    """Blind random extra allocation; controls for the budget itself."""
    total, state, used = e07syn.random_allocation(deltas, budget, rng)
    steps = _wrap_with_steps(deltas, state, list(range(len(deltas))))
    return total, state, used, steps


def single_step_index_allocation(deltas, budget):
    """Gain index I_i = d1_i / c1 recomputed after every round (equal costs)."""
    total, state, used, raw_steps = e07syn.index_allocation(deltas, budget)
    steps = [dict(step, epoch=e) for e, step in enumerate(raw_steps)]
    return total, state, used, steps


def two_step_package_allocation(deltas, budget):
    """Fixed two-step package d2/c2 reserving two rounds; falls back to d1/c1."""
    n = len(deltas)
    state = [0] * n
    used = 0
    steps = []
    epoch = 0
    while used < budget:
        best_i, best_idx, package = -1, 0.0, False
        for i in range(n):
            h = state[i]
            if h + 2 <= len(deltas[i]) and used + 2 <= budget:
                idx = (deltas[i][h] + deltas[i][h + 1]) / 2.0
                if idx > best_idx:
                    best_idx, best_i, package = idx, i, True
        if best_i < 0:
            for i in range(n):
                h = state[i]
                if h < len(deltas[i]):
                    idx = deltas[i][h]
                    if idx > best_idx:
                        best_idx, best_i, package = idx, i, False
        if best_i < 0:
            break
        h = state[best_i]
        n_steps = 2 if package else 1
        for k in range(n_steps):
            steps.append({"request": best_i, "round": h + k + 1,
                          "gain": deltas[best_i][h + k], "epoch": epoch})
            epoch += 1
        state[best_i] += n_steps
        used += n_steps
    total = sum(deltas[i][h] for i in range(n) for h in range(state[i]))
    return total, state, used, steps


def tail_index_allocation(deltas, budget):
    """Indivisible full-continuation package, I = d_tail / c_tail."""
    n = len(deltas)
    state = [0] * n
    used = 0
    steps = []
    epoch = 0
    while True:
        best_i, best_idx = -1, 0.0
        for i in range(n):
            length = len(deltas[i])
            if state[i] == 0 and used + length <= budget:
                idx = sum(deltas[i]) / length
                if idx > best_idx:
                    best_idx, best_i = idx, i
        if best_i < 0:
            break
        for k in range(len(deltas[best_i])):
            steps.append({"request": best_i, "round": k + 1,
                          "gain": deltas[best_i][k], "epoch": epoch})
            epoch += 1
        state[best_i] = len(deltas[best_i])
        used += len(deltas[best_i])
    total = sum(sum(deltas[i]) for i in range(n) if state[i] > 0)
    return total, state, used, steps


def rolling_window_dp_allocation(deltas, budget, dp_window):
    """Rolling-horizon exact DP over arrival-ordered chunks (approximation).

    Each chunk of ``dp_window`` requests receives at most its proportional
    share of the remaining budget (scaled by remaining capacity); the exact DP
    of e07_synthetic is solved inside the chunk on predicted gains and the
    unspent remainder rolls forward.  This is NOT the joint optimum over the
    full window; the exact full-window oracle on realized gains is computed
    separately by e07_synthetic.prefix_knapsack (O(N B H)).
    """
    n = len(deltas)
    state = [0] * n
    used = 0
    steps = []
    epoch = 0
    start = 0
    while start < n and used < budget:
        chunk = list(range(start, min(start + dp_window, n)))
        capacity = sum(len(deltas[i]) for i in chunk)
        remaining_capacity = sum(len(deltas[i]) for i in range(start, n))
        remaining_budget = budget - used
        chunk_budget = min(remaining_budget,
                           math.ceil(remaining_budget * capacity / remaining_capacity))
        chunk_deltas = [deltas[i] for i in chunk]
        _, chunk_rounds = e07syn.exact_dp(chunk_deltas, chunk_budget)
        for j, i in enumerate(chunk):
            for h in range(chunk_rounds[j]):
                steps.append({"request": i, "round": h + 1, "gain": deltas[i][h], "epoch": epoch})
                epoch += 1
            state[i] = chunk_rounds[j]
            used += chunk_rounds[j]
        start += dp_window
    total = sum(deltas[i][h] for i in range(n) for h in range(state[i]))
    return total, state, used, steps


ALLOCATOR_NAMES = ("equal_depth", "random", "single_step_index", "two_step_package",
                   "tail_index", "rolling_window_dp")


def run_offline_allocators(pred_deltas, budget, rng, dp_window):
    """Every compared allocator on the predicted trajectories of one tier."""
    results = {
        "equal_depth": equal_depth_allocation(pred_deltas, budget),
        "random": random_allocation(pred_deltas, budget, rng),
        "single_step_index": single_step_index_allocation(pred_deltas, budget),
        "two_step_package": two_step_package_allocation(pred_deltas, budget),
        "tail_index": tail_index_allocation(pred_deltas, budget),
        "rolling_window_dp": rolling_window_dp_allocation(pred_deltas, budget, dp_window),
    }
    return results


# --------------------------------------------------------------------------
# Realized evaluation, completion order, stratification
# --------------------------------------------------------------------------

def evaluate_allocation(window, true_deltas, state, steps, top_k, arrival_aligned=False):
    """Realized metrics of an allocation chosen on predicted trajectories."""
    requests = window["requests"]
    n = len(requests)
    gains = [sum(true_deltas[i][h] for h in range(state[i])) for i in range(n)]
    nll = [requests[i]["stop_cost"] - gains[i] for i in range(n)]
    completion = completion_epochs(requests, state, steps, arrival_aligned=arrival_aligned)
    return {
        "mean_nll": statistics.fmean(nll),
        "mean_rounds": 1.0 + statistics.fmean(state),
        "mean_expert_calls": top_k * (1.0 + statistics.fmean(state)),
        "used_extra_rounds": int(sum(state)),
        "rounds_per_request": [int(s) for s in state],
        "realized_total_gain": sum(gains),
        "completion": completion,
        "stratified_nll": stratified_nll(requests, nll),
    }


def completion_epochs(requests, state, steps, arrival_aligned=False):
    """Completion order of the execution schedule (NOT wall-clock time).

    A request with no extra rounds completes on arrival at its base depth;
    others complete at their last executed round.  For offline allocators the
    schedule is computed with the whole window visible, so only the
    completion order/rank is meaningful (``arrival_aligned=False``): a
    request can be scheduled before its arrival index and waiting-vs-arrival
    is not defined.  For the online ledger allocator every execution happens
    at an arrival step at which the request is already active
    (``arrival_aligned=True``, steps carry ``arrival_step``), so waiting =
    completion arrival step minus arrival index is nonnegative.
    """
    step_key = "arrival_step" if arrival_aligned else "epoch"
    last = {}
    for step in steps:
        last[step["request"]] = step[step_key]
    rows = []
    for i, req in enumerate(requests):
        finish = req["arrival"] if state[i] == 0 else last.get(i, req["arrival"])
        row = {
            "request": i,
            "arrival": req["arrival"],
            "extra_rounds": int(state[i]),
            "completion_epoch": int(finish),
        }
        if arrival_aligned:
            row["waiting_epochs"] = int(finish - req["arrival"])
        rows.append(row)
    order = sorted(rows, key=lambda r: (r["completion_epoch"], r["arrival"]))
    rank = {row["request"]: k for k, row in enumerate(order)}
    by_nll_gain = sorted(range(len(requests)), key=lambda i: -requests[i]["true_delta"])
    top_q = by_nll_gain[: max(1, len(requests) // 4)]
    bottom_q = by_nll_gain[-max(1, len(requests) // 4):]
    result = {
        "units": ("arrival steps of the online stream, not wall-clock time" if arrival_aligned
                  else "decision epochs of the offline schedule, not wall-clock time"),
        "arrival_aligned": arrival_aligned,
        "order": [r["request"] for r in order],
        "mean_completion_rank": statistics.fmean(rank.values()),
        "mean_completion_rank_top_gain_quartile": statistics.fmean(rank[i] for i in top_q),
        "mean_completion_rank_bottom_gain_quartile": statistics.fmean(rank[i] for i in bottom_q),
        "per_request": rows,
    }
    if arrival_aligned:
        result["mean_waiting_epochs"] = statistics.fmean(r["waiting_epochs"] for r in rows)
        result["max_waiting_epochs"] = max(r["waiting_epochs"] for r in rows)
        result["mean_waiting_top_gain_quartile"] = statistics.fmean(
            rows[i]["waiting_epochs"] for i in top_q)
        result["mean_waiting_bottom_gain_quartile"] = statistics.fmean(
            rows[i]["waiting_epochs"] for i in bottom_q)
    else:
        result["waiting_note"] = ("waiting-vs-arrival is undefined for offline allocators: the "
                                  "schedule is computed with the whole window visible, so a request "
                                  "can be scheduled before its arrival index")
    return result


def stratified_nll(requests, nll):
    """Per-position-quartile realized NLL (position is the length proxy)."""
    positions = [r["position"] for r in requests]
    if any(p is None for p in positions):
        return {"available": False, "reason": "no position field in artifacts"}
    order = sorted(range(len(requests)), key=lambda i: positions[i])
    quartiles = [order[k * len(order) // 4:(k + 1) * len(order) // 4] for k in range(4)]
    return {
        "available": True,
        "stratifier": "position quartile (length proxy; position 0 = shortest)",
        "quartile_mean_nll": [statistics.fmean(nll[i] for i in q) for q in quartiles],
        "quartile_mean_position": [statistics.fmean(positions[i] for i in q) for q in quartiles],
    }


# --------------------------------------------------------------------------
# Hard streaming ledger (scenario B in service-window form)
# --------------------------------------------------------------------------

def credit_schedule(n_requests, total_credits):
    """Integer per-arrival credit supply; the remainder goes to the earliest arrivals."""
    base = total_credits // n_requests
    extra = total_credits % n_requests
    return [base + (1 if m < extra else 0) for m in range(n_requests)]


def online_ledger_replay(window, pred_deltas, true_deltas, budget, top_k,
                         arrival_order=None, batch_size=1):
    """Streaming hard-ledger replay of the online single-step index allocator.

    Requests arrive in ``arrival_order`` in batches of ``batch_size``; each
    arrival supplies its share of the tier's expert-call credits.  After each
    batch arrival the allocator spends affordable rounds on the active
    requests by recomputed index I_i = d1_i / c1.  Executions are debited only
    when affordable (price signals never replace this check) and the prefix
    invariant cum_executed <= cum_supply is verified on every prefix.
    Returns (ledger_entries, violations, final_state, steps).
    """
    n = len(window["requests"])
    order = list(arrival_order) if arrival_order is not None else list(range(n))
    supply = credit_schedule(n, budget * top_k)
    state = [0] * n
    balance = 0
    cum_executed = 0
    cum_supply = 0
    entries = []
    violations = []
    steps = []
    active = []
    arrived = 0
    for batch_start in range(0, n, batch_size):
        batch = order[batch_start:batch_start + batch_size]
        for i in batch:
            balance += supply[i]
            cum_supply += supply[i]
            active.append(i)
            arrived += 1
        # spend while some active request has a positive affordable index
        while balance >= top_k:
            best_i, best_idx = -1, 0.0
            for i in active:
                h = state[i]
                if h < len(pred_deltas[i]):
                    idx = pred_deltas[i][h] / top_k
                    if idx > best_idx:
                        best_idx, best_i = idx, i
            if best_i < 0:
                break
            h = state[best_i]
            state[best_i] += 1
            balance -= top_k
            cum_executed += top_k
            steps.append({"request": best_i, "round": h + 1,
                          "gain": pred_deltas[best_i][h],
                          "arrival_step": arrived - 1})
        entries.append({
            "arrival_step": len(entries),
            "arrived_requests": [int(i) for i in batch],
            "supply": int(sum(supply[i] for i in batch)),
            "cum_supply": int(cum_supply),
            "cum_executed": int(cum_executed),
            "balance": int(balance),
        })
        if cum_executed > cum_supply or balance < 0:
            violations.append(len(entries) - 1)
    return entries, violations, state, steps


# --------------------------------------------------------------------------
# Small-window exact DP: computable-range upper bound
# --------------------------------------------------------------------------

def small_window_dp_gap(window, true_deltas, pred_deltas, budget_rounds_tiers,
                        dp_window, n_instances, seed):
    """Exact DP on realized gains over small sub-windows (diagnostic reference).

    Heuristic allocations are chosen on predicted gains and evaluated on
    realized gains; their realized value is compared to the exact-DP optimum
    of the same sub-window.  The absolute regret is always reported; the
    relative gap is null on instances whose optimum is zero (dividing by ~0
    would hide negative realized value behind a reported gap of 0).  The
    exact full-window oracle is tractable separately via
    ``e07_synthetic.prefix_knapsack`` (O(N B H)) and is reported per tier in
    the main output; this small-window DP is kept as a per-instance gap
    diagnostic.
    """
    rng = random.Random(seed + 1)
    requests = window["requests"]
    n = len(requests)
    capacity_per_request = window["extra_rounds"]
    out = []
    for avg_rounds in budget_rounds_tiers:
        sub_budget = min(int(round((avg_rounds - 1.0) * dp_window)),
                         dp_window * capacity_per_request)
        agg = {}
        n_zero_optimal = 0
        for _ in range(n_instances):
            idx = sorted(rng.sample(range(n), min(dp_window, n)))
            sub_true = [true_deltas[i] for i in idx]
            sub_pred = [pred_deltas[i] for i in idx]
            opt, _ = e07syn.exact_dp(sub_true, sub_budget)
            zero_optimal = opt <= 1e-12
            n_zero_optimal += int(zero_optimal)
            for name, alloc in (
                    ("equal_depth", lambda d: e07syn.round_robin_allocation(d, sub_budget)[1]),
                    ("single_step_index", lambda d: e07syn.index_allocation(d, sub_budget)[1]),
                    ("two_step_package", lambda d: e07syn.two_step_package_allocation(d, sub_budget)[1])):
                state = alloc(sub_pred)
                realized = sum(sub_true[i][h] for i in range(len(idx)) for h in range(state[i]))
                abs_regret = opt - realized
                a = agg.setdefault(name, {"regrets": [], "gaps": []})
                a["regrets"].append(abs_regret)
                a["gaps"].append(None if zero_optimal else abs_regret / opt)
        out.append({
            "budget_avg_rounds": avg_rounds,
            "sub_window_budget_rounds": sub_budget,
            "n_zero_optimal_instances": n_zero_optimal,
            "mean_abs_regret_vs_exact_dp": {
                name: statistics.fmean(v["regrets"]) for name, v in agg.items()},
            "max_abs_regret_vs_exact_dp": {
                name: max(v["regrets"]) for name, v in agg.items()},
            "mean_rel_gap_vs_exact_dp": {
                name: (statistics.fmean(g for g in v["gaps"] if g is not None)
                       if any(g is not None for g in v["gaps"]) else None)
                for name, v in agg.items()},
            "max_rel_gap_vs_exact_dp": {
                name: (max(g for g in v["gaps"] if g is not None)
                       if any(g is not None for g in v["gaps"]) else None)
                for name, v in agg.items()},
        })
    return {
        "dp_window": dp_window,
        "n_instances_per_tier": n_instances,
        "upper_bound_note": ("Exact DP on realized gains over dp_window-request sub-windows; "
                             "a per-instance gap diagnostic. The exact full-window oracle is "
                             "computed separately by prefix_knapsack (separable prefix DP, "
                             "O(N B H)) and reported per tier; the rolling_window_dp allocator "
                             "remains an explicit approximation, not the joint optimum. "
                             "Absolute regret is always reported; relative gaps are null on "
                             "zero-optimum instances. A relative gap above 1 means the "
                             "heuristic's realized total gain is negative — allocations chosen "
                             "on predicted gains can lose value on mispredicted requests."),
        "per_tier": out,
    }


# --------------------------------------------------------------------------
# Sensitivity: reorder, batch size, length/doc mix
# --------------------------------------------------------------------------

def stream_aligned_window(window, arrival_order):
    """Copy of the window whose per-request arrival is the position in the stream order.

    The online ledger allocator executes a request's rounds only at arrival
    steps at which the request is already active, so completion/waiting are
    measured against the stream position, not the original window index.
    """
    positions = {request: pos for pos, request in enumerate(arrival_order)}
    requests = [dict(r, arrival=positions[i]) for i, r in enumerate(window["requests"])]
    return dict(window, requests=requests)


def reorder_sensitivity(window, pred_deltas, true_deltas, budget, top_k, n_seeds, seed,
                        dp_window=8):
    """Arrival-order sensitivity of the online AND offline allocators.

    The online ledger allocator depends on the arrival order by construction.
    Offline allocators are NOT all order-invariant: the equal-depth
    round-robin remainder follows the arrival order and rolling_window_dp
    chunks the window in arrival order, so both can change their allocation
    (and realized gain) under reordering.  All offline allocators are
    re-evaluated under every ordering; the index allocators
    (single_step_index, two_step_package, tail_index) are order-invariant by
    construction and serve as invariance controls.  Orderings: shuffles plus
    adversarial length/doc mixes.
    """
    requests = window["requests"]
    rng = random.Random(seed + 2)
    rng_offline = random.Random(seed + 3)
    variants = {}
    for s in range(n_seeds):
        order = list(range(len(requests)))
        rng.shuffle(order)
        variants[f"shuffle_{s}"] = order
    positions = [r["position"] if r["position"] is not None else 0 for r in requests]
    doc_ids = [r["doc_id"] if r["doc_id"] is not None else 0 for r in requests]
    variants["longest_first"] = sorted(range(len(requests)), key=lambda i: (-positions[i], i))
    variants["shortest_first"] = sorted(range(len(requests)), key=lambda i: (positions[i], i))
    variants["grouped_by_doc"] = sorted(range(len(requests)), key=lambda i: (doc_ids[i], i))

    rows = {}
    for name, order in variants.items():
        entries, violations, state, steps = online_ledger_replay(
            window, pred_deltas, true_deltas, budget, top_k, arrival_order=order)
        stream = stream_aligned_window(window, order)
        metrics = evaluate_allocation(stream, true_deltas, state, steps, top_k,
                                      arrival_aligned=True)
        # offline allocators re-run on the reordered window: allocation
        # decisions and realized gains can change with the arrival order
        perm_pred = [pred_deltas[i] for i in order]
        perm_true = [true_deltas[i] for i in order]
        offline = {}
        for alloc_name, (_, perm_state, _, _) in run_offline_allocators(
                perm_pred, budget, rng_offline, dp_window).items():
            gain_by_orig = [0.0] * len(order)
            for j, orig in enumerate(order):
                gain_by_orig[orig] = sum(perm_true[j][:perm_state[j]])
            nll = [requests[i]["stop_cost"] - gain_by_orig[i] for i in range(len(order))]
            offline[alloc_name] = {
                "mean_nll": statistics.fmean(nll),
                "realized_total_gain": sum(gain_by_orig),
                "used_extra_rounds": int(sum(perm_state)),
            }
        rows[name] = {
            "mean_nll": metrics["mean_nll"],
            "used_extra_rounds": metrics["used_extra_rounds"],
            "mean_waiting_epochs": metrics["completion"]["mean_waiting_epochs"],
            "ledger_violations": violations,
            "offline": offline,
        }
    nlls = [row["mean_nll"] for row in rows.values()]
    offline_names = sorted(rows[next(iter(rows))]["offline"])
    return {
        "budget_extra_rounds": budget,
        "note": ("Both the online ledger allocator and the offline allocators are "
                 "re-evaluated under every ordering. The online allocator is "
                 "order-dependent by construction; among the offline allocators, "
                 "equal_depth (round-robin remainder follows arrival order) and "
                 "rolling_window_dp (arrival-ordered chunk boundaries) are "
                 "order-dependent, while single_step_index, two_step_package and "
                 "tail_index are order-invariant by construction and act as "
                 "invariance controls."),
        "adversarial_orderings": ["longest_first", "shortest_first", "grouped_by_doc"],
        "per_ordering": rows,
        "mean_nll_range": [min(nlls), max(nlls)],
        "mean_nll_std": statistics.pstdev(nlls),
        "offline_mean_nll_range": {
            name: [min(row["offline"][name]["mean_nll"] for row in rows.values()),
                   max(row["offline"][name]["mean_nll"] for row in rows.values())]
            for name in offline_names
        },
    }


def batch_size_sensitivity(window, pred_deltas, true_deltas, budget, top_k, batch_sizes):
    """Arrival batch-size sensitivity of the online index allocator."""
    rows = {}
    for b in batch_sizes:
        entries, violations, state, steps = online_ledger_replay(
            window, pred_deltas, true_deltas, budget, top_k, batch_size=b)
        stream = stream_aligned_window(window, list(range(len(window["requests"]))))
        metrics = evaluate_allocation(stream, true_deltas, state, steps, top_k,
                                      arrival_aligned=True)
        rows[str(b)] = {
            "mean_nll": metrics["mean_nll"],
            "used_extra_rounds": metrics["used_extra_rounds"],
            "mean_waiting_epochs": metrics["completion"]["mean_waiting_epochs"],
            "ledger_violations": violations,
        }
    return {"budget_extra_rounds": budget, "per_batch_size": rows}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def sanitize(payload):
    if isinstance(payload, float) and not math.isfinite(payload):
        return None
    if isinstance(payload, dict):
        return {key: sanitize(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [sanitize(value) for value in payload]
    return payload


def run(input_dir, e03_summary_path, window_size=64,
        budget_rounds_tiers=DEFAULT_BUDGET_ROUNDS, seed=20261025,
        dp_window=8, n_dp_instances=16, n_reorder_seeds=8,
        batch_sizes=DEFAULT_BATCH_SIZES, top_k=2) -> dict:
    window = load_window(input_dir, e03_summary_path, window_size, seed)
    extra_rounds = window["extra_rounds"]
    requests = window["requests"]
    full_budget = window_size * extra_rounds

    tiers = {}
    for avg_rounds in budget_rounds_tiers:
        budget = min(int(round((avg_rounds - 1.0) * window_size)), full_budget)
        tiers[f"{avg_rounds:g}"] = {"avg_rounds": avg_rounds, "extra_rounds_budget": budget,
                                    "expert_call_budget": budget * top_k}

    profiles_out = {}
    for profile in PROFILES:
        pred_deltas = window_trajectories(window, profile, "pred_delta")
        true_deltas = window_trajectories(window, profile, "true_delta")
        rng = random.Random(seed)
        tier_rows = {}
        for tier_key, tier in tiers.items():
            budget = tier["extra_rounds_budget"]
            allocator_rows = {}
            for name, (pred_total, state, used, steps) in run_offline_allocators(
                    pred_deltas, budget, rng, dp_window).items():
                metrics = evaluate_allocation(window, true_deltas, state, steps, top_k)
                metrics["predicted_total_gain"] = pred_total
                allocator_rows[name] = metrics
            # exact full-window label-informed oracle (separable prefix DP,
            # O(N B H)); a lower bound on NLL in this interpolation
            # environment, never a deployable policy
            oracle_gain, oracle_rounds = e07syn.prefix_knapsack(true_deltas, budget)
            oracle_gains = [sum(true_deltas[i][:oracle_rounds[i]]) for i in range(len(requests))]
            oracle_nll = [requests[i]["stop_cost"] - oracle_gains[i] for i in range(len(requests))]
            tier_oracle = {
                "realized_total_gain": oracle_gain,
                "mean_nll": statistics.fmean(oracle_nll),
                "used_extra_rounds": int(sum(oracle_rounds)),
                "rounds_per_request": [int(k) for k in oracle_rounds],
                "deployable": False,
                "note": ("Label-informed exact offline oracle on realized gains "
                         "(prefix_knapsack; separable, budget may be left unspent). "
                         "This is a lower bound on achievable NLL inside the "
                         "endpoint-interpolated environment, not a deploy gain and "
                         "not a new LM result."),
            }
            # hard streaming ledger for the online single-step index allocator
            entries, violations, on_state, on_steps = online_ledger_replay(
                window, pred_deltas, true_deltas, budget, top_k)
            stream = stream_aligned_window(window, list(range(len(requests))))
            online_metrics = evaluate_allocation(stream, true_deltas, on_state, on_steps, top_k,
                                                 arrival_aligned=True)
            allocator_rows["online_single_step_index"] = {
                **online_metrics,
                "hard_ledger": {
                    "credit_supply_per_arrival": credit_schedule(len(requests), budget * top_k),
                    "entries": entries,
                    "prefix_invariant": "cum_executed <= cum_supply on every prefix",
                    "violations": violations,
                    "violation_count": len(violations),
                },
            }
            tier_rows[tier_key] = {**tier, "allocators": allocator_rows,
                                   "full_window_label_informed_oracle": tier_oracle}
        dp_gap = small_window_dp_gap(window, true_deltas, pred_deltas, budget_rounds_tiers,
                                     dp_window, n_dp_instances, seed)
        mid_tier = sorted(tiers, key=lambda k: tiers[k]["extra_rounds_budget"])[len(tiers) // 2 - 1]
        mid_budget = tiers[mid_tier]["extra_rounds_budget"]
        profiles_out[profile] = {
            "intermediate_depth_note": INTERMEDIATE_DEPTH_NOTE,
            "budget_tiers": tier_rows,
            "small_window_exact_dp_gap": dp_gap,
            "sensitivity": {
                "reorder": reorder_sensitivity(window, pred_deltas, true_deltas, mid_budget,
                                               top_k, n_reorder_seeds, seed, dp_window),
                "batch_size": batch_size_sensitivity(window, pred_deltas, true_deltas, mid_budget,
                                                     top_k, batch_sizes),
                "length_mix": {
                    "note": ("Stratified per-quartile NLL is reported per allocator inside "
                             "each budget tier (stratified_nll, position quartile as length "
                             "proxy); adversarial length/doc arrival orderings are in the "
                             "reorder block."),
                    "mid_tier": mid_tier,
                },
            },
        }

    summary = {
        "experiment": "E07 offline trace replay: 64-request service window on frozen E03/E01 artifacts",
        "date": datetime.date.today().isoformat(),
        "seed": seed,
        "offline_replay": True,
        "latency_note": LATENCY_NOTE,
        "input": {
            "input_dir": str(input_dir),
            "e03_summary": str(e03_summary_path),
            "backbone_hash": window["backbone_hash"],
            "hash_verified_against_e03_summary": True,
            "split": window["split"],
            "window_size": window_size,
            "rounds": window["rounds"],
            "extra_rounds_per_request": extra_rounds,
            "top_k": top_k,
            "students": "E01-frozen value students, reused as-is; no refitting, h is the only input",
        },
        "allocators": {
            "equal_depth": "blind round-robin; max-min fair equal-depth baseline",
            "random": "blind random extra allocation; controls for the budget itself",
            "single_step_index": "gain index I = d1/c1, recomputed after every round",
            "two_step_package": "fixed two-step package d2/c2 reserving two rounds (2 Top-2 calls)",
            "tail_index": "indivisible full-continuation package, I = d_tail/c_tail",
            "rolling_window_dp": ("rolling-horizon exact DP over arrival-ordered chunks of "
                                  f"{dp_window} requests on predicted gains; explicit approximation, "
                                  "not the joint full-window optimum"),
            "online_single_step_index": ("streaming version of the single-step index with a hard "
                                         "per-arrival credit ledger; executions only when affordable"),
        },
        "budget_tiers_note": ("Four tiers at the E03 average-round budgets; the extra-round budget "
                              "is (avg_rounds - 1) * window_size, and spending is never forced — "
                              "predicted nonpositive gains are not funded."),
        "window_state_indices": [r["state_index"] for r in requests],
        "profiles": profiles_out,
        "scope_notes": [
            INTERMEDIATE_DEPTH_NOTE,
            LATENCY_NOTE,
            "Requests are states sampled without replacement from the frozen test split; quality is "
            "mean per-request NLL at the baseline action, budget is counted in Top-2 expert calls.",
            "The exact full-window oracle on realized gains IS computed per tier via the separable "
            "prefix DP (prefix_knapsack, O(N B H)); it is a label-informed lower bound on NLL in "
            "this interpolation environment, never a deployable policy. The small-window exact DP "
            "gap block remains as a per-instance realized-gap diagnostic.",
        ],
    }
    return sanitize(summary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="runs/e01/coverage_remediation")
    parser.add_argument("--e03-summary", default="runs/e03/pilot/e03_summary.json")
    parser.add_argument("--output-dir", default="runs/e07/replay")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--budget-rounds", type=float, nargs="+",
                        default=list(DEFAULT_BUDGET_ROUNDS),
                        help="average-rounds budget tiers, as in E03 (default: 1.5 2 3 4)")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--dp-window", type=int, default=8,
                        help="sub-window size of the exact-DP upper bound and rolling DP chunks")
    parser.add_argument("--n-dp-instances", type=int, default=16)
    parser.add_argument("--n-reorder-seeds", type=int, default=8)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--top-k", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    summary = run(
        args.input_dir,
        args.e03_summary,
        window_size=args.window_size,
        budget_rounds_tiers=tuple(args.budget_rounds),
        seed=args.seed,
        dp_window=args.dp_window,
        n_dp_instances=args.n_dp_instances,
        n_reorder_seeds=args.n_reorder_seeds,
        batch_sizes=tuple(args.batch_sizes),
        top_k=args.top_k,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "e07_replay.json"
    atomic_json(out_path, summary)
    print(f"wrote {out_path}")
    print(f"offline_replay: {summary['offline_replay']} ({summary['latency_note'][:80]}...)")
    for profile in PROFILES:
        tiers = summary["profiles"][profile]["budget_tiers"]
        for key, tier in tiers.items():
            alloc = tier["allocators"]
            viol = alloc["online_single_step_index"]["hard_ledger"]["violation_count"]
            print(f"[{profile}] tier {key}: budget={tier['extra_rounds_budget']} rounds | "
                  f"NLL equal_depth={alloc['equal_depth']['mean_nll']:.4f} "
                  f"index={alloc['single_step_index']['mean_nll']:.4f} "
                  f"pkg={alloc['two_step_package']['mean_nll']:.4f} "
                  f"tail={alloc['tail_index']['mean_nll']:.4f} "
                  f"rollDP={alloc['rolling_window_dp']['mean_nll']:.4f} | "
                  f"ledger violations={viol}")


if __name__ == "__main__":
    main()
