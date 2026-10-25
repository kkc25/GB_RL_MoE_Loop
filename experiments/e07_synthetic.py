"""E07 synthetic part: cross-request budget allocation under a hard budget.

Scenario A (service window): N <= 8 requests, each with a per-round
improvement trajectory of length H <= 4, share a budget of B extra Top-2
group calls. An exact DP over (rounds_used_per_request, remaining_budget)
gives the computable-range upper bound; the gain-index heuristic
I_i = d1_i / c1 is compared against it, together with round-robin, random
and fixed two-step-package (d2/c2) baselines. Two trajectory families are
built: diminishing marginal returns, and invest-first-benefit-later. The
latter makes the greedy ratio fail; the failure samples are kept and
reported, not deleted.

Scenario B (streaming ledger): each token supplies K expert credits,
R_m = R_{m-1} + K - q_m >= 0, with an optional carryover cap whose overflow
is voided. The whole per-token ledger is kept, and the prefix invariant
cum_executed(q) <= K * m is checked on every prefix.

SeqTopK audit: the plan-documented reading of public v2 eq. (7) — at token
m the prefix top-(mK) set of cumulative scores is selected, and calls
already spent on earlier tokens cannot be withdrawn — is replayed on the
E=4, K=1 counterexample scores (.40,.30,.20,.10) -> (.49,.48,.02,.01),
which really executes 1+2=3 calls against a prefix budget of 2. The
set-growth ("virtual history") accounting that would hide the overdraw is
reported next to the real executed-call ledger. This audits the public
formula only; the authors' repository implementation was not reviewed.

The real-LM service-load part (E03-calibrated controller, 64-request trace
replay, request-reorder/batch/length-mix sensitivity, GPU bucketing) is not
executed here and is marked as such in the output.
"""

import argparse
import datetime
import json
import random
from functools import lru_cache

TWO_TOP2_CREDITS = 4  # one fixed two-step package = two Top-2 calls


# --------------------------------------------------------------------------
# Scenario A: exact DP and allocation heuristics on the finite model
# --------------------------------------------------------------------------

def exact_dp(deltas, budget):
    """Optimal total improvement under a budget of extra rounds.

    deltas[i][h] is the improvement request i gets from its (h+1)-th extra
    round. Budget and rounds are in equal-cost units (one Top-2 group call).
    Spending is never forced: the DP may leave budget unused.
    Returns (optimal_value, rounds_per_request).
    """
    n = len(deltas)
    budget = min(budget, sum(len(d) for d in deltas))

    @lru_cache(maxsize=None)
    def best(state, b):
        if b == 0:
            return 0.0
        v = 0.0
        for i in range(n):
            h = state[i]
            if h < len(deltas[i]):
                nxt = list(state)
                nxt[i] = h + 1
                cand = deltas[i][h] + best(tuple(nxt), b - 1)
                if cand > v:
                    v = cand
        return v

    state = tuple([0] * n)
    value = best(state, budget)
    rounds = [0] * n
    b = budget
    while b > 0:
        cur = best(state, b)
        if cur <= 1e-12:
            break
        for i in range(n):
            h = state[i]
            if h < len(deltas[i]):
                nxt = list(state)
                nxt[i] = h + 1
                nxt = tuple(nxt)
                if cur - (deltas[i][h] + best(nxt, b - 1)) <= 1e-9:
                    state = nxt
                    rounds[i] += 1
                    b -= 1
                    break
    return value, rounds


def index_allocation(deltas, budget, cost=1.0):
    """Single-step gain index I_i = d1_i / c1, recomputed after every round.

    Requests with d1 <= 0 are not funded; budget is not force-spent.
    Returns (total_improvement, rounds_per_request, used_budget, steps).
    """
    n = len(deltas)
    state = [0] * n
    total, used = 0.0, 0
    steps = []
    while used < budget:
        best_i, best_idx = -1, 0.0
        for i in range(n):
            h = state[i]
            if h < len(deltas[i]):
                idx = deltas[i][h] / cost
                if idx > best_idx:
                    best_idx, best_i = idx, i
        if best_i < 0:
            break
        h = state[best_i]
        total += deltas[best_i][h]
        state[best_i] += 1
        used += 1
        steps.append({"request": best_i, "round": h + 1,
                      "index": best_idx, "gain": deltas[best_i][h]})
    return total, state, used, steps


def two_step_package_allocation(deltas, budget, cost=1.0):
    """Fixed two-step package index d2/c2; a package reserves two rounds.

    Falls back to the single-step index when no full package is affordable
    or available. Returns (total, rounds_per_request, used_budget).
    """
    n = len(deltas)
    state = [0] * n
    total, used = 0.0, 0
    while used < budget:
        best_i, best_idx, package = -1, 0.0, False
        for i in range(n):
            h = state[i]
            if h + 2 <= len(deltas[i]) and used + 2 <= budget:
                idx = (deltas[i][h] + deltas[i][h + 1]) / (2.0 * cost)
                if idx > best_idx:
                    best_idx, best_i, package = idx, i, True
        if best_i < 0:
            for i in range(n):
                h = state[i]
                if h < len(deltas[i]):
                    idx = deltas[i][h] / cost
                    if idx > best_idx:
                        best_idx, best_i, package = idx, i, False
        if best_i < 0:
            break
        h = state[best_i]
        if package:
            total += deltas[best_i][h] + deltas[best_i][h + 1]
            state[best_i] += 2
            used += 2
        else:
            total += deltas[best_i][h]
            state[best_i] += 1
            used += 1
    return total, state, used


def round_robin_allocation(deltas, budget):
    """Fair equal-depth baseline, blind to gains; spends while anyone has rounds."""
    n = len(deltas)
    state = [0] * n
    total, used = 0.0, 0
    i = 0
    while used < budget and any(state[j] < len(deltas[j]) for j in range(n)):
        if state[i] < len(deltas[i]):
            total += deltas[i][state[i]]
            state[i] += 1
            used += 1
        i = (i + 1) % n
    return total, state, used


def random_allocation(deltas, budget, rng):
    """Random extra allocation, blind to gains; controls for the budget itself."""
    n = len(deltas)
    state = [0] * n
    total, used = 0.0, 0
    while used < budget:
        active = [j for j in range(n) if state[j] < len(deltas[j])]
        if not active:
            break
        j = rng.choice(active)
        total += deltas[j][state[j]]
        state[j] += 1
        used += 1
    return total, state, used


# --------------------------------------------------------------------------
# Trajectory families and the delayed-benefit failure instances
# --------------------------------------------------------------------------

def make_diminishing(rng, H):
    a = rng.uniform(0.2, 1.5)
    r = rng.uniform(0.4, 0.85)
    return [a * (r ** h) for h in range(H)]


def make_delayed(rng, H):
    if H < 2:
        return make_diminishing(rng, H)
    delay = rng.randint(1, H - 1)
    cost = rng.uniform(0.05, 0.3)
    peak = rng.uniform(1.0, 3.0)
    return ([-cost] * delay
            + [peak * (0.6 ** k) for k in range(H - delay)])


def sample_instance(rng, n, H, p_delayed=0.3):
    return [make_delayed(rng, H) if rng.random() < p_delayed
            else make_diminishing(rng, H) for _ in range(n)]


def delayed_benefit_instances():
    """Hand-built invest-first-benefit-later cases; both failures are kept.

    deep_delay: the gain arrives at round 3, so even the two-step package
    index at h=0 is negative and both greedy variants walk away.
    shallow_delay: the gain arrives at round 2, inside the package window,
    so d2/c2 invests while d1/c1 does not.
    """
    return {
        "deep_delay": {
            "deltas": [[-0.2, -0.2, 5.0, 0.5], [0.4, 0.25, 0.15, 0.05]],
            "budget": 4,
        },
        "shallow_delay": {
            "deltas": [[-0.1, 2.0, 0.4, 0.2], [0.4, 0.25, 0.15, 0.05]],
            "budget": 4,
        },
    }


def evaluate_instance(deltas, budget, rng):
    opt, opt_rounds = exact_dp(deltas, budget)
    idx_total, idx_rounds, idx_used, _ = index_allocation(deltas, budget)
    pkg_total, _, pkg_used = two_step_package_allocation(deltas, budget)
    rr_total, _, rr_used = round_robin_allocation(deltas, budget)
    rnd_total, _, rnd_used = random_allocation(deltas, budget, rng)

    def gap(v):
        return (opt - v) / opt if opt > 1e-12 else 0.0

    return {
        "optimal": opt, "optimal_rounds": opt_rounds,
        "heuristics": {
            "index_d1_c1": {"value": idx_total, "used": idx_used,
                            "rel_gap": gap(idx_total)},
            "two_step_package_d2_c2": {"value": pkg_total, "used": pkg_used,
                                       "rel_gap": gap(pkg_total)},
            "round_robin": {"value": rr_total, "used": rr_used,
                            "rel_gap": gap(rr_total)},
            "random": {"value": rnd_total, "used": rnd_used,
                       "rel_gap": gap(rnd_total)},
        },
    }


def dp_gap_curve(seed=20261025, n_instances=20, N=6, H=4,
                 budgets=(2, 4, 6, 8, 10, 12)):
    rng = random.Random(seed)
    curve = []
    for B in budgets:
        agg = {}
        for _ in range(n_instances):
            deltas = sample_instance(rng, N, H)
            row = evaluate_instance(deltas, B, rng)
            for name, h in row["heuristics"].items():
                a = agg.setdefault(name, {"gaps": [], "used": 0})
                a["gaps"].append(h["rel_gap"])
                a["used"] += h["used"]
        curve.append({
            "budget": B,
            "n_instances": n_instances,
            "heuristics": {
                name: {
                    "mean_rel_gap": sum(a["gaps"]) / len(a["gaps"]),
                    "max_rel_gap": max(a["gaps"]),
                    "mean_used_budget": a["used"] / n_instances,
                } for name, a in agg.items()
            },
        })
    return {"N": N, "H": H, "seed": seed, "curve": curve}


# --------------------------------------------------------------------------
# SeqTopK public-formula audit (formula only; repo implementation not reviewed)
# --------------------------------------------------------------------------

def seqtopk_v2_trace(score_seq, K):
    """Replay the prefix top-(mK) rule on a per-token score sequence.

    At token m the top-(mK) experts of the cumulative score are selected;
    calls spent on earlier tokens belong to those tokens' hidden states and
    cannot be withdrawn, so token m really executes |S_m| fresh calls. The
    'virtual_new' column is the set-growth accounting that understates
    executed calls; the hard check is on real_cum_executed <= m*K.
    """
    E = len(score_seq[0])
    cum = [0.0] * E
    union = set()
    real_total = 0
    rows = []
    for m, s in enumerate(score_seq, start=1):
        cum = [a + b for a, b in zip(cum, s)]
        order = sorted(range(E), key=lambda e: (-cum[e], e))
        S = set(order[:min(m * K, E)])
        real_total += len(S)
        rows.append({
            "token": m,
            "prefix_set": sorted(S),
            "executed_calls": len(S),
            "virtual_new": len(S - union),
            "real_cum_executed": real_total,
            "prefix_budget": m * K,
            "violation": real_total > m * K,
        })
        union |= S
    return rows


def seqtopk_counterexample():
    scores = [[0.40, 0.30, 0.20, 0.10], [0.49, 0.48, 0.02, 0.01]]
    K = 1
    trace = seqtopk_v2_trace(scores, K)
    ledger_entries, ledger_violations = simulate_ledger(
        K, [row["executed_calls"] for row in trace])
    return {
        "E": len(scores[0]), "K": K, "scores": scores,
        "formula_trace": trace,
        "real_executed_total": trace[-1]["real_cum_executed"],
        "prefix_budget_total": len(scores) * K,
        "violation": any(row["violation"] for row in trace),
        "virtual_accounting_total": sum(r["virtual_new"] for r in trace),
        "hard_ledger_replay": ledger_entries,
        "hard_ledger_violations": ledger_violations,
        "note": ("Audits the public v2 eq. (7) formula only; the authors' "
                 "repository implementation was not reviewed, so this is not "
                 "claimed as an author-acknowledged code bug."),
    }


# --------------------------------------------------------------------------
# Scenario B: streaming per-token hard ledger
# --------------------------------------------------------------------------

def simulate_ledger(K, desired_calls, carryover_cap=None):
    """Hard budget ledger: R_m = R_{m-1} + K - q_m >= 0, R_0 = 0.

    Executed calls are debited only when affordable; insufficient balance
    clamps q_m (price signals never replace this check). With a carryover
    cap, overflow above the cap is voided and not counted as executed.
    The prefix invariant cum_executed <= K*m is checked on every prefix.
    Returns (entries, violations).
    """
    R = 0
    cum = 0
    entries = []
    violations = []
    for m, want in enumerate(desired_calls, start=1):
        R += K
        q = min(want, R)
        R -= q
        voided = 0
        if carryover_cap is not None and R > carryover_cap:
            voided = R - carryover_cap
            R = carryover_cap
        cum += q
        if cum > K * m or R < 0:
            violations.append(m)
        entries.append({
            "token": m, "supply": K, "requested": want, "executed": q,
            "voided_overflow": voided, "balance": R,
            "cum_executed": cum, "prefix_budget": K * m,
        })
    return entries, violations


def ledger_demo():
    # K >= 2 guarantees the mandatory base Top-2 round every token; K = 3
    # leaves 1 credit/token of slack so surpluses and the cap are exercised.
    K = 3
    desired = [2, 2, 2, 2, 2, 4, 2, 2, 2, 6]
    uncapped, v1 = simulate_ledger(K, desired)
    capped, v2 = simulate_ledger(K, desired, carryover_cap=4)
    return {
        "K": K, "desired_calls": desired, "carryover_cap": 4,
        "uncapped": uncapped, "capped": capped,
        "uncapped_violations": v1, "capped_violations": v2,
        "prefix_invariant_holds": not v1 and not v2,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(seed=20261025):
    rng = random.Random(seed)
    delayed = {}
    for name, inst in delayed_benefit_instances().items():
        row = evaluate_instance(inst["deltas"], inst["budget"], rng)
        delayed[name] = {"deltas": inst["deltas"], "budget": inst["budget"],
                         **row}
    return {
        "experiment": "E07_synthetic",
        "date": datetime.date.today().isoformat(),
        "seed": seed,
        "scenario_a_exact_dp": {
            "model": ("N<=8 requests, H<=4 rounds each, equal cost per extra "
                      "Top-2 group call; exact DP over (rounds_used, budget) "
                      "is the computable-range upper bound only"),
            "delayed_benefit_instances": delayed,
            "dp_gap": dp_gap_curve(seed=seed),
        },
        "seqtopk_counterexample": seqtopk_counterexample(),
        "scenario_b_hard_ledger": ledger_demo(),
        "service_load": {
            "status": "not_executed",
            "reason": ("Real-LM service-window load (E03-calibrated delta "
                       "controller, 64-request x 4-budget trace replay, "
                       "request-reorder / batch-size / length-mix "
                       "sensitivity, GPU same-round bucketing) requires the "
                       "serving stack and is outside this synthetic script."),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="E07 synthetic budget scheduling")
    ap.add_argument("--output", required=True, help="path for the JSON results")
    ap.add_argument("--seed", type=int, default=20261025)
    args = ap.parse_args(argv)
    result = run(seed=args.seed)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    gap = result["scenario_a_exact_dp"]["dp_gap"]["curve"]
    worst = max(h["max_rel_gap"] for row in gap for h in row["heuristics"].values())
    print(f"wrote {args.output}")
    print(f"service_load: {result['service_load']['status']}")
    print(f"seqtopk counterexample violation: "
          f"{result['seqtopk_counterexample']['violation']}")
    print(f"ledger prefix invariant holds: "
          f"{result['scenario_b_hard_ledger']['prefix_invariant_holds']}")
    print(f"worst max_rel_gap over curve: {worst:.4f}")


if __name__ == "__main__":
    main()
