"""E08 sub-experiments C/D/E: capacity & prices, switching stability, audit gates.

Implements the synthetic-mechanism checks of
plans/loop_moe_research_plan_2026-10-25/experiments/E08_stability_and_synthetic_mechanisms.md
  C1 (sec.4) fixed-state capacity solver verification (N=8 enumeration, N=32 exact DP)
  C2 (sec.4) drifting value matrices, empirical stability only
  D1 (sec.5) C6 two-stable-matrix switching counterexample + stabilizers
  D2 (sec.5) gated by plan sec.1/sec.5(D3): learning prototype starts only after
             main-line stability phenomena are observed; recorded as not-run.
  E1 (sec.6) prefix-invariance causality audit of routing rules
  E2 (sec.6) execution-layer budget ledger incl. the C8 counterexample
  E3 (sec.6) order vs architecture separation and Top-2 pair counting

All instances are small synthetic CPU problems. Results verify controller
mechanisms or refute over-strong claims; they do not imply anything about
natural-language models.

Usage: python3 experiments/e08_cde.py --output runs/e08_cde_cde.json [--only c1,c2,d1,e1,e2,e3]
"""

import argparse
import itertools
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

PLAN_PATH = "plans/loop_moe_research_plan_2026-10-25/experiments/E08_stability_and_synthetic_mechanisms.md"
SYNTHETIC_SCOPE = (
    "All numbers come from small synthetic CPU instances constructed for mechanism "
    "verification and falsification (plan E08 sections 4-6). They validate or refute "
    "controller properties (feasibility, causality, ledger correctness, switching "
    "stability). They are not evidence that any mechanism helps natural-language "
    "modeling, and no language-model training was performed."
)

PENALTY = 2.0    # marginal overload congestion cost c_e(j) = PENALTY * max(0, j - cap_e)
STOP_COST = 1.0  # explicit social cost of the stop/dummy action


# ---------------------------------------------------------------------------
# C1 helpers: congestion game over Top-1 assignments with a stop/dummy expert.
# The dummy expert has value 0, unlimited capacity and explicit STOP_COST; the
# solver never silently drops tokens.
# ---------------------------------------------------------------------------

def make_scores(kind, n, e, rng):
    if kind == "uniform":
        return rng.random((n, e))
    if kind == "concentrated":
        v = rng.random((n, e)) * 0.2
        hot = np.where(rng.random(n) < 0.7, 0, rng.integers(0, e, n))
        v[np.arange(n), hot] += 1.0
        return v
    if kind == "close_second":
        v = rng.random((n, e))
        best = v.argmax(axis=1)
        for i in range(n):
            others = [x for x in range(e) if x != best[i]]
            second = others[int(rng.integers(0, len(others)))]
            v[i, second] = v[i, best[i]] - 0.05
        return v
    raise ValueError(kind)


def augment(values):
    return np.concatenate([values, np.zeros((values.shape[0], 1))], axis=1)


def profile_loads(profile, e_real):
    loads = [0] * e_real
    for a in profile:
        if a < e_real:
            loads[a] += 1
    return loads


def overload_of(profile, caps):
    return sum(max(0, l - c) for l, c in zip(profile_loads(profile, len(caps)), caps))


def total_value(profile, v_aug):
    return float(sum(v_aug[n, a] for n, a in enumerate(profile)))


def congestion_cost(profile, caps):
    return sum(
        PENALTY * max(0, j - c)
        for l, c in zip(profile_loads(profile, len(caps)), caps)
        for j in range(1, l + 1)
    )


def potential_of(profile, v_aug, caps):
    return total_value(profile, v_aug) - congestion_cost(profile, caps)


def social_cost_of(profile, caps):
    loads = profile_loads(profile, len(caps))
    delay = sum(l * PENALTY * max(0, l - c) for l, c in zip(loads, caps))
    stops = sum(1 for a in profile if a == len(caps))
    return delay + STOP_COST * stops


def token_utility(v_aug, caps, profile, n, e):
    e_real = len(caps)
    if e == e_real:
        return float(v_aug[n, e])
    load = sum(1 for m, a in enumerate(profile) if a == e and m != n) + 1
    return float(v_aug[n, e] - PENALTY * max(0, load - caps[e]))


def greedy_feasible_assign(weight, caps):
    """Assign each token to its best expert with remaining capacity, else dummy."""
    n_tok, e_real = weight.shape
    remaining = list(caps)
    profile = [None] * n_tok
    ordered = np.sort(weight, axis=1)
    margin = ordered[:, -1] - (ordered[:, -2] if e_real > 1 else 0.0)
    for n in np.argsort(-margin, kind="stable"):
        for e in np.argsort(-weight[n], kind="stable"):
            if remaining[e] > 0:
                profile[n] = int(e)
                remaining[e] -= 1
                break
        else:
            profile[n] = e_real
    return profile


def exact_capacity_dp(values, caps):
    """Exact capacity-constrained optimum via DP over used-capacity tuples."""
    n_tok, e_real = values.shape
    states = {tuple([0] * e_real): (0.0, [])}
    for n in range(n_tok):
        new = {}
        for used, (val, path) in states.items():
            cands = [(used, val, e_real)]
            for e in range(e_real):
                if used[e] < caps[e]:
                    u2 = list(used)
                    u2[e] += 1
                    cands.append((tuple(u2), val + float(values[n, e]), e))
            for key, v2, e in cands:
                if key not in new or new[key][0] < v2:
                    new[key] = (v2, path + [e])
        states = new
    val, path = max(states.values(), key=lambda t: t[0])
    return val, path


def exact_capacity_value(values, caps):
    """Value-only vectorized DP (mixed-radix state array); same optimum as
    exact_capacity_dp but without profile backtracking."""
    shape = tuple(c + 1 for c in caps)
    dp = np.full(shape, -np.inf)
    dp[(0,) * len(caps)] = 0.0
    for n in range(values.shape[0]):
        new = dp.copy()
        for e, cap in enumerate(caps):
            shifted = np.full(shape, -np.inf)
            dst = [slice(None)] * len(caps)
            src = [slice(None)] * len(caps)
            dst[e] = slice(1, cap + 1)
            src[e] = slice(0, cap)
            shifted[tuple(dst)] = dp[tuple(src)] + values[n, e]
            new = np.maximum(new, shifted)
        dp = new
    return float(dp.max())


def method_independent_top1(v_aug, caps):
    return v_aug.argmax(axis=1).tolist()


def method_sync_best_response(v_aug, caps, max_rounds=50):
    profile = v_aug.argmax(axis=1).tolist()
    n_tok, e_all = v_aug.shape
    potentials = [potential_of(profile, v_aug, caps)]
    overloads = [overload_of(profile, caps)]
    visited = {tuple(profile)}
    cycled = False
    converged = False
    for _ in range(max_rounds):
        new_profile = list(profile)
        for n in range(n_tok):
            cur_u = token_utility(v_aug, caps, profile, n, profile[n])
            best_e, best_u = profile[n], cur_u
            for e in range(e_all):
                u = token_utility(v_aug, caps, profile, n, e)
                if u > best_u + 1e-12:
                    best_e, best_u = e, u
            new_profile[n] = best_e
        if new_profile == profile:
            converged = True
            break
        profile = new_profile
        potentials.append(potential_of(profile, v_aug, caps))
        overloads.append(overload_of(profile, caps))
        key = tuple(profile)
        if key in visited:
            cycled = True
            break
        visited.add(key)
    return profile, {"potentials": potentials, "per_round_overload": overloads,
                     "cycled": cycled, "converged": converged,
                     "rounds": len(potentials) - 1}


def method_async_strict_best_response(v_aug, caps, max_sweeps=200):
    profile = v_aug.argmax(axis=1).tolist()
    n_tok, e_all = v_aug.shape
    potentials = [potential_of(profile, v_aug, caps)]
    for _ in range(max_sweeps):
        changed = False
        for n in range(n_tok):
            cur = profile[n]
            cur_u = token_utility(v_aug, caps, profile, n, cur)
            best_e, best_u = cur, cur_u
            for e in range(e_all):
                if e == cur:
                    continue
                u = token_utility(v_aug, caps, profile, n, e)
                if u > best_u + 1e-12:
                    best_e, best_u = e, u
            if best_e != cur:
                profile[n] = best_e
                potentials.append(potential_of(profile, v_aug, caps))
                changed = True
        if not changed:
            break
    monotone = all(potentials[i + 1] > potentials[i] - 1e-9 for i in range(len(potentials) - 1))
    return profile, {"potentials": potentials, "potential_monotone": monotone,
                     "updates": len(potentials) - 1}


def method_one_price_update(v_aug, caps, eta=1.0):
    e_real = len(caps)
    top1 = v_aug[:, :e_real].argmax(axis=1).tolist()
    loads = profile_loads(top1, e_real)
    prices = np.array([eta * max(0, l - c) for l, c in zip(loads, caps)])
    adjusted = v_aug.copy()
    adjusted[:, :e_real] -= prices
    return adjusted.argmax(axis=1).tolist(), prices.tolist()


def method_price_hard_fix(v_aug, caps, eta=1.0):
    e_real = len(caps)
    top1 = v_aug[:, :e_real].argmax(axis=1).tolist()
    loads = profile_loads(top1, e_real)
    prices = np.array([eta * max(0, l - c) for l, c in zip(loads, caps)])
    adjusted = v_aug[:, :e_real] - prices
    profile = greedy_feasible_assign(adjusted, caps)
    return profile, prices.tolist()


def method_soft_primal_dual(values, caps, k_iters, tau=0.05, eta=0.5):
    n_tok, e_real = values.shape
    caps_arr = np.asarray(caps, dtype=float)
    prices = np.zeros(e_real)
    avg = np.zeros((n_tok, e_real))
    last_overloads, avg_overloads = [], []
    for k in range(k_iters):
        z = (values - prices) / tau
        z -= z.max(axis=1, keepdims=True)
        x = np.exp(z)
        x /= x.sum(axis=1, keepdims=True)
        load = x.sum(axis=0)
        prices = np.maximum(0.0, prices + eta * (load - caps_arr))
        avg += x
        last_overloads.append(float(np.maximum(0.0, load - caps_arr).sum()))
        avg_overloads.append(float(np.maximum(0.0, avg.sum(axis=0) / (k + 1) - caps_arr).sum()))
    rounded = greedy_feasible_assign(avg / k_iters, caps)
    return {"last_iterate_soft_overload": last_overloads,
            "average_iterate_soft_overload": avg_overloads,
            "final_prices": prices.tolist(),
            "rounded_profile": rounded}


def enumerate_profiles(values, caps):
    """Exhaustive E^N enumeration: value, potential, feasibility, Nash count."""
    n_tok, e_real = values.shape
    v_aug = augment(values)
    best_feasible = None
    best_unconstrained = -math.inf
    nash_count = 0
    for prof in itertools.product(range(e_real), repeat=n_tok):
        prof = list(prof)
        val = total_value(prof, v_aug)
        best_unconstrained = max(best_unconstrained, val)
        if overload_of(prof, caps) == 0:
            best_feasible = val if best_feasible is None else max(best_feasible, val)
        if all(
            token_utility(v_aug, caps, prof, n, prof[n])
            >= max(token_utility(v_aug, caps, prof, n, e) for e in range(e_real + 1)) - 1e-12
            for n in range(n_tok)
        ):
            nash_count += 1
    return {"profiles": e_real ** n_tok,
            "best_feasible_value_no_dummy": best_feasible,
            "best_unconstrained_value": best_unconstrained,
            "nash_equilibria_count": nash_count}


def c1_instances(seed):
    rng = np.random.default_rng(seed)
    specs = []
    for kind in ("uniform", "concentrated", "close_second"):
        v8 = make_scores(kind, 8, 3, rng)
        for capname, caps in (("uniform_caps", [3, 3, 3]),
                              ("skewed_caps", [4, 2, 2]),
                              ("insufficient_caps", [3, 2, 2])):
            specs.append({"name": f"n8_{kind}_{capname}", "values": v8, "caps": caps,
                          "enumerate": True})
        v32 = make_scores(kind, 32, 4, rng)
        for capname, caps in (("uniform_caps", [9, 9, 9, 9]),
                              ("skewed_caps", [14, 8, 6, 4]),
                              ("insufficient_caps", [10, 8, 6, 4])):
            specs.append({"name": f"n32_{kind}_{capname}", "values": v32, "caps": caps,
                          "enumerate": False})
    return specs


def evaluate_profile(profile, v_aug, caps, oracle_value):
    value = total_value(profile, v_aug)
    overload = overload_of(profile, caps)
    return {"profile": [int(a) for a in profile],
            "value": value,
            "feasible": overload == 0,
            "overload": overload,
            "value_gap_vs_oracle": oracle_value - value,
            "potential": potential_of(profile, v_aug, caps),
            "social_cost": social_cost_of(profile, caps),
            "stop_dummy_count": sum(1 for a in profile if a == len(caps))}


def run_c1_instance(spec):
    values, caps = spec["values"], spec["caps"]
    v_aug = augment(values)
    t0 = time.perf_counter()
    oracle_value, oracle_profile = exact_capacity_dp(values, caps)
    oracle_ms = (time.perf_counter() - t0) * 1e3

    methods = {}
    t0 = time.perf_counter()
    p = method_independent_top1(v_aug, caps)
    methods["independent_top1"] = evaluate_profile(p, v_aug, caps, oracle_value)
    methods["independent_top1"]["time_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    p, info = method_sync_best_response(v_aug, caps)
    methods["sync_best_response"] = evaluate_profile(p, v_aug, caps, oracle_value)
    methods["sync_best_response"].update(info)
    methods["sync_best_response"]["potential_monotone"] = all(
        info["potentials"][i + 1] >= info["potentials"][i] - 1e-9
        for i in range(len(info["potentials"]) - 1))
    methods["sync_best_response"]["time_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    p, info = method_async_strict_best_response(v_aug, caps)
    methods["async_strict_best_response"] = evaluate_profile(p, v_aug, caps, oracle_value)
    methods["async_strict_best_response"].update(info)
    methods["async_strict_best_response"]["time_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    p, prices = method_one_price_update(v_aug, caps)
    methods["one_price_update"] = evaluate_profile(p, v_aug, caps, oracle_value)
    methods["one_price_update"]["prices"] = prices
    methods["one_price_update"]["time_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    p, prices = method_price_hard_fix(v_aug, caps)
    methods["price_preference_hard_capacity_fix"] = evaluate_profile(p, v_aug, caps, oracle_value)
    methods["price_preference_hard_capacity_fix"]["prices"] = prices
    methods["price_preference_hard_capacity_fix"]["time_ms"] = (time.perf_counter() - t0) * 1e3

    soft = {}
    for k in (1, 2, 4, 8, 32, 128):
        t0 = time.perf_counter()
        r = method_soft_primal_dual(values, caps, k)
        entry = evaluate_profile(r.pop("rounded_profile"), v_aug, caps, oracle_value)
        entry.update(r)
        entry["time_ms"] = (time.perf_counter() - t0) * 1e3
        soft[str(k)] = entry
    methods["soft_primal_dual"] = soft

    methods["exact_capacity_matching"] = evaluate_profile(oracle_profile, v_aug, caps, oracle_value)
    methods["exact_capacity_matching"]["time_ms"] = oracle_ms

    out = {"name": spec["name"], "n_tokens": int(values.shape[0]),
           "n_experts": int(values.shape[1]), "caps": caps,
           "total_real_capacity": sum(caps),
           "oracle_value": oracle_value,
           "methods": methods}
    if spec["enumerate"]:
        t0 = time.perf_counter()
        enum = enumerate_profiles(values, caps)
        enum["time_ms"] = (time.perf_counter() - t0) * 1e3
        if enum["best_feasible_value_no_dummy"] is not None:
            enum["dp_matches_enumeration"] = abs(
                enum["best_feasible_value_no_dummy"] - oracle_value) < 1e-9
        else:
            enum["dp_matches_enumeration"] = None
        out["enumeration"] = enum
    return out


def run_c1(seed=0):
    instances = [run_c1_instance(s) for s in c1_instances(seed)]
    gate = {
        "hard_capacity_fix_feasible_all_instances": all(
            inst["methods"]["price_preference_hard_capacity_fix"]["feasible"]
            for inst in instances),
        "dp_oracle_matches_enumeration_all_n8": all(
            inst["enumeration"]["dp_matches_enumeration"] in (True, None)
            for inst in instances if "enumeration" in inst),
    }
    return {"description": "C1 fixed-state capacity solver verification; values, potentials, "
                           "social cost and feasibility computed exactly; dummy/stop action has "
                           "explicit cost and is never silent.",
            "penalty": PENALTY, "stop_cost": STOP_COST,
            "instances": instances, "gates": gate}


# ---------------------------------------------------------------------------
# C2: value matrix drifts after each batch of expert calls. Empirical
# stability only; fixed-problem theorems are not carried over (plan sec.4).
# ---------------------------------------------------------------------------

C2_DELTAS = (0.0, 0.01, 0.1, 1.0)
C2_PATTERNS = ("smooth", "alternating_hotspot", "burst_hotspot")


def c2_drift(values, pattern, delta, t, mid, state, rng):
    if delta == 0.0:
        return
    if pattern == "smooth":
        values += delta * 0.1 * rng.standard_normal(values.shape)
    elif pattern == "alternating_hotspot":
        hot = t % 2
        values[:, hot] += delta
        if t > 0:
            values[:, 1 - hot] -= delta
    elif pattern == "burst_hotspot":
        if t == mid:
            state["burst"] = 3.0 * delta
            values[:, 2] += state["burst"]
        elif t > mid:
            values[:, 2] -= state["burst"]
            state["burst"] *= 0.8
            values[:, 2] += state["burst"]
    else:
        raise ValueError(pattern)


def run_c2_cell(values0, caps, pattern, delta, method, rounds, rng, eta=0.5):
    n_tok, e_real = values0.shape
    values = values0.copy()
    prices = np.zeros(e_real)
    state = {}
    mid = rounds // 2
    pre_overloads, realized_values, regrets, decision_ms = [], [], [], []
    for t in range(rounds):
        scores = values - prices if method == "price_hard_fix" else values
        t0 = time.perf_counter()
        pre_profile = scores.argmax(axis=1).tolist()
        pre_overload = overload_of(pre_profile, caps)
        profile = greedy_feasible_assign(scores, caps)
        decision_ms.append((time.perf_counter() - t0) * 1e3)
        assert overload_of(profile, caps) == 0, "hard capacity fix must be feasible"
        oracle_value = exact_capacity_value(values, caps)
        realized = sum(float(values[n, a]) for n, a in enumerate(profile) if a < e_real)
        realized_values.append(realized)
        regrets.append(oracle_value - realized)
        pre_overloads.append(pre_overload)
        if method == "price_hard_fix":
            loads = np.array(profile_loads(pre_profile, e_real), dtype=float)
            prices = np.maximum(0.0, prices + eta * (loads - np.asarray(caps, dtype=float)))
        c2_drift(values, pattern, delta, t, mid, state, rng)
    return {"pattern": pattern, "delta": delta, "method": method,
            "rounds": rounds,
            "per_round_pre_fix_overload": pre_overloads,
            "avg_pre_fix_overload": float(np.mean(pre_overloads)),
            "max_pre_fix_overload": int(max(pre_overloads)),
            "post_fix_overload_sum": 0,
            "total_realized_value": float(sum(realized_values)),
            "total_regret_vs_per_round_oracle": float(sum(regrets)),
            "avg_decision_ms": float(np.mean(decision_ms)),
            "max_decision_ms": float(max(decision_ms))}


def run_c2(seed=1, rounds=20, decision_budget_ms=50.0):
    rng = np.random.default_rng(seed)
    n_tok, e_real = 32, 4
    caps = [8, 8, 8, 8]
    values0 = rng.random((n_tok, e_real))
    cells = []
    for pattern in C2_PATTERNS:
        for delta in C2_DELTAS:
            for method in ("top1_hard_fix", "price_hard_fix"):
                cell_rng = np.random.default_rng(seed + 1000)
                cells.append(run_c2_cell(values0, caps, pattern, delta, method,
                                         rounds, cell_rng))
    gate = {
        "hard_capacity_fix_feasible_all_cells": all(
            c["post_fix_overload_sum"] == 0 for c in cells),
        "max_decision_ms": max(c["max_decision_ms"] for c in cells),
        "decision_budget_ms": decision_budget_ms,
    }
    gate["controller_overhead_within_budget"] = gate["max_decision_ms"] <= decision_budget_ms
    gate["ready_for_real_model"] = (
        gate["hard_capacity_fix_feasible_all_cells"]
        and gate["controller_overhead_within_budget"])
    return {"description": "C2 drifting-value empirical stability. Prices update per batch; "
                           "value drift applies after each round of executed expert calls. "
                           "No fixed-problem theorem is transferred to this dynamic setting.",
            "n_tokens": n_tok, "n_experts": e_real, "caps": caps,
            "price_timescale": "per_batch", "value_drift_timescale": "per_expert_call_round",
            "cells": cells, "gates": gate}


# ---------------------------------------------------------------------------
# D1: C6 switching counterexample. A and B each have spectral radius 0.5 but
# rho(BA) = 2.25 + sqrt(5) > 1. All computations in float64.
# ---------------------------------------------------------------------------

D1_A = np.array([[0.5, 2.0], [0.0, 0.5]])
D1_B = np.array([[0.5, 0.0], [2.0, 0.5]])
D1_H0 = np.array([1.0, 1.0])
D1_LENGTHS = (1, 2, 4, 8, 16, 32)


def spectral_radius(m):
    return float(max(abs(np.linalg.eigvals(m))))


def d1_route_mats(name, k, rng=None):
    if name == "AAAA":
        return [D1_A] * k
    if name == "BBBB":
        return [D1_B] * k
    if name == "ABAB":
        return [D1_A if i % 2 == 0 else D1_B for i in range(k)]
    if name == "random":
        return [D1_A if r < 0.5 else D1_B for r in rng.random(k)]
    raise ValueError(name)


def d1_run_route(mats, h0=D1_H0):
    h = h0.astype(np.float64).copy()
    product = np.eye(2)
    norms = [float(np.linalg.norm(h))]
    for m in mats:
        h = m @ h
        product = m @ product
        norms.append(float(np.linalg.norm(h)))
    return {"norms": norms, "final_norm": norms[-1],
            "product_spectral_radius": spectral_radius(product)}


def d1_greedy_route(mats_pool, k, h0=D1_H0):
    h = h0.astype(np.float64).copy()
    chosen = []
    for _ in range(k):
        cands = [(float(np.linalg.norm(m @ h)), idx) for idx, m in enumerate(mats_pool)]
        _, idx = min(cands)
        h = mats_pool[idx] @ h
        chosen.append(idx)
    return chosen


def run_d1():
    check = {
        "spectral_radius_A": spectral_radius(D1_A),
        "spectral_radius_B": spectral_radius(D1_B),
        "spectral_radius_BA": spectral_radius(D1_B @ D1_A),
        "spectral_radius_BA_closed_form": 2.25 + math.sqrt(5.0),
    }
    routes = {}
    for k in D1_LENGTHS:
        rng = np.random.default_rng(1000 + k)
        entry = {}
        for name in ("AAAA", "BBBB", "ABAB", "random"):
            r = d1_run_route(d1_route_mats(name, k, rng))
            r.pop("norms")
            entry[name] = r
        chosen = d1_greedy_route([D1_A, D1_B], k)
        g = d1_run_route([ [D1_A, D1_B][i] for i in chosen ])
        g.pop("norms")
        g["chosen_route"] = "".join("A" if i == 0 else "B" for i in chosen)
        entry["greedy_min_norm_growth"] = g
        routes[str(k)] = entry

    damping = {}
    for alpha in (0.1, 0.25, 0.5, 1.0):
        a_d = (1 - alpha) * np.eye(2) + alpha * D1_A
        b_d = (1 - alpha) * np.eye(2) + alpha * D1_B
        r = d1_run_route([a_d if i % 2 == 0 else b_d for i in range(32)])
        damping[str(alpha)] = {
            "spectral_radius_A_damped": spectral_radius(a_d),
            "spectral_radius_B_damped": spectral_radius(b_d),
            "spectral_radius_BA_damped": spectral_radius(b_d @ a_d),
            "alternating_k32_final_norm": r["final_norm"],
            "alternating_stable": spectral_radius(b_d @ a_d) < 1.0,
        }

    norm_scaling = {}
    for q in (0.8, 0.95, 1.0, 1.1):
        sig_a = float(np.linalg.svd(D1_A, compute_uv=False)[0])
        sig_b = float(np.linalg.svd(D1_B, compute_uv=False)[0])
        a_s = D1_A * min(1.0, q / sig_a)
        b_s = D1_B * min(1.0, q / sig_b)
        r_alt = d1_run_route([a_s if i % 2 == 0 else b_s for i in range(32)])
        rng = np.random.default_rng(7)
        r_rand = d1_run_route([a_s if x < 0.5 else b_s for x in rng.random(32)])
        norm_scaling[str(q)] = {
            "spectral_norm_A_before": sig_a, "spectral_norm_B_before": sig_b,
            "spectral_norm_A_after": float(np.linalg.svd(a_s, compute_uv=False)[0]),
            "spectral_norm_B_after": float(np.linalg.svd(b_s, compute_uv=False)[0]),
            "alternating_k32_final_norm": r_alt["final_norm"],
            "random_k32_final_norm": r_rand["final_norm"],
        }

    k_oracle = 8
    best_uncon, best_allowed = math.inf, math.inf
    for bits in itertools.product((0, 1), repeat=k_oracle):
        mats = [D1_A if b == 0 else D1_B for b in bits]
        n = d1_run_route(mats)["final_norm"]
        best_uncon = min(best_uncon, n)
        if bits.count(0) in (0, k_oracle):
            best_allowed = min(best_allowed, n)
    switching_graph = {
        "constraint": "only self-transitions allowed (no A<->B switches); the expanding "
                      "combination is forbidden",
        "horizon": k_oracle,
        "allowed_paths": 2,
        "total_paths": 2 ** k_oracle,
        "branching_factor_allowed": 1,
        "branching_factor_unconstrained": 2,
        "best_unconstrained_final_norm": best_uncon,
        "best_allowed_final_norm": best_allowed,
        "oracle_loss": best_allowed - best_uncon,
    }

    findings = {
        "single_expert_stable": check["spectral_radius_A"] < 1 and check["spectral_radius_B"] < 1,
        "alternating_unstable": check["spectral_radius_BA"] > 1,
        "pure_route_k32_norm": routes["32"]["AAAA"]["final_norm"],
        "alternating_k32_norm": routes["32"]["ABAB"]["final_norm"],
        "damping_sufficient_at_all_alpha": all(
            v["alternating_stable"] for v in damping.values()),
    }
    return {"description": "D1 C6 switching counterexample in float64; routes compared at "
                           "lengths 1,2,4,8,16,32 with product spectral radii; stabilizers: "
                           "residual damping, spectral-NORM scaling (not spectral radius), "
                           "switching-graph constraint.",
            "checks": check, "routes_by_length": routes,
            "stabilizers": {"residual_damping": damping,
                            "spectral_norm_scaling": norm_scaling,
                            "switching_graph_constraint": switching_graph},
            "findings": findings}


def run_d2():
    return {"status": "not_run_gated",
            "reason": "Plan sec.1: run CPU exact counterexamples first and pick learning "
                      "experiments only after main-line problems appear. Plan sec.5 (D3) gate: "
                      "invest only if the real model shows (a) systematic NLL degradation with "
                      "longer loops, (b) clear numerical-stability differences across expert "
                      "sequences, or (c) the controller repeatedly choosing reproducible "
                      "expansion paths. None of these signals is present in this run, so the "
                      "small learning-expert prototype (D2) is intentionally not executed.",
            "gate_conditions": ["longer_loop_systematic_nll_degradation",
                                "expert_sequence_stability_differences",
                                "controller_reproducible_expansion_paths"],
            "gate_conditions_met": False}


# ---------------------------------------------------------------------------
# E1: prefix-invariance causality audit. Prefix positions 0..2 are fixed;
# suffix positions 3..5 are swapped between two designed score matrices.
# Routing rules that read suffix scores must show route/logit changes on the
# prefix; causal rules must be bitwise identical. float64, no dropout, fixed
# positions (plan sec.6 E1 and counterexample C7).
# ---------------------------------------------------------------------------

E1_PREFIX = np.array([
    [0.90, 0.85, 0.10, 0.05],
    [0.80, 0.75, 0.20, 0.10],
    [0.70, 0.65, 0.30, 0.15],
])
E1_SUFFIX_A = np.array([
    [0.50, 0.10, 0.05, 0.02],
    [0.48, 0.10, 0.05, 0.02],
    [0.46, 0.10, 0.05, 0.02],
])
E1_SUFFIX_B = np.array([
    [0.10, 5.00, 0.20, 0.15],
    [0.10, 4.90, 0.20, 0.15],
    [0.10, 4.80, 0.20, 0.15],
])
E1_PREFIX_LEN = 3
E1_K = 1


def mech_per_token_top1(scores, k=E1_K):
    return [set(np.argsort(-row, kind="stable")[:k].tolist()) for row in scores]


def mech_prefix_budget_online(scores, k=E1_K):
    """C8-fixed online rule: at step m only rows <= m are read and executed
    calls are limited by the real remaining budget mK - spent. A position with
    no remaining quota gets an empty selection."""
    n_pos, n_exp = scores.shape
    spent = 0
    out = []
    for m in range(1, n_pos + 1):
        ranked = sorted(
            ((scores[t][e], t, e) for t in range(m) for e in range(n_exp)),
            reverse=True)
        current = [(s, e) for s, t, e in ranked[: m * k] if t == m - 1]
        quota = max(0, m * k - spent)
        chosen = sorted(current, reverse=True)[:quota]
        spent += len(chosen)
        out.append({e for _, e in chosen})
    return out


def mech_global_sinkhorn(scores, k=E1_K, iters=50):
    m = scores.copy()
    for _ in range(iters):
        m = m / m.sum(axis=1, keepdims=True)
        m = m / (m.sum(axis=0, keepdims=True) / (m.shape[0] / m.shape[1]))
    return [set(np.argsort(-row, kind="stable")[:k].tolist()) for row in m]


def mech_sequence_topk(scores, k=E1_K):
    n_pos, n_exp = scores.shape
    budget = n_pos * k
    flat = sorted(
        ((scores[t, e], t, e) for t in range(n_pos) for e in range(n_exp)),
        reverse=True)[:budget]
    out = [set() for _ in range(n_pos)]
    for _, t, e in flat:
        out[t].add(e)
    for t in range(n_pos):
        if not out[t]:
            out[t] = {int(scores[t].argmax())}
    return out


def mech_batch_prices(scores, k=E1_K):
    top1 = scores.argmax(axis=1)
    loads = np.array([(top1 == e).sum() for e in range(scores.shape[1])], dtype=float)
    adjusted = scores - loads
    return [set(np.argsort(-row, kind="stable")[:k].tolist()) for row in adjusted]


def mech_expert_choice(scores, k=E1_K):
    n_pos, n_exp = scores.shape
    quota = int(math.ceil(n_pos * k / n_exp))
    assigned = {}
    for e in range(n_exp):
        order = sorted(
            (t for t in range(n_pos) if t not in assigned),
            key=lambda t: -scores[t, e])[:quota]
        for t in order:
            assigned[t] = e
    out = []
    for t in range(n_pos):
        out.append({assigned.get(t, int(scores[t].argmax()))})
    return out


def mech_token_group(scores, k=E1_K, group=2):
    n_pos, n_exp = scores.shape
    order = sorted(range(n_pos), key=lambda t: -scores[t].max())
    out = [set() for _ in range(n_pos)]
    for g in range(0, n_pos, group):
        e = (g // group) % n_exp
        for t in order[g:g + group]:
            out[t] = {e}
    return out


E1_MECHANISMS = {
    "per_token_top1_causal_control": mech_per_token_top1,
    "prefix_budget_online_causal_control": mech_prefix_budget_online,
    "global_sinkhorn": mech_global_sinkhorn,
    "sequence_topk": mech_sequence_topk,
    "batch_prices": mech_batch_prices,
    "expert_choice": mech_expert_choice,
    "token_group_assignment": mech_token_group,
}
E1_CAUSAL_CONTROLS = {"per_token_top1_causal_control", "prefix_budget_online_causal_control"}


def e1_forward(scores_layer1, mech):
    """Two-layer toy model with fixed parameters; returns per-layer selections
    and position-wise logits. Hidden states depend only on position index."""
    rng = np.random.default_rng(42)
    n_pos, n_exp = scores_layer1.shape
    d = 4
    h = rng.standard_normal((n_pos, d)) * 0.5
    experts = [np.diag(rng.random(d) + 0.5) for _ in range(n_exp)]
    w2 = rng.standard_normal((n_exp, d))
    w_out = rng.standard_normal((n_exp, d))

    sel1 = mech(scores_layer1)
    mix1 = np.stack([
        (np.mean([experts[e] @ h[t] for e in sel1[t]], axis=0)
         if sel1[t] else np.zeros(d)) for t in range(n_pos)])
    h1 = h + mix1
    scores2 = h1 @ w2.T
    sel2 = mech(scores2)
    mix2 = np.stack([
        (np.mean([experts[e] @ h1[t] for e in sel2[t]], axis=0)
         if sel2[t] else np.zeros(d)) for t in range(n_pos)])
    h2 = h1 + mix2
    logits = h2 @ w_out.T
    return sel1, sel2, logits


def run_e1(tol=1e-12):
    results = {}
    for name, mech in E1_MECHANISMS.items():
        sel1_a, sel2_a, logits_a = e1_forward(np.vstack([E1_PREFIX, E1_SUFFIX_A]), mech)
        sel1_b, sel2_b, logits_b = e1_forward(np.vstack([E1_PREFIX, E1_SUFFIX_B]), mech)
        mismatch_l1 = sum(sel1_a[t] != sel1_b[t] for t in range(E1_PREFIX_LEN))
        mismatch_l2 = sum(sel2_a[t] != sel2_b[t] for t in range(E1_PREFIX_LEN))
        logit_diff = float(np.max(np.abs(
            logits_a[:E1_PREFIX_LEN] - logits_b[:E1_PREFIX_LEN])))
        violated = mismatch_l1 > 0 or mismatch_l2 > 0 or logit_diff > tol
        results[name] = {
            "prefix_route_mismatches_layer1": mismatch_l1,
            "prefix_route_mismatches_layer2": mismatch_l2,
            "prefix_logit_max_abs_diff": logit_diff,
            "declared_causal_control": name in E1_CAUSAL_CONTROLS,
            "causality_violation_detected": violated,
            "gate_pass": (not violated) if name in E1_CAUSAL_CONTROLS else None,
        }
    summary = {
        "causal_controls_exact_zero": all(
            not results[n]["causality_violation_detected"] for n in E1_CAUSAL_CONTROLS),
        "noncausal_rules_all_flagged": all(
            results[n]["causality_violation_detected"]
            for n in E1_MECHANISMS if n not in E1_CAUSAL_CONTROLS),
    }
    return {"description": "E1 prefix-invariance audit: fixed prefix positions 0..2, suffix "
                           "swapped between two designed score matrices; per-layer prefix "
                           "routes and prefix logits compared in float64 with fixed positions "
                           "and no dropout. Violations disqualify a rule from causal "
                           "autoregressive deployment (system-level shared congestion is a "
                           "separate issue from same-sequence future-token information).",
            "prefix_len": E1_PREFIX_LEN, "n_positions": 6, "n_experts": 4, "k": E1_K,
            "tolerance": tol, "mechanisms": results, "summary": summary}


# ---------------------------------------------------------------------------
# E2: execution-layer budget ledger. Includes the C8 counterexample against
# the published Online SeqTopK budget claim (arXiv:2511.06494v2 eq.(7)) and a
# ledger-fixed variant. Counting is done where experts actually execute, never
# inferred from the current top-set size.
# ---------------------------------------------------------------------------

C8_ROWS = [[0.40, 0.30, 0.20, 0.10],
           [0.49, 0.48, 0.02, 0.01]]


def run_online_seqtopk(score_rows, k, ledger_fix):
    """Returns per-step ledger. selected_in_cache is what a naive top-set
    length inference would report; new_calls is counted at execution."""
    spent = 0
    ledger = []
    for m in range(1, len(score_rows) + 1):
        n_exp = len(score_rows[0])
        ranked = sorted(
            ((score_rows[t][e], t, e) for t in range(m) for e in range(n_exp)),
            reverse=True)
        cache = ranked[: m * k]
        current = sorted(((s, e) for s, t, e in cache if t == m - 1), reverse=True)
        executed = current if not ledger_fix else current[: max(0, m * k - spent)]
        spent += len(executed)
        ledger.append({
            "step": m,
            "budget_mK": m * k,
            "selected_in_cache": len(cache),
            "current_token_candidates": len(current),
            "new_calls": len(executed),
            "executed_experts": [e for _, e in executed],
            "cumulative_calls": spent,
            "remaining_budget": m * k - spent,
        })
    return ledger


def e2_counting_metrics(ledger, params_per_expert=1000, flops_per_call=2048,
                        pad_multiple=4, n_experts=4):
    calls = [(row["step"], e) for row in ledger for e in row["executed_experts"]]
    unique_experts = {e for _, e in calls}
    padded = sum(
        int(math.ceil(1 / pad_multiple) * pad_multiple) * n_experts for _ in ledger)
    return {
        "unique_expert_params": len(unique_experts) * params_per_expert,
        "total_expert_calls": len(calls),
        "token_expert_pairs": len(calls),
        "executed_kernels": len(set(calls)),
        "padded_matrix_elements": padded,
        "estimated_flops": len(calls) * flops_per_call,
    }


def run_e2(seed=3, stress_instances=200):
    c8_unfixed = run_online_seqtopk(C8_ROWS, 1, ledger_fix=False)
    c8_fixed = run_online_seqtopk(C8_ROWS, 1, ledger_fix=True)
    c8 = {
        "rows": C8_ROWS, "k": 1,
        "unfixed_ledger": c8_unfixed,
        "fixed_ledger": c8_fixed,
        "unfixed_total_calls": c8_unfixed[-1]["cumulative_calls"],
        "unfixed_budget_final": c8_unfixed[-1]["budget_mK"],
        "unfixed_exceeds_budget": c8_unfixed[-1]["cumulative_calls"]
                                  > c8_unfixed[-1]["budget_mK"],
        "naive_topset_inference_would_report": c8_unfixed[-1]["selected_in_cache"],
        "naive_inference_wrong": c8_unfixed[-1]["selected_in_cache"]
                                 != c8_unfixed[-1]["cumulative_calls"],
        "fixed_total_calls": c8_fixed[-1]["cumulative_calls"],
        "fixed_within_budget": all(r["cumulative_calls"] <= r["budget_mK"]
                                   for r in c8_fixed),
    }
    rng = np.random.default_rng(seed)
    fixed_violations = 0
    unfixed_violations = 0
    for _ in range(stress_instances):
        n_pos = int(rng.integers(2, 9))
        rows = rng.random((n_pos, 4))
        rows /= rows.sum(axis=1, keepdims=True)
        k = int(rng.integers(1, 3))
        lu = run_online_seqtopk(rows.tolist(), k, ledger_fix=False)
        lf = run_online_seqtopk(rows.tolist(), k, ledger_fix=True)
        unfixed_violations += any(r["cumulative_calls"] > r["budget_mK"] for r in lu)
        fixed_violations += any(r["cumulative_calls"] > r["budget_mK"] for r in lf)
    stress = {"instances": stress_instances,
              "unfixed_budget_violations": unfixed_violations,
              "fixed_budget_violations": fixed_violations}
    metrics = {"c8_fixed": e2_counting_metrics(c8_fixed),
               "c8_unfixed": e2_counting_metrics(c8_unfixed),
               "note": "each counting basis is reported separately; branch/search "
                       "compute must be charged to generation-label or deployment cost "
                       "as appropriate"}
    return {"description": "E2 execution-layer budget ledger: per-step new calls, cumulative "
                           "calls, budget and remaining quota are recorded where experts "
                           "actually execute. C8 shows the published Online SeqTopK rule can "
                           "execute 3 calls against an mK=2 budget while a top-set-length "
                           "inference reports 2; the ledger fix restores the guarantee.",
            "c8_counterexample": c8, "randomized_stress": stress,
            "counting_metrics": metrics,
            "gate_pass": c8["fixed_within_budget"] and fixed_violations == 0
                         and c8["unfixed_exceeds_budget"] and c8["naive_inference_wrong"]}


# ---------------------------------------------------------------------------
# E3: order and architecture must be compared in separate tables (C9/C10).
# Scalar experts A(h)=2h, B(h)=h+1, h0=1; all parameters fixed.
# ---------------------------------------------------------------------------

def run_e3():
    a = lambda h: 2.0 * h
    b = lambda h: h + 1.0
    h0 = 1.0
    router_comparison = {
        "structure": "serial two Top-1 calls, identical experts/parameters/compute; "
                     "only the selection order changes",
        "paths": {
            "A->B": b(a(h0)),
            "B->A": a(b(h0)),
            "A->A": a(a(h0)),
            "B->B": b(b(h0)),
        },
    }
    router_comparison["same_set_order_dependent"] = (
        router_comparison["paths"]["A->B"] != router_comparison["paths"]["B->A"])
    alpha = 0.5
    architecture_comparison = {
        "note": "separate table: architectures differ in composition, depth and kernels; "
                "equal expert-call count does not make them the same function class",
        "serial_two_calls": {"output": b(a(h0)), "kernels": 2, "sequential_depth": 2,
                             "mix_ops": 0},
        "parallel_top2_mean": {"output": 0.5 * a(h0) + 0.5 * b(h0), "kernels": 2,
                               "sequential_depth": 1, "mix_ops": 1},
        "parallel_top2_residual": {"output": h0 + alpha * a(h0) + (1 - alpha) * b(h0),
                                   "kernels": 2, "sequential_depth": 1, "mix_ops": 1},
        "single_call": {"output": a(h0), "kernels": 1, "sequential_depth": 1,
                        "mix_ops": 0},
    }
    n_experts = 8
    pairs = list(itertools.combinations(range(n_experts), 2))
    pair_counting = {
        "n_experts": n_experts,
        "top2_pair_actions": len(pairs),
        "pair_labels": [f"({i},{j})" for i, j in pairs],
        "treated_as_independent_expert_actions_would_give": n_experts,
        "pair_labels_required": True,
        "top2_pair_h4_leaf_paths": len(pairs) ** 4,
    }
    gate = (len(pairs) == math.comb(8, 2) == 28
            and router_comparison["same_set_order_dependent"])
    return {"description": "E3 order/architecture separation: A->B, B->A, A->A, B->B and "
                           "parallel mixes compared with all parameters fixed; router "
                           "comparisons share the identical compute structure, architecture "
                           "comparisons are reported in a separate table; Top-2 candidates "
                           "over 8 experts are counted and labeled as 28 pairs, never as 8 "
                           "independent actions.",
            "h0": h0, "router_comparison": router_comparison,
            "architecture_comparison": architecture_comparison,
            "pair_counting": pair_counting, "gate_pass": gate}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

RUNNERS = {"c1": run_c1, "c2": run_c2, "d1": run_d1, "d2": run_d2,
           "e1": run_e1, "e2": run_e2, "e3": run_e3}


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="runs/e08_cde_cde.json",
                        help="path for the JSON result file")
    parser.add_argument("--only", default=None,
                        help="comma-separated subset of c1,c2,d1,d2,e1,e2,e3")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    selected = list(RUNNERS) if args.only is None else [
        s.strip() for s in args.only.split(",")]
    unknown = set(selected) - set(RUNNERS)
    if unknown:
        parser.error(f"unknown sub-experiments: {sorted(unknown)}")

    started = time.perf_counter()
    results = {}
    for name in selected:
        t0 = time.perf_counter()
        results[name] = RUNNERS[name]()
        results[name]["wall_time_s"] = time.perf_counter() - t0

    record = {
        "experiment": "E08 sub-experiments C/D/E (capacity/prices, switching "
                      "stability, causality/budget/architecture audit gates)",
        "plan": PLAN_PATH,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": args.seed,
        "synthetic_scope": SYNTHETIC_SCOPE,
        "total_wall_time_s": time.perf_counter() - started,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                   default=_json_default) + "\n",
                        encoding="utf-8")
    print(json.dumps({"output": str(out_path), "sections": selected,
                      "total_wall_time_s": record["total_wall_time_s"]},
                     ensure_ascii=False))
    return record


if __name__ == "__main__":
    main()
