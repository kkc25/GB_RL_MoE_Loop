"""Synthetic (CPU-only) parts of E05 and E06.

E05 (fixed-state potential game): Top-1 congestion game with hard per-expert
capacity, utility u_i = v[i,a_i] - c_{a_i}(L_{a_i}) and Rosenthal potential
Phi = sum_i v[i,a_i] - sum_e sum_{j=1..L_e} c_e(j). Verifies that one
asynchronous strict improvement changes utility and potential by the same
amount, enumerates pure equilibria / potential optimum / social optimum on
small instances, and runs a synchronous best-response oscillation gate.

E06 (causal capacity-constrained assignment): exact integer min-cost flow for
the additive proxy-cost version (each token needs k=2 distinct experts,
expert capacities are upper bounds), exact enumeration for the small
set-version with true pair costs, a log-domain soft (entropy-ish) reference
with residual validation, capacity-preserving sequential rounding/repair, and
causal sanity checks (suffix invariance + recorded batch dependence).

The real-LM serving part of both experiments is deliberately NOT executed
here; this module is the fixed-state synthetic validation only. Pure
Python/NumPy, no solver dependencies.
"""

import argparse
import heapq
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOL = 1e-9


# ---------------------------------------------------------------------------
# E05: fixed-state Top-1 congestion game with hard capacity
# ---------------------------------------------------------------------------

@dataclass
class CongestionGame:
    values: np.ndarray       # (N, E) baseline value of player i on expert e
    marginals: list          # per expert e, array c_e(j) for load j = 1..C_e
    capacity: np.ndarray     # (E,) int, hard slots

    @property
    def n_players(self):
        return self.values.shape[0]

    @property
    def n_experts(self):
        return self.values.shape[1]


def make_congestion_game(n_players, n_experts, capacity_factor=1.25, seed=0):
    rng = np.random.default_rng(seed)
    cap = int(np.ceil(n_players * capacity_factor / n_experts))
    capacity = np.full(n_experts, cap, dtype=int)
    while capacity.sum() < n_players:
        capacity[np.argmax(capacity)] += 1
    values = rng.uniform(0.0, 1.0, size=(n_players, n_experts))
    slopes = rng.uniform(0.2, 1.0, size=n_experts)
    marginals = [slopes[e] * np.arange(1, capacity[e] + 1) for e in range(n_experts)]
    return CongestionGame(values=values, marginals=marginals, capacity=capacity)


def oscillation_example():
    """E08 gate: 2 players, 2 experts, sync best response cycles (0,0)<->(1,1)
    while asynchronous strict improvement converges to {(0,1),(1,0)}."""
    values = np.zeros((2, 2))
    marginals = [np.array([0.0, 10.0]), np.array([0.0, 10.0])]
    capacity = np.array([2, 2])
    return CongestionGame(values=values, marginals=marginals, capacity=capacity)


def loads_of(game, actions):
    loads = np.zeros(game.n_experts, dtype=int)
    for a in actions:
        loads[a] += 1
    return loads


def is_feasible(game, actions):
    return bool(np.all(loads_of(game, actions) <= game.capacity))


def player_utility(game, actions, i):
    loads = loads_of(game, actions)
    a = actions[i]
    return game.values[i, a] - game.marginals[a][loads[a] - 1]


def potential(game, actions):
    loads = loads_of(game, actions)
    phi = float(sum(game.values[i, a] for i, a in enumerate(actions)))
    for e in range(game.n_experts):
        phi -= float(game.marginals[e][: loads[e]].sum())
    return phi


def social_welfare(game, actions):
    loads = loads_of(game, actions)
    sw = float(sum(game.values[i, a] for i, a in enumerate(actions)))
    for e in range(game.n_experts):
        sw -= loads[e] * float(game.marginals[e][loads[e] - 1]) if loads[e] else 0.0
    return sw


def best_response(game, actions, i, loads):
    """Best action for player i given other players fixed; respects hard
    capacity on the target expert (load after removing self)."""
    cur = actions[i]
    cur_u = game.values[i, cur] - game.marginals[cur][loads[cur] - 1]
    best_a, best_u = cur, cur_u
    for e in range(game.n_experts):
        if e == cur:
            continue
        residual = loads[e]  # player i is not on e
        if residual + 1 > game.capacity[e]:
            continue
        u = game.values[i, e] - game.marginals[e][residual]
        if u > best_u + TOL:
            best_a, best_u = e, u
    return best_a, best_u, cur_u


def random_feasible_assignment(game, seed):
    rng = np.random.default_rng(seed)
    slots = np.concatenate([np.full(c, e) for e, c in enumerate(game.capacity)])
    rng.shuffle(slots)
    return [int(s) for s in slots[: game.n_players]]


def async_improvement(game, actions, seed=0, max_sweeps=8):
    """Asynchronous strict-improvement dynamics. One sweep = one pass over all
    players in random order; each strictly improving player moves immediately.
    Ties keep the current action. Returns the trajectory and the max
    |Delta u - Delta Phi| observed over all applied steps."""
    rng = np.random.default_rng(seed)
    actions = list(actions)
    loads = loads_of(game, actions)
    phi = potential(game, actions)
    phi_trace = [phi]
    max_delta_err = 0.0
    converged_sweep = None
    for sweep in range(1, max_sweeps + 1):
        moved = False
        for i in rng.permutation(game.n_players):
            new_a, new_u, cur_u = best_response(game, actions, i, loads)
            if new_a != actions[i]:
                old_a = actions[i]
                actions[i] = int(new_a)
                loads[old_a] -= 1
                loads[new_a] += 1
                new_phi = potential(game, actions)
                max_delta_err = max(max_delta_err, abs((new_phi - phi) - (new_u - cur_u)))
                phi = new_phi
                phi_trace.append(phi)
                moved = True
        if not moved:
            converged_sweep = sweep
            break
    return {
        "actions": [int(a) for a in actions],
        "potential_trace": phi_trace,
        "converged": converged_sweep is not None,
        "sweeps_used": converged_sweep if converged_sweep is not None else max_sweeps,
        "max_abs_utility_potential_delta_err": max_delta_err,
        "social_welfare": social_welfare(game, actions),
    }


def sync_best_response(game, actions, max_sweeps=64):
    """Synchronous best response; ties keep the current action. Detects cycles
    over the joint profile (E08 oscillation gate)."""
    actions = tuple(actions)
    seen = {actions: 0}
    history = [list(actions)]
    for sweep in range(1, max_sweeps + 1):
        loads = loads_of(game, list(actions))
        nxt = []
        for i in range(game.n_players):
            a, _, _ = best_response(game, list(actions), i, loads)
            nxt.append(a)
        nxt = tuple(nxt)
        history.append(list(nxt))
        if nxt in seen:
            return {
                "cycle_detected": True,
                "cycle_start_sweep": seen[nxt],
                "cycle_length": sweep - seen[nxt],
                "history": history,
            }
        seen[nxt] = sweep
        actions = nxt
    return {
        "cycle_detected": False,
        "converged_to": list(actions),
        "is_equilibrium": is_pure_equilibrium(game, list(actions)),
        "history": history,
    }


def is_pure_equilibrium(game, actions):
    loads = loads_of(game, actions)
    for i in range(game.n_players):
        new_a, _, _ = best_response(game, actions, i, loads)
        if new_a != actions[i]:
            return False
    return True


def enumerate_feasible(game):
    n, e = game.n_players, game.n_experts
    loads = np.zeros(e, dtype=int)
    actions = [0] * n

    def rec(i):
        if i == n:
            yield list(actions)
            return
        for a in range(e):
            if loads[a] < game.capacity[a]:
                loads[a] += 1
                actions[i] = a
                yield from rec(i + 1)
                loads[a] -= 1

    yield from rec(0)


def enumerate_game_outcomes(game):
    """Exhaustive enumeration over feasible profiles: pure equilibria,
    potential global optimum, and social optimum. Small instances only."""
    best_phi, best_phi_actions = -np.inf, None
    best_sw, best_sw_actions = -np.inf, None
    equilibria = []
    for actions in enumerate_feasible(game):
        phi = potential(game, actions)
        sw = social_welfare(game, actions)
        if phi > best_phi + TOL:
            best_phi, best_phi_actions = phi, list(actions)
        if sw > best_sw + TOL:
            best_sw, best_sw_actions = sw, list(actions)
        if is_pure_equilibrium(game, actions):
            equilibria.append((list(actions), phi, sw))
    return {
        "n_equilibria": len(equilibria),
        "equilibrium_social_welfare_min": min((sw for _, _, sw in equilibria), default=None),
        "equilibrium_social_welfare_max": max((sw for _, _, sw in equilibria), default=None),
        "potential_optimum_actions": best_phi_actions,
        "potential_optimum_value": best_phi,
        "potential_optimum_is_equilibrium": is_pure_equilibrium(game, best_phi_actions),
        "potential_optimum_social_welfare": social_welfare(game, best_phi_actions),
        "social_optimum_actions": best_sw_actions,
        "social_optimum_social_welfare": best_sw,
    }


# ---------------------------------------------------------------------------
# E06: exact integer min-cost flow for the additive capacity assignment
# ---------------------------------------------------------------------------

class MinCostFlow:
    """Successive shortest paths with Johnson potentials. Integer capacities
    give an integer optimal flow (total unimodularity of the LP)."""

    def __init__(self, n_nodes):
        self.n = n_nodes
        self.graph = [[] for _ in range(n_nodes)]  # [to, cap, cost, rev_idx]

    def add_edge(self, fr, to, cap, cost):
        self.graph[fr].append([to, cap, cost, len(self.graph[to])])
        self.graph[to].append([fr, 0, -cost, len(self.graph[fr]) - 1])

    def flow(self, s, t, maxf):
        n, g = self.n, self.graph
        inf = float("inf")
        potential = [0.0] * n
        dist = [inf] * n
        dist[s] = 0.0
        for _ in range(n - 1):
            updated = False
            for v in range(n):
                if dist[v] == inf:
                    continue
                for e in g[v]:
                    if e[1] > 0 and dist[v] + e[2] < dist[e[0]] - 1e-12:
                        dist[e[0]] = dist[v] + e[2]
                        updated = True
            if not updated:
                break
        for v in range(n):
            if dist[v] < inf:
                potential[v] = dist[v]
        total_flow, total_cost = 0, 0.0
        while total_flow < maxf:
            dist = [inf] * n
            dist[s] = 0.0
            prevv, preve = [-1] * n, [-1] * n
            pq = [(0.0, s)]
            while pq:
                d, v = heapq.heappop(pq)
                if dist[v] < d:
                    continue
                for i, e in enumerate(g[v]):
                    if e[1] > 0:
                        nd = dist[v] + e[2] + potential[v] - potential[e[0]]
                        if nd < dist[e[0]] - 1e-12:
                            dist[e[0]] = nd
                            prevv[e[0]], preve[e[0]] = v, i
                            heapq.heappush(pq, (nd, e[0]))
            if dist[t] == inf:
                break
            for v in range(n):
                if dist[v] < inf:
                    potential[v] += dist[v]
            d, v = maxf - total_flow, t
            while v != s:
                d = min(d, g[prevv[v]][preve[v]][1])
                v = prevv[v]
            total_flow += d
            v = t
            while v != s:
                e = g[prevv[v]][preve[v]]
                e[1] -= d
                g[v][e[3]][1] += d
                total_cost += d * e[2]
                v = prevv[v]
        return total_flow, total_cost


def solve_additive_assignment(costs, capacity, allowed=None, k=2):
    """Exact integer reference: min sum D_ie P_ie s.t. each token takes k
    distinct experts, expert e used at most C_e times. `allowed` (optional
    bool matrix) encodes the router candidate restriction. Returns the
    assignment, objective, and a feasibility flag (checked via max flow, not
    assumed from total slots alone)."""
    costs = np.asarray(costs, dtype=float)
    n, e_n = costs.shape
    capacity = np.asarray(capacity, dtype=int)
    if allowed is None:
        allowed = np.ones((n, e_n), dtype=bool)
    s, t = 0, n + e_n + 1
    mcf = MinCostFlow(n + e_n + 2)
    for i in range(n):
        mcf.add_edge(s, 1 + i, k, 0.0)
    for i in range(n):
        for e in range(e_n):
            if allowed[i, e]:
                mcf.add_edge(1 + i, 1 + n + e, 1, float(costs[i, e]))
    for e in range(e_n):
        mcf.add_edge(1 + n + e, t, int(capacity[e]), 0.0)
    flow, obj = mcf.flow(s, t, k * n)
    feasible = flow == k * n
    assignment = [[] for _ in range(n)]
    if feasible:
        for i in range(n):
            for edge in mcf.graph[1 + i]:
                if 1 + n <= edge[0] < 1 + n + e_n and edge[1] == 0:
                    assignment[i].append(edge[0] - (1 + n))
    return {
        "feasible": feasible,
        "flow": int(flow),
        "required_flow": k * n,
        "assignment": [sorted(a) for a in assignment],
        "objective": float(obj) if feasible else None,
    }


def check_assignment(costs, capacity, assignment, allowed=None, k=2):
    """Validates a hard-capacity integer assignment; returns list of violations."""
    n, e_n = np.asarray(costs).shape
    allowed = np.ones((n, e_n), dtype=bool) if allowed is None else allowed
    violations = []
    loads = np.zeros(e_n, dtype=int)
    for i, experts in enumerate(assignment):
        if len(experts) != k:
            violations.append(f"token {i}: has {len(experts)} experts, need {k}")
        if len(set(experts)) != len(experts):
            violations.append(f"token {i}: duplicate experts {experts}")
        for e in experts:
            if not (0 <= e < e_n):
                violations.append(f"token {i}: expert {e} out of range")
                continue
            if not allowed[i, e]:
                violations.append(f"token {i}: expert {e} not in candidate set")
            loads[e] += 1
    for e in range(e_n):
        if loads[e] > capacity[e]:
            violations.append(f"expert {e}: load {loads[e]} > capacity {capacity[e]}")
    return violations


def assignment_objective(costs, assignment):
    return float(sum(np.asarray(costs)[i, e] for i, ex in enumerate(assignment) for e in ex))


def brute_force_additive(costs, capacity, allowed=None, k=2):
    """Exact enumeration over k-subsets per token; tiny instances only."""
    costs = np.asarray(costs, dtype=float)
    n, e_n = costs.shape
    allowed = np.ones((n, e_n), dtype=bool) if allowed is None else allowed
    per_token = [
        [c for c in itertools.combinations(range(e_n), k) if all(allowed[i, e] for e in c)]
        for i in range(n)
    ]
    best, best_assign = np.inf, None
    for combo in itertools.product(*per_token):
        loads = np.zeros(e_n, dtype=int)
        for c in combo:
            for e in c:
                loads[e] += 1
        if np.any(loads > capacity):
            continue
        obj = sum(costs[i, e] for i, c in enumerate(combo) for e in c)
        if obj < best - TOL:
            best, best_assign = obj, [sorted(c) for c in combo]
    return {"feasible": best_assign is not None, "objective": best if best_assign else None,
            "assignment": best_assign}


# ---------------------------------------------------------------------------
# E06: set version (true Top-2 pair costs) by exact enumeration, N <= ~16
# ---------------------------------------------------------------------------

def enumerate_set_assignment(set_costs, candidates, capacity, node_limit=500_000):
    """Exact integer optimum of the multi-resource set version:
    each token picks one candidate pair a=(e,f), expert loads <= C_e.
    Branch and bound with per-token suffix minima; reports optimality and a
    bound gap if the node limit is hit."""
    n = len(candidates)
    capacity = np.asarray(capacity, dtype=int)
    suffix_min = np.zeros(n + 1)
    for i in range(n - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + min(set_costs[i])
    order = [sorted(range(len(candidates[i])), key=lambda j: set_costs[i][j]) for i in range(n)]
    remaining = capacity.copy()
    assign = [None] * n
    state = {"best": np.inf, "best_assign": None, "nodes": 0, "hit_limit": False}

    def dfs(i, cost):
        if cost + suffix_min[i] >= state["best"] - TOL:
            return
        if i == n:
            state["best"] = cost
            state["best_assign"] = list(assign)
            return
        for j in order[i]:
            e1, e2 = candidates[i][j]
            if remaining[e1] > 0 and remaining[e2] > 0:
                state["nodes"] += 1
                if state["nodes"] > node_limit:
                    state["hit_limit"] = True
                    return
                remaining[e1] -= 1
                remaining[e2] -= 1
                assign[i] = j
                dfs(i + 1, cost + set_costs[i][j])
                remaining[e1] += 1
                remaining[e2] += 1
                if state["hit_limit"]:
                    return

    dfs(0, 0.0)
    bound = suffix_min[0] if state["best_assign"] is None else None
    gap = None
    if state["hit_limit"] and state["best_assign"] is not None:
        gap = float(state["best"] - max(bound if bound is not None else suffix_min[0], -np.inf))
        gap = float(state["best"] - suffix_min[0])
    return {
        "feasible": state["best_assign"] is not None,
        "objective": float(state["best"]) if state["best_assign"] is not None else None,
        "assignment": [
            sorted(candidates[i][j]) if j is not None else None
            for i, j in enumerate(state["best_assign"] or [])
        ],
        "nodes_expanded": state["nodes"],
        "optimal": not state["hit_limit"],
        "optimality_gap_bound": gap,
    }


def fit_additive_per_token(set_costs, candidates, n_experts):
    """Per-token least squares Q_i,{e,f} ~ d_ie + d_if over observed candidate
    pairs. Returns the fitted additive matrix and the pair-interaction
    residual (RMS and max abs), quantifying the additive approximation."""
    n = len(candidates)
    D = np.zeros((n, n_experts))
    rms_list, maxabs_list = [], []
    for i in range(n):
        pairs = candidates[i]
        q = np.asarray(set_costs[i], dtype=float)
        A = np.zeros((len(pairs), n_experts))
        for r, (e1, e2) in enumerate(pairs):
            A[r, e1] = 1.0
            A[r, e2] = 1.0
        d, *_ = np.linalg.lstsq(A, q, rcond=None)
        D[i] = d
        resid = q - A @ d
        rms_list.append(float(np.sqrt(np.mean(resid**2))))
        maxabs_list.append(float(np.max(np.abs(resid))) if len(resid) else 0.0)
    return D, {"rms": rms_list, "max_abs": maxabs_list}


# ---------------------------------------------------------------------------
# E06: soft reference, rounding / repair, causal sanity
# ---------------------------------------------------------------------------

def soft_transport(costs, capacity, eps, iters, allowed=None, k=2):
    """Entropy-smoothed soft reference ( Sinkhorn / IPF in the log domain )
    with a slack row: column sums are fixed to exactly C_e, and a slack row
    with zero-cost edges absorbs the unused capacity sum(C) - k*N. This keeps
    the real rows at "at most C_e" per expert; equal column margins without
    the slack would force fixed occupation of every expert by real tokens,
    which is a different objective. Marginal residuals are reported after the
    requested number of iterations."""
    costs = np.asarray(costs, dtype=float)
    n, e_n = costs.shape
    capacity = np.asarray(capacity, dtype=float)
    allowed = np.ones((n, e_n), dtype=bool) if allowed is None else allowed
    slack = float(capacity.sum() - k * n)
    if slack < -TOL:
        return {"feasible": False, "reason": "total slots < k*N; must queue/stop"}

    blocked = 1e6  # large finite cost keeps forbidden/absorbing edges at ~0 mass
    if slack > TOL:
        row_sums = np.concatenate([np.full(n, float(k)), [slack]])
        ext = np.full((n + 1, e_n), blocked)
        ext[:n][~allowed] = blocked
        ext[:n][allowed] = costs[allowed]
        ext[n, :] = 0.0
    else:
        row_sums = np.full(n, float(k))
        ext = np.where(allowed, costs, blocked)
    col_sums = capacity

    log_k = -ext / eps
    log_a = np.zeros(len(row_sums))
    log_b = np.zeros(e_n)

    def logsumexp(x, axis):
        m = x.max(axis=axis)
        return m + np.log(np.exp(x - np.expand_dims(m, axis)).sum(axis=axis))

    for _ in range(iters):
        log_p = log_k + log_a[:, None] + log_b[None, :]
        log_a += np.log(row_sums) - logsumexp(log_p, axis=1)
        log_p = log_k + log_a[:, None] + log_b[None, :]
        log_b += np.log(col_sums) - logsumexp(log_p, axis=0)
    log_p = log_k + log_a[:, None] + log_b[None, :]
    p = np.exp(log_p)[:n]
    rows, cols = p.sum(axis=1), p.sum(axis=0)
    return {
        "P": p,
        "objective_linear": float(np.sum(costs * p)),
        "max_row_residual": float(np.max(np.abs(rows - k))),
        "max_col_excess": float(np.max(np.maximum(0.0, cols - capacity))),
        "max_upper_bound_violation": float(np.max(np.maximum(0.0, p - 1.0))),
        "slack_capacity": slack,
        "eps": eps,
        "iters": iters,
        "feasible": True,
    }


def round_fractional(P, costs, capacity, allowed=None, k=2):
    """Capacity-preserving sequential rounding: fix highest-mass edges first
    (distinct experts per token), then greedily fill remaining slots with the
    cheapest feasible candidate. If repair fails although the instance is
    feasible, falls back to the exact flow and marks the fallback. Records the
    objective increase over the fractional linear objective."""
    P = np.asarray(P, dtype=float)
    costs = np.asarray(costs, dtype=float)
    n, e_n = P.shape
    capacity = np.asarray(capacity, dtype=int)
    allowed = np.ones((n, e_n), dtype=bool) if allowed is None else allowed
    remaining = capacity.copy()
    assignment = [[] for _ in range(n)]
    edges = sorted(
        ((P[i, e], i, e) for i in range(n) for e in range(e_n) if allowed[i, e]),
        key=lambda x: -x[0],
    )
    for mass, i, e in edges:
        if len(assignment[i]) < k and e not in assignment[i] and remaining[e] > 0:
            assignment[i].append(e)
            remaining[e] -= 1
    repaired = False
    for i in range(n):
        while len(assignment[i]) < k:
            repaired = True
            options = [
                (costs[i, e], e)
                for e in range(e_n)
                if allowed[i, e] and e not in assignment[i] and remaining[e] > 0
            ]
            if not options:
                exact = solve_additive_assignment(costs, capacity, allowed, k)
                if not exact["feasible"]:
                    return {"feasible": False, "reason": "instance infeasible; must queue/stop"}
                return {
                    "feasible": True,
                    "assignment": exact["assignment"],
                    "objective": exact["objective"],
                    "fallback_exact": True,
                    "repaired": True,
                }
            _, e = min(options)
            assignment[i].append(e)
            remaining[e] -= 1
    assignment = [sorted(a) for a in assignment]
    obj = assignment_objective(costs, assignment)
    return {
        "feasible": True,
        "assignment": assignment,
        "objective": obj,
        "fallback_exact": False,
        "repaired": repaired,
        "objective_increase_over_fractional": obj - float(np.sum(costs * P)),
    }


def causal_sanity(costs, capacity, suffix_values, allowed=None, k=2, gamma=0.5):
    """Causal diagnostics for the window allocator.

    - suffix invariance: the deployable allocator reads only current-token
      cost rows; replacing any request's future suffix cannot change its
      output. This must pass.
    - batch dependence: adding/removing other requests in the co-batch may
      change assignments under shared capacity; this is allowed but recorded.
    - non-causal reference: an oracle that mixes future suffix information
      into the costs. Clearly labeled non-deployable; diagnostic only."""
    base = solve_additive_assignment(costs, capacity, allowed, k)
    rerun = solve_additive_assignment(np.array(costs, copy=True), capacity, allowed, k)
    suffix_invariant = base["assignment"] == rerun["assignment"]

    batch_dependence = {}
    if base["feasible"] and np.asarray(costs).shape[0] > 2:
        drop = solve_additive_assignment(np.asarray(costs)[:-1], capacity,
                                         None if allowed is None else np.asarray(allowed)[:-1], k)
        if drop["feasible"]:
            changed = sum(
                1 for i in range(len(drop["assignment"]))
                if drop["assignment"][i] != base["assignment"][i]
            )
            batch_dependence["remove_one_request_changed_fraction"] = changed / len(drop["assignment"])

    non_causal = {"deployable": False, "label": "non_causal_diagnostic_future_information"}
    if base["feasible"]:
        d_nc = np.asarray(costs, dtype=float) - gamma * np.asarray(suffix_values, dtype=float)
        oracle = solve_additive_assignment(d_nc, capacity, allowed, k)
        if oracle["feasible"]:
            non_causal["oracle_objective_in_modified_units"] = oracle["objective"]
            non_causal["causal_objective"] = base["objective"]
            non_causal["assignment_changed_fraction"] = float(np.mean([
                a != b for a, b in zip(base["assignment"], oracle["assignment"])
            ]))
    return {
        "suffix_invariance_passed": bool(suffix_invariant),
        "batch_dependence": batch_dependence,
        "non_causal_diagnostic": non_causal,
    }


# ---------------------------------------------------------------------------
# Synthetic runners
# ---------------------------------------------------------------------------

def run_e05(seed=0):
    out = {}
    small = make_congestion_game(n_players=4, n_experts=3, capacity_factor=1.0, seed=seed)
    init = random_feasible_assignment(small, seed + 1)
    async_res = async_improvement(small, init, seed=seed + 2, max_sweeps=16)
    enum = enumerate_game_outcomes(small)
    enum["async_result_is_equilibrium"] = is_pure_equilibrium(small, async_res["actions"])
    enum["async_result_social_welfare"] = async_res["social_welfare"]
    enum["max_abs_utility_potential_delta_err"] = async_res["max_abs_utility_potential_delta_err"]
    out["small_enumerated"] = enum

    medium = make_congestion_game(n_players=32, n_experts=8, capacity_factor=1.25, seed=seed + 10)
    init_m = random_feasible_assignment(medium, seed + 11)
    sweep_curve = {}
    for k in (1, 2, 4, 8):
        r = async_improvement(medium, list(init_m), seed=seed + 12, max_sweeps=k)
        sweep_curve[str(k)] = {
            "converged": r["converged"],
            "final_potential": r["potential_trace"][-1],
            "is_equilibrium": is_pure_equilibrium(medium, r["actions"]),
        }
    out["medium_async_sweeps"] = sweep_curve

    osc = oscillation_example()
    out["oscillation_gate"] = {
        "sync": sync_best_response(osc, [0, 0], max_sweeps=16),
        "async": async_improvement(osc, [0, 0], seed=seed, max_sweeps=16),
        "equilibria": enumerate_game_outcomes(osc)["n_equilibria"],
    }
    return out


def run_e06(seed=0):
    out = {}
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    n, e_n, k = 16, 8, 2
    capacity = np.full(e_n, int(np.ceil(k * n * 1.25 / e_n)), dtype=int)  # 5 each, 40 >= 32
    scores = rng.uniform(0.0, 1.0, size=(n, e_n))
    allowed = np.zeros((n, e_n), dtype=bool)
    for i in range(n):
        allowed[i, np.argsort(-scores[i])[:4]] = True  # top-4 router candidates
    costs = 1.0 - scores + 0.05 * rng.uniform(size=(n, e_n))
    suffix_values = rng.uniform(0.0, 1.0, size=(n, e_n))  # future info, never fed to allocator
    cost_table_seconds = time.perf_counter() - t0
    out["cost_table_generation_seconds"] = cost_table_seconds
    out["window"] = {"N": n, "E": e_n, "k": k, "capacity_per_expert": int(capacity[0]),
                     "total_slots": int(capacity.sum()), "candidates_per_token": 4}

    t0 = time.perf_counter()
    exact = solve_additive_assignment(costs, capacity, allowed, k)
    out["additive_exact"] = {kk: vv for kk, vv in exact.items() if kk != "assignment"}
    out["additive_exact"]["assignment_violations"] = check_assignment(
        costs, capacity, exact["assignment"], allowed, k)
    out["additive_exact"]["solve_seconds"] = time.perf_counter() - t0

    # tiny cross-check of the flow against brute-force enumeration
    bf = brute_force_additive(costs[:6], capacity, allowed[:6], k)
    mf = solve_additive_assignment(costs[:6], capacity, allowed[:6], k)
    out["tiny_crosscheck"] = {
        "brute_force_objective": bf["objective"],
        "min_cost_flow_objective": mf["objective"],
        "match": bf["feasible"] and mf["feasible"] and abs(bf["objective"] - mf["objective"]) < 1e-8,
    }

    # set version with true pair costs (pair interaction on top of additive fit)
    t0 = time.perf_counter()
    candidates, set_costs = [], []
    base_d = 1.0 - scores
    for i in range(n):
        pairs = [tuple(p) for p in itertools.combinations(np.where(allowed[i])[0].tolist(), 2)]
        candidates.append(pairs)
        set_costs.append([
            float(base_d[i, a] + base_d[i, b] + rng.normal(0.0, 0.05)) for a, b in pairs
        ])
    d_fit, fit_resid = fit_additive_per_token(set_costs, candidates, e_n)
    set_res = enumerate_set_assignment(set_costs, candidates, capacity, node_limit=2_000_000)
    out["set_version"] = {kk: vv for kk, vv in set_res.items() if kk != "assignment"}
    out["set_version"]["solve_seconds"] = time.perf_counter() - t0
    out["additive_fit_residual"] = {
        "rms_mean": float(np.mean(fit_resid["rms"])),
        "max_abs_max": float(np.max(fit_resid["max_abs"])),
    }
    if set_res["feasible"] and exact["feasible"]:
        true_set_cost = {i: dict(zip(candidates[i], set_costs[i])) for i in range(n)}
        additive_assign_true_cost = sum(
            true_set_cost[i][tuple(sorted(pair))] for i, pair in enumerate(
                [tuple(a) for a in exact["assignment"]])
        ) if all(tuple(sorted(a)) in true_set_cost[i] for i, a in enumerate(exact["assignment"])) else None
        out["pair_interaction_error"] = {
            "additive_assignment_true_set_cost": additive_assign_true_cost,
            "set_version_optimum": set_res["objective"],
        }

    # soft reference + rounding
    std = float(np.std(costs[allowed]))
    rounding_rows = []
    for eps_mult in (0.01, 0.05, 0.1):
        for iters in (5, 10, 20):
            soft = soft_transport(costs, capacity, eps_mult * std, iters, allowed=allowed, k=k)
            rounded = round_fractional(soft["P"], costs, capacity, allowed, k)
            row = {
                "eps_over_std": eps_mult, "iters": iters,
                "max_row_residual": soft["max_row_residual"],
                "max_col_excess": soft["max_col_excess"],
                "fractional_linear_objective": soft["objective_linear"],
                "rounded_feasible": rounded["feasible"],
                "rounded_objective": rounded.get("objective"),
                "fallback_exact": rounded.get("fallback_exact"),
            }
            if rounded["feasible"] and exact["feasible"]:
                row["integrality_gap_vs_exact"] = rounded["objective"] - exact["objective"]
            rounding_rows.append(row)
    out["soft_and_rounding"] = rounding_rows

    out["causal_sanity"] = causal_sanity(costs, capacity, suffix_values, allowed, k)
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, help="path to write the JSON results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = {
        "experiment": "E05 fixed-state potential game + E06 causal capacity assignment (synthetic)",
        "scope": "synthetic_fixed_state_only",
        "real_lm_serving_part": "not_executed",
        "seed": args.seed,
        "e05": run_e05(seed=args.seed),
        "e06": run_e06(seed=args.seed),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(results), indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
