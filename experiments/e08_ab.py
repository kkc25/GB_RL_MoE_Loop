"""E08 A/B sub-experiments: sufficient state, delayed reward, synthetic mechanisms.

Implements the A/B part of
plans/loop_moe_research_plan_2026-10-25/experiments/E08_stability_and_synthetic_mechanisms.md

  A1  Sufficient-state negative control: exhaustive finite-horizon DP vs an
      explicit history tree, over 20 random transition tables x 3 cost configs.
  A2  Information-erasure positive control (counterexample C1): h-only /
      h+recent-expert / h+ordered-history controllers under label-noise sweep.
  A3  Finite function class: 8-bit / 12-bit parity, linear vs MLP h-only vs
      h+history (constructed cumulative-parity features, input-derived only).
  B1  Delayed-reward planning: exact 2-token env (C5) + random 8-token/4-step
      envs with planted low-then-high chains; uniform budget, one-step greedy,
      two-step rollout, pruned two-step, exact budget DP, supervised Q^{pi0}.
  B2  Unlearnable counterfactual boundary (C2): label-visible oracle vs
      label-blind per-expert loss regressors across a signal-strength sweep.
  B3  Ordered composition: 4 non-commuting affine experts mod p, router-only
      training with exact experts, then small-MLP approximate experts.

All conclusions are restricted to these synthetic environments; they verify
mechanisms and code correctness, not natural-language behaviour.

CLI:
    python experiments/e08_ab.py --output results.json [--subexp A1,A2,...]
                                 [--seed 0] [--quick]
"""

import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn

LOG2 = math.log(2.0)
ORACLE_NLL = -math.log(0.99)  # 0.01005033585350145 (verified_toy_calculations.json)


# --------------------------------------------------------------------------- #
# A1: sufficient-state negative control
# --------------------------------------------------------------------------- #

def _max_group_spread(keys, values):
    """Max (group max - group min) of values grouped by integer keys."""
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    v = values[order]
    starts = np.flatnonzero(np.concatenate(([True], k[1:] != k[:-1])))
    gmax = np.maximum.reduceat(v, starts)
    gmin = np.minimum.reduceat(v, starts)
    return float(np.max(gmax - gmin)) if len(starts) else 0.0


def _a1_sample_table(rng, E, S, support=3):
    """Random transition tensor P[a, i, j] (row-stochastic, sparse support)
    and terminal loss L[i]."""
    P = np.zeros((E, S, S), dtype=np.float64)
    for a in range(E):
        for i in range(S):
            nxt = rng.choice(S, size=support, replace=False)
            probs = rng.dirichlet(np.ones(support))
            P[a, i, nxt] = probs
    L = rng.uniform(0.0, 1.0, size=S)
    return P, L


def _a1_full_state_dp(P, L, costs, B, H, no_repeat):
    """Exact DP over the full sufficient state (node, round, remaining budget
    [, used-expert mask]).  Stopping is always allowed at value -L[node]."""
    E, S = P.shape[0], P.shape[1]
    costs = np.asarray(costs, dtype=np.int64)
    if no_repeat:
        M = 1 << E
        V = np.zeros((H + 1, S, B + 1, M), dtype=np.float64)
        V[H, :, :, :] = -L[:, None, None]
        for t in range(H - 1, -1, -1):
            for b in range(B + 1):
                for m in range(M):
                    best = -L  # stop anywhere
                    for a in range(E):
                        if costs[a] > b or (m >> a) & 1:
                            continue
                        q = P[a] @ V[t + 1, :, b - costs[a], m | (1 << a)]
                        best = np.maximum(best, q)
                    V[t, :, b, m] = best
        return V
    V = np.zeros((H + 1, S, B + 1), dtype=np.float64)
    V[H, :, :] = -L[:, None]
    for t in range(H - 1, -1, -1):
        for b in range(B + 1):
            best = -L.copy()
            for a in range(E):
                if costs[a] > b:
                    continue
                q = P[a] @ V[t + 1, :, b - costs[a]]
                best = np.maximum(best, q)
            V[t, :, b] = best
    return V


def _a1_history_tree(P, L, costs, B, H, no_repeat, group_with_mask, i0=0):
    """Explicit history tree.  A level-t history is encoded as a base-(E*S)
    integer whose digits are (action, next_node) pairs, so the current node is
    idx % S.  Optimal history-conditioned continuation values are computed by
    backward induction over the tree WITHOUT assuming they depend only on the
    current state; we then group histories by state key and measure the spread.
    Returns (root_value, max_spread, n_unique_states)."""
    E, S = P.shape[0], P.shape[1]
    ES = E * S
    costs = np.asarray(costs, dtype=np.int64)
    M = 1 << E

    # forward: remaining budget and used masks per level
    rems = [np.array([B], dtype=np.int64)]
    masks = [np.array([0], dtype=np.int64)]
    for t in range(H):
        n_t = ES ** t
        step_cost = np.tile(np.repeat(costs, S), n_t)
        rems.append(np.repeat(rems[t], ES) - step_cost)
        if no_repeat:
            step_bit = np.tile(np.repeat(1 << np.arange(E), S), n_t)
            masks.append(np.repeat(masks[t], ES) | step_bit)
        else:
            masks.append(np.zeros(ES ** (t + 1), dtype=np.int64))

    def state_keys(t):
        n_t = ES ** t
        cur = np.arange(n_t, dtype=np.int64) % S
        key = (t * (B + 1) + rems[t]) * S + cur
        if group_with_mask:
            key = key * M + masks[t]
        return key

    # terminal level
    n_H = ES ** H
    V = -L[np.arange(n_H, dtype=np.int64) % S]
    max_spread = _max_group_spread(state_keys(H), V)
    n_states = len(np.unique(state_keys(H)))

    for t in range(H - 1, -1, -1):
        n_t = ES ** t
        idx = np.arange(n_t, dtype=np.int64)
        cur = idx % S
        V_prev = V
        V = -L[cur].copy()  # stop option
        base = idx * ES
        for a in range(E):
            feas = rems[t] >= costs[a]
            if no_repeat:
                feas &= ((masks[t] >> a) & 1) == 0
            child = base[:, None] + a * S + np.arange(S, dtype=np.int64)[None, :]
            cont = (P[a][cur] * V_prev[child]).sum(axis=1)
            cont = np.where(feas, cont, -np.inf)
            V = np.maximum(V, cont)
        sp = _max_group_spread(state_keys(t), V)
        max_spread = max(max_spread, sp)
        n_states += len(np.unique(state_keys(t)))
    return float(V[0]), max_spread, n_states


def run_a1(seed=0, n_tables=20, E=4, H=4, S=6, B=5,
           cost_configs=((0, 0, 0, 0), (2, 2, 2, 2), (5, 5, 5, 5)), tol=1e-9):
    """Negative control: with a sufficient state (node, round, budget [, mask])
    a history-conditioned policy cannot beat Markov DP, and continuation values
    of one state reached via different histories must coincide.  Costs are in
    cents (0 / 0.02 / 0.05 with budget 0.05)."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    records = []
    max_spread_sufficient = 0.0
    max_root_diff = 0.0
    insufficiency_spreads = []
    for table_id in range(n_tables):
        P, L = _a1_sample_table(rng, E, S)
        for costs in cost_configs:
            for no_repeat in (False, True):
                Vdp = _a1_full_state_dp(P, L, costs, B, H, no_repeat)
                root_dp = float(Vdp[0, 0, B, 0] if no_repeat else Vdp[0, 0, B])
                root_tree, spread, n_states = _a1_history_tree(
                    P, L, costs, B, H, no_repeat, group_with_mask=no_repeat)
                max_spread_sufficient = max(max_spread_sufficient, spread)
                max_root_diff = max(max_root_diff, abs(root_tree - root_dp))
                rec = {"table": table_id, "costs_cents": list(costs),
                       "no_repeat_mask": no_repeat,
                       "root_value_dp": root_dp, "root_value_history_tree": root_tree,
                       "state_value_spread": spread, "n_states_visited": n_states}
                if no_repeat:
                    # Diagnostic: drop the used-expert mask from the state key.
                    # The checker must be able to detect the insufficiency.
                    _, spread_omit, _ = _a1_history_tree(
                        P, L, costs, B, H, no_repeat, group_with_mask=False)
                    rec["spread_when_mask_omitted"] = spread_omit
                    insufficiency_spreads.append(spread_omit)
                records.append(rec)
    max_spread_insufficient = max(insufficiency_spreads)
    gates = {
        "sufficient_state_value_spread<=tol": max_spread_sufficient <= tol,
        "history_tree_root==dp_root<=tol": max_root_diff <= tol,
        "omitting_mask_detected>1e-6": max_spread_insufficient > 1e-6,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": {
            "n_tables": n_tables, "cost_configs_cents": [list(c) for c in cost_configs],
            "max_state_value_spread_sufficient": max_spread_sufficient,
            "max_abs_root_diff_history_vs_dp": max_root_diff,
            "max_spread_when_mask_omitted": max_spread_insufficient,
            "n_condition_evaluations": len(records),
            "tolerance": tol,
        },
        "per_condition": records,
        "environment_scope": (
            "Synthetic finite MDP only: E=4 actions, H=4 rounds, 6 nodes, "
            "budget 0.05, per-action costs in {0, 0.02, 0.05}, 20 seeded random "
            "sparse transition tables, float64 CPU, exhaustive enumeration. "
            "Verifies Markov-sufficiency of the declared state; says nothing "
            "about natural-language MoE hidden states."),
        "runtime_sec": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- #
# A2: information-erasure positive control (C1)
# --------------------------------------------------------------------------- #

def _linear_error(Xtr, ytr, Xte, yte, steps, lr, seed):
    torch.manual_seed(seed)
    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).float().unsqueeze(1)
    Xte_t = torch.from_numpy(Xte).float()
    yte_t = torch.from_numpy(yte).float()
    w = torch.zeros(Xtr.shape[1], 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(Xtr_t @ w + b, ytr_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (Xte_t @ w + b > 0).float().squeeze(1)
    return float((pred != yte_t).float().mean().item())


def _a2_dataset(rng, n, noise_dim, zeta, two_rounds):
    X = rng.integers(0, 2, size=n)
    h = rng.normal(size=(n, noise_dim))  # erased state + irrelevant noise
    e1 = X.copy()                       # round-1 routing records X, then h is zeroed
    Y = X ^ (rng.random(n) < zeta).astype(np.int64)
    feats = {"h": h}
    if two_rounds:
        Z = rng.integers(0, 2, size=n)  # round-2 routes on an unrelated bit
        feats["recent"] = np.eye(2)[Z]
        feats["history"] = np.concatenate([np.eye(2)[e1], np.eye(2)[Z]], axis=1)
    else:
        feats["recent"] = np.eye(2)[e1]
        feats["history"] = np.eye(2)[e1]
    return feats, Y


def run_a2(seed=0, noise_levels=(0.0, 0.1, 0.25, 0.5), n_train=100_000,
           n_test=20_000, noise_dim=32, steps=300, lr=0.1):
    """C1: round-1 routing records input bit X, hidden state is then erased;
    Y = X with label noise zeta.  Theoretical errors: h-only = 1/2,
    history-aware = zeta.  Extended two-round variant routes round 2 on an
    unrelated bit Z so that h+recent-expert also drops to chance while the
    ordered history still recovers Y."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    results = {}
    for variant, two_rounds in (("core_C1", False), ("extended_two_round", True)):
        per_zeta = {}
        for zeta in noise_levels:
            ftr, ytr = _a2_dataset(rng, n_train, noise_dim, zeta, two_rounds)
            fte, yte = _a2_dataset(rng, n_test, noise_dim, zeta, two_rounds)
            row = {}
            for ci, ctrl in enumerate(("h", "recent", "history")):
                ctr, cte = ftr[ctrl], fte[ctrl]
                if ctrl == "h":
                    pass
                elif ctrl == "recent":
                    ctr = np.concatenate([ftr["h"], ctr], axis=1)
                    cte = np.concatenate([fte["h"], cte], axis=1)
                else:
                    ctr = np.concatenate([ftr["h"], ctr], axis=1)
                    cte = np.concatenate([fte["h"], cte], axis=1)
                row[ctrl] = _linear_error(ctr, ytr, cte, yte, steps, lr,
                                          seed + (1 if two_rounds else 0) * 10 + ci)
            row["theory_h_only_error"] = 0.5
            row["theory_history_error"] = zeta
            per_zeta[str(zeta)] = row
        results[variant] = per_zeta

    gates = {}
    core = results["core_C1"]
    for zeta in (0.0, 0.1, 0.25):
        row = core[str(zeta)]
        gates[f"core zeta={zeta}: h_only>=0.45"] = row["h"] >= 0.45
        gates[f"core zeta={zeta}: history<=zeta+0.06"] = row["history"] <= zeta + 0.06
        gates[f"core zeta={zeta}: recent<=zeta+0.06"] = row["recent"] <= zeta + 0.06
    row = core["0.5"]
    for ctrl in ("h", "recent", "history"):
        gates[f"core zeta=0.5: {ctrl}>=0.44 (no super-chance)"] = row[ctrl] >= 0.44
    ext = results["extended_two_round"]
    gates["extended zeta=0: recent>=0.45"] = ext["0.0"]["recent"] >= 0.45
    gates["extended zeta=0: history<=0.06"] = ext["0.0"]["history"] <= 0.06

    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": results,
        "config": {"n_train": n_train, "n_test": n_test, "noise_dim": noise_dim,
                   "steps": steps, "lr": lr, "model": "linear logistic probe"},
        "environment_scope": (
            "Synthetic C1 counterexample only: binary X, erased hidden state + "
            "32-dim irrelevant Gaussian noise, label noise zeta in "
            f"{tuple(noise_levels)}. Hidden state is deliberately insufficient; "
            "a positive history gain here is NOT evidence that real MoE needs "
            "memory. Train/test sizes "
            f"{n_train}/{n_test}, linear probes, CPU."),
        "runtime_sec": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- #
# A3: finite function class - parity readability
# --------------------------------------------------------------------------- #

def _mlp(din, hidden):
    return nn.Sequential(nn.Linear(din, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1))


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


def _train_clf(model, X, y, steps, lr, seed):
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float().unsqueeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward()
        opt.step()
    return model


def _acc(model, X, y):
    with torch.no_grad():
        pred = (model(torch.from_numpy(X).float()) > 0).float().squeeze(1)
    return float((pred == torch.from_numpy(y).float()).float().mean().item())


def _parity_data(n_bits):
    X = ((np.arange(2 ** n_bits)[:, None] >> np.arange(n_bits - 1, -1, -1)[None, :])
         & 1).astype(np.float64)
    cum_par = np.mod(np.cumsum(X, axis=1), 2)  # input-derived history feature
    y = cum_par[:, -1].astype(np.int64)
    return X, cum_par, y


def _a3_one_setting(seed, n_bits, n_train, mlp_steps, lin_steps):
    X, cum_par, y = _parity_data(n_bits)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    tr, te = perm[:n_train], perm[n_train:]
    Xh_tr = np.concatenate([X[tr], cum_par[tr]], axis=1)
    Xh_te = np.concatenate([X[te], cum_par[te]], axis=1)
    Xh_all = np.concatenate([X, cum_par], axis=1)

    hidden_small, hidden_hist, hidden_big = 32, 32, 36
    # parameter-matched big h-only MLP vs the h+history MLP
    while _n_params(_mlp(n_bits, hidden_big)) < _n_params(_mlp(2 * n_bits, hidden_hist)):
        hidden_big += 1

    models = {
        "linear_h_only": (nn.Linear(n_bits, 1), X[tr], X[te], lin_steps, 0.1),
        "mlp_small_h_only": (_mlp(n_bits, hidden_small), X[tr], X[te], mlp_steps, 3e-3),
        "mlp_big_h_only_param_matched": (_mlp(n_bits, hidden_big), X[tr], X[te],
                                         mlp_steps, 3e-3),
        "linear_h_plus_history": (nn.Linear(2 * n_bits, 1), Xh_tr, Xh_te, lin_steps, 0.1),
        "mlp_h_plus_history": (_mlp(2 * n_bits, hidden_hist), Xh_tr, Xh_te,
                               mlp_steps, 3e-3),
    }
    out = {"n_bits": n_bits, "n_train": n_train, "n_test": len(te),
           "param_counts": {}, "test_acc": {}, "full_space_acc": {}}
    for name, (model, xtr, xte, steps, lr) in models.items():
        _train_clf(model, xtr, y[tr], steps, lr, seed + len(name))
        out["param_counts"][name] = _n_params(model)
        out["test_acc"][name] = _acc(model, xte, y[te])
        full_x = Xh_all if "history" in name else X
        out["full_space_acc"][name] = _acc(model, full_x, y)
    return out


def run_a3(seed=0, mlp_steps=3000, lin_steps=1000, scaled_steps=8000):
    """8-bit parity with exhaustive 256-point evaluation plus a 12-bit random
    split.  The history feature (cumulative parity) is constructed from the
    input only; it adds no Bayes information but turns a hard-to-represent
    function into a readable one.  A scaled-data control trains the same small
    h-only MLP on all 256 8-bit inputs to support the function-class/data
    explanation."""
    t0 = time.perf_counter()
    r8 = _a3_one_setting(seed + 1, n_bits=8, n_train=128,
                         mlp_steps=mlp_steps, lin_steps=lin_steps)
    r12 = _a3_one_setting(seed + 2, n_bits=12, n_train=2048,
                          mlp_steps=mlp_steps, lin_steps=lin_steps)

    X8, _, y8 = _parity_data(8)
    scaled = _mlp(8, 32)
    _train_clf(scaled, X8, y8, scaled_steps, 5e-3, seed + 3)
    scaled_full_acc = _acc(scaled, X8, y8)

    gates = {
        "8bit linear_h+history_test>=0.99": r8["test_acc"]["linear_h_plus_history"] >= 0.99,
        "12bit linear_h+history_test>=0.99": r12["test_acc"]["linear_h_plus_history"] >= 0.99,
        "8bit linear_h_only<=0.65": r8["test_acc"]["linear_h_only"] <= 0.65,
        "12bit linear_h_only<=0.65": r12["test_acc"]["linear_h_only"] <= 0.65,
        "scaled_full_data_mlp_full256>=0.90": scaled_full_acc >= 0.90,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": {
            "parity_8bit": r8,
            "parity_12bit": r12,
            "scaled_control_small_mlp_trained_on_all_256": {
                "full_space_acc": scaled_full_acc,
                "note": "same h-only MLP catches up with full data -> "
                        "function-class/readability explanation"},
        },
        "environment_scope": (
            "Synthetic parity only: 8-bit exhaustive (256 inputs, 128/128 random "
            "split) and 12-bit (4096 inputs, 2048/2048 random split). History "
            "features are cumulative input parities (input-derived, never read "
            "labels). Results do not imply natural-language generalization."),
        "runtime_sec": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- #
# B1: delayed-reward planning
# --------------------------------------------------------------------------- #

def _b1_prefix_values(gains):
    """prefix[k] = total gain of spending the first k steps on a chain."""
    return np.concatenate(([0.0], np.cumsum(gains)))


def _b1_dp(gains, costs, budget):
    """Exact finite-budget DP (knapsack over tokens, per-token ordered chain)."""
    n_tokens, n_steps = gains.shape
    dp = np.full(budget + 1, -np.inf)
    dp[0] = 0.0
    for i in range(n_tokens):
        pref = _b1_prefix_values(gains[i])
        new = dp.copy()
        for k in range(1, n_steps + 1):
            c = k * costs[i]
            if c > budget:
                break
            new[c:] = np.maximum(new[c:], dp[: budget + 1 - c] + pref[k])
        dp = new
    return float(dp.max())


def _b1_greedy(gains, costs, budget, horizon, candidate_mask=None):
    """Greedy controller: at each decision, score every token by the sum of
    its next `horizon` chain gains (respecting remaining budget), execute ONE
    step on the argmax, repeat.  horizon=1 is one-step marginal gain."""
    n_tokens, n_steps = gains.shape
    pos = np.zeros(n_tokens, dtype=np.int64)
    total = 0.0
    b = budget
    steps_taken = np.zeros(n_tokens, dtype=np.int64)
    while True:
        best_i, best_score = -1, 0.0
        for i in range(n_tokens):
            if candidate_mask is not None and not candidate_mask[i]:
                continue
            if pos[i] >= n_steps or costs[i] > b:
                continue
            score = 0.0
            for h in range(horizon):
                if pos[i] + h >= n_steps or (h + 1) * costs[i] > b:
                    break
                score += gains[i, pos[i] + h]
            if score > best_score:
                best_i, best_score = i, score
        if best_i < 0:
            break
        total += gains[best_i, pos[best_i]]
        pos[best_i] += 1
        steps_taken[best_i] += 1
        b -= costs[best_i]
    return total, steps_taken


def _b1_uniform(gains, costs, budget):
    """Fixed uniform budget spread: round-robin one step per token while
    affordable."""
    n_tokens, n_steps = gains.shape
    pos = np.zeros(n_tokens, dtype=np.int64)
    total, b = 0.0, budget
    moved = True
    while moved:
        moved = False
        for i in range(n_tokens):
            if pos[i] < n_steps and costs[i] <= b:
                total += gains[i, pos[i]]
                pos[i] += 1
                b -= costs[i]
                moved = True
    return total, pos


def _b1_gen_env(rng, n_tokens=8, n_steps=4, cost_cfg="uniform"):
    gains = rng.uniform(0.2, 3.0, size=(n_tokens, n_steps))
    n_delayed = int(rng.integers(1, 3))  # plant 1-2 low-then-high chains
    delayed = rng.choice(n_tokens, size=n_delayed, replace=False)
    high_pos = {}
    for i in delayed:
        hp = int(rng.integers(1, 3))  # high gain at position 1 or 2
        gains[i] = rng.uniform(0.05, 0.3, size=n_steps)
        gains[i, hp] = rng.uniform(7.0, 10.0)
        high_pos[i] = hp
    if cost_cfg == "uniform":
        costs = np.ones(n_tokens, dtype=np.int64)
        budget = 4
    else:
        costs = rng.choice(np.array([1, 2, 4]), size=n_tokens)
        budget = 8
    return gains, costs, budget, {int(k): v for k, v in high_pos.items()}


def _b1_features(gains, costs, budget, pos, b_left, i):
    past = gains[i, :pos[i]]
    return np.array([
        pos[i] / gains.shape[1],
        past.sum() / 10.0,
        past[-1] if len(past) else 0.0,
        costs[i] / 4.0,
        b_left / max(budget, 1),
    ])


def _b1_supervised_q(seed, n_train_envs=25, n_samples=20_000, steps=1500,
                     batch=512, n_tokens=8, n_steps=4):
    """Supervised Q^{pi0} controller: regress the realized return of taking one
    step on token i and then following uniform-random pi0, on features visible
    at deployment (no labels of future gains)."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    Xs, ys = [], []
    envs = [_b1_gen_env(rng, n_tokens, n_steps, "uniform")
            for _ in range(n_train_envs)]
    while len(Xs) < n_samples:
        gains, costs, budget, _ = envs[int(rng.integers(len(envs)))]
        pos = np.zeros(n_tokens, dtype=np.int64)
        b = budget
        while b >= costs.min():
            cand = [i for i in range(n_tokens)
                    if pos[i] < n_steps and costs[i] <= b]
            if not cand:
                break
            i = cand[int(rng.integers(len(cand)))]
            # realized return of stepping i now, then pi0
            sim_pos = pos.copy()
            sim_b = b
            ret = gains[i, sim_pos[i]]
            sim_pos[i] += 1
            sim_b -= costs[i]
            while True:
                c2 = [j for j in range(n_tokens)
                      if sim_pos[j] < n_steps and costs[j] <= sim_b]
                if not c2:
                    break
                j = c2[int(rng.integers(len(c2)))]
                ret += gains[j, sim_pos[j]]
                sim_pos[j] += 1
                sim_b -= costs[j]
            Xs.append(_b1_features(gains, costs, budget, pos, b, i))
            ys.append(ret)
            # advance the pi0 episode
            pos[i] += 1
            b -= costs[i]
            if not any(pos[j] < n_steps and costs[j] <= b for j in range(n_tokens)):
                break
    X = np.array(Xs[:n_samples])
    y = np.array(ys[:n_samples])
    n_fit = int(0.8 * len(X))
    torch.manual_seed(seed)
    model = _mlp(X.shape[1], 64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(X[:n_fit]).float()
    yt = torch.from_numpy(y[:n_fit]).float().unsqueeze(1)
    for _ in range(steps):
        idx = torch.randint(0, n_fit, (batch,))
        opt.zero_grad()
        loss = lossf(model(Xt[idx]), yt[idx])
        loss.backward()
        opt.step()
    train_time = time.perf_counter() - t0

    # calibration on held-out samples
    with torch.no_grad():
        pred = model(torch.from_numpy(X[n_fit:]).float()).squeeze(1).numpy()
    real = y[n_fit:]
    order = np.argsort(pred)
    bins = np.array_split(order, 10)
    cal_err = max(float(abs(pred[b].mean() - real[b].mean())) for b in bins if len(b))
    corr = float(np.corrcoef(pred, real)[0, 1])

    def act(gains_env, costs_env, budget_env):
        pos = np.zeros(n_tokens, dtype=np.int64)
        total, b = 0.0, budget_env
        while True:
            cand = [i for i in range(n_tokens)
                    if pos[i] < n_steps and costs_env[i] <= b]
            if not cand:
                break
            feats = np.stack([_b1_features(gains_env, costs_env, budget_env,
                                           pos, b, i) for i in cand])
            with torch.no_grad():
                q = model(torch.from_numpy(feats).float()).squeeze(1).numpy()
            i = cand[int(np.argmax(q))]
            total += gains_env[i, pos[i]]
            pos[i] += 1
            b -= costs_env[i]
        return total, pos

    return act, {"n_samples": len(X), "train_time_sec": train_time,
                 "calibration_max_bin_gap": cal_err,
                 "calibration_pearson_corr": corr}


def run_b1(seed=0, n_envs=30, quick=False):
    """C5 exact env plus random 8-token/4-step envs with planted low-then-high
    chains.  Uniform vs {1,2,4} token costs are evaluated separately."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    # --- exact 2-token env (C5) ---
    gains_c5 = np.array([[0.1, 9.9], [6.0, 0.0]])
    costs_c5 = np.ones(2, dtype=np.int64)
    dp_c5 = _b1_dp(gains_c5, costs_c5, 2)
    greedy_c5, _ = _b1_greedy(gains_c5, costs_c5, 2, horizon=1)
    two_c5, _ = _b1_greedy(gains_c5, costs_c5, 2, horizon=2)
    uniform_c5, _ = _b1_uniform(gains_c5, costs_c5, 2)
    pruned_mask = gains_c5[:, 0] >= np.median(gains_c5[:, 0])
    pruned_c5, _ = _b1_greedy(gains_c5, costs_c5, 2, horizon=2,
                              candidate_mask=pruned_mask)
    exact = {"dp": dp_c5, "one_step": greedy_c5, "two_step": two_c5,
             "uniform_budget": uniform_c5, "two_step_pruned": pruned_c5,
             "regret_one_step": dp_c5 - greedy_c5,
             "regret_two_step": dp_c5 - two_c5,
             "regret_two_step_pruned": dp_c5 - pruned_c5}

    # --- supervised Q controller (trained on uniform-cost envs) ---
    q_act, q_meta = _b1_supervised_q(seed + 100,
                                     n_samples=6_000 if quick else 20_000,
                                     steps=600 if quick else 1500)

    # --- random envs ---
    random_results = {}
    for cfg in ("uniform", "nonuniform"):
        methods = ("uniform_budget", "one_step", "two_step", "two_step_pruned",
                   "supervised_q", "dp")
        sums = {m: 0.0 for m in methods}
        delayed_hit = {m: 0 for m in methods}
        per_env = []
        envs = [_b1_gen_env(rng, cost_cfg=cfg) for _ in range(n_envs)]
        for gains, costs, budget, high_pos in envs:
            dp_v = _b1_dp(gains, costs, budget)
            vals = {
                "uniform_budget": _b1_uniform(gains, costs, budget),
                "one_step": _b1_greedy(gains, costs, budget, horizon=1),
                "two_step": _b1_greedy(gains, costs, budget, horizon=2),
                "two_step_pruned": _b1_greedy(
                    gains, costs, budget, horizon=2,
                    candidate_mask=gains[:, 0] >= np.median(gains[:, 0])),
                "dp": (dp_v, None),
            }
            if cfg == "uniform":
                vals["supervised_q"] = q_act(gains, costs, budget)
            else:
                vals["supervised_q"] = (np.nan, None)
            row = {"dp_value": dp_v, "regret": {}}
            for m in methods:
                v, pos = vals[m]
                if m == "supervised_q" and cfg != "uniform":
                    row["regret"][m] = None
                    continue
                sums[m] += dp_v - v
                if high_pos and pos is not None:
                    ok = all(pos[i] > hp for i, hp in high_pos.items())
                    delayed_hit[m] += int(ok)
                row["regret"][m] = dp_v - v
            per_env.append(row)
        n_with_chains = n_envs
        random_results[cfg] = {
            "mean_regret": {m: (sums[m] / n_envs if not (m == "supervised_q" and cfg != "uniform") else None)
                            for m in methods},
            "delayed_chain_fully_exploited_frac": {
                m: (delayed_hit[m] / n_with_chains if not (m == "supervised_q" and cfg != "uniform") else None)
                for m in methods},
            "n_envs": n_envs,
            "per_env_regret_sample": per_env[:5],
        }

    gates = {
        "exact_env_dp==10": abs(exact["dp"] - 10.0) < 1e-9,
        "exact_env_one_step==6.1": abs(exact["one_step"] - 6.1) < 1e-9,
        "exact_env_two_step==10": abs(exact["two_step"] - 10.0) < 1e-9,
        "exact_env_pruned_irreducible_gap>3": exact["regret_two_step_pruned"] > 3.0,
        "random_uniform_dp_regret==0": abs(random_results["uniform"]["mean_regret"]["dp"]) < 1e-9,
        "random_nonuniform_dp_regret==0": abs(random_results["nonuniform"]["mean_regret"]["dp"]) < 1e-9,
        "random_uniform_two_step<=one_step":
            random_results["uniform"]["mean_regret"]["two_step"]
            <= random_results["uniform"]["mean_regret"]["one_step"] + 1e-9,
        "random_nonuniform_two_step<=one_step":
            random_results["nonuniform"]["mean_regret"]["two_step"]
            <= random_results["nonuniform"]["mean_regret"]["one_step"] + 1e-9,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": {
            "exact_c5_env": exact,
            "random_envs": random_results,
            "supervised_q_training": q_meta,
        },
        "environment_scope": (
            "Synthetic delayed-reward envs only: exact C5 (2 tokens, budget 2, "
            "chains A=(0.1,9.9), B=(6,0)) and 30 seeded random 8-token/4-step "
            "envs with 1-2 planted low-then-high chains; costs uniform {1} and "
            "non-uniform {1,2,4} evaluated separately. Supervised Q controller "
            "is trained and evaluated on uniform-cost envs only. Recovered "
            "two-step gains here do not imply the same structure exists in "
            "natural language."),
        "runtime_sec": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- #
# B2: unlearnable counterfactual boundary (C2)
# --------------------------------------------------------------------------- #

def _b2_data(rng, n, signal, feat_dim=8):
    x = rng.integers(0, 2, size=n)
    feats = np.concatenate([x[:, None].astype(np.float64),
                            rng.normal(size=(n, feat_dim - 1))], axis=1)
    q = 0.5 + 0.5 * signal  # P(Y = x)
    y = np.where(rng.random(n) < q, x, 1 - x)
    return feats, x, y


def _nll_of_action(y, p1):
    return -(y * math.log(p1) + (1 - y) * math.log(1 - p1))


def _b2_fit_loss_regressor(feats, targets, steps, lr, seed):
    torch.manual_seed(seed)
    Xt = torch.from_numpy(feats).float()
    yt = torch.from_numpy(targets).float().unsqueeze(1)
    model = nn.Sequential(nn.Linear(feats.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward()
        opt.step()
    return model


def run_b2(seed=0, signals=(0.0, 0.25, 0.5, 0.75, 1.0), n_train=40_000,
           n_test=20_000, steps=600, lr=5e-3):
    """C2: X carries tunable signal about Y ~ Bern(1/2).  Expert 1 outputs
    P(y=1)=0.99, expert 2 outputs 0.01, stop outputs 0.5.  Per-expert loss
    regressors are trained on train splits only; deployment sees inputs only.
    The label-visible oracle is always near -log(0.99); a label-blind
    controller can never beat the achievable optimum (log 2 at zero signal)."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    curves = {}
    for s in signals:
        ftr, xtr, ytr = _b2_data(rng, n_train, s)
        fte, xte, yte = _b2_data(rng, n_test, s)
        nll1_tr = np.array([_nll_of_action(y, 0.99) for y in ytr])
        nll2_tr = np.array([_nll_of_action(y, 0.01) for y in ytr])
        r1 = _b2_fit_loss_regressor(ftr, nll1_tr, steps, lr, seed + 11)
        r2 = _b2_fit_loss_regressor(ftr, nll2_tr, steps, lr, seed + 12)
        with torch.no_grad():
            p1 = r1(torch.from_numpy(fte).float()).squeeze(1).numpy()
            p2 = r2(torch.from_numpy(fte).float()).squeeze(1).numpy()
        pred_losses = np.stack([p1, p2, np.full(len(p1), LOG2)], axis=1)
        choice = pred_losses.argmin(axis=1)
        nll_true = np.stack([
            np.array([_nll_of_action(y, 0.99) for y in yte]),
            np.array([_nll_of_action(y, 0.01) for y in yte]),
            np.full(len(yte), LOG2)], axis=1)
        controller_loss = float(nll_true[np.arange(len(yte)), choice].mean())
        oracle_loss = float(nll_true[:, :2].min(axis=1).mean())
        stop_loss = LOG2
        # achievable optimum for a label-blind controller (knows the true q)
        q = 0.5 + 0.5 * s
        l1 = -(q * math.log(0.99) + (1 - q) * math.log(0.01))
        l2 = -((1 - q) * math.log(0.99) + q * math.log(0.01))
        optimal = min(l1, l2, LOG2)
        curves[str(s)] = {
            "stop_loss": stop_loss,
            "label_visible_oracle_loss": oracle_loss,
            "held_out_controller_loss": controller_loss,
            "achievable_optimum_label_blind": optimal,
            "action_fraction": {"expert1": float((choice == 0).mean()),
                                 "expert2": float((choice == 1).mean()),
                                 "stop": float((choice == 2).mean())},
        }
    gates = {}
    for s in signals:
        row = curves[str(s)]
        gates[f"s={s}: oracle~0.01005"] = abs(row["label_visible_oracle_loss"]
                                             - ORACLE_NLL) < 0.01
        gates[f"s={s}: controller>=optimum-0.02 (no super-Bayes)"] = \
            row["held_out_controller_loss"] >= row["achievable_optimum_label_blind"] - 0.02
    gates["s=0: controller>=log2-0.02"] = curves["0.0"]["held_out_controller_loss"] >= LOG2 - 0.02
    gates["s=0: controller<=log2+0.05"] = curves["0.0"]["held_out_controller_loss"] <= LOG2 + 0.05
    gates["s=1: controller<=0.05 (signal usable)"] = curves["1.0"]["held_out_controller_loss"] <= 0.05
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": {"curves_vs_signal": curves,
                    "theory": {"stop_log2": LOG2, "oracle_minus_log_0.99": ORACLE_NLL}},
        "config": {"n_train": n_train, "n_test": n_test, "steps": steps,
                   "model": "per-expert MLP(8->32->1) MSE loss regressors"},
        "environment_scope": (
            "Synthetic C2 only: X = 1 bit + 7 noise dims, Y = X with "
            "probability 0.5+0.5*signal, signal sweep "
            f"{tuple(signals)}. Guards against claiming a value model learned "
            "Q* from label-visible cherry-picking; conclusions are limited to "
            "this action set {0.99, 0.01, 0.5}."),
        "runtime_sec": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- #
# B3: ordered composition with non-commuting affine experts
# --------------------------------------------------------------------------- #

def _b3_experts(p):
    a = [2, 3, 4, 5]
    b = [1, 0, 2, 3]
    return a, b


def _b3_apply(e, x, a, b, p):
    return (a[e] * x + b[e]) % p


def _b3_compose(seq, x0, a, b, p):
    x = x0
    for e in seq:
        x = _b3_apply(e, x, a, b, p)
    return x


def _b3_noncommuting_pairs(a, b, p):
    pairs = []
    for e in range(4):
        for f in range(4):
            if e >= f:
                continue
            for x in range(p):
                if _b3_apply(f, _b3_apply(e, x, a, b, p), a, b, p) != \
                        _b3_apply(e, _b3_apply(f, x, a, b, p), a, b, p):
                    pairs.append((e, f))
                    break
    return pairs


def _b3_router_features(x, t, desc, L_max, p):
    fx = np.zeros(p)
    fx[x] = 1.0
    ft = np.zeros(L_max)
    ft[t] = 1.0
    fd = np.zeros(L_max * 5)
    for k in range(L_max):
        fd[k * 5 + (desc[k] if k < len(desc) else 4)] = 1.0
    return np.concatenate([fx, ft, fd])


def _b3_all_sequences(L_max_train=4):
    seqs = []
    for L in range(1, L_max_train + 1):
        seqs.extend(_b3_seqs_of_len(L))
    return seqs


def _b3_seqs_of_len(L):
    if L == 0:
        return [()]
    return [s + (e,) for s in _b3_seqs_of_len(L - 1) for e in range(4)]


def _b3_pick_holdout(seqs, seed, frac=0.15):
    """Hold out complete length-4 paths while keeping every single op and
    every ordered adjacent pair covered by the training set."""
    rng = np.random.default_rng(seed)
    len4 = [s for s in seqs if len(s) == 4]
    rest = [s for s in seqs if len(s) < 4]
    for attempt in range(200):
        idx = rng.choice(len(len4), size=int(frac * len(len4)), replace=False)
        holdout = {len4[i] for i in idx}
        train = rest + [s for s in len4 if s not in holdout]
        singles = {e for s in train for e in s}
        pairs = {(s[k], s[k + 1]) for s in train for k in range(len(s) - 1)}
        if singles == {0, 1, 2, 3} and len(pairs) == 16:
            return train, sorted(holdout)
    raise RuntimeError("could not find a coverage-preserving holdout")


def run_b3(seed=0, p=97, router_steps=4000, expert_steps=1500,
           n_x0_train=8, n_x0_test=20, quick=False):
    """Four non-commuting affine experts F_e(x)=a_e x + b_e (mod p).  Phase 1
    fixes exact expert functions and trains only the router (per-step
    cross-entropy on the requested operation).  Phase 2 replaces experts with
    small MLPs trained on a subset of inputs, exposing the interaction between
    function-approximation error and control error.  Lengths 5-6 are reported
    separately as extrapolation, never mixed into the H<=4 table."""
    t0 = time.perf_counter()
    a, b = _b3_experts(p)
    noncomm = _b3_noncommuting_pairs(a, b, p)
    L_max = 6
    seqs = _b3_all_sequences(4)
    train_seqs, holdout_seqs = _b3_pick_holdout(seqs, seed)
    rng = np.random.default_rng(seed + 1)
    x0_train = rng.choice(p, size=n_x0_train, replace=False)
    x0_test = rng.choice(p, size=n_x0_test, replace=False)

    # --- phase 1: router with exact experts ---
    dim = p + L_max + L_max * 5
    Xs, ys = [], []
    for s in train_seqs:
        desc = list(s)
        for x0 in x0_train:
            x = int(x0)
            for t, e in enumerate(s):
                Xs.append(_b3_router_features(x, t, desc, L_max, p))
                ys.append(e)
                x = _b3_apply(e, x, a, b, p)
    X = np.array(Xs)
    y = np.array(ys)
    torch.manual_seed(seed)
    router = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 4))
    opt = torch.optim.Adam(router.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    for _ in range(router_steps):
        opt.zero_grad()
        loss = lossf(router(Xt), yt)
        loss.backward()
        opt.step()

    def rollout(seq_list, x0_list, expert_fn):
        correct, total = 0, 0
        per_len = {}
        for s in seq_list:
            for x0 in x0_list:
                x = int(x0)
                desc = list(s)
                for t in range(len(s)):
                    f = _b3_router_features(x, t, desc, L_max, p)
                    with torch.no_grad():
                        e = int(router(torch.from_numpy(f).float()).argmax().item())
                    x = expert_fn(e, x)
                target = _b3_compose(s, int(x0), a, b, p)
                correct += int(x == target)
                total += 1
                per_len.setdefault(len(s), []).append(int(x == target))
        return (correct / total if total else 0.0,
                {k: float(np.mean(v)) for k, v in sorted(per_len.items())})

    exact_fn = lambda e, x: _b3_apply(e, x, a, b, p)
    rng_eval = np.random.default_rng(seed + 2)
    random_seqs = [tuple(rng_eval.integers(0, 4, size=4)) for _ in range(100)]
    reversed_seqs = [tuple(reversed(s)) for s in train_seqs if len(s) == 4][:100]
    repeated_seqs = [(e, e, e, e) for e in range(4)]
    extra_seqs = [tuple(rng_eval.integers(0, 4, size=L))
                  for L in (5, 6) for _ in range(50)]

    acc_train, pl_train = rollout(train_seqs, x0_test, exact_fn)
    acc_hold, pl_hold = rollout(holdout_seqs, x0_test, exact_fn)
    acc_rand, _ = rollout(random_seqs, x0_test, exact_fn)
    acc_rev, _ = rollout(reversed_seqs, x0_test, exact_fn)
    acc_rep, _ = rollout(repeated_seqs, x0_test, exact_fn)
    acc_extra, pl_extra = rollout(extra_seqs, x0_test, exact_fn)

    # --- phase 2: small MLP approximate experts ---
    # Scalar + cyclic input encoding; the target is the output angle on the
    # circle (sin, cos), decoded by nearest class angle.  Trained on 60/97
    # inputs per expert, leaving genuine partial approximation error whose
    # compounding over loop depth interacts with control error.
    def x_feat(x):
        return np.array([x / p, math.sin(2 * math.pi * x / p),
                         math.cos(2 * math.pi * x / p)])

    class_ang = np.stack([np.array([math.sin(2 * math.pi * k / p),
                                    math.cos(2 * math.pi * k / p)])
                          for k in range(p)])

    def train_mlp_expert(e):
        rng_e = np.random.default_rng(seed + 10 + e)
        x_tr = rng_e.choice(p, size=60, replace=False)
        x_te = np.array([x for x in range(p) if x not in set(x_tr)])
        torch.manual_seed(seed + 20 + e)
        net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64),
                            nn.ReLU(), nn.Linear(64, 2))
        opt = torch.optim.Adam(net.parameters(), lr=5e-3)
        lossf = nn.MSELoss()
        Xin = torch.from_numpy(np.stack([x_feat(int(x)) for x in x_tr])).float()
        Yout = torch.from_numpy(np.stack([class_ang[_b3_apply(e, int(x), a, b, p)]
                                          for x in x_tr])).float()
        for _ in range(expert_steps):
            opt.zero_grad()
            loss = lossf(net(Xin), Yout)
            loss.backward()
            opt.step()

        def predict(xs):
            with torch.no_grad():
                out = net(torch.from_numpy(
                    np.stack([x_feat(int(x)) for x in xs])).float()).numpy()
            return ((out[:, None, :] - class_ang[None, :, :]) ** 2).sum(-1).argmin(1)

        tgt_te = np.array([_b3_apply(e, int(x), a, b, p) for x in x_te])
        return predict, float((predict(x_te) == tgt_te).mean())

    mlp_experts, per_expert_acc = [], []
    for e in range(4):
        predict_e, acc_e = train_mlp_expert(e)
        mlp_experts.append(predict_e)
        per_expert_acc.append(acc_e)

    def mlp_fn(e, x):
        return int(mlp_experts[e]([x])[0])

    acc_hold_mlp, pl_hold_mlp = rollout(holdout_seqs, x0_test, mlp_fn)
    acc_rand_mlp, pl_rand_mlp = rollout(random_seqs, x0_test, mlp_fn)

    gates = {
        "experts_noncommuting": len(noncomm) >= 1,
        "holdout_combos_exact_experts>=0.90": acc_hold >= 0.90,
        "train_combos_exact_experts>=0.95": acc_train >= 0.95,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": {
            "modulus_p": p,
            "expert_coeffs": {"a": a, "b": b},
            "noncommuting_pairs": noncomm,
            "n_train_sequences": len(train_seqs),
            "n_holdout_sequences": len(holdout_seqs),
            "phase1_exact_experts": {
                "rollout_acc_train_combos": acc_train,
                "rollout_acc_holdout_combos": acc_hold,
                "per_length_holdout": pl_hold,
                "controls": {
                    "reversed_paths": acc_rev,
                    "repeated_same_expert": acc_rep,
                    "random_equal_length_paths": acc_rand,
                },
            },
            "extrapolation_length_5_6_exact_experts": {
                "rollout_acc": acc_extra, "per_length": pl_extra,
                "label": "EXTRAPOLATION - not comparable to H<=4 budget table"},
            "phase2_mlp_experts": {
                "per_expert_heldout_input_acc": per_expert_acc,
                "rollout_acc_holdout_combos": acc_hold_mlp,
                "per_length_holdout": pl_hold_mlp,
                "rollout_acc_random_paths": acc_rand_mlp,
                "per_length_random": pl_rand_mlp,
            },
        },
        "environment_scope": (
            f"Synthetic ordered composition only: 4 affine experts mod {p}, "
            "train lengths 1-4 with held-out complete paths (all single ops "
            "and adjacent pairs still covered in training). Phase-1 router "
            "supervision reads the operation description; phase-2 MLP experts "
            "are trained on 60/97 inputs each. Concludes only that the control "
            "mechanism can learn ordered composition; does not claim real "
            "experts have prescribed semantics."),
        "runtime_sec": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

SUBEXPS = {"A1": run_a1, "A2": run_a2, "A3": run_a3,
           "B1": run_b1, "B2": run_b2, "B3": run_b3}


def main():
    ap = argparse.ArgumentParser(description="E08 A/B synthetic mechanism experiments")
    ap.add_argument("--output", required=True, help="path to write JSON results")
    ap.add_argument("--subexp", default=",".join(SUBEXPS),
                    help="comma-separated subset of A1,A2,A3,B1,B2,B3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="reduced sizes for smoke tests")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = {}
    for name in [s.strip().upper() for s in args.subexp.split(",") if s.strip()]:
        if name not in SUBEXPS:
            raise ValueError(f"unknown subexperiment {name}")
        kwargs = {"seed": args.seed}
        if args.quick:
            if name == "A1":
                kwargs["n_tables"] = 4
            elif name == "A2":
                kwargs.update(n_train=8_000, n_test=4_000, steps=150)
            elif name == "A3":
                kwargs.update(mlp_steps=800, lin_steps=400, scaled_steps=2000)
            elif name == "B1":
                kwargs.update(n_envs=6)
            elif name == "B2":
                kwargs.update(n_train=8_000, n_test=4_000, steps=300)
            elif name == "B3":
                kwargs.update(router_steps=2500, expert_steps=800,
                              n_x0_train=6, n_x0_test=10)
        kwargs = {k: v for k, v in kwargs.items()
                  if k in SUBEXPS[name].__code__.co_varnames}
        results[name] = SUBEXPS[name](**kwargs)

    out = {
        "experiment": "E08 A/B: stability and synthetic mechanisms",
        "plan": "plans/loop_moe_research_plan_2026-10-25/experiments/E08_stability_and_synthetic_mechanisms.md",
        "seed": args.seed,
        "quick_mode": args.quick,
        "environment": {"python": __import__("sys").version.split()[0],
                        "numpy": np.__version__, "torch": torch.__version__,
                        "device": "cpu", "dtype": "float64 (numpy) / float32 (torch)"},
        "subexperiments": results,
        "overall_passed": all(r["passed"] for r in results.values()),
    }
    def _jsonable(o):
        if isinstance(o, (np.bool_, np.integer, np.floating)):
            return o.item()
        raise TypeError(f"not serializable: {type(o)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=_jsonable)
    for name, r in results.items():
        print(f"[{name}] passed={r['passed']} runtime={r['runtime_sec']:.1f}s")
        for g, ok in r["gates"].items():
            print(f"    {'PASS' if ok else 'FAIL'}  {g}")
    print(f"overall_passed={out['overall_passed']} -> {args.output}")


if __name__ == "__main__":
    main()
