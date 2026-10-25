"""Reproduce mathematical sanity checks; no GPU or ML training is performed.

Run: python3 verify_counterexamples.py
All examples are intentionally small and use the Python standard library.
"""

from itertools import product
from math import comb, log, sqrt
from pathlib import Path
import json


def seqtopk_executed(rows, k):
    executed = []
    for m in range(1, len(rows) + 1):
        ranked = sorted(
            ((rows[t][e], t, e) for t in range(m) for e in range(len(rows[t]))),
            reverse=True,
        )[:m * k]
        executed.append(sum(t == m - 1 for _, t, _ in ranked))
    return executed


def potential(profile, values, cost):
    loads = [profile.count(e) for e in range(len(values[0]))]
    return sum(values[n][a] for n, a in enumerate(profile)) - sum(
        cost(e, j) for e, load in enumerate(loads) for j in range(1, load + 1)
    )


def utility(profile, n, values, cost):
    e = profile[n]
    return values[n][e] - cost(e, profile.count(e))


def check_exact_potential():
    values = [[4, 5], [5, 4]]
    cost = lambda e, j: 0 if j == 1 else 10
    for profile in product(range(2), repeat=2):
        for n in range(2):
            alt = tuple((1 - a) if i == n else a for i, a in enumerate(profile))
            du = utility(alt, n, values, cost) - utility(profile, n, values, cost)
            dp = potential(alt, values, cost) - potential(profile, values, cost)
            assert abs(du - dp) < 1e-12
    low = (0, 1)
    assert all(
        utility(low, n, values, cost) > utility(
            tuple((1 - a) if i == n else a for i, a in enumerate(low)), n, values, cost
        )
        for n in range(2)
    )
    assert potential(low, values, cost) == 8
    assert potential((1, 0), values, cost) == 10


def matmul2(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def real_spectral_radius2(a):
    trace = a[0][0] + a[1][1]
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    discriminant = trace * trace - 4 * det
    assert discriminant >= 0, "This helper is only used for real-eigenvalue examples."
    roots = ((trace + sqrt(discriminant)) / 2, (trace - sqrt(discriminant)) / 2)
    return max(abs(r) for r in roots)


def check_pigou():
    cost = lambda e, load: load if e == 0 else 2.5
    social = lambda a: sum(cost(e, a.count(e)) for e in a)
    potentials = {
        a: sum(cost(e, j) for e in range(2) for j in range(1, a.count(e) + 1))
        for a in product(range(2), repeat=2)
    }
    equilibrium = min(potentials, key=potentials.get)
    assert equilibrium == (0, 0)
    optimum = min(social(a) for a in potentials)
    assert social(equilibrium) == 4 and optimum == 3.5
    return social(equilibrium), optimum


def check_delayed_gain():
    gains = [[0.1, 9.9], [6.0, 0.0]]
    counts = [0, 0]
    greedy = 0.0
    for _ in range(2):
        n = max((n for n in range(2) if counts[n] < 2), key=lambda n: gains[n][counts[n]])
        greedy += gains[n][counts[n]]
        counts[n] += 1
    optimum = max(sum(gains[0][:k]) + sum(gains[1][:2 - k]) for k in range(3))
    assert abs(greedy - 6.1) < 1e-12 and optimum == 10.0
    return greedy, optimum


def check_price_cycle():
    prices = [0, 0]
    trajectories = []
    for _ in range(4):
        expert = min(range(2), key=lambda e: prices[e])
        load = [0, 0]
        load[expert] = 2
        prices = [max(0, prices[e] + load[e] - 1) for e in range(2)]
        trajectories.append({"load": load, "updated_prices": prices[:]})
    assert trajectories[1] == trajectories[3]
    assert all(max(step["load"]) == 2 for step in trajectories)
    return trajectories


def main():
    check_exact_potential()
    a = [[0.5, 2], [0, 0.5]]
    b = [[0.5, 0], [2, 0.5]]
    pigou_equilibrium, pigou_optimal = check_pigou()
    delayed_greedy, delayed_optimal = check_delayed_gain()
    price_cycle = check_price_cycle()
    rows = [[0.4, 0.3, 0.2, 0.1], [0.49, 0.48, 0.02, 0.01]]
    assert all(abs(sum(row) - 1) < 1e-12 for row in rows)
    counts = seqtopk_executed(rows, 1)
    assert counts == [1, 2] and sum(counts) > 2
    # The same current prefix receives different allocation under two suffixes.
    prefix = [0.4, 0.3, 0.2, 0.1]
    first_counts = []
    for suffix in [[0.27, 0.26, 0.24, 0.23], [0.70, 0.11, 0.10, 0.09]]:
        ranked = sorted(
            ((score, t) for t, row in enumerate([prefix, suffix]) for score in row),
            reverse=True,
        )[:4]
        first_counts.append(sum(t == 0 for _, t in ranked))
    assert first_counts == [2, 3]
    record = {
        "spectral_radius_A": real_spectral_radius2(a),
        "spectral_radius_B": real_spectral_radius2(b),
        "spectral_radius_BA": real_spectral_radius2(matmul2(b, a)),
        "top1_h4_leaf_paths": 8 ** 4,
        "top1_h4_all_stopping_prefixes": sum(8 ** i for i in range(5)),
        "top1_h4_expert_evaluations_shared_tree": sum(8 ** i for i in range(1, 5)),
        "top2_pair_h4_leaf_paths": comb(8, 2) ** 4,
        "oracle_random_target_nll": -log(0.99),
        "deploy_fixed_extreme_nll": -0.5 * (log(0.99) + log(0.01)),
        "deploy_uniform_nll": log(2),
        "seqtopk_counterexample_calls": counts,
        "seqtopk_prefix_budget": 2,
        "offline_prefix_calls_with_different_suffixes": first_counts,
        "suboptimal_strict_equilibrium_potential": 8,
        "better_equilibrium_potential": 10,
        "pigou_equilibrium_social_cost": pigou_equilibrium,
        "pigou_optimal_social_cost": pigou_optimal,
        "delayed_gain_greedy_reward": delayed_greedy,
        "delayed_gain_optimal_reward": delayed_optimal,
        "price_cycle": price_cycle,
        "executed_checks": "Exact finite examples and algebra only; no language-model training.",
    }
    path = Path(__file__).with_name("verified_toy_calculations.json")
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
