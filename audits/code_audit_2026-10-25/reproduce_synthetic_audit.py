"""Independent CPU checks against the pinned repository; source files unchanged.

Run: python reproduce_synthetic_audit.py --repo ../GB_RL_MoE_Loop --output synthetic_audit_checks.json
Torch-dependent imports are avoided by extracting only pure function AST nodes
from e07_replay.py. That extraction is explicitly recorded below.
"""
import argparse
import ast
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prefix_knapsack(deltas, budget):
    """Exact separable prefix allocation, O(N B H), including optional stop."""
    budget = min(budget, sum(map(len, deltas)))
    dp = np.full(budget + 1, -np.inf)
    dp[0] = 0.0
    parents = []
    for row in deltas:
        pref = np.r_[0.0, np.cumsum(row)]
        new = np.full(budget + 1, -np.inf)
        choice = np.zeros(budget + 1, dtype=int)
        for k, gain in enumerate(pref):
            if k > budget:
                break
            candidate = dp[:budget + 1 - k] + gain
            improved = candidate > new[k:]
            new[k:][improved] = candidate[improved]
            choice[k:][improved] = k
        parents.append(choice)
        dp = new
    b = int(np.argmax(dp))
    value = float(dp[b])
    allocation = []
    for choice in reversed(parents):
        k = int(choice[b])
        allocation.append(k)
        b -= k
    allocation.reverse()
    return value, allocation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1] / 'GB_RL_MoE_Loop')
    ap.add_argument('--output', type=Path, default=Path(__file__).with_name('synthetic_audit_checks.json'))
    args = ap.parse_args()
    repo = args.repo.resolve()
    e07 = load_file('audit_e07_synthetic', repo / 'experiments/e07_synthetic.py')
    e06 = load_file('audit_e05_e06', repo / 'experiments/e05_e06_synthetic.py')
    wanted = {'_wrap_with_steps', 'equal_depth_allocation', 'rolling_window_dp_allocation', 'small_window_dp_gap'}
    source = ast.parse((repo / 'experiments/e07_replay.py').read_text())
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {'e07syn': e07, 'random': random, 'statistics': statistics, 'math': math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(repo / 'experiments/e07_replay.py'), 'exec'), namespace)
    out = {'repository_commit': 'f2bef422f7315f30618f00c690bf82810d1e85ac',
           'execution': 'CPU NumPy and pure Python; no model weights or GPU; e07_replay pure AST functions extracted to avoid importing Torch'}

    # train/exact.py:31, F log p / |paths|, has a different gradient from sum p F.
    logit = math.log(4.0)
    probability = 1.0 / (1.0 + math.exp(-logit))
    costs = [1.0, 2.0]
    correct = probability * (1-probability) * (costs[1]-costs[0])
    implemented = (-probability*costs[0] + (1-probability)*costs[1]) / 2.0
    lr = 0.1
    objective = lambda z: costs[0] + (costs[1]-costs[0])/(1.0+math.exp(-z))
    assert correct > 0 and implemented < 0
    assert objective(logit-lr*implemented) > objective(logit)
    out['exact_trainer_missing_probability'] = {
        'costs': costs, 'p_action_1': probability,
        'correct_gradient': correct, 'implemented_gradient': implemented,
        'objective_before': objective(logit),
        'objective_after_implemented_step': objective(logit-lr*implemented),
        'objective_after_correct_step': objective(logit-lr*correct),
        'source': 'train/exact.py:30-33',
    }

    first = [[10.0], [0.0]]
    second = list(reversed(first))
    equal_first = namespace['equal_depth_allocation'](first, 1)
    equal_second = namespace['equal_depth_allocation'](second, 1)
    assert equal_first[0] != equal_second[0]
    rolling_a = namespace['rolling_window_dp_allocation']([[1.0], [100.0], [2.0], [3.0]], 2, 2)
    rolling_b = namespace['rolling_window_dp_allocation']([[1.0], [2.0], [100.0], [3.0]], 2, 2)
    assert rolling_a[0] != rolling_b[0]
    out['offline_order_dependence'] = {
        'equal_depth_gain_original': equal_first[0], 'equal_depth_gain_reversed': equal_second[0],
        'rolling_dp_gain_original': rolling_a[0], 'rolling_dp_gain_reordered': rolling_b[0],
        'source': 'experiments/e07_replay.py:579-581,612-613',
    }

    zero_opt = namespace['small_window_dp_gap'](
        {'requests': [{}, {}], 'extra_rounds': 1}, [[-1.0], [-1.0]], [[1.0], [1.0]],
        [1.5], 2, 1, 0)
    assert zero_opt['per_tier'][0]['mean_rel_gap_vs_exact_dp']['single_step_index'] == 0
    out['zero_oracle_gap_hidden'] = {
        'true_gains': [[-1.0], [-1.0]], 'predicted_gains': [[1.0], [1.0]],
        'budget': 1, 'oracle_value': 0.0, 'heuristic_value': -1.0,
        'true_absolute_regret': 1.0, 'reported': zero_opt,
        'source': 'experiments/e07_replay.py:539',
    }

    rng = np.random.default_rng(20261025)
    checked = 0
    max_error = 0.0
    for _ in range(100):
        n, h = int(rng.integers(1,6)), int(rng.integers(1,5))
        deltas = rng.normal(size=(n,h)).tolist()
        budget = int(rng.integers(0,n*h+1))
        reference, _ = e07.exact_dp(deltas, budget)
        value, alloc = prefix_knapsack(deltas, budget)
        achieved = sum(sum(row[:k]) for row,k in zip(deltas,alloc))
        assert sum(alloc) <= budget
        error = max(abs(reference-value), abs(achieved-value))
        assert error < 1e-9
        max_error = max(max_error,error)
        checked += 1
    large = rng.normal(size=(64,3)).tolist()
    start = time.perf_counter()
    large_value, large_allocation = prefix_knapsack(large,192)
    elapsed = time.perf_counter()-start
    out['full_window_dp_is_tractable'] = {
        'random_small_cases_vs_repository_exact_dp': checked, 'maximum_error': max_error,
        'large_n': 64, 'large_h': 3, 'large_budget': 192,
        'runtime_seconds_this_cpu': elapsed, 'value': large_value,
        'used_extra_rounds': sum(large_allocation),
        'source': 'experiments/e07_replay.py:283-290,508-552',
    }

    soft = e06.soft_transport(np.array([[0.0,100.0]]), np.array([2,2]), eps=1.0, iters=1000, k=2)
    hard = e06.solve_additive_assignment(np.array([[0.0,100.0]]), np.array([2,2]), k=2)
    assert soft['P'].max() > 1.9
    assert soft['feasible'] and hard['feasible']
    out['soft_transport_not_top2_feasible'] = {
        'P': soft['P'].tolist(), 'feasible_flag': soft['feasible'],
        'row_residual': soft['max_row_residual'],
        'edge_upper_bound_violation': soft['max_upper_bound_violation'],
        'soft_linear_objective': soft['objective_linear'],
        'distinct_top2_exact_objective': hard['objective'],
        'note': 'Helper exposes the violation but run_e06 drops it from summary; hard flow and rounding remain capacity safe.',
    }

    # Distinct Top-2 structural infeasibility despite enough total capacity.
    allowed = np.array([[True,False]])
    impossible_soft = e06.soft_transport(np.array([[0.0,1.0]]),np.array([2,2]),1.0,20,allowed=allowed,k=2)
    impossible_hard = e06.solve_additive_assignment(np.array([[0.0,1.0]]),np.array([2,2]),allowed=allowed,k=2)
    assert impossible_soft['feasible'] and not impossible_hard['feasible']
    out['soft_feasible_flag_ignores_structural_infeasibility'] = {
        'soft_feasible': impossible_soft['feasible'], 'hard_feasible': impossible_hard['feasible'],
        'required_distinct_experts': 2, 'allowed_distinct_experts': 1,
    }

    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
