"""Reproduce only the E07 oracle and current NumPy E09 gradient audit.

Requirements: Python 3.10+, NumPy, this bundle's reproduce_synthetic_audit.py,
and the read-only repository snapshot including the uploaded JSON/JSONL files.
No Torch, model checkpoint, GPU, training, or full test suite is required.

Usage:
    python recheck_e07_e09.py --repo ../GB_RL_MoE_Loop --output e07_e09_recheck.json

E07 is an OFFLINE label-informed oracle for a known, separable window. Its
intermediate gains are the repository's endpoint-interpolation assumptions,
not measured T2/T3 losses, an online allocator, or a deployable value predictor.
E09 executes the CURRENT NumPy controlled environment. It does not validate
the separate legacy Torch train/exact.py implementation.

Original pure functions are AST-extracted unchanged where importing an entire
module would unnecessarily import Torch. The prefix-knapsack solver and finite
differences are independent audit computations, explicitly identified below.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import statistics

import numpy as np

from reproduce_synthetic_audit import load_file, prefix_knapsack


PINNED_COMMIT = "f2bef422f7315f30618f00c690bf82810d1e85ac"
FD_COORDINATES = ((0, 0), (0, 11), (1, 1), (2, 2), (3, 3), (4, 4))
FD_EPSILON = 1e-5


def extract_functions(path: Path, names: set[str], namespace: dict):
    """Execute original FunctionDef AST nodes, with no body modifications."""
    source = ast.parse(path.read_text())
    nodes = [node for node in source.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    found = {node.name for node in nodes}
    if found != names:
        raise ValueError(f"Missing requested original functions in {path}: {names - found}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def literal_assignment(path: Path, name: str):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise ValueError(f"No literal assignment {name} in {path}")


def recheck_e07(repo: Path):
    source = repo / "experiments/e07_replay.py"
    published = json.loads((repo / "runs/e07/m105_replay/e07_replay.json").read_text())
    e03 = json.loads((repo / "runs/e03/m105_100m/e03_summary.json").read_text())
    records_path = repo / "runs/e01/m105_100m/records_test.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    expected_hash = published["input"]["backbone_hash"]
    assert expected_hash == e03["backbone_hash"]
    assert all(row["backbone_hash"] == expected_hash and row["split"] == "test" for row in records)
    seed = int(published["seed"])
    window_size = int(published["input"]["window_size"])
    indices = sorted(random.Random(seed).sample(range(len(records)), window_size))
    assert indices == published["window_state_indices"], "Sampled window differs from uploaded E07"
    stop = [float(records[i]["immediate"][records[i]["baseline_action"]]) for i in indices]
    continued = [float(records[i]["q_pi0"][records[i]["baseline_action"]]) for i in indices]
    deltas = [a - b for a, b in zip(stop, continued)]
    extra_rounds = int(published["input"]["extra_rounds_per_request"])
    assert extra_rounds == int(published["input"]["rounds"]) - 1
    profiles = literal_assignment(source, "PROFILES")
    tiers = literal_assignment(source, "DEFAULT_BUDGET_ROUNDS")
    original = extract_functions(source, {"profile_weights", "trajectory"},
                                 {"PROFILE_RATIO": literal_assignment(source, "PROFILE_RATIO")})
    result = {
        "scope": "Offline label-informed oracle for a known separable window; rounded E01 endpoint labels and assumed interpolation, not real intermediate-depth labels or a deployable/online policy.",
        "n_states": len(records), "seed": seed, "window_indices": indices,
        "window_matches_uploaded_e07": True, "backbone_hash_matches_uploaded_e03": True,
        "stop_nll": statistics.fmean(stop), "T4_nll": statistics.fmean(continued),
        "positive_tail_gains": sum(delta > 0 for delta in deltas),
        "extra_rounds_per_request": extra_rounds,
        "method": {
            "original_code": "Unmodified AST functions profile_weights and trajectory from experiments/e07_replay.py",
            "audit_solver": "prefix_knapsack imported from sibling reproduce_synthetic_audit.py; independent O(N*B*H) multiple-choice prefix knapsack with optional stop",
            "assumptions": "Fixed, additive, per-request prefix gains; full window and gains known before allocation; no inter-request coupling, streaming credit constraints, or GPU latency model.",
            "rounded_label_precision": "The uploaded JSONL serializes losses to six decimal places; no missing .pt state tensor is reconstructed.",
        },
        "profiles": {},
    }
    for profile in profiles:
        rows = [original["trajectory"](delta, extra_rounds, profile) for delta in deltas]
        output_rows = []
        for avg_rounds in tiers:
            budget = min(int(round((avg_rounds - 1.0) * window_size)), window_size * extra_rounds)
            value, allocation = prefix_knapsack(rows, budget)
            achieved = sum(sum(row[:count]) for row, count in zip(rows, allocation))
            assert sum(allocation) <= budget and abs(value - achieved) < 1e-10
            assert all(0 <= count <= len(row) for count, row in zip(allocation, rows))
            output_rows.append({
                "budget_extra_rounds": budget, "oracle_gain": value,
                "oracle_nll": result["stop_nll"] - value / window_size,
                "used_extra_rounds": sum(allocation), "allocation": allocation,
            })
        result["profiles"][profile] = output_rows
    return result


def recheck_e09(repo: Path):
    controlled = load_file("audit_recheck_controlled", repo / "data/controlled.py")
    environment = controlled.ControlledEnvironment()
    weights = controlled.reference_weights()
    original = extract_functions(repo / "experiments/e09_controlled.py",
                                 {"_moments", "estimator_moments_for_tree"},
                                 {"np": np, "C": controlled})
    results = []
    max_rloo_error = 0.0
    max_fd_error = 0.0
    for instance in environment.instances:
        tree = controlled.enumerate_tree(instance, weights, environment.price)
        exact = controlled.exact_gradient(tree)
        rloo = controlled.rloo2_expected_gradient(tree)
        moments = original["estimator_moments_for_tree"](tree)
        rloo_error = float(np.max(np.abs(exact - rloo)))
        fd = []
        for row, column in FD_COORDINATES:
            plus, minus = weights.copy(), weights.copy()
            plus[row, column] += FD_EPSILON
            minus[row, column] -= FD_EPSILON
            # Independent central differences of the original expected objective;
            # this is not a copied score-gradient formula or a Torch trainer call.
            estimate = (
                controlled.exact_objective(controlled.enumerate_tree(instance, plus, environment.price))
                - controlled.exact_objective(controlled.enumerate_tree(instance, minus, environment.price))
            ) / (2.0 * FD_EPSILON)
            error = abs(estimate - float(exact[row, column]))
            max_fd_error = max(max_fd_error, error)
            fd.append({"coordinate": [row, column], "analytic": float(exact[row, column]),
                       "finite_difference": estimate, "absolute_error": error})
        mass = sum(leaf.probability for leaf in tree)
        mean_score_error = float(np.max(np.abs(controlled.mean_score(tree))))
        original_e09_bias = moments["rloo2_loo"]["bias_max_abs"]
        max_rloo_error = max(max_rloo_error, rloo_error, original_e09_bias)
        assert abs(mass - 1.0) < 1e-12
        assert len(tree) == len(controlled.all_paths(controlled.path_limit(instance.budget)))
        assert rloo_error < 1e-10 and original_e09_bias < 1e-10
        assert mean_score_error < 1e-12
        assert max(item["absolute_error"] for item in fd) < 1e-7
        results.append({
            "instance": instance.name, "budget": instance.budget,
            "terminal_paths": len(tree), "probability_mass": mass,
            "expected_objective": controlled.exact_objective(tree),
            "rloo_vs_exact_max_abs": rloo_error,
            "original_e09_rloo_moment_bias_max_abs": original_e09_bias,
            "mean_score_max_abs": mean_score_error, "finite_difference_coordinates": fd,
        })
    return {
        "scope": "Current NumPy controlled.py and original E09 pure estimator functions; NOT legacy Torch train/exact.py, neural LM RL, or a rerun of training.",
        "execution": "Direct import of data/controlled.py avoids unrelated data/__init__.py Torch imports; two original E09 function AST nodes are executed unchanged with this module bound as C.",
        "rloo_vs_exact_max_abs": max_rloo_error,
        "exact_vs_finite_difference_6_coordinates_max_abs": max_fd_error,
        "finite_difference_epsilon": FD_EPSILON,
        "finite_difference_coordinates": [list(coord) for coord in FD_COORDINATES],
        "terminal_paths_per_instance": [item["terminal_paths"] for item in results],
        "total_probability_per_instance": [item["probability_mass"] for item in results],
        "per_instance": results,
    }


def compare_previous(e07: dict, e09: dict):
    """Compare stable values, preserving the original audit JSON files."""
    audit_dir = Path(__file__).resolve().parent
    comparison = {}
    path = audit_dir / "e07_reconstructed_oracle.json"
    if path.exists():
        previous = json.loads(path.read_text())
        assert e07["window_indices"] == previous["window_indices"]
        assert e07["positive_tail_gains"] == previous["positive_tail_gains"]
        errors = [abs(e07[key] - previous[key]) for key in ("stop_nll", "T4_nll")]
        for profile, rows in previous["profiles"].items():
            for old, new in zip(rows, e07["profiles"][profile]):
                for key in ("budget_extra_rounds", "used_extra_rounds"):
                    assert old[key] == new[key]
                errors.extend(abs(old[key] - new[key]) for key in ("oracle_gain", "oracle_nll"))
        assert max(errors) < 1e-12
        comparison["e07"] = {"previous_file": path.name, "passed": True, "max_abs_numeric_difference": max(errors)}
    else:
        comparison["e07"] = {"previous_file_present": False}
    path = audit_dir / "e09_gradient_recheck.json"
    if path.exists():
        previous = json.loads(path.read_text())
        assert e09["terminal_paths_per_instance"] == previous["terminal_paths_per_instance"]
        mass_error = max(abs(a - b) for a, b in zip(e09["total_probability_per_instance"], previous["total_probability_per_instance"]))
        assert mass_error < 1e-12
        assert e09["rloo_vs_exact_max_abs"] < 1e-10 and previous["rloo_vs_exact_max_abs"] < 1e-10
        assert e09["exact_vs_finite_difference_6_coordinates_max_abs"] < 1e-7
        assert previous["exact_vs_finite_difference_6_coordinates_max_abs"] < 1e-7
        comparison["e09"] = {
            "previous_file": path.name, "passed": True,
            "probability_mass_max_abs_difference": mass_error,
            "previous_rloo_vs_exact_max_abs": previous["rloo_vs_exact_max_abs"],
            "previous_six_coordinate_fd_max_abs": previous["exact_vs_finite_difference_6_coordinates_max_abs"],
            "finite_difference_comparison": "The historical compact JSON did not record coordinate indices. This script fixes and publishes six coordinates; compare the <1e-7 gate, not bitwise equality of the old numerical-error scalar.",
        }
    else:
        comparison["e09"] = {"previous_file_present": False}
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1] / "GB_RL_MoE_Loop")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("e07_e09_recheck.json"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    e07, e09 = recheck_e07(repo), recheck_e09(repo)
    source_paths = ("experiments/e07_replay.py", "data/controlled.py", "experiments/e09_controlled.py",
                    "runs/e01/m105_100m/records_test.jsonl", "runs/e07/m105_replay/e07_replay.json",
                    "runs/e03/m105_100m/e03_summary.json")
    out = {
        "intended_repository_commit": PINNED_COMMIT,
        "source_sha256": {path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in source_paths},
        "e07": e07, "e09": e09,
        "comparison_to_prior_audit": compare_previous(e07, e09),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {args.output.resolve()}")
    print("E07: OFFLINE known separable window; assumed intermediate gains; no online/deployment or GPU claim.")
    for profile, rows in e07["profiles"].items():
        print(profile + ": " + ", ".join(f"B={row['budget_extra_rounds']} NLL={row['oracle_nll']:.12f}" for row in rows))
    print("E09: CURRENT NumPy controlled environment, NOT legacy Torch exact trainer; no training rerun.")
    print(f"RLOO/exact max error={e09['rloo_vs_exact_max_abs']:.12g}; six-coordinate FD max error={e09['exact_vs_finite_difference_6_coordinates_max_abs']:.12g}")
    print("All stated reproduction and comparison checks passed.")


if __name__ == "__main__":
    main()
