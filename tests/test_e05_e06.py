"""Unit tests for experiments/e05_e06_synthetic.py (E05/E06 synthetic parts)."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "e05_e06_synthetic.py"
_spec = importlib.util.spec_from_file_location("e05_e06_synthetic", _MODULE_PATH)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


class TestE05PotentialIdentity(unittest.TestCase):
    def setUp(self):
        self.game = M.make_congestion_game(n_players=6, n_experts=4, capacity_factor=1.25, seed=3)

    def test_single_strict_improvement_delta_equals_potential_delta(self):
        rng = np.random.default_rng(0)
        checked = 0
        for _ in range(50):
            actions = M.random_feasible_assignment(self.game, seed=int(rng.integers(1 << 30)))
            loads = M.loads_of(self.game, actions)
            for i in range(self.game.n_players):
                new_a, new_u, cur_u = M.best_response(self.game, actions, i, loads)
                if new_a != actions[i]:
                    phi_before = M.potential(self.game, actions)
                    old = actions[i]
                    actions[i] = new_a
                    phi_after = M.potential(self.game, actions)
                    self.assertAlmostEqual(phi_after - phi_before, new_u - cur_u, places=9)
                    actions[i] = old
                    checked += 1
        self.assertGreater(checked, 0)

    def test_async_improvement_delta_error_and_monotone_potential(self):
        init = M.random_feasible_assignment(self.game, seed=1)
        res = M.async_improvement(self.game, init, seed=2, max_sweeps=32)
        self.assertLess(res["max_abs_utility_potential_delta_err"], 1e-9)
        trace = res["potential_trace"]
        for a, b in zip(trace, trace[1:]):
            self.assertGreater(b, a - 1e-12)
        self.assertTrue(res["converged"])
        self.assertTrue(M.is_pure_equilibrium(self.game, res["actions"]))
        self.assertTrue(M.is_feasible(self.game, res["actions"]))


class TestE05Enumeration(unittest.TestCase):
    def setUp(self):
        self.game = M.make_congestion_game(n_players=4, n_experts=3, capacity_factor=1.0, seed=5)
        self.enum = M.enumerate_game_outcomes(self.game)

    def test_enumeration_finds_equilibria(self):
        self.assertGreaterEqual(self.enum["n_equilibria"], 1)

    def test_potential_optimum_is_equilibrium(self):
        self.assertTrue(self.enum["potential_optimum_is_equilibrium"])

    def test_social_optimum_dominates_every_equilibrium(self):
        self.assertGreaterEqual(
            self.enum["social_optimum_social_welfare"] + 1e-9,
            self.enum["equilibrium_social_welfare_max"],
        )

    def test_async_result_is_enumerated_equilibrium(self):
        init = M.random_feasible_assignment(self.game, seed=7)
        res = M.async_improvement(self.game, init, seed=8, max_sweeps=32)
        self.assertTrue(M.is_pure_equilibrium(self.game, res["actions"]))
        self.assertLessEqual(
            res["social_welfare"], self.enum["social_optimum_social_welfare"] + 1e-9
        )


class TestE05OscillationGate(unittest.TestCase):
    def test_sync_cycles_and_async_converges(self):
        game = M.oscillation_example()
        sync = M.sync_best_response(game, [0, 0], max_sweeps=16)
        self.assertTrue(sync["cycle_detected"])
        self.assertEqual(sync["cycle_length"], 2)
        async_res = M.async_improvement(game, [0, 0], seed=0, max_sweeps=16)
        self.assertTrue(async_res["converged"])
        self.assertTrue(M.is_pure_equilibrium(game, async_res["actions"]))
        enum = M.enumerate_game_outcomes(game)
        self.assertEqual(enum["n_equilibria"], 2)


class TestE06MinCostFlow(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.costs = rng.uniform(0.0, 1.0, size=(6, 4))
        self.capacity = np.array([3, 3, 3, 3])

    def test_matches_brute_force(self):
        bf = M.brute_force_additive(self.costs, self.capacity)
        mf = M.solve_additive_assignment(self.costs, self.capacity)
        self.assertTrue(bf["feasible"] and mf["feasible"])
        self.assertAlmostEqual(bf["objective"], mf["objective"], places=8)

    def test_hard_capacity_and_distinct_experts(self):
        mf = M.solve_additive_assignment(self.costs, self.capacity)
        violations = M.check_assignment(self.costs, self.capacity, mf["assignment"])
        self.assertEqual(violations, [])

    def test_infeasible_total_slots_detected(self):
        capacity = np.array([1, 1, 1, 1])  # 4 slots < 2*6 required
        mf = M.solve_additive_assignment(self.costs, capacity)
        self.assertFalse(mf["feasible"])
        self.assertLess(mf["flow"], mf["required_flow"])

    def test_structural_infeasibility_detected(self):
        # total slots suffice but every token only has the same 2 candidates
        allowed = np.zeros((6, 4), dtype=bool)
        allowed[:, 0] = True
        allowed[:, 1] = True
        mf = M.solve_additive_assignment(self.costs, np.array([5, 6, 0, 0]), allowed)
        self.assertFalse(mf["feasible"])

    def test_candidate_restriction_respected(self):
        rng = np.random.default_rng(13)
        allowed = np.zeros((6, 4), dtype=bool)
        for i in range(6):
            allowed[i, rng.choice(4, size=3, replace=False)] = True
        mf = M.solve_additive_assignment(self.costs, np.array([4, 4, 4, 4]), allowed)
        self.assertTrue(mf["feasible"])
        violations = M.check_assignment(self.costs, np.array([4, 4, 4, 4]), mf["assignment"], allowed)
        self.assertEqual(violations, [])


class TestE06SetVersion(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.n, self.e = 6, 4
        self.capacity = np.array([3, 3, 3, 3])
        import itertools

        self.candidates = [list(itertools.combinations(range(self.e), 2)) for _ in range(self.n)]
        self.set_costs = [list(rng.uniform(0.0, 2.0, size=len(c))) for c in self.candidates]

    def test_enumeration_is_optimal_and_feasible(self):
        res = M.enumerate_set_assignment(self.set_costs, self.candidates, self.capacity)
        self.assertTrue(res["feasible"])
        self.assertTrue(res["optimal"])
        violations = M.check_assignment(
            np.zeros((self.n, self.e)), self.capacity, res["assignment"]
        )
        self.assertEqual(violations, [])
        # cross-check against unconstrained-product brute force
        import itertools

        best = np.inf
        for combo in itertools.product(*[range(len(c)) for c in self.candidates]):
            loads = np.zeros(self.e, dtype=int)
            cost = 0.0
            for i, j in enumerate(combo):
                for e in self.candidates[i][j]:
                    loads[e] += 1
                cost += self.set_costs[i][j]
            if np.all(loads <= self.capacity):
                best = min(best, cost)
        self.assertAlmostEqual(res["objective"], best, places=9)

    def test_additive_fit_recovers_truly_additive_costs(self):
        rng = np.random.default_rng(19)
        d_true = rng.uniform(0.0, 1.0, size=(self.n, self.e))
        exact_costs = [
            [d_true[i, a] + d_true[i, b] for a, b in self.candidates[i]] for i in range(self.n)
        ]
        d_fit, resid = M.fit_additive_per_token(exact_costs, self.candidates, self.e)
        self.assertLess(max(resid["max_abs"]), 1e-8)
        # pair costs only pin down sums; recoverability is checked via residuals
        for i in range(self.n):
            for a, b in self.candidates[i][:3]:
                self.assertAlmostEqual(d_fit[i, a] + d_fit[i, b], d_true[i, a] + d_true[i, b], places=7)


class TestE06SoftAndRounding(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(23)
        self.costs = rng.uniform(0.0, 1.0, size=(8, 6))
        self.capacity = np.array([3, 3, 3, 3, 3, 3])

    def test_soft_transport_residuals_shrink_with_iterations(self):
        std = float(np.std(self.costs))
        soft5 = M.soft_transport(self.costs, self.capacity, 0.1 * std, 5)
        soft20 = M.soft_transport(self.costs, self.capacity, 0.1 * std, 20)
        self.assertLessEqual(
            soft20["max_row_residual"], soft5["max_row_residual"] + 1e-12
        )
        self.assertLess(soft20["max_row_residual"], 0.2)
        self.assertLess(soft20["max_col_excess"], 0.2)
        # P <= 1 is not enforced by the two-margin soft solver; the violation
        # is reported as a residual and repaired by rounding, not asserted away.
        self.assertGreaterEqual(soft20["max_upper_bound_violation"], 0.0)
        self.assertTrue(np.isfinite(soft20["max_upper_bound_violation"]))

    def test_rounding_is_feasible_and_not_better_than_exact(self):
        soft = M.soft_transport(self.costs, self.capacity, 0.05 * float(np.std(self.costs)), 20)
        rounded = M.round_fractional(soft["P"], self.costs, self.capacity)
        self.assertTrue(rounded["feasible"])
        self.assertFalse(rounded["fallback_exact"])
        violations = M.check_assignment(self.costs, self.capacity, rounded["assignment"])
        self.assertEqual(violations, [])
        exact = M.solve_additive_assignment(self.costs, self.capacity)
        self.assertGreaterEqual(rounded["objective"], exact["objective"] - 1e-8)
        self.assertGreaterEqual(rounded["objective_increase_over_fractional"], -1e-8)

    def test_rounding_is_identity_on_integer_solution(self):
        exact = M.solve_additive_assignment(self.costs, self.capacity)
        P = np.zeros_like(self.costs)
        for i, experts in enumerate(exact["assignment"]):
            for e in experts:
                P[i, e] = 1.0
        rounded = M.round_fractional(P, self.costs, self.capacity)
        self.assertTrue(rounded["feasible"])
        self.assertEqual(rounded["assignment"], exact["assignment"])
        self.assertAlmostEqual(
            rounded["objective_increase_over_fractional"], 0.0, places=9
        )

    def test_rounding_reports_infeasible_instance(self):
        P = np.full((6, 4), 0.5)
        capacity = np.array([1, 1, 1, 1])
        rounded = M.round_fractional(P, self.costs[:6, :4], capacity)
        self.assertFalse(rounded["feasible"])
        self.assertIn("queue", rounded["reason"])


class TestE06CausalSanity(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(29)
        self.costs = rng.uniform(0.0, 1.0, size=(8, 6))
        self.suffix = rng.uniform(0.0, 1.0, size=(8, 6))
        self.capacity = np.array([3, 3, 3, 3, 3, 3])

    def test_suffix_invariance_passes(self):
        out = M.causal_sanity(self.costs, self.capacity, self.suffix)
        self.assertTrue(out["suffix_invariance_passed"])

    def test_batch_dependence_is_recorded(self):
        out = M.causal_sanity(self.costs, self.capacity, self.suffix)
        frac = out["batch_dependence"]["remove_one_request_changed_fraction"]
        self.assertGreaterEqual(frac, 0.0)
        self.assertLessEqual(frac, 1.0)

    def test_non_causal_diagnostic_is_labeled_non_deployable(self):
        out = M.causal_sanity(self.costs, self.capacity, self.suffix)
        diag = out["non_causal_diagnostic"]
        self.assertFalse(diag["deployable"])
        self.assertIn("non_causal", diag["label"])


class TestCLI(unittest.TestCase):
    def test_cli_writes_real_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "e05_e06.json"
            import subprocess
            import sys

            proc = subprocess.run(
                [sys.executable, str(_MODULE_PATH), "--output", str(out), "--seed", "0"],
                capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(out.read_text())
            self.assertEqual(data["real_lm_serving_part"], "not_executed")
            e05 = data["e05"]
            self.assertLess(
                e05["small_enumerated"]["max_abs_utility_potential_delta_err"], 1e-9
            )
            self.assertTrue(e05["oscillation_gate"]["sync"]["cycle_detected"])
            e06 = data["e06"]
            self.assertTrue(e06["additive_exact"]["feasible"])
            self.assertEqual(e06["additive_exact"]["assignment_violations"], [])
            self.assertTrue(e06["tiny_crosscheck"]["match"])
            self.assertTrue(e06["causal_sanity"]["suffix_invariance_passed"])
            self.assertTrue(all(r["rounded_feasible"] for r in e06["soft_and_rounding"]))


if __name__ == "__main__":
    unittest.main()
