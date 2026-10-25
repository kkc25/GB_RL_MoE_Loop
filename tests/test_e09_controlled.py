"""Unit tests for experiments.e09_controlled (E09 synthetic policy-gradient comparison).

Covers: legality of the shared-policy rollouts, the supervised-signal
gradient (finite differences), exact estimator moments (unbiasedness and
variance ordering), the enumeration-calibrated Monte Carlo gradient check,
training-signal comparisons (all reduce the exact batch objective; supervised
recovers the designed optima), budget compliance, and the CLI JSON contract.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import controlled as C
from experiments import e09_controlled as e09

ENV = C.ControlledEnvironment()
W0 = C.reference_weights()

_BATCH_MOMENTS = None


def batch_moments():
    """Cached exact estimator moments at the reference weights."""
    global _BATCH_MOMENTS
    if _BATCH_MOMENTS is None:
        _BATCH_MOMENTS = e09.batch_estimator_variance(ENV, W0)
    return _BATCH_MOMENTS


def instance_index(name):
    return next(i for i, x in enumerate(ENV.instances) if x.name == name)


class TestSharedPolicyRollouts(unittest.TestCase):
    def test_sampled_rollout_is_legal_and_consistent(self):
        rng = np.random.default_rng(0)
        for index, instance in enumerate(ENV.instances):
            for _ in range(50):
                path, objective, score, calls = e09.sample_rollout(ENV, index, W0, rng)
                self.assertGreaterEqual(calls, 1)
                self.assertLessEqual(calls, ENV.limit(instance))
                self.assertEqual(len(path), calls)
                self.assertTrue(all(0 <= e < C.NUM_EXPERTS for e in path))
                self.assertEqual(score.shape, W0.shape)
                terminal = C.run_path(instance.z0, path)[-1]
                expected = C.terminal_loss(terminal, instance.target) + ENV.price * calls
                self.assertAlmostEqual(objective, expected, places=12)

    def test_sampled_objective_mean_matches_exact(self):
        # Enumeration-calibrated: 5 sigma of the sample mean, sigma from the tree.
        index = instance_index("two_step_complementary")
        tree = C.enumerate_tree(ENV.instances[index], W0, ENV.price)
        exact = C.exact_objective(tree)
        var = sum(l.probability * (l.objective - exact) ** 2 for l in tree)
        rng = np.random.default_rng(1)
        n = 3000
        mean = sum(e09.sample_rollout(ENV, index, W0, rng)[1] for _ in range(n)) / n
        self.assertLess(abs(mean - exact), 5.0 * np.sqrt(var / n))

    def test_greedy_rollout_terminates_within_budget(self):
        for index, instance in enumerate(ENV.instances):
            path, objective, calls = e09.greedy_rollout(ENV, index, W0)
            self.assertGreaterEqual(calls, 1)
            self.assertLessEqual(calls, ENV.limit(instance))
            self.assertTrue(np.isfinite(objective))


class TestSupervisedSignal(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        eps, tol = 1e-6, 1e-5
        loss0, grad = e09.supervised_loss_and_gradient(ENV, W0)
        self.assertGreater(loss0, 0.0)
        for index in np.ndindex(W0.shape):
            plus, minus = W0.copy(), W0.copy()
            plus[index] += eps
            minus[index] -= eps
            fd = (e09.supervised_loss_and_gradient(ENV, plus)[0]
                  - e09.supervised_loss_and_gradient(ENV, minus)[0]) / (2 * eps)
            self.assertAlmostEqual(grad[index], fd, delta=tol, msg=f"W{index}")

    def test_training_recovers_all_designed_optima(self):
        result = e09.train_supervised(
            ENV, W0, e09.SUPERVISED_STEPS, e09.SUPERVISED_LR, eval_every=50)
        init_objective = result["curve"][0]["exact_batch_objective"]
        self.assertLess(result["final_exact_batch_objective"], init_objective)
        table = e09.evaluate_policy(ENV, result["weights"], ENV.instances, "designed")
        self.assertEqual(table["n_recovered"], len(ENV.instances))
        for row in table["instances"].values():
            self.assertTrue(row["path_matches"])
        self.assertEqual(result["cost"]["sampled_rollouts"], 0)


class TestEstimatorMoments(unittest.TestCase):
    def test_unbiased_estimators(self):
        per_instance, _, _ = batch_moments()
        for name, rows in per_instance.items():
            for estimator in ("reinforce_b0", "reinforce_mean_baseline", "rloo2_loo"):
                self.assertLess(rows[estimator]["bias_max_abs"], 1e-10,
                                msg=f"{name}/{estimator}")

    def test_self_mean_baseline_is_biased(self):
        per_instance, _, _ = batch_moments()
        biases = [rows["self_mean_g2"]["bias_max_abs"] for rows in per_instance.values()]
        self.assertGreater(max(biases), 1e-3)

    def test_rloo2_variance_below_unbaselined_reinforce(self):
        _, batch, _ = batch_moments()
        self.assertLess(batch["rloo2_loo"]["total_variance"],
                        batch["reinforce_b0"]["total_variance"])
        self.assertLess(batch["self_mean_g2"]["total_variance"],
                        batch["rloo2_loo"]["total_variance"])
        for name, rows in batch.items():
            self.assertGreaterEqual(rows["total_variance"], 0.0, msg=name)

    def test_mc_check_against_exact_gradient(self):
        check = e09.monte_carlo_gradient_check(ENV, W0, samples=300, seed=7)
        self.assertTrue(check["passed"], msg=check)
        self.assertEqual(check["coordinates_over_tolerance"], 0)

    def test_rloo2_expectation_equals_exact_gradient(self):
        for instance in ENV.instances:
            tree = C.enumerate_tree(instance, W0, ENV.price)
            err = np.max(np.abs(C.rloo2_expected_gradient(tree) - C.exact_gradient(tree)))
            self.assertLess(err, 1e-12, msg=instance.name)


class TestTrainingSignals(unittest.TestCase):
    def test_exact_training_reduces_objective(self):
        result = e09.train_exact(ENV, W0, 120, e09.EXACT_LR, eval_every=40)
        self.assertLess(result["final_exact_batch_objective"],
                        result["curve"][0]["exact_batch_objective"])
        self.assertEqual(result["cost"]["enumerated_trajectories_per_update"], 360)

    def test_rloo2_training_reduces_objective(self):
        result = e09.train_rloo2(ENV, W0, 300, e09.RLOO_LR, seed=3,
                                 clip=e09.RLOO_CLIP, eval_every=50)
        self.assertLess(result["final_exact_batch_objective"],
                        result["curve"][0]["exact_batch_objective"])
        self.assertEqual(result["cost"]["sampled_rollouts"], 300 * 2 * len(ENV))

    def test_held_out_recovery_rates_are_valid(self):
        instances = e09.validation_instances(8, 11)
        self.assertEqual(len(instances), 8)
        val_env = C.ControlledEnvironment(instances, ENV.price)
        table = e09.evaluate_policy(val_env, W0, instances, "held_out")
        self.assertEqual(table["n_instances"], 8)
        self.assertTrue(0.0 <= table["recovery_rate"] <= 1.0)


class TestBudget(unittest.TestCase):
    def test_no_hard_violations_and_limits_respected(self):
        result = e09.train_supervised(ENV, W0, 20, e09.SUPERVISED_LR, eval_every=10)
        report = e09.budget_report(
            ENV, {"supervised": result["weights"], "initial": W0}, 100)
        self.assertEqual(report["hard_budget_violations"], 0)
        for method_rows in report["overspend"].values():
            for name, row in method_rows.items():
                instance = ENV.instances[instance_index(name)]
                self.assertLessEqual(row["expected_calls"], row["hard_limit"] + 1e-12)
                self.assertEqual(row["hard_limit"], ENV.limit(instance))
                self.assertGreaterEqual(row["expected_calls"], 1.0)


class TestCLI(unittest.TestCase):
    def test_output_json_written(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "e09.json")
            e09.main(["--output", out, "--quick", "--seed", "5"])
            with open(out) as f:
                r = json.load(f)
        self.assertEqual(r["experiment"], "E09_controlled")
        self.assertEqual(r["real_lm_rl"]["status"], "not_executed")
        self.assertIn("synthetic", r["scope"])
        for key in ("gradient_verification", "training", "validation",
                    "complementary_two_step", "budget", "findings"):
            self.assertIn(key, r)
        gv = r["gradient_verification"]
        self.assertLess(
            gv["rloo2_expectation_equals_exact_gradient_max_abs_err"], 1e-12)
        self.assertTrue(gv["monte_carlo_rloo2_mean_vs_exact"]["passed"])
        self.assertEqual(r["budget"]["hard_budget_violations"], 0)
        # All three training signals improve the exact batch objective.
        init = r["environment"]["init_exact_batch_objective"]
        tr = r["training"]
        self.assertLess(tr["supervised_oracle_imitation"]["final_exact_batch_objective"], init)
        self.assertLess(tr["exact_joint_gradient"]["final_exact_batch_objective"], init)
        self.assertLess(tr["rloo2_onpolicy"]["final_objective_mean"], init)
        # Complementary two-step cases are reported with their non-myopic structure.
        for name in ("two_step_complementary", "two_step_complementary_2"):
            case = r["complementary_two_step"][name]
            self.assertTrue(case["first_optimal_call_increases_distance"])
            self.assertTrue(
                case["methods"]["supervised_oracle_imitation"]["recovered"])
        # Validation covers both the designed set and held-out instances.
        for method in ("supervised_oracle_imitation", "exact_joint_gradient",
                       "rloo2_onpolicy_best_seed", "initial"):
            self.assertEqual(r["validation"][method]["designed"]["n_instances"], 6)
            self.assertGreater(r["validation"][method]["held_out"]["n_instances"], 0)

    def test_seed_reproducible(self):
        self.assertEqual(e09.run(seed=2, quick=True), e09.run(seed=2, quick=True))


if __name__ == "__main__":
    unittest.main()
