"""Unit tests for data.controlled (known non-commutative finite-state control environment).

Covers: operator algebra, enumerable optimal paths (1-step vs 2-step with
complementary actions), STOP/budget semantics matching the main project
(plan sec. 3.4/4.2, P0b), observation legality, and the FP64 exact
objective / G=2 RLOO expected-gradient checks of plan sec. 7.
"""

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import controlled as C

W = C.reference_weights()

# Exact expected objectives E[F] at reference_weights(), price = 0.05,
# computed by full tree enumeration (verified against finite differences in
# TestExactMath); hardcoded as regression constants.
EXPECTED_OBJECTIVES = {
    "one_step_exact": 0.2736880476923995,
    "one_step_near": 0.2661440190801576,
    "two_step_complementary": 0.6523754647924489,
    "two_step_complementary_2": 0.6473233940633042,
    "budget_one": 0.7853821776163419,
    "budget_two": 0.6664486785726119,
}
EXPECTED_BATCH_OBJECTIVE = 0.548560296969544

EXPECTED_OPTIMA = {
    "one_step_exact": (0,),
    "one_step_near": (0,),
    "two_step_complementary": (1, 3),
    "two_step_complementary_2": (3, 2),
    "budget_one": (3,),
    "budget_two": (1, 3),
}


def instance_by_name(name):
    return next(i for i in C.DEFAULT_INSTANCES if i.name == name)


class TestOperators(unittest.TestCase):
    def test_all_ordered_pairs_noncommuting(self):
        for i, j in itertools.permutations(range(C.NUM_EXPERTS), 2):
            for z in (0, 1, 13, 50, 96):
                lhs = C.apply_operator(i, C.apply_operator(j, z))
                rhs = C.apply_operator(j, C.apply_operator(i, z))
                self.assertNotEqual(lhs, rhs, msg=f"operators {i},{j} commute at z={z}")

    def test_apply_matches_definition(self):
        for e, (a, b) in enumerate(C.OPERATORS):
            for z in (0, 1, 48, 96):
                self.assertEqual(C.apply_operator(e, z), (a * z + b) % C.MODULUS)

    def test_circular_distance(self):
        self.assertEqual(C.circular_distance(0, 0), 0)
        self.assertEqual(C.circular_distance(10, 58), 48)  # diameter
        self.assertEqual(C.circular_distance(96, 1), 2)  # wraps around
        self.assertEqual(C.circular_distance(65, 10), C.circular_distance(10, 65))

    def test_terminal_loss_bounds_and_zero(self):
        self.assertEqual(C.terminal_loss(65, 65), 0.0)
        self.assertEqual(C.terminal_loss(10, 58), 1.0)  # (48/48)^2
        for u, v in itertools.product(range(0, 97, 12), repeat=2):
            self.assertTrue(0.0 <= C.terminal_loss(u, v) <= 1.0)


class TestPathEnumeration(unittest.TestCase):
    def test_terminal_path_counts(self):
        self.assertEqual(len(C.all_paths(1)), 4)
        self.assertEqual(len(C.all_paths(2)), 4 + 16)
        self.assertEqual(len(C.all_paths(3)), 4 + 16 + 64)  # 84, the P0b tree

    def test_at_least_one_call(self):
        # No empty path exists: every terminal trajectory calls >= 1 expert.
        for path in C.all_paths(3):
            self.assertGreaterEqual(len(path), 1)

    def test_path_objective_definition(self):
        inst = instance_by_name("two_step_complementary")
        # Optimal composition (1, 3): z = 10 -> 20 -> 65, exact hit.
        self.assertEqual(C.run_path(inst.z0, (1, 3)), (10, 20, 65))
        self.assertEqual(C.path_objective(inst, (1, 3)), 0.0 + 2 * C.DEFAULT_PRICE)

    def test_path_objective_enforces_budget(self):
        inst = instance_by_name("budget_one")
        with self.assertRaises(ValueError):
            C.path_objective(inst, (1, 3))  # 2 calls under B = 1
        with self.assertRaises(ValueError):
            C.path_objective(inst, ())  # zero calls are never legal


class TestOptimalPaths(unittest.TestCase):
    def test_documented_optima_are_unique_and_exact(self):
        for inst in C.DEFAULT_INSTANCES:
            evaluations = C.enumerate_objectives(inst)
            best = C.optimal_path(inst)
            self.assertEqual(best.path, EXPECTED_OPTIMA[inst.name], msg=inst.name)
            runner_up = min(ev.objective for ev in evaluations if ev.path != best.path)
            self.assertLess(best.objective, runner_up, msg=f"{inst.name} optimum not unique")

    def test_some_instances_one_step_optimal_some_two_step(self):
        lengths = {inst.name: len(C.optimal_path(inst).path) for inst in C.DEFAULT_INSTANCES}
        self.assertEqual(lengths["one_step_exact"], 1)
        self.assertEqual(lengths["one_step_near"], 1)
        self.assertEqual(lengths["two_step_complementary"], 2)
        self.assertEqual(lengths["two_step_complementary_2"], 2)

    def test_one_step_near_beats_exact_two_step(self):
        # Reaching the target exactly with 2 calls costs more than stopping
        # after 1 call one unit away: the budget is not spent for its own sake.
        inst = instance_by_name("one_step_near")
        best = C.optimal_path(inst)
        self.assertEqual(best.path, (0,))
        self.assertAlmostEqual(best.loss, 1.0 / C.DIAMETER**2)
        exact_two_step = C.path_objective(inst, (0, 0))
        self.assertEqual(C.run_path(inst.z0, (0, 0))[-1], inst.target)
        self.assertLess(best.objective, exact_two_step)

    def test_complementary_actions_look_worse_after_one_step(self):
        # Plan sec. 4.3 counterexample as data: the first call of the optimal
        # composition strictly increases the distance to the target — past the
        # initial state and past the greedy one-step call — and only the
        # second call reaches the target.
        for name, d_initial, d_mid, d_greedy in (
            ("two_step_complementary", 42, 45, 30),
            ("two_step_complementary_2", 31, 39, 32),
        ):
            inst = instance_by_name(name)
            best = C.optimal_path(inst)
            states = C.run_path(inst.z0, best.path)
            self.assertEqual(C.circular_distance(inst.z0, inst.target), d_initial)
            self.assertEqual(C.circular_distance(states[1], inst.target), d_mid)
            self.assertEqual(C.circular_distance(states[2], inst.target), 0)
            greedy = min(C.circular_distance(C.apply_operator(e, inst.z0), inst.target) for e in range(4))
            self.assertEqual(greedy, d_greedy)
            self.assertGreater(d_mid, d_initial)
            self.assertGreater(d_mid, greedy)

    def test_myopic_greedy_is_suboptimal(self):
        for name in ("two_step_complementary", "two_step_complementary_2"):
            inst = instance_by_name(name)
            one_step = [ev for ev in C.enumerate_objectives(inst) if len(ev.path) == 1]
            greedy = min(one_step, key=lambda ev: ev.objective)
            self.assertLess(C.optimal_path(inst).objective, greedy.objective)

    def test_budget_monotonicity(self):
        # Plan sec. 4.4: larger budget can only help (same task, B = 1/2/3).
        j1 = C.optimal_path(instance_by_name("budget_one")).objective
        j2 = C.optimal_path(instance_by_name("budget_two")).objective
        j3 = C.optimal_path(instance_by_name("two_step_complementary")).objective
        self.assertGreaterEqual(j1, j2)
        self.assertGreaterEqual(j2, j3)
        self.assertEqual(j2, j3)  # B = 2 already affords the optimal 2-step path


class TestRolloutSemantics(unittest.TestCase):
    def setUp(self):
        self.env = C.ControlledEnvironment()

    def test_first_call_forced_continue(self):
        state = self.env.reset(0)
        mask = self.env.legal_decisions(state)
        self.assertTrue(mask.continue_forced)
        self.assertFalse(mask.stop_allowed)
        self.assertEqual(mask.experts, (0, 1, 2, 3))
        with self.assertRaises(ValueError):
            self.env.step(state, C.STOP)

    def test_forced_exit_at_limit(self):
        # budget_one has limit 1: after the first call the exit is forced.
        state = self.env.reset(4)
        state = self.env.step(state, 3)
        self.assertTrue(state.done)
        with self.assertRaises(ValueError):
            self.env.step(state, 0)
        with self.assertRaises(ValueError):
            self.env.step(state, C.STOP)

    def test_stochastic_stop_after_one_call_under_budget(self):
        state = self.env.reset(0)  # one_step_exact, B = 3
        state = self.env.step(state, 0)
        self.assertFalse(state.done)
        mask = self.env.legal_decisions(state)
        self.assertTrue(mask.stop_allowed)
        self.assertFalse(mask.stop_forced)
        self.assertFalse(mask.continue_forced)
        state = self.env.step(state, C.STOP)
        self.assertTrue(state.done)
        # F(tau) = loss + price * calls with exactly one call, target hit.
        self.assertEqual(self.env.objective(state), 0.0 + self.env.price)

    def test_budget_two_forces_exit_at_two_calls(self):
        state = self.env.reset(5)  # budget_two
        state = self.env.step(state, 1)
        state = self.env.step(state, 3)
        self.assertTrue(state.done)
        self.assertEqual(state.z, instance_by_name("budget_two").target)
        self.assertEqual(self.env.objective(state), 0.0 + 2 * self.env.price)

    def test_rollout_matches_enumerated_path(self):
        index = 2  # two_step_complementary, B = 3: STOP ends the 2-step path
        state = self.env.reset(index)
        for decision in (1, 3, C.STOP):
            state = self.env.step(state, decision)
        self.assertTrue(state.done)
        enumerated = {ev.path: ev for ev in self.env.enumerate_objectives(index)}
        self.assertEqual(self.env.objective(state), enumerated[(1, 3)].objective)
        self.assertEqual(self.env.objective(state), self.env.optimal(index).objective)

    def test_full_rollout_with_reference_policy_terminates(self):
        # Sample episodes deterministically (argmax) under the linear
        # factorized policy; every episode ends within the limit.
        for index in range(len(self.env)):
            state = self.env.reset(index)
            while not state.done:
                obs = self.env.observe(state)
                mask = self.env.legal_decisions(state)
                p_stop, p_experts = C.action_probabilities(W, obs)
                if mask.continue_forced:
                    decision = int(np.argmax(p_experts))
                elif mask.stop_forced:
                    decision = C.STOP
                else:
                    decision = C.STOP if p_stop > 0.5 else int(np.argmax(p_experts))
                state = self.env.step(state, decision)
            self.assertLessEqual(state.t, self.env.limit(state.instance))
            self.assertGreaterEqual(state.t, 1)
            self.assertTrue(np.isfinite(self.env.objective(state)))


class TestObservation(unittest.TestCase):
    def setUp(self):
        self.env = C.ControlledEnvironment()

    def test_shape_dtype_and_layout(self):
        state = self.env.reset(0)
        obs = self.env.observe(state)
        self.assertEqual(obs.shape, (C.OBSERVATION_DIM,))
        self.assertEqual(obs.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertEqual(obs[11], 1.0)  # bias
        self.assertEqual(obs[3], 0.0)  # round t = 0
        self.assertEqual(obs[4], 1.0)  # remaining budget 3/3
        self.assertEqual(obs[5], 1.0)  # requested budget 3/3
        self.assertEqual(obs[6], self.env.price)
        self.assertTrue(np.all(obs[7:11] == 0.0))  # no previous expert yet

    def test_previous_expert_onehot_after_first_call(self):
        state = self.env.step(self.env.reset(0), 2)
        obs = self.env.observe(state)
        self.assertEqual(obs[7 + 2], 1.0)
        self.assertEqual(obs[7:11].sum(), 1.0)

    def test_observation_contains_only_input_and_prefix_state(self):
        # Legality (plan sec. 3.3): the observation is a function of the task
        # input (z0, target, budget) and the executed prefix (z, t, previous
        # expert). Two instances that differ only in target yield observations
        # differing only at the target entry; nothing in the vector is the
        # loss value, the optimal action, or a future state.
        a = self.env.observe(C.EnvState(C.Instance("a", 10, 65, 3), 20, 1, 1, False))
        b = self.env.observe(C.EnvState(C.Instance("b", 10, 66, 3), 20, 1, 1, False))
        differing = np.flatnonzero(a != b)
        self.assertEqual(list(differing), [2])
        state = self.env.step(self.env.reset(2), 1)
        obs = self.env.observe(state)
        loss = self.env.terminal_loss(state)
        self.assertNotIn(loss, set(obs))  # the true loss is never a feature


class TestExactMath(unittest.TestCase):
    """FP64 exact-tree checks (plan sec. 7; implementation gate < 1e-7)."""

    def test_probability_mass_sums_to_one(self):
        for inst in C.DEFAULT_INSTANCES:
            tree = C.enumerate_tree(inst, W)
            mass = sum(leaf.probability for leaf in tree)
            self.assertAlmostEqual(mass, 1.0, places=12, msg=inst.name)
            self.assertEqual(len(tree), len(C.all_paths(C.path_limit(inst.budget))))

    def test_exact_objectives_match_reference_constants(self):
        for inst in C.DEFAULT_INSTANCES:
            objective = C.exact_objective(C.enumerate_tree(inst, W))
            self.assertAlmostEqual(
                objective, EXPECTED_OBJECTIVES[inst.name], places=14, msg=inst.name
            )

    def test_batch_objective_constant(self):
        objective, gradient = C.batch_exact_objective_and_gradient(C.ControlledEnvironment(), W)
        self.assertAlmostEqual(objective, EXPECTED_BATCH_OBJECTIVE, places=14)
        self.assertEqual(gradient.shape, W.shape)

    def test_score_function_gradient_vs_finite_difference(self):
        # Plan sec. 8.5 gate: FP64 small-tree gradient error < 1e-7.
        for name in ("two_step_complementary", "budget_one"):
            inst = instance_by_name(name)
            tree = C.enumerate_tree(inst, W)
            grad = C.exact_gradient(tree)
            fd = C.finite_difference_gradient(inst, W)
            self.assertLess(np.max(np.abs(grad - fd)), 1e-7, msg=name)

    def test_rloo_g2_expected_gradient_equals_exact_gradient(self):
        # Plan eq. (11): exact expectation over trajectory pairs, not Monte Carlo.
        for inst in C.DEFAULT_INSTANCES:
            tree = C.enumerate_tree(inst, W)
            grad = C.exact_gradient(tree)
            rloo = C.rloo2_expected_gradient(tree)
            self.assertLess(np.max(np.abs(rloo - grad)), 1e-10, msg=inst.name)

    def test_self_mean_baseline_shrinks_gradient_by_half(self):
        # Including the sample in its own baseline gives (G-1)/G = 1/2 of the
        # gradient (plan sec. 5.2) — it must not be called the unbiased one.
        inst = instance_by_name("two_step_complementary")
        tree = C.enumerate_tree(inst, W)
        grad = C.exact_gradient(tree)
        self_mean = C.self_mean_gradient(tree)
        self.assertLess(np.max(np.abs(self_mean - 0.5 * grad)), 1e-10)

    def test_mean_score_is_zero(self):
        for inst in C.DEFAULT_INSTANCES:
            tree = C.enumerate_tree(inst, W)
            self.assertLess(np.max(np.abs(C.mean_score(tree))), 1e-12, msg=inst.name)

    def test_forced_decisions_carry_no_score(self):
        # budget_one: the single decision is the forced first CONTINUE, so the
        # score contains no STOP-logit row; the forced exit adds nothing.
        tree = C.enumerate_tree(instance_by_name("budget_one"), W)
        self.assertEqual(len(tree), 4)
        for leaf in tree:
            self.assertEqual(leaf.path and len(leaf.path), 1)
            self.assertTrue(np.all(leaf.score[0] == 0.0))
            self.assertGreater(np.abs(leaf.score[1:]).sum(), 0.0)

    def test_tree_leaf_paths_cover_all_terminal_paths(self):
        inst = instance_by_name("two_step_complementary")
        tree = C.enumerate_tree(inst, W)
        self.assertEqual(sorted(leaf.path for leaf in tree), sorted(C.all_paths(3)))


if __name__ == "__main__":
    unittest.main()
