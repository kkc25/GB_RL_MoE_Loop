"""Unit tests for experiments/e08_ab.py (E08 A/B synthetic mechanism sub-experiments).

Standard-library unittest; run with:
    python -m unittest tests.test_e08_ab -v
from the repository root.  Uses quick/reduced configurations so the suite
finishes in a few minutes on CPU.
"""

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "e08_ab", REPO_ROOT / "experiments" / "e08_ab.py")
e08_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e08_ab)

LOG2 = math.log(2.0)


class TestA1SufficientState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = e08_ab.run_a1(seed=0, n_tables=3)

    def test_negative_control_passes(self):
        self.assertTrue(self.res["passed"], self.res["gates"])

    def test_history_cannot_beat_markov_dp(self):
        self.assertLessEqual(self.res["metrics"]["max_abs_root_diff_history_vs_dp"], 1e-9)
        self.assertLessEqual(
            self.res["metrics"]["max_state_value_spread_sufficient"], 1e-9)

    def test_checker_detects_omitted_state_variable(self):
        # Deliberately dropping the used-expert mask from the state key must
        # produce a detectable spread: the checker is not vacuously passing.
        self.assertGreater(
            self.res["metrics"]["max_spread_when_mask_omitted"], 1e-6)

    def test_all_conditions_enumerated(self):
        # 3 tables x 3 cost configs x 2 constraint variants
        self.assertEqual(self.res["metrics"]["n_condition_evaluations"], 18)


class TestA2InformationErasure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = e08_ab.run_a2(seed=0, n_train=8_000, n_test=4_000, steps=150)

    def test_passes(self):
        self.assertTrue(self.res["passed"], self.res["gates"])

    def test_h_only_at_chance(self):
        core = self.res["metrics"]["core_C1"]
        for zeta in ("0.0", "0.1", "0.25", "0.5"):
            self.assertGreaterEqual(core[zeta]["h"], 0.44, zeta)

    def test_history_recovers_up_to_label_noise(self):
        core = self.res["metrics"]["core_C1"]
        for zeta, z in (("0.0", 0.0), ("0.1", 0.1), ("0.25", 0.25)):
            self.assertLessEqual(core[zeta]["history"], z + 0.06, zeta)

    def test_ordered_history_beats_recent_expert_when_recent_is_erased(self):
        ext = self.res["metrics"]["extended_two_round"]["0.0"]
        self.assertGreaterEqual(ext["recent"], 0.45)
        self.assertLessEqual(ext["history"], 0.06)


class TestA3ParityReadability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = e08_ab.run_a3(seed=0, mlp_steps=800, lin_steps=400,
                                scaled_steps=2000)

    def test_passes(self):
        self.assertTrue(self.res["passed"], self.res["gates"])

    def test_history_features_make_parity_readable(self):
        m = self.res["metrics"]
        self.assertGreaterEqual(
            m["parity_8bit"]["test_acc"]["linear_h_plus_history"], 0.99)
        self.assertGreaterEqual(
            m["parity_12bit"]["test_acc"]["linear_h_plus_history"], 0.99)

    def test_linear_h_only_cannot_represent_parity(self):
        m = self.res["metrics"]
        self.assertLessEqual(m["parity_8bit"]["test_acc"]["linear_h_only"], 0.65)
        self.assertLessEqual(m["parity_12bit"]["test_acc"]["linear_h_only"], 0.65)

    def test_param_matched_big_mlp_at_least_as_many_params(self):
        m = self.res["metrics"]["parity_8bit"]["param_counts"]
        self.assertGreaterEqual(m["mlp_big_h_only_param_matched"],
                                m["mlp_h_plus_history"])


class TestB1DelayedReward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = e08_ab.run_b1(seed=0, n_envs=6, quick=True)

    def test_passes(self):
        self.assertTrue(self.res["passed"], self.res["gates"])

    def test_exact_c5_env_matches_verified_values(self):
        # verified_toy_calculations.json: greedy 6.1, optimal 10.0
        exact = self.res["metrics"]["exact_c5_env"]
        self.assertAlmostEqual(exact["one_step"], 6.1, places=9)
        self.assertAlmostEqual(exact["two_step"], 10.0, places=9)
        self.assertAlmostEqual(exact["dp"], 10.0, places=9)
        self.assertGreater(exact["regret_two_step_pruned"], 3.0)

    def test_dp_zero_regret_and_two_step_dominates(self):
        for cfg in ("uniform", "nonuniform"):
            rr = self.res["metrics"]["random_envs"][cfg]["mean_regret"]
            self.assertAlmostEqual(rr["dp"], 0.0, places=9)
            self.assertLessEqual(rr["two_step"], rr["one_step"] + 1e-9)

    def test_supervised_q_reports_calibration_and_samples(self):
        meta = self.res["metrics"]["supervised_q_training"]
        self.assertGreater(meta["n_samples"], 0)
        self.assertGreaterEqual(meta["calibration_max_bin_gap"], 0.0)
        self.assertIn("train_time_sec", meta)


class TestB2UnpredictableLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = e08_ab.run_b2(seed=0, n_train=8_000, n_test=4_000, steps=300)

    def test_passes(self):
        self.assertTrue(self.res["passed"], self.res["gates"])

    def test_no_super_bayes_at_zero_signal(self):
        row = self.res["metrics"]["curves_vs_signal"]["0.0"]
        self.assertGreaterEqual(row["held_out_controller_loss"], LOG2 - 0.02)
        self.assertAlmostEqual(row["label_visible_oracle_loss"],
                               -math.log(0.99), delta=0.01)

    def test_controller_never_beats_achievable_optimum(self):
        for s, row in self.res["metrics"]["curves_vs_signal"].items():
            self.assertGreaterEqual(
                row["held_out_controller_loss"],
                row["achievable_optimum_label_blind"] - 0.02, s)


class TestB3OrderedComposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = e08_ab.run_b3(seed=0, router_steps=2500, expert_steps=800,
                                n_x0_train=6, n_x0_test=10)

    def test_passes(self):
        self.assertTrue(self.res["passed"], self.res["gates"])

    def test_experts_actually_noncommuting(self):
        self.assertGreaterEqual(len(self.res["metrics"]["noncommuting_pairs"]), 1)

    def test_holdout_composition_with_exact_experts(self):
        p1 = self.res["metrics"]["phase1_exact_experts"]
        self.assertGreaterEqual(p1["rollout_acc_holdout_combos"], 0.90)
        self.assertGreaterEqual(p1["rollout_acc_train_combos"], 0.95)

    def test_component_coverage_preserved(self):
        # holdout removes complete paths only; every single op must remain
        # in training (component holdout != complete-path holdout)
        train_seqs, holdout = e08_ab._b3_pick_holdout(
            e08_ab._b3_all_sequences(4), seed=0)
        singles = {e for s in train_seqs for e in s}
        self.assertEqual(singles, {0, 1, 2, 3})
        self.assertTrue(all(len(s) == 4 for s in holdout))

    def test_extrapolation_reported_separately(self):
        extra = self.res["metrics"]["extrapolation_length_5_6_exact_experts"]
        self.assertIn("EXTRAPOLATION", extra["label"])
        self.assertEqual(set(extra["per_length"]), {5, 6})

    def test_fixed_task_forced_path_intervention(self):
        # Same input x0 and same target compose(s, x0); only the executed
        # expert order is forced.  This is the causal order intervention.
        interv = self.res["metrics"]["phase1_exact_experts"][
            "fixed_task_forced_path_intervention"]
        self.assertIn("fixed", interv["semantics"])
        # sanity: forcing the correct path must hit the original target
        self.assertAlmostEqual(interv["forced_correct_path_sanity_acc"], 1.0)
        # forcing a wrong order on the SAME task must break the answer
        self.assertLess(interv["forced_reversed_path_acc_vs_original_target"], 0.5)
        self.assertLess(interv["forced_random_path_acc_vs_original_target"], 0.5)
        self.assertGreater(interv["n_cases"], 0)

    def test_program_changed_controls_renamed_and_labeled(self):
        # The old reverse/random/repeat controls change BOTH the program and
        # its correct answer; the metric group name must say so.
        p1 = self.res["metrics"]["phase1_exact_experts"]
        self.assertNotIn("controls", p1)
        renamed = p1["controls_changed_program_and_target"]
        self.assertIn("program", renamed["note"])
        for key in ("reversed_paths", "repeated_same_expert", "random_equal_length_paths"):
            self.assertIn(key, renamed)


class TestCLI(unittest.TestCase):
    def test_cli_writes_json_with_required_fields(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "e08.json"
            proc = subprocess.run(
                [sys.executable, "experiments/e08_ab.py",
                 "--output", str(out), "--quick", "--subexp", "A1,B1"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=900)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(out.read_text())
            self.assertIn("overall_passed", data)
            self.assertTrue(data["overall_passed"])
            subs = data["subexperiments"]
            self.assertEqual(set(subs), {"A1", "B1"})
            for name, r in subs.items():
                self.assertIn("passed", r)
                self.assertIn("metrics", r)
                self.assertIn("gates", r)
                self.assertIn("environment_scope", r)
                self.assertTrue(r["passed"], f"{name} failed: {r['gates']}")


if __name__ == "__main__":
    unittest.main()
