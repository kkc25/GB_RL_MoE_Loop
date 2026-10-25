"""Unit tests for experiments.e08_cde (E08 sub-experiments C/D/E)."""

import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import e08_cde as X


class TestC1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = X.run_c1(seed=0)
        cls.n8 = [i for i in cls.r["instances"] if i["n_tokens"] == 8]

    def test_enumeration_size_and_dp_match(self):
        for inst in self.n8:
            enum = inst["enumeration"]
            self.assertEqual(enum["profiles"], 6561)
            if enum["best_feasible_value_no_dummy"] is not None:
                self.assertTrue(enum["dp_matches_enumeration"], inst["name"])

    def test_hard_capacity_fix_feasible_everywhere(self):
        self.assertTrue(self.r["gates"]["hard_capacity_fix_feasible_all_instances"])

    def test_oracle_dominates_feasible_methods(self):
        for inst in self.r["instances"]:
            for name, m in inst["methods"].items():
                if name == "soft_primal_dual":
                    entries = m.values()
                else:
                    entries = [m]
                for e in entries:
                    if e["feasible"]:
                        self.assertGreaterEqual(
                            inst["oracle_value"] + 1e-9, e["value"],
                            f"{inst['name']}:{name}")

    def test_async_strict_best_response_potential_monotone(self):
        for inst in self.r["instances"]:
            self.assertTrue(
                inst["methods"]["async_strict_best_response"]["potential_monotone"],
                inst["name"])

    def test_last_and_average_iterate_reported_separately(self):
        for inst in self.r["instances"]:
            soft = inst["methods"]["soft_primal_dual"]
            self.assertEqual(set(soft), {"1", "2", "4", "8", "32", "128"})
            for entry in soft.values():
                self.assertIn("last_iterate_soft_overload", entry)
                self.assertIn("average_iterate_soft_overload", entry)

    def test_insufficient_capacity_uses_priced_dummy(self):
        inst = next(i for i in self.n8 if "insufficient" in i["name"])
        self.assertLess(inst["total_real_capacity"], inst["n_tokens"])
        fixed = inst["methods"]["price_preference_hard_capacity_fix"]
        self.assertTrue(fixed["feasible"])
        self.assertGreater(fixed["stop_dummy_count"], 0)


class TestC2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = X.run_c2(seed=1, rounds=8)

    def test_all_cells_present(self):
        self.assertEqual(len(self.r["cells"]), 3 * 4 * 2)

    def test_hard_fix_feasible_and_gate(self):
        for cell in self.r["cells"]:
            self.assertEqual(cell["post_fix_overload_sum"], 0)
        self.assertTrue(self.r["gates"]["hard_capacity_fix_feasible_all_cells"])
        self.assertTrue(self.r["gates"]["controller_overhead_within_budget"])

    def test_delta_zero_top1_has_no_drift_effect(self):
        for cell in self.r["cells"]:
            if cell["delta"] == 0.0 and cell["method"] == "top1_hard_fix":
                ovs = cell["per_round_pre_fix_overload"]
                self.assertEqual(ovs, [ovs[0]] * len(ovs))

    def test_drift_changes_pre_fix_behavior(self):
        def cell(pattern, method, delta):
            return next(c for c in self.r["cells"] if c["pattern"] == pattern
                        and c["method"] == method and c["delta"] == delta)
        changed = any(
            cell(p, m, 0.0)["per_round_pre_fix_overload"]
            != cell(p, m, 1.0)["per_round_pre_fix_overload"]
            for p in X.C2_PATTERNS for m in ("top1_hard_fix", "price_hard_fix"))
        self.assertTrue(changed)


class TestD1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = X.run_d1()

    def test_c6_spectral_radii(self):
        c = self.r["checks"]
        self.assertAlmostEqual(c["spectral_radius_A"], 0.5, places=12)
        self.assertAlmostEqual(c["spectral_radius_B"], 0.5, places=12)
        self.assertAlmostEqual(c["spectral_radius_BA"], 2.25 + math.sqrt(5), places=12)
        self.assertGreater(c["spectral_radius_BA"], 1.0)

    def test_single_stable_alternating_explodes(self):
        k32 = self.r["routes_by_length"]["32"]
        self.assertLess(k32["AAAA"]["final_norm"], 1e-6)
        self.assertLess(k32["BBBB"]["final_norm"], 1e-6)
        self.assertGreater(k32["ABAB"]["final_norm"], 1e6)
        self.assertLess(k32["greedy_min_norm_growth"]["final_norm"], 1e-6)

    def test_damping_not_assumed_sufficient(self):
        damp = self.r["stabilizers"]["residual_damping"]
        self.assertFalse(self.r["findings"]["damping_sufficient_at_all_alpha"])
        self.assertGreater(damp["1.0"]["spectral_radius_BA_damped"], 1.0)
        self.assertGreater(damp["0.5"]["spectral_radius_BA_damped"], 1.0)

    def test_spectral_norm_scaling_contracts(self):
        ns = self.r["stabilizers"]["spectral_norm_scaling"]
        self.assertLess(ns["0.8"]["alternating_k32_final_norm"], 1.0)
        self.assertLessEqual(ns["0.8"]["spectral_norm_A_after"], 0.8 + 1e-12)
        self.assertLessEqual(ns["0.8"]["spectral_norm_B_after"], 0.8 + 1e-12)
        self.assertGreater(ns["0.8"]["spectral_norm_A_before"], 1.0)

    def test_switching_graph_reduces_actions(self):
        sg = self.r["stabilizers"]["switching_graph_constraint"]
        self.assertEqual(sg["allowed_paths"], 2)
        self.assertEqual(sg["total_paths"], 256)
        self.assertAlmostEqual(sg["oracle_loss"], 0.0, places=12)

    def test_d2_gated_not_run(self):
        d2 = X.run_d2()
        self.assertEqual(d2["status"], "not_run_gated")
        self.assertFalse(d2["gate_conditions_met"])


class TestE1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = X.run_e1()

    def test_causal_controls_exactly_invariant(self):
        for name in X.E1_CAUSAL_CONTROLS:
            m = self.r["mechanisms"][name]
            self.assertEqual(m["prefix_route_mismatches_layer1"], 0, name)
            self.assertEqual(m["prefix_route_mismatches_layer2"], 0, name)
            self.assertEqual(m["prefix_logit_max_abs_diff"], 0.0, name)
            self.assertTrue(m["gate_pass"], name)

    def test_every_noncausal_rule_flagged(self):
        for name, m in self.r["mechanisms"].items():
            if name not in X.E1_CAUSAL_CONTROLS:
                self.assertTrue(m["causality_violation_detected"], name)
                self.assertGreater(
                    m["prefix_route_mismatches_layer1"]
                    + m["prefix_route_mismatches_layer2"], 0, name)
        self.assertTrue(self.r["summary"]["noncausal_rules_all_flagged"])
        self.assertTrue(self.r["summary"]["causal_controls_exact_zero"])

    def test_prefix_budget_rule_respects_budget(self):
        scores = np.vstack([X.E1_PREFIX, X.E1_SUFFIX_A])
        sel = X.mech_prefix_budget_online(scores, k=1)
        self.assertEqual(len(sel), scores.shape[0])
        self.assertTrue(all(len(s) <= 1 for s in sel))


class TestE2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = X.run_e2(seed=3)

    def test_c8_unfixed_exceeds_budget(self):
        c8 = self.r["c8_counterexample"]
        self.assertEqual(c8["unfixed_total_calls"], 3)
        self.assertEqual(c8["unfixed_budget_final"], 2)
        self.assertTrue(c8["unfixed_exceeds_budget"])
        self.assertEqual([r["new_calls"] for r in c8["unfixed_ledger"]], [1, 2])

    def test_naive_topset_inference_is_wrong(self):
        c8 = self.r["c8_counterexample"]
        self.assertEqual(c8["naive_topset_inference_would_report"], 2)
        self.assertTrue(c8["naive_inference_wrong"])

    def test_ledger_fix_restores_budget(self):
        c8 = self.r["c8_counterexample"]
        self.assertEqual(c8["fixed_total_calls"], 2)
        self.assertTrue(c8["fixed_within_budget"])
        for row in c8["fixed_ledger"]:
            self.assertLessEqual(row["cumulative_calls"], row["budget_mK"])
            self.assertEqual(row["remaining_budget"],
                             row["budget_mK"] - row["cumulative_calls"])

    def test_randomized_stress(self):
        s = self.r["randomized_stress"]
        self.assertGreater(s["unfixed_budget_violations"], 0)
        self.assertEqual(s["fixed_budget_violations"], 0)
        self.assertTrue(self.r["gate_pass"])

    def test_counting_bases_separate(self):
        metrics = self.r["counting_metrics"]["c8_fixed"]
        for key in ("unique_expert_params", "total_expert_calls",
                    "token_expert_pairs", "executed_kernels",
                    "padded_matrix_elements", "estimated_flops"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["total_expert_calls"], 2)


class TestE3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = X.run_e3()

    def test_c9_path_values(self):
        paths = self.r["router_comparison"]["paths"]
        self.assertAlmostEqual(paths["A->B"], 3.0)
        self.assertAlmostEqual(paths["B->A"], 4.0)
        self.assertAlmostEqual(paths["A->A"], 4.0)
        self.assertAlmostEqual(paths["B->B"], 3.0)
        self.assertTrue(self.r["router_comparison"]["same_set_order_dependent"])

    def test_parallel_mix_is_different_function_class(self):
        arch = self.r["architecture_comparison"]
        self.assertAlmostEqual(arch["parallel_top2_mean"]["output"], 2.0)
        self.assertAlmostEqual(arch["parallel_top2_residual"]["output"], 3.0)
        self.assertNotEqual(arch["serial_two_calls"]["output"],
                            arch["parallel_top2_mean"]["output"])

    def test_top2_pairs_counted_as_pairs(self):
        pc = self.r["pair_counting"]
        self.assertEqual(pc["top2_pair_actions"], 28)
        self.assertEqual(len(pc["pair_labels"]), 28)
        self.assertIn("(3,7)", pc["pair_labels"])
        self.assertEqual(pc["treated_as_independent_expert_actions_would_give"], 8)
        self.assertEqual(pc["top2_pair_h4_leaf_paths"], 28 ** 4)
        self.assertTrue(self.r["gate_pass"])


class TestCLI(unittest.TestCase):
    def test_output_json_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "e08.json")
            record = X.main(["--output", out, "--only", "d1,e3"])
            self.assertTrue(os.path.exists(out))
            loaded = json.load(open(out, encoding="utf-8"))
            self.assertEqual(set(loaded["results"]), {"d1", "e3"})
            self.assertIn("synthetic_scope", loaded)
            self.assertIn("C/D/E", record["experiment"])

    def test_unknown_section_rejected(self):
        with self.assertRaises(SystemExit):
            X.main(["--output", "/tmp/never_written.json", "--only", "c9"])


if __name__ == "__main__":
    unittest.main()
