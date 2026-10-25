import itertools
import json
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import e07_synthetic as e07


def brute_force_optimal(deltas, budget):
    best = 0.0
    for rounds in itertools.product(*[range(len(d) + 1) for d in deltas]):
        if sum(rounds) > budget:
            continue
        val = sum(deltas[i][h] for i in range(len(deltas)) for h in range(rounds[i]))
        best = max(best, val)
    return best


class TestExactDP(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = random.Random(7)
        for _ in range(30):
            n = rng.randint(1, 4)
            H = rng.randint(1, 4)
            deltas = e07.sample_instance(rng, n, H)
            budget = rng.randint(1, n * H)
            opt, _ = e07.exact_dp(deltas, budget)
            self.assertAlmostEqual(opt, brute_force_optimal(deltas, budget), places=9)

    def test_dp_upper_bounds_heuristics(self):
        rng = random.Random(11)
        for _ in range(20):
            deltas = e07.sample_instance(rng, rng.randint(2, 8), rng.randint(2, 4))
            budget = rng.randint(1, 12)
            opt, _ = e07.exact_dp(deltas, budget)
            for val in (e07.index_allocation(deltas, budget)[0],
                        e07.two_step_package_allocation(deltas, budget)[0],
                        e07.round_robin_allocation(deltas, budget)[0],
                        e07.random_allocation(deltas, budget, rng)[0]):
                self.assertLessEqual(val, opt + 1e-9)

    def test_budget_and_horizon_respected(self):
        rng = random.Random(13)
        for _ in range(20):
            deltas = e07.sample_instance(rng, 5, 4)
            budget = rng.randint(1, 10)
            for rounds in (e07.exact_dp(deltas, budget)[1],
                           e07.index_allocation(deltas, budget)[1]):
                self.assertLessEqual(sum(rounds), budget)
                for i, r in enumerate(rounds):
                    self.assertLessEqual(r, len(deltas[i]))

    def test_no_forced_spending_on_nonpositive_gain(self):
        deltas = [[-0.5, -0.4], [-0.1, -0.2]]
        total, _, used, _ = e07.index_allocation(deltas, 4)
        self.assertEqual(used, 0)
        self.assertEqual(total, 0.0)
        opt, rounds = e07.exact_dp(deltas, 4)
        self.assertAlmostEqual(opt, 0.0)
        self.assertEqual(sum(rounds), 0)


class TestTrajectories(unittest.TestCase):
    def test_diminishing_greedy_is_optimal(self):
        deltas = [[1.0, 0.6, 0.3, 0.1], [0.8, 0.5, 0.2, 0.05]]
        budget = 5
        opt, _ = e07.exact_dp(deltas, budget)
        idx = e07.index_allocation(deltas, budget)[0]
        self.assertAlmostEqual(idx, opt, places=9)

    def test_deep_delay_greedy_fails_and_failure_is_kept(self):
        inst = e07.delayed_benefit_instances()["deep_delay"]
        opt, _ = e07.exact_dp(inst["deltas"], inst["budget"])
        idx_total, _, idx_used, _ = e07.index_allocation(inst["deltas"], inst["budget"])
        self.assertAlmostEqual(opt, 5.1, places=9)
        self.assertAlmostEqual(idx_total, 0.85, places=9)
        self.assertEqual(idx_used, inst["budget"])  # greedily burns all on B
        pkg = e07.two_step_package_allocation(inst["deltas"], inst["budget"])[0]
        self.assertLess(pkg, opt - 1e-9)  # deep delay defeats the package too

    def test_shallow_delay_package_recovers(self):
        inst = e07.delayed_benefit_instances()["shallow_delay"]
        idx = e07.index_allocation(inst["deltas"], inst["budget"])[0]
        pkg = e07.two_step_package_allocation(inst["deltas"], inst["budget"])[0]
        self.assertAlmostEqual(idx, 0.85, places=9)
        self.assertGreater(pkg, idx + 1.0)


class TestSeqTopKCounterexample(unittest.TestCase):
    def test_formula_overdraws_prefix_budget(self):
        ce = e07.seqtopk_counterexample()
        self.assertEqual([r["executed_calls"] for r in ce["formula_trace"]], [1, 2])
        self.assertEqual(ce["real_executed_total"], 3)
        self.assertEqual(ce["prefix_budget_total"], 2)
        self.assertTrue(ce["violation"])
        self.assertTrue(ce["formula_trace"][-1]["violation"])
        self.assertFalse(ce["formula_trace"][0]["violation"])

    def test_virtual_accounting_hides_overdraw(self):
        ce = e07.seqtopk_counterexample()
        self.assertEqual([r["virtual_new"] for r in ce["formula_trace"]], [1, 1])
        self.assertEqual(ce["virtual_accounting_total"], 2)

    def test_hard_ledger_blocks_overdraw(self):
        ce = e07.seqtopk_counterexample()
        executed = [e["executed"] for e in ce["hard_ledger_replay"]]
        self.assertEqual(executed, [1, 1])
        self.assertEqual(ce["hard_ledger_violations"], [])


class TestHardLedger(unittest.TestCase):
    def test_prefix_invariant_every_prefix(self):
        demo = e07.ledger_demo()
        self.assertTrue(demo["prefix_invariant_holds"])
        for entries in (demo["uncapped"], demo["capped"]):
            for e in entries:
                self.assertLessEqual(e["cum_executed"], e["prefix_budget"])
                self.assertGreaterEqual(e["balance"], 0)

    def test_base_round_always_affordable(self):
        demo = e07.ledger_demo()
        for e in demo["uncapped"] + demo["capped"]:
            self.assertGreaterEqual(e["executed"], 2)

    def test_carryover_cap_voids_overflow(self):
        demo = e07.ledger_demo()
        self.assertGreater(sum(e["voided_overflow"] for e in demo["capped"]), 0)
        self.assertEqual(sum(e["voided_overflow"] for e in demo["uncapped"]), 0)
        for e in demo["capped"]:
            self.assertLessEqual(e["balance"], demo["carryover_cap"])

    def test_uncapped_carries_surplus_forward(self):
        entries, violations = e07.simulate_ledger(3, [2, 2, 2, 6])
        self.assertEqual(violations, [])
        self.assertEqual([e["executed"] for e in entries], [2, 2, 2, 6])
        self.assertEqual(entries[-1]["balance"], 0)


class TestCLI(unittest.TestCase):
    def test_output_json_written(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "e07.json")
            e07.main(["--output", out])
            with open(out) as f:
                r = json.load(f)
        self.assertEqual(r["experiment"], "E07_synthetic")
        self.assertEqual(r["service_load"]["status"], "not_executed")
        self.assertIn("scenario_a_exact_dp", r)
        self.assertIn("seqtopk_counterexample", r)
        self.assertIn("scenario_b_hard_ledger", r)
        self.assertTrue(r["seqtopk_counterexample"]["violation"])
        self.assertTrue(r["scenario_b_hard_ledger"]["prefix_invariant_holds"])
        delayed = r["scenario_a_exact_dp"]["delayed_benefit_instances"]
        self.assertGreater(delayed["deep_delay"]["heuristics"]["index_d1_c1"]["rel_gap"], 0.5)
        curve = r["scenario_a_exact_dp"]["dp_gap"]["curve"]
        self.assertTrue(all(row["budget"] >= h["mean_used_budget"]
                            for row in curve
                            for h in row["heuristics"].values()))

    def test_seed_reproducible(self):
        self.assertEqual(e07.run(seed=1), e07.run(seed=1))


if __name__ == '__main__':
    unittest.main()
