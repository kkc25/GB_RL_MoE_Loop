import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import resource_accounting as ra
from eval import verify_router_math as vm


class TestRouterMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = vm.run()

    def test_terminal_paths_39(self):
        self.assertEqual(self.r['terminal_paths'], 39)
        self.assertEqual(self.r['expert_count'], 3)
        self.assertEqual(self.r['max_calls'], 3)

    def test_total_probability(self):
        self.assertAlmostEqual(self.r['total_probability'], 1.0, places=15)

    def test_expected_loss(self):
        self.assertAlmostEqual(self.r['expected_loss'], 0.19722830591662754, places=15)

    def test_pg_vs_finite_difference(self):
        self.assertLess(self.r['max_abs_pg_vs_fd'], 1e-9)
        self.assertAlmostEqual(self.r['max_abs_pg_vs_fd'], 6.06e-12, delta=1e-12)

    def test_rloo2_exact_expectation(self):
        self.assertLess(self.r['max_abs_rloo2_vs_exact_pg'], 1e-12)
        self.assertAlmostEqual(self.r['max_abs_rloo2_vs_exact_pg'], 2.95e-17, delta=1e-16)

    def test_self_mean_half_gradient(self):
        self.assertLess(self.r['max_abs_self_mean_vs_half_pg'], 1e-12)
        self.assertAlmostEqual(self.r['max_abs_self_mean_vs_half_pg'], 1.65e-17, delta=1e-16)

    def test_mean_score_zero(self):
        self.assertLess(self.r['max_abs_mean_score'], 1e-12)
        self.assertAlmostEqual(self.r['max_abs_mean_score'], 2.60e-17, delta=1e-16)

    def test_potential_shaping_telescoping(self):
        self.assertAlmostEqual(self.r['potential_shaping_shift'], -0.7, places=15)

    def test_myopic_counterexample(self):
        m = self.r['myopic_counterexample']
        self.assertFalse(m['greedy_continues'])
        self.assertTrue(m['optimal_continues'])
        self.assertAlmostEqual(m['stop_cost'], 1.0)
        self.assertAlmostEqual(m['two_call_cost'], 0.4)

    def test_expected_vs_marginal_nll(self):
        self.assertAlmostEqual(self.r['expected_route_nll'], 1.203973, places=6)
        self.assertAlmostEqual(self.r['marginal_nll'], 0.693147, places=6)
        self.assertGreater(self.r['expected_route_nll'], self.r['marginal_nll'])

    def test_halt_gradient_vs_fd(self):
        self.assertLess(self.r['max_abs_exact_halt_vs_fd'], 1e-9)


class TestResourceAccounting(unittest.TestCase):
    def test_parameter_counts(self):
        c = ra.all_counts()
        self.assertEqual(c['pilot']['inference_total'], 126_044_169)
        self.assertEqual(c['pilot']['training_total_with_critic'], 126_044_298)
        self.assertEqual(c['main']['inference_total'], 280_269_065)
        self.assertEqual(c['main']['training_total_with_critic'], 280_269_194)
        self.assertEqual(c['scale_optional']['inference_total'], 906_373_641)
        self.assertEqual(c['scale_optional']['training_total_with_critic'], 906_373_770)

    def test_controller_params(self):
        c = ra.all_counts()
        self.assertEqual(c['main']['controller'], 206_345)
        self.assertEqual(c['main']['inference_total'] - c['main']['backbone'],
                         98_432 + 106_752 + 1_161)
        for cfg in c.values():
            self.assertEqual(
                cfg['training_total_with_critic'] - cfg['inference_total'], 129)

    def test_memory_gib(self):
        c = ra.all_counts()
        self.assertAlmostEqual(c['pilot']['full_training_base_GiB'], 1.878, places=3)
        self.assertAlmostEqual(c['main']['full_training_base_GiB'], 4.176, places=3)
        self.assertAlmostEqual(c['scale_optional']['full_training_base_GiB'], 13.506, places=3)
        self.assertAlmostEqual(c['pilot']['frozen_environment_base_GiB'], 0.237, places=3)
        self.assertAlmostEqual(c['main']['frozen_environment_base_GiB'], 0.525, places=3)
        self.assertAlmostEqual(c['scale_optional']['frozen_environment_base_GiB'], 1.691, places=3)

    def test_flops_constants(self):
        self.assertAlmostEqual(ra.FIXED_MFLOPS, 144.703488, places=9)
        self.assertAlmostEqual(ra.EXPERT_MFLOPS, 4.718592, places=9)
        self.assertAlmostEqual(ra.GATE_MFLOPS, 0.012288, places=9)
        self.assertAlmostEqual(ra.POLICY_MFLOPS, 0.410880, places=9)

    def test_flops_table(self):
        expected = {
            (4, 8): (201.326592, 204.761088),
            (6, 10): (210.763776, 215.044608),
            (8, 12): (220.200960, 225.328128),
            (12, 16): (239.075328, 245.895168),
            (16, 16): (257.949696, 264.818688),
        }
        self.assertEqual([ (row['K'], row['D']) for row in ra.flops_table() ],
                         list(expected.keys()))
        for (K, D), (no_gc, with_gc) in expected.items():
            self.assertAlmostEqual(ra.flops_without_gate_controller(K), no_gc, places=6)
            self.assertAlmostEqual(ra.flops(K, D), with_gc, places=6)

    def test_loop_saving_4_to_2_calls(self):
        saving = 1 - ra.flops_without_gate_controller(8) / ra.flops_without_gate_controller(16)
        self.assertAlmostEqual(saving, 0.1463, places=4)

    def test_shared_prefix(self):
        sp = ra.shared_prefix_analysis()
        self.assertAlmostEqual(sp['shared_prefix_MFLOPs_per_token'], 108.62592, places=6)
        self.assertAlmostEqual(sp['two_routes_with_shared_prefix'], 342.030336, places=6)
        self.assertAlmostEqual(sp['relative_matrix_saving'], 0.241039, places=6)

    def test_full_head(self):
        fh = ra.full_head_analysis()
        self.assertAlmostEqual(fh['full_head_MFLOPs'], 50.331648, places=9)
        self.assertAlmostEqual(fh['head_to_expert_ratio'], 10.6667, places=4)
        self.assertAlmostEqual(
            fh['stop_plus_8_branch_readout_MFLOPs_per_queried_state'],
            490.733568, places=6)

    def test_token_quota(self):
        self.assertEqual(ra.TOKENS_PER_UPDATE, 262_144)
        q1 = ra.token_quota(1_000_000_000)
        self.assertEqual(q1['full_updates'], 3815)
        self.assertEqual(q1['actual_tokens'], 1_000_079_360)
        q2 = ra.token_quota(2_000_000_000)
        self.assertEqual(q2['full_updates'], 7630)
        self.assertEqual(q2['actual_tokens'], 2_000_158_720)


if __name__ == '__main__':
    unittest.main()
