import unittest

from experiments.profile_fixed_depth import latency_summary, randomized_depth_order


class FixedDepthProfileTests(unittest.TestCase):
    def test_depth_order_is_deterministic_and_complete(self):
        first = randomized_depth_order([1, 2, 4], 2, 3, seed=7)
        self.assertEqual(first, randomized_depth_order([1, 2, 4], 2, 3, seed=7))
        self.assertEqual(sorted(first), [1, 2, 4])

    def test_latency_summary_uses_cuda_time_and_repeat_variation(self):
        records = []
        for repeat in range(2):
            for depth, milliseconds in ((1, 50 + repeat), (2, 100 + repeat), (4, 150 + repeat)):
                records.append({"repeat": repeat, "depth": depth, "event_ms": milliseconds, "wall_ms": milliseconds + 2, "valid_tokens": 100})
        summary = latency_summary(records, [1, 2, 4])
        self.assertEqual(summary["1"]["calls"], 2)
        self.assertEqual(summary["1"]["event_ms"]["p50"], 50.5)
        self.assertGreater(summary["1"]["speedup_vs_r2"]["mean"], 0.9)
        self.assertEqual(summary["2"]["speedup_vs_r2"]["mean"], 0.0)
        self.assertLess(summary["4"]["speedup_vs_r2"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()
