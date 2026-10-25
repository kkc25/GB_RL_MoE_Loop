from __future__ import annotations

import unittest

from experiments.e12_equivalence import run


class E12EquivalenceTests(unittest.TestCase):
    def test_linear_bids_are_central_router(self):
        result = run()
        self.assertTrue(result["equivalent"])
        self.assertLess(result["max_absolute_difference"], 1e-12)


if __name__ == "__main__":
    unittest.main()
