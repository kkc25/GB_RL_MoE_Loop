from __future__ import annotations

import tempfile
import unittest

from analysis.report import build_report


class AnalysisReportTests(unittest.TestCase):
    def test_missing_math_checks_are_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            report = build_report(directory)
        self.assertFalse(report["math_gate_pass"])
        self.assertEqual(report["math_gate_status"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
