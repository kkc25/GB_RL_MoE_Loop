"""Regression tests for the 2026-10-25 code audit registry/goal-report items
covering experiments/registry.py and analysis/goal_report.py.

They fail on the pre-repair tree and must pass after the repair:

* the experiment-level status was overwritten by whichever component was
  updated last, so e.g. E09 showed "finished" while its neural component was
  blocked_by_precondition; the status must be aggregated from all components
  deterministically, independent of update order;
* component GPU hours defaulted to 0.0 even when nobody recorded them, so
  real but unrecorded GPU work was indistinguishable from verified CPU-only
  work; missing hours must aggregate to unknown (null), not 0;
* analysis/goal_report.py must stay compatible with the aggregated status and
  unknown (null) GPU hours.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.goal_report import build_report
from experiments.registry import Registry, initial_registry

ROOT = Path(__file__).resolve().parents[1]


class RegistryStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "registry.json"

    def test_last_component_does_not_override_experiment_status(self):
        registry = Registry(self.path)
        registry.update("E09", "neural", "blocked_by_precondition")
        registry.update("E09", "controlled", "finished")
        self.assertEqual(registry.data["experiments"]["E09"]["status"], "blocked_by_precondition")

    def test_status_aggregation_is_order_independent(self):
        registry = Registry(self.path)
        registry.update("E09", "controlled", "finished")
        registry.update("E09", "neural", "blocked_by_precondition")
        self.assertEqual(registry.data["experiments"]["E09"]["status"], "blocked_by_precondition")

    def test_all_components_finished_marks_experiment_finished(self):
        registry = Registry(self.path)
        registry.update("E03", "tau_sweep", "finished", gpu_hours=0.1)
        registry.update("E03", "confirmation", "finished", gpu_hours=0.2)
        self.assertEqual(registry.data["experiments"]["E03"]["status"], "finished")

    def test_failed_component_marks_experiment_failed(self):
        registry = Registry(self.path)
        registry.update("E07", "offline_oracle", "finished")
        registry.update("E07", "online_check", "failed")
        self.assertEqual(registry.data["experiments"]["E07"]["status"], "failed")

    def test_stopped_component_not_hidden_by_finished(self):
        registry = Registry(self.path)
        registry.update("E11", "scale_out", "stopped_by_evidence")
        registry.update("E11", "s0", "finished")
        self.assertEqual(registry.data["experiments"]["E11"]["status"], "stopped_by_evidence")

    def test_running_component_marks_experiment_running(self):
        registry = Registry(self.path)
        registry.update("E01", "labels", "finished")
        registry.update("E01", "student", "running")
        self.assertEqual(registry.data["experiments"]["E01"]["status"], "running")


class RegistryGpuHoursTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "registry.json"

    def test_missing_gpu_hours_aggregate_to_unknown(self):
        registry = Registry(self.path)
        registry.update("E00", "c0", "finished", gpu_hours=0.0203)
        registry.update("E00", "m105_b2", "finished")  # hours never recorded
        record = registry.data["experiments"]["E00"]
        self.assertIsNone(record["components"]["m105_b2"]["gpu_hours"])
        self.assertIsNone(record["gpu_hours"], "unrecorded GPU work must aggregate to unknown, not 0")

    def test_all_known_gpu_hours_sum(self):
        registry = Registry(self.path)
        registry.update("E00", "c0", "finished", gpu_hours=0.0203)
        registry.update("E00", "m105_b2", "finished", gpu_hours=9.63)
        record = registry.data["experiments"]["E00"]
        self.assertAlmostEqual(record["gpu_hours"], 9.6503)

    def test_explicit_zero_hours_stays_known(self):
        registry = Registry(self.path)
        registry.update("E08", "cpu_only", "finished", gpu_hours=0.0)
        self.assertEqual(registry.data["experiments"]["E08"]["gpu_hours"], 0.0)

    def test_legacy_registry_with_numeric_hours_still_sums(self):
        legacy = initial_registry()
        legacy["experiments"]["E00"]["components"] = {
            "c0": {"status": "finished", "evidence": [], "metrics": {}, "gpu_hours": 0.0203, "updated_at": "2026-10-24T00:00:00+00:00"},
            "m105_b2": {"status": "finished", "evidence": [], "metrics": {}, "gpu_hours": 9.6279, "updated_at": "2026-10-24T00:00:00+00:00"},
        }
        self.path.write_text(json.dumps(legacy, indent=2) + "\n")
        registry = Registry(self.path)
        registry.update("E00", "b0", "finished", gpu_hours=0.5)
        record = registry.data["experiments"]["E00"]
        self.assertAlmostEqual(record["gpu_hours"], 0.0203 + 9.6279 + 0.5)
        self.assertEqual(record["status"], "finished")

    def test_cli_flags_unchanged_and_missing_hours_unknown(self):
        base = [sys.executable, str(ROOT / "experiments" / "registry.py"), "--path", str(self.path)]
        subprocess.run(
            base + ["--experiment", "E11", "--component", "s0", "--status", "finished", "--gpu-hours", "1.5"],
            cwd=ROOT, check=True, capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            base + ["--experiment", "E11", "--component", "scale_out", "--status", "stopped_by_evidence"],
            cwd=ROOT, check=True, capture_output=True, text=True, timeout=120,
        )
        record = json.loads(self.path.read_text())["experiments"]["E11"]
        self.assertEqual(record["status"], "stopped_by_evidence")
        self.assertIsNone(record["gpu_hours"])


class GoalReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry_path = self.root / "runs" / "registry.json"

    def test_goal_report_handles_aggregated_status_and_unknown_hours(self):
        (self.root / "evidence").mkdir(parents=True)
        (self.root / "evidence" / "e09_controlled.json").write_text("{}\n")
        registry = Registry(self.registry_path)
        registry.update("E09", "neural", "blocked_by_precondition")
        registry.update("E09", "controlled", "finished", evidence=["evidence/e09_controlled.json"], gpu_hours=2.0)
        report = build_report(self.registry_path)
        experiment = report["experiments"]["E09"]
        self.assertEqual(experiment["status"], "blocked_by_precondition")
        self.assertIsNone(experiment["gpu_hours"])
        self.assertTrue(experiment["gpu_hours_unknown"])
        self.assertEqual(experiment["evidence"], [{"path": "evidence/e09_controlled.json", "exists": True}])
        self.assertEqual(report["missing_evidence"], [])
        self.assertEqual(report["terminal_statuses"]["blocked_by_precondition"], 1)
        self.assertEqual(report["terminal_statuses"]["finished"], 0)

    def test_goal_report_on_fresh_registry(self):
        Registry(self.registry_path).save()
        report = build_report(self.registry_path)
        self.assertEqual(report["complete_experiment_records"], 13)
        self.assertFalse(report["all_have_components"])
        for experiment in report["experiments"].values():
            self.assertEqual(experiment["status"], "planned")
            self.assertFalse(experiment["gpu_hours_unknown"])


if __name__ == "__main__":
    unittest.main()
