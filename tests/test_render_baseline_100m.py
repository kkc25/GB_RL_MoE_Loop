import json
import tempfile
import unittest
from pathlib import Path

from experiments.render_baseline_100m_report import render


class BaselineReportTests(unittest.TestCase):
    def test_render_is_deterministic_and_status_update_is_idempotent(self):
        arm = {
            "parameters": 100,
            "input_tokens": 99999744,
            "supervised_tokens": 99000000,
            "r1_nll": 5.0,
            "elapsed_seconds": 10.0,
            "allocated_gpu_hours": 0.01,
            "peak_memory_bytes": 1000,
        }
        profile = {"mean_event_ms_r1": 50.0, "effective_tokens_per_second_r1": 20000.0, "speedup_vs_r2": {"mean": 0.05}}
        data = {
            "arms": {"loop_r1": arm, "dense_parameter": dict(arm, r1_nll=5.1)},
            "profiles": {"loop_r1": profile, "dense_parameter": profile},
            "loop_minus_dense_nll": -0.1,
            "loop_speedup_vs_dense": 0.1,
            "next_stage_decision": "controlled_replication",
            "decision_protocol_sha256": "abc",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation.json"
            report = root / "report.md"
            status = root / "status.md"
            validation.write_text(json.dumps(data))
            status.write_text("# Current\n")
            first = render(validation, report, status)
            render(validation, report, status)
            text = report.read_text()
            self.assertIn("-0.100000000", text)
            self.assertIn("controlled_replication", text)
            self.assertEqual(status.read_text().count("## 100M 强基线对照"), 1)
            self.assertEqual(first["decision"], "controlled_replication")


if __name__ == "__main__":
    unittest.main()
