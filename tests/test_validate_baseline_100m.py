import json
import tempfile
import unittest
from pathlib import Path

from experiments.validate_baseline_100m import validate_profile
from train.recovery_common import sha256_file


class BaselineProfileValidationTests(unittest.TestCase):
    def make_root(self, directory):
        root = Path(directory)
        arm = root / "loop_r1"
        profile = arm / "profile_final"
        profile.mkdir(parents=True)
        (root / "loop_r1.json").write_text("{}\n")
        (arm / "latest.pt").write_bytes(b"checkpoint")
        protocol = {
            "config_sha256": sha256_file(root / "loop_r1.json"),
            "checkpoint_sha256": sha256_file(arm / "latest.pt"),
            "packed_file_sha256": {"input_ids": "a", "segment_ids": "b", "doc_map": "c"},
        }
        (profile / "profile_protocol.json").write_text(json.dumps(protocol))
        (profile / "measurements.jsonl").write_text("{}\n")
        summary = {
            "valid_for_latency_claims": True,
            "exclusive_at_start": True,
            "exclusive_at_end": True,
            "measurements_sha256": sha256_file(profile / "measurements.jsonl"),
            "latency": {
                str(depth): {
                    "calls": 1000,
                    "event_ms": {"mean": 50.0},
                    "effective_tokens_per_second": 20000.0,
                    "speedup_vs_r2": {"mean": 0.1},
                }
                for depth in (1, 2, 4)
            },
        }
        (profile / "summary.json").write_text(json.dumps(summary))
        return root

    def test_valid_profile_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = validate_profile(self.make_root(temporary), "loop_r1")
            self.assertEqual(result["mean_event_ms_r1"], 50.0)

    def test_checkpoint_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_root(temporary)
            (root / "loop_r1" / "latest.pt").write_bytes(b"different")
            with self.assertRaises(AssertionError):
                validate_profile(root, "loop_r1")


class BaselineDecisionTests(unittest.TestCase):
    def test_frozen_decision_branches(self):
        from experiments.validate_baseline_100m import decide_baseline

        self.assertEqual(decide_baseline(-0.01, 0.02), "controlled_replication")
        self.assertEqual(decide_baseline(0.01, 0.08), "efficiency_only_redesign")
        self.assertEqual(decide_baseline(-0.01, -0.02), "stop_or_fundamental_redesign")
        self.assertEqual(decide_baseline(0.001, 0.02), "stop_or_fundamental_redesign")


if __name__ == "__main__":
    unittest.main()
