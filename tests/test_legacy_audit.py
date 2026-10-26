"""Regression tests for the 2026-10-25 code audit (audits/code_audit_2026-10-25).

Each test encodes a failure reproduced against the legacy code; they fail on
the pre-repair tree and must pass after the repair:

* ``train/exact.py`` optimised ``mean(F * log_pi) / |T|`` per path: no
  probability weighting of the score-function terms and an extra ``1/|T|``
  scaling, so the optimised quantity disagreed with the recorded expected
  objective (analytic two-leaf fixture: true gradient +0.16, legacy -0.2).
* ``eval/oracle.py`` enumerated forced paths deeper than the budget; the
  truncated duplicates inflated the probability mass (experts=2, max_calls=2,
  budget=1 -> mass 3.0 instead of 1.0 over the 2 unique legal paths).
* ``run_recovery.py --mode evaluate`` never loaded a checkpoint and exited 0
  on a randomly initialised model even when the checkpoint did not exist.
* ``eval/causality.py`` exited 1 because the aggregate entry unconditionally
  built a ``target_only`` multi-adaptive configuration that ``ModelConfig``
  rejects, instead of recording the rejection as expected behaviour.
"""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ControllerConfig, ExperimentConfig, ModelConfig
from data.affine import sample_batch
from data.recovery import build_datasets, build_manifest
from eval import causality
from eval.oracle import exact_path_batch
from model import LoopMoE
from policy import Controller
from recovery_config import RecoveryConfig
from train.common import CostLedger
from train.recovery import evaluate_recovery
from train.recovery_common import RecoveryLedger, recovery_checkpoint
from data.recovery import recovery_collate

ROOT = Path(__file__).resolve().parents[1]


class Recorder:
    def __init__(self):
        self.rows = []

    def write(self, **kwargs):
        self.rows.append(kwargs)


class ExactEnumerationTests(unittest.TestCase):
    """eval/oracle.py must enumerate only unique legal paths."""

    def setUp(self):
        torch.manual_seed(17)
        config = ModelConfig(
            layers=2, hidden_size=16, attention_heads=2, head_dim=8,
            intermediate_size=16, experts_per_layer=2, adaptive_layers=(1,),
            max_calls=2, target_only=True,
        )
        self.model = LoopMoE(config).eval()
        self.controller = Controller(config, ControllerConfig(projected_hidden=8, gru_hidden_size=8)).eval()
        self.input_ids = torch.tensor([[1, 5, 3]])
        self.labels = torch.tensor([[-100, -100, 6]])

    def enumerate_mass(self, budget):
        mass = 0.0
        count = 0
        for _, _, log_probability in exact_path_batch(
            self.model, self.controller, self.input_ids, self.labels,
            budget, 0.0, torch.device("cpu"), "float32",
        ):
            mass += float(log_probability.detach().exp().sum())
            count += 1
        return count, mass

    def test_short_budget_enumerates_unique_legal_paths(self):
        # budget=1, max_calls=2, experts=2: only the 2 depth-1 paths are
        # legal; legacy code enumerated 6 and returned probability mass 3.0.
        count, mass = self.enumerate_mass(1)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(mass, 1.0, places=5)

    def test_full_budget_probability_mass(self):
        count, mass = self.enumerate_mass(2)
        self.assertEqual(count, 6)
        self.assertAlmostEqual(mass, 1.0, places=5)

    def test_objective_price_matches_forward_price(self):
        # The training/evaluation objective must add price * cost with the
        # same price that the routed forward observes.
        for price in (0.0, 0.3):
            for path, objective, _ in exact_path_batch(
                self.model, self.controller, self.input_ids, self.labels,
                2, price, torch.device("cpu"), "float32",
            ):
                with torch.no_grad():
                    output = self.model(
                        self.input_ids, self.labels, controller=self.controller,
                        budget=2, price=price, forced_path=path,
                        collect_rollout=True, keep_logits=False,
                    )
                reference = output.rollout.token_losses.float() + price * output.rollout.costs.float()
                torch.testing.assert_close(objective, reference)


class ExactGradientTests(unittest.TestCase):
    """train_exact must optimise the same expected objective it records:
    the analytic two-leaf fixture has true gradient +0.16 (legacy: -0.2)."""

    def test_gradient_matches_analytic_expectation(self):
        trainer = importlib.import_module("train.exact")
        config = ExperimentConfig()
        config.model.experts_per_layer = 2
        config.model.max_calls = 1
        config.data.min_depth = config.data.max_depth = 1
        config.train.updates = 1
        config.train.batch_size = 2
        config.train.controller_learning_rate = 0.01
        config.train.validation_interval = 10
        config.train.dtype = "float32"
        model = torch.nn.Linear(1, 1)
        controller = torch.nn.Module()
        controller.register_parameter("z", torch.nn.Parameter(torch.tensor(math.log(4.0))))

        def analytic_paths(model, controller, input_ids, labels, *args):
            n = input_ids.shape[0]
            yield (0,), torch.ones(n), torch.nn.functional.logsigmoid(-controller.z).expand(n)
            yield (1,), torch.full((n,), 2.0), torch.nn.functional.logsigmoid(controller.z).expand(n)

        original_paths, original_evaluate = trainer.exact_path_batch, trainer.evaluate
        trainer.exact_path_batch = analytic_paths
        trainer.evaluate = lambda *args, **kwargs: {"fixture": True}
        recorder = Recorder()
        try:
            trainer.train_exact(model, controller, config, torch.device("cpu"), CostLedger(), recorder)
        finally:
            trainer.exact_path_batch, trainer.evaluate = original_paths, original_evaluate
        # E[F] = 1 * sigmoid(-z) + 2 * sigmoid(z); d/dz at z = log 4 is +0.16.
        self.assertAlmostEqual(float(controller.z.grad), 0.16, places=6)
        exact_rows = [row for row in recorder.rows if row.get("phase") == "exact"]
        self.assertEqual(len(exact_rows), 1)
        self.assertAlmostEqual(exact_rows[0]["expected_objective"], 1.8, places=6)
        self.assertAlmostEqual(exact_rows[0]["probability_mass"], 1.0, places=6)


class RecoveryEvaluateTests(unittest.TestCase):
    """--mode evaluate must load and validate a checkpoint and must fail
    when it does not exist or does not match the current task data."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.output_dir = root / "run"
        config = RecoveryConfig()
        config.model = ModelConfig(
            layers=2, hidden_size=32, attention_heads=2, head_dim=16,
            intermediate_size=48, experts_per_layer=2, adaptive_layers=(1,),
            max_calls=2, target_only=True,
        )
        config.task.stage = "c0_identity"
        config.run.device = "cpu"
        config.run.dtype = "float32"
        config.run.batch_size = 128
        config.run.output_dir = str(self.output_dir)
        self.config = config
        self.config_path = root / "config.json"
        config.save(self.config_path)

    def run_cli(self, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "run_recovery.py"),
             "--config", str(self.config_path), "--mode", "evaluate", *extra],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )

    def make_checkpoint(self, data_hash=None):
        torch.manual_seed(3)
        model = LoopMoE(self.config.model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        manifest = build_manifest(self.config.task.stage, self.config.task.seed)
        path = self.output_dir / "best_validation.pt"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        recovery_checkpoint(
            path, self.config, model, optimizer, 7, RecoveryLedger(),
            manifest["sha256"] if data_hash is None else data_hash,
        )
        return path, model, manifest

    def test_missing_checkpoint_is_an_error(self):
        # Legacy behaviour: exit code 0 evaluating a random model.
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0, result.stdout[-2000:])
        self.assertIn("checkpoint", (result.stderr + result.stdout).lower())

    def test_evaluate_loads_and_validates_checkpoint(self):
        checkpoint_path, model, manifest = self.make_checkpoint()
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = json.loads((self.output_dir / "stage_result.json").read_text())
        self.assertEqual(summary["checkpoint_selection"], str(checkpoint_path))
        self.assertEqual(summary["checkpoint_update"], 7)
        # Metrics must come from the checkpoint weights: recompute them
        # in-process from the same state dict.
        datasets = build_datasets(self.config.task.stage, self.config.task.seed)
        reloaded = LoopMoE(self.config.model)
        reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"])
        for split, dataset in datasets.items():
            expected = evaluate_recovery(
                reloaded, dataset, recovery_collate, self.config,
                torch.device("cpu"), self.config.run.fixed_depth,
            )
            for key in ("ce_full", "accuracy"):
                self.assertAlmostEqual(summary["metrics"][split][key], expected[key], places=6)

    def test_evaluate_rejects_data_hash_mismatch(self):
        self.make_checkpoint(data_hash="0" * 64)
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0, result.stdout[-2000:])
        self.assertIn("hash", (result.stderr + result.stdout).lower())

    def test_evaluate_resume_path(self):
        checkpoint_path, _, _ = self.make_checkpoint()
        missing = self.run_cli("--resume", str(self.output_dir / "does_not_exist.pt"))
        self.assertNotEqual(missing.returncode, 0, missing.stdout[-2000:])
        ok = self.run_cli("--resume", str(checkpoint_path))
        self.assertEqual(ok.returncode, 0, ok.stderr[-2000:])
        summary = json.loads((self.output_dir / "stage_result.json").read_text())
        self.assertEqual(summary["checkpoint_selection"], str(checkpoint_path))


class CausalityEntryTests(unittest.TestCase):
    """The causality aggregate entry must record the rejected illegal
    target-only multi-adaptive configuration as expected behaviour and still
    produce the complete report (legacy: exit code 1)."""

    EXPECTED_KEYS = {
        "dtype", "device", "fixed_depth_calls_target_only",
        "fixed_depth_calls_full_sequence", "budget", "prefix_invariance",
        "batch_invariance", "sampled_batch_invariance_informational",
        "sampled_reproducibility", "zero_delta_greedy_vs_fixed", "replay",
        "multi_adaptive", "multi_adaptive_target_only_probe",
    }

    def test_probe_raises_on_illegal_config(self):
        # The probe itself surfaces the ModelConfig rejection; run() is
        # responsible for recording it (tests/test_model_policy.py locks this
        # raising contract).
        with self.assertRaises(ValueError):
            causality.check_multi_adaptive_target_only(
                torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long),
            )

    def test_run_records_config_rejection(self):
        report = causality.run(seed=0)
        probe = report["multi_adaptive_target_only_probe"]
        self.assertTrue(probe["config_rejected"])
        self.assertTrue(probe["rejection_expected"])
        self.assertIn("target_only", probe["error"])
        self.assertEqual(set(report), self.EXPECTED_KEYS)

    def test_cli_outputs_complete_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            result = subprocess.run(
                [sys.executable, str(ROOT / "eval" / "causality.py"),
                 "--seed", "0", "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=600,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            report = json.loads(result.stdout)
            self.assertEqual(set(report), self.EXPECTED_KEYS)
            probe = report["multi_adaptive_target_only_probe"]
            self.assertTrue(probe["config_rejected"])
            self.assertTrue(probe["rejection_expected"])
            self.assertEqual(json.loads(output.read_text()), report)


if __name__ == "__main__":
    unittest.main()
