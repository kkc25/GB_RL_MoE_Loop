import tempfile
import unittest
from pathlib import Path

import torch

from analysis.goal_report import build_report
from config import ControllerConfig, ExperimentConfig, ModelConfig
from data.affine import sample_batch
from eval.oracle import exact_path_batch
from experiments.registry import Registry
from model import LoopMoE
from policy import Controller
from train.router_rloo import evaluate


class FollowupContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model_config = ModelConfig(
            layers=2,
            hidden_size=32,
            attention_heads=2,
            head_dim=16,
            intermediate_size=48,
            experts_per_layer=2,
            adaptive_layers=(1,),
            max_calls=2,
            target_only=True,
        )
        self.model = LoopMoE(self.model_config).eval()
        self.controller = Controller(self.model_config, ControllerConfig(projected_hidden=16, gru_hidden_size=16)).eval()
        batch = sample_batch("train", 2, 19, 0, (1, 2))
        self.input_ids = batch["input_ids"]
        self.labels = batch["labels"]

    def test_exact_rejects_multiple_supervised_positions(self):
        labels = self.labels.clone()
        labels[:, 0] = 1
        with self.assertRaises(NotImplementedError):
            next(exact_path_batch(self.model, self.controller, self.input_ids, labels, 2, 0.0, torch.device("cpu"), "float32"))

    def test_evaluation_passes_configured_price(self):
        config = ExperimentConfig()
        config.model = self.model_config
        config.train.validation_batches = 1
        config.train.batch_size = 2
        config.train.budget = 2
        config.train.lambda_cost = 0.3
        seen = []
        original = self.model.forward

        def recording_forward(*args, **kwargs):
            seen.append(kwargs["price"])
            return original(*args, **kwargs)

        self.model.forward = recording_forward
        evaluate(self.model, self.controller, config, torch.device("cpu"))
        self.assertEqual(seen, [0.3])

    def test_blocked_reason_survives_later_finished_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Registry(Path(temporary) / "registry.json")
            registry.update("E09", "neural", "blocked_by_precondition", stop_reason="student missing", next_trigger="positive student")
            registry.update("E09", "controlled", "finished")
            record = registry.data["experiments"]["E09"]
            self.assertEqual(record["status"], "blocked_by_precondition")
            self.assertEqual(record["stop_reason"], "student missing")
            self.assertEqual(record["next_trigger"], "positive student")

    def test_nested_registry_resolves_repo_relative_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "runs" / "repair" / "summary.json"
            evidence.parent.mkdir(parents=True)
            evidence.write_text("{}\n")
            registry_path = root / "runs" / "repair" / "registry.json"
            registry = Registry(registry_path)
            registry.update("E03", "confirmation", "finished", evidence=["runs/repair/summary.json"])
            report = build_report(registry_path, repo_root=root)
            self.assertEqual(report["missing_evidence"], [])
            self.assertTrue(report["experiments"]["E03"]["evidence"][0]["exists"])


if __name__ == "__main__":
    unittest.main()
