import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from eval.counterfactual import PathCosts
from experiments.followup_b_recompute import (
    analytic_mix,
    selection_diagnostics,
    validate_backbone_config,
    validate_states_identity,
)
from lm_config import LMExperimentConfig
from train.recovery_common import sha256_file


class FollowupBTests(unittest.TestCase):
    def test_analytic_mix_is_exact_per_state_expectation(self):
        stop = torch.tensor([1.0, 3.0])
        cont = torch.tensor([5.0, 7.0])
        self.assertTrue(torch.equal(analytic_mix(stop, cont, 0.0), stop))
        self.assertTrue(torch.equal(analytic_mix(stop, cont, 1.0), cont))
        self.assertTrue(torch.equal(analytic_mix(stop, cont, 0.25), torch.tensor([2.0, 4.0])))
        with self.assertRaises(ValueError):
            analytic_mix(stop, cont, 1.5)

    def test_selection_diagnostics_uses_actual_global_ids_and_fallback(self):
        costs = PathCosts(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            torch.tensor([0, 1]),
            torch.zeros(2, 3, 2, dtype=torch.long),
            4,
            torch.tensor([[10, 20, 21], [23, 10, 24]]),
        )
        constant = selection_diagnostics(costs, torch.tensor([1, 1]), 20)
        self.assertEqual(constant["actual_global_action_counts"], {"10": 1, "20": 1})
        self.assertEqual(constant["fallback_fraction"], 0.5)
        group = selection_diagnostics(costs, torch.tensor([0, 1]), torch.tensor([10, 10]))
        self.assertEqual(group["fallback_fraction"], 0.0)


class StatesIdentityTests(unittest.TestCase):
    """B/D entry validation: cached states must match the checkpoint, the
    corpus doc-maps and the config before any forward consumes them."""

    ROLES = ("controller_train", "newtest")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.checkpoint = root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"backbone weights")
        self.backbone_hash = sha256_file(self.checkpoint)
        self.doc_map_hashes = {role: f"{index:064x}" for index, role in enumerate(self.ROLES, start=1)}
        self.corpus = root / "corpus"
        self.corpus.mkdir()
        (self.corpus / "manifest.json").write_text(json.dumps({
            "splits": {role: {"doc_map_sha256": digest} for role, digest in self.doc_map_hashes.items()},
        }))
        self.states = {
            role: {"backbone_hash": self.backbone_hash, "corpus_doc_map_sha256": self.doc_map_hashes[role]}
            for role in self.ROLES
        }

    def tearDown(self):
        self._tmp.cleanup()

    def test_matching_inputs_verify(self):
        identity = validate_states_identity(self.states, checkpoint_path=self.checkpoint, corpus_dir=self.corpus)
        self.assertEqual(identity["checkpoint_sha256"], self.backbone_hash)
        self.assertIn("corpus_manifest_sha256", identity)

    def test_wrong_checkpoint_rejected_before_use(self):
        self.checkpoint.write_bytes(b"different backbone weights")
        with self.assertRaisesRegex(ValueError, "backbone hash"):
            validate_states_identity(self.states, checkpoint_path=self.checkpoint, corpus_dir=self.corpus)

    def test_wrong_doc_map_rejected_before_use(self):
        self.states["newtest"]["corpus_doc_map_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "doc-map hash"):
            validate_states_identity(self.states, checkpoint_path=self.checkpoint, corpus_dir=self.corpus)

    def test_missing_backbone_hash_rejected(self):
        del self.states["controller_train"]["backbone_hash"]
        with self.assertRaisesRegex(ValueError, "backbone hash"):
            validate_states_identity(self.states, checkpoint_path=self.checkpoint)

    def test_backbone_config_must_match_checkpoint_embedding(self):
        config = LMExperimentConfig()
        validate_backbone_config(config, {"config": {"model": asdict(config.model)}})
        tampered = asdict(config.model)
        tampered["layers"] = config.model.layers + 1
        with self.assertRaisesRegex(ValueError, "model section"):
            validate_backbone_config(config, {"config": {"model": tampered}})


if __name__ == "__main__":
    unittest.main()
