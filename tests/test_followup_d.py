import json
import tempfile
import unittest
from pathlib import Path

import torch

import experiments.followup_d_learnability as followup_d
from eval.counterfactual import PathCosts
from experiments.followup_b_recompute import validate_backbone_config, validate_states_identity
from experiments.followup_d_learnability import (
    CandidateAdvantageScorer,
    advantage_targets,
    candidate_predict,
    fit_candidate,
    pair_router_features,
    subset_costs,
)
from train.recovery_common import sha256_file


class FollowupDTests(unittest.TestCase):
    def test_advantage_is_relative_to_each_states_baseline(self):
        costs = PathCosts(
            torch.zeros(2, 2),
            torch.tensor([[3.0, 2.0], [4.0, 1.0]]),
            torch.tensor([1, 0]),
            torch.zeros(2, 2, 2, dtype=torch.long),
            4,
            torch.tensor([[10, 20], [10, 20]]),
        )
        self.assertTrue(torch.equal(advantage_targets(costs), torch.tensor([[1.0, 0.0], [0.0, -3.0]])))
        subset = subset_costs(costs, torch.tensor([1]))
        self.assertTrue(torch.equal(subset.global_action_ids, torch.tensor([[10, 20]])))
        self.assertTrue(torch.equal(subset.baseline_action, torch.tensor([0])))

    def test_candidate_scorer_reads_actual_action_identity(self):
        torch.manual_seed(0)
        model = CandidateAdvantageScorer(8, 4, 28, hidden_dim=12)
        h = torch.randn(1, 8)
        router = torch.randn(1, 4)
        pairs = torch.randn(1, 2, 2)
        first = model(h, router, pairs, torch.tensor([[3, 3]]))
        second = model(h, router, pairs, torch.tensor([[3, 7]]))
        self.assertEqual(first.shape, (1, 2))
        self.assertNotEqual(float(first[0, 1]), float(second[0, 1]))

    def test_candidate_fit_and_predict_smoke(self):
        torch.manual_seed(1)
        n, candidates, experts = 16, 4, 4
        h = torch.randn(n, 8)
        router = torch.randn(n, experts)
        actions = torch.tensor([[[0, 1], [0, 2], [1, 2], [2, 3]]]).expand(n, -1, -1)
        global_ids = torch.tensor([[0, 1, 4, 9]]).expand(n, -1).clone()
        pair = pair_router_features(router, actions)
        target = torch.randn(n, candidates)
        fitted = fit_candidate(h, router, pair, global_ids, target, epochs=1, seed=0, hidden_dim=12, batch_size=8)
        prediction = candidate_predict(fitted["model"], h, router, pair, global_ids, fitted["normalization"])
        self.assertEqual(prediction.shape, (n, candidates))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertEqual(fitted["optimizer_updates"], 2)


class FollowupDEntryIdentityTests(unittest.TestCase):
    """Plan D must wire the same states/config identity validation as B before
    any router-feature forward over the cached states."""

    ROLES = ("controller_train", "selection", "newtest")

    def test_entry_uses_shared_identity_validators(self):
        self.assertIs(followup_d.validate_states_identity, validate_states_identity)
        self.assertIs(followup_d.validate_backbone_config, validate_backbone_config)

    def test_role_states_with_wrong_backbone_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"backbone weights")
            backbone_hash = sha256_file(checkpoint)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "manifest.json").write_text(json.dumps({
                "splits": {role: {"doc_map_sha256": f"{index:064x}"} for index, role in enumerate(self.ROLES, start=1)},
            }))
            states = {
                role: {"backbone_hash": backbone_hash, "corpus_doc_map_sha256": f"{index:064x}"}
                for index, role in enumerate(self.ROLES, start=1)
            }
            validate_states_identity(states, checkpoint_path=checkpoint, corpus_dir=corpus)
            states["selection"]["backbone_hash"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "selection"):
                validate_states_identity(states, checkpoint_path=checkpoint, corpus_dir=corpus)


if __name__ == "__main__":
    unittest.main()
