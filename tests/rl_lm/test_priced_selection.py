"""Spec 16.1: pinned R1 bound, cost-aware selection and resume policy identity."""
import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from rl_lm.trainer import Trainer


class Config:
    def __init__(self, price):
        self.values = {"cost.lambda": price, "evaluation.r2_quality_tolerance_nll": 0.005}

    def get(self, key):
        return self.values[key]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.identity = {"checkpoint_sha256": "a" * 64, "max_calls": 1}

    def trainer(self, price=0.02, stage="r2"):
        trainer = Trainer.__new__(Trainer)
        trainer.config = Config(price)
        trainer.stage = stage
        trainer.evaluator = SimpleNamespace(r1_identity=self.identity)
        trainer.selection_policy = trainer._selection_policy()
        trainer.selection_state = {"best": None, "candidates": [], "evaluations": [],
                                   "policy": copy.deepcopy(trainer.selection_policy)}
        trainer.run_dir = Path(self.tmp.name)
        trainer.model = trainer.actor = None
        trainer.successful_updates = 0
        trainer.tie_tolerance = 1e-6
        trainer.rmax = 2
        trainer._save_checkpoint = lambda kind: None
        return trainer

    def evaluate(self, trainer, nll, rounds, *, r1_nll=5., identity=None):
        metrics = {"greedy_nll": nll, "mean_rounds": rounds,
                   "vs_selected_r1": {"identity": identity or self.identity, "nll": r1_nll}}
        trainer.evaluator = lambda *args: metrics
        result = trainer._evaluate_and_select()
        trainer.successful_updates += 1
        return result

    def test_price_can_prefer_higher_nll_with_lower_cost(self):
        t = self.trainer()
        self.evaluate(t, 5., 1.)
        self.evaluate(t, 4.99, 2.)  # better NLL but worse priced objective
        self.assertEqual(t.selection_state["best"]["update"], 0)
        self.evaluate(t, 4.97, 2.)
        self.assertEqual(t.selection_state["best"]["update"], 2)

    def test_infeasible_candidate_never_becomes_best_even_if_score_better(self):
        t = self.trainer()
        self.evaluate(t, 5., 2.)
        record = self.evaluate(t, 5.006, 1.)
        self.assertFalse(record["eligible"])
        self.assertEqual(t.selection_state["best"]["update"], 0)
        empty = self.trainer()
        self.evaluate(empty, 5.006, 1.)
        self.assertIsNone(empty.selection_state["best"])

    def test_infeasible_initial_is_recorded_and_later_feasible_can_win(self):
        t = self.trainer()
        t.config.to_dict = lambda: {}
        t.source_identity = None
        t.evaluator = lambda *args: {"greedy_nll": 5.006, "mean_rounds": 1.,
                                    "vs_selected_r1": {"identity": self.identity, "nll": 5.}}
        saved = []
        t._save_checkpoint = saved.append
        t._initialize_run()
        self.assertEqual(saved, ["initial", "latest"])
        self.assertIsNone(t.selection_state["best"])
        t.successful_updates = 1
        self.evaluate(t, 5., 1.)
        self.assertEqual(t.selection_state["best"]["update"], 1)
        self.assertEqual(saved[-1], "best")

    def test_all_infeasible_finish_reports_no_candidate(self):
        t = self.trainer()
        self.evaluate(t, 5.006, 1.)
        t.frozen_hash_at_start = "test-hash"
        t.ledger = SimpleNamespace(merge_peak=lambda _: None, to_dict=lambda: {})
        t.device = "cpu"
        t.start_time = 0
        t.run_id = "test"
        t.initialization = "fresh"
        t.planned_updates = t.stop_after = 1
        t.lr_scale = 1.
        t.lr_backoffs = 0
        with patch("rl_lm.trainer.model_parameter_hash", return_value="test-hash"):
            summary = t._finish("completed")
        self.assertEqual(summary["selection_status"], "no_feasible_candidate")
        self.assertIsNone(summary["best"])
        self.assertFalse((t.run_dir / "checkpoint_best.pt").exists())

    def test_zero_price_r2_still_enforces_bound(self):
        t = self.trainer(price=0.)
        self.evaluate(t, 5.006, 1.)
        self.assertIsNone(t.selection_state["best"])
        self.evaluate(t, 5.005, 1.)  # inclusive bound
        self.assertIsNotNone(t.selection_state["best"])

    def test_ties_prefer_fewer_rounds_then_earlier_update(self):
        t = self.trainer()
        self.evaluate(t, 4.98, 2.)
        self.evaluate(t, 5., 1.)
        self.evaluate(t, 5., 1.)
        self.assertEqual(t.selection_state["best"]["update"], 1)

    def test_missing_reference_fails_closed(self):
        t = self.trainer()
        t.evaluator = None
        with self.assertRaisesRegex(ValueError, "reference-checkpoint"):
            t._selection_policy()
        t.evaluator = lambda *args: {"greedy_nll": 5., "mean_rounds": 1.}
        with self.assertRaisesRegex(ValueError, "pinned R1"):
            t._evaluate_and_select()

    def test_nonfinite_or_changed_reference_rejected(self):
        t = self.trainer()
        self.evaluate(t, 5., 1.)
        with self.assertRaisesRegex(ValueError, "NLL changed"):
            self.evaluate(t, 5., 1., r1_nll=5.1)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.evaluate(t, float("nan"), 1.)
        with self.assertRaisesRegex(ValueError, "pinned R1"):
            self.evaluate(t, 5., 1., identity={"checkpoint_sha256": "b" * 64})

    def test_exact_resume_requires_same_policy_and_pinned_reference(self):
        t = self.trainer()
        self.evaluate(t, 5., 1.)
        t._validate_selection_resume(t.selection_state)
        legacy = copy.deepcopy(t.selection_state)
        del legacy["policy"]
        with self.assertRaisesRegex(ValueError, "legacy R2"):
            t._validate_selection_resume(legacy)
        changed = copy.deepcopy(t.selection_state)
        changed["policy"]["r1_checkpoint_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            t._validate_selection_resume(changed)
        missing = copy.deepcopy(t.selection_state)
        del missing["r1_reference_nll"]
        with self.assertRaisesRegex(ValueError, "pinned R1"):
            t._validate_selection_resume(missing)

    def test_production_constructor_rejects_r2_without_reference(self):
        from rl_lm.config import RLConfig
        from tests.rl_lm.test_resume import toy_provider, toy_r2_config_dict
        from tests.rl_lm.tiny_lm import build_tiny_model
        with self.assertRaisesRegex(ValueError, "reference-checkpoint"):
            Trainer(RLConfig.from_dict(toy_r2_config_dict(Path(self.tmp.name) / "missing")),
                    build_tiny_model(), train_contexts=toy_provider(64, 1234), evaluator=None,
                    device="cpu", source_identity={"test": True})

    def test_r2_checkpoint_roundtrip_and_legacy_resume_rejection(self):
        import torch
        from tests.rl_lm.test_resume import make_trainer, toy_r2_config_dict
        run = Path(self.tmp.name) / "r2"
        config = toy_r2_config_dict(run)
        config["evaluation"]["interval_updates"] = 1
        initial = make_trainer(config, stop_after=1)
        initial.run()
        checkpoint = run / "checkpoint_latest.pt"
        resumed = make_trainer(config, resume_path=checkpoint, stop_after=2)
        self.assertEqual(resumed.selection_state, initial.selection_state)
        resumed.run()
        self.assertEqual(resumed.successful_updates, 2)
        payload = torch.load(checkpoint, weights_only=False)
        del payload["selection_state"]["policy"]
        torch.save(payload, checkpoint)
        with self.assertRaisesRegex(ValueError, "legacy R2"):
            make_trainer(config, resume_path=checkpoint, stop_after=3)

    def test_r1_legacy_selection_and_resume_unchanged(self):
        t = self.trainer(price=0., stage="r1")
        self.evaluate(t, 5.1, 1.)
        self.evaluate(t, 5., 1.)
        self.assertEqual(t.selection_state["best"]["update"], 1)
        t._validate_selection_resume({})
        self.assertNotIn("selection_score", t.selection_state["best"])


if __name__ == "__main__":
    unittest.main()
