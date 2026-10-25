from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from experiments import e03_halting as e03
from model.top2 import enumerate_top2_actions
from train.value_student import ValueStudent

DIM = 8
N_ACTIONS = 28
ROUNDS = 4
HASH = "a" * 64


def make_split(directory: Path, split: str, n: int, backbone_hash: str = HASH, seed: int = 0,
               true_delta: torch.Tensor | None = None) -> dict:
    generator = torch.Generator().manual_seed(seed)
    h = torch.randn(n, DIM, generator=generator)
    baseline_action = torch.randint(N_ACTIONS, (n,), generator=generator)
    immediate = 6.0 + 0.1 * torch.randn(n, N_ACTIONS, generator=generator)
    if true_delta is None:
        true_delta = 0.05 * torch.randn(n, generator=generator)
    q_pi0 = immediate + 0.01 * torch.randn(n, N_ACTIONS, generator=generator)
    rows = torch.arange(n)
    q_pi0[rows, baseline_action] = immediate[rows, baseline_action] - true_delta
    data = {
        "h": h,
        "targets": torch.randint(100, (n,), generator=generator),
        "positions": torch.arange(n),
        "doc_ids": torch.zeros(n, dtype=torch.long),
        "seq_indices": torch.zeros(n, dtype=torch.long),
        "split": split,
        "immediate": immediate,
        "q_pi0": q_pi0,
        "baseline_action": baseline_action,
        "actions": enumerate_top2_actions(8),
        "rounds": ROUNDS,
        "backbone_hash": backbone_hash,
    }
    torch.save(data, directory / f"states_{split}.pt")
    return data


def make_students(directory: Path, seed: int = 0) -> None:
    payload = {}
    for index, name in enumerate(("immediate", "q_pi0")):
        torch.manual_seed(seed + index)
        student = ValueStudent(DIM, N_ACTIONS, 16)
        payload[name] = {
            "state_dict": student.state_dict(),
            "mean": torch.zeros(1, DIM),
            "std": torch.ones(1, DIM),
            "hidden_dim": 16,
            "history": [],
        }
    torch.save(payload, directory / "students.pt")


def make_input_dir(directory: Path, n_train: int = 32, n_eval: int = 24) -> dict:
    splits = {
        "train": make_split(directory, "train", n_train, seed=1),
        "validation": make_split(directory, "validation", n_eval, seed=2),
        "test": make_split(directory, "test", n_eval, seed=3),
    }
    make_students(directory)
    return splits


class EvaluateDecisionsTests(unittest.TestCase):
    def test_hand_computed_metrics(self):
        stop = torch.tensor([1.0, 2.0, 3.0, 4.0])
        cont = torch.tensor([0.5, 2.5, 3.0, 5.0])
        mask = torch.tensor([True, False, False, True])
        result = e03.evaluate_decisions(mask, stop, cont, rounds=4, top_k=2, fixed_t4_cost=cont)
        cost = torch.tensor([0.5, 2.0, 3.0, 5.0])
        self.assertAlmostEqual(result["mean_nll"]["mean"], cost.mean().item())
        self.assertAlmostEqual(result["mean_nll"]["se"], (cost.std(unbiased=False) / 2).item())
        self.assertAlmostEqual(result["mean_rounds"]["mean"], 2.5)
        self.assertAlmostEqual(result["mean_expert_calls"]["mean"], 5.0)
        self.assertAlmostEqual(result["stop_rate"], 0.5)
        # state 1 stopped although continuing was better (delta = -0.5 > 0? no: delta=stop-cont=-0.5)
        # premature stops: states where mask False and stop-cont > 0 -> state 2 (delta 0.0? stop-cont=0)
        # deltas: [0.5, -0.5, 0.0, -1.0]; stop states: 1, 2 -> premature only state 1? delta[1]=-0.5 no;
        # state 2 delta 0.0 not > 0 -> no premature stops here
        self.assertAlmostEqual(result["premature_stop"]["rate"], 0.0)
        self.assertIsNone(result["premature_stop"]["nll_lost"]["mean"])
        # wasteful: continue states 0, 3 with delta <= 0 -> state 3 only (delta -1.0)
        self.assertAlmostEqual(result["wasteful_continue"]["rate"], 0.25)
        self.assertAlmostEqual(result["wasteful_continue"]["excess_nll"]["mean"], 1.0)
        self.assertEqual(result["wasteful_continue"]["extra_expert_calls_each"], 6)
        delta_vs_t4 = cost - cont
        self.assertAlmostEqual(result["nll_delta_vs_fixed_t4"]["mean"], delta_vs_t4.mean().item())

    def test_premature_stop_accounting(self):
        stop = torch.tensor([2.0, 1.0])
        cont = torch.tensor([1.0, 1.0])
        mask = torch.tensor([False, False])
        result = e03.evaluate_decisions(mask, stop, cont, rounds=4, top_k=2, fixed_t4_cost=cont)
        self.assertAlmostEqual(result["premature_stop"]["rate"], 0.5)
        self.assertAlmostEqual(result["premature_stop"]["nll_lost"]["mean"], 1.0)
        self.assertAlmostEqual(result["wasteful_continue"]["rate"], 0.0)

    def test_mask_validation(self):
        stop = torch.zeros(3)
        with self.assertRaises(ValueError):
            e03.evaluate_decisions(torch.tensor([1, 0, 1]), stop, stop, rounds=4, top_k=2, fixed_t4_cost=stop)


class ThresholdTests(unittest.TestCase):
    def test_select_tau_respects_budget(self):
        frontier = [
            {"tau": -1.0, "mean_nll": 1.00, "mean_rounds": 4.0},
            {"tau": 0.0, "mean_nll": 1.02, "mean_rounds": 2.0},
            {"tau": 1.0, "mean_nll": 1.05, "mean_rounds": 1.5},
            {"tau": 2.0, "mean_nll": 1.50, "mean_rounds": 1.0},
        ]
        chosen = e03.select_tau(frontier, 2.0)
        self.assertEqual(chosen["tau"], 0.0)
        chosen = e03.select_tau(frontier, 1.5)
        self.assertEqual(chosen["tau"], 1.0)
        self.assertIsNone(e03.select_tau(frontier, 0.5))

    def test_select_tau_tie_prefers_fewer_rounds(self):
        frontier = [
            {"tau": 0.0, "mean_nll": 1.0, "mean_rounds": 2.0},
            {"tau": 1.0, "mean_nll": 1.0, "mean_rounds": 1.5},
        ]
        self.assertEqual(e03.select_tau(frontier, 2.0)["tau"], 1.0)

    def test_frontier_endpoints_are_degenerate(self):
        delta = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        stop = torch.ones(4)
        cont = torch.zeros(4)
        frontier = e03.calibration_frontier(delta, stop, cont, rounds=4, grid_size=8)
        rounds_seen = sorted({point["mean_rounds"] for point in frontier})
        self.assertEqual(rounds_seen[0], 1.0)
        self.assertEqual(rounds_seen[-1], 4.0)
        for point in frontier:
            expected = 1.0 + 3.0 * (delta.double() > point["tau"]).double().mean().item()
            self.assertAlmostEqual(point["mean_rounds"], expected)

    def test_calibration_bins(self):
        delta_hat = torch.arange(8, dtype=torch.float64)
        true_delta = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        bins = e03.calibration_bins(delta_hat, true_delta, 4)
        self.assertEqual(len(bins), 4)
        self.assertEqual([entry["count"] for entry in bins], [2, 2, 2, 2])
        self.assertAlmostEqual(bins[0]["predicted_delta_mean"], 0.5)
        self.assertAlmostEqual(bins[0]["realized_delta"]["mean"], 0.0)
        self.assertAlmostEqual(bins[-1]["realized_delta"]["mean"], 3.0)


class EndToEndTests(unittest.TestCase):
    def run_pilot(self, directory: Path, budgets=(1.5, 4.0), **overrides):
        splits = make_input_dir(directory)
        output = directory / "out"
        e03.main(["--input-dir", str(directory), "--output-dir", str(output),
                  "--budgets", *[str(budget) for budget in budgets]])
        summary = json.loads((output / "e03_summary.json").read_text())
        return splits, summary

    def test_end_to_end_fixed_policies_match_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            splits, summary = self.run_pilot(Path(temporary))
            test = splits["test"]
            rows = torch.arange(test["h"].shape[0])
            stop = test["immediate"][rows, test["baseline_action"]].double()
            cont = test["q_pi0"][rows, test["baseline_action"]].double()
            self.assertAlmostEqual(summary["fixed"]["T1"]["mean_nll"]["mean"], stop.mean().item())
            self.assertAlmostEqual(summary["fixed"]["T4"]["mean_nll"]["mean"], cont.mean().item())
            self.assertAlmostEqual(summary["fixed"]["T1"]["mean_rounds"]["mean"], 1.0)
            self.assertAlmostEqual(summary["fixed"]["T4"]["mean_expert_calls"]["mean"], 8.0)
            self.assertEqual(summary["backbone_hash"], HASH)
            self.assertTrue(summary["hash_verified_across_splits"])
            self.assertEqual(summary["nll_tolerance_nat"], 0.005)

    def test_threshold_frozen_on_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            splits, summary = self.run_pilot(Path(temporary))
            calibration = summary["calibration"]["value_student"]
            self.assertEqual(calibration["split"], "validation")
            for budget, entry in calibration["budgets"].items():
                if not entry["feasible"]:
                    continue
                self.assertTrue(entry["frozen_before_test"])
                self.assertLessEqual(entry["validation_mean_rounds"], float(budget) + 1e-12)
                chosen = e03.select_tau(calibration["frontier"], float(budget))
                self.assertAlmostEqual(entry["tau"], chosen["tau"])
            # the frozen validation tau must reproduce the reported test metrics
            test = splits["test"]
            rows = torch.arange(test["h"].shape[0])
            stop = test["immediate"][rows, test["baseline_action"]].double()
            cont = test["q_pi0"][rows, test["baseline_action"]].double()
            students = e03.load_students(Path(temporary))
            signal = e03.build_signals({"train": splits["train"], "validation": splits["validation"],
                                        "test": splits["test"]}, students)[0]["value_student"]
            for budget, entry in calibration["budgets"].items():
                if not entry["feasible"]:
                    continue
                mask = signal["test"] > entry["tau"]
                expected = torch.where(mask, cont, stop).mean().item()
                reported = summary["test"][f"value_student_budget_{budget}"]["mean_nll"]["mean"]
                self.assertAlmostEqual(reported, expected)

    def test_oracle_separated_and_dominates(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, summary = self.run_pilot(Path(temporary))
            self.assertFalse(summary["oracle"]["deployable"])
            self.assertIn("note", summary["oracle"])
            for key in summary["test"]:
                self.assertNotIn("oracle", key)
            oracle_nll = summary["oracle"]["oracle_tau_0"]["mean_nll"]["mean"]
            self.assertLessEqual(oracle_nll, summary["fixed"]["T1"]["mean_nll"]["mean"] + 1e-12)
            self.assertLessEqual(oracle_nll, summary["fixed"]["T4"]["mean_nll"]["mean"] + 1e-12)

    def test_entropy_not_fabricated(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, summary = self.run_pilot(Path(temporary))
            signals = summary["signals"]
            self.assertTrue(signals["value_student"]["available"])
            for name in ("entropy", "maxprob", "logit_margin", "hidden_change"):
                self.assertFalse(signals[name]["available"])
                self.assertTrue(signals[name]["reason"])
                for key in summary["test"]:
                    self.assertNotIn(name, key)

    def test_entropy_signal_used_when_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            splits = make_input_dir(directory)
            for split, states in splits.items():
                generator = torch.Generator().manual_seed(7)
                states["entropy"] = torch.rand(states["h"].shape[0], generator=generator)
                torch.save(states, directory / f"states_{split}.pt")
            output = directory / "out"
            e03.main(["--input-dir", str(directory), "--output-dir", str(output),
                      "--budgets", "2.5"])
            summary = json.loads((output / "e03_summary.json").read_text())
            self.assertTrue(summary["signals"]["entropy"]["available"])
            self.assertIn("entropy_budget_2.5", summary["test"])
            self.assertIn("entropy", summary["calibration"])

    def test_hash_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            make_input_dir(directory)
            make_split(directory, "test", 24, backbone_hash="b" * 64, seed=3)
            with self.assertRaises(ValueError):
                e03.load_artifacts(directory)

    def test_missing_split_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            make_input_dir(directory)
            (directory / "states_test.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                e03.load_artifacts(directory)

    def test_max_states_caps_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            make_input_dir(directory, n_train=32, n_eval=24)
            splits, _, _ = e03.load_artifacts(directory, max_states=10)
            for split in ("train", "validation", "test"):
                self.assertEqual(splits[split]["h"].shape[0], 10)
                self.assertEqual(splits[split]["immediate"].shape[0], 10)

    def test_tolerance_flag_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, summary = self.run_pilot(Path(temporary))
            for key, metrics in summary["test"].items():
                self.assertIn("nll_within_tolerance_vs_fixed_t4", metrics)
                delta = metrics["nll_delta_vs_fixed_t4"]["mean"]
                self.assertEqual(metrics["nll_within_tolerance_vs_fixed_t4"], delta <= 0.005)

    def test_calibration_curve_bins_cover_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            splits, summary = self.run_pilot(Path(temporary))
            curve = summary["calibration_curve_test"]
            total = sum(entry["count"] for entry in curve)
            self.assertEqual(total, splits["test"]["h"].shape[0])
            for entry in curve:
                self.assertIn("predicted_delta_mean", entry)
                self.assertIn("realized_delta", entry)
                self.assertIsNotNone(entry["realized_delta"]["se"])


if __name__ == "__main__":
    unittest.main()
