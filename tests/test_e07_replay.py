"""Unit tests for experiments/e07_replay.py (E07 offline trace replay)."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import e07_replay as replay
from model.top2 import enumerate_top2_actions
from train.value_student import ValueStudent

DIM = 8
N_ACTIONS = 28
ROUNDS = 4
HASH = "b" * 64


def make_split(directory: Path, split: str, n: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    h = torch.randn(n, DIM, generator=generator)
    baseline_action = torch.randint(N_ACTIONS, (n,), generator=generator)
    immediate = 6.0 + 0.1 * torch.randn(n, N_ACTIONS, generator=generator)
    true_delta = 0.05 * torch.randn(n, generator=generator)
    q_pi0 = immediate + 0.01 * torch.randn(n, N_ACTIONS, generator=generator)
    rows = torch.arange(n)
    q_pi0[rows, baseline_action] = immediate[rows, baseline_action] - true_delta
    torch.save({
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
        "backbone_hash": HASH,
    }, directory / f"states_{split}.pt")


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


def make_input_dir(directory: Path, n_states: int = 48) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    make_split(directory, "train", n_states, seed=1)
    make_split(directory, "validation", n_states, seed=2)
    make_split(directory, "test", n_states, seed=3)
    make_students(directory)
    (directory / "e03_summary.json").write_text(json.dumps({"backbone_hash": HASH}))
    return directory


def tiny_run_kwargs(directory: Path) -> dict:
    return {
        "input_dir": directory,
        "e03_summary_path": directory / "e03_summary.json",
        "window_size": 16,
        "budget_rounds_tiers": (1.5, 2.0, 3.0, 4.0),
        "seed": 5,
        "dp_window": 4,
        "n_dp_instances": 2,
        "n_reorder_seeds": 2,
        "batch_sizes": (1, 4),
        "top_k": 2,
    }


class TestProfiles(unittest.TestCase):
    def test_weights_sum_to_one(self):
        for profile in replay.PROFILES:
            for n_rounds in (1, 2, 3, 5):
                weights = replay.profile_weights(n_rounds, profile)
                self.assertEqual(len(weights), n_rounds)
                self.assertAlmostEqual(sum(weights), 1.0, places=12)

    def test_profile_shapes(self):
        diminishing = replay.profile_weights(3, "diminishing")
        delayed = replay.profile_weights(3, "delayed")
        self.assertGreater(diminishing[0], diminishing[-1])
        self.assertLess(delayed[0], delayed[-1])
        self.assertAlmostEqual(diminishing[0], delayed[-1], places=12)

    def test_trajectory_anchors_endpoint(self):
        for delta in (0.7, -0.4, 0.0):
            for profile in replay.PROFILES:
                gains = replay.trajectory(delta, 3, profile)
                self.assertAlmostEqual(sum(gains), delta, places=12)

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):
            replay.profile_weights(3, "flat")


class TestAllocators(unittest.TestCase):
    def setUp(self):
        self.deltas = [[0.5, 0.3, 0.1], [0.4, 0.2, 0.2], [0.1, 0.05, 0.05]]

    def test_budget_and_horizon_respected(self):
        rng = random.Random(3)
        for budget in (0, 1, 3, 7, 9, 100):
            for name, (_, state, used, _) in replay.run_offline_allocators(
                    self.deltas, budget, rng, dp_window=2).items():
                self.assertLessEqual(sum(state), budget, name)
                self.assertEqual(sum(state), used, name)
                for i, r in enumerate(state):
                    self.assertLessEqual(r, len(self.deltas[i]), name)

    def test_no_forced_spending_on_nonpositive_gain(self):
        deltas = [[-0.3, -0.2, -0.1], [-0.5, -0.4, -0.4]]
        rng = random.Random(1)
        results = replay.run_offline_allocators(deltas, 6, rng, dp_window=2)
        for name in ("single_step_index", "two_step_package", "tail_index", "rolling_window_dp"):
            self.assertEqual(results[name][2], 0, name)
        # blind baselines still spend; that is their role as controls
        self.assertGreater(results["equal_depth"][2], 0)
        self.assertGreater(results["random"][2], 0)

    def test_equal_depth_is_max_min_fair(self):
        _, state, used, _ = replay.equal_depth_allocation(self.deltas, 7)
        self.assertEqual(used, 7)
        self.assertLessEqual(max(state) - min(state), 1)

    def test_two_step_package_reserves_two_rounds(self):
        deltas = [[1.0, 1.0, 0.01], [0.05, 0.05, 0.05]]
        _, state, used, steps = replay.two_step_package_allocation(deltas, 2)
        self.assertEqual(state[0], 2)
        self.assertEqual(used, 2)
        self.assertEqual([s["round"] for s in steps], [1, 2])

    def test_tail_index_indivisible(self):
        _, state, used, _ = replay.tail_index_allocation(self.deltas, 4)
        for r in state:
            self.assertIn(r, (0, 3))
        self.assertEqual(used, sum(state))

    def test_steps_cover_allocation(self):
        rng = random.Random(9)
        for name, (_, state, used, steps) in replay.run_offline_allocators(
                self.deltas, 5, rng, dp_window=2).items():
            self.assertEqual(len(steps), used, name)
            for i, r in enumerate(state):
                self.assertEqual(sum(1 for s in steps if s["request"] == i), r, name)
            self.assertEqual([s["epoch"] for s in steps], list(range(used)), name)

    def test_random_allocator_deterministic_per_seed(self):
        a = replay.random_allocation(self.deltas, 4, random.Random(7))
        b = replay.random_allocation(self.deltas, 4, random.Random(7))
        self.assertEqual(a[1], b[1])

    def test_rolling_dp_matches_exact_on_single_chunk(self):
        rng = random.Random(11)
        deltas = [[0.9, 0.1, 0.0], [0.3, 0.3, 0.3]]
        _, state, _, _ = replay.rolling_window_dp_allocation(deltas, 3, dp_window=2)
        from experiments import e07_synthetic as e07syn
        opt, opt_rounds = e07syn.exact_dp(deltas, 3)
        realized = sum(deltas[i][h] for i in range(2) for h in range(state[i]))
        self.assertAlmostEqual(realized, opt, places=9)
        self.assertLessEqual(sum(state), 3)


class TestOnlineLedger(unittest.TestCase):
    def make_window(self, n=12):
        rng = random.Random(17)
        return {
            "rounds": ROUNDS,
            "extra_rounds": ROUNDS - 1,
            "requests": [
                {"state_index": i, "arrival": i, "stop_cost": 6.0,
                 "continue_cost": 6.0 - rng.uniform(-0.2, 0.5),
                 "true_delta": 0.0, "pred_delta": 0.0,
                 "position": i, "doc_id": 0}
                for i in range(n)
            ],
        }

    def test_prefix_invariant_every_prefix_and_batch(self):
        window = self.make_window()
        n = len(window["requests"])
        rng = random.Random(19)
        pred = [[rng.uniform(-0.1, 0.4) for _ in range(3)] for _ in range(n)]
        true = [[rng.uniform(-0.1, 0.4) for _ in range(3)] for _ in range(n)]
        for budget in (0, 2, 6, 12, 36):
            for batch in (1, 3, 12):
                entries, violations, state, _ = replay.online_ledger_replay(
                    window, pred, true, budget, 2, batch_size=batch)
                self.assertEqual(violations, [])
                for e in entries:
                    self.assertLessEqual(e["cum_executed"], e["cum_supply"])
                    self.assertGreaterEqual(e["balance"], 0)
                self.assertLessEqual(sum(state) * 2, budget * 2)

    def test_credit_schedule_sums_to_budget(self):
        schedule = replay.credit_schedule(64, 100)
        self.assertEqual(len(schedule), 64)
        self.assertEqual(sum(schedule), 100)
        self.assertLessEqual(max(schedule) - min(schedule), 1)

    def test_zero_budget_executes_nothing(self):
        window = self.make_window()
        pred = [[1.0, 1.0, 1.0] for _ in window["requests"]]
        entries, violations, state, steps = replay.online_ledger_replay(
            window, pred, pred, 0, 2)
        self.assertEqual(sum(state), 0)
        self.assertEqual(steps, [])
        self.assertEqual(violations, [])


class TestCompletionAndStratification(unittest.TestCase):
    def test_completion_epochs_and_ranks(self):
        window = {
            "rounds": ROUNDS, "extra_rounds": 3,
            "requests": [
                {"state_index": i, "arrival": i, "stop_cost": 6.0, "continue_cost": 5.0,
                 "true_delta": 1.0 - 0.1 * i, "pred_delta": 1.0 - 0.1 * i,
                 "position": 10 * i, "doc_id": 0}
                for i in range(8)
            ],
        }
        true = [[1.0, 0.5, 0.25] for _ in range(8)]
        state = [3, 0, 3, 0, 3, 0, 3, 0]
        steps = [{"request": i, "round": r, "gain": 0.1, "epoch": e}
                 for e, (i, r) in enumerate(
                     [(0, 1), (2, 1), (4, 1), (6, 1), (0, 2), (2, 2), (4, 2), (6, 2),
                      (0, 3), (2, 3), (4, 3), (6, 3)])]
        metrics = replay.evaluate_allocation(window, true, state, steps, 2)
        completion = metrics["completion"]
        self.assertFalse(completion["arrival_aligned"])
        self.assertIn("waiting_note", completion)
        self.assertNotIn("mean_waiting_epochs", completion)
        rows = {r["request"]: r for r in completion["per_request"]}
        self.assertEqual(rows[1]["completion_epoch"], 1)  # no extra rounds: completes on arrival
        self.assertEqual(rows[0]["completion_epoch"], 8)
        self.assertEqual(completion["order"], [1, 3, 5, 7, 0, 2, 4, 6])
        self.assertIn("not wall-clock", completion["units"])
        self.assertLess(completion["mean_completion_rank_top_gain_quartile"],
                        completion["mean_completion_rank_bottom_gain_quartile"])
        strat = metrics["stratified_nll"]
        self.assertTrue(strat["available"])
        self.assertEqual(len(strat["quartile_mean_nll"]), 4)

    def test_online_completion_waiting_nonnegative(self):
        window = {
            "rounds": ROUNDS, "extra_rounds": 3,
            "requests": [
                {"state_index": i, "arrival": i, "stop_cost": 6.0, "continue_cost": 5.0,
                 "true_delta": 0.3, "pred_delta": 0.3, "position": i, "doc_id": 0}
                for i in range(10)
            ],
        }
        deltas = [[0.2, 0.1, 0.05] for _ in range(10)]
        order = [3, 7, 1, 9, 0, 2, 4, 5, 6, 8]
        _, violations, state, steps = replay.online_ledger_replay(
            window, deltas, deltas, 10, 2, arrival_order=order, batch_size=2)
        self.assertEqual(violations, [])
        stream = replay.stream_aligned_window(window, order)
        metrics = replay.evaluate_allocation(stream, deltas, state, steps, 2,
                                             arrival_aligned=True)
        completion = metrics["completion"]
        self.assertTrue(completion["arrival_aligned"])
        for r in completion["per_request"]:
            self.assertGreaterEqual(r["waiting_epochs"], 0)
        self.assertIn("arrival steps", completion["units"])

    def test_stratification_unavailable_without_positions(self):
        requests = [{"position": None, "stop_cost": 1.0} for _ in range(4)]
        strat = replay.stratified_nll(requests, [1.0] * 4)
        self.assertFalse(strat["available"])


class TestWindowLoading(unittest.TestCase):
    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            directory = make_input_dir(Path(d))
            bad = Path(d) / "bad_summary.json"
            bad.write_text(json.dumps({"backbone_hash": "c" * 64}))
            with self.assertRaises(ValueError):
                replay.load_window(directory, bad, 8, 0)

    def test_window_uses_only_legal_signals(self):
        with tempfile.TemporaryDirectory() as d:
            directory = make_input_dir(Path(d))
            window = replay.load_window(directory, directory / "e03_summary.json", 16, 0)
            self.assertEqual(len(window["requests"]), 16)
            self.assertEqual(window["extra_rounds"], ROUNDS - 1)
            indices = [r["state_index"] for r in window["requests"]]
            self.assertEqual(len(set(indices)), 16)
            for r in window["requests"]:
                self.assertAlmostEqual(r["true_delta"],
                                       r["stop_cost"] - r["continue_cost"], places=9)


class TestCLI(unittest.TestCase):
    def run_tiny(self, directory):
        out_dir = Path(directory) / "out"
        input_dir = make_input_dir(Path(directory) / "artifacts")
        kwargs = tiny_run_kwargs(input_dir)
        replay.main([
            "--input-dir", str(kwargs["input_dir"]),
            "--e03-summary", str(kwargs["e03_summary_path"]),
            "--output-dir", str(out_dir),
            "--window-size", str(kwargs["window_size"]),
            "--budget-rounds", "1.5", "2", "3", "4",
            "--seed", str(kwargs["seed"]),
            "--dp-window", str(kwargs["dp_window"]),
            "--n-dp-instances", str(kwargs["n_dp_instances"]),
            "--n-reorder-seeds", str(kwargs["n_reorder_seeds"]),
            "--batch-sizes", "1", "4",
        ])
        with open(out_dir / "e07_replay.json") as f:
            return json.load(f)

    def test_end_to_end_json(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.run_tiny(d)
        self.assertTrue(result["offline_replay"])
        self.assertIn("expert calls", result["latency_note"])
        self.assertIn("not", result["latency_note"])
        self.assertEqual(result["input"]["window_size"], 16)
        self.assertTrue(result["input"]["hash_verified_against_e03_summary"])
        for profile in replay.PROFILES:
            block = result["profiles"][profile]
            self.assertIn("not measured", block["intermediate_depth_note"])
            self.assertEqual(len(block["budget_tiers"]), 4)
            for tier in block["budget_tiers"].values():
                alloc = tier["allocators"]
                for name in ("equal_depth", "random", "single_step_index", "two_step_package",
                             "tail_index", "rolling_window_dp", "online_single_step_index"):
                    self.assertIn(name, alloc)
                ledger = alloc["online_single_step_index"]["hard_ledger"]
                self.assertEqual(ledger["violation_count"], 0)
                for e in ledger["entries"]:
                    self.assertLessEqual(e["cum_executed"], e["cum_supply"])
                for name in ("equal_depth", "single_step_index", "rolling_window_dp"):
                    self.assertIn("completion", alloc[name])
                    self.assertIn("order", alloc[name]["completion"])
                    self.assertIn("quartile_mean_nll", alloc[name]["stratified_nll"])
                self.assertLessEqual(alloc["single_step_index"]["used_extra_rounds"],
                                     tier["extra_rounds_budget"])
            dp = block["small_window_exact_dp_gap"]
            self.assertEqual(len(dp["per_tier"]), 4)
            self.assertIn("approximation", dp["upper_bound_note"])
            sensitivity = block["sensitivity"]
            self.assertIn("longest_first", sensitivity["reorder"]["per_ordering"])
            self.assertIn("grouped_by_doc", sensitivity["reorder"]["per_ordering"])
            for row in sensitivity["reorder"]["per_ordering"].values():
                self.assertEqual(row["ledger_violations"], [])
            self.assertEqual(set(sensitivity["batch_size"]["per_batch_size"]), {"1", "4"})
        # no wall-clock/latency metric may appear anywhere
        forbidden = ("wall_clock", "latency_ms", "throughput_tokens_per_sec", "gpu_seconds")
        self.assertFalse(any(key in json.dumps(result) for key in forbidden))

    def test_seed_reproducible(self):
        with tempfile.TemporaryDirectory() as d:
            input_dir = make_input_dir(Path(d))
            kwargs = tiny_run_kwargs(input_dir)
            self.assertEqual(replay.run(**kwargs), replay.run(**kwargs))


if __name__ == "__main__":
    unittest.main()
