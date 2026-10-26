from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.e12_bidding import (
    ExpertBidAdapters,
    bidding_actions,
    load_states,
    match_adapter_hidden,
    adapter_parameter_count,
    central_parameter_count,
    pair_action_matrix,
    run_experiment,
)
from model.top2 import pair_to_action

BACKBONE_HASH = "ab" * 32


def make_states(split: str, n_states: int, h_dim: int, experts: int, seed: int, backbone_hash: str = BACKBONE_HASH) -> dict:
    generator = torch.Generator().manual_seed(seed)
    actions = torch.combinations(torch.arange(experts), 2)
    n_actions = actions.shape[0]
    h = torch.randn(n_states, h_dim, generator=generator)
    weights = torch.randn(h_dim, n_actions, generator=generator)
    q_pi0 = (h @ weights).square() + 0.1 * torch.randn(n_states, n_actions, generator=generator) + 3.0
    immediate = q_pi0 + 0.5 * torch.rand(n_states, n_actions, generator=generator)
    return {
        "h": h,
        "targets": torch.randint(100, (n_states,), generator=generator),
        "positions": torch.arange(n_states) % 64,
        "doc_ids": torch.arange(n_states) % 7,
        "seq_indices": torch.arange(n_states) % 11,
        "split": split,
        "immediate": immediate.float(),
        "q_pi0": q_pi0.float(),
        "baseline_action": torch.randint(n_actions, (n_states,), generator=generator),
        "actions": actions,
        "rounds": 4,
        "backbone_hash": backbone_hash,
    }


def write_input_dir(root: Path, h_dim: int = 16, experts: int = 4, backbone_hash: str = BACKBONE_HASH) -> Path:
    input_dir = root / "input"
    input_dir.mkdir(parents=True)
    sizes = {"train": 128, "validation": 32, "test": 32}
    for offset, (split, n) in enumerate(sizes.items()):
        torch.save(make_states(split, n, h_dim, experts, seed=100 + offset, backbone_hash=backbone_hash), input_dir / f"states_{split}.pt")
    return input_dir


class PairActionMatrixTests(unittest.TestCase):
    def test_matches_pair_to_action(self):
        experts = 8
        matrix = pair_action_matrix(experts)
        self.assertTrue((matrix.diagonal() == -1).all())
        for anchor in range(experts):
            for second in range(experts):
                if anchor == second:
                    continue
                expected = pair_to_action(torch.tensor([[anchor, second]]), experts=experts).item()
                self.assertEqual(matrix[anchor, second].item(), expected)
                self.assertEqual(matrix[second, anchor].item(), expected)

    def test_covers_all_actions(self):
        experts = 4
        matrix = pair_action_matrix(experts)
        n_actions = experts * (experts - 1) // 2
        covered = sorted(matrix[matrix >= 0].tolist())
        self.assertEqual(covered, [action for action in range(n_actions) for _ in range(2)])


class HiddenMatchingTests(unittest.TestCase):
    def test_matched_hidden_within_five_percent(self):
        central = central_parameter_count(512, 128, 28)
        width = match_adapter_hidden(central, 528, 8)
        total = 8 * adapter_parameter_count(528, width)
        self.assertLess(abs(total - central) / central, 0.05)

    def test_matched_hidden_at_least_one(self):
        width = match_adapter_hidden(central_parameter_count(16, 4, 6), 20, 4)
        self.assertGreaterEqual(width, 1)


class BiddingSelectionTests(unittest.TestCase):
    def test_selection_contains_anchor_and_excludes_self(self):
        torch.manual_seed(0)
        experts, h_dim = 4, 8
        adapters = ExpertBidAdapters(h_dim, experts, hidden_dim=4)
        h = torch.randn(16, h_dim)
        mean, std = h.mean(dim=0, keepdim=True), h.std(dim=0, keepdim=True)
        anchors = torch.randint(experts, (16,))
        pa = pair_action_matrix(experts)
        selected, seconds = bidding_actions(adapters, h, mean, std, anchors, pa)
        self.assertFalse((seconds == anchors).any())
        pairs = torch.stack([torch.minimum(anchors, seconds), torch.maximum(anchors, seconds)], dim=-1)
        self.assertTrue((selected == pair_to_action(pairs, experts=experts)).all())

    def test_summary_modes_change_input_dim(self):
        plain = ExpertBidAdapters(8, 4, 4, summary_mode="none")
        learned = ExpertBidAdapters(8, 4, 4, summary_mode="learned", summary_dim=3)
        fixed = ExpertBidAdapters(8, 4, 4, summary_mode="checkpoint", checkpoint_summary=torch.randn(4, 5))
        self.assertEqual(plain.input_dim, 8 + 4)
        self.assertEqual(learned.input_dim, 8 + 4 + 3)
        self.assertEqual(fixed.input_dim, 8 + 4 + 5)
        bids = learned.bids_all_anchors(torch.randn(6, 8))
        self.assertEqual(tuple(bids.shape), (6, 4, 4))


class AuditCounterexampleTests(unittest.TestCase):
    """Counterexamples from the 2026-10-25 code audit (finding F12).

    Both tests fail on the pre-repair implementation: the first because the
    first-layer einsum summed the expert axis, the second because the
    ``_apply`` method overrode ``nn.Module._apply`` and broke ``.to()``.
    """

    def test_each_expert_reads_only_its_own_summary(self):
        adapters = ExpertBidAdapters(
            1, 2, 1, summary_mode="checkpoint", checkpoint_summary=torch.tensor([[2.0], [7.0]])
        )
        with torch.no_grad():
            adapters.w1.zero_()
            adapters.w1[:, -1, 0] = torch.tensor([3.0, 5.0])
            adapters.w2.fill_(1.0)
            adapters.w3.fill_(1.0)
        h = torch.zeros(1, 1)
        anchors = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            before = adapters.bids_for_anchor(h, anchors)
        expected = F.gelu(F.gelu(torch.tensor([[6.0, 35.0]])))
        torch.testing.assert_close(before, expected)
        with torch.no_grad():
            adapters.summary[1, 0] += 1.0
            after = adapters.bids_for_anchor(h, anchors)
        torch.testing.assert_close(before[:, 0], after[:, 0])
        self.assertNotEqual(float(before[0, 1]), float(after[0, 1]))

    def test_module_to_and_dtype_conversion_remain_available(self):
        adapters = ExpertBidAdapters(2, 3, 4)
        moved = adapters.to(device="cpu", dtype=torch.float64)
        self.assertIs(moved, adapters)
        self.assertTrue(all(p.dtype == torch.float64 for p in adapters.parameters()))
        output = adapters.bids_for_anchor(torch.zeros(2, 2, dtype=torch.float64), torch.tensor([0, 1]))
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(tuple(output.shape), (2, 3))


class CentralRestrictedAlignmentTests(unittest.TestCase):
    """The restricted central deployment must face the exact bidding contract."""

    def test_restricted_selection_matches_anchor_candidates_and_argmin(self):
        from experiments.e12_bidding import central_restricted_actions

        torch.manual_seed(0)
        experts = 4
        pa = pair_action_matrix(experts)
        n_actions = experts * (experts - 1) // 2
        predictions = torch.randn(16, n_actions)
        anchors = torch.randint(experts, (16,))
        selected, seconds = central_restricted_actions(predictions, anchors, pa)
        self.assertFalse((seconds == anchors).any())
        pairs = torch.stack([torch.minimum(anchors, seconds), torch.maximum(anchors, seconds)], dim=-1)
        self.assertTrue((selected == pair_to_action(pairs, experts=experts)).all())
        for i in range(16):
            cells = pa[anchors[i]]
            valid = cells >= 0
            self.assertEqual(int(valid.sum()), experts - 1)
            expected = valid.nonzero().flatten()[predictions[i][cells[valid]].argmin()]
            self.assertEqual(seconds[i].item(), expected.item())

    def test_restricted_differs_from_unrestricted_only_in_candidate_space(self):
        from experiments.e12_bidding import central_restricted_actions

        experts = 4
        pa = pair_action_matrix(experts)
        n_actions = experts * (experts - 1) // 2
        predictions = torch.zeros(1, n_actions)
        anchors = torch.tensor([2])
        # put the global minimum on a pair not containing the anchor
        outside = pair_to_action(torch.tensor([[0, 1]]), experts=experts).item()
        predictions[0, outside] = -1.0
        selected, seconds = central_restricted_actions(predictions, anchors, pa)
        self.assertNotEqual(selected.item(), outside)
        self.assertNotEqual(seconds.item(), anchors.item())
        self.assertEqual(predictions.argmin(dim=-1).item(), outside)


class LoadStatesTests(unittest.TestCase):
    def test_hash_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as root:
            input_dir = write_input_dir(Path(root))
            states = make_states("test", 8, 16, 4, seed=5, backbone_hash="ff" * 32)
            torch.save(states, input_dir / "states_test.pt")
            with self.assertRaises(ValueError):
                load_states(input_dir)

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                load_states(root)


class EndToEndTests(unittest.TestCase):
    def run_pipeline(self, root: Path) -> dict:
        input_dir = write_input_dir(root / "case")
        output_dir = root / "case" / "output"
        return run_experiment(input_dir, output_dir, hidden_dim=16, epochs=2, batch_size=32, seed=0)

    def test_end_to_end_summary(self):
        with tempfile.TemporaryDirectory() as root:
            summary = self.run_pipeline(Path(root))
            written = Path(root) / "case" / "output" / "e12_bidding_summary.json"
            self.assertTrue(written.exists())
            loaded = json.loads(written.read_text())
            self.assertEqual(loaded["backbone_hash"], BACKBONE_HASH)
            for arm in ("central", "bidding", "bidding_summary", "central_irrelevant"):
                self.assertIn(arm, loaded["systems"])
            for key in ("bidding_on_same_cells", "bidding_summary_on_same_cells", "central_on_same_cells"):
                self.assertTrue(torch.isfinite(torch.tensor(loaded["mse_comparison"][key])))
            for policy in ("baseline", "random", "central_argmin", "bidding_learned_anchor",
                           "bidding_baseline_pair_first_anchor", "bidding_summary_learned_anchor"):
                self.assertIn(policy, loaded["policies"])
            for policy in ("central_restricted_learned_anchor", "central_restricted_baseline_pair_first_anchor",
                           "bidding_summary_baseline_pair_first_anchor"):
                self.assertIn(policy, loaded["policies"])
            self.assertIn("unrestricted", loaded["deployment_alignment"]["unrestricted_central"])
            self.assertEqual(loaded["training_budget"]["weight_decay"], 0.0)
            for arm in ("bidding", "bidding_summary"):
                adapter_file = Path(root) / "case" / "output" / f"adapters_{arm}.pt"
                self.assertTrue(adapter_file.exists())
                saved = torch.load(adapter_file, map_location="cpu", weights_only=False)
                self.assertIn("state_dict", saved)
                self.assertEqual(saved["config"]["backbone_hash"], BACKBONE_HASH)
                self.assertEqual(saved["config"]["training_budget"]["weight_decay"], 0.0)
            self.assertLess(loaded["policies"]["central_argmin"]["mean_cost"], loaded["policies"]["random"]["mean_cost"])
            matching = loaded["parameter_matching"]["bidding"]
            self.assertLess(abs(matching["relative_difference"]), 0.2)
            self.assertEqual(loaded["control_overhead"]["bids_collected_per_state"]["bidding"], 3)
            self.assertEqual(
                loaded["systems"]["central"]["optimizer_updates"],
                loaded["systems"]["bidding"]["optimizer_updates"],
            )
            self.assertIn("no incentive-compatibility claim", loaded["scope"])

    def test_deterministic_given_seed(self):
        with tempfile.TemporaryDirectory() as root:
            first = self.run_pipeline(Path(root) / "a")
            second = self.run_pipeline(Path(root) / "b")
            self.assertEqual(
                first["systems"]["central"]["test_mse_all_candidates"],
                second["systems"]["central"]["test_mse_all_candidates"],
            )
            self.assertEqual(
                first["mse_comparison"]["bidding_on_same_cells"],
                second["mse_comparison"]["bidding_on_same_cells"],
            )

    def test_checkpoint_summary_mode_requires_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            input_dir = write_input_dir(Path(root) / "case")
            with self.assertRaises(ValueError):
                run_experiment(input_dir, None, hidden_dim=16, epochs=1, batch_size=32, summary_mode="checkpoint")


if __name__ == "__main__":
    unittest.main()
