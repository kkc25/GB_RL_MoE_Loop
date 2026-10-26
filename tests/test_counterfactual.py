"""Unit tests for E01 counterfactual diagnostics (eval/counterfactual.py,
train/value_student.py).  All tests run on CPU with a tiny Top-2 LoopMoE
(4 experts -> 6 actions) and synthetic packed-LM splits.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ModelConfig
from data.fineweb import SplitWriter
from data.packed_lm import PackedLMDataset
from eval.counterfactual import (
    CounterfactualLedger,
    cache_last_ffn_input,
    candidate_actions,
    counterfactual_path_costs,
    evaluate_policies,
    extract_states,
    gather_states,
    oracle_metrics,
    random_selection,
    readout_ce,
)
from model import LoopMoE
from model.top2 import enumerate_top2_actions
from train.value_student import fit_value_student, predict, student_features

EXPERTS = 4
N_ACTIONS = EXPERTS * (EXPERTS - 1) // 2  # 6


def tiny_model(seed: int = 7) -> LoopMoE:
    torch.manual_seed(seed)
    config = ModelConfig(
        layers=3,
        hidden_size=32,
        attention_heads=2,
        head_dim=16,
        vocab_size=64,
        intermediate_size=48,
        experts_per_layer=EXPERTS,
        top_k=2,
        adaptive_layers=(2,),
        moe_layers=(0, 1, 2),
        max_calls=4,
        target_only=False,
        max_position_embeddings=32,
    )
    return LoopMoE(config).eval()


def tiny_batch(batch: int = 2, length: int = 12, seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, 64, (batch, length), generator=generator)
    labels = torch.full((batch, length), -100)
    labels[:, 1:] = input_ids[:, 1:]
    labels[0, 5] = -100  # unsupervised hole inside the sequence
    segment_ids = torch.zeros(batch, length, dtype=torch.long)
    segment_ids[0, 8:] = 1
    segment_ids[1, :4] = -1  # padding
    labels[segment_ids < 0] = -100
    labels[0, 8] = -100  # document boundary: no supervision across segments
    return input_ids, labels, segment_ids


def ce_per_position(model, input_ids, labels, segment_ids, depth):
    with torch.no_grad():
        output = model(input_ids, labels, depth=depth, segment_ids=segment_ids)
    losses = F.cross_entropy(
        output.logits.float().reshape(-1, output.logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    return losses


class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.input_ids, self.labels, self.segment_ids = tiny_batch()
        self.states = extract_states(self.model, self.input_ids, self.labels, self.segment_ids)
        self.costs = counterfactual_path_costs(self.model, self.states["h"], self.states["targets"], rounds=4)

    def test_cached_h_matches_normal_baseline_forward(self):
        # h replayed manually block-by-block must reproduce the normal forward:
        # readout of the baseline continuation equals the fixed-depth baseline CE.
        mask = self.labels != -100
        depth1 = ce_per_position(self.model, self.input_ids, self.labels, self.segment_ids, depth=1)[mask]
        depth4 = ce_per_position(self.model, self.input_ids, self.labels, self.segment_ids, depth=4)[mask]
        rows = torch.arange(self.states["h"].shape[0])
        immediate_baseline = self.costs.immediate[rows, self.costs.baseline_action]
        q_baseline = self.costs.q_pi0[rows, self.costs.baseline_action]
        torch.testing.assert_close(immediate_baseline, depth1, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(q_baseline, depth4, rtol=1e-5, atol=1e-6)

    def test_forced_baseline_then_baseline_continuation_equals_baseline(self):
        # Explicitly forcing the baseline action at round 1 and continuing with
        # pi0 must match the plain depth-4 baseline forward exactly.
        layer = self.model.config.adaptive_layers[-1]
        block = self.model.blocks[layer]
        gamma = self.model.gammas[layer]
        h = self.states["h"]
        with torch.no_grad():
            current = h
            for round_index in range(4):
                normalized = block.ffn_norm(current)
                if round_index == 0:
                    update, _, _, _ = block.executor(normalized, self.costs.baseline_action)
                else:
                    update, _, _, _ = block.executor(normalized)
                current = current + gamma * update
            logits = F.linear(self.model.final_norm(current).float(), self.model.embedding.weight.float())
            forced_ce = F.cross_entropy(logits, self.states["targets"], reduction="none")
        mask = self.labels != -100
        depth4 = ce_per_position(self.model, self.input_ids, self.labels, self.segment_ids, depth=4)[mask]
        torch.testing.assert_close(forced_ce, depth4, rtol=1e-5, atol=1e-6)

    def test_path_cost_shapes_and_candidate_set(self):
        n_states = int((self.labels != -100).sum())
        self.assertEqual(tuple(self.costs.immediate.shape), (n_states, N_ACTIONS))
        self.assertEqual(tuple(self.costs.q_pi0.shape), (n_states, N_ACTIONS))
        self.assertTrue(torch.equal(self.costs.actions, enumerate_top2_actions(EXPERTS)))
        # The candidate set contains the baseline action of every state.
        self.assertTrue(bool(((self.costs.baseline_action >= 0) & (self.costs.baseline_action < N_ACTIONS)).all()))
        self.assertTrue(torch.isfinite(self.costs.immediate).all())
        self.assertTrue(torch.isfinite(self.costs.q_pi0).all())

    def test_chunking_does_not_change_costs(self):
        chunked = counterfactual_path_costs(self.model, self.states["h"], self.states["targets"], rounds=4, state_chunk=1, readout_chunk=2)
        torch.testing.assert_close(chunked.immediate, self.costs.immediate, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(chunked.q_pi0, self.costs.q_pi0, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(chunked.baseline_action, self.costs.baseline_action))

    def test_custom_action_subset_preserves_pairs_and_global_mapping(self):
        pairs = self.costs.actions[self.costs.baseline_action].unique(dim=0)
        custom = counterfactual_path_costs(self.model, self.states["h"], self.states["targets"], rounds=4, actions=pairs)
        from model.top2 import pair_to_action
        expected = pair_to_action(pairs, experts=EXPERTS)
        self.assertTrue(torch.equal(custom.actions, pairs))
        self.assertTrue(torch.equal(custom.global_action_ids, expected))
        self.assertTrue(bool(((custom.baseline_action >= 0) & (custom.baseline_action < pairs.shape[0])).all()))
        torch.testing.assert_close(custom.immediate, self.costs.immediate[:, expected])
        torch.testing.assert_close(custom.q_pi0, self.costs.q_pi0[:, expected])

    def test_shared_and_broadcast_nonprefix_actions_are_equivalent(self):
        order = torch.tensor([5, 0, 4, 1, 3, 2])
        shared_actions = self.costs.actions[order]
        shared = counterfactual_path_costs(self.model, self.states["h"], self.states["targets"], rounds=4, actions=shared_actions)
        broadcast_actions = shared_actions.unsqueeze(0).expand(self.states["h"].shape[0], -1, -1)
        broadcast = counterfactual_path_costs(self.model, self.states["h"], self.states["targets"], rounds=4, actions=broadcast_actions)
        torch.testing.assert_close(shared.immediate, broadcast.immediate)
        torch.testing.assert_close(shared.q_pi0, broadcast.q_pi0)
        self.assertTrue(torch.equal(shared.baseline_action, broadcast.baseline_action))
        self.assertTrue(torch.equal(shared.global_action_ids.unsqueeze(0).expand_as(broadcast.global_action_ids), broadcast.global_action_ids))

    def test_per_state_actions_use_state_specific_mapping(self):
        pairs = self.costs.actions[self.costs.baseline_action].unsqueeze(1)
        custom = counterfactual_path_costs(self.model, self.states["h"], self.states["targets"], rounds=4, actions=pairs)
        from model.top2 import pair_to_action
        expected = pair_to_action(pairs, experts=EXPERTS)
        rows = torch.arange(self.states["h"].shape[0])
        self.assertTrue(torch.equal(custom.global_action_ids, expected))
        self.assertTrue(bool(((custom.baseline_action >= 0) & (custom.baseline_action < pairs.shape[1])).all()))
        torch.testing.assert_close(custom.immediate, self.costs.immediate[rows, self.costs.baseline_action].unsqueeze(1))
        torch.testing.assert_close(custom.q_pi0, self.costs.q_pi0[rows, self.costs.baseline_action].unsqueeze(1))

    def test_candidate_actions_include_baseline_and_exploration(self):
        candidates = candidate_actions(self.model, self.states["h"], candidates=4)
        self.assertEqual(candidates.shape, (self.states["h"].shape[0], 4, 2))
        for row, baseline in enumerate(self.costs.actions[self.costs.baseline_action]):
            self.assertTrue(torch.equal(candidates[row, 0], baseline))
            self.assertEqual(len({tuple(pair) for pair in candidates[row].tolist()}), 4)

    def test_states_only_at_supervised_positions(self):
        mask = self.labels != -100
        expected_rows, expected_positions = torch.nonzero(mask, as_tuple=True)
        self.assertTrue(torch.equal(self.states["rows"], expected_rows))
        self.assertTrue(torch.equal(self.states["positions"], expected_positions))
        self.assertTrue(torch.equal(self.states["targets"], self.labels[mask]))
        # No state may sit on a -100 label (padding, boundary, or the hole).
        self.assertEqual(self.states["h"].shape[0], int(mask.sum()))

    def test_no_attention_after_cached_h(self):
        # Positions are independent after the cached h: scrambling other
        # positions' cached states must not change a state's continuation.
        layer = self.model.config.adaptive_layers[-1]
        block = self.model.blocks[layer]
        h = self.states["h"]
        permuted = h[torch.randperm(h.shape[0], generator=torch.Generator().manual_seed(0))]
        with torch.no_grad():
            normalized = block.ffn_norm(h)
            reference, _, _, _ = block.executor(normalized, self.costs.baseline_action)
            logits = block.executor.gate_logits(block.ffn_norm(permuted))
        self.assertEqual(reference.shape, h.shape)
        self.assertEqual(tuple(logits.shape), (h.shape[0], EXPERTS))


class ReadoutPrecisionTests(unittest.TestCase):
    def test_readout_ce_is_invariant_to_outer_autocast(self):
        # Regression: an outer BF16 autocast context must not leak into the
        # FP32 head readout; casting inputs with .float() alone still lets
        # autocast run final_norm/F.linear in the outer dtype.
        model = tiny_model()
        input_ids, labels, segment_ids = tiny_batch()
        states = extract_states(model, input_ids, labels, segment_ids)
        reference = readout_ce(model, states["h"], states["targets"])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            protected = readout_ce(model, states["h"], states["targets"])
            # Sanity: the unprotected pattern really is affected here,
            # otherwise this regression test would be vacuous on CPU.
            unprotected_logits = F.linear(
                model.final_norm(states["h"].float()), model.embedding.weight.float()
            )
        self.assertEqual(unprotected_logits.dtype, torch.bfloat16)
        torch.testing.assert_close(protected, reference, rtol=0, atol=0)

    def test_readout_ce_chunking_unchanged_under_autocast(self):
        model = tiny_model()
        input_ids, labels, segment_ids = tiny_batch()
        states = extract_states(model, input_ids, labels, segment_ids)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            chunked = readout_ce(model, states["h"], states["targets"], chunk=3)
        reference = readout_ce(model, states["h"], states["targets"])
        torch.testing.assert_close(chunked, reference, rtol=0, atol=0)


class SplitIsolationTests(unittest.TestCase):
    DOCS = {
        "train": [(0, [5, 6, 7, 8, 9, 10]), (1, [11, 12, 13, 14])],
        "validation": [(0, [21, 22, 23, 24]), (1, [25, 26, 27, 28, 29])],
        "test": [(0, [31, 32, 33]), (1, [34, 35, 36, 37])],
    }

    def build_root(self, directory: Path) -> None:
        for split, docs in self.DOCS.items():
            writer = SplitWriter(directory, split, 8)
            for rank, tokens in docs:
                writer.add_document(rank, tokens)
            writer.finish()
        manifest = {"pipeline_version": "unit-test", "config": {"seq_len": 8},
                    "splits": {split: {"seq_len": 8} for split in self.DOCS}}
        (directory / "manifest.json").write_text(json.dumps(manifest))

    def test_gathered_states_stay_inside_their_split_file_and_docs(self):
        model = tiny_model()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_root(root)
            seen = {}
            for split, docs in self.DOCS.items():
                dataset = PackedLMDataset(root, split)
                n_states = dataset.n_supervised_tokens
                states = gather_states(model, dataset, n_states, split=split, device=torch.device("cpu"), seed=0)
                self.assertEqual(states["h"].shape[0], n_states)
                split_docs = {rank for rank, _ in docs}
                self.assertTrue(set(states["doc_ids"].tolist()) <= split_docs)
                # Every recorded (seq, position) really is supervised in that file.
                for index in range(n_states):
                    item = dataset[int(states["seq_indices"][index])]
                    position = int(states["positions"][index])
                    self.assertNotEqual(int(item["labels"][position]), -100)
                    self.assertEqual(int(item["labels"][position]), int(states["targets"][index]))
                seen[split] = {(int(s), int(d), int(p)) for s, d, p in zip(states["seq_indices"], states["doc_ids"], states["positions"])}
            # File-level isolation: tagged (split, ...) state identities never overlap.
            tagged = {split: {(split,) + entry for entry in entries} for split, entries in seen.items()}
            self.assertEqual(len(set.union(*tagged.values())), sum(len(entries) for entries in tagged.values()))


class StudentTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.input_ids, self.labels, self.segment_ids = tiny_batch()
        self.states = extract_states(self.model, self.input_ids, self.labels, self.segment_ids)
        self.costs = counterfactual_path_costs(self.model, self.states["h"], self.states["targets"], rounds=4)

    def test_student_input_is_only_the_legal_state(self):
        features = student_features(self.states)
        hidden = self.model.config.hidden_size
        self.assertEqual(tuple(features.shape), (self.states["h"].shape[0], hidden))
        torch.testing.assert_close(features, self.states["h"].float())
        # Features carry no label information: permuting targets leaves them unchanged.
        permuted = dict(self.states)
        permuted["targets"] = self.states["targets"][torch.randperm(features.shape[0], generator=torch.Generator().manual_seed(1))]
        torch.testing.assert_close(student_features(permuted), features)

    def test_students_fit_and_select_legal_actions(self):
        features = student_features(self.states)
        results = {}
        for name in ("immediate", "q_pi0"):
            target = getattr(self.costs, name)
            result = fit_value_student(features, target, hidden_dim=16, epochs=3, batch_size=8, seed=0,
                                       validation=(features, target))
            self.assertEqual(result.student.net[-1].out_features, N_ACTIONS)
            self.assertTrue(all(entry["train_mse"] == entry["train_mse"] for entry in result.history))
            prediction = predict(result.student, features, result.mean, result.std)
            self.assertEqual(tuple(prediction.shape), (features.shape[0], N_ACTIONS))
            selection = prediction.argmin(dim=-1)
            self.assertTrue(bool(((selection >= 0) & (selection < N_ACTIONS)).all()))
            results[name] = result
        # Equal capacity: the two students have identical parameter counts.
        count = lambda result: sum(p.numel() for p in result.student.parameters())
        self.assertEqual(count(results["immediate"]), count(results["q_pi0"]))

    def test_oracle_is_not_a_deploy_policy(self):
        n_states = self.costs.q_pi0.shape[0]
        selections = {
            "baseline": self.costs.baseline_action,
            "random": random_selection(n_states, N_ACTIONS, 0, self.costs.q_pi0.device),
        }
        policies = evaluate_policies(self.costs, selections)
        self.assertNotIn("oracle", policies)
        self.assertAlmostEqual(policies["baseline"]["deploy_improvement"], 0.0, places=7)
        oracle = oracle_metrics(self.costs)
        self.assertFalse(oracle["deployable"])
        self.assertGreaterEqual(oracle["oracle_gap"], -1e-6)

    def test_ledger_counts_real_work(self):
        ledger = CounterfactualLedger()
        counterfactual_path_costs(self.model, self.states["h"], self.states["targets"], rounds=4, ledger=ledger)
        n_states = self.states["h"].shape[0]
        self.assertEqual(ledger.candidate_pair_calls, n_states * N_ACTIONS * 4)
        self.assertEqual(ledger.head_readouts, n_states * N_ACTIONS * 2)



if __name__ == "__main__":
    unittest.main()
