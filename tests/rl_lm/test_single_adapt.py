"""Supervised adaptation contracts: differentiable readout, whitelist, resume."""

import copy
import itertools
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from experiments import rl_lm_single_adapt as adapt_cli
from experiments.rl_lm_single_adapt import (
    ADAPT_CHECKPOINT_FORMAT, ADAPT_OBJECTIVE_VERSION, AdaptTrainer,
    _interleaved_timings, _terminal_ffn_timing_paths, adapt_lr, build_student,
    pair_selection_reference_identity,
)
from rl_lm.single_config import SingleAdaptConfig
from rl_lm.single_executor import single_argmax_expert, single_ffn_update
from rl_lm.single_readout import (
    adapt_ce_loss,
    adapt_kl_loss,
    differentiable_logits,
    relative_update_error,
)
from tests.rl_lm.tiny_lm import build_tiny_model

TOY_WHITELIST = ("blocks.1.moe.pool.w1", "blocks.1.moe.pool.w2", "blocks.1.moe.pool.w3")


def _toy_config(run_dir, max_updates=20, batch=8):
    return SingleAdaptConfig({
        "spec_version": "RL-LM-Single-v1.0",
        "stage": "adapt",
        "backbone": {
            "config_path": "toy.json",
            "checkpoint_path": None,
            "expected_checkpoint_sha256": "0" * 64,
            "adaptive_layer": 1,
            "freeze_all": True,
            "prefix_depth": 1,
        },
        "data": {
            "corpus_dir": None,
            "tokenizer_path": None,
            "pair_cache_root": "toy_pair",
            "cache_root": "toy_single",
            "train_split": "controller_train",
            "selection_split": "selection",
            "calibration_split": "calibration",
            "confirmation_split": "newtest",
            "train_states": 64,
            "selection_states": 16,
            "calibration_states": 8,
            "state_sampling": "uniform_valid_token_without_replacement",
            "sampling_seed": 1,
            "prefix_sequence_batch": 2,
        },
        "adaptation": {
            "learning_rate": 1e-3,
            "final_learning_rate": 1e-6,
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "batch_contexts": batch,
            "max_updates": max_updates,
            "warmup_updates": 5,
            "kl_coefficient": 1.0,
            "update_error_coefficient": 0.1,
            "update_error_epsilon": 1e-8,
            "eval_interval_updates": 100,
            "selection_gate_tolerance_nll": 0.02,
            "trainable_whitelist": list(TOY_WHITELIST),
        },
        "environment": {
            "min_calls": 1,
            "max_calls": 1,
            "gamma_source": "backbone_config",
            "training_backend": "canonical_all_experts_single_v1",
            "deployment_backend": "sparse_single_v1",
            "expert_chunk": 64,
            "head_chunk": 64,
            "frozen_compute_dtype": "float32",
            "residual_dtype": "float32",
            "head_contract": "bf16_linear_fp32_ce_v1",
        },
        "runtime": {"device": "cpu", "require_cuda": False, "allow_tf32": False},
        "logging": {"interval_updates": 10},
        "output_dir": str(run_dir),
    })


def _toy_cache(n_states=64, dim=32, vocab=64, seed=5):
    rng = np.random.default_rng(seed)
    return {
        "h0": rng.standard_normal((n_states, dim)).astype(np.float32),
        "targets": rng.integers(0, vocab, n_states).astype(np.int64),
        "episode_ids": [{"episode_id": f"toy:{i}"} for i in range(n_states)],
    }


def _manual_pair_update(model, n, z):
    """Independent reference: Python lexicographic pair search + scalar FFNs."""
    pool = model.blocks[model.config.adaptive_layers[-1]].moe.pool
    outputs = []
    for row in range(len(n)):
        pair = max(itertools.combinations(range(8), 2),
                   key=lambda ij: float(z[row, ij[0]] + z[row, ij[1]]))
        weights = torch.softmax(z[row, list(pair)].float(), dim=0)
        update = torch.zeros_like(n[row])
        for slot, expert in enumerate(pair):
            hidden = torch.nn.functional.silu(torch.nn.functional.linear(n[row], pool.w1[expert]))
            hidden = hidden * torch.nn.functional.linear(n[row], pool.w3[expert])
            update = update + weights[slot] * torch.nn.functional.linear(hidden, pool.w2[expert])
        outputs.append(update)
    return torch.stack(outputs)


class DifferentiableReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.model = build_tiny_model()

    def test_grad_flows_through_frozen_head_to_hidden(self):
        h = torch.randn(6, 32, requires_grad=True)
        logits = differentiable_logits(self.model, h)
        self.assertEqual(tuple(logits.shape), (6, 64))
        self.assertTrue(logits.requires_grad)
        loss = adapt_ce_loss(logits, torch.randint(0, 64, (6,)))
        loss.backward()
        self.assertIsNotNone(h.grad)
        self.assertGreater(float(h.grad.abs().sum()), 0.0)
        self.assertIsNone(self.model.embedding.weight.grad)
        self.assertFalse(self.model.embedding.weight.requires_grad)

    def test_kl_requires_detached_teacher(self):
        h_t = torch.randn(4, 32, requires_grad=True)
        h_s = torch.randn(4, 32, requires_grad=True)
        teacher_logits = differentiable_logits(self.model, h_t)
        student_logits = differentiable_logits(self.model, h_s)
        with self.assertRaises(ValueError):
            adapt_kl_loss(teacher_logits, student_logits)
        kl = adapt_kl_loss(teacher_logits.detach(), student_logits)
        kl.backward()
        self.assertIsNotNone(h_s.grad)
        self.assertIsNone(h_t.grad)

    def test_kl_is_zero_for_identical_logits(self):
        h = torch.randn(4, 32)
        logits = differentiable_logits(self.model, h)
        kl = adapt_kl_loss(logits.detach(), logits.detach().clone().requires_grad_(True))
        self.assertAlmostEqual(float(kl), 0.0, places=5)

    def test_relative_update_error_is_per_token(self):
        u_t = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        u_s = torch.tensor([[1.1, 0.0], [0.0, 4.04]])
        expected = ((0.1 ** 2) / (1.0 ** 2 + 1e-8) + (0.04 ** 2) / (4.0 ** 2 + 1e-8)) / 2
        value = relative_update_error(u_s.requires_grad_(True), u_t)
        self.assertAlmostEqual(float(value), expected, places=6)
        # A single global batch-norm ratio would give a different number.
        global_ratio = float((u_s - u_t).square().sum() / (u_t.square().sum() + 1e-8))
        self.assertNotAlmostEqual(float(value), global_ratio, places=6)
        with self.assertRaises(ValueError):
            relative_update_error(u_s, u_t.requires_grad_(True))


class AdaptWhitelistTests(unittest.TestCase):
    def test_whitelist_is_exactly_terminal_pool(self):
        teacher = build_tiny_model()
        student = build_student(teacher, TOY_WHITELIST)
        trainable = sorted(name for name, p in student.named_parameters() if p.requires_grad)
        self.assertEqual(trainable, sorted(TOY_WHITELIST))

    def test_adapt_step_only_moves_terminal_pool(self):
        torch.manual_seed(21)
        teacher = build_tiny_model()
        student = build_student(teacher, TOY_WHITELIST)
        before_student = {name: p.detach().clone() for name, p in student.named_parameters()}
        before_teacher = {name: p.detach().clone() for name, p in teacher.named_parameters()}
        layer = 1
        h0 = torch.randn(16, 32)
        targets = torch.randint(0, 64, (16,))
        with torch.no_grad():
            n = teacher.blocks[layer].ffn_norm(h0).float()
            z = teacher.blocks[layer].moe.gate(n)
            experts = single_argmax_expert(z.float())
            u_t = _manual_pair_update(teacher, n, z)
            logits_t = differentiable_logits(teacher, h0 + teacher.gammas[layer] * u_t).detach()
        u_s = single_ffn_update(student.blocks[layer].moe.pool, n, experts)
        logits_s = differentiable_logits(student, h0 + student.gammas[layer] * u_s)
        loss = adapt_ce_loss(logits_s, targets) + adapt_kl_loss(logits_t, logits_s) + 0.1 * relative_update_error(u_s, u_t)
        loss.backward()
        named_s = dict(student.named_parameters())
        for name, parameter in student.named_parameters():
            if name in TOY_WHITELIST:
                self.assertIsNotNone(parameter.grad, name)
            else:
                self.assertIsNone(parameter.grad, name)
        for name, parameter in teacher.named_parameters():
            self.assertIsNone(parameter.grad, name)
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-2)
        optimizer.step()
        moved = set()
        for name, parameter in student.named_parameters():
            if not torch.equal(parameter.detach(), before_student[name]):
                moved.add(name)
        self.assertTrue(moved <= set(TOY_WHITELIST), moved)
        self.assertTrue(moved, "adaptation did not move any parameter")
        for name, parameter in teacher.named_parameters():
            self.assertTrue(torch.equal(parameter.detach(), before_teacher[name]), name)


class PairTeacherRegressionTests(unittest.TestCase):
    def test_initial_student_distills_original_pair_not_identical_single(self):
        cache = _toy_cache()
        # A non-tied gate with largest expert ID greater than the second-best
        # catches taking the first sorted pair ID as the single selection.
        cache["h0"][:, 0] = 1.0
        teacher = build_tiny_model()
        with torch.no_grad():
            teacher.blocks[1].moe.gate.weight.zero_()
            teacher.blocks[1].moe.gate.weight[5, 0] = 0.2
            teacher.blocks[1].moe.gate.weight[2, 0] = 0.1
        with tempfile.TemporaryDirectory() as tmp:
            trainer = AdaptTrainer(_toy_config(tmp), teacher, train_cache=cache,
                                   device=torch.device("cpu"), run_dir=tmp)
            indices = trainer._epoch_permutation()[:trainer.batch]
            h0 = torch.from_numpy(cache["h0"][indices])
            targets = torch.from_numpy(cache["targets"][indices])
            with torch.no_grad():
                n = teacher.blocks[1].ffn_norm(h0).float()
                z = teacher.blocks[1].moe.gate(n)
                u_t = _manual_pair_update(teacher, n, z)
                u_s = single_ffn_update(teacher.blocks[1].moe.pool, n, z.argmax(-1))
                logits_t = differentiable_logits(teacher, h0 + teacher.gammas[1] * u_t)
                logits_s = differentiable_logits(teacher, h0 + teacher.gammas[1] * u_s)
                expected_ce = torch.nn.functional.cross_entropy(logits_s, targets)
                log_t = logits_t.log_softmax(-1)
                log_s = logits_s.log_softmax(-1)
                expected_kl = (log_t.exp() * (log_t - log_s)).sum(-1).mean()
                expected_error = ((u_s - u_t).square().sum(-1) /
                                  (u_t.square().sum(-1) + 1e-8)).mean()
            entry = trainer._update()
            self.assertGreater(float(expected_error), 0.01)
            self.assertAlmostEqual(entry["update_error"], float(expected_error), places=5)
            self.assertAlmostEqual(entry["kl"], float(expected_kl), places=7)
            self.assertAlmostEqual(entry["ce"], float(expected_ce), places=6)
            self.assertEqual(entry["expert_sample_counts"], [0, 0, 0, 0, 0, trainer.batch, 0, 0])
            # Old wrong-teacher implementation has identical initial single
            # teacher/student updates: error=0, and fails this regression.
            self.assertGreater(entry["update_error"], 0.01)
            self.assertEqual(trainer.ledger.physical_expert_rows, 3 * trainer.batch)
            self.assertEqual(trainer.ledger.physical_head_rows, 2 * trainer.batch)

    def test_profile_interleaves_order_and_keeps_samples(self):
        calls = []
        def pair():
            calls.append("pair")
            return 2.0
        def single():
            calls.append("single")
            return 1.0
        samples = _interleaved_timings(pair, single, lambda fn: fn(), warmup=1, repeats=4)
        self.assertEqual(calls, ["pair", "single", "pair", "single", "single", "pair",
                                 "pair", "single", "single", "pair"])
        self.assertEqual(samples, {"pair": [2.0] * 4, "single": [1.0] * 4})

    def test_selection_reference_binds_actual_data_and_numerics(self):
        config = _toy_config("unused")
        ids = torch.tensor([[1, 2, 3]])
        seg = torch.tensor([[0, 0, 0]])
        labels = torch.tensor([[2, 3, -100]])
        def identity(**changes):
            args = dict(config=config, input_ids=ids, segment_ids=seg, labels=labels,
                        checkpoint_sha256="checkpoint-a", backbone_config_sha256="backbone-a")
            return pair_selection_reference_identity(**(args | changes))
        baseline = identity()
        self.assertEqual(identity(), baseline)
        for changes in ({"input_ids": ids + 1}, {"segment_ids": seg + 1},
                        {"labels": labels + 1}, {"checkpoint_sha256": "checkpoint-b"},
                        {"backbone_config_sha256": "backbone-b"}):
            self.assertNotEqual(identity(**changes), baseline)
        config.data["environment"]["head_chunk"] = 32
        self.assertNotEqual(identity(), baseline)

    def test_timing_paths_include_fresh_gate_and_use_distinct_model_weights(self):
        teacher = build_tiny_model()
        student = build_student(teacher, TOY_WHITELIST)
        with torch.no_grad():
            student.blocks[1].moe.pool.w2.mul_(3.0)
        h0 = torch.randn(8, 32)
        with mock.patch.object(teacher.blocks[1].moe.gate, "forward",
                               wraps=teacher.blocks[1].moe.gate.forward) as pair_gate, \
             mock.patch.object(student.blocks[1].moe.gate, "forward",
                               wraps=student.blocks[1].moe.gate.forward) as single_gate:
            pair_fn, single_fn = _terminal_ffn_timing_paths(teacher, student, h0, compute_dtype="float32")
            self.assertEqual(pair_gate.call_count, 0)
            self.assertEqual(single_gate.call_count, 0)
            pair_h = pair_fn()
            single_h = single_fn()
            pair_fn()
            single_fn()
            self.assertEqual(pair_gate.call_count, 2)
            self.assertEqual(single_gate.call_count, 2)
        with torch.no_grad():
            n = teacher.blocks[1].ffn_norm(h0).float()
            z = teacher.blocks[1].moe.gate(n)
            expected_pair = h0 + teacher.gammas[1] * _manual_pair_update(teacher, n, z)
            expected_single = h0 + student.gammas[1] * single_ffn_update(student.blocks[1].moe.pool, n, z.argmax(-1))
        torch.testing.assert_close(pair_h, expected_pair)
        torch.testing.assert_close(single_h, expected_single)


class AdaptScheduleTests(unittest.TestCase):
    def test_warmup_then_cosine_to_final(self):
        self.assertAlmostEqual(adapt_lr(1, peak=1e-5, final=1e-6, warmup=50, total=1000), 2e-7)
        self.assertAlmostEqual(adapt_lr(50, peak=1e-5, final=1e-6, warmup=50, total=1000), 1e-5)
        self.assertAlmostEqual(adapt_lr(1000, peak=1e-5, final=1e-6, warmup=50, total=1000), 1e-6)
        mid = adapt_lr(525, peak=1e-5, final=1e-6, warmup=50, total=1000)
        self.assertTrue(1e-6 < mid < 1e-5)


class AdaptResumeTests(unittest.TestCase):
    def test_resume_matches_continuous_run(self):
        cache = _toy_cache()
        with tempfile.TemporaryDirectory() as tmp:
            dir_a = Path(tmp) / "a"
            trainer_a = AdaptTrainer(_toy_config(dir_a, max_updates=10), build_tiny_model(), train_cache=cache,
                                     device=torch.device("cpu"), run_dir=dir_a, stop_after=10)
            trainer_a.run(lambda student, update: {"update": update})

            dir_b = Path(tmp) / "b"
            trainer_b1 = AdaptTrainer(_toy_config(dir_b, max_updates=10), build_tiny_model(), train_cache=cache,
                                      device=torch.device("cpu"), run_dir=dir_b, stop_after=4)
            trainer_b1.run(lambda student, update: {"update": update})
            trainer_b2 = AdaptTrainer(_toy_config(dir_b, max_updates=10), build_tiny_model(), train_cache=cache,
                                      device=torch.device("cpu"), run_dir=dir_b,
                                      resume_path=dir_b / "adapt_checkpoint_latest.pt", stop_after=10)
            trainer_b2.run(lambda student, update: {"update": update})

            for name in TOY_WHITELIST:
                wa = dict(trainer_a.student.named_parameters())[name]
                wb = dict(trainer_b2.student.named_parameters())[name]
                self.assertTrue(torch.equal(wa.detach(), wb.detach()), name)
            self.assertEqual(trainer_b2.successful_updates, 10)
            metrics = (dir_b / "metrics.jsonl").read_text().splitlines()
            self.assertEqual(len(metrics), 10)

    def test_cli_existing_fresh_run_preserves_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _toy_config(tmp)
            artifacts = {name: ("historical:" + name).encode() for name in
                         ("resolved_config.json", "pair_teacher_selection.json", "metrics.jsonl")}
            for name, content in artifacts.items():
                (Path(tmp) / name).write_bytes(content)
            with mock.patch.object(sys, "argv", ["adapt", "--config", "unused"]), \
                 mock.patch.object(adapt_cli, "resolve_single_config", return_value=config), \
                 mock.patch.object(adapt_cli, "load_backbone") as load:
                with self.assertRaises(FileExistsError):
                    adapt_cli.main()
                load.assert_not_called()
            self.assertEqual({p.name: p.read_bytes() for p in Path(tmp).iterdir()}, artifacts)

    def test_cli_incompatible_resume_preserves_all_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _toy_config(tmp)
            trainer = AdaptTrainer(config, build_tiny_model(), train_cache=_toy_cache(),
                                   device=torch.device("cpu"), run_dir=tmp)
            original = trainer._payload()
            for name in ("resolved_config.json", "pair_teacher_selection.json", "metrics.jsonl"):
                (Path(tmp) / name).write_bytes(("historical:" + name).encode())
            for incompatible in ("legacy", "objective", "config"):
                payload = copy.deepcopy(original)
                if incompatible == "legacy":
                    payload["format_version"] = "rl-lm-single-adapt-v1"
                elif incompatible == "objective":
                    payload["objective_version"] = "single-teacher-v1"
                else:
                    payload["resolved_config"]["adaptation"]["max_updates"] += 1
                resume = Path(tmp) / "adapt_checkpoint_latest.pt"
                torch.save(payload, resume)
                before = {p.name: p.read_bytes() for p in Path(tmp).iterdir()}
                with mock.patch.object(sys, "argv", ["adapt", "--config", "unused", "--resume", str(resume)]), \
                     mock.patch.object(adapt_cli, "resolve_single_config", return_value=config), \
                     mock.patch.object(adapt_cli, "load_backbone") as load:
                    with self.assertRaises(ValueError):
                        adapt_cli.main()
                    load.assert_not_called()
                self.assertEqual({p.name: p.read_bytes() for p in Path(tmp).iterdir()}, before)

    def test_resume_rejects_legacy_teacher_or_changed_objective(self):
        cache = _toy_cache()
        with tempfile.TemporaryDirectory() as tmp:
            config = _toy_config(tmp)
            trainer = AdaptTrainer(config, build_tiny_model(), train_cache=cache,
                                   device=torch.device("cpu"), run_dir=tmp)
            payload = trainer._payload()
            self.assertEqual(payload["format_version"], ADAPT_CHECKPOINT_FORMAT)
            self.assertEqual(payload["objective_version"], ADAPT_OBJECTIVE_VERSION)
            path = Path(tmp) / "incompatible.pt"
            for field, value, error in (
                ("format_version", "rl-lm-single-adapt-v1", "corrected pair-teacher v2"),
                ("objective_version", "single-teacher-v1", "objective version mismatch"),
            ):
                changed = dict(payload)
                changed[field] = value
                torch.save(changed, path)
                with self.assertRaisesRegex(ValueError, error):
                    AdaptTrainer(config, build_tiny_model(), train_cache=cache,
                                 device=torch.device("cpu"), run_dir=tmp, resume_path=path)

    def test_adaptation_distills_perturbed_student_back(self):
        cache = _toy_cache()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            trainer = AdaptTrainer(_toy_config(run_dir, max_updates=40), build_tiny_model(), train_cache=cache,
                                   device=torch.device("cpu"), run_dir=run_dir, stop_after=40)
            named = dict(trainer.student.named_parameters())
            with torch.no_grad():
                generator = torch.Generator().manual_seed(99)
                for name in TOY_WHITELIST:
                    named[name].add_(0.05 * torch.randn(named[name].shape, generator=generator))
                    trainer.initial_pool[name] = named[name].detach().clone()
            trainer.run(lambda student, update: {"update": update})
            lines = [__import__("json").loads(row) for row in (run_dir / "metrics.jsonl").read_text().splitlines()]
            for row in lines:
                self.assertTrue(np.isfinite(row["loss"]), row)
            first_kl = sum(row["kl"] for row in lines[:5]) / 5
            last_kl = sum(row["kl"] for row in lines[-5:]) / 5
            first_err = sum(row["update_error"] for row in lines[:5]) / 5
            last_err = sum(row["update_error"] for row in lines[-5:]) / 5
            self.assertLess(last_kl, first_kl)
            self.assertLess(last_err, first_err)
            counts = np.sum([row["expert_sample_counts"] for row in lines], axis=0)
            self.assertEqual(int(counts.sum()), 40 * 8)
            self.assertGreater(int((counts > 0).sum()), 1)


if __name__ == "__main__":
    unittest.main()
