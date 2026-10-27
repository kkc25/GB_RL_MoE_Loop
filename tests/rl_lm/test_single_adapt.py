"""Supervised adaptation contracts: differentiable readout, whitelist, resume."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.rl_lm_single_adapt import AdaptTrainer, adapt_lr, build_student
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
            u_t = single_ffn_update(teacher.blocks[layer].moe.pool, n, experts)
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
