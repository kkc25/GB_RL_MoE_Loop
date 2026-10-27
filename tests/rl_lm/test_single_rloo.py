"""Single-branch RLOO: toy environment rollout, monitored update, resume.

Mirrors the pair branch's T19/T20 contracts with the 9-action single
parameterization: the toy config keeps the production ``adaptive_layer: 11``
schema value; terminal-layer enforcement on the 2-layer fixture is done by
the executor.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.rl_lm_single_train import (
    SingleEnvironment,
    SingleTrainer,
    adapted_single_reference_ce,
    single_replay_log_joint,
)
from rl_lm.rng import make_uniform_table
from rl_lm.single_actor import SingleActor
from rl_lm.single_config import SingleRLConfig
from rl_lm.single_executor import (
    ArgmaxSingleTeacherPolicy,
    SingleLedger,
    make_single_executor,
    single_argmax_expert,
    single_ffn_update,
)
from rl_lm.readout import terminal_ce
from tests.rl_lm.tiny_lm import TINY_DIM, TINY_VOCAB, build_tiny_model

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs" / "rl_lm" / "single_rloo.json"

VOLATILE = {"run_id", "environment_seconds", "actor_seconds", "wall_time_seconds_cumulative", "ledger"}


def toy_config_dict(workdir) -> dict:
    data = json.loads(BASE_CONFIG.read_text())
    data["output_dir"] = str(workdir)
    data["runtime"]["device"] = "cpu"
    data["runtime"]["require_cuda"] = False
    data["optimization"]["batch_contexts"] = 8
    data["optimization"]["replay_microbatch_contexts"] = 3
    data["evaluation"]["interval_updates"] = 4
    data["logging"]["interval_updates"] = 1000
    data["data"]["train_states"] = 64
    return data


class ToyProvider:
    def __init__(self, count: int, seed: int = 9) -> None:
        generator = torch.Generator().manual_seed(seed)
        self._h0 = torch.randn(count, TINY_DIM, generator=generator) * 1.5
        self._targets = torch.randint(0, TINY_VOCAB, (count,), generator=generator)
        self._ref_ce = torch.rand(count, generator=generator) + 1.0
        self._episode_ids = tuple(f"toy-single-v1:seq-{i // 13}:pos-{i}" for i in range(count))

    def __len__(self) -> int:
        return self._h0.shape[0]

    def gather(self, indices, device):
        index = torch.as_tensor(list(indices), dtype=torch.int64)
        return SimpleNamespace(
            h0=self._h0[index].to(device),
            targets=self._targets[index].to(device),
            ref_ce=self._ref_ce[index].to(device),
            episode_ids=tuple(self._episode_ids[int(i)] for i in index),
        )


class StubEvaluator:
    def __call__(self, model, actor, update):
        return {"greedy_nll": 5.0 - 1e-3 * update, "mean_rounds": 1.0}


def make_trainer(config_dict, *, resume_path=None, stop_after=None) -> SingleTrainer:
    config = SingleRLConfig(copy.deepcopy(config_dict))
    model = build_tiny_model()
    return SingleTrainer(
        config,
        model,
        train_contexts=ToyProvider(64),
        evaluator=StubEvaluator(),
        device=torch.device("cpu"),
        resume_path=resume_path,
        stop_after=stop_after,
    )


class SingleRolloutTests(unittest.TestCase):
    def test_rollout_replay_and_shapes(self):
        torch.manual_seed(17)
        model = build_tiny_model()
        ledger = SingleLedger()
        executor = make_single_executor(model, "canonical_all_experts_single_v1", compute_dtype="float32", ledger=ledger)
        environment = SingleEnvironment(model, max_calls=1, executor=executor, ledger=ledger)
        provider = ToyProvider(16)
        contexts = provider.gather(range(4), torch.device("cpu"))
        actor = SingleActor(2 * TINY_DIM + 2 * 8 + 4)
        uniforms = make_uniform_table(17, 0, contexts.episode_ids, 4, 1)
        rollout, records = environment.rollout(contexts, actor, uniforms, policy_version=0)
        self.assertEqual(tuple(rollout.terminal_ce.shape), (4, 4))
        self.assertTrue(bool((rollout.rounds == 1).all()))
        self.assertTrue(bool((records.actions >= 1).all()))
        self.assertTrue(bool((records.actions <= 8).all()))
        self.assertEqual(tuple(records.old_log_probs_all.shape), (16, 9))
        replay = single_replay_log_joint(actor, records, 4, 4)
        self.assertTrue(torch.allclose(replay, rollout.old_log_joint, atol=1e-5, rtol=1e-5))
        # canonical backend: physical/logical expert ratio exactly 8.
        self.assertEqual(ledger.logical_expert_rows, 16)
        self.assertEqual(ledger.physical_expert_rows, 8 * 64 * 1)

    def test_reference_ce_is_adapted_single_argmax(self):
        model = build_tiny_model()
        cache = {
            "h0": (torch.randn(12, TINY_DIM) * 1.5).numpy(),
            "targets": torch.randint(0, TINY_VOCAB, (12,)).numpy(),
        }
        ref = adapted_single_reference_ce(model, cache, torch.device("cpu"), chunk=5, compute_dtype="float32")
        layer = model.config.layers - 1
        block = model.blocks[layer]
        gamma = float(model.gammas[layer])
        with torch.no_grad():
            for row in range(12):
                h = torch.from_numpy(cache["h0"][row:row + 1])
                n = block.ffn_norm(h).float()
                z = block.moe.gate(n)
                expert = single_argmax_expert(z.float())
                update = single_ffn_update(block.moe.pool, n, expert)
                ce = terminal_ce(model, h + gamma * update, torch.from_numpy(cache["targets"][row:row + 1]))
                self.assertAlmostEqual(float(ref[row]), float(ce[0]), places=5)


class SingleResumeTests(unittest.TestCase):
    def test_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as tmp:
            dir_a = Path(tmp) / "a"
            trainer_a = make_trainer(toy_config_dict(dir_a), stop_after=8)
            trainer_a.run()
            dir_b = Path(tmp) / "b"
            trainer_b1 = make_trainer(toy_config_dict(dir_b), stop_after=4)
            trainer_b1.run()
            trainer_b2 = make_trainer(toy_config_dict(dir_b), resume_path=dir_b / "checkpoint_latest.pt", stop_after=8)
            trainer_b2.run()

            for (name_a, pa), (name_b, pb) in zip(
                trainer_a.actor.state_dict().items(), trainer_b2.actor.state_dict().items()
            ):
                self.assertEqual(name_a, name_b)
                self.assertTrue(torch.equal(pa, pb), name_a)
            self.assertEqual(trainer_b2.successful_updates, 8)

            metrics_a = [json.loads(row) for row in (dir_a / "metrics.jsonl").read_text().splitlines()]
            metrics_b = [json.loads(row) for row in (dir_b / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(metrics_a), len(metrics_b))
            for row_a, row_b in zip(metrics_a, metrics_b):
                for key in row_a:
                    if key in VOLATILE:
                        continue
                    self.assertEqual(row_a[key], row_b[key], f"update {row_a['update']} key {key}")

    def test_refusing_start_without_gate(self):
        config_dict = toy_config_dict(Path(tempfile.mkdtemp()) / "run")
        config = SingleRLConfig(copy.deepcopy(config_dict)).validate()
        self.assertEqual(config.get("cost.unit"), "extra_single_expert_round")
        self.assertEqual(config.get("policy.action_space"), "single8_stop_v1")
        self.assertEqual(config.get("policy.distribution"), "factorized_joint9")


if __name__ == "__main__":
    unittest.main()
