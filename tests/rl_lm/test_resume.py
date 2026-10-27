"""T19: exact resume — 20 continuous updates vs 10 + save + new process + 10.

Spec 12.4 / 14: on the CPU toy environment, training 20 updates in one
process must produce bitwise-identical per-update episode IDs, uniform
tables, actions, losses, LRs and final actor/optimizer/scheduler/cursor/RNG/
best/ledger state as training 10 updates, checkpointing, and resuming in a
*new* process for 10 more.  The resumed half is driven through the real
checkpoint path (``Trainer(resume_path=...)``) in a subprocess so hidden
in-process state cannot leak.

Also covered: the CLI rejects ``--resume`` together with
``--warmstart-actor``, warmstart creates a *new* run that inherits only the
actor trunk/pair head (stop_head zeroed, fresh optimizer/cursor/ledger), and
R2 warmstart unfreezes a zeroed stop_head (spec 7.4 / 12.3).

The toy config keeps the production ``adaptive_layer: 11`` field (the
RLConfig schema value for the 100M backbone); terminal-layer enforcement on
the actual model is done by the executor, which validates
``layer == model.config.layers - 1`` for the 2-layer tiny fixture.

Run directly as a driver: ``python -m tests.rl_lm.test_resume --driver
--workdir DIR --mode continuous|first|resumed``.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from rl_lm.config import RLConfig
from rl_lm.ledger import RLLedger
from rl_lm.trainer import GreedyCacheEvaluator, TensorContextProvider, Trainer
from tests.rl_lm.tiny_lm import TINY_DIM, TINY_VOCAB, build_tiny_model

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs" / "rl_lm" / "pair_r1.json"

CONTINUOUS_UPDATES = 20
SPLIT_AT = 10

# Keys that legitimately differ between the continuous and resumed runs.
VOLATILE_METRIC_KEYS = {
    "run_id",
    "initialization",
    "environment_seconds",
    "actor_seconds",
    "wall_time_seconds_cumulative",
    "ledger",  # compared separately from the checkpoint payload
}


def toy_config_dict(workdir) -> dict:
    data = json.loads(BASE_CONFIG.read_text())
    data["output_dir"] = str(workdir)
    data["runtime"]["device"] = "cpu"
    data["runtime"]["require_cuda"] = False
    data["optimization"]["batch_contexts"] = 8
    data["optimization"]["replay_microbatch_contexts"] = 3  # exercises microbatch replay
    data["evaluation"]["interval_updates"] = 5
    data["logging"]["interval_updates"] = 1000  # keep driver stdout quiet
    data["data"]["train_states"] = 64
    return data


def toy_r2_config_dict(workdir) -> dict:
    data = toy_config_dict(workdir)
    data["stage"] = "r2"
    data["environment"]["max_calls"] = 2
    data["optimization"]["learning_rate"] = 0.00005
    return data


def toy_provider(count: int, seed: int) -> TensorContextProvider:
    generator = torch.Generator().manual_seed(seed)
    h0 = torch.randn(count, TINY_DIM, generator=generator) * 1.5
    targets = torch.randint(0, TINY_VOCAB, (count,), generator=generator)
    ref_ce = torch.rand(count, generator=generator) + 1.0
    episode_ids = tuple(f"toy-corpus-v1:seq-{i // 13}:pos-{i}" for i in range(count))
    return TensorContextProvider(h0, targets, ref_ce, episode_ids)


def make_trainer(config_dict, *, resume_path=None, warmstart_path=None, stop_after=None) -> Trainer:
    config = RLConfig.from_dict(copy.deepcopy(config_dict))
    model = build_tiny_model()
    trainer = Trainer(
        config,
        model,
        train_contexts=toy_provider(64, 1234),
        evaluator=None,
        device=torch.device("cpu"),
        resume_path=resume_path,
        warmstart_path=warmstart_path,
        stop_after=stop_after,
    )
    selection = toy_provider(8, 777)
    trainer.evaluator = GreedyCacheEvaluator(
        trainer.environment, selection.gather(list(range(8)), torch.device("cpu"))
    )
    return trainer


def driver(argv) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", action="store_true")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--mode", choices=["continuous", "first", "resumed"], required=True)
    args = parser.parse_args(argv)
    workdir = Path(args.workdir)
    config_dict = toy_config_dict(workdir)
    if args.mode == "continuous":
        trainer = make_trainer(config_dict, stop_after=CONTINUOUS_UPDATES)
    elif args.mode == "first":
        trainer = make_trainer(config_dict, stop_after=SPLIT_AT)
    else:
        trainer = make_trainer(
            config_dict,
            resume_path=workdir / "checkpoint_latest.pt",
            stop_after=CONTINUOUS_UPDATES,
        )
    summary = trainer.run()
    print(json.dumps({"status": summary["status"], "updates": summary["successful_updates"]}))


def run_driver(workdir, mode) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tests.rl_lm.test_resume", "--driver", "--workdir", str(workdir), "--mode", mode],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"driver {mode} failed:\n{result.stdout}\n{result.stderr}")


def load_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def normalized_metrics(entries: list[dict]) -> list[dict]:
    return [{k: v for k, v in entry.items() if k not in VOLATILE_METRIC_KEYS} for entry in entries]


def compare_optimizer_state(first, second, path: str = "optimizer") -> None:
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor), f"{path}: type mismatch"
        if not torch.equal(first.cpu(), second.cpu()):
            raise AssertionError(f"{path}: tensor mismatch")
        return
    if isinstance(first, dict):
        assert set(first) == set(second), f"{path}: key mismatch {set(first) ^ set(second)}"
        for key in first:
            compare_optimizer_state(first[key], second[key], f"{path}.{key}")
        return
    if isinstance(first, (list, tuple)):
        assert len(first) == len(second), f"{path}: length mismatch"
        for index, (a, b) in enumerate(zip(first, second)):
            compare_optimizer_state(a, b, f"{path}[{index}]")
        return
    if isinstance(first, float):
        assert first == second, f"{path}: {first} != {second}"
        return
    assert first == second, f"{path}: {first!r} != {second!r}"


class ExactResumeTests(unittest.TestCase):
    """T19: continuous 20 vs 10 + new-process resume + 10, bitwise."""

    def test_continuous_vs_new_process_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            dir_a = Path(tmp) / "continuous"
            dir_b = Path(tmp) / "split"
            run_driver(dir_a, "continuous")
            run_driver(dir_b, "first")
            run_driver(dir_b, "resumed")

            metrics_a = load_metrics(dir_a / "metrics.jsonl")
            metrics_b = load_metrics(dir_b / "metrics.jsonl")
            self.assertEqual(len(metrics_a), CONTINUOUS_UPDATES)
            self.assertEqual(len(metrics_b), CONTINUOUS_UPDATES)

            norm_a = normalized_metrics(metrics_a)
            norm_b = normalized_metrics(metrics_b)
            self.assertEqual(norm_a, norm_b, "per-update episode ids / uniforms / actions / losses differ")
            # Spot-report the key identity fields for the record.
            for key in ("batch_episode_sha256", "uniforms_sha256", "actions_sha256", "actor_loss", "lr"):
                self.assertEqual(
                    [entry[key] for entry in metrics_a],
                    [entry[key] for entry in metrics_b],
                    f"{key} mismatch",
                )
            # The resumed process continues the same run identity.
            self.assertEqual(metrics_b[0]["run_id"], metrics_b[-1]["run_id"])
            self.assertEqual(metrics_b[0]["initialization"], "fresh")
            self.assertEqual(metrics_b[-1]["initialization"], "resumed")

            payload_a = torch.load(dir_a / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
            payload_b = torch.load(dir_b / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
            for key in ("actor_state_dict", "optimizer_state_dict"):
                compare_optimizer_state(payload_a[key], payload_b[key], key)
            self.assertEqual(payload_a["scheduler_state"], payload_b["scheduler_state"])
            self.assertEqual(payload_a["training_state"], payload_b["training_state"])
            self.assertEqual(payload_a["selection_state"], payload_b["selection_state"])
            self.assertEqual(payload_a["ledger"], payload_b["ledger"])
            self.assertTrue(
                torch.equal(payload_a["rng_state"]["torch"], payload_b["rng_state"]["torch"]),
                "torch RNG state differs",
            )
            self.assertTrue(
                np_equal(payload_a["rng_state"]["numpy"], payload_b["rng_state"]["numpy"]),
                "numpy RNG state differs",
            )
            self.assertEqual(payload_a["rng_state"]["python"], payload_b["rng_state"]["python"])
            self.assertEqual(payload_a["stage"], payload_b["stage"])
            # Evaluations and best tracking survived the process boundary.
            evals_a = load_metrics(dir_a / "evaluations.jsonl")
            evals_b = load_metrics(dir_b / "evaluations.jsonl")
            self.assertEqual(evals_a, evals_b)
            self.assertEqual(payload_b["selection_state"]["best"]["update"], payload_a["selection_state"]["best"]["update"])
            print(
                "\nT19: 20-update continuous run and 10+resume+10 split run are bitwise identical"
                f" (final actor_loss={metrics_a[-1]['actor_loss']:.6e},"
                f" epoch={payload_a['training_state']['epoch']},"
                f" cursor={payload_a['training_state']['next_batch_cursor']})"
            )

    def test_cli_rejects_resume_with_warmstart(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "experiments.rl_lm_train",
                "--config", str(BASE_CONFIG),
                "--resume", "a.pt", "--warmstart-actor", "b.pt",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not allowed with argument", result.stderr)


def np_equal(first, second) -> bool:
    import numpy as np

    if len(first) != len(second):
        return False
    for a, b in zip(first, second):
        if isinstance(a, np.ndarray):
            if not np.array_equal(a, b):
                return False
        elif a != b:
            return False
    return True


class WarmstartTests(unittest.TestCase):
    """Spec 12.3 / 7.4: warmstart is a new run with actor-only inheritance."""

    def test_warmstart_new_run_actor_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent_dir = Path(tmp) / "parent"
            parent = make_trainer(toy_config_dict(parent_dir), stop_after=3)
            parent_summary = parent.run()
            self.assertEqual(parent_summary["successful_updates"], 3)

            child_dir = Path(tmp) / "child"
            child = make_trainer(
                toy_config_dict(child_dir),
                warmstart_path=parent_dir / "checkpoint_latest.pt",
                stop_after=2,
            )
            self.assertEqual(child.initialization, "warmstart")
            self.assertNotEqual(child.run_id, parent.run_id)
            self.assertEqual(child.successful_updates, 0)
            self.assertEqual(child.epoch, 0)
            self.assertEqual(child.epoch_cursor, 0)
            self.assertEqual(child.ledger, RLLedger())
            parent_actor = torch.load(
                parent_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=False
            )["actor_state_dict"]
            child_state = child.actor.state_dict()
            for key, value in parent_actor.items():
                if key.startswith("stop_head."):
                    self.assertTrue(bool((child_state[key] == 0).all()), "stop_head must be reset to zero")
                else:
                    self.assertTrue(torch.equal(child_state[key].cpu(), value), f"{key} not inherited")
            self.assertFalse(child.actor.stop_head.weight.requires_grad, "R1 stop_head stays frozen")
            self.assertEqual(child.warmstart_from["parent_run_id"], parent.run_id)
            summary = child.run()
            self.assertEqual(summary["successful_updates"], 2)
            self.assertTrue((child_dir / "checkpoint_initial.pt").exists())

    def test_r2_warmstart_unfreezes_zeroed_stop_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent_dir = Path(tmp) / "parent_r1"
            parent = make_trainer(toy_config_dict(parent_dir), stop_after=2)
            parent.run()
            child_dir = Path(tmp) / "child_r2"
            child = make_trainer(
                toy_r2_config_dict(child_dir),
                warmstart_path=parent_dir / "checkpoint_latest.pt",
                stop_after=2,
            )
            self.assertTrue(child.actor.stop_head.weight.requires_grad)
            self.assertTrue(bool((child.actor.stop_head.weight == 0).all()))
            summary = child.run()
            self.assertEqual(summary["successful_updates"], 2)
            self.assertEqual(child.rmax, 2)


if __name__ == "__main__":
    if "--driver" in sys.argv:
        driver(sys.argv[1:])
    else:
        unittest.main()
