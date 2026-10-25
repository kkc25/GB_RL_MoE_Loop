"""E01 minimal real diagnostic: counterfactual Top-2 path costs + value students.

Pipeline (frozen backbone, no backbone updates anywhere):

1.  Load an E00 LoopMoE checkpoint and hash it (backbone hash).
2.  Gather supervised states (labels != -100 only) from the packed-LM
    train/validation/test split files, keeping states isolated by file/doc.
3.  Cache h entering the last adaptive FFN, then really execute all 28 Top-2
    candidate paths per state (forced action at round 1, baseline pi0
    continuation to T=rounds), accumulating FP32 CE in chunks.
4.  Train two equal-capacity MLP students (immediate cost, Q^{pi0}) on train
    states with validation-state selection.
5.  On held-out test states report the real executed cost of baseline,
    random, and both students (deploy improvement vs baseline), plus the
    label-informed oracle in a clearly separated section, student MSE, and
    the cost ledger.

Example (small CPU pilot, defaults are already small):

    python run_e01.py --config configs/e00_debug67_b2.json \
        --checkpoint runs/e00/debug67_b2/latest.pt --device cpu \
        --train-states 256 --validation-states 64 --test-states 64 \
        --epochs 5 --output-dir runs/e01/pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import torch

from data.packed_lm import PackedLMDataset
from eval.counterfactual import (
    CounterfactualLedger,
    counterfactual_path_costs,
    evaluate_policies,
    gather_states,
    oracle_metrics,
    random_selection,
)
from lm_config import LMExperimentConfig
from model import LoopMoE
from train.common import MetricWriter, seed_everything
from train.recovery_common import atomic_json, sha256_file
from train.value_student import fit_value_student, mse, predict, student_features


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="default: <train.output_dir>/latest.pt from the config")
    parser.add_argument("--train-states", type=int, default=256)
    parser.add_argument("--validation-states", type=int, default=64)
    parser.add_argument("--test-states", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", default="runs/e01/pilot")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1, help="fixed depth of earlier adaptive layers, if any")
    parser.add_argument("--seq-batch", type=int, default=2)
    parser.add_argument("--state-chunk", type=int, default=32)
    parser.add_argument("--readout-chunk", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--student-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def save_records(path: Path, states: dict, costs, backbone_hash: str) -> None:
    with path.open("w") as handle:
        for index in range(states["h"].shape[0]):
            record = {
                "split": states["split"],
                "seq": int(states["seq_indices"][index]),
                "doc": int(states["doc_ids"][index]),
                "position": int(states["positions"][index]),
                "target": int(states["targets"][index]),
                "baseline_action": int(costs.baseline_action[index]),
                "immediate": [round(value, 6) for value in costs.immediate[index].tolist()],
                "q_pi0": [round(value, 6) for value in costs.q_pi0[index].tolist()],
                "backbone_hash": backbone_hash,
            }
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = LMExperimentConfig.load(args.config)
    config.save(output_dir / "resolved_config.json")
    checkpoint_path = Path(args.checkpoint or f"{config.train.output_dir}/latest.pt")
    backbone_hash = sha256_file(checkpoint_path)
    seed_everything(args.seed)
    device = torch.device(args.device)
    ledger = CounterfactualLedger()
    writer = MetricWriter(output_dir / "metrics.jsonl")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LoopMoE(config.model)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint_path),
        "backbone_hash": backbone_hash,
    }
    atomic_json(output_dir / "environment.json", environment)

    requested = {"train": args.train_states, "validation": args.validation_states, "test": args.test_states}
    states, costs = {}, {}
    for offset, (split, n_states) in enumerate(requested.items()):
        dataset = PackedLMDataset(config.train.data_dir, split)
        states[split] = gather_states(
            model, dataset, n_states, split=split, device=device, dtype=args.dtype,
            seed=args.seed + 101 * offset, seq_batch=args.seq_batch, depth=args.depth, ledger=ledger,
        )
        costs[split] = counterfactual_path_costs(
            model, states[split]["h"], states[split]["targets"],
            rounds=args.rounds, state_chunk=args.state_chunk, readout_chunk=args.readout_chunk,
            dtype=args.dtype, ledger=ledger,
        )
        torch.save(
            {**{key: value for key, value in states[split].items() if key != "split"},
             "split": split, "immediate": costs[split].immediate, "q_pi0": costs[split].q_pi0,
             "baseline_action": costs[split].baseline_action, "actions": costs[split].actions,
             "rounds": args.rounds, "backbone_hash": backbone_hash},
            output_dir / f"states_{split}.pt",
        )
        save_records(output_dir / f"records_{split}.jsonl", states[split], costs[split], backbone_hash)
        ledger.merge_peak(device)

    train_x, validation_x, test_x = (student_features(states[split]) for split in ("train", "validation", "test"))
    students = {}
    for name, target in (("immediate", "immediate"), ("q_pi0", "q_pi0")):
        result = fit_value_student(
            train_x, getattr(costs["train"], target),
            hidden_dim=args.hidden_dim, epochs=args.epochs, lr=args.lr,
            batch_size=args.student_batch_size, seed=args.seed,
            validation=(validation_x, getattr(costs["validation"], target)),
        )
        ledger.student_optimizer_updates += result.optimizer_updates
        ledger.student_train_samples += train_x.shape[0] * args.epochs
        students[name] = result
        writer.write(phase="student", student=name, best_validation_mse=result.best_validation_mse)
    torch.save(
        {name: {"state_dict": result.student.state_dict(), "mean": result.mean, "std": result.std,
                "hidden_dim": args.hidden_dim, "history": result.history} for name, result in students.items()},
        output_dir / "students.pt",
    )

    test_costs = costs["test"]
    n_test = test_costs.q_pi0.shape[0]
    n_actions = test_costs.q_pi0.shape[1]
    selections = {
        "baseline": test_costs.baseline_action,
        "random": random_selection(n_test, n_actions, args.seed, device),
        "student_immediate": predict(students["immediate"].student, test_x, students["immediate"].mean, students["immediate"].std).argmin(dim=-1),
        "student_q_pi0": predict(students["q_pi0"].student, test_x, students["q_pi0"].mean, students["q_pi0"].std).argmin(dim=-1),
    }
    policies = evaluate_policies(test_costs, selections)
    oracle = oracle_metrics(test_costs)
    student_mse = {
        name: mse(result.student, test_x, getattr(test_costs, name), result.mean, result.std)
        for name, result in students.items()
    }
    writer.write(phase="heldout", **{f"policy_{key}": value["mean_cost"] for key, value in policies.items()})
    writer.write(phase="heldout_oracle", **oracle)

    ledger.elapsed_seconds = time.monotonic() - started
    ledger.merge_peak(device)
    ledger.save(output_dir / "cost_ledger.json")

    summary = {
        "experiment": "E01 counterfactual path space and deployable learnability",
        "checkpoint": str(checkpoint_path),
        "backbone_hash": backbone_hash,
        "rounds": args.rounds,
        "n_actions": n_actions,
        "n_states": requested,
        "state_isolation": "train/validation/test states come from separate packed-LM split files; (split, doc, seq, position) recorded per state",
        "policies": policies,
        "oracle": {**oracle, "note": "label-informed lower bound on cost; reported separately and never counted as deploy improvement"},
        "students": {
            name: {"test_mse": student_mse[name], "best_validation_mse": result.best_validation_mse}
            for name, result in students.items()
        },
        "ledger": ledger.__dict__,
    }
    deploy_gain = policies["student_q_pi0"]["deploy_improvement"]
    if oracle["oracle_gap"] > 0:
        summary["descriptive_recovery_ratio"] = deploy_gain / oracle["oracle_gap"]
    atomic_json(output_dir / "e01_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
