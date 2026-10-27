"""Training CLI (spec 13.4 / 20): wiring only, no algorithm implementation.

``--resume`` and ``--warmstart-actor`` are mutually exclusive; ``--stop-after``
is an absolute lifetime cap on successful updates (never "N more from now")
and combines with either entry.  A run is only accepted after a complete
preflight (``preflight.json`` with status ``ready`` in the output dir); the
trainer never reads the confirmation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data.packed_lm import PackedLMDataset
from rl_lm.cache import cache_identity, load_cache
from rl_lm.cli import load_backbone, resolve_config
from rl_lm.inference import CanonicalTeacherPolicy, forward_with_last_ffn_policy
from rl_lm.trainer import CacheContextProvider, Trainer
from train.recovery_common import sha256_file


class OnlineSelectionEvaluator:
    """Spec 16.1 selection evaluation: full production greedy forward NLL.

    Runs the fixed number of selection sequences through
    ``forward_with_last_ffn_policy`` with the deployment backend and computes
    token-weighted mean NLL over all valid labels; the canonical teacher
    reference NLL is computed once and cached so ``delta_nll`` /
    ``gain_nll`` share one frozen denominator.
    """

    def __init__(self, config, model, device) -> None:
        self.config = config
        self.device = device
        dataset = PackedLMDataset(
            config.get("data.corpus_dir"),
            config.get("data.selection_split"),
            limit=int(config.get("evaluation.online_selection_sequences")),
        )
        if len(dataset) < 1:
            raise ValueError("selection split is empty")
        items = [dataset[i] for i in range(len(dataset))]
        self.input_ids = torch.from_numpy(np.stack([item["input_ids"] for item in items])).to(device)
        self.segment_ids = torch.from_numpy(np.stack([item["segment_ids"] for item in items])).to(device)
        self.labels = torch.from_numpy(np.stack([item["labels"] for item in items])).to(device)
        self.valid = self.labels != -100
        if not bool(self.valid.any()):
            raise ValueError("selection sequences contain no valid labels")
        self.reference_nll = None
        self.reference_rounds = None

    def _forward_nll(self, policy) -> tuple[float, float]:
        output = forward_with_last_ffn_policy(
            self._model,
            policy,
            self.input_ids,
            self.segment_ids,
            max_calls=int(self.config.get("environment.max_calls")),
            backend=self.config.get("environment.deployment_backend"),
            compute_dtype=self.config.get("environment.frozen_compute_dtype"),
            chunk=int(self.config.get("environment.expert_chunk")),
            head_chunk=int(self.config.get("environment.head_chunk")),
        )
        # Chunk over positions: the full [B*S,V] FP32 CE buffer would not fit
        # next to the BF16 logits tensor (observed OOM at the real 100M scale).
        flat_logits = output.logits.reshape(-1, output.logits.shape[-1])
        flat_labels = self.labels.reshape(-1)
        ce_sum = 0.0
        rows = 8192
        for start in range(0, flat_labels.shape[0], rows):
            ce_sum += float(
                F.cross_entropy(
                    flat_logits[start : start + rows].float(),
                    flat_labels[start : start + rows],
                    reduction="none",
                    ignore_index=-100,
                ).sum()
            )
        nll = ce_sum / int(self.valid.sum())
        mean_rounds = float(output.rounds.reshape(-1)[self.valid.reshape(-1)].float().mean())
        return nll, mean_rounds

    def __call__(self, model, actor, update: int) -> dict:
        self._model = model
        greedy_nll, mean_rounds = self._forward_nll(actor)
        if self.reference_nll is None:
            self.reference_nll, self.reference_rounds = self._forward_nll(CanonicalTeacherPolicy())
        return {
            "greedy_nll": greedy_nll,
            "mean_rounds": mean_rounds,
            "reference_nll": self.reference_nll,
            "reference_rounds": self.reference_rounds,
            "delta_nll": greedy_nll - self.reference_nll,
            "gain_nll": self.reference_nll - greedy_nll,
            "eval_backend": self.config.get("environment.deployment_backend"),
            "eval_sequences": int(self.input_ids.shape[0]),
            "eval_tokens": int(self.valid.sum()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--corpus-dir")
    parser.add_argument("--tokenizer")
    parser.add_argument("--device")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", help="exact resume of the same run; mutually exclusive with --warmstart-actor")
    start.add_argument("--warmstart-actor", help="new run inheriting only the actor weights")
    parser.add_argument("--stop-after", type=int, help="absolute lifetime cap on successful updates")
    args = parser.parse_args()

    config = resolve_config(
        args.config,
        checkpoint=args.checkpoint,
        corpus_dir=args.corpus_dir,
        tokenizer=args.tokenizer,
        device=args.device,
    )
    output_dir = Path(config.get("output_dir"))
    if args.resume is not None and Path(args.resume).parent.resolve() != output_dir.resolve():
        raise ValueError("--resume must point at a checkpoint inside this run's output_dir")
    preflight_path = output_dir / "preflight.json"
    if not preflight_path.exists():
        raise RuntimeError("preflight.json missing; run experiments.rl_lm_preflight until it passes")
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "ready":
        raise RuntimeError(f"preflight status is {preflight.get('status')!r}; trainer requires 'ready'")

    device = torch.device(config.get("runtime.device"))
    model, _ = load_backbone(config, device)

    backbone_identity = {
        "checkpoint_path": config.get("backbone.checkpoint_path"),
        "checkpoint_sha256": sha256_file(config.get("backbone.checkpoint_path")),
        "config_path": config.get("backbone.config_path"),
        "config_sha256": sha256_file(config.get("backbone.config_path")),
        "tokenizer_sha256": sha256_file(config.get("data.tokenizer_path")),
    }
    data_identity = {}
    caches = {}
    for role, split in (
        ("controller_train", config.get("data.train_split")),
        ("selection", config.get("data.selection_split")),
        ("calibration", config.get("data.calibration_split")),
    ):
        expected = cache_identity(
            model,
            model.config,
            checkpoint_path=config.get("backbone.checkpoint_path"),
            config_path=config.get("backbone.config_path"),
            tokenizer_path=config.get("data.tokenizer_path"),
            data_dir=config.get("data.corpus_dir"),
            split=split,
            sampling_seed=config.get("data.sampling_seed"),
            backend_version=config.get("environment.training_backend"),
            head_contract=config.get("environment.head_contract"),
            expert_chunk=config.get("environment.expert_chunk"),
        )
        caches[role] = load_cache(Path(config.get("data.cache_root")) / split, expected)
        data_identity[role] = expected

    trainer = Trainer(
        config,
        model,
        train_contexts=CacheContextProvider(caches["controller_train"]),
        evaluator=OnlineSelectionEvaluator(config, model, device),
        device=device,
        resume_path=args.resume,
        warmstart_path=args.warmstart_actor,
        stop_after=args.stop_after,
        backbone_identity=backbone_identity,
        data_identity=data_identity,
    )
    summary = trainer.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
