"""S1 supervised adaptation CLI (spec 18.2/18.3, task card G).

Pipeline:

1. derive the S1 caches (new ``rl-lm-single-cache-v1`` identity) from the
   pair-v1 payload after re-verifying every prefix-defining hash against the
   loaded teacher; single baseline records are recomputed with the frozen
   teacher at amplitude 1;
2. adapt only ``blocks.11.moe.pool.w1/w2/w3`` of a student copy with
   L = CE(student, y) + KL(teacher || student) + 0.1 * per-token relative
   squared update error (pre-gamma updates), teacher fully detached;
3. every 100 updates run the full-selection online evaluation (fixed argmax
   single-expert rule, amplitude 1) against the original pair teacher;
4. at the end, measure the real terminal-FFN execution time of the single
   sparse executor vs the original top-2 executor, then decide the gate:
   selection NLL degradation <= 0.02 nat AND faster real single execution.
   The adapted student checkpoint is written with its sha256 regardless;
   ``gate.json`` records the pass/fail that ``rl_lm_single_train`` requires.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.packed_lm import PackedLMDataset
from rl_lm.cache import load_cache
from rl_lm.cli import load_backbone, write_resolved_config
from rl_lm.inference import CanonicalTeacherPolicy, forward_with_last_ffn_policy
from rl_lm.readout import terminal_ce
from rl_lm.single_config import (
    SingleAdaptConfig,
    resolve_single_config,
    single_cache_identity,
)
from rl_lm.single_executor import (
    ArgmaxSingleTeacherPolicy,
    SingleLedger,
    single_argmax_expert,
    single_ffn_update,
    single_online_forward,
    make_single_executor,
    SINGLE_SPARSE_BACKEND,
)
from rl_lm.single_readout import adapt_ce_loss, adapt_kl_loss, differentiable_logits, relative_update_error
from train.recovery_common import atomic_json, sha256_file

ADAPT_CHECKPOINT_FORMAT = "rl-lm-single-adapt-v1"


# ------------------------------------------------------------------ cache


def _recompute_single_baseline(model, h0: np.ndarray, targets: np.ndarray, device, chunk: int, compute_dtype: str) -> list[dict]:
    """Single-expert teacher records for every cached state (amplitude 1)."""
    layer = int(model.config.adaptive_layers[-1])
    block = model.blocks[layer]
    gamma = float(model.gammas[layer])
    records = []
    count = h0.shape[0]
    with torch.no_grad():
        for start in range(0, count, chunk):
            h = torch.from_numpy(np.ascontiguousarray(h0[start:start + chunk])).to(device)
            y = torch.from_numpy(np.ascontiguousarray(targets[start:start + chunk])).to(device)
            n = block.ffn_norm(h).float()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(compute_dtype == "bfloat16")):
                z = block.moe.gate(n)
            z = z.float()
            experts = single_argmax_expert(z)
            top2 = z.topk(2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(compute_dtype == "bfloat16")):
                update = single_ffn_update(block.moe.pool, n, experts)
            hidden = h + gamma * update
            ce = terminal_ce(model, hidden, y, chunk=2048)
            for row in range(h.shape[0]):
                records.append(
                    {
                        "single_expert": int(experts[row]),
                        "gate_margin": float(margin[row]),
                        "single_teacher_ce": float(ce[row]),
                    }
                )
    return records


def derive_single_cache(config, model, split: str, device) -> Path:
    """Create (or load) the S1 cache for ``split`` with the new identity.

    The numeric prefix payload is reused only after every prefix-defining
    field of the pair manifest is re-verified against the loaded teacher;
    the manifest is a new cache/environment version and the single baseline
    is always recomputed.
    """
    pair_dir = Path(config.get("data.pair_cache_root")) / split
    output_dir = Path(config.get("data.cache_root")) / split
    pair_manifest = json.loads((pair_dir / "manifest.json").read_text())
    if pair_manifest.get("cache_format") != "rl-lm-cache-v1":
        raise ValueError(f"{pair_dir} is not a pair-v1 cache")
    identity = single_cache_identity(
        model,
        checkpoint_path=config.get("backbone.checkpoint_path"),
        config_path=config.get("backbone.config_path"),
        tokenizer_path=config.get("data.tokenizer_path"),
        data_dir=config.get("data.corpus_dir"),
        split=split,
        sampling_seed=config.get("data.sampling_seed"),
        expert_chunk=config.get("environment.expert_chunk"),
        pair_manifest=pair_manifest,
    )
    if output_dir.exists():
        load_cache(output_dir, identity)  # validates the existing manifest
        return output_dir
    staging = output_dir.with_name(output_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in ("h0.npy", "targets.npy", "positions.npy", "sequence_indices.npy", "episode_ids.jsonl", "sampling_report.json"):
        shutil.copy2(pair_dir / name, staging / name)
    h0 = np.load(pair_dir / "h0.npy", mmap_mode="r")
    targets = np.load(pair_dir / "targets.npy", mmap_mode="r")
    pair_baseline = [json.loads(line) for line in (pair_dir / "baseline.jsonl").read_text().splitlines()]
    single_records = _recompute_single_baseline(
        model, h0, targets, device, chunk=4096, compute_dtype=config.get("environment.frozen_compute_dtype")
    )
    if len(single_records) != len(pair_baseline):
        raise RuntimeError("single baseline length mismatch")
    rows = []
    for single, pair in zip(single_records, pair_baseline):
        rows.append(single | {"pair_canonical_ce": pair["canonical_ce"]})
    with (staging / "baseline.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    atomic_json(staging / "manifest.json", identity | {"n": int(h0.shape[0]), "d": int(h0.shape[1])})
    staging.rename(output_dir)
    return output_dir


# ------------------------------------------------------------------ student


WHITELIST = ("blocks.11.moe.pool.w1", "blocks.11.moe.pool.w2", "blocks.11.moe.pool.w3")


def build_student(teacher, whitelist=WHITELIST) -> torch.nn.Module:
    """Deep copy of the backbone; only the whitelisted pool tensors trainable."""
    student = copy.deepcopy(teacher)
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    named = dict(student.named_parameters())
    for name in whitelist:
        if name not in named:
            raise ValueError(f"whitelist parameter {name} not present in the student")
        named[name].requires_grad_(True)
    trainable = sorted(name for name, p in student.named_parameters() if p.requires_grad)
    if trainable != sorted(whitelist):
        raise RuntimeError(f"trainable whitelist mismatch: {trainable}")
    return student


def assert_frozen_clean(teacher, student, whitelist=WHITELIST) -> None:
    for name, parameter in teacher.named_parameters():
        if parameter.requires_grad or parameter.grad is not None:
            raise RuntimeError(f"teacher parameter {name} is trainable or carries a gradient")
    for name, parameter in student.named_parameters():
        if name in whitelist:
            continue
        if parameter.requires_grad or parameter.grad is not None:
            raise RuntimeError(f"frozen student parameter {name} is trainable or carries a gradient")


def adapt_lr(j: int, *, peak: float, final: float, warmup: int, total: int) -> float:
    """1-based schedule: linear warmup, then cosine decay to ``final``."""
    if j < 1:
        raise ValueError("j is 1-based")
    if j <= warmup:
        return peak * j / warmup
    progress = min(1.0, (j - warmup) / max(1, total - warmup))
    return final + (peak - final) * (1.0 + math.cos(math.pi * progress)) / 2.0


# ------------------------------------------------------------------ eval


def _selection_tensors(config, device):
    dataset = PackedLMDataset(config.get("data.corpus_dir"), config.get("data.selection_split"))
    items = [dataset[i] for i in range(len(dataset))]
    input_ids = torch.from_numpy(np.stack([item["input_ids"] for item in items])).to(device)
    segment_ids = torch.from_numpy(np.stack([item["segment_ids"] for item in items])).to(device)
    labels = torch.from_numpy(np.stack([item["labels"] for item in items])).to(device)
    return input_ids, segment_ids, labels


def _nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    ce_sum = 0.0
    rows = 8192
    for start in range(0, flat_labels.shape[0], rows):
        ce_sum += float(
            torch.nn.functional.cross_entropy(
                flat_logits[start:start + rows].float(),
                flat_labels[start:start + rows],
                reduction="none",
                ignore_index=-100,
            ).sum()
        )
    return ce_sum / int((labels != -100).sum())


def pair_teacher_selection_nll(teacher, input_ids, segment_ids, labels, *, compute_dtype, chunk, head_chunk, seq_batch=8) -> float:
    """Full-selection NLL of the original pair teacher (canonical pair, top-2 weights)."""
    total = 0.0
    tokens = 0
    for start in range(0, input_ids.shape[0], seq_batch):
        ids = input_ids[start:start + seq_batch]
        seg = segment_ids[start:start + seq_batch]
        lab = labels[start:start + seq_batch]
        output = forward_with_last_ffn_policy(
            teacher,
            CanonicalTeacherPolicy(),
            ids,
            seg,
            max_calls=1,
            backend="sparse_merged_v1",
            compute_dtype=compute_dtype,
            chunk=chunk,
            head_chunk=head_chunk,
        )
        flat_logits = output.logits.reshape(-1, output.logits.shape[-1])
        flat_labels = lab.reshape(-1)
        rows = 8192
        for r in range(0, flat_labels.shape[0], rows):
            total += float(
                torch.nn.functional.cross_entropy(
                    flat_logits[r:r + rows].float(), flat_labels[r:r + rows], reduction="none", ignore_index=-100
                ).sum()
            )
        tokens += int((lab != -100).sum())
    return total / tokens


def single_selection_nll(student, input_ids, segment_ids, labels, *, compute_dtype, chunk, head_chunk, seq_batch=8) -> float:
    """Full-selection NLL of the fixed argmax single-expert rule (amplitude 1)."""
    total = 0.0
    tokens = 0
    for start in range(0, input_ids.shape[0], seq_batch):
        ids = input_ids[start:start + seq_batch]
        seg = segment_ids[start:start + seq_batch]
        lab = labels[start:start + seq_batch]
        logits, _rounds = single_online_forward(
            student,
            ArgmaxSingleTeacherPolicy(),
            ids,
            seg,
            max_calls=1,
            backend=SINGLE_SPARSE_BACKEND,
            compute_dtype=compute_dtype,
            chunk=chunk,
            head_chunk=head_chunk,
        )
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = lab.reshape(-1)
        rows = 8192
        for r in range(0, flat_labels.shape[0], rows):
            total += float(
                torch.nn.functional.cross_entropy(
                    flat_logits[r:r + rows].float(), flat_labels[r:r + rows], reduction="none", ignore_index=-100
                ).sum()
            )
        tokens += int((lab != -100).sum())
    return total / tokens


def measure_executor_improvement(student, device, *, rows: int = 8192, warmup: int = 30, repeats: int = 200) -> dict:
    """Real terminal-FFN execution time: original top-2 vs single sparse.

    Both paths run on the same random hidden batch under BF16 production
    precision with CUDA-event timing; the single executor must be faster for
    the gate to pass.
    """
    layer = int(student.config.adaptive_layers[-1])
    block = student.blocks[layer]
    gamma = float(student.gammas[layer])
    torch.manual_seed(20261027)
    h = torch.randn(rows, student.config.hidden_size, device=device, dtype=torch.float32)
    with torch.no_grad():
        n = block.ffn_norm(h).float()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            z = block.moe.gate(n)
        experts = single_argmax_expert(z.float())

        def original_top2():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                update, _, _, _ = block.executor(n)
            return h + gamma * update

        def single_sparse():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                update = single_ffn_update(block.moe.pool, n, experts)
            return h + gamma * update

        def time_fn(fn):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fn()
            end.record()
            torch.cuda.synchronize(device)
            return start.elapsed_time(end) / repeats

        original_ms = time_fn(original_top2)
        single_ms = time_fn(single_sparse)
    return {
        "rows": rows,
        "warmup": warmup,
        "repeats": repeats,
        "original_top2_ms": original_ms,
        "single_sparse_ms": single_ms,
        "speedup": original_ms / single_ms,
        "improved": single_ms < original_ms,
    }


# ------------------------------------------------------------------ trainer


class AdaptTrainer:
    def __init__(self, config, teacher, *, train_cache, device, run_dir, resume_path=None, stop_after=None) -> None:
        self.config = config
        self.teacher = teacher
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.peak_lr = float(config.get("adaptation.learning_rate"))
        self.final_lr = float(config.get("adaptation.final_learning_rate"))
        self.warmup = int(config.get("adaptation.warmup_updates"))
        self.planned = int(config.get("adaptation.max_updates"))
        self.batch = int(config.get("adaptation.batch_contexts"))
        self.clip = float(config.get("adaptation.gradient_clip"))
        self.kl_coeff = float(config.get("adaptation.kl_coefficient"))
        self.update_coeff = float(config.get("adaptation.update_error_coefficient"))
        self.eval_interval = int(config.get("adaptation.eval_interval_updates"))
        self.log_interval = int(config.get("logging.interval_updates"))
        self.gamma = float(teacher.gammas[int(teacher.config.adaptive_layers[-1])])
        self.compute_dtype = config.get("environment.frozen_compute_dtype")
        self.whitelist = tuple(config.get("adaptation.trainable_whitelist"))
        self.stop_after = self.planned if stop_after is None else min(int(stop_after), self.planned)
        if self.stop_after < 1:
            raise ValueError("--stop-after must be a positive absolute cap")

        self._h0 = train_cache["h0"]
        self._targets = train_cache["targets"]
        self._episode_ids = tuple(row["episode_id"] for row in train_cache["episode_ids"])
        if len(self._h0) < self.batch:
            raise ValueError("train cache has fewer contexts than the batch")

        self.ledger = SingleLedger()
        self.resumed = resume_path is not None
        if self.resumed:
            self._restore(Path(resume_path))
        else:
            if (self.run_dir / "adapt_checkpoint_latest.pt").exists():
                raise FileExistsError("run dir already contains an adaptation run; use --resume")
            seed = 20261027
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.student = build_student(teacher, self.whitelist).to(device)
            self.optimizer = torch.optim.AdamW(
                [p for p in self.student.parameters() if p.requires_grad],
                lr=0.0,
                betas=tuple(config.get("adaptation.betas")),
                eps=float(config.get("adaptation.epsilon")),
                weight_decay=float(config.get("adaptation.weight_decay")),
            )
            self.successful_updates = 0
            self.epoch = 0
            self.epoch_cursor = 0
            self._permutation = None
            self._visited = set()
            self.initial_pool = {
                name: dict(self.student.named_parameters())[name].detach().clone()
                for name in self.whitelist
            }
        self.layer = int(teacher.config.adaptive_layers[-1])
        self.teacher_block = teacher.blocks[self.layer]
        self.student_block = self.student.blocks[self.layer]
        self.start_time = time.time()

    # ------------------------------ data

    def _epoch_permutation(self) -> np.ndarray:
        key = json.dumps(["rl-lm-single-adapt-permutation-v1", self.epoch], separators=(",", ":"))
        seed = int.from_bytes(__import__("hashlib").sha256(key.encode()).digest()[:16], "little")
        return np.random.Generator(np.random.Philox(seed)).permutation(len(self._h0))

    def _next_batch_indices(self):
        if self._permutation is None:
            self._permutation = self._epoch_permutation()
        indices = [int(i) for i in self._permutation[self.epoch_cursor:self.epoch_cursor + self.batch]]
        self.epoch_cursor += self.batch
        if self.epoch_cursor >= (len(self._h0) // self.batch) * self.batch:
            self.epoch += 1
            self.epoch_cursor = 0
            self._permutation = None
        return indices

    # ------------------------------ update

    def _update(self) -> dict:
        assert_frozen_clean(self.teacher, self.student, self.whitelist)
        indices = self._next_batch_indices()
        index = torch.as_tensor(indices, dtype=torch.int64)
        h0 = torch.from_numpy(np.ascontiguousarray(self._h0[index.numpy()])).to(self.device)
        targets = torch.from_numpy(np.ascontiguousarray(self._targets[index.numpy()])).to(self.device)

        with torch.no_grad():
            n = self.teacher_block.ffn_norm(h0).float()
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=(self.compute_dtype == "bfloat16")):
                z = self.teacher_block.moe.gate(n)
            experts = single_argmax_expert(z.float())
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=(self.compute_dtype == "bfloat16")):
                u_teacher = single_ffn_update(self.teacher_block.moe.pool, n, experts)
            h_teacher = h0 + self.gamma * u_teacher
            teacher_logits = differentiable_logits(self.teacher, h_teacher).detach()

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=(self.compute_dtype == "bfloat16")):
            u_student = single_ffn_update(self.student_block.moe.pool, n, experts)
        h_student = h0 + self.gamma * u_student
        student_logits = differentiable_logits(self.student, h_student)

        ce = adapt_ce_loss(student_logits, targets)
        kl = adapt_kl_loss(teacher_logits, student_logits)
        update_error = relative_update_error(u_student, u_teacher)
        loss = ce + self.kl_coeff * kl + self.update_coeff * update_error

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in self.student.named_parameters():
            if name in self.whitelist:
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"missing or non-finite gradient on {name}")
            elif parameter.grad is not None:
                raise RuntimeError(f"frozen student parameter {name} received a gradient")
        for name, parameter in self.teacher.named_parameters():
            if parameter.grad is not None:
                raise RuntimeError(f"teacher parameter {name} received a gradient")

        sample_counts = torch.bincount(experts.cpu(), minlength=8)
        named = dict(self.student.named_parameters())
        grad_norms = []
        for expert in range(8):
            sq = 0.0
            for short in ("w1", "w2", "w3"):
                grad = named[f"blocks.{self.layer}.moe.pool.{short}"].grad
                sq += float(grad[expert].square().sum())
            grad_norms.append(math.sqrt(sq))
        before = {name: named[name].detach().clone() for name in self.whitelist}

        trainable = [p for p in self.student.parameters() if p.requires_grad]
        total_norm = float(nn.utils.clip_grad_norm_(trainable, self.clip))
        if not math.isfinite(total_norm):
            raise RuntimeError("non-finite gradient norm")
        lr = adapt_lr(
            self.successful_updates + 1,
            peak=self.peak_lr,
            final=self.final_lr,
            warmup=self.warmup,
            total=self.planned,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.ledger.optimizer_attempts += 1
        self.successful_updates += 1
        self.ledger.successful_updates += 1
        self.ledger.context_visits += len(indices)
        self._visited.update(indices)
        self.ledger.unique_train_contexts = len(self._visited)
        self.ledger.merge_peak(self.device)

        delta_step = []
        delta_init = []
        for expert in range(8):
            step_sq = 0.0
            init_sq = 0.0
            for short in ("w1", "w2", "w3"):
                name = f"blocks.{self.layer}.moe.pool.{short}"
                step_sq += float((named[name].detach() - before[name])[expert].square().sum())
                init_sq += float((named[name].detach() - self.initial_pool[name])[expert].square().sum())
            delta_step.append(math.sqrt(step_sq))
            delta_init.append(math.sqrt(init_sq))

        return {
            "update": self.successful_updates,
            "lr": lr,
            "loss": float(loss.detach()),
            "ce": float(ce.detach()),
            "kl": float(kl.detach()),
            "update_error": float(update_error.detach()),
            "grad_norm_unclipped": total_norm,
            "grad_clipped": total_norm > self.clip,
            "expert_sample_counts": sample_counts.tolist(),
            "expert_grad_norms": grad_norms,
            "expert_weight_delta_step": delta_step,
            "expert_weight_delta_since_init": delta_init,
            "wall_time_seconds_cumulative": time.time() - self.start_time,
        }

    # ------------------------------ checkpoint

    def _payload(self) -> dict:
        return {
            "format_version": ADAPT_CHECKPOINT_FORMAT,
            "spec_version": "RL-LM-Single-v1.0",
            "saved_at_update": self.successful_updates,
            "resolved_config": self.config.to_dict(),
            "config_sha256": self.config.sha256,
            "student_state_dict": {k: v.detach().cpu() for k, v in self.student.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_state": {
                "successful_updates": self.successful_updates,
                "epoch": self.epoch,
                "next_batch_cursor": self.epoch_cursor,
                "visited_contexts": sorted(self._visited),
            },
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": list(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,
            },
            "initial_pool": {k: v.cpu() for k, v in self.initial_pool.items()},
            "ledger": self.ledger.to_dict(),
        }

    def _save(self) -> Path:
        path = self.run_dir / "adapt_checkpoint_latest.pt"
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            torch.save(self._payload(), handle)
            handle.flush()
            __import__("os").fsync(handle.fileno())
        __import__("os").replace(temporary, path)
        return path

    def _restore(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != ADAPT_CHECKPOINT_FORMAT:
            raise ValueError("not an S1 adaptation checkpoint")
        if payload.get("spec_version") != "RL-LM-Single-v1.0":
            raise ValueError("spec version mismatch")
        stored = copy.deepcopy(payload["resolved_config"])
        current = copy.deepcopy(self.config.to_dict())
        stored["runtime"]["device"] = current["runtime"]["device"] = "<agnostic>"
        if stored != current:
            raise ValueError("exact resume requires the identical resolved config (device may differ)")
        self.student = build_student(self.teacher, self.whitelist).to(self.device)
        self.student.load_state_dict(payload["student_state_dict"])
        self.optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=0.0,
            betas=tuple(self.config.get("adaptation.betas")),
            eps=float(self.config.get("adaptation.epsilon")),
            weight_decay=float(self.config.get("adaptation.weight_decay")),
        )
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)
        state = payload["training_state"]
        self.successful_updates = int(state["successful_updates"])
        self.epoch = int(state["epoch"])
        self.epoch_cursor = int(state["next_batch_cursor"])
        self._visited = set(state.get("visited_contexts", []))
        self._permutation = None
        rng = payload["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([entry.cpu() for entry in rng["cuda"]])
        self.initial_pool = {k: v.to(self.device) for k, v in payload["initial_pool"].items()}
        self.ledger = SingleLedger.from_dict(payload["ledger"])
        if self.stop_after < self.successful_updates:
            raise ValueError("--stop-after is below the completed updates")

    # ------------------------------ run

    def run(self, evaluate) -> dict:
        metrics_path = self.run_dir / "metrics.jsonl"
        evaluations = []
        if not self.resumed:
            evaluations.append(evaluate(self.student, self.successful_updates))
        while self.successful_updates < self.stop_after:
            entry = self._update()
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
            if entry["update"] % self.log_interval == 0:
                print(json.dumps({"update": entry["update"], "loss": entry["loss"], "ce": entry["ce"], "kl": entry["kl"], "update_error": entry["update_error"]}))
            if self.successful_updates % self.eval_interval == 0:
                evaluations.append(evaluate(self.student, self.successful_updates))
            self._save()
        if not evaluations or evaluations[-1]["update"] != self.successful_updates:
            evaluations.append(evaluate(self.student, self.successful_updates))
        return {"updates": self.successful_updates, "evaluations": evaluations}


# ------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--corpus-dir")
    parser.add_argument("--tokenizer")
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()

    config = resolve_single_config(
        args.config, SingleAdaptConfig,
        checkpoint=args.checkpoint, corpus_dir=args.corpus_dir, tokenizer=args.tokenizer, device=args.device,
    )
    output_dir = Path(config.get("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, output_dir / "resolved_config.json")
    device = torch.device(config.get("runtime.device"))
    teacher, _ = load_backbone(config, device)
    config.validate(backbone_model=teacher.config, runtime=True)

    caches = {}
    for split in (config.get("data.train_split"), config.get("data.selection_split"), config.get("data.calibration_split")):
        cache_dir = derive_single_cache(config, teacher, split, device)
        pair_manifest = json.loads((Path(config.get("data.pair_cache_root")) / split / "manifest.json").read_text())
        identity = single_cache_identity(
            teacher,
            checkpoint_path=config.get("backbone.checkpoint_path"),
            config_path=config.get("backbone.config_path"),
            tokenizer_path=config.get("data.tokenizer_path"),
            data_dir=config.get("data.corpus_dir"),
            split=split,
            sampling_seed=config.get("data.sampling_seed"),
            expert_chunk=config.get("environment.expert_chunk"),
            pair_manifest=pair_manifest,
        )
        caches[split] = load_cache(cache_dir, identity)
    atomic_json(
        output_dir / "preflight.json",
        {
            "status": "ready",
            "spec_version": "RL-LM-Single-v1.0",
            "stage": "adapt",
            "checkpoint_sha256": sha256_file(config.get("backbone.checkpoint_path")),
            "caches": {split: {"states": int(caches[split]["manifest"]["n"])} for split in caches},
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    input_ids, segment_ids, labels = _selection_tensors(config, device)
    reference_path = output_dir / "pair_teacher_selection.json"
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        if reference["checkpoint_sha256"] != sha256_file(config.get("backbone.checkpoint_path")):
            raise ValueError("cached pair teacher reference belongs to a different checkpoint")
        reference_nll = float(reference["nll"])
    else:
        reference_nll = pair_teacher_selection_nll(
            teacher, input_ids, segment_ids, labels,
            compute_dtype=config.get("environment.frozen_compute_dtype"),
            chunk=int(config.get("environment.expert_chunk")),
            head_chunk=int(config.get("environment.head_chunk")),
        )
        atomic_json(
            reference_path,
            {
                "nll": reference_nll,
                "checkpoint_sha256": sha256_file(config.get("backbone.checkpoint_path")),
                "sequences": int(input_ids.shape[0]),
                "tokens": int((labels != -100).sum()),
                "rule": "canonical pair teacher, sparse_merged_v1, full selection split",
            },
        )

    def evaluate(student, update: int) -> dict:
        nll = single_selection_nll(
            student, input_ids, segment_ids, labels,
            compute_dtype=config.get("environment.frozen_compute_dtype"),
            chunk=int(config.get("environment.expert_chunk")),
            head_chunk=int(config.get("environment.head_chunk")),
        )
        record = {
            "update": int(update),
            "student_single_nll": nll,
            "pair_teacher_nll": reference_nll,
            "degradation_nll": nll - reference_nll,
            "gate_tolerance_nll": float(config.get("adaptation.selection_gate_tolerance_nll")),
            "eval_sequences": int(input_ids.shape[0]),
            "eval_tokens": int((labels != -100).sum()),
        }
        with (output_dir / "evaluations.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        return record

    trainer = AdaptTrainer(
        config, teacher,
        train_cache=caches[config.get("data.train_split")],
        device=device,
        run_dir=output_dir,
        resume_path=args.resume,
        stop_after=args.stop_after,
    )
    started = time.perf_counter()
    result = trainer.run(evaluate)
    trainer.ledger.record_wall_time(time.perf_counter() - started)
    trainer.ledger.save(output_dir / "cost_ledger.json")

    final_eval = result["evaluations"][-1]
    timing = measure_executor_improvement(trainer.student, device)
    atomic_json(output_dir / "executor_timing.json", timing)
    quality_ok = final_eval["degradation_nll"] <= float(config.get("adaptation.selection_gate_tolerance_nll"))
    gate = {
        "passed": bool(quality_ok and timing["improved"]),
        "quality_ok": bool(quality_ok),
        "final_degradation_nll": final_eval["degradation_nll"],
        "gate_tolerance_nll": float(config.get("adaptation.selection_gate_tolerance_nll")),
        "timing_ok": bool(timing["improved"]),
        "timing": timing,
        "updates_completed": result["updates"],
        "single_rl_allowed": bool(quality_ok and timing["improved"]),
    }

    student_path = output_dir / "student_final.pt"
    temporary = student_path.with_name(student_path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(
            {
                "format_version": ADAPT_CHECKPOINT_FORMAT,
                "spec_version": "RL-LM-Single-v1.0",
                "kind": "adapted_student",
                "saved_at_update": result["updates"],
                "student_state_dict": {k: v.detach().cpu() for k, v in trainer.student.state_dict().items()},
                "teacher_checkpoint_sha256": sha256_file(config.get("backbone.checkpoint_path")),
                "gate": gate,
            },
            handle,
        )
        handle.flush()
        __import__("os").fsync(handle.fileno())
    __import__("os").replace(temporary, student_path)
    gate["student_checkpoint"] = str(student_path)
    gate["student_checkpoint_sha256"] = sha256_file(student_path)
    atomic_json(output_dir / "gate.json", gate)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
