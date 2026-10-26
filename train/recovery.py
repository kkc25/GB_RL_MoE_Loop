from __future__ import annotations

import math
import time
import torch
from torch.nn import functional as F

from eval.recovery_metrics import query_loss_and_metrics
from train.common import MetricWriter, autocast_context
from train.recovery_common import RecoveryLedger, query_labels, recovery_checkpoint, restore_ledger_fields, restore_rng_state, sha256_file


def batch_indices(size: int, batch_size: int, update: int, seed: int, shuffle: bool) -> list[int]:
    start = update * batch_size
    remaining = min(batch_size, size)
    indices = []
    while len(indices) < remaining:
        epoch, offset = divmod(start, size)
        if shuffle:
            generator = torch.Generator().manual_seed(seed + epoch)
            permutation = torch.randperm(size, generator=generator).tolist()
        else:
            permutation = list(range(size))
        take = min(remaining - len(indices), size - offset)
        indices.extend(permutation[offset : offset + take])
        start += take
    return indices


def move_recovery_batch(batch: dict, device: torch.device) -> dict:
    moved = dict(batch)
    for key in ("input_ids", "query_positions", "target_answer_ids", "attention_mask"):
        if key in batch:
            moved[key] = batch[key].to(device)
    if "labels" in batch:
        moved["labels"] = batch["labels"].to(device)
    return moved


def recovery_learning_rate(update: int, total: int, peak: float, warmup: int) -> float:
    if update < warmup:
        return peak * (update + 1) / max(1, warmup)
    progress = (update - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, progress))))


@torch.no_grad()
def evaluate_recovery(model, dataset, collate, config, device: torch.device, depth: int) -> dict:
    model.eval()
    totals = {"ce_sum": 0.0, "conditional_sum": 0.0, "mass_sum": 0.0, "correct_count": 0, "eval_count": 0}
    modulus = dataset.modulus
    answer_ids = torch.arange(5, 5 + modulus, device=device)
    for start in range(0, len(dataset), config.run.batch_size):
        indices = list(range(start, min(start + config.run.batch_size, len(dataset))))
        batch = move_recovery_batch(collate([dataset[index] for index in indices]), device)
        labels = query_labels(batch)
        with autocast_context(device, config.run.dtype):
            output = model(batch["input_ids"], labels, depth=depth)
        _, metrics = query_loss_and_metrics(
            output.logits,
            batch["input_ids"],
            batch["query_positions"],
            batch["target_answer_ids"],
            answer_ids,
        )
        totals["ce_sum"] += metrics["ce_sum"].item()
        totals["conditional_sum"] += metrics["ce_conditional_sum"].item()
        totals["mass_sum"] += metrics["answer_mass_sum"].item()
        totals["correct_count"] += metrics["correct_count"].item()
        totals["eval_count"] += metrics["eval_count"]
    count = totals["eval_count"]
    return {
        "ce_full": totals["ce_sum"] / count,
        "ce_answer_conditional": totals["conditional_sum"] / count,
        "answer_mass": totals["mass_sum"] / count,
        "correct_count": totals["correct_count"],
        "eval_count": count,
        "accuracy": totals["correct_count"] / count,
        "depth": depth,
    }


def train_recovery(model, datasets: dict, collate, config, device: torch.device, ledger: RecoveryLedger, writer: MetricWriter, data_hash: str) -> None:
    run = config.run
    train_set = datasets["train"]
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": run.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=run.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    start_update = 0
    best_validation_loss = float("inf")
    schedule = {"updates": run.updates, "learning_rate": run.learning_rate, "warmup_updates": run.warmup_updates}
    if run.resume:
        checkpoint = torch.load(run.resume, map_location=device, weights_only=False)
        if checkpoint["data_hash"] != data_hash:
            raise ValueError("checkpoint data hash does not match current task")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restored_ledger = RecoveryLedger(**restore_ledger_fields(checkpoint["ledger"], config.model.top_k))
        ledger.__dict__.update(restored_ledger.__dict__)
        start_update = checkpoint["update"]
        best_validation_loss = checkpoint.get("best_validation_loss")
        if best_validation_loss is None:
            best_validation_loss = float("inf")
        previous_schedule = checkpoint.get("schedule")
        if previous_schedule is not None and previous_schedule != schedule:
            ledger.schedule_phases.append({"schedule": previous_schedule, "ended_at_update": start_update})
            writer.write(phase="schedule", update=start_update, previous_schedule=previous_schedule, current_schedule=schedule)
        restore_rng_state(checkpoint)
    started = time.monotonic()
    last_accounted = started

    def accumulate_elapsed() -> None:
        nonlocal last_accounted
        now = time.monotonic()
        ledger.elapsed_seconds += now - last_accounted
        if device.type == "cuda":
            ledger.allocated_gpu_hours += (now - last_accounted) / 3600
        last_accounted = now

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    answer_ids = torch.arange(5, 5 + train_set.modulus, device=device)
    for update in range(start_update, run.updates):
        indices = batch_indices(len(train_set), run.batch_size, update, run.seed, run.shuffle_each_epoch)
        batch = move_recovery_batch(collate([train_set[index] for index in indices]), device)
        labels = query_labels(batch)
        if run.depth_schedule == "fixed":
            depth = run.fixed_depth
        elif run.depth_schedule == "random":
            generator = torch.Generator(device="cpu").manual_seed(run.seed + update)
            depth = int(torch.randint(1, config.model.max_calls + 1, (1,), generator=generator).item())
        else:
            raise ValueError(f"unknown depth schedule {run.depth_schedule}")
        lr = recovery_learning_rate(update, run.updates, run.learning_rate, run.warmup_updates)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, run.dtype):
            output = model(batch["input_ids"], labels, depth=depth)
        loss, metrics = query_loss_and_metrics(output.logits, batch["input_ids"], batch["query_positions"], batch["target_answer_ids"], answer_ids)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), run.gradient_clip)
        optimizer.step()
        targets = batch["target_answer_ids"].numel()
        rounds = int(output.calls.sum().item())
        ledger.optimizer_updates += 1
        ledger.supervised_targets += targets
        ledger.input_tokens += int(batch["attention_mask"].sum().item()) if "attention_mask" in batch else batch["input_ids"].numel()
        ledger.ffn_rounds += rounds
        ledger.expert_token_pairs += rounds * config.model.top_k
        if update % run.log_interval == 0:
            writer.write(phase="train", update=update, loss=loss.item(), depth=depth, lr=lr, grad_norm=float(gradient_norm), supervised_targets=ledger.supervised_targets)
        if (update + 1) % run.validation_interval == 0 or update + 1 == run.updates:
            validation_results = {}
            for split, dataset in datasets.items():
                result = evaluate_recovery(model, dataset, collate, config, device, run.fixed_depth)
                validation_results[split] = result
                writer.write(phase="validation", update=update + 1, split=split, **result)
            selection_split = "id_validation" if "id_validation" in validation_results else "train"
            selection_loss = validation_results[selection_split]["ce_full"]
            accumulate_elapsed()
            if device.type == "cuda":
                ledger.peak_memory_bytes = max(ledger.peak_memory_bytes, torch.cuda.max_memory_allocated(device))
            if selection_loss < best_validation_loss:
                best_validation_loss = selection_loss
                recovery_checkpoint(f"{run.output_dir}/best_validation.pt", config, model, optimizer, update + 1, ledger, data_hash, best_validation_loss, schedule)
            model.train()
            recovery_checkpoint(f"{run.output_dir}/latest.pt", config, model, optimizer, update + 1, ledger, data_hash, best_validation_loss, schedule)
    accumulate_elapsed()
    if device.type == "cuda":
        ledger.peak_memory_bytes = max(ledger.peak_memory_bytes, torch.cuda.max_memory_allocated(device))
    ledger.save(f"{run.output_dir}/cost_ledger.json")
    checkpoint_path = f"{run.output_dir}/latest.pt"
    writer.write(phase="checkpoint", update=run.updates, path=checkpoint_path, sha256=sha256_file(checkpoint_path))
