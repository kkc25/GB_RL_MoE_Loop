from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F

from data.packed_lm import PackedLMBatchSampler, PackedLMDataset
from train.common import MetricWriter, autocast_context, learning_rate_at
from train.recovery_common import atomic_torch_save, restore_rng_state, sha256_file


def packed_batch(dataset, indices, device):
    items = [dataset[index] for index in indices]
    return {
        "input_ids": torch.from_numpy(__import__("numpy").stack([item["input_ids"] for item in items])).to(device),
        "labels": torch.from_numpy(__import__("numpy").stack([item["labels"] for item in items])).to(device),
        "segment_ids": torch.from_numpy(__import__("numpy").stack([item["segment_ids"] for item in items])).to(device),
        "n_tokens": sum(item["n_tokens"] for item in items),
        "n_supervised_tokens": sum(item["n_supervised_tokens"] for item in items),
    }


def lm_loss(logits, labels):
    valid = labels != -100
    count = valid.sum()
    loss_sum = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum")
    return loss_sum / count.clamp_min(1), loss_sum, count


@torch.no_grad()
def evaluate_lm(model, dataset, config, device, depth, max_batches):
    model.eval()
    loss_sum, count = 0.0, 0
    sampler = PackedLMBatchSampler(dataset, config.train.batch_size, shuffle=False, drop_last=False)
    for batch_index, indices in enumerate(sampler):
        if batch_index >= max_batches:
            break
        batch = packed_batch(dataset, indices, device)
        with autocast_context(device, config.train.dtype):
            output = model(batch["input_ids"], batch["labels"], depth=depth, segment_ids=batch["segment_ids"])
        _, batch_sum, batch_count = lm_loss(output.logits, batch["labels"])
        loss_sum += batch_sum.item()
        count += batch_count.item()
    nll = loss_sum / max(1, count)
    return {"nll": nll, "ppl": float(torch.tensor(nll).exp()), "supervised_tokens": count, "depth": depth}


def save_lm_checkpoint(path, config, model, optimizer, update, ledger, best_nll, schedule, data_hash):
    atomic_torch_save(path, {
        "config": asdict(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "ledger": ledger,
        "best_nll": best_nll,
        "schedule": schedule,
        "data_hash": data_hash,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    })


def train_lm(model, train_dataset, validation_dataset, config, device, writer, data_hash=None):
    run = config.train
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": run.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=run.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    ledger = {"updates": 0, "input_tokens": 0, "supervised_tokens": 0, "ffn_rounds": 0, "expert_token_pairs": 0, "elapsed_seconds": 0.0, "allocated_gpu_hours": 0.0, "peak_memory_bytes": 0, "schedule_phases": []}
    schedule = {"updates": run.updates, "learning_rate": run.learning_rate, "warmup_fraction": run.warmup_fraction}
    start_update = 0
    best_nll = float("inf")
    if run.resume:
        checkpoint = torch.load(run.resume, map_location=device, weights_only=False)
        checkpoint_hash = checkpoint.get("data_hash")
        if checkpoint_hash is not None and data_hash is not None and checkpoint_hash != data_hash:
            raise ValueError(f"checkpoint data hash {checkpoint_hash} does not match current data hash {data_hash}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = checkpoint["update"]
        ledger = checkpoint["ledger"]
        if "ffn_rounds" not in ledger:
            # Legacy checkpoints stored per-token FFN rounds as "expert_calls".
            ledger["ffn_rounds"] = ledger.pop("expert_calls", 0)
        else:
            ledger.pop("expert_calls", None)
        ledger.setdefault("expert_token_pairs", ledger["ffn_rounds"] * config.model.top_k)
        ledger.setdefault("schedule_phases", [])
        best_nll = checkpoint.get("best_nll", float("inf"))
        previous_schedule = checkpoint.get("schedule")
        if previous_schedule is not None and previous_schedule != schedule:
            ledger["schedule_phases"].append({"schedule": previous_schedule, "ended_at_update": start_update})
            writer.write(phase="schedule", update=start_update, previous_schedule=previous_schedule, current_schedule=schedule)
        restore_rng_state(checkpoint)
    batches_per_epoch = len(train_dataset) // run.batch_size
    started = time.monotonic()
    last_accounted = started

    def accumulate_elapsed() -> None:
        nonlocal last_accounted
        now = time.monotonic()
        ledger["elapsed_seconds"] += now - last_accounted
        if device.type == "cuda":
            ledger["allocated_gpu_hours"] += (now - last_accounted) / 3600
        last_accounted = now

    model.train()
    iterator = None
    current_epoch = None
    for update in range(start_update, run.updates):
        epoch, start_batch = divmod(update, batches_per_epoch)
        if epoch != current_epoch:
            sampler = PackedLMBatchSampler(train_dataset, run.batch_size, shuffle=True, seed=run.seed, epoch=epoch, start_batch=start_batch)
            iterator = iter(sampler)
            current_epoch = epoch
        indices = next(iterator)
        batch = packed_batch(train_dataset, indices, device)
        if run.depth_schedule == "random":
            depth = 1 + (update + run.seed) % config.model.max_calls
        else:
            depth = run.fixed_depth
        lr = learning_rate_at(update, run.updates, run.learning_rate, run.warmup_fraction)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, run.dtype):
            output = model(batch["input_ids"], batch["labels"], depth=depth, segment_ids=batch["segment_ids"])
        loss, _, supervised = lm_loss(output.logits, batch["labels"])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), run.gradient_clip)
        optimizer.step()
        rounds = int(output.calls.sum().item())
        ledger["updates"] += 1
        ledger["input_tokens"] += batch["n_tokens"]
        ledger["supervised_tokens"] += int(supervised.item())
        ledger["ffn_rounds"] += rounds
        ledger["expert_token_pairs"] += rounds * config.model.top_k
        if update % run.log_interval == 0:
            writer.write(phase="train", update=update, loss=loss.item(), depth=depth, lr=lr, grad_norm=float(grad_norm), supervised_tokens=ledger["supervised_tokens"])
        if (update + 1) % run.validation_interval == 0 or update + 1 == run.updates:
            metrics = evaluate_lm(model, validation_dataset, config, device, run.fixed_depth, run.validation_batches)
            writer.write(phase="validation", update=update + 1, **metrics)
            accumulate_elapsed()
            if device.type == "cuda":
                ledger["peak_memory_bytes"] = max(ledger["peak_memory_bytes"], torch.cuda.max_memory_allocated(device))
            if metrics["nll"] < best_nll:
                best_nll = metrics["nll"]
                save_lm_checkpoint(f"{run.output_dir}/best_validation.pt", config, model, optimizer, update + 1, ledger, best_nll, schedule, data_hash)
            save_lm_checkpoint(f"{run.output_dir}/latest.pt", config, model, optimizer, update + 1, ledger, best_nll, schedule, data_hash)
            model.train()
    accumulate_elapsed()
    if device.type == "cuda":
        ledger["peak_memory_bytes"] = max(ledger["peak_memory_bytes"], torch.cuda.max_memory_allocated(device))
    Path(f"{run.output_dir}/cost_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    writer.write(phase="checkpoint", path=f"{run.output_dir}/latest.pt", sha256=sha256_file(f"{run.output_dir}/latest.pt"))
    return ledger
