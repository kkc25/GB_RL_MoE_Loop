from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F

from data.packed_lm import PackedLMBatchSampler, PackedLMDataset
from train.common import MetricWriter, autocast_context, learning_rate_at
from train.recovery_common import atomic_torch_save, sha256_file


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


def save_lm_checkpoint(path, config, model, optimizer, update, ledger):
    atomic_torch_save(path, {
        "config": asdict(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "ledger": ledger,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    })


def train_lm(model, train_dataset, validation_dataset, config, device, writer):
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
    ledger = {"updates": 0, "input_tokens": 0, "supervised_tokens": 0, "expert_calls": 0, "elapsed_seconds": 0.0, "allocated_gpu_hours": 0.0, "peak_memory_bytes": 0}
    start_update = 0
    if run.resume:
        checkpoint = torch.load(run.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = checkpoint["update"]
        ledger = checkpoint["ledger"]
    sampler_size = len(PackedLMBatchSampler(train_dataset, run.batch_size, shuffle=True, seed=run.seed).batches_per_epoch * [0])
    batches_per_epoch = len(train_dataset) // run.batch_size
    started = time.monotonic()
    model.train()
    best_nll = float("inf")
    for update in range(start_update, run.updates):
        epoch, start_batch = divmod(update, batches_per_epoch)
        sampler = PackedLMBatchSampler(train_dataset, run.batch_size, shuffle=True, seed=run.seed, epoch=epoch, start_batch=start_batch)
        indices = next(iter(sampler))
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
        ledger["updates"] += 1
        ledger["input_tokens"] += batch["n_tokens"]
        ledger["supervised_tokens"] += int(supervised.item())
        ledger["expert_calls"] += int(output.calls.sum().item())
        if update % run.log_interval == 0:
            writer.write(phase="train", update=update, loss=loss.item(), depth=depth, lr=lr, grad_norm=float(grad_norm), supervised_tokens=ledger["supervised_tokens"])
        if (update + 1) % run.validation_interval == 0 or update + 1 == run.updates:
            metrics = evaluate_lm(model, validation_dataset, config, device, run.fixed_depth, run.validation_batches)
            writer.write(phase="validation", update=update + 1, **metrics)
            save_lm_checkpoint(f"{run.output_dir}/latest.pt", config, model, optimizer, update + 1, ledger)
            if metrics["nll"] < best_nll:
                best_nll = metrics["nll"]
                save_lm_checkpoint(f"{run.output_dir}/best_validation.pt", config, model, optimizer, update + 1, ledger)
            model.train()
    elapsed = time.monotonic() - started
    ledger["elapsed_seconds"] += elapsed
    ledger["allocated_gpu_hours"] += elapsed / 3600 if device.type == "cuda" else 0.0
    if device.type == "cuda":
        ledger["peak_memory_bytes"] = max(ledger["peak_memory_bytes"], torch.cuda.max_memory_allocated(device))
    Path(f"{run.output_dir}/cost_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    writer.write(phase="checkpoint", path=f"{run.output_dir}/latest.pt", sha256=sha256_file(f"{run.output_dir}/latest.pt"))
    return ledger
