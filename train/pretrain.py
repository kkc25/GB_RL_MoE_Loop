from __future__ import annotations

import torch

from config import ExperimentConfig
from data.affine import sample_batch
from .common import CostLedger, MetricWriter, autocast_context, learning_rate_at, move_batch, save_checkpoint


def pretrain(model, config: ExperimentConfig, device: torch.device, ledger: CostLedger, writer: MetricWriter, controller=None) -> None:
    train = config.train
    updates = train.pretrain_updates
    if updates <= 0:
        return
    model.train()
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": train.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=train.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    lengths = tuple(range(config.data.min_depth, config.data.max_depth + 1))
    generator = torch.Generator(device=device).manual_seed(config.train.seed + 101)
    for step in range(updates):
        offset = step * train.batch_size
        batch = sample_batch("train", train.batch_size, config.data.seed, offset, lengths)
        input_ids, labels = move_batch(batch, device)
        depth = int(torch.randint(1, config.model.max_calls + 1, (1,), generator=generator, device=device).item())
        lr = learning_rate_at(step, updates, train.learning_rate, train.warmup_fraction)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, train.dtype):
            output = model(input_ids, labels, depth=depth)
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train.gradient_clip)
        optimizer.step()
        tokens = input_ids.numel()
        ledger.input_tokens += tokens
        ledger.path_forward_tokens += tokens
        ledger.backward_tokens += tokens
        ledger.optimizer_updates += 1
        ledger.update_peak(device)
        if step % train.log_interval == 0:
            writer.write(phase="pretrain", step=step, offset=offset, loss=output.loss.item(), depth=depth, lr=lr, grad_norm=float(grad_norm), input_tokens=ledger.input_tokens)
        if (step + 1) % train.validation_interval == 0 or step + 1 == updates:
            save_checkpoint(f"{train.output_dir}/pretrain_latest.pt", config, model, controller, optimizer, step + 1, offset + train.batch_size, ledger)
    model.eval()
