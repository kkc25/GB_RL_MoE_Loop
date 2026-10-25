from __future__ import annotations

import torch
from torch import nn

from config import ExperimentConfig
from data.affine import sample_batch
from train.common import CostLedger, MetricWriter, autocast_context, learning_rate_at, move_batch


class HaltHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(hidden)).squeeze(-1).float()


def halt_distribution(model, head, input_ids, labels, device, dtype):
    losses, hazards = [], []
    positions = labels != -100
    with torch.no_grad(), autocast_context(device, dtype):
        for depth in range(1, model.config.max_calls + 1):
            output = model(input_ids, labels, depth=depth)
            losses.append(output.token_losses.float())
            hazards.append(output.hidden[positions].detach())
    logits = [head(hidden) for hidden in hazards]
    continuation = []
    remaining = torch.ones_like(logits[0])
    probabilities = []
    for index, logit in enumerate(logits):
        if index + 1 == len(logits):
            probabilities.append(remaining)
        else:
            stop = torch.sigmoid(logit)
            probabilities.append(remaining * stop)
            continuation.append(remaining * (1.0 - stop))
            remaining = continuation[-1]
    return torch.stack(losses), torch.stack(probabilities), torch.stack(logits)


@torch.no_grad()
def evaluate_halt(model, head, config, device, split="validation"):
    losses, expectations, accuracies = [], [], []
    lengths = tuple(range(config.data.min_depth, config.data.max_depth + 1))
    for batch_index in range(config.train.validation_batches):
        batch = sample_batch(split, config.train.batch_size, config.data.seed, batch_index * config.train.batch_size, lengths)
        input_ids, labels = move_batch(batch, device)
        depth_losses, probabilities, _ = halt_distribution(model, head, input_ids, labels, device, config.train.dtype)
        losses.append(depth_losses)
        expectations.append((probabilities * depth_losses).sum(dim=0))
        depths = torch.arange(1, config.model.max_calls + 1, device=device).float()
        accuracies.append((probabilities * depths[:, None]).sum(dim=0))
    return {"loss": torch.cat(expectations).mean().item(), "expected_cost": torch.cat(accuracies).mean().item()}


def train_halt(model, config: ExperimentConfig, device: torch.device, ledger: CostLedger, writer: MetricWriter) -> HaltHead:
    train = config.train
    model.eval()
    model.requires_grad_(False)
    head = HaltHead(config.model.hidden_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=train.controller_learning_rate, weight_decay=0.0)
    lengths = tuple(range(config.data.min_depth, config.data.max_depth + 1))
    depths = torch.arange(1, config.model.max_calls + 1, device=device).float()
    for step in range(train.updates):
        batch = sample_batch("train", train.batch_size, config.data.seed, step * train.batch_size, lengths)
        input_ids, labels = move_batch(batch, device)
        losses, probabilities, _ = halt_distribution(model, head, input_ids, labels, device, train.dtype)
        objective = (probabilities * (losses.detach() + train.lambda_cost * depths[:, None])).sum(dim=0).mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), train.gradient_clip)
        lr = learning_rate_at(step, train.updates, train.controller_learning_rate, train.warmup_fraction)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        tokens = input_ids.numel()
        ledger.input_tokens += tokens
        ledger.path_forward_tokens += tokens * config.model.max_calls
        ledger.optimizer_updates += 1
        ledger.update_peak(device)
        writer.write(phase="halt", step=step, objective=objective.item(), expected_cost=(probabilities * depths[:, None]).sum(dim=0).mean().item(), grad_norm=float(grad_norm))
        if (step + 1) % train.validation_interval == 0 or step + 1 == train.updates:
            writer.write(phase="validation", step=step + 1, algorithm="exact_halt", **evaluate_halt(model, head, config, device))
            torch.save({"head": head.state_dict(), "step": step + 1}, f"{train.output_dir}/halt_latest.pt")
    model.requires_grad_(True)
    return head
