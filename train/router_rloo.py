from __future__ import annotations

import torch

from config import ExperimentConfig
from data.affine import sample_batch
from .common import CostLedger, MetricWriter, autocast_context, learning_rate_at, move_batch, save_checkpoint


@torch.no_grad()
def evaluate(model, controller, config: ExperimentConfig, device: torch.device, split: str = "validation", routed: bool = True) -> dict[str, float]:
    train = config.train
    was_training = model.training
    model.eval()
    if controller is not None:
        controller.eval()
    losses, costs, difficulties, correct, count = [], [], [], 0, 0
    lengths = tuple(range(config.data.min_depth, config.data.max_depth + 1))
    for batch_index in range(train.validation_batches):
        offset = batch_index * train.batch_size
        batch = sample_batch(split, train.batch_size, config.data.seed, offset, lengths)
        input_ids, labels = move_batch(batch, device)
        with autocast_context(device, train.dtype):
            output = model(
                input_ids,
                labels,
                depth=config.model.max_calls,
                controller=controller if routed else None,
                budget=train.budget if routed else None,
                price=train.lambda_cost,
                sample=False,
                collect_rollout=routed,
            )
        losses.append(output.token_losses.detach().float())
        difficulties.extend(item["length"] for item in batch["meta"])
        if routed:
            costs.append(output.rollout.costs.detach().float())
        else:
            costs.append(torch.full_like(output.token_losses, float(config.model.max_calls)))
        positions = labels != -100
        predictions = output.logits.argmax(dim=-1)
        correct += int((predictions[positions] == labels[positions]).sum().item())
        count += int(positions.sum().item())
    if was_training:
        model.train()
    flat_losses = torch.cat(losses)
    flat_costs = torch.cat(costs)
    result = {
        "loss": flat_losses.mean().item(),
        "cost": flat_costs.mean().item(),
        "accuracy": correct / max(1, count),
        "examples": count,
    }
    difficulties = torch.tensor(difficulties, device=flat_losses.device)
    for length in lengths:
        selected = difficulties == length
        if bool(selected.any()):
            result[f"loss_length_{length}"] = flat_losses[selected].mean().item()
            result[f"cost_length_{length}"] = flat_costs[selected].mean().item()
    return result


def train_rloo(model, controller, config: ExperimentConfig, device: torch.device, ledger: CostLedger, writer: MetricWriter, start_step: int = 0, start_offset: int = 0) -> None:
    train = config.train
    model.eval()
    model.requires_grad_(False)
    controller.train()
    controller.requires_grad_(True)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=train.controller_learning_rate, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
    lengths = tuple(range(config.data.min_depth, config.data.max_depth + 1))
    best_loss = float("inf")
    batch_size = train.batch_size
    groups = train.rollout_groups
    if groups < 2:
        raise ValueError("RLOO requires at least two independent routes")
    for step in range(start_step, train.updates):
        if ledger.input_tokens >= train.input_token_limit:
            break
        offset = start_offset + (step - start_step) * batch_size
        batch = sample_batch("train", train.batch_size, config.data.seed, offset, lengths)
        input_ids, labels = move_batch(batch, device)
        rollouts = []
        with torch.no_grad(), autocast_context(device, train.dtype):
            for group in range(groups):
                generator = torch.Generator(device=device).manual_seed(config.train.seed + step * groups + group)
                output = model(
                    input_ids,
                    labels,
                    controller=controller,
                    budget=train.budget,
                    price=train.lambda_cost,
                    sample=True,
                    generator=generator,
                    collect_rollout=True,
                    keep_logits=False,
                )
                rollouts.append(output.rollout)
        sequence_losses = torch.stack([route.token_losses.float() for route in rollouts])
        costs = torch.stack([route.costs.float() for route in rollouts])
        objectives = sequence_losses + train.lambda_cost * costs
        loo = (objectives.sum(dim=0, keepdim=True) - objectives) / (groups - 1)
        advantages = (objectives - loo).detach()
        log_joints = []
        with autocast_context(device, train.dtype):
            for route in rollouts:
                from policy.rollout import replay_logprob
                log_joint = replay_logprob(controller, route)
                if not config.model.target_only:
                    log_joint = log_joint.reshape(batch_size, input_ids.shape[1]).sum(dim=-1)
                log_joints.append(log_joint)
        actor_loss = (advantages * torch.stack(log_joints)).mean()
        lr = learning_rate_at(step, train.updates, train.controller_learning_rate, train.warmup_fraction)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), train.gradient_clip)
        optimizer.step()
        tokens = input_ids.numel()
        ledger.input_tokens += tokens
        ledger.path_forward_tokens += tokens * groups
        ledger.optimizer_updates += 1
        ledger.update_peak(device)
        if step % train.log_interval == 0:
            writer.write(
                phase="rloo",
                step=step,
                offset=offset,
                actor_loss=actor_loss.item(),
                nll=sequence_losses.mean().item(),
                cost=costs.mean().item(),
                advantage_std=advantages.std().item(),
                lr=lr,
                grad_norm=float(grad_norm),
                input_tokens=ledger.input_tokens,
                path_forward_tokens=ledger.path_forward_tokens,
            )
        if (step + 1) % train.validation_interval == 0 or step + 1 == train.updates:
            metrics = evaluate(model, controller, config, device)
            writer.write(phase="validation", step=step + 1, algorithm="rloo_greedy", **metrics)
            save_checkpoint(f"{train.output_dir}/rloo_latest.pt", config, model, controller, optimizer, step + 1, offset + batch_size, ledger)
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                save_checkpoint(f"{train.output_dir}/rloo_best.pt", config, model, controller, optimizer, step + 1, offset + batch_size, ledger)
            controller.train()
    model.requires_grad_(True)
