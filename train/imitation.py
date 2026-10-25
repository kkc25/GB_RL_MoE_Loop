from __future__ import annotations

import torch

from config import ExperimentConfig
from data.affine import sample_batch
from eval.oracle import all_paths
from policy.rollout import replay_logprob
from train.common import CostLedger, MetricWriter, autocast_context, learning_rate_at, move_batch
from train.router_rloo import evaluate


def train_imitation(model, controller, config: ExperimentConfig, device: torch.device, ledger: CostLedger, writer: MetricWriter) -> None:
    train = config.train
    model.eval()
    model.requires_grad_(False)
    controller.train()
    controller.requires_grad_(True)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=train.controller_learning_rate, weight_decay=0.0)
    lengths = tuple(range(config.data.min_depth, config.data.max_depth + 1))
    paths = all_paths(config.model.experts_per_layer, config.model.max_calls)
    for step in range(train.updates):
        batch = sample_batch("train", train.batch_size, config.data.seed, step * train.batch_size, lengths)
        input_ids, labels = move_batch(batch, device)
        rollouts, objectives = [], []
        with torch.no_grad(), autocast_context(device, train.dtype):
            for path in paths:
                output = model(input_ids, labels, controller=controller, budget=train.budget, forced_path=path, collect_rollout=True, keep_logits=False)
                rollouts.append(output.rollout)
                objectives.append(output.rollout.token_losses.float() + train.lambda_cost * output.rollout.costs.float())
        objectives = torch.stack(objectives)
        best_values, best_paths = objectives.min(dim=0)
        optimizer.zero_grad(set_to_none=True)
        imitation_loss = 0.0
        for path_index, rollout in enumerate(rollouts):
            selected = best_paths == path_index
            if not bool(selected.any()):
                continue
            with autocast_context(device, train.dtype):
                log_probability = replay_logprob(controller, rollout)
                if not config.model.target_only:
                    log_probability = log_probability.reshape(input_ids.shape[0], input_ids.shape[1]).sum(dim=-1)
            loss = -(log_probability * selected.float()).sum() / input_ids.shape[0]
            loss.backward()
            imitation_loss += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), train.gradient_clip)
        lr = learning_rate_at(step, train.updates, train.controller_learning_rate, train.warmup_fraction)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        tokens = input_ids.numel()
        ledger.input_tokens += tokens
        ledger.path_forward_tokens += tokens * len(paths)
        ledger.oracle_forward_tokens += tokens * len(paths)
        ledger.optimizer_updates += 1
        ledger.update_peak(device)
        writer.write(phase="imitation", step=step, loss=imitation_loss, oracle_objective=best_values.mean().item(), grad_norm=float(grad_norm), lr=lr)
        if (step + 1) % train.validation_interval == 0 or step + 1 == train.updates:
            writer.write(phase="validation", step=step + 1, algorithm="imitation_greedy", **evaluate(model, controller, config, device))
    model.requires_grad_(True)
