from __future__ import annotations

import torch

from config import ExperimentConfig
from data.affine import sample_batch
from eval.oracle import all_paths, exact_path_batch
from train.common import CostLedger, MetricWriter, learning_rate_at, move_batch
from train.router_rloo import evaluate


def train_exact(model, controller, config: ExperimentConfig, device: torch.device, ledger: CostLedger, writer: MetricWriter) -> None:
    train = config.train
    model.eval()
    model.requires_grad_(False)
    controller.train()
    controller.requires_grad_(True)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=train.controller_learning_rate, weight_decay=0.0)
    lengths = tuple(range(config.data.min_depth, config.data.max_depth + 1))
    paths = all_paths(config.model.experts_per_layer, config.model.max_calls)
    for step in range(train.updates):
        offset = step * train.batch_size
        batch = sample_batch("train", train.batch_size, config.data.seed, offset, lengths)
        input_ids, labels = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        expected = 0.0
        probability_mass = 0.0
        best = None
        reference = None
        for path, objective, log_probability in exact_path_batch(model, controller, input_ids, labels, train.budget, train.lambda_cost, device, train.dtype):
            ((objective.detach() * log_probability).mean() / len(paths)).backward()
            probability = log_probability.detach().exp()
            expected += (probability * objective).mean().item()
            probability_mass += probability.mean().item()
            candidate = objective.detach()
            reference = candidate if reference is None else reference
            best = candidate if best is None else torch.minimum(best, candidate)
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
        writer.write(phase="exact", step=step, expected_objective=expected, probability_mass=probability_mass, oracle_gap=(reference - best).mean().item(), grad_norm=float(grad_norm), lr=lr)
        if (step + 1) % train.validation_interval == 0 or step + 1 == train.updates:
            metrics = evaluate(model, controller, config, device)
            writer.write(phase="validation", step=step + 1, algorithm="exact_greedy", **metrics)
    model.requires_grad_(True)
