from __future__ import annotations

import torch
from torch.nn import functional as F

from eval.counterfactual import cache_last_ffn_input, last_adaptive_layer
from train.value_student import predict


class DynamicHaltingPolicy:
    def __init__(self, students: dict, tau: float) -> None:
        self.students = students
        self.tau = tau

    def to(self, device: torch.device) -> None:
        for entry in self.students.values():
            entry["student"].to(device)
            entry["mean"] = entry["mean"].to(device)
            entry["std"] = entry["std"].to(device)

    @torch.no_grad()
    def forward(self, model, input_ids: torch.Tensor, segment_ids: torch.Tensor | None = None, trace: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self.to(input_ids.device)
        layer = last_adaptive_layer(model)
        block = model.blocks[layer]
        h = cache_last_ffn_input(model, input_ids, segment_ids, depth=1)
        normalized = block.ffn_norm(h)
        flat_normalized = normalized.reshape(-1, normalized.shape[-1])
        actions = block.executor.baseline_actions(flat_normalized)
        flat_h = h.reshape(-1, h.shape[-1])
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            stop_cost = predict(self.students["immediate"]["student"], flat_h, self.students["immediate"]["mean"], self.students["immediate"]["std"])
            continue_cost = predict(self.students["q_pi0"]["student"], flat_h, self.students["q_pi0"]["mean"], self.students["q_pi0"]["std"])
        rows = torch.arange(actions.shape[0], device=actions.device)
        delta = stop_cost[rows, actions] - continue_cost[rows, actions]
        continue_mask = delta > self.tau
        update, _, _, _ = block.executor(normalized)
        current = h + model.gammas[layer] * update
        current_flat = current.reshape(-1, current.shape[-1])
        rounds = torch.ones(actions.shape, dtype=torch.long, device=actions.device)
        for _ in range(1, model.config.max_calls):
            if not bool(continue_mask.any()):
                break
            selected = block.ffn_norm(current_flat[continue_mask])
            update, _, _, _ = block.executor(selected)
            current_flat = current_flat.clone()
            current_flat[continue_mask] = current_flat[continue_mask] + model.gammas[layer] * update
            rounds[continue_mask] += 1
        current = current_flat.reshape(h.shape)
        logits = F.linear(model.final_norm(current), model.embedding.weight)
        rounds = rounds.reshape(input_ids.shape)
        if trace is not None:
            trace.update({
                "h": h,
                "baseline_action": actions.reshape(input_ids.shape),
                "delta": delta.reshape(input_ids.shape),
                "continue_mask": continue_mask.reshape(input_ids.shape),
                "rounds": rounds,
                "hidden": current,
            })
        return logits, rounds
