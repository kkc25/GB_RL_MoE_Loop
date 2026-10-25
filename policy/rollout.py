from __future__ import annotations

from dataclasses import dataclass, field
import torch

from .controller import Controller


@dataclass
class DecisionBatch:
    observation: torch.Tensor
    indices: torch.Tensor
    stopped: torch.Tensor
    expert: torch.Tensor
    forced_continue: torch.Tensor
    gate_logits: torch.Tensor
    gate_probabilities: torch.Tensor


@dataclass
class RouteRollout:
    loss: torch.Tensor
    token_losses: torch.Tensor
    costs: torch.Tensor
    decisions: list[DecisionBatch] = field(default_factory=list)
    episodes: int = 0
    calls: torch.Tensor | None = None
    logits: torch.Tensor | None = None


def replay_logprob(controller: Controller, rollout: RouteRollout) -> torch.Tensor:
    memory = controller.initial_memory(rollout.episodes, rollout.decisions[0].observation.device)
    joint = torch.zeros(rollout.episodes, dtype=torch.float32, device=memory.device)
    for decision in rollout.decisions:
        output = controller(
            decision.observation,
            memory.index_select(0, decision.indices),
            decision.gate_logits,
            decision.gate_probabilities,
        )
        memory.index_copy_(0, decision.indices, output.memory.to(memory.dtype))
        log_prob = controller.decision_log_prob(output, decision.stopped, decision.expert, decision.forced_continue)
        joint.index_add_(0, decision.indices, log_prob)
    return joint
