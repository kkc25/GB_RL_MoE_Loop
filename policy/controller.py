from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from config import ControllerConfig, ModelConfig


@dataclass
class ControllerOutput:
    memory: torch.Tensor
    stop_logit: torch.Tensor
    expert_log_probs: torch.Tensor
    reference: torch.Tensor


class Controller(nn.Module):
    def __init__(self, model: ModelConfig, config: ControllerConfig) -> None:
        super().__init__()
        self.model_config = model
        self.config = config
        self.experts = model.experts_per_layer
        self.adaptive_layers = len(model.adaptive_layers)
        self.feature_size = config.projected_hidden + self.experts + self.adaptive_layers + 8
        self.projection = nn.Linear(model.hidden_size, config.projected_hidden, bias=True)
        self.gru = nn.GRUCell(self.feature_size, config.gru_hidden_size)
        self.actor = nn.Linear(config.gru_hidden_size, self.experts + 1, bias=True)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        self.actor.bias.data[0] = math.log(config.initial_stop_hazard / (1.0 - config.initial_stop_hazard))

    def initial_memory(self, episodes: int, device: torch.device) -> torch.Tensor:
        return self.gru.weight_hh.new_zeros(episodes, self.config.gru_hidden_size, device=device)

    def observe(
        self,
        normalized: torch.Tensor,
        previous_expert: torch.Tensor,
        adaptive_layer: torch.Tensor,
        rounds: torch.Tensor,
        remaining: torch.Tensor,
        requested: torch.Tensor,
        price: float,
        previous_update: torch.Tensor | None,
        previous_state: torch.Tensor | None,
        remaining_layers: torch.Tensor,
    ) -> torch.Tensor:
        episodes = normalized.shape[0]
        layer_onehot = F.one_hot(adaptive_layer, self.adaptive_layers).to(normalized.dtype)
        if previous_update is None:
            relative_update = torch.zeros(episodes, device=normalized.device)
            cosine = torch.ones(episodes, device=normalized.device)
        else:
            relative_update = previous_update.float().norm(dim=-1) / normalized.float().norm(dim=-1).clamp_min(1e-8)
            if previous_state is None:
                cosine = torch.ones(episodes, device=normalized.device)
            else:
                cosine = F.cosine_similarity(previous_state.float(), normalized.float(), dim=-1)
        log_rms = normalized.float().square().mean(dim=-1).add(1e-8).log().mul(0.5)
        maximum = float(self.adaptive_layers * self.model_config.max_calls)
        scalars = torch.stack(
            (
                rounds.float() / self.model_config.max_calls,
                remaining.float() / maximum,
                requested.float() / maximum,
                torch.full_like(rounds, float(price), dtype=torch.float32),
                relative_update,
                cosine,
                log_rms,
                remaining_layers.float() / self.adaptive_layers,
            ),
            dim=-1,
        ).to(normalized.dtype)
        return torch.cat((normalized, previous_expert.to(normalized.dtype), layer_onehot, scalars), dim=-1)

    def forward(
        self,
        observation: torch.Tensor,
        memory: torch.Tensor,
        gate_logits: torch.Tensor,
        gate_probabilities: torch.Tensor,
    ) -> ControllerOutput:
        hidden_size = self.model_config.hidden_size
        projected = self.projection(observation[:, :hidden_size])
        features = torch.cat((projected, observation[:, hidden_size:]), dim=-1)
        new_memory = self.gru(features, memory)
        logits = self.actor(new_memory)
        gate_logits = gate_logits.float()
        tempered = F.softmax(gate_logits / self.config.reference_temperature, dim=-1)
        original = gate_probabilities.float()
        reference = self.config.reference_tempered_weight * tempered + (1.0 - self.config.reference_tempered_weight) * original
        reference = reference.clamp_min(torch.finfo(torch.float32).tiny)
        reference = reference / reference.sum(dim=-1, keepdim=True)
        expert_logits = reference.log() + logits[:, 1:].float()
        return ControllerOutput(new_memory, logits[:, 0].float(), F.log_softmax(expert_logits, dim=-1), reference)

    def decision_log_prob(self, output: ControllerOutput, stopped: torch.Tensor, expert: torch.Tensor, forced_continue: torch.Tensor) -> torch.Tensor:
        stop_log_prob = F.logsigmoid(output.stop_logit)
        continue_log_prob = F.logsigmoid(-output.stop_logit)
        selected = output.expert_log_probs.gather(-1, expert.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        normal = torch.where(stopped, stop_log_prob, continue_log_prob + selected)
        return torch.where(forced_continue, selected, normal)
