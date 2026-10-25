from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from config import ModelConfig
from policy.controller import Controller
from policy.rollout import DecisionBatch, RouteRollout
from .moe import DenseSwiGLU, MoE, RMSNorm
from .loop import episode_random_keys, splitmix_uniform, validate_budget
from .top2 import Top2MoE


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    token_losses: torch.Tensor | None
    calls: torch.Tensor
    hidden: torch.Tensor | None = None
    rollout: RouteRollout | None = None


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.head_dim
        self.hidden = config.hidden_size
        self.rope_base = config.rope_base
        self.q = nn.Linear(self.hidden, self.hidden, bias=False)
        self.k = nn.Linear(self.hidden, self.hidden, bias=False)
        self.v = nn.Linear(self.hidden, self.hidden, bias=False)
        self.o = nn.Linear(self.hidden, self.hidden, bias=False)

    def rope(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-2]
        half = self.head_dim // 2
        frequencies = self.rope_base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
        positions = torch.arange(length, device=x.device, dtype=torch.float32)
        angles = positions[:, None] * frequencies[None, :]
        cos = angles.cos()[None, None, :, :].to(x.dtype)
        sin = angles.sin()[None, None, :, :].to(x.dtype)
        first, second = x[..., :half], x[..., half:]
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        if segment_ids is None:
            attended = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, is_causal=True)
        else:
            same_document = segment_ids.unsqueeze(1) == segment_ids.unsqueeze(2)
            causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
            valid = segment_ids >= 0
            allowed = same_document & causal.unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)
            attended = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, attn_mask=allowed.unsqueeze(1))
        return self.o(attended.transpose(1, 2).contiguous().view(batch, length, self.hidden))


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.moe = MoE(config.hidden_size, config.intermediate_size, config.experts_per_layer) if layer_index in config.moe_layers else DenseSwiGLU(config.hidden_size, config.intermediate_size)
        self.executor = Top2MoE(self.moe, freeze=False) if layer_index in config.moe_layers and config.top_k == 2 else self.moe


class LoopMoE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(Block(config, layer) for layer in range(config.layers))
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.adaptive = {layer: index for index, layer in enumerate(config.adaptive_layers)}
        self.gammas = [config.adaptive_residual_scale if layer in self.adaptive else config.nonadaptive_residual_scale for layer in range(config.layers)]
        self.apply(self.initialize)
        for layer, block in enumerate(self.blocks):
            if layer not in config.moe_layers:
                continue
            for weight in (block.moe.pool.w1, block.moe.pool.w2, block.moe.pool.w3):
                nn.init.normal_(weight, std=0.02)

    @staticmethod
    def initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def supervised_positions(self, labels: torch.Tensor | None, batch: int, length: int, device: torch.device) -> torch.Tensor:
        if labels is None or not self.config.target_only:
            return torch.full((batch,), length - 1, dtype=torch.long, device=device)
        mask = labels != -100
        if bool((mask.sum(dim=-1) != 1).any()):
            raise ValueError("target_only requires exactly one supervised position per sequence")
        return mask.long().argmax(dim=-1)

    def fixed_depth(self, h: torch.Tensor, block: Block, depth: int | torch.Tensor, positions: torch.Tensor | None, gamma: float, expert_path: tuple[int, ...] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, dim = h.shape
        base = h
        if positions is not None:
            normalized = block.ffn_norm(h)
            indices = None if expert_path is None else torch.full((batch, length), expert_path[0], dtype=torch.long, device=h.device)
            update, _, _, _ = block.executor(normalized, indices)
            base = h + gamma * update
            selected_h = base[torch.arange(batch, device=h.device), positions]
            selected_depth = depth - 1
        else:
            selected_h = h.reshape(-1, dim)
            selected_depth = depth
        if isinstance(selected_depth, int):
            depths = torch.full((selected_h.shape[0],), selected_depth, dtype=torch.long, device=h.device)
        else:
            depths = selected_depth.reshape(-1)
        updated = selected_h
        extra_calls = torch.zeros(selected_h.shape[0], dtype=torch.long, device=h.device)
        maximum = int(depths.max().item()) if depths.numel() else 0
        for round_index in range(maximum):
            active = depths > round_index
            if not bool(active.any()):
                continue
            normalized = block.ffn_norm(updated[active])
            path_index = round_index + (1 if positions is not None else 0)
            indices = None if expert_path is None else torch.full((normalized.shape[0],), expert_path[path_index], dtype=torch.long, device=h.device)
            update, _, _, _ = block.executor(normalized, indices)
            updated = updated.clone()
            updated[active] = updated[active] + gamma * update
            extra_calls += active.long()
        if positions is None:
            calls = extra_calls.reshape(batch, length)
            return updated.reshape(batch, length, dim), calls
        result = base.clone()
        result[torch.arange(batch, device=h.device), positions] = updated
        full_calls = torch.ones(batch, length, dtype=torch.long, device=h.device)
        full_calls[torch.arange(batch, device=h.device), positions] = extra_calls + 1
        return result, full_calls

    def route_layer(
        self,
        h: torch.Tensor,
        block: Block,
        adaptive_index: int,
        controller: Controller,
        positions: torch.Tensor | None,
        budget: torch.Tensor,
        memory: torch.Tensor,
        previous_expert: torch.Tensor,
        remaining: torch.Tensor,
        rounds: torch.Tensor,
        price: float,
        sample: bool,
        generator: torch.Generator | None,
        random_keys: torch.Tensor,
        random_counters: torch.Tensor,
        decisions: list[DecisionBatch],
        gamma: float,
        forced_path: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, dim = h.shape
        base = h
        if positions is None:
            current = h.reshape(-1, dim)
        else:
            normalized = block.ffn_norm(h)
            base_update, _, _, _ = block.executor(normalized)
            base = h + gamma * base_update
            current = h[torch.arange(batch, device=h.device), positions]
        episodes = current.shape[0]
        previous_expert = torch.zeros_like(previous_expert)
        rounds = torch.zeros_like(rounds)
        previous_update = None
        previous_state = None
        calls = torch.zeros(episodes, dtype=torch.long, device=h.device)
        alive = torch.ones(episodes, dtype=torch.bool, device=h.device)
        remaining_layers = torch.full_like(rounds, len(self.config.adaptive_layers) - adaptive_index - 1)
        while True:
            first = rounds == 0
            legal = alive & (first | ((rounds < self.config.max_calls) & (remaining >= remaining_layers + 1)))
            active_indices = torch.nonzero(legal, as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                break
            active_h = current.index_select(0, active_indices)
            normalized = block.ffn_norm(active_h)
            gate_logits = block.moe.gate(normalized)
            gate_probabilities = F.softmax(gate_logits.float(), dim=-1)
            observation = controller.observe(
                normalized,
                previous_expert.index_select(0, active_indices),
                torch.full((active_indices.numel(),), adaptive_index, dtype=torch.long, device=h.device),
                rounds.index_select(0, active_indices),
                remaining.index_select(0, active_indices),
                budget.index_select(0, active_indices),
                price,
                None if previous_update is None else previous_update.index_select(0, active_indices),
                None if previous_state is None else previous_state.index_select(0, active_indices),
                remaining_layers.index_select(0, active_indices),
            )
            output = controller(observation, memory.index_select(0, active_indices), gate_logits, gate_probabilities)
            memory.index_copy_(0, active_indices, output.memory.to(memory.dtype))
            forced = first.index_select(0, active_indices)
            active_rounds = rounds.index_select(0, active_indices)
            if forced_path is not None:
                stopped = active_rounds >= len(forced_path)
                path_expert = forced_path[min(int(active_rounds.max().item()), len(forced_path) - 1)]
                expert = torch.full_like(active_rounds, path_expert)
            elif sample:
                active_keys = random_keys.index_select(0, active_indices)
                active_counters = random_counters.index_select(0, active_indices)
                stop_uniform = splitmix_uniform(active_keys, adaptive_index, active_rounds, active_counters, 0).to(h.device)
                expert_uniform = splitmix_uniform(active_keys, adaptive_index, active_rounds, active_counters, 1).to(h.device)
                stopped = stop_uniform < torch.sigmoid(output.stop_logit)
                cumulative = output.expert_log_probs.exp().cumsum(dim=-1)
                expert = (expert_uniform.unsqueeze(-1) >= cumulative).sum(dim=-1).clamp_max(self.config.experts_per_layer - 1)
                random_counters[active_indices] += 1
            else:
                stopped = output.stop_logit >= 0
                expert = output.expert_log_probs.argmax(dim=-1)
            stopped = stopped & ~forced
            alive[active_indices[stopped]] = False
            decisions.append(DecisionBatch(observation, active_indices, stopped, expert, forced, gate_logits, gate_probabilities))
            continued_indices = active_indices[~stopped]
            if continued_indices.numel() == 0:
                break
            continued_expert = expert[~stopped]
            continued_normalized = normalized[~stopped]
            update, _, _, _ = block.executor(continued_normalized, continued_expert)
            current = current.clone()
            current[continued_indices] = current[continued_indices] + gamma * update
            calls[continued_indices] += 1
            remaining[continued_indices] -= 1
            rounds[continued_indices] += 1
            previous_expert[continued_indices] = F.one_hot(continued_expert, self.config.experts_per_layer).to(previous_expert.dtype)
            if previous_update is None:
                previous_update = torch.zeros_like(current)
                previous_state = torch.zeros_like(current)
            previous_update.index_copy_(0, continued_indices, update)
            previous_state.index_copy_(0, continued_indices, continued_normalized)
        if positions is None:
            return current.reshape(batch, length, dim), calls, memory, previous_expert, remaining, rounds
        result = base.clone()
        result[torch.arange(batch, device=h.device), positions] = current
        return result, calls, memory, previous_expert, remaining, rounds

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        depth: int | torch.Tensor = 1,
        controller: Controller | None = None,
        budget: int | torch.Tensor | None = None,
        price: float = 0.0,
        sample: bool = False,
        generator: torch.Generator | None = None,
        collect_rollout: bool = False,
        keep_logits: bool = True,
        expert_path: tuple[int, ...] | None = None,
        forced_path: tuple[int, ...] | None = None,
        episode_ids: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> ModelOutput:
        batch, length = input_ids.shape
        positions = self.supervised_positions(labels, batch, length, input_ids.device) if self.config.target_only else None
        h = self.embedding(input_ids)
        total_calls = torch.zeros(batch, length, dtype=torch.long, device=input_ids.device)
        decisions: list[DecisionBatch] = []
        if controller is not None:
            if self.config.top_k != 1:
                raise NotImplementedError("Top-2 controller training uses the separate 28-action policy")
            episodes = batch if positions is not None else batch * length
            memory = controller.initial_memory(episodes, input_ids.device)
            previous_expert = torch.zeros(episodes, self.config.experts_per_layer, device=input_ids.device)
            if isinstance(budget, int):
                requested = torch.full((episodes,), budget, dtype=torch.long, device=input_ids.device)
            elif budget is None:
                raise ValueError("budget is required for routed forward")
            else:
                requested = budget.reshape(-1).to(input_ids.device)
            validate_budget(requested, len(self.config.adaptive_layers), self.config.max_calls)
            remaining = requested.clone()
            rounds = torch.zeros_like(requested)
            route_seed = generator.initial_seed() if generator is not None else 0
            random_keys = episode_random_keys(input_ids, requested, route_seed, episode_ids, positions)
            random_counters = torch.zeros_like(requested)
        for layer_index, block in enumerate(self.blocks):
            h = h + block.attention(block.attention_norm(h), segment_ids)
            if layer_index not in self.adaptive:
                normalized = block.ffn_norm(h)
                update, _, _, _ = block.executor(normalized)
                h = h + self.gammas[layer_index] * update
                total_calls += 1
                continue
            if controller is None:
                h, calls = self.fixed_depth(h, block, depth, positions if layer_index == self.config.adaptive_layers[-1] else None, self.gammas[layer_index], expert_path)
                total_calls += calls
            else:
                routed_positions = positions if layer_index == self.config.adaptive_layers[-1] else None
                h, calls, memory, previous_expert, remaining, rounds = self.route_layer(
                    h, block, self.adaptive[layer_index], controller, routed_positions, requested,
                    memory, previous_expert, remaining, rounds, price, sample, generator, random_keys, random_counters, decisions,
                    self.gammas[layer_index], forced_path,
                )
                if routed_positions is None:
                    total_calls += calls.reshape(batch, length)
                else:
                    total_calls += 1
                    total_calls[torch.arange(batch, device=h.device), routed_positions] += calls - 1
        logits = F.linear(self.final_norm(h), self.embedding.weight)
        token_losses = None
        loss = None
        if labels is not None:
            losses = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), labels.reshape(-1), ignore_index=-100, reduction="none").reshape(batch, length)
            token_losses = losses.sum(dim=-1) / (labels != -100).sum(dim=-1).clamp_min(1)
            loss = token_losses.mean()
        rollout = None
        if collect_rollout:
            if controller is None or loss is None:
                raise ValueError("rollout collection requires a controller and labels")
            if positions is None:
                costs = total_calls.float() - float(self.config.layers - len(self.config.adaptive_layers))
                costs = costs.mean(dim=-1)
            else:
                costs = total_calls[torch.arange(batch, device=h.device), positions].float() - float(self.config.layers - len(self.config.adaptive_layers))
            rollout = RouteRollout(loss, token_losses, costs, decisions, batch if positions is not None else batch * length, total_calls, logits if keep_logits else None)
        return ModelOutput(logits, loss, token_losses, total_calls, h, rollout)
