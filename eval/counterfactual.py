"""E01 counterfactual Top-2 path diagnostics on a frozen LoopMoE backbone.

The backbone blocks are replayed manually to cache the hidden state ``h``
entering the final adaptive (looping) FFN, exactly as ``LoopMoE.forward``
computes it.  Because the adaptive layer is the last block, no attention
follows it, so every supervised position can be continued independently:
force each of the 28 Top-2 actions at round 1, then continue with the frozen
baseline policy pi0 (router Top-2, frozen-logit renormalized weights) up to
``rounds`` total calls, reading out FP32 cross-entropy against the true
next-token label after round 1 (immediate, T=1) and after the final round
(Q^{pi0}, T=rounds).  Forcing the baseline action reproduces the model's
normal fixed-depth forward exactly, which the tests verify.

Candidates are processed in chunks and the FP32 CE is accumulated on the fly,
so the full ``states x candidates x vocab`` logits tensor is never
materialized.  States are only taken at positions with ``labels != -100``
and are gathered per packed-LM split file, so train/validation/test states
are isolated by file and by document.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from data.packed_lm import PackedLMBatchSampler
from model.top2 import Top2MoE, baseline_top2_pairs, enumerate_top2_actions, pair_to_action
from train.common import autocast_context


@dataclass
class CounterfactualLedger:
    backbone_prefix_tokens: int = 0
    candidate_pair_calls: int = 0
    head_readouts: int = 0
    student_train_samples: int = 0
    student_optimizer_updates: int = 0
    elapsed_seconds: float = 0.0
    peak_memory_bytes: int = 0

    def merge_peak(self, device: torch.device) -> None:
        if device.type == "cuda":
            self.peak_memory_bytes = max(self.peak_memory_bytes, torch.cuda.max_memory_allocated(device))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


@dataclass
class PathCosts:
    """Real measured costs of every candidate path, per state.

    ``immediate`` and ``q_pi0`` are (n_states, n_actions) FP32 CE values.
    ``q_pi0[:, a]`` is the T=``rounds`` cost of forcing action ``a`` at
    round 1 and following the baseline policy afterwards; the candidate set
    always contains the baseline action, so ``q_pi0`` gathered at
    ``baseline_action`` is the ordinary depth-``rounds`` baseline cost.
    """

    immediate: torch.Tensor
    q_pi0: torch.Tensor
    baseline_action: torch.Tensor
    actions: torch.Tensor
    rounds: int
    global_action_ids: torch.Tensor | None = None


def last_adaptive_layer(model) -> int:
    layer = int(model.config.adaptive_layers[-1])
    block = model.blocks[layer]
    if not isinstance(block.executor, Top2MoE):
        raise TypeError(f"last adaptive layer {layer} does not use the Top-2 executor")
    return layer


@torch.no_grad()
def cache_last_ffn_input(model, input_ids: torch.Tensor, segment_ids: torch.Tensor | None = None, depth: int = 1) -> torch.Tensor:
    """Replay the backbone up to the last adaptive layer's FFN.

    Mirrors ``LoopMoE.forward`` block by block (attention with the same
    block-diagonal ``segment_ids`` mask, dense/baseline FFN updates with the
    same residual scales) and returns ``h`` entering the final adaptive FFN,
    before ``ffn_norm``.  No attention is applied to ``h`` afterwards.
    ``depth`` only affects earlier adaptive layers, if any.
    """
    last = last_adaptive_layer(model)
    h = model.embedding(input_ids)
    for layer_index, block in enumerate(model.blocks):
        h = h + block.attention(block.attention_norm(h), segment_ids)
        if layer_index == last:
            return h
        if layer_index in model.adaptive:
            h, _ = model.fixed_depth(h, block, depth, None, model.gammas[layer_index])
        else:
            normalized = block.ffn_norm(h)
            update, _, _, _ = block.executor(normalized)
            h = h + model.gammas[layer_index] * update
    raise RuntimeError("last adaptive layer not reached")


@torch.no_grad()
def extract_states(model, input_ids, labels, segment_ids=None, doc_ids=None, depth: int = 1, dtype: str = "float32") -> dict:
    """Cache h and keep only supervised positions (``labels != -100``)."""
    device = input_ids.device
    with autocast_context(device, dtype):
        h = cache_last_ffn_input(model, input_ids, segment_ids, depth)
    mask = labels != -100
    rows, positions = torch.nonzero(mask, as_tuple=True)
    if doc_ids is None:
        doc_ids = torch.full_like(labels, -1)
    return {
        "h": h[rows, positions].float(),
        "targets": labels[rows, positions],
        "positions": positions,
        "rows": rows,
        "doc_ids": doc_ids[rows, positions],
    }


@torch.no_grad()
def gather_states(model, dataset, n_states: int, *, split: str, device: torch.device, dtype: str = "float32", seed: int = 0, seq_batch: int = 2, depth: int = 1, ledger: CounterfactualLedger | None = None) -> dict:
    """Gather up to ``n_states`` supervised states from one packed split file.

    Sequences are visited in a deterministic seeded order; every returned
    state carries its split file, sequence index, document id, position and
    target, so states from different split files never mix.
    """
    collected = []
    total = 0
    sampler = PackedLMBatchSampler(dataset, seq_batch, shuffle=True, seed=seed, epoch=0, drop_last=False)
    for indices in sampler:
        items = [dataset[index] for index in indices]
        input_ids = torch.from_numpy(__import__("numpy").stack([item["input_ids"] for item in items])).to(device)
        labels = torch.from_numpy(__import__("numpy").stack([item["labels"] for item in items])).to(device)
        segment_ids = torch.from_numpy(__import__("numpy").stack([item["segment_ids"] for item in items])).to(device)
        doc_ids = torch.from_numpy(__import__("numpy").stack([item["doc_ids"] for item in items])).to(device)
        batch = extract_states(model, input_ids, labels, segment_ids, doc_ids, depth, dtype)
        batch["seq_indices"] = torch.tensor(indices, dtype=torch.long, device=device)[batch["rows"]]
        collected.append(batch)
        total += batch["h"].shape[0]
        if ledger is not None:
            ledger.backbone_prefix_tokens += int(input_ids.numel())
        if total >= n_states:
            break
    if total < n_states:
        raise ValueError(f"split {split!r} only yielded {total} supervised states, requested {n_states}")
    merged = {key: torch.cat([batch[key] for batch in collected]) for key in ("h", "targets", "positions", "doc_ids", "seq_indices")}
    merged = {key: value[:n_states] for key, value in merged.items()}
    merged["split"] = split
    return merged


@torch.no_grad()
def readout_ce(model, h: torch.Tensor, targets: torch.Tensor, chunk: int = 256, ledger: CounterfactualLedger | None = None) -> torch.Tensor:
    """FP32 cross-entropy through final_norm + tied head, accumulated per chunk."""
    weight = model.embedding.weight
    pieces = []
    for start in range(0, h.shape[0], chunk):
        normed = model.final_norm(h[start : start + chunk])
        logits = F.linear(normed.float(), weight.float())
        pieces.append(F.cross_entropy(logits, targets[start : start + chunk], reduction="none"))
    if ledger is not None:
        ledger.head_readouts += int(h.shape[0])
    return torch.cat(pieces) if pieces else torch.empty(0)


@torch.no_grad()
def counterfactual_path_costs(
    model,
    h: torch.Tensor,
    targets: torch.Tensor,
    *,
    rounds: int = 4,
    actions: torch.Tensor | None = None,
    state_chunk: int = 32,
    readout_chunk: int = 256,
    dtype: str = "float32",
    ledger: CounterfactualLedger | None = None,
) -> PathCosts:
    """Measure immediate (T=1) and Q^{pi0} (T=``rounds``) costs of every action.

    Round 1 forces each candidate Top-2 action; rounds 2..``rounds`` follow
    the frozen baseline policy.  All candidate paths use the same number of
    expert calls, so costs are equal-compute by construction.  Candidates
    are processed ``state_chunk`` states at a time and the head readout is
    accumulated per ``readout_chunk`` rows; full candidate x vocab logits
    are never stored.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    layer = last_adaptive_layer(model)
    block = model.blocks[layer]
    gamma = model.gammas[layer]
    device = h.device
    experts = model.config.experts_per_layer
    if actions is None:
        actions = enumerate_top2_actions(experts, device=device)
    else:
        actions = actions.to(device)
    if actions.ndim not in (2, 3) or actions.shape[-1] != 2:
        raise ValueError("actions must have shape (n_actions, 2) or (n_states, n_actions, 2)")
    n_actions = actions.shape[-2]
    n_states = h.shape[0]
    global_action_ids = pair_to_action(actions, experts=experts)
    if actions.ndim == 2:
        local_action_ids = torch.full((experts * (experts - 1) // 2,), -1, dtype=torch.long, device=device)
        local_action_ids[global_action_ids] = torch.arange(n_actions, device=device)
    else:
        if actions.shape[0] != n_states:
            raise ValueError("per-state actions must share the state dimension")
        local_action_ids = torch.full((n_states, experts * (experts - 1) // 2), -1, dtype=torch.long, device=device)
        local_action_ids[torch.arange(n_states, device=device).unsqueeze(1), global_action_ids] = torch.arange(n_actions, device=device)
    immediate = torch.empty(n_states, n_actions, dtype=torch.float32, device=device)
    q_pi0 = torch.empty(n_states, n_actions, dtype=torch.float32, device=device)
    baseline_action = torch.empty(n_states, dtype=torch.long, device=device)
    for start in range(0, n_states, state_chunk):
        hc = h[start : start + state_chunk]
        tc = targets[start : start + state_chunk]
        size = hc.shape[0]
        with autocast_context(device, dtype):
            normalized = block.ffn_norm(hc)
            logits = block.executor.gate_logits(normalized)
            baseline_global = pair_to_action(baseline_top2_pairs(logits.float()), experts=experts)
            if actions.ndim == 2:
                baseline_local = local_action_ids[baseline_global]
            else:
                baseline_local = local_action_ids[torch.arange(start, start + size, device=device), baseline_global]
            if bool((baseline_local < 0).any()):
                raise ValueError("candidate actions must contain every state's baseline action")
            baseline_action[start : start + size] = baseline_local
            expanded = hc.unsqueeze(1).expand(size, n_actions, hc.shape[-1]).reshape(-1, hc.shape[-1])
            current = expanded
            for round_index in range(rounds):
                normalized = block.ffn_norm(current)
                if round_index == 0:
                    chunk_actions = actions if actions.ndim == 2 else actions[start : start + size]
                    expanded_actions = chunk_actions.unsqueeze(0).expand(size, -1, -1).reshape(-1, 2) if actions.ndim == 2 else chunk_actions.reshape(-1, 2)
                    update, _, _, _ = block.executor(normalized, pairs=expanded_actions)
                else:
                    update, _, _, _ = block.executor(normalized)
                current = current + gamma * update
                if round_index == 0:
                    after_first = current
            if ledger is not None:
                ledger.candidate_pair_calls += size * n_actions * rounds
        repeated_targets = tc.unsqueeze(1).expand(size, n_actions).reshape(-1)
        immediate[start : start + size] = readout_ce(model, after_first, repeated_targets, readout_chunk, ledger).reshape(size, n_actions)
        q_pi0[start : start + size] = readout_ce(model, current, repeated_targets, readout_chunk, ledger).reshape(size, n_actions)
    return PathCosts(immediate, q_pi0, baseline_action, actions, rounds, global_action_ids)


def candidate_actions(model, h: torch.Tensor, candidates: int = 4) -> torch.Tensor:
    if candidates < 1:
        raise ValueError("candidates must be positive")
    layer = last_adaptive_layer(model)
    block = model.blocks[layer]
    normalized = block.ffn_norm(h)
    logits = block.executor.gate_logits(normalized).float()
    baseline = baseline_top2_pairs(logits)
    all_actions = enumerate_top2_actions(model.config.experts_per_layer, device=h.device)
    scores = logits[:, all_actions].sum(dim=-1)
    order = torch.argsort(scores, dim=-1, descending=True)
    output = []
    for row, baseline_pair in enumerate(baseline):
        selected = [baseline_pair]
        for action_index in order[row].tolist():
            candidate = all_actions[action_index]
            if not any(torch.equal(candidate, previous) for previous in selected):
                selected.append(candidate)
            if len(selected) == candidates - 1:
                break
        for action_index in reversed(order[row].tolist()):
            candidate = all_actions[action_index]
            if not any(torch.equal(candidate, previous) for previous in selected):
                selected.append(candidate)
                break
        if len(selected) < candidates:
            raise ValueError("not enough distinct legal candidates")
        output.append(torch.stack(selected))
    return torch.stack(output)


def random_selection(n_states: int, n_actions: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(n_actions, (n_states,), generator=generator).to(device)


def evaluate_policies(costs: PathCosts, selections: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Held-out real executed cost of deployable label-free policies.

    Every candidate path was really executed when ``costs`` was measured, so
    the executed cost of a selected action is the measured cost of that path.
    ``deploy_improvement`` is the mean per-state cost reduction against the
    baseline policy.  Oracle selections must not be passed here; use
    ``oracle_metrics`` instead.
    """
    rows = torch.arange(costs.q_pi0.shape[0], device=costs.q_pi0.device)
    baseline = costs.q_pi0[rows, costs.baseline_action]
    report = {}
    for name, selection in selections.items():
        if selection.shape != costs.baseline_action.shape:
            raise ValueError(f"selection {name!r} has shape {tuple(selection.shape)}")
        selection = selection.to(device=costs.q_pi0.device, dtype=torch.long)
        if bool(((selection < 0) | (selection >= costs.q_pi0.shape[1])).any()):
            raise ValueError(f"selection {name!r} contains an out-of-range candidate index")
        executed = costs.q_pi0[rows, selection]
        immediate = costs.immediate[rows, selection]
        improvement = baseline - executed
        count = max(1, improvement.numel())
        report[name] = {
            "mean_cost": executed.mean().item(),
            "mean_immediate_cost": immediate.mean().item(),
            "deploy_improvement": improvement.mean().item(),
            "deploy_improvement_se": (improvement.std(unbiased=False) / count**0.5).item(),
        }
    return report


def oracle_metrics(costs: PathCosts) -> dict:
    """Label-informed oracle gap; reported separately, never a deploy gain."""
    rows = torch.arange(costs.q_pi0.shape[0], device=costs.q_pi0.device)
    baseline = costs.q_pi0[rows, costs.baseline_action]
    best, selection = costs.q_pi0.min(dim=-1)
    gap = baseline - best
    count = max(1, gap.numel())
    return {
        "mean_cost": best.mean().item(),
        "oracle_gap": gap.mean().item(),
        "oracle_gap_se": (gap.std(unbiased=False) / count**0.5).item(),
        "mean_oracle_action_is_baseline": (selection == costs.baseline_action).float().mean().item(),
        "deployable": False,
    }
