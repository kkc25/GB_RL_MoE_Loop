"""Structural causality, budget and replay checks for the routed Loop-MoE.

Plan reference: `eval/causality.py` — prefix、batch、KV、预算和 replay 检查.

All checks run on CPU with float64 models so structural identities can use
strict tolerances. There is no KV cache in the current implementation, so the
checks cover the no-cache reference path. Every check function returns a plain
dict; `run()` aggregates them into a JSON-able report and `main()` prints it.

Notes on earlier core-file issues (all resolved in production code):

* ``Controller.initial_memory`` now allocates via ``new_zeros`` on the GRU
  weight and inherits the module dtype. ``patch_initial_memory_dtype`` below
  is a redundant legacy workaround retained only for API compatibility;
  ``build_pair(patch_memory_dtype=False)`` exercises the same FP64 math.
* ``route_layer`` resets ``rounds`` per adaptive layer and reserves budget
  for the remaining adaptive layers, so ``max_calls`` caps per-layer calls
  and budgets above it are reachable (``check_multi_adaptive`` reports
  ``budget_fully_usable``).
* ``target_only=True`` with several adaptive layers is rejected by
  ``ModelConfig`` validation; ``run()`` records that rejection as expected
  behaviour in ``multi_adaptive_target_only_probe`` instead of crashing.
* Sampled routing derives per-episode random keys from the input content, so
  draws no longer shift when batch composition changes
  (``check_sampled_batch_invariance`` still reports the measured diffs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.nn import functional as F

from config import ControllerConfig, ModelConfig
from data.affine import AffineDataset, affine_collate
from model.transformer import LoopMoE
from policy.controller import Controller
from policy.rollout import replay_logprob


def small_model_config(adaptive_layers=(3,), target_only=True, **overrides) -> ModelConfig:
    kwargs = dict(
        layers=4,
        hidden_size=64,
        attention_heads=4,
        head_dim=16,
        vocab_size=512,
        intermediate_size=128,
        experts_per_layer=4,
        adaptive_layers=adaptive_layers,
        target_only=target_only,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def patch_initial_memory_dtype(controller: Controller) -> Controller:
    """Instance-level workaround: initial_memory ignores the module dtype."""

    def initial_memory(self, episodes: int, device: torch.device) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        return torch.zeros(episodes, self.config.gru_hidden_size, device=device, dtype=dtype)

    controller.initial_memory = types.MethodType(initial_memory, controller)
    return controller


def build_pair(
    adaptive_layers=(3,),
    target_only=True,
    seed: int = 0,
    patch_memory_dtype: bool = True,
    **model_overrides,
) -> tuple[LoopMoE, Controller]:
    torch.manual_seed(seed)
    model_config = small_model_config(adaptive_layers, target_only, **model_overrides)
    controller_config = ControllerConfig(projected_hidden=32, gru_hidden_size=32)
    model = LoopMoE(model_config).double().eval()
    controller = Controller(model_config, controller_config).double().eval()
    if patch_memory_dtype:
        patch_initial_memory_dtype(controller)
    return model, controller


def make_items(n: int = 4, seed: int = 0):
    dataset = AffineDataset("train", num_samples=n, seed=seed)
    return [dataset[i] for i in range(n)]


def force_continue(controller: Controller, value: float = -30.0) -> float:
    """Pin the stop bias so greedy routing never stops; returns the old bias."""
    old = float(controller.actor.bias.data[0])
    controller.actor.bias.data[0] = value
    return old


def restore_stop_bias(controller: Controller, value: float) -> None:
    controller.actor.bias.data[0] = value


def target_positions(model: LoopMoE, labels: torch.Tensor) -> torch.Tensor:
    batch, length = labels.shape
    return model.supervised_positions(labels, batch, length, labels.device)


# ---------------------------------------------------------------------------
# Call counting
# ---------------------------------------------------------------------------


def check_fixed_depth_calls(model: LoopMoE, ids, labels, depths=(1, 2, 3)) -> dict:
    """Fixed-depth mode: non-target tokens get exactly one FFN per layer and
    the target token gets exactly `depth` calls at the (single) adaptive layer."""
    layers = model.config.layers
    positions = target_positions(model, labels)
    batch = ids.shape[0]
    rows = torch.arange(batch)
    report = {}
    for depth in depths:
        with torch.no_grad():
            out = model(ids, labels=labels, depth=depth)
        target_calls = out.calls[rows, positions]
        nontarget_mask = torch.ones_like(out.calls, dtype=torch.bool)
        nontarget_mask[rows, positions] = False
        nontarget_calls = out.calls[nontarget_mask]
        report[str(depth)] = {
            "target_min": int(target_calls.min()),
            "target_max": int(target_calls.max()),
            "nontarget_min": int(nontarget_calls.min()),
            "nontarget_max": int(nontarget_calls.max()),
            "expected_target": layers - 1 + depth,
            "expected_nontarget": layers,
        }
    return report


def check_fixed_depth_calls_full_sequence(model: LoopMoE, ids, depths=(1, 2, 3)) -> dict:
    """target_only=False: every token gets `depth` calls at the adaptive layer."""
    layers = model.config.layers
    report = {}
    for depth in depths:
        with torch.no_grad():
            out = model(ids, depth=depth)
        report[str(depth)] = {
            "min": int(out.calls.min()),
            "max": int(out.calls.max()),
            "expected": layers - 1 + depth,
        }
    return report


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def check_budget(model, controller, ids, labels, budgets=(1, 2, 3), seeds=(0, 1, 2)) -> dict:
    """Routed mode, single adaptive layer: per-episode adaptive calls must
    stay within [1, budget] for greedy and for several sampling seeds."""
    layers = model.config.layers
    positions = target_positions(model, labels)
    rows = torch.arange(ids.shape[0])
    report = {}
    for budget in budgets:
        variants = {"greedy": None}
        variants.update({f"sampled_seed_{s}": s for s in seeds})
        entry = {}
        for name, seed in variants.items():
            generator = None if seed is None else torch.Generator().manual_seed(seed)
            with torch.no_grad():
                out = model(
                    ids, labels=labels, controller=controller, budget=budget,
                    sample=seed is not None, generator=generator,
                )
            calls = out.calls[rows, positions] - (layers - 1)
            entry[name] = {
                "min_calls": int(calls.min()),
                "max_calls": int(calls.max()),
                "violations": int(((calls > budget) | (calls < 1)).sum()),
            }
        report[str(budget)] = entry
    return report


# ---------------------------------------------------------------------------
# Causality: prefix and batch invariance (greedy / fixed depth, deterministic)
# ---------------------------------------------------------------------------


def check_prefix_invariance(model, controller, ids, labels, cut: int, budget=3, depth=2, seed=123) -> dict:
    """Replacing the suffix after `cut` must not change logits or call counts
    at positions before `cut`."""
    generator = torch.Generator().manual_seed(seed)
    replaced = ids.clone()
    replaced[:, cut:] = torch.randint(0, model.config.vocab_size, ids[:, cut:].shape, generator=generator)
    prefix = slice(0, cut)
    with torch.no_grad():
        fixed_a = model(ids, depth=depth)
        fixed_b = model(replaced, depth=depth)
        routed_a = model(ids, labels=labels, controller=controller, budget=budget, sample=False)
        routed_b = model(replaced, labels=labels, controller=controller, budget=budget, sample=False)
    return {
        "fixed_max_logit_diff": float((fixed_a.logits[:, prefix] - fixed_b.logits[:, prefix]).abs().max()),
        "fixed_max_call_diff": int((fixed_a.calls[:, prefix] - fixed_b.calls[:, prefix]).abs().max()),
        "routed_max_logit_diff": float((routed_a.logits[:, prefix] - routed_b.logits[:, prefix]).abs().max()),
        "routed_max_call_diff": int((routed_a.calls[:, prefix] - routed_b.calls[:, prefix]).abs().max()),
    }


def check_batch_invariance(model, controller, items, budget=3, depth=2) -> dict:
    """Running a sequence alone vs inside a batch must give identical results."""
    batch = affine_collate(items)
    ids, labels = batch["input_ids"], batch["labels"]
    report = {"fixed": [], "routed": []}
    with torch.no_grad():
        big_fixed = model(ids, labels=labels, depth=depth)
        big_routed = model(ids, labels=labels, controller=controller, budget=budget, sample=False)
        for i, item in enumerate(items):
            solo_ids = item["input_ids"].unsqueeze(0)
            solo_labels = item["labels"].unsqueeze(0)
            n = solo_ids.shape[1]
            solo_fixed = model(solo_ids, labels=solo_labels, depth=depth)
            solo_routed = model(solo_ids, labels=solo_labels, controller=controller, budget=budget, sample=False)
            report["fixed"].append({
                "max_logit_diff": float((solo_fixed.logits[0, :n] - big_fixed.logits[i, :n]).abs().max()),
                "max_call_diff": int((solo_fixed.calls[0, :n] - big_fixed.calls[i, :n]).abs().max()),
            })
            report["routed"].append({
                "max_logit_diff": float((solo_routed.logits[0, :n] - big_routed.logits[i, :n]).abs().max()),
                "max_call_diff": int((solo_routed.calls[0, :n] - big_routed.calls[i, :n]).abs().max()),
            })
    return report


def check_sampled_batch_invariance(model, controller, items, budget=3, seed=5) -> dict:
    """Sampled draws derive from content-hashed per-episode keys, so batch
    composition must not change a sequence's route; any residual diff is
    reported here."""
    batch = affine_collate(items)
    ids, labels = batch["input_ids"], batch["labels"]
    with torch.no_grad():
        big = model(ids, labels=labels, controller=controller, budget=budget,
                    sample=True, generator=torch.Generator().manual_seed(seed))
        solo_ids = items[0]["input_ids"].unsqueeze(0)
        solo_labels = items[0]["labels"].unsqueeze(0)
        solo = model(solo_ids, labels=solo_labels, controller=controller, budget=budget,
                     sample=True, generator=torch.Generator().manual_seed(seed))
    n = solo_ids.shape[1]
    return {
        "max_logit_diff": float((solo.logits[0, :n] - big.logits[0, :n]).abs().max()),
        "max_call_diff": int((solo.calls[0, :n] - big.calls[0, :n]).abs().max()),
    }


def check_sampled_reproducibility(model, controller, ids, labels, budget=3, seed=5) -> dict:
    with torch.no_grad():
        a = model(ids, labels=labels, controller=controller, budget=budget,
                  sample=True, generator=torch.Generator().manual_seed(seed))
        b = model(ids, labels=labels, controller=controller, budget=budget,
                  sample=True, generator=torch.Generator().manual_seed(seed))
    return {
        "max_logit_diff": float((a.logits - b.logits).abs().max()),
        "max_call_diff": int((a.calls - b.calls).abs().max()),
    }


# ---------------------------------------------------------------------------
# Zero-delta controller vs fixed-depth greedy
# ---------------------------------------------------------------------------


def check_zero_delta_greedy(model, controller, ids, labels) -> dict:
    """The zero-initialised actor adds no delta, so greedy routing picks the
    gate argmax (like fixed-depth MoE) and stops at the first opportunity
    (initial stop hazard 0.5 -> stop logit 0 -> greedy STOP). With the stop
    bias pinned negative it must continue to the legality limit. Both routes
    must reproduce fixed-depth runs at the matching depth."""
    max_calls = model.config.max_calls
    with torch.no_grad():
        routed1 = model(ids, labels=labels, controller=controller, budget=max_calls,
                        sample=False, collect_rollout=True)
        fixed1 = model(ids, labels=labels, depth=1)
        old = force_continue(controller)
        routed_max = model(ids, labels=labels, controller=controller, budget=max_calls, sample=False)
        fixed_max = model(ids, labels=labels, depth=max_calls)
        restore_stop_bias(controller, old)
    experts_match = all(
        bool((d.expert == d.gate_logits.double().argmax(dim=-1)).all()) for d in routed1.rollout.decisions
    )
    return {
        "stop_after_forced_round_logit_diff": float((routed1.logits - fixed1.logits).abs().max()),
        "stop_after_forced_round_call_diff": int((routed1.calls - fixed1.calls).abs().max()),
        "forced_continue_logit_diff": float((routed_max.logits - fixed_max.logits).abs().max()),
        "forced_continue_call_diff": int((routed_max.calls - fixed_max.calls).abs().max()),
        "experts_match_gate_argmax": experts_match,
    }


# ---------------------------------------------------------------------------
# Decision accounting and replay
# ---------------------------------------------------------------------------


def decision_accounting(rollout) -> dict:
    """Per-episode decision structure: the first decision of every episode is
    a forced continue, at most one STOP is recorded per episode, a recorded
    STOP is terminal, and #decisions == #calls + #stops (no logprob is
    recorded when the loop ends on legality, i.e. a forced STOP)."""
    episodes = rollout.episodes
    counts = torch.zeros(episodes, dtype=torch.long)
    forced = torch.zeros(episodes, dtype=torch.long)
    stopped = torch.zeros(episodes, dtype=torch.long)
    seen = torch.zeros(episodes, dtype=torch.bool)
    stopped_before = torch.zeros(episodes, dtype=torch.bool)
    first_forced_ok = True
    stop_terminal_ok = True
    for d in rollout.decisions:
        idx = d.indices
        if bool(stopped_before[idx].any()):
            stop_terminal_ok = False
        first = ~seen[idx]
        if not bool((d.forced_continue == first).all()):
            first_forced_ok = False
        seen[idx] = True
        stopped_before[idx] |= d.stopped
        counts.index_add_(0, idx, torch.ones_like(idx))
        forced.index_add_(0, idx, d.forced_continue.long())
        stopped.index_add_(0, idx, d.stopped.long())
    return {
        "counts": counts,
        "forced": forced,
        "stopped": stopped,
        "first_decision_forced": first_forced_ok,
        "stop_is_terminal": stop_terminal_ok,
        "at_most_one_stop": bool((stopped <= 1).all()),
        "exactly_one_forced": bool((forced == 1).all()),
    }


def manual_joint_logprob(controller: Controller, rollout) -> tuple[torch.Tensor, float]:
    """Independent recomputation of the joint per-episode log probability.

    Rebuilds GRU memory from the stored observations, recomputes the
    tempered reference from the stored gate logits by the documented formula,
    and applies the stop/expert factorisation directly (forced continue ->
    expert term only; STOP -> stop term only; continue -> continue + expert).
    Returns (joint logprob in float64, max |reference diff|)."""
    device = rollout.decisions[0].observation.device
    memory = controller.initial_memory(rollout.episodes, device)
    joint = torch.zeros(rollout.episodes, dtype=torch.float64, device=device)
    hidden = controller.model_config.hidden_size
    weight = controller.config.reference_tempered_weight
    temperature = controller.config.reference_temperature
    max_ref_diff = 0.0
    for d in rollout.decisions:
        projected = controller.projection(d.observation[:, :hidden])
        features = torch.cat((projected, d.observation[:, hidden:]), dim=-1)
        new_memory = controller.gru(features, memory.index_select(0, d.indices))
        memory = memory.clone()
        memory[d.indices] = new_memory
        actor = controller.actor(new_memory).float()
        gate = d.gate_logits.float()
        reference = weight * F.softmax(gate / temperature, dim=-1) + (1.0 - weight) * d.gate_probabilities.float()
        reference = reference.clamp_min(torch.finfo(torch.float32).tiny)
        reference = reference / reference.sum(dim=-1, keepdim=True)
        expert_log_probs = F.log_softmax(reference.log() + actor[:, 1:], dim=-1)
        selected = expert_log_probs.gather(-1, d.expert.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        log_prob = torch.where(
            d.forced_continue,
            selected,
            torch.where(d.stopped, F.logsigmoid(actor[:, 0]), F.logsigmoid(-actor[:, 0]) + selected),
        )
        joint.index_add_(0, d.indices, log_prob.double())
        # Cross-check the reference distribution stored via ControllerOutput.
        output = controller(d.observation, memory.index_select(0, d.indices), d.gate_logits, d.gate_probabilities)
        max_ref_diff = max(max_ref_diff, float((output.reference.double() - reference.double()).abs().max()))
    return joint, max_ref_diff


def check_replay(model, controller, ids, labels, budget=3, seed=0) -> dict:
    """Sampled rollout with a perturbed (non-zero-delta) actor: replay_logprob
    must match both the per-decision decision_log_prob path and the fully
    manual recomputation, and the accounting invariants must hold."""
    layers = model.config.layers
    torch.manual_seed(seed + 1000)
    with torch.no_grad():
        controller.actor.weight.add_(0.1 * torch.randn_like(controller.actor.weight))
        controller.actor.bias.add_(0.1 * torch.randn_like(controller.actor.bias))
        out = model(ids, labels=labels, controller=controller, budget=budget, sample=True,
                    generator=torch.Generator().manual_seed(seed + 1), collect_rollout=True)
        rollout = out.rollout
        joint_replay = replay_logprob(controller, rollout)
        joint_dlp = torch.zeros(rollout.episodes, dtype=torch.float64)
        memory = controller.initial_memory(rollout.episodes, ids.device)
        for d in rollout.decisions:
            output = controller(d.observation, memory.index_select(0, d.indices), d.gate_logits, d.gate_probabilities)
            memory = memory.clone()
            memory[d.indices] = output.memory
            log_prob = controller.decision_log_prob(output, d.stopped, d.expert, d.forced_continue)
            joint_dlp.index_add_(0, d.indices, log_prob.double())
        joint_manual, max_ref_diff = manual_joint_logprob(controller, rollout)
    positions = target_positions(model, labels)
    episode_calls = out.calls[torch.arange(ids.shape[0]), positions] - (layers - 1)
    accounting = decision_accounting(rollout)
    return {
        "replay_vs_decision_log_prob": float((joint_replay.double() - joint_dlp).abs().max()),
        "replay_vs_manual": float((joint_replay.double() - joint_manual).abs().max()),
        "reference_formula_max_diff": max_ref_diff,
        "episodes": int(rollout.episodes),
        "decisions": len(rollout.decisions),
        "counts_equal_calls_plus_stops": bool((accounting["counts"] == episode_calls + accounting["stopped"]).all()),
        "first_decision_forced": accounting["first_decision_forced"],
        "stop_is_terminal": accounting["stop_is_terminal"],
        "at_most_one_stop": accounting["at_most_one_stop"],
        "exactly_one_forced": accounting["exactly_one_forced"],
    }


# ---------------------------------------------------------------------------
# RLOO auxiliary constructions (training code is not delivered yet)
# ---------------------------------------------------------------------------


def rloo_groups(input_ids: torch.Tensor, budgets: torch.Tensor) -> list[list[int]]:
    """Group rollout copies of the same input: identical tokens AND identical
    budget are required for trajectories to share an RLOO baseline."""
    groups: dict[tuple, list[int]] = {}
    for i in range(input_ids.shape[0]):
        key = (tuple(input_ids[i].tolist()), int(budgets[i]))
        groups.setdefault(key, []).append(i)
    return list(groups.values())


def leave_one_out_advantages(rewards: torch.Tensor, groups: list[list[int]]) -> torch.Tensor:
    """RLOO leave-one-out baseline: advantage_i = r_i - mean(r_j, j != i).
    Singleton groups get advantage 0 (no independent baseline available)."""
    rewards = rewards.double()
    advantages = torch.zeros_like(rewards)
    for group in groups:
        for i in group:
            others = [j for j in group if j != i]
            if others:
                advantages[i] = rewards[i] - rewards[others].mean()
    return advantages


# ---------------------------------------------------------------------------
# Multi-adaptive-layer configuration (full-sequence routing)
# ---------------------------------------------------------------------------


def check_multi_adaptive(model, controller, ids, labels, budgets=(2, 3, 4, 5, 6)) -> dict:
    """adaptive_layers=(1, 3), target_only=False: budget reserve must keep
    every token's total adaptive calls within [1, budget], and full-sequence
    replays must have batch*length episodes. `budget_fully_usable` reports
    whether a never-stopping policy can actually spend the whole budget."""
    layers = model.config.layers
    n_adaptive = len(model.config.adaptive_layers)
    batch, length = ids.shape
    report = {}
    for budget in budgets:
        with torch.no_grad():
            old = force_continue(controller)
            out = model(ids, labels=labels, controller=controller, budget=budget,
                        sample=False, collect_rollout=True)
            restore_stop_bias(controller, old)
            joint = replay_logprob(controller, out.rollout)
        adaptive_calls = out.calls - (layers - n_adaptive)
        report[str(budget)] = {
            "min_adaptive_calls": int(adaptive_calls.min()),
            "max_adaptive_calls": int(adaptive_calls.max()),
            "violations": int(((adaptive_calls > budget) | (adaptive_calls < 1)).sum()),
            "budget_fully_usable": bool((adaptive_calls == budget).all()),
            "episodes": int(out.rollout.episodes),
            "expected_episodes": batch * length,
            "joint_shape": list(joint.shape),
            "joint_finite": bool(torch.isfinite(joint).all()),
            "decision_index_max": max(int(d.indices.max()) for d in out.rollout.decisions),
            "calls_shape": list(out.calls.shape),
        }
    return report


def check_multi_adaptive_target_only(ids, labels, budget=6, seed=0) -> dict:
    """Probe: target_only=True with adaptive_layers=(1, 3). This configuration
    is illegal: ModelConfig rejects it by raising ValueError, and that
    rejection is the expected behaviour (run() records it instead of
    crashing; the production constraint is deliberately not relaxed). If the
    constraint is ever lifted, the probe below measures the routed call
    structure directly."""
    model, controller = build_pair(adaptive_layers=(1, 3), target_only=True, seed=seed)
    with torch.no_grad():
        old = force_continue(controller)
        out = model(ids, labels=labels, controller=controller, budget=budget, sample=False,
                    collect_rollout=True)
        restore_stop_bias(controller, old)
    positions = target_positions(model, labels)
    rows = torch.arange(ids.shape[0])
    nontarget_mask = torch.ones_like(out.calls, dtype=torch.bool)
    nontarget_mask[rows, positions] = False
    nontarget = out.calls[nontarget_mask]
    return {
        "nontarget_min_calls": int(nontarget.min()),
        "nontarget_max_calls": int(nontarget.max()),
        "expected_nontarget_calls": model.config.layers,
        "episodes": ids.shape[0],
        "decision_index_max": max(int(d.indices.max()) for d in out.rollout.decisions),
    }


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def run(seed: int = 0) -> dict:
    items = make_items(4, seed=seed)
    batch = affine_collate(items)
    ids, labels = batch["input_ids"], batch["labels"]

    model_to, controller_to = build_pair(target_only=True, seed=seed)
    model_fs, controller_fs = build_pair(target_only=False, seed=seed)
    model_multi, controller_multi = build_pair(adaptive_layers=(1, 3), target_only=False, seed=seed)

    try:
        target_only_probe = check_multi_adaptive_target_only(ids, labels, seed=seed)
    except ValueError as error:
        # The illegal target-only multi-adaptive configuration is rejected by
        # ModelConfig; record the rejection as expected behaviour.
        target_only_probe = {
            "config_rejected": True,
            "rejection_expected": True,
            "error": str(error),
        }

    return {
        "dtype": "float64",
        "device": "cpu",
        "fixed_depth_calls_target_only": check_fixed_depth_calls(model_to, ids, labels),
        "fixed_depth_calls_full_sequence": check_fixed_depth_calls_full_sequence(model_fs, ids),
        "budget": check_budget(model_to, controller_to, ids, labels),
        "prefix_invariance": check_prefix_invariance(model_fs, controller_fs, ids, labels, cut=10),
        "batch_invariance": check_batch_invariance(model_fs, controller_fs, items),
        "sampled_batch_invariance_informational": check_sampled_batch_invariance(model_fs, controller_fs, items),
        "sampled_reproducibility": check_sampled_reproducibility(model_to, controller_to, ids, labels),
        "zero_delta_greedy_vs_fixed": check_zero_delta_greedy(model_to, controller_to, ids, labels),
        "replay": check_replay(model_to, controller_to, ids, labels, seed=seed),
        "multi_adaptive": check_multi_adaptive(model_multi, controller_multi, ids, labels),
        "multi_adaptive_target_only_probe": target_only_probe,
    }


def main():
    parser = argparse.ArgumentParser(description="Structural causality/budget/replay checks (CPU, FP64).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None,
                        help="Optional path to write the JSON report. Default: only print to stdout.")
    args = parser.parse_args()
    text = json.dumps(run(seed=args.seed), indent=2)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
