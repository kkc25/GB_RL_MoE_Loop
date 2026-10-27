"""S1 single-expert RLOO training CLI (spec 18.3, task card G).

Allowed to start only after the adaptation gate passed: ``gate.json`` in the
adapt run dir must say ``single_rl_allowed: true`` and the adapted student
checkpoint hash is pinned into the resolved config.

Differences from the pair branch (all required by spec 18.3):

- action space ``single8_stop_v1``: joint 9-class actor, prior
  ``0.9 * one_hot(argmax gate) + 0.1 / 8``, previous-expert one-hot feature;
- one expert call per round; the ledger counts ``logical_single_rounds`` and
  ``cost.lambda`` is denominated per extra single-expert round;
- reward reference CE is the adapted fixed single-expert R1 readout
  (precomputed per context from the frozen adapted student); the original
  pair teacher remains the external quality reference in evaluations;
- the canonical training backend computes all 8 experts and gathers one,
  so physical/logical expert rows have ratio exactly 8 (accounted);
- independent run dir, checkpoint format, RNG domain reuse and resume.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from data.packed_lm import PackedLMDataset
from experiments.rl_lm_single_adapt import ADAPT_CHECKPOINT_FORMAT, ADAPT_OBJECTIVE_VERSION
from rl_lm.checkpoint import (
    capture_rng_state,
    move_optimizer_state,
    restore_rng_state,
    save_atomic,
    update_pointer_manifest,
)
from rl_lm.cli import load_backbone, write_resolved_config
from rl_lm.objectives import rloo_advantage
from rl_lm.readout import terminal_ce
from rl_lm.reporting import stop_metrics
from rl_lm.rng import make_uniform_table
from rl_lm.single_actor import SingleActor, SingleDecisionRecords
from rl_lm.single_config import SingleRLConfig, resolve_single_config, single_cache_identity
from rl_lm.single_executor import (
    JOINT_SINGLE_ACTIONS,
    SINGLE_ACTIONS,
    STOP_ACTION,
    ArgmaxSingleTeacherPolicy,
    SingleLedger,
    make_single_executor,
    single_argmax_expert,
    single_ffn_update,
    single_greedy_lowest_id,
    single_online_forward,
    _single_safe_actions,
    _single_safe_state,
)
from rl_lm.single_executor import SINGLE_SPARSE_BACKEND
from rl_lm.trainer import categorical_kl, scheduled_base_lr
from rl_lm.cache import load_cache
from train.recovery_common import atomic_json, sha256_file

CHECKPOINT_FORMAT = "rl-lm-single-checkpoint-v1"
PERMUTATION_DOMAIN = "rl-lm-single-epoch-permutation-v1"
REPLAY_ATOL = 1e-5
REPLAY_RTOL = 1e-5
EXPERTS = 8


class UpdateRejected(RuntimeError):
    pass


# ------------------------------------------------------------------ rollout


@torch.no_grad()
def _inverse_cdf_sample(log_probs: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    if uniforms.dtype != torch.float64:
        raise ValueError("uniforms must remain float64")
    if not bool(((uniforms >= 0) & (uniforms < 1)).all()):
        raise ValueError("uniforms must lie in [0,1)")
    if bool(torch.isnan(log_probs).any()) or bool(torch.isposinf(log_probs).any()):
        raise ValueError("invalid log probabilities")
    probabilities = log_probs.detach().double().exp()
    normalizer = probabilities.sum(-1, keepdim=True)
    if not bool((normalizer > 0).all()):
        raise ValueError("each row needs at least one legal action")
    probabilities = probabilities / normalizer
    cdf = probabilities.cumsum(-1)
    cdf[:, -1] = 1.0
    return (cdf <= uniforms[:, None]).sum(-1).long()


class SingleEnvironment:
    """Frozen terminal-FFN single-expert environment (R1 in single-v1)."""

    def __init__(self, model, *, max_calls: int, executor, head_chunk: int = 64, price_lambda: float = 0.0, ledger: SingleLedger | None = None) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if price_lambda < 0:
            raise ValueError("lambda must be nonnegative (nat per extra single-expert round)")
        if executor.model is not model:
            raise ValueError("executor must wrap the same model")
        self.model = model
        self.max_calls = int(max_calls)
        self.executor = executor
        self.head_chunk = int(head_chunk)
        self.price_lambda = float(price_lambda)
        self.ledger = ledger

    @torch.no_grad()
    def rollout(self, contexts, actor, uniforms: torch.Tensor, *, policy_version: int = 0):
        if uniforms.ndim != 3 or uniforms.dtype != torch.float64:
            raise ValueError("uniforms must be float64 [B,G,Rmax]")
        batch, groups, slots = uniforms.shape
        if contexts.h0.shape[0] != batch or slots != self.max_calls:
            raise ValueError("uniforms shape does not match contexts or Rmax")
        device = contexts.h0.device
        count = batch * groups
        u = uniforms.reshape(count, slots)
        h0 = contexts.h0.repeat_interleave(groups, dim=0)
        targets = contexts.targets.repeat_interleave(groups)
        ref_ce = contexts.ref_ce.repeat_interleave(groups)
        h = h0.clone()
        rounds = torch.zeros(count, dtype=torch.int64, device=device)
        done = torch.zeros(count, dtype=torch.bool, device=device)
        previous_expert = torch.full((count,), -1, dtype=torch.int64, device=device)

        rec_features, rec_reference, rec_stop_allowed = [], [], []
        rec_actions, rec_flat_ids, rec_log_probs = [], [], []

        for slot in range(slots):
            active = ~done
            if not bool(active.any()):
                break
            safe_h, safe_h0, safe_rounds, safe_previous = _single_safe_state(h, h0, rounds, previous_expert, active)
            observation = self.executor.observe(safe_h, safe_h0, safe_rounds, safe_previous, max_calls=self.max_calls)
            stop_allowed = safe_rounds >= 1
            log_probs = actor(observation.features, observation.reference, stop_allowed)
            if tuple(log_probs.shape) != (count, JOINT_SINGLE_ACTIONS):
                raise ValueError("actor must return joint log-probs [N,9]")
            if bool(torch.isnan(log_probs).any()) or bool(torch.isposinf(log_probs).any()):
                raise ValueError("actor produced invalid log-probs")
            actions = _inverse_cdf_sample(log_probs, u[:, slot])
            if bool(((actions == STOP_ACTION) & ~stop_allowed & active).any()):
                raise RuntimeError("illegal STOP sampled at r=0; fail fast instead of remapping")

            rows = torch.nonzero(active, as_tuple=False).squeeze(-1)
            rec_features.append(observation.features[rows].clone())
            rec_reference.append(observation.reference[rows].clone())
            rec_stop_allowed.append(stop_allowed[rows].clone())
            rec_actions.append(actions[rows].clone())
            rec_flat_ids.append(rows.clone())
            rec_log_probs.append(log_probs[rows].clone())

            take = active & (actions > STOP_ACTION)
            voluntary_stop = active & (actions == STOP_ACTION)
            proposed = self.executor.step_single(safe_h, observation, _single_safe_actions(actions, active))
            h = torch.where(take[:, None], proposed, h)
            previous_expert = torch.where(take, actions - 1, previous_expert)
            rounds = rounds + take.to(torch.int64)
            done = done | voluntary_stop | (rounds >= self.max_calls)

            if self.ledger is not None:
                self.ledger.actor_decision_rows += int(rows.numel())
                self.ledger.actor_padding_rows += int(count - rows.numel())
                self.ledger.logical_single_rounds += int(take.sum())
                self.ledger.logical_expert_rows += int(take.sum())

        if not bool(done.all()):
            raise RuntimeError("environment ended with unfinished episodes")
        ce = terminal_ce(self.model, h, targets, chunk=self.head_chunk, ledger=self.ledger)
        returns = ref_ce - ce - self.price_lambda * (rounds - 1).float()

        actions_all = torch.cat(rec_actions)
        log_probs_all = torch.cat(rec_log_probs)
        flat_ids_all = torch.cat(rec_flat_ids)
        chosen = log_probs_all.gather(1, actions_all[:, None]).squeeze(1)
        joint = chosen.new_zeros(count)
        joint = joint.index_add(0, flat_ids_all, chosen)

        records = SingleDecisionRecords(
            features=torch.cat(rec_features),
            reference=torch.cat(rec_reference),
            stop_allowed=torch.cat(rec_stop_allowed),
            actions=actions_all,
            flat_episode_ids=flat_ids_all,
            old_log_probs_all=log_probs_all,
        )
        rollout = SimpleNamespace(
            terminal_ce=ce.reshape(batch, groups),
            rounds=rounds.reshape(batch, groups),
            returns=returns.reshape(batch, groups),
            old_log_joint=joint.reshape(batch, groups),
            policy_version=policy_version,
        )
        if self.ledger is not None:
            self.ledger.rollout_episodes += count
        return rollout, records


# ------------------------------------------------------------------ replay


def single_replay_log_joint(actor, records: SingleDecisionRecords, batch: int, groups: int) -> torch.Tensor:
    total = int(batch) * int(groups)
    if records.flat_episode_ids.numel() and not bool((records.flat_episode_ids < total).all()):
        raise ValueError("flat_episode_ids exceed B*G - 1")
    if bool((records.actions < 0).any()) or bool((records.actions >= JOINT_SINGLE_ACTIONS).any()):
        raise ValueError("actions must be legal joint actions in [0,8]")
    log_probs = actor(records.features, records.reference, records.stop_allowed)
    if tuple(log_probs.shape) != (records.features.shape[0], JOINT_SINGLE_ACTIONS):
        raise ValueError("actor must return joint log-probs [K,9]")
    chosen = log_probs.gather(1, records.actions[:, None]).squeeze(1)
    joint = chosen.new_zeros(total)
    joint = joint.index_add(0, records.flat_episode_ids, chosen)
    return joint.reshape(int(batch), int(groups))


def _context_record_mask(records: SingleDecisionRecords, context_ids: torch.Tensor, groups: int) -> torch.Tensor:
    context_ids = torch.as_tensor(context_ids, dtype=torch.int64, device=records.flat_episode_ids.device)
    contexts = records.flat_episode_ids // int(groups)
    return (contexts[:, None] == context_ids[None, :]).any(dim=1)


def _filter_records(records: SingleDecisionRecords, mask: torch.Tensor) -> SingleDecisionRecords:
    return SingleDecisionRecords(
        features=records.features[mask],
        reference=records.reference[mask],
        stop_allowed=records.stop_allowed[mask],
        actions=records.actions[mask],
        flat_episode_ids=records.flat_episode_ids[mask],
        old_log_probs_all=records.old_log_probs_all[mask],
    )


def single_microbatch_pg_loss(actor, records, advantage, batch, groups, context_ids) -> torch.Tensor:
    if advantage.shape != (int(batch), int(groups)):
        raise ValueError("advantage must be the full [B,G] tensor")
    if advantage.requires_grad:
        raise ValueError("advantage must be detached")
    sub = _filter_records(records, _context_record_mask(records, context_ids, groups))
    joint = single_replay_log_joint(actor, sub, batch, groups)
    return -(advantage * joint).sum() / (int(batch) * int(groups))


def iter_context_microbatches(batch: int, microbatch: int):
    for start in range(0, int(batch), int(microbatch)):
        yield torch.arange(start, min(start + int(microbatch), int(batch)))


# ------------------------------------------------------------------ contexts


class SingleCacheContextProvider:
    """Contexts over the S1 cache; ref_ce is the adapted single-R1 reference."""

    def __init__(self, cache: dict, ref_ce: np.ndarray) -> None:
        self._h0 = cache["h0"]
        self._targets = cache["targets"]
        self._ref_ce = np.asarray(ref_ce, dtype=np.float32)
        self._episode_ids = tuple(row["episode_id"] for row in cache["episode_ids"])
        if not (len(self._h0) == len(self._targets) == len(self._ref_ce) == len(self._episode_ids)):
            raise ValueError("cache arrays and reference disagree in length")

    def __len__(self) -> int:
        return len(self._h0)

    def gather(self, indices, device):
        index = [int(i) for i in indices]
        return SimpleNamespace(
            h0=torch.from_numpy(np.ascontiguousarray(self._h0[index])).to(device),
            targets=torch.from_numpy(np.ascontiguousarray(self._targets[index])).to(device),
            ref_ce=torch.from_numpy(self._ref_ce[index].copy()).to(device),
            episode_ids=tuple(self._episode_ids[i] for i in index),
        )


@torch.no_grad()
def adapted_single_reference_ce(model, cache: dict, device, *, chunk: int = 2048, compute_dtype: str = "bfloat16") -> np.ndarray:
    """Fixed reference: adapted student executing the argmax single expert (amplitude 1)."""
    layer = int(model.config.adaptive_layers[-1])
    block = model.blocks[layer]
    gamma = float(model.gammas[layer])
    h0 = cache["h0"]
    targets = cache["targets"]
    out = np.zeros(h0.shape[0], dtype=np.float64)
    for start in range(0, h0.shape[0], chunk):
        h = torch.from_numpy(np.ascontiguousarray(h0[start:start + chunk])).to(device)
        y = torch.from_numpy(np.ascontiguousarray(targets[start:start + chunk])).to(device)
        n = block.ffn_norm(h).float()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(compute_dtype == "bfloat16")):
            z = block.moe.gate(n)
            experts = single_argmax_expert(z.float())
            update = single_ffn_update(block.moe.pool, n, experts)
        ce = terminal_ce(model, h + gamma * update, y, chunk=2048)
        out[start:start + chunk] = ce.double().cpu().numpy()
    return out


# ------------------------------------------------------------------ trainer


class SingleTrainer:
    def __init__(self, config, model, *, train_contexts, evaluator=None, device=None, resume_path=None, stop_after=None,
                 backbone_identity=None, data_identity=None, adapted_identity=None) -> None:
        self.config = config.validate()
        self.model = model.eval()
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                raise ValueError("the adapted student backbone must be fully frozen before RL")
        self.device = torch.device(device) if device is not None else torch.device(self.config.get("runtime.device"))
        self.run_dir = Path(self.config.get("output_dir"))
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.actor_seed = int(self.config.get("actor_seed"))
        self.rmax = int(self.config.get("environment.max_calls"))
        self.batch = int(self.config.get("optimization.batch_contexts"))
        self.groups = int(self.config.get("optimization.groups"))
        self.microbatch = int(self.config.get("optimization.replay_microbatch_contexts"))
        self.planned_updates = int(self.config.get("optimization.updates"))
        self.peak_lr = float(self.config.get("optimization.learning_rate"))
        self.warmup = int(self.config.get("optimization.warmup_updates"))
        self.final_ratio = float(self.config.get("optimization.final_lr_ratio"))
        self.grad_clip = float(self.config.get("optimization.gradient_clip"))
        self.betas = tuple(self.config.get("optimization.betas"))
        self.eps = float(self.config.get("optimization.epsilon"))
        self.weight_decay = float(self.config.get("optimization.weight_decay"))
        self.max_kl_mean = float(self.config.get("optimization.max_update_kl_mean"))
        self.max_kl_state = float(self.config.get("optimization.max_update_kl_state"))
        self.max_retries = int(self.config.get("optimization.max_retries_per_update"))
        self.max_backoffs = int(self.config.get("optimization.max_lr_backoffs_per_run"))
        self.eval_interval = int(self.config.get("evaluation.interval_updates"))
        self.log_interval = int(self.config.get("logging.interval_updates"))
        self.tie_tolerance = float(self.config.get("evaluation.selection_tie_tolerance"))
        self.stop_after = self.planned_updates if stop_after is None else min(int(stop_after), self.planned_updates)
        if self.stop_after < 1:
            raise ValueError("--stop-after must be a positive absolute cap")

        self.train_contexts = train_contexts
        if len(train_contexts) < self.batch:
            raise ValueError("train cache has fewer contexts than the batch")
        self.batches_per_epoch = len(train_contexts) // self.batch
        self.evaluator = evaluator

        hidden = int(self.model.config.hidden_size)
        self.feature_dim = 2 * hidden + 2 * EXPERTS + 4

        self.ledger = SingleLedger()
        self.executor = make_single_executor(
            self.model,
            self.config.get("environment.training_backend"),
            chunk=int(self.config.get("environment.expert_chunk")),
            compute_dtype=self.config.get("environment.frozen_compute_dtype"),
            ledger=self.ledger,
        )
        self.environment = SingleEnvironment(
            self.model,
            max_calls=self.rmax,
            executor=self.executor,
            head_chunk=int(self.config.get("environment.head_chunk")),
            price_lambda=float(self.config.get("cost.lambda")),
            ledger=self.ledger,
        )

        self.backbone_identity = dict(backbone_identity or {})
        self.data_identity = dict(data_identity or {})
        self.adapted_identity = dict(adapted_identity or {})
        self.frozen_hash_at_start = _model_parameter_hash(self.model)

        self.successful_updates = 0
        self.epoch = 0
        self.epoch_cursor = 0
        self._permutation = None
        self._visited = set()
        self.lr_scale = 1.0
        self.lr_backoffs = 0
        self.selection_state = {"best": None, "candidates": [], "evaluations": []}
        self.start_time = time.time()
        self.resumed = resume_path is not None

        if self.resumed:
            self._restore_run(Path(resume_path))
        else:
            if (self.run_dir / "checkpoint_latest.pt").exists():
                raise FileExistsError("run dir already contains a run; use --resume")
            random.seed(self.actor_seed)
            np.random.seed(self.actor_seed)
            torch.manual_seed(self.actor_seed)
            self.actor = SingleActor(self.feature_dim, train_stop=False).to(self.device)
            self.optimizer = torch.optim.AdamW(
                [p for p in self.actor.parameters() if p.requires_grad],
                lr=0.0, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay,
            )
            self.run_id = "rl-lm-single-{}-{}".format(
                self.config.sha256[:12], datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            )
            self.created_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------- scheduling

    def base_lr(self, j: int) -> float:
        return scheduled_base_lr(j, peak=self.peak_lr, warmup=self.warmup, total=self.planned_updates, final_ratio=self.final_ratio)

    # ------------------------------------------------------------------- data

    def _epoch_permutation(self) -> np.ndarray:
        key = json.dumps([PERMUTATION_DOMAIN, self.actor_seed, self.epoch], separators=(",", ":"))
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], "little")
        return np.random.Generator(np.random.Philox(seed)).permutation(len(self.train_contexts))

    def _next_batch_indices(self):
        if self._permutation is None:
            self._permutation = self._epoch_permutation()
        indices = [int(i) for i in self._permutation[self.epoch_cursor:self.epoch_cursor + self.batch]]
        self.epoch_cursor += self.batch
        if self.epoch_cursor >= self.batches_per_epoch * self.batch:
            self.epoch += 1
            self.epoch_cursor = 0
            self._permutation = None
        return indices

    # ----------------------------------------------------------------- update

    def _check_update_preconditions(self) -> None:
        trainable = [p for p in self.actor.parameters() if p.requires_grad]
        grouped = [p for group in self.optimizer.param_groups for p in group["params"]]
        if len(grouped) != len(trainable) or any(a is not b for a, b in zip(grouped, trainable)):
            raise RuntimeError("optimizer parameter whitelist does not match the actor's trainable parameters")
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad or parameter.grad is not None:
                raise RuntimeError(f"frozen LM parameter {name} is trainable or carries a gradient")

    def _begin_update(self) -> SimpleNamespace:
        self._check_update_preconditions()
        indices = self._next_batch_indices()
        contexts = self.train_contexts.gather(indices, self.device)
        uniforms = make_uniform_table(self.actor_seed, self.successful_updates, contexts.episode_ids, self.groups, self.rmax).to(self.device)
        env_start = time.perf_counter()
        rollout, records = self.environment.rollout(contexts, self.actor, uniforms, policy_version=self.successful_updates)
        env_seconds = time.perf_counter() - env_start
        if rollout.policy_version != self.successful_updates:
            raise RuntimeError("rollout policy_version mismatch")
        advantage = rloo_advantage(rollout.returns)
        with torch.no_grad():
            check = single_replay_log_joint(self.actor, records, self.batch, self.groups)
        replay_error = float((check - rollout.old_log_joint).abs().max())
        if not bool(torch.isclose(check, rollout.old_log_joint, atol=REPLAY_ATOL, rtol=REPLAY_RTOL).all()):
            raise RuntimeError(f"pre-update replay disagrees with the rollout (max |diff|={replay_error:.3e})")

        actor_start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for context_ids in iter_context_microbatches(self.batch, self.microbatch):
            loss = single_microbatch_pg_loss(self.actor, records, advantage, self.batch, self.groups, context_ids)
            loss.backward()
            total_loss += float(loss.detach())
        trainable = [p for p in self.actor.parameters() if p.requires_grad]
        for parameter in trainable:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError("actor gradient missing or non-finite")
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                raise RuntimeError("frozen LM received a gradient during actor backward")
        grad_norm = float(nn.utils.clip_grad_norm_(trainable, self.grad_clip))
        if not math.isfinite(grad_norm):
            raise RuntimeError("non-finite gradient norm")
        actor_seconds = time.perf_counter() - actor_start
        return SimpleNamespace(
            indices=indices, contexts=contexts, uniforms=uniforms, rollout=rollout, records=records,
            advantage=advantage, replay_error=replay_error, loss=total_loss, grad_norm=grad_norm,
            clipped=grad_norm > self.grad_clip, env_seconds=env_seconds, actor_seconds=actor_seconds,
        )

    def _current_log_probs(self, records) -> torch.Tensor:
        with torch.no_grad():
            return self.actor(records.features, records.reference, records.stop_allowed)

    def _restore_snapshot(self, snapshot) -> None:
        self.actor.load_state_dict(copy.deepcopy(snapshot["actor"]))
        self.optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        move_optimizer_state(self.optimizer, self.device)
        self.lr_scale = snapshot["lr_scale"]
        self.lr_backoffs = snapshot["lr_backoffs"]

    def _guarded_step(self, records) -> dict:
        j = self.successful_updates + 1
        snapshot = {
            "actor": copy.deepcopy(self.actor.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "lr_scale": self.lr_scale,
            "lr_backoffs": self.lr_backoffs,
        }
        attempts = 0
        while True:
            lr = self.base_lr(j) * self.lr_scale
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.step()
            attempts += 1
            self.ledger.optimizer_attempts += 1
            if not all(bool(torch.isfinite(p).all()) for p in self.actor.parameters()):
                self._restore_snapshot(snapshot)
                self._write_update_rejected(j, attempts, "nan_parameters", None)
                raise UpdateRejected(f"update {j} produced non-finite actor parameters; run terminated")
            kl = categorical_kl(records.old_log_probs_all, self._current_log_probs(records))
            kl_mean, kl_max = float(kl.mean()), float(kl.max())
            if kl_mean <= self.max_kl_mean and kl_max <= self.max_kl_state:
                return {"attempts": attempts, "lr": lr, "lr_scale": self.lr_scale, "kl_mean": kl_mean, "kl_max": kl_max}
            if attempts > self.max_retries or self.lr_backoffs >= self.max_backoffs:
                reason = "kl_budget_exhausted" if self.lr_backoffs >= self.max_backoffs else "kl_retry_exhausted"
                self._restore_snapshot(snapshot)
                self._write_update_rejected(j, attempts, reason, {"kl_mean": kl_mean, "kl_max": kl_max})
                raise UpdateRejected(f"update {j} rejected: step KL mean={kl_mean:.4g} max={kl_max:.4g}")
            self._restore_snapshot(snapshot)
            self.lr_backoffs += 1
            self.lr_scale *= 0.5

    def _update(self) -> dict:
        info = self._begin_update()
        step = self._guarded_step(info.records)
        self.successful_updates += 1
        self.ledger.successful_updates += 1
        self.ledger.context_visits += len(info.indices)
        self._visited.update(info.indices)
        self.ledger.unique_train_contexts = len(self._visited)
        self.ledger.merge_peak(self.device)
        entry = self._metrics_entry(info, step)
        with (self.run_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        if entry["update"] % self.log_interval == 0:
            print(json.dumps({"update": entry["update"], "actor_loss": entry["actor_loss"],
                              "greedy_nll_best": (self.selection_state["best"] or {}).get("greedy_nll"), "lr": entry["lr"]}))
        return entry

    # ----------------------------------------------------------------- metrics

    def _metrics_entry(self, info, step) -> dict:
        rollout, records, advantage = info.rollout, info.records, info.advantage
        returns = rollout.returns
        same = (returns == returns[:, :1]).all(dim=1)
        lp = records.old_log_probs_all
        legal = torch.isfinite(lp)
        p = torch.where(legal, lp.exp(), torch.zeros_like(lp))
        plp = torch.where(legal, p * lp, torch.zeros_like(lp))
        entropy = -plp.sum(dim=-1)
        chosen = lp.gather(1, records.actions[:, None]).squeeze(1)
        expert_mask = records.actions > STOP_ACTION
        expert_usage = torch.bincount((records.actions[expert_mask] - 1).cpu(), minlength=EXPERTS)
        teacher = records.reference.argmax(dim=-1)
        agreement = float(((records.actions[expert_mask] - 1) == teacher[expert_mask]).float().mean()) if bool(expert_mask.any()) else 0.0
        rounds_hist = torch.bincount(rollout.rounds.reshape(-1).cpu(), minlength=self.rmax + 1).tolist()
        return {
            "run_id": self.run_id,
            "stage": "r1-single",
            "policy_version": self.successful_updates,
            "update": self.successful_updates,
            "epoch": self.epoch,
            "data_cursor": self.epoch_cursor,
            "actor_seed": self.actor_seed,
            "lambda": float(self.config.get("cost.lambda")),
            "cost_unit": self.config.get("cost.unit"),
            "batch_contexts": self.batch,
            "batch_episode_sha256": hashlib.sha256(json.dumps(list(info.contexts.episode_ids), separators=(",", ":")).encode()).hexdigest(),
            "uniforms_sha256": hashlib.sha256(info.uniforms.cpu().numpy().tobytes()).hexdigest(),
            "actions_sha256": hashlib.sha256(records.actions.cpu().numpy().tobytes()).hexdigest(),
            "actor_loss": info.loss,
            "sampled_terminal_nll": float(rollout.terminal_ce.mean()),
            "ref_nll_adapted_single": float(info.contexts.ref_ce.mean()),
            "reward_without_cost": float((info.contexts.ref_ce[:, None] - rollout.terminal_ce).mean()),
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std(unbiased=False)),
            "advantage_mean": float(advantage.mean()),
            "advantage_std": float(advantage.std(unbiased=False)),
            "zero_adv_context_fraction": float(same.float().mean()),
            "rounds_histogram": rounds_hist,
            **stop_metrics(records.actions, records.stop_allowed, max_calls=self.rmax),
            "expert_usage": expert_usage.tolist(),
            "teacher_expert_agreement": agreement,
            "legal_entropy_mean": float(entropy.mean()),
            "joint_max_prob_mean": float(p.max(dim=-1).values.mean()),
            "expert_max_prob_mean": float(p[:, 1:].max(dim=-1).values.mean()),
            "chosen_logprob_min": float(chosen.min()),
            "chosen_logprob_max": float(chosen.max()),
            "grad_norm_unclipped": info.grad_norm,
            "grad_clipped": info.clipped,
            "lr": step["lr"],
            "lr_scale": step["lr_scale"],
            "step_kl_mean": step["kl_mean"],
            "step_kl_max": step["kl_max"],
            "retry_count": step["attempts"] - 1,
            "lr_backoffs_total": self.lr_backoffs,
            "replay_max_abs_error": info.replay_error,
            "frozen_lm_grad_check": "ok",
            "finiteness_check": "ok",
            "environment_seconds": info.env_seconds,
            "actor_seconds": info.actor_seconds,
            "wall_time_seconds_cumulative": time.time() - self.start_time,
            "ledger": self.ledger.to_dict(),
        }

    # -------------------------------------------------------------- evaluation

    def _evaluate_and_select(self) -> dict:
        if self.evaluator is None:
            raise RuntimeError("no selection evaluator wired")
        metrics = self.evaluator(self.model, self.actor, self.successful_updates)
        candidate = {"update": self.successful_updates, "greedy_nll": float(metrics["greedy_nll"]), "mean_rounds": float(metrics["mean_rounds"])}
        record = dict(candidate)
        record["metrics"] = metrics
        self.selection_state["evaluations"].append(record)
        self.selection_state["candidates"].append(candidate)
        with (self.run_dir / "evaluations.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        best = self.selection_state["best"]
        is_better = best is None
        if best is not None:
            delta = candidate["greedy_nll"] - best["greedy_nll"]
            if delta < -self.tie_tolerance:
                is_better = True
            elif abs(delta) < self.tie_tolerance:
                if candidate["mean_rounds"] < best["mean_rounds"]:
                    is_better = True
                elif candidate["mean_rounds"] == best["mean_rounds"] and candidate["update"] < best["update"]:
                    is_better = True
        if is_better:
            self.selection_state["best"] = dict(candidate, path="checkpoint_best.pt")
            self._save_checkpoint("best")
        return record

    # -------------------------------------------------------------- checkpoint

    def _checkpoint_payload(self) -> dict:
        return {
            "format_version": CHECKPOINT_FORMAT,
            "spec_version": "RL-LM-Single-v1.0",
            "run_id": self.run_id,
            "stage": "r1",
            "created_at": self.created_at,
            "saved_at_update": self.successful_updates,
            "resolved_config": self.config.to_dict(),
            "config_sha256": self.config.sha256,
            "backbone_identity": dict(self.backbone_identity) | {"frozen_parameters_sha256": self.frozen_hash_at_start},
            "adapted_identity": self.adapted_identity,
            "data_identity": self.data_identity,
            "action_contract": {"action_space": "single8_stop_v1", "joint_actions": JOINT_SINGLE_ACTIONS},
            "feature_schema": self.config.get("policy.feature_schema"),
            "feature_dim": self.feature_dim,
            "numerical_contract": {
                "training_backend": self.config.get("environment.training_backend"),
                "expert_chunk": int(self.config.get("environment.expert_chunk")),
                "head_chunk": int(self.config.get("environment.head_chunk")),
                "frozen_compute_dtype": self.config.get("environment.frozen_compute_dtype"),
                "residual_dtype": self.config.get("environment.residual_dtype"),
                "head_contract": self.config.get("environment.head_contract"),
                "gamma": self.executor.gamma,
                "physical_logical_expert_ratio": 8,
            },
            "actor_state_dict": {k: v.detach().cpu() for k, v in self.actor.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state": {
                "successful_updates": self.successful_updates,
                "planned_updates": self.planned_updates,
                "warmup_updates": self.warmup,
                "final_lr_ratio": self.final_ratio,
                "peak_lr": self.peak_lr,
                "lr_scale": self.lr_scale,
                "lr_backoffs": self.lr_backoffs,
            },
            "training_state": {
                "successful_updates": self.successful_updates,
                "next_update": self.successful_updates + 1,
                "epoch": self.epoch,
                "next_batch_cursor": self.epoch_cursor,
                "batches_per_epoch": self.batches_per_epoch,
                "permutation_domain": PERMUTATION_DOMAIN,
                "visited_contexts": sorted(self._visited),
            },
            "rng_state": capture_rng_state(),
            "selection_state": copy.deepcopy(self.selection_state),
            "ledger": self.ledger.to_dict(),
        }

    def _save_checkpoint(self, kind: str) -> Path:
        path = save_atomic(self.run_dir / f"checkpoint_{kind}.pt", self._checkpoint_payload())
        update_pointer_manifest(self.run_dir, kind, path)
        return path

    def _restore_run(self, path: Path) -> None:
        if path.parent.resolve() != self.run_dir.resolve():
            raise ValueError("an exact resume must point at a checkpoint of this run's output_dir")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != CHECKPOINT_FORMAT or payload.get("spec_version") != "RL-LM-Single-v1.0":
            raise ValueError("not an S1 RL checkpoint of this spec version")
        stored = copy.deepcopy(payload["resolved_config"])
        current = copy.deepcopy(self.config.to_dict())
        stored["runtime"]["device"] = current["runtime"]["device"] = "<agnostic>"
        if stored != current:
            raise ValueError("exact resume requires the identical resolved configuration; only runtime.device may differ")
        training = payload["training_state"]["successful_updates"]
        scheduler = payload["scheduler_state"]["successful_updates"]
        ledger = payload["ledger"]["successful_updates"]
        if not (training == scheduler == ledger):
            raise ValueError("inconsistent successful_updates in checkpoint")
        self.actor = SingleActor(self.feature_dim, train_stop=False)
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.actor.to(self.device)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.actor.parameters() if p.requires_grad],
            lr=0.0, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay,
        )
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        move_optimizer_state(self.optimizer, self.device)
        scheduler_state = payload["scheduler_state"]
        self.lr_scale = float(scheduler_state["lr_scale"])
        self.lr_backoffs = int(scheduler_state["lr_backoffs"])
        training_state = payload["training_state"]
        self.successful_updates = int(training_state["successful_updates"])
        self.epoch = int(training_state["epoch"])
        self.epoch_cursor = int(training_state["next_batch_cursor"])
        self._visited = set(training_state["visited_contexts"])
        self.ledger = SingleLedger.from_dict(payload["ledger"])
        self.executor.ledger = self.ledger
        self.environment.ledger = self.ledger
        self.selection_state = copy.deepcopy(payload["selection_state"])
        self.run_id = payload["run_id"]
        self.created_at = payload["created_at"]
        restore_rng_state(payload["rng_state"])
        stored_frozen = (payload.get("backbone_identity") or {}).get("frozen_parameters_sha256")
        if stored_frozen is not None and stored_frozen != self.frozen_hash_at_start:
            raise ValueError("frozen backbone parameter hash changed since the checkpoint was written")
        if self.stop_after < self.successful_updates:
            raise ValueError("--stop-after is below the completed updates")

    def _write_update_rejected(self, update, attempts, reason, kl) -> None:
        atomic_json(self.run_dir / "update_rejected.json", {
            "update": update, "attempts": attempts, "reason": reason, "kl": kl,
            "lr_scale": self.lr_scale, "lr_backoffs": self.lr_backoffs,
            "successful_updates": self.successful_updates, "ledger": self.ledger.to_dict(),
        })

    # -------------------------------------------------------------------- run

    def run(self) -> dict:
        try:
            if not self.resumed:
                atomic_json(self.run_dir / "resolved_config.json", self.config.to_dict())
                self._evaluate_and_select()
                self._save_checkpoint("initial")
                if self.selection_state["best"] is None:
                    raise RuntimeError("update-0 evaluation did not initialize the best record")
                self._save_checkpoint("latest")
            while self.successful_updates < self.stop_after:
                self._update()
                if self.successful_updates % self.eval_interval == 0:
                    self._evaluate_and_select()
                self._save_checkpoint("latest")
            return self._finish("completed" if self.successful_updates >= self.planned_updates else "stopped_at_cap")
        except Exception as exc:
            atomic_json(self.run_dir / "failure.json", {
                "exception_type": type(exc).__name__, "message": str(exc),
                "successful_updates_completed": self.successful_updates,
            })
            raise

    def _finish(self, status: str) -> dict:
        if _model_parameter_hash(self.model) != self.frozen_hash_at_start:
            raise RuntimeError("frozen backbone parameter hash changed during training")
        self.ledger.merge_peak(self.device)
        wall = time.time() - self.start_time
        self.ledger.record_wall_time(wall)
        atomic_json(self.run_dir / "cost_ledger.json", self.ledger.to_dict() | {"wall_time_seconds": wall})
        atomic_json(self.run_dir / "selection_manifest.json", {
            "rule": "lowest greedy_nll on selection; ties prefer fewer rounds then earlier update",
            "best": self.selection_state["best"],
            "candidates": self.selection_state["candidates"],
        })
        summary = {
            "status": status,
            "run_id": self.run_id,
            "successful_updates": self.successful_updates,
            "planned_updates": self.planned_updates,
            "stop_after": self.stop_after,
            "lr_scale": self.lr_scale,
            "lr_backoffs": self.lr_backoffs,
            "best": self.selection_state["best"],
            "ledger": self.ledger.to_dict(),
            "frozen_lm_hash_stable": True,
            "run_dir": str(self.run_dir),
        }
        atomic_json(self.run_dir / "final_summary.json", summary)
        return summary


def _hash_tensors(tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _model_parameter_hash(model) -> str:
    return _hash_tensors(model.parameters())


# ------------------------------------------------------------------ evaluator


class SingleOnlineSelectionEvaluator:
    """Greedy joint-9 full-forward NLL on selection sequences.

    References: the adapted fixed single-expert R1 rule (primary, same
    environment) and the original pair teacher (external quality contrast).
    """

    def __init__(self, config, adapted_model, teacher_model, device) -> None:
        self.config = config
        self.device = device
        dataset = PackedLMDataset(
            config.get("data.corpus_dir"),
            config.get("data.selection_split"),
            limit=int(config.get("evaluation.online_selection_sequences")),
        )
        items = [dataset[i] for i in range(len(dataset))]
        self.input_ids = torch.from_numpy(np.stack([item["input_ids"] for item in items])).to(device)
        self.segment_ids = torch.from_numpy(np.stack([item["segment_ids"] for item in items])).to(device)
        self.labels = torch.from_numpy(np.stack([item["labels"] for item in items])).to(device)
        self.valid = self.labels != -100
        from experiments.rl_lm_single_adapt import pair_teacher_selection_nll, single_selection_nll

        self.adapted_reference_nll = single_selection_nll(
            adapted_model, self.input_ids, self.segment_ids, self.labels,
            compute_dtype=config.get("environment.frozen_compute_dtype"),
            chunk=int(config.get("environment.expert_chunk")),
            head_chunk=int(config.get("environment.head_chunk")),
        )
        self.pair_teacher_nll = pair_teacher_selection_nll(
            teacher_model, self.input_ids, self.segment_ids, self.labels,
            compute_dtype=config.get("environment.frozen_compute_dtype"),
            chunk=int(config.get("environment.expert_chunk")),
            head_chunk=int(config.get("environment.head_chunk")),
        )

    def _forward_nll(self, model, policy) -> tuple[float, float]:
        logits, rounds = single_online_forward(
            model,
            policy,
            self.input_ids,
            self.segment_ids,
            max_calls=int(self.config.get("environment.max_calls")),
            backend=self.config.get("environment.deployment_backend"),
            compute_dtype=self.config.get("environment.frozen_compute_dtype"),
            chunk=int(self.config.get("environment.expert_chunk")),
            head_chunk=int(self.config.get("environment.head_chunk")),
        )
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = self.labels.reshape(-1)
        ce_sum = 0.0
        rows = 8192
        for start in range(0, flat_labels.shape[0], rows):
            ce_sum += float(
                torch.nn.functional.cross_entropy(
                    flat_logits[start:start + rows].float(), flat_labels[start:start + rows],
                    reduction="none", ignore_index=-100,
                ).sum()
            )
        nll = ce_sum / int(self.valid.sum())
        mean_rounds = float(rounds.reshape(-1)[self.valid.reshape(-1)].float().mean())
        return nll, mean_rounds

    def __call__(self, model, actor, update: int) -> dict:
        greedy_nll, mean_rounds = self._forward_nll(model, actor)
        return {
            "greedy_nll": greedy_nll,
            "mean_rounds": mean_rounds,
            "reference_nll_adapted_single": self.adapted_reference_nll,
            "pair_teacher_nll_external": self.pair_teacher_nll,
            "delta_nll_vs_adapted_single": greedy_nll - self.adapted_reference_nll,
            "delta_nll_vs_pair_teacher": greedy_nll - self.pair_teacher_nll,
            "eval_backend": self.config.get("environment.deployment_backend"),
            "eval_sequences": int(self.input_ids.shape[0]),
            "eval_tokens": int(self.valid.sum()),
        }


# ------------------------------------------------------------------ main


def validate_adaptation_gate(gate: dict, adapted_sha: str) -> None:
    if gate.get("objective_version") != ADAPT_OBJECTIVE_VERSION:
        raise ValueError("legacy adaptation gate cannot authorize repaired pair-teacher single-RL")
    if not gate.get("single_rl_allowed"):
        raise ValueError("adaptation gate does not allow single-RL")
    if gate.get("student_checkpoint_sha256") != adapted_sha:
        raise ValueError("adaptation gate belongs to a different student checkpoint")


def reference_ce_identity(cache: dict, adapted_sha: str, compute_dtype: str) -> dict:
    # Bind actual payload/order, not only a manifest that might have stale bytes.
    digest = hashlib.sha256()
    for key in ("h0", "targets"):
        array = np.asarray(cache[key])
        digest.update(json.dumps([key, str(array.dtype), list(array.shape)]).encode())
        for start in range(0, len(array), 2048):
            digest.update(np.ascontiguousarray(array[start:start + 2048]).tobytes())
    digest.update(json.dumps(cache["episode_ids"], sort_keys=True).encode())
    return {"version": "adapted_single_reference_ce_v2", "adapted_checkpoint_sha256": adapted_sha,
            "cache_payload_sha256": digest.hexdigest(), "cache_manifest": cache["manifest"],
            "compute_dtype": compute_dtype, "head_contract": "production_bf16_head_v1",
            "expert_rows_per_chunk": 2048, "head_chunk": 2048}


def validate_reference_ce_identity(meta: dict, expected: dict) -> None:
    if meta.get("identity") != expected:
        raise ValueError("cached reference CE identity mismatch (checkpoint/cache/order/numerics); use a new run directory")


def validate_adapted_objective(payload: dict) -> None:
    if (payload.get("format_version") != ADAPT_CHECKPOINT_FORMAT
            or payload.get("objective_version") != ADAPT_OBJECTIVE_VERSION):
        raise ValueError("legacy/wrong-objective adapted checkpoint cannot start repaired single-RL; run a new pair-teacher adaptation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--corpus-dir")
    parser.add_argument("--tokenizer")
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()

    config = resolve_single_config(
        args.config, SingleRLConfig,
        checkpoint=args.checkpoint, corpus_dir=args.corpus_dir, tokenizer=args.tokenizer, device=args.device,
    )
    adapt_dir = Path(config.get("adapted.adapt_run_dir"))
    gate_path = adapt_dir / "gate.json"
    if not gate_path.exists():
        raise RuntimeError("adaptation gate.json missing; run experiments.rl_lm_single_adapt first")
    gate = json.loads(gate_path.read_text())
    if gate.get("objective_version") != ADAPT_OBJECTIVE_VERSION:
        raise ValueError("legacy adaptation gate cannot authorize repaired pair-teacher single-RL; run a new adaptation")
    if not gate.get("single_rl_allowed"):
        raise RuntimeError(f"adaptation gate did not pass ({gate_path}); single-RL is not allowed to start")
    adapted_path = Path(config.get("adapted.checkpoint_path") or gate["student_checkpoint"])
    adapted_sha = sha256_file(adapted_path)
    validate_adaptation_gate(gate, adapted_sha)
    if config.get("adapted.expected_checkpoint_sha256") is None:
        config.data["adapted"]["expected_checkpoint_sha256"] = adapted_sha
    config.validate_adapted_hash(adapted_sha)

    output_dir = Path(config.get("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.get("runtime.device"))

    teacher, _ = load_backbone(config, device)
    config.validate(backbone_model=teacher.config, runtime=True)
    payload = torch.load(adapted_path, map_location=device, weights_only=False)
    if payload.get("kind") != "adapted_student" or payload.get("spec_version") != "RL-LM-Single-v1.0":
        raise ValueError("adapted checkpoint is not an S1 adapted student")
    validate_adapted_objective(payload)
    if payload.get("teacher_checkpoint_sha256") != sha256_file(config.get("backbone.checkpoint_path")):
        raise ValueError("adapted student was not derived from this teacher checkpoint")
    model = copy.deepcopy(teacher)
    model.load_state_dict(payload["student_state_dict"])
    model.eval().requires_grad_(False)

    data_identity = {}
    caches = {}
    for split in (config.get("data.train_split"), config.get("data.selection_split")):
        pair_manifest = json.loads((Path(config.get("data.pair_cache_root")) / split / "manifest.json").read_text())
        identity = single_cache_identity(
            teacher,
            checkpoint_path=config.get("backbone.checkpoint_path"),
            config_path=config.get("backbone.config_path"),
            tokenizer_path=config.get("data.tokenizer_path"),
            data_dir=config.get("data.corpus_dir"),
            split=split,
            sampling_seed=config.get("data.sampling_seed"),
            expert_chunk=config.get("environment.expert_chunk"),
            pair_manifest=pair_manifest,
        )
        caches[split] = load_cache(Path(config.get("data.cache_root")) / split, identity)
        data_identity[split] = identity

    reference_path = output_dir / "reference_ce.npy"
    reference_meta_path = output_dir / "reference_ce.json"
    reference_identity = reference_ce_identity(caches[config.get("data.train_split")], adapted_sha,
                                               config.get("environment.frozen_compute_dtype"))
    if reference_path.exists():
        meta = json.loads(reference_meta_path.read_text())
        validate_reference_ce_identity(meta, reference_identity)
        ref_ce = np.load(reference_path)
    else:
        ref_ce = adapted_single_reference_ce(
            model, caches[config.get("data.train_split")], device,
            compute_dtype=config.get("environment.frozen_compute_dtype"),
        )
        np.save(reference_path, ref_ce)
        atomic_json(reference_meta_path, {
            "identity": reference_identity,
            "adapted_checkpoint_sha256": adapted_sha,
            "rule": "adapted student, fixed argmax single expert, amplitude 1, R1",
            "states": int(ref_ce.shape[0]),
            "mean_ce": float(ref_ce.mean()),
        })

    backbone_identity = {
        "teacher_checkpoint_path": config.get("backbone.checkpoint_path"),
        "teacher_checkpoint_sha256": sha256_file(config.get("backbone.checkpoint_path")),
        "config_path": config.get("backbone.config_path"),
        "config_sha256": sha256_file(config.get("backbone.config_path")),
        "tokenizer_sha256": sha256_file(config.get("data.tokenizer_path")),
    }
    adapted_identity = {
        "adapted_checkpoint_path": str(adapted_path),
        "adapted_checkpoint_sha256": adapted_sha,
        "adapt_gate": gate,
    }
    trainer = SingleTrainer(
        config,
        model,
        train_contexts=SingleCacheContextProvider(caches[config.get("data.train_split")], ref_ce),
        evaluator=SingleOnlineSelectionEvaluator(config, model, teacher, device),
        device=device,
        resume_path=args.resume,
        stop_after=args.stop_after,
        backbone_identity=backbone_identity,
        data_identity=data_identity,
        adapted_identity=adapted_identity,
    )
    summary = trainer.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
