"""Training state machine: sample -> replay -> exactly one monitored update.

Implements spec sections 11.2-11.5 and 12:

- each update draws B distinct contexts from the fixed epoch permutation
  (rebuilt with a fixed seed at each epoch boundary, never padded by
  repetition), regenerates the full G x T uniform table keyed by
  ``(actor_seed, successful_update, episode_id, g)``, collects real
  on-policy rollouts in ``no_grad``, computes G-wise leave-one-out
  advantages, verifies the pre-update replay against the stored old
  log-joints (FP32 atol=rtol=1e-5), backprops the PG loss once (optionally in
  context microbatches with the global ``B*G`` reduction), clips the grad
  norm once at 1.0, and performs a single KL-monitored optimizer step;
- the learning rate schedule is indexed by *successful* updates only
  (``j = successful_updates + 1``; warmup then cosine to 10% of peak over the
  planned 2000 updates), and the KL-rollback LR multiplier ``lr_scale`` is
  persistent training state;
- the monitored step snapshots actor + optimizer + scheduler state with a
  real deepcopy *before* stepping, measures the full categorical KL(old||new)
  over the visited decision states, allows at most one retry on the same old
  parameters and gradients with the LR halved, enforces a run-level backoff
  budget (default 4), and on terminal rejection restores the snapshot, writes
  ``update_rejected.json`` and ends the run without saving a half-updated
  checkpoint — rollout/optimizer resource costs stay in the ledger;
- rollouts are discarded immediately after a successful update; there is no
  PPO ratio, no multi-epoch reuse, no critic.

Evaluation and selection follow spec 16.1: every new run evaluates at
update 0 and saves ``checkpoint_initial.pt`` as the first candidate;
fixed-interval evaluations retain NLL selection for R1; new
R2 uses a versioned cost objective subject to a pinned selected-R1 quality
bound. An exact resume never resets or reinterprets that record.
"""

from __future__ import annotations

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

from rl_lm.actions import JOINT_ACTIONS, STOP_ACTION, action_table_identity
from rl_lm.actor import PairActor
from rl_lm.checkpoint import (
    CHECKPOINT_FORMAT,
    SPEC_VERSION,
    capture_rng_state,
    move_optimizer_state,
    restore_rng_state,
    resume_exact,
    save_atomic,
    update_pointer_manifest,
    warmstart_actor as load_warmstart_payload,
)
from rl_lm.environment import (
    LastFFNEnvironment,
    action_for_safe_execution,
    make_safe_observation_state,
)
from rl_lm.executor import FrozenLastFFNExecutor
from rl_lm.inference import greedy_lowest_id
from rl_lm.ledger import RLLedger
from rl_lm.objectives import rloo_advantage
from rl_lm.readout import terminal_ce
from rl_lm.replay import iter_context_microbatches, microbatch_pg_loss, replay_log_joint
from rl_lm.reporting import stop_metrics
from rl_lm.rng import make_uniform_table
from rl_lm.training_state import (
    advance_epoch_batch, capture_update_snapshot, epoch_permutation, restore_update_snapshot,
)
from rl_lm.types import ContextBatch
from train.recovery_common import atomic_json, sha256_file

EXPERTS = 8
PAIR_ACTIONS = 28
PERMUTATION_DOMAIN = "rl-lm-epoch-permutation-v1"
REPLAY_ATOL = 1e-5
REPLAY_RTOL = 1e-5


class UpdateRejected(RuntimeError):
    """The monitored optimizer step exceeded its retry/backoff budget."""


def scheduled_base_lr(j: int, *, peak: float, warmup: int, total: int, final_ratio: float) -> float:
    """LR of the j-th (1-based) attempted update; indexed by successful updates."""
    if j < 1:
        raise ValueError("j is 1-based")
    if warmup < 1 or total <= warmup:
        raise ValueError("require 1 <= warmup < total")
    if j <= warmup:
        return peak * j / warmup
    progress = (j - warmup) / (total - warmup)
    return peak * (final_ratio + (1.0 - final_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0)


def categorical_kl(old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor:
    """Per-state full categorical KL(old||new) over legal actions, [K].

    Illegal STOP entries carry -inf in the old distribution and are skipped
    (never 0 * -inf); the new distribution must be finite wherever the old
    one is.
    """
    if old_log_probs.shape != new_log_probs.shape or old_log_probs.ndim != 2:
        raise ValueError("KL inputs must share [K,A] shape")
    old = old_log_probs.detach().double()
    new = new_log_probs.detach().double()
    legal = torch.isfinite(old)
    if bool((~torch.isfinite(new[legal])).any()):
        raise ValueError("new log-probs must be finite on every previously legal action")
    p = torch.where(legal, old.exp(), torch.zeros_like(old))
    diff = torch.where(legal, old - new, torch.zeros_like(old))
    return (p * diff).sum(dim=-1)


class TensorContextProvider:
    """In-memory context provider (toy fixtures and tests)."""

    def __init__(self, h0: torch.Tensor, targets: torch.Tensor, ref_ce: torch.Tensor, episode_ids) -> None:
        batch = ContextBatch(h0=h0, targets=targets, ref_ce=ref_ce, episode_ids=tuple(episode_ids))
        self._h0 = batch.h0.cpu()
        self._targets = batch.targets.cpu()
        self._ref_ce = batch.ref_ce.cpu()
        self._episode_ids = batch.episode_ids

    def __len__(self) -> int:
        return self._h0.shape[0]

    def gather(self, indices, device) -> ContextBatch:
        index = torch.as_tensor(list(indices), dtype=torch.int64)
        return ContextBatch(
            h0=self._h0[index].to(device),
            targets=self._targets[index].to(device),
            ref_ce=self._ref_ce[index].to(device),
            episode_ids=tuple(self._episode_ids[int(i)] for i in index),
        )


class CacheContextProvider:
    """Context provider over a validated prefix cache (``rl_lm.cache.load_cache``)."""

    def __init__(self, cache: dict) -> None:
        if not isinstance(cache.get("verified_identity"), dict) or not cache["verified_identity"].get("content_sha256"):
            raise ValueError("cache provider requires content-verified load_cache output")
        self.verified_cache_identity = copy.deepcopy(cache["verified_identity"])
        self._h0 = cache["h0"]
        self._targets = cache["targets"]
        self._ref_ce = np.asarray([row["canonical_ce"] for row in cache["baseline"]], dtype=np.float32)
        self._episode_ids = tuple(row["episode_id"] for row in cache["episode_ids"])
        if not (len(self._h0) == len(self._targets) == len(self._ref_ce) == len(self._episode_ids)):
            raise ValueError("cache arrays and identity rows disagree in length")

    def __len__(self) -> int:
        return len(self._h0)

    def gather(self, indices, device) -> ContextBatch:
        index = [int(i) for i in indices]
        return ContextBatch(
            h0=torch.from_numpy(np.ascontiguousarray(self._h0[index])).to(device),
            targets=torch.from_numpy(np.ascontiguousarray(self._targets[index])).to(device),
            ref_ce=torch.from_numpy(self._ref_ce[index].copy()).to(device),
            episode_ids=tuple(self._episode_ids[i] for i in index),
        )


@torch.no_grad()
def greedy_rollout(environment: LastFFNEnvironment, contexts: ContextBatch, actor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic joint-argmax rollout; returns (per-context CE, rounds)."""
    batch = contexts.h0.shape[0]
    device = contexts.h0.device
    h0 = contexts.h0
    h = h0.clone()
    rounds = torch.zeros(batch, dtype=torch.int64, device=device)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    previous_pair = torch.full((batch, 2), -1, dtype=torch.int64, device=device)
    for _ in range(environment.max_calls):
        active = ~done
        if not bool(active.any()):
            break
        safe_h, safe_h0, safe_rounds, safe_previous = make_safe_observation_state(h, h0, rounds, previous_pair, active)
        observation = environment.executor.observe(safe_h, safe_h0, safe_rounds, safe_previous, max_calls=environment.max_calls)
        stop_allowed = safe_rounds >= 1
        log_probs = actor(observation.features, observation.pair_reference, stop_allowed)
        if tuple(log_probs.shape) != (batch, JOINT_ACTIONS):
            raise ValueError("actor must return joint log-probs [N,29]")
        actions = greedy_lowest_id(log_probs)
        if bool(((actions == STOP_ACTION) & ~stop_allowed & active).any()):
            raise RuntimeError("illegal greedy STOP at r=0")
        take_pair = active & (actions > STOP_ACTION)
        voluntary_stop = active & (actions == STOP_ACTION)
        proposed = environment.executor.step_pair(safe_h, observation, action_for_safe_execution(actions, active))
        h = torch.where(take_pair[:, None], proposed, h)
        pairs = environment.executor.table[(actions.clamp_min(1) - 1)]
        previous_pair = torch.where(take_pair[:, None], pairs, previous_pair)
        rounds = rounds + take_pair.to(torch.int64)
        done = done | voluntary_stop | (rounds >= environment.max_calls)
    if not bool(done.all()):
        raise RuntimeError("greedy rollout ended with unfinished episodes")
    ce = terminal_ce(environment.model, h, contexts.targets, chunk=environment.head_chunk, ledger=environment.ledger)
    return ce, rounds


class GreedyCacheEvaluator:
    """Greedy joint-argmax NLL over a fixed cache-side context set.

    This is the deterministic diagnostic evaluator; the CLI wires the online
    full-forward selection evaluator of spec 16.1 instead.
    """

    def __init__(self, environment: LastFFNEnvironment, contexts: ContextBatch) -> None:
        self.environment = environment
        self.contexts = contexts

    def __call__(self, model, actor, update: int) -> dict:
        ce, rounds = greedy_rollout(self.environment, self.contexts, actor)
        greedy_nll = float(ce.mean())
        result = {
            "greedy_nll": greedy_nll,
            "mean_rounds": float(rounds.float().mean()),
            "eval_backend": "cache_greedy_canonical_v1",
            "eval_states": int(self.contexts.h0.shape[0]),
        }
        reference = float(self.contexts.ref_ce.mean())
        result["reference_nll"] = reference
        result["delta_nll"] = greedy_nll - reference
        result["gain_nll"] = reference - greedy_nll
        return result


def _hash_tensors(tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def model_parameter_hash(model) -> str:
    return _hash_tensors(model.parameters())


def _json_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class Trainer:
    """On-policy RLOO trainer with transactional updates and exact resume."""

    def __init__(
        self,
        config,
        model,
        *,
        train_contexts,
        evaluator=None,
        device=None,
        run_dir=None,
        resume_path=None,
        warmstart_path=None,
        stop_after=None,
        backbone_identity=None,
        data_identity=None,
        source_identity=None,
    ) -> None:
        if resume_path is not None and warmstart_path is not None:
            raise ValueError("--resume and --warmstart-actor are mutually exclusive")
        self.config = config.validate()
        self.model = model.eval()
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                raise ValueError("the LM backbone must be fully frozen before training")
        self.device = torch.device(device) if device is not None else torch.device(self.config.get("runtime.device"))
        self.run_dir = Path(run_dir) if run_dir is not None else Path(self.config.get("output_dir"))
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.stage = self.config.get("stage")
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

        if stop_after is None:
            self.stop_after = self.planned_updates
        else:
            if int(stop_after) < 1:
                raise ValueError("--stop-after must be a positive absolute cap")
            self.stop_after = min(int(stop_after), self.planned_updates)

        self.train_contexts = train_contexts
        total_contexts = len(train_contexts)
        if total_contexts < self.batch:
            raise ValueError(f"train cache has {total_contexts} contexts, fewer than batch {self.batch}")
        self.batches_per_epoch = total_contexts // self.batch
        self.evaluator = evaluator
        self.selection_policy = self._selection_policy()

        hidden = int(self.model.config.hidden_size)
        self.feature_dim = 2 * hidden + 2 * EXPERTS + 4

        self.ledger = RLLedger()
        self.executor = FrozenLastFFNExecutor(
            self.model,
            chunk=int(self.config.get("environment.expert_chunk")),
            compute_dtype=self.config.get("environment.frozen_compute_dtype"),
            ledger=self.ledger,
        )
        self.environment = LastFFNEnvironment(
            self.model,
            max_calls=self.rmax,
            executor=self.executor,
            head_chunk=int(self.config.get("environment.head_chunk")),
            price_lambda=float(self.config.get("cost.lambda")),
            ledger=self.ledger,
        )

        self.source_identity = source_identity
        self.backbone_identity = dict(backbone_identity or {})
        self.data_identity = dict(data_identity or {})
        verified_cache_identity = getattr(train_contexts, "verified_cache_identity", None)
        if verified_cache_identity is not None:
            # Caller-supplied source identity cannot override verified payload identity.
            self.data_identity["verified_train_cache"] = copy.deepcopy(verified_cache_identity)
        self.frozen_hash_at_start = model_parameter_hash(self.model)

        self.successful_updates = 0
        self.epoch = 0
        self.epoch_cursor = 0
        self._permutation = None
        self._visited = set()
        self.lr_scale = 1.0
        self.lr_backoffs = 0
        self.selection_state = {"best": None, "candidates": [], "evaluations": [],
                                "policy": copy.deepcopy(self.selection_policy)}
        self.warmstart_from = None
        self.start_time = time.time()
        self.resumed = resume_path is not None

        if self.resumed:
            self.initialization = "resumed"
            self._restore_run(Path(resume_path))
        else:
            self.initialization = "fresh"
            self._start_new_run(warmstart_path)

    # ------------------------------------------------------------------ setup

    def _build_actor(self) -> PairActor:
        torch.manual_seed(self.actor_seed)
        return PairActor(self.feature_dim, train_stop=(self.stage == "r2")).to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.actor.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=0.0, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)

    def _start_new_run(self, warmstart_path) -> None:
        from rl_lm.run_directory import guard_run_directory

        guard_run_directory(self.run_dir, resume=False)
        if (self.run_dir / "checkpoint_latest.pt").exists():
            raise FileExistsError(
                f"{self.run_dir} already contains a run; exact state can only be continued with --resume"
            )
        random.seed(self.actor_seed)
        np.random.seed(self.actor_seed)
        self.actor = self._build_actor()
        if warmstart_path is not None:
            self.initialization = "warmstart"
            payload = load_warmstart_payload(
                warmstart_path,
                feature_schema=self.config.get("policy.feature_schema"),
                feature_dim=self.feature_dim,
                action_sha256=action_table_identity()["sha256"],
            )
            parent = payload["actor_state_dict"]
            inherited = {k: v for k, v in parent.items() if not k.startswith("stop_head.")}
            self.actor.load_state_dict(inherited, strict=False)
            nn.init.zeros_(self.actor.stop_head.weight)
            nn.init.zeros_(self.actor.stop_head.bias)
            self.actor.stop_head.requires_grad_(self.stage == "r2")
            self.warmstart_from = {
                "path": str(warmstart_path),
                "sha256": sha256_file(warmstart_path),
                "parent_run_id": payload.get("run_id"),
                "parent_successful_updates": payload["training_state"]["successful_updates"],
            }
        self.optimizer = self._build_optimizer()
        if self.source_identity is None:
            from experiments.r1_closed_loop import source_identity

            self.source_identity = source_identity()
        self.run_id = "rl-lm-{}-{}".format(
            self.config.sha256[:12], datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        )
        self.created_at = datetime.now(timezone.utc).isoformat()

    def _restore_run(self, resume_path: Path) -> None:
        payload = resume_exact(resume_path, resolved_config=self.config.to_dict())
        if resume_path.parent.resolve() != self.run_dir.resolve():
            raise ValueError("an exact resume must point at a checkpoint of this run's output_dir")
        self._validate_resume_identities(payload)
        self._validate_selection_resume(payload["selection_state"])
        self.actor = self._build_actor()
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.actor.to(self.device)
        self.optimizer = self._build_optimizer()
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        move_optimizer_state(self.optimizer, self.device)
        scheduler = payload["scheduler_state"]
        self.lr_scale = float(scheduler["lr_scale"])
        self.lr_backoffs = int(scheduler["lr_backoffs"])
        training = payload["training_state"]
        self.successful_updates = int(training["successful_updates"])
        self.epoch = int(training["epoch"])
        self.epoch_cursor = int(training["next_batch_cursor"])
        self._visited = set(training["visited_contexts"])
        self.ledger = RLLedger.from_dict(payload["ledger"])
        self.executor.ledger = self.ledger
        self.environment.ledger = self.ledger
        self.selection_state = copy.deepcopy(payload["selection_state"])
        self.run_id = payload["run_id"]
        self.created_at = payload["created_at"]
        self.warmstart_from = payload.get("warmstart_from")
        if self.source_identity is None:
            self.source_identity = payload.get("source_identity")
        restore_rng_state(payload["rng_state"])
        stored_frozen = (payload.get("backbone_identity") or {}).get("frozen_parameters_sha256")
        if stored_frozen is not None and stored_frozen != self.frozen_hash_at_start:
            raise ValueError("frozen backbone parameter hash changed since the checkpoint was written")
        if self.stop_after < self.successful_updates:
            raise ValueError(
                f"--stop-after {self.stop_after} is below the completed {self.successful_updates} updates"
            )

    def _validate_resume_identities(self, payload: dict) -> None:
        """Pin actual inputs, not just unchanged paths in the resolved config.

        Git commit labels are informational: identical trees/diffs/untracked
        Python bytes may continue across a metadata-only commit. All other
        source fields remain pinned; this is not an implicit code migration.
        """
        def normalized(value):
            if isinstance(value, dict):
                return {key: normalized(item) for key, item in value.items() if key != "commit"}
            if isinstance(value, list):
                return [normalized(item) for item in value]
            return value

        if self.source_identity is None:
            from experiments.r1_closed_loop import source_identity

            self.source_identity = source_identity()
        expected = {
            "backbone_identity": self.backbone_identity | {"frozen_parameters_sha256": self.frozen_hash_at_start},
            "data_identity": self.data_identity,
            "source_identity": self.source_identity,
            "numerical_contract": self._numerical_contract(),
            "action_contract": action_table_identity(),
            "feature_schema": self.config.get("policy.feature_schema"),
            "feature_dim": self.feature_dim,
        }
        for key, current in expected.items():
            if normalized(payload.get(key)) != normalized(current):
                raise ValueError(f"exact resume {key} mismatch; changed inputs require a new run")

    def _numerical_contract(self) -> dict:
        return {
            "training_backend": self.config.get("environment.training_backend"),
            "expert_chunk": int(self.config.get("environment.expert_chunk")),
            "head_chunk": int(self.config.get("environment.head_chunk")),
            "frozen_compute_dtype": self.config.get("environment.frozen_compute_dtype"),
            "residual_dtype": self.config.get("environment.residual_dtype"),
            "head_contract": self.config.get("environment.head_contract"),
            "gamma": self.executor.gamma,
        }

    # ------------------------------------------------------------- scheduling

    def base_lr(self, j: int) -> float:
        return scheduled_base_lr(
            j, peak=self.peak_lr, warmup=self.warmup, total=self.planned_updates, final_ratio=self.final_ratio
        )

    def upcoming_lr(self) -> float:
        return self.base_lr(self.successful_updates + 1) * self.lr_scale

    # ------------------------------------------------------------------- data

    def _epoch_permutation(self) -> np.ndarray:
        return epoch_permutation(domain=PERMUTATION_DOMAIN, actor_seed=self.actor_seed,
                                 epoch=self.epoch, count=len(self.train_contexts))

    def _next_batch_indices(self) -> list[int]:
        if self._permutation is None:
            self._permutation = self._epoch_permutation()
        indices, self.epoch, self.epoch_cursor, self._permutation = advance_epoch_batch(
            self._permutation, epoch=self.epoch, cursor=self.epoch_cursor,
            batch=self.batch, batches_per_epoch=self.batches_per_epoch,
        )
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
        uniforms = make_uniform_table(
            self.actor_seed, self.successful_updates, contexts.episode_ids, self.groups, self.rmax
        ).to(self.device)
        env_start = time.perf_counter()
        rollout, records = self.environment.rollout(
            contexts, self.actor, uniforms, policy_version=self.successful_updates
        )
        env_seconds = time.perf_counter() - env_start
        if rollout.policy_version != self.successful_updates:
            raise RuntimeError("rollout policy_version does not match the successful update count")
        advantage = rloo_advantage(rollout.returns)
        with torch.no_grad():
            check = replay_log_joint(self.actor, records, self.batch, self.groups)
        replay_error = float((check - rollout.old_log_joint).abs().max())
        if not bool(torch.isclose(check, rollout.old_log_joint, atol=REPLAY_ATOL, rtol=REPLAY_RTOL).all()):
            raise RuntimeError(
                f"pre-update replay disagrees with the rollout (max |diff|={replay_error:.3e}); "
                "the update is stopped before touching the optimizer"
            )

        actor_start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for context_ids in iter_context_microbatches(self.batch, self.microbatch):
            loss = microbatch_pg_loss(self.actor, records, advantage, self.batch, self.groups, context_ids)
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
            indices=indices,
            contexts=contexts,
            uniforms=uniforms,
            rollout=rollout,
            records=records,
            advantage=advantage,
            replay_error=replay_error,
            loss=total_loss,
            grad_norm=grad_norm,
            clipped=grad_norm > self.grad_clip,
            env_seconds=env_seconds,
            actor_seconds=actor_seconds,
        )

    def _current_log_probs(self, records) -> torch.Tensor:
        with torch.no_grad():
            return self.actor(records.features, records.pair_reference, records.stop_allowed)

    def _restore_snapshot(self, snapshot) -> None:
        self.lr_scale, self.lr_backoffs = restore_update_snapshot(
            self.actor, self.optimizer, snapshot, device=self.device,
        )

    def _guarded_step(self, records) -> dict:
        """One monitored optimizer step with at most one halved-LR retry."""
        j = self.successful_updates + 1
        snapshot = capture_update_snapshot(
            self.actor, self.optimizer, lr_scale=self.lr_scale, lr_backoffs=self.lr_backoffs,
        )
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
                return {
                    "attempts": attempts,
                    "lr": lr,
                    "lr_scale": self.lr_scale,
                    "kl_mean": kl_mean,
                    "kl_max": kl_max,
                }
            if attempts > self.max_retries or self.lr_backoffs >= self.max_backoffs:
                reason = "kl_budget_exhausted" if self.lr_backoffs >= self.max_backoffs else "kl_retry_exhausted"
                self._restore_snapshot(snapshot)
                self._write_update_rejected(j, attempts, reason, {"kl_mean": kl_mean, "kl_max": kl_max})
                raise UpdateRejected(
                    f"update {j} rejected: step KL mean={kl_mean:.4g} max={kl_max:.4g} "
                    f"after {attempts} attempt(s); run terminated without a half-updated checkpoint"
                )
            self._restore_snapshot(snapshot)
            self.lr_backoffs += 1
            self.lr_scale *= 0.5

    def _commit_update(self, info: SimpleNamespace, step: dict) -> None:
        self.successful_updates += 1
        self.ledger.successful_updates += 1
        self.ledger.context_visits += len(info.indices)
        self._visited.update(info.indices)
        self.ledger.unique_train_contexts = len(self._visited)
        self.ledger.merge_peak(self.device)

    def _update(self) -> dict:
        info = self._begin_update()
        step = self._guarded_step(info.records)
        self._commit_update(info, step)
        entry = self._metrics_entry(info, step)
        self._append_metrics(entry)
        # Rollout trajectories are discarded here; nothing is reused (no PPO epochs).
        del info
        return entry

    # ----------------------------------------------------------------- metrics

    def _metrics_entry(self, info: SimpleNamespace, step: dict) -> dict:
        rollout, records, advantage = info.rollout, info.records, info.advantage
        contexts = info.contexts
        returns = rollout.returns
        same = (returns == returns[:, :1]).all(dim=1)
        lp = records.old_log_probs_all
        legal = torch.isfinite(lp)
        p = torch.where(legal, lp.exp(), torch.zeros_like(lp))
        plp = torch.where(legal, p * lp, torch.zeros_like(lp))
        entropy = -plp.sum(dim=-1)
        chosen = lp.gather(1, records.actions[:, None]).squeeze(1)
        pair_mask = records.actions > STOP_ACTION
        pair_usage = torch.bincount((records.actions[pair_mask] - 1).cpu(), minlength=PAIR_ACTIONS)
        table = self.executor.table.cpu()
        expert_usage = torch.zeros(EXPERTS, dtype=torch.int64)
        if bool(pair_mask.any()):
            used_pairs = table[(records.actions[pair_mask] - 1).cpu()]
            flat = used_pairs.reshape(-1)
            expert_usage = torch.bincount(flat, minlength=EXPERTS)
        teacher = records.pair_reference.argmax(dim=-1)
        agreement = float(((records.actions[pair_mask] - 1) == teacher[pair_mask]).float().mean()) if bool(pair_mask.any()) else 0.0
        paths: dict[int, list[int]] = {}
        for flat_id, action in zip(records.flat_episode_ids.tolist(), records.actions.tolist()):
            paths.setdefault(flat_id, []).append(action)
        distinct = []
        for b in range(self.batch):
            seqs = [tuple(paths[b * self.groups + g]) for g in range(self.groups)]
            distinct.append(len(set(seqs)))
        rounds_hist = torch.bincount(rollout.rounds.reshape(-1).cpu(), minlength=self.rmax + 1).tolist()
        entry = {
            "run_id": self.run_id,
            "initialization": self.initialization,
            "stage": self.stage,
            "policy_version": self.successful_updates,
            "update": self.successful_updates,
            "epoch": self.epoch,
            "data_cursor": self.epoch_cursor,
            "actor_seed": self.actor_seed,
            "lambda": float(self.config.get("cost.lambda")),
            "batch_contexts": self.batch,
            "batch_episode_sha256": _json_sha256(list(contexts.episode_ids)),
            "uniforms_sha256": hashlib.sha256(info.uniforms.cpu().numpy().tobytes()).hexdigest(),
            "actions_sha256": hashlib.sha256(records.actions.cpu().numpy().tobytes()).hexdigest(),
            "actor_loss": info.loss,
            "sampled_terminal_nll": float(rollout.terminal_ce.mean()),
            "ref_nll": float(contexts.ref_ce.mean()),
            "reward_without_cost": float((contexts.ref_ce[:, None] - rollout.terminal_ce).mean()),
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std(unbiased=False)),
            "advantage_mean": float(advantage.mean()),
            "advantage_std": float(advantage.std(unbiased=False)),
            "zero_adv_context_fraction": float(same.float().mean()),
            "distinct_paths_per_context": float(np.mean(distinct)),
            "rounds_histogram": rounds_hist,
            **stop_metrics(records.actions, records.stop_allowed, max_calls=self.rmax),
            "pair_usage": pair_usage.tolist(),
            "expert_usage": expert_usage.tolist(),
            "teacher_pair_agreement": agreement,
            "legal_entropy_mean": float(entropy.mean()),
            "joint_max_prob_mean": float(p.max(dim=-1).values.mean()),
            "pair_max_prob_mean": float(p[:, 1:].max(dim=-1).values.mean()),
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
        return entry

    def _append_metrics(self, entry: dict) -> None:
        with (self.run_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        if entry["update"] % self.log_interval == 0:
            print(json.dumps({"update": entry["update"], "actor_loss": entry["actor_loss"],
                              "greedy_nll_best": (self.selection_state["best"] or {}).get("greedy_nll"),
                              "lr": entry["lr"]}))

    # -------------------------------------------------------------- evaluation

    def _selection_policy(self) -> dict:
        price = float(self.config.get("cost.lambda"))
        if self.stage == "r1":
            # R1 has fixed cost and retains historical selection semantics.
            return {"version": "legacy_greedy_nll_v1"}
        identity = getattr(self.evaluator, "r1_identity", None)
        digest = (identity or {}).get("checkpoint_sha256", "")
        if (self.stage != "r2" or not identity or identity.get("max_calls") != 1
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("R2 selection requires a validated --r1-reference-checkpoint")
        tolerance = float(self.config.get("evaluation.r2_quality_tolerance_nll"))
        if tolerance != 0.005:
            raise ValueError("R2 selection requires the spec's 0.005 NLL tolerance")
        return {"version": "r2_selected_r1_bound_v1", "lambda": price,
                "quality_tolerance_nll": tolerance, "r1_checkpoint_sha256": digest}

    def _validate_selection_resume(self, state: dict) -> None:
        stored = state.get("policy", {"version": "legacy_greedy_nll_v1"})
        if stored != self.selection_policy:
            raise ValueError("selection policy/reference changed; legacy R2 runs cannot exact-resume under the new rule")
        if "lambda" in self.selection_policy and not math.isfinite(state.get("r1_reference_nll", float("nan"))):
            raise ValueError("R2 resume lacks its pinned R1 selection NLL")

    def _evaluate_and_select(self) -> dict:
        if self.evaluator is None:
            raise RuntimeError("no selection evaluator wired; refusing to track checkpoints blind")
        metrics = self.evaluator(self.model, self.actor, self.successful_updates)
        candidate = {
            "update": self.successful_updates,
            "greedy_nll": float(metrics["greedy_nll"]),
            "mean_rounds": float(metrics["mean_rounds"]),
        }
        score_key = "greedy_nll"
        eligible = True
        if "lambda" in self.selection_policy:
            reference = metrics.get("vs_selected_r1") or {}
            identity = reference.get("identity") or {}
            if identity.get("checkpoint_sha256") != self.selection_policy["r1_checkpoint_sha256"]:
                raise ValueError("R2 selection evaluation changed or omitted the pinned R1 reference")
            r1_nll = float(reference.get("nll", float("nan")))
            if not all(math.isfinite(v) for v in (r1_nll, candidate["greedy_nll"], candidate["mean_rounds"])):
                raise ValueError("R2 selection metrics must be finite")
            if not 1 <= candidate["mean_rounds"] <= self.rmax:
                raise ValueError("R2 selection mean_rounds outside the execution contract")
            pinned = self.selection_state.setdefault("r1_reference_nll", r1_nll)
            if r1_nll != pinned:
                raise ValueError("pinned R1 selection NLL changed since initial evaluation")
            candidate["selection_score"] = candidate["greedy_nll"] + self.selection_policy["lambda"] * (candidate["mean_rounds"] - 1)
            candidate["quality_bound_nll"] = pinned + self.selection_policy["quality_tolerance_nll"]
            eligible = candidate["greedy_nll"] <= candidate["quality_bound_nll"]
            candidate["eligible"] = eligible
            score_key = "selection_score"
        record = dict(candidate)
        record["metrics"] = metrics
        self.selection_state["evaluations"].append(record)
        self.selection_state["candidates"].append(candidate)
        with (self.run_dir / "evaluations.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        best = self.selection_state["best"]
        is_better = eligible and best is None
        if eligible and best is not None:
            delta = candidate[score_key] - best[score_key]
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
            "spec_version": SPEC_VERSION,
            "run_id": self.run_id,
            "stage": self.stage,
            "created_at": self.created_at,
            "saved_at_update": self.successful_updates,
            "resolved_config": self.config.to_dict(),
            "config_sha256": self.config.sha256,
            "source_identity": self.source_identity,
            "backbone_identity": dict(self.backbone_identity)
            | {"frozen_parameters_sha256": self.frozen_hash_at_start},
            "data_identity": self.data_identity,
            "action_contract": action_table_identity(),
            "feature_schema": self.config.get("policy.feature_schema"),
            "feature_dim": self.feature_dim,
            "numerical_contract": self._numerical_contract(),
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
            "warmstart_from": self.warmstart_from,
        }

    def _save_checkpoint(self, kind: str) -> Path:
        path = save_atomic(self.run_dir / f"checkpoint_{kind}.pt", self._checkpoint_payload())
        update_pointer_manifest(self.run_dir, kind, path)
        return path

    def _write_update_rejected(self, update: int, attempts: int, reason: str, kl: dict | None) -> None:
        atomic_json(
            self.run_dir / "update_rejected.json",
            {
                "update": update,
                "attempts": attempts,
                "reason": reason,
                "kl": kl,
                "lr_scale": self.lr_scale,
                "lr_backoffs": self.lr_backoffs,
                "successful_updates": self.successful_updates,
                "ledger": self.ledger.to_dict(),
            },
        )

    # -------------------------------------------------------------------- run

    def _initialize_run(self) -> None:
        atomic_json(self.run_dir / "resolved_config.json", self.config.to_dict())
        if self.source_identity is not None:
            atomic_json(self.run_dir / "source_identity.json", self.source_identity)
        initial = self._evaluate_and_select()
        self._save_checkpoint("initial")
        if self.selection_state["best"] is None and "lambda" not in self.selection_policy:
            raise RuntimeError("update-0 evaluation did not initialize the best record")
        self._save_checkpoint("latest")
        return initial

    def run(self) -> dict:
        try:
            if not self.resumed:
                self._initialize_run()
            while self.successful_updates < self.stop_after:
                self._update()
                if self.successful_updates % self.eval_interval == 0:
                    self._evaluate_and_select()
                self._save_checkpoint("latest")
            return self._finish("completed" if self.successful_updates >= self.planned_updates else "stopped_at_cap")
        except Exception as exc:
            self._record_failure(exc)
            raise

    def _finish(self, status: str) -> dict:
        frozen_now = model_parameter_hash(self.model)
        if frozen_now != self.frozen_hash_at_start:
            raise RuntimeError("frozen backbone parameter hash changed during training")
        self.ledger.merge_peak(self.device)
        wall = time.time() - self.start_time
        atomic_json(self.run_dir / "cost_ledger.json", self.ledger.to_dict() | {"wall_time_seconds": wall})
        atomic_json(
            self.run_dir / "selection_manifest.json",
            {
                "selection_status": "selected" if self.selection_state["best"] else "no_feasible_candidate",
                "policy": self.selection_policy,
                "r1_reference_nll": self.selection_state.get("r1_reference_nll"),
                "rule": ("lowest NLL + lambda*(mean_rounds-1), subject to pinned R1 NLL + 0.005; ties within "
                         if "lambda" in self.selection_policy else "legacy lowest greedy_nll on selection; ties within ") +
                f"{self.tie_tolerance} prefer fewer rounds then earlier update",
                "best": self.selection_state["best"],
                "candidates": self.selection_state["candidates"],
            },
        )
        summary = {
            "selection_status": "selected" if self.selection_state["best"] else "no_feasible_candidate",
            "status": status,
            "run_id": self.run_id,
            "initialization": self.initialization,
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
        (self.run_dir / "final_summary.md").write_text(self._summary_markdown(summary))
        return summary

    def _summary_markdown(self, summary: dict) -> str:
        best = summary["best"] or {}
        lines = [
            "# RL-LM pair run summary",
            "",
            f"- run_id: `{summary['run_id']}`",
            f"- status: **{summary['status']}**",
            f"- stage: {self.stage}, initialization: {summary['initialization']}",
            f"- successful updates: {summary['successful_updates']} / planned {summary['planned_updates']}"
            f" (stop-after cap {summary['stop_after']})",
            f"- LR scale: {summary['lr_scale']}, cumulative KL backoffs: {summary['lr_backoffs']}",
            f"- selection status: {summary.get('selection_status', 'selected' if best else 'no_feasible_candidate')}",
            f"- best: update {best.get('update')}, greedy_nll {best.get('greedy_nll')},"
            f" mean_rounds {best.get('mean_rounds')}",
            "",
            "## 已实现",
            "- on-policy RLOO 采样→replay→单次受监控 update;原子 checkpoint 与精确恢复。",
            "",
            "## 已验证",
            "- CPU toy 契约测试见 tests/rl_lm/test_resume.py (T19) 与 test_update_transaction.py (T20)。",
            "",
            "## 尚未运行",
            "- 真实 100M checkpoint 上的 GPU 训练(任务卡 E)不在本交付内。",
            "",
            "## 未通过",
            "- 无(若本 run 以 failure 结束,见 failure.json / update_rejected.json)。",
            "",
        ]
        return "\n".join(lines)

    def _record_failure(self, exc: Exception) -> None:
        latest = self.run_dir / "checkpoint_latest.pt"
        atomic_json(
            self.run_dir / "failure.json",
            {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "successful_updates_completed": self.successful_updates,
                "last_complete_checkpoint": str(latest) if latest.exists() else None,
            },
        )
        try:
            (self.run_dir / "final_summary.md").write_text(
                self._summary_markdown(
                    {
                        "status": f"failed: {type(exc).__name__}",
                        "run_id": self.run_id,
                        "initialization": self.initialization,
                        "successful_updates": self.successful_updates,
                        "planned_updates": self.planned_updates,
                        "stop_after": self.stop_after,
                        "lr_scale": self.lr_scale,
                        "lr_backoffs": self.lr_backoffs,
                        "best": self.selection_state.get("best"),
                        "ledger": self.ledger.to_dict(),
                        "frozen_lm_hash_stable": None,
                        "run_dir": str(self.run_dir),
                    }
                )
            )
        except Exception:
            pass


def train(config, model, **kwargs) -> dict:
    """Build a Trainer and run it to the stop-after cap or planned budget."""
    return Trainer(config, model, **kwargs).run()
