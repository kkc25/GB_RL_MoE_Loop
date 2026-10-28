"""Bounded, deterministic training-only learning-rate and stopping policy.

The caller supplies the SAME fixed full-training-population probe at every
observation. There is deliberately no model, optimizer, RNG, minibatch-loss,
selection, or held-out-data dependency here. The caller retains its hard update
ceiling and best-train-checkpoint logic; ``stop`` is an optimization diagnostic,
not evidence that a method is unlearnable or a generalization verdict.

``ineffective_update`` is an adapter-only witness: actual native-BF16 post-add
change is zero despite nonzero finite gradients since the previous probe. A
small residual norm or rounded residual alone is NOT this witness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
import math


SCHEMA = 'mixtral-train-only-adaptive-fit-v1'
MAX_UPDATES = 10000


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f'{name} must be an integer in [{lower},{upper}]')


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be finite numeric')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{name} must be finite numeric') from exc
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite numeric')
    return result


@dataclass(frozen=True)
class AdaptiveFitConfig:
    enabled: bool = False
    eval_interval: int = 16
    patience: int = 3
    min_delta: float = 1e-4
    lr_factor: float = .5
    min_lr: float = 1e-5
    max_lr: float = .02
    growth_factor: float = 2.
    max_growths: int = 2
    min_updates: int = 64
    stop_patience: int = 3

    @classmethod
    def from_dict(cls, value):
        """An absent config keeps historical fixed-rate/fixed-budget behavior."""
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError('adaptive must be an object or None')
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError('unknown adaptive configuration fields: ' +
                             ', '.join(sorted(map(str, unknown))))
        config = cls(**value)
        config.validate()
        return config

    def validate(self):
        if type(self.enabled) is not bool:
            raise ValueError('adaptive.enabled must be boolean')
        for name in ('eval_interval', 'patience', 'stop_patience'):
            _integer(getattr(self, name), 'adaptive.' + name, 1, MAX_UPDATES)
        _integer(self.min_updates, 'adaptive.min_updates', 0, MAX_UPDATES)
        _integer(self.max_growths, 'adaptive.max_growths', 0, 100)
        for name in ('min_delta', 'lr_factor', 'min_lr', 'max_lr', 'growth_factor'):
            _finite(getattr(self, name), 'adaptive.' + name)
        if not 0 <= self.min_delta <= 100:
            raise ValueError('adaptive.min_delta must be in [0,100]')
        if not 0 < self.lr_factor < 1:
            raise ValueError('adaptive.lr_factor must be in (0,1)')
        if not 0 < self.min_lr <= self.max_lr <= 1:
            raise ValueError('adaptive requires 0 < min_lr <= max_lr <= 1')
        if not 1 < self.growth_factor <= 100:
            raise ValueError('adaptive.growth_factor must be in (1,100]')


class AdaptiveFitController:
    """One independent instance per module/stage; no global state is modified.

    Patience counts fixed-probe observations, not optimizer updates. The first
    observation (normally update zero) establishes a baseline, never an action.
    Consecutive no-progress observations reduce LR; bounded BF16 ineffectiveness
    recovery takes precedence over reductions. Growth additionally requires no
    worsening from the previous/best loss AND no new meaningful progress.

    Stopping needs ``stop_patience`` consecutive non-improving observations made
    *at* the minimum LR, after both coverage and ``min_updates`` are satisfied.
    Arriving at the LR floor does not count as having tested that new LR. Missing
    coverage/early observations cannot consume stopping patience. ``eval_interval``
    is owned/enforced by the trainer, which can also observe its final update.
    """

    def __init__(self, config=None, *, initial_lr):
        if not isinstance(config, AdaptiveFitConfig):
            config = AdaptiveFitConfig.from_dict(config)
        config.validate()
        initial_lr = _finite(initial_lr, 'initial_lr')
        if not 0 < initial_lr <= 1:
            raise ValueError('initial_lr must be in (0,1]')
        if config.enabled and not config.min_lr <= initial_lr <= config.max_lr:
            raise ValueError('initial_lr must lie within adaptive min_lr/max_lr')
        self.config = config
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.last_update = None
        self.best_train_loss = None
        self.progress_anchor_loss = None
        self.last_train_loss = None
        self.plateau_observations = 0
        self.ineffective_observations = 0
        self.floor_plateau_observations = 0
        self.growths = 0
        self.reductions = 0
        self.coverage_complete = False
        self.stopped = False
        self.stop_reason = None
        self.decisions = []

    def _state(self):
        return {
            'current_lr': self.lr,
            'last_update': self.last_update,
            'best_train_loss': self.best_train_loss,
            'progress_anchor_loss': self.progress_anchor_loss,
            'last_train_loss': self.last_train_loss,
            'plateau_observations': self.plateau_observations,
            'ineffective_observations': self.ineffective_observations,
            'floor_plateau_observations': self.floor_plateau_observations,
            'growths': self.growths,
            'reductions': self.reductions,
            'coverage_complete': self.coverage_complete,
            'stopped': self.stopped,
            'stop_reason': self.stop_reason,
            'observation_count': len(self.decisions),
        }

    def snapshot(self):
        """Complete, detached, finite JSON audit state (at most 10001 events)."""
        return {'schema': SCHEMA, 'config': asdict(self.config),
                'initial_lr': self.initial_lr, **self._state(),
                'decisions': copy.deepcopy(self.decisions)}

    @classmethod
    def from_snapshot(cls, snapshot):
        """Restore by deterministic replay; reject incomplete/tampered state.

        This restores the policy only, not optimizer/model/RNG state or a trainer
        run. It makes saved controller counters independently auditable.
        """
        if not isinstance(snapshot, dict) or snapshot.get('schema') != SCHEMA:
            raise ValueError('invalid adaptive snapshot schema')
        events = snapshot.get('decisions')
        if not isinstance(events, list) or len(events) > MAX_UPDATES + 1:
            raise ValueError('invalid adaptive snapshot decisions')
        try:
            controller = cls(AdaptiveFitConfig.from_dict(snapshot['config']),
                             initial_lr=snapshot['initial_lr'])
            for event in events:
                if not isinstance(event, dict):
                    raise ValueError('invalid adaptive snapshot event')
                controller.observe(event['update'], event['train_loss'],
                                   event['coverage_complete'], event['ineffective_update'])
            if (json.dumps(controller.snapshot(), sort_keys=True, allow_nan=False) !=
                    json.dumps(snapshot, sort_keys=True, allow_nan=False)):
                raise ValueError('adaptive snapshot does not match deterministic replay')
        except (KeyError, TypeError) as exc:
            raise ValueError('incomplete adaptive snapshot') from exc
        return controller

    def observe(self, update, train_loss, coverage_complete, ineffective_update=False):
        """Return ``action/lr/stop/reason/state`` using training feedback only.

        Invalid/nonfinite feedback raises before ANY state mutation. A returned
        LR applies to the next optimizer update. ``state`` omits history to avoid
        recursive/quadratic logs; ``snapshot()`` contains the entire history.
        """
        _integer(update, 'update', 0, MAX_UPDATES)
        loss = _finite(train_loss, 'train_loss')
        if type(coverage_complete) is not bool or type(ineffective_update) is not bool:
            raise ValueError('coverage_complete and ineffective_update must be boolean')
        if self.stopped:
            raise ValueError('adaptive controller already stopped')
        if self.last_update is not None and update <= self.last_update:
            raise ValueError('adaptive observations require strictly increasing updates')

        baseline = self.last_train_loss is None
        improved = baseline or loss < self.progress_anchor_loss - self.config.min_delta
        nonworsening = (not baseline and loss <= self.last_train_loss
                       and loss <= self.best_train_loss)
        action, reason = 'hold', 'baseline' if baseline else 'awaiting_plateau_patience'
        if improved:
            self.progress_anchor_loss = loss
            self.plateau_observations = 0
            self.ineffective_observations = 0
            self.floor_plateau_observations = 0
            if not baseline:
                reason = 'train_loss_improved'
        elif self.config.enabled:
            self.plateau_observations += 1
            if ineffective_update and nonworsening:
                self.ineffective_observations += 1
            else:
                self.ineffective_observations = 0

            if (self.ineffective_observations >= self.config.patience
                    and self.growths < self.config.max_growths
                    and self.lr < self.config.max_lr):
                self.lr = min(self.config.max_lr, self.lr * self.config.growth_factor)
                self.growths += 1
                self.plateau_observations = 0
                self.ineffective_observations = 0
                self.floor_plateau_observations = 0
                action, reason = 'grow_lr', 'ineffective_native_bf16_update'
            elif self.lr <= self.config.min_lr:
                if update < self.config.min_updates:
                    self.floor_plateau_observations = 0
                    reason = 'minimum_updates_not_reached'
                elif not coverage_complete:
                    self.floor_plateau_observations = 0
                    reason = 'training_coverage_incomplete'
                else:
                    self.floor_plateau_observations += 1
                    reason = 'awaiting_min_lr_stop_patience'
                    if self.floor_plateau_observations >= self.config.stop_patience:
                        if (self.ineffective_observations > 0
                                and self.growths < self.config.max_growths
                                and self.lr < self.config.max_lr):
                            # A shorter stopping patience must not preempt a
                            # still-available, evidenced BF16 recovery attempt.
                            reason = 'awaiting_native_bf16_recovery_patience'
                        else:
                            action, reason = 'stop', 'train_plateau_at_min_lr'
                            self.stopped = True
                            self.stop_reason = reason
            elif self.plateau_observations >= self.config.patience:
                self.lr = max(self.config.min_lr, self.lr * self.config.lr_factor)
                self.reductions += 1
                self.plateau_observations = 0
                self.ineffective_observations = 0
                self.floor_plateau_observations = 0
                action, reason = 'reduce_lr', 'train_loss_plateau'

        if not self.config.enabled:
            action, reason = 'hold', 'adaptive_disabled'
        self.last_update = update
        self.last_train_loss = loss
        self.best_train_loss = loss if self.best_train_loss is None else min(loss, self.best_train_loss)
        self.coverage_complete = coverage_complete
        self.decisions.append({'update': update, 'train_loss': loss,
                               'coverage_complete': coverage_complete,
                               'ineffective_update': ineffective_update,
                               'action': action, 'lr': self.lr,
                               'stop': self.stopped, 'reason': reason})
        return {'action': action, 'lr': self.lr, 'stop': self.stopped,
                'reason': reason, 'state': self._state()}
