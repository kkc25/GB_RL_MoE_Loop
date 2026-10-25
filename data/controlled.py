"""Controlled non-commutative finite-state control environment (plan sec. 7/8.1/8.2, P0b).

A small environment with known transitions, built to validate the controller
algorithms (exact expected objective, RLOO G=2, halt semantics) on a problem
where the optimal policy is enumerable and non-myopic by construction.

State and operators:
    State z in Z_97. Each of the E = 4 experts is a fixed affine operator
    f_e(z) = (a_e * z + b_e) mod 97; every ordered pair of operators is
    non-commuting, so action order matters. An instance is (z0, target,
    budget); the terminal loss is the squared circular distance to the target,
    (d/48)^2 in [0, 1], and the objective is the plan's positive-loss
    convention F(tau) = loss + lambda * calls (sec. 4.1, 5.2).

STOP / budget semantics — identical to the main project (sec. 3.4, 4.2, P0b):
    * At least one call: the first decision is a forced CONTINUE, only the
      expert choice contributes to the log-probability.
    * Budget B in {1, 2, 3} and max depth 3; the per-episode limit is
      min(B, MAX_CALLS). When the limit is reached the exit is forced, carries
      probability 1, and contributes no score.
    * Otherwise the policy is factorized: p_stop = sigmoid(z_stop),
      p_e = softmax(expert logits); CONTINUE with expert e has log-probability
      log(1 - p_stop) + log p_e, STOP has log p_stop.
    Terminal expert paths for B = 3 number 4 + 16 + 64 = 84, exactly the P0b
    tree, so every instance has an enumerable optimal path. Some instances are
    1-step optimal, some 2-step optimal; the 2-step ones contain complementary
    actions: the first call moves the state *away* from the target (worse than
    the greedy one-step call), and only the composition reaches it — the
    sec. 4.3 myopic counterexample realized as data.

Observation legality (sec. 3.3): observations contain only quantities
available at deployment — current state, the task input (z0, target, budget),
round/budget/price scalars, and the previous expert one-hot (all-zero before
the first call). They never contain the terminal loss value, the optimal
action or path, or any future state. The loss is a deterministic function of
visible state, but its value is not a feature.

All exact math here is NumPy FP64; no GPU or LLM is involved. The intended
trainer usage:

    env = ControlledEnvironment()
    state = env.reset(index)
    while not state.done:
        obs = env.observe(state)               # FP64 feature vector
        mask = env.legal_decisions(state)      # STOP/expert legality
        state = env.step(state, decision)      # validates legality
    objective = env.objective(state)           # loss + price * calls

and for exact/oracle baselines:

    env.enumerate_objectives(index)            # all terminal paths, F values
    env.optimal(index)                         # enumerable optimal path
    enumerate_tree(instance, weights)          # leaves with prob/score
    exact_objective(tree), exact_gradient(tree), rloo2_expected_gradient(tree)
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np

MODULUS = 97
DIAMETER = 48  # max circular distance on Z_97; normalizes the loss to [0, 1]
NUM_EXPERTS = 4  # E = 4 (P0b actor expert head)
MAX_CALLS = 3  # max depth 3 (P0b)
BUDGETS = (1, 2, 3)  # per-episode budget B (P0b)
DEFAULT_BUDGET = 3
DEFAULT_PRICE = 0.05  # lambda, in loss units per call

STOP = -1  # decision id for STOP; expert decisions are 0..NUM_EXPERTS-1

# f_e(z) = (a z + b) mod 97. Every ordered pair is non-commuting (checked in
# tests): translations commute only with translations, scalings only with
# scalings, and (2, 1) vs (3, 5) fails b*(c-1) == d*(a-1).
OPERATORS: Tuple[Tuple[int, int], ...] = (
    (1, 1),  # e0: z -> z + 1
    (2, 0),  # e1: z -> 2z
    (2, 1),  # e2: z -> 2z + 1
    (3, 5),  # e3: z -> 3z + 5
)

# Observation layout (OBSERVATION_DIM = 12), all deployment-legal quantities:
#   0..2   circular encoding of current z, initial z0, target, each in [-1, 1]
#   3      round t / MAX_CALLS
#   4      remaining budget (budget - t) / MAX_CALLS
#   5      requested budget / MAX_CALLS
#   6      price lambda
#   7..10  previous expert one-hot (all zeros before the first call)
#   11     bias 1.0
OBSERVATION_DIM = 12


@dataclass(frozen=True)
class Instance:
    """One control problem: drive z0 to target in at most `budget` calls."""

    name: str
    z0: int
    target: int
    budget: int = DEFAULT_BUDGET

    def __post_init__(self):
        if not (0 <= self.z0 < MODULUS and 0 <= self.target < MODULUS):
            raise ValueError(f"state out of range: {self}")
        if self.budget not in BUDGETS:
            raise ValueError(f"budget must be in {BUDGETS}: {self.budget}")


# Fixed instance set. Optimality claims below are verified by full enumeration
# in tests/test_controlled.py; the paths are the *unique* optima at
# price = DEFAULT_PRICE:
#   one_step_exact:            optimal (0,), loss 0
#   one_step_near:             optimal (0,), loss 1/2304 — stopping early beats
#                              spending a second call to hit the target exactly
#   two_step_complementary:    optimal (1, 3); after e1 the distance grows
#                              42 -> 45 (greedy one-step: 30), then e3 hits
#   two_step_complementary_2:  optimal (3, 2); distance 31 -> 39 -> 0
#   budget_one / budget_two:   same task as two_step_complementary with B = 1/2,
#                              exercising forced exits and the truncated tree
DEFAULT_INSTANCES: Tuple[Instance, ...] = (
    Instance("one_step_exact", 10, 11, 3),
    Instance("one_step_near", 10, 12, 3),
    Instance("two_step_complementary", 10, 65, 3),
    Instance("two_step_complementary_2", 11, 77, 3),
    Instance("budget_one", 10, 65, 1),
    Instance("budget_two", 10, 65, 2),
)


# ---------------------------------------------------------------------------
# Environment dynamics (pure functions, exact integer arithmetic)
# ---------------------------------------------------------------------------


def apply_operator(expert: int, z: int) -> int:
    """f_e(z) = (a_e z + b_e) mod 97."""
    if not 0 <= expert < NUM_EXPERTS:
        raise ValueError(f"unknown expert: {expert}")
    a, b = OPERATORS[expert]
    return (a * z + b) % MODULUS


def circular_distance(u: int, v: int) -> int:
    """min((u - v) mod 97, (v - u) mod 97), in {0, ..., 48}."""
    diff = abs(u - v) % MODULUS
    return min(diff, MODULUS - diff)


def terminal_loss(z: int, target: int) -> float:
    """Squared normalized circular distance, in [0, 1]."""
    return (circular_distance(z, target) / DIAMETER) ** 2


def run_path(z0: int, path: Sequence[int]) -> Tuple[int, ...]:
    """States visited by an expert path, including z0: (z0, z1, ..., z_T)."""
    states = [z0]
    z = z0
    for e in path:
        z = apply_operator(e, z)
        states.append(z)
    return tuple(states)


def path_limit(budget: int) -> int:
    """Per-episode call cap: min(B, MAX_CALLS)."""
    if budget not in BUDGETS:
        raise ValueError(f"budget must be in {BUDGETS}: {budget}")
    return min(budget, MAX_CALLS)


def all_paths(limit: int = MAX_CALLS) -> List[Tuple[int, ...]]:
    """All terminal expert paths: lengths 1..limit (4 + 16 + 64 = 84 at 3).

    Same convention as eval.oracle.all_paths: a path is the tuple of expert
    calls; STOP is implicit at its end.
    """
    if not 1 <= limit <= MAX_CALLS:
        raise ValueError(f"limit must be in 1..{MAX_CALLS}: {limit}")
    return [
        path
        for depth in range(1, limit + 1)
        for path in itertools.product(range(NUM_EXPERTS), repeat=depth)
    ]


def path_objective(instance: Instance, path: Sequence[int], price: float = DEFAULT_PRICE) -> float:
    """F(tau) = terminal loss + price * calls (plan sec. 4.1, positive loss)."""
    if not 1 <= len(path) <= path_limit(instance.budget):
        raise ValueError(f"path {path} violates the budget of {instance}")
    z_end = run_path(instance.z0, path)[-1]
    return terminal_loss(z_end, instance.target) + price * len(path)


class PathEvaluation(NamedTuple):
    path: Tuple[int, ...]
    loss: float
    cost: float
    objective: float


def enumerate_objectives(instance: Instance, price: float = DEFAULT_PRICE) -> List[PathEvaluation]:
    """Objective of every terminal path of the instance, in all_paths order."""
    out = []
    for path in all_paths(path_limit(instance.budget)):
        loss = terminal_loss(run_path(instance.z0, path)[-1], instance.target)
        cost = float(len(path))
        out.append(PathEvaluation(path, loss, cost, loss + price * cost))
    return out


def optimal_path(instance: Instance, price: float = DEFAULT_PRICE) -> PathEvaluation:
    """The enumerable optimal path (first in all_paths order on ties)."""
    return min(enumerate_objectives(instance, price), key=lambda ev: (ev.objective, ev.path))


# ---------------------------------------------------------------------------
# Rollout interface for trainers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvState:
    instance: Instance
    z: int
    t: int
    previous_expert: int  # -1 before the first (forced) call
    done: bool


class DecisionMask(NamedTuple):
    """Legality of the next decision, mirroring sec. 3.4/4.2 semantics."""

    stop_allowed: bool  # a stochastic STOP may be sampled
    stop_forced: bool  # exit is forced: probability 1, no score
    continue_forced: bool  # first call: CONTINUE is forced, STOP illegal
    experts: Tuple[int, ...]  # callable experts (all of them while t < limit)


def _circular_encoding(u: int) -> float:
    """Wraparound-continuous encoding of a residue into [-1, 1]."""
    return (((u + DIAMETER) % MODULUS) - DIAMETER) / DIAMETER


class ControlledEnvironment:
    """Rollout + enumeration API for the default instance set.

    Deliberately NumPy/FP64 and model-free: a trainer supplies its own policy
    (e.g. the linear factorized policy below, or a torch module reading
    `observe`), consumes `legal_decisions` masks, and scores episodes with
    `objective`. Enumeration methods support the exact-objective and oracle
    baselines of P0b.
    """

    def __init__(
        self,
        instances: Sequence[Instance] = DEFAULT_INSTANCES,
        price: float = DEFAULT_PRICE,
    ):
        if not instances:
            raise ValueError("at least one instance is required")
        self.instances: Tuple[Instance, ...] = tuple(instances)
        self.price = float(price)
        self.num_experts = NUM_EXPERTS
        self.max_calls = MAX_CALLS

    def __len__(self) -> int:
        return len(self.instances)

    def reset(self, index: int) -> EnvState:
        instance = self.instances[index]
        return EnvState(instance, instance.z0, 0, -1, False)

    def limit(self, instance: Instance) -> int:
        return path_limit(instance.budget)

    def legal_decisions(self, state: EnvState) -> DecisionMask:
        if state.done:
            raise ValueError("episode is already terminal")
        limit = self.limit(state.instance)
        if state.t == 0:
            # At least one call: forced CONTINUE (plan sec. 3.4).
            return DecisionMask(False, False, True, tuple(range(NUM_EXPERTS)))
        if state.t == limit:
            # Budget/round limit: forced exit, probability 1, no score.
            return DecisionMask(True, True, False, ())
        return DecisionMask(True, False, False, tuple(range(NUM_EXPERTS)))

    def observe(self, state: EnvState) -> np.ndarray:
        """FP64 observation vector; see the OBSERVATION_DIM layout above."""
        instance = state.instance
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float64)
        obs[0] = _circular_encoding(state.z)
        obs[1] = _circular_encoding(instance.z0)
        obs[2] = _circular_encoding(instance.target)
        obs[3] = state.t / MAX_CALLS
        obs[4] = (instance.budget - state.t) / MAX_CALLS
        obs[5] = instance.budget / MAX_CALLS
        obs[6] = self.price
        if state.previous_expert >= 0:
            obs[7 + state.previous_expert] = 1.0
        obs[11] = 1.0
        return obs

    def step(self, state: EnvState, decision: int) -> EnvState:
        """Apply STOP or an expert call; illegal decisions raise ValueError."""
        mask = self.legal_decisions(state)
        if decision == STOP:
            if not mask.stop_allowed or mask.continue_forced:
                raise ValueError(f"STOP is not legal in state {state}")
            return EnvState(state.instance, state.z, state.t, state.previous_expert, True)
        if not 0 <= decision < NUM_EXPERTS:
            raise ValueError(f"unknown decision: {decision}")
        if mask.stop_forced:
            raise ValueError(f"state {state} is at the call limit; STOP is forced")
        return EnvState(
            state.instance,
            apply_operator(decision, state.z),
            state.t + 1,
            decision,
            state.t + 1 == self.limit(state.instance),
        )

    def terminal_loss(self, state: EnvState) -> float:
        return terminal_loss(state.z, state.instance.target)

    def objective(self, state: EnvState) -> float:
        """F(tau) of a terminal state: loss + price * calls."""
        if not state.done:
            raise ValueError("objective is defined at terminal states")
        return self.terminal_loss(state) + self.price * state.t

    def enumerate_objectives(self, index: int) -> List[PathEvaluation]:
        return enumerate_objectives(self.instances[index], self.price)

    def optimal(self, index: int) -> PathEvaluation:
        return optimal_path(self.instances[index], self.price)


# ---------------------------------------------------------------------------
# Factorized linear policy and exact finite-tree math (NumPy FP64)
#
# Policy: logits = W @ obs with W of shape (NUM_EXPERTS + 1, OBSERVATION_DIM).
# Row 0 is the STOP logit, rows 1..E the expert residual logits. Action
# probabilities follow plan sec. 3.4: p_stop = sigmoid(z_stop),
# p_e = softmax(expert logits); the first call is a forced CONTINUE (only the
# expert softmax contributes) and reaching the call limit is a forced exit
# (probability 1, no score). Score rows below are d log pi / d W.
# ---------------------------------------------------------------------------


def reference_weights() -> np.ndarray:
    """Fixed FP64 policy weights for the exact checks (no randomness)."""
    k, j = np.meshgrid(
        np.arange(NUM_EXPERTS + 1, dtype=np.float64),
        np.arange(OBSERVATION_DIM, dtype=np.float64),
        indexing="ij",
    )
    return 0.2 * np.sin(0.7 * (k + 1.0) * (j + 1.0) + 0.3) + 0.05 * np.cos(0.3 * j - 0.1 * k)


def action_probabilities(weights: np.ndarray, observation: np.ndarray) -> Tuple[float, np.ndarray]:
    """(p_stop, p_experts) from the factorized policy (plan sec. 3.4)."""
    logits = weights @ observation
    stop_logit = float(logits[0])
    p_stop = 1.0 / (1.0 + math.exp(-stop_logit))
    expert_logits = logits[1:] - np.max(logits[1:])
    p_experts = np.exp(expert_logits)
    p_experts /= p_experts.sum()
    return p_stop, p_experts


def decision_log_prob_and_score(
    weights: np.ndarray,
    observation: np.ndarray,
    decision: int,
    forced_continue: bool = False,
) -> Tuple[float, np.ndarray]:
    """log pi(decision) and its gradient wrt W for one stochastic decision.

    forced_continue models the first call (sec. 3.4): only the expert softmax
    contributes. The forced exit at the call limit is never passed here — it
    has probability 1 and an empty score.
    """
    p_stop, p_experts = action_probabilities(weights, observation)
    score = np.zeros_like(weights)
    if forced_continue:
        if decision == STOP:
            raise ValueError("STOP is illegal under a forced CONTINUE")
        onehot = np.zeros(NUM_EXPERTS)
        onehot[decision] = 1.0
        score[1:] = np.outer(onehot - p_experts, observation)
        return float(np.log(p_experts[decision])), score
    if decision == STOP:
        score[0] = (1.0 - p_stop) * observation
        return math.log(p_stop), score
    onehot = np.zeros(NUM_EXPERTS)
    onehot[decision] = 1.0
    score[0] = -p_stop * observation
    score[1:] = np.outer(onehot - p_experts, observation)
    return math.log(1.0 - p_stop) + float(np.log(p_experts[decision])), score


class Leaf(NamedTuple):
    """A terminal trajectory of the exact tree."""

    probability: float
    loss: float
    cost: float
    objective: float  # F(tau) = loss + price * cost
    score: np.ndarray  # sum of d log pi / d W over the stochastic decisions
    path: Tuple[int, ...]  # expert calls; STOP is implicit at the end


def enumerate_tree(
    instance: Instance,
    weights: np.ndarray,
    price: float = DEFAULT_PRICE,
) -> List[Leaf]:
    """All terminal trajectories with probabilities and score-function terms.

    Same enumeration contract as eval/verify_router_math.py but with the
    project's factorized STOP/expert policy and per-instance budget.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (NUM_EXPERTS + 1, OBSERVATION_DIM):
        raise ValueError(f"weights must have shape {(NUM_EXPERTS + 1, OBSERVATION_DIM)}")
    limit = path_limit(instance.budget)
    env = ControlledEnvironment((instance,), price)
    leaves: List[Leaf] = []

    def visit(z: int, t: int, previous: int, prob: float, score: np.ndarray, path: Tuple[int, ...]):
        if t == limit:  # forced exit: probability 1, no score
            loss = terminal_loss(z, instance.target)
            leaves.append(Leaf(prob, loss, float(t), loss + price * t, score, path))
            return
        state = EnvState(instance, z, t, previous, False)
        observation = env.observe(state)
        if t >= 1:  # stochastic STOP is legal after at least one call
            logp, sc = decision_log_prob_and_score(weights, observation, STOP)
            loss = terminal_loss(z, instance.target)
            leaves.append(Leaf(prob * math.exp(logp), loss, float(t), loss + price * t, score + sc, path))
        for e in range(NUM_EXPERTS):
            logp, sc = decision_log_prob_and_score(weights, observation, e, forced_continue=(t == 0))
            visit(apply_operator(e, z), t + 1, e, prob * math.exp(logp), score + sc, path + (e,))

    visit(instance.z0, 0, -1, 1.0, np.zeros_like(weights), ())
    return leaves


def exact_objective(tree: Sequence[Leaf]) -> float:
    """E[F] over the enumerated tree."""
    return float(sum(leaf.probability * leaf.objective for leaf in tree))


def exact_gradient(tree: Sequence[Leaf]) -> np.ndarray:
    """Score-function gradient of E[F] (plan eq. 9 with b = 0)."""
    grad = sum(leaf.probability * leaf.objective * leaf.score for leaf in tree)
    return np.asarray(grad, dtype=np.float64)


def mean_score(tree: Sequence[Leaf]) -> np.ndarray:
    """E[sum of scores]; identically zero for a normalized tree."""
    return np.asarray(sum(leaf.probability * leaf.score for leaf in tree), dtype=np.float64)


def rloo2_expected_gradient(tree: Sequence[Leaf]) -> np.ndarray:
    """Exact expectation of the G=2 RLOO estimator (plan eq. 11).

    Sums over all ordered pairs of independent trajectories — an exact
    expectation, not Monte Carlo:
        1/2 * (F1 - F2) * (score1 - score2).
    """
    grad = np.zeros_like(tree[0].score)
    for a in tree:
        for b in tree:
            grad += a.probability * b.probability * 0.5 * (a.objective - b.objective) * (a.score - b.score)
    return grad


def self_mean_gradient(tree: Sequence[Leaf]) -> np.ndarray:
    """G=2 estimator whose baseline includes the sample itself; equals grad/2."""
    grad = np.zeros_like(tree[0].score)
    for a in tree:
        for b in tree:
            mean = 0.5 * (a.objective + b.objective)
            grad += a.probability * b.probability * 0.5 * (
                (a.objective - mean) * a.score + (b.objective - mean) * b.score
            )
    return grad


def finite_difference_gradient(
    instance: Instance,
    weights: np.ndarray,
    price: float = DEFAULT_PRICE,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central finite differences of the exact expected objective in W."""
    fd = np.zeros_like(weights)
    for index in np.ndindex(weights.shape):
        plus = weights.copy()
        minus = weights.copy()
        plus[index] += eps
        minus[index] -= eps
        fd[index] = (
            exact_objective(enumerate_tree(instance, plus, price))
            - exact_objective(enumerate_tree(instance, minus, price))
        ) / (2.0 * eps)
    return fd


def batch_exact_objective_and_gradient(
    env: ControlledEnvironment,
    weights: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Mean exact objective and its gradient over the environment's instances.

    This is the P0b exact-training baseline: exhaust every path of every
    training input instead of sampling (plan sec. 8.2).
    """
    trees = [enumerate_tree(instance, weights, env.price) for instance in env.instances]
    objective = sum(exact_objective(tree) for tree in trees) / len(trees)
    gradient = sum(exact_gradient(tree) for tree in trees) / len(trees)
    return objective, np.asarray(gradient, dtype=np.float64)
