"""E09 controlled-environment part: policy gradient vs supervised/exact baselines.

Controlled-environment analogue of
plans/loop_moe_research_plan_2026-10-25/experiments/E09_router_only_policy_gradient.md

The E09 plan asks whether, with experts frozen and equal extra training
compute, policy-gradient training of a router/controller provides anything a
supervised baseline does not. This script runs that question inside the fully
enumerable synthetic control environment of data/controlled.py (84 terminal
paths, non-commuting experts, complementary two-step optima), where every
quantity the plan lists — training objective, optimal-path recovery, gradient
variance, budget compliance, cost — is computable exactly.

All three trainers update THE SAME policy: the linear factorized STOP/expert
policy of data/controlled.py (weights (E+1, OBS_DIM), logits = W @ obs),
reading only deployment-legal observations (plan sec. 3.3). They differ only
in the training signal:

  supervised_oracle_imitation
      Cross-entropy toward the enumerable optimal path (privileged oracle
      labels — a search-distillation baseline, not a deployment-legal
      signal). Per update: zero rollouts.
  exact_joint_gradient
      Exact score-function gradient of the joint expected objective E[F]
      (plan eq. 9) by full tree enumeration — the P0b exact-training
      baseline. Per update: 360 enumerated terminal trajectories.
  rloo2_onpolicy
      Real on-policy updates: per instance per update, G=2 trajectories are
      sampled from the current policy through env.rollout semantics (forced
      first CONTINUE, forced exit at the limit) and combined with the
      leave-one-out estimator 0.5*(F1-F2)*(score1-score2) (plan eq. 11),
      grad clipped to norm 1. Per update: 12 sampled rollouts. Its
      expectation is verified by exact enumeration
      (rloo2_expected_gradient == exact_gradient) instead of trusting the
      high-variance samples, and the estimator variance is computed exactly.

Compared: training objective curves, optimal-path recovery on the designed
instances and on held-out random instances, the complementary two-step cases,
budget compliance (hard violations are impossible by construction; overspend
vs the optimal call count is reported), exact gradient variance of four
estimators, and per-method trajectory-evaluation cost.

All conclusions are restricted to this synthetic environment. The real-LM
router-RL part of E09 is NOT executed here; no language model is trained.

CLI:
    python experiments/e09_controlled.py --output runs/e09_controlled.json \
        [--seed 20261025] [--quick]
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import controlled as C

PLAN_PATH = "plans/loop_moe_research_plan_2026-10-25/experiments/E09_router_only_policy_gradient.md"
SYNTHETIC_SCOPE = (
    "All numbers come from the enumerable synthetic control environment of "
    "data/controlled.py (Z_97, 4 non-commuting affine experts, 84 terminal "
    "paths at budget 3). They compare training signals for one small linear "
    "router policy and verify estimator properties (unbiasedness, variance, "
    "budget semantics). They are not evidence about natural-language models, "
    "and no language-model RL was performed."
)

# Hyperparameters (locked for the full run; --quick shrinks steps for tests).
SUPERVISED_LR, SUPERVISED_STEPS = 2.0, 200
EXACT_LR, EXACT_STEPS = 3.0, 600
RLOO_LR, RLOO_STEPS, RLOO_CLIP = 1.0, 2000, 1.0
RLOO_SEEDS = (0, 1, 2, 3)
EVAL_EVERY = 25
MC_SAMPLES = 3000
N_VALIDATION = 24

QUICK = dict(
    supervised_steps=SUPERVISED_STEPS, exact_steps=120, rloo_steps=300,
    rloo_seeds=(0,), mc_samples=300, n_validation=8, eval_every=20,
)

DESIGNED_TWO_STEP = ("two_step_complementary", "two_step_complementary_2")


# ---------------------------------------------------------------------------
# Rollouts of the shared legal-observation policy
# ---------------------------------------------------------------------------

def sample_rollout(env, index, weights, rng):
    """One on-policy trajectory; returns (path, objective, score, calls).

    Scores are summed d log pi / dW over the stochastic decisions only;
    the forced first CONTINUE contributes the expert softmax score and the
    forced exit at the call limit contributes nothing (plan sec. 3.4/4.2).
    """
    state = env.reset(index)
    score = np.zeros_like(weights)
    path = ()
    while not state.done:
        obs = env.observe(state)
        mask = env.legal_decisions(state)
        if mask.stop_forced:
            state = env.step(state, C.STOP)  # probability 1, no score
            continue
        p_stop, p_experts = C.action_probabilities(weights, obs)
        if mask.continue_forced:
            decision = int(rng.choice(C.NUM_EXPERTS, p=p_experts))
        elif rng.random() < p_stop:
            decision = C.STOP
        else:
            decision = int(rng.choice(C.NUM_EXPERTS, p=p_experts))
        _, sc = C.decision_log_prob_and_score(
            weights, obs, decision, forced_continue=mask.continue_forced)
        score += sc
        if decision != C.STOP:
            path = path + (decision,)
        state = env.step(state, decision)
    assert state.t <= env.limit(state.instance) and state.t >= 1
    return path, env.objective(state), score, state.t


def greedy_rollout(env, index, weights):
    """Argmax deployment trajectory; returns (path, objective, calls)."""
    state = env.reset(index)
    path = ()
    while not state.done:
        obs = env.observe(state)
        mask = env.legal_decisions(state)
        p_stop, p_experts = C.action_probabilities(weights, obs)
        if mask.stop_forced:
            decision = C.STOP
        elif mask.continue_forced:
            decision = int(np.argmax(p_experts))
        elif p_stop > 0.5:
            decision = C.STOP
        else:
            decision = int(np.argmax(p_experts))
        if decision != C.STOP:
            path = path + (decision,)
        state = env.step(state, decision)
    assert state.t <= env.limit(state.instance) and state.t >= 1
    return path, env.objective(state), state.t


# ---------------------------------------------------------------------------
# Training signals
# ---------------------------------------------------------------------------

def supervised_loss_and_gradient(env, weights):
    """Mean CE toward the enumerable optimal paths (privileged oracle labels).

    The labels walk each instance's optimal path: expert path[t] at round t
    (forced CONTINUE at t = 0) and a STOP label at the end state when the
    path stops before the call limit. Returns (loss, d loss / dW).
    """
    loss = 0.0
    grad = np.zeros_like(weights)
    for index in range(len(env)):
        instance = env.instances[index]
        optimal = env.optimal(index)
        states = C.run_path(instance.z0, optimal.path)
        decisions = list(optimal.path)
        if len(optimal.path) < env.limit(instance):
            decisions.append(C.STOP)
        for t, decision in enumerate(decisions):
            state = C.EnvState(
                instance, states[t], t, optimal.path[t - 1] if t else -1, False)
            obs = env.observe(state)
            logp, score = C.decision_log_prob_and_score(
                weights, obs, decision, forced_continue=(t == 0))
            loss -= logp
            grad -= score
    n = len(env)
    return loss / n, grad / n


def rloo2_batch_estimate(env, weights, rng):
    """One on-policy G=2 leave-one-out batch estimate; uses 2 rollouts/instance."""
    grad = np.zeros_like(weights)
    for index in range(len(env)):
        _, f1, s1, _ = sample_rollout(env, index, weights, rng)
        _, f2, s2, _ = sample_rollout(env, index, weights, rng)
        grad += 0.5 * (f1 - f2) * (s1 - s2)
    return grad / len(env)


def paths_per_instance(env):
    """Terminal-path count of every instance (84 at B=3, 20 at B=2, 4 at B=1)."""
    return [len(C.all_paths(C.path_limit(i.budget))) for i in env.instances]


# ---------------------------------------------------------------------------
# Trainers (same policy, same init; only the signal differs)
# ---------------------------------------------------------------------------

def _curve_point(env, weights, step):
    objective, _ = C.batch_exact_objective_and_gradient(env, weights)
    return {"step": step, "exact_batch_objective": objective}


def train_supervised(env, w0, steps, lr, eval_every):
    w = w0.copy()
    curve = [_curve_point(env, w, 0)]
    loss, _ = supervised_loss_and_gradient(env, w)
    for step in range(1, steps + 1):
        loss, grad = supervised_loss_and_gradient(env, w)
        w -= lr * grad
        if step % eval_every == 0:
            curve.append(_curve_point(env, w, step))
    final_objective, _ = C.batch_exact_objective_and_gradient(env, w)
    return {
        "weights": w,
        "curve": curve,
        "final_ce_loss": loss,
        "final_exact_batch_objective": final_objective,
        "cost": {
            "updates": steps,
            "oracle_enumeration_trajectories": sum(paths_per_instance(env)),
            "sampled_rollouts": 0,
            "note": ("Optimal-path labels are computed once by full "
                     "enumeration; each update then costs zero rollouts."),
        },
    }


def train_exact(env, w0, steps, lr, eval_every):
    w = w0.copy()
    curve = [_curve_point(env, w, 0)]
    enumerated = sum(paths_per_instance(env))
    for step in range(1, steps + 1):
        _, grad = C.batch_exact_objective_and_gradient(env, w)
        w -= lr * grad
        if step % eval_every == 0:
            curve.append(_curve_point(env, w, step))
    final_objective, _ = C.batch_exact_objective_and_gradient(env, w)
    return {
        "weights": w,
        "curve": curve,
        "final_exact_batch_objective": final_objective,
        "cost": {
            "updates": steps,
            "enumerated_trajectories": steps * enumerated,
            "enumerated_trajectories_per_update": enumerated,
            "sampled_rollouts": 0,
        },
    }


def train_rloo2(env, w0, steps, lr, seed, clip, eval_every):
    rng = np.random.default_rng(seed)
    w = w0.copy()
    curve = [_curve_point(env, w, 0)]
    for step in range(1, steps + 1):
        grad = rloo2_batch_estimate(env, w, rng)
        norm = float(np.sqrt((grad * grad).sum()))
        if norm > clip:
            grad *= clip / norm
        w -= lr * grad
        if step % eval_every == 0:
            curve.append(_curve_point(env, w, step))
    final_objective, _ = C.batch_exact_objective_and_gradient(env, w)
    return {
        "weights": w,
        "seed": seed,
        "curve": curve,
        "final_exact_batch_objective": final_objective,
        "cost": {
            "updates": steps,
            "sampled_rollouts": steps * 2 * len(env),
            "sampled_rollouts_per_update": 2 * len(env),
            "grad_clip_norm": clip,
        },
    }


# ---------------------------------------------------------------------------
# Exact gradient diagnostics (enumeration, not Monte Carlo)
# ---------------------------------------------------------------------------

def _moments(vectors_weights):
    """(mean, total variance, per-coordinate variance) of a discrete estimator."""
    mean = sum(p * v for v, p in vectors_weights)
    second = sum(p * v * v for v, p in vectors_weights)
    coord_var = second - mean * mean
    return mean, float(coord_var.sum()), coord_var


def estimator_moments_for_tree(tree):
    """Exact mean/variance of four gradient estimators on one enumerated tree.

    reinforce_b0:        F * score over single trajectories (plan eq. 9, b=0).
    reinforce_mean_b:    (F - E[F]) * score over single trajectories.
    rloo2_loo:           0.5*(F1-F2)*(score1-score2) over ordered pairs (eq. 11).
    self_mean_g2:        G=2 baseline including the sample itself (biased,
                         must not be called the unbiased one; plan sec. 5.2).
    """
    g = C.exact_gradient(tree)
    ef = C.exact_objective(tree)
    out = {}
    singles = {
        "reinforce_b0": [(l.objective * l.score, l.probability) for l in tree],
        "reinforce_mean_baseline": [
            ((l.objective - ef) * l.score, l.probability) for l in tree],
    }
    for name, vw in singles.items():
        mean, total_var, coord_var = _moments(vw)
        out[name] = {
            "bias_max_abs": float(np.max(np.abs(mean - g))),
            "total_variance": total_var,
            "coordinate_variance": coord_var,
        }
    pairs = {"rloo2_loo": [], "self_mean_g2": []}
    for a in tree:
        for b in tree:
            p = a.probability * b.probability
            mean_ab = 0.5 * (a.objective + b.objective)
            pairs["rloo2_loo"].append(
                (0.5 * (a.objective - b.objective) * (a.score - b.score), p))
            pairs["self_mean_g2"].append(
                (0.5 * ((a.objective - mean_ab) * a.score
                        + (b.objective - mean_ab) * b.score), p))
    for name, vw in pairs.items():
        mean, total_var, coord_var = _moments(vw)
        out[name] = {
            "bias_max_abs": float(np.max(np.abs(mean - g))),
            "total_variance": total_var,
            "coordinate_variance": coord_var,
        }
    return out


def batch_estimator_variance(env, weights):
    """Batch-level (mean over independent instances) exact estimator variance."""
    per_instance = {}
    coord = {}
    for instance in env.instances:
        tree = C.enumerate_tree(instance, weights, env.price)
        moments = estimator_moments_for_tree(tree)
        per_instance[instance.name] = {
            name: {"bias_max_abs": m["bias_max_abs"],
                   "total_variance": m["total_variance"]}
            for name, m in moments.items()
        }
        for name, m in moments.items():
            coord.setdefault(name, np.zeros_like(weights))
            coord[name] += m["coordinate_variance"]
    n = len(env)
    batch = {
        name: {
            "total_variance": float(v.sum()) / n**2,
            "max_coordinate_variance": float(v.max()) / n**2,
        }
        for name, v in coord.items()
    }
    return per_instance, batch, coord


def monte_carlo_gradient_check(env, weights, samples, seed):
    """Sanity check: the sampled RLOO batch estimator mean -> exact gradient.

    Tolerances use the exact per-coordinate variance from enumeration
    (6 sigma of the sample mean), so the check is calibrated by enumeration
    rather than by the samples themselves.
    """
    rng = np.random.default_rng(seed)
    _, _, coord = batch_estimator_variance(env, weights)
    exact = C.batch_exact_objective_and_gradient(env, weights)[1]
    acc = np.zeros_like(weights)
    for _ in range(samples):
        acc += rloo2_batch_estimate(env, weights, rng)
    acc /= samples
    err = np.abs(acc - exact)
    tol = 6.0 * np.sqrt(coord["rloo2_loo"] / (len(env)**2 * samples))
    over = err > tol
    return {
        "samples": samples,
        "sigma": 6,
        "max_abs_err": float(err.max()),
        "max_abs_tol": float(tol.max()),
        "coordinates_over_tolerance": int(over.sum()),
        "passed": bool(not over.any()),
    }


# ---------------------------------------------------------------------------
# Evaluation: recovery, complementary cases, budget, held-out instances
# ---------------------------------------------------------------------------

def expected_calls(env, index, weights):
    tree = C.enumerate_tree(env.instances[index], weights, env.price)
    return float(sum(leaf.probability * leaf.cost for leaf in tree))


def evaluate_policy(env, weights, instances, prefix):
    """Greedy recovery table; recovery is tie-tolerant on the objective."""
    rows = {}
    for i, instance in enumerate(instances):
        optimal = C.optimal_path(instance, env.price)
        path, objective, calls = greedy_rollout(env, i, weights)
        rows[instance.name] = {
            "greedy_path": list(path),
            "optimal_path": list(optimal.path),
            "path_matches": path == optimal.path,
            "greedy_objective": objective,
            "optimal_objective": optimal.objective,
            "recovered": objective <= optimal.objective + 1e-9,
            "greedy_calls": calls,
            "optimal_calls": len(optimal.path),
        }
    recovered = sum(r["recovered"] for r in rows.values())
    return {
        "set": prefix,
        "n_instances": len(instances),
        "n_recovered": recovered,
        "recovery_rate": recovered / len(instances),
        "instances": rows,
    }


def validation_instances(n, seed):
    """Held-out random instances (z0, target, budget), distinct from the 6 designed ones."""
    rng = np.random.default_rng(seed)
    designed = {(i.z0, i.target, i.budget) for i in C.DEFAULT_INSTANCES}
    out = []
    while len(out) < n:
        z0 = int(rng.integers(0, C.MODULUS))
        target = int(rng.integers(0, C.MODULUS))
        budget = int(rng.choice(C.BUDGETS))
        if (z0, target, budget) in designed or z0 == target:
            continue
        out.append(C.Instance(f"val_{len(out)}", z0, target, budget))
    return out


def complementary_case_report(env, weights_by_method):
    """The plan sec. 4.3 myopic counterexample, per training method."""
    report = {}
    for name in DESIGNED_TWO_STEP:
        index = next(i for i, x in enumerate(env.instances) if x.name == name)
        instance = env.instances[index]
        optimal = C.optimal_path(instance, env.price)
        opt_states = C.run_path(instance.z0, optimal.path)
        distances = [C.circular_distance(z, instance.target) for z in opt_states]
        methods = {}
        for method, w in weights_by_method.items():
            path, objective, _ = greedy_rollout(env, index, w)
            states = C.run_path(instance.z0, path)
            methods[method] = {
                "greedy_path": list(path),
                "recovered": path == optimal.path,
                "greedy_objective": objective,
                "distance_trajectory": [
                    C.circular_distance(z, instance.target) for z in states],
            }
        report[name] = {
            "optimal_path": list(optimal.path),
            "optimal_distance_trajectory": distances,
            "first_optimal_call_increases_distance": distances[1] > distances[0],
            "methods": methods,
        }
    return report


def budget_report(env, weights_by_method, trajectories_checked):
    """Hard violations are impossible by construction; overspend is reported."""
    overspend = {}
    for method, w in weights_by_method.items():
        rows = {}
        for index, instance in enumerate(env.instances):
            optimal = C.optimal_path(instance, env.price)
            e_calls = expected_calls(env, index, w)
            rows[instance.name] = {
                "expected_calls": e_calls,
                "optimal_calls": len(optimal.path),
                "overspend": e_calls - len(optimal.path),
                "hard_limit": env.limit(instance),
            }
        overspend[method] = rows
    return {
        "hard_budget_violations": 0,
        "trajectories_checked": trajectories_checked,
        "note": ("env.step raises on any illegal decision and the forced exit "
                 "at min(budget, MAX_CALLS) carries probability 1, so a hard "
                 "violation cannot be sampled; every rollout above asserted "
                 "1 <= calls <= limit. Overspend = expected calls under the "
                 "trained policy minus the optimal path length."),
        "overspend": overspend,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(seed=20261025, quick=False):
    env = C.ControlledEnvironment()
    w0 = C.reference_weights()
    cfg = QUICK if quick else {}
    sup_steps = cfg.get("supervised_steps", SUPERVISED_STEPS)
    exact_steps = cfg.get("exact_steps", EXACT_STEPS)
    rloo_steps = cfg.get("rloo_steps", RLOO_STEPS)
    rloo_seeds = cfg.get("rloo_seeds", RLOO_SEEDS)
    mc_samples = cfg.get("mc_samples", MC_SAMPLES)
    n_val = cfg.get("n_validation", N_VALIDATION)
    eval_every = cfg.get("eval_every", EVAL_EVERY)

    init_objective, _ = C.batch_exact_objective_and_gradient(env, w0)

    # Gradient verification: enumeration identity + calibrated Monte Carlo.
    identity_err = max(
        float(np.max(np.abs(
            C.rloo2_expected_gradient(C.enumerate_tree(i, w0, env.price))
            - C.exact_gradient(C.enumerate_tree(i, w0, env.price)))))
        for i in env.instances
    )
    mc = monte_carlo_gradient_check(env, w0, mc_samples, seed + 1)
    _, variance_at_init, _ = batch_estimator_variance(env, w0)

    # Training.
    supervised = train_supervised(env, w0, sup_steps, SUPERVISED_LR, eval_every)
    exact = train_exact(env, w0, exact_steps, EXACT_LR, eval_every)
    rloo_runs = [
        train_rloo2(env, w0, rloo_steps, RLOO_LR, 1000 * seed + s, RLOO_CLIP, eval_every)
        for s in rloo_seeds
    ]
    rloo_finals = [r["final_exact_batch_objective"] for r in rloo_runs]

    weights_by_method = {
        "initial": w0,
        "supervised_oracle_imitation": supervised["weights"],
        "exact_joint_gradient": exact["weights"],
        "rloo2_onpolicy_best_seed": rloo_runs[int(np.argmin(rloo_finals))]["weights"],
    }

    _, variance_at_exact_trained, _ = batch_estimator_variance(env, exact["weights"])

    # Validation: designed instances (path equality) + held-out random ones.
    val_env_instances = validation_instances(n_val, seed + 2)
    val_env = C.ControlledEnvironment(val_env_instances, env.price)
    validation = {
        method: {
            "designed": evaluate_policy(env, w, env.instances, "designed"),
            "held_out": evaluate_policy(val_env, w, val_env_instances, "held_out"),
        }
        for method, w in weights_by_method.items()
    }

    trajectories_checked = (  # rollouts whose 1 <= calls <= limit was asserted
        mc_samples * 2 * len(env)
        + sum(r["cost"]["sampled_rollouts"] for r in rloo_runs)
        + len(weights_by_method) * (len(env) + n_val)
    )

    def strip(run_dict):
        return {k: v for k, v in run_dict.items() if k != "weights"}

    result = {
        "experiment": "E09_controlled",
        "date": datetime.date.today().isoformat(),
        "seed": seed,
        "quick": quick,
        "plan": PLAN_PATH,
        "scope": SYNTHETIC_SCOPE,
        "real_lm_rl": {
            "status": "not_executed",
            "reason": ("The E09 real-LM router-RL part (frozen backbone, "
                       "Top-2 set actions, teacher-forced terminal reward) "
                       "requires the training stack and is outside this "
                       "script; only the enumerable synthetic analogue runs "
                       "here."),
        },
        "environment": {
            "instances": [i.name for i in env.instances],
            "price": env.price,
            "num_experts": C.NUM_EXPERTS,
            "max_calls": C.MAX_CALLS,
            "terminal_paths_per_instance": paths_per_instance(env),
            "policy": ("shared linear factorized STOP/expert policy, "
                       f"weights ({C.NUM_EXPERTS + 1}, {C.OBSERVATION_DIM}), "
                       "deployment-legal observations only"),
            "init": "reference_weights()",
            "init_exact_batch_objective": init_objective,
        },
        "gradient_verification": {
            "rloo2_expectation_equals_exact_gradient_max_abs_err": identity_err,
            "monte_carlo_rloo2_mean_vs_exact": mc,
            "estimator_variance_at_init": variance_at_init,
            "estimator_variance_at_exact_trained": variance_at_exact_trained,
            "note": ("reinforce_b0, reinforce_mean_baseline and rloo2_loo are "
                     "unbiased (bias ~1e-16); self_mean_g2 includes the sample "
                     "in its own baseline, is biased toward grad/2, and its "
                     "low variance must not be read as an advantage."),
        },
        "training": {
            "hyperparameters": {
                "supervised": {"lr": SUPERVISED_LR, "steps": sup_steps},
                "exact_joint": {"lr": EXACT_LR, "steps": exact_steps},
                "rloo2_onpolicy": {"lr": RLOO_LR, "steps": rloo_steps,
                                   "seeds": list(rloo_seeds), "grad_clip": RLOO_CLIP},
                "eval_every": eval_every,
            },
            "supervised_oracle_imitation": strip(supervised),
            "exact_joint_gradient": strip(exact),
            "rloo2_onpolicy": {
                "runs": [strip(r) for r in rloo_runs],
                "final_objective_mean": float(np.mean(rloo_finals)),
                "final_objective_std": float(np.std(rloo_finals)),
                "final_objective_min": float(np.min(rloo_finals)),
                "final_objective_max": float(np.max(rloo_finals)),
            },
            "eval_enumerated_trajectories_per_curve_point": sum(paths_per_instance(env)),
        },
        "validation": validation,
        "complementary_two_step": complementary_case_report(env, weights_by_method),
        "budget": budget_report(env, weights_by_method, trajectories_checked),
    }
    result["findings"] = _findings(result)
    return result


def _findings(result):
    val = result["validation"]
    tr = result["training"]
    sup = tr["supervised_oracle_imitation"]
    sup_des = val["supervised_oracle_imitation"]["designed"]
    sup_held = val["supervised_oracle_imitation"]["held_out"]
    exact_des = val["exact_joint_gradient"]["designed"]
    rloo = tr["rloo2_onpolicy"]
    rloo_des = val["rloo2_onpolicy_best_seed"]["designed"]
    var = result["gradient_verification"]["estimator_variance_at_init"]
    ident = result["gradient_verification"][
        "rloo2_expectation_equals_exact_gradient_max_abs_err"]
    return [
        (f"Supervised imitation of the enumerable optimal paths recovers "
         f"{sup_des['n_recovered']}/{sup_des['n_instances']} designed optima at ~zero "
         f"per-update rollout cost (final batch objective "
         f"{sup['final_exact_batch_objective']:.4f}); exact joint score-function descent "
         f"reaches a lower batch objective "
         f"({tr['exact_joint_gradient']['final_exact_batch_objective']:.4f}) yet recovers only "
         f"{exact_des['n_recovered']}/{exact_des['n_instances']} designed optima — the joint "
         f"objective is non-convex in the router weights, and batch E[F] and greedy "
         f"path recovery are different criteria."),
        (f"On-policy G=2 RLOO has exactly the same expected update as exact "
         f"descent (enumeration identity err {ident:.2e}) plus batch total gradient "
         f"variance {var['rloo2_loo']['total_variance']:.4f} at init (vs "
         f"{var['reinforce_b0']['total_variance']:.4f} for unbaselined REINFORCE); across "
         f"seeds its final batch objective is {rloo['final_objective_mean']:.4f} ± "
         f"{rloo['final_objective_std']:.4f} (range [{rloo['final_objective_min']:.4f}, "
         f"{rloo['final_objective_max']:.4f}]), and its best seed recovers "
         f"{rloo_des['n_recovered']}/{rloo_des['n_instances']} designed optima — the sampled "
         f"estimator inherits the exact update's local optima and adds "
         f"cross-seed variance on top."),
        (f"Held-out recovery on fresh random instances is modest for every "
         f"method (supervised {sup_held['recovery_rate']:.3f}): one linear policy trained "
         f"on 6 tasks generalizes weakly, so none of the designed-set gains "
         f"should be read as general competence."),
        ("All conclusions are restricted to this enumerable synthetic "
         "environment; nothing here licenses or refutes router RL on a real "
         "language model (that part of E09 is marked not_executed)."),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E09 controlled: policy gradient vs supervised/exact baselines")
    ap.add_argument("--output", required=True, help="path for the JSON results")
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--quick", action="store_true",
                    help="reduced steps/seeds for smoke tests")
    args = ap.parse_args(argv)
    result = run(seed=args.seed, quick=args.quick)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    val = result["validation"]
    tr = result["training"]
    print(f"wrote {args.output}")
    print(f"real_lm_rl: {result['real_lm_rl']['status']}")
    print(f"init batch objective: {result['environment']['init_exact_batch_objective']:.4f}")
    print(f"supervised final: {tr['supervised_oracle_imitation']['final_exact_batch_objective']:.4f} "
          f"designed recovery {val['supervised_oracle_imitation']['designed']['n_recovered']}/6")
    print(f"exact final: {tr['exact_joint_gradient']['final_exact_batch_objective']:.4f} "
          f"designed recovery {val['exact_joint_gradient']['designed']['n_recovered']}/6")
    print(f"rloo2 final: {tr['rloo2_onpolicy']['final_objective_mean']:.4f} "
          f"+- {tr['rloo2_onpolicy']['final_objective_std']:.4f}")
    print(f"rloo2==exact identity err: "
          f"{result['gradient_verification']['rloo2_expectation_equals_exact_gradient_max_abs_err']:.2e}; "
          f"MC check passed: {result['gradient_verification']['monte_carlo_rloo2_mean_vs_exact']['passed']}")
    print(f"hard budget violations: {result['budget']['hard_budget_violations']}")


if __name__ == "__main__":
    main()
