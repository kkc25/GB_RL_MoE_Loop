import argparse
import json
from pathlib import Path

import numpy as np

H, E, COST = 3, 3, 0.07
W = np.array([[.1, -.2, .3, .05], [-.15, .25, .1, -.05], [.2, .1, -.1, .1], [-.1, .05, .2, -.15]], dtype=float)
A = np.array([[[.8, .2], [-.1, .6]], [[.4, -.7], [.5, .3]], [[-.2, .8], [.4, .7]]])
v = np.array([[.1, -.1], [-.05, .2], [.2, .05]])
h0 = np.array([.3, -.2])


def leaves(w):
    out = []

    def visit(h, t, p, score, path):
        if t == H:
            loss = .5 * (np.dot([1., -.6], h) - .8) ** 2 + COST * t
            out.append((p, loss, score.copy(), path))
            return
        f = np.r_[h, t / H, 1.]
        mask = np.array([t > 0, True, True, True])
        logits = w @ f
        logits = np.where(mask, logits, -np.inf)
        prob = np.exp(logits - np.max(logits))
        prob /= prob.sum()
        for a in np.flatnonzero(mask):
            grad = np.outer(np.eye(E + 1)[a] - prob, f)
            if a == 0:
                loss = .5 * (np.dot([1., -.6], h) - .8) ** 2 + COST * t
                out.append((p * prob[a], loss, score + grad, path + (a,)))
            else:
                nxt = h + .4 * np.tanh(A[a - 1] @ h + v[a - 1])
                visit(nxt, t + 1, p * prob[a], score + grad, path + (a,))

    visit(h0, 0, 1., np.zeros_like(w), ())
    return out


def objective(w):
    return sum(p * f for p, f, g, a in leaves(w))


def run():
    tree = leaves(W)
    grad = sum(p * f * g for p, f, g, a in tree)
    score_mean = sum(p * g for p, f, g, a in tree)
    eps = 1e-5
    fd = np.zeros_like(W)
    for idx in np.ndindex(W.shape):
        plus = W.copy()
        minus = W.copy()
        plus[idx] += eps
        minus[idx] -= eps
        fd[idx] = (objective(plus) - objective(minus)) / (2 * eps)
    loo = np.zeros_like(W)
    self_mean = np.zeros_like(W)
    for p1, f1, g1, a1 in tree:
        for p2, f2, g2, a2 in tree:
            loo += p1 * p2 * .5 * ((f1 - f2) * g1 + (f2 - f1) * g2)
            mean = (f1 + f2) / 2
            self_mean += p1 * p2 * .5 * ((f1 - mean) * g1 + (f2 - mean) * g2)
    np.testing.assert_allclose(grad, fd, atol=1e-9)
    np.testing.assert_allclose(loo, grad, atol=1e-12)
    np.testing.assert_allclose(self_mean, .5 * grad, atol=1e-12)
    np.testing.assert_allclose(score_mean, 0, atol=1e-12)
    r = np.array([-.1, -.1, -.2])
    phi = np.array([.7, .4, -.1, 0.])
    shaped = r + phi[1:] - phi[:-1]
    np.testing.assert_allclose(shaped.sum(), r.sum() - phi[0])
    stop_now = 1.
    after_one = 1.1
    after_two = .2
    c = .1
    greedy_continues = (stop_now - after_one) > c
    optimal_continues = (after_two + 2 * c) < stop_now
    losses = np.array([1., 1.1, .2])
    z = np.array([.2, -.4])
    haz = 1 / (1 + np.exp(-z))

    def halt_obj(z):
        h = 1 / (1 + np.exp(-z))
        q = np.array([h[0], (1 - h[0]) * h[1], (1 - h[0]) * (1 - h[1])])
        return q @ (losses + .1 * np.arange(1, 4))

    g_halt = np.array([
        haz[0] * (1 - haz[0]) * ((losses[0] + .1) - (haz[1] * (losses[1] + .2) + (1 - haz[1]) * (losses[2] + .3))),
        (1 - haz[0]) * haz[1] * (1 - haz[1]) * ((losses[1] + .2) - (losses[2] + .3))])
    fd_halt = []
    for k in range(2):
        p = z.copy()
        m = z.copy()
        p[k] += eps
        m[k] -= eps
        fd_halt.append((halt_obj(p) - halt_obj(m)) / (2 * eps))
    np.testing.assert_allclose(g_halt, fd_halt, atol=1e-9)
    result = {
        'kind': 'CPU exact enumeration and finite differences; not an LLM experiment',
        'expert_count': E, 'max_calls': H, 'terminal_paths': len(tree),
        'total_probability': sum(x[0] for x in tree), 'expected_loss': objective(W),
        'max_abs_pg_vs_fd': float(np.max(np.abs(grad - fd))),
        'max_abs_rloo2_vs_exact_pg': float(np.max(np.abs(loo - grad))),
        'max_abs_self_mean_vs_half_pg': float(np.max(np.abs(self_mean - .5 * grad))),
        'max_abs_mean_score': float(np.max(np.abs(score_mean))),
        'potential_shaping_shift': float(shaped.sum() - r.sum()),
        'myopic_counterexample': {'stop_cost': stop_now, 'two_call_cost': after_two + 2 * c,
                                  'greedy_continues': greedy_continues, 'optimal_continues': optimal_continues},
        'expected_route_nll': float(-np.log([.9, .1]).mean()),
        'marginal_nll': float(-np.log(np.mean([.9, .1]))),
        'max_abs_exact_halt_vs_fd': float(np.max(np.abs(g_halt - fd_halt))),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='Exact finite-tree math checks; no LLM/GPU training.')
    parser.add_argument('--output', default=None,
                        help='Optional path to write the JSON result. Default: only print to stdout.')
    args = parser.parse_args()
    r = run()
    text = json.dumps(r, indent=2)
    if args.output:
        Path(args.output).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
