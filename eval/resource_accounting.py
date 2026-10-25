import argparse
import json
from math import ceil

V, E = 32768, 8

CONFIGS = [
    ("pilot", 8, 512, 1024),
    ("main", 12, 768, 1024),
    ("scale_optional", 16, 1024, 2048),
]

MAIN_L, MAIN_D, MAIN_M, S = 12, 768, 1024, 1024

FIXED_MFLOPS = (8 * MAIN_L * MAIN_D * MAIN_D + 4 * MAIN_L * S * MAIN_D + 2 * V * MAIN_D) / 1e6
EXPERT_MFLOPS = 6 * MAIN_D * MAIN_M / 1e6
GATE_MFLOPS = 2 * MAIN_D * E / 1e6
POLICY_MFLOPS = (2 * MAIN_D * 128 + 6 * 128 * (148 + 128) + 2 * 128 * 9) / 1e6

TOKENS_PER_UPDATE = 4 * 8 * 1024 * 8


def counts(L, d, m):
    backbone = V * d + 4 * L * d * d + 3 * L * E * d * m + (2 * L + 1) * d + L * d * E
    projection = d * 128 + 128
    gru = 3 * (148 * 128 + 128 * 128 + 2 * 128)
    actor = 128 * 9 + 9
    controller = projection + gru + actor
    critic = 128 + 1
    infer = backbone + controller
    train = infer + critic
    return {
        "backbone": backbone, "controller": controller,
        "inference_total": infer, "training_total_with_critic": train,
        "full_training_base_GiB": 16 * train / 2**30,
        "frozen_environment_base_GiB": (2 * backbone + 16 * (controller + critic)) / 2**30,
    }


def all_counts():
    return {name: counts(L, d, m) for name, L, d, m in CONFIGS}


def flops(K, D):
    return FIXED_MFLOPS + (EXPERT_MFLOPS + GATE_MFLOPS) * (8 + K) + POLICY_MFLOPS * D


def flops_without_gate_controller(K):
    return FIXED_MFLOPS + EXPERT_MFLOPS * (8 + K)


FLOPS_TABLE = [(4, 8), (6, 10), (8, 12), (12, 16), (16, 16)]


def flops_table():
    return [{"K": K, "D": D, "matrix_MFLOPs_per_token": flops(K, D)} for K, D in FLOPS_TABLE]


def shared_prefix_analysis(K=8, D=12):
    attention_one_layer = (8 * MAIN_D * MAIN_D + 4 * S * MAIN_D) / 1e6
    shared = 8 * (attention_one_layer + EXPERT_MFLOPS + GATE_MFLOPS) + attention_one_layer
    two_full = 2 * flops(K, D)
    two_shared = shared + 2 * (flops(K, D) - shared)
    return {
        "shared_prefix_MFLOPs_per_token": shared,
        "two_routes_with_shared_prefix": two_shared,
        "relative_matrix_saving": 1 - two_shared / two_full,
    }


def full_head_analysis():
    full_head = 2 * V * MAIN_D / 1e6
    return {
        "full_head_MFLOPs": full_head,
        "head_to_expert_ratio": full_head / EXPERT_MFLOPS,
        "stop_plus_8_branch_readout_MFLOPs_per_queried_state": 8 * EXPERT_MFLOPS + 9 * full_head,
    }


def token_quota(requested):
    steps = ceil(requested / TOKENS_PER_UPDATE)
    return {"tokens": requested, "full_updates": steps, "actual_tokens": steps * TOKENS_PER_UPDATE}


def report():
    return {
        "configs": all_counts(),
        "flops_constants_MFLOPs": {
            "fixed": FIXED_MFLOPS, "expert": EXPERT_MFLOPS,
            "gate": GATE_MFLOPS, "policy": POLICY_MFLOPS,
        },
        "flops_table": flops_table(),
        "shared_prefix": shared_prefix_analysis(),
        "full_head": full_head_analysis(),
        "tokens_per_update": TOKENS_PER_UPDATE,
        "token_quotas": [token_quota(r) for r in (1_000_000_000, 2_000_000_000)],
    }


def main():
    parser = argparse.ArgumentParser(description='Parameter and resource accounting (explicit matrix FLOPs estimates).')
    parser.add_argument('--output', default=None,
                        help='Optional path to write the JSON report. Default: only print to stdout.')
    args = parser.parse_args()
    text = json.dumps(report(), indent=2)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
