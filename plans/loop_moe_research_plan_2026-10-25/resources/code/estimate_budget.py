#!/usr/bin/env python3
"""Analytic budget calculator; no GPU benchmark and no measured speed claims.
Usage: python3 estimate_budget.py --loop-blocks 1 --loops 4 --tokens 1000000000
Optional --tokens-per-second is a USER-SUPPLIED measured/assumed one-GPU speed.
"""
import argparse
import json


def estimate(layers=12, width=512, experts=8, top_k=2, intermediate=512,
             vocab=32768, sequence=1024, loop_blocks=1, loops=4):
    assert 1 <= top_k <= experts
    assert 0 <= loop_blocks <= layers
    assert loops >= 1 and sequence >= 1
    L, d, E, k, m, V, S = layers, width, experts, top_k, intermediate, vocab, sequence
    parts = {
        "tied_embedding": V * d,
        "attention_projections": L * 4 * d * d,
        "expert_weights": L * E * 3 * d * m,
        "router_weights": L * d * E,
        "rmsnorm_weights": (2 * L + 1) * d,
    }
    total_passes = L - loop_blocks + loop_blocks * loops
    forward = {
        "attention_projections": 2 * L * 4 * d * d,
        "expert_matmuls": 2 * total_passes * k * 3 * d * m,
        "router_matmuls": 2 * total_passes * d * E,
        "one_lm_head": 2 * d * V,
        "attention_scores_values_ideal_causal": 2 * L * (S + 1) * d,
    }
    full_square_attention = 4 * L * S * d
    fwd = sum(forward.values())
    p = sum(parts.values())
    return {
        "status": "analytic_only_not_measured",
        "assumptions": ["MHA with four d-by-d projections", "SwiGLU three matrices",
                        "no linear biases", "shared parameters across FFN loops",
                        "two RMSNorm vectors per layer and one final vector",
                        "one router per layer", "tied embedding and LM head",
                        "one output-head evaluation per token", "multiply-add counted as 2 FLOPs",
                        "normalization, nonlinearities, softmax, indexing, communication excluded"],
        "parameter_parts": parts,
        "parameters_total": p,
        "model_state_gib_16_bytes_per_parameter": 16 * p / (2 ** 30),
        "ffn_passes_across_all_layers": total_passes,
        "forward_flops_per_token_parts": forward,
        "forward_flops_per_token_ideal_causal": fwd,
        "forward_flops_per_token_full_square_attention": fwd - forward["attention_scores_values_ideal_causal"] + full_square_attention,
        "training_flops_per_token_3x_forward_approx": 3 * fwd,
        "extra_final_block_topk_step_forward_flops": 2 * k * 3 * d * m + 2 * d * E,
        "each_extra_full_vocab_readout_flops": 2 * d * V,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intermediate", type=int, default=512)
    ap.add_argument("--loop-blocks", type=int, default=1)
    ap.add_argument("--loops", type=float, default=4)
    ap.add_argument("--sequence", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=1_000_000_000)
    ap.add_argument("--tokens-per-second", type=float, default=None)
    args = ap.parse_args()
    out = estimate(intermediate=args.intermediate, loop_blocks=args.loop_blocks,
                   loops=args.loops, sequence=args.sequence)
    out["training_flops_total_approx"] = out["training_flops_per_token_3x_forward_approx"] * args.tokens
    if args.tokens_per_second is not None:
        assert args.tokens_per_second > 0
        out["user_supplied_tokens_per_second"] = args.tokens_per_second
        out["one_gpu_hours_at_supplied_speed"] = args.tokens / args.tokens_per_second / 3600
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
