from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch


def run(hidden_size=32, experts=8, samples=128, seed=0):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(samples, hidden_size, generator=generator, dtype=torch.float64)
    weights = torch.randn(experts, hidden_size, generator=generator, dtype=torch.float64)
    bias = torch.randn(experts, generator=generator, dtype=torch.float64)
    decentralized = torch.stack([hidden @ weights[expert] + bias[expert] for expert in range(experts)], dim=-1)
    centralized = hidden @ weights.T + bias
    difference = (decentralized - centralized).abs().max().item()
    return {
        "scope": "algebraic_equivalence_only_not_value_learning",
        "hidden_size": hidden_size,
        "experts": experts,
        "samples": samples,
        "max_absolute_difference": difference,
        "equivalent": difference < 1e-12,
        "conclusion": "linear_per_expert_bids_are_a_central_linear_router_reparameterization",
        "value_label_experiment": "awaiting_E01_same_hash_labels",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
