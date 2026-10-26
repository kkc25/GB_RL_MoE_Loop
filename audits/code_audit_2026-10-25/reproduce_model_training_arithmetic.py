"""CPU-only audit arithmetic and sampler benchmark; does not modify the repository.

Run from any directory: python reproduce_model_training_arithmetic.py
The sampler is imported directly from the pinned repository snapshot.
Timings are this audit host's CPU observations, not A6000 performance claims.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent / "GB_RL_MoE_Loop"
OUTPUT = Path(__file__).with_name("model_training_arithmetic.json")


def parameter_count(c):
    d, L, m, E, V = (c[k] for k in ("hidden_size", "layers", "intermediate_size", "experts_per_layer", "vocab_size"))
    moe = c.get("moe_layers")
    n_moe = L if moe is None else len(moe)
    return V*d + 4*L*d*d + 3*d*m*(n_moe*E + L-n_moe) + (2*L+1)*d + n_moe*d*E


def ffn_matrix_flops(c, depth):
    d, L, m, E = (c[k] for k in ("hidden_size", "layers", "intermediate_size", "experts_per_layer"))
    moe = c.get("moe_layers")
    moe = set(range(L) if moe is None else moe)
    adaptive = set(c["adaptive_layers"])
    return sum(6*d*m*(c["top_k"] if i in moe else 1)*(depth if i in adaptive else 1) for i in range(L))


def main():
    configs = {}
    for path in sorted((ROOT / "configs").glob("e00*.json")):
        c = json.loads(path.read_text())["model"]
        configs[path.stem] = {"parameters": parameter_count(c), "ffn_flops_per_token_at_R1": ffn_matrix_flops(c, 1), "ffn_flops_per_token_at_R4": ffn_matrix_flops(c, 4)}
    assert configs["e00_debug67_b2"]["parameters"] == 67_170_816
    assert configs["e00_m105_b2"]["parameters"] == 104_919_552
    assert configs["e00_dense_b4"]["parameters"] == 104_870_400

    summary = json.loads((ROOT / "runs/e00/m105_b2/run_summary.json").read_text())
    b0 = json.loads((ROOT / "runs/e00/debug67_b0/run_summary.json").read_text())
    ledger = summary["ledger"]
    assert b0["ledger"]["expert_calls"] == 4883*1024*12
    assert ledger["expert_calls"] == sum(1024*(11+1+(u+17)%4) for u in range(97657))

    spec = importlib.util.spec_from_file_location("audit_packed_lm", ROOT / "data/packed_lm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads((ROOT / "runs/data/fineweb_edu_100m/manifest.json").read_text())
    N, K = manifest["splits"]["train"]["n_sequences"], 12
    start = time.perf_counter()
    reconstructed = [next(iter(module.PackedLMBatchSampler(N, 1, seed=17, epoch=0, start_batch=i))) for i in range(K)]
    rebuild_seconds = time.perf_counter()-start
    start = time.perf_counter()
    iterator = iter(module.PackedLMBatchSampler(N, 1, seed=17, epoch=0))
    cached = [next(iterator) for _ in range(K)]
    one_iterator_seconds = time.perf_counter()-start
    assert reconstructed == cached

    result = {
        "commit": "f2bef422f7315f30618f00c690bf82810d1e85ac",
        "configuration_arithmetic": configs,
        "m105_input_tokens_per_update": ledger["input_tokens"]/ledger["updates"],
        "m105_supervised_tokens_per_update": ledger["supervised_tokens"]/ledger["updates"],
        "m105_ledger_named_expert_calls": ledger["expert_calls"],
        "m105_actual_expert_token_pairs_including_padding": 2*ledger["expert_calls"],
        "b0_ledger_named_expert_calls": b0["ledger"]["expert_calls"],
        "b0_actual_expert_token_pairs_including_padding": 2*b0["ledger"]["expert_calls"],
        "validation_tokens": summary["validation"]["depth_4"]["supervised_tokens"],
        "sampler_reproduction": {"manifest_number_of_sequences": N, "steps": K, "identical_sequence_order": reconstructed == cached, "rebuild_every_update_seconds": rebuild_seconds, "one_iterator_seconds": one_iterator_seconds, "time_ratio_for_these_K_steps": rebuild_seconds/one_iterator_seconds, "scope": "audit_host_CPU_only_not_GPU_performance_estimate"},
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
