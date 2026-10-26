"""Recompute E10 document coverage and tolerance-aware signs from public JSONL.

This uses six-decimal serialized costs, not the missing full-precision tensor.
No model is trained or evaluated. Floating arithmetic here uses NumPy float64.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent / "GB_RL_MoE_Loop")
    args = parser.parse_args()
    run = args.repo / "runs/e10/pilot"
    summary = json.loads((run / "e10_summary.json").read_text())
    result = {
        "scope": "Coverage is exact from uploaded records. Gamma signs are an exploratory recomputation from rounded six-decimal costs, not a corrected full-precision experiment.",
        "source_commit": "f2bef422f7315f30618f00c690bf82810d1e85ac",
        "arithmetic": "numpy.float64",
        "coverage": {},
        "source_sha256": {},
    }
    records = {}
    for split in ("train", "validation", "test"):
        source = run / f"records_{split}.jsonl"
        records[split] = [json.loads(line) for line in source.read_text().splitlines()]
        rows = records[split]
        result["coverage"][split] = {
            "states": len(rows),
            "document_counts": dict(Counter(str(row["doc"]) for row in rows)),
            "sequence_counts": dict(Counter(str(row["seq"]) for row in rows)),
        }
        result["source_sha256"][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    q = np.asarray([row["pair_q"] for row in records["test"]], dtype=np.float64).reshape(-1, 28, 28)
    anchors = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 2))
    means = []
    for a0, b0 in anchors:
        gamma = q[:, :, b0, None] + q[:, a0, None, :] - q - q[:, a0, b0, None, None]
        means.append(gamma.mean(axis=0))
    values = np.stack(means)
    positive = values > 0
    coded = positive.any(axis=0) & (~positive).any(axis=0)
    eps = 1e-6
    strict = (values > eps).any(axis=0) & (values < -eps).any(axis=0)
    result["signs"] = {
        "anchors": [list(pair) for pair in anchors],
        "original_full_precision_summary_fraction": summary["reference_sensitivity"]["sign_flip_pair_fraction"],
        "recomputed_original_boolean_rule_on_rounded_costs": float(coded.mean()),
        "epsilon_nat": eps,
        "strict_positive_negative_fraction_on_rounded_costs": float(strict.mean()),
        "strict_positive_negative_pairs": int(strict.sum()),
        "all_pairs": int(strict.size),
        "interpretation": "The zero-band illustrates sensitivity and corrects the boolean definition on this serialized table. It does not recover the missing full-precision Gamma tensor; use original states_test.pt before publishing corrected full-precision statistics.",
    }
    destination = Path(__file__).with_name("e10_records_recheck.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
