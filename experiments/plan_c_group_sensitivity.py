from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.evaluate_plan_c import document_bootstrap
from train.recovery_common import atomic_json, sha256_file


def main():
    root = Path("runs/followup_2026_10_26/plan_c")
    output = Path("runs/next_2026_10_26/plan_c_group_sensitivity.json")
    per = json.loads((root / "per_document.json").read_text())
    records = [json.loads(line) for line in (root / "confirmation_corpus/confirmation_documents.jsonl").read_text().splitlines()]
    by_rank = {record["rank"]: record for record in records}
    ranks = np.asarray(per["document_ranks"], dtype=np.int64)
    counts = np.asarray(per["counts"], dtype=np.float64)
    losses = {key: np.asarray(value, dtype=np.float64) for key, value in per["loss_sums"].items()}
    near = np.asarray([by_rank[int(rank)]["near_hash"] for rank in ranks])
    groups, inverse = np.unique(near, return_inverse=True)
    group_counts = np.bincount(inverse, weights=counts)
    group_diff = np.bincount(inverse, weights=losses["1"] - losses["2"])
    leave_one_out = []
    for index in range(len(ranks)):
        keep = np.arange(len(ranks)) != index
        leave_one_out.append(float((losses["1"][keep] - losses["2"][keep]).sum() / counts[keep].sum()))
    result = {
        "scope": "post-hoc normalized-text-group sensitivity; original frozen document-level analysis retained",
        "document_ids": len(ranks),
        "normalized_text_groups": len(groups),
        "document_primary": document_bootstrap(losses["1"] - losses["2"], counts, 10000, 0),
        "group_sensitivity": document_bootstrap(group_diff, group_counts, 10000, 0),
        "leave_one_document_out_estimate_range": [min(leave_one_out), max(leave_one_out)],
        "valid_tokens": int(counts.sum()),
        "per_document_sha256": sha256_file(root / "per_document.json"),
        "confirmation_documents_sha256": sha256_file(root / "confirmation_corpus/confirmation_documents.jsonl"),
    }
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
