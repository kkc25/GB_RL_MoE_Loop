from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.confirmation_corpus import prepare_confirmation_corpus
from data.fineweb import load_tokenizer
from data.prepare_fineweb import fetch_documents
from experiments.controller_exclusions import write_used_document_exclusions
from experiments.r1_closed_loop import source_identity
from train.recovery_common import atomic_json, sha256_file


def write_controller_exclusions(controller_dir, states_path, output_path):
    controller_dir = Path(controller_dir)
    states = torch.load(states_path, map_location="cpu", weights_only=False)
    used_ranks = set(states["stats"]["doc_ids"])
    heldout_roles = {"selection", "calibration", "newtest"}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"heldout_roles": 0, "controller_train_states": 0}
    with (controller_dir / "documents.jsonl").open() as source, output_path.open("w") as output:
        for line in source:
            row = json.loads(line)
            if row["role"] in heldout_roles:
                output.write(json.dumps(row) + "\n")
                counts["heldout_roles"] += 1
            elif row["role"] == "controller_train" and int(row["rank"]) in used_ranks:
                output.write(json.dumps(row) + "\n")
                counts["controller_train_states"] += 1
    if counts["controller_train_states"] != len(used_ranks):
        raise ValueError("not every sampled controller_train document was excluded")
    return output_path, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")
    parser.add_argument("--tokenizer", default="runs/data/fineweb_edu_100m/tokenizer.json")
    parser.add_argument("--config", default="configs/e00_m105_b2.json")
    parser.add_argument("--checkpoint", default="runs/e00/m105_b2/latest.pt")
    parser.add_argument("--output-dir", default="runs/followup_2026_10_26/plan_c")
    parser.add_argument("--tokens", type=int, default=1_100_000)
    parser.add_argument("--pool-bytes", type=int, default=13_200_000)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    plan_path = output_dir / "evaluation_plan.json"
    if plan_path.exists():
        raise FileExistsError(f"frozen evaluation plan already exists: {plan_path}")
    exclusions = output_dir / "exclusions"
    original = [
        write_used_document_exclusions(Path(path).parent, exclusions / f"original_{index}.jsonl")
        for index, path in enumerate((
            "runs/data/fineweb_edu_100m/documents.jsonl",
            "runs/data/fineweb_edu_debug/documents.jsonl",
        ))
    ]
    controller, controller_counts = write_controller_exclusions(
        "runs/repair_2026_10_25/controller_corpus",
        "runs/repair_2026_10_25/r2_full/states_controller_train.pt",
        exclusions / "controller_used.jsonl",
    )
    doc_source, shards = fetch_documents(args.revision, limit_shards=1)
    corpus_dir = output_dir / "confirmation_corpus"
    manifest = prepare_confirmation_corpus(
        doc_source,
        corpus_dir,
        tokenizer=load_tokenizer(Path(args.tokenizer)),
        revision=args.revision,
        source="HuggingFaceFW/fineweb-edu:sample-10BT",
        confirmation_tokens=args.tokens,
        pool_bytes=args.pool_bytes,
        exclude_documents=[*original, controller],
        tokenizer_path=args.tokenizer,
        tokenizer_info={"source_shards": shards},
    )
    plan = {
        "protocol": "fixed-depth-natural-token-confirmation-v1",
        "created_before_evaluation": True,
        "source": source_identity(),
        "config_path": args.config,
        "config_sha256": sha256_file(args.config),
        "checkpoint_path": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "corpus_manifest_sha256": sha256_file(corpus_dir / "manifest.json"),
        "frozen_document_ids_sha256": manifest["frozen_document_ids_sha256"],
        "depths": [1, 2, 4],
        "primary_comparison": "D = sum(L_R1 - L_R2) / N_valid_tokens",
        "secondary_comparison": "R1 - R4 and R2 - R4, descriptive",
        "tolerance_nat_per_token": 0.005,
        "decision_rule": "one-sided 95% document-bootstrap upper bound of D < 0.005",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 0,
        "cluster": "document",
        "weighting": "natural valid-token counts; all packed sequences; no per-document cap",
        "minimum_supervised_tokens": 1_000_000,
        "packed_real_tokens": args.tokens,
        "evaluation_batch_size": 1,
        "dtype": "bfloat16",
        "controller_training": "none; fixed depths only",
        "controller_exclusion_counts": controller_counts,
        "sample_size_changes_after_results": "forbidden",
    }
    atomic_json(plan_path, plan)
    print(json.dumps({"plan": str(plan_path), "corpus": manifest}, indent=2))


if __name__ == "__main__":
    main()
