from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.controller_corpus import prepare_controller_corpus
from data.fineweb import load_tokenizer
from data.prepare_fineweb import fetch_documents
from experiments.controller_exclusions import write_used_document_exclusions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--exclude-documents", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-tokens", type=int, default=50000)
    parser.add_argument("--calibration-tokens", type=int, default=50000)
    parser.add_argument("--newtest-tokens", type=int, default=50000)
    parser.add_argument("--controller-train-tokens", type=int, default=200000)
    args = parser.parse_args()
    doc_source, shards = fetch_documents(args.revision, limit_shards=1)
    output_dir = Path(args.output_dir)
    exclusion_paths = [
        write_used_document_exclusions(Path(path).parent, output_dir / "exclusions" / f"{index}.jsonl")
        for index, path in enumerate(args.exclude_documents)
    ]
    manifest = prepare_controller_corpus(
        doc_source,
        Path(args.output_dir),
        tokenizer=load_tokenizer(Path(args.tokenizer)),
        revision=args.revision,
        source="HuggingFaceFW/fineweb-edu:sample-10BT",
        selection_tokens=args.selection_tokens,
        calibration_tokens=args.calibration_tokens,
        newtest_tokens=args.newtest_tokens,
        controller_train_tokens=args.controller_train_tokens,
        exclude_documents=exclusion_paths,
        tokenizer_path=args.tokenizer,
        tokenizer_info={"source_shards": shards},
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
