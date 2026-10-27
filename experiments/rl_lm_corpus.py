"""CLI: build the RL-LM-Pair-v1 four-role packed corpus from raw JSONL.

Spec section 4.6: this entry adapts historical corpus identities, applies
exclusions, and produces the new controller_train/selection/calibration/
newtest splits.  It never downloads data and never fits a tokenizer; all
inputs are pinned local files with recorded hashes.  ``rl_lm_prepare`` only
verifies an existing corpus and builds prefix caches; missing files here are
hard errors, never silently replaced by historical test data.

Example (spec section 20)::

    python -m experiments.rl_lm_corpus --config configs/rl_lm/corpus.json \
        --source-jsonl /path/pinned_raw_documents.jsonl \
        --source-revision EXACT_REVISION --tokenizer "$RL_TOKENIZER" \
        --output-dir "$RL_CORPUS" --exclude-corpus-dir /path/old_corpus ...
"""

from __future__ import annotations

import argparse
import json

from rl_lm.config import CorpusConfig
from rl_lm.corpus import build_rl_corpus


def apply_overrides(data: dict, args: argparse.Namespace) -> dict:
    """CLI values override null/path fields; exclusion dirs are appended."""
    data = dict(data)
    if args.source_jsonl is not None:
        data["source_jsonl"] = args.source_jsonl
    if args.source_revision is not None:
        data["source_revision"] = args.source_revision
    if args.tokenizer is not None:
        data["tokenizer_path"] = args.tokenizer
    if args.output_dir is not None:
        data["output_dir"] = args.output_dir
    extra = [entry for group in args.exclude_corpus_dir or [] for entry in group]
    if extra:
        merged = list(data["exclude_corpus_dirs"])
        for entry in extra:
            if entry not in merged:
                merged.append(entry)
        data["exclude_corpus_dirs"] = merged
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="corpus JSON config (CorpusConfig)")
    parser.add_argument("--source-jsonl", default=None, help="local UTF-8 JSONL of id/text rows")
    parser.add_argument("--source-revision", default=None, help="exact source revision, recorded verbatim")
    parser.add_argument("--tokenizer", default=None, help="already-trained tokenizer.json")
    parser.add_argument("--output-dir", default=None, help="target corpus directory (must be empty/new)")
    parser.add_argument(
        "--exclude-corpus-dir",
        action="append",
        nargs="+",
        default=None,
        metavar="DIR",
        help="one or more complete historical corpus directories; repeatable",
    )
    args = parser.parse_args()

    config = CorpusConfig(apply_overrides(CorpusConfig.load(args.config).to_dict(), args))
    config.validate_required()
    coverage = build_rl_corpus(config)
    print(json.dumps(coverage, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
