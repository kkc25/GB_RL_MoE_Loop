"""CLI: build the FineWeb-Edu LM dataset for the 126M/280M experiments.

Downloads ``HuggingFaceFW/fineweb-edu`` (config ``sample-10BT``) at an
*explicitly pinned* revision and runs the local pipeline in
``data/fineweb.py``: fingerprint-based splits, byte-level BPE tokenizer on a
fixed budget of training text, exact validation/test token budgets,
length-1024 packing with document-boundary metadata, and a provenance
manifest.

The revision is never guessed or defaulted: ``--revision`` is required and is
validated against the Hub before any download.  Get the current commit hash
from https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/commits or via
``HfApi.dataset_info("HuggingFaceFW/fineweb-edu").sha``.

Example (real run, needs pyarrow in the environment for parquet reading):

    python -m data.prepare_fineweb \
        --revision <commit-sha> \
        --out-dir runs/data/fineweb_edu_10bt \
        --train-tokens 2000000000

Offline / pre-downloaded mode (still requires an explicit revision label,
which is recorded verbatim in the manifest):

    python -m data.prepare_fineweb \
        --revision <commit-sha> \
        --from-jsonl raw_docs.jsonl \
        --out-dir runs/data/fineweb_edu_10bt

The gb conda env ships transformers/tokenizers/huggingface_hub but neither
``datasets`` nor ``pyarrow``; parquet shards therefore cannot be parsed there
and the fetch mode exits with a clear error.  Everything else (splitting,
tokenizer training, packing, manifest) only needs the standard library plus
``tokenizers`` and ``numpy``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import fineweb as FW


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_fineweb",
        description="Prepare the FineWeb-Edu sample-10BT LM dataset (pinned revision).",
    )
    parser.add_argument(
        "--revision",
        required=True,
        help="Exact HuggingFace dataset revision (commit sha). Required; never "
        "defaults to a moving branch, so runs stay reproducible.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for splits + manifest.")
    parser.add_argument(
        "--from-jsonl",
        default=None,
        help="Offline mode: read documents from this JSONL file (one "
        '{"id": ..., "text": ...} per line) instead of downloading from the Hub.',
    )
    parser.add_argument("--max-docs", type=int, default=None, help="Cap on documents read (smoke runs).")
    parser.add_argument("--limit-shards", type=int, default=None, help="Cap on parquet shards downloaded.")
    parser.add_argument("--seq-len", type=int, default=FW.SEQ_LEN)
    parser.add_argument("--vocab-size", type=int, default=FW.VOCAB_SIZE,
                        help="BPE vocabulary size including special tokens.")
    parser.add_argument("--tokenizer-bytes", type=int, default=FW.TOKENIZER_TRAIN_BYTES,
                        help="Bytes of training text used to fit the tokenizer.")
    parser.add_argument("--val-tokens", type=int, default=FW.VAL_TOKENS)
    parser.add_argument("--test-tokens", type=int, default=FW.TEST_TOKENS)
    parser.add_argument("--train-tokens", type=int, default=None,
                        help="Deterministic first-D-token budget for train; default keeps all.")
    parser.add_argument("--val-pool-bytes", type=int, default=None,
                        help="Override the validation document-pool byte target.")
    parser.add_argument("--test-pool-bytes", type=int, default=None,
                        help="Override the test document-pool byte target.")
    return parser


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading FineWeb-Edu parquet shards requires 'pyarrow', which is not "
            "installed in this environment (the gb env has neither datasets nor "
            "pyarrow). Install it (e.g. 'pip install pyarrow') or pre-convert the "
            "shards and use --from-jsonl."
        ) from exc
    return pq


def list_shards(revision: str, limit: Optional[int] = None) -> List[str]:
    """Sorted sample-10BT parquet shard names at the pinned revision.

    Listing at ``revision`` also validates that the revision exists; an
    unknown revision raises instead of silently falling back to a branch.
    """
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(FW.REPO_ID, revision=revision)
    shards = sorted(
        item.rfilename for item in info.siblings if item.rfilename.startswith(f"sample/{FW.SOURCE_CONFIG.replace('sample-', '')}/")
        and item.rfilename.endswith(".parquet")
    )
    if not shards:
        raise RuntimeError(
            f"no sample-10BT parquet shards found in {FW.REPO_ID} at revision {revision}"
        )
    return shards[:limit] if limit else shards


def fetch_documents(
    revision: str,
    *,
    limit_shards: Optional[int] = None,
    max_docs: Optional[int] = None,
) -> Tuple[callable, List[str]]:
    """Download shards at the pinned revision; return (doc_source, shard list)."""
    from huggingface_hub import hf_hub_download

    pq = _require_pyarrow()
    shards = list_shards(revision, limit_shards)
    local_paths = [
        hf_hub_download(FW.REPO_ID, name, repo_type="dataset", revision=revision)
        for name in shards
    ]

    def doc_source() -> Iterator[Tuple[str, str]]:
        produced = 0
        for path in local_paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=1024, columns=["id", "text"]):
                for doc_id, text in zip(
                    batch.column("id").to_pylist(), batch.column("text").to_pylist()
                ):
                    if max_docs is not None and produced >= max_docs:
                        return
                    produced += 1
                    yield str(doc_id), text

    return doc_source, shards


def jsonl_documents(path: str, max_docs: Optional[int] = None) -> callable:
    def doc_source() -> Iterator[Tuple[str, str]]:
        docs = FW.load_raw_docs_jsonl(Path(path))
        yield from (docs if max_docs is None else itertools.islice(docs, max_docs))

    return doc_source


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.from_jsonl:
        doc_source = jsonl_documents(args.from_jsonl, args.max_docs)
        shards = None
    else:
        doc_source, shards = fetch_documents(
            args.revision, limit_shards=args.limit_shards, max_docs=args.max_docs
        )
    manifest = FW.prepare_documents(
        doc_source,
        Path(args.out_dir),
        revision=args.revision,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        tokenizer_bytes=args.tokenizer_bytes,
        val_tokens=args.val_tokens,
        test_tokens=args.test_tokens,
        train_tokens=args.train_tokens,
        val_pool_bytes=args.val_pool_bytes,
        test_pool_bytes=args.test_pool_bytes,
        source_shards=shards,
    )
    summary = {
        "revision": manifest["revision"],
        "n_documents": manifest["n_documents"],
        "split_documents": manifest["split_documents"],
        "tokenizer_sha256": manifest["tokenizer"]["sha256"],
        "document_list_sha256": manifest["document_list_sha256"],
        "tokens": {k: v["n_tokens"] for k, v in manifest["splits"].items()},
        "sequences": {k: v["n_sequences"] for k, v in manifest["splits"].items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main()
