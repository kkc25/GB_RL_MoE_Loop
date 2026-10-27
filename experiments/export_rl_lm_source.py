"""CLI: export pinned FineWeb-Edu raw documents to a local UTF-8 JSONL.

Task card E step 1 (spec 4.6 item 1): the RL corpus builder only accepts a
local JSONL of ``{"id", "text"}`` rows with a recorded SHA256 and exact
source revision.  This entry produces that file through the existing pinned
downloader ``data.prepare_fineweb.fetch_documents`` — the revision is
mandatory and validated against the Hub, the local HF cache is reused when
the shard is already present, and the revision is never silently changed.

The output directory must be new/empty; existing files are never overwritten.
A sidecar ``<name>.manifest.json`` records the file hash, revision, shard
list and row count so the corpus config can pin the exact bytes.

Example::

    python -m experiments.export_rl_lm_source \
        --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
        --limit-shards 1 \
        --output data/rl_lm_pair_v1/source/fineweb_edu_sample10bt_shard0.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import fineweb as FW
from data.fineweb import sha256_file
from data.prepare_fineweb import fetch_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revision", required=True, help="exact dataset revision (commit sha)")
    parser.add_argument("--output", required=True, help="target JSONL path (must not exist)")
    parser.add_argument("--limit-shards", type=int, default=None, help="cap on parquet shards exported")
    parser.add_argument("--max-docs", type=int, default=None, help="cap on exported documents (smoke runs)")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"output already exists, refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    doc_source, shards = fetch_documents(
        args.revision, limit_shards=args.limit_shards, max_docs=args.max_docs
    )
    n_rows = 0
    with open(output, "w", encoding="utf-8") as handle:
        for doc_id, text in doc_source():
            handle.write(
                json.dumps({"id": doc_id, "text": text}, ensure_ascii=False, sort_keys=True) + "\n"
            )
            n_rows += 1
    file_sha = sha256_file(output)
    manifest = {
        "pipeline_version": "rl-lm-source-export-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": FW.REPO_ID,
        "source_config": FW.SOURCE_CONFIG,
        "revision": args.revision,
        "shards": shards,
        "limit_shards": args.limit_shards,
        "max_docs": args.max_docs,
        "n_rows": n_rows,
        "output_file": output.name,
        "output_sha256": file_sha,
    }
    manifest_path = output.parent / (output.name + ".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
