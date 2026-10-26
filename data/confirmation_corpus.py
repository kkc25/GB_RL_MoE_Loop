from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

from data.controller_corpus import DocSource, build_doc_records, encode_document, load_exclusion_hashes, mark_excluded
from data.fineweb import SplitWriter, doc_fingerprint, sha256_file


def confirmation_group_key(near_hash: str) -> str:
    return hashlib.sha256(f"fixed-depth-confirmation-v1:{near_hash}".encode()).hexdigest()


def prepare_confirmation_corpus(doc_source: DocSource, out_dir: Path, *, tokenizer, revision: str, source: str, confirmation_tokens: int, pool_bytes: int, exclude_documents, seq_len: int = 1024, tokenizer_path=None, tokenizer_info=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = build_doc_records(doc_source())
    exclusion = load_exclusion_hashes(exclude_documents)
    excluded = mark_excluded(records, exclusion)
    groups = {}
    for record, is_excluded in zip(records, excluded):
        if is_excluded:
            continue
        group = groups.setdefault(record["near_hash"], {"records": [], "n_bytes": 0})
        group["records"].append(record)
        group["n_bytes"] += record["n_bytes"]
    selected = []
    selected_bytes = 0
    for near_hash, group in sorted(groups.items(), key=lambda item: (confirmation_group_key(item[0]), item[0])):
        if selected_bytes >= pool_bytes:
            break
        selected.extend(group["records"])
        selected_bytes += group["n_bytes"]
    selected_hashes = {record["doc_hash"] for record in selected}
    selected_path = out_dir / "confirmation_documents.jsonl"
    with selected_path.open("w") as handle:
        for record in selected:
            handle.write(json.dumps(record | {"role": "confirmation"}, sort_keys=True) + "\n")
    rank_of_hash = {record["doc_hash"]: record["rank"] for record in selected}
    writer = SplitWriter(out_dir, "confirmation", seq_len, max_tokens=confirmation_tokens)
    for doc_id, text in doc_source():
        document_hash = doc_fingerprint(str(doc_id), text)
        if document_hash in selected_hashes:
            if not writer.add_document(rank_of_hash[document_hash], encode_document(tokenizer, text)):
                break
    info = writer.finish()
    if info["n_tokens"] != confirmation_tokens:
        raise RuntimeError(f"confirmation pool has {info['n_tokens']} tokens, needs {confirmation_tokens}")
    used_ranks = set()
    with (out_dir / "confirmation.doc_map.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            used_ranks.update(int(segment["doc"]) for segment in row["segments"])
    selected_by_rank = {record["rank"]: record for record in selected}
    frozen = {
        "document_ids": [selected_by_rank[rank]["id"] for rank in sorted(used_ranks)],
        "document_ranks": sorted(used_ranks),
        "document_hashes": [selected_by_rank[rank]["doc_hash"] for rank in sorted(used_ranks)],
        "near_hashes": sorted({selected_by_rank[rank]["near_hash"] for rank in used_ranks}),
    }
    frozen_path = out_dir / "frozen_document_ids.json"
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
    tokenizer_info = dict(tokenizer_info or {})
    if tokenizer_path is not None:
        tokenizer_info.update({"path": str(tokenizer_path), "sha256": sha256_file(tokenizer_path)})
    tokenizer_info.update({"vocab_size": tokenizer.get_vocab_size(), "trained_by": "caller"})
    paths = [exclude_documents] if isinstance(exclude_documents, (str, Path)) else list(exclude_documents)
    manifest = {
        "pipeline_version": "fixed-depth-confirmation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "revision": revision,
        "config": {"confirmation_tokens": confirmation_tokens, "pool_bytes": pool_bytes, "seq_len": seq_len},
        "environment": {"python": platform.python_version()},
        "n_source_documents": len(records),
        "n_excluded_documents": sum(excluded),
        "n_eligible_documents": len(records) - sum(excluded),
        "n_selected_documents": len(selected),
        "n_frozen_documents": len(used_ranks),
        "selected_bytes": selected_bytes,
        "confirmation_documents_sha256": sha256_file(selected_path),
        "frozen_document_ids_sha256": sha256_file(frozen_path),
        "exclusion": {
            "documents_files": [str(path) for path in paths],
            "documents_sha256": [sha256_file(path) for path in paths],
            "n_doc_hashes": len(exclusion["doc_hashes"]),
            "n_text_hashes": len(exclusion["text_hashes"]),
        },
        "tokenizer": tokenizer_info,
        "splits": {"confirmation": info},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
