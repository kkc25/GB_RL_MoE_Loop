"""RL-LM-Pair-v1 corpus identity adaptation, exclusion, and four-role build.

Implements spec section 4.6 / task card B for the corpus path only:

1. Input is a local UTF-8 JSONL of ``{"id", "text"}`` rows; the actual file
   SHA256, source id and exact revision are recorded.  No network access and
   no tokenizer fitting happens here.
2. ``collect_exclusions`` reads every ``exclude_corpus_dirs`` manifest and
   walks **all** declared ``splits.*.doc_map_file`` entries (hash-checked) to
   find the actually packed document ranks.  Ranks are joined with the
   document list **inside that corpus only**.  The regular format uses
   ``document_list_file/document_list_sha256``; ``fixed-depth-confirmation-v1``
   is read through an explicit adapter for ``confirmation_documents.jsonl``
   (hash-checked via ``confirmation_documents_sha256``) where the in-line
   ``rank`` field is authoritative, not the file line order, and selected
   documents that were never packed do not count as used.  Unknown formats,
   missing declared files, hash mismatches, and rank join failures are hard
   errors.
3. ``additional_exclusion_jsonl`` files carry already-identified historical
   exposures without a full corpus directory; each file is hash-recorded and
   every row must carry at least one of ``doc_hash``/``text_hash``.
4. Historical documents lacking ``near_hash`` are rebuilt by a **complete**
   scan of the new raw source *before* any exclusion or role assignment, so a
   near-duplicate variant appearing earlier in the source than the original
   is still excluded.  Unresolvable entries mark the corpus
   ``development_only``; they never silently pass as independent.
5. The pre-filtered source is written as an immutable filtered JSONL and the
   existing ``data.controller_corpus.prepare_controller_corpus`` builder is
   called with a factory that re-opens that file on every pass.  Duplicate
   ``doc_hash`` inputs keep the first occurrence and are counted.
6. Everything is written to a staging directory and committed to the target
   only after all validations pass (including actual >= requested tokens for
   all four roles, which the underlying builder does not enforce for
   ``controller_train``).  A non-empty output directory is refused.

The confirmation split of the produced corpus is only a candidate: when
``development_only`` is set, downstream ``--split newtest`` confirmation must
refuse this corpus until the unresolved identities are rebuilt.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from data.controller_corpus import (
    ROLES,
    near_duplicate_hash,
    prepare_controller_corpus,
)
from data.fineweb import (
    doc_fingerprint,
    load_tokenizer,
    sha256_file,
    text_fingerprint,
)

CORPUS_BUILD_VERSION = "rl-lm-corpus-build-v1"
CONFIRMATION_PIPELINE_VERSION = "fixed-depth-confirmation-v1"
FILTERED_FILENAME = "filtered_documents.jsonl"
COVERAGE_FILENAME = "exclusion_coverage.json"


class CorpusBuildError(RuntimeError):
    """Hard failure of any corpus identity, exclusion, or budget check."""


# --------------------------------------------------------------------------
# Strict raw source reading
# --------------------------------------------------------------------------

def read_source_jsonl(path: Union[str, Path]) -> Iterator[Tuple[str, str]]:
    """Stream ``(doc_id, text)`` from a UTF-8 JSONL with string id/text.

    Unknown shapes are hard errors; ids are never silently coerced.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusBuildError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or "id" not in row or "text" not in row:
                raise CorpusBuildError(f"{path}:{lineno}: each row needs 'id' and 'text'")
            if not isinstance(row["id"], str) or not isinstance(row["text"], str):
                raise CorpusBuildError(f"{path}:{lineno}: 'id' and 'text' must be strings")
            yield row["id"], row["text"]


@dataclass
class SourceScan:
    """Complete pass over the raw source, before exclusion or assignment."""

    records: List[Dict[str, object]]  # first occurrence per doc_hash, source order
    n_lines: int
    n_duplicate_doc_hash: int
    near_by_doc_hash: Dict[str, str]
    near_by_text_hash: Dict[str, str]


def scan_source(path: Union[str, Path]) -> SourceScan:
    """Hash every raw line; keep the first occurrence of each doc_hash."""
    records: List[Dict[str, object]] = []
    seen: Set[str] = set()
    n_lines = 0
    n_duplicate = 0
    near_by_doc: Dict[str, str] = {}
    near_by_text: Dict[str, str] = {}
    for doc_id, text in read_source_jsonl(path):
        n_lines += 1
        text_hash = text_fingerprint(text)
        doc_hash = doc_fingerprint(doc_id, text)
        near_hash = near_duplicate_hash(text)
        near_by_doc.setdefault(doc_hash, near_hash)
        near_by_text.setdefault(text_hash, near_hash)
        if doc_hash in seen:
            n_duplicate += 1
            continue
        seen.add(doc_hash)
        records.append(
            {
                "id": doc_id,
                "text": text,
                "doc_hash": doc_hash,
                "text_hash": text_hash,
                "near_hash": near_hash,
            }
        )
    return SourceScan(records, n_lines, n_duplicate, near_by_doc, near_by_text)


# --------------------------------------------------------------------------
# Exclusion collection from historical corpus directories
# --------------------------------------------------------------------------

def _read_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise CorpusBuildError(f"missing {what}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusBuildError(f"invalid JSON in {what} {path}: {exc}") from exc


def _verify_sha(path: Path, expected: object, what: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise CorpusBuildError(f"{what} has no declared sha256 for {path.name}")
    if not path.is_file():
        raise CorpusBuildError(f"declared {what} file is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise CorpusBuildError(
            f"{what} hash mismatch for {path}: manifest says {expected}, actual {actual}"
        )
    return actual


def _used_ranks_from_doc_maps(corpus_dir: Path, manifest: dict) -> Tuple[Set[int], List[str]]:
    """Collect segment doc integers from every declared split doc map."""
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise CorpusBuildError(f"{corpus_dir}: manifest declares no splits")
    used: Set[int] = set()
    names: List[str] = []
    for split_name in sorted(splits):
        info = splits[split_name]
        if not isinstance(info, dict) or "doc_map_file" not in info:
            raise CorpusBuildError(
                f"{corpus_dir}: split {split_name!r} declares no doc_map_file"
            )
        doc_map = corpus_dir / str(info["doc_map_file"])
        _verify_sha(doc_map, info.get("doc_map_sha256"), f"doc map of split {split_name!r}")
        names.append(split_name)
        with open(doc_map, "r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    segments = row["segments"]
                except (json.JSONDecodeError, KeyError) as exc:
                    raise CorpusBuildError(
                        f"{doc_map}:{lineno}: malformed doc map row: {exc}"
                    ) from exc
                for segment in segments:
                    try:
                        used.add(int(segment["doc"]))
                    except (KeyError, TypeError, ValueError) as exc:
                        raise CorpusBuildError(
                            f"{doc_map}:{lineno}: segment without integer doc: {exc}"
                        ) from exc
    return used, names


def _rows_by_rank(rows: Iterator[dict], path: Path, what: str) -> Dict[int, dict]:
    by_rank: Dict[int, dict] = {}
    for lineno, row in enumerate(rows, start=1):
        if "rank" not in row:
            raise CorpusBuildError(f"{path}:{lineno}: {what} row has no in-line rank")
        try:
            rank = int(row["rank"])
        except (TypeError, ValueError) as exc:
            raise CorpusBuildError(f"{path}:{lineno}: non-integer rank: {exc}") from exc
        if rank in by_rank:
            raise CorpusBuildError(f"{path}:{lineno}: duplicate rank {rank}")
        by_rank[rank] = row
    return by_rank


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusBuildError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise CorpusBuildError(f"{path}:{lineno}: expected a JSON object")
            yield row


def _document_table(corpus_dir: Path, manifest: dict) -> Tuple[Dict[int, dict], str, str, str]:
    """Locate this corpus' document table; returns (rows_by_rank, format, file, sha)."""
    if "document_list_file" in manifest:
        doc_path = corpus_dir / str(manifest["document_list_file"])
        sha = _verify_sha(doc_path, manifest.get("document_list_sha256"), "document list")
        rows = _rows_by_rank(_iter_jsonl(doc_path), doc_path, "document list")
        return rows, str(manifest.get("pipeline_version", "document-list-v1")), doc_path.name, sha
    if (
        manifest.get("pipeline_version") == CONFIRMATION_PIPELINE_VERSION
        or "confirmation_documents_sha256" in manifest
    ):
        # fixed-depth-confirmation-v1 adapter: the table is not written in rank
        # order, so the in-line rank field is the only join key; the selected
        # document count is not the packed/used count.
        doc_path = corpus_dir / "confirmation_documents.jsonl"
        sha = _verify_sha(
            doc_path, manifest.get("confirmation_documents_sha256"), "confirmation documents"
        )
        rows = _rows_by_rank(_iter_jsonl(doc_path), doc_path, "confirmation documents")
        return rows, CONFIRMATION_PIPELINE_VERSION, doc_path.name, sha
    raise CorpusBuildError(
        f"{corpus_dir}: unknown corpus format; neither document_list_file nor "
        f"{CONFIRMATION_PIPELINE_VERSION} confirmation_documents.jsonl is declared"
    )


def _row_hashes(row: dict, path: str, rank: int) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    doc_hash = row.get("doc_hash")
    text_hash = row.get("text_hash")
    near_hash = row.get("near_hash")
    if not doc_hash and not text_hash:
        raise CorpusBuildError(
            f"{path}: document row rank={rank} carries neither doc_hash nor text_hash"
        )
    return (
        str(doc_hash) if doc_hash else None,
        str(text_hash) if text_hash else None,
        str(near_hash) if near_hash else None,
    )


@dataclass
class ExclusionSets:
    doc_hashes: Set[str] = field(default_factory=set)
    text_hashes: Set[str] = field(default_factory=set)
    near_hashes: Set[str] = field(default_factory=set)
    pending_near: List[Dict[str, Optional[str]]] = field(default_factory=list)
    sources: List[Dict[str, object]] = field(default_factory=list)
    additional_files: List[Dict[str, object]] = field(default_factory=list)

    def add_row(
        self,
        doc_hash: Optional[str],
        text_hash: Optional[str],
        near_hash: Optional[str],
        origin: str,
    ) -> bool:
        """Record one historical document; returns True when near_hash is missing."""
        if doc_hash:
            self.doc_hashes.add(doc_hash)
        if text_hash:
            self.text_hashes.add(text_hash)
        if near_hash:
            self.near_hashes.add(near_hash)
            return False
        self.pending_near.append(
            {"doc_hash": doc_hash, "text_hash": text_hash, "origin": origin}
        )
        return True


def _collect_corpus_dir(corpus_dir: Path, exclusions: ExclusionSets) -> None:
    if not corpus_dir.is_dir():
        raise CorpusBuildError(f"exclude corpus dir does not exist: {corpus_dir}")
    manifest_path = corpus_dir / "manifest.json"
    manifest = _read_json(manifest_path, "corpus manifest")
    manifest_sha = sha256_file(manifest_path)

    used_ranks, split_names = _used_ranks_from_doc_maps(corpus_dir, manifest)
    by_rank, fmt, table_file, table_sha = _document_table(corpus_dir, manifest)

    n_missing_near = 0
    n_used = 0
    by_doc_hash = {
        str(row["doc_hash"]): row for row in by_rank.values() if row.get("doc_hash")
    }
    for rank in sorted(used_ranks):
        row = by_rank.get(rank)
        if row is None:
            raise CorpusBuildError(
                f"{corpus_dir}: doc map rank {rank} cannot be joined to {table_file}; "
                "ranks are only valid inside their own corpus"
            )
        doc_hash, text_hash, near_hash = _row_hashes(row, str(corpus_dir), rank)
        n_missing_near += exclusions.add_row(doc_hash, text_hash, near_hash, str(corpus_dir))
        n_used += 1

    # The tokenizer training documents are exposed as well (fineweb format).
    tokenizer_doc_hashes = (
        manifest.get("tokenizer_corpus", {}) or {}
    ).get("doc_hashes", []) or []
    n_tokenizer = 0
    for doc_hash in tokenizer_doc_hashes:
        doc_hash = str(doc_hash)
        row = by_doc_hash.get(doc_hash)
        if row is None:
            # Only the content identity is available; still exclude by doc_hash.
            n_missing_near += exclusions.add_row(doc_hash, None, None, str(corpus_dir))
        else:
            _, text_hash, near_hash = _row_hashes(row, str(corpus_dir), int(row["rank"]))
            n_missing_near += exclusions.add_row(doc_hash, text_hash, near_hash, str(corpus_dir))
        n_tokenizer += 1

    exclusions.sources.append(
        {
            "path": str(corpus_dir),
            "format": fmt,
            "manifest_sha256": manifest_sha,
            "document_table_file": table_file,
            "document_table_sha256": table_sha,
            "splits_traversed": split_names,
            "n_used_documents": n_used,
            "n_tokenizer_corpus_doc_hashes": n_tokenizer,
            "n_missing_near_hash": n_missing_near,
        }
    )


def _collect_additional(path: Path, exclusions: ExclusionSets) -> None:
    if not path.is_file():
        raise CorpusBuildError(f"additional exclusion file does not exist: {path}")
    file_sha = sha256_file(path)
    n_rows = 0
    n_missing_near = 0
    for lineno, row in enumerate(_iter_jsonl(path), start=1):
        doc_hash = row.get("doc_hash")
        text_hash = row.get("text_hash")
        if not doc_hash and not text_hash:
            raise CorpusBuildError(
                f"{path}:{lineno}: exclusion row carries neither doc_hash nor text_hash"
            )
        near_hash = row.get("near_hash")
        n_missing_near += exclusions.add_row(
            str(doc_hash) if doc_hash else None,
            str(text_hash) if text_hash else None,
            str(near_hash) if near_hash else None,
            str(path),
        )
        n_rows += 1
    exclusions.additional_files.append(
        {
            "path": str(path),
            "sha256": file_sha,
            "n_rows": n_rows,
            "n_missing_near_hash": n_missing_near,
        }
    )


def collect_exclusions(
    exclude_corpus_dirs: Sequence[Union[str, Path]],
    additional_exclusion_jsonl: Sequence[Union[str, Path]] = (),
) -> ExclusionSets:
    """Union doc/text/near hash identities of every historical exposure.

    Doc integers from doc maps are joined with in-corpus document tables only;
    hashes are the sole cross-corpus identity.  Every declared file is
    hash-verified; unknown formats and join failures raise CorpusBuildError.
    """
    exclusions = ExclusionSets()
    for directory in exclude_corpus_dirs:
        _collect_corpus_dir(Path(directory), exclusions)
    for path in additional_exclusion_jsonl:
        _collect_additional(Path(path), exclusions)
    return exclusions


def rebuild_near_hashes(
    exclusions: ExclusionSets, scan: SourceScan
) -> Dict[str, int]:
    """Rebuild missing historical near_hash values from a complete source scan.

    The scan must cover the **entire** raw source (not post-filter candidates)
    so that a variant appearing before its original still resolves the group.
    Returns rebuild statistics; ``n_unresolved > 0`` means independence cannot
    be confirmed and the corpus must stay ``development_only``.
    """
    rebuilt = 0
    unresolved: List[Dict[str, Optional[str]]] = []
    for entry in exclusions.pending_near:
        near_hash = None
        if entry["doc_hash"]:
            near_hash = scan.near_by_doc_hash.get(entry["doc_hash"])
        if near_hash is None and entry["text_hash"]:
            near_hash = scan.near_by_text_hash.get(entry["text_hash"])
        if near_hash is None:
            unresolved.append(entry)
        else:
            exclusions.near_hashes.add(near_hash)
            rebuilt += 1
    exclusions.pending_near = unresolved
    return {
        "n_pending": rebuilt + len(unresolved),
        "n_rebuilt": rebuilt,
        "n_unresolved": len(unresolved),
    }


# --------------------------------------------------------------------------
# Filtered source and staged build
# --------------------------------------------------------------------------

def write_filtered_jsonl(
    records: Sequence[Dict[str, object]], exclusions: ExclusionSets, out_path: Path
) -> Dict[str, int]:
    """Write the immutable pre-filtered source; returns filter statistics."""
    n_kept = 0
    with open(out_path, "w", encoding="utf-8") as handle:
        for record in records:
            if (
                record["doc_hash"] in exclusions.doc_hashes
                or record["text_hash"] in exclusions.text_hashes
                or record["near_hash"] in exclusions.near_hashes
            ):
                continue
            handle.write(
                json.dumps(
                    {"id": record["id"], "text": record["text"]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            n_kept += 1
    return {"n_kept": n_kept, "n_excluded": len(records) - n_kept}


def _role_intersections(staging: Path, exclusions: ExclusionSets) -> Dict[str, Dict[str, int]]:
    """Verify the built roles never intersect the exclusion identities."""
    intersections = {
        role: {"doc_hash": 0, "text_hash": 0, "near_hash": 0} for role in ROLES
    }
    for row in _iter_jsonl(staging / "documents.jsonl"):
        role = row.get("role")
        if role not in intersections:
            continue
        if row.get("doc_hash") in exclusions.doc_hashes:
            intersections[role]["doc_hash"] += 1
        if row.get("text_hash") in exclusions.text_hashes:
            intersections[role]["text_hash"] += 1
        if row.get("near_hash") in exclusions.near_hashes:
            intersections[role]["near_hash"] += 1
    return intersections


def build_rl_corpus(config, *, tokenizer=None) -> Dict[str, object]:
    """Build the four-role RL corpus from a CorpusConfig (rl_lm.config).

    Stages into ``<output_dir>.staging`` and commits only after every check
    passes.  Returns the coverage report (also written to
    ``exclusion_coverage.json`` inside the corpus directory).
    """
    config.validate_required()
    data = config.data
    output_dir = Path(data["output_dir"])
    if output_dir.is_file():
        raise CorpusBuildError(f"output path is an existing file: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CorpusBuildError(f"output directory is not empty, refusing to overwrite: {output_dir}")

    source_path = Path(data["source_jsonl"])
    if not source_path.is_file():
        raise CorpusBuildError(f"source JSONL does not exist: {source_path}")
    tokenizer_path = Path(data["tokenizer_path"])
    if not tokenizer_path.is_file():
        raise CorpusBuildError(f"tokenizer file does not exist: {tokenizer_path}")
    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_path)
    source_sha = sha256_file(source_path)

    exclusions = collect_exclusions(
        data["exclude_corpus_dirs"], data["additional_exclusion_jsonl"]
    )
    scan = scan_source(source_path)
    rebuild_stats = rebuild_near_hashes(exclusions, scan)
    development_only = rebuild_stats["n_unresolved"] > 0

    staging = output_dir.parent / (output_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        filtered_path = staging / FILTERED_FILENAME
        filter_stats = write_filtered_jsonl(scan.records, exclusions, filtered_path)
        filtered_sha = sha256_file(filtered_path)
        if filter_stats["n_kept"] == 0:
            raise CorpusBuildError("exclusion removed every source document")

        def doc_source() -> Iterator[Tuple[str, str]]:
            # Factory re-opens the immutable filtered file on every pass so
            # each builder scan sees identical bytes in identical order.
            return read_source_jsonl(filtered_path)

        tokens = data["tokens"]
        manifest = prepare_controller_corpus(
            doc_source,
            staging,
            tokenizer=tokenizer,
            revision=data["source_revision"],
            source=data["source_id"],
            seq_len=int(data["sequence_length"]),
            selection_tokens=int(tokens["selection"]),
            calibration_tokens=int(tokens["calibration"]),
            newtest_tokens=int(tokens["newtest"]),
            controller_train_tokens=int(tokens["controller_train"]),
            exclude_documents=None,
            tokenizer_path=tokenizer_path,
            tokenizer_info={"via": "rl_lm.corpus", "filtered_source_sha256": filtered_sha},
        )

        role_tokens: Dict[str, Dict[str, int]] = {}
        shortfalls: List[str] = []
        for role in ROLES:
            requested = int(tokens[role])
            actual = int(manifest["splits"][role]["n_tokens"])
            role_tokens[role] = {"requested": requested, "actual": actual}
            if actual < requested:
                shortfalls.append(
                    f"{role}: actual {actual} < requested {requested} "
                    f"(shortfall {requested - actual})"
                )
        if shortfalls:
            raise CorpusBuildError(
                "insufficient tokens for requested roles; samples are never "
                "duplicated to fill the gap: " + "; ".join(shortfalls)
            )

        intersections = _role_intersections(staging, exclusions)
        bad = {
            role: counts for role, counts in intersections.items() if any(counts.values())
        }
        if bad:
            raise CorpusBuildError(f"excluded identities leaked into built roles: {bad}")

        coverage: Dict[str, object] = {
            "version": CORPUS_BUILD_VERSION,
            "spec_version": data["spec_version"],
            "development_only": development_only,
            "source": {
                "path": str(source_path),
                "sha256": source_sha,
                "source_id": data["source_id"],
                "revision": data["source_revision"],
            },
            "filtered_documents": {
                "file": FILTERED_FILENAME,
                "sha256": filtered_sha,
                "n_documents": filter_stats["n_kept"],
            },
            "n_source_lines": scan.n_lines,
            "n_source_documents": len(scan.records),
            "n_duplicate_doc_hash": scan.n_duplicate_doc_hash,
            "n_excluded_documents": filter_stats["n_excluded"],
            "exclusion_sets": {
                "n_doc_hashes": len(exclusions.doc_hashes),
                "n_text_hashes": len(exclusions.text_hashes),
                "n_near_hashes": len(exclusions.near_hashes),
            },
            "near_hash_rebuild": rebuild_stats,
            "unresolved_near_hash": exclusions.pending_near,
            "exclusion_sources": exclusions.sources,
            "additional_exclusion_files": exclusions.additional_files,
            "role_intersections": intersections,
            "role_tokens": role_tokens,
        }
        coverage_path = staging / COVERAGE_FILENAME
        coverage_path.write_text(
            json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        manifest["rl_lm_corpus"] = {
            "build_version": CORPUS_BUILD_VERSION,
            "spec_version": data["spec_version"],
            "source_jsonl": str(source_path),
            "source_jsonl_sha256": source_sha,
            "filtered_documents_file": FILTERED_FILENAME,
            "filtered_documents_sha256": filtered_sha,
            "exclusion_coverage_file": COVERAGE_FILENAME,
            "exclusion_coverage_sha256": sha256_file(coverage_path),
            "development_only": development_only,
            "n_duplicate_doc_hash": scan.n_duplicate_doc_hash,
            "config_sha256": config.sha256,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if output_dir.exists():
        os.rmdir(output_dir)  # only reachable when empty; non-empty was refused
    os.rename(staging, output_dir)
    return coverage
