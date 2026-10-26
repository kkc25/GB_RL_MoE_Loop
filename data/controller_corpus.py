"""Controller-role corpus builder (repair plan R2, audits/code_audit_2026-10-25 sec. 4.1).

Builds the four frozen controller roles -- ``controller_train``, ``selection``,
``calibration``, ``newtest`` -- as packed-LM splits readable by
``data.packed_lm.PackedLMDataset``.  Unlike ``data/fineweb.py`` this module
never fits a tokenizer: the caller supplies an already-trained byte-level BPE
(``tokenizers.Tokenizer``), so split assignment cannot leak held-out text into
a tokenizer training step and the backbone's tokenizer is reused verbatim.

Role assignment invariants:

1. Documents are grouped by a *near-duplicate* fingerprint (case/punctuation/
   whitespace-insensitive, see ``near_duplicate_hash``); exact duplicates share
   a near-duplicate group by construction.  Every document of a group gets the
   same role -- token rows of one document are never scattered across roles.
2. Assignment order is a stable hash of the group fingerprint
   (``role_group_key``), independent of source order and of Python hash
   randomization.  The lowest-keyed groups fill ``newtest`` first (the frozen
   confirmation split), then ``calibration``, then ``selection``; everything
   else is ``controller_train``.
3. Documents listed in an existing ``documents.jsonl`` (e.g. the backbone's
   ``data/fineweb.py`` manifest document list) are excluded by ``doc_hash`` or
   ``text_hash``; exclusion is group-level, so near-duplicates of an excluded
   document are excluded too.  Excluded documents are recorded with role
   ``"excluded"`` in the new document list.

Token budgets for ``selection`` / ``calibration`` / ``newtest`` are enforced
exactly (deterministic first tokens of the role's documents, an error is
raised if the pool is too small); ``controller_train`` takes everything else,
optionally capped.  Output uses ``data.fineweb.SplitWriter`` and a
``manifest.json`` recording the full document list, the exclusion provenance,
the tokenizer identity supplied by the caller, and per-role file hashes.

Everything here is local-file / in-memory only and testable on synthetic
documents without network access.
"""

from __future__ import annotations

import hashlib
import json
import platform
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from data.fineweb import (
    BYTES_PER_TOKEN_ESTIMATE,
    SEQ_LEN,
    SPLIT_POOL_SAFETY,
    DocRecord,
    SplitWriter,
    doc_fingerprint,
    sha256_file,
    text_fingerprint,
)

ROLES = ("controller_train", "selection", "calibration", "newtest")
EXCLUDED_ROLE = "excluded"

PIPELINE_VERSION = "controller-corpus-v1"
_ROLE_SALT = b"controller-split-v1\x00"

BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"

DocSource = Callable[[], Iterator[Tuple[str, str]]]


# --------------------------------------------------------------------------
# Near-duplicate grouping
# --------------------------------------------------------------------------

def normalize_near_text(text: str) -> str:
    """Aggressive canonical form for the near-duplicate fingerprint.

    NFC-normalized, case-folded, every non-alphanumeric character replaced by
    a space, whitespace runs collapsed.  Documents differing only in case,
    punctuation, or whitespace share a near-duplicate group; exact duplicates
    (same ``text_fingerprint``) always share one too.
    """
    text = unicodedata.normalize("NFC", text).casefold()
    chars = [ch if ch.isalnum() else " " for ch in text]
    return " ".join("".join(chars).split())


def near_duplicate_hash(text: str) -> str:
    """sha256 of the aggressive canonical form; the near-duplicate group id."""
    return hashlib.sha256(normalize_near_text(text).encode("utf-8")).hexdigest()


def role_group_key(near_hash: str) -> int:
    """Deterministic uniform key in [0, 2**256) for role assignment."""
    return int.from_bytes(hashlib.sha256(_ROLE_SALT + near_hash.encode("ascii")).digest(), "big")


def build_doc_records(docs: Iterable[Tuple[str, str]]) -> List[DocRecord]:
    """One record per document, in source order (rank = order of appearance).

    Same fields as ``data.fineweb.build_doc_records`` plus ``near_hash``, the
    near-duplicate group fingerprint used for role assignment and for
    group-level exclusion.
    """
    records = []
    for rank, (doc_id, text) in enumerate(docs):
        records.append(
            {
                "rank": rank,
                "id": str(doc_id),
                "text_hash": text_fingerprint(text),
                "doc_hash": doc_fingerprint(str(doc_id), text),
                "near_hash": near_duplicate_hash(text),
                "n_bytes": len(text.encode("utf-8")),
            }
        )
    return records


# --------------------------------------------------------------------------
# Exclusion of previously used / locked documents
# --------------------------------------------------------------------------

def load_exclusion_hashes(paths: Union[str, Path, Sequence[Union[str, Path]]]) -> Dict[str, set]:
    """Read ``doc_hash`` / ``text_hash`` values from existing documents.jsonl files.

    Accepts a single path or a sequence of paths; every line must be a JSON
    object carrying at least one of the two hash fields (the
    ``data/fineweb.py`` document list carries both).
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    doc_hashes: set = set()
    text_hashes: set = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("doc_hash"):
                    doc_hashes.add(str(row["doc_hash"]))
                if row.get("text_hash"):
                    text_hashes.add(str(row["text_hash"]))
    return {"doc_hashes": doc_hashes, "text_hashes": text_hashes}


def mark_excluded(records: Sequence[DocRecord], exclusion: Dict[str, set]) -> List[bool]:
    """Group-level exclusion mask, one flag per record.

    A record is excluded when its own ``doc_hash``/``text_hash`` matches, or
    when any member of its near-duplicate group matches.
    """
    excluded_hashes = {
        rec["near_hash"]
        for rec in records
        if rec["doc_hash"] in exclusion["doc_hashes"] or rec["text_hash"] in exclusion["text_hashes"]
    }
    return [rec["near_hash"] in excluded_hashes for rec in records]


# --------------------------------------------------------------------------
# Role assignment
# --------------------------------------------------------------------------

def assign_roles(
    records: Sequence[DocRecord],
    *,
    newtest_bytes: int,
    calibration_bytes: int,
    selection_bytes: int,
    excluded: Optional[Sequence[bool]] = None,
) -> Dict[str, List[DocRecord]]:
    """Assign each document to a controller role by near-duplicate group.

    Groups are ordered by ``role_group_key``; the lowest-keyed groups fill the
    ``newtest`` pool until ``newtest_bytes`` of UTF-8 text are covered, then
    ``calibration``, then ``selection``; all remaining groups are
    ``controller_train``.  Excluded records never receive a role.  Within each
    role the documents keep their source (rank) order.
    """
    if excluded is None:
        excluded = [False] * len(records)
    by_group: Dict[str, dict] = {}
    for rec, is_excluded in zip(records, excluded):
        if is_excluded:
            continue
        group = by_group.setdefault(
            rec["near_hash"], {"key": role_group_key(rec["near_hash"]), "n_bytes": 0}
        )
        group["n_bytes"] += rec["n_bytes"]

    role_of_group: Dict[str, str] = {}
    seen = {"newtest": 0, "calibration": 0, "selection": 0}
    budgets = {"newtest": newtest_bytes, "calibration": calibration_bytes, "selection": selection_bytes}
    for near_hash, group in sorted(by_group.items(), key=lambda kv: (kv[1]["key"], kv[0])):
        role = "controller_train"
        for candidate in ("newtest", "calibration", "selection"):
            if seen[candidate] < budgets[candidate]:
                role = candidate
                seen[candidate] += group["n_bytes"]
                break
        role_of_group[near_hash] = role

    roles: Dict[str, List[DocRecord]] = {role: [] for role in ROLES}
    for rec, is_excluded in zip(records, excluded):
        if not is_excluded:
            roles[role_of_group[rec["near_hash"]]].append(rec)
    return roles


# --------------------------------------------------------------------------
# Encoding with a caller-supplied tokenizer (never trained here)
# --------------------------------------------------------------------------

def encode_document(
    tokenizer,
    text: str,
    *,
    bos_token: str = BOS_TOKEN,
    eos_token: str = EOS_TOKEN,
) -> List[int]:
    """Token ids for one document: [bos] + content tokens + [eos].

    The tokenizer must already define the two special tokens; this function
    never mutates or trains it.
    """
    bos_id = tokenizer.token_to_id(bos_token)
    eos_id = tokenizer.token_to_id(eos_token)
    if bos_id is None or eos_id is None:
        raise ValueError(
            f"tokenizer must define {bos_token!r} and {eos_token!r}; "
            "pass an already-trained tokenizer with these special tokens"
        )
    ids = [int(bos_id)] + tokenizer.encode(text, add_special_tokens=False).ids + [int(eos_id)]
    if ids and max(ids) > 65535:
        raise ValueError("token ids must fit uint16 packed storage")
    return ids


# --------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------

def write_document_list(
    records: Sequence[DocRecord],
    roles: Dict[str, List[DocRecord]],
    excluded: Sequence[bool],
    out_path: Path,
) -> str:
    """Write the full document list with roles ("excluded" where applicable)."""
    role_of_doc = {
        rec["doc_hash"]: role for role, recs in roles.items() for rec in recs
    }
    out_path = Path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec, is_excluded in zip(records, excluded):
            row = dict(rec)
            row["role"] = EXCLUDED_ROLE if is_excluded else role_of_doc[rec["doc_hash"]]
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return sha256_file(out_path)


# --------------------------------------------------------------------------
# End-to-end local pipeline
# --------------------------------------------------------------------------

def prepare_controller_corpus(
    doc_source: DocSource,
    out_dir: Path,
    *,
    tokenizer,
    revision: str,
    source: str = "controller-corpus",
    seq_len: int = SEQ_LEN,
    selection_tokens: int,
    calibration_tokens: int,
    newtest_tokens: int,
    controller_train_tokens: Optional[int] = None,
    selection_pool_bytes: Optional[int] = None,
    calibration_pool_bytes: Optional[int] = None,
    newtest_pool_bytes: Optional[int] = None,
    exclude_documents: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    tokenizer_path: Optional[Union[str, Path]] = None,
    tokenizer_info: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build the four controller-role splits and write splits + manifest.

    ``doc_source`` is a factory returning a *fresh* iterator of
    ``(doc_id, text)`` in deterministic source order; it is consumed twice
    (records, packing).  ``tokenizer`` is an already-trained BPE supplied by
    the caller and is only ever used for encoding.  ``exclude_documents``
    points at one or more existing ``documents.jsonl`` files whose
    ``doc_hash``/``text_hash`` (and their near-duplicate groups) are kept out
    of every controller role.  ``revision`` is recorded verbatim; this
    function never invents one.
    """
    if not revision:
        raise ValueError("an explicit source revision is required")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if selection_pool_bytes is None:
        selection_pool_bytes = int(selection_tokens * BYTES_PER_TOKEN_ESTIMATE * SPLIT_POOL_SAFETY)
    if calibration_pool_bytes is None:
        calibration_pool_bytes = int(calibration_tokens * BYTES_PER_TOKEN_ESTIMATE * SPLIT_POOL_SAFETY)
    if newtest_pool_bytes is None:
        newtest_pool_bytes = int(newtest_tokens * BYTES_PER_TOKEN_ESTIMATE * SPLIT_POOL_SAFETY)

    # Pass 1: document identities, exclusion, role assignment.
    records = build_doc_records(doc_source())
    if exclude_documents is None:
        exclusion = {"doc_hashes": set(), "text_hashes": set()}
        exclusion_paths: List[str] = []
        exclusion_sha: List[str] = []
    else:
        exclusion = load_exclusion_hashes(exclude_documents)
        paths = [exclude_documents] if isinstance(exclude_documents, (str, Path)) else list(exclude_documents)
        exclusion_paths = [str(p) for p in paths]
        exclusion_sha = [sha256_file(Path(p)) for p in paths]
    excluded = mark_excluded(records, exclusion)
    roles = assign_roles(
        records,
        newtest_bytes=newtest_pool_bytes,
        calibration_bytes=calibration_pool_bytes,
        selection_bytes=selection_pool_bytes,
        excluded=excluded,
    )
    doc_list_sha = write_document_list(records, roles, excluded, out_dir / "documents.jsonl")

    # Pass 2: encode + budget + pack each role in deterministic (rank) order.
    role_stats: Dict[str, dict] = {}
    for role, budget, exact in (
        ("newtest", newtest_tokens, True),
        ("calibration", calibration_tokens, True),
        ("selection", selection_tokens, True),
        ("controller_train", controller_train_tokens, False),
    ):
        rank_of_hash = {rec["doc_hash"]: int(rec["rank"]) for rec in roles[role]}
        writer = SplitWriter(out_dir, role, seq_len, max_tokens=budget)
        for doc_id, text in doc_source():
            rank = rank_of_hash.get(doc_fingerprint(str(doc_id), text))
            if rank is None:
                continue
            if not writer.add_document(rank, encode_document(tokenizer, text)):
                break  # token budget reached
        info = writer.finish()
        if exact and info["n_tokens"] < budget:
            raise RuntimeError(
                f"role {role!r} only has {info['n_tokens']} tokens, needs {budget}; "
                "increase the role pool bytes or add source data"
            )
        role_stats[role] = info

    if tokenizer_info is None:
        tokenizer_info = {}
    else:
        tokenizer_info = dict(tokenizer_info)
    if tokenizer_path is not None:
        tokenizer_info.setdefault("path", str(tokenizer_path))
        tokenizer_info.setdefault("sha256", sha256_file(Path(tokenizer_path)))
    tokenizer_info.setdefault("vocab_size", int(tokenizer.get_vocab_size()))
    tokenizer_info["trained_by"] = "caller"

    manifest: Dict[str, object] = {
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "revision": revision,
        "config": {
            "seq_len": seq_len,
            "controller_train_tokens": controller_train_tokens,
            "selection_tokens": selection_tokens,
            "calibration_tokens": calibration_tokens,
            "newtest_tokens": newtest_tokens,
            "selection_pool_bytes": selection_pool_bytes,
            "calibration_pool_bytes": calibration_pool_bytes,
            "newtest_pool_bytes": newtest_pool_bytes,
            "role_salt": _ROLE_SALT.decode("ascii", errors="replace"),
            "bos_token": BOS_TOKEN,
            "eos_token": EOS_TOKEN,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "n_documents": len(records),
        "n_excluded_documents": int(sum(excluded)),
        "role_documents": {role: len(recs) for role, recs in roles.items()},
        "document_list_file": "documents.jsonl",
        "document_list_sha256": doc_list_sha,
        "exclusion": {
            "documents_files": exclusion_paths,
            "documents_sha256": exclusion_sha,
            "n_doc_hashes": len(exclusion["doc_hashes"]),
            "n_text_hashes": len(exclusion["text_hashes"]),
        },
        "tokenizer": tokenizer_info,
        "splits": role_stats,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
