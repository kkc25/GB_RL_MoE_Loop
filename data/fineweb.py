"""FineWeb-Edu data pipeline for the 126M/280M LM experiments (plan sec. 8.1 B).

Source: HuggingFaceFW/fineweb-edu, config ``sample-10BT``, pinned to an
explicit dataset revision (never guessed; the CLI requires ``--revision``).

Pipeline invariants (plan sec. 8.1 B):

1. Splits are assigned by document identity / de-duplication fingerprint
   *before* the tokenizer is fitted.  Validation and test documents never
   enter the tokenizer training corpus.  Documents whose normalized texts
   collide (duplicates) always land in the same split.
2. The tokenizer is a byte-level BPE trained on a fixed budget (default
   200 MB) of UTF-8 text taken deterministically from training documents
   only.  Vocabulary size is 32768 *including* special tokens.
3. Validation and test are exactly ``val_tokens`` / ``test_tokens`` tokens
   (default 5M each): the deterministic first tokens of the documents
   assigned to those splits.  Training uses the deterministic first
   ``train_tokens`` tokens of the remaining documents (default: all).
4. Sequences have length ``seq_len`` (default 1024).  Packing preserves
   document boundaries: every token carries a per-sequence segment id
   (-1 = padding), and a per-sequence doc map records
   ``(document index, token offset, length)`` for each segment, which is
   everything a block-diagonal attention mask and per-document loss
   accounting need.
5. Provenance: the manifest records the dataset revision, the full document
   list (id, fingerprints, split) plus its sha256, the tokenizer file
   sha256, the complete script configuration, library versions, and the
   actual token counts per split.

Everything in this module is local-file / in-memory only; network access
lives in ``data/prepare_fineweb.py``.  ``prepare_documents`` accepts a
``doc_source`` factory (a callable returning a fresh iterator of
``(doc_id, text)``) so the full pipeline is testable on synthetic
documents without touching the network.
"""

from __future__ import annotations

import hashlib
import json
import platform
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

REPO_ID = "HuggingFaceFW/fineweb-edu"
SOURCE_CONFIG = "sample-10BT"

SEQ_LEN = 1024
VOCAB_SIZE = 32768
TOKENIZER_TRAIN_BYTES = 200_000_000  # 200 MB of UTF-8 text
VAL_TOKENS = 5_000_000
TEST_TOKENS = 5_000_000

# Rough UTF-8-bytes-per-token estimate for English web text with a byte-level
# BPE, times a safety margin, used only to size the validation/test document
# pools *before* the tokenizer exists.  The final counts are enforced exactly
# after tokenization (take_first_tokens) and an error is raised if the pool
# turns out too small.
BYTES_PER_TOKEN_ESTIMATE = 4.0
SPLIT_POOL_SAFETY = 3.0

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>"]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

PIPELINE_VERSION = "fineweb-pipeline-v1"
_SPLIT_SALT = b"fineweb-split-v1\x00"

DocRecord = Dict[str, object]
DocSource = Callable[[], Iterator[Tuple[str, str]]]


# --------------------------------------------------------------------------
# Document identity and de-duplication fingerprints
# --------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Canonical form used for the de-duplication fingerprint.

    NFC-normalized, with every whitespace run collapsed to a single space
    and leading/trailing whitespace removed.  Case is preserved: the
    fingerprint is meant to catch (near-)exact duplicates, not paraphrases.
    """
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def text_fingerprint(text: str) -> str:
    """sha256 of the normalized text; identical texts share a split."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def doc_fingerprint(doc_id: str, text: str) -> str:
    """sha256 binding the source document id to its text fingerprint."""
    h = hashlib.sha256()
    h.update(str(doc_id).encode("utf-8"))
    h.update(b"\x00")
    h.update(text_fingerprint(text).encode("ascii"))
    return h.hexdigest()


def split_key(text_fp: str) -> int:
    """Deterministic uniform key in [0, 2**256) for split assignment."""
    return int.from_bytes(hashlib.sha256(_SPLIT_SALT + text_fp.encode("ascii")).digest(), "big")


def build_doc_records(docs: Iterable[Tuple[str, str]]) -> List[DocRecord]:
    """One record per document, in source order (rank = order of appearance).

    The rank defines the deterministic within-split ordering: it is fixed by
    the pinned revision's shard order, not by Python hash randomization.
    """
    records = []
    for rank, (doc_id, text) in enumerate(docs):
        text_fp = text_fingerprint(text)
        records.append(
            {
                "rank": rank,
                "id": str(doc_id),
                "text_hash": text_fp,
                "doc_hash": doc_fingerprint(str(doc_id), text),
                "n_bytes": len(text.encode("utf-8")),
            }
        )
    return records


def assign_splits(
    records: Sequence[DocRecord],
    *,
    val_bytes: int,
    test_bytes: int,
) -> Dict[str, List[DocRecord]]:
    """Assign each document to train/validation/test by fingerprint.

    Assignment is keyed on the *text* fingerprint so that duplicate texts
    (different ids, same content) always share a split.  Unique texts are
    ordered by ``split_key``; the lowest-keyed texts fill the test pool until
    ``test_bytes`` of UTF-8 text are covered, then the validation pool until
    ``val_bytes``, and everything else is training data.  Within each split
    the documents keep their source (rank) order.

    The pools deliberately over-request bytes (see BYTES_PER_TOKEN_ESTIMATE
    and SPLIT_POOL_SAFETY); the exact token budgets are enforced later, after
    tokenization.
    """
    by_text: Dict[str, dict] = {}
    for rec in records:
        group = by_text.setdefault(
            rec["text_hash"], {"key": split_key(rec["text_hash"]), "n_bytes": 0}
        )
        group["n_bytes"] += rec["n_bytes"]

    split_of_text: Dict[str, str] = {}
    test_seen = val_seen = 0
    for text_fp, group in sorted(by_text.items(), key=lambda kv: (kv[1]["key"], kv[0])):
        if test_seen < test_bytes:
            split_of_text[text_fp] = "test"
            test_seen += group["n_bytes"]
        elif val_seen < val_bytes:
            split_of_text[text_fp] = "validation"
            val_seen += group["n_bytes"]
        else:
            split_of_text[text_fp] = "train"

    splits: Dict[str, List[DocRecord]] = {"train": [], "validation": [], "test": []}
    for rec in records:
        splits[split_of_text[rec["text_hash"]]].append(rec)
    return splits


# --------------------------------------------------------------------------
# Tokenizer: byte-level BPE on a fixed budget of training text
# --------------------------------------------------------------------------

def collect_tokenizer_corpus(
    docs: Iterable[Tuple[str, str]],
    train_hashes: set,
    max_bytes: int,
) -> Tuple[List[str], List[str], int]:
    """Deterministic first ``max_bytes`` of UTF-8 text from training docs.

    ``docs`` must be iterated in the same deterministic (source) order used
    for ``build_doc_records``.  Returns (texts, doc_hashes_used, total_bytes).
    The final document is truncated on a UTF-8 character boundary when it
    would exceed the budget.
    """
    texts: List[str] = []
    used: List[str] = []
    total = 0
    for doc_id, text in docs:
        if total >= max_bytes:
            break
        fp = doc_fingerprint(str(doc_id), text)
        if fp not in train_hashes:
            continue
        data = text.encode("utf-8")
        remaining = max_bytes - total
        if len(data) > remaining:
            # Cut on a character boundary; the truncated tail is dropped.
            data = data[:remaining]
            text = data.decode("utf-8", errors="ignore")
            data = text.encode("utf-8")
        texts.append(text)
        used.append(fp)
        total += len(data)
    return texts, used, total


def train_byte_bpe(
    texts: Iterable[str],
    out_path: Path,
    *,
    vocab_size: int = VOCAB_SIZE,
    special_tokens: Sequence[str] = SPECIAL_TOKENS,
) -> Dict[str, object]:
    """Train a byte-level BPE and save it as ``out_path`` (tokenizer.json).

    ``vocab_size`` includes the special tokens, which occupy the first ids in
    the order given (asserted after training).  Training with the
    ``tokenizers`` library is deterministic for a fixed corpus, vocabulary
    size, and library version.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(iter(texts), trainer=trainer)

    for expected_id, tok in enumerate(special_tokens):
        actual = tokenizer.token_to_id(tok)
        if actual != expected_id:
            raise RuntimeError(
                f"special token {tok!r} got id {actual}, expected {expected_id}"
            )
    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab > vocab_size:
        raise RuntimeError(f"trained vocab {actual_vocab} exceeds requested {vocab_size}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    import tokenizers as _tk

    return {
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "vocab_size": actual_vocab,
        "special_tokens": list(special_tokens),
        "tokenizers_version": _tk.__version__,
    }


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def encode_document(tokenizer, text: str) -> List[int]:
    """Token ids for one document: [BOS] + content tokens + [EOS]."""
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return [BOS_ID] + ids + [EOS_ID]


# --------------------------------------------------------------------------
# Token budgets and packing
# --------------------------------------------------------------------------

def take_first_tokens(
    encoded_docs: Iterable[Tuple[int, List[int]]],
    max_tokens: int,
) -> Tuple[List[Tuple[int, List[int]]], int]:
    """Deterministic first ``max_tokens`` tokens of an encoded document stream.

    The last document is truncated to its prefix when it crosses the budget.
    Returns (trimmed docs, actual token count).
    """
    out: List[Tuple[int, List[int]]] = []
    total = 0
    for doc_index, tokens in encoded_docs:
        if total >= max_tokens:
            break
        piece = tokens[: max_tokens - total]
        if piece:
            out.append((doc_index, piece))
            total += len(piece)
    return out, total


@dataclass
class PackedSequence:
    input_ids: np.ndarray      # (seq_len,) uint16
    segment_ids: np.ndarray    # (seq_len,) int32; -1 = padding
    segments: List[dict]       # [{"doc": rank, "offset": int, "length": int}]


class DocumentPacker:
    """Incremental packing of encoded documents into fixed-length sequences.

    Documents are concatenated in the order they are added and cut at exactly
    ``seq_len`` tokens; a document may span consecutive sequences.  Every
    token belongs to exactly one segment whose doc index (rank) and
    within-document offset are recorded, so a block-diagonal (document
    boundary) attention mask and per-document losses can be reconstructed.
    Completed sequences are handed to the ``emit`` callback.
    """

    def __init__(self, seq_len: int, emit: Callable[[PackedSequence], None], pad_id: int = PAD_ID):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.seq_len = seq_len
        self.pad_id = pad_id
        self._emit = emit
        self._ids: List[int] = []
        self._segments: List[dict] = []
        self.n_tokens = 0  # real (non-padding) tokens accepted so far

    def add(self, doc_index: int, tokens: Sequence[int]) -> None:
        offset, n = 0, len(tokens)
        while offset < n:
            take = min(self.seq_len - len(self._ids), n - offset)
            self._segments.append({"doc": int(doc_index), "offset": offset, "length": take})
            self._ids.extend(tokens[offset : offset + take])
            offset += take
            self.n_tokens += take
            if len(self._ids) == self.seq_len:
                self._flush(padding=0)

    def _flush(self, padding: int) -> None:
        seg_index: List[int] = []
        for pos, seg in enumerate(self._segments):
            seg_index.extend([pos] * seg["length"])
        sequence = PackedSequence(
            input_ids=np.asarray(self._ids + [self.pad_id] * padding, dtype=np.uint16),
            segment_ids=np.asarray(seg_index + [-1] * padding, dtype=np.int32),
            segments=[dict(seg) for seg in self._segments],
        )
        self._ids = []
        self._segments = []
        self._emit(sequence)

    def finish(self, keep_partial: bool = True) -> Dict[str, int]:
        """Flush the final partial sequence; returns padding/drop counts."""
        n_padding = n_dropped = 0
        if self._ids:
            if keep_partial:
                n_padding = self.seq_len - len(self._ids)
                self._flush(padding=n_padding)
            else:
                n_dropped = len(self._ids)
                self.n_tokens -= n_dropped
                self._ids = []
                self._segments = []
        return {"n_padding": n_padding, "n_dropped_tokens": n_dropped}


def pack_documents(
    encoded_docs: Iterable[Tuple[int, Sequence[int]]],
    seq_len: int,
    *,
    keep_partial: bool = True,
    pad_id: int = PAD_ID,
) -> Tuple[List[PackedSequence], Dict[str, int]]:
    """In-memory packing (see DocumentPacker); for tests and small splits.

    The final partial sequence is kept (padded, segment id -1) when
    ``keep_partial`` is set, which is how validation/test hit their exact
    token budgets.
    """
    sequences: List[PackedSequence] = []
    packer = DocumentPacker(seq_len, sequences.append, pad_id=pad_id)
    for doc_index, tokens in encoded_docs:
        if tokens:
            packer.add(doc_index, tokens)
    tail = packer.finish(keep_partial=keep_partial)
    stats = {
        "n_tokens": packer.n_tokens,
        "n_sequences": len(sequences),
        "n_padding": tail["n_padding"],
        "n_dropped_tokens": tail["n_dropped_tokens"],
    }
    return sequences, stats


class SplitWriter:
    """Streaming split writer: budget enforcement + packing + disk output.

    Accepts documents one at a time in deterministic rank order, keeps at
    most ``max_tokens`` real tokens (trimming the crossing document to its
    prefix), packs them into ``seq_len`` sequences, and writes:

      ``{split}.input_ids.u16.bin``   (n_sequences, seq_len) uint16
      ``{split}.segment_ids.i32.bin`` (n_sequences, seq_len) int32, -1 = pad
      ``{split}.doc_map.jsonl``       one line per sequence:
                                      {"seq": i, "segments": [{doc, offset, length}, ...]}

    Memory use is O(seq_len) regardless of split size.
    """

    def __init__(self, out_dir: Path, split: str, seq_len: int, max_tokens: Optional[int] = None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self._ids_path = out_dir / f"{split}.input_ids.u16.bin"
        self._seg_path = out_dir / f"{split}.segment_ids.i32.bin"
        self._map_path = out_dir / f"{split}.doc_map.jsonl"
        self._f_ids = open(self._ids_path, "wb")
        self._f_seg = open(self._seg_path, "wb")
        self._f_map = open(self._map_path, "w", encoding="utf-8")
        self._packer = DocumentPacker(seq_len, self._emit)
        self.n_sequences = 0
        self.exhausted = False  # True once max_tokens has been reached
        self._docs: set = set()

    @property
    def n_tokens(self) -> int:
        return self._packer.n_tokens

    def add_document(self, rank: int, tokens: Sequence[int]) -> bool:
        """Add one encoded document; returns False once the budget is full."""
        if self.exhausted:
            return False
        tokens = list(tokens)
        if self.max_tokens is not None:
            remaining = self.max_tokens - self.n_tokens
            if remaining <= 0:
                self.exhausted = True
                return False
            if len(tokens) > remaining:
                tokens = tokens[:remaining]
                self.exhausted = True
        if tokens:
            self._docs.add(int(rank))
            self._packer.add(rank, tokens)
        return not self.exhausted

    def _emit(self, sequence: PackedSequence) -> None:
        self._f_ids.write(sequence.input_ids.tobytes())
        self._f_seg.write(sequence.segment_ids.tobytes())
        self._f_map.write(
            json.dumps({"seq": self.n_sequences, "segments": sequence.segments}) + "\n"
        )
        self.n_sequences += 1

    def finish(self) -> Dict[str, object]:
        tail = self._packer.finish(keep_partial=True)
        self._f_ids.close()
        self._f_seg.close()
        self._f_map.close()
        return {
            "n_sequences": self.n_sequences,
            "seq_len": self.seq_len,
            "n_tokens": self.n_tokens,
            "n_padding": tail["n_padding"],
            "n_documents": len(self._docs),
            "input_ids_file": self._ids_path.name,
            "segment_ids_file": self._seg_path.name,
            "doc_map_file": self._map_path.name,
            "input_ids_dtype": "uint16",
            "segment_ids_dtype": "int32",
            "input_ids_sha256": sha256_file(self._ids_path),
            "segment_ids_sha256": sha256_file(self._seg_path),
            "doc_map_sha256": sha256_file(self._map_path),
        }


# --------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_document_list(records: Sequence[DocRecord], splits: Dict[str, List[DocRecord]], out_path: Path) -> str:
    """Write the full document list (one JSON record per line) and return its sha256."""
    split_of_doc = {
        rec["doc_hash"]: split for split, recs in splits.items() for rec in recs
    }
    out_path = Path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            row = dict(rec)
            row["split"] = split_of_doc[rec["doc_hash"]]
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return sha256_file(out_path)


# --------------------------------------------------------------------------
# End-to-end local pipeline
# --------------------------------------------------------------------------

def prepare_documents(
    doc_source: DocSource,
    out_dir: Path,
    *,
    revision: str,
    source: str = REPO_ID,
    source_config: str = SOURCE_CONFIG,
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_SIZE,
    tokenizer_bytes: int = TOKENIZER_TRAIN_BYTES,
    val_tokens: int = VAL_TOKENS,
    test_tokens: int = TEST_TOKENS,
    train_tokens: Optional[int] = None,
    val_pool_bytes: Optional[int] = None,
    test_pool_bytes: Optional[int] = None,
    source_shards: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Run the full local pipeline and write splits + manifest to ``out_dir``.

    ``doc_source`` is a factory returning a *fresh* iterator of
    ``(doc_id, text)`` in the deterministic source order; it is consumed up
    to three times (records, tokenizer corpus, packing).  ``revision`` is
    recorded verbatim in the manifest; this function never invents one.
    """
    if not revision:
        raise ValueError("an explicit dataset revision is required")
    if vocab_size > 65535:
        raise ValueError("vocab_size must fit uint16 storage")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if val_pool_bytes is None:
        val_pool_bytes = int(val_tokens * BYTES_PER_TOKEN_ESTIMATE * SPLIT_POOL_SAFETY)
    if test_pool_bytes is None:
        test_pool_bytes = int(test_tokens * BYTES_PER_TOKEN_ESTIMATE * SPLIT_POOL_SAFETY)

    # Pass 1: document identities / fingerprints, then split assignment.
    records = build_doc_records(doc_source())
    splits = assign_splits(records, val_bytes=val_pool_bytes, test_bytes=test_pool_bytes)
    doc_list_sha = write_document_list(records, splits, out_dir / "documents.jsonl")

    # Pass 2: tokenizer corpus from training documents only.
    train_hashes = {rec["doc_hash"] for rec in splits["train"]}
    corpus_texts, corpus_doc_hashes, corpus_bytes = collect_tokenizer_corpus(
        doc_source(), train_hashes, tokenizer_bytes
    )
    if not corpus_texts:
        raise RuntimeError("no training documents available for tokenizer fitting")
    tok_info = train_byte_bpe(corpus_texts, out_dir / "tokenizer.json", vocab_size=vocab_size)
    tokenizer = load_tokenizer(tok_info["path"])

    # Pass 3: encode + budget + pack each split in deterministic (source/rank)
    # order.  Documents are streamed and packed incrementally, so memory stays
    # bounded even for multi-billion-token training splits.
    split_stats: Dict[str, dict] = {}
    for split, budget, exact in (
        ("validation", val_tokens, True),
        ("test", test_tokens, True),
        ("train", train_tokens, False),
    ):
        rank_of_hash = {rec["doc_hash"]: int(rec["rank"]) for rec in splits[split]}
        writer = SplitWriter(out_dir, split, seq_len, max_tokens=budget)
        for doc_id, text in doc_source():
            rank = rank_of_hash.get(doc_fingerprint(str(doc_id), text))
            if rank is None:
                continue
            if not writer.add_document(rank, encode_document(tokenizer, text)):
                break  # token budget reached
        info = writer.finish()
        if exact and info["n_tokens"] < budget:
            raise RuntimeError(
                f"split {split!r} only has {info['n_tokens']} tokens, needs {budget}; "
                "increase the split pool bytes or add source data"
            )
        split_stats[split] = info

    manifest: Dict[str, object] = {
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "source_config": source_config,
        "revision": revision,
        "source_shards": list(source_shards) if source_shards else None,
        "config": {
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "tokenizer_bytes": tokenizer_bytes,
            "val_tokens": val_tokens,
            "test_tokens": test_tokens,
            "train_tokens": train_tokens,
            "val_pool_bytes": val_pool_bytes,
            "test_pool_bytes": test_pool_bytes,
            "special_tokens": list(SPECIAL_TOKENS),
            "bytes_per_token_estimate": BYTES_PER_TOKEN_ESTIMATE,
            "split_pool_safety": SPLIT_POOL_SAFETY,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "n_documents": len(records),
        "split_documents": {k: len(v) for k, v in splits.items()},
        "document_list_file": "documents.jsonl",
        "document_list_sha256": doc_list_sha,
        "tokenizer": tok_info,
        "tokenizer_corpus": {
            "n_documents": len(corpus_doc_hashes),
            "n_bytes": corpus_bytes,
            "doc_hashes": corpus_doc_hashes,
        },
        "splits": split_stats,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_raw_docs_jsonl(path: Path) -> Iterator[Tuple[str, str]]:
    """Stream ``(doc_id, text)`` pairs from a JSONL file with id/text fields."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield str(row["id"]), row["text"]
