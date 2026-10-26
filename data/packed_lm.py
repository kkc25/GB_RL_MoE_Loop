"""Memory-mapped reader and deterministic batch sampler for packed LM splits.

Consumes the on-disk format produced by ``data/fineweb.py`` (``SplitWriter`` /
``prepare_documents``):

  ``{split}.input_ids.u16.bin``   (n_sequences, seq_len) uint16
  ``{split}.segment_ids.i32.bin`` (n_sequences, seq_len) int32, -1 = padding
  ``{split}.doc_map.jsonl``       one line per sequence:
                                  {"seq": i, "segments": [{doc, offset, length}, ...]}
  ``manifest.json``               (optional) used to infer ``seq_len`` when it
                                  is not passed explicitly.

``PackedLMDataset`` memory-maps the two binary files (constant RAM regardless
of split size) and returns, for each packed sequence, the token ids, the
next-token labels shifted right by one position, the per-token segment ids,
and per-token document identity.  Supervision never crosses a document
boundary: the label of the last token of every segment is ``IGNORE_INDEX``
(-100), as are all padding positions and the final position of the sequence
(whose continuation lives in the next sequence on disk).

The document-boundary metadata needed by the model (block-diagonal attention
mask, per-document loss accounting) is preserved: every item carries the raw
``segments`` doc-map record, and ``block_diagonal_attention_mask`` builds the
same-document visibility mask from a segment-id vector.

``PackedLMBatchSampler`` yields deterministic index batches: a fixed
``(seed, epoch)`` pair always produces the same permutation, and
``start_batch`` resumes mid-epoch after a checkpoint restart.  The dataset
itself also takes ``offset``/``limit`` to resume at sequence granularity or
to carve out a contiguous shard.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

IGNORE_INDEX = -100

_IDS_SUFFIX = ".input_ids.u16.bin"
_SEG_SUFFIX = ".segment_ids.i32.bin"
_MAP_SUFFIX = ".doc_map.jsonl"


def legacy_data_fingerprint(manifest: Dict) -> str:
    return hashlib.sha256(f'{manifest["document_list_sha256"]}:{manifest["tokenizer"]["sha256"]}'.encode()).hexdigest()


def packed_data_fingerprint(manifest: Dict) -> str:
    splits = {
        split: {
            key: manifest["splits"][split][key]
            for key in ("input_ids_sha256", "segment_ids_sha256", "doc_map_sha256", "seq_len")
        }
        for split in ("train", "validation")
    }
    payload = {
        "document_list_sha256": manifest["document_list_sha256"],
        "tokenizer_sha256": manifest["tokenizer"]["sha256"],
        "splits": splits,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def split_paths(root: Union[str, Path], split: str) -> Dict[str, Path]:
    """On-disk paths of the three files that make up one packed split."""
    root = Path(root)
    return {
        "input_ids": root / f"{split}{_IDS_SUFFIX}",
        "segment_ids": root / f"{split}{_SEG_SUFFIX}",
        "doc_map": root / f"{split}{_MAP_SUFFIX}",
    }


def _infer_seq_len(root: Path, split: str) -> int:
    """Read ``seq_len`` for ``split`` from ``manifest.json``."""
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            f"seq_len not given and no manifest.json found in {root}; "
            "pass seq_len explicitly"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        return int(manifest["splits"][split]["seq_len"])
    except KeyError as exc:
        raise ValueError(
            f"manifest.json in {root} has no seq_len for split {split!r}"
        ) from exc


def _load_doc_map(path: Path) -> List[List[dict]]:
    """One ``segments`` list per packed sequence, in file order."""
    segments: List[List[dict]] = []
    with open(path, "r", encoding="utf-8") as f:
        for expected_seq, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("seq", expected_seq)) != expected_seq:
                raise ValueError(
                    f"{path}: doc_map entry out of order "
                    f"(expected seq {expected_seq}, got {row.get('seq')})"
                )
            segments.append(
                [
                    {
                        "doc": int(seg["doc"]),
                        "offset": int(seg["offset"]),
                        "length": int(seg["length"]),
                    }
                    for seg in row["segments"]
                ]
            )
    return segments


def block_diagonal_attention_mask(segment_ids: Sequence[int]) -> np.ndarray:
    """Same-document visibility mask for one packed sequence.

    Returns a boolean ``(seq_len, seq_len)`` array where entry ``[t, s]`` is
    True iff positions ``t`` and ``s`` belong to the same non-padding
    segment.  The model should combine this with its causal mask to obtain
    block-diagonal causal attention; padding rows/columns are all False.
    """
    seg = np.asarray(segment_ids, dtype=np.int64)
    same = seg[:, None] == seg[None, :]
    valid = seg >= 0
    return same & valid[:, None] & valid[None, :]


class PackedLMDataset:
    """Memory-mapped view over one packed split.

    Parameters
    ----------
    root:
        Directory containing the split files (and optionally manifest.json).
    split:
        Split name prefix, e.g. ``"train"``, ``"validation"``, ``"test"``.
    seq_len:
        Sequence length.  Inferred from ``manifest.json`` when omitted.
    offset:
        First packed sequence of the view (resume at sequence granularity).
    limit:
        Maximum number of sequences in the view; ``None`` = to end of file.

    Item ``__getitem__(i)`` returns a dict with:

      ``index``               global sequence index (offset + i)
      ``input_ids``           (seq_len,) int64 token ids (0 = pad)
      ``labels``              (seq_len,) int64 next-token targets,
                              IGNORE_INDEX at every segment-final token,
                              at padding, and at the last sequence position
      ``segment_ids``         (seq_len,) int32 as stored (-1 = padding)
      ``doc_ids``             (seq_len,) int64 document rank per token,
                              -1 at padding
      ``segments``            doc-map record: [{"doc", "offset", "length"}]
      ``n_tokens``            real (non-padding) tokens in the sequence
      ``n_supervised_tokens`` positions whose label is not IGNORE_INDEX
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        *,
        seq_len: Optional[int] = None,
        offset: int = 0,
        limit: Optional[int] = None,
    ):
        root = Path(root)
        self.root = root
        self.split = split
        if seq_len is None:
            seq_len = _infer_seq_len(root, split)
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.seq_len = seq_len

        paths = split_paths(root, split)
        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"missing {name} file: {path}")

        ids_bytes = paths["input_ids"].stat().st_size
        row_bytes = self.seq_len * np.dtype(np.uint16).itemsize
        if ids_bytes % row_bytes:
            raise ValueError(
                f"{paths['input_ids']}: {ids_bytes} bytes is not a whole number of "
                f"({self.seq_len},) uint16 rows"
            )
        n_sequences = ids_bytes // row_bytes

        self._ids = np.memmap(
            paths["input_ids"], dtype=np.uint16, mode="r", shape=(n_sequences, self.seq_len)
        )
        self._segs = np.memmap(
            paths["segment_ids"], dtype=np.int32, mode="r", shape=(n_sequences, self.seq_len)
        )
        if self._segs.size != self._ids.size:
            raise ValueError(
                f"{paths['segment_ids']} does not match the input_ids shape "
                f"({n_sequences}, {self.seq_len})"
            )

        self._doc_map = _load_doc_map(paths["doc_map"])
        if len(self._doc_map) != n_sequences:
            raise ValueError(
                f"{paths['doc_map']}: {len(self._doc_map)} entries for "
                f"{n_sequences} binary sequences"
            )

        self.n_sequences_disk = n_sequences
        if not 0 <= offset <= n_sequences:
            raise ValueError(f"offset {offset} out of range [0, {n_sequences}]")
        self.offset = offset
        stop = n_sequences if limit is None else min(n_sequences, offset + max(0, limit))
        self._stop = stop

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._stop - self.offset

    def __getitem__(self, index: int) -> Dict[str, object]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        n = len(self)
        if not -n <= index < n:
            raise IndexError(index)
        if index < 0:
            index += n
        global_index = self.offset + index

        seg_ids = np.asarray(self._segs[global_index])  # (seq_len,) int32
        ids = np.asarray(self._ids[global_index], dtype=np.int64)
        segments = self._doc_map[global_index]

        labels = np.full(self.seq_len, IGNORE_INDEX, dtype=np.int64)
        if self.seq_len > 1:
            same_next = (seg_ids[1:] == seg_ids[:-1]) & (seg_ids[:-1] >= 0)
            labels[:-1][same_next] = ids[1:][same_next]

        doc_ids = np.full(self.seq_len, -1, dtype=np.int64)
        doc_of_segment = np.asarray([seg["doc"] for seg in segments], dtype=np.int64)
        real = seg_ids >= 0
        if doc_of_segment.size:
            doc_ids[real] = doc_of_segment[seg_ids[real]]

        n_tokens = int(real.sum())
        n_supervised = int((labels != IGNORE_INDEX).sum())

        return {
            "index": global_index,
            "input_ids": ids,
            "labels": labels,
            "segment_ids": seg_ids.copy(),
            "doc_ids": doc_ids,
            "segments": [dict(seg) for seg in segments],
            "n_tokens": n_tokens,
            "n_supervised_tokens": n_supervised,
        }

    # ------------------------------------------------------------------
    # Token accounting over the view
    # ------------------------------------------------------------------

    @property
    def n_tokens(self) -> int:
        """Real (non-padding) tokens across the whole view."""
        return sum(
            seg["length"]
            for segments in self._doc_map[self.offset : self._stop]
            for seg in segments
        )

    @property
    def n_supervised_tokens(self) -> int:
        """Positions with a non-IGNORE label: one lost per segment.

        Every segment contributes ``length - 1`` supervised targets: its last
        token is never supervised because the next token belongs to a
        different document, to padding, or to the next sequence.
        """
        return sum(
            seg["length"] - 1
            for segments in self._doc_map[self.offset : self._stop]
            for seg in segments
        )

    def token_counts(self) -> Dict[str, int]:
        """Aggregate counts for logging / loss normalization."""
        return {
            "n_sequences": len(self),
            "n_tokens": self.n_tokens,
            "n_supervised_tokens": self.n_supervised_tokens,
            "n_padding": len(self) * self.seq_len - self.n_tokens,
        }


class PackedLMBatchSampler:
    """Deterministic batch sampler over a ``PackedLMDataset`` (or any length).

    The permutation depends only on ``(seed, epoch)``: identical arguments
    always reproduce identical batches, on any machine.  ``start_batch``
    skips the first batches of the epoch, which is how a training loop
    resumes mid-epoch after loading a checkpoint (resume offset = number of
    optimizer steps already taken in the current epoch).

    Iterating yields lists of dataset indices.  With ``shuffle=False`` the
    order is sequential (validation/test evaluation).
    """

    def __init__(
        self,
        dataset: Union[PackedLMDataset, int],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        epoch: int = 0,
        drop_last: bool = True,
        start_batch: int = 0,
    ):
        n = len(dataset) if hasattr(dataset, "__len__") else int(dataset)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if n < 0:
            raise ValueError("dataset size must be non-negative")
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        self.drop_last = drop_last
        self.start_batch = start_batch

    @property
    def batches_per_epoch(self) -> int:
        if self.n == 0:
            return 0
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        """Number of batches iteration will yield from ``start_batch`` on."""
        return max(0, self.batches_per_epoch - self.start_batch)

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch; changes the permutation deterministically."""
        self.epoch = epoch

    def _order(self) -> List[int]:
        order = list(range(self.n))
        if self.shuffle:
            key = int.from_bytes(
                hashlib.sha256(f"packed-lm-sampler-v1:{self.seed}:{self.epoch}".encode()).digest()[:8],
                "big",
            )
            random.Random(key).shuffle(order)
        return order

    def __iter__(self) -> Iterator[List[int]]:
        order = self._order()
        batches = [
            order[i : i + self.batch_size]
            for i in range(0, self.n, self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        yield from batches[self.start_batch :]
