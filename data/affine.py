"""Mod-97 affine program composition task (P0 synthetic data, plan sec. 8.1/8.2).

Task: given an initial value z0 and a short program of affine functions
f_{a,b}(z) = (a*z + b) % 97 applied left to right, predict the final value.

Token layout (fixed vocab V = 512):
    0       PAD
    1       BOS
    2       EOS
    3       QRY   query delimiter; the answer symbol follows it
    4       FNC   function marker, followed by the two symbol tokens a, b
    5..101  symbol tokens for values 0..96 (sym(v) = 5 + v)
    102..511 reserved (per-function/family tokens, diagnostics, future use)

Sequence format (k functions):
    [BOS, sym(z0), FNC, sym(a1), sym(b1), ..., FNC, sym(ak), sym(bk),
     QRY, sym(answer), EOS]
Labels are -100 everywhere except QRY, whose target is the following answer.

Splits are by program family, not by random instance: each function is typed
T (translation a==1), S (scaling b==0), or G (general), and a program family
is the tuple of per-step types. Whole families are assigned to
train/validation/test deterministically per program length (length/difficulty
stratification), so no family appears in two splits.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

MODULUS = 97
VOCAB_SIZE = 512

PAD = 0
BOS = 1
EOS = 2
QRY = 3
FNC = 4
SYM_BASE = 5
SYM_END = SYM_BASE + MODULUS  # exclusive; symbols occupy 5..101
IGNORE_INDEX = -100

# Function types used to define program families. Identity (a==1, b==0) and
# constant (a==0) maps are excluded from random generation: identities are
# no-ops and constants destroy the dependency on the initial value.
TYPE_TRANSLATION = "T"  # a == 1, b != 0
TYPE_SCALING = "S"      # b == 0, a not in {0, 1}
TYPE_GENERAL = "G"      # a not in {0, 1}, b != 0
FUNCTION_TYPES = (TYPE_TRANSLATION, TYPE_SCALING, TYPE_GENERAL)

DEFAULT_LENGTHS = (1, 2, 3, 4)
DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_VAL_FRAC = 0.15


def sym(value: int) -> int:
    """Token id for a value in 0..96."""
    if not 0 <= value < MODULUS:
        raise ValueError(f"value out of range: {value}")
    return SYM_BASE + value


def inv_sym(token: int) -> int:
    """Value encoded by a symbol token."""
    if not SYM_BASE <= token < SYM_END:
        raise ValueError(f"not a symbol token: {token}")
    return token - SYM_BASE


def _stable_int(*parts) -> int:
    """Deterministic 64-bit hash of the given parts (stable across runs)."""
    msg = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")


def function_type(a: int, b: int) -> str:
    if a == 0:
        return "Z"
    if a == 1:
        return TYPE_TRANSLATION if b != 0 else "I"
    if b == 0:
        return TYPE_SCALING
    return TYPE_GENERAL


@dataclass(frozen=True)
class AffineFunction:
    """f(z) = (a*z + b) % 97."""

    a: int
    b: int

    def __post_init__(self):
        if not (0 <= self.a < MODULUS and 0 <= self.b < MODULUS):
            raise ValueError(f"parameters out of range: a={self.a}, b={self.b}")

    def apply(self, z: int) -> int:
        return (self.a * z + self.b) % MODULUS

    @property
    def type(self) -> str:
        return function_type(self.a, self.b)

    def compose(self, other: "AffineFunction") -> "AffineFunction":
        """self ∘ other: apply other first, then self."""
        return AffineFunction(
            (self.a * other.a) % MODULUS,
            (self.a * other.b + self.b) % MODULUS,
        )

    def inverse(self) -> "AffineFunction":
        """f^{-1} with f^{-1}(f(z)) == z; requires a != 0."""
        if self.a == 0:
            raise ValueError("constant map is not invertible")
        a_inv = pow(self.a, -1, MODULUS)
        return AffineFunction(a_inv, (-a_inv * self.b) % MODULUS)


@dataclass(frozen=True)
class Program:
    """An initial value and a sequence of affine functions applied in order."""

    z0: int
    functions: Tuple[AffineFunction, ...]

    def run(self) -> int:
        z = self.z0
        for f in self.functions:
            z = f.apply(z)
        return z

    @property
    def length(self) -> int:
        return len(self.functions)

    @property
    def family(self) -> Tuple[str, ...]:
        return tuple(f.type for f in self.functions)

    @property
    def difficulty(self) -> Tuple[int, int]:
        """(program length, number of multiplicative steps) — metadata only."""
        n_mult = sum(1 for f in self.functions if f.type != TYPE_TRANSLATION)
        return (self.length, n_mult)


def encode_program(program: Program) -> Tuple[List[int], List[int]]:
    """Return causal labels that predict the answer at the query delimiter."""
    ids = [BOS, sym(program.z0)]
    for f in program.functions:
        ids.extend([FNC, sym(f.a), sym(f.b)])
    ids.append(QRY)
    answer_pos = len(ids)
    ids.extend([sym(program.run()), EOS])
    labels = [IGNORE_INDEX] * len(ids)
    labels[answer_pos - 1] = ids[answer_pos]
    return ids, labels


def answer_position(program_length: int) -> int:
    """Index of the supervised answer token for a program of this length."""
    return 2 + 3 * program_length + 1


def sequence_length(program_length: int) -> int:
    return 2 + 3 * program_length + 3


def all_families(length: int) -> List[Tuple[str, ...]]:
    return list(itertools.product(FUNCTION_TYPES, repeat=length))


def family_split_map(
    length: int,
    seed: int = 0,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> Dict[str, List[Tuple[str, ...]]]:
    """Deterministically assign every family of this length to a split.

    Families are ordered by a seeded hash and cut into train/val/test, so the
    assignment is reproducible, disjoint, and stratified per length. Val and
    test each get at least one family.
    """
    fams = sorted(all_families(length), key=lambda f: _stable_int("split", seed, length, f))
    n = len(fams)
    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * val_frac))
    if n_val + n_test >= n:  # degenerate tiny case; keep train non-empty
        n_val = n_test = 1
    n_train = n - n_val - n_test
    return {
        "train": fams[:n_train],
        "validation": fams[n_train:n_train + n_val],
        "test": fams[n_train + n_val:],
    }


def families_for_split(split: str, length: int, seed: int = 0) -> List[Tuple[str, ...]]:
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split: {split}")
    return family_split_map(length, seed=seed)[split]


def _sample_function_of_type(rng: random.Random, ftype: str) -> AffineFunction:
    if ftype == TYPE_TRANSLATION:
        return AffineFunction(1, rng.randrange(1, MODULUS))
    if ftype == TYPE_SCALING:
        return AffineFunction(rng.randrange(2, MODULUS), 0)
    if ftype == TYPE_GENERAL:
        return AffineFunction(rng.randrange(2, MODULUS), rng.randrange(1, MODULUS))
    raise ValueError(f"cannot sample function of type {ftype!r}")


def sample_program(
    rng: random.Random,
    split: str,
    length: int,
    split_seed: int = 0,
) -> Program:
    """Sample a random instance whose family belongs to the given split."""
    family = rng.choice(families_for_split(split, length, seed=split_seed))
    funcs = tuple(_sample_function_of_type(rng, t) for t in family)
    z0 = rng.randrange(MODULUS)
    return Program(z0, funcs)


class AffineDataset(Dataset):
    """Reproducible map-style dataset of mod-97 affine programs.

    Item i is fully determined by (seed, offset + i); lengths cycle through
    `lengths` so every bucket is evenly represented (stratification). Each
    item is a dict with input_ids, labels, and metadata (split, length,
    family, difficulty, z0, answer, parameters).
    """

    def __init__(
        self,
        split: str = "train",
        num_samples: int = 1024,
        lengths: Sequence[int] = DEFAULT_LENGTHS,
        seed: int = 0,
        offset: int = 0,
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"unknown split: {split}")
        if not lengths or min(lengths) < 1:
            raise ValueError("lengths must be positive")
        self.split = split
        self.num_samples = num_samples
        self.lengths = tuple(lengths)
        self.seed = seed
        self.offset = offset
        # Fail fast if some length has no family in this split.
        for k in self.lengths:
            if not families_for_split(split, k, seed=seed):
                raise ValueError(f"split {split!r} has no family of length {k}")

    def __len__(self) -> int:
        return self.num_samples

    def make_program(self, index: int) -> Program:
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        length = self.lengths[(self.offset + index) % len(self.lengths)]
        rng = random.Random(_stable_int("instance", self.seed, self.offset + index))
        return sample_program(rng, self.split, length, split_seed=self.seed)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        program = self.make_program(index)
        ids, labels = encode_program(program)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "length": program.length,
            "family": "".join(program.family),
            "difficulty": program.difficulty,
            "z0": program.z0,
            "answer": program.run(),
        }


def affine_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad a list of dataset items into a batch; padded labels are -100."""
    max_len = max(item["input_ids"].numel() for item in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), PAD, dtype=torch.long)
    labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].numel()
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        attention_mask[i, :n] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "meta": [
            {k: v for k, v in item.items() if k not in ("input_ids", "labels")}
            for item in batch
        ],
    }


def sample_batch(
    split: str,
    batch_size: int,
    seed: int = 0,
    offset: int = 0,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
) -> Dict[str, torch.Tensor]:
    """Deterministic batch: items offset .. offset+batch_size-1 of the split.

    The same (split, seed, offset) always yields the same batch; different
    offsets give disjoint, reproducible batches without keeping a generator.
    """
    ds = AffineDataset(
        split=split, num_samples=batch_size, lengths=lengths, seed=seed, offset=offset
    )
    return affine_collate([ds[i] for i in range(batch_size)])


# ---------------------------------------------------------------------------
# Diagnostics: known complementary / non-commuting operator structure.
# These give affine-side diagnostics analogous to the controlled
# non-commutative operator environment: ground truth is exact, so a policy
# that must compose operators in a specific order can be checked precisely.
# ---------------------------------------------------------------------------


def complementary_program(z0: int, a: int, b: int) -> Program:
    """[f, f^{-1}] with known answer z0 (f then its inverse is the identity)."""
    f = AffineFunction(a, b)
    return Program(z0, (f, f.inverse()))


def noncommuting_pair(
    a1: int = 2, b1: int = 1, a2: int = 1, b2: int = 1
) -> Tuple[AffineFunction, AffineFunction]:
    """A pair (f, g) with f∘g != g∘f. The default is a fixed known example:
    f(z)=2z+1, g(z)=z+1 give f∘g(z)=2z+3 but g∘f(z)=2z+2 (mod 97)."""
    f = AffineFunction(a1, b1)
    g = AffineFunction(a2, b2)
    if f.compose(g) == g.compose(f):
        raise ValueError("the given pair commutes; pick other parameters")
    return f, g


def order_diagnostic_programs(
    z0: int, f: AffineFunction, g: AffineFunction
) -> Tuple[Program, Program]:
    """Programs [f, g] and [g, f] with exactly known, generally different
    answers, for testing sensitivity to operator order. Functions run left to
    right: [f, g] computes (g∘f)(z0) and [g, f] computes (f∘g)(z0)."""
    return Program(z0, (f, g)), Program(z0, (g, f))
