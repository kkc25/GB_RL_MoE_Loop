"""Recovery-curriculum datasets (recovery plan sec. 4/5, appendix A/B).

Label contract (sec. 4.1): the model input is exactly ``[BOS, x, program,
QRY]``; the answer never appears in the input. Supervision is a single
cross-entropy target ``target_answer_id`` at the QRY position. One example
carries exactly one supervised target.

Token layout is inherited from :mod:`data.affine` (V = 512, PAD/BOS/EOS/QRY/
FNC = 0..4, ``sym(v) = 5 + v``), so the same embedding table and head serve
every stage. Moduli 11/13/97 all fit inside the symbol range 5..101.

Stages (sec. 5.4):
    c0_identity   p = 97, empty program, all 97 inputs (full-table control)
    c0_fixed      p = 97, fixed single function 2x+1, all 97 inputs
    c1            p = 11, explicit one-hop (a, b, x), a != 0, 1,210 instances,
                  fixed-seed 70/15/15 instance split (train/id_validation/
                  id_test) covering every digit and primitive role
    c2            p = 13, ops {(1,1),(2,1),(3,1),(5,1)}, lengths 1-3
    c3            p = 97, same op set and split structure as c2

For c2/c3 the five splits are train / id_validation / id_test /
program_validation / program_test. Primitive (length-1) full tables are
training-only. Length-2 programs are split 12/2/2 and length-3 programs
44/10/10 into train/program_validation/program_test; inside each
train program the inputs x are split 9/2/2 (p = 13) or 67/15/15 (p = 97,
same fractions) into train/id_validation/id_test.

Every instance carries three identities (sec. 5.3): ``program_id`` (the
operation string), ``canonical`` ((A, B) of the composed affine map), and
``instance_id`` ((program_id, x)).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .affine import BOS, EOS, FNC, PAD, QRY, SYM_BASE, VOCAB_SIZE, inv_sym, sym

# Split names. c0 uses only "train"; c1 uses train/id_validation/id_test;
# c2/c3 use all five.
TRAIN = "train"
ID_VALIDATION = "id_validation"
ID_TEST = "id_test"
PROGRAM_VALIDATION = "program_validation"
PROGRAM_TEST = "program_test"
FIVE_SPLITS = (TRAIN, ID_VALIDATION, ID_TEST, PROGRAM_VALIDATION, PROGRAM_TEST)
INSTANCE_SPLITS = (TRAIN, ID_VALIDATION, ID_TEST)

STAGES = ("c0_identity", "c0_fixed", "c1", "c2", "c3")

DEFAULT_SPLIT_SEED = 0

# Composition-control op library (sec. 5.4); invertible mod 13 and mod 97.
COMPOSITION_OPS: Tuple[Tuple[int, int], ...] = ((1, 1), (2, 1), (3, 1), (5, 1))
LENGTH2_PROGRAM_COUNTS = (12, 2, 2)   # train / program_validation / program_test
LENGTH3_PROGRAM_COUNTS = (44, 10, 10)
# Within-train-program x split, per modulus (same 9/2/2 fractions of 13).
X_SPLIT_COUNTS = {13: (9, 2, 2), 97: (67, 15, 15)}

C1_MODULUS = 11
C1_SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

# Expected manifest counts (sec. 5.4) — asserted by build and by tests.
EXPECTED_COUNTS = {
    "c0_identity": {"train": 97},
    "c0_fixed": {"train": 97},
    "c1": {"train": 847, "id_validation": 182, "id_test": 181},
    "c2": {"train": 556, "id_validation": 112, "id_test": 112,
           "program_validation": 156, "program_test": 156},
    "c3": {"train": 4140, "id_validation": 840, "id_test": 840,
           "program_validation": 1164, "program_test": 1164},
}

STAGE_MODULUS = {"c0_identity": 97, "c0_fixed": 97, "c1": 11, "c2": 13, "c3": 97}


def _stable_int(*parts) -> int:
    """Deterministic 64-bit hash of the given parts (stable across runs)."""
    msg = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Independent interpreter (sec. 4.2): ground truth never depends on the
# token encoding; audit re-derives labels from decoded sequences with these.
# ---------------------------------------------------------------------------


def apply_op(x: int, op: Tuple[int, int], p: int) -> int:
    a, b = op
    return (a * x + b) % p


def run_program(x: int, ops: Sequence[Tuple[int, int]], p: int) -> int:
    for op in ops:
        x = apply_op(x, op, p)
    return x


def canonical_affine(ops: Sequence[Tuple[int, int]], p: int) -> Tuple[int, int]:
    """(A, B) of the composed map x -> (A*x + B) % p (plan eq. 3)."""
    A, B = 1, 0
    for a, b in ops:
        A, B = a * A % p, (a * B + b) % p
    return A, B


def program_id(ops: Sequence[Tuple[int, int]]) -> str:
    """Operation-string identity; the empty program is 'id'."""
    if not ops:
        return "id"
    return ";".join(f"f{a},{b}" for a, b in ops)


def instance_id(ops: Sequence[Tuple[int, int]], x: int, p: int) -> str:
    return f"p={p}|{program_id(ops)}|x={x}"


# ---------------------------------------------------------------------------
# Instances and encoding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One supervised example: initial value x and an affine program."""

    x: int
    ops: Tuple[Tuple[int, int], ...]
    modulus: int
    split: str

    @property
    def answer(self) -> int:
        return run_program(self.x, self.ops, self.modulus)

    @property
    def program_id(self) -> str:
        return program_id(self.ops)

    @property
    def canonical(self) -> Tuple[int, int]:
        return canonical_affine(self.ops, self.modulus)

    @property
    def instance_id(self) -> str:
        return instance_id(self.ops, self.x, self.modulus)

    @property
    def program_length(self) -> int:
        return len(self.ops)


def encode_instance(instance: Instance) -> List[int]:
    """Input ids ``[BOS, sym(x), (FNC, sym(a), sym(b))*..., QRY]``.

    The answer and EOS are deliberately absent (sec. 4.1: the main path does
    not append the answer to the input).
    """
    ids = [BOS, sym(instance.x)]
    for a, b in instance.ops:
        ids.extend([FNC, sym(a), sym(b)])
    ids.append(QRY)
    return ids


def query_position(instance: Instance) -> int:
    """Index of the QRY token; QRY is always the last input token."""
    return 2 + 3 * instance.program_length


def decode_input(ids: Sequence[int], p: int) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
    """Parse ``[BOS, x, program, QRY]`` back into (x, ops); inverse of
    :func:`encode_instance`, used by the D0 audit as an independent reader."""
    ids = [int(t) for t in ids]
    if len(ids) < 3 or ids[0] != BOS or ids[-1] != QRY:
        raise ValueError("input must be [BOS, x, program..., QRY]")
    body = ids[1:-1]
    if (len(body) - 1) % 3 != 0:
        raise ValueError("malformed program encoding")
    x = inv_sym(body[0])
    ops = []
    for i in range(1, len(body), 3):
        if body[i] != FNC:
            raise ValueError("expected FNC marker in program encoding")
        ops.append((inv_sym(body[i + 1]), inv_sym(body[i + 2])))
    for v in [x] + [c for op in ops for c in op]:
        if v >= p:
            raise ValueError(f"value {v} outside modulus {p}")
    return x, tuple(ops)


class RecoveryDataset(Dataset):
    """Map-style dataset over a fixed list of instances (one stage/split)."""

    def __init__(self, instances: Sequence[Instance], stage: str, split: str):
        self.instances = list(instances)
        self.stage = stage
        self.split = split
        self.modulus = STAGE_MODULUS[stage]

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> Dict:
        inst = self.instances[index]
        ids = encode_instance(inst)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "query_position": query_position(inst),
            "target_answer_id": sym(inst.answer),
            "meta": {
                "stage": self.stage,
                "split": self.split,
                "modulus": inst.modulus,
                "x": inst.x,
                "ops": inst.ops,
                "program_id": inst.program_id,
                "canonical_a": inst.canonical[0],
                "canonical_b": inst.canonical[1],
                "instance_id": inst.instance_id,
                "program_length": inst.program_length,
                "answer": inst.answer,
            },
        }


def recovery_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Right-pad a list of items; QRY positions refer to the padded tensor."""
    max_len = max(item["input_ids"].numel() for item in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), PAD, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    query_positions = torch.zeros(bsz, dtype=torch.long)
    target_answer_ids = torch.zeros(bsz, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].numel()
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = 1
        query_positions[i] = item["query_position"]
        target_answer_ids[i] = item["target_answer_id"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "query_positions": query_positions,
        "target_answer_ids": target_answer_ids,
        "meta": [item["meta"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Stage construction
# ---------------------------------------------------------------------------


def _c0_instances(task: str) -> List[Instance]:
    if task == "c0_identity":
        ops: Tuple[Tuple[int, int], ...] = ()
    elif task == "c0_fixed":
        ops = ((2, 1),)
    else:
        raise ValueError(f"unknown c0 task: {task}")
    return [Instance(x, ops, 97, TRAIN) for x in range(97)]


def _c1_instances(seed: int) -> Dict[str, List[Instance]]:
    p = C1_MODULUS
    triples = [(a, b, x) for a in range(1, p) for b in range(p) for x in range(p)]
    rng = random.Random(_stable_int("recovery-c1-split", seed))
    rng.shuffle(triples)
    n = len(triples)
    n_train = round(n * C1_SPLIT_FRACTIONS[0])
    n_val = round(n * C1_SPLIT_FRACTIONS[1])
    assert n_train == EXPECTED_COUNTS["c1"]["train"]
    split_of = {}
    for i, t in enumerate(triples):
        if i < n_train:
            split_of[t] = TRAIN
        elif i < n_train + n_val:
            split_of[t] = ID_VALIDATION
        else:
            split_of[t] = ID_TEST
    out = {name: [] for name in INSTANCE_SPLITS}
    for (a, b, x), split in split_of.items():
        out[split].append(Instance(x, ((a, b),), p, split))
    for name in out:
        out[name].sort(key=lambda inst: inst.instance_id)
    # The fixed seed must cover every digit and every primitive role in train.
    train = out[TRAIN]
    seen_a = {inst.ops[0][0] for inst in train}
    seen_b = {inst.ops[0][1] for inst in train}
    seen_x = {inst.x for inst in train}
    assert seen_a == set(range(1, p)), "train split misses a multiplier digit"
    assert seen_b == set(range(p)), "train split misses an offset digit"
    assert seen_x == set(range(p)), "train split misses an input digit"
    return out


def _split_programs(
    programs: List[Tuple[Tuple[int, int], ...]],
    counts: Tuple[int, int, int],
    seed: int,
    tag: str,
) -> Tuple[list, list, list]:
    rng = random.Random(_stable_int("recovery-program-split", seed, tag))
    ordered = list(programs)
    rng.shuffle(ordered)
    n_train, n_val, n_test = counts
    assert n_train + n_val + n_test == len(ordered)
    return (
        ordered[:n_train],
        ordered[n_train:n_train + n_val],
        ordered[n_train + n_val:],
    )


def _composition_instances(p: int, seed: int) -> Dict[str, List[Instance]]:
    out = {name: [] for name in FIVE_SPLITS}
    # Length 1: primitive full tables are training-only (sec. 5.4).
    for op in COMPOSITION_OPS:
        for x in range(p):
            out[TRAIN].append(Instance(x, (op,), p, TRAIN))
    # Same seed => identical program split for p = 13 and p = 97.
    for length, counts in ((2, LENGTH2_PROGRAM_COUNTS), (3, LENGTH3_PROGRAM_COUNTS)):
        programs = list(itertools.product(COMPOSITION_OPS, repeat=length))
        train_prog, val_prog, test_prog = _split_programs(
            programs, counts, seed, f"length{length}"
        )
        used = {op for prog in train_prog for op in prog}
        assert used == set(COMPOSITION_OPS), "train programs miss a primitive"
        n_x_train, n_x_val, n_x_test = X_SPLIT_COUNTS[p]
        assert n_x_train + n_x_val + n_x_test == p
        for prog in train_prog:
            xs = list(range(p))
            rng = random.Random(_stable_int("recovery-x-split", seed, p, prog))
            rng.shuffle(xs)
            for i, x in enumerate(xs):
                if i < n_x_train:
                    split = TRAIN
                elif i < n_x_train + n_x_val:
                    split = ID_VALIDATION
                else:
                    split = ID_TEST
                out[split].append(Instance(x, prog, p, split))
        for prog in val_prog:
            for x in range(p):
                out[PROGRAM_VALIDATION].append(Instance(x, prog, p, PROGRAM_VALIDATION))
        for prog in test_prog:
            for x in range(p):
                out[PROGRAM_TEST].append(Instance(x, prog, p, PROGRAM_TEST))
    for name in out:
        out[name].sort(key=lambda inst: inst.instance_id)
    return out


def build_instances(
    stage: str, seed: int = DEFAULT_SPLIT_SEED
) -> Dict[str, List[Instance]]:
    """All instances of one stage, grouped by split, with count checks."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage!r}")
    if stage in ("c0_identity", "c0_fixed"):
        splits = {TRAIN: _c0_instances(stage)}
    elif stage == "c1":
        splits = _c1_instances(seed)
    else:
        splits = _composition_instances(STAGE_MODULUS[stage], seed)
    for name, expected in EXPECTED_COUNTS[stage].items():
        actual = len(splits.get(name, []))
        if actual != expected:
            raise AssertionError(
                f"{stage}/{name}: expected {expected} instances, got {actual}"
            )
    return splits


def build_datasets(
    stage: str, seed: int = DEFAULT_SPLIT_SEED
) -> Dict[str, RecoveryDataset]:
    """Map-style datasets for every non-empty split of a stage."""
    return {
        name: RecoveryDataset(insts, stage, name)
        for name, insts in build_instances(stage, seed).items()
        if insts
    }


# ---------------------------------------------------------------------------
# Manifest and hashes (sec. 8.6 / D0: split manifest + data hash)
# ---------------------------------------------------------------------------


def _hash_instance_ids(instances: Sequence[Instance]) -> str:
    h = hashlib.sha256()
    for inst in sorted(instances, key=lambda i: i.instance_id):
        h.update(inst.instance_id.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_manifest(stage: str, seed: int = DEFAULT_SPLIT_SEED) -> Dict:
    """Per-split instance counts and content hashes, plus a manifest hash."""
    splits = build_instances(stage, seed)
    manifest = {
        "stage": stage,
        "modulus": STAGE_MODULUS[stage],
        "split_seed": seed,
        "label_contract": "answer_at_qry_not_in_input",
        "splits": {
            name: {"count": len(insts), "sha256": _hash_instance_ids(insts)}
            for name, insts in sorted(splits.items())
        },
    }
    manifest["total_instances"] = sum(s["count"] for s in manifest["splits"].values())
    payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    return manifest


def manifest_json(manifest: Dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# D0 audit (sec. 4.2): independent label recomputation, prefix-conflict and
# target-count checks over at least 1,000 examples or the full finite table.
# ---------------------------------------------------------------------------


@dataclass
class AuditReport:
    checked: int = 0
    meets_min_count: bool = False
    label_mismatches: List[Tuple[str, int, int]] = field(default_factory=list)
    prefix_conflicts: List[Tuple[str, int, int]] = field(default_factory=list)
    query_errors: List[str] = field(default_factory=list)
    target_errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.label_mismatches
            or self.prefix_conflicts
            or self.query_errors
            or self.target_errors
        )

    def assert_ok(self) -> "AuditReport":
        if not self.ok:
            raise AssertionError(
                "D0 audit failed: "
                f"{len(self.label_mismatches)} label mismatches, "
                f"{len(self.prefix_conflicts)} prefix conflicts, "
                f"{len(self.query_errors)} query errors, "
                f"{len(self.target_errors)} target errors"
            )
        return self


def audit_dataset(dataset: RecoveryDataset, min_count: int = 1000) -> AuditReport:
    """Recompute every label with the independent interpreter and check the
    data contract. Deterministic tasks must show zero unexplained conflicts.

    Every item is checked: at least ``min_count`` examples, or the full
    finite table when the domain is smaller. ``report.meets_min_count``
    records whether the 1,000-example threshold was reached; small full
    tables are audited in full instead.
    """
    report = AuditReport()
    prefix_labels: Dict[Tuple[int, ...], int] = {}
    p = dataset.modulus
    legal_targets = {sym(v) for v in range(p)}
    for i in range(len(dataset)):
        item = dataset[i]
        ids = [int(t) for t in item["input_ids"]]
        report.checked += 1
        # Exactly one QRY, at the recorded position, and it ends the input.
        if ids.count(QRY) != 1:
            report.query_errors.append(
                f"{item['meta']['instance_id']}: {ids.count(QRY)} QRY tokens"
            )
            continue
        q_idx = ids.index(QRY)
        if q_idx != int(item["query_position"]) or q_idx != len(ids) - 1:
            report.query_errors.append(
                f"{item['meta']['instance_id']}: QRY at {q_idx}, "
                f"recorded {item['query_position']}, length {len(ids)}"
            )
            continue
        target = int(item["target_answer_id"])
        if target not in legal_targets:
            report.target_errors.append(
                f"{item['meta']['instance_id']}: target {target} not a legal "
                f"answer symbol for p={p}"
            )
            continue
        try:
            x, ops = decode_input(ids, p)
        except ValueError as exc:
            report.query_errors.append(f"{item['meta']['instance_id']}: {exc}")
            continue
        expected = run_program(x, ops, p)
        if sym(expected) != target:
            report.label_mismatches.append(
                (item["meta"]["instance_id"], expected, inv_sym(target))
            )
        prefix = tuple(ids[: q_idx + 1])
        if prefix in prefix_labels and prefix_labels[prefix] != target:
            report.prefix_conflicts.append(
                (item["meta"]["instance_id"], prefix_labels[prefix], target)
            )
        else:
            prefix_labels[prefix] = target
    report.meets_min_count = report.checked >= min_count
    return report


def audit_batch(batch: Dict[str, torch.Tensor], modulus: int) -> List[str]:
    """Batch-level D0 checks: right padding, mask contiguity, QRY position
    and target legality on the actual padded tensors."""
    errors: List[str] = []
    input_ids = batch["input_ids"]
    mask = batch["attention_mask"]
    qpos = batch["query_positions"]
    targets = batch["target_answer_ids"]
    legal = {sym(v) for v in range(modulus)}
    for i in range(input_ids.shape[0]):
        n = int(mask[i].sum())
        if not bool((mask[i, :n] == 1).all()) or not bool((mask[i, n:] == 0).all()):
            errors.append(f"row {i}: attention mask is not right-contiguous")
        if n == 0 or int(input_ids[i, n - 1]) != QRY:
            errors.append(f"row {i}: last visible token is not QRY")
        if int(qpos[i]) != n - 1 or int(input_ids[i, qpos[i]]) != QRY:
            errors.append(f"row {i}: query position {int(qpos[i])} mismatch")
        if int(targets[i]) not in legal:
            errors.append(f"row {i}: illegal target {int(targets[i])}")
        if bool((input_ids[i, n:] == PAD).all()) is False:
            errors.append(f"row {i}: padding is not PAD")
    return errors
