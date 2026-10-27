from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import os
from pathlib import Path
import random

import numpy as np
import torch

from data.packed_lm import verify_packed_split
from eval.counterfactual import cache_last_ffn_input, last_adaptive_layer
from experiments.r1_closed_loop import source_identity
from model.top2 import Top2MoE, baseline_top2_pairs
from rl_lm.readout import terminal_ce
from rl_lm.selector import canonical_pair
from train.common import autocast_context
from train.recovery_common import atomic_json, sha256_file

CACHE_FORMAT = "rl-lm-cache-v2"
PAYLOAD_FORMAT = "rl-lm-cache-payload-v2"
ARRAY_FILES = ("h0", "targets", "positions", "sequence_indices")
JSON_FILES = ("episode_ids.jsonl", "baseline.jsonl", "sampling_report.json")


def load_document_identities(corpus_dir):
    corpus_dir = Path(corpus_dir)
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    if manifest.get("document_list_file"):
        filename = manifest["document_list_file"]
        expected = manifest["document_list_sha256"]
    elif manifest.get("pipeline_version") == "fixed-depth-confirmation-v1":
        filename = "confirmation_documents.jsonl"
        expected = manifest["confirmation_documents_sha256"]
    else:
        raise ValueError("unknown corpus document identity format")
    path = corpus_dir / filename
    if sha256_file(path) != expected:
        raise ValueError("document identity file hash mismatch")
    records = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        records[int(row["rank"])] = {key: row.get(key) for key in ("id", "doc_hash", "text_hash", "near_hash")}
    return records


def plan_contexts(dataset, n_states, *, seed, corpus_id, document_records=None):
    if n_states <= 0:
        raise ValueError("n_states must be positive")
    contexts = []
    for local_index in range(len(dataset)):
        item = dataset[local_index]
        global_index = int(item["index"])
        for position in np.nonzero(item["labels"] != -100)[0]:
            position = int(position)
            doc_id = int(item["doc_ids"][position])
            identity = (document_records or {}).get(doc_id, {})
            contexts.append(identity | {
                "sequence_index": global_index,
                "position": position,
                "doc_id": doc_id,
                "episode_id": f"{corpus_id}:{global_index}:{position}",
            })
    if len(contexts) < n_states:
        raise ValueError(f"split has only {len(contexts)} valid states, requested {n_states}")
    random.Random(seed).shuffle(contexts)
    return contexts[:n_states]


def cache_identity(model, config, *, checkpoint_path, config_path, tokenizer_path, data_dir, split, sampling_seed, backend_version, head_contract, expert_chunk):
    _require_legacy_prefix(model)
    layer = last_adaptive_layer(model)
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    packed = verify_packed_split(data_dir, split, manifest)
    return {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "packed_sha256": packed,
        "document_manifest_sha256": sha256_file(Path(data_dir) / "manifest.json"),
        "layer_index": layer,
        "gamma": float(model.gammas[layer]),
        "prefix_depth": 1,
        "prefix_dtype": "bfloat16",
        "backend_version": backend_version,
        "expert_chunk": int(expert_chunk),
        "head_contract": head_contract,
        "cache_format": CACHE_FORMAT,
        "split": split,
        "sampling_seed": sampling_seed,
        "source": source_identity(),
    }


def validate_cache_identity(manifest, expected):
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"cache identity mismatch at {key}: {manifest.get(key)!r} != {value!r}")
    return True


def _write_jsonl(path, rows):
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _sampling_report(contexts, dataset, excluded_doc_ids):
    documents = {}
    for context in contexts:
        documents[context["doc_id"]] = documents.get(context["doc_id"], 0) + 1
    pairs = {(context["sequence_index"], context["position"]) for context in contexts}
    intersection = sorted(documents.keys() & set(excluded_doc_ids))
    if intersection:
        raise ValueError(f"sampled documents intersect exclusions: {intersection[:8]}")
    return {
        "states": len(contexts),
        "documents": len(documents),
        "max_document_weight": max(documents.values()) / len(contexts),
        "duplicate_positions": len(contexts) - len(pairs),
        "excluded_doc_ids_present": intersection,
        "near_duplicate_coverage": "unavailable in packed reader; corpus-level near_hash must be checked separately",
        "sampling": "uniform valid token without replacement",
        "dataset_offset": dataset.offset,
    }


def _baseline_records(model, dataset, contexts, device, dtype):
    by_sequence = {}
    for row, context in enumerate(contexts):
        by_sequence.setdefault(context["sequence_index"], []).append((row, context))
    records = [None] * len(contexts)
    layer = last_adaptive_layer(model)
    block = model.blocks[layer]
    with torch.no_grad():
        for sequence_index, entries in by_sequence.items():
            local = sequence_index - dataset.offset
            if not 0 <= local < len(dataset):
                raise ValueError("cached global sequence index is outside dataset view")
            item = dataset[local]
            input_ids = torch.from_numpy(item["input_ids"]).unsqueeze(0).to(device)
            labels = torch.from_numpy(item["labels"]).unsqueeze(0).to(device)
            segment_ids = torch.from_numpy(item["segment_ids"]).unsqueeze(0).to(device)
            with autocast_context(device, dtype):
                output = model(input_ids, labels, depth=1, segment_ids=segment_ids)
                h0 = cache_last_ffn_input(model, input_ids, segment_ids, depth=1)
            for row, context in entries:
                position = context["position"]
                h = h0[0, position].float().unsqueeze(0)
                normalized = block.ffn_norm(h)
                gate = block.executor.gate_logits(normalized).float().reshape(1, -1)
                canonical_index = int(canonical_pair(gate)[0])
                pair = canonical_pair_table()[canonical_index].to(device)
                legacy_pair = baseline_top2_pairs(gate)[0]
                margin = float(gate.topk(2).values[0, 0] - gate.topk(2).values[0, 1])
                with autocast_context(device, dtype):
                    update, _, _, _ = block.executor(normalized, pairs=pair.unsqueeze(0))
                    canonical_hidden = h + model.gammas[layer] * update
                target = torch.tensor([item["labels"][position]], dtype=torch.long, device=device)
                canonical_ce = terminal_ce(model, canonical_hidden, target)[0]
                legacy_ce = terminal_ce(model, output.hidden[0, position].float().unsqueeze(0), target)[0]
                records[row] = {
                    "canonical_pair_index": canonical_index,
                    "canonical_pair": pair.tolist(),
                    "legacy_pair": legacy_pair.tolist(),
                    "pair_difference": not torch.equal(pair, legacy_pair),
                    "gate_margin": margin,
                    "canonical_ce": float(canonical_ce),
                    "legacy_ce": float(legacy_ce),
                }
    return records


def canonical_pair_table():
    from rl_lm.actions import make_pair_table
    return make_pair_table()


def _require_legacy_prefix(model):
    if any(isinstance(module, Top2MoE) and getattr(module, "dispatch_backend", "legacy") != "legacy"
           for module in model.modules()):
        raise ValueError("canonical cache requires legacy prefix dispatch; optimized dispatch is profiling-only")


def build_cache(model, dataset, contexts, *, output_dir, identity, dtype="bfloat16", sequence_batch=2):
    _require_legacy_prefix(model)
    if sequence_batch < 1 or not contexts:
        raise ValueError("sequence_batch and context count must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty cache: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".staging-", dir=output_dir.parent))
    try:
        model.eval().requires_grad_(False)
        device = next(model.parameters()).device
        count = len(contexts)
        width = model.config.hidden_size
        h0 = np.zeros((count, width), dtype=np.float32)
        targets = np.zeros(count, dtype=np.int64)
        positions = np.zeros(count, dtype=np.int64)
        sequences = np.zeros(count, dtype=np.int64)
        by_sequence = {}
        for row, context in enumerate(contexts):
            by_sequence.setdefault(context["sequence_index"], []).append((row, context))
        sequence_entries = list(by_sequence.items())
        with torch.no_grad():
            for start in range(0, len(sequence_entries), sequence_batch):
                entries = sequence_entries[start : start + sequence_batch]
                items = [dataset[sequence_index - dataset.offset] for sequence_index, _ in entries]
                input_ids = torch.from_numpy(np.stack([item["input_ids"] for item in items])).to(device)
                segment_ids = torch.from_numpy(np.stack([item["segment_ids"] for item in items])).to(device)
                with autocast_context(device, dtype):
                    prefix = cache_last_ffn_input(model, input_ids, segment_ids, depth=1)
                for batch_row, (_, rows) in enumerate(entries):
                    item = items[batch_row]
                    for row, context in rows:
                        position = context["position"]
                        h0[row] = prefix[batch_row, position].float().cpu().numpy()
                        targets[row] = item["labels"][position]
                        positions[row] = position
                        sequences[row] = context["sequence_index"]
        np.save(staging / "h0.npy", h0)
        np.save(staging / "targets.npy", targets)
        np.save(staging / "positions.npy", positions)
        np.save(staging / "sequence_indices.npy", sequences)
        _write_jsonl(staging / "episode_ids.jsonl", contexts)
        baseline = _baseline_records(model, dataset, contexts, device, dtype)
        _write_jsonl(staging / "baseline.jsonl", baseline)
        identity = dict(identity)
        excluded = identity.pop("excluded_doc_ids", set())
        atomic_json(staging / "sampling_report.json", _sampling_report(contexts, dataset, excluded))
        seal_cache(staging, identity, count=count, width=width)
        os.replace(staging, output_dir)
        return output_dir
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _finite_json(value):
    if isinstance(value, dict):
        return all(_finite_json(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_json(v) for v in value)
    return not isinstance(value, float) or bool(np.isfinite(value))


def _read_payload(directory, count, width):
    arrays = {name: np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False) for name in ARRAY_FILES}
    for name, array in arrays.items():
        shape = (count, width) if name == "h0" else (count,)
        dtype = np.dtype("float32" if name == "h0" else "int64")
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"cache {name} dtype/shape mismatch")
        for start in range(0, count, 65536):
            if not np.isfinite(array[start:start + 65536]).all():
                raise ValueError(f"cache {name} contains non-finite values")
        if name != "h0" and np.any(array < 0):
            raise ValueError(f"cache {name} contains negative values")
    rows = {name: [json.loads(line) for line in (directory / (name + ".jsonl")).read_text().splitlines()]
            for name in ("episode_ids", "baseline")}
    sampling = json.loads((directory / "sampling_report.json").read_text())
    if any(len(value) != count for value in rows.values()) or sampling.get("states") != count:
        raise ValueError("cache payload row count mismatch")
    if not _finite_json(rows) or not _finite_json(sampling):
        raise ValueError("cache JSON contains non-finite values")
    episodes = rows["episode_ids"]
    if any(not isinstance(row.get("episode_id"), str) or not row["episode_id"] for row in episodes):
        raise ValueError("cache episode IDs must be nonempty strings")
    if len({row["episode_id"] for row in episodes}) != count:
        raise ValueError("cache episode IDs must be unique")
    for index, row in enumerate(episodes):
        if row["sequence_index"] != int(arrays["sequence_indices"][index]) or row["position"] != int(arrays["positions"][index]):
            raise ValueError("cache episode position/sequence mismatch")
    for row in rows["baseline"]:
        required = ("single_teacher_ce", "gate_margin", "single_expert", "pair_canonical_ce") if "single_teacher_ce" in row else ("canonical_ce", "legacy_ce", "gate_margin", "canonical_pair_index")
        if any(key not in row or not isinstance(row[key], (int, float)) or isinstance(row[key], bool) for key in required):
            raise ValueError("cache baseline record schema mismatch")
    return arrays | rows | {"sampling_report": sampling}


def seal_cache(directory, identity, *, count, width):
    """Seal newly generated bytes only; never use to migrate unverified old caches.

    Callers generate into a private staging directory and publish by rename.
    """
    directory = Path(directory)
    if count <= 0 or width <= 0:
        raise ValueError("cache dimensions must be positive")
    payload = _read_payload(directory, count, width)
    files = {}
    for name in ARRAY_FILES:
        path = directory / (name + ".npy")
        array = payload[name]
        files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size,
                            "dtype": str(array.dtype), "shape": list(array.shape)}
    for name in JSON_FILES:
        path = directory / name
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    content = {"format": PAYLOAD_FORMAT, "n": count, "d": width, "files": files}
    manifest = identity | {"n": count, "d": width, "payload": content, "content_sha256": _canonical_hash(content)}
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def load_cache(output_dir, expected_identity):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    content = manifest.get("payload", {})
    if content.get("format") != PAYLOAD_FORMAT:
        raise ValueError("unverified legacy cache: rebuild from the verified backbone and packed corpus into a new cache directory; in-place certification is not supported")
    validate_cache_identity(manifest, expected_identity)
    if _canonical_hash(content) != manifest.get("content_sha256"):
        raise ValueError("cache content identity mismatch")
    count, width = manifest["n"], manifest["d"]
    if type(count) is not int or type(width) is not int or count <= 0 or width <= 0 or content.get("n") != count or content.get("d") != width:
        raise ValueError("cache dimensions mismatch")
    expected_files = {name + ".npy" for name in ARRAY_FILES} | set(JSON_FILES)
    if set(content.get("files", {})) != expected_files:
        raise ValueError("cache payload file inventory mismatch")
    for name, entry in content["files"].items():
        path = output_dir / name
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"cache payload hash mismatch: {name}")
    payload = _read_payload(output_dir, count, width)
    for name in ARRAY_FILES:
        entry = content["files"][name + ".npy"]
        if entry.get("dtype") != str(payload[name].dtype) or entry.get("shape") != list(payload[name].shape):
            raise ValueError(f"cache array metadata mismatch: {name}")
    verified = {key: value for key, value in manifest.items() if key not in ("payload", "n", "d")}
    return payload | {"manifest": manifest, "verified_identity": verified}
