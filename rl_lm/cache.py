from __future__ import annotations

import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from data.packed_lm import verify_packed_split
from eval.counterfactual import cache_last_ffn_input, last_adaptive_layer
from experiments.r1_closed_loop import source_identity
from model.top2 import baseline_top2_pairs
from rl_lm.readout import terminal_ce
from rl_lm.selector import canonical_pair
from train.common import autocast_context
from train.recovery_common import atomic_json, sha256_file

CACHE_FORMAT = "rl-lm-cache-v1"


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


def build_cache(model, dataset, contexts, *, output_dir, identity, dtype="bfloat16", sequence_batch=2):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty cache: {output_dir}")
    staging = output_dir.with_name(output_dir.name + ".staging")
    staging.mkdir(parents=True, exist_ok=False)
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
    with torch.no_grad():
        for start in range(0, len(by_sequence), sequence_batch):
            entries = list(by_sequence.items())[start : start + sequence_batch]
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
    excluded = identity.pop("excluded_doc_ids", set())
    atomic_json(staging / "sampling_report.json", _sampling_report(contexts, dataset, excluded))
    atomic_json(staging / "manifest.json", identity | {"n": count, "d": width})
    os.replace(staging, output_dir)
    return output_dir


def load_cache(output_dir, expected_identity):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    validate_cache_identity(manifest, expected_identity)
    episode_ids = [json.loads(line) for line in (output_dir / "episode_ids.jsonl").read_text().splitlines()]
    baseline = [json.loads(line) for line in (output_dir / "baseline.jsonl").read_text().splitlines()]
    return {
        "manifest": manifest,
        "h0": np.load(output_dir / "h0.npy", mmap_mode="r"),
        "targets": np.load(output_dir / "targets.npy", mmap_mode="r"),
        "positions": np.load(output_dir / "positions.npy", mmap_mode="r"),
        "sequence_indices": np.load(output_dir / "sequence_indices.npy", mmap_mode="r"),
        "episode_ids": episode_ids,
        "baseline": baseline,
        "sampling_report": json.loads((output_dir / "sampling_report.json").read_text()),
    }
