from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from data.packed_lm import PackedLMDataset
from rl_lm.cache import build_cache, cache_identity, load_document_identities, plan_contexts
from rl_lm.cli import load_backbone, resolve_config, write_resolved_config
from rl_lm.run_directory import guard_run_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--corpus-dir")
    parser.add_argument("--tokenizer")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = resolve_config(args.config, checkpoint=args.checkpoint, corpus_dir=args.corpus_dir, tokenizer=args.tokenizer, device=args.device)
    guard_run_directory(Path(config.get("output_dir")))
    write_resolved_config(config, Path(config.get("output_dir")) / "resolved_config.json")
    device = torch.device(config.get("runtime.device"))
    model, _ = load_backbone(config, device)
    corpus_id = config.sha256[:16]
    document_records = load_document_identities(config.get("data.corpus_dir"))
    results = {}
    for role, split, states_key in (
        ("controller_train", config.get("data.train_split"), "train_states"),
        ("selection", config.get("data.selection_split"), "selection_states"),
        ("calibration", config.get("data.calibration_split"), "calibration_states"),
    ):
        dataset = PackedLMDataset(config.get("data.corpus_dir"), split)
        contexts = plan_contexts(dataset, config.get(f"data.{states_key}"), seed=config.get("data.sampling_seed"), corpus_id=corpus_id, document_records=document_records)
        identity = cache_identity(model, model.config, checkpoint_path=config.get("backbone.checkpoint_path"), config_path=config.get("backbone.config_path"), tokenizer_path=config.get("data.tokenizer_path"), data_dir=config.get("data.corpus_dir"), split=split, sampling_seed=config.get("data.sampling_seed"), backend_version=config.get("environment.training_backend"), head_contract=config.get("environment.head_contract"), expert_chunk=config.get("environment.expert_chunk"))
        output_dir = Path(config.get("data.cache_root")) / split
        build_cache(model, dataset, contexts, output_dir=output_dir, identity=identity, dtype=config.get("environment.frozen_compute_dtype"), sequence_batch=config.get("data.prefix_sequence_batch"))
        results[role] = {"cache_dir": str(output_dir), "states": len(contexts)}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
