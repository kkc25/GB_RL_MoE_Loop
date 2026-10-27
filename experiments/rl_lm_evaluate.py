"""CLI: post-training selection/confirmation evaluation for RL-LM-Pair-v1.

Spec sections 13.4 / 16.1-16.3 / 20.  Reports the registered baselines on the
chosen split with one frozen denominator (``canonical_pair_r1`` online NLL):

- ``legacy_top2_r1``: original production top-2, fixed one round;
- ``canonical_pair_r1``: canonical teacher pair through the deployment backend;
- ``initial_actor_greedy``: ``checkpoint_initial.pt`` of the same run;
- ``learned_greedy``: joint argmax of the candidate actor (main metric);
- ``learned_sampled``: fixed evaluation random stream, G_eval samples per
  cache state (diagnostic matching the training objective).

The main NLL is token-weighted; the paired interval aggregates per
near-duplicate group first and bootstraps groups (never tokens).  ``--split
newtest`` additionally requires ``--candidate-manifest`` and refuses a corpus
marked ``development_only``; the trainer has no way to read newtest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data.packed_lm import PackedLMDataset, verify_packed_split
from rl_lm.actor import PairActor
from rl_lm.actions import action_table_identity
from rl_lm.checkpoint import load_checkpoint
from rl_lm.cache import cache_identity, load_cache
from rl_lm.cli import load_backbone, resolve_config
from rl_lm.environment import LastFFNEnvironment
from rl_lm.executor import FrozenLastFFNExecutor
from rl_lm.inference import CanonicalTeacherPolicy, forward_with_last_ffn_policy
from rl_lm.ledger import RLLedger
from rl_lm.reporting import METRIC_SCHEMA_VERSION, reference_metrics, evaluation_readout_identity
from rl_lm.rng import make_uniform_table
from rl_lm.trainer import CacheContextProvider
from train.common import autocast_context
from train.recovery_common import atomic_json, sha256_file

EVAL_STREAM_UPDATE = 1_000_000_000  # reserved stream id; training updates stay < planned_updates
BOOTSTRAP_SEED = 20261027


def verified_evaluation_scope(corpus_dir, split, n_sequences):
    """Freeze this result as candidate['confirmation_scope'] BEFORE confirmation.

    Legacy candidates have no verified denominator: create a new candidate
    manifest with this scope, rather than relabeling an existing report.
    This reads and verifies data only; it never evaluates an actor.
    """
    corpus_dir = Path(corpus_dir)
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    packed = verify_packed_split(corpus_dir, split, manifest)
    if n_sequences < 1:
        raise ValueError("evaluation sequence count must be positive")
    if manifest.get("document_list_file"):
        filename = manifest["document_list_file"]
        expected = manifest.get("document_list_sha256")
    elif manifest.get("pipeline_version") == "fixed-depth-confirmation-v1":
        filename = "confirmation_documents.jsonl"
        expected = manifest.get("confirmation_documents_sha256")
    else:
        raise ValueError("unknown corpus document identity format")
    document_bytes = (corpus_dir / filename).read_bytes()
    document_sha = hashlib.sha256(document_bytes).hexdigest()
    if document_sha != expected:
        raise ValueError("document identity file hash mismatch")
    groups = {}
    for line in document_bytes.decode("utf-8").splitlines():
        row = json.loads(line)
        rank = int(row["rank"])
        if rank in groups or not row.get("near_hash"):
            raise ValueError("document grouping requires unique ranks and nonempty near_hash")
        groups[rank] = str(row["near_hash"])
    dataset = PackedLMDataset(corpus_dir, split, limit=n_sequences)
    if len(dataset) != n_sequences:
        raise ValueError("requested evaluation sequence count exceeds split size")
    sequence_ids = []
    for i in range(len(dataset)):
        item = dataset[i]
        sequence_ids.append(int(item["index"]))
        for rank in np.unique(item["doc_ids"][item["labels"] != -100]):
            if int(rank) not in groups:
                raise ValueError(f"doc rank {rank} missing from the corpus document list")
    scope = {
        "format_version": "rl-lm-confirmation-scope-v1",
        "corpus_manifest_sha256": sha256_file(corpus_dir / "manifest.json"),
        "split": split,
        "packed_sha256": packed,
        "document_list_file": filename,
        "document_list_sha256": document_sha,
        "sequence_count": n_sequences,
        "sequence_ids": sequence_ids,
        "sequence_ids_sha256": hashlib.sha256(
            json.dumps(sequence_ids, separators=(",", ":")).encode()).hexdigest(),
    }
    return scope, dataset, groups


def _validate_actor_reference(payload, config, model, *, require_r1=False) -> None:
    stored = payload.get("resolved_config", {})
    environment = stored.get("environment", {})
    stage = "r1" if require_r1 else config.get("stage")
    calls = 1 if require_r1 else config.get("environment.max_calls")
    price = 0 if require_r1 else config.get("cost.lambda")
    if (payload.get("stage") != stage or stored.get("stage") != stage
            or environment.get("max_calls") != calls or environment.get("min_calls") != 1
            or stored.get("cost", {}).get("lambda") != price):
        raise ValueError("actor checkpoint stage/call-limit/price mismatch")
    if payload.get("action_contract") != action_table_identity():
        raise ValueError("actor action contract mismatch")
    if (payload.get("feature_schema") != config.get("policy.feature_schema")
            or payload.get("feature_dim") != 2 * model.config.hidden_size + 20):
        raise ValueError("actor feature contract mismatch")
    if stored.get("policy") != config.to_dict().get("policy"):
        raise ValueError("actor policy contract mismatch")
    backbone = payload.get("backbone_identity", {})
    for key, path in (("checkpoint_sha256", config.get("backbone.checkpoint_path")),
                      ("config_sha256", config.get("backbone.config_path")),
                      ("tokenizer_sha256", config.get("data.tokenizer_path"))):
        if backbone.get(key) != sha256_file(path):
            raise ValueError(f"actor backbone mismatch: {key}")
    for role, split in (("controller_train", config.get("data.train_split")),
                        ("selection", config.get("data.selection_split"))):
        expected = cache_identity(model, model.config,
            checkpoint_path=config.get("backbone.checkpoint_path"), config_path=config.get("backbone.config_path"),
            tokenizer_path=config.get("data.tokenizer_path"), data_dir=config.get("data.corpus_dir"),
            split=split, sampling_seed=config.get("data.sampling_seed"),
            backend_version=config.get("environment.training_backend"), head_contract=config.get("environment.head_contract"),
            expert_chunk=config.get("environment.expert_chunk"))
        actual = payload.get("data_identity", {}).get(role, {})
        for key, value in expected.items():
            if key != "source" and actual.get(key) != value:
                raise ValueError(f"actor data/numerical mismatch: {role}.{key}")
    for key in ("head_chunk", "frozen_compute_dtype", "residual_dtype", "head_contract"):
        if payload.get("numerical_contract", {}).get(key) != config.get("environment." + key):
            raise ValueError(f"actor numerical mismatch: {key}")


def _validate_r1_reference(payload, config, model) -> None:
    _validate_actor_reference(payload, config, model, require_r1=True)


def _load_actor(config, checkpoint_path: Path, device, *, r1_model=None, model=None) -> PairActor:
    payload = load_checkpoint(checkpoint_path)
    if r1_model is not None:
        _validate_r1_reference(payload, config, r1_model)
    elif model is not None:
        _validate_actor_reference(payload, config, model)
    feature_dim = int(payload["feature_dim"])
    actor = PairActor(feature_dim, train_stop=(config.get("stage") == "r2"))
    actor.load_state_dict(payload["actor_state_dict"])
    return actor.to(device).eval()


def _token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Legacy native-head CE and explicit full-logits parity oracle."""
    # Chunked over positions: a monolithic [B*S,V] FP32 CE does not fit in
    # memory next to the BF16 logits at the real 100M scale.
    ce = torch.zeros(labels.shape, dtype=torch.float32)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    flat_ce = ce.reshape(-1)
    rows = 8192
    for start in range(0, flat_labels.shape[0], rows):
        flat_ce[start : start + rows] = F.cross_entropy(
            flat_logits[start : start + rows].float(),
            flat_labels[start : start + rows],
            reduction="none",
            ignore_index=-100,
        ).cpu()
    return ce


def _online_nll(config, model, policy, batch, device, *, max_calls=None, full_logits_reference=False) -> dict:
    output = forward_with_last_ffn_policy(
        model,
        policy,
        batch["input_ids"],
        batch["segment_ids"],
        max_calls=int(config.get("environment.max_calls")) if max_calls is None else max_calls,
        backend=config.get("environment.deployment_backend"),
        compute_dtype=config.get("environment.frozen_compute_dtype"),
        chunk=int(config.get("environment.expert_chunk")),
        head_chunk=int(config.get("environment.head_chunk")),
        output_mode="logits" if full_logits_reference else "token_ce",
        labels=None if full_logits_reference else batch["labels"],
    )
    # Full logits remain an explicit diagnostic oracle, never the default.
    ce = _token_ce(output.logits, batch["labels"]) if full_logits_reference else output.token_ce
    valid = batch["valid"].cpu()
    return {
        "nll": float(ce.sum() / valid.sum()),
        "mean_rounds": float(output.rounds.cpu().reshape(-1)[valid.reshape(-1)].float().mean()),
        "ce": ce,
    }


def _legacy_nll(model, batch, device) -> dict:
    with torch.no_grad(), autocast_context(device, "bfloat16"):
        # Preserve the native monolithic head contract, but do not ask the
        # backbone to allocate an unused full FP32 CE/loss before chunked CE.
        legacy = model(batch["input_ids"], labels=None, depth=1, segment_ids=batch["segment_ids"])
    ce = _token_ce(legacy.logits, batch["labels"])
    return {"nll": float(ce.sum() / batch["valid"].sum()), "mean_rounds": 1.0, "ce": ce}


def _grouped_ce(ce: torch.Tensor, valid: torch.Tensor, doc_ids: torch.Tensor, group_of_rank) -> dict:
    """Aggregate CE sums and token counts per near-duplicate group."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    ce_flat = ce.reshape(-1).cpu().numpy()
    valid_flat = valid.reshape(-1).cpu().numpy()
    docs_flat = doc_ids.reshape(-1).cpu().numpy()
    for value, is_valid, rank in zip(ce_flat, valid_flat, docs_flat):
        if not is_valid:
            continue
        group = group_of_rank.get(int(rank))
        if group is None:
            raise ValueError(f"doc rank {rank} missing from the corpus document list")
        sums[group] = sums.get(group, 0.0) + float(value)
        counts[group] = counts.get(group, 0) + 1
    return {"sum": sums, "n": counts}


def _paired_bootstrap(learned: dict, reference: dict, *, repetitions: int, confidence: float) -> dict:
    groups = sorted(set(learned["sum"]) & set(reference["sum"]))
    if not groups:
        raise ValueError("no shared near-duplicate groups between candidates")
    diff = np.array([learned["sum"][g] - reference["sum"][g] for g in groups], dtype=np.float64)
    tokens = np.array(
        [min(learned["n"][g], reference["n"][g]) for g in groups], dtype=np.float64
    )
    if not np.allclose(
        [learned["n"][g] for g in groups], [reference["n"][g] for g in groups]
    ):
        raise ValueError("paired candidates disagree on per-group token counts")
    point = float(diff.sum() / tokens.sum())
    rng = np.random.Generator(np.random.Philox(BOOTSTRAP_SEED))
    stats = np.empty(repetitions, dtype=np.float64)
    for rep in range(repetitions):
        take = rng.integers(0, len(groups), size=len(groups))
        stats[rep] = diff[take].sum() / tokens[take].sum()
    alpha = 1.0 - confidence
    return {
        "groups": len(groups),
        "tokens": int(tokens.sum()),
        "delta_nll": point,
        "ci_low": float(np.quantile(stats, alpha / 2)),
        "ci_high": float(np.quantile(stats, 1 - alpha / 2)),
        "repetitions": repetitions,
        "confidence": confidence,
    }


def _sampled_cache_nll(config, model, actor, cache, device, n_states: int) -> dict:
    ledger = RLLedger()
    executor = FrozenLastFFNExecutor(
        model,
        chunk=int(config.get("environment.expert_chunk")),
        compute_dtype=config.get("environment.frozen_compute_dtype"),
        ledger=ledger,
    )
    environment = LastFFNEnvironment(
        model,
        max_calls=int(config.get("environment.max_calls")),
        executor=executor,
        head_chunk=int(config.get("environment.head_chunk")),
        price_lambda=float(config.get("cost.lambda")),
        ledger=ledger,
    )
    provider = CacheContextProvider(cache)
    n_states = min(n_states, len(provider))
    contexts = provider.gather(list(range(n_states)), device)
    groups = int(config.get("evaluation.stochastic_samples_per_context"))
    uniforms = make_uniform_table(
        int(config.get("actor_seed")),
        EVAL_STREAM_UPDATE,
        contexts.episode_ids,
        groups,
        int(config.get("environment.max_calls")),
    ).to(device)
    rollout, _ = environment.rollout(contexts, actor, uniforms, policy_version=0)
    ref = contexts.ref_ce
    return {
        "sampled_nll": float(rollout.terminal_ce.mean()),
        "reference_nll": float(ref.mean()),
        "gain_nll": float(ref[:, None].mean() - rollout.terminal_ce.mean()),
        "mean_rounds": float(rollout.rounds.float().mean()),
        "states": n_states,
        "samples_per_context": groups,
        "eval_stream_update": EVAL_STREAM_UPDATE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="resolved run config JSON")
    parser.add_argument("--actor-checkpoint", required=True, help="candidate actor checkpoint (checkpoint_best.pt)")
    parser.add_argument("--r1-reference-checkpoint", help="strongest R1 checkpoint chosen on selection BEFORE confirmation; never selected on this split")
    parser.add_argument("--split", required=True, choices=["selection", "calibration", "newtest"])
    parser.add_argument("--mode", default="online", choices=["online"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidate-manifest", default=None, help="required for --split newtest")
    parser.add_argument("--sequences", type=int, default=None, help="override online evaluation sequence count")
    parser.add_argument("--sampled-states", type=int, default=2048, help="cache states for the sampled diagnostic")
    parser.add_argument("--output", default=None, help="report JSON path")
    parser.add_argument("--overwrite-output", action="store_true", help="explicitly replace an existing report; omitted by default")
    args = parser.parse_args()

    config = resolve_config(args.config, device=args.device)
    device = torch.device(config.get("runtime.device"))
    corpus_dir = Path(config.get("data.corpus_dir"))
    corpus_manifest = json.loads((corpus_dir / "manifest.json").read_text())
    actor_path = Path(args.actor_checkpoint)
    output = Path(args.output) if args.output else actor_path.parent / f"evaluate_{args.split}_metrics_v4.json"
    if output.exists() and not args.overwrite_output:
        raise FileExistsError(f"report already exists: {output}; choose a new --output or explicitly --overwrite-output")
    actor_sha = sha256_file(actor_path)

    candidate = None
    candidate_sha = None
    if args.split == "newtest":
        if args.candidate_manifest is None:
            raise SystemExit("--split newtest requires --candidate-manifest PATH")
        candidate_bytes = Path(args.candidate_manifest).read_bytes()
        candidate = json.loads(candidate_bytes)
        candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
        required = {"actor_checkpoint_sha256", "price_lambda", "backend", "head_contract", "selection_rule"}
        missing = required - set(candidate)
        if missing:
            raise SystemExit(f"candidate manifest missing fields: {sorted(missing)}")
        if candidate["actor_checkpoint_sha256"] != actor_sha:
            raise SystemExit("candidate manifest actor hash does not match --actor-checkpoint")
        for key, value in (("price_lambda", config.get("cost.lambda")),
                           ("backend", config.get("environment.deployment_backend")),
                           ("head_contract", config.get("environment.head_contract"))):
            if candidate[key] != value:
                raise ValueError(f"candidate manifest {key} mismatch")
        if candidate.get("r1_reference_checkpoint_sha256") and not args.r1_reference_checkpoint:
            raise ValueError("candidate manifest pins an R1 reference; --r1-reference-checkpoint required")
        if (corpus_manifest.get("rl_lm_corpus") or {}).get("development_only"):
            raise SystemExit(
                "corpus is development_only; the independent confirmation entry refuses it "
                "until exclusion coverage is rebuilt"
            )

    if args.split == "newtest":
        if "confirmation_scope" not in candidate:
            raise ValueError("legacy candidate lacks confirmation_scope; freeze a NEW candidate "
                             "using verified_evaluation_scope before confirmation; do not relabel old reports")
        n_sequences = candidate["confirmation_scope"]["sequence_count"]
        if args.sequences is not None and args.sequences != n_sequences:
            raise ValueError("--sequences differs from frozen confirmation scope")
    else:
        n_sequences = (args.sequences if args.sequences is not None
                       else int(config.get("evaluation.online_selection_sequences")))
    evaluated_scope, dataset, group_of_rank = verified_evaluation_scope(
        corpus_dir, args.split, n_sequences)
    if candidate is not None and candidate["confirmation_scope"] != evaluated_scope:
        raise ValueError("candidate confirmation scope mismatch")
    model, _ = load_backbone(config, device)
    items = [dataset[i] for i in range(len(dataset))]
    batch = {
        "input_ids": torch.from_numpy(np.stack([item["input_ids"] for item in items])).to(device),
        "segment_ids": torch.from_numpy(np.stack([item["segment_ids"] for item in items])).to(device),
        "labels": torch.from_numpy(np.stack([item["labels"] for item in items])).to(device),
    }
    batch["valid"] = batch["labels"] != -100
    doc_ids = torch.from_numpy(np.stack([item["doc_ids"] for item in items])).to(device)

    learned_actor = _load_actor(config, actor_path, device, model=model)
    initial_path = actor_path.parent / "checkpoint_initial.pt"
    initial_actor = _load_actor(config, initial_path, device, model=model) if initial_path.exists() else None

    r1_identity = None
    if args.r1_reference_checkpoint:
        r1_path = Path(args.r1_reference_checkpoint)
        r1_sha = sha256_file(r1_path)
        if args.split == "newtest" and candidate.get("r1_reference_checkpoint_sha256") != r1_sha:
            raise ValueError("candidate manifest must pin r1_reference_checkpoint_sha256 before newtest")
        r1_actor = _load_actor(config, r1_path, device, r1_model=model)
        r1_identity = {"name": "selected_r1_greedy", "checkpoint": str(r1_path),
                       "checkpoint_sha256": r1_sha, "max_calls": 1,
                       "selection": "caller-declared strongest R1, frozen before this evaluation",
                       "uncertainty": "point estimate only; canonical-reference CI does not apply"}
    results: dict[str, dict] = {}
    results["legacy_top2_r1"] = _legacy_nll(model, batch, device)
    results["canonical_pair_r1"] = _online_nll(config, model, CanonicalTeacherPolicy(), batch, device, max_calls=1)
    if int(config.get("environment.max_calls")) == 2:
        results["canonical_pair_r2"] = _online_nll(config, model, CanonicalTeacherPolicy(), batch, device, max_calls=2)
    if r1_identity is not None:
        results["selected_r1_greedy"] = _online_nll(config, model, r1_actor, batch, device, max_calls=1)
    if initial_actor is not None:
        results["initial_actor_greedy"] = _online_nll(config, model, initial_actor, batch, device)
    results["learned_greedy"] = _online_nll(config, model, learned_actor, batch, device)

    reference_nll = results["canonical_pair_r1"]["nll"]
    summary = {
        name: {"nll": res["nll"], "mean_rounds": res["mean_rounds"],
               "delta_nll": res["nll"] - reference_nll, "gain_nll": reference_nll - res["nll"]}
        for name, res in results.items()
    }

    # Cache-side sampled diagnostic (fixed evaluation stream, G_eval samples).
    if args.split in (
        config.get("data.train_split"),
        config.get("data.selection_split"),
        config.get("data.calibration_split"),
    ):
        expected = cache_identity(
            model, model.config,
            checkpoint_path=config.get("backbone.checkpoint_path"),
            config_path=config.get("backbone.config_path"),
            tokenizer_path=config.get("data.tokenizer_path"),
            data_dir=config.get("data.corpus_dir"),
            split=args.split,
            sampling_seed=config.get("data.sampling_seed"),
            backend_version=config.get("environment.training_backend"),
            head_contract=config.get("environment.head_contract"),
            expert_chunk=config.get("environment.expert_chunk"),
        )
        cache = load_cache(Path(config.get("data.cache_root")) / args.split, expected)
        sampled = {"learned_sampled": _sampled_cache_nll(config, model, learned_actor, cache, device, args.sampled_states)}
        if initial_actor is not None:
            sampled["initial_actor_sampled"] = _sampled_cache_nll(
                config, model, initial_actor, cache, device, args.sampled_states
            )
    else:
        sampled = {}

    learned_groups = _grouped_ce(results["learned_greedy"]["ce"], batch["valid"], doc_ids, group_of_rank)
    canonical_groups = _grouped_ce(results["canonical_pair_r1"]["ce"], batch["valid"], doc_ids, group_of_rank)
    paired = _paired_bootstrap(
        learned_groups,
        canonical_groups,
        repetitions=int(config.get("evaluation.bootstrap_repetitions")),
        confidence=float(config.get("evaluation.bootstrap_confidence")),
    )

    report = {
        "report_schema_version": "rl-lm-verified-evaluation-v4",
        "spec_version": config.get("spec_version"),
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "evaluation_readout": evaluation_readout_identity(config),
        "evaluation_readout_scope": "online policy variants only; sampled_cache uses its existing terminal reward readout",
        "evaluation_readout_exceptions": {
            "legacy_top2_r1": {
                "version": "native_full_logits_chunked_fp32_ce8192_v1",
                "head_chunk": None,
                "compute_dtype": "bfloat16" if device.type == "cuda" else "float32",
                "ce_dtype": "float32", "ce_window": 8192,
                "materializes_full_logits": True,
                "unused_model_loss": False,
            },
        },
        "online_gain_reference": {"name": "canonical_pair_r1", "max_calls": 1},
        "historical_note": "v1 mislabeled the configured-depth teacher as canonical_pair_r1 in R2; v2 separates R1 and forced R2. Historical artifacts are not rewritten.",
        "learned_vs_selected_r1": reference_metrics(results["learned_greedy"]["nll"], results["selected_r1_greedy"]["nll"], r1_identity) if r1_identity else None,
        "selected_r1_status": "provided" if r1_identity else "not_provided; no claim of gain over strongest R1",
        "learned_vs_forced_r2": reference_metrics(results["learned_greedy"]["nll"], results["canonical_pair_r2"]["nll"], {"name": "canonical_pair_r2", "max_calls": 2}) if "canonical_pair_r2" in results else None,
        "stage": config.get("stage"),
        "split": args.split,
        "mode": args.mode,
        "corpus_dir": str(corpus_dir),
        "corpus_development_only": bool((corpus_manifest.get("rl_lm_corpus") or {}).get("development_only")),
        "actor_checkpoint": str(actor_path),
        "actor_checkpoint_sha256": actor_sha,
        "candidate_manifest": args.candidate_manifest,
        "candidate_manifest_sha256": candidate_sha,
        "candidate_manifest_content": candidate,
        "evaluated_data_identity": evaluated_scope,
        "sequences": int(batch["input_ids"].shape[0]),
        "valid_tokens": int(batch["valid"].sum()),
        "backend": config.get("environment.deployment_backend"),
        "head_contract": config.get("environment.head_contract"),
        "online": summary,
        "sampled_cache": sampled,
        "paired_bootstrap_learned_greedy_minus_canonical": paired,
        "minimum_practical_nll_gain": float(config.get("evaluation.minimum_practical_nll_gain")),
    }
    if output.exists() and not args.overwrite_output:
        raise FileExistsError(f"report appeared during evaluation: {output}; refusing overwrite")
    atomic_json(output, report)
    printable = {k: v for k, v in report.items() if k != "sampled_cache" or True}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
