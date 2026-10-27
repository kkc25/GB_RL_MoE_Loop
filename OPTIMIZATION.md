# Verified integrity and opt-in runtime optimization

For an operator checklist in Chinese, see [MIGRATION.md](MIGRATION.md).

This update is based on main `ead65b2`. It does **not** establish better trained
quality, GPU throughput, or end-to-end latency. Existing research results remain
historical records and are not regenerated or reinterpreted.

## What changes by default

- Strict configuration validation rejects boolean/nonfinite numerical values and
  invalid optimizer, schedule, cadence, chunk, state-count and seed domains before
  model allocation.
- Cache v2 verifies actual payload bytes, schemas, shapes, counts and finiteness;
  pair, single and adaptation checkpoints pin the verified cache identity.
  Single-RL reference CE sidecars additionally verify file/array digests, dtype,
  shape and finiteness, and checkpoints pin the actual reference payload.
- Evaluation verifies the actual split and document-grouping identities. Formal
  confirmation additionally pins its sequence count/IDs and hashes in a new
  candidate manifest's `confirmation_scope`.
- Production head output uses one preallocated tensor instead of retaining all
  chunk tensors before concatenation. Head GEMM chunk shapes, tail padding, dtype,
  no-grad behavior and order remain unchanged.
- Pair/single trainers share narrow deep-copy update snapshots and deterministic
  epoch-cursor helpers. Objectives, KL decisions, gradients, selection and cost
  ledgers stay policy-specific. Rollback does not erase consumed work or gradients.
- Profiling uses inference/no-grad consistently and rotates variant order across
  interleaved repeats. Report v3 records backend identities and timing scopes.

## What stays opt-in

### Active terminal deployment

Set `environment.deployment_backend` in a **new** resolved run configuration to
`sparse_active_merged_v3`, or pass this backend to the profiler. This compacts
continuing rows: stopped rows perform no terminal expert work. It retains row
order and scatters completed updates back without changing stopped hidden states.

`canonical_all_experts_v1` remains the fixed-shape training-reward reference.
`sparse_merged_v1` and `sparse_merged_v2` retain their historical behavior and IDs.
All shipped deployment defaults remain v1. Compaction changes GEMM batch shapes;
CPU equality is not evidence of CUDA BF16 path equality.

Small CPU execution-count regression, 8 rows and at most 2 rounds:

| Policy | v2 physical expert rows | v3 physical expert rows |
|---|---:|---:|
| Pair, then all STOP | 32 | 16 |
| Pair, then half STOP | 32 | 24 |
| Pair, then all continue | 32 | 32 |

Actual linear-call input rows independently check these counters. They describe
terminal expert work, **not a measured whole-model speedup**.

### Merged prefix dispatch

The profiler accepts `--prefix-backend merged`; the default is `legacy`. The
candidate merges the two Top-2 assignment slots into one expert bucket while
preserving weighting and gradient paths. Its setter selects only nonterminal
Top2MoE modules and records per-module versioned identities. It is a runtime
profiling choice, not a change to serialized model parameters/configuration.

Canonical cache generation/training must not silently inherit an optimized
prefix: cache identity/build/derivation refuse nonlegacy runtime dispatch.
Promoting it requires a separately versioned numerical/environment contract.

## Migration and preservation

1. Keep historical run directories and their old code/protocol references.
2. Rebuild pair and derived single caches into **new directories**. Do not edit a
   legacy manifest's version or invent checksums to label it verified.
3. Use fresh output directories. Legacy or mismatched cache-bound checkpoints are
   not exact continuations of the new contract; a new run is required.
4. Freeze a new confirmation candidate before evaluating held-out data. The
   `verified_evaluation_scope(corpus_dir, "newtest", sequence_count)` helper reads
   and verifies data without evaluating the actor; put its first returned value
   under `confirmation_scope`. Retain model/reference hashes and all other
   candidate fields. Do not tune the scope after reading held-out metrics.
5. Use a new profiler output directory. Versioned defaults and overwrite guards
   preserve previous evidence.

Example **GPU validation commands, not executed here** (supply existing authorized
checkpoint, corpus, tokenizer and actor paths as required by each entrypoint):

```bash
python -m experiments.rl_lm_verify_r1 --deployment-backend sparse_active_merged_v3 \
  --output-dir runs/validation/new_active_r1
python -m experiments.rl_lm_profile --config PATH_TO_RESOLVED_CONFIG \
  --deployment-backend sparse_active_merged_v3 --prefix-backend legacy \
  --output-dir runs/validation/new_terminal_profile
python -m experiments.rl_lm_profile --config PATH_TO_RESOLVED_CONFIG \
  --deployment-backend sparse_merged_v1 --prefix-backend merged \
  --output-dir runs/validation/new_prefix_profile
```

Validate each candidate independently before combining them. Compare full-model
quality/latency as well as selector and executor components. Preserve dense,
original-pair and strongest R1 baselines; changing physical work is not proof of
research benefit.

## Verification gates

CPU tests cover payload corruption, rehashed cache replacement on resume,
confirmation-scope tampering, invalid configs, profiler mode/order, physical work,
FP32 closed-loop actions/STOP/CE, readout equality, prefix forward/backward
parity, forced retry/rollback/backoff, retained costs and interrupted exact resume.
CUDA tests remain separate skips when no GPU is available. Run the full suite:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -v
```

Final local and remote results are recorded in the pull request. No GPU training,
CUDA speed measurements or new quality experiments are performed by this update.


## Subsequent bounded extension

See [RUN_CONTRACT.md](RUN_CONTRACT.md) for run-approval/state-count isolation,
streaming production evaluation and the profiler's complete factorial controls.
Cache/reference integrity, active deployment and merged-prefix candidates above
remain intact; no new backend is promoted automatically.
