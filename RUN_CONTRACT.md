# Run contracts and bounded evaluation

This update follows main `4e3332d`. It closes cross-entrypoint approval/count
mismatches without changing objectives, sample counts, selection cadence or
training-reward numerical contracts.

## One approval describes one verified experiment

`rl_lm/run_contract.py` shares role loading and exact count checks across pair,
single and adaptation entrypoints. The contract schema is
`rl-lm-run-contract-v1`, with a canonical SHA256 of the resolved configuration
and all three development roles' split names, requested/actual state counts and
verified cache identities. Confirmation data is not loaded by training/preflight.

All four role names must be nonempty, distinct leaf identifiers. Train, selection
and calibration cannot use `newtest` or the configured confirmation role. Name
checks supplement, not replace, the corpus's byte/document isolation.

Pair preflight uses atomic `checking -> ready` or `checking -> failed` records.
The training entrypoint independently verifies actual inputs and compares the
entire contract. A stale, modified or absent approval is not sufficient. An
8-state approval cannot authorize a subsequently configured 64-state run, even
if the cache remains internally valid and a later preflight failed.

Only `runtime.device` is normalized for documented device portability. The actual
preflight environment is recorded separately; dtype, chunks, backend, CUDA
requirements and other configuration changes are not ignored. Existing checkpoint
RNG/device restrictions still apply; normalization does not promise arbitrary
CUDA-to-CPU exact continuation.

Single/adaptation bind the common contract to checkpoint identities as well as
using the same role/count verification. New or mismatched data requires a new
run. See [MIGRATION.md](MIGRATION.md), including the treatment of protected
populated directories and legacy approvals.

## Streaming readout, not a new numerical objective

The explicit `output_mode="token_ce"` pair/single inference path runs the same
prefix, terminal policy, action/STOP logic and sequence batching, then streams
head tiles into a bounded vocabulary buffer. Default public full-logits APIs
remain available unchanged.

- Head chunks, tail padding, compute dtype and no-grad behavior are unchanged.
- CE keeps the original 8192-position FP32 windows, ignored-label zeros and
  per-window device reduction order.
- Output retains CPU per-token CE for group aggregation/bootstrap, not only an
  average. Training's chunk-64 reward CE remains a separate API.
- Pair/single selection and adaptation production-reference evaluation use the
  streaming path; differentiable FP32 adaptation training does not.
- Reports record `streaming_fixed_head_fp32_ce8192_v1`. The numerical head
  contract and objective remain unchanged. Full-logits reference paths remain
  explicit numerical test oracles.

With vocab32768 and the default BF16 head, the reusable 8192-row logits buffer is
512 MiB rather than retaining all 128×1024 rows (8 GiB). FP32 CE inputs/workspaces,
head tiles, model/prefix state and transfers are **additional** allocations.
This is a capacity calculation, not measured total VRAM or a latency result.

The original native Top-2 baseline is intentionally excluded from chunked-head
replacement: its monolithic head GEMM is part of that reference. It still uses
full logits, but no longer computes an unused full loss before the separately
chunked evaluation CE. Do not label the entire evaluation program fully bounded
by the streaming buffer or infer faster wall time from reduced storage.

## Complete end-to-end controls

The optional profiler mode interleaves all four cells for each identical policy
and call cap:

| Prefix | Terminal |
|---|---|
| legacy | sparse_merged_v1 |
| legacy | sparse_active_merged_v3 |
| merged | sparse_merged_v1 |
| merged | sparse_active_merged_v3 |

Native original-MoE controls are separately restored to completely legacy
execution. The historical terminal above is the online v1 executor, not an alias
for native original MoE. Inputs, policy weights and call caps remain matched;
actual runtime preparation and verification occur outside the timed interval,
including before each measured invocation. Component fixtures use legacy prefix.
Each cell reports its actual effective prefix/terminal identity independently
of the top-level requested settings. New profile schema/output version is v4.

Example for a future authorized GPU validation run, **not executed here**:

```bash
python -m experiments.rl_lm_profile --config PATH_TO_RESOLVED_CONFIG \
  --r1-actor PATH_TO_R1_ACTOR --r2-actor PATH_TO_R2_ACTOR \
  --factorial-end-to-end --output-dir runs/validation/fresh_factorial_v4
```

Supply the other required backbone, dense-model, corpus and tokenizer paths from
your environment. First validate numerical/action/STOP parity on the target GPU,
then measure quality, full-model latency and component costs. The profiler does
not certify quality or automatically promote a candidate backend.

## Tests and scope

Regressions cover role aliases/holdout misuse, stale ready records, independently
changed counts/content, historical-output preservation, device-only approval
compatibility, contract-bound resume, full-logits versus streaming CE/NLL/action
parity, allocation bounds, unequal-group bootstrap and untimed factorial state
switching. CPU and mocked timer checks do not establish CUDA parity/speed.

Run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -v
```

Final local test counts and exact-head CI are recorded in the PR. Historical
artifacts are preserved; no GPU experiment or new training campaign is launched.
