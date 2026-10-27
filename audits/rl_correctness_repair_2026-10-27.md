# RL correctness repair and bounded validation — 2026-10-27

## Upgrade and validation status

Baseline: `0eccfc52258e327612959ee9faf906abf9a8f2e3`
(`research/rl-lm-pair-v1`). This bounded upgrade corrects the frozen pair-teacher
objective, versions incompatible adaptation artifacts, adds opt-in merged sparse
dispatch, aligns terminal timing boundaries, clarifies report references and STOP
denominators, and repairs pair-R2 selection. Canonical training and legacy sparse-v1
backend meanings are preserved. No new GPU quality or speed claim is established.

| Evidence | Result |
|---|---|
| Earlier CPU suite before R2-selection correction | `python -m unittest discover -s tests -t . -v`: 756 tests, 98.314 s, OK (skipped=1); 755 passed, one skipped |
| Clean-baseline precision reproducer | `0eccfc5`: cross-chunk FP32 exact equality fails at 2.38e-7; same-chunk autocast-isolation equality passes |
| Final complete CPU suite | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -v`: 771 tests in 98.883 s, OK (skipped=1); 770 passed. The skipped test requires CUDA BF16 head validation. |
| Syntax / dependency / whitespace checks | `compileall`, `pip check`, all six affected CLI `--help` entrypoints and `git diff --check` passed. |
| R2-selection regressions | `tests.rl_lm.test_priced_selection`: 12 focused CPU tests passed; real R2 resume and missing-reference constructor cases included; final aggregate included below |
| CUDA FP32/BF16 real-state closed-loop parity | Not established |
| Actual target-workload latency / memory | Not established |
| Held-out quality / learning / oracle diagnostic | Not run |

CPU test success does not establish CUDA numerical parity, learning gain or
end-to-end speed. A skip is not a pass. CI configuration alone is not a CI result.

### Pair-R2 selection migration

All new R2 runs, **including lambda=0**, require a preselected compatible R1
checkpoint. The selector minimizes `NLL + lambda * (mean_rounds - 1)` among
candidates with `NLL <= pinned_best_R1_NLL + 0.005`; score ties within `1e-6`
prefer fewer rounds, then earlier checkpoints. The reference's checkpoint hash
and initial selection NLL are pinned. Policy version: `r2_selected_r1_bound_v1`.
Old R2 checkpoints without compatible policy/reference metadata cannot exact-resume
under the corrected rule. R1 keeps `legacy_greedy_nll_v1` selection and compatibility.
This repairs §16.1 selection, not merely display of a reference metric.

### Minimal usage

These are examples, not executed experiments. Replace paths with validated local
artifacts and use a new output/run directory when semantics change.

```bash
# Pair deployment/profile opt-in; the default remains sparse_merged_v1.
python -m experiments.rl_lm_profile --config /path/to/pair-config.json \
  --deployment-backend sparse_merged_v2 --output-dir /path/to/new-profile

# New pair-R2 run: actor initialization and quality reference are separate inputs.
python -m experiments.rl_lm_train --config /path/to/r2-config.json \
  --warmstart-actor /path/to/r1-best.pt \
  --r1-reference-checkpoint /path/to/r1-best.pt
```

For pair evaluation config, set `environment.deployment_backend` to
`sparse_merged_v2` only when explicitly testing the new backend; keep
`environment.training_backend=canonical_all_experts_v1`. The single-expert
backend is a different implementation, not renamed by this pair-backend option.
For exact R2 resume, pass the same `--r1-reference-checkpoint` alongside `--resume`
and retain the original run configuration. Do not combine resume with warmstart.

## Traceability: contract → defect or risk → verification → caveat

The contract descriptions summarize project-relevant requirements; private source
attachments are not reproduced. Test descriptions are required checks, not claims
that every listed check has already been implemented or executed.

| Contract | Baseline defect or risk | Planned/new regression and source boundary | Experimental caveat |
|---|---|---|---|
| Single spec §18.2: original frozen **pair** teacher; same `h0`; pair-softmax pre-gamma update and its own residual logits; student amplitude-1 gate argmax | `AdaptTrainer._update` distilled a frozen **single** update and logits instead | `tests/rl_lm/test_single_adapt.py`: independent hand-computed unequal pair/single target; nontrivial second expert, nonuniform weights, ties; verify CE/KL/update terms, teacher detach/hash, student whitelist; `experiments/rl_lm_single_adapt.py` | Correct teacher fixes the objective, not evidence adaptation helps. Do not infer teacher update by residual subtraction or division by gamma. |
| §18.2–18.4: frozen environment and explicit derived-artifact identity | Old objective checkpoints/gates could otherwise be misrepresented as exact continuation of corrected training | Reject old/missing objective/format on adaptation resume and downstream single-RL gate/artifact loading; checkpoint identity tests in adaptation/single-RL suites; inspect `experiments/rl_lm_single_train.py` | New objective requires a new run and compatible artifact chain. Valid unchanged prefix payload may be reused only after dependency checks; old reward/teacher targets may not. |
| Pair action IDs 1–28 lexicographic, STOP=0; pair-softmax mixture; frozen production numerical contract | Sparse-v1 loops over slot × expert despite its name; performance opportunity, not an incorrect mathematical transition | `tests/rl_lm/test_merged_executor_v2.py`: all pairs, empty buckets, skew, empty/single/noncontiguous rows, invalid IDs, chunk edges; independent manual update and ledger; `rl_lm/executor.py` | Merge assignments per expert, reconstruct row/slot outputs, mix in specified order. Variable GEMM shapes may change BF16 outputs; CPU closeness does not prove identical paths. |
| Canonical fixed chunks preserve episode reward identity; deployment backend is separately named | Silently replacing v1 would change backend provenance and potentially reward reproducibility | Factory/config acceptance tests for **opt-in** `sparse_merged_v2`; retain `canonical_all_experts_v1` and `sparse_merged_v1`; online/transition tests check STOP no-write, max calls, action/CE traces | Do not promote v2 as canonical training reward or claim speed until actual workload parity/profile passes. Python/nonzero overhead remains. |
| Spec production BF16 vs diagnostic FP32 vs differentiable adaptation head are distinct | Shared-looking readout paths can accidentally remove gradients or mix numerical references | Preserve `test_readout.py`, `test_single_adapt.py`, `test_numerical_contract.py`; gradient through frozen head into student experts, explicit precision modes | Broad readout/trainer extraction is deferred; diagnostic FP32 improvement is not production BF16 gain. |
| §18.2 speed gate is original pair vs adapted single; equal timed boundaries | Single routing was precomputed, pair routing was inside timed work; synthetic hidden states only | Timing-path regression instruments norm/gate/selection for both candidates; profile schema checks scope and model identities; `experiments/rl_lm_single_adapt.py`, `experiments/rl_lm_profile.py` | Terminal-only synthetic timing cannot establish end-to-end deployment speed. Same-weight kernel control is additional, not replacement for original-teacher comparison. |
| Pair spec §16.1: all R2 candidates obey pinned R1 quality bound and priced objective | Baseline selected greedy NLL alone with rounds only as a tie-break | `tests/rl_lm/test_priced_selection.py`: cost tradeoff, infeasible candidates, lambda=0 bound, ties, missing/changed reference, exact-resume migration and unchanged R1 | Twelve focused regressions pass; final aggregate also passed; older R2 artifacts must not be relabeled as selected under the new rule. |
| Reporting must identify denominator, reference, sign, and scope | STOP fraction over all recorded decisions can be mistaken for conditional r=1 STOP rate; generic baseline fields can conceal baseline differences | `rl_lm/reporting.py` and `tests/rl_lm/test_reporting.py`: legacy alias unchanged; explicit per-decision vs eligible-r1 fractions/counts; no eligible decisions → null; invalid forbidden STOP rejected; reference identity/sign tests | For v1 max calls 1/2, eligible decisions correspond to r=1. Do not generalize that name to larger horizons. Report original pair and adapted-single baselines separately. |
| RLOO sums voluntary joint logprobs; independent group leave-one-out; KL is transactional guard | Small observed KL/no rollback is not proof of missing guard; prior strength is intentional | Retain `test_rloo_math.py`, replay/resume and `test_update_transaction.py` restore/retry/cost-retention tests; no estimator rewrite | Default prior epsilon=.1 is specified, not a discovered bug. Exploration/objective changes require separately versioned ablations. |
| Terminal layer only; no causal-label features; data/source/checkpoint identity | Reusing old rewards after expert updates or treating exposed data as fresh invalidates claims | Preserve causality/cache/corpus/config-role tests; frozen teacher and per-step student weight-cache invalidation; disjoint-role manifests | All-layer RL, actor+expert joint training, and broad backbone dispatch changes are outside this repair. |

### CPU portability and CI scope

The CPU portability correction in `tests/test_counterfactual.py` retains exact
equality for same-shape autocast isolation. Cross-chunk FP32 comparison uses a
`1e-6` tolerance because the unchanged baseline exhibited `2.38e-7` rounding
difference; this is a test-portability adjustment, not a model change or relaxed
BF16 deployment gate. Minimal test requirements and CPU CI are added separately;
the final aggregate passed as recorded above; no remote CI outcome is asserted.

## Historical findings are not a current defect list

- The Oct25 audit targeted `f2bef42`; subsequent `661eb57` and `571f9f6`
  reviews cover different states. Raw-state policy mismatch, candidate-ID mapping,
  simple-rule fallback, E12 axes/API, and several recovery/accounting issues were
  subsequently repaired. Do not reopen them solely because an old report lists them.
- The Oct26 status index documents later readout, actual-byte identity and isolated
  efficiency work; its older “RL not started” stopping decision predates renewed
  pair-RL implementation at `5e89315` and the `0eccfc5` branch. Neither historical
  “not run” nor historical pass counts describe this repair's current execution.
- Current baseline evidence identifies the wrong adaptation teacher and asymmetric
  terminal timing. Existing transactional KL restore/backoff/retry is real code,
  not a missing implementation. Sparse-v1 dispatch overhead is a performance risk,
  not proof of bad RL mathematics.
- Preserve negative student/candidate-aware/depth/RL runs and the limits of the
  fixed-R1 results. Earlier oracle gaps are label-informed; tiny rule gains and
  cost-matched depth comparisons do not establish deployable RL superiority.
  No corrected code result erases a scientifically valid negative finding.

## Fair measurements and reporting

Use three separate names and matched boundaries:

1. **Executor only:** precomputed normalized rows, routes and weights on both
   sides; report whether residual mixing is included consistently.
2. **Terminal routing + FFN:** norm, gate, selection, expert work, mixture/residual
   on both sides. Compare original frozen pair weights with adapted single weights;
   also retain same-weight kernel controls and legacy-v1/v2 pair comparisons.
3. **End-to-end:** original input through prefix once, observation/actor, every
   executed round and final head on both sides, with identical CE/logits boundary.
   No full original forward followed by overwriting the terminal result.

Declare backend, checkpoint/teacher/objective IDs, dtype, chunk size, device,
software/source snapshot, batch/sequence/active-row shapes and data provenance.
Use real cached/natural states plus synthetic skew controls; separate cold and
steady-state; warm up and repeat in interleaved order. Record host wall/GPU-event
time, reset peak memory per mode, and distinguish logical rounds, expert-token
assignments, physical all-expert/padded work, actor/head work, and valid quality
tokens from all computed positions. Do not transfer a synthetic terminal speed
ratio to full-model speed or attribute a shared-kernel gain to learned routing.

Report `delta_nll = candidate - reference` and `gain_nll = reference - candidate`.
Legacy STOP fraction remains per recorded decision; add conditional r=1 with an
explicit eligible count and null for zero eligibility. Every reference needs
model/backend/precision identity. Production natural-token NLL and clustered
paired uncertainty are primary; cache metrics, sampled actions, training loss,
oracle performance and diagnostic head precision remain labeled diagnostics.

## Bounded diagnostic task cards — UNEXECUTED

### D0 — preregister before labels or learning

Freeze a small document/normalized-group-disjoint train/selection/final partition,
excluding previously exposed confirmation data. Record exact IDs/hashes, source,
checkpoint, backend, target and action semantics, seed, selection procedure,
quality tolerance and minimum worthwhile headroom. Specify row counts, full action
counts (8 or 28), rounds, repeats, expert/head work and a hard runtime/memory cap.
Estimate cost from a permitted tiny pilot, not ordinary LM token throughput.
Do not launch these cards as part of the code repair.

**Exit:** an inspectable manifest and budget exist, or stop with the missing
prerequisite. Do not tune on the final partition or silently increase the cap.

### D1 — finite first-action headroom

Evaluate **all 8 single or all 28 pair first actions** on the declared small states
under one fixed terminal/horizon definition. Include the proper fixed teacher
baseline; STOP only where legal. Reuse expert outputs when mathematically valid,
stream/chunk vocabulary heads, and count all labeling costs. This is **not** full
multi-step path enumeration, a full-path oracle, or proof about extra depth.

Report oracle gain, robust-positive fraction and action/margin distributions with
numerical uncertainty. Keep single and pair protocols/references separate.

**Exit:** no worthwhile robust headroom → preserve the negative result and stop
learning spend. Headroom exists → proceed only inside the preregistered budget.

### D2 — identify where deployability fails

On identical splits/action labels compare fixed teacher, random (analytic expected
cost where available), label-informed oracle, simple cost-sensitive supervised
policy, then a bounded RL arm. Only deployment-visible features may enter policies.
Fit on train, choose on selection, freeze once before final evaluation. Match
quality/cost baselines and account for labeling, training and evaluation work.

Measure held-out regret and quality/cost, disagreement/margins; for RL also entropy,
effective action coverage, nonzero-advantage rate, visits, reward variance, gradient
norm, actual parameter delta and KL. No full seed grid or automatic scale-up.

**Interpretation / stop gates:**
- No oracle headroom: intervention/backbone opportunity is the bottleneck; stop.
- Oracle headroom but no supervised generalization: investigate labels, features or
  state distribution within the bounded diagnostic, not larger RL by default.
- Supervised generalizes but RL fails: isolate exploration/credit assignment before
  changing objectives; keep default protocol and any ablation separately versioned.
- Quality improves but full-model latency does not: backend/batch/control overhead
  problem, not established efficiency; retain the negative deployment result.
- Gate inconclusive or budget exhausted: report uncertainty and stop this card;
  do not silently expand into a larger experiment.

## Scope boundary

No whole-project rewrite, earlier-layer executor replacement, all-layer RL,
joint actor/expert optimization, new CUDA/Triton stack, 300M expansion, broad
hyperparameter grid or server training is part of this repair. Promote a backend
only after stated parity and measured target-workload benefit; retain canonical
and legacy paths when it fails. Correctness tests cannot promise scientific gain.

## Artifact preservation and CPU environment

Adaptation validates the output directory and resume objective/config before any
CLI output writes. Rejected legacy resumes leave existing files unchanged.
Profile defaults are metric/backend-versioned and refuse an existing report;
v2 R1-verification defaults use a separate output directory from historical v1.
CPU validation uses Python 3.12, torch 2.12.1+cpu, numpy 2.5.2, and tokenizers
0.22.2. Original CUDA experiment requirements are retained, not replaced.
