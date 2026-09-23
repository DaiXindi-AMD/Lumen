# Fresh MXFP4 Training/Optimization Notes

### [2026-09-20 kfd-remeasurement-reset]
- User directive: all profiling and timing artifacts produced before this entry
  are invalid as performance evidence for the current optimization decision.
  The historical entries below are retained only as a work log and must not be
  cited to accept or reject a candidate in this round.
- Hardware access was rechecked directly: `/dev/kfd` is readable/writable by
  the current user through the `render` group, PyTorch sees 8 AMD Instinct
  MI350X devices, and no training process was occupying them at session start.
- Frozen source snapshot before new measurements: Lumen commit
  `9d85c8adb5159cc5765680bbc4c2230bb00e74e4`; tracked binary diff SHA256
  `a167b3d9e97404b7d7aebeb83cb535999e364ea588c0ed0b3d6734edf29439c3`;
  porcelain-status SHA256
  `3701ca5435893c0647ffb0957008ea72091b33f02c6d11f99b73d376d7d9bdd2`.
- New matched comparison will use the same Qwen3-8B initialization, data order,
  batch geometry, FSDP2 policy, reduction dtype, attention/norm/RoPE/cross
  entropy paths, optimizer, and scheduler. The intentional precision-policy
  difference is BF16 linears versus MXFP4 linears with the configured BF16
  guard layers. A fresh per-session autotune cache will be generated rather
  than reading an earlier cache.
- Acceptance gate: MXFP4 steady-state step time must be at most 62.5% of the
  newly measured matched BF16 time (at least 1.6x), with finite losses/gradient
  norms and no material short-run validation regression. Longer convergence is
  required before making a full training-quality claim.
- Status: open; no current performance number is accepted until new BF16 and
  MXFP4 runs complete under this frozen protocol.

### [2026-09-20 kfd-remeasurement-mxfp4-smoke]
- New evidence only: an 8-GPU, three-step MXFP4 smoke run completed from the
  frozen source snapshot using the matched candidate protocol (FSDP2
  `full_shard`, BF16 reduction, retained parameters, no activation
  checkpointing, AITER attention, Lumen norm, fused RoPE/cross entropy, and five
  BF16 tail layers).
- Runtime routing reported 217 quantized linear modules and 36 BF16-skipped
  linears. The newly created session-local autotune cache has SHA256
  `a0993a928ed8a821617b31da3261b932b823d1be87ccce4a441b4d9c01fa99d`;
  no earlier cache was read.
- Step 1 included compilation/autotuning and took 43546.9 ms, so it is excluded
  from performance. Steps 2 and 3 took 6625.5 and 6585.5 ms. These two samples
  are only a smoke signal, not an accepted throughput estimate.
- Loss stayed finite (12.7957 to 12.7886), gradient norm stayed finite
  (5.781e0 to 5.688e0), validation loss at step 3 was 12.7886, and peak
  allocated memory was 138.9 GiB/GPU. No runtime fallback, OOM, NaN, skipped
  update, or kernel failure was logged.
- A known `torch.library` weakref cleanup traceback appeared after the explicit
  `Training complete` message; torchrun returned exit code 0. Treat it as a
  teardown defect, not as a successful-path kernel failure, while continuing
  to record it.
- Artifact:
  `/home/xdai/profile-results/lumen-mxfp4-remeasure-20260920/smoke_mxfp4/`.
- Status: smoke passed; matched BF16 smoke and longer unprofiled timing remain.

### [2026-09-20 kfd-remeasurement-bf16-smoke-and-pairing-gap]
- New matched BF16 smoke completed with the same FSDP2, BF16 reduction,
  retained-parameter, no-checkpoint, attention/norm/RoPE/cross-entropy, data,
  batch, and optimizer settings as the MXFP4 smoke. Steps 2 and 3 took 8776.8
  and 8300.0 ms (mean/median 8538.4 ms); peak memory was 133.8 GiB/GPU.
- Relative to the MXFP4 smoke's two post-JIT samples (mean/median 6605.5 ms),
  the provisional ratio is 1.2926x. Two samples per arm are insufficient for a
  throughput decision, but they show that the current candidate is not
  obviously at the 1.6x target and justify paired profiling.
- BF16 loss/gradient norm stayed finite, final validation loss was finite, and
  no runtime fallback, OOM, NaN, skipped update, or kernel failure was logged.
  The same post-success `torch.library` weakref teardown traceback appeared.
- Correctness audit found that the shuffled pretraining DataLoader has no
  independent generator. Its order therefore depends on the global Torch RNG
  state after precision-specific model transformation. Until an explicit
  rank-local data generator and batch fingerprints are added, differences in
  rank-0 training loss cannot be treated as paired precision evidence. The
  deterministic validation loader remains usable as a coarse smoke screen.
- Artifact:
  `/home/xdai/profile-results/lumen-mxfp4-remeasure-20260920/smoke_bf16/`.
- Status: performance smoke passed; strict accuracy pairing is open and must be
  fixed before formal timing/accuracy acceptance runs.

### [2026-09-20 strict-paired-baseline-ab]
- Source snapshot for both arms: Lumen commit
  `9d85c8adb5159cc5765680bbc4c2230bb00e74e4`, tracked diff SHA256
  `744c590a9ca5b151d44245d059b4da98ff8c363d6b99afb4e28d493524b260e8`,
  AITER commit `e35bb17f4f815903bf73598facedbb321e15af28`.
- The two direct `torchrun` commands were identical except `--mode`. Both used
  Qwen3-8B, seq8192, MBS2, GBS128/GA8, FSDP2 `full_shard`, retained parameters,
  no activation checkpointing, BF16 reduction, AITER attention, Lumen norm,
  fused RoPE/cross entropy, seed 1234, and the frozen train/validation files.
- Pairing evidence passed: BF16 and MXFP4 had identical pre-quantization model
  init SHA256
  `b63b3be6fba1242f837ed8e677ddfcb25d3252b7b6e709d3cf5d3d5b9072dd12`,
  and every rank's step-1 input/label rolling SHA256 matched across arms.
- Unprofiled step 11-30 results: BF16 mean/median 8162.27/8113.75 ms; MXFP4
  tail-5 mean/median 6117.07/6028.50 ms. The new strict speedup is therefore
  1.3343x mean and 1.3459x median, below 1.6x. At the measured BF16 speed, the
  MXFP4 target is 5101.42/5071.09 ms, leaving 1015.65/957.41 ms to remove.
- Short-run numerical screen: final global validation NLL was 9.1534 BF16 and
  9.1540 MXFP4; delta NLL +0.0006 corresponds to approximately +0.0600%
  perplexity. All logged losses and gradient norms were finite. This passes a
  smoke non-regression screen but is not a long-horizon convergence claim.
- Peak allocated memory was 133.8 GiB/GPU BF16 and 138.9 GiB/GPU MXFP4. No
  fallback, OOM, NaN, skipped update, or kernel failure was logged. The known
  post-success `torch.library` teardown traceback remained.
- Artifacts:
  `/home/xdai/profile-results/lumen-mxfp4-remeasure-20260920/baseline_ab_bf16/`
  and
  `/home/xdai/profile-results/lumen-mxfp4-remeasure-20260920/baseline_ab_mxfp4/`.
- Status: open; strict 1.6x target not met. Next action is a newly measured
  matched BF16/MXFP4 profile pair, followed by single-variable A/B changes.

This file intentionally contains only evidence collected after the 2026-09-20
rollback. Per the user's instruction, earlier saved profiling results are not
used as performance evidence in this optimization round.

### [2026-09-20 clean-fresh-exact-a4w4-sweep]
- New evidence only: a single-GPU exact-shape sweep ran under the exclusive GPU
  lock against the frozen clean-fresh Lumen/AITER trees.  The runner recorded
  identical before/after tree digests and exit status 0.  All 73 tested
  candidates produced finite output and passed the full-output FP32-reference
  gate at SNR >= 25 dB; the three winning/incumbent paths measured about
  55.6 dB.
- `(16384,4096,12288)`: the incumbent ASM 256x256 split-0 remained best.  Its
  2048-sample mean/median/P99/max were
  0.381654/0.380444/0.425457/0.617247 ms.  The previously suspected
  hundreds-of-milliseconds kernel tail was not reproduced, so it is not being
  treated as a stable optimization target.
- `(12288,4096,16384)`: ASM 128x512 split-0 beat the incumbent preshuffled
  Triton default.  Over 2048 samples, Triton mean/median/P99 were
  0.610137/0.605647/0.660300 ms and ASM were
  0.380438/0.368884/0.433159 ms.  This is 1.6038x by mean and 1.6418x by
  median; the ASM result was bitwise equal to the incumbent output in the
  correctness check.  At the freshly measured 496 calls/step this has a
  kernel-only estimate of about 114-117 ms/step, large enough for an
  integration experiment but only about 11% of the current roughly 1 s target
  gap.
- `(16384,12288,4096)`: ASM 256x256 split-0 beat the incumbent ASM 128x512 by
  only 1.0042x on the 2048-sample medians, an estimated 1.4 ms/step at 744
  calls.  This is below the materiality gate and is rejected for now.
- The known post-success `torch.library` weakref teardown traceback remained;
  it did not change the successful process exit or result files.
- Artifact:
  `/home/xdai/profile-results/lumen-mxfp4-a4w4-fresh-20260920-191450/`.
  Shape JSON SHA256 values are `27e094f8...` for 16384x4096x12288,
  `4fb7ac32...` for 12288x4096x16384, and `a818699c...` for
  16384x12288x4096.
- Status: open.  Advance only the 12288x4096x16384 ASM candidate to at least
  three fresh-process interleaved ABBA runs before changing dispatch or
  training configuration.

### [2026-09-20 clean-fresh-a4w4-abba]
- The `(12288,4096,16384)` preshuffled-Triton incumbent and ASM 128x512
  split-0 candidate were measured in three independent fresh Python processes
  with CUDA events and interleaved `ABBA`, `BAAB`, and `ABBA` ordering.  Each
  process collected 1024 quads, or 2048 launches per backend, after warmup.
- All three processes passed finite output, full-output FP32-reference SNR
  (55.6165 dB for both paths), and bitwise equality between incumbent and
  candidate.  Per-process ratios of means were 1.8163x, 1.7942x, and 1.8292x;
  ratios of medians were 1.8174x, 1.8104x, and 1.8311x.  ASM won 3071 of 3072
  paired quads.
- Pooled mean fell from 0.636613 to 0.351098 ms and pooled median from
  0.637317 to 0.350204 ms.  The process-preserving block-bootstrap 95% CI for
  the mean ratio was [1.8023x, 1.8196x].  At 496 fresh-profile calls/step, the
  kernel-only saving estimate is 141.6 ms/step.
- The original runner returned 10 only after all three result JSONs were
  complete: its post-run idle check observed a transient KFD proc directory
  whose `/proc/<pid>` had already disappeared.  A separate robust finalizer
  confirmed the GPU was idle and that both repository tree digests still
  matched preflight, then generated the bootstrap summary without rerunning or
  modifying the measurements.
- Artifact:
  `/home/xdai/profile-results/lumen-mxfp4-a4w4-abba-fresh-20260920-192136/`.
- Decision: accept the kernel-local candidate.  Add its exact gfx950/256-CU
  tuning row in AITER so Lumen's existing fail-closed ASM registry can expose
  it automatically; then verify actual Lumen dispatch before 8-GPU timing.

### [2026-09-20 clean-fresh-a4w4-tuned-row]
- Added exact AITER tuning row `(gfx950,256,12288,4096,16384)` selecting
  `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E`, split-K 0.  The
  recorded 351.0976 us is the pooled ABBA mean; derived throughput fields are
  4697.46 TFLOP/s and 668.99 GB/s under AITER's A4W4 accounting.
- Static validation: the CSV has exactly one row for the key; both explicit
  and default Lumen ASM lookup resolve the expected symbol/split; six focused
  Lumen registry/cache tests passed.
- The broad AITER CSV suite reported 23 passing tests plus two failures in
  pre-existing A8W8/A8W8-blockscale-preshuffle model-table duplicate checks.
  Neither failure references or mutates the A4W4 row.  A focused A4W4-only
  validation is still required before the GPU dispatch check.
- Status: open; no end-to-end benefit is claimed until actual Lumen dispatch
  and 8-GPU control/candidate/control timing pass.

### [2026-09-20 clean-fresh-a4w4-dispatch-check]
- Focused AITER validation for the A4W4 table passed 2/2: the tuned CSV has no
  duplicate A4W4 keys, and merging the generic/model A4W4 tables has no shape
  collision.  The earlier broad-suite failures remain isolated to unrelated
  A8W8 tables.
- Fresh single-GPU integration used identical preshuffled operands and the
  actual Lumen `gemm_mxfp4_dispatch`.  A high-priority duplicate-key overlay
  disabled only `(12288,4096,16384)` for the control while leaving every other
  source table unchanged.  Control selected `shuffled`; candidate selected
  `asm` with the exact 128x512 split-0 symbol, manifest hash, and code-object
  hash recorded in the result.
- Both paths were finite and measured 55.6165 dB against the full-output FP32
  reference.  Over 256 steady CUDA-event samples, control mean/median were
  0.607110/0.605766 ms and candidate were 0.368107/0.365265 ms, or
  1.6493x/1.6584x.  The candidate cache profile recorded the existing
  `protected_asm_consensus_required` policy rather than an unverified fallback.
- Artifact:
  `/home/xdai/profile-results/lumen-mxfp4-a4w4-dispatch-fresh-20260920-193157/`.
- Status: dispatch integration passed.  Next gate is a frozen 8-GPU
  control/candidate/control training comparison with fresh caches.

### [2026-09-20 fresh-qwen3-8b-baseline-profile]
- Goal: optimize the Qwen3-8B FSDP2 MXFP4 training path toward 1.6x the matched
  BF16 step speed without a material accuracy regression.
- Frozen comparison: 8 x MI350X gfx950; Qwen3-8B; C4
  `c4_train_1k_repeat4.jsonl`; sequence length 8192; micro batch 2; global batch
  128; gradient accumulation 8; seed 1234; FSDP2 `full_shard`; gradient
  checkpointing on; MXFP4 communication off; retain-accumulated-params off;
  AITER attention, Lumen norm, fused RoPE, and fused cross entropy on. BF16 and
  MXFP4 use the same data, initialization, optimizer schedule, and common
  kernels. The intended precision differences are MXFP4 linears with a five
  layer BF16 tail, plus the current implementation's FP32 gradient-reduction
  dtype versus BF16 reduction in the BF16 reference.
- Fresh correctness evidence: the current MXFP4 op/dispatch suite completed 14
  tests and the fused-cross-entropy reference test completed 1 test. The fresh
  five-step training profile completed 5/5 with no logged NaN, skipped update,
  MXFP4 rejection, or emergency BF16 fallback. Loss was 12.7916 -> 12.7852,
  grad norm stayed finite (5.656e0-5.938e0), and peak memory was 57.5 GiB/GPU.
- Fresh profile evidence: rank 0, training steps 4-5 only, with evaluation
  outside the window. `nccl:_all_gather_base` used 3.245 s CUDA across 1184
  calls, or 1.6225 s and 592 calls per step. `nccl:_reduce_scatter_base` used
  0.246 s across 74 calls, or 37 calls per step. The 592 all-gathers equal 74
  FSDP units x 8 accumulation micro-batches, showing that parameters are
  re-gathered throughout the accumulation window.
- Possible optimization: enable the existing
  `--fsdp-retain-accumulated-params` option as a single-variable A/B. It changes
  parameter residency, not arithmetic, reduction order, or precision. Expected
  effect is fewer all-gathers at the cost of higher peak memory.
- Evidence artifacts:
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/profile-base-mxfp4/`.
  `profile.txt` SHA256 is
  `99363fc9ea5f6178b7585a58b34a9fc2b4954367ec9eda889ba957ed62f3464d`;
  `trace.json` SHA256 is
  `01238981686f71f0111f811cc8e112e1cb7a34842552fea5bd536e9f76e08f7d`.
- Follow-up: the retain, no-checkpoint, reduction-dtype, BF16 reference, and
  rejected shard-grad experiments are recorded below.
- Status: resolved (the selected recipe met the 1.6x deployable-baseline target
  in fresh long-window timing; a same-policy format-only comparison remains
  below 1.6x and is reported explicitly).

### [2026-09-20 retain-accumulated-params-ab]
- Symptom/hypothesis: the baseline re-gathers each FSDP unit for every
  accumulation micro-batch, so retaining full parameters across the
  accumulation window may remove redundant all-gathers without changing the
  training arithmetic.
- Single-variable comparison: same Qwen3-8B recipe, seed, data, precision, and
  gradient-checkpointing settings as the fresh baseline; only
  `--fsdp-retain-accumulated-params` changed from off to on.
- Mechanism evidence from fresh profiles: all-gather calls fell from 592 to 81
  per step (-86.3%).  Two-step NCCL interval union fell from 3493.087 ms to
  1452.110 ms and exposed communication fell from 1395.127 ms to 1008.799 ms.
  Peak allocated memory increased from 57.5 to 70.5 GiB/GPU.  The two profiled
  steps contained 160-341 ms collective/kernel outliers and host stalls, so
  their wall time is not used to decide the speedup.
- Steady-state evidence: fresh unprofiled 40-step A/B runs used steps 11-40 for
  the primary window.  Retain-off mean/median were 8344.1/8369.4 ms and
  retain-on mean/median were 7901.3/7888.5 ms, for 1.056x/1.061x speedup.  On
  the later steps 21-40, mean/median were 8330.7/8379.8 ms off versus
  7860.4/7841.5 ms on, for 1.060x/1.069x.  Nineteen of twenty paired later
  steps were faster and the paired median saving was 380.8 ms.
- Accuracy screen: all losses and gradient norms remained finite with no
  skipped updates.  At step 40, retain-off train/validation loss was
  8.5506/8.5105 and retain-on was 8.5268/8.4277.  These runs predate explicit
  seeding of Python's RNG, which supplies MXFP4 stochastic-rounding seeds, so
  their curve difference is not attributable to retain.  They are only a
  finite/no-obvious-divergence screen; the seeded comparison below is the
  accuracy gate.
- Decision: keep retain-accumulated-params in the candidate optimization stack
  when the additional approximately 13 GiB/GPU fits.  Do not infer its speedup
  from the noisy two-step profiler window; use the unprofiled steady-state A/B.
- Evidence artifacts:
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/profile-retain-mxfp4/`,
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/timing-retain-base/`,
  and
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/timing-retain-on/`.
- Next check: explicitly disable inference KV caching in training forwards,
  validate that change, then profile gradient checkpointing as an isolated
  single variable before testing the retain + no-checkpoint combination.
- Status: resolved (retain accepted as a conditional memory-for-speed
  optimization).

### [2026-09-20 fsdp2-reduce-dtype-control]
- Comparison gap: the current FSDP2 automatic mixed-precision policy reduces
  BF16 gradients in BF16 but reduces FP8/MXFP4 gradients in FP32.  This also
  controls the accumulation dtype while gradient synchronization is disabled,
  so the existing BF16 run is not a same-reduction-policy reference.
- Change: Qwen3 now accepts
  `--fsdp-reduce-dtype {auto,bf16,fp32}` for FSDP2.  `auto` preserves the prior
  behavior exactly; explicit `fp32` enables a BF16 reference with the same
  reduction/accumulation dtype as MXFP4.  Old programmatic callers that omit
  the field also retain `auto` behavior.
- CPU/mock validation: 12 focused tests passed for CLI parsing, launcher
  passthrough, backward-compatible defaults, BF16/MXFP4/FP8 policy selection,
  explicit overrides, and invalid-value rejection.  The broader Qwen3 CPU
  suite plus the FSDP2 policy mock class passed 58 tests with the one CUDA
  cross-entropy test deselected.  No CUDA tensor or kernel was launched.
- GPU control result: the seeded BF16 no-checkpoint + retain recipe with
  explicit FP32 reduction completed 40/40 steps.  Steps 21-40 mean/median were
  8293.87/8170.60 ms and peak memory was 149.0 GiB/GPU.  The otherwise matched
  BF16-reduction run used 8168.25/8114.45 ms and 133.8 GiB/GPU.  Step-40
  train/validation loss was 8.5406/8.4514 with FP32 reduction versus
  8.5408/8.4517 with BF16 reduction, so the control changed memory and modestly
  changed timing without a visible short-run loss penalty.
- Evidence artifact:
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/timing-nogc-retain-bf16-fp32reduce-seeded/`.
  The log SHA256 is
  `840ec054488dde9b7f8016ab923e6a8cbf7d0a694739563c62c3e7040dcd1ebe`.
- Status: resolved (comparison control implemented and measured).

### [2026-09-20 no-gradient-checkpointing]
- Hypothesis: once training forwards explicitly pass `use_cache=False`, removing
  gradient checkpointing should delete forward recomputation without silently
  enabling the inference KV cache.
- Code/correctness guard: pretraining and SFT forwards now pass
  `use_cache=False`.  Model initialization keeps the same seed on every data
  parallel rank, while Python's MXFP4 stochastic-rounding stream is seeded with
  `seed + global_rank`; this makes seeded A/B runs reproducible without sharing
  the same rank-local rounding stream.
- Fresh profile evidence, retain off: wall fell from 8404.6 to 7362.8 ms in the
  two-step trace; non-communication compute fell from 7032.9 to 5851.2 ms.
  MXFP4 GEMMs fell from 6944 to 5208 calls/step, dual quantization from 5208 to
  3472, FMHA forward from 576 to 288, and total GPU kernels from 34974 to
  28382.  All-gather stayed at 592 calls/step, proving the change deleted
  recomputation rather than changing FSDP communication.  Peak memory rose
  from 57.5 to 141.3 GiB/GPU.
- Fresh seeded 40-step evidence, retain off: steps 21-40 mean/median were
  6402.3/6319.3 ms versus the checkpoint baseline's 8330.7/8379.8 ms, for
  1.301x/1.326x.  Step-40 train/validation loss was 8.6393/8.5091; all losses
  and gradient norms were finite, with no fallback, skipped update, OOM, NaN,
  or Inf.
- Evidence artifacts:
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/profile-nogc-mxfp4/`
  and
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/timing-nogc-base-seeded/`.
- Status: resolved (accepted when the activation-memory increase fits).

### [2026-09-20 no-checkpoint-plus-retain]
- Combined recipe: `full_shard`, no gradient checkpointing, retain accumulated
  parameters, MBS2/GBS128/GA8, five BF16 tail layers, MXFP4 communication off,
  and FP32 reduction.
- Fresh seeded 40-step result: steps 11-40 mean/median were
  6132.52/6041.10 ms; steps 21-40 were 6106.13/5994.70 ms.  Relative to the
  no-checkpoint retain-off run, the later window improved by 1.0485x mean and
  1.0541x median, with 19/20 paired steps faster.  Peak memory was
  154.2 GiB/GPU.
- Accuracy screen: step-40 train/validation loss was 8.6037/8.5247 and every
  parsed loss and gradient norm was finite.  Against fresh stock BF16
  (10478.88/10446.10 ms mean/median on steps 21-40), this recipe reached
  1.7161x/1.7426x.  Its validation loss was 0.0731 nats, or 0.865%, above the
  stock BF16 value.  This is a short 40-step gate, not a long-run convergence
  guarantee.
- Same-policy BF16 control: no-checkpoint + retain BF16 used
  8168.25/8114.45 ms on steps 21-40 and 133.8 GiB/GPU.  Thus the same-policy
  MXFP4 advantage with FP32 reduction still differed in reduction dtype and was
  only 1.3377x/1.3536x; this motivated the explicit reduction controls above.
- Evidence artifacts:
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/timing-nogc-retain-seeded/`,
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/timing-base-bf16-current/`,
  and
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/timing-nogc-retain-bf16-seeded/`.
- Status: resolved (accepted as the base optimized recipe before reduction
  dtype tuning).

### [2026-09-20 shard-grad-op-candidate]
- Hypothesis: `shard_grad_op` plus retain could remove the remaining final
  forward/backward parameter regather while preserving arithmetic.
- Fresh profile mechanism: all-gather fell to 37 calls/step, versus 592 for
  no-checkpoint full-shard without retain.  Compute call counts were unchanged:
  5208 MXFP4 GEMMs, 3472 dual quantizations, and 288 FMHA forwards per step.
  Peak memory was 153.2 GiB/GPU.  The mechanism was real, but profiler wall
  time was not used for the decision because of run-to-run FMHA and collective
  variation.
- Fresh seeded 40-step decision: steps 21-40 mean/median were
  6072.12/6012.20 ms, versus full-shard retain's 6106.13/5994.70 ms.  Mean and
  median moved in opposite directions and neither change reached 1%.  The
  candidate validation loss was 8.5956, 0.0709 above the seeded full-shard
  value, and its printed gradient trajectory also diverged materially.
- Evidence artifacts:
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/profile-nogc-retain-shardgrad-mxfp4/`
  and
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/timing-nogc-retain-shardgrad-seeded/`.
  The long-run log SHA256 is
  `fb51053f91dbf72b0c90bd844b54030d53c986336020fea81949d90a0a6cc116`.
- Status: rejected (no stable speed or memory win, with a worse short-run
  numerical screen).

### [2026-09-20 mxfp4-bf16-reduction]
- Hypothesis: use BF16 gradient accumulation/reduction for MXFP4, matching the
  trusted BF16 policy, to halve reduce-scatter bytes and remove FP32-to-BF16
  gradient cast kernels.  This changes numerical policy, so it requires a
  seeded loss gate rather than being accepted from timing alone.
- Fresh profile mechanism: reduce-scatter stayed at 37 calls/step.  On the
  comparable non-outlier step its interval union fell from 121.656 ms with
  FP32 reduction to 61.274 ms with BF16, a 49.63% reduction.  The 37 per-step
  FP32-to-BF16 cast kernels disappeared.  MXFP4 GEMM, dual-quantization, FMHA,
  and residual BF16 GEMM call counts were unchanged.  Peak memory fell from
  154.2 to 139.0 GiB/GPU.
- Fresh seeded 40-step result: steps 11-40 mean/median were
  5970.07/5858.20 ms; steps 21-40 were 5940.40/5848.65 ms.  Against FP32
  reduction this was 1.0279x mean and 1.0250x median on the later window;
  19/20 paired later steps were faster, with a 115.05 ms paired median saving.
- Target result: against fresh stock BF16, steps 21-40 mean/median speedup was
  1.7640x/1.7861x.  Against same-policy BF16 with BF16 reduction it was
  1.3750x/1.3874x, so the 1.6x target is achieved for the deployable optimized
  recipe but not as a pure precision-format-only comparison.
- Accuracy screen: step-40 train/validation loss was 8.5136/8.4138, which was
  0.0270/0.0378 below the fresh stock BF16 values and 0.0901/0.1109 below the
  FP32-reduction MXFP4 run.  All values were finite and no runtime fallback,
  skipped update, OOM, NaN, or Inf occurred.  The favorable 40-step result is
  evidence of no short-horizon regression, not proof of better long-run
  convergence.
- Evidence artifacts:
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/profile-nogc-retain-bf16reduce-mxfp4/`
  and
  `/home/xdai/profile-results/lumen-mxfp4-fresh-20260920/timing-nogc-retain-bf16reduce-seeded/`.
  The long-run log SHA256 is
  `51e8b8b89a2cf1fe335ea44616f2520d123cd031401f6363a3f25c80c59f0da5`.
- Status: resolved (accepted as the selected optimized recipe).

### [2026-09-20 final-focused-validation]
- Scope: final validation of the fresh MXFP4 optimization changes in
  `examples/qwen3/train_qwen3_fsdp.py`, `lumen/models/fsdp.py`, and their
  focused model tests.
- Command: `PYTHONPATH=/home/xdai/Lumen pytest -q tests/models/test_qwen3_fsdp_pretrain.py tests/models/test_fsdp2.py`.
- Result: 70 passed, 0 failed in 177.46 seconds; process exit code 0.
- Teardown note: two identical `torch.library` weakref cleanup tracebacks
  (`ValueError: too many values to unpack (expected 2)`) appeared after the
  successful pytest summary. They did not change the exit code and are not a
  test or training failure.
- Status: resolved.

### [2026-09-20 strict-paired-accuracy-evidence]
- Comparison gap: the pretraining loader used PyTorch's process-global RNG for
  shuffle, so precision-specific RNG consumption before iterator construction
  could change BF16/MXFP4 sample order.  Existing logs also did not provide a
  direct initialization or first-update batch identity check.
- Change: pretraining shuffle now uses a dedicated CPU `torch.Generator` seeded
  only from the training seed and global rank.  Optional
  `LUMEN_PAIRED_RUN_EVIDENCE=1` logging hashes rank-0 model parameters before
  quantization and rolls each rank's CPU `input_ids`/`labels` across all
  micro-batches of optimizer step 1.  The batch digest is disabled after that
  step; the environment switch defaults off.
- CPU/mock validation: 5 focused pairing tests passed.  The full Qwen3 model
  test file with its CUDA-only fused-cross-entropy case deselected passed 49
  tests.  `git diff --check` and `py_compile` passed.  Successful pytest exits
  still emitted the known `torch.library` weakref cleanup traceback.
- Status: resolved (the next BF16/MXFP4 pair can prove identical initialization
  and per-rank first-update data directly from its fresh logs).

### [2026-09-20 strict-remeasurement-baseline]
- Scope reset: all timing/profile evidence produced before
  `/home/xdai/profile-results/lumen-mxfp4-remeasure-20260920/` is excluded from
  the current optimization decision.  The entries above remain historical
  engineering notes only and are not evidence for this remeasurement.
- Strict paired setup: Qwen3-8B, 8x gfx950, sequence length 8192, MBS 2,
  GBS 128, GA 8, FSDP2 `full_shard`, retained accumulated parameters, no
  gradient checkpointing, BF16 reductions in both arms, identical seed 1234,
  AITER attention, Lumen norm, fused RoPE/cross entropy, and five BF16 tail
  layers plus a BF16 `lm_head` in the MXFP4 arm.
- Pairing proof: BF16 and MXFP4 emitted the same pre-quantization model SHA256
  `b63b3be6fba1242f837ed8e677ddfcb25d3252b7b6e709d3cf5d3d5b9072dd12`,
  and every rank's first-update input/label rolling digest matched its peer.
- Fresh 30-step result, steps 11-30: BF16 mean/median 8162.27/8113.75 ms;
  MXFP4 mean/median 6117.07/6028.50 ms; speedup 1.3343x/1.3459x.  MXFP4 needs
  about 5101/5071 ms to reach 1.6x, leaving a 1016/957 ms mean/median gap.
- Accuracy screen: step-30 validation NLL was 9.1534 BF16 versus 9.1540
  MXFP4 (delta +0.0006 nats, approximately +0.060% perplexity).  This passes
  the short-horizon gate but is not a long-run convergence claim.
- Peak allocated memory was 133.8 GiB BF16 versus 138.9 GiB MXFP4.
- Evidence artifacts: `baseline_ab_bf16/` and `baseline_ab_mxfp4/` below the
  remeasurement root.
- Status: open (accuracy is acceptable, but the strict same-policy speed target
  is not met).

### [2026-09-20 strict-remeasurement-profile]
- Fresh rank-0 profiler window: steps 7-8, outside the accepted timing window.
  Both arms used the strict setup above; profiler step times are not used for
  the throughput claim.
- Raw GPU kernel sums from Chrome trace were 17675.610 ms BF16 and
  12209.599 ms MXFP4 over two steps (1.448x).  Unioning kernel/memcpy intervals
  across GPU streams gave 17089.589/11825.457 ms busy time and
  300.064/827.055 ms gaps inside the first-to-last-GPU-event envelope.  Thus
  MXFP4 adds about 263.5 ms/step of visible GPU-idle/host-launch gap.
- BF16 kernel decomposition over two steps: BF16 GEMM 10364.137 ms,
  attention 3477.071 ms, collectives 1592.487 ms.  MXFP4 decomposition:
  MXFP4 GEMM 2681.319 ms, residual BF16 GEMM 2238.828 ms, attention
  3393.606 ms, collectives 941.142 ms, and MXFP4 quant/layout kernels
  705.770 ms.  Categories are disjoint raw `cat=kernel` name classifications;
  profiler operator rows are not added again.
- MXFP4 CPU evidence: `QuantizedLinearFunction` forward/backward self CPU is
  about 2.960 s over two steps, and 109602 `hipPointerGetAttribute` calls are
  visible.  The current ASM `runtime_snapshot()` costs about 30.2-30.5 us per
  hot call in a fresh matching-environment microbenchmark; at 3224 MXFP4 GEMMs
  per step this exposes roughly 97 ms/step before duplicate weight-layout
  checks.
- Artifact SHA256: BF16 trace
  `55b75b97bb174b6f6e1eea09e61248820d255be664711d60b5ebe5c5038dec61`;
  MXFP4 trace
  `9dad3b6de4633c674b33b00b0382d6996da680de08d58f063d001d9c280d66e5`.
  Full profile/table/log hashes are stored beside each artifact.
- Evidence artifacts: `profile_base_bf16/` and `profile_base_mxfp4/` below the
  remeasurement root.
- Conclusion: GEMM replacement is effective, but residual BF16 GEMMs,
  quant/layout work, and host/dispatch gaps cap end-to-end speedup.  First
  optimization candidate is opt-in freezing of ASM registry metadata with
  explicit cache invalidation; it is low risk but cannot alone close the
  approximately one-second target gap.
- Status: open (profile-driven optimization in progress).

### [2026-09-20 asm-registry-freeze]
- Hypothesis: an immutable training deployment does not need to `stat` every
  configured tuned table, manifest, artifact directory, and selected code
  object before every MXFP4 GEMM.  A process-start opt-in can cache snapshots
  until the existing explicit `mxfp4_asm.clear_caches()` invalidation path.
- Change: `LUMEN_MXFP4_ASM_FREEZE_REGISTRY=1` enables a bounded snapshot cache
  keyed by registry epoch, device identity, and GEMM shape.  Default live-poll
  semantics remain unchanged.  Device-discovery failures are not cached, and
  explicit clear/reconfigure invalidates frozen snapshots.  The fused B-layout
  trust check now consumes one coherent snapshot rather than independent ASM
  config and identity reads.
- Focused validation: 44 MXFP4 ASM tests passed; `py_compile` and
  `git diff --check` passed.  The known successful-exit `torch.library`
  weakref traceback remains.
- CPU mechanism: matching-environment hot `runtime_snapshot()` latency fell
  from 30.3 us median to 0.70 us median (about 43x).  This microbenchmark is
  mechanism evidence only, not a step-time claim.
- Fresh ON->OFF->ON 30-step sandwich, steps 11-30: ON means were
  6159.525 and 6158.535 ms; OFF mean was 6327.255 ms.  The average OFF-minus-ON
  saving was 168.225 ms/step (2.66%).  Combined ON median was 6070.85 ms versus
  6240.15 ms OFF.  Per-step paired differences were noisy because NUMA
  balancing remained enabled, but both bracketing ON runs reproduced to within
  1 ms in their means.
- All three arms had the same model-init SHA and per-rank first-step data
  hashes.  The switch changes metadata lookup only, not arithmetic.  The two
  ON validation NLL values were 9.1121 and 9.1130; OFF was 9.1636, illustrating
  that the current two-batch stochastic-MXFP4 validation has much larger
  run-to-run variance than this non-numerical change.
- Evidence artifacts: `opt1_freeze_on_ba/`, `opt1_freeze_off_ba/`, and
  `opt1_freeze_on_bab2/` below the remeasurement root.  Log SHA256 values are
  respectively `9dd66bd2a7cada08381acdd54836e6246762b2ae17701d9c8f667fce6e4deff1`,
  `74122044976ef577e6e66e86a15c189913e86d4339e0b877dbf0ea57a1814adc`,
  and `b49d5a7a6c8998f377cb6b6b34d17de32a070b7317637e2fb44c3deb51f6ba94`.
- Status: accepted as an opt-in static-deployment optimization; insufficient by
  itself to close the 1.6x target gap.

### [2026-09-20 asm-registry-freeze-result-invalidated]
- Pre-GPU source/hash audit found that all three `opt1_freeze_*` run metadata
  files recorded the same `lumen/ops/quantize/mxfp4_asm.py` SHA256
  `113dbd659a124c1abf57012ba5a1468e788538310bee7f0534b31b86d1f84313` as
  the current file, and that source contains no read of
  `LUMEN_MXFP4_ASM_FREEZE_REGISTRY`. A whole-tree search found the variable
  only in the later weight-cache candidate and its tests.
- The launch commands did pass `FREEZE_REGISTRY=1/0/1`; the failure was not a
  missing environment setting. A temporary freeze implementation used by an
  earlier CPU microbenchmark was overwritten at 05:16:33, before the first GPU
  arm started at 05:19:42. The GPU run hashes and preserved bytecode both prove
  that the executed source did not consume the switch.
- Therefore the labelled ON/OFF/ON runs did not switch registry snapshot
  behavior. Their 168.225 ms/step spread is system/order/NUMA variation, not an
  optimization effect, and must not be cited as accepted performance evidence.
- The earlier CPU microbenchmark described a short-lived implementation rather
  than the code captured by the GPU run hashes. A real default-live/opt-in-frozen
  implementation, focused invalidation tests, and a new single-variable GPU
  sandwich are required before this candidate can be accepted.
- Status: rejected as invalid experimental attribution; implementation and
  remeasurement reopened.

### [2026-09-20 asm-registry-freeze-reimplemented]
- A real opt-in implementation now consumes
  `LUMEN_MXFP4_ASM_FREEZE_REGISTRY=1`. It caches one coherent
  `RuntimeSnapshot` by registry epoch, CUDA device index, arch/CU identity, and
  GEMM shape; default live polling is unchanged.
- `clear_caches()` and the existing configure path advance the epoch and clear
  frozen snapshots. Transient device/current-device failures and explicitly
  uncacheable snapshots are retried rather than cached. Concurrent first misses
  serialize publication.
- Independent review found a manifest-generation coherence bug in the first
  draft. The final version uses one captured manifest for table admission,
  selected artifact identity, and fingerprinting, and rechecks split-K support
  against that captured artifact. Loaded code-object baselines remain
  fail-closed across manifest-root changes.
- Validation rerun by the root agent: 43 focused registry/dispatch CPU tests
  and 17 weight-cache tests passed; `py_compile` and `git diff --check` passed.
- Fresh ABBA metadata microbenchmark on the final source, 5,000 hot calls per
  arm at shape 16384x4096x4096: OFF medians 32.72/32.77 us, ON medians
  1.83/1.80 us, or 18.04x. Both modes resolved the same gfx950 ASM symbol and
  split policy. This proves the switch is active but is not a step-time claim.
- Status: implementation validated; fresh 8-GPU single-variable sandwich is
  still required before accepting any training speedup.

### [2026-09-20 opt12-pilot-and-measurement-lock-hardening]
- The first post-implementation 30-step control run completed as
  `opt1_real_freeze_off_a` with freeze=0 and fast-hit=0. Steps 11-30 were
  6210.83 ms mean, 6122.25 ms median, 5856.73 ms P10, and 6753.55 ms P90.
  Peak memory was 138.9 GiB/GPU; final train/validation NLL was 9.1088/9.1111;
  all 30 steps and all eight first-update batch hashes were present and finite.
  There was no fallback, OOM, NaN/Inf, skipped update, or runtime error. The
  known post-success torch.library weakref teardown traceback remained.
- This run is retained only as a pilot, not as an arm in the final attribution
  matrix. Audit found that the preceding `opt12_real_prewarm` was actually
  freeze=0/fast-hit=0, so it did not exercise the two new optimized paths.
  Audit also found that the runner's bundle omitted dispatch.py, the runner
  itself, and loaded AITER artifacts.
- The runner was hardened before the accepted matrix: unique case directories,
  explicit source/cache before files, expected-cache fail-fast, dispatch.py,
  runner, AITER JIT modules and the selected gfx950 FP4 code objects in the
  before/after bundle, full Lumen/AITER tree-state before/after checks, and an
  explicit successful train exit-status artifact. The analyzer now requires
  and reports the full ordered step sequence, finite values, model/batch hashes,
  completion, fallback/OOM/NaN/Inf/runtime-error indicators, and the log hash.
- Next: run a real freeze=1/fast-hit=1 three-step smoke under the hardened lock,
  then restart a fresh A/B/C/B/A 30-step matrix. The pilot above must not be
  combined with that matrix.
- Status: measurement gap found and contained; accepted GPU attribution remains
  open.

### [2026-09-20 opt12-locked-gpu-smoke]
- A new three-step smoke explicitly ran both optimized paths:
  `LUMEN_MXFP4_ASM_FREEZE_REGISTRY=1` and
  `LUMEN_MXFP4_WEIGHT_CACHE_FAST_HIT=1`.
- The strengthened source/AITER bundle was identical before and after at
  `619691975670eed723e0f683f8e7b9227b9fee678be1c070ac7f9e491e751a53`.
  The autotune cache was also identical before and after at
  `7091528fe1de631e1d9dafc5d2621157fa2d55200bbd70ed3e64327253cd9a4e`;
  the log loaded nine cached decisions and generated no new tuning decision.
  Lumen and AITER diff/status hashes were unchanged during the run.
- All 3/3 steps completed. Post-first-step times were 6345.1 and 6500.3 ms;
  these two values are smoke evidence only. Final train/validation NLL was
  12.8035/12.7884, gradients were finite, peak memory was 138.9 GiB/GPU, and
  there was no fallback, OOM, NaN/Inf, skipped update, or runtime error.
- Artifact:
  `/home/xdai/profile-results/lumen-mxfp4-remeasure-20260920/opt12_locked_smoke_on/`.
- Status: freeze+fast-hit GPU smoke passed; begin a new locked A/B/C/B/A timing
  matrix, excluding the earlier pilot from attribution.

### [2026-09-20 opt12-locked-matrix-a1]
- Formal locked A1 control: freeze=0, fast-hit=0, tail5, no profiler. Steps
  11-30 were 6105.265 ms mean, 5960.2 ms median, 5859.46 ms P10, and
  6521.62 ms P90. Full sequence:
  `5994.8,7045.3,5934.9,5985.5,5905.1,6072.6,6554.2,6054.0,5920.0,6402.7,6518.0,5866.2,5880.1,5860.0,5868.1,5854.6,5865.6,5854.5,6336.0,6333.1` ms.
- Peak memory was 138.9 GiB/GPU; final train/validation NLL was 9.1144/9.1624;
  all values were finite. All 30 ordered steps, the expected model hash, and
  all eight batch hashes were present. No fallback, OOM, NaN/Inf, skipped
  update, or runtime error was found.
- Source/AITER bundle, repository state, and autotune cache were unchanged
  before/after. Artifact: `opt12_locked_a1_off/` under the remeasurement root.
- Status: A1 accepted; no optimization conclusion until B1/C/B2/A2 complete.

### [2026-09-20 opt12-locked-matrix-b1]
- Formal locked B1: freeze=1, fast-hit=0. Steps 11-30 were 6264.67 ms
  mean, 6125.3 ms median, 5838.26 ms P10, and 6837.96 ms P90. Full sequence:
  `5876.5,6118.2,6423.1,6570.1,6017.5,6821.7,6132.4,6984.3,6369.0,7116.4,6271.0,6534.2,6604.6,6058.3,5852.9,5780.7,5840.0,5822.6,6057.4,6042.5` ms.
- Peak memory was 138.9 GiB/GPU; final train/validation NLL was 9.1364/9.1228.
  All integrity, finite, completion, model/batch-hash, and cache gates passed.
- B1 alone was slower than A1 by 159.405 ms mean and 165.1 ms median, opposite
  the expected micro-mechanism. This is not a rejection because NUMA balancing
  is enabled and the distributions contain large, differently positioned tail
  events. Await B2/A2 symmetric attribution.
- Status: B1 accepted as a measurement arm; freeze decision remains open.

### [2026-09-20 opt12-locked-matrix-c]
- Formal locked C: freeze=1, fast-hit=1. Steps 11-30 were 5934.065 ms mean,
  5824.35 ms median, 5774.53 ms P10, and 6246.05 ms P90. Full sequence:
  `5832.3,6422.9,6037.9,5953.7,5796.5,5792.9,6039.6,5923.9,5788.7,5786.6,6315.8,5781.1,5776.0,5888.3,6185.8,5816.4,5775.3,5767.6,5761.7,6238.3` ms.
- Peak memory was 138.9 GiB/GPU; final train/validation NLL was 9.1219/9.1178.
  All integrity, finite, completion, model/batch-hash, and cache gates passed.
- C was 330.605 ms mean and 300.95 ms median faster than B1, but B2 is still
  required to separate fast-hit from time/order/NUMA drift.
- Status: C accepted as a measurement arm; fast-hit decision remains open.

### [2026-09-20 opt12-locked-matrix-b2]
- Formal locked B2: freeze=1, fast-hit=0. Steps 11-30 were 6203.62 ms mean,
  6242.7 ms median, 5816.47 ms P10, and 6554.75 ms P90. Full sequence:
  `6026.1,6273.6,6496.6,5892.1,5865.9,5810.8,5805.9,6275.6,6223.2,7007.4,6262.2,6551.5,6345.9,5817.1,6193.6,6584.0,5978.1,6055.0,6285.4,6322.4` ms.
- Peak memory was 138.9 GiB/GPU; final train/validation NLL was 9.1188/9.1115.
  All integrity, finite, completion, model/batch-hash, and cache gates passed.
- Averaging B1/B2 gives 6234.145 ms mean and 6184.0 ms median. C is therefore
  300.08 ms (4.81%) faster by mean and 359.65 ms (5.82%) faster by median.
  This is consistent across both B brackets and is provisionally attributed to
  fast-hit; finish A2 before final matrix acceptance.
- Status: B2 accepted; fast-hit provisional positive, freeze still open.

### [2026-09-20 opt12-locked-matrix-a2-and-decision]
- Formal locked A2: freeze=0, fast-hit=0. Steps 11-30 were 6070.96 ms mean,
  5874.0 ms median, 5861.14 ms P10, and 6453.96 ms P90. Full sequence:
  `6199.3,7007.1,6110.7,5957.7,6174.0,5871.6,5876.4,5868.2,5866.6,6443.8,6545.4,5862.8,5869.0,5862.8,5860.1,5867.2,5861.2,5860.6,6275.0,6179.7` ms.
- Peak memory was 138.9 GiB/GPU; final train/validation NLL was 9.1144/9.1108.
  All integrity, finite, completion, model/batch-hash, and cache gates passed.
- Symmetric attribution using the arm means: A1/A2 average 6088.1125 ms;
  B1/B2 average 6234.145 ms. Registry freeze alone was 146.0325 ms slower
  (speed ratio 0.9766x) and won only 7/20 same-step bracket comparisons, so it
  is rejected as a standalone speed optimization under this NUMA-enabled run.
- C (freeze+fast-hit) was 5934.065 ms mean and 5824.35 ms median. Against the
  B bracket it saved 300.08 ms mean (1.0506x) and won 15/20 step comparisons;
  against the A bracket it saved 154.0475 ms mean (1.0260x) and won 17/20.
  Because the current correctness contract enables fast-hit only together with
  frozen registry metadata, keep the pair as one opt-in stack, not freeze alone.
- Validation NLLs were A1 9.1624, B1 9.1228, C 9.1178, B2 9.1115, A2 9.1108.
  All runs used identical initialization and per-rank first-update batch hashes;
  the spread is treated as short-run stochastic-MXFP4 variance, with no sign of
  material regression from the non-arithmetic cache changes.
- Status: opt12 stack provisionally accepted for a 2.60% mean step-speed gain;
  standalone registry freeze rejected. Fresh profile and larger candidates are
  still required for the 1.6x goal.

### [2026-09-20 opt12-locked-fresh-profile]
- Scope: this entry uses only the post-rollback, post-`/dev/kfd`-authorization
  profiles `profile_opt12_locked_mxfp4/` and `profile_opt12_locked_bf16/`.
  No earlier profile artifact is used in this attribution.
- Integrity passed for both arms. The source/AITER bundle remained
  `619691975670eed723e0f683f8e7b9227b9fee678be1c070ac7f9e491e751a53`,
  the autotune cache remained
  `7091528fe1de631e1d9dafc5d2621157fa2d55200bbd70ed3e64327253cd9a4e`,
  Lumen/AITER tree-state hashes were unchanged, and both recorded exit status
  zero. Model-init SHA and all eight rank-local first-update batch hashes match.
- The rank-0 profiler covers optimizer steps 7-8 (GA8) on device 0. Profiler
  wall time is attribution evidence only, not the final throughput metric.
  MXFP4 step times were 5878.8/6199.9 ms and BF16 step times were
  8326.2/8274.2 ms. Both runs completed 8/8 steps with finite losses and
  gradients, no fallback/OOM/NaN/Inf/skipped update/runtime failure, and only
  the known post-success `torch.library` weakref teardown traceback.
- GPU timeline, reported as two-step total / per-step: BF16 kernel raw sum
  16872.436/8436.218 ms, kernel+memcpy interval union 16307.721/8153.861 ms,
  first-to-last GPU envelope 16596.813/8298.406 ms, and idle gaps
  289.091/144.546 ms. MXFP4 raw sum was 12168.915/6084.457 ms, union
  11693.833/5846.916 ms, envelope 12069.132/6034.566 ms, and idle gaps
  375.299/187.649 ms. MXFP4 therefore saved 2306.944 ms/step of GPU union
  busy time but exposed 43.103 ms/step more gap. Raw-kernel speed ratio was
  1.3865x and union-busy ratio was 1.3946x.
- Disjoint kernel classification, two-step total / per-step: MXFP4 A4W4 GEMM
  2947.342/1473.671 ms, residual BF16 GEMM 2243.273/1121.636 ms, MXFP4
  quant/layout 770.450/385.225 ms, attention 3407.710/1703.855 ms, and NCCL
  collectives 494.908/247.454 ms. BF16 had BF16 GEMM
  10400.979/5200.490 ms, attention 3462.990/1731.495 ms, and collectives
  726.133/363.067 ms. Collective calls and tensor shapes match, so their
  duration difference is treated as scheduling/contention noise rather than an
  MXFP4 algorithmic gain.
- The linear-chain replacement is effective: BF16 GEMM fell by
  4078.853 ms/step while A4W4 cost 1473.671 ms/step, a net GEMM-family saving
  of 2605.183 ms/step. Quant/layout gives back 385.225 ms/step, leaving a net
  linear-chain reduction of about 2219.958 ms/step, 94.4% of the total raw
  kernel reduction. Attention is nearly unchanged (-27.640 ms/step).
- MXFP4 residual BF16 GEMMs are 864 calls/step. The three vocabulary-sized
  `lm_head` GEMMs are 24 calls and about 469.605 ms/step; the five protected
  transformer tail layers account for the other 840 calls and about
  652.032 ms/step. The profile therefore directly supports testing tail5 to
  tail0 while retaining the BF16 `lm_head`. The 652 ms is only a removable
  BF16-work upper bound because replacement A4W4 and quant kernels add cost.
- MXFP4 quant/layout is dominated by `_dual_layout_quant_mxfp4` at
  372.172 ms/step and 3472 launches/step (96.6% of classified quant/layout).
  MXFP4 also issues 8680 more kernels over two steps than BF16. Kernel-launch
  runtime self time increases by about 238.582 ms/step, but visible GPU idle
  grows only 43.103 ms/step, so kernel work/launch count is the primary issue;
  pure host starvation is secondary.
- Host/runtime evidence per step: MXFP4 `QuantizedLinearFunctionBackward`
  self/total CPU was about 962.191/1769.450 ms, forward 315.247/598.595 ms,
  and `aiter::_gemm_a4w4_asm` 363.178/584.077 ms. MXFP4
  `hipModuleLaunchKernel` was 15364 calls and 911.601 ms self time,
  `hipPointerGetAttribute` 54801 calls and 51.318 ms. BF16 `aten::mm` was
  6072 calls and 180.795/562.054 ms self/total CPU. Synchronize and blocking
  memcpy rows mostly measure waiting for GPU work and are not counted as
  independently removable CPU overhead.
- Reproducible parser:
  `/home/xdai/profile-results/lumen-mxfp4-remeasure-20260920/analyze_opt12_locked_profiles.py`.
  Trace SHA256 values are MXFP4
  `a6f78a8d23ac6845f9b4965b8ac89ac7a82055248e562ca6e1a38ca641243f05`
  and BF16
  `cbb11a3b46a3d4f4d824a8b93b5db2c59483c56bc108e55899b76b0911769e6f`.
- Decision: run a tail0 three-step GPU smoke, then a new unprofiled
  tail5/tail0/tail5 30-step sandwich with stronger matched validation. If the
  arithmetic/accuracy gate passes, keep tail0 provisionally; it is not expected
  by itself to close the full 1.6x gap. The next MXFP4-specific target is the
  dual-layout quantization/descriptor path.
- Status: fresh profile accepted for attribution; tail0 experiment pending.

### [2026-09-20 opt13-tail0-smoke]
- Profile-driven candidate: reduce the BF16 tail from five transformer layers
  to zero while retaining the independently protected BF16 `lm_head`. No code
  change was needed; only `TAIL_BF16=0` changed the arithmetic policy.
- The fresh 8-GPU smoke routed exactly 252 `nn.Linear` modules through MXFP4
  and skipped exactly one BF16 output layer, matching the audited 36-layer
  Qwen3-8B module inventory. It loaded the same nine locked autotune decisions.
- All 3/3 optimizer steps completed. Step times were 32275.1, 6813.0, and
  5542.6 ms; the first includes startup and the other two are smoke evidence
  only. Train loss remained finite (12.7926 to 12.8033), grad norm remained
  finite (5.719e0 to 5.812e0), strengthened eight-batch validation NLL was
  12.7917, and peak allocated memory was 139.7 GiB/GPU.
- Model-init SHA and all eight rank-local first-update batch hashes were
  present. Source bundle, autotune cache, and Lumen/AITER tree-state hashes
  were unchanged before/after; shell exit status was zero. There was no
  fallback, OOM, NaN/Inf, skipped update, or runtime error. The known
  post-success `torch.library` weakref teardown traceback remained.
- Artifact:
  `/home/xdai/profile-results/lumen-mxfp4-remeasure-20260920/opt13_tail0_locked_smoke/`;
  log SHA256
  `45e96d00fa42136bf1236bb0ac298ac61ada2a7e0dc158c1dde0c52f35bf3da0`.
- Decision: smoke passed. Proceed to an unprofiled tail5/tail0/tail5 sandwich,
  using steps 11-30 and matched eight-batch validation for performance and
  short-horizon numerical attribution.
- Status: resolved as a smoke gate; timing/accuracy decision pending.

### [2026-09-20 opt13-tail5-tail0-tail5-sandwich]
- Scope: fresh post-rollback, post-`/dev/kfd`-authorization measurements only.
  The unprofiled 30-step runs used the locked Qwen3-8B 8xMI350X setup
  (sequence length 8192, MBS 2, GBS 128, GA 8), identical source/AITER and
  autotune-cache bundles, and steps 11-30 as the timing window. NUMA balancing
  remained enabled, so the decision uses the symmetric tail5/tail0/tail5
  bracket rather than an isolated run.
- Tail5 A1 measured 5849.695 ms mean and 5783.05 ms median; tail0 B measured
  5643.965 ms mean and 5513.25 ms median; tail5 A2 measured 5922.575 ms mean
  and 5857.55 ms median. Tail0 saved 242.17 ms versus the bracket mean
  (1.04291x) and 307.05 ms versus the bracket median (1.05569x), and won
  17/20 same-numbered step comparisons against the midpoint of the two
  controls. The two tail5 controls drifted 1.238% by mean and 1.280% by
  median, which is acceptable only because the symmetric control was used.
- The three arms completed all 30/30 optimizer steps with finite losses and
  gradients and no fallback, OOM, NaN/Inf, skipped update, or runtime error.
  Model initialization, all eight rank-local first-update batch hashes,
  source/cache bundles, and before/after tree-state hashes matched. Peak
  allocated memory was 138.9 GiB for tail5 and 139.7 GiB for tail0.
- Matched eight-batch validation NLL was 9.1433 for tail5 A1, 9.1966 for
  tail0 B, and 9.1758 for tail5 A2. The tail5 bracket mean was 9.15955 and
  its natural A1/A2 span was 0.03250. Tail0 was +0.03705 NLL versus the
  bracket mean (about +3.774% perplexity), slightly outside the predeclared
  short-run natural-span gate.
- Decision: the performance gate passed, but the short-horizon accuracy gate
  failed. Do not accept tail0 and do not relax the gate after seeing the
  result. Tail0 may only be reconsidered through a separately designed
  >=200-step strictly paired or multi-seed accuracy experiment. Tail1/tail2
  remain possible independent candidates but require new symmetric empirical
  measurements; no linear extrapolation from tail0 is allowed.
- Reproducible parser:
  `/home/xdai/profile-results/lumen-mxfp4-remeasure-20260920/analyze_tail_sandwich.py`.
- Status: tail0 rejected from the optimization stack; continue with the
  dual-layout quantization and repeated host/descriptor paths.

### [2026-09-20 opt14-qwen-a4w4-asm-shape-tuning]
- Scope: only fresh post-rollback, post-`/dev/kfd`-authorization artifacts
  below the remeasurement root are used. Four exact Qwen3-8B A4W4 shapes were
  added to the per-model tuned table so they dispatch to already-installed
  gfx950 ASM code objects instead of the shuffled Triton fallback:
  `(1024,4096,16384)` uses the 128x128 kernel and
  `(12288,4096,16384)`, `(16384,1024,4096)`, and
  `(16384,4096,1024)` use the 128x512 kernel, all with split-K zero.
- Correctness: all four candidate outputs were bitwise equal to the prior
  shuffled-Triton outputs. The independently generated control and candidate
  caches resolved those four shapes to `shuffled` and `asm`, respectively,
  with matching manifest, code-object, symbol, split policy, and recorded
  artifact hashes.
- Formal unprofiled A/B/A result, steps 11-30: the control bracket was
  5910.265 ms mean and 5809.425 ms median; the ASM candidate was 5754.675 ms
  mean and 5656.500 ms median. This is a 1.027037x improvement by both mean
  and median, saving 155.590 ms/step, with 18/20 same-step midpoint wins.
  Control drift was 0.320% by mean and 0.607% by median.
- All arms completed 30/30 steps with finite values and no fallback, OOM,
  NaN/Inf, skipped update, or runtime failure. Source/cache/tree/model/batch
  locks passed. Candidate validation NLL was 9.1393 versus a 9.1916 control
  bracket; this is treated only as no observed short-run regression, not as a
  quality improvement claim.
- Artifacts: `opt14_a4w4_control_a1/`, `opt14_a4w4_candidate_b/`,
  `opt14_a4w4_control_a2/`, and `opt14_a4w4_sandwich_analysis.json` below the
  remeasurement root.
- Status: accepted into the optimization stack.

### [2026-09-20 opt15-forward-activation-descriptor-cache]
- Hypothesis: q/k/v projections and gate/up projections consume the same
  forward activation, so an exact-match, forward-RTN-only descriptor cache can
  remove duplicate dual-layout MXFP4 quantization without changing backward
  stochastic rounding. The implementation is explicit opt-in through
  `LUMEN_MXFP4_ACTIVATION_DESCRIPTOR_CACHE=1`; default behavior remains off.
- Focused validation: 23 lifecycle tests passed. Independent review found no
  high- or medium-risk issue for the target single-thread eager training path.
  Every autograd consumer receives a fresh `FP8Descriptor`; only immutable
  quantized tensors are shared. Input/sign identity and mutation version,
  layouts, dtype/device, stream, and quantization options are all keyed.
  Backward SR is never cached. Two low-risk limitations remain: `.data` or raw
  kernel mutation can bypass Tensor versioning, and cross-thread destruction
  can retain one dead cache entry until the owner thread calls again or exits.
- Fresh paired profile, steps 7-8: target dual-layout launches fell exactly
  from 3472 to 2728 per step (-744, -21.43%). Target raw GPU time fell from
  353.937 to 297.374 ms/step (-56.563 ms, -15.98%), total GPU raw from
  6055.066 to 5771.871 ms/step (-283.195 ms, -4.68%), and GPU union busy from
  5802.034 to 5559.768 ms/step (-242.266 ms, -4.18%). Peak allocated memory
  fell from 138.9 to 132.7 GiB/GPU. Profile/source/cache/tree integrity passed.
- Formal unprofiled off/on/off result, steps 11-30: control bracket was
  5809.510 ms mean and 5715.250 ms median; cache-on was 5843.795 ms mean and
  5736.350 ms median. Cache-on was therefore slower by 34.285 ms mean and
  21.100 ms median (0.9941x/0.9963x) and won only 9/20 same-step midpoint
  comparisons. It fails the predeclared >=1% mean/median and >=14/20 win gate.
  The A1/A2 mean drift was 0.548%; median drift was 2.049%, reinforcing that
  profiler savings must not be substituted for the failed wall-time result.
- Validation NLL was 9.1413 cache-on versus a 9.1304 control midpoint, delta
  +0.0109 nats (about +1.096% perplexity), within the controls' 0.0176 NLL
  natural span. All arms completed 30/30 with finite values and no fallback,
  OOM, NaN/Inf, skipped update, or runtime failure; all pairing/provenance
  checks passed.
- Artifacts: `profile_opt15_cache_off/`, `profile_opt15_cache_on/`,
  `opt15_descriptor_cache_profile_analysis.json`, `opt15_cache_off_a1/`,
  `opt15_cache_on_b/`, `opt15_cache_off_a2/`, and
  `opt15_descriptor_cache_sandwich_analysis.json` below the remeasurement root.
- Status: rejected as a step-speed optimization. Keep default off; it may be
  useful only as an explicit memory-saving experiment and is not credited
  toward the 1.6x target.

### [2026-09-20 kfd-fresh-host-fastpath-preparation]
- Scope: preparation only under
  `/home/xdai/profile-results/lumen-mxfp4-kfd-rerun-20260920-094252/`;
  no GPU timing or profiling result was produced or reused by this step.
- Source audit: registry freeze is exact-value opt-in and caches coherent ASM
  snapshots by registry epoch, CUDA device, arch/CU identity, and GEMM shape.
  Explicit `mxfp4_asm.clear_caches()` advances the epoch; transient discovery
  failures are uncacheable; a changed already-loaded code object remains
  fail-closed until process restart. Weight-cache fast-hit is active only when
  both freeze and fast-hit equal `1`, and its key covers weight version/object
  identity, GEMM rows, block size, autotune decision epoch, and registry epoch.
  Optimizer hooks clear module and Parameter caches after each step.
- Prepared a fail-closed smoke + A1/B1/C/B2/A2 runner, integrity analyzer, and
  protocol document in the fresh root. The analyzer uses steps 11-30, paired
  circular moving-block bootstrap, source/cache/tree/model/batch locks, and
  predeclared mean/median/win/drift/numerical gates. The three-step C smoke is
  required by default before the five 30-step arms.
- Preparation-time locks for the current source stack are source bundle
  `b9fb84b2f8474203d3707e60068b5dc021331ee53fcf4a7c1f9d892dc27d13a8`,
  cache namespace `shape128x512_candidate`, and cache SHA256
  `898df1fd413733af31186ab3204b352c1a5d57b8e17dcd83edb7ed69ed9d8067`.
  A dry-run provenance check passed without starting a GPU process.
- CPU validation on current source: weight-cache lifecycle/fast-hit suite
  `17 passed`; ASM registry/table/artifact/freeze suite `40 passed`; FSDP
  optimizer-hook wiring `4 passed`. All pytest commands exited zero. The known
  post-success `torch.library._del_library` cleanup traceback remained. An
  earlier attempt to hide every GPU failed during test collection because
  Triton found zero active drivers and then attempted an unavailable JAX
  fallback; that was an environment-selection failure, not a code-test failure.
- Status: runner and analysis protocol ready; no host-fastpath performance
  conclusion exists until the fresh GPU smoke and full matrix are executed.

### [2026-09-20 kfd-fresh-shape128x512-step-aba]
- Scope: only artifacts under
  `/home/xdai/profile-results/lumen-mxfp4-kfd-rerun-20260920-094252/` were
  used. The tested change added one exact Qwen3 table entry for
  `(M,N,K)=(12288,4096,16384)`, selecting the installed gfx950
  `BpreShuffle_128x512` ASM symbol with split-K zero instead of the current
  shuffled-Triton path.
- Mechanism evidence passed: a fresh trace replaced all 496 target Triton
  launches with the selected ASM symbol. Target raw GPU time fell from
  323.365 to 182.999 ms over the profiled window, but the trace wall improvement
  was only 15.369 ms and was not used as the acceptance decision.
- Formal unprofiled A/B/A used 30 optimizer steps per arm and steps 11-30 for
  timing. A1 control was 6009.485/5892.250 ms mean/median, candidate B was
  6042.135/5883.400 ms, and A2 control was 6007.510/5889.200 ms. The symmetric
  control was 6008.4975 ms mean and 5890.725 ms run-median midpoint. Candidate
  mean was 33.6375 ms slower (`0.994433x`); median was only 7.325 ms faster
  (`1.001245x`). It won 13/20 paired steps.
- The paired IID bootstrap 95% speedup CI was `[0.977681, 1.009322]` and the
  saving CI was `[-136.935, 55.548]` ms. Independent circular moving-block
  bootstraps with block lengths 2, 3, 4, 5, 6, and 10 also crossed no-effect.
  A1/A2 drift was only 0.0329% mean and 0.0518% median, so the rejection is not
  explained by control drift.
- Integrity passed: all arms completed 30/30 with exit zero, finite losses and
  gradients, identical model-init and all eight first-update batch hashes, and
  unchanged source/cache/tree locks. Candidate validation NLL was 9.1238 versus
  a 9.13705 control bracket, a same-precision short-run screen only. Strict
  zero-traceback failed solely due to the known post-success `torch.library`
  cleanup defect.
- Decision: reject and roll back the exact-shape row. The production Qwen3
  tuned table is byte-identical to the saved control table after rollback.
  Microbenchmark and trace-local kernel wins are insufficient without a stable
  end-to-end step win.
- Artifact:
  `/home/xdai/profile-results/lumen-mxfp4-kfd-rerun-20260920-094252/shape128x512_aba_result.json`.
- Status: resolved; candidate rejected and production row removed.

### [2026-09-20 kfd-fresh-existing-asm-topkeys-retune]
- Scope: only post-rollback measurements under
  `/home/xdai/profile-results/lumen-mxfp4-kfd-rerun-20260920-094252/`
  were used. Existing installed gfx950 ASM symbols were exhaustively checked
  for the two largest current ASM keys using production packing/marshalling;
  no new kernel or tuning-table row was introduced.
- For `(M,N,K)=(16384,12288,4096)`, 34 split-K-zero symbols were bitwise
  correct. All tested split-K 1/2/3 cases failed the low-risk bitwise gate.
  The incumbent `BpreShuffle_128x512`, split-K zero, measured 451.975 us;
  the closest challenger `BpreShuffle_256x256`, split-K zero, measured
  454.801 us. Challenger/incumbent speed ratio was 0.992926x with 95% CI
  `[0.991805, 0.994042]`, so the incumbent remains selected.
- For `(16384,4096,12288)`, `BpreShuffle_128x512`, split-K zero, beat the
  incumbent `BpreShuffle_256x256`, split-K zero, in two independent processes:
  389.835 -> 388.343 us (1.003840x, CI `[1.002879, 1.005003]`) and
  385.584 -> 384.925 us (1.001713x, CI `[1.000779, 1.002635]`). The estimated
  end-to-end contribution is only about 0.8 ms/step (roughly 0.013%), below the
  threshold for an eight-GPU A/B/A campaign.
- Decision: preserve the production tables unchanged. The first challenger is
  slower; the second is a reproducible micro-winner but too small to matter for
  the 1.6x target and is not credited as a training-step optimization.
- Artifact:
  `/home/xdai/profile-results/lumen-mxfp4-kfd-rerun-20260920-094252/asm_topkeys_retune_results.md`.
- Status: resolved; no production configuration change.

### [2026-09-20 kfd-fresh-host-fastpath-matrix]
- Scope: fresh eight-GPU smoke plus unprofiled A1/B1/C/B2/A2 matrix under
  `/home/xdai/profile-results/lumen-mxfp4-kfd-rerun-20260920-094252/`.
  The five 30-step arms used steps 11-30, the locked `baseline_fresh` cache,
  identical source/tuned-table/tree/model/batch provenance, and the order
  live-control -> freeze-only -> freeze+fast-hit -> freeze-only -> live-control.
- Arm mean/median step times were: A1 live 5993.925/5866.350 ms; B1 freeze
  6170.645/6113.750 ms; C freeze+fast-hit 6123.250/5997.500 ms; B2 freeze
  6202.765/6058.200 ms; A2 live 6033.225/5963.300 ms. A and B bracket drift
  stayed below the predeclared 3% limit.
- Registry freeze alone was slower: 0.972016x mean and 0.971878x run-level
  median, 7/20 wins, moving-block-bootstrap 95% speedup CI
  `[0.947636, 0.997555]`. It failed every performance gate.
- Fast-hit relative to the freeze bracket reached 1.010363x mean and 1.014752x
  median, but only 11/20 wins and CI `[0.974143, 1.043076]`; the apparent gain
  is not statistically stable under the preregistered gate.
- The deployable freeze+fast-hit pair versus the live-control bracket was
  slower: 0.982089x mean and 0.986215x median, 10/20 wins, CI
  `[0.961236, 1.004948]`. It therefore cannot be retained even though the
  incremental fast-hit point estimate narrowly exceeded 1% versus freeze.
- All arms completed 30/30 with finite losses/gradients, exit zero, and peak
  memory 138.9 GiB/GPU. The numerical screen passed: C validation NLL 9.1439
  versus live-control bracket 9.1393, delta +0.0046 nats, below the +0.03 gate.
  The only traceback was the known post-success `torch.library._del_library`
  cleanup defect, so the strict zero-traceback gate remains separately failed.
- Decision: reject registry freeze both alone and paired with weight-cache
  fast-hit; keep both flags default-off and do not credit them toward the 1.6x
  target. Because the end-to-end gate failed, no follow-up A/C profiler trace
  is warranted.
- Artifacts: `host_fastpath_fresh_analysis.json` and
  `host_fastpath_fresh_analysis.md` below the fresh root.
- Status: resolved; candidate rejected on wall time, accuracy screen passed.

### [2026-09-20 kfd-fresh-mbs4-ga4-smoke]
- Scope: only new measurements under
  `/home/xdai/profile-results/lumen-mxfp4-kfd-rerun-20260920-094252/` were
  used. The matched feasibility smoke changed batch geometry to MBS4/GA4 while
  keeping GBS128, sequence length 8192, FSDP2 full-shard, BF16 reduction,
  retained accumulated parameters, no gradient checkpointing, five BF16 tail
  layers, initialization, data order, source bundle, and precision-independent
  settings fixed.
- Both arms completed 3/3 optimizer steps with exit zero, finite loss and
  gradient norm, matching model initialization SHA256, and matching step-1
  batch digests on all eight ranks (`microbatches=4`). Source/tree locks held;
  the MXFP4-generated cache was frozen at SHA256
  `924fb81af7502727c42914530583f8e8a8f940eed34800dff4d3475950ae603d`
  for the BF16 arm.
- Excluding startup/autotune step 1, MXFP4 steps 2/3 were 6824.7/5979.4 ms
  (mean 6402.05 ms) and BF16 steps 2/3 were 8388.3/8515.6 ms (mean
  8451.95 ms), a smoke-only ratio of 1.3202x. The corresponding 1.6x target is
  5282.47 ms, leaving a 1119.58 ms smoke gap. These two samples are a screen,
  not an accepted throughput estimate.
- Peak allocated memory was 231.9 GiB/GPU MXFP4 and 227.7 GiB/GPU BF16.
  Board monitoring observed approximately 244.8 GiB used in both arms, leaving
  only about 6.9 GiB minimum headroom. Validation NLL was 12.7893 MXFP4 versus
  12.8007 BF16; this is only a finite/no-obvious-regression smoke check.
- The sole traceback started after `Training complete` and was the known
  `torch.library._del_library` cleanup defect. No OOM, fallback, NaN/Inf,
  skipped update, or training-time runtime failure was found.
- Decision: correctness smoke passed, but screen out MBS4/GA4 as the next
  performance direction. Its provisional ratio is far from 1.6x and the board
  memory margin is too small to justify a four-arm 30-step campaign before a
  structural MXFP4-chain optimization. No formal speed claim is made.
- Artifacts: `mbs4_fresh_smoke_analysis.json` and
  `mbs4_fresh_smoke_analysis.md` in the fresh evidence root.
- Status: resolved as a feasibility screen; formal ABBA intentionally not run.

### [2026-09-20 invalid-concurrent-mbs4-abba]
- A read-only sub-agent accidentally launched an MBS4 ABBA runner without GPU
  coordination. BF16 A1 completed while the root task's first SwiGLU
  microbenchmark overlapped part of it; MXFP4 B1 was interrupted after metadata.
- The A1/B1 directories and that first console-only microbenchmark are invalid
  by the exclusive-GPU protocol and are excluded from every conclusion. They
  are retained only for auditability and are marked in
  `INVALID_CONCURRENT_RUNS.md` in the fresh evidence root.
- Current process inspection after interruption found no `torchrun`, training,
  profiler, or benchmark process. Future GPU launches are root-coordinated only.
- Status: resolved by invalidation; no timing value from these artifacts is used.

### [2026-09-20 kfd-fresh-swiglu-kernel-correctness]
- Scope: new AITER SwiGLU validation only; no pre-existing profiling result was
  used as performance evidence. The tested AITER checkout is
  `e35bb17f4f815903bf73598facedbb321e15af28` with the local split-SwiGLU diff.
- A read-only review found two blockers before benchmarking: packed backward
  could read out of bounds when grad shape differed, and split backward omitted
  the BF16/FP16 cut points from eager `F.silu(gate) * up` autograd.
- The shared tiled forward/backward kernels now carry an explicit numeric mode.
  Legacy packed `swiglu_fwd/bwd` preserve their prior FP32-intermediate
  semantics, while `swiglu_fwd_split/bwd_split` preserve eager activation and
  multiplication rounding. Both modes remain one kernel implementation and the
  mode is included in the trace repr.
- Public contracts now validate shape, dtype, CUDA device, zero-copy flattening,
  preallocated outputs, and output aliasing. All empty cases, including `D=0`,
  return before division/view/grid construction. Packed backward validates the
  exact expected grad shape before launch.
- Fresh single-GPU AITER test result:
  `56 passed in 4.81s`, pytest exit zero. Coverage includes BF16/FP16, 1D,
  leading-empty and zero-width tensors, padded row strides, preallocated
  backward outputs, invalid dtype/layout/shape, alias rejection, and real eager
  autograd comparison. Legacy packed forward/backward matched the prior formulas
  element-for-element. The only post-success traceback was the known
  `torch.library._del_library` cleanup defect.
- Status: correctness gate passed; no step-speed credit is assigned. Next check
  is a fresh independent-process interleaved tile comparison before any tuning
  value is written to production configuration.

### [2026-09-20 kfd-fresh-swiglu-tile-confirmation]
- Exact Qwen3 shape `M=16384,D=12288` was measured in three independent Python
  processes. Each process used 60 alternating-order pairs, 10 launches per
  CUDA-event sample, and verified elementwise-identical outputs before timing.
- Forward `4x1024/8-warps` versus current `8x1024/2-warps` produced process
  speedups `0.996320x`, `0.978291x`, and `0.995216x`; it was slower in all three.
  Across 180 pairs it measured `0.989883x` by mean and `0.991269x` by median.
- Backward `1x512/4-warps` versus current `8x1024/2-warps` produced
  `1.003166x`, `0.996900x`, and `1.004472x`. The direction changed across
  processes and the aggregate point estimate was only `1.001505x` by mean.
- Decision: reject both single-sweep winners and retain the current tile. No
  external production config is added for an unstable or negative result.
- Artifact: `swiglu_tile_compare_analysis.md` in the fresh KFD evidence root.
- Status: resolved; proceed to real HF autograd integration and measurement.

### [2026-09-20 kfd-fresh-lumen-swiglu-autograd]
- Lumen now exposes a separate-input custom-autograd SwiGLU path with guarded
  AITER dispatch and a logged PyTorch fallback. Qwen3 integration patches only
  model instances, after `quant.enable` and before `apply_fsdp2`; it does not
  mutate the global Transformers class or replace gate/up/down linears.
- Targeted Lumen tests passed: `58 passed in 5.56s`, including actual GPU
  forward/backward, forced fallback, AITER rejection fallback, per-instance
  Qwen3 patching, SiLU guard, and launcher opt-in. The first test invocation
  lacked the repo `PYTHONPATH` and failed during import before collection; the
  corrected command passed. The known post-success torch-library cleanup
  traceback remained.
- Fresh exact-shape real-autograd benchmark used three independent processes.
  Mean forward was `0.443893 -> 0.285685 ms` (`1.553787x`). Mean
  forward+backward was `1.233774 -> 0.755137 ms` (`1.633841x`), saving
  `0.478637 ms/call`. Every process reported forward/dup SNR `inf`, dgate SNR
  `122.107 dB`, and finite outputs.
- With 36 MLPs and 8 accumulation microbatches, the kernel-local upper-bound is
  `137.85 ms/step`. This is not step-speed evidence; it only passes the gate to
  run an eight-GPU smoke and symmetric wall-time experiment.
- Artifact: `swiglu_autograd_analysis.md` in the fresh KFD evidence root.
- Status: integration correctness and microbenchmark gates passed; end-to-end
  smoke and wall-time measurement remain.

### [2026-09-20 kfd-fresh-swiglu-post-review-gpu-recheck]
- After hardening AITER's preallocated-output overlap checks and restoring the
  packed API's noncontiguous/broadcast compatibility, the fresh single-GPU
  targeted suite collected 59 tests and completed 58 before one numerical
  assertion failed.
- The only mismatch was one BF16 `dgate` element out of 3,158,016 at shape
  `(257, 12288)`: absolute error `0.001953125`, relative error `0.0045776`.
  Forward, `dup`, all FP16 cases, and the other BF16 cases passed. The process
  exited 1; the later `_del_library` tracebacks were teardown noise and do not
  override the failed pytest result.
- This is not accepted as a correctness pass. The immediate check is a
  multi-seed ULP/SNR comparison to distinguish a one-ULP exp2 approximation
  boundary from a kernel semantic error; the tolerance must not be relaxed
  without that evidence.
- Follow-up covered 64 seeds and 202,113,024 BF16 `dgate` elements. Only nine
  elements differed (`4.45e-8`), never more than one per seed; all differences
  were exactly one BF16 ULP, none exceeded one ULP, and the worst SNR was
  `108.951 dB`. A reproduced element sat at the BF16 rounding midpoint, where
  Triton/FMA ordering and ATen chose adjacent representable values.
- The test now applies a BF16-`dgate`-specific gate (`max ULP <= 1`, at most
  `max(1, ceil(numel / 1,000,000))` changed elements, SNR >= 40 dB), while
  forward, `dup`, and non-BF16 assertions remain unchanged. A deterministic
  midpoint regression case was added.
- Fresh single-GPU rerun: `60 passed in 5.05s`, exit zero; Ruff, Black 26.3,
  `py_compile`, and `git diff --check` also passed. The known post-success
  `_del_library` tracebacks remained.
- The post-review Lumen integration rerun also passed on one GPU:
  `58 passed in 5.65s`, exit zero, covering the custom autograd path, fallback,
  Qwen3 instance patching, and launcher flag. Its only later tracebacks were the
  same known `_del_library` cleanup defect.
- Status: resolved as a one-ULP numerical boundary; AITER correctness gate
  passed for the campaign.

### [2026-09-20 kfd-fresh-swiglu-campaign-lock-final]
- The SwiGLU smoke/A1-B-A2 harness now fails closed on candidate rejection,
  uses a shared host GPU lock, rejects non-service or unreadable KFD clients,
  hashes the Qwen config/tokenizer and both train/validation datasets, and only
  permits the exact post-success `torch.library._del_library` traceback shape.
- Runtime provenance is checked with `/usr/bin/python3` from the training
  environment: imported AITER is
  `/home/xdai/aiter/aiter/__init__.py`, both split APIs are callable, commit is
  `e35bb17f4f815903bf73598facedbb321e15af28`, and the frozen AITER dirty-tree
  digest is `23123e9ef7dd102f08a62bd7a4a454f28bf9bf8598c18097dc71829a054bbacc`.
- Each arm clears inherited profiler variables and CUDA/ROCm device-visibility
  overrides before installing the frozen eight-GPU environment.
- Final frozen bundles: direct source
  `180e57960f670ccfd347b6a48e61f64aa45ddb4394dc3c27f0b56dfc9c1733f5`,
  SwiGLU source
  `9ca2dd17296b51b1350882b1e8713a45ba2ca74c938733089999499c9e4f2b48`,
  workload `da4b31dde571a889aceae845da762c20c9f71976644269e718a58ecc8b2fe3dd`,
  and cache `990179febd8be6f8bc6cb3a637f656c45bea4d8a4b4404c1a9e250751a9f24c1`.
- `bash -n`, `py_compile`, Ruff lint/format, analyzer self-test, and the final
  sanitized `DRY_RUN=1` preflight all exited zero. No GPU process was started.
- Status: resolved; the next action is the real eight-GPU campaign.

### [2026-09-20 kfd-fresh-swiglu-campaign-rejection-and-rollback]
- The post-lock eight-GPU campaign completed in the required order: three-step
  fused smoke, A1 split fusion off, B split fusion on, and A2 split fusion off.
  All formal arms completed 30/30 updates with identical model initialization,
  matching first-update batch hashes on every rank, finite loss/gradient norms,
  and no fallback, OOM, NaN/Inf, skipped update, or training-time exception.
- The original campaign exit remains `2`. Its analyzer incorrectly required the
  smoke arm (`--train-samples 640`) to have the same shuffled batch hashes as
  the formal timing arms (`--train-samples 4096`). The formal A1/B/A2 arms do
  match each other exactly, so a separately labelled post-campaign audit was
  generated without rewriting the original status.
- Formal steps 11-30: A1 off mean/median 6080.585/5883.750 ms, B on
  5937.530/5859.050 ms, and A2 off 6036.440/5882.550 ms. Relative to the
  symmetric off bracket, B was 1.020376x by mean but only 1.004113x by the
  run-level median; paired wins were 16/20 and the block-4, 100,000-sample
  moving-block bootstrap 95% speedup CI was [1.005951x, 1.036395x]. Control
  drift was 0.7286% mean and 0.0204% median.
- The numerical screen also failed: B validation NLL was 9.1892 versus 9.1404
  for the control bracket, delta +0.0488 nats (about +5.001% perplexity), above
  the predeclared +0.03 gate. Peak allocation fell from 138.9 to 125.4 GiB/GPU,
  but memory savings do not override the failed median and accuracy gates.
- Decision: reject the Qwen3 separate-input split-SwiGLU candidate. The Lumen
  runtime wiring, CLI/launcher switch, probe, integration module, and candidate
  tests were removed locally. The independent packed Megatron SwiGLU path and
  AITER packed-kernel correctness hardening (bounds, empty tensors, alias and
  noncontiguous handling, and legacy FP32-intermediate semantics) were kept.
- Audit artifacts:
  `/home/xdai/profile-results/lumen-mxfp4-kfd-rerun-20260920-094252/swiglu_fresh_audit_analysis.json`
  and `swiglu_fresh_audit_analysis.md`.
- Status: resolved; no speedup from this candidate is credited. The next
  BF16/MXFP4 baseline and profile must use a new result directory, new cache,
  and the rolled-back source fingerprint; no earlier profile is accepted for
  selecting the next optimization.

### [2026-09-20 postrollback-fresh-baseline-and-profile]
- Scope: only the new eight-GPU KFD campaign under
  `/home/xdai/profile-results/lumen-mxfp4-postrollback-20260920-143648/` is
  accepted as performance evidence. It used the rolled-back Lumen source,
  a newly generated MXFP4 autotune cache, and the fixed Qwen3-8B recipe:
  sequence length 8192, MBS2, GBS128/GA8, FSDP2 `full_shard`, retained
  accumulated parameters, no gradient checkpointing, BF16 reduction, MXFP4
  communication off, and five BF16 tail layers.
- Formal unprofiled A1/B/A2 steps 11-30: BF16 A1 mean/median
  `8315.810/8216.000 ms`, MXFP4 B `6018.175/5923.350 ms`, and BF16 A2
  `8153.315/8113.050 ms`. The symmetric BF16 reference is
  `8234.5625/8164.525 ms`, so MXFP4 reaches `1.368282x/1.378363x`. The
  block-4, 100,000-sample moving-block bootstrap 95% speedup interval is
  `[1.342426x, 1.396570x]`, with 20/20 paired wins. To reach `1.6x`, the
  current MXFP4 mean/median must fall to `5146.602/5102.828 ms`, leaving
  `871.573/820.522 ms` or `14.482%` mean reduction.
- Short-run numerical screen: final validation NLL was `9.1771` for the BF16
  bracket and `9.1398` for MXFP4, delta `-0.0373` nats. All logged losses,
  gradient norms, learning rates, timings, memory values, and validation
  values were finite. This is evidence against an immediate regression, not a
  long-horizon convergence claim.
- Fresh rank-0 profile, steps 7-8: BF16 envelope/busy/idle was
  `8306.540/8167.005/139.535 ms` per step; MXFP4 was
  `6095.169/5627.773/467.397 ms`. MXFP4 therefore exposes an additional
  `327.862 ms/step` of GPU idle. Its exclusive categories were A4W4
  `1330.058 ms`, residual BF16 GEMM `1116.219 ms`, quant/layout
  `402.305 ms`, attention `1670.905 ms`, collectives `143.069 ms`, activation
  `453.366 ms`, optimizer `24.545 ms`, and other `487.307 ms` per step.
- The largest MXFP4-specific kernels were the ASM 256x256 A4W4 family
  (`558.658 ms/step`, 2480 calls), shuffled-Triton A4W4
  (`465.148 ms/step`, 1984 calls), ASM 128x512 A4W4
  (`311.366 ms/step`, 744 calls), and dual-layout MXFP4 quantization
  (`342.984 ms/step`, 3472 calls). Quant/layout's complete interval-union
  ceiling is only `409.264 ms/step`, so deleting quantization alone cannot
  close the `871.573 ms` target gap; a structural combination is required.
- Integrity: every training arm and the campaign exited zero; model-init hash,
  all eight rank-local first-update batch hashes, source bundle, cache, and
  tree state matched. Strict zero-traceback is false: every arm emitted only
  the exact allowlisted `torch.library._del_library` teardown traceback after
  `Training complete`. The explicit post-success teardown allowlist gate and
  the overall comparison-integrity gate pass.
- Known provenance limitations: model/tokenizer/data bytes, the campaign
  runner itself, and the AITER config-cache directory were not frozen; GPU
  idleness was checked only at campaign start. The next candidate harness must
  close these gaps.
- Status: baseline/profile accepted for this post-rollback round. The next
  candidate must be selected from this trace and validated with fresh
  correctness, microbenchmark, smoke, and symmetric unprofiled timing.

### [2026-09-20 fresh-tail5-tail4-tail5-bracket]
- Scope: only the fresh artifacts under
  `/home/xdai/profile-results/lumen-mxfp4-tail-sweep-20260920-152755/` are
  used for this decision. The fixed recipe was Qwen3-8B on 8 x MI350X,
  sequence length 8192, MBS2, GBS128/GA8, FSDP2 `full_shard`, retained
  accumulated parameters, no gradient checkpointing, BF16 reduction, MXFP4
  communication off, and the same seed/data/model initialization.
- The original campaign and the supplemental tail5 A2 arm both exited zero.
  The supplement held the global GPU lock, checked KFD idle before and after,
  and revalidated the frozen Lumen tree, AITER tree, workload, direct runner,
  source bundle, MXFP4 autotune cache, and empty AITER config-cache directory.
  The five formal arms have identical model-init and all eight rank-local
  first-update batch hashes.
- Steps 11-30: BF16 A1/A2 means were `8262.180/8163.160 ms`; their per-step
  midpoint mean/median was `8212.670/8159.150 ms`. MXFP4 tail5 A1/A2 were
  `6026.375/5872.200 ms` and `6158.810/6072.550 ms` by mean/median; the
  tail5 per-step midpoint was `6092.5925/5979.800 ms`. Tail5 A2 drifted
  `+2.1976%` by mean and `+3.4118%` by median versus A1, so the candidate
  decision uses the symmetric midpoint rather than either control alone.
- Tail4 measured `5921.750/5753.350 ms`. Relative to the tail5 midpoint it is
  `1.028850x` faster by mean and `1.039360x` by median, saving
  `170.8425/190.525 ms`; it won `15/20` paired steps. A block-4,
  100,000-sample circular moving-block bootstrap gives a mean-speedup 95%
  interval of `[1.016480x, 1.040547x]`.
- Relative to the fresh BF16 midpoint, tail4 is `1.386865x` by mean and
  `1.418156x` by median. The 1.6x target is not met: tail4 remains
  `788.831 ms/step` above the mean target and `653.881 ms/step` above the
  median target.
- Short-run validation NLL was `9.1769` for the BF16 midpoint, `9.13865` for
  the tail5 midpoint, and `9.1484` for tail4. Tail4 is `+0.00975` nats versus
  tail5 (about `+0.980%` perplexity) but `-0.0285` nats versus BF16, passing
  the predeclared `+0.03` short-run screen. This is not a long-horizon
  convergence or time-to-quality claim.
- All logged training/validation values were finite; there was no fallback,
  OOM, NaN/Inf, skipped update, or training-time kernel failure. Each arm only
  emitted the known post-success `torch.library._del_library` teardown
  traceback after `Training complete`.
- Decision: adopt tail4 for the next fresh profile/optimization round. It is a
  measured recipe-level speed improvement with a passing short-run numerical
  screen, but it does not by itself approach the remaining 1.6x gap and still
  requires longer matched-token convergence validation before a final quality
  claim.
- Analyzer artifacts: `tail4_analysis.json` and `tail4_analysis.md` in the
  fresh result directory. Analyzer execution, assertion-based self-test, Ruff
  lint/format, and `py_compile` all passed.

### [2026-09-20 tail4-current-only-profile-remeasurement]
- Scope reset: every profiling, bottleneck, and target-gap number in this entry
  comes only from
  `/home/xdai/profile-results/lumen-mxfp4-tail4-profile-20260920-161914/`.
  No earlier trace or timing directory is used to select the next candidate.
- Integrity passed: both eight-GPU profile arms exited zero; their commands
  differ only by `--mode`; source, tree, workload, direct-runner, profiler,
  autotune-cache, and empty AITER-config-cache locks match; pre-quantization
  model initialization and all eight rank-local first-update batch hashes
  match; all logged values are finite; and no fallback, OOM, NaN/Inf, skipped
  update, or training-time failure occurred. The only tracebacks are the known
  post-success `torch.library._del_library` cleanup defect.
- Rank-0 device-0 profiler window, steps 7-8: BF16 profiler span / GPU envelope
  / busy union / idle was `8258.955 / 8256.071 / 8102.045 / 154.026 ms` per
  step. MXFP4 tail4 was
  `6276.937 / 6271.525 / 5705.489 / 566.036 ms` per step. MXFP4 removes
  `2396.557 ms/step` of GPU-busy work but adds `412.010 ms/step` of idle, so
  its profiler-span speedup is only `1.315762x`.
- Current-profile target accounting: `8258.955 / 1.6 = 5161.847 ms/step`;
  MXFP4 therefore has a fresh diagnostic gap of `1115.091 ms/step`. This is a
  profiler-window sizing result, not the final throughput acceptance metric;
  acceptance requires a fresh unprofiled symmetric campaign.
- MXFP4 disjoint raw work per step includes A4W4 GEMM `1376.827 ms` across
  `5376` calls, residual BF16 GEMM `989.238 ms` across `696` calls,
  attention `1679.277 ms`, quant/layout `370.247 ms` across `4480` calls,
  activation/residual elementwise `619.864 ms`, collectives `345.142 ms`,
  norm `223.720 ms`, RoPE `73.195 ms`, cross entropy `55.412 ms`,
  copy/memset `118.813 ms`, and optimizer `15.820 ms`.
- Largest format-specific families are ASM A4W4 256x256
  (`574.110 ms/step`, `2560` calls), shuffled-Triton A4W4
  (`476.506 ms/step`, `2048` calls), dual-layout quantization
  (`358.360 ms/step`, `3584` calls), and ASM A4W4 128x512
  (`326.212 ms/step`, `768` calls). FMHA backward/forward are
  `1331.843/292.362 ms/step` but are common-path work and are not the first
  lever for improving the MXFP4/BF16 ratio.
- MXFP4 has exactly `4480` more physical GPU events and `4480` more host
  launches per step than BF16, equal to the full quant/layout call count.
  However, deleting all quant/layout intervals has only a `369.866 ms/step`
  union ceiling, far below the `1115.091 ms/step` gap. Quant-only work cannot
  reach 1.6x.
- The overlap-safe union of A4W4 and quant/layout is `1735.789 ms/step`.
  Adding all `412.010 ms/step` of excess idle gives an extreme, explicitly
  causal-unproven ceiling of `2147.800 ms/step`; idle is not double-counted
  with the joint busy-time union. This structural chain is the only measured
  MXFP4-specific aggregate large enough to cover the fresh gap.
- Residual BF16 work is not a one-layer ceiling. The three vocabulary/lm-head
  shapes contribute `470.385 ms/step`; the four protected transformer tail
  layers contribute `518.854 ms/step`. One additional tail layer therefore has
  only a `129.713 ms/step` gross raw-work ceiling, before subtracting its
  replacement A4W4 and quantization work and considering overlap.
- Shape attribution limitation: all `43276` BF16 physical events have a direct
  `External id`. MXFP4 has `1792` events over two steps (`896/step`) without
  one; they are convert (`224/step`), packed-transpose (`224/step`), and
  scale-swizzle (`448/step`) helpers, and correlation containment recovered no
  producer shapes. Direct shape mapping remains complete for dual-layout
  quantization and the main A4W4 GEMM families.
- Accuracy must not be accepted from this trace: profile-run validation NLL was
  `12.3956` BF16 versus `12.7720` MXFP4, delta `+0.3764`. If the `+0.03` gate
  were applied, this eight-step profiled run would fail. Both arms are finite,
  but a fresh unprofiled matched run is required for the numerical decision.
- Next candidate: audit and microbenchmark `(M,N,K)=(12288,4096,16384)` first,
  one backend/tile change at a time. The current trace plus its frozen cache
  uniquely map this shuffled-Triton wgrad group to `327.223 ms/step` across
  `512` calls, or `68.67%` of the whole shuffled family. Any new or modified
  GPU kernel belongs in AITER; Lumen owns only probe/dispatch and integration.
  Require exact correctness/SNR, no production-shape regression above the
  predeclared bound, then an eight-GPU smoke and symmetric unprofiled A/B/A
  campaign before crediting step speed.
- Status: current-only profile analysis complete; no new GPU workload was
  launched while preparing this entry.

### [2026-09-20 user-requested-fastpath-rollback-and-fresh-reset]
- User directive: do not use any existing timing or profiler artifact as
  evidence for the next optimization decision. All performance conclusions
  below this reset must come from a newly created result directory, fresh
  MXFP4 autotune cache, and newly executed BF16/MXFP4 runs on the authorized
  KFD devices.
- Rolled back only the latest registry-freeze and weight-cache pre-resolution
  fast-hit experiment. Removed both environment-variable paths and their
  dedicated tests while preserving live file-aware ASM resolution, coherent
  single-snapshot identity checks, artifact fail-closed validation, generic
  backend replay invalidation, and the existing per-step MXFP4 weight cache.
- The dirty Lumen/AITER worktrees were not globally restored. Unrelated user
  changes, the untracked but required `mxfp4_asm.py`, and independent AITER
  SwiGLU correctness work remain untouched.
- Static validation passed: `git diff --check` and `py_compile` both exited
  zero. Focused regression under the shared GPU lock passed `13` tests with
  `289` deselected; pytest exited zero. The only later traceback was the known
  `torch.library._del_library` interpreter-teardown defect.
- Final broader rollback regression passed `55` tests with `247` deselected in
  `4.75s`; pytest exited zero. The only post-test traceback remained the same
  known `torch.library._del_library` interpreter-teardown defect.
- Status: rollback complete; fresh baseline campaign and fresh profiling are
  required before selecting or crediting any optimization.

### [2026-09-20 full-fresh-rerun-reset]
- User directive: discard every pre-existing timing and profiler artifact for
  the next decision and perform both BF16 and MXFP4 measurements again using
  the explicitly authorized KFD devices.
- The in-progress BF16-only profiler was interrupted during model construction
  because its analyzer paired against an older MXFP4 trace. It produced no
  completed profiling artifact and is invalid for this round.
- The most recent A4W4 candidate was rolled back by deleting only the exact
  gfx950 `(12288,4096,16384)` 128x512 tuning row. The AITER table now matches
  its repository `HEAD` byte-for-byte; unrelated dirty Lumen and AITER changes
  remain untouched.
- Fresh acceptance protocol: create a new result directory and MXFP4 autotune
  cache; run an unprofiled BF16/MXFP4/BF16 bracket with identical initialization
  and per-rank first-update batch digests; then profile both precisions from the
  same frozen source/cache snapshot. Existing profiles cannot select a new
  optimization candidate.
- Status: reset and exact rollback complete; fresh GPU campaign pending.

### [2026-09-21 fresh-loss-readback-aba-rejection]
- Scope: only the new eight-GPU A/B/A artifacts under
  `/home/xdai/profile-results/lumen-mxfp4-loss-readback-fresh-20260920-234356/`
  are used. The order was baseline A1, `--defer-loss-readback` candidate B,
  baseline A2; all arms used the same frozen source bundle and fresh autotune
  cache, Qwen3-8B inputs, initialization, and 8-rank batch/validation digests.
- Steps 11-30: A1 mean/median was `6081.715/5956.800 ms`, A2 was
  `6233.330/6115.950 ms`, and the per-step control midpoint was
  `6157.523/6015.850 ms`. Candidate B was `6494.455/6482.800 ms`, or
  `0.948120x/0.927971x`; it lost `16/20` paired steps. The block-4 circular
  moving-block bootstrap 95% mean-speedup interval was
  `[0.928711x, 0.970060x]` over 100,000 fixed-seed resamples.
- The candidate timing cutoff occurred before its final deferred `.item()`, so
  the reported B time is biased in its favor. It was still slower by
  `336.933 ms` on the mean. Fresh trace causality review also bounded directly
  attributable MXFP4 post-item idle at about `19.1 ms/step`, confirming this
  is not a material path to the 1.6x target.
- Validation NLL was `9.1313` for B versus `9.1712` for the A1/A2 midpoint;
  all loss/gradient values were finite. Integrity passed `37/37`: train and
  postflight exits, KFD idle checks, source/cache/tree/runtime hashes, model
  init, and all rank-local data digests matched. The only traceback was the
  known post-success `torch.library._del_library` teardown defect.
- Decision: reject the candidate and do not credit it toward speedup. The
  `--defer-loss-readback` parser option and training-loop branch were removed
  exactly; `py_compile`, CLI absence check, and `git diff --check` passed.
- Status: resolved; proceed to a structural MXFP4 gate/up projection fusion
  that reduces quantization, GEMM, and autograd-launch counts together.

### [2026-09-21 root-retention-profile-and-semantic-bracket]
- Scope: the mechanism profile under
  `/home/xdai/profile-results/lumen-mxfp4-root-retain-fresh-20260921-005209/`
  and the independent A2 control under
  `/home/xdai/profile-results/lumen-mxfp4-root-retain-semantic-a2-fresh-20260921-022208/`.
  The candidate is the opt-in `--fsdp-retain-root-params` flag; the default
  remains off.
- Integrity passed for all three arms. Source bundle
  `0f32e3453e11efe3f0bb221765e3f7ae28205a6a7cda6da16d8d25280485ede2`,
  MXFP4 cache `6364a0737b4eba0fe644c973af7655b70d53ae0c57725fa66c07123f152f3f24`,
  Lumen diff `23efd2c14fcae30fa47ac6b4513a3948ae7b6514ff01e5cf952b93081215fd0b`,
  and AITER diff `4444e47732a2a3aa96d20393a293e49f7142719123a608b5beb884b2d01b6b1a`
  match. Training, postflight, provenance, and idle checks exited zero.
- Mechanism: root FSDP all-gathers fell exactly from 9 to 1 per step. The
  root annotation union fell from about 515.653 to 214.160 ms/step and peak
  allocation from 138.9 to 137.8 GiB. The short profiled wall measurements did
  not establish a repeatable step-time improvement, so the annotation delta is
  not credited as throughput.
- Numerical bracket: A1/B/A2 step-3 losses were
  `12.7801 / 12.7775 / 12.7763`; B lies inside the control span. Step-2 grad
  norms were `5.812 / 5.812 / 5.844`, so the repeated control itself moves by
  about 0.55%. Validation was `12.7872 / 12.7886 / 12.7868`; B is +0.0016
  above the control midpoint and remains outside the original 0.001 gate.
- Decision: hold, do not credit. The implementation appears mechanically
  consistent and the training deltas are compatible with existing attention /
  kernel nondeterminism, but the preregistered semantic failure cannot be
  waived. A deterministic A1/B/A2 with every-batch, MXFP4 Philox stream,
  full-precision loss/grad, root grad-shard, parameter, and Adam-state hashes is
  required before this opt-in flag can be accepted. It is not the next
  performance candidate because the profile did not show a useful wall-time
  ceiling.

### [2026-09-21 fresh-dual-layout-tile-sweep]
- Scope: only the new artifacts under
  `/home/xdai/profile-results/lumen-mxfp4-dual-layout-tiles-fresh-20260921-025101/`
  are used. Three fresh processes swept BM/BN
  `64x32, 128x32, 256x32, 64x64, 128x64, 256x64`, with different orders,
  10 warmups and 30 CUDA-event samples per tile. The exact dense Qwen cases
  were activation `(16384,4096)` with direct shuffled-column storage, separate
  gate/up gradient `(16384,12288)`, and prospective packed gradient
  `(16384,24576)`.
- Integrity passed: the run exited zero, held the global GPU lock, accepted
  only the PID-1 root `gpuagent` service process, pinned imports to
  `/home/xdai/Lumen` and `/home/xdai/aiter`, and observed identical pre/post
  Lumen and AITER tree hashes. All 54 tile/case/process comparisons were RTN
  bitwise equal to the current `256x32` output. The only tracebacks were the
  known post-success `torch.library._del_library` teardown defect.
- Median of the three process medians for `256x32` was `0.097501 ms` on the
  activation, `0.212562 ms` on the separate gradient, and `0.377444 ms` on the
  packed gradient. It won every case. BM128/BN32 achieved only
  `0.911045x / 0.856404x / 0.865808x` versus `256x32`; the closest challenger,
  BM256/BN64, achieved `0.970531x / 0.940698x / 0.971083x`.
- Analysis: GPT-OSS's BM64->128 result does not transfer to this dense kernel.
  It lacks per-expert metadata scans to amortize, and reducing BM below 256
  adds workgroups and repeated tile setup. Increasing BN to 64 also loses on
  every aggregate median.
- Decision: reject all tile changes and keep production `256x32`; no source or
  end-to-end candidate follows from this sweep. Continue with structural packed
  gate/up projection, which removes quant and GEMM consumers together rather
  than only retuning an already efficient producer.

### [2026-09-21 fresh-lm-head-vocabulary-quantization]
- Scope: only fresh single-GPU MI350X measurements under
  `/home/xdai/profile-results/lumen-mxfp4-lm-head-quant-fresh-20260921/`
  and the separate public-AITER A16WFP4 sweep under
  `/home/xdai/profile-results/lumen-mxfp4-lm-head-a16fp4-sweep-fresh-20260921-035615/`
  are used. CUDA-event timing used 5 warmups and 15 samples. No production
  Lumen/AITER source or configuration was changed; the work added benchmark
  scripts/results only.
- Exact Qwen3-8B output shape is `(M,V,K)=(16384,151936,4096)`.
  `M*V=2,489,319,424` exceeds signed int32, and a complete BF16 logits tensor
  occupies about 4.6367 GiB. The current `quantize_output_layer` MXFP4 path
  therefore still takes its int32-safety fallback to BF16. Current training
  also materializes the complete logits tensor before fused cross entropy;
  AITER's existing chunked cross entropy chunks token rows, not vocabulary.
- Final `results_v3.json` forward medians were: BF16 `17.776 ms`; two-chunk
  A4W4 `8.035 ms` (`2.212x`, one-time weight quantization `0.643 ms`);
  FP8 per-tensor `13.491 ms` (`1.318x`, weight quantization `2.175 ms`);
  FP8 per-token `16.518 ms` (`1.076x`, weight quantization `4.156 ms`); and
  FP8 blockwise 1x128 `26.333 ms` (`0.675x`, weight quantization `0.569 ms`).
  These are forward-kernel measurements only and exclude cross entropy, dX,
  dW, and end-to-end scheduling.
- Real-checkpoint accuracy used the Qwen3-8B `lm_head`, real final hidden
  states from three C4 samples, 589 next-token positions, and the complete
  151,936-token vocabulary. A4W4 measured SNR `14.159 dB`, KL mean/p99
  `0.115665/0.457124`, top-1 agreement `78.27%`, and true-label delta NLL
  `+0.086155`; the weight-only FP4 oracle measured `17.841 dB`,
  `0.064682/0.230306`, `85.40%`, and `+0.061486`. Both are rejected for
  accuracy.
- On the same checkpoint screen, FP8 per-tensor measured SNR `28.522 dB`, KL
  mean/p99 `0.005688/0.023675`, top-1 `93.38%`, and delta NLL `+0.009396`;
  per-token measured `28.984 dB`, `0.004745/0.018854`, `95.93%`, and
  `-0.004386`; blockwise 1x128 measured `30.516 dB`,
  `0.004181/0.015585`, `96.10%`, and `-0.002310`. Against the conservative
  gate KL mean <= `0.002`, KL p99 <= `0.01`, and delta NLL <= `+0.01`, no FP8
  path passes all precision criteria: per-tensor is only a prototype candidate,
  per-token has too little speed gain, and blockwise is slower than BF16.
- True weight-only A16W8 used a separate fresh BF16 median of `17.689 ms`.
  Its plain and preshuffled forwards were `30.170 ms` (`0.586x`) and
  `40.204 ms` (`0.440x`), with one-time weight quantization `0.581 ms`.
  Checkpoint accuracy was SNR `31.862 dB`, KL mean/p99
  `0.003853/0.015322`, top-1 `96.43%`, and delta NLL `-0.001352`.
  It is numerically strongest among the tested weight-only paths but is
  rejected because the forward is 1.71x slower than its BF16 reference and it
  still misses the conservative KL thresholds.
- The separately swept public AITER `gemm_a16wfp4_preshuffle` reached
  `30.733 ms` versus its fresh BF16 reference `17.853 ms`: only `0.581x`
  BF16 speed, or 1.721x slower. The implementation quantizes BF16 activation
  tiles inside the GEMM and executes FP4 x FP4 `dot_scaled`, so it is a fused
  A4W4 path rather than true W4A16. Its small MXFP4-representable reference
  check validates layout/arithmetic only; it is not real-checkpoint accuracy
  evidence. Reject it before CE/backward integration.
- FP8 per-tensor vocabulary chunking improved kernel-only median from
  `13.299/13.282` ms at 2/4 chunks to `12.834 ms` at 8 chunks. With a cached
  weight across GA8, the theoretical savings were
  `32.934/33.083/36.681 ms/update`. The required `torch.cat` full-logits
  materialization raised medians to `15.345/15.400/15.173 ms` and reduced
  those theoretical savings to `16.565/16.143/17.965 ms/update`; this is too
  small to materially close the 1.6x step target and remains neither CE nor
  backward nor end-to-end timing.
- A direct preallocated full-output experiment cannot be credited: writing
  AITER GEMM results into noncontiguous vocabulary slices disagreed with the
  standard chunk outputs by max-absolute error `10.5-10.75` (`10.53125` for
  8 chunks). This indicates an AITER strided-output/split-K correctness issue,
  not a valid no-copy speed result.
- Decision: keep production `lm_head` BF16. A4W4/FP4 are rejected for
  checkpoint accuracy; A16W8, public A16WFP4, and blockwise FP8 are rejected
  for performance; per-token FP8 is not material. Per-tensor FP8 with eight
  vocabulary chunks is the only candidate worth an opt-in prototype, but its
  current accuracy is not conservatively accepted and vocabulary quantization
  alone cannot be the main path to 1.6x.
- Next check: first add an AITER kernel-level correctness test for the
  noncontiguous preallocated-output failure. Only after that passes, build an
  opt-in Lumen custom-autograd prototype with eight vocabulary chunks, shared
  FP8 activation quantization, optimizer-step weight caching, contiguous
  preallocated logits, existing BF16/FP32 CE, and BF16 dX/dW/master weights.
  Require a complete forward+CE+backward benchmark before an eight-GPU A/B,
  then a paired `3 seeds x 200 steps` quality run with validation-delta-NLL
  95% upper bound approximately `+0.01` before any production adoption.
- Status: resolved for the current decision; production remains BF16 and the
  guarded opt-in investigation is open.

### [2026-09-21 fresh-lm-head-backward-layout-audit]
- Production-path correction: the default Qwen3 MXFP4 trainer does not enable
  `quantize_output_layer` or `lumen_linear`; their defaults remain false, and
  the quantization patcher skips `lm_head`. The observed 217 quantized and 36
  skipped linears independently match 31 active transformer layers x 7 plus
  five protected layers x 7 and one BF16 `lm_head`. Therefore the default
  output layer is stock HuggingFace `nn.Linear`, not
  `QuantizedLinearFunction`.
- The local Qwen3-8B config has `tie_word_embeddings=false`, so `lm_head` owns
  an independent `(151936,4096)` BF16 weight and requires a full dW.
- Fresh exact-shape layout measurements under the exclusive GPU lock used
  `(M,V,K)=(16384,151936,4096)`. Stock `nn.Linear` forward+dX+dW measured
  `54.898 ms`. Native PyTorch no-copy dX/dW measured `18.542/19.385 ms`;
  AITER inferred or explicit transpose-view paths measured
  `18.621-18.627/19.323-19.373 ms`. Exact-shape AITER outputs were bitwise
  equal to native `torch.mm`.
- Counterfactual explicit-copy paths measured `29.964 ms` for dX and
  `30.074 ms` for dW, including `13.026 ms` for `weight.T` copy, `11.100 ms`
  for `dY.T` copy, and `0.284 ms` for `X.T` copy. These copies occur only in
  the opt-in `QuantizedLinearFunction(scaling_type="none")` fallback and are
  dead for the default production `lm_head`; they cannot be credited as a
  production optimization opportunity.
- Decision: reject a stride-aware AITER BF16 dX/dW integration for the current
  production path. Native PyTorch already consumes the transpose views without
  materializing GEMM-operand copies; AITER's best dW result is only about 0.3%
  faster and dX is slower.
- Separate remaining cost: root FSDP2 copies/upcasts the full unsharded
  `lm_head` gradient into its flat reduce-scatter input. With the current FP32
  reduction policy, the `lm_head` staging contribution is about `2.319 GiB`.
  This is FSDP communication staging, not transpose materialization, and must
  be investigated independently from GEMM layout.
- Artifact:
  `/home/xdai/profile-results/lumen-mxfp4-lm-head-bwd-layout-fresh-20260921-050243/`.
- Status: resolved; keep native PyTorch for production `lm_head` backward and
  remove BF16 transpose-copy elimination from the main optimization plan.

### [2026-09-21 fresh-vocabulary-hybrid-and-input-embedding]
- Follow-up question: determine whether a precision-preserving vocabulary-layer
  quantization can accelerate the current Qwen3-8B MXFP4 training path. New
  measurements only were collected under the exclusive GPU lock in
  `/home/xdai/profile-results/lumen-mxfp4-vocab-hybrid-fresh-20260921-053214/`;
  no production Lumen/AITER source or configuration was changed.
- Real-checkpoint accuracy used the Qwen3-8B `lm_head`, complete 151,936-token
  vocabulary, final hidden states from three C4 samples, and 589 next-token
  positions. The raw 8-chunk per-tensor FP8 candidate measured KL mean/p99
  `0.005629/0.023158`, top-1 agreement `93.72%`, and delta NLL `+0.009360`, so
  it again failed the conservative KL gates (`0.002/0.01`).
- A label-aware mixed candidate recomputed the approximate top-k logits and
  true-label logit from the BF16 master weight. Top-4 remained just outside the
  gates (`0.002265/0.010969`), while top-8 passed the static screen with KL
  mean/p99 `0.001454/0.006992`, top-1 agreement `100%`, and delta NLL
  `-0.002802`. This establishes numerical potential only, not convergence.
- Exact-shape `(M,V,K)=(16384,151936,4096)` CUDA-event medians were BF16
  forward `17.705 ms`, cached-weight 8-chunk FP8 including activation quant and
  full-output `cat` `15.017 ms` (`1.179x`), top-8 selection alone `12.120 ms`,
  nine selected BF16 dot products `1.678 ms`, and the complete mixed forward
  `29.650 ms` (`0.597x`). The precision-preserving candidate is therefore
  materially slower than BF16.
- With GA8, raw FP8 would save only about `19.35 ms/update` after one measured
  `2.155 ms` weight quantization, before CE/backward; it already fails accuracy.
  Even an unrealistically free fused top-8 selection and scatter would leave
  only about `5.93 ms/update` theoretical saving after selected-dot and weight
  quantization, about `0.62%` of the current `957.41 ms` target gap.
- The untied input embedding was also measured at 16,384 lookups from a
  `(151936,4096)` table. BF16 lookup median was only `0.0517 ms`; current FP8
  per-tensor/per-row gather-cast-scale medians were `0.2259/0.2303 ms`
  (`0.229x/0.224x` BF16). Local output SNR was `31.409/31.512 dB`, but the
  performance gate fails before any expensive end-to-end accuracy study.
- Decision: retain BF16 for both input embedding and production `lm_head`.
  Reject raw FP8 for precision and top-8+label correction for performance; do
  not advance either to 8-GPU E2E or multi-seed convergence. Vocabulary-layer
  quantization cannot materially close the 1.6x target on this workload.
- Artifact:
  `/home/xdai/profile-results/lumen-mxfp4-vocab-hybrid-fresh-20260921-053214/analysis.md`.
- Status: resolved negative result; continue higher-Amdahl transformer and
  FSDP optimization work.

### [2026-09-21 fresh-current-wgrad-asm-formal-aba]
- Scope: fresh single-variable control/candidate/control campaign under
  `/home/xdai/profile-results/lumen-mxfp4-current-wgrad-asm-fresh-20260921-063804/`.
  All three arms used 8 GPUs, Qwen3-8B, sequence length 8192, MBS2, GBS128/GA8,
  FSDP2 `full_shard`, BF16 reduction, retained accumulated parameters, no
  gradient checkpointing, five BF16 tail layers, seed 1234, and 16 validation
  batches. The only intended variable was ASM admission for A4W4 WGrad shape
  `(M,N,K)=(12288,4096,16384)`.
- Kernel correctness preceded E2E timing. Three independent seeds produced
  bitwise-identical plain, shuffled, ASM, and chunked dequantized-FP32 outputs.
  Actual Lumen dispatch measured shuffled medians `0.602666/0.606707 ms` around
  candidate ASM median `0.369164 ms`, all at `55.6165 dB` SNR. Candidate
  identity was exact ASM `128x512`, split-K 0, manifest hash prefix
  `4e0dce9b9642d4fb`, and code-object hash prefix `c32d5356e5d92b77`.
- Formal step 11-30 means/medians were control A `6190.065/6000.400 ms`,
  candidate B `5964.930/5878.500 ms`, and control C `6148.465/5919.750 ms`.
  Against the A/C per-step midpoint, B saved `204.335 ms` by mean and
  `121.350 ms` by median, for `1.034256x/1.017602x`; it won 17/20 paired
  steps. A block-4 circular moving-block bootstrap with 100,000 resamples gave
  a speedup 95% CI of `[1.018162x,1.048172x]` and saving CI
  `[108.039,289.290] ms`.
- Accuracy/integrity passed the predeclared short-run gates. Validation NLL was
  `9.1684/9.1369/9.1344` for A/B/C; candidate delta versus the control midpoint
  was `-0.0145`. All arms completed 30/30 updates with finite values, identical
  model-init, rank-local first-batch and validation-batch digests, frozen
  source/cache/runtime hashes, zero train/postflight/KFD statuses, and only the
  allowlisted post-success `torch.library._del_library` traceback. Analyzer
  integrity was `39/39` and decision `accept`; this is not a long-horizon
  convergence claim.
- Production integration: added the exact admitted row to
  `examples/qwen3/configs/qwen3_8b_a4w4_blockscale_tuned_gemm.csv`, selecting
  `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E` with split-K 0.
  The AITER CSV validator reported 0 errors, 24 focused Lumen ASM/autotune tests
  passed, and explicit three-table lookup found exactly one target row with the
  expected artifact identity. A fresh real Lumen production-table dispatch at
  `/home/xdai/profile-results/lumen-mxfp4-production-row-dispatch-fresh-ZgQQAb3k/`
  selected ASM, measured `0.365284 ms` median over 256 CUDA-event samples, and
  passed the FP32-reference gate at `55.6165 dB`.
- Decision: accept and retain this model-specific row. It materially improves
  current MXFP4 step time but does not close the same-policy 1.6x goal; a fresh
  final matched BF16/MXFP4 run remains required after the next structural
  optimization(s).
- Status: resolved and integrated; continue with the highest-Amdahl
  A4W4-plus-quantization structural chain.

### [2026-09-21 packed-gate-up-fsdp2-smoke]
- A fresh 8-GPU Qwen3-8B FSDP2 `full_shard` smoke exercised the opt-in
  `--mxfp4-pack-gate-up` path for three optimizer updates with sequence length
  8192, MBS2, GA8, BF16 reduction, retained accumulated parameters, no
  activation checkpointing, and five BF16 tail layers. Runtime reported 217
  quantized linears and 31 packed Qwen3 MLPs before FSDP2 wrapping.
- The run completed with `torchrun=0`, postflight/provenance status 0, and an
  unchanged source-bundle digest. Steps 2 and 3 measured `6206.3/5994.5 ms`;
  these two samples are smoke evidence only, not an accepted throughput
  estimate. Losses `12.7898/12.7974/12.8033`, gradient norms
  `5.781/5.719/5.812`, and final validation loss `12.7892` were finite.
- No packed cache/runtime fallback, OOM, NaN, skipped update, or kernel failure
  was logged. The 40 `packed gate/up disabled` warnings are exactly the five
  intentionally BF16 tail MLPs on eight ranks, rather than failures in the 31
  quantized prefix MLPs. The known post-success `torch.library._del_library`
  teardown traceback remains.
- Artifact:
  `/home/xdai/profile-results/lumen-mxfp4-packed-gate-up-e2e-fresh-20260921-v1/smoke_candidate_v2/`.
- Status: FSDP2 smoke passed. A longer fresh paired control/candidate campaign
  is still required before accepting performance or short-run accuracy.

### [2026-09-21 current-production-wrapper-packed-swiglu-smoke]
- Scope: fresh eight-GPU smoke under
  `/home/xdai/profile-results/lumen-mxfp4-packed-swiglu-smoke-current-fresh-20260921-1t7Vjn/`
  after the benchmark was corrected to call the current production
  `lumen.ops.fused_swiglu.packed_swiglu` wrapper. The run used Qwen3-8B,
  sequence length 8192, MBS2/GA8, FSDP2 `full_shard`, BF16 reduction,
  retained accumulated parameters, no activation checkpointing, five BF16
  tail layers, eight validation batches, and a cache absent at preflight.
- Functional routing passed: runtime reported 217 quantized linears, 36 BF16
  skips, and 31 packed Qwen3 MLPs. Exactly 40 `packed gate/up disabled`
  warnings were emitted, equal to five intentionally BF16 tail MLPs on each
  of eight ranks. The fresh Triton cache contains the direct execution
  artifacts `fused_silu_mul_kernel.hsaco`, `_swiglu_bwd_kernel.hsaco`, and two
  `_dynamic_mxfp4_quant_blockscale_kernel.hsaco` specializations.
- All three updates completed with finite loss/gradient norm:
  `12.7898/5.781`, `12.7974/5.719`, and `12.8049/5.812`; final validation NLL
  was `12.7897`. Step times were `61029.1`, `6414.8`, and `6735.5 ms`; the
  first includes fresh compilation/autotuning, and the remaining two are
  smoke samples only, not throughput evidence. Peak allocation was
  `125.2 GiB/GPU` after startup.
- Integrity passed: `torchrun`, `tee`, ROCm-SMI, post-run idle, and provenance
  statuses were all zero; KFD was idle before, immediately before launch, and
  after; the source bundle remained
  `991d1fbccd6308cb8a1e183feb5d78e6163beb8cc767608bf70e4667a7d16b3e`;
  all recorded Lumen/AITER tree/runtime hashes were stable; and the fresh
  autotune cache finished at
  `027eda2d1350e09d9e28ae11db3758d54eb61f1f2ab8c3d14e0b70a952a32880`.
  Pairing and validation evidence each covered unique ranks 0--7 with eight
  microbatches/batches per rank.
- No traceback, OOM, NaN/Inf, runtime fallback, or skipped update occurred
  before `Training complete`. Sixteen post-success tracebacks, exactly two per
  rank, were the known `torch.library._del_library` teardown defect. AITER's
  `module_mha_fwd` probe warning does not establish a new CK compile: the
  actually imported `module_fmha_v3_fwd/bwd` binaries predated this run.
- Status: production-wrapper FSDP2 smoke passed. Build a fresh union autotune
  cache covering the control-only and packed-only shapes, prove two replay
  passes leave it unchanged, then run a 30-step control/candidate/control
  campaign before crediting any step-speed or short-run numerical benefit.

### [2026-09-21 packed-gate-up-formal-aba-rejection]
- Scope: fresh formal control/candidate/control campaign under
  `/home/xdai/profile-results/lumen-mxfp4-packed-gate-up-formal-current-fresh-20260921-MjAFbw/`.
  The three 30-step arms used Qwen3-8B, 8 x MI350X, sequence length 8192,
  MBS2, GBS128/GA8, FSDP2 `full_shard`, retained accumulated parameters, no
  activation checkpointing, BF16 reduction, five BF16 tail layers, and 16
  validation batches. The only intended algorithmic delta was
  `--mxfp4-pack-gate-up` in the candidate.
- Cache attribution passed. A fresh control build produced nine decisions;
  the candidate build extended the same namespace to exactly twelve control
  plus packed shapes. Control and candidate replay both left the union cache
  unchanged at SHA256
  `b85f60c3bb2e83a3fe130b15a054007ab86846009f177f9ab713f565453d5ae6`.
  Independent copies with that same hash were used by all three formal arms.
- Routing/integrity passed for this run: all arms completed 30/30 updates with
  torchrun/tee/postflight/provenance/KFD statuses zero; source, tree, runtime,
  and cache hashes stayed frozen; model initialization and all rank-local
  train/validation batch digests matched. Candidate routing reported 31 packed
  MLPs and exactly 40 expected BF16-tail warnings. The packed Triton artifacts
  were present. Only the allowlisted post-success `torch.library` teardown
  tracebacks occurred.
- Steps 11--30: control A1 mean/median `6065.805/5933.400 ms`, candidate B
  `5942.625/5689.750 ms`, and control A2 `5947.675/5889.600 ms`. Against the
  per-step A1/A2 midpoint, B saved `64.115 ms` by mean and `209.125 ms` by
  median, for point estimates `1.010789x/1.039022x`, but won only `12/20`
  paired steps. The block-4 circular moving-block bootstrap 95% speedup CI was
  `[0.947550x,1.077913x]` and the saving CI was
  `[-326.928,441.855] ms`; both include regression/no effect. Control mean
  drift was `1.967%`, below the 3% gate, while candidate timing stdev was a
  high `526.6 ms` under enabled NUMA balancing.
- Numerical screen passed but does not rescue the speed decision. Validation
  NLL was `9.1303/9.1460/9.1383` for A1/B/A2, so B was `+0.0117` versus the
  control midpoint, below the predeclared `+0.03` limit. Peak allocation fell
  from `138.9` to `125.2 GiB/GPU`. Training losses and gradient norms were
  finite, but the analyzer's NLL gate is only a short-run one-sided
  non-regression check and is not a convergence claim.
- Decision: reject packed gate/up as a step-speed optimization under the
  preregistered gate. Do not credit its favorable median or memory reduction
  toward the 1.6x target. Keep the implementation opt-in/default-off while it
  remains useful for controlled memory research; do not enable it in the
  production recipe based on this campaign.
- Harness audit for future campaigns: the saved runner's status-22 KFD retry
  reads `/proc/<pid>/comm` with a broken Bash redirection expression; fallback
  detection only matches `falling back`; the analyzer does not parse the four
  union-build/replay cases; and packed-warning counting has an ordering blind
  spot. These defects did not cause this rejection because every v2 arm had a
  direct clean postflight, no fallback text was present, union hashes/shapes
  were checked independently, and the 40 warnings all preceded the enable
  report. Do not edit the frozen artifact; fix these checks in the next copied
  harness.
- Status: resolved negative result. Continue from a new fresh profile or a
  higher-Amdahl structural candidate; the same-policy 1.6x objective remains
  open.

### [2026-09-21 fresh-post-packed-profile-and-qkv-screen]
- Scope: fresh eight-GPU BF16/MXFP4 profiling after packed gate/up was formally
  rejected. The campaign root is
  `/home/xdai/profile-results/lumen-mxfp4-post-packed-profile-fresh-20260921-Q8LeDT/`;
  it built an absent MXFP4 autotune cache, replayed it byte-identically, then
  profiled optimizer steps 7--8 with packed gate/up off and the accepted
  `(12288,4096,16384)` WGrad ASM route on.
- An independent post-campaign audit reads the raw logs/traces without importing
  the original analyzer and passes 17/17 checks. It verifies exact stage order,
  all-zero case statuses, idle KFD boundaries, nine exact cache shapes, cache
  byte identity, full WGrad symbol/tile/split/manifest/code-object identity,
  paired commands, source/tree/workload/runtime hashes, matching initialization
  and rank-local train/validation digests, finite execution, and only the known
  post-success `torch.library._del_library` tracebacks. Profiler start -> step 7
  -> step 8 -> stop and 16 DataLoader/two optimizer annotations are exact;
  trace-vs-log span error is `2.378 ms` BF16 and `1.563 ms` MXFP4, below the
  tightened `3 ms` gate.
- Direct raw-trace interval analysis gives BF16/MXFP4 profiler spans
  `8620.489/6127.332 ms/step`, GPU busy unions `8468.491/5651.919 ms/step`,
  and a diagnostic speedup of `1.406891x`. The profile-derived `1.6x` target is
  `5387.806 ms/step`, leaving `739.526 ms/step` (`12.069%`) to remove. MXFP4
  A4W4 plus quant/layout has a `1548.463 ms/step` overlap-safe union; MXFP4
  also exposes `322.359 ms/step` more GPU idle than BF16.
- Packed-QKV attribution validates all `3,472` quantized-linear forward and
  backward events against Qwen's repeated seven-projection topology and uses
  forward sequence numbers to identify backward consumers. Current QKV work
  has a topology-mapped overlap-safe union of `324.974 ms/step`. A deliberately
  order-independent shape-only bound is `187.363--464.177 ms/step`, with a
  `325.770 ms/step` Q/O-symmetry point. These values cover only
  `25.34--62.77%` of the `739.526 ms` gap (`44.05%` at the symmetry point).
  They are impossible zero-cost-replacement ceilings, not predicted savings:
  packed forward/dgrad/wgrad, shared quantization, output splits, and gradient
  concatenation still have to execute.
- Structural launch accounting is exact: 31 quantized layers x 8 microbatches
  gives 248 calls per individual projection and phase. Packing Q/K/V removes
  496 forward GEMMs, 496 dgrad GEMMs, 496 wgrad GEMMs, and 496 forward input
  quantizations per update, replacing them with shapes
  `(16384,6144,4096)`, `(16384,4096,6144)`, and `(6144,4096,16384)`.
- The intrusive eight-step profiler validation NLL was `12.3836` BF16 and
  `12.7678` MXFP4. It is recorded only as finite-run evidence and must not be
  used for quality acceptance; every candidate still requires a fresh matched
  unprofiled accuracy gate.
- Artifacts: `post_audit_v2.py/.json/.md`,
  `analyze_packed_qkv_chain.py`, and `packed_qkv_chain_raw_trace.json/.md` in
  the campaign root. Raw traces and logs were not modified.
- Status: profile integrity accepted for diagnostic use. Packed QKV cannot
  close the target alone but its measured current-work ceiling is material
  enough to justify exact packed-shape correctness and microbenchmarking before
  any production integration.

### [2026-09-21 packed-qkv-full-chain-smoke-gate-correction]
- A new exact-shape single-GPU packed-QKV full-chain benchmark was added at
  `benchmarks/bench_mxfp4_qkv.py`. Its control calls the three production
  Lumen-patched MXFP4 Q/K/V linears; its prototype candidate keeps the same
  three source Parameters and uses one `_mxfp4_forward_core` /
  `_mxfp4_backward_core` chain. The candidate includes output splitting,
  combined-gradient assembly, separate Q/K/V dW returns, and a conservative
  BF16-cat weight-cache build that is outside the microbatch hot path.
- The first fresh smoke under the exclusive GPU lock stopped at the
  correctness gate before producing a performance result. The production
  control itself measured output SNR `15.3221--15.3264 dB`, dX SNR
  `13.7714 dB`, and Q/K/V dW SNR `5.7415--5.7543 dB` versus BF16 at exact
  `(M,H,Q,KV)=(16384,4096,4096,1024)`. Thus the inherited small-shape
  `10 dB` dW threshold incorrectly rejects the trusted current MXFP4 path.
- The benchmark gate is corrected symmetrically: both arms retain absolute
  floors of 12 dB output, 10 dB dX, and 5 dB dW, and the packed arm may not
  regress any same-seed metric by more than 0.5 dB versus the production
  control. No candidate-specific tolerance was introduced.
- Failed smoke artifact:
  `/home/xdai/profile-results/lumen-mxfp4-packed-qkv-chain-smoke-fresh-20260921-3FPEDr/`.
  It is correctness-gate evidence only and contains no accepted timing result.
- Status: benchmark debugging progressed; rerun the corrected smoke in a new
  directory before any packed-QKV performance claim.

### [2026-09-21 packed-qkv-full-chain-formal-acceptance]
- Scope: only the new exact-shape campaign under
  `/home/xdai/profile-results/lumen-mxfp4-packed-qkv-chain-formal-fresh-20260921-v9dKs1/`
  is used. It ran one fallback-chain diagnostic and eight independent fast
  processes, four initial AB and four initial BA, with 128 hot pairs, 32
  batched pairs at 16 repeats, 32 direct GA8 update pairs, and three
  stochastic-rounding correctness seeds per process.
- Integrity passed every fail-closed gate. All nine processes exited zero;
  Lumen/AITER heads and source fingerprint
  `4fe4da0498077eb3db456dd4bc3d850c6f8128ce833b1345be2968982fefe0d6`
  were stable; all executed benchmark copies matched SHA256
  `0b265ebbb02d46447d688ac2adf006355467913453998b7178b7ac6eda94fca4`;
  the three tuned-table hashes, autotune identity, eight native shape keys,
  and selected backend vectors matched across processes. The fallback process
  observed every selected backend and never used `dequant_bf16`.
- Correctness passed across all 27 seed records. Global minimum SNR was
  `15.317965 dB` for outputs, `13.766171 dB` for dX, and `14.273864 dB` for
  dW, above the 12/10/5 dB floors. Worst packed-versus-control same-seed SNR
  regression was only `0.006645 dB`, below the 0.5 dB limit. Every packed dW
  was contiguous and every noncontiguous downstream-gradient check was finite.
- Memory passed: each fast process measured exactly `22 MiB` more packed
  incremental peak and zero post-call resident increase versus control, below
  the preregistered 64 MiB limit.
- Primary inference used 100,000 fixed-seed replicates with AB/BA-stratified
  process resampling and circular block length four within each process. Hot
  full forward+backward was `7.218542 -> 3.051595 ms`, saving `4.166947 ms`
  at `2.365498x`; saving CI was `[4.044078,4.286438] ms` and speedup CI was
  `[2.296396x,2.428698x]`. All 8/8 process means and both strata improved.
- Batched confirmation was `6.505154 -> 2.500725 ms`, `2.601307x`, with
  saving CI `[3.959440,4.059116] ms` and speedup CI
  `[2.544654x,2.659401x]`. Direct one-layer GA8 update timing was
  `53.027225 -> 20.310072 ms`, `2.610883x`, with saving CI
  `[32.187495,33.171478] ms` and speedup CI
  `[2.562609x,2.643952x]`. Block-8 and process-only sensitivity analyses keep
  the same conclusion.
- Decision: accept packed QKV as an exact-shape single-GPU full-chain
  microbenchmark candidate. Do not credit the benchmark's 31-layer projection
  as step speed: production cache/autograd/Qwen integration, FSDP2 tests, an
  eight-GPU smoke, and a fresh symmetric unprofiled training campaign remain
  required.
- Reproducible analyzer and reports are `analyze_campaign.py`, `analysis.json`,
  and `analysis.md` in the campaign root. Analyzer SHA256 is
  `bdb63ac834b6af75ba819694bd2130f6a58f4bc97ba2933119b8e877962fa570`.
- Status: microbenchmark gate passed; begin guarded opt-in production
  integration while preserving the three original Q/K/V Parameters.

### [2026-09-21 packed-qkv-production-fsdp2-smoke]
- Scope: fresh eight-GPU Qwen3-8B FSDP2 smoke under
  `/home/xdai/profile-results/lumen-mxfp4-packed-qkv-e2e-smoke-fresh-20260921-ZEHwxA/`.
  It used sequence length 8192, MBS2, GBS128/GA8, `full_shard`, retained
  accumulated parameters, no activation checkpointing, BF16 reduction, five
  BF16 tail layers, eight validation batches, and only the additional
  `--mxfp4-pack-qkv` flag. The run held the global GPU lock and was pinned to
  NUMA node 0; `/dev/kfd` was idle at initial, prelaunch, and post-run checks.
- Current-source gates passed before launch: `py_compile` succeeded; the
  focused Qwen3 integration/CLI suite passed 9 tests; and the one-GPU packed
  QKV/gate-up op subset passed 5 tests. Both pytest processes emitted only the
  known successful-exit `torch.library._del_library` cleanup traceback.
- Runtime routing was exact: quantization enabled 217 linears and skipped 36;
  packed QKV enabled on 31 attention layers. Exactly 40 packed-QKV disabled
  warnings were emitted, all with reason `all Q/K/V projections must already
  be quantized`, matching five intentionally BF16 tail layers on eight ranks.
  No other fallback text was present.
- All 3/3 optimizer updates completed. Step 1 included fresh autotuning and
  took `59895.1 ms`; smoke-only post-startup steps 2/3 were
  `5760.0/5579.1 ms`. Losses were `12.7898/12.7974/12.8014`, gradient norms
  `5.781/5.719/5.781`, final eight-batch validation NLL was `12.7888`, and
  peak allocated memory was `134.7 GiB/GPU`. These two timing samples are not
  accepted throughput evidence.
- Integrity passed: torchrun, tee, ROCm-SMI, post-idle, and provenance statuses
  were all zero; source bundle before/after was byte-identical at
  `0f3601c8c5ad7df87183c5aba82a915bf94f44e5a15125051da0ebf884365c34`;
  all eight rank-local step-1 and validation digests were present; and the
  fresh candidate cache SHA256 is
  `c607f48ca2c7c800c768ce7bb919fc81e7e2edbf9f76de8f82ec4e30bb126caf`.
  The cache contains the three packed shapes `(16384,6144,4096)`,
  `(16384,4096,6144)`, and `(6144,4096,16384)`, all selecting protected native
  ASM paths, and six remaining production shapes. Sixteen post-success
  tracebacks, exactly two per rank, are the known `torch.library` cleanup
  defect; no training-time runtime error, OOM, NaN/Inf, skipped update, or
  unexpected fallback was found.
- Decision: the production packed-QKV integration passes the real FSDP2 smoke
  gate. Do not credit the two post-startup timings as speedup. Next build a
  fresh union cache by exercising both unpacked control and packed candidate,
  prove control and candidate replay leave it byte-identical, then run the
  predeclared 30-step control/candidate/control campaign with 16 validation
  batches and block-4 bootstrap analysis.
- Status: smoke resolved; formal end-to-end performance and short-run accuracy
  attribution remain open.

### [2026-09-21 packed-qkv-union-cache-replay]
- Question: can unpacked control and packed-QKV candidate be measured against
  one immutable current-source autotune cache, without online profiling or
  backend-choice asymmetry?
- Fresh artifact:
  `/home/xdai/profile-results/lumen-mxfp4-packed-qkv-formal-fresh-20260921-5d0u3R/`.
- The control fresh-build produced nine choices (SHA256 `b802b524...`). The
  packed arm loaded those nine and added exactly the three packed F/D/W shapes,
  producing a 12-choice union cache at SHA256
  `811419593463317ace7d141fc6a055873b9f61b504515e6061338c6a4a3334e3`.
- Control replay and packed replay both loaded all 12 decisions, emitted no
  online-autotune event, and preserved the cache byte-for-byte. All four arms
  reported torchrun/tee/ROCm/post-idle/provenance status 0 and idle KFD
  boundaries; the frozen source bundle was `2cad461d...` throughout.
- Cache identity: schema 6, gfx950, backend fingerprint `dd7fa0173867fffa`,
  tuned-table fingerprint `f970d344c7dec601`, nine ASM and three AITER Triton
  shuffled choices. Packed `(16384,6144,4096)`, `(16384,4096,6144)`, and
  `(6144,4096,16384)` all select native ASM.
- `decision_scope=single_device` is not a consensus defect for this cache:
  current multi-device unanimous evidence is required only for named FlyDSL
  promotions; protected ASM and exact `plain`/`shuffled` identities follow
  their dedicated replay rules. No consensus fields were rewritten.
- Packed build/replay each routed 31 quantized attention layers and emitted the
  exact 40 expected BF16-tail warnings. The known `torch.library._del_library`
  traces occur only after successful completion.
- Status: resolved. Use the exact union SHA for every formal arm. Run a separate
  untimed per-rank shape probe, then a node-0-pinned 30-step A/B/A with steps
  11-30, block-4 bootstrap, `<3%` control drift, and validation delta-NLL gate.
- Detailed report:
  `/home/xdai/profile-results/lumen-mxfp4-packed-qkv-formal-fresh-20260921-5d0u3R/union_cache_analysis.md`.

### [2026-09-21 packed-qkv-route-probe-instrumentation-miss]
- The first fresh untimed route-probe attempt completed both three-step,
  eight-GPU arms successfully and replayed the immutable 12-shape cache, but
  the analyzer stopped fail-closed because none of the expected per-rank
  `mxfp4-shapes-rank*.csv` files existed.
- The runner recorded `shape_log_enabled=1`, and the rank-aware entrypoint was
  used, but relying on the module's interpreter-exit hook did not produce the
  files. No route or performance conclusion is accepted from this attempt.
- Preserve `route_control/`, `route_candidate/`, and the nonzero
  `route-probes-exit-status.txt` as failed-instrumentation evidence. The next
  attempt must use new arm names and make the rank entrypoint explicitly set
  the logger's module path, flush it in `finally`, and fail the rank if the CSV
  was not written.
- Status: open harness defect; repair and rerun the route probe before any
  formal A/B/A timing.

### [2026-09-21 packed-qkv-route-probe-v2-preflight-false-positive]
- The repaired second route-probe attempt did not launch `torchrun` or touch
  the GPUs. Its control-arm prelaunch KFD check classified PID 2506629 as a
  known GPU workload because that short-lived source-review shell command
  contained the literal strings `rank_entry.py` and `train_qwen3_fsdp.py`.
- Evidence is preserved in `route_control_v2/kfd-prelaunch.txt`; the run has no
  `train.log`, `train-exit-status.txt`, or postflight artifact, and the driver
  recorded `route-v2-probes-exit-status.txt=1`.
- This is a conservative harness false positive, not a training or kernel
  failure. Do not overwrite the partial v2 directory. The next attempt uses
  isolated `route_control_v3` / `route_candidate_v3` names and v3 shared/status
  files, after the source-review process has exited and the route source bundle
  is recomputed.
- Status: resolved as a prelaunch-only attempt; v3 route evidence remains open.

### [2026-09-21 packed-qkv-route-probe-v3-pass]
- Scope: fresh untimed eight-GPU route probes under
  `/home/xdai/profile-results/lumen-mxfp4-packed-qkv-formal-fresh-20260921-5d0u3R/`
  using isolated `route_control_v3` and `route_candidate_v3` arms. Shape
  logging was intentionally enabled, so their step times are diagnostic only
  and are not accepted throughput evidence.
- Integrity passed `40/40` fail-closed checks. Both arms completed 3/3 updates,
  emitted finite loss/gradient/validation values, matched the same model-init
  and all rank-local train/validation digests, returned zero train/postflight
  statuses, and had idle KFD boundaries. Only the known post-success
  `torch.library._del_library` teardown tracebacks occurred.
- All sixteen per-rank shape CSVs were written. Every control rank issued the
  exact nine unpacked shape/count records. Every candidate rank removed the
  three old KV shapes, halved the shared Q-width calls, and issued exactly the
  packed forward/dgrad/wgrad shapes `(16384,6144,4096)`,
  `(16384,4096,6144)`, and `(6144,4096,16384)` with their expected counts.
- Both arms loaded the same immutable 12-choice cache at SHA256
  `811419593463317ace7d141fc6a055873b9f61b504515e6061338c6a4a3334e3`
  without online autotuning or cache mutation. All three packed shapes selected
  the preregistered native ASM symbols, tiles, split-K values, manifests, and
  code objects.
- Control validation NLL was `12.7869`; candidate was `12.7874`. This tiny
  three-step difference is only a finite-route smoke signal. Accuracy and
  speed decisions remain gated on the 30-step symmetric unprofiled A/B/A.
- Report: `route_probe_analysis.json` / `route_probe_analysis.md`; source bundle
  SHA256 `a8dfeff49c4e90808431e2338b3c845aa45ffb185d953f5be02f74499aa5c578`.
- Status: route gate passed. Freeze the formal source bundle and run the
  30-step control/candidate/control campaign with shape logging disabled.

### [2026-09-21 packed-qkv-formal-aba-accept]
- Scope: fresh eight-GPU, node-0-pinned, unprofiled control/candidate/control
  campaign under
  `/home/xdai/profile-results/lumen-mxfp4-packed-qkv-formal-fresh-20260921-5d0u3R/`.
  The sole measured-arm difference was `--mxfp4-pack-qkv`; shape logging was
  disabled and all arms replayed the immutable 12-choice cache at SHA256
  `811419593463317ace7d141fc6a055873b9f61b504515e6061338c6a4a3334e3`.
- Integrity passed `54/54` gates. All arms completed 30/30 updates with zero
  torchrun, tee, ROCm-SMI, post-idle, and provenance statuses; KFD was idle at
  every arm boundary; source bundle
  `c545d12f0c22a53675e49ca436822cab9ac2508f4468cc11f5a236b39ae3dba9`,
  cache, repository/runtime snapshots, initialization hash, and all rank-local
  training/validation batch digests were paired and frozen. The candidate
  enabled 31 packed attention layers, emitted exactly the expected 40 BF16-tail
  warnings, and the prior 40/40 route probe proved the three packed shapes used
  their preregistered native ASM paths. Only the known post-success
  `torch.library._del_library` teardown tracebacks occurred.
- Steps 11--30 independently reparse to control A1 mean/median
  `5823.630/5723.000 ms`, packed candidate `5626.865/5575.150 ms`, and control
  A2 `5824.105/5723.100 ms`. Against the same-step A1/A2 midpoint, packed QKV
  saved `197.003 ms/step` by mean and `147.375 ms/step` by median, for
  `1.035011x/1.026780x`; it won `17/20` paired positions. Control mean/median
  drift was only `0.0082%/0.0017%`, below the preregistered 3% gates.
- The fixed-seed 100,000-resample paired circular moving-block bootstrap with
  block length four reported a speedup 95% CI of
  `[1.013824x,1.056741x]` and a saving CI of
  `[78.465,317.030] ms/step`, both excluding no benefit. An independent raw-log
  reproduction matched both intervals exactly; block lengths one through six
  also kept the speedup lower bound above one. Because there is only one
  candidate process, this inference covers the paired step sequence rather
  than arbitrary fresh-process variance.
- The 16-batch validation NLLs were A1 `9.1268`, candidate `9.1243`, and A2
  `9.1714`; candidate delta versus the control midpoint was `-0.0248`, passing
  the preregistered one-sided `<= +0.03` screen. Peak allocation fell from
  `138.9` to `134.7 GiB/GPU`. This is a short-run numerical screen, not a
  convergence claim.
- Decision: accept packed QKV as an opt-in/default-off incremental MXFP4
  optimization and credit only its formally measured `147--197 ms/step`
  saving. Do not infer the final BF16/MXFP4 ratio by combining separate-run
  point estimates. Before making it a default or claiming progress to 1.6x,
  run checkpoint save/load/reshard validation, a paired `3 seeds x >=200
  steps` quality campaign, a fresh post-change profile, and a final matched
  BF16/MXFP4 timing comparison.
- Reports: `formal_aba_analysis.json`, `formal_aba_analysis.md`, and
  `formal-aba-exit-status.txt=0` in the campaign root.
- Status: formal incremental performance gate accepted; long-horizon quality
  and final same-policy 1.6x target remain open.

### [2026-09-21 packed-qkv-state-dict-cache-regression]
- Added a focused CPU regression for the packed-QKV cache lifecycle across a
  strict `state_dict` load. The test first builds the real mock Q/K/V packed
  cache, loads different Q/K/V values, and proves that the original three
  `Parameter` objects and checkpoint keys are preserved while no
  `_mxfp4_w_cache*` attribute enters the state dict.
- The load increments the live Q/K/V parameter versions without mutating the
  recorded old cache metadata. The next packed lookup therefore misses and
  rebuilds all three quantized sources; the mock quantizer call count rises
  from three to six, and the rebuilt cache records the loaded versions and the
  original Parameter identities.
- Root rerun results:
  `tests/quantize/test_mxfp4_weight_cache_hook.py -k 'qkv_state_dict_load or qkv_cache'`
  passed `3` tests, and
  `tests/models/test_qwen3_fsdp_pretrain.py -k 'mxfp4_qkv and (state_dict or checkpoint)'`
  passed `2` tests. `git diff --check` passed. Both pytest processes exited
  zero and emitted only the known interpreter-exit
  `torch.library._del_library` traceback.
- This closes ordinary state-dict/cache invalidation only. It does not prove
  FSDP2 distributed-checkpoint resharding; a real `2-rank save -> 8-rank load`
  DCP validation remains in progress before packed QKV can be defaulted.
- Status: ordinary checkpoint lifecycle passed; cross-world-size FSDP2 gate
  remains open.

### [2026-09-21 packed-qkv-fsdp2-dcp-stale-cache-fix]
- A real `2-rank save -> 8-rank load` FSDP2 distributed-checkpoint gate first
  reproduced a correctness defect: DCP restored gathered Q/K/V Parameter bytes
  in place while preserving their Python identity and `_version`. The packed
  QKV version/source/layout cache key therefore accepted FP4 bytes derived from
  the pre-load weights. The failed attempt is preserved as
  `attempt-2-stale-cache-failed` rather than overwritten.
- Production fix: `lumen/quantize/__init__.py` now provides one shared cache
  invalidator covering module-owned and Parameter-owned MXFP4 caches, plus an
  idempotent root `load_state_dict` post-hook. Registration is defensive at
  generic `quant.enable(MXFP4)`, direct packed-QKV enable, direct packed
  gate/up enable, and the optimizer-hook helper. Optimizer post-step and model
  post-load use the same invalidation implementation.
- Latest-source validation froze source digest
  `f977b1a508fac624ddce068b76651c0162786aed361dbe020759cd9a5c08b976`.
  Save, load, and overall statuses were all zero; source before/after/current
  matched; every KFD boundary was idle; both phases ran under the global GPU
  lock and `numactl --cpunodebind=0 --membind=0`.
- On all eight load ranks, every packed cache attribute was absent immediately
  after `set_state_dict`. The next packed forward rebuilt distinct cache/data
  objects with different pointers and data hashes from the deliberately stale
  pre-load cache. Packed routing stayed active and BF16 fallback calls/messages
  were zero.
- Q/K/V model hashes and AdamW `step`, `exp_avg`, and `exp_avg_sq` hashes
  matched between the 2-rank save and 8-rank load. Model and optimizer
  Parameter identities were preserved, optimizer step restored from `1` and
  advanced to `2`, and post-load forward/backward/AdamW completed.
- The checkpoint retained only original model/optimizer keys and was unchanged
  by load: metadata SHA256
  `82fc852d6069fc1684cc7aff33fd2b457070c4f313a964f741e0b37ed3bbe54e`,
  tree SHA256
  `b60f97b6c64d04dcd3e3791fe32abfe46c927205799b33a348dde2b1985e3580`.
  Reports are `save-report.json` SHA256
  `bcb2d7d1a60926028ac26abdd6e61d5d24f480960459cf05b37925e270d4a679`
  and `load-report.json` SHA256
  `8ff62f96a209afcdd89d4a047a957dc2712abf3b0e2d2698eff73dde80b2a428`
  under
  `/home/xdai/profile-results/lumen-qkv-fsdp2-dcp-reshard-fresh-20260921-1npYfh/`.
- Independent source and artifact reviews found no blocker for the next
  packed-QKV profile. A repository-level CPU/gloo FSDP2+DCP regression and
  direct native-Megatron entry coverage remain desirable before merge, but the
  external cross-world-size GPU gate already proves the production path used
  by this campaign.
- Root post-fix rerun on the current source passed: `py_compile` and
  `git diff --check` exited zero; cache/load/optimizer selection passed `7`
  tests; packed-QKV state/checkpoint selection passed `2` tests. Both pytest
  processes exited zero and emitted only the already-known interpreter-exit
  `torch.library._del_library` traceback.
- Status: resolved; cross-world-size checkpoint/cache lifecycle gate passed.

### [2026-09-21 packed-qkv-profile-supplemental-raw-audit]
- The frozen post-packed-QKV profile campaign remains formally failed:
  `campaign-exit-status.txt=1`, the original report remains `28/29`, and none
  of the original analyzer/report/checksum/status files were modified.
- A new independent streaming audit reparsed both raw traces without importing
  the original analyzer. BF16 versus packed MXFP4 profiler span was
  `8147.580700/5627.233340 ms/step`, GPU envelope
  `8144.675805/5622.878307`, busy union
  `7995.501713/5424.654942`, and idle `149.174092/198.223364`; every value
  matches the frozen JSON exactly. Raw trace SHA256 values remain
  `8282d6998ec504274dd05c956bf6b787de3a5842c5253f933eb25f884fa6b53b`
  BF16 and
  `5e5138fb2f0b752ccd96bcdb4763e2632f14740c80dec105997cb7f2d4a2606f`
  packed.
- The sole failed signature gate is an analyzer assumption, not missing work.
  The raw packed trace contains exactly `496` Q/K/V gradient `aten::cat`
  events (`248/step`), each nested in `SplitWithSizesBackward0`, assembling
  input shapes `[2,8192,4096]`, `[2,8192,1024]`, and `[2,8192,1024]` into
  width `6144`. Their child GPU cat-copy work is also exactly `496` events,
  `87.146260 ms` over two steps.
- Packed dW splitting is present as exactly `496` `aten::split_with_sizes`
  events on stride-`[4096,1]` `[6144,4096]` inputs. Splitting dimension zero
  produces already-contiguous Q/K/V row views, so the source's
  `part.contiguous()` calls are no-ops and correctly emit zero profiler events.
  Requiring nonzero `packed_dw_contiguous` was invalid.
- Supplemental artifacts are `profile_post_audit.py/.json/.md` under
  `/home/xdai/profile-results/lumen-mxfp4-packed-qkv-profile-fresh-20260921-prepared/`;
  SHA256 values are respectively
  `2f9e3e2204b09ae73204c7e89f31bcf6966e6f5babbf2b76397b7285ea99d32d`,
  `ca13a6f4d010b78952811ad4fcfc2cfedfb20366b6e0404b6e875ba961b8bb9d`,
  and
  `ca08b6a0d41158f5fbe9e598935be61fb9a95f790e687cdb4e94ad76a36555a9`.
- Decision: supplemental raw-trace execution integrity passes and no GPU rerun
  is needed for this analyzer defect. The original formal FAIL remains
  preserved; profiler timing is diagnostic only and is not throughput credit.

### [2026-09-21 packed-qkv-plus-split-swiglu-smoke]
- Scope: fresh three-step, eight-GPU smoke under
  `/home/xdai/profile-results/lumen-mxfp4-split-swiglu-fresh-20260921-192245/`
  for the incremental candidate `--mxfp4-pack-qkv --mxfp4-fuse-swiglu`.
  The run used the current Qwen3-8B FSDP2 full-shard recipe, GA8, retained
  accumulated parameters, no gradient checkpointing, BF16 reduction, and five
  BF16 tail layers.
- Routing evidence reported 31 MXFP4 MLPs using the split-SwiGLU integration
  and 31 attention layers using packed QKV. The warnings for the five protected
  BF16 tail layers were expected. All eight rank-local shape logs were written.
- Step 1 included JIT/startup and took `56083.5 ms`; smoke-only steps 2 and 3
  took `5946.2/5455.7 ms`. These two values are not throughput evidence and no
  speedup is credited from them.
- Loss remained finite (`12.7831 -> 12.7985`), gradient norm remained finite
  (`5.750e0 -> 5.812e0`), step-3 validation loss was `12.7845`, and peak
  allocation was `123.1 GiB/GPU`. No training-time OOM, NaN/Inf, skipped
  update, kernel failure, or unexpected fallback was found.
- Postflight passed: torchrun/tee/ROCm-SMI/post-idle/provenance statuses were
  all zero; KFD was idle before launch and after completion. The immutable
  12-choice union cache remained byte-identical at SHA256
  `811419593463317ace7d141fc6a055873b9f61b504515e6061338c6a4a3334e3`,
  and the frozen source bundle remained byte-identical at SHA256
  `e78cdd48d924dd835c666a8fc4b59f9ef6809fc9e5523d0c731300bffb0b6e75`.
- Decision: smoke gate passed. The next performance decision requires a fresh
  unprofiled control/candidate/control campaign comparing packed QKV alone
  against packed QKV plus split SwiGLU over at least steps 11--30, with paired
  data/provenance, bootstrap analysis, and validation-NLL checks.
- Status: smoke resolved; formal incremental performance and accuracy
  attribution remain open.

### [2026-09-21 vocabulary-quantization-evidence-audit]
- A read-only reparse confirmed the principal timing arithmetic in
  `/home/xdai/profile-results/lumen-mxfp4-vocab-hybrid-fresh-20260921-053214/`:
  BF16/FP8/hybrid `lm_head` medians were `17.705320/15.016805/29.650009 ms`,
  and BF16 versus FP8 input-embedding lookup was
  `0.0517005/0.225942--0.230303 ms`. The current implementations therefore
  remain rejected: the precision-repaired head is materially slower and the
  quantized embedding is about 4.4x slower.
- The raw-FP8 GA8 saving estimate reparses to `19.353 ms/update` after the
  assumed once-per-update `2.155123 ms` whole-weight quantization. Even with
  top-k selection/scatter treated as free, retaining the measured BF16
  selected-dot correction leaves only `5.932 ms/update`, or `0.620%` of the
  then-current `957.41 ms` target gap. These are optimistic forward-only
  estimates, not end-to-end credit.
- Precision caveat: the hybrid accuracy script covered only 589 tokens from
  three pretrained-checkpoint samples. Its aggregation averages per-sample
  p99 values rather than computing the pooled-token p99, and similarly averages
  non-linear SNR/max-absolute summaries. Therefore the reported top-8 static
  gate is only a weak feasibility signal, not a formal precision pass or a
  training-convergence guarantee.
- Additional evidence gaps are no CE/dX/dW/FSDP timing, no multi-process ABBA
  interval, and no demonstrated optimizer-step cache lifecycle in the actual
  sharded training path. These gaps cannot rescue a candidate that already
  loses its local performance gate, but they prohibit claiming the raw-FP8
  point estimate as a deployable speedup.
- Decision: keep input embedding and `lm_head` BF16. Do not advance the current
  raw FP8, top-8 correction, or gather-dequant embedding implementations to an
  eight-GPU campaign. A future, distinct memory-oriented investigation may
  evaluate exact vocabulary-chunk projection plus online CE/backward fusion,
  but it must start with a complete forward+CE+dX+dW benchmark and pooled-token
  numerical metrics.
- Status: resolved negative result; the earlier top-8 wording is downgraded
  from a precision pass to a static feasibility signal.

### [2026-09-21 packed-qkv-plus-split-swiglu-route-probe]
- A fresh untimed eight-GPU route probe completed under
  `/home/xdai/profile-results/lumen-mxfp4-split-swiglu-fresh-20260921-192245/`
  before the formal throughput campaign. The probe used the same Qwen3-8B
  FSDP2 full-shard, GA8, BF16-reduction, five-BF16-tail-layer candidate path
  as the preceding smoke run.
- All eight ranks recorded exactly `806` successful AITER split-SwiGLU
  forwards and `744` successful backwards, with zero AITER failures and zero
  eligible calls reaching the wrapped original MLP forward. Each rank reported
  `31` enabled/eligible MLPs and exactly five expected static warnings with
  reason `both projections must already be quantized` for the protected BF16
  tail layers.
- Training completed three finite steps (`5703.8/5448.9 ms` for smoke-only
  steps 2/3), with finite loss and gradient norm and validation loss `12.7853`.
  These timings remain smoke evidence only and are not throughput credit.
- Route analysis passed `105/105` integrity checks. Torchrun, tee, ROCm-SMI,
  post-idle, provenance, source-freeze, and cache-freeze checks passed; the
  source bundle was
  `68740943351c154eb3a871276dcbb38641fdd16ece421503233517ae30dc6295`
  and the immutable cache remained
  `811419593463317ace7d141fc6a055873b9f61b504515e6061338c6a4a3334e3`.
- Decision: runtime routing and autograd coverage pass. Advance to the frozen,
  unprofiled packed-QKV control / packed-QKV+split-SwiGLU candidate /
  packed-QKV control A/B/A campaign; do not infer a speedup from this probe.
- Status: route gate resolved; formal incremental performance and validation
  attribution remain open.

### [2026-09-21 packed-qkv-plus-split-swiglu-formal-aba]
- A fresh unprofiled eight-GPU A/B/A campaign compared packed QKV alone
  (`formal_control_a1`, `formal_control_a2`) against packed QKV plus
  split-SwiGLU (`formal_candidate_b`) under the frozen Qwen3-8B FSDP2
  full-shard, seq-8192, MBS2, GBS128/GA8, BF16-reduction recipe. The only
  intended command delta was `--mxfp4-fuse-swiglu`.
- Over the paired step-11--30 window, the control midpoint mean/median was
  `5654.060/5582.425 ms`; the candidate was `5534.220/5455.900 ms`.
  Mean/median speedups were `1.021654x/1.023190x`, with paired mean/median
  savings of `119.840/124.600 ms` and `19/20` paired wins.
- The paired circular moving-block bootstrap used block length 4 and 100,000
  resamples. Its 95% speedup interval was `[1.018325x, 1.024800x]`, and its
  saving interval was `[102.720, 135.918] ms`; both exclude no benefit.
- A1/B/A2 validation NLL was `9.1371/9.1273/9.1364`. Candidate delta against
  the control midpoint was `-0.00945`, so the short-run quality gate passed.
  All losses and gradient norms were finite.
- Control mean/median drift was only `0.133%/0.067%`. All `46/46` integrity
  checks passed, including paired model/data/validation evidence, intended
  command delta, routing, source/cache/tree stability, KFD/postflight, and
  clean torchrun/tee exit. The known post-success `torch.library` cleanup
  traceback remained and was correctly classified as teardown-only.
- Peak allocated memory was `134.7 GiB/GPU` in both controls and
  `123.1 GiB/GPU` in the candidate, an observed reduction of `11.6 GiB/GPU`.
- Artifact:
  `/home/xdai/profile-results/lumen-mxfp4-split-swiglu-fresh-20260921-192245/formal_aba_analysis.md`.
- Decision: accept split-SwiGLU as an incremental packed-QKV optimization.
  This is not by itself a fresh final BF16-versus-MXFP4 target claim; a final
  matched comparison is still required after the remaining candidate work.
- Status: resolved accepted incremental candidate.

### [2026-09-21 fresh-tail5-through-tail2-palindrome]
- Scope: fresh eight-GPU Stage-1 tail-reduction campaign under
  `/home/xdai/profile-results/lumen-mxfp4-tail-reduction-fresh-20260921-xxgdel/`.
  The fixed Qwen3-8B recipe used sequence length 8192, MBS2, GBS128/GA8,
  FSDP2 `full_shard`, retained accumulated parameters, no activation
  checkpointing, BF16 reduction, packed QKV, split SwiGLU, seed 1234, and 16
  validation batches.  `lm_head` remained independently protected in BF16.
- The formal order was
  `tail5_a1 -> tail4_b1 -> tail3_c1 -> tail2_d -> tail3_c2 -> tail4_b2 -> tail5_a2`.
  The original unified terminal was reclaimed at its approximately 25-minute
  lifetime boundary while the first `tail3_c2` process was still initializing,
  before step 1.  That no-sample directory is preserved as
  `interrupted-tail3_c2-session-timeout-20260921-220814/`.  A separately
  audited continuation reran the same label from scratch and completed the
  remaining palindrome suffix with the original driver, source bundle, cache,
  repository trees, runtime modules, and AITER artifacts unchanged.
- Integrity passed all `84/84` frozen analyzer checks.  Both standard and
  continuation exit statuses are zero.  Every formal arm completed 30/30
  updates, used identical model initialization and rank-local training and
  validation digests, loaded the same nine-decision cache at SHA256
  `c607f48ca2c7c800c768ce7bb919fc81e7e2edbf9f76de8f82ec4e30bb126caf`,
  and kept source bundle
  `6485456e71c7695c56556b2281d26125f80a0cd4bebb94ada58521aff48d86e9`
  frozen.  No training-time fallback, OOM, NaN/Inf, skipped update, or kernel
  failure occurred; only the known post-success `torch.library` teardown
  traceback was present.
- Step-11--30 midpoint timing was `5510.530/5461.675 ms` mean/median for
  tail-5, `5417.315/5382.575 ms` for tail-4, `5341.615/5290.500 ms` for
  tail-3, and `5272.025/5210.100 ms` for tail-2.  Relative to tail-5, tail-2
  saved `238.505/251.575 ms` and reached `1.045240x/1.048286x`, with 17/20
  paired wins and a block-4, 100,000-resample speedup CI of
  `[1.035046x,1.055322x]`.
- Every one-layer marginal passed independently: tail-5 -> tail-4 was
  `1.017207x` mean with CI `[1.010003x,1.024666x]`; tail-4 -> tail-3 was
  `1.014172x` with CI `[1.008550x,1.019597x]`; tail-3 -> tail-2 was
  `1.013200x` with CI `[1.007912x,1.018461x]`.  Each transition won 18/20
  paired positions.  Replicate mean drift was only 0.154% tail-5, 0.064%
  tail-4, and 0.118% tail-3, below the frozen 3% limit.
- Validation NLL was `9.13275` for the tail-5 bracket, `9.15050` tail-4,
  `9.15540` tail-3, and `9.15370` tail-2.  Tail-2 delta was `+0.02095` nats
  (approximately +2.117% perplexity), below the preregistered `+0.03` screen.
  Peak allocation fell from 123.1 GiB/GPU at tail-5 to 122.1 GiB/GPU at
  tail-2.  This remains a 30-step, one-seed screen rather than a convergence
  guarantee.
- Decision: select tail-2 for the next confirmation stage.  Continue with a
  fresh tail-2/tail-1/tail-0 boundary experiment while preserving BF16
  `lm_head`; any surviving candidate still requires an independent matched
  BF16 comparison and `3 seeds x >=200 steps` before default adoption or a
  training-quality claim.  The overall same-policy BF16/MXFP4 1.6x objective
  remains open.
- Analyzer artifacts are `stage1_analysis.json` SHA256
  `da8cab075017866cd39762118779a666254dff3bfb67eeeb4b3965208a966b9a`
  and `stage1_analysis.md` SHA256
  `ac6d6ebd5d3f66024ce82eeda2f0cd420d8a440151244921f2dc30aceb81883d`.
- Status: Stage-1 resolved; tail-2 accepted for boundary confirmation only.

### [2026-09-22 fresh-tail2-tail1-tail0-boundary]
- Scope: fresh eight-GPU Stage-2 boundary campaign under
  `/home/xdai/profile-results/lumen-mxfp4-tail-boundary-fresh-20260921-xIqY6u/`.
  The fixed Qwen3-8B recipe used sequence length 8192, MBS2, GBS128/GA8,
  FSDP2 `full_shard`, retained accumulated parameters, no activation
  checkpointing, BF16 reduction, packed QKV, split SwiGLU, seed 1234, and 16
  validation batches. The vocabulary `lm_head` remained independently
  protected in BF16 in every arm.
- The formal order was
  `tail2_a1 -> tail1_b1 -> tail0_c -> tail1_b2 -> tail2_a2`. An earlier
  phase-2 launch was terminated before `torchrun`; its preserved directory
  `interrupted-tail1_b2-launch-before-train-20260922-004639/` has no training
  log or step and is excluded. The later `tail1_b2` and `tail2_a2` arms
  completed normally.
- An independent parser that does not import the frozen analyzer passes all
  `98/98` integrity checks: every valid formal arm completed 30/30 updates,
  model/data/validation hashes match, source/cache/tree/runtime provenance is
  frozen, arm order and non-overlap pass, all values are finite, and no
  training-time fallback, OOM, NaN/Inf, skipped update, or kernel failure is
  present. Only the known post-success `torch.library._del_library` traceback
  occurs.
- Step-11--30 midpoint mean/median was `5268.785/5209.300 ms` for tail-2,
  `5184.065/5121.600 ms` for tail-1, and `5096.590/5044.600 ms` for tail-0.
  Tail-2 -> tail-1 reached `1.016342x/1.017124x`, saved
  `84.720/87.700 ms`, and won `17/20` paired positions, but its block-4,
  100,000-resample circular-bootstrap speedup CI was
  `[0.998298x,1.034793x]`; it fails only the predeclared requirement that the
  lower bound be strictly above one.
- Tail-1 -> tail-0 passed its marginal gate at
  `1.017163x/1.015264x`, `19/20` wins, and CI
  `[1.011516x,1.022699x]`. Tail-0 versus the tail-2 endpoint also passed at
  `1.033786x/1.032649x`, `18/20` wins, and CI
  `[1.017257x,1.050164x]`. Nevertheless, the frozen conservative selection
  chain requires tail-1 to pass first, so tail-0 cannot advance by skipping
  the failed intermediate gate.
- Validation NLL was `9.16175` tail-2, `9.15760` tail-1, and `9.16790`
  tail-0. Tail-1 delta versus tail-2 was `-0.00415`; tail-0 delta versus
  tail-2 was `+0.00615`, inside the `+0.01` screen. The non-gated tail-0
  versus tail-1 delta was `+0.01030`, reinforcing that tail-0 remains a
  provisional numerical risk despite passing the endpoint screen. Tail-2 and
  tail-1 replicate mean/median drift stayed below 0.3%, well inside the 3%
  limit.
- The frozen analyzer remains preserved as failed. Its `order_checks` uses
  `zip(ordered, ordered[1:], strict=True)` and the analogous `starts` call,
  which compare sequences of unequal length and always raise `ValueError`.
  It also rejects two expected post-completion rank-0 shape-log flush lines in
  the smoke arm. The original phase-2 status remains `1`, and no original
  official report or campaign-complete sentinel was fabricated.
- Supplemental v2 imports the frozen analyzer and narrowly overrides only the
  two faulty contracts. It exits zero with `98/98` checks and reaches the same
  no-selection decision as the fully independent parser. Artifacts are
  `analyze_stage2_supplemental_v2.py`,
  `stage2_analysis_supplemental_v2.json/.md`, and
  `independent_runtime_audit.py/.md` in the campaign root.
- Decision: retain tail-2. Do not launch the final BF16/tail-0/BF16 comparison
  or credit tail-0 toward the 1.6x goal yet. The next experiment is a fresh,
  higher-power 50-step
  `tail2_a1 -> tail1_b1 -> tail0_c -> tail1_b2 -> tail2_a2` palindrome with a
  40-step timing window. Repeating the full chain avoids pooling across
  campaigns or advancing tail-0 from a post-hoc prerequisite repair; only a
  fully passing preregistered repeat can unlock a separate fresh BF16
  comparison and multi-seed long-run quality validation.
- Status: Stage-2 resolved as inconclusive for further tail reduction; the
  overall same-policy BF16/MXFP4 1.6x objective remains open.

### [2026-09-22 fresh-tail0-bf16-final-confirmation]
- Scope: fresh eight-GPU `BF16 A1 -> MXFP4 tail0 -> BF16 A2` confirmation under
  `/home/xdai/profile-results/lumen-mxfp4-tail0-bf16-confirm-fresh-20260922-YTeWY7/`.
  The fixed Qwen3-8B recipe used sequence length 8192, MBS2, GBS128/GA8,
  FSDP2 `full_shard`, retained accumulated parameters, no activation
  checkpointing, BF16 reduction, packed QKV, split SwiGLU, seed 1234, 16
  validation batches, and no MXFP4 communication. The vocabulary `lm_head`
  remained the sole BF16 linear in the MXFP4 arm.
- Integrity passed all `235/235` frozen analyzer checks. An independent parser
  that did not import the analyzer also passed `166/166` runtime/integrity
  checks, all `134/134` source-receipt artifact hashes, and all `67/67`
  phase-1 manifest entries. Every formal arm completed 50/50 updates with
  matching model initialization and eight-rank training/validation digests,
  finite values, frozen source/cache/tree/runtime artifacts, and no training
  fallback, OOM, NaN/Inf, skipped update, or kernel failure. The only
  tracebacks were the known post-success `torch.library._del_library` defect.
- Step-11--50 BF16 A1/A2 mean and median were
  `8083.5275/8072.35 ms` and `8073.1775/8062.75 ms`. Their same-position
  midpoint was `8078.3525/8065.85 ms`; replicate drift was only
  `-0.1280%/-0.1189%`. MXFP4 tail0 was `5079.4900/5018.25 ms`, saving
  `2998.8625/3047.6000 ms` and winning all `40/40` paired positions.
- Tail0 nevertheless failed the preregistered performance gate: mean speedup
  was `1.590387x` (<1.6), median speedup was `1.607303x`, and the paired
  circular block-4, 100,000-resample speedup interval was
  `[1.576344x,1.603899x]`, whose lower bound is below 1.6.
- The short-run quality gate failed materially. BF16 A1/A2 validation NLL was
  `7.8764/7.8870`, midpoint `7.8817`; MXFP4 tail0 was `8.0262`, for
  `delta NLL +0.1445` (approximately +15.55% perplexity), above the `+0.01`
  limit. This one-process stochastic-MXFP4 result is not a convergence claim,
  but it cannot be waived under the frozen gate.
- The first phase-2 invocation failed before acquiring the GPU lock because
  the absolute-path unittest command is cwd-sensitive under Python 3.10. It
  produced no `bf16_a2` or KFD-prelaunch artifact. Its exact exit-status file
  was preserved outside the campaign, and the unchanged runner was restarted
  from the campaign root, where the same suite passed 20/20. The measured run
  and final phase-2 status then exited zero.
- Decision: reject tail0 as the default precision policy. Retain tail2 as the
  conservative working policy while creating a completely new BF16/tail2
  profiling campaign from the current source and a fresh cache. Do not reuse
  old trace numbers to choose the next optimization. Any later tail0 or
  projection-level protection candidate requires a new matched campaign and,
  before default adoption, `3 seeds x >=200 steps` with a one-sided validation
  delta-NLL upper confidence bound at or below `+0.01`.
- Status: resolved negative confirmation; same-policy 1.6x objective remains
  open.

### [2026-09-22 fresh-tail2-profile-attempt1-harness-gate]
- A new tail-2 BF16/MXFP4 profiling campaign was started under
  `/home/xdai/profile-results/lumen-mxfp4-tail2-profile-fresh-20260922-MA4B5n/`.
  Preflight and all 33 CPU/static tests passed. The three-step MXFP4 smoke
  completed with `torchrun=0`, finite loss/gradient/validation values, exact
  238-quantized/15-BF16 routing, all eight route/shape reports, idle KFD
  postflight, and frozen source/workload state.
- The harness then stopped fail-closed before cache replay or either formal
  profiler arm. Its new live-ASM verifier compared Lumen's persisted 16-hex
  SHA256 identity prefixes with full 64-hex file digests, so the correct cache
  identity `4e0dce9b9642d4fb` was rejected against the same manifest's full
  digest `4e0dce9b9642d4fb...`. This was a harness representation mismatch, not a
  kernel, training, routing, or cache-content failure. No performance result
  was produced or accepted.
- The complete failed attempt was preserved unchanged as
  `/home/xdai/profile-results/lumen-mxfp4-tail2-profile-fresh-20260922-MA4B5n-attempt1-manifest-prefix-failure/`.
  A fresh campaign root was created from only the six harness inputs. The live
  verifier now compares the exact 16-hex representation used by
  `mxfp4_asm._file_sha256`, while the independent full f4gemm-directory digest
  continues to freeze complete artifact bytes before and after every arm.
- Post-fix validation passed: all nine ASM cache identities matched the live
  manifest and code objects, 33/33 tests passed, Bash syntax and Ruff
  lint/format passed, and a `/tmp` dry-run passed. The fresh root contains no
  campaign output or bytecode/cache artifact.
- Status: attempt 1 invalid for profiling and preserved; corrected full
  campaign rerun pending.

### [2026-09-22 fresh-tail2-profile-rerun-and-analyzer-gate-audit]
- Scope: the corrected fresh eight-GPU campaign completed under
  `/home/xdai/profile-results/lumen-mxfp4-tail2-profile-fresh-20260922-MA4B5n/`
  using Qwen3-8B, sequence length 8192, MBS2, GBS128/GA8, FSDP2 full-shard,
  retained accumulated parameters, no activation checkpointing, BF16
  reduction, packed QKV, split SwiGLU, two protected BF16 transformer tail
  layers, and a BF16 `lm_head`. The smoke arm built a fresh nine-decision
  autotune cache; replay loaded exactly those nine decisions without online
  tuning. Both formal arms profiled steps 7--8 and used 16 validation batches.
- All four training arms and all postflight checks succeeded: every
  `torchrun`, `tee`, provenance, ROCm-SMI, and idle-KFD status was zero; source,
  workload, cache, runtime-module, tuned-table, live ASM manifest/symbol/tile,
  and f4gemm-directory identities remained frozen. The MXFP4 route was exactly
  238 quantized Linear modules, 15 BF16 Linear modules, 34 packed-QKV layers,
  and 34 split-SwiGLU layers. The known `torch.library._del_library` weakref
  traceback occurred only after `Training complete` and did not affect exit
  status.
- The frozen analyzer exited one after writing its complete JSON/Markdown
  report, but independent inspection proved both failed gates were analyzer
  defects rather than invalid measurements. First, it compared first-update
  hashes across smoke/replay and formal arms even though the stages used
  different sampler cardinalities (`train_samples=640` versus 1024). Within
  the formal pair, model-init SHA and all eight rank-local first-update hashes
  were identical; validation hashes were identical across every arm. Second,
  the analyzer compared two-step host-event totals against per-step values.
  Observed packed-QKV forward/backward counts were exactly 544 total / 272 per
  step, and ordinary quantized-Linear forward/backward counts were exactly
  2176 total / 1088 per step, matching the static route. Both traces parsed to
  EOF, passed their trace contracts, and resolved all events with an External
  ID; the original frozen report remains marked FAIL and is not rewritten.
- Fresh profiler diagnostics were 8139.274 ms/step BF16 versus 5401.819
  ms/step MXFP4-tail2, a profiler-only 1.507x ratio. GPU busy-union/envelope
  were 7986.135/8136.335 ms for BF16 and 5027.283/5396.644 ms for tail2; idle
  inside the GPU envelope was 150.200 versus 369.361 ms/step. Formal profile
  validation NLL was 12.3831 BF16 and 12.7741 tail2. These intrusive eight-step
  values are diagnostic only and are not used as an accuracy acceptance run.
- The fresh tail2 GPU decomposition per step was: A4W4 GEMM raw/union
  1200.515/1196.536 ms; MXFP4 quant/layout 351.020/350.694 ms; their joint
  overlap-safe union 1542.964 ms; residual BF16 `lm_head` 476.895/476.877 ms;
  the two protected tail layers 260.275/259.775 ms; attention
  1684.751/1683.177 ms; collectives 183.670/183.631 ms; norm
  228.014/224.968 ms; RoPE 75.167/74.223 ms; cross entropy
  53.725/53.721 ms; optimizer 53.580/53.355 ms; copy/memset
  69.636/69.618 ms; and unclassified work only 10.854/10.739 ms.
- A4W4 exact-shape attribution showed the largest rows were forward
  `(M,N,K)=(16384,12288,4096)` at 226.561 ms/step on ASM tile 128x512,
  wgrad `(12288,4096,16384)` at 203.198 ms/step on 128x512, and dgrad
  `(16384,4096,12288)` at 193.402 ms/step on 256x256. Packed-QKV forward,
  dgrad, and wgrad contributed 60.538, 51.094, and 48.905 ms/step for the
  6144-wide packed shapes; the packed wgrad selected 192x256. All selected
  rows used ASM, log2 K-split zero, the frozen manifest identity, and verified
  code objects.
- Quant/layout broke down to 116.682 ms/step activation dual-layout quant,
  221.988 ms gradient dual-layout quant, 5.127 ms packed transpose, 2.006 ms
  scale swizzle, and 5.217 ms conversion/helper work. Host runtime recorded
  23,137 launches/step and 41,574 `hipPointerGetAttribute` calls/step; their
  raw CPU times were 1056.360 and 35.856 ms/step, but these asynchronous host
  spans overlap GPU work and cannot be added to the device decomposition.
- The trace's collective producer name is the generic `record_param_comms`;
  the actual operation must be read from the event's `Collective name` arg.
  The frozen analyzer therefore mislabeled all 238 two-step calls as
  `collective_other`. Independent decoding gives 162 all-gathers, 74
  reduce-scatters, and two all-reduces across the two-step window.
- Fresh Amdahl conclusion: the two protected BF16 tail layers alone occupy
  about 259.8 ms/step, or about 129.9 ms per layer, while A4W4 plus
  quant/layout occupies 1543.0 ms/step. The next precision-policy experiment
  is therefore a new unprofiled tail2/tail1/tail2 bracket, followed by
  projection-level BF16 protection only if whole-layer tail1/tail0 cannot meet
  the quality gate. Kernel/microbenchmark tuning remains separate and no local
  timing improvements will be arithmetically added to predict E2E speed.
- Status: fresh traces accepted for diagnosis after independent correction of
  the two analyzer defects; final speed/quality acceptance remains pending and
  still requires fresh unprofiled symmetric measurement.

### [2026-09-22 fresh-tail2-tail1-tail0-high-power-confirmation]
- Scope: fresh eight-GPU 50-step palindrome under
  `/home/xdai/profile-results/lumen-mxfp4-tail210-confirm-fresh-20260922-vDKaSe/`.
  The order was `tail2_a1 -> tail1_b1 -> tail0_c -> tail1_b2 -> tail2_a2`;
  steps 11--50 supplied 40 timing points per arm. The Qwen3-8B recipe, source
  tree, nine-choice MXFP4 cache, initialization, rank-local training batches,
  and validation batches were frozen and paired. `lm_head` stayed BF16.
- Integrity passed all 106/106 analyzer checks. Every arm completed 50/50
  updates with finite values, zero train/postflight/KFD/provenance statuses,
  no online autotuning, fallback, OOM, NaN/Inf, skipped update, or kernel
  failure. Tail-2 and tail-1 replicate mean drift was 0.223% and 0.007%.
- Tail-2 midpoint mean/median was `5276.855/5192.075 ms`; tail-1 midpoint was
  `5179.141/5107.525 ms`; tail-0 was `5087.678/5020.950 ms`. Tail-2 -> tail-1
  saved `97.714 ms` by mean at `1.018867x`, 35/40 wins, and block-4 95%
  speedup CI `[1.001514x,1.037322x]`. Tail-1 -> tail-0 saved `91.464 ms` at
  `1.017978x`, 34/40 wins, CI `[1.011266x,1.024388x]`. The full tail-2 ->
  tail-0 change saved `189.178 ms` at `1.037183x`, 32/40 wins, CI
  `[1.019950x,1.055637x]`.
- Validation NLL was `7.93110` for the tail-2 midpoint, `7.92055` tail-1, and
  `7.93990` tail-0. Relative to tail-2, tail-1 was `-0.01055` and tail-0 was
  `+0.00880`, both inside the frozen `+0.01` endpoint screen. This is a
  one-seed 50-step screen, not a convergence claim.
- Reconciliation: the repeat now satisfies the prerequisite chain that the
  earlier lower-power boundary run missed, but it does not override the
  separate matched BF16/tail-0/BF16 rejection. That run measured only
  `1.590387x` by mean and a material BF16-relative validation delta NLL of
  `+0.1445`. Tail-0 therefore remains rejected; tail-1 is the aggressive
  whole-layer candidate for further precision-preserving optimization.
- Status: resolved. Continue with projection-level BF16 protection in the last
  layer and structural MXFP4 launch/materialization reductions; do not adopt
  tail-0 or claim 1.6x.

### [2026-09-22 last-layer-projection-guard-preparation]
- Fresh tail-2 profiling attributes `259.775 ms/step` of overlap-safe BF16 GEMM
  time to the two protected transformer layers, while tail-0 fails the matched
  BF16 quality gate. The next precision-policy experiment narrows protection
  inside only the final layer instead of changing optimizer/data settings.
- Added default-off CLI option `--mxfp4-last-layer-bf16-projections` to the
  Qwen3 FSDP trainer. It restores named final-layer projections to their
  original BF16 forward after generic MXFP4 patching and before packed-QKV,
  split-SwiGLU, or FSDP2 integration. Existing defaults and all arithmetic are
  unchanged when the option is absent. The option is restricted to MXFP4,
  full-parameter training, and `--num-layers-at-end-in-bf16 0`.
- Focused parser/restoration tests passed `2` tests with `83` deselected in
  `5.43s`; `py_compile` and `git diff --check` passed. The only later traceback
  was the known interpreter-exit `torch.library._del_library` defect. Ruff
  still reports two pre-existing E741 names in the already-modified training
  script; neither is in this change.
- Next check: an eight-GPU route smoke for final-layer `o_proj+down_proj` BF16,
  followed by a fresh symmetric screen against whole-layer tail-1 and a
  down-only variant. No speed or quality credit exists yet.
- Status: implementation prepared; GPU validation open.

### [2026-09-22 last-layer-projection-guard-smoke]
- The fresh eight-GPU route smoke completed under
  `/home/xdai/profile-results/lumen-mxfp4-projection-guard-smoke-fresh-20260922-f68RA4/`
  for tail-0 with final-layer `self_attn.o_proj` and `mlp.down_proj` restored
  to BF16. This was a three-step route/correctness smoke only; its step times
  are not performance evidence.
- All launch and postflight statuses were zero (`torchrun`, `tee`, ROCm-SMI,
  idle-KFD, and provenance). The source bundle was unchanged before/after at
  `6a9d96950ed4f8c6c6e9546b3502525f419800e5bd898aa55fda6a24378de40f`.
  Training completed 3/3 updates with finite losses `12.7920`, `12.7999`, and
  `12.8064`, finite grad norms `5.812e+00`, `5.719e+00`, and `5.812e+00`, and
  finite 16-batch validation loss `12.7890`.
- Every rank reported exactly the same unquantized Linear set:
  `model.layers.35.self_attn.o_proj`, `model.layers.35.mlp.down_proj`, and
  `lm_head`. `lm_head` remained BF16 and unquantized. All eight ranks enabled
  36 packed-QKV and 36 split-SwiGLU layers and recorded nonzero success counts
  (`qkv_linear_success=1440`, `swiglu_fwd_success=1440`,
  `swiglu_bwd_success=864`). All rank-local MXFP4 shape CSVs were byte
  identical and contained nine distinct shapes.
- No fallback, OOM, NaN/Inf, skipped update, or kernel failure occurred. The
  only tracebacks were the already-known post-success
  `torch.library._del_library` weakref cleanup defect after `Training
  complete`; process exit status remained zero.
- Decision: the projection guard is route-correct and safe to admit into a
  new unprofiled 50-step symmetric screen against whole-layer tail-1. Test
  `tail1 -> o_proj+down_proj -> down_proj-only -> o_proj+down_proj -> tail1`
  with steps 11--50, frozen source/data/cache, 16 validation batches, and the
  existing paired/block-bootstrap gates. Do not infer speed from this smoke.
- Status: resolved route/correctness smoke; E2E speed and accuracy screen open.

### [2026-09-22 projection-guard-review-hardening]
- A read-only integration review confirmed that final-layer `o_proj` and
  `down_proj` protection composes with the accepted packed-QKV and split-SwiGLU
  stack: the fused wrappers still call those output projections, and FSDP2
  MXFP4 communication ignores modules whose `_quant_enabled` flag is false.
- The review also found that the initial experimental CLI over-advertised
  Q/K/V and gate/up protection. The restore helper left
  `_quant_scaling_type=mxfp4` and other quantization metadata behind, while the
  strict packed/fused eligibility checks treat that metadata as an MXFP4
  marker. Those five choices could therefore fail startup or mix precision and
  structural policy unexpectedly. The helper also mutated modules before
  validating every requested target, so a later validation error could leave
  a partially restored model.
- Hardened the default-off experiment before any formal measurement. The CLI
  now exposes only the two residual-output projections required by this
  campaign (`o_proj`, `down_proj`). The helper resolves canonical final-layer
  targets, validates all of them first (including the MXFP4 scaling type), then
  restores them atomically. It removes quantization-only manager/backend/tensor
  metadata, derived FP8/MXFP4 caches on both module and Parameter owners, and
  stale frozen-weight tags while retaining `_quant_enabled=False` as explicit
  route state.
- Focused parser/restoration/cache/atomicity tests passed `3` tests with `83`
  deselected in `5.07s`; `py_compile` and `git diff --check` passed. Ruff still
  reports only the same two pre-existing E741 names in the already-modified
  trainer. The only post-success traceback was the known
  `torch.library._del_library` interpreter-exit defect.
- A separate redundant tail1-bracket preparation was stopped. Its latest
  smoke attempt failed during initialization with `LLVM ERROR: ... No space
  left on device`; it produced no accepted timing or accuracy result. The GPU
  lock is idle, and the failed attempt is retained as
  `/home/xdai/profile-results/lumen-mxfp4-tail1-bracket-fresh-20260922-WBIbMI-attempt4-disk-full/`.
  Current free space recovered to approximately 14 GiB after compiler
  temporary files were released; no user artifact was deleted.
- Decision: because the source changed after the earlier successful route
  smoke, the formal projection-policy campaign must start with a new fresh
  `o_proj+down_proj` route smoke and a new autotune cache. Do not reuse the
  earlier smoke cache or any interrupted tail1-bracket artifact.
- Status: helper hardened and CPU-tested; fresh GPU smoke/formal screen pending.
