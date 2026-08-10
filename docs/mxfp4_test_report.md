# MXFP4 test report

The single place MXFP4 test status is recorded. Anything about what is tested,
what passes, what is deliberately not tested and why, and what each release gate
is set to belongs here rather than in a new document. Append to the changelog at
the bottom on every update; do not silently rewrite earlier entries, since the
point of the file is being able to see when a gap opened or closed.

Performance numbers live in [`mxfp4_training_report.md`](mxfp4_training_report.md);
this file only carries the thresholds that gate a release.

## Running the suite

```bash
scripts/run_mxfp4_tests.sh          # one status line per group, logs in /tmp/mxfp4_tests
OUT=/tmp/x MEGATRON_ROOT=... scripts/run_mxfp4_tests.sh
```

Two environment facts that cost time when they are missed:

- `tests/models/` needs Megatron importable. The runner puts
  `/home/xdai/Megatron-LM` (`core_r0.15.0_rocm`) on `PYTHONPATH`; without it the
  module tests do not fail, they fail to *collect*, which is easy to read as
  "no MXFP4 tests here".
- Groups run as separate pytest processes. An abort inside an AITER JIT op takes
  the whole session down, and running everything in one process hides every
  result behind the first crash.

## Status — 2026-08-10, 8x MI350X (gfx950)

| Group | Tests | Result |
|---|---:|---|
| `test_quantize.py -k "mxfp4 or hadamard or unaligned or dividing"` | 105 | pass |
| `test_linear.py -k mxfp4` | 8 | pass |
| `test_mxfp4_dual_layout_shuffle.py` | 3 | pass |
| `test_mxfp4_fused_act_scale.py` | 2 | pass |
| `test_mxfp4_fwd_wgrad_operand.py` | 3 | pass |
| `test_mxfp4_weight_cache_hook.py` | 5 | pass |
| `test_mxfp4_weight_operand_fusion.py` | 2 | pass |
| `test_mxfp4_backward_optimization.py` | 5 | pass |
| `test_grad_quant.py` (whole file) | 14 | pass |
| `tests/models/ -k mxfp4` | 3 | pass |
| **Total** | **150** | **pass**, 84 s |

Known unrelated breakage on the same machine, present on `HEAD` with no local
changes, so not an MXFP4 regression:

- `tests/ops/test_linear.py` — 6 failures (`none`, `dynamic`, `per_token` FP8
  scaling at `M128_K4096_N4096` and `M256_K4096_N1024`), all from
  `lld invocation failed` while Triton builds the kernel.

## What MXFP4 needs from the FP8 CI matrix

The FP8 matrix is drawn over FP8's feature surface. MXFP4's surface is a
fraction of it, so most rows have no subject: `rmsnorm.py`, `attention.py`,
`grouped_linear.py` and `cross_entropy.py` contain zero MXFP4 references. Copying
those rows across produces tests that either skip or, worse, exercise the BF16
fallback and report green.

| FP8 matrix row | MXFP4 | Why |
|---|---|---|
| Op-level correctness | **Have it** | 105 cases, bit-exact against torchAO where a reference exists |
| Edge values | **Need it** | The one gap that has already cost us a real defect — see below |
| Quantization lifecycle | **Have it** | RTN, SR, 1D/2D tiles, dual layout, Hadamard fusion |
| FSDP + Megatron (2+ GPU) | **Need it** | Nothing multi-GPU exists; `tests/env/` has no MXFP4 |
| TP + SP | **Need it** | Never run with MXFP4; blocks the TP comm-GEMM overlap work |
| CP (A2A, P2P) | Not yet | MXFP4 has never run under CP; no subject to test |
| Comm overlap | Not yet | Same |
| Fixed-seed loss regression | **Need it** | Highest-value missing gate |
| Overflow monitoring | **Redefine** | Written for FP8 delayed scaling's amax history. MXFP4 is E8M0 per block with no history; the analogous metric is block scale saturation, not overflow rate |
| Determinism (same seed, identical output) | **Redefine** | Not achievable as written — see below |
| Perf matrix: 7B/70B x 5 scaling types | **Collapses** | MXFP4 is one recipe on one validated model. The axes that matter are TP width and whether the tuned GEMM table still covers the shapes |

Two things MXFP4 needs that the FP8 matrix does not contain at all:

- **GEMM backend dispatch.** Three AITER FP4 kernels, per-shape selection, and a
  tuned table that only applies on exact `(gfx, cu_num, M, N, K)` matches. FP8
  has no such structure. An untuned shape once returned 0.6 dB of garbage
  through AITER's unvalidated default kernel. Covered today by
  `test_mxfp4_backends_are_interchangeable` and the autotune tests.
- **torchAO as an external oracle.** FP8 has no equivalent; MXFP4 can be checked
  bit-for-bit against it, which is a much stronger statement than an SNR
  threshold. This is what caught the defect below.

### Determinism needs a different definition

The FP8 row asks that repeated runs with the same seed produce identical
results. For MXFP4 that gate is red by construction: WGrad uses stochastic
rounding, and `philox_seed` falls back to `random.randint()` off Python's global
RNG when the caller does not pin it. Runs are reproducible in where they land,
not in the bits they get there with.

So the gate has to be **noise-floor-aware** — measure the run-to-run spread
first, then set the threshold above it.

### Which part of the curve to read

Across two documented repeats of the current default (`ab_lumlinear`,
`ab_lumlinear2`, identical config, 200 steps):

| Metric | Spread between identical runs |
|---|---:|
| Loss at iteration 15 | 7.0% |
| Whole-trajectory mean per-step deviation | 0.33% |
| Final iteration alone | 0.104% |
| **Mean of the last 50 iterations** | **0.096%** |

The early curve is chaotic and a single final step is one sample, so both report
noise. The tail mean is the metric worth gating on. The same pair on the previous
linear path gives 0.056%, so the current path is noisier — the floor has to be
re-measured when the path changes rather than carried over.

### The floor, measured on five runs

Two runs give a range, not a distribution. Extending to five (2026-08-10,
`ab_lumlinear`, `ab_lumlinear2`, `noise1..3`, all `MBS=2 GBS=32 SEQ=8192 SEED=1234`
on 8 GPUs) widened the range from 0.096% to 0.223% with nothing else changed,
which is the argument for setting the threshold from the **standard deviation**
rather than the range: the range is a function of how many runs were taken.

| | mean | stdev | range | gate |
|---|---:|---:|---:|---:|
| Loss, tail-50 mean | 6.753586 | 0.088% | 0.223% | **±0.352%** (4 stdev) |
| Median step time | 1430.3 ms | 0.093% | 0.210% | **+1.0%** |

Baseline frozen at `examples/qwen3/configs/mxfp4_loss_baseline.json`. All five
runs sit between -0.131% and +0.093% of it.

### The harness

```bash
scripts/mxfp4_loss_regression.py calibrate LOG...            # spread across identical runs
scripts/mxfp4_loss_regression.py record -o baseline.json LOG...
scripts/mxfp4_loss_regression.py check -b baseline.json LOG  # non-zero on regression
```

It refuses to compare runs whose iteration counts differ, and fails on any NaN or
skipped iteration. Only a slowdown counts against the step-time gate.

### What the gate is proven to catch, and what it is not

Verified against four controls:

| Control | Loss delta | Step delta | Caught |
|---|---:|---:|---|
| Previous linear path (`ab_base`) | -0.118% | +5.8% | yes, on step |
| `--grad-quant-type mxfp4` | -0.034% | +29.7% | yes, on step |
| BF16 run, different config | -29.5% | +649% | yes, on both, plus iteration mismatch |
| No BF16 tail (`TAIL_BF16=0`) | -0.058% | +0.064% | **no** |

The step-time axis is proven. The loss axis is not, and the last row is why:
dropping the BF16 tail puts five more layers of a 36-layer model into MXFP4, and
at 200 steps that moves the tail mean by less than the noise floor. The training
report has 8B diverging around step 1300 without that tail (§1.5, §6.3), so the
change does matter — just not inside this horizon.

**A 200-step gate cannot see a regression that takes 1300 steps to show.** It
catches performance regressions, NaN, skipped iterations and gross divergence.
Anything subtler needs a longer-horizon variant of the same baseline, which is
the natural next step for this harness.

Worth noting for anyone trying to build a better control: every MXFP4 env toggle
that exists today (`LUMEN_MXFP4_ASM`, `_PRESHUFFLE`, `_AUTOTUNE`,
`DISABLE_WEIGHT_CACHE`) is bit-exact by design, so none of them can produce a
numerics-only perturbation. That is a good property of the code and an
inconvenient one for validating this gate.

Two side findings from the calibration runs, both inside the noise floor at this
horizon rather than proven absent: `--grad-quant-type mxfp4` costs -0.034% of
loss for +29.7% of step time, and the BF16 tail buys -0.058%.

## Defects found

### 2026-08-10 — out-of-bounds read and write in the MXFP4 quantize kernels

**Severity: silent wrong results plus a write past the end of a tensor.** Fixed.

`_convert_to_mxfp4_kernel` and its siblings address their tiles with no bounds
masks, but the launchers built the grid with `cdiv` over a fixed tile of 64 rows.
Any row count that is not a multiple of 64 left the last program reading past the
input and writing past the scale tensor. The assertions only required
divisibility by `block_size` (32), so `M = 96` was accepted and quietly corrupted.

Four entry points were affected: `convert_to_mxfp4`, `convert_to_mxfp4_2d`,
`hadamard_quant_mxfp4`, `dual_layout_quant_mxfp4`, plus
`dequant_hadamard_quant_mxfp4`.

Why it survived: every shape in `MXFP4_SHAPES` is a multiple of 64, and so is
every Qwen3-8B shape after tensor parallelism, so no training run was affected.
It is the "edge values" row of the matrix — the one nobody had written.

Two symptoms made it visible. Against torchAO, `M = 96` and `M = 224` produced
*wrong* packed values, not merely unstable ones. And the same call run twice
returned different scales, because the out-of-bounds region picked up whatever
the allocator had recycled.

The fix picks the largest power-of-two tile that divides the dimension
(`_dividing_block`) instead of masking every load and store. Every training shape
is a multiple of the cap, so they keep the same tile and the same performance;
only misaligned shapes drop to a smaller tile. It also removed the
`CompilationError` that `dual_layout_quant_mxfp4` raised at `M` of 96, 160 and 224.

Pinned by `test_mxfp4_1d_rtn_unaligned_rows_vs_torchao`,
`test_mxfp4_quant_unaligned_rows_are_reproducible` and
`test_mxfp4_dividing_block_keeps_full_tile_on_aligned_shapes`. All 13 of the new
cases that can fail do fail on the pre-fix code.

Confirmed neutral end to end, which is what the claim about training shapes
requires: a 200-step run on the fixed code lands +0.087% in loss and -0.066% in
step time against a baseline recorded before the fix — both inside the noise
floor, on a config where both fixes should be no-ops by construction.

### 2026-08-10 — `test_mxfp4_backward_optimization.py` was never running

The file pinned `sys.modules['lumen'].__path__` to a git worktree that no longer
exists, so it failed to import and pytest reported a collection error rather than
a failure. It had also reimplemented all seven of the ops it was supposed to
cover, so even the two cases that ran were testing copies of production logic that
had since drifted — the copied kernel wrapper was missing the `hmat_ptr` argument
added with the RHT matrix.

Now imports the production ops directly. Its two remaining failures were the
out-of-bounds defect above (DGrad off by 48.1, forward SNR 0.7 dB); all five pass.

The lesson worth keeping: a test that reimplements the op it covers cannot fail
when the op changes, which is the only time it matters.

### 2026-08-10 — MXFP4 activations were row-padded and never unpadded

**Severity: loud failure, no silent corruption.** Fixed.

`quantize_input` padded the row count up to a multiple of 32 for both operands
on the MXFP4 path and discarded the original size, so any token count that was
not a multiple of 32 reached the GEMM with extra rows. `quantized_linear` then
failed its final `view` — at `M = 1, K = 256, N = 512` it got 16384 elements
where it wanted 512, because M had become 32.

Only the 2D weight tiles and the swizzled scale layout are organised along rows;
the row-wise activation path takes any M, and after the tile fix above it takes
any M correctly. The pad now applies only to those two cases.

Found by adding `mxfp4` to `test_linear.py`'s scaling-type parametrization,
which is what put an M of 1 through the path for the first time. Pinned by
`test_fp8_linear_m1[mxfp4]`.

## Release thresholds

Filled in where there is measured data; the rest stay open rather than being
given a plausible-looking number.

| Metric | Threshold | Basis |
|---|---|---|
| Test pass rate | 100% on gfx950 | 150/150 today |
| Quantize vs torchAO | bit-exact, `atol=0 rtol=0` | Holds on every shape with a torchAO counterpart |
| GEMM backends vs each other | bit-identical | Precondition for autotune; if it ever fails, disable autotune rather than loosen this |
| Linear SNR, fwd / dX / dW | 12 / 12 / 10 dB | Measured 15.3 / 17.4-17.8 / 15.9-18.9 across `LINEAR_SHAPES` |
| Step time regression | +1% of 1430.3 ms | Stdev over five runs is 0.093%, so 1% is well outside it |
| Loss tail-50 mean | ±0.352% of 6.753586 | 4 stdev over the same five runs |
| Peak memory | 150.4 GiB of 251.7 | Current 8-GPU Qwen3-8B measurement |
| Block scale saturation | TBD | Replaces FP8's overflow rate; needs instrumentation first |

## Gaps, in priority order

1. **A longer-horizon loss baseline.** The 200-step gate is calibrated and
   working, but demonstrably cannot see a change that takes ~1300 steps to
   surface. The same harness against a 1500-step baseline would close the one
   class of regression nothing currently catches.
2. **Tuned-table coverage.** An untuned shape is correct but silently falls back
   off the ASM path, so a change in TP width or sequence length can cost
   performance with nothing reporting it. A test that the shipped CSV covers the
   shapes the training scripts issue would catch it.
4. **TP > 1 correctness.** Prerequisite for the comm-GEMM overlap work.
5. `torch.compile` compatibility — no MXFP4 coverage, low priority until
   something depends on it.

Closed: MXFP4 now runs through `tests/ops/test_linear.py`'s scaling-type
parametrization, which is what surfaced the row-padding defect.

## Changelog

- **2026-08-10b** — Added the loss regression gate. Five same-config 200-step
  runs put the tail-50 noise floor at 0.088% stdev, and the gate is set at
  4 stdev (±0.352%) with step time at +1%. Baseline frozen at
  `examples/qwen3/configs/mxfp4_loss_baseline.json`. Verified on four controls:
  it catches the three that regress step time and misses the one that only
  perturbs numerics, because at 200 steps that perturbation is itself inside the
  noise. That bound is now written down rather than assumed away. Also confirmed
  the day's two production fixes are neutral end to end.
- **2026-08-10** — First edition. Established a 150-test baseline on 8x MI350X
  and added `scripts/run_mxfp4_tests.sh`. Recorded which rows of the FP8 CI
  matrix apply to MXFP4 and which two need redefining rather than adopting.
  Three defects came out of the first honest run: the out-of-bounds read/write
  in the quantize launchers, the un-runnable
  `test_mxfp4_backward_optimization.py` that had been testing copies of the ops
  rather than the ops, and the activation row padding that was never unwound.
  All three are fixed and pinned; the coverage they were hiding behind was the
  "edge values" row nobody had written.
