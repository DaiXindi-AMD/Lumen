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
results. For MXFP4 that gate is red by construction:

- WGrad uses stochastic rounding, and `philox_seed` falls back to
  `random.randint()` off Python's global RNG when the caller does not pin it.
- Empirically, two 200-step runs of an identical configuration differ by 0.25%
  relative in mean loss and 1.8 ms in median step time (2026-08-10).

So the useful gate is a **noise-floor-aware** one: measure the run-to-run spread
first, then set the threshold above it. A bit-exactness requirement would either
fail constantly or force SR off, which changes the numerics being shipped.

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
| Step time regression | > 1% of 1429.0 ms | Noise floor is 1.8 ms on the current path (~0.13%), so 1% is comfortably outside it |
| Peak memory | 150.4 GiB of 251.7 | Current 8-GPU Qwen3-8B measurement |
| Loss drift vs BF16 | TBD | Needs the fixed-seed harness below; must be set above the 0.25% run-to-run spread |
| Block scale saturation | TBD | Replaces FP8's overflow rate; needs instrumentation first |

## Gaps, in priority order

1. **Fixed-seed loss regression with a noise-floor gate.** The highest-value
   missing test and the only one that would catch a numerics regression that does
   not crash. Needs the spread measured before the threshold is set.
2. **Multi-GPU smoke.** 8-GPU Megatron, ~200 steps, asserting the run completes
   and the loss lands in a band. Nothing multi-GPU covers MXFP4 today.
3. **Tuned-table coverage.** An untuned shape is correct but silently falls back
   off the ASM path, so a change in TP width or sequence length can cost
   performance with nothing reporting it. A test that the shipped CSV covers the
   shapes the training scripts issue would catch it.
4. **TP > 1 correctness.** Prerequisite for the comm-GEMM overlap work.
5. `torch.compile` compatibility — no MXFP4 coverage, low priority until
   something depends on it.

Closed: MXFP4 now runs through `tests/ops/test_linear.py`'s scaling-type
parametrization, which is what surfaced the row-padding defect.

## Changelog

- **2026-08-10** — First edition. Established a 150-test baseline on 8x MI350X
  and added `scripts/run_mxfp4_tests.sh`. Recorded which rows of the FP8 CI
  matrix apply to MXFP4 and which two need redefining rather than adopting.
  Three defects came out of the first honest run: the out-of-bounds read/write
  in the quantize launchers, the un-runnable
  `test_mxfp4_backward_optimization.py` that had been testing copies of the ops
  rather than the ops, and the activation row padding that was never unwound.
  All three are fixed and pinned; the coverage they were hiding behind was the
  "edge values" row nobody had written.
