# MXFP4 GEMM backend selection

Lumen's MXFP4 GEMM was slower than BF16. The cause was not quantization overhead
and not a Lumen kernel — Lumen owns no FP4 GEMM kernel. It was calling the
slowest of the several FP4 GEMM kernels AITER ships. This change makes the
backend choice per shape.

## Background

Both Lumen and AMD's TransformerEngine fork dispatch FP4 GEMMs into AITER; neither
writes its own. They pick different entry points:

| | Lumen (before) | TransformerEngine |
|---|---|---|
| AITER entry | `gemm_afp4wfp4` | `gemm_a4w4_asm` / `gemm_a4w4_blockscale` |
| Implementation | Triton | handwritten ASM / CK |
| Kernel selection | none | `a4w4_blockscale_tuned_gemm.csv` by `(gfx, cu_num, M, N, K)` |
| split-K | none | from the tuned table |
| B operand layout | row-major | preshuffled into (16,16) tiles |
| Scale layout | as produced | padded to 256×8 and swizzled |

Lumen can now reach all three kernels. Which one wins depends on the shape, so
`gemm_mxfp4_dispatch` chooses per call.

## Step 1 — the shuffled-layout Triton kernel

AITER's Triton FP4 GEMM has a sibling, `gemm_afp4wfp4_preshuffle`, which reads the
B operand and both scale tensors from a tiled layout. It needs a shuffle prologue
but is far faster once the weight is large.

- `lumen/ops/dispatch.py` — `_probe_aiter_triton_gemm_mxfp4_preshuffle()`, which
  requires both the kernel and the `shuffle_weight` / `shuffle_scale_gemm` helpers
  that build its layout.
- `lumen/ops/quantize/linear.py`
  - `_gemm_mxfp4_aiter_preshuffle()` — shuffles the B operand and both scales,
    then calls the shuffled kernel. Scale-shuffle tiling is looked up per
    architecture (`gfx950` → `(32, 8)`, `gfx1250` → `(16, 4)`); AITER's defaults
    target gfx1250 and silently mislay the scales on gfx950.
  - `_mxfp4_preshuffle_eligible()` — the shape predicate, overridable with
    `LUMEN_MXFP4_PRESHUFFLE=0|1`.

`_MXFP4_PRESHUFFLE_MIN_WEIGHT_BYTES = 16 MiB` of packed FP4 weight. Past that the
GEMM turns weight-streaming bound and the coalesced tile reads outweigh the
prologue; below it the prologue is not amortised and the plain kernel wins. The
check also enforces `N % 16 == 0` (the shuffle emits `N // 16` tiles) and
`M >= 32`.

## Step 2 — the prebuilt ASM/CK kernels

AITER ships 35 prebuilt gfx950 F4GEMM binaries under `hsa/gfx950/f4gemm/`, all
`BpreShuffle` variants, and Lumen already vendors them. They come with a per-shape
kernel and split-K choice from a tuned table, which is what makes them faster than
anything Triton autotune finds.

- `lumen/ops/dispatch.py` — `_probe_aiter_gemm_mxfp4_asm()`, requiring the
  `gemm_a4w4` dispatcher, the `get_GEMM_config` tuned table, and the two layout
  helpers.
- `lumen/ops/quantize/linear.py`
  - `_pad_and_swizzle_mxfp4_scale()` — pads an E8M0 scale tensor to
    `(roundup(rows, 256), roundup(K/32, 8))` and rewrites it into the swizzled
    order the kernels read.
  - `_gemm_mxfp4_aiter_asm()` — builds the operand layout and hands off to
    `aiter.gemm_a4w4`, which owns the tuned lookup, the ASM-vs-CK choice, and
    slicing the row-padded output back to M.
  - `_mxfp4_asm_tuned()` / `_mxfp4_asm_eligible()` — the gates below, overridable
    with `LUMEN_MXFP4_ASM=0|1`.

Dispatch order is now ASM → shuffled Triton → plain Triton → dequantize + BF16.

### Getting the scale layout right

Nothing documents this layout; TransformerEngine emits it straight out of its
quantize kernel, so the ordering is only visible in
`compute_scale_shuffle_index`. A wrong guess does not raise — it mislays scales —
so each candidate was scored against the plain Triton kernel:

| scale layout | result |
|---|---|
| padded only | 9.94 dB — wrong |
| padded, swizzled, left in the natural `(rows/32, k32*32)` view | illegal memory access |
| padded, swizzled, reshaped back to `(rows_pad, k32_pad)` | bit-exact |

The last is what ships. The kernels index a `(rows_pad, k32_pad)` buffer through a
permuted flat offset, so the swizzle has to preserve that shape — handing them
`shuffle_scale_gemm`'s natural view walks off the end of the tensor. Row padding
to 256 rather than 32 follows TransformerEngine; both were measured correct, and
the tensor is small enough that the difference does not matter.

### Two gates

**A tuned config must exist.** Without one, `aiter.gemm_a4w4` falls back to a
default kernel choice that is not validated for correctness: at
`(M, N, K) = (64, 64, 128)` it silently returns garbage, 0.6 dB against the plain
Triton kernel. This surfaced as a failure in the pre-existing
`test_mxfp4_gemm_vs_torchao_gemm[64x128x64]`. Every shape that *does* hit the
tuned table was bit-exact, so `_mxfp4_asm_tuned()` gates on the lookup — which
also confines us to the shapes AMD benchmarked, the ones the ASM path wins on.

**The weight must be large enough.** Like the shuffled Triton kernel, the ASM path
pays a layout prologue — a weight shuffle plus a pad+swizzle of both scale
tensors. Sweeping N at M=8192 puts the crossover in a different place depending on
whether the CPU gets to run ahead of the GPU:

| packed weight | plain (batch / sync) | ASM (batch / sync) |
|---|---|---|
| 8 MiB | 0.127 / 0.179 | 0.143 / 0.226 |
| 12 MiB | 0.175 / 0.215 | 0.162 / 0.262 |
| 16 MiB | 0.235 / 0.264 | 0.192 / 0.296 |
| 24 MiB | 0.326 / 0.350 | 0.277 / 0.362 |
| 28 MiB | 1.073 / 1.105 | 0.279 / 0.345 |
| 32 MiB | 0.541 / 0.563 | 0.354 / 0.431 |
| 56 MiB | 2.296 / 2.365 | 0.610 / 0.637 |

*batch* keeps the queue full so launch overhead is hidden; *sync* times every call
separately, which is what `benchmarks/bench_utils.cuda_timer` does and which
exposes the prologue's launch chain. ASM has several more dispatched ops than
plain, so *sync* penalises it more: the crossover sits at ~10 MiB under *batch*
but between 24 MiB (plain 3% ahead, within noise) and 28 MiB (ASM 3.2x ahead)
under *sync*. `_MXFP4_ASM_MIN_WEIGHT_BYTES = 26 MiB` sits in that gap, gating on
the pessimistic measurement since a launch-bound training step cannot hide the
difference.

On Llama 3.1 8B this puts both MLP projections on ASM and both attention
projections on plain Triton. Note the plain kernel is not a smooth function of
size — 28 MiB at `N=4096, K=14336` costs 1.073 ms while 35 MiB at
`N=5120, K=14336` costs 0.466 ms — so the threshold is a coarse proxy, not a fit.

Step 3 replaces both of these thresholds, for reasons the next section covers:
a constant fitted to one model's shapes turned out to *lose* on another.

## Result

Through `gemm_mxfp4_dispatch` on one MI350X (gfx950), M=8192, 100 iters, ms/iter:

| layer | N | K | weight | plain | shuffled | ASM | dispatch | selected |
|---|---|---|---|---|---|---|---|---|
| qkv_proj | 6144 | 4096 | 12 MiB | 0.228 | 0.273 | — | 0.221 | plain |
| o_proj | 4096 | 4096 | 8 MiB | 0.176 | 0.218 | — | 0.176 | plain |
| gate_up_proj | 28672 | 4096 | 56 MiB | 2.460 | 1.367 | 0.646 | 0.633 | ASM |
| down_proj | 4096 | 14336 | 28 MiB | 1.257 | 0.437 | 0.339 | 0.341 | ASM |

Per-layer forward GEMM total:

| | ms | vs BF16 |
|---|---|---|
| BF16 | 2.925 | 1.00x |
| before both steps | 4.170 | **0.70x** |
| after step 1 | 2.141 | 1.37x |
| after step 2 | 1.369 | **2.14x** |

**3.05x faster than before, 1.56x faster than step 1 alone, and MXFP4 GEMM now
runs at more than twice BF16 speed instead of losing to it.** Three consecutive
runs gave layer totals of 1.358, 1.363 and 1.369 ms. The plain kernel at
`down_proj` is the one unstable entry, ranging 1.05–1.26 ms across those runs.

This is short of the 1.044 ms the step-1 writeup projected for ASM, because that
projection timed the GEMM alone. The 0.3 ms difference is the layout prologue.

These are Llama forward GEMMs only. Step 3 measures whole training layers across
models, where the picture is considerably less flattering.

## Step 3 — measure instead of guessing, and tune per model

Steps 1 and 2 each ended in a hand-measured byte threshold. Pointing them at
Qwen3 showed why that does not survive a change of model.

### The thresholds were overfitted

Qwen3-8B's intermediate size is 12288 against Llama's 14336, so its MLP weights
pack to 24 MiB rather than 28 MiB — just under `_MXFP4_ASM_MIN_WEIGHT_BYTES`.
Every one of its GEMMs was excluded from the ASM path. Qwen3-0.6B is smaller
still: its largest weight is 12 MiB, below even step 1's 16 MiB gate, so all 21
of its GEMMs ran the plain kernel.

Worse, step 1's threshold was actively harmful off-Llama. Summing a full
training layer at 8192 tokens (all three GEMMs of all seven projections):

| model | plain | static thresholds | autotune |
|---|---|---|---|
| Qwen3-0.6B | 2.106 ms | 2.106 ms (1.00x) | 2.004 ms (1.05x) |
| Qwen3-8B | 4.566 ms | 5.089 ms (**0.90x**) | 4.463 ms (1.02x) |
| Llama 3.1 8B | 7.432 ms | 5.870 ms (1.27x) | 4.799 ms (**1.55x**) |

The static policy made Qwen3-8B **10% slower than doing nothing**, because the
shuffled kernel loses to plain at almost all of its shapes. Step 1's large gains
came from plain Triton being pathological at `N=28672` specifically — 2.44 ms
where the shape's size implies about 1.05 ms — which Llama hits and Qwen3 does
not. Measuring beats the constant on all three models, including the one it was
fitted to.

### Autotune

`lumen/ops/quantize/mxfp4_autotune.py` times the backends that can legally run a
shape on its first call and caches the winner under `(M, N, K)`. A model issues
only a couple of dozen distinct shapes, so this costs about a second, once.

This is only sound because the three backends are **bit-for-bit identical** —
verified across Qwen3 and Llama shapes, forward and both backward GEMMs, by
`test_mxfp4_backends_are_interchangeable`. Were they merely close, a timing flip
between runs would silently change a training job's numerics.

Three details keep the decisions stable, all of which matter because a decision
is cached for the life of the process — a bad measurement is not self-correcting.

- The candidates are timed **round-robin**, not one after another. Measured back
  to back, whichever went first absorbed the cold caches and clock ramp-up and
  could permanently lose a contest it deserved to win.
- Each backend's time is the **median** of 11 iterations.
- A layout-rewriting backend must beat the plain kernel by 5% (`_SWITCH_MARGIN`)
  to be chosen. Without it, run-to-run spread was enough to rank the same shape
  differently on two measurements.

The dispatch layer itself costs 1.7–3.4 µs per call once the decision is cached,
about 1% of a 0.2 ms GEMM.

`_mxfp4_asm_eligible` and `_mxfp4_preshuffle_eligible` still hold the byte
thresholds, but only as the fallback for `LUMEN_MXFP4_AUTOTUNE=0`. The hard
constraints moved to `_mxfp4_asm_supported` and `_mxfp4_preshuffle_supported`.

Both dispatch paths go through the autotuner. The `try_backends` path
(`LUMEN_FAST_QUANT_DISPATCH=0`) puts the chosen backend first and keeps the others
behind it, so a backend that rejects the operands at runtime still degrades rather
than raising.

### Getting a new model's shapes into the tuned table

An untuned shape cannot use the ASM path at all, and must not: AITER picks an
unvalidated default kernel for it. `scripts/mxfp4_tune_shapes.py` runs the four
stages — collect, untuned, tune, verify. Stage 1 drives the real
`quantized_linear` with `LUMEN_MXFP4_GEMM_SHAPE_LOG` set rather than deriving
shapes on paper, because each linear issues three GEMMs and the backward pair
permutes the dimensions (a wgrad's M is the output width and its K is the token
count). It also takes `--tp`, since tensor parallelism changes the shapes.

For Qwen3-8B and 0.6B together this found 19 distinct shapes, 6 already tuned.
AITER's tuner covered the other 13 in 157 s, trying 702 candidates (20 CK
instances plus 34 preshuffle ASM tiles per shape).

Stage 4 is an independent correctness gate. The tuner validates its own picks
against a dequantized f32 reference and rejects anything whose `errRatio` — the
fraction of mismatched elements — exceeds 5%, which is loose. All 1470 rows AITER
ships have `errRatio` exactly 0, and all 13 new rows came out **bit-exact against
the plain Triton kernel**, so none were dropped. Any that were not would be
excluded from the shipped CSV rather than trusted.

The result is checked in at `examples/qwen3/configs/a4w4_blockscale_tuned_gemm.csv`.
Both Qwen3 training scripts wire it up themselves in `mxfp4` mode via
`mxfp4_autotune.configure()`, which colon-joins it with AITER's own table so the
two merge rather than one replacing the other. Setting `AITER_CONFIG_GEMM_A4W4`
by hand still wins, for overriding without editing code.

`configure()` does not have to run before `import aiter` — AITER reads the
variable when it first looks up a config, not at import — but it must run before
the first MXFP4 GEMM, after which the lookup is cached. It warns if it is called
too late rather than silently doing nothing.

The first run after adding rows also needs `AITER_REBUILD=1`, so the CK kernels
for them get built.

`--mxfp4-autotune-cache` persists the per-shape decisions to JSON. The point is
less the ~140 ms of measurement it saves than reproducibility: several shapes sit
within a few percent of `_SWITCH_MARGIN`, so without a cache two processes can
rank the backends differently. The file records the architecture it was measured
on and is ignored elsewhere.

### The prologue is now the binding constraint

With the table widened, the ASM kernel is reachable everywhere on Qwen3-8B but
only wins by about 5%. The tuner clocks `8192x4096x12288` at 199 µs; through
Lumen it measures 300 µs against plain's 317 µs. The missing 100 µs is the layout
prologue, rebuilt on every call.

So the tuned table does not make Qwen3-8B measurably faster *today* — it roughly
doubles how many shapes can reach ASM, and raises the ceiling, but the prologue
eats the difference:

| Qwen3-8B | without the table | with it |
|---|---|---|
| autotune, wall time | 4.589 ms (1.02x) | 4.519 ms (1.04x) |
| shapes choosing ASM | 3 | 6 |
| ceiling with no prologue | 3.443 ms (1.35x) | 3.252 ms (**1.44x**) |

Timing the ASM kernel with the layout hoisted out gives what step 4 would leave:

| model | autotune | prologue-free |
|---|---|---|
| Qwen3-0.6B | 2.004 ms (1.05x) | 1.268 ms (**1.66x**) |
| Qwen3-8B | 4.463 ms (1.02x) | 3.184 ms (**1.43x**) |
| Llama 3.1 8B | 4.799 ms (1.55x) | 3.915 ms (**1.90x**) |

On Qwen3 nearly all the remaining headroom is prologue, not kernel. The table is a
prerequisite for collecting it, not the win itself.

### Step 4: making the prologue cheap without fusing it

Fusing the layout into the quantizer turned out to be the wrong first move. The
three GEMMs take their B operand from three *different* Lumen Triton kernels
(`convert_to_mxfp4_2d` for fprop, `transpose_packed_fp4` for dgrad,
`hadamard_quant_mxfp4` for wgrad) and no buffer is shared between them, so fusing
means editing three store paths. Worse, the layout would have to be chosen at
quantize time, before autotune knows which backend wins — a shuffled buffer is
unusable by the plain kernel.

Timing the prologue's parts first showed most of it was not inherent:

| | before | after |
|---|---|---|
| `shuffle_weight`, 12 MiB | 41.9 µs (1.2 TB/s) | 23.7 µs (2.1 TB/s) |
| `shuffle_weight`, 28 MiB | 77.5 µs (1.5 TB/s) | 33.7 µs (3.5 TB/s) |
| pad+swizzle, one scale | 30.4 µs | 18.6 µs |

Two causes, both fixed without touching a kernel:

- **The pad was usually a no-op.** Scales need rows aligned to 256 and columns
  (K/32) to 8, which every realistic training shape already satisfies. The
  `torch.zeros` + slice copy was building a byte-identical duplicate for ~17 µs.
  It now runs only when the shape is genuinely unaligned.
- **The weight shuffle was copying one byte at a time.** The permutation leaves
  the innermost 16 bytes contiguous at both ends, so it is a transpose of 16-byte
  units; AITER expresses it over a `uint8` view, which cannot vectorize. Viewing
  the same bytes as `int64` roughly doubles throughput. `_shuffle_mxfp4_weight`
  does this and falls back to AITER for gfx1250, unaligned or non-contiguous
  operands, and weights under 4 MiB, where the copy is launch-bound and the wide
  view measures ~2 µs slower.

Both are bit-exact with what they replaced. Measured A/B in one process,
alternating between old and new so clock drift hits both equally:

| shape | plain | old prologue | new prologue |
|---|---|---|---|
| q3-8b gate fprop | 382.3 µs | 358.0 (1.07x) | 328.3 (**1.16x**) |
| q3-8b gate dgrad | 347.2 µs | 313.8 (1.11x) | 287.7 (**1.21x**) |
| q3-8b down wgrad | 351.9 µs | 325.2 (1.08x) | 298.5 (**1.18x**) |
| q3-8b qkv fprop | 233.8 µs | 256.9 (0.91x) | 228.1 (**1.02x**) |

That is a flat ~27 µs off every ASM call, which flips several Qwen3-8B shapes
from losing to winning. Per-layer totals, three repeats each:

| model | before step 4 | after step 4 | shapes on ASM | prologue-free ceiling |
|---|---|---|---|---|
| Qwen3-0.6B | 1.05x | 1.03x | 2–4 of 21 | 1.63x |
| Qwen3-8B | 1.02x | **1.12x** | 3 → 9 of 21 | 1.48x |
| Llama 3.1 8B | 1.55x | 1.59x | 6 of 21 | 1.92x |

Qwen3-0.6B is unchanged: its GEMMs are small enough that ASM loses on nearly
every shape regardless of prologue cost, and autotune keeps picking plain.

A fused cast+shuffle quantizer is still worth roughly another 1.3x on Qwen3-8B,
but it is now a second-order fix rather than the whole gap.

### Step 5: what the profile actually said

All of the above is GEMM-level. End to end on Qwen3-8B (8192 tokens/step, one
GPU, medians of two runs), it barely registers:

| policy | step time | vs plain |
|---|---|---|
| plain | 1102.6 ms | 1.00x |
| static thresholds | 1130.9 ms | 0.97x |
| autotune | 1097.8 ms | 1.004x |

So the whole backend-selection effort is worth 0.4% of a training step, and its
real value is avoiding the 2.6% regression the static policy would have shipped.

Profiling a step (`--profile-steps 2` on the pretrain script) explains why:

| | per step | share |
|---|---|---|
| `aten::copy_` | 267 ms | 24.6% |
| AdamW | 126 ms | 11.6% |
| `aten::mul` | 120 ms | 11.0% |
| flash attention backward | 119 ms | 10.9% |
| **all MXFP4 GEMMs** | **102 ms** | **9.4%** |
| BF16 `mm` (tail layers) | 75 ms | 6.9% |

The GEMMs are 9.4% of the step, so driving them to zero would buy 9.4%. The
copies are 2.6x larger, and grouping them by operand shape pointed straight at
the wgrad path: `[12288, 8192]` at 43 ms/step and `[4096, 8192]` at 42 ms/step,
together 7.8% of the step, were `grad_flat.t().contiguous()` and
`input_bf16.t().contiguous()`.

Those copies existed only by habit. `_fused_hadamard_quant_mxfp4_kernel` already
addressed its input through both strides, and `hadamard_quant_mxfp4` was calling
`.contiguous()` before handing it over. Passing the transposed view straight
through removes the materialisation:

| wgrad operand | materialise | view | speedup |
|---|---|---|---|
| q3-8b down `input_t` 12288x8192 | 628.0 µs | 234.8 µs | 2.67x |
| q3-8b gate `grad_t` 12288x8192 | 623.9 µs | 230.6 µs | 2.71x |
| q3-8b qkv `grad_t` 6144x8192 | 322.8 µs | 134.7 µs | 2.40x |
| q3-8b o_proj `input_t` 4096x8192 | 209.2 µs | 103.6 µs | 2.02x |

Bit-exact, not merely close. Stochastic rounding derives its randomness from
tile-local `tl.arange` plus a fixed philox offset rather than from the address,
so a strided read sees the same random stream; `_convert_to_mxfp4_kernel`'s tile
decomposition is unchanged. `test_hadamard_quant_reads_transpose_without_materialising`
pins this, because a quietly different gradient would be far worse than a slow one.

End to end that is worth **1.078x** (1097.8 ms to 1018.4 ms), and `aten::copy_`
drops from 24.6% to 17.4% of the step — about twenty times the end-to-end effect
of all the backend-selection work above, from a much smaller change.

### Step 6: the copies that were left

Two of the `aten::copy_` entries step 5 left unexplained have since been dealt with.

**The activation dequant in wgrad.** `convert_from_mxfp4(...).t().contiguous()` wrote
a full BF16 `(M, K)` intermediate and then copied it transposed.
`dequant_transpose_mxfp4` reads packed FP4 `(M, K/2)` and writes BF16 `(K, M)` in one
launch, so neither the intermediate nor the copy exists. Bit-exact with the two-op
form (`bench_mxfp4_backward_ops.py::test_fused_dequant_transpose_is_bit_exact`).

**The weight quantization repeated per micro-batch.** RTN is deterministic and the
BF16 weight does not move within an optimizer step, so every micro-batch after the
first was re-deriving a bit-identical FP4 weight and re-running the pre-transpose —
and gradient checkpointing paid for it a second time. A module-level cache
(`module._mxfp4_w_cache`, cleared by an optimizer post-step hook) collapses that to
once per step. It costs 4.8 GB/GPU on Qwen3-8B, since every quantized layer's FP4
weight now stays live for the whole step. Reuse is bit-exact by construction
(`test_mxfp4_backward_optimization.py::test_weight_caching_gemm_correctness`);
`LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` turns it off.

Together with the FP4 parameter all-gather and the wgrad rounding fix, measured on
8 GPUs at 65536 tokens/step (the configuration the model actually trains in, rather
than step 5's single-GPU 8192):

| build | median step | vs BF16 |
|---|---|---|
| BF16 | 928.0 ms | 1.00x |
| MXFP4 through step 5 (`7d1841b`) | 1061.8 ms | 0.87x |
| MXFP4 through step 6 (`035431e`) | **869.4 ms** | **1.067x** |

**MXFP4 is now faster than BF16 end to end**, by 6.7%, at 20.90 GB against BF16's
15.30. The four step-6 changes were not A/B'd individually — the machine was shared
and isolated micro-benchmarks of the same shape varied 2x with inconsistent
direction, so only the end-to-end medians are trustworthy. Full write-up, including
the failed attempts, in [`mxfp4_optimization_report.md`](mxfp4_optimization_report.md).

## Tests

```
pytest tests/ops/test_quantize.py -k "mxfp4 or hadamard" -q   # 47 passed
pytest benchmarks/bench_mxfp4_gemm.py -v -s            # 5 passed
pytest benchmarks/bench_mxfp4_gemm_models.py -v -s     # 3 passed
```

Running the whole of `test_quantize.py` aborts in an unrelated FP8/torchao test;
that predates this work.

Step 1:

- `test_mxfp4_preshuffle_gemm_matches_plain` — SNR of the shuffled kernel against
  the plain one at both MLP shapes, plus an exact check that the dispatcher routes
  them to the shuffled kernel.
- `test_mxfp4_preshuffle_eligibility` — the predicate accepts the MLP shapes,
  rejects the attention shapes, and rejects `N % 16 != 0` and `M < 32`.

Step 2:

- `test_mxfp4_asm_gemm_matches_plain` — the ASM/CK path is bit-exact against the
  plain kernel at three shapes, the output is sliced back to M, and the
  dispatcher's output matches.
- `test_mxfp4_asm_eligibility` — accepts the MLP shapes, rejects the attention
  shapes on size, rejects `N % 16 != 0` and `K % 32 != 0` on tiling, and rejects
  the untuned `(64, 64, 128)`.
- `test_mxfp4_asm_scale_pad_and_swizzle_roundtrip` — the padded, swizzled scales
  round-trip through `unshuffle_scale_gemm`, keep the shape the ASM kernel indexes
  against, and zero-fill the padding, at aligned, odd-row and wide-K shapes.

Step 3:

- `test_mxfp4_backends_are_interchangeable` — every available backend is
  bit-identical to the plain one. This is the precondition for autotune; if it
  ever fails, autotune must be disabled rather than fixed up.
- `test_mxfp4_autotune_picks_and_caches` — the choice is one of the legal
  backends, is cached under the shape, and is not re-measured.
- `test_mxfp4_autotune_cache_roundtrip` — a persisted decision is reused, and one
  recorded on a different architecture is ignored.
- `test_mxfp4_shape_log_records_all_three_gemms` — one `quantized_linear` call
  logs `(M,N,K)`, `(M,K,N)` and `(N,K,M)`, so the collector sees fprop, dgrad and
  wgrad.

Also re-run green with `LUMEN_FAST_QUANT_DISPATCH=0` (the `try_backends` path
rather than the cached fast path), with `LUMEN_MXFP4_ASM` set to `0` and `1`, and
with `LUMEN_MXFP4_AUTOTUNE=0` (36 passed, 1 skipped — the autotune test skips
itself when the path it covers is off).

The ASM/CK path came out bit-exact against the plain kernel on every Llama and
Qwen3 shape and on a deliberately unaligned `M = 300`; the shuffled path is
bit-exact too. Both layout changes are value-preserving.

Unrelated pre-existing breakage:

- `tests/ops/test_quantize.py` aborts the whole process at
  `test_quant_fp8_tensorwise_vs_torchao[fp8_dtype0-dtype_in1-64x128]` inside an
  AITER JIT op, so the file cannot currently be run end to end — only under
  `-k mxfp4`. Nothing here touches the FP8 path.
- Most of `tests/modules/` and several `tests/ops/` files fail to collect with
  `NameError: USE_FP8E5M2_BWD` from `lumen/kernels/attention/attention_impl.py`.

## Not done yet

**Fold the shuffle into quantization.** This is now the only large win left, and
on Qwen3 it is most of the remaining headroom: 1.43x on 8B and 1.66x on 0.6B,
against the 1.02x and 1.05x autotune delivers today. TransformerEngine's
`cast_transpose_mxfp4_shuffled.cuh` does cast, transpose, Hadamard and shuffle in
one HIP kernel, where the shuffle is only a different store address and costs
nothing. Lumen's equivalent work is spread across `convert_to_mxfp4_2d`,
`transpose_packed_fp4`, `dequant_transpose_mxfp4` and `hadamard_quant_mxfp4` —
four store paths, though the two `.t().contiguous()` copies that used to sit
between them are gone (steps 5 and 6). Folding the shuffle into each quantize
kernel's store address would erase the prologue — and would make autotune's job
easy, since with a free layout the fast kernels win nearly everywhere.

**Share autotune decisions across ranks.** Each rank measures independently.
Because the backends are bit-exact this cannot diverge numerically, only in
speed, but having rank 0 decide and broadcast — or shipping a
`LUMEN_MXFP4_AUTOTUNE_CACHE` produced by a warmup run — would make a distributed
job's behaviour reproducible.

**Retune when the shape set changes.** Tuning is per `(gfx, cu_num, M, N, K)`, so
a new model, a different tensor-parallel width, or a different tokens-per-step all
produce shapes the table may not hold. `scripts/mxfp4_tune_shapes.py` is the
repeatable path; nothing about it is Qwen3-specific.
