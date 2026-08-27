# MXFP4 vs BF16 vs FP8 — training step-time benchmark

Companion to the BF16/FP8 reference report a colleague measured on 8x MI325X,
extended to add MXFP4 as a third precision. The reference report compared two
precisions across two stacks (TE and Lumen); this one compares three precisions
on Lumen, on 8x MI350X, with the reference report's own logs re-parsed alongside
so its numbers and these are read with the same ruler.

Design and accuracy of the MXFP4 path itself live in
[`mxfp4_training_report.md`](mxfp4_training_report.md); test status and release
gates in [`mxfp4_test_report.md`](mxfp4_test_report.md). This file is only the
cross-precision, cross-model step-time comparison.

Headline: **MXFP4 is 1.41x-1.52x faster per step than BF16 and 1.28x-1.36x
faster than FP8 delayed, at 1.9%-2.5% less peak memory.**

The FP8 arm was re-measured on 2026-08-14 against the AITER the upstream repo
pins rather than the stock ROCm one this machine had installed, which is what
made the reference report's `hybrid` recipe runnable here at all. With that
recipe FP8 delayed is worth **1.09x-1.14x** over BF16 — against 1.00x-1.04x for
the `fp8_e4m3` recipe the first version of this report was forced to use, and
1.38x-1.43x on MI325X. Re-running everything inside the coworker's own Docker
image (next section) reaches 1.15x-1.20x, which is the best seen on gfx950 and
still 15%-19% short. §8.1 is what was wrong and §9 is the gap that remains.

Both results were re-measured over 350 steps on real C4 text with a held-out
validation set (§12) and came back to three digits, so neither is an artefact of
short runs or mock data.

## 2026-08-19: exact coworker Docker environment re-run

All 16 canonical FP8 cells added on 2026-08-14 were re-run in the coworker's
environment: Lumen `c410f2cb`, AITER `e42f5791a` (`lumen/triton_kernels`) and
`zhangdanyangamd/lumen@sha256:92f90b3d92dfb5e9fa942e28dfab340f53a1bc0d9b8986f8a8623d8da67903b6`.
Six matching BF16 controls were also run so every ratio below is internal to this
environment. Step time uses the same median of iterations after iteration 10.

That MI325X checkout is not directly runnable on gfx950. Its fused
cast+transpose extension accepts only the gfx942 FNUZ FP8 types, so the isolated
reproduction worktree adds dispatch and HIP conversion for gfx950 OCP E4M3/E5M2.
The kernel matches PyTorch bit-for-bit for row and transposed outputs for both
types and matches amax. The launchers were parameterized only to select the
second FP8 recipe, `--lumen-linear`, and external C4 data. Native-linear cells
set `LUMEN_FUSED_RES_BWD=0` because this AITER's RMSNorm backward does not accept
the `residual_grad` argument; the existing correct fallback costs about 0.4% in
the commit that introduced the fused form. C4 physically repeats the train file
twice instead of using the later `allow_repeat` dataset option. These are
explicit hardware/harness compatibility deltas, not silently claimed as the
unmodified MI325X commit.

### 50-step mock-data core

| Model | BF16 ms | FP8 hybrid ms / speedup | FP8 E4M3 ms / speedup | hybrid vs E4M3 | report hybrid speedup |
|---|---:|---:|---:|---:|---:|
| Llama2-7B | 7730.7 | 6559.7 / **1.179x** | 7247.8 / 1.067x | 1.105x | 1.140x |
| Llama3.1-8B | 9233.1 | 7873.0 / **1.173x** | 8778.5 / 1.052x | 1.115x | 1.121x |
| Qwen3-8B | 9571.6 | 8335.5 / **1.148x** | 9194.2 / 1.041x | 1.103x | 1.093x |

The exact coworker environment therefore does produce more FP8 acceleration
than the report: the hybrid speedup ratio rises by 3.4%, 4.6% and 5.1%
respectively. It does **not** close the MI325X gap. These remain 15%-19% below
the reference's 1.384x-1.432x, and both BF16 and FP8 absolute step times are
slower than the report's newer source. The ratio improved because BF16 lost more,
not because gfx950 FP8 reached MI325X throughput.

### 50-step native-linear probe

| Model | Recipe | native ms | vs matching BF16 | native vs patched Megatron |
|---|---|---:|---:|---:|
| Llama2-7B | hybrid | 6323.8 | **1.222x** | 1.037x |
| Llama2-7B | E4M3 | 7062.8 | 1.095x | 1.026x |
| Qwen3-8B | hybrid | 8055.1 | **1.188x** | 1.035x |
| Qwen3-8B | E4M3 | 8986.4 | 1.065x | 1.023x |

The older environment makes the native path 2.3%-3.7% faster rather than
inside the report's ±1.6% band. Even its best two ratios remain 14%-16% short of
the MI325X reference.

### 350-step C4 with held-out validation

| Model | BF16 ms / val | hybrid ms / speedup / val | E4M3 ms / speedup / val | report hybrid speedup |
|---|---:|---:|---:|---:|
| Llama2-7B | 7842.6 / 5.4972 | 6542.9 / **1.199x** / 5.5090 | 7374.8 / 1.063x / 5.5028 | 1.127x |
| Llama3.1-8B | 9215.4 / 6.2434 | 7894.9 / **1.167x** / 6.2290 | 8795.5 / 1.048x / 6.2298 | 1.123x |
| Qwen3-8B | 9686.1 / 6.1208 | 8446.1 / **1.147x** / 6.1092 | 9290.7 / 1.043x / 6.1108 | 1.092x |

All 16 FP8 cells and all controls completed with zero NaN and zero skipped
iterations. On C4, hybrid and E4M3 held-out loss differ by at most 0.0062 and
both stay within 0.0143 of BF16. The longer runs confirm the same performance
finding as the core matrix: this environment raises gfx950 FP8 to roughly
1.15x-1.20x, not to the coworker's 1.38x-1.43x.

## 1. Scope — what was run

| Group | Runs | Steps | Purpose |
|---|---:|---:|---|
| Core matrix | 9 | 50 | 3 models x {BF16, FP8 delayed, MXFP4} |
| C4 matrix (§12) | 9 | 350 | The same nine cells on real data, with held-out validation |
| Smoke pass | 9 | 3 | Prove each recipe starts before spending 50 steps on it |
| Noise floor | 6 | 50 | BF16 and MXFP4 repeated per model, to size the error bar |
| Tuned A4W4 table A/B | 3 | 50 | Price the tuned MXFP4 GEMM table per model |
| FP8 integration probe | 2 | 50 | Test whether FP8's flat result is the recipe or the path |
| FP8 on the upstream AITER (§8.1) | 16 | 50 / 350 | Both FP8 recipes re-measured: core, native path, C4 |
| BF16 control on that AITER | 2 | 50 | Show the environment change does not move the baseline |
| FP8 smoke on that AITER | 2 | 3 | Prove `hybrid` no longer dies in the first backward |
| **Total measured** | **58** | | ~21 h wall on the 8-GPU node |

Plus two AITER tuning jobs (49 s together) that produced the A4W4 table rows
Llama2-7B and Llama3.1-8B were missing, and one shape-collection pass per model.

Three models, matching the reference report: **Llama2-7B** (6.74 B params),
**Llama3.1-8B** (8.03 B), **Qwen3-8B** (8.19 B). The parameter counts agree with
the reference report's to the digit it published, so these are its models.

## 2. Environment

| | |
|---|---|
| GPU | 8x AMD Instinct MI350X VF, gfx950, 256 CU, 251.7 GiB HBM each |
| Reference report GPU | 8x MI325X, gfx942 — **different hardware, see §9** |
| ROCm / driver | 7.2.4 / 6.16.13 |
| PyTorch | 2.13.0+rocm7.2 (HIP 7.2.53211) |
| Lumen, BF16 / MXFP4 arms | `94560b7`, branch `feature/mxfp4` |
| Lumen, FP8 arms | `ba54880`, branch `bench/fp8-latest-upstream` — the same tree with upstream `origin/main` merged |
| AITER, BF16 / MXFP4 arms | `667d6c66` — stock ROCm/aiter, with four Lumen Triton kernels and `swiglu_fwd/bwd` hand-copied in |
| AITER, FP8 arms | `ecfff3fa8` from `ZhangDanyang-AMD/aiter`, the pin upstream `origin/main` carries, plus `swiglu_fwd/bwd` and `hipb_mm_mixed` re-applied from the `e42f5791a` pin recorded in `third_party/aiter` — see §8.1 |
| Megatron-LM | `1b754411` |
| Launch | native `torchrun`, 8 ranks — the `lumen:dev` image was unavailable |

Parallelism is TP=1, PP=1, CP=1, DP=8 with the distributed optimizer and no
activation recompute, for every cell.

The FP8 rows therefore sit on a different AITER from the BF16 and MXFP4 rows,
which is a caliber difference and has to be treated as one. Two things bound it.
The `fp8_e4m3` arm was re-run on the new AITER and reproduced the old numbers to
within 0.5% on all eight cells it shares with the first version of this report,
so the environment on its own moves nothing. And a BF16 control on two models
came back at +1.60% (Llama2-7B) and -0.58% (Qwen3-8B) — no systematic shift, both
inside the 1.53% noise floor of §10. Ratios below are quoted against the BF16 arm
measured on the same AITER wherever one exists; the two places where it does not
say so.

## 3. Method

Each cell runs `examples/scripts/train_pretrain.sh` through
`examples/scripts/run_precision_matrix.sh`, so the two arms of any comparison
differ only in the quantization recipe: same seed (1234), same mock corpus, same
trainer arguments, same runtime environment.

**Step time is the median of the per-iteration times after iteration 10.** The
first several steps carry JIT compilation and MXFP4's per-shape GEMM autotune
probing, and the eval steps cost several times a train step; a mean over the
whole run mostly measures those. The reference report used a worst-3-trimmed
mean instead. Re-parsing its committed logs with the median gives numbers within
0.5% of what it published on all six cells, so the two statistics agree and its
rows below are directly comparable:

| Cell | Reference report | Median, this parser | Delta |
|---|---:|---:|---:|
| Llama2-7B BF16 | 11520 | 11538.0 | +0.16% |
| Llama2-7B FP8 | 8028 | 8059.0 | +0.39% |
| Llama3.1-8B BF16 | 13607 | 13646.0 | +0.29% |
| Llama3.1-8B FP8 | 9879 | 9860.9 | -0.18% |
| Qwen3-8B BF16 | 14677 | 14613.2 | -0.44% |
| Qwen3-8B FP8 | 10315 | 10334.8 | +0.19% |

TFLOP/s/GPU uses the reference report's formula, `6*P*GBS*seq/step_time/8`, kept
identical so its columns and these stay comparable. It counts only the GEMMs and
ignores attention, so treat it as a throughput index rather than achieved FLOPs;
it is low by a few percent, more so at 8192 sequence length.

`GBS x seq` is 1,048,576 tokens/step for all three models, so tok/s/GPU and step
time carry the same information here. The collector refuses to divide two cells
with different token counts, which is what stops a batch-size mismatch from
being reported as a speedup.

### Recipes

| | BF16 | FP8 delayed | MXFP4 |
|---|---|---|---|
| Format | bf16 | `hybrid` (E4M3 forward, E5M2 backward) and `fp8_e4m3`, both measured | `mxfp4`, E2M1 |
| Scaling | — | delayed, amax history 1024, algo `max` | blockwise, 32-element |
| Quantized linears | — | all: 128 Llama / 144 Qwen3 | all but the last 5 layers: 108 / 124 |
| BF16 layers kept | — | none | last 5 |
| Integration | — | `nn.Linear` monkeypatch | `--lumen-linear` |

The FP8 arm reproduces the reference report's recipe exactly, including its
`hybrid` format, its FP8-only fusion switches and its `nn.Linear` integration
path. `fp8_e4m3` is carried alongside it because that is what the first version
of this report measured — `hybrid` crashed then, for reasons that turned out to
be environmental (§8.1) — and keeping both is what separates the recipe's effect
from the environment's.

**The fusion switches, since most of them default to off.** `train_pretrain.sh`
exports twelve for every FP8 cell, matching the reference report's §7:
`LUMEN_PREFER_HIPBLASLT`, `LUMEN_FUSED_QUANT_TRANSPOSE_CPP`,
`LUMEN_FUSED_QUANT_AMAX`, `LUMEN_FUSED_QUANT_SCALE`, `LUMEN_FUSED_CAST_TRANSPOSE`,
`LUMEN_FUSED_CAST_TRANSPOSE_V2`, `LUMEN_FUSED_SWIGLU_QUANT`,
`LUMEN_FUSED_NORM_QUANT`, `LUMEN_FUSED_NORM_QUANT_V2`, `LUMEN_TRANSPOSE_CACHE`,
`LUMEN_FAST_QUANT_DISPATCH`, `LUMEN_WEIGHT_QUANT_ONCE`. Four more are set for
*every* arm by the matrix harness — `LUMEN_FUSED_SWIGLU`,
`LUMEN_FUSED_RESIDUAL_NORM`, `LUMEN_FUSED_RES_BWD`, `LUMEN_SKIP_BACKEND_SYNC` —
alongside `--use-distributed-optimizer`, `--overlap-grad-reduce` and
`--lumen-attn-backend csrc`, all three confirmed in the parsed-argument dump of
every log. This matters because these are read at import time and most default to
`0`: a run launched outside `train_pretrain.sh` silently measures a different,
slower FP8 path, which is what happened to the profiling scripts §9 cites.

The one §7 item not reproduced is the gfx942 round-to-zero attention conversion
(`how_v3_bf16_cvt=2`), which is specific to the reference report's hardware.

The last two rows are an asymmetry between the FP8 and MXFP4 arms rather than a
choice; §8.3 and §9 cover what it does and does not explain.

## 4. Step time — the core result

50 iterations, median after iteration 10, 8x MI350X. MXFP4 rows are the arm with
a complete tuned A4W4 table (`asm 11/11`), which is what now ships for all three
models; §7 prices the table. §12 repeats this table for 350 steps on real C4 data
and reproduces the MXFP4 ratios to within 0.001x.

FP8 rows are on the AITER of §8.1 and are quoted against the BF16 control
measured on that same AITER where one exists — Llama2-7B and Qwen3-8B. Llama3.1-8B
has no control, so its FP8 rows are quoted against the original BF16 arm and
carry the §2 caveat.

| Model | Precision | ms/iter | vs BF16 | TFLOP/s/GPU | tok/s/GPU | peak mem | GEMM |
|---|---|---:|---:|---:|---:|---:|---|
| Llama2-7B | BF16 | 6475.1 | 1.000x | 818.4 | 20242 | 0.5349 | — |
| Llama2-7B | BF16, §8.1 AITER | 6578.9 | 0.984x | 805.5 | 19923 | 0.5349 | — |
| Llama2-7B | **FP8 hybrid** | **5772.7** | **1.140x** | 918.0 | 22705 | 0.5682 | — |
| Llama2-7B | FP8 e4m3 | 6336.5 | 1.038x | 836.3 | 20685 | 0.5686 | — |
| Llama2-7B | **MXFP4** | **4247.9** | **1.524x** | 1247.5 | 30856 | 0.5245 | asm 11/11 |
| Llama3.1-8B | BF16 | 8209.5 | 1.000x | 769.3 | 15966 | 0.6094 | — |
| Llama3.1-8B | **FP8 hybrid** | **7322.6** | **1.121x** | 862.4 | 17900 | 0.6342 | — |
| Llama3.1-8B | FP8 e4m3 | 8039.6 | 1.021x | 785.5 | 16303 | 0.6336 | — |
| Llama3.1-8B | **MXFP4** | **5720.7** | **1.435x** | 1103.9 | 22912 | 0.5939 | asm 11/11 |
| Qwen3-8B | BF16 | 8573.6 | 1.000x | 751.1 | 15288 | 0.6539 | — |
| Qwen3-8B | BF16, §8.1 AITER | 8524.1 | 1.006x | 755.5 | 15377 | 0.6539 | — |
| Qwen3-8B | **FP8 hybrid** | **7800.6** | **1.093x** | 825.5 | 16803 | 0.6772 | — |
| Qwen3-8B | FP8 e4m3 | 8559.9 | 0.996x | 752.3 | 15312 | 0.6765 | — |
| Qwen3-8B | **MXFP4** | **6097.2** | **1.406x** | 1056.2 | 21497 | 0.6387 | asm 11/11 |

Pairwise, on this stack, FP8 read as the `hybrid` recipe:

| Model | MXFP4 vs BF16 | MXFP4 vs FP8 | FP8 vs BF16 | FP8 hybrid vs FP8 e4m3 |
|---|---:|---:|---:|---:|
| Llama2-7B | 1.524x | 1.359x | 1.140x | 1.098x |
| Llama3.1-8B | 1.435x | 1.280x | 1.121x | 1.098x |
| Qwen3-8B | 1.406x | 1.279x | 1.093x | 1.097x |

MXFP4 wins on all three models by 40%-52% against BF16 and by 28%-36% against
FP8. That second margin used to be 40%-49%, and the difference is entirely FP8's:
switching it from `fp8_e4m3` to the `hybrid` recipe the reference report used is
worth a flat 1.098x on every model (the last column) — 8.9% off the step time.

That 1.098x is not the number format — E4M3 and E5M2 are the same width and cost
the same in a GEMM. It is which wgrad routine the format selects. `linear.py`
branches on `grad_fp8.dtype == input_data.dtype`: matched dtypes take
`_gemm_wgrad_hipblas`, which hands hipBLASLt the strided view `grad_fp8.t()` and
relies on it to apply `HIPBLAS_OP_T` internally, while mixed dtypes take
`gemm_wgrad_mixed`, which consumes the *contiguous* grad transpose that
`LUMEN_FUSED_QUANT_TRANSPOSE_CPP` already produced and so hands over a TN pair.
Its docstring predicts 6%-14% for exactly that on large-K wgrad shapes, and
`.claude/tmp-training-bugs.md` measured the strided operand at 2.28x the TN one
at the Qwen3-8B wgrad shape. So the `hybrid` arm is faster because the E4M3 arm
takes the stride path, which is a Lumen-side finding about `fp8_e4m3` and not a
property of hybrid: an E4M3 recipe routed through the transposed operand should
recover the same 1.098x, and that is worth testing before anyone concludes hybrid
is the faster arithmetic.

### Why the three MXFP4 ratios differ, and why that is not a kernel result

MXFP4 spans 1.406x to 1.524x across the three models, and the ordering is stable
across the 50-step and 350-step matrices. It is worth saying what that spread is,
because the natural reading — that the kernels suit Llama2-7B better than
Qwen3-8B — is not what the numbers say.

MXFP4 accelerates the transformer linears outside the BF16 tail and nothing else.
Attention, the norms and the LM head stay BF16 in every arm, so each model has an
Amdahl ceiling fixed by its architecture. Computing that composition from the
launcher configs (`scripts/mxfp4_speedup_composition.py`) and solving
`1/((1-p) + p/s) = measured` for the effective speedup `s` of the quantized part:

| Model | quantizable `p` | attention | LM head | BF16 tail | measured | implied `s` |
|---|---:|---:|---:|---:|---:|---:|
| Llama2-7B | 75.5% | 8.7% | 1.8% | 14.0% | 1.524x | **1.84x** |
| Llama3.1-8B | 67.2% | 14.3% | 6.0% | 12.5% | 1.435x | **1.82x** |
| Qwen3-8B | 66.6% | 15.7% | 6.9% | 10.7% | 1.406x | **1.77x** |

**The kernels deliver about the same 1.8x everywhere.** The end-to-end ordering
follows `p` exactly, and `p` is architecture. Llama2-7B leads because it runs at
sequence 4096 where the other two run at 8192, which holds its unquantized
attention to 8.7% against their 14%-16%, and because its 32000-token vocabulary
makes the BF16 LM head almost free at 1.8%.

Qwen3-8B trails Llama3.1-8B on two counts at identical sequence length, token
count and near-identical linear FLOPs: its 36 layers carry 12.5% more attention
than Llama3.1-8B's 32 (+1.4 points), and its 151936-token vocabulary makes the
BF16 LM head 18.5% larger (+0.9 points). Working the other way, and worth noting
because it is counter-intuitive, Qwen3-8B has the *smallest* BF16-tail penalty of
the three — five fixed layers is a smaller share of 36 than of 32 — which offsets
part of the loss but not all of it.

That leaves roughly 3% of genuinely kernel-side difference in `s`. Tuned-table
coverage is not the cause: §7 shows all three at `asm 11/11`. The likely residue
is Qwen3-8B's FFN 12288 shape being slightly less efficient on the assembly
kernels than 14336 or 11008, plus QK-LayerNorm (§8.2), which costs about 0.6% of
the step in HBM traffic and does not appear in a FLOP model at all because it is
bandwidth-bound rather than compute-bound.

The FP8 column orders the same way — 1.140x, 1.121x, 1.093x — for the same
reason, which is a useful cross-check: two independent quantization paths ranking
the three models identically is what an architectural explanation predicts and a
kernel-quality explanation does not.

Treat `s` as an effective figure, not a measured GEMM speedup: the model assumes
time proportional to FLOPs with fixed forward/backward multipliers and does not
separately account for bandwidth-bound operators, communication or the optimizer.
The robust finding is not any single `s` but that all three land within 4% of
each other.

The reference report's MI325X rows, for context. Read the two blocks as separate
experiments — different silicon and an older Lumen — not as a delta:

| Model | Precision | ms/iter | vs BF16 |
|---|---|---:|---:|
| Llama2-7B | BF16 | 11538.0 | 1.000x |
| Llama2-7B | FP8 delayed (hybrid) | 8059.0 | 1.432x |
| Llama3.1-8B | BF16 | 13646.0 | 1.000x |
| Llama3.1-8B | FP8 delayed (hybrid) | 9860.9 | 1.384x |
| Qwen3-8B | BF16 | 14613.2 | 1.000x |
| Qwen3-8B | FP8 delayed (hybrid) | 10334.8 | 1.414x |

## 5. Peak memory

`mem usages` is a fraction of the 251.7 GiB HBM; absolute values are that
fraction times 251.7.

| Model | BF16 | FP8 delayed | MXFP4 |
|---|---:|---:|---:|
| Llama2-7B | 134.6 GiB | 143.1 GiB (+6.3%) | 132.0 GiB (-1.9%) |
| Llama3.1-8B | 153.4 GiB | 159.5 GiB (+4.0%) | 149.5 GiB (-2.5%) |
| Qwen3-8B | 164.6 GiB | 170.3 GiB (+3.5%) | 160.8 GiB (-2.3%) |

FP8 delayed costs *more* memory than BF16 on all three models, which is expected
for this recipe: the parameters stay BF16 and delayed scaling adds a 1024-entry
amax history per tensor plus the transpose cache its fusions rely on.

The FP8 column is the same for both recipes: `hybrid` peaks at 0.5682 / 0.6342 /
0.6772 of HBM against `fp8_e4m3`'s 0.5686 / 0.6336 / 0.6765, i.e. within 0.1% on
all three models. The 8.9% of §4 is bought with time, not bytes — E5M2 and E4M3
tensors are the same size, and the contiguous grad transpose the mixed path
consumes was already being materialized by the fused cast+transpose.

MXFP4's saving is small for the same reason — weights and optimizer state are
still BF16/FP32 and only the GEMM operands are 4-bit, so what it saves is
activation and cast-buffer traffic, not parameter storage. Anyone reading "4-bit"
as "4x less memory" is reading the wrong axis; the win here is time, not bytes.
FP4 parameter all-gather, which would move that axis, is
[`mxfp4_training_report.md`](mxfp4_training_report.md) §2.8 and not enabled here.

## 6. Numerical behaviour — and its limits

No run produced a NaN or a skipped iteration: **0/0 on all 49 runs.** Loss
trajectories over the 50 steps:

| Model | BF16 | FP8 hybrid | FP8 e4m3 | MXFP4 |
|---|---|---|---|---|
| Llama2-7B | 11.594 -> 2.209 | 11.721 -> 2.210 | 11.721 -> 2.214 | 11.615 -> 2.212 |
| Llama3.1-8B | 12.989 -> 5.095 | 13.000 -> 5.143 | 13.000 -> 5.085 | 12.931 -> 5.242 |
| Qwen3-8B | 13.244 -> 2.280 | 13.092 -> 2.275 | 13.092 -> 2.279 | 13.141 -> 2.277 |

The two FP8 recipes start from the same loss to six digits, as they must — they
differ only in the backward — and end within 0.004 of each other on Llama2-7B and
Qwen3-8B. Llama3.1-8B's 0.058 gap is the one worth noting, and §12 is where it
gets tested on real data rather than random tokens.

**This is a crash test, not an accuracy result** — §12 is where real loss curves
live. The corpus is uniformly random
token ids, so a loss falling from ~12 to ~2 in 50 steps is the model fitting the
one-step statistics of noise, and it does that whatever the arithmetic. The
useful signal is the absence of divergence, NaNs and loss-scale collapse, plus
final losses landing within ~0.15 of each other. The reference report shares this
limitation, since it uses mock data too.

MXFP4 accuracy evidence that does mean something is elsewhere: 12/12 operator
tests against torchAO, and the C4 runs — Qwen3-0.6B over 10k steps and Qwen3-8B
over 5k — in [`mxfp4_training_report.md`](mxfp4_training_report.md) §3-§4. Those
runs are also where the last-5-layers-BF16 setting comes from: 8B diverged around
step 1300 without it, which no 50-step mock run would ever have caught.

## 7. The tuned A4W4 GEMM table is worth 7%-11%

AITER reaches its prebuilt gfx950 A4W4 assembly kernels only for shapes listed in
a tuned table, keyed on exact M/N/K. A shape that misses falls back to Triton.
Only Qwen3-8B had a Lumen table; Llama2-7B and Llama3.1-8B ran most of their
MXFP4 GEMMs on Triton and their first measurements understated the path.

Collecting the shapes with `scripts/mxfp4_tune_shapes.py`, tuning the misses, and
verifying every row bit-exact against the Triton reference: 9 shapes for
Llama2-7B, 7 for Llama3.1-8B, 16/16 verified, 0 dropped, 49 s of tuning in total
because the CK kernels were already built. Same recipe either side, fresh
autotune cache so the backend choice is actually re-probed:

| Model | Partial table | Full table | Gain |
|---|---:|---:|---:|
| Llama2-7B | 4780.0 (asm 2/11) | 4247.9 (asm 11/11) | -11.1% (1.125x) |
| Llama3.1-8B | 6156.4 (asm 4/11) | 5720.7 (asm 11/11) | -7.1% (1.076x) |
| Qwen3-8B | 6660.9 (asm 3/11) | 6097.2 (asm 11/11) | -8.5% (1.092x) |

Qwen3-8B's row is the A/B run backwards — its tuned table already shipped, so the
partial-table arm was produced by removing it. That it lands in the same 7%-11%
band as the two forward experiments is the useful part.

The tables now ship as
`examples/{llama2,llama31}/configs/*_a4w4_blockscale_tuned_gemm.csv`, assembled
onto the stock AITER table by `train_pretrain.sh`. **This is a standing trap for
new models and new batch shapes:** the table keys on exact M/N/K, so changing
MBS, sequence length or TP silently drops back to Triton and costs ~10% with
nothing in the log to say so. The `gemm` column exists to make that visible, and
it reads the autotune cache because the per-shape probe runs once and a later run
that finds the cache logs nothing about what it chose.

## 8. Deviations from the reference report

### 8.1 The hybrid recipe works — the crash was the wrong AITER, not this stack

**This section replaces the first version's finding.** It reported that the
reference report's `hybrid` FP8 (E4M3 forward, E5M2 backward) dies in the first
backward here:

```
RuntimeError: expected mat1 and mat2 to have the same dtype,
but got: c10::Float8_e5m2 != c10::Float8_e4m3fn
```

That string is a `TORCH_CHECK` inside AITER's own `gradlib/csrc/hipbsolgemm.cu`.
The AITER this machine had installed carries it; the AITER the upstream Lumen repo
pins deletes it, generalizes the matrix-layout creation to a per-operand input
dtype, and adds `hipb_mm_mixed`. `gemm_wgrad_mixed` reaches that code through a
plain `hipb_mm(grad_t, input_fp8, ...)`, which is why hybrid backward was
unreachable here and worked for the reference report.

What was actually installed, and what upstream pins:

| | commit | what it is |
|---|---|---|
| installed, and used by the first version of this report | `667d6c669` | stock ROCm/aiter, 2026-07-15, with four Lumen Triton kernels and `swiglu_fwd/bwd` hand-copied in |
| pin recorded in `third_party/aiter` | `e42f5791a` | `ZhangDanyang-AMD/aiter`, "FP8 GEMM + quant/attention opts for Lumen" — has `hipb_mm_mixed` |
| pin on upstream `origin/main` today | `ecfff3fa8` | a direct descendant of `667d6c669` carrying the same Lumen kernels as a clean cherry-pick |

The FP8 arm now runs on `ecfff3fa8`, because it is the newer of the two upstream
pins and shares `667d6c669`'s ROCm base where `e42f5791a` would have moved it back
seven weeks. Its cherry-pick dropped two pieces Lumen calls, both re-applied from
`e42f5791a`: `swiglu_fwd/bwd`, reached by `LUMEN_FUSED_SWIGLU=1` on every arm, and
`hipb_mm_mixed` with the `.cu` half above.

Two details worth keeping:

- **The Python kernels were already at parity.** Six of the eight files that had
  been hand-copied into the installed tree are byte-identical to what `ecfff3fa8`
  ships. What was missing was the compiled half, which no amount of copying
  `.py` files reaches.
- **AITER's JIT does not notice source changes.** It rebuilds only when the `.so`
  is absent or the arch changed (`jit/core.py:1259`). Editing a `.cu` and
  re-running silently reuses the stale module, which is a good way to measure the
  old kernel and believe you measured the new one.

Verified before any step time was taken: `hipb_mm` at (1024,4096)x(4096,4096)
against a dequantized reference gives relative error 0.0029 for E4M3xE4M3 and
0.0027 for E5M2xE4M3, and a 3-step Llama2-7B hybrid smoke completes 3/3 with 0
NaN and a loss within 0.004 of the E4M3 arm.

**Lumen itself was not behind.** `git cherry` puts upstream's FP8 work — `f542a09`
"FP8 hybrid training optimizations for gfx942" and `6d09c31` — as already
patch-equivalent on `feature/mxfp4`. The branch was four commits behind
`origin/main`, all of them dsv4, an opt-in BF16 tuned GEMM, and an mmap
checkpointing fix; those are merged in now and none of them touches the dense FP8
path. So the gap was the environment, not the source, and §9 is where that leaves
the result.

The first version's closing claim — that the format is "unlikely to explain §9",
since E4M3 and E5M2 are the same width and cost the same in the GEMM — was right
about the arithmetic and wrong about the consequence. The format is read by a
dtype test that selects a different wgrad routine, and that is worth 8.9% (§4).

### 8.2 Qwen3-8B keeps `--qk-layernorm` on all three arms

The reference report's Qwen3 runs had it off. Turning it on for all three
precisions keeps this matrix internally consistent, at the cost of making the
Qwen3 rows not directly comparable to pre-8/10 Qwen3 logs (archived under
`examples/qwen3/results/stale_pre_0810/`).

### 8.3 MXFP4 and FP8 do not use the same integration path

MXFP4 runs on `--lumen-linear`, Megatron's parallel linears replaced by Lumen's;
FP8 runs on the `nn.Linear` monkeypatch, because that is what the reference
report measured. The native path is worth about 5.6% on MXFP4, so the two arms
differ by more than their number format. §11 removes this difference and finds it
changes nothing for FP8, so the core table is not distorted by it.

A second asymmetry cuts the other way: MXFP4 keeps its last 5 layers in BF16 and
so quantizes 108 of 128 linears on the Llama models and 124 of 144 on Qwen3-8B,
where FP8 quantizes all of them. MXFP4 wins by 40%+ while quantizing *fewer*
layers, so this asymmetry does not flatter it — it makes the MXFP4 numbers
conservative. It does mean the two arms carry different accuracy risk, and the
tail-BF16 setting is load-bearing for MXFP4 at longer horizons (§6).

## 9. FP8 delayed is worth 1.09x-1.14x here, at most 1.20x anywhere, and still short of MI325X

**Corrected on 2026-08-14.** The first version of this section reported FP8 at
1.00x-1.03x and called it indistinguishable from BF16. That was measured with the
`fp8_e4m3` recipe, which §8.1 forced on it. With the reference report's own
`hybrid` recipe, on the AITER upstream pins, FP8 is well clear of the noise floor:

| Model | MI325X, reference | MI350X, `hybrid` | MI350X, `fp8_e4m3` | short of MI325X by |
|---|---:|---:|---:|---:|
| Llama2-7B | 1.432x | 1.140x | 1.038x | 25.7% |
| Llama3.1-8B | 1.384x | 1.121x | 1.021x | 23.4% |
| Qwen3-8B | 1.414x | 1.093x | 0.996x | 29.4% |

So the headline finding changes and the open question narrows but survives. FP8
delayed *is* faster than BF16 on gfx950 — 9.3% to 14.0%, six to nine times the
1.53% error bar — and it is still 23%-29% short of what the same recipe bought on
MI325X.

**Updated 2026-08-19, and this is now the tighter bound.** Re-running all 16 FP8
cells inside the coworker's own Docker image (see the section above §1) lifts
hybrid to 1.148x-1.179x on the 50-step core and 1.147x-1.199x on C4 — better than
this report's stack, but still **15%-19% short** of MI325X. The rest of this
section was written against the 23%-29% figure; read it as the mechanism
discussion, and 15%-19% as the gap that actually has to be explained.

What moved is still mostly the baseline. Between the reference report's stack on
MI325X and this one on MI350X, with FP8 read as `hybrid` on both sides:

| Model | BF16 got faster by | FP8 got faster by |
|---|---:|---:|
| Llama2-7B | 1.782x | 1.396x |
| Llama3.1-8B | 1.662x | 1.347x |
| Qwen3-8B | 1.704x | 1.325x |

BF16 gained 1.66x-1.78x; FP8 only 1.33x-1.40x. Had FP8 scaled like BF16, Qwen3-8B
would sit near 6065 ms — almost exactly MXFP4's 6097 ms. The question is still not
why FP8 is slow but why it picked up less of the generational gain than BF16 did.

Ruled out, or bounded:

- **Partly the recipe, and that part is now fixed.** `hybrid` versus `fp8_e4m3` is
  worth a flat 1.098x (§4), 8.9% off the step time, which is about a third of the
  gap to MI325X. §4 also shows the mechanism is not the arithmetic but which wgrad
  routine the dtype test selects.
- **Not the AITER the runs were measured on.** The `fp8_e4m3` arm re-run on
  `ecfff3fa8` reproduces the first version's numbers to within 0.5% on all eight
  shared cells, and the BF16 control moves +1.60% / -0.58%. The environment was
  wrong in a way that made `hybrid` unreachable (§8.1); it was not silently
  costing the E4M3 arm throughput.
- **Not a silent fallback to BF16.** The FP8 arm reports
  `Quantization enabled on 128/144 nn.Linear layers (format=hybrid, scaling=delayed)`
  and its peak memory is 3.5%-6.3% *above* BF16, which is the amax history and
  transpose cache being allocated. It is quantizing.
- **Not the integration path.** §11 re-runs both FP8 recipes on `--lumen-linear`,
  the path MXFP4 uses. Hybrid lands at 1.122x (Llama2-7B) and 1.099x (Qwen3-8B),
  within 1.6% of its `nn.Linear` numbers — the wiring is not it.
- **Not the fusion switches.** All twelve FP8 switches the reference report's §7
  credits are exported for every FP8 cell here, and all 26 FP8 logs confirm the
  hipBLASLt forward path was taken (§3). Audited 2026-08-17 precisely because a
  switch left at its default would have been the cheap explanation. It is not.
- **Not the software environment as a whole, which is the strongest of these.**
  The 2026-08-19 re-run took the coworker's entire stack — their Lumen commit,
  their AITER, their Docker image — and still reached only 1.148x-1.179x on
  gfx950. That closes the last "our build is wrong" hypothesis, since the build
  was theirs. Note *how* it improved: their BF16 is also slower, so the ratio rose
  because the baseline lost more, not because FP8 got faster in absolute terms.

What is left, after the environment hypothesis fell:

1. A roofline ceiling. **Still cannot be the whole story:** MXFP4, on the same
   model, batch shape and linears, finds 1.41x-1.52x through the same integration.
   The headroom exists and FP8 reaches under half of it.
2. The FP8 quantization, cast-transpose and GEMM kernels never got gfx950 tuning —
   they earned the reference report its 1.4x on gfx942. §4's wgrad finding is the
   one instance measured under this report's own configuration: the E4M3 route
   hands hipBLASLt a strided operand, and `.claude/tmp-training-bugs.md` measures
   1.66x-2.28x sitting between the as-called and TN layouts at these shapes.

**A caveat on the rest of that note.** Its kernel-by-kernel numbers — the
~3.5 ms/layer of unfused quantization plumbing, and the finding that the forward
GEMM never reaches hipBLASLt — come from standalone scripts that set no `LUMEN_*`
variables, so they ran at the source defaults. Those defaults are *not* this
report's configuration: `train_pretrain.sh` turns on twelve FP8 fusion switches
for every FP8 cell here (§3), including `LUMEN_PREFER_HIPBLASLT` — the exact flag
that decides the forward GEMM's backend, since `linear.py:1617` picks hipBLASLt
when it is set and Triton when it is not. All 26 FP8 logs behind this report
contain `hipBLASLt workspace pre-allocated`, so their forward GEMM does reach
hipBLASLt and their quantization is fused. Those two figures
describe an unswitched FP8 path and should not be read as the mechanism behind
the 23%-29% here until they are re-taken under the harness environment.

Confirming the rest needs exactly that: the per-op profile re-run with the
benchmark's switches on, comparing FP8 and MXFP4 kernel time at the same GEMM
shapes, which is still out of scope here. With recipe, AITER, integration path,
fusion switches and the entire software environment now eliminated, that profile
is the only remaining way to turn "the kernels were tuned for gfx942" from the
surviving hypothesis into a measured one.

**Until that is done, the FP8 column should be read as "FP8 delayed as currently
implemented, on gfx950" and not as a statement about what FP8 can do on this
hardware.** The MXFP4-vs-BF16 comparison does not depend on it.

## 10. Noise floor

Two identical runs per configuration, same seed, sequentially on an otherwise
idle node:

| Model | BF16 | MXFP4 |
|---|---:|---:|
| Llama2-7B | 6475.1 / 6508.8 — 0.52% | 4780.0 / 4853.1 — 1.53% |
| Llama3.1-8B | 8209.5 / 8203.4 — 0.07% | 6156.4 / 6146.9 — 0.15% |
| Qwen3-8B | 8573.6 / 8525.4 — 0.57% | 6097.2 / 6110.4 — 0.22% |

Worst spread 1.53%, most under 0.6%. Use 1.5% as the error bar on any single
number in this report.

So: MXFP4's 40%-52% margin is at least 26x the worst spread seen here, and is
real. FP8 e4m3's 0.4%-3.8% straddles the noise floor, while FP8 hybrid's
9.2%-14.0% and §7's 7%-11% table effect clear it comfortably.

A third data point on spread came free from the FP8 re-measurement of §2: the
same BF16 cell, same seed, on the two AITER builds, gives 6475.1 / 6578.9 on
Llama2-7B (1.60%) and 8573.6 / 8524.1 on Qwen3-8B (0.58%). That is the same
order as the repeat spread above, so switching AITER did not move BF16 by more
than run-to-run noise — but 1.6% is large enough that a cross-AITER FP8 ratio
should not be read past its second digit. Where it matters, §4 and §12 say which
baseline each ratio uses.

The two BF16 repeats reproduced their loss curves digit for digit. The MXFP4
repeats did not — final loss moved by 0.004 (Llama2-7B), 0.042 (Llama3.1-8B) and
0.005 (Qwen3-8B) — so the MXFP4 path is not bitwise reproducible across runs at a
fixed seed, though BF16 on the same harness is. The cause was not chased here.
It is small next to the precision differences being measured, but worth knowing
before anyone tries to bisect an MXFP4 loss curve by diffing two runs.

## 11. FP8 on the native linear path

Closing the §8.3 asymmetry: same FP8 delayed recipe, `--lumen-linear` added, so
the only remaining difference from the MXFP4 arm is the number format. Run on the
fastest and the slowest model of the three, and now for both FP8 recipes. Ratios
are against the BF16 arm on the matching AITER — 6578.9 and 8524.1 for the FP8
rows, the original 6475.1 and 8573.6 for the MXFP4 rows.

| Model | Recipe | FP8 `nn.Linear` | FP8 native | native vs `nn.Linear` |
|---|---|---:|---:|---:|
| Llama2-7B | hybrid | 5772.7 (1.140x) | 5863.1 (**1.122x**) | +1.57% |
| Llama2-7B | e4m3 | 6336.5 (1.038x) | 6377.7 (**1.032x**) | +0.65% |
| Qwen3-8B | hybrid | 7800.6 (1.093x) | 7754.9 (**1.099x**) | -0.59% |
| Qwen3-8B | e4m3 | 8559.9 (0.996x) | 8542.0 (**0.998x**) | -0.21% |

For scale, MXFP4 on that same native path is 4247.9 (1.524x) and 6097.2 (1.406x).

**The integration path is still not the explanation.** Moving FP8 onto the same
linears MXFP4 uses changes step time by between -0.6% and +1.6% across the four
cells, all inside the 1.53% noise floor, and the hybrid arm stays at 1.09x-1.14x
either way. MXFP4 through those same linears is still 1.28x-1.36x ahead of the
better FP8 recipe (§4). Whatever FP8 is missing on gfx950 is in the quantization
and GEMM kernels it dispatches, not in how it is wired into Megatron — which
leaves hypothesis 2 of §9 as the one to test.

One side result worth keeping, and it survives both recipes: the native path cuts
FP8's peak memory from 0.5682 to 0.5389 on Llama2-7B and 0.6772 to 0.6494 on
Qwen3-8B under hybrid — i.e. from +6.2% over BF16 to +0.7%, and from +3.6% to
-0.7% — and to 0.5386 / 0.6491 under e4m3. It avoids the transpose caching the
`nn.Linear` path relies on. So if the FP8 arm is kept, it should be on
`--lumen-linear`: same speed to within noise, several GiB cheaper.

Also confirmed here: with `--lumen-linear` the run really does execute delayed
scaling (`Enabled FP8 (scaling=delayed) on 128 Lumen parallel linear modules`),
so the old defect where the native path silently downgraded the recipe to FP8
blockwise is genuinely fixed.

## 12. 350 steps on real data (C4)

Everything above is 50 steps on random tokens. That measures step time honestly
but cannot say whether a precision *learns*, which is the limitation §6 flags.
This section repeats the same nine cells for 350 steps on C4 with a held-out
validation file, to ask two things: does the step-time result survive a 7x
longer run on real data, and do the three precisions track each other's loss.

Same harness, same seed, same recipes, same tuned A4W4 tables; the corpus and
the step count are the only changes.

| Model | Precision | ms/iter | vs BF16 | vs BF16 at 50 steps | tok/s/GPU | peak mem | train loss | held-out val |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Llama2-7B | BF16 | 6558.9 | 1.000x | 1.000x | 19984 | 0.5349 | 11.197 -> 5.494 | 5.497 |
| Llama2-7B | FP8 delayed | 6442.2 | 1.018x | 1.025x | 20346 | 0.5686 | 11.185 -> 5.501 | 5.502 |
| Llama2-7B | **MXFP4** | **4301.4** | **1.525x** | 1.524x | 30472 | 0.5245 | 11.190 -> 5.560 | 5.561 |
| Llama3.1-8B | BF16 | 8222.2 | 1.000x | 1.000x | 15941 | 0.6094 | 12.569 -> 6.282 | 6.245 |
| Llama3.1-8B | FP8 delayed | 8030.8 | 1.024x | 1.026x | 16321 | 0.6327 | 12.579 -> 6.271 | 6.242 |
| Llama3.1-8B | **MXFP4** | **5726.4** | **1.436x** | 1.435x | 22889 | 0.5939 | 12.587 -> 6.342 | 6.300 |
| Qwen3-8B | BF16 | 8669.7 | 1.000x | 1.000x | 15118 | 0.6539 | 12.777 -> 6.170 | 6.217 |
| Qwen3-8B | FP8 delayed | 8704.4 | 0.996x | 1.002x | 15058 | 0.6763 | 12.774 -> 6.181 | 6.230 |
| Qwen3-8B | **MXFP4** | **6169.1** | **1.405x** | 1.406x | 21246 | 0.6388 | 12.785 -> 6.244 | 6.285 |

**The step-time result reproduces, to three digits.** MXFP4 lands at 1.525x,
1.436x and 1.405x against 1.524x, 1.435x and 1.406x from the 50-step mock
matrix — an order of magnitude inside the 1.5% noise floor, on a different
corpus at 7x the length. FP8 E4M3 stays flat, and Qwen3-8B's arm is now 0.996x,
i.e. marginally *slower* than BF16; with §10's error bar that is still
"indistinguishable from BF16", but it removes any reading of §9 as an artefact
of short runs or mock data. The hybrid recipe reproduces just as tightly on the
newer AITER — see below.

**The loss curves are now real, and MXFP4 tracks BF16.** Held-out validation
loss, MXFP4 minus BF16: +0.064 (Llama2-7B), +0.055 (Llama3.1-8B), +0.068
(Qwen3-8B). FP8 sits within ±0.013. The MXFP4 gaps are small, consistent across
three models, and in the expected direction — it is the coarsest arithmetic
here — and no run produced a NaN or a skipped iteration.

Read that as corroboration, not as an accuracy verdict. 350 steps at 1.05 M
tokens each is 367 M tokens, about a third of a percent of a real pretraining
budget, and it is well short of the ~1300-step horizon where 8B diverged without
the tail-BF16 setting ([`mxfp4_training_report.md`](mxfp4_training_report.md)
§1.5). What it does establish is that the MXFP4 path learns on real text at a
rate indistinguishable-in-shape from BF16, which no amount of random-token
running could show.

### The FP8 rows, re-measured on the newer AITER

Six cells, both recipes on all three models, same corpus, same seed, same 350
steps, on the `ecfff3fa8` build of §2. Ratios are against the BF16 rows of the
table above, which is a cross-AITER comparison; the 50-step controls of §4 put
that mismatch at +1.6% (Llama2-7B) and -0.6% (Qwen3-8B) on BF16 itself, so read
the third digit with that in mind.

| Model | Recipe | ms/iter | vs BF16 | vs BF16 at 50 steps | tok/s/GPU | peak mem | train loss | held-out val |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Llama2-7B | **hybrid** | **5820.0** | **1.127x** | 1.140x | 22521 | 0.5684 | 11.185 -> 5.501 | 5.503 |
| Llama2-7B | e4m3 | 6426.6 | 1.021x | 1.038x | 20395 | 0.5686 | 11.185 -> 5.501 | 5.503 |
| Llama3.1-8B | **hybrid** | **7319.6** | **1.123x** | 1.121x | 17907 | 0.6332 | 12.579 -> 6.262 | 6.235 |
| Llama3.1-8B | e4m3 | 8029.9 | 1.024x | 1.021x | 16323 | 0.6328 | 12.579 -> 6.253 | 6.233 |
| Qwen3-8B | **hybrid** | **7937.1** | **1.092x** | 1.093x | 16514 | 0.6769 | 12.774 -> 6.178 | 6.229 |
| Qwen3-8B | e4m3 | 8688.4 | 0.998x | 0.996x | 15086 | 0.6766 | 12.774 -> 6.186 | 6.229 |

**The hybrid gain is not a short-run artefact.** 1.127x / 1.123x / 1.092x at 350
steps on C4 against 1.140x / 1.121x / 1.093x at 50 steps on mock data: the two
larger models agree to 0.2%, and Llama2-7B's 1.1% spread is roughly the BF16
baseline mismatch noted above. The e4m3 rows likewise reproduce their own 1.02x /
1.02x / 1.00x. So §4's conclusion — that the recipe is worth ~9% and the rest of
the MI325X gap is not — holds on real text at 7x the length.

**Both recipes learn the same.** Held-out validation, hybrid minus e4m3: +0.0002
(Llama2-7B), +0.0017 (Llama3.1-8B), +0.0002 (Qwen3-8B). The E5M2 backward costs
nothing detectable at this horizon. Against BF16 both FP8 arms land within
±0.012, and in inconsistent directions — Llama3.1-8B's FP8 rows are *below* its
BF16 row — which is the signature of run-to-run scatter rather than a precision
effect. Train loss is identical to three decimals on Llama2-7B and differs by
0.008-0.009 on the other two, again unsigned. No NaN, no skipped iteration,
across all six runs.

The `held-out val` column is the end-of-training validation-set number, the same
basis as the table above, so the two tables can be compared directly.

### Two data-sizing constraints, both learned the hard way

The first attempt at this run died at iteration 210 of 350 with a bare
`StopIteration` out of the dataloader. Both causes are worth recording, because
neither announces itself until the run is hours in:

1. **The training set has to cover the step count.** 350 steps x 256 sequences
   asks for 89,600 samples; the C4 file holds 81,294. `PretrainTextDataset` now
   takes `allow_repeat`, set only for the training set, so it wraps to the
   requested count instead of ending the run — here that means the last 10% of
   the run sees data it has seen once before, which is fine for a paired
   comparison and is logged as `data repeats 1.10 times`.
2. **The validation set is consumed, not re-read.** Each eval pass takes
   `EVAL_ITERS x GBS` *fresh* sequences and the validation dataset deliberately
   does not wrap, so a held-out loss is never averaged over duplicated samples.
   The budget is `(TRAIN_STEPS / EVAL_INTERVAL + 1) x EVAL_ITERS x GBS`. That is
   what actually killed the first attempt: iteration 210 was the sixth eval, and
   the file only had five in it.

The binding constraint is Llama3.1-8B, whose tokenizer turns the held-out file
into 1,228 samples of 8193 tokens (against 2,890 for Llama2-7B at 4097). This
run therefore uses `EVAL_INTERVAL=50` and `EVAL_ITERS=1`: eight eval points
costing 1,024 samples, leaving about 1.6 evals of headroom. Raising either knob,
or the step count, needs that arithmetic redone first.

## 13. Reproducing

```bash
# full 9-cell matrix
FP8_FORMAT=fp8_e4m3 bash examples/scripts/run_precision_matrix.sh

# one slice
MODELS=qwen3_8b PRECISIONS="bf16 mxfp4" bash examples/scripts/run_precision_matrix.sh

# noise floor
REPEAT_TAG=rep2 PRECISIONS=mxfp4 bash examples/scripts/run_precision_matrix.sh

# FP8 on the native linear path (§11)
FP8_FORMAT=fp8_e4m3 FP8_LUMEN_LINEAR=1 REPEAT_TAG=lumlin \
  PRECISIONS=fp8 bash examples/scripts/run_precision_matrix.sh

# the 350-step C4 matrix of §12, with live wandb
TRAIN_STEPS=350 LOG_TAG=c4_350 CACHE_TAG=c4_350 FP8_FORMAT=fp8_e4m3 \
  EVAL_INTERVAL=50 EVAL_ITERS=1 \
  WANDB_PROJECT=lumen-precision-matrix-mi350x-c4 \
  TRAIN_JSONL=$PWD/examples/qwen3/results/c4_data/c4_train_1k.jsonl \
  VALID_JSONL=$PWD/examples/qwen3/results/c4_data/c4_valid_heldout.jsonl \
  bash examples/scripts/run_precision_matrix.sh

# the table
python examples/scripts/collect_precision_matrix.py --markdown
```

The FP8 cells added on 2026-08-14 differ only in `FP8_FORMAT` and the tag. `na`
marks the newer AITER of §2, `hyb` the hybrid recipe; drop `FP8_FORMAT` entirely
and `train_pretrain.sh` defaults to `hybrid`.

```bash
# §4 core FP8, both recipes            (add MODELS=... to slice)
FP8_FORMAT=hybrid   LOG_TAG=na_hyb PRECISIONS=fp8 bash examples/scripts/run_precision_matrix.sh
FP8_FORMAT=fp8_e4m3 LOG_TAG=na     PRECISIONS=fp8 bash examples/scripts/run_precision_matrix.sh

# §4 BF16 controls on the same build — needed for any FP8 ratio quoted here
LOG_TAG=na PRECISIONS=bf16 MODELS="llama2_7b qwen3_8b" \
  bash examples/scripts/run_precision_matrix.sh

# §11 native linear, hybrid
FP8_FORMAT=hybrid FP8_LUMEN_LINEAR=1 LOG_TAG=na_hyb_lumlin \
  PRECISIONS=fp8 MODELS="llama2_7b qwen3_8b" bash examples/scripts/run_precision_matrix.sh

# §12 C4 350-step, hybrid (same corpus and eval budget as the 350-step matrix above)
TRAIN_STEPS=350 LOG_TAG=na_hyb_c4_350 CACHE_TAG=na_hyb_c4_350 FP8_FORMAT=hybrid \
  EVAL_INTERVAL=50 EVAL_ITERS=1 PRECISIONS=fp8 \
  TRAIN_JSONL=$PWD/examples/qwen3/results/c4_data/c4_train_1k.jsonl \
  VALID_JSONL=$PWD/examples/qwen3/results/c4_data/c4_valid_heldout.jsonl \
  bash examples/scripts/run_precision_matrix.sh
```

`LOG_TAG` and `REPEAT_TAG` are interchangeable for the log name — the harness
takes `${REPEAT_TAG:-${LOG_TAG:-}}` — and these cells were run with `LOG_TAG`.

`CACHE_TAG` names only the MXFP4 autotune cache, so on an FP8-only slice it is
inert — it is set above only so the command stays copy-pasteable for a slice that
does include MXFP4. Getting the AITER build right is the part these commands do
*not* capture — see §2 and the bug note for the two pieces that have to be
back-ported onto the upstream pin.

`collect_precision_matrix.py` hides tagged runs by default so the core matrix
stays readable; pass `--all-tags` to see the cells above.

Cells whose log already exists are skipped, so the matrix is restartable; `FORCE=1`
re-measures one anyway. Runs are strictly sequential — each wants all 8 GPUs, and
two at once would make every number wrong rather than just slow. `DRY_RUN=1`
prints the `torchrun` command without touching a GPU, which is how to diff two
arms' command lines before spending an hour measuring them.

Run index with status and wall time: `examples/precision_matrix_runs.tsv`.
Logs: `examples/{llama2,llama31,qwen3}/results/lumen_*.log`.

### W&B

The 50-step matrix ran without `--wandb-project`, so Megatron never built its
wandb writer and those runs had no dashboard. All 20 were replayed out of their
logs afterwards into
[`lumen-precision-matrix-mi350x`](https://wandb.ai/daixindi-amd/lumen-precision-matrix-mi350x):

```bash
bash examples/scripts/backfill_precision_matrix_wandb.sh      # DRY_RUN=1 to check first
```

The §12 C4 runs did not need this: they set `WANDB_PROJECT` at launch and streamed
live into
[`lumen-precision-matrix-mi350x-c4`](https://wandb.ai/daixindi-amd/lumen-precision-matrix-mi350x-c4),
so their timestamps and durations are real. The 2026-08-14 FP8 re-measurement is
the exception in both directions: it ran without `WANDB_PROJECT`, so none of those
16 cells has a dashboard, live or backfilled. Their numbers come from the logs and
the collector only. Note that `--wandb-project` alone is
not enough — Megatron gates every `wandb_writer.log()` behind the *TensorBoard*
writer, so without `--tensorboard-dir` the run appears in the project, connects,
and uploads nothing. `train_pretrain.sh` sets both together for that reason.

Each backfilled run carries `backfilled_from_log` in its config, and its
wall-clock timestamps are the replay's rather than the training run's — step time comes from
Megatron's own per-iteration number, so the curves are the run's, but do not read
the run duration or start time as real.

Every non-BF16 run also carries `delta/lm_loss_vs_baseline` and
`delta/speedup_vs_baseline` against the same model's BF16 arm. Those are paired:
every cell used SEED=1234 over the same corpus, so at a given iteration both arms
have seen the same batches. The BF16 repeats are paired against the first BF16 run,
which makes the noise floor of §10 readable as a curve rather than a single number.

MXFP4 run names encode the GEMM table and the assembly coverage it bought, since
§7 shows that is worth 7%-11% and nothing else in the dashboard reveals it:
`mxfp4-tunedtable-asm11of11` reached an AITER assembly kernel for all 11 shapes,
`mxfp4-stocktable-asm{2,3,4}of11` fell back to Triton for the rest. The same pair
is also in config as `a4w4_table` and `a4w4_asm_shapes`, so the dashboard can
group on it. Comparing a `tunedtable` run against a `stocktable` one compares two
kernel sets, not two precisions.

The three stock-table arms are one condition, not two, even though they were
reached differently: Llama2-7B and Llama3.1-8B were measured before their tuned
CSV existed, Qwen3-8B by removing the CSV it already had. Their coverage differs
(2, 4 and 3 of 11) only because AITER's stock table happens to cover each model's
shapes differently.

## 14. Conclusions

1. **MXFP4 is the fastest training precision available here on all three models**:
   1.524x (Llama2-7B), 1.435x (Llama3.1-8B), 1.406x (Qwen3-8B) over BF16, and
   1.28x-1.36x over the best FP8 recipe, at slightly *lower* peak memory. Margins
   are far outside the 1.53% noise floor.
2. **A model without a complete tuned A4W4 table gives up 7%-11%** of that. Any
   new model, or any change to MBS / sequence length / TP, needs its shapes
   collected and tuned, and the `gemm` column checked for `asm N/N`.
3. **FP8 delayed is worth 1.09x-1.14x on gfx950, against 1.38x-1.43x on MI325X**
   — and only with the `hybrid` recipe; `fp8_e4m3` gives 1.00x-1.04x, inside the
   noise floor. The recipe alone is 1.098x of that, 8.9% off the step time and
   about a third of the gap to MI325X; §4 shows the mechanism is which wgrad
   routine the dtype test selects rather than the arithmetic itself. Re-running
   the whole matrix in the coworker's own Docker image reaches 1.15x-1.20x, so
   **15%-19% is the real gap** and it is not the recipe, the AITER, the fusion
   switches, the integration path, or the software environment (§9). The surviving
   candidate is that FP8's quantization and GEMM kernels never got gfx950 tuning.
   This is still the highest-value follow-up in this report, and it is not an
   MXFP4 issue — MXFP4 reaches 1.41x-1.52x through the same linears.
4. **Use `hybrid`, and put the arm on `--lumen-linear`.** Hybrid takes 8.9% less
   step time than e4m3 at equal memory and no measurable accuracy cost over 350
   C4 steps (§12). The native path costs nothing in step time and drops memory from
   +6.2% over BF16 to +0.7% on Llama2-7B and from +3.6% to -0.7% on Qwen3-8B (§11).
5. **The hybrid crash was an AITER version problem, not a Lumen one** (§8.1). The
   `hipb_mm` guard that rejected E5M2 x E4M3 operands is gone upstream; with that
   build the reference report's recipe runs, which is what makes the comparison in
   point 3 a like-for-like one.
6. **The step-time result holds on real data at 7x the length** (§12): MXFP4 at
   1.525x, 1.436x and 1.405x over 350 C4 steps against 1.524x, 1.435x and 1.406x
   over 50 mock steps, and FP8 hybrid at 1.127x, 1.123x and 1.092x against 1.140x,
   1.121x and 1.093x. Neither result is an artefact of short runs or random tokens.
7. **This is still not a full accuracy result.** 0 NaN / 0 skipped everywhere, and
   §12's held-out validation loss puts MXFP4 within +0.068 of BF16 on real text —
   but 367 M tokens is far short of the ~1300-step horizon where 8B diverged
   without tail-BF16. MXFP4 accuracy rests on the operator tests and the longer C4
   runs in [`mxfp4_training_report.md`](mxfp4_training_report.md), and on keeping
   the last 5 layers in BF16.

## Changelog

- **2026-08-24** — Explained the MXFP4 spread in §4. Decomposing each model's step
  into quantizable and non-quantizable work shows the 1.406x-1.524x range is an
  Amdahl effect of architecture, not kernel quality: the implied speedup of the
  quantized part is 1.84x / 1.82x / 1.77x, within 4% across the three. Qwen3-8B
  ranks last because 36 layers at sequence 8192 and a 151936-token vocabulary
  leave it the least to quantize. Adds
  `scripts/mxfp4_speedup_composition.py`. No measured number changes.
- **2026-08-19** — Re-ran all 16 FP8 cells plus 6 BF16 controls inside the
  coworker's own environment (their Lumen commit, their AITER `e42f5791a`, their
  Docker image), reported in the section above §1. Hybrid reaches 1.148x-1.179x
  on the 50-step core and 1.147x-1.199x on C4 — better than this report's stack,
  still 15%-19% short of MI325X, and the ratio improves because BF16 slows down
  rather than because FP8 speeds up. This eliminates the software environment as
  the explanation for the gap: the build was theirs. §9 now carries 15%-19% as the
  bound to explain and lists the environment among the ruled-out causes. Required
  gfx950 OCP E4M3/E5M2 support in the fused cast+transpose extension (verified
  bit-exact against PyTorch) and `LUMEN_FUSED_RES_BWD=0` on native-linear cells;
  both are documented there as deliberate compatibility deltas.
- **2026-08-17** — Audited the FP8 optimization switches against the reference
  report's §6/§7 after a question about whether they were all on. They are: §3
  now enumerates the twelve `train_pretrain.sh` exports and the four harness-wide
  ones, and records that `--use-distributed-optimizer`, `--overlap-grad-reduce`
  and `--lumen-attn-backend csrc` are confirmed in every log's parsed arguments.
  The audit also found that the per-op profile §9 was citing had been taken with
  those switches at their defaults (off), so its "forward GEMM never reaches
  hipBLASLt" and "~3.5 ms/layer unfused plumbing" figures describe a different
  configuration than the one measured here; §9 now says so and the wgrad layout
  result is the only per-op evidence left standing. No measured number changes.
- **2026-08-14** — Re-measured every FP8 cell on the newer AITER of §2, after
  finding the original FP8 numbers were taken on a build missing the compiled
  pieces the FP8 path depends on. 16 FP8 cells (core, native-linear, C4 350-step;
  both recipes), 2 BF16 controls, 4 smokes. The `hybrid` recipe of §8.1 no longer
  crashes — the mixed-dtype `hipb_mm` guard is fixed upstream — and it is worth
  1.09x-1.14x over BF16 against e4m3's 1.00x-1.04x, holding to within 0.2% over
  350 C4 steps at no measurable accuracy cost. Revised §2, §3, §4, §5, §6, §8.1,
  §9, §10, §11, §12, §13 and the conclusions accordingly; the MXFP4-vs-BF16
  result is untouched.
- **2026-08-12** — Added §12: the same nine cells re-run for 350 steps on C4 with
  a held-out validation set. Step-time results reproduce within the noise floor;
  the loss curves are real for the first time here. Records the two data-sizing
  constraints that killed the first attempt at iteration 210, and the
  `allow_repeat` fix for the first of them.
- **2026-08-11** — First version. 29 runs on 8x MI350X: 9-cell core matrix, 6
  noise-floor repeats, 3 tuned-table A/Bs, 2 FP8 integration probes, 9 smoke
  passes. Added Llama2-7B and Llama3.1-8B A4W4 tuned tables (16 shapes, all
  bit-exact). Recorded the hybrid-FP8 crash and the flat FP8-vs-BF16 result on
  gfx950.
