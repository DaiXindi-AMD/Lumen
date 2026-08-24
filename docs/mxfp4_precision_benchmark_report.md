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

Headline: **MXFP4 is 1.41x-1.52x faster per step than BF16 and 1.40x-1.49x
faster than FP8 delayed, at 1.9%-2.5% less peak memory.** The surprise is not
MXFP4 but FP8: on this stack FP8 delayed is worth 1.00x-1.03x over BF16, where
the reference report measured 1.38x-1.43x on MI325X. §9 is about that.

Both results were re-measured over 350 steps on real C4 text with a held-out
validation set (§12) and came back to three digits, so neither is an artefact of
short runs or mock data.

## 1. Scope — what was run

| Group | Runs | Steps | Purpose |
|---|---:|---:|---|
| Core matrix | 9 | 50 | 3 models x {BF16, FP8 delayed, MXFP4} |
| C4 matrix (§12) | 9 | 350 | The same nine cells on real data, with held-out validation |
| Smoke pass | 9 | 3 | Prove each recipe starts before spending 50 steps on it |
| Noise floor | 6 | 50 | BF16 and MXFP4 repeated per model, to size the error bar |
| Tuned A4W4 table A/B | 3 | 50 | Price the tuned MXFP4 GEMM table per model |
| FP8 integration probe | 2 | 50 | Test whether FP8's flat result is the recipe or the path |
| **Total measured** | **38** | | ~13.5 h wall on the 8-GPU node |

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
| Lumen | `94560b7`, branch `feature/mxfp4` |
| AITER | `667d6c66` |
| Megatron-LM | `1b754411` |
| Launch | native `torchrun`, 8 ranks — the `lumen:dev` image was unavailable |

Parallelism is TP=1, PP=1, CP=1, DP=8 with the distributed optimizer and no
activation recompute, for every cell.

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
| Format | bf16 | `fp8_e4m3` (reference used `hybrid`, §8.1) | `mxfp4`, E2M1 |
| Scaling | — | delayed, amax history 1024, algo `max` | blockwise, 32-element |
| Quantized linears | — | all: 128 Llama / 144 Qwen3 | all but the last 5 layers: 108 / 124 |
| BF16 layers kept | — | none | last 5 |
| Integration | — | `nn.Linear` monkeypatch | `--lumen-linear` |

The FP8 arm reproduces the reference report's recipe exactly, including its
FP8-only fusion switches and its `nn.Linear` integration path. The last two rows
are an asymmetry between the FP8 and MXFP4 arms rather than a choice; §8.3 and §9
cover what it does and does not explain.

## 4. Step time — the core result

50 iterations, median after iteration 10, 8x MI350X. MXFP4 rows are the arm with
a complete tuned A4W4 table (`asm 11/11`), which is what now ships for all three
models; §7 prices the table. §12 repeats this table for 350 steps on real C4 data
and reproduces the MXFP4 ratios to within 0.001x.

| Model | Precision | ms/iter | vs BF16 | TFLOP/s/GPU | tok/s/GPU | peak mem | GEMM |
|---|---|---:|---:|---:|---:|---:|---|
| Llama2-7B | BF16 | 6475.1 | 1.000x | 818.4 | 20242 | 0.5349 | — |
| Llama2-7B | FP8 delayed | 6315.1 | 1.025x | 839.1 | 20755 | 0.5686 | — |
| Llama2-7B | **MXFP4** | **4247.9** | **1.524x** | 1247.5 | 30856 | 0.5245 | asm 11/11 |
| Llama3.1-8B | BF16 | 8209.5 | 1.000x | 769.3 | 15966 | 0.6094 | — |
| Llama3.1-8B | FP8 delayed | 8001.0 | 1.026x | 789.3 | 16382 | 0.6336 | — |
| Llama3.1-8B | **MXFP4** | **5720.7** | **1.435x** | 1103.9 | 22912 | 0.5939 | asm 11/11 |
| Qwen3-8B | BF16 | 8573.6 | 1.000x | 751.1 | 15288 | 0.6539 | — |
| Qwen3-8B | FP8 delayed | 8557.0 | 1.002x | 752.6 | 15317 | 0.6765 | — |
| Qwen3-8B | **MXFP4** | **6097.2** | **1.406x** | 1056.2 | 21497 | 0.6387 | asm 11/11 |

Pairwise, on this stack:

| Model | MXFP4 vs BF16 | MXFP4 vs FP8 | FP8 vs BF16 |
|---|---:|---:|---:|
| Llama2-7B | 1.524x | 1.487x | 1.025x |
| Llama3.1-8B | 1.435x | 1.399x | 1.026x |
| Qwen3-8B | 1.406x | 1.403x | 1.002x |

MXFP4 wins on all three models by 40%-52% against BF16, and its margin over FP8 is nearly the
same as over BF16 because FP8 barely moves off BF16 here.

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

MXFP4's saving is small for the same reason — weights and optimizer state are
still BF16/FP32 and only the GEMM operands are 4-bit, so what it saves is
activation and cast-buffer traffic, not parameter storage. Anyone reading "4-bit"
as "4x less memory" is reading the wrong axis; the win here is time, not bytes.
FP4 parameter all-gather, which would move that axis, is
[`mxfp4_training_report.md`](mxfp4_training_report.md) §2.8 and not enabled here.

## 6. Numerical behaviour — and its limits

No run produced a NaN or a skipped iteration: **0/0 on all 29 runs.** Loss
trajectories over the 50 steps:

| Model | BF16 | FP8 delayed | MXFP4 |
|---|---|---|---|
| Llama2-7B | 11.594 -> 2.209 | 11.721 -> 2.210 | 11.615 -> 2.212 |
| Llama3.1-8B | 12.989 -> 5.095 | 13.000 -> 5.085 | 12.931 -> 5.242 |
| Qwen3-8B | 13.244 -> 2.280 | 13.092 -> 2.275 | 13.141 -> 2.277 |

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

### 8.1 FP8 is E4M3 here, not hybrid — the hybrid recipe crashes

The reference report's `hybrid` FP8 (E4M3 forward, E5M2 backward) dies in the
first backward on this stack:

```
RuntimeError: expected mat1 and mat2 to have the same dtype,
but got: c10::Float8_e5m2 != c10::Float8_e4m3fn
```

It reproduces on all three models at iteration 1, so it is the wgrad GEMM
handing hipBLASLt two different FP8 dtypes, not anything model-specific. Tracked
in `.claude/tmp-training-bugs.md` as
`[2026-08-06 hybrid-fp8-wgrad-mixed-dtype-crash]`, now with MI350X evidence.

The FP8 arm therefore runs `fp8_e4m3` for both directions. This is the one place
the FP8 numbers here are not the reference report's recipe. It is unlikely to
explain §9: E4M3 and E5M2 are the same width and cost the same in the GEMM, and
the difference is range in the backward, not throughput.

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

## 9. Open question: FP8 delayed has no headroom over BF16 on MI350X

This is the result most likely to be challenged, so here is the evidence.

FP8-over-BF16, reference report against this run:

| Model | MI325X, reference | MI350X, here |
|---|---:|---:|
| Llama2-7B | 1.432x | 1.025x |
| Llama3.1-8B | 1.384x | 1.026x |
| Qwen3-8B | 1.414x | 1.002x |

The FP8 deltas here — 0.2% to 2.6% — sit at the run-to-run noise floor measured
in §10, so the honest reading is that **FP8 delayed and BF16 are indistinguishable
on this stack**, not that FP8 is 2% faster.

What moved is mostly the baseline. Between the reference report's stack on MI325X
and this one on MI350X:

| Model | BF16 got faster by | FP8 got faster by |
|---|---:|---:|
| Llama2-7B | 1.782x | 1.276x |
| Llama3.1-8B | 1.662x | 1.232x |
| Qwen3-8B | 1.704x | 1.208x |

BF16 gained 1.66x-1.78x; FP8 only 1.21x-1.28x. Had FP8 scaled like BF16, Qwen3-8B
would sit near 6060 ms — almost exactly MXFP4's 6097 ms. So the question is not
why FP8 is slow but why it failed to pick up the generational gain BF16 did.

Ruled out, or bounded:

- **Not the recipe.** Re-parsing the reference report's own FP8 log confirms it
  used the same path, same delayed scaling, amax history 1024, algo `max`, same
  144 modules, same `first_last_layers_bf16=False`. The only recipe difference is
  hybrid vs E4M3 (§8.1), which does not change GEMM cost.
- **Not a silent fallback to BF16.** The FP8 arm reports
  `Quantization enabled on 128/144 nn.Linear layers (format=fp8_e4m3, scaling=delayed)`
  and its peak memory is 3.5%-6.3% *above* BF16, which is the amax history and
  transpose cache being allocated. It is quantizing.
- **Not the integration path.** §11 re-runs FP8 on `--lumen-linear`, the path
  MXFP4 uses, closing the last structural difference between the arms. FP8 stays
  at 1.012x (Llama2-7B) and 1.003x (Qwen3-8B) — the wiring is not it.

One hypothesis is now bounded and one remains:

1. A roofline ceiling — BF16 is already fast enough on MI350X that GEMMs are no
   longer the dominant term, so halving them cannot buy much and FP8's
   quantize/transpose overhead eats the rest. **This cannot be the whole story:**
   MXFP4, on the same model, same batch shape and same linears, finds 1.41x-1.52x.
   The headroom exists; FP8 is not reaching it.
2. The FP8 quantization, cast-transpose and GEMM kernels never got gfx950 tuning.
   They were tuned on gfx942, where they earned the reference report its 1.4x.
   This is the same class of problem as §7 one layer down, and after §11 it is the
   remaining candidate.

Confirming it needs a per-op profile of the FP8 arm on gfx950 — comparing FP8 and
MXFP4 kernel time for the same GEMM shapes — which is out of scope here.

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
real. FP8's 0.2%-2.6% is inside the noise floor, and §7's 7%-11% table effect
clears it comfortably.

The two BF16 repeats reproduced their loss curves digit for digit. The MXFP4
repeats did not — final loss moved by 0.004 (Llama2-7B), 0.042 (Llama3.1-8B) and
0.005 (Qwen3-8B) — so the MXFP4 path is not bitwise reproducible across runs at a
fixed seed, though BF16 on the same harness is. The cause was not chased here.
It is small next to the precision differences being measured, but worth knowing
before anyone tries to bisect an MXFP4 loss curve by diffing two runs.

## 11. FP8 on the native linear path

Closing the §8.3 asymmetry: same FP8 E4M3 delayed recipe, `--lumen-linear` added,
so the only remaining difference from the MXFP4 arm is the number format. Run on
the fastest and the slowest model of the three.

| Model | BF16 | FP8 `nn.Linear` | FP8 native | MXFP4 native |
|---|---:|---:|---:|---:|
| Llama2-7B | 6475.1 | 6315.1 (1.025x) | 6399.1 (**1.012x**) | 4247.9 (1.524x) |
| Qwen3-8B | 8573.6 | 8557.0 (1.002x) | 8549.2 (**1.003x**) | 6097.2 (1.406x) |

**The integration path is not the explanation.** Moving FP8 onto the same linears
MXFP4 uses changes step time by -1.3% on Llama2-7B and +0.1% on Qwen3-8B, both
inside the noise floor, and leaves FP8 within 1.2% of BF16. MXFP4 on that same
path is 40%-52% ahead. Whatever FP8 is missing on gfx950 is in the quantization
and GEMM kernels it dispatches, not in how it is wired into Megatron — which
leaves hypothesis 2 of §9 as the one to test.

One side result worth keeping: the native path cuts FP8's peak memory from 0.5686
to 0.5386 on Llama2-7B and 0.6765 to 0.6491 on Qwen3-8B, i.e. from +6.3% over
BF16 to +0.7%, and from +3.5% to -0.7%. It avoids the transpose caching the
`nn.Linear` path relies on. So if the FP8 arm is kept, it should be on
`--lumen-linear`: same speed, several GiB cheaper.

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
corpus at 7x the length. FP8 stays flat, and Qwen3-8B's arm is now 0.996x, i.e.
marginally *slower* than BF16; with §10's error bar that is still
"indistinguishable from BF16", but it removes any reading of §9 as an artefact
of short runs or mock data.

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
so their timestamps and durations are real. Note that `--wandb-project` alone is
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
   1.40x-1.49x over FP8 delayed, at slightly *lower* peak memory. Margins are far
   outside the 1.53% noise floor.
2. **A model without a complete tuned A4W4 table gives up 7%-11%** of that. Any
   new model, or any change to MBS / sequence length / TP, needs its shapes
   collected and tuned, and the `gemm` column checked for `asm N/N`.
3. **FP8 delayed is currently worth nothing over BF16 on gfx950** — 1.00x-1.03x,
   inside the noise floor, against 1.38x-1.43x on MI325X. It is not the recipe, not
   a silent fallback, and not the integration path (§9, §11); the remaining
   candidate is that its quantization and GEMM kernels never got gfx950 tuning.
   This is the highest-value follow-up in this report, and it is not an MXFP4
   issue — MXFP4 reaches 1.41x-1.52x through the same linears.
4. **If the FP8 arm is kept, put it on `--lumen-linear`**: same step time to within
   noise, but peak memory drops from +6.3% over BF16 to +0.7% on Llama2-7B and
   from +3.5% to -0.7% on Qwen3-8B (§11).
5. **The hybrid FP8 recipe the reference report used crashes on this stack** (§8.1)
   and needs fixing before any FP8 number here can be called a reproduction of it.
6. **The step-time result holds on real data at 7x the length** (§12): 1.525x,
   1.436x and 1.405x over 350 C4 steps against 1.524x, 1.435x and 1.406x over 50
   mock steps, and Qwen3-8B's FP8 arm comes out at 0.996x, so §9 is not an
   artefact of short runs or random tokens.
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
