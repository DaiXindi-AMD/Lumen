# Temporary Training Bug Notes

This file lives at `.cursor/tmp-training-bugs.md` relative to the `Lumen` repo root. Read the whole file at the start of every new Lumen training debug session.

Use it to keep track of possible bugs found during testing. Do not treat any entry here as proof. Re-check against the current reference diff and current repro before acting.

Treat any fresh return to the same debugging problem as a new debug session:

- a new chat or agent session
- a new day or work block
- returning after unrelated work
- starting a new round of debug after prior tests finished

Write back only meaningful tests or experiments that change confidence in a hypothesis, such as a new repro, written diff, backend toggle, layerwise compare, kernel test, or targeted integration check. Do not log every identical rerun. Do log negative results that rule a suspicion out.

## Open

### [2026-08-06 fp8-per-tensor-path-slower-than-bf16]
- Symptom: Qwen3-8B FP8 (e4m3, delayed) ran at 3178 ms/step against MXFP4's 1521 ms on the same 8×MI350X, same everything else. Two separate causes found, one fixed, one still open.
- Identified cause, deliberately left unfixed (worth ~166 ms/step of GPU plus the logging): `fast_transpose_fp8` falls back to `.t().contiguous()` because the installed aiter at `/home/xdai/aiter` has no `ops/triton/quant/fast_transpose.py` — Lumen vendors it under `third_party/aiter/` and nothing copies it over, the same gap the launcher already papers over for `cross_entropy`. The fallback is 3.8-4.2× slower than the kernel at the wgrad shapes (0.567 vs 2.296 ms for one layer's four linears), and it logs per call: 709k warnings, 78 MB log, ~3000 lines/step across 8 ranks. Copying the two vendored files in makes the probe return True and the kernel is bit-identical to `.t().contiguous()` at (16384,4096), (16384,12288) and (4096,16384) — but the copy was reverted and the fallback left as-is, because FP8 is being held fixed as the comparison baseline. Fix it only as a deliberate FP8 change, not as a side effect of MXFP4 work.
- Second cause: even with the transpose kernel in place, one fwd+bwd through `quantized_linear` at Qwen3-8B shapes costs **21.8 ms in FP8 delayed against 16.4 ms in BF16 and 13.5 ms in MXFP4** (sum over qkv/attn_out/gate_up/down, 16384 tokens). FP8 is the only one of the three that loses to BF16. The GEMM is not the problem: `hipb_mm` on `float8_e4m3fn` does 16384×4096×4096 in 0.279 ms (1970 TFLOP/s). So the loss is in quantization, transposes, and amax bookkeeping around the GEMM, not in the GEMM.
- Next check: profile one FP8 linear fwd+bwd kernel by kernel and compare against the MXFP4 breakdown in `docs/mxfp4_training_report.md` §5.12. The MXFP4 path got a fused dual-layout quantizer that emits both the DGrad and WGrad operands from one read; FP8 still quantizes, then transposes, then re-quantizes per operand, which is the shape of the gap.
- Bearing on the MXFP4-vs-FP8 comparison: any "MXFP4 is N× faster than FP8" number from this stack is measured against an unoptimized FP8 path, not against FP8 as a format. State it that way.
- Kernel-by-kernel profile done (gate_up, 16384 tokens, one fwd+bwd): BF16 **8.39 ms / 4 launches**, FP8 delayed **10.89 ms / 29 launches**, MXFP4 **4.59 ms / 14 launches**. FP8's GEMMs sum to 6.80 ms and everything else is **4.08 ms**: unfused `clamp` (0.860) + `elementwise_manual_unroll` (1.102, the `.t().contiguous()` fallback) + type convert (0.630) + `ConvertToFloat8E4M3fnOp` (0.582) + `_amax_abs_kernel` (0.505) + `data_to_scale` (0.192). MXFP4 does the same job in one `_dual_layout_quant_mxfp4_kernel` at 0.551 ms. So ~3.5 ms/layer/micro-batch is unfused quantization plumbing.
- Third cause, new and larger than expected — **the FP8 GEMMs run at BF16 speed because the operands are in the wrong layout**. Tracing `hipb_mm` through one `quantized_linear` fwd+bwd shows only two calls reach hipBLASLt (dgrad and wgrad); the **forward never gets there at all** and lands on Triton `_gemm_a8w8_kernel` (1.877 ms, 1758 TF/s). Of the two that do, dgrad passes B as `(24576,4096)` stride `(4096,1)` and wgrad passes A as `(24576,16384)` stride `(1,24576)` — K is strided in both, so hipBLASLt cannot use the TN FP8 matrix-core path. Measured at the dgrad shape: as-called **2.201 ms / 1498 TF/s** vs TN **1.328 ms / 2483 TF/s** (1.66×); at the wgrad shape: as-called **2.914 ms / 1132 TF/s** vs TN **1.277 ms / 2583 TF/s** (2.28×).
- Ruled out — it is *not* hipBLASLt heuristic/tuning. Scanning every `hipb_findallsols` candidate against the default `sol=-1` at four Qwen3-8B shapes gains only 1.03–1.10×, and `sol=-1` in TN layout already does gate_up fwd at 1.436 ms / 2296 TF/s versus BF16's 2.659 ms / 1241 TF/s. FP8 hardware is delivering ~1.85× BF16 when fed correctly; a tuned solution table would not have found this.
- The layout problem and the transpose fallback are the same root cause: with no `fast_transpose` in the installed aiter, the path can neither materialize a K-contiguous transposed operand cheaply nor avoid handing the strided view straight to the GEMM.
- Headroom if all three were fixed: 3 × ~1.33 ms TN GEMM + a fused quantizer ≈ **4.6 ms against BF16's 8.39** — FP8 should be ~1.8× faster than BF16 instead of 1.30× slower. None of this is a property of the FP8 format.
- Environment mismatch found, and it is *not* the main cause. `third_party/aiter` pins `e42f5791a` ("FP8 GEMM + quant/attention opts for Lumen", 2026-07-21) but the machine imports `/home/xdai/aiter` at `667d6c669` (2026-07-15), a different repo missing `quant/fast_transpose.py`, `_triton_kernels/quant/fast_transpose.py`, `_triton_kernels/quant/quant_fp8_blockwise.py` and `_triton_kernels/quant/quant_mxfp8.py`. The gap is bad enough that **`origin/main` cannot even be imported here** — `lumen/ops/quantize/ops.py:21` hard-imports `quant_fp8_blockwise`; only our try/except makes the tree importable. Copying the four files in (same untracked-file pattern the launcher already uses for `cross_entropy`) makes `_transpose_2d_kernel` run at 0.087 ms in place of the 1.102 ms `.t().contiguous()` pair. **Verified bit-identical**: out/dX/dW hashes `a12ddb18144a5d7d` / `61813d5c850a5a6e` / `02d771088dea1f8a` before and after, layer 11.078 → 10.677 ms.
- But it only buys 5%: gate_up fwd+bwd goes 10.886 → 10.359 ms and **the three GEMMs are untouched** (2.773 / 2.083 / 1.832, forward still on Triton). So the layout and unfused-quant causes are in Lumen's FP8 code, not in the environment. Correct aiter alone does not make FP8 beat BF16.
- **2026-08-24 solution-only tuning, with FP8 implementation frozen.** An external tuner enumerated every `hipb_findallsols` candidate for the four Qwen3 forward and four hybrid-wgrad calls, timed with CUDA events, and checked outputs against `sol=-1`. The forward candidates looked 1.03x-1.28x faster in isolation but both contiguous-transpose and strided-transpose tables caused illegal GPU memory access in a full 8-GPU first forward, so they are rejected. The four wgrad candidates passed a 3-step 8-GPU forward/backward smoke and a 30-step run: median iterations 11-30 **8188.4 ms** versus the untuned 50-step median **8335.5 ms**, a **147.1 ms / 1.018x** gain. Against the paired BF16 9571.6 ms, Qwen FP8 improves from 1.148x to **1.169x**. The tuned run completed 30/30 with zero NaN/skipped. No layout, fusion, recipe, or FP8 code was changed; only the existing `LUMEN_TUNED_GEMM` hook selected wgrad solution indices.
- Status: open — root cause identified in three parts; the aiter part is fixable without touching FP8 code, the other two need changes to the FP8 path, which is owned by another author and held as the frozen comparison baseline. Hand the layout finding upstream rather than patching locally.

### [2026-08-06 hybrid-fp8-wgrad-mixed-dtype-crash]
- Symptom: Qwen3-8B with `--linear-fp8 --linear-fp8-format hybrid --linear-fp8-scaling delayed` dies in the first backward on all 8 ranks: `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::Float8_e5m2 != c10::Float8_e4m3fn`, at `lumen/ops/quantize/linear.py:840` (`gemm_wgrad_mixed` → `hipb_mm`).
- Possible bug: `gemm_wgrad_mixed` is written for exactly this pair (docstring: "grad_fp8 is E5M2, input_fp8 is E4M3"), but its hipBLASLt path passes both operands to `hipb_mm`, which on the installed AITER rejects mismatched FP8 dtypes. The mixed-dtype Triton kernel (`gemm_a8w8_mixed`) sits below it as the `_probe_aiter_hipblas()`-false branch only, so a machine that *has* hipBLASLt can never reach the fallback. Hybrid FP8 backward therefore looks unreachable on this stack.
- Evidence so far: 4-layer 5-step smoke, 8×MI350X, `/tmp/smoke_fp8.log` (first failure at line 1260). Same smoke with `--linear-fp8-format fp8_e4m3` (single dtype, everything else identical) completes 5/5 iterations, loss 12.0→11.3, 0 NaN.
- Next check: confirm whether the installed `hipb_mm` binding ever accepted mixed FP8 dtypes, then either route mixed pairs to `gemm_a8w8_mixed` regardless of hipBLASLt, or catch the dtype RuntimeError as a logged fallback. Benchmark before adopting: the Triton mixed kernel may be slow enough that hybrid should instead be rejected at config time rather than silently made slow.
- Not blocking the MXFP4-vs-FP8 comparison, which uses `fp8_e4m3` delayed.
- 2026-08-11: reproduces on **Llama2-7B** too (3-step smoke, 8×MI350X, `examples/llama2/results/lumen_llama2_7b_smoke_fp8.log:4871`, same `Float8_e5m2 != c10::Float8_e4m3fn` out of `aiter/jit/core.py` wrapper on all 8 ranks). So it is model-independent, not a Qwen3 shape thing, and it lands via Megatron's `--fp8-format hybrid` as well as `--linear-fp8-format hybrid` — `_FP8_FORMAT_MAP` sends both to the same resolved format.
- Bearing on the BF16/FP8/MXFP4 precision matrix: the reference FP8 report (`ref/Lumen-fp8-test-report-for-reference-only.md` §8) ran hybrid delayed on MI325X, so the recipe cannot be reproduced on this stack at all. The matrix runs its FP8 arm at `FP8_FORMAT=fp8_e4m3` and the report has to say so — an E4M3 backward is a different numerical recipe, not just a different flag.
- Status: open

## Ruled Out

Move disproved suspicions here instead of deleting them.

### [2026-08-02 lumen-vs-te-mxfp4-convergence-gap]
- Suspicion: Lumen's MXFP4 implementation converges better than TE's MXFP4. At each stack's own defaults Lumen led by 0.069 nats held-out at 1000 steps (Qwen3-8B, C4, 8×MI350X, seed/corpus/order shared), and the paired per-step gap widened monotonically.
- Ruled out by: 4-arm ladder, all sharing seed 1234 and sample order. A = Lumen (tail 5, H on wgrad operands), B = TE (no tail, no H), C = TE (tail 5, no H), D = TE (tail 5, H on all operands via `NVTE_MXFP4_USE_HADAMARD=1`).
- Evidence (paired Δ, second-half mean): C−B = −0.0058 (BF16 tail), D−C = −0.0565 (Hadamard), A−D = −0.0043 (everything else). Sums to the observed −0.0665. Held-out val@1k: A 5.7793 vs D 5.7833.
- Conclusion: the gap was ~85% recipe (Hadamard), 9% BF16 tail, ~0.004 nats residual. No evidence of an implementation-level convergence advantage.
- Secondary finding: TE rotates every operand (fprop activation/weight included), Lumen only the two WGrad operands, and the wider rotation did *not* help — D is 0.004 nats behind A. Consistent with FP4 range overflow being gradient-side. Do not widen Lumen's Hadamard scope on the theory that more rotation is better.
- Still open (throughput, not correctness): TE is 1.29× faster per step at matched recipe (1555 vs 2007 ms); the fused norm/rope/attention gap is not separated from the quantized linears.
- Status: ruled out

## Resolved

### [2026-07-21 mxfp4-8b-late-loss-spike]
- Symptom: 8B MXFP4 (after FSDP2 fix) crashes at step 1275-3600 (varies by config). Loss spikes to 11.9375.
- Root cause: FP4 dynamic range overflow (max=6.0) on late-training outliers + all layers quantized including sensitive末尾layers.
- Fix: Deterministic Hadamard H16 (G=32→16, sign=all+1) + last 5/36 layers kept in BF16 (`first_last_layers_bf16=True, num_layers_at_end_in_bf16=5`). Per arXiv:2605.09825 + arXiv:2509.25149.
- Verification: 8B MXFP4, lr=1e-4, 5000 steps, zero divergence, val_loss 7.07→5.74.
- Status: resolved

### [2026-07-20 mxfp4-8b-fsdp2-nan-grads]
- Symptom: MXFP4 8B with FSDP2 multi-GPU: 397/399 NaN grads on step 1 (without grad ckpt).
- Root cause: FSDP2 reshards weight after forward; saved FP4 weight references stale data.
- Fix: backward re-quantizes from ctx.weight_ref (BF16 master, all-gathered by FSDP2).
- Status: resolved

### [2026-07-20 mxfp4-8b-no-convergence]
- Symptom: MXFP4 training on Qwen3-8B appeared to produce zero learning at any lr (6e-5, 2e-5, 3e-6). Loss stayed at ~12.75 then jumped to 11.9375 (uniform distribution).
- Root cause: **Not a code bug — hyperparameter sensitivity.** MXFP4's 4-bit quantization noise requires a higher lr (~1e-4) and shorter warmup (~50 steps) than BF16 at 8B scale. With lr=6e-5 and 200-step warmup, the signal-to-noise ratio during warmup was too low for the model to escape the initial random plateau.
- Evidence:
  - Single-layer gradient test at 4096×4096: MXFP4 gradients are correct (norm matches BF16, zero%=0, no NaN)
  - Single-GPU 8B MXFP4 without FSDP: trains perfectly (12.76→7.33 in 50 steps, lr=1e-4, clip=1.0)
  - Multi-GPU 8B MXFP4 with FSDP2: trains perfectly with lr=1e-4, warmup=50 (12.8→6.2 in 200 steps, val_loss 7.08)
  - Same FSDP2 setup fails with lr=6e-5, warmup=200
  - 0.6B MXFP4 tolerates lr=6e-5 because smaller models have smoother loss landscapes
- Fix: Use lr=1e-4, warmup=50 steps, grad_clip=1.0 for 8B MXFP4 pretraining. The 0.6B default of lr=6e-5 does not transfer to 8B.
- Status: resolved

### [2026-06-26 bw2d-fused-swiglu-quant-1d-scale]
- Symptom: 70B blockwise2d LoRA (run_blockwise2d_v2.sh, image lumen/llama2:dev) dies in forward at `linear_fc2` with `IndexError: Dimension out of range (expected [-1,0], got 1)` at `lumen/ops/quantize/linear.py:_gemm_blockscale_bpreshuffle` → `scale_a.transpose(0, 1)`.
- Root cause: `LUMEN_FUSED_SWIGLU_QUANT=1` was active and *working* — fc2 consumed the fused cache — but `try_fused_swiglu_fp8` (`lumen/models/_swiglu_fp8_fuse.py`) hard-coded `dynamic_per_tensor_quant_fp8_i8_with_amax`, producing a per-tensor **1D** scale `(1,)`. fc2's `scaling_type="blockwise2d"` routed to `gemm_blockscale` → bpreshuffle, which expects a 2D `(M, K/128)` scale and called `transpose(0,1)` on the 1D scale. Structural mismatch: the fusion bridge only ever implemented per-tensor granularity. (`_fp8_store_activation`/`_SwiGLU_FP8Store` also use dynamic, but those only serve backward — they return bf16 to fc2, not the GEMM scale.)
- Fix (3 files):
  1. `_swiglu_fp8_fuse.py`: added `set_fused_swiglu_scaling(scaling_type, block_size)` + module state; `try_fused_swiglu_fp8` now dispatches on scaling_type — blockwise/blockwise2d uses `get_hip_quant(QuantType.per_1x128)` → 2D scale (matching `_quant_blockwise2d_activation`), amax=None; per-tensor path unchanged. Skips fusion if block_size≠128 or width not divisible.
  2. `lumen/models/megatron.py`: `enable_fp8_for_parallel_linear` calls `set_fused_swiglu_scaling(scaling_type, block_size)` so the bridge knows the global granularity.
  3. `lumen/ops/quantize/linear.py`: `gemm_blockscale` bpreshuffle except now also catches `IndexError, ValueError` → graceful CK/Triton fallback instead of crashing the run.
- Verification: run_blockwise2d_v2.sh completed all 30/30 steps, loss converged to ~1.88, 0 NaN / 0 skipped iters, ~7.47 s/iter, no IndexError / memory fault / type error.
- Note: 4 stale `lumen/llama2:dev` checkpoint-conversion containers (convert_to_megatron*, failed 11-12h ago on TESpecProvider / megatron.core.mpu) were concurrently mutating the repo (reverting core.py, churning scripts) — removed before the verifying run.

## Entry Template

```markdown
### [YYYY-MM-DD session-name]
- Symptom:
- Possible bug:
- Evidence so far:
- Next check:
- Status: open | ruled out | resolved
```
