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

### [2026-07-20 mxfp4-8b-no-convergence]
- Symptom: MXFP4 training on Qwen3-8B (random init, C4, FSDP2 8-way) produces no loss reduction at any learning rate. Loss stays at ~12.75 (random init level) then jumps to 11.9375 (= ln(151936), uniform distribution). Tested lr=6e-5, 2e-5, 3e-6 — all fail identically.
- Context: Same setup works perfectly on Qwen3-0.6B (lr=6e-5 trains to val_loss 6.34). BF16 + Lumen attn/norm/rope on 8B works fine (lr=6e-5, loss 12.75→7.28 in 200 steps). The problem is isolated to MXFP4 linear quantization on 8B.
- Possible bug: MXFP4 backward (DGrad or WGrad) produces zero or near-zero gradients at 8B scale. The 4-bit quantization + RHT + packed transpose chain may be truncating gradient signal when matrix dimensions are 4096×4096+ (vs 1024×1024 at 0.6B). Key suspect areas:
  1. `transpose_packed_fp4` on (4096, 6144/2) weight — nibble packing correctness at large N
  2. WGrad `hadamard_transform` with G=32 on M-dim after transpose — may amplify quantization error at larger M
  3. `gemm_afp4wfp4` numerical accuracy at 4096×4096 — AITER kernel may have a shape-dependent bug
  4. E8M0 scale underflow: 4096-dim vectors may have smaller per-block amax, pushing scales to minimum and zeroing out gradients
- Evidence so far:
  - 0.6B MXFP4 trains correctly (val_loss 6.34, matching BF16 within +0.045)
  - 8B BF16 trains correctly (val_loss 6.05 in 5000 steps)
  - 8B BF16 + Lumen attn/norm/rope trains correctly (loss 12.75→7.28)
  - 8B MXFP4 at lr=3e-6 (20x lower than BF16) still shows zero learning
  - 12/12 MXFP4 op tests pass at small matrix sizes (128×256)
- Next check: Single-layer gradient comparison: run one forward+backward on a single 8B linear layer (4096×4096), compare MXFP4 grad vs BF16 grad. Check if grad is all-zero or has NaN. Check `transpose_packed_fp4` output at 4096×6144 scale.
- Status: open

## Ruled Out

Move disproved suspicions here instead of deleting them.

## Resolved

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
