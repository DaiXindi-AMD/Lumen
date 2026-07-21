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

### [2026-07-20 mxfp4-8b-fsdp2-nan-grads]
- Symptom: MXFP4 8B with FSDP2 multi-GPU produces NaN/Inf gradients in **every** layer on step 1 (without gradient checkpointing). With gradient checkpointing, training runs for ~1250 steps then spikes to loss=11.9375.
- Root cause: **FSDP2 save_for_backward incompatibility.** MXFP4 forward saves packed FP4 weight + scale via `ctx.save_for_backward()`. FSDP2 `full_shard` reshards weights after forward. When backward runs, the saved FP4 weight tensor references the resharded (partial) data, but MXFP4 backward (`transpose_packed_fp4`, `gemm_afp4wfp4`, `convert_from_mxfp4`) expects the full weight — operating on stale/partial data produces NaN.
- Evidence:
  - 1 GPU, no FSDP, no grad_ckpt: **0 NaN, 0 Inf** (all 399 grads OK)
  - 1 GPU, FSDP2, no grad_ckpt: **0 NaN, 0 Inf** (world_size=1, no resharding)
  - 2 GPU, FSDP2, no grad_ckpt: **397 NaN grads on step 1**
  - 8 GPU, FSDP2, no grad_ckpt: **397 NaN grads on step 1**
  - 8 GPU, FSDP2, WITH grad_ckpt: trains ~1250 steps then spikes (grad_ckpt recomputes forward, partially avoiding the stale-tensor issue, but doesn't fix it)
  - BF16 8B with FSDP2: works perfectly (BF16 backward doesn't use save_for_backward FP4 data)
- NOT the cause: Hadamard sign (random vs deterministic), SR vs RTN, learning rate, Fprop Hadamard — all tested and ruled out
- Fix needed: MXFP4 backward must access FSDP2 all-gathered weight, not saved FP4 weight. Options: (a) save `ctx.weight_ref` and re-quantize weight in backward after FSDP2 all-gather; (b) use `ctx.save_for_backward` with FSDP2-aware tensors; (c) always recompute weight quantization in backward (like gradient checkpointing does implicitly).
- Status: open

## Ruled Out

Move disproved suspicions here instead of deleting them.

## Resolved

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
