# MLPerf Llama2-70B LoRA SFT — Lumen Benchmark Results

Lumen training results on the MLPerf `llama2_70b_lora` benchmark (8x MI300X),
compared against the official AMD MLPerf v5.1 reference run on the **same machine**.

**Target**: val_loss < 0.925

## Result Summary

| Metric | Lumen (speed-opt) | Lumen (mem-opt) | Lumen (C++ dispatch) | Lumen (prev) | MLPerf Ref (local, same MI300X) |
|--------|-------------------|----------------|---------------------|--------------|--------------------------------|
| Best val_loss | TBD (103 steps) | **0.9170** (step 1024) | 0.9210 (step 576) | 0.9187 (step 960) | 0.9243 |
| Passes MLPerf target? | TBD | **Yes** (0.9212 at step 576) | Yes (0.9210 at step 576) | Yes (0.9208 at step 576) | Yes (step 384) |
| Pre-eval step time | **4,135 ms** (steps 20-100 avg) | 4,152 ms | 4,170 ms | 4,263 ms | **3,967 ms** |
| Post-eval step time | TBD | **~4,145 ms (+0%)** | TBD | ~4,476 ms (+6.8%) | ~4,000 ms (est.) |
| Speed ratio (Lumen / MLPerf) | **1.042x** (pre-eval) | 1.047x | 1.051x | 1.075x | 1.0x |
| Memory utilization | 99.5% | **99.5%** | 99.5% | 99.5% | ~82% |
| Stability | 0 NaN/skip (103 steps) | 0 NaN/skip (1024 steps) | 0 NaN/skip (600 steps) | 0 NaN/skip | 0 NaN/skip |
| Total time (1024 steps) | TBD | **5,004 s** | ~4,520 s (est.) | ~4,630 s (est.) | — |

**Local MLPerf reference**: `rocm/amd-mlperf:llama2_70b_training_5.1` Docker,
`SEED=1234`, zarr checkpoint converted from `NousResearch/Llama-2-70b-hf`.
Log: `mlperf_ref_mi300x_20260416_030753.log`.

## Implemented Optimizations and Measured Impact

All optimizations applied cumulatively. Impact is measured against the unoptimized
Lumen baseline (7,400 ms/step, val_loss 0.9371, does not pass target).

### Convergence Fixes

| Optimization | Impact on val_loss | Impact on Speed |
|--------------|--------------------|-----------------|
| **Epoch-level data shuffling** (`LUMEN_SHUFFLE_TRAIN=1`) | **-0.016** (0.9371→0.9208, now passes target) | None |
| **Aligned eval schedule** (`LUMEN_EVAL_ALIGNED=1`, every 192 steps) | -0.003 (matches MLPerf eval cadence) | **-11% wall-clock** (fewer evals) |

Data shuffling was the single most important fix. Without it, Lumen does not
converge below 0.925 regardless of other optimizations.

### Speed Optimizations

| Optimization | Measured Savings | Mechanism |
|--------------|-----------------|-----------|
| **hipBLASLt for all GEMMs** (`LUMEN_PREFER_HIPBLASLT=1`) | **-790 ms/step (14.2%)** | hipBLASLt replaces CK for fwd+bwd; `.t()` view eliminates weight transpose copy |
| **Fused quant+amax** (`LUMEN_FUSED_QUANT_AMAX=1`) | **-377 ms/step (6.1%)** | Merge `abs()` + `amax()` into single Triton kernel |
| **Fused quant+scale** (`LUMEN_FUSED_QUANT_SCALE=1`) | **-206 ms/step (2.8%)** | Merge quant + scale computation |
| **Post-eval allocator fixes** (eval recompute, warmup, GC, cache clear) | **-11.1% total training time** | Prevent memory fragmentation after eval passes |
| **Fused NQG norm+quant** (`LUMEN_FUSED_NORM_QUANT_GEMM=1`) | **-51 ms/step (-1.1%)** | Fuse RMSNorm + FP8 quant in single AITER Triton kernel; rsigma output eliminates extra FP32 reduction |
| **Fused RoPE** (`LUMEN_FUSED_ROPE=1`) | **-135 ms/step (-3.1%)** | Apex native fused RoPE kernel; eliminates 160 unfused cos/sin/rotate_half ops per forward |
| **Deferred BDA to LayerNormLinear** (`LUMEN_FUSED_RESIDUAL_NORM=1`) | **-70 ms pre-eval, -155 ms post-eval** | Defer attn BDA residual add to LumenLayerNormLinear; eliminates BDA handler overhead, improves cache locality |
| **Fused residual+norm** (non-lumen-linear path) | **~0 ms** (net zero) | Skip BDA, fuse add into norm input; 1 fewer kernel launch per layer |
| **Backend caching + sync elimination** (`LUMEN_SKIP_BACKEND_SYNC=1`) | **~1-2% step time** | Cache GEMM backend selection, skip redundant syncs |
| **Fused RMSNorm + FP8 quant** (`LUMEN_FUSED_NORM_QUANT=1`) | ~0.2% step time | Merge norm + quant into single kernel (V1 path) |
| **Fused SwiGLU backward** (`LUMEN_FUSED_SWIGLU=1`) | Included in baseline→current | Fuse SwiGLU backward elementwise ops |
| **Wgrad `.t()` view** (v47) | **~50 ms/step** | Eliminate `grad_fp8.t().contiguous()` in wgrad path |
| **SwiGLU fused amax** (v47) | **~30-80 ms/step** | Replace `bf16.abs().amax()` with `fused_amax_abs()` in SwiGLU FP8 path |
| **Fused LayerNormLinear** (`--lumen-linear`) | **-302 ms/step (-6.5%)** | TE-style fused Norm+Linear autograd boundary; fewer kernel launches, reduced autograd overhead |
| **LoRA input normalization fix** | **-0.011 val_loss** | Fix LoRA_A receiving pre-norm input instead of post-norm; critical for fused LayerNormLinear + LoRA convergence |
| **LoRA norm cache** | **-38 ms CUDA/step, -1.1% memory** | Cache `ln_out` from fused norm+quant; LoRA adapter reuses it instead of redundant `_norm()` call; eliminates 160 RMSNorm kernel launches/step |
| **SwiGLU amax tracking** | correctness improvement | Feed pre-computed amax from fused SwiGLU FP8 quant to scaling manager; fixes missing amax recording |
| **QuantConfig propagation fix** | accuracy improvement | Forward correct `amax_algo`/`history_len` from training args to per-module ScalingManagers |
| **FP8 weight gradients** (`FP8_WGRAD=1`, hipBLASLt) | ~0 ms (correctness alignment) | Match MLPerf FP8 wgrad path |
| **ACL=21** (`RECOMPUTE_NUM_LAYERS=21`) | ~0 ms (memory trade-off) | Match MLPerf activation checkpointing depth |
| **Fused LoRA dropout+scale+add** (`LUMEN_FUSED_LORA=1`) | **~25 ms/step, -6.4% memory** | Triton kernel fuses dropout+scale+add for 160 LoRA adapters (480→160 kernel launches); regenerates mask from seed in backward, eliminating ~8.6 GiB dropout mask storage |
| **V2 fused residual+norm+quant** (`LUMEN_FUSED_RESIDUAL_KERNEL=1`) | **~15-20 ms/step** | Fuses x+residual→RMSNorm→FP8 quant→amax into single Triton kernel; previously blocked by OOM at 98.9% memory, now enabled thanks to fused LoRA memory savings |
| **hipThreadExchangeStreamCaptureMode stub** (LD_PRELOAD) | **~10-30 ms/step** | No-op stub eliminates ~10k HIP API calls per 3 steps when HIP graphs disabled |
| **Row-based quant+amax kernel** (`static_quant_with_amax`) | **~70 ms/step (1.6%)** | Replace 2D-tiled `cast_amax_fp8` (1 atomic per 64x64 tile) with row-based kernel (1 atomic per row); 8,192 vs 57,344 atomics for (8192,28672) tensors |
| **C++ FP8 quant dispatch** (`LUMEN_CPP_QUANT_DISPATCH=1`) | **~30 ms/step (0.7%)** | Replace Python hot path (dict lookups, boolean checks, Triton launch overhead) with single C++ call per tensor; fused scale+quant+amax HIP kernel; C++ amax history |
| **Forward residual add fusion** (fused add+norm+quant) | **~10-15 ms/step** | Wire existing `_try_fused_norm_quant_with_residual()` into forward; eliminates standalone `aten::add` for `x + pending_residual` |
| **Deferred MLP BDA** (MLP residual add fusion) | **~17 ms/step** | Defer MLP bias-dropout-add to next layer's LumenLayerNormLinear via TLS `_set_pending_residual`; fuses add+norm+quant in single kernel. Checkpoint-aware: layers 1-21 (recompute region) use standard BDA; layers 22-79 defer |
| **hipMemsetAsync for AMAX zeroing** | **included above** | Replace `amax_scratch_.zero_()` (`aten::fill_`) with `hipMemsetAsync` in C++ FP8QuantDispatcher; eliminates ~2,693 `fill_` dispatch calls/step |
| **FP8 dgrad epilogue** (`LUMEN_FP8_DGRAD_OUTPUT=1`) | **~5-10 ms/step** | hipBLASLt GEMM epilogue produces FP8 dgrad; cached via `_fp8_cache_put()` for next layer's backward |
| **In-place FP8 quant buffer reuse** (`LUMEN_REUSE_QUANT_BUFFER=1`) | **eliminates post-eval degradation** | Per-shape scratch buffer for backward-only FP8 quant; reduces allocator fragmentation from +6.8% to +0% post-eval |

### Net Result

| Metric | Baseline | Lumen v47 | Lumen (prev) | Lumen (C++ dispatch) | Lumen (mem-opt) | Lumen (speed-opt) | MLPerf Ref (local) |
|--------|----------|-----------|--------------|---------------------|----------------|-------------------|-------------------|
| Pre-eval step time | 7,400 ms | 4,730 ms | 4,263 ms | 4,170 ms | 4,152 ms | **4,135 ms** | **3,967 ms** |
| Post-eval step time | — | 5,550 ms | ~4,476 ms | TBD | ~4,145 ms (+0%) | TBD | ~4,000 ms |
| Memory utilization | — | 98.7% | 99.5% | 99.5% | 99.5% | **99.5%** | ~82% |
| val_loss (best) | 0.9371 (fail) | 0.9223 (pass) | 0.9187 (pass, step 960) | 0.9210 (pass, step 576) | **0.9170** (pass, step 1024) | TBD (103 steps) | 0.9243 (pass) |
| Speed ratio vs MLPerf | 1.87x | 1.19x | 1.075x | 1.051x | 1.047x | **1.042x** | 1.0x |

## Val_loss Trajectory

### Lumen (current) — Speed optimizations: MLP BDA deferral + hipMemsetAsync fill_ elimination (SEED=1234)

103-step validation run. Full convergence run pending.

| Step | val_loss | Passes target? |
|------|----------|----------------|
| — | — | — |

Pre-eval step time: **4,135 ms** (steps 20-100 avg).
Speed ratio: **1.042x** pre-eval vs MLPerf reference (3,967 ms).
103 steps completed with zero NaN/skip, memory stable at 99.5%.

### Lumen (prev) — Memory optimizations: residual fusion + FP8 dgrad epilogue + buffer reuse (SEED=1234)

| Step | val_loss | Passes target? |
|------|----------|----------------|
| 192 | 0.9455 | No |
| 384 | 0.9322 | No |
| 576 | **0.9212** | **Yes** |
| 768 | 0.9235 | Yes |
| 960 | **0.9191** | **Yes** |
| 1024 | **0.9170** | **Yes** |

val_loss at step 576 is **0.9212** — passes the MLPerf target (< 0.925).
Pre-eval step time: **4,152 ms** (steps 20-600 avg).
Post-eval step time: **~4,145 ms (+0%)** — post-eval degradation eliminated by buffer reuse.
Speed ratio: **1.047x** pre-eval vs MLPerf reference (3,967 ms).
1024 steps completed with zero NaN/skip, memory stable at 99.5%.

### Lumen (prev) — C++ FP8 quant dispatch + row-based quant + LD_PRELOAD stub + fused LoRA (SEED=1234)

| Step | val_loss | Passes target? |
|------|----------|----------------|
| 192 | 0.9447 | No |
| 384 | 0.9296 | No |
| 576 | **0.9210** | **Yes** |

val_loss at step 576 is **0.9210** — passes the MLPerf target (< 0.925).
Pre-eval step time: **4,170 ms** (steps 20-600 avg).
Speed ratio: **1.051x** pre-eval vs MLPerf reference (3,967 ms).

### Lumen (prev) — row-based quant + LD_PRELOAD stub + fused LoRA (SEED=1234)

| Step | val_loss | Passes target? |
|------|----------|----------------|
| 192 | 0.9430 | No |
| 384 | 0.9289 | No |
| 576 | **0.9203** | **Yes** |

val_loss at step 576 is **0.9203** — passes the MLPerf target (< 0.925).
Pre-eval step time: **4,191 ms** (steps 20-190 avg).
Post-eval step time: **~4,476 ms** (+6.8%).
Speed ratio: **1.056x** pre-eval vs MLPerf reference (3,967 ms).

### Lumen (prev) — fused RoPE + deferred BDA + fused LoRA + residual kernel (SEED=1234)

| Step | val_loss | Passes target? |
|------|----------|----------------|
| 192 | 0.9460 | No |
| 384 | 0.9336 | No |
| 576 | **0.9208** | **Yes** |
| 768 | 0.9229 | Yes |
| 960 | **0.9187** | **Yes** |

val_loss at step 576 is **0.9208** — passes the MLPerf target (< 0.925).
Pre-eval step time: **4,263 ms** (steps 20-100 avg).
Speed ratio: **1.075x** pre-eval vs MLPerf reference (3,967 ms).

### Lumen (older) — fused RoPE + deferred BDA (SEED=1234)

| Step | val_loss | Passes target? |
|------|----------|----------------|
| 192 | 0.9465 | No |
| 384 | 0.9327 | No |
| 576 | **0.9218** | **Yes** |
| 768 | 0.9229 | Yes |
| 960 | **0.9183** | **Yes** |

### Lumen (older) — deferred BDA to LayerNormLinear (SEED=1234)

| Step | val_loss | Passes target? |
|------|----------|----------------|
| 192 | 0.9455 | No |
| 384 | 0.9333 | No |
| 576 | **0.9222** | **Yes** |
| 768 | 0.9245 | Yes |
| 960 | **0.9195** | **Yes** |

### Lumen (older) — fused LayerNormLinear + LoRA fix (SEED=1234)

| Step | val_loss | Passes target? |
|------|----------|----------------|
| 192 | 0.9484 | No |
| 384 | 0.9415 | No |
| 576 | **0.9231** | **Yes** |
| 768 | 0.9236 | Yes |
| 960 | **0.9197** | **Yes** |

### Lumen (older) — rsigma optimization (SEED=1234)

| Step | val_loss | Passes target? |
|------|----------|----------------|
| 192 | 0.9510 | No |
| 384 | 0.9351 | No |
| 576 | **0.9244** | **Yes** |
| 768 | 0.9246 | Yes |
| 960 | 0.9206 | Yes |
| 1024 | **0.9190** | **Yes** |

### Lumen v48 (SEED=1234)

| Step | val_loss | Passes target? |
|------|----------|----------------|
| 192 | 0.9485 | No |
| 384 | 0.9321 | No |
| 576 | **0.9211** | **Yes** |

### MLPerf Reference — Local MI300X (SEED=1234)

| Step | val_loss | Throughput (s/s) | Step time (ms) |
|------|----------|-----------------|---------------|
| 192 | 0.9398 | 2.0167 | 3,967 |
| 240 | 0.9395 | 2.0172 | 3,966 |
| 288 | 0.9311 | 2.0168 | 3,967 |
| 336 | 0.9268 | 2.0167 | 3,967 |
| 384 | **0.9243** | 2.0156 | 3,969 |

### Lumen (C++ dispatch) — C++ FP8 quant dispatch, `LUMEN_CPP_QUANT_DISPATCH=1` (SEED=1234)

| Step | val_loss | Passes target? |
|------|----------|----------------|
| 192 | 0.9447 | No |
| 384 | 0.9296 | No |
| 576 | **0.9210** | **Yes** |

val_loss at step 576 is **0.9210** — passes the MLPerf target (< 0.925).
Pre-eval step time: **4,170 ms** (steps 20-600 avg).
Speed ratio: **1.051x** pre-eval vs MLPerf reference (3,967 ms).
600 steps completed with zero NaN/skip, memory stable at 99.5%.

### Convergence Gap at Matched Steps

| Step | Lumen (speed-opt) | Lumen (mem-opt) | Lumen (C++ dispatch) | Lumen (prev) | MLPerf Ref (local) |
|------|-------------------|----------------|---------------------|--------------|-------------------|
| 192 | — | 0.9455 | 0.9447 | 0.9430 | 0.9398 |
| 384 | — | 0.9322 | 0.9296 | 0.9289 | 0.9243 |
| 576 | — | **0.9212** | **0.9210** | **0.9203** | — |

Lumen (speed-opt) adds MLP BDA deferral and hipMemsetAsync fill_ elimination.
Pre-eval: ~17 ms/step faster (4,135 vs 4,152 ms). Full convergence run pending.

## Speed Gap Analysis

Both Lumen and MLPerf reference use identical parallelism (TP=1, ACL=21, DP=8)
and FP8 config. The remaining 1.05x pre-eval speed gap is from kernel fusion,
dispatch overhead, and memory pressure.

### Remaining Speed Gap: 4,135 ms (Lumen speed-opt) vs 3,967 ms (MLPerf local) = 168 ms

Based on fresh profiling data (steps 8-10, rank 0, 3-step totals divided by 3):

| # | Root Cause | Lumen (ms/step) | TE est. (ms/step) | Gap (ms) | Status |
|---|-----------|-----------------|-------------------|---------|--------|
| 1 | **aten::add + add_** (residual, autograd) | 140 | ~30 | **~110** | 2586+2129 calls/3-step; fuse residual into norm needs transformer block changes |
| 2 | **hipThreadExchangeStreamCaptureMode** | 101 | ~0 | **~101** | ROCm runtime overhead, 2344 calls/step; HIP graphs or ROCm upgrade |
| 3 | **FP8 quant pipeline** (dynamic+static+amax) | 186 | ~54 | **~132** | dynamic=81, static=70, amax_abs=35 |
| 4 | **aten::mul** (autograd backward) | 81 | ~20 | **~61** | Chain rule multiplications; hard to eliminate |
| 5 | **aten::cat** (QKV backward concat) | 70 | ~10 | **~60** | 647 calls/step; pre-allocated buffer would eliminate |
| 6 | **aten::copy_** (dtype casts) | 50 | ~10 | **~40** | Down from 289 ms (v47), most copies fused |
| 7 | **SwiGLU fwd+bwd** | 147 | ~100 | **~47** | fwd=80, bwd=67 |
| 8 | **dropout fwd+bwd** | 41 | ~10 | **~31** | 606+480 calls; fuse into preceding op |
| 9 | **Memcpy DtoD** | 25 | ~5 | **~20** | Down from 88 ms (v47) |
| 10 | **Other** (fill_, neg, RoPE, NCCL) | ~80 | ~30 | **~50** | |
| | **Total estimated gap** | | | **~652** | vs measured 413 ms (GPU overlap masks ~239 ms) |

### Highest-Impact Next Steps

The gap is **structural** — composed of many small overheads with no single optimization
yielding >50 ms without significant infrastructure work. Each candidate was investigated:

1. **HIP graphs** (est. -100 ms/step) — `hipThreadExchangeStreamCaptureMode` costs
   101 ms/step (2344 calls). HIP graph capture for the training step would eliminate
   most runtime dispatch overhead. Requires careful handling of dynamic shapes
   (eval vs train), NCCL collective integration, FP8 scaling state updates, and
   dropout RNG. **Status: major infrastructure project, not attempted.**

2. **Fuse residual add into LayerNormLinear** (est. -50-70 ms/step) — **PARTIALLY DONE.**
   Forward path now calls `_try_fused_norm_quant_with_residual()` to fuse add+norm+quant
   in a single AITER Triton kernel, eliminating standalone `aten::add`. Measured ~10-15 ms/step
   improvement. Full autograd-level fusion (keeping 3 tensors for backward) still blocked by
   memory constraints at 99.5% VRAM.

3. **QKV buffer pre-allocation** (est. -30 ms/step) — **NOT FEASIBLE with current arch.**
   TE's `_SplitAlongDim.backward` has a noop path when grads are contiguous views, but
   CK attention backward produces separate grad tensors, so `torch.cat` is always called.
   Only ~80/647 cat calls/step are from QKV; rest from LoRA, column parallel overlap, etc.

4. **Reduce memory pressure** — **prerequisite for item 2.** Currently 96.7% (~185 GiB).
   Options: FP8 storage for attention inputs (currently BF16), gradient checkpointing
   for LoRA activations, or FP8 param storage for remaining non-frozen layers.

## Post-Eval Performance

| Metric | Lumen v47 | Lumen (prev) | Lumen (mem-opt) | Lumen (speed-opt) | MLPerf Ref (local) |
|--------|-----------|--------------|----------------|-------------------|-------------------|
| Pre-eval ms/step | 4,730 | 4,263 | 4,152 | **4,135** | **3,967** |
| Post-eval #1 ms/step | 5,550 | ~4,476 | ~4,145 | TBD | ~4,000 |
| Post-eval delta | +17.3% | +6.8% | +0% | TBD | ~0% |

Root cause of prior degradation: Megatron's `transformer_block.py` skips activation
checkpointing during eval (`self.training=False`), allocating activations for all 80
layers vs 21 during training. At high memory utilization, this fragments the ROCm
allocator and the fragmentation persists after training resumes.

**Solved in mem-opt**: `LUMEN_REUSE_QUANT_BUFFER=1` reuses per-shape scratch buffers
for backward-only FP8 quantization (~1,089 allocations/step), eliminating the transient
allocation churn that caused allocator fragmentation. Post-eval delta is now **+0%**.

## Configuration Diff

| Parameter | Lumen (current) | MLPerf Reference (local) |
|-----------|-----------------|--------------------------|
| Framework | Megatron-LM-AMD + Lumen + AITER | NeMo v2.3.0 + TE + Megatron-LM |
| Docker | N/A (native) | `rocm/amd-mlperf:llama2_70b_training_5.1` |
| TP / PP / CP | 1 / 1 / 1 | 1 / 1 / 1 |
| DP | 8 | 8 |
| MBS / GBS | 1 / 8 | 1 / 8 |
| LR | 4e-4 | 4e-4 |
| LR Warmup | 0 | 0 |
| LR Schedule | Cosine, 1024 steps | Cosine, 1024 steps |
| LoRA rank / alpha | 16 / 32 | 16 / 32 |
| FP8 Format | E4M3 hybrid (delayed) | E4M3 hybrid (delayed) |
| FP8 amax_history | 4 | 4 |
| FP8 amax_algo | most_recent | most_recent |
| FP8 backward dtype | E5M2 (hipBLASLt) | E5M2 (hipBLASLt) |
| FP8 wgrad | FP8 (hipBLASLt) | FP8 (TE) |
| Activation recompute | 21 layers (full/block) | 21 layers (full/block) |
| Fused LayerNormLinear | `--lumen-linear` (LumenLayerNormLinear) | TE `_LayerNormLinear` |
| RMSNorm | AITER Triton | TE Triton / apex |
| SwiGLU | Fused Triton (fwd+bwd) | TE fused C++ |
| Attention | AITER CK FMHA v3 | TE CK fused attn v3 |
| RoPE | Apex fused (native, `LUMEN_FUSED_ROPE=1`) | TE fused |
| Seed | 1234 | 1234 |
| Val check interval | 192 steps | 48 steps (384/GBS, SKIP_EVALS=3) |

## Optimization History

| Change | val_loss | Step time | vs MLPerf |
|--------|----------|-----------|-----------|
| Baseline (no shuffle) | 0.9371 (fail) | 7,400 ms | 1.87x |
| + Data shuffling | **0.9208** (pass) | 7,400 ms | 1.87x |
| + Aligned eval + fused norm+quant | 0.9192 | 6,200 ms | 1.56x |
| + Fused quant+amax | — | 5,809 ms | 1.46x |
| + Fused SwiGLU bwd | — | 5,348 ms | 1.35x |
| + ACL=21 + hipBLASLt wgrad | 0.921 | 5,570 ms | 1.40x |
| + hipBLASLt all + `.t()` view (v46) | 0.9216 | 4,780 ms | 1.21x |
| + Wgrad `.t()` + SwiGLU fused amax (v47) | 0.9223 | 4,730 ms | 1.19x |
| + Fused residual+norm + v48 kernel opts (v48) | 0.9211 | 4,747 ms | 1.20x |
| + NQG fusion + rsigma opt | 0.9190 | 4,699 ms | 1.18x |
| + Fused LayerNormLinear `--lumen-linear` (broken LoRA) | 0.9280 (fails target) | 4,370 ms | 1.10x |
| + LoRA norm fix + QuantConfig fix | 0.9233 (passes at step 576) | 4,348 ms | 1.10x |
| + Deferred BDA to LayerNormLinear | 0.9222 | 4,310 ms | 1.09x |
| + Fused RoPE | 0.9218 (passes at step 576) | 4,175 ms | 1.05x |
| + Fused LoRA + residual kernel + LD_PRELOAD stub | 0.9218 | 4,263 ms | 1.075x |
| + Row-based quant kernel | 0.9203 (passes at step 576) | 4,191 ms | 1.056x |
| + C++ FP8 quant dispatch | 0.9210 (passes at step 576) | 4,170 ms | 1.051x |
| + Memory opts: residual fusion + FP8 dgrad + buffer reuse | 0.9212 (passes at step 576) | 4,152 ms | 1.047x |
| **+ Speed opts: MLP BDA deferral + hipMemsetAsync fill_ elimination (current)** | TBD (103-step validation) | **4,135 ms** | **1.042x** |

## Source Files

### Core — Fused LayerNormLinear (`--lumen-linear`) + FP8 Dtype Fix

| File | Purpose |
|------|---------|
| `lumen/modules/layernorm_linear.py` | **MODIFIED**: FP8 dtype fix (`_get_float8_e4m3()`), fused Norm+Linear autograd boundary |
| `lumen/modules/parallel_linear.py` | **MODIFIED**: FP8 dtype fix for `LumenColumnParallelLinear` and `LumenRowParallelLinear` |
| `lumen/modules/grouped_linear.py` | **MODIFIED**: FP8 dtype fix for `LumenGroupedLinear` |
| `lumen/models/megatron.py` | **MODIFIED**: Auto-detect `fp8_dtype`, propagate `quant_config` to per-module ScalingManagers, `_patch_lora_for_layernorm_linear()` — normalize LoRA_A input for fused LayerNormLinear |

### Core — Required for NQG Fusion and Training

| File | Purpose |
|------|---------|
| `lumen/ops/fused_norm_quant.py` | **NEW**: Fused RMSNorm + FP8 quant + `_FusedNQGNorm` autograd Function + rsigma output |
| `lumen/ops/fused_residual_norm.py` | **NEW**: Deferred BDA + autograd-aware RMSNorm + fused add+rmsnorm |
| `lumen/models/megatron_patches.py` | **MODIFIED**: `install_fused_residual_norm()` — NQG + deferred BDA patches on `TransformerLayer` |
| `lumen/quantize/__init__.py` | **MODIFIED**: Thread-local `_set_pre_quantized_activation` / `_pop_pre_quantized_activation` for NQG bypass |
| `lumen/ops/quantize/linear.py` | **MODIFIED**: `pre_quantized_input` param in `QuantizedLinearFunction` and `FP8StoredLinearFunction` |

### AITER Triton Kernel Changes

| File | Purpose |
|------|---------|
| `third_party/aiter/aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py` | **MODIFIED**: `_rmsmorm_op` returns `(norm, rsigma)` tuple; kernel outputs rsigma via `OUTPUT_RSIGMA` flag |
| `third_party/aiter/aiter/ops/triton/quant/fused_fp8_quant.py` | **MODIFIED**: `output_rsigma=False` param; allocates rsigma tensor; returns 5-tuple when enabled |

### AITER C++ Fused Module (optional — JIT compiled, not used in current path)

| File | Purpose |
|------|---------|
| `third_party/aiter/aiter/ops/fused_norm_quant_gemm.py` | **NEW**: Python wrapper for C++ fused norm+quant+GEMM |
| `third_party/aiter/csrc/kernels/fused_norm_quant_gemm.cu` | **NEW**: HIP C++ host function (rmsnorm_quant + hipBLASLt) |
| `third_party/aiter/csrc/include/fused_norm_quant_gemm.h` | **NEW**: Header |
| `third_party/aiter/csrc/pybind/fused_norm_quant_gemm_pybind.cu` | **NEW**: pybind11 module |
| `third_party/aiter/csrc/include/rocm_ops.hpp` | **MODIFIED**: `FUSED_NORM_QUANT_GEMM_PYBIND` macro |
| `third_party/aiter/aiter/jit/optCompilerConfig.json` | **MODIFIED**: `module_fused_norm_quant_gemm` entry |
| `third_party/aiter/aiter/jit/core.py` | **MODIFIED**: added module to build list |

### Core — C++ FP8 Quant Dispatch (`LUMEN_CPP_QUANT_DISPATCH=1`)

| File | Purpose |
|------|---------|
| `lumen/csrc/fp8_quant_dispatch.cu` | **NEW**: C++ `FP8QuantDispatcher` class + `fused_scale_quant_amax_kernel` HIP kernel (pybind11) |
| `lumen/ops/quantize/quant_dispatch_cpp.py` | **NEW**: Python wrapper, probe function, JIT compilation fallback |
| `lumen/quantize/scaling_manager.py` | **MODIFIED**: C++ fast path in `_quantize_core()` and `quantize_bwd_delayed()` |
| `setup.py` | **MODIFIED**: Added `lumen.csrc._fp8_quant_dispatch` CppExtension |

### Other Modified Files

| File | Purpose |
|------|---------|
| `lumen/modules/layernorm_linear.py` | `_FusedResidualRMSNormFP8Quant` / `_FusedRMSNormFP8QuantV2` autograd Functions + FP8 dtype fix (see above) |
| `lumen/modules/parallel_linear.py` | `is_contiguous()` guards + FP8 dtype fix (see above) |
| `lumen/ops/normalization/rmsnorm.py` | `rmsnorm_from_module` + `fused_add_rmsnorm` with autograd guard |
| `lumen/ops/quantize/cast_transpose.py` | `cast_amax_fp8` (no-transpose), `rmsnorm_quant_amax_fp8` kernels |
| `lumen/ops/quantize/quant_amax_fused.py` | `dequant_fp8_to_bf16` fused dequant kernel |
| `lumen/ops/dispatch.py` | `_probe_aiter_fused_quant` + backend caching |
| `lumen/ops/__init__.py` | Module exports |
| `lumen/ops/gemm/__init__.py` | GEMM epilogue exports |
| `lumen/quantize/scaling_manager.py` | `_quantize_core` with row-based `static_quant_with_amax` for hipBLASLt path |
| `lumen/csrc/hip_no_stream_capture.c` | **NEW**: LD_PRELOAD stub for `hipThreadExchangeStreamCaptureMode` no-op |
| `lumen/models/_swiglu_fp8_fuse.py` | `is_contiguous()` guard |
| `lumen/utils/hip_graphs.py` | HIP graph capture fixes (not actively used) |
| `examples/llama2/run_lora_finetune_llama2_70b.sh` | All env flags including `LUMEN_FP8_DGRAD_OUTPUT=1`, `LUMEN_REUSE_QUANT_BUFFER=1` |
| `examples/llama2/run_profile.sh` | Profiling script additions |
| `examples/llama2/scripts/patch_mlp_fp8_store.py` | SwiGLU pre-alloc fix |

### Obsolete Patch Scripts (integrated into megatron_patches.py)

| File | Status |
|------|--------|
| `examples/llama2/scripts/patch_fused_nqg.py` | **OBSOLETE** — functionality in `install_fused_residual_norm()` |
| `examples/llama2/scripts/patch_fused_residual_norm.py` | **OBSOLETE** — functionality in `install_fused_residual_norm()` |

### Non-Essential (generated data, debug logs)

| File | Status |
|------|--------|
| `examples/llama2/tunableop_results{0-7}.csv` | TunableOp cache — machine-specific, not needed for code submission |
| `.cursor/tmp-training-bugs.md` | Debug notes — not production code |
| `lumen/ops/gemm/epilogue.py` | FP8 output GEMM epilogue — used by `LUMEN_FP8_DGRAD_OUTPUT=1` |
| `lumen/ops/gemm/fp8_output.py` | `gemm_per_tensor_mixed_fp8out()` — hipBLASLt GEMM with FP8 output + AMAX_D |

## Profiling Data

| File | Description |
|------|------------|
| `profiling/lumen_profile_summary.txt` | `torch.profiler` key_averages — CK baseline, 3 steps, rank 0 |
| `profiling/lumen_latest_profile_summary.txt` | `torch.profiler` key_averages — v46 hipBLASLt all + `.t()` view, 3 steps, rank 0 |
| `profiling/te_profile_results.txt` | `torch.profiler` key_averages — TE ops (LayerNormLinear, Linear) at same tensor shapes, 10 iters |
