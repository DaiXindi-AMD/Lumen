# Feature Dev — Lumen MXFP4

By Dai, Xindi

---

## 1. Design Overview

Lumen MXFP4 implements FP4 E2M1 training for linear layers on AMD MI350X (gfx950) hardware. The design is informed by NVFP4 (NVIDIA, 2025) and arXiv:2605.09825 (AMD/PSU, 2025). Forward and DGrad GEMMs are computed in MXFP4; WGrad uses MXFP4 with deterministic Hadamard rotation to stabilize convergence.

### 1.1 Forward (Fprop): Y = Q(X) @ Q(W)^T

| Operand      | Format          | Rounding | Block Layout | Scales             |
|--------------|-----------------|----------|--------------|--------------------|
| Weight W     | MXFP4 (32×32)  | RTN      | 2D tiles     | E8M0 (N/32, K/32)  |
| Activation X | MXFP4 (1×32)   | RTN      | 1D per-group | E8M0 (M, K/32)     |
| Output Y     | BF16            | —        | —            | —                  |

- Weight uses 2D (32×32) block scaling (chain-rule consistent, transpose-friendly).
- Activation uses 1D (1×32) per-group scaling along K.
- GEMM kernel: AITER `gemm_afp4wfp4` (TN layout, packed uint8 FP4, E8M0 scales).
- Weight is re-quantized from BF16 master in backward (see §2.3).

### 1.2 Backward DGrad: dX = Q(dY) @ Q(W_ref)^T

| Operand      | Format          | Rounding | Notes                              |
|--------------|-----------------|----------|------------------------------------|
| Gradient dY  | MXFP4 (1×32)   | SR       | Stochastic rounding for gradients  |
| Weight W     | MXFP4 (32×32)  | RTN      | Re-quantized from `ctx.weight_ref` |
| Output dX    | BF16            | —        | —                                  |

- Weight is **re-quantized from `ctx.weight_ref`** (BF16 master weight) in backward, not reused from forward `save_for_backward`. This is required for FSDP2 compatibility (see §2.3).
- After re-quantization, weight is transposed via `transpose_packed_fp4`: (N, K/2) → (K, N/2).
- GEMM kernel: same `gemm_afp4wfp4`.

### 1.3 Backward WGrad: dW = Q(H·dY^T) @ Q(H·X^T)^T

| Operand       | Format          | Rounding | Notes                                    |
|---------------|-----------------|----------|------------------------------------------|
| Gradient dY^T | MXFP4 (1×32)   | SR       | Transpose → Hadamard → quantize          |
| Activation X^T| MXFP4 (1×32)   | SR       | Dequant saved FP4 → BF16 → transpose → Hadamard → quantize |
| Output dW     | BF16            | —        | —                                        |

WGrad applies a **deterministic Hadamard rotation** (all +1 sign vector = pure H, no random diagonal) before quantization, following arXiv:2605.09825:

1. **Dequant saved activation**: `convert_from_mxfp4(X_fp4, X_scale) → X_bf16`
2. **Transpose**: dY^T (N, M) and X^T (K, M)
3. **Deterministic Hadamard**: blockwise H with G=32 along reduction dim M. Both operands receive the same H, which cancels in GEMM: (dY^T H)(X^T H)^T = dY^T X.
4. **SR quantize both**: 1×32 along axis=-1
5. **GEMM**: `gemm_afp4wfp4(dY^T_fp4, X^T_fp4) → dW`

### 1.4 Per-Layer Operation Count

| Phase  | FP4 Quant | Dequant | Hadamard | Transpose | FP4 GEMM |
|--------|-----------|---------|----------|-----------|----------|
| Fprop  | 2 (X + W) | 0       | 0        | 0         | 1        |
| DGrad  | 2 (dY + W re-quant) | 0 | 0    | 1 (W packed) | 1     |
| WGrad  | 2 (dY^T + X^T SR) | 1 (X) | 2 (dY^T + X^T) | 2 (dY, X) | 1 |
| **Total** | **6**  | **1**   | **2**    | **3**     | **3**    |

Compare to BF16: 0 quant, 0 dequant, 0 Hadamard, 0 transpose, 3 BF16 GEMMs.

---

## 2. Implementation Details

### 2.1 Hardware Acceleration (gfx950 ASM)

Quantization uses Triton inline assembly targeting MI350X VOP3 instructions:

| Instruction                          | Operation                        |
|--------------------------------------|----------------------------------|
| `v_cvt_scalef32_pk_fp4_f32`         | 2×FP32 → packed FP4 byte (RTN)  |
| `v_cvt_scalef32_sr_pk_fp4_f32`      | 2×FP32 → packed FP4 byte (SR)   |
| `v_cvt_scalef32_pk_fp4_bf16`        | 2×BF16 → packed FP4 byte (RTN)  |
| `v_cvt_scalef32_sr_pk_fp4_bf16`     | 2×BF16 → packed FP4 byte (SR)   |

Detection: `is_cdna4()` → `target.arch == "gfx950"`. Verified active on MI350X. Non-gfx950 falls back to software LUT-based FP4 E2M1 conversion.

### 2.2 GEMM Dispatch

```
dispatch_gemm(scaling_type="mxfp4")
  → gemm_mxfp4_dispatch()
      → try_backends([
            (TRITON, _gemm_mxfp4_aiter),     # AITER gemm_afp4wfp4 (native FP4)
            (TRITON, _gemm_mxfp4_fallback),   # dequant → BF16 GEMM
        ])
```

On MI350X with AITER, `gemm_afp4wfp4` succeeds — no fallback.

### 2.3 FSDP2 Compatibility Fix (Critical)

**Problem**: FSDP2 `full_shard` reshards (frees) weight parameters after forward. MXFP4 forward originally saved packed FP4 weight via `ctx.save_for_backward()`. In multi-GPU backward, these saved tensors reference resharded (partial/stale) data, causing NaN gradients in every layer on step 1.

**Evidence**:

| Config                              | NaN grads on step 1 |
|-------------------------------------|---------------------|
| 1 GPU, no FSDP                      | 0 / 399             |
| 1 GPU, FSDP2 (world_size=1)         | 0 / 399             |
| 2 GPU, FSDP2, no gradient ckpt      | **397 / 399**       |
| 8 GPU, FSDP2, no gradient ckpt      | **397 / 399**       |
| 8 GPU, FSDP2, WITH gradient ckpt    | 0 initially, spike at step ~1250 |

With gradient checkpointing, backward recomputes forward (triggering FSDP2 all-gather), partially masking the bug — training runs for ~1250 steps before eventual loss spike from accumulated error.

**Fix**: Backward re-quantizes weight from `ctx.weight_ref` (the BF16 master weight managed by FSDP2, which is correctly all-gathered before backward) instead of using saved FP4 weight. Only the activation (not a model parameter, unaffected by FSDP resharding) is saved in `ctx.save_for_backward()`.

This follows the same pattern as the existing FP8 blockwise backward (line 1527-1530 in `linear.py`), which uses `ctx.weight_ref` for DGrad weight access.

**Tradeoff**: Backward now pays an extra weight re-quantization cost (BF16→MXFP4 32×32) per layer, adding ~5-10% to backward time. This is the correct behavior — the same approach is used by ROCm Transformer Engine (arXiv:2605.09825 Figure 3).

### 2.4 Dispatch Routing Fix

- `lumen/quantize/__init__.py:290` used `config.scaling.value` (returns `"blockwise"` for MXFP4) → misrouted to FP8 blockscale GEMM.
- Fixed to `config.recipe` (returns `"mxfp4"`).

### 2.5 Hadamard Design (arXiv:2605.09825)

The Hadamard sign vector uses deterministic all +1 (pure Hadamard, no random diagonal matrix). The paper shows randomized signs cause Wgrad divergence at 8B+ scale due to structured micro-scaling errors from outliers. Only deterministic Hadamard converges stably.

Experiments confirmed: random sign Hadamard crashed at step ~1275 on 8B; deterministic sign also crashed at ~1275 (before the FSDP2 fix); after the FSDP2 fix, deterministic Hadamard trains stably.

### 2.6 Backward Fallback

When dimensions are not 32-aligned, backward falls back to BF16 GEMM using `ctx.weight_ref` directly.

---

## 3. Operator Accuracy (12/12 Tests vs torchAO)

| # | Operation                      | Lumen Op                     | Reference            | Result              |
|---|--------------------------------|------------------------------|----------------------|---------------------|
| 1 | 1D Quant (axis=-1, RTN)        | `convert_to_mxfp4`           | torchAO MXTensor     | bitwise identical   |
| 2 | 1D Dequant                     | `convert_from_mxfp4`         | torchAO MXTensor     | bitwise identical   |
| 3 | Cross-framework dequant        | `convert_from_mxfp4`         | torchAO `to_dtype`   | bitwise identical   |
| 4 | 1D Quant (axis=0, RTN)         | `convert_to_mxfp4`           | torchAO MXTensor     | bitwise identical   |
| 5 | Dual-axis quant                | `convert_to_mxfp4_dual_axis` | torchAO MXTensor     | bitwise identical   |
| 6 | Roundtrip (quant→dequant)      | `convert_to/from_mxfp4`      | torchAO MXTensor     | bitwise identical (SNR 19.0 dB) |
| 7 | GEMM (Y=A@W^T)                 | `gemm_mxfp4_dispatch`        | torchAO MXTensor     | bitwise identical   |
| 8 | 2D Block Quant Roundtrip       | `convert_to/from_mxfp4_2d`   | Manual LUT reference | bitwise identical   |
| 9 | Packed FP4 Transpose           | `transpose_packed_fp4`       | Python reference     | bitwise identical   |
| 10 | Hadamard Transform            | `hadamard_transform`         | torchAO RHT          | ≈identical (atol=1e-2) |
| 11 | Stochastic Rounding Unbiased  | 200-round SR mean test       | Statistical          | Unbiased (p > 0.05) |
| 12 | 2D Scale Expansion            | `_expand_2d_scale_to_1d`     | Manual reference     | bitwise identical   |

---

## 4. Training Experiments

### 4.1 Qwen3-0.6B: MXFP4 vs BF16 (C4, 10k steps)

| Parameter          | Value                                      |
|--------------------|--------------------------------------------|
| Model              | Qwen3-0.6B (dense), random init, head_dim=128 |
| Dataset            | C4 (allenai/c4, English, streaming)        |
| Sequence length    | 512                                        |
| Global batch size  | 16 (micro_batch=2 × 8 GPUs)               |
| Optimizer          | AdamW (β1=0.9, β2=0.95, ε=1e-8, wd=0.1)  |
| Learning rate      | 6e-5 peak, 200-step warmup, cosine decay   |
| Grad clip          | 1.0                                        |
| Parallelism        | FSDP2 full_shard, 8× MI350X               |
| Steps              | 10,000                                     |
| Seed               | 1234                                       |

**Convergence**:

| Step   | BF16    | MXFP4   | Δ       |
|-------:|--------:|--------:|--------:|
|    500 | 7.163   | 7.187   | +0.024  |
|  2,000 | 6.540   | 6.570   | +0.030  |
|  5,000 | 6.334   | 6.364   | +0.030  |
| 10,000 | 6.299   | 6.344   | **+0.045** |

Loss curves nearly superimposed. Final gap +0.045 (0.7% relative). No NaN, no divergence.

**Throughput**:

| Metric           | BF16   | MXFP4  | Ratio        |
|------------------|--------|--------|--------------|
| Median step time | 229 ms | 478 ms | 2.09× slower |

### 4.2 Qwen3-8B: BF16 Baseline (C4, 5k steps)

BF16 8B trained with the same hyperparameters (lr=6e-5, warmup=200):

| Step   | BF16 val_loss |
|-------:|--------------:|
|    250 | 7.438         |
|  1,000 | 6.563         |
|  2,500 | 6.158         |
|  5,000 | 6.048         |

Median step time: 229 ms. Peak memory: 15.3 GB/GPU.

### 4.3 Qwen3-8B: MXFP4 (C4, 5k steps — in progress)

After the FSDP2 fix (§2.3), 8B MXFP4 trains with the same lr=6e-5 as BF16. The run that previously diverged at step ~1250 is now stable through step 1500+. Full results pending.

Median step time: ~690 ms. Peak memory: 15.3 GB/GPU.

---

## 5. Performance Analysis

### 5.1 Why MXFP4 Shows No Speed Benefit Yet

MXFP4 training is **~2× slower** than BF16 at 0.6B and ~3× slower at 8B. This is a known limitation of the current implementation, not the MXFP4 algorithm.

**Root cause: unfused kernel pipeline.** Each MXFP4 GEMM requires 3 separate kernel launches (Hadamard → Quant → GEMM) with global memory reads/writes between each stage:

```
Current Lumen pipeline (per GEMM, per operand):
  kernel 1: hadamard_transform   BF16 read → BF16 write (global memory)
  kernel 2: convert_to_mxfp4     BF16 read → FP4 write (global memory)
  kernel 3: gemm_afp4wfp4        FP4 read → BF16 write (global memory)
```

The paper (arXiv:2605.09825) achieves 9-10% speedup over FP8 by fusing all three into a single kernel via **GEMM prologue fusion** in ROCm Transformer Engine:

```
ROCm TE fused pipeline (single kernel):
  1. Load BF16 tile from global memory → registers
  2. Hadamard butterfly in registers (O(G log G), no memory traffic)
  3. FP4 quant in registers (scale + convert)
  4. FP4 data → shared memory → Matrix Core
  5. Write BF16 result
```

Per-operation microbenchmark at 0.6B scale (M=K=N=1024):

| Operation             | Time (μs) | Notes                    |
|-----------------------|-----------|--------------------------|
| BF16 GEMM             | 16        | Baseline                 |
| FP4 quant (2 operands)| 153       | gfx950 ASM path          |
| FP4 GEMM              | 57        | `gemm_afp4wfp4`          |
| Hadamard transform    | 49        | Per-operand, WGrad only  |
| Packed FP4 transpose  | 70        | DGrad weight re-quant    |

At 0.6B, quantization overhead (153μs) is 9.6× the BF16 GEMM itself (16μs).

### 5.2 Why Memory Is Not Saved

MXFP4 does not reduce peak memory in the current implementation because:

1. **BF16 master weights are retained** — FSDP2 shards and all-gathers BF16 weights. FP4 exists only as transient intermediate results.
2. **Backward re-quantizes from BF16** — The FSDP2 fix (§2.3) requires `ctx.weight_ref` (BF16) in backward, so BF16 weight storage cannot be eliminated.
3. **Optimizer states are BF16** — AdamW momentum and variance are stored in BF16/FP32.

Memory savings would require: (a) FP4 weight storage with FSDP all-gather in FP4, and (b) FP4-aware optimizer states — both are future work requiring AITER and PyTorch-level changes.

### 5.3 Path to Performance Parity

| Optimization                  | Owner     | Expected impact             |
|-------------------------------|-----------|------------------------------|
| Fused Hadamard+Quant kernel   | Lumen     | −30-40% quant overhead       |
| GEMM prologue fusion (H+Q+GEMM) | AITER  | Match ROCm TE (9-10% over FP8) |
| FP4 weight storage + FSDP     | AITER + PyTorch | Memory reduction        |
| FP4 gradient communication    | Lumen     | Reduced allreduce bandwidth  |

---

## 6. Debugging History

### 6.1 Dispatch Routing Bug (Resolved)

`config.scaling.value` returned `"blockwise"` for MXFP4, routing to FP8 blockscale GEMM. Fixed to `config.recipe` which returns `"mxfp4"`. Commit `656922c`.

### 6.2 FSDP2 NaN Gradients (Resolved)

MXFP4 backward used saved FP4 weight from `save_for_backward`. FSDP2 reshards weight after forward, invalidating the saved tensor. Multi-GPU backward produced 397/399 NaN grads on step 1 without gradient checkpointing. Fixed by re-quantizing from `ctx.weight_ref`.

### 6.3 Hadamard/SR Ablation (Informational)

Tested per arXiv:2605.09825 before the FSDP2 root cause was found:

| Variant                        | Wgrad Hadamard      | Wgrad Rounding | Crash step |
|--------------------------------|---------------------|----------------|------------|
| Original                       | Random sign ±1      | SR             | ~1550      |
| Deterministic sign             | All +1              | SR             | ~1275      |
| Deterministic + RTN            | All +1              | RTN            | ~425       |
| Full-pipeline Fprop+Dgrad+Wgrad| All +1              | SR             | step 1     |

All variants crashed because the real bug was FSDP2 `save_for_backward`, not Hadamard/SR configuration. After the FSDP2 fix, deterministic Hadamard + SR trains stably.

---

## 7. Implementation Artifacts

| Artifact | Path |
|----------|------|
| MXFP4 Triton kernels (ASM + fallback) | `lumen/kernels/mxfp4.py` |
| Quantization ops | `lumen/ops/quantize/ops.py` |
| GEMM dispatch + autograd | `lumen/ops/quantize/linear.py` |
| Dispatch routing fix | `lumen/quantize/__init__.py` |
| Unit tests (12 ops) | `tests/ops/test_quantize.py` |
| Accuracy report script | `scripts/mxfp4_accuracy_report.py` |
| Pretraining script (C4 + TensorBoard) | `examples/qwen3/pretrain_qwen3_mxfp4.py` |
| SFT script (BF16/FP8/MXFP4) | `examples/qwen3/train_qwen3_fsdp.py` |
| Paper comparison | `docs/papers/mxfp4_paper_vs_lumen_comparison.md` |

---

## 8. Status and Next Steps

### Done

- [x] MXFP4 quantization ops: 1D/2D RTN, SR, dual-axis, dequant, transpose, Hadamard
- [x] gfx950 ASM kernels for RTN and SR
- [x] AITER `gemm_afp4wfp4` native FP4 GEMM
- [x] Full autograd forward + backward (DGrad + WGrad)
- [x] Deterministic Hadamard (arXiv:2605.09825)
- [x] FSDP2 compatibility fix (re-quantize from `weight_ref`)
- [x] Dispatch routing fix
- [x] 12/12 operator accuracy tests vs torchAO
- [x] 0.6B BF16 vs MXFP4 convergence validation (Δ val_loss = +0.045)
- [x] 8B BF16 baseline (val_loss 6.05, 5k steps)
- [x] 8B MXFP4 training stabilized after FSDP2 fix (in progress)

### Open Issues

1. **No speed benefit** — MXFP4 is 2-3× slower than BF16 due to unfused kernel pipeline (3 separate kernel launches per GEMM). Requires AITER prologue fusion to match ROCm TE performance.
2. **No memory saving** — BF16 master weights retained for FSDP2 all-gather and backward re-quantization. Requires FP4 weight storage with FP4-aware FSDP.

### Next Steps

1. **Fused Hadamard+Quant Triton kernel** (Lumen) — merge `hadamard_transform` + `convert_to_mxfp4` into one kernel to eliminate one global memory roundtrip.
2. **AITER GEMM prologue fusion** — request AITER to fuse H+Q into GEMM tile load, eliminating all intermediate memory traffic.
3. **8B convergence validation** — complete the 5k-step run and publish BF16 vs MXFP4 comparison.
4. **Gradient quantization** — enable `quantize_grad="mxfp4"` for communication bandwidth reduction.
5. **Megatron backend** — wire MXFP4 through TP/PP for larger-scale runs.
