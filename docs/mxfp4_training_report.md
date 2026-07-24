# Feature Dev — Lumen MXFP4

By Dai, Xindi

---

## 1. Design Overview

Lumen MXFP4 implements FP4 E2M1 training for linear layers on AMD MI350X (gfx950) hardware. The design is informed by NVFP4 (NVIDIA, arXiv:2509.25149) and arXiv:2605.09825 (AMD/PSU, 2025). Forward and DGrad GEMMs are computed in MXFP4; WGrad uses MXFP4 with deterministic Hadamard rotation to stabilize convergence. Following both papers, the most sensitive layers (last ~15% of transformer blocks) are kept in BF16 (see §1.5).

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

### 1.3 Backward WGrad: dW = Q(dY^T·H) @ Q(X^T·H)^T

| Operand       | Format          | Rounding | Notes                                    |
|---------------|-----------------|----------|------------------------------------------|
| Gradient dY^T | MXFP4 (1×32)   | SR       | Transpose → Hadamard → quantize          |
| Activation X^T| MXFP4 (1×32)   | SR       | Dequant saved FP4 → BF16 → transpose → Hadamard → quantize |
| Output dW     | BF16            | —        | —                                        |

WGrad applies a **deterministic Hadamard rotation** (all +1 sign vector = pure H, no random diagonal) before quantization, following arXiv:2605.09825:

1. **Dequant saved activation**: `convert_from_mxfp4(X_fp4, X_scale) → X_bf16`
2. **Transpose**: dY^T (N, M) and X^T (K, M)
3. **Deterministic Hadamard**: blockwise H with G=16 along reduction dim M. Both operands receive the same H, which cancels in GEMM: (dY^T H)(X^T H)^T = dY^T X.
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

### 1.5 Mixed Precision: Last Layers in BF16

Both papers find that the **final linear layers are the most sensitive to FP4** and recommend keeping ~15% of layers (末尾为主) in higher precision for convergence stability. This is now enabled by default for MXFP4 in the pretraining script:

```218:221:examples/qwen3/pretrain_qwen3_mxfp4.py
    # MXFP4: keep last ~15% layers in BF16 (NVFP4 paper §4:末尾层最敏感)
    tail_bf16 = args.mode == "mxfp4"
    num_layers = getattr(config, "num_hidden_layers", 0)
    tail_count = max(1, round(num_layers * 0.15)) if tail_bf16 else 0
```

- Controlled by `QuantConfig.first_last_layers_bf16` + `num_layers_at_end_in_bf16` (`lumen/quantize/config.py:211-218`). Matching layers are skipped during patching (stay BF16 / unpatched), and `lm_head` is also skipped.
- MXFP4 default: `num_layers_at_start_in_bf16=0`, `num_layers_at_end_in_bf16=round(0.15·num_layers)` (≥1).
- **Note**: the 8B run in §4.3 (loss spike) predates this mechanism (all layers were MXFP4); it needs re-validation with the current default (see §8).

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

Detection: `is_cdna4()` → `target.backend == "hip" and target.arch == "gfx950"`. Verified active on MI350X. Fallbacks when the ASM path is not taken:
- **Quant (non-gfx950, RTN)**: AITER `dynamic_mxfp4_quant` Triton kernel.
- **Quant (SR, or when AITER MXFP4 quant is unavailable)**: Lumen's software E2M1 packing kernel (threshold-based nibble encoding in `_pack_fp4`, `USE_ASM=False`).
- **Dequant**: AITER `mxfp4_to_f32` / `e8m0_to_f32` when available, else a pure-Python E2M1 lookup-table path.

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

This follows the same pattern as the existing FP8 **blockwise (1D)** backward in `linear.py` (the `scaling_type == "blockwise"` DGrad branch, ~line 1530-1537), which prefers `ctx.weight_ref` (the BF16 master) to build the columnwise-requantized weight for DGrad. (The `blockwise2d` path instead transposes the saved 2D-tile weight directly, since its square-tile scale is transpose-symmetric.)

**Tradeoff**: Backward now pays an extra weight re-quantization cost (BF16→MXFP4 32×32) per layer, adding ~5-10% to backward time. This is the correct behavior — the same approach is used by ROCm Transformer Engine (arXiv:2605.09825 Figure 3).

### 2.4 Dispatch Routing Fix

- `lumen/quantize/__init__.py:290` used `config.scaling.value` (returns `"blockwise"` for MXFP4) → misrouted to FP8 blockscale GEMM.
- Fixed to `config.recipe` (returns `"mxfp4"`).

### 2.5 Hadamard Design (arXiv:2605.09825)

- **Block size G=16** (not 32): arXiv:2605.09825 shows H16 is 8% faster than H32 and equally stable; arXiv:2509.25149 also uses d=16 as recommended Hadamard size. Code: `_MXFP4_RHT_G = 16` in `linear.py:74`.
- **Deterministic sign vector** (all +1, no random diagonal): arXiv:2605.09825 proves randomized signs cause Wgrad divergence at 8B+ scale due to structured micro-scaling errors from outliers. Code: `torch.ones(_MXFP4_RHT_G, ...)` in `linear.py:81`.

### 2.6 Backward Fallback

When the M/N/K dimensions are not 32-aligned — or when the FP4 GEMM/quant kernel raises `AssertionError`/`RuntimeError` — backward falls back to BF16 GEMM using `ctx.weight_ref` directly for DGrad and the dequantized saved activation for WGrad.

---

## 3. Operator Accuracy (12/12 Tests vs torchAO)

| # | Operation                      | Lumen Op                     | Reference            | Result              |
|---|--------------------------------|------------------------------|----------------------|---------------------|
| 1 | 1D Quant (axis=-1, RTN)        | `convert_to_mxfp4`           | torchAO MXTensor     | bitwise identical   |
| 2 | 1D Dequant                     | `convert_from_mxfp4`         | torchAO MXTensor     | bitwise identical   |
| 3 | Cross-framework dequant        | `convert_from_mxfp4`         | torchAO `to_dtype`   | bitwise identical   |
| 4 | 1D Quant (axis=0, RTN)         | `convert_to_mxfp4`           | torchAO MXTensor     | bitwise identical   |
| 5 | Dual-axis quant                | `convert_to_mxfp4_dual_axis` | torchAO MXTensor     | bitwise identical   |
| 6 | Roundtrip (quant→dequant)      | `convert_to/from_mxfp4`      | torchAO MXTensor     | matches torchAO roundtrip bitwise; roundtrip SNR ≈19.0 dB vs FP32 |
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

### 4.2 Qwen3-8B: MXFP4 vs BF16 (C4, 5k steps, same lr)

Fair head-to-head comparison. BF16 baseline uses **pure PyTorch** (zero Lumen dependency): standard `AutoModelForCausalLM` + PyTorch FSDP2 (`pretrain_qwen3_bf16_baseline.py`). MXFP4 uses Lumen with H16 + last 5/36 layers BF16 + AITER attention/norm/RoPE.

| Parameter          | Value                                      |
|--------------------|--------------------------------------------|
| Model              | Qwen3-8B (dense), random init, 36 layers, head_dim=128 |
| Dataset            | C4 (allenai/c4, English, streaming)        |
| Sequence length    | 512                                        |
| Global batch size  | 16 (micro_batch=2 × 8 GPUs)               |
| Optimizer          | AdamW (β1=0.9, β2=0.95, ε=1e-8, wd=0.1)  |
| Learning rate      | **1e-4** peak, 50-step warmup, cosine decay |
| Grad clip          | 1.0                                        |
| Parallelism        | FSDP2 full_shard, 8× MI350X               |
| Steps              | 5,000                                      |
| Seed               | 1234                                       |
| BF16 framework     | Pure PyTorch (no Lumen, no AITER)          |
| MXFP4 framework    | Lumen + AITER attn/norm/RoPE, last 5 layers BF16 |

**Convergence (same lr, same seed, same data)**:

| Step  | BF16 val_loss | MXFP4 val_loss | Δ (MXFP4 − BF16) |
|------:|--------------:|---------------:|------------------:|
|   500 | 6.711         | 6.707          | −0.004            |
| 1,000 | 6.357         | 6.356          | −0.001            |
| 1,500 | 6.141         | 6.147          | +0.006            |
| 2,000 | 5.986         | 5.993          | +0.007            |
| 2,500 | 5.854         | 5.863          | +0.010            |
| 3,000 | 5.769         | 5.782          | +0.013            |
| 3,500 | 5.718         | 5.741          | +0.022            |
| 4,000 | 5.699         | 5.737          | +0.038            |
| 4,500 | 5.695         | 5.737          | +0.042            |
| 5,000 | 5.694         | 5.738          | **+0.044**        |

Loss curves nearly superimposed through 5000 steps. Final gap **+0.044 (0.8% relative)**. No NaN, no divergence, no loss spike. MXFP4 trains stably at the same lr as BF16.

**Throughput**:

| Metric           | BF16 (pure PyTorch) | MXFP4 (Lumen) | Ratio        |
|------------------|---------------------|----------------|--------------|
| Median step time | 334 ms              | 623 ms         | 1.86× slower |
| Peak memory      | 15.3 GB             | 15.3 GB        | identical    |

**Debug history**: Before stabilization, 8B MXFP4 crashed at step ~1275-3600 with all layers quantized. See §6.3 for the full ablation.

---

## 5. Performance Analysis

### 5.1 Why MXFP4 Shows No Speed Benefit Yet

MXFP4 training is **~2.1× slower** than BF16 at 0.6B (478 vs 229 ms) and **~1.9× slower** at 8B (630 vs 329 ms; see §4.3). This is a known limitation of the current implementation, not the MXFP4 algorithm.

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

| Optimization                  | Owner     | Expected impact             | Status |
|-------------------------------|-----------|------------------------------|--------|
| Fused Hadamard+Quant kernel   | Lumen     | ~~−30-40% quant overhead~~   | **Tested: no speedup.** Triton butterfly/matmul has higher overhead than `torch.matmul` + separate quant kernel at these tile sizes. See `feature/mxfp4-kernel-fusion` branch. |
| GEMM prologue fusion (H+Q+GEMM) | AITER  | Match ROCm TE (9-10% over FP8) | **Required.** The only path to actual speedup — fuse H+Q into GEMM tile load in `gemm_afp4wfp4` kernel. |
| FP4 weight storage + FSDP     | AITER + PyTorch | Memory reduction        | Not started |
| FP4 gradient communication    | Lumen     | Reduced allreduce bandwidth  | Not started |

---

## 6. Debugging History

### 6.1 Dispatch Routing Bug (Resolved)

`config.scaling.value` returned `"blockwise"` for MXFP4, routing to FP8 blockscale GEMM. Fixed to `config.recipe` which returns `"mxfp4"`. Commit `656922c`.

### 6.2 FSDP2 NaN Gradients (Resolved)

MXFP4 backward used saved FP4 weight from `save_for_backward`. FSDP2 reshards weight after forward, invalidating the saved tensor. Multi-GPU backward produced 397/399 NaN grads on step 1 without gradient checkpointing. Fixed by re-quantizing from `ctx.weight_ref`.

### 6.3 8B Convergence Ablation

8B MXFP4 convergence was stabilized through iterative debugging. All runs used lr=1e-4, warmup=50, grad_clip=1.0, FSDP2 8×MI350X. The table shows the full history from FSDP2 fix to stable training:

| # | Hadamard | Sign | G  | Tail BF16 | FSDP2 fix | Result |
|---|----------|------|----|-----------|-----------|--------|
| 1 | Random   | ±1   | 32 | none      | before    | step ~1550 crash |
| 2 | Determ.  | +1   | 32 | none      | before    | step ~1275 crash |
| 3 | Determ.  | +1   | 32 | none      | **after** | step ~1275 crash (FSDP2 fix alone not sufficient) |
| 4 | Random   | ±1   | 32 | none      | after     | step ~1525 crash |
| 5 | Determ.  | +1   | **16** | **last 5 layers** | after | **5000 steps, no crash** ✅ |

Key insights:
- FSDP2 fix (§6.2) was necessary but not sufficient for 8B stability
- Hadamard sign (random vs deterministic) alone did not fix the crash
- The combination of **H16 + last 5/36 layers BF16** eliminated the late loss spike
- Both papers agree: end-of-network layers are most sensitive to FP4, and ~15% in BF16 is the primary stabilization lever

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
| Paper comparison (AMD MXFP4) | `docs/papers/mxfp4_paper_vs_lumen_comparison.md` |
| Paper comparison (NVFP4 + hyperparams) | `docs/papers/nvfp4_paper_vs_lumen_comparison.md` |
| BF16 baseline script (pure PyTorch) | `examples/qwen3/pretrain_qwen3_bf16_baseline.py` |
| Debug flow (full history) | `docs/mxfp4_debug_flow.md` |
| Status report | `docs/mxfp4_status_report.md` |

---

## 8. Status and Next Steps

### Done

- [x] MXFP4 quantization ops: 1D/2D RTN, SR, dual-axis, dequant, transpose, Hadamard
- [x] gfx950 ASM kernels for RTN and SR
- [x] AITER `gemm_afp4wfp4` native FP4 GEMM
- [x] Full autograd forward + backward (DGrad + WGrad)
- [x] Deterministic Hadamard H16 (arXiv:2605.09825)
- [x] FSDP2 compatibility fix (re-quantize from `weight_ref`)
- [x] Dispatch routing fix (`config.recipe`)
- [x] 12/12 operator accuracy tests vs torchAO (bitwise identical)
- [x] 0.6B BF16 vs MXFP4 convergence validation (C4, 10k steps, Δ val_loss = +0.045)
- [x] Last ~15% layers BF16 enabled by default for MXFP4 (§1.5)
- [x] **8B MXFP4 stable 5000 steps** (H16 + last 5 layers BF16, val_loss 5.74, zero divergence)
- [x] **8B fair comparison**: BF16 (pure PyTorch) vs MXFP4, same lr=1e-4, Δ val_loss = **+0.044** (0.8%), §4.2
- [x] Fused H+Q kernel attempted — no speedup (Triton butterfly overhead > torch.matmul), see `feature/mxfp4-kernel-fusion` branch

### Open Issues

1. **No speed benefit** — MXFP4 is ~1.86× slower than BF16 at 8B (623 vs 334 ms) due to unfused kernel pipeline. Requires AITER GEMM prologue fusion (§5.3).
2. **No memory saving** — BF16 master weights retained for FSDP2 and backward re-quantization. Requires FP4 weight storage with FP4-aware FSDP.

### Next Steps

1. **AITER GEMM prologue fusion** — fuse H+Q into `gemm_afp4wfp4` tile load stage. This is the **only path** to actual speedup — Lumen-side fusion was tested and provides no benefit (§5.3).
2. **Gradient quantization** — enable `quantize_grad="mxfp4"` for communication bandwidth reduction in multi-node training.
3. **Megatron backend** — wire MXFP4 through TP/PP for larger-scale runs.
4. **Late-training BF16 switchover** — NVFP4 paper §4.1/Appendix D: switch Fprop to BF16 during LR decay phase to close the ~0.8% loss gap with ~6% extra compute.
