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

- Weight uses 2D (32×32) block scaling (chain-rule consistent, transpose-invariant).
- Activation uses 1D (1×32) per-group scaling along K.
- GEMM kernel: one of three AITER FP4 kernels, picked per shape by measurement (see §2.2).
- The **pre-transposed** FP4 weight is what goes into `save_for_backward` for DGrad to
  reuse — not both W and W^T, since DGrad only consumes W^T.
- Weight quantization and its pre-transpose are cached per module and invalidated after
  `optimizer.step()`, so under gradient accumulation they run once per optimizer step
  rather than once per micro-batch (see §2.7).

### 1.2 Backward DGrad: dX = Q(dY) @ W_cached^T

| Operand      | Format          | Rounding | Notes                              |
|--------------|-----------------|----------|------------------------------------|
| Gradient dY  | MXFP4 (1×32)   | SR       | Stochastic rounding for gradients  |
| Weight W^T   | MXFP4 (32×32)  | RTN      | Cached from forward, pre-transposed |
| Output dX    | BF16            | —        | —                                  |

- Weight FP4 data and pre-transposed FP4 weight are **reused from forward** via `save_for_backward`. The FP4 tensors are freshly allocated by `quantize_input` (not views of the BF16 param), so they survive FSDP2 resharding (see §2.3).
- 2D (32×32) block scales are transpose-invariant: transposing the FP4 data and scales preserves quantization correctness (NVFP4 paper §4.3, Quartet II).
- Pre-transposed weight (`transpose_packed_fp4` + `scale.t()`) is computed in forward to move work off the backward critical path.
- GEMM kernel: same `gemm_afp4wfp4`.

### 1.3 Backward WGrad: dW = fused_HQ(dY^T) @ fused_HQ(X^T)^T

| Operand       | Format          | Rounding | Notes                                    |
|---------------|-----------------|----------|------------------------------------------|
| Gradient dY^T | MXFP4 (1×32)   | **SR**   | Transposed **view** → fused Hadamard+Quant |
| Activation X^T| MXFP4 (1×32)   | **RTN**  | Fused dequant+transpose → fused Hadamard+Quant |
| Output dW     | BF16            | —        | —                                        |

Rounding follows NVFP4 §4.4 / Appendix E.3: **stochastic rounding on the gradient only**.
On activations SR buys little and can diverge, so X^T stays round-to-nearest. (Both
operands used SR until `40e2691`.)

WGrad applies a **deterministic Hadamard rotation** (all +1 sign vector = pure H, no random diagonal) before quantization, following arXiv:2605.09825. The Hadamard transform and FP4 quantization are **fused into a single kernel** (`hadamard_quant_mxfp4`), eliminating one global memory roundtrip per operand:

1. **Activation**: `dequant_transpose_mxfp4(X_fp4, X_scale) → X^T` in BF16 `(K, M)`.
   One kernel: the BF16 `(M, K)` intermediate that a separate dequant would write is
   never materialised, and the result lands dense.
2. **Gradient**: `grad_flat.t()`, left as a **view**. `hadamard_quant_mxfp4` addresses
   its input through both strides, so it reads the transpose directly.
3. **Fused Hadamard+Quant** (single kernel per operand):
   - Load BF16 tile from global memory → registers
   - Hadamard-16 butterfly entirely in registers (zero memory traffic)
   - FP4 quantization in registers (scale computation + pack), SR for dY^T / RTN for X^T
   - Write packed FP4 + E8M0 scales
4. **GEMM**: `gemm_mxfp4_dispatch(dY^T_fp4, X^T_fp4) → dW`

When `M % 16 != 0` the Hadamard is skipped and both operands go through plain
`convert_to_mxfp4` instead (same SR/RTN split). That branch needs a dense operand,
which is the second reason the activation goes through the fused kernel rather than
staying a view.

### 1.4 Per-Layer Operation Count

**Current**:

| Phase  | FP4 Quant | Dequant+Transpose | Fused H+Q | Transpose | FP4 GEMM |
|--------|-----------|---------|-----------|-----------|----------|
| Fprop  | 2 (X + W\*) | 0     | 0         | 1 (W pre-T\*) | 1     |
| DGrad  | 1 (dY SR) | 0       | 0         | 0 (reuses fprop) | 1   |
| WGrad  | 0         | 1 (X, fused) | 2 (dY^T SR + X^T RTN) | 0 (dY^T is a view) | 1 |
| **Total** | **3**  | **1**   | **2**     | **1**     | **3**    |

\* Under gradient accumulation the two weight ops are amortised to once per optimizer
step by the module cache, so a cache-hit micro-batch runs 1 quant (X) + 0 transposes in
fprop.

**Original (before any of the optimization rounds)**:

| Phase  | FP4 Quant | Dequant | Hadamard | Transpose | FP4 GEMM |
|--------|-----------|---------|----------|-----------|----------|
| Fprop  | 2 (X + W) | 0       | 0        | 0         | 1        |
| DGrad  | 2 (dY + W re-quant) | 0 | 0    | 1 (W packed) | 1     |
| WGrad  | 2 (dY^T + X^T SR) | 1 (X) | 2 (dY^T + X^T) | 2 (dY, X) | 1 |
| **Total** | **6**  | **1**   | **2**    | **3**     | **3**    |

**Savings**: quantization 6→3, separate Hadamard 2→0 (fused into H+Q), transposes 3→1
(weight transpose moved to fprop and cached; dY^T became a view; X^T folded into the
dequant kernel), and the standalone activation dequant became a fused dequant+transpose.

Compare to BF16: 0 quant, 0 dequant, 0 Hadamard, 0 transpose, 3 BF16 GEMMs.

### 1.5 Mixed Precision: Last Layers in BF16

Both papers find that the **final linear layers are the most sensitive to FP4** and recommend keeping ~15% of layers — the tail ones above all — in higher precision for convergence stability. This is now enabled by default for MXFP4 in the pretraining script:

```265:268:examples/qwen3/pretrain_qwen3_mxfp4.py
    # MXFP4: keep last ~15% layers in BF16 (NVFP4 paper §4:末尾层最敏感)
    tail_bf16 = args.mode == "mxfp4"
    num_layers = getattr(config, "num_hidden_layers", 0)
    tail_count = max(1, round(num_layers * 0.15)) if tail_bf16 else 0
```

- Controlled by `QuantConfig.first_last_layers_bf16` + `num_layers_at_end_in_bf16` (`lumen/quantize/config.py:219-221`). Matching layers are skipped during patching (stay BF16 / unpatched), and `lm_head` is also skipped.
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

Detection: `is_cdna4()` → `target.arch == "gfx950"`. Verified active on MI350X. Non-gfx950 falls back to software LUT-based FP4 E2M1 conversion.

### 2.2 GEMM Dispatch

Lumen owns no FP4 GEMM kernel; it reaches three that AITER ships, and which one wins
depends on the shape, so the choice is made per shape by measuring:

```
dispatch_gemm(scaling_type="mxfp4")
  → gemm_mxfp4_dispatch()
      → _mxfp4_probe_backends()          # once per process: which kernels exist here
      → _mxfp4_choose_backend()          # per (M, N, K), cached by mxfp4_autotune
          asm       aiter.gemm_a4w4               (prebuilt ASM/CK, tuned table)
          shuffled  gemm_afp4wfp4_preshuffle      (Triton, tiled B + scales)
          plain     gemm_afp4wfp4                 (Triton, row-major)
      → fast path (LUMEN_FAST_QUANT_DISPATCH=1, default):
            _MXFP4_BACKENDS[name](...)   # module-level dict, no list/lambda building
      → slow path (LUMEN_FAST_QUANT_DISPATCH=0):
            try_backends([chosen, ...others, (TRITON, _gemm_mxfp4_fallback)])
```

On MI350X with AITER all three are reachable and the dequant→BF16 fallback never runs.
The fast path costs 1.7–3.4 µs per call once the shape's decision is cached, about 1% of
a 0.2 ms GEMM. `_fast_mxfp4_gemm_fn` still exists but now only gates whether the plain
kernel is reachable at all.

Measured decisions from the Qwen3-8B run in §4.4 (8192 tokens/GPU, autotune's own
11-iteration medians, logged at startup):

| (M, N, K) | plain | shuffled | asm | chosen |
|---|---|---|---|---|
| 8192×4096×4096 | 0.310 | 0.383 | 0.362 | plain |
| 8192×1024×4096 | 0.260 | 0.311 | 0.353 | plain |
| 8192×4096×1024 | 0.133 | 0.178 | 0.174 | plain |
| 4096×4096×8192 | 0.169 | 0.211 | 0.185 | plain |
| 1024×4096×8192 | 0.150 | 0.150 | 0.155 | plain |
| 8192×12288×4096 | 0.394 | 0.491 | **0.340** | **asm** |
| 8192×4096×12288 | 0.370 | 0.413 | **0.303** | **asm** |
| 12288×4096×8192 | 0.362 | 0.418 | **0.305** | **asm** |
| 4096×12288×8192 | 0.352 | 0.432 | **0.308** | **asm** |

Nine distinct shapes for 21 GEMMs per layer (7 projections × 3). The four ASM shapes are
exactly the MLP projections' GEMMs — gate/up/down, 9 of the 21 instances. The attention
projections all stay on plain Triton. Mechanism and history in
[`mxfp4_gemm_backend_selection.md`](mxfp4_gemm_backend_selection.md).

### 2.3 FSDP2 Compatibility: Weight Caching

**Background**: FSDP2 `full_shard` reshards (frees) weight **parameters** after forward. The original MXFP4 implementation avoided saving FP4 weight via `save_for_backward` and instead re-quantized from `ctx.weight_ref` in backward.

**Current approach**: Forward pre-transposes the FP4 weight (`transpose_packed_fp4(w_fp4)`
+ `w_scale.t()`) and puts **that** into `save_for_backward` — the only form DGrad
consumes. These are freshly allocated tensors derived from `quantize_input` output (not
views of the BF16 parameter), so they are **not affected by FSDP2 resharding**. This
follows the same pattern as the FP8 blockwise path, which also saves `weight_desc.data`
and `weight_desc.scale`. It removes both the BF16→FP4 re-quantization and the packed
transpose from the backward critical path:

```1659:1673:lumen/ops/quantize/linear.py
        elif scaling_type == "mxfp4":
            # Reuse pre-transposed weight from module cache if available.
            _wt_cached = getattr(weight_desc.data, "_mxfp4_wt_cached", None)
            if _wt_cached is not None:
                w_fp4_t, w_scale_t = _wt_cached
            else:
                from lumen.ops.quantize.ops import transpose_packed_fp4
                w_fp4_t = transpose_packed_fp4(weight_desc.data)
                w_scale_t = weight_desc.scale.t().contiguous()
            ctx.save_for_backward(
                input_desc.data,
                input_desc.scale,
                w_fp4_t,
                w_scale_t,
            )
```

An earlier revision attached these to `ctx._mxfp4_w_fp4_t` / `ctx._mxfp4_w_scale_t`
instead; `660705f` moved them into `save_for_backward` so autograd tracks them normally.

**Historical note**: this is the *second* time the FP4 weight has been saved for backward.
The first attempt was blamed for the FSDP2 NaN storm and removed in `8bcf2d9` (§6.2). The
argument for bringing it back in `23644ea` is that `quantize_input` allocates fresh tensors
whose storage FSDP2 does not own, so resharding the parameter cannot invalidate them.
Empirically it holds — Qwen3-8B on 8 GPUs has trained past 1250 steps on this path with a
monotone loss curve — which means the original stale-tensor diagnosis was not the real
mechanism. What in `8bcf2d9` actually stopped the NaNs was never isolated. See
`docs/mxfp4_debug_flow.md` for the original trace.

### 2.4 Dispatch Routing Fix

- The patching path used `config.scaling.value`, which returns `"blockwise"` for MXFP4, so MXFP4 GEMMs were misrouted to the FP8 blockscale kernel (`GROUP_K` 128 against MXFP4's block size 32).
- Fixed to `config.recipe`, which returns `"mxfp4"`. Now `lumen/quantize/__init__.py:304` (`__init__.py:290` at the time of the fix, `656922c`).

### 2.5 Hadamard Design (arXiv:2605.09825)

- **Block size G=16** (not 32): arXiv:2605.09825 shows H16 is 8% faster than H32 and equally stable; arXiv:2509.25149 also uses d=16 as recommended Hadamard size. Code: `_MXFP4_RHT_G = 16` in `linear.py:80`.
- **Deterministic sign vector** (all +1, no random diagonal): arXiv:2605.09825 proves randomized signs cause Wgrad divergence at 8B+ scale due to structured micro-scaling errors from outliers. Code: `torch.ones(_MXFP4_RHT_G, ...)` in `linear.py:87`.

### 2.6 Backward Fallback

When dimensions are not 32-aligned, backward falls back to BF16 GEMM using `ctx.weight_ref` directly. The same fallback catches an `AssertionError` / `RuntimeError` from any FP4 kernel mid-path.

### 2.7 Cross-Micro-Batch Weight Cache

MXFP4 weight quantization is RTN, hence deterministic: the same BF16 tensor quantizes to
bit-identical FP4 every time. Within one optimizer step the BF16 weight does not change,
so every micro-batch after the first was re-deriving an identical FP4 weight plus an
identical pre-transpose — and gradient checkpointing paid for it again when it re-ran the
forward.

The cache lives on the module (`module._mxfp4_w_cache`), with the pre-transposed pair
hanging off the FP4 tensor as `_mxfp4_wt_cached`. A post-step hook on the optimizer
clears it:

```828:833:lumen/quantize/__init__.py
    def _post_step(opt, args, kwargs):
        for m in model.modules():
            if hasattr(m, "_mxfp4_w_cache"):
                del m._mxfp4_w_cache

    optimizer.register_step_post_hook(_post_step)
```

Registered by `register_mxfp4_weight_optimizer_hooks(model, opt)`; the pretraining script
does this in `mxfp4` mode. `LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` turns it off for A/B runs.

FSDP2-safe for the same reason §2.3 is: the cached tensors are independent allocations,
not views of the parameter. The cost is memory — the FP4 weight and its pre-transpose for
every quantized layer stay live for the whole step, measured at +4.8 GB/GPU on Qwen3-8B
(§4.4).

### 2.8 FP4 Parameter All-Gather

`MXFP4CommTensor` (`lumen/quantize/comm_tensor.py`) wraps a BF16 parameter and gives
FSDP2 two hooks: `fsdp_pre_all_gather` quantizes the local BF16 shard to packed MXFP4
(2D 32×32 scales), `fsdp_post_all_gather` dequantizes back to BF16. The wire carries
**3.99x fewer bytes** (0.5 byte/element + E8M0 scales vs 2 bytes). Everything downstream
— forward, gradients, optimizer — sees a normal BF16 weight.

Requires `N % (32 × world_size) == 0` and `K % 32 == 0` so a rank's dim-0 shard never
splits a 32-row tile; weights that fail the check keep BF16 all-gather. Qwen3-8B on 8
GPUs wraps 217 weights.

The gathered weight is `RTN(W)` rather than `W`. This does not accumulate — the optimizer
updates the full-precision sharded master, only the gathered copy is rounded — and since
the forward quantizes it to FP4 with the same block size and rounding anyway, the FP4
operand the GEMM sees is unchanged.

Enabled by default in `mxfp4` mode, off with `--no-mxfp4-comm`. **Whether it is a net win
is unmeasured**: `convert_from_mxfp4_2d` is pure PyTorch and materialises several
full-size intermediates including an int64, which may cost more than the saved bandwidth
on a single node. See
[`mxfp4_optimization_report.md`](mxfp4_optimization_report.md) 改动 7.

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

**Throughput** (this run, seq 512, before the optimization rounds of §5.1):

| Metric           | BF16   | MXFP4  | Ratio        |
|------------------|--------|--------|--------------|
| Median step time | 229 ms | 478 ms | 2.09× slower |

The 0.6B configuration has not been re-measured since; §4.4 covers 8B, which is where the
optimization work was targeted. Do not read the 2.09× as current.

### 4.2 Qwen3-8B: BF16 Baseline (C4, 5k steps)

BF16 8B trained with the same hyperparameters (lr=6e-5, warmup=200):

| Step   | BF16 val_loss |
|-------:|--------------:|
|    500 | 6.929         |
|  1,500 | 6.380         |
|  2,500 | 6.158         |
|  3,500 | 6.062         |
|  5,000 | 6.048         |

Stable convergence throughout. Median step time: 329 ms. Peak memory: 15.3 GB/GPU.

### 4.3 Qwen3-8B: MXFP4 (C4 — stabilized with H16 + tail BF16)

**History**: After the FSDP2 fix (§2.3), 8B MXFP4 with all layers quantized and H32 crashed at step ~1275-3600 (loss spike to 11.94). See §6.3 for ablation details.

**Current (with H16 + last 5 layers BF16, §1.5 + §2.5)**: 5000 steps, zero divergence.

Config: lr=1e-4, warmup=50, cosine decay, grad_clip=1.0, FSDP2, `--aiter-attn --lumen-norm --fuse-rope`, last 5/36 layers BF16.

| Step  | BF16 val_loss (lr=6e-5) | MXFP4 val_loss (lr=1e-4) |
|------:|------------------------:|-------------------------:|
|   250 | 7.438                   | 7.071                    |
|   500 | 6.929                   | 6.715                    |
| 1,000 | 6.563                   | 6.366                    |
| 1,500 | 6.380                   | 6.153                    |
| 2,000 | 6.252                   | 5.999                    |
| 2,500 | 6.158                   | 5.870                    |
| 3,000 | 6.091                   | 5.790                    |
| 3,500 | 6.062                   | 5.748                    |
| 4,000 | 6.049                   | 5.744                    |
| 5,000 | 6.048                   | **5.743**                |

**Note**: MXFP4 val_loss is lower than BF16 because MXFP4 used lr=1e-4 (higher effective lr, more aggressive training) while BF16 used lr=6e-5. The comparison validates convergence stability rather than absolute quality parity — a fair head-to-head comparison at the same lr would require re-running BF16 at lr=1e-4.

**Throughput** (this run, seq 512, before the optimization rounds of §5.1):

| Metric           | BF16   | MXFP4  | Ratio        |
|------------------|--------|--------|--------------|
| Median step time | 329 ms | 630 ms | 1.91× slower |
| Peak memory      | 15.3 GB | 15.3 GB | identical  |

Superseded — see §4.4.

### 4.4 Qwen3-8B: Current Throughput (65536 tokens/step)

Qwen3-8B, 8× MI350X, FSDP2 full_shard, C4 streaming, seq 2048 × micro_batch 4 × 8 GPUs
= 65536 tokens/step, lr 3e-4, warmup 200. BF16 and MXFP4 differ only in `--mode`.
Step time is the median of the logged `step_time_ms` over steps ≥ 50 (C4 streaming
stalls produce occasional multi-second outliers that inflate the mean by ~40% but leave
the median stable to within ±1.5% across several windows).

| Build | Median step | P25 | Fastest | Peak mem/GPU | vs BF16 |
|---|---|---|---|---|---|
| BF16 | 928.0 ms | 917.0 | 910.7 | 15.30 GB | 1.00× |
| MXFP4, GEMM round only (`7d1841b`) | 1061.8 ms | 1001.1 | 989.0 | 16.10 GB | 0.87× |
| MXFP4, current (`035431e`) | **869.4 ms** | 865.0 | 860.2 | **20.90 GB** | **1.067×** |

**MXFP4 is now 6.7% faster than BF16**, at the cost of 5.6 GB more memory (+37%) from
the weight cache (§2.7). The gain over the previous MXFP4 build is 1.221×, and it comes
from the four changes in §5.1 D–G collectively — they have not been A/B'd individually.

Convergence on the current build (WGrad rounding changed in §5.1 G, so not point-comparable
with §4.3):

| Step | 500 | 700 | 900 | 1000 | 1100 | 1200 |
|---|---|---|---|---|---|---|
| val_loss | 5.1367 | 4.9633 | 4.7664 | 4.6359 | 4.5352 | 4.4441 |

Monotone, no spikes. This run was at step ~1250/3000 when the numbers above were taken.

---

## 5. Performance Analysis

### 5.1 Optimizations Applied

Three rounds, in the order they landed:

**A. Forward weight caching for DGrad** — Forward pre-transposes the FP4 weight and saves
that via `save_for_backward`. DGrad reuses it instead of re-quantizing from BF16 and
transposing in backward. Eliminates 1 weight re-quantization + 1 packed transpose from the
backward critical path.

**B. Fused Hadamard+Quant kernel in WGrad** — Replaces separate `hadamard_transform` + `convert_to_mxfp4` with a single `hadamard_quant_mxfp4` kernel that performs the Hadamard-16 butterfly entirely in registers and quantizes to FP4 without writing intermediate BF16 to global memory. Eliminates 2 kernel launches + 2 global memory roundtrips per layer.

**C. Fast GEMM dispatch** — `gemm_mxfp4_dispatch` probes once, then selects a backend by
shape from a module-level dict, bypassing `try_backends`'s list and closure construction.
1.7–3.4 µs per call, ~1% of a 0.2 ms GEMM (§2.2).

**D. Per-shape GEMM backend selection + vectorized weight shuffle** — Three AITER FP4
kernels, chosen by measuring each shape on its first call; the shuffled path's weight
layout conversion was also rewritten to be vectorized. Together they take Qwen3-8B's
summed per-layer GEMM time from 0.90× BF16 to 1.12×, but only +0.4% end to end. The real
value was avoiding the −2.6% regression that hand-fitted byte thresholds would have
shipped. Details in [`mxfp4_gemm_backend_selection.md`](mxfp4_gemm_backend_selection.md).

**E. WGrad stops materialising transposes** — `hadamard_quant_mxfp4` addresses its input
through both strides, so `grad_flat.t()` can stay a view. The two `.t().contiguous()`
copies it replaced were 7.8% of a step, more GPU time than all the MXFP4 GEMMs together.

**F. Fused dequant+transpose kernel** — `dequant_transpose_mxfp4` reads packed FP4
`(M, K/2)` and writes BF16 `(K, M)` in one launch, so WGrad's activation operand never
materialises the BF16 `(M, K)` intermediate and still lands dense. Bit-exact with
`convert_from_mxfp4(...).t().contiguous()`.

**G. Cross-micro-batch weight cache + FP4 all-gather + WGrad RTN** — §2.7, §2.8, and the
NVFP4 §4.4 rounding fix. Together with F these are the 1.221× in §4.4.

**Kernel launch reduction per layer**:

| Path    | Original | Current | Eliminated |
|---------|--------------------------|-------------------------|------------|
| Fprop   | 3 (quant_X + quant_W + GEMM) | 4 (quant_X + quant_W + transpose_W + GEMM), or 2 on a weight-cache hit | −1, or −1 net with the cache |
| DGrad   | 4 (quant_dY + requant_W + transpose_W + GEMM) | 2 (quant_dY + GEMM) | **−2** |
| WGrad   | 6 (dequant_X + H_dY + quant_dY + H_X + quant_X + GEMM) | 4 (fused dequant+transpose_X + fused_HQ_dY + fused_HQ_X + GEMM) | **−2** |
| **Total** | **13** | **10**, or 8 on a weight-cache hit | **−3 to −5** |

### 5.2 Remaining Performance Gap

The main remaining bottleneck is the **unfused GEMM prologue**: quant and GEMM remain separate kernel launches. The ideal implementation (ROCm Transformer Engine, arXiv:2605.09825) fuses quant into the GEMM tile load:

```
Ideal fused pipeline (single kernel):
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
| Fused H+Q (1 operand) | ~35       | Replaces H(49)+Q(76) separately |
| Packed FP4 transpose  | 70        | In forward, and skipped entirely on a weight-cache hit |

The Qwen3-8B profile tells the same story at scale: over one step, the 21 MXFP4 GEMMs per
layer summed to less GPU time than the two `.t().contiguous()` copies in WGrad that §5.1 E
removed. Quantization and layout, not the GEMM, dominate what is left.

### 5.3 Memory Usage

MXFP4 costs memory rather than saving it, in three places:

| Source | Qwen3-8B, 8 GPU, measured peak | Note |
|---|---|---|
| BF16 baseline | 15.30 GB | — |
| + FP4 operands, saved FP4 weight + pre-transpose, all-gather scratch | 16.10 GB (+0.8) | Per-layer, freed as backward walks down |
| + cross-micro-batch weight cache (§2.7) | **20.90 GB (+4.8)** | Every quantized layer stays live for the whole step |

BF16 master weights are still retained for FSDP2, so there is no weight-storage saving to
offset any of this. `LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` trades the §4.4 speed back for
the 4.8 GB. Caching only the pre-transposed form (DGrad is its only consumer) would
recover roughly half.

### 5.4 Remaining Path to Performance Parity

| Optimization                  | Owner     | Status        | Measured impact             |
|-------------------------------|-----------|---------------|-----------------------------|
| Forward weight caching (save_for_backward) | Lumen | **Done** ✅ | Eliminates DGrad re-quant + transpose |
| Fused Hadamard+Quant kernel   | Lumen     | **Done** ✅   | −2 kernels, −2 mem roundtrips per layer |
| Fast MXFP4 GEMM dispatch      | Lumen     | **Done** ✅   | 1.7–3.4 µs/call, ~1% of a GEMM |
| Per-shape backend selection + vectorized shuffle | Lumen | **Done** ✅ | GEMM-level 0.90× → 1.12×; +0.4% end to end |
| WGrad transpose materialization removed | Lumen | **Done** ✅ | **+7.8%** end to end |
| Fused dequant+transpose kernel | Lumen    | **Done** ✅   | −1 kernel, −1 full BF16 intermediate in WGrad |
| Cross-micro-batch weight cache | Lumen    | **Done** ✅   | Part of the 1.221× in §4.4; costs +4.8 GB |
| FP4 parameter all-gather      | Lumen     | **Done**, unverified | 3.99× less all-gather traffic; net effect unmeasured |
| GEMM prologue fusion (H+Q+GEMM) | AITER   | Future        | Match ROCm TE (9-10% over FP8) |
| Triton `convert_from_mxfp4_2d` | Lumen    | Future        | Would make §2.8 unambiguously a win |
| FP4 weight storage + FSDP     | AITER + PyTorch | Future  | Memory reduction            |
| FP4 gradient communication    | Lumen     | Future        | Reduced reduce-scatter bandwidth |

---

## 6. Debugging History

### 6.1 Dispatch Routing Bug (Resolved)

`config.scaling.value` returned `"blockwise"` for MXFP4, routing to FP8 blockscale GEMM. Fixed to `config.recipe` which returns `"mxfp4"`. Commit `656922c`.

### 6.2 FSDP2 NaN Gradients (Resolved)

Multi-GPU backward produced 397/399 NaN grads on step 1 without gradient checkpointing. Forward was saving the packed FP4 weight via `save_for_backward`; the diagnosis at the time was that FSDP2's post-forward reshard invalidated it. `8bcf2d9` stopped saving the FP4 weight and re-quantized from `ctx.weight_ref` in backward instead. The NaNs went away, at the cost of a full weight quantization plus a packed transpose in backward.

`23644ea` then put the saved FP4 weight back (§2.3, §5.1 A) on the grounds that the quant kernel's outputs are fresh allocations rather than parameter views. That path has since trained Qwen3-8B on 8 GPUs past 1250 steps with a monotone loss curve, so the stale-tensor explanation does not survive: if it were correct, the current code would NaN. The NaNs were real and `8bcf2d9` did stop them, but the mechanism was something else in that commit — it also rerouted the unaligned BF16 fallback to `weight_ref` — and was never isolated. Worth revisiting if FP4 weight reuse ever misbehaves under a new parallelism configuration.

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
| Quantization ops (incl. `dequant_transpose_mxfp4`, `hadamard_quant_mxfp4`) | `lumen/ops/quantize/ops.py` |
| GEMM dispatch + autograd | `lumen/ops/quantize/linear.py` |
| Per-shape backend measurement + cache | `lumen/ops/quantize/mxfp4_autotune.py` |
| Weight cache + optimizer hook, dispatch routing | `lumen/quantize/__init__.py` |
| FP4 all-gather wrapper (`MXFP4CommTensor`) | `lumen/quantize/comm_tensor.py` |
| FSDP2 wrapping + eligibility checks | `lumen/models/fsdp.py` |
| Unit tests (12 ops) | `tests/ops/test_quantize.py` |
| Accuracy report script | `scripts/mxfp4_accuracy_report.py` |
| Offline shape tuning script | `scripts/mxfp4_tune_shapes.py` |
| Pretraining script (C4 + TensorBoard) | `examples/qwen3/pretrain_qwen3_mxfp4.py` |
| SFT script (BF16/FP8/MXFP4) | `examples/qwen3/train_qwen3_fsdp.py` |
| Paper comparison (AMD MXFP4) | `docs/papers/mxfp4_paper_vs_lumen_comparison.md` |
| Paper comparison (NVFP4 + hyperparams) | `docs/papers/nvfp4_paper_vs_lumen_comparison.md` |
| Optimization report (measured, changes 1–8) | `docs/mxfp4_optimization_report.md` |
| GEMM backend selection deep-dive | `docs/mxfp4_gemm_backend_selection.md` |
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
- [x] FSDP2 compatibility (FP4 outputs via `save_for_backward`, `660705f`; §2.3, §6.2)
- [x] Dispatch routing fix (`config.recipe`)
- [x] 12/12 operator accuracy tests vs torchAO (bitwise identical)
- [x] 0.6B BF16 vs MXFP4 convergence validation (C4, 10k steps, Δ val_loss = +0.045)
- [x] 8B BF16 baseline (C4, 5k steps, val_loss 6.05)
- [x] Last ~15% layers BF16 enabled by default for MXFP4 (§1.5)
- [x] **8B MXFP4 stable 5000 steps** (H16 + last 5 layers BF16, val_loss 5.74, zero divergence, §4.3)
- [x] **Forward FP4 weight caching** — DGrad reuses forward's FP4 weight instead of re-quantizing from BF16 (2D block scales are transpose-invariant)
- [x] **Pre-transposed weight** — weight transpose moved from backward to forward
- [x] **Fused Hadamard+Quant kernel** — `hadamard_quant_mxfp4` wired into WGrad backward (−2 kernels, −2 mem roundtrips)
- [x] **Fast MXFP4 GEMM dispatch** — probe once, then a per-shape dict lookup (§2.2)
- [x] **Per-shape GEMM backend selection** — 3 AITER kernels measured per shape and cached (`mxfp4_autotune.py`)
- [x] **WGrad transpose materialization removed** — strided kernel access instead of `.t().contiguous()`, +7.8% end to end
- [x] **Fused dequant+transpose kernel** — `dequant_transpose_mxfp4`, bit-exact with the two-op form
- [x] **WGrad activation uses RTN, not SR** — NVFP4 §4.4: SR belongs on gradients only (§1.3)
- [x] **Cross-micro-batch weight cache** — module-level, invalidated by an optimizer post-step hook (§2.7)
- [x] **FP4 parameter all-gather** — `MXFP4CommTensor`, 3.99× less wire traffic (§2.8)
- [x] **Operation count reduced** — 3 quant ops (was 6), 2 fused H+Q, 1 fused dequant+transpose, 0 materialized transposes
- [x] **8B MXFP4 faster than BF16** — 869.4 ms vs 928.0 ms, 1.067× (§4.4)

### Open Issues

1. **GEMM prologue fusion** — quant and GEMM remain separate kernel launches. Full fusion requires AITER changes. This is the largest remaining gap (§5.2).
2. **Memory regressed, not saved** — 15.30 → 20.90 GB (§5.3). BF16 master weights are still retained for FSDP2, and the weight cache adds 4.8 GB. True savings need FP4 weight storage with FP4-aware FSDP.
3. **FP4 all-gather benefit unverified** — the dequant path is pure PyTorch and may cost more than the bandwidth it saves on a single node (§2.8).
4. **Changes 5–8 were not A/B'd individually** — the 1.221× in §4.4 is their combined effect, measured on a shared machine where isolated micro-benchmarks were unreliable.
5. **BF16 vs MXFP4 not compared at same lr in §4.3** — 8B BF16 ran at lr=6e-5, MXFP4 at lr=1e-4. §4.4 uses the same lr for both.

### Next Steps

1. **Finish the current 3000-step 8B run** for a same-lr convergence comparison against the BF16 baseline at the new build.
2. **Isolate the FP4 all-gather** with a dedicated `--no-mxfp4-comm` A/B on an idle machine, and port `convert_from_mxfp4_2d` to Triton if the gather is bandwidth-bound.
3. **Cache only the pre-transposed weight** to recover ~half of the 4.8 GB (§5.3).
4. **AITER GEMM prologue fusion** — request AITER to fuse H+Q into GEMM tile load, eliminating all intermediate memory traffic.
4. **Gradient quantization** — enable `quantize_grad="mxfp4"` for communication bandwidth reduction in multi-node training.
5. **Megatron backend** — wire MXFP4 through TP/PP for larger-scale runs.
