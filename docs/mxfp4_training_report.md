# Feature Dev — Lumen MXFP4

By Dai, Xindi

Test status, coverage gaps and release gates live in
[`mxfp4_test_report.md`](mxfp4_test_report.md), which is the only place MXFP4
test results are recorded.

---

## 1. Design Overview

Lumen MXFP4 implements FP4 E2M1 training for linear layers on AMD MI350X (gfx950) hardware. The design is informed by NVFP4 (NVIDIA, arXiv:2509.25149) and arXiv:2605.09825 (AMD/PSU, 2025). Forward and DGrad GEMMs are computed in MXFP4; WGrad uses MXFP4 with deterministic Hadamard rotation (AMD/PSU) to stabilize convergence. Following the NVFP4 paper, the most sensitive layers (last ~15% of transformer blocks) are kept in BF16 (see §1.5).

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

The NVFP4 paper (§4, Appendix E.2) finds that the **final linear layers are the most sensitive to FP4** and recommends keeping ~15% of layers — the tail ones above all — in higher precision for convergence stability. The AMD/PSU MXFP4 paper does not study layer-position sensitivity at all: it quantizes every transformer linear layer against an FP8 baseline and attributes instability to the WGrad pass. This is now enabled by default for MXFP4 in the pretraining script:

```265:268:examples/qwen3/pretrain_qwen3_mxfp4.py
    # MXFP4: keep last ~15% layers in BF16 (NVFP4 paper §4:末尾层最敏感)
    tail_bf16 = args.mode == "mxfp4"
    num_layers = getattr(config, "num_hidden_layers", 0)
    tail_count = max(1, round(num_layers * 0.15)) if tail_bf16 else 0
```

- Controlled by `QuantConfig.first_last_layers_bf16` + `num_layers_at_end_in_bf16` (`lumen/quantize/config.py:219-221`). Matching layers are skipped during patching (stay BF16 / unpatched), and `lm_head` is also skipped.
- MXFP4 default: `num_layers_at_start_in_bf16=0`, `num_layers_at_end_in_bf16=round(0.15·num_layers)` (≥1).
- **Note**: the 8B run in §4.3 (loss spike) predates this mechanism (all layers were MXFP4); it needs re-validation with the current default (see §8).
- **Cost**: 78.8 ms/step per layer held in BF16, flat across the sweep, so the default tail of 5 is 6.4% of step time on Qwen3-8B (§5.14). The same section shows why the matching accuracy question cannot be answered by a short run.

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

### 2.9 Megatron Backend

Megatron reaches MXFP4 through the same `quant.enable` patching as FSDP2 — it
patches Megatron's own `ColumnParallelLinear` / `RowParallelLinear`, so no
Lumen-specific module is involved. Three things differ from the FSDP2 path:

- **Format selection.** Megatron's `--fp8-format` has no MX formats among its
  argparse choices, so the format comes from Lumen's own
  `--linear-fp8-format mxfp4`. It defaults to `None` and only then takes
  precedence over `--fp8-format`, so existing FP8 runs are unaffected.
  `_override_te_args_for_lumen` also pins the block size to 32 for MXFP4 —
  a mismatched `--linear-fp8-block-size` would hand the FP4 GEMM scales of the
  wrong shape rather than fail.
- **Weight cache invalidation.** `install_mxfp4_weight_cache_hook()` wraps
  `setup_model_and_optimizer` to register the §2.7 hook, because Megatron's
  `ChainedOptimizer` / `DistributedOptimizer` are not `torch.optim.Optimizer`
  subclasses and have no `register_step_post_hook` (the helper wraps `step()`
  in that case). Without it the cached FP4 weight never expires and the run
  silently trains against the step-0 weights.
- **Do not pass `--lumen-linear`.** It swaps in `LumenColumnParallelLinear`,
  which `enable_fp8_for_parallel_linear` configures from the
  `--linear-fp8-scaling` string (`"blockwise"`) rather than the resolved
  recipe (`"mxfp4"`) — the same misrouting shape as §6.1, on a different path.

Tail BF16 works unchanged: `--first-last-layers-bf16` and
`--num-layers-at-{start,end}-in-bf16` are Megatron-native args that
`LumenConfig.from_args` picks up, and `_build_bf16_skip_prefixes` keys off
Megatron's global 1-indexed `layer_number`, so it stays correct under pipeline
parallelism.

The FP4 all-gather of §2.8 is FSDP2-only — it hangs off
`fsdp_pre_all_gather` / `fsdp_post_all_gather`, which Megatron's distributed
optimizer never calls. Launcher: `examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh`
(TP=PP=1), measured in §4.5.

**Pin Megatron to `core_r0.15.0_rocm`.** The patches in `megatron_patches.py`
carry copies of Megatron internals and track that lineage. On `rocm_dev` HEAD,
`TransformerLayer._forward_attention` gained a `padding_mask` that
`_patched_fwd_attn` forwards straight into `Attention.forward`, which does not
accept it. HEAD also passes `config` and `pg_collection` to the model provider;
that part is handled — the provider takes both, forwards `config`, and drops
`pg_collection` because Megatron defaults it to the same
`ProcessGroupCollection.use_mpu_process_groups()` that `GPTModel` derives on its
own. The `padding_mask` mismatch is not.

The launcher also runs without TransformerEngine or apex, in which case Megatron
falls back to `WrappedTorchNorm` and torch Adam, and rope fusion has to be turned
off with `--no-rope-fusion` — `apply_rope_fusion` defaults to on and asserts on a
fused kernel only TE or Lumen's apex bridge supplies.

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

### 4.5 Qwen3-8B: Megatron Backend, MXFP4 vs BF16 (C4, 200 steps)

Qwen3-8B 36 layers from random init, 8× MI350X, Megatron with TP=PP=CP=1 and the
distributed optimizer, seq 8192 × micro-batch 2 × global batch 32 = 262k
tokens/step, lr 1e-5 cosine, warmup 2, last 5 layers BF16. Both runs come from
`examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh` and differ only in `PRECISION`.
Corpus is the first C4 `en` train shard trimmed to 53.07M tokens = 6477 sequences;
200 steps consume 6400 of them, so nothing repeats — but only by a 1.2% margin.

| iter | 1 | 25 | 50 | 75 | 100 | 125 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|
| MXFP4 | 12.780 | 8.950 | 7.807 | 7.208 | 6.942 | 6.779 | 6.798 | 6.809 |
| BF16 | 12.775 | 9.036 | 7.884 | 7.210 | 6.938 | 6.769 | 6.783 | 6.790 |
| Δ | +0.005 | −0.086 | −0.077 | −0.002 | +0.005 | +0.010 | +0.015 | +0.019 |

Final loss on the eval batches 6.6785 (MXFP4) vs 6.6574 (BF16), Δ = +0.0211
(+0.32%), with zero NaN and zero skipped iterations in both. Both runs share seed,
data path and order, so a given iteration sees the identical batch in both and the
per-step Δ is a paired quantity; the run's second eval pass, on a different batch,
gives +0.0242, so the two paired estimates agree to 0.003.

**These two numbers are not held-out validation loss.** `PretrainTextDataset`
builds train, valid and test each from its own path starting at chunk 0 and ignores
`--split`, and both runs pointed all three paths at the same file — so the "eval"
batches are the first sequences of the training set, seen in step 1. The figure
measures how the two precisions memorise an early batch, which is why it belongs
with the training curve above rather than being read as generalization. The paired
Δ is unaffected: both arms measure the same batch. `VALID_JSONL` (§4.6) now takes a
disjoint corpus slice so later runs report a real held-out number.

The per-step Δ is indistinguishable from zero between steps 60 and 95 (mean
−0.0065, sd 0.0084) — quantization noise on a randomly initialised model, not a
quality edge — first stays positive for ten consecutive steps at 94, and then
drifts upward rather than levelling off:

| Δ window mean | 100–124 | 125–149 | 150–174 | 175–200 |
|---|---|---|---|---|
| lm loss Δ | +0.0086 | +0.0174 | +0.0186 | +0.0203 |
| sd | 0.0032 | 0.0079 | 0.0019 | 0.0017 |

So at 200 steps the gap is still widening, consistent with the 0.6B run climbing
from +0.024 at 500 steps to +0.045 at 10k (§4.1). Nothing here shows where it
settles, so this validates the path rather than the recipe at scale.

Step time, median over iterations 11–200. Iteration 1 costs 20.3 s under MXFP4 and
20.9 s under BF16 and iteration 2 costs ~3 s in both, so that warmup is allocator
and kernel load common to both builds, not MXFP4 autotune; both reach steady state
by iteration 3, which makes the 11–200 window conservative rather than load-bearing:

| Config | Build | Median | p25 | p75 | Fastest | tokens/s | vs BF16 |
|---|---|---|---|---|---|---|---|
| gbs 32, 262k tok/step | BF16 | 2804 ms | 2801 | 2808 | 2781 | 93,496 | 1.00× |
| | MXFP4 | **2477 ms** | 2474 | 2503 | 2460 | 105,831 | **1.132×** |
| gbs 128, 1049k tok/step | BF16 | 10712 ms | 10693 | 10768 | 10675 | 97,893 | 1.00× |
| | MXFP4 | **9352 ms** | 9328 | 9393 | 9313 | 112,126 | **1.145×** |

p25/p75 sit within 1% of the median in all four rows, and the two configurations
agree to within 1.2% despite a 4× difference in batch size. Not comparable to the
1.067× of §4.4: this host has neither TransformerEngine nor apex, so both runs fall
back to torch norms, torch Adam and unfused RoPE, and BF16 loses more to that than
MXFP4 does — part of the 1.13× is a weakened baseline.

Device memory occupancy (`1 - free/total` on rank 0, so it includes the caching
allocator's reserved pool) is 155.3 GiB for MXFP4 against 169.5 GiB for BF16 of the
251.7 GiB available — **14.2 GiB less**, the opposite direction from the +5.6 GB
that §5.3 measures under FSDP2. The plausible cause is that Megatron's distributed
optimizer shards master weights and optimizer state across the 8 ranks, so the
weight cache's per-rank increment is diluted while the activation saving is not.
That is unattributed: no per-tensor accounting was done, and this metric is not
`max_memory_allocated`.

Both runs launched without `--wandb-project`, which is what Megatron gates its
wandb writer on, so their metrics existed only in the stdout logs. They were
replayed into `daixindi-amd/qwen3-8b-mxfp4` as `megatron-mxfp4-8b-c4-200`,
`megatron-bf16-8b-c4-200` and the paired-difference run
`megatron-mxfp4-vs-bf16-delta-c4-200` by
`examples/qwen3/scripts/wandb_backfill_megatron_log.py` — the curves are the
training run's, the wall-clock timestamps are the replay's. Set `WANDB_PROJECT` on
the launcher to log live instead; note Megatron writes wandb from the last rank,
not rank 0.

These two runs used `--eval-iters 1 --eval-interval 200`, i.e. one evaluation of a
single 32-sequence batch at the very end, which is why every `val/*` metric has one
data point and wandb can only draw it as a bar. The launcher now defaults to
`EVAL_INTERVAL = TRAIN_STEPS/10` and `EVAL_ITERS=2` so validation is a curve. The
ceiling on `EVAL_ITERS` is how many sequences `VALID_JSONL` holds, not anything in
`SPLIT`: `PretrainTextDataset` never reads `--split`. Size that file for
`(TRAIN_STEPS/EVAL_INTERVAL) × EVAL_ITERS × GBS` sequences, or `__getitem__` wraps
modulo its length and successive eval points land on repeated data.

The gbs 128 rows above come from a 50-step run on the launcher's mock corpus
(repeated random token ids), which reached Δ val_loss = +0.0014. That corpus is
memorisable within 50 steps, so treat it as a smoke signal for the path and a
throughput data point, not a convergence measurement.

### 4.6 Qwen3-8B: Lumen MXFP4 vs TransformerEngine MXFP4 (C4, 1000 steps)

Both arms are MXFP4, each configured the way its own stack ships — so this measures
two products, not one recipe implemented twice (§4.6.1 spells out what that costs
in interpretability). 1000 steps, seq 8192 × micro-batch 2 × global batch 32, lr
1e-5 cosine, seed 1234, 8× MI350X, TP=PP=CP=1, distributed optimizer.

| | Lumen MXFP4 | TE MXFP4 |
|---|---|---|
| launcher | `train_qwen3_8b.sh` | `train_qwen3_8b_te.sh` |
| entry | `pretrain_llama31.py --backend megatron` | `examples/qwen3/scripts/pretrain_qwen3_te.py` |
| Megatron | `core_r0.15.0_rocm` | `rocm_dev` |
| quant selection | `--linear-fp8-format mxfp4` | `--fp4-format e2m1 --fp4-recipe mxfp4` |
| Hadamard | on (H16) | off (`MXFP4BlockScaling.use_hadamard` default) |
| BF16 layers | last 5 | none |
| transformer impl | `local` + Lumen modules | `transformer_engine` |
| attention | Lumen csrc | TE fused (CK) |
| norm / rope fusion | torch RMSNorm, no rope fusion | TE fused |
| median step time | 2007.2 ms | **1425.3 ms** |
| p25 / p75 | 2005.7 / 2009.5 | 1424.1 / 1427.1 |
| tokens/s | 130,602 | **183,922** |
| peak mem (HBM frac) | **0.6215** | 0.6355 |
| NaN / skipped iters | 0 / 0 | 0 / 0 |

Held-out validation loss, on a corpus slice disjoint from the training data
(`VALID_JSONL`, 2 × 32 sequences per eval):

| iter | 100 | 300 | 500 | 700 | 900 | 1000 |
|---|---|---|---|---|---|---|
| Lumen | 6.8427 | 6.2061 | 5.9997 | 5.8411 | 5.7585 | **5.7793** |
| TE | 6.8649 | 6.2566 | 6.0546 | 5.9073 | 5.8306 | **5.8479** |
| Δ (Lumen − TE) | −0.0222 | −0.0505 | −0.0549 | −0.0662 | −0.0720 | −0.0686 |

The two results point opposite ways, and both are large enough to matter:

- **TE is 1.41× faster** per step (Lumen needs 40.8% more time for the same batch).
  Step times are extremely tight in both arms (p75−p25 under 4 ms), so this is not
  measurement noise.
- **Lumen converges better by 0.069 nats** on held-out data at step 1000, and the
  paired per-step gap widens monotonically rather than closing: window means
  −0.0477 (201–400), −0.0590 (401–600), −0.0668 (601–800), −0.0692 (801–1000). The
  arms share seed, corpus and sample order, so each iteration sees the identical
  batch and the Δ is paired; sd within the last three windows is ≤0.0064.

Runs: [`z64anp9i`](https://wandb.ai/daixindi-amd/qwen3-8b-mxfp4/runs/z64anp9i) (Lumen),
[`82zq2fia`](https://wandb.ai/daixindi-amd/qwen3-8b-mxfp4/runs/82zq2fia) (TE),
[`c7ezs6sm`](https://wandb.ai/daixindi-amd/qwen3-8b-mxfp4/runs/c7ezs6sm) (paired Δ).

#### 4.6.1 What this comparison does and does not isolate

Neither number is attributable to the MXFP4 GEMMs alone:

- **The recipes differ.** Lumen applies the H16 Hadamard transform and keeps the
  last 5 layers in BF16; TE does neither by default. Both of Lumen's choices cost
  throughput and buy accuracy, which is the shape of the result above — so the
  headline is "Lumen's default recipe trades ~41% step time for 0.069 nats", not
  "Lumen's kernels are slower". §4.7 runs those extra arms and confirms it: 85% of
  the gap is the Hadamard, 9% the BF16 tail, and ~0.004 nats is everything else.
- **The surrounding kernels differ.** TE supplies fused norm, fused rope and CK
  fused attention; the Lumen arm runs torch RMSNorm (no apex), no rope fusion, and
  Lumen's csrc attention. Part of the 1.41× is stack-level fusion outside the
  quantized linears.
- **The Megatron versions differ** (`core_r0.15.0_rocm` vs `rocm_dev`), because the
  FP4 recipe plumbing (`megatron/core/fp4_utils.py`) only exists on `rocm_dev` while
  Lumen is pinned to the release branch.

What it does isolate is the data: `pretrain_qwen3_te.py` loads Lumen's
`PretrainTextDataset` by file path — not as `lumen.models.llama31.dataset`, which
would execute Lumen's Megatron patches through the package `__init__` — and
reproduces Lumen's batch construction call for call. Both arms therefore consume
the same token stream in the same order, which is what makes the per-step Δ paired.
Hyperparameters live in the shared `examples/qwen3/scripts/qwen3_8b_common_args.sh` so the two
launchers cannot drift apart.

#### 4.6.2 TE on ROCm: the two blockers worth knowing

TE had to be built from ROCm's fork (`dev`, 2.15.0.dev0) for gfx950; two failures
cost the most time and neither is obvious from the error text:

1. **Fused attention silently disappears without CK.** Building with
   `NVTE_FUSED_ATTN_CK=0` leaves only AOTriton, and AOTriton refuses GQA outright
   (`num_attn_heads != num_gqa_groups` → unsupported, in
   `fused_attn_rocm/fused_attn_aotriton.cpp`). Qwen3-8B is 32 heads / 8 KV groups,
   so TE fell through to `UnfusedDotProductAttention`, whose `torch.baddbmm` tries
   to materialise an 8 GiB attention matrix at seq 8192 and OOMs. With CK built in,
   the same shape runs fused at 0.57 GiB peak. For any GQA model on ROCm, CK is not
   optional.
2. **The CK-JIT build needs `CXX` pinned.** `ck_jit_build.py` captures
   `os.environ.get("CXX") or "c++"` and hands it to a compiler interceptor; when
   that resolution lands on `hipcc`, aiter's ABI probe runs `hipcc -v`, which exits
   1 with no input files, and the build dies as `[AITER-BUILD] CK-JIT build failed`
   with no mention of a compiler. Building with `CXX=/usr/bin/c++` fixes it.

Also required, and unrelated to TE itself: `ROCM_PATH=/opt/rocm-7.2.4` (this host's
`/opt/rocm` is an empty directory, so cmake finds neither hipblaslt nor
`.info/version`), cmake ≥ 3.25 (hipblaslt's config uses `block()`), and
`NVTE_ROCM_ARCH=gfx950` to avoid also building gfx942 — that variable lives in
`build_tools/rocm_utils.cmake`, not in `build_tools/utils.py`.

One TE-specific wart affects tooling, not results: importing TE succeeds but tearing
it down raises in torch 2.13's `torch.library` cleanup and then aborts on heap
corruption while unloading native libs, after everything has been flushed. Preflight
checks in `train_qwen3_8b_te.sh` therefore look for a sentinel on stdout instead of
trusting the exit status.

### 4.7 Aligning the recipe: what "same configuration" can and cannot mean

§4.6 compared each stack at its own defaults, which left three confounders bundled
together. This section removes the two that are actually alignable and measures each
separately. The config diff was written down before the runs started; the finding
that changed the design is that **the two stacks' Hadamard transforms are not the
same transform**, so "both have Hadamard on" is not the same recipe.

Where each stack applies its fixed 16-point Hadamard:

| operand | Lumen | TE (`use_hadamard=True`) |
|---|---|---|
| fprop activation | no | **yes** |
| fprop weight | no | **yes** |
| dgrad `grad_output` | no | **yes** |
| dgrad weight | no (reuses forward's cached FP4) | **yes** |
| wgrad `grad_output^T` | **yes** (SR on) | **yes** |
| wgrad `activation^T` | **yes** (RTN) | **yes** |

Lumen rotates only the two WGrad operands, via `hadamard_quant_mxfp4` in
`lumen/ops/quantize/linear.py` (and silently falls back to plain quantization when
`M % 16 != 0`); the fprop and DGrad GEMMs consume unrotated FP4. TE's rotation is
fused into the MXFP4 cast kernel and therefore applies to every tensor any recipe
quantizer touches, rowwise and columnwise
(`common/cast/mxfp4/cast_transpose_mxfp4_shuffled.cuh`). So enabling TE's Hadamard
does not reproduce Lumen's recipe — it applies a strictly wider rotation. Lumen also
has no switch to turn its Hadamard off; that would take a code change.

That leaves the BF16 tail as the one knob that aligns exactly, and it does align
exactly: `get_fp4_context()` in `megatron/core/fp4_utils.py` returns `nullcontext()`
for tail layers at both init and forward, which is the same "these layers never see
FP4" semantics as Lumen's skip-prefix set in `lumen/quantize/__init__.py`. One trap:
`--num-layers-at-start-in-bf16` defaults to **1**, not 0, so a tail-only request has
to pass `0` explicitly or layer 0 silently leaves FP4 as well. Both launchers now do.

The resulting ladder, all four arms sharing seed 1234, corpus, sample order and
held-out set:

| arm | stack | BF16 tail | Hadamard |
|---|---|---|---|
| A | Lumen | last 5 | wgrad operands |
| B | TE | none | off |
| C | TE | last 5 | off |
| D | TE | last 5 | on (all operands) |

- **B → C** isolates the BF16 tail inside one stack.
- **C → D** isolates TE's Hadamard inside one stack.
- **A vs D** and **A vs C** are the closest achievable Lumen/TE comparisons; neither
  is an exact recipe match, and they bracket Lumen's rotation scope from above and
  below.

Already matched before this section, and worth stating because it is easy to assume
otherwise: the output layer is BF16 in **both** stacks. Lumen skips it explicitly
(`quantize_output_layer=False`), and in Megatron `GPTModel.output_layer` is built and
called outside any quantization context, so the TE arm never quantizes it either.

Two confounders survive this section and cannot be removed without more work: the
fused norm/rope/attention gap (a throughput effect, mathematically neutral for loss)
and the Megatron branch difference. TE's rounding policy for MXFP4 was not audited
against Lumen's "SR on gradients, RTN on activations" split, so that remains an
unquantified recipe difference too.

#### 4.7.1 Result: the convergence gap was the recipe, not the implementation

| arm | median step | tokens/s | train loss @1k | held-out val @1k | peak mem |
|---|---|---|---|---|---|
| A Lumen, tail 5, H on wgrad | 2007.2 ms | 130,602 | 5.7649 | **5.7793** | 0.6215 |
| B TE, no tail, no H | 1425.3 ms | 183,922 | 5.8300 | 5.8479 | 0.6355 |
| C TE, tail 5, no H | 1529.2 ms | 171,426 | 5.8262 | 5.8423 | 0.6035 |
| D TE, tail 5, H on all | 1555.3 ms | 168,549 | 5.7692 | **5.7833** | 0.6035 |

All four arms: zero NaN, zero skipped iterations, p75 − p25 under 7 ms.

Decomposing §4.6's headline gap, using paired per-iteration deltas over the second
half of training (identical batches, so these add up exactly):

| contrast | Δ mean (nats) | sd | share of the 0.0665 gap |
|---|---|---|---|
| C − B: BF16 tail | −0.0058 | 0.0017 | 9% |
| D − C: TE's Hadamard | −0.0565 | 0.0040 | 85% |
| A − D: everything else | −0.0043 | 0.0046 | 6% |
| **A − B: total (§4.6)** | **−0.0665** | 0.0058 | 100% |

**Roughly 85% of Lumen's convergence advantage was the Hadamard transform, and
almost none of it was the implementation.** Once TE runs the same two recipe
choices, its held-out loss lands at 5.7833 against Lumen's 5.7793 — a residual of
0.004 nats, or 0.07%. That residual is consistently negative across all four
windows, so it is probably real rather than noise, but it is also the bucket that
still contains the Hadamard-scope difference, the unaudited rounding policy, the
fused-kernel gap and the Megatron branch difference. Whatever Lumen's MXFP4 GEMM
path does differently from TE's, it is worth at most ~0.004 nats here.

Two results worth noting beyond the headline:

- **Wider Hadamard is not better.** TE rotates every operand including the fprop
  activation and weight; Lumen rotates only the two WGrad operands. The wider
  rotation did not win — D is 0.004 nats *behind* A. Rotating the forward pass buys
  nothing measurable at this scale, which is consistent with FP4 dynamic-range
  overflow being a gradient-side problem (§4.3, and the H16 + tail fix recorded in
  the bug notes).
- **The two recipe knobs have very different price tags.** In TE, Hadamard costs
  1.7% step time (1529 → 1555 ms) because it is fused into the cast kernel, and it
  buys 0.057 nats. The BF16 tail costs 7.3% (1425 → 1529 ms) and buys only 0.006
  nats — BF16 layers are simply slower than FP4 ones. The tail earns its place as
  divergence insurance at longer horizons (§4.3), not as a loss optimization at
  1000 steps. It does, however, *reduce* peak memory (0.6355 → 0.6035), presumably
  by removing those layers' FP4 cast and cache buffers.

On speed the §4.6 conclusion survives recipe alignment, smaller: at matched recipe
TE is **1.29×** faster per step (1555 vs 2007 ms), down from 1.41× at defaults.
That remainder is the fused norm/rope/attention gap plus whatever the quantized
linears cost differently, and this comparison does not separate those two.

Runs: [`i79coc3c`](https://wandb.ai/daixindi-amd/qwen3-8b-mxfp4/runs/i79coc3c) (C),
[`nhi8mghf`](https://wandb.ai/daixindi-amd/qwen3-8b-mxfp4/runs/nhi8mghf) (D).
Logs: `te_qwen3_8b_c4_1k_tail5{,_had}_mxfp4.log`. Reproduce with
`TAIL_BF16=5 HADAMARD={0,1} bash examples/qwen3/scripts/train_qwen3_8b_te.sh`.

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

### 5.5 Where the 1.29× against TE actually goes

§4.7 left one question open: at matched recipe TE still runs a step in 1555 ms
against Lumen's 2007 ms. Both arms were re-run for 25 steps with
`--use-pytorch-profiler` on rank 0 (2 profiled iterations, same 8192×2 shapes; the
profiled iterations clocked 2044 and 1543 ms, so the profiler is not distorting
the comparison). Aggregating device kernel time by category, per step:

| category | Lumen | TE | Δ |
|---|---|---|---|
| FP4 + BF16 GEMM | 668.5 | 536.7 | **+131.8** |
| elementwise (rope, cat, misc) | 321.8 | 179.8 | **+142.1** |
| Hadamard (separate kernel) | 124.2 | 0.0 | **+124.2** |
| RMSNorm | 101.7 | 48.1 | **+53.6** |
| grad accumulation add | 422.4 | 384.6 | +37.8 |
| quant / cast | 112.8 | 94.9 | +17.9 |
| comm (NCCL) | 131.5 | 122.8 | +8.7 |
| other + optimizer | 42.9 | 25.8 | +17.1 |
| attention | 401.2 | 421.0 | **−19.8** |
| fused rope | 0.0 | 30.6 | −30.6 |
| **total kernel time** | **2327.2** | **1844.3** | **+482.9** |
| kernel launches | 11,945 | 6,072 | +5,873 |

The 483 ms of kernel time matches the 452 ms of wall clock, so this is GPU-bound
work, not launch overhead — despite Lumen issuing roughly twice as many kernels.
Four causes, largest first.

**1. Quantization fusion (+142 ms combined).** Lumen runs three kernels where TE
runs one. Lumen: `_fused_hadamard_quant_mxfp4` (496 launches, 124.2 ms),
`_convert_to_mxfp4` (744, 76.4 ms), `_dequant_transpose_mxfp4` (248, 28.1 ms) —
1,488 launches, 228.7 ms. TE: four template instantiations of
`te_mxfp4::cast_transpose_mxfp4_shuffled` (620 launches, 91.1 ms) which do cast,
transpose, Hadamard and scale-shuffle in a single pass over the tensor. Same
recipe, 2.5× the cost. The `dequant_transpose` kernel is pure overhead of Lumen's
design: it reconstructs BF16 from the saved FP4 activation to build the WGrad
operand, work TE never does because its columnwise cast already emitted it.

**2. FP4 GEMM kernel selection (+132 ms).** Both stacks can reach AITER's assembly
kernel `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256`. TE uses it for 620 of its 744
FP4 GEMMs. Lumen uses it for 186 of 682 and sends the rest to Triton kernels at
several times the cost:

| Lumen backend | shapes | calls/step | ms/step | ms/call |
|---|---|---|---|---|
| `asm` (AITER assembly, = TE's kernel) | 3 | 186 | 32.4 | 0.174 |
| `plain` (Triton `gemm_afp4wfp4`) | 7 | 434 | 280.1 | 0.645 |
| `shuffled` (Triton preshuffle) | 1 | 62 | 68.6 | 1.107 |

Lumen's FP4 GEMMs total 381.1 ms against TE's 273.5 ms for the same layers. The
cause is `_mxfp4_asm_supported()`, which gates the assembly path on
`get_GEMM_config(M, N, K) is not None` — AITER's tuned-shape table. Only **3 of
Qwen3-8B's 11 MXFP4 GEMM shapes are in that table**; the other 8 return "not found
tuned config, will use default config". The gate is not paranoia: on an untuned
shape the default config silently returned garbage once (0.6 dB at (64,64,128),
§2.2). So the fix is to extend the table rather than drop the gate, and
`scripts/mxfp4_tune_shapes.py` already exists to do exactly that — collect a
model's shapes, tune them, and verify bit-exactness. The stale
`qwen3-8b.shapes.csv` at the repo root is from an older sequence length and does
not cover the current M=16384 shapes. This cause is closed in §5.6.

**3. Unfused RoPE and the QKV concatenation (+111 ms net).** Lumen's elementwise
total is 321.8 ms against TE's 179.8. The excess is `CatArrayBatchedCopy` (360
launches, 59.9 ms, against TE's 72 launches and 13.6 ms) plus
`elementwise_kernel_manual_unroll` (576 launches, 40.2 ms) — the rotate-half and
concatenate sequence of an unfused rotary embedding. TE spends 30.6 ms total in
`fused_rope_forward_kernel` and `fused_rope_backward_kernel` and skips the rest.
This is the apex/TE fusion gap, and it is the cheapest of the four to close.

**4. RMSNorm (+54 ms).** Torch's `vectorized_layer_norm_kernel` (42.4 ms),
`cuComputeGradInput` (35.8 ms) and a Triton `rmsnorm_bwd` (11.8 ms) total 101.7 ms
over 727 launches, against TE's tuned fwd/bwd/bwd_finalize trio at 48.1 ms.

Three things that are *not* causes, and one shared cost:

- **Attention is not a differentiator** — Lumen is 19.8 ms *faster*. Both stacks
  land on the same AITER kernels (`fmha_bwd_hd128_bf16_causal[_br]_a32`,
  `fmha_fwd_hd128_bf16_causal`); TE additionally runs `ck_fused_attn::dk_dv_reduce`
  (9.2 ms). §4.6 listed TE's CK fused attention as part of its advantage; the trace
  does not support that. CK mattered for making TE *runnable* on a GQA model at all
  (§4.6.2), not for its speed.
- **The BF16 GEMMs are a clean control** — the three Tensile `Cijk_*` kernels cost
  249.5 ms in Lumen and 249.4 ms in TE. The BF16 tail layers do identical work in
  both stacks, which is a good check that the two traces are comparable.
- **Communication is equal** — the same `ncclDevKernel_Generic_1`, 131.5 vs
  122.8 ms.
- **Gradient accumulation is the single largest kernel in both traces and is
  shared.** `CUDAFunctor_add<float>` runs 586 times per step in *both* arms, 377.2 ms
  in Lumen and 364.9 ms in TE — 16–21% of the step, spent on a plain fp32 add. Both
  launchers pass `--no-gradient-accumulation-fusion` from the shared config. This is
  the largest single optimization available and it is not a Lumen-vs-TE issue at
  all; it would pay off in both arms.

Ranked by expected return: extending AITER's tuned table (+132 ms, tooling already
written), fusing the quantization kernels (+142 ms, the largest but the deepest
change), enabling rope fusion (+111 ms, cheapest), and a fused RMSNorm (+54 ms).
Together they account for 429 of the 483 ms gap. Separately, gradient-accumulation
fusion is worth ~370 ms to whichever stack enables it.

### 5.6 Closing cause 2: extending AITER's tuned table

Cause 2 from §5.5 is now fixed. `scripts/mxfp4_tune_shapes.py` was run over the 11
shapes the 1000-step job actually issued, taken from its autotune log rather than
re-derived on paper — the paper derivation gets Megatron's *fused* qkv (N=6144) and
gate_up (N=24576) wrong, and those are two of the eight missing shapes. AITER's
tuner found a prebuilt assembly kernel for all 8 in 25 s, and all 8 are bit-exact
against the plain Triton kernel, so none were dropped. The rows live in
`examples/qwen3/configs/qwen3_8b_a4w4_blockscale_tuned_gemm.csv` and
`train_qwen3_8b.sh` now points `AITER_CONFIG_GEMM_A4W4` at them. They key on the
exact M/N/K, so they only fire at MBS × SEQ_LEN = 16384 with TP=1.

With the table extended, autotune measures the assembly path against Triton on
every shape and it wins on 9 of 11, by 15–44%:

| M, N, K | winner | asm | plain | shuffled | asm gain |
|---|---|---|---|---|---|
| 4096, 4096, 16384 | plain | 0.221 | 0.234 | 0.278 | +6% |
| 4096, 12288, 16384 | asm | 0.458 | 0.647 | 0.703 | +29% |
| 6144, 4096, 16384 | asm | 0.293 | 0.395 | 0.380 | +26% |
| 16384, 4096, 4096 | plain | 0.270 | 0.286 | 0.328 | +6% |
| 16384, 4096, 6144 | asm | 0.316 | 0.374 | 0.415 | +16% |
| 16384, 4096, 12288 | asm | 0.453 | 0.679 | 0.683 | +33% |
| 16384, 4096, 24576 | asm | 0.735 | 1.304 | 1.171 | +44% |
| 16384, 6144, 4096 | asm | 0.337 | 0.396 | 0.489 | +15% |
| 16384, 12288, 4096 | asm | 0.518 | 0.697 | 0.827 | +26% |
| 16384, 24576, 4096 | asm | 0.924 | 1.363 | 1.560 | +32% |
| 24576, 4096, 16384 | asm | 0.775 | 1.306 | 1.214 | +41% |

(ms/call, best of the per-shape measurements.) The two shapes that still pick
Triton are within 6%, close enough that the winner flips between runs; there is
nothing left on the table there.

Re-running the same 25-step profiling job, everything else identical:

| | median step | vs TE | ratio |
|---|---|---|---|
| Lumen, stock table | 2041.4 ms | +485.3 | 1.31× |
| Lumen, extended table | **1943.2 ms** | **+387.1** | **1.25×** |
| TE (arm D) | 1556.1 ms | — | — |

A 25-step job reads about 40 ms high against steady state. At 200 steps on the
training corpus the extended table settles at **1900.0 ms**, against 2007.2 ms for
the 1000-step run on the stock table — the same ~100 ms, and the number to quote
against the 2803.8 ms BF16 baseline (§5.8).

98 ms of the predicted 132 ms, and in the trace the GEMM category falls from
668.5 to 550.2 ms — against TE's 536.7 ms, so GEMM is now within +13.6 ms and is
no longer a differentiator. The prediction overshot because two shapes stayed on
Triton and because Lumen's autotune probes cost a little more once there is a
third candidate to measure.

TE does not benefit from this. Its log carries the same "not found tuned config"
warnings, but its trace shows 744 of 744 FP4 GEMMs already on the assembly kernels
— TE reaches them through an AITER entry point that does not consult the tuned
table. So the comparison in §4.7 and §5.5 is unaffected: the 98 ms is a real
narrowing of the gap, not a baseline that moved.

What remains of the 387 ms is the fusion work: elementwise +163.7, Hadamard
+127.6, RMSNorm +60.1, comm +44.2, quant/cast +22.1, offset by TE's fused rope
(−30.6) and Lumen's faster attention (−24.3).

### 5.7 Closing cause 3: the rope flag the launcher was not passing

Cause 3 turned out to be a missing flag rather than missing work. The trace's
excess elementwise time is a rotate-half: `aten::cat` 507×, `aten::neg` 288×,
`aten::mul` 578× per step. Megatron does have a fused path, and it does not need
apex — `rope_utils` imports `fused_apply_rotary_pos_emb` from TE, which is
installed. But enabling it bought only 4 ms, because the Lumen backend never reads
Megatron's switch:

```12:12:lumen/models/megatron.py
    args.apply_rope_fusion = getattr(args, "lumen_fused_rope", False)
```

`model_provider` overwrites `apply_rope_fusion` from Lumen's own
`--lumen-fused-rope`, which routes to AITER's kernel instead of TE's or apex's.
The launcher never passed it, so rope ran unfused no matter what Megatron was
told. Passing it is worth **137.7 ms/step** (1943.2 → 1805.5 on the 25-step job),
almost exactly the 134 ms §5.5 predicted, and the flag is now in
`train_qwen3_8b.sh`. Over 200 steps the loss curves sit on top of each other
(step 100: 6.9271 unfused vs 6.9267 fused; step 200: 6.8016 vs 6.8025), so this
is a free kernel swap, not a numerics trade.

### 5.8 Where MXFP4 stands against BF16 and TE

All six runs below are the same shape (Qwen3-8B, 36 layers, seq 8192 × mbs 2,
gbs 32, TP=1, 8×MI350X), median of the run's steps after the first 10:

| | step | vs BF16 | TE's lead |
|---|---|---|---|
| BF16 baseline | 2803.8 ms | — | 1.80× |
| Lumen MXFP4, stock AITER table | 2007.2 ms | 28.4% faster | 1.29× |
| Lumen MXFP4, + extended table (§5.6) | 1900.0 ms | 32.2% faster | 1.22× |
| Lumen MXFP4, + fused rope (§5.7) | 1804.8 ms | 35.6% faster | 1.16× |
| **Lumen MXFP4, + dual-layout quant (§5.11)** | **1791.6 ms** | **36.1% faster** | **1.15×** |
| TE MXFP4, matched recipe (arm D) | 1555.3 ms | 44.5% faster | 1.00× |
| TE MXFP4, stock recipe (arm B) | 1425.3 ms | 49.2% faster | 0.92× |

Two cheap fixes moved MXFP4 from 28.4% to 35.6% faster than BF16 and cut TE's lead
from 1.29× to 1.16×. Note the BF16 baseline runs the *same* unfused RMSNorm, so
these two fixes improve the MXFP4/BF16 ratio rather than cancelling out of it.

The remaining 249.5 ms against TE is almost exactly the fusion work still on the
list: Hadamard as a separate kernel (+127.6), RMSNorm (+60.1), comm (+44.2) and
quant/cast (+22.1) sum to 254 ms. Closing the quantization fusion and RMSNorm
would put the two stacks at parity; nothing found so far suggests a structural
reason Lumen has to stay behind.

### 5.9 Two attempts on that remainder that did not work

Both were tried end to end at 200 steps against the §5.8 baseline and both were
reverted. They are recorded because the reasoning behind each looks sound enough
to be attempted again.

**Swapping the QK norm onto Lumen's kernel: 49.1 ms slower.** Qwen3 runs a norm on
q and k as well as the two block norms, and `_patch_norms_in_spec` only walked the
layer's own norms, so the QK pair stayed on Megatron's `WrappedTorchNorm` while
everything else ran Lumen's. Teaching the walk to descend into the attention spec
is a two-line change and it does reach them — and the step got *slower*, 1804.8 →
1853.9 ms.

The trace explains it. Per step both stacks run 288 norms. TE does all 288 in
47.3 ms, a flat 0.164 ms each. Lumen's Triton kernel does the block norms
(N=4096) in 0.160 ms each, matching TE exactly; the QK norms (N=128) cost 0.585
ms each on torch, and *more* than that on Lumen's kernel. AITER's `rms_norm` uses
a persistent grid with `BLOCK_SIZE = next_pow2(N)`, so at N=128 each iteration
moves 256 bytes and most of the wavefront idles.

So the whole 60.1 ms norm gap is QK norm at narrow N, and it is not reachable by
re-pointing `LNImpl`: Lumen already owns the norms it is good at, and the one it
is missing needs a kernel that tiles several rows per program. Until that kernel
exists, torch is the better fallback for N=128 and the walk should stay as it is.

**Loading the Hadamard tile transposed: no change at all.** The wgrad path calls
`hadamard_quant_mxfp4(grad_flat.t(), ...)` on a view, so the kernel's innermost
index carries the long stride. That looked like the reason the kernel spends half
its 127.6 ms in 62 calls of ~1.0 ms — one per layer per micro-batch, matching
fc1's (24576, 16384) gradient — which works out to ~0.8 TB/s against ~2 TB/s on
the dense operands.

Loading the tile as (BLOCK_N, BLOCK_M) and applying `tl.trans` moved neither the
kernel (127.6 → 124.2 ms, inside run-to-run noise) nor the step (−0.1 ms), and the
duration histogram kept the same 62 slow calls. The flag was confirmed to be set
on the real shapes, so the new path did run.

### 5.10 What actually limits the Hadamard+quant kernel

`/tmp` benchmarks had been reporting 26 ms for a call the trace timed at 1.0 ms,
which is what sent §5.9 after the wrong cause. Timing with CUDA events over
tensors that stay allocated for the whole measurement reproduces the trace to
within 0.2%: 127.8 ms/step against the trace's 127.6, per-call 1.029 ms on fc1's
gradient. Every measurement below comes from that harness, which turns a
10-minute training run into a 6-second one.

The view penalty is real and shape-independent — the *same* (4096, 16384) operand
runs at 1219 GB/s as a view and 2268 GB/s dense. But it is not the load, and it is
not fixable from the wrapper:

| | dense | transposed view |
|---|---|---|
| load alone (kernel stripped to a tile load + store) | 5129 GB/s | 3570 GB/s |
| same load written as `tl.trans` of the swapped tile | — | 3512 GB/s |
| full kernel | 1955 GB/s | 1150 GB/s |

Three things follow. The kernel is nowhere near load-bound: dense loads sustain
5129 GB/s while the full kernel manages 1955. `tl.trans` is indistinguishable from
the plain strided load, so §5.9's rewrite never could have helped — Triton folds
the transpose into layout inference and lands on the same access pattern. And the
view costs 1.70× in the full kernel against 1.44× in the isolated load, so the
layout the load forces on `x` is also making the Hadamard butterfly and `_pack_fp4`
do more cross-lane work. Sweeping BLOCK_M from 32 to 512 does not move the view
off ~1150 GB/s in either direction.

That points at the fusion in §8.8 rather than at any repair of the present kernel.
`grad_flat` is already read densely once per linear by DGrad's `convert_to_mxfp4`
(55.7 ms/step) and then again as a view by the WGrad Hadamard (94.2 ms/step). One
kernel reading it dense once and writing both layouts, transposing through LDS,
drops per-element traffic from 5.06 to ~3.06 bytes and removes the view penalty
outright. §5.11 built it.

### 5.11 Dual-layout gradient quantization

`dual_layout_quant_mxfp4` reads a tile of `grad_flat` once and emits both MXFP4
layouts the backward needs: row-major along n for DGrad, and Hadamard-rotated,
transposed, blocked along m for WGrad. Both outputs' scale blocks fall entirely
inside the tile, so it needs no cross-tile reduction — only that BLOCK_M and
BLOCK_N are whole numbers of quant blocks and BLOCK_M a whole number of Hadamard
groups. With SR off it is bit-exact against both kernels it replaces, on all five
shapes tested; with SR on it draws from its own philox stream, deliberately
separate from the row-major one so the two GEMMs do not share rounding noise.

It is a smaller win than the traffic reduction suggests:

| | per element | per layer per micro-batch | achieved |
|---|---|---|---|
| `convert_to_mxfp4` + `hadamard_quant_mxfp4(x.t())` | 5.06 B | 2.356 ms | 1150–2268 GB/s |
| `dual_layout_quant_mxfp4` at (128, 64) | 3.06 B | 2.030 ms | 850–990 GB/s |

40% less traffic, but the fused kernel sustains roughly half the bandwidth —
holding both quantization paths and the transposed tile at once costs more than
the second read did. That leaves −20.2 ms/step predicted, and 1804.8 → **1791.6
ms/step** measured, with loss tracking the baseline (6.8025 → 6.8012 at step 200).
BLOCK_M was swept from 32 to 256: it sets the transposed output's contiguous write
run, and (128, 64) beat (64, 64) by 5%.

The remaining headroom is in that bandwidth number, not in further fusion — a
dense read alone sustains 5129 GB/s (§5.10), so the kernel is compute- and
register-bound, and the next thing to look at is its register pressure.

**Superseded (2026-08-26): "register-bound" is wrong.** §5.19 compiles the kernel
and finds **zero spills on all five production shapes** (VGPR 84–118, 4–5
waves/SIMD), so there is no register pressure to relieve; §5.17 closes tile and
warp tuning and §5.19 closes occupancy. The limiter is raw instruction count:
66–78% plain VALU, a third of it Philox. Read those sections before spending time
here.

### 5.12 WGrad's activation operand, emitted by the forward quantizer

At parity (Lumen 1526.5 ms, TE 1526.5 ms median) the two stacks were profiled
kernel-by-kernel over three steps of Qwen3-8B. Lumen was already ahead on the
GEMMs (790.7 ms vs 816.1), the norms (103.0 vs 135.0) and attention (970.1 vs
1004.3, TE paying an extra `dk_dv_reduce`); the whole remaining deficit sat in
quantization, 286.6 ms against 205.6.

The structural difference: TE's forward `cast_transpose` emits both operands at
once, where Lumen's forward emitted only the row-major one and had backward
rebuild WGrad's from the stored FP4 (`dequant_hadamard_quant_mxfp4`) — decode,
rotate, requantize, a second full pass over the activation. Measured per layer
per micro-batch at Qwen3-8B's shapes:

| | (16384, 4096) ×3 | (16384, 12288) ×1 |
|---|---|---|
| forward `convert_to_mxfp4` | 68.3 us | 156.3 us |
| backward `dequant_hadamard_quant` | 60.9 us | 174.4 us |
| **current total** | **129.2 us** | **330.7 us** |
| forward `dual_layout_quant` (both) | 79.4 us | 197.9 us |
| | 1.63× | 1.67× |

22.3 ms → 13.5 ms per micro-batch across 31 quantized layers, and 124 fewer
kernel launches. The operand is also the WGrad GEMM's B, so the quantizer stores
it pre-shuffled (`shuffle_col`, the same treatment §5.11's weights get) once the
autotuner has picked the shape's backend.

The activation is now quantized once instead of twice — WGrad no longer reads a
value that has been through FP4 and back — so dW cannot match the old path
bit-for-bit. Over 200 steps the loss difference against the previous code
(mean 0.047, last-50 mean 0.016) is the same size as the difference between two
runs of the new code against each other (0.047 / 0.015): run-to-run noise, no
bias.

Two side effects worth noting. The row-major activation stays saved for the BF16
fallback, so activation memory grows by one FP4 copy: 144.0 → 150.4 GiB peak of
the 251.7 GiB card. And the RTN quantizers stopped drawing philox counters they
never read — an all-RTN call that draws from Python's RNG also shifts the stream
every SR caller after it reads, which made the forward and backward operands
impossible to compare.

Paired against TE at 200 steps each (median of iterations 21–200):

| | median | mean |
|---|---|---|
| TE (4 runs) | 1526.2–1526.7 ms | 1531.4–1538.3 ms |
| Lumen before | 1526.3, 1526.8 ms | 1530.7, 1531.3 ms |
| **Lumen after** | **1513.7, 1515.7 ms** | **1516.1, 1525.9 ms** |

Lumen finishes 11–13 ms/step (0.7–0.8%) ahead of TransformerEngine. Superseded on
8/10 by §5.13 — every Lumen number above was measured on the patched-Megatron
linears, which the launcher no longer uses.

### 5.13 Re-measured on the native parallel linears (8/10)

`--lumen-linear` became the launcher default once the recipe-routing fix made it
safe (it had been configuring the native modules from the `--linear-fp8-scaling`
string, so an MXFP4 run silently executed FP8 blockwise). That moves the whole
ladder's reference point, so the numbers that anyone still cites were re-measured
against it. Same protocol throughout: 200 steps, GBS 32, seq 8192, 36 layers, c4,
median over iterations 2–200.

| Arm | median | vs new default |
|---|---|---|
| **Lumen, native linears (new default)** | **1429.0, 1430.8 ms** | — |
| Lumen, patched Megatron linears (the old default) | 1513.4, 1514.0 ms | +84 |
| TE, re-run today | 1528.5 ms | +99 |

TE reproduces its recorded 1526.2–1526.7 to within 2 ms, which is what licenses
comparing today's Lumen numbers to the rest of this section. **Lumen now finishes
~99 ms/step (6.5%) ahead of TransformerEngine**, up from 0.8%, without any new
kernel work — the gap was sitting behind a CLI flag.

The four ladder rungs that are still a runtime toggle were re-priced on the new
default. The rest are code commits from 8/3–8/5 that all predate the routing fix,
so `--lumen-linear` on them runs FP8 blockwise and their deltas cannot be
re-measured without backporting the fix to each one; those rows stand as measured
on the old path.

| Rung | Toggle | Old path | New default |
|---|---|---|---|
| 1 | `--lumen-fused-rope` | −95.2 | **−88.4** |
| 5 | Cross-micro-batch weight cache | −25.5 | not wired to native linears (§5.15) |
| 8 | Cached scale swizzle | −15.6 | −1.5 (noise) |
| 13 | `gc.freeze()` after warmup | −5.1 | −1.1 (noise) |

Read the last two cache/GC rows as *unresolvable*, not as zero. Two identical
runs of the new default differ by 1.8 ms in the median — three times the 0.6 ms
spread the old path showed — and both deltas sit inside that. Separating them
needs a profiler or many seeds, not another A/B. `gc.freeze()` is additionally
the wrong thing to judge by median, since its documented effect is tail latency;
its mean is 14 ms above the new default's, against an 8 ms mean spread between
identical runs, so even that is only suggestive. The weight-cache result had a
different cause: the switch only reached `quantize._replace_forward`, while the
new default uses native Lumen linears. §5.15 wires the existing cache to that
path and re-measures it.

### 5.14 What the BF16 tail costs, per layer

§4.7 priced the tail as a single 7.3% block on TE. On Lumen the same knob was swept
one setting at a time to get its marginal cost, since that is what decides whether
a partial tail is worth anything. Six arms, `TAIL_BF16` = 5 through 0, which on 36
layers leaves 31 through 36 layers in MXFP4. Protocol: Megatron
backend, 300 steps, GBS 128, seq 8192, c4 with a disjoint held-out valid file,
lr 1e-4 with warmup 50 and cosine decay, median over iterations 21–300. Each arm's
precision assignment was read back from `enable_fp8_for_parallel_linear`'s count
(124/20 through 144/0 quantized-vs-BF16 modules in increments of 4, against 144 =
36 layers × 4 linears), not assumed from the flag.

| Arm | MXFP4 layers | median | IQR | vs default | speedup |
|---|---|---|---|---|---|
| `TAIL_BF16=5` (default) | 31 | 6168.6 ms | 24.2 | — | — |
| `TAIL_BF16=4` | 32 | 6087.2 ms | 20.9 | −81.4 | 1.32% |
| `TAIL_BF16=3` | 33 | 6012.1 ms | 18.1 | −156.6 | 2.54% |
| `TAIL_BF16=2` | 34 | 5924.8 ms | 19.0 | −243.9 | 3.95% |
| `TAIL_BF16=1` | 35 | 5851.4 ms | 18.5 | −317.2 | 5.14% |
| `TAIL_BF16=0` | 36 | 5774.7 ms | 15.5 | **−393.9** | **6.39%** |

The marginal cost is flat: 81.4, 75.1, 87.3, 73.4 and 76.7 ms for the five
successive layers, 78.8 ms/layer overall. This is the one rung in this section that a
median can resolve cleanly — the interquartile range is 15–24 ms against a
smallest arm-to-arm gap of 81 ms. It is also why the earlier 30-step attempt at
the same question failed: there the run-to-run spread of the median was 54 ms,
the same size as one layer's effect, so only the 5-layer endpoint was
separable. The endpoint agrees with that earlier measurement (6.39% here, 6.58%
on the mock corpus at a 6099.7 ms baseline).

**On the loss side these runs settle nothing, and show why.** Held-out loss at
step 300 was 5.7172 / 5.5935 / 5.7043 / 5.7434 / 5.7824 / 5.7537 for tails
5 / 4 / 3 / 2 / 1 / 0.
Tail 4 quantizes *one* more layer than the default and comes out 0.124 nats
*better*, while tail 0 quantizes five more and is only 0.037 nats worse. The
ordering remains non-monotonic after filling both missing arms, so the run-level
spread is at least ~0.12 nats and the smaller differences sit well inside it. A likely
mechanism is that the arms are not actually paired: WGrad applies stochastic
rounding to the gradient (§1.3), so changing which layers are quantized changes
the random stream, not just the precision. §4.7's independent TE measurement —
the tail buying 0.006 nats at 1000 steps — is 20× below this noise floor, so a
short run was never going to see the effect either way. Separating it needs
several seeds per arm.

What the arms do agree on is the absence of any instability signal at this
horizon: zero NaN and zero skipped iterations in all six, gradient-norm medians
within 0.88–0.96, p95 near 3.0, and no iteration above 10 anywhere. That is not
evidence that the tail is removable. The failure it exists to prevent is a
late-horizon divergence, and per §6.3 every run that diverged used H32; H16
*without* a tail is a combination that has never been run to the horizon where
rows 2–4 broke. 314M tokens at lr 1e-4 without a spike lowers the prior on a
problem; it does not test the tail risk.

### 5.15 Native linears were re-quantizing every weight per micro-batch

A fresh two-iteration PyTorch trace of the current default reproduced §5.5 to
within 1% kernel by kernel. It also made one supposedly closed rung impossible:
`_convert_to_mxfp4_kernel` and `_transpose_packed_fp4_kernel` each ran **992
times/step**, exactly 124 quantized linears × 8 gradient-accumulation
micro-batches. Their CPU correlation points to `QuantizedLinearFunction` forward
with its BF16 weight input, not an activation layout operation.

The cache in `_mxfp4_cached_weight` was correct but only called from
`quantize._replace_forward`, the patched-Megatron path. Since §5.13 made native
`Lumen*ParallelLinear` the default, `_do_gemm` had bypassed it and re-derived
byte-identical RTN weights and DGrad transposes on every micro-batch. The native
path now caches the two weight operands on the Parameter and passes them through
`fp8_weight_cache` / `fp8_weight_scale`; the existing optimizer-step hook clears
both the patched-module and native-Parameter cache locations.

The second trace confirms the structural result:

| current default, per step | before | native cache | delta |
|---|---:|---:|---:|
| weight convert calls / time | 992 / 62.5 ms | 124 / 4.3 ms | −58.2 ms |
| packed transpose calls / time | 992 / 71.3 ms | 124 / 4.8 ms | −66.5 ms |
| quantize/layout total | 575.2 ms | 454.3 ms | −120.9 ms |
| main-compute-stream busy | 7338.9 ms | 7246.6 ms | −92.3 ms |

The profiler stretches a normal ~6.1 s iteration to ~7.3 s of compute-stream
busy time, so the last two rows are attribution, not a step-time claim. A
profiler-free 60-step A/B on the same code, median over iterations 21–60, gives
6093.3 ms with `LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` and **6019.9 ms** with the
cache: **−73.5 ms / 1.21%** (means 6104.2 / 6040.8). Both complete with zero NaN
or skipped iterations and loss@60 differs by 0.0018. The price is ~2.8 GiB/GPU
for both packed weight layouts and expanded scales.

After this fix the largest MXFP4-owned non-GEMM kernel is unambiguous:
`_dual_layout_quant_mxfp4_kernel`. The trace put it at 412.5 ms/step; §5.16 shows
that figure counts a validation pass, and the training-step cost is **322.7 ms at
`TAIL_BF16=5` (1984 calls) and 380.3 ms at `TAIL_BF16=0` (2304 calls)** — the
`quantize / layout` category around it being 359.1 and 423.8 ms respectively.
§5.11 already showed it is compute/register-bound rather than bandwidth-bound, so
register pressure and tile occupancy looked like the next MXFP4-specific target.

**They were not, and §5.17–§5.19 close all three axes — do not sweep them again.**
Tile and warp configuration: the launcher's BLOCK_M/BLOCK_N rule is already the
fastest of 27 configurations on every shape, and `num_warps` 8 and 16 are slower
than the default 4 everywhere. Occupancy: `waves_per_eu` unset, i.e. the
scheduler's own choice, beats every explicit floor on every shape. Register
pressure: the premise is simply false — the compiled kernel has **zero spills on
all five production shapes** at VGPR 84–118 and 4–5 waves/SIMD. Adding
`num_stages` and the MFMA-shape knobs on top, the whole launch-parameter space is
worth 1.01× in aggregate with two of five shapes *slower*.

The real limiter is the instruction count: the stream is 66–78% plain VALU, and a
third of it (796 of 2415 instructions on `grad gate_up`) is Philox generating
stochastic-rounding dither. That is what §5.20 attacks, via `SR_PHILOX_ROUNDS`.

Attention is larger but is BF16 work, and grad accumulation's available headroom
is ruled out in the 2026-08-19 findings.

### 5.16 The whole speedup against BF16, kernel by kernel

Everything before this section prices MXFP4 against TransformerEngine or against
its own previous commit. This one asks the other question — how far ahead of BF16
the stack actually is at the production shape, and what the remaining distance to
1.6× is made of.

Both arms ran back to back on the same machine, same launcher, same mock corpus,
differing only in the quantization recipe: Qwen3-8B, 36 layers, GBS 128, seq 8192,
MBS 2 (8 gradient-accumulation micro-batches per rank), TP=1, 8×MI350X, 60 steps,
median of iterations 21–60. Validation was pushed outside the run so no eval work
lands in the step time.

| | median | mean | spread (min–max) |
|---|---:|---:|---|
| BF16 | 8520.7 ms | 8539.5 ms | 8482.4 – 8635.5 |
| **MXFP4, `TAIL_BF16=5`** | **5984.6 ms** | 5999.1 ms | 5961.2 – 6090.3 |

**MXFP4 is 1.424× BF16** — 2536.1 ms/step faster, 29.8% less time. Both arms were
re-measured on `feature/mxfp4` on 2026-08-27 and reproduce at 8510.2 and 6009.5 ms
(1.416×); see §5.21.

#### Where those 2536 ms come from

Two-iteration PyTorch traces of each arm, categorised with
`scripts/summarize_torch_trace.py`. Every call count below matches its closed
form exactly, so the split between quantized and BF16 work is a measurement
rather than an attribution: BF16 issues 3480 Tensile GEMM calls
(36 layers × 4 linears × 3 GEMMs × 8 micro-batches + 24 for the vocab
projection), MXFP4 issues 2976 FP4 GEMM calls (31 × 4 × 3 × 8) plus 504 Tensile
calls (5 × 4 × 3 × 8 + 24).

| ms/iter | BF16 | MXFP4 | delta |
|---|---:|---:|---:|
| linear GEMM | 5084.5 | 2196.6 | **−2887.9** |
| — of which FP4 | — | 1107.2 | |
| — of which BF16 (tail + vocab) | — | 1089.4 | |
| quantize / layout | — | 359.1 | **+359.1** |
| attention | 1731.7 | 1734.5 | +2.8 |
| grad accumulate | 608.0 | 660.5 | +52.5 |
| optimizer | 320.4 | 341.0 | +20.6 |
| norm | 313.8 | 275.8 | −38.0 |
| collectives | 246.0 | 260.3 | +14.3 |
| activation | 244.9 | 237.2 | −7.7 |
| rope / embedding | 170.2 | 157.0 | −13.2 |
| everything else | 176.7 | 171.0 | −5.7 |
| **total kernel time** | **8896.2** | **6392.9** | **−2503.3** |

The kernel-time delta lands within 33 ms (1.3%) of the measured step delta, so
nothing material is hiding outside the kernels. The speedup is one line: the
linear GEMMs get 2888 ms cheaper and quantization hands 359 of it back.

Solving the two GEMM rows for the per-layer cost (36L + vocab = 5084.5,
5L + vocab = 1089.4) gives **128.9 ms per BF16 layer per step** and 445.0 ms for
the vocab projection, which is BF16 in both arms. A quantized layer costs
1107.2/31 = 35.7 ms of FP4 GEMM plus 359.1/31 = 11.6 ms of quantization:

| per layer per step | BF16 | MXFP4 |
|---|---:|---:|
| linear GEMM | 128.9 ms | 35.7 ms (**3.61×**) |
| quantization | — | 11.6 ms |
| **total** | **128.9 ms** | **47.3 ms** (**2.73×**) |

The FP4 GEMM is not the thing left to fix. At 4703 TFLOP of linear work per step
across 31 layers it sustains 4.25 PFLOPS against BF16's 1.18 PFLOPS on the same
GEMMs — a 3.6× ratio where the hardware's FP4:BF16 arithmetic ratio is 4:1, i.e.
the FP4 path is running at about the same fraction of its peak as the BF16 path
is of its own.

#### What 1.6× would take

1.6× means a 5325.4 ms step, 659.2 ms below where MXFP4 is. Only two line items
in the table are MXFP4's to spend:

- **The BF16 tail.** Quantizing the last 5 layers replaces 5 × 128.9 ms with
  5 × 47.3 ms: −408 ms of kernel time. This is a recipe change, not an
  optimization — §6.3 is why the tail exists.
- **Quantization itself.** 359.1 ms, of which `_dual_layout_quant_mxfp4_kernel`
  is 322.7. Removing all of it is the absolute bound, and §5.17 finds no way to
  remove any of it.

Together they are 767 ms of kernel time, roughly 720 ms of step — just past the
bar, and only if quantization becomes free. Everything else in the table is work
both arms do identically: attention (1734.5), gradient accumulation (660.5), the
optimizer (341.0), norms, collectives, activation and rope sum to 3665 ms/iter
that no quantization recipe touches, and the vocab projection adds 445 ms of BF16
GEMM on top. Cutting shared work does raise the ratio, but slowly — it would take
1853 ms off *both* arms to reach 1.6× that way.

So the honest ceiling at this shape, if every transformer linear and its
quantization were free, is 8520.7 / 3873.9 = **2.20×**, and the reachable number
today is the `TAIL_BF16=0` recipe at roughly **1.52×**. 1.6× is not available
from MXFP4-side work on Qwen3-8B at GBS 128 / seq 8192.

#### A profiling-recipe correction

The traces behind §5.5, §5.14 and §5.15 were taken with `TRAIN_STEPS=22` and no
`EVAL_INTERVAL` override, which makes Megatron evaluate every 2 steps — and one
of those validation passes falls inside the `--profile-step-start 18
--profile-step-end 20` window. Every forward-side count in those tables is
therefore exactly double what a training step issues: `fmha_fwd` 576 against 288,
FP4 GEMM 3968 against 2976, `_dual_layout_quant_mxfp4_kernel` 2976 against 1984.
It also explains the "profiler stretches the step ~20%" note in §5.15 — summed
kernel time was 132.7% of the step because it included work the step does not do.

Re-profiling with `EVAL_INTERVAL=100000` reproduces every count in closed form
and lands at 106.8% of the step. Backward-side rows were never affected, so the
rankings those sections drew still hold; the forward-side absolute values do not.
The corrected summaries are at `examples/qwen3/results/prof_clean_{mxfp4,bf16}/`.
**Any future profiling run has to disable eval in the window.**

### 5.17 Two things that did not make the dual-layout quantizer faster

§5.11 left the kernel at "compute- and register-bound, and the next thing to look
at is its register pressure". Both attempts at that are negative results.

**Relaunching it differently: no configuration beats the current rule.**
`benchmarks/bench_dual_layout_tiles.py` sweeps BLOCK_M ∈ {64, 128, 256},
BLOCK_N ∈ {32, 64, 128} and `num_warps` ∈ {4, 8, 16} — 27 configurations — on
each of the five shapes a step issues, under both call recipes. The launcher's
existing rule picks the fastest of the 27 on every shape, and `num_warps` 8 and
16 are slower than Triton's default 4 everywhere. Summed over one call of each
shape: 1.032 ms for the default against 1.032 ms for the per-shape best.

| shape | recipe | tile | ms | GB/s |
|---|---|---|---:|---:|
| (16384, 24576) | gradient, SR | (256, 64) | 0.493 | 2093 |
| (16384, 12288) | activation, RTN | (256, 64) | 0.210 | 2454 |
| (16384, 6144) | gradient, SR | (256, 32) | 0.149 | 1730 |
| (16384, 4096) | gradient, SR | (256, 32) | 0.105 | 1645 |
| (16384, 4096) | activation, RTN | (256, 32) | 0.075 | 2284 |

Note these are 1.6–2.5 TB/s, not the 850–990 GB/s §5.11 recorded — the matrix-unit
rotation and the shortened Philox have moved the kernel a long way since. Against
~8 TB/s of HBM it is still not bandwidth-bound, but the remaining headroom is not
addressable by launching the same kernel differently.

**Dropping the redundant FP32 widening: 0.7%, inside noise.** The kernel loads a
BF16 tile and immediately widens it to FP32 for the row-major operand, even
though both consumers have a BF16 form that produces the same bytes — the scale's
round-even tests the same mantissa bit at either width, and `_pack_fp4` has a
`v_cvt_scalef32_pk_fp4_bf16` path. That is a whole second copy of the tile live
across the A path, so it looked like the register pressure §5.11 pointed at.
Passing the loaded tile straight through moved the summed five-shape time from
1.032 to 1.025 ms. Reverted: that is below the harness's run-to-run spread, and it
is not worth moving a numerics path for.

What is left is the kernel's own arithmetic — the two amax reductions, the
E2M1 conversions and the in-register transpose — not how it is tiled or how wide
its operands are.

### 5.18 The fastest recipe measured, and why the step is 573 ms longer than its kernels

§5.16 ends on a projection — "the reachable number today is the `TAIL_BF16=0`
recipe at roughly 1.52×". This section measures it, then profiles it, because a
recipe that removes 644 ms of BF16 GEMM changes which bottleneck is next.

All arms are 60 steps at the §5.16 shape (Qwen3-8B 36L, GBS 128, seq 8192, MBS 2,
TP=1, 8×MI350X), median of iterations 21–60, eval outside the run:

| arm | median | mean | IQR | vs BF16 | loss@60 |
|---|---:|---:|---:|---:|---:|
| BF16 | 8520.7 ms | 8539.5 | 52.2 | 1.000× | 2.272813 |
| MXFP4 `TAIL_BF16=5` | 5984.6 ms | 5999.1 | 31.0 | 1.424× | 2.271212 |
| MXFP4 `TAIL_BF16=0` | 5591.0 ms | 5614.7 | 25.6 | **1.524×** | 2.270299 |
| … + `--no-check-for-nan-in-loss-and-grad` | 5575.2 ms | 5593.0 | 56.6 | 1.528× | 2.272596 |
| … + `--overlap-param-gather` | 5572.1 ms | 5585.9 | 37.7 | 1.529× | 2.269602 |
| `TAIL_BF16=0` + `--use-nccl-ub --disable-symmetric-registration --ddp-pad-buckets-for-high-nccl-busbw` | 5597.0 ms | 5606.2 | 16.3 | 1.522× | 2.272475 |
| `TAIL_BF16=0` + `LUMEN_SR_PHILOX_ROUNDS=4` (§5.20) | 5558.2 ms | 5578.0 | 37.1 | 1.533× | 2.272955 |
| … + both DDP flags — **best measured** | **5552.2 ms** | 5573.4 | 58.5 | **1.535×** | 2.273108 |

The projection was right: `TAIL_BF16=0` lands at 1.524×, against 1.52× predicted
from the kernel table. Every DDP flag below it is a negative result — 18.9 ms
across the first two against IQRs of 25.6 and 37.7, and 6.0 ms the *wrong* way for
the NCCL transport pair against an IQR of 16.3 — and they are reported here only
because the profile made them look like the obvious next move.

#### The `TAIL_BF16=0` profile

Clean two-iteration trace, eval disabled, at
`examples/qwen3/results/prof_clean_tail0/`. Every count is closed form: FP4 GEMM
3456 = 36 × 4 × 3 × 8, `_dual_layout_quant_mxfp4_kernel` 2304 = 36 × 4 × 2 × 8,
Tensile GEMM 24 = 3 × 8 for the vocab projection and nothing else, `fmha_fwd`
288 = 36 × 8.

| ms/iter | `TAIL_BF16=5` | `TAIL_BF16=0` |
|---|---:|---:|
| attention | 1734.5 | 1551.1 |
| FP4 GEMM | 1107.2 | 1283.2 |
| BF16 Tensile GEMM | 1089.4 | 448.0 (vocab only) |
| grad accumulate | 660.5 | 670.3 |
| quantize / layout | 359.1 | 423.8 |
| collectives | 260.3 | 406.3 |
| optimizer | 341.0 | 354.0 |
| everything else | 840.9 | 843.0 |
| **total kernel time** | **6392.9** | **5979.7** |

Two rows move for reasons that are not the recipe. **Attention drops 183 ms for
identical work** — `fmha_bwd` is the same kernel at the same 288 calls, 1336.5 ms
against 1157.8. Replacing BF16 GEMMs with FP4 ones lowers the power the GEMMs
draw and the rest of the model clocks higher, which is a real effect on the step
but means cross-trace comparisons of a "shared" category carry roughly 10%
uncertainty. **Collectives rise 146 ms** for the same bytes; that one is the
subject of the rest of this section.

#### The step is 573 ms longer than the compute stream is busy

`scripts/trace_compute_stream_gaps.py` splits the busiest kernel stream into busy
and idle and asks what is on the other streams during each idle interval.

| | BF16 | MXFP4 `TAIL_BF16=5` | MXFP4 `TAIL_BF16=0` |
|---|---:|---:|---:|
| step | 8520.7 ms | 5984.6 | 5591.0 |
| compute stream busy | 8196.3 | 5623.0 | 5044.8 |
| compute stream idle | 388.3 | 388.2 | **573.3** |
| — of which a collective is in flight | 128.0 | 150.9 | **302.6** |

The idle is not spread out. At `TAIL_BF16=0` two intervals per iteration, 293.2
and 245.6 ms, are 100% covered by a collective and account for over half of it.
They sit at 85% of the way through the step, where the last gradient buckets
become ready.

**Exposed communication grows as MXFP4 gets faster.** The bytes are fixed by the
parameter count, not the recipe: gradients reduce-scatter in FP32
(`accumulate_allreduce_grads_in_fp32`, so 8B × 4 B = 32 GB) and parameters
all-gather in BF16 (16 GB), identically in all three arms. What shrinks is the
backward pass available to hide them behind, so the same collective goes from 128
to 303 ms of exposure. This is the structural cost of making the compute faster
without touching the communication, and it is why the `TAIL_BF16=0` step improved
by 394 ms when its kernel time fell by 413.

#### Two Megatron knobs that do not close it, and why

The host trace names the stall precisely. Inside
`distributed_data_parallel.py:448 hook` → `register_grad_ready` →
`start_grad_sync` → `param_and_grad_buffer.py:175 check_grads` →
`rerun_state_machine.py:436 validate_result` there is an `aten::item`, and it
blocks for the full 293 ms. `check_grads` takes an L2 norm of each bucket's
gradient buffer and hands it to two `validate_result` calls — one `torch.isnan`,
one `torch.isinf` — each of which is a device sync in the backward hook.
`check_for_nan_in_grad` defaults False in `DistributedDataParallelConfig`, but
`training.py:975` overwrites it from `args.check_for_nan_in_loss_and_grad`, which
defaults True. It costs 578.8 ms/iter of host block time in the MXFP4 arm and
927.9 in BF16, across 268 syncs per iteration.

That made it look like the cause. It is not: `--no-check-for-nan-in-loss-and-grad`
turns the syncs off (verified as `check_for_nan_in_grad=False` in the config the
run logs) and buys **15.8 ms**, inside the IQR. The sync was waiting on the same
collective the compute stream was waiting on, so removing it lets the host run
ahead into work that has nowhere to go. `--overlap-param-gather` adds a further
**3.1 ms**, also noise — the exposure is on the gradient reduce-scatter, and
overlapping the parameter all-gather does not touch it.

A third flag pair aimed straight at the transport is also negative:
`--use-nccl-ub --disable-symmetric-registration --ddp-pad-buckets-for-high-nccl-busbw`
measures **5597.0 ms, 6.0 ms slower** than the baseline at an IQR of 16.3. (Plain
`--use-nccl-ub` dies first: `nccl_allocator.py:135` raises "symmetric setting with
torch.cuda.MemPool requires higher PyTorch version", so the local-registration
opt-out is required on this PyTorch.) The result is consistent with what the flag
claims — user-buffer registration lowers the SM count communication takes *while
overlapped*, and pad-buckets changes transfer sizes. Neither touches the reason
the collective is exposed in the first place.

So the exposure does not appear to be reachable from DDP-level configuration. Two
readings remain, and they are not distinguished yet:

- **Wire time.** The FP32 gradient reduce-scatter moves 32 GB; at 8 ranks that is
  28 GB of bus volume, and 292 ms for it is roughly 100 GB/s. Halving it by
  reducing in BF16 is not available: `--grad-reduce-in-bf16` sets
  `DDPConfig.grad_reduce_in_fp32=False`, which makes `grad_dtype` the parameter
  dtype (`distributed_data_parallel.py:168`), and `param.main_grad` is a view into
  a buffer allocated at that dtype (`param_and_grad_buffer.py:722`, `:752`). The
  flag therefore moves the *accumulation* of all 8 micro-batches into BF16, not
  just the wire — roughly 64-way accumulation at 8 mantissa bits once the
  cross-rank reduce is included. Megatron's own comment at `arguments.py:759` says
  bf16 requires fp32 gradient accumulation, and it has been observed to diverge in
  RL. Out of scope.
- **Rank skew.** `partition_buckets`' rule 2 gives each bucket group a single
  bucket when no FP8 buffer is present, and the trace has 226 collective calls per
  iteration against ~200 buckets at `bucket_size=40000000` — so this is many small
  reduce-scatters, not two large ones. Two of them nonetheless account for 292 of
  the 406.3 ms, which is what a collective looks like when it spins waiting for the
  slowest rank. Only rank 0 was profiled, so this cannot be confirmed from these
  traces; `--profile-ranks 0 1 2 3 4 5 6 7` and comparing arrival times is the
  experiment. Both RCCL and AITER print a NUMA-auto-balancing warning every run
  (`sudo sysctl kernel.numa_balancing=0`, needs root), which is the cheapest thing
  to rule out first.

Either way this is shared-infrastructure work rather than MXFP4 work.

#### The 1.6× arithmetic, with the profile behind it

1.6× is a 5325.4 ms step, 246.7 ms below the best arm measured. Three routes,
all closed:

- **Shared work, including all the idle above.** Taking a constant *c* off both
  arms needs (8520.7 − c)/(5572.1 − c) = 1.6, i.e. *c* = 658 ms. The MXFP4 arm
  only has 573.3 ms of idle in total and BF16 has 388.3, so perfect scheduling in
  both arms does not get there — it would have to come out of kernels both arms
  run, such as a 40% cut to attention, which makes the ratio look better without
  making MXFP4 faster.
- **Quantization.** It is the only MXFP4-owned line item left at `TAIL_BF16=0`:
  423.8 ms/iter. Reaching 1.6× means removing 246.7 of it, 58%. §5.17's tile
  sweep and FP32-widening attempt both came back empty, and §5.11 already ruled
  out memory traffic as the limiter.
- **The vocab projection**, the last 448.0 ms of BF16 GEMM. Quantizing it needs
  FP4 kernels at N = 151936, and the tuned A4W4 table covers N ∈ {4096, 6144,
  16384, 24576} only, so all three of its GEMMs would fall back to Triton — which
  the 2026-08-19 wgrad-accumulation measurement prices well above the ASM path at
  these sizes. It is also the layer every FP4 paper keeps in high precision, and
  §5.14's C4 sweep could not resolve tail accuracy to better than ~0.12 nats.

**The answer stands: 1.6× is not available at Qwen3-8B / GBS 128 / seq 8192.** At
the time of writing the best measured recipe was 5572.1 ms against BF16's
8520.7 ms, or 1.53×; §5.20 then took it to 5552.2 ms (1.535×) by cutting Philox
rounds, which does not change the conclusion.

### 5.19 What the quantizer's time is actually spent on

With the collective ruled out at the configuration level (§5.18), the quantizer is
the only MXFP4-owned item left. Its launch parameters are now exhausted in both
directions. §5.11's tile sweep had already found the launcher's BLOCK_M/BLOCK_N
rule optimal and `num_warps` 8 and 16 slower than the default 4 everywhere;
`--axis pipeline` adds `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim` and
`kpack`, and the aggregate over one call of each production shape is **1.01×,
with two of the five shapes slower**, every win inside run-to-run noise.

`num_stages` doing nothing has a mechanical reason worth recording:
`_dual_layout_quant_mxfp4_kernel` contains no loop. It is one `tl.load`, two
`_calculate_fp4_scales`/`_pack_fp4` pairs, one Hadamard, four stores — straight-line
code, so there is no loop nest for software pipelining to stage. `waves_per_eu=0`
(unset, i.e. let the scheduler choose) also beat every explicit floor on every
shape. Nothing about *how* this kernel is launched is on the table any more.

So `--axis features` asks the other question — which *computation* a rewrite should
attack. Each row turns one thing off and holds the rest at what production passes.
None is a shippable configuration; they are cost attributions:

| shape | production | SR → RTN | MFMA → butterfly | swizzle off |
|---|---:|---:|---:|---:|
| grad gate_up (N=24576) | 0.482 ms | **−26%** | +101% | +6% |
| grad qkv (N=6144) | 0.141 ms | **−22%** | +63% | +6% |
| grad o/down (N=4096) | 0.099 ms | **−26%** | +65% | +4% |
| act down (N=12288) | 0.192 ms | — | +130% | +4% |
| act qkv/o/gate_up (N=4096) | 0.074 ms | — | +96% | +3% |

Three things follow, and one of them contradicts a comment in the source:

- **Stochastic rounding costs 22–26% of the gradient-path quantizer.**
  `_generate_randval`'s docstring claims the four-words-per-round trick dropped SR
  to "nothing measurable (0.146 → 0.059 vs 0.057 RTN)". That is stale — the trick
  is real and load-bearing, but at production tiles and shapes SR is back to a
  quarter of the kernel on every gradient shape. It is the largest addressable
  feature. `SR_PHILOX_ROUNDS` is already reduced from Philox's standard 10 to 7 and
  could go lower, since SR dither has no cryptographic requirement — but that is a
  numerics change and has to go through §5.14's harness.
- **The MFMA Hadamard is load-bearing, not a tuning artifact.** Falling back to the
  butterfly costs 63–130%. Leave it alone.
- **Fusing the scale swizzle into the quantizer is net positive.** It costs 3–6%
  here while removing the standalone `swizzle_mxfp4_scale` pass, 27.24 ms/iter.

None of it reaches 1.6×. Making SR entirely free is worth ~24% of the gradient half
of 423.8 ms/iter, call it 51 ms, which moves 1.529× to 1.543×. The step needs 247 ms.
A rewrite would have to roughly **double** the whole kernel — production sustains
1734–2687 GB/s against ~8 TB/s of HBM, so the headroom exists on paper, but the
butterfly result says the kernel is genuinely compute-bound on the rotation and
pack rather than parked on a bad memory schedule. A successful 2× rewrite lands
423.8 → ~212 ms, i.e. 5360 ms and **1.589×** — still short, and only on the
assumption that a 2× rewrite lands at all.

### 5.20 Why a hand-written kernel will not double it, and what does help

Before committing to a HIP rewrite, `benchmarks/inspect_dual_layout_asm.py` asks
what the compiler is already achieving. The answer closes the case for a rewrite
as a *scheduling* exercise:

| shape | tile | VGPR | spills | LDS | occ (waves/SIMD) | instructions |
|---|---|---:|---:|---:|---:|---:|
| grad gate_up | (256, 64) | 118 | **0** | 32 KB | 4 | 2415 |
| act down | (256, 64) | 117 | **0** | 32 KB | 4 | 1554 |
| grad qkv | (256, 32) | 88 | **0** | 16 KB | 5 | 1331 |
| grad o/down | (256, 32) | 88 | **0** | 16 KB | 5 | 1330 |
| act qkv/o/gate_up | (256, 32) | 84 | **0** | 16 KB | 5 | 885 |

Zero spills everywhere, 4–5 waves/SIMD, and the instruction stream is 66–78%
plain VALU against 0.6–1.0% MFMA and well under 1% memory. There is no register
pressure to relieve and no memory schedule to fix; instruction count *is* time.
A rewrite would have to remove arithmetic, and the AMDGCN says which arithmetic
there is to remove. On `grad gate_up`, of 2415 instructions:

| what | instructions | share |
|---|---:|---:|
| Philox (`v_xor_b32` 386, `v_mul_lo_u32` 210, `v_mul_hi_u32` 200) | **796** | **33%** |
| address/bit arithmetic (`v_lshlrev` 135, `v_and` 100, `v_add_u32` 89, `v_perm_b32` 88, `v_bfe` 45, …) | ~468 | 19% |
| amax reduction (`v_max_f32` 152 + 48, `v_max3_f32` 40, `v_max_u16` 22) | ~262 | 11% |
| FP4 convert/pack (`v_cvt_scalef32_sr_pk_fp4_f32`) | 64 | 3% |
| Hadamard (`v_mfma_f32_16x16x16_bf16`) | 16 | 0.7% |

A third of the kernel is generating random numbers. That is not something a
rewrite is better at than Triton — it is Philox, and it costs what it costs.

**The hardware PRNG is not a way out.** gfx950 has `v_prng_b32`, one instruction,
and `v_cvt_scalef32_sr_pk_fp4_f32` already uses it internally to derive the second
FP4 lane's dither from the seed it is handed — so half of production's SR noise is
already hardware PRNG output, which makes "use it for everything" the obvious
idea. `benchmarks/probe_prng_b32.py` characterises it and the idea fails: it is a
pure, bijective, *linear* map (popcount not preserved, so not a bit permutation;
some output bit equals some input bit exactly, so a GF(2) map with a pass-through
field — an xorshift step). Applied to already-uniform input it is uniform (worst
bit |p−0.5| = 0.0011 against a 0.00195 tolerance, byte χ²/df = 1.00); applied to
the counter sequence 0,1,2,… it fails outright (|p−0.5| = 0.5, χ²/df = 4112), and
`prng(0) = 0` is a fixed point. It stirs entropy, it does not create any. Being a
bijection on 32 bits, a chain of any length is a deterministic function of its
seed and carries 32 bits of entropy total, so dithering a 32-element quantization
block from one chain gives all 32 elements deterministically related noise —
which is precisely the independence SR exists to provide. The hardware's own use
shares one seed across *two* adjacent elements, a bounded trade; 32 is not.

**What does help is buying the same entropy for fewer rounds.** The Philox
instruction count is near-linear in `SR_PHILOX_ROUNDS`, which Lumen already runs
at 7 against Random123's default of 10. `benchmarks/bench_sr_philox_rounds.py`
sweeps it for both speed and the property a reduction could break — SR is
unbiased when E[dequant(quant(x))] = x, so the mean residual over many draws
should be sampling noise and nothing more:

| rounds | grad gate_up | grad qkv | grad o/down | Philox share | resid mean | resid std |
|---|---:|---:|---:|---:|---:|---:|
| 7 (current) | 0.502 ms | 0.159 ms | 0.107 ms | 33% | 0.00004 | 0.02049 |
| 5 | 1.08× | 1.06× | 1.08× | 25% | 0.00016 | 0.02047 |
| **4** | **1.13×** | **1.16×** | **1.14×** | 20% | 0.00009 | **0.02012** |
| 3 | 1.18× | 1.20× | 1.17× | 15% | −0.00033 | 0.09922 |
| 2 | 1.22× | 1.25× | 1.26× | 9% | 0.00032 | 0.16511 |

The mean stays at sampling noise throughout — SR remains unbiased even at 2 rounds,
as the hardware instruction guarantees. The **standard deviation** is what fails:
flat at 0.0205 through 4 rounds, then 4.8× worse at 3 and 8× at 2. That is the
dither losing independence across the block, showing up as per-element error that
no longer averages out. **Four rounds is the floor**: statistically
indistinguishable from seven, 1.13–1.16× on every gradient shape.

`SR_PHILOX_ROUNDS` now reads `LUMEN_SR_PHILOX_ROUNDS` at import, because it has
to be fixed before tracing — Triton 3.7 rejects a global that changed after
compiling, so a sweep cannot patch it in place.

Which leaves the honest accounting for a rewrite. Philox at 4 rounds is 20% of
the stream and irreducible; the MFMA Hadamard is 0.7% and already optimal; amax is
11% and is a reduction tree with little slack. The one place a hand-written kernel
should beat Triton is the 19% of address and bit arithmetic, where precomputed
offsets and strength reduction might recover half. That is ~10% of the stream, or
about **1.1× on the kernel** — not 2×. Stacked on the round reduction, the
quantizer might reach ~1.3× overall, 423.8 → ~326 ms/iter, which is ~98 ms off the
step: **5474 ms, or 1.557×**.

**So the conclusion of §5.18 survives the deeper look, with the ceiling raised
slightly: the reachable range at this shape is ~1.55–1.56×, and 1.6× is not
available.** The quantizer is not badly written; a third of it is buying entropy
and the rest is close to the arithmetic it needs.

#### Measured end to end

`LUMEN_SR_PHILOX_ROUNDS=4` at `TAIL_BF16=0`, 60 steps, median of iterations 21–60:
**5558.2 ms against the 5591.0 ms baseline, −32.8 ms, 1.533×**, loss@60 2.272955
against the baseline's 2.270299 and inside the spread of every other arm
(2.2696–2.2731). Stacking the two DDP flags from §5.18 gives the best figure
measured in this campaign, **5552.2 ms, or 1.535×**.

Read the −32.8 ms honestly: it is ~0.9× the IQR (37.1), so it sits at the edge of
what a 40-iteration median resolves. Two things argue it is real rather than
drift. It is the largest single-change delta measured here — against −15.8 and
−18.9 ms for the DDP flags and +6.0 for the NCCL transport pair — and, unlike
those, it was *predicted before it was measured*: the microbenchmark's 1.13–1.16×
on gradient shapes implies ~29–45 ms of step time, and 32.8 landed inside that
window. A confirmation matching an a-priori prediction is stronger evidence than
its own effect size. The DDP flags adding a further 6.0 ms remains noise, exactly
as it was on its own.

#### Through the precision harness

Step time is the cheap half. The round reduction changes gradient numerics, so it
went through `examples/scripts/run_precision_matrix.sh` at the configuration
§5.14's C4 tail sweep used — 300 steps of real C4 with held-out validation and a
held-out test slice, `qwen3_8b`, `TAIL_BF16=0`, seed 1234, both arms sharing one
autotune cache so the per-shape backend choices are pinned and the round count is
the only difference.

`TAIL_BF16=0` is deliberate: it quantizes every layer, so it is the arm that pays
the most SR and the strictest test of a dither change.

A two-arm comparison would not have been interpretable on its own, because the
MXFP4 path is not bitwise reproducible at a fixed seed (§10: the Qwen3-8B repeat
moved 0.005 while both BF16 repeats reproduced digit for digit). So the control
was run twice to measure that noise at this exact configuration:

| arm | held-out valid | test | step time |
|---|---:|---:|---:|
| rounds=7 (A) | 6.3924 | 6.4004 | 5659.8 ms |
| rounds=7 (B, identical repeat) | 6.4024 | 6.4096 | 5651.8 ms |
| **rounds=4** | **6.3856** | **6.3924** | **5629.1 ms** |

| | validation | test |
|---|---:|---:|
| same-config noise, two identical rounds=7 runs | 0.0100 | 0.0092 |
| rounds=4 vs the rounds=7 mean | **−0.0119** | **−0.0126** |
| as multiples of the noise sd (s from n=2) | −1.67 sd | −1.94 sd |

**The verdict is no evidence of harm.** The effect is the same order as the noise
of re-running the identical configuration — 1.7–1.9 sd against a standard
deviation estimated from two samples, which is not significant at any conventional
bar — and both held-out slices move in the *better* direction. A dither that had
lost independence would raise loss, not lower it, and §5.20's residual statistics
already showed why: at 4 rounds the residual std is 0.02012 against 7 rounds'
0.02049, while 3 rounds is 0.09922. Four rounds is not a degraded dither, it is
the same dither computed with less mixing.

The step time reproduces too: −26.6 ms against the two-run rounds=7 mean, on a
different corpus and step count from the 60-step measurement's −32.8 ms.

Three limits worth stating plainly. 300 steps bounds short-horizon behaviour and
says nothing about long-horizon convergence. It says nothing about RL, where
gradient-precision changes have bitten this project before — a different
mechanism (accumulation dtype, §5.18) but the same class of caution. And the noise
sd rests on two samples.

So `SR_PHILOX_ROUNDS` **stays at 7 by default**. The knob exists, the win is real
but small — 26–33 ms, ~0.5% of the step, 1.524× → 1.533× — and that is not worth
changing gradient numerics by default until it has ridden along in a long run.
Every figure here is at `TAIL_BF16=0`; §5.21 prices the same knob at the shipped
`TAIL_BF16=5` recipe, where it is worth 0.21% rather than 0.48%, and adds the
argument that carries more weight than either number — 4 is the last value before
the dither collapses, so it has no margin to spare as a default.
`LUMEN_SR_PHILOX_ROUNDS=4` announces itself on rank 0 when set, because the round
count is not a Megatron argument and would otherwise leave no trace in the log.

### 5.21 Re-measured on `feature/mxfp4` (2026-08-27)

Every step time in §5.15–§5.20 was measured on `bench/mxfp4-ablation-staircase`.
The optimization work has since been migrated to `feature/mxfp4`, which also
carries the module-boundary fix to the BF16-skip prefixes (§5.13) that the bench
branch did not. A migration that moves ten commits across branches can drop a
wiring change without failing anything, so the headline arms were re-measured on
the branch that now owns them.

Same shape and launch as §5.18 — Qwen3-8B 36L, GBS 128, seq 8192, MBS 2, TP=1,
8×MI350X, native launch, seed 1234, `EVAL_INTERVAL=100000` so no eval lands
inside the window, and the same `mxfp4_autotune_qwen3_8b.json` so per-shape
backend choices are pinned the same way. Shortened to 40 steps with the median
taken over iterations 21–40: the warm-up plateau is reached by iteration 21
(§5.20's logs show 5749 ms at iteration 5 settling to 5576–5597 from 21 on), so
the change costs window width, not steady state.

| arm | median | mean | IQR | vs BF16 | §5.18 median | Δ |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 8510.2 ms | 8528.7 | 16.6 | 1.000× | 8520.7 ms | −10.5 ms |
| MXFP4 `TAIL_BF16=5` | 6009.5 ms | 6022.8 | 24.6 | 1.416× | 5984.6 ms | +24.9 ms |
| MXFP4 `TAIL_BF16=0` | 5598.1 ms | 5598.3 | 9.1 | **1.520×** | 5591.0 ms | +7.1 ms |

All three reproduce. The two ratios move by 0.008 and 0.004, and every step-time
delta sits at or under the IQR of the arm it belongs to — BF16 −10.5 against 16.6,
`TAIL_BF16=0` +7.1 against 9.1. `TAIL_BF16=5`'s +24.9 is the one worth naming
honestly: that is about 1.0× its own IQR and 0.8× the 31.0 the §5.18 arm had, so
it is at the edge of what a 20-iteration median resolves rather than comfortably
inside it. Nothing here separates the branches; what it rules out is a migration
that silently cost performance.

Logs `lumen_qwen3_8b_recheck0827_bf16.log`,
`lumen_qwen3_8b_recheck0827tail5_mxfp4.log` and
`lumen_qwen3_8b_recheck0827tail0_mxfp4.log`. Step time only — these are 40-step
mock-corpus runs, so they carry no loss figure comparable to §5.18's loss@60.

#### `SR_PHILOX_ROUNDS=4` priced at both recipes, and why 7 is still the default

§5.20 measured rounds=4 only at `TAIL_BF16=0`, where it was worth −32.8 ms. But
the *shipped* recipe is `TAIL_BF16=5`, and the knob cannot be worth the same
there: five of 36 layers stay in BF16 and pay no SR at all, and the step carries
644 ms more BF16 GEMM in the denominator. So it was measured at both.

Both rounds=4 arms came back with a larger IQR than their rounds=7 counterpart
(64.1 against 24.6, 47.1 against 9.1), which is not something fewer Philox rounds
can cause. Machine interference is one-sided — a disturbed iteration is slower,
never faster — so the lower quantiles are the robust read and are quoted
alongside the median rather than instead of it:

| recipe | rounds=7 | rounds=4 | Δ median | Δ p25 | Δ fastest | vs BF16 |
|---|---:|---:|---:|---:|---:|---|
| `TAIL_BF16=5` (default) | 6009.5 ms | 5999.9 ms | −9.6 | −12.8 | −13.0 | 1.416× → 1.418× |
| `TAIL_BF16=0` | 5598.1 ms | 5571.4 ms | −26.7 | −28.6 | −31.6 | 1.520× → 1.527× |

`TAIL_BF16=0`'s −27 to −32 ms agrees with §5.20's −32.8 and the precision
harness's −26.6, so the effect is real and its size is settled. The new number is
the other row: **at the recipe that ships, the knob is worth about 13 ms, 0.21% of
the step.**

**The default stays at 7, and not because 4 looks bad.** It does not. Residual std
at 4 rounds matches 7 within 3%, the 300-step harness put r4's held-out loss
*below* r7's on both slices, and both arms above ran 40/40 steps with zero NaN.
Two other things decide it:

1. 0.21% at the shipped recipe is too little to spend a gradient-numerics change
   on. §5.20 reached the same conclusion from the tail0 figure alone, which at
   0.48% was the more generous of the two.
2. **4 has no margin below it.** Residual std is flat across 7, 5 and 4 (0.0205)
   and then jumps to 0.0992 at 3 — a 4.8× step. Four is the last good value. For
   an opt-in knob that is fine, because whoever sets it has measured their own
   configuration. For a default it is not: if some future shape, sequence length
   or recipe shifts dither quality even slightly, 7 degrades gradually while 4
   falls off a cliff, and the symptom is a loss curve that slowly gets worse
   rather than anything that raises.

So the guidance is per-recipe rather than global. A run already at
`TAIL_BF16=0` can set `LUMEN_SR_PHILOX_ROUNDS=4` for roughly 29 ms, having
accepted that it sits on the edge of the plateau; the knob announces itself on
rank 0 so the choice leaves a trace in the log. At the default `TAIL_BF16=5`
there is little point.

Logs `lumen_qwen3_8b_recheck0827tail5srp4_mxfp4.log` and
`lumen_qwen3_8b_recheck0827tail0srp4_mxfp4.log`.

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
- Per the NVFP4 paper, end-of-network layers are most sensitive to FP4, and ~15% in BF16 is the primary stabilization lever (the AMD/PSU paper makes no such claim; its lever is the deterministic Hadamard)
- Row 5 moved **two** knobs at once (G 32→16 and the tail), so the table does not attribute the fix to either alone. Every diverging row is H32; **H16 without a tail is untested**. That is the arm to run before treating the tail as optional, and it has to reach the horizon where rows 2–4 broke — a 300-step Megatron sweep at GBS 128 stays clean in all four tail settings (§5.14), which is far too early to count.

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
| Megatron wiring (format arg, weight-cache hook) | `lumen/models/megatron.py` |
| Pretraining script (C4 + TensorBoard) | `examples/qwen3/pretrain_qwen3_mxfp4.py` |
| Megatron launcher (Qwen3-8B, TP=PP=1) | `examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh` |
| Megatron training body (docker + native) | `examples/qwen3/scripts/train_qwen3_8b.sh` |
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
- [x] **Megatron backend runs end to end** — 8B/36L, TP=PP=1, C4 200 steps, Δ val_loss = +0.0211 vs BF16, 1.13× throughput, zero NaN (§4.5)
- [x] **Lumen MXFP4 vs TE MXFP4 on Megatron, 1000 steps** — held-out val_loss 5.7793 vs 5.8479 (Lumen better by 0.069 nats), step time 2007 vs 1425 ms (TE 1.41× faster), zero NaN in both (§4.6)
- [x] **TransformerEngine built for gfx950 with CK fused attention** — required for any GQA model on ROCm; AOTriton rejects GQA outright (§4.6.2)
- [x] **Recipe/implementation split resolved** — 4-arm ladder attributes 85% of Lumen's convergence advantage to the Hadamard, 9% to the BF16 tail, ~0.004 nats to everything else; TE remains 1.29× faster at matched recipe (§4.7)
- [x] **Dual-layout gradient quantization** — `dual_layout_quant_mxfp4` emits both MXFP4 layouts from one dense read, bit-exact against the two kernels it replaces, −13.2 ms/step (§5.11)
- [x] **Step time re-verified on `feature/mxfp4`** (2026-08-27) — BF16 8510.2 ms, MXFP4 `TAIL_BF16=5` 6009.5 ms (1.416×), `TAIL_BF16=0` 5598.1 ms (1.520×); every delta at or under the arm's own IQR, so the migration off the ablation branch cost nothing (§5.21)

### Open Issues

1. **GEMM prologue fusion** — quant and GEMM remain separate kernel launches. Full fusion requires AITER changes. This is the largest remaining gap (§5.2).
2. **Memory regressed, not saved** — 15.30 → 20.90 GB (§5.3). BF16 master weights are still retained for FSDP2, and the weight cache adds 4.8 GB. True savings need FP4 weight storage with FP4-aware FSDP.
3. **FP4 all-gather benefit unverified** — the dequant path is pure PyTorch and may cost more than the bandwidth it saves on a single node (§2.8).
4. **Changes 5–8 were not A/B'd individually** — the 1.221× in §4.4 is their combined effect, measured on a shared machine where isolated micro-benchmarks were unreliable.
5. **BF16 vs MXFP4 not compared at same lr in §4.3** — 8B BF16 ran at lr=6e-5, MXFP4 at lr=1e-4. §4.4 uses the same lr for both.
6. **Megatron patches break on `rocm_dev` HEAD** — `_patched_fwd_attn` forwards Megatron's new `padding_mask` into `Attention.forward`, which rejects it (§2.9). Pinning to `core_r0.15.0_rocm` is a workaround, not a fix.

### Next Steps

1. **Finish the current 3000-step 8B run** for a same-lr convergence comparison against the BF16 baseline at the new build.
2. **Isolate the FP4 all-gather** with a dedicated `--no-mxfp4-comm` A/B on an idle machine, and port `convert_from_mxfp4_2d` to Triton if the gather is bandwidth-bound.
3. **Cache only the pre-transposed weight** to recover ~half of the 4.8 GB (§5.3).
4. **AITER GEMM prologue fusion** — request AITER to fuse H+Q into GEMM tile load, eliminating all intermediate memory traffic.
4. **Gradient quantization** — enable `quantize_grad="mxfp4"` for communication bandwidth reduction in multi-node training.
5. **Megatron backend** — TP=PP=1 runs and stays within +0.32% of BF16 over 200 C4
   steps, but the gap is still widening at step 200 (§4.5). Needs a run long enough
   to see whether it plateaus, and TP/PP
   is worth attempting. TP will also re-shape the GEMMs, so the §2.2 per-shape
   backend decisions have to be re-measured.
6. ~~**Tune Qwen3-8B's 8 missing MXFP4 GEMM shapes into AITER's table**~~ — done,
   −98 ms/step, GEMM now within +13.6 ms of TE (§5.6). Re-tune whenever the token
   count or TP width changes, since the rows key on the exact M/N/K.
7. ~~**Enable rope fusion**~~ — done, −137.7 ms by passing `--lumen-fused-rope`
   (§5.7). **The +60 ms norm gap is still open, but it is not an `LNImpl` swap**
   (§5.9): it is entirely QK norm at N=128, where AITER's `rms_norm` moves 256
   bytes per iteration and loses to torch. It needs a Lumen RMSNorm that tiles
   several rows per program; pointing the spec at the existing kernel costs
   49.1 ms instead of saving 60.
8. ~~**Fuse the gradient's two quantizations**~~ — done, −13.2 ms (§5.11). The
   activation half is still open: `dequant_transpose_mxfp4` + `hadamard_quant_mxfp4`
   remain two passes, and the first only exists because the WGrad operand has to be
   rebuilt from the saved FP4 activation. Folding the dequant into the rotation
   would remove a full BF16 (K, M) roundtrip. §5.11 suggested cutting the kernel's
   register pressure first, since it only sustains ~900 GB/s against a 5129 GB/s
   dense-read ceiling — **that suggestion is dead: §5.19 measures zero spills on
   every production shape, and tile, occupancy and register tuning are all closed
   as negative results (§5.17, §5.19).** The gap is instruction count, not
   scheduling. Use
   the CUDA-event harness from §5.10, which reproduces the trace to 0.2% in 6
   seconds, rather than full training runs.
9. **Gradient-accumulation fusion** — ~370 ms/step, the largest single kernel in
   both stacks' traces, currently disabled by `--no-gradient-accumulation-fusion`
   in the shared launcher config. Not a Lumen-vs-TE gap; it would pay in both.
10. **Do not widen Lumen's Hadamard scope** — §4.7 shows TE's all-operand rotation
   does not beat Lumen's WGrad-only rotation.
11. **Decide the BF16 tail on evidence** — it costs 78.8 ms/step per layer, 6.4% for
   the default 5 (§5.14), which makes it the largest recipe-level speedup still on
   the table. Two things are needed and neither is another short A/B. Several seeds
   per arm, because one sweep put `TAIL_BF16=4` 0.124 nats *ahead* of the default
   while `TAIL_BF16=0` was 0.037 behind, so the run-level spread swamps the effect.
   And an H16-no-tail run carried to the horizon where §6.3's rows 2–4 diverged,
   since that combination has never been tested and divergence is a tail risk that
   no mean over 300 steps can rule out.
