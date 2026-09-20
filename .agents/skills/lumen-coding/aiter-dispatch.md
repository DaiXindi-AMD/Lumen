# AITER Dispatch Guide

## Overview

New GPU kernels of every backend belong in AITER; Lumen owns dispatch, argument adaptation, and fallback. Existing `lumen/kernels/` code is legacy, not a destination for new kernels. Never assume AITER or one of its backends is available on an AMD target: use cached probes.

## `try_backends()` -- Core Dispatch

```python
from lumen.ops.dispatch import try_backends, build_fallback_chain, Backend

chain = build_fallback_chain({
    Backend.ASM: _asm_fn,       # preferred: lowest latency on gfx94x
    Backend.CK: _ck_fn,         # fallback: ASM requires K%64==0
    Backend.TRITON: _triton_fn, # fallback only if its probe succeeded
})
return try_backends(chain, x, op_name="my_op")
```

**Priority order:** ASM -> CK -> Triton is a common default; use the actual op/scaling-specific chain below. A PyTorch fallback exists only where the op explicitly supports it.

**Exception handling:** Inspect the target `try_backends()` catch list. The migration checkout also catches `IndexError` and `KeyError` for AITER config-lookup failures, in addition to `RuntimeError`, `NotImplementedError`, `TypeError`, `ValueError`, Triton `CompilationError`, and `OutOfResources`.

**Synchronization:** Surface asynchronous GPU errors before fallback decisions in correctness/debug validation. Let the current dispatcher own synchronization: the migration checkout synchronizes during warmup, caches a successful entry by stable label, and skips sync on the cached path or with `LUMEN_SKIP_BACKEND_SYNC=1`. Graph capture requires warmup. Do not add unconditional synchronization to timed, captured, or compiled paths. The compile guide below is a design reference, not proof those mechanisms are implemented.

## Probe Functions

Every AITER import is guarded by a `@functools.lru_cache(maxsize=1)` probe:

```python
@functools.lru_cache(maxsize=1)
def _probe_aiter_asm_gemm():
    try:
        from aiter import asm_gemm  # noqa: F401
        return True
    except (ImportError, OSError):
        return False
```

**Rule:** When adding a new AITER backend, add a probe in `dispatch.py` and gate the backend in `build_fallback_chain`.

## Backend forcing (verify the checkout)

The source design proposed `LUMEN_BACKEND=triton` as a single-backend override. The migration checkout's `build_fallback_chain()` does not read it, so setting it alone does not establish which backend ran. Inspect the implementation, select a supported explicit backend or isolate the chain in a targeted test, and verify the selected entry in logs. If a later checkout implements the override, use its documented behavior.

## FP8 GEMM Backend Dispatch

| Scaling Type | Backends (priority) | Tuning Config |
|---|---|---|
| delayed/dynamic | hipBLASLt -> CK (`gemm_a8w8_CK`) -> Triton | `a8w8_tuned_gemm.csv` |
| per_token | Triton only (`gemm_a8w8_per_token_scale`) | None |
| blockwise/blockwise2d | CK (`gemm_a8w8_blockscale`) -> Triton | `a8w8_blockscale_tuned_gemm.csv` |
| mxfp8 | Triton only (`gemm_mxfp8`) | None |

## BF16 GEMM

All GEMM kernels use TN layout: `Y = A @ W^T`, weight is always `(N, K)`.

- `tuned_gemm` / `hipb_mm` / `F.linear` require matching dtypes -- cast after dequant
- Triton `gemm_a16w16`: BLOCK_SIZE_K >= 128 on gfx942
- ASM: requires K % 64 == 0 and N % 64 == 0
- Backward dgrad: pass `weight.t().contiguous()` as `w`

## FP8 Attention Dispatch

| Quant Type | Autograd Function | Scale Shape |
|---|---|---|
| `blockwise` | `AttentionTritonFunction` | 1D: `[B, H, ceil(S/bm)]` |
| `blockwise2d` | `AttentionTritonBlockwise2DFunction` | 2D: `[B, H, S//bm, D//bn]` |
| `mxfp8` | `AttentionTritonMXFP8Function` | MXFP8 block scales |

BF16 attention routes through AITER CK (`FlashAttnFunc`) as primary, Triton as fallback.

## Grouped GEMM

- BF16: `aiter.ops.triton.gmm`
- FP8 per-token / mxfp8: fused MOE GEMM kernels (`moe_gemm_per_token`, `moe_gemm_mxfp8`)
- FP8 blockwise: `moe_gemm_a8w8_blockscale` with `RoutingData` (built from `group_sizes`)
- Sequential fallback: iterates per-group when no fused kernel
- wgrad: `ptgmm` (permuted-tensor GMM)

## RoPE Dispatch (AITER Triton, no PyTorch fallback)

| Priority | Kernel | Layout | Use Case |
|----------|--------|--------|----------|
| 1 | `rope_cached_thd_positions_2c_fwd` | THD | Q+K pair (one launch) |
| 2 | `rope_cached_fwd` | SBHD | Single tensor RoPE |

Layout adaptation: Lumen BHSD -> AITER SBHD/THD via permute before/after kernel call. Extended API includes `rope_fwd_2d` (vision) and `rope_fwd_3d` (video).

## MoE Fused Dispatch

| Component | Kernel | Notes |
|-----------|--------|-------|
| Token sorting | `moe_align_block_size_triton` | Triton block-padded sort |
| Fused GEMM | `fused_moe` (AITER) | End-to-end sort+GEMM+weight-mul |
| TopK | `aiter.topk_softmax` | ASM-backed |
| Permute | `aiter.moe_sorting_fwd` | |
| Unpermute | `aiter.moe_sum` | |

## Adding a New Op

1. Add probe function in `dispatch.py`: `_probe_aiter_<name>()`
2. Build fallback chain in the op function: `build_fallback_chain({Backend.ASM: ..., Backend.CK: ..., Backend.TRITON: ...})`
3. Call `try_backends(chain, *args, op_name="<name>")`
4. Add per-lambda comments explaining why each fallback exists
5. Add tests comparing against PyTorch reference using `compute_snr`
