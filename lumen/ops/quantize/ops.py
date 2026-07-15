###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Low-level quantization/dequantization ops wrapping Triton and C++ kernels.

These are pure stateless functions — no autograd, no scaling history.  For
autograd-aware quantized linear, see :mod:`~.linear`.  For the nn.Module
wrapper, see :mod:`lumen.modules.quantize`.
"""

import logging
import random
from typing import Optional, Tuple

import torch
import triton
from aiter.ops.quant import static_per_tensor_quant
from aiter.ops.triton._triton_kernels.quant.quant_fp8_blockwise import (
    quant_fp8_blockwise_for_act_grad_kernel,
    quant_fp8_blockwise_kernel,
    quant_fp8_blockwise_segment_m_kernel,
)
try:
    from aiter.ops.triton._triton_kernels.quant.quant_fp8_blockwise import (
        requant_fp8_row_to_col_kernel,
    )
    _HAVE_REQUANT_ROW_TO_COL = True
except ImportError:
    requant_fp8_row_to_col_kernel = None  # type: ignore[assignment]
    _HAVE_REQUANT_ROW_TO_COL = False
from aiter.ops.triton._triton_kernels.quant.quant_mxfp8 import (
    _convert_from_mxfp8_kernel,
    _convert_to_mxfp8_kernel,
)
from torch.library import triton_op, wrap_triton

logger = logging.getLogger(__name__)


def is_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


# ---------------------------------------------------------------------------
# Tensorwise Quantization
# ---------------------------------------------------------------------------


def quant_fp8_tensorwise_impl(x, scale, dtype):
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    static_per_tensor_quant(out, x, scale)
    return out


def dequant_fp8_tensorwise_impl(x, scale_inv, dtype):
    return x.to(dtype) * scale_inv


# ---------------------------------------------------------------------------
# Blockwise Quantization
# ---------------------------------------------------------------------------


@triton_op("lumen::quant_fp8_blockwise_impl", mutates_args=())
def quant_fp8_blockwise_impl(
    x: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor using blockwise FP8 scaling along *axis*.

    Returns ``(x_fp8, x_scales)`` where ``x_scales`` is per-block in float32.
    """
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    assert axis in (-2, -1, 0, 1), f"axis must be 0 or 1 (or -1, -2), got {axis}"
    axis = axis % 2

    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)
    scales_shape = (triton.cdiv(M, block_size), N) if axis == 0 else (M, triton.cdiv(N, block_size))
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    wrap_triton(quant_fp8_blockwise_kernel)[grid](
        x,
        x_fp8,
        x_scales,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
        axis,
    )
    return x_fp8, x_scales


@quant_fp8_blockwise_impl.register_fake
def quant_fp8_blockwise_impl_meta(
    x: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, "Input must be 2D"
    assert axis in (-2, -1, 0, 1), f"axis must be 0 or 1 (or -1, -2), got {axis}"
    axis = axis % 2
    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)
    scales_shape = (triton.cdiv(M, block_size), N) if axis == 0 else (M, triton.cdiv(N, block_size))
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)
    return x_fp8, x_scales


@triton_op("lumen::quant_fp8_blockwise_dual_axis_impl", mutates_args=())
def quant_fp8_blockwise_dual_axis_impl(
    x: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused dual-axis blockwise FP8 quant: emit row-wise (1×B) AND col-wise (B×1) in one pass.

    Loads each BLOCK×BLOCK tile of BF16 ``x`` once and writes both layouts —
    saves one BF16 read + amax pass vs calling ``quant_fp8_blockwise_impl``
    twice.  Used in the blockwise2d (Jet-RL §4.2) backward where the same
    grad tensor feeds both DGrad (1×128 along axis=1) and WGrad (128×1 along
    axis=0).

    Returns ``(x_fp8_row, x_scales_row, x_fp8_col, x_scales_col)`` with
    shapes ``(M, N)``, ``(M, N/B)``, ``(M, N)``, ``(M/B, N)``.  Scales are
    stored as dequant multipliers (``amax / FP8_MAX``), matching the
    convention of ``quant_fp8_blockwise_impl``.
    """
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    M, N = x.shape

    x_fp8_row = torch.empty((M, N), dtype=dtype, device=x.device)
    x_fp8_col = torch.empty((M, N), dtype=dtype, device=x.device)
    x_scales_row = torch.empty((M, triton.cdiv(N, block_size)), dtype=torch.float32, device=x.device)
    x_scales_col = torch.empty((triton.cdiv(M, block_size), N), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    wrap_triton(quant_fp8_blockwise_for_act_grad_kernel)[grid](
        x,
        x_fp8_row,
        x_scales_row,
        x_fp8_col,
        x_scales_col,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
    )
    return x_fp8_row, x_scales_row, x_fp8_col, x_scales_col


@quant_fp8_blockwise_dual_axis_impl.register_fake
def quant_fp8_blockwise_dual_axis_impl_meta(
    x: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, "Input must be 2D"
    M, N = x.shape
    return (
        torch.empty((M, N), dtype=dtype, device=x.device),
        torch.empty((M, triton.cdiv(N, block_size)), dtype=torch.float32, device=x.device),
        torch.empty((M, N), dtype=dtype, device=x.device),
        torch.empty((triton.cdiv(M, block_size), N), dtype=torch.float32, device=x.device),
    )


if _HAVE_REQUANT_ROW_TO_COL:
    @triton_op("lumen::requant_fp8_row_to_col", mutates_args=())
    def requant_fp8_row_to_col(
        x_fp8: torch.Tensor,
        x_scales: torch.Tensor,
        dtype: torch.dtype,
        block_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-quantize FP8 (row-wise 1×block) → FP8 (col-wise block×1) without BF16 roundtrip.

        ``x_fp8`` is (M, K) FP8 with row-wise 1×block_size scales ``x_scales`` (M, K//block_size).
        Returns ``(y_fp8, y_scales)`` where ``y_fp8`` is (M, K) FP8 and ``y_scales`` is
        (M//block_size, K) float32 col-wise dequant scales.  Used in the blockwise/blockwise2d
        WGrad backward to avoid a BF16 intermediate activation copy.
        """
        assert x_fp8.is_contiguous() and x_fp8.dim() == 2, "x_fp8 must be 2D and contiguous"
        M, K = x_fp8.shape
        y_fp8 = torch.empty((M, K), dtype=dtype, device=x_fp8.device)
        y_scales = torch.empty((triton.cdiv(M, block_size), K), dtype=torch.float32, device=x_fp8.device)
        grid = (triton.cdiv(M, block_size), triton.cdiv(K, block_size))
        wrap_triton(requant_fp8_row_to_col_kernel)[grid](
            x_fp8, x_scales, y_fp8, y_scales, M, K, block_size, torch.finfo(dtype).max,
        )
        return y_fp8, y_scales
else:
    def requant_fp8_row_to_col(x_fp8, x_scales, dtype, block_size=128):  # type: ignore[misc]
        raise RuntimeError(
            "requant_fp8_row_to_col_kernel not available in this aiter build; "
            "overlay third_party/aiter/aiter/ops/triton/_triton_kernels/quant/quant_fp8_blockwise.py"
        )


if _HAVE_REQUANT_ROW_TO_COL:
    @requant_fp8_row_to_col.register_fake
    def requant_fp8_row_to_col_meta(
        x_fp8: torch.Tensor,
        x_scales: torch.Tensor,
        dtype: torch.dtype,
        block_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x_fp8.dim() == 2, "x_fp8 must be 2D"
        M, K = x_fp8.shape
        return (
            torch.empty((M, K), dtype=dtype, device=x_fp8.device),
            torch.empty((triton.cdiv(M, block_size), K), dtype=torch.float32, device=x_fp8.device),
        )


def quant_fp8_blockwise_segment_m_impl(
    x: torch.Tensor,
    batch_size: int,
    seg_lens: torch.Tensor,
    seg_indptr: torch.Tensor,
    scales_seg_indptr: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
):
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)

    scales_shape = (triton.cdiv(M, block_size) + batch_size, N)
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, block_size) + seg_lens.shape[0], triton.cdiv(N, block_size))
    quant_fp8_blockwise_segment_m_kernel[grid](
        x,
        x_fp8,
        x_scales,
        N,
        batch_size,
        seg_indptr,
        scales_seg_indptr,
        block_size,
        torch.finfo(dtype).max,
    )
    return x_fp8, x_scales


# ---------------------------------------------------------------------------
# MXFP8 Conversion
# ---------------------------------------------------------------------------


@triton_op("lumen::convert_to_mxfp8", mutates_args={})
def convert_to_mxfp8(
    data_hp: torch.Tensor,
    block_size: int = 64,
    axis: int = -1,
    is_2d_block: bool = False,
    use_sr: bool = False,
    use_asm: Optional[bool] = None,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    float8_dtype_pt: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch._check(
        data_hp.shape[axis] % block_size == 0,
        f"tensor shape ({data_hp.shape}) at axis={axis} is not divisible by {block_size}",
    )
    assert not is_2d_block or data_hp.size(-2) % block_size == 0
    assert data_hp.dtype in [torch.float32, torch.bfloat16]
    if use_asm is None:
        use_asm = is_cdna4() and float8_dtype_pt == torch.float8_e4m3fn
    elif use_asm and float8_dtype_pt == torch.float8_e5m2:
        use_asm = False
        logger.warning(f"ASM mode doesn't support {float8_dtype_pt}, falling back to non-ASM implementation")

    data_hp = data_hp.transpose(axis, -1)
    data_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, data_shape[-1])
    data_lp = torch.empty(data_shape, dtype=float8_dtype_pt, device=data_hp.device).reshape(-1, data_shape[-1])

    if is_2d_block:
        scales_shape = (*data_shape[:-2], data_shape[-2] // block_size, data_shape[-1] // block_size)
    else:
        scales_shape = (*data_shape[:-1], data_shape[-1] // block_size)
    scales = torch.ones(scales_shape, dtype=torch.uint8, device=data_hp.device).reshape(-1, scales_shape[-1])
    stride_xm, stride_xn = data_hp.stride()
    stride_ym, stride_yn = data_lp.stride()
    stride_sm, stride_sn = scales.stride()
    M, N = data_hp.shape

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))

    assert M % block_size == 0, "tensor M shape must align to block size"
    assert N % block_size == 0, "tensor N shape must align to block size"

    BLOCK_M = 64 if M >= 64 else M
    BLOCK_N = 64 if N >= 64 else N
    BLOCK_M = block_size if BLOCK_M < block_size else BLOCK_M
    BLOCK_N = block_size if BLOCK_N < block_size else BLOCK_N

    if philox_seed is None:
        philox_seed = random.randint(0, 2**31 - 2)
    if philox_offset is None:
        philox_offset = random.randint(0, 2**31 - 2)
    wrap_triton(_convert_to_mxfp8_kernel)[grid](
        data_hp,
        data_lp,
        scales,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_SR=use_sr,
        USE_ASM=use_asm,
    )

    return data_lp.reshape(data_shape).transpose(axis, -1), scales.reshape(scales_shape).transpose(axis, -1)


@triton_op("lumen::convert_from_mxfp8", mutates_args={})
def convert_from_mxfp8(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = 64,
    axis: int = -1,
    is_2d_block: bool = False,
    use_asm: Optional[bool] = None,
) -> torch.Tensor:
    assert output_dtype in [torch.float32, torch.bfloat16]
    if use_asm is None:
        use_asm = is_cdna4() and data_lp.dtype == torch.float8_e4m3fn
    elif use_asm and data_lp.dtype == torch.float8_e5m2:
        use_asm = False
        logger.warning(f"ASM mode doesn't support {data_lp.dtype}, falling back to non-ASM implementation")

    data_lp = data_lp.transpose(axis, -1)
    scales = scales.transpose(axis, -1)
    orig_shape = data_lp.shape
    data_lp = data_lp.reshape(-1, orig_shape[-1])

    scales = scales.reshape(-1, orig_shape[-1] // block_size)
    data_hp = data_lp.new_empty(orig_shape, dtype=output_dtype).reshape(-1, orig_shape[-1])

    stride_xm, stride_xn = data_lp.stride()
    stride_ym, stride_yn = data_hp.stride()
    stride_sm, stride_sn = scales.stride()
    M, N = data_hp.shape

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))

    BLOCK_M = 64 if M >= 64 else M
    BLOCK_N = 64 if N >= 64 else N
    wrap_triton(_convert_from_mxfp8_kernel)[grid](
        data_lp,
        data_hp,
        scales,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_ASM=use_asm,
    )
    return data_hp.reshape(orig_shape).transpose(axis, -1)


@convert_to_mxfp8.register_fake
def _fake_convert_to_mxfp8(
    data_hp: torch.Tensor,
    block_size: int = 64,
    axis: int = -1,
    is_2d_block: bool = False,
    use_sr: bool = False,
    use_asm: Optional[bool] = None,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    float8_dtype_pt: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_hp = data_hp.transpose(axis, -1)
    orig_shape = data_hp.shape

    data_lp = data_hp.new_empty(data_hp.shape, dtype=float8_dtype_pt)
    if is_2d_block:
        scales_shape = (*orig_shape[:-2], orig_shape[-2] // block_size, orig_shape[-1] // block_size)
    else:
        scales_shape = (*orig_shape[:-1], orig_shape[-1] // block_size)

    scales = data_hp.new_empty(scales_shape, dtype=torch.uint8)
    return data_lp, scales.transpose(axis, -1)


@convert_from_mxfp8.register_fake
def _fake_convert_from_mxfp8(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = 64,
    axis: int = -1,
    is_2d_block: bool = False,
    use_asm: Optional[bool] = None,
) -> torch.Tensor:
    data_hp = data_lp.new_empty(data_lp.shape, dtype=output_dtype)
    return data_hp


# ---------------------------------------------------------------------------
# MXFP4 Conversion
# ---------------------------------------------------------------------------

_AITER_MXFP4_QUANT_AVAILABLE: Optional[bool] = None
_AITER_FP4_UTILS_AVAILABLE: Optional[bool] = None


def _probe_aiter_mxfp4_quant() -> bool:
    global _AITER_MXFP4_QUANT_AVAILABLE
    if _AITER_MXFP4_QUANT_AVAILABLE is not None:
        return _AITER_MXFP4_QUANT_AVAILABLE
    try:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant  # noqa: F401
        _AITER_MXFP4_QUANT_AVAILABLE = True
    except ImportError:
        _AITER_MXFP4_QUANT_AVAILABLE = False
    return _AITER_MXFP4_QUANT_AVAILABLE


def _probe_aiter_fp4_utils() -> bool:
    global _AITER_FP4_UTILS_AVAILABLE
    if _AITER_FP4_UTILS_AVAILABLE is not None:
        return _AITER_FP4_UTILS_AVAILABLE
    try:
        from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32  # noqa: F401
        _AITER_FP4_UTILS_AVAILABLE = True
    except ImportError:
        _AITER_FP4_UTILS_AVAILABLE = False
    return _AITER_FP4_UTILS_AVAILABLE


def convert_to_mxfp4(
    data_hp: torch.Tensor,
    block_size: int = 32,
    axis: int = -1,
    use_sr: bool = False,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert BF16/FP32 -> packed MXFP4 (uint8) + E8M0 scales (uint8).

    Uses AITER ``dynamic_mxfp4_quant`` for round-to-nearest (RTN) by default.
    Uses Lumen SR kernel when ``use_sr=True`` (for gradient quantization only —
    SR on forward tensors is detrimental per NVFP4 paper §4.4).

    Returns:
        (data_fp4, scales) — packed uint8 + uint8 E8M0 scales.
    """
    assert data_hp.dtype in (torch.float32, torch.bfloat16)

    if axis == 0 or axis == -2:
        data_hp = data_hp.transpose(-2, -1).contiguous()

    orig_shape = data_hp.shape
    data_2d = data_hp.reshape(-1, orig_shape[-1]).contiguous()
    M, N = data_2d.shape

    assert N % block_size == 0, f"N={N} not divisible by block_size={block_size}"
    assert N % 2 == 0, f"N={N} must be even for packing"

    use_asm = is_cdna4()

    if not use_sr and not use_asm and _probe_aiter_mxfp4_quant():
        # AITER RTN path (non-ASM fallback)
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        data_bf16 = data_2d.to(torch.bfloat16) if data_2d.dtype != torch.bfloat16 else data_2d
        fp4_packed, scales_e8m0 = dynamic_mxfp4_quant(data_bf16)
        fp4_packed = fp4_packed.view(torch.uint8)
        scales_e8m0 = scales_e8m0.view(torch.uint8)
        expected_scale_cols = N // block_size
        scales_e8m0 = scales_e8m0[:M, :expected_scale_cols].contiguous()
    else:
        # Lumen unified kernel: ASM (gfx950) or software fallback
        from lumen.kernels.mxfp4 import _convert_to_mxfp4_kernel

        if philox_seed is None:
            philox_seed = random.randint(0, 2**31 - 2)
        if philox_offset is None:
            philox_offset = random.randint(0, 2**31 - 2)

        fp4_packed = torch.empty((M, N // 2), dtype=torch.uint8, device=data_2d.device)
        scales_e8m0 = torch.empty((M, N // block_size), dtype=torch.uint8, device=data_2d.device)

        BLOCK_M = min(64, M) if M >= 64 else M
        BLOCK_N = min(64, N) if N >= 64 else N
        BLOCK_N = max(BLOCK_N, block_size)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _convert_to_mxfp4_kernel[grid](
            data_2d, fp4_packed, scales_e8m0,
            data_2d.stride(0), data_2d.stride(1),
            fp4_packed.stride(0), fp4_packed.stride(1),
            scales_e8m0.stride(0), scales_e8m0.stride(1),
            philox_seed, philox_offset,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            QUANT_BLOCK_SIZE=block_size,
            IS_2D_BLOCK=False,
            USE_SR=use_sr,
            USE_ASM=use_asm,
        )

    out_shape = (*orig_shape[:-1], N // 2)
    scale_shape = (*orig_shape[:-1], N // block_size)

    if axis == 0 or axis == -2:
        return fp4_packed.reshape(out_shape).transpose(-2, -1).contiguous(), \
               scales_e8m0.reshape(scale_shape).transpose(-2, -1).contiguous()

    return fp4_packed.reshape(out_shape), scales_e8m0.reshape(scale_shape)


def convert_to_mxfp4_2d(
    data_hp: torch.Tensor,
    block_size: int = 32,
    use_sr: bool = False,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert BF16/FP32 -> packed MXFP4 with 2D (block×block) tile scaling.

    Each block×block tile shares a single E8M0 scale. The quantized
    representation is transpose-invariant: transposing the packed data
    and the 2D scale grid produces the same quantized values, preserving
    the chain rule across forward and backward passes (NVFP4 paper §4.3).

    Uses the same ``_convert_to_mxfp4_kernel`` as 1D but with
    ``IS_2D_BLOCK=True``, which makes ``_calculate_fp4_scales`` compute
    per-tile (block×block) amax instead of per-row-block (1×block) amax.

    Returns:
        (data_fp4_packed, scales_2d) where scales_2d has shape
        ``(M//block, N//block)`` in uint8 E8M0 format.
    """
    assert data_hp.dtype in (torch.float32, torch.bfloat16)
    orig_shape = data_hp.shape
    data_2d = data_hp.reshape(-1, orig_shape[-1]).contiguous()
    M, N = data_2d.shape

    assert M % block_size == 0, f"M={M} not divisible by block_size={block_size}"
    assert N % block_size == 0, f"N={N} not divisible by block_size={block_size}"

    sm, sn = M // block_size, N // block_size
    use_asm = is_cdna4()

    from lumen.kernels.mxfp4 import _convert_to_mxfp4_kernel

    if philox_seed is None:
        philox_seed = random.randint(0, 2**31 - 2)
    if philox_offset is None:
        philox_offset = random.randint(0, 2**31 - 2)

    fp4_packed = torch.empty((M, N // 2), dtype=torch.uint8, device=data_2d.device)
    scales_2d = torch.empty((sm, sn), dtype=torch.uint8, device=data_2d.device)

    BLOCK_M = min(64, M) if M >= 64 else M
    BLOCK_N = min(64, N) if N >= 64 else N
    BLOCK_M = max(BLOCK_M, block_size)
    BLOCK_N = max(BLOCK_N, block_size)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _convert_to_mxfp4_kernel[grid](
        data_2d, fp4_packed, scales_2d,
        data_2d.stride(0), data_2d.stride(1),
        fp4_packed.stride(0), fp4_packed.stride(1),
        scales_2d.stride(0), scales_2d.stride(1),
        philox_seed, philox_offset,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        IS_2D_BLOCK=True,
        USE_SR=use_sr,
        USE_ASM=use_asm,
    )

    out_shape = (*orig_shape[:-1], N // 2)
    return fp4_packed.reshape(out_shape), scales_2d


def convert_from_mxfp4(
    data_fp4: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 32,
    axis: int = -1,
) -> torch.Tensor:
    """Convert packed MXFP4 + E8M0 scales -> BF16/FP32.

    Uses AITER ``mxfp4_to_f32`` + ``e8m0_to_f32`` when available.
    Falls back to a pure-Python dequant path otherwise.
    """
    assert output_dtype in (torch.float32, torch.bfloat16)

    if axis == 0 or axis == -2:
        data_fp4 = data_fp4.transpose(-2, -1).contiguous()
        scales = scales.transpose(-2, -1).contiguous()

    orig_packed_shape = data_fp4.shape
    data_flat = data_fp4.reshape(-1, orig_packed_shape[-1])
    scales_flat = scales.reshape(-1, scales.shape[-1])
    M, N_packed = data_flat.shape
    N = N_packed * 2

    if _probe_aiter_fp4_utils():
        from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32
        values = mxfp4_to_f32(data_flat.view(torch.uint8))
        scale_f32 = e8m0_to_f32(scales_flat.view(torch.uint8))
        scale_expanded = scale_f32.unsqueeze(-1).expand(
            M, N // block_size, block_size
        ).reshape(M, N)
        result = (values * scale_expanded).to(output_dtype)
    else:
        # Pure-Python fallback: unpack nibbles via lookup table
        _mxfp4_lut = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32, device=data_flat.device,
        )
        unpacked = data_flat.view(torch.uint8).repeat_interleave(2, dim=-1)
        unpacked[..., ::2] = unpacked[..., ::2] & 0xF
        unpacked[..., 1::2] = unpacked[..., 1::2] >> 4
        values = _mxfp4_lut[unpacked.long()]

        scale_f32 = torch.pow(2.0, scales_flat.view(torch.uint8).to(torch.float32) - 127.0)
        scale_expanded = scale_f32.unsqueeze(-1).expand(
            M, N // block_size, block_size
        ).reshape(M, N)
        result = (values * scale_expanded).to(output_dtype)

    out_shape = (*orig_packed_shape[:-1], N)
    if axis == 0 or axis == -2:
        return result.reshape(out_shape).transpose(-2, -1).contiguous()
    return result.reshape(out_shape)


def convert_from_mxfp4_2d(
    data_fp4: torch.Tensor,
    scales_2d: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 32,
) -> torch.Tensor:
    """Dequantize packed MXFP4 with 2D (block×block) E8M0 scales -> BF16/FP32.

    ``scales_2d`` has shape ``(M//block, N//block)`` — one E8M0 scale per tile.
    """
    assert output_dtype in (torch.float32, torch.bfloat16)

    orig_packed_shape = data_fp4.shape
    data_flat = data_fp4.reshape(-1, orig_packed_shape[-1])
    M, N_packed = data_flat.shape
    N = N_packed * 2

    sm, sn = scales_2d.shape[-2], scales_2d.shape[-1]
    assert sm == M // block_size and sn == N // block_size, (
        f"scales_2d shape {scales_2d.shape} doesn't match data shape ({M}, {N}) "
        f"with block_size={block_size}"
    )

    # Unpack nibbles
    _mxfp4_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32, device=data_flat.device,
    )
    unpacked = data_flat.view(torch.uint8).repeat_interleave(2, dim=-1)
    unpacked[..., ::2] = unpacked[..., ::2] & 0xF
    unpacked[..., 1::2] = unpacked[..., 1::2] >> 4
    values = _mxfp4_lut[unpacked.long()]  # (M, N)

    # E8M0 scales → float: 2^(stored_exp - 127)
    scale_f32 = torch.pow(
        2.0, scales_2d.view(torch.uint8).to(torch.float32) - 127.0
    )  # (sm, sn)
    # Expand 2D scales to per-element: (sm, 1, sn, 1) → (sm, block, sn, block) → (M, N)
    scale_expanded = (
        scale_f32.view(sm, 1, sn, 1)
        .expand(sm, block_size, sn, block_size)
        .reshape(M, N)
    )
    result = (values * scale_expanded).to(output_dtype)

    out_shape = (*orig_packed_shape[:-1], N)
    return result.reshape(out_shape)


def convert_to_mxfp4_dual_axis(
    data_hp: torch.Tensor,
    block_size: int = 32,
    use_sr: bool = True,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Emit both axis=1 (1x32) and axis=0 (32x1) MXFP4 quantizations.

    Uses different PRNG offsets for the two axes to avoid correlation.

    Returns:
        (row_fp4, row_scales, col_fp4, col_scales)
    """
    if philox_seed is None:
        philox_seed = random.randint(0, 2**31 - 2)
    if philox_offset is None:
        philox_offset = random.randint(0, 2**31 - 2)

    row_fp4, row_scales = convert_to_mxfp4(
        data_hp, block_size=block_size, axis=-1,
        use_sr=use_sr, philox_seed=philox_seed, philox_offset=philox_offset,
    )
    col_fp4, col_scales = convert_to_mxfp4(
        data_hp, block_size=block_size, axis=0,
        use_sr=use_sr, philox_seed=philox_seed + 1, philox_offset=philox_offset,
    )
    return row_fp4, row_scales, col_fp4, col_scales


def transpose_packed_fp4(data_fp4: torch.Tensor) -> torch.Tensor:
    """Transpose a packed MXFP4 matrix: (M, N//2) -> (N, M//2).

    Uses Lumen Triton kernel (no AITER equivalent).
    """
    from lumen.kernels.mxfp4 import _transpose_packed_fp4_kernel

    M, N_packed = data_fp4.shape
    N = N_packed * 2
    assert M % 2 == 0, f"M={M} must be even for packed transpose"

    output = torch.empty((N, M // 2), dtype=torch.uint8, device=data_fp4.device)

    BLOCK_M = min(32, M)
    BLOCK_N_PACKED = min(16, N_packed)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_packed, BLOCK_N_PACKED))
    _transpose_packed_fp4_kernel[grid](
        data_fp4, output,
        M, N_packed,
        data_fp4.stride(0), data_fp4.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N_PACKED=BLOCK_N_PACKED,
    )
    return output


def hadamard_transform(
    x: torch.Tensor,
    sign_vector: torch.Tensor,
    g: int = 64,
) -> torch.Tensor:
    """Apply blockwise Random Hadamard Transform.

    Applies H_g @ diag(S) to blocks of g elements along the last dimension.
    Uses Lumen Triton kernel (no AITER equivalent).
    """
    from lumen.kernels.mxfp4 import _hadamard_transform_kernel

    assert x.shape[-1] % g == 0, f"N={x.shape[-1]} not divisible by g={g}"
    assert (g & (g - 1)) == 0, f"g={g} must be a power of 2"

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x_2d.shape
    output = torch.empty_like(x_2d)

    grid = (M, N // g)
    _hadamard_transform_kernel[grid](
        x_2d, output, sign_vector,
        M, N,
        x_2d.stride(0), x_2d.stride(1),
        output.stride(0), output.stride(1),
        G=g,
    )
    return output.reshape(orig_shape)
