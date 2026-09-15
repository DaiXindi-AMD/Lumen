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

import functools
import logging
import os
import random
from typing import Optional, Tuple

import torch
import triton
from aiter.ops.quant import static_per_tensor_quant
try:
    from aiter.ops.triton._triton_kernels.quant.quant_fp8_blockwise import (
        quant_fp8_blockwise_for_act_grad_kernel,
        quant_fp8_blockwise_kernel,
        quant_fp8_blockwise_segment_m_kernel,
    )
except (ImportError, ModuleNotFoundError):
    quant_fp8_blockwise_for_act_grad_kernel = None  # type: ignore[assignment]
    quant_fp8_blockwise_kernel = None  # type: ignore[assignment]
    quant_fp8_blockwise_segment_m_kernel = None  # type: ignore[assignment]
try:
    from aiter.ops.triton._triton_kernels.quant.quant_fp8_blockwise import (
        requant_fp8_row_to_col_kernel,
    )
    _HAVE_REQUANT_ROW_TO_COL = True
except (ImportError, ModuleNotFoundError):
    requant_fp8_row_to_col_kernel = None  # type: ignore[assignment]
    _HAVE_REQUANT_ROW_TO_COL = False
try:
    from aiter.ops.triton._triton_kernels.quant.quant_mxfp8 import (
        _convert_from_mxfp8_kernel,
        _convert_to_mxfp8_kernel,
    )
except (ImportError, ModuleNotFoundError):
    _convert_from_mxfp8_kernel = None  # type: ignore[assignment]
    _convert_to_mxfp8_kernel = None  # type: ignore[assignment]
from torch.library import triton_op, wrap_triton

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def _triton_target(device: int):
    """Triton's target for one device, asked once.

    ``get_current_target()`` queries the HIP runtime for the device properties
    on every call. The quantize path asks per launch, and a training step makes
    hundreds of those, which is dead CPU time in front of kernels the GPU is
    already waiting for. A device's architecture cannot change under a live
    process, so the answer is cacheable; keying on the device keeps it right for
    a process that switches between unlike GPUs.
    """
    return triton.runtime.driver.active.get_current_target()


def is_cdna4():
    target = _triton_target(torch.cuda.current_device())
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


def triton_arch() -> str:
    """Architecture name of the current device, e.g. ``gfx950``.

    Same answer as AITER's ``get_arch()``, without its per-call device query.
    """
    target = _triton_target(torch.cuda.current_device())
    return "" if target is None else target.arch


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
# MXFP4 Conversion (implemented in AITER)
# ---------------------------------------------------------------------------

# Preserve the old Lumen tuning knob while ownership moves into AITER.  The
# AITER-prefixed name wins when both are set.
if "LUMEN_SR_PHILOX_ROUNDS" in os.environ:
    os.environ.setdefault(
        "AITER_MXFP4_SR_PHILOX_ROUNDS",
        os.environ["LUMEN_SR_PHILOX_ROUNDS"],
    )

# The RHT-bearing APIs below must stay distinct from plain convert_to_mxfp4:
# they perform the original training-path rotation inside AITER before packing.
from aiter.ops.mxfp4 import (  # noqa: E402
    convert_from_mxfp4,
    convert_from_mxfp4_2d,
    convert_to_mxfp4,
    convert_to_mxfp4_2d,
    dequant_hadamard_quant_mxfp4,
    dequant_transpose_mxfp4,
    dual_layout_quant_mxfp4,
    hadamard_quant_mxfp4,
    hadamard_transform,
    mxfp4_data_shuffle_supported,
    mxfp4_scale_swizzle_supported,
    swizzle_expanded_mxfp4_scale,
    swizzle_mxfp4_scale,
    transpose_packed_fp4,
)


def convert_to_mxfp4_dual_axis(
    data_hp: torch.Tensor,
    block_size: int = 32,
    use_sr: bool = True,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose Lumen's row-wise and column-wise compatibility outputs.

    This is two calls to AITER's one-axis conversion API, not an AITER kernel.
    Keep the composition here so AITER's public surface contains only reusable
    operators.  It is also deliberately separate from
    ``dual_layout_quant_mxfp4``, whose second output includes Lumen's H16 RHT.
    """
    # Preserve the original Lumen API's seeded stream split exactly.  This
    # helper predates the AITER migration and callers may use explicit seeds
    # when comparing the two axis layouts.
    if philox_seed is None:
        philox_seed = random.randint(0, 2**31 - 2)
    if philox_offset is None:
        philox_offset = random.randint(0, 2**31 - 2)

    row_fp4, row_scales = convert_to_mxfp4(
        data_hp,
        block_size=block_size,
        axis=-1,
        use_sr=use_sr,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
    )
    col_fp4, col_scales = convert_to_mxfp4(
        data_hp,
        block_size=block_size,
        axis=0,
        use_sr=use_sr,
        philox_seed=philox_seed + 1,
        philox_offset=philox_offset,
    )
    return row_fp4, row_scales, col_fp4, col_scales
