# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for Lumen quantization ops, comparing against torchao reference implementation."""

import csv
import json
import os

import pytest
import torch
import triton
from triton.compiler.errors import CompilationError
from conftest import compute_snr
from torchao.kernel.blockwise_quantization import fp8_blockwise_act_quant
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import (
    MXTensor,
)
from torchao.prototype.mx_formats.mx_tensor import to_dtype as torchao_to_dtype
from torchao.prototype.mx_formats.mx_tensor import to_mx as torchao_to_mx
from torchao.quantization.quant_primitives import (
    _dequantize_affine_float8,
    _quantize_affine_float8,
)

from lumen.ops.quantize import (
    convert_from_mxfp4,
    convert_from_mxfp4_2d,
    convert_from_mxfp8,
    convert_to_mxfp4,
    convert_to_mxfp4_2d,
    convert_to_mxfp4_dual_axis,
    convert_to_mxfp8,
    dequant_fp8_tensorwise_impl,
    dequant_hadamard_quant_mxfp4,
    dequant_transpose_mxfp4,
    dual_layout_quant_mxfp4,
    hadamard_quant_mxfp4,
    hadamard_transform,
    is_cdna4,
    quant_fp8_blockwise_impl,
    quant_fp8_tensorwise_impl,
    swizzle_mxfp4_scale,
    transpose_packed_fp4,
)
from lumen.ops.quantize import flydsl_mxfp4, mxfp4_asm, mxfp4_autotune
from lumen.ops.quantize.linear import (
    _MXFP4_ASM_ARCHS,
    _MXFP4_SCALE_SHUFFLE_TILING,
    _expand_2d_scale_to_1d,
    _gemm_mxfp4_aiter,
    _gemm_mxfp4_aiter_asm,
    _gemm_mxfp4_aiter_preshuffle,
    _gemm_mxfp4_flydsl,
    _mxfp4_asm_eligible,
    _mxfp4_asm_supported,
    _mxfp4_asm_tuned,
    _mxfp4_choose_backend,
    _mxfp4_preshuffle_eligible,
    _mxfp4_preshuffle_supported,
    _MXFP4_WIDE_SHUFFLE_MIN_BYTES,
    _pad_and_swizzle_mxfp4_scale,
    _shuffle_mxfp4_weight,
    gemm_mxfp4_dispatch,
)

# ---------------------------------------------------------------------------
# Tensorwise FP8
# ---------------------------------------------------------------------------

SHAPES = [(64, 128), (128, 256), (256, 512)]
SHAPE_IDS = [f"{m}x{n}" for m, n in SHAPES]


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype_in", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_quant_fp8_tensorwise_vs_torchao(shape, dtype_in, fp8_dtype):
    """Compare Lumen tensorwise quant against torchao _quantize_affine_float8."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(*shape, device="cuda", dtype=dtype_in)
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.abs().max().float().clamp(min=1e-6)
    scale_torchao = amax / fp8_max
    scale_lumen = scale_torchao

    try:
        x_fp8_lumen = quant_fp8_tensorwise_impl(x, scale_lumen, fp8_dtype)
    except AttributeError as e:
        if "_static_per_tensor_quant_cuda" in str(e):
            pytest.skip(f"AITER HIP tensorwise quant unavailable (JIT rebuild needed): {e}")
        raise
    x_fp8_torchao = _quantize_affine_float8(x, scale_torchao, fp8_dtype)

    torch.testing.assert_close(
        x_fp8_lumen.float(),
        x_fp8_torchao.float(),
        atol=1e-2,
        rtol=1e-2,
        msg="FP8 quant outputs should match",
    )

    x_deq_lumen = _dequantize_affine_float8(x_fp8_lumen, scale_torchao, torch.float32)
    x_deq_torchao = _dequantize_affine_float8(x_fp8_torchao, scale_torchao, torch.float32)
    snr = compute_snr(x.float(), x_deq_lumen)
    assert snr >= 8.0, f"SNR {snr:.1f} dB too low"
    torch.testing.assert_close(x_deq_lumen, x_deq_torchao, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype_in", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_dequant_fp8_tensorwise_vs_torchao(shape, dtype_in, fp8_dtype):
    """Quantize with torchao, dequant with both Lumen and torchao, compare."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(*shape, device="cuda", dtype=dtype_in)
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.abs().max().float().clamp(min=1e-6)
    scale = amax / fp8_max

    x_fp8 = _quantize_affine_float8(x, scale, fp8_dtype)
    x_deq_lumen = dequant_fp8_tensorwise_impl(x_fp8, scale, dtype_in)
    x_deq_torchao = _dequantize_affine_float8(x_fp8, scale, dtype_in)

    torch.testing.assert_close(
        x_deq_lumen.float(),
        x_deq_torchao.float(),
        atol=1e-2,
        rtol=1e-2,
        msg="Dequant outputs should match",
    )
    snr = compute_snr(x.float(), x_deq_lumen.float())
    assert snr >= 8.0, f"SNR {snr:.1f} dB too low"


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_quant_fp8_tensorwise_zeros(shape, fp8_dtype):
    """Both implementations should map zeros to zero."""
    x = torch.zeros(*shape, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    try:
        x_fp8_lumen = quant_fp8_tensorwise_impl(x, scale, fp8_dtype)
    except AttributeError as e:
        if "_static_per_tensor_quant_cuda" in str(e):
            pytest.skip(f"AITER HIP tensorwise quant unavailable (JIT rebuild needed): {e}")
        raise
    x_fp8_torchao = _quantize_affine_float8(x, scale, fp8_dtype)

    torch.testing.assert_close(x_fp8_lumen.float(), x_fp8_torchao.float())
    assert (x_fp8_lumen == 0).all()
    assert (x_fp8_torchao == 0).all()


# ---------------------------------------------------------------------------
# Blockwise FP8
# ---------------------------------------------------------------------------

BLOCK_SIZE = 128


def _blockwise_quant_ref(x, block_size, fp8_dtype):
    """Pure PyTorch blockwise FP8 quantization reference (axis=1 only)."""
    M, N = x.shape
    fp8_max = torch.finfo(fp8_dtype).max
    x_f32 = x.float()
    x_blocked = x_f32.reshape(M, N // block_size, block_size)
    amax = x_blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scales = amax / fp8_max  # (M, N//block_size, 1)
    x_scaled = (x_blocked / scales).clamp(-fp8_max, fp8_max)
    x_fp8 = x_scaled.reshape(M, N).to(fp8_dtype)
    scales = scales.squeeze(-1)  # (M, N//block_size)
    return x_fp8, scales


@pytest.mark.parametrize("shape", [(128, 256), (256, 512)], ids=["128x256", "256x512"])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_quant_fp8_blockwise_vs_torchao(shape, fp8_dtype):
    """Compare Lumen blockwise (axis=1) against torchao or PyTorch reference."""
    M, N = shape
    if N % BLOCK_SIZE != 0:
        pytest.skip(f"N={N} not divisible by block_size={BLOCK_SIZE}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    x_fp8_lumen, scales_lumen = quant_fp8_blockwise_impl(x, fp8_dtype, axis=1, block_size=BLOCK_SIZE)
    try:
        x_fp8_ref, scales_ref = fp8_blockwise_act_quant(x, BLOCK_SIZE, fp8_dtype)
    except (AssertionError, RuntimeError):
        x_fp8_ref, scales_ref = _blockwise_quant_ref(x, BLOCK_SIZE, fp8_dtype)

    x_deq_lumen = _dequantize_affine_float8(x_fp8_lumen, scales_lumen, torch.float32)
    x_deq_ref = _dequantize_affine_float8(x_fp8_ref, scales_ref, torch.float32)

    snr_lumen = compute_snr(x.float(), x_deq_lumen)
    snr_ref = compute_snr(x.float(), x_deq_ref)
    # e5m2 has lower precision (2 mantissa bits) → lower SNR expected
    snr_floor = 4.0 if fp8_dtype == torch.float8_e5m2 else 8.0
    assert snr_lumen >= snr_floor, f"Lumen SNR {snr_lumen:.1f} dB too low"
    assert snr_ref >= snr_floor, f"Reference SNR {snr_ref:.1f} dB too low"
    tol = 0.5 if fp8_dtype == torch.float8_e5m2 else 1e-1
    torch.testing.assert_close(x_deq_lumen, x_deq_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("shape", [(256, 128), (512, 256)], ids=["256x128", "512x256"])
def test_quant_fp8_blockwise_axis0(shape):
    """Self-roundtrip for axis=0 (no torchao equivalent)."""
    M, N = shape
    if M % BLOCK_SIZE != 0:
        pytest.skip(f"M={M} not divisible by block_size={BLOCK_SIZE}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    x_fp8, scales = quant_fp8_blockwise_impl(x, torch.float8_e4m3fn, axis=0, block_size=BLOCK_SIZE)
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scales.shape == (triton.cdiv(M, BLOCK_SIZE), N)

    x_deq = _dequantize_affine_float8(x_fp8, scales, torch.float32)
    snr = compute_snr(x.float(), x_deq)
    assert snr >= 8.0, f"SNR {snr:.1f} dB too low"


# ---------------------------------------------------------------------------
# MXFP8
# ---------------------------------------------------------------------------

MX_BLOCK_SIZES = [32, 64]
MX_SHAPES = [(64, 128), (128, 256)]


@pytest.mark.parametrize("shape", MX_SHAPES, ids=[f"{m}x{n}" for m, n in MX_SHAPES])
@pytest.mark.parametrize("block_size", MX_BLOCK_SIZES)
def test_mxfp8_vs_torchao(shape, block_size):
    """Compare Lumen MXFP8 quant outputs against torchao, then cross-dequant."""
    M, N = shape
    if N % block_size != 0:
        pytest.skip(f"N={N} not divisible by block_size={block_size}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    data_lp_lumen, scales_lumen = convert_to_mxfp8(
        x.float(),
        block_size=block_size,
        axis=-1,
        float8_dtype_pt=torch.float8_e4m3fn,
        philox_seed=42,
        philox_offset=0,
    )

    scale_ref, data_lp_ref = torchao_to_mx(
        x.float().cpu().contiguous(),
        torch.float8_e4m3fn,
        block_size,
        scaling_mode=ScaleCalculationMode.EVEN,
    )

    # Compare quantized FP8 data
    lumen_flat = data_lp_lumen.cpu().flatten().view(torch.float8_e4m3fn).view(torch.uint8)
    ref_flat = data_lp_ref.flatten().view(torch.uint8)
    assert (
        lumen_flat.numel() == ref_flat.numel()
    ), f"FP8 data size mismatch: Lumen {lumen_flat.numel()} vs torchao {ref_flat.numel()}"
    fp8_match = (lumen_flat == ref_flat).float().mean().item()
    assert fp8_match >= 0.95, f"FP8 data match rate {fp8_match:.2%} < 95%"

    # Compare scales (torchao returns float8_e8m0fnu, Lumen returns uint8; bitwise reinterpret)
    s_lumen = scales_lumen.cpu().flatten()
    s_ref = scale_ref.flatten().view(torch.uint8)
    assert s_lumen.numel() == s_ref.numel(), f"Scale size mismatch: Lumen {s_lumen.numel()} vs torchao {s_ref.numel()}"
    scale_match = (s_lumen == s_ref).float().mean().item()
    assert scale_match >= 0.95, f"Scale match rate {scale_match:.2%} < 95%"

    # Cross-dequant: Lumen quant → torchao dequant
    _e8m0 = scale_ref.dtype  # float8_e8m0fnu
    x_deq_lumen_cpu = torchao_to_dtype(
        data_lp_lumen.cpu(),
        scales_lumen.cpu().view(_e8m0),
        torch.float8_e4m3fn,
        block_size,
        torch.float32,
    )

    # MXFP8 uses block scaling + E8M0 exponent-only scales → lower SNR than per-tensor
    snr = compute_snr(x.float().cpu(), x_deq_lumen_cpu)
    assert snr >= 6.0, f"SNR {snr:.1f} dB too low"
    assert not torch.isnan(x_deq_lumen_cpu).any()
    assert not torch.isinf(x_deq_lumen_cpu).any()


@pytest.mark.parametrize("shape", MX_SHAPES, ids=[f"{m}x{n}" for m, n in MX_SHAPES])
@pytest.mark.parametrize("block_size", MX_BLOCK_SIZES)
def test_mxfp8_scale_and_data_agreement_with_torchao(shape, block_size):
    """Verify Lumen and torchao produce matching scales (>95%) and FP8 data (>90%)."""
    M, N = shape
    if N % block_size != 0:
        pytest.skip(f"N={N} not divisible by block_size={block_size}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    data_lp_lumen, scales_lumen = convert_to_mxfp8(
        x.float(),
        block_size=block_size,
        axis=-1,
        float8_dtype_pt=torch.float8_e4m3fn,
        philox_seed=42,
        philox_offset=0,
    )
    scale_ref, data_lp_ref = torchao_to_mx(
        x.float().cpu().contiguous(),
        torch.float8_e4m3fn,
        block_size,
        scaling_mode=ScaleCalculationMode.EVEN,
    )

    # Scales agreement (torchao returns float8_e8m0fnu; bitwise reinterpret to uint8)
    s_lumen = scales_lumen.cpu().flatten()
    s_ref = scale_ref.flatten().view(torch.uint8)
    assert s_lumen.numel() == s_ref.numel(), f"Scale size mismatch: Lumen {s_lumen.numel()} vs torchao {s_ref.numel()}"
    scale_match = (s_lumen == s_ref).float().mean().item()
    assert scale_match >= 0.95, f"Scale match rate {scale_match:.2%} < 95%"

    # FP8 data agreement
    d_lumen = data_lp_lumen.cpu().flatten().view(torch.float8_e4m3fn).view(torch.uint8)
    d_ref = data_lp_ref.flatten().view(torch.uint8)
    assert (
        d_lumen.numel() == d_ref.numel()
    ), f"FP8 data size mismatch: Lumen {d_lumen.numel()} vs torchao {d_ref.numel()}"
    data_match = (d_lumen == d_ref).float().mean().item()
    assert data_match >= 0.90, f"FP8 data match rate {data_match:.2%} < 90%"


@pytest.mark.parametrize("shape", MX_SHAPES, ids=[f"{m}x{n}" for m, n in MX_SHAPES])
@pytest.mark.parametrize("block_size", MX_BLOCK_SIZES)
def test_mxfp8_vs_torchao_mxtensor(shape, block_size):
    """Compare Lumen quantized tensors AND roundtrip vs torchao MXTensor API."""
    M, N = shape
    if N % block_size != 0:
        pytest.skip(f"N={N} not divisible by block_size={block_size}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    data_lp_lumen, scales_lumen = convert_to_mxfp8(
        x.float(),
        block_size=block_size,
        axis=-1,
        float8_dtype_pt=torch.float8_e4m3fn,
        philox_seed=42,
        philox_offset=0,
    )

    mx_ref = MXTensor.to_mx(
        x.float().cpu().contiguous(),
        torch.float8_e4m3fn,
        block_size,
        scaling_mode=ScaleCalculationMode.EVEN,
    )

    # Compare quantized FP8 data directly
    data_lp_ref = mx_ref.qdata.flatten()
    data_lp_lumen_flat = data_lp_lumen.cpu().flatten().view(torch.float8_e4m3fn)
    assert (
        data_lp_lumen_flat.numel() == data_lp_ref.numel()
    ), f"FP8 data size mismatch: Lumen {data_lp_lumen_flat.numel()} vs torchao {data_lp_ref.numel()}"
    fp8_match = (data_lp_lumen_flat.view(torch.uint8) == data_lp_ref.view(torch.uint8)).float().mean().item()
    assert fp8_match >= 0.95, f"FP8 data match rate {fp8_match:.2%} < 95%"

    # Compare E8M0 scales directly (bitwise reinterpret float8_e8m0fnu → uint8)
    scales_ref = mx_ref.scale.flatten().view(torch.uint8)
    scales_lumen_flat = scales_lumen.cpu().flatten()
    assert (
        scales_lumen_flat.numel() == scales_ref.numel()
    ), f"Scale size mismatch: Lumen {scales_lumen_flat.numel()} vs torchao {scales_ref.numel()}"
    scale_match = (scales_lumen_flat == scales_ref).float().mean().item()
    assert scale_match >= 0.95, f"Scale match rate {scale_match:.2%} < 95%"

    # Compare dequantized results
    x_deq_lumen = convert_from_mxfp8(
        data_lp_lumen,
        scales_lumen,
        output_dtype=torch.float32,
        block_size=block_size,
        axis=-1,
    )
    x_deq_torchao = mx_ref.dequantize()

    # MXFP8 uses block scaling + E8M0 exponent-only scales → lower SNR than per-tensor
    snr = compute_snr(x.float().cpu(), x_deq_lumen.cpu())
    assert snr >= 6.0, f"SNR {snr:.1f} dB too low"
    torch.testing.assert_close(
        x_deq_lumen.cpu(),
        x_deq_torchao.cpu(),
        atol=1e-1,
        rtol=1e-1,
    )


def test_mxfp8_zeros():
    """Both implementations should handle zeros."""
    M, N = 64, 128
    block_size = 64
    x = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    data_lp, scales = convert_to_mxfp8(
        x.float(),
        block_size=block_size,
        axis=-1,
        float8_dtype_pt=torch.float8_e4m3fn,
    )
    x_deq = convert_from_mxfp8(data_lp, scales, block_size=block_size, axis=-1)
    torch.testing.assert_close(x_deq, x.float())

    scale_ref, data_ref = torchao_to_mx(
        x.float().cpu(),
        torch.float8_e4m3fn,
        block_size,
        scaling_mode=ScaleCalculationMode.EVEN,
    )
    x_deq_ref = torchao_to_dtype(
        data_ref,
        scale_ref,
        torch.float8_e4m3fn,
        block_size,
        torch.float32,
    )
    torch.testing.assert_close(x_deq_ref, x.float().cpu())


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("shape", MX_SHAPES, ids=[f"{m}x{n}" for m, n in MX_SHAPES])
@pytest.mark.parametrize("block_size", MX_BLOCK_SIZES)
def test_mxfp8_dtype_variants(fp8_dtype, shape, block_size):
    """Test MXFP8 with different FP8 element dtypes, compared against torchao."""
    M, N = shape
    if N % block_size != 0:
        pytest.skip(f"N={N} not divisible by block_size={block_size}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    data_lp, scales = convert_to_mxfp8(
        x.float(),
        block_size=block_size,
        axis=-1,
        float8_dtype_pt=fp8_dtype,
        philox_seed=42,
        philox_offset=0,
    )

    scale_ref, data_lp_ref = torchao_to_mx(
        x.float().cpu().contiguous(),
        fp8_dtype,
        block_size,
        scaling_mode=ScaleCalculationMode.EVEN,
    )

    # Compare FP8 data
    d_lumen = data_lp.cpu().flatten().view(fp8_dtype).view(torch.uint8)
    d_ref = data_lp_ref.flatten().view(torch.uint8)
    assert (
        d_lumen.numel() == d_ref.numel()
    ), f"FP8 data size mismatch: Lumen {d_lumen.numel()} vs torchao {d_ref.numel()}"
    data_match = (d_lumen == d_ref).float().mean().item()
    assert data_match >= 0.95, f"FP8 data match rate {data_match:.2%} < 95%"

    # Compare scales (torchao returns float8_e8m0fnu; bitwise reinterpret to uint8)
    s_lumen = scales.cpu().flatten()
    s_ref = scale_ref.flatten().view(torch.uint8)
    assert s_lumen.numel() == s_ref.numel(), f"Scale size mismatch: Lumen {s_lumen.numel()} vs torchao {s_ref.numel()}"
    scale_match = (s_lumen == s_ref).float().mean().item()
    assert scale_match >= 0.95, f"Scale match rate {scale_match:.2%} < 95%"

    # Compare dequantized results
    x_deq = convert_from_mxfp8(data_lp, scales, block_size=block_size, axis=-1)
    x_deq_ref = torchao_to_dtype(
        data_lp_ref,
        scale_ref,
        fp8_dtype,
        block_size,
        torch.float32,
    )

    assert not torch.isnan(x_deq).any()
    assert not torch.isinf(x_deq).any()
    # MXFP8 uses block scaling + E8M0 exponent-only scales → lower SNR than per-tensor
    snr = compute_snr(x.float().cpu(), x_deq.cpu())
    assert snr >= 6.0, f"Lumen roundtrip SNR {snr:.1f} dB too low"
    torch.testing.assert_close(
        x_deq.cpu(),
        x_deq_ref.cpu(),
        atol=1e-1,
        rtol=1e-1,
    )


# ---------------------------------------------------------------------------
# MXFP4
# ---------------------------------------------------------------------------

MXFP4_BLOCK_SIZE = 32
MXFP4_SHAPES = [(64, 128), (128, 256)]

# Every shape above has a row count that is a multiple of 64, the tile height
# the quantize kernels pick for anything that large. The kernels address their
# tiles without bounds masks, so a row count that is not a multiple of the tile
# used to leave the last program reading past the input and writing past the
# scale tensor -- silently, and only visible as a scale that changed between two
# identical calls.
MXFP4_UNALIGNED_ROWS = [96, 160, 224]


def _poison_free_blocks():
    """Recycle the allocator's free blocks with non-zero bytes.

    An unwritten output is only detectable if the memory it lands on differs
    between the two calls, so it has to be dirtied in between.
    """
    for _ in range(3):
        junk = torch.full((8 << 20,), 0xAB, dtype=torch.uint8, device="cuda")
        del junk


def _require_mxfp4_dtype():
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("torch.float4_e2m1fn_x2 unavailable in this PyTorch build")


def _torchao_mxfp4_dequant(data_fp4, scales, block_size=MXFP4_BLOCK_SIZE):
    return torchao_to_dtype(
        data_fp4.cpu().contiguous(),
        scales.cpu().contiguous().view(torch.float8_e8m0fnu),
        torch.float4_e2m1fn_x2,
        block_size,
        torch.float32,
    )


@pytest.mark.parametrize(
    "dim,cap,floor,expected",
    [
        (8192, 64, 1, 64),
        (4096, 64, 32, 64),
        (12288, 64, 32, 64),
        (8192, 256, 32, 256),
        (8192, 128, 32, 128),
        (96, 64, 32, 32),
        (224, 64, 32, 32),
        (300, 64, 1, 4),
    ],
)
def test_mxfp4_dividing_block_keeps_full_tile_on_aligned_shapes(dim, cap, floor, expected):
    """The tile only shrinks for shapes it would otherwise overrun.

    Every training shape is a multiple of the cap, so shrinking must not cost
    them anything -- that is what makes fitting the tile preferable to masking
    every load and store in the kernel.
    """
    from lumen.ops.quantize.ops import _dividing_block

    block = _dividing_block(dim, cap, floor)
    assert block == expected
    assert dim % block == 0


@pytest.mark.parametrize("M", MXFP4_UNALIGNED_ROWS)
def test_mxfp4_1d_rtn_unaligned_rows_vs_torchao(M):
    """A row count that does not divide the kernel tile must still be exact."""
    _require_mxfp4_dtype()
    N = 256
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    try:
        data_fp4, scales = convert_to_mxfp4(
            x.float(), block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False,
        )
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 RTN quant unavailable on this hardware/build: {e}")

    mx_ref = MXTensor.to_mx(
        x.float().cpu().contiguous(),
        torch.float4_e2m1fn_x2,
        MXFP4_BLOCK_SIZE,
        scaling_mode=ScaleCalculationMode.EVEN,
    )
    torch.testing.assert_close(scales.cpu(), mx_ref.scale.view(torch.uint8), atol=0, rtol=0)
    torch.testing.assert_close(data_fp4.cpu(), mx_ref.qdata.view(torch.uint8), atol=0, rtol=0)


@pytest.mark.parametrize("M", MXFP4_UNALIGNED_ROWS)
def test_mxfp4_quant_unaligned_rows_are_reproducible(M):
    """Two identical quantize calls must agree byte for byte on every path.

    Covers the paths torchAO has no counterpart for -- 2D tiles, fused Hadamard,
    dual layout -- where writing past the output shows up only as a result that
    moves when the allocator hands back different memory.
    """
    N = 256
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    sign = torch.randint(0, 2, (N // 16, 16), device="cuda", dtype=torch.bfloat16) * 2 - 1

    paths = {
        "1d": lambda: convert_to_mxfp4(x.float(), block_size=MXFP4_BLOCK_SIZE, use_sr=False),
        "2d": lambda: convert_to_mxfp4_2d(x.float(), block_size=MXFP4_BLOCK_SIZE),
        "hadamard": lambda: hadamard_quant_mxfp4(
            x, sign, block_size=MXFP4_BLOCK_SIZE, use_sr=False,
        ),
        "dual_layout": lambda: dual_layout_quant_mxfp4(
            x, sign, block_size=MXFP4_BLOCK_SIZE,
            use_sr_row=False, use_sr_transposed=False,
        ),
    }

    for name, run in paths.items():
        _poison_free_blocks()
        first = tuple(t.clone() for t in run())
        _poison_free_blocks()
        second = run()
        for i, (a, b) in enumerate(zip(first, second)):
            assert torch.equal(a, b), (
                f"{name} output {i} changed between two identical calls at M={M}"
            )


@pytest.mark.parametrize("shape", MXFP4_SHAPES, ids=[f"{m}x{n}" for m, n in MXFP4_SHAPES])
def test_mxfp4_1d_rtn_vs_torchao_mxtensor(shape):
    """Compare Lumen 1x32 MXFP4 RTN quant/dequant against TorchAO MXTensor."""
    _require_mxfp4_dtype()
    M, N = shape
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    try:
        data_fp4, scales = convert_to_mxfp4(
            x.float(), block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False,
        )
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 RTN quant unavailable on this hardware/build: {e}")

    mx_ref = MXTensor.to_mx(
        x.float().cpu().contiguous(),
        torch.float4_e2m1fn_x2,
        MXFP4_BLOCK_SIZE,
        scaling_mode=ScaleCalculationMode.EVEN,
    )

    torch.testing.assert_close(scales.cpu(), mx_ref.scale.view(torch.uint8), atol=0, rtol=0)
    torch.testing.assert_close(data_fp4.cpu(), mx_ref.qdata.view(torch.uint8), atol=0, rtol=0)

    x_deq_lumen = convert_from_mxfp4(
        data_fp4, scales, output_dtype=torch.float32, block_size=MXFP4_BLOCK_SIZE,
    )
    x_deq_ref = mx_ref.dequantize(torch.float32)
    torch.testing.assert_close(x_deq_lumen.cpu(), x_deq_ref, atol=0, rtol=0)


@pytest.mark.parametrize("shape", MXFP4_SHAPES, ids=[f"{m}x{n}" for m, n in MXFP4_SHAPES])
def test_mxfp4_1d_rtn_cross_dequant_with_torchao(shape):
    """TorchAO should dequantize Lumen's packed MXFP4 payload identically."""
    _require_mxfp4_dtype()
    M, N = shape
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    try:
        data_fp4, scales = convert_to_mxfp4(
            x.float(), block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False,
        )
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 RTN quant unavailable on this hardware/build: {e}")

    x_deq_lumen = convert_from_mxfp4(
        data_fp4, scales, output_dtype=torch.float32, block_size=MXFP4_BLOCK_SIZE,
    )
    x_deq_torchao = _torchao_mxfp4_dequant(data_fp4, scales)
    torch.testing.assert_close(x_deq_lumen.cpu(), x_deq_torchao, atol=0, rtol=0)


@pytest.mark.parametrize("shape", MXFP4_SHAPES, ids=[f"{m}x{n}" for m, n in MXFP4_SHAPES])
def test_mxfp4_2d_rtn_roundtrip_snr(shape):
    """Lumen 32x32 MXFP4 weight quantization has no TorchAO MXTensor equivalent."""
    _require_mxfp4_dtype()
    M, N = shape
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    try:
        data_fp4, scales_2d = convert_to_mxfp4_2d(
            x.float(), block_size=MXFP4_BLOCK_SIZE, use_sr=False,
        )
        x_deq = convert_from_mxfp4_2d(
            data_fp4, scales_2d, output_dtype=torch.float32, block_size=MXFP4_BLOCK_SIZE,
        )
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 2D quant unavailable on this hardware/build: {e}")

    snr = compute_snr(x.float(), x_deq)
    assert snr >= 4.0, f"MXFP4 2D roundtrip SNR {snr:.1f} dB too low"
    assert not torch.isnan(x_deq).any()
    assert not torch.isinf(x_deq).any()


@pytest.mark.parametrize("shape", MXFP4_SHAPES, ids=[f"{m}x{n}" for m, n in MXFP4_SHAPES])
def test_mxfp4_transpose_packed_matches_unpack_reference(shape):
    """Packed FP4 transpose should match unpack -> transpose -> repack reference."""
    _require_mxfp4_dtype()
    M, N = shape
    torch.manual_seed(11)
    torch.cuda.manual_seed(11)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    try:
        data_fp4, _ = convert_to_mxfp4(
            x.float(), block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False,
        )
        transposed = transpose_packed_fp4(data_fp4)
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 transpose unavailable on this hardware/build: {e}")

    unpacked = data_fp4.cpu().repeat_interleave(2, dim=-1)
    unpacked[..., ::2] = unpacked[..., ::2] & 0xF
    unpacked[..., 1::2] = unpacked[..., 1::2] >> 4
    ref_unpacked_t = unpacked.t().contiguous()
    ref = ref_unpacked_t[..., ::2] | (ref_unpacked_t[..., 1::2] << 4)

    torch.testing.assert_close(transposed.cpu(), ref, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(64, 128), (128, 256)], ids=["64x128", "128x256"])
def test_mxfp4_hadamard_transform_matches_torchao_matrix(shape):
    """Lumen blockwise RHT should match TorchAO's explicit 16x16 RHT matrix."""
    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import get_rht_matrix

    M, N = shape
    # TorchAO's helper currently exposes only 16x16 Hadamard matrices. Lumen's
    # runtime MXFP4 path uses g=32, but the same kernel supports g=16, which lets
    # us compare the operator against TorchAO's reference matrix directly.
    g = 16
    torch.manual_seed(17)
    torch.cuda.manual_seed(17)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    sign = torch.where(
        torch.rand(g, device="cuda") >= 0.5,
        torch.ones(g, device="cuda"),
        -torch.ones(g, device="cuda"),
    )

    try:
        y = hadamard_transform(x, sign, g=g)
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen Hadamard transform unavailable on this hardware/build: {e}")

    h = get_rht_matrix(tuple(sign.cpu().to(torch.int8).tolist()), "cpu", torch.float32, g)
    ref = (x.float().cpu().reshape(M, N // g, g) @ h).reshape(M, N)
    torch.testing.assert_close(y.float().cpu(), ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", MXFP4_SHAPES, ids=[f"{m}x{n}" for m, n in MXFP4_SHAPES])
def test_mxfp4_axis0_quant_vs_torchao(shape):
    """Lumen axis=0 quant should match torchAO quant on the transposed input."""
    _require_mxfp4_dtype()
    M, N = shape
    torch.manual_seed(55)
    torch.cuda.manual_seed(55)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    try:
        data_fp4, scales = convert_to_mxfp4(
            x.float(), block_size=MXFP4_BLOCK_SIZE, axis=0, use_sr=False,
        )
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 axis=0 quant unavailable: {e}")

    mx_ref = MXTensor.to_mx(
        x.float().t().contiguous().cpu(),
        torch.float4_e2m1fn_x2,
        MXFP4_BLOCK_SIZE,
        scaling_mode=ScaleCalculationMode.EVEN,
    )

    ref_data = mx_ref.qdata.view(torch.uint8)
    ref_scales = mx_ref.scale.view(torch.uint8)

    # Lumen axis=0 returns transposed packed data: (N//2, M) scales: (N//block, M)
    # torchAO quantizes x.T along axis=-1: data (N, M//2), scales (N, M//block)
    # Lumen's axis=0 transposes before and after, so the packed output shape is
    # (N//2, M) for data and (N//block, M) for scales.
    torch.testing.assert_close(
        data_fp4.t().contiguous().cpu(), ref_data, atol=0, rtol=0,
    )
    torch.testing.assert_close(
        scales.t().contiguous().cpu(), ref_scales, atol=0, rtol=0,
    )


@pytest.mark.parametrize("shape", MXFP4_SHAPES, ids=[f"{m}x{n}" for m, n in MXFP4_SHAPES])
def test_mxfp4_dual_axis_vs_torchao(shape):
    """Both axes from dual_axis should match independent torchAO quantizations."""
    _require_mxfp4_dtype()
    M, N = shape
    torch.manual_seed(77)
    torch.cuda.manual_seed(77)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    try:
        row_fp4, row_scales, col_fp4, col_scales = convert_to_mxfp4_dual_axis(
            x.float(), block_size=MXFP4_BLOCK_SIZE, use_sr=False,
        )
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 dual_axis quant unavailable: {e}")

    mx_row = MXTensor.to_mx(
        x.float().cpu().contiguous(),
        torch.float4_e2m1fn_x2,
        MXFP4_BLOCK_SIZE,
        scaling_mode=ScaleCalculationMode.EVEN,
    )
    torch.testing.assert_close(
        row_fp4.cpu(), mx_row.qdata.view(torch.uint8), atol=0, rtol=0,
    )
    torch.testing.assert_close(
        row_scales.cpu(), mx_row.scale.view(torch.uint8), atol=0, rtol=0,
    )

    mx_col = MXTensor.to_mx(
        x.float().t().contiguous().cpu(),
        torch.float4_e2m1fn_x2,
        MXFP4_BLOCK_SIZE,
        scaling_mode=ScaleCalculationMode.EVEN,
    )
    torch.testing.assert_close(
        col_fp4.t().contiguous().cpu(), mx_col.qdata.view(torch.uint8), atol=0, rtol=0,
    )
    torch.testing.assert_close(
        col_scales.t().contiguous().cpu(), mx_col.scale.view(torch.uint8), atol=0, rtol=0,
    )


@pytest.mark.parametrize(
    "M,K",
    [(4096, 4096), (2048, 12288), (512, 128)],
    ids=["square", "wide", "small"],
)
def test_mxfp4_dequant_hadamard_quant_matches_two_pass(M, K):
    """The fused WGrad activation path must equal the two kernels it replaces.

    Backward needs the stored FP4 activation rotated and transposed. Doing it as
    dequant-transpose then Hadamard-quant writes a BF16 (K, M) buffer worth four
    times the bytes of either FP4 end; the fused kernel keeps it in registers.
    With round-to-nearest the arithmetic is identical, so bit-equality is the
    right bar -- anything looser would hide a mislaid scale block.
    """
    _require_mxfp4_dtype()
    torch.manual_seed(3)
    torch.cuda.manual_seed(3)
    G = 16
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    sign = torch.where(torch.rand(G, device="cuda") < 0.5, -1.0, 1.0).to(torch.bfloat16)

    try:
        fp4, scales = convert_to_mxfp4(x, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False)
        ref_bf16 = dequant_transpose_mxfp4(fp4, scales, block_size=MXFP4_BLOCK_SIZE)
        ref_q, ref_s = hadamard_quant_mxfp4(
            ref_bf16, sign, block_size=MXFP4_BLOCK_SIZE, g=G, use_sr=False,
        )
        got_q, got_s = dequant_hadamard_quant_mxfp4(
            fp4, scales, sign, block_size=MXFP4_BLOCK_SIZE, g=G, use_sr=False,
        )
    except (AssertionError, RuntimeError, CompilationError) as e:
        pytest.skip(f"Lumen MXFP4 kernels unavailable: {e}")

    assert tuple(got_q.shape) == (K, M // 2)
    assert tuple(got_s.shape) == (K, M // MXFP4_BLOCK_SIZE)
    torch.testing.assert_close(got_q, ref_q, atol=0, rtol=0)
    torch.testing.assert_close(got_s, ref_s, atol=0, rtol=0)


@pytest.mark.parametrize(
    "M,K",
    [(4096, 4096), (2048, 12288), (512, 256)],
    ids=["square", "wide", "small"],
)
def test_mxfp4_wgrad_operand_shuffle_matches_reference(M, K):
    """Storing the WGrad B operand shuffled must equal shuffling it afterwards.

    The GEMM reads this operand through a hardcoded permuted offset, so bytes
    landing in the wrong place would quietly multiply the wrong elements instead
    of failing. Only bit-equality against the separate shuffle catches that.
    """
    _require_mxfp4_dtype()
    pytest.importorskip("aiter")
    from lumen.ops.quantize.linear import _shuffle_mxfp4_weight
    from lumen.ops.quantize.ops import mxfp4_data_shuffle_supported

    if not is_cdna4():
        pytest.skip("gfx950 B-operand layout")

    torch.manual_seed(5)
    G = 16
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    sign = torch.where(torch.rand(G, device="cuda") < 0.5, -1.0, 1.0).to(torch.bfloat16)

    # No skip-on-failure here: the hardware gates above already decide whether
    # this shape is supported, so anything raised past them is a real defect.
    fp4, scales = convert_to_mxfp4(x, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False)
    assert mxfp4_data_shuffle_supported(K, M // 2)
    plain, plain_s = dequant_hadamard_quant_mxfp4(
        fp4, scales, sign, block_size=MXFP4_BLOCK_SIZE, g=G, use_sr=False,
    )
    fused, fused_s = dequant_hadamard_quant_mxfp4(
        fp4, scales, sign, block_size=MXFP4_BLOCK_SIZE, g=G, use_sr=False,
        shuffle_data=True,
    )

    ref = _shuffle_mxfp4_weight(plain, arch="gfx950")
    assert tuple(fused.shape) == tuple(plain.shape)
    torch.testing.assert_close(fused.view(torch.uint8), ref.view(torch.uint8), atol=0, rtol=0)
    # The shuffle only reorders the packed data; scales are untouched.
    torch.testing.assert_close(fused_s, plain_s, atol=0, rtol=0)


@pytest.mark.parametrize(
    "N_out,K_in",
    [(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)],
    ids=["qkv", "proj", "fc1", "fc2"],
)
def test_mxfp4_transposed_weight_shuffle_matches_reference(N_out, K_in):
    """The pre-transposed weight may be stored shuffled without changing a byte.

    DGrad reads this operand through the GEMM's permuted offsets, so writing it
    in that order during the transpose has to land every byte exactly where the
    separate shuffle would have put it.
    """
    _require_mxfp4_dtype()
    pytest.importorskip("aiter")
    from lumen.ops.quantize.linear import _shuffle_mxfp4_weight
    from lumen.ops.quantize.ops import mxfp4_data_shuffle_supported, transpose_packed_fp4

    if not is_cdna4():
        pytest.skip("gfx950 B-operand layout")

    torch.manual_seed(7)
    w = torch.randn(N_out, K_in, device="cuda", dtype=torch.bfloat16)
    w_fp4, _ = convert_to_mxfp4(w, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False)

    assert mxfp4_data_shuffle_supported(K_in, N_out // 2)
    plain = transpose_packed_fp4(w_fp4)
    fused = transpose_packed_fp4(w_fp4, shuffle_data=True)

    assert tuple(fused.shape) == (K_in, N_out // 2)
    torch.testing.assert_close(fused, _shuffle_mxfp4_weight(plain, arch="gfx950"), atol=0, rtol=0)


@pytest.mark.parametrize(
    "N_out,K_in",
    [(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)],
    ids=["qkv", "proj", "fc1", "fc2"],
)
def test_mxfp4_weight_quant_fused_shuffle_matches_reference(N_out, K_in):
    """The forward weight operand may be quantized straight into the GEMM order.

    Nothing but the GEMM reads it, so the quantizer stores it shuffled and the
    separate permuting pass over the whole FP4 weight goes away. The transpose
    that DGrad's operand comes from then has to read that layout back, and both
    ends have to land byte for byte where the two-pass path put them.
    """
    _require_mxfp4_dtype()
    pytest.importorskip("aiter")
    from lumen.ops.quantize.linear import _shuffle_mxfp4_weight
    from lumen.ops.quantize.ops import (
        convert_to_mxfp4_2d,
        mxfp4_data_shuffle_supported,
        transpose_packed_fp4,
    )

    if not is_cdna4():
        pytest.skip("gfx950 B-operand layout")

    torch.manual_seed(9)
    w = torch.randn(N_out, K_in, device="cuda", dtype=torch.bfloat16)

    assert mxfp4_data_shuffle_supported(N_out, K_in // 2)
    plain, plain_s = convert_to_mxfp4_2d(w, block_size=MXFP4_BLOCK_SIZE, use_sr=False)
    fused, fused_s = convert_to_mxfp4_2d(
        w, block_size=MXFP4_BLOCK_SIZE, use_sr=False, shuffle_data=True,
    )

    torch.testing.assert_close(fused, _shuffle_mxfp4_weight(plain, arch="gfx950"), atol=0, rtol=0)
    # Only the packed data moves; the tile scales are laid out as before.
    torch.testing.assert_close(fused_s, plain_s, atol=0, rtol=0)

    # DGrad's operand comes off the same tensor, so the transpose has to be able
    # to read the shuffled form back.
    torch.testing.assert_close(
        transpose_packed_fp4(fused, in_shuffled=True),
        transpose_packed_fp4(plain),
        atol=0, rtol=0,
    )
    torch.testing.assert_close(
        transpose_packed_fp4(fused, in_shuffled=True, shuffle_data=True),
        transpose_packed_fp4(plain, shuffle_data=True),
        atol=0, rtol=0,
    )


@pytest.mark.parametrize(
    "N_out,K_in",
    [(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)],
    ids=["qkv", "proj", "fc1", "fc2"],
)
@pytest.mark.parametrize("transpose", [False, True], ids=["forward", "dgrad"])
def test_mxfp4_expanded_scale_swizzle_matches_two_step(N_out, K_in, transpose):
    """Expanding 2D tile scales inside the swizzle must not move a byte.

    Both operand layouts of an MXFP4 weight read their scales through the GEMM's
    permuted offsets, so replicating the tile scale in the load has to land
    exactly where expand-then-swizzle would have put it.
    """
    _require_mxfp4_dtype()
    pytest.importorskip("aiter")
    from lumen.ops.quantize.linear import _expand_2d_scale_to_1d
    from lumen.ops.quantize.ops import convert_to_mxfp4_2d, swizzle_expanded_mxfp4_scale

    if not is_cdna4():
        pytest.skip("gfx950 scale layout")

    torch.manual_seed(11)
    w = torch.randn(N_out, K_in, device="cuda", dtype=torch.bfloat16)
    _, scale_2d = convert_to_mxfp4_2d(w, block_size=MXFP4_BLOCK_SIZE)

    src = scale_2d.t().contiguous() if transpose else scale_2d
    rows = (K_in if transpose else N_out)
    ref = swizzle_mxfp4_scale(_expand_2d_scale_to_1d(src, (rows, 0)))
    got = swizzle_expanded_mxfp4_scale(
        scale_2d, block_size=MXFP4_BLOCK_SIZE, transpose=transpose,
    )

    assert tuple(got.shape) == tuple(ref.shape)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    "rows,cols",
    [(16384, 128), (16384, 384), (24576, 512), (4096, 512), (32, 8)],
    ids=["act", "act-wide", "wgrad", "wgrad-narrow", "single-tile"],
)
def test_mxfp4_scale_swizzle_matches_aiter(rows, cols):
    """The scale swizzle must reproduce AITER's layout byte for byte.

    The GEMM addresses scales through a hardcoded permuted offset, so a single
    misplaced byte silently scales the wrong block rather than failing -- only
    bit-equality catches that.
    """
    pytest.importorskip("aiter")
    from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm

    if not is_cdna4():
        pytest.skip("gfx950 scale layout")

    torch.manual_seed(0)
    scales = torch.randint(0, 256, (rows, cols), dtype=torch.uint8, device="cuda")

    ref = shuffle_scale_gemm(scales, arch="gfx950", preshuffle_factor=32, scale_kwidth=8)
    got = swizzle_mxfp4_scale(scales)

    assert tuple(got.shape) == (rows // 32, cols * 32)
    torch.testing.assert_close(got, ref.reshape(got.shape), atol=0, rtol=0)


@pytest.mark.parametrize(
    "M,N", [(4096, 4096), (2048, 12288)], ids=["square", "wide"],
)
def test_mxfp4_dual_layout_rotation_matches_reference_hadamard(M, N):
    """The transposed output must be the reference rotation, quantized.

    The quantizers apply the rotation on the matrix unit as one ±1/4 matrix
    rather than as a sign multiply plus a butterfly. That is only free because
    BF16 holds both the matrix and the operand exactly, and a lost bit would
    show up as a rounding difference rather than a failure, so compare the
    packed result against the torch reference bit for bit.
    """
    _require_mxfp4_dtype()
    torch.manual_seed(11)
    G = 16
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    sign = torch.where(torch.rand(G, device="cuda") < 0.5, -1.0, 1.0).to(torch.bfloat16)

    try:
        _, _, got_b, got_bs = dual_layout_quant_mxfp4(
            x, sign, block_size=MXFP4_BLOCK_SIZE, g=G,
            use_sr_row=False, use_sr_transposed=False,
        )
        # dY^T rotated along its last axis is what WGrad consumes. Rotating in
        # FP32 keeps the reference where the kernel is: the operands are
        # BF16-exact but each rotated value is a sum of 16 of them.
        ref_rot = hadamard_transform(x.t().contiguous().float(), sign, g=G)
        ref_b, ref_bs = convert_to_mxfp4(
            ref_rot, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False,
        )
    except (AssertionError, RuntimeError, CompilationError) as e:
        pytest.skip(f"Lumen MXFP4 kernels unavailable: {e}")

    torch.testing.assert_close(got_b, ref_b, atol=0, rtol=0)
    torch.testing.assert_close(got_bs, ref_bs, atol=0, rtol=0)


@pytest.mark.parametrize(
    "M,N", [(4096, 4096), (2048, 12288)], ids=["square", "wide"],
)
def test_mxfp4_dual_layout_fused_swizzle_matches_separate_pass(M, N):
    """Storing scales pre-swizzled must equal quantizing then swizzling.

    The GEMM reads scales through a fixed permuted offset, so a wrong index in
    the fused store mis-scales blocks instead of failing; only bit-equality
    against the two-step path catches it.
    """
    _require_mxfp4_dtype()
    if not is_cdna4():
        pytest.skip("gfx950 scale layout")

    torch.manual_seed(7)
    G = 16
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    sign = torch.where(torch.rand(G, device="cuda") < 0.5, -1.0, 1.0).to(torch.bfloat16)
    kw = dict(block_size=MXFP4_BLOCK_SIZE, g=G, use_sr_row=False, use_sr_transposed=False)

    try:
        ref_a, ref_as, ref_b, ref_bs = dual_layout_quant_mxfp4(x, sign, **kw)
        got_a, got_as, got_b, got_bs = dual_layout_quant_mxfp4(
            x, sign, swizzle_scale=True, **kw
        )
    except (AssertionError, RuntimeError, CompilationError) as e:
        pytest.skip(f"Lumen MXFP4 kernels unavailable: {e}")

    # The packed data is untouched by the scale layout.
    torch.testing.assert_close(got_a, ref_a, atol=0, rtol=0)
    torch.testing.assert_close(got_b, ref_b, atol=0, rtol=0)
    torch.testing.assert_close(got_as, swizzle_mxfp4_scale(ref_as), atol=0, rtol=0)
    torch.testing.assert_close(got_bs, swizzle_mxfp4_scale(ref_bs), atol=0, rtol=0)


def test_mxfp4_dequant_hadamard_fused_swizzle_matches_separate_pass():
    """The WGrad activation quantizer's swizzled store must match the two-step path."""
    _require_mxfp4_dtype()
    if not is_cdna4():
        pytest.skip("gfx950 scale layout")

    torch.manual_seed(5)
    M, K, G = 2048, 4096, 16
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    sign = torch.where(torch.rand(G, device="cuda") < 0.5, -1.0, 1.0).to(torch.bfloat16)

    try:
        fp4, scales = convert_to_mxfp4(x, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False)

        def run(swz):
            return dequant_hadamard_quant_mxfp4(
                fp4, scales, sign, block_size=MXFP4_BLOCK_SIZE, g=G,
                use_sr=False, swizzle_scale=swz,
            )

        ref_q, ref_s = run(False)
        got_q, got_s = run(True)
    except (AssertionError, RuntimeError, CompilationError) as e:
        pytest.skip(f"Lumen MXFP4 kernels unavailable: {e}")

    torch.testing.assert_close(got_q, ref_q, atol=0, rtol=0)
    torch.testing.assert_close(got_s, swizzle_mxfp4_scale(ref_s), atol=0, rtol=0)


@pytest.mark.parametrize(
    "M,K", [(4096, 4096), (2048, 12288)], ids=["square", "wide"],
)
def test_mxfp4_convert_fused_swizzle_matches_separate_pass(M, K):
    """The activation quantizer's swizzled store must match the two-step path."""
    _require_mxfp4_dtype()
    if not is_cdna4():
        pytest.skip("gfx950 scale layout")

    torch.manual_seed(3)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    try:
        ref_q, ref_s = convert_to_mxfp4(x, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False)
        got_q, got_s = convert_to_mxfp4(
            x, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False, swizzle_scale=True,
        )
    except (AssertionError, RuntimeError, CompilationError) as e:
        pytest.skip(f"Lumen MXFP4 kernels unavailable: {e}")

    torch.testing.assert_close(got_q, ref_q, atol=0, rtol=0)
    torch.testing.assert_close(got_s, swizzle_mxfp4_scale(ref_s), atol=0, rtol=0)


def test_mxfp4_dequant_hadamard_reads_swizzled_input_scale():
    """Reading a pre-swizzled input scale must equal reading the row-major one.

    Forward stores the activation's scales in the GEMM layout, so WGrad's
    requantizer gathers them through the permutation instead of a separate pass
    putting them back. A wrong index there mis-scales blocks rather than
    failing, so only equality against the row-major read catches it.
    """
    _require_mxfp4_dtype()
    if not is_cdna4():
        pytest.skip("gfx950 scale layout")

    torch.manual_seed(5)
    M, K, G = 2048, 4096, 16
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    sign = torch.where(torch.rand(G, device="cuda") < 0.5, -1.0, 1.0).to(torch.bfloat16)

    try:
        fp4, scales = convert_to_mxfp4(x, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False)
        fp4_s, scales_s = convert_to_mxfp4(
            x, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False, swizzle_scale=True,
        )
        ref_q, ref_s = dequant_hadamard_quant_mxfp4(
            fp4, scales, sign, block_size=MXFP4_BLOCK_SIZE, g=G, use_sr=False,
        )
        got_q, got_s = dequant_hadamard_quant_mxfp4(
            fp4_s, scales_s, sign, block_size=MXFP4_BLOCK_SIZE, g=G, use_sr=False,
            in_scale_swizzled=True,
        )
    except (AssertionError, RuntimeError, CompilationError) as e:
        pytest.skip(f"Lumen MXFP4 kernels unavailable: {e}")

    torch.testing.assert_close(got_q, ref_q, atol=0, rtol=0)
    torch.testing.assert_close(got_s, ref_s, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["asm", "shuffled", "plain"])
def test_mxfp4_gemm_accepts_fused_swizzled_scales(backend):
    """Every GEMM backend must read a fused-swizzle scale as the plain one.

    The quantizer cannot know which backend its scales will reach, so it emits
    the swizzled layout and each backend either reads it directly or undoes it.
    Feeding both layouts of the same quantization through the same kernel is
    what proves that contract holds -- a backend that ignored the tag would
    still return a plausible-looking tensor.
    """
    _require_mxfp4_dtype()
    if not is_cdna4():
        pytest.skip("gfx950 scale layout")
    from lumen.ops.quantize.linear import (
        _MXFP4_BACKENDS,
        _mark_mxfp4_scale_swizzled,
        _mxfp4_can_fuse_scale_swizzle,
    )

    torch.manual_seed(11)
    M, N, K = 512, 512, 4096
    G, blk = 16, MXFP4_BLOCK_SIZE
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    sign = torch.where(torch.rand(G, device="cuda") < 0.5, -1.0, 1.0).to(torch.bfloat16)

    if not _mxfp4_can_fuse_scale_swizzle((M, K // blk), (K, M // blk)):
        pytest.skip(f"({M}, {K}) scales do not meet the fused-swizzle alignment")

    # Same philox stream both ways, so only the scale layout differs.
    seed, off = 1234, 99
    kw = dict(block_size=blk, g=G, use_sr_row=False, use_sr_transposed=False,
              philox_seed=seed, philox_offset=off)
    try:
        a_fp4, a_scale, _, _ = dual_layout_quant_mxfp4(x, sign, **kw)
        a_fp4_s, a_scale_s, _, _ = dual_layout_quant_mxfp4(x, sign, swizzle_scale=True, **kw)
        w_fp4, w_scale = convert_to_mxfp4(w, block_size=blk, axis=-1, use_sr=False)
    except (AssertionError, RuntimeError, CompilationError) as e:
        pytest.skip(f"Lumen MXFP4 kernels unavailable: {e}")

    torch.testing.assert_close(a_fp4_s, a_fp4, atol=0, rtol=0)
    _mark_mxfp4_scale_swizzled(a_scale_s)

    gemm = _MXFP4_BACKENDS[backend]
    try:
        ref = gemm(a_fp4, w_fp4, a_scale, w_scale)
        got = gemm(a_fp4_s, w_fp4, a_scale_s, w_scale)
    except (AssertionError, RuntimeError, NotImplementedError) as e:
        pytest.skip(f"{backend} backend rejected these operands: {e}")

    torch.testing.assert_close(got, ref, atol=0, rtol=0)


def test_mxfp4_stochastic_rounding_unbiased():
    """SR quant-dequant should be unbiased: mean over many rounds ≈ original."""
    _require_mxfp4_dtype()
    torch.manual_seed(99)
    torch.cuda.manual_seed(99)
    M, N = 64, 128
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    x_f32 = x.float()

    num_rounds = 200
    deq_sum = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    sr_last_fp4 = None

    for i in range(num_rounds):
        try:
            sr_fp4, sr_scales = convert_to_mxfp4(
                x_f32, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=True,
                philox_seed=i, philox_offset=0,
            )
        except (AssertionError, RuntimeError, CompilationError) as e:
            pytest.skip(f"Lumen MXFP4 SR quant unavailable: {e}")

        deq = convert_from_mxfp4(
            sr_fp4, sr_scales, output_dtype=torch.float32, block_size=MXFP4_BLOCK_SIZE,
        )
        deq_sum += deq
        sr_last_fp4 = sr_fp4

    rtn_fp4, _ = convert_to_mxfp4(
        x_f32, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False,
    )

    mean_deq = deq_sum / num_rounds

    # Unbiasedness: mean should be close to original within FP4 quantization noise
    abs_err = (mean_deq - x_f32).abs()
    max_abs_err = abs_err.max().item()
    assert max_abs_err < 1.0, f"SR mean max error {max_abs_err:.4f} too large (expect < 1.0)"

    mean_abs_err = abs_err.mean().item()
    assert mean_abs_err < 0.15, f"SR mean abs error {mean_abs_err:.4f} too large (expect < 0.15)"

    # SR should produce different packed bytes from RTN for at least some elements
    assert not torch.equal(sr_last_fp4, rtn_fp4), "SR and RTN produced identical outputs"


def test_mxfp4_stochastic_rounding_neighbours_draw_independently():
    """Adjacent elements must not share a random word.

    One Philox round yields four 32-bit words and the kernel spends all four,
    handing them to four adjacent output bytes rather than throwing three away.
    Broadcasting a single word across those bytes instead would be just as fast
    and just as unbiased, but would correlate the rounding noise of neighbours
    and defeat the point of stochastic rounding. This pins the distinction.
    """
    _require_mxfp4_dtype()
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    M, N = 256, 128
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16).float()

    # Elements sharing a round span 4 packed bytes = 8 columns.
    SPAN = 8
    rounds = 32
    noise = []
    for i in range(rounds):
        try:
            fp4, scales = convert_to_mxfp4(
                x, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=True,
                philox_seed=i, philox_offset=0,
            )
        except (AssertionError, RuntimeError, CompilationError) as e:
            pytest.skip(f"Lumen MXFP4 SR quant unavailable: {e}")
        deq = convert_from_mxfp4(
            fp4, scales, output_dtype=torch.float32, block_size=MXFP4_BLOCK_SIZE,
        )
        noise.append(deq - x)
    noise = torch.stack(noise)  # (rounds, M, N)

    flat = noise[:, :, :SPAN].permute(2, 0, 1).reshape(SPAN, -1)
    corr = torch.corrcoef(flat)
    off_diag = corr[~torch.eye(SPAN, dtype=torch.bool, device=corr.device)]
    max_corr = off_diag.abs().max().item()
    assert max_corr < 0.2, (
        f"noise of elements sharing a Philox round correlates at {max_corr:.3f}; "
        "they are drawing from the same word"
    )


@pytest.mark.parametrize(
    "M,K,N", [(64, 128, 64), (128, 256, 128)],
    ids=["64x128x64", "128x256x128"],
)
def test_mxfp4_gemm_vs_torchao_gemm(M, K, N):
    """Lumen MXFP4 GEMM (1D scales) should match torchAO MXTensor matmul."""
    _require_mxfp4_dtype()
    torch.manual_seed(33)
    torch.cuda.manual_seed(33)

    a_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_hp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    # Lumen: quant both with 1D scales, then GEMM
    try:
        a_fp4, a_scales = convert_to_mxfp4(
            a_hp.float(), block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False,
        )
        w_fp4, w_scales = convert_to_mxfp4(
            w_hp.float(), block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False,
        )
        y_lumen = gemm_mxfp4_dispatch(a_fp4, w_fp4, a_scales, w_scales)
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 GEMM unavailable: {e}")

    # torchAO: quant both with MXTensor, then matmul (dequant→FP32 matmul)
    mx_a = MXTensor.to_mx(
        a_hp.float().cpu().contiguous(),
        torch.float4_e2m1fn_x2,
        MXFP4_BLOCK_SIZE,
        scaling_mode=ScaleCalculationMode.EVEN,
    )
    mx_w = MXTensor.to_mx(
        w_hp.float().cpu().contiguous(),
        torch.float4_e2m1fn_x2,
        MXFP4_BLOCK_SIZE,
        scaling_mode=ScaleCalculationMode.EVEN,
    )
    y_torchao = (mx_a.dequantize(torch.float32) @ mx_w.dequantize(torch.float32).t())

    # Lumen dequant reference (should match torchAO dequant — verified by other tests)
    a_deq = convert_from_mxfp4(
        a_fp4, a_scales, output_dtype=torch.float32, block_size=MXFP4_BLOCK_SIZE,
    )
    w_deq = convert_from_mxfp4(
        w_fp4, w_scales, output_dtype=torch.float32, block_size=MXFP4_BLOCK_SIZE,
    )
    y_lumen_deq = a_deq @ w_deq.t()

    # Lumen GEMM vs Lumen dequant-matmul (self-consistency, SNR)
    snr_self = compute_snr(y_lumen_deq, y_lumen.float())
    # Both paths consume identical FP4 values and scales, so quantization noise
    # cancels; only accumulation order remains. A single-digit floor would let
    # a wrong kernel/backend pass while still looking like ordinary FP4 error.
    assert snr_self >= 40.0, f"Lumen GEMM self-consistency SNR {snr_self:.1f} dB too low"

    # Lumen dequant-matmul vs torchAO dequant-matmul (cross-framework, bitwise)
    torch.testing.assert_close(y_lumen_deq.cpu(), y_torchao, atol=0, rtol=0)


@pytest.mark.parametrize(
    "M,N,K",
    [(2048, 28672, 4096), (2048, 4096, 14336)],
    ids=["gate_up", "down_proj"],
)
def test_mxfp4_preshuffle_gemm_matches_plain(M, N, K):
    """Shuffled-layout MXFP4 GEMM must be numerically identical to the plain one.

    The shuffled kernel only rearranges how the B operand and the scales are laid
    out in memory, so both kernels consume the same values and must agree to
    within GEMM reduction-order noise.
    """
    _require_mxfp4_dtype()
    if os.environ.get("LUMEN_MXFP4_PRESHUFFLE") is not None:
        pytest.skip("LUMEN_MXFP4_PRESHUFFLE overrides the policy this test pins down")
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    a_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_hp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05

    a_fp4, a_scales = convert_to_mxfp4(
        a_hp, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False,
    )
    w_fp4, w_scales = convert_to_mxfp4_2d(
        w_hp, block_size=MXFP4_BLOCK_SIZE, use_sr=False,
    )

    assert _mxfp4_preshuffle_eligible(a_fp4, w_fp4), "shape should select the shuffled path"

    try:
        y_plain = _gemm_mxfp4_aiter(a_fp4, w_fp4, a_scales, w_scales)
        y_shuf = _gemm_mxfp4_aiter_preshuffle(a_fp4, w_fp4, a_scales, w_scales)
    except (AssertionError, RuntimeError, NotImplementedError) as e:
        pytest.skip(f"AITER MXFP4 GEMM unavailable: {e}")

    # Bit-exact, not merely close: the autotuner is free to swap backends between
    # runs, so anything less would make results depend on a timing measurement.
    torch.testing.assert_close(y_shuf, y_plain, atol=0, rtol=0)

    y_dispatch = gemm_mxfp4_dispatch(a_fp4, w_fp4, a_scales, w_scales)
    torch.testing.assert_close(y_dispatch, y_shuf, atol=0, rtol=0)


def test_mxfp4_preshuffle_eligibility():
    """Only large, 16-row-aligned weights should take the shuffle prologue."""
    _require_mxfp4_dtype()
    if os.environ.get("LUMEN_MXFP4_PRESHUFFLE") is not None:
        pytest.skip("LUMEN_MXFP4_PRESHUFFLE overrides the policy this test pins down")

    def _operands(M, N, K):
        a = torch.empty((M, K // 2), dtype=torch.uint8, device="cuda")
        w = torch.empty((N, K // 2), dtype=torch.uint8, device="cuda")
        return a, w

    # Llama-8B attention projections are below the weight-size threshold.
    assert not _mxfp4_preshuffle_eligible(*_operands(2048, 4096, 4096))
    assert not _mxfp4_preshuffle_eligible(*_operands(2048, 6144, 4096))

    # MLP projections are above it.
    assert _mxfp4_preshuffle_eligible(*_operands(2048, 28672, 4096))
    assert _mxfp4_preshuffle_eligible(*_operands(2048, 4096, 14336))

    # N must tile by 16, and the kernel needs M >= 32.
    assert not _mxfp4_preshuffle_eligible(*_operands(2048, 28680, 4096))
    assert not _mxfp4_preshuffle_eligible(*_operands(16, 28672, 4096))


@pytest.mark.parametrize(
    "M,N,K",
    [(2048, 28672, 4096), (2048, 4096, 14336), (2048, 6144, 4096)],
    ids=["gate_up", "down_proj", "qkv_proj"],
)
def test_mxfp4_asm_gemm_matches_plain(M, N, K):
    """The A4W4 ASM kernels must agree with the plain Triton MXFP4 GEMM.

    Lumen only rewrites the operand layout for these kernels -- the B tiling and
    the padded, swizzled scales -- so both paths consume the same values. A wrong
    layout does not raise, it just mislays scales, which is exactly what this
    catches.
    """
    _require_mxfp4_dtype()
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    a_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_hp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05

    a_fp4, a_scales = convert_to_mxfp4(
        a_hp, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False,
    )
    w_fp4, w_scales = convert_to_mxfp4_2d(
        w_hp, block_size=MXFP4_BLOCK_SIZE, use_sr=False,
    )

    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() not in _MXFP4_ASM_ARCHS or not _mxfp4_asm_tuned(M, N, K):
        pytest.skip(f"no tuned A4W4 kernel for {M}x{N}x{K} on {get_arch()}")

    try:
        y_plain = _gemm_mxfp4_aiter(a_fp4, w_fp4, a_scales, w_scales)
        y_asm = _gemm_mxfp4_aiter_asm(a_fp4, w_fp4, a_scales, w_scales)
    except (AssertionError, RuntimeError, NotImplementedError) as e:
        pytest.skip(f"AITER A4W4 MXFP4 GEMM unavailable: {e}")

    assert y_asm.shape == (M, N), f"ASM output should be sliced back to M, got {y_asm.shape}"
    # Bit-exact, not merely close: the autotuner is free to swap backends between
    # runs, so anything less would make results depend on a timing measurement.
    torch.testing.assert_close(y_asm, y_plain, atol=0, rtol=0)

    y_dispatch = gemm_mxfp4_dispatch(a_fp4, w_fp4, a_scales, w_scales)
    torch.testing.assert_close(y_dispatch, y_asm, atol=0, rtol=0)


def test_mxfp4_asm_weight_operand_cache_follows_the_scales():
    """The cached weight layout must not survive a change of scales.

    The shuffled weight and its swizzled scales are memoized on the FP4 weight
    tensor so forward and DGrad across micro-batches don't rebuild them. A weight
    tensor re-used with different scales must miss, because a stale cache here
    would not raise -- it would quietly compute with the old scales.
    """
    _require_mxfp4_dtype()
    torch.manual_seed(17)
    torch.cuda.manual_seed(17)
    M, N, K = 2048, 6144, 4096

    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() not in _MXFP4_ASM_ARCHS or not _mxfp4_asm_tuned(M, N, K):
        pytest.skip(f"no tuned A4W4 kernel for {M}x{N}x{K} on {get_arch()}")

    a_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_hp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    a_fp4, a_scales = convert_to_mxfp4(a_hp, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False)
    w_fp4, w_scales = convert_to_mxfp4_2d(w_hp, block_size=MXFP4_BLOCK_SIZE, use_sr=False)

    try:
        first = _gemm_mxfp4_aiter_asm(a_fp4, w_fp4, a_scales, w_scales)
    except (AssertionError, RuntimeError, NotImplementedError) as e:
        pytest.skip(f"AITER A4W4 MXFP4 GEMM unavailable: {e}")

    assert hasattr(
        w_fp4, "_mxfp4_preshuffled_operands"
    ), "shared ASM/FlyDSL weight operands were not cached"

    # A second call on the same weight must hit the cache and change nothing.
    torch.testing.assert_close(
        _gemm_mxfp4_aiter_asm(a_fp4, w_fp4, a_scales, w_scales), first, atol=0, rtol=0,
    )

    # Same weight tensor, different scales: the result must follow the new scales.
    other_scales = torch.full_like(w_scales, 128)
    got = _gemm_mxfp4_aiter_asm(a_fp4, w_fp4, a_scales, other_scales)
    expected = _gemm_mxfp4_aiter(a_fp4, w_fp4, a_scales, other_scales)
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_mxfp4_asm_eligibility():
    """Only large, 16-tileable, tuned shapes should take the ASM layout prologue."""
    _require_mxfp4_dtype()
    if os.environ.get("LUMEN_MXFP4_ASM") is not None:
        pytest.skip("LUMEN_MXFP4_ASM overrides the policy this test pins down")

    def _operands(M, N, K):
        a = torch.empty((M, K // 2), dtype=torch.uint8, device="cuda")
        w = torch.empty((N, K // 2), dtype=torch.uint8, device="cuda")
        return a, w

    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() not in _MXFP4_ASM_ARCHS:
        assert not _mxfp4_asm_eligible(*_operands(2048, 28672, 4096))
        pytest.skip(f"A4W4 ASM path is gfx950-only, running on {get_arch()}")

    # Llama-8B MLP projections clear the weight-size threshold.
    assert _mxfp4_asm_eligible(*_operands(2048, 28672, 4096))  # 56 MiB
    assert _mxfp4_asm_eligible(*_operands(2048, 4096, 14336))  # 28 MiB

    # The attention projections do not: the layout prologue is not amortised.
    assert not _mxfp4_asm_eligible(*_operands(2048, 6144, 4096))  # 12 MiB
    assert not _mxfp4_asm_eligible(*_operands(2048, 4096, 4096))  # 8 MiB

    # N must tile by 16, and so must the packed K dim (K by 32).
    assert not _mxfp4_asm_eligible(*_operands(2048, 28680, 4096))
    assert not _mxfp4_asm_eligible(*_operands(2048, 28672, 4080))

    # Untuned shapes must not reach the direct ASM launch.
    assert not _mxfp4_asm_tuned(64, 64, 128)
    assert not _mxfp4_asm_eligible(*_operands(64, 64, 128))


@pytest.mark.parametrize(
    "entry,k,expected",
    [
        (
            {
                "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
                "splitK": 0,
            },
            128,
            (
                "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
                0,
            ),
        ),
        (
            {
                "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E",
                "splitK": "3",
            },
            2048,
            (
                "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E",
                3,
            ),
        ),
        (
            {
                "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
                "splitK": 2,
            },
            128,
            None,
        ),
        (
            {
                "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
                "splitK": 0,
                "libtype": "other",
            },
            128,
            None,
        ),
        ({"kernelName": "_ZN5aiter17generic_dispatchE", "splitK": 1}, 128, None),
        (
            {
                "kernelName": "_ZN5aiter44f4gemm_bf16_per1x32Fp4_noBpreShuffle_128x256E",
                "splitK": 1,
            },
            128,
            None,
        ),
        ({"kernelName": "", "splitK": 0}, 128, None),
        (
            {
                "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E"
            },
            128,
            None,
        ),
        (
            {
                "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
                "splitK": None,
            },
            128,
            None,
        ),
        (
            {
                "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
                "splitK": float("nan"),
            },
            128,
            None,
        ),
        (
            {
                "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
                "splitK": float("inf"),
            },
            128,
            None,
        ),
        (
            {
                "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
                "splitK": True,
            },
            128,
            None,
        ),
        (
            {
                "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
                "splitK": 1.5,
            },
            128,
            None,
        ),
        (
            {
                "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
                "splitK": -1,
            },
            128,
            None,
        ),
        (
            {
                "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
                "splitK": 4,
            },
            128,
            None,
        ),
        (
            {
                "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E",
                "splitK": 2,
            },
            130,
            None,
        ),
        (None, 128, None),
        (["malformed"], 128, None),
    ],
)
def test_mxfp4_asm_config_accepts_only_explicit_asm_symbols(entry, k, expected):
    """The ASM backend accepts only entries with an explicit ASM identity."""
    assert mxfp4_asm.validate_tuned_entry(entry, k, "gfx950") == expected


def test_mxfp4_asm_config_uses_runtime_registry(monkeypatch):
    """Linear dispatch gets both arch and CU identity from the ASM registry."""
    from lumen.ops.quantize import linear as linear_mod

    expected = (
        "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
        0,
    )
    calls = []

    def _lookup(M, N, K):
        calls.append((M, N, K))
        return expected

    monkeypatch.setattr(mxfp4_asm, "lookup_runtime", _lookup)
    assert linear_mod._mxfp4_asm_config(64, 128, 256) == expected
    assert calls == [(64, 128, 256)]


def test_mxfp4_asm_runtime_lookup_uses_aiter_device_identity(monkeypatch):
    """The table key follows AITER's runtime arch/CU overrides exactly."""
    from aiter.jit.utils import chip_info

    calls = []

    def _lookup(M, N, K, *, arch, cu_num, paths=None):
        calls.append((M, N, K, arch, cu_num, paths))
        return ("asm-symbol", 0)

    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 304)
    monkeypatch.setattr(mxfp4_asm, "lookup", _lookup)
    assert mxfp4_asm.lookup_runtime(64, 128, 256) == ("asm-symbol", 0)
    assert calls == [(64, 128, 256, "gfx950", 304, None)]


def _write_mxfp4_asm_table(path, rows, *, include_gfx=True, include_libtype=False):
    fields = (["gfx"] if include_gfx else []) + [
        "cu_num",
        "M",
        "N",
        "K",
        "kernelName",
        "splitK",
    ]
    if include_libtype:
        fields.append("libtype")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def isolated_mxfp4_asm_artifact_baselines():
    """Model a fresh process for tests of AITER's process-lifetime loader state."""
    with mxfp4_asm._runtime_artifact_lock:
        baselines = dict(mxfp4_asm._runtime_artifact_baselines)
        warnings = set(mxfp4_asm._runtime_artifact_warnings)
        mxfp4_asm._runtime_artifact_baselines.clear()
        mxfp4_asm._runtime_artifact_warnings.clear()
    try:
        yield
    finally:
        with mxfp4_asm._runtime_artifact_lock:
            mxfp4_asm._runtime_artifact_baselines.clear()
            mxfp4_asm._runtime_artifact_baselines.update(baselines)
            mxfp4_asm._runtime_artifact_warnings.clear()
            mxfp4_asm._runtime_artifact_warnings.update(warnings)


def test_mxfp4_asm_table_filters_before_applying_path_priority(tmp_path):
    """An unrelated high-priority row cannot hide a lower-priority ASM row."""
    high = tmp_path / "high.csv"
    low = tmp_path / "low.csv"
    shape = {"gfx": "gfx950", "cu_num": 256, "M": 64, "N": 128, "K": 256}
    symbol = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"
    _write_mxfp4_asm_table(
        high,
        [{**shape, "kernelName": "unapproved_implementation", "splitK": 0}],
    )
    _write_mxfp4_asm_table(
        low,
        [{**shape, "kernelName": symbol, "splitK": 0}],
    )
    mxfp4_asm.clear_caches()
    assert mxfp4_asm.lookup(
        64, 128, 256, arch="gfx950", cu_num=256, paths=[str(high), str(low)]
    ) == (symbol, 0)

    faster_looking = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"
    _write_mxfp4_asm_table(
        high,
        [{**shape, "kernelName": faster_looking, "splitK": 0}],
    )
    mxfp4_asm.clear_caches()
    assert mxfp4_asm.lookup(
        64, 128, 256, arch="gfx950", cu_num=256, paths=[str(high), str(low)]
    ) == (faster_looking, 0)


def test_mxfp4_asm_table_prefers_exact_then_fine_then_coarse_m(tmp_path):
    table = tmp_path / "padded.csv"
    exact = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"
    fine = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"
    coarse = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_96x128E"
    base = {"gfx": "gfx950", "cu_num": 256, "N": 128, "K": 256, "splitK": 0}
    _write_mxfp4_asm_table(
        table,
        [
            {**base, "M": 257, "kernelName": exact},
            {**base, "M": 288, "kernelName": fine},
            {**base, "M": 512, "kernelName": coarse},
        ],
    )
    mxfp4_asm.clear_caches()
    paths = [str(table)]
    assert mxfp4_asm.lookup(
        257, 128, 256, arch="gfx950", cu_num=256, paths=paths
    ) == (exact, 0)

    _write_mxfp4_asm_table(
        table,
        [
            {**base, "M": 288, "kernelName": fine},
            {**base, "M": 512, "kernelName": coarse},
        ],
    )
    mxfp4_asm.clear_caches()
    assert mxfp4_asm.lookup(
        257, 128, 256, arch="gfx950", cu_num=256, paths=paths
    ) == (fine, 0)

    _write_mxfp4_asm_table(
        table,
        [{**base, "M": 512, "kernelName": coarse}],
    )
    mxfp4_asm.clear_caches()
    assert mxfp4_asm.lookup(
        257, 128, 256, arch="gfx950", cu_num=256, paths=paths
    ) == (coarse, 0)


@pytest.mark.parametrize(
    "m,n,level,expected",
    [
        (256, 4096, 0, 256),
        (257, 4096, 0, 288),
        (1024, 4096, 0, 1024),
        (1025, 4096, 0, 1088),
        (4096, 4096, 0, 4096),
        (4097, 4096, 0, 4224),
        (8193, 4096, 1, 16384),
        (8193, 4097, 1, 8192),
    ],
)
def test_mxfp4_asm_padded_m_matches_tuned_lookup(m, n, level, expected):
    assert mxfp4_asm.padded_m(m, n, level) == expected


def test_mxfp4_asm_table_rejects_ambiguous_key_and_wrong_libtype(tmp_path):
    duplicate = tmp_path / "duplicate.csv"
    tagged = tmp_path / "tagged.csv"
    shape = {"gfx": "gfx950", "cu_num": 256, "M": 64, "N": 128, "K": 256}
    first = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"
    second = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"
    _write_mxfp4_asm_table(
        duplicate,
        [
            {**shape, "kernelName": first, "splitK": 0},
            {**shape, "kernelName": second, "splitK": 0},
        ],
    )
    _write_mxfp4_asm_table(
        tagged,
        [{**shape, "kernelName": first, "splitK": 0, "libtype": "other"}],
        include_libtype=True,
    )
    mxfp4_asm.clear_caches()
    assert mxfp4_asm.lookup(
        64, 128, 256, arch="gfx950", cu_num=256, paths=[str(duplicate)]
    ) is None
    assert mxfp4_asm.lookup(
        64, 128, 256, arch="gfx950", cu_num=256, paths=[str(tagged)]
    ) is None


def test_mxfp4_asm_legacy_table_is_gfx950_only(tmp_path):
    table = tmp_path / "legacy.csv"
    symbol = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"
    _write_mxfp4_asm_table(
        table,
        [
            {
                "cu_num": 256,
                "M": 64,
                "N": 128,
                "K": 256,
                "kernelName": symbol,
                "splitK": 0,
            }
        ],
        include_gfx=False,
    )
    mxfp4_asm.clear_caches()
    assert mxfp4_asm.lookup(
        64, 128, 256, arch="gfx950", cu_num=256, paths=[str(table)]
    ) == (symbol, 0)
    assert mxfp4_asm.lookup(
        64, 128, 256, arch="gfx942", cu_num=256, paths=[str(table)]
    ) is None


def test_mxfp4_asm_artifact_identity_tracks_code_object_changes(
    tmp_path, monkeypatch
):
    symbol = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"
    manifest = tmp_path / "f4gemm_bf16_per1x32Fp4.csv"
    code_object = tmp_path / "kernel.co"
    manifest.write_text(
        "tile_M,tile_N,splitK,bpreshuffle,knl_name,co_name\n"
        f"64,128,0,1,{symbol},{code_object.name}\n"
    )
    code_object.write_bytes(b"first-build")
    monkeypatch.setattr(mxfp4_asm, "_manifest_path", lambda _arch: manifest)
    mxfp4_asm.clear_caches()
    first = mxfp4_asm.kernel_artifact(symbol, "gfx950")
    assert first is not None

    code_object.write_bytes(b"second-build-with-a-different-size")
    second = mxfp4_asm.kernel_artifact(symbol, "gfx950")
    assert second is not None
    assert second["code_object_sha256"] != first["code_object_sha256"]


def test_mxfp4_asm_runtime_snapshot_tracks_all_live_files(
    tmp_path, monkeypatch, caplog, isolated_mxfp4_asm_artifact_baselines
):
    """Table changes reload, while changed loaded artifacts require restart."""
    from aiter.jit.utils import chip_info

    first_symbol = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"
    second_symbol = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E"
    table = tmp_path / "tuned.csv"
    manifest = tmp_path / "f4gemm_bf16_per1x32Fp4.csv"
    first_code = tmp_path / "first.co"
    second_code = tmp_path / "second.co"
    replacement_code = tmp_path / "replacement-longer-name.co"
    first_code.write_bytes(b"first")
    second_code.write_bytes(b"second")
    replacement_code.write_bytes(b"replacement")
    manifest.write_text(
        "tile_M,tile_N,splitK,bpreshuffle,knl_name,co_name\n"
        f"64,128,1,1,{first_symbol},{first_code.name}\n"
        f"128,256,1,1,{second_symbol},{second_code.name}\n"
    )
    shape = {"gfx": "gfx950", "cu_num": 256, "M": 64, "N": 128, "K": 256}
    _write_mxfp4_asm_table(
        table,
        [{**shape, "kernelName": first_symbol, "splitK": 0}],
    )
    monkeypatch.setenv(mxfp4_asm.TUNED_CONFIG_ENV, str(table))
    manifest_lookups = 0

    def _manifest_path(_arch):
        nonlocal manifest_lookups
        manifest_lookups += 1
        return manifest

    monkeypatch.setattr(mxfp4_asm, "_manifest_path", _manifest_path)
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    mxfp4_asm.clear_caches()

    first = mxfp4_asm.runtime_snapshot(64, 128, 256)
    assert first.config == (first_symbol, 0)
    assert mxfp4_asm.snapshot_identity(first)["code_object"] == first_code.name

    # A hot dispatch must reuse the manifest captured for the registry token
    # when it validates the selected code object. It still stats the manifest
    # and code object on every call, but must not rediscover the search path.
    manifest_lookups = 0
    unchanged = mxfp4_asm.runtime_snapshot(64, 128, 256)
    assert unchanged.validation_token == first.validation_token
    assert manifest_lookups == 1

    _write_mxfp4_asm_table(
        table,
        [{**shape, "kernelName": second_symbol, "splitK": 0}],
    )
    second = mxfp4_asm.runtime_snapshot(64, 128, 256)
    assert second.config == (second_symbol, 0)
    assert second.validation_token != first.validation_token

    _write_mxfp4_asm_table(
        table,
        [{**shape, "kernelName": second_symbol, "splitK": 1}],
    )
    # Ensure the filesystem signature changes even on filesystems whose
    # sub-second timestamp granularity cannot distinguish the two rewrites.
    with table.open("a") as output:
        output.write("\n")
    split_changed = mxfp4_asm.runtime_snapshot(64, 128, 256)
    assert split_changed.config == (second_symbol, 1)
    assert split_changed.validation_token != second.validation_token

    manifest.write_text(
        "tile_M,tile_N,splitK,bpreshuffle,knl_name,co_name\n"
        f"128,256,1,1,{second_symbol},{replacement_code.name}\n"
    )
    with caplog.at_level("WARNING", logger=mxfp4_asm.__name__):
        third = mxfp4_asm.runtime_snapshot(64, 128, 256)
    assert third.config is None
    assert mxfp4_asm.snapshot_identity(third) is None
    assert third.validation_token != split_changed.validation_token
    assert "until process restart" in caplog.text

    replacement_code.write_bytes(b"replacement-with-different-size")
    fourth = mxfp4_asm.runtime_snapshot(64, 128, 256)
    assert fourth.config is None
    assert fourth.validation_token != third.validation_token
    assert mxfp4_asm.snapshot_identity(fourth) is None

    mxfp4_asm.clear_caches()
    after_clear = mxfp4_asm.runtime_snapshot(64, 128, 256)
    assert after_clear.config is None
    assert mxfp4_asm.snapshot_identity(after_clear) is None


def test_mxfp4_asm_runtime_snapshot_uses_one_manifest_generation(
    tmp_path, monkeypatch, isolated_mxfp4_asm_artifact_baselines
):
    """Table admission and launch identity use the same captured manifest."""
    from aiter.jit.utils import chip_info

    symbol = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"
    table = tmp_path / "tuned.csv"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_manifest = first_dir / "f4gemm_bf16_per1x32Fp4.csv"
    second_manifest = second_dir / first_manifest.name
    first_code = first_dir / "kernel.co"
    second_code = second_dir / "kernel.co"
    first_code.write_bytes(b"no-split-k")
    second_code.write_bytes(b"split-k-capable")
    first_manifest.write_text(
        "tile_M,tile_N,splitK,bpreshuffle,knl_name,co_name\n"
        f"64,128,0,1,{symbol},{first_code.name}\n"
    )
    second_manifest.write_text(
        "tile_M,tile_N,splitK,bpreshuffle,knl_name,co_name\n"
        f"64,128,1,1,{symbol},{second_code.name}\n"
    )
    _write_mxfp4_asm_table(
        table,
        [
            {
                "gfx": "gfx950",
                "cu_num": 256,
                "M": 64,
                "N": 128,
                "K": 256,
                "kernelName": symbol,
                "splitK": 1,
            }
        ],
    )
    manifest_calls = 0

    def _changing_manifest(_arch):
        nonlocal manifest_calls
        manifest_calls += 1
        return first_manifest if manifest_calls == 1 else second_manifest

    monkeypatch.setenv(mxfp4_asm.TUNED_CONFIG_ENV, str(table))
    monkeypatch.setattr(mxfp4_asm, "_manifest_path", _changing_manifest)
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    mxfp4_asm.clear_caches()

    snapshot = mxfp4_asm.runtime_snapshot(64, 128, 256)
    assert snapshot.config is None
    assert snapshot.identity_items is None
    assert manifest_calls == 1

    mxfp4_asm.clear_caches()
    manifest_calls = 0
    assert len(mxfp4_asm.table_fingerprint()) == 16
    assert manifest_calls == 1


def test_mxfp4_asm_runtime_snapshot_retries_initially_missing_artifact(
    tmp_path, monkeypatch, isolated_mxfp4_asm_artifact_baselines
):
    """Installing an artifact before its first valid use refreshes the table view."""
    from aiter.jit.utils import chip_info

    symbol = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"
    table = tmp_path / "tuned.csv"
    manifest = tmp_path / "f4gemm_bf16_per1x32Fp4.csv"
    code_object = tmp_path / "kernel.co"
    shape = {"gfx": "gfx950", "cu_num": 256, "M": 64, "N": 128, "K": 256}
    _write_mxfp4_asm_table(
        table,
        [{**shape, "kernelName": symbol, "splitK": 0}],
    )
    monkeypatch.setenv(mxfp4_asm.TUNED_CONFIG_ENV, str(table))
    monkeypatch.setattr(
        mxfp4_asm,
        "_manifest_path",
        lambda _arch: manifest if manifest.is_file() else None,
    )
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    mxfp4_asm.clear_caches()

    missing = mxfp4_asm.runtime_snapshot(64, 128, 256)
    assert missing.config is None

    code_object.write_bytes(b"installed-before-first-use")
    manifest.write_text(
        "tile_M,tile_N,splitK,bpreshuffle,knl_name,co_name\n"
        f"64,128,1,1,{symbol},{code_object.name}\n"
    )
    available = mxfp4_asm.runtime_snapshot(64, 128, 256)
    assert available.config == (symbol, 0)
    assert mxfp4_asm.snapshot_identity(available)["code_object"] == code_object.name
    assert available.validation_token != missing.validation_token


def test_mxfp4_asm_launch_uses_direct_api(monkeypatch):
    """The named ASM backend must call the direct ASM API with its exact config."""
    from aiter.ops import gemm_op_a4w4
    from lumen.ops.quantize import linear as linear_mod
    from lumen.ops.quantize import ops as quantize_ops

    M, N, K = 33, 16, 128
    a = torch.empty((M, K // 2), dtype=torch.uint8)
    w = torch.empty((N, K // 2), dtype=torch.uint8)
    scale_a = torch.empty((M, K // 32), dtype=torch.uint8)
    scale_w = torch.empty((N, K // 32), dtype=torch.uint8)
    calls = []

    def _direct_asm(a_arg, w_arg, sa_arg, sw_arg, out, **kwargs):
        calls.append((a_arg, w_arg, sa_arg, sw_arg, out, kwargs))

    def _generic_dispatch(*_args, **_kwargs):
        pytest.fail("MXFP4 ASM path called AITER's generic A4W4 dispatcher")

    asm_symbol = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_asm_config",
        lambda *_shape: pytest.fail("re-read ASM config after dispatch validation"),
    )
    monkeypatch.setattr(quantize_ops, "triton_arch", lambda: "gfx950")
    monkeypatch.setattr(linear_mod, "_shuffle_mxfp4_weight", lambda x, arch=None: x)
    monkeypatch.setattr(linear_mod, "_pad_and_swizzle_mxfp4_scale", lambda x, *_args: x)
    monkeypatch.setattr(gemm_op_a4w4, "gemm_a4w4_asm", _direct_asm)
    monkeypatch.setattr(gemm_op_a4w4, "gemm_a4w4", _generic_dispatch)
    result = _gemm_mxfp4_aiter_asm(
        a, w, scale_a, scale_w, asm_config=(asm_symbol, 3)
    )

    assert result.shape == (M, N)
    assert len(calls) == 1
    assert calls[0][4].shape == (64, N)
    assert calls[0][5] == {
        "kernelName": asm_symbol,
        "bpreshuffle": True,
        "log2_k_split": 3,
    }


@pytest.mark.parametrize(
    "rows,cols", [(2048, 128), (300, 128), (2048, 448)],
    ids=["aligned", "odd_rows", "wide_k"],
)
def test_mxfp4_asm_scale_pad_and_swizzle_roundtrip(rows, cols):
    """Padded+swizzled scales must round-trip, and keep the shape ASM indexes.

    ``shuffle_scale_gemm`` natively returns ``(rows // 32, cols * 32)``; handing
    that view to the ASM kernel reads out of bounds, so the helper has to fold it
    back to the padded 2D shape.
    """
    _require_mxfp4_dtype()
    from aiter.ops.triton.utils._triton.arch_info import get_arch
    from aiter.ops.triton.utils.shuffle import unshuffle_scale_gemm

    arch = get_arch()
    tiling = _MXFP4_SCALE_SHUFFLE_TILING.get(arch)
    if arch != "gfx950" or tiling is None:
        pytest.skip(f"scale swizzle round-trip is gfx950-only, running on {arch}")

    scale = torch.randint(100, 200, (rows, cols), dtype=torch.uint8, device="cuda")
    swizzled = _pad_and_swizzle_mxfp4_scale(scale, arch, tiling)

    rows_pad = -(-rows // 256) * 256
    cols_pad = -(-cols // 8) * 8
    assert swizzled.shape == (rows_pad, cols_pad)
    assert swizzled.is_contiguous()

    recovered = unshuffle_scale_gemm(
        swizzled.reshape(rows_pad // 32, cols_pad * 32), arch=arch
    )
    torch.testing.assert_close(recovered[:rows, :cols], scale, atol=0, rtol=0)
    # Padding must be zero-filled, not stale memory.
    assert recovered[rows:].eq(0).all()
    assert recovered[:, cols:].eq(0).all()


def test_mxfp4_backends_are_interchangeable():
    """Every backend admitted to autotune must agree with the existing path."""
    _require_mxfp4_dtype()
    torch.manual_seed(19)
    torch.cuda.manual_seed(19)

    # Non-square, and small enough to stay quick, but still 16-tileable.
    M, N, K = 2048, 4096, 14336
    a_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_hp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    a_fp4, a_scales = convert_to_mxfp4(a_hp, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False)
    w_fp4, w_scales = convert_to_mxfp4(w_hp, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False)

    try:
        ref = _gemm_mxfp4_aiter(a_fp4, w_fp4, a_scales, w_scales)
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"AITER MXFP4 GEMM unavailable: {e}")

    checked = 0
    if _mxfp4_preshuffle_supported(a_fp4, w_fp4):
        torch.testing.assert_close(
            _gemm_mxfp4_aiter_preshuffle(a_fp4, w_fp4, a_scales, w_scales),
            ref, atol=0, rtol=0,
        )
        checked += 1
    if _mxfp4_asm_supported(a_fp4, w_fp4):
        torch.testing.assert_close(
            _gemm_mxfp4_aiter_asm(a_fp4, w_fp4, a_scales, w_scales),
            ref, atol=0, rtol=0,
        )
        checked += 1
    if flydsl_mxfp4.available():
        for config in flydsl_mxfp4.supported_configs(M, N, K):
            torch.testing.assert_close(
                _gemm_mxfp4_flydsl(
                    config.name, a_fp4, w_fp4, a_scales, w_scales
                ),
                ref,
                atol=0.1,
                rtol=0.1,
            )
            checked += 1
    assert checked, "no alternative backend was available to compare against"


@pytest.mark.parametrize("rows,cols", [(2048, 3072), (1024, 4096), (512, 1024)])
def test_hadamard_quant_reads_transpose_without_materialising(rows, cols):
    """Quantizing x^T as a view must equal quantizing a materialised x^T.

    The wgrad path relies on this to skip two large .contiguous() copies. It
    holds only because stochastic rounding draws from tile-local indices and a
    fixed philox offset rather than from the address, so the strided read sees
    the same random stream. If that ever changes this test catches it, since a
    silently different gradient is far worse than a slow one.
    """
    _require_mxfp4_dtype()
    from lumen.ops.quantize.linear import _get_mxfp4_rht_sign

    torch.manual_seed(13)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    sign = _get_mxfp4_rht_sign(x.device)
    seed, offset = 4242, 99

    view_t = x.t()
    assert not view_t.is_contiguous()

    got_fp4, got_scale = hadamard_quant_mxfp4(
        view_t, sign, block_size=MXFP4_BLOCK_SIZE, g=16, use_sr=True,
        philox_seed=seed, philox_offset=offset,
    )
    ref_fp4, ref_scale = hadamard_quant_mxfp4(
        view_t.contiguous(), sign, block_size=MXFP4_BLOCK_SIZE, g=16, use_sr=True,
        philox_seed=seed, philox_offset=offset,
    )

    torch.testing.assert_close(got_fp4, ref_fp4, atol=0, rtol=0)
    torch.testing.assert_close(got_scale, ref_scale, atol=0, rtol=0)


@pytest.mark.parametrize("m,n,k", [(1024, 768, 512), (2048, 1024, 1024), (512, 512, 256)])
def test_mxfp4_backward_gradients_track_the_bf16_reference(m, n, k):
    """Both gradients must stay close to the BF16 reference.

    The wgrad path feeds its two operands through different routes — the
    gradient as a strided view of grad_flat, the activation through the fused
    dequant+transpose kernel — and a mix-up between them still produces
    plausibly-shaped output. Measured SNR sits near 12.9 dB (dW) and 13.7 dB
    (dX); the floor here is well under that but far above the near-zero SNR a
    transposed or mismatched operand would give.
    """
    _require_mxfp4_dtype()
    from lumen.ops.quantize.linear import quantized_linear

    torch.manual_seed(0)
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_(True)
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    grad_out = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.1

    quantized_linear(x, w, scaling_type="mxfp4").backward(grad_out)
    torch.nn.functional.linear(x_ref, w_ref).backward(grad_out)

    assert w.grad is not None, "backward did not populate the weight grad"
    assert x.grad is not None, "backward did not populate the input grad"
    assert compute_snr(w_ref.grad, w.grad) > 11.0
    assert compute_snr(x_ref.grad, x.grad) > 11.0


@pytest.mark.parametrize("m", [1000, 513, 33])
def test_mxfp4_backward_ragged_m_stays_in_fp4(m):
    """A ragged token count must not change which arithmetic backward runs.

    M is seq x mbs: a last batch, a variable-length run and a MoE expert's
    token count are all ragged, so this is ordinary rather than exceptional.
    Requiring M % 32 sent the whole layer to BF16, and that path took DGrad
    from the BF16 master weight while WGrad went through the dequantized
    activation. Measured at M=1000 before this was fixed: dX 101.7 dB against
    the unquantized reference -- effectively exact, because it *was* the
    unquantized computation -- next to 13.7 dB for an aligned M=2048 on the
    same shapes. A last batch was getting a differently conditioned gradient
    from every other batch in the run, and nothing said so.

    Hence the upper bound: this asserts the gradient carries FP4 quantization
    error, not merely that it is close to correct.
    """
    _require_mxfp4_dtype()
    from lumen.ops.quantize.linear import quantized_linear

    torch.manual_seed(0)
    k, n = 512, 768
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_(True)
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    grad_out = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.1

    quantized_linear(x, w, scaling_type="mxfp4").backward(grad_out)
    torch.nn.functional.linear(x_ref, w_ref).backward(grad_out)

    assert x.grad.shape == x.shape, "padding rows were not sliced back off"
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()

    dx_snr = compute_snr(x_ref.grad, x.grad)
    dw_snr = compute_snr(w_ref.grad, w.grad)
    assert dx_snr > 11.0, f"dX is not a correct gradient ({dx_snr:.1f} dB)"
    assert dw_snr > 11.0, f"dW is not a correct gradient ({dw_snr:.1f} dB)"
    assert dx_snr < 40.0, (
        f"dX at {dx_snr:.1f} dB is too good to have gone through FP4 -- backward "
        "took this shape to BF16 against the master weight"
    )


def test_mxfp4_backward_bf16_fallback_uses_the_quantized_weight():
    """When backward does fall back, both gradients must describe one forward.

    The fallback dequantizes the saved activation for WGrad, so taking DGrad
    from ctx.weight_ref -- the untouched BF16 master -- made the two terms the
    gradient of two different functions. The forward ran FP4; DGrad has to see
    the weight the forward saw.
    """
    _require_mxfp4_dtype()
    from lumen.ops.quantize import linear as linear_mod

    torch.manual_seed(3)
    m, n, k = 256, 768, 512
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_(True)
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    grad_out = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.1

    out = linear_mod.quantized_linear(x, w, scaling_type="mxfp4")

    # Drive the fallback the way a kernel rejecting these operands would, which
    # is the only way in now that a ragged M is padded rather than refused.
    real_dispatch = linear_mod.gemm_mxfp4_dispatch

    def _reject(*args, **kwargs):
        raise RuntimeError("pretend this shape has no kernel")

    linear_mod.gemm_mxfp4_dispatch = _reject
    try:
        out.backward(grad_out)
    finally:
        linear_mod.gemm_mxfp4_dispatch = real_dispatch

    against_master = (grad_out.float() @ w.float())
    snr_master = compute_snr(against_master, x.grad)
    assert snr_master > 11.0, f"dX is not a correct gradient at all ({snr_master:.1f} dB)"
    assert snr_master < 40.0, (
        f"dX matches the unquantized master weight to {snr_master:.1f} dB, so DGrad "
        "bypassed the quantized weight the forward used"
    )


def test_mxfp4_backward_fallback_decodes_a_shuffled_quantized_weight(monkeypatch):
    """A rejected DGrad kernel must not make a shuffled cache use the master."""
    _require_mxfp4_dtype()
    from lumen.ops.quantize import linear as linear_mod
    from lumen.quantize import _mxfp4_cached_weight

    torch.manual_seed(31)
    m = n = k = 256
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_(True)
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    grad_out = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.1

    # Model a later micro-batch whose fprop and DGrad backends have already
    # selected the shuffled B layout.
    monkeypatch.setattr(linear_mod, "_mxfp4_can_fuse_b_shuffle", lambda *_args: True)
    owner = torch.nn.Module()
    weight_data, weight_scale = _mxfp4_cached_weight(
        owner, w, None, None, "mxfp4", None, 32, gemm_rows=m,
    )
    assert getattr(weight_data._mxfp4_wt_cached[0], "_mxfp4_data_shuffled", False)

    reject = False

    def _dispatch(a_fp4, w_fp4, *_args):
        if reject:
            raise RuntimeError("force DGrad fallback after a shuffled cache")
        return torch.zeros(
            (a_fp4.shape[0], w_fp4.shape[0]),
            device=a_fp4.device,
            dtype=torch.bfloat16,
        )

    monkeypatch.setattr(linear_mod, "gemm_mxfp4_dispatch", _dispatch)
    out = linear_mod.quantized_linear(
        x,
        w,
        scaling_type="mxfp4",
        block_size=32,
        fp8_weight_cache=weight_data,
        fp8_weight_scale=weight_scale,
    )
    reject = True
    out.backward(grad_out)

    against_master = grad_out.float() @ w.float()
    snr_master = compute_snr(against_master, x.grad)
    assert 11.0 < snr_master < 40.0, (
        f"DGrad matches the BF16 master to {snr_master:.1f} dB; the shuffled "
        "FP4 weight was not decoded for fallback"
    )


def test_mxfp4_ignores_a_pre_quantized_fp8_activation():
    """An FP8 activation cache must never reach the MXFP4 GEMM.

    The fused SwiGLU bridge hands the next GEMM a per-tensor FP8 activation to
    reuse. MXFP4 needs FP4 elements paired with E8M0 block scales, so the cache
    is unusable here; taking it anyway drops an IndexError on the scale's rank
    inside AITER's afp4wfp4 GEMM, several frames from the cause. The result must
    match the run that was given no cache at all.
    """
    _require_mxfp4_dtype()
    from lumen.ops.quantize.linear import quantized_linear
    from lumen.quantize.config import _get_float8_e4m3

    torch.manual_seed(0)
    m, n, k = 512, 512, 256
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.05
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02

    fp8_dtype = _get_float8_e4m3()
    fp8_max = torch.finfo(fp8_dtype).max
    scale = (x.abs().max().float() / fp8_max).reshape(1)
    x_fp8 = (x.float() / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)

    got = quantized_linear(x, w, scaling_type="mxfp4", pre_quantized_input=(x_fp8, scale))
    expected = quantized_linear(x, w, scaling_type="mxfp4")

    torch.testing.assert_close(got, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "n,k",
    [
        (12288, 4096),   # above the size gate, takes the wide-view path
        (4096, 12288),
        (3072, 1024),    # below the gate, falls back to AITER
        (48, 128),       # tiny, still has to agree
    ],
)
def test_mxfp4_weight_shuffle_matches_aiter(n, k):
    """The vectorized B-operand shuffle must be byte-identical to AITER's.

    It reinterprets the packed-fp4 bytes as int64 to let the copy vectorize, so
    any mistake in the reshape would silently feed the ASM kernel a wrong but
    plausible-looking weight layout.
    """
    from aiter.ops.shuffle import shuffle_weight

    torch.manual_seed(5)
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")

    ref = shuffle_weight(w, layout=(16, 16)).view(torch.uint8)
    got = _shuffle_mxfp4_weight(w).view(torch.uint8)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


def test_mxfp4_weight_shuffle_falls_back_when_wide_view_invalid():
    """Non-contiguous, unaligned, and gfx1250 operands must take AITER's path."""
    from aiter.ops.shuffle import shuffle_weight

    torch.manual_seed(6)
    n, kp = 4096, 4096  # 16 MiB, comfortably over the size gate
    w = torch.randint(0, 256, (n, kp), dtype=torch.uint8, device="cuda")
    ref = shuffle_weight(w, layout=(16, 16)).view(torch.uint8)

    # gfx1250 uses a different WMMA layout, so the wide view would be wrong.
    torch.testing.assert_close(
        _shuffle_mxfp4_weight(w, arch="gfx1250").view(torch.uint8), ref, atol=0, rtol=0
    )

    non_contig = torch.randint(
        0, 256, (n, kp * 2), dtype=torch.uint8, device="cuda"
    )[:, :kp]
    assert not non_contig.is_contiguous()
    torch.testing.assert_close(
        _shuffle_mxfp4_weight(non_contig).view(torch.uint8),
        shuffle_weight(non_contig, layout=(16, 16)).view(torch.uint8),
        atol=0, rtol=0,
    )

    assert n * kp >= _MXFP4_WIDE_SHUFFLE_MIN_BYTES


def test_mxfp4_aligned_scale_swizzle_does_not_alias_input():
    """The aligned fast path must still hand back storage of its own.

    It skips the padded copy that used to guarantee a fresh buffer, so an
    accidental view would let a caller's scale tensor be corrupted downstream.
    """
    _require_mxfp4_dtype()
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    arch = get_arch()
    tiling = _MXFP4_SCALE_SHUFFLE_TILING.get(arch)
    if arch != "gfx950" or tiling is None:
        pytest.skip(f"scale swizzle is gfx950-only, running on {arch}")

    torch.manual_seed(7)
    scale = torch.randint(0, 256, (512, 128), dtype=torch.uint8, device="cuda")
    before = scale.clone()

    out = _pad_and_swizzle_mxfp4_scale(scale, arch, tiling)
    assert out.data_ptr() != scale.data_ptr()

    out.fill_(0)
    torch.testing.assert_close(scale, before, atol=0, rtol=0)


@pytest.mark.parametrize(
    "timings,expected",
    [
        (
            {"asm": 1.0, "flydsl_64x128x256": 1.0 / 1.05},
            "flydsl_64x128x256",
        ),
        ({"asm": 100.0, "flydsl_64x128x256": 97.0}, "asm"),
        ({"asm": 100.0, "flydsl_64x128x256": 94.0}, "flydsl_64x128x256"),
        (
            {
                "asm": 100.0,
                "shuffled": 90.0,
                "flydsl_64x128x256": 94.0,
            },
            "flydsl_64x128x256",
        ),
        ({"shuffled": 90.0, "flydsl_64x128x256": 94.0}, "shuffled"),
        ({"flydsl_64x128x256": 94.0}, "asm"),
        ({"asm": 100.0, "unknown_challenger": 50.0}, "asm"),
    ],
)
def test_mxfp4_autotune_protects_asm_incumbent(timings, expected):
    """Only a FlyDSL result at least 5% faster may replace working ASM."""
    winner, _reason = mxfp4_autotune._select_winner(timings, incumbent="asm")
    assert winner == expected


def test_mxfp4_autotune_winner_is_independent_of_candidate_order():
    """Persisted per-shape decisions must not depend on registration order."""
    timings = [
        ("asm", 100.0),
        ("shuffled", 90.0),
        ("flydsl_64x128x256", 94.0),
        ("flydsl_64x256x256", 96.0),
    ]
    forward = mxfp4_autotune._select_winner(dict(timings), incumbent="asm")
    reverse = mxfp4_autotune._select_winner(
        dict(reversed(timings)), incumbent="asm"
    )
    assert forward[0] == reverse[0] == "flydsl_64x128x256"


def test_mxfp4_autotune_validates_flydsl_before_timing():
    """A numerically bad FlyDSL configuration never enters the performance race."""
    reference = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    candidates = [
        ("asm", lambda: reference),
        ("flydsl_64x128x256", lambda: reference.clone()),
        ("flydsl_64x256x256", lambda: reference + 10),
    ]
    admitted, validation = mxfp4_autotune._validate_flydsl_candidates(
        (8, 8, 256), candidates, incumbent="asm"
    )
    assert [name for name, _fn in admitted] == ["asm", "flydsl_64x128x256"]
    assert validation["flydsl_64x128x256"].startswith("passed")
    assert validation["flydsl_64x256x256"].startswith("rejected")


def test_mxfp4_autotune_validates_every_output_element():
    """A bad interior value omitted by the former sample is still rejected."""
    reference = torch.zeros((64, 64), dtype=torch.float32)
    bad = reference.clone()
    # Row/column 8 was not part of the former edge-and-17-point-grid sample.
    bad[8, 8] = 10
    admitted, validation = mxfp4_autotune._validate_flydsl_candidates(
        (64, 64, 256),
        [
            ("asm", lambda: reference),
            ("flydsl_64x128x256", lambda: bad),
        ],
        incumbent="asm",
    )
    assert [name for name, _fn in admitted] == ["asm"]
    assert validation["flydsl_64x128x256"].startswith("rejected")


def test_mxfp4_autotune_never_selects_flydsl_without_a_reference(monkeypatch):
    """A sole FlyDSL candidate cannot enter dispatch without profile evidence."""
    key = (64, 128, 256)
    name = "flydsl_64x128x256"
    monkeypatch.setattr(mxfp4_autotune, "AUTOTUNE_ENABLED", True)
    monkeypatch.setattr(mxfp4_autotune, "_capturing", lambda: False)
    mxfp4_autotune.clear()
    try:
        with pytest.raises(RuntimeError, match="cannot be selected"):
            mxfp4_autotune.pick_backend(
                key,
                [(name, lambda: torch.zeros((2, 2)))],
                fallback=name,
                incumbent=name,
            )
        with pytest.raises(RuntimeError, match="no existing backend"):
            mxfp4_autotune._measure(
                key,
                [
                    (name, lambda: torch.zeros((2, 2))),
                    (
                        "flydsl_64x256x256",
                        lambda: torch.zeros((2, 2)),
                    ),
                ],
                incumbent=name,
            )
    finally:
        mxfp4_autotune.clear()


def test_mxfp4_autotune_does_not_cache_when_asm_measurement_fails(monkeypatch):
    """A missing ASM timing cannot permanently promote FlyDSL or a fallback."""
    key = (64, 128, 256)
    monkeypatch.setattr(mxfp4_autotune, "AUTOTUNE_ENABLED", True)
    monkeypatch.setattr(mxfp4_autotune, "_capturing", lambda: False)
    monkeypatch.setattr(
        mxfp4_autotune,
        "_measure",
        lambda *_args, **_kwargs: (
            "shuffled",
            {"shuffled": 1.0, "flydsl_64x128x256": 0.9},
            "ASM measurement failed",
            {"flydsl_64x128x256": "passed against shuffled"},
        ),
    )
    mxfp4_autotune.clear()
    try:
        assert (
            mxfp4_autotune.pick_backend(
                key,
                [
                    ("asm", lambda: None),
                    ("flydsl_64x128x256", lambda: None),
                    ("shuffled", lambda: None),
                ],
                fallback="asm",
                incumbent="asm",
            )
            == "shuffled"
        )
        assert mxfp4_autotune.cached(key) is None
    finally:
        mxfp4_autotune.clear()


def test_mxfp4_autotune_rejects_when_every_profile_candidate_fails(monkeypatch):
    """A failed label must not be returned as though measurement succeeded."""
    key = (64, 128, 256)

    def fail():
        raise RuntimeError("injected profile failure")

    monkeypatch.setattr(mxfp4_autotune, "_WARMUP_ITERS", 1)
    with pytest.raises(RuntimeError, match="no backend survived profile warmup"):
        mxfp4_autotune._measure(
            key,
            [("asm", fail), ("plain", fail)],
            incumbent="asm",
        )


def test_mxfp4_autotune_profile_records_exact_backend_identities(monkeypatch):
    """The cache keeps the concrete ASM symbol and FlyDSL tile behind labels."""
    key = (64, 128, 256)
    flydsl_name = "flydsl_64x128x256"
    asm_symbol = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E"
    asm_identity = mxfp4_asm.identity((asm_symbol, 0), "gfx950")
    assert asm_identity is not None
    identities = {
        "asm": asm_identity,
        flydsl_name: flydsl_mxfp4.config_identity(
            flydsl_mxfp4.get_config(flydsl_name)
        ),
    }
    monkeypatch.setattr(mxfp4_autotune, "AUTOTUNE_ENABLED", True)
    monkeypatch.setattr(mxfp4_autotune, "_capturing", lambda: False)
    monkeypatch.setattr(
        mxfp4_autotune,
        "_measure",
        lambda *_args, **_kwargs: (
            "asm",
            {"asm": 1.0, flydsl_name: 0.97},
            "FlyDSL challenger did not clear margin",
            {flydsl_name: "passed against asm"},
        ),
    )
    mxfp4_autotune.clear()
    try:
        chosen = mxfp4_autotune.pick_backend(
            key,
            [("asm", lambda: None), (flydsl_name, lambda: None)],
            fallback="asm",
            incumbent="asm",
            identities=identities,
        )
        assert chosen == "asm"
        assert mxfp4_autotune._profiles[key]["identities"] == identities
        assert mxfp4_autotune._profiles[key]["validation"] == {
            flydsl_name: "passed against asm"
        }
        assert mxfp4_autotune.cached_profile_supports(
            key,
            "asm",
            expected_identities={"asm": identities["asm"]},
            required_incumbent="asm",
        )
    finally:
        mxfp4_autotune.clear()


def test_mxfp4_flydsl_shape_gate_matches_scale_layout_contract(monkeypatch):
    """The vendored scale reader requires complete 256-value K chunks."""
    monkeypatch.setattr(flydsl_mxfp4, "_ENABLED", True)
    assert not flydsl_mxfp4.supported_configs(64, 128, 128)
    assert not flydsl_mxfp4.supported_configs(64, 128, 384)
    assert {
        config.name for config in flydsl_mxfp4.supported_configs(64, 128, 256)
    } == {
        "flydsl_32x128x256",
        "flydsl_64x128x128",
        "flydsl_64x128x256",
    }


def test_mxfp4_flydsl_profiles_every_legal_upstream_config(monkeypatch):
    """M heuristics must not remove a legal contender before measurement."""
    monkeypatch.setattr(flydsl_mxfp4, "_ENABLED", True)
    monkeypatch.setattr(flydsl_mxfp4, "FLYDSL_RUNTIME_VERSION", "0.3.2")
    four_wave = "flydsl_4wave_256x256x256"
    expected = set(flydsl_mxfp4.backend_names()) - {four_wave}
    for M in (1, 32, 64, 128, 511, 512, 5133, 8192):
        actual = {
            config.name
            for config in flydsl_mxfp4.supported_configs(M, 8192, 8192)
        }
        assert actual == (expected | {four_wave} if M % 256 == 0 else expected)


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((256, 256, 512), False),
        ((256, 256, 768), False),
        ((256, 256, 1024), True),
        ((256, 256, 1280), False),
        ((256, 256, 1536), True),
        ((255, 256, 1024), False),
        ((256, 255, 1024), False),
        ((256, 256, 1152), False),
    ],
)
def test_mxfp4_4wave_legality_is_fail_closed(monkeypatch, shape, expected):
    """The fixed 256x256 kernel only admits shapes its pipeline can execute."""
    name = "flydsl_4wave_256x256x256"
    monkeypatch.setattr(flydsl_mxfp4, "_ENABLED", True)
    monkeypatch.setattr(flydsl_mxfp4, "FLYDSL_RUNTIME_VERSION", "0.3.2")
    names = {config.name for config in flydsl_mxfp4.supported_configs(*shape)}
    assert (name in names) is expected


def test_mxfp4_4wave_rejects_old_flydsl_runtime(monkeypatch):
    """FlyDSL 0.2.4 lacks APIs required by the 4-wave source."""
    name = "flydsl_4wave_256x256x256"
    monkeypatch.setattr(flydsl_mxfp4, "_ENABLED", True)
    monkeypatch.setattr(flydsl_mxfp4, "FLYDSL_RUNTIME_VERSION", "0.2.4")

    names = {
        config.name
        for config in flydsl_mxfp4.supported_configs(256, 256, 1024)
    }
    assert name not in names
    assert "flydsl_64x128x256" in names


@pytest.mark.parametrize(
    "version,expected",
    [
        ("0.3.1", False),
        ("0.3.2", True),
        ("0.3.2+local", True),
        ("0.4.0.dev1", True),
        ("unavailable", False),
        ("0.3", False),
    ],
)
def test_mxfp4_4wave_runtime_version_helper(version, expected):
    assert flydsl_mxfp4.four_wave_runtime_supported(version) is expected


def test_mxfp4_4wave_has_independent_identity_and_registration(monkeypatch):
    """The 4-wave candidate cannot reuse another FlyDSL kernel's profile."""
    from lumen.ops.quantize import linear as linear_mod

    monkeypatch.setattr(flydsl_mxfp4, "FLYDSL_RUNTIME_VERSION", "0.3.2")
    generic = flydsl_mxfp4.get_config("flydsl_64x256x256")
    four_wave = flydsl_mxfp4.get_config("flydsl_4wave_256x256x256")
    generic_identity = flydsl_mxfp4.config_identity(generic)
    four_wave_identity = flydsl_mxfp4.config_identity(four_wave)

    assert four_wave.name in flydsl_mxfp4.backend_names()
    assert four_wave.name in linear_mod._MXFP4_BACKENDS
    assert linear_mod._MXFP4_BACKEND_KIND[four_wave.name].value == "flydsl"
    assert four_wave_identity["kernel_family"] == "mxfp4_4wave"
    assert four_wave_identity["source_revision"] == flydsl_mxfp4.FLYDSL_KERNEL_REVISION
    assert four_wave_identity["source_sha256"] != generic_identity["source_sha256"]
    assert four_wave_identity["runtime_version"] == "0.3.2"
    assert four_wave_identity["waves_per_eu"] == 1
    assert four_wave_identity["use_xcd_remap"] is True


def test_mxfp4_4wave_uses_shared_preshuffled_operands(monkeypatch):
    """The new candidate consumes the same B/scale layout as strict ASM."""
    from lumen.ops.quantize import linear as linear_mod
    from lumen.ops.quantize import ops as quantize_ops

    name = "flydsl_4wave_256x256x256"
    a = torch.empty((256, 512), dtype=torch.uint8)
    w = torch.empty((256, 512), dtype=torch.uint8)
    sa = torch.empty((256, 32), dtype=torch.uint8)
    sw = torch.empty((256, 32), dtype=torch.uint8)
    w_shuffled = torch.empty_like(w)
    sa_shuffled = torch.empty_like(sa)
    sw_shuffled = torch.empty_like(sw)
    seen = {}

    monkeypatch.setattr(quantize_ops, "triton_arch", lambda: "gfx950")
    monkeypatch.setattr(
        linear_mod, "_expand_2d_scale_to_1d", lambda scale, _shape: scale
    )
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_preshuffled_operands",
        lambda weight, scale, arch, tiling: (w_shuffled, sw_shuffled),
    )
    monkeypatch.setattr(
        linear_mod,
        "_pad_and_swizzle_mxfp4_scale",
        lambda scale, arch, tiling: sa_shuffled,
    )

    def _run(config, actual_a, actual_w, actual_sa, actual_sw):
        seen.update(
            config=config,
            a=actual_a,
            w=actual_w,
            sa=actual_sa,
            sw=actual_sw,
        )
        return "output"

    monkeypatch.setattr(flydsl_mxfp4, "run", _run)
    assert linear_mod._gemm_mxfp4_flydsl(name, a, w, sa, sw) == "output"
    assert seen["config"] == flydsl_mxfp4.get_config(name)
    assert seen["a"] is a
    assert seen["w"] is w_shuffled
    assert seen["sa"] is sa_shuffled
    assert seen["sw"] is sw_shuffled


def test_mxfp4_4wave_must_clear_protected_asm_margin():
    """Registration never statically promotes the measured 4-wave kernel."""
    name = "flydsl_4wave_256x256x256"
    assert mxfp4_autotune._select_winner(
        {"asm": 100.0, name: 96.0}, incumbent="asm"
    )[0] == "asm"
    assert mxfp4_autotune._select_winner(
        {"asm": 100.0, name: 100.0 / 1.05}, incumbent="asm"
    )[0] == name


def test_mxfp4_flydsl_identity_tracks_runtime_version(monkeypatch):
    """A compiler/runtime change must invalidate shape-profile evidence."""
    config = flydsl_mxfp4.get_config("flydsl_64x128x256")
    before_identity = flydsl_mxfp4.config_identity(config)
    before_fingerprint = flydsl_mxfp4.backend_fingerprint()

    monkeypatch.setattr(flydsl_mxfp4, "FLYDSL_RUNTIME_VERSION", "test-new-runtime")

    assert flydsl_mxfp4.config_identity(config) != before_identity
    assert flydsl_mxfp4.backend_fingerprint() != before_fingerprint
    assert flydsl_mxfp4.config_identity(config)["runtime_version"] == "test-new-runtime"


def test_mxfp4_flydsl_source_identity_is_cached():
    """The replay guard must not reread kernel source on every GEMM hot-path call."""
    flydsl_mxfp4._kernel_source_sha256.cache_clear()
    flydsl_mxfp4._wrapper_source_sha256.cache_clear()
    try:
        first = flydsl_mxfp4._kernel_source_sha256()
        second = flydsl_mxfp4._kernel_source_sha256()
        assert first == second
        assert flydsl_mxfp4._kernel_source_sha256.cache_info().hits == 1
        wrapper_first = flydsl_mxfp4._wrapper_source_sha256()
        wrapper_second = flydsl_mxfp4._wrapper_source_sha256()
        assert wrapper_first == wrapper_second
        assert flydsl_mxfp4._wrapper_source_sha256.cache_info().hits == 1
    finally:
        flydsl_mxfp4._kernel_source_sha256.cache_clear()
        flydsl_mxfp4._wrapper_source_sha256.cache_clear()


def test_mxfp4_flydsl_identity_tracks_wrapper_source(monkeypatch):
    """A wrapper-overhead change must invalidate measured shape winners."""
    config = flydsl_mxfp4.get_config("flydsl_4wave_256x256x256")
    before_identity = flydsl_mxfp4.config_identity(config)
    before_fingerprint = flydsl_mxfp4.backend_fingerprint()

    monkeypatch.setattr(
        flydsl_mxfp4, "_wrapper_source_sha256", lambda: "changed-wrapper"
    )

    assert flydsl_mxfp4.config_identity(config) != before_identity
    assert flydsl_mxfp4.backend_fingerprint() != before_fingerprint
    assert (
        flydsl_mxfp4.config_identity(config)["wrapper_source_sha256"]
        == "changed-wrapper"
    )


def test_mxfp4_4wave_identity_tracks_dma_intrinsic_env(monkeypatch):
    """Switching the generated DMA path must invalidate shape profiles."""
    four_wave = flydsl_mxfp4.get_config("flydsl_4wave_256x256x256")
    preshuffle = flydsl_mxfp4.get_config("flydsl_64x256x256")

    monkeypatch.setenv("FP4_DMA_INTRINSIC", "0")
    identity_0 = flydsl_mxfp4.config_identity(four_wave)
    fingerprint_0 = flydsl_mxfp4.backend_fingerprint()
    preshuffle_0 = flydsl_mxfp4.config_identity(preshuffle)

    monkeypatch.setenv("FP4_DMA_INTRINSIC", "1")
    identity_1 = flydsl_mxfp4.config_identity(four_wave)
    fingerprint_1 = flydsl_mxfp4.backend_fingerprint()
    preshuffle_1 = flydsl_mxfp4.config_identity(preshuffle)

    assert identity_0["fp4_dma_intrinsic"] == 0
    assert identity_1["fp4_dma_intrinsic"] == 1
    assert identity_0 != identity_1
    assert fingerprint_0 != fingerprint_1
    assert preshuffle_0["fp4_dma_intrinsic"] == "not_applicable"
    assert preshuffle_1 == preshuffle_0


def test_mxfp4_flydsl_missing_runtime_metadata_fails_closed(monkeypatch):
    """An unidentifiable FlyDSL compiler must never enter profiling or replay."""
    def _missing(_distribution):
        raise flydsl_mxfp4.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(flydsl_mxfp4.importlib.metadata, "version", _missing)
    assert flydsl_mxfp4._installed_runtime_version() == "unavailable"

    monkeypatch.setattr(flydsl_mxfp4, "_ENABLED", True)
    monkeypatch.setattr(flydsl_mxfp4, "FLYDSL_RUNTIME_VERSION", "unavailable")
    flydsl_mxfp4.available.cache_clear()
    try:
        assert not flydsl_mxfp4.available()
    finally:
        flydsl_mxfp4.available.cache_clear()


def test_mxfp4_flydsl_missing_rocm_linker_fails_closed(monkeypatch, tmp_path):
    """Direct callers must not discover failure only after JIT compilation."""
    missing_root = tmp_path / "missing-rocm"
    monkeypatch.setattr(flydsl_mxfp4, "_ENABLED", True)
    monkeypatch.setattr(flydsl_mxfp4, "FLYDSL_RUNTIME_VERSION", "0.3.2")
    monkeypatch.setattr(
        flydsl_mxfp4, "_configured_rocm_root", lambda: str(missing_root)
    )
    flydsl_mxfp4._rocm_toolchain_identity.cache_clear()
    flydsl_mxfp4.available.cache_clear()
    try:
        assert not flydsl_mxfp4.available()
    finally:
        flydsl_mxfp4.available.cache_clear()
        flydsl_mxfp4._rocm_toolchain_identity.cache_clear()


def test_mxfp4_autotune_picks_and_caches():
    """Autotune must return a legal backend and reuse it on later calls."""
    _require_mxfp4_dtype()
    if not mxfp4_autotune.AUTOTUNE_ENABLED:
        pytest.skip("LUMEN_MXFP4_AUTOTUNE=0 disables the path this test covers")

    torch.manual_seed(23)
    M, N, K = 2048, 4096, 14336
    a_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_hp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    a_fp4, a_scales = convert_to_mxfp4(a_hp, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False)
    w_fp4, w_scales = convert_to_mxfp4(w_hp, block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False)

    mxfp4_autotune.clear()
    key = (M, N, K)
    assert mxfp4_autotune.cached(key) is None

    try:
        chosen = _mxfp4_choose_backend(a_fp4, w_fp4, a_scales, w_scales)
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"AITER MXFP4 GEMM unavailable: {e}")

    assert chosen in ("asm", "shuffled", "plain", *flydsl_mxfp4.backend_names())
    assert mxfp4_autotune.cached(key) == chosen
    # Second call must not re-measure.
    assert _mxfp4_choose_backend(a_fp4, w_fp4, a_scales, w_scales) == chosen
    mxfp4_autotune.clear()


def test_mxfp4_autotune_decision_epoch_tracks_all_mutations(monkeypatch):
    """Resolved dispatch entries can invalidate on every choice/profile change."""
    original_path = mxfp4_autotune._CACHE_PATH
    mxfp4_autotune.clear()
    try:
        mxfp4_autotune._CACHE_PATH = ""
        before = mxfp4_autotune.decision_epoch()
        mxfp4_autotune._load_cache()
        assert mxfp4_autotune.decision_epoch() == before + 1

        protected_key = (1, 2, 3)
        before = mxfp4_autotune.decision_epoch()
        assert (
            mxfp4_autotune._remember_protected_asm(
                protected_key, {"asm": {"implementation": "asm"}}
            )
            == "asm"
        )
        assert mxfp4_autotune.decision_epoch() == before + 1

        before = mxfp4_autotune.decision_epoch()
        mxfp4_autotune.forget(protected_key)
        assert mxfp4_autotune.decision_epoch() == before + 1

        measured_key = (4, 5, 6)
        monkeypatch.setattr(mxfp4_autotune, "AUTOTUNE_ENABLED", True)
        monkeypatch.setattr(mxfp4_autotune, "_consensus_required", lambda: False)
        monkeypatch.setattr(mxfp4_autotune, "_capturing", lambda: False)
        monkeypatch.setattr(
            mxfp4_autotune,
            "_measure",
            lambda *_args, **_kwargs: (
                "plain",
                {"plain": 1.0, "shuffled": 2.0},
                "test",
                {},
            ),
        )
        before = mxfp4_autotune.decision_epoch()
        assert (
            mxfp4_autotune.pick_backend(
                measured_key,
                [("plain", lambda: None), ("shuffled", lambda: None)],
                fallback="plain",
                incumbent="plain",
                identities={"plain": {"implementation": "triton"}},
            )
            == "plain"
        )
        assert mxfp4_autotune.decision_epoch() == before + 1

        before = mxfp4_autotune.decision_epoch()
        mxfp4_autotune.clear()
        assert mxfp4_autotune.decision_epoch() == before + 1
    finally:
        mxfp4_autotune._CACHE_PATH = original_path
        mxfp4_autotune.clear()


def test_mxfp4_autotune_cache_roundtrip(tmp_path):
    """A persisted decision is reused, and one from another GPU is not."""
    cache = tmp_path / "autotune.json"
    key = (8192, 12288, 4096)

    mxfp4_autotune.clear()
    original_path = mxfp4_autotune._CACHE_PATH
    mxfp4_autotune._CACHE_PATH = str(cache)
    try:
        cache.write_text(json.dumps({
            "schema": mxfp4_autotune._CACHE_SCHEMA,
            "arch": mxfp4_autotune._arch(),
            "backends": mxfp4_autotune._backend_fingerprint(),
            "tuned_tables": mxfp4_autotune._tuned_table_fingerprint(),
            "choices": {"8192,12288,4096": "asm"},
        }))
        mxfp4_autotune._load_cache()
        assert mxfp4_autotune.cached(key) == "asm"

        # A cache measured elsewhere says nothing about this GPU.
        mxfp4_autotune.clear()
        cache.write_text(json.dumps({
            "schema": mxfp4_autotune._CACHE_SCHEMA,
            "arch": "gfx000-not-a-real-arch",
            "backends": mxfp4_autotune._backend_fingerprint(),
            "tuned_tables": mxfp4_autotune._tuned_table_fingerprint(),
            "choices": {"8192,12288,4096": "asm"},
        }))
        mxfp4_autotune._load_cache()
        assert mxfp4_autotune.cached(key) is None
    finally:
        mxfp4_autotune._CACHE_PATH = original_path
        mxfp4_autotune.clear()


def test_mxfp4_autotune_cache_rejects_old_schema_and_unknown_backend(tmp_path):
    """Only current, explicitly registered backend identities may be replayed."""
    cache = tmp_path / "autotune.json"
    key = (8192, 12288, 4096)
    original_path = mxfp4_autotune._CACHE_PATH
    mxfp4_autotune._CACHE_PATH = str(cache)
    try:
        common = {
            "arch": mxfp4_autotune._arch(),
            "backends": mxfp4_autotune._backend_fingerprint(),
            "tuned_tables": mxfp4_autotune._tuned_table_fingerprint(),
        }

        cache.write_text(
            json.dumps(
                {
                    **common,
                    "schema": mxfp4_autotune._CACHE_SCHEMA - 1,
                    "choices": {"8192,12288,4096": "asm"},
                }
            )
        )
        mxfp4_autotune.clear()
        mxfp4_autotune._load_cache()
        assert mxfp4_autotune.cached(key) is None

        cache.write_text(
            json.dumps(
                {
                    **common,
                    "schema": mxfp4_autotune._CACHE_SCHEMA,
                    "choices": {"8192,12288,4096": "unregistered_backend"},
                }
            )
        )
        mxfp4_autotune.clear()
        mxfp4_autotune._load_cache()
        assert mxfp4_autotune.cached(key) is None
    finally:
        mxfp4_autotune._CACHE_PATH = original_path
        mxfp4_autotune.clear()


def test_mxfp4_autotune_cache_tracks_flydsl_revision(tmp_path, monkeypatch):
    """A FlyDSL source/config change invalidates earlier shape measurements."""
    cache = tmp_path / "autotune.json"
    key = (8192, 12288, 4096)
    original_path = mxfp4_autotune._CACHE_PATH
    mxfp4_autotune._CACHE_PATH = str(cache)
    try:
        cache.write_text(
            json.dumps(
                {
                    "schema": mxfp4_autotune._CACHE_SCHEMA,
                    "arch": mxfp4_autotune._arch(),
                    "backends": mxfp4_autotune._backend_fingerprint(),
                    "tuned_tables": mxfp4_autotune._tuned_table_fingerprint(),
                    "choices": {"8192,12288,4096": "asm"},
                }
            )
        )
        monkeypatch.setattr(
            flydsl_mxfp4,
            "FLYDSL_KERNEL_REVISION",
            flydsl_mxfp4.FLYDSL_KERNEL_REVISION + "-changed",
        )
        mxfp4_autotune.clear()
        mxfp4_autotune._load_cache()
        assert mxfp4_autotune.cached(key) is None
    finally:
        mxfp4_autotune._CACHE_PATH = original_path
        mxfp4_autotune.clear()


def test_mxfp4_autotune_fingerprint_tracks_asm_registry_policy(monkeypatch):
    """Changing ASM admission/lookup code invalidates prior profile evidence."""
    before = mxfp4_autotune._backend_fingerprint()
    monkeypatch.setattr(
        mxfp4_asm,
        "registry_fingerprint",
        lambda: "asm-only-test-policy-change",
    )
    assert mxfp4_autotune._backend_fingerprint() != before


def test_mxfp4_autotune_cache_rejects_a_different_tuned_table(tmp_path):
    """``asm`` is only meaningful against the table it was measured with.

    The cache persists on a shared results directory and outlives the run that
    earned it. Replaying a decision with a changed ASM-only map could compare a
    FlyDSL tile with a different symbol/split policy. ``arch`` cannot catch
    this: the GPU has not changed.
    """
    cache = tmp_path / "autotune.json"
    table = tmp_path / "tuned.csv"
    table.write_text(
        "gfx,cu_num,M,N,K,kernelName,splitK\n"
        "gfx950,256,8192,12288,4096,"
        "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E,0\n"
    )
    key = (8192, 12288, 4096)

    original_path = mxfp4_autotune._CACHE_PATH
    original_env = os.environ.get(mxfp4_autotune.AITER_TUNED_CONFIG_ENV)
    mxfp4_autotune._CACHE_PATH = str(cache)
    mxfp4_autotune.clear()
    try:
        os.environ[mxfp4_autotune.AITER_TUNED_CONFIG_ENV] = str(table)
        cache.write_text(json.dumps({
            "schema": mxfp4_autotune._CACHE_SCHEMA,
            "arch": mxfp4_autotune._arch(),
            "backends": mxfp4_autotune._backend_fingerprint(),
            "tuned_tables": mxfp4_autotune._tuned_table_fingerprint(),
            "choices": {"8192,12288,4096": "asm"},
        }))
        mxfp4_autotune._load_cache()
        assert mxfp4_autotune.cached(key) == "asm"

        # Same file, different rows: a table tuned for another model.
        mxfp4_autotune.clear()
        table.write_text(
            "gfx,cu_num,M,N,K,kernelName,splitK\n"
            "gfx950,256,64,64,128,"
            "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E,0\n"
        )
        mxfp4_autotune._load_cache()
        assert mxfp4_autotune.cached(key) is None, "replayed asm against a different table"

        # And no table at all is likewise a different world.
        mxfp4_autotune.clear()
        os.environ.pop(mxfp4_autotune.AITER_TUNED_CONFIG_ENV, None)
        mxfp4_autotune._load_cache()
        assert mxfp4_autotune.cached(key) is None

        # Contents, not paths: the same table bind-mounted elsewhere still hits.
        mxfp4_autotune.clear()
        os.environ[mxfp4_autotune.AITER_TUNED_CONFIG_ENV] = str(table)
        cache.write_text(json.dumps({
            "schema": mxfp4_autotune._CACHE_SCHEMA,
            "arch": mxfp4_autotune._arch(),
            "backends": mxfp4_autotune._backend_fingerprint(),
            "tuned_tables": mxfp4_autotune._tuned_table_fingerprint(),
            "choices": {"8192,12288,4096": "asm"},
        }))
        moved = tmp_path / "elsewhere"
        moved.mkdir()
        (moved / "tuned.csv").write_text(table.read_text())
        os.environ[mxfp4_autotune.AITER_TUNED_CONFIG_ENV] = str(moved / "tuned.csv")
        mxfp4_autotune._load_cache()
        assert mxfp4_autotune.cached(key) == "asm"
    finally:
        if original_env is None:
            os.environ.pop(mxfp4_autotune.AITER_TUNED_CONFIG_ENV, None)
        else:
            os.environ[mxfp4_autotune.AITER_TUNED_CONFIG_ENV] = original_env
        mxfp4_autotune._CACHE_PATH = original_path
        mxfp4_autotune.clear()


def test_mxfp4_autotune_cache_write_cannot_tear_the_file(tmp_path, monkeypatch):
    """Every rank writes this path at exit, so a failed write must not destroy it."""
    cache = tmp_path / "autotune.json"
    good = json.dumps({
        "schema": mxfp4_autotune._CACHE_SCHEMA,
        "arch": mxfp4_autotune._arch(),
        "backends": mxfp4_autotune._backend_fingerprint(),
        "tuned_tables": mxfp4_autotune._tuned_table_fingerprint(),
        "choices": {"1,2,3": "plain"},
    })
    cache.write_text(good)

    original_path = mxfp4_autotune._CACHE_PATH
    mxfp4_autotune._CACHE_PATH = str(cache)
    mxfp4_autotune.clear()
    try:
        mxfp4_autotune._choice[(4, 5, 6)] = "asm"
        mxfp4_autotune._cache_dirty = True

        def _die(*args, **kwargs):
            raise OSError("disk full halfway through")

        monkeypatch.setattr(mxfp4_autotune.json, "dump", _die)
        mxfp4_autotune._save_cache()
        assert cache.read_text() == good, "a failed write truncated the existing cache"
        assert not list(tmp_path.glob("*.tmp")), "left a temp file behind"

        monkeypatch.undo()
        mxfp4_autotune._cache_dirty = True
        mxfp4_autotune._save_cache()
        written = json.loads(cache.read_text())
        assert written["choices"]["4,5,6"] == "asm"
        assert written["decision_scope"] == mxfp4_autotune._SINGLE_DEVICE_SCOPE
        assert written["profile_device_count"] == 1
        assert written["tuned_tables"] == mxfp4_autotune._tuned_table_fingerprint()
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        mxfp4_autotune._CACHE_PATH = original_path
        mxfp4_autotune.clear()


def test_mxfp4_choose_backend_rechecks_a_cached_decision(monkeypatch):
    """A cached name that these operands cannot legally run must be re-measured.

    The blob's table fingerprint covers a cache written elsewhere, but not a
    table narrowed inside the process, so the dispatcher checks legality before
    trusting the decision rather than after.
    """
    from lumen.ops.quantize import linear as linear_mod

    key = (2048, 4096, 14336)
    a = torch.empty((key[0], key[2] // 2), dtype=torch.uint8)
    w = torch.empty((key[1], key[2] // 2), dtype=torch.uint8)

    mxfp4_autotune.clear()
    linear_mod._mxfp4_legality_cache.clear()
    try:
        # Measured when the ASM kernels were reachable; they no longer are.
        mxfp4_autotune._choice[key] = "asm"
        monkeypatch.setattr(
            linear_mod,
            "_mxfp4_backend_legality",
            lambda k, x, y, *_rest: (False, False, ()),
        )
        monkeypatch.setattr(
            linear_mod.mxfp4_autotune,
            "pick_backend",
            lambda k, c, fallback=None, incumbent=None, identities=None: "plain",
        )

        assert linear_mod._mxfp4_choose_backend(a, w, None, None) == "plain"
        assert mxfp4_autotune.cached(key) is None, "kept a decision it could not honour"

        # A cached name that is still legal is used as-is, with no re-measure.
        mxfp4_autotune._choice[key] = "plain"
        monkeypatch.setattr(
            linear_mod.mxfp4_autotune,
            "pick_backend",
            lambda k, c, fallback=None, incumbent=None, identities=None: pytest.fail(
                "re-measured a legal cached decision"
            ),
        )
        assert linear_mod._mxfp4_choose_backend(a, w, None, None) == "plain"
    finally:
        mxfp4_autotune.clear()
        linear_mod._mxfp4_legality_cache.clear()


def test_mxfp4_resolved_backend_cache_revalidates_live_tokens(monkeypatch):
    """Hot replay skips deep profile checks but not file or decision changes."""
    from lumen.ops.quantize import linear as linear_mod

    key = (32, 64, 256)
    a = torch.empty((key[0], key[2] // 2), dtype=torch.uint8)
    w = torch.empty((key[1], key[2] // 2), dtype=torch.uint8)
    symbol = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"
    state = {
        "epoch": 7,
        "snapshot": mxfp4_asm.RuntimeSnapshot(
            "gfx950",
            256,
            (symbol, 0),
            ("files", 1),
            (
                ("implementation", "asm"),
                ("kernel_name", symbol),
                ("log2_k_split", 0),
                ("code_object_sha256", "first"),
            ),
            True,
        ),
    }
    profile_checks = []

    mxfp4_autotune.clear()
    linear_mod._mxfp4_legality_cache.clear()
    linear_mod._mxfp4_resolved_backend_cache.clear()
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_gemm_probed", True)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_gemm_fn", None)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_asm_ok", True)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_preshuffle_ok", False)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_flydsl_ok", False)
    monkeypatch.setattr(
        mxfp4_asm, "runtime_snapshot", lambda *_shape: state["snapshot"]
    )
    monkeypatch.setattr(mxfp4_autotune, "cached", lambda _key: "asm")
    monkeypatch.setattr(
        mxfp4_autotune, "decision_epoch", lambda: state["epoch"]
    )
    monkeypatch.setattr(mxfp4_autotune, "record_shape", lambda *_args, **_kwargs: None)

    def _supports(_key, name, **kwargs):
        profile_checks.append((name, kwargs["expected_identities"]))
        return True

    monkeypatch.setattr(mxfp4_autotune, "cached_profile_supports", _supports)

    first = linear_mod._mxfp4_resolve_backend(a, w, None, None)
    second = linear_mod._mxfp4_resolve_backend(a, w, None, None)
    assert first.name == second.name == "asm"
    assert first.asm_snapshot.config == second.asm_snapshot.config == (symbol, 0)
    assert len(profile_checks) == 1

    state["snapshot"] = mxfp4_asm.RuntimeSnapshot(
        "gfx950",
        256,
        (symbol, 1),
        ("files", 2),
        (
            ("implementation", "asm"),
            ("kernel_name", symbol),
            ("log2_k_split", 1),
            ("code_object_sha256", "second"),
        ),
        True,
    )
    changed_files = linear_mod._mxfp4_resolve_backend(a, w, None, None)
    assert changed_files.asm_snapshot.config == (symbol, 1)
    assert len(profile_checks) == 2

    state["epoch"] += 1
    linear_mod._mxfp4_resolve_backend(a, w, None, None)
    assert len(profile_checks) == 3
    linear_mod._mxfp4_resolved_backend_cache.clear()
    mxfp4_autotune.clear()


def test_mxfp4_resolution_cache_key_supports_shape_only_validation():
    """Offline cache generation need not allocate real model-sized tensors."""
    from lumen.ops.quantize import linear as linear_mod

    class _ShapeOnly:
        def __init__(self, shape):
            self.shape = shape

    key = (16384, 4096, 4096)
    shape_only = _ShapeOnly((key[0], key[2] // 2))
    assert linear_mod._mxfp4_resolution_cache_key(key, shape_only, False) == (
        "shape-only",
        None,
        key,
        False,
    )

    real_tensor = torch.empty((1, 1))
    assert linear_mod._mxfp4_resolution_cache_key(key, real_tensor, True) == (
        "cpu",
        None,
        key,
        True,
    )


def test_mxfp4_resolver_does_not_offer_unavailable_plain_backend(monkeypatch):
    """A successful optional probe must not imply that plain Triton exists."""
    from lumen.ops.quantize import linear as linear_mod

    key = (32, 64, 256)
    a = torch.empty((key[0], key[2] // 2), dtype=torch.uint8)
    w = torch.empty((key[1], key[2] // 2), dtype=torch.uint8)
    symbol = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"
    snapshot = mxfp4_asm.RuntimeSnapshot(
        "gfx950",
        256,
        (symbol, 0),
        ("test",),
        (("implementation", "asm"), ("kernel_name", symbol)),
        True,
    )

    mxfp4_autotune.clear()
    linear_mod._mxfp4_legality_cache.clear()
    linear_mod._mxfp4_resolved_backend_cache.clear()
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_gemm_probed", True)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_gemm_fn", None)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_asm_ok", True)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_preshuffle_ok", False)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_flydsl_ok", False)
    monkeypatch.setattr(mxfp4_asm, "runtime_snapshot", lambda *_shape: snapshot)

    def _pick(_key, candidates, fallback=None, incumbent=None, identities=None):
        assert [name for name, _fn in candidates] == ["asm"]
        assert fallback == incumbent == "asm"
        assert set(identities) == {"asm"}
        return "asm"

    monkeypatch.setattr(linear_mod.mxfp4_autotune, "pick_backend", _pick)
    try:
        assert linear_mod._mxfp4_choose_backend(a, w, None, None) == "asm"
    finally:
        mxfp4_autotune.clear()
        linear_mod._mxfp4_legality_cache.clear()
        linear_mod._mxfp4_resolved_backend_cache.clear()


def test_mxfp4_cached_asm_is_dropped_when_tuned_row_changes(monkeypatch):
    """A later non-ASM tuned row cannot inherit an earlier ``asm`` decision."""
    from lumen.ops.quantize import linear as linear_mod
    from lumen.ops.quantize import ops as quantize_ops

    key = (32, 64, 256)
    a = torch.empty((key[0], key[2] // 2), dtype=torch.uint8)
    w = torch.empty((key[1], key[2] // 2), dtype=torch.uint8)
    row = {
        "kernelName": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
        "splitK": 0,
    }

    mxfp4_autotune.clear()
    linear_mod._mxfp4_legality_cache.clear()
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_gemm_probed", True)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_gemm_fn", lambda *_args: None)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_asm_ok", True)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_preshuffle_ok", False)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_flydsl_ok", False)
    monkeypatch.setattr(quantize_ops, "triton_arch", lambda: "gfx950")
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_asm_config",
        lambda *_shape: mxfp4_asm.validate_tuned_entry(row, key[2], "gfx950"),
    )
    def _snapshot(*shape):
        config = linear_mod._mxfp4_asm_config(*shape)
        identity = linear_mod._mxfp4_asm_identity(config) if config else None
        return mxfp4_asm.RuntimeSnapshot(
            "gfx950",
            256,
            config,
            ("test-row", repr(row)),
            tuple(identity.items()) if identity is not None else None,
            True,
        )

    monkeypatch.setattr(mxfp4_asm, "runtime_snapshot", _snapshot)
    try:
        assert linear_mod._mxfp4_backend_legality(key, a, w)[0]
        mxfp4_autotune._choice[key] = "asm"
        mxfp4_autotune._profiles[key] = {
            "winner": "asm",
            "incumbent": "asm",
            "switch_margin": mxfp4_autotune._SWITCH_MARGIN,
            "timings_ms": {"asm": 1.0},
            "identities": {"asm": linear_mod._mxfp4_asm_identity((row["kernelName"], 0))},
        }
        assert mxfp4_autotune.cached_profile_supports(
            key,
            "asm",
            expected_identities={
                "asm": linear_mod._mxfp4_asm_identity((row["kernelName"], 0))
            },
            required_incumbent="asm",
        )

        row = {"kernelName": "unapproved_implementation", "splitK": 0}

        def _pick(
            _key, candidates, fallback=None, incumbent=None, identities=None
        ):
            assert [name for name, _fn in candidates] == ["plain"]
            assert fallback == incumbent == "plain"
            assert identities["plain"]["implementation"] == "triton"
            return "plain"

        monkeypatch.setattr(linear_mod.mxfp4_autotune, "pick_backend", _pick)
        assert linear_mod._mxfp4_choose_backend(a, w, None, None) == "plain"
        assert mxfp4_autotune.cached(key) is None
    finally:
        mxfp4_autotune.clear()
        linear_mod._mxfp4_legality_cache.clear()


def test_mxfp4_cached_flydsl_requires_profile_evidence(monkeypatch):
    """A backend label alone cannot opt a shape into an unprofiled FlyDSL kernel."""
    from lumen.ops.quantize import linear as linear_mod

    key = (64, 128, 256)
    name = "flydsl_64x128x256"
    a = torch.empty((key[0], key[2] // 2), dtype=torch.uint8)
    w = torch.empty((key[1], key[2] // 2), dtype=torch.uint8)

    mxfp4_autotune.clear()
    linear_mod._mxfp4_legality_cache.clear()
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_gemm_probed", True)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_gemm_fn", lambda *_args: None)
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_backend_legality",
        lambda *_args: (False, False, (name,)),
    )
    try:
        mxfp4_autotune._choice[key] = name

        def _pick(
            _key, candidates, fallback=None, incumbent=None, identities=None
        ):
            assert [candidate for candidate, _fn in candidates] == [name, "plain"]
            assert fallback == incumbent == "plain"
            return "plain"

        monkeypatch.setattr(linear_mod.mxfp4_autotune, "pick_backend", _pick)
        assert linear_mod._mxfp4_choose_backend(a, w, None, None) == "plain"
        assert mxfp4_autotune.cached(key) is None
    finally:
        mxfp4_autotune.clear()
        linear_mod._mxfp4_legality_cache.clear()


@pytest.mark.parametrize(
    "old_symbol,old_split,new_symbol",
    [
        (
            "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E",
            0,
            "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
        ),
        (
            "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E",
            1,
            "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E",
        ),
    ],
    ids=["symbol", "split_k"],
)
def test_mxfp4_cached_asm_is_reprofiled_when_identity_changes(
    monkeypatch, old_symbol, old_split, new_symbol
):
    """The ASM symbol and split-K are both part of its performance identity."""
    from lumen.ops.quantize import linear as linear_mod
    from lumen.ops.quantize import ops as quantize_ops

    key = (32, 64, 256)
    a = torch.empty((key[0], key[2] // 2), dtype=torch.uint8)
    w = torch.empty((key[1], key[2] // 2), dtype=torch.uint8)
    config = (new_symbol, 0)

    mxfp4_autotune.clear()
    linear_mod._mxfp4_legality_cache.clear()
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_gemm_probed", True)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_gemm_fn", lambda *_args: None)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_asm_ok", True)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_preshuffle_ok", False)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_flydsl_ok", False)
    monkeypatch.setattr(quantize_ops, "triton_arch", lambda: "gfx950")
    monkeypatch.setattr(linear_mod, "_mxfp4_asm_config", lambda *_shape: config)
    identity = mxfp4_asm.identity(config, "gfx950")
    monkeypatch.setattr(
        mxfp4_asm,
        "runtime_snapshot",
        lambda *_shape: mxfp4_asm.RuntimeSnapshot(
            "gfx950",
            256,
            config,
            ("test-config", config),
            tuple(identity.items()) if identity is not None else None,
            True,
        ),
    )
    try:
        mxfp4_autotune._choice[key] = "asm"
        mxfp4_autotune._profiles[key] = {
            "winner": "asm",
            "incumbent": "asm",
            "switch_margin": mxfp4_autotune._SWITCH_MARGIN,
            "timings_ms": {"asm": 1.0},
            "identities": {
                "asm": mxfp4_asm.identity(
                    (old_symbol, old_split), "gfx950"
                )
            },
        }

        def _pick(
            _key, candidates, fallback=None, incumbent=None, identities=None
        ):
            assert [name for name, _fn in candidates] == ["asm", "plain"]
            assert incumbent == "asm"
            assert identities["asm"]["kernel_name"] == new_symbol
            return "asm"

        monkeypatch.setattr(linear_mod.mxfp4_autotune, "pick_backend", _pick)
        assert linear_mod._mxfp4_choose_backend(a, w, None, None) == "asm"
        assert mxfp4_autotune.cached(key) is None
    finally:
        mxfp4_autotune.clear()
        linear_mod._mxfp4_legality_cache.clear()


def test_mxfp4_cached_flydsl_requires_exact_identity_and_asm_race():
    """A FlyDSL replay proves both its tile and the ASM incumbent it beat."""
    key = (64, 128, 256)
    name = "flydsl_64x128x256"
    flydsl_identity = flydsl_mxfp4.config_identity(flydsl_mxfp4.get_config(name))
    asm_symbol = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E"
    asm_identity = mxfp4_asm.identity((asm_symbol, 0), "gfx950")
    assert asm_identity is not None
    profile = {
        "winner": name,
        "incumbent": "asm",
        "switch_margin": mxfp4_autotune._SWITCH_MARGIN,
        "timings_ms": {"asm": 1.0, name: 0.94},
        "validation": {name: "passed against asm"},
        "identities": {"asm": asm_identity, name: flydsl_identity},
        "confirmation": {
            "challenger": name,
            "status": "passed",
            "winner": name,
            "switch_margin": mxfp4_autotune._SWITCH_MARGIN,
            "timed_rounds": mxfp4_autotune._balanced_round_count(
                mxfp4_autotune._TIMED_ITERS
            ),
            "timed_sample_counts": {
                "asm": mxfp4_autotune._balanced_round_count(
                    mxfp4_autotune._TIMED_ITERS
                ),
                name: mxfp4_autotune._balanced_round_count(
                    mxfp4_autotune._TIMED_ITERS
                ),
            },
            "timings_ms": {"asm": 1.0, name: 0.94},
        },
    }

    mxfp4_autotune.clear()
    try:
        mxfp4_autotune._choice[key] = name
        mxfp4_autotune._profiles[key] = profile
        expected = {"asm": asm_identity, name: flydsl_identity}
        assert mxfp4_autotune.cached_profile_supports(
            key,
            name,
            expected_identities=expected,
            required_incumbent="asm",
        )

        profile["identities"][name] = {**flydsl_identity, "tile_k": 128}
        assert not mxfp4_autotune.cached_profile_supports(
            key,
            name,
            expected_identities=expected,
            required_incumbent="asm",
        )

        profile["identities"][name] = flydsl_identity
        profile["incumbent"] = "plain"
        assert not mxfp4_autotune.cached_profile_supports(
            key,
            name,
            expected_identities=expected,
            required_incumbent="asm",
        )
    finally:
        mxfp4_autotune.clear()


def test_mxfp4_fused_b_shuffle_requires_replayable_profile(monkeypatch):
    """A cached label alone must never change the quantizer's B layout."""
    from lumen.ops.quantize import linear as linear_mod
    from lumen.ops.quantize import ops as quantize_ops

    key = (64, 128, 256)
    flydsl_name = "flydsl_64x128x256"
    asm_identity = {"implementation": "asm", "kernel_name": "approved"}
    flydsl_identity = {
        "implementation": "flydsl",
        "source_revision": "tested",
        "tile_m": 64,
        "tile_n": 128,
        "tile_k": 256,
    }
    snapshot_state = {
        "value": mxfp4_asm.RuntimeSnapshot(
            "gfx950",
            256,
            ("approved", 0),
            ("registry", 1),
            tuple(asm_identity.items()),
            True,
        )
    }
    monkeypatch.setattr(
        quantize_ops, "mxfp4_data_shuffle_supported", lambda *_shape: True
    )
    monkeypatch.setattr(
        mxfp4_asm, "runtime_snapshot", lambda *_shape: snapshot_state["value"]
    )
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_asm_config",
        lambda *_shape: pytest.fail("performed an independent ASM config read"),
    )
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_asm_identity",
        lambda _config: pytest.fail("performed an independent ASM identity read"),
    )
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_flydsl_identity",
        lambda name: flydsl_identity if name == flydsl_name else None,
    )

    mxfp4_autotune.clear()
    try:
        mxfp4_autotune._choice[key] = "asm"
        assert not linear_mod._mxfp4_can_fuse_b_shuffle(key, 128, 128)

        mxfp4_autotune._profiles[key] = {
            "winner": "asm",
            "incumbent": "asm",
            "switch_margin": mxfp4_autotune._SWITCH_MARGIN,
            "timings_ms": {"asm": 1.0},
            "identities": {"asm": asm_identity},
        }
        assert linear_mod._mxfp4_can_fuse_b_shuffle(key, 128, 128)

        mxfp4_autotune._choice[key] = flydsl_name
        mxfp4_autotune._profiles.pop(key)
        assert not linear_mod._mxfp4_can_fuse_b_shuffle(key, 128, 128)

        # Exact 5% boundary: 1 / 1.05 is sufficient, anything slower is not.
        mxfp4_autotune._profiles[key] = {
            "winner": flydsl_name,
            "incumbent": "asm",
            "switch_margin": mxfp4_autotune._SWITCH_MARGIN,
            "timings_ms": {
                "asm": 1.0,
                flydsl_name: 1.0 / mxfp4_autotune._SWITCH_MARGIN,
            },
            "validation": {flydsl_name: "passed against asm"},
            "identities": {
                "asm": asm_identity,
                flydsl_name: flydsl_identity,
            },
            "confirmation": {
                "challenger": flydsl_name,
                "status": "passed",
                "winner": flydsl_name,
                "switch_margin": mxfp4_autotune._SWITCH_MARGIN,
                "timed_rounds": mxfp4_autotune._balanced_round_count(
                    mxfp4_autotune._TIMED_ITERS
                ),
                "timed_sample_counts": {
                    "asm": mxfp4_autotune._balanced_round_count(
                        mxfp4_autotune._TIMED_ITERS
                    ),
                    flydsl_name: mxfp4_autotune._balanced_round_count(
                        mxfp4_autotune._TIMED_ITERS
                    ),
                },
                "timings_ms": {
                    "asm": 1.0,
                    flydsl_name: 1.0 / mxfp4_autotune._SWITCH_MARGIN,
                },
            },
        }
        assert linear_mod._mxfp4_can_fuse_b_shuffle(key, 128, 128)

        mxfp4_autotune._profiles[key]["timings_ms"][flydsl_name] = 0.96
        assert not linear_mod._mxfp4_can_fuse_b_shuffle(key, 128, 128)

        # The Triton preshuffle layout is equally irreversible. A cached label
        # alone is insufficient, and an exact measured identity is required.
        snapshot_state["value"] = mxfp4_asm.RuntimeSnapshot(
            "gfx950", 256, None, ("registry", 2), None, True
        )
        mxfp4_autotune._choice[key] = "shuffled"
        mxfp4_autotune._profiles.pop(key)
        assert not linear_mod._mxfp4_can_fuse_b_shuffle(key, 128, 128)

        shuffled_identity = linear_mod._mxfp4_triton_identity("shuffled")
        mxfp4_autotune._profiles[key] = {
            "winner": "shuffled",
            "incumbent": "shuffled",
            "switch_margin": mxfp4_autotune._SWITCH_MARGIN,
            "timings_ms": {"shuffled": 1.0},
            "identities": {"shuffled": shuffled_identity},
        }
        assert linear_mod._mxfp4_can_fuse_b_shuffle(key, 128, 128)

        mxfp4_autotune._profiles[key]["identities"]["shuffled"] = {
            "implementation": "triton",
            "entrypoint": "different_kernel",
        }
        assert not linear_mod._mxfp4_can_fuse_b_shuffle(key, 128, 128)

        snapshot_state["value"] = mxfp4_asm.RuntimeSnapshot(
            "unknown", 0, None, ("discovery-failed",), None, False
        )
        assert not linear_mod._mxfp4_can_fuse_b_shuffle(key, 128, 128)
    finally:
        mxfp4_autotune.clear()


def test_mxfp4_cached_flydsl_is_reprofiled_when_asm_appears(monkeypatch):
    """A FlyDSL-vs-Triton result cannot bypass a newly available ASM entry."""
    from lumen.ops.quantize import linear as linear_mod
    from lumen.ops.quantize import ops as quantize_ops

    key = (64, 128, 256)
    name = "flydsl_64x128x256"
    a = torch.empty((key[0], key[2] // 2), dtype=torch.uint8)
    w = torch.empty((key[1], key[2] // 2), dtype=torch.uint8)
    asm_symbol = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E"

    mxfp4_autotune.clear()
    linear_mod._mxfp4_legality_cache.clear()
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_gemm_probed", True)
    monkeypatch.setattr(linear_mod, "_fast_mxfp4_gemm_fn", lambda *_args: None)
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_backend_legality",
        lambda *_args: (True, False, (name,)),
    )
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_asm_config",
        lambda *_shape: (asm_symbol, 0),
    )
    monkeypatch.setattr(quantize_ops, "triton_arch", lambda: "gfx950")
    asm_identity = mxfp4_asm.identity((asm_symbol, 0), "gfx950")
    monkeypatch.setattr(
        mxfp4_asm,
        "runtime_snapshot",
        lambda *_shape: mxfp4_asm.RuntimeSnapshot(
            "gfx950",
            256,
            (asm_symbol, 0),
            ("test-config", asm_symbol, 0),
            tuple(asm_identity.items()) if asm_identity is not None else None,
            True,
        ),
    )
    try:
        mxfp4_autotune._choice[key] = name
        mxfp4_autotune._profiles[key] = {
            "winner": name,
            "incumbent": "plain",
            "switch_margin": mxfp4_autotune._SWITCH_MARGIN,
            "timings_ms": {"plain": 1.0, name: 0.9},
            "validation": {name: "passed against plain"},
            "identities": {
                "plain": {
                    "implementation": "triton",
                    "entrypoint": "gemm_afp4wfp4",
                },
                name: linear_mod._mxfp4_flydsl_identity(name),
            },
        }

        def _pick(
            _key, candidates, fallback=None, incumbent=None, identities=None
        ):
            assert [candidate for candidate, _fn in candidates] == [
                "asm",
                name,
                "plain",
            ]
            assert incumbent == "asm"
            assert identities["asm"]["kernel_name"] == asm_symbol
            return "asm"

        monkeypatch.setattr(linear_mod.mxfp4_autotune, "pick_backend", _pick)
        assert linear_mod._mxfp4_choose_backend(a, w, None, None) == "asm"
        assert mxfp4_autotune.cached(key) is None
    finally:
        mxfp4_autotune.clear()
        linear_mod._mxfp4_legality_cache.clear()


def test_mxfp4_remeasure_excludes_plain_for_shuffled_weight(monkeypatch):
    """A lapsed decision must not remeasure a shuffled operand as row-major."""
    from lumen.ops.quantize import linear as linear_mod

    key = (2048, 4096, 14336)
    a = torch.empty((key[0], key[2] // 2), dtype=torch.uint8)
    w = torch.empty((key[1], key[2] // 2), dtype=torch.uint8)
    linear_mod._mark_mxfp4_data_shuffled(w)

    mxfp4_autotune.clear()
    linear_mod._mxfp4_legality_cache.clear()
    try:
        mxfp4_autotune._choice[key] = "asm"
        monkeypatch.setattr(
            linear_mod,
            "_mxfp4_backend_legality",
            lambda k, x, y, *_rest: (False, True, ()),
        )
        monkeypatch.setattr(linear_mod, "_mxfp4_preshuffle_eligible", lambda x, y: False)

        def _pick(
            _key, candidates, fallback=None, incumbent=None, identities=None
        ):
            assert [name for name, _fn in candidates] == ["shuffled"]
            assert fallback == "shuffled"
            assert incumbent == "shuffled"
            assert identities["shuffled"]["implementation"] == "triton"
            return "shuffled"

        monkeypatch.setattr(linear_mod.mxfp4_autotune, "pick_backend", _pick)
        assert linear_mod._mxfp4_choose_backend(a, w, None, None) == "shuffled"
    finally:
        mxfp4_autotune.clear()
        linear_mod._mxfp4_legality_cache.clear()


def test_mxfp4_dispatch_launches_the_validated_asm_snapshot(monkeypatch):
    """The direct launch cannot re-read a different tuned-table config."""
    from lumen.ops.quantize import linear as linear_mod

    M, N, K = 32, 64, 256
    a = torch.empty((M, K // 2), dtype=torch.uint8)
    w = torch.empty((N, K // 2), dtype=torch.uint8)
    scale_a = torch.empty((M, K // 32), dtype=torch.uint8)
    scale_w = torch.empty((N, K // 32), dtype=torch.uint8)
    config = ("validated-symbol", 2)
    snapshot = mxfp4_asm.RuntimeSnapshot(
        "gfx950", 256, config, ("files", 1), None, True
    )
    resolution = linear_mod._MXFP4BackendResolution(
        "asm", snapshot, True, False, ()
    )
    launched = []
    output = object()

    monkeypatch.setattr(linear_mod, "_FAST_QUANT_DISPATCH", True)
    monkeypatch.setattr(linear_mod, "_mxfp4_probe_backends", lambda: True)
    monkeypatch.setattr(
        linear_mod, "_mxfp4_resolve_backend", lambda *_args, **_kwargs: resolution
    )

    def _launch(*_args, asm_config=None):
        launched.append(asm_config)
        return output

    monkeypatch.setattr(linear_mod, "_gemm_mxfp4_aiter_asm", _launch)
    assert linear_mod.gemm_mxfp4_dispatch(a, w, scale_a, scale_w) is output
    assert launched == [config]


def test_mxfp4_dispatch_fallback_lock_is_scoped_to_shape(monkeypatch):
    """One shape's BF16 verdict must not bypass another shape's working kernel."""
    from lumen.ops import dispatch as dispatch_mod
    from lumen.ops.quantize import linear as linear_mod

    dispatch_mod._backend_cache.clear()
    monkeypatch.setattr(dispatch_mod, "_SKIP_BACKEND_SYNC", True)
    monkeypatch.setattr(linear_mod, "_FAST_QUANT_DISPATCH", False)
    monkeypatch.setattr(linear_mod, "_mxfp4_probe_backends", lambda: True)
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_resolve_backend",
        lambda *_args, **_kwargs: linear_mod._MXFP4BackendResolution(
            "plain",
            mxfp4_asm.RuntimeSnapshot(
                "gfx950", 256, None, ("test",), None, True
            ),
            False,
            False,
            (),
        ),
    )

    bad_m, good_m, n, k = 32, 64, 64, 128
    bad_a = torch.empty((bad_m, k // 2), dtype=torch.uint8)
    good_a = torch.empty((good_m, k // 2), dtype=torch.uint8)
    weight = torch.empty((n, k // 2), dtype=torch.uint8)
    scale_w = torch.empty((n, k // 32), dtype=torch.uint8)

    def _plain(a, *_args):
        if a.shape[0] == bad_m:
            raise RuntimeError("this shape has no plain kernel")
        return "plain"

    def _unavailable(*_args):
        raise RuntimeError("backend unavailable in this test")

    monkeypatch.setitem(linear_mod._MXFP4_BACKENDS, "asm", _unavailable)
    monkeypatch.setitem(linear_mod._MXFP4_BACKENDS, "shuffled", _unavailable)
    monkeypatch.setitem(linear_mod._MXFP4_BACKENDS, "plain", _plain)
    monkeypatch.setattr(linear_mod, "_gemm_mxfp4_fallback", lambda *_args: "dequant_bf16")

    try:
        # Lock only the bad shape onto the degraded path.
        for _ in range(dispatch_mod._BACKEND_WARMUP_CALLS):
            assert linear_mod.gemm_mxfp4_dispatch(bad_a, weight, None, scale_w) == "dequant_bf16"

        # The old key omitted M/N/K, so this call inherited that lock and never
        # reached its working plain backend.
        assert linear_mod.gemm_mxfp4_dispatch(good_a, weight, None, scale_w) == "plain"
    finally:
        dispatch_mod._backend_cache.clear()


def test_mxfp4_configure_wires_tuned_table_and_cache(tmp_path):
    """configure() sets both env knobs, defers to ones already set, and merges."""
    cache = tmp_path / "autotune.json"
    tuned = tmp_path / "tuned.csv"
    tuned.write_text("gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n")

    original_env = os.environ.get(mxfp4_autotune.AITER_TUNED_CONFIG_ENV)
    original_cache = mxfp4_autotune._CACHE_PATH
    os.environ.pop(mxfp4_autotune.AITER_TUNED_CONFIG_ENV, None)
    mxfp4_autotune._CACHE_PATH = ""
    mxfp4_autotune.clear()
    try:
        applied = mxfp4_autotune.configure(
            tuned_config=str(tuned), autotune_cache=str(cache)
        )
        assert str(tuned) in applied["tuned_config"]
        assert applied["autotune_cache"] == str(cache)
        # AITER's own table has to stay in the list; the two cover different shapes.
        assert applied["tuned_config"].count(":") >= 1

        # A second call must not stomp what is already configured.
        other = tmp_path / "other.csv"
        other.write_text("gfx\n")
        again = mxfp4_autotune.configure(tuned_config=str(other))
        assert str(other) not in again["tuned_config"]

        # A path that does not exist is reported, not silently written in.
        os.environ.pop(mxfp4_autotune.AITER_TUNED_CONFIG_ENV, None)
        missing = mxfp4_autotune.configure(tuned_config=str(tmp_path / "nope.csv"))
        assert missing["tuned_config"] == ""
    finally:
        if original_env is None:
            os.environ.pop(mxfp4_autotune.AITER_TUNED_CONFIG_ENV, None)
        else:
            os.environ[mxfp4_autotune.AITER_TUNED_CONFIG_ENV] = original_env
        mxfp4_autotune._CACHE_PATH = original_cache
        mxfp4_autotune.clear()


def test_mxfp4_configure_keeps_table_order_and_skips_missing(tmp_path):
    """Several tables stay in the given order, and a missing one is dropped.

    Order is what decides which row wins for a shape both tables carry, so a
    model's own table has to stay ahead of the generic one.
    """
    model = tmp_path / "model.csv"
    generic = tmp_path / "generic.csv"
    for f in (model, generic):
        f.write_text("gfx,cu_num,M,N,K,kernelId,splitK,us\n")

    original_env = os.environ.get(mxfp4_autotune.AITER_TUNED_CONFIG_ENV)
    original_cache = mxfp4_autotune._CACHE_PATH
    os.environ.pop(mxfp4_autotune.AITER_TUNED_CONFIG_ENV, None)
    mxfp4_autotune._CACHE_PATH = ""
    mxfp4_autotune.clear()
    try:
        applied = mxfp4_autotune.configure(tuned_config=[str(model), str(generic)])
        listed = applied["tuned_config"].split(":")
        assert listed[:2] == [str(model), str(generic)]

        # One unusable entry must not cost the others.
        os.environ.pop(mxfp4_autotune.AITER_TUNED_CONFIG_ENV, None)
        applied = mxfp4_autotune.configure(
            tuned_config=[str(tmp_path / "nope.csv"), str(generic)]
        )
        listed = applied["tuned_config"].split(":")
        assert listed[0] == str(generic)
        assert "nope.csv" not in applied["tuned_config"]
    finally:
        if original_env is None:
            os.environ.pop(mxfp4_autotune.AITER_TUNED_CONFIG_ENV, None)
        else:
            os.environ[mxfp4_autotune.AITER_TUNED_CONFIG_ENV] = original_env
        mxfp4_autotune._CACHE_PATH = original_cache
        mxfp4_autotune.clear()


def test_mxfp4_shape_log_records_all_three_gemms(tmp_path):
    """The collector must see fprop, dgrad and wgrad from a single linear.

    This is what makes tuning generalise: the backward shapes permute the dims
    (a wgrad's M is the output width, its K is the token count) and are easy to
    derive wrongly by hand.
    """
    _require_mxfp4_dtype()
    from lumen.ops.quantize.linear import quantized_linear

    log = tmp_path / "shapes.csv"
    mxfp4_autotune.clear()
    original = mxfp4_autotune._SHAPE_LOG_PATH
    mxfp4_autotune._SHAPE_LOG_PATH = str(log)
    try:
        M, N, K = 1024, 768, 512
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        quantized_linear(x, w, scaling_type="mxfp4").sum().backward()
        mxfp4_autotune._save_shape_log()

        seen = {
            (int(r["M"]), int(r["N"]), int(r["K"]))
            for r in csv.DictReader(log.open())
        }
    finally:
        mxfp4_autotune._SHAPE_LOG_PATH = original
        mxfp4_autotune.clear()

    assert (M, N, K) in seen, f"fprop shape missing from {seen}"
    assert (M, K, N) in seen, f"dgrad shape missing from {seen}"
    assert (N, K, M) in seen, f"wgrad shape missing from {seen}"


@pytest.mark.parametrize(
    "M,K", [(128, 256), (256, 512)],
    ids=["128x256", "256x512"],
)
def test_mxfp4_expand_2d_scale_to_1d(M, K):
    """2D scale expansion should replicate each tile scale across block_size rows."""
    block = MXFP4_BLOCK_SIZE
    sm, sn = M // block, K // block
    scale_2d = torch.randint(100, 200, (sm, sn), dtype=torch.uint8, device="cuda")

    expanded = _expand_2d_scale_to_1d(scale_2d, (M, K), block_size=block)
    assert expanded.shape == (M, sn), f"Expected ({M}, {sn}), got {expanded.shape}"

    for tile_row in range(sm):
        for row_in_tile in range(block):
            global_row = tile_row * block + row_in_tile
            torch.testing.assert_close(
                expanded[global_row], scale_2d[tile_row], atol=0, rtol=0,
            )

    # Passthrough: 1D scale with matching row count should return unchanged
    scale_1d = torch.randint(100, 200, (M, sn), dtype=torch.uint8, device="cuda")
    result_1d = _expand_2d_scale_to_1d(scale_1d, (M, K), block_size=block)
    assert result_1d.data_ptr() == scale_1d.data_ptr()


@pytest.mark.parametrize("shape", MXFP4_SHAPES, ids=[f"{m}x{n}" for m, n in MXFP4_SHAPES])
def test_mxfp4_dequant_2d_vs_manual_reference(shape):
    """2D dequant should match a manual unpack+scale reference."""
    _require_mxfp4_dtype()
    M, N = shape
    torch.manual_seed(13)
    torch.cuda.manual_seed(13)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    try:
        data_fp4, scales_2d = convert_to_mxfp4_2d(
            x.float(), block_size=MXFP4_BLOCK_SIZE, use_sr=False,
        )
        y = convert_from_mxfp4_2d(
            data_fp4, scales_2d, output_dtype=torch.float32,
            block_size=MXFP4_BLOCK_SIZE,
        )
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 2D quant/dequant unavailable: {e}")

    # Manual reference on CPU
    block = MXFP4_BLOCK_SIZE
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    packed = data_fp4.cpu().view(torch.uint8)
    unpacked = packed.repeat_interleave(2, dim=-1)
    unpacked[..., ::2] = unpacked[..., ::2] & 0xF
    unpacked[..., 1::2] = unpacked[..., 1::2] >> 4
    values = lut[unpacked.long()]

    sm, sn = scales_2d.shape[-2], scales_2d.shape[-1]
    scale_f32 = torch.pow(
        2.0, scales_2d.cpu().view(torch.uint8).to(torch.float32) - 127.0,
    )
    scale_expanded = (
        scale_f32.view(sm, 1, sn, 1)
        .expand(sm, block, sn, block)
        .reshape(M, N)
    )
    ref = values * scale_expanded

    torch.testing.assert_close(y.cpu(), ref, atol=0, rtol=0)


@pytest.mark.parametrize("shape", MXFP4_SHAPES, ids=[f"{m}x{n}" for m, n in MXFP4_SHAPES])
def test_mxfp4_roundtrip_quant_dequant_vs_torchao_roundtrip(shape):
    """Full Lumen roundtrip should match full torchAO roundtrip bitwise."""
    _require_mxfp4_dtype()
    M, N = shape
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    try:
        data_fp4, scales = convert_to_mxfp4(
            x.float(), block_size=MXFP4_BLOCK_SIZE, axis=-1, use_sr=False,
        )
        x_deq_lumen = convert_from_mxfp4(
            data_fp4, scales, output_dtype=torch.float32,
            block_size=MXFP4_BLOCK_SIZE,
        )
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 roundtrip unavailable: {e}")

    mx_ref = MXTensor.to_mx(
        x.float().cpu().contiguous(),
        torch.float4_e2m1fn_x2,
        MXFP4_BLOCK_SIZE,
        scaling_mode=ScaleCalculationMode.EVEN,
    )
    x_deq_torchao = mx_ref.dequantize(torch.float32)

    torch.testing.assert_close(x_deq_lumen.cpu(), x_deq_torchao, atol=0, rtol=0)


@pytest.mark.parametrize(
    "rank,should_write",
    [(None, True), ("0", True), ("1", False), ("not-a-rank", True)],
    ids=["no-rank", "rank-zero", "rank-one", "invalid-rank"],
)
def test_mxfp4_autotune_cache_has_one_distributed_writer(
    tmp_path, monkeypatch, rank, should_write
):
    """Only parseable nonzero global ranks skip the shared cache write."""
    cache = tmp_path / "autotune.json"
    original_path = mxfp4_autotune._CACHE_PATH
    mxfp4_autotune.clear()
    mxfp4_autotune._CACHE_PATH = str(cache)
    if rank is None:
        monkeypatch.delenv("RANK", raising=False)
    else:
        monkeypatch.setenv("RANK", rank)
    monkeypatch.setattr(mxfp4_autotune, "_arch", lambda: "gfx950")
    monkeypatch.setattr(mxfp4_autotune, "_backend_fingerprint", lambda: "backends")
    monkeypatch.setattr(mxfp4_autotune, "_runtime_metadata", lambda: {})
    monkeypatch.setattr(
        mxfp4_autotune, "_tuned_table_fingerprint", lambda: "tables"
    )
    try:
        mxfp4_autotune._choice[(64, 128, 256)] = "asm"
        mxfp4_autotune._cache_dirty = True
        mxfp4_autotune._save_cache()
        assert cache.exists() is should_write
        if should_write:
            assert json.loads(cache.read_text())["choices"] == {
                "64,128,256": "asm"
            }
    finally:
        mxfp4_autotune._CACHE_PATH = original_path
        mxfp4_autotune.clear()


def test_mxfp4_autotune_fingerprint_tracks_lumen_wrapper_sources(monkeypatch):
    """Wrapper or selection-policy edits must invalidate old timing evidence."""
    hashes = {
        "mxfp4_autotune.py": "autotune-v1",
        "linear.py": "linear-v1",
    }
    monkeypatch.setattr(mxfp4_autotune, "_runtime_metadata", lambda: {})
    monkeypatch.setattr(mxfp4_asm, "registry_fingerprint", lambda: "asm-registry")
    monkeypatch.setattr(
        flydsl_mxfp4, "backend_fingerprint", lambda: "flydsl-backend"
    )
    monkeypatch.setattr(
        mxfp4_autotune,
        "_file_sha256",
        lambda path: hashes[os.path.basename(path)],
    )

    original = mxfp4_autotune._backend_fingerprint()
    hashes["linear.py"] = "linear-v2"
    assert mxfp4_autotune._backend_fingerprint() != original

    hashes["linear.py"] = "linear-v1"
    hashes["mxfp4_autotune.py"] = "autotune-v2"
    assert mxfp4_autotune._backend_fingerprint() != original


@pytest.mark.parametrize(
    "iterations,expected",
    [
        (3, ["a", "b", "c", "c", "b", "a", "a", "b", "c"]),
        (
            4,
            [
                "a",
                "b",
                "c",
                "c",
                "b",
                "a",
                "a",
                "b",
                "c",
                "c",
                "b",
                "a",
            ],
        ),
    ],
    ids=["odd-iterations", "even-iterations"],
)
def test_mxfp4_autotune_alternates_timed_candidate_order(
    monkeypatch, iterations, expected
):
    """Timed rounds alternate canonical/reverse order without changing ties."""
    calls = []

    class FakeEvent:
        def __init__(self, enable_timing=False):
            assert enable_timing

        def record(self):
            pass

        def elapsed_time(self, _end):
            return 1.0

    monkeypatch.setattr(mxfp4_autotune, "_WARMUP_ITERS", 0)
    monkeypatch.setattr(mxfp4_autotune, "_TIMED_ITERS", iterations)
    monkeypatch.setattr(mxfp4_autotune.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(mxfp4_autotune.torch.cuda, "synchronize", lambda: None)

    candidates = [
        (name, lambda name=name: calls.append(name)) for name in ("a", "b", "c")
    ]
    winner, timings, _reason, _validation = mxfp4_autotune._measure(
        (1, 1, 1), candidates
    )
    assert calls == expected
    assert winner == "a"
    assert list(timings) == ["a", "b", "c"]


def test_mxfp4_autotune_timed_failure_preserves_canonical_live_order(monkeypatch):
    """A reverse-round failure must not reverse the next round's live list."""
    calls = []
    counts = {name: 0 for name in ("a", "b", "c")}

    class FakeEvent:
        def __init__(self, enable_timing=False):
            assert enable_timing

        def record(self):
            pass

        def elapsed_time(self, _end):
            return 1.0

    def candidate(name):
        def run():
            calls.append(name)
            counts[name] += 1
            if name == "b" and counts[name] == 2:
                raise RuntimeError("timed failure")

        return run

    monkeypatch.setattr(mxfp4_autotune, "_WARMUP_ITERS", 0)
    monkeypatch.setattr(mxfp4_autotune, "_TIMED_ITERS", 4)
    monkeypatch.setattr(mxfp4_autotune.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(mxfp4_autotune.torch.cuda, "synchronize", lambda: None)

    candidates = [(name, candidate(name)) for name in ("a", "b", "c")]
    winner, timings, _reason, _validation = mxfp4_autotune._measure(
        (1, 1, 1), candidates
    )
    assert calls == ["a", "b", "c", "c", "b", "a", "a", "c", "c", "a"]
    assert winner == "a"
    assert list(timings) == ["a", "c"]


def test_mxfp4_autotune_alternates_warmup_and_removes_failures(monkeypatch):
    """Warmup alternates order and keeps a canonical list after a failure."""
    calls = []
    counts = {name: 0 for name in ("a", "b", "c")}
    synchronizations = 0

    class FakeEvent:
        def __init__(self, enable_timing=False):
            assert enable_timing

        def record(self):
            pass

        def elapsed_time(self, _end):
            return 1.0

    def candidate(name):
        def run():
            calls.append(name)
            counts[name] += 1
            if name == "b" and counts[name] == 2:
                raise RuntimeError("warmup failure")

        return run

    def synchronize():
        nonlocal synchronizations
        synchronizations += 1

    monkeypatch.setattr(mxfp4_autotune, "_WARMUP_ITERS", 3)
    monkeypatch.setattr(mxfp4_autotune, "_TIMED_ITERS", 1)
    monkeypatch.setattr(mxfp4_autotune.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(mxfp4_autotune.torch.cuda, "synchronize", synchronize)

    candidates = [(name, candidate(name)) for name in ("a", "b", "c")]
    winner, timings, _reason, _validation = mxfp4_autotune._measure(
        (1, 1, 1), candidates
    )
    assert calls == ["a", "b", "c", "c", "b", "a", "a", "c", "a", "c"]
    assert synchronizations == len(calls) - 1
    assert winner == "a"
    assert list(timings) == ["a", "c"]


def _run_mxfp4_pairwise_gate_stub(monkeypatch, confirmation_fly_time):
    flydsl_name = "flydsl_64x128x256"
    calls = []
    current = {"name": None}
    timed_counts = {"asm": 0, flydsl_name: 0}

    class FakeEvent:
        def __init__(self, enable_timing=False):
            assert enable_timing

        def record(self):
            pass

        def elapsed_time(self, _end):
            name = current["name"]
            timed_counts[name] += 1
            if timed_counts[name] <= 3:
                return 100.0 if name == "asm" else 94.0
            return 100.0 if name == "asm" else confirmation_fly_time

    def candidate(name):
        def run():
            current["name"] = name
            calls.append(name)

        return run

    monkeypatch.setattr(
        mxfp4_autotune,
        "_validate_flydsl_candidates",
        lambda _key, candidates, _incumbent: (
            list(candidates),
            {flydsl_name: "passed against asm"},
        ),
    )
    monkeypatch.setattr(mxfp4_autotune, "_WARMUP_ITERS", 0)
    monkeypatch.setattr(mxfp4_autotune, "_TIMED_ITERS", 3)
    monkeypatch.setattr(mxfp4_autotune.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(mxfp4_autotune.torch.cuda, "synchronize", lambda: None)
    evidence = {}
    result = mxfp4_autotune._measure(
        (64, 128, 256),
        [("asm", candidate("asm")), (flydsl_name, candidate(flydsl_name))],
        incumbent="asm",
        profile_evidence=evidence,
    )
    return flydsl_name, calls, evidence, result


def test_mxfp4_autotune_pairwise_rejects_boundary_initial_win(monkeypatch):
    """A noisy full-list win cannot replace ASM when balanced ABBA disagrees."""
    flydsl_name, calls, evidence, result = _run_mxfp4_pairwise_gate_stub(
        monkeypatch, confirmation_fly_time=97.0
    )
    winner, timings, reason, _validation = result
    confirmation = evidence["confirmation"]

    assert timings == {"asm": 100.0, flydsl_name: 94.0}
    assert winner == "asm"
    assert "did not clear" in reason
    assert confirmation["status"] == "rejected"
    assert confirmation["timings_ms"] == {"asm": 100.0, flydsl_name: 97.0}
    assert confirmation["timed_rounds"] == 4
    assert confirmation["timed_sample_counts"] == {"asm": 4, flydsl_name: 4}
    assert calls[-8:] == [
        "asm",
        flydsl_name,
        flydsl_name,
        "asm",
        "asm",
        flydsl_name,
        flydsl_name,
        "asm",
    ]


def test_mxfp4_autotune_pairwise_accepts_confirmed_win(monkeypatch):
    """A FlyDSL challenger replaces ASM only when balanced ABBA confirms it."""
    flydsl_name, _calls, evidence, result = _run_mxfp4_pairwise_gate_stub(
        monkeypatch, confirmation_fly_time=94.0
    )
    winner, _timings, reason, _validation = result
    confirmation = evidence["confirmation"]

    assert winner == flydsl_name
    assert "cleared" in reason
    assert confirmation["status"] == "passed"
    assert confirmation["winner"] == flydsl_name


def test_mxfp4_autotune_pairwise_odd_count_has_position_neutral_median(
    monkeypatch,
):
    """An 81-iteration request becomes 82 balanced samples with a true median."""
    flydsl_name = "flydsl_64x128x256"
    calls = []
    current = {"name": None, "position": None}

    class FakeEvent:
        def __init__(self, enable_timing=False):
            assert enable_timing

        def record(self):
            pass

        def elapsed_time(self, _end):
            if current["name"] == "asm":
                return 90.0 if current["position"] == 0 else 110.0
            return 80.0 if current["position"] == 0 else 100.0

    def candidate(name):
        def run():
            current["name"] = name
            current["position"] = len(calls) % 2
            calls.append(name)

        return run

    monkeypatch.setattr(mxfp4_autotune, "_WARMUP_ITERS", 0)
    monkeypatch.setattr(mxfp4_autotune, "_TIMED_ITERS", 81)
    monkeypatch.setattr(mxfp4_autotune.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(mxfp4_autotune.torch.cuda, "synchronize", lambda: None)

    winner, confirmation = mxfp4_autotune._confirm_asm_flydsl(
        (64, 128, 256),
        ("asm", candidate("asm")),
        (flydsl_name, candidate(flydsl_name)),
    )
    assert winner == flydsl_name
    assert confirmation["timed_rounds"] == 82
    assert confirmation["timed_sample_counts"] == {"asm": 82, flydsl_name: 82}
    assert confirmation["timings_ms"] == {"asm": 100.0, flydsl_name: 90.0}
    for name in ("asm", flydsl_name):
        positions = [index % 2 for index, call in enumerate(calls) if call == name]
        assert positions.count(0) == positions.count(1) == 41


def test_mxfp4_autotune_profile_persists_pairwise_evidence(monkeypatch):
    """The cached profile retains the independent timings and decision reason."""
    key = (64, 128, 256)
    flydsl_name = "flydsl_64x128x256"
    confirmation = {
        "challenger": flydsl_name,
        "status": "passed",
        "winner": flydsl_name,
        "reason": "pairwise confirmation cleared 1.050x margin",
        "timings_ms": {"asm": 100.0, flydsl_name: 94.0},
        "timed_sample_counts": {"asm": 4, flydsl_name: 4},
    }

    def measure(_key, _candidates, incumbent=None, profile_evidence=None):
        assert incumbent == "asm"
        profile_evidence["confirmation"] = confirmation
        return (
            flydsl_name,
            {"asm": 100.0, flydsl_name: 94.0},
            confirmation["reason"],
            {flydsl_name: "passed against asm"},
        )

    monkeypatch.setattr(mxfp4_autotune, "AUTOTUNE_ENABLED", True)
    monkeypatch.setattr(mxfp4_autotune, "_capturing", lambda: False)
    monkeypatch.setattr(mxfp4_autotune, "_measure", measure)
    mxfp4_autotune.clear()
    mxfp4_autotune._cache_loaded = True
    try:
        chosen = mxfp4_autotune.pick_backend(
            key,
            [("asm", lambda: None), (flydsl_name, lambda: None)],
            fallback="asm",
            incumbent="asm",
        )
        assert chosen == flydsl_name
        assert mxfp4_autotune._profiles[key]["confirmation"] == confirmation
        assert mxfp4_autotune._profiles[key]["reason"] == confirmation["reason"]
    finally:
        mxfp4_autotune.clear()


def test_mxfp4_autotune_pairwise_failure_is_fail_closed(monkeypatch):
    """Any failure during the independent race retains the ASM incumbent."""
    flydsl_name = "flydsl_64x128x256"
    calls = []
    flydsl_calls = 0

    def asm():
        calls.append("asm")

    def flydsl():
        nonlocal flydsl_calls
        flydsl_calls += 1
        calls.append(flydsl_name)
        if flydsl_calls == 2:
            raise RuntimeError("injected confirmation failure")

    monkeypatch.setattr(mxfp4_autotune, "_WARMUP_ITERS", 1)
    monkeypatch.setattr(mxfp4_autotune, "_TIMED_ITERS", 3)
    monkeypatch.setattr(mxfp4_autotune.torch.cuda, "synchronize", lambda: None)

    winner, confirmation = mxfp4_autotune._confirm_asm_flydsl(
        (64, 128, 256),
        ("asm", asm),
        (flydsl_name, flydsl),
    )
    assert calls == ["asm", flydsl_name, flydsl_name]
    assert winner == "asm"
    assert confirmation["status"] == "failed"
    assert confirmation["failed_backend"] == flydsl_name
    assert confirmation["failed_phase"] == "warmup"
    assert confirmation["winner"] == "asm"


def _mxfp4_unanimous_consensus_profile(
    name="flydsl_64x128x256", device_count=8, worst_ratio=0.94
):
    samples = mxfp4_autotune._balanced_round_count(
        mxfp4_autotune._TIMED_ITERS
    )
    identities = {
        "asm": {"implementation": "asm", "kernel_name": "approved"},
        name: {"implementation": "flydsl", "config": name},
    }
    return {
        "winner": name,
        "incumbent": "asm",
        "identities": identities,
        "switch_margin": mxfp4_autotune._SWITCH_MARGIN,
        "timings_ms": {"asm": 1.0, name: 0.94},
        "validation": {name: "passed against asm"},
        "confirmation": {
            "challenger": name,
            "status": "passed",
            "winner": name,
            "switch_margin": mxfp4_autotune._SWITCH_MARGIN,
            "timed_rounds": samples,
            "timed_sample_counts": {"asm": samples, name: samples},
            "timings_ms": {"asm": 1.0, name: 0.94},
        },
        "decision_scope": mxfp4_autotune._CONSENSUS_SCOPE,
        "profile_device_count": device_count,
        "consensus": {
            "policy": "unanimous",
            "backend": name,
            "arch": "gfx950",
            "device_count": device_count,
            "correctness": {"passed": device_count, "total": device_count},
            "gate": {"passed": device_count, "total": device_count},
            "worst_ratio": worst_ratio,
        },
    }


def _write_mxfp4_consensus_cache(cache, profile, key=(64, 128, 256)):
    cache.write_text(
        json.dumps(
            {
                "schema": mxfp4_autotune._CACHE_SCHEMA,
                "arch": "gfx950",
                "decision_scope": mxfp4_autotune._CONSENSUS_SCOPE,
                "profile_device_count": 8,
                "backends": "backends",
                "tuned_tables": "tables",
                "choices": {mxfp4_autotune._cache_key(key): profile["winner"]},
                "profiles": {mxfp4_autotune._cache_key(key): profile},
            }
        )
    )


def test_mxfp4_multigpu_cold_cache_keeps_and_caches_protected_asm(monkeypatch):
    """Ranks do not independently profile a FlyDSL promotion without consensus."""
    key = (64, 128, 256)
    name = "flydsl_64x128x256"
    identities = {
        "asm": {"implementation": "asm", "kernel_name": "approved"},
        name: {"implementation": "flydsl", "config": name},
    }
    calls = []

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("LUMEN_MXFP4_REQUIRE_CONSENSUS", raising=False)
    monkeypatch.setattr(mxfp4_autotune, "_CACHE_PATH", "")
    monkeypatch.setattr(
        mxfp4_autotune,
        "_measure",
        lambda *_args, **_kwargs: pytest.fail("multi-GPU cold path profiled"),
    )
    mxfp4_autotune.clear()
    mxfp4_autotune._cache_loaded = True
    try:
        candidates = [
            ("asm", lambda: calls.append("asm")),
            (name, lambda: calls.append(name)),
        ]
        assert mxfp4_autotune.pick_backend(
            key,
            candidates,
            fallback="asm",
            incumbent="asm",
            identities=identities,
        ) == "asm"
        assert calls == []
        assert mxfp4_autotune.cached(key) == "asm"
        assert mxfp4_autotune.cached_profile_supports(
            key,
            "asm",
            expected_identities={"asm": identities["asm"]},
            required_incumbent="asm",
        )
        # The second call is a true cached hit, not a forget/recreate loop.
        assert mxfp4_autotune.pick_backend(
            key,
            candidates,
            fallback="asm",
            incumbent="asm",
            identities=identities,
        ) == "asm"
        assert calls == []
    finally:
        mxfp4_autotune.clear()


def test_mxfp4_multigpu_loads_unanimous_flydsl_consensus(tmp_path, monkeypatch):
    """A complete 8/8 cache can replay one job-wide FlyDSL decision."""
    cache = tmp_path / "autotune.json"
    key = (64, 128, 256)
    name = "flydsl_64x128x256"
    profile = _mxfp4_unanimous_consensus_profile(name)
    _write_mxfp4_consensus_cache(cache, profile, key)

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("LUMEN_MXFP4_REQUIRE_CONSENSUS", raising=False)
    monkeypatch.setattr(mxfp4_autotune, "_CACHE_PATH", str(cache))
    monkeypatch.setattr(mxfp4_autotune, "_arch", lambda: "gfx950")
    monkeypatch.setattr(
        mxfp4_autotune, "_backend_fingerprint", lambda: "backends"
    )
    monkeypatch.setattr(
        mxfp4_autotune, "_tuned_table_fingerprint", lambda: "tables"
    )
    mxfp4_autotune.clear()
    try:
        mxfp4_autotune._load_cache()
        assert mxfp4_autotune.cached(key) == name
        assert mxfp4_autotune.cached_profile_supports(
            key,
            name,
            expected_identities=profile["identities"],
            required_incumbent="asm",
        )
    finally:
        mxfp4_autotune.clear()


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "top_scope",
        "top_device_count",
        "backend",
        "arch",
        "device_count",
        "entry_device_count",
        "correctness",
        "gate",
        "worst_ratio",
    ],
)
def test_mxfp4_multigpu_rejects_incomplete_flydsl_consensus(
    tmp_path, monkeypatch, invalid
):
    """Missing, partial, or below-margin evidence fails closed before replay."""
    cache = tmp_path / f"autotune-{invalid}.json"
    key = (64, 128, 256)
    profile = _mxfp4_unanimous_consensus_profile()
    evidence = profile["consensus"]
    if invalid == "missing":
        profile.pop("consensus")
    elif invalid == "backend":
        evidence["backend"] = "flydsl_wrong_config"
    elif invalid == "arch":
        evidence["arch"] = "gfx942"
    elif invalid == "device_count":
        evidence["device_count"] = 7
    elif invalid == "entry_device_count":
        profile["profile_device_count"] = 7
    elif invalid == "correctness":
        evidence["correctness"]["passed"] = 7
    elif invalid == "gate":
        evidence["gate"]["passed"] = 7
    elif invalid == "worst_ratio":
        evidence["worst_ratio"] = 1.0 / mxfp4_autotune._SWITCH_MARGIN + 1e-6
    elif invalid not in ("top_scope", "top_device_count"):
        raise AssertionError(f"unhandled invalid consensus case: {invalid}")
    _write_mxfp4_consensus_cache(cache, profile, key)
    if invalid in ("top_scope", "top_device_count"):
        blob = json.loads(cache.read_text())
        if invalid == "top_scope":
            blob["decision_scope"] = mxfp4_autotune._SINGLE_DEVICE_SCOPE
        else:
            blob["profile_device_count"] = 7
        cache.write_text(json.dumps(blob))

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("LUMEN_MXFP4_REQUIRE_CONSENSUS", raising=False)
    monkeypatch.setattr(mxfp4_autotune, "_CACHE_PATH", str(cache))
    monkeypatch.setattr(mxfp4_autotune, "_arch", lambda: "gfx950")
    monkeypatch.setattr(
        mxfp4_autotune, "_backend_fingerprint", lambda: "backends"
    )
    monkeypatch.setattr(
        mxfp4_autotune, "_tuned_table_fingerprint", lambda: "tables"
    )
    mxfp4_autotune.clear()
    try:
        mxfp4_autotune._load_cache()
        assert mxfp4_autotune.cached(key) is None
        assert mxfp4_autotune.cached_profile(key) is None
    finally:
        mxfp4_autotune.clear()


def test_mxfp4_multigpu_consensus_survives_local_asm_fallback(
    tmp_path, monkeypatch
):
    """Adding a missing-shape ASM choice cannot invalidate loaded consensus."""
    cache = tmp_path / "autotune.json"
    consensus_key = (64, 128, 256)
    fallback_key = (128, 256, 512)
    name = "flydsl_64x128x256"
    profile = _mxfp4_unanimous_consensus_profile(name)
    _write_mxfp4_consensus_cache(cache, profile, consensus_key)

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("LUMEN_MXFP4_REQUIRE_CONSENSUS", raising=False)
    monkeypatch.setattr(mxfp4_autotune, "_CACHE_PATH", str(cache))
    monkeypatch.setattr(mxfp4_autotune, "_arch", lambda: "gfx950")
    monkeypatch.setattr(
        mxfp4_autotune, "_backend_fingerprint", lambda: "backends"
    )
    monkeypatch.setattr(
        mxfp4_autotune, "_tuned_table_fingerprint", lambda: "tables"
    )
    monkeypatch.setattr(mxfp4_autotune, "_runtime_metadata", lambda: {})
    mxfp4_autotune.clear()
    try:
        mxfp4_autotune._load_cache()
        assert mxfp4_autotune._remember_protected_asm(
            fallback_key, {"asm": {"implementation": "asm"}}
        ) == "asm"
        assert mxfp4_autotune._cache_decision_scope == (
            mxfp4_autotune._CONSENSUS_SCOPE
        )
        assert mxfp4_autotune._cache_profile_device_count == 8
        assert mxfp4_autotune.cached_profile_supports(
            consensus_key,
            name,
            expected_identities=profile["identities"],
            required_incumbent="asm",
        )
        mxfp4_autotune._save_cache()
        persisted = json.loads(cache.read_text())
        assert persisted["decision_scope"] == mxfp4_autotune._CONSENSUS_SCOPE
        assert persisted["profile_device_count"] == 8
        assert persisted["choices"] == {
            mxfp4_autotune._cache_key(consensus_key): name,
            mxfp4_autotune._cache_key(fallback_key): "asm",
        }
    finally:
        mxfp4_autotune.clear()
        assert mxfp4_autotune._cache_decision_scope == (
            mxfp4_autotune._SINGLE_DEVICE_SCOPE
        )
        assert mxfp4_autotune._cache_profile_device_count == 1


@pytest.mark.parametrize(
    "world_size,override",
    [("1", None), ("8", "0")],
    ids=["single-device", "explicit-multigpu-override"],
)
def test_mxfp4_online_autotune_behavior_is_preserved_when_consensus_not_required(
    monkeypatch, world_size, override
):
    """Single-device profiling and the explicit escape hatch keep old behavior."""
    key = (64, 128, 256)
    name = "flydsl_64x128x256"

    def measure(_key, _candidates, incumbent=None, profile_evidence=None):
        assert incumbent == "asm"
        profile_evidence["confirmation"] = {
            "status": "passed",
            "winner": name,
        }
        return (
            name,
            {"asm": 1.0, name: 0.9},
            "pairwise confirmation cleared 1.050x margin",
            {name: "passed against asm"},
        )

    monkeypatch.setenv("WORLD_SIZE", world_size)
    if override is None:
        monkeypatch.delenv("LUMEN_MXFP4_REQUIRE_CONSENSUS", raising=False)
    else:
        monkeypatch.setenv("LUMEN_MXFP4_REQUIRE_CONSENSUS", override)
    monkeypatch.setattr(mxfp4_autotune, "_CACHE_PATH", "")
    monkeypatch.setattr(mxfp4_autotune, "AUTOTUNE_ENABLED", True)
    monkeypatch.setattr(mxfp4_autotune, "_capturing", lambda: False)
    monkeypatch.setattr(mxfp4_autotune, "_measure", measure)
    mxfp4_autotune.clear()
    mxfp4_autotune._cache_loaded = True
    try:
        assert mxfp4_autotune.pick_backend(
            key,
            [("asm", lambda: None), (name, lambda: None)],
            fallback="asm",
            incumbent="asm",
        ) == name
        profile = mxfp4_autotune.cached_profile(key)
        assert profile["decision_scope"] == mxfp4_autotune._SINGLE_DEVICE_SCOPE
        assert profile["profile_device_count"] == 1
    finally:
        mxfp4_autotune.clear()
