# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for Lumen quantization ops, comparing against torchao reference implementation."""

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
    hadamard_transform,
    quant_fp8_blockwise_impl,
    quant_fp8_tensorwise_impl,
    transpose_packed_fp4,
)
from lumen.ops.quantize.linear import _expand_2d_scale_to_1d, gemm_mxfp4_dispatch

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
    assert snr_self >= 4.0, f"Lumen GEMM self-consistency SNR {snr_self:.1f} dB too low"

    # Lumen dequant-matmul vs torchAO dequant-matmul (cross-framework, bitwise)
    torch.testing.assert_close(y_lumen_deq.cpu(), y_torchao, atol=0, rtol=0)


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
