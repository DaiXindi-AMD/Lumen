###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Lumen-owned MXFP4 Triton kernels.

Contains only kernels that AITER does not provide:
  - ASM-accelerated MXFP4 quantization (RTN + SR) via gfx950 VOP3 instructions
  - Software fallback MXFP4 quantization (RTN + SR)
  - Packed FP4 transpose
  - Blockwise Hadamard transform (RHT)

For kernels that AITER provides (RTN quant via dynamic_mxfp4_quant, GEMM,
dequant), see ``lumen/ops/quantize/ops.py`` which wraps AITER directly.

gfx950 ISA instructions used:
  - ``v_cvt_scalef32_pk_fp4_f32``    : 2×FP32 → packed FP4 byte (RTN)
  - ``v_cvt_scalef32_sr_pk_fp4_f32`` : 2×FP32 → packed FP4 byte (SR)
  - ``v_cvt_scalef32_pk_fp4_bf16``   : 2×BF16 → packed FP4 byte (RTN)
  - ``v_cvt_scalef32_sr_pk_fp4_bf16``: 2×BF16 → packed FP4 byte (SR)

ASM SR is unbiased without pre-scaling — no correction factors needed.
"""

import math

import triton
import triton.language as tl

FP4_E2M1_MAX = 6.0
_E2M1_EMAX = 2  # largest normal biased exponent for E2M1


# ---------------------------------------------------------------------------
# Scale calculation (shared by RTN and SR paths)
# ---------------------------------------------------------------------------


@triton.jit
def _calculate_fp4_scales(
    x,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
):
    """Compute E8M0 block scales for MXFP4 quantization.

    Follows the same pattern as AITER's ``_calculate_scales`` for MXFP8 but
    with ``target_max_pow2 = 2`` (FP4 E2M1 emax).
    """
    E8M0_EXPONENT_BIAS: tl.constexpr = 127

    tl.static_assert(BLOCK_N % QUANT_BLOCK_SIZE == 0)
    if IS_2D_BLOCK:
        tl.static_assert(BLOCK_M % QUANT_BLOCK_SIZE == 0)

    if x.type.element_ty == tl.float32:
        hp_int_dtype = tl.int32
        hp_mbits: tl.constexpr = 23
        hp_ebits: tl.constexpr = 8
        hp_exp_bias: tl.constexpr = 127
    else:
        hp_int_dtype = tl.int16
        hp_mbits: tl.constexpr = 7
        hp_ebits: tl.constexpr = 8
        hp_exp_bias: tl.constexpr = 127

    sbits: tl.constexpr = 1
    # FP4 E2M1: 1 mantissa bit, max exponent pow2 = 2
    mbits: tl.constexpr = 1
    target_max_pow2: tl.constexpr = 2

    NEW_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    if IS_2D_BLOCK:
        NEW_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        x_r = x.reshape(NEW_BLOCK_M, QUANT_BLOCK_SIZE, NEW_BLOCK_N, QUANT_BLOCK_SIZE)
        x_r = tl.permute(x_r, (0, 2, 1, 3))
        max_abs = tl.max(tl.abs(x_r), axis=-1)
        max_abs = tl.max(max_abs, axis=-1)
    else:
        x_r = x.reshape(BLOCK_M, NEW_BLOCK_N, QUANT_BLOCK_SIZE)
        max_abs = tl.max(tl.abs(x_r), axis=-1)
    max_abs = max_abs.to(x.type.element_ty)

    # Round-even on the scale (adaptive rounding from pytorch/ao)
    max_abs = max_abs.to(hp_int_dtype, bitcast=True)
    val_to_add = 1 << (hp_mbits - mbits - 1)
    mask = ((1 << (hp_ebits + sbits)) - 1) << hp_mbits
    max_abs = (max_abs + val_to_add) & mask

    extracted_pow2 = ((max_abs >> hp_mbits) & 0b11111111) - hp_exp_bias
    scale_e8m0_unbiased = extracted_pow2 - target_max_pow2

    scale_e8m0_unbiased = tl.minimum(
        tl.maximum(scale_e8m0_unbiased, -1 * E8M0_EXPONENT_BIAS), E8M0_EXPONENT_BIAS + 1
    )
    scale_e8m0_biased = scale_e8m0_unbiased + E8M0_EXPONENT_BIAS

    return scale_e8m0_biased.to(tl.uint8)


# ---------------------------------------------------------------------------
# FP4 packing: ASM (gfx950) + software fallback
# ---------------------------------------------------------------------------


@triton.jit
def _generate_randval(m, n, philox_seed, philox_offset):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    rng_offsets = philox_offset + ms[:, None] * n + ns[None, :]
    r1, _, _, _ = tl.randint4x(philox_seed, rng_offsets)
    return r1


@triton.jit
def _pack_fp4(
    x,
    scales,
    philox_seed,
    philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
    USE_SR: tl.constexpr = False,
    USE_ASM: tl.constexpr = False,
):
    """Pack BF16/FP32 values into MXFP4 (2 FP4 nibbles per byte).

    ASM path uses gfx950 ``v_cvt_scalef32_[sr_]pk_fp4_f32`` instructions.
    These instructions take the E8M0 scale as a uint8 and produce a packed
    byte with bits[3:0]=fp4(src_lo) and bits[7:4]=fp4(src_hi).

    For SR, the hardware uses the seed from Src1 for the first FP4 lane,
    then internally derives a new random value via v_prng_b32 for the second
    lane. No software pre-scaling is needed — the result is unbiased.
    """
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    HALF_QUANT_BLOCK_SIZE: tl.constexpr = QUANT_BLOCK_SIZE // 2
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE

    # Split into pairs for the pk (pack-2) instruction
    x0, x1 = tl.split(x.reshape(BLOCK_M, HALF_BLOCK_N, 2))

    # E8M0 scale → FP32 power-of-2 for the conversion instruction
    scales = tl.where(scales < 1, 1, scales)
    scales_fp32 = (scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    F32_MIN_NORMAL: tl.constexpr = 2**-126
    min_frag = (F32_MIN_NORMAL).to(tl.float32)
    scales_fp32 = tl.where(scales_fp32 < min_frag, min_frag, scales_fp32)

    if IS_2D_BLOCK:
        SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        scales_fp32 = (
            scales_fp32.expand_dims(axis=(1, 3))
            .broadcast_to(SCALE_BLOCK_M, QUANT_BLOCK_SIZE, SCALE_BLOCK_N, HALF_QUANT_BLOCK_SIZE)
            .reshape(BLOCK_M, HALF_BLOCK_N)
        )
    else:
        scales_fp32 = (
            scales_fp32.expand_dims(axis=2)
            .broadcast_to(BLOCK_M, SCALE_BLOCK_N, HALF_QUANT_BLOCK_SIZE)
            .reshape(BLOCK_M, HALF_BLOCK_N)
        )

    if USE_SR:
        randval0 = _generate_randval(BLOCK_M, BLOCK_N, philox_seed, philox_offset)
    else:
        randval0 = 0

    if USE_ASM:
        if x0.type.element_ty == tl.float32:
            if not USE_SR:
                # RTN: v_cvt_scalef32_pk_fp4_f32  dst, src0(f32_lo), src1(f32_hi), src2(scale)
                y = tl.inline_asm_elementwise(
                    asm="v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];",
                    constraints="=&v,v,v,v",
                    args=[x0, x1, scales_fp32],
                    dtype=tl.uint8,
                    is_pure=True,
                    pack=1,
                )
            else:
                # SR: v_cvt_scalef32_sr_pk_fp4_f32  dst, src0(u64=hi|lo), src1(seed), src2(scale)
                x_packed = (
                    x1.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
                ) | x0.to(tl.uint32, bitcast=True)
                y = tl.inline_asm_elementwise(
                    asm="v_cvt_scalef32_sr_pk_fp4_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];",
                    constraints="=&v,v,v,v",
                    args=[x_packed, randval0, scales_fp32],
                    dtype=tl.uint8,
                    is_pure=True,
                    pack=1,
                )
        else:
            # BF16 input
            if not USE_SR:
                x_packed_bf16 = (
                    x1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
                ) | x0.to(tl.uint16, bitcast=True)
                y = tl.inline_asm_elementwise(
                    asm="v_cvt_scalef32_pk_fp4_bf16 $0, $1, $2 op_sel:[0,0,0,0];",
                    constraints="=&v,v,v",
                    args=[x_packed_bf16, scales_fp32],
                    dtype=tl.uint8,
                    is_pure=True,
                    pack=1,
                )
            else:
                x_packed_bf16 = (
                    x1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
                ) | x0.to(tl.uint16, bitcast=True)
                y = tl.inline_asm_elementwise(
                    asm="v_cvt_scalef32_sr_pk_fp4_bf16 $0, $1, $2, $3 op_sel:[0,0,0,0];",
                    constraints="=&v,v,v,v",
                    args=[x_packed_bf16, randval0, scales_fp32],
                    dtype=tl.uint8,
                    is_pure=True,
                    pack=1,
                )

        # Output is already packed: bits[3:0]=fp4(x0), bits[7:4]=fp4(x1)
        # Reshape from (BLOCK_M, HALF_BLOCK_N) to the packed output layout
        y = y.reshape(BLOCK_M, HALF_BLOCK_N)
    else:
        # Software fallback: manual FP4 E2M1 conversion
        x_scaled = x / scales_fp32.expand_dims(axis=2).broadcast_to(
            BLOCK_M, HALF_BLOCK_N, 2
        ).reshape(BLOCK_M, BLOCK_N)

        if USE_SR:
            noise = randval0.to(tl.float32) * (1.0 / 4294967296.0)
            x_scaled = x_scaled + (noise - 0.5) * 0.01

        # Clamp and round to nearest FP4 E2M1 representable value
        abs_val = tl.abs(x_scaled)
        sign = tl.where(x_scaled < 0.0, 1.0, 0.0)

        # FP4 E2M1 levels: 0, 0.5, 1, 1.5, 2, 3, 4, 6
        # Boundaries: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
        code = tl.zeros_like(abs_val, dtype=tl.uint8)
        code = tl.where(abs_val >= 0.25, tl.full(code.shape, 1, dtype=tl.uint8), code)
        code = tl.where(abs_val >= 0.75, tl.full(code.shape, 2, dtype=tl.uint8), code)
        code = tl.where(abs_val >= 1.25, tl.full(code.shape, 3, dtype=tl.uint8), code)
        code = tl.where(abs_val >= 1.75, tl.full(code.shape, 4, dtype=tl.uint8), code)
        code = tl.where(abs_val >= 2.50, tl.full(code.shape, 5, dtype=tl.uint8), code)
        code = tl.where(abs_val >= 3.50, tl.full(code.shape, 6, dtype=tl.uint8), code)
        code = tl.where(abs_val >= 5.00, tl.full(code.shape, 7, dtype=tl.uint8), code)

        sign_u8 = sign.to(tl.uint8)
        fp4_code = (sign_u8 << 3) | code

        # Pack two FP4 codes into one uint8 (AITER convention: even=low nibble)
        codes_reshaped = fp4_code.reshape(BLOCK_M, HALF_BLOCK_N, 2)
        even = codes_reshaped[:, :, 0:1].reshape(BLOCK_M, HALF_BLOCK_N)
        odd = codes_reshaped[:, :, 1:2].reshape(BLOCK_M, HALF_BLOCK_N)
        y = even | (odd << 4)

    return y


# ---------------------------------------------------------------------------
# Main conversion kernel: BF16/FP32 → packed MXFP4 + E8M0 scales
# ---------------------------------------------------------------------------


@triton.jit
def _convert_to_mxfp4_kernel(
    x_ptr, y_ptr, s_ptr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_sm, stride_sn,
    philox_seed, philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
    USE_SR: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """BF16/FP32 → packed MXFP4 with E8M0 block scales.

    Computes E8M0 shared exponent scales, then quantizes each element to
    FP4 E2M1 using either ASM instructions (gfx950) or software fallback.
    Two FP4 values are packed per output byte.

    When ``USE_ASM=True`` and ``USE_SR=True``, the hardware SR is unbiased
    without any pre-scaling — no correction factors (4/3 or 16/9) are needed
    on the GEMM output.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_xn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    if IS_2D_BLOCK:
        SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        offs_sm = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    else:
        offs_sm = offs_m

    offs_x = offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
    offs_s = offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn

    x = tl.load(x_ptr + offs_x)

    scales = _calculate_fp4_scales(
        x,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
    )

    y = _pack_fp4(
        x, scales,
        philox_seed, philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
        USE_SR=USE_SR,
        USE_ASM=USE_ASM,
    )

    # Store packed FP4 output (HALF_BLOCK_N bytes per row)
    offs_yn = pid_n * HALF_BLOCK_N + tl.arange(0, HALF_BLOCK_N)
    offs_y = offs_m[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    tl.store(y_ptr + offs_y, y)
    tl.store(s_ptr + offs_s, scales)


# ---------------------------------------------------------------------------
# Packed FP4 transpose kernel
# ---------------------------------------------------------------------------


@triton.jit
def _transpose_packed_fp4_kernel(
    in_ptr, out_ptr,
    M, N_packed: tl.constexpr,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N_PACKED: tl.constexpr,
):
    """Transpose packed FP4: (M, N//2) -> (N, M//2).

    Unpacks nibbles, transposes, repacks.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    N = N_packed * 2
    BLOCK_N: tl.constexpr = BLOCK_N_PACKED * 2
    BLOCK_M_HALF: tl.constexpr = BLOCK_M // 2

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn_packed = pid_n * BLOCK_N_PACKED + tl.arange(0, BLOCK_N_PACKED)

    mask = (rm[:, None] < M) & (rn_packed[None, :] < N_packed)
    packed = tl.load(in_ptr + rm[:, None] * stride_im + rn_packed[None, :] * stride_in,
                     mask=mask, other=0).to(tl.uint8)

    # AITER convention: even in low nibble, odd in high nibble
    even = packed & 0x0F
    odd = (packed >> 4) & 0x0F
    unpacked = tl.reshape(tl.join(even, odd), (BLOCK_M, BLOCK_N))

    transposed = tl.trans(unpacked)

    reshaped = tl.reshape(transposed, (BLOCK_N, BLOCK_M_HALF, 2))
    t_even = tl.reshape(reshaped[:, :, 0:1], (BLOCK_N, BLOCK_M_HALF))
    t_odd = tl.reshape(reshaped[:, :, 1:2], (BLOCK_N, BLOCK_M_HALF))
    repacked = t_even | (t_odd << 4)

    rn_full = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm_packed = pid_m * BLOCK_M_HALF + tl.arange(0, BLOCK_M_HALF)
    out_mask = (rn_full[:, None] < N) & (rm_packed[None, :] < (M // 2))
    tl.store(out_ptr + rn_full[:, None] * stride_om + rm_packed[None, :] * stride_on,
             repacked, mask=out_mask)


# ---------------------------------------------------------------------------
# Blockwise Hadamard Transform kernel
# ---------------------------------------------------------------------------


@triton.jit
def _hadamard_transform_kernel(
    x_ptr, out_ptr, sign_ptr,
    M, N: tl.constexpr,
    stride_xm, stride_xn,
    stride_om, stride_on,
    G: tl.constexpr,
):
    """Blockwise Random Hadamard Transform via butterfly algorithm O(G log G).

    Applies H_g @ diag(S) to blocks of G elements along the last dimension.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m + tl.arange(0, 1)
    rn = pid_n * G + tl.arange(0, G)

    mask = (rm[:, None] < M) & (rn[None, :] < N)
    x = tl.load(x_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn,
                mask=mask, other=0.0).to(tl.float32)

    signs = tl.load(sign_ptr + tl.arange(0, G)).to(tl.float32)
    x = x * signs[None, :]

    h = 1
    log2_G: tl.constexpr = tl.constexpr(int(math.log2(G)))
    for _ in range(log2_G):
        x_reshaped = tl.reshape(x, (1, G // (2 * h), 2, h))
        top = tl.reshape(x_reshaped[:, :, 0:1, :], (1, G // (2 * h), h))
        bot = tl.reshape(x_reshaped[:, :, 1:2, :], (1, G // (2 * h), h))
        new_top = top + bot
        new_bot = top - bot
        x = tl.reshape(tl.join(new_top, new_bot), (1, G))
        h = h * 2

    x = x * (1.0 / tl.sqrt(float(G)))

    tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on, x, mask=mask)
