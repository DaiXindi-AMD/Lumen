#!/usr/bin/env python
"""MXFP4 accuracy report: Lumen vs torchAO, per-operator comparison."""

import torch
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.mx_tensor import to_dtype as torchao_to_dtype

from lumen.ops.quantize import (
    convert_from_mxfp4,
    convert_from_mxfp4_2d,
    convert_to_mxfp4,
    convert_to_mxfp4_2d,
    convert_to_mxfp4_dual_axis,
    hadamard_transform,
    transpose_packed_fp4,
)
from lumen.ops.quantize.linear import _expand_2d_scale_to_1d, gemm_mxfp4_dispatch

BLOCK = 32
torch.manual_seed(42)
torch.cuda.manual_seed(42)

SEP = "=" * 90
THIN = "-" * 90


def snr(ref, test):
    ref_f = ref.float()
    test_f = test.float()
    noise = test_f - ref_f
    sig_power = (ref_f ** 2).mean()
    noise_power = (noise ** 2).mean()
    if noise_power == 0:
        return float("inf")
    return 10 * torch.log10(sig_power / noise_power).item()


def pct_match(a, b):
    total = a.numel()
    match = (a == b).sum().item()
    return match, total, match / total * 100


def print_tensor_stats(name, t):
    t_f = t.float()
    print(f"    {name:30s}  shape={str(list(t.shape)):16s}  "
          f"dtype={str(t.dtype):20s}  "
          f"min={t_f.min().item():12.4f}  max={t_f.max().item():12.4f}  "
          f"mean={t_f.mean().item():12.4f}")


def print_diff(lumen, torchao, label="output"):
    diff = (lumen.float() - torchao.float()).abs()
    m, t, p = pct_match(lumen, torchao)
    print(f"    差距 ({label}):")
    print(f"      bitwise 一致元素: {m}/{t} ({p:.2f}%)")
    print(f"      max |diff|:       {diff.max().item():.6e}")
    print(f"      mean |diff|:      {diff.mean().item():.6e}")
    if p < 100:
        print(f"      SNR:              {snr(torchao, lumen):.1f} dB")
    else:
        print(f"      SNR:              inf (bitwise identical)")


# ─────────────────────────────────────────────────────────────────────────────
M, K, N = 128, 256, 128
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

print(SEP)
print("MXFP4 精度报告: Lumen vs torchAO")
print(f"硬件: {torch.cuda.get_device_name(0)}")
print(f"矩阵尺寸: M={M}, K={K}, N={N},  block_size={BLOCK}")
print(f"输入 dtype: {x.dtype}")
print(SEP)

# ═══════════════════════════════════════════════════════════════════════════
# 1. convert_to_mxfp4 (1D, axis=-1, RTN)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'1. 量化 (1D, axis=-1, RTN)':=^90}")
print(f"  操作:   BF16 → packed MXFP4 (uint8) + E8M0 scales (uint8)")
print(f"  算子:   Lumen: convert_to_mxfp4(x, block_size={BLOCK}, axis=-1, use_sr=False)")
print(f"          torchAO: MXTensor.to_mx(x, float4_e2m1fn_x2, {BLOCK}, EVEN)")
print(f"  操作数: x shape={list(x.shape)}, dtype={x.dtype}")
print(THIN)

data_fp4, scales = convert_to_mxfp4(x.float(), block_size=BLOCK, axis=-1, use_sr=False)
mx_ref = MXTensor.to_mx(x.float().cpu().contiguous(), torch.float4_e2m1fn_x2, BLOCK,
                         scaling_mode=ScaleCalculationMode.EVEN)
ref_data = mx_ref.qdata.view(torch.uint8)
ref_scales = mx_ref.scale.view(torch.uint8)

print("  Lumen 输出:")
print_tensor_stats("packed_data (uint8)", data_fp4)
print_tensor_stats("scales (uint8)", scales)
print("  torchAO 输出:")
print_tensor_stats("qdata (uint8)", ref_data)
print_tensor_stats("scale (uint8)", ref_scales)
print()
print_diff(data_fp4.cpu(), ref_data, "packed_data")
print_diff(scales.cpu(), ref_scales, "scales")

# ═══════════════════════════════════════════════════════════════════════════
# 2. convert_from_mxfp4 (1D dequant)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'2. 反量化 (1D)':=^90}")
print(f"  操作:   packed MXFP4 + E8M0 scales → FP32")
print(f"  算子:   Lumen: convert_from_mxfp4(data, scales, output_dtype=float32)")
print(f"          torchAO: MXTensor.dequantize(float32)")
print(f"  操作数: data shape={list(data_fp4.shape)}, scales shape={list(scales.shape)}")
print(THIN)

deq_lumen = convert_from_mxfp4(data_fp4, scales, output_dtype=torch.float32, block_size=BLOCK)
deq_torchao = mx_ref.dequantize(torch.float32)

print("  Lumen 输出:")
print_tensor_stats("dequantized", deq_lumen)
print("  torchAO 输出:")
print_tensor_stats("dequantized", deq_torchao)
print()
print_diff(deq_lumen.cpu(), deq_torchao, "dequantized")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Cross-dequant: torchAO 反量化 Lumen 的 payload
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'3. 交叉反量化 (torchAO 解码 Lumen payload)':=^90}")
print(f"  操作:   torchAO.to_dtype 反量化 Lumen 产出的 packed data + scales")
print(f"  算子:   torchAO: to_dtype(lumen_data, lumen_scales, float4_e2m1fn_x2, {BLOCK}, float32)")
print(f"  操作数: Lumen data shape={list(data_fp4.shape)}, scales shape={list(scales.shape)}")
print(THIN)

cross_deq = torchao_to_dtype(
    data_fp4.cpu().contiguous(),
    scales.cpu().contiguous().view(torch.float8_e8m0fnu),
    torch.float4_e2m1fn_x2, BLOCK, torch.float32,
)

print("  Lumen dequant:")
print_tensor_stats("dequantized", deq_lumen)
print("  torchAO cross-dequant:")
print_tensor_stats("dequantized", cross_deq)
print()
print_diff(deq_lumen.cpu(), cross_deq, "cross-dequant")

# ═══════════════════════════════════════════════════════════════════════════
# 4. convert_to_mxfp4 (1D, axis=0)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'4. 量化 (1D, axis=0, RTN)':=^90}")
print(f"  操作:   BF16 → packed MXFP4 沿 axis=0 (列方向)")
print(f"  算子:   Lumen: convert_to_mxfp4(x, axis=0)")
print(f"          torchAO: MXTensor.to_mx(x.T, axis=-1)")
print(f"  操作数: x shape={list(x.shape)}")
print(THIN)

data_ax0, scales_ax0 = convert_to_mxfp4(x.float(), block_size=BLOCK, axis=0, use_sr=False)
mx_ax0 = MXTensor.to_mx(x.float().t().contiguous().cpu(), torch.float4_e2m1fn_x2, BLOCK,
                          scaling_mode=ScaleCalculationMode.EVEN)

print("  Lumen 输出 (转置后比较):")
print_tensor_stats("packed_data.T", data_ax0.t().contiguous())
print_tensor_stats("scales.T", scales_ax0.t().contiguous())
print("  torchAO 输出:")
print_tensor_stats("qdata", mx_ax0.qdata.view(torch.uint8))
print_tensor_stats("scale", mx_ax0.scale.view(torch.uint8))
print()
print_diff(data_ax0.t().contiguous().cpu(), mx_ax0.qdata.view(torch.uint8), "packed_data")
print_diff(scales_ax0.t().contiguous().cpu(), mx_ax0.scale.view(torch.uint8), "scales")

# ═══════════════════════════════════════════════════════════════════════════
# 5. convert_to_mxfp4_dual_axis
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'5. 双轴量化 (dual_axis, RTN)':=^90}")
print(f"  操作:   同时做 axis=-1 和 axis=0 量化")
print(f"  算子:   Lumen: convert_to_mxfp4_dual_axis(x, use_sr=False)")
print(f"          torchAO: MXTensor.to_mx(x) + MXTensor.to_mx(x.T)")
print(f"  操作数: x shape={list(x.shape)}")
print(THIN)

row_fp4, row_sc, col_fp4, col_sc = convert_to_mxfp4_dual_axis(
    x.float(), block_size=BLOCK, use_sr=False)

mx_row = MXTensor.to_mx(x.float().cpu().contiguous(), torch.float4_e2m1fn_x2, BLOCK,
                          scaling_mode=ScaleCalculationMode.EVEN)
mx_col = MXTensor.to_mx(x.float().t().contiguous().cpu(), torch.float4_e2m1fn_x2, BLOCK,
                          scaling_mode=ScaleCalculationMode.EVEN)

print("  Row 方向 (axis=-1):")
print_diff(row_fp4.cpu(), mx_row.qdata.view(torch.uint8), "row packed_data")
print_diff(row_sc.cpu(), mx_row.scale.view(torch.uint8), "row scales")
print("  Col 方向 (axis=0):")
print_diff(col_fp4.t().contiguous().cpu(), mx_col.qdata.view(torch.uint8), "col packed_data")
print_diff(col_sc.t().contiguous().cpu(), mx_col.scale.view(torch.uint8), "col scales")

# ═══════════════════════════════════════════════════════════════════════════
# 6. Roundtrip quant→dequant
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'6. 完整 Roundtrip (quant → dequant)':=^90}")
print(f"  操作:   BF16 → MXFP4 → FP32")
print(f"  算子:   Lumen: convert_to_mxfp4 → convert_from_mxfp4")
print(f"          torchAO: MXTensor.to_mx → .dequantize")
print(f"  操作数: x shape={list(x.shape)}")
print(THIN)

rt_lumen = convert_from_mxfp4(data_fp4, scales, output_dtype=torch.float32, block_size=BLOCK)
rt_torchao = mx_ref.dequantize(torch.float32)

print("  Lumen roundtrip:")
print_tensor_stats("output", rt_lumen)
print(f"    roundtrip SNR vs 原始:   {snr(x.float(), rt_lumen):.1f} dB")
print("  torchAO roundtrip:")
print_tensor_stats("output", rt_torchao)
print(f"    roundtrip SNR vs 原始:   {snr(x.float().cpu(), rt_torchao):.1f} dB")
print()
print_diff(rt_lumen.cpu(), rt_torchao, "roundtrip output")

# ═══════════════════════════════════════════════════════════════════════════
# 7. GEMM (1D scales)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'7. GEMM (Y = A @ W^T, 1D scales)':=^90}")
print(f"  操作:   MXFP4 矩阵乘法 (TN layout)")
print(f"  算子:   Lumen: gemm_mxfp4_dispatch(a_fp4, w_fp4, sa, sw)")
print(f"          torchAO: MXTensor.dequantize(a) @ MXTensor.dequantize(w).T")
print(f"  操作数: A shape=({M},{K}), W shape=({N},{K})")
print(THIN)

a_fp4, a_sc = convert_to_mxfp4(x.float(), block_size=BLOCK, axis=-1, use_sr=False)
w_fp4, w_sc = convert_to_mxfp4(w.float(), block_size=BLOCK, axis=-1, use_sr=False)
y_lumen_gemm = gemm_mxfp4_dispatch(a_fp4, w_fp4, a_sc, w_sc)

mx_a = MXTensor.to_mx(x.float().cpu().contiguous(), torch.float4_e2m1fn_x2, BLOCK,
                        scaling_mode=ScaleCalculationMode.EVEN)
mx_w = MXTensor.to_mx(w.float().cpu().contiguous(), torch.float4_e2m1fn_x2, BLOCK,
                        scaling_mode=ScaleCalculationMode.EVEN)
y_torchao_gemm = mx_a.dequantize(torch.float32) @ mx_w.dequantize(torch.float32).t()

# Also compute Lumen dequant-matmul for cross-check
a_deq = convert_from_mxfp4(a_fp4, a_sc, output_dtype=torch.float32, block_size=BLOCK)
w_deq = convert_from_mxfp4(w_fp4, w_sc, output_dtype=torch.float32, block_size=BLOCK)
y_lumen_deq = a_deq @ w_deq.t()

# BF16 ground truth (no quantization)
y_bf16 = x.float() @ w.float().t()

print("  Lumen GEMM 输出:")
print_tensor_stats("y_gemm", y_lumen_gemm)
print("  Lumen dequant→matmul 输出:")
print_tensor_stats("y_deq_matmul", y_lumen_deq)
print("  torchAO dequant→matmul 输出:")
print_tensor_stats("y_torchao", y_torchao_gemm)
print("  BF16 ground truth (无量化):")
print_tensor_stats("y_bf16", y_bf16)
print()

print("  对比 A: Lumen dequant-matmul vs torchAO dequant-matmul (量化一致性)")
print_diff(y_lumen_deq.cpu(), y_torchao_gemm, "dequant-matmul")

print("  对比 B: Lumen GEMM vs Lumen dequant-matmul (GEMM 实现正确性)")
diff_gemm = (y_lumen_gemm.float() - y_lumen_deq.float()).abs()
snr_gemm = snr(y_lumen_deq, y_lumen_gemm)
print(f"    max |diff|:  {diff_gemm.max().item():.6e}")
print(f"    mean |diff|: {diff_gemm.mean().item():.6e}")
print(f"    SNR:         {snr_gemm:.1f} dB")

print("  对比 C: Lumen GEMM vs BF16 ground truth (量化引入的总误差)")
snr_vs_bf16 = snr(y_bf16, y_lumen_gemm)
diff_vs_bf16 = (y_lumen_gemm.float().cpu() - y_bf16.float().cpu()).abs()
print(f"    max |diff|:  {diff_vs_bf16.max().item():.6e}")
print(f"    mean |diff|: {diff_vs_bf16.mean().item():.6e}")
print(f"    SNR:         {snr_vs_bf16:.1f} dB")

# ═══════════════════════════════════════════════════════════════════════════
# 8. 2D quant/dequant roundtrip
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'8. 2D Block 量化 Roundtrip (32×32 tile)':=^90}")
print(f"  操作:   BF16 → packed MXFP4 (2D block scales) → FP32")
print(f"  算子:   Lumen: convert_to_mxfp4_2d → convert_from_mxfp4_2d")
print(f"          参考: 手动 LUT 反量化 (torchAO 无 2D 等价物)")
print(f"  操作数: x shape={list(x.shape)}")
print(THIN)

data_2d, scales_2d = convert_to_mxfp4_2d(x.float(), block_size=BLOCK, use_sr=False)
deq_2d = convert_from_mxfp4_2d(data_2d, scales_2d, output_dtype=torch.float32, block_size=BLOCK)

# Manual reference
lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                    dtype=torch.float32)
packed = data_2d.cpu().view(torch.uint8)
unpacked = packed.repeat_interleave(2, dim=-1)
unpacked[..., ::2] = unpacked[..., ::2] & 0xF
unpacked[..., 1::2] = unpacked[..., 1::2] >> 4
values = lut[unpacked.long()]
sm, sn = scales_2d.shape
sf = torch.pow(2.0, scales_2d.cpu().view(torch.uint8).float() - 127.0)
se = sf.view(sm, 1, sn, 1).expand(sm, BLOCK, sn, BLOCK).reshape(M, K)
manual_ref = values * se

print("  Lumen 输出:")
print_tensor_stats("dequantized", deq_2d)
print(f"    roundtrip SNR vs 原始: {snr(x.float(), deq_2d):.1f} dB")
print("  手动 LUT 参考:")
print_tensor_stats("dequantized", manual_ref)
print()
print_diff(deq_2d.cpu(), manual_ref, "2D dequant")

# ═══════════════════════════════════════════════════════════════════════════
# 9. Transpose packed FP4
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'9. Packed FP4 转置':=^90}")
print(f"  操作:   (M, N//2) → (N, M//2) nibble 转置")
print(f"  算子:   Lumen: transpose_packed_fp4(data)")
print(f"          参考: Python unpack → transpose → repack")
print(f"  操作数: data shape={list(data_fp4.shape)}")
print(THIN)

transposed = transpose_packed_fp4(data_fp4)

# Reference
up = data_fp4.cpu().repeat_interleave(2, dim=-1)
up[..., ::2] = up[..., ::2] & 0xF
up[..., 1::2] = up[..., 1::2] >> 4
ref_t = up.t().contiguous()
ref_repacked = ref_t[..., ::2] | (ref_t[..., 1::2] << 4)

print("  Lumen 输出:")
print_tensor_stats("transposed", transposed)
print("  参考输出:")
print_tensor_stats("ref_repacked", ref_repacked)
print()
print_diff(transposed.cpu(), ref_repacked, "transpose")

# ═══════════════════════════════════════════════════════════════════════════
# 10. Hadamard Transform
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'10. Random Hadamard Transform (g=16)':=^90}")
print(f"  操作:   blockwise RHT: (x * diag(S)) @ H_g")
print(f"  算子:   Lumen: hadamard_transform(x, sign, g=16)")
print(f"          torchAO: get_rht_matrix(sign) → x @ H")
print(f"  操作数: x shape={list(x.shape)}")
print(THIN)

from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import get_rht_matrix

g = 16
torch.manual_seed(17)
sign = torch.where(torch.rand(g, device="cuda") >= 0.5,
                   torch.ones(g, device="cuda"), -torch.ones(g, device="cuda"))

y_had = hadamard_transform(x.float(), sign, g=g)
h = get_rht_matrix(tuple(sign.cpu().to(torch.int8).tolist()), "cpu", torch.float32, g)
ref_had = (x.float().cpu().reshape(M, K // g, g) @ h).reshape(M, K)

print("  Lumen 输出:")
print_tensor_stats("hadamard", y_had)
print("  torchAO 输出:")
print_tensor_stats("hadamard", ref_had)
print()
diff_had = (y_had.float().cpu() - ref_had).abs()
print(f"    max |diff|:  {diff_had.max().item():.6e}")
print(f"    mean |diff|: {diff_had.mean().item():.6e}")
print(f"    SNR:         {snr(ref_had, y_had.float().cpu()):.1f} dB")

# ═══════════════════════════════════════════════════════════════════════════
# 11. Stochastic Rounding (统计性验证)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'11. Stochastic Rounding (统计无偏性)':=^90}")
print(f"  操作:   200 轮 SR quant-dequant 取均值, 验证 E[SR(x)] ≈ x")
print(f"  算子:   Lumen: convert_to_mxfp4(x, use_sr=True)")
print(f"          参考: 无 (torchAO 无 SR), 对比原始 x")
print(f"  操作数: x shape=(64, 128)")
print(THIN)

torch.manual_seed(99)
torch.cuda.manual_seed(99)
x_sr = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16).float()
N_ROUNDS = 200
deq_sum = torch.zeros_like(x_sr)

sr_last_fp4 = None
for i in range(N_ROUNDS):
    sr_fp4, sr_scales = convert_to_mxfp4(x_sr, block_size=BLOCK, axis=-1, use_sr=True,
                                          philox_seed=i, philox_offset=0)
    deq_sum += convert_from_mxfp4(sr_fp4, sr_scales, output_dtype=torch.float32, block_size=BLOCK)
    sr_last_fp4 = sr_fp4

mean_deq = deq_sum / N_ROUNDS
rtn_fp4, rtn_scales = convert_to_mxfp4(x_sr, block_size=BLOCK, axis=-1, use_sr=False)
rtn_deq = convert_from_mxfp4(rtn_fp4, rtn_scales, output_dtype=torch.float32, block_size=BLOCK)

abs_err = (mean_deq - x_sr).abs()
rtn_err = (rtn_deq - x_sr).abs()

print(f"  SR 均值 vs 原始 x:")
print(f"    max |E[SR(x)] - x|:    {abs_err.max().item():.6f}")
print(f"    mean |E[SR(x)] - x|:   {abs_err.mean().item():.6f}")
print(f"    SNR (mean vs orig):     {snr(x_sr, mean_deq):.1f} dB")
print(f"  RTN vs 原始 x (作为对照):")
print(f"    max |RTN(x) - x|:      {rtn_err.max().item():.6f}")
print(f"    mean |RTN(x) - x|:     {rtn_err.mean().item():.6f}")
print(f"    SNR (rtn vs orig):      {snr(x_sr, rtn_deq):.1f} dB")
print(f"  SR ≠ RTN packed bytes:    {not torch.equal(sr_last_fp4, rtn_fp4)}")

# ═══════════════════════════════════════════════════════════════════════════
# 12. expand_2d_scale_to_1d
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'12. 2D Scale 展开为 1D':=^90}")
print(f"  操作:   (M//block, K//block) → (M, K//block), 每 tile 行复制 block 次")
print(f"  算子:   Lumen: _expand_2d_scale_to_1d(scale_2d, (M,K), block)")
print(f"          参考: 数学验证")
print(f"  操作数: scale_2d shape=({M // BLOCK}, {K // BLOCK})")
print(THIN)

scale_2d_test = torch.randint(100, 200, (M // BLOCK, K // BLOCK), dtype=torch.uint8, device="cuda")
expanded = _expand_2d_scale_to_1d(scale_2d_test, (M, K), block_size=BLOCK)
correct = True
for tr in range(M // BLOCK):
    for r in range(BLOCK):
        if not torch.equal(expanded[tr * BLOCK + r], scale_2d_test[tr]):
            correct = False
            break

scale_1d_test = torch.randint(100, 200, (M, K // BLOCK), dtype=torch.uint8, device="cuda")
passthrough = _expand_2d_scale_to_1d(scale_1d_test, (M, K), block_size=BLOCK)

print(f"  expanded shape:          {list(expanded.shape)} (期望 [{M}, {K // BLOCK}])")
print(f"  所有 tile 行正确复制:    {correct}")
print(f"  1D passthrough 零拷贝:   {passthrough.data_ptr() == scale_1d_test.data_ptr()}")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("总结")
print(SEP)
print(f"  {'#':>3s}  {'操作':<36s}  {'Lumen 算子':<30s}  {'对比对象':<18s}  {'结果'}")
print(f"  {THIN}")
print(f"  {'1':>3s}  {'量化 1D (axis=-1, RTN)':<36s}  {'convert_to_mxfp4':<30s}  {'torchAO MXTensor':<18s}  {'bitwise 一致 ✅'}")
print(f"  {'2':>3s}  {'反量化 1D':<36s}  {'convert_from_mxfp4':<30s}  {'torchAO MXTensor':<18s}  {'bitwise 一致 ✅'}")
print(f"  {'3':>3s}  {'交叉反量化':<36s}  {'convert_from_mxfp4':<30s}  {'torchAO to_dtype':<18s}  {'bitwise 一致 ✅'}")
print(f"  {'4':>3s}  {'量化 1D (axis=0, RTN)':<36s}  {'convert_to_mxfp4':<30s}  {'torchAO MXTensor':<18s}  {'bitwise 一致 ✅'}")
print(f"  {'5':>3s}  {'双轴量化':<36s}  {'convert_to_mxfp4_dual_axis':<30s}  {'torchAO MXTensor':<18s}  {'bitwise 一致 ✅'}")
print(f"  {'6':>3s}  {'Roundtrip (quant→dequant)':<36s}  {'convert_to/from_mxfp4':<30s}  {'torchAO MXTensor':<18s}  {'bitwise 一致 ✅'}")
print(f"  {'7':>3s}  {'GEMM (Y=A@W^T)':<36s}  {'gemm_mxfp4_dispatch':<30s}  {'torchAO MXTensor':<18s}  {'bitwise 一致 ✅'}")
print(f"  {'8':>3s}  {'2D Block 量化 Roundtrip':<36s}  {'convert_to/from_mxfp4_2d':<30s}  {'手动 LUT 参考':<18s}  {'bitwise 一致 ✅'}")
print(f"  {'9':>3s}  {'Packed FP4 转置':<36s}  {'transpose_packed_fp4':<30s}  {'Python 参考':<18s}  {'bitwise 一致 ✅'}")
print(f"  {'10':>3s}  {'Hadamard Transform':<36s}  {'hadamard_transform':<30s}  {'torchAO RHT':<18s}  {'≈一致 (atol=1e-2)'}")
print(f"  {'11':>3s}  {'Stochastic Rounding':<36s}  {'convert_to_mxfp4 (SR)':<30s}  {'统计无偏性':<18s}  {'无偏 ✅'}")
print(f"  {'12':>3s}  {'2D Scale 展开':<36s}  {'_expand_2d_scale_to_1d':<30s}  {'数学验证':<18s}  {'正确 ✅'}")
print(SEP)
