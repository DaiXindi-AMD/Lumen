# MXFP4 精度报告: Lumen vs torchAO

| 项 | 值 |
|---|---|
| 硬件 | AMD Instinct MI350X VF |
| 矩阵尺寸 (第 1–12 节) | M=128, K=256, N=128, block_size=32 |
| 矩阵尺寸 (第 13 节) | Qwen3-8B 生产 shape, 见该节 |
| 输入 dtype | `torch.bfloat16` |


#### 1. 量化 (1D, axis=-1, RTN)

- **操作**: BF16 → packed MXFP4 (uint8) + E8M0 scales (uint8)
- **Lumen 算子**: `convert_to_mxfp4(x, block_size=32, axis=-1, use_sr=False)`
- **对比算子**: torchAO `MXTensor.to_mx(x, float4_e2m1fn_x2, 32, EVEN)`
- **操作数**: x shape=`[128, 256]`, dtype=`torch.bfloat16`

| 来源 | 张量 | shape | dtype | min | max | mean |
|---|---|---|---|---|---|---|
| Lumen | `packed_data` | `[128, 128]` | `uint8` | 0.0000 | 255.0000 | 117.3615 |
| Lumen | `scales` | `[128, 8]` | `uint8` | 125.0000 | 127.0000 | 125.9492 |
| torchAO | `qdata` | `[128, 128]` | `uint8` | 0.0000 | 255.0000 | 117.3615 |
| torchAO | `scale` | `[128, 8]` | `uint8` | 125.0000 | 127.0000 | 125.9492 |

| 对比项 | bitwise 一致元素 | max \|diff\| | mean \|diff\| | SNR |
|---|---|---|---|---|
| packed_data | 16384/16384 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |
| scales | 1024/1024 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |


#### 2. 反量化 (1D)

- **操作**: packed MXFP4 + E8M0 scales → FP32
- **Lumen 算子**: `convert_from_mxfp4(data, scales, output_dtype=float32)`
- **对比算子**: torchAO `MXTensor.dequantize(float32)`
- **操作数**: data shape=`[128, 128]`, scales shape=`[128, 8]`

| 来源 | 张量 | shape | dtype | min | max | mean |
|---|---|---|---|---|---|---|
| Lumen | `dequantized` | `[128, 256]` | `float32` | -4.0000 | 4.0000 | -0.0052 |
| torchAO | `dequantized` | `[128, 256]` | `float32` | -4.0000 | 4.0000 | -0.0052 |

| 对比项 | bitwise 一致元素 | max \|diff\| | mean \|diff\| | SNR |
|---|---|---|---|---|
| dequantized | 32768/32768 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |


#### 3. 交叉反量化 (torchAO 解码 Lumen payload)

- **操作**: torchAO.to_dtype 反量化 Lumen 产出的 packed data + scales
- **Lumen 算子**: `convert_from_mxfp4`
- **对比算子**: torchAO `to_dtype(lumen_data, lumen_scales, float4_e2m1fn_x2, 32, float32)`
- **操作数**: Lumen data shape=`[128, 128]`, scales shape=`[128, 8]`

| 来源 | 张量 | shape | dtype | min | max | mean |
|---|---|---|---|---|---|---|
| Lumen | `dequantized` | `[128, 256]` | `float32` | -4.0000 | 4.0000 | -0.0052 |
| torchAO cross-dequant | `dequantized` | `[128, 256]` | `float32` | -4.0000 | 4.0000 | -0.0052 |

| 对比项 | bitwise 一致元素 | max \|diff\| | mean \|diff\| | SNR |
|---|---|---|---|---|
| cross-dequant | 32768/32768 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |


#### 4. 量化 (1D, axis=0, RTN)

- **操作**: BF16 → packed MXFP4 沿 axis=0 (列方向)
- **Lumen 算子**: `convert_to_mxfp4(x, axis=0)`
- **对比算子**: torchAO `MXTensor.to_mx(x.T, axis=-1)`
- **操作数**: x shape=`[128, 256]`

| 来源 | 张量 | shape | dtype | min | max | mean |
|---|---|---|---|---|---|---|
| Lumen (转置后) | `packed_data.T` | `[256, 64]` | `uint8` | 0.0000 | 255.0000 | 117.7158 |
| Lumen (转置后) | `scales.T` | `[256, 4]` | `uint8` | 125.0000 | 127.0000 | 125.9346 |
| torchAO | `qdata` | `[256, 64]` | `uint8` | 0.0000 | 255.0000 | 117.7158 |
| torchAO | `scale` | `[256, 4]` | `uint8` | 125.0000 | 127.0000 | 125.9346 |

| 对比项 | bitwise 一致元素 | max \|diff\| | mean \|diff\| | SNR |
|---|---|---|---|---|
| packed_data | 16384/16384 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |
| scales | 1024/1024 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |


#### 5. 双轴量化 (dual_axis, RTN)

- **操作**: 同时做 axis=-1 和 axis=0 量化
- **Lumen 算子**: `convert_to_mxfp4_dual_axis(x, use_sr=False)`
- **对比算子**: torchAO `MXTensor.to_mx(x)` + `MXTensor.to_mx(x.T)`
- **操作数**: x shape=`[128, 256]`

| 对比项 | bitwise 一致元素 | max \|diff\| | mean \|diff\| | SNR |
|---|---|---|---|---|
| row (axis=-1) packed_data | 16384/16384 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |
| row (axis=-1) scales | 1024/1024 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |
| col (axis=0) packed_data | 16384/16384 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |
| col (axis=0) scales | 1024/1024 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |


#### 6. 完整 Roundtrip (quant → dequant)

- **操作**: BF16 → MXFP4 → FP32
- **Lumen 算子**: `convert_to_mxfp4` → `convert_from_mxfp4`
- **对比算子**: torchAO `MXTensor.to_mx` → `.dequantize`
- **操作数**: x shape=`[128, 256]`

| 来源 | 张量 | shape | dtype | min | max | mean | roundtrip SNR vs 原始 x |
|---|---|---|---|---|---|---|---|
| Lumen | `output` | `[128, 256]` | `float32` | -4.0000 | 4.0000 | -0.0052 | 19.0 dB |
| torchAO | `output` | `[128, 256]` | `float32` | -4.0000 | 4.0000 | -0.0052 | 19.0 dB |

| 对比项 | bitwise 一致元素 | max \|diff\| | mean \|diff\| | SNR |
|---|---|---|---|---|
| roundtrip output | 32768/32768 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |


#### 7. GEMM (Y = A @ W^T, 1D scales)

- **操作**: MXFP4 矩阵乘法 (TN layout)
- **Lumen 算子**: `gemm_mxfp4_dispatch(a_fp4, w_fp4, sa, sw)`
- **对比算子**: torchAO `MXTensor.dequantize(a) @ MXTensor.dequantize(w).T`
- **操作数**: A shape=`(128, 256)`, W shape=`(128, 256)`

| 来源 | 张量 | shape | dtype | min | max | mean |
|---|---|---|---|---|---|---|
| Lumen GEMM | `y_gemm` | `[128, 128]` | `bfloat16` | -56.5000 | 61.2500 | -0.0262 |
| Lumen dequant→matmul | `y_deq_matmul` | `[128, 128]` | `float32` | -56.6250 | 61.2500 | -0.0261 |
| torchAO dequant→matmul | `y_torchao` | `[128, 128]` | `float32` | -56.6250 | 61.2500 | -0.0261 |
| BF16 ground truth (无量化) | `y_bf16` | `[128, 128]` | `float32` | -60.7451 | 63.3518 | -0.0213 |

| 对比 | 内容 | 验的是 | bitwise 一致元素 | max \|diff\| | mean \|diff\| | SNR |
|---|---|---|---|---|---|---|
| A | Lumen dequant-matmul vs torchAO dequant-matmul | 量化器一致性 | 16384/16384 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |
| B | Lumen GEMM vs Lumen dequant-matmul | GEMM 实现正确性 | 11215/16384 (68.45%) | 1.250e-01 | 1.423e-02 | 55.1 dB |
| C | Lumen GEMM vs BF16 ground truth | 量化引入的总误差 | 0/16384 (0.00%) | 1.055e+01 | 2.033e+00 | 16.0 dB |

> 对比 A 两侧都是先反量化再用 torch matmul, 验的是量化器, **不经过** `gemm_mxfp4_dispatch`; 对比 B 才真正跑 GEMM, 累加顺序不同, 不应期望 bitwise。

| 本节使用的后端 | 结论 |
|---|---|
| asm | 不可用 — (128,128,256) 不在 AITER tuned 表中 |
| Triton | 使用 |

> 这个 shape 只覆盖 Triton 路径; 生产用的 asm 路径见第 13 节。


#### 8. 2D Block 量化 Roundtrip (32×32 tile)

- **操作**: BF16 → packed MXFP4 (2D block scales) → FP32
- **Lumen 算子**: `convert_to_mxfp4_2d` → `convert_from_mxfp4_2d`
- **对比算子**: 手动 LUT 反量化 (torchAO 无 2D 等价物)
- **操作数**: x shape=`[128, 256]`

| 来源 | 张量 | shape | dtype | min | max | mean | roundtrip SNR vs 原始 x |
|---|---|---|---|---|---|---|---|
| Lumen | `dequantized` | `[128, 256]` | `float32` | -4.0000 | 4.0000 | -0.0056 | 17.8 dB |
| 手动 LUT 参考 | `dequantized` | `[128, 256]` | `float32` | -4.0000 | 4.0000 | -0.0056 | — |

| 对比项 | bitwise 一致元素 | max \|diff\| | mean \|diff\| | SNR |
|---|---|---|---|---|
| 2D dequant | 32768/32768 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |


#### 9. Packed FP4 转置

- **操作**: (M, N//2) → (N, M//2) nibble 转置
- **Lumen 算子**: `transpose_packed_fp4(data)`
- **对比算子**: Python unpack → transpose → repack
- **操作数**: data shape=`[128, 128]`

| 来源 | 张量 | shape | dtype | min | max | mean |
|---|---|---|---|---|---|---|
| Lumen | `transposed` | `[256, 64]` | `uint8` | 0.0000 | 255.0000 | 116.9971 |
| 参考 | `ref_repacked` | `[256, 64]` | `uint8` | 0.0000 | 255.0000 | 116.9971 |

| 对比项 | bitwise 一致元素 | max \|diff\| | mean \|diff\| | SNR |
|---|---|---|---|---|
| transpose | 16384/16384 (100.00%) | 0.000e+00 | 0.000e+00 | inf (bitwise 一致) |


#### 10. Random Hadamard Transform (g=16)

- **操作**: blockwise RHT: (x * diag(S)) @ H_g
- **Lumen 算子**: `hadamard_transform(x, sign, g=16)`
- **对比算子**: torchAO `get_rht_matrix(sign)` → `x @ H`
- **操作数**: x shape=`[128, 256]`

| 来源 | 张量 | shape | dtype | min | max | mean |
|---|---|---|---|---|---|---|
| Lumen | `hadamard` | `[128, 256]` | `float32` | -3.6895 | 3.7975 | -0.0038 |
| torchAO | `hadamard` | `[128, 256]` | `float32` | -3.6895 | 3.7975 | -0.0038 |

| 对比项 | bitwise 一致元素 | max \|diff\| | mean \|diff\| | SNR |
|---|---|---|---|---|
| hadamard | 32766/32768 (99.99%) | 5.960e-08 | 3.638e-12 | 186.6 dB |

> Lumen 在 GPU 上做块内蝶形运算, 参考实现是 CPU 上的稠密矩阵乘, 浮点结合律不同, 不应期望 bitwise。


#### 11. Stochastic Rounding (统计无偏性)

- **操作**: 200 轮 SR quant-dequant 取均值, 验证 E[SR(x)] ≈ x
- **Lumen 算子**: `convert_to_mxfp4(x, use_sr=True)`
- **对比算子**: 无 (torchAO 无 SR), 对比原始 x
- **操作数**: x shape=`[64, 128]`

| 对比原始 x | max \|err\| | mean \|err\| | SNR |
|---|---|---|---|
| SR 200 轮均值 E[SR(x)] | 0.281250 | 0.008452 | 35.5 dB |
| RTN 单次 (对照) | 0.500000 | 0.085620 | 19.0 dB |

| 检查项 | 结果 |
|---|---|
| SR ≠ RTN packed bytes | True |
| SR 均值误差 < RTN 误差 | True |


#### 12. 2D Scale 展开为 1D

- **操作**: (M//block, K//block) → (M, K//block), 每 tile 行复制 block 次
- **Lumen 算子**: `_expand_2d_scale_to_1d(scale_2d, (M, K), block)`
- **对比算子**: 数学验证
- **操作数**: scale_2d shape=`(4, 8)`

| 检查项 | 结果 |
|---|---|
| expanded shape | `[128, 8]` (期望 `[128, 8]`) |
| 所有 tile 行正确复制 | True |
| 1D passthrough 零拷贝 | True |


#### 13. 生产 shape 的 GEMM 后端 (asm vs Triton)

- **操作**: 在 Qwen3-8B 实际发出的 11 个 MXFP4 GEMM shape 上比对两个后端
- **Lumen 算子**: `_gemm_mxfp4_aiter_asm` (AITER 汇编) vs `_gemm_mxfp4_aiter` (Triton)
- **对比算子**: 互为参考 — 两者若不逐位一致, 说明 dispatch 会引入后端相关的数值差异
- **操作数**: 见下表 (Y = A(M,K) @ W(N,K)^T, block_size=32)

> 这一节存在的原因: 上面各节用的是 (128, 256, 128), 该 shape 不在 AITER tuned 表中, asm 会被拒绝, 因此只覆盖了 Triton。补表后生产上多数 shape 走 asm, 需要单独验证。

| M | N | K | asm 可用 | asm vs Triton | SNR vs dequant-matmul |
|---|---|---|---|---|---|
| 4096 | 4096 | 16384 | 是 | bitwise 一致 ✅ | 55.6 dB |
| 4096 | 12288 | 16384 | 是 | bitwise 一致 ✅ | 55.6 dB |
| 6144 | 4096 | 16384 | 是 | bitwise 一致 ✅ | 55.6 dB |
| 16384 | 4096 | 4096 | 是 | bitwise 一致 ✅ | 55.6 dB |
| 16384 | 4096 | 6144 | 是 | bitwise 一致 ✅ | 55.6 dB |
| 16384 | 4096 | 12288 | 是 | bitwise 一致 ✅ | 55.6 dB |
| 16384 | 4096 | 24576 | 是 | bitwise 一致 ✅ | 55.6 dB |
| 16384 | 6144 | 4096 | 是 | bitwise 一致 ✅ | 55.6 dB |
| 16384 | 12288 | 4096 | 是 | bitwise 一致 ✅ | 跳过 (fp32 参考过大) |
| 16384 | 24576 | 4096 | 是 | bitwise 一致 ✅ | 跳过 (fp32 参考过大) |
| 24576 | 4096 | 16384 | 是 | bitwise 一致 ✅ | 55.6 dB |

| 结论 | 值 |
|---|---|
| asm 与 Triton 逐位一致 | 11/11 个 shape |
| asm vs dequant-matmul SNR | 最低 55.6 dB, 最高 55.6 dB |

> SNR 那一列在所有 shape 上几乎相同, 因为它被 GEMM 输出的 bf16 舍入定住了 (bf16 8 位尾数 → ~59 dB 上限), 而不是在反映累加误差。真正有区分度的是 `asm vs Triton` 一列: 两个后端逐位一致, 意味着 dispatch 按速度选后端不会改变数值结果。


#### 总结

| # | 操作 | Lumen 算子 | 对比对象 | 结果 |
|---|---|---|---|---|
| 1 | 量化 1D (axis=-1, RTN) | `convert_to_mxfp4` | torchAO MXTensor | bitwise 一致 ✅ |
| 2 | 反量化 1D | `convert_from_mxfp4` | torchAO MXTensor | bitwise 一致 ✅ |
| 3 | 交叉反量化 | `convert_from_mxfp4` | torchAO to_dtype | bitwise 一致 ✅ |
| 4 | 量化 1D (axis=0, RTN) | `convert_to_mxfp4` | torchAO MXTensor | bitwise 一致 ✅ |
| 5 | 双轴量化 | `convert_to_mxfp4_dual_axis` | torchAO MXTensor | bitwise 一致 ✅ |
| 6 | Roundtrip (quant→dequant) | `convert_to/from_mxfp4` | torchAO MXTensor | bitwise 一致 ✅ |
| 7a | GEMM 量化一致性 (dequant→matmul) | `convert_to/from_mxfp4` | torchAO MXTensor | bitwise 一致 ✅ |
| 7b | GEMM 实现正确性 | `gemm_mxfp4_dispatch` | Lumen dequant-matmul | 55.1 dB ✅ |
| 8 | 2D Block 量化 Roundtrip | `convert_to/from_mxfp4_2d` | 手动 LUT 参考 | bitwise 一致 ✅ |
| 9 | Packed FP4 转置 | `transpose_packed_fp4` | Python 参考 | bitwise 一致 ✅ |
| 10 | Hadamard Transform | `hadamard_transform` | torchAO RHT | 186.6 dB ✅ |
| 11 | Stochastic Rounding | `convert_to_mxfp4` (SR) | 统计无偏性 | 无偏 ✅ |
| 12 | 2D Scale 展开 | `_expand_2d_scale_to_1d` | 数学验证 | 正确 ✅ |
| 13 | 生产 shape asm vs Triton (11/11) | `_gemm_mxfp4_aiter_asm` | plain Triton | bitwise 一致 ✅ |

> 第 1–12 节使用 (M=128, K=256, N=128)。该 shape 不在 AITER tuned 表中, asm 后端会被拒绝, 因此第 7b 项只覆盖 Triton 实现; 生产实际走的 asm 路径由第 13 项在真实 shape 上覆盖。

