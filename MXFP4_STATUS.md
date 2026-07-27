# MXFP4 分支状态（`feature/mxfp4`）

> 验证 Lumen **MXFP4 量化**实现的正确性，方法是与 **torchAO** 逐算子对比精度。
> **结论：10 / 10 测试全部通过，汇编路径与 torchAO 逐位一致。**

---

## 1. 任务目标
- 仓库：`~/Lumen`，分支：`feature/mxfp4`。
- 参照实现：**torchAO**（源码 `~/ao`，已装为 editable `torchao 0.18.0`）。
- 对比测试：`tests/ops/test_quantize.py`。

## 2. 硬件 / 环境
- 8× AMD MI350X（gfx950 / CDNA4），ROCm 7.2，`torch 2.13.0+rocm7.2`，`triton 3.7.0`。
- `import torchao` 可用。

## 3. 环境准备（避免重复踩坑）
1. **AITER 版本必须用 fork 版**：子模块 `third_party/aiter`，pin 在
   `ZhangDanyang-AMD/aiter@43d5f579`（分支 `lumen/triton_kernels`）。
   ```bash
   git submodule update --init third_party/aiter
   ```
2. **Triton 3.7 导入 bug 需绕过**：`/tmp/tritonshim/sitecustomize.py`
   （让 `triton.runtime.jit.get_def_col_number` 失败时返回 1）。

## 4. 复现命令
```bash
cd ~/Lumen
PYTHONPATH=/tmp/tritonshim:$PWD/third_party/aiter \
  python -m pytest tests/ops/test_quantize.py -v -k mxfp4 -p no:cacheprovider
```

## 5. 已修复的 Bug
除注明外均在 `lumen/kernels/mxfp4.py`。

1. **编译**：`min_frag = (F32_MIN_NORMAL).to(tl.float32)` → 改为 `min_frag = F32_MIN_NORMAL`。
2. **编译**：`tl.zeros_like(abs_val, dtype=tl.uint8)` → 改为 `tl.zeros_like(abs_val).to(tl.uint8)`。
3. **编译**：切片索引 `x[:, :, 0:1]`（Triton 3.7 不支持）→ 改用 `tl.split`。
4. **汇编崩溃**：inline asm 输出 `dtype=tl.uint8`（8 位无法映射到 VGPR）→ 改为 `dtype=tl.uint32` + mask。
5. **★数值核心 bug**（`_calculate_fp4_scales`）：无符号整数下溢导致 scale = ∞ → 输出全 NaN。
   修复：减法前 `.to(tl.int32)`。
6. **API 导出缺失**（`lumen/ops/quantize/__init__.py`）：补上 MXFP4 函数导出。
7. **Hadamard kernel 编译失败**：Triton 3.7 的 `tl.static_range` 归纳变量被 trace 为 tensor，
   无法用于 `tl.reshape`。修复：改用 PyTorch 显式矩阵乘法 `(x * sign) @ H_g`。

## 6. 测试详情：10 / 10 通过

### 对 TorchAO 的精度对比

| 路径 | scale 匹配 | data 匹配 | 结论 |
|---|---|---|---|
| **汇编路径**（gfx950 默认） | 100% | **100%** | 与 torchAO **逐位一致** |
| 软件路径 | 100% | 98.5% | 1.5% 差异来自手动取整的边界 |

### 逐个测试说明

| # | 测试名 | 对比的 Lumen 算子 | 对比的 TorchAO 参照 | 验证内容 |
|---|---|---|---|---|
| 1-2 | `test_mxfp4_1d_rtn_vs_torchao_mxtensor` ×2 shape | `convert_to_mxfp4` (1D, RTN) | `MXTensor.to_mx` (FP4 E2M1, EVEN mode) | scale **逐位一致** (`atol=0`) + packed data **逐位一致** + dequant **逐位一致** |
| 3-4 | `test_mxfp4_1d_rtn_cross_dequant_with_torchao` ×2 shape | `convert_from_mxfp4` (Lumen dequant) | `torchao.to_dtype` (TorchAO dequant) | Lumen 量化数据交给 TorchAO 反量化，结果与 Lumen 自身反量化 **逐位一致** (`atol=0`) |
| 5-6 | `test_mxfp4_2d_rtn_roundtrip_snr` ×2 shape | `convert_to_mxfp4_2d` + `convert_from_mxfp4_2d` (32×32 block) | 无（TorchAO 无 2D block MXTensor） | 自身 roundtrip SNR ≥ 4 dB，无 NaN/Inf |
| 7-8 | `test_mxfp4_transpose_packed_matches_unpack_reference` ×2 shape | `transpose_packed_fp4` | Python unpack→transpose→repack 参照 | Triton kernel 输出与纯 Python nibble 解包转置重打包 **逐位一致** (`atol=0`) |
| 9-10 | `test_mxfp4_hadamard_transform_matches_torchao_matrix` ×2 shape | `hadamard_transform` (g=16) | `torchao.get_rht_matrix` → 显式 `x @ H` | Lumen RHT 输出与 TorchAO 16×16 Hadamard 矩阵乘法 **一致** (`atol=1e-2`) |

### 测试覆盖的算子总结

| Lumen 算子 | 功能 | 有 TorchAO 对比 |
|---|---|---|
| `convert_to_mxfp4` | BF16/FP32 → packed MXFP4 + E8M0 scale (1D block) | ✅ MXTensor.to_mx |
| `convert_from_mxfp4` | packed MXFP4 → BF16/FP32 反量化 (1D block) | ✅ torchao.to_dtype |
| `convert_to_mxfp4_2d` | BF16/FP32 → packed MXFP4 + E8M0 scale (2D 32×32 block) | ❌ TorchAO 无 2D 等价 |
| `convert_from_mxfp4_2d` | packed MXFP4 → BF16/FP32 反量化 (2D 32×32 block) | ❌ 同上 |
| `transpose_packed_fp4` | packed nibble 矩阵转置 (M,N/2) → (N,M/2) | ❌ 与纯 Python 参照对比 |
| `hadamard_transform` | blockwise Random Hadamard Transform | ✅ torchao.get_rht_matrix |
| `convert_to_mxfp4_dual_axis` | 同时做行/列两方向量化 | ❌ 未测试 |

## 7. 改动文件
```
lumen/kernels/mxfp4.py          # 编译修复 + 汇编 dtype 修复 + scale 无符号下溢修复
lumen/ops/quantize/__init__.py  # 导出 MXFP4 API
lumen/ops/quantize/ops.py       # Hadamard: Triton kernel → PyTorch matmul
```

## 8. 待办 / 可选
- [x] 修复 Hadamard kernel 使 10/10 通过
- [ ] 固化 Triton shim 绕过（给 aiter fork 提 patch，或写进 conftest）
- [ ] 补充 `convert_to_mxfp4_dual_axis` 测试
- [ ] 若拿到 CDNA4 ISA 白皮书，复核汇编 `v_cvt_scalef32_[sr_]pk_fp4_*` 的 `op_sel` 语义
