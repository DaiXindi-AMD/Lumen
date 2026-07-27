# MXFP4 反向传播性能优化总结

**作者**: Dai, Xindi  
**日期**: 2026-07-27  
**分支**: `worktree-mxfp4-perf-optimization` (基于 `feature/mxfp4-kernel-fusion`)  
**PR**: https://github.com/DaiXindi-AMD/Lumen/pull/1

---

## 1. 背景与动机

Lumen MXFP4 训练在 Qwen3-8B 上收敛正确（val_loss 与 BF16 差距 <1%），但性能比 BF16 慢约 2 倍。瓶颈在于反向传播中的冗余量化和未融合的算子流水线。

**优化前每层操作统计**:

| 阶段   | FP4 量化 | 反量化 | Hadamard | 转置 | FP4 GEMM |
|--------|----------|--------|----------|------|----------|
| 前向   | 2 (X+W)  | 0      | 0        | 0    | 1        |
| DGrad  | 2 (dY + W重新量化) | 0 | 0 | 1 (W packed) | 1 |
| WGrad  | 2 (dY^T + X^T) | 1 (X) | 2 (dY^T + X^T) | 2 (dY, X) | 1 |
| **合计** | **6** | **1** | **2** | **3** | **3** |

核心问题:
1. **DGrad 中重复量化权重**: 前向已将权重从 BF16 量化为 FP4，但反向又从 `ctx.weight_ref`（BF16 主权重）重新量化一次，加上 packed FP4 转置操作
2. **WGrad 中 Hadamard 和量化分离**: Hadamard 变换和 FP4 量化是两个独立的 kernel launch，中间有一次全局内存读写往返
3. **GEMM 调度开销**: 每次 `gemm_mxfp4_dispatch` 都通过 `try_backends()` 构建 lambda 列表

---

## 2. 优化方案

### 2.1 前向缓存 FP4 权重用于 DGrad 复用

**原理**: 2D (32x32) block scaling 具有转置不变性 —— 转置量化后的 FP4 数据和 2D scale 矩阵后，反量化结果与先反量化再转置完全一致（误差为 0）。这是 NVFP4 论文 §4.3 和 Quartet II 论文的核心设计。

**实现**: 前向中将 `quantize_input()` 产生的 FP4 权重（packed uint8 + E8M0 scales）通过 `save_for_backward` 保存。同时预计算转置形式 `transpose_packed_fp4(w_fp4)` + `w_scale.t()`。反向 DGrad 直接使用缓存的转置权重，无需从 BF16 重新量化。

**FSDP2 安全性**: FP4 张量是 `quantize_input()` 新分配的独立张量（非 BF16 参数的 view），FSDP2 resharding 不会影响它们。FP8 blockwise 路径已使用相同模式成功保存 `weight_desc.data/scale`。

**消除的操作**: 1 次 BF16→FP4 重新量化 + 1 次 packed FP4 转置（从反向移至前向）。

```python
# 前向 (linear.py:1332-1347)
w_fp4_t = transpose_packed_fp4(weight_desc.data)
w_scale_t = weight_desc.scale.t().contiguous()
ctx.save_for_backward(input_desc.data, input_desc.scale, w_fp4_t, w_scale_t)

# 反向 DGrad (linear.py:1748-1751)
input_data, input_scale, weight_data, weight_scale = ctx.saved_tensors
grad_input = gemm_mxfp4_dispatch(g_fp4, weight_data, g_scale, weight_scale)
```

### 2.2 融合 Hadamard+Quant kernel 接入反向 WGrad

**原理**: `_fused_hadamard_quant_mxfp4_kernel` 已在 `feature/mxfp4-kernel-fusion` 分支实现，在寄存器中完成 Hadamard-16 蝶形变换后直接量化为 FP4，无中间 BF16 全局内存写入。但此前未接入反向传播。

**实现**: WGrad 中将分离的 `hadamard_transform()` + `convert_to_mxfp4()` 替换为单个 `hadamard_quant_mxfp4()` 调用。

**消除的操作**: 每层 2 次 kernel launch + 2 次全局内存往返。

```python
# 优化前 (4 次 kernel launch)
grad_t_rht = hadamard_transform(grad_t, sign_m, g=16)      # kernel 1
grad_t_fp4, grad_t_scale = convert_to_mxfp4(grad_t_rht)    # kernel 2
input_t_rht = hadamard_transform(input_t, sign_m, g=16)     # kernel 3
input_t_fp4, input_t_scale = convert_to_mxfp4(input_t_rht)  # kernel 4

# 优化后 (2 次 kernel launch，fusion 省掉中间 BF16 读写)
grad_t_fp4, grad_t_scale = hadamard_quant_mxfp4(grad_t, sign_m)    # kernel 1
input_t_fp4, input_t_scale = hadamard_quant_mxfp4(input_t, sign_m)  # kernel 2
```

### 2.3 MXFP4 GEMM 快速调度路径

**原理**: `gemm_mxfp4_dispatch` 每次调用都通过 `try_backends()` 构建 lambda 列表。MXFP4 训练每层 3 次 GEMM，8B 模型 36 层 × 每步 3 次 = 108 次调度。

**实现**: 首次 probe 后缓存结果，后续调用直接走 `_gemm_mxfp4_aiter`，跳过 `try_backends` 开销。

```python
_fast_mxfp4_gemm_fn = None
_fast_mxfp4_gemm_probed = False

def gemm_mxfp4_dispatch(a_fp4, w_fp4, scale_a, scale_w):
    global _fast_mxfp4_gemm_fn, _fast_mxfp4_gemm_probed
    if _FAST_QUANT_DISPATCH:
        if not _fast_mxfp4_gemm_probed:
            _fast_mxfp4_gemm_probed = True
            if _probe_aiter_triton_gemm_mxfp4():
                _fast_mxfp4_gemm_fn = _gemm_mxfp4_aiter
        if _fast_mxfp4_gemm_fn is not None:
            return _fast_mxfp4_gemm_fn(a_fp4, w_fp4, scale_a, scale_w)
    # fallback...
```

### 2.4 AITER 导入修复

AITER 重构了内部 Triton kernel 模块路径（`_triton_kernels.quant.quant_fp8_blockwise` 和 `_triton_kernels.attention.fp8_attention_kernel` 不再存在），导致 Lumen 所有模块无法导入。将这些顶层 import 改为 `try/except` 保护，使 BF16 和 MXFP4 路径不依赖 FP8 blockwise kernel。

---

## 3. 优化效果

### 3.1 操作统计对比

| 阶段 | FP4 量化 | 反量化 | 融合 H+Q | 转置 | FP4 GEMM |
|------|----------|--------|----------|------|----------|
| 前向 | 2 (X+W) | 0 | 0 | 1 (W 预转置) | 1 |
| DGrad | 1 (dY SR) | 0 | 0 | 0 (缓存) | 1 |
| WGrad | 0 | 1 (X) | 2 (dY^T + X^T) | 2 (dY, X) | 1 |
| **合计** | **3** | **1** | **2** | **3** | **3** |

**节省**: 量化操作 6→3（-50%），分离 Hadamard kernel 2→0（融合为 H+Q），kernel launch 总计 ~13→~10。

### 3.2 Kernel Launch 统计

| 路径 | 优化前 | 优化后 | 减少 |
|------|--------|--------|------|
| 前向 | 3 | 4（+1 预转置） | — |
| DGrad | 4（重量化+转置+量化dY+GEMM） | 2（量化dY+GEMM） | **-2** |
| WGrad | 6（反量化+H×2+Q×2+GEMM） | 4（反量化+HQ×2+GEMM） | **-2** |
| **总计** | **13** | **10** | **-3** |

---

## 4. 正确性验证

### 4.1 单元测试 (5/5 通过)

| 测试 | 结果 | 关键指标 |
|------|------|----------|
| 2D block scale 转置不变性 | **通过** | 误差 = 0（bit-identical） |
| 融合 H+Q 等价性 | **通过** | SNR = 29.4 dB |
| 权重缓存 GEMM 正确性 | **通过** | 误差 = 0（RTN 确定性量化） |
| 前向+反向 vs BF16 | **通过** | 前向 SNR=15.3dB, WGrad SNR=14.2dB |
| 训练收敛测试 | **通过** | MXFP4 = 1.12x BF16 loss（50 步） |

### 4.2 端到端训练对比 (Qwen3-8B, C4, 2000 步)

**配置**: lr=1e-4, warmup=50, grad_clip=1.0, FSDP2, 8×MI350X, seed=1234, seq_len=512

| 步数 | BF16 val_loss | MXFP4 val_loss | 差值 | 比率 |
|------|---------------|----------------|------|------|
| 250 | 7.069 | 7.063 | -0.006 | 0.999 |
| 500 | 6.699 | 6.712 | +0.013 | 1.002 |
| 1000 | 6.335 | 6.345 | +0.010 | 1.002 |
| 1500 | 6.199 | 6.226 | +0.027 | 1.004 |
| 2000 | **6.188** | **6.223** | **+0.035** | **1.006** |

**最终 val_loss 差距: +0.035（相对 0.6%）**。全程无 NaN，无发散。曲线高度吻合。

### 4.3 性能

| 指标 | BF16 | MXFP4 | 比率 |
|------|------|-------|------|
| 中位步时 | 292 ms | 573 ms | 1.96x 慢 |
| 峰值内存 | 15.3 GB | 15.3 GB | 相同 |

MXFP4 仍慢于 BF16 约 2x，主要瓶颈是量化和 GEMM 仍为分离 kernel（未做 GEMM prologue fusion）。这是 AITER 层面的工作。

---

## 5. 代码变更清单

| 文件 | 变更 | 说明 |
|------|------|------|
| `lumen/ops/quantize/linear.py` | +176/−115 | 核心优化：权重缓存、融合 H+Q 接入、快速调度 |
| `lumen/ops/quantize/ops.py` | +19/−10 | AITER blockwise kernel 导入保护 |
| `lumen/kernels/attention/attention_impl.py` | +29/−23 | AITER attention kernel 导入保护 |
| `docs/mxfp4_training_report.md` | +84/−63 | 更新操作统计、数据流、性能分析 |
| `tests/test_mxfp4_backward_optimization.py` | +582 (新建) | 5 个独立测试验证优化正确性 |

---

## 6. 论文对齐分析

| 论文技术 | Lumen 状态 | 说明 |
|----------|------------|------|
| 2D block scaling 转置不变性 (NVFP4 §4.3) | **已实现** | 32x32 E8M0 scales，forward 缓存复用 |
| 确定性 Hadamard H16 (arXiv:2605.09825) | **已实现** | 全+1 sign vector，G=16 |
| 融合 Hadamard+Quant kernel | **已实现** | 寄存器内蝶形+量化，零内存流量 |
| 末尾 ~15% 层保持 BF16 (NVFP4 §4) | **已实现** | 8B: 最后 5/36 层 BF16 |
| SR 用于梯度 / RTN 用于权重和激活 | **已实现** | NVFP4 §4.4 |
| GEMM prologue fusion (H+Q+GEMM 一体) | **未实现** | 需要 AITER 支持 |
| FP4 权重存储 + FSDP all-gather | **未实现** | 需要 PyTorch + AITER 支持 |

---

## 7. 后续工作

1. **AITER GEMM prologue fusion** — 将 Hadamard+量化融入 GEMM tile load，消除所有中间内存流量。这是达到论文级性能（比 FP8 快 9-10%）的关键。
2. **重新跑 benchmark** — 在优化后的代码上对比 0.6B 和 8B 的步时变化。
3. **FP4 梯度通信** — 用 MXFP4 量化 allreduce 梯度，减少多节点通信带宽。
4. **Megatron TP/PP 支持** — 将 MXFP4 接入 Megatron 的张量/流水线并行。
