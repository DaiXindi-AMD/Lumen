# MXFP4 Training Status Report

**Author**: Dai, Xindi
**Date**: 2026-07-22
**Hardware**: 8× AMD Instinct MI350X (gfx950, 256GB HBM each)
**Branch**: `feature/mxfp4`

---

## 1. Executive Summary

MXFP4 (Microscaling FP4 E2M1) 训练链路已在 Lumen 上实现并完成 8B 规模验证。

- **0.6B**: BF16 vs MXFP4 loss 曲线几乎重合（Δ val_loss = +0.045, 0.7%），验证了链路正确性。
- **8B**: 经历 3 轮 bug 修复（dispatch 路由、FSDP2 stale tensor、FP4 dynamic range overflow），最终通过 **确定性 Hadamard H16 + 末尾 5/36 层保留 BF16** 稳定收敛，5000 步无发散（val_loss 7.07→5.74）。
- **性能**: MXFP4 比 BF16 慢 ~1.9×（8B: 630ms vs 329ms），原因是量化和 GEMM 未做 kernel fusion。性能收益需要 AITER GEMM prologue fusion 支持。

---

## 2. 实验结果

### 2.1 Qwen3-0.6B: MXFP4 vs BF16（C4, 10k steps）

**共同配置**: 随机初始化, C4 streaming, FSDP2 full_shard 8×MI350X, seq_len=512, lr=6e-5, cosine decay, warmup=200, grad_clip=1.0, seed=1234。BF16 和 MXFP4 除精度外完全相同。

**收敛对比**:

| Step   | BF16 val_loss | MXFP4 val_loss | Δ (MXFP4 − BF16) |
|-------:|--------------:|---------------:|------------------:|
|    500 | 7.163         | 7.187          | +0.024            |
|  1,000 | 6.786         | 6.811          | +0.025            |
|  2,000 | 6.540         | 6.570          | +0.030            |
|  5,000 | 6.334         | 6.364          | +0.030            |
| 10,000 | 6.299         | 6.344          | **+0.045**        |

两条 val_loss 曲线几乎完全重合，尖峰和谷的细节趋势一致。最终差距 +0.045（0.7%），在 FP4 量化噪声的预期范围内。无 NaN、无 Inf、无发散。不需要调整任何超参（BF16 和 MXFP4 用完全相同的 lr/warmup/clip）。

**耗时**:

| 指标              | BF16     | MXFP4    | 比值           |
|-------------------|----------|----------|----------------|
| Median step time  | 229 ms   | 478 ms   | 2.09× 慢       |
| 显存 (per GPU)    | 3.1 GB   | 3.1 GB   | 无差异          |

### 2.2 Qwen3-8B: BF16 Baseline（C4, 5k steps）

**配置**: 随机初始化, C4, FSDP2 8×MI350X, lr=6e-5, 同上。

| Step  | BF16 val_loss |
|------:|--------------:|
|   500 | 6.929         |
| 1,500 | 6.380         |
| 2,500 | 6.158         |
| 3,500 | 6.062         |
| 5,000 | 6.048         |

全程稳定收敛。Median step time 329 ms，显存 15.3 GB/GPU。

### 2.3 Qwen3-8B: MXFP4（C4, 5k steps — 稳定收敛）

**配置**: lr=1e-4, warmup=50, cosine decay, grad_clip=1.0, FSDP2 8×MI350X, `--aiter-attn --lumen-norm --fuse-rope`。确定性 Hadamard H16 + 末尾 5/36 层 BF16（当前默认配置）。

**注**: BF16 baseline 用 lr=6e-5（§2.2），MXFP4 用 lr=1e-4（8B MXFP4 需要更高 lr 突破初始化 plateau）。因此 val_loss 不做精度差距对比，仅验证收敛稳定性。

| Step  | BF16 val_loss (lr=6e-5) | MXFP4 val_loss (lr=1e-4) |
|------:|------------------------:|-------------------------:|
|   500 | 6.929                   | 6.715                    |
| 1,000 | 6.563                   | 6.366                    |
| 1,500 | 6.380                   | 6.153                    |
| 2,000 | 6.252                   | 5.999                    |
| 2,500 | 6.158                   | 5.870                    |
| 3,000 | 6.091                   | 5.790                    |
| 3,500 | 6.062                   | 5.748                    |
| 5,000 | 6.048                   | **5.743**                |

5000 步全程稳定，无 NaN、无 loss spike、无发散。此前全量化（无末尾 BF16）的运行在 step ~1275-3600 崩溃（见 `docs/mxfp4_debug_flow.md`）。

**耗时**:

| 指标              | BF16     | MXFP4    | 比值           |
|-------------------|----------|----------|----------------|
| Median step time  | 329 ms   | 630 ms   | 1.91× 慢       |
| 显存 (per GPU)    | 15.3 GB  | 15.3 GB  | 无差异          |

---

## 3. 链路设计细节

### 3.1 整体架构

按 NVFP4 (NVIDIA, 2025) 和 arXiv:2605.09825 (AMD/PSU, 2025) 的方案，每个线性层的 forward 和 backward 全部使用 MXFP4 GEMM。

```
Megatron / HF Model
  └─ Linear Layer (patched by Lumen)
       └─ QuantizedLinearFunction (autograd.Function)
            ├─ Fprop:  Q_fp4(X) @ Q_fp4(W)^T     → Y (BF16)
            ├─ DGrad:  Q_fp4(dY) @ Q_fp4(W_ref)^T → dX (BF16)
            └─ WGrad:  Q_fp4(H·dY^T) @ Q_fp4(H·X^T)^T → dW (BF16)
```

### 3.2 Forward (Fprop): Y = Q(X) @ Q(W)^T

| 操作数        | 量化方式          | 舍入 | Block Layout | Scales             |
|--------------|-----------------|------|-------------|-------------------|
| Activation X | MXFP4 1×32      | RTN  | 1D per-group | E8M0 (M, K/32)    |
| Weight W     | MXFP4 32×32     | RTN  | 2D tile      | E8M0 (N/32, K/32) |
| Output Y     | BF16            | —    | —           | —                 |

- Weight 用 2D (32×32) block scaling（transpose-friendly，backward 兼容）
- Activation 用 1D (1×32) per-group scaling 沿 K 维（与论文附录 D 结论一致）
- GEMM kernel: AITER `gemm_afp4wfp4`（TN layout, packed uint8 FP4, E8M0 scales）
- 量化用 gfx950 ASM 指令：`v_cvt_scalef32_pk_fp4_f32`（RTN）
- Forward **只存 activation FP4** 到 `save_for_backward`，不存 weight FP4（FSDP2 兼容，见 §4.1）

### 3.3 Backward DGrad: dX = Q(dY) @ Q(W_ref)^T

| 操作数        | 量化方式          | 舍入 | 来源                           |
|--------------|-----------------|------|-------------------------------|
| Gradient dY  | MXFP4 1×32      | SR   | 当前 step 的 grad_output        |
| Weight W     | MXFP4 32×32     | RTN  | 从 `ctx.weight_ref` (BF16) 重新量化 |
| Output dX    | BF16            | —    | —                             |

- **Weight 从 BF16 master weight 重新量化**，不复用 forward 存的 FP4。原因：FSDP2 在 forward 后 reshard weight，saved FP4 tensor 引用的是 resharded（不完整）数据。`ctx.weight_ref` 在 backward 时由 FSDP2 自动 all-gather，数据完整。
- 重新量化后做 `transpose_packed_fp4`: (N, K/2) → (K, N/2)
- 论文（arXiv:2605.09825 Figure 3）也是对 DGrad 的 weight 做重新量化（W^T×H），不是复用 forward 的

### 3.4 Backward WGrad: dW = Q(H·dY^T) @ Q(H·X^T)^T

| 操作数         | 量化方式          | 舍入 | 处理流程                                |
|---------------|-----------------|------|----------------------------------------|
| Gradient dY^T | MXFP4 1×32      | SR   | transpose → Hadamard → SR quantize     |
| Activation X^T| MXFP4 1×32      | SR   | dequant FP4 → BF16 → transpose → Hadamard → SR quantize |
| Output dW     | BF16            | —    | —                                      |

WGrad 的处理流程：

1. **Dequant saved activation**: `convert_from_mxfp4(X_fp4, X_scale) → X_bf16`
2. **Transpose**: dY^T (N, M) 和 X^T (K, M)
3. **确定性 Hadamard 旋转**: 沿 reduction dim M 做 blockwise Hadamard (G=16, sign=全+1)。两个操作数用同一个 H，在 GEMM 内消掉：(dY^T H)(X^T H)^T = dY^T HH^T X = dY^T X。确定性 sign（不用随机 ±1）是按 arXiv:2605.09825 的结论 — 随机 sign 在 Wgrad 全量化时导致发散。G=16（而非 32）按两篇论文推荐，kernel 快 8% 且同等稳定。
4. **SR 量化**: 1×32 沿 axis=-1
5. **GEMM**: `gemm_afp4wfp4(dY^T_fp4, X^T_fp4) → dW`

### 3.5 每层操作数统计

| Phase  | FP4 Quant | Dequant | Hadamard | Transpose | FP4 GEMM |
|--------|-----------|---------|----------|-----------|----------|
| Fprop  | 2 (X + W) | 0       | 0        | 0         | 1        |
| DGrad  | 2 (dY + W 重量化) | 0 | 0     | 1 (W packed) | 1     |
| WGrad  | 2 (dY^T + X^T) | 1 (X) | 2 (dY^T + X^T) | 2 (dY, X) | 1 |
| **总计** | **6**    | **1**   | **2**    | **3**     | **3**    |

BF16 对比：0 quant, 0 dequant, 0 Hadamard, 0 transpose, 3 BF16 GEMMs。

### 3.6 混合精度：末尾层保留 BF16

按 NVFP4 (NVIDIA, arXiv:2509.25149) §4 和 arXiv:2605.09825 的结论——**末尾线性层对 FP4 量化最敏感**，两篇都建议保留 ~15% 层（以末尾为主）在高精度。该机制已接入训练脚本：

```218:221:examples/qwen3/pretrain_qwen3_mxfp4.py
    # MXFP4: keep last ~15% layers in BF16 (NVFP4 paper §4:末尾层最敏感)
    tail_bf16 = args.mode == "mxfp4"
    num_layers = getattr(config, "num_hidden_layers", 0)
    tail_count = max(1, round(num_layers * 0.15)) if tail_bf16 else 0
```

- 底层由 `QuantConfig.first_last_layers_bf16` + `num_layers_at_end_in_bf16` 控制（`lumen/quantize/config.py:211-218`），patch 阶段跳过这些层（保持 BF16 unpatched），同时跳过 `lm_head`。
- mxfp4 模式下**默认开启**：`num_layers_at_start_in_bf16=0`，`num_layers_at_end_in_bf16=round(0.15·num_layers)`（至少 1 层）。
- **注意**：§2.3 的 8B loss spike 实验是在该机制**接入之前**跑的（末尾层也走了 MXFP4），因此不能代表当前默认配置的稳定性，需重跑验证（见 Issue #1）。

### 3.7 硬件加速确认

量化使用 MI350X (gfx950) 原生 FP4 ASM 指令：

| 指令                                 | 操作                           |
|--------------------------------------|-------------------------------|
| `v_cvt_scalef32_pk_fp4_f32`         | 2×FP32 → packed FP4 (RTN)     |
| `v_cvt_scalef32_sr_pk_fp4_f32`      | 2×FP32 → packed FP4 (SR)      |
| `v_cvt_scalef32_pk_fp4_bf16`        | 2×BF16 → packed FP4 (RTN)     |
| `v_cvt_scalef32_sr_pk_fp4_bf16`     | 2×BF16 → packed FP4 (SR)      |

已验证 `is_cdna4() = True`，ASM 路径活跃，不走软件 fallback。

GEMM 使用 AITER `gemm_afp4wfp4`（native FP4 Triton kernel），已验证在 MI350X 上成功执行，不 fallback 到 dequant+BF16。

---

## 4. 调试过程和发现

### 4.1 Bug #1: GEMM Dispatch 路由错误（已修复）

**现象**: MXFP4 GEMM 报 `AssertionError: GROUP_K must equal BLOCK_SIZE_K`。

**原因**: `lumen/quantize/__init__.py:290` 用 `config.scaling.value` 确定 `scaling_type`。MXFP4 的 `config.scaling = ScalingType.BLOCKWISE`，所以 `scaling.value = "blockwise"`，导致 MXFP4 GEMM 被路由到 FP8 `gemm_a8w8_blockscale`（GROUP_K=128），与 MXFP4 block_size=32 不兼容。

**修复**: 改用 `config.recipe`（对 MXFP4 返回 `"mxfp4"`），正确路由到 `gemm_mxfp4_dispatch`。

**Commit**: `656922c`

### 4.2 Bug #2: FSDP2 save_for_backward NaN（已修复）

**现象**: 8B MXFP4 + FSDP2 多卡训练，step 1 即产生 397/399 NaN 梯度（无 gradient checkpointing 时）。

**排查过程**:
1. 单层梯度对比（4096×4096）：MXFP4 vs BF16 梯度 norm 一致，0 NaN → 排除 MXFP4 backward 算法错误
2. 单卡无 FSDP：0 NaN → 排除模型规模问题
3. 单卡 FSDP2 (world_size=1)：0 NaN → 排除 FSDP2 本身
4. 2 卡 FSDP2：**397 NaN** → 定位到多卡 FSDP2 resharding

**Root cause**: FSDP2 `full_shard` 在 forward 后 reshard weight 参数（释放非本 rank 的 shard）。MXFP4 forward 通过 `ctx.save_for_backward()` 保存了 packed FP4 weight，但这个 tensor 引用的底层存储在 reshard 后已被释放或覆盖。Backward 时对 stale 数据做 `transpose_packed_fp4` 和 `gemm_afp4wfp4`，产生 NaN。

**为什么 gradient checkpointing 掩盖了问题**: gradient checkpointing 不用 `save_for_backward` 的 activation/weight，而是在 backward 时重跑 forward（触发 FSDP2 all-gather），部分避免了 stale tensor 问题。但累积误差最终还是导致 step ~1250 崩溃。

**证据表**:

| 配置                                | Step 1 NaN grads |
|------------------------------------|------------------|
| 1 GPU, 无 FSDP                     | 0 / 399          |
| 1 GPU, FSDP2 (world_size=1)        | 0 / 399          |
| 2 GPU, FSDP2, 无 gradient ckpt     | **397 / 399**    |
| 8 GPU, FSDP2, 无 gradient ckpt     | **397 / 399**    |
| 8 GPU, FSDP2, 有 gradient ckpt     | 0 → spike at ~1250 |

**修复**: backward 不用 saved FP4 weight，改为从 `ctx.weight_ref`（FSDP2 管理的 BF16 master weight，backward 时自动 all-gather）重新做 MXFP4 量化。只在 `save_for_backward` 中保存 activation（不是模型参数，不受 FSDP resharding 影响）。

这与 FP8 blockwise 路径的做法一致（`linear.py:1527-1530` 用 `ctx.weight_ref` 做 DGrad），也与论文 Figure 3 一致（论文也是对每个 GEMM pass 重新量化 weight）。

### 4.3 Hadamard / SR 消融实验（信息性）

按 arXiv:2605.09825 做了消融。注意：这些实验是在 FSDP2 fix 之前做的，所有变体都崩了，因为真正的 root cause 是 FSDP2 stale tensor，不是 Hadamard/SR 配置。

| 变体                    | Wgrad Hadamard    | Wgrad 舍入 | Crash step |
|------------------------|-------------------|-----------|------------|
| 原始实现                | 随机 sign ±1       | SR        | ~1550      |
| 确定性 sign             | 全 +1              | SR        | ~1275      |
| 确定性 + RTN            | 全 +1              | RTN       | ~425       |
| 全链路 Fprop+Dgrad+Wgrad| 全 +1              | SR        | step 1     |

FSDP2 fix 后，所有全量化变体仍在 step ~1275-3600 崩溃（FSDP2 fix 是必要条件但不充分）。

### 4.4 8B Loss Spike（已解决 ✅）

**最终修复**: 确定性 Hadamard **H16**（G=32→16） + **末尾 5/36 层保留 BF16**（§3.6）。

| 配置                           | 结果                          |
|-------------------------------|-------------------------------|
| 全量化, H32, 随机 sign          | step ~1550 crash              |
| 全量化, H32, 确定性 sign        | step ~1275 crash              |
| **H16 + 末尾 5 层 BF16**       | **5000 步稳定, val_loss 5.74** |

两篇论文均指出末尾层对 FP4 最敏感，保留 ~15% 在 BF16 是头号稳定性建议。H16 比 H32 快 8% 且同等稳定。详细排查流程见 `docs/mxfp4_debug_flow.md`。

---

## 5. 性能分析

### 5.1 为什么没有速度收益

MXFP4 设计上的加速来自 FP4 的 2× 理论算力。但我们当前的实现中，量化开销远大于 GEMM 计算本身。

**根因：unfused kernel pipeline。** 每次 MXFP4 GEMM 需要 3 次独立 kernel launch，每次都有 global memory 读写：

```
当前 Lumen pipeline（每个 GEMM, 每个操作数）:
  kernel 1: hadamard_transform     BF16 读 → BF16 写 (global memory)
  kernel 2: convert_to_mxfp4       BF16 读 → FP4+scale 写 (global memory)
  kernel 3: gemm_afp4wfp4          FP4 读 → BF16 写 (global memory)
```

论文的 ROCm Transformer Engine 把三者融合在一个 kernel 里：

```
ROCm TE fused pipeline（单个 kernel）:
  1. 从 global memory 加载 BF16 tile → 寄存器
  2. 寄存器内做 Hadamard butterfly (O(G log G), 零 memory traffic)
  3. 寄存器内做 FP4 quant (scale + convert)
  4. FP4 数据 → shared memory → Matrix Core
  5. 写回 BF16 结果
```

0.6B 实测微基准（M=K=N=1024）:

| 操作                  | 时间 (μs)  | 说明                          |
|----------------------|-----------|-------------------------------|
| BF16 GEMM            | 16        | Baseline                      |
| FP4 量化 (2个操作数)    | 153       | gfx950 ASM path               |
| FP4 GEMM             | 57        | `gemm_afp4wfp4`               |
| Hadamard 变换         | 49        | 每操作数, 仅 WGrad             |
| Packed FP4 转置       | 70        | DGrad weight                  |

量化开销（153μs）是 BF16 GEMM 本身（16μs）的 **9.6×**。

### 5.2 为什么没有显存收益

1. **BF16 master weight 必须保留** — FSDP2 all-gather 的是 BF16 weight。backward 也从 BF16 重新量化。FP4 只是 transient 中间结果。
2. **Optimizer state 是 BF16/FP32** — AdamW 的 momentum 和 variance 不受 MXFP4 影响。
3. **Activation 存储** — forward 存的 FP4 activation 比 BF16 小 4×，但只占总显存的一小部分。

### 5.3 到性能收益的路径

| 优化                          | 负责方         | 预期收益                        |
|------------------------------|--------------|-------------------------------|
| Fused Hadamard+Quant kernel  | Lumen        | 减少 ~30-40% 量化开销           |
| GEMM prologue fusion (H+Q+GEMM) | AITER    | 达到论文水平（比 FP8 快 9-10%）    |
| FP4 weight 常驻存储 + FSDP     | AITER + PyTorch | 显存减少                     |
| FP4 梯度通信                   | Lumen        | 减少 allreduce 带宽            |

---

## 6. 算子精度验证

12 个 MXFP4 原语全部通过 torchAO MXTensor 对比验证：

| #  | 操作                         | 对比基准              | 结果              |
|----|-----------------------------|--------------------|-------------------|
| 1  | 1D 量化 (axis=-1, RTN)       | torchAO MXTensor   | bitwise 一致       |
| 2  | 1D 反量化                     | torchAO MXTensor   | bitwise 一致       |
| 3  | 跨框架反量化                   | torchAO to_dtype   | bitwise 一致       |
| 4  | 1D 量化 (axis=0, RTN)        | torchAO MXTensor   | bitwise 一致       |
| 5  | 双轴量化                      | torchAO MXTensor   | bitwise 一致       |
| 6  | Roundtrip (量化→反量化)        | torchAO MXTensor   | bitwise 一致 (SNR 19.0 dB) |
| 7  | GEMM (Y=A@W^T)              | torchAO MXTensor   | bitwise 一致       |
| 8  | 2D Block 量化 Roundtrip       | 手动 LUT 参考        | bitwise 一致       |
| 9  | Packed FP4 转置               | Python 参考         | bitwise 一致       |
| 10 | Hadamard 变换                 | torchAO RHT        | ≈一致 (atol=1e-2)  |
| 11 | 随机舍入无偏性                  | 200 轮统计           | 无偏 (p > 0.05)    |
| 12 | 2D Scale 展开                 | 手动参考             | bitwise 一致       |

---

## 7. 代码改动记录

| Commit    | 内容                                                        |
|-----------|-------------------------------------------------------------|
| `4e6c828` | feat: MXFP4 量化 ops 初始实现                                |
| `ea4d4ee` | fix: MXFP4 kernel bugs, 10/10 tests pass vs torchAO         |
| `9ed2070` | fix: SR ASM kernel shape mismatch, 12/12 tests pass         |
| `99b8249` | docs: MXFP4 accuracy report script                          |
| `656922c` | fix: `config.recipe` dispatch routing（修复 GEMM 路由错误）    |
| `a25ccf4` | feat: SFT 脚本加 `--mode mxfp4` + TensorBoard               |
| `66ab5de` | docs: 训练报告 + 预训练脚本 C4 streaming                      |
| `0edf68a` | chore: `--model` 参数 + GPU 显存 TensorBoard                 |
| `7674afd` | debug: FSDP2 NaN 诊断 + 确定性 Hadamard + 消融实验            |
| `38d4ae0` | docs: 报告重写                                               |
| `8bcf2d9` | fix: FSDP2 backward 改用 `ctx.weight_ref` 重量化 weight       |
| `a72e8c6` | debug: 8B late loss spike 诊断 — FP4 dynamic range overflow   |
| `745de8f` | fix: 确定性 H16 + 末尾 5 层 BF16，8B 5000 步稳定收敛          |

---

## 8. Open Issues

### Issue #1: 8B Loss Spike ~~（已解决 ✅）~~

通过确定性 H16 + 末尾 5 层 BF16 解决。5000 步稳定，val_loss 5.74。详见 §4.4。

### Issue #2: 无速度/显存收益

- **现象**: MXFP4 比 BF16 慢 ~1.9×（8B: 630ms vs 329ms），显存相同
- **原因**: 3 次独立 kernel launch，量化开销远大于 GEMM 计算
- **下一步**: (1) Lumen 做 Hadamard+Quant fused kernel; (2) AITER 做 GEMM prologue fusion

### Issue #3: BF16 vs MXFP4 未在同一 lr 下对比

- 8B BF16 用 lr=6e-5，MXFP4 用 lr=1e-4。公平的精度对比需要在同一 lr 下跑两者。

---

## 9. 下一步建议

1. **公平对比**: 重跑 BF16 8B 用 lr=1e-4（与 MXFP4 相同），做 head-to-head val_loss 对比
2. **Lumen fused Hadamard+Quant kernel** — 合并两个 Triton kernel，减少一次 global memory roundtrip
3. **给 AITER 提 GEMM prologue fusion 需求** — 这是达到论文性能水平的关键依赖
4. **梯度量化** — `quantize_grad="mxfp4"` 减少多节点 allreduce 带宽
5. **Megatron 路径** — 接入 TP/PP 支持更大规模训练

> 相关文档：`docs/mxfp4_debug_flow.md`（完整调试流程）、`docs/papers/nvfp4_paper_vs_lumen_comparison.md`（NVFP4 超参核对）、`docs/papers/mxfp4_paper_vs_lumen_comparison.md`（AMD MXFP4 对比）。
