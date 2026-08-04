# MXFP4 Training Status Report

**Author**: Dai, Xindi
**Date**: 2026-07-22（§2、§4 的实验结果）；2026-07-30（§3、§5、§7–§9 已按当前代码更新）
**Hardware**: 8× AMD Instinct MI350X (gfx950, 256GB HBM each)
**Branch**: `feature/mxfp4`（当前 HEAD `035431e`）

---

## 1. Executive Summary

MXFP4 (Microscaling FP4 E2M1) 训练链路已在 Lumen 上实现并完成 8B 规模验证。

- **0.6B**: BF16 vs MXFP4 loss 曲线几乎重合（Δ val_loss = +0.045, 0.7%），验证了链路正确性。
- **8B**: 经历 3 轮 bug 修复（dispatch 路由、FSDP2 stale tensor、FP4 dynamic range overflow），最终通过 **确定性 Hadamard H16 + 末尾 5/36 层保留 BF16** 稳定收敛，5000 步无发散（val_loss 7.07→5.74）。
- **性能（本文初版，07-22）**: MXFP4 比 BF16 慢 ~1.9×（8B: 630ms vs 329ms），量化和 GEMM 未做 kernel fusion。
- **性能（当前 HEAD，07-30）**: **MXFP4 比 BF16 快 6.7%**（8B seq2048: 869.4 ms vs 928.0 ms），代价是显存
  从 15.30 GB 涨到 20.90 GB。经过两轮共 8 项优化，逐项实测数据见
  [`mxfp4_optimization_report.md`](mxfp4_optimization_report.md)，汇总见 §5。

> **本文定位**：§2 和 §4 是 07-22 那一版代码的实验记录，保留原样作为历史；§3、§5、§7–§9
> 已对齐当前 HEAD。设计细节的权威版本是
> [`mxfp4_training_report.md`](mxfp4_training_report.md)。

---

## 2. 实验结果（07-22 版本代码）

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

0.6B 此后没有重测（优化工作都对着 8B 做），这个 2.09× 不代表当前状态；当前数字见 §5.1。

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

## 3. 链路设计细节（当前 HEAD `035431e`）

### 3.1 整体架构

按 NVFP4 (NVIDIA, 2025) 和 arXiv:2605.09825 (AMD/PSU, 2025) 的方案，每个线性层的 forward 和 backward 全部使用 MXFP4 GEMM。

```
Megatron / HF Model
  └─ Linear Layer (patched by Lumen)
       └─ QuantizedLinearFunction (autograd.Function)
            ├─ Fprop:  Q_fp4(X) @ Q_fp4(W)^T        → Y  (BF16)
            │            权重量化 + 预转置按 module 缓存，一个 optimizer step 只做一次
            ├─ DGrad:  Q_fp4(dY) @ W^T_cached        → dX (BF16)
            │            复用 forward 存的预转置 FP4 权重，不重新量化
            └─ WGrad:  HQ_sr(dY^T_view) @ HQ_rtn(X^T)^T → dW (BF16)
                         dY^T 是 view；X^T 走 fused dequant+transpose
```

### 3.2 Forward (Fprop): Y = Q(X) @ Q(W)^T

| 操作数        | 量化方式          | 舍入 | Block Layout | Scales             |
|--------------|-----------------|------|-------------|-------------------|
| Activation X | MXFP4 1×32      | RTN  | 1D per-group | E8M0 (M, K/32)    |
| Weight W     | MXFP4 32×32     | RTN  | 2D tile      | E8M0 (N/32, K/32) |
| Output Y     | BF16            | —    | —           | —                 |

- Weight 用 2D (32×32) block scaling（transpose-invariant，backward 可直接复用）
- Activation 用 1D (1×32) per-group scaling 沿 K 维（与论文附录 D 结论一致）
- GEMM kernel: 三个 AITER FP4 kernel 之一，按 (M,N,K) 首次实测选定并缓存（§3.8）
- 量化用 gfx950 ASM 指令：`v_cvt_scalef32_pk_fp4_f32`（RTN）
- Forward 存 activation FP4，**以及权重的 packed FP4 + 预转置形式**，都走 `save_for_backward`。
  这些是量化 kernel 新分配的 tensor，不是 BF16 参数的 view，因此 FSDP2 reshard 影响不到它们（§4.2）。
- 权重量化和预转置按 module 缓存（`module._mxfp4_w_cache`），optimizer post-step hook 失效。
  梯度累积下每个 optimizer step 只做一次，而不是每个 micro-batch 一次。
  `LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` 可关掉。

### 3.3 Backward DGrad: dX = Q(dY) @ W^T_cached

| 操作数        | 量化方式          | 舍入 | 来源                           |
|--------------|-----------------|------|-------------------------------|
| Gradient dY  | MXFP4 1×32      | SR   | 当前 step 的 grad_output        |
| Weight W^T   | MXFP4 32×32     | RTN  | **复用 forward 的预转置 FP4 权重** |
| Output dX    | BF16            | —    | —                             |

- **不再从 BF16 重新量化**。2D (32×32) block scale 转置不变，转置 FP4 数据和 scale 即得到
  数学上等价的 W^T 量化结果（NVFP4 §4.3, Quartet II），所以 forward 量化一次就够。
- `transpose_packed_fp4`（(N, K/2) → (K, N/2)）连同 `scale.t()` 都在 forward 做，
  从 backward 的关键路径上挪走；weight cache 命中时连这一步也省掉。
- 与论文（arXiv:2605.09825 Figure 3）的差异：论文每个 GEMM pass 重新量化 weight。
  复用是合法的，因为 RTN 是确定性的，同一个 BF16 权重每次量化结果 bit 级一致。

### 3.4 Backward WGrad: dW = HQ(dY^T) @ HQ(X^T)^T

| 操作数         | 量化方式          | 舍入 | 处理流程                                |
|---------------|-----------------|------|----------------------------------------|
| Gradient dY^T | MXFP4 1×32      | **SR**   | `.t()` 保持为 view → fused Hadamard+Quant |
| Activation X^T| MXFP4 1×32      | **RTN**  | fused dequant+transpose → fused Hadamard+Quant |
| Output dW     | BF16            | —    | —                                      |

舍入按 NVFP4 §4.4 / 附录 E.3：**只在梯度上用 SR**。activation 上 SR 收益极小且可能发散，
所以 X^T 走 RTN。（`40e2691` 之前两个操作数都是 SR。）

WGrad 的处理流程：

1. **Fused dequant+transpose**: `dequant_transpose_mxfp4(X_fp4, X_scale) → X^T` BF16 `(K, M)`。
   一个 kernel 直接产出转置结果，分开做时那个 BF16 `(M, K)` 中间量根本不落地。
2. **dY^T 保持 view**: `grad_flat.t()` 不 `.contiguous()`。`hadamard_quant_mxfp4` 按两个 stride
   寻址，直接读转置。这两次 `.t().contiguous()` 拷贝原先占一步的 7.8%，比所有 MXFP4 GEMM 加起来还多。
3. **Fused Hadamard+Quant**（每个操作数一个 kernel）：沿 reduction dim M 做 blockwise Hadamard
   (G=16, sign=全+1)，butterfly 全在寄存器里，接着寄存器内量化写出 packed FP4 + scale。
   两个操作数用同一个 H，在 GEMM 内消掉：(dY^T H)(X^T H)^T = dY^T HH^T X = dY^T X。
   确定性 sign（不用随机 ±1）按 arXiv:2605.09825 的结论 —— 随机 sign 在 Wgrad 全量化时导致发散。
   G=16（而非 32）按 arXiv:2605.09825：H16 kernel 快 8% 且同等稳定。（NVFP4 论文对 NVFP4 也用 16×16，
   但对 MXFP4 是用 d=32 去对齐 block 大小的，所以这里跟的是 AMD/PSU 那篇。）
4. **GEMM**: `gemm_mxfp4_dispatch(dY^T_fp4, X^T_fp4) → dW`

`M % 16 != 0` 时跳过 Hadamard，两个操作数走普通 `convert_to_mxfp4`（SR/RTN 分工不变）。
这条分支要求操作数 dense，是 activation 走 fused kernel 而不像 dY^T 那样保持 view 的第二个原因。

### 3.5 每层操作数统计

**当前**：

| Phase  | FP4 Quant | Dequant+Transpose | Fused H+Q | Transpose | FP4 GEMM |
|--------|-----------|---------|----------|-----------|----------|
| Fprop  | 2 (X + W\*) | 0     | 0        | 1 (W 预转置\*) | 1     |
| DGrad  | 1 (dY SR) | 0       | 0        | 0（复用 fprop） | 1     |
| WGrad  | 0         | 1 (X, fused) | 2 (dY^T SR + X^T RTN) | 0（dY^T 是 view） | 1 |
| **总计** | **3**    | **1**   | **2**    | **1**     | **3**    |

\* 梯度累积下这两个权重操作被 module cache 摊薄到每 optimizer step 一次，
命中的 micro-batch 在 fprop 只有 1 次量化（X）、0 次转置。

**07-22 版本**：

| Phase  | FP4 Quant | Dequant | Hadamard | Transpose | FP4 GEMM |
|--------|-----------|---------|----------|-----------|----------|
| Fprop  | 2 (X + W) | 0       | 0        | 0         | 1        |
| DGrad  | 2 (dY + W 重量化) | 0 | 0     | 1 (W packed) | 1     |
| WGrad  | 2 (dY^T + X^T) | 1 (X) | 2 (dY^T + X^T) | 2 (dY, X) | 1 |
| **总计** | **6**    | **1**   | **2**    | **3**     | **3**    |

量化 6→3，独立 Hadamard 2→0（融进 H+Q），转置 3→1（权重转置挪到 fprop 并缓存、
dY^T 变 view、X^T 折进 dequant kernel），独立的 activation dequant 变成 fused dequant+transpose。

BF16 对比：0 quant, 0 dequant, 0 Hadamard, 0 transpose, 3 BF16 GEMMs。

### 3.6 混合精度：末尾层保留 BF16

按 NVFP4 (NVIDIA, arXiv:2509.25149) §4 / 附录 E.2 的结论——**末尾线性层对 FP4 量化最敏感**，该文建议保留 ~15% 层（以末尾为主）在高精度。arXiv:2605.09825 (AMD/PSU) 没有这条：它对所有 transformer linear layer 全量化（基线是 FP8），稳定手段只有确定性 Hadamard。该机制已接入训练脚本：

```265:268:examples/qwen3/pretrain_qwen3_mxfp4.py
    # MXFP4: keep last ~15% layers in BF16 (NVFP4 paper §4:末尾层最敏感)
    tail_bf16 = args.mode == "mxfp4"
    num_layers = getattr(config, "num_hidden_layers", 0)
    tail_count = max(1, round(num_layers * 0.15)) if tail_bf16 else 0
```

- 底层由 `QuantConfig.first_last_layers_bf16` + `num_layers_at_end_in_bf16` 控制（`lumen/quantize/config.py:219-221`），patch 阶段跳过这些层（保持 BF16 unpatched），同时跳过 `lm_head`。
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

GEMM 走 AITER native FP4 kernel，已验证在 MI350X 上成功执行，不 fallback 到 dequant+BF16。

### 3.8 GEMM 后端选择（每形状实测）

同一个 shape 有三条可用路径，首次调用时实测三者、缓存最快的那个
（`lumen/ops/quantize/mxfp4_autotune.py`）：

| 后端 | Kernel | 说明 |
|---|---|---|
| ASM | `gemm_a4w4_asm` | 需要权重 shuffle 成 ASM 布局，大 shape 上最快 |
| shuffled Triton | `gemm_afp4wfp4` + shuffled 权重 | 中间档 |
| plain Triton | `gemm_afp4wfp4` | 无预处理，小 shape 上最快 |

Qwen3-8B 每层 21 次 GEMM 调用、去重后 9 个不同 shape。实测选中 ASM 的是 4 个 shape，
全是 MLP 的（gate/up 的 fprop 与 down 的 dgrad 同形状、gate/up 的 dgrad 与 down 的 fprop
同形状，所以这 4 个 shape 覆盖 21 次调用里的 9 次）；剩下 5 个 attention shape 留在
plain Triton。首次 probe 之后调度层本身只是一次 dict 查表，实测 1.7–3.4 µs/次，
约为一次 0.2 ms GEMM 的 1%。

这项工作在 GEMM 层面把 Qwen3-8B 从 0.90x BF16 拉到 1.12x，端到端只值 +0.4%；
真正的价值是避免了手工阈值方案 −2.6% 的倒退。完整数据见
[`mxfp4_gemm_backend_selection.md`](mxfp4_gemm_backend_selection.md)。

### 3.9 FP4 参数 all-gather

`MXFP4CommTensor`（`lumen/quantize/comm_tensor.py`）包住 BF16 参数，给 FSDP2 提供两个钩子：
all-gather 前把本 rank 的 BF16 shard 量化成 packed MXFP4（2D 32×32 scale），
all-gather 后再 dequant 回 BF16。线上字节数从 2 byte/元素降到 0.5 byte/元素加 E8M0 scale，
实测 **3.99x 更少**。optimizer、forward、梯度看到的都还是正常的 BF16 权重。

需要 `N % (32 × world_size) == 0` 且 `K % 32 == 0`，保证一个 rank 的 dim-0 shard 不会切开
32 行的 tile；不满足的权重（如 vocab embedding）继续走 BF16 all-gather。
Qwen3-8B 8 卡下包住 217 个权重。mxfp4 模式默认开启，`--no-mxfp4-comm` 关掉。

**收益尚未单独验证**：`convert_from_mxfp4_2d` 是纯 PyTorch，会物化好几个全尺寸中间量
（含一个 int64），单节点上可能比省下的带宽更贵。

---

## 4. 调试过程和发现

### 4.1 Bug #1: GEMM Dispatch 路由错误（已修复）

**现象**: MXFP4 GEMM 报 `AssertionError: GROUP_K must equal BLOCK_SIZE_K`。

**原因**: `lumen/quantize/__init__.py:290`（当前为 304 行）用 `config.scaling.value` 确定 `scaling_type`。MXFP4 的 `config.scaling = ScalingType.BLOCKWISE`，所以 `scaling.value = "blockwise"`，导致 MXFP4 GEMM 被路由到 FP8 `gemm_a8w8_blockscale`（GROUP_K=128），与 MXFP4 block_size=32 不兼容。

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

**当时的修复**（`8bcf2d9`）: backward 不用 saved FP4 weight，改为从 `ctx.weight_ref`（FSDP2 管理的 BF16 master weight，backward 时自动 all-gather）重新做 MXFP4 量化。只在 `save_for_backward` 中保存 activation。NaN 消失，代价是 backward 多一次权重量化加一次 packed 转置。

**后续（重要）**: `23644ea` 又把 saved FP4 weight 加回来了（理由：量化 kernel 的输出是新分配的
tensor，不是参数的 view，FSDP2 reshard 动不到它），当前 HEAD 走的就是这条路，Qwen3-8B 8 卡已经
跑过 1250 步、loss 单调。也就是说**上面这个 "stale tensor" 根因判断站不住** —— 如果它成立，
当前代码就该 NaN。NaN 是真的，`8bcf2d9` 也确实治住了，但真实机制是那个 commit 里的别的改动
（它同时把非对齐 BF16 fallback 也改成用 `weight_ref`），一直没有单独定位。
如果哪天在新的并行配置下 FP4 权重复用出问题，从这里查。

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

NVFP4 论文指出末尾层对 FP4 最敏感，保留 ~15% 在 BF16 是其头号稳定性建议（AMD/PSU 那篇不涉及层位置敏感度）。H16 比 H32 快 8% 且同等稳定。详细排查流程见 `docs/mxfp4_debug_flow.md`。

---

## 5. 性能分析

### 5.1 当前性能（HEAD `035431e`）

Qwen3-8B, 8× MI350X, FSDP2 full_shard, C4 streaming, seq 2048 × micro_batch 4 × 8 卡
= 65536 tokens/step, lr 3e-4, warmup 200。BF16 与 MXFP4 只差一个 `--mode`。
step time 取 step ≥ 50 的 `step_time_ms` 中位数（C4 streaming 偶发数秒级卡顿会把均值抬高
约 40%，中位数在多个窗口间稳定在 ±1.5% 内）。

| 版本 | 中位步时 | P25 | 最快 | 峰值显存/卡 | vs BF16 |
|---|---|---|---|---|---|
| BF16 | 928.0 ms | 917.0 | 910.7 | 15.30 GB | 1.00x |
| MXFP4，仅第一轮 GEMM 优化（`7d1841b`） | 1061.8 ms | 1001.1 | 989.0 | 16.10 GB | 0.87x |
| MXFP4，当前（`035431e`） | **869.4 ms** | 865.0 | 860.2 | **20.90 GB** | **1.067x** |

八项优化的逐项实测（含失败的尝试和方法论教训）见
[`mxfp4_optimization_report.md`](mxfp4_optimization_report.md)。摘要：

| 优化 | 端到端实测 |
|---|---|
| 每形状后端选择 + 向量化 weight shuffle | GEMM 层面 0.90x → 1.12x，端到端 +0.4% |
| wgrad 去掉转置物化（`.t()` 保持 view） | **+7.8%** |
| fused dequant+transpose kernel | 与后三项合计 1.221x |
| 跨 micro-batch 权重缓存 | 同上，代价 +4.8 GB |
| FP4 all-gather | 通信字节 3.99x 更少，净收益未单独验证 |
| WGrad activation 改 RTN | 精度修正，非性能 |

### 5.2 剩下的差距：unfused GEMM prologue

MXFP4 设计上的加速来自 FP4 的 2× 理论算力，而量化开销至今仍与 GEMM 同量级。
现在量化和 Hadamard 已经融进一个 kernel，但**量化和 GEMM 仍是两次独立 launch**：

```
当前 Lumen pipeline（每个 GEMM, 每个操作数）:
  kernel 1: hadamard_quant_mxfp4   BF16 读 → FP4+scale 写 (global memory)  ← 已融合
  kernel 2: gemm_afp4wfp4          FP4 读 → BF16 写 (global memory)
```

论文的 ROCm Transformer Engine 把它们融合在一个 kernel 里：

```
ROCm TE fused pipeline（单个 kernel）:
  1. 从 global memory 加载 BF16 tile → 寄存器
  2. 寄存器内做 Hadamard butterfly (O(G log G), 零 memory traffic)
  3. 寄存器内做 FP4 quant (scale + convert)
  4. FP4 数据 → shared memory → Matrix Core
  5. 写回 BF16 结果
```

0.6B 实测微基准（M=K=N=1024，07-22 采集）:

| 操作                  | 时间 (μs)  | 说明                          |
|----------------------|-----------|-------------------------------|
| BF16 GEMM            | 16        | Baseline                      |
| FP4 量化 (2个操作数)    | 153       | gfx950 ASM path               |
| FP4 GEMM             | 57        | `gemm_afp4wfp4`               |
| Hadamard 变换         | 49        | 每操作数, 仅 WGrad（现已融进 H+Q）|
| Packed FP4 转置       | 70        | 现在在 fprop，权重缓存命中时完全跳过 |

量化开销（153μs）是 BF16 GEMM 本身（16μs）的 **9.6×**。Qwen3-8B 的 profile 在更大规模上
给出同样的结论：一步之内，每层 21 个 MXFP4 GEMM 加起来的 GPU 时间还不如 wgrad 里那两次
`.t().contiguous()` 拷贝多。**瓶颈是量化和布局，不是 GEMM。**

### 5.3 显存：不省，反而多

| 来源 | Qwen3-8B 8 卡实测峰值 | 说明 |
|---|---|---|
| BF16 baseline | 15.30 GB | — |
| + FP4 操作数、saved FP4 权重及预转置、all-gather scratch | 16.10 GB (+0.8) | 逐层分配，backward 走过即释放 |
| + 跨 micro-batch 权重缓存（§3.2） | **20.90 GB (+4.8)** | 所有量化层的 FP4 权重整个 step 常驻 |

原因不变：

1. **BF16 master weight 必须保留** — FSDP2 all-gather 的是 BF16 weight，optimizer 也更新它。
2. **Optimizer state 是 BF16/FP32** — AdamW 的 momentum 和 variance 不受 MXFP4 影响。
3. **Activation 存储** — forward 存的 FP4 activation 比 BF16 小 4×，但只占总显存的一小部分。

`LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` 可以用 §5.1 的速度换回那 4.8 GB。
只缓存预转置形式（DGrad 唯一的消费者）大约能省回一半。

### 5.4 到进一步收益的路径

| 优化                          | 负责方         | 状态 / 实测                     |
|------------------------------|--------------|-------------------------------|
| Fused Hadamard+Quant kernel  | Lumen        | **已完成** ✅ −2 kernel、−2 次 memory roundtrip |
| 每形状 GEMM 后端选择            | Lumen        | **已完成** ✅ GEMM 层 0.90x → 1.12x |
| wgrad 去掉转置物化              | Lumen        | **已完成** ✅ **+7.8%**         |
| fused dequant+transpose      | Lumen        | **已完成** ✅ −1 kernel、−1 个全尺寸中间量 |
| 跨 micro-batch 权重缓存         | Lumen        | **已完成** ✅ 代价 +4.8 GB       |
| FP4 参数 all-gather            | Lumen        | **已完成**，收益未验证 ⚠️        |
| GEMM prologue fusion (H+Q+GEMM) | AITER    | 未做 — 达到论文水平（比 FP8 快 9-10%）|
| `convert_from_mxfp4_2d` 改 Triton | Lumen   | 未做 — 让 FP4 all-gather 明确变成收益 |
| FP4 weight 常驻存储 + FSDP     | AITER + PyTorch | 未做 — 显存减少               |
| FP4 梯度通信                   | Lumen        | 未做 — 减少 reduce-scatter 带宽  |

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

07-27 之后的性能优化（本文 §5 覆盖）：

| Commit    | 内容                                                        |
|-----------|-------------------------------------------------------------|
| `23644ea` | perf: 复用 forward 的 FP4 权重、融合 Hadamard+Quant、快速调度   |
| `660705f` | fix: 预转置权重改走 `save_for_backward`，加 5 项 backward 测试   |
| `d545804` | perf: 跨 micro-batch 权重缓存 + optimizer post-step hook       |
| `026e398` | perf: 加环境变量开关 + A/B 实测结果                            |
| `827c941` | perf: fused dequant+transpose kernel、预转置权重缓存            |
| `3975266` | feat: FSDP2 FP4 all-gather（`MXFP4CommTensor`）               |
| `40e2691` | fix: WGrad activation 改 RTN（NVFP4 §4.4：SR 只用于梯度）       |
| `7d1841b` | 第一轮 GEMM 优化的收尾（§5.1 的中间测量点）                     |
| `30f5277` | merge: weight-cache 分支并入 `feature/mxfp4`                  |
| `035431e` | test: backward 对齐 BF16 参考；FP4 all-gather 可关（当前 HEAD）  |

---

## 8. Open Issues

### Issue #1: 8B Loss Spike ~~（已解决 ✅）~~

通过确定性 H16 + 末尾 5 层 BF16 解决。5000 步稳定，val_loss 5.74。详见 §4.4。

### Issue #2: 无速度收益 ~~（已解决 ✅）~~

- **07-22**: MXFP4 比 BF16 慢 ~1.9×（8B seq512: 630ms vs 329ms）
- **当前**: MXFP4 比 BF16 **快 6.7%**（8B seq2048: 869.4ms vs 928.0ms），见 §5.1
- **仍在的差距**: 量化和 GEMM 还是两次独立 launch，需要 AITER 做 GEMM prologue fusion（§5.2）

### Issue #3: 显存反而多了 5.6 GB

- 15.30 → 20.90 GB，其中 4.8 GB 是跨 micro-batch 权重缓存（§5.3）。
- BF16 master weight 仍必须保留，没有任何权重存储收益来抵这部分。
- 缓解手段：只缓存预转置形式（省回约一半），或 `LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` 换回速度。

### Issue #4: FP4 all-gather 收益未验证

- `convert_from_mxfp4_2d` 是纯 PyTorch，物化多个全尺寸中间量（含 int64），
  单节点上可能比省下的 3.99x 带宽更贵。需要在空闲机器上单独 A/B（§3.9）。

### Issue #5: 改动 5–8 未逐项 A/B

- §5.1 的 1.221x 是四项合起来的效果。当时机器上有其他任务，隔离的微基准同一形状能差 2 倍
  且方向不一致，所以只取了端到端中位数。

### Issue #6: FSDP2 NaN 的真实机制未定位

- `8bcf2d9` 治住了 NaN，但当前代码又回到了被它判定为根因的做法且稳定运行（§4.2）。
  说明当年的根因判断不对，真实原因不明。

### Issue #7: BF16 vs MXFP4 未在同一 lr 下对比（§2.3）

- §2.3 的 8B BF16 用 lr=6e-5，MXFP4 用 lr=1e-4。§5.1 的性能对比两者 lr 相同（3e-4），
  但那次运行还在进行中（约 1250/3000 步），收敛数据尚不完整。

---

## 9. 下一步建议

1. **跑完当前 3000 步 8B 运行** — 同 lr 下与 BF16 baseline 做 head-to-head val_loss 对比
2. **单独 A/B FP4 all-gather** — 空闲机器上用 `--no-mxfp4-comm` 验证；若确认带宽瓶颈，
   把 `convert_from_mxfp4_2d` 改写成 Triton kernel
3. **只缓存预转置权重** — 省回约一半的 4.8 GB（§5.3）
4. **给 AITER 提 GEMM prologue fusion 需求** — 这是达到论文性能水平的关键依赖
5. **梯度量化** — `quantize_grad="mxfp4"` 减少多节点 reduce-scatter 带宽
6. **Megatron 路径** — 接入 TP/PP 支持更大规模训练

> 相关文档：[`mxfp4_training_report.md`](mxfp4_training_report.md)（设计细节权威版本）、
> [`mxfp4_optimization_report.md`](mxfp4_optimization_report.md)（八项优化逐项实测）、
> [`mxfp4_gemm_backend_selection.md`](mxfp4_gemm_backend_selection.md)（后端选择深入）、
> `docs/mxfp4_debug_flow.md`（完整调试流程）、
> `docs/papers/nvfp4_paper_vs_lumen_comparison.md`（NVFP4 超参核对）、
> `docs/papers/mxfp4_paper_vs_lumen_comparison.md`（AMD MXFP4 对比）。
