# MXFP4 8B 训练调试全流程

**Author**: Dai, Xindi
**Date**: 2026-07-22
**Branch**: `feature/mxfp4`

本文记录 MXFP4 从 0.6B 验证到 8B 收敛的完整 debug 过程，包括遇到的三个 bug、排查逻辑和最终修复。

---

## 阶段一：0.6B 链路验证（成功）

**目标**: 验证 MXFP4 端到端链路是否正确。

**过程**:
1. 发现 AITER `gemm_afp4wfp4` import 路径是 `aiter.ops.triton.gemm.basic.gemm_afp4wfp4`（非顶层 re-export），probe 成功
2. 写了 `pretrain_qwen3_mxfp4.py` 预训练脚本（Qwen3-0.6B 随机初始化 + wikitext-2）
3. 首次跑报 `ModuleNotFoundError: No module named 'aiter.ops.triton._triton_kernels.quant.quant_fp8_blockwise'` — 系统装的 AITER 版本不含该文件，需要用 `PYTHONPATH` 加 Lumen 自带的 `third_party/aiter`

**Bug #1: GEMM dispatch 路由错误**
- **现象**: `AssertionError: GROUP_K must equal BLOCK_SIZE_K`
- **排查**: `dispatch_gemm` 检查 `scaling_type`，发现传进来是 `"blockwise"` 而不是 `"mxfp4"`
- **根因**: `lumen/quantize/__init__.py:290` 用 `config.scaling.value`，MXFP4 的 scaling 是 `ScalingType.BLOCKWISE`，`.value = "blockwise"`，被路由到 FP8 `gemm_a8w8_blockscale`
- **修复**: 改用 `config.recipe`（MXFP4 返回 `"mxfp4"`）
- **验证**: 0.6B wikitext-2 跑 50 步，loss 12.1→7.4，正常

**0.6B BF16 vs MXFP4 对比（C4, 10k 步）**:
- val_loss 差距 +0.045（0.7%），两条曲线几乎重合
- MXFP4 median step time 478ms vs BF16 229ms（2.09× 慢）
- 确认了 gfx950 ASM 路径 (`is_cdna4()=True`) 和 AITER native FP4 GEMM 都在工作

---

## 阶段二：8B 首次尝试（失败 — 不收敛）

**现象**: 8B MXFP4 + FSDP2 8 卡，任何 lr 都不收敛。loss 卡在 ~12.75（随机初始化水平）或跳到 11.9375（= ln(151936)，模型退化为均匀分布）。

**排查步骤**:
1. **试降 lr**: 6e-5 → 2e-5 → 3e-6，全部失败 → 排除 "lr 太高"
2. **单层梯度对比**: 在 4096×4096 矩阵上，MXFP4 梯度 norm 和 BF16 一致，zero%=0，无 NaN → **排除 MXFP4 backward 算法错误**
3. **单卡无 FSDP**: lr=1e-4，loss 12.76→7.33（50 步） → **排除模型规模问题**
4. **多卡 FSDP2 + lr=1e-4**: loss 正常下降 → 定位到 **lr=6e-5 对 8B MXFP4 太低**

**结论**: 初步判断为超参敏感性 — MXFP4 量化噪声需要更高 lr 突破初始化 plateau。用 lr=1e-4 + warmup=50 能跑。

---

## 阶段三：8B late loss spike（Bug #2 — FSDP2 交互）

**现象**: 用 lr=1e-4 跑 5000 步，前 1500 步正常（val_loss 7.07→6.15），step 1550 突然崩到 11.9375。

**排查步骤**:
1. 试确定性 Hadamard（按 arXiv:2605.09825 建议）→ **同样在 step ~1275 崩**
2. 对比 BF16 baseline 和 BF16+Lumen attn/norm/rope → **都正常** → 问题隔离到 MXFP4 线性层
3. 写 `diag_crash_v2.py` 逐 step 监控 per-layer NaN/grad norm：
   - Step 1402: loss 看似正常 (7.19)，但 **325 个参数有 NaN 梯度**，集中在 layer 0
   - Step 1403: loss=11.94，grad norm 26.7M，权重变 NaN → 不可恢复

**Bug #2: FSDP2 `save_for_backward` stale tensor**
- **根因**: FSDP2 `full_shard` 在 forward 后 reshard weight（释放非本 rank 的 shard）。MXFP4 forward 通过 `save_for_backward()` 保存了 packed FP4 weight tensor，但底层存储已被 FSDP2 释放/覆盖。Backward 时对 stale 数据做转置和 GEMM，产生 NaN。
- **证据**:

  | 配置 | Step 1 NaN grads |
  |------|-----------------|
  | 1 GPU, 无 FSDP | 0 / 399 |
  | 1 GPU, FSDP2 (world_size=1) | 0 / 399 |
  | 2 GPU, FSDP2, 无 grad ckpt | **397 / 399** |
  | 8 GPU, FSDP2, 有 grad ckpt | 0 初始 → step ~1250 崩 |

  Gradient checkpointing 掩盖了问题——它不用 `save_for_backward` 的 weight，而是 backward 时重跑 forward（触发 FSDP2 all-gather），部分绕过了 stale tensor。但累积误差最终仍导致崩溃。

- **修复**: backward 不复用 saved FP4 weight，改为从 `ctx.weight_ref`（BF16 master weight，FSDP2 在 backward 时自动 all-gather）重新量化。只在 `save_for_backward` 中保存 activation。

---

## 阶段四：FSDP2 fix 后的 late spike（Bug #3 — FP4 dynamic range overflow）

**现象**: FSDP2 fix 后 8B 能训到 step 3500（val_loss 6.14），但 step ~3600 仍然 loss spike。

**排查**:
- 逐 step 监控 `fwd_max`（forward 激活最大值）: 在 crash step 达到 12-14+，超过 FP4 E2M1 max=6.0
- NaN 出现在 backward 梯度而非 forward logits — forward 量化时 >6.0 的值被静默 clip
- 0.6B 从不崩溃（1024 维度下激活值始终在 FP4 范围内）

**论文分析**:
1. arXiv:2605.09825 (AMD): Wgrad 全量化是发散主因；**确定性 Hadamard**（非随机符号）+ H16 是关键
2. arXiv:2509.25149 (NVIDIA): **末尾 ~15% 层保留 BF16** 是头号建议；H=16 是推荐 Hadamard 块大小

**修复（3 项配合）**:
1. `_MXFP4_RHT_G`: 32 → **16**（AMD 论文推荐，kernel 快 8%）
2. `_get_mxfp4_rht_sign()`: `torch.randint` → **`torch.ones`**（确定性符号，消除结构化微缩放误差）
3. 末尾 ~15% 层保留 BF16: `first_last_layers_bf16=True`, `num_layers_at_end_in_bf16=round(0.15×num_layers)`（Qwen3-8B 36 层 → 最后 5 层 BF16）

**验证**: 同一配置（lr=1e-4, warmup=50, grad_clip=1.0）跑 5000 步，**全程无崩溃**：

| Step  | MXFP4 val_loss (修复前, 全量化) | MXFP4 val_loss (修复后, 末5层BF16) |
|------:|-------------------------------:|----------------------------------:|
|   500 | 崩 (lr=6e-5) / 6.73 (lr=1e-4) | 6.71                              |
| 1,500 | 6.15 → step 1550 崩            | 6.15 ✅ 通过                      |
| 2,500 | —                              | 5.87                              |
| 3,500 | —                              | 5.75                              |
| 5,000 | —                              | **5.74**                          |

---

## 修复总结

| Bug | 根因 | 修复 | 影响 |
|-----|------|------|------|
| #1 Dispatch 路由 | `config.scaling.value` 返回 `"blockwise"` | 改用 `config.recipe` 返回 `"mxfp4"` | MXFP4 GEMM 被错误路由到 FP8 blockscale |
| #2 FSDP2 NaN | `save_for_backward` 的 FP4 weight 被 FSDP2 reshard 后失效 | backward 从 `ctx.weight_ref` (BF16) 重新量化 | 多卡训练 step 1 即 NaN |
| #3 Late spike | FP4 dynamic range overflow + 末尾层敏感 | H16 + 确定性 Hadamard + 末 15% 层 BF16 | 8B 训练 step ~1275-3600 崩溃 |

**当前状态**: 8B MXFP4 预训练 5000 步稳定收敛（val_loss 5.74），无 NaN，无发散。
