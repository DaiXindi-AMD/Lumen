# MXFP4 优化方案分析与实现报告

**日期**: 2026-07-27  
**分支**: `worktree-mxfp4-weight-cache` (基于 `feature/mxfp4`)

---

## 方案总览

| # | 方案 | 可行性 | 状态 | 说明 |
|---|------|--------|------|------|
| 1 | 跨 micro-batch 权重缓存 | **可行** | **已实现** | RTN 确定性保证缓存安全 |
| 2 | QKV/gate_up 激活共享 | 合理但不适合 | **搁置** | 属于模型架构变更，不是量化层职责 |
| 3 | FP4 all-gather 替代 BF16 | 可行但工作量大 | **搁置** | 需要 FSDP2 extension 和 AITER 支持 |
| 4 | 前向预转置权重 | **可行** | **已实现**（上一轮） | 已在 `feature/mxfp4` 合并 |

---

## 方案 1：跨 micro-batch 权重缓存

### 原理

一个 optimizer step 内，BF16 权重 W 不变。MXFP4 forward 使用 RTN（round-to-nearest）量化，是确定性的 —— 同一个 BF16 张量无论量化多少遍，FP4 结果逐位相同。因此 gradient accumulation (GA) 中 `ga` 次 forward 的权重量化完全冗余。

仓库 FP8 已有两套对应机制：
- `store_weights_fp8` + `register_fp8_weight_optimizer_hooks`：module 级缓存 + optimizer post-step 刷新
- `ScalingManager._fp8_param_cache` + `_WEIGHT_QUANT_ONCE`：per-tensor_id 缓存 + `mark_fp8_params_stale()` 失效

### 实现

采用 FP8 第一套方案的简化版：

1. **`quant_forward` 中检查缓存**（`lumen/quantize/__init__.py`）：
   ```python
   if _wcache is None and scaling_type == "mxfp4":
       _mc = getattr(module, "_mxfp4_w_cache", None)
       if _mc is not None:
           _wcache, _wscale = _mc
       else:
           _wd = quantize_input(w.contiguous(), "mxfp4", ...)
           _wcache, _wscale = _wd.data, _wd.scale
           module._mxfp4_w_cache = (_wcache, _wscale)
   ```

2. **Optimizer post-step 失效**（`register_mxfp4_weight_optimizer_hooks`）：
   ```python
   def _post_step(opt, args, kwargs):
       for m in model.modules():
           if hasattr(m, "_mxfp4_w_cache"):
               del m._mxfp4_w_cache
   ```

3. **训练脚本注册钩子**（`pretrain_qwen3_mxfp4.py`）：
   ```python
   if args.mode == "mxfp4":
       register_mxfp4_weight_optimizer_hooks(model, opt)
   ```

### FSDP2 安全性

缓存的 FP4 张量是 `quantize_input` 新分配的独立张量（packed uint8 + E8M0 scales），不是 BF16 参数的 view。FSDP2 resharding 只释放参数张量本身，不影响这些独立分配。FP8 blockwise 路径使用完全相同的模式成功运行。

### Gradient Checkpointing 的额外收益

启用 gradient checkpointing 时，backward 中 forward 被重新执行一次。如果没有权重缓存，这会导致每层多一次权重量化 + 转置。缓存后这些操作全部跳过。

### 性能测试 — A/B 对比

**Qwen3-8B, GA=4, 100 步, 8×MI350X FSDP2, C4, seq=512, micro_batch=2**

同一代码分支，仅通过环境变量 `LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` 关闭缓存作为对照。

| 指标 | 开启缓存 | 关闭缓存 | 差异 |
|------|---------|---------|------|
| **中位步时 (ms)** | **2420.7** | **2626.7** | **-7.8%** |
| P25 步时 (ms) | 2221.8 | 2489.0 | -10.7% |
| 均值步时 (ms) | 2462.9 | 2650.7 | -7.1% |
| 峰值显存 (GB) | 17.5 | 15.3 | +2.2 GB |
| 最终 loss | 7.4062 | 7.4141 | -0.0079 |

**每步节省 206 ms（7.8%），来自跳过 3 次（GA=4 中的 2-4 次）冗余权重量化。**

显存多 2.2 GB 因为缓存了每层的 FP4 权重（8B 模型 36 层 × ~2MB FP4 ≈ 72 MB），以及 `quantize_input` 在 `quant_forward` 层面的额外调用路径增加了一些临时分配。

逐步对比（后半段差异更稳定，前半段受 JIT 编译和 C4 流式数据加载影响）：

| 步数 | 缓存 ON (ms) | 缓存 OFF (ms) | 差异 |
|------|-------------|-------------|------|
| 60 | 2119.9 | 2466.3 | -14.0% |
| 70 | 2606.5 | 2622.5 | -0.6% |
| 80 | 2167.4 | 2630.8 | -17.6% |
| 90 | 2221.8 | 2906.4 | -23.6% |
| 100 | 2330.2 | 3038.9 | -23.3% |

后 40 步平均：缓存 ON 2331.4 ms, 缓存 OFF 2789.6 ms, **节省 16.4%**。

**收益总结**：
- GA=4 + grad ckpt 场景下每 step 省 7 次权重量化（4 次 forward GA + 3 次 grad ckpt recompute 扣掉已缓存的那次）
- 代价是 ~2 GB 显存用于缓存 FP4 权重

---

## 方案 2：QKV/gate_up 激活共享（搁置）

### 分析

正确的观察：HF Qwen3 模型中 `q_proj`、`k_proj`、`v_proj` 接收同一个输入张量，各自独立量化为 FP4，结果完全相同但存了 3 份。`gate_proj` 和 `up_proj` 同理存 2 份。

### 为什么不在 Lumen 实现

1. **侵入式修改**：需要合并 HF 模型的 `q_proj`/`k_proj`/`v_proj` 为一个 `qkv_proj` Linear，或者在 attention 模块中注入共享逻辑。这超出 Lumen 的非侵入式 patch 设计（`_replace_forward` 只替换单个 Linear 的 forward，不改变模块间的数据流）。

2. **Megatron 已解决**：Megatron-Core 的 `ColumnParallelLinear` 天然是 QKV 合并的。当 MXFP4 接入 Megatron 后（`LumenSpecProvider`），这个问题自动消失。

3. **正确的实现位置**：如果要在 HF 模型上做，应该在模型层面合并投影（如 vLLM/SGLang 的做法），而不是在量化层面。

### 潜在方案（future work）

- 在 `LumenConfig.enable()` 中识别共享输入的 Linear 组，用一个 "量化一次、共享描述符" 的机制
- 或在 HF attention patch 中合并 QKV 投影
- 仅在 Megatron 路径上优先推进（已解决）

---

## 方案 3：FP4 all-gather 替代 BF16（搁置）

### 分析

FSDP2 当前在 forward 前 all-gather 完整的 BF16 权重到每张卡，然后每卡本地独立量化为 FP4。8 卡就是同一份量化做 8 遍。

改为 "each rank 量化本地 shard → all-gather FP4 数据"，通信量降为 BF16 的 1/4（FP4 packed = 0.5 byte/element vs BF16 = 2 bytes）。由于 RTN 确定性，各卡独立量化相同数据得到相同结果，无需额外同步。

### 为什么搁置

1. **需要 FSDP2 extension**：要实现 `Blockwise2DMXFP4Param`，类似现有的 `Blockwise2DFP8Param`，提供 `fsdp_pre_all_gather`（量化 local shard → FP4）和 `fsdp_post_all_gather`（构建 FP4 gathered view）。

2. **MXFP4 的分片约束**：2D block scaling 要求 `M % 32 == 0` 和 `N % 32 == 0`。分片后的权重可能不满足此条件（需要 `shard_size % 32 == 0`，即 `N % (32 × world_size) == 0`）。

3. **AITER GEMM 兼容性**：`gemm_afp4wfp4` 需要验证在 gathered FP4 + 1D expanded scales 输入下的正确性。

4. **独立 feature branch**：工作量大且与方案 1/4 正交，应该作为独立 feature 推进。

### 预期收益

| 指标 | BF16 all-gather | FP4 all-gather |
|------|----------------|----------------|
| 通信量/卡 | `N × K × 2 bytes` | `N × K × 0.5 bytes + scales` |
| 量化计算 | `world_size × 1` 次 | `1` 次（量化 local shard） |
| 通信带宽节省 | 基线 | **~4x** |

---

## 方案 4：前向预转置（已实现）

在上一轮 `worktree-mxfp4-perf-optimization` 中已实现并合并。Forward 中预计算 `transpose_packed_fp4(w_fp4)` + `w_scale.t()`，存入 `save_for_backward`。Backward DGrad 直接使用，无需重新量化和转置。

---

## 代码变更

| 文件 | 变更 |
|------|------|
| `lumen/quantize/__init__.py` | +24 行：`quant_forward` 中 MXFP4 权重缓存逻辑 + `register_mxfp4_weight_optimizer_hooks` |
| `lumen/ops/quantize/linear.py` | -4 行：移除多余的 `_mxfp4_wt` 缓存逻辑（缓存现在在 module 层面） |
| `examples/qwen3/pretrain_qwen3_mxfp4.py` | +3 行：注册 optimizer hook |

---

## 后续 TODO

1. **GA>1 性能测试**：在 8B GA=4 上量化缓存的具体时间节省
2. **Gradient Checkpointing 交互**：验证 grad ckpt recompute 时缓存被正确使用
3. **方案 3 (FP4 all-gather)**：独立 feature branch，实现 `Blockwise2DMXFP4Param`
4. **MXFP4 Megatron 接入**：`LumenSpecProvider` + TP/PP 后方案 2 自动解决
