# MXFP4 优化方案分析与实现报告

**日期**: 2026-07-27（状态与代码引用已按当前 `feature/mxfp4` / `035431e` 校准）  
**分支**: `worktree-mxfp4-weight-cache` (基于 `feature/mxfp4`)，已由 `30f5277` 合入

---

## 方案总览

| # | 方案 | 可行性 | 状态 | 说明 |
|---|------|--------|------|------|
| 1 | 跨 micro-batch 权重缓存 | **可行** | **已实现** | RTN 确定性保证缓存安全 |
| 2 | QKV/gate_up 激活共享 | 合理但不适合 | **搁置** | 属于模型架构变更，不是量化层职责 |
| 3 | FP4 all-gather 替代 BF16 | 可行 | **已实现**，收益待验证 | 本文写作时判为"搁置"，随后由 `3975266` / `e7776ff` 实现为 `MXFP4CommTensor`，走的是 BF16-in/BF16-out 而非本文设想的 FP4 直通，绕开了当时列的三个障碍 |
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

1. **`quant_forward` 中检查缓存**（`lumen/quantize/__init__.py`）。缓存里存的是
   `(FP4 权重, scale)` 二元组，预转置形式另挂在 FP4 张量的 `_mxfp4_wt_cached` 属性上
   （见下文「优化 A」）：

   ```519:538:lumen/quantize/__init__.py
               if (
                   _wcache is None
                   and scaling_type == "mxfp4"
                   and _os.environ.get("LUMEN_MXFP4_DISABLE_WEIGHT_CACHE") != "1"
               ):
                   _mc = getattr(module, "_mxfp4_w_cache", None)
                   if _mc is not None:
                       _wcache, _wscale = _mc[:2]
                   else:
                       from lumen.ops.quantize.linear import quantize_input as _qi
                       from lumen.ops.quantize.ops import transpose_packed_fp4 as _tp
                       _wd = _qi(
                           w.contiguous(), "mxfp4", fp8_dtype, block_size,
                           None, None, is_weight=True,
                       )
                       _wcache, _wscale = _wd.data, _wd.scale
                       _wt = _tp(_wcache)
                       _wst = _wscale.t().contiguous()
                       _wcache._mxfp4_wt_cached = (_wt, _wst)
                       module._mxfp4_w_cache = (_wcache, _wscale)
   ```

   `LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` 是 A/B 对照用的关闭开关。

2. **Optimizer post-step 失效**（`register_mxfp4_weight_optimizer_hooks`）：

   ```828:833:lumen/quantize/__init__.py
       def _post_step(opt, args, kwargs):
           for m in model.modules():
               if hasattr(m, "_mxfp4_w_cache"):
                   del m._mxfp4_w_cache

       optimizer.register_step_post_hook(_post_step)
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

## 方案 3：FP4 all-gather 替代 BF16（已实现，收益待验证）

### 分析

FSDP2 当前在 forward 前 all-gather 完整的 BF16 权重到每张卡，然后每卡本地独立量化为 FP4。8 卡就是同一份量化做 8 遍。

改为 "each rank 量化本地 shard → all-gather FP4 数据"，通信量降为 BF16 的 1/4（FP4 packed = 0.5 byte/element vs BF16 = 2 bytes）。由于 RTN 确定性，各卡独立量化相同数据得到相同结果，无需额外同步。

### 本文当时判为搁置的三个障碍，以及实际怎么绕开的

1. **需要 FSDP2 extension** —— 确实需要，但不必是本文设想的 `Blockwise2DMXFP4Param`。
   实现出来的是 `MXFP4CommTensor`（`lumen/quantize/comm_tensor.py`），**BF16 进 /
   BF16 出**：`fsdp_pre_all_gather` 把本 rank 的 BF16 shard 量化成 packed MXFP4，
   `fsdp_post_all_gather` 用 `convert_from_mxfp4_2d` 反量化回 BF16。module 拿到的
   还是普通 BF16 权重，forward、optimizer、梯度全都不用改。

2. **MXFP4 的分片约束** —— 没绕开，是显式检查的：`N % (block_size × world_size) == 0`
   且 `K % block_size == 0`，不满足的权重不包，保持 BF16 all-gather
   （`_wrap_params_as_mxfp4_comm` 里 `skipped` 计数）。Qwen3-8B 8 卡下 217 个权重
   全部满足。

3. **AITER GEMM 兼容性** —— 因为选了 BF16-out，gathered 结果是普通 BF16，
   `gemm_afp4wfp4` 那边什么都不用验证。代价是多一次反量化，见下。

`--no-mxfp4-comm` 可以关掉；mxfp4 模式下默认开启：

```295:295:examples/qwen3/pretrain_qwen3_mxfp4.py
            fsdp_mxfp4_comm=(args.mode == "mxfp4" and not args.no_mxfp4_comm),
```

### 实测与代价

| 指标 | BF16 all-gather | FP4 all-gather（实现值） |
|------|----------------|----------------|
| 线上字节 | `N × K × 2 bytes` | `N × K × 0.5 bytes + scales`，实测 **3.99x 更少** |
| 每卡量化计算 | 每 rank 量化完整权重 | 每 rank 只量化自己的 shard（1/world_size） |
| 额外开销 | — | gather 后一次 `convert_from_mxfp4_2d` 反量化 |

数值上 gather 回来的权重是 `RTN(W)`。这不累积（optimizer 更新的是全精度分片 master），
而且 forward 本来就要用同样的 block size 和 RTN 把它量化成 FP4，所以 GEMM 看到的 FP4
操作数不变。

**收益尚未验证**：`convert_from_mxfp4_2d` 是纯 PyTorch，一次调用物化多个全尺寸中间量，
其中 `unpacked.long()` 是 int64（FP4 数据的 16 倍字节）。单节点 xGMI 带宽很高，省下的
3/4 参数通信未必抵得上这个反量化。`--no-mxfp4-comm` 就是为跑对照那一臂加的，对照还没跑。
要转正，反量化得换成 Triton kernel。

---

## 方案 4：前向预转置（已实现）

在上一轮 `worktree-mxfp4-perf-optimization` 中已实现并合并。Forward 中预计算 `transpose_packed_fp4(w_fp4)` + `w_scale.t()`，存入 `save_for_backward`。Backward DGrad 直接使用，无需重新量化和转置。

---

## 追加优化：预转置缓存 + 融合 dequant+transpose kernel

### 优化 A：预转置缓存

除了 FP4 权重，也缓存其预转置形式 `(w_fp4_t, w_scale_t)`。这样
`QuantizedLinearFunction.forward` 中的 `transpose_packed_fp4` 被完全跳过（包括
gradient checkpointing 重算时的 forward 调用）。

存放位置是 FP4 权重张量自己的属性 `_wcache._mxfp4_wt_cached = (_wt, _wst)`，不是
`module._mxfp4_w_cache` 元组的第三项 —— 这样 forward 只要拿到 FP4 权重就能取到它的
转置形式，不必再回头找 module：

```1660:1667:lumen/ops/quantize/linear.py
            # Reuse pre-transposed weight from module cache if available.
            _wt_cached = getattr(weight_desc.data, "_mxfp4_wt_cached", None)
            if _wt_cached is not None:
                w_fp4_t, w_scale_t = _wt_cached
            else:
                from lumen.ops.quantize.ops import transpose_packed_fp4
                w_fp4_t = transpose_packed_fp4(weight_desc.data)
                w_scale_t = weight_desc.scale.t().contiguous()
```

### 优化 B：融合 dequant+transpose Triton kernel

WGrad 中 `convert_from_mxfp4(input_data, input_scale)` 写出 BF16 (M, K)，然后 `.t().contiguous()` 再读写为 (K, M)。新 kernel `_dequant_transpose_mxfp4_kernel` 直接读 packed FP4 (M, K/2) + E8M0 scales，在单个 kernel launch 中 dequant + 写入 transposed BF16 (K, M)，省掉一次 BF16 全矩阵读写。

kernel 实现（`lumen/kernels/mxfp4.py`）：
- `_fp4_e2m1_decode`：4-bit FP4 E2M1 code → float32 的 LUT 解码
- `_dequant_transpose_mxfp4_kernel`：读 packed FP4 → unpack nibbles → LUT dequant → expand scales → `tl.trans` → 写 BF16

---

## 完整 A/B 性能对比

**Qwen3-8B, GA=4, micro_batch=2, seq_len=512, 100 步, 8×MI350X FSDP2, C4**

三档都在同一分支上，靠 `LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` 关缓存做对照。

| 指标 | 无优化 | 权重缓存 | 权重缓存 + 优化 A/B |
|------|--------|---------|---------|
| **中位步时 (ms)** | **2622** | **2330** | **1983** |
| 均值步时 (ms) | 2651 | 2391 | 2063 |
| 最快步时 (ms) | 2402 | 2120 | 1782 |
| 峰值显存 (GB) | 15.3 | 17.5 | 20.3 |
| 最终 loss | 7.4141 | 7.4062 | 7.3828 |

| 相对基线 | 无优化 | 权重缓存 | 权重缓存 + 优化 A/B |
|---------|--------|---------|---------|
| vs 无优化 | — | **-11.1%** | **-24.4%** |
| vs 权重缓存 | — | — | **-14.9%** |

**每 micro-batch 时间**：656 ms → 583 ms → **496 ms**（-24.4%）

> 这一组是 **GA=4 / seq_len=512** 的配置，一步含 4 个 micro-batch，所以绝对步时比
> 其他文档里的数字大得多。合入 `feature/mxfp4` 后在 **seq 2048 × mbs 4 × 8 卡、
> 无 GA** 的标准配置下复测，MXFP4 端到端为 869.4 ms，已快过 BF16 的 928.0 ms
> —— 见 [`mxfp4_optimization_report.md`](mxfp4_optimization_report.md)「验证二」。
> 那份的 1.221x 是本文三项加上 FP4 all-gather、WGrad RTN 修正的合计，无 GA 时权重
> 缓存的收益会小于这里的 -11.1%（一步只有一次 forward，缓存只在 gradient
> checkpointing 重算时才命中）。

---

## 代码变更

| 文件 | 变更 | 说明 |
|------|------|------|
| `lumen/quantize/__init__.py` | +30 行 | 权重缓存 + 预转置缓存 + optimizer hook |
| `lumen/ops/quantize/linear.py` | +10/-8 行 | 使用缓存 transpose + 融合 dequant_transpose |
| `lumen/ops/quantize/ops.py` | +37 行 | `dequant_transpose_mxfp4` wrapper |
| `lumen/kernels/mxfp4.py` | +90 行 | `_fp4_e2m1_decode` + `_dequant_transpose_mxfp4_kernel` |
| `examples/qwen3/pretrain_qwen3_mxfp4.py` | +3 行 | 注册 optimizer hook |

本分支后续（方案 3 与相关修正）：

| 文件 | 说明 |
|------|------|
| `lumen/quantize/comm_tensor.py` | `MXFP4CommTensor`：FSDP2 FP4 all-gather 的两个钩子 |
| `lumen/models/fsdp.py` | `_wrap_params_as_mxfp4_comm` + `fsdp_mxfp4_comm` 开关、对齐检查 |
| `examples/qwen3/pretrain_qwen3_mxfp4.py` | `--no-mxfp4-comm` 关闭开关 |
| `lumen/ops/quantize/linear.py` | WGrad 激活改回 RTN（`use_sr=False`，两处调用点） |
| `tests/ops/test_quantize.py` | `test_mxfp4_backward_gradients_track_the_bf16_reference`（dW 12.9 dB / dX 13.7 dB） |

---

## 后续 TODO

1. ~~**方案 3 (FP4 all-gather)**~~：已实现为 `MXFP4CommTensor`（BF16-in/BF16-out），
   但收益未验证 —— 先跑 `--no-mxfp4-comm` 的对照臂
2. **把 `convert_from_mxfp4_2d` 换成 Triton kernel**：方案 3 转正的前提，现在是纯
   PyTorch，物化多个全尺寸中间量（含一个 int64）
3. **MXFP4 Megatron 接入**：`LumenSpecProvider` + TP/PP 后方案 2 自动解决
4. **显存优化**：本文配置下 20.3 GB（+5 GB vs 无优化），标准配置下实测 20.9 GB
   （vs BF16 15.3 GB）。可考虑只缓存预转置形式（DGrad 只用它），或 micro-batch
   结束时按需释放
