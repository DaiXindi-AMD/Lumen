# MXFP4 训练性能优化记录

本文记录 MXFP4 训练路径优化的全过程：每项改动的动机、实测数据、核心代码，以及
最后的收敛与性能验证。硬件为 MI350X（gfx950），模型为 Qwen3-8B / Qwen3-0.6B。

分两轮。第一轮（改动 1–4）冲的是 GEMM 和它周边的拷贝，在单卡上测；第二轮（改动 5–8）
来自 `worktree-mxfp4-weight-cache`，冲的是跨 micro-batch 的冗余量化和参数通信，随
`30f5277` 合入，在 8 卡上测。第二轮才让 MXFP4 端到端超过 BF16。

后端选择机制本身（autotune、调表流程）见
[`mxfp4_gemm_backend_selection.md`](mxfp4_gemm_backend_selection.md)，本文侧重优化项与量化数据。

---

## 结论摘要

前四项在单卡 8192 tokens/步下测量；后四项来自 `worktree-mxfp4-weight-cache`，随
`30f5277` 合入，在 8 卡 65536 tokens/步下测量。

| 改动 | GEMM / 算子层面 | 端到端（Qwen3-8B） |
|---|---|---|
| 1. autotune 后端选择 | Qwen3-8B 0.90x → 1.02x | +0.4%，并消除 −2.6% 倒退 |
| 2. 跳过冗余 scale padding | 前处理 92 → 60 µs | 含在下行 |
| 3. 向量化 weight shuffle | Qwen3-8B 1.02x → 1.12x | 含在上行的 +0.4% 内 |
| 4. **wgrad 去掉转置物化** | 单个操作数 2.0–2.7x | **+7.8%** |
| 5. **跨 micro-batch 权重缓存** | 省掉每次冗余 forward 的权重量化+预转置 | 见下（4 项合计） |
| 6. **融合 dequant+transpose kernel** | 省掉一份 BF16 (M,K) 中间缓冲 | 见下 |
| 7. FP4 all-gather（`MXFP4CommTensor`） | 参数通信字节 3.99x 更少 | 见下，**收益未单独验证** |
| 8. WGrad 激活改回 RTN | 非性能项，是正确性修正 | — |

**端到端**（Qwen3-8B，8× MI350X FSDP2 full_shard，C4 streaming，seq 2048 × mbs 4 ×
8 卡 = 65536 tokens/步，lr 3e-4 warmup 200；取日志中 step ≥ 50 的 `step_time_ms` 中位数）：

| 构建 | 中位步时 | P25 | 最快 | 显存/卡 |
|---|---|---|---|---|
| BF16 | 928.0 ms | 917.0 | 910.7 | 15.30 GB |
| MXFP4，改动 1–4 后（`7d1841b`） | 1061.8 ms | 1001.1 | 989.0 | 16.10 GB |
| MXFP4，改动 1–8 后（`035431e`） | **869.4 ms** | 865.0 | 860.2 | **20.90 GB** |

**MXFP4 现在比 BF16 快 6.7%（928.0 → 869.4 ms），代价是显存多 5.6 GB（+37%）。**
改动 5–8 合计把 MXFP4 从比 BF16 慢 14% 变成快 6.7%，即 1061.8 → 869.4 ms（1.221x）。

**必须同时记住的三件事：**

1. 改动 1–3 我是**按假设**优化 GEMM 的，端到端只值 0.4%；改动 4 是 **profile 指出来**的，
   值 7.8%。后者代码量最小，收益是前者的 20 倍。
2. 改动 5–8 的 1.221x 是**四项合计**，没有逐项 A/B。`LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1`
   和 `--no-mxfp4-comm` 两个开关就是为了做这个拆分而加的，但拆分还没跑。特别是改动 7
   很可能是**负收益**（下文），不要把 1.221x 记成"四项都有效"。
3. 显存从 16.1 GB 涨到 20.9 GB，全部来自改动 5 缓存的 FP4 权重及其预转置。在这个
   配置下换来了速度，但它是**按层数线性增长**的，更大的模型或更长的序列可能换不起。

---

## 改动 1：autotune 后端选择

### 问题

MXFP4 GEMM 有三个后端（plain Triton / shuffled Triton / AITER ASM）。原先用硬编码的
字节阈值路由，阈值是在 Llama 3.1 8B 上量出来的。Qwen3-8B 的 intermediate 是 3072
（Llama 是 14336），权重尺寸落在阈值的另一侧，导致：

- Qwen3-8B 每层 GEMM 从 4.809 ms 变成 5.329 ms，**比不做任何优化还慢 10%**
- Qwen3-0.6B 21 个形状全部落回 plain，优化完全没生效

### 做法

改成首次调用时实测三个后端、按 `(M, N, K)` 缓存决策。要点：

- **交错轮询计时**，而不是顺序跑完一个再跑下一个——否则第一个候选会吃掉冷缓存而蒙冤
- 取 **11 次的中位数**而非均值，抵抗离群值
- **5% 切换余量**：只有明显更快才从 plain 切走，避免决策在噪声中反复横跳
- 决策可持久化到磁盘并带 arch 标记，让远程训练复现 warmup 阶段的选择

```python
# lumen/ops/quantize/mxfp4_autotune.py
AUTOTUNE_ENABLED = os.environ.get("LUMEN_MXFP4_AUTOTUNE", "1") == "1"
_SWITCH_MARGIN = 1.05   # 只有比 plain 快 5% 以上才切换
_WARMUP_ITERS = 3
_TIMED_ITERS = 11       # 中位数而非均值，抵抗离群
```

配套 `scripts/mxfp4_tune_shapes.py` 提供四阶段调表流程（collect / untuned / tune /
verify），其中 verify 阶段要求与 plain Triton **逐位一致**才入库，拒绝 AITER 默认的
"5% 元素不匹配"容差。

### 结果（每层训练 GEMM 合计，三轮重复）

| 模型 | plain | 静态阈值 | autotune |
|---|---|---|---|
| Qwen3-0.6B | 2.345 ms | 2.345 (1.00x) | 2.271 (1.03x) |
| Qwen3-8B | 4.809 ms | 5.329 (**0.90x**) | 4.292 (1.12x) |
| Llama-3.1-8B | 7.754 ms | 5.986 (1.30x) | 4.836 (1.60x) |

---

## 改动 2：跳过冗余的 scale padding

### 问题

ASM 后端每次调用都要把两个 scale 张量 pad 并 swizzle。拆开计时发现，一个 1 MiB 的
scale 要花 30 µs——而按带宽算只需要 **0.2 µs**。

```
scale (8192, 128) -> padded (8192, 128) = 1024 KiB
needs padding: rows False, cols False        <-- 根本不需要 pad

  torch.zeros alloc+memset          11.8 us
  + slice copy                      17.2 us
  shuffle_scale_gemm alone          21.7 us
  .reshape().contiguous()            5.9 us   <-- 是 no-op
  whole _pad_and_swizzle            34.2 us

  empty kernel launch floor         11.6 us   <-- 启动开销地板
  1.00 MiB at ~5 TB/s would be       0.2 us
```

三个事实：scale 要求行对齐 256、列（K/32）对齐 8，**真实训练形状全部满足**；
`torch.zeros` 那 11.8 µs 恰好等于空 kernel 的启动地板，也就是它什么也没干；
整个 pad 在造一份逐字节相同的副本。

### 代码

```python
# lumen/ops/quantize/linear.py::_pad_and_swizzle_mxfp4_scale
    if (rows, cols) == (rows_pad, cols_pad):
        # Training shapes are almost always aligned already (rows is the token
        # count or a hidden dim, cols is K/32). Allocating and filling a copy
        # that is byte-identical to the input costs ~17us of launch overhead per
        # scale, which is real money next to a ~250us GEMM.
        padded = scale if scale.is_contiguous() else scale.contiguous()
    else:
        padded = torch.zeros(
            (rows_pad, cols_pad), dtype=scale.dtype, device=scale.device
        )
        padded[:rows, :cols] = scale
```

### 结果

两个 swizzle 各 30.4 → 18.6 µs，前处理合计 92.2 → 59.8 µs。

**新增风险与对应测试**：跳过拷贝后，返回值可能变成调用方 scale 张量的视图（原先
`padded` 永远是新副本）。`test_mxfp4_aligned_scale_swizzle_does_not_alias_input` 钉住
这一点。

---

## 改动 3：向量化 weight shuffle

### 问题

`shuffle_weight(w, layout=(16,16))` 对 24 MiB 权重要 41.9 µs，算下来只有 **1.2 TB/s**，
而 MI350X 的 HBM 带宽是 ~8 TB/s。

AITER 的实现是 6 维 view + permute + `.contiguous()`：

```python
# aiter/ops/shuffle.py
x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
x_ = x_.permute(0, 1, 3, 4, 2, 5)
x_ = x_.contiguous()
```

### 一次失败的尝试

我先写了专用 Triton 内核（按输出索引、保证 store 连续），结果是 **0.66x——比 aiter 还慢**。
原因是 gather load 每 16 字节就跨行一次，合并访存很差。这条弯路说明：**先假设"手写内核
更快"是错的，得先弄清楚慢在哪。**

### 正确的解法

关键观察：这个置换的**最内层 16 字节在源和目标里都连续**，所以它本质是 16 字节单元的
转置。AITER 用 `uint8` 视图表达，PyTorch 的 TensorIterator 只能逐字节搬，无法向量化。
把同样的字节按 `int64` 看，拷贝就能向量化——不用写任何内核。

```python
# lumen/ops/quantize/linear.py
_MXFP4_WIDE_SHUFFLE_MIN_BYTES = 4 << 20

def _shuffle_mxfp4_weight(w_fp4, arch=None):
    """AITER's ``layout=(16, 16)`` B-operand shuffle, over wider elements."""
    from aiter.ops.shuffle import shuffle_weight
    dtype = w_fp4.dtype
    w = w_fp4
    if hasattr(torch, "float4_e2m1fn_x2") and dtype == torch.float4_e2m1fn_x2:
        w = w.view(torch.uint8)
    if (
        arch == "gfx1250"                      # 不同的 WMMA 布局
        or w.ndim != 2
        or not w.is_contiguous()
        or w.numel() < _MXFP4_WIDE_SHUFFLE_MIN_BYTES
        or w.shape[0] % _MXFP4_SHUFFLE_N_MULTIPLE
        or w.shape[1] % 32
    ):
        return shuffle_weight(w_fp4, layout=(16, 16))

    n, kp = w.shape
    wide = w.view(torch.int64).view(n // 16, 16, kp // 32, 2, 2)
    wide = wide.permute(0, 2, 3, 1, 4).contiguous()
    return wide.view(torch.uint8).view(n, kp).view(dtype)
```

### 结果（全部逐位一致）

| 权重 (N,K) | aiter uint8 | 我的 triton | **view int64** |
|---|---|---|---|
| q3-8b gate (12288, 4096) | 41.9 µs | 64.0 | **23.7** |
| q3-8b down (4096, 12288) | 41.4 | 63.9 | **23.4** |
| llama gate (28672, 4096) | 77.5 | 126.0 | **33.7** |
| q3-0.6b gate (3072, 1024) | **20.2** | 22.4 | 22.3 |

带宽 1.2 → 2.1–3.5 TB/s。小张量（0.6B）反而略差 2 µs，因为那里启动开销主导、搬数据
根本不是瓶颈——所以有 4 MiB 的尺寸门槛，低于它退回 AITER。

---

## 改动 2+3 合并效果

跨进程测量有 ~1% 漂移，会淹没要测的效果，所以**在同一进程内交错跑新旧两条路径**：

| 形状 | plain | 旧前处理 | 新前处理 |
|---|---|---|---|
| q3-8b gate fprop | 382.3 µs | 358.0 (1.07x) | 328.3 (**1.16x**) |
| q3-8b gate dgrad | 347.2 µs | 313.8 (1.11x) | 287.7 (**1.21x**) |
| q3-8b down wgrad | 351.9 µs | 325.2 (1.08x) | 298.5 (**1.18x**) |
| q3-8b qkv fprop | 233.8 µs | 256.9 (0.91x) | 228.1 (**1.02x**) |
| llama gate fprop | 2456.1 µs | 635.1 (3.87x) | 612.8 (**4.01x**) |

每次 ASM 调用稳定省 22–30 µs。这个固定收益让不少形状从"不划算"翻成"划算"：
Qwen3-8B 选中 ASM 的形状从 **3 个变成 9 个**，整模型 1.02x → 1.12x。

---

## 转折点：profile 说了什么

到此为止全是 GEMM 层面的数。端到端一测（Qwen3-8B，8192 tokens/步，单卡，两轮中位数）：

| 策略 | 步时间 | vs plain |
|---|---|---|
| plain | 1102.6 ms | 1.00x |
| 静态阈值 | 1130.9 ms | 0.97x |
| autotune | 1097.8 ms | 1.004x |

**整个后端选择工作端到端只值 0.4%**，真实价值是避免了静态策略 −2.6% 的倒退。

于是给训练脚本加了 profiler 开关（`--profile-steps N`，可选 `--profile-out` 导 chrome
trace），实测一步的时间去向：

| 项 | 每步 | 占比 |
|---|---|---|
| **`aten::copy_`** | **267 ms** | **24.6%** |
| AdamW | 126 ms | 11.6% |
| `aten::mul` | 120 ms | 11.0% |
| flash attention 反向 | 119 ms | 10.9% |
| **全部 MXFP4 GEMM** | **102 ms** | **9.4%** |
| BF16 `mm`（tail 层） | 75 ms | 6.9% |

**GEMM 只占 9.4%——把它变成零也只有 9.4%。**而拷贝是它的 2.6 倍。按操作数形状拆开：

| 形状 | 每步 | 次数/步 |
|---|---|---|
| `[12288, 8192]` | 43 ms | 93 |
| `[4096, 8192]` | 42 ms | 279 |
| `[192946432]`（FSDP 扁平参数） | 29 ms | 72 |
| `[4, 2048, 4096]` | 23 ms | 509 |

前两行合计 7.8%，直指 wgrad 路径。

---

## 改动 4：wgrad 去掉转置物化（收益最大）

### 问题

```python
input_bf16 = convert_from_mxfp4(input_data, input_scale, ...)
grad_t = grad_flat.t().contiguous()      # 物化一份转置副本
input_t = input_bf16.t().contiguous()    # down_proj 这个是 201 MB
grad_t_fp4,  _ = hadamard_quant_mxfp4(grad_t,  ...)
input_t_fp4, _ = hadamard_quant_mxfp4(input_t, ...)
```

每个 linear 的反向都要把数据完整搬三遍，只为了拿到一个转置。

### 关键发现

`_fused_hadamard_quant_mxfp4_kernel` **本来就用两个 stride 定址**：

```python
# lumen/kernels/mxfp4.py
offs_x = offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
x = tl.load(x_ptr + offs_x).to(tl.float32)
```

它完全能直接读转置视图。是包装函数里一句 `.contiguous()` 把这个能力挡住了。所以
**不需要写任何新内核**——删掉那句、把视图传进去就行。

### 为什么能保证逐位一致

这一点在动手前就必须确认，因为悄悄改变梯度比慢得多要糟。随机舍入的随机数来源：

```python
# lumen/kernels/mxfp4.py::_generate_randval
def _generate_randval(m, n, philox_seed, philox_offset):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    rng_offsets = philox_offset + ms[:, None] * n + ns[None, :]
    r1, _, _, _ = tl.randint4x(philox_seed, rng_offsets)
    return r1
```

只用 tile 内的 `tl.arange` 和固定 offset，**不依赖元素地址**。所以 strided 读看到的
随机流与 dense 读完全相同，且 tile 划分未变 → 结果逐位相同。

### 代码

```python
# lumen/ops/quantize/ops.py::hadamard_quant_mxfp4
    orig_shape = x.shape
    # Deliberately not .contiguous(): the kernel addresses x through both
    # strides, so a transposed view works as-is. Callers wanting x^T would
    # otherwise have to materialise it, which on Qwen3-8B's wgrad path cost
    # more GPU time than the wgrad GEMM itself.
    x_2d = x.reshape(-1, orig_shape[-1])
```

当时两个操作数都改成了视图：

```python
# lumen/ops/quantize/linear.py，wgrad 路径（本改动当时的形态）
                    grad_t = grad_flat.t()
                    input_t = input_bf16.t()
```

### 结果（全部逐位一致）

| wgrad 操作数 | 物化 | 视图 | 加速 |
|---|---|---|---|
| down `input_t` 12288×8192 | 628.0 µs | 234.8 µs | 2.67x |
| gate `grad_t` 12288×8192 | 623.9 µs | 230.6 µs | 2.71x |
| qkv `grad_t` 6144×8192 | 322.8 µs | 134.7 µs | 2.40x |
| o_proj `input_t` 4096×8192 | 209.2 µs | 103.6 µs | 2.02x |

端到端 **1097.8 → 1018.4 ms（1.078x）**，`aten::copy_` 占比 24.6% → 17.4%。

**后续变化**：`grad_t` 至今仍是视图，但 `input_t` 已被改动 6 换成融合 kernel —— 视图
省掉了转置拷贝，却让量化 kernel 变成跨步读；融合 kernel 两头都省。当前代码是：

```2087:2092:lumen/ops/quantize/linear.py
                    grad_t = grad_flat.t()
                    # Fused, so no separate BF16 (M, K) dequant buffer is written.
                    # It also lands dense, which the non-RHT quantizer below needs.
                    input_t = dequant_transpose_mxfp4(
                        input_data, input_scale, block_size=mxfp4_block,
                    )
```

---

## 改动 5：跨 micro-batch 权重缓存

### 问题

一个 optimizer step 内 BF16 权重不变，而 MXFP4 权重量化走 RTN，是确定性的 —— 同一个
BF16 张量量化多少遍，FP4 结果逐位相同。所以 gradient accumulation 里每个 micro-batch
的 forward 都在重算一份完全一样的 FP4 权重，外加一次 `transpose_packed_fp4` 预转置。
开了 gradient checkpointing 还要再多算一遍（backward 重跑 forward）。

### 做法

在 module 上挂一层缓存，optimizer 走完 `step()` 再失效。仓库里 FP8 已有两套同类机制
（`store_weights_fp8` + `register_fp8_weight_optimizer_hooks`，以及
`ScalingManager._fp8_param_cache`），这里取了前者的简化版：

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

预转置形式挂在 FP4 张量自己的属性上（`_mxfp4_wt_cached`），forward 直接取用，取不到
才自己算：

```1659:1673:lumen/ops/quantize/linear.py
        elif scaling_type == "mxfp4":
            # Reuse pre-transposed weight from module cache if available.
            _wt_cached = getattr(weight_desc.data, "_mxfp4_wt_cached", None)
            if _wt_cached is not None:
                w_fp4_t, w_scale_t = _wt_cached
            else:
                from lumen.ops.quantize.ops import transpose_packed_fp4
                w_fp4_t = transpose_packed_fp4(weight_desc.data)
                w_scale_t = weight_desc.scale.t().contiguous()
            ctx.save_for_backward(
                input_desc.data,
                input_desc.scale,
                w_fp4_t,
                w_scale_t,
            )
```

注意 `save_for_backward` 存的是**预转置形式**（`w_fp4_t`），不是 W 和 W^T 两份 ——
DGrad 只需要 W^T，存两份没有用处。

失效钩子挂在 optimizer 上：

```828:833:lumen/quantize/__init__.py
    def _post_step(opt, args, kwargs):
        for m in model.modules():
            if hasattr(m, "_mxfp4_w_cache"):
                del m._mxfp4_w_cache

    optimizer.register_step_post_hook(_post_step)
```

### FSDP2 安全性

缓存的是 `quantize_input` 新分配的独立张量（packed uint8 + E8M0 scales），不是 BF16
参数的 view。FSDP2 reshard 只释放参数张量本身，不影响这些独立分配。FP8 blockwise
路径用的是同一个模式。

### 代价

显存。每个量化层缓存 FP4 权重 + 预转置 FP4 权重，Qwen3-8B 每层权重 192.9M 元素
→ FP4 各 96.4 MB，两份约 193 MB/层，31 个量化层合起来接近 6 GB。实测峰值显存
16.1 → 20.9 GB（+4.8 GB），量级吻合。

---

## 改动 6：融合 dequant+transpose kernel

### 问题

WGrad 要把 forward 存的 FP4 激活变成 BF16 `(K, M)`。原来分两步：
`convert_from_mxfp4` 写出完整的 BF16 `(M, K)`，再转置。down_proj 在 8192 tokens 下
那个中间缓冲是 192 MiB，写一遍读一遍就扔。

改动 4 把转置改成视图后，拷贝没了，但 `hadamard_quant_mxfp4` 变成跨步读，而且非 RHT
分支下游的 `convert_to_mxfp4` 会路由到 AITER 的量化 kernel，那个要求操作数连续。

### 做法

新 Triton kernel 直接读 packed FP4 `(M, K/2)` + E8M0 scales，在一个 launch 里
dequant 完写出转置好的 BF16 `(K, M)`：中间那份 BF16 `(M, K)` 根本不存在，且落地是连续的。

```897:907:lumen/ops/quantize/ops.py
def dequant_transpose_mxfp4(
    data_fp4: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """Fused dequant + transpose: packed FP4 (M, K/2) + 1D scales → BF16 (K, M).

    Equivalent to ``convert_from_mxfp4(data, scales).t().contiguous()`` but
    eliminates one full BF16 (M, K) intermediate write.
    """
    from lumen.kernels.mxfp4 import _dequant_transpose_mxfp4_kernel
```

kernel 侧是 `_dequant_transpose_mxfp4_kernel`（`lumen/kernels/mxfp4.py`）：
读 packed FP4 → 拆 nibble → LUT 反量化 → 展开 scales → `tl.trans` → 写 BF16。

与 `convert_from_mxfp4(...).t().contiguous()` **逐位一致**，由
`benchmarks/bench_mxfp4_backward_ops.py::TestWgradActivationPrep::test_fused_dequant_transpose_is_bit_exact`
在 K=4096 和 K=12288 两个形状上钉住。

### 算子级数字待补

这一项的单算子 A/B 需要一张空闲卡：写这份文档时 8 张卡都在跑训练，测出来同一形状
三次能差 2 倍，不能入库。`benchmarks/bench_mxfp4_backward_ops.py` 是复现用的，
在空闲卡上跑 `pytest benchmarks/bench_mxfp4_backward_ops.py -v -s`，它会把
物化 / 视图 / 融合三条路径交错计时并按层汇总。

---

## 改动 7：FP4 参数 all-gather（收益未验证）

### 做法

`MXFP4CommTensor` 包住 BF16 参数，给 FSDP2 提供两个钩子：all-gather 前把本 rank 的
BF16 shard 量化成 packed MXFP4（2D 32×32 block scales），gather 完再反量化回 BF16。
线上字节数从 2 byte/元素降到 0.5 byte/元素加 E8M0 scales，实测 **3.99x**。optimizer、
梯度和 forward 都只看到正常的 BF16 权重。

对齐要求 `N % (32 × world_size) == 0` 且 `K % 32 == 0`，不满足的权重保持 BF16。
Qwen3-8B 8 卡下包住了 **217** 个权重（训练日志里的
`> MXFP4CommTensor wrapping: 217 weights`）。

数值上：gather 回来的权重是 `RTN(W)`，比 BF16 master 少一次舍入。这不会累积 ——
optimizer 更新的是分片的全精度 master，只有 gather 出来的副本被舍入；而且 forward
本来就要用同样的 block size 和 RTN 把它量化成 FP4，所以 GEMM 看到的 FP4 操作数不变
（`TestFP4AllGather::test_roundtrip_is_rtn_quantization` 钉住这一点）。

### 为什么怀疑它是负收益

`convert_from_mxfp4_2d` 是纯 PyTorch，一次调用要物化好几个全尺寸中间量，其中
`unpacked.long()` 是 int64，**单这一个就是 FP4 数据的 16 倍字节**：

```696:705:lumen/ops/quantize/ops.py
    # Unpack nibbles
    _mxfp4_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32, device=data_flat.device,
    )
    unpacked = data_flat.view(torch.uint8).repeat_interleave(2, dim=-1)
    unpacked[..., ::2] = unpacked[..., ::2] & 0xF
    unpacked[..., 1::2] = unpacked[..., 1::2] >> 4
    values = _mxfp4_lut[unpacked.long()]  # (M, N)
```

单节点 8 卡的 xGMI 带宽很高，省下来的 3/4 参数通信字节未必抵得上这个反量化。所以
`035431e` 加了 `--no-mxfp4-comm` 把它关掉，**就是为了能跑对照的那一臂**；对照还没跑。
上表 869.4 ms 是**开着** FP4 all-gather 测的，所以它是净收益 1.221x 里的一部分，
但不知道是正的还是负的。

要真正划算，反量化得换成 Triton kernel，或者干脆让 forward 直接消费 gather 回来的
FP4（省掉反量化再量化的往返）——后者要求 gathered FP4 的 2D scales 布局能直接喂
`gemm_afp4wfp4`。

---

## 改动 8：WGrad 激活改回 RTN（正确性修正）

WGrad 量化两个操作数，梯度 dY^T 和激活 X^T，原来两个都用随机舍入。NVFP4 §4.4 和
附录 E.3 把 SR 限定在梯度上：用在激活上收益很小，还可能发散。DGrad 本来就是对的
（梯度 SR，权重复用 forward 缓存的 RTN 结果）。

```2099:2104:lumen/ops/quantize/linear.py
                        # NVFP4 §4.4 / E.3: stochastic rounding belongs on the
                        # gradient only. On activations it buys little and can
                        # diverge, so the activation stays round-to-nearest.
                        input_t_fp4, input_t_scale = hadamard_quant_mxfp4(
                            input_t, sign_m, block_size=mxfp4_block, g=rht_g, use_sr=False,
                        )
```

Hadamard 分支和非 RHT 分支两处都改了。这不是性能项，但它改变梯度数值，所以上表里
MXFP4 那一行的 loss 曲线不能和改动 5–8 之前的直接逐点对比。

---

## 验证一：改动 1–4 后的 8 卡 2000 步训练

Qwen3-8B，8× MI350X FSDP full_shard，c4 streaming，seq 2048 × mbs 4 × 8 卡 =
65536 tokens/步，lr 3e-4 warmup 200，MXFP4 与 BF16 除 `--mode` 外配置完全一致。

| | val@500 | val@1000 | val@1500 | val@2000 | 中位步时间 | 显存/卡 |
|---|---|---|---|---|---|---|
| MXFP4（`7d1841b`） | 5.3977 | 4.6242 | 4.2641 | 4.2336 | 1061.8 ms | 16.10 GB |
| BF16 | 5.3461 | 4.5637 | 4.2152 | 4.1852 | 928.0 ms | 15.30 GB |
| 差 | +0.052 | +0.061 | +0.049 | +0.048 | | |

**收敛健康**：差距稳定在 ~0.05 nats 且不随训练扩大，说明量化没有累积性损害。

此时 **MXFP4 比 BF16 慢 14%，显存多 5%**，在这个配置下还不是性能优化手段。

> 步时间口径：日志里 `step_time_ms` 取 step ≥ 50 的中位数。C4 streaming 的取数停顿
> 会造成偶发的 2–12 秒离群步，均值会被抬高 40%（MXFP4 1537 ms、BF16 1437 ms），
> 中位数不受影响，且在 step ≥ 0 / 50 / 100 / 200 四个窗口下都稳定在 ±1.5% 内。
> 本文早期版本对这两个 run 记的是 1075.1 / 925.7 ms，与此处口径略有差异，结论一致。

---

## 验证二：改动 1–8 后（当前 HEAD `035431e`）

同一配置、同一口径，MXFP4 重跑：

| | 中位步时 | P25 | 最快 | 显存/卡 | vs BF16 |
|---|---|---|---|---|---|
| BF16 | 928.0 ms | 917.0 | 910.7 | 15.30 GB | 1.00x |
| MXFP4 改动 1–4（`7d1841b`） | 1061.8 ms | 1001.1 | 989.0 | 16.10 GB | 0.87x |
| MXFP4 改动 1–8（`035431e`） | **869.4 ms** | 865.0 | 860.2 | 20.90 GB | **1.067x** |

**MXFP4 第一次比 BF16 快** —— 快 6.7%，代价是显存多 5.6 GB（+37%）。相对改动 1–4
是 1.221x。

同一 HEAD 的另一个 run（10000 步配置，同参数）P25 866.6 ms、最快 858.8 ms，与上表
的 865.0 / 860.2 一致；它的中位数 916.7 ms 偏高是取数停顿更多，不是构建不同。

收敛（HEAD run，改动 8 改过 WGrad 舍入，所以不与验证一逐点对比）：

| step | 500 | 700 | 900 | 1000 | 1100 | 1200 |
|---|---|---|---|---|---|---|
| val_loss | 5.1367 | 4.9633 | 4.7664 | 4.6359 | 4.5352 | 4.4441 |

单调下降，无 spike。写这份文档时该 run 跑到 step ~1250/3000，尚未结束，上面的步时
统计基于 step 10–1250。

**这个 1.221x 是改动 5–8 的合计**，没有逐项拆分。拆分的开关都在
（`LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1`、`--no-mxfp4-comm`），但对照还没跑；改动 7
尤其可能是负的，见该节。

---

## 方法论教训

这轮里几次差点得出错误结论，记录下来：

1. **跨进程比较不可信。** 分别跑新旧配置时机器有 ~1% 漂移，方向随机，足以淹没 10% 以内
   的效果。必须在同一进程内交错跑。

2. **各部分的耗时不等于流水线的耗时。** 单独计时每个 op 时它们各自付满启动延迟；串在
   一起时 CPU 派发与 GPU 执行重叠。改动后各部件明显变快而总时间没变，不是改动无效，
   是这两个数不可加。

3. **小模型端到端测不出东西。** Qwen3-0.6B 三轮下来，三种策略的差异完全被 ±4% 的运行
   间噪声淹没；Qwen3-8B 同配置内部波动只有 0.14%，才测得出 0.4% 的差别。

4. **按假设优化 vs 按 profile 优化。** 前者（改动 1–3）花了绝大部分时间，端到端 0.4%；
   后者（改动 4）代码量最小，端到端 7.8%。**先 profile。**

5. **GEMM 不是唯一的量化开销。** 改动 1–4 把 GEMM 层面做到 1.12x，端到端只有 0.4%；
   改动 5–8 一行 GEMM 代码都没动，端到端 1.221x。真正贵的是"每个 micro-batch 重算一遍
   同样的权重"和"为了转置物化整个矩阵"这类冗余，不是 kernel 选得对不对。

6. **测量前先看卡是不是空的。** 整理这份文档时 8 张卡都在跑训练，同一形状的算子 A/B
   三次能差 2 倍，方向还不固定。端到端步时（本身就是在训练里测的）不受影响，算子级
   数字则完全不可用 —— 宁可留空也不要入库。`rocm-smi --showpids` 花两秒。

---

## 测试

```
pytest tests/ops/test_quantize.py -k "mxfp4 or hadamard" -q      # 47 passed
pytest benchmarks/bench_mxfp4_gemm.py -v -s                      # 5 passed
pytest benchmarks/bench_mxfp4_gemm_models.py -v -s               # 3 passed
pytest benchmarks/bench_mxfp4_backward_ops.py -v -s              # 5 passed（需空闲卡）
```

改动 1–4 新增的关键测试：

| 测试 | 守住什么 |
|---|---|
| `test_mxfp4_weight_shuffle_matches_aiter` | int64 视图 shuffle 与 AITER 逐字节一致（4 个形状） |
| `test_mxfp4_weight_shuffle_falls_back_when_wide_view_invalid` | gfx1250 / 非连续 / 小张量正确退回 |
| `test_mxfp4_aligned_scale_swizzle_does_not_alias_input` | 跳过 pad 后不返回调用方 scale 的视图 |
| `test_hadamard_quant_reads_transpose_without_materialising` | 转置视图量化与物化结果逐位一致 |
| `test_mxfp4_backends_are_interchangeable` | 三个后端逐位一致（autotune 的正确性前提） |

改动 5–8 新增的：

| 测试 | 守住什么 |
|---|---|
| `test_mxfp4_backward_gradients_track_the_bf16_reference` | dW / dX 对 `F.linear` 的 SNR。实测 dW 12.9 dB、dX 13.7 dB，断言下限 11 dB。WGrad 两个操作数走不同路径（梯度是跨步视图，激活走融合 kernel），弄反了照样能出形状合理的梯度，所以需要一个数值参照 |
| `TestWgradActivationPrep::test_fused_dequant_transpose_is_bit_exact` | 融合 dequant+transpose 与 `convert_from_mxfp4(...).t().contiguous()` 逐位一致 |
| `TestFP4AllGather::test_roundtrip_is_rtn_quantization` | gather 回来的权重重新量化后与直接量化原权重得到同一份 FP4，即 FP4 all-gather 不改变 forward |

完整跑 `test_quantize.py` 会在一个 FP8/torchao 测试处 abort，该问题早于本轮改动
（stash 掉全部改动后同样复现）。

---

## 已知遗留

profile 数字来自改动 1–4 之后、改动 5–8 之前的单卡 8192 tokens/步 run，占比在
当前 HEAD 上应已变化（步时降了 18%），**尚未重新 profile**：

| 项 | 每步 | 说明 |
|---|---|---|
| `aten::copy_` 剩余部分 | ~175 ms | FSDP 扁平参数 29 ms、`[4,2048,4096]` 23 ms 等尚未归因 |
| `aten::mul` | 120 ms | 疑似反量化/缩放，未拆解 |
| MXFP4 GEMM 前处理 | ~60 µs/次 | 融合 cast+shuffle 可再拿 GEMM 层面 1.3x，端到端约 3–4% |
| ~~wgrad 的 `convert_from_mxfp4`~~ | — | 已由改动 6 的融合 kernel 解决 |

新增的遗留项：

| 项 | 说明 |
|---|---|
| 改动 5–8 未逐项 A/B | 开关都在，跑一遍 `--no-mxfp4-comm` 和 `LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` 就能拆出四项各自的贡献。优先做改动 7，它最可能是负的 |
| `convert_from_mxfp4_2d` 是纯 PyTorch | FP4 all-gather 的反量化端，物化多个全尺寸中间量（含 int64）。换成 Triton kernel 是让改动 7 转正的前提 |
| 权重缓存的 +4.8 GB | 按层数线性增长。可考虑只缓存预转置形式（DGrad 只用它），或在 micro-batch 结束时按需释放 |
| 算子级数字缺口 | 改动 6 只有逐位一致性，没有可信的算子计时；需要空闲卡跑 `bench_mxfp4_backward_ops.py` |
| 未重新 profile | 上表占比是旧构建的，`aten::copy_` 是否还是第一大头未知 |

融合 cast+shuffle 量化内核的代价评估：三个 GEMM 的 B 操作数由三个不同内核产出
（`convert_to_mxfp4_2d` / `transpose_packed_fp4` / `hadamard_quant_mxfp4`），互不复用；
且布局必须在量化时决定，而那时 autotune 还不知道会不会选 ASM，写成 shuffled 后 plain
后端就消费不了。改动面大、收益 3–4%，优先级应低于上面两行。

---

## 环境注意事项

- **wandb 0.28.1 与 `--num-workers > 0` 冲突**：wandb 的 telemetry import hook 在 fork 出的
  DataLoader worker 中抛 `ForkedError`，会直接打死训练。用 `--num-workers 0`，或在
  `wandb.init()` 之前构造 dataloader。
- `AITER_CONFIG_GEMM_A4W4` 若在 shell 中导出，会让
  `test_mxfp4_configure_wires_tuned_table_and_cache` 失败——该测试正是在验证"尊重已有
  环境变量"。跑测试时用 `env -u AITER_CONFIG_GEMM_A4W4`。
