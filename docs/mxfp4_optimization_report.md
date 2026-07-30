# MXFP4 训练性能优化记录

本文记录一轮 MXFP4 训练路径优化的全过程：每项改动的动机、实测数据、核心代码，以及
最后的收敛与性能验证。硬件为 MI350X（gfx950），模型为 Qwen3-8B / Qwen3-0.6B。

后端选择机制本身（autotune、调表流程）见
[`mxfp4_gemm_backend_selection.md`](mxfp4_gemm_backend_selection.md)，本文侧重优化项与量化数据。

---

## 结论摘要

| 改动 | GEMM 层面 | 端到端（Qwen3-8B） |
|---|---|---|
| 1. autotune 后端选择 | Qwen3-8B 0.90x → 1.02x | +0.4%，并消除 −2.6% 倒退 |
| 2. 跳过冗余 scale padding | 前处理 92 → 60 µs | 含在下行 |
| 3. 向量化 weight shuffle | Qwen3-8B 1.02x → 1.12x | 含在上行的 +0.4% 内 |
| 4. **wgrad 去掉转置物化** | — | **+7.8%** |

最终：Qwen3-8B 单卡步时间 **1097.8 ms → 1018.4 ms（1.078x）**，全部改动逐位一致。

**必须同时记住的两件事：**

1. 改动 1–3 我是**按假设**优化 GEMM 的，端到端只值 0.4%；改动 4 是 **profile 指出来**的，
   值 7.8%。后者代码量最小，收益是前者的 20 倍。
2. 即便如此，**MXFP4 端到端仍比 BF16 慢 16%**（1075 vs 926 ms/步），显存还多 5%。
   目前 MXFP4 在本配置下不是性能优化手段。

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

```python
# lumen/ops/quantize/linear.py，wgrad 路径
                    # Left as views: hadamard_quant_mxfp4 indexes through both
                    # strides, so it can read the transpose directly. Materialising
                    # these two was ~85 ms of every Qwen3-8B step, more than all
                    # the MXFP4 GEMMs put together.
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

---

## 最终验证：8 卡 2000 步训练

Qwen3-8B，8× MI350X FSDP full_shard，c4 streaming，seq 2048 × mbs 4 × 8 卡 =
65536 tokens/步，lr 3e-4 warmup 200，MXFP4 与 BF16 除 `--mode` 外配置完全一致。

| | val@500 | val@1000 | val@1500 | val@2000 | 中位步时间 | 显存/卡 |
|---|---|---|---|---|---|---|
| MXFP4 | 5.3977 | 4.6242 | 4.2641 | 4.2336 | 1075.1 ms | 16.1 GB |
| BF16 | 5.3461 | 4.5637 | 4.2152 | 4.1852 | 925.7 ms | 15.3 GB |
| 差 | +0.052 | +0.061 | +0.049 | +0.048 | | |

**收敛健康**：差距稳定在 ~0.05 nats 且不随训练扩大，说明量化没有累积性损害。

**但 MXFP4 比 BF16 慢 16%，显存还多 5%。** 经过本轮全部优化后，MXFP4 在这个配置下
依然不是性能优化手段。要让它真正划算，还需要解决下面列出的开销。

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

---

## 测试

```
pytest tests/ops/test_quantize.py -k "mxfp4 or hadamard" -q   # 47 passed
pytest benchmarks/bench_mxfp4_gemm.py -v -s                   # 5 passed
pytest benchmarks/bench_mxfp4_gemm_models.py -v -s            # 3 passed
```

本轮新增的关键测试：

| 测试 | 守住什么 |
|---|---|
| `test_mxfp4_weight_shuffle_matches_aiter` | int64 视图 shuffle 与 AITER 逐字节一致（4 个形状） |
| `test_mxfp4_weight_shuffle_falls_back_when_wide_view_invalid` | gfx1250 / 非连续 / 小张量正确退回 |
| `test_mxfp4_aligned_scale_swizzle_does_not_alias_input` | 跳过 pad 后不返回调用方 scale 的视图 |
| `test_hadamard_quant_reads_transpose_without_materialising` | 转置视图量化与物化结果逐位一致 |
| `test_mxfp4_backends_are_interchangeable` | 三个后端逐位一致（autotune 的正确性前提） |

完整跑 `test_quantize.py` 会在一个 FP8/torchao 测试处 abort，该问题早于本轮改动
（stash 掉全部改动后同样复现）。

---

## 已知遗留

按当前 profile 的剩余大头排序：

| 项 | 每步 | 说明 |
|---|---|---|
| `aten::copy_` 剩余部分 | ~175 ms | FSDP 扁平参数 29 ms、`[4,2048,4096]` 23 ms 等尚未归因 |
| `aten::mul` | 120 ms | 疑似反量化/缩放，未拆解 |
| MXFP4 GEMM 前处理 | ~60 µs/次 | 融合 cast+shuffle 可再拿 GEMM 层面 1.3x，端到端约 3–4% |
| wgrad 的 `convert_from_mxfp4` | 未单独计量 | 反向把激活反量化回 BF16 再重量化，可能可以省掉 |

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
