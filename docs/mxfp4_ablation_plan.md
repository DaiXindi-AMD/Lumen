# MXFP4 性能优化消融实验 —— 修改与运行 plan

**目标：**回答"当前 HEAD 这一整套 MXFP4 优化方案里，每一项历史优化各贡献了多少性能提升"，产出一张对齐 `docs/performance optimization for reference only.md` 第二节形态的 step-time 阶梯表 + 阶梯图。

**环境（冻结，不可改）：**Qwen3-8B · 8×MI350X (gfx950) · Megatron backend · TP1/PP1/CP1/DP8 · seq 8192 · MBS 2 · GBS 128（= 8 个梯度累积 micro-batch）· tail 5/36 层 BF16 · SEED 1234

---

## 一、设计原则（已锁定）

1. **代码基准始终是今天的 HEAD。**不 checkout 任何历史 commit 当实验版本。
2. **correctness fixes 始终保持 HEAD 的当前状态**，所有 arm 一致，不参与消融。判定规则：**pin 在 HEAD 上仍然 live 的修复集**。
3. **纯性能优化尽可能全部 disable/revert**，构造 stripped baseline `S0`。
4. **按这些优化最初引入的时间顺序逐项 enable**，形成 cumulative add-one-in 阶梯。
5. **历史 commit 只用来回答两个问题**：这个优化什么时候加入的、对应哪些代码变化。

> 由此带来的直接后果，需要在报告里写明：本实验**不复现任何历史报告里的数字**。所有 arm 都是 tail-BF16 生效（124 个量化 module，`dfa8618` 已 live），历史上 8/10 那批 144-module 的数字（1513.4 / 1429.0 ms、loss 中心 6.753586）在本实验中不可复现，也不作为对比基准。

---

## 二、新分支

当前 `bench/fp8-latest-upstream` 有未提交修改（`docs/mxfp4_precision_benchmark_report.md`、`examples/scripts/train_pretrain.sh`、`third_party/aiter`），先处理干净再切：

```bash
cd ~/Lumen
git status --short                 # 确认要保留哪些改动
git stash push -u -m "wip before ablation branch"   # 或先 commit
git checkout -b bench/mxfp4-ablation-staircase
git stash pop                      # 如需带上 wip
```

**分支纪律：**本分支只做两件事——(1) 新增 `LUMEN_ABL_*` 消融开关，(2) 新增 launcher 层 env 开关。**所有新开关的默认值必须等于 HEAD 当前行为**，即不加任何 env 时，本分支与 HEAD 逐位一致、step time 无差异。这一条用第七节的 `S23 == HEAD` 对照验证。

---

## 三、correctness fixes：全实验启用，不参与消融

| commit | 修复内容 | 在 HEAD 是否 live | 对本实验的实际影响 |
| --- | --- | --- | --- |
| `dfa8618` | `--first-last-layers-bf16` 在 `--lumen-linear` 路径上生效（`_build_bf16_skip_prefixes`） | live | **有实际影响**：所有 arm 都是 124 个量化 module，不是 144 |
| `47b8841` | MXFP4 tile 越界读 | live | 训练 shape 全部 64 对齐，不触发，等效 no-op |
| `3812b0f` | scale padding unwind | live | 同上，不触发 |
| `0109d35` | `_cached_weight_operands` 引用环泄漏 | live | 保留；泄漏点由 `1be93f8` 引入，早期 arm 本无此代码 |
| `8bcf2d9` | FSDP2 backward 从 `ctx.weight_ref` 重新量化权重 | **仅部分 live** | live 的部分只有 unaligned / kernel-rejected 的 BF16 fallback（`linear.py:2509`），训练 shape 下不走到；主路径的每步重量化已被 `23644ea` 取代，它属于消融项 `A1` 的 legacy 路径 |
| `ead7610` / `656922c` | recipe/dispatch 路由（`--lumen-linear` 不再误跑 FP8 blockwise；`config.recipe` 正确分发 MXFP4） | live | 前置条件，所有 arm 必须有 |
| `8f8ae19` | launcher 不再用通用表遮蔽 model 专用 A4W4 表 | live | **关键**：见第六节 autotune cache 隔离 |

`8bcf2d9` 是这套规则唯一的边界情形：它同时被"correctness fix"和"被后来的性能优化取代"覆盖。按 pin-live-set 规则处理后没有歧义——live 的 fallback 分支保留在所有 arm，被取代的主路径归入消融项。

---

## 四、消融项清单（按引入时间排序）

三类 off 机制：**[E]** 已有 env 开关，零代码改动；**[N]** 需新增 flag，但 legacy 分支在 HEAD 已存在，只需 gate；**[L]** 需新增 flag 且需要重建 legacy 调用路径。

HEAD 上 `convert_to_mxfp4` / `convert_from_mxfp4` / `transpose_packed_fp4` / `hadamard_transform` / `hadamard_quant_mxfp4` / `dual_layout_quant_mxfp4` / `dequant_hadamard_quant_mxfp4` / `dequant_transpose_mxfp4` **全部仍然存在**（`lumen/ops/quantize/ops.py`），所以 **[L]** 类都是重新接线，没有一项需要写 kernel。

| # | 引入日期 | commit | 优化项 | off 机制 | 开关 | 代码位点 | 工作量 | 预期类型 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 07-26 21:24 | `23644ea` | DGrad 复用 forward 已转置的 FP4 权重 | N | `LUMEN_ABL_DGRAD_WEIGHT_REUSE` | `linear.py` mxfp4 backward | 中 | 无损 |
| A2 | 07-26 21:24 | `23644ea` | WGrad 走融合 Hadamard+Quant | N | `LUMEN_ABL_FUSED_HQ_WGRAD` | `linear.py` + `hadamard_quant_mxfp4` | 低 | 需验证位等价 |
| A3 | 07-26 21:24 | `23644ea` | 快速 GEMM dispatch（跳过 fallback 链） | **E** | `LUMEN_FAST_QUANT_DISPATCH=0` | `linear.py:68/1571` | 0 | 无损 |
| A4 | 07-27 21:50 | `d545804` | 跨 micro-batch 权重缓存 | **E** | `LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` | `megatron.py:1290`, `quantize/__init__.py:473` | 0 | **换显存 +4.8 GB** |
| A5 | 07-28 00:28 | `827c941` | 融合 dequant+transpose kernel | N | `LUMEN_ABL_DEQUANT_TRANSPOSE` | `dequant_transpose_mxfp4` 调用点 | 低 | 无损 |
| A6 | 07-29 23:21 | `95512ed` | **preshuffle（shuffled-layout Triton）backend 进入 dispatch** | L | `LUMEN_ABL_MXFP4_SHUF_BACKEND` | `_mxfp4_choose_backend:1526` 的 `shuf_ok` | 低 | 无损 |
| A7 | 07-29 23:21 | `95512ed` | **ASM/CK backend 进入 dispatch** | L | `LUMEN_ABL_MXFP4_ASM_BACKEND` | `_mxfp4_choose_backend:1525` 的 `asm_ok` | 低 | 无损 |
| A8 | 07-29 23:21 | `95512ed` | 实测 per-shape autotune 取代静态字节阈值 | **E** | `LUMEN_MXFP4_AUTOTUNE=0` | `mxfp4_autotune.py:41`, `linear.py:1544` | 0 | 无损 |
| A9 | 07-29 23:24 | `0bb7f8f` | 跳过冗余 scale padding | N | `LUMEN_ABL_SCALE_PAD_SKIP` | `_pad_and_swizzle_mxfp4_scale:1340` | 极低 | 无损 |
| A10 | 07-29 23:24 | `0bb7f8f` | 向量化（int64）wide 权重 shuffle | N | `LUMEN_ABL_VEC_SHUFFLE` | `_shuffle_mxfp4_weight:1093` | 极低 | 无损 |
| A11 | 07-29 23:25 | `7ef406f` | WGrad 不再物化转置（改传 view） | L | `LUMEN_ABL_WGRAD_VIEWS` | `linear.py` wgrad 分支 | 低 | 无损 |
| A12 | 08-03 21:29 | `332a403` | Qwen3-8B 专用 A4W4 tuned 表 | **E** | `AITER_CONFIG_GEMM_A4W4` 指向不含本模型 shape 的表 | `train_pretrain.sh:325` | 0 | 无损 |
| A13 | 08-03 21:29 | `332a403` | 融合 RoPE | launcher | `FUSED_ROPE=0` (新增) | `train_pretrain.sh:207` | 极低 | 无损 |
| A14 | 08-06 00:01:22 | `1be93f8` | 跳过 RTN 下用不到的 philox 采样 | N | `LUMEN_ABL_RTN_SKIP_PHILOX` | `ops.py` 量化核调用 | 低 | 需验证位等价 |
| ~~A15~~ | 08-06 00:01:22 | `1be93f8` | MFMA 实现的 H16 butterfly | **不设 arm，统一设置** | —— | `hadamard_transform:898` | 0 | 见 4.4 |
| A16 | 08-06 00:01:22 | `1be93f8` | 融合 dequant+Hadamard+quant | N | `LUMEN_ABL_FUSED_DHQ` | `dequant_hadamard_quant_mxfp4:1152` | 低 | 无损 |
| A17 | 08-06 00:01:22 | `1be93f8` | 缓存/融合 scale swizzle | N | `LUMEN_ABL_SWIZZLE_CACHE` | `_shuffle_mxfp4_scale`, `_cached_weight_operands:1381` | 低 | 无损 |
| A18 | 08-06 00:01:22 | `1be93f8` | forward 直接产出 WGrad 激活算子 | **N（legacy 分支已 live）** | `LUMEN_ABL_FWD_WGRAD_OPERAND` | `linear.py:1869` 产出、`:2475` legacy | 极低 | 换显存 +6.4 GiB；**备注：dW 少一轮量化，数值更优**（7.1） |
| A19 | 08-06 00:01:22 | `1be93f8` | dual-layout 梯度量化 | N | `LUMEN_ABL_DUAL_LAYOUT` | `linear.py:2461` | 低 | 无损；SR 重抽签（7.1） |
| A20 | 08-06 00:01:22 | `1be93f8` | quantizer 直接产出 shuffled B | N | `LUMEN_ABL_QUANT_EMIT_SHUFFLE` | `linear.py:1158/1437` | 低 | 无损 |
| A21 | 08-06 00:01:37 | `f01c39f` | narrow-N RMSNorm 反向特化 | L | `LUMEN_ABL_NARROW_N_RMSNORM` | `ops/normalization/rmsnorm.py` | 中 | 无损（非 MXFP4 专属） |
| A22 | 08-06 00:01:37 | `f01c39f` | attention QKV strided view | L | `LUMEN_ABL_ATTN_QKV_VIEWS` | `ops/attention/` | 中 | 无损（非 MXFP4 专属） |
| A23 | 08-06 00:01:37 | `f01c39f` | seq-major attention 输出 | L | `LUMEN_ABL_ATTN_SEQ_MAJOR` | `ops/attention/` | 中 | 无损（非 MXFP4 专属） |
| A24 | 08-06 00:01:37 | `f01c39f` | `gc.freeze` | **E** | `LUMEN_GC_FREEZE=0` | `megatron.py:1347` | 0 | 无损 |
| A25 | 08-10 01:14 | `b7459ef` | Lumen 原生 parallel linear 默认开 | launcher | `MXFP4_LUMEN_LINEAR=0` (新增) | `train_pretrain.sh:269` | 极低 | 无损 |

**已判定不设 arm：**`38414e0`（08-07，"stop making backward guess how forward saved its operands"）为纯 refactor —— 统一 `saved_tensors` 的存放顺序、把 fused WGrad 算子 scale 的 swizzle 状态改成如实记录的一个 bool，单文件 9 增 5 删，`tests/ops` 失败集合前后不变。无性能含义，不入阶梯。

### 4.1 `95512ed` 必须拆成三项（A6 / A7 / A8）

`git log -S` 显示 `_gemm_mxfp4_aiter_preshuffle`、`gemm_afp4wfp4_preshuffle`、`_gemm_mxfp4_asm`、`gemm_a4w4`、`_mxfp4_asm_supported`、`_mxfp4_preshuffle_eligible`、`_MXFP4_ASM_MIN_WEIGHT_BYTES`、`_pad_and_swizzle_mxfp4_scale` **全部**首次出现在 `95512ed`。也就是说这一个 commit 同时做了三件独立的事：把 preshuffle backend 接进 dispatch、把 ASM/CK backend 接进 dispatch、以及用实测 autotune 取代静态阈值。把它当成单一 "autotune" arm 会把两个新 GEMM backend 的收益全部错记到 autotune 名下。

### 4.2 现有 env var 不能当 ablation 机制（重要陷阱）

`_mxfp4_choose_backend` 的候选表来自 `_mxfp4_asm_supported` / `_mxfp4_preshuffle_supported`：

```
asm_ok  = _fast_mxfp4_asm_ok        and _mxfp4_asm_supported(a_fp4, w_fp4)
shuf_ok = _fast_mxfp4_preshuffle_ok and _mxfp4_preshuffle_supported(a_fp4, w_fp4)
```

`LUMEN_MXFP4_ASM` / `LUMEN_MXFP4_PRESHUFFLE` 只在 `_mxfp4_asm_eligible` / `_mxfp4_preshuffle_eligible` 里被读，而这两个函数只决定 `static` —— autotune 未命中时的**兜底**选择。因此：

> **`LUMEN_MXFP4_ASM=0` / `LUMEN_MXFP4_PRESHUFFLE=0` 在 autotune 打开时并不会把这两个 backend 从候选表里去掉**，autotune 仍会实测并选中它们。

所以 A6 / A7 的开关必须切在候选表这一层（强制 `shuf_ok` / `asm_ok` 为 False），不能复用现有 env var。这是本 plan 里唯一一处会**静默**产出错误阶梯的地方：用现有 env var 会得到"关掉 ASM 也没变慢"的假结论。现有两个 env var 在本实验中只保留诊断用途（确认某 arm 实际走了哪个 backend）。

### 4.3 A7（ASM backend）的三方依赖 —— 预计在自己的 rung 上接近 0

ASM 能否真正被用到，取决于三件事同时成立：

1. **backend 已接入 dispatch**（A7 本身，07-29）；
2. **该 shape 在 AITER 有 tuned kernel**（`_mxfp4_asm_tuned` → `get_GEMM_config`），这要靠 A12 的 Qwen3-8B tuned 表（08-03）；
3. **选中它**：静态策略的门限是 26 MiB，而代码注释明确写了这个阈值"按另一个模型的 shape 拟合，差 2 MiB 把 Qwen3-8B 的 24 MiB MLP 权重排除在外"——所以只有 A8（实测 autotune）打开后，MLP shape 才会真的选 ASM。

结论：按时间顺序，**A7 在自己的 rung 上的降幅预计接近 0**，真正的跳变会出现在 A12（tuned 表）和 A8（autotune）。报告里必须写明这一点，否则一个 ≈0 的 rung 会被误读成"ASM 没用"。同理，A9（scale padding 跳过）和 A10（wide shuffle）优化的都是 A6/A7 两个 backend 的 prologue，它们在时间上正好落在 A6/A7 之后，依赖自动满足——这也是 4.5 那个顺序错误必须修正的原因。

### 4.4 A15（MFMA H16 butterfly）统一设置，不设 arm

Hadamard 本身属于 **recipe**（`745de8f` 定的 H16 + 确定性），A15 只是它的 MFMA 实现。按"recipe 一律 pin 在当前状态"的规则，H16 butterfly 在**所有 arm 上统一用 HEAD 的 MFMA 实现**，不做消融。相应地不注册 `LUMEN_ABL_MFMA_H16` 开关，`lumen/utils/ablation.py` 里以注释记下这个决定，避免后来者以为漏了一项。

这样也顺带消掉了原先最大的一处不确定性：A15 是否位等价不再影响 arm 列表——它不在阶梯上，其成本计入 `S0` 基线。若日后要单独评估 Hadamard 实现的性能，属于 recipe-cost 报告，不是本消融。

### 4.5 07-29 的时间顺序（已修正）

`95512ed` 23:21:32 → `0bb7f8f` 23:24:07 → `7ef406f` 23:25:01。本 plan 初稿把 `0bb7f8f` / `7ef406f` 排在 `95512ed` 之前，是错的。若按错误顺序跑，A9/A10 的 rung 上还没有任何 backend 调用 `_pad_and_swizzle_mxfp4_scale` 或 wide shuffle，两项降幅会全是 0，而它们的真实收益会被后面的 A6/A7 吸收。

**不纳入消融的项：**

- `3975266` / `e7776ff` FP4 all-gather —— **仅 FSDP2 路径**，本实验是 Megatron track，代码不参与，不设 arm。
- `745de8f` 的 H16 policy / tail-BF16 policy、`40e2691` 的 WGrad SR→RTN —— 这些是 **recipe**，改的是数学不是速度，pin 在当前状态，不设 arm。若要评估它们的性能代价，属于另一份 recipe-cost 报告。
- `LUMEN_MXFP4_ASM` / `LUMEN_MXFP4_PRESHUFFLE` **这两个 env var 本身**不对应任何 commit，是诊断后门，不设 arm。但它们所控制的底层优化——ASM backend 与 preshuffle backend 首次进入 production dispatch——确实改变了正常执行路径，已作为 A7 / A6 列入阶梯（见 4.1–4.3）。**区分"runtime switch"与"它控制的优化"是这里的关键**：前者不是历史优化，后者是。

**launcher 需新增的两个 env 开关**（照 `FP8_LUMEN_LINEAR` 的既有写法）：

```bash
# train_pretrain.sh
[ "${MXFP4_LUMEN_LINEAR:-1}" = "1" ] && QUANT_ARGS+=(--lumen-linear)
[ "${FUSED_ROPE:-1}" = "1" ] && ROPE_ARGS+=(--lumen-fused-rope)
```

---

## 五、实验矩阵（cumulative add-one-in）

25 个 arm（`S0` + A1–A25 去掉不设 arm 的 A15）。`S0` 全关；每个 `Sn` 比上一 arm 多打开一项。`S24` 应与不带任何 env 的 HEAD 一致。

`S0` 的 stripped baseline = 全部 `LUMEN_ABL_*=0` + `LUMEN_FAST_QUANT_DISPATCH=0` + `LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` + `LUMEN_MXFP4_AUTOTUNE=0` + `LUMEN_GC_FREEZE=0` + `AITER_CONFIG_GEMM_A4W4=<不含本模型 shape 的表>` + `FUSED_ROPE=0` + `MXFP4_LUMEN_LINEAR=0`。此时 MXFP4 GEMM 只有 plain Triton 一条路（A6/A7 关掉后候选表里只剩 `plain`）。

| arm | 新增打开的项 | 相对上一 arm 的 env 变化 |
| --- | --- | --- |
| S0 | —（stripped baseline） | 见上 |
| S1 | A1 DGrad 权重复用 | `LUMEN_ABL_DGRAD_WEIGHT_REUSE=1` |
| S2 | A2 融合 H+Q | `LUMEN_ABL_FUSED_HQ_WGRAD=1` |
| S3 | A3 快速 dispatch | `LUMEN_FAST_QUANT_DISPATCH=1` |
| S4 | A4 权重缓存 | 去掉 `LUMEN_MXFP4_DISABLE_WEIGHT_CACHE` |
| S5 | A5 融合 dequant+transpose | `LUMEN_ABL_DEQUANT_TRANSPOSE=1` |
| S6 | A6 **preshuffle backend 进 dispatch** | `LUMEN_ABL_MXFP4_SHUF_BACKEND=1` |
| S7 | A7 **ASM backend 进 dispatch** | `LUMEN_ABL_MXFP4_ASM_BACKEND=1`（预计降幅 ≈0，见 4.3） |
| S8 | A8 实测 autotune | `LUMEN_MXFP4_AUTOTUNE=1` |
| S9 | A9 scale padding 跳过 | `LUMEN_ABL_SCALE_PAD_SKIP=1` |
| S10 | A10 向量化 wide shuffle | `LUMEN_ABL_VEC_SHUFFLE=1` |
| S11 | A11 WGrad view | `LUMEN_ABL_WGRAD_VIEWS=1` |
| S12 | A12 tuned A4W4 表 | `AITER_CONFIG_GEMM_A4W4=<qwen3_8b tuned>` |
| S13 | A13 融合 RoPE | `FUSED_ROPE=1` |
| S14 | A14 philox 跳过 | `LUMEN_ABL_RTN_SKIP_PHILOX=1` |
| S15 | A16 融合 DHQ | `LUMEN_ABL_FUSED_DHQ=1` |
| S16 | A17 swizzle 缓存 | `LUMEN_ABL_SWIZZLE_CACHE=1` |
| S17 | A18 forward 产出 WGrad 算子 | `LUMEN_ABL_FWD_WGRAD_OPERAND=1` |
| S18 | A19 dual-layout 量化 | `LUMEN_ABL_DUAL_LAYOUT=1` |
| S19 | A20 quantizer 产出 shuffle | `LUMEN_ABL_QUANT_EMIT_SHUFFLE=1` |
| S20 | A21 narrow-N RMSNorm | `LUMEN_ABL_NARROW_N_RMSNORM=1` |
| S21 | A22 QKV view | `LUMEN_ABL_ATTN_QKV_VIEWS=1` |
| S22 | A23 seq-major attention | `LUMEN_ABL_ATTN_SEQ_MAJOR=1` |
| S23 | A24 `gc.freeze` | `LUMEN_GC_FREEZE=1` |
| S24 | A25 原生 parallel linear | `MXFP4_LUMEN_LINEAR=1` → 等于 HEAD 默认 |

A15 不占 arm（4.4），所以 S15 起与 A 编号错开一位；A 编号绑定 commit，不重排。

**可选合并（若 arm 数需要压缩到参考报告的 ~14 行）：**A14/A16–A20 合并为"operand-layout 组"（1 个 arm），A21–A23 合并为"shared kernel 组"（1 个 arm）。**A6/A7/A8 不可合并**——把它们并回单一 "autotune" arm 正是 4.1 要避免的错误归因。建议先全跑，报告里再决定合并展示。

---

## 六、运行协议

**入口唯一：**`examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh`（native 或 docker 二选一后固定不变），配置参数一律不改。

```bash
ARM=S0
RESULTS_DIR=~/Lumen/examples/qwen3/results/ablation/${ARM} \
LUMEN_MXFP4_AUTOTUNE_CACHE=~/Lumen/examples/qwen3/results/ablation/${ARM}/autotune.json \
TRAIN_STEPS=60 SEED=1234 \
<该 arm 的 env 差异> \
  bash examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh
```

硬性要求：

1. **每个 arm 独立的 autotune cache 路径。**launcher 里已有注释警告过这一点：用缺表的配置跑过一次，cache 会把每个 shape 都钉死成 "用 Triton"，后续 arm 全部继承。**共用 cache 会静默毁掉整条阶梯。**
2. **60 步，丢弃前 15 步**（warmup + autotune 探测），对剩余 45 步的 per-iteration elapsed time 取**中位数**。
3. **每 arm 重复 2 次**；两次中位数相差 >1.5% 则跑第 3 次并报告 spread。
4. **每 arm 记录**：step time 中位数、`torch.cuda.max_memory_allocated` 峰值、`rocm-smi` 峰值占用、前 50 步 loss 序列、dispatch 实际选中的 backend（每个 GEMM shape）。
5. **同一台机器、同一镜像、同一 `third_party/aiter` 构建**，全程不重新编译。arm 之间只改 env。

---

## 七、验证关卡（跑阶梯之前必须过）

**Phase 0 —— 开关忠实性。**按 `tests/ops/` 约定，为每个新增 `LUMEN_ABL_*` 加一条测试：同一输入下 legacy 路径 vs 优化路径，用 `compute_snr` 比对。

- 声明"无损"的项（A1、A5–A13、A16–A24）要求 **位等价**（`torch.testing.assert_close` 零容差），不满足则该项必须重新归类。A6/A7/A8/A12 尤其要验：代码注释声称三个 MXFP4 backend 位位相同，这条断言是整条 GEMM 阶梯"纯速度"定性的全部依据。
- 标注"需验证位等价"的项（A2、A14）单独跑：A14 若跳过 philox 采样会改变全局 RNG 消耗序列，需确认在 dropout=0 的本配置下不影响任何下游采样。A15 已按 4.4 统一设置，不再需要这项判定。
- **A6/A7 的 gate 需额外一条集成断言**：关闭时 `mxfp4_autotune.record_shape` 记录的 backend 必须只有 `plain`（见 4.2）。
- **`38414e0` 的代码判定已完成**：纯 refactor，不入阶梯（见第四节末）。
- 命名照 `test_<op>_<variant>` / `test_matches_reference`。

### 7.1 Phase 0 已得结论

分支 `bench/mxfp4-ablation-staircase`，开关模块 `lumen/utils/ablation.py`（17 个开关，默认全开，未注册名字直接抛 `KeyError` —— 打错的开关静默什么都不做，会产出一个测错东西的 arm），测试 `tests/ops/test_mxfp4_ablation_switches.py`（16 passed，含 gfx950 实机）。

| 项 | 判定 | 依据 |
| --- | --- | --- |
| A6 / A7 backend gate | **有效** | 实测：开关关闭后候选表只剩 `plain`，且能覆盖 autotune cache 里残留的 `asm`/`shuffled` 决定 |
| A9 scale padding 跳过 | **位等价** | `test_scale_pad_skip_switch_is_bit_exact`，gfx950 实跑，`atol=0 rtol=0` |
| A10 向量化 wide shuffle | **位等价** | `test_vec_shuffle_switch_is_bit_exact`，同上 |
| A17 swizzle 缓存层 | **位等价** | 关闭后每次重建，产出与缓存值逐位相同 |
| A19 dual-layout 梯度量化 | **位等价（RTN）/ SR 重抽签** | 见下 |
| A18 forward 产出 WGrad 算子 | 非位等价，**dW 数值更优** | 见下 |

**A19：数学一致，只有 SR 抽签不同。**实测（gfx950，同一 philox seed/offset）：`use_sr=False` 时 fused 与两次调用的 data 与 scale **全部逐位相同**；`use_sr=True` 时 **scale 仍逐位相同**，只有约 54% 的已舍入尾数不同——融合核按自己的 tiling 把 philox counter 映射到元素，两个独立核的映射不同。对精确旋转结果取 SNR，三个随机种子下两种形式相差均在 ±0.05 dB 内（约 16.0 dB），**没有哪一种舍得更好**。

结论：A19 是纯速度优化，其唯一数值足迹是一次无偏重抽签。backward 的梯度量化 SR 是开着的，所以 A19 这一级 loss 会有"换随机种子"量级的扰动，不代表质量变化——报告里按此措辞。

**A18：备注为数值收益。**该算子比 backward 重建的少一轮量化，因此 dW 无法逐位相同，且**更接近 BF16 参考**（`tests/ops/test_mxfp4_fwd_wgrad_operand.py` 直接断言 fused 的 SNR 不低于重建路径）。这是一项附带的数值**改善**，作为该优化的备注写进报告即可，不需要当作问题处理。只需在第八节的 loss 断言里把这一级列为预期会动、且方向应当变好。

**A18 与 A19 共用一个 kernel。**`dual_layout_quant_mxfp4` 同时服务 forward 的 WGrad 激活算子（A18）和 backward 的梯度量化（A19），两项各自 gate 不同调用点。测试因此断言"融合调用次数恰好减一、两次调用形式恰好加一"，而不是断言归零——否则会误判开关无效。

**Phase 1 —— `S24 == HEAD`。**S24 的 step time 与 loss 必须与不带任何 env 的 HEAD 在噪声内一致。不一致说明 gate 写错或有开关默认值不等于 HEAD 行为。

**Phase 2 —— `S0` 冒烟。**确认 stripped baseline 能跑完 60 步不 OOM、不 NaN。S0 是历史上从未存在过的组合，这一步是纯粹的新配置风险。若 S0 慢到不可接受，把该 arm 的步数降到 30 并在报告中标注。

**Phase 3 —— 全阶梯。**25 arm × 2 次。

**Phase 4 —— 出报告。**对齐参考文档结构：结论速览表 / 阶梯表（单步降幅 + 累计降幅 + 类型）/ 每步细节 / 瓶颈构成变化 / 试过但没采用。

---

## 八、这个设计自带的两个额外产出

1. **loss 中性证明。**每个 arm 都记录前 50 步 loss。除 A18（已知改变 dW，见 7.1）以及 Phase 0 判定为非位等价的项之外，**其余整条阶梯的 loss 曲线应当完全重合**。任何一级 rung 让 loss 动了而事先没被 Phase 0 标记，就直接证明该项不是无损优化——这比事后论证强得多，也正好回答"哪些优化是无损的"。
2. **换显存项的定价。**逐 arm 的峰值显存曲线会把 A4（权重缓存）和 A18（forward 产出 WGrad 算子）这两处速度换显存的台阶画出来，给出"每省 1 ms 花多少 GB"的直接读数。
3. **backend 归因链。**每 arm 记录 dispatch 实际选中的 backend，A6→A7→A8→A12 四级会把"ASM 到底是被什么解锁的"完整画出来——这是现有报告里一直缺的一环。

---

## 九、风险与缓解

| 风险 | 缓解 |
| --- | --- |
| legacy 路径不忠实（gate 出来的旧行为并不等于当年的旧行为） | Phase 0 位等价测试；对 **[L]** 类逐个 diff 对应 commit 的旧代码，在测试注释里写明依据的 commit |
| 累计归因不可交换：重叠收益全归时间上更早的那一项（如权重缓存在原生 linear 之前，会吃掉大部分收益） | 报告中明确写"按引入时间的累计归因，非 Shapley 贡献"；对怀疑重叠的对（A4/A25、A5/A16、A17/A20、A6/A7/A8/A12）补一组 leave-one-out 作为脚注 |
| **用现有 `LUMEN_MXFP4_ASM` / `_PRESHUFFLE` 当开关**，autotune 仍从候选表里选中它们 → 得到"关掉也没变慢"的假结论 | A6/A7 的 gate 必须切在 `_mxfp4_choose_backend` 的 `asm_ok` / `shuf_ok` 上；Phase 0 增加一条断言：A6/A7 关闭时 `mxfp4_autotune.record_shape` 记录的 backend 只能是 `plain` |
| A7 在自己 rung 上降幅 ≈0 被误读为"ASM 无用" | 报告中把 A7 / A12 / A8 三项作为一组解释（4.3），并给出"三者齐备后 ASM 的实际收益"作为组内小结 |
| autotune cache 跨 arm 污染 | 每 arm 独立 cache 路径 + 每 arm 结束后校验 cache 里记录的 backend 与日志一致 |
| A19 这一级 loss 会动（SR 重抽签），被误读成"该优化有损" | 报告中预先声明：A19 的 scale 逐位相同、SNR 无差异，扰动等价于换随机种子（7.1） |
| S0 组合从未存在，可能 OOM 或触发未测过的路径 | Phase 2 冒烟；必要时把 A4 权重缓存提前到 S0（作为 baseline 的一部分）并在报告中标注基线定义 |
| 25 arm × 2 次的机时超预算 | 先跑 S0 / S6 / S8 / S12 / S19 / S24 六个关键点确认阶梯形状，再补齐中间 arm |
| 时间顺序搞错导致某项在自己 rung 上恒为 0（初稿已犯过一次，见 4.4） | arm 顺序一律以 `git log -1 --date=iso` 的精确时间戳为准，不用日期；每个 arm 上线前确认它依赖的代码在前序 arm 已激活 |

---

## 十、执行顺序

1. 切分支 `bench/mxfp4-ablation-staircase`（第二节）。
2. 加 launcher 两个 env 开关（`MXFP4_LUMEN_LINEAR`、`FUSED_ROPE`）+ 17 个 `LUMEN_ABL_*` gate，默认值全部等于 HEAD 行为。其中 A6/A7 切在 `_mxfp4_choose_backend` 的候选表层（见 4.2），不要复用 `LUMEN_MXFP4_ASM` / `_PRESHUFFLE`。已落地 8 个：A6、A7、A9、A10、A17、A18、A19、A20。
3. 写 Phase 0 测试，跑通位等价判定，据结果冻结最终 arm 列表。
4. Phase 1 / Phase 2 关卡。
5. Phase 3 全阶梯，结果落 `examples/qwen3/results/ablation/`。
6. Phase 4 出报告。
