# AITER #5542 合并后的 Lumen 集成计划

**日期**：2026-09-20

**Lumen 基线**：`codex/aiter-kernel-migration @ 0f2cc90398ad05379d4eff125cfd365aaa71a7ce`

**当前 AITER pin**：`third_party/aiter @ f4f95c3d548b19e1824070b9daf075908a410a9b`

**目标**：AITER #5542 合并后，让 Lumen 只通过 upstream AITER 公共 API 使用 fused SiLU-and-multiply forward/backward，并移除 fork-only API 与 vendor workaround。

**提交策略**：AITER 继续逐算子 PR；Lumen 的 API 适配、测试、vendor 清理和 submodule bump 合为一个提交。

> 本文是合并后的执行清单，不要求把 PR 临时 head 固定进 Lumen。文中的 AITER revision 必须替换为最终进入 `ROCm/aiter:main` 的 commit。

## 1. 最终状态

Lumen 保留自己的稳定包装函数，但内部改为调用 upstream AITER：

| Lumen 接口 | AITER 公共接口 | 输入/输出 |
| --- | --- | --- |
| `fused_swiglu(y)` | `aiter.ops.triton.activation.fused_silu_mul` | `y[..., 2N] = [gate, up]`，输出 `[..., N]` |
| `fused_swiglu_backward(grad_output, y, out=None)` | `aiter.ops.triton.activation.silu_and_mul_backward` | `grad_output[..., N]` + `y[..., 2N]`，输出 `[..., 2N]` |

不再使用 fork 中的临时名称：

```python
swiglu_fwd
swiglu_bwd
```

生产调用链应为：

```text
Megatron fused_bias_swiglu
  -> Lumen import-time patch
  -> lumen.ops.fused_swiglu
  -> aiter.ops.triton.activation public API
  -> AITER Triton kernel
```

这次迁移不在 Lumen 新增或复制任何 GPU kernel；kernel、repr、unit test、benchmark 和 tuning config 继续由 AITER 维护。

#5542 最新审查版本已把 backward 放进现有 activation family，与 `fused_silu_mul` 共用模块和 activation helper。Lumen 不应再要求 AITER 提供 `swiglu_*` compatibility alias，也不应在自己仓库实现相同 kernel。

改动总览：

| 文件 | 合并后动作 |
| --- | --- |
| `.gitmodules`、`third_party/aiter` | 所有依赖 PR 合并后切到 `ROCm/aiter:main` 并固定最终 SHA |
| `lumen/ops/fused_swiglu.py` | 从旧 `swiglu_*` 名称迁到两个公共 activation API，并透传 `out=` |
| `lumen/ops/dispatch.py` | 成对探测正式 forward/backward symbol |
| `lumen/patches/runtime/megatron_import.py` | FP8 chunked backward 在 fused 模式下直接写预分配 output |
| `examples/scripts/train_pretrain.sh` | 更新 API preflight；显式启用 fusion 时缺 API 直接失败 |
| `examples/qwen3/Dockerfile`、`third_party/aiter_vendor/` | 保持 vendor 单文件 overlay 已删除，并从目标 AITER pin 重建镜像 |
| `tests/ops/`、`tests/patches/` | 更新 API mock，增加 probe、installer、`out=` 和 fallback 覆盖 |
| `lumen/ops/quantize/linear.py` | 不因 #5542 修改；保留原有 MXFP4 RHT/H16 链路 |

## 2. 合并前置条件

### 2.1 不要 pin PR head

截至本文日期，#5542 仍为 open，当前 head 为 `9920a8ab9b56dab94853906dfa0a0fd18a54208d`。这个 SHA 只用于审查，不能作为 Lumen 的最终依赖。

合并后先取得最终 upstream revision，并验证公共 API：

```bash
gh pr view 5542 --repo ROCm/aiter \
  --json state,mergeCommit,url

git -C third_party/aiter fetch origin main
git -C third_party/aiter merge-base --is-ancestor \
  <PR_5542_MERGE_SHA> <TARGET_AITER_SHA>

git -C third_party/aiter show <TARGET_AITER_SHA>:aiter/ops/triton/activation.py \
  | rg 'def (fused_silu_mul|silu_and_mul_backward)'
```

验收条件：

- #5542 状态为 `MERGED`。
- `TARGET_AITER_SHA` 是 #5542 merge commit 的后代。
- 两个函数都能从 `aiter.ops.triton.activation` 导入。
- 最终 AITER 代码仍含 backward kernel repr、单测、benchmark，以及 gfx942/gfx950 的 tuning config。

### 2.2 何时切换到 `ROCm/aiter:main`

当前 Lumen migration commit 不只依赖 #5542，还依赖以下 upstream PR：

| PR | Lumen 所需能力 |
| --- | --- |
| [#5531](https://github.com/ROCm/aiter/pull/5531) | stochastic MXFP4 quantization |
| [#5538](https://github.com/ROCm/aiter/pull/5538) | packed FP4 logical transpose |
| [#5542](https://github.com/ROCm/aiter/pull/5542) | fused SiLU-and-multiply backward |
| [#5548](https://github.com/ROCm/aiter/pull/5548) | 32×32 block-scaled MXFP4 quantization |

只有 Lumen 实际引用的这些 API 都已进入 upstream main，才把 `.gitmodules` 改为：

```ini
[submodule "third_party/aiter"]
    path = third_party/aiter
    url = https://github.com/ROCm/aiter.git
    branch = main
```

然后把 gitlink 固定到同时包含全部依赖的一个 upstream commit。不要使用 `git submodule update --remote` 作为提交结果；Lumen 必须记录可复现的精确 SHA。

如果只有 #5542 先合并，其余 PR 尚未合并，可以先在 Lumen 临时分支完成 API 适配和测试，但不要把最终 submodule 从集成 fork 切到 upstream main，也不要重新复制 #5542 kernel 到 `third_party/aiter_vendor/`。

## 3. Lumen 逐文件修改

### 3.1 `lumen/ops/fused_swiglu.py`

保留 Lumen 函数名，避免 Megatron patch 和调用方跟随 AITER 改名。只替换内部 import，并给 backward 增加 `out` 透传：

```python
def fused_swiglu(y: torch.Tensor) -> torch.Tensor:
    from aiter.ops.triton.activation import fused_silu_mul

    return fused_silu_mul(y)


def fused_swiglu_backward(
    grad_output: torch.Tensor,
    y: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    from aiter.ops.triton.activation import silu_and_mul_backward

    return silu_and_mul_backward(grad_output, y, out=out)
```

AITER #5542 的 backward contract：

- `y` 必须 contiguous，最后一维为正偶数，布局是 `[gate, up]`。
- 支持 FP16、BF16、FP32。
- `grad_output` 与 `y` 同 dtype/device；允许 strided tensor。
- 可选 `out` 必须与 `y` shape/dtype/device 一致且 contiguous。
- 当前 tuning config 仅覆盖 gfx942 和 gfx950。

### 3.2 `lumen/ops/dispatch.py`

`_probe_aiter_swiglu()` 必须成对探测正式 forward/backward API：

```python
from aiter.ops.triton.activation import (  # noqa: F401
    fused_silu_mul as _forward,
    silu_and_mul_backward as _backward,
)
```

不能只探测 forward。否则旧版 AITER 会让安装器成功替换 forward，却在第一次 backward 时失败。

该 probe 只证明符号可导入，不证明当前 GPU 有 tuning config。`LUMEN_FUSED_SWIGLU=1` 的支持范围应明确为 gfx942/gfx950；启动预检或集成测试必须在 kernel 真正执行后再判定可用，不能用 import success 代替运行验证。

### 3.3 `lumen/patches/runtime/megatron_import.py`

`install_fused_swiglu_triton()` 仍同时替换 Megatron 的 `swiglu` 和 `swiglu_back`，并保留现有幂等标记。不要只替换 backward。

FP8 input-store 的 chunked backward 已经预分配完整 `result`。#5542 支持 `out=` 后，fused patch 已安装时应直接写入该 slice，避免每个 chunk 先分配临时 gradient 再 copy：

```python
out_chunk = result[s:e]
if getattr(_swiglu_mod, "_lumen_triton_swiglu_patched", False):
    _swiglu_mod.swiglu_back(grad_chunk, inp_chunk, out=out_chunk)
else:
    out_chunk.copy_(_swiglu_mod.swiglu_back(grad_chunk, inp_chunk))
```

这里的条件分支是必要的：Megatron 原始 `swiglu_back` 只接受两个参数。`result[s:e]` 是完整行切片，必须用测试确认它满足 AITER 对 contiguous `out` 的要求。

保持当前注册顺序：先安装 `swiglu_fp8`，再安装 `fused_swiglu_triton`。这样 `_PatchedSwiGLUFunction.backward()` 在执行时能看到最终替换后的 `_swiglu_mod.swiglu_back`。

### 3.4 `examples/scripts/train_pretrain.sh`

将旧 API 预检改为：

```bash
python -c \
  "from aiter.ops.triton.activation import fused_silu_mul, silu_and_mul_backward"
```

当 `LUMEN_FUSED_SWIGLU=1` 时，缺少任一符号都应打印实际 `aiter.__file__` 并退出，而不是 warning 后让训练静默使用 Megatron fallback。这样性能结果不会被误标为 fused SwiGLU 结果。

建议预检同时输出：

```bash
python -c "import aiter; print(aiter.__file__)"
```

本机当前 import 指向 `/home/xdai/aiter/aiter/__init__.py`，并不等于 Lumen 的 `third_party/aiter`。验证时必须检查训练进程或容器中的真实 import 路径。

### 3.5 `third_party/aiter_vendor/` 与 Qwen3 image

最终 upstream AITER 已提供公共实现后，以下临时 vendor 文件必须保持删除：

```text
third_party/aiter_vendor/aiter/ops/triton/activation.py
third_party/aiter_vendor/aiter/ops/triton/_triton_kernels/activation.py
```

同时不得在 `examples/qwen3/Dockerfile` 中重新覆盖 `/opt/aiter` 的这两个单文件。#5542 本身是 Triton Python + JSON config 变更，但最终 pin 还会包含其他 upstream 变化；正式镜像应从新 submodule revision 重建基础镜像，使 `/opt/aiter` 的 Python 和编译扩展来自同一 revision。只有确认两个 revision 之间没有相关编译扩展变化时，才可把“仅复制 Python package”作为临时验证方式。

submodule bump 后必须重建 Qwen3 image；只 bind-mount Lumen 源码不会更新镜像中的 `/opt/aiter`：

```bash
docker build -f Dockerfile -t lumen/tests:latest .
bash examples/qwen3/build_mxfp4_lumen_image.sh

python -c "import aiter; print(aiter.__file__)"
python -c \
  "from aiter.ops.triton.activation import fused_silu_mul, silu_and_mul_backward"
```

后两条命令应在实际训练容器内执行。

### 3.6 测试文件

更新 `tests/ops/test_fused_swiglu_api.py`：

- fake activation 暴露 `fused_silu_mul` 和 `silu_and_mul_backward`。
- 断言 forward/backward 分别调用这两个正式 API。
- 增加 `out=` identity 与原样透传测试。
- 保留 non-contiguous `grad_output` 覆盖。

扩充 `tests/ops/test_dispatch.py`：

- 两个符号都存在时 probe 为 `True`。
- 缺少任一符号时 probe 为 `False`。
- 每个 case 清理 `_probe_aiter_swiglu.cache_clear()`，避免缓存污染。

扩充 `tests/patches/test_megatron_import_patches.py`：

- 环境变量关闭时不替换。
- probe 失败时保留 Megatron 原函数并输出一次明确 warning。
- probe 成功时 forward/backward 成对替换，重复安装幂等。
- FP8 chunked path 在 fused 模式下把 `result[s:e]` 作为 `out` 传入。
- 非 fused fallback 仍调用 Megatron 两参数 backward。

### 3.7 注释与文档清理

本次触碰到的 Python 和 shell 代码只保留说明 contract 或非显然 fallback 原因的短注释。性能数字放在 benchmark/验证记录中；不要把长篇实现过程、审查对话或生成式说明写入源码。

## 4. Megatron 与 FSDP 的实际影响

### 4.1 Megatron 是本次直接消费者

Qwen3 MXFP4 launcher 已设置 `LUMEN_FUSED_SWIGLU=1`。Megatron import patch 会把 `megatron.core.fusions.fused_bias_swiglu` 的 forward/backward 指向 Lumen wrapper，因此 #5542 合并后会直接进入生产训练链路。

训练“跑完”不足以证明新 kernel 被使用。profile 或 trace 中必须出现类似：

```text
_silu_and_mul_backward_kernel_BLOCK_M_*_BLOCK_N_*
```

同时日志中不能出现 “AITER SwiGLU kernels not available” 或 fallback 提示。

### 4.2 FSDP/Hugging Face 当前不消费 #5542

Qwen3 FSDP 路径使用 Hugging Face `Qwen3MLP`，`gate_proj` 和 `up_proj` 分别产生 tensor，没有 Megatron 的 packed `[gate, up]` 输入，也不会安装 `fused_bias_swiglu` monkey patch。因此：

- 本次 Lumen 改动不会让 FSDP 自动调用 #5542。
- 不要为了复用现有 API 在 FSDP 中额外 `torch.cat((gate, up), dim=-1)`；这个大 tensor 的分配和带宽很可能抵消融合收益。
- FSDP2 MXFP4 训练只作为 AITER dependency bump 的 negative-control regression，不作为 #5542 kernel coverage。
- 如果未来要让 FSDP 使用该 kernel，应另立任务，优先设计 fused gate/up projection 输出布局或在 AITER 增加真正适合两个输入的公共 op。

## 5. RHT/H16 必须保持不变

#5542 只替换 SwiGLU elementwise forward/backward，不得改变 MXFP4 quantization、transpose、GEMM 或 RHT 的语义和顺序。

以下生产路径保持原样：

- `lumen/ops/quantize/linear.py` 中 forward-side 的 `dual_layout_quant_mxfp4(...)`，第二份 layout 继续包含原有 blockwise H16 RHT。
- MXFP4 backward 中 gradient 的 `dual_layout_quant_mxfp4(...)`，继续同时生成 DGrad row layout 和带 RHT 的 WGrad transposed layout。
- activation WGrad operand 的 `dequant_hadamard_quant_mxfp4(...)`，继续执行 dequant → transpose → H16 RHT → requant。
- 原本使用 stochastic rounding 的 gradient 路径继续使用 stochastic rounding；activation 路径仍按现有决定使用 round-to-nearest。

禁止把上述带 RHT 的生产 API 退化为普通 `convert_to_mxfp4(...)`，也不要因为接入 #5542 修改 RHT sign、group size、scale swizzle 或 packed-data shuffle。

## 6. 验证顺序

### 6.1 静态与 CPU integration tests

```bash
pytest -q tests/ops/test_fused_swiglu_api.py
pytest -q tests/ops/test_dispatch.py
pytest -q tests/patches/test_megatron_import_patches.py

ruff check \
  lumen/ops/fused_swiglu.py \
  lumen/ops/dispatch.py \
  lumen/patches/runtime/megatron_import.py \
  tests/ops/test_fused_swiglu_api.py \
  tests/ops/test_dispatch.py \
  tests/patches/test_megatron_import_patches.py

git diff --check
```

### 6.2 GPU 数值测试

在 gfx942 和 gfx950 上通过 AITER 公共 wrapper 调用，至少覆盖：

| Case | 目的 |
| --- | --- |
| gate width `1` | 防止最小合法宽度编译回归 |
| BF16 input last dim `24576` / grad last dim `12288` | Qwen3-8B TP1 生产 shape |
| TP-sharded width | 验证 tensor-parallel local shape |
| strided `grad_output` | 验证 #5542 支持的 stride contract |
| explicit contiguous `out` | 验证 FP8 chunked backward 的零临时分配路径 |
| empty leading rows | 验证空 batch 行为 |

结果与 PyTorch autograd reference 比较，使用 Lumen 的 `compute_snr` / `check_close` 或有依据的 dtype tolerance。不能只检查 shape 或“没有抛异常”。

还要对 forward 做 BF16/FP16 回归：upstream `fused_silu_mul` 当前在乘法前把 SiLU 结果 cast 回输入 dtype，而旧 vendored forward 保持 FP32 到乘法结束。若 loss/gradient 超出既有噪声范围，应先在 AITER 修正并补测试，不要在 Lumen 恢复一份私有 kernel。

### 6.3 Megatron Qwen3 MXFP4

先运行 2–5 step smoke：

```bash
TRAIN_STEPS=2 \
LUMEN_SKIP_BACKEND_SYNC=0 \
LAUNCH=native \
bash examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh
```

验收：

- forward/backward、optimizer step 完成，无 NaN/Inf。
- 实际 import 的 AITER 是目标 upstream revision。
- trace 中出现 #5542 backward kernel。
- 没有 silent fallback。
- RHT-bearing MXFP4 路径仍被命中。

然后固定 commit、数据、seed、并行配置、RHT sign 和所有 MXFP4 参数，做 `LUMEN_FUSED_SWIGLU=0` 与 `1` 的 50-step 配对 A/B；正式验收可扩到 200 step。比较 loss、grad norm、峰值显存和 step-time。

`examples/qwen3/configs/mxfp4_loss_baseline.json` 的已有 200-step baseline 使用 GBS=32，而 launcher 默认值可能不同。只有精确复现相同 GBS、数据、seed、LR、tail-BF16 和 RHT 配置时才能直接使用该 JSON 的阈值；否则以同环境、同 commit 的 ON/OFF 配对结果为准。

### 6.4 FSDP2 regression

```bash
MODEL_PATH=<model-path> \
TRAIN_DATA_PATH=<data-path> \
TRAIN_STEPS=2 \
bash examples/qwen3/run_qwen3_fsdp_mxfp4_pretrain.sh
```

这里只验证 submodule bump 没有破坏 FSDP2 MXFP4：训练步完成、loss/grad finite、RHT 路径未变化。该 profile 不应被要求出现 #5542 kernel。

## 7. Landing、回滚与完成标准

### 7.1 Landing

最终只提交一个 Lumen integration commit，包含：

- `.gitmodules` 切到 `ROCm/aiter`（仅在全部依赖已 upstream 时）。
- `third_party/aiter` pin 到包含所有所需 PR 的 upstream SHA。
- `swiglu_fwd/swiglu_bwd` 到正式 API 的适配。
- probe、preflight、FP8 chunked `out=` 和对应测试。
- vendor 文件及 Docker 单文件 overlay 的删除或保持删除。
- 本文档。

建议执行顺序：

```bash
# 1. 在干净的 Lumen integration worktree 中确认没有意外改动。
git status --short

# 2. 修改 .gitmodules 后同步 URL，并初始化 submodule。
git submodule sync -- third_party/aiter
git submodule update --init --recursive third_party/aiter

# 3. 固定到已验证包含全部依赖的 upstream commit。
git -C third_party/aiter fetch origin main
git -C third_party/aiter checkout --detach <TARGET_AITER_SHA>

# 4. 完成 Lumen API/test/container 改动，再统一验证和提交。
```

如果 `third_party/aiter` 已有本地修改，先停止并保存或转移这些改动；不要用 reset/checkout 覆盖。完成后使用 `git diff --submodule=log` 审核实际 pin 变化。

提交前检查：

```bash
rg -n 'swiglu_fwd|swiglu_bwd|third_party/aiter_vendor' \
  lumen examples tests .gitmodules

git diff --submodule=log --check
git status --short
```

除明确的历史说明外，第一条命令不应在生产代码、launcher 或测试中找到旧 API/vendor 依赖。

### 7.2 Rollback

运行时首选回滚：

```bash
LUMEN_FUSED_SWIGLU=0
```

这只恢复 Megatron 原始 SwiGLU 实现，不触碰 MXFP4 quantization 和 RHT。

如果 upstream AITER pin 同时导致其他 MXFP4 API 回归，则 revert 整个 Lumen integration commit，回到已知可用的 AITER pin；不要只恢复一份 vendored activation kernel，因为那会重新制造与 upstream 漂移的重复实现。

### 7.3 完成标准

- Lumen 不再引用 `swiglu_fwd`、`swiglu_bwd` 或 vendored activation kernel。
- Lumen 只调用 AITER public wrapper，不导入 `_triton_kernels`。
- mock/probe/patch integration tests 全部通过。
- gfx942、gfx950 的目标 GPU 数值测试通过。
- Megatron Qwen3 MXFP4 profile 确认实际运行 #5542 backward kernel。
- Megatron paired A/B 的 loss 与 grad norm 对齐，性能无不可接受回退。
- FSDP2 MXFP4 regression 通过，但不宣称它已使用 #5542。
- 原有 dual-layout 与 dequant→transpose→H16 RHT→requant 路径保持不变。
