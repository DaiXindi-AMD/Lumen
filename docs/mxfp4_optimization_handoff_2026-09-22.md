# Lumen Qwen3-8B MXFP4 训练优化交接

更新时间：2026-09-22（US/Central）

## 2026-09-28 补充：当前阻塞与下一次 fresh campaign

当前本机运行树重新核验为：

```text
Lumen: /home/xdai/Lumen
branch: dev/mxfp4
HEAD: 6b9aee1569247eca20937c14319ba6adcd69e0cb
tracked diff SHA256: 346860766945762e587bff65b1cbf9486e73ab0f5bd884a86f7fb9fe9527021a

AITER runtime: /home/xdai/aiter
branch: bench/ecfff3f-lumen
HEAD: e35bb17f4f815903bf73598facedbb321e15af28
tracked diff SHA256: a5476078484e426e951c2a5e62233cb20f26c103f93656e02a690e294e9f9aa0
```

正式 runner 中 BF16/MXFP4 两臂均开启 `--aiter-attn`；当前没有
`gc.freeze()` 或对应环境开关。Megatron 入口使用 `aiter_csrc` attention，FSDP
convenience launcher 默认不启用 `--aiter-attn`，但本轮 formal harness 明确启用。

新的 projection-guard campaign 已在 CPU 侧修复并冻结，备份位于本分支：

```text
recovery/projection_guard_harness_2026-09-28/
```

正式实验顺序是：

```text
tail1_a1 -> guard_o_down_b1 -> guard_down_c -> guard_o_down_b2 -> tail1_a2
```

每臂 50 step，统计 step 11--50。A 是完整最后一层 BF16；B 只保护最后一层
`o_proj + down_proj`；C 只保护最后一层 `down_proj`；三者均保留 BF16
`lm_head`。harness 已通过 `bash -n`、analyzer self-test、9/9 unittest、Ruff
lint/format 和从 `/tmp` 发起的 no-GPU dry-run。它没有产生任何新的速度或 loss
结果，不能作为性能证据。

2026-09-28 的宿主只读检查显示八张 MI350X 仍被 root-owned vLLM 占用，每卡约
94--95% VRAM；没有终止这些进程，也没有启动训练。新机器或 GPU 空闲后，先按
recovery README 恢复精确路径与源码，再运行 `dry-run`，随后只运行 `phase1`。
必须先审计新 smoke/cache 和 A1/B1/C，才允许执行 `phase2`。

AITER 当前 dirty WIP 的更新 portable backup 已远端核验：

```text
repository: https://github.com/DaiXindi-AMD/aiter.git
branch: backup/2026-09-28/ecfff3f-lumen-portable-wip
commit: 5d7178517e5f3947b7496fa5994696aa2fb9c50d
```

该分支携带基于精确 `e35bb17f4` 的 Git bundle、17 文件 hash manifest、patch
和恢复说明。不要把旧的 `35e796da...` 整提交 cherry-pick 到 `e35bb17f4`；两者
历史并不线性。

`split/gc-freeze-hook` 的 `8badcb9e...` 只包装 Megatron `train_step`，不会作用
于当前 Qwen FSDP2 训练循环；它还与现有 weight-cache hook 文件发生冲突，不能
直接合并。若后续测试 GC freeze，应为 FSDP 增加独立 default-off 开关，对 BF16
和 MXFP4 在相同 completed-step 边界对称执行 `gc.collect(); gc.freeze();
dist.barrier()`，并把完整计时窗口放在 freeze 之后。测试必须 mock 全局 GC
调用，并记录 GC pause、freeze count 与 host RSS。历史性能数字不能用于当前
源码的接受判断。

## 0. 一分钟结论

本轮目标是让同一训练配置下的 MXFP4 达到 BF16 step 速度的至少
`1.6x`，同时把训练精度损失控制在可接受范围内。

截至交接时，**目标尚未完成**。不得把任何局部 kernel 加速、profiler
比例、跨 campaign 点估计或仅 median 达标解释成最终达标。

最强的同 campaign 正式结果来自 MXFP4 `tail0`：

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|:---:|
| BF16 midpoint mean | `8078.3525 ms` | — | reference |
| MXFP4 tail0 mean | `5079.4900 ms` | `<=5048.9703 ms` | FAIL |
| Mean speedup | `1.590387x` | `>=1.6x` | FAIL |
| Median speedup | `1.607303x` | `>=1.6x` | PASS |
| Block-4 bootstrap 95% CI | `[1.576344x,1.603899x]` | lower `>=1.6x` | FAIL |
| BF16 midpoint validation NLL | `7.8817` | — | reference |
| MXFP4 tail0 validation NLL | `8.0262` | — | FAIL |
| Delta NLL | `+0.1445` | `<=+0.01` | FAIL |
| Perplexity delta | about `+15.55%` | — | materially worse |

因此：

- `tail0` 明确否决，不能作为默认精度策略。
- `tail1` 是当前速度/短程数值表现最好的 whole-layer 候选，但还没有完成
  同一 fresh campaign 的 BF16/tail1/BF16 最终确认，不能声称达到 `1.6x`。
- 保守工作策略仍是 `tail2`；但 tail2 同样没有 current-source、同 campaign 的
  BF16/tail2/BF16 最终确认。“保守”只表示比 tail1/tail0 多保留 BF16 层，不是
  已证明达到最终速度或长期精度门。`lm_head` 和 input embedding 均保持 BF16。
- 最后启动的 projection-guard campaign 在 smoke 后因 analyzer 的 shape
  count 预期错误 fail-closed，未进入正式 A/B/C arms。
- 用户已要求停止继续优化；本交接只记录状态，不代表恢复实验。

## 1. 最重要的迁移警告

### 1.1 已建立的远端 recovery snapshots

2026-09-22 已对两个 dirty worktree 做了只读审计，并建立或核验了以下恢复点：

| 仓库 | 远端分支 | 精确 commit | 用途 |
|---|---|---|---|
| `DaiXindi-AMD/Lumen` | `backup/2026-09-22/mxfp4-optimization-wip` | `74771d140c334cabc4c1d0023a525a1328b9a2b4` | 当前 Lumen 相关 dirty 源码、测试、benchmark、诊断脚本和 fresh 实验总账 |
| `DaiXindi-AMD/aiter` | `backup/2026-09-22/fused-swiglu-dual-layout-wip` | `35e796da188e2131d004e5b17391b7b8e836d852` | 当前 AITER 的 split-SwiGLU、two-way MXFP4 quant 和 fused dual-layout WIP |

两条远端 ref 均已用 live `git ls-remote` 核验。它们是 **recovery snapshot，
不是 PR-ready 分支**，不得整分支直接合并到产品分支。

Lumen recovery commit 基于本交接文档 commit
`8d7a8a8e0410ab87409bb97d0849e26b4d4becb8`，保存 34 个文件，包含：

- 当前链路所需的 Qwen3 CLI/FSDP2、packed QKV、split SwiGLU、MXFP4
  forward/backward、ASM registry、autotune/cache、weight-cache invalidation、
  tuned CSV、测试和 benchmark；
- default-off 或未完成的 projection guard、root FSDP retention、FlyDSL；
- 已作为速度方案否决但可用于历史复现的 packed gate/up；
- Hadamard 诊断脚本，但不含 PNG/HTML/JSON 生成物；
- fresh-reset 后的完整实验总账，移动到
  `docs/mxfp4_fresh_experiment_log_2026-09-22.md`，内容 SHA256 仍为
  `7c6e0f3da5944047df69056e705bc321cd2e7083347fbff88d3103802cb5887d`；
- 分支内的分类清单 `docs/mxfp4_wip_snapshot_manifest_2026-09-22.md`。

Lumen snapshot 明确排除了 `.agents/`、`AGENTS.md`、`.gitignore` 的 agent
scratch hunk、Hadamard 生成图、GPU trace/compiler cache/result 目录，以及未改变
gitlink 的 dirty `third_party/aiter`。后者唯一差异是无关 A8W8 CSV 的 CRLF/LF
换行转换。

AITER 当前 checkout 是：

```text
path:       /home/xdai/aiter
branch:     bench/ecfff3f-lumen
HEAD:       e35bb17f4f815903bf73598facedbb321e15af28
Lumen pin:  ecfff3fa80f906c5c421a35a7f5e52842f000559
relation:   HEAD is one commit ahead of the Lumen pin
```

AITER 远端 recovery commit 中的 17 个相关路径已逐一比较 Git blob hash，
与当前 dirty worktree **17/17 字节一致**。其中当前 Lumen 实际依赖 11 个：

```text
aiter/ops/triton/_triton_kernels/activation.py
aiter/ops/triton/activation.py
aiter/ops/triton/utils/_triton/activation.py
op_tests/triton_tests/fusions/test_fused_silu_mul.py
op_tests/op_benchmarks/triton/bench_swiglu.py
aiter/ops/triton/_triton_kernels/quant/quant.py
aiter/ops/triton/quant/quant.py
aiter/ops/triton/quant/__init__.py
aiter/ops/triton/utils/_triton/shuffle.py
op_tests/triton_tests/quant/test_quant_mxfp4_2way.py
op_tests/op_benchmarks/triton/bench_quant_mxfp4_2way.py
```

另外 6 个是当前 Lumen 尚未调用的 fused SwiGLU dual-layout WIP：

```text
aiter/ops/triton/_triton_kernels/quant/fused_swiglu_dual_layout_mxfp4.py
aiter/ops/triton/quant/fused_swiglu_dual_layout_mxfp4.py
aiter/ops/triton/configs/quant/gfx950-FUSED-SWIGLU-DUAL-LAYOUT-MXFP4.json
aiter/ops/triton/utils/quant_config_utils.py
op_tests/triton_tests/quant/test_fused_swiglu_dual_layout_mxfp4.py
op_tests/op_benchmarks/triton/bench_fused_swiglu_dual_layout_mxfp4.py
```

该 WIP 已有 repr、public wrapper、gfx950 config、unit test 和 benchmark，目标是
一次产生 BF16 SwiGLU、row-major MXFP4 和 H16-transposed MXFP4；但尚未接入
Lumen，也没有在本次备份审计中重跑 GPU correctness/performance。

新机器恢复 Lumen snapshot：

```bash
git clone https://github.com/DaiXindi-AMD/Lumen.git
cd Lumen
git fetch origin backup/2026-09-22/mxfp4-optimization-wip
git switch -c recovery/mxfp4 FETCH_HEAD
test "$(git rev-parse HEAD)" = 74771d140c334cabc4c1d0023a525a1328b9a2b4
```

新机器获取 AITER snapshot：

```bash
git clone https://github.com/ROCm/aiter.git
cd aiter
git fetch https://github.com/DaiXindi-AMD/aiter.git \
  refs/heads/backup/2026-09-22/fused-swiglu-dual-layout-wip
test "$(git rev-parse FETCH_HEAD)" = 35e796da188e2131d004e5b17391b7b8e836d852
```

AITER backup commit 的父提交是 Dai fork 的较新 `main`，不是当前
`e35bb17` 的线性后继；两者 merge-base 为 `5f930ea50a36`，左右提交数为
`9/595`。若必须复原旧运行树，应先 checkout 目标 AITER base，再从
`FETCH_HEAD` **按上面 17 个路径执行 `git restore --source=FETCH_HEAD
--worktree -- <paths>`**。不要直接 cherry-pick 整个 backup commit。

### 1.2 为什么没有把混合 dirty tree 直接提交到 `dev/mxfp4`

实验时的 Lumen source HEAD 是：

```text
repository:       /home/xdai/Lumen
branch:           dev/mxfp4
experiment HEAD:  9d85c8adb5159cc5765680bbc4c2230bb00e74e4
handoff commit:   8d7a8a8e0410ab87409bb97d0849e26b4d4becb8
upstream:         origin/dev/mxfp4
```

原来的 3 个功能提交是：

```text
9d85c8a feat(fsdp): fuse the pretraining cross-entropy to free logit-sized buffers
db0f330 feat(fsdp): dial activation recompute with --grad-checkpoint-layers
d8bb173 feat(fsdp): retain accumulated parameters across the FSDP2 window
```

当前 dirty 文件内同时混有已接受路径、已拒绝/default-off 实验、尚未完成的
projection guard，以及按仓库边界最终应移到 AITER 的 FlyDSL kernel。直接把
整棵工作树提交到共享 `dev/mxfp4` 会把这些状态错误地包装成同一个可合并实现。
因此采用独立 backup branch 保存完整字节，再由下一位 agent 按 hunk 拆成可审查
提交。`dev/mxfp4` 只保存交接文档和此前已经独立提交的改动。

**旧机器上的 dirty tree 仍未被切分、清理、stash 或覆盖。** 不要对它执行
`git reset --hard`、`git clean`，也不要把 recovery branch 直接当作最终方案。

### 1.3 当前 source snapshot 指纹

最后一次 projection-guard smoke 冻结到的运行源码状态：

```text
source_bundle_sha256=fef672a40fb77a0bab43740302c0ac571a8f65fb0c7ab67d89cb70f19b7790a1
lumen_commit=9d85c8adb5159cc5765680bbc4c2230bb00e74e4
lumen_diff_sha256=346860766945762e587bff65b1cbf9486e73ab0f5bd884a86f7fb9fe9527021a
aiter_commit=e35bb17f4f815903bf73598facedbb321e15af28
aiter_diff_sha256=a5476078484e426e951c2a5e62233cb20f26c103f93656e02a690e294e9f9aa0
lumen_tree_sha256=1027a15985c3695d503c99bfe202b414d60d3393c598284ba6214b01d95ef1db
aiter_tree_sha256=97d328fed8375a723c0bd3fd3e6972a6fe274e490716af490f4ba52d72cba7d0
runtime_modules_sha256=e6aeddca57b89a79c0f9d04c4708b4574c8737d58f07df8c2fbd099edb873c65
```

添加本交接文档及其 commit 会改变 Git tree/status，因此未来 campaign 必须重新
建立 source manifest，不能继续使用旧 campaign 的冻结检查。

## 2. 环境与固定 workload

### 2.1 软件和硬件

```text
GPU:          8 x AMD Instinct MI350X VF
architecture: gfx950
PyTorch:      2.13.0+rocm7.2
HIP:          7.2.53211
Lumen import: /home/xdai/Lumen/lumen/__init__.py
AITER import: /home/xdai/aiter/aiter/__init__.py
GPU lock:     /home/xdai/profile-results/.lumen-gpu-exclusive.lock
```

长期可见的 KFD PID `2537950` 是系统 `/usr/local/bin/gpuagent`，历史检查中为
`0 VRAM / 0 CU`，不是训练任务。仍应在每次新 run 前重新核对，而不是永久白名单
任意 PID。

### 2.2 仓库状态

```text
Lumen:
  path:   /home/xdai/Lumen
  branch: dev/mxfp4
  commit: 9d85c8adb5159cc5765680bbc4c2230bb00e74e4
  remote origin: https://github.com/ZhangDanyang-AMD/Lumen.git
  remote mine:   https://github.com/DaiXindi-AMD/Lumen.git

AITER runtime checkout:
  path:   /home/xdai/aiter
  branch: bench/ecfff3f-lumen
  commit: e35bb17f4f815903bf73598facedbb321e15af28
  remote origin: https://github.com/ROCm/aiter.git
```

Lumen 的 `third_party/aiter` gitlink 指向 `ecfff3fa...`，但实际 Python import
来自外部 editable checkout `/home/xdai/aiter`。后续 agent 必须同时验证
`import lumen`、`import aiter` 的真实路径和 commit，不能只看 submodule。

### 2.3 模型、数据和训练参数

```text
model/tokenizer: /home/xdai/models/Qwen3-8B
train data:      /home/xdai/fp8-coworker-repro/data/c4_train_1k_repeat4.jsonl
validation:      /home/xdai/fp8-coworker-repro/data/c4_valid_heldout.jsonl
sequence length: 8192
micro batch:     2
global batch:    128
grad accumulation: 8
tokens/update:   1,048,576
seed:            1234
FSDP:            version 2, full_shard
reduction dtype: BF16
activation checkpointing: off
MXFP4 communication: off
attention:       AITER
normalization:   Lumen
RoPE:            fused
cross entropy:   fused
validation:      16 batches / 256 samples in final campaigns
```

正式 50-step campaign 使用 `TRAIN_SAMPLES=6400`，统计 steps 11--50，共
40 个 paired positions。所有正式 timing arms 必须关闭 profiler 和 shape logging。

## 3. 证据规则

### 3.1 Fresh reset 边界

用户在 2026-09-20 明确要求回退最新 fast-path 实验，并要求后续优化不得使用
已有 profiling 选择新候选。该 reset 之后的结论必须来自：

- 新 result directory；
- absent-at-start 的 fresh MXFP4 autotune cache；
- 新执行的 BF16/MXFP4 或 control/candidate/control；
- 相同初始化、数据顺序、训练参数和运行源码；
- 每个 rank 的配对 batch/validation digest；
- KFD idle、source/cache/tree/runtime provenance 检查。

reset 前的数字可以解释优化链路历史，但不能用于新的接受或最终达标声明。
尤其是曾出现的 `1.764x` “optimized recipe vs stock BF16”结果，同时改变了
checkpointing、FSDP retention 和 reduction policy，不是当前要求的
same-policy BF16/MXFP4 格式对照，而且已被用户的 fresh reset 排除。不要把它写成
最终成功。

### 3.2 三类性能证据必须分开

1. **Kernel microbenchmark**：只证明某一 exact shape 的局部实现；不能外推
   step speed。
2. **Profiler trace**：用于定位 GPU busy、idle、launch 和 kernel family；
   profiler span 不是正式吞吐。
3. **Unprofiled E2E**：只有冻结的 A/B/A 或 palindrome、warmup 后窗口统计，
   才能作为 step speed 决策。

不同 campaign 的点估计不能相加，也不能用一个 campaign 的 BF16 除以另一个
campaign 的 MXFP4 形成“最终比例”。

### 3.3 最终速度与精度门

最终 same-policy BF16/MXFP4 确认至少要求：

- mean speedup `>=1.6x`；
- median speedup `>=1.6x`；
- paired block-4 circular moving-block bootstrap，100,000 resamples，95% CI
  下界 `>=1.6x`；
- 至少 `28/40` paired wins；
- BF16 A1/A2 mean、median drift 均 `<3%`；
- candidate validation `delta NLL <= +0.01`；
- 所有 integrity、pairing、routing、cache、finite-value、exit-status 检查通过。

50-step one-seed validation 仍只是筛选。默认采用前还要求至少：

```text
3 seeds x >=200 optimizer steps
one-sided 95% upper confidence bound of validation delta NLL <= +0.01
```

统计口径注意：早期 baseline 表中的“midpoint median”是两个 control run median
的平均；后期 palindrome/final analyzer 通常先按同一 step position 对两个 control
取 midpoint，再对 midpoint 序列取 median。两者不必完全相等。另外，后文
WGrad、packed QKV、split SwiGLU 表中的第二个 saving 是 paired per-step
difference 的 median，不是简单的 `control median - candidate median`。

## 4. 当前 MXFP4 执行链路

当前优化训练路径概念上是：

```text
Qwen3-8B initialization
  -> deterministic rank-local data generator / pairing evidence
  -> generic MXFP4 Linear patching
  -> optional final-layer BF16 projection restoration
  -> packed QKV patching
  -> split SwiGLU patching
  -> FSDP2 full_shard
       + retained accumulated decoder-layer parameters
       + BF16 gradient reduction
  -> no activation checkpointing / use_cache=False
  -> AITER attention + Lumen norm + fused RoPE + fused CE
  -> BF16 lm_head
```

当前重点代码位置：

| 功能 | 路径 |
|---|---|
| Qwen3 训练 CLI、pairing、projection guard | `examples/qwen3/train_qwen3_fsdp.py` |
| launcher 参数 | `examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh`、`examples/scripts/train_pretrain.sh` |
| FSDP2 retained params / reduce dtype | `lumen/models/fsdp.py` |
| Qwen packed QKV / split SwiGLU patching | `lumen/models/qwen3.py` |
| guarded AITER dispatch | `lumen/ops/dispatch.py` |
| packed/split SwiGLU autograd | `lumen/ops/fused_swiglu.py` |
| MXFP4 Linear forward/backward | `lumen/ops/quantize/linear.py` |
| ASM registry | `lumen/ops/quantize/mxfp4_asm.py` |
| autotune/cache | `lumen/ops/quantize/mxfp4_autotune.py` |
| weight/cache invalidation | `lumen/quantize/__init__.py` |
| model-specific A4W4 table | `examples/qwen3/configs/qwen3_8b_a4w4_blockscale_tuned_gemm.csv` |

关键运行开关：

```text
--fsdp-retain-accumulated-params
--fsdp-reduce-dtype bf16
--no-grad-checkpointing
--mxfp4-pack-qkv
--mxfp4-fuse-swiglu
--num-layers-at-end-in-bf16 <N>
--mxfp4-last-layer-bf16-projections {o_proj,down_proj}
```

最后一个 projection 选项是 default-off 实验开关，只允许：

- `--mode mxfp4`；
- full-parameter training（`--lora-rank 0`）；
- `--num-layers-at-end-in-bf16 0`；
- `o_proj` 和/或 `down_proj`，不再允许 Q/K/V 或 gate/up。

## 5. Fresh baseline 与总体进展

回退后的 fresh baseline 使用 tail5、BF16 `lm_head`、retained accumulated
params、无 activation checkpointing、BF16 reduction：

| Arm | Mean | Median | Validation NLL |
|---|---:|---:|---:|
| BF16 A1 | `8315.810 ms` | `8216.000 ms` | `9.1770` |
| MXFP4 tail5 | `6018.175 ms` | `5923.350 ms` | `9.1398` |
| BF16 A2 | `8153.315 ms` | `8113.050 ms` | `9.1772` |

对称 BF16 midpoint：

- mean/median：`8234.5625/8164.525 ms`
- MXFP4 speedup：`1.368282x/1.378363x`
- 95% CI：`[1.342426x,1.396570x]`
- 距离 `1.6x` mean target：`871.573 ms/step`
- validation delta NLL：`-0.0373`

artifact：
`/home/xdai/profile-results/lumen-mxfp4-postrollback-20260920-143648/`

经过 WGrad ASM、packed QKV、split SwiGLU 和 tail reduction 后，tail0 的
正式 mean 已降到 `5079.490 ms`，把 mean 目标差距缩到约 `30.52 ms`；但
tail0 的精度显著失败。这个进展只说明性能接近目标，不等于存在可采用配置。

## 6. 已接受或保留的优化

### 6.1 FSDP2 retained accumulated parameters

机制：在 GA8 accumulation window 内保留 decoder-layer full parameters，减少
重复 all-gather。

历史单变量证据：

- all-gather：`592 -> 81 calls/step`，减少 `86.3%`；
- steady-state mean：`8344.1 -> 7901.3 ms`，约 `1.056x`；
- peak memory：`57.5 -> 70.5 GiB/GPU`。

这是 memory-for-speed 策略，后续所有 fresh 正式 campaign 都继续使用并重新
验证了该运行组合。它本身的最初数字在 reset 之前，只作历史机制说明。

### 6.2 关闭 activation checkpointing

机制：明确传 `use_cache=False` 后关闭训练重计算。

历史机制证据：

- MXFP4 GEMM：`6944 -> 5208 calls/step`；
- dual quant：`5208 -> 3472 calls/step`；
- FMHA forward：`576 -> 288 calls/step`；
- unprofiled mean/median：`8330.7/8379.8 -> 6402.3/6319.3 ms`；
- peak memory：`57.5 -> 141.3 GiB/GPU`。

后续 fresh campaigns 均使用 no-checkpoint；原始单变量数字只作历史说明。

### 6.3 BF16 gradient reduction

机制：相对 FP32 reduction，减少 reduce-scatter bytes 和 FP32->BF16 cast。

历史机制证据：

- comparable reduce-scatter union：`121.656 -> 61.274 ms`；
- 37 个 per-step cast kernel 消失；
- peak memory：`154.2 -> 139.0 GiB/GPU`；
- later-window mean speedup：`1.0279x`。

后续 BF16 与 MXFP4 正式对照均统一使用 BF16 reduction，避免把 reduction
policy 差异冒充精度格式收益。

### 6.4 exact WGrad ASM route

目标 shape：

```text
(M,N,K)=(12288,4096,16384)
ASM symbol: _ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E
split-K: 0
```

Kernel/dispatch：

- shuffled median：`0.602666/0.606707 ms`
- ASM median：`0.369164 ms`
- SNR：两者均 `55.6165 dB`
- exact production dispatch 复测：`0.365284 ms`

8-GPU unprofiled A/B/A，steps 11--30：

| Control midpoint mean/median | Candidate mean/median | Mean / paired-median saving | Speedup | 95% CI |
|---:|---:|---:|---:|---:|
| `6169.265/5981.975 ms` | `5964.930/5878.500 ms` | `204.335/121.350 ms` | `1.034256x/1.017602x` | `[1.018162x,1.048172x]` |

- paired wins：`17/20`
- validation delta NLL：`-0.0145`
- integrity：`39/39`

结论：接受并保留 model-specific tuning row。

artifact：
`/home/xdai/profile-results/lumen-mxfp4-current-wgrad-asm-fresh-20260921-063804/`

### 6.5 packed QKV

结构变化：把每层 Q/K/V 的三套 forward、dgrad、wgrad 和重复 activation
quantization 合并为 packed shapes，同时保留原始三个 Parameter 和 state-dict
keys。

Exact-shape full-chain microbenchmark：

- forward+backward：`7.218542 -> 3.051595 ms`，`2.365498x`
- direct one-layer GA8 update：`53.027225 -> 20.310072 ms`，`2.610883x`
- 输出/dX/dW correctness gates 均通过
- microbenchmark 不能直接乘 31 层计入 step speed

正式 8-GPU A/B/A：

| Control midpoint | Packed QKV | Mean / paired-median saving | Speedup | 95% CI |
|---:|---:|---:|---:|---:|
| `5823.868/5724.450 ms` | `5626.865/5575.150 ms` | `197.003/147.375 ms` | `1.035011x/1.026780x` | `[1.013824x,1.056741x]` |

- paired wins：`17/20`
- validation delta NLL：`-0.0248`
- control drift：`0.0082%/0.0017%`
- integrity：`54/54`
- peak memory：`138.9 -> 134.7 GiB/GPU`

后续还修复了真实 FSDP2 DCP `2-rank save -> 8-rank load` 场景下的 stale
Q/K/V MXFP4 cache：DCP 会原地恢复 Parameter bytes 而不改变 identity/version，
因此必须由 root load-state post-hook 清理 module/Parameter 两侧缓存。修复后的
save/load/optimizer-state/route 检查通过。

artifact：
`/home/xdai/profile-results/lumen-mxfp4-packed-qkv-formal-fresh-20260921-5d0u3R/`

### 6.6 split SwiGLU

当前接受的是 separate gate/up projection 后的 split SwiGLU AITER autograd，
不是被拒绝的旧 packed-gate/up 方案。

正式 8-GPU A/B/A：

| Control midpoint | Candidate | Mean / paired-median saving | Speedup | 95% CI |
|---:|---:|---:|---:|---:|
| `5654.060/5582.425 ms` | `5534.220/5455.900 ms` | `119.840/124.600 ms` | `1.021654x/1.023190x` | `[1.018325x,1.024800x]` |

- paired wins：`19/20`
- validation delta NLL：`-0.00945`
- integrity：`46/46`
- peak memory：`134.7 -> 123.1 GiB/GPU`

artifact：
`/home/xdai/profile-results/lumen-mxfp4-split-swiglu-fresh-20260921-192245/`

### 6.7 BF16 tail 从 5 减到 2

Stage-1 palindrome：

| BF16 tail | Mean | Median | Speedup vs tail5 | Delta NLL vs tail5 |
|---:|---:|---:|---:|---:|
| 5 | `5510.530 ms` | `5461.675 ms` | `1.000000x` | `+0.00000` |
| 4 | `5417.315 ms` | `5382.575 ms` | `1.017207x` | `+0.01775` |
| 3 | `5341.615 ms` | `5290.500 ms` | `1.031622x` | `+0.02265` |
| 2 | `5272.025 ms` | `5210.100 ms` | `1.045240x` | `+0.02095` |

每一步 tail5->4、4->3、3->2 的 mean speedup CI 下界均高于 1；tail2
通过当时 `+0.03` 的短程 screen，因此被选入下一阶段。它不是长期收敛证明。

artifact：
`/home/xdai/profile-results/lumen-mxfp4-tail-reduction-fresh-20260921-xxgdel/`

## 7. 最新 fresh profiler 诊断

tail2 profiler campaign 使用 packed QKV、split SwiGLU、BF16 reduction、
retained accumulated params、no checkpoint 和 BF16 `lm_head`。

### 7.1 Profiler span

| Metric | BF16 | MXFP4 tail2 |
|---|---:|---:|
| Profiler span | `8139.274 ms/step` | `5401.819 ms/step` |
| GPU busy union | `7986.135 ms/step` | `5027.283 ms/step` |
| GPU envelope | `8136.335 ms/step` | `5396.644 ms/step` |
| Idle in envelope | `150.200 ms/step` | `369.361 ms/step` |

profiler-only 比值约 `1.507x`。它只用于定位热点，不能作为最终 throughput
或 accuracy 声明。

### 7.2 tail2 overlap-safe decomposition

| Category | Raw | Union |
|---|---:|---:|
| A4W4 GEMM | `1200.515 ms` | `1196.536 ms` |
| MXFP4 quant/layout | `351.020 ms` | `350.694 ms` |
| A4W4 + quant/layout joint union | — | `1542.964 ms` |
| BF16 `lm_head` | `476.895 ms` | `476.877 ms` |
| 两个 BF16 tail layers | `260.275 ms` | `259.775 ms` |
| Attention | `1684.751 ms` | `1683.177 ms` |
| Collectives | `183.670 ms` | `183.631 ms` |
| Norm | `228.014 ms` | `224.968 ms` |
| RoPE | `75.167 ms` | `74.223 ms` |
| Cross entropy | `53.725 ms` | `53.721 ms` |
| Optimizer | `53.580 ms` | `53.355 ms` |
| Copy/memset | `69.636 ms` | `69.618 ms` |
| Unclassified | `10.854 ms` | `10.739 ms` |

这些 category 不能随意相加；joint union 才是处理 overlap 后的局部上限。

主要 exact A4W4 shapes：

| Role / shape `(M,N,K)` | Time/step | Route |
|---|---:|---|
| MLP forward `(16384,12288,4096)` | `226.561 ms` | ASM 128x512 |
| WGrad `(12288,4096,16384)` | `203.198 ms` | ASM 128x512 |
| DGrad `(16384,4096,12288)` | `193.402 ms` | ASM 256x256 |
| packed-QKV forward | `60.538 ms` | ASM |
| packed-QKV dgrad | `51.094 ms` | ASM |
| packed-QKV wgrad | `48.905 ms` | ASM 192x256 |

Quant/layout：

- activation dual-layout quant：`116.682 ms/step`
- gradient dual-layout quant：`221.988 ms/step`
- packed transpose：`5.127 ms/step`
- scale swizzle：`2.006 ms/step`
- conversion/helper：`5.217 ms/step`

Host side 记录到：

- `23,137` launches/step；
- `41,574 hipPointerGetAttribute` calls/step；
- raw CPU time `1056.360/35.856 ms/step`。

这些 CPU spans 与 GPU 异步重叠，不能直接加到 GPU decomposition。它们只说明
launch/dispatch 数量仍很高。

原始 analyzer 报告标记两个 FAIL，但独立复核确认是 analyzer contract 缺陷：

1. 把 sampler cardinality 不同的 smoke/replay 与 formal arm 的 first-update
   hash 强行比较；formal pair 内部其实完全匹配。
2. 把 two-step host-event totals 与 per-step expected counts 比较；实际 packed
   与普通 Linear 调用次数和静态拓扑一致。

原报告保持 FAIL，未被改写；trace 只作为诊断证据使用。

artifact：
`/home/xdai/profile-results/lumen-mxfp4-tail2-profile-fresh-20260922-MA4B5n/`

## 8. tail2 / tail1 / tail0 高功效正式消融

50-step palindrome：

```text
tail2_a1 -> tail1_b1 -> tail0_c -> tail1_b2 -> tail2_a2
```

统计 steps 11--50：

| Policy | Mean | Median | Validation NLL |
|---|---:|---:|---:|
| tail2 midpoint | `5276.855 ms` | `5192.075 ms` | `7.93110` |
| tail1 midpoint | `5179.141 ms` | `5107.525 ms` | `7.92055` |
| tail0 | `5087.678 ms` | `5020.950 ms` | `7.93990` |

| Transition | Mean speedup | Median speedup | Mean saving | Wins | 95% CI |
|---|---:|---:|---:|---:|---:|
| tail2 -> tail1 | `1.018867x` | `1.016554x` | `97.714 ms` | `35/40` | `[1.001514x,1.037322x]` |
| tail1 -> tail0 | `1.017978x` | `1.017243x` | `91.464 ms` | `34/40` | `[1.011266x,1.024388x]` |
| tail2 -> tail0 | `1.037183x` | `1.034082x` | `189.178 ms` | `32/40` | `[1.019950x,1.055637x]` |

短程 NLL 相对 tail2：

- tail1：`-0.01055`，PASS；
- tail0：`+0.00880`，在这一轮 tail-relative gate 内 PASS；
- tail0 相对 tail1：`+0.01935`，是风险信号。

注意：tail0 对 tail2 的相对 NLL 通过，不能替代 tail0 对 BF16 的质量检查。
后者在独立正式 campaign 中失败了 `+0.1445`。这正是不能只在量化策略之间
比较、必须保留 BF16 reference 的原因。

artifact：
`/home/xdai/profile-results/lumen-mxfp4-tail210-confirm-fresh-20260922-vDKaSe/`

## 9. tail0 对 BF16 的最终确认

正式顺序：

```text
BF16 A1 -> MXFP4 tail0 B -> BF16 A2
```

每个 arm 50 steps，统计 11--50。完整性 `235/235`，独立 parser
`166/166`，所有 source receipts `134/134`，phase-1 manifest `67/67`。

| Arm | Mean | Median | Validation NLL |
|---|---:|---:|---:|
| BF16 A1 | `8083.5275 ms` | `8072.35 ms` | `7.8764` |
| MXFP4 tail0 | `5079.4900 ms` | `5018.25 ms` | `8.0262` |
| BF16 A2 | `8073.1775 ms` | `8062.75 ms` | `7.8870` |
| BF16 midpoint | `8078.3525 ms` | `8065.85 ms` | `7.8817` |

结果：

- mean speedup：`1.590387x`；
- median speedup：`1.607303x`；
- 95% CI：`[1.576344x,1.603899x]`；
- paired wins：`40/40`；
- BF16 replicate drift：`-0.1280%/-0.1189%`；
- delta NLL：`+0.1445`；
- perplexity delta：约 `+15.55%`。

结论：**速度门和精度门都失败，tail0 拒绝。**

artifact：
`/home/xdai/profile-results/lumen-mxfp4-tail0-bf16-confirm-fresh-20260922-YTeWY7/`

## 10. 词表层和 embedding 量化结论

### 10.1 Exact shape 与生产约束

```text
lm_head: (M,V,K)=(16384,151936,4096)
M*V=2,489,319,424 > signed int32
full BF16 logits ~= 4.6367 GiB
tie_word_embeddings=false
```

当前 production `lm_head` 是 stock HuggingFace `nn.Linear`。默认没有打开
`quantize_output_layer` 或 `lumen_linear`，并且当前 MXFP4 output path 会因
`M*V > 2^31` 的索引安全检查回退 BF16。

### 10.2 `lm_head` forward microbenchmark

| Path | Median | Relative to BF16 | Accuracy conclusion |
|---|---:|---:|---|
| BF16 | `17.776 ms` | `1.000x` | reference |
| A4W4 two chunks | `8.035 ms` | `2.212x` | rejected |
| FP8 per-tensor | `13.491 ms` | `1.318x` | KL gate failed |
| FP8 per-token | `16.518 ms` | `1.076x` | too little speed gain |
| FP8 blockwise 1x128 | `26.333 ms` | `0.675x` | slower |
| A16W8 plain | `30.170 ms` | `0.586x` | slower |
| A16W8 preshuffled | `40.204 ms` | `0.440x` | slower |
| public AITER A16WFP4 | `30.733 ms` | `0.581x` | slower |

真实 checkpoint、589 token、完整 vocabulary 的静态精度：

| Candidate | Logits SNR | KL mean/p99 | Top-1 | Delta NLL |
|---|---:|---:|---:|---:|
| A4W4 | `14.159 dB` | `0.115665/0.457124` | `78.27%` | `+0.086155` |
| FP4 weight-only oracle | `17.841 dB` | `0.064682/0.230306` | `85.40%` | `+0.061486` |
| FP8 per-tensor | `28.522 dB` | `0.005688/0.023675` | `93.38%` | `+0.009396` |
| FP8 per-token | `28.984 dB` | `0.004745/0.018854` | `95.93%` | `-0.004386` |
| FP8 blockwise | `30.516 dB` | `0.004181/0.015585` | `96.10%` | `-0.002310` |
| A16W8 | `31.862 dB` | `0.003853/0.015322` | `96.43%` | `-0.001352` |

保守门：KL mean `<=0.002`、KL p99 `<=0.01`、delta NLL `<=+0.01`。
没有候选同时通过性能和精度。

8-chunk FP8 per-tensor 包含 activation quant 和完整-output `torch.cat` 后为
`15.173 ms`，相对 fresh BF16 `17.683 ms`，GA8 且每 update 只量化一次
weight 的理论节省仅 `17.965 ms/update`。这还不含 CE、dX、dW、FSDP 和
调度，因此不能记作 step 收益。

直接写入 noncontiguous vocabulary slice 的 no-copy 实验产生
`10.5--10.75` max-abs error，说明 AITER strided-output/split-K correctness
存在问题；这些时间无效。

### 10.3 保精度混合修正

8-chunk raw FP8 的 top-8 和 true-label logit 用 BF16 master weight 重算后：

- KL mean/p99：`0.001454/0.006992`；
- top-1：`100%`；
- delta NLL：`-0.002802`。

静态精度通过，但性能为 `29.650 ms`，只有 BF16 的 `0.597x`，明确更慢。

### 10.4 Input embedding

| Path | Median | Relative |
|---|---:|---:|
| BF16 lookup | `0.0517 ms` | `1.000x` |
| FP8 per-tensor gather/cast/scale | `0.2259 ms` | `0.229x` |
| FP8 per-row gather/cast/scale | `0.2303 ms` | `0.224x` |

结论：input embedding 与 `lm_head` 都保持 BF16。词表层量化不能成为达到
`1.6x` 的主路径。

artifacts：

- `/home/xdai/profile-results/lumen-mxfp4-lm-head-quant-fresh-20260921/`
- `/home/xdai/profile-results/lumen-mxfp4-lm-head-a16fp4-sweep-fresh-20260921-035615/`
- `/home/xdai/profile-results/lumen-mxfp4-vocab-hybrid-fresh-20260921-053214/`
- `/home/xdai/profile-results/lumen-mxfp4-lm-head-bwd-layout-fresh-20260921-050243/`

## 11. 已拒绝或暂缓的其他尝试

| Candidate | Fresh evidence | Decision |
|---|---|---|
| Deferred loss readback | `6494.455 ms` vs control `6157.523 ms`; `0.948120x`; CI `[0.928711x,0.970060x]` | rejected and code removed |
| Root FSDP parameter retention | root all-gathers `9 -> 1`, but no repeatable wall win; validation `+0.0016` missed original `+0.001` gate | hold, no credit |
| Registry freeze | hot metadata lookup much faster locally, but E2E freeze path regressed | rejected/default-off, later rolled back |
| Registry freeze + weight-cache fast hit | deployable pair `0.982089x` vs live control; CI crossed/no benefit | rejected and rolled back |
| Activation descriptor cache | profiler removed 744 quant launches and 56.563 ms raw target work, but E2E `0.9941x` mean | rejected/default-off |
| `shard_grad_op` | mean/median disagreed, no material win, numerical trajectory worse | rejected |
| MBS4/GA4 | smoke ratio `1.3202x`, peak board use about `244.8 GiB`, only ~6.9 GiB headroom | screened out |
| Dual-layout BM/BN retune | current `256x32` won all exact shapes; GPT-OSS BM128 result did not transfer | keep `256x32` |
| Packed gate/up | mean `1.010789x`, but only `12/20` wins and CI `[0.947550x,1.077913x]` | rejected for speed; opt-in/default-off only |
| Old separate-input split SwiGLU | median gate and NLL `+0.0488` failed | rolled back |
| Current split SwiGLU | new implementation passed `1.021654x` mean, CI and NLL gates | accepted |
| tail0 | `1.590387x` mean, CI lower `<1.6`, delta NLL `+0.1445` | rejected |

外部 GPT-OSS 优化材料中最可借鉴、且在本链路得到验证的原则是：

- 必须按 production exact shape 调优；相似 kernel 的 tile 结论不能直接迁移；
- 优先消除重复 quant/layout、GEMM 和 autograd launch，而不是只改一个小 kernel；
- 保留 microbenchmark -> route smoke -> unprofiled E2E -> quality 的验证阶梯；
- 用逐项消融和对称 control，不能把多个局部收益直接相加。

## 12. 最后一个未完成实验：projection guard

### 12.1 目的

tail0 速度接近目标但精度失败，因此尝试只保留最后一层的敏感 residual-output
投影为 BF16：

```text
A: tail1 whole layer BF16
B: tail0 + final o_proj + down_proj BF16
C: tail0 + final down_proj BF16
```

计划正式顺序：

```text
fresh B route/cache smoke
tail1 A1 -> B1 -> C -> B2 -> tail1 A2
```

正式 arm 原计划每个 50 steps，统计 11--50，16 validation batches。速度门为
mean/median `>=1.003x`、wins `>=28/40`、95% CI lower `>1`，A/B drift
`<3%`；B/C 相对 tail1 midpoint 的 delta NLL 必须 `<=+0.01`。

### 12.2 已完成 smoke

目录：
`/home/xdai/profile-results/lumen-mxfp4-projection-guard-formal-fresh-20260922-4GQFhw/`

route B 正确：

```text
unquantized linears:
  model.layers.35.self_attn.o_proj
  model.layers.35.mlp.down_proj
  lm_head
packed QKV layers: 36
split SwiGLU layers: 36
lm_head dtype: BF16
lm_head quant enabled: false
```

Smoke：

- step 1：`63515.3 ms`，包含 fresh compile/autotune，不可计入吞吐；
- step 2：`5243.2 ms`；
- step 3：`5082.9 ms`；
- loss：`12.7744 -> 12.7962 -> 12.7983`；
- grad norm：finite；
- validation NLL：`12.7908`；
- peak memory：`121.4 GiB/GPU`；
- 9 distinct MXFP4 shapes；
- training/postflight 均成功，仅有已知 post-success teardown traceback。

这些只是 smoke 数字，不能用于速度或精度选择。

### 12.3 campaign 为什么停止

phase1 status 为 `2`。analyzer smoke gate 通过 `43/44`，唯一失败项是
`smoke_exact_rank_shape_inventory`。

根因不是 kernel 或训练失败，而是 projection analyzer 直接复用了 tail0 的
`EXPECTED_SMOKE_SHAPES`。恢复最后一层 `o_proj/down_proj` 为 BF16 后，相关
MXFP4 调用数理应下降，但 analyzer 仍要求 tail0 count。

| Shape | Tail0 frozen expectation | Guard-B observed |
|---|---:|---:|
| `(4096,4096,16384)` | `864` | `840` |
| `(4096,12288,16384)` | `864` | `840` |
| `(6144,4096,16384)` | `864` | `864` |
| `(12288,4096,16384)` | `1728` | `1728` |
| `(16384,4096,4096)` | `2304` | `2240` |
| `(16384,4096,6144)` | `864` | `864` |
| `(16384,4096,12288)` | `3168` | `3128` |
| `(16384,6144,4096)` | `1440` | `1440` |
| `(16384,12288,4096)` | `3744` | `3720` |

所有 8 ranks 都报告相同 route 与相同 observed counts。正式
`tail1_a1/B1/C/B2/A2` 均未启动，因此没有正式结果。

### 12.4 若恢复该实验

不要在旧目录直接修改 analyzer 后继续 phase1：本交接 commit 已改变仓库状态，
旧 campaign 也已经 fail-closed。正确做法是：

1. 新建全新的 result root。
2. 复制 harness 输入，不复制旧 cache/arm outputs。当前 runner 还硬编码依赖
   `/home/xdai/profile-results/lumen-mxfp4-a4w4-e2e-fresh-20260920-193907/run_case.sh`；
   新机器必须同时迁移该文件，或把 harness 改成自包含的相对路径后重新冻结。
3. 为 A/B/C 分别从静态 topology 推导 expected shape counts，避免再次复用
   tail0 常量。
4. 更新 analyzer tests，覆盖三种 policy 的 exact per-rank shape inventory。
5. 重新跑 `bash -n`、`py_compile`、analyzer self-test、unittest 和 dry-run。
6. 从 absent cache 开始重新跑 B smoke。
7. smoke 全部通过后再运行完整 A1/B1/C/B2/A2。
8. 不得把此次 smoke 的 `5243.2/5082.9 ms` 加入正式统计。

## 13. AITER fused SwiGLU dual-layout MXFP4：当前状态

外部 AITER dirty checkout 已有公共 wrapper：

```text
/home/xdai/aiter/aiter/ops/triton/quant/fused_swiglu_dual_layout_mxfp4.py
```

它一次读取 gate/up 并生成：

- BF16 SwiGLU activation；
- row-packed MXFP4 + row scale，供 down-projection forward；
- H16-rotated/transposed column MXFP4 + col scale，供 down-projection WGrad；
- 可选 scale swizzle 和 column B-payload shuffle。

当前约束：gfx950、BF16、contiguous 2-D、M/N 为 32 的倍数。production
exact shape 是 SwiGLU output `(16384,12288)`，对应 down projection
`(M,N,K)=(16384,4096,12288)`。

当前证据：

- AITER targeted suite：`14 passed, 1 skipped`；
- production parity JSON：`all_passed=true`；
- `(16384,4096)`、`(16384,12288)`、`(16384,24576)` 的 activation、row
  data/scale、column data/scale 均与 unfused reference bitwise equal；
- 只有已知的 post-success `torch.library._del_library` traceback；
- 尚无可引用的 production-shape 性能报告；
- 尚未接入 Lumen；
- 尚无 8-GPU smoke 或 unprofiled E2E 结果。

artifact：
`/home/xdai/profile-results/lumen-mxfp4-fused-swiglu-dual-layout-fresh-20260921-203226/`

建议接入边界：

```text
AITER public API
  -> Lumen dispatch probe / try_backends
  -> Lumen custom autograd returns BF16 activation
     and marks quantized layout tensors non-differentiable
  -> down_proj consumes explicit pre-quantized row/column bundle
  -> fallback to current split_swiglu + independent dual-layout quant
```

不要从 Lumen 直接 import AITER private kernel。新 GPU kernel 继续归 AITER；
Lumen 只负责 probe、dispatch、参数传递、autograd/context 和 fallback。

验证顺序必须是：

1. AITER public-wrapper correctness；
2. Lumen dispatch/autograd/fallback tests；
3. exact production-shape microbenchmark；
4. 8-GPU route/correctness smoke；
5. fresh unprofiled A/B/A；
6. 若有性能收益，再做 BF16 final bracket 与多 seed quality。

## 14. 当前 dirty worktree 清单

### 14.1 Lumen tracked modifications

```text
.gitignore
benchmarks/bench_mxfp4_gemm.py
benchmarks/bench_mxfp4_gemm_models.py
examples/qwen3/configs/a4w4_blockscale_tuned_gemm.csv
examples/qwen3/configs/qwen3_8b_a4w4_blockscale_tuned_gemm.csv
examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh
examples/qwen3/train_qwen3_fsdp.py
examples/scripts/train_pretrain.sh
lumen/models/fsdp.py
lumen/ops/dispatch.py
lumen/ops/fused_swiglu.py
lumen/ops/quantize/__init__.py
lumen/ops/quantize/linear.py
lumen/ops/quantize/mxfp4_autotune.py
lumen/quantize/__init__.py
tests/models/test_fsdp2.py
tests/models/test_qwen3_fsdp_pretrain.py
tests/ops/test_dispatch.py
tests/ops/test_linear.py
tests/ops/test_quantize.py
tests/quantize/test_mxfp4_weight_cache_hook.py
third_party/aiter (dirty gitlink)
```

当前 tracked diff 规模约 `8208 insertions / 873 deletions`。多个文件可能混有
用户原有改动，不能批量 commit 或 reset。

### 14.2 Lumen important untracked paths

```text
AGENTS.md
benchmarks/bench_mxfp4_exact_shape_parity.py
benchmarks/bench_mxfp4_gate_up.py
benchmarks/bench_mxfp4_qkv.py
examples/qwen3/hadamard_outlier_analysis.py
examples/qwen3/hadamard_wgrad_analysis.py
lumen/models/qwen3.py
lumen/ops/quantize/mxfp4_asm.py
lumen/ops/quantize/flydsl_mxfp4.py
tests/ops/test_fused_swiglu.py
```

另有 `.agents/`、`.codex/`、`hadamard_analysis/`、`hadamard_wgrad_analysis/`
和 `lumen/kernels/flydsl/` 等未跟踪目录。先判断归属再迁移或提交。

### 14.3 AITER dirty paths

```text
modified:
  aiter/ops/triton/_triton_kernels/activation.py
  aiter/ops/triton/_triton_kernels/quant/quant.py
  aiter/ops/triton/activation.py
  aiter/ops/triton/quant/__init__.py
  aiter/ops/triton/quant/quant.py
  op_tests/triton_tests/fusions/test_fused_silu_mul.py

untracked and relevant:
  aiter/ops/triton/_triton_kernels/quant/fused_swiglu_dual_layout_mxfp4.py
  aiter/ops/triton/configs/quant/
  aiter/ops/triton/quant/fused_swiglu_dual_layout_mxfp4.py
  aiter/ops/triton/utils/_triton/activation.py
  aiter/ops/triton/utils/_triton/shuffle.py
  aiter/ops/triton/utils/quant_config_utils.py
  op_tests/op_benchmarks/triton/bench_fused_swiglu_dual_layout_mxfp4.py
  op_tests/op_benchmarks/triton/bench_quant_mxfp4_2way.py
  op_tests/op_benchmarks/triton/bench_swiglu.py
  op_tests/triton_tests/quant/test_fused_swiglu_dual_layout_mxfp4.py
  op_tests/triton_tests/quant/test_quant_mxfp4_2way.py
```

因此，AITER 的 fused kernel 也不会因本次 Lumen 文档 push 自动出现在新机器。

## 15. 新机器接手步骤

### 15.1 先恢复源码，不要先跑 GPU

1. 拉取 Lumen `dev/mxfp4`，确认本文档存在。
2. 确认提交链包含 `d8bb173`、`db0f330`、`9d85c8a` 和本文档提交。
3. 从旧机器迁移或人工重建 Lumen dirty source；不要把实验 cache、trace 当源码。
4. 从旧机器迁移或人工重建 `/home/xdai/aiter` dirty source。
5. 比较关键 source manifest；仅 commit 相同并不足够。
6. 确认 editable import 指向新机器实际 checkout，而不是系统包或 submodule。

本次交接提交推送到：

```text
remote URL: https://github.com/ZhangDanyang-AMD/Lumen.git
branch:     dev/mxfp4
```

新机器的 remote 名称不一定叫 `origin`，因此应按 URL/ref 核对，不要只相信本地
remote alias。

建议先执行：

```bash
python - <<'PY'
import aiter
import lumen
import torch

print("lumen", lumen.__file__)
print("aiter", aiter.__file__)
print("torch", torch.__version__, "hip", torch.version.hip)
print("devices", torch.cuda.device_count())
PY
```

### 15.2 恢复 workload

必须复制或重新校验：

```text
/home/xdai/models/Qwen3-8B
/home/xdai/fp8-coworker-repro/data/c4_train_1k_repeat4.jsonl
/home/xdai/fp8-coworker-repro/data/c4_valid_heldout.jsonl
```

最后一次 source manifest 记录的关键输入 hashes：

```text
train data: 13fb674babc993cb88b705ff5d55eccfefb1f5a96e71c072b8977e8c93dcd486
val data:   b1e66138eca1d6c43069c4f2e8a4084fb3051ad0107a93c768e9f48779101c35
config:     f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30
tokenizer:  aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
```

### 15.3 迁移实验资料

`/home/xdai/profile-results` 不在 Git 中。本文档中的路径只是旧机器位置。
完整主要目录约 38 GiB；如果空间有限，优先复制：

- `*analysis.md` / `*analysis.json`；
- `protocol.md`；
- runner/analyzer/test scripts；
- `run-meta.txt`、status、source/cache/tree manifests；
- 正式 arm 的 `train.log`；
- 只有需要重新解析 profiler 时才复制大型 `trace.json` 和 compiler cache。

还必须单独复制 `/home/xdai/Lumen/.codex/tmp-training-bugs.md`。它是未跟踪文件，
不会随本文档 commit/push 到新机器。projection-guard 若要原样恢复，还必须复制：

```text
/home/xdai/profile-results/lumen-mxfp4-a4w4-e2e-fresh-20260920-193907/run_case.sh
```

更稳妥的后续做法是把 runner 复制进新的 campaign root、改为相对路径，再重新
生成 source manifest；不要继续依赖旧机器绝对路径。

主要目录大小：

| Artifact root | Approx size |
|---|---:|
| postrollback baseline | `2.6 GiB` |
| WGrad ASM campaign | `12 GiB` |
| packed QKV formal | `2.1 GiB` |
| split SwiGLU formal | `2.1 GiB` |
| tail5--tail2 | `6.3 GiB` |
| tail2/tail1/tail0 | `2.1 GiB` |
| tail0/BF16 confirmation | `2.1 GiB` |
| tail2 profiler | `6.6 GiB` |
| failed projection guard | `2.1 GiB` |
| AITER fused kernel correctness | `895 MiB` |

### 15.4 恢复实验的建议顺序

如果用户重新授权继续优化：

1. 先读取本文件和 `.codex/tmp-training-bugs.md` 全文。
2. 校验 Lumen/AITER import、commit、dirty source、模型与数据 hashes。
3. 检查磁盘空间；此前出现过 LLVM `No space left on device`，新 fresh cache
   约可消耗 2 GiB，profile trace 还会额外占数 GiB。
4. 检查 `/dev/kfd` 权限、所有 KFD clients 和独占锁，不允许并发 GPU workload。
5. 先修 projection-guard shape-count analyzer，在新目录做 CPU/static tests 和
   dry-run，再 fresh smoke。
6. 完成 A1/B1/C/B2/A2 后，只有通过 speed/NLL/integrity 链的 policy 才能进入
   fresh BF16/candidate/BF16 final confirmation。
7. projection policy 如果仍不够，再接入 AITER fused SwiGLU dual-layout，按
   correctness -> Lumen integration -> exact microbench -> smoke -> E2E 的顺序。
8. 最终候选通过 `1.6x` 短程门后，再跑 `3 seeds x >=200 steps` quality。

## 16. 已知坑和不可重复的错误路径

- `torch.library._del_library` 的
  `ValueError: too many values to unpack (expected 2)` 在 `Training complete`
  之后出现，进程 exit 仍为 0。只能精确 allowlist 该 teardown，不得泛化忽略
  traceback。
- profiler 自身显著扰动 step time；正式 speed 必须 unprofiled。
- smoke、route probe 和 shape logging 的时间都不能计入 throughput。
- 不同 `train_samples` 会改变 shuffled sampler ordering；不要跨 smoke/formal
  直接比较 first-update hashes。
- analyzer 计数必须明确是 two-step total 还是 per-step，不能混用。
- packed dW split 产生 contiguous views 时，`part.contiguous()` 是 no-op；不能
  强制要求 profiler 中出现 contiguous kernel。
- KFD 检查不能只按命令行字符串匹配，否则审计 shell 自身可能被误判为 GPU
  workload。
- 不要并发运行子 agent 的 GPU benchmark；曾有一次 MBS4 ABBA 因并发被整轮
  判无效。
- terminal/session lifetime 可能在约 25 分钟回收；长 campaign 应分 phase，
  每个 arm 不允许原地 resume。
- NUMA balancing 曾显著扩大 step 方差；正式 runner 应固定 NUMA policy 并记录。
- AITER cache 和 Lumen MXFP4 autotune cache 是不同层级，都必须冻结。
- 当前 raw results 和 dirty source 不在远端；远端文档不是可执行快照。

## 17. 全部优化尝试的速度与 loss 总账

这一节是给接手 agent 的统一索引，避免只看到成功项。这里的“尝试”按独立优化
假设统计；同一候选的编译 smoke、route probe、cache replay 和正式 A/B/A 不重复
冒充多个收益。必要时仍列出较早或失败的重复 campaign，以解释为什么后来重新
测量或改变结论。

统一口径：

- speedup 均写成 `reference_time / candidate_time`，大于 1 才是 candidate 更快；
- `Delta NLL = candidate validation NLL - reference validation NLL`，负数更好；
- `--` 表示没有可辩护的 matched E2E timing 或 matched validation reference；
- smoke、profiler 和 kernel microbenchmark 会明确标注，不能当正式 step speed；
- 2026-09-20 最后一次 fresh reset 之前的数据只用于工程历史，不能用于当前接受；
- 不同表中独立 campaign 的 speedup 禁止相乘或相加。

### 17.1 最后一次 full-fresh reset 后的正式或决策级尝试

| 顺序 | 尝试 / 对照 | Mean / median speedup | Delta validation NLL | 结果与原因 |
|---:|---|---:|---:|---|
| 1 | deferred loss readback / normal readback | `0.948120x / 0.927971x` | `-0.0399` | 明确变慢，删除实现 |
| 2 | retain root FSDP params / default root behavior | `--`；短 profiler 未建立稳定 wall-time 收益 | `+0.0016` | 原始 `+0.001` 语义门失败；hold、不得计速 |
| 3 | dual-layout quant BM128/BN32 / current BM256/BN32 | micro only：activation / separate-grad / packed-grad 为 `0.911045x / 0.856404x / 0.865808x` | `--` | 三个 exact shape 全慢，保留 256x32 |
| 4 | exact WGrad ASM / shuffled path | E2E `1.034256x / 1.017602x` | `-0.0145` | 95% CI `[1.018162x,1.048172x]`，接受 |
| 5 | packed gate/up / separate gate/up | `1.010789x / 1.039022x` | `+0.0117` | 仅 `12/20` wins，CI `[0.947550x,1.077913x]`；拒绝为速度优化 |
| 6 | packed QKV / separate Q/K/V | `1.035011x / 1.026780x` | `-0.0248` | 95% CI `[1.013824x,1.056741x]`，接受 |
| 7 | current split SwiGLU / unfused activation path，在 packed QKV 上增量 | `1.021654x / 1.023190x` | `-0.00945` | 95% CI `[1.018325x,1.024800x]`，接受 |
| 8 | tail5 -> tail4 | `1.017207x / 1.014696x` | `+0.01775` vs tail5 | 速度 CI 下界 >1；通过当时 `+0.03` screen |
| 9 | tail4 -> tail3 | `1.014172x / 1.017404x` | `+0.00490` vs tail4 | 速度 CI 下界 >1；短程通过 |
| 10 | tail3 -> tail2 | `1.013200x / 1.015432x` | `-0.00170` vs tail3 | 速度 CI 下界 >1；短程通过 |
| 11 | tail5 -> tail2 整体 | `1.045240x / 1.048286x` | `+0.02095` vs tail5 | 选 tail2 进入下一阶段，非长期精度证明 |
| 12 | 第一次 30-step tail2 -> tail1 boundary | `1.016342x / 1.017124x` | `-0.00415` | CI `[0.998298x,1.034793x]` 穿过 1；不得接受 |
| 13 | 第一次 30-step tail1 -> tail0 boundary | `1.017163x / 1.015264x` | tail0 相对 tail1 `+0.01030` | 局部速度通过，但前置 tail1 gate 失败，不能越级选择 |
| 14 | 50-step high-power tail2 -> tail1 | `1.018867x / 1.016554x` | `-0.01055` | CI `[1.001514x,1.037322x]`；tail1 成为激进候选 |
| 15 | 50-step high-power tail1 -> tail0 | `1.017978x / 1.017243x` | `+0.01935` | 速度通过但相对 tail1 的质量风险明显 |
| 16 | 50-step high-power tail2 -> tail0 | `1.037183x / 1.034082x` | `+0.00880` | tail-relative screen 通过，不替代 BF16 reference |
| 17 | final tail0 MXFP4 / same-policy BF16 | `1.590387x / 1.607303x` | `+0.1445` | mean、CI lower 与精度门都失败；tail0 拒绝 |
| 18 | tail2 profiler / BF16 profiler | profiler-only `1.507x` | profile-only `+0.3910` (`12.7741-12.3831`) | 只作热点诊断，不作速度或精度接受 |
| 19 | final-layer `o_proj+down_proj` BF16 guard / tail1、down-only | `--` | `--` | route smoke 成功；正式 A/B/C arms 未运行 |
| 20 | AITER fused SwiGLU + dual-layout MXFP4 / unfused chain | `--` | `--` | correctness `14 passed, 1 skipped`；未接 Lumen、无 E2E |

上表中的 tail loss 差值使用同一 campaign 的相邻或明确 endpoint reference。
其中最容易误读的是 tail0：它在 tail2-relative screen 中只有 `+0.00880`，但在
真正 BF16-relative final campaign 中是 `+0.1445`；最终决策必须以后者为准。

### 17.2 较早、后来被 reset 或更严格复测取代的尝试

以下数字说明优化路径如何形成，但不能越过 fresh-reset 边界用于当前接受。

| 尝试 | 当时测得的速度 | 当时测得的 loss / NLL 差距 | 最终处理 |
|---|---:|---:|---|
| strict tail5 MXFP4 / BF16 baseline | `1.3343x / 1.3459x` | `+0.0006` | 旧基线，后来 fresh baseline 取代 |
| postrollback tail5 MXFP4 / same-policy BF16 baseline | `1.368282x / 1.378363x` | `-0.0373` | 优化链路起点；发生在最后一次 full-fresh candidate reset 前，只作历史锚点 |
| retain accumulated decoder params / retain off | `1.056x / 1.061x`；later window `1.060x / 1.069x` | observed `8.4277-8.5105=-0.0828`，但 SR seed 未严格配对 | 机制有效并保留；旧 loss 只作 finite screen |
| no activation checkpointing / checkpointing | `1.301x / 1.326x` | observed `8.5091-8.5105=-0.0014`，非严格配对 | 内存允许时保留；后续 fresh campaigns 固定使用 |
| no-checkpoint + retain / no-checkpoint only | `1.0485x / 1.0541x` | observed `+0.0156`；相对 stock BF16 为 `+0.0731` | 组合保留；不能把 `1.7161x` 跨 policy 结果当格式收益 |
| `shard_grad_op` / `full_shard` retain | `1.005601x` mean、`0.997089x` median | `+0.0709` | mean/median 方向冲突且数值更差，拒绝 |
| BF16-reference BF16 reduction / FP32 reduction control | `1.015379x / 1.006920x` | `+0.0003` (`8.4517-8.4514`) | 用于证明 reduction policy 影响；不是 MXFP4 格式收益 |
| MXFP4 BF16 reduction / MXFP4 FP32 reduction | `1.0279x / 1.0250x` | `-0.1109` validation NLL | 保留；同 policy BF16/MXFP4 当时仍仅 `1.3750x/1.3874x` |
| 首次 registry-freeze ON/OFF/ON | 表面约 `1.027314x` | 标签间 NLL 大幅波动 | 运行源码实际未消费开关，整组 attribution 无效 |
| reimplemented freeze + fast-hit，早期 locked matrix | vs live `1.0260x` mean；vs freeze `1.0506x` | observed `-0.0188` vs live midpoint，受随机性影响 | 当时仅 provisional；被下一次 fresh matrix 推翻 |
| exact WGrad 128x512，较早 step A/B/A | `0.994433x` mean、`1.001245x` median | `-0.01325` | CI 穿过无效点，回退 tuning row；后来全 fresh 正式复测才接受 |
| registry freeze only，fresh matrix | `0.972016x / 0.971878x` | 没有独立可靠的 freeze-only Delta NLL attribution | 拒绝/default-off |
| freeze + weight-cache fast-hit / live control | `0.982089x / 0.986215x` | `+0.0046` | deployable pair 变慢，拒绝并回退 |
| MBS4/GA4 / MBS2/GA8 | smoke-only BF16/MXFP4 ratio `1.3202x` | `-0.0114` (`12.7893-12.8007`) | 仅两样本且显存余量约 6.9 GiB，筛掉 |
| old separate-input split SwiGLU | `1.020376x / 1.004113x` | `+0.0488` | median 与精度门失败，回退；不是当前 split 实现 |
| early tail5 -> tail0 sandwich | `1.04291x / 1.05569x` | `+0.03705` | 当时精度门失败；后来重新设计 tail2/1/0 campaign |
| four exact-shape A4W4 ASM rows | `1.027037x / 1.027037x` | `-0.0523` | 当时接受；用户 fresh reset 后不得继续引用作当前证据 |
| activation descriptor cache | `0.9941x / 0.9963x` | `+0.0109` | profiler 减少 744 launches，但 E2E 变慢，拒绝/default-off |
| postrollback tail5 -> tail4 bracket | `1.028850x / 1.039360x` | `+0.00975` vs tail5；`-0.0285` vs BF16 | 当时通过，后来完整 tail5--tail2 palindrome 取代 |
| tail4 intrusive profiler / BF16 profiler | profiler-only `1.315762x` | profile-only `+0.3764` | 仅诊断，不能作精度接受 |

此外，existing-ASM top-key retune 只产生 kernel micro 结果：一个 challenger
`0.992926x`（更慢），另一个在两个进程中仅 `1.003840x/1.001713x`，估计约
`0.8 ms/step`，因此未进入 E2E；没有 matched training NLL。

### 17.3 词表层与 input embedding 的逐候选速度/loss

这些都是 exact-shape forward 或静态 checkpoint 屏幕，不是 8-GPU 完整训练
step。之所以逐项列出，是为了防止新机器再次从已经失败的方向开始。

| Candidate | 局部速度相对 BF16 | Delta NLL | 决策 |
|---|---:|---:|---|
| `lm_head` two-chunk A4W4 | `2.212x` | `+0.086155` | 明显精度失败 |
| FP4 weight-only oracle | `--`，只做静态 oracle | `+0.061486` | 即使不算 activation quant 仍精度失败 |
| FP8 per-tensor | `1.318x` | `+0.009396` | NLL 勉强但 KL mean/p99 失败 |
| FP8 per-token | `1.076x` | `-0.004386` | 速度收益太小且 KL 仍未全过 |
| FP8 blockwise 1x128 | `0.675x` | `-0.002310` | 更慢 |
| A16W8 plain | `0.586x` | `-0.001352` | 更慢且 KL 未全过 |
| A16W8 preshuffled | `0.440x` | `-0.001352`，同一权重量化精度屏幕 | 更慢 |
| public AITER A16WFP4 | `0.581x` | `--`，无 real-checkpoint accuracy | 更慢，且实际为 tile 内 A4W4 |
| 8-chunk raw FP8，含 activation quant + full-output cat | `1.179034x` | `+0.009360` | KL 失败；GA8 理论仅省约 `19.35 ms/update` |
| 8-chunk FP8 + top-8/label BF16 修正 | `0.597x` | `-0.002802` | 静态精度改善但明显更慢 |
| input embedding FP8 per-tensor | `0.229x` | `--`，只测 local SNR `31.409 dB` | 约 4.4x 更慢 |
| input embedding FP8 per-row | `0.224x` | `--`，只测 local SNR `31.512 dB` | 约 4.5x 更慢 |

`lm_head` BF16 backward layout 也做过独立审计：native no-copy dX/dW 已经生效；
AITER best dW 只有约 `1.003x`，dX 反而更慢，且没有 E2E/NLL 对照，因此未进入
生产集成。结论仍是 input embedding 和 `lm_head` 保持 BF16。

### 17.4 只到 microbenchmark、smoke 或 correctness 的尝试

| 尝试 | 已有速度信息 | loss / 精度信息 | 为什么没有 step-speed 结论 |
|---|---:|---|---|
| WGrad ASM exact-shape micro | 约 `1.63--1.66x` vs shuffled；production median `0.365284 ms` | SNR `55.6165 dB`，无 training NLL | 后续已有正式 E2E，正式数字见 17.1 |
| old SwiGLU tile retune | forward `0.989883x`，backward `1.001505x` | elementwise / ULP correctness；无 NLL | forward 更慢、backward 不稳定，未进 E2E |
| old separate-input SwiGLU exact autograd | forward `1.553787x`，fwd+bwd `1.633841x` | dgate SNR >=`108.951 dB` after ULP audit；无 NLL | 后续 E2E 精度失败，见 17.2 |
| packed QKV full-chain | fwd+bwd `2.365498x`；one-layer GA8 `2.610883x` | packed/control SNR regression最多 `0.006645 dB`；无 NLL | 后续 E2E 只有 `1.035011x`，不得乘 31 层 |
| packed gate/up smoke | post-startup `6206.3/5994.5 ms`，无 matched ratio | absolute validation `12.7892` | smoke 后正式 A/B/A 拒绝 |
| packed QKV smoke | post-startup `5760.0/5579.1 ms`，无 matched ratio | absolute validation `12.7888` | route/correctness only；正式结果见 17.1 |
| packed QKV + split SwiGLU smoke/probe | `5946.2/5455.7` 与 `5703.8/5448.9 ms`，无 matched ratio | absolute validation `12.7845/12.7853` | 正式结果见 17.1 |
| projection guard `o_proj+down_proj` smoke | post-JIT `5243.2/5082.9 ms`，无 matched ratio | absolute validation `12.7908` | formal 未启动，不得从 smoke 选策略 |
| AITER fused SwiGLU dual-layout | 无可引用 production-shape timing | bitwise parity pass，targeted suite `14 passed, 1 skipped` | 未接 Lumen、未测训练 loss/E2E |
| packed-QKV DCP stale-cache fix | `--` | save/load 后参数、optimizer、route/hash 正确 | correctness fix，不是速度候选 |

### 17.5 没有结果、结果无效或被 fail-closed 的执行尝试

| 尝试 | speedup | Delta NLL | 状态 |
|---|---:|---:|---|
| concurrent MBS4 ABBA | `--` | `--` | 与别的 GPU workload 并发，整轮无效 |
| packed-QKV first route probe | `--` | `--` | shape CSV 未 flush，instrumentation fail-closed |
| packed-QKV route probe v2 | `--` | `--` | KFD preflight 误报，未启动 `torchrun` |
| tail2 profiler attempt 1 | `--` | `--` | 16-hex/full-SHA 表示不一致，formal profiler 未启动 |
| interrupted tail palindrome suffix | `--` | `--` | session timeout，零 step；后来从该 arm 重新完整运行 |
| redundant tail1 bracket attempt | `--` | `--` | LLVM `No space left on device`，初始化期失败 |
| projection-guard formal campaign | `--` | `--` | smoke 成功，但 analyzer 复用 tail0 shape counts；A1/B1/C/B2/A2 均未启动 |

这张 fail-closed 表的 `--` 不是遗漏，而是明确表示没有合法的 paired 数据。
接手 agent 不得从残留日志中的单个 step、编译后 smoke 或 profiler span 补造
speedup/loss 结论。

## 18. 关键 artifact 索引

| 内容 | 路径 |
|---|---|
| 全部实验总账 | `/home/xdai/Lumen/.codex/tmp-training-bugs.md` |
| post-reset BF16/MXFP4 baseline | `/home/xdai/profile-results/lumen-mxfp4-postrollback-20260920-143648/` |
| WGrad ASM formal A/B/A | `/home/xdai/profile-results/lumen-mxfp4-current-wgrad-asm-fresh-20260921-063804/` |
| packed QKV formal | `/home/xdai/profile-results/lumen-mxfp4-packed-qkv-formal-fresh-20260921-5d0u3R/` |
| packed QKV DCP reshard | `/home/xdai/profile-results/lumen-qkv-fsdp2-dcp-reshard-fresh-20260921-1npYfh/` |
| split SwiGLU formal | `/home/xdai/profile-results/lumen-mxfp4-split-swiglu-fresh-20260921-192245/` |
| vocabulary quantization | `/home/xdai/profile-results/lumen-mxfp4-lm-head-quant-fresh-20260921/` |
| vocabulary hybrid + embedding | `/home/xdai/profile-results/lumen-mxfp4-vocab-hybrid-fresh-20260921-053214/` |
| tail5--tail2 stage 1 | `/home/xdai/profile-results/lumen-mxfp4-tail-reduction-fresh-20260921-xxgdel/` |
| tail2/tail1/tail0 confirmation | `/home/xdai/profile-results/lumen-mxfp4-tail210-confirm-fresh-20260922-vDKaSe/` |
| tail0/BF16 final confirmation | `/home/xdai/profile-results/lumen-mxfp4-tail0-bf16-confirm-fresh-20260922-YTeWY7/` |
| tail2 fresh profiler | `/home/xdai/profile-results/lumen-mxfp4-tail2-profile-fresh-20260922-MA4B5n/` |
| projection guard successful earlier smoke | `/home/xdai/profile-results/lumen-mxfp4-projection-guard-smoke-fresh-20260922-f68RA4/` |
| projection guard stopped formal attempt | `/home/xdai/profile-results/lumen-mxfp4-projection-guard-formal-fresh-20260922-4GQFhw/` |
| projection harness shared runner dependency | `/home/xdai/profile-results/lumen-mxfp4-a4w4-e2e-fresh-20260920-193907/run_case.sh` |
| fused SwiGLU dual-layout correctness | `/home/xdai/profile-results/lumen-mxfp4-fused-swiglu-dual-layout-fresh-20260921-203226/` |

## 19. 最终决策表

| 项目 | 状态 |
|---|---|
| Same-policy `>=1.6x` mean speedup | **未达到** |
| Same-policy 95% CI lower `>=1.6x` | **未达到** |
| 可接受短程 BF16-relative accuracy | tail0 **失败**；tail1 **尚未最终确认** |
| 长期 convergence | **未运行** |
| 当前保守策略 | tail2 + BF16 `lm_head`；尚无 final BF16 bracket |
| 当前激进候选 | tail1 + BF16 `lm_head`；尚无 final BF16 bracket |
| 最快但否决策略 | tail0 + BF16 `lm_head` |
| 词表层量化 | 拒绝，embedding/lm_head 保持 BF16 |
| WGrad ASM | 接受 |
| packed QKV | 接受 |
| split SwiGLU | 接受 |
| packed gate/up | 拒绝为速度优化，保留 opt-in/default-off |
| projection guard | route smoke 成功；formal 未运行，harness gate 失败 |
| AITER fused SwiGLU dual-layout | correctness 已过；未接 Lumen、未测 E2E |

最终可对外陈述的结论只能是：

> Lumen Qwen3-8B MXFP4 已通过多个 fresh、配对、8-GPU 消融把 tail5 baseline
> 推进到接近 `1.6x` 的 tail0 性能点，但 tail0 在同 campaign BF16 对照中只有
> `1.590387x` mean，95% CI 下界为 `1.576344x`，并出现 `+0.1445` validation
> NLL 回归。因此“`1.6x` 且精度可接受”仍未实现。当前最合理的继续方向是先
> 完成最后一层 projection-level BF16 guard 的 fresh 正式筛选，再评估 AITER
> fused SwiGLU + dual-layout MXFP4 的结构性集成。
