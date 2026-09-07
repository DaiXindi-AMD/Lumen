# SFT Experiment Report — v1 → v5e Full Evolution

## Overview

Nine rounds of SFT iteration (v1→v5e). Overall 50% → **74.1%**, sandbox compile rate 0% → **21.9% (42/192)**.
Key turning point: v5d proved that negative-example strategy is whack-a-mole (eliminating old hallucinations spawns new ones); v5e pivoted to positive-example flooding
(kernel code share 35%→52%), and sandbox compile rate doubled in one step from 9.9% to 21.9%.
SFT phase complete; the model has sufficient baseline capability to transition to RL (RFT → DAPO).

**HuggingFace**: https://huggingface.co/Zhangdanyang/Qwen2.5-Coder-SFT-v5e

## Version Evolution Summary

| Version | Sample Count | Kernel Rate | Val Loss | Overall | Sandbox Compile | Core Issue |
|------|--------|---------|----------|---------|---------|---------|
| Base | — | — | — | 50.4% | — | Doesn't understand FlyDSL |
| v1 | 2,808 | 18% | 1.030 | 56.3% | — | 82% non-code data |
| v2 | 2,916 | 59% | 0.985 | 72.4% | 0% | 37.6% truncation + r=32 |
| v3 | 3,096 | 60% | 0.974 | not tested | 0% | 81% import hallucination |
| v4 | 2,344 | 55% | 0.984 | 60.3% | 0.5% | degradation due to reduced dataset size |
| v5 | 2,596 | 58% | pending | 69.3% | 0% | secondary API hallucination |
| v5b | 2,792 | 60% | pending | 76.5% | 4.7% (9/192) | new variant hallucinations + syntax residue |
| v5c | 2,943 | 62% | pending | 73.6% | 10.4% (20/192) | import style + flyc API hallucination |
| v5d | 3,102 | 63% | pending | not tested | 9.9% (19/192) | whack-a-mole failure: old hallucinations eliminated, new ones emerge |
| **v5e** | **3,889** | **52%** | **pending** | **74.1%** | **21.9% (42/192)** | **SFT complete, transition to RL** |

## v1: Baseline — 82% Non-Code Data

### Problem

82% of SFT training data contains no FlyDSL kernel code:

```
@flyc.kernel occurrence rate:  354/2808 (13%)
fx.* API:             471/2808 (17%)
Actual kernel code:     500/2808 (18%)
QA/docs/refusal:        2308/2808 (82%)

By source:
  documentation_qa (669):        8% contain kernel
  ai_annotated_instruction (375): 2% contain kernel
  refusal_boundary (135):         0% contain kernel
  augmentation_tile (105):       75% contain kernel  ← high quality but too few
```

### Results

- Overall 56.3% (base 50.4%, only +5.9%)
- `@flyc.kernel` usage dropped from base 96% to 40% — model learned to write documentation answers instead of kernels
- L4-L5 barely improved

### Scripts

No special scripts; used original `flydsl-agent-dataset`.

---

## v2: Data Resampling + Kernel Extraction + r=64 + seq=16384

### Problem

Two root causes from v1:
1. **Data ratio imbalance** — kernel code only 18%
2. **Training capacity limits** — r=32 insufficient capacity, seq=8192 truncated 37.6% of data

### Solution

1. **Extract kernel code from CPT data**: 54 FlyDSL kernel files → 108 SFT pairs
2. **Weighted resampling**: kernel-high sources 5x, QA 0.3x, refusal 0.1x
3. **LoRA r=32 → r=64**: trainable params 0.81% → 1.61%
4. **seq=8192 → 16384**: truncation rate 37.6% → 19.2%

```
Script: experiments/flydsl-agent/dataprocess/enhance_sft_data.py
```

### Results

| Metric | v1 | v2 | Δ |
|------|-----|-----|-----|
| Overall | 56.3% | **72.4%** | +16.1% |
| @flyc.kernel | 40% | **92%** | +52% |
| fx.* API | 68% | **92%** | +24% |
| import flydsl | 96% | **100%** | +4% |
| L1 | 87% | **100%** | +13% |
| L3 GEMM | 64% | **76%** | +12% |
| L4 FP8 GEMM | 26% | **49%** | +23% |

### Remaining Issues

RFT Stage 1 used v2 model to generate 208 candidates → **sandbox compile pass rate 0%**.

---

## v3: Import Hallucination Fix

### Problem

RFT Stage 1 found that v2 model-generated code had **81% hallucinated import paths**:

```
v2 model-generated import paths vs real FlyDSL API:

  454x  × from flydsl.allocators import SharedAllocator
          → ✓ from flydsl.utils.smem_allocator import SmemAllocator

  299x  × from flydsl.core.fx import fx_make_variable_value_type_impl
          → ✓ import flydsl.expr as fx

  261x  × import flydsl as fx
          → ✓ import flydsl.expr as fx

  188x  × from flydsl.gpu.wmma.mma.mma_config import WmmaMmaConfig
          → ✓ from flydsl.expr.rocdl import cluster

  99x   × import flydsl as flyc
          → ✓ import flydsl.compiler as flyc

Correct import ratio: 18.6% (995 / 5349)
```

**Contradiction**: SFT training data had `import flydsl.compiler as flyc` 1375 times and
`import flydsl.expr as fx` 1570 times, but the model still generated wrong imports.
Reason: these correct imports were buried in long kernel code; the model did not learn them independently.

### Solution

Added 60 import-focused SFT samples × 3 repeats = 180 samples:

| Type | Count | Content |
|------|------|------|
| Import templates | 24 | "Write FlyDSL standard imports" → correct import block |
| Kernel skeletons | 6 | kernel framework with correct import + @flyc.kernel + @flyc.jit |
| Import correction | 30 | 15 common wrong imports → correct import + explanation |

```
Script: experiments/flydsl-agent/dataprocess/fix_import_sft.py
```

### Results

Val loss: 0.985 → 0.974 (improved). Benchmark and RFT not fully tested due to v4 data issues discovered later.

---

## v4: Hardware-Feature Mismatch Cleaning

### Problem (Most Severe)

RFT Stage 1 still had 0% compile pass rate. Deep analysis found **76% of gfx950 training samples**
used features that don't exist on gfx950 hardware in assistant code:

```
Among 2741 gfx950-related samples, 2071 (76%) have mismatches:
  × mxfp4 (FP4):    1051 — FP4 is gfx1250 (MI450) exclusive
  × tdm (TDM ops):   591 — TDM is gfx1250 exclusive
  × wmma (WMMA):     429 — gfx950 uses MFMA, not WMMA
```

### Root Cause Trace

Data pipeline `process_all_v2.py` `augmentation_hardware` logic had a bug:
it took gfx1250 kernels (with WMMA/TDM/mxfp4) and only changed the hardware name to adapt for gfx950,
without removing gfx1250-exclusive features.

```
Source analysis:
  augmentation_hardware:     415 errors ← most
  augmentation_tile:         354
  augmentation_pipeline:     226
  kernel_reverse_annotation: 159

Actual source files:
  kernels/wmma_gemm_gfx1250.py         → labeled as gfx950 (wrong!)
  kernels/moe_gemm_2stage_wmma_gfx1250.py → labeled as gfx950 (wrong!)
  kernels/gemm_fp8fp4_gfx1250.py       → labeled as gfx950 (wrong!)
```

### Hardware-Feature Compatibility Matrix

| Feature | gfx942 (MI300X) | gfx950 (MI350X) | gfx1250 (MI450) |
|---------|:---:|:---:|:---:|
| MFMA | ✅ | ✅ | ❌ |
| WMMA | ❌ | ❌ | ✅ |
| TDM | ❌ | ❌ | ✅ |
| mxfp4 (FP4) | ❌ | ❌ | ✅ |
| preshuffle | ✅ | ✅ | ❌ |
| swizzle_xor16 | ✅ | ✅ | ❌ |
| pipeline | ✅ | ✅ | ✅ |
| SmemAllocator | ✅ | ✅ | ✅ |
| split_k | ✅ | ✅ | ✅ |

### Solution

Discarded all SFT samples whose assistant code contained hardware-feature mismatches:

```
First regex version (too broad):  3096 → 1532 (discarded 50%) — falsely removed 812
  \bfp4\b matched "FP4" text in comments (e.g. "MFMA + FP4/MFMA-scale")
  \bwmma\b matched WMMA branches in arch dispatch code

Second regex version (precise):  3096 → 2344 (discarded 24%) — only matches code calls
  wmma_\w+( / WmmaAtom / from.*wmma import
  tdm_ops. / TdmCopy / tdm_load
  mxfp4_quant / fp4_gemm / wfp4 / afp4

Script: experiments/flydsl-agent/dataprocess/clean_hw_features.py
```

### v4 Candidate Error Detailed Analysis

RFT Stage 1 used v4 model to generate 192 candidates; error distribution:

| Error Type | Count | Share | Notes |
|---------|------|------|------|
| **Syntax errors** | 174 | 90.6% | |
| └ markdown not cleaned | 98 | 51% | code wrapped in ``` |
| └ invalid syntax | 110 | 57% | includes illegal characters in comments |
| └ unterminated string | 18 | 9% | |
| **Missing @flyc decorator** | 7 | 3.6% | has import but no kernel |
| **Correct structure but wrong API** | 6 | 3.1% | see cases below |
| **Imports still wrong** | 3 | 1.6% | v3 fix mostly worked, some residue |
| **Partially correct** | 2 | 1.0% | |
| **✅ Compile passed** | **1** | **0.5%** | paged_attn/gfx950 |

#### Syntax Error Details (174, 90.6%)

Biggest issue: **model output contains markdown code blocks** (104/192 = 54%):
```
Model output:
  ```python
  import flydsl.compiler as flyc
  ...
  ```

What verify.py received:
  ```python           ← this line causes SyntaxError
  import flydsl.compiler as flyc
```

verify script already had strip markdown logic (v2 fix), but couldn't match all formats.
Some candidates had markdown nested in the middle of output rather than at the start.

#### Correct Structure but Wrong API (6, 3.1%)

Case 1 — hallucinated API:
```python
import flydsl.compiler as flyc  # ✓ correct
import flydsl.expr as fx        # ✓ correct
from flydsl.allocators import SmemAllocator  # × does not exist

@flyc.kernel(
    grid_dim=(fx.ceil_div(fx.Arg("B"), 8), ...),  # × fx.Arg does not exist
    smem_bytes=fx.Arg("T") * 4 * 32,              # × fx.Arg does not exist
)
def rmsnorm(
    x: fx.DeviceArray(3, dtype=torch.float32),    # × fx.DeviceArray does not exist
```
Correct API: `fx.Tensor`, `fx.Constexpr[int]`, `from flydsl.utils.smem_allocator import SmemAllocator`

Case 2 — mixed with JAX:
```python
import flydsl.compiler as flyc  # ✓
import flydsl.expr as fx        # ✓
import jax                      # × FlyDSL doesn't use JAX
import jax.numpy as jnp         # × hallucination
```

Case 3 — hallucinated type system:
```python
import flydsl.compiler as flyc  # ✓
import flydsl.expr as fx        # ✓
import flydsl.types as ft       # × flydsl.types does not exist

@flyc.kernel
def softmax_fma(x: ft.Tensor, y: ft.Tensor):  # × ft.Tensor does not exist
```
Correct: `x: fx.Tensor`

#### The Only Candidate That Passed Compile

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def paged_attn_fwd(
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    logsumexp: fx.Tensor,
    y: fx.Tensor,
    num_heads: int,
    split_k: int,
):
    num_seqs = q.layout[0]
    ...
```
Correctly used `import flydsl.compiler as flyc`, `import flydsl.expr as fx`, `@flyc.kernel`, `fx.Tensor`.

### v4 Results

| Metric | v2 | v4 | Δ | Reason |
|------|-----|-----|-----|------|
| Overall | 72.4% | 60.3% | -12.1% | reduced dataset size |
| L1 | 100% | 100% | = | |
| L2 | 100% | 80% | -20% | layernorm/rope regression |
| L3 | 76% | 76% | = | |
| L4 | 49% | 26% | -23% | FP8 GEMM 49%→14% |
| L5 | 38% | 20% | -18% | preshuffle regression |
| @flyc.kernel | 92% | 64% | -28% | fewer complex kernel samples |
| Pipeline | 12% | 0% | -12% | pipeline samples cleaned out |
| Swizzle | 4% | 0% | -4% | swizzle samples cleaned out |
| **Sandbox compile** | **0%** | **0.5%** | **+0.5%** | **first breakthrough!** |

---

---

## v5: Add gfx950 Kernels + Gluon Tutorials + Secondary Hallucination Discovery

### Improvements

1. **+156 real gfx950 kernel code samples** — 52 pairs ×3 from 27 gfx950-compatible kernels in FlyDSL repo
2. **+18 no-markdown templates** — emphasize "output raw Python, no markdown"
3. **+18 API type corrections** — `fx.DeviceArray`→`fx.Tensor` and 9 other corrections ×2
4. **+60 Gluon GEMM optimization tutorials** — v0→v9 progressive optimization (520→1489 TFLOPS)

### Benchmark Results

| Metric | v4 | **v5** | Δ |
|------|-----|--------|-----|
| L1 | 100% | **100%** | = |
| L2 | 80% | **100%** | +20% |
| L3 | 76% | **72%** | -4% |
| L4 | 26% | **37%** | +11% |
| L5 | 20% | **38%** | +18% |
| **Overall** | **60%** | **69%** | **+9%** |

v5 improved substantially overall: L2 back to 100%, L4 +11%, L5 +18%.

### Sandbox Compile 0% — Dual Root Cause Analysis

**Original analysis (before clean_code fix)**: only 18 of 192 candidates passed static checks; rest failed on syntax errors.

**Re-analysis after clean_code fix**:

```
Syntax error root cause breakdown (192 candidates):
  Special token leakage:     126 (66%) ← <|fim_middle|>, <|file_sep|> not stripped
  Markdown wrapping:           54 (28%) ← ``` wrapping
  Code truncation:                19 (10%) ← max_new_tokens=4096 insufficient, unclosed brackets
  System prompt leakage:       5 (3%)  ← model output system prompt as code

After clean_code() fix:
  Enhanced regex: strip <|fim_*|>, <|file_sep|>, <|endoftext|>, <|im_start/end|>
  Enhanced markdown: strip ``` in middle positions
  Truncation handling: truncate at position when residual <| remains

Effect:
  Syntax pass: 18/192 (9%) → 64/192 (33%)  ← +46 candidates
  Sandbox pass: 0/18 → 0/64               ← all failed on ImportError
```

**Key finding: syntax issues can be solved at the code level without retraining.**

The real bottleneck for sandbox compile failure is **secondary API hallucination** — all 64 syntax-correct candidates failed on nonexistent imports like `from flydsl.expr import Expr`:

```
Model-generated code (top-level imports correct, secondary imports hallucinated):

  import flydsl.compiler as flyc    # ✓ correct 
  import flydsl.expr as fx          # ✓ correct
  from flydsl.expr import Expr      # × does not exist! should use fx.Tensor
  from flydsl.expr import ArithOp   # × does not exist! should use from flydsl.expr import arith
  from flydsl.expr import dtypes    # × does not exist! types are fx.Float32 etc. direct attributes
  from flydsl.utils import Layout   # × Layout is in fx.Layout, not utils

Root cause: model doesn't know FlyDSL module structure tree:
  flydsl/
  ├── compiler (flyc) — @flyc.kernel, @flyc.jit
  ├── expr (fx) — 222 public names (types/functions/layout algebra)
  │   ├── arith — arithmetic ops (addf, mulf...)
  │   ├── buffer_ops — buffer operations
  │   ├── rocdl — ROCDL/MFMA instructions
  │   └── typing — T, Vector
  ├── utils.smem_allocator — SmemAllocator, SmemPtr
  └── runtime.device — get_rocm_arch

Model knows to import from flydsl.expr but not which names exist vs don't.
Need to teach the full import chain tree — "need X → which module → how to import".
```

**Code fixes** (done, no retrain needed):
- `verify_candidates.py::clean_code()` — enhanced special token / markdown / truncation handling
- `generate_candidates.py` — `max_new_tokens` 4096 → 6144

Script: `experiments/flydsl-agent/rft-stage1/verify_candidates.py`

### Solution (v5b direction)

Generated import chain navigation data (25 pairs ×4 = 100 samples), teaching three reasoning modes:
1. **Forward navigation**: "I need SmemAllocator" → flydsl → utils → smem_allocator → SmemAllocator
2. **Module reference card**: "What's in flydsl.expr?" → full list (222 names)
3. **Reverse negation**: "flydsl.expr.Expr does not exist" → should use fx.Tensor

Script: `experiments/flydsl-agent/dataprocess/fix_import_chain.py`

---

---

## v5b: Import Chain Navigation + API Reference — Sandbox Compile Breakthrough

### Improvements

1. **+100 import chain navigation samples** — "need X → which module → how to import" (25 pairs ×4)
2. **+96 API reference + hallucination correction** — full module API lists + 17 wrong import corrections (32 pairs ×3)
3. **clean_code() code fix** — enhanced special token / markdown strip (no retrain needed)
4. **max_new_tokens 4096→6144** — avoid long kernel truncation

### Benchmark Results

| Metric | v5 | **v5b** | Δ | Target |
|------|-----|---------|------|--------|
| L1 | 100% | **100%** | = | 90% ✅ |
| L2 | 100% | **100%** | = | 85% ✅ |
| L3 | 72% | **76%** | +4% | 70% ✅ |
| L4 | 37% | **51%** | +14% | 50% ✅ |
| L5 | 38% | **55%** | +17% | 20% ✅ |
| **Overall** | **69%** | **76.5%** | **+7.5%** | **60% ✅** |

**All 5 levels met targets for the first time.** L4 37%→51% exceeded 50% target for the first time.

### Sandbox Compile: 0% → 4.7% (9/192)

9 kernels passed real FlyDSL compile, covering 9/12 operators:

| Operator | Pass Count | Lines of Code | Import Pattern |
|------|--------|---------|------------|
| layernorm | 1 | 53 | flyc + fx + arith,buffer_ops + SmemAllocator |
| gemm | 1 | 97 | flyc + fx + flyrt |
| mla | 2 | 90, 116 | flyc + fx + arith,buffer_ops |
| softmax | 1 | 34 | flyc + fx |
| flash_attn | 1 | 65 | flyc + fx |
| topk | 1 | 29 | flyc + fx |
| custom | 1 | 75 | flyc + fx + typing.Tensor |
| paged_attn | 1 | 118 | flyc + fx + arith,buffer_ops,rocdl |

Operators not passing: moe (0), rmsnorm (0), quant (0), rope (0)

### Remaining Error Analysis

```
192 candidate error distribution (v5b):
  Syntax errors (still after clean_code):  98 (51%)  ← need generation-stage fix
  Static pass but sandbox ImportError:    85 (44%)  ← new variant hallucinations
  Sandbox compile passed:                   9 (4.7%) ← breakthrough!

New variant hallucinations (85 candidates):
  5x  from flydsl.expr import fx              → should use import flydsl.expr as fx
  5x  import flydsl.expr.ops as ops           → flydsl.expr.ops does not exist
  6x  from flydsl.expr.types import F16/BF16  → flydsl.expr.types does not exist, use fx.Float16
  2x  from flydsl.expr import dtypes          → old issue residue
  2x  from flydsl.runtime import rocdl        → rocdl is in flydsl.expr.rocdl
  2x  from flydsl.utils import div_up         → div_up not in utils
```

### v5c Direction

1. **Generation-stage fix**: add stop tokens to avoid special token leakage, reduce 51% syntax errors
2. **New variant correction data**: target `flydsl.expr.types`, `flydsl.expr.ops`, `from flydsl.expr import fx`
3. **Increase candidate count N=32**: improve per-spec compile pass probability

---

## Key Lessons by Version

| Version | Lesson |
|------|------|
| v1 | Training data content ratio directly determines model capability direction |
| v2 | LoRA rank and seq_length are hard constraints — insufficient means insufficient |
| v3 | Model can learn concepts but not necessarily correct call paths — need dedicated correction data |
| v4 | Wrong data is more harmful than no data — 76% mismatch data taught impossible code patterns |
| v4 | After cleaning, dataset halved causing advanced capability regression — need to replenish with correct data |
| v5 | Hallucinations are layered: v3 fixed level-1 import paths, v5 found level-2 API name hallucinations |
| v5 | Module structure knowledge can't be learned implicitly from code examples — need explicit chain navigation teaching |
| v5b | Import chain navigation data directly effective — sandbox pass rate 0%→4.7%, proves explicit module structure teaching works |
| v5b | Hallucinations are a long-tail problem — fixing old ones (Expr/dtypes) spawns new ones (expr.types/expr.ops) |
| v5b | 66% of syntax errors are post-processing issues, fixable at code level without retrain |

---

## v5c: Full Module Structure Digest (30 Modules) — Sandbox Compile Breakthrough Again

### Improvements

1. **+111 full module structure digest samples** — covers all 30 FlyDSL modules (compiler/*, expr/*, runtime/*, utils/*)
   - 12 kernel generation (system prompt embeds full module tree) ×3 repeat
   - 5 module QA reference (per-module API lists) ×3 repeat
   - 20 negative examples (explicit negation of high-frequency hallucination paths) ×3 repeat
2. **MODULE_DIGEST expansion** — from flydsl.expr to all compiler/expr/runtime/utils submodules
3. **"DOES NOT EXIST" list expansion** — 20 high-frequency hallucination paths (including `from flydsl.expr import fx`, `flydsl.expr.types`, etc.)

### Benchmark Results

| Metric | v5b | **v5c** | Δ | Target |
|------|------|---------|------|--------|
| L1 | 100% | **100%** | = | 90% ✅ |
| L2 | 100% | **100%** | = | 85% ✅ |
| L3 | 76% | **80%** | +4% | 70% ✅ |
| L4 | 51% | **43%** | -8% | 50% ❌ |
| L5 | 55% | **45%** | -10% | 20% ✅ |
| **Overall** | **76.5%** | **73.6%** | **-2.9%** | **60% ✅** |

L3 improved to 80%, but L4/L5 regressed. Module digest data may have diluted advanced kernel sample weight.

### Sandbox Compile: 4.7% → 10.4% (20/192)

**Sandbox compile pass rate doubled!** 20 kernels passed compile, covering 9/12 operators:

| Operator | v5b | **v5c** | Δ |
|------|-----|---------|---|
| gemm | 1 | **3** | +2 |
| layernorm | 1 | **3** | +2 |
| flash_attn | 1 | **3** | +2 |
| paged_attn | 1 | **3** | +2 |
| mla | 2 | **2** | = |
| moe | 0 | **2** | +2 |
| topk | 1 | **2** | +1 |
| softmax | 1 | **1** | = |
| custom | 1 | **1** | = |
| rmsnorm | 0 | **0** | = |
| quant | 0 | **0** | = |
| rope | 0 | **0** | = |

### Remaining Error Analysis (172 failed candidates)

#### Static Failures (53/192, 27.6%)

| Reason | Count |
|------|------|
| Syntax errors | 50 |
| Code too short (<15 lines) | 3 |

#### Sandbox Failures — Detailed Classification (119/192, 62%)

**Category 1: Import Style Errors (42 occurrences)**

| Error Pattern | Count | Correct Usage |
|---------|------|---------|
| `from flydsl.expr import fx` | 17 | `import flydsl.expr as fx` |
| `from flydsl.compiler import flyc` | 7 | `import flydsl.compiler as flyc` |
| `flydsl.expr.func` (does not exist) | 6 | no such module |
| `from flydsl import X` | 4 | `import flydsl.compiler as flyc` |
| `flydsl.expr._expr` (does not exist) | 3 | no such module |
| `flydsl.expr.ops` (does not exist) | 3 | `from flydsl.expr import arith` |
| `flydsl.runtime.rocm` (does not exist) | 1 | `from flydsl.expr import rocdl` |
| `flydsl.utils.gemm_test_utils` (does not exist) | 1 | no such module |

**Category 2: API Name Hallucinations (97 occurrences, most severe)**

| Error Pattern | Count | Correct Usage |
|---------|------|---------|
| `flyc.kernel_context` | 13 | flyc only has `.kernel/.jit/.compile` |
| `from flydsl.expr import types` | 6 | types are fx.* direct attributes |
| `from flydsl.expr import expr` | 4 | does not exist |
| `from flydsl.expr import utils` | 4 | does not exist |
| `from flydsl.expr import dtypes` | 4 | use fx.Float32 etc. |
| `from flydsl.expr import memory` | 3 | does not exist |
| `from flydsl.expr import atomics` | 3 | does not exist |
| `from flydsl.expr import type_traits` | 3 | does not exist |
| `flyc.SmemAllocator` | 3 | `from flydsl.utils.smem_allocator import SmemAllocator` |
| `flyc.get_shared_memory` | 2 | does not exist |
| `flyc.launch` | 2 | does not exist |
| Others (`enums`, `ir`, `f32`, `smem`) | 2 each | does not exist |

**Category 3: External Dependencies / Other (10 occurrences)**

| Error Pattern | Count |
|---------|------|
| Uses flyc.* but no import | 6 |
| `import jax` | 3 |
| `import triton` | 1 |

### Core Conclusions

1. **Module digest effective** — sandbox compile rate 4.7%→10.4% (doubled), proves teaching module structure works
2. **New hallucinations keep emerging** — fixing old ones (types/ops) spawns new ones (atomics/enums/type_traits/kernel_context)
3. **Two core error patterns unresolved**:
   - `from flydsl.expr import fx` (17×) — should be `import flydsl.expr as fx`
   - `from flydsl.compiler import flyc` (7×) — should be `import flydsl.compiler as flyc`
4. **flyc API hallucinations** — model thinks flyc has `kernel_context`/`SmemAllocator`/`launch` etc.

### v5d Direction

1. **Expand DOES NOT EXIST list** — add all hallucination names discovered in v5c
2. **Increase positive-example repetition** — more correct usage of `import flydsl.expr as fx` / `import flydsl.compiler as flyc`
3. **flyc API correction** — explicitly teach flyc only has `.kernel/.jit/.compile` three attributes
4. **Import style correction** — specifically target `from X import Y` vs `import X as Y` patterns

---

## v5d: Expanded Negative Examples + Correction — Whack-a-Mole Failure

### Improvements

1. **Negative examples expanded 20→58** — added atomics/enums/type_traits/func/ir/kernel_context/rocm etc.
2. **Import correction +7 pairs** — `from flydsl.expr import fx` → `import flydsl.expr as fx` etc.
3. **flyc API boundary + QA** — explicitly flyc only has kernel/jit/compile
4. **Correct kernel skeletons +6** — gemm/softmax/rmsnorm/rope/flash_attn/topk full code

### Sandbox Results: 19/192 (9.9%)

| Operator | v5c | v5d | Δ |
|------|-----|-----|---|
| rmsnorm | 0 | **4** | +4 ✅ |
| rope | 0 | **2** | +2 ✅ |
| quant | 0 | **1** | +1 ✅ |
| topk | 2 | **3** | +1 |
| paged_attn | 3 | 2 | -1 |
| gemm | 3 | 1 | -2 |
| layernorm | 3 | 1 | -2 |
| flash_attn | 3 | 1 | -2 |

Three previously zero-pass operators (rmsnorm/rope/quant) passed for the first time, but other operators regressed.

### Key Finding — Whack-a-Mole Effect

| Hallucination Pattern | v5c | v5d | Notes |
|---------|-----|-----|------|
| `flyc.kernel_context` | 13 | **0** | eliminated ✅ |
| `flyc.load` | 0 | **16** | newly emerged ❌ |
| `flyc.barrier` | 0 | **3** | newly emerged ❌ |
| `from flydsl.expr import kernel` | 0 | **5** | newly emerged ❌ |
| `from flydsl.expr import fx` | 17 | **17** | stubbornly unchanged ❌ |

**Conclusion: negative-example strategy is essentially whack-a-mole — eliminate one known hallucination, model invents a new one.**
`from flydsl.expr import fx` (17×) completely unaffected by correction data, showing model preference for
the generic Python pattern `from X import Y`; insufficient data volume to flip the preference.

### Strategy Pivot (v5e)

Stop adding negative examples (bottomless pit); instead greatly increase correct kernel code share:
- Extract 247 deduplicated correct kernels, 3 prompt variants each → +741 samples
- 8 mini-kernel templates ×5 → +40 samples (short and precise, focused on correct imports)
- flyc boundary QA ×3 → +6 samples
- Correct kernel ratio: 35% → **52%**

Script: `experiments/flydsl-agent/dataprocess/boost_correct_kernels.py`

---

## Key Lessons by Version

| Version | Lesson |
|------|------|
| v1 | Training data content ratio directly determines model capability direction |
| v2 | LoRA rank and seq_length are hard constraints — insufficient means insufficient |
| v3 | Model can learn concepts but not necessarily correct call paths — need dedicated correction data |
| v4 | Wrong data is more harmful than no data — 76% mismatch data taught impossible code patterns |
| v4 | After cleaning, dataset halved causing advanced capability regression — need to replenish with correct data |
| v5 | Hallucinations are layered: v3 fixed level-1 import paths, v5 found level-2 API name hallucinations |
| v5 | Module structure knowledge can't be learned implicitly from code examples — need explicit chain navigation teaching |
| v5b | Import chain navigation data directly effective — sandbox pass rate 0%→4.7%, proves explicit module structure teaching works |
| v5b | Hallucinations are a long-tail problem — fixing old ones (Expr/dtypes) spawns new ones (expr.types/expr.ops) |
| v5b | 66% of syntax errors are post-processing issues, fixable at code level without retrain |
| v5c | Module structure digest doubled sandbox compile rate (4.7%→10.4%), but hallucinations are bottomless |
| v5c | Core import pattern (`import X as Y` vs `from X import Y`) needs heavy positive-example repetition to override |
| v5d | Negative-example strategy is whack-a-mole — eliminate old hallucinations (kernel_context), new ones emerge (flyc.load/barrier) |
| v5d | `from flydsl.expr import fx` stubbornly unchanged — model prefers generic Python pattern, need positive-example flooding to flip |
| **v5e** | **Positive-example flooding works — kernel ratio 35%→52% doubled sandbox compile rate from 9.9% to 21.9%** |
| **v5e** | **`from flydsl.expr import fx` still stubborn (17×), but verified: even if import fixed, all fail on other hallucinations — SFT ceiling reached** |
| **v5e** | **21.9% sandbox pass rate provides sufficient positive signal for RFT; SFT phase complete** |

---

## v5e: Positive-Example Flooding Strategy — Final SFT Version

### Improvements

Strategy shift: from exhaustive negative examples (whack-a-mole) to greatly increasing correct kernel code share.

1. **+741 boosted kernels** — 247 deduplicated correct kernels ×3 prompt variants
2. **+40 mini-kernels** — 8 compact templates ×5 repeats (focused on correct import block)
3. **+6 flyc boundary QA** — explicitly flyc only has kernel/jit/compile
4. **+270 module digest** — reuse v5d full negative list + QA

Dataset: 2832 (v5b base) + 787 (boost) + 270 (module digest) = **3889 samples**
Correct kernel ratio: 35% → **52%**

### Benchmark Results

| Metric | v5b | v5c | v5d | **v5e** | Target |
|------|------|------|------|---------|--------|
| L1 | 100% | 100% | — | **100%** | 90% ✅ |
| L2 | 100% | 100% | — | **100%** | 85% ✅ |
| L3 | 76% | 80% | — | **72%** | 70% ✅ |
| L4 | 51% | 43% | — | **49%** | 50% ❌ (1% short) |
| L5 | 55% | 45% | — | **50%** | 20% ✅ |
| **Overall** | **76.5%** | **73.6%** | — | **74.1%** | **60% ✅** |

### Sandbox Compile: 9.9% → 21.9% (42/192)

**Sandbox compile rate doubled again!** Covers **11/12 operators**:

| Operator | v5c | v5d | **v5e** |
|------|-----|-----|---------|
| mla | 2 | 1 | **6** |
| moe | 2 | 1 | **5** |
| softmax | 1 | 1 | **5** |
| rope | 0 | 2 | **5** |
| custom | 1 | 1 | **5** |
| gemm | 3 | 1 | **4** |
| layernorm | 3 | 1 | **4** |
| topk | 2 | 3 | **3** |
| rmsnorm | 0 | 4 | **2** |
| flash_attn | 3 | 1 | **2** |
| quant | 0 | 1 | **1** |
| paged_attn | 3 | 2 | **0** |
| **Total** | **20** | **19** | **42** |

### Remaining Error Analysis

```
192 candidate distribution:
  Static pass:    134 (69.8%)  ← improved vs v5d (124)
  Sandbox pass:     42 (21.9%)  ← 2.2× v5d (19)
  Sandbox fail:     92 (47.9%)

Stubborn import pattern (17× in v5c/v5d/v5e):
  from flydsl.expr import fx: 17×
  → Verified: even if this line is fixed, all 17 candidates fail on subsequent hallucinations (import rocdl / import arith etc.)
  → Conclusion: not worth continuing to fix at SFT level

flyc attribute hallucinations (new batch):
  flyc.Stage: 13, flyc.SmemAllocator: 12, flyc.stage: 7
  flyc.if_: 6, flyc.build: 5, flyc.launch: 3
  → Whack-a-mole continues: old ones (kernel_context/load) eliminated, new ones (Stage/if_) emerge
```

### SFT Phase Conclusion

**SFT phase complete here.** Key data points:
- Overall 74.1% (stable in 73-76% range; diminishing returns from further SFT iteration)
- Sandbox compile 21.9% (2× the 10% target, sufficient positive signal for RFT)
- 11/12 operator coverage
- Hallucination problem hit SFT ceiling — need RL (compile feedback) to break through further

**Next step**: transition to RL phase (RFT → DAPO)
- Stage A: use v5e model to generate N=16 candidates → sandbox verify → diversity-preserving RFT
- Stage B: Single-Turn DAPO (compile/correctness reward)
- Stage C: Multi-Turn DAPO + PrimeEcho (performance optimization)

Script: `experiments/flydsl-agent/dataprocess/boost_correct_kernels.py`

## File Inventory

| File | Description |
|------|------|
| `sft/eval_sft.py` | 25-question 5-level benchmark |
| `sft/train_sft.py` | FSDP2 SFT trainer |
| `sft/dataset.py` | SFT dataset + answer-only loss masking |
| `sft/config_sft.sh` | training hyperparameter config |
| `sft/run_sft.sh` | Docker training launch script |
| `cpt/export_hf.py` | DCP→HF format export |
| `dataprocess/enhance_sft_data.py` | v2: kernel extraction + resampling |
| `dataprocess/fix_import_sft.py` | v3: import correction data |
| `dataprocess/clean_hw_features.py` | v4: hw-feature mismatch cleaning |
| `dataprocess/enhance_sft_v5.py` | v5: gfx950 kernel + no-markdown + API correction |
| `dataprocess/add_gluon_tutorials.py` | v5: Gluon GEMM tutorials |
| `dataprocess/fix_import_chain.py` | v5b: import chain navigation |
| `dataprocess/fix_api_hallucination.py` | v5b: API reference + hallucination correction |
| `dataprocess/add_module_digest.py` | v5c/v5d: full module structure digest (30 modules) |
| `dataprocess/boost_correct_kernels.py` | v5e: positive-example flooding (247 kernel ×3 + 8 mini ×5) |
| `rft-stage1/generate_candidates.py` | RFT candidate generator |
| `rft-stage1/verify_candidates.py` | sandbox verifier |
| `sandbox/Dockerfile` | FlyDSL-Gym sandbox |
