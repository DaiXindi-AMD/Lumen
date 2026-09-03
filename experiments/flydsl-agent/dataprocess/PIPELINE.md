# FlyDSL Agent Dataset Pipeline (v5e)

> Nine SFT iterations → Overall 74.1%, sandbox compile 21.9% → transitioning to RFT

## 1. Overall Architecture

```
FlyDSL + aiter + gpu-docs
         │
    ┌────┴────┐
    ▼         ▼
 manifest  gpu-docs (confidential)
    │         │
    ▼         ▼
 CPT/SFT/RL  refusal SFT
    │
    ├── v2: kernel extraction + resampling
    ├── v3: import correction
    ├── v4: hw-feature cleanup
    ├── v5/v5b: gfx950 kernels + import chains + API reference
    ├── v5c/v5d: module structure digest + negative list
    └── v5e: correct kernel boost (positive flooding)
         │
         ▼
    SFT v5e (3,889 samples, 52% kernel, sandbox 21.9%)
         │
         ▼
    RFT Stage A (100 specs × 16 candidates → verified → short SFT)
```

| Output | Count | Description |
|------|------|------|
| SFT v5e | 3,889 train + 264 val | 52% correct kernel code, Overall 74.1% |
| RL specs | 2,563 train + 287 val | 213 gfx950 specs |
| Sandbox | 21.9% (42/192) | 11/12 operators pass |

---

## 2. Nine-Round SFT Iteration Summary

### Data Quality Issues and Fixes

| Version | Issue Found | Fix | Effect |
|------|-----------|---------|------|
| v1→v2 | 82% non-code data | kernel extraction + weighted sampling | Overall 56%→72% |
| v2→v3 | 81% import hallucination | 60 import correction ×3 | val_loss improved |
| v3→v4 | 76% hw-feature mismatch | drop gfx1250 features on gfx950 | sandbox 0%→0.5% |
| v4→v5 | data reduction caused regression | +156 real gfx950 kernels | Overall 60%→69% |
| v5→v5b | secondary API hallucination | import chain navigation + API reference | sandbox 0%→4.7% |
| v5b→v5c | unknown module structure | 30-module full digest | sandbox 4.7%→10.4% |
| v5c→v5d | new hallucinations keep appearing | expand negatives 20→58 | whack-a-mole failed (9.9%) |
| **v5d→v5e** | **negative examples bottomless** | **positive flooding: kernel ratio 35%→52%** | **sandbox 9.9%→21.9%** |

### Key Lessons

1. **Positive > negative** — v5d proved enumerating "does not exist" is whack-a-mole (eliminate `flyc.kernel_context`, `flyc.load` appears). v5e used correct kernel code flooding (247 ×3 + mini ×5) to double sandbox compile rate.
2. **`from flydsl.expr import fx` is the SFT ceiling** — 17 occurrences across v5c/v5d/v5e; model inherits Python inertia from pretraining, SFT data volume cannot flip it. Verified: even if import is fixed, all 17 candidates fail on subsequent hallucinations — not worth continuing to fix.
3. **Bad data is worse than no data** — v4 removed 24% of data, key metrics actually improved.

---

## 3. v5e Data Composition

### Source Distribution

| Source | Count | Description |
|------|------|------|
| augmentation_tile | ~266 | tile size variants |
| augmentation_pipeline | ~195 | pipeline depth variants |
| boost_correct_kernel | **741** | **247 deduplicated correct kernels ×3 prompt variants (v5e new)** |
| module_digest_negative | 174 | 58 hallucination types ×3 (v5d) |
| import_fix_template/correction | ~120 | import correction |
| kernel_reverse_annotation | ~97 | code→instruction |
| kernel_code_synthesis | ~79 | kernels extracted from CPT |
| boost_mini_kernel | **40** | **8 compact templates ×5 (v5e new)** |
| module_digest_kernel/qa | ~57 | module digest kernel + QA |
| Other | ~2120 | doc Q&A, gfx950 kernels, Gluon tutorials, etc. |
| **Total** | **3,889** | |

### Correct Import Pattern Coverage

```
import flydsl.compiler as flyc    — ~1800 samples in assistant
import flydsl.expr as fx          — ~1800 samples
@flyc.kernel                      — ~1800 samples
@flyc.jit                         — ~1600 samples
from flydsl.expr import arith     — ~800 samples
SmemAllocator                     — ~600 samples
rocdl                             — ~700 samples
```

---

## 4. Sandbox Compile Evolution

```
v2:  0/208 = 0%      — 81% import hallucination
v4:  1/192 = 0.5%    — first breakthrough
v5b: 9/192 = 4.7%    — import chains effective
v5c: 20/192 = 10.4%  — module digest doubled ✓ (>10% target)
v5d: 19/192 = 9.9%   — negatives ineffective
v5e: 42/192 = 21.9%  — positive flooding doubled again ✓✓

By operator (v5e):
  mla: 6, moe: 5, softmax: 5, rope: 5, custom: 5
  gemm: 4, layernorm: 4, topk: 3, rmsnorm: 2
  flash_attn: 2, quant: 1, paged_attn: 0
```

---

## 5. RFT Phase (In Progress)

```
v5e model (sandbox 21.9%)
    │
    ▼ generate_candidates.py (84 specs × 16 = 1344 candidates)
    │
    ▼ verify_candidates.py --use-sandbox (~290 pass)
    │
    ▼ build_rft_dataset.py (merge with v5e SFT data, ×2 repeat)
    │
    ▼ train_sft.py (1 epoch, lr=5e-6, on v5e merged model)
    │
    ▼ RFT v1 model → benchmark + sandbox eval
```

Target: sandbox compile rate > 30%

---

## 6. Script Reference

| Phase | Script | Responsibility |
|------|------|------|
| Base | `process_all_v2.py` | Main pipeline: scan→manifest→CPT/SFT/RL |
| v2 | `enhance_sft_data.py` | kernel extraction + weighted sampling |
| v3 | `fix_import_sft.py` | import correction pairs |
| v4 | `clean_hw_features.py` | hw-feature mismatch cleanup |
| v5 | `enhance_sft_v5.py` | gfx950 kernel + API correction |
| v5 | `add_gluon_tutorials.py` | Gluon GEMM tutorials |
| v5b | `fix_import_chain.py` | import chain navigation data |
| v5b | `fix_api_hallucination.py` | API reference + hallucination correction |
| v5c/d | `add_module_digest.py` | 30-module structure digest + negative list |
| v5e | `boost_correct_kernels.py` | correct kernel pattern boost |
| RFT | `rft-stage1/generate_candidates.py` | candidate generation |
| RFT | `rft-stage1/verify_candidates.py` | sandbox verification |
| RFT | `rft-stage1/build_rft_dataset.py` | build RFT dataset |

---

*v5e · 2026-06-25 · SFT phase complete, transitioning to RFT*
