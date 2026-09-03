# RFT Stage A Experiment Report — Diversity-Preserving Rejection Fine-Tuning

## v5f RFT (Current, 2026-07-06)

### Overview

Stage A RFT on top of SFT v5f (format-aligned).
v5f model generates candidates at scale for 122 gfx950 specs → FlyDSL-Gym sandbox verification → retain all compilation-passing implementations → 1 epoch training.

**HuggingFace**: [Zhangdanyang/Qwen2.5-Coder-RFT-v5f](https://huggingface.co/Zhangdanyang/Qwen2.5-Coder-RFT-v5f)

### Model Evolution Path

| Stage | API Score | Sandbox Compile Rate | L5 Expert |
|------|-----------|-----------|-----------|
| SFT v5e | 74% | 22% (12 specs) | 50% |
| SFT v5f (+format) | 74% | 38% (122 specs) | 50% |
| **RFT v5f** | **75%** | **53%** (122 specs) | **57%** |

### Three-Level Verification Results

Sandbox verifier now supports three levels: compile, run, correctness.

| Level | Passed / 1952 | Pass Rate | Description |
|------|-------------|--------|------|
| Static check | 1598 | 82% | Python syntax + FlyDSL pattern + ≥15 lines |
| **Compile** | 944 | **48%** | FlyDSL JIT import success (no import/type error) |
| **Run** | 211 | **11%** | entry point callable (no crash/OOM/timeout) |
| **Correctness** | 0 | **0%** | Output does not match PyTorch reference |

Run-but-incorrect reason distribution:

| Reason | Count | Description |
|------|------|------|
| INCORRECT | 149 | quant kernel logic error (max_diff > atol) |
| Shape mismatch | 52 | topk/rmsnorm output dimensions differ from reference |
| Returned None | 10 | in-place kernel but result not matchable |

> **Key conclusion**: The model learned FlyDSL API syntax (48% compile pass) and partial host-side structure (11% run),
> but all kernel internal compute logic is wrong (0% correct). This is expected — SFT+RFT trained only on compile pass/fail,
> with no correctness reward. **Runtime correctness is the target of RL Stage B/C.**

### Pipeline Execution

#### Step 1: Candidate Generation (v5f model)

| Item | Configuration |
|------|------|
| Model | Qwen2.5-Coder-SFT-v5f |
| Spec source | 213 gfx950 specs → sample 122 (uniform per operator) |
| Candidates per spec | N=16 |
| Total candidates | 1,952 |
| Generation temperature | temperature=0.8, top_p=0.95 |
| Prompt styles | 3 rotated (precise / natural / optimization) |
| Duration | ~23h (single GPU cuda:0) |

#### Step 2: Sandbox Verification (compile level)

| Stage | Passed | Pass Rate |
|------|--------|--------|
| Total candidates | 1,952 | 100% |
| Static check | 1,598 | 82% |
| **FlyDSL-Gym sandbox compile** | **1,026** | **53%** |
| Diversity filter | 1,026 | 53% |

Compile pass rate vs v5f SFT (pre-RFT): 38% → **53%** (+15pp)

**All 12 operator sandbox pass rate comparison**:

| Operator | v5e | v5f SFT | **RFT v5f** | Delta (RFT-v5f) |
|------|-----|---------|-------------|-----------------|
| rmsnorm | 2 (17%) | 41 (43%) | **57 (59%)** | +17pp |
| quant | 1 (8%) | 100 (37%) | **158 (58%)** | +21pp |
| softmax | 5 (42%) | 71 (49%) | **84 (58%)** | +9pp |
| layernorm | 4 (33%) | 31 (39%) | **45 (56%)** | +18pp |
| custom | 5 (42%) | 120 (44%) | **150 (55%)** | +11pp |
| topk | 3 (25%) | 48 (43%) | **58 (52%)** | +9pp |
| gemm | 4 (33%) | 96 (35%) | **137 (50%)** | +15pp |
| rope | 5 (42%) | 34 (35%) | **48 (50%)** | +15pp |
| paged_attn | 0 (0%) | 26 (32%) | **40 (50%)** | +18pp |
| mla | 6 (50%) | 43 (38%) | **55 (49%)** | +11pp |
| moe | 5 (42%) | 60 (42%) | **69 (48%)** | +6pp |
| flash_attn | 2 (17%) | 63 (23%) | **125 (46%)** | +23pp |

All 12 operators improved compile pass rate; flash_attn had the largest gain (+23pp).

#### Step 3: RFT Dataset Construction

| Item | Count |
|------|------|
| Sandbox-passing candidates | 733 |
| RFT pairs (×2 repeat) | 1,466 |
| v5f SFT data | 6,809 |
| **Total after merge** | **8,275** |
| RFT data share | 17.7% |

#### Step 4: RFT Training

| Item | Configuration |
|------|------|
| Base model | Qwen2.5-Coder-SFT-v5f (merged) |
| Training epochs | 1 epoch |
| MAX_STEPS | 1,035 |
| LR | 5e-6 |
| LoRA | r=64, alpha=128, dropout=0.05 |
| seq_length | 16384 |
| GBS | 8 |
| val_loss | 0.9615 → 0.9533 |
| Duration | ~2h (8xMI350X) |

### Benchmark (RFT vs v5f vs v5e)

| Level | v5e | v5f | **RFT v5f** | Delta (RFT-v5f) |
|-------|-----|-----|-------------|-----------------|
| L1 | 100% | 100% | **100%** | 0% |
| L2 | 100% | 100% | **100%** | 0% |
| L3 | 72% | 72% | 68% | -4% |
| L4 | 49% | 46% | **49%** | **+3%** |
| L5 | 50% | 50% | **57%** | **+7%** |
| **Overall** | 74% | 74% | **75%** | **+1%** |

| Metric | RFT | v5f | Target |
|--------|-----|-----|--------|
| Format compliance | 96% | 96% | >= 90% ✅ |
| Sandbox compilation | 100% | 100% | >= 80% ✅ |

L5 gains: blockscale_gemm +25pp, moe_2stage +12pp, preshuffle_gemm +12pp.

### Artifacts

| File | Location |
|------|------|
| v5f candidates (122 specs) | `rft-results/candidates_v5f_gfx950.jsonl` |
| RFT candidates (122 specs, RFT model) | `rft-results/candidates_rft_v5f_gfx950.jsonl` |
| Compile-level verification | `rft-results/verify_stats_rft_v5f_gfx950.json` |
| Three-level verification (incl. runtime + correctness) | `rft-results/verify_stats_rft_v5f_runtime.json` |
| RFT training data | `rft-results/rft_v5f_train.jsonl` (8,275 samples) |
| RFT training log | `rft-results/rft_v5f_train.log` |
| Benchmark | `rft-results/benchmark_rft_v5f.json` |
| HuggingFace | [Zhangdanyang/Qwen2.5-Coder-RFT-v5f](https://huggingface.co/Zhangdanyang/Qwen2.5-Coder-RFT-v5f) |

### Next Steps

SFT + RFT phase complete. Verifier supports three-level checks (compile/run/correctness).
Correctness 0% indicates RL phase correctness reward is needed to train compute logic:

- **Stage B**: Single-Turn DAPO — compile + correctness reward, 100-200 steps
- **Stage C**: Multi-Turn DAPO + HRD + PrimeEcho — 3 iteration rounds (generate→fix→optimize)

---

# RFT v1 (archived, based on SFT v5e)

### Overview

Initial RFT on SFT v5e. 84 specs, 1344 candidates.

**Result: sandbox compile rate 21.9% → 30.7%, L4 first met target (54%), 12/12 operator full coverage.**

**HuggingFace**: [Zhangdanyang/Qwen2.5-Coder-RFT-v1](https://huggingface.co/Zhangdanyang/Qwen2.5-Coder-RFT-v1)

| Metric | SFT v5e | RFT v1 | Δ |
|------|---------|--------|---|
| Overall | 74.1% | 74.6% | +0.5% |
| Sandbox compile | 21.9% | 30.7% | +8.8% |
| Operator coverage | 11/12 | 12/12 | +1 |
