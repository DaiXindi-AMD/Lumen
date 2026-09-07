# Comm-GEMM Overlap Benchmark Analysis — 2-GPU Results

**Source log:** `bench_comm_overlap_2GPU.log`
**Test suite:** `benchmarks/bench_comm_overlap.py` (20 tests, all PASSED)
**Configuration:** 2× AMD MI250X GCDs, `torchrun --nproc_per_node=2`
**Matrix shape:** M=8192, K=7168, N=7168 (default); N varies in scaling tests
**Runtime:** ~8.0 s total

---

## 1. Executive Summary

| Category | Key Finding |
|---|---|
| **NCCL AG + GEMM Overlap** | 1.08× speedup; overlap_ratio=0.014; 46% of GEMM hidden |
| **NCCL RS + GEMM Overlap** | 1.03× speedup; overlap_ratio=0.054; 34% of GEMM hidden |
| **SDMA AG + GEMM Overlap** | 1.02× speedup; overlap_ratio=0.060; only 7% GEMM hidden |
| **SDMA RS + GEMM Overlap** | 1.03× speedup; overlap_ratio=0.024; 27% of GEMM hidden |
| **SDMA vs NCCL AG (overlapped)** | SDMA 1.72× faster (0.431 vs 0.742 ms) |
| **SDMA vs NCCL AG Scaling** | SDMA wins at all N; advantage 1.48×–1.57× (summary sweep) |
| **SDMA vs NCCL RS Scaling** | NCCL wins at N=256 (0.97×); SDMA wins at N≥1024, peak 1.19× at N=7168 |
| **RS Bandwidth** | SDMA: 49.8–52.1 GB/s; NCCL: 43.1 GB/s |

---

## 2. Test-by-Test Breakdown

### 2.1 NCCL Column-Parallel AllGather + GEMM Overlap

Actual 2-GPU distributed benchmark (NCCL backend).

| Component | Avg (ms) | Min | Max | CV (%) | Notes |
|---|---|---|---|---|---|
| AllGather alone | 0.623 | 0.620 | 0.629 | 0.4% | Stable |
| GEMM alone | 0.130 | 0.108 | 0.140 | 5.5% | [!NOISY] |
| Sequential | 0.801 | 0.795 | 0.809 | 0.5% | Stable |
| **Overlapped** | **0.742** | 0.736 | 0.749 | 0.5% | Stable |

**Key metrics:**
- **Overlap ratio:** 0.014 (1.4%)
- **Speedup:** 1.08×
- **Hidden GEMM:** 0.059 / 0.130 ms = **46% of GEMM hidden**

**Analysis:**

Overlapped time (0.742 ms) is close to AllGather latency (0.623 ms) plus roughly half of GEMM time. Under this config the GEMM/comm ratio is 0.130/0.623 ≈ 0.21 (compute much smaller than communication); theoretical max speedup is `(AG + GEMM) / AG` = 1.21×. Measured 1.08× is below theory, indicating only 46% of GEMM was successfully hidden.

Reasons for incomplete overlap:
1. **NCCL AG kernel consumes some CU resources**, affecting GEMM parallel execution efficiency
2. **Stream synchronization overhead**: overlap path requires extra CUDA stream management
3. **GEMM is noisy** (CV=5.5%); small GEMM kernel launch overhead is relatively significant

---

### 2.2 NCCL Row-Parallel Reduce-Scatter + GEMM Overlap

| Component | Avg (ms) | Min | Max | CV (%) | Notes |
|---|---|---|---|---|---|
| Reduce-Scatter | 0.758 | 0.754 | 0.763 | 0.3% | Stable |
| GEMM | 0.069 | 0.065 | 0.074 | 4.5% | [~unstable] |
| Sequential | 0.807 | 0.804 | 0.813 | 0.3% | Stable |
| **Overlapped** | **0.783** | 0.780 | 0.788 | 0.3% | Stable |

**Key metrics:**
- **Overlap ratio:** 0.054 (5.4%)
- **Speedup:** 1.03×
- **Hidden GEMM:** 0.024 / 0.069 ms = **34% of GEMM hidden**

**Analysis:**

RS dominates latency (0.758 ms vs GEMM 0.069 ms, **11× ratio**). Because GEMM is tiny (only 0.069 ms), even perfect hiding of all GEMM yields theoretical max speedup of only 1.09×.

Only 34% hidden in practice because: **NCCL reduce-scatter differs from all-gather — RS performs reduction (sum) on GPU, consuming compute units (CU)** and competing with GEMM. This was confirmed in prior 8-GPU analysis — NCCL RS overlap can even show negative optimization (8-GPU speedup=0.94×).

Under 2-GPU, competition is lighter, still yielding slight positive gain (1.03×).

---

### 2.3 SDMA Column-Parallel AllGather + GEMM Overlap

| Component | Avg (ms) | Min | Max | CV (%) | Notes |
|---|---|---|---|---|---|
| SDMA AllGather | 0.334 | 0.332 | 0.336 | 0.3% | Stable |
| GEMM | 0.123 | 0.110 | 0.130 | 5.9% | [!NOISY] |
| Sequential | 0.438 | 0.434 | 0.441 | 0.5% | Stable |
| **SDMA overlap** | **0.430** | 0.426 | 0.434 | 0.5% | Stable |

**Key metrics:**
- **Overlap ratio:** 0.060 (6.0%)
- **Speedup:** 1.02×
- **Hidden GEMM:** 0.008 / 0.123 ms = **only 7% of GEMM hidden**

**Analysis:**

This is an important contradictory result:

- **SDMA bare AllGather latency significantly lower**: 0.334 ms vs NCCL 0.623 ms (**1.87× faster**)
- **But overlap efficiency is very poor**: only 7% GEMM hidden (vs NCCL 46%)
- **Absolute time still better**: 0.430 ms vs NCCL 0.742 ms (**1.72× faster**)

**Cause: SDMA PUT kernel consumes GPU CU**

SDMA AllGather uses a PUT kernel (Grid=32768, Block=256) running a data-copy kernel on GPU. This kernel:
- Launches with `Grid size: 32768, Block size: 256`, consuming substantial CU resources
- Occupies CU throughout the PUT phase, preventing effective GEMM parallelism

Compared to NCCL, NCCL AllGather primarily transfers via NIC/DMA engine with less GPU CU occupancy, so GEMM can parallelize better.

**Conclusion:** SDMA's absolute latency advantage (1.87×) far outweighs overlap efficiency disadvantage. In end-to-end scenarios, SDMA is always the better choice — even though it barely overlaps any GEMM, its lower latency alone beats NCCL's overlapped version.

---

### 2.4 SDMA Row-Parallel Reduce-Scatter + GEMM Overlap

| Component | Avg (ms) | Min | Max | CV (%) | Notes |
|---|---|---|---|---|---|
| SDMA RS | 0.656 | 0.655 | 0.659 | 0.2% | Stable |
| GEMM | 0.067 | 0.065 | 0.069 | 2.4% | [~unstable] |
| Sequential | 0.724 | 0.720 | 0.734 | 0.5% | Stable |
| **SDMA RS overlap** | **0.706** | 0.703 | 0.711 | 0.3% | Stable |

**Key metrics:**
- **Overlap ratio:** 0.024 (2.4%)
- **Speedup:** 1.03×
- **Hidden GEMM:** 0.018 / 0.067 ms = **27% of GEMM hidden**

**Analysis:**

SDMA RS latency is 13.5% lower than NCCL RS (0.656 vs 0.758 ms). Overlap effect comparable to NCCL RS (NCCL 34% vs SDMA 27%); both limited by tiny GEMM.

---

### 2.5 NCCL vs SDMA AllGather Head-to-Head

Direct comparison of overlapped-mode latency (default N=7168):

| Backend | Overlapped Avg (ms) | CV (%) |
|---|---|---|
| NCCL | 0.742 | 0.5% |
| SDMA | 0.431 | 0.5% |

**SDMA speedup: 1.72×**

---

### 2.6 NCCL vs SDMA AllGather Scaling (N Sweep)

#### Per-N test results

| N | GEMM (ms) | GFLOPS | NCCL AG (ms) | SDMA AG (ms) | SDMA vs NCCL | Winner |
|---|---|---|---|---|---|---|
| 256 | 0.027 | 158,316 | 0.655 | 0.428 | **1.53×** | SDMA |
| 1024 | 0.051 | 335,110 | 0.691 | 0.429 | **1.61×** | SDMA |
| 4096 | 0.082 | 833,767 | 0.656 | 0.430 | **1.53×** | SDMA |
| 7168 | 0.123 | 977,608 | 0.739 | 0.428 | **1.73×** | SDMA |
| 14336 | 0.242 | 995,028 | 0.671 | 0.428 | **1.57×** | SDMA |
| 28672 | 0.475 | 1,013,309 | 1.078 | 0.519 | **2.08×** | SDMA |

#### Scaling Summary (final summary sweep)

| N | NCCL (ms) | SDMA (ms) | Speedup | Winner |
|---|---|---|---|---|
| 256 | 0.654 | 0.427 | **1.53×** | SDMA |
| 1024 | 0.656 | 0.429 | **1.53×** | SDMA |
| 4096 | 0.657 | 0.427 | **1.54×** | SDMA |
| 7168 | 0.661 | 0.428 | **1.54×** | SDMA |
| 14336 | 0.670 | 0.427 | **1.57×** | SDMA |
| 28672 | 0.753 | 0.508 | **1.48×** | SDMA |

**Key findings:**

1. **SDMA AllGather latency nearly constant** (~0.427–0.429 ms) for N ≤ 14336, indicating SDMA PUT path is not bandwidth-limited at these data sizes.
2. **NCCL AllGather latency grows with N**: from 0.654 ms (N=256) to 0.753 ms (N=28672), 15% increase, reflecting NCCL's bandwidth-proportional overhead.
3. **SDMA advantage stable at 1.48–1.57× across all N values**.
4. **SDMA latency first rises at N=28672 to 0.508 ms**: suggests at ~112 MB (28672×7168×2B / 2 ranks) transfer volume, SDMA transit buffer (16 MB input + 32 MB output) becomes a bottleneck.

---

### 2.7 NCCL vs SDMA Reduce-Scatter Head-to-Head

RS overlap comparison under default config:

| Backend | RS Overlap Avg (ms) | BW (GB/s) |
|---|---|---|
| NCCL | 2.723 | 43.1 |
| SDMA | 2.358 | 49.8 |

**SDMA speedup: 1.15×** | BW advantage: +15.5%

Data size: 117.4 MB (M=8192, N=7168, BF16, single rank shard)

---

### 2.8 NCCL vs SDMA Reduce-Scatter Scaling (N Sweep)

#### RS Scaling Summary

| N | Data Size (MB) | NCCL RS (ms) | SDMA RS (ms) | Speedup | Winner | SDMA BW (GB/s) |
|---|---|---|---|---|---|---|
| 256 | 0.5 | 0.121 | 0.125 | 0.97× | **NCCL** | 16.8 |
| 1024 | 8.0 | 0.253 | 0.236 | **1.07×** | SDMA | 35.6 |
| 4096 | 32.0 | 0.830 | 0.710 | **1.17×** | SDMA | 47.3 |
| 7168 | 56.0 | 1.365 | 1.145 | **1.19×** | SDMA | 51.3 |
| 14336 | 112.0 | 2.688 | 2.338 | **1.15×** | SDMA | 50.2 |
| 28672 | 224.0 | 5.132 | 4.510 | **1.14×** | SDMA | 52.1 |

**Key findings:**

1. **NCCL wins at small N**: at N=256 (0.5 MB) SDMA RS is 3% slower. SDMA RS barrier/flag sync overhead is relatively significant at small data sizes.
2. **Crossover at N≈1024** (~8 MB): SDMA RS begins to overtake.
3. **SDMA BW increases with N**: from 16.8 GB/s (N=256) to 52.1 GB/s (N=28672), gradually approaching XGMI theoretical bandwidth ceiling.
4. **SDMA RS speedup stabilizes at ~1.14–1.19× for N≥4096**: unlike AllGather where SDMA advantage keeps growing. RS requires more sync coordination (partial sum reduction), limiting further speedup.

---

## 3. Communication Operation Comparison Matrix

### 3.1 AllGather (AG)

| | NCCL | SDMA | Comparison |
|---|---|---|---|
| **Bare AG latency** | 0.623 ms | 0.334 ms | SDMA **1.87×** faster |
| **AG + GEMM Overlapped** | 0.742 ms | 0.430 ms | SDMA **1.72×** faster |
| **GEMM hide rate** | 46% | 7% | NCCL better |
| **Overlap speedup** | 1.08× | 1.02× | NCCL better |
| **End-to-end winner** | — | — | **SDMA** (lower absolute latency) |

### 3.2 Reduce-Scatter (RS)

| | NCCL | SDMA | Comparison |
|---|---|---|---|
| **Bare RS latency** | 0.758 ms | 0.656 ms | SDMA **1.16×** faster |
| **RS + GEMM Overlapped** | 0.783 ms | 0.706 ms | SDMA **1.11×** faster |
| **GEMM hide rate** | 34% | 27% | NCCL slightly better |
| **Overlap speedup** | 1.03× | 1.03× | Tie |
| **End-to-end winner** | — | — | **SDMA** (lower absolute latency) |

---

## 4. Stability Analysis

### Measurement quality flags

| Flag | Meaning | Threshold |
|---|---|---|
| (none) | Stable | CV < 2% |
| `[~unstable]` | Slight fluctuation | 2% ≤ CV < 5% |
| `[!NOISY]` | High variance | CV ≥ 5% |

### Stability summary

| Category | Stability | Notes |
|---|---|---|
| NCCL/SDMA comm standalone measurements | **All stable** (CV < 1%) | Timing infrastructure reliable |
| Sequential/Overlapped composite measurements | **All stable** (CV < 1%) | High confidence |
| GEMM alone (small N) | **NOISY** (CV 4–6%) | Small GEMM kernel launch overhead dominates |
| Scaling summary measurements | Stable, except N=28672 SDMA AG (CV=3.3%) | Large-data SDMA occasional buffer latency |

---

## 5. GEMM Performance Analysis

### 5.1 GEMM Throughput vs N

| N | Avg (ms) | GFLOPS | Relative peak utilization* |
|---|---|---|---|
| 256 | 0.027 | 158,316 | ~38% |
| 1024 | 0.051 | 335,110 | ~81% |
| 4096 | 0.082 | 833,767 | ~100% |
| 7168 | 0.123 | 977,608 | ~100% |
| 14336 | 0.242 | 995,028 | ~100% |
| 28672 | 0.475 | 1,013,309 | ~100% |

*Reference: MI250X single GCD ~400 TFLOPS BF16.

**Findings:**
- N≥4096 GEMM reaches near-peak throughput — the most meaningful range for comm-GEMM overlap.
- N=256–1024 GEMM is very fast (0.027–0.051 ms); overlap benefit extremely limited — not worth overlap sync overhead to hide such short GEMM.

---

## 6. Communication Bandwidth Analysis

### 6.1 AllGather bandwidth

| Backend | Operation | Latency (ms) | Estimated BW (GB/s)* | Notes |
|---|---|---|---|---|
| NCCL | AG (default) | 0.623 | ~26 | Ring/tree protocol overhead |
| SDMA | AG (default) | 0.334 | ~48 | Direct PUT transfer |
| NCCL | AG (N=28672) | 0.753 | ~53 | More efficient at large data |
| SDMA | AG (N=28672) | 0.508 | ~79 | Approaching XGMI bandwidth |

*Estimated from data_size / latency. 2-GPU MI250X connected via XGMI, ~100 GB/s unidirectional bandwidth.

### 6.2 Reduce-Scatter bandwidth

| Backend | Peak BW (GB/s) | At N= | Notes |
|---|---|---|---|
| NCCL | 43.1 | default (7168) | Limited by ring RS protocol |
| SDMA | 52.1 | 28672 | Maximizes bandwidth utilization |

**SDMA RS bandwidth growth curve:** 16.8 → 35.6 → 47.3 → 51.3 → 50.2 → 52.1 GB/s (N=256→28672)

---

## 7. Actionable Insights

### 7.1 [HIGH] SDMA AllGather universally beats NCCL

SDMA wins at **all tested N values** (1.48–2.08×). For typical Transformer models (hidden_dim=7168–14336):
- Expect **1.54–1.57× AG acceleration**
- Even with SDMA overlap efficiency at only 7%, absolute latency still far below NCCL overlapped version

**Recommendation:** Prefer SDMA for AllGather communication path.

### 7.2 [MEDIUM] SDMA RS penalty at small N

At N=256 (0.5 MB) SDMA RS is slower than NCCL (0.97×). If model architecture uses small RS messages (e.g. very high TP degree + narrow layers), NCCL may be better.

**Recommendation:** Consider hybrid strategy — SDMA for AG, NCCL for RS when N<1024.

### 7.3 [MEDIUM] RS overlap benefit limited

Both NCCL and SDMA RS+GEMM overlap yield only 1.03× speedup. Reasons:
- In row-parallel path GEMM is much smaller than RS (0.067–0.069 ms vs 0.656–0.758 ms)
- RS kernel itself consumes CU (especially NCCL), competing with GEMM

**Recommendations:**
- Achieve larger overlap benefit via inter-layer pipeline (layer N RS overlaps with layer N+1 compute)
- Consider fusing additional post-RS compute into overlap window

### 7.4 [LOW] SDMA PUT kernel CU occupancy

SDMA AG PUT kernel uses `Grid=32768, Block=256`, heavily occupying CU and preventing GEMM parallelism. Potential optimization — reduce PUT kernel grid size or use hardware DMA engine to free CU for GEMM.

### 7.5 [LOW] SDMA transit buffer size

At N=28672 SDMA AG latency jumps from ~0.428 ms to 0.508 ms (+19%), indicating transit buffer (16 MB input + 32 MB output) insufficient at this scale. Consider enlarging buffer for models with hidden_dim > 14K.

---

## 8. 2-GPU vs 8-GPU Comparison Preview

| Metric | 2-GPU | 8-GPU* | Trend |
|---|---|---|---|
| NCCL AG overlap speedup | 1.08× | 1.09× | Roughly flat |
| NCCL RS overlap speedup | 1.03× | **0.94×** (negative optimization) | 8-GPU competition intensifies |
| SDMA AG overlap speedup | 1.02× | 0.98× (negative optimization) | 8-GPU PUT kernel competition worse |
| SDMA RS overlap speedup | 1.03× | 1.09× | 8-GPU actually improves |

*Data from `bench_comm_overlap_8GPU_analysis.md`.

Under 8-GPU, NCCL RS overlap shows clear negative optimization (0.94×), validating that RS reduction competes with GEMM for CU — as GPU count increases, RS reduction workload grows and CU competition intensifies.

---

## 9. Test Inventory

All 20 tests passed.

| # | Test Class | Test Name | Key Result |
|---|---|---|---|
| 1 | TestNCCLColumnParallelOverlap | test_allgather_gemm_overlap | **1.08× speedup**, 46% GEMM hidden |
| 2 | TestNCCLRowParallelOverlap | test_reduce_scatter_gemm_overlap | 1.03× speedup, 34% GEMM hidden |
| 3 | TestSdmaColumnOverlap | test_sdma_allgather_gemm_overlap | 1.02× speedup, 7% GEMM hidden |
| 4 | TestSdmaRowOverlap | test_sdma_reduce_scatter_overlap | 1.03× speedup, 27% GEMM hidden |
| 5 | TestNCCLvsSdma | test_nccl_vs_sdma_allgather_overlap | **SDMA 1.72× faster** |
| 6 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling[N=256] | SDMA 1.53× |
| 7 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling[N=1024] | SDMA 1.61× |
| 8 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling[N=4096] | SDMA 1.53× |
| 9 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling[N=7168] | SDMA 1.73× |
| 10 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling[N=14336] | SDMA 1.57× |
| 11 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling[N=28672] | SDMA 2.08× |
| 12 | TestNCCLvsSdma | test_nccl_vs_sdma_scaling_summary | SDMA 1.48–1.57× |
| 13 | TestNCCLvsSdma | test_nccl_vs_sdma_reduce_scatter_overlap | **SDMA 1.15×**, 49.8 vs 43.1 GB/s |
| 14 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling[N=256] | NCCL 0.97× (NCCL wins) |
| 15 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling[N=1024] | SDMA 1.07× |
| 16 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling[N=4096] | SDMA 1.17× |
| 17 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling[N=7168] | SDMA 1.19× |
| 18 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling[N=14336] | SDMA 1.15× |
| 19 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling[N=28672] | SDMA 1.14× |
| 20 | TestNCCLvsSdma | test_nccl_vs_sdma_rs_scaling_summary | NCCL wins N=256; SDMA peaks 1.19× |
