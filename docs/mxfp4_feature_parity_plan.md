# Lumen MXFP4 Training Feature Parity: Execution Plan & Tracker

By Dai, Xindi — last audit **2026-08-10** against `feature/mxfp4` @ `3ae8c92`.

---

## 0. How this document differs from the FP8 one

This mirrors the structure of *Lumen FP8 Training Feature Parity*, but the two projects are
at opposite corners of the same grid, so the emphasis is deliberately different.

| | FP8 | MXFP4 |
|---|---|---|
| Shape of the problem | Wide surface (~61 features, 80% supported), depth still to be proven | One path (linear GEMM), proven deep, surface almost empty |
| Against TE | Feature-by-feature checklist | Head-to-head at matched recipe: **already ahead on both step time and loss** |
| Biggest risk | M2 communication, M5 validation | M5 validation — **zero CI, zero multi-GPU tests** |
| Numerical reference | BF16 only | BF16 **and** torchAO (bit-exact, third-party) |

Three categories exist here that have no FP8 counterpart, and they belong near the front
rather than buried in an appendix:

1. **Quantization recipe** (§6.1). Hadamard scope, which operand gets stochastic rounding,
   the BF16 tail fraction, block size 32. These decide whether the model converges at all,
   and the ablation attributing the convergence advantage is already done (§10).
2. **Third-party numerical reference** (§3). 12/12 ops are bit-identical against torchAO,
   reproducible from a committed script. FP8 has no equivalent oracle.
3. **Tuned GEMM table maintenance** (§13). The table keys on exact M/N/K. Changing sequence
   length, micro-batch or TP width silently drops every shape back to Triton with no error.
   This is a standing operational cost, not a one-time task.

---

## 1. Current state in one paragraph

MXFP4 trains Qwen3-8B end to end on 8×MI350X through two backends (Megatron with stock
parallel linears, and FSDP2 with HF `nn.Linear`). At a matched recipe against ROCm
TransformerEngine's MXFP4 it is **1513.7 ms/step against TE's 1526.2** and converges
**0.069 nats better** at 1000 steps, with zero NaN in either arm. Everything outside the
linear layer — attention, normalization fusion, MoE, fused MLP — has no MXFP4 path. Only
TP=PP=CP=1 has ever been run. No CI executes a single MXFP4 test.

---

## 2. Roadmap (milestone timeline)

| Milestone | Name | Scope focus | Acceptance criteria (summary) | Owner | Target | Status |
|---|---|---|---|---|---|---|
| M0 | Scope & contract freeze | Decide per feature: pursue, defer (with reason), or declare out of scope | Every cell in §6 is one of SUPPORTED / PARTIAL / MISSING / DEFERRED / N/A — none blank | @Dai, Xindi | TBD | NOT STARTED |
| M1 | Core quantization & GEMM | Recipe, A4W4 GEMM dispatch, Hadamard, dual-layout quant, weight cache | Ops bit-exact vs torchAO; 8B converges; step time beats BF16 | @Dai, Xindi | 2026-08-05 | **DONE** |
| M2 | Communication | Megatron FP4 all-gather, TP comm-GEMM overlap, gradient comm quantization | FP4 all-gather A/B'd on an idle machine; overlap path covers the MXFP4 backend | @Dai, Xindi | TBD | NOT STARTED |
| M3 | Runtime | HIP graphs, activation recompute, CPU offload, torch.compile under MXFP4 | Each runs without falling back, output identical to eager | @Dai, Xindi | TBD | NOT STARTED |
| M4 | Ecosystem coverage | Attention, fused RMSNorm+quant, MoE grouped linear, fused MLP | Each targeted feature passes a functional test; deferred ones carry a written reason | @Dai, Xindi | TBD | NOT STARTED |
| M5 | Validation & regression | CI matrix, multi-GPU integration, loss regression gates, determinism | Green CI on MI350X; no regression beyond thresholds | TBD | TBD | **NOT STARTED — highest risk** |
| M6 | Release gate | Final sign-off | All preceding green; tuned-table applicability recorded in the release checklist | TBD | TBD | NOT STARTED |

M1 is done in substance; it is not signed off because M5 does not exist to sign it off with.

---

## 3. Reference baselines

Three baselines, each answering a different question. Mixing them is the most common way to
draw a wrong conclusion from a run.

| Baseline | Answers | How it is run |
|---|---|---|
| **BF16** | Is MXFP4 worth enabling at all? | Same launcher, `PRECISION=bf16`. Note it runs the *same* unfused ops elsewhere, so shared optimizations do not cancel out of the ratio |
| **ROCm TE MXFP4, matched recipe (arm D)** | Is Lumen's implementation competitive? | TE with last 5 layers BF16 + Hadamard, so only the implementation differs, not the recipe |
| **torchAO** | Are the quantization ops numerically correct? | `scripts/mxfp4_accuracy_report.py`, bit-exact comparison per op |

TE's *stock* recipe (arm B, no BF16 tail, no Hadamard) is faster still, but it is not a
parity target — it is a different numerical recipe that converges worse. Quote it only as
context.

---

## 4. Functional testing (cross-phase)

### Objectives

- Every MXFP4 op is bit-exact or within a stated SNR bound against torchAO / PyTorch.
- Switching a model between BF16 and MXFP4 changes only what the recipe says it should.
- Fallback paths (AITER absent, shape ineligible, non-gfx950) degrade correctly rather than
  silently producing wrong numbers.

### Coverage today

| Layer | State | Where |
|---|---|---|
| Op-level correctness vs torchAO, fixed seed | **Good** — ~97 parametrized cases | `tests/ops/test_quantize.py` |
| Quant/dequant lifecycle, scale, swizzle/shuffle layouts | **Good** | `tests/ops/test_quantize.py`, `tests/ops/test_mxfp4_dual_layout_shuffle.py` |
| Fusion equivalence (fused emit vs two-pass) | **Good** | `tests/ops/test_mxfp4_fused_act_scale.py`, `tests/ops/test_mxfp4_fwd_wgrad_operand.py`, `tests/quantize/test_mxfp4_weight_operand_fusion.py` |
| Weight-cache invalidation | **Good** | `tests/quantize/test_mxfp4_weight_cache_hook.py` |
| Stochastic rounding | **Partial** — unbiasedness and neighbour independence tested; no bit-exact same-seed repeat | `tests/ops/test_quantize.py` |
| Edge values (zero, NaN/Inf, overflow, extreme scale) | **Partial** — NaN/Inf only checked incidentally after a 2D roundtrip | — |
| Misaligned shapes (N not a multiple of the block) | **Partial** — eligibility is asserted, the actual quant path is never exercised | — |
| Module-level integration in Megatron-Core | **Missing** — only a static check that the weight-cache hook is installed | `tests/models/test_megatron_entrypoint_parity.py` |
| Multi-GPU (TP / SP / CP / FSDP2) | **Missing** — zero tests | — |
| End-to-end smoke vs BF16 | **Partial** — a 50-step in-process convergence check; Megatron smoke is a shell script read by hand | `tests/test_mxfp4_backward_optimization.py` |
| Fallback when a backend is unavailable | **Partial** — per-kernel only; no "AITER missing ⇒ layer falls back to BF16" test | — |
| Determinism across repeated runs | **Partial** — seeds are fixed, but autotune picks backends by timing and is never checked for cross-run stability | — |
| Cross-commit loss regression | **Missing** — done by hand in wandb | — |

`tests/ops/test_linear.py` does not include `mxfp4` in `ALL_SCALING_TYPES` /
`BWD_SCALING_TYPES`, so the linear layer's MXFP4 path is only covered indirectly.
`tests/test_mxfp4_backward_optimization.py` hard-codes a worktree path, is not a standard
pytest module, and is currently in the local `lastfailed` set — it should be fixed or folded
into the main tree.

### Acceptance thresholds

| Metric | Threshold | Note |
|---|---|---|
| Op correctness vs torchAO | Bit-exact where the recipe is deterministic; SNR ≥ stated bound where it is not | Deterministic = RTN paths with SR off |
| Loss drift vs BF16 | TBD per model scale | 8B currently +0.021 val_loss at 200 steps, +0.069 *better* than TE at 1000 |
| NaN / Inf | Zero, on every path | Already holds across all recorded runs |
| Determinism | Same seed ⇒ identical loss trajectory with SR off | Not currently enforced |
| Fallback | Every fallback either produces correct numbers or raises — never silently wrong | The one historical violation is recorded in §11 |

---

## 5. Performance testing (cross-phase)

### KPIs

Step time (median after the first 10 steps — the mean is skewed by validation steps),
tokens/s, peak VRAM, and the per-category kernel breakdown from a profiled run.

### Matrix

| Dimension | Values |
|---|---|
| Scenario | Pretrain (C4). Finetune not yet covered |
| Scale | 8×MI350X. 1 / 2 GPU not covered |
| Model | Qwen3-8B (36 layers). 0.6B for fast convergence checks |
| Config | BF16, MXFP4, TE MXFP4 (matched recipe) |
| Parallelism | TP=PP=CP=1 only |

### Thresholds

| Metric | Threshold |
|---|---|
| Step time vs BF16 | Must stay ahead; currently 1513.7 vs 2803.8 ms |
| Step time vs TE matched recipe | Must not regress below parity; currently ahead by 0.8% |
| Peak VRAM | Must fit the card with headroom; currently 150.4 of 251.7 GiB |
| Regression per change | Any change is A/B'd at 200 steps against the immediately preceding config, one variable at a time |

### Measurement rules (learned the hard way)

- Time with CUDA events over tensors that stay allocated for the whole measurement. A `/tmp`
  microbenchmark once reported 26 ms for a call the trace timed at 1.0 ms and sent a whole
  optimization down the wrong path.
- Take the **median**, not the mean.
- Re-run the TE control arm in the same session. Over 8/4–8/5 it stayed at 1526–1527 ms
  across seven runs, which is what makes the comparison trustworthy.
- Check the vendored AITER kernels are actually present. A missing `fast_transpose` cost the
  FP8 baseline 166 ms/step while only emitting warnings; the launcher now checks for this.

---

## 6. Feature parity tracker

Status vocabulary: **SUPPORTED** (implemented and exercised), **PARTIAL** (implemented but
not wired, or wired but never validated), **MISSING**, **DEFERRED** (with reason), **N/A**.

### 6.1 Quantization recipe

| Feature | TE MXFP4 | Lumen MXFP4 | Status |
|---|---|---|---|
| E2M1, block 32, E8M0 scales | Yes | 1D per-group and 2D tile scaling | SUPPORTED |
| Hadamard rotation | All operands | **WGrad's two operands only**, deterministic H16 | SUPPORTED |
| Stochastic rounding | Gradients | Gradients; RTN for activation and weight | SUPPORTED |
| BF16 tail layers | Configurable | Last ~15% by default | SUPPORTED |
| Dual-layout gradient quantization | — (folded into `cast_transpose`) | `dual_layout_quant_mxfp4`, one read emits both layouts | SUPPORTED (Lumen-only) |
| NVFP4 (E4M3 second-level scale) | `nvfp4` (TE ≥ 2.7) | `QuantFormat.FP4` is a placeholder, raises `NotImplementedError` | MISSING |

The Hadamard scope difference is a deliberate divergence, not a gap: the 4-arm ablation shows
TE's all-operand rotation does **not** beat Lumen's WGrad-only rotation on convergence.

### 6.2 Linear / GEMM

| Feature | Lumen MXFP4 | Status |
|---|---|---|
| Megatron stock `Column/RowParallelLinear` | `quant.enable()` patch — this is the official path | SUPPORTED |
| HF `nn.Linear` under FSDP2 | Same patch | SUPPORTED |
| `LumenColumn/RowParallelLinear` | The enable path now passes the resolved recipe, so `--lumen-linear` routes MXFP4 instead of silently running FP8 blockwise; unit-tested, but the launcher still defaults to the patched Megatron linears until a 200-step A/B clears the native ones | PARTIAL |
| `LumenLayerNormLinear` | GEMM side works; fused norm→quant supports FP8 delayed only | PARTIAL |
| `LumenGroupedLinear` / MoE | API accepts a scaling type, defaults to `none`, no end-to-end validation | PARTIAL |
| `LumenFusedMLP` / `LumenGatedMLP` | BF16 / FP8 branches only | MISSING |
| Gradient accumulation fusion | MXFP4 backward supports `main_grad.add_` | SUPPORTED |
| Delay wgrad (`_DeferredWgrad`) | MXFP4 WGrad can be deferred | SUPPORTED |
| FP4 activation store | Forward explicitly sets `fp8_activation_store = False` | MISSING |
| A4W4 GEMM: 3 backends, per-shape autotune, tuned table | 9 of 11 Qwen3-8B shapes reach the assembly kernel | SUPPORTED |

### 6.3 Attention

| Feature | Lumen MXFP4 | Status |
|---|---|---|
| MXFP4 DPA / MHA | No MXFP4 reference anywhere in the attention modules; only `--lumen-fp8-attn` exists | MISSING |
| Attention gradient quantization to MXFP4 | `--grad-quant-type mxfp4` now reaches the attention and norm backward paths; default stays `None` and no run has enabled it | PARTIAL |

### 6.4 Normalization

| Feature | Lumen MXFP4 | Status |
|---|---|---|
| Fused RMSNorm + MXFP4 quant | `rmsnorm_blockwise` and `rmsnorm_mxfp8` exist; no MXFP4 branch | MISSING |
| Narrow-N RMSNorm fwd/bwd (per-head QK norm) | Lumen-owned row-tiling kernels, dispatched on `N ≤ 512` | SUPPORTED |

The narrow-N kernel is not a quantization feature, but it lands on the same step time and is
one of the largest single wins recorded in §10, so it belongs in the table.

### 6.5 Cross-entropy

| Feature | Lumen MXFP4 | Status |
|---|---|---|
| MXFP4 in the loss path | The output layer is skipped by default; CE runs on BF16 logits | N/A |

### 6.6 Communication

| Feature | Lumen MXFP4 | Status |
|---|---|---|
| FSDP2 FP4 parameter all-gather (`MXFP4CommTensor`) | Implemented, 3.99× less wire traffic on paper; the **net** effect has never been isolated | PARTIAL |
| Megatron FP4 parameter all-gather | No equivalent path | MISSING |
| TP comm-GEMM overlap × MXFP4 | Overlap lives in the Lumen native parallel modules; MXFP4 runs on the stock ones | MISSING |
| Gradient communication quantization | `--grad-quant-type mxfp4` is now accepted by both the Megatron and FSDP parsers and reaches `ScalingManager._round_to_mxfp4` (round-trip measures 15.8 dB SNR); no training run has used it yet | PARTIAL |

### 6.7 Parallelism and runtime

| Feature | Lumen MXFP4 | Status |
|---|---|---|
| FSDP2 | Weight cache, FP4 `save_for_backward`, comm tensor — all exercised | SUPPORTED |
| Megatron, TP=PP=CP=1 | 8B / 36 layers / 1000 steps | SUPPORTED |
| TP > 1 | Never run. Reshapes every GEMM, so the tuned table has to be re-measured | MISSING |
| PP > 1 | BF16 tail keys on `layer_number`, so it is semantically safe; unvalidated | PARTIAL |
| Context parallel | No MXFP4 code or test | MISSING |
| HIP graphs | `set_graph_capture_mode` exists; no MXFP4-specific test | PARTIAL |
| Activation recompute / CPU offload | Generic path; recompute re-quantizes, amortized by the weight cache | PARTIAL |
| torch.compile | No test, no guarantee | MISSING |

---

## 7. Summary scorecard

| Category | Features | Supported | Partial | Missing | N/A |
|---|---|---|---|---|---|
| Quantization recipe | 6 | 5 | 0 | 1 | 0 |
| Linear / GEMM | 10 | 5 | 3 | 2 | 0 |
| Attention | 2 | 0 | 1 | 1 | 0 |
| Normalization | 2 | 1 | 0 | 1 | 0 |
| Cross-entropy | 1 | 0 | 0 | 0 | 1 |
| Communication | 4 | 0 | 2 | 2 | 0 |
| Parallelism / runtime | 8 | 2 | 3 | 3 | 0 |
| **Total** | **33** | **13 (39%)** | **9** | **10** | **1** |

The 39% is not comparable to FP8's 80%: the denominators are different feature sets, and
MXFP4's supported 39% is the part that has been benchmarked against a production competitor.

---

## 8. Lumen-only features

| # | Feature | Description |
|---|---|---|
| 1 | WGrad-only Hadamard | Rotating only WGrad's operands converges as well as TE's all-operand rotation at lower cost. Backed by a 4-arm ablation |
| 2 | Dual-layout gradient quantization | One dense read emits both the row-major (DGrad) and Hadamard-rotated transposed (WGrad) MXFP4 layouts |
| 3 | Forward-emitted WGrad activation operand | The forward writes both operands; the backward never rebuilds one from stored FP4 |
| 4 | Per-shape GEMM backend autotune | Three AITER FP4 kernels measured on first call and cached per shape |
| 5 | FP4 parameter all-gather | `MXFP4CommTensor` under FSDP2 |

---

## 9. Remaining work

### Still missing

| # | Feature | Category | Note |
|---|---|---|---|
| 1 | MXFP4 attention | Attention | |
| 2 | Fused RMSNorm + MXFP4 | Normalization | |
| 3 | `LumenFusedMLP` / `LumenGatedMLP` MXFP4 | Linear | |
| 4 | FP4 activation store | Linear | |
| 5 | Megatron FP4 parameter all-gather | Communication | |
| 6 | TP comm-GEMM overlap × MXFP4 | Communication | Blocked on wiring the native parallel linears first |
| 7 | TP > 1 validation | Parallelism | Also invalidates the tuned table |
| 8 | Context parallel | Parallelism | |
| 9 | torch.compile support | Runtime | |
| 10 | NVFP4 | Recipe | |

### Deferred

| # | Feature | Reason |
|---|---|---|
| 1 | Fusing quantization into the GEMM prologue | Requires AITER kernel changes. This is the last structural gap against TE's `cast_transpose_mxfp4_shuffled` |
| 2 | Cross-entropy / lm_head in MXFP4 | Deliberate — the output layer is the most precision-sensitive and is ~1% of GEMM time |
| 3 | MoE grouped linear end-to-end | No MoE model in the current target set; the API hook exists for when there is one |

### Still partial (implemented, not reachable)

| # | Feature | Remaining work | Cost |
|---|---|---|---|
| 1 | `--grad-quant-type mxfp4` | Wiring done. Needs a training run to find out what gradient quantization costs in loss | A 200-step A/B |
| 2 | Native parallel linears + MXFP4 | Wiring done. Needs the A/B against the patched Megatron linears before `--lumen-linear` goes back in the launcher; that is also what unblocks TP comm-GEMM overlap, which only exists on the native modules | A 200-step A/B |
| 3 | FSDP2 FP4 all-gather | Isolate with a dedicated A/B on an idle machine; port `convert_from_mxfp4_2d` to Triton if the gather turns out bandwidth-bound | Days |

---

## 10. Recently completed

Performance work, 8/3 – 8/5, all measured as 200-step medians with one variable per run.
Cumulative **1900.0 → 1513.7 ms/step (−20.3%)**, which moved TE from 1.22× ahead to 0.8%
behind. The table lists the changes that paid; two that measured neutral are omitted here and
covered below. Full ladder in `mxfp4_training_report.md` §5.7–§5.12.

| # | Change | Δ ms/step |
|---|---|---|
| 1 | Enable `--lumen-fused-rope` (the flag was never passed) | −95.2 |
| 2 | Narrow-N RMSNorm backward, row-tiling kernel | −64.3 |
| 3 | Stop drawing philox counters on RTN paths | −52.3 |
| 4 | Hadamard-16 butterfly via MFMA | −47.1 |
| 5 | Weight cache wired into the Megatron path (two A/Bs) | −25.5 |
| 6 | Fuse dequant + Hadamard + quant into one pass | −17.7 |
| 7 | Attention q/k/v passed as strided views | −15.9 |
| 8 | Cache the scale swizzle | −15.6 |
| 9 | Forward emits the WGrad activation operand | −14.6 |
| 10 | Dual-layout gradient quantization | −13.0 |
| 11 | Weight B-operand shuffle emitted by the quantizer | −12.6 |
| 12 | Attention output written in seq-major order | −8.1 |
| 13 | `gc.freeze()` after warmup | −5.1 (tail latency 11.84 s → 0.54 s per 200 steps) |

Two attempts were reverted and are recorded so they are not retried blind: pointing the QK
norm at Lumen's existing RMSNorm forward (+49.1 ms — AITER's kernel moves 256 bytes per
iteration at N=128), and loading the Hadamard tile transposed (no change — Triton folds
`tl.trans` into layout inference).

Correctness work completed earlier: 12/12 ops bit-exact vs torchAO, 8B stable for 5000 steps,
convergence attribution across a 4-arm recipe ladder, dispatch-routing and FSDP2 NaN fixes.

---

## 11. Known issues

| # | Issue | Severity | Affected | Status |
|---|---|---|---|---|
| 1 | Memory regresses rather than improves. FSDP2 8B: 15.30 → 20.90 GB peak, of which 4.8 GB is the cross-micro-batch weight cache. Megatron 8B at seq 8192: 144.0 → 150.4 GiB of 251.7, the delta being one extra FP4 copy per saved activation. BF16 master weights are retained either way, so there is no weight-storage saving to offset it | Medium | Memory footprint | Open. Caching only the pre-transposed form would recover roughly half. `LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1` trades the speed back |
| 2 | FP4 all-gather benefit unverified — the dequant path is pure PyTorch and may cost more than the bandwidth it saves on a single node | Medium | Communication | Open, see §9 |
| 3 | Tuned GEMM table silently stops applying when M/N/K change | Medium | Performance reproducibility | Mitigated by §13; no automatic detection yet |
| 4 | An untuned shape once returned garbage (0.6 dB at 64×64×128) through the default assembly config | High | Correctness | Resolved — the assembly path is gated on the tuned-shape table, and `scripts/mxfp4_tune_shapes.py` verifies bit-exactness before adding a row |
| 5 | The weight-cache invalidation hook was unregistered on the Megatron path, freezing FP4 weights at step 0 with no error | Critical | Correctness | Resolved — `install_mxfp4_weight_cache_hook()` wraps `setup_model_and_optimizer` |
| 6 | `tests/test_mxfp4_backward_optimization.py` hard-codes a worktree path, is not a standard pytest module, and currently fails locally | Low | Test hygiene | Open |

---

## 12. Phase pages

Each phase below is a candidate child page; the goal, checklist and exit criteria are given
here so the split is mechanical.

### Phase 1 — Core quantization & GEMM (M1) — DONE

**Goal.** A numerically correct MXFP4 linear layer that is faster than BF16.

Checklist: recipe ops (1D/2D RTN, SR, Hadamard, dequant, transpose) ✅ · A4W4 GEMM with
per-shape backend selection ✅ · weight cache with optimizer-step invalidation ✅ ·
dual-layout and fused-emit quantizers ✅ · BF16 tail ✅ · bit-exactness vs torchAO ✅.

Exit criteria: 8B converges over 1000 steps with zero NaN ✅ · step time beats BF16 ✅ ·
every op bit-exact or within a stated SNR bound ✅.

### Phase 2 — Communication (M2) — NOT STARTED

**Goal.** Make MXFP4 reduce wire traffic, not just compute.

Checklist: isolate the FSDP2 FP4 all-gather with the existing `--no-mxfp4-comm` flag
(`examples/qwen3/pretrain_qwen3_mxfp4.py`) on an idle machine — the flag is already there,
the A/B has simply never been run · port
`convert_from_mxfp4_2d` to Triton if the gather is bandwidth-bound · add an FP4 all-gather
path for Megatron · A/B `--grad-quant-type mxfp4`, which is now exposed but unmeasured · wire
TP comm-GEMM overlap once the native parallel linears are the default.

Debug checklist: confirm the comm tensor survives FSDP2's internal `chunk`/`copy_` · check
that `MixedPrecisionPolicy` does not upcast the subclass, which silently erases the benefit ·
compare NCCL byte counts, not step time, when attributing the gain.

Exit criteria: measured (not computed) reduction in all-gather bytes and step time · loss
unchanged.

### Phase 3 — Runtime (M3) — NOT STARTED

**Goal.** MXFP4 survives graph capture, recompute, offload and compile.

Checklist: HIP-graph-capture an MXFP4 step and diff the output against eager · confirm
recompute does not double-quantize beyond what the weight cache absorbs · run under
`torch.compile` and record which graph breaks appear.

Exit criteria: each runs without falling back; output identical to eager.

### Phase 4 — Ecosystem coverage (M4) — NOT STARTED

**Goal.** Close the surface, or write down why a cell stays open.

Checklist: MXFP4 attention · `rmsnorm_mxfp4` fused quant · fused MLP · MoE grouped linear ·
FP4 activation store.

Note before starting: measure first. Several of these sit on ops that are already a small
share of the step; the fused-MLP and attention paths in particular should be justified by a
trace before any kernel is written.

### Phase 5 — Validation & regression (M5) — NOT STARTED, highest risk

**Goal.** Make a regression impossible to merge unnoticed.

Checklist:

1. Add `mxfp4` to `tests/ops/test_linear.py`'s scaling-type lists.
2. Add edge-value tests: zero, NaN/Inf, extreme scale, shapes where N is not a multiple of
   the block (assert the documented behaviour, whether that is padding or a clear error).
3. Add a bit-exact same-seed repeat test with SR off, and a stability check on autotune's
   backend selection across runs.
4. Add a multi-GPU integration test: Megatron 8-rank, a handful of steps, loss compared to a
   recorded reference.
5. Fix or fold in `tests/test_mxfp4_backward_optimization.py`.
6. Add a CI workflow that runs the MXFP4 test set on MI350X, with the gfx950-only tests
   skipped elsewhere rather than silently passing.
7. Automate the cross-commit loss comparison — a 200-step run with a drift threshold, rather
   than reading wandb by eye.

Exit criteria: green CI on target hardware · a regression in any of the §5 thresholds fails
the build.

### Phase 6 — Release gate (M6) — NOT STARTED

Checklist: all preceding milestones green · §13 recorded in the release notes · known issues
either resolved or accepted in writing.

---

## 13. Operational note: the tuned GEMM table

`examples/qwen3/configs/qwen3_8b_a4w4_blockscale_tuned_gemm.csv` holds tuned assembly-kernel
rows for Qwen3-8B's 11 MXFP4 GEMM shapes, pointed at by `AITER_CONFIG_GEMM_A4W4`. Extending
it was worth 98 ms/step.

The rows key on **exact M/N/K**. They only fire at MBS × SEQ = 16384 with TP = 1. Any of the
following invalidates the table, with no error and no warning — the shapes simply fall back
to Triton and the step gets slower:

- changing sequence length or micro-batch size,
- changing TP width (which re-splits N and K),
- adding or resizing layers.

Re-tune with `scripts/mxfp4_tune_shapes.py`, which collects the shapes a run actually issued
from its autotune log — deriving them on paper gets Megatron's fused qkv (N=6144) and gate_up
(N=24576) wrong, and those were two of the eight originally missing shapes. Every tuned row
is verified bit-exact against the plain Triton kernel before being added.

Any performance number in this document is only valid for the configuration it was measured
at. State the shape alongside the number.
