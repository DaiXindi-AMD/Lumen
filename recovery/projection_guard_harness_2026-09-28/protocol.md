# Fresh MXFP4 final-layer projection-guard campaign

## Decision objective

Measure whether replacing the final whole BF16 transformer layer (`A`) with
only final-layer `o_proj + down_proj` in BF16 (`B`), or only final-layer
`down_proj` in BF16 (`C`), saves end-to-end step time while keeping short-run
validation NLL within the predeclared budget.  `lm_head` remains BF16 for all
policies and is not included in the projection names.

No timing or NLL sample from an earlier campaign enters this decision.  The
older projection smoke is design provenance only; the hardened trainer source
must be measured again from an absent campaign cache.

The campaign freezes local copies of the shared runner and inherited analyzer.
The eight rank-local policy-B CSVs under `fixtures/policy_b_smoke/` are a
CPU-only analyzer regression fixture copied from the old 2026-09-22 B smoke;
they are not timing, accuracy, cache, or launch evidence for this campaign.

## Frozen policies and order

The first run is an untimed three-step route/cache smoke using policy B.  The
five unprofiled formal arms then run in this exact order:

```text
route_guard_o_down_fresh
tail1_a1 -> guard_o_down_b1 -> guard_down_c -> guard_o_down_b2 -> tail1_a2
```

- A: `--num-layers-at-end-in-bf16 1`, no projection override.
- B: tail 0 plus `--mxfp4-last-layer-bf16-projections o_proj down_proj`.
- C: tail 0 plus `--mxfp4-last-layer-bf16-projections down_proj`.

Phase 1 runs the fresh smoke and `A1 -> B1 -> C`.  Phase 2 validates every
frozen phase-1 byte before running `B2 -> A2` and the final analyzer.  Partial
arm directories are never resumed in place.

## Frozen workload

- 8 x MI350X/gfx950, NUMA node 0; Qwen3-8B initialized from scratch.
- Sequence 8192, micro batch 2, global batch 128, accumulation 8.
- 50 optimizer steps per formal arm; `TRAIN_SAMPLES=6400`.
- Statistics use steps 11--50 only (40 paired positions).
- FSDP2 full shard, retained accumulated parameters, BF16 reduction, no
  activation checkpointing, no MXFP4 communication.
- Packed QKV and split SwiGLU enabled in every policy.
- Seed 1234; 16 validation batches / 256 samples.
- Formal arms are unprofiled; only the fresh smoke writes shape logs.

The campaign holds the GPU-exclusive lock and checks idle KFD boundaries.  It
uses one cache namespace: the smoke must create it from absence and every
formal arm must replay the same immutable cache with no online-autotune event.
Every live KFD PID must have a completely readable identity.  A missing
`comm`, executable, command line, owner, or parent PID is treated as busy; only
an exact root-owned PID-1 `/usr/local/bin/gpuagent` identity is allowlisted.

## Route invariants

The campaign-local entrypoint records route inventory only during model
construction; it does not wrap forward or backward calls.  Every rank of every
arm must report exactly:

| Policy | Unquantized `nn.Linear` modules | packed QKV | split SwiGLU |
|:--|:--|--:|--:|
| A | final-layer q/k/v/o, gate/up/down, and `lm_head` | 35 | 35 |
| B | final-layer `o_proj`, `down_proj`, and `lm_head` | 36 | 36 |
| C | final-layer `down_proj` and `lm_head` | 36 | 36 |

For all policies, `lm_head` must be exactly one BF16 module with quantization
disabled.  The smoke must additionally prove the exact nine gfx950 ASM cache
choices and identical per-rank shape inventories.

Shape CSV `calls` values are totals for the complete three-step smoke, not
per-step values.  The frozen projection-policy totals are:

| `(M, N, K)` | A | B | C |
|:--|--:|--:|--:|
| `(4096, 4096, 16384)` | 840 | 840 | 864 |
| `(4096, 12288, 16384)` | 840 | 840 | 840 |
| `(6144, 4096, 16384)` | 840 | 864 | 864 |
| `(12288, 4096, 16384)` | 1680 | 1728 | 1728 |
| `(16384, 4096, 4096)` | 2240 | 2240 | 2304 |
| `(16384, 4096, 6144)` | 840 | 864 | 864 |
| `(16384, 4096, 12288)` | 3080 | 3128 | 3128 |
| `(16384, 6144, 4096)` | 1400 | 1440 | 1440 |
| `(16384, 12288, 4096)` | 3640 | 3720 | 3720 |

Only the B row is runtime-gating in this campaign because only the fresh B
smoke enables shape logging.  All eight frozen B fixtures must satisfy it in
the CPU-only analyzer tests.

First-update training-batch hashes are compared only among formal arms with
the same `--train-samples` sampler cardinality.  The smoke is never used as a
formal-arm batch-hash reference: seeded shuffling over a different cardinality
does not preserve the first batch.  Model initialization and fixed-cardinality
validation evidence retain their independent comparisons.

## Predeclared speed gates

Series A and B are per-position midpoints of their bracketing arms; C is its
single formal process.  Each comparison `A -> B`, `B -> C`, and `A -> C` must
meet all four gates:

- mean speedup at least `1.003x`;
- median speedup at least `1.003x`;
- at least `28/40` paired wins;
- paired circular moving-block bootstrap (block 4, fixed seed, 100,000
  resamples) has a 95% mean-speedup lower bound strictly greater than 1.

A and B replicate mean and median drift must each be strictly below 3%.

The `1.003x` threshold is fixed before execution from the available Amdahl
budget: the already measured whole-layer tail1-to-tail0 change was only about
`1.018x` across seven projections, while B-to-C changes one projection.  A
`1.01x` marginal gate would therefore reject a useful single-projection change
by construction.  At a roughly 5.1-second step, `1.003x` still requires about
15 ms of mean benefit, plus the independent win-count and bootstrap gates.

## Predeclared accuracy and selection gates

The A control NLL is the mean of A1/A2 and B is the mean of B1/B2.  Both B and
C require `delta_nll <= +0.01` versus A.  C-minus-B is reported as a non-gating
risk signal.

B advances only if integrity, both replicate-drift checks, A->B performance,
and B accuracy pass.  C advances only if that B chain plus B->C and A->C
performance and C accuracy all pass.  Otherwise retain the least aggressive
policy whose complete chain passed; if integrity fails, make no selection.
This 50-step one-seed screen is not a convergence proof.

## Provenance hardening

The campaign-local runner's full dirty-tree digest is retained, but the campaign sets
a local `core.excludesFile` containing only `.codex/`.  This prevents mutable
Codex runtime ledgers from invalidating an otherwise unchanged run.  No model,
training, Lumen, AITER, config, binary, data, or harness path is excluded.

Independently, every actual source/input path printed by the shared runner is
compared byte-for-byte across arms; the aggregate source bundle, explicit
source manifest, commits, dirty trees, runtime modules, code objects, tuning
tables, model/tokenizer/data bytes, harness, cache, and policy commands all
remain fail-closed.

## Commands

```bash
/usr/bin/bash /home/xdai/profile-results/lumen-mxfp4-projection-guard-formal-fresh-20260928-XWXD4L/run_projection_guard.sh dry-run
/usr/bin/bash /home/xdai/profile-results/lumen-mxfp4-projection-guard-formal-fresh-20260928-XWXD4L/run_projection_guard.sh phase1
/usr/bin/bash /home/xdai/profile-results/lumen-mxfp4-projection-guard-formal-fresh-20260928-XWXD4L/run_projection_guard.sh phase2
```

`dry-run` validates frozen files, commits, hashes, analyzer self-tests, and
unittests without requiring `/dev/kfd` and without launching a GPU process.
Phase execution separately requires readable/writable `/dev/kfd`, readable KFD
process inventory, NUMA balancing, the GPU lock, and an idle fail-closed KFD
boundary.
