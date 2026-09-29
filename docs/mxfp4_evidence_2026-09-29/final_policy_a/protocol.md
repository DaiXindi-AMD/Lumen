# Fresh final Policy A BF16 bracket protocol

## Purpose

This write-once campaign measures a fresh, same-source bracket:

`smoke_tail1 -> bf16_a1 -> mxfp4_policy_a -> bf16_a2`

Policy A is exactly one whole final transformer layer in BF16 (`tail1`) with
packed QKV and split SwiGLU enabled everywhere eligible. It has no final-layer
projection override. The BF16 arms use no MXFP4-only CLI flags.

This is a 50-step, one-seed screening result. Even a pass is not a default
adoption or long-horizon training-quality conclusion; that requires at least
three seeds and at least 200 steps with the registered quality analysis.

## Immutable inputs

- Lumen: branch `dev/mxfp4`, commit
  `6b9aee1569247eca20937c14319ba6adcd69e0cb`.
- AITER: commit `e35bb17f4f815903bf73598facedbb321e15af28`.
- Eight MI350X/gfx950 devices, CPU and memory bound to NUMA node 0, kernel
  automatic NUMA balancing set to `0`.
- Qwen3-8B scratch initialization; sequence 8192; microbatch 2; global batch
  128; gradient accumulation 8; seed 1234.
- FSDP2 `full_shard`, retained accumulated parameters, no gradient
  checkpointing, BF16 reduction, no MXFP4 communication.
- AITER attention, Lumen norm, fused RoPE, fused cross entropy.
- 6,400 training samples for each 50-step formal arm; 16 validation batches
  and 256 validation samples.
- Profiling disabled and
  `LUMEN_MXFP4_ACTIVATION_DESCRIPTOR_CACHE=0`.
- One campaign-local immutable MXFP4 cache namespace and one campaign-local
  empty AITER configuration cache.

The copied `run_case.sh`, `inventory_entry.py`, `analyze_tail210_base.py`, and
`provenance-git-excludes` are hash-pinned. Every arm records the exact source
bundle, full Lumen/AITER tree state excluding only `.codex/` runtime logs,
runtime `.so` set, gfx950 f4gemm directory, model/tokenizer/data inputs, tuned
tables, cache bytes, command, and environment-relevant metadata.

## Phase 1

Phase 1 requires a pristine campaign root: no cache, case directory, status,
sentinel, analysis, or completion artifact may already exist. It takes the
global GPU lock and verifies KFD is idle with fail-closed PID identity checks.

1. Run `smoke_tail1` for three steps through `inventory_entry.py`, with shape
   logging enabled and a previously absent cache namespace.
2. Validate the newly generated cache and Policy A route before continuing.
3. Run `bf16_a1` for 50 steps through the direct trainer entry.
4. Run `mxfp4_policy_a` for 50 steps through the direct trainer entry.
5. Validate the complete prefix, record the phase boundary as idle, and write
   an exact SHA256 manifest of every phase-1 artifact.
6. Write `phase1-complete.txt` only after the manifest is complete.

The smoke route must show, on every rank, 245 quantized linears and these eight
unquantized linears: final-layer Q/K/V/O, gate/up/down, and `lm_head`.
Packed QKV and split SwiGLU counts must each be 35; `lm_head` must occur once,
remain BF16, and have `_quant_enabled=False`. All nine registered Policy A
shape totals must match exactly and select ASM.

The fresh cache must be schema 6 for gfx950 with exactly nine ASM choices,
single-device decision scope/count, the protected-ASM consensus policy and
reason, switch margin 1.05, and exact kernel, split-K, manifest, and code-object
identities. Formal arms must see the same cache bytes before and after; the
candidate must load nine cached decisions without online autotuning.

## Phase 2

Phase 2 is not a continuation of a partial arm. Before taking the GPU lock it
revalidates:

- every phase-1 byte against `phase1-artifacts.sha256`;
- the phase-1 sentinel and zero exit status;
- all harness and trainer hashes;
- workload, source manifests, cache, tree, runtime module, AITER config-cache,
  and f4gemm directory identities;
- the complete prefix analyzer result.

It then verifies KFD idle, runs only `bf16_a2` for 50 steps, verifies the final
idle boundary, writes the formal completion sentinel, and runs the final
analyzer. A partial or failed phase is preserved and must be replaced by a new
unique campaign root; it is never resumed in place.

## Pairing and safety gates

- All four runs have the same pre-quantization model-init SHA256.
- Rank 0--7 first-update input/label digests match exactly across all runs.
- Rank 0--7 validation input/label digests match exactly across all runs.
- Formal LR sequences and normalized commands match; the only formal command
  difference is the registered precision stack.
- All steps, loss, gradient norm, LR, timing, memory, and validation NLL are
  present and finite.
- No OOM, NaN/Inf, skipped update, runtime/backend fallback, kernel failure,
  unexpected disabled path, online autotune, profiler, or case overlap.
- Only the exact known post-success `torch.library._del_library` teardown
  traceback is allowed.
- Every case and both phase boundaries must be KFD-idle, with the inherited
  exclusive lock verified by the runner.

## Registered result

Only formal steps 11--50 are used: 40 same-position pairs. For each step, the
BF16 control is the midpoint of `bf16_a1` and `bf16_a2`.

All gates are required:

- mean speedup `>= 1.6x`;
- median speedup `>= 1.6x`;
- Policy A wins at least `28/40` paired steps;
- fixed-seed 100,000-resample paired circular moving-block bootstrap, block
  length 4, has 95% mean-speedup lower bound `>= 1.6x`;
- BF16 A1/A2 mean drift and median drift each have absolute value strictly
  below `3%`;
- `Policy A validation NLL - mean(BF16_A1 NLL, BF16_A2 NLL) <= +0.01`;
- every integrity, source/cache/tree/runtime, route, pairing, order, no-overlap,
  exit, KFD, and log-safety check passes.

## Commands

Static verification only; launches no GPU process:

```bash
/usr/bin/bash /home/xdai/profile-results/lumen-mxfp4-final-policy-a-bf16-bracket-fresh-20260928-fOB5j0/run_final_policy_a_bf16_bracket.sh dry-run
```

GPU phases, only in an authorized environment where `/dev/kfd` is readable and
writable and the machine is idle:

```bash
/usr/bin/bash /home/xdai/profile-results/lumen-mxfp4-final-policy-a-bf16-bracket-fresh-20260928-fOB5j0/run_final_policy_a_bf16_bracket.sh phase1
/usr/bin/bash /home/xdai/profile-results/lumen-mxfp4-final-policy-a-bf16-bracket-fresh-20260928-fOB5j0/run_final_policy_a_bf16_bracket.sh phase2
```
