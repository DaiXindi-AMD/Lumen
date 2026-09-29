# Fresh BF16 / MXFP4 Policy A profiling protocol

## Purpose

This directory is a fresh-output, fail-closed harness for diagnostic profiling
of the current conservative Qwen3-8B MXFP4 policy.  It creates exactly four
arms, in this order:

```text
smoke_mxfp4_policy_a
replay_mxfp4_policy_a
profile_bf16
profile_mxfp4_policy_a
```

The first two arms build and then replay a campaign-local MXFP4 autotune cache.
The final two arms collect matched rank-0/device-0 profiles.  No cache, trace,
timing sample, route report, or analysis value from another result directory is
an input to this campaign.

Profiler wall time and raw kernel-duration sums are diagnostic evidence only.
Any end-to-end throughput claim, including the 1.6x BF16 target, must be
confirmed later by a fresh unprofiled symmetric campaign.

## Fixed workload

- 8 AMD GPUs, with the process bound to NUMA CPU node 0 and memory node 0.
- Qwen3-8B initialized from config with BF16 parameters.
- Sequence length 8192, micro-batch 2, global batch 128, accumulation 8.
- FSDP2 `full_shard`, retained accumulated parameters, no activation
  checkpointing, BF16 reduction.
- AITER attention, Lumen norm, fused RoPE, and fused cross entropy.
- MXFP4 communication disabled and activation-descriptor caching disabled.
- Seed 1234, paired model/data evidence enabled.
- Validation uses 16 batches and 256 samples.
- The eight formal updates consume exactly 1,024 training samples globally
  (`8 updates x global batch 128`); there is no unused formal-data tail.
- MXFP4 uses packed QKV, split SwiGLU, and one complete BF16 transformer tail
  layer (Policy A).
- `lm_head` remains BF16 in every MXFP4 arm.

The formal profile arms run eight optimizer steps.  Steps 1--6 are warmup;
only optimizer steps 7 and 8 are profiled.  Validation runs after the profiler
has closed at step 8.  Shape recording in the PyTorch profiler is enabled;
Python hot-path shape CSV logging and copy/contiguous monkey-patching are off.

After command normalization, the two formal commands may differ only in
precision and the MXFP4-only packed-QKV/split-SwiGLU flags.  The tail argument
is passed identically to both arms so it cannot hide another workload delta.

## Fresh cache and route gates

`smoke_mxfp4_policy_a` runs three optimizer steps from an absent cache namespace.
Every rank writes a route JSON and a shape CSV.  The smoke must establish:

- exactly 245 quantized transformer linears;
- exactly 8 BF16-skipped linears: seven in transformer layer 35 and the
  independent vocabulary projection;
- `lm_head` is BF16 and has no quantization flag;
- packed QKV and split SwiGLU are enabled on exactly 35 transformer layers;
- exactly one protected-tail QKV warning and one protected-tail SwiGLU warning
  per rank;
- no eligible original QKV/SwiGLU execution, backend fallback, OOM, NaN/Inf,
  skipped update, or kernel failure;
- on every rank, exactly 1,400 packed-QKV forwards, 1,400 split-SwiGLU
  forwards, and 840 split-SwiGLU backwards across training plus validation;
- a valid cache is created, and every rank records the same non-empty set of
  exact A4W4 shapes.

`replay_mxfp4_policy_a` runs the same three-step workload against that frozen
cache.  Online autotuning is forbidden.  The cache SHA256, cache JSON,
per-shape backend choice, ASM symbol/tile/split/manifest/code-object identity,
route counters, and rank-local shape records must remain identical.

Only after replay passes is the frozen cache copied byte-for-byte into the two
independent formal namespaces.  Each formal arm also receives its own initially
empty `AITER_CONFIG_CACHE_DIR`, which must remain byte-identical across that
arm.

## Source and execution integrity

Before launching a GPU process, the runner:

1. requires every declared output path to be absent;
2. obtains `/home/xdai/profile-results/.lumen-gpu-exclusive.lock`;
3. requires read/write access to `/dev/kfd` and readable KFD process state;
4. rejects non-service KFD clients using KFD PID plus `/proc/<pid>` owner,
   executable, command, and parent PID evidence;
5. records Lumen/AITER commits and full dirty-tree digests;
6. hashes the runner, route entrypoint, analyzer, both CPU/static test files,
   runtime source files, imported JIT modules, tuned tables,
   model/tokenizer configuration, train/validation data, AITER manifests, and
   gfx950 FP4 code objects into `source-audit.json`.

The same source, repository, workload, runtime-module, tuned-table, and f4gemm
digests are rechecked before and after every arm and at campaign postflight.
The campaign root must contain only the six frozen harness inputs at preflight;
any other pre-existing file or directory is rejected before a GPU is touched.
Cache and AITER-config-cache expectations are checked independently per arm.
Persisted ASM manifest/code-object identities use Lumen's exact 16-hex SHA256
prefix representation; the independent full f4gemm-directory digest protects
the complete artifact bytes before and after every arm.
All status and completion files are written atomically.  A partial or failed
arm is never resumed in place; its directory must be preserved under a renamed
`interrupted-*` path before a new campaign directory is prepared.

All harness-side Python invocations use `-B` and
`PYTHONDONTWRITEBYTECODE=1`.  Preflight rejects `__pycache__`, `.pyc`, and
`.pyo` artifacts, and the final manifest excludes those names defensively.

KFD workload detection is based on KFD PIDs and `/proc` identity.  `pgrep` is
only supplemental.  A disappearing stale KFD directory may be rechecked for a
short bounded interval, but a persistent non-service client fails the run.

## Standard artifacts

Each arm directory contains:

```text
train.log
trace.json                     # real trace for formal arms; explicit disabled sentinel for route arms
profile.txt                    # real table for formal arms; explicit disabled sentinel for route arms
profile_shapes.txt             # formal profile arms only
run-meta.txt
train-exit-status.txt
postflight-status.txt
kfd-before.txt
kfd-prelaunch.txt
kfd-after.txt
kfd-after-attempts.txt
cache-before.sha256
cache-after.sha256
source-bundle-before.sha256
source-bundle-after.sha256
tree-state-after.txt
```

The route arms additionally contain `route-rank<N>.json` and
`mxfp4-shapes-rank<N>.csv` for all eight ranks.

Campaign-level artifacts include:

```text
profile-meta.txt
source-audit.json
campaign-progress.log
campaign-artifacts.sha256
profile_analysis.json
profile_analysis.md
profile-complete.txt
policy_a-profile-exit-status.txt
```

## Trace and analyzer requirements

The analyzer must parse each new trace to EOF and freeze its SHA256.  It must
require exactly two optimizer-step annotations for steps 7 and 8, sixteen
microbatch forward/backward intervals, correct concrete `Profiler armed` /
step-7 / step-8 / primary `Profiler wrote` log order, matching trace profiler
boundaries, and at most 3 ms disagreement between trace and log profiler spans. Physical
GPU events must belong only to rank 0/device 0 and lie within the selected
window.

The report must derive, only from these traces:

- profiler span, GPU envelope, busy interval union, idle/gaps;
- A4W4 forward, dgrad, and wgrad by exact `(M,N,K)`, symbol, backend, tile,
  split policy, calls, raw work, and interval union;
- residual BF16 GEMM split into protected tail layers, `lm_head`, and other;
- dual-layout activation/gradient quantization, weight quantization, packed
  transpose, scale swizzle, and conversion/helper kernels;
- packed-QKV forward/dgrad/wgrad, split/cat/copy, Q/K RMSNorm, and RoPE;
- split-SwiGLU and other activation/residual kernels;
- attention forward/backward, norm, RoPE, cross entropy, collectives by op,
  optimizer, copy/memset, and unclassified GPU work;
- CPU/operator costs for quantized-linear forward/backward,
  `hipModuleLaunchKernel`, `hipPointerGetAttribute`, synchronization, and
  blocking copies;
- overlap-safe A4W4-plus-quant union, BF16-versus-policy_a category deltas, and an
  Amdahl ceiling based on interval unions.

Missing `External ID` events must be counted and reported with the resolution
method.  They may not be silently discarded.  Raw duration sums are workload
and call-count diagnostics; overlapping raw sums must never be added to predict
step-time savings.

A row-wise packed dW split may already be contiguous.  The analyzer must not
require a no-op `aten::contiguous` event.  Adjacent-order checks must compare
equal-length sequences, for example `zip(items[:-1], items[1:], strict=True)`.

Analysis is deliberately two-phase.  The pre-final pass writes the JSON and
Markdown reports while the three final sentinels are absent.  The runner then
writes the final progress line, hashes every immutable campaign artifact,
writes `profile-complete.txt` and the zero exit sentinel, and invokes the
analyzer with `--verify-completion`.  That second pass is read-only and must
validate the manifest and completion hashes without rewriting either signed
analysis report.

## Invocation

CPU/static validation, safe from any working directory:

```bash
/usr/bin/bash /home/xdai/profile-results/lumen-mxfp4-policy-a-profile-fresh-20260928-BZluZ4/run_policy_a_profile.sh dry-run
```

Authorized GPU execution:

```bash
/usr/bin/bash /home/xdai/profile-results/lumen-mxfp4-policy-a-profile-fresh-20260928-BZluZ4/run_policy_a_profile.sh run
```

Do not run it concurrently with another GPU campaign.
