# Lumen MXFP4 dirty-worktree recovery snapshot

Date: 2026-09-22 (US/Central)

This branch is a machine-migration snapshot of the relevant dirty files from
`/home/xdai/Lumen` on `dev/mxfp4` at base commit
`8d7a8a8e0410ab87409bb97d0849e26b4d4becb8`. It is not a PR-ready change and
must not be merged wholesale.

## Recovery ref

```text
repository: https://github.com/DaiXindi-AMD/Lumen.git
branch:     backup/2026-09-22/mxfp4-optimization-wip
```

The exact snapshot commit is recorded in the main handoff document after the
branch is pushed.

## What is preserved

The snapshot keeps the complete relevant tracked-file diffs because accepted
and experimental changes share files and cannot be safely reconstructed by
copying only selected hunks.

Core/current-path material includes:

- Qwen3 training CLI, deterministic pairing evidence, FSDP2 BF16 reduction,
  retained accumulated parameters, packed QKV, split SwiGLU, and projection
  guard plumbing.
- MXFP4 linear forward/backward, fail-closed ASM dispatch, autotune/cache
  integrity, weight-cache invalidation, and model-specific A4W4 tuning rows.
- Unit/integration tests and the packed-QKV benchmark.

Experimental or default-off material is intentionally preserved for recovery:

- FlyDSL MXFP4 prototypes and their benchmark hooks. New GPU kernels in
  `lumen/kernels/flydsl/` violate the intended Lumen/AITER ownership boundary
  and must move to AITER before any production submission.
- Packed gate/up experiments, which were rejected as a speed optimization and
  remain opt-in/default-off.
- Root-parameter FSDP retention, which is on hold because wall-time benefit was
  not reproducible.
- The unfinished final-layer projection guard campaign.
- Hadamard diagnostic scripts and exact-shape diagnostic benchmarks.

The fresh-reset experiment ledger is archived as
`docs/mxfp4_fresh_experiment_log_2026-09-22.md`; its source file SHA256 was
`7c6e0f3da5944047df69056e705bc321cd2e7083347fbff88d3103802cb5887d`.

## What is deliberately excluded

- `.agents/`, `AGENTS.md`, and other local agent-skill metadata.
- `.gitignore`, whose only dirty hunk ignored an agent scratch file.
- `hadamard_analysis/` and `hadamard_wgrad_analysis/` PNG/HTML/JSON generated
  outputs.
- The dirty `third_party/aiter` worktree. Its gitlink did not change; the only
  submodule file difference was an unrelated A8W8 CRLF-to-LF conversion.
- GPU traces, compiler caches, result directories, datasets, and model files.

## AITER dependency

The runtime AITER checkout was `/home/xdai/aiter`, branch
`bench/ecfff3f-lumen`, base `e35bb17f4f815903bf73598facedbb321e15af28`.
Its 17 relevant dirty code files are backed up byte-for-byte at:

```text
repository: https://github.com/DaiXindi-AMD/aiter.git
branch:     backup/2026-09-22/fused-swiglu-dual-layout-wip
commit:     35e796da188e2131d004e5b17391b7b8e836d852
```

That AITER snapshot is based on a much newer fork revision, so restore its 17
paths onto the intended AITER base; do not blindly cherry-pick the whole
snapshot commit.

## Status labels

- Accepted/current: exact WGrad ASM route, packed QKV, split SwiGLU, FSDP2
  retained accumulated decoder parameters, no activation checkpointing, and
  BF16 gradient reduction.
- Conservative candidate: whole-layer BF16 `tail2`.
- Best whole-layer speed candidate still needing a fresh final campaign:
  `tail1`.
- Rejected for accuracy: `tail0`.
- Rejected/default-off: packed gate/up.
- Unfinished/default-off: final-layer projection guard and FlyDSL prototype.

No GPU tests were rerun while creating this recovery snapshot.
