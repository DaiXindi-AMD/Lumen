# Projection-guard campaign recovery

This directory preserves the CPU-validated 2026-09-28 campaign harness for
the next fresh Qwen3-8B MXFP4 measurement. It is recovery material, not a
performance result and not a production default.

## Frozen source state

```text
Lumen runtime path: /home/xdai/Lumen
Lumen branch:       dev/mxfp4
Lumen commit:       6b9aee1569247eca20937c14319ba6adcd69e0cb
Lumen tracked diff: 346860766945762e587bff65b1cbf9486e73ab0f5bd884a86f7fb9fe9527021a

AITER runtime path: /home/xdai/aiter
AITER branch:       bench/ecfff3f-lumen
AITER commit:       e35bb17f4f815903bf73598facedbb321e15af28
AITER tracked diff: a5476078484e426e951c2a5e62233cb20f26c103f93656e02a690e294e9f9aa0
```

The actual Python runtime resolves AITER from the sibling checkout, not from
Lumen's `third_party/aiter` submodule. Preserve both dirty worktrees exactly;
do not reset, clean, or blindly merge them.

## Experiment

The campaign starts with an untimed policy-B route/cache smoke and then runs:

```text
tail1_a1 -> guard_o_down_b1 -> guard_down_c -> guard_o_down_b2 -> tail1_a2
```

- A: one complete final transformer layer in BF16.
- B: tail zero, with final-layer `o_proj` and `down_proj` in BF16.
- C: tail zero, with only final-layer `down_proj` in BF16.
- `lm_head` remains BF16 in every arm.
- Formal arms use 50 optimizer updates and steps 11--50 for timing.
- All arms use the same 8-GPU Qwen3-8B, seq-8192, MBS2, GBS128/GA8,
  FSDP2 full-shard, retained-parameter, no-checkpoint, BF16-reduction,
  packed-QKV, split-SwiGLU, and AITER-attention policy.

See `protocol.md` for the complete predeclared speed, drift, integrity, and
validation-NLL gates.

## Validation completed before backup

```text
bash -n run_projection_guard.sh: PASS
bash -n run_case.sh: PASS
analyzer --self-test: PASS
cwd-independent unittest: 9/9 PASS
ruff check --no-cache: PASS
ruff format --check --no-cache: PASS
absolute-path no-GPU dry-run: PASS
```

The focused Lumen implementation tests could not be recollected in the
sandbox because AITER/Triton import requires a visible GPU driver. No test body
ran, so that attempt is neither a pass nor a source failure.

## Restore and run

Copy this directory to the exact path expected by the scripts:

```bash
/home/xdai/profile-results/lumen-mxfp4-projection-guard-formal-fresh-20260928-XWXD4L
```

Verify every file with `SHA256SUMS`, restore the exact Lumen and AITER source
states above, and first run:

```bash
/usr/bin/bash /home/xdai/profile-results/lumen-mxfp4-projection-guard-formal-fresh-20260928-XWXD4L/run_projection_guard.sh dry-run
```

Only after all eight GPUs are idle and every KFD client identity is readable:

```bash
/usr/bin/bash /home/xdai/profile-results/lumen-mxfp4-projection-guard-formal-fresh-20260928-XWXD4L/run_projection_guard.sh phase1
```

Audit the smoke/cache and A1/B1/C artifacts before allowing `phase2`. Never
kill another GPU workload to make this experiment run.

## Related AITER recovery branch

The portable AITER dirty-code backup is remotely verified at:

```text
repository: https://github.com/DaiXindi-AMD/aiter.git
branch:     backup/2026-09-28/ecfff3f-lumen-portable-wip
commit:     5d7178517e5f3947b7496fa5994696aa2fb9c50d
```

Use its recovery README and file manifest rather than cherry-picking an
unrelated upstream history wholesale.
