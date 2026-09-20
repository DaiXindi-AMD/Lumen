---
name: lumen-coding
description: "Write, review, or design Lumen modules and ops, including AITER dispatch, FP8 quantization, torch.compile, and Megatron/FSDP integration. Pair with lumen-aiter when changing kernels or coordinating both repositories."
---

# Lumen Development Guide

## Codex use and source

Ported from `Lumen/.claude/skills/lumen-coding/` at Lumen commit `03102ffb954b`; all companion guides are included. Repository paths below are relative to the target Lumen checkout, and commands run from its root. Read its current `CLAUDE.md` for project conventions.

Feature counts, coverage inventories, benchmark measurements, and implementation plans in this skill and its references are historical snapshots. Check the target code and hardware before treating them as current support or running a command. Apply only the requested workflow; roadmap checklists are not permission to implement unrelated milestones.


## Mission

Lumen is a drop-in replacement for NVIDIA TransformerEngine (TE) within Megatron-Core, targeting AMD GPUs (gfx94x / gfx950). Every module, op, and kernel provides a feature-equivalent alternative to TE's interfaces.

## Architecture

```
Megatron-Core
  +- lumen/models/megatron.py   (LumenSpecProvider)
       +- lumen/modules/         (nn.Module wrappers)
            +- lumen/ops/        (stateless dispatch functions)
                 +- AITER public API  (Triton, Gluon, HIP, CK, ASM kernels)
                 +- lumen/kernels/    (existing legacy code; no new GPU kernels)
```

| Layer | Owns | Does NOT own |
|-------|------|-------------|
| `modules/` | State (params, buffers), Megatron API shape | Kernel calls |
| `ops/` | Backend dispatch, argument marshalling, fallback | Param management |
| AITER | New GPU kernels, public wrappers, kernel tests, benchmarks, tuning | Lumen parameter and training state |
| `lumen/kernels/` | Existing legacy implementations | New GPU kernels |

**New GPU kernels belong in AITER, not in Lumen.** Lumen only dispatches.

Use [lumen-aiter](../lumen-aiter/SKILL.md) for cross-repository changes. `third_party/aiter/` is the declared dependency location; an external AITER checkout can provide the actual editable install. Verify the imported package path and commit before choosing where to edit.

## Coding Conventions

### Constants -- derive from spec

```python
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
scale = amax / FP8_E4M3_MAX  # NOT: scale = 448.0
```

### No TE language in docstrings

```python
# BAD: """Drop-in replacement for TE's TEColumnParallelLinear."""
# GOOD: """Column-parallel linear using Lumen GEMM."""
```

### Fallback Logging

Every fallback path must log why and comment the intent:

```python
def my_op(x):
    try:
        return _asm_path(x)
    except RuntimeError as e:
        logger.warning("my_op: ASM failed (%s), falling back to Triton", e)
        return _triton_path(x)
```

`try_backends()` chains need per-lambda comments. Conditional fallbacks need comments explaining why.

### First-Principles Codegen

1. Clarify the problem in one sentence
2. Identify constraints (memory bandwidth, register pressure, API contract)
3. Derive solution from constraints, not analogy
4. Add abstraction only when it removes duplication across >=2 call sites

### Anti-Patterns

- Abstraction with one subclass -> use plain function
- Copying TE API blindly -> shape API by Lumen's own requirements

## Code Review Checklist

- [ ] Constants derived from hardware facts, not magic numbers
- [ ] No unnecessary abstraction (YAGNI)
- [ ] `try_backends()` uses the op/scaling-specific priority order (not one universal chain)
- [ ] AITER imports guarded by `_probe_aiter_*()` probes
- [ ] Fallback paths logged with reason
- [ ] FP8 scaling type handled for both forward and backward
- [ ] No TE "drop-in replacement" language in docstrings
- [ ] Tests use `compute_snr` / `check_close` against references
- [ ] Fallback paths covered by tests
- [ ] Performance-critical path free of Python-level overhead

### Handling Review Feedback

- **Verify before implementing** -- check against codebase reality
- **Push back with reasoning** if a suggestion breaks functionality or violates YAGNI
- **No performative agreement** -- state what changed or ask for clarification
- **One fix at a time**, test each before proceeding

## Feature Parity

Scorecard: ~61 features, ~47 supported (77%), 4 Lumen-only, ~2 missing, ~6 partial, ~2 deferred.

For complete feature matrix and TE mapping -> see [reference.md](reference.md).

---

## Subsection Guides

Detailed domain-specific guides are in separate files. Read when working in that area:

- **[AITER Dispatch](aiter-dispatch.md)** -- `try_backends()`, probe functions, GEMM/attention backend dispatch tables, fallback chain design
- **[Quantize Manager](quantize-manager.md)** -- ScalingManager lifecycle, FP8 scaling types, blockwise2d dual behavior, amax history
- **[torch.compile Compatibility](torch-compile.md)** -- Compile Guard Dual-Mode design, 6 blockers, implementation phases, `LUMEN_BACKEND` env var
- **[Megatron & FSDP Integration](megatron-fsdp.md)** -- LumenSpecProvider, norm patching, FSDP1/FSDP2, CLI args, example scripts, QuantConfig wiring
