---
name: lumen-aiter
description: Develop, review, or prepare submissions for Lumen–AITER kernel integrations. Use when deciding repository ownership, adding or tuning AITER kernels called by Lumen, debugging an AITER-backed Lumen op, or preparing AITER commits and PRs. Covers kernel repr, tests, benchmarks, shared utils, tuning configuration, and submission conventions.
---

# Lumen–AITER Joint Development

Implement the requested feature across the correct ownership boundary, with evidence for kernel correctness and Lumen integration. Apply only the relevant part when the request concerns one repository.

## Establish the target

- Locate the user's Lumen and AITER checkouts; read their applicable project instructions. For Lumen, read `CLAUDE.md`. For AITER, read `CONTRIBUTE.md`, `aiter/ops/triton/README.md`, and `.github/instructions/aiter-ops-triton.instructions.md` when applicable. Read `aiter/ops/triton/configs/CLAUDE.md` before editing tuned JSON or its loaders.
- Record each target commit, branch, dirty state, and the active environment's imported `lumen` / `aiter` paths. Lumen declares `third_party/aiter/`, but the actual editable install may point to a sibling checkout or worktree. An empty submodule directory is not proof AITER is unavailable; installing or updating it is not an automatic prerequisite.
- Distinguish the Lumen dependency fork from `ROCm/aiter` upstream. Choose the patch destination from the requested target and actual imported code; record the AITER revision required by the Lumen change.
- The references were checked on 2026-09-18 against Lumen `03102ffb954b` and upstream AITER `1834cd14b33ff`. That AITER revision uses the nested config layout. Older branches may not have its loaders or title automation: inspect the target before applying commands, and do not migrate an entire tree as a side effect of an unrelated fix.

## Ownership boundary

| Work | Owner and location |
| --- | --- |
| Parameters, buffers, training lifecycle, Megatron/HF/FSDP integration | Lumen `modules/`, `models/`, `quantize/`, or the relevant integration package |
| Backend choice, argument/layout adaptation, capability probes, fallback | Lumen `ops/`, using `try_backends()` and `dispatch.py` |
| New GPU kernels: Triton, Gluon, HIP, CK, ASM and other backends | AITER, exposed through its public Python wrapper/API |
| Kernel tests, microbenchmarks, tuning configuration, reusable kernel helpers | AITER, in the matching category and shared utility directories |
| Integration correctness, fallback coverage, training and end-to-end performance | Lumen `tests/`, `benchmarks/`, and relevant examples |

All **new GPU kernels belong in AITER**, including kernels needed only by a Lumen feature. Existing `lumen/kernels/` files do not authorize new kernels there. Do not turn a focused change into an unrequested migration of existing kernels. Lumen consumes the AITER public wrapper; it must not import or launch private `_triton_kernels` / `_gluon_kernels` bodies. AITER must not depend on Lumen to implement or test its operator.

## The user's seven AITER requirements

1. **All kernels must have a repr decorator.** Every launchable Triton/Gluon kernel uses `make_kernel_repr` and `@triton.jit(repr=...)` / `@gluon.jit(repr=...)`, including meaningful compile-time/tuning keys. For HIP/CK/ASM or another backend, inspect its real naming/registration mechanism and explain the applicability of this requirement; never invent a Python `@repr` API or claim an unsupported decorator was applied. Device-only JIT helpers are not separately launchable kernels.
2. **All kernels must have unit tests.** Add or extend AITER numerical tests against a trusted reference; exercise the public wrapper and the relevant shapes, dtypes, layouts, edge conditions, and backward path where supported. A Lumen end-to-end test alone does not satisfy this requirement.
3. **All kernels must have benchmarks.** Add or extend the matching AITER benchmark and record comparable baseline/changed measurements on the target GPU. If hardware is unavailable, deliver the runnable benchmark and explicitly leave execution unverified.
4. **All kernels must be in the right folders with tuning config files.** Keep wrappers, kernel bodies, tests, benchmarks, and configuration in their designated locations. Use the target family's shared loader and external tuning files; no hidden per-op tuning dictionaries in Python.
5. **No duplicate or nearly duplicate kernels.** Search existing AITER wrappers, kernels, helpers, and configurations first. Compare semantics, dtypes, layout, fusion, shape limits, and backward support. Reuse or extend a matching implementation; state the concrete capability gap before creating a new kernel.
6. **Common functions, including activations, go in utils.** Reuse shared activation, shuffle, architecture, config, and logging helpers. New reusable helpers go in the appropriate `utils/` layer; shared split-K reductions use `_triton_kernels/common/`. Do not hide copies in wrappers, tests, or benchmarks.
7. **Comments stay succinct: 1–2 sentences describing what and why.** State non-obvious behavior or constraints. Public wrappers still document their contract—arguments/layout/config, return, unsupported cases—in compact form. Put extended performance analysis in the PR or benchmark report.

These user requirements govern new/modified kernel work even when older examples omit them. Existing unrelated kernels are not a mandate for a repository-wide cleanup.

## Work and validation

1. State the required semantics, supported input contract, and ownership split. Search for reuse before choosing implementation files.
2. For kernel changes, read [AITER kernel development](references/aiter-kernels.md). For configuration or tuning, also read [AITER tuning](references/aiter-tuning.md).
3. Implement the AITER public API and required kernel artifacts, then connect Lumen through capability probes and op-specific dispatch. Read [Lumen integration](references/lumen-integration.md) for numerical, fallback, and test constraints.
4. Validate in order: AITER kernel unit tests through the wrapper → smallest Lumen integration/fallback test → short end-to-end confirmation if the task requires it. Run benchmarks after correctness. Verify which backend actually ran so fallback cannot disguise a failed optimization.
5. Report the changed files by repository, exact revisions/environment, commands and results, measured performance, and remaining hardware/test limitations. Do not mark skipped GPU tests as passing.

When debugging training, use [lumen-training](../lumen-training/SKILL.md): preserve already-aligned reference settings and localize the mismatch. For broader Lumen development use [lumen-coding](../lumen-coding/SKILL.md), [lumen-test](../lumen-test/SKILL.md), or [lumen-benchmark](../lumen-benchmark/SKILL.md). RL and collective scheduling are covered by [lumen-rl](../lumen-rl/SKILL.md) and [lumen-sdma](../lumen-sdma/SKILL.md).

## Commits and PRs

Read [AITER submission rules](references/aiter-submissions.md) before preparing commit messages or PR content.

- A new kernel gets its own AITER PR containing its wrapper, tests, benchmark, and necessary configs; split unrelated fixes/refactors into follow-ups. Keep Lumen integration as a separate repository patch and state the dependency.
- AITER contributions require DCO sign-off (`git commit -s`, using the configured contributor identity). The verified contribution guide mandates PR title tags but does **not** mandate a single Git commit subject grammar; distinguish the documented rule from suggested commit naming.
- Authors supply type prefixes such as `[Bugfix]`, `[Feature]`, `[Kernel]`, `[Perf]`, `[Test]`, `[Hardware]`, `[Misc]`. On current upstream, the title workflow derives component tags such as `[Triton/Gluon]`, `[CK]`, and `[HIP]` from changed files.
- Pass the target checkout's formatting/lint and relevant CI checks, and include actual unit-test and benchmark evidence in the PR. Config moves are pure rename commits; content changes follow separately.
- Preparing a patch or skill does not itself request a commit, push, PR publication, label change, or dependency bump. Carry out those actions only within the user's requested workflow and existing authorization.
