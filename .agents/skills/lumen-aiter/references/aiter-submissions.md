# AITER commits and pull requests

Read when splitting changes, preparing a commit message, formatting a PR title/body, or checking submission readiness.

## Authoritative sources

Checked against upstream AITER `1834cd14b33ff7fd366c8ba6e6486e41df027b70` on 2026-09-18:

- [CONTRIBUTE.md: PR title, description, DCO, code quality](https://github.com/ROCm/aiter/blob/1834cd14b33ff7fd366c8ba6e6486e41df027b70/CONTRIBUTE.md).
- [Triton review rules: one concern per PR](https://github.com/ROCm/aiter/blob/1834cd14b33ff7fd366c8ba6e6486e41df027b70/.github/instructions/aiter-ops-triton.instructions.md).
- [PR title/label automation](https://github.com/ROCm/aiter/blob/1834cd14b33ff7fd366c8ba6e6486e41df027b70/.github/workflows/pr-title-tags.yaml).
- [Current formatting/lint CI](https://github.com/ROCm/aiter/blob/1834cd14b33ff7fd366c8ba6e6486e41df027b70/.github/workflows/pre-checks.yaml) and [local pre-commit hook](https://github.com/ROCm/aiter/blob/1834cd14b33ff7fd366c8ba6e6486e41df027b70/.githooks/pre-commit).

Recheck the target branch; neither older local documentation nor recent commit-message examples override its actual contribution rules.

## Scope and commit construction

- One concern per PR. A new kernel gets its own PR with its necessary wrapper, unit tests, benchmark, and tuning configs. An unrelated refactor, another kernel fix, or independent retuning goes into a follow-up PR.
- Lumen integration and AITER implementation remain separate repository changes. Link their API/revision dependency when preparing submissions. A submodule update, if requested, must point to the intended AITER commit rather than an unreviewable local working tree.
- Config moves/renames must be pure `git mv` commits (100% similarity); make content changes in a separate subsequent commit.
- AITER requires the Developer Certificate of Origin. When committing is part of the authorized task, use `git commit -s` with the contributor's configured name/email to add `Signed-off-by`. Do not invent an identity or forge a different person's sign-off. This is a DCO trailer, not a requirement to GPG-sign with `-S`.

**Git commit naming:** the verified contribution guide specifies PR title syntax and DCO, but does not impose a single commit subject grammar, subject-length limit, or branch-name format. Existing commits use both bracket tags and `fix(scope):`/`feat(scope):` forms. Do not claim Conventional Commits is an AITER requirement.

For a new series without a user-specific convention, a useful **recommendation** is a concise type-prefixed imperative subject matching the PR vocabulary, with the op and concrete change:

```text
[Bugfix] Correct FP8 scale layout in grouped GEMM
[Perf] Tune RMSNorm for gfx950
[Kernel] Add fused normalization and quantization
```

Use the body for why, constraints, relevant test/benchmark evidence, and seeded-config provenance. Preserve the user's chosen naming convention when compatible with actual project requirements.

## PR title naming

Current upstream has two independent tag classes:

| Author-supplied type | Meaning |
| --- | --- |
| `[Bugfix]` | Correctness or other bug fix |
| `[Feature]` | New feature/operator |
| `[Kernel]` | New kernel or kernel optimization |
| `[Perf]` | Performance optimization |
| `[Test]` | Test changes |
| `[Hardware]` | Hardware support or architecture-specific work |
| `[Misc]` | Other work |

The action adds canonical **component** tags based on changed files:
`[Triton/Gluon]`, `[ASM]`, `[HIP]`, `[CK]`, `[OPUS]`, `[FlyDSL]`, `[CI]`, `[JIT]`, `[Build]`, `[Config]`, `[Docs]`.

- The title contains at most three component tags; labels carry the full set. `[Config]` applies to config-only changes and `[Docs]` to docs-only changes according to the workflow's path rules.
- Hand-written variants such as `[TRITON]`, `[Gluon]`, `[CK_TILE]`, and `[Doc]` are normalized. Type/hardware/other non-component tags are preserved.
- Authors can supply `[Perf] Tune RMSNorm for gfx950`; on a qualifying Triton PR, the action produces `[Triton/Gluon] [Perf] Tune RMSNorm for gfx950`. Use canonical tags in a prepared final title if helpful; automation owns component classification.
- The verified workflow skips draft PRs and runs when ready for review and on its configured PR events. Do not promise that a draft's title has already been rewritten.
- `no-auto-title` is an available opt-out label, not a label to add by default. Changing labels or PR metadata must stay within the requested task.
- Older branches without this workflow may document manual component prefixes such as `[Triton]`. Use their actual rulebook when explicitly backporting to those branches.

## Quality checks and evidence

- Run the checkout's Black/Ruff checks on changed Python files and clang-format-18 checks on changed C++/HIP files as applicable. Complete the relevant required CI/test checks; avoid reformatting unrelated files.
- Read the workflow for tool versions. At the verified upstream revision, CI pins `ruff==0.16.0`, while `CONTRIBUTE.md` still suggests `ruff==0.15.7` and `black==26.3.0`; the effective CI pin takes precedence for reproducing lint results. Black CI uses `psf/black@stable`. Do not claim a documentation pin is the running CI version.
- `.githooks/install` enables the repo hook. The hook edits files and re-stages them (copyright, whitespace, formatting, Ruff fixes), so inspect the staged diff after it runs, especially when changes were partially staged. No global Git settings are needed for a per-commit `-s`.
- Include kernel unit-test and benchmark commands, actual output/results, hardware, software and commit versions, and before/after comparisons. Record regressions as well as wins. If a GPU or test was unavailable, leave that validation item unverified.
- Include bandwidth utilization and roofline/arithmetic-intensity evidence for kernel performance work as required by `CONTRIBUTE.md`, with estimates labeled. Keep extended analysis out of kernel comments.
- New dependencies need justification; prefer existing PyTorch, ROCm/HIP, CK, and Triton facilities. Document API/behavior changes and update affected docs. Convention changes under Triton also update its README and review instructions.

## PR body

Use the target repository's template. The verified `CONTRIBUTE.md` proposes these fields; fill them with concrete evidence and omit irrelevant narrative:

- **Summary / Motivation:** the failing or missing case and resulting behavior.
- **Changes:** kernel/API/config changes and compatibility impact.
- **Performance:** hardware and shape/dtype/layout, before/after latency, improvement/regressions, utilization and roofline evidence where applicable.
- **Testing:** exact unit/integration/benchmark commands and outcomes; distinguish passed, skipped, and not run.
- **Documentation / Dependencies / Breaking Changes:** applicable updates, dependency justification, and migration details.

Do not pre-check hardware or test boxes that were not executed. For cross-repository work, state which Lumen patch consumes which AITER API and revision. Preparing these artifacts does not itself authorize publishing or contacting maintainers.
