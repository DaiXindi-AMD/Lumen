# MXFP4 handoff evidence snapshot

This directory preserves the small, reviewable evidence needed after the
2026-09-29 machine migration. The authoritative interpretation is in
`../mxfp4_optimization_handoff_2026-09-22.md`; filenames retain their original
campaign names.

Included:

- `projection_guard/`: formal Policy A/B/C protocol, runner, analyzer, tests,
  frozen eight-rank shape fixture, final JSON/Markdown, and campaign receipts.
- `final_policy_a/`: fresh BF16/Policy-A/BF16 protocol, runner, analyzer,
  tests, final report, and validation receipt.
- `policy_a_profile/`: original profiler report, route instrumentation, and the
  independent supplemental audit that corrected only path/sampler interpretation.
- `aiter_bwd_v1/` and `aiter_bwd_v2/`: first failure and component-labelled
  diagnostic logs/runners; the final 23-pass source and benchmark are in the
  AITER recovery branch documented by the main handoff.
- `dual_layout_wrapper/`: accepted exact-shape result JSON and console log
  rejecting direct AITER wrapper substitution.
- `fused_forward/`: the three fresh-process CSV/log pairs for the positive
  micro-only forward fusion screen.

Not included:

- multi-gigabyte profiler traces;
- Triton/LLVM caches and compiled code objects;
- full training logs and model/data artifacts;
- ROCm-SMI snapshots already summarized by the campaign receipts.

These files are evidence and reproduction helpers, not a license to combine
speedups across campaigns. Re-establish source, dataset, cache, environment,
KFD-idle, and workload hashes before running a new candidate.

The four CPU-only analyzer/harness test files in this archive are
self-contained and should report `51 passed` when run together with
`python -m pytest -q`.
