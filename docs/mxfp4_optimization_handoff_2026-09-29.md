# Lumen MXFP4 current handoff

The authoritative detailed guide is
[`mxfp4_optimization_handoff_2026-09-22.md`](mxfp4_optimization_handoff_2026-09-22.md).
Its historical filename is retained so existing recovery links keep working;
the document itself is updated through 2026-09-29. Read sections 0 and 20 first.

Current selected working policy (short-run quality gate only):

- Policy A: final complete transformer layer and `lm_head` in BF16, packed QKV,
  split SwiGLU, AITER attention, FSDP2 full-shard, retained accumulated params,
  no activation checkpointing, BF16 reduction.
- BF16 midpoint / MXFP4 mean: `8077.2050 / 5182.0975 ms`.
- Mean / median speedup: `1.558675x / 1.576439x`.
- Block-4 95% CI: `[1.542335x,1.573744x]`.
- Validation delta NLL: `+0.00790` (short-run quality gate passes).
- The requested `1.6x` speed target is not reached.

Recovery refs:

- Lumen: `ZhangDanyang-AMD/Lumen`, branch `dev/mxfp4`; code snapshot ancestor
  `6da1a42f7bd176227391568d4f80da91fda13398`.
- Lumen fallback mirror: `DaiXindi-AMD/Lumen`, branch
  `backup/2026-09-29/mxfp4-final-handoff`; it must resolve to the same final
  handoff tip as the primary branch at delivery time.
- AITER: `DaiXindi-AMD/aiter`, branch
  `backup/2026-09-29/mxfp4-current-handoff`, commit
  `58fb9ee213b592627b1c5f668996469925b9f725`.
- Small evidence archive: `mxfp4_evidence_2026-09-29/`.

Do not start with a new E2E run. Restore and verify both repositories, read the
full attempt ledger in `.codex/tmp-training-bugs.md`, and follow the two-stage
backward redesign gate sequence in section 20.7 of the detailed guide.
