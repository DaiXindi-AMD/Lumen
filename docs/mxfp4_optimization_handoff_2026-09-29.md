# Lumen MXFP4 current handoff

The authoritative detailed guide is
[`mxfp4_optimization_handoff_2026-09-22.md`](mxfp4_optimization_handoff_2026-09-22.md).
Its historical filename is retained so existing recovery links keep working;
the document itself is updated through 2026-10-07. Read sections 0 and 20 first.

2026-10-07 AITER/Lumen migration update:

- ROCm/aiter #5542 merged as
  `48b13652fd05ed6e65d82204d3e040373ff74708`. Lumen should call
  `aiter.ops.triton.activation.silu_and_mul_backward` through its public
  wrapper; Megatron is the direct consumer, while the current HF/FSDP path is
  not.
- The merged backward is gfx950-only. Architecture-guard follow-up #6124 is
  still open at `b207635449d945d9e445df3f0d55717c7dab3d2d`; until it lands in the
  pinned AITER, Lumen must check the input device architecture and use a logged
  fallback on non-gfx950 devices.
- Full migration still waits for separate AITER PRs for
  `dual_layout_quant_mxfp4` and `dequant_hadamard_quant_mxfp4`. Preserve the
  existing dual-layout, backward DGrad/WGrad, and dequant -> transpose -> H16
  -> requant RHT positions.

Current selected working policy (short-run quality gate only):

- Policy A: final complete transformer layer and `lm_head` in BF16, packed QKV,
  split SwiGLU, AITER attention, FSDP2 full-shard, retained accumulated params,
  no activation checkpointing, BF16 reduction.
- BF16 midpoint / MXFP4 mean: `8077.2050 / 5182.0975 ms`.
- Mean / median speedup: `1.558675x / 1.576439x`.
- Block-4 95% CI: `[1.542335x,1.573744x]`.
- Validation delta NLL: `+0.00790` (short-run quality gate passes).
- The requested `1.6x` speed target is not reached.

2026-10-10 dispatch audit addendum (records what the `6da1a42` commit message
does not state):

- `6da1a42f7bd176227391568d4f80da91fda13398` ("preserve the optimized Qwen3
  training stack") also changed the MXFP4 GEMM backend dispatch:
  `_gemm_mxfp4_aiter_asm` now calls
  `aiter.ops.gemm_op_a4w4.gemm_a4w4_asm` directly with a validated ASM tuned
  entry, replacing the generic `aiter.gemm_a4w4` call whose tuned selector
  could pick ASM or CK. The Lumen MXFP4 GEMM registry (forward, DGrad, WGrad)
  therefore contains no CK candidate. The AITER generic `gemm_a4w4` API keeps
  its CK selector but is bypassed on this path.
- CK is not removed from the training stack: AITER attention keeps the CK
  `mha_fwd` / `mha_bwd` fallback for shapes outside the v3 constraints. The
  saved Policy A profile shows attention producers `aiter::fmha_v3_fwd/bwd`
  (ASM v3), and the current AITER compile registration maps those v3 modules
  to `asm_mha_fwd/bwd.cu` with `-DENABLE_CK=0`. The trainer's historical
  "CK FMHA" help text did not describe the actual default path.

Recovery refs:

- Lumen: `ZhangDanyang-AMD/Lumen`, branch `dev/mxfp4`; code snapshot ancestor
  `6da1a42f7bd176227391568d4f80da91fda13398`.
- Lumen fallback mirror: `DaiXindi-AMD/Lumen`, branch
  `backup/2026-09-29/mxfp4-final-handoff`; it must resolve to the same final
  handoff tip as the primary branch at delivery time.
- AITER: `DaiXindi-AMD/aiter`, branch
  `backup/2026-09-29/mxfp4-current-handoff`, commit
  `7c4a9a496600bb38496ee5c88a18551c0fd76ea7`.
- Lumen AITER-migration recovery: `DaiXindi-AMD/Lumen`, branch
  `backup/2026-09-29/aiter-kernel-migration-handoff`, commit
  `6c19736a6d50093f622970285937d80556b364fd`.
- Small evidence archive: `mxfp4_evidence_2026-09-29/`.

Do not start with a new E2E run. Restore and verify both repositories, read the
full attempt ledger in `.codex/tmp-training-bugs.md`, and follow the two-stage
backward redesign gate sequence in section 20.7 of the detailed guide.
