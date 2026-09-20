# Lumen integration constraints

Read for Lumen changes that consume AITER APIs, or for isolating an AITER-backed training mismatch.

Sources: Lumen [CLAUDE.md](https://github.com/DaiXindi-AMD/Lumen/blob/03102ffb954b749e14e7e89b22f8b0fbeceacc0e/CLAUDE.md), [dispatch.py](https://github.com/DaiXindi-AMD/Lumen/blob/03102ffb954b749e14e7e89b22f8b0fbeceacc0e/lumen/ops/dispatch.py), and the six migrated Lumen skills. Checked 2026-09-18. Read the target checkout's current rules and implementation when they differ.

## Lumen owns integration

Lumen supplies AMD training interfaces compatible with the Megatron-Core workflow, without editing Megatron-Core sources for each feature. Modules own state; ops own dispatch, marshalling, and fallback; new GPU kernels and reusable low-level functions belong in AITER.

- Derive constants from dtype/hardware/math rather than unexplained literals. Add abstraction only to remove real duplication or encapsulate a non-obvious invariant.
- Generated code comments/docstrings describe the Lumen operation directly. Do not call it a TransformerEngine “drop-in replacement” or mention TE classes just to describe an equivalent class; actual runtime imports/class references remain valid.
- Do not bypass the public AITER API by copying or directly launching private kernels in Lumen.

## Capability and dispatch

- Guard AITER availability with `try/except` inside `_probe_aiter_*()` functions in `lumen/ops/dispatch.py`. New backend probes use `@functools.lru_cache(maxsize=1)`.
- Use `try_backends()` and, where appropriate, `build_fallback_chain()` to select available backend candidates. AITER and Triton are not guaranteed present merely because the GPU is AMD.
- Fallbacks log the reason and have brief intent comments, including per-lambda comments in dispatch chains. Preserve the actual op/scaling priority, not a universal ASM-first order.
- For correctness/debugging, synchronize before concluding an asynchronous kernel succeeded. The migration checkout's dispatcher synchronizes during warmup, then caches a successful backend by stable label; graph capture requires warmup and the cached path skips sync. Preserve these mechanisms instead of adding unconditional synchronization to a timed/compiled/captured path. Revalidate the implementation before relying on compile-mode behavior from an old design document.
- If several candidates share a backend enum, preserve distinct entry labels so a warmup cache does not select a different kernel after a shape-dependent chain change. Record the actual selected kernel/backend in correctness and performance evidence.

## Numerical constraints from the Lumen guide

| Path | Constraint |
| --- | --- |
| BF16 GEMM | `Y = A @ W.T`, with weight `(N, K)`; match operand dtypes after dequantization. The cited gfx942 Triton path requires `BLOCK_SIZE_K >= 128`; the cited ASM path requires `K % 64 == 0` and `N % 64 == 0`. Verify these for the specific kernel used. |
| delayed/dynamic FP8 GEMM | hipBLASLt → CK `gemm_a8w8_CK` → Triton; CK tuning uses `a8w8_tuned_gemm.csv`. |
| per-token FP8 GEMM | Triton `gemm_a8w8_per_token_scale`; do not assume the scalar-scale backend chain. |
| blockwise FP8 GEMM | CK `gemm_a8w8_blockscale` → Triton; CK tuning uses `a8w8_blockscale_tuned_gemm.csv`. |
| MXFP8 GEMM | Triton `gemm_mxfp8`, subject to hardware and layout support. |
| Full FP8 backward | The documented transpose path supports scalar scales (`delayed`, `dynamic`; `none` is unquantized). Per-token/blockwise/blockwise2d/MXFP8 scale layouts do not survive a simple `weight_fp8.t()`. Validate an explicitly implemented alternative; do not enable unsupported backward or retune aligned scaling to hide errors. |
| Normalization backward | Prefer a backend that saves required intermediates; the guide prefers Triton when gradients are needed. The documented ASM norm is LayerNorm-only. |

The CK CSVs above are backend-specific and distinct from the nested **Triton/Gluon JSON** configuration tree. Old Lumen tables saying “no tuned config” describe the then-current path, not an exemption from providing configs for new AITER kernels.

## Integration tests and validation order

- AITER unit tests first, through the public wrapper, then Lumen op/module/fallback tests, then short training confirmation when requested. Include a test that disables/fails the primary backend and checks a correct fallback.
- FP8/GPU tests create tensors and move modules to `cuda` (PyTorch ROCm retains this device spelling). Skip clearly when the required hardware/backend is absent.
- Use identical input values for reference and implementation; ensure reference tensors whose `.grad` is inspected are leaves, for example `(torch.randn(...) * 0.02).requires_grad_(True)`. Assert gradients are non-None before SNR checks.
- Match tensor-parallel dimensions: row-parallel input with `input_is_parallel=True` has width `in_features / tp_size`.
- Pass required arguments such as `attention_mask=None` even in benchmarks. The documented quantized attention path auto-maps `aiter_csrc` to `aiter_triton`; verify actual selection.
- MXFP8 attention needs `padded_head_dim % quant_block_size == 0`; use valid block/sequence/head sizes.
- Reuse Lumen `compute_snr`, `check_close`, or strict reference assertions with justified dtype/path tolerances. Numerical tests must do more than inspect shape or merely complete backward.
- Lumen performance tests call real Lumen APIs, use CUDA-event timing, and report end-to-end effects separately from AITER microbenchmarks. Distributed benchmark groups use `cpu:gloo,cuda:nccl` with device selection, and their existing cleanup fixtures.
- For divergence, read the repo-local Codex training bug note, freeze reference config/data/checkpoint, and locate the first mismatch before modifying aligned knobs. AITER evidence justifies kernel validation first; an untrustworthy training run should be stopped with its evidence retained.

Detailed routes: [coding](../../lumen-coding/SKILL.md), [tests](../../lumen-test/SKILL.md), [training debugging](../../lumen-training/SKILL.md), [benchmarks](../../lumen-benchmark/SKILL.md).
