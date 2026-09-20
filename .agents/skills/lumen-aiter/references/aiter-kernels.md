# AITER kernel development

Read for implementation or review of kernels, public wrappers, unit tests, or benchmarks. Paths here are relative to the AITER repository, not Lumen.

## Sources and scope

Verified 2026-09-18 at upstream commit `1834cd14b33ff7fd366c8ba6e6486e41df027b70`:

- [Triton/Gluon maintainer guide](https://github.com/ROCm/aiter/blob/1834cd14b33ff7fd366c8ba6e6486e41df027b70/aiter/ops/triton/README.md).
- [Scoped review rules](https://github.com/ROCm/aiter/blob/1834cd14b33ff7fd366c8ba6e6486e41df027b70/.github/instructions/aiter-ops-triton.instructions.md).
- [General contribution guide](https://github.com/ROCm/aiter/blob/1834cd14b33ff7fd366c8ba6e6486e41df027b70/CONTRIBUTE.md).

The specific Triton/Gluon rules take precedence over old standalone-test examples in the general contribution guide for that subtree. Recheck instructions in the actual target revision.

## Reuse before implementation

Search the semantic operation, aliases, relevant dtype/fusion, and nearby kernels with `rg`; search filenames with `rg --files`. Inspect wrappers and their limitations, not just matching names. Check:

- `aiter/ops/triton/utils/`: shared config, shuffle, architecture, activation, logging, and repr helpers.
- `aiter/ops/triton/_triton_kernels/common/`: shared reduction machinery.
- Existing Triton/Gluon and other backend kernels, matching unit tests, and tuned configurations.

Prefer extending an existing op's supported cases or public wrapper. A rename, changed tiling, or slightly different activation is not sufficient justification for a duplicate kernel. Record an actual semantic or hardware constraint if specialization is necessary.

## Folder and API contract

| Artifact | Triton/Gluon location |
| --- | --- |
| Public wrapper | `aiter/ops/triton/<category>/<op>.py` |
| Triton body | `aiter/ops/triton/_triton_kernels/<category>/<op>.py` |
| Gluon body | `aiter/ops/triton/_gluon_kernels/`, with `<arch>/<category>/` for architecture-specific implementations |
| Unit test | `op_tests/triton_tests/<category>/test_<op>.py` |
| Benchmark | `op_tests/op_benchmarks/triton/bench_<op>.py` |
| Tuning | `aiter/ops/triton/configs/<arch>/<backend>/<op>/<d_type>/` |

Categories include `gemm/{basic,batched,feed_forward,fused}`, `attention`, `moe`, `normalization`, `quant`, `rope`, `fusions`, `comms`, `conv`, `gated_delta_net`, and `kimi_delta_attn`; inspect the closest existing family. The config `<op>` segment follows the loader's family scheme, not necessarily the wrapper's full category path.

- Do not add flat top-level op modules or launchable JIT bodies in public wrapper modules.
- Use absolute categorized imports, for example `from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16`. Legacy flat imports remain for compatibility; new code must not use them. Do not use relative imports.
- Lumen, tests, and benchmarks call the public wrapper. Private body imports/launches belong inside AITER's wrapper layer.
- HIP/CK/ASM implementations follow their existing `csrc/`, `hsa/`, `aiter/ops/` API and `op_tests/` conventions, with backend-specific configs and benchmarks. Check neighboring registration/JIT/build definitions instead of imposing Triton paths on them.

## Representation and shared helpers

Use the actual shared repr callback, not Python's builtin `repr` as a decorator:

```python
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_kernel_repr = make_kernel_repr(
    "_my_kernel",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"],
)

@triton.jit(repr=_kernel_repr)
def _my_kernel(...):
    ...
```

This is a schematic declaration: use the actual kernel signature and its meaningful specialization keys. Gluon uses `@gluon.jit(repr=_kernel_repr)`. Every independently launchable entry/stage gets repr metadata; JIT helpers that only execute inside another kernel are not separate entry kernels. When touching another backend, establish its trace-naming support rather than pretending this Python decorator applies to C++ or assembly.

- Shared activations and other common operations belong in `utils/`, not in op-local copies. Host/PyTorch helpers live in `utils/`; device/torch-free helpers live in `utils/_triton/`.
- New kernel modules and config loaders must remain torch-free. Put allocation, torch dtype handling, and annotations that require torch in public wrappers. Existing imports in old modules do not justify introducing torch to a new/previously torch-free module. Standalone `utils/_triton/tunning/` harnesses are an upstream exception.
- Reuse `utils/shuffle.py` for weights/scales; do not duplicate `view → permute → contiguous` helpers in wrappers or tests. Use `shuffle_scale_moe(..., return_layout=True)` for layout labels, and the current name `moe_weight_decode_view`.
- Split-K uses the shared `_gemm_splitk_reduce_kernel` / `_batched_gemm_splitk_reduce_kernel` in `_triton_kernels/common/splitk_reduce.py`, also for Gluon first stages.
- Select architecture behavior with `gfx*` identifiers. Allocate outputs/workspaces on the input tensor's device (`device=x.device`), not hardcoded `device="cuda"` or `.cuda()` in library code.

## Numerical tests and benchmarks

A kernel ships with its wrapper, numerical test, benchmark, and tuning files. Extend existing matching artifacts when that provides real coverage without duplication.

- Triton/Gluon tests use pytest under the matching category and shared `*_test_utils.py` / `utils/` helpers. Test forward, supported backward, supported layouts/dtypes, non-aligned sizes, and relevant edge conditions against a trusted numerical implementation. Use identical inputs for the reference and implementation.
- Exercise the wrapper's own config resolution. Do not hardcode tuning dictionaries, pass literal `config=` overrides, or run tile/autotune sweeps in correctness tests.
- Assert correctness; printing/logging a failure is not a test failure. Do not add ad-hoc reporting `__main__` blocks or tensor/timing dumps to Triton unit tests.
- Put essential failure details in assertion messages or WARNING/ERROR logs: the suite sets `AITER_LOG_LEVEL=WARNING`, hiding INFO diagnostics.
- Follow adjacent benchmark harnesses. Call the same public API and shared shuffle/config utilities. Report GPU/arch, ROCm/PyTorch/Triton versions, commits, shapes/dtypes/layouts, warmup/repetitions, latency and regression cases. Include bandwidth/utilization and arithmetic intensity/roofline evidence where applicable to a kernel PR; label estimates.
- Run targeted pytest tests and the corresponding benchmark from the AITER root. Inspect the benchmark CLI before choosing parameters. General HIP/CK tests may use standalone `op_tests/test_<op>.py`; do not copy that pattern into new Triton tests.

## Comments and logging

Use one or two explanatory sentences per comment; retain compact API information in wrapper docstrings. Avoid tutorials and narration of each line. Put longer analysis in PR or benchmark artifacts.

Use the AITER logger with lazy formatting (`logger.info("shape=%s", x.shape)`), choosing `%s` for optional values/tensors/tuples, `%d` for integer counts, `%f` for real scalars. Debug logging uses `logger.debug` and `AITER_LOG_LEVEL=DEBUG`; do not reconfigure the root logger or lower only the logger's level.

If a requested refactor changes a convention, update both the Triton README and the scoped review instructions in that PR.
