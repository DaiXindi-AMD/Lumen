"""Ablation switches for the MXFP4 optimization staircase.

Each switch names one historical performance optimization and, when turned off,
restores the code path that preceded it. Everything defaults to on, so an
unset environment reproduces HEAD exactly -- ``docs/mxfp4_ablation_plan.md``
Phase 1 checks that.

The switches exist to measure one optimization's contribution to the current
scheme, not to be shipped: they are read through this module rather than as
module-level constants so a test can flip one without reimporting the world.

Reads are cached because several of these sit inside per-GEMM dispatch. Tests
must call ``reset()`` (or use ``overridden()``) after touching the environment.
"""

import contextlib
import os

# name -> (commit that introduced the optimization, what turning it off restores)
_REGISTRY = {
    # 23644ea, 2026-07-26
    "DGRAD_WEIGHT_REUSE": (
        "23644ea",
        "backward re-quantizes the weight from ctx.weight_ref instead of "
        "reusing the pre-transposed FP4 weight forward saved",
    ),
    "FUSED_HQ_WGRAD": (
        "23644ea",
        "wgrad runs hadamard_transform then convert_to_mxfp4 as two kernels",
    ),
    # 827c941, 2026-07-28
    "DEQUANT_TRANSPOSE": (
        "827c941",
        "transpose goes through convert_from_mxfp4().t().contiguous() instead "
        "of the fused dequant+transpose kernel",
    ),
    # 95512ed, 2026-07-29 -- one commit, three independent optimizations
    "MXFP4_SHUF_BACKEND": (
        "95512ed",
        "the shuffled-layout Triton GEMM leaves the dispatch candidate list",
    ),
    "MXFP4_ASM_BACKEND": (
        "95512ed",
        "the prebuilt A4W4 ASM/CK GEMM leaves the dispatch candidate list",
    ),
    # 0bb7f8f, 2026-07-29
    "SCALE_PAD_SKIP": (
        "0bb7f8f",
        "an already-aligned scale is still padded into a fresh copy",
    ),
    "VEC_SHUFFLE": (
        "0bb7f8f",
        "the B-operand shuffle goes back to AITER's byte-wide shuffle_weight",
    ),
    # 7ef406f, 2026-07-29
    "WGRAD_VIEWS": (
        "7ef406f",
        "wgrad materializes its transposes instead of passing strided views",
    ),
    # 1be93f8, 2026-08-06 -- one squash, seven independent optimizations
    "RTN_SKIP_PHILOX": (
        "1be93f8",
        "the quantizer draws philox values RTN rounding never reads",
    ),
    # The H16 butterfly's MFMA form is deliberately absent: the Hadamard is part
    # of the recipe, so it is pinned to one setting across every arm rather than
    # ablated (docs/mxfp4_ablation_plan.md section 4.5).
    "FUSED_DHQ": (
        "1be93f8",
        "dequant, Hadamard and quantize run as separate kernels",
    ),
    "SWIZZLE_CACHE": (
        "1be93f8",
        "the scale swizzle is recomputed per call instead of cached",
    ),
    "FWD_WGRAD_OPERAND": (
        "1be93f8",
        "backward rebuilds the WGrad activation operand instead of consuming "
        "the one forward emitted",
    ),
    "DUAL_LAYOUT": (
        "1be93f8",
        "the gradient is quantized twice, once per layout",
    ),
    "QUANT_EMIT_SHUFFLE": (
        "1be93f8",
        "the B operand is shuffled at GEMM time instead of by the quantizer",
    ),
    # f01c39f, 2026-08-06 -- shared with the non-quantized path
    "NARROW_N_RMSNORM": (
        "f01c39f",
        "RMSNorm backward drops its large-M-small-N specialization",
    ),
    "ATTN_QKV_VIEWS": (
        "f01c39f",
        "attention copies its QKV operands instead of reading strided views",
    ),
    "ATTN_SEQ_MAJOR": (
        "f01c39f",
        "attention writes a batch-major output and permutes it",
    ),
}

_cache: dict[str, bool] = {}


def enabled(name):
    """True when optimization ``name`` is active. Default is on.

    Raises for an unregistered name: a mistyped switch that silently did
    nothing would produce a staircase arm that measures the wrong thing.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown ablation switch {name!r}; registered: "
            f"{', '.join(sorted(_REGISTRY))}"
        )
    hit = _cache.get(name)
    if hit is None:
        hit = os.environ.get(f"LUMEN_ABL_{name}", "1") != "0"
        _cache[name] = hit
    return hit


def reset():
    """Drop cached reads, so a later ``enabled()`` sees the environment again."""
    _cache.clear()


@contextlib.contextmanager
def overridden(**switches):
    """Run a block with switches forced, e.g. ``overridden(VEC_SHUFFLE=False)``."""
    for name in switches:
        if name not in _REGISTRY:
            raise KeyError(f"unknown ablation switch {name!r}")
    saved = dict(_cache)
    _cache.update({name: bool(value) for name, value in switches.items()})
    try:
        yield
    finally:
        _cache.clear()
        _cache.update(saved)


def active_overrides():
    """The switches currently turned off, for a run to log which arm it is."""
    return sorted(name for name in _REGISTRY if not enabled(name))


def describe(name):
    """``(commit, what turning it off restores)`` for ``name``."""
    return _REGISTRY[name]
