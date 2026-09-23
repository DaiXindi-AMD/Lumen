###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Backend dispatcher with automatic ASM → FlyDSL → Triton fallback.

Each operator registers its available backends via :func:`try_backends`.
On each call the dispatcher walks the priority chain and returns the
first successful result, logging fallbacks as warnings.
"""

import functools
import logging
import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

_SKIP_BACKEND_SYNC = os.environ.get("LUMEN_SKIP_BACKEND_SYNC", "0") == "1"

# op_name -> winning backend label; also op_name + ":hits" / ":prev" / ":warned".
_backend_cache: Dict[str, Any] = {}
_BACKEND_WARMUP_CALLS = 3

_IN_GRAPH_CAPTURE = False


def set_graph_capture_mode(active: bool):
    """Toggle graph-capture flag for dispatch safety."""
    global _IN_GRAPH_CAPTURE
    _IN_GRAPH_CAPTURE = active


try:
    from triton.compiler.errors import CompilationError as _TritonCompilationError
except ImportError:
    _TritonCompilationError = None

try:
    from triton.runtime.errors import OutOfResources as _TritonOutOfResources
except ImportError:
    _TritonOutOfResources = None

logger = logging.getLogger(__name__)


class Backend(Enum):
    ASM = "asm"
    FLYDSL = "flydsl"
    TRITON = "triton"
    HIPBLAS = "hipblas"


FALLBACK_ORDER = [Backend.ASM, Backend.FLYDSL, Backend.TRITON]


# ---------------------------------------------------------------------------
# Lazy import helpers — detect available AITER sub-packages once
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_norm():
    try:
        from aiter.ops.triton.normalization.norm import layer_norm as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_rmsnorm():
    try:
        from aiter.ops.triton.normalization.rmsnorm import rms_norm as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_gemm():
    try:
        from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_gemm_bf16():
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_tuned_gemm_bf16():
    try:
        from aiter.tuned_gemm import gemm_a16w16 as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_asm_norm():
    try:
        from aiter.ops.norm import layernorm2d_with_add_asm as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_quant():
    """Check if AITER HIP quant ops are available."""
    try:
        from aiter.ops.quant import per_tensor_quant_hip as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_quant():
    """Check if AITER Triton quant ops are available."""
    try:
        from aiter.ops.quant import per_tensor_quant_triton as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_quant_mxfp4_2way():
    """Check if AITER's two-source MXFP4 weight quantizer is available."""
    try:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant_2way as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_hipblas():
    """Check if AITER hipBLASLt GEMM ops are available."""
    try:
        from aiter.ops.gradlib import hipb_mm as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_hipblas_fp8_output() -> bool:
    """Check if ``torch._scaled_mm`` supports FP8 output dtype.

    Uses ``torch._scaled_mm`` (not ``hipb_mm``) for FP8 output because
    hipb_mm's ``out_dtype=FP8`` corrupts the hipBLASLt tuning cache.
    This probe verifies the API exists without running a real GEMM.
    """
    try:
        return hasattr(torch, "_scaled_mm")
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_fused_quant():
    try:
        from aiter.ops.triton.quant.fused_fp8_quant import fused_rms_fp8_per_tensor_static_quant as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_fused_silu_mul_fp8():
    """True if AITER fused SwiGLU + per-tensor FP8 quant kernel is importable."""
    try:
        from aiter.ops.triton.quant.fused_fp8_quant import fused_silu_mul_fp8_per_tensor_static_quant as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_gmm():
    try:
        from aiter.ops.triton.gmm import gmm as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_deepgemm():
    try:
        from aiter.ops.deepgemm import deepgemm as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_gemm_mxfp8():
    """Check if AITER Triton MXFP8 GEMM is available."""
    try:
        from aiter.ops.triton.gemm.basic.gemm_mxfp8 import gemm_mxfp8 as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_gemm_mxfp4():
    """Check if AITER Triton MXFP4 GEMM (gemm_afp4wfp4) is available."""
    try:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4 as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_gemm_mxfp4_preshuffle():
    """Check if AITER Triton MXFP4 GEMM with shuffled operand layout is available.

    Needs both the kernel and the shuffle helpers that build its layout.
    """
    try:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle as _  # noqa: F401
        from aiter.ops.triton.utils.shuffle import (  # noqa: F401
            shuffle_scale_gemm as _s,
            shuffle_weight as _w,
        )

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_gemm_mxfp4_asm():
    """Check if AITER's prebuilt A4W4 ASM MXFP4 GEMM is available.

    Needs the direct ASM entry point and the two layout helpers that build the
    operand layout the kernels read. Lumen owns the filtered ASM metadata view.
    """
    try:
        from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm as _  # noqa: F401
        from aiter.ops.shuffle import shuffle_weight as _w  # noqa: F401
        from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm as _s  # noqa: F401

        return True
    except (ImportError, OSError, RuntimeError, KeyError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_flydsl_gemm_mxfp4():
    """Check whether Lumen's pinned gfx950 FlyDSL MXFP4 kernel is importable."""
    try:
        from lumen.ops.quantize.flydsl_mxfp4 import available

        return available()
    except (ImportError, OSError, RuntimeError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_rope_cached():
    """Check if AITER Triton cached RoPE (SBHD layout) is available."""
    try:
        from aiter.ops.triton.rope.rope import rope_cached_fwd as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_rope_cached_2c():
    """Check if AITER Triton cached RoPE 2-component (THD Q+K) is available."""
    try:
        from aiter.ops.triton.rope.rope import rope_cached_thd_positions_2c_fwd as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_rope_2d():
    """Check if AITER Triton 2D RoPE is available."""
    try:
        from aiter.ops.triton.rope.rope import rope_fwd_2d as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_rope_3d():
    """Check if AITER Triton 3D RoPE is available."""
    try:
        from aiter.ops.triton.rope.rope import rope_fwd_3d as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_moe_topk_softmax():
    """Check if AITER fused topk+softmax ASM kernel is available."""
    try:
        from aiter.ops.moe_op import topk_softmax as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_softmax_topk():
    """Check if AITER softmax_topk HIP binding is available."""
    try:
        from aiter.ops.moe_op import softmax_topk as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_moe_aux_loss():
    """Check if AITER Triton moe_aux_loss kernels are available."""
    try:
        from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_bwd as _bwd  # noqa: F401
        from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_fwd as _fwd  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_moe_sorting():
    """Check if AITER moe_sorting HIP kernel is available."""
    try:
        from aiter.ops.moe_sorting import moe_sorting_fwd as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_moe_sum():
    """Check if AITER moe_sum ASM kernel is available."""
    try:
        from aiter.ops.moe_op import moe_sum as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_moe_gemm_per_token():
    """Check if AITER fused per-token MOE GEMM is available."""
    try:
        from aiter.ops.triton.moe.moe_gemm_per_token import moe_gemm_per_token as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_moe_gemm_mxfp8():
    """Check if AITER fused MXFP8 MOE GEMM is available."""
    try:
        from aiter.ops.triton.moe.moe_gemm_mxfp8 import moe_gemm_mxfp8 as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_moe_align():
    """Check if AITER Triton MoE block alignment kernel is available."""
    try:
        from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_triton_fused_moe():
    """Check if AITER Triton fused MoE (sort+GEMM) kernel is available."""
    try:
        from aiter.ops.triton.moe.moe_op import fused_moe as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_fused_gated():
    """Check if AITER Triton fused gated feed-forward is available."""
    try:
        from aiter.ops.triton.gemm.feed_forward import ff_a16w16_fused_gated as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_fused_ungated():
    """Check if AITER Triton fused ungated feed-forward is available."""
    try:
        from aiter.ops.triton.gemm.feed_forward import ff_a16w16_fused_ungated as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_swiglu():
    """Check if AITER Triton fused SwiGLU fwd/bwd kernels are available."""
    try:
        from aiter.ops.triton.activation import (  # noqa: F401
            swiglu_bwd as _swiglu_bwd,
            swiglu_fwd as _swiglu_fwd,
        )

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_swiglu_split():
    """Check if AITER's eager-compatible split SwiGLU kernels are available."""
    try:
        from aiter.ops.triton.activation import (  # noqa: F401
            swiglu_bwd_split as _swiglu_bwd_split,
            swiglu_fwd_split as _swiglu_fwd_split,
        )

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_fast_transpose():
    """Check if AITER Triton fast 2D transpose kernel is available."""
    try:
        from aiter.ops.triton.quant.fast_transpose import fast_transpose_2d as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_fused_add_rmsnorm_pad():
    """Check if AITER Triton fused add+RMSNorm+pad kernel is available."""
    try:
        from aiter.ops.triton.normalization.fused_add_rmsnorm_pad import fused_add_rmsnorm_pad as _  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


@functools.lru_cache(maxsize=1)
def _probe_aiter_fused_gemm_blockscale_mul_add():
    """Check if AITER fused blockscale GEMM + mul/add epilogue is available."""
    try:
        from aiter.ops.triton.gemm.fused.fused_gemm_a8w8_blockscale_mul_add import (  # noqa: F401
            fused_gemm_a8w8_blockscale_mul_add as _,
        )

        return True
    except (ImportError, OSError):
        return False


# ---------------------------------------------------------------------------
# Core fallback dispatcher
# ---------------------------------------------------------------------------


def _entry_label(entry: Tuple) -> str:
    """Stable identity for one fallback-chain entry.

    ``Backend`` on its own is not unique: ``gemm_mxfp4`` offers three
    ``Backend.TRITON`` entries (preshuffled, row-major, and the dequant→BF16
    fallback), so a chain that reuses a ``Backend`` passes an explicit label as
    a third tuple element.
    """
    if len(entry) >= 3:
        return entry[2]
    return entry[0].value


def try_backends(
    backends: List[Tuple[Backend, Callable]],
    *args,
    op_name: str = "op",
    slow_labels: Sequence[str] = (),
    **kwargs,
) -> Any:
    """Try each ``(backend, fn)`` pair in order; return first success.

    On ``RuntimeError``, ``NotImplementedError``, ``TypeError``,
    ``ValueError``, ``IndexError``, ``KeyError``, or Triton
    ``CompilationError`` from a backend, logs a warning and falls
    through to the next.  ``IndexError`` / ``KeyError`` are included
    because AITER JIT wrappers raise ``map::at`` when a kernel config
    lookup fails.  If all fail, raises the last exception.

    After a backend succeeds ``_BACKEND_WARMUP_CALLS`` consecutive times
    for a given ``op_name``, the winning backend is cached by *label* and
    subsequent calls skip the fallback chain entirely.  Caching the label
    rather than the list position matters for any op that rebuilds its chain
    per call: a position means a different backend once the order or length
    changes, so an index cached from one shape can hand a later shape the
    fallback -- or a kernel that shape is not allowed to use.  A label that is
    no longer on offer simply misses and re-runs the chain.

    Labels named in ``slow_labels`` are reported at ``warning`` rather than
    ``debug`` when they win, because locking onto a degraded path is not
    something a run should have to read debug logs to discover.

    ``torch.cuda.synchronize()`` is issued during warmup for error
    detection.  After warmup (or when ``LUMEN_SKIP_BACKEND_SYNC=1``),
    sync is skipped to reduce host-device round-trip overhead.
    """
    _catchable = (
        RuntimeError,
        NotImplementedError,
        TypeError,
        ValueError,
        IndexError,
        KeyError,
    )
    if _TritonCompilationError is not None:
        _catchable = _catchable + (_TritonCompilationError,)
    if _TritonOutOfResources is not None:
        _catchable = _catchable + (_TritonOutOfResources,)

    cached_label = _backend_cache.get(op_name)
    if cached_label is not None:
        for entry in backends:
            if _entry_label(entry) == cached_label:
                try:
                    return entry[1](*args, **kwargs)
                except _catchable as exc:
                    # The lock is a warmup shortcut, not a promise that these
                    # operands still suit the winner. Forget it and re-run the
                    # chain rather than propagating, so the degradation the
                    # chain exists to provide survives warmup.
                    logger.warning(
                        "%s: cached %s backend failed (%s), re-running the fallback chain",
                        op_name,
                        cached_label,
                        exc,
                    )
                    for suffix in ("", ":hits", ":prev"):
                        _backend_cache.pop(op_name + suffix, None)
                break

    if _IN_GRAPH_CAPTURE:
        raise RuntimeError(
            f"{op_name}: no cached backend during CUDA graph capture. "
            "Run warmup before graph capture."
        )

    last_exc = None
    for entry in backends:
        backend, fn = entry[0], entry[1]
        label = _entry_label(entry)
        try:
            result = fn(*args, **kwargs)
            if (
                torch.cuda.is_available()
                and not _SKIP_BACKEND_SYNC
                and not _IN_GRAPH_CAPTURE
            ):
                torch.cuda.synchronize()

            # Count consecutive wins for *this* backend. Reading the running
            # count before comparing would let a switch of backend inherit the
            # previous one's streak and lock on its first success.
            if _backend_cache.get(op_name + ":prev") == label:
                hit_count = _backend_cache.get(op_name + ":hits", 0) + 1
            else:
                hit_count = 1
            _backend_cache[op_name + ":hits"] = hit_count
            _backend_cache[op_name + ":prev"] = label

            if hit_count >= _BACKEND_WARMUP_CALLS:
                _backend_cache[op_name] = label
                if label in slow_labels and not _backend_cache.get(op_name + ":warned"):
                    _backend_cache[op_name + ":warned"] = True
                    logger.warning(
                        "%s: locked to the %s fallback after %d successes -- every later "
                        "call with this operand shape takes it. The faster backends were "
                        "tried first and failed; check the warnings above.",
                        op_name,
                        label,
                        hit_count,
                    )
                else:
                    logger.debug(
                        "%s: locked to %s backend (%s) after %d successes",
                        op_name,
                        backend.value,
                        label,
                        hit_count,
                    )

            return result
        except _catchable as exc:
            logger.warning(
                "%s: %s backend failed (%s), trying next...",
                op_name,
                backend.value,
                exc,
            )
            # Deliberately *not* resetting the streak here. The winner's count
            # has to survive a failure earlier in the chain, or a backend that
            # is never first can never lock -- which left every call paying the
            # cost of a kernel known to reject these operands, plus a warning.
            last_exc = exc
    raise RuntimeError(
        f"{op_name}: all backends exhausted. Last error: {last_exc}"
    ) from last_exc


def build_fallback_chain(
    candidates: Dict[Backend, Optional[Callable]],
    order: List[Backend] = FALLBACK_ORDER,
) -> List[Tuple[Backend, Callable]]:
    """Build an ordered list of (backend, callable) from candidates.

    ``None`` values (unavailable backends) are skipped.
    """
    chain = []
    for b in order:
        fn = candidates.get(b)
        if fn is not None:
            chain.append((b, fn))
    return chain
