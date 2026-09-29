# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fresh exact-shape MXFP4 gate/up packing feasibility benchmark.

This benchmark compares the Qwen3-8B-sized MLP input projection in two forms:

* A: two independently patched ``nn.Linear`` modules followed by eager
  ``silu(gate) * up``.
* B: one wide projection backed by the original gate/up Parameters and the
  production ``_mxfp4_cached_weight_pair`` / ``mxfp4_gate_up_linear`` APIs.

Both arms exercise the production MXFP4 cache, autograd, and GEMM dispatcher.
Legacy ``parameter`` and ``cat`` modes remain as diagnostic oracles, but only
``pair`` represents the proposed production parameter and cache lifecycle.

The output directory and MXFP4 autotune cache must be new.  This prevents a
previous run's backend decisions from being mistaken for cold-start evidence.
No result from this microbenchmark is an end-to-end training-step claim.

Examples::

    python -m benchmarks.bench_mxfp4_gate_up --dry-run
    python -m benchmarks.bench_mxfp4_gate_up \
        --output-dir /tmp/lumen-mxfp4-gate-up-001
    python -m benchmarks.bench_mxfp4_gate_up \
        --output-dir /tmp/lumen-mxfp4-gate-up-trace-001 --trace
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.bench_utils import (
    BenchResult,
    cuda_timer,
    print_report_with_table,
    trace_fn,
)


M = 16_384
H = 4_096
INTERMEDIATE = 12_288
PACKED_INTERMEDIATE = 2 * INTERMEDIATE
MXFP4_BLOCK_SIZE = 32

LOGICAL_GEMM_SHAPES = {
    "forward_current": (M, INTERMEDIATE, H),
    "forward_packed": (M, PACKED_INTERMEDIATE, H),
    "dgrad_current": (M, H, INTERMEDIATE),
    "dgrad_packed": (M, H, PACKED_INTERMEDIATE),
    "wgrad_current": (INTERMEDIATE, H, M),
    "wgrad_packed": (PACKED_INTERMEDIATE, H, M),
}

EXPECTED_CALL_COUNTS = {
    "current": {
        "quantized_linear_forward": 2,
        "forward_gemm": 2,
        "dgrad_gemm": 2,
        "wgrad_gemm": 2,
        "swiglu": 1,
    },
    "packed": {
        "quantized_linear_forward": 1,
        "forward_gemm": 1,
        "dgrad_gemm": 1,
        "wgrad_gemm": 1,
        "split_views": 2,
        "swiglu": 1,
    },
}


class GateUpBank(nn.Module):
    """Own the two production linears and the one-wide feasibility oracle."""

    def __init__(self, device: torch.device):
        super().__init__()
        kwargs = {"bias": False, "device": device, "dtype": torch.bfloat16}
        self.gate = nn.Linear(H, INTERMEDIATE, **kwargs)
        self.up = nn.Linear(H, INTERMEDIATE, **kwargs)
        self.packed = nn.Linear(H, PACKED_INTERMEDIATE, **kwargs)

        with torch.no_grad():
            self.gate.weight.normal_(mean=0.0, std=0.02)
            self.up.weight.normal_(mean=0.0, std=0.02)
            # Avoid charging this one-time oracle construction to warm timings.
            self.packed.weight[:INTERMEDIATE].copy_(self.gate.weight)
            self.packed.weight[INTERMEDIATE:].copy_(self.up.weight)


class _AiterSplitSwiGLU(torch.autograd.Function):
    """Autograd bridge for AITER's eager-equivalent split SwiGLU kernels."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        from aiter.ops.triton.activation import swiglu_fwd_split

        ctx.save_for_backward(gate, up)
        return swiglu_fwd_split(gate, up)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from aiter.ops.triton.activation import swiglu_bwd_split

        gate, up = ctx.saved_tensors
        return swiglu_bwd_split(grad_output.contiguous(), gate, up)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_even_int(value: str) -> int:
    parsed = _positive_int(value)
    if parsed % 2:
        raise argparse.ArgumentTypeError("must be even for order-balanced pairs")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _trim_percentage(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < 50.0:
        raise argparse.ArgumentTypeError("must be in [0, 50)")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new directory for results, traces, autotune cache, and shape log",
    )
    parser.add_argument("--device", type=_nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=20_260_921)
    parser.add_argument("--warmup", type=_nonnegative_int, default=10)
    parser.add_argument("--iters", type=_positive_int, default=30)
    parser.add_argument(
        "--paired-iters",
        type=_positive_even_int,
        default=64,
        help="interleaved ABBA/BAAB full forward+backward sample pairs",
    )
    parser.add_argument(
        "--batch-repeats",
        type=_positive_int,
        default=32,
        help="full forward+backward calls per batched throughput arm",
    )
    parser.add_argument(
        "--batch-pairs",
        type=_positive_even_int,
        default=12,
        help="order-balanced current/packed batched-throughput pairs",
    )
    parser.add_argument(
        "--paired-disable-python-gc",
        action="store_true",
        help="diagnostic only: disable Python cyclic GC during paired measurements",
    )
    parser.add_argument("--trim-pct", type=_trim_percentage, default=10.0)
    parser.add_argument(
        "--arm-order",
        choices=("ab", "ba"),
        default="ab",
        help="warm measurement order: A=current and B=packed",
    )
    parser.add_argument(
        "--dispatch-mode",
        choices=("fallback-chain", "fast"),
        default="fallback-chain",
        help=(
            "fallback-chain records the actually successful runtime backend; "
            "fast matches LUMEN_FAST_QUANT_DISPATCH=1 but exposes only the "
            "autotuner selection"
        ),
    )
    parser.add_argument(
        "--current-swiglu",
        choices=("eager", "aiter"),
        default="eager",
        help="SwiGLU implementation for the two-linear current arm",
    )
    parser.add_argument(
        "--packed-swiglu",
        choices=("eager", "aiter"),
        default="eager",
        help="SwiGLU implementation for the one-wide-linear packed arm",
    )
    parser.add_argument(
        "--packed-weight-source",
        choices=("pair", "parameter", "cat"),
        default="pair",
        help=(
            "pair uses the production two-Parameter packed API; parameter uses "
            "the one-wide feasibility Parameter; cat materializes a transient "
            "BF16 concatenation before reusing one parent-owned MXFP4 cache"
        ),
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="skip the exact-shape BF16/current/packed SNR report",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="write one current and one packed full-forward/backward trace",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print shapes and expected calls without importing Lumen or using a GPU",
    )
    return parser


def _shape_payload() -> dict[str, Any]:
    return {
        "dimensions": {
            "tokens_m": M,
            "hidden_h": H,
            "intermediate_i": INTERMEDIATE,
            "packed_intermediate_2i": PACKED_INTERMEDIATE,
            "mxfp4_block_size": MXFP4_BLOCK_SIZE,
        },
        "tensors": {
            "input": [M, H],
            "gate_weight": [INTERMEDIATE, H],
            "up_weight": [INTERMEDIATE, H],
            "packed_weight": [PACKED_INTERMEDIATE, H],
            "swiglu_output": [M, INTERMEDIATE],
        },
        "logical_gemm_shapes_mnk": {
            name: list(shape) for name, shape in LOGICAL_GEMM_SHAPES.items()
        },
        "expected_calls_per_forward_backward": EXPECTED_CALL_COUNTS,
    }


def _dry_run(args: argparse.Namespace) -> None:
    arguments = vars(args).copy()
    arguments["output_dir"] = (
        str(args.output_dir) if args.output_dir is not None else None
    )
    payload = {
        "mode": "dry-run",
        "uses_gpu": False,
        "production_path": (
            "nn.Linear -> lumen.quantize.enable(format='mxfp4') -> quantized_linear"
        ),
        "comparison": {
            "a": f"two independent linears + {args.current_swiglu} SwiGLU",
            "b": (
                f"one logical wide linear from {args.packed_weight_source} + "
                f"{args.packed_swiglu} SwiGLU"
            ),
        },
        "arguments": arguments,
        **_shape_payload(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _prepare_fresh_output(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --dry-run is used")

    world_size = os.environ.get("WORLD_SIZE", "1")
    if world_size != "1":
        raise RuntimeError(
            "this is a single-GPU microbenchmark; WORLD_SIZE must be 1, "
            f"got {world_size!r}"
        )

    if "lumen.ops.quantize.mxfp4_autotune" in sys.modules:
        raise RuntimeError(
            "MXFP4 autotune was imported before fresh paths were configured; "
            "run this benchmark in a new Python process"
        )

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    # Keep the exact executed benchmark alongside every result.  This file is
    # intentionally an untracked experiment, so a Git commit alone cannot
    # reconstruct the measured source after later edits.
    shutil.copy2(
        Path(__file__).resolve(),
        output_dir / "bench_mxfp4_gate_up.executed.py",
    )

    autotune_cache = output_dir / "mxfp4_autotune.json"
    shape_log = output_dir / "mxfp4_shapes.csv"
    os.environ["LUMEN_MXFP4_AUTOTUNE"] = "1"
    os.environ["LUMEN_MXFP4_AUTOTUNE_CACHE"] = str(autotune_cache)
    os.environ["LUMEN_MXFP4_GEMM_SHAPE_LOG"] = str(shape_log)
    os.environ["LUMEN_FAST_QUANT_DISPATCH"] = (
        "0" if args.dispatch_mode == "fallback-chain" else "1"
    )
    # ``cuda_timer`` gives these environment variables precedence over call
    # arguments. Remove inherited values so the CLI remains authoritative and
    # the explicit zero-warmup/one-iteration cold observations stay cold.
    for variable in (
        "LUMEN_BENCH_WARMUP",
        "LUMEN_BENCH_ITERS",
        "LUMEN_BENCH_TRIM_PCT",
    ):
        os.environ.pop(variable, None)
    return output_dir, autotune_cache, shape_log


def _load_lumen_api() -> dict[str, Any]:
    """Import Lumen only after fresh autotune paths are in the environment."""
    try:
        import aiter
    except (ImportError, OSError) as exc:
        raise RuntimeError("AITER is required for the production MXFP4 path") from exc

    import lumen.quantize as quant
    from lumen.ops import dispatch as lumen_dispatch
    from lumen.ops.quantize import mxfp4_autotune
    from lumen.ops.quantize import linear as quantized_linear_module

    return {
        "aiter": aiter,
        "dispatch": lumen_dispatch,
        "linear": quantized_linear_module,
        "mxfp4_autotune": mxfp4_autotune,
        "quant": quant,
    }


def _clear_mxfp4_weight_cache(owner: Any) -> None:
    for attribute in (
        "_mxfp4_w_cache",
        "_mxfp4_w_cache_version",
        "_mxfp4_w_cache_sources",
    ):
        if hasattr(owner, attribute):
            delattr(owner, attribute)


def _bench_result_payload(result: BenchResult) -> dict[str, Any]:
    return dataclasses.asdict(result)


def _comparison_payload(current: BenchResult, packed: BenchResult) -> dict[str, Any]:
    return {
        "a_current_ms": current.avg_ms,
        "b_packed_ms": packed.avg_ms,
        "speedup_a_over_b": current.avg_ms / packed.avg_ms,
        "saving_ms": current.avg_ms - packed.avg_ms,
    }


def _measure_arms(
    arm_order: str,
    current: Callable[[], BenchResult],
    packed: Callable[[], BenchResult],
) -> tuple[BenchResult, BenchResult]:
    """Measure A/B in the requested order while returning canonical A, B."""
    if arm_order == "ab":
        current_result = current()
        packed_result = packed()
    else:
        packed_result = packed()
        current_result = current()
    return current_result, packed_result


def _run_arms(
    arm_order: str, current: Callable[[], Any], packed: Callable[[], Any]
) -> None:
    """Run untimed arm setup in the same order as the warm measurements."""
    if arm_order == "ab":
        current()
        packed()
    else:
        packed()
        current()


def _measure_interleaved_full(
    *,
    current: Callable[[], Any],
    packed: Callable[[], Any],
    initial_order: str,
    warmup: int,
    iters: int,
    disable_python_gc: bool,
) -> dict[str, Any]:
    """Measure adjacent current/packed calls in an order-balanced sequence."""

    def order_for(index: int) -> str:
        # Starting with AB gives ABBA over four pairs; starting with BA gives
        # BAAB. Both arms therefore occupy first and second position equally.
        base = ("ab", "ba", "ba", "ab")
        order = base[index % len(base)]
        if initial_order == "ab":
            return order
        return order.translate(str.maketrans("ab", "ba"))

    gc.collect()
    gc_events: list[dict[str, Any]] = []
    active_call: dict[str, Any] = {"phase": "idle", "index": None, "arm": None}

    def gc_callback(phase: str, info: dict[str, Any]) -> None:
        gc_events.append(
            {
                "phase": phase,
                "generation": info.get("generation"),
                "collected": info.get("collected"),
                "uncollectable": info.get("uncollectable"),
                **active_call,
            }
        )

    gc_was_enabled = gc.isenabled()
    gc.callbacks.append(gc_callback)
    if disable_python_gc:
        gc.disable()
    current_ms: list[float] = []
    packed_ms: list[float] = []
    pair_orders: list[str] = []
    try:
        for index in range(warmup):
            for label in order_for(index):
                active_call.update(phase="warmup", index=index, arm=label)
                (current if label == "a" else packed)()
        torch.cuda.synchronize()

        for index in range(iters):
            pair_order = order_for(index)
            pair_orders.append(pair_order)
            for label in pair_order:
                active_call.update(phase="timed", index=index, arm=label)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                (current if label == "a" else packed)()
                end.record()
                end.synchronize()
                elapsed = float(start.elapsed_time(end))
                (current_ms if label == "a" else packed_ms).append(elapsed)
    finally:
        active_call.update(phase="idle", index=None, arm=None)
        if gc_callback in gc.callbacks:
            gc.callbacks.remove(gc_callback)
        if gc_was_enabled and not gc.isenabled():
            gc.enable()

    savings = [a - b for a, b in zip(current_ms, packed_ms)]
    ratios = [a / b for a, b in zip(current_ms, packed_ms)]
    packed_wins = sum(value > 0.0 for value in savings)
    current_wins = sum(value < 0.0 for value in savings)
    return {
        "pair_orders": pair_orders,
        "current_ms": current_ms,
        "packed_ms": packed_ms,
        "paired_saving_ms": savings,
        "paired_speedup": ratios,
        "current_mean_ms": sum(current_ms) / len(current_ms),
        "packed_mean_ms": sum(packed_ms) / len(packed_ms),
        "ratio_of_means": sum(current_ms) / sum(packed_ms),
        "current_median_ms": statistics.median(current_ms),
        "packed_median_ms": statistics.median(packed_ms),
        "median_paired_saving_ms": statistics.median(savings),
        "median_paired_speedup": statistics.median(ratios),
        "current_wins": current_wins,
        "packed_wins": packed_wins,
        "ties": len(savings) - current_wins - packed_wins,
        "pairs": len(savings),
        "python_gc_disabled": disable_python_gc,
        "python_gc_events": gc_events,
    }


def _measure_batched_full(
    *,
    current: Callable[[], Any],
    packed: Callable[[], Any],
    initial_order: str,
    repeats: int,
    pairs: int,
    disable_python_gc: bool,
) -> dict[str, Any]:
    """Measure layer-chain throughput with many calls inside each event pair."""

    def order_for(index: int) -> str:
        base = ("ab", "ba", "ba", "ab")
        order = base[index % len(base)]
        if initial_order == "ab":
            return order
        return order.translate(str.maketrans("ab", "ba"))

    gc.collect()
    gc_was_enabled = gc.isenabled()
    if disable_python_gc:
        gc.disable()
    try:
        for label in order_for(0):
            fn = current if label == "a" else packed
            for _ in range(repeats):
                fn()
        torch.cuda.synchronize()

        current_ms: list[float] = []
        packed_ms: list[float] = []
        pair_orders: list[str] = []
        for index in range(pairs):
            pair_order = order_for(index)
            pair_orders.append(pair_order)
            for label in pair_order:
                fn = current if label == "a" else packed
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):
                    fn()
                end.record()
                end.synchronize()
                elapsed_per_call = float(start.elapsed_time(end)) / repeats
                (current_ms if label == "a" else packed_ms).append(elapsed_per_call)
    finally:
        if gc_was_enabled and not gc.isenabled():
            gc.enable()

    savings = [a - b for a, b in zip(current_ms, packed_ms)]
    ratios = [a / b for a, b in zip(current_ms, packed_ms)]
    packed_wins = sum(value > 0.0 for value in savings)
    current_wins = sum(value < 0.0 for value in savings)
    return {
        "pair_orders": pair_orders,
        "repeats_per_arm": repeats,
        "pairs": pairs,
        "current_ms_per_call": current_ms,
        "packed_ms_per_call": packed_ms,
        "paired_saving_ms_per_call": savings,
        "paired_speedup": ratios,
        "current_mean_ms_per_call": sum(current_ms) / len(current_ms),
        "packed_mean_ms_per_call": sum(packed_ms) / len(packed_ms),
        "ratio_of_means": sum(current_ms) / sum(packed_ms),
        "median_paired_saving_ms_per_call": statistics.median(savings),
        "median_paired_speedup": statistics.median(ratios),
        "current_wins": current_wins,
        "packed_wins": packed_wins,
        "ties": len(savings) - current_wins - packed_wins,
        "python_gc_disabled": disable_python_gc,
    }


def _tensor_error(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    if reference.shape != actual.shape:
        raise RuntimeError(
            f"correctness shape mismatch: {tuple(reference.shape)} != {tuple(actual.shape)}"
        )

    # Use FP32 reductions so the report measures quantization error rather than
    # overflowing a BF16 sum.  Correctness runs outside every timed region.
    reference_fp32 = reference.detach().float()
    actual_fp32 = actual.detach().float()
    difference = actual_fp32 - reference_fp32
    reference_norm = float(torch.linalg.vector_norm(reference_fp32).item())
    actual_norm = float(torch.linalg.vector_norm(actual_fp32).item())
    error_norm = float(torch.linalg.vector_norm(difference).item())
    difference.abs_()
    max_abs = float(difference.max().item())
    del reference_fp32, actual_fp32, difference

    if error_norm == 0.0:
        snr_db = float("inf")
    elif reference_norm == 0.0:
        snr_db = float("-inf")
    else:
        snr_db = 20.0 * math.log10(reference_norm / error_norm)
    relative_l2 = 0.0 if reference_norm == 0.0 else error_norm / reference_norm
    return {
        "snr_db": snr_db,
        "relative_l2": relative_l2,
        "max_abs": max_abs,
        "finite": all(
            math.isfinite(value)
            for value in (reference_norm, actual_norm, error_norm, max_abs)
        ),
    }


def _correctness_report(
    current_forward: Callable[[], torch.Tensor],
    packed_forward: Callable[[], torch.Tensor],
    x: torch.Tensor,
    dy: torch.Tensor,
    bank: GateUpBank,
    seed: int,
    packed_weight_source: str,
) -> dict[str, Any]:
    """Compare final output, dX, and split dW against BF16 and each other."""
    reference_output = F.silu(F.linear(x, bank.gate.weight)) * F.linear(
        x, bank.up.weight
    )
    reference_dx, reference_dw_gate, reference_dw_up = torch.autograd.grad(
        reference_output,
        (x, bank.gate.weight, bank.up.weight),
        grad_outputs=dy,
    )

    random.seed(seed + 1)
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 1)
    current_output = current_forward()
    current_dx, current_dw_gate, current_dw_up = torch.autograd.grad(
        current_output,
        (x, bank.gate.weight, bank.up.weight),
        grad_outputs=dy,
    )

    random.seed(seed + 1)
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 1)
    packed_output = packed_forward()
    if packed_weight_source == "parameter":
        packed_dx, packed_dw = torch.autograd.grad(
            packed_output,
            (x, bank.packed.weight),
            grad_outputs=dy,
        )
        packed_dw_gate, packed_dw_up = packed_dw.split(INTERMEDIATE, dim=0)
    else:
        packed_dx, packed_dw_gate, packed_dw_up = torch.autograd.grad(
            packed_output,
            (x, bank.gate.weight, bank.up.weight),
            grad_outputs=dy,
        )

    groups = {
        "current_vs_bf16": (
            ("output", reference_output, current_output),
            ("dinput", reference_dx, current_dx),
            ("dweight_gate", reference_dw_gate, current_dw_gate),
            ("dweight_up", reference_dw_up, current_dw_up),
        ),
        "packed_vs_bf16": (
            ("output", reference_output, packed_output),
            ("dinput", reference_dx, packed_dx),
            ("dweight_gate", reference_dw_gate, packed_dw_gate),
            ("dweight_up", reference_dw_up, packed_dw_up),
        ),
        "packed_vs_current": (
            ("output", current_output, packed_output),
            ("dinput", current_dx, packed_dx),
            ("dweight_gate", current_dw_gate, packed_dw_gate),
            ("dweight_up", current_dw_up, packed_dw_up),
        ),
    }
    report = {
        comparison: {
            name: _tensor_error(reference, actual)
            for name, reference, actual in tensors
        }
        for comparison, tensors in groups.items()
    }
    report["packed_dweight_split"] = {
        "gate_shape": list(packed_dw_gate.shape),
        "up_shape": list(packed_dw_up.shape),
        "gate_is_view": packed_dw_gate._base is not None,
        "up_is_view": packed_dw_up._base is not None,
    }

    finite = all(
        metric["finite"]
        for comparison in groups
        for metric in report[comparison].values()
    )
    report["all_finite"] = finite
    if not finite:
        raise RuntimeError("non-finite value found in gate/up correctness report")
    return report


def _runtime_dispatch_report(
    dispatch_module: Any,
    autotune_module: Any,
) -> dict[str, Any]:
    """Report selected and actually successful backends without mutating state."""
    dispatch_cache = getattr(dispatch_module, "_backend_cache", {})
    report: dict[str, Any] = {}
    suffixes = (":hits", ":prev", ":warned")

    for phase, shape in LOGICAL_GEMM_SHAPES.items():
        prefix = f"gemm_mxfp4:{shape[0]}x{shape[1]}x{shape[2]}:"
        base_names = set()
        for key in dispatch_cache:
            if not isinstance(key, str) or not key.startswith(prefix):
                continue
            base = key
            for suffix in suffixes:
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            base_names.add(base)

        runtime_paths = []
        observed_labels = set()
        for base in sorted(base_names):
            locked = dispatch_cache.get(base)
            previous = dispatch_cache.get(base + ":prev")
            if locked is not None:
                observed_labels.add(str(locked))
            if previous is not None:
                observed_labels.add(str(previous))
            runtime_paths.append(
                {
                    "op_name": base,
                    "locked_backend": locked,
                    "last_successful_backend": previous,
                    "consecutive_successes": dispatch_cache.get(base + ":hits", 0),
                    "slow_fallback_warning_emitted": bool(
                        dispatch_cache.get(base + ":warned", False)
                    ),
                }
            )

        selected = autotune_module.cached(shape)
        runtime_observed = bool(observed_labels)
        report[phase] = {
            "shape_mnk": list(shape),
            "selected_backend": selected,
            "autotune_profile": autotune_module.cached_profile(shape),
            "runtime_paths": runtime_paths,
            "runtime_backends_observed": sorted(observed_labels),
            "runtime_observation_available": runtime_observed,
            "runtime_fallback_observed": (
                any(label != selected for label in observed_labels)
                if runtime_observed
                else None
            ),
            "dequant_bf16_fallback_observed": (
                "dequant_bf16" in observed_labels if runtime_observed else None
            ),
        }
    return report


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _command_output(command: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"unavailable: {exc}"
    if completed.returncode != 0:
        return f"unavailable (exit {completed.returncode}): {completed.stderr.strip()}"
    return completed.stdout.strip()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_bytes(command: list[str], cwd: Path) -> bytes:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        return f"unavailable: {exc}".encode()
    if completed.returncode != 0:
        return (
            f"unavailable (exit {completed.returncode}): "
            f"{completed.stderr.decode(errors='replace').strip()}"
        ).encode()
    return completed.stdout


def _archive_git_state(repo_root: Path, prefix: str, output_dir: Path) -> dict[str, str]:
    head = _git_bytes(["git", "rev-parse", "HEAD"], repo_root)
    status = _git_bytes(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal", "-z"],
        repo_root,
    )
    diff = _git_bytes(["git", "diff", "--binary"], repo_root)
    files = {
        "head": output_dir / f"{prefix}_head.txt",
        "status": output_dir / f"{prefix}_status_porcelain_z.bin",
        "diff": output_dir / f"{prefix}_tracked_diff.patch",
    }
    files["head"].write_bytes(head)
    files["status"].write_bytes(status)
    files["diff"].write_bytes(diff)
    return {
        f"{name}_path": str(path)
        for name, path in files.items()
    } | {
        f"{name}_sha256": _sha256_path(path)
        for name, path in files.items()
    }


def _repository_root(source_file: str | None) -> Path | None:
    if source_file is None:
        return None
    source_path = Path(source_file).resolve()
    return next((parent for parent in source_path.parents if (parent / ".git").exists()), None)


def _loaded_aiter_binary_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if not source:
            continue
        path = Path(source).resolve()
        if path.suffix not in (".so", ".hsaco") or "/aiter/" not in str(path):
            continue
        if path.is_file():
            hashes[str(path)] = _sha256_path(path)
    return dict(sorted(hashes.items()))


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _time_backward_only(
    *,
    label: str,
    forward: Callable[[], torch.Tensor],
    parameters: tuple[torch.Tensor, ...],
    dy: torch.Tensor,
    warmup: int,
    iters: int,
    trim_pct: float,
    split_packed_weight_grad: bool,
) -> BenchResult:
    """Build once, then time only autograd over a retained production graph."""
    output = forward()
    torch.cuda.synchronize()

    def backward() -> tuple[torch.Tensor, ...]:
        gradients = torch.autograd.grad(
            output,
            parameters,
            grad_outputs=dy,
            retain_graph=True,
        )
        if not split_packed_weight_grad:
            return gradients
        dinput, packed_dweight = gradients
        dweight_gate, dweight_up = packed_dweight.split(INTERMEDIATE, dim=0)
        return dinput, dweight_gate, dweight_up

    return cuda_timer(
        backward,
        warmup=warmup,
        iters=iters,
        trim_pct=trim_pct,
        label=label,
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is required; use --dry-run for CPU validation")
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("this PyTorch build has no float4_e2m1fn_x2 dtype")
    if args.device >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {args.device} does not exist; "
            f"visible device count is {torch.cuda.device_count()}"
        )

    output_dir, autotune_cache, shape_log = _prepare_fresh_output(args)
    api = _load_lumen_api()
    quant = api["quant"]
    repo_root = Path(__file__).resolve().parents[1]
    aiter_root = _repository_root(getattr(api["aiter"], "__file__", None))
    git_archives = {
        "lumen": _archive_git_state(repo_root, "lumen", output_dir),
    }
    if aiter_root is not None:
        git_archives["aiter"] = _archive_git_state(
            aiter_root, "aiter", output_dir
        )

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    quant.assert_mxfp4_arch_supported()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()

    bank = GateUpBank(device)
    manager = quant.enable(
        bank,
        format="mxfp4",
        scaling="blockwise",
        block_size=MXFP4_BLOCK_SIZE,
    )
    for name, layer in (
        ("gate", bank.gate),
        ("up", bank.up),
        ("packed", bank.packed),
    ):
        if not getattr(layer, "_quant_enabled", False):
            raise RuntimeError(f"quant.enable did not patch the {name} nn.Linear")
        if getattr(layer, "_quant_scaling_type", None) != "mxfp4":
            raise RuntimeError(f"{name} nn.Linear is not on the MXFP4 path")

    packed_cache_owner = nn.Module()
    packed_tensor_id = "gate_up_packed.weight"
    packed_activation_tensor_id = "gate_up_packed.activation"
    packed_fp8_dtype = bank.gate._lumen_fp8_dtype

    def packed_projection(input_tensor: torch.Tensor) -> torch.Tensor:
        if args.packed_weight_source == "parameter":
            return bank.packed(input_tensor)

        if args.packed_weight_source == "pair":
            weight_cache, weight_scale = quant._mxfp4_cached_weight_pair(
                packed_cache_owner,
                bank.gate.weight,
                bank.up.weight,
                packed_fp8_dtype,
                MXFP4_BLOCK_SIZE,
                gemm_rows=input_tensor.numel() // input_tensor.shape[-1],
            )
            return api["linear"].mxfp4_gate_up_linear(
                input_tensor,
                bank.gate.weight,
                bank.up.weight,
                weight_cache,
                weight_scale,
                scaling_manager=manager,
                fp8_dtype=packed_fp8_dtype,
                block_size=MXFP4_BLOCK_SIZE,
                tensor_id=packed_tensor_id,
            )

        # Legacy diagnostic: retain two Parameters but materialize a BF16 cat.
        packed_weight = torch.cat((bank.gate.weight, bank.up.weight), dim=0)
        weight_cache, weight_scale = quant._mxfp4_cached_weight(
            packed_cache_owner,
            packed_weight,
            None,
            None,
            "mxfp4",
            packed_fp8_dtype,
            MXFP4_BLOCK_SIZE,
            gemm_rows=input_tensor.numel() // input_tensor.shape[-1],
        )
        return api["linear"].quantized_linear(
            input_tensor,
            packed_weight,
            None,
            scaling_manager=manager,
            backend="auto",
            scaling_type="mxfp4",
            fp8_dtype=packed_fp8_dtype,
            block_size=MXFP4_BLOCK_SIZE,
            tensor_id=packed_tensor_id,
            quantize_activation=True,
            fp8_wgrad=True,
            fp8_weight_cache=weight_cache,
            fp8_weight_scale=weight_scale,
            activation_tensor_id=packed_activation_tensor_id,
        )

    # Qwen feeds RMSNorm output into gate/up projections, so an order-one input
    # is representative.  A 0.05 standard deviation makes the SwiGLU local
    # gradients cancellation-dominated and understates MXFP4 dW SNR.
    x = torch.empty((M, H), device=device, dtype=torch.bfloat16).normal_(std=1.0)
    x.requires_grad_(True)
    dy = torch.empty((M, INTERMEDIATE), device=device, dtype=torch.bfloat16).normal_(
        std=0.01
    )
    torch.cuda.reset_peak_memory_stats(device)

    def current_forward() -> torch.Tensor:
        gate = bank.gate(x)
        up = bank.up(x)
        if args.current_swiglu == "aiter":
            return _AiterSplitSwiGLU.apply(gate, up)
        return F.silu(gate) * up

    def packed_forward() -> torch.Tensor:
        packed = packed_projection(x)
        if args.packed_swiglu == "aiter":
            # Exercise the production Lumen autograd wrapper rather than a
            # benchmark-local copy of the AITER calling convention.
            from lumen.ops.fused_swiglu import packed_swiglu

            return packed_swiglu(packed)
        gate, up = packed.split(INTERMEDIATE, dim=-1)
        return F.silu(gate) * up

    def current_full() -> tuple[torch.Tensor, ...]:
        output = current_forward()
        return torch.autograd.grad(
            output,
            (x, bank.gate.weight, bank.up.weight),
            grad_outputs=dy,
        )

    def packed_full() -> tuple[torch.Tensor, ...]:
        output = packed_forward()
        if args.packed_weight_source == "parameter":
            dinput, packed_dweight = torch.autograd.grad(
                output,
                (x, bank.packed.weight),
                grad_outputs=dy,
            )
            dweight_gate, dweight_up = packed_dweight.split(INTERMEDIATE, dim=0)
            return dinput, dweight_gate, dweight_up
        return torch.autograd.grad(
            output,
            (x, bank.gate.weight, bank.up.weight),
            grad_outputs=dy,
        )

    # Each shape's first invocation includes production autotuning, JIT work,
    # first-use dispatcher checks, and an MXFP4 weight-cache miss.  These two
    # one-shot observations are provenance, not an order-neutral speedup claim.
    cold_current = cuda_timer(
        current_full,
        warmup=0,
        iters=1,
        trim_pct=0.0,
        label="cold_first_full_current",
    )
    cold_packed = cuda_timer(
        packed_full,
        warmup=0,
        iters=1,
        trim_pct=0.0,
        label="cold_first_full_packed",
    )

    first_backend_report = _runtime_dispatch_report(
        api["dispatch"], api["mxfp4_autotune"]
    )
    missing_backends = [
        phase
        for phase, record in first_backend_report.items()
        if record["selected_backend"] is None
    ]
    if missing_backends:
        raise RuntimeError(
            "fresh production calls did not cache a backend for: "
            + ", ".join(missing_backends)
        )

    # The first cache is necessarily constructed before its consumer's backend
    # has been selected.  Rebuild it once so warm measurements use the layout
    # production uses on subsequent optimizer steps.
    cache_owners = [bank.gate, bank.up]
    cache_owners.append(
        bank.packed if args.packed_weight_source == "parameter" else packed_cache_owner
    )
    for layer in cache_owners:
        _clear_mxfp4_weight_cache(layer)
    _run_arms(args.arm_order, current_full, packed_full)
    torch.cuda.synchronize()
    for name, layer in (
        ("gate", bank.gate),
        ("up", bank.up),
        ("packed", cache_owners[-1]),
    ):
        if not hasattr(layer, "_mxfp4_w_cache"):
            raise RuntimeError(
                f"{name} has no warm MXFP4 weight cache; timing would include "
                "weight quantization on every iteration"
            )
        if args.packed_weight_source == "pair" and name == "packed":
            expected_version = (bank.gate.weight._version, bank.up.weight._version)
        else:
            expected_version = (
                layer.weight._version
                if hasattr(layer, "weight")
                else 0  # torch.cat creates a fresh logical packed weight at version 0.
            )
        if getattr(layer, "_mxfp4_w_cache_version", None) != expected_version:
            raise RuntimeError(f"{name} MXFP4 weight-cache version is stale")
        if args.packed_weight_source == "pair" and name == "packed":
            expected_sources = (id(bank.gate.weight), id(bank.up.weight))
            if getattr(layer, "_mxfp4_w_cache_sources", None) != expected_sources:
                raise RuntimeError("packed MXFP4 pair cache source identities are stale")

    paired_full = _measure_interleaved_full(
        current=current_full,
        packed=packed_full,
        initial_order=args.arm_order,
        warmup=args.warmup,
        iters=args.paired_iters,
        disable_python_gc=args.paired_disable_python_gc,
    )
    batched_full = _measure_batched_full(
        current=current_full,
        packed=packed_full,
        initial_order=args.arm_order,
        repeats=args.batch_repeats,
        pairs=args.batch_pairs,
        disable_python_gc=args.paired_disable_python_gc,
    )
    gc.collect()
    torch.cuda.empty_cache()

    timings = [cold_current, cold_packed]
    warm_forward_current, warm_forward_packed = _measure_arms(
        args.arm_order,
        lambda: cuda_timer(
            current_forward,
            warmup=args.warmup,
            iters=args.iters,
            trim_pct=args.trim_pct,
            label="warm_forward_current",
        ),
        lambda: cuda_timer(
            packed_forward,
            warmup=args.warmup,
            iters=args.iters,
            trim_pct=args.trim_pct,
            label="warm_forward_packed",
        ),
    )
    timings.extend((warm_forward_current, warm_forward_packed))

    def measure_backward_current() -> BenchResult:
        result = _time_backward_only(
            label="warm_backward_current",
            forward=current_forward,
            parameters=(x, bank.gate.weight, bank.up.weight),
            dy=dy,
            warmup=args.warmup,
            iters=args.iters,
            trim_pct=args.trim_pct,
            split_packed_weight_grad=False,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def measure_backward_packed() -> BenchResult:
        packed_parameters = (
            (x, bank.packed.weight)
            if args.packed_weight_source == "parameter"
            else (x, bank.gate.weight, bank.up.weight)
        )
        result = _time_backward_only(
            label="warm_backward_packed",
            forward=packed_forward,
            parameters=packed_parameters,
            dy=dy,
            warmup=args.warmup,
            iters=args.iters,
            trim_pct=args.trim_pct,
            split_packed_weight_grad=args.packed_weight_source == "parameter",
        )
        gc.collect()
        torch.cuda.empty_cache()
        return result

    warm_backward_current, warm_backward_packed = _measure_arms(
        args.arm_order,
        measure_backward_current,
        measure_backward_packed,
    )
    timings.extend((warm_backward_current, warm_backward_packed))

    warm_full_current, warm_full_packed = _measure_arms(
        args.arm_order,
        lambda: cuda_timer(
            current_full,
            warmup=args.warmup,
            iters=args.iters,
            trim_pct=args.trim_pct,
            label="warm_full_current",
        ),
        lambda: cuda_timer(
            packed_full,
            warmup=args.warmup,
            iters=args.iters,
            trim_pct=args.trim_pct,
            label="warm_full_packed",
        ),
    )
    timings.extend((warm_full_current, warm_full_packed))

    correctness = (
        {"skipped": True}
        if args.skip_correctness
        else _correctness_report(
            current_forward,
            packed_forward,
            x,
            dy,
            bank,
            args.seed,
            args.packed_weight_source,
        )
    )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    def measure_peak_allocated(fn: Callable[[], Any]) -> dict[str, int]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        result = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)
        del result
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "baseline_bytes": before,
            "peak_bytes": peak,
            "increment_bytes": peak - before,
        }

    peak_memory_by_arm = {
        "current": measure_peak_allocated(current_full),
        "packed": measure_peak_allocated(packed_full),
    }

    fp8_dtype = bank.gate._lumen_fp8_dtype

    def build_cache(owner: Any, weight: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return quant._mxfp4_cached_weight(
            owner,
            weight,
            None,
            None,
            "mxfp4",
            fp8_dtype,
            MXFP4_BLOCK_SIZE,
            gemm_rows=M,
        )

    def current_cache_miss() -> tuple[Any, Any]:
        _clear_mxfp4_weight_cache(bank.gate)
        _clear_mxfp4_weight_cache(bank.up)
        return (
            build_cache(bank.gate, bank.gate.weight),
            build_cache(bank.up, bank.up.weight),
        )

    def prepacked_cache_miss() -> tuple[torch.Tensor, ...]:
        _clear_mxfp4_weight_cache(bank.packed)
        return build_cache(bank.packed, bank.packed.weight)

    packed_cat_owner = nn.Module()

    def bf16_cat_only() -> torch.Tensor:
        return torch.cat((bank.gate.weight.detach(), bank.up.weight.detach()), dim=0)

    def packed_cache_miss_with_cat() -> tuple[torch.Tensor, ...]:
        _clear_mxfp4_weight_cache(packed_cat_owner)
        packed_weight = bf16_cat_only()
        return build_cache(packed_cat_owner, packed_weight)

    pair_cache_owner = nn.Module()

    def packed_cache_miss_with_pair() -> tuple[torch.Tensor, ...]:
        _clear_mxfp4_weight_cache(pair_cache_owner)
        return quant._mxfp4_cached_weight_pair(
            pair_cache_owner,
            bank.gate.weight,
            bank.up.weight,
            fp8_dtype,
            MXFP4_BLOCK_SIZE,
            gemm_rows=M,
        )

    cache_current = cuda_timer(
        current_cache_miss,
        warmup=args.warmup,
        iters=args.iters,
        trim_pct=args.trim_pct,
        label="cache_miss_current_two_weights",
    )
    cache_prepacked = cuda_timer(
        prepacked_cache_miss,
        warmup=args.warmup,
        iters=args.iters,
        trim_pct=args.trim_pct,
        label="cache_miss_packed_prepacked_weight",
    )
    cat_only = cuda_timer(
        bf16_cat_only,
        warmup=args.warmup,
        iters=args.iters,
        trim_pct=args.trim_pct,
        label="bf16_cat_gate_up",
    )
    cache_with_cat = cuda_timer(
        packed_cache_miss_with_cat,
        warmup=args.warmup,
        iters=args.iters,
        trim_pct=args.trim_pct,
        label="cache_miss_packed_with_bf16_cat",
    )
    cache_with_pair = cuda_timer(
        packed_cache_miss_with_pair,
        warmup=args.warmup,
        iters=args.iters,
        trim_pct=args.trim_pct,
        label="cache_miss_packed_two_source_pair",
    )
    timings.extend(
        (cache_current, cache_prepacked, cat_only, cache_with_cat, cache_with_pair)
    )

    trace_paths: dict[str, str] = {}
    if args.trace:
        # Restore module-owned warm caches after the destructive miss timings.
        for layer in cache_owners:
            _clear_mxfp4_weight_cache(layer)
        current_full()
        packed_full()
        torch.cuda.synchronize()
        trace_paths["current_full"] = trace_fn(
            current_full,
            str(output_dir / "current_full_trace.json"),
            warmup=1,
            active=1,
            label=f"current: two MXFP4 linears + {args.current_swiglu} SwiGLU",
        )
        trace_paths["packed_full"] = trace_fn(
            packed_full,
            str(output_dir / "packed_full_trace.json"),
            warmup=1,
            active=1,
            label=f"packed: one wide MXFP4 linear + {args.packed_swiglu} SwiGLU",
        )

    backend_report = _runtime_dispatch_report(api["dispatch"], api["mxfp4_autotune"])
    if args.dispatch_mode == "fallback-chain":
        missing_runtime_evidence = [
            phase
            for phase, record in backend_report.items()
            if not record["runtime_observation_available"]
        ]
        if missing_runtime_evidence:
            raise RuntimeError(
                "fallback-chain mode produced no runtime backend evidence for: "
                + ", ".join(missing_runtime_evidence)
            )

    git_archives_post = {
        "lumen": _archive_git_state(repo_root, "lumen_post", output_dir),
    }
    if aiter_root is not None:
        git_archives_post["aiter"] = _archive_git_state(
            aiter_root, "aiter_post", output_dir
        )

    cold_first_calls = {
        "comparable": False,
        "reason": (
            "ordered first calls include different shape-specific autotuning/JIT "
            "work and are reported only as cold-start provenance"
        ),
        "order": ["current", "packed"],
        "current": _bench_result_payload(cold_current),
        "packed": _bench_result_payload(cold_packed),
    }
    comparisons = {
        "warm_forward": _comparison_payload(warm_forward_current, warm_forward_packed),
        "warm_backward": _comparison_payload(
            warm_backward_current, warm_backward_packed
        ),
        "warm_full": _comparison_payload(warm_full_current, warm_full_packed),
        "cache_miss_prepacked": _comparison_payload(cache_current, cache_prepacked),
        "cache_miss_including_bf16_cat": _comparison_payload(
            cache_current, cache_with_cat
        ),
        "cache_miss_actual_two_source_pair": _comparison_payload(
            cache_current, cache_with_pair
        ),
    }

    aiter = api["aiter"]
    device_properties = torch.cuda.get_device_properties(device)
    metadata = {
        "fresh_run": True,
        "started_utc": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "finished_utc": datetime.now(tz=timezone.utc).isoformat(),
        "duration_seconds": time.time() - started,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "device_index": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "device_arch": getattr(device_properties, "gcnArchName", "unknown"),
        "device_total_memory_bytes": device_properties.total_memory,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "lumen_git_head": _command_output(["git", "rev-parse", "HEAD"], repo_root),
        "lumen_git_status_short": _command_output(
            ["git", "status", "--short"], repo_root
        ).splitlines(),
        "git_archives": git_archives,
        "git_archives_post": git_archives_post,
        "benchmark_source_snapshot": str(
            output_dir / "bench_mxfp4_gate_up.executed.py"
        ),
        "benchmark_source_sha256": _sha256_path(
            output_dir / "bench_mxfp4_gate_up.executed.py"
        ),
        "lumen_quantize_origin": str(getattr(quant, "__file__", "unknown")),
        "aiter_origin": str(getattr(aiter, "__file__", "unknown")),
        "loaded_aiter_binary_sha256": _loaded_aiter_binary_hashes(),
        "aiter_version_attribute": getattr(aiter, "__version__", None),
        "aiter_distribution_version": _package_version("aiter"),
        "autotune_enabled": api["mxfp4_autotune"].AUTOTUNE_ENABLED,
        "fast_quant_dispatch": api["linear"]._FAST_QUANT_DISPATCH,
        "dispatch_mode": args.dispatch_mode,
        "autotune_cache": str(autotune_cache),
        "autotune_cache_written_at_process_exit": True,
        "shape_log": str(shape_log),
        "aiter_tuned_config": os.environ.get("AITER_CONFIG_GEMM_A4W4", ""),
        "flydsl_enabled": os.environ.get("LUMEN_MXFP4_FLYDSL", "0") == "1",
        "weight_cache_disabled": (
            os.environ.get("LUMEN_MXFP4_DISABLE_WEIGHT_CACHE", "0") == "1"
        ),
        "dgrad_hadamard_enabled": (
            os.environ.get("LUMEN_MXFP4_DGRAD_HADAMARD", "0") == "1"
        ),
        "skip_backend_sync": os.environ.get("LUMEN_SKIP_BACKEND_SYNC", "0") == "1",
        "quant_manager_type": type(manager).__qualname__,
        "fp8_dtype": str(fp8_dtype),
        "api_path": (
            "nn.Linear patched by lumen.quantize.enable -> production "
            "quantized_linear autograd/dispatch"
        ),
        "cold_first_call_order": ["current", "packed"],
        "warm_arm_order": args.arm_order,
        "current_swiglu": args.current_swiglu,
        "packed_swiglu": args.packed_swiglu,
        "packed_weight_source": args.packed_weight_source,
        "scope_note": (
            "The packed arm uses "
            f"{args.packed_weight_source} as its logical wide-weight source. "
            "The pair mode preserves distinct gate/up Parameters and uses the "
            "production two-source compact cache; cat includes a transient BF16 "
            "concatenation, while parameter is a one-wide feasibility oracle. "
            "This is not an end-to-end training-step result."
        ),
    }
    arguments = vars(args).copy()
    arguments["output_dir"] = str(output_dir)
    payload = {
        "metadata": metadata,
        "arguments": arguments,
        "comparison_definition": {
            "a_current": (
                "two patched MXFP4 nn.Linear calls + "
                f"{args.current_swiglu} SwiGLU"
            ),
            "b_packed": (
                "one logical wide MXFP4 linear from "
                f"{args.packed_weight_source} + split views + "
                f"{args.packed_swiglu} SwiGLU"
            ),
        },
        **_shape_payload(),
        "timings": {result.name: _bench_result_payload(result) for result in timings},
        "cold_first_calls": cold_first_calls,
        "comparisons": comparisons,
        "paired_full": paired_full,
        "batched_full": batched_full,
        "backend_after_first_calls": first_backend_report,
        "backend_final": backend_report,
        "correctness": correctness,
        "peak_memory_by_arm": peak_memory_by_arm,
        "traces": trace_paths,
    }

    serializable = _jsonable(payload)
    result_path = output_dir / "gate_up_mxfp4_results.json"
    result_path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print_report_with_table(
        "MXFP4 gate/up packing, M=16384 H=4096 I=12288",
        timings,
    )
    print(
        "COLD comparable=false order=current,packed "
        f"current_ms={cold_current.avg_ms:.6f} "
        f"packed_ms={cold_packed.avg_ms:.6f}"
    )
    for name, comparison in comparisons.items():
        print(
            "RESULT "
            f"name={name} "
            f"a_ms={comparison['a_current_ms']:.6f} "
            f"b_ms={comparison['b_packed_ms']:.6f} "
            f"speedup={comparison['speedup_a_over_b']:.6f} "
            f"saving_ms={comparison['saving_ms']:.6f}"
        )
    print(
        "PAIRED_FULL "
        f"pairs={paired_full['pairs']} "
        f"current_mean_ms={paired_full['current_mean_ms']:.6f} "
        f"packed_mean_ms={paired_full['packed_mean_ms']:.6f} "
        f"ratio_of_means={paired_full['ratio_of_means']:.6f} "
        f"median_speedup={paired_full['median_paired_speedup']:.6f} "
        f"median_saving_ms={paired_full['median_paired_saving_ms']:.6f} "
        f"packed_wins={paired_full['packed_wins']}/{paired_full['pairs']} "
        f"current_wins={paired_full['current_wins']}/{paired_full['pairs']}"
    )
    print(
        "BATCHED_FULL "
        f"pairs={batched_full['pairs']} "
        f"repeats={batched_full['repeats_per_arm']} "
        f"current_mean_ms_per_call={batched_full['current_mean_ms_per_call']:.6f} "
        f"packed_mean_ms_per_call={batched_full['packed_mean_ms_per_call']:.6f} "
        f"ratio_of_means={batched_full['ratio_of_means']:.6f} "
        f"median_speedup={batched_full['median_paired_speedup']:.6f} "
        f"median_saving_ms_per_call={batched_full['median_paired_saving_ms_per_call']:.6f} "
        f"packed_wins={batched_full['packed_wins']}/{batched_full['pairs']} "
        f"current_wins={batched_full['current_wins']}/{batched_full['pairs']}"
    )
    for phase, record in backend_report.items():
        runtime = ",".join(record["runtime_backends_observed"]) or "unobserved"
        print(
            "BACKEND "
            f"phase={phase} "
            f"shape={tuple(record['shape_mnk'])} "
            f"selected={record['selected_backend']} "
            f"runtime={runtime} "
            f"fallback={record['runtime_fallback_observed']} "
            f"dequant_bf16={record['dequant_bf16_fallback_observed']}"
        )
    print(f"RESULT_JSON path={result_path}")
    print(
        f"SCOPE packed_weight_source={args.packed_weight_source}; "
        "it is not end-to-end training-step evidence"
    )
    return serializable


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.dry_run:
        _dry_run(args)
        return
    try:
        _run(args)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
