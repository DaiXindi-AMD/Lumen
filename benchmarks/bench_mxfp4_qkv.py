# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact-shape MXFP4 packed-QKV feasibility benchmark for Qwen3-8B.

The control uses three production Lumen-patched ``nn.Linear`` projections.
The candidate keeps the same three Parameters but executes one packed MXFP4
forward, dgrad, and wgrad.  The benchmark includes split-gradient assembly,
weight-cache construction, multi-seed BF16 accuracy, peak memory, and paired
full forward/backward timing.  It is a feasibility benchmark, not an E2E step
speed claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.bench_utils import BenchResult, cuda_timer, trace_fn

M = 16_384
H = 4_096
Q = 4_096
K = 1_024
V = 1_024
PACKED = Q + K + V
BLOCK = 32
QUANTIZED_LAYERS = 31
GRAD_ACCUMULATION = 8

LOGICAL_GEMM_SHAPES = {
    "control_forward_q": (M, Q, H),
    "control_forward_kv": (M, K, H),
    "control_dgrad_q": (M, H, Q),
    "control_dgrad_kv": (M, H, K),
    "control_wgrad_q": (Q, H, M),
    "control_wgrad_kv": (K, H, M),
    "packed_forward": (M, PACKED, H),
    "packed_dgrad": (M, H, PACKED),
    "packed_wgrad": (PACKED, H, M),
}


class QKVBank(nn.Module):
    """Own the original Q/K/V Parameters used by both benchmark arms."""

    def __init__(self, device: torch.device):
        super().__init__()
        kwargs = {"bias": False, "device": device, "dtype": torch.bfloat16}
        self.q = nn.Linear(H, Q, **kwargs)
        self.k = nn.Linear(H, K, **kwargs)
        self.v = nn.Linear(H, V, **kwargs)
        with torch.no_grad():
            for projection in (self.q, self.k, self.v):
                projection.weight.normal_(mean=0.0, std=0.02)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=_nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=20_260_921)
    parser.add_argument("--warmup", type=_nonnegative_int, default=10)
    parser.add_argument("--iters", type=_positive_int, default=30)
    parser.add_argument("--paired-iters", type=_positive_int, default=64)
    parser.add_argument("--batch-repeats", type=_positive_int, default=16)
    parser.add_argument("--batch-pairs", type=_positive_int, default=12)
    parser.add_argument("--update-pairs", type=_positive_int, default=12)
    parser.add_argument("--correctness-seeds", type=_positive_int, default=3)
    parser.add_argument("--arm-order", choices=("ab", "ba"), default="ab")
    parser.add_argument(
        "--dispatch-mode", choices=("fallback-chain", "fast"), default="fast"
    )
    parser.add_argument(
        "--tuned-config",
        action="append",
        type=Path,
        default=None,
        help=(
            "A4W4 tuned CSV, in priority order. By default the Qwen3 model, "
            "Lumen generic, and installed AITER generic tables are used."
        ),
    )
    parser.add_argument("--trace", action="store_true")
    return parser


def _clear_weight_cache(owner: Any) -> None:
    for name in (
        "_mxfp4_w_cache",
        "_mxfp4_w_cache_version",
        "_mxfp4_w_cache_sources",
        "_mxfp4_w_cache_metadata",
    ):
        if hasattr(owner, name):
            delattr(owner, name)


def _tensor_error(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    if reference.shape != actual.shape:
        raise RuntimeError(f"shape mismatch: {reference.shape} != {actual.shape}")
    ref = reference.detach().float()
    out = actual.detach().float()
    diff = out - ref
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    error_norm = float(torch.linalg.vector_norm(diff).item())
    max_abs = float(diff.abs().max().item())
    finite = bool(torch.isfinite(out).all().item())
    del ref, out, diff
    if error_norm == 0.0:
        snr = float("inf")
    elif ref_norm == 0.0:
        snr = float("-inf")
    else:
        snr = 20.0 * math.log10(ref_norm / error_norm)
    return {
        "snr_db": snr,
        "relative_l2": error_norm / ref_norm if ref_norm else float("inf"),
        "max_abs": max_abs,
        "finite": finite and math.isfinite(error_norm) and math.isfinite(max_abs),
    }


def _result(result: BenchResult) -> dict[str, Any]:
    return dataclasses.asdict(result)


def _paired_measure(
    current: Callable[[], Any],
    packed: Callable[[], Any],
    *,
    initial_order: str,
    warmup: int,
    iters: int,
    repeats: int = 1,
) -> dict[str, Any]:
    base = ("ab", "ba", "ba", "ab")

    def order(index: int) -> str:
        value = base[index % len(base)]
        return (
            value
            if initial_order == "ab"
            else value.translate(str.maketrans("ab", "ba"))
        )

    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    current_ms: list[float] = []
    packed_ms: list[float] = []
    orders: list[str] = []
    try:
        for index in range(warmup):
            for arm in order(index):
                fn = current if arm == "a" else packed
                for _ in range(repeats):
                    fn()
        torch.cuda.synchronize()

        for index in range(iters):
            sample_order = order(index)
            orders.append(sample_order)
            for arm in sample_order:
                fn = current if arm == "a" else packed
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):
                    fn()
                end.record()
                end.synchronize()
                value = float(start.elapsed_time(end)) / repeats
                (current_ms if arm == "a" else packed_ms).append(value)
    finally:
        if gc_was_enabled:
            gc.enable()

    savings = [left - right for left, right in zip(current_ms, packed_ms)]
    ratios = [left / right for left, right in zip(current_ms, packed_ms)]
    return {
        "orders": orders,
        "repeats": repeats,
        "current_ms": current_ms,
        "packed_ms": packed_ms,
        "saving_ms": savings,
        "speedup": ratios,
        "current_mean_ms": statistics.fmean(current_ms),
        "packed_mean_ms": statistics.fmean(packed_ms),
        "ratio_of_means": sum(current_ms) / sum(packed_ms),
        "current_median_ms": statistics.median(current_ms),
        "packed_median_ms": statistics.median(packed_ms),
        "median_saving_ms": statistics.median(savings),
        "median_speedup": statistics.median(ratios),
        "packed_wins": sum(value > 0 for value in savings),
        "pairs": len(savings),
        "python_gc_disabled": True,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_backend_report(
    dispatch_module: Any, autotune_module: Any
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    cache = getattr(dispatch_module, "_backend_cache", {})
    for label, shape in LOGICAL_GEMM_SHAPES.items():
        prefix = f"gemm_mxfp4:{shape[0]}x{shape[1]}x{shape[2]}:"
        labels = set()
        for key, value in cache.items():
            if (
                isinstance(key, str)
                and key.startswith(prefix)
                and not key.endswith((":hits", ":warned"))
            ):
                if value is not None:
                    labels.add(str(value))
        report[label] = {
            "shape_mnk": list(shape),
            "selected_backend": autotune_module.cached(shape),
            "autotune_profile": autotune_module.cached_profile(shape),
            "runtime_backends": sorted(labels),
        }
    return report


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse {args.output_dir}")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("this benchmark requires WORLD_SIZE=1")
    if "lumen.ops.quantize.mxfp4_autotune" in sys.modules:
        raise RuntimeError("Lumen was imported before fresh cache paths were set")

    args.output_dir.mkdir(parents=True)
    shutil.copy2(Path(__file__), args.output_dir / "bench_mxfp4_qkv.executed.py")
    autotune_path = args.output_dir / "mxfp4_autotune.json"
    shape_log = args.output_dir / "mxfp4_shapes.csv"
    os.environ["LUMEN_MXFP4_AUTOTUNE"] = "1"
    os.environ["LUMEN_MXFP4_AUTOTUNE_CACHE"] = str(autotune_path)
    os.environ["LUMEN_MXFP4_GEMM_SHAPE_LOG"] = str(shape_log)
    os.environ["LUMEN_FAST_QUANT_DISPATCH"] = (
        "0" if args.dispatch_mode == "fallback-chain" else "1"
    )
    for name in ("LUMEN_BENCH_WARMUP", "LUMEN_BENCH_ITERS", "LUMEN_BENCH_TRIM_PCT"):
        os.environ.pop(name, None)

    import aiter

    repo_root = Path(__file__).resolve().parents[1]
    if args.tuned_config is None:
        tuned_configs = [
            repo_root
            / "examples/qwen3/configs/qwen3_8b_a4w4_blockscale_tuned_gemm.csv",
            repo_root / "examples/qwen3/configs/a4w4_blockscale_tuned_gemm.csv",
            Path(aiter.__file__).resolve().parent
            / "configs/a4w4_blockscale_tuned_gemm.csv",
        ]
    else:
        tuned_configs = [path.expanduser().resolve() for path in args.tuned_config]
    missing_configs = [path for path in tuned_configs if not path.is_file()]
    if missing_configs:
        raise FileNotFoundError(f"missing tuned config(s): {missing_configs}")
    os.environ["AITER_CONFIG_GEMM_A4W4"] = os.pathsep.join(
        str(path) for path in tuned_configs
    )

    import lumen
    import lumen.quantize as quant
    from lumen.ops import dispatch
    from lumen.ops.quantize import mxfp4_qkv_linear
    from lumen.ops.quantize import mxfp4_autotune

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    quant.assert_mxfp4_arch_supported()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    bank = QKVBank(device)
    manager = quant.enable(bank, format="mxfp4", scaling="blockwise", block_size=BLOCK)
    for name, projection in (("q", bank.q), ("k", bank.k), ("v", bank.v)):
        if not getattr(projection, "_quant_enabled", False):
            raise RuntimeError(f"quant.enable did not patch {name}")

    packed_owner = nn.Module()
    fp8_dtype = bank.q._lumen_fp8_dtype

    def packed_cache(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return quant._mxfp4_cached_weight_qkv(
            packed_owner,
            bank.q.weight,
            bank.k.weight,
            bank.v.weight,
            fp8_dtype,
            BLOCK,
            gemm_rows=input_tensor.numel() // input_tensor.shape[-1],
        )

    x = torch.empty((M, H), device=device, dtype=torch.bfloat16).normal_(std=1.0)
    x.requires_grad_(True)
    grad_q = torch.empty((M, Q), device=device, dtype=torch.bfloat16).normal_(std=0.01)
    grad_k = torch.empty((M, K), device=device, dtype=torch.bfloat16).normal_(std=0.01)
    grad_v = torch.empty((M, V), device=device, dtype=torch.bfloat16).normal_(std=0.01)
    parameters = (x, bank.q.weight, bank.k.weight, bank.v.weight)

    def current_forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return bank.q(x), bank.k(x), bank.v(x)

    def candidate_forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, scale = packed_cache(x)
        packed = mxfp4_qkv_linear(
            x,
            bank.q.weight,
            bank.k.weight,
            bank.v.weight,
            data,
            scale,
            scaling_manager=manager,
            fp8_dtype=fp8_dtype,
            block_size=BLOCK,
            tensor_id="qkv_packed.weight",
        )
        # Match the three independent nn.Linear output contract. Plain split
        # views retain a 6144-element row stride and can make Q/K norm or the
        # attention backend materialize hidden copies downstream.
        return tuple(output.contiguous() for output in packed.split((Q, K, V), dim=-1))

    grad_outputs = (grad_q, grad_k, grad_v)

    def current_full() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            current_forward(), parameters, grad_outputs=grad_outputs
        )

    def candidate_full() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            candidate_forward(), parameters, grad_outputs=grad_outputs
        )

    def current_update() -> tuple[torch.Tensor, ...]:
        for owner in (bank.q, bank.k, bank.v):
            _clear_weight_cache(owner)
        result: tuple[torch.Tensor, ...] = ()
        for _ in range(GRAD_ACCUMULATION):
            result = current_full()
        return result

    def candidate_update() -> tuple[torch.Tensor, ...]:
        _clear_weight_cache(packed_owner)
        result: tuple[torch.Tensor, ...] = ()
        for _ in range(GRAD_ACCUMULATION):
            result = candidate_full()
        return result

    started = time.time()
    cold_current = cuda_timer(current_full, warmup=0, iters=1, label="cold_current")
    cold_candidate = cuda_timer(candidate_full, warmup=0, iters=1, label="cold_packed")

    # First use selects backends before all cache layouts are known. Rebuild once.
    for owner in (bank.q, bank.k, bank.v, packed_owner):
        _clear_weight_cache(owner)
    current_full()
    candidate_full()
    torch.cuda.synchronize()
    for name, owner in (
        ("q", bank.q),
        ("k", bank.k),
        ("v", bank.v),
        ("packed", packed_owner),
    ):
        if not hasattr(owner, "_mxfp4_w_cache"):
            raise RuntimeError(f"{name} MXFP4 weight cache was not populated")

    paired = _paired_measure(
        current_full,
        candidate_full,
        initial_order=args.arm_order,
        warmup=args.warmup,
        iters=args.paired_iters,
    )
    batched = _paired_measure(
        current_full,
        candidate_full,
        initial_order=args.arm_order,
        warmup=1,
        iters=args.batch_pairs,
        repeats=args.batch_repeats,
    )
    update_paired = _paired_measure(
        current_update,
        candidate_update,
        initial_order=args.arm_order,
        warmup=1,
        iters=args.update_pairs,
    )
    forward_current = cuda_timer(
        current_forward,
        warmup=args.warmup,
        iters=args.iters,
        trim_pct=10.0,
        label="forward_current",
    )
    forward_candidate = cuda_timer(
        candidate_forward,
        warmup=args.warmup,
        iters=args.iters,
        trim_pct=10.0,
        label="forward_packed",
    )

    def backward_only(
        forward: Callable[[], tuple[torch.Tensor, ...]], label: str
    ) -> BenchResult:
        outputs = forward()

        def invoke():
            return torch.autograd.grad(
                outputs, parameters, grad_outputs=grad_outputs, retain_graph=True
            )

        return cuda_timer(
            invoke, warmup=args.warmup, iters=args.iters, trim_pct=10.0, label=label
        )

    backward_current = backward_only(current_forward, "backward_current")
    backward_candidate = backward_only(candidate_forward, "backward_packed")

    def bf16_forward():
        return (
            F.linear(x, bank.q.weight),
            F.linear(x, bank.k.weight),
            F.linear(x, bank.v.weight),
        )

    correctness: list[dict[str, Any]] = []
    reference_outputs = bf16_forward()
    reference_grads = torch.autograd.grad(
        reference_outputs, parameters, grad_outputs=grad_outputs
    )
    # The exact production QKV WGrad is reduction-heavy and measures about
    # 5.7 dB versus BF16, so a small-shape 10 dB gate would reject the control
    # itself. Keep an absolute floor, then require the packed path not to lose
    # more than 0.5 dB relative to the same-seed production control.
    thresholds = {"output": 12.0, "dinput": 10.0, "dweight": 5.0}
    max_regression_db = 0.5
    for offset in range(args.correctness_seeds):
        seed = args.seed + 100 + offset
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        control_outputs = current_forward()
        control_grads = torch.autograd.grad(
            control_outputs, parameters, grad_outputs=grad_outputs
        )
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        packed_outputs = candidate_forward()
        packed_grads = torch.autograd.grad(
            packed_outputs, parameters, grad_outputs=grad_outputs
        )
        record: dict[str, Any] = {"seed": seed, "control": {}, "packed": {}}
        for arm, outputs, gradients in (
            ("control", control_outputs, control_grads),
            ("packed", packed_outputs, packed_grads),
        ):
            record[arm]["output_q"] = _tensor_error(reference_outputs[0], outputs[0])
            record[arm]["output_k"] = _tensor_error(reference_outputs[1], outputs[1])
            record[arm]["output_v"] = _tensor_error(reference_outputs[2], outputs[2])
            record[arm]["dinput"] = _tensor_error(reference_grads[0], gradients[0])
            record[arm]["dweight_q"] = _tensor_error(reference_grads[1], gradients[1])
            record[arm]["dweight_k"] = _tensor_error(reference_grads[2], gradients[2])
            record[arm]["dweight_v"] = _tensor_error(reference_grads[3], gradients[3])
            record[arm]["weight_grad_contiguous"] = [
                bool(gradient.is_contiguous()) for gradient in gradients[1:]
            ]
            metrics = record[arm]
            metric_names = (
                "output_q",
                "output_k",
                "output_v",
                "dinput",
                "dweight_q",
                "dweight_k",
                "dweight_v",
            )
            if any(not metrics[name]["finite"] for name in metric_names):
                raise RuntimeError(f"{arm} produced non-finite values at seed {seed}")
            if any(
                metrics[name]["snr_db"] < thresholds["output"]
                for name in ("output_q", "output_k", "output_v")
            ):
                raise RuntimeError(f"{arm} output SNR failed at seed {seed}: {metrics}")
            if metrics["dinput"]["snr_db"] < thresholds["dinput"]:
                raise RuntimeError(f"{arm} dinput SNR failed at seed {seed}: {metrics}")
            if any(
                metrics[name]["snr_db"] < thresholds["dweight"]
                for name in ("dweight_q", "dweight_k", "dweight_v")
            ):
                raise RuntimeError(
                    f"{arm} dweight SNR failed at seed {seed}: {metrics}"
                )
        for metric_name in (
            "output_q",
            "output_k",
            "output_v",
            "dinput",
            "dweight_q",
            "dweight_k",
            "dweight_v",
        ):
            control_snr = record["control"][metric_name]["snr_db"]
            packed_snr = record["packed"][metric_name]["snr_db"]
            if packed_snr + max_regression_db < control_snr:
                raise RuntimeError(
                    "packed QKV regressed versus the production control at "
                    f"seed {seed} for {metric_name}: packed={packed_snr:.4f} dB, "
                    f"control={control_snr:.4f} dB"
                )
        correctness.append(record)

    # One explicit non-contiguous downstream-gradient check.
    noncontiguous_grads = tuple(
        torch.randn((rows, M), device=device, dtype=torch.bfloat16).t()
        for rows in (Q, K, V)
    )
    if any(gradient.is_contiguous() for gradient in noncontiguous_grads):
        raise RuntimeError("failed to construct non-contiguous downstream gradients")
    noncontiguous_result = torch.autograd.grad(
        candidate_forward(), parameters, grad_outputs=noncontiguous_grads
    )
    noncontiguous_ok = all(
        bool(torch.isfinite(value).all().item()) for value in noncontiguous_result
    )
    if not noncontiguous_ok:
        raise RuntimeError("packed QKV failed non-contiguous downstream-gradient check")

    # Correctness tensors are much larger than the caches under study. Release
    # them before the arm-isolated peak measurements so they do not dominate
    # the reported baseline or keep both autograd graphs alive.
    del (
        reference_outputs,
        reference_grads,
        control_outputs,
        control_grads,
        packed_outputs,
        packed_grads,
        noncontiguous_grads,
        noncontiguous_result,
    )
    torch.cuda.synchronize()
    gc.collect()

    def clear_all_weight_caches() -> None:
        for owner in (bank.q, bank.k, bank.v, packed_owner):
            _clear_weight_cache(owner)

    def peak(fn: Callable[[], Any]) -> dict[str, int]:
        clear_all_weight_caches()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        values = fn()
        torch.cuda.synchronize()
        maximum = torch.cuda.max_memory_allocated(device)
        del values
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated(device)
        return {
            "baseline_bytes": before,
            "peak_bytes": maximum,
            "increment_bytes": maximum - before,
            "resident_after_bytes": resident,
            "resident_delta_bytes": resident - before,
        }

    peak_memory = {"current": peak(current_full), "packed": peak(candidate_full)}

    def current_cache_miss():
        for owner in (bank.q, bank.k, bank.v):
            _clear_weight_cache(owner)
        return current_forward()

    def packed_cache_miss():
        _clear_weight_cache(packed_owner)
        return candidate_forward()

    cache_current = cuda_timer(
        current_cache_miss,
        warmup=args.warmup,
        iters=args.iters,
        trim_pct=10.0,
        label="cache_miss_current_three_weights",
    )
    cache_packed = cuda_timer(
        packed_cache_miss,
        warmup=args.warmup,
        iters=args.iters,
        trim_pct=10.0,
        label="cache_miss_packed_including_bf16_cat",
    )

    traces: dict[str, str] = {}
    if args.trace:
        for owner in (bank.q, bank.k, bank.v, packed_owner):
            _clear_weight_cache(owner)
        current_full()
        candidate_full()
        torch.cuda.synchronize()
        traces["current"] = trace_fn(
            current_full,
            str(args.output_dir / "current_full_trace.json"),
            warmup=1,
            active=1,
            label="current three MXFP4 Q/K/V projections",
        )
        traces["packed"] = trace_fn(
            candidate_full,
            str(args.output_dir / "packed_full_trace.json"),
            warmup=1,
            active=1,
            label="one packed MXFP4 QKV projection",
        )

    hot_saving = paired["current_mean_ms"] - paired["packed_mean_ms"]
    fixed_order_cache_miss_full_saving = cache_current.avg_ms - cache_packed.avg_ms
    local_update_saving = (
        update_paired["current_mean_ms"] - update_paired["packed_mean_ms"]
    )
    projected_step_saving = QUANTIZED_LAYERS * local_update_saving
    backend_report = _runtime_backend_report(dispatch, mxfp4_autotune)
    for label, record in backend_report.items():
        selected = record["selected_backend"]
        if selected is None or selected == "dequant_bf16":
            raise RuntimeError(f"{label} did not select a native MXFP4 backend")
        runtime_backends = record["runtime_backends"]
        if args.dispatch_mode == "fallback-chain" and (
            not runtime_backends or selected not in runtime_backends
        ):
            raise RuntimeError(
                f"{label} fallback-chain evidence disagrees with selection: "
                f"selected={selected}, observed={runtime_backends}"
            )
    payload = {
        "metadata": {
            "fresh_run": True,
            "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "lumen_origin": lumen.__file__,
            "aiter_origin": aiter.__file__,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(device),
            "dispatch_mode": args.dispatch_mode,
            "tuned_configs": [
                {"path": str(path), "sha256": _sha256(path)} for path in tuned_configs
            ],
            "benchmark_sha256": _sha256(
                args.output_dir / "bench_mxfp4_qkv.executed.py"
            ),
            "implementation": (
                "production mxfp4_qkv_linear with compact-only Q/K/V cache; "
                "Q/K/V split outputs are materialized contiguously so downstream "
                "copy cost is included conservatively"
            ),
            "correctness_seed_scope": (
                "fixed operands with independent stochastic-rounding seeds"
            ),
            "cold_timing_scope": (
                "ordered diagnostic only; candidate reuses imports and probes"
            ),
        },
        "arguments": {**vars(args), "output_dir": str(args.output_dir)},
        "shapes_mnk": {
            name: list(shape) for name, shape in LOGICAL_GEMM_SHAPES.items()
        },
        "timings": {
            result.name: _result(result)
            for result in (
                cold_current,
                cold_candidate,
                forward_current,
                forward_candidate,
                backward_current,
                backward_candidate,
                cache_current,
                cache_packed,
            )
        },
        "paired_full": paired,
        "batched_full": batched,
        "paired_update_ga8": update_paired,
        "correctness": correctness,
        "correctness_thresholds_db": thresholds,
        "max_packed_regression_vs_control_db": max_regression_db,
        "noncontiguous_grad": {"finite": noncontiguous_ok},
        "peak_memory": peak_memory,
        "backend": backend_report,
        "projection": {
            "quantized_layers": QUANTIZED_LAYERS,
            "gradient_accumulation": GRAD_ACCUMULATION,
            "hot_saving_ms_per_layer_microbatch": hot_saving,
            "fixed_order_cache_miss_full_saving_ms": (
                fixed_order_cache_miss_full_saving
            ),
            "measured_saving_ms_per_layer_update": local_update_saving,
            "estimated_step_saving_ms": projected_step_saving,
            "warning": "projection only; attention/FSDP scheduling may change E2E",
        },
        "traces": traces,
    }
    output_path = args.output_dir / "qkv_mxfp4_results.json"
    output_path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
    )
    print(
        "PAIRED_FULL "
        f"current_mean_ms={paired['current_mean_ms']:.6f} "
        f"packed_mean_ms={paired['packed_mean_ms']:.6f} "
        f"speedup={paired['ratio_of_means']:.6f} "
        f"saving_ms={hot_saving:.6f} "
        f"wins={paired['packed_wins']}/{paired['pairs']}"
    )
    print(
        "PROJECTED_STEP "
        f"saving_ms={projected_step_saving:.6f} "
        f"layers={QUANTIZED_LAYERS} ga={GRAD_ACCUMULATION}"
    )
    print(
        "PAIRED_UPDATE_GA8 "
        f"current_mean_ms={update_paired['current_mean_ms']:.6f} "
        f"packed_mean_ms={update_paired['packed_mean_ms']:.6f} "
        f"speedup={update_paired['ratio_of_means']:.6f} "
        f"wins={update_paired['packed_wins']}/{update_paired['pairs']}"
    )
    print(f"RESULT_JSON path={output_path}")
    return payload


def main() -> None:
    args = _parser().parse_args()
    try:
        _run(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
