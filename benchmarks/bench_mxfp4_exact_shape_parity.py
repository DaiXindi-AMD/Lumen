# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check exact-shape MXFP4 GEMM backend parity on shared quantized operands.

The gate/up packing experiment can change WGrad from Triton preshuffled GEMM
to a tuned ASM GEMM.  This diagnostic removes stochastic rounding and all
autograd/SwiGLU effects: every backend consumes the same packed FP4 bytes and
row-major E8M0 scales.  A chunked FP32 dequantized matmul is the independent
reference, so the largest Qwen3-8B WGrad shape does not require a full FP32
copy of both operands at once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import torch

from benchmarks.bench_utils import cuda_timer
from lumen.ops.quantize.linear import (
    _gemm_mxfp4_aiter,
    _gemm_mxfp4_aiter_asm,
    _gemm_mxfp4_aiter_preshuffle,
    _mxfp4_asm_config,
)
from lumen.ops.quantize.ops import convert_from_mxfp4


SHAPES = {
    "wgrad_current": (12_288, 4_096, 16_384),
    "wgrad_packed": (24_576, 4_096, 16_384),
}
BLOCK_SIZE = 32


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20_260_929)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--reference-chunk-rows", type=int, default=256)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True
    )
    if completed.returncode:
        return f"unavailable (exit {completed.returncode}): {completed.stderr.strip()}"
    return completed.stdout.strip()


def _error(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    reference_f32 = reference.float()
    actual_f32 = actual.float()
    difference = actual_f32 - reference_f32
    signal = float(reference_f32.square().sum().item())
    noise = float(difference.square().sum().item())
    max_abs = float(difference.abs().max().item())
    mismatch = int(torch.count_nonzero(actual != reference).item())
    snr_db = float("inf") if noise == 0.0 else 10.0 * math.log10(signal / noise)
    return {
        "bitwise_equal": mismatch == 0,
        "mismatched_elements": mismatch,
        "max_abs": max_abs,
        "snr_db": snr_db,
    }


def _chunked_reference_error(
    *,
    a_fp4: torch.Tensor,
    w_fp4: torch.Tensor,
    scale_a: torch.Tensor,
    scale_w: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    chunk_rows: int,
) -> dict[str, dict[str, Any]]:
    """Compare BF16 backend outputs with FP32 dequantized matmul by row chunk."""
    w_f32 = convert_from_mxfp4(
        w_fp4, scale_w, output_dtype=torch.float32, block_size=BLOCK_SIZE
    )
    accumulators = {
        name: {"signal": 0.0, "noise": 0.0, "max_abs": 0.0, "mismatch": 0}
        for name in outputs
    }
    for start in range(0, a_fp4.shape[0], chunk_rows):
        stop = min(start + chunk_rows, a_fp4.shape[0])
        a_f32 = convert_from_mxfp4(
            a_fp4[start:stop],
            scale_a[start:stop],
            output_dtype=torch.float32,
            block_size=BLOCK_SIZE,
        )
        reference = torch.mm(a_f32, w_f32.t()).to(torch.bfloat16)
        signal = float(reference.float().square().sum().item())
        for name, output in outputs.items():
            actual = output[start:stop]
            difference = actual.float() - reference.float()
            record = accumulators[name]
            record["signal"] += signal
            record["noise"] += float(difference.square().sum().item())
            record["max_abs"] = max(
                record["max_abs"], float(difference.abs().max().item())
            )
            record["mismatch"] += int(torch.count_nonzero(actual != reference).item())
        del a_f32, reference

    report = {}
    for name, record in accumulators.items():
        noise = record["noise"]
        report[name] = {
            "bitwise_equal": record["mismatch"] == 0,
            "mismatched_elements": record["mismatch"],
            "max_abs": record["max_abs"],
            "snr_db": (
                float("inf")
                if noise == 0.0
                else 10.0 * math.log10(record["signal"] / noise)
            ),
        }
    return report


def _time(fn: Callable[[], torch.Tensor], label: str, warmup: int, iters: int) -> dict[str, Any]:
    result = cuda_timer(fn, warmup=warmup, iters=iters, trim_pct=10.0, label=label)
    return {
        "avg_ms": result.avg_ms,
        "min_ms": result.min_ms,
        "max_ms": result.max_ms,
        "median_ms": result.median_ms,
        "p95_ms": result.p95_ms,
    }


def _run_shape(
    name: str,
    shape: tuple[int, int, int],
    seed: int,
    warmup: int,
    iters: int,
    chunk_rows: int,
) -> dict[str, Any]:
    m, n, k = shape
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Random packed values plus bounded E8M0 scales isolate GEMM/layout math
    # from any quantizer or stochastic-rounding implementation.
    a_fp4 = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device="cuda")
    w_fp4 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scale_a = torch.randint(
        124, 129, (m, k // BLOCK_SIZE), dtype=torch.uint8, device="cuda"
    )
    scale_w = torch.randint(
        124, 129, (n, k // BLOCK_SIZE), dtype=torch.uint8, device="cuda"
    )

    calls: dict[str, Callable[[], torch.Tensor]] = {
        "plain": lambda: _gemm_mxfp4_aiter(a_fp4, w_fp4, scale_a, scale_w),
        "shuffled": lambda: _gemm_mxfp4_aiter_preshuffle(
            a_fp4, w_fp4, scale_a, scale_w
        ),
    }
    asm_config = _mxfp4_asm_config(m, n, k)
    if asm_config is not None:
        calls["asm"] = lambda: _gemm_mxfp4_aiter_asm(
            a_fp4, w_fp4, scale_a, scale_w, asm_config=asm_config
        )

    outputs = {backend: fn() for backend, fn in calls.items()}
    torch.cuda.synchronize()
    plain = outputs["plain"]
    backend_parity = {
        backend: _error(plain, output)
        for backend, output in outputs.items()
        if backend != "plain"
    }
    reference = _chunked_reference_error(
        a_fp4=a_fp4,
        w_fp4=w_fp4,
        scale_a=scale_a,
        scale_w=scale_w,
        outputs=outputs,
        chunk_rows=chunk_rows,
    )
    timings = {
        backend: _time(fn, f"{name}_{backend}", warmup, iters)
        for backend, fn in calls.items()
    }
    return {
        "shape_mnk": list(shape),
        "asm_config": list(asm_config) if asm_config is not None else None,
        "backend_parity_vs_plain": backend_parity,
        "dequant_fp32_reference": reference,
        "timings": timings,
    }


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is required")
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("this PyTorch build has no packed FP4 dtype")
    if args.reference_chunk_rows <= 0:
        raise ValueError("--reference-chunk-rows must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse {output_dir}")
    output_dir.mkdir(parents=True)
    source_snapshot = output_dir / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), source_snapshot)

    torch.cuda.set_device(args.device)
    config_paths = [
        Path(value).resolve()
        for value in os.environ.get("AITER_CONFIG_GEMM_A4W4", "").split(":")
        if value
    ]
    results = {
        name: _run_shape(
            name,
            shape,
            args.seed + index,
            args.warmup,
            args.iters,
            args.reference_chunk_rows,
        )
        for index, (name, shape) in enumerate(SHAPES.items())
    }
    payload = {
        "metadata": {
            "fresh_run": True,
            "hostname": platform.node(),
            "device": torch.cuda.get_device_name(args.device),
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "seed": args.seed,
            "lumen_head": _git(["git", "rev-parse", "HEAD"], Path(__file__).resolve().parents[1]),
            "lumen_status": _git(["git", "status", "--short"], Path(__file__).resolve().parents[1]).splitlines(),
            "source_snapshot": str(source_snapshot),
            "source_sha256": _sha256(source_snapshot),
            "aiter_config_paths": [str(path) for path in config_paths],
            "aiter_config_sha256": {
                str(path): _sha256(path) for path in config_paths if path.is_file()
            },
        },
        "arguments": vars(args) | {"output_dir": str(output_dir)},
        "results": results,
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"RESULT_JSON path={result_path}")


if __name__ == "__main__":
    main()
