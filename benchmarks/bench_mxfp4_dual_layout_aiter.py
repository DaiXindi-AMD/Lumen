# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare Lumen and AITER dual-layout MXFP4 on the Qwen3 gradient shape.

The benchmark calls both production-facing APIs with identical tensors and
Philox streams.  It first requires bitwise output parity, then uses interleaved
CUDA-event samples for the quantizer alone and for the real split-SwiGLU
backward plus two gradient quantizers.  Results are a kernel-chain screen, not
an end-to-end training-speed claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


M = 16_384
D = 12_288
GATE_SEED = 0x1234
GATE_OFFSET = 0x100000
UP_SEED = 0x5678
UP_OFFSET = 0x200000
SWIZZLE_SCALE = True


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20_260_929)
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--pairs", type=_positive_int, default=32)
    return parser


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def _git_root(path: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip()).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_exact(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]):
    if len(actual) != len(expected):
        raise AssertionError(f"output count differs: {len(actual)} != {len(expected)}")
    for index, (got, want) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(
            got,
            want,
            atol=0,
            rtol=0,
            msg=lambda message: f"output {index} mismatch: {message}",
        )


def _event_ms(function: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _summary(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)

    def quantile(fraction: float) -> float:
        index = round(fraction * (len(ordered) - 1))
        return ordered[index]

    return {
        "count": len(samples),
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p20_ms": quantile(0.2),
        "p80_ms": quantile(0.8),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _interleaved(
    lumen_function: Callable[[], object],
    aiter_function: Callable[[], object],
    *,
    warmup: int,
    pairs: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        lumen_function()
        aiter_function()
    torch.cuda.synchronize()

    samples = {"lumen": [], "aiter": []}
    for pair in range(pairs):
        order = (
            (("lumen", lumen_function), ("aiter", aiter_function))
            if pair % 2 == 0
            else (("aiter", aiter_function), ("lumen", lumen_function))
        )
        sequence = (*order, *reversed(order))
        for name, function in sequence:
            samples[name].append(_event_ms(function))

    lumen_summary = _summary(samples["lumen"])
    aiter_summary = _summary(samples["aiter"])
    return {
        "lumen": lumen_summary,
        "aiter": aiter_summary,
        "speedup_mean": lumen_summary["mean_ms"] / aiter_summary["mean_ms"],
        "speedup_median": (lumen_summary["median_ms"] / aiter_summary["median_ms"]),
        "raw_samples_ms": samples,
    }


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device required")
    torch.cuda.set_device(args.device)

    from aiter.ops.triton.activation import swiglu_bwd_split
    from aiter.ops.triton.quant import dual_layout_quant_mxfp4 as aiter_dual
    from lumen.ops.quantize.linear import _get_mxfp4_rht_sign
    from lumen.ops.quantize.ops import dual_layout_quant_mxfp4 as lumen_dual

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    x = torch.randn((M, D), dtype=torch.bfloat16, device="cuda")
    grad = torch.randn_like(x)
    gate = torch.randn_like(x)
    up = torch.randn_like(x)
    sign = _get_mxfp4_rht_sign(x.device)

    def lumen_quant(
        value: torch.Tensor, seed: int, offset: int
    ) -> tuple[torch.Tensor, ...]:
        return lumen_dual(
            value,
            sign,
            use_sr_row=True,
            use_sr_transposed=True,
            philox_seed=seed,
            philox_offset=offset,
            swizzle_scale=SWIZZLE_SCALE,
        )

    def aiter_quant(
        value: torch.Tensor, seed: int, offset: int
    ) -> tuple[torch.Tensor, ...]:
        return aiter_dual(
            value,
            use_sr=True,
            philox_seed=seed,
            philox_offset=offset,
            swizzle_scale=SWIZZLE_SCALE,
        )

    lumen_x = lumen_quant(x, GATE_SEED, GATE_OFFSET)
    aiter_x = aiter_quant(x, GATE_SEED, GATE_OFFSET)
    torch.cuda.synchronize()
    _assert_exact(aiter_x, lumen_x)
    del lumen_x, aiter_x

    def lumen_chain() -> tuple[torch.Tensor, ...]:
        dgate, dup = swiglu_bwd_split(grad, gate, up)
        return (
            dgate,
            dup,
            *lumen_quant(dgate, GATE_SEED, GATE_OFFSET),
            *lumen_quant(dup, UP_SEED, UP_OFFSET),
        )

    def aiter_chain() -> tuple[torch.Tensor, ...]:
        dgate, dup = swiglu_bwd_split(grad, gate, up)
        return (
            dgate,
            dup,
            *aiter_quant(dgate, GATE_SEED, GATE_OFFSET),
            *aiter_quant(dup, UP_SEED, UP_OFFSET),
        )

    lumen_outputs = lumen_chain()
    aiter_outputs = aiter_chain()
    torch.cuda.synchronize()
    _assert_exact(aiter_outputs, lumen_outputs)
    del lumen_outputs, aiter_outputs

    results = {
        "dual_layout": _interleaved(
            lambda: lumen_quant(x, GATE_SEED, GATE_OFFSET),
            lambda: aiter_quant(x, GATE_SEED, GATE_OFFSET),
            warmup=args.warmup,
            pairs=args.pairs,
        ),
        "swiglu_bwd_plus_two_dual_layout": _interleaved(
            lumen_chain,
            aiter_chain,
            warmup=args.warmup,
            pairs=args.pairs,
        ),
    }

    source = Path(__file__).resolve()
    lumen_dual_source = Path(lumen_dual.__code__.co_filename).resolve()
    aiter_dual_source = Path(aiter_dual.__code__.co_filename).resolve()
    swiglu_source = Path(swiglu_bwd_split.__code__.co_filename).resolve()
    lumen_root = _git_root(lumen_dual_source)
    aiter_root = _git_root(aiter_dual_source)
    payload = {
        "metadata": {
            "fresh_output": True,
            "hostname": platform.node(),
            "device": torch.cuda.get_device_name(args.device),
            "torch": torch.__version__,
            "rocm": torch.version.hip,
            "lumen_head": _git(["rev-parse", "HEAD"], lumen_root),
            "aiter_head": _git(["rev-parse", "HEAD"], aiter_root),
            "lumen_status": _git(["status", "--short"], lumen_root).splitlines(),
            "aiter_status": _git(["status", "--short"], aiter_root).splitlines(),
            "source": str(source),
            "source_sha256": _sha256(source),
            "implementation_sources": {
                "lumen_dual_layout": {
                    "path": str(lumen_dual_source),
                    "sha256": _sha256(lumen_dual_source),
                },
                "aiter_dual_layout": {
                    "path": str(aiter_dual_source),
                    "sha256": _sha256(aiter_dual_source),
                },
                "aiter_swiglu": {
                    "path": str(swiglu_source),
                    "sha256": _sha256(swiglu_source),
                },
            },
        },
        "arguments": {
            "output_dir": str(output_dir),
            "device": args.device,
            "seed": args.seed,
            "warmup": args.warmup,
            "pairs": args.pairs,
            "shape": [M, D],
            "swizzle_scale": SWIZZLE_SCALE,
        },
        "correctness": "bitwise_equal",
        "results": results,
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"RESULT_JSON={result_path}")


if __name__ == "__main__":
    main()
