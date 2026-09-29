# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 GEMM backend selection benchmark.

Exercises ``gemm_mxfp4_dispatch`` at Llama 3.1 8B linear shapes across Lumen's
explicit ASM, FlyDSL and Triton implementations, to show what each buys and why
the choice has to be made per shape rather than globally.

The benchmark reports every legal FlyDSL configuration beside the exact ASM
entry selected for the shape. The dispatch policy retains a working ASM result
unless the fastest validated FlyDSL configuration is at least five percent
faster; Triton remains the fallback rather than an eligible ASM replacement.

Run:
    LUMEN_MXFP4_FLYDSL=1 python -m pytest benchmarks/bench_mxfp4_gemm.py -v -s
"""

import os

import pytest
import torch

from benchmarks.bench_utils import cuda_timer, print_report_with_table
from benchmarks.conftest import AITER
from lumen.ops.quantize import flydsl_mxfp4, mxfp4_autotune
from lumen.ops.quantize.linear import (
    _gemm_mxfp4_aiter,
    _gemm_mxfp4_aiter_asm,
    _gemm_mxfp4_aiter_preshuffle,
    _gemm_mxfp4_flydsl,
    _mxfp4_asm_config,
    _mxfp4_asm_supported,
    _mxfp4_flydsl_backend_names,
    _mxfp4_preshuffle_eligible,
    gemm_mxfp4_dispatch,
)
from lumen.ops.quantize.ops import convert_to_mxfp4, convert_to_mxfp4_2d

HIDDEN = 4096
FFN_HIDDEN = 14336
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

MXFP4_BLOCK = 32
TOKENS = 8192

LAYERS = [
    ("qkv_proj", NUM_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM, HIDDEN),
    ("o_proj", HIDDEN, HIDDEN),
    ("gate_up_proj", 2 * FFN_HIDDEN, HIDDEN),
    ("down_proj", HIDDEN, FFN_HIDDEN),
]


def _quantized_operands(M, N, K):
    a_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_hp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    a_fp4, a_s = convert_to_mxfp4(a_hp, block_size=MXFP4_BLOCK, axis=-1, use_sr=False)
    w_fp4, w_s = convert_to_mxfp4_2d(w_hp, block_size=MXFP4_BLOCK, use_sr=False)
    del a_hp, w_hp
    return a_fp4, w_fp4, a_s, w_s


@AITER
@pytest.mark.parametrize("layer,N,K", LAYERS, ids=[layer for layer, _, _ in LAYERS])
def test_mxfp4_gemm_backend_choice(layer, N, K):
    """Compare explicit MXFP4 backends and verify the cached dispatch choice."""
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("torch.float4_e2m1fn_x2 unavailable in this PyTorch build")

    M = TOKENS
    a_fp4, w_fp4, a_s, w_s = _quantized_operands(M, N, K)
    reference = _gemm_mxfp4_aiter(a_fp4, w_fp4, a_s, w_s)

    # Settle the autotune decision up front. Its measurement allocates output
    # buffers for every candidate, and letting that happen inside the timed run
    # charges the dispatcher for one-time work it does not do per call.
    gemm_mxfp4_dispatch(a_fp4, w_fp4, a_s, w_s)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    results = [
        cuda_timer(
            lambda a=a_bf16, w=w_bf16: torch.mm(a, w.t()),
            label=f"{layer} bf16",
        ),
    ]
    del a_bf16, w_bf16
    torch.cuda.empty_cache()

    results.append(
        cuda_timer(
            lambda a=a_fp4, w=w_fp4, sa=a_s, sw=w_s: _gemm_mxfp4_aiter(
                a, w, sa, sw
            ),
            label=f"{layer} mxfp4 plain",
        )
    )
    try:
        results.append(
            cuda_timer(
                lambda a=a_fp4, w=w_fp4, sa=a_s, sw=w_s: (
                    _gemm_mxfp4_aiter_preshuffle(a, w, sa, sw)
                ),
                label=f"{layer} mxfp4 shuffled",
            )
        )
    except (RuntimeError, NotImplementedError, AssertionError) as e:
        pytest.skip(f"shuffled MXFP4 GEMM unavailable: {e}")

    # Benchmark every explicitly tuned ASM shape. The static profitability
    # threshold is a dispatch fallback, not a reason to omit an incumbent from
    # the per-shape ASM-versus-FlyDSL comparison.
    asm_ok = _mxfp4_asm_supported(a_fp4, w_fp4)
    asm_config = _mxfp4_asm_config(M, N, K) if asm_ok else None
    asm = None
    if asm_ok:
        torch.testing.assert_close(
            _gemm_mxfp4_aiter_asm(a_fp4, w_fp4, a_s, w_s),
            reference,
            atol=0,
            rtol=0,
        )
        results.append(
            cuda_timer(
                lambda a=a_fp4, w=w_fp4, sa=a_s, sw=w_s: (
                    _gemm_mxfp4_aiter_asm(a, w, sa, sw)
                ),
                label=f"{layer} mxfp4 asm",
            )
        )
        asm = results[-1].avg_ms

    flydsl_times = {}
    if flydsl_mxfp4.available():
        for name in _mxfp4_flydsl_backend_names(a_fp4, w_fp4):
            torch.testing.assert_close(
                _gemm_mxfp4_flydsl(name, a_fp4, w_fp4, a_s, w_s),
                reference,
                atol=0.1,
                rtol=0.1,
            )
            results.append(
                cuda_timer(
                    lambda name=name, a=a_fp4, w=w_fp4, sa=a_s, sw=w_s: (
                        _gemm_mxfp4_flydsl(name, a, w, sa, sw)
                    ),
                    label=f"{layer} mxfp4 {name}",
                )
            )
            flydsl_times[name] = results[-1].avg_ms

    del reference
    torch.cuda.empty_cache()

    results.append(
        cuda_timer(
            lambda a=a_fp4, w=w_fp4, sa=a_s, sw=w_s: gemm_mxfp4_dispatch(
                a, w, sa, sw
            ),
            label=f"{layer} mxfp4 dispatch",
        )
    )
    print_report_with_table(f"MXFP4 GEMM  {layer}  M={M} N={N} K={K}", results)

    plain, shuffled, dispatch = results[1].avg_ms, results[2].avg_ms, results[-1].avg_ms
    times = {"plain": plain, "shuffled": shuffled}
    if asm is not None:
        times["asm"] = asm
    times.update(flydsl_times)

    # Which backend the dispatcher settled on is a property of the autotuner, not
    # of the static thresholds, so ask it rather than re-deriving it.
    chosen = mxfp4_autotune.cached((M, N, K)) or "plain"
    asm_identity = (
        f"{asm_config[0]} splitK={asm_config[1]}" if asm_config else "n/a"
    )
    print(
        f"  asm_supported={asm_ok}  asm_identity={asm_identity}  "
        f"shuffle_eligible={_mxfp4_preshuffle_eligible(a_fp4, w_fp4)}  "
        f"plain={plain:.3f}ms  shuffled={shuffled:.3f}ms  "
        f"asm={'n/a' if asm is None else f'{asm:.3f}ms'}  "
        f"flydsl={flydsl_times or 'n/a'}  "
        f"dispatch={dispatch:.3f}ms  chose={chosen}"
    )

    expected = times.get(chosen, plain)
    assert dispatch <= expected * 1.20, (
        f"dispatch {dispatch:.3f}ms far above its chosen backend {chosen} {expected:.3f}ms"
    )
    if asm_ok and flydsl_mxfp4.is_backend_name(chosen):
        profile = mxfp4_autotune.cached_profile((M, N, K))
        assert profile is not None
        profile_times = profile["timings_ms"]
        assert (
            profile_times[chosen] * mxfp4_autotune._SWITCH_MARGIN
            <= profile_times["asm"]
        ), "FlyDSL replaced ASM without clearing the protected margin"


@AITER
def test_mxfp4_gemm_layer_total():
    """Sum the four projections to show the per-layer forward GEMM effect."""
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("torch.float4_e2m1fn_x2 unavailable in this PyTorch build")

    M = TOKENS
    totals = {"bf16": 0.0, "plain": 0.0, "shuffled": 0.0, "dispatch": 0.0}

    for layer, N, K in LAYERS:
        a_fp4, w_fp4, a_s, w_s = _quantized_operands(M, N, K)
        a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

        totals["bf16"] += cuda_timer(
            lambda a=a_bf16, w=w_bf16: torch.mm(a, w.t())
        ).avg_ms
        del a_bf16, w_bf16
        torch.cuda.empty_cache()

        totals["plain"] += cuda_timer(
            lambda a=a_fp4, w=w_fp4, sa=a_s, sw=w_s: _gemm_mxfp4_aiter(
                a, w, sa, sw
            )
        ).avg_ms
        # What step 1 shipped: shuffled Triton where it wins, plain otherwise.
        step1 = (
            _gemm_mxfp4_aiter_preshuffle
            if _mxfp4_preshuffle_eligible(a_fp4, w_fp4)
            else _gemm_mxfp4_aiter
        )
        totals["shuffled"] += cuda_timer(
            lambda fn=step1, a=a_fp4, w=w_fp4, sa=a_s, sw=w_s: fn(
                a, w, sa, sw
            )
        ).avg_ms
        totals["dispatch"] += cuda_timer(
            lambda a=a_fp4, w=w_fp4, sa=a_s, sw=w_s: gemm_mxfp4_dispatch(
                a, w, sa, sw
            )
        ).avg_ms

        del a_fp4, w_fp4, a_s, w_s
        torch.cuda.empty_cache()

    print(f"\n  per-layer forward GEMM total, M={M}")
    for k, v in totals.items():
        print(f"    {k:<10} {v:7.3f} ms   ({totals['bf16'] / v:.2f}x vs bf16)")
    print(
        f"    dispatch is {totals['plain'] / totals['dispatch']:.2f}x the plain path, "
        f"{totals['shuffled'] / totals['dispatch']:.2f}x the shuffled-Triton path"
    )


if __name__ == "__main__":
    os.environ.setdefault("LUMEN_BENCH_ITERS", "30")
    pytest.main([__file__, "-v", "-s"])
