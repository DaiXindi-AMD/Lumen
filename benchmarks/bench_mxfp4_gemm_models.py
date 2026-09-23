# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-layer MXFP4 training GEMM cost, across models and backend-selection policies.

``bench_mxfp4_gemm.py`` covers the forward GEMMs of one model. This one covers a
whole training layer -- all three GEMMs of every projection -- for several models,
and compares the three ways of choosing a backend:

    plain      always the plain Triton kernel (what Lumen did before any of this)
    static     the hand-measured byte thresholds
    autotune   protected ASM incumbent versus per-shape FlyDSL challengers

The point of the comparison is that the static thresholds were fitted to Llama
3.1 8B and do not transfer: Qwen3-8B's MLP weights are 24 MiB against Llama's
28 MiB, which puts them on the wrong side of a constant chosen for the other
model.

Run:
    LUMEN_MXFP4_FLYDSL=1 python -m pytest benchmarks/bench_mxfp4_gemm_models.py -v -s
"""

import os

import pytest
import torch

from benchmarks.bench_utils import cuda_timer
from benchmarks.conftest import AITER
from lumen.ops.quantize import flydsl_mxfp4, mxfp4_autotune
from lumen.ops.quantize.linear import (
    _gemm_mxfp4_aiter,
    _gemm_mxfp4_aiter_asm,
    _gemm_mxfp4_aiter_preshuffle,
    _gemm_mxfp4_flydsl,
    _mxfp4_asm_config,
    _mxfp4_asm_eligible,
    _mxfp4_asm_supported,
    _mxfp4_flydsl_backend_names,
    _mxfp4_preshuffle_eligible,
    _mxfp4_preshuffle_supported,
)
from lumen.ops.quantize.ops import convert_to_mxfp4

MXFP4_BLOCK = 32
TOKENS = 8192

# name: (hidden, intermediate, q_dim, kv_dim)
MODELS = {
    "Qwen3-0.6B": (1024, 3072, 16 * 128, 8 * 128),
    "Qwen3-8B": (4096, 12288, 32 * 128, 8 * 128),
    "Llama-3.1-8B": (4096, 14336, 32 * 128, 8 * 128),
}


def _projections(hidden, inter, q_dim, kv_dim):
    return [
        ("q_proj", q_dim, hidden),
        ("k_proj", kv_dim, hidden),
        ("v_proj", kv_dim, hidden),
        ("o_proj", hidden, q_dim),
        ("gate_proj", inter, hidden),
        ("up_proj", inter, hidden),
        ("down_proj", hidden, inter),
    ]


def _layer_gemms(hidden, inter, q_dim, kv_dim, tokens):
    """Every (M, N, K) a training step issues for one transformer layer.

    Each linear contributes three: the forward GEMM, and the two backward ones,
    which permute the dims -- dgrad swaps the output and reduction widths, wgrad
    reduces over the token count.
    """
    out = []
    for name, n_out, k_in in _projections(hidden, inter, q_dim, kv_dim):
        out.append((f"{name}.fprop", tokens, n_out, k_in))
        out.append((f"{name}.dgrad", tokens, k_in, n_out))
        out.append((f"{name}.wgrad", n_out, k_in, tokens))
    return out


def _asm_kernel_only(a_fp4, w_fp4, a_s, w_s):
    """The ASM GEMM with the operand layout already built.

    Timing the direct entry point with layout operands already prepared isolates
    kernel time from the shuffle/swizzle prologue included in normal dispatch.
    """
    from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    from lumen.ops.quantize.linear import (
        _MXFP4_SCALE_SHUFFLE_TILING,
        _pad_and_swizzle_mxfp4_scale,
    )

    arch = get_arch()
    tiling = _MXFP4_SCALE_SHUFFLE_TILING[arch]
    M = a_fp4.shape[0]
    N = w_fp4.shape[0]
    K = a_fp4.shape[1] * 2
    asm_config = _mxfp4_asm_config(M, N, K)
    if asm_config is None:
        raise RuntimeError(f"no explicit ASM config for {(M, N, K)}")
    kernel_name, split_k = asm_config
    w_shuf = shuffle_weight(w_fp4, layout=(16, 16))
    sa = _pad_and_swizzle_mxfp4_scale(a_s, arch, tiling)
    sw = _pad_and_swizzle_mxfp4_scale(w_s, arch, tiling)
    out = torch.empty(
        (((M + 31) // 32) * 32, N), dtype=torch.bfloat16, device=a_fp4.device
    )
    return lambda: gemm_a4w4_asm(
        a_fp4,
        w_shuf,
        sa,
        sw,
        out,
        kernelName=kernel_name,
        bpreshuffle=True,
        log2_k_split=split_k,
    )


def _time_backends(M, N, K):
    """ms/iter for each backend that can legally run this shape."""
    a_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_hp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    a_fp4, a_s = convert_to_mxfp4(a_hp, block_size=MXFP4_BLOCK, axis=-1, use_sr=False)
    w_fp4, w_s = convert_to_mxfp4(w_hp, block_size=MXFP4_BLOCK, axis=-1, use_sr=False)
    del a_hp, w_hp
    torch.cuda.empty_cache()

    reference = _gemm_mxfp4_aiter(a_fp4, w_fp4, a_s, w_s)
    times = {
        "plain": cuda_timer(
            lambda a=a_fp4, w=w_fp4, sa=a_s, sw=w_s: _gemm_mxfp4_aiter(
                a, w, sa, sw
            )
        ).avg_ms
    }
    legal_static = "plain"

    if _mxfp4_preshuffle_supported(a_fp4, w_fp4):
        times["shuffled"] = cuda_timer(
            lambda a=a_fp4, w=w_fp4, sa=a_s, sw=w_s: (
                _gemm_mxfp4_aiter_preshuffle(a, w, sa, sw)
            )
        ).avg_ms
        if _mxfp4_preshuffle_eligible(a_fp4, w_fp4):
            legal_static = "shuffled"
    if _mxfp4_asm_supported(a_fp4, w_fp4):
        asm_output = _gemm_mxfp4_aiter_asm(a_fp4, w_fp4, a_s, w_s)
        torch.testing.assert_close(asm_output, reference, atol=0, rtol=0)
        del asm_output
        times["asm"] = cuda_timer(
            lambda a=a_fp4, w=w_fp4, sa=a_s, sw=w_s: _gemm_mxfp4_aiter_asm(
                a, w, sa, sw
            )
        ).avg_ms
        times["asm_nolayout"] = cuda_timer(
            _asm_kernel_only(a_fp4, w_fp4, a_s, w_s)
        ).avg_ms
        if _mxfp4_asm_eligible(a_fp4, w_fp4):
            legal_static = "asm"
    if flydsl_mxfp4.available():
        for name in _mxfp4_flydsl_backend_names(a_fp4, w_fp4):
            flydsl_output = _gemm_mxfp4_flydsl(name, a_fp4, w_fp4, a_s, w_s)
            torch.testing.assert_close(
                flydsl_output, reference, atol=0.1, rtol=0.1
            )
            del flydsl_output
            times[name] = cuda_timer(
                lambda name=name, a=a_fp4, w=w_fp4, sa=a_s, sw=w_s: (
                    _gemm_mxfp4_flydsl(name, a, w, sa, sw)
                )
            ).avg_ms

    del reference, a_fp4, w_fp4, a_s, w_s
    torch.cuda.empty_cache()
    return times, legal_static


@AITER
@pytest.mark.parametrize("model", list(MODELS), ids=list(MODELS))
def test_mxfp4_layer_backend_policies(model):
    """Sum a training layer's GEMMs under each backend-selection policy."""
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("torch.float4_e2m1fn_x2 unavailable in this PyTorch build")

    hidden, inter, q_dim, kv_dim = MODELS[model]
    totals = {"plain": 0.0, "static": 0.0, "autotune": 0.0, "no_prologue": 0.0}
    picks = {"static": {}, "autotune": {}}
    measured = {}

    print(f"\n  {model}  tokens={TOKENS}")
    print(f"  {'gemm':<18} {'M':>6} {'N':>6} {'K':>6} {'wMiB':>6} "
          f"{'plain':>8} {'shuf':>8} {'asm':>8} {'flydsl':>8} "
          f"{'asm-noprol':>11}  {'static':<9} {'auto':<24}")

    for label, M, N, K in _layer_gemms(hidden, inter, q_dim, kv_dim, TOKENS):
        key = (M, N, K)
        if key not in measured:
            times, static_pick = _time_backends(M, N, K)
            profile_times = {
                name: timing
                for name, timing in times.items()
                if name != "asm_nolayout"
            }
            incumbent = "asm" if "asm" in profile_times else static_pick
            auto_pick, _reason = mxfp4_autotune._select_winner(
                profile_times, incumbent=incumbent
            )
            if incumbent == "asm":
                assert auto_pick == "asm" or flydsl_mxfp4.is_backend_name(
                    auto_pick
                )
                if flydsl_mxfp4.is_backend_name(auto_pick):
                    assert (
                        profile_times[auto_pick]
                        * mxfp4_autotune._SWITCH_MARGIN
                        <= profile_times["asm"]
                    )
            measured[key] = (times, static_pick, auto_pick)
        # The dispatcher caches per shape, so repeated shapes reuse the decision
        # rather than re-rolling it against measurement noise.
        times, static_pick, auto_pick = measured[key]

        totals["plain"] += times["plain"]
        totals["static"] += times[static_pick]
        totals["autotune"] += times[auto_pick]
        totals["no_prologue"] += min(
            times[auto_pick], times.get("asm_nolayout", float("inf"))
        )
        for k, v in (("static", static_pick), ("autotune", auto_pick)):
            picks[k][v] = picks[k].get(v, 0) + 1

        def show(name, width=8):
            return f"{times[name]:{width}.3f}" if name in times else f"{'-':>{width}}"

        flydsl_names = [name for name in times if flydsl_mxfp4.is_backend_name(name)]
        flydsl_best = (
            min(flydsl_names, key=lambda name: times[name]) if flydsl_names else None
        )
        flydsl_time = f"{times[flydsl_best]:8.3f}" if flydsl_best else f"{'-':>8}"
        print(f"  {label:<18} {M:>6} {N:>6} {K:>6} {N * K / 2 / 1024 / 1024:>6.1f} "
              f"{show('plain')} {show('shuffled')} {show('asm')} {flydsl_time} "
              f"{show('asm_nolayout', 11)}  {static_pick:<9} {auto_pick:<24}")
        asm_config = _mxfp4_asm_config(M, N, K) if "asm" in times else None
        identities = []
        if asm_config is not None:
            identities.append(f"asm={asm_config[0]} splitK={asm_config[1]}")
        if flydsl_best is not None:
            identities.append(f"flydsl_best={flydsl_best}")
        if identities:
            print(f"    identities: {', '.join(identities)}")

    print(f"\n  per-layer training GEMM total, {model}:")
    for k in ("plain", "static", "autotune"):
        speedup = totals["plain"] / totals[k]
        chosen = "  ".join(f"{n}={c}" for n, c in sorted(picks.get(k, {}).items()))
        print(f"    {k:<11} {totals[k]:7.3f} ms  ({speedup:.2f}x vs plain)   {chosen}")
    print(
        f"    {'no-prologue':<11} {totals['no_prologue']:7.3f} ms  "
        f"({totals['plain'] / totals['no_prologue']:.2f}x vs plain)   "
        f"what a fused cast+shuffle quantize kernel would leave on the table"
    )
    print(
        f"    autotune is {totals['static'] / totals['autotune']:.2f}x "
        f"the static-threshold policy"
    )

if __name__ == "__main__":
    os.environ.setdefault("LUMEN_BENCH_ITERS", "30")
    pytest.main([__file__, "-v", "-s"])
