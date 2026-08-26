# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Speed and unbiasedness of the MXFP4 quantizer against Philox round count.

Report §5.19 attributed 22-26% of the gradient-path dual-layout quantizer to
stochastic rounding, and the AMDGCN says where it goes: on ``grad gate_up``,
386 ``v_xor_b32`` + 210 ``v_mul_lo_u32`` + 200 ``v_mul_hi_u32`` = 796 of 2415
instructions, a third of the whole kernel, all of it Philox. Each round is two
mulhi, two mullo and a pair of xors, so the count is close to linear in
``SR_PHILOX_ROUNDS``.

``probe_prng_b32.py`` ruled out replacing Philox with the hardware PRNG: that
instruction is a linear bijection, so it stirs entropy but never creates it, and
one chain cannot dither a quantization block independently. Which leaves buying
the same entropy for fewer rounds. Lumen already runs 7 against the Random123
default of 10; the paper's own analysis puts Philox-4x32 through BigCrush at 7
and reports no failures from 6, and Monte Carlo practice commonly uses 4.

SR's requirement is weaker than a general PRNG's. It needs the dither uniform
enough that the rounding is unbiased and the errors across a quantization block
are not correlated -- there is no adversary and no long-range structure to worry
about. So this measures both halves of the trade: kernel time, and the actual
statistical property the reduction could break.

The bias figure is the one that decides it. For a block-scaled FP4 grid, SR is
unbiased when E[dequant(quant(x))] == x, so the mean residual over many draws
should fall as 1/sqrt(draws) and nothing else. A round count that biases the
rounding shows up as a residual mean that stops falling.

Run:
    python benchmarks/bench_sr_philox_rounds.py [--rounds 7 4 3 2]
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.bench_utils import cuda_timer, require_cuda  # noqa: E402
from benchmarks.bench_dual_layout_tiles import (  # noqa: E402
    BLOCK,
    G,
    SHAPES,
    TOKENS,
    _alloc,
    _default_blocks,
    _launcher,
)

# Only the gradient shapes pay for SR at all.
GRAD_SHAPES = [(label, N) for label, N, recipe in SHAPES if recipe == "grad"]


def set_rounds(n: int):
    """Rebuild the kernel module with a different Philox round count.

    ``SR_PHILOX_ROUNDS_C`` is a ``tl.constexpr`` closed over by the jitted
    functions, so it has to be fixed before tracing. Patching the attribute is
    not enough and not even allowed -- Triton 3.7 raises "Global variable
    SR_PHILOX_ROUNDS_C has changed since we compiled this kernel" -- so set the
    environment and reload the module, which builds fresh JITFunctions.
    """
    os.environ["LUMEN_SR_PHILOX_ROUNDS"] = str(n)
    import lumen.kernels.mxfp4 as k

    k = importlib.reload(k)
    assert k.SR_PHILOX_ROUNDS == n, f"reload did not take: {k.SR_PHILOX_ROUNDS} != {n}"
    return k


def philox_instruction_count(M: int, N: int) -> tuple[int, int]:
    """(philox instructions, total instructions) in the compiled kernel."""
    from benchmarks.inspect_dual_layout_asm import classify, compile_one

    compiled, _, _ = compile_one(M, N, "grad")
    asm = compiled.asm.get("amdgcn", "")
    _, total = classify(asm)
    philox = 0
    for raw in asm.splitlines():
        line = raw.strip()
        if line.split(" ")[0] in ("v_xor_b32_e32", "v_mul_lo_u32", "v_mul_hi_u32"):
            philox += 1
    return philox, total


def _dequant(q: torch.Tensor, s: torch.Tensor, block: int) -> torch.Tensor:
    """Packed MXFP4 + E8M0 scales -> fp32, matching the kernel's convention.

    The kernel builds the multiplier as ``(scale_byte << 23)`` bitcast to fp32,
    i.e. the byte lands in the exponent field, so the value is 2^(byte-127).
    """
    from aiter.utility.fp4_utils import mxfp4_to_f32

    codes = mxfp4_to_f32(q)
    mult = (s.to(torch.int32) << 23).view(torch.float32)
    return codes * mult.repeat_interleave(block, dim=-1)


def residual_stats(n_draws: int = 512) -> tuple[float, float]:
    """Mean and std of dequant(quant(x)) - x over independent SR draws.

    Row-major layout only: the transposed one rotates before quantizing, so its
    residual is not comparable elementwise.
    """
    from lumen.ops.quantize.ops import convert_to_mxfp4

    torch.manual_seed(0)
    x = torch.randn((256, BLOCK * 8), dtype=torch.bfloat16, device="cuda")
    acc = torch.zeros((256, BLOCK * 8), dtype=torch.float32, device="cuda")
    for i in range(n_draws):
        q, s = convert_to_mxfp4(x, BLOCK, axis=-1, use_sr=True,
                                philox_seed=1234 + i, philox_offset=0)
        acc += _dequant(q, s, BLOCK)
    resid = acc / n_draws - x.to(torch.float32)
    # Normalise by mean|x| so the figure is comparable across round counts.
    denom = x.to(torch.float32).abs().mean().clamp_min(1e-6)
    return (resid.mean() / denom).item(), (resid.std() / denom).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, nargs="+", default=[7, 6, 5, 4, 3, 2])
    ap.add_argument("--draws", type=int, default=512,
                    help="SR draws for the unbiasedness check")
    args = ap.parse_args()
    require_cuda()

    baseline = {}
    print(f"{'rounds':>6} {'shape':<22} {'ms':>8} {'vs 7 rounds':>12} "
          f"{'philox insts':>13} {'of total':>9}")
    print("-" * 78)

    for rounds in args.rounds:
        set_rounds(rounds)
        torch.manual_seed(0)
        sign = torch.randint(0, 2, (G,), device="cuda", dtype=torch.float32) * 2 - 1
        for label, N in GRAD_SHAPES:
            M = TOKENS
            from lumen.ops.quantize.ops import mxfp4_scale_swizzle_supported

            swizzle = (mxfp4_scale_swizzle_supported(M, N // BLOCK)
                       and mxfp4_scale_swizzle_supported(N, M // BLOCK))
            bm, bn = _default_blocks(M, N)
            x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
            outs = _alloc(M, N, swizzle)
            run = _launcher(x, outs, sign, bm, bn, 4, use_sr=True,
                            swizzle=swizzle, shuffle=False)
            run()
            torch.cuda.synchronize()
            ms = cuda_timer(run, warmup=10, iters=40, trim_pct=10.0, label=label).median_ms
            ph, tot = philox_instruction_count(M, N)
            baseline.setdefault(label, ms)
            print(f"{rounds:>6} {label:<22} {ms:>8.3f} "
                  f"{baseline[label] / ms:>11.2f}x {ph:>13} {ph / tot * 100:>8.1f}%")
            del x, outs
            torch.cuda.empty_cache()
        print()

    print("=" * 78)
    print("unbiasedness: mean and std of dequant(quant(x)) - x over "
          f"{args.draws} SR draws,\nnormalised by mean|x|. SR is unbiased when the "
          "mean is pure sampling noise,\nso it should sit near 1/sqrt(draws) = "
          f"{1 / args.draws ** 0.5:.4f} and not grow as rounds fall.")
    print()
    print(f"{'rounds':>6} {'resid mean':>12} {'resid std':>12}")
    print("-" * 32)
    for rounds in args.rounds:
        set_rounds(rounds)
        try:
            mean, std = residual_stats(args.draws)
            print(f"{rounds:>6} {mean:>12.5f} {std:>12.5f}")
        except Exception as e:
            print(f"{rounds:>6} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
