# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tile and warp-count sweep for the MXFP4 dual-layout quantizer.

``_dual_layout_quant_mxfp4_kernel`` is the largest non-GEMM kernel left in a
Qwen3-8B MXFP4 step (412.5 ms of 6069, report §5.15). Its launcher picks
BLOCK_M/BLOCK_N from a rule written for hidden=4096 and never passes
``num_warps``, so every shape runs Triton's default 4. The tile holds the whole
(BLOCK_M, BLOCK_N) input in registers twice -- once row-major for the A operand,
once transposed for B -- which at the default (256, 64) is 64 fp32 values per
lane before either output is packed, so the launch config is a register-pressure
choice rather than a tiling one and has to be measured per shape.

This sweeps both, on the six shapes a training step actually issues, under each
of the two call recipes (activation: RTN both sides; gradient: SR both sides).

Result on 8xMI350X, 27 configs per shape: the launcher's rule already picks the
fastest of them everywhere, and num_warps 8 and 16 are slower than the default 4
on every shape. The kernel sustains 1.6-2.5 TB/s against ~8 TB/s of HBM, so it is
still register- or compute-bound as report §5.11 said -- but that headroom is not
reachable by relaunching it *tiled* differently.

``--axis pipeline`` sweeps the launch parameters the tile sweep did not touch, at
the launcher's own tile choice. Triton 3.7's HIP backend takes five beyond
num_warps: ``num_stages`` and ``waves_per_eu`` (software pipelining depth and the
occupancy floor the scheduler targets, both of which trade against the register
pressure that makes this kernel slow), and ``matrix_instr_nonkdim``, ``kpack``,
``schedule_hint``, which reach the MFMA the Hadamard rotation runs on. These are
the cheap thing to rule out before rewriting the kernel in HIP.

Run:
    python benchmarks/bench_dual_layout_tiles.py [--axis tiles|pipeline]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.bench_utils import cuda_timer, require_cuda  # noqa: E402
from lumen.ops.quantize.ops import (  # noqa: E402
    _dividing_block,
    _rht_matrix_bf16,
    mxfp4_data_shuffle_supported,
    mxfp4_scale_swizzle_supported,
)

BLOCK = 32
G = 16
TOKENS = 16384  # mbs 2 x seq 8192, the token count every Qwen3-8B GEMM sees

# (label, N, recipe). Recipe "act" is the forward activation saved for WGrad
# (RTN both layouts, scales swizzled); "grad" is the backward gradient
# (SR both layouts). Ns are Qwen3-8B's: hidden 4096, fused qkv 6144,
# ffn 12288, fused gate_up 24576.
SHAPES = [
    ("grad gate_up", 24576, "grad"),
    ("act  down", 12288, "act"),
    ("grad qkv", 6144, "grad"),
    ("grad o/down", 4096, "grad"),
    ("act  qkv/o/gate_up", 4096, "act"),
]


def _default_blocks(M: int, N: int) -> tuple[int, int]:
    """What lumen.ops.quantize.ops.dual_layout_quant_mxfp4 picks today."""
    block_m = _dividing_block(M, 256, floor=max(BLOCK, G))
    block_n = _dividing_block(N, 64 if N >= 8192 else 32, floor=BLOCK)
    return block_m, block_n


def _alloc(M: int, N: int, swizzle: bool):
    from lumen.kernels.mxfp4 import MXFP4_SCALE_STRIPE

    n_scale_a, n_scale_b = N // BLOCK, M // BLOCK
    shape_a = (M // MXFP4_SCALE_STRIPE, n_scale_a * MXFP4_SCALE_STRIPE) if swizzle else (M, n_scale_a)
    shape_b = (N // MXFP4_SCALE_STRIPE, n_scale_b * MXFP4_SCALE_STRIPE) if swizzle else (N, n_scale_b)
    dev = "cuda"
    return (
        torch.empty((M, N // 2), dtype=torch.uint8, device=dev),
        torch.empty(shape_a, dtype=torch.uint8, device=dev),
        torch.empty((N, M // 2), dtype=torch.uint8, device=dev),
        torch.empty(shape_b, dtype=torch.uint8, device=dev),
    )


def _launcher(x, outs, sign, block_m, block_n, num_warps, *, use_sr, swizzle, shuffle,
              **launch_opts):
    """One pre-bound launch of the kernel, so timing excludes argument setup."""
    from lumen.kernels.mxfp4 import _dual_layout_quant_mxfp4_kernel

    M, N = x.shape
    a, a_s, b, b_s = outs
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    hmat = _rht_matrix_bf16(sign, G)
    from lumen.ops.quantize.ops import is_cdna4

    kwargs = dict(
        BLOCK_M=block_m, BLOCK_N=block_n,
        QUANT_BLOCK_SIZE=BLOCK,
        USE_SR_A=use_sr, USE_SR_B=use_sr,
        USE_ASM=is_cdna4(),
        SWIZZLE_SCALE=swizzle,
        NUM_SCALE_COLS_A=N // BLOCK, NUM_SCALE_COLS_B=M // BLOCK,
        SHUFFLE_B=shuffle, NUM_PACKED_COLS_B=M // 2,
        USE_MFMA=True,
        num_warps=num_warps,
    )
    kwargs.update({k: v for k, v in launch_opts.items() if v is not None})
    args = (
        x, a, a_s, b, b_s, sign, hmat,
        x.stride(0), x.stride(1),
        a.stride(0), a.stride(1), a_s.stride(0), a_s.stride(1),
        b.stride(0), b.stride(1), b_s.stride(0), b_s.stride(1),
        1234, 0, 0x9E3779B9,
    )

    def run():
        _dual_layout_quant_mxfp4_kernel[grid](*args, **kwargs)

    return run


def _bytes_moved(M: int, N: int) -> int:
    """Input read once, both FP4 layouts and both E8M0 scale planes written."""
    return M * N * 2 + M * N // 2 + M * N // 32 * 2


def _pipeline_configs():
    """Launch parameters the tile sweep never varied.

    Two stages, so the grid stays affordable. First the two that act on register
    pressure and occupancy directly; then the MFMA-shape knobs, crossed only with
    the survivors, since they only reach the Hadamard rotation.
    """
    coarse = [
        dict(num_stages=s, waves_per_eu=w)
        for s, w in itertools.product((1, 2, 3, 4), (0, 1, 2, 3, 4))
    ]
    refine = [
        dict(matrix_instr_nonkdim=k, kpack=p)
        for k, p in itertools.product((16, 32), (1, 2))
    ]
    return coarse, refine


def _fmt(cfg: dict) -> str:
    short = {"num_stages": "s", "waves_per_eu": "w", "matrix_instr_nonkdim": "nk",
             "kpack": "kp"}
    return ",".join(f"{short.get(k, k)}{v}" for k, v in sorted(cfg.items()))


def sweep_pipeline():
    """Sweep num_stages/waves_per_eu (then MFMA shape) at the launcher's tiles."""
    require_cuda()
    torch.manual_seed(0)
    sign = torch.randint(0, 2, (G,), device="cuda", dtype=torch.float32) * 2 - 1
    coarse, refine = _pipeline_configs()

    print(f"{'shape':<22} {'cfg':<22} {'ms':>8} {'GB/s':>8} {'vs default':>11}")
    print("-" * 76)
    totals = {}

    for label, N, recipe in SHAPES:
        M = TOKENS
        use_sr = recipe == "grad"
        swizzle = mxfp4_scale_swizzle_supported(M, N // BLOCK) and mxfp4_scale_swizzle_supported(
            N, M // BLOCK
        )
        shuffle = recipe == "act" and mxfp4_data_shuffle_supported(N, M // 2)
        x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
        outs = _alloc(M, N, swizzle)
        gb = _bytes_moved(M, N) / 1e9
        bm, bn = _default_blocks(M, N)

        def timed(cfg):
            try:
                run = _launcher(x, outs, sign, bm, bn, 4, use_sr=use_sr,
                                swizzle=swizzle, shuffle=shuffle, **cfg)
                run()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"{label:<22} {_fmt(cfg):<22} skipped: {type(e).__name__}")
                return None
            return cuda_timer(run, warmup=10, iters=40, trim_pct=10.0,
                              label=f"{label} {_fmt(cfg)}").median_ms

        # Triton's own defaults are the baseline: nothing passed at all.
        base = timed({})
        rows = []
        for cfg in coarse:
            ms = timed(cfg)
            if ms is not None:
                rows.append((ms, cfg))
        rows.sort(key=lambda r: r[0])
        for cfg in refine:
            for _, top in rows[:2]:
                ms = timed({**top, **cfg})
                if ms is not None:
                    rows.append((ms, {**top, **cfg}))
        rows.sort(key=lambda r: r[0])

        print(f"{label:<22} {'(triton defaults)':<22} {base:>8.3f} {gb / base * 1e3:>8.0f} "
              f"{'1.00x':>11}")
        for ms, cfg in rows[:4]:
            print(f"{label:<22} {_fmt(cfg):<22} {ms:>8.3f} {gb / ms * 1e3:>8.0f} "
                  f"{base / ms:>10.2f}x")
        print()
        totals[(label, N)] = (base, rows[0])
        del x, outs
        torch.cuda.empty_cache()

    print("=" * 76)
    print("best launch config per shape, tiles fixed at the launcher's choice")
    now = best = 0.0
    for (label, N), (base, (ms, cfg)) in totals.items():
        now += base
        best += ms
        print(f"  {label:<22} N={N:<6} {base:.3f} ms  ->  {ms:.3f} ms  "
              f"{base / ms:.2f}x  [{_fmt(cfg)}]")
    print(f"\n  sum over one call of each: {now:.3f} -> {best:.3f} ms  ({now / best:.2f}x)")
    print("\n  A speedup here is an upper bound on the step-time win: the quantizer is")
    print("  423.8 ms/iter of a 5591 ms step, so 1.10x on it is ~39 ms, or 0.007x")
    print("  on the ratio. Verify end to end before believing it.")


def sweep_features():
    """Attribute the kernel's time to the work it does, not to how it launches.

    Both launch-parameter sweeps came back negative, so the remaining question is
    which *computation* to attack in a rewrite. Each row turns off one thing and
    holds the rest at what production passes for that shape, so the delta is that
    feature's cost. None of these rows is a legal configuration to ship -- turning
    SR or the Hadamard off changes numerics -- they are cost attributions.
    """
    require_cuda()
    torch.manual_seed(0)
    sign = torch.randint(0, 2, (G,), device="cuda", dtype=torch.float32) * 2 - 1

    print(f"{'shape':<22} {'variant':<20} {'ms':>8} {'GB/s':>8} {'of prod':>9} {'saves':>8}")
    print("-" * 80)

    for label, N, recipe in SHAPES:
        M = TOKENS
        prod_sr = recipe == "grad"
        swizzle_ok = mxfp4_scale_swizzle_supported(M, N // BLOCK) and mxfp4_scale_swizzle_supported(
            N, M // BLOCK
        )
        prod_shuffle = recipe == "act" and mxfp4_data_shuffle_supported(N, M // 2)
        x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
        gb = _bytes_moved(M, N) / 1e9
        bm, bn = _default_blocks(M, N)

        def timed(*, use_sr, swizzle, shuffle, mfma):
            outs = _alloc(M, N, swizzle)
            try:
                run = _launcher(x, outs, sign, bm, bn, 4, use_sr=use_sr,
                                swizzle=swizzle, shuffle=shuffle, USE_MFMA=mfma)
                run()
                torch.cuda.synchronize()
                return cuda_timer(run, warmup=10, iters=40, trim_pct=10.0,
                                  label=label).median_ms
            except Exception as e:
                print(f"{label:<22} skipped: {type(e).__name__}: {e}")
                return None
            finally:
                del outs
                torch.cuda.empty_cache()

        prod = dict(use_sr=prod_sr, swizzle=swizzle_ok, shuffle=prod_shuffle, mfma=True)
        base = timed(**prod)
        variants = [("production", prod)]
        if prod_sr:
            variants.append(("no SR (RTN)", {**prod, "use_sr": False}))
        variants.append(("butterfly not MFMA", {**prod, "mfma": False}))
        if swizzle_ok:
            variants.append(("no scale swizzle", {**prod, "swizzle": False}))
        if prod_shuffle:
            variants.append(("no B shuffle", {**prod, "shuffle": False}))

        for name, cfg in variants:
            ms = timed(**cfg)
            if ms is None:
                continue
            saves = "-" if name == "production" else f"{(base - ms) / base * 100:+.0f}%"
            print(f"{label:<22} {name:<20} {ms:>8.3f} {gb / ms * 1e3:>8.0f} "
                  f"{ms / base:>8.2f}x {saves:>8}")
        print()
        del x
        torch.cuda.empty_cache()

    print("=" * 80)
    print("A feature's cost is what a rewrite could win by making it free. The")
    print("quantizer is 423.8 ms of a 5591 ms step and 1.6x needs 247 ms off the")
    print("step, so only a feature worth >58% of the kernel is on the path to 1.6x.")


def sweep_tiles():
    require_cuda()
    torch.manual_seed(0)
    sign = torch.randint(0, 2, (G,), device="cuda", dtype=torch.float32) * 2 - 1

    # BLOCK_M must be a whole number of quant blocks and Hadamard groups;
    # BLOCK_N a whole number of quant blocks. Both bounded above by what the
    # register file can hold twice over.
    candidates = [
        (bm, bn, w)
        for bm, bn, w in itertools.product((64, 128, 256), (32, 64, 128), (4, 8, 16))
        if bm * bn <= 256 * 128
    ]

    print(f"{'shape':<22} {'cfg':<18} {'ms':>8} {'GB/s':>8} {'vs default':>11}")
    print("-" * 72)
    best_overall = {}

    for label, N, recipe in SHAPES:
        M = TOKENS
        use_sr = recipe == "grad"
        swizzle = mxfp4_scale_swizzle_supported(M, N // BLOCK) and mxfp4_scale_swizzle_supported(
            N, M // BLOCK
        )
        # Only the activation recipe ever asks for the B-operand shuffle, and
        # only when the WGrad backend for the shape reads that order.
        shuffle = recipe == "act" and mxfp4_data_shuffle_supported(N, M // 2)
        x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
        outs = _alloc(M, N, swizzle)
        gb = _bytes_moved(M, N) / 1e9

        dm, dn = _default_blocks(M, N)
        rows = []
        for bm, bn, w in candidates:
            if M % bm or N % bn:
                continue
            try:
                run = _launcher(x, outs, sign, bm, bn, w, use_sr=use_sr,
                                swizzle=swizzle, shuffle=shuffle)
                run()
                torch.cuda.synchronize()
            except Exception as e:  # a tile the compiler cannot fit
                print(f"{label:<22} ({bm},{bn},w{w}) skipped: {type(e).__name__}")
                continue
            ms = cuda_timer(run, warmup=10, iters=40, trim_pct=10.0,
                            label=f"{label} {bm}x{bn}w{w}").median_ms
            rows.append((ms, bm, bn, w))

        base = next((r for r in rows if r[1] == dm and r[2] == dn and r[3] == 4), None)
        rows.sort()
        for ms, bm, bn, w in rows[:4] + ([base] if base and base != rows[0] else []):
            tag = "  <- default" if (bm, bn, w) == (dm, dn, 4) else ""
            rel = f"{base[0] / ms:.2f}x" if base else "-"
            print(f"{label:<22} ({bm:>3},{bn:>3},w{w}){'':<5} {ms:>8.3f} {gb / ms * 1e3:>8.0f} "
                  f"{rel:>11}{tag}")
        print()
        best_overall[(label, N, recipe)] = (rows[0], base)
        del x, outs
        torch.cuda.empty_cache()

    print("=" * 72)
    print("best config per shape")
    total_now = total_best = 0.0
    for (label, N, recipe), (best, base) in best_overall.items():
        if base is None:
            continue
        total_now += base[0]
        total_best += best[0]
        print(f"  {label:<22} N={N:<6} default ({base[1]},{base[2]},w4) {base[0]:.3f} ms"
              f"  ->  ({best[1]},{best[2]},w{best[3]}) {best[0]:.3f} ms"
              f"  {base[0] / best[0]:.2f}x")
    print(f"\n  sum over one call of each: {total_now:.3f} -> {total_best:.3f} ms"
          f"  ({total_now / total_best:.2f}x)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", choices=("tiles", "pipeline", "features"), default="tiles",
                    help="tiles: BLOCK_M/BLOCK_N/num_warps (a settled negative result). "
                         "pipeline: num_stages/waves_per_eu/MFMA shape (also negative). "
                         "features: attribute the kernel's time to the work it does.")
    args = ap.parse_args()
    {"tiles": sweep_tiles, "pipeline": sweep_pipeline, "features": sweep_features}[args.axis]()


if __name__ == "__main__":
    main()
